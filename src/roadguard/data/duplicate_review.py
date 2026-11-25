"""Apply a reviewed exclusion to a verified Unified release image pair."""

import json
from collections import Counter
from pathlib import Path

from roadguard.data.dataset import ImageRecord
from roadguard.data.unified_release import REVISION
from roadguard.data.validation import Issue, ValidationReport, sha256

POLICY = "reviewed_exact_cross_split_pair_v1"
RESOLVABLE = {"duplicate_image", "split_leakage"}


def select_reviewed_pairs(
    report: ValidationReport, review_path: Path,
) -> tuple[list[ImageRecord], set[Issue], dict]:
    """Keep the reviewed canonical pair and account only for its exact findings."""
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if (
        not isinstance(review, dict)
        or review.get("policy") != POLICY
        or review.get("revision") != REVISION
        or not isinstance(review.get("rationale"), str)
        or not review["rationale"].strip()
    ):
        raise ValueError(f"Unrecognized Unified duplicate review: {review_path}")

    indexed = {row.relative_image.as_posix(): row for row in report.records}
    matches = []
    for role in ("canonical", "excluded"):
        entry = review.get(role)
        if not isinstance(entry, dict) or not isinstance(entry.get("image"), str):
            raise ValueError(f"Duplicate review must identify its {role} pair.")
        row = indexed.get(entry["image"])
        if (
            row is None
            or row.image_hash != review.get("image_sha256")
            or row.pixel_hash != review.get("pixel_sha256")
            or row.label.relative_to(report.root).as_posix() != entry.get("label")
            or sha256(row.label) != entry.get("label_sha256")
        ):
            raise ValueError(f"Reviewed {role} image or label failed the pinned identity check.")
        matches.append(row)

    canonical, excluded = matches
    if (
        canonical.split != "train"
        or excluded.split != "val"
        or canonical.provenance != excluded.provenance
        or Counter(box.class_id for box in canonical.boxes)
        != Counter(box.class_id for box in excluded.boxes)
    ):
        raise ValueError("The reviewed copies must match the publisher split and annotation audit.")

    pair = {row.relative_image.as_posix() for row in matches}
    resolved = {
        issue for issue in report.issues
        if issue.level == "error" and issue.code in RESOLVABLE and set(issue.paths) == pair
    }
    if len(resolved) != len(RESOLVABLE) or {issue.code for issue in resolved} != RESOLVABLE:
        raise ValueError("Expected both the exact-duplicate and publisher split-overlap findings.")

    records = [row for row in report.records if row is not excluded]
    audit = {
        **review,
        "resolved_findings": dict(Counter(issue.code for issue in sorted(
            resolved, key=lambda issue: issue.code,
        ))),
        "sample_pairs_excluded": 1,
        "annotations_excluded": len(excluded.boxes),
    }
    return records, resolved, audit
