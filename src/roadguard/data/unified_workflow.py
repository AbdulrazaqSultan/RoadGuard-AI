"""Validate and prepare the Unified release and publish offline EDA artifacts."""

import csv
import io
import json
import logging
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from roadguard.config import Config
from roadguard.data.annotations import read_yolo
from roadguard.data.dataset import CLASS_NAMES, RDD_CLASSES, ImageRecord, Provenance
from roadguard.data.diagnostics import DiagnosticThresholds, flagged_boxes, summarize_records
from roadguard.data.duplicate_review import select_reviewed_pairs
from roadguard.data.prepare import publish_snapshot, read_snapshot, snapshot_summary
from roadguard.data.split import assign_source_splits, read_groups
from roadguard.data.unified_figures import save_unified_figures
from roadguard.data.unified_release import ARCHIVES, REPOSITORY, REVISION
from roadguard.data.validation import ValidationReport, validate_dataset
from roadguard.utils.paths import write_json, write_text

LOGGER = logging.getLogger(__name__)

RISKS = [
    "RDD2022 route and capture sequence IDs are unavailable. Related frames can span partitions.",
    "Filename groups cover recognizable UAV tiles and RoadDamageVision renditions.",
    "Decoded-pixel hashes detect exact copies. Crops and re-encoded near-duplicates need review.",
    "Source and country imbalance can hide domain shift in pooled validation metrics.",
    "Diagnostic box flags identify candidates for review. Keep original images and labels intact.",
]

RECOMMENDATIONS = [
    "After Stage 3 approval, start with pretrained YOLO11n at 640 px, batch 8, CUDA AMP and seed 42.",
    "Track validation AP and recall per class and source before changing augmentation or sampling.",
    "Use the training box-size distribution to decide whether to compare 640 px and 960 px.",
    "Consider source-balanced sampling and minority-class augmentation if validation supports them.",
    "Defer loss weighting and focal-loss experiments to a controlled follow-up.",
]


def _validation_files(
    report: ValidationReport, metrics: Path, *, name: str = "dataset_validation_unified",
) -> dict:
    summary = report.summary()
    write_json(metrics / f"{name}.json", summary)
    lines = [
        "# Unified dataset validation", "",
        f"Status **{'passed' if report.valid else 'blocked'}**", "",
        f"- Images found — {report.images_found:,}",
        f"- Labels found — {report.labels_found:,}",
        f"- Valid pairs — {len(report.records):,}", "",
        "## Findings", "",
    ]
    lines.extend(f"- {code} — {count:,}" for code, count in summary["issue_counts"].items())
    if not report.issues:
        lines.append("No findings.")
    lines.extend(["", f"See `{name}.json` for each affected file and detail."])
    write_text(metrics / f"{name}.md", "\n".join(lines) + "\n")
    return summary


def _quality(
    rows: list[ImageRecord], thresholds: DiagnosticThresholds, output: Path, split: str,
) -> dict:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow([
        "source_image", "split", "source", "annotation", "class_id", "normalized_area",
        "pixel_area", "pixel_width", "pixel_height", "aspect_ratio", "flags",
    ])
    flags: Counter[str] = Counter()
    images = set()
    objects = 0
    for row, index, box, values, labels in flagged_boxes(rows, thresholds):
        writer.writerow([
            row.relative_image.as_posix(), split if split == "train" else row.split,
            row.provenance.source, index, box.class_id, *asdict(values).values(), "|".join(labels),
        ])
        flags.update(labels)
        images.add(row.relative_image)
        objects += 1
    write_text(output, stream.getvalue())
    return {
        "analysis_scope": split,
        "thresholds": asdict(thresholds),
        "flags": dict(flags),
        "flagged_objects": objects,
        "flagged_images": len(images),
    }


def _artifact_paths(root: Path, config: Config) -> tuple[Path, Path]:
    experiment = config["experiment"]
    return root / experiment["metrics_root"], root / experiment["figures_root"] / "dataset" / "unified"


def _path(root: Path, child: Path) -> str:
    return child.relative_to(root).as_posix() if child.is_relative_to(root) else child.as_posix()


def _eda(root: Path, config: Config, rows: list[ImageRecord], fingerprint: str) -> dict:
    metrics, figures = _artifact_paths(root, config)
    try:
        import matplotlib  # noqa: F401
    except ImportError as error:
        raise RuntimeError("Install the RoadGuard eda extra to generate dataset figures.") from error
    thresholds = DiagnosticThresholds(**config["data"]["box_diagnostics"])
    stats = summarize_records(rows)
    review = metrics / "unified" / fingerprint / "box_review.csv"
    quality = _quality(rows, thresholds, review, "train")
    paths = save_unified_figures(
        rows, stats, figures / fingerprint, seed=config["training"]["seed"],
        count=config["experiment"]["sample_images_per_class"], thresholds=thresholds,
    )
    analysis = {
        "split": "train",
        "statistics": stats,
        "box_diagnostics": quality,
        "review_csv": _path(root, review),
        "figures": [_path(root, path) for path in paths],
    }
    write_json(metrics / "unified" / fingerprint / "training_eda.json", analysis)
    return analysis


def _report_base(root: Path, report: ValidationReport, validation: dict) -> dict:
    return {
        "dataset": "Unified Road Defect Dataset",
        "source_repository": REPOSITORY,
        "revision": REVISION,
        "local_root": _path(root, report.root),
        "archive_bytes": sum(
            (report.root / archive).stat().st_size
            for archive in ARCHIVES if (report.root / archive).is_file()
        ),
        "extracted_image_label_bytes": sum(
            file.stat().st_size for file in (report.root / "extracted").rglob("*") if file.is_file()
        ),
        "images_found": report.images_found,
        "labels_found": report.labels_found,
        "validated_pairs": len(report.records),
        "validation_issue_counts": validation["issue_counts"],
        "publisher_class_coverage": validation.get("class_coverage", {}),
        "verified_publisher_class_map": {
            class_id: {"code": code, "name": CLASS_NAMES[class_id]}
            for code, class_id in RDD_CLASSES.items()
        } if report.layout else None,
        "layout": report.layout,
        "fingerprint": None,
        "snapshot": None,
        "splits": None,
        "training_eda": None,
        "key_risks": RISKS.copy(),
        "training_recommendations": RECOMMENDATIONS,
    }


def process_unified(root: Path, config: Config, *, validate_only: bool = False) -> Path | None:
    """Keep preparation behind the complete validation gate."""
    data = config["data"]
    report = validate_dataset(root / data["dataset_root"], "unified")
    metrics, _ = _artifact_paths(root, config)
    validation = _validation_files(report, metrics)
    output = metrics / "dataset_report.json"
    summary = _report_base(root, report, validation)
    summary["input_pair_statistics"] = summarize_records(report.records)
    records = report.records
    accounted = set()
    audit = None
    if data.get("duplicate_review") and report.records:
        try:
            records, accounted, audit = select_reviewed_pairs(report, root / data["duplicate_review"])
        except (OSError, ValueError) as error:
            summary["duplicate_review_error"] = str(error)
    open_errors = [
        issue for issue in report.issues if issue.level == "error" and issue not in accounted
    ]
    summary["unresolved_publisher_findings"] = dict(Counter(issue.code for issue in open_errors))
    if audit:
        audit_path = metrics / "unified_duplicate_resolution.json"
        write_json(audit_path, audit)
        summary["duplicate_resolution"] = _path(root, audit_path)
        LOGGER.info("Applied reviewed duplicate exclusion: %s", audit_path)
    if not records or open_errors or "duplicate_review_error" in summary:
        review = metrics / "unified" / "publisher_box_review.csv"
        summary.update(
            status="validation_failed",
            box_diagnostics=_quality(
                report.records, DiagnosticThresholds(**data["box_diagnostics"]), review, "publisher",
            ),
            review_csv=_path(root, review),
        )
        write_json(output, summary)
        raise ValueError(f"Unified validation blocked preparation. Review {metrics}.")

    try:
        groups = read_groups(root / data["groups_csv"], records) if data["groups_csv"] else None
        splits = assign_source_splits(records, data["split"], config["training"]["seed"], groups)
    except (OSError, ValueError) as error:
        report.add("invalid_split", (report.root,), str(error))
        summary.update(status="split_failed", validation_issue_counts=_validation_files(
            report, metrics,
        )["issue_counts"])
        write_json(output, summary)
        raise
    if validate_only:
        LOGGER.info("Unified preparation gate passed for %s unique images", f"{len(records):,}")
        return None

    source = {
        "repository": REPOSITORY, "revision": REVISION,
        "split_policy": "source_class_balanced_groups_v1",
        "group_source": "csv" if groups is not None else "recognized_filename_parents",
    }
    if audit:
        source["duplicate_review"] = audit
    snapshot = publish_snapshot(
        root / data["processed_root"], splits, version=data["dataset_version"],
        seed=config["training"]["seed"], ratios=data["split"], groups=groups, source_release=source,
    )
    LOGGER.info("Validating the prepared snapshot independently: %s", snapshot)
    prepared = validate_dataset(snapshot, "yolo_split")
    if len(prepared.records) != len(records):
        prepared.add("prepared_count_mismatch", (snapshot,), "Unexpected usable pair count.")
    prepared_name = "dataset_validation_unified_prepared"
    prepared_result = _validation_files(prepared, metrics, name=prepared_name)
    summary["prepared_validation"] = {
        "passed": prepared.valid,
        "fingerprint": snapshot.name,
        "report": _path(root, metrics / f"{prepared_name}.json"),
        "issue_counts": prepared_result["issue_counts"],
    }
    if not prepared.valid:
        summary.update(status="prepared_validation_failed", rejected_snapshot=_path(root, snapshot))
        write_json(output, summary)
        raise ValueError(f"The prepared snapshot failed independent validation. Review {metrics}.")

    split_stats = {name: summarize_records(rows, geometry=False) for name, rows in splits.items()}
    for name, stats in split_stats.items():
        missing = [label for label, count in stats["instances_per_class"].items() if not count]
        if missing:
            summary["key_risks"].append(f"{name} has no annotations for {', '.join(missing)}.")
    write_json(root / "data" / "splits" / f"{snapshot.name}.json", snapshot_summary(read_snapshot(snapshot)))
    summary.update(
        status="prepared", fingerprint=snapshot.name, snapshot=_path(root, snapshot),
        split_policy=source, splits=split_stats,
        prepared_pair_statistics=summarize_records(records, geometry=False),
        snapshot_bytes=sum(file.stat().st_size for file in snapshot.rglob("*") if file.is_file()),
    )
    write_json(output, summary)
    summary["training_eda"] = _eda(root, config, splits["train"], snapshot.name)
    write_json(metrics / "unified" / snapshot.name / "dataset_report.json", summary)
    write_json(output, summary)
    LOGGER.info("Prepared Unified snapshot %s; report %s", snapshot, output)
    return snapshot


def load_unified_report(root: Path, config: Config) -> dict | None:
    """Read saved evidence for the notebook without scanning data or training."""
    metrics, _ = _artifact_paths(root, config)
    path = metrics / "dataset_report.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def reanalyze_unified(root: Path, config: Config) -> dict:
    """Recreate training-only figures from an existing local prepared snapshot."""
    report = load_unified_report(root, config)
    if not report or report["status"] != "prepared":
        raise ValueError("Prepare a validated Unified snapshot before generating training EDA.")
    snapshot = root / report["snapshot"]
    manifest = read_snapshot(snapshot)
    rows = []
    for entry in manifest["files"]:
        if entry["split"] != "train":
            continue
        image = snapshot / entry["image"]
        label = snapshot / entry["label"]
        boxes, errors = read_yolo(label)
        if errors:
            raise ValueError(f"Invalid prepared label {label}: {errors[0]}")
        rows.append(ImageRecord(
            image, label, Path(entry["source_path"]), "train", entry["width"], entry["height"],
            entry["image_sha256"], entry["pixel_sha256"], boxes,
            Provenance(entry["source"], entry["country"], entry["view"], entry["capture_group"]),
        ))
    report["training_eda"] = _eda(root, config, rows, snapshot.name)
    metrics, _ = _artifact_paths(root, config)
    write_json(metrics / "dataset_report.json", report)
    write_json(metrics / "unified" / snapshot.name / "dataset_report.json", report)
    return report
