"""Protect the Unified import, provenance, grouping and manifest contract."""

import csv
import json
import shutil
import tarfile
from io import BytesIO
from pathlib import Path

import pytest
import yaml
from PIL import Image

from roadguard.config import load_config
from roadguard.data.dataset import CLASS_NAMES
from roadguard.data.diagnostics import DiagnosticThresholds, flagged_boxes, summarize_records
from roadguard.data.duplicate_review import POLICY, select_reviewed_pairs
from roadguard.data.extract_unified import extract_archive
from roadguard.data.prepare import publish_snapshot, verify_snapshot
from roadguard.data.split import assign_source_splits
from roadguard.data.unified import discover_layout, infer_provenance, read_class_config
from roadguard.data.unified_release import REVISION
from roadguard.data.unified_workflow import process_unified
from roadguard.data.validation import sha256, validate_dataset

RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


@pytest.fixture
def unified(tmp_path):
    """Create a publisher-style local extract under a wrapper directory."""
    root = tmp_path / "raw"
    data = root / "extracted" / "wrapper"
    root.mkdir()
    (root / "rdd_merged.yaml").write_text(
        yaml.safe_dump(
            {
                "path": "/unused/publisher/path",
                "train": "images/train",
                "val": "images/val",
                "nc": 4,
                "names": [
                    "Longitudinal Crack (D00)",
                    "Transverse Crack (D10)",
                    "Alligator Crack (D20)",
                    "Pothole (D40)",
                ],
            }
        ),
        encoding="utf-8",
    )

    def add(name, number, split="train", labels=None):
        image = data / "images" / split / (name + ".png")
        label = data / "labels" / split / (name + ".txt")
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 32), (number, 80, 90)).save(image)
        if labels is None:
            labels = f"{number % 4} 0.5 0.5 0.4 0.3\n"
        label.write_text(labels, encoding="utf-8")
        return image

    return root, add


def test_publisher_class_mapping_and_layout(unified):
    root, add = unified
    add("rdd_India_1", 1)
    add("rdv_Spain_2", 2, "val")
    config, parts = discover_layout(root)
    assert config.name == "rdd_merged.yaml"
    assert [split for _, _, split in parts] == ["train", "val"]
    assert parts[0][0] == root / "extracted" / "wrapper" / "images" / "train"
    settings = read_class_config(config)
    settings["names"][0] = "Longitudinal Crack (D40)"
    config.write_text(yaml.safe_dump(settings), encoding="utf-8")
    with pytest.raises(ValueError, match="Class ID 0"):
        discover_layout(root)
    assert validate_dataset(root, "unified").issues[0].code == "invalid_unified_layout"
    settings["names"][0], settings["names"][3] = CLASS_NAMES[3], CLASS_NAMES[0]
    config.write_text(yaml.safe_dump(settings), encoding="utf-8")
    with pytest.raises(ValueError, match="Class ID 0"):
        discover_layout(root)


def test_source_inference_keeps_reliable_parent_images():
    rdd = infer_provenance("rdd_China_MotorBike_China_MotorBike_001344")
    assert (rdd.source, rdd.country, rdd.view) == ("RDD2022", "China", "motorbike")
    assert rdd.capture_group is None
    tile_a = infer_provenance("uav_lr_00418_top_right")
    tile_b = infer_provenance("uav_lr_00418_bottom_left")
    assert tile_a.capture_group == tile_b.capture_group == "uav:lr_00418"
    rdv = infer_provenance("rdv_China_Drone_000745_jpg.rf.b00e2e0ea819aa167b1b708e50a6818c")
    assert rdv.capture_group == "rdv:china_drone_000745"
    assert rdv.country == "China"
    assert infer_provenance("rdv_unknown_1").country is None
    assert infer_provenance("unrecognizable").source == "unknown"


def test_repartition_is_deterministic_and_source_balanced(unified, tmp_path):
    root, add = unified
    for index in range(72):
        domain = ("rdd_Japan", "rdv_Spain", "uav_original")[index % 3]
        add(f"{domain}_{index}", index, "train" if index % 10 < 8 else "val")
    add("uav_lr_00001_top_left", 100, "train")
    add("uav_lr_00001_top_right", 101, "val")
    report = validate_dataset(root, "unified")
    assert report.valid
    rows = report.records
    splits = assign_source_splits(rows, RATIOS, 42)
    assert splits == assign_source_splits(rows[::-1], RATIOS, 42)
    assert sum(map(len, splits.values())) == 74
    assert {len({row.provenance.source for row in group}) for group in splits.values()} == {3}
    assert {
        len({box.class_id for row in group for box in row.boxes}) for group in splits.values()
    } == {4}
    assert sum(any(row.provenance.capture_group for row in group) for group in splits.values()) == 1
    kwargs = {
        "version": "test-fixture",
        "seed": 42,
        "ratios": RATIOS,
        "source_release": {"revision": "fixture"},
    }
    snapshot = publish_snapshot(tmp_path / "processed", splits, **kwargs)
    assert publish_snapshot(tmp_path / "processed", splits, **kwargs) == snapshot
    manifest = verify_snapshot(snapshot)
    assert manifest["schema"] == 2
    with (snapshot / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        entries = list(csv.DictReader(stream))
    entry = next(row for row in entries if row["source"] == "RDD2022")
    assert entry["country"] == "Japan"
    assert entry["width"] == "64"
    assert entry["objects"] == "1"
    assert len(entry["image_sha256"]) == 64
    assert json.loads((snapshot / "metadata.json").read_text())["fingerprint"] == snapshot.name


def test_duplicates_and_invalid_boxes_block_preparation(unified, tmp_path):
    root, add = unified
    original = add("rdd_Japan_0", 0)
    duplicate = add("rdv_China_0", 4, "val")
    shutil.copyfile(original, duplicate)
    add("uav_0", 8, labels="0 0.5 0.5 -0.2 0.3\n2 0.5 0.5 nan 0.2\n")
    report = validate_dataset(root, "unified")
    report_counts = report.summary()["issue_counts"]
    assert report_counts == {"invalid_annotation": 2, "duplicate_image": 1, "split_leakage": 1}
    with pytest.raises(ValueError, match="duplicates"):
        assign_source_splits(report.records, RATIOS, 42)
    config = load_config(overrides=[f"data.dataset_root={root.as_posix()}"])
    with pytest.raises(ValueError, match="blocked preparation"):
        process_unified(tmp_path, config)
    summary = json.loads((tmp_path / "artifacts/metrics/dataset_report.json").read_text())
    assert summary["status"] == "validation_failed"
    assert summary["fingerprint"] is None
    assert not (tmp_path / "data/processed").exists()
    assert original.is_file() and duplicate.is_file()


@pytest.fixture
def reviewed_duplicate(unified, tmp_path):
    root, add = unified
    raw_paths = [
        add("rdd_Japan_1", 1, labels="0 0.5 0.5 0.2 0.4\n"),
        add("rdd_Japan_2", 2, "val", "0 0.51 0.5 0.2 0.4\n"),
    ]
    shutil.copyfile(*raw_paths)
    add("uav_3", 3)
    add("rdv_4", 4)
    report = validate_dataset(root, "unified")
    first, second = (next(row for row in report.records if row.image == p) for p in raw_paths)
    review = {
        "policy": POLICY,
        "revision": REVISION,
        "rationale": "Fixture for an exact image pair with a reviewed geometry difference.",
        "image_sha256": first.image_hash,
        "pixel_sha256": first.pixel_hash,
    }
    for role, row in (("canonical", first), ("excluded", second)):
        review[role] = {
            "image": row.relative_image.as_posix(),
            "label": row.label.relative_to(root).as_posix(),
            "label_sha256": sha256(row.label),
        }
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    return report, review_path, add, first, second


def test_reviewed_duplicate_keeps_exact_canonical_annotation(reviewed_duplicate, tmp_path):
    report, review_path, _, canonical, excluded = reviewed_duplicate
    records, resolutions, audit = select_reviewed_pairs(report, review_path)
    assert canonical in records and excluded not in records
    assert canonical.boxes[0].x == 0.5
    assert excluded.boxes[0].x == 0.51
    assert {issue.code for issue in resolutions} == {"duplicate_image", "split_leakage"}
    assert audit["annotations_excluded"] == 1
    assert not report.valid
    config = load_config(
        overrides=[
            f"data.dataset_root={report.root.as_posix()}",
            f"data.duplicate_review={review_path.as_posix()}",
        ]
    )
    assert process_unified(tmp_path, config, validate_only=True) is None
    assert (tmp_path / "artifacts/metrics/unified_duplicate_resolution.json").is_file()
    assert canonical.image.is_file() and excluded.image.is_file()


def test_review_rejects_changed_label_bytes(reviewed_duplicate):
    report, review_path, _, _, excluded = reviewed_duplicate
    excluded.label.write_text("0 0.52 0.5 0.2 0.4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pinned identity"):
        select_reviewed_pairs(validate_dataset(report.root, "unified"), review_path)


def test_review_keeps_unrelated_validation_errors_blocking(reviewed_duplicate, tmp_path):
    report, review_path, add, _, _ = reviewed_duplicate
    add("uav_5", 5, labels="0 0.5 0.5 0.0 0.4\n")
    config = load_config(
        overrides=[
            f"data.dataset_root={report.root.as_posix()}",
            f"data.duplicate_review={review_path.as_posix()}",
        ]
    )
    with pytest.raises(ValueError, match="blocked preparation"):
        process_unified(tmp_path, config)
    summary = json.loads((tmp_path / "artifacts/metrics/dataset_report.json").read_text())
    assert summary["unresolved_publisher_findings"] == {"invalid_annotation": 1}
    assert summary["fingerprint"] is None


def test_box_diagnostics_preserve_labels(unified):
    root, add = unified
    add("rdd_Japan_0", 0, labels="0 0.5 0.5 0.01 0.6\n3 0.5 0.5 0.8 0.8\n")
    add("rdd_Japan_1", 1, "val", "")
    rows = validate_dataset(root, "unified").records
    flags = list(flagged_boxes(rows, DiagnosticThresholds(0.001, 1024, 0.25, 10)))
    assert len(flags) == 2
    assert "elongated" in flags[0][-1]
    assert "large_area" in flags[1][-1]
    summary = summarize_records(rows)
    assert summary["objects"] == 2
    assert summary["negatives"] == summary["multi_class_images"] == 1
    assert summary["largest_to_smallest_class_ratio"] is None


def _tar(path: Path, name: str) -> None:
    with tarfile.open(path, "w:gz") as stream:
        entry = tarfile.TarInfo(name)
        entry.size = 4
        stream.addfile(entry, BytesIO(b"test"))


def test_archive_paths_and_duplicates_are_checked(tmp_path):
    archive = tmp_path / "archive.tar.gz"
    target = tmp_path / "files"
    target.mkdir()
    _tar(archive, "images/val/test.jpg")
    assert extract_archive(archive, target) == (1, 4)
    with pytest.raises(FileExistsError):
        extract_archive(archive, target)
    _tar(archive, "../escape")
    with pytest.raises(ValueError, match="Unsupported member"):
        extract_archive(archive, target)
    assert not (tmp_path / "escape").exists()
