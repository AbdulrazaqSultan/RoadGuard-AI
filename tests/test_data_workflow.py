"""Exercise deterministic groups, prepared snapshots and dataset statistics."""

import pytest
import yaml

from roadguard.data import prepare
from roadguard.data.prepare import publish_snapshot, verify_snapshot, write_dataset_yaml
from roadguard.data.split import assign_splits
from roadguard.data.statistics import sample_images, summarize
from roadguard.data.validation import validate_dataset

RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def test_group_split_and_reuse(tmp_path, make_pair):
    for index in range(10):
        make_pair(index)
    records = validate_dataset(tmp_path, "yolo").records
    groups = {record.relative_image.as_posix(): str(i // 2) for i, record in enumerate(records)}
    first = assign_splits(records, RATIOS, 42, groups)
    assert first == assign_splits(records[::-1], RATIOS, 42, groups)
    assert sum(map(len, first.values())) == 10
    assert all(first.values())
    for rows in first.values():
        partition_groups = {groups[row.relative_image.as_posix()] for row in rows}
        assert len(rows) == 2 * len(partition_groups)

    kwargs = {"version": "synthetic", "seed": 42, "ratios": RATIOS, "groups": groups}
    snapshot = publish_snapshot(tmp_path / "output", first, **kwargs)
    assert snapshot == publish_snapshot(tmp_path / "output", first, **kwargs)
    manifest = verify_snapshot(snapshot)
    assert validate_dataset(snapshot, "yolo_split").valid

    descriptor = tmp_path / "train.yaml"
    write_dataset_yaml(descriptor, snapshot, include_test=False)
    assert "test" not in yaml.safe_load(descriptor.read_text(encoding="utf-8"))
    (snapshot / manifest["files"][0]["label"]).write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Modified"):
        verify_snapshot(snapshot)


def test_training_snapshot_verification_never_reads_test_files(tmp_path, make_pair, monkeypatch):
    for index in range(8):
        make_pair(index)
    records = validate_dataset(tmp_path, "yolo").records
    split = assign_splits(records, RATIOS, 42)
    snapshot = publish_snapshot(
        tmp_path / "output", split, version="fixture", seed=42, ratios=RATIOS
    )
    original_hash = prepare.sha256

    def record_hash(path):
        assert path.parent.name != "test"
        return original_hash(path)

    monkeypatch.setattr(prepare, "sha256", record_hash)
    manifest = verify_snapshot(snapshot, splits=("train", "val"))
    assert len(manifest["files"]) == 8


def test_presupplied_group_crossing_splits_is_rejected(tmp_path, make_pair):
    for index, split in enumerate(("train", "val", "test")):
        make_pair(index, split=split)
    records = validate_dataset(tmp_path, "yolo_split").records
    groups = dict.fromkeys((row.relative_image.as_posix() for row in records), "same_route")
    with pytest.raises(ValueError, match="spans multiple splits"):
        assign_splits(records, RATIOS, 42, groups)


def test_statistics_count_images_boxes_and_distinct_classes(tmp_path, make_pair):
    make_pair(1, "0 0.5 0.5 0.4 0.4\n0 0.5 0.5 0.2 0.4\n3 0.5 0.5 0.4 0.4")
    make_pair(2, "")
    report = validate_dataset(tmp_path, "yolo")
    samples = sample_images(report.records, 3, 42)
    stats = summarize(report.records, samples)
    assert stats.total_images == 2
    assert stats.total_annotations == 3
    assert stats.empty_images == stats.multi_defect_images == 1
    assert stats.annotations_per_class["longitudinal_crack"] == 2
    assert stats.images_per_class["longitudinal_crack"] == 1
    assert stats.box_aspect_ratio == [2.0, 1.0, 2.0]
    assert stats.image_resolution == {"20x10": 2}
    assert len(stats.samples_per_class["pothole"]) == 1
