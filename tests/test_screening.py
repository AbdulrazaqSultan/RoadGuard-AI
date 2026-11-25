"""Keep Stage 3B screening validation-only and sensitive to deployment tradeoffs."""

import pytest
import yaml

from roadguard.data.dataset import CLASS_NAMES
from roadguard.evaluation.screening import pareto_models
from roadguard.training.screening import inventory, verify_descriptor


def test_screening_descriptor_rejects_a_test_reference(tmp_path):
    snapshot = tmp_path / "snapshot"
    descriptor = {
        "path": snapshot.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": CLASS_NAMES,
    }
    file = tmp_path / "dataset.yaml"
    file.write_text(yaml.safe_dump(descriptor), encoding="utf-8")
    assert verify_descriptor(tmp_path, snapshot) == file
    descriptor["test"] = "images/test"
    file.write_text(yaml.safe_dump(descriptor), encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected descriptor"):
        verify_descriptor(tmp_path, snapshot)


def test_inventory_detects_new_cache_files(tmp_path):
    (tmp_path / "labels").mkdir()
    before = inventory(tmp_path)
    (tmp_path / "labels" / "train.cache").write_bytes(b"sample cache")
    assert inventory(tmp_path) != before


def test_screening_frontier_considers_speed_and_class_specific_accuracy():
    reference = {
        "model": "balanced",
        "map50_95": 0.2,
        "recall": 0.4,
        "D40_recall": 0.45,
        "D00_ap50_95": 0.2,
        "D10_ap50_95": 0.2,
        "D20_ap50_95": 0.2,
        "D40_ap50_95": 0.2,
        "total_latency_ms": 3.0,
        "checkpoint_bytes": 6000000,
    }
    accurate = reference | {"model": "highest-map", "map50_95": 0.21, "total_latency_ms": 5.0}
    potholes = reference | {"model": "best-pothole-recall", "map50_95": 0.19, "D40_recall": 0.50}
    dominated = reference | {"model": "dominated", "map50_95": 0.18, "total_latency_ms": 4.0}
    assert pareto_models([reference, accurate, potholes, dominated]) == [
        "balanced",
        "highest-map",
        "best-pothole-recall",
    ]
