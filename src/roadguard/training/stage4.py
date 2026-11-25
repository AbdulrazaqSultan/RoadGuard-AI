"""Validate the selected Stage 4 inputs against archived evidence."""

import json
from pathlib import Path

from roadguard.config import load_config
from roadguard.data.prepare import verify_snapshot
from roadguard.data.validation import sha256
from roadguard.models.detection.yolo import load_detector
from roadguard.training.archive import open_experiment
from roadguard.training.screening import (
    FINGERPRINT,
    verify_descriptor,
    verify_inventory,
    verify_raw_archives,
)
from roadguard.utils.device import environment_info, require_cuda

EXPERIMENT = "stage4_yolo26s_long_training"
PROFILE = Path("configs/training/stage4_yolo26s.yaml")


def stage4_preflight(root: Path) -> tuple[dict, dict]:
    """Check CUDA, checkpoint provenance and the frozen train/validation protocol."""
    experiments = root / "artifacts/experiments"
    comparison = json.loads((experiments / "stage3b_comparison.json").read_text(encoding="utf-8"))
    integrity = json.loads((experiments / "stage3b_integrity.json").read_text(encoding="utf-8"))
    if (
        comparison["dataset_fingerprint"] != FINGERPRINT
        or comparison["split"] != "val"
        or comparison["test_evaluated"]
        or comparison["recommendation"]["model"] != "YOLO26s"
        or integrity["status"] != "passed"
        or integrity["snapshot_fingerprint"] != FINGERPRINT
        or integrity["test_contents_opened"]
        or integrity["test_evaluated"]
    ):
        raise ValueError("Stage 4 requires the integrity-verified Stage 3B YOLO26s selection.")
    selected = next(row for row in comparison["models"] if row["model"] == "YOLO26s")
    source = open_experiment(experiments / selected["run"])
    config = load_config(root, profile=root / PROFILE)
    archive = source.metadata
    original = archive["config"]
    expected_training = original["training"] | {"epochs": 50, "early_stopping": 10}
    snapshot = root / original["data"]["processed_root"] / FINGERPRINT
    weights = root / selected["best_checkpoint"]
    if (
        archive["status"] != "completed"
        or archive["selection_split"] != "val"
        or archive["dataset_fingerprint"] != FINGERPRINT
        or source.best is None
        or selected["best_checkpoint_sha256"] != source.best["sha256"]
        or weights != source.directory / source.best["checkpoint"]
        or sha256(weights) != source.best["sha256"]
    ):
        raise ValueError("The Stage 3B validation-best checkpoint failed its provenance check.")
    expected_model = original["model"] | {"weights": weights.relative_to(root).as_posix()}
    if (
        config["data"] != original["data"]
        or config["training"] != expected_training
        or config["model"] != expected_model
        or config["experiment"]["experiment_name"] != EXPERIMENT
        or config["experiment"]["experiments_root"] != "artifacts/experiments"
        or snapshot.resolve() != snapshot
    ):
        raise ValueError(
            "Stage 4 must reuse Stage 3B settings with the approved budget and patience."
        )
    verify_descriptor(source.directory, snapshot)
    require_cuda()
    versions = environment_info()
    if (
        versions["gpu"] != "NVIDIA GeForce RTX 4070 Laptop GPU"
        or versions != archive["identity"]["versions"]
    ):
        raise ValueError(
            "The installed CUDA environment differs from the recorded Stage 3B environment."
        )
    checkpoint = load_detector(weights, trained=True)
    if (
        Path(checkpoint.model.yaml.get("yaml_file", "")).stem != "yolo26s"
        or bool(checkpoint.model.end2end) != archive["end_to_end_detection"]
    ):
        raise ValueError(
            "The selected checkpoint does not match the archived YOLO26s configuration."
        )
    report = {
        "status": "passed",
        "fingerprint": FINGERPRINT,
        "snapshot": str(snapshot),
        "config": config,
        "environment": versions,
        "platform": archive["platform"],
        "cuda_torchvision_nms": "passed",
        "protected_inventory": integrity["protected_inventory"],
        "initialization": {
            "method": "warm_start_new_optimizer",
            "checkpoint": weights.relative_to(root).as_posix(),
            "sha256": source.best["sha256"],
            "stage3b_run": source.directory.name,
            "stage3b_metrics": archive["validation_metrics"],
        },
        "test_contents_opened": False,
        "test_evaluated": False,
    }
    verify_inventory(root, report)
    raw_hashes = verify_raw_archives(root / "data/raw/unified")
    if raw_hashes != integrity["raw_archive_sha256"]:
        raise ValueError("Raw archive checksums differ from the prior integrity record.")
    report["raw_archive_sha256"] = raw_hashes
    manifest = verify_snapshot(snapshot, splits=("train", "val"))
    report["train_val_checksum_counts"] = {
        split: sum(row["split"] == split for row in manifest["files"]) for split in ("train", "val")
    }
    return config, report
