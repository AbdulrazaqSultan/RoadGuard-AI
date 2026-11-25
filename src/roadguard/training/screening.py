"""Guard the frozen validation-only Stage 3B screening protocol."""

import hashlib
import json
import os
import platform
from pathlib import Path

import yaml

from roadguard.data.dataset import CLASS_NAMES
from roadguard.data.prepare import verify_snapshot
from roadguard.data.unified_release import ARCHIVES
from roadguard.data.validation import sha256
from roadguard.training.archive import open_experiment
from roadguard.utils.device import environment_info, require_cuda

FINGERPRINT = "0eb0721e553c9eec39850e9648c7af04a38c6b13cbc67d98788b38ceab89629f"
WEIGHT_RELEASE = "https://github.com/ultralytics/assets/releases/tag/v8.4.0"
VARIANTS = ("n", "s")
WEIGHT_HASHES = {
    "n": "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef",
    "s": "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b",
}


def inventory(root: Path) -> dict:
    """Fingerprint paths, sizes and modification times without opening image or label contents."""
    rows = []
    for directory, folders, files in os.walk(root):
        base = Path(directory)
        rows.extend(
            (base.joinpath(name).relative_to(root).as_posix(), "directory") for name in folders
        )
        for name in files:
            path = base / name
            stat = path.stat()
            rows.append((path.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns))
    if not rows:
        raise ValueError(f"Cannot inventory missing or empty dataset tree: {root}")
    digest = hashlib.sha256(json.dumps(sorted(rows)).encode()).hexdigest()
    return {"entries": len(rows), "path_size_mtime_sha256": digest}


def verify_descriptor(run: Path, snapshot: Path, *, include_test: bool = False) -> Path:
    """Require the exact frozen descriptor for the authorized partitions."""
    path = run / "dataset.yaml"
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        "path": snapshot.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": CLASS_NAMES,
    }
    if include_test:
        expected["test"] = "images/test"
    if settings != expected:
        raise ValueError(f"Unexpected descriptor for the approved snapshot: {path}")
    return path


def stage3a_protocol(experiments: Path, snapshot: Path) -> tuple[dict, dict[str, str]]:
    """Read the completed YOLO11 runs and recover their common training protocol."""
    report = json.loads((experiments / "stage3_comparison.json").read_text(encoding="utf-8"))
    if report["dataset_fingerprint"] != FINGERPRINT or report["split"] != "val":
        raise ValueError("Stage 3A must use the approved validation protocol.")
    runs = {
        f"YOLO11{variant}": report[role]["run"]
        for variant, role in zip(VARIANTS, ("baseline", "challenger"), strict=True)
    }
    configs = []
    for name in runs.values():
        path = experiments / name
        run = open_experiment(path)
        if (
            run.metadata["status"] != "completed"
            or run.metadata["dataset_fingerprint"] != FINGERPRINT
            or run.metadata["selection_split"] != "val"
            or not run.metadata["effective_amp"]
            or len(run.epoch_metrics) != 5
        ):
            raise ValueError(f"Stage 3A run does not match its recorded protocol: {name}")
        verify_descriptor(path, snapshot)
        configs.append(run.metadata["config"])
    first, second = configs
    evaluation = ("confidence_threshold", "evaluation_confidence", "iou_threshold")
    if first["training"] != second["training"] or any(
        first["model"][key] != second["model"][key] for key in evaluation
    ):
        raise ValueError("Stage 3A training or validation settings differ.")
    settings = first["training"]
    if tuple(settings[key] for key in ("epochs", "image_size", "batch_size", "seed", "amp")) != (
        5,
        640,
        8,
        42,
        True,
    ):
        raise ValueError("Expected the approved five-epoch Stage 3A protocol.")
    return first, runs


def check_pretrained(weights: Path, variant: str) -> dict:
    """Verify the installed backend can rebuild the pretrained YOLO26 architecture."""
    from copy import deepcopy

    from ultralytics.nn.tasks import DetectionModel

    from roadguard.models.detection.yolo import load_detector

    digest = sha256(weights)
    if digest != WEIGHT_HASHES[variant]:
        raise ValueError(f"Official YOLO26 release checksum mismatch: {weights}")
    model = load_detector(weights)
    architecture = model.model.yaml
    if Path(architecture.get("yaml_file", "")).stem != f"yolo26{variant}":
        raise ValueError(f"Expected the official YOLO26{variant} detection architecture.")
    candidate = DetectionModel(deepcopy(architecture), nc=len(CLASS_NAMES), verbose=False)
    candidate.load(model.model, verbose=False)
    if getattr(candidate.model[-1], "one2one_cv2", None) is None:
        raise ValueError(
            f"Installed Ultralytics cannot construct the YOLO26 detection head: {weights}"
        )
    return {
        "path": str(weights),
        "sha256": digest,
        "release": WEIGHT_RELEASE,
        "architecture": f"YOLO26{variant}",
        "dual_detection_heads": True,
        "default_end_to_end_inference": bool(candidate.end2end),
    }


def preflight(root: Path) -> tuple[dict, dict]:
    """Verify content for train/validation and register identity for the protected trees."""
    raw = root / "data/raw/unified"
    snapshot = root / "data/processed" / FINGERPRINT
    if snapshot.resolve() != snapshot:
        raise ValueError("The frozen dataset must resolve to the approved prepared path.")
    config, baselines = stage3a_protocol(root / "artifacts/experiments", snapshot)
    if config["data"]["snapshot_fingerprint"] != FINGERPRINT:
        raise ValueError("Stage 3A configuration has a different snapshot pin.")
    require_cuda()
    weights_dir = root / "artifacts/models/pretrained"
    weights = {key: check_pretrained(weights_dir / f"yolo26{key}.pt", key) for key in VARIANTS}
    before = {"raw": inventory(raw), "snapshot": inventory(snapshot)}
    archives = verify_raw_archives(raw)
    manifest = verify_snapshot(snapshot, splits=("train", "val"))
    train_val = {
        split: sum(row["split"] == split for row in manifest["files"]) for split in ("train", "val")
    }
    if train_val != {"train": 24147, "val": 3019}:
        raise ValueError(f"Unexpected frozen train/validation counts: {train_val}")
    return config, {
        "status": "passed",
        "snapshot": str(snapshot),
        "fingerprint": FINGERPRINT,
        "train_val_checksum_counts": train_val,
        "test_contents_opened": False,
        "cuda_and_torchvision_nms": "passed",
        "environment": environment_info()
        | {
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "protected_inventory": before,
        "raw_archive_sha256": archives,
        "weights": weights,
        "stage3a_runs": baselines,
    }


def verify_inventory(root: Path, preflight_report: dict) -> dict:
    """Stop if any protected file path, size or modification time changed."""
    result = {
        "raw": inventory(root / "data/raw/unified"),
        "snapshot": inventory(Path(preflight_report["snapshot"])),
    }
    if result != preflight_report["protected_inventory"]:
        raise RuntimeError("Frozen snapshot or raw data changed. Screening stopped.")
    return result


def verify_raw_archives(raw: Path) -> dict[str, str]:
    """Confirm that the source archives still match the pinned dataset release."""
    observed = {}
    for relative, (size, expected) in ARCHIVES.items():
        path = raw / relative
        if path.stat().st_size != size or sha256(path) != expected:
            raise ValueError(f"Raw archive checksum mismatch: {path}")
        observed[relative] = expected
    return observed
