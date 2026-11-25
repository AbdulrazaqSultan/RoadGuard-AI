"""Capture the selected Stage 4 detector's validation outputs without changing the model."""

import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, TextIO

from roadguard.data.dataset import CLASS_NAMES, RDD_CLASSES
from roadguard.data.prepare import verify_snapshot, write_dataset_yaml
from roadguard.data.validation import sha256
from roadguard.evaluation.evaluator import validate_checkpoint
from roadguard.evaluation.metrics import detection_metrics
from roadguard.models.detection.yolo import load_detector
from roadguard.training.archive import open_experiment
from roadguard.training.experiment import reserve_run
from roadguard.training.screening import (
    FINGERPRINT,
    verify_descriptor,
    verify_inventory,
    verify_raw_archives,
)
from roadguard.utils.device import environment_info, require_cuda
from roadguard.utils.paths import write_json

LOGGER = logging.getLogger(__name__)
EXPERIMENT = "stage5_validation_error_analysis"
STAGE4 = Path("artifacts/experiments/stage4_yolo26s_long_training-bfeb3ff5d6c9-001")
BEST = STAGE4 / "weights/best.pt"
BEST_SHA256 = "ef989dee5457f29bd70942eb610d0dc8c5ba81da6e8099a898747b5defae61c6"
CODE = {index: code for code, index in RDD_CLASSES.items()}
CAPTURE = "validation_capture.jsonl"
MATCH_IOU = 0.5


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_protected_inputs(
    root: Path, snapshot: Path, baseline: dict, *, splits: tuple[str, ...] = ("val",)
) -> tuple[dict, dict]:
    """Verify the requested partition bytes, raw archives, checkpoint and inventory."""
    if (
        snapshot != root / "data/processed" / FINGERPRINT
        or snapshot.resolve() != snapshot
        or (root / BEST).resolve() != root / BEST
        or sha256(root / BEST) != BEST_SHA256
    ):
        raise ValueError("Expected the exact frozen snapshot and Stage 4 epoch-49 best.pt.")
    inventory = verify_inventory(root, baseline | {"snapshot": str(snapshot)})
    raw = verify_raw_archives(root / "data/raw/unified")
    if raw != baseline["raw_archive_sha256"]:
        raise ValueError("Raw archive hashes differ from Stage 4.")
    manifest = verify_snapshot(snapshot, splits=splits)
    evidence = {
        "status": "passed",
        "snapshot_fingerprint": FINGERPRINT,
        "validation_checksums_match": "val" in splits,
        "verified_checksum_splits": list(splits),
        "protected_inventory": inventory,
        "raw_archive_sha256": raw,
        "checkpoint_sha256": BEST_SHA256,
        "test_contents_opened": "test" in splits,
        "test_evaluated": False,
    }
    return manifest, evidence


def _capture_validator(snapshot: Path, paths: set[str], stream: TextIO, seen: set[str]) -> type:
    """Use native NMS, labels, metrics and geometry, with a read-only export hook."""
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils import ops

    class CaptureValidator(DetectionValidator):
        def plot_val_samples(self, *args: Any, **kwargs: Any) -> None:
            """Use the project's curated examples in place of first-batch figures."""

        def plot_predictions(self, *args: Any, **kwargs: Any) -> None:
            """Use the project's curated examples in place of first-batch figures."""

        def update_metrics(self, preds: list[dict], batch: dict) -> None:
            super().update_metrics(preds, batch)
            for index, prediction in enumerate(preds):
                prepared = self._prepare_batch(index, batch)
                image = Path(prepared["im_file"]).relative_to(snapshot).as_posix()
                if image not in paths or image in seen:
                    raise ValueError(f"Unexpected validator image: {image}")
                seen.add(image)
                scaled = self.scale_preds(prediction, prepared)
                target = self.scale_preds({"bboxes": prepared["bboxes"]}, prepared)
                height, width = prepared["ori_shape"]

                def boxes(tensor: Any, size: tuple[int, int]) -> list[list[float]]:
                    tensor = tensor.float()
                    return (ops.xyxy2xywh(tensor) / tensor.new_tensor(size * 2)).tolist()

                row = {
                    "image": image,
                    "width": width,
                    "height": height,
                    "labels": [
                        [int(cid), *box]
                        for cid, box in zip(
                            prepared["cls"].tolist(),
                            boxes(target["bboxes"], (width, height)),
                            strict=True,
                        )
                    ],
                    "predictions": [
                        [int(cid), confidence, *box]
                        for cid, confidence, box in zip(
                            scaled["cls"].tolist(),
                            scaled["conf"].tolist(),
                            boxes(scaled["bboxes"], (width, height)),
                            strict=True,
                        )
                    ],
                }
                stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")

    return CaptureValidator


def evaluation_identity(archived: dict, environment: dict) -> dict:
    """Recover the final detection validation settings used by Stage 5."""
    config = archived["config"]
    model = config["model"]
    return {
        "split": "val",
        "snapshot": FINGERPRINT,
        "checkpoint_sha256": BEST_SHA256,
        "imgsz": config["training"]["image_size"],
        "batch_size": 2 * archived["effective_batch_size"],
        "workers": config["training"]["workers"],
        "device": 0,
        "cpu_threads": None,
        "quantize": 16 if archived["effective_amp"] else None,
        "evaluation_confidence_floor": model["evaluation_confidence"],
        "operating_confidence": model["confidence_threshold"],
        "nms_iou": model["iou_threshold"],
        "match_iou": MATCH_IOU,
        "environment": environment,
    }


def capture_predictions(
    detector: Any,
    snapshot: Path,
    output: Path,
    paths: set[str],
    config: dict,
    identity: dict,
    *,
    filename: str = CAPTURE,
    plots: bool = False,
) -> dict:
    """Run one native split pass and publish a complete image-level prediction capture."""
    split = identity["split"]
    descriptor = verify_descriptor(output, snapshot, include_test=split == "test")
    seen: set[str] = set()
    temporary = output / f"{filename}.partial"
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        metrics = validate_checkpoint(
            detector,
            snapshot,
            descriptor,
            output,
            config,
            split=split,
            batch=identity["batch_size"],
            validator=_capture_validator(snapshot, paths, stream, seen),
            quantize=identity["quantize"],
            plots=plots,
        )
    if seen != paths:
        raise ValueError(f"Incomplete capture. Missing {len(paths - seen)} {split} images.")
    temporary.replace(output / filename)
    return detection_metrics(metrics)


def capture_validation(root: Path) -> Path:
    """Verify Stage 4 evidence and run a single complete native validation pass."""
    experiments = root / "artifacts/experiments"
    result = _read(experiments / "stage4_result.json")
    stage4 = open_experiment(root / STAGE4)
    archived = stage4.metadata
    prior = _read(experiments / "stage4_integrity.json")
    if (
        result["status"] != "passed"
        or result["selected_checkpoint"] != BEST.as_posix()
        or result["test_evaluated"]
        or stage4.best is None
        or stage4.best["sha256"] != BEST_SHA256
        or stage4.best["epoch"] != 49
        or archived["selection_split"] != "val"
        or archived["model_name"] != "YOLO26s"
        or archived["dataset_fingerprint"] != FINGERPRINT
        or prior["status"] != "passed"
        or prior["snapshot_fingerprint"] != FINGERPRINT
        or prior["test_contents_opened"]
        or prior["test_evaluated"]
        or list(CODE.values()) != ["D00", "D10", "D20", "D40"]
    ):
        raise ValueError("Stage 4 selection or integrity records do not match the approved inputs.")
    config = archived["config"]
    snapshot = root / config["data"]["processed_root"] / FINGERPRINT
    verify_descriptor(root / STAGE4, snapshot)
    LOGGER.info("Verifying the checkpoint, raw archives and frozen validation files")
    manifest, evidence = verify_protected_inputs(root, snapshot, prior)
    paths = {row["image"] for row in manifest["files"] if row["split"] == "val"}
    if len(paths) != 3019:
        raise ValueError("Stage 5 requires all 3,019 approved validation images.")
    require_cuda()
    environment = environment_info()
    if environment != archived["identity"]["versions"]:
        raise ValueError("Stage 5 requires the verified Stage 4 CUDA environment.")
    detector = load_detector(root / BEST, trained=True)
    if bool(detector.model.end2end) != archived["end_to_end_detection"]:
        raise ValueError("Unexpected YOLO26s inference mode.")
    identity = evaluation_identity(archived, environment)
    experiment_id, output = reserve_run(experiments, EXPERIMENT, identity)
    metadata = {
        "experiment_id": experiment_id,
        "status": "inference",
        "identity": identity,
        "snapshot_path": str(snapshot),
        "checkpoint": BEST.as_posix(),
        "checkpoint_epoch": stage4.best["epoch"],
        "model": archived["model_name"],
        "classes": {code: CLASS_NAMES[index] for index, code in CODE.items()},
        "stage4_validation_metrics": archived["validation_metrics"],
        "class_order": list(CODE.values()),
        "val_images": len(paths),
    }
    write_json(output / "preflight.json", evidence)
    write_json(output / "metadata.json", metadata)
    descriptor = output / "dataset.yaml"
    write_dataset_yaml(descriptor, snapshot, include_test=False)
    started = perf_counter()
    LOGGER.info("Capturing validation predictions in %s", output)
    try:
        metrics = capture_predictions(detector, snapshot, output, paths, config, identity)
        write_json(output / "validation_metrics.json", metrics)
        metadata.update(status="captured", inference_elapsed_seconds=perf_counter() - started)
    except BaseException:
        metadata["status"] = "failed_inference"
        raise
    finally:
        write_json(output / "metadata.json", metadata)
    return output
