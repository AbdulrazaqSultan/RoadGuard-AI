"""Freeze the Stage 5 protocol and run the final test inference exactly once."""

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import yaml
from filelock import FileLock

from roadguard.data.dataset import CLASS_NAMES, SPLITS
from roadguard.data.prepare import write_dataset_yaml
from roadguard.data.validation import sha256
from roadguard.evaluation.stage5 import (
    BEST,
    BEST_SHA256,
    CODE,
    STAGE4,
    capture_predictions,
    evaluation_identity,
    verify_protected_inputs,
)
from roadguard.models.detection.yolo import load_detector
from roadguard.training.archive import open_experiment
from roadguard.training.experiment import reserve_run
from roadguard.training.screening import FINGERPRINT, verify_descriptor
from roadguard.utils.device import environment_info, require_cuda
from roadguard.utils.paths import write_json

EXPERIMENT = "stage6_final_test_evaluation"
CAPTURE = "test_capture.jsonl"
METRICS = "final_test_metrics.json"
LOGGER = logging.getLogger(__name__)
SIZE_CUTOFFS = (0.0058138503442412, 0.024389603605958765)
VALIDATION_REFERENCE = {
    "map50_95": 0.3234,
    "map50": 0.6395,
    "precision": 0.6935,
    "recall": 0.5808,
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def split_checksums(manifest: dict) -> dict:
    """Summarize the independently verified image and label hashes per partition."""
    records = {split: [] for split in SPLITS}
    for entry in manifest["files"]:
        files = [(entry[kind], entry[f"{kind}_sha256"]) for kind in ("image", "label")]
        records[entry["split"]].extend(files)
    return {
        split: {
            "images": len(items) // 2,
            "labels": len(items) // 2,
            "path_and_sha256_digest": hashlib.sha256(
                json.dumps(sorted(items), separators=(",", ":")).encode()
            ).hexdigest(),
        }
        for split, items in records.items()
    }


def _freeze(root: Path, experiments: Path) -> tuple[dict, dict, dict, Path]:
    stage4 = open_experiment(root / STAGE4)
    best, archived = stage4.best, stage4.metadata
    selected = read_json(experiments / "stage4_result.json")
    stage5 = read_json(experiments / "stage5_result.json")
    stage5_run = root / stage5["run"]
    prior = read_json(stage5_run / "metadata.json")
    integrity = stage5["integrity"]
    sizes = stage5["size_definition"]
    if (
        best is None
        or best["checkpoint"] != "weights/best.pt"
        or best["sha256"] != BEST_SHA256
        or best["epoch"] != 49
        or archived["status"] != "completed"
        or archived["model_name"] != "YOLO26s"
        or archived["dataset_fingerprint"] != FINGERPRINT
        or archived["selection_split"] != "val"
        or selected["status"] != "passed"
        or selected["test_evaluated"]
        or selected["selected_checkpoint"] != BEST.as_posix()
        or (root / BEST).resolve() != root / BEST
        or sha256(root / BEST) != BEST_SHA256
        or stage5["status"] != "passed"
        or stage5 != read_json(stage5_run / "stage5_result.json")
        or integrity != read_json(stage5_run / "integrity.json")
        or not stage5["stage6_recommended"]
        or integrity["status"] != "passed"
        or integrity["test_contents_opened"]
        or integrity["test_evaluated"]
        or prior["status"] != "passed"
        or integrity["snapshot_fingerprint"] != FINGERPRINT
        or integrity["checkpoint_sha256"] != BEST_SHA256
        or prior["checkpoint"] != BEST.as_posix()
        or prior["checkpoint_epoch"] != 49
        or (sizes["small_max_inclusive"], sizes["medium_max_inclusive"]) != SIZE_CUTOFFS
        or sizes != read_json(stage5_run / "size_analysis.json")["definition"]
        or prior["classes"] != {code: CLASS_NAMES[index] for index, code in CODE.items()}
    ):
        raise ValueError("Stage 4 selection or Stage 5 frozen evidence does not match.")
    config = archived["config"]
    snapshot = root / config["data"]["processed_root"] / FINGERPRINT
    verify_descriptor(root / STAGE4, snapshot)
    verify_descriptor(stage5_run, snapshot)
    require_cuda()
    environment = environment_info()
    identity = evaluation_identity(archived, environment)
    if (
        environment != archived["identity"]["versions"]
        or prior["identity"] != identity
        or any(
            round(archived["validation_metrics"][key], 4) != score
            for key, score in VALIDATION_REFERENCE.items()
        )
    ):
        raise ValueError("Evaluation configuration or Stage 4 validation references changed.")
    args = yaml.safe_load((root / STAGE4 / "args.yaml").read_text(encoding="utf-8"))
    from ultralytics.cfg import DEFAULT_CFG_DICT

    if (
        args["max_det"] != DEFAULT_CFG_DICT["max_det"]
        or args["agnostic_nms"]
        or DEFAULT_CFG_DICT["agnostic_nms"]
        or args["nms"] is not None
        or DEFAULT_CFG_DICT["nms"] is not None
        or archived["end_to_end_detection"]
        or args["conf"] != identity["evaluation_confidence_floor"]
        or args["iou"] != identity["nms_iou"]
        or args["imgsz"] != identity["imgsz"]
    ):
        raise ValueError("The native inference limits differ from the approved NMS protocol.")
    freeze = {
        "stage5_run": stage5["run"],
        "stage5_result_sha256": sha256(stage5_run / "stage5_result.json"),
        "stage4_checkpoint": BEST.as_posix(),
        "stage4_epoch": best["epoch"],
        "stage5_identity": prior["identity"],
        "test_identity": identity | {"split": "test"},
        "size_definition": sizes,
        "stage4_reference": VALIDATION_REFERENCE,
        "stage4_full_validation_metrics": archived["validation_metrics"],
        "stage5_validation_size_performance": stage5["size_analysis"],
        "class_mapping": prior["classes"],
        "stage4_config": config,
        "nms": {
            "mode": "class-aware standard NMS",
            "iou": identity["nms_iou"],
            "agnostic": DEFAULT_CFG_DICT["agnostic_nms"],
            "max_detections": DEFAULT_CFG_DICT["max_det"],
            "multi_label": True,
            "match_iou": identity["match_iou"],
            "rectangular_validation_batches": True,
        },
        "configuration_unchanged_since_stage5": True,
        "test_images_expected": archived["dataset_summary"]["test"]["images"],
        "training_occurred": False,
    }
    return freeze, integrity, archived, snapshot


def evaluate_test_once(root: Path) -> Path:
    """Acquire the one-shot slot, verify hashes, and archive the only test pass."""
    experiments = root / "artifacts/experiments"
    with FileLock(str(experiments / ".stage6_test.lock"), timeout=0):
        if list((root / STAGE4 / "evaluations").glob("test-*")) or any(
            read_json(path)["test_evaluation_started"]
            for path in experiments.glob(f"{EXPERIMENT}-*/metadata.json")
        ):
            raise ValueError("Final test inference already started. Use the archived capture.")
        freeze, prior, archived, snapshot = _freeze(root, experiments)
        detector = load_detector(root / BEST, trained=True)
        if bool(detector.model.end2end) != archived["end_to_end_detection"]:
            raise ValueError("Unexpected YOLO26s inference mode.")
        identity = freeze["test_identity"]
        experiment_id, output = reserve_run(
            experiments,
            EXPERIMENT,
            identity
            | {
                "size_definition": freeze["size_definition"],
            },
        )
        metadata = {
            "experiment_id": experiment_id,
            "status": "preflight",
            "model": archived["model_name"],
            "checkpoint": BEST.as_posix(),
            "checkpoint_epoch": freeze["stage4_epoch"],
            "snapshot_path": str(snapshot),
            "identity": identity,
            "training_occurred": False,
            "test_evaluation_started": False,
        }
        file = output / "metadata.json"
        write_json(file, metadata)
        write_json(output / "frozen_protocol.json", freeze)
        LOGGER.info("Frozen model and Stage 5 settings in %s", output)
        try:
            manifest, evidence = verify_protected_inputs(root, snapshot, prior, splits=SPLITS)
            preflight = evidence | {"splits": split_checksums(manifest)}
            write_json(output / "preflight.json", preflight)
            paths = {row["image"] for row in manifest["files"] if row["split"] == "test"}
            if len(paths) != freeze["test_images_expected"]:
                raise ValueError("The test set does not contain the approved frozen image count.")
            write_dataset_yaml(output / "dataset.yaml", snapshot, include_test=True)
            verify_descriptor(output, snapshot, include_test=True)
        except BaseException:
            metadata["status"] = "failed_preflight"
            write_json(file, metadata)
            raise
        metadata.update(
            status="inference",
            test_evaluation_started=True,
            test_evaluation_started_utc=datetime.now(UTC).isoformat(),
        )
        write_json(file, metadata)
        started = perf_counter()
        LOGGER.info("Starting the one final test pass on %d images", len(paths))
        try:
            metrics = capture_predictions(
                detector,
                snapshot,
                output,
                paths,
                config=freeze["stage4_config"],
                identity=identity,
                filename=CAPTURE,
                plots=True,
            )
            write_json(output / METRICS, metrics)
            metadata.update(
                status="captured",
                test_images=len(paths),
                test_evaluation_passes=1,
                capture_sha256=sha256(output / CAPTURE),
                inference_seconds=perf_counter() - started,
            )
        except BaseException:
            metadata["status"] = "failed_inference"
            raise
        finally:
            write_json(file, metadata)
        return output
