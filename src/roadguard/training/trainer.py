"""Fine-tune YOLO detectors using only the training and validation partitions."""

import json
import logging
import platform
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from roadguard.config import Config
from roadguard.data.prepare import snapshot_summary, verify_snapshot, write_dataset_yaml
from roadguard.data.validation import sha256
from roadguard.evaluation.metrics import detection_metrics
from roadguard.models.detection.yolo import load_detector
from roadguard.training.callbacks import attach_callbacks
from roadguard.training.experiment import reserve_run
from roadguard.training.label_cache import isolated_label_cache
from roadguard.training.review import archive_review
from roadguard.utils.device import environment_info, require_cuda
from roadguard.utils.paths import write_json
from roadguard.utils.seed import seed_everything

LOGGER = logging.getLogger(__name__)
MODEL_VARIANTS = {"n", "s", "m"}
MODEL_FAMILIES = {"yolo11", "yolo26"}


def _training_args(settings: dict, model: dict) -> dict[str, Any]:
    if settings["scheduler"] not in {"cosine", "linear"}:
        raise ValueError("The scheduler must be cosine or linear.")
    if not 0 <= settings["seed"] < 2**32 - 1:
        raise ValueError("Training seed must be between 0 and 2**32 - 2.")
    size = settings["image_size"]
    if size < 32 or size % 32 or min(settings["batch_size"], settings["epochs"]) < 1:
        raise ValueError(
            "Use a positive batch size and epoch count and an image size divisible by 32."
        )
    if min(settings["workers"], settings["early_stopping"]) < 0:
        raise ValueError("Workers and early stopping must be nonnegative.")
    if settings["deterministic"] and (settings["cudnn_benchmark"] or settings["tf32"]):
        raise ValueError("Disable deterministic mode to opt in to cuDNN benchmark or TF32.")
    if not 0 < model["evaluation_confidence"] < 1 or not 0 < model["iou_threshold"] <= 1:
        raise ValueError("Use valid evaluation confidence and NMS IoU thresholds.")
    return {
        "device": 0,
        "epochs": settings["epochs"],
        "imgsz": size,
        "batch": settings["batch_size"],
        "workers": settings["workers"],
        "optimizer": settings["optimizer"],
        "lr0": settings["learning_rate"],
        "weight_decay": settings["weight_decay"],
        "cos_lr": settings["scheduler"] == "cosine",
        "patience": settings["early_stopping"],
        "save_period": settings["save_period"],
        "amp": settings["amp"],
        "deterministic": settings["deterministic"],
        "seed": settings["seed"],
        "conf": model["evaluation_confidence"],
        "iou": model["iou_threshold"],
        "multi_scale": 0.0,
        "rect": False,
        "save": True,
        "cache": False,
        "pretrained": True,
        "plots": True,
        "val": True,
        **settings["augmentation"],
    }


def _resume_run(directory: Path, identity: dict) -> tuple[dict, Path]:
    from ultralytics.nn.tasks import load_checkpoint

    with (directory / "metadata.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata["status"] == "completed" or metadata["identity"] != identity:
        raise ValueError(
            "Resume requires an unfinished run with the original configuration and data."
        )
    checkpoint = directory / "weights" / "last.pt"
    if not checkpoint.is_file():
        raise ValueError(f"No last checkpoint in {directory}")
    _, state = load_checkpoint(checkpoint)
    if state.get("optimizer") is None or state["epoch"] + 1 >= identity["training"]["epochs"]:
        raise ValueError("The checkpoint has no resumable optimizer state or remaining epochs.")
    # A resumed backend run must see the same files and the existing validation-best checkpoint.
    if not (directory / "weights" / "best.pt").is_file():
        raise ValueError("The original validation-best checkpoint is missing.")
    return metadata, checkpoint


def train(root: Path, config: Config, snapshot: Path, *, resume: Path | None = None) -> Path:
    """Start a tracked run, or explicitly continue an interrupted run in place."""
    settings, model, outputs = config["training"], config["model"], config["experiment"]
    device = require_cuda(settings["device"], deterministic=settings["deterministic"])
    arguments = _training_args(settings, model)
    family = model.get("model_family", "yolo11")
    if (
        family not in MODEL_FAMILIES
        or model["model_variant"] not in MODEL_VARIANTS
        or not model["pretrained"]
    ):
        raise ValueError("Select a pretrained YOLO11 or YOLO26 detection variant: n, s or m.")
    snapshot = snapshot.resolve()
    frozen = config["data"].get("snapshot_fingerprint")
    if frozen and snapshot.name != frozen:
        raise ValueError(f"This training profile requires the frozen snapshot {frozen}.")
    LOGGER.info("Verifying frozen training and validation files in %s", snapshot)
    manifest = verify_snapshot(snapshot, splits=("train", "val"))
    for split in ("train", "val"):
        if not any(
            entry["split"] == split and (snapshot / entry["label"]).stat().st_size
            for entry in manifest["files"]
        ):
            raise ValueError(f"The {split} partition has no target annotations.")

    model_name = f"{family}{model['model_variant']}"
    weights = root / (model["weights"] or f"{model_name}.pt")
    if not weights.is_file():
        raise FileNotFoundError(
            f"Place trusted pretrained weights at {weights} or set model.weights."
        )
    versions = environment_info()
    identity = {
        "dataset": snapshot.name,
        "model": model,
        "weights_sha256": sha256(weights),
        "training": settings,
        "versions": versions,
    }
    seed_everything(settings["seed"])
    if resume is None:
        experiment_id, directory = reserve_run(
            root / outputs["experiments_root"], outputs["experiment_name"], identity
        )
        metadata = {
            "experiment_id": experiment_id,
            "experiment_name": outputs["experiment_name"],
            "dataset_version": manifest["dataset_version"],
            "dataset_fingerprint": snapshot.name,
            "dataset_summary": snapshot_summary(manifest),
            "fitted_split": "train",
            "selection_split": "val",
            "model_name": f"{family.upper()}{model['model_variant']}",
            "pretrained_weights": {
                "path": str(weights.resolve()),
                "sha256": identity["weights_sha256"],
            },
            "platform": {"version": platform.platform(), "machine": platform.machine()},
            "device": str(device),
            "gpu": versions["gpu"],
            "identity": identity,
            "config": config,
        }
        checkpoint = weights
    else:
        directory = resume.resolve()

    # Lock a run through completion or failure so two resume processes cannot share it.
    from filelock import FileLock, Timeout

    try:
        with FileLock(str(directory / ".run.lock"), timeout=0):
            if resume is not None:
                metadata, checkpoint = _resume_run(directory, identity)
            cache = root / "data" / "interim" / "ultralytics_label_cache"
            with isolated_label_cache(snapshot, cache):
                _fit(
                    directory,
                    metadata,
                    checkpoint,
                    snapshot,
                    manifest,
                    arguments,
                    resume is not None,
                )
    except Timeout as error:
        raise RuntimeError(f"Another process is writing this run: {directory}") from error
    return directory


def _fit(
    directory: Path,
    metadata: dict,
    checkpoint: Path,
    snapshot: Path,
    manifest: dict,
    args: dict,
    resume: bool,
) -> None:
    output = directory / "metadata.json"
    dataset = directory / "dataset.yaml"
    write_dataset_yaml(dataset, snapshot, include_test=False)
    metadata["status"] = "running"
    metadata["data_paths"] = {
        "descriptor": str(dataset),
        "train_images": str(snapshot / "images" / "train"),
        "train_labels": str(snapshot / "labels" / "train"),
        "validation_images": str(snapshot / "images" / "val"),
        "validation_labels": str(snapshot / "labels" / "val"),
    }
    metadata.setdefault("started_utc", datetime.now(UTC).isoformat())
    write_json(output, metadata)
    try:
        import torch

        started = perf_counter()
        torch.cuda.reset_peak_memory_stats()
        model = load_detector(checkpoint, trained=resume)
        attach_callbacks(model, metadata["identity"]["training"])
        LOGGER.info("Starting CUDA training in %s", directory)
        common = {"data": str(dataset), "save_dir": str(directory)}
        if resume:
            results = model.train(resume=str(checkpoint), device=0, **common)
        else:
            results = model.train(**(args | common))
        if results is None or not (directory / "weights" / "best.pt").is_file():
            raise RuntimeError("No validation results or best checkpoint were saved.")
        metrics = detection_metrics(results)
        best = directory / "weights" / "best.pt"
        checkpoint_report = directory / "best_checkpoint.json"
        best_details = (
            json.loads(checkpoint_report.read_text(encoding="utf-8"))
            if checkpoint_report.is_file()
            else {"epoch": None}
        )
        best_details.update(
            checkpoint=best.relative_to(directory).as_posix(),
            bytes=best.stat().st_size,
            sha256=sha256(best),
            validation_map50_95=metrics["map50_95"],
        )
        write_json(checkpoint_report, best_details)
        backend = model.trainer
        final_epoch = backend.epoch + 1
        stopper = backend.stopper
        stale_epochs = final_epoch - stopper.best_epoch
        patience = metadata["config"]["training"]["early_stopping"]
        exhausted = bool(patience and stale_epochs >= patience)
        stopped_early = final_epoch < args["epochs"] and exhausted
        if stopped_early:
            stop_reason = "validation_patience_exhausted"
        elif final_epoch == args["epochs"]:
            stop_reason = "epoch_budget_completed"
        else:
            stop_reason = "other_backend_stop"
        metadata.update(
            status="completed",
            best_validation_metric=metrics["map50_95"],
            validation_metrics=metrics,
            effective_amp=bool(model.trainer.amp),
            effective_batch_size=int(model.trainer.batch_size),
            parameters=sum(parameter.numel() for parameter in model.model.parameters()),
            end_to_end_detection=bool(model.model.end2end),
            elapsed_seconds=perf_counter() - started,
            stopping={
                "final_epoch": final_epoch,
                "epoch_budget": args["epochs"],
                "stopped_early": stopped_early,
                "reason": stop_reason,
                "patience": patience,
                "epochs_without_improvement": stale_epochs,
                "patience_reached": exhausted,
                "stopper_best_epoch": stopper.best_epoch,
                "stopper_best_fitness": float(stopper.best_fitness),
            },
            peak_torch_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_torch_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            validation_review_status="pending",
        )
    except BaseException:
        metadata["status"] = "interrupted"
        raise
    finally:
        write_json(output, metadata)

    try:
        archive_review(model, snapshot, manifest, directory, metadata["config"])
    except Exception:
        LOGGER.exception("Training completed; validation prediction review could not be archived.")
        metadata["validation_review_status"] = "failed"
    else:
        metadata["validation_review_status"] = "saved"
    finally:
        write_json(output, metadata)
