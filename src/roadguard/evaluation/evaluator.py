"""Evaluate a completed run's best checkpoint with its recorded settings."""

import json
from pathlib import Path
from typing import Any

from roadguard.data.prepare import verify_snapshot, write_dataset_yaml
from roadguard.data.validation import sha256
from roadguard.evaluation.metrics import detection_metrics
from roadguard.models.detection.yolo import load_detector
from roadguard.training.experiment import reserve_run
from roadguard.training.label_cache import isolated_label_cache
from roadguard.utils.device import environment_info, require_cuda
from roadguard.utils.paths import write_json


def validate_checkpoint(
    detector: Any,
    snapshot: Path,
    descriptor: Path,
    output: Path,
    config: dict,
    *,
    split: str,
    batch: int,
    **options: Any,
) -> Any:
    """Apply the recorded validation protocol with a cache outside the snapshot."""
    settings, model_settings = (config[key] for key in ("training", "model"))
    with isolated_label_cache(snapshot, output / "label_cache", splits=(split,)):
        return detector.val(
            data=str(descriptor),
            split=split,
            device=0,
            batch=batch,
            imgsz=settings["image_size"],
            workers=settings["workers"],
            conf=model_settings["evaluation_confidence"],
            iou=model_settings["iou_threshold"],
            save_dir=str(output),
            **({"plots": True} | options),
        )


def evaluate(run: Path, snapshot: Path, split: str, *, batch_size: int | None = None) -> Path:
    """Record validation or explicit test metrics separately from training."""
    device = require_cuda()
    if split not in {"val", "test"}:
        raise ValueError("Choose val or test for evaluation.")
    with (run / "metadata.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata["status"] != "completed":
        raise ValueError("Evaluate a completed run.")
    snapshot = snapshot.resolve()
    if snapshot.name != metadata["dataset_fingerprint"]:
        raise ValueError("Evaluation must use the exact snapshot recorded by training.")
    verify_snapshot(snapshot, splits=(split,))
    checkpoint = run / "weights" / "best.pt"
    settings, model_settings = (metadata["config"][key] for key in ("training", "model"))
    batch = settings["batch_size"] if batch_size is None else batch_size
    if batch < 1:
        raise ValueError("Evaluation batch size must be positive.")
    identity = {
        "split": split,
        "snapshot": snapshot.name,
        "checkpoint_sha256": sha256(checkpoint),
        "batch_size": batch,
        "environment": environment_info(),
    }
    evaluation_id, output = reserve_run(run / "evaluations", split, identity)
    yaml_path = output / "dataset.yaml"
    write_dataset_yaml(yaml_path, snapshot, include_test=split == "test")

    import torch

    torch.cuda.reset_peak_memory_stats(device)
    detector = load_detector(checkpoint, trained=True)
    parameters = sum(parameter.numel() for parameter in detector.model.parameters())
    metrics = validate_checkpoint(
        detector, snapshot, yaml_path, output, metadata["config"], split=split, batch=batch
    )
    report = output / "metrics.json"
    write_json(
        report,
        {
            **identity,
            "experiment_id": metadata["experiment_id"],
            "evaluation_id": evaluation_id,
            "image_size": settings["image_size"],
            "confidence_floor": model_settings["evaluation_confidence"],
            "nms_iou_threshold": model_settings["iou_threshold"],
            "parameters": parameters,
            "model_size_bytes": checkpoint.stat().st_size,
            "peak_torch_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
            **detection_metrics(metrics),
        },
    )
    return report
