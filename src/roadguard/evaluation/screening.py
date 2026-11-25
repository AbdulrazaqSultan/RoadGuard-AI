"""Compare the four screened architectures using archived validation evidence."""

import csv
import io
import json
import re
from math import isfinite
from pathlib import Path

from roadguard.data.dataset import CLASS_NAMES, RDD_CLASSES
from roadguard.data.validation import sha256
from roadguard.models.detection.yolo import load_detector
from roadguard.training.archive import open_experiment
from roadguard.training.screening import FINGERPRINT, verify_descriptor
from roadguard.utils.paths import write_json, write_text

_TRAIN_MEMORY = re.compile(r"\b[1-5]/5[ \t]+([0-9.]+)G\b")
_ACCURACY = ("map50_95", "recall", "D40_recall", *(f"{code}_ap50_95" for code in RDD_CLASSES))
_DEPLOYMENT = ("total_latency_ms", "checkpoint_bytes")


def _peak_logged_vram(log_path: Path) -> float | None:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return max((float(value) for value in _TRAIN_MEMORY.findall(text)), default=None)


def _stage3a_log(root: Path, name: str, run: Path) -> Path:
    for path in sorted((root / "artifacts/logs").glob(f"stage3-{name.lower()}-*.log")):
        if run.name in path.read_text(encoding="utf-8", errors="replace"):
            return path
    raise FileNotFoundError(f"Could not find the original training log for {run.name}.")


def _read_model(root: Path, name: str, run_path: Path, snapshot: Path, protocol: dict) -> dict:
    run = open_experiment(run_path)
    metadata = run.metadata
    if (
        metadata["status"] != "completed"
        or metadata["dataset_fingerprint"] != snapshot.name
        or metadata["selection_split"] != "val"
        or metadata["fitted_split"] != "train"
        or metadata["config"]["training"] != protocol
        or not metadata["effective_amp"]
        or not run.best
        or len(run.epoch_metrics) != protocol["epochs"]
    ):
        raise ValueError(f"{name} is outside the common screening protocol.")
    verify_descriptor(run_path, snapshot)
    checkpoint = run_path / run.best["checkpoint"]
    if sha256(checkpoint) != run.best["sha256"]:
        raise ValueError(f"{name} best checkpoint failed its recorded checksum.")
    parameters = metadata.get("parameters")
    if parameters is None:
        # Stage 3A already archived its validation scores. Read architecture only.
        model = load_detector(checkpoint, trained=True)
        parameters = sum(parameter.numel() for parameter in model.model.parameters())
    metrics = metadata["validation_metrics"]
    speed = metrics["latency_ms_per_image"]
    preprocess, inference, postprocess = (
        speed[key] for key in ("preprocess", "inference", "postprocess")
    )
    total = preprocess + inference + postprocess
    if (
        not all(isfinite(value) and value >= 0 for value in (preprocess, inference, postprocess))
        or total <= 0
    ):
        raise ValueError(f"{name} has invalid archived latency values.")
    log_path = (
        run_path / "stdout.log" if name.startswith("YOLO26") else _stage3a_log(root, name, run_path)
    )
    row = {
        "model": name,
        "run": run_path.name,
        "best_checkpoint": checkpoint.relative_to(root).as_posix(),
        "best_checkpoint_sha256": run.best["sha256"],
        **{key: metrics[key] for key in ("map50_95", "map50", "precision", "recall")},
        **{
            f"{code}_ap50_95": metrics["per_class"][CLASS_NAMES[class_id]]["ap50_95"]
            for code, class_id in RDD_CLASSES.items()
        },
        "D40_recall": metrics["per_class"][CLASS_NAMES[RDD_CLASSES["D40"]]]["recall"],
        "best_epoch": run.best["epoch"],
        "parameter_count": parameters,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "peak_logged_training_vram_gib": _peak_logged_vram(log_path),
        "peak_torch_cuda_reserved_bytes": metadata.get("peak_torch_cuda_reserved_bytes"),
        "preprocess_latency_ms": preprocess,
        "inference_latency_ms": inference,
        "postprocess_latency_ms": postprocess,
        "total_latency_ms": total,
        "estimated_detector_fps": 1000 / total,
    }
    return row


def pareto_models(rows: list[dict]) -> list[str]:
    """Identify models with no peer that improves every accuracy and deployment metric."""

    def score(row: dict) -> tuple:
        return (*(row[key] for key in _ACCURACY), *(-row[key] for key in _DEPLOYMENT))

    scored = [(row["model"], score(row)) for row in rows]
    return [
        name
        for name, values in scored
        if not any(
            other != name
            and all(a >= b for a, b in zip(alternate, values, strict=True))
            and alternate != values
            for other, alternate in scored
        )
    ]


def screening_comparison(root: Path) -> dict:
    """Read the integrity-verified Stage 3B results and both original Stage 3A archives."""
    experiments = root / "artifacts/experiments"
    preflight = json.loads((experiments / "stage3b_preflight.json").read_text(encoding="utf-8"))
    integrity = json.loads((experiments / "stage3b_integrity.json").read_text(encoding="utf-8"))
    snapshot = Path(preflight["snapshot"])
    if (
        integrity["status"] != "passed"
        or snapshot != root / "data/processed" / FINGERPRINT
        or integrity["snapshot_fingerprint"] != FINGERPRINT
        or integrity["test_evaluated"]
    ):
        raise ValueError("Stage 3B dataset integrity is not verified.")
    runs = preflight["stage3a_runs"] | integrity["runs"]
    if set(runs) != {"YOLO11n", "YOLO11s", "YOLO26n", "YOLO26s"}:
        raise ValueError("All four screened model archives are required.")
    protocol = open_experiment(experiments / runs["YOLO11n"]).metadata["config"]["training"]
    rows = [
        _read_model(root, name, experiments / directory, snapshot, protocol)
        for name, directory in runs.items()
    ]
    return {
        "stage": "3B",
        "dataset_fingerprint": FINGERPRINT,
        "split": "val",
        "test_evaluated": False,
        "training_protocol": protocol,
        "latency_measurement": {
            "source": "Ultralytics final validation at imgsz=640 on the RTX 4070 Laptop GPU.",
            "unit": "milliseconds per image, averaged over batched validation",
            "total": "preprocessing + inference + postprocessing",
            "estimated_fps_scope": (
                "Detector throughput. Camera, disk I/O, GPS, severity and networking "
                "are outside the estimate."
            ),
        },
        "vram_measurement": (
            "Highest sampled CUDA reservation in training progress, in GiB. "
            "The actual training peak may be higher."
        ),
        "selection_evidence": {
            "accuracy": list(_ACCURACY),
            "practical_deployment": list(_DEPLOYMENT),
            "training_memory_is_context": True,
        },
        "pareto_models": pareto_models(rows),
        "models": rows,
    }


def save_screening_comparison(root: Path, report: dict, recommendation: dict) -> Path:
    """Archive the measured results and the stated accuracy/deployment tradeoff."""
    if recommendation["model"] not in report["pareto_models"]:
        raise ValueError("The recommendation must be supported by the comparison frontier.")
    output = root / "artifacts/experiments/stage3b_comparison.json"
    write_json(output, report | {"recommendation": recommendation})
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(report["models"][0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(report["models"])
    write_text(output.with_suffix(".csv"), stream.getvalue())
    return output
