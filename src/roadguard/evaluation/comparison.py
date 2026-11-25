"""Compare completed detector runs under a shared validation protocol."""

from pathlib import Path

from roadguard.training.archive import ExperimentArchive, open_experiment
from roadguard.utils.paths import write_json

METRICS = ("map50_95", "map50", "precision", "recall", "macro_f1")


def _row(run: ExperimentArchive) -> dict:
    metadata = run.metadata
    if (
        metadata["status"] != "completed"
        or metadata.get("selection_split") != "val"
        or not run.best
    ):
        raise ValueError(f"A completed validation-selected run is required: {run.directory}")
    return {
        "run": run.directory.name,
        "model_variant": metadata["config"]["model"]["model_variant"],
        "best_epoch": run.best["epoch"],
        "epochs_completed": len(run.epoch_metrics),
        "checkpoint": run.best["checkpoint"],
        "checkpoint_sha256": run.best["sha256"],
        "validation_metrics": metadata["validation_metrics"],
    }


def compare_validation_runs(baseline: Path, challenger: Path, output: Path) -> dict:
    """Save matched validation metrics and select the higher mAP50–95 checkpoint."""
    first, second = map(open_experiment, (baseline, challenger))
    rows = {"baseline": _row(first), "challenger": _row(second)}
    baseline_meta, challenger_meta = first.metadata, second.metadata
    fingerprint = baseline_meta["dataset_fingerprint"]
    if fingerprint != challenger_meta["dataset_fingerprint"]:
        raise ValueError("Both runs must use the same prepared snapshot.")
    first_cfg, second_cfg = baseline_meta["config"], challenger_meta["config"]
    model_evaluation_fields = ("evaluation_confidence", "iou_threshold")
    if (
        first_cfg["training"] != second_cfg["training"]
        or baseline_meta["effective_amp"] != challenger_meta["effective_amp"]
        or rows["baseline"]["epochs_completed"] != rows["challenger"]["epochs_completed"]
        or any(
            first_cfg["model"][key] != second_cfg["model"][key] for key in model_evaluation_fields
        )
    ):
        raise ValueError("A matched comparison requires the same training and validation protocol.")
    if rows["baseline"]["model_variant"] == rows["challenger"]["model_variant"]:
        raise ValueError("Choose different model variants for this comparison.")

    baseline_metrics = rows["baseline"]["validation_metrics"]
    challenger_metrics = rows["challenger"]["validation_metrics"]
    changes = {key: challenger_metrics[key] - baseline_metrics[key] for key in METRICS}
    winner = "challenger" if changes["map50_95"] > 0 else "baseline"
    report = {
        "dataset_fingerprint": fingerprint,
        "split": "val",
        "selection_metric": "map50_95",
        "tie_break": "baseline",
        "test_evaluated": False,
        "seed": first_cfg["training"]["seed"],
        "training": first_cfg["training"],
        "selected_run": rows[winner]["run"],
        "challenger_minus_baseline": changes,
        **rows,
    }
    write_json(output, report)
    return report
