"""Shared validation, splitting and EDA entry points for scripts and notebooks."""

import logging
from dataclasses import asdict
from pathlib import Path

from roadguard.config import Config
from roadguard.data.dataset import CLASS_NAMES
from roadguard.data.prepare import publish_snapshot
from roadguard.data.split import assign_splits, read_groups
from roadguard.data.statistics import DatasetStatistics, sample_images, save_charts, summarize
from roadguard.data.unified_workflow import process_unified, reanalyze_unified
from roadguard.data.validation import ValidationReport, validate_dataset
from roadguard.utils.paths import write_json

LOGGER = logging.getLogger(__name__)


def inspect_input(root: Path, config: Config) -> ValidationReport:
    """Inspect configured input and write its report even when errors are found."""
    data = config["data"]
    report = validate_dataset(root / data["dataset_root"], data["source_format"])
    path = root / config["experiment"]["metrics_root"] / "dataset_validation.json"
    write_json(path, report.summary())
    LOGGER.info("Dataset validation report: %s", path)
    if not report.valid:
        raise ValueError(f"Dataset validation failed. Review {path}.")
    return report


def partitions_for(root: Path, config: Config, report: ValidationReport) -> tuple[dict, dict | None]:
    """Resolve group assignments, retaining any existing split labels."""
    data = config["data"]
    groups = read_groups(root / data["groups_csv"], report.records) if data["groups_csv"] else None
    splits = assign_splits(report.records, data["split"], config["training"]["seed"], groups)
    return splits, groups


def prepare(root: Path, config: Config, *, validate_only: bool = False) -> Path | None:
    """Validate input and optionally publish a reusable versioned YOLO snapshot."""
    if config["data"]["source_format"] == "unified":
        return process_unified(root, config, validate_only=validate_only)
    report = inspect_input(root, config)
    try:
        splits, groups = partitions_for(root, config, report)
    except (OSError, ValueError) as error:
        report.add("invalid_split", (report.root,), str(error))
        write_json(
            root / config["experiment"]["metrics_root"] / "dataset_validation.json",
            report.summary(),
        )
        raise
    if validate_only:
        return None
    data = config["data"]
    snapshot = publish_snapshot(
        root / data["processed_root"],
        splits,
        version=data["dataset_version"],
        seed=config["training"]["seed"],
        ratios=data["split"],
        groups=groups,
    )
    split_summary = {
        name: {
            "images": len(records),
            "classes_present": sorted({
                CLASS_NAMES[box.class_id] for row in records for box in row.boxes
            }),
        }
        for name, records in splits.items()
    }
    write_json(root / "data" / "splits" / f"{snapshot.name}.json", split_summary)
    LOGGER.info("Prepared snapshot: %s", snapshot)
    return snapshot


def analyze(root: Path, config: Config) -> DatasetStatistics | dict:
    """Analyze only the designated training partition to keep holdouts blind."""
    if config["data"]["source_format"] == "unified":
        return reanalyze_unified(root, config)
    report = inspect_input(root, config)
    splits, _ = partitions_for(root, config, report)
    experiment = config["experiment"]
    rows = splits["train"]
    samples = sample_images(rows, experiment["sample_images_per_class"], config["training"]["seed"])
    statistics = summarize(rows, samples)
    write_json(
        root / experiment["metrics_root"] / "dataset_statistics.json",
        {"split": "train", **asdict(statistics)},
    )
    save_charts(statistics, samples, root / experiment["figures_root"] / "dataset")
    LOGGER.info("Analyzed %d training images", statistics.total_images)
    return statistics
