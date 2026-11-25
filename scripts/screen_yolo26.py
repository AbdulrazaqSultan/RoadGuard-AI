"""Run exactly two Stage 3B training jobs using the archived Stage 3A protocol."""

import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from roadguard.cli import run
from roadguard.data.prepare import verify_snapshot
from roadguard.data.validation import sha256
from roadguard.training.screening import (
    VARIANTS,
    preflight,
    verify_descriptor,
    verify_inventory,
    verify_raw_archives,
)
from roadguard.utils.paths import project_root, write_json


def train_variant(
    root: Path,
    config: dict,
    report: dict,
    variant: str,
    log_directory: Path,
) -> str:
    """Archive one CUDA process and its effective frozen validation protocol."""
    experiments = root / config["experiment"]["experiments_root"]
    name = f"stage3b_yolo26{variant}_screening"
    if any(experiments.glob(f"{name}-*")):
        raise ValueError(f"A Stage 3B archive already exists for {name}.")
    weights = report["weights"][variant]
    if sha256(Path(weights["path"])) != weights["sha256"]:
        raise ValueError(f"Pretrained checkpoint changed: {weights['path']}")
    profile = deepcopy(config)
    profile["experiment"]["experiment_name"] = name
    profile["model"].update(model_family="yolo26", model_variant=variant, weights=weights["path"])
    snapshot = Path(report["snapshot"])
    profile_path = log_directory / f"{name}.json"
    write_json(profile_path, profile)
    stdout_path = log_directory / f"{name}.stdout.log"
    stderr_path = log_directory / f"{name}.stderr.log"
    print(f"Launching {name}. Training log {stderr_path}", flush=True)
    env = os.environ | {
        "YOLO_OFFLINE": "true",
        "YOLO_AUTOINSTALL": "false",
        "MPLBACKEND": "Agg",
        "PYTHONIOENCODING": "utf-8",
        "WANDB_MODE": "disabled",
    }
    args = [
        sys.executable,
        "-u",
        "-B",
        str(root / "scripts/train_detector.py"),
        "--config",
        str(profile_path),
        "--dataset",
        str(snapshot),
    ]
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        completed = subprocess.run(
            args,
            cwd=root,
            env=env,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )

    matches = list(experiments.glob(f"{name}-*"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one run archive for {name}. Review {stderr_path}.")
    archive = matches[0]
    for source, filename in (
        (profile_path, "screening_config.json"),
        (stdout_path, "stdout.log"),
        (stderr_path, "stderr.log"),
    ):
        shutil.copyfile(source, archive / filename)
    metadata_path = archive / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        pretrained_release=weights["release"],
        logs={"training": "stderr.log", "stdout": "stdout.log", "stderr": "stderr.log"},
    )
    write_json(metadata_path, metadata)
    if completed.returncode != 0 or metadata["status"] != "completed":
        raise RuntimeError(f"Training failed for {name}. Review {archive / 'stderr.log'}.")
    if (
        metadata["validation_review_status"] != "saved"
        or metadata["effective_batch_size"] != 8
        or not metadata["effective_amp"]
        or metadata["end_to_end_detection"]
    ):
        raise ValueError(f"{name} left the matched protocol or has incomplete outputs.")
    verify_descriptor(archive, snapshot)
    verify_inventory(root, report)
    print(f"Completed {archive.name}", flush=True)
    return archive.name


def main() -> None:
    """Preflight, train the two variants, and audit preservation of the dataset."""
    root = project_root()
    os.environ.setdefault("YOLO_OFFLINE", "true")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    experiments = root / "artifacts/experiments"
    if any(experiments.glob("stage3b_yolo26*_screening-*")):
        raise ValueError("Stage 3B experiment archives already exist.")
    log_directory = root / "artifacts/logs" / f"stage3b-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    log_directory.mkdir(parents=True)
    config, report = preflight(root)
    write_json(experiments / "stage3b_preflight.json", report)
    print("Stage 3B preflight passed. Using the frozen Stage 3A training protocol.", flush=True)
    runs = {}
    for variant in VARIANTS:
        runs[f"YOLO26{variant}"] = train_variant(root, config, report, variant, log_directory)
    verify_snapshot(Path(report["snapshot"]), splits=("train", "val"))
    raw_archives = verify_raw_archives(root / "data/raw/unified")
    write_json(
        experiments / "stage3b_integrity.json",
        {
            "status": "passed",
            "snapshot_fingerprint": report["fingerprint"],
            "train_val_checksums_match": True,
            "test_contents_opened": False,
            "test_evaluated": False,
            "protected_inventory": verify_inventory(root, report),
            "raw_archive_sha256": raw_archives,
            "runs": runs,
        },
    )
    print("Stage 3B training and final dataset integrity checks passed.", flush=True)


if __name__ == "__main__":
    run(main)
