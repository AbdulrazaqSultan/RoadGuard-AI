"""Run the approved Stage 4 profile once and leave a reviewable validation archive."""

import contextlib
import csv
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from filelock import FileLock

from roadguard.cli import run
from roadguard.training.stage4 import EXPERIMENT, PROFILE, stage4_preflight
from roadguard.utils.paths import project_root, write_json

GPU_FIELDS = ("temperature.gpu", "utilization.gpu", "memory.used", "power.draw")


def _gpu_samples(job: subprocess.Popen, log: Path) -> dict:
    """Sample device-level telemetry at startup and every 60 seconds."""
    values = {field: [] for field in GPU_FIELDS}
    samples = 0
    with log.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("utc", *GPU_FIELDS))
        while job.poll() is None:
            try:
                sample = subprocess.run(
                    [
                        "nvidia-smi",
                        "--id=0",
                        f"--query-gpu={','.join(GPU_FIELDS)}",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                row = sample.stdout.strip().split(",")
                if len(row) == len(GPU_FIELDS):
                    samples += 1
                    writer.writerow(
                        (datetime.now(UTC).isoformat(), *(part.strip() for part in row))
                    )
                    stream.flush()
                    for field, part in zip(GPU_FIELDS, row, strict=True):
                        with contextlib.suppress(ValueError):
                            values[field].append(float(part))
            except (OSError, subprocess.SubprocessError) as error:
                print(f"GPU telemetry unavailable: {error}", file=sys.stderr, flush=True)
                job.wait()
                break
            with contextlib.suppress(subprocess.TimeoutExpired):
                job.wait(timeout=60)
    return {
        "sample_count": samples,
        "interval_seconds": 60,
        "scope": "whole_gpu",
        "measures": {
            field: {"mean": sum(series) / len(series), "max": max(series)}
            for field, series in values.items()
            if series
        },
    }


def _run_stage4(root: Path, experiments: Path, status_file: Path) -> None:
    status = {"state": "preflight"}
    write_json(status_file, status)
    try:
        _, preflight = stage4_preflight(root)
        write_json(experiments / "stage4_preflight.json", preflight)
        status["state"] = "training"
        write_json(status_file, status)
        log = root / "artifacts/logs" / datetime.now(UTC).strftime("stage4-%Y%m%dT%H%M%SZ")
        log.mkdir(parents=True)
        status["log_directory"] = log.relative_to(root).as_posix()
        write_json(status_file, status)
        print(f"Stage 4 preflight passed. Logs: {log}", flush=True)
        with (
            (log / "stdout.log").open("w", encoding="utf-8") as stdout,
            (log / "stderr.log").open("w", encoding="utf-8") as stderr,
            subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-B",
                    str(root / "scripts/train_detector.py"),
                    "--config",
                    str(root / PROFILE),
                    "--dataset",
                    preflight["snapshot"],
                ],
                cwd=root,
                env=os.environ
                | {
                    "YOLO_OFFLINE": "true",
                    "YOLO_AUTOINSTALL": "false",
                    "WANDB_MODE": "disabled",
                    "MPLBACKEND": "Agg",
                    "PYTHONIOENCODING": "utf-8",
                },
                stdout=stdout,
                stderr=stderr,
            ) as job,
        ):
            telemetry = _gpu_samples(job, log / "gpu_telemetry.csv")
        write_json(log / "gpu_telemetry.json", telemetry)
        candidates = list(experiments.glob(f"{EXPERIMENT}-*"))
        if len(candidates) == 1:
            [directory] = candidates
            status["run"] = directory.name
            for name in ("stdout.log", "stderr.log", "gpu_telemetry.csv", "gpu_telemetry.json"):
                shutil.copy2(log / name, directory / name)
            shutil.copy2(root / PROFILE, directory / PROFILE.name)
        if job.returncode:
            raise RuntimeError(
                f"Training exited with code {job.returncode}. See {log / 'stderr.log'}."
            )
        if len(candidates) != 1:
            raise RuntimeError("Expected exactly one Stage 4 experiment folder after training.")
        status["state"] = "quality_gates"
        write_json(status_file, status)

        from roadguard.training.stage4_report import finalize_stage4

        result = finalize_stage4(root, directory, preflight, telemetry)
        status["state"] = result["status"]
    except BaseException as error:
        status.update(state="failed", error=str(error))
        raise
    finally:
        write_json(status_file, status)


def main() -> None:
    root = project_root()
    experiments = root / "artifacts/experiments"
    with FileLock(str(experiments / ".stage4.lock"), timeout=0):
        if any(experiments.glob(f"{EXPERIMENT}-*")):
            raise RuntimeError(
                "A Stage 4 run already exists. Review its archive before proceeding."
            )
        _run_stage4(root, experiments, experiments / "stage4_status.json")


if __name__ == "__main__":
    run(main)
