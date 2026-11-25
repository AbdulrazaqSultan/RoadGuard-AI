"""Read saved experiment results without training, inference or CUDA."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path


def _json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


@dataclass(frozen=True)
class ExperimentArchive:
    """Paths and structured values for a saved training attempt."""

    directory: Path
    metadata: dict
    epoch_metrics: list[dict]
    best: dict | None
    validation_review: dict | None

    def plots(self, *names: str) -> list[Path]:
        """Find available trainer plots by name."""
        return [self.directory / name for name in names if (self.directory / name).is_file()]


def list_experiments(directory: Path) -> list[str]:
    """List reviewable experiment attempts."""
    return sorted(path.parent.name for path in directory.glob("*/metadata.json"))


def open_experiment(directory: Path) -> ExperimentArchive:
    """Read snapshots of saved metrics and plots for a completed or interrupted run."""
    metadata = _json(directory / "metadata.json")
    if metadata is None:
        raise FileNotFoundError(f"No experiment metadata in {directory}")
    history = directory / "results.csv"
    rows = []
    if history.is_file():
        with history.open(encoding="utf-8", newline="") as stream:
            rows = [
                {
                    key.strip(): value.strip()
                    for key, value in row.items()
                    if key is not None and isinstance(value, str)
                }
                for row in csv.DictReader(stream)
            ]
    return ExperimentArchive(
        directory,
        metadata,
        rows,
        _json(directory / "best_checkpoint.json"),
        _json(directory / "validation_review" / "review.json"),
    )


def reopen_validation_review(run: Path, snapshot: Path) -> dict:
    """Regenerate missing validation review artifacts from the saved checkpoint only."""
    from roadguard.data.prepare import verify_snapshot
    from roadguard.data.validation import sha256
    from roadguard.models.detection.yolo import load_detector
    from roadguard.training.review import archive_review
    from roadguard.utils.device import require_cuda

    archive = open_experiment(run)
    if archive.metadata["status"] != "completed":
        raise ValueError("Use a completed training attempt for prediction review.")
    if archive.validation_review is not None and all(
        (run / row["figure"]).is_file() for row in archive.validation_review["samples"]
    ):
        return archive.validation_review
    snapshot = snapshot.resolve()
    if archive.metadata["dataset_fingerprint"] != snapshot.name:
        raise ValueError("Use the exact prepared snapshot recorded by training.")
    manifest = verify_snapshot(snapshot, splits=("val",))
    checkpoint = run / "weights" / "best.pt"
    checksum = archive.best.get("sha256") if archive.best else None
    if checksum and checksum != sha256(checkpoint):
        raise ValueError("Best checkpoint checksum differs from the run archive.")
    require_cuda()
    return archive_review(
        load_detector(checkpoint, trained=True),
        snapshot,
        manifest,
        run,
        archive.metadata["config"],
    )
