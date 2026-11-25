"""Load local, trusted Ultralytics YOLO detection checkpoints."""

from pathlib import Path
from typing import Any

from roadguard.data.dataset import CLASS_NAMES


def load_detector(weights: Path, *, trained: bool = False) -> Any:
    """Load a detection checkpoint and optionally enforce the RoadGuard taxonomy."""
    if weights.suffix.lower() != ".pt" or not weights.is_file():
        raise ValueError(f"A local .pt checkpoint is required: {weights}")
    from ultralytics import YOLO

    detector = YOLO(str(weights.resolve()))
    if detector.task != "detect":
        raise ValueError(f"Expected detection weights: {weights}")
    if trained and detector.names != CLASS_NAMES:
        raise ValueError(f"Checkpoint does not match the RoadGuard detection classes: {weights}")
    return detector
