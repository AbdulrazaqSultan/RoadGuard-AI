"""Independent, schematic road clip and deterministic detections for software checks."""

from pathlib import Path
from typing import Any

from roadguard.data.dataset import Box
from roadguard.evaluation.box_matching import Prediction
from roadguard.vehicle.events import pixel_bounds

FRAMES = 48
FPS = 15.0
SIZE = (640, 360)


def _potholes(index: int) -> tuple[Box, Box]:
    return (
        Box(3, 0.34 + 0.0007 * index, 0.60 + 0.002 * index, 0.13, 0.08),
        Box(3, 0.69 - 0.0005 * index, 0.67 + 0.0015 * index, 0.10, 0.09),
    )


def write_software_video(path: Path) -> None:
    """Create a fixture from geometry; no images or frames are read from a dataset."""
    import cv2
    import numpy as np

    width, height = SIZE
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    try:
        if not writer.isOpened():
            raise OSError(f"Could not create the synthetic MP4: {path}")
        rng = np.random.default_rng(42)
        texture = rng.integers(-7, 8, (height, width, 1), dtype=np.int16)
        for index in range(FRAMES):
            frame = np.full((height, width, 3), (158, 180, 144), dtype=np.uint8)
            cv2.fillPoly(
                frame,
                [np.array([(265, 78), (371, 78), (635, 359), (5, 359)], dtype=np.int32)],
                (78, 79, 81),
            )
            cv2.line(frame, (265, 78), (5, 359), (180, 180, 180), 2)
            cv2.line(frame, (371, 78), (635, 359), (180, 180, 180), 2)
            for base in (88, 149, 222, 305):
                y = min(height - 12, base + index % 14)
                cv2.line(frame, (320, y), (320, y + 10), (205, 217, 226), 2)
            for box, visible in zip(
                _potholes(index), (2 <= index <= 23, 7 <= index <= 29), strict=True
            ):
                if visible:
                    left, top, right, bottom = pixel_bounds(box, width, height)
                    center = (left + right) // 2, (top + bottom) // 2
                    axes = (right - left) // 2, (bottom - top) // 2
                    cv2.ellipse(frame, center, axes, -12, 0, 360, (49, 49, 53), -1)
                    cv2.ellipse(frame, center, axes, -12, 0, 360, (104, 105, 107), 2)
            if index == 32:
                cv2.line(frame, (335, 292), (408, 302), (40, 42, 44), 2)
            frame = np.clip(frame.astype(np.int16) + texture, 0, 255).astype(np.uint8)
            cv2.putText(
                frame,
                "SYNTHETIC SOFTWARE INPUT",
                (14, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (20, 29, 24),
                2,
            )
            writer.write(frame)
    finally:
        writer.release()


class ScriptedDetections:
    """Inject known observations, including a gap, simultaneous defects and one transient."""

    def __init__(self) -> None:
        self.index = 0

    def __call__(self, pixels: Any) -> list[Prediction]:
        index = self.index
        self.index += 1
        first, second = _potholes(index)
        result = []
        if 2 <= index <= 10 or 15 <= index <= 23:
            result.append(Prediction(first, 0.96 if index == 20 else 0.76))
        if 7 <= index <= 29:
            result.append(Prediction(second, 0.94 if index == 17 else 0.73))
        if index == 32:
            result.append(Prediction(Box(1, 0.58, 0.82, 0.11, 0.04), 0.82))
        return result
