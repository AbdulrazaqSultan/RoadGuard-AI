"""OpenCV frame acquisition shared by video, cameras and future stream adapters."""

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import Any, Protocol


@dataclass(frozen=True)
class Frame:
    index: int
    timestamp_s: float | None
    captured_at: str | None
    pixels: Any

    @property
    def dimensions(self) -> tuple[int, int]:
        height, width = self.pixels.shape[:2]
        return width, height


class FrameSource(Protocol):
    """A decoded-frame stream with optional FPS and reliable EOF signaling."""

    fps: float | None
    is_live: bool

    def read(self) -> Frame | None: ...


class OpenCVSource:
    """Read BGR frames from a local video path or a camera device index."""

    def __init__(self, source: str | int) -> None:
        if (isinstance(source, int) and source < 0) or not str(source).strip():
            raise ValueError("Provide a video path or a nonnegative camera index.")
        self.source = source
        self.is_live = isinstance(source, int)
        self.fps: float | None = None
        self.reported_frames: int | None = None
        self.timestamp_fallbacks = 0
        self._capture = None
        self._index = 0
        self._previous_timestamp: float | None = None

    def __enter__(self) -> "OpenCVSource":
        import cv2

        self._capture = cv2.VideoCapture(self.source)
        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            raise OSError(f"OpenCV could not open source: {self.source}")
        try:
            self._index = 0
            self._previous_timestamp = None
            self.timestamp_fallbacks = 0
            self._started = perf_counter()
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            self.fps = float(fps) if math.isfinite(fps) and fps > 0 else None
            frames = self._capture.get(cv2.CAP_PROP_FRAME_COUNT)
            self.reported_frames = (
                int(frames) if not self.is_live and math.isfinite(frames) and frames > 0 else None
            )
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *args: Any) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def read(self) -> Frame | None:
        import cv2

        if self._capture is None:
            raise RuntimeError("Open the frame source before reading.")
        ok, pixels = self._capture.read()
        if not ok or pixels is None or not pixels.size:
            if self.is_live:
                raise OSError(f"Camera read failed or device disconnected: {self.source}")
            if self.reported_frames is not None and self._index < self.reported_frames:
                raise OSError(f"Video decode stopped after {self._index}/{self.reported_frames}.")
            return None
        if self.is_live:
            timestamp, captured_at = perf_counter() - self._started, datetime.now(UTC).isoformat()
        else:
            captured_at = None
            timestamp = self._capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            previous = self._previous_timestamp
            if (
                not math.isfinite(timestamp)
                or timestamp < 0
                or (previous is not None and timestamp <= previous)
            ):
                self.timestamp_fallbacks += 1
                timestamp = (
                    max(self._index / self.fps, (previous or 0.0) + 1 / self.fps)
                    if self.fps and previous is not None
                    else (self._index / self.fps if self.fps else None)
                )
        frame = Frame(self._index, timestamp, captured_at, pixels)
        self._index += 1
        self._previous_timestamp = timestamp
        return frame
