"""Record self-contained defect events, evidence and explicit location provenance."""

import csv
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from roadguard.data.dataset import CLASS_NAMES, RDD_CLASSES, Box
from roadguard.vehicle.tracking import EventCandidate

CLASS_CODES = {index: code for code, index in RDD_CLASSES.items()}
PIPELINE_VERSION = "1.1"


@dataclass(frozen=True)
class Location:
    latitude: float | None
    longitude: float | None
    accuracy_m: float | None
    timestamp: str | None
    source: str
    simulated: bool


class LocationProvider(Protocol):
    def locate(self, seconds: float | None, captured_at: str | None) -> Location: ...


class NoLocation:
    def locate(self, seconds: float | None, captured_at: str | None) -> Location:
        return Location(None, None, None, captured_at, "unavailable", False)


@dataclass(frozen=True)
class SimulatedRoute:
    latitude: float
    longitude: float
    longitude_step_per_second: float

    def __post_init__(self) -> None:
        import math

        if (
            not -90 <= self.latitude <= 90
            or not -180 <= self.longitude <= 180
            or not math.isfinite(self.longitude_step_per_second)
        ):
            raise ValueError("Invalid simulated route.")

    def locate(self, seconds: float | None, captured_at: str | None) -> Location:
        longitude = self.longitude + (seconds or 0.0) * self.longitude_step_per_second
        longitude = (longitude + 180) % 360 - 180
        return Location(self.latitude, longitude, None, captured_at, "simulated_route", True)


def unknown_severity(box: Box) -> str:
    """Keep repair severity separate from detector confidence."""
    return "Unknown"


@dataclass(frozen=True)
class RoadDefectEvent:
    event_id: str
    source_id: str
    class_id: str
    class_name: str
    confidence: float
    first_seen_frame: int
    last_seen_frame: int
    first_seen_timestamp_s: float | None
    last_seen_timestamp_s: float | None
    representative_frame: int
    video_timestamp_s: float | None
    captured_at: str | None
    recorded_at: str
    detection_count: int
    track_ids: tuple[int, ...]
    bbox_xywh_normalized: tuple[float, float, float, float]
    frame_width: int
    frame_height: int
    location: Location
    severity: str
    evidence_frame_path: str
    crop_path: str | None
    model_name: str
    checkpoint: str | None
    session_id: str
    session_event_number: int
    pipeline_version: str = PIPELINE_VERSION


def pixel_bounds(box: Box, width: int, height: int) -> tuple[int, int, int, int]:
    left = max(0, min(width - 1, round((box.x - box.width / 2) * width)))
    top = max(0, min(height - 1, round((box.y - box.height / 2) * height)))
    right = max(left + 1, min(width, round((box.x + box.width / 2) * width)))
    bottom = max(top + 1, min(height, round((box.y + box.height / 2) * height)))
    return left, top, right, bottom


class EventRecorder:
    """Stream completed events into matching JSONL and CSV and save one evidence frame each."""

    def __init__(
        self,
        output: Path,
        run_id: str,
        source_id: str,
        location: LocationProvider,
        severity: Callable[[Box], str],
        model_name: str,
        checkpoint: str | None,
        *,
        crops: bool,
    ) -> None:
        self.output, self.run_id, self.source_id = output, run_id, source_id
        self.session_id = str(uuid4())
        self.location, self.severity = location, severity
        self.model_name, self.checkpoint = model_name, checkpoint
        self.crops = crops

    def __enter__(self) -> "EventRecorder":
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "evidence").mkdir(exist_ok=True)
        if self.crops:
            (self.output / "crops").mkdir(exist_ok=True)
        self._json = (self.output / "events.jsonl.partial").open("w", encoding="utf-8")
        try:
            self._csv = (self.output / "events.csv.partial").open("w", encoding="utf-8", newline="")
        except BaseException:
            self._json.close()
            raise
        self._writer = csv.DictWriter(
            self._csv, fieldnames=[item.name for item in fields(RoadDefectEvent)]
        )
        self._writer.writeheader()
        return self

    def __exit__(self, exc_type: Any, *args: Any) -> None:
        self._json.close()
        self._csv.close()
        if exc_type is None:
            for filename in ("events.jsonl", "events.csv"):
                (self.output / f"{filename}.partial").replace(self.output / filename)

    def record(self, candidate: EventCandidate) -> RoadDefectEvent:
        import cv2

        if candidate.best is None:
            raise ValueError("An event requires at least one observed detection.")
        frame, prediction = candidate.best
        box = prediction.box
        event_id = str(uuid4())
        evidence = f"evidence/{event_id}.jpg"
        crop = f"crops/{event_id}.jpg" if self.crops else None
        width, height = frame.dimensions
        if not cv2.imwrite(str(self.output / evidence), frame.pixels):
            raise OSError(f"Could not save {evidence}.")
        if crop is not None:
            left, top, right, bottom = pixel_bounds(box, width, height)
            if not cv2.imwrite(str(self.output / crop), frame.pixels[top:bottom, left:right]):
                raise OSError(f"Could not save {crop}.")
        event = RoadDefectEvent(
            event_id,
            self.source_id,
            CLASS_CODES[box.class_id],
            CLASS_NAMES[box.class_id],
            prediction.confidence,
            candidate.first_frame,
            candidate.last_frame,
            candidate.first_timestamp,
            candidate.last_timestamp,
            frame.index,
            frame.timestamp_s,
            frame.captured_at,
            datetime.now(UTC).isoformat(),
            candidate.observations,
            tuple(sorted(candidate.track_ids)),
            box.coordinates,
            width,
            height,
            self.location.locate(frame.timestamp_s, frame.captured_at),
            self.severity(box),
            evidence,
            crop,
            self.model_name,
            self.checkpoint,
            self.session_id,
            candidate.sequence,
        )
        data = asdict(event)
        self._json.write(json.dumps(data, allow_nan=False) + "\n")
        self._writer.writerow(
            {
                key: json.dumps(value, separators=(",", ":"))
                if isinstance(value, dict | list | tuple)
                else value
                for key, value in data.items()
            }
        )
        self._json.flush()
        self._csv.flush()
        return event
