"""Associate road-surface observations and merge short, strongly overlapping fragments."""

import math
from dataclasses import dataclass, field, replace

from roadguard.data.dataset import Box
from roadguard.evaluation.box_matching import Prediction, box_iou
from roadguard.vehicle.source import Frame


class TemporalAssociator:
    """Greedy, class-consistent, one-to-one assignment in normalized image coordinates."""

    def __init__(
        self,
        *,
        max_gap_frames: int,
        min_iou: float,
        max_center_distance: float,
        max_area_ratio: float,
    ) -> None:
        if not (
            max_gap_frames >= 1
            and 0 < min_iou <= 1
            and 0 <= max_center_distance <= 1
            and math.isfinite(max_area_ratio)
            and max_area_ratio >= 1
        ):
            raise ValueError("Invalid temporal association settings.")
        self.max_gap_frames = max_gap_frames
        self.min_iou = min_iou
        self.max_center_distance = max_center_distance
        self.max_area_ratio = max_area_ratio
        self._active: dict[int, tuple[int, Box]] = {}
        self._next_id = 1

    def close(self, track_id: int) -> None:
        """Retire an active anchor at a pipeline-level event boundary."""
        del self._active[track_id]

    def update(
        self, frame: int, detections: list[Prediction]
    ) -> tuple[list[tuple[int, Prediction]], list[int]]:
        expired = [
            track_id
            for track_id, (last, _) in self._active.items()
            if frame - last > self.max_gap_frames
        ]
        for track_id in expired:
            del self._active[track_id]
        candidates = []
        for track_id, (_, previous) in self._active.items():
            for index, detection in enumerate(detections):
                box = detection.box
                if previous.class_id != box.class_id:
                    continue
                overlap = box_iou(previous, box)
                distance = math.hypot(previous.x - box.x, previous.y - box.y)
                areas = previous.width * previous.height, box.width * box.height
                if overlap >= self.min_iou:
                    candidates.append((0, -overlap, distance, track_id, index))
                elif (
                    distance <= self.max_center_distance
                    and min(areas) > 0
                    and max(areas) / min(areas) <= self.max_area_ratio
                ):
                    candidates.append((1, distance, -overlap, track_id, index))
        used_tracks: set[int] = set()
        matches: dict[int, int] = {}
        for *_, track_id, index in sorted(candidates):
            if index not in matches and track_id not in used_tracks:
                matches[index] = track_id
                used_tracks.add(track_id)
        assigned = []
        for index, detection in enumerate(detections):
            track_id = matches.get(index)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
            self._active[track_id] = frame, detection.box
            assigned.append((track_id, detection))
        return assigned, expired


@dataclass
class EventCandidate:
    sequence: int
    first_frame: int
    first_timestamp: float | None
    last_frame: int
    last_timestamp: float | None
    last_box: Box
    observations: int = 0
    track_ids: set[int] = field(default_factory=set)
    best: tuple[Frame, Prediction] | None = None

    def observe(self, frame: Frame, track_id: int, prediction: Prediction) -> None:
        self.last_frame, self.last_timestamp = frame.index, frame.timestamp_s
        self.last_box = prediction.box
        self.observations += 1
        self.track_ids.add(track_id)
        if self.best is None or prediction.confidence > self.best[1].confidence:
            self.best = replace(frame, pixels=frame.pixels.copy()), prediction


class DuplicateSuppressor:
    """Resume a retired same-class event only with close time and strong local overlap."""

    def __init__(self, *, max_gap_frames: int, min_iou: float, minimum_observations: int) -> None:
        if max_gap_frames < 1 or not 0 < min_iou <= 1 or minimum_observations < 1:
            raise ValueError("Invalid duplicate suppression settings.")
        self.max_gap_frames = max_gap_frames
        self.min_iou = min_iou
        self.minimum_observations = minimum_observations
        self._pending: dict[int, EventCandidate] = {}
        self._next_sequence = 1

    def retire(self, event: EventCandidate) -> None:
        self._pending[event.sequence] = event

    def begin(self, frame: Frame, box: Box, *, max_span_frames: int = 0) -> EventCandidate:
        nearby = [
            (box_iou(box, event.last_box), event.last_frame, -event.sequence)
            for event in self._pending.values()
            if box.class_id == event.last_box.class_id
            and 0 < frame.index - event.last_frame <= self.max_gap_frames
            and (not max_span_frames or frame.index - event.first_frame < max_span_frames)
        ]
        overlap, _, negative_sequence = max(nearby, default=(0.0, -1, 0))
        if overlap >= self.min_iou:
            return self._pending.pop(-negative_sequence)
        sequence = self._next_sequence
        self._next_sequence += 1
        return EventCandidate(
            sequence, frame.index, frame.timestamp_s, frame.index, frame.timestamp_s, box
        )

    def ready(self, frame: int | None = None) -> list[EventCandidate]:
        ids = [
            sequence
            for sequence, event in self._pending.items()
            if frame is None or frame - event.last_frame > self.max_gap_frames
        ]
        return [self._pending.pop(sequence) for sequence in sorted(ids)]
