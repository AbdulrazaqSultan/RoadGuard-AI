"""Process frames sequentially with a shared detector, tracking and recording path."""

import json
import math
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

from roadguard.evaluation.box_matching import Prediction
from roadguard.vehicle.events import CLASS_CODES, EventRecorder, pixel_bounds
from roadguard.vehicle.source import FrameSource
from roadguard.vehicle.tracking import DuplicateSuppressor, EventCandidate, TemporalAssociator


class AnnotatedVideo:
    """Write a headless annotated MP4 with explicit track labels."""

    def __init__(self, path: Path, fps: float) -> None:
        self.path, self.fps = path, fps
        self._writer = None

    def write(self, pixels: Any, assignments: list[tuple[int, Prediction]]) -> None:
        import cv2

        height, width = pixels.shape[:2]
        if self._writer is None:
            self._shape = width, height
            codec = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(self.path), codec, self.fps, self._shape)
            if not self._writer.isOpened():
                raise OSError(f"Could not open annotated video writer: {self.path}")
        if (width, height) != self._shape:
            raise ValueError("Frame dimensions changed during annotated video output.")
        drawn = pixels.copy()
        for track_id, prediction in assignments:
            box = prediction.box
            x1, y1, x2, y2 = pixel_bounds(box, width, height)
            color = ((255, 208, 62), (85, 193, 255), (87, 223, 124), (108, 125, 255))[box.class_id]
            cv2.rectangle(drawn, (x1, y1), (x2 - 1, y2 - 1), color, 2)
            label = f"{CLASS_CODES[box.class_id]} {prediction.confidence:.2f} T{track_id}"
            origin = x1, max(13, y1 - 5)
            cv2.putText(drawn, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (12, 12, 12), 3)
            cv2.putText(drawn, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        self._writer.write(drawn)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def _filter_predictions(detections: list[Prediction], confidence: float) -> list[Prediction]:
    accepted = []
    for prediction in detections:
        box = prediction.box
        if (
            box.class_id not in CLASS_CODES
            or not 0 <= prediction.confidence <= 1
            or not all(math.isfinite(value) and 0 <= value <= 1 for value in box.coordinates)
        ):
            raise ValueError("Detector returned an invalid road-defect prediction.")
        if prediction.confidence >= confidence and box.width > 0 and box.height > 0:
            accepted.append(prediction)
    return accepted


def run_pipeline(
    source: FrameSource,
    infer: Callable[[Any], list[Prediction]],
    recorder: EventRecorder,
    settings: dict,
    *,
    mode: str,
    confidence: float,
) -> dict:
    """Run a sequential video/camera loop and measure all processing and pacing."""
    if mode not in {"throughput", "realtime"}:
        raise ValueError("Choose throughput or realtime.")
    if mode == "realtime" and not source.is_live and not source.fps:
        raise ValueError("Real-time video simulation requires a known source FPS.")
    tracker = TemporalAssociator(**settings["association"])
    duplicates = DuplicateSuppressor(**settings["duplicates"])
    video = AnnotatedVideo(
        recorder.output / "annotated_video.mp4", source.fps or settings["output_fps_fallback"]
    )
    active: dict[int, EventCandidate] = {}
    frames = raw_count = event_count = discarded = merged = merged_fragments = late_frames = 0
    detector_seconds = frame_seconds = pacing_seconds = 0.0
    boundaries = []
    span_seconds = settings.get("max_event_span_seconds", 0.0)
    if not math.isfinite(span_seconds) or span_seconds < 0:
        raise ValueError("Maximum event span must be a finite, nonnegative number of seconds.")
    max_span = (
        max(1, math.ceil(span_seconds * (source.fps or settings["output_fps_fallback"])))
        if span_seconds
        else 0
    )
    first_timestamp = None
    end_reason = "frame_limit"
    planned = settings["frame_limit"]
    started = perf_counter()

    def finalize(candidates: list[EventCandidate]) -> None:
        nonlocal event_count, discarded, merged, merged_fragments
        for candidate in candidates:
            if candidate.observations >= duplicates.minimum_observations:
                recorder.record(candidate)
                event_count += 1
                merged += candidate.observations - 1
                merged_fragments += len(candidate.track_ids) - 1
            else:
                discarded += candidate.observations

    with ExitStack() as resources:
        resources.enter_context(recorder)
        resources.callback(video.close)
        records = resources.enter_context(
            (recorder.output / "frame_detections.jsonl").open("w", encoding="utf-8")
        )
        while not planned or frames < planned:
            frame_start = perf_counter()
            frame = source.read()
            if frame is None:
                end_reason = "end_of_file"
                break
            delay = 0.0
            if mode == "realtime" and not source.is_live:
                timestamp = frame.timestamp_s
                if timestamp is None:
                    timestamp = frame.index / source.fps
                if first_timestamp is None:
                    first_timestamp = timestamp
                    pacing_start = perf_counter()
                due = pacing_start + timestamp - first_timestamp
                before = perf_counter()
                if due > before:
                    sleep(due - before)
                    delay = perf_counter() - before
                    pacing_seconds += delay
                if perf_counter() - due > 1 / source.fps:
                    late_frames += 1
            inferred = perf_counter()
            predictions = _filter_predictions(infer(frame.pixels), confidence)
            detector_seconds += perf_counter() - inferred
            if max_span:
                for track_id, event in tuple(active.items()):
                    if frame.index - event.first_frame >= max_span:
                        tracker.close(track_id)
                        boundaries.append({"track_id": track_id, "boundary_frame": frame.index})
                        finalize([active.pop(track_id)])
            assigned, expired = tracker.update(frame.index, predictions)
            for track_id in expired:
                duplicates.retire(active.pop(track_id))
            for track_id, prediction in assigned:
                if track_id not in active:
                    active[track_id] = duplicates.begin(
                        frame, prediction.box, max_span_frames=max_span
                    )
                active[track_id].observe(frame, track_id, prediction)
            finalize(duplicates.ready(frame.index))
            if settings["write_video"]:
                video.write(frame.pixels, assigned)
            records.write(
                json.dumps(
                    {
                        "frame_index": frame.index,
                        "source_timestamp_s": frame.timestamp_s,
                        "captured_at": frame.captured_at,
                        "dimensions": frame.dimensions,
                        "detections": [
                            {
                                "track_id": track_id,
                                "class_id": CLASS_CODES[det.box.class_id],
                                "confidence": det.confidence,
                                "bbox_xywh_normalized": det.box.coordinates,
                            }
                            for track_id, det in assigned
                        ],
                    },
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            if not frames:
                width, height = frame.dimensions
            frames += 1
            raw_count += len(predictions)
            frame_seconds += perf_counter() - frame_start - delay
        for candidate in active.values():
            duplicates.retire(candidate)
        finalize(duplicates.ready())
        if not frames:
            raise ValueError("The source supplied no decodable frames.")
        if raw_count != event_count + merged + discarded:
            raise RuntimeError("Event observation counts do not reconcile.")
    elapsed = perf_counter() - started
    return {
        "mode": mode,
        "source_fps": source.fps,
        "processed_frames": frames,
        "frame_width": width,
        "frame_height": height,
        "max_event_span_frames": max_span,
        "event_span_boundaries": boundaries,
        "software_skipped_frames": 0,
        "capture_drops_observable": False,
        "realtime_late_frames": late_frames if mode == "realtime" and not source.is_live else None,
        "mean_detector_ms": detector_seconds * 1000 / frames,
        "mean_total_frame_ms_excluding_intentional_wait": frame_seconds * 1000 / frames,
        "effective_fps": frames / elapsed,
        "wall_seconds": elapsed,
        "intentional_pacing_seconds": pacing_seconds,
        "raw_detections": raw_count,
        "final_events": event_count,
        "merged_detections": merged,
        "merged_track_fragments": merged_fragments,
        "discarded_short_track_detections": discarded,
        "duplicate_reduction_ratio": merged / raw_count if raw_count else None,
        "raw_to_event_reduction_ratio": (raw_count - event_count) / raw_count
        if raw_count
        else None,
        "end_reason": end_reason,
    }
