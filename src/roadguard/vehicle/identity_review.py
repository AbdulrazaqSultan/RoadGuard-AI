"""Diagnose event continuity and render its supporting frames."""

import math
from collections import defaultdict
from itertools import pairwise
from pathlib import Path


def sample_observations(hits: list[dict], limit: int = 5) -> list[dict]:
    """Take evenly spaced observations, including both ends, without inventing boxes."""
    if limit < 2:
        raise ValueError("Use at least two contact-sheet samples.")
    count = min(limit, len(hits))
    if count < 2:
        return hits[:]
    return [hits[round(i * (len(hits) - 1) / (count - 1))] for i in range(count)]


def _diagnostic(hits: list[dict], label: str, group: str, policy: dict, event: dict | None) -> dict:
    boxes = [row["bbox_xywh_normalized"] for row in hits]
    frames = [row["frame"] for row in hits]
    areas = [box[2] * box[3] for box in boxes]
    gaps = [after - before - 1 for before, after in pairwise(frames)]
    jump = max((max(a, b) / min(a, b) for a, b in pairwise(areas)), default=1.0)
    drift = math.dist(boxes[0][:2], boxes[-1][:2])
    duration = frames[-1] - frames[0] + 1
    flags = []
    if duration > policy["long_frames"]:
        flags.append("long_span")
    if drift > policy["center_drift"]:
        flags.append("large_center_drift")
    if jump > policy["area_jump_ratio"]:
        flags.append("abrupt_area_change")
    if max(gaps, default=0) > policy["gap_frames"]:
        flags.append("long_gap")
    scores = [row["confidence"] for row in hits]
    best = max(hits, key=lambda row: row["confidence"])
    samples = sample_observations(hits, 9 if "long_span" in flags else 5)
    return {
        "variant": label,
        "kind": group,
        "event_id": event["event_id"] if event else None,
        "track_ids": sorted({row["track_id"] for row in hits}),
        "class_id": hits[0]["class_id"],
        "start_frame": frames[0],
        "end_frame": frames[-1],
        "duration_frames": duration,
        "detections": len(hits),
        "gaps": [gap for gap in gaps if gap],
        "max_gap_frames": max(gaps, default=0),
        "first_bbox_xywh": boxes[0],
        "last_bbox_xywh": boxes[-1],
        "center_drift_normalized": drift,
        "last_to_first_area_ratio": areas[-1] / areas[0],
        "max_adjacent_area_ratio": jump,
        "representative_frame": (
            event["representative_frame"] if group == "event" else best["frame"]
        ),
        "confidence_min": min(scores),
        "confidence_max": max(scores),
        "flags": flags,
        "samples": samples,
    }


def analyze_identity(
    frames: list[dict], events: list[dict], label: str, policy: dict
) -> list[dict]:
    """Include transient tracks, promoted tracks, and finalized events spanning fragments."""
    tracks: dict[int, list[dict]] = defaultdict(list)
    event_by_track = {}
    for event in events:
        for track in event["track_ids"]:
            if track in event_by_track:
                raise ValueError(f"Track T{track} belongs to multiple events.")
            event_by_track[track] = event
    for frame in frames:
        for row in frame["detections"]:
            tracks[row["track_id"]].append(row | {"frame": frame["frame_index"]})
    diagnostics = [
        _diagnostic(hits, label, "track", policy, event_by_track.get(track))
        for track, hits in sorted(tracks.items())
    ]
    for event in events:
        hits = sorted(
            (row for track in event["track_ids"] for row in tracks[track]),
            key=lambda row: row["frame"],
        )
        if len(hits) != event["detection_count"] or any(
            row["class_id"] != event["class_id"] for row in hits
        ):
            raise ValueError(f"Observations do not reconcile for {event['event_id']}.")
        diagnostics.append(_diagnostic(hits, label, "event", policy, event))
    return diagnostics


def render_contact_sheets(source: Path, output: Path, diagnostics: list[dict]) -> None:
    """Decode selected frames once and make contact sheets for flagged tracks and events."""
    import cv2
    import numpy as np

    from roadguard.data.dataset import Box
    from roadguard.vehicle.events import pixel_bounds
    from roadguard.vehicle.source import OpenCVSource

    flagged = [row for row in diagnostics if row["flags"]]
    if not flagged:
        return
    output.mkdir(parents=True, exist_ok=True)
    requests = {item["frame"] for row in flagged for item in row["samples"]}
    frames = {}
    with OpenCVSource(str(source)) as video:
        while requests - frames.keys():
            frame = video.read()
            if frame is None:
                raise OSError("A selected frame was not found in the video.")
            if frame.index in requests:
                frames[frame.index] = cv2.resize(
                    frame.pixels, (600, 338), interpolation=cv2.INTER_AREA
                )
    for index, row in enumerate(flagged, 1):
        key = row["event_id"] or "unconfirmed track"
        samples = row["samples"]
        tiles = []
        for sample in samples:
            pixels = frames[sample["frame"]].copy()
            x1, y1, x2, y2 = pixel_bounds(Box(0, *sample["bbox_xywh_normalized"]), 600, 338)
            cv2.rectangle(pixels, (x1, y1), (x2 - 1, y2 - 1), (103, 241, 240), 2)
            tile = np.full((374, 600, 3), 24, dtype=np.uint8)
            tile[36:] = pixels
            title = (
                f"Frame {sample['frame']}   T{sample['track_id']}   {sample['class_id']}   "
                f"confidence {sample['confidence']:.2f}"
            )
            cv2.putText(tile, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (248, 248, 248), 1)
            tiles.append(tile)
        while len(tiles) % 3:
            tiles.append(np.full((374, 600, 3), 24, dtype=np.uint8))
        header = np.full((65, 1800, 3), 24, dtype=np.uint8)
        title = f"{row['variant']} {row['kind']} | {key}"
        cv2.putText(header, title, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (248, 248, 248), 1)
        cv2.putText(
            header,
            ", ".join(row["flags"]),
            (12, 53),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (103, 241, 240),
            1,
        )
        sheet = np.vstack([header, *(np.hstack(tiles[i : i + 3]) for i in range(0, len(tiles), 3))])
        file = output / f"{row['variant']}-{row['kind']}-{index:02d}.jpg"
        if not cv2.imwrite(str(file), sheet):
            raise OSError(f"Could not save {file}.")
        row["contact_sheet"] = file.as_posix()
