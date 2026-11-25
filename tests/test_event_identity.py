import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from roadguard.data.dataset import Box
from roadguard.evaluation.box_matching import Prediction
from roadguard.vehicle.event_store import CaptureSession, EventStore, read_events
from roadguard.vehicle.events import EventRecorder, NoLocation, unknown_severity
from roadguard.vehicle.identity_review import analyze_identity, sample_observations
from roadguard.vehicle.pipeline import run_pipeline
from roadguard.vehicle.source import Frame
from roadguard.vehicle.tracking import DuplicateSuppressor


class Pixels:
    shape = (90, 160, 3)

    def copy(self):
        return self


def _images(monkeypatch):
    def save(path, pixels):
        Path(path).write_bytes(b"evidence fixture")
        return True

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(imwrite=save))


def _record(path: Path, source: str, monkeypatch) -> tuple[CaptureSession, list[dict]]:
    _images(monkeypatch)
    recorder = EventRecorder(
        path,
        "reused-readable-name",
        source,
        NoLocation(),
        unknown_severity,
        "YOLO26s",
        "best.pt",
        crops=False,
    )
    hit = Prediction(Box(3, 0.4, 0.5, 0.2, 0.3), 0.8)
    frame = Frame(0, 0.0, None, Pixels())
    event = DuplicateSuppressor(max_gap_frames=8, min_iou=0.55, minimum_observations=1).begin(
        frame, hit.box
    )
    event.observe(frame, 1, hit)
    with recorder:
        written = recorder.record(event)
    assert written.event_id in written.evidence_frame_path
    assert written.session_event_number == 1
    assert UUID(written.event_id).version == UUID(recorder.session_id).version == 4
    session = CaptureSession(
        recorder.session_id,
        source,
        "mobile_camera",
        "device-id",
        "start",
        "end",
        None,
        160,
        90,
        "best.pt",
        written.pipeline_version,
        path.as_posix(),
    )
    return session, read_events(path / "events.jsonl")


def test_archive_identity_and_sqlite_reopen(tmp_path, monkeypatch):
    first, events = _record(tmp_path / "first", "camera-a", monkeypatch)
    second, identical_geometry = _record(tmp_path / "second", "camera-a", monkeypatch)
    assert first.session_id != second.session_id
    assert events[0]["event_id"] != identical_geometry[0]["event_id"]
    database = tmp_path / "events.db"
    store = EventStore(database)
    assert store.save_session(first, events) == {"inserted": 1, "duplicates_blocked": 0}
    assert store.save_session(second, identical_geometry)["inserted"] == 1
    reopened = EventStore(database)
    assert reopened.get_event(events[0]["event_id"]) == events[0]
    assert reopened.get_event(str(UUID(int=0))) is None
    assert reopened.verify_session(first.session_id, events) == 1
    assert reopened.save_session(first, events) == {"inserted": 0, "duplicates_blocked": 1}
    with pytest.raises(ValueError, match="ID collision"):
        reopened.save_session(first, [events[0] | {"severity": "High"}])
    assert reopened.inspect()["total_unique_events"] == 2
    with closing(sqlite3.connect(database)) as connection:
        columns = connection.execute(
            "SELECT location_latitude,location_longitude,severity FROM events"
        ).fetchall()
        assert columns == [(None, None, "Unknown")] * 2
        names = [
            row[1] for row in connection.execute("PRAGMA table_info(events)") if row[1] != "id"
        ]
        projection = ",".join(names)
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE.*event_id"):
            connection.execute(f"INSERT INTO events ({projection}) SELECT {projection} FROM events")


def test_event_duration_boundary_cannot_rejoin_through_pending_tracks(tmp_path, monkeypatch):
    _images(monkeypatch)

    class Frames:
        fps, is_live = 2.0, False

        def __init__(self):
            self.stream = iter(Frame(index, index / 2, None, Pixels()) for index in range(8))

        def read(self):
            return next(self.stream, None)

    recorder = EventRecorder(
        tmp_path,
        "duration-test",
        "source",
        NoLocation(),
        unknown_severity,
        "YOLO26s",
        "best.pt",
        crops=False,
    )
    config = {
        "association": {
            "max_gap_frames": 3,
            "min_iou": 0.25,
            "max_center_distance": 0.035,
            "max_area_ratio": 2.5,
        },
        "duplicates": {"max_gap_frames": 8, "min_iou": 0.55, "minimum_observations": 1},
        "max_event_span_seconds": 1.5,
        "write_video": False,
        "output_fps_fallback": 2,
        "frame_limit": 0,
    }
    box = Box(3, 0.3, 0.4, 0.2, 0.2)
    report = run_pipeline(
        Frames(),
        lambda _: [Prediction(box, 0.8)],
        recorder,
        config,
        mode="throughput",
        confidence=0.25,
    )
    assert report["raw_detections"] == 8
    assert report["final_events"] == 3
    assert report["merged_track_fragments"] == 0
    assert [row["boundary_frame"] for row in report["event_span_boundaries"]] == [3, 6]
    rows = read_events(tmp_path / "events.jsonl")
    assert [(row["first_seen_frame"], row["last_seen_frame"]) for row in rows] == [
        (0, 2),
        (3, 5),
        (6, 7),
    ]


def test_identity_diagnostics_detect_suspicious_span_and_sample_observations():
    hit = {
        "track_id": 48,
        "class_id": "D20",
        "confidence": 0.8,
        "bbox_xywh_normalized": [0.5, 0.7, 0.25, 0.25],
    }
    frames = [{"frame_index": index, "detections": [hit]} for index in range(265)]
    policy = {"long_frames": 90, "center_drift": 0.25, "area_jump_ratio": 2.5, "gap_frames": 3}
    [track] = analyze_identity(frames, [], "baseline", policy)
    assert track["flags"] == ["long_span"]
    assert track["duration_frames"] == 265
    assert len(track["samples"]) == 9
    assert track["samples"][0]["frame"] == 0 and track["samples"][-1]["frame"] == 264
    observations = json.loads(json.dumps(track["samples"]))
    assert sample_observations(observations, 3) == observations[::4]
