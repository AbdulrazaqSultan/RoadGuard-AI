import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from roadguard.data.dataset import Box
from roadguard.evaluation.box_matching import Prediction
from roadguard.vehicle.events import EventRecorder, NoLocation, SimulatedRoute, unknown_severity
from roadguard.vehicle.pipeline import run_pipeline
from roadguard.vehicle.software_fixture import FRAMES, ScriptedDetections
from roadguard.vehicle.source import Frame, OpenCVSource
from roadguard.vehicle.stage7 import checked_source
from roadguard.vehicle.tracking import TemporalAssociator


class Pixels:
    shape = (360, 640, 3)
    size = 360 * 640 * 3

    def copy(self):
        return Pixels()


class Capture:
    def __init__(self, *, open_ok=True, reads=1, expected=1):
        self.open_ok = open_ok
        self.reads = reads
        self.expected = expected
        self.released = False

    def isOpened(self):
        return self.open_ok

    def release(self):
        self.released = True

    def get(self, key):
        return {"fps": 15.0, "count": self.expected, "timestamp": 0.0}[key]

    def read(self):
        if self.reads:
            self.reads -= 1
            return True, Pixels()
        return False, None


def _use_capture(monkeypatch, capture):
    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(
            VideoCapture=lambda source: capture,
            CAP_PROP_FPS="fps",
            CAP_PROP_FRAME_COUNT="count",
            CAP_PROP_POS_MSEC="timestamp",
        ),
    )


def test_video_reports_dimensions_timestamps_and_eof(monkeypatch):
    capture = Capture(reads=2, expected=2)
    _use_capture(monkeypatch, capture)
    with OpenCVSource("independent.mp4") as source:
        first, second = source.read(), source.read()
        assert first.dimensions == (640, 360)
        assert first.timestamp_s == 0.0
        assert second.timestamp_s == pytest.approx(1 / 15)
        assert second.captured_at is None
        assert source.read() is None
    assert capture.released


@pytest.mark.parametrize(
    ("source_id", "opened", "reads", "match"),
    [
        (0, True, 0, "disconnected"),
        ("broken.mp4", True, 0, "decode stopped"),
        ("missing.mp4", False, 0, "could not open"),
    ],
)
def test_source_failures_release_capture(monkeypatch, source_id, opened, reads, match):
    capture = Capture(reads=reads, open_ok=opened)
    _use_capture(monkeypatch, capture)
    with pytest.raises(OSError, match=match), OpenCVSource(source_id) as source:
        source.read()
    assert capture.released


def test_source_paths_guard_frozen_inputs_and_invalid_options(tmp_path):
    with pytest.raises(ValueError, match="nonnegative"):
        OpenCVSource(-1)
    with pytest.raises(ValueError, match="exactly one"):
        checked_source(tmp_path, None, None)
    path = tmp_path / "data" / "processed" / "original.mp4"
    with pytest.raises(ValueError, match="protected"):
        checked_source(tmp_path, path, None)
    with pytest.raises(FileNotFoundError):
        checked_source(tmp_path, tmp_path / "missing.mp4", None)


def test_association_is_one_to_one_and_class_aware():
    tracker = TemporalAssociator(
        max_gap_frames=3,
        min_iou=0.25,
        max_center_distance=0.035,
        max_area_ratio=2.5,
    )
    original = Prediction(Box(3, 0.4, 0.5, 0.1, 0.1), 0.8)
    same = Prediction(Box(3, 0.41, 0.5, 0.1, 0.1), 0.7)
    another = Prediction(Box(3, 0.50, 0.5, 0.1, 0.1), 0.8)
    different = Prediction(Box(0, 0.4, 0.5, 0.1, 0.1), 0.9)
    initial, _ = tracker.update(0, [original])
    following, _ = tracker.update(1, [same, another, different])
    assert following[0][0] == initial[0][0]
    assert len({index for index, _ in following}) == 3
    assert tracker.update(5, [original])[1] == [1, 2, 3]


def test_location_modes_and_default_severity():
    assert NoLocation().locate(1.0, None).latitude is None
    point = SimulatedRoute(1.0, 2.0, 0.01).locate(2.0, "local time")
    assert point.simulated is True
    assert point.source == "simulated_route"
    assert point.longitude == pytest.approx(2.02)
    assert point.accuracy_m is None
    assert unknown_severity(Box(3, 0.5, 0.5, 0.2, 0.2)) == "Unknown"


def test_pipeline_merges_fragment_selects_evidence_and_serializes(monkeypatch, tmp_path):
    class Frames:
        fps, is_live = 15.0, False

        def __init__(self):
            self.items = iter(Frame(n, n / 15, None, Pixels()) for n in range(FRAMES))

        def read(self):
            return next(self.items, None)

    def write_image(path, pixels):
        Path(path).write_bytes(b"test-only evidence stub")
        return True

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(imwrite=write_image))
    settings = {
        "association": {
            "max_gap_frames": 3,
            "min_iou": 0.25,
            "max_center_distance": 0.035,
            "max_area_ratio": 2.5,
        },
        "duplicates": {"max_gap_frames": 8, "min_iou": 0.55, "minimum_observations": 3},
        "write_video": False,
        "output_fps_fallback": 15,
        "frame_limit": 0,
    }
    recorder = EventRecorder(
        tmp_path,
        "fixture",
        "fixture.mp4",
        SimulatedRoute(0, 0, 0.00002),
        unknown_severity,
        "scripted_fixture",
        None,
        crops=False,
    )
    result = run_pipeline(
        Frames(),
        ScriptedDetections(),
        recorder,
        settings,
        mode="throughput",
        confidence=0.25,
    )
    assert result["end_reason"] == "end_of_file"
    assert result["processed_frames"] == FRAMES
    assert result["raw_detections"] == 42
    assert result["final_events"] == 2
    assert result["merged_track_fragments"] == 1
    assert result["merged_detections"] == 39
    assert result["discarded_short_track_detections"] == 1
    records = sorted(
        map(json.loads, (tmp_path / "events.jsonl").read_text().splitlines()),
        key=lambda row: row["first_seen_frame"],
    )
    first, second = records
    assert first["track_ids"] == [1, 3]
    assert (first["first_seen_frame"], first["last_seen_frame"]) == (2, 23)
    assert first["representative_frame"] == 20
    assert second["representative_frame"] == 17
    assert first["detection_count"] == 18
    assert first["class_id"] == "D40"
    assert first["location"]["simulated"] is True
    assert first["severity"] == "Unknown"
    assert (tmp_path / first["evidence_frame_path"]).is_file()
    with (tmp_path / "events.csv").open(newline="") as stream:
        assert {row["event_id"] for row in csv.DictReader(stream)} == {
            row["event_id"] for row in records
        }
