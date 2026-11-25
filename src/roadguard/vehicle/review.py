"""Verify the Stage 7 archive and publish the software and field-validation status."""

import csv
import json
from dataclasses import fields
from pathlib import Path

from roadguard.data.validation import sha256
from roadguard.utils.paths import write_json, write_text
from roadguard.utils.quality import run_quality_gates
from roadguard.vehicle.events import RoadDefectEvent
from roadguard.vehicle.software_fixture import FRAMES
from roadguard.vehicle.stage7 import EXPERIMENT, read_json, verify_frozen_state

NOTEBOOK = Path("notebooks/07_real_time_vehicle_pipeline_review.ipynb")
DOCUMENT = Path("docs/stage7_real_time_vehicle_pipeline_ar.md")


def verify_outputs(directory: Path, metrics: dict, settings: dict) -> list[dict]:
    """Parse both event formats, reconcile all frames, and decode the output video."""
    import cv2

    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    with (directory / "events.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        names = [field.name for field in fields(RoadDefectEvent)]
        legacy = [name for name in names if name not in {"session_id", "session_event_number"}]
        if reader.fieldnames not in (names, legacy):
            raise ValueError(f"Unexpected event CSV schema: {directory}.")
    if len(events) != len(rows) or len(events) != metrics["final_events"]:
        raise ValueError(f"JSONL/CSV event totals do not match: {directory}.")
    if len({event["event_id"] for event in events}) != len(events):
        raise ValueError("Duplicate event IDs in the finished archive.")
    decoded = 0
    raw = 0
    with (directory / "frame_detections.jsonl").open(encoding="utf-8") as stream:
        for decoded, line in enumerate(stream, 1):
            frame = json.loads(line)
            if frame["frame_index"] != decoded - 1:
                raise ValueError("Missing or reordered frame detection record.")
            raw += len(frame["detections"])
    if raw != metrics["raw_detections"] or decoded != metrics["processed_frames"]:
        raise ValueError("Raw detection statistics do not match the saved capture.")
    for event, row in zip(events, rows, strict=True):
        for key, value in event.items():
            csv_value = row[key]
            if isinstance(value, dict | list):
                agrees = json.loads(csv_value) == value
            else:
                agrees = csv_value == ("" if value is None else str(value))
            if not agrees:
                raise ValueError(f"JSONL and CSV disagree on {key}.")
        if (
            not event["first_seen_frame"]
            <= event["representative_frame"]
            <= event["last_seen_frame"]
        ):
            raise ValueError("Evidence frame falls outside the event span.")
        for key in ("evidence_frame_path", "crop_path"):
            relative = event[key]
            if relative is None:
                continue
            path = (directory / relative).resolve()
            if not path.is_relative_to(directory) or cv2.imread(str(path)) is None:
                raise ValueError(f"Evidence could not be decoded: {relative}.")
    if settings["write_video"]:
        from roadguard.vehicle.source import OpenCVSource

        video_frames = 0
        with OpenCVSource(str(directory / "annotated_video.mp4")) as source:
            while source.read() is not None:
                video_frames += 1
        if video_frames != metrics["processed_frames"]:
            raise ValueError("Annotated video doesn't contain all processed frames.")
    return events


def finalize_stage7(root: Path, run: Path) -> Path:
    """Verify frozen inputs and generated artifacts, execute notebook and write the report."""
    from roadguard.vehicle.stage7_document import arabic_report

    experiments = root / "artifacts/experiments"
    if run.parent != experiments or not run.name.startswith(EXPERIMENT + "-"):
        raise ValueError("Select a Stage 7 run under artifacts/experiments.")
    metadata_file = run / "metadata.json"
    metadata = read_json(metadata_file)
    if metadata["status"] not in {"measured", "partial_pass", "failed_quality_gates"}:
        raise ValueError("Complete the Stage 7 execution first.")
    stage6 = read_json(experiments / "stage6_result.json")
    integrity = verify_frozen_state(root, stage6)
    if integrity != read_json(run / "preflight.json"):
        raise ValueError("Frozen inputs changed during Stage 7.")
    write_json(run / "integrity.json", integrity)
    metrics = read_json(run / "runtime_metrics.json")
    settings = metadata["identity"]["config"]["vehicle"]
    events = {key: verify_outputs(run / key, row, settings) for key, row in metrics.items()}
    synthetic = metadata["identity"]["source_kind"] == "synthetic_software"
    if synthetic:
        fixture = metrics["event_fixture"]
        if (
            any(row["processed_frames"] != FRAMES for row in metrics.values())
            or any(row["end_reason"] != "end_of_file" for row in metrics.values())
            or sha256(run / "synthetic_input.mp4") != metadata["synthetic_input_sha256"]
            or fixture["raw_detections"] != 42
            or fixture["final_events"] != 2
            or fixture["merged_detections"] != 39
            or fixture["merged_track_fragments"] != 1
            or fixture["discarded_short_track_detections"] != 1
            or any(not event["location"]["simulated"] for event in events["event_fixture"])
        ):
            raise ValueError("The Stage 7 software fixture did not produce its expected results.")
    report = run / "stage7_result.json"
    result = {
        "status": "quality_gates_pending",
        "run": metadata["run"],
        "model": metadata["model"],
        "checkpoint": metadata["checkpoint"],
        "source_kind": metadata["identity"]["source_kind"],
        "source_provenance": metadata["source_provenance"],
        "field_validation_status": "pending",
        "runtime": metrics,
        "integrity": integrity,
        "output_verified": True,
    }
    write_json(report, result)
    gates = run_quality_gates(root, run, NOTEBOOK, notebook_env={"ROADGUARD_STAGE7_RUN": str(run)})
    status = "partial_pass" if all(gates.values()) else "failed_quality_gates"
    metadata.update(status=status, quality_gates=gates)
    result.update(status=status, quality_gates=gates)
    write_text(root / DOCUMENT, arabic_report(result, metadata, stage6))
    write_json(metadata_file, metadata)
    write_json(report, result)
    write_json(experiments / "stage7_result.json", result)
    if status == "failed_quality_gates":
        raise RuntimeError(f"Stage 7 quality gates failed: {gates}")
    return report
