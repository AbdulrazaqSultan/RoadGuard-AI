"""Replay archived predictions to validate conservative event identity and persistence."""

import logging
from dataclasses import asdict
from pathlib import Path

from roadguard.data.dataset import RDD_CLASSES, Box
from roadguard.data.validation import sha256
from roadguard.evaluation.box_matching import Prediction
from roadguard.evaluation.stage5_review import write_csv
from roadguard.training.experiment import reserve_run
from roadguard.utils.paths import write_json
from roadguard.utils.quality import run_quality_gates
from roadguard.vehicle.event_store import CaptureSession, EventStore, read_events
from roadguard.vehicle.identity_review import analyze_identity, render_contact_sheets
from roadguard.vehicle.review import NOTEBOOK, verify_outputs
from roadguard.vehicle.stage7 import (
    checked_source,
    measure_pipeline,
    read_json,
    verify_frozen_state,
)

ARCHIVES = (
    "stage7_user_video_validation-006b339eb3a9-001",
    "stage7_user_video_validation-75e52feb1ebd-001",
)
EXPERIMENT = "stage7_1_event_identity"
LOGGER = logging.getLogger(__name__)


def _without_track_ids(rows: list[dict]) -> list[dict]:
    return [
        row
        | {
            "detections": [
                {key: value for key, value in hit.items() if key != "track_id"}
                for hit in row["detections"]
            ]
        }
        for row in rows
    ]


def _flags(rows: list[dict]) -> dict:
    return {
        "suspicious_tracks": sum(row["kind"] == "track" and bool(row["flags"]) for row in rows),
        "long_tracks": sum(row["kind"] == "track" and "long_span" in row["flags"] for row in rows),
        "suspicious_events": sum(row["kind"] == "event" and bool(row["flags"]) for row in rows),
    }


def _policy(fps: float, settings: dict) -> dict:
    import math

    return {
        "long_frames": math.ceil(fps * settings["max_event_span_seconds"]),
        "center_drift": 0.25,
        "area_jump_ratio": settings["association"]["max_area_ratio"],
        "gap_frames": settings["association"]["max_gap_frames"],
    }


def replay_and_persist(root: Path, config: dict) -> Path:
    """Use archived predictions and the shared acquisition, association and recording path."""
    experiments = root / "artifacts/experiments"
    stage6 = read_json(experiments / "stage6_result.json")
    before = verify_frozen_state(root, stage6)
    settings = config["vehicle"]
    database = root / settings["event_database"]["path"]
    identity = {
        "baselines": list(ARCHIVES),
        "vehicle": settings,
        "checkpoint": stage6["checkpoint"],
    }
    _, run = reserve_run(experiments, EXPERIMENT, identity)
    output_config = config | {"vehicle": settings | {"write_video": False}}
    write_json(run / "pipeline_config.json", output_config)
    data = []
    baseline_diagnostics = []
    for index, name in enumerate(ARCHIVES, 1):
        archive = experiments / name
        original = read_json(archive / "stage7_result.json")
        media = read_json(archive / "video_source.json")
        source = checked_source(root, Path(media["project_copy"]), None)
        if not isinstance(source, str) or sha256(Path(source)) != media["sha256"]:
            raise ValueError("Field-video input checksum changed.")
        old = archive / "throughput"
        predictions = old / "frame_detections.jsonl"
        rows = read_events(predictions)
        events = read_events(old / "events.jsonl")
        signature = {
            p.name: sha256(p) for p in (predictions, old / "events.jsonl", old / "events.csv")
        }
        metrics = original["runtime"]["throughput"]
        if len(rows) != media["total_frames"] or len(events) != metrics["final_events"]:
            raise ValueError("Baseline archive is incomplete.")
        policy = _policy(media["fps"], settings)
        reviewed = analyze_identity(rows, events, "baseline", policy)
        for row in reviewed:
            if index == 2 and row["track_ids"] == [48]:
                row["flags"].append("changing_road_details_on_visual_review")
        render_contact_sheets(
            Path(source), run / "figures/contact_sheets" / f"video{index}", reviewed
        )
        baseline_diagnostics.extend(row | {"video": f"video{index}"} for row in reviewed)
        data.append((source, rows, metrics, signature, media, policy))
    write_json(run / "baseline_identity.json", baseline_diagnostics)
    LOGGER.info("Archived the baseline identity review before applying the duration safeguard")

    comparisons = []
    revised_diagnostics = []
    consistency = []
    export = []
    store = EventStore(database)
    for index, (source, rows, baseline, signature, media, policy) in enumerate(data, 1):
        folder = run / f"video{index}"
        archived_rows = iter(rows)

        def infer(_pixels, stream=archived_rows):
            return [
                Prediction(
                    Box(RDD_CLASSES[row["class_id"]], *row["bbox_xywh_normalized"]),
                    row["confidence"],
                )
                for row in next(stream)["detections"]
            ]

        runtime = measure_pipeline(
            folder,
            source,
            "throughput",
            "YOLO26s",
            output_config,
            infer,
            location_mode="none",
            checkpoint=stage6["checkpoint"],
            gpu=False,
        )
        regenerated = read_events(folder / "frame_detections.jsonl")
        if _without_track_ids(regenerated) != _without_track_ids(rows):
            raise ValueError("Replay changed the archived detection data or frame metadata.")
        events = verify_outputs(folder, runtime, output_config["vehicle"])
        session = CaptureSession(**read_json(folder / "session.json"))
        if any(
            event["severity"] != "Unknown"
            or event["location"]["simulated"]
            or event["location"]["latitude"] is not None
            or event["location"]["longitude"] is not None
            for event in events
        ):
            raise ValueError("Unexpected location or severity data.")
        first = store.save_session(session, events)
        reopened = EventStore(database)
        retry = reopened.save_session(session, events)
        stored = reopened.verify_session(session.session_id, events)
        if retry != {"inserted": 0, "duplicates_blocked": stored}:
            raise ValueError("The idempotent retry did not preserve the event count.")
        consistency.append(
            {
                "video": f"video{index}",
                "session_id": session.session_id,
                "jsonl_rows": len(events),
                "csv_rows": len(events),
                "sqlite_rows": stored,
                "first_write": first,
                "retry": retry,
                "evidence_verified": True,
                "severity": "Unknown",
                "location": "disabled",
            }
        )
        reviewed = analyze_identity(regenerated, events, "safeguard", policy)
        render_contact_sheets(
            Path(source), run / "figures/contact_sheets" / f"video{index}", reviewed
        )
        revised_diagnostics.extend(row | {"video": f"video{index}"} for row in reviewed)
        old = experiments / ARCHIVES[index - 1] / "throughput"
        if signature != {name: sha256(old / name) for name in signature} or (
            sha256(Path(source)) != media["sha256"]
        ):
            raise ValueError("The source video or Stage 7 archives changed during Stage 7.1.")
        comparisons.append(
            {
                "video": f"video{index}",
                "filename": media["original_filename"],
                "policy": policy,
                "source_sha256": media["sha256"],
                "baseline_events": baseline["final_events"],
                "replay_events": len(events),
                "raw_detections": runtime["raw_detections"],
                "frames": runtime["processed_frames"],
                "boundary_count": len(runtime["event_span_boundaries"]),
                "boundaries": runtime["event_span_boundaries"],
                "baseline_flags": _flags(
                    [row for row in baseline_diagnostics if row["video"] == f"video{index}"]
                ),
                "replay_flags": _flags(reviewed),
                "predictions_unchanged": True,
            }
        )
        for event in events:
            keys = (
                "event_id",
                "session_id",
                "class_id",
                "class_name",
                "confidence",
                "first_seen_frame",
                "last_seen_frame",
                "detection_count",
                "severity",
            )
            export.append(
                {key: event[key] for key in keys}
                | {"evidence_frame_path": (folder / event["evidence_frame_path"]).as_posix()}
            )
    all_rows = baseline_diagnostics + revised_diagnostics
    write_csv(
        run / "suspicious_tracks.csv",
        [
            {key: value for key, value in row.items() if key != "samples"}
            for row in all_rows
            if row["flags"]
        ],
    )
    write_csv(run / "stage7_1_database_summary.csv", export)
    database_report = store.inspect() | {
        "database": str(database),
        "sessions": consistency,
        "batch_unique_events": sum(row["sqlite_rows"] for row in consistency),
        "duplicate_attempts_blocked": sum(
            row["retry"]["duplicates_blocked"] for row in consistency
        ),
        "jsonl_csv_sqlite_consistent": True,
    }
    write_json(run / "database_consistency.json", database_report)
    after = verify_frozen_state(root, stage6)
    if before != after:
        raise ValueError("Frozen artifact integrity changed during the replay.")
    analysis = {
        "inputs": comparisons,
        "tracks_and_events": all_rows,
        "physical_over_merges_confirmed": None,
        "visual_finding": "Baseline T48 crosses changing road detail despite stable image overlap.",
        "interpretation": "An identity-risk case, not a count of confirmed physical defects.",
        "detector_inference_rerun": False,
    }
    write_json(run / "event_identity_analysis.json", analysis)
    write_json(run / "integrity.json", after)
    write_json(
        run / "metadata.json",
        {
            "status": "measured",
            "run": run.relative_to(root).as_posix(),
            "identity": identity,
            "sessions": [
                asdict(CaptureSession(**read_json(run / row["video"] / "session.json")))
                for row in consistency
            ],
        },
    )
    return run


def finalize_stage7_1(root: Path, run: Path) -> Path:
    experiments = root / "artifacts/experiments"
    if run.parent != experiments or not run.name.startswith(f"{EXPERIMENT}-"):
        raise ValueError("Select a recorded Stage 7.1 archive.")
    metadata_path = run / "metadata.json"
    metadata = read_json(metadata_path)
    consistency_path = run / "database_consistency.json"
    consistency = read_json(consistency_path)
    database = Path(consistency["database"])
    if not database.is_file():
        raise FileNotFoundError(database)
    reopened = EventStore(database)
    reopened.inspect()
    for session in consistency["sessions"]:
        events = read_events(run / session["video"] / "events.jsonl")
        reopened.verify_session(session["session_id"], events)
        if any(reopened.get_event(event["event_id"]) != event for event in events):
            raise ValueError(
                "An archived event could not be retrieved after restarting the process."
            )
    consistency["reopened_in_separate_process"] = True
    write_json(consistency_path, consistency)
    gates = run_quality_gates(root, run, NOTEBOOK, notebook_env={"ROADGUARD_STAGE71_RUN": str(run)})
    status = "partial_pass" if all(gates.values()) else "failed_quality_gates"
    metadata.update(status=status, quality_gates=gates)
    write_json(metadata_path, metadata)
    write_json(experiments / "stage7_1_result.json", metadata)
    if status == "failed_quality_gates":
        raise RuntimeError(f"Stage 7.1 quality gates failed: {gates}")
    return metadata_path
