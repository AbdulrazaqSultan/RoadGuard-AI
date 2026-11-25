"""Prepare and score a manually adjudicated road-defect identity sample."""

import csv
import json
import re
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path
from uuid import UUID

from roadguard.data.validation import sha256
from roadguard.evaluation.stage5_review import write_csv
from roadguard.training.experiment import reserve_run
from roadguard.utils.paths import write_json
from roadguard.utils.quality import run_quality_gates
from roadguard.vehicle.event_store import read_events
from roadguard.vehicle.identity_review import render_contact_sheets, sample_observations
from roadguard.vehicle.review import NOTEBOOK
from roadguard.vehicle.stage7 import checked_source, read_json, verify_frozen_state
from roadguard.vehicle.stage7_1 import ARCHIVES

EXPERIMENT = "stage7_2_event_ground_truth"
PRIORITIES = {
    "video1": {
        21: "area change",
        28: "gap",
        53: "neighbor",
        56: "neighbor",
        69: "adjacent",
        70: "adjacent",
        73: "normal",
    },
    "video2": {
        9: "gap",
        25: "fragment merge",
        31: "normal transverse",
        42: "gap",
        48: "duration rule",
        52: "duration rule",
        55: "fragment merge",
        60: "duration rule",
    },
}
STATUSES = {"CONFIRMED", "AMBIGUOUS", "NOT_REVIEWABLE"}
REVIEW_COLUMNS = (
    "review_status",
    "ground_truth_defect_ids",
    "physical_class",
    "spurious_confirmed",
    "reviewer",
    "notes",
)


def _archive(root: Path, run: Path) -> dict:
    if run.parent != root / "artifacts/experiments" or not run.name.startswith(f"{EXPERIMENT}-"):
        raise ValueError("Select a Stage 7.2 review archive.")
    return read_json(run / "metadata.json")


def _protected_hashes(root: Path, prior: Path) -> dict:
    files = [root / "data/runtime/roadguard_events.db"]
    for archive in ARCHIVES:
        folder = root / "artifacts/experiments" / archive / "throughput"
        files.extend(
            folder / name for name in ("events.jsonl", "events.csv", "frame_detections.jsonl")
        )
    for video in PRIORITIES:
        files.extend(
            prior / video / name
            for name in (
                "events.jsonl",
                "events.csv",
                "frame_detections.jsonl",
            )
        )
    return {path.relative_to(root).as_posix(): sha256(path) for path in files}


def _readonly_database(root: Path, prior: Path, cases: list[dict]) -> dict:
    database = root / "data/runtime/roadguard_events.db"
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as conn:
        rows = conn.execute(
            """SELECT event_id, payload_json, location_latitude, location_longitude,
               location_simulated, severity FROM events"""
        ).fetchall()
        sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        integrity = [row[0] for row in conn.execute("PRAGMA integrity_check")]
        foreign_keys = list(conn.execute("PRAGMA foreign_key_check"))
        objects = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    events = {row[0]: json.loads(row[1]) for row in rows}
    expected = {
        row["event_id"]: row
        for video in PRIORITIES
        for row in read_events(prior / video / "events.jsonl")
    }
    if (
        integrity != ["ok"]
        or foreign_keys
        or len(rows) != len(events)
        or any(UUID(event_id).version != 4 for event_id in events)
        or events != expected
        or any(row[2:] != (None, None, 0, "Unknown") for row in rows)
        or any(case["event_id"] not in events for case in cases)
    ):
        raise ValueError("Read-only database verification failed.")
    return {
        "database": database.as_posix(),
        "access": "read_only",
        "database_schema_changed": False,
        "review_table_present": "event_ground_truth_review" in objects,
        "integrity_check": "ok",
        "foreign_key_check": "ok",
        "total_events": len(rows),
        "event_ids_unique": True,
        "event_ids_uuid4": True,
        "total_sessions": sessions,
        "review_uuid_links_verified": len(cases),
        "gps_coordinates": None,
        "severity": "Unknown",
    }


def _frame_hits(path: Path, tracks: list[int]) -> list[dict]:
    return [
        detection | {"frame": frame["frame_index"]}
        for frame in read_events(path)
        for detection in frame["detections"]
        if detection["track_id"] in tracks
    ]


def prepare_review(root: Path) -> Path:
    experiments = root / "artifacts/experiments"
    state = read_json(experiments / "stage7_1_result.json")
    if state["status"] != "partial_pass" or not all(state["quality_gates"].values()):
        raise ValueError("Stage 7.1 integrity and quality checks must have passed.")
    prior = root / state["run"]
    frozen = verify_frozen_state(root, read_json(experiments / "stage6_result.json"))
    hashes = _protected_hashes(root, prior)
    archive = read_json(prior / "event_identity_analysis.json")
    _, run = reserve_run(experiments, EXPERIMENT, {"stage7_1": state["run"], "sample": PRIORITIES})
    cases = []
    media_paths = {}
    for index, source_name in enumerate(ARCHIVES, 1):
        video = f"video{index}"
        stage7 = experiments / source_name
        media = read_json(stage7 / "video_source.json")
        source = Path(checked_source(root, Path(media["project_copy"]), None))
        if sha256(source) != media["sha256"]:
            raise ValueError("Independent video does not match its archived hash.")
        media_paths[video] = {"path": str(source), "sha256": media["sha256"]}
        selected = []
        found = set()
        events = {event["event_id"]: event for event in read_events(prior / video / "events.jsonl")}
        for row in archive["tracks_and_events"]:
            if row["video"] != video or row["variant"] != "safeguard" or row["kind"] != "event":
                continue
            track = row["track_ids"][0]
            if track not in PRIORITIES[video]:
                continue
            found.add(track)
            reason = PRIORITIES[video][track]
            hits = _frame_hits(prior / video / "frame_detections.jsonl", row["track_ids"])
            if len(hits) != row["detections"]:
                raise ValueError("A review case does not match the archived observations.")
            samples = sample_observations(hits)
            event_id = row["event_id"]
            event = events[event_id]
            case = {
                "case_id": f"V{index}-T{track}",
                "video": video,
                "source_file": source.name,
                "event_id": event_id,
                "session_id": event["session_id"],
                "start_frame": row["start_frame"],
                "end_frame": row["end_frame"],
                "predicted_class": row["class_id"],
                "confidence_max": row["confidence_max"],
                "track_ids": row["track_ids"],
                "review_reason": reason,
                "representative_frame": row["representative_frame"],
            }
            selected.append(row | {"case": case, "flags": [reason], "samples": samples})
        if found != PRIORITIES[video].keys():
            raise ValueError(f"The review sample for {video} is incomplete.")
        if video == "video2":
            hits = _frame_hits(stage7 / "throughput/frame_detections.jsonl", [48])
            if len(hits) != 265:
                raise ValueError("The baseline T48 record differs from the reviewed sequence.")
            original = next(
                row
                for row in archive["tracks_and_events"]
                if row["variant"] == "baseline"
                and row["kind"] == "event"
                and row["video"] == video
                and row["track_ids"] == [48]
            )
            selected.append(
                original
                | {
                    "flags": ["T48 baseline, full-span samples"],
                    "samples": sample_observations(hits, 9),
                }
            )
            selected.append(
                original
                | {
                    "flags": ["T48 boundary frames at 254 and 344"],
                    "samples": [
                        hit for hit in hits if hit["frame"] in (252, 253, 254, 342, 343, 344)
                    ],
                }
            )
        render_contact_sheets(source, run / "figures/contact_sheets" / video, selected)
        for row in selected:
            sheet = Path(row["contact_sheet"]).relative_to(run).as_posix()
            if "case" in row:
                cases.append(row["case"] | {"contact_sheet": sheet})
        if video == "video2":
            review = {
                "baseline_event_id": selected[-2]["event_id"],
                "track_id": 48,
                "baseline_frames": [164, 428],
                "boundary_frames": [254, 344],
                "contact_sheet": Path(selected[-2]["contact_sheet"]).relative_to(run).as_posix(),
                "boundary_sheet": Path(selected[-1]["contact_sheet"]).relative_to(run).as_posix(),
                "affected_case_ids": ["V2-T48", "V2-T52", "V2-T60"],
            }
            write_json(run / "t48_review.json", review)
    write_json(run / "cases.json", cases)
    write_csv(
        run / "manual_review_template.csv",
        [case | dict.fromkeys(REVIEW_COLUMNS, "") for case in cases],
    )
    write_json(run / "database_consistency.json", _readonly_database(root, prior, cases))
    write_json(
        run / "metadata.json",
        {
            "status": "awaiting_review",
            "run": run.relative_to(root).as_posix(),
            "stage7_1_run": state["run"],
            "frozen_inputs": frozen,
            "protected_hashes": hashes,
            "independent_sources": media_paths,
            "prediction_count": 42,
            "sample_count": len(cases),
        },
    )
    return run


def score_adjudications(cases: list[dict], decisions: list[dict]) -> tuple[list[dict], dict]:
    """Score explicit manual correspondences. No visual labels are inferred from boxes."""
    by_case = {row["case_id"]: row for row in decisions}
    if by_case.keys() != {case["case_id"] for case in cases} or len(by_case) != len(decisions):
        raise ValueError("Each sampled event must have one explicit decision.")
    associations = defaultdict(set)
    gt_video = {}
    rows = []
    for case in cases:
        decision = by_case[case["case_id"]]
        status = decision["review_status"]
        ids = json.loads(decision["ground_truth_defect_ids"] or "[]")
        reject_as_spurious = decision["spurious_confirmed"].lower() == "true"
        if (
            status not in STATUSES
            or not decision["notes"].strip()
            or not decision["reviewer"].strip()
            or not isinstance(ids, list)
            or not all(isinstance(key, str) and re.fullmatch(r"GT\d{3,}", key) for key in ids)
            or len(ids) != len(set(ids))
            or (status != "CONFIRMED" and (ids or reject_as_spurious))
            or (status == "CONFIRMED" and bool(ids) == reject_as_spurious)
        ):
            raise ValueError(f"Invalid physical-defect decision for {case['case_id']}.")
        for key in ids:
            if gt_video.setdefault(key, case["video"]) != case["video"]:
                raise ValueError("A GT ID cannot span these independently reviewed clips.")
            associations[key].add(case["event_id"])
        rows.append(
            case
            | {col: decision[col] for col in REVIEW_COLUMNS}
            | {
                "ground_truth_defect_ids": ids,
            }
        )
    fragmented = {key for key, linked in associations.items() if len(linked) > 1}
    for row in rows:
        ids = row["ground_truth_defect_ids"]
        if row["review_status"] != "CONFIRMED":
            verdict = "AMBIGUOUS"
        elif not ids:
            verdict = "SPURIOUS"
        elif len(ids) > 1:
            verdict = "OVER_MERGED"
        elif ids[0] in fragmented:
            verdict = "FRAGMENTED"
        else:
            verdict = "CORRECT"
        row["error_type"] = verdict
    counts = Counter(row["error_type"] for row in rows)
    reviewed = sum(row["review_status"] == "CONFIRMED" for row in rows)
    return rows, {
        "reviewed_predicted_events": len(rows),
        "reviewed_confirmed_physical_defects": len(associations),
        "non_ambiguous_reviewed_events": reviewed,
        "correct_events": counts["CORRECT"],
        "over_merged_events": counts["OVER_MERGED"],
        "fragmented_predicted_events": counts["FRAGMENTED"],
        "fragmented_gt_defects": len(fragmented),
        "spurious_events": counts["SPURIOUS"],
        "ambiguous_events": sum(row["review_status"] == "AMBIGUOUS" for row in rows),
        "not_reviewable_events": sum(row["review_status"] == "NOT_REVIEWABLE" for row in rows),
        "event_identity_accuracy": counts["CORRECT"] / reviewed if reviewed else None,
        "over_merge_rate": counts["OVER_MERGED"] / reviewed if reviewed else None,
        "fragmentation_rate": len(fragmented) / len(associations) if associations else None,
        "scope": "purposefully selected inspection sample, reviewed correspondences only",
    }


def finish_review(root: Path, run: Path, adjudications: Path, t48_notes: Path) -> Path:
    metadata = _archive(root, run)
    cases = read_json(run / "cases.json")
    prior = root / metadata["stage7_1_run"]
    with adjudications.open(encoding="utf-8", newline="") as stream:
        decisions = list(csv.DictReader(stream))
    rows, metrics = score_adjudications(cases, decisions)
    write_csv(run / "event_identity_ground_truth.csv", rows)
    write_json(run / "event_identity_metrics.json", metrics)
    denominators = {
        "event_identity_accuracy": ("correct_events", "non_ambiguous_reviewed_events"),
        "over_merge_rate": ("over_merged_events", "non_ambiguous_reviewed_events"),
        "fragmentation_rate": ("fragmented_gt_defects", "reviewed_confirmed_physical_defects"),
    }
    metric_rows = {name: {"metric": name, "value": value} for name, value in metrics.items()}
    for rate, (numerator, denominator) in denominators.items():
        metric_rows[rate].update(numerator=metrics[numerator], denominator=metrics[denominator])
    write_csv(run / "event_identity_metrics.csv", list(metric_rows.values()))
    t48 = read_json(run / "t48_review.json") | read_json(t48_notes)
    affected = [row for row in rows if row["case_id"] in t48["affected_case_ids"]]
    if len(affected) != 3:
        raise ValueError("The two duration boundaries require review of all three affected events.")
    write_csv(
        run / "three_second_rule_review.csv",
        [row | {"safeguard_identity_effect": t48["safeguard_effect"]} for row in affected],
    )
    write_json(run / "t48_review.json", t48)
    after = verify_frozen_state(root, read_json(root / "artifacts/experiments/stage6_result.json"))
    if (
        _protected_hashes(root, prior) != metadata["protected_hashes"]
        or after != metadata["frozen_inputs"]
        or any(
            sha256(Path(source["path"])) != source["sha256"]
            for source in metadata["independent_sources"].values()
        )
    ):
        raise ValueError("A protected input changed during Stage 7.2.")
    write_json(
        run / "database_consistency.json",
        _readonly_database(root, prior, cases)
        | {
            "database_and_earlier_event_archives_unchanged": True,
        },
    )
    gates = run_quality_gates(root, run, NOTEBOOK, notebook_env={"ROADGUARD_STAGE72_RUN": str(run)})
    status = "partial_pass" if all(gates.values()) else "failed_quality_gates"
    metadata.update(status=status, quality_gates=gates, no_training=True, no_new_inference=True)
    write_json(run / "metadata.json", metadata)
    write_json(root / "artifacts/experiments/stage7_2_result.json", metadata)
    if status == "failed_quality_gates":
        raise RuntimeError(f"Stage 7.2 quality gates failed: {gates}")
    return run
