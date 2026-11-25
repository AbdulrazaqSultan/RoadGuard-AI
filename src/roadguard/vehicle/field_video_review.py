"""Verify and summarize a saved independent-video run without rerunning detection."""

import json
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path
from statistics import median

from roadguard.data.dataset import Box
from roadguard.data.validation import sha256
from roadguard.evaluation.box_matching import box_iou
from roadguard.utils.paths import write_json
from roadguard.utils.quality import run_quality_gates
from roadguard.vehicle.events import CLASS_CODES
from roadguard.vehicle.review import NOTEBOOK, verify_outputs
from roadguard.vehicle.stage7 import read_json, verify_frozen_state

PREFIX = "stage7_user_video_validation-"


def summarize_associations(directory: Path, events: list[dict]) -> dict:
    """Count observations, distinct temporal groups and their adjacent-frame overlap."""
    counts = Counter()
    tracks = defaultdict(list)
    frames_with_detections = 0
    with (directory / "frame_detections.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            frames_with_detections += bool(row["detections"])
            for detection in row["detections"]:
                counts[detection["class_id"]] += 1
                tracks[detection["track_id"]].append((row["frame_index"], detection))
    adjacent = []
    groups = []
    for track_id, sequence in sorted(tracks.items()):
        hits = [detection for _, detection in sequence]
        classes = {item["class_id"] for item in hits}
        if len(classes) != 1:
            raise ValueError(f"Track T{track_id} changes detection class.")
        for (first, a), (next_frame, b) in pairwise(sequence):
            if first >= next_frame:
                raise ValueError(f"Track T{track_id} repeats or reorders frames.")
            if next_frame == first + 1:
                adjacent.append(
                    box_iou(Box(0, *a["bbox_xywh_normalized"]), Box(0, *b["bbox_xywh_normalized"]))
                )
        groups.append(
            {
                "track_id": track_id,
                "class_id": hits[0]["class_id"],
                "frames": [frame for frame, _ in sequence],
                "observations": len(hits),
                "peak_confidence": max(item["confidence"] for item in hits),
            }
        )
    event_counts = Counter(event["class_id"] for event in events)
    return {
        "detections_by_class": {code: counts[code] for code in CLASS_CODES.values()},
        "events_by_class": {code: event_counts[code] for code in CLASS_CODES.values()},
        "temporal_groups": len(groups),
        "linked_observations": counts.total() - len(groups),
        "adjacent_frame_links": len(adjacent),
        "median_adjacent_frame_box_iou": median(adjacent) if adjacent else None,
        "single_observation_groups": sum(group["observations"] == 1 for group in groups),
        "frames_with_detections": frames_with_detections,
        "groups": groups,
    }


def finalize_user_video(root: Path, run: Path) -> Path:
    """Reconcile source hashes, full-video outputs, hardware and quality checks."""
    import cv2

    experiments = root / "artifacts/experiments"
    if run.parent != experiments or not run.name.startswith(PREFIX):
        raise ValueError("Select the archived Stage 7 user-video run.")
    file = run / "metadata.json"
    metadata = read_json(file)
    if metadata["status"] not in {"measured", "partial_pass", "failed_quality_gates"}:
        raise ValueError("The video run must be fully measured before review.")
    media = read_json(run / "video_source.json")
    source, original = Path(media["project_copy"]), Path(media["original_path"])
    expected = media["sha256"]
    identity = metadata["identity"]
    if (
        not source.resolve().is_relative_to(root / "data/field_validation/user_supplied")
        or Path(identity["source"]) != source
        or {identity["source_sha256"], sha256(original), sha256(source)} != {expected}
        or media["provenance"] != metadata["source_provenance"]
    ):
        raise ValueError("The source or archived provenance does not match the attachment.")
    integrity = verify_frozen_state(root, read_json(experiments / "stage6_result.json"))
    if integrity != read_json(run / "preflight.json"):
        raise ValueError(
            "Frozen detector or dataset evidence changed during field-video validation."
        )
    integrity.update(original_video_sha256=expected, copied_video_sha256=expected)
    write_json(run / "integrity.json", integrity)
    runtime = read_json(run / "runtime_metrics.json")
    metrics = runtime["throughput"]
    settings = identity["config"]["vehicle"]
    directory = run / "throughput"
    if (
        metrics["processed_frames"] != media["decoded_frames"]
        or media["total_frames"] != media["decoded_frames"]
        or metrics["source_fps"] != media["fps"]
        or metrics["software_skipped_frames"] != 0
        or metrics["end_reason"] != "end_of_file"
        or metrics["inference_device"] != "cuda:0"
        or metadata["environment"]["gpu"] != "NVIDIA GeForce RTX 4070 Laptop GPU"
        or metrics["location_mode"] != "none"
    ):
        raise ValueError("Incomplete video, unexpected GPU, or unapproved location mode.")
    events = verify_outputs(directory, metrics, settings)
    if any(
        event["severity"] != "Unknown"
        or event["location"]["source"] != "unavailable"
        or event["location"]["simulated"]
        or event["location"]["latitude"] is not None
        or event["location"]["longitude"] is not None
        for event in events
    ):
        raise ValueError("An event contains unexpected location or severity data.")
    reader = cv2.VideoCapture(str(directory / "annotated_video.mp4"))
    try:
        if not reader.isOpened() or (
            int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            reader.get(cv2.CAP_PROP_FPS),
        ) != (media["width"], media["height"], media["fps"]):
            raise ValueError("Annotated MP4 dimensions or timing differ from the source.")
    finally:
        reader.release()
    associations = summarize_associations(directory, events)
    metrics.update({key: value for key, value in associations.items() if key != "groups"})
    write_json(run / "runtime_metrics.json", runtime)
    write_json(directory / "runtime_metrics.json", metrics)
    write_json(run / "temporal_associations.json", associations)
    observations = read_json(run / "field_observations.json")
    if not observations["review_images"] or any(
        not (run / rel).is_file() for rel in observations["review_images"]
    ):
        raise ValueError("The visual observations require saved review evidence.")
    summary = {
        "status": "quality_gates_pending",
        "run": metadata["run"],
        "model": metadata["model"],
        "checkpoint": metadata["checkpoint"],
        "source_kind": metadata["identity"]["source_kind"],
        "source_provenance": metadata["source_provenance"],
        "field_validation_status": "independent_real_road_video_completed",
        "vehicle_hardware_validation_status": "pending",
        "runtime": runtime,
        "video_source": media,
        "visual_observations": observations,
        "integrity": integrity,
        "output_verified": True,
    }
    report = run / "stage7_result.json"
    write_json(report, summary)
    gates = run_quality_gates(root, run, NOTEBOOK, notebook_env={"ROADGUARD_STAGE7_RUN": str(run)})
    status = "partial_pass" if all(gates.values()) else "failed_quality_gates"
    metadata.update(status=status, quality_gates=gates, video_source="video_source.json")
    summary.update(status=status, quality_gates=gates)
    write_json(file, metadata)
    write_json(report, summary)
    write_json(experiments / "stage7_user_video_result.json", summary)
    if status == "failed_quality_gates":
        raise RuntimeError(f"Stage 7 user-video quality gates failed: {gates}")
    return report
