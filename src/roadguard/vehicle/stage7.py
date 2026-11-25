"""Guard the frozen detector and archive software or supplied-source vehicle runs."""

import json
import logging
import math
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any

from roadguard.data.prepare import read_snapshot
from roadguard.data.validation import sha256
from roadguard.evaluation.stage5 import BEST, BEST_SHA256
from roadguard.inference.detector import Detector
from roadguard.training.experiment import reserve_run
from roadguard.training.screening import FINGERPRINT, verify_inventory, verify_raw_archives
from roadguard.utils.device import environment_info
from roadguard.utils.paths import write_json
from roadguard.vehicle.event_store import CaptureSession, EventStore, read_events
from roadguard.vehicle.events import (
    PIPELINE_VERSION,
    EventRecorder,
    NoLocation,
    SimulatedRoute,
    unknown_severity,
)
from roadguard.vehicle.pipeline import run_pipeline
from roadguard.vehicle.software_fixture import ScriptedDetections, write_software_video
from roadguard.vehicle.source import OpenCVSource

EXPERIMENT = "stage7_real_time_vehicle_pipeline"
PROFILE = Path("configs/vehicle/stage7.yaml")
LOGGER = logging.getLogger(__name__)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_frozen_state(root: Path, stage6: dict) -> dict:
    """Check the checkpoint, manifest and archives without opening split media."""
    prior = stage6["integrity"]
    snapshot = root / "data/processed" / FINGERPRINT
    checkpoint = root / BEST
    if (
        stage6["status"] != "passed"
        or stage6["checkpoint"] != BEST.as_posix()
        or prior["status"] != "passed"
        or prior["snapshot_fingerprint"] != FINGERPRINT
        or prior["checkpoint_sha256"] != BEST_SHA256
        or checkpoint.resolve() != checkpoint
        or sha256(checkpoint) != BEST_SHA256
    ):
        raise ValueError("Stage 7 requires the exact detector and snapshot accepted in Stage 6.")
    read_snapshot(snapshot)
    inventory = verify_inventory(root, prior | {"snapshot": str(snapshot)})
    archives = verify_raw_archives(root / "data/raw/unified")
    if archives != prior["raw_archive_sha256"]:
        raise ValueError("Raw archive hashes changed since Stage 6.")
    return {
        "status": "passed",
        "snapshot_fingerprint": FINGERPRINT,
        "checkpoint_sha256": BEST_SHA256,
        "protected_inventory": inventory,
        "raw_archive_sha256": archives,
        "training_occurred": False,
        "stage6_test_reused_for_field_validation": False,
    }


def checked_source(root: Path, video: Path | None, camera: int | None) -> str | int:
    """Require an explicit capture source and exclude protected dataset/experiment media."""
    if (video is None) == (camera is None):
        raise ValueError("Select exactly one independent video or camera device.")
    if camera is not None:
        if camera < 0:
            raise ValueError("Camera index must be nonnegative.")
        return camera
    assert video is not None
    file = video.resolve()
    for protected in ("data/raw", "data/processed", "artifacts/experiments"):
        if file.is_relative_to((root / protected).resolve()):
            raise ValueError(f"Use independent input outside protected {protected}.")
    if not file.is_file():
        raise FileNotFoundError(file)
    return str(file)


def _gpu_temperature() -> float | None:
    try:
        sample = subprocess.run(
            [
                "nvidia-smi",
                "--id=0",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=4,
            check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        value = float(sample.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return value if math.isfinite(value) else None


def measure_pipeline(
    output: Path,
    source: str | int,
    mode: str,
    name: str,
    config: dict,
    infer: Any,
    *,
    location_mode: str,
    checkpoint: str | None,
    gpu: bool,
    warmup_ms: float | None = None,
    store: EventStore | None = None,
) -> dict:
    settings = config["vehicle"]
    location_settings = settings["location"]
    location = NoLocation()
    if location_mode == "mock":
        location = SimulatedRoute(
            location_settings["mock_latitude"],
            location_settings["mock_longitude"],
            location_settings["mock_longitude_step_per_second"],
        )
    source_id = Path(source).name if isinstance(source, str) else f"camera:{source}"
    recorder = EventRecorder(
        output,
        f"{output.parent.name}-{output.name}",
        source_id,
        location,
        unknown_severity,
        name,
        checkpoint,
        crops=settings["write_crops"],
    )
    temperature = []
    started_at = datetime.now(UTC).isoformat()
    if gpu:
        import torch

        torch.cuda.reset_peak_memory_stats(0)
        temperature.append(_gpu_temperature())
    with OpenCVSource(source) as frames:
        report = run_pipeline(
            frames,
            infer,
            recorder,
            settings,
            mode=mode,
            confidence=config["model"]["confidence_threshold"],
        )
        report["reported_source_frames"] = frames.reported_frames
        report["timestamp_fallbacks"] = frames.timestamp_fallbacks
    if gpu:
        temperature.append(_gpu_temperature())
    session = CaptureSession(
        recorder.session_id,
        source_id,
        "recorded_video" if isinstance(source, str) else "usb_camera",
        str(source),
        started_at,
        datetime.now(UTC).isoformat(),
        report["source_fps"],
        report["frame_width"],
        report["frame_height"],
        checkpoint,
        PIPELINE_VERSION,
        output.as_posix(),
    )
    write_json(output / "session.json", asdict(session))
    if store is not None:
        persisted = read_events(output / "events.jsonl")
        report["database_write"] = store.save_session(session, persisted)
        store.verify_session(session.session_id, persisted)
    report.update(
        predictor=name,
        location_mode=location_mode,
        severity="Unknown",
        session_id=session.session_id,
        warmup_ms_excluded=warmup_ms,
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(0) if gpu else None,
        gpu_temperature_c=[sample for sample in temperature if sample is not None],
    )
    write_json(output / "runtime_metrics.json", report)
    LOGGER.info(
        "%s processed %d frames, %d detections and %d events",
        output.name,
        report["processed_frames"],
        report["raw_detections"],
        report["final_events"],
    )
    return report


def _validate_config(config: dict, stage6: dict, root: Path) -> str:
    settings, model = config["vehicle"], config["model"]
    required = read_json(root / stage6["run"] / "frozen_protocol.json")["test_identity"]
    if (
        model["model_family"] != "yolo26"
        or model["model_variant"] != "s"
        or model["weights"] != BEST.as_posix()
        or config["training"]["image_size"] != required["imgsz"]
        or model["confidence_threshold"] != required["operating_confidence"]
        or model["iou_threshold"] != required["nms_iou"]
    ):
        raise ValueError("Use the frozen checkpoint, image size and operational confidence/NMS.")
    if (
        settings["device"] not in {"auto", "cuda", "cpu"}
        or settings["severity"] != "unknown"
        or settings["location"]["mode"] not in {"none", "mock"}
        or settings["cpu_threads"] < 1
        or settings["frame_limit"] < 0
        or not math.isfinite(settings["output_fps_fallback"])
        or settings["output_fps_fallback"] <= 0
    ):
        raise ValueError("Invalid vehicle runtime, location or severity settings.")
    return settings["device"]


def run_stage7(
    root: Path, config: dict, source: str | int | None, mode: str, provenance: str
) -> Path:
    """Execute the same sequential core for file playback, camera and software fixture."""
    experiments = root / config["experiment"]["experiments_root"]
    if experiments.resolve() != root / "artifacts/experiments":
        raise ValueError("Archive Stage 7 beneath the project's artifacts/experiments directory.")
    stage6 = read_json(experiments / "stage6_result.json")
    requested = _validate_config(config, stage6, root)
    evidence = verify_frozen_state(root, stage6)
    environment = environment_info()
    device = requested
    if requested == "auto":
        device = "cuda" if environment["cuda_available"] else "cpu"
    if device == "cpu":
        import torch

        torch.set_num_threads(config["vehicle"]["cpu_threads"])
    source_kind = (
        "synthetic_software"
        if source is None
        else ("camera" if isinstance(source, int) else "prerecorded_video")
    )
    identity = {
        "mode": mode,
        "source_kind": source_kind,
        "source": source,
        "benchmark_warmup_frames": 1 if source is None else 0,
        "source_sha256": sha256(Path(source)) if isinstance(source, str) else None,
        "checkpoint_sha256": BEST_SHA256,
        "config": config,
        "device": device,
        "pipeline_version": PIPELINE_VERSION,
    }
    _, run = reserve_run(experiments, config["experiment"]["experiment_name"], identity)
    metadata = {
        "status": "running",
        "run": run.relative_to(root).as_posix(),
        "identity": identity,
        "environment": environment,
        "model": "YOLO26s",
        "checkpoint": BEST.as_posix(),
        "checkpoint_epoch": 49,
        "predict_precision": "FP16" if device == "cuda" else "FP32",
        "source_provenance": provenance,
        "training_occurred": False,
        "stage6_test_reused_for_field_validation": False,
    }
    write_json(run / "metadata.json", metadata)
    write_json(run / "preflight.json", evidence)
    write_json(run / "pipeline_config.json", config)
    LOGGER.info("Stage 7 running in %s on %s", run, device)
    try:
        warm_frame = None
        if source is None:
            synthetic = run / "synthetic_input.mp4"
            write_software_video(synthetic)
            source = str(synthetic)
            metadata["synthetic_input_sha256"] = sha256(synthetic)
            with OpenCVSource(source) as video:
                warm_frame = video.read()
            if warm_frame is None:
                raise ValueError("The synthetic benchmark input has no decodable frames.")
            modes = ("throughput", "realtime")
        else:
            modes = (mode,)
        metrics = {}
        database = config["vehicle"]["event_database"]
        store = (
            EventStore(root / database["path"])
            if (source_kind != "synthetic_software" and database["enabled"])
            else None
        )
        for benchmark in modes:
            detector = Detector(root / BEST, device=device)
            infer = partial(
                detector.predict_frame,
                image_size=config["training"]["image_size"],
                confidence=config["model"]["confidence_threshold"],
                iou=config["model"]["iou_threshold"],
            )
            warmup_ms = None
            if warm_frame is not None:
                start = perf_counter()
                infer(warm_frame.pixels)
                warmup_ms = (perf_counter() - start) * 1000
            metrics[benchmark] = measure_pipeline(
                run / benchmark,
                source,
                benchmark,
                "YOLO26s",
                config,
                infer,
                location_mode=config["vehicle"]["location"]["mode"],
                checkpoint=BEST.as_posix(),
                gpu=device == "cuda",
                warmup_ms=warmup_ms,
                store=store,
            )
            actual_device = detector.execution_device
            if actual_device != ("cuda:0" if device == "cuda" else "cpu"):
                raise RuntimeError(f"Expected {device} but the predictor used {actual_device}.")
            metrics[benchmark]["inference_device"] = actual_device
            write_json(run / benchmark / "runtime_metrics.json", metrics[benchmark])
            del detector, infer
        if source_kind == "synthetic_software":
            metrics["event_fixture"] = measure_pipeline(
                run / "event_fixture",
                source,
                "throughput",
                "scripted_fixture",
                config,
                ScriptedDetections(),
                location_mode="mock",
                checkpoint=None,
                gpu=False,
            )
        write_json(run / "runtime_metrics.json", metrics)
        metadata["status"] = "measured"
    except BaseException:
        metadata["status"] = "interrupted"
        raise
    finally:
        write_json(run / "metadata.json", metadata)
    return run
