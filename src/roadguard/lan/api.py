"""Authenticated, single-worker LAN API for recorded videos and Android notifications."""

import logging
import math
import secrets
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID, uuid4

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, StringConstraints

from roadguard.lan.firebase import FirebaseDeliveryError, FirebasePublisher, probe_event
from roadguard.lan.location import LocationFix, read_trace
from roadguard.lan.store import OperationsStore, operations_event

LOGGER = logging.getLogger(__name__)
MAX_BYTES = 150 * 1024 * 1024
MAX_SECONDS = 60
EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".3gp"}
WorkflowStatus = Literal["new", "reviewing", "confirmed", "scheduled", "resolved", "dismissed"]
Priority = Literal["normal", "important", "critical"]
AddressText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=150)]


class CameraAddress(BaseModel):
    address_region: AddressText | None = None
    address_district: AddressText | None = None
    address_city: AddressText | None = None
    address_area: AddressText | None = None
    address_street: AddressText | None = None
    address_landmark: AddressText | None = None


class Workflow(BaseModel):
    status: WorkflowStatus | None = None
    priority_status: Priority | None = None


class Registration(BaseModel):
    token: str = Field(min_length=20, max_length=4096, pattern=r"^[\w:-]+$")


class InvalidVideo(ValueError):
    pass


def create_app(
    root: Path,
    key: str,
    process: Callable[[Path, Path], list[dict]],
    publisher: FirebasePublisher | None = None,
) -> FastAPI:
    if len(key) < 24 or not key.isascii():
        raise ValueError("Provide a random ROADGUARD_LAN_TOKEN of at least 24 ASCII characters.")
    store = OperationsStore(root)
    devices: dict[UUID, str] = {}
    tested_at: dict[UUID, float] = {}
    notifications_lock = threading.Lock()

    def require_pairing(authorization: str = Header(default="")) -> None:
        if not secrets.compare_digest(authorization.encode(), f"Bearer {key}".encode()):
            raise HTTPException(401, "pairing_required")

    app = FastAPI(
        title="RoadGuardAI LAN",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        dependencies=[Depends(require_pairing)],
    )

    def sync(events: list[dict], fields: tuple[str, ...] | None = None) -> bool:
        if publisher is None or not publisher.mirror:
            return True
        ok = True
        for event in events:
            try:
                publisher.publish(event, fields)
            except FirebaseDeliveryError as exc:
                LOGGER.warning("Firestore: %s", exc)
                ok = False
        return ok

    def process_file(job_id: str, video: Path, origin: str, fixes: tuple[LocationFix, ...]) -> None:
        try:
            records = process(video, video.parent / "result")
            events = [operations_event(record, origin, fixes) for record in records]
            outcome = None if sync(events) else "firestore_sync_failed"
            store.finish(job_id, events, outcome)
        except InvalidVideo as exc:
            store.set_job(job_id, "failed", str(exc))
        except Exception:
            LOGGER.exception("Processing failed for job %s", job_id)
            store.set_job(job_id, "failed", "analysis_failed")
        finally:
            video.unlink(missing_ok=True)

    @app.get("/health")
    def health():
        return {"ok": True, "push_configured": publisher is not None}

    @app.post("/jobs", status_code=202)
    async def upload(
        incoming: Request,
        tasks: BackgroundTasks,
        extension: str = Header(alias="X-Video-Extension"),
        origin: Literal["mobile_gallery", "mobile_camera"] = Header(alias="X-RoadGuard-Origin"),
        location_trace: str | None = Header(default=None, alias="X-RoadGuard-Location"),
    ):
        extension = extension.lower()
        if extension not in EXTENSIONS:
            raise HTTPException(415, "unsupported_video")
        if location_trace is not None and origin != "mobile_camera":
            raise HTTPException(422, "camera_location_only")
        try:
            fixes = read_trace(location_trace)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        try:
            advertised = int(incoming.headers.get("content-length", "0"))
        except ValueError:
            raise HTTPException(400, "invalid_length") from None
        if advertised > MAX_BYTES:
            raise HTTPException(413, "video_too_large")
        job_id = str(uuid4())
        if not store.reserve(job_id):
            raise HTTPException(409, "server_busy")
        video = root / "jobs" / job_id / f"mobile_video{extension}"
        accepted = False
        failure = "upload_interrupted"
        try:
            video.parent.mkdir(parents=True)
            size = 0
            with video.open("xb") as stream:
                async for packet in incoming.stream():
                    size += len(packet)
                    if size > MAX_BYTES:
                        failure = "video_too_large"
                        raise HTTPException(413, failure)
                    stream.write(packet)
            if not size:
                failure = "empty_video"
                raise HTTPException(400, failure)
            store.set_job(job_id, "processing")
            tasks.add_task(process_file, job_id, video, origin, fixes)
            accepted = True
        finally:
            if not accepted:
                store.set_job(job_id, "failed", failure)
                video.unlink(missing_ok=True)
        return {"id": job_id, "state": "processing"}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: UUID):
        result = store.job(str(job_id))
        if result is None:
            raise HTTPException(404, "job_not_found")
        return result

    @app.get("/events")
    def events():
        return store.events()

    def require_event(event_id: UUID) -> tuple[str, dict]:
        result = store.event(str(event_id))
        if result is None:
            raise HTTPException(404, "event_not_found")
        return result

    @app.get("/events/{event_id}")
    def get_event(event_id: UUID):
        return require_event(event_id)[1]

    @app.patch("/events/{event_id}/address")
    def save_address(event_id: UUID, address: CameraAddress, tasks: BackgroundTasks):
        result = store.save_address(str(event_id), address.model_dump(exclude_none=True))
        if result is None:
            raise HTTPException(404, "event_not_found")
        event, fields = result
        if event.get("location_source") != "camera_device":
            raise HTTPException(409, "camera_location_unavailable")
        if fields:
            tasks.add_task(sync, [event], fields)
        return event

    @app.get("/events/{event_id}/evidence")
    def evidence(event_id: UUID):
        job_id, event = require_event(event_id)
        path = root / "jobs" / job_id / "result" / "evidence" / f"{event_id}.jpg"
        if not event["evidence_available"] or not path.is_file():
            raise HTTPException(404, "evidence_unavailable")
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    def deliver(installation: UUID, token: str, event: dict) -> None:
        try:
            publisher.send(token, event)
        except FirebaseDeliveryError as exc:
            LOGGER.warning("FCM: %s", exc)
            if exc.code == "UNREGISTERED":
                with notifications_lock:
                    if devices.get(installation) == token:
                        devices.pop(installation, None)
                        tested_at.pop(installation, None)

    @app.patch("/events/{event_id}")
    def update(event_id: UUID, workflow: Workflow, tasks: BackgroundTasks):
        changes = workflow.model_dump(exclude_none=True)
        if not changes:
            raise HTTPException(422, "missing_workflow_changes")
        result = store.update(str(event_id), changes)
        if result is None:
            raise HTTPException(404, "event_not_found")
        event, should_alert = result
        if publisher is not None:
            tasks.add_task(sync, [event], (*changes, "updated_at"))
            if should_alert:
                with notifications_lock:
                    registrations = list(devices.items())
                for installation, token in registrations:
                    tasks.add_task(deliver, installation, token, event)
        return event

    @app.put("/installations/{installation}")
    def register(installation: UUID, registration: Registration):
        if publisher is None:
            raise HTTPException(503, "push_not_configured")
        with notifications_lock:
            if installation not in devices and len(devices) >= 30:
                raise HTTPException(409, "installation_limit")
            for previous, token in list(devices.items()):
                if token == registration.token and previous != installation:
                    devices.pop(previous)
                    tested_at.pop(previous, None)
            devices[installation] = registration.token
        return {"registered": True}

    @app.delete("/installations/{installation}")
    def unregister(installation: UUID):
        with notifications_lock:
            devices.pop(installation, None)
            tested_at.pop(installation, None)
        return {"registered": False}

    @app.post("/installations/{installation}/test")
    def test_push(installation: UUID):
        if publisher is None:
            raise HTTPException(503, "push_not_configured")
        with notifications_lock:
            token = devices.get(installation)
            if token is None:
                raise HTTPException(404, "installation_not_registered")
            now = time.monotonic()
            if now - tested_at.get(installation, float("-inf")) < 10:
                raise HTTPException(429, "test_cooldown")
            tested_at[installation] = now
        try:
            publisher.send(token, probe_event(str(uuid4())), test=True)
        except FirebaseDeliveryError as exc:
            LOGGER.warning("FCM test: %s", exc)
            raise HTTPException(502, exc.code) from None
        return {"accepted_by_fcm": True}

    return app


class VideoProcessor:
    def __init__(self, project: Path, device: str) -> None:
        self.project = project
        self.device = device
        self.detector = None

    def __call__(self, video: Path, output: Path) -> list[dict]:
        import torch

        from roadguard.config import load_config
        from roadguard.data.validation import sha256
        from roadguard.evaluation.stage5 import BEST, BEST_SHA256
        from roadguard.inference.detector import Detector
        from roadguard.utils.paths import write_json
        from roadguard.vehicle.event_store import read_events
        from roadguard.vehicle.events import EventRecorder, NoLocation, unknown_severity
        from roadguard.vehicle.pipeline import run_pipeline
        from roadguard.vehicle.source import OpenCVSource

        config = load_config(
            self.project,
            profile=self.project / "configs/vehicle/stage7_1.yaml",
            extra_defaults={"vehicle": "vehicle/default.yaml"},
        )
        settings = config["vehicle"]
        checkpoint = self.project / BEST
        try:
            with OpenCVSource(str(video)) as frames:
                maximum_frames = math.floor(MAX_SECONDS * (frames.fps or 30))
                if frames.reported_frames is None or frames.fps is None:
                    raise InvalidVideo("unknown_video_duration")
                if frames.reported_frames > maximum_frames or frames.fps > 120:
                    raise InvalidVideo("video_too_long")
                settings["frame_limit"] = min(maximum_frames + 1, frames.reported_frames + 1)
                if self.detector is None:
                    if sha256(checkpoint) != BEST_SHA256:
                        raise RuntimeError("Frozen checkpoint verification failed")
                    device = self.device
                    if device == "auto":
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                    if device == "cpu":
                        torch.set_num_threads(settings["cpu_threads"])
                    self.detector = Detector(checkpoint, device=device)

                def infer(pixels):
                    return self.detector.predict_frame(
                        pixels,
                        image_size=config["training"]["image_size"],
                        confidence=config["model"]["confidence_threshold"],
                        iou=config["model"]["iou_threshold"],
                    )

                recorder = EventRecorder(
                    output,
                    output.parent.name,
                    output.parent.name,
                    NoLocation(),
                    unknown_severity,
                    "YOLO26s",
                    BEST.as_posix(),
                    crops=settings["write_crops"],
                )
                report = run_pipeline(
                    frames,
                    infer,
                    recorder,
                    settings,
                    mode="throughput",
                    confidence=config["model"]["confidence_threshold"],
                )
        except OSError:
            raise InvalidVideo("video_decode_failed") from None
        if report["end_reason"] != "end_of_file":
            raise InvalidVideo("video_too_long")
        write_json(output / "runtime_metrics.json", report)
        return read_events(output / "events.jsonl")
