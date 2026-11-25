"""Operations records isolated from the vehicle pipeline's SQLite database."""

import json
import sqlite3
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path

from roadguard.lan.location import (
    PLACE_ACCURACY_M,
    STREET_ACCURACY_M,
    LocationFix,
    event_location,
)


def stamp() -> str:
    return datetime.now(UTC).isoformat()


class OperationsStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.database = root / "operations.db"
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    detail TEXT,
                    events INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(id),
                    body TEXT NOT NULL
                );
                UPDATE jobs SET state = 'failed', detail = 'interrupted'
                    WHERE state IN ('receiving', 'processing');
                """
            )

    @contextmanager
    def connect(self):
        with closing(sqlite3.connect(self.database, timeout=8)) as conn, conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn

    def reserve(self, job_id: str) -> bool:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            active = conn.execute(
                "SELECT 1 FROM jobs WHERE state IN ('receiving', 'processing') LIMIT 1"
            ).fetchone()
            if active:
                return False
            conn.execute("INSERT INTO jobs (id,state) VALUES (?, 'receiving')", (job_id,))
            return True

    def set_job(self, job_id: str, state: str, detail: str | None = None) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE jobs SET state=?,detail=? WHERE id=?", (state, detail, job_id))

    def finish(self, job_id: str, events: list[dict], detail: str | None) -> None:
        with self.connect() as conn:
            conn.executemany(
                "INSERT INTO events VALUES (?,?,?)",
                [(event["event_id"], job_id, json.dumps(event)) for event in events],
            )
            conn.execute(
                "UPDATE jobs SET state='completed',events=?,detail=? WHERE id=?",
                (len(events), detail, job_id),
            )

    def job(self, job_id: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            return dict(row) if row else None

    def events(self) -> list[dict]:
        with self.connect() as conn:
            return [
                json.loads(row["body"])
                for row in conn.execute("SELECT body FROM events ORDER BY rowid DESC")
            ]

    def event(self, event_id: str) -> tuple[str, dict] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
            return (row["job_id"], json.loads(row["body"])) if row else None

    def update(self, event_id: str, changes: dict) -> tuple[dict, bool] | None:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT body FROM events WHERE id=?", (event_id,)).fetchone()
            if not row:
                return None
            event = json.loads(row["body"])
            elevated = changes.get("priority_status") in {"important", "critical"}
            send = elevated and changes["priority_status"] != event["priority_status"]
            event.update(changes, updated_at=stamp())
            conn.execute("UPDATE events SET body=? WHERE id=?", (json.dumps(event), event_id))
            return event, send

    def save_address(self, event_id: str, address: dict) -> tuple[dict, tuple[str, ...]] | None:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT body FROM events WHERE id=?", (event_id,)).fetchone()
            if row is None:
                return None
            event = json.loads(row["body"])
            if event.get("location_source") != "camera_device":
                return event, ()
            accuracy = event["location_accuracy_m"]
            if accuracy > STREET_ACCURACY_M:
                address.pop("address_street", None)
            if accuracy > PLACE_ACCURACY_M:
                address.pop("address_landmark", None)
            changes = {key: value for key, value in address.items() if not event.get(key)}
            if changes:
                event.update(changes, address_provider="android_geocoder")
                conn.execute("UPDATE events SET body=? WHERE id=?", (json.dumps(event), event_id))
                return event, (*changes, "address_provider")
            return event, ()


def operations_event(record: dict, origin: str, fixes: tuple[LocationFix, ...] = ()) -> dict:
    fields = (
        "event_id",
        "session_id",
        "class_id",
        "class_name",
        "confidence",
        "first_seen_frame",
        "last_seen_frame",
        "representative_frame",
        "detection_count",
        "source_id",
        "captured_at",
    )
    event = {key: record[key] for key in fields} | {
        "source_type": origin,
        "status": "new",
        "priority_status": "normal",
        "severity": "Unknown",
        "evidence_image_url": None,
        "evidence_local_path": None,
        "evidence_available": bool(record.get("evidence_frame_path")),
        "latitude": None,
        "longitude": None,
        "location_available": False,
        "created_at": record["recorded_at"],
        "updated_at": record["recorded_at"],
    }
    if origin == "mobile_camera":
        event.update(event_location(record.get("video_timestamp_s"), fixes))
    return event
