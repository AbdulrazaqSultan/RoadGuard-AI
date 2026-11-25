"""Transactional SQLite storage for completed RoadGuard capture sessions."""

import json
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import UUID

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY NOT NULL,
    source_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_locator TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT NOT NULL,
    source_fps REAL,
    frame_width INTEGER NOT NULL,
    frame_height INTEGER NOT NULL,
    checkpoint_path TEXT,
    pipeline_version TEXT NOT NULL,
    archive_dir TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    event_id TEXT NOT NULL UNIQUE,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    source_id TEXT NOT NULL,
    class_id TEXT NOT NULL CHECK(class_id IN ('D00','D10','D20','D40')),
    class_name TEXT NOT NULL,
    confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
    first_seen_frame INTEGER NOT NULL,
    last_seen_frame INTEGER NOT NULL,
    representative_frame INTEGER NOT NULL,
    detection_count INTEGER NOT NULL CHECK(detection_count > 0),
    track_count INTEGER NOT NULL CHECK(track_count > 0),
    bbox_x1 REAL NOT NULL, bbox_y1 REAL NOT NULL, bbox_x2 REAL NOT NULL, bbox_y2 REAL NOT NULL,
    video_timestamp REAL,
    captured_at TEXT,
    evidence_frame_path TEXT NOT NULL,
    crop_path TEXT,
    location_latitude REAL,
    location_longitude REAL,
    location_source TEXT NOT NULL,
    location_simulated INTEGER NOT NULL CHECK(location_simulated IN (0, 1)),
    severity TEXT NOT NULL,
    model_name TEXT NOT NULL,
    checkpoint_path TEXT,
    pipeline_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    CHECK(first_seen_frame <= representative_frame AND representative_frame <= last_seen_frame)
);
CREATE INDEX IF NOT EXISTS events_by_session ON events(session_id);
PRAGMA user_version = 1;
"""


@dataclass(frozen=True)
class CaptureSession:
    session_id: str
    source_id: str
    source_type: str
    source_locator: str
    started_at: str
    ended_at: str
    source_fps: float | None
    frame_width: int
    frame_height: int
    checkpoint_path: str | None
    pipeline_version: str
    archive_dir: str


def _uuid(value: str) -> None:
    parsed = UUID(value)
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError("Expected a canonical UUID4.")


def _event_fields(event: dict) -> dict:
    names = (
        "event_id",
        "session_id",
        "source_id",
        "class_id",
        "class_name",
        "confidence",
        "first_seen_frame",
        "last_seen_frame",
        "representative_frame",
        "detection_count",
        "captured_at",
        "evidence_frame_path",
        "crop_path",
        "severity",
        "model_name",
        "pipeline_version",
    )
    x, y, w, h = event["bbox_xywh_normalized"]
    location = event["location"]
    return {name: event[name] for name in names} | {
        "track_count": len(event["track_ids"]),
        "bbox_x1": x - w / 2,
        "bbox_y1": y - h / 2,
        "bbox_x2": x + w / 2,
        "bbox_y2": y + h / 2,
        "video_timestamp": event["video_timestamp_s"],
        "location_latitude": location["latitude"],
        "location_longitude": location["longitude"],
        "location_source": location["source"],
        "location_simulated": int(location["simulated"]),
        "checkpoint_path": event["checkpoint"],
        "created_at": event["recorded_at"],
        "payload_json": json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False),
    }


def read_events(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


class EventStore:
    """Persist a whole session atomically and reject conflicting reuse of an ID."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise ValueError(f"Unsupported RoadGuard SQLite schema version {version}.")
            if not version:
                connection.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def save_session(self, session: CaptureSession, events: list[dict]) -> dict[str, int]:
        """Return inserted and blocked counts; exact retries leave stored records unchanged."""
        _uuid(session.session_id)
        payload = asdict(session)
        inserted = duplicates = 0
        placeholders = ",".join("?" for _ in payload)
        with closing(self._connect()) as connection, connection:
            connection.execute(
                f"INSERT INTO sessions ({','.join(payload)}) VALUES ({placeholders}) "
                "ON CONFLICT(session_id) DO NOTHING",
                tuple(payload.values()),
            )
            record = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session.session_id,)
            ).fetchone()
            if dict(record) != payload:
                raise ValueError("The existing session ID belongs to different metadata.")
            for event in events:
                event_id = event["event_id"]
                _uuid(event_id)
                if (
                    event["session_id"] != session.session_id
                    or event["source_id"] != session.source_id
                ):
                    raise ValueError("Event does not belong to this capture session.")
                data = _event_fields(event)
                columns, parameters = ",".join(data), ",".join("?" for _ in data)
                cursor = connection.execute(
                    f"INSERT INTO events ({columns}) VALUES ({parameters}) "
                    "ON CONFLICT(event_id) DO NOTHING",
                    tuple(data.values()),
                )
                if cursor.rowcount:
                    inserted += 1
                else:
                    existing = connection.execute(
                        f"SELECT {columns} FROM events WHERE event_id = ?", (event_id,)
                    ).fetchone()
                    if dict(existing) != data:
                        raise ValueError(f"Event ID collision with a different payload: {event_id}")
                    duplicates += 1
        return {"inserted": inserted, "duplicates_blocked": duplicates}

    def get_event(self, event_id: str) -> dict | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT payload_json FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def verify_session(self, session_id: str, events: list[dict]) -> int:
        expected = {event["event_id"]: _event_fields(event) for event in events}
        with closing(self._connect()) as connection:
            persisted = connection.execute(
                "SELECT * FROM events WHERE session_id = ?", (session_id,)
            )
            rows = [dict(row) for row in persisted]
        for row in rows:
            del row["id"]
        actual = {row["event_id"]: row for row in rows}
        if len(events) != len(rows) or len(expected) != len(events) or actual != expected:
            raise ValueError(f"SQLite and archived events differ for session {session_id}.")
        return len(rows)

    def inspect(self) -> dict:
        """Check physical integrity, foreign keys, unique ID constraint and row totals."""
        with closing(self._connect()) as connection:
            integrity = [row[0] for row in connection.execute("PRAGMA integrity_check")]
            foreign_keys = list(connection.execute("PRAGMA foreign_key_check"))
            unique = connection.execute(
                """SELECT 1 FROM pragma_index_list('events') AS i
                   WHERE i."unique" = 1 AND i.partial = 0
                   AND (SELECT group_concat(name) FROM pragma_index_info(i.name)) = 'event_id'"""
            ).fetchone()
            field = connection.execute(
                """SELECT "notnull" FROM pragma_table_info('events') WHERE name = 'event_id'"""
            ).fetchone()
            sessions = connection.execute("SELECT count(*) FROM sessions").fetchone()[0]
            total = connection.execute("SELECT count(*) FROM events").fetchone()[0]
            classes = dict(
                connection.execute("SELECT class_id, count(*) FROM events GROUP BY class_id")
            )
        if integrity != ["ok"] or foreign_keys or not unique or not field or field[0] != 1:
            raise ValueError("SQLite integrity, foreign key or event ID constraint check failed.")
        return {
            "integrity_check": "ok",
            "foreign_key_check": "ok",
            "event_id_unique_not_null": True,
            "total_sessions": sessions,
            "total_unique_events": total,
            "events_by_class": classes,
        }
