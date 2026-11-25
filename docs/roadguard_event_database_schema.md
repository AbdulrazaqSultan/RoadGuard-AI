# RoadGuard event database

Stage 7.1 stores completed capture sessions in SQLite at `data/runtime/roadguard_events.db`.
The schema is initialized by `EventStore` in `src/roadguard/vehicle/event_store.py`
and recorded as SQLite `user_version=1`.

| Table | Key and constraints | Contents |
|---|---|---|
| `sessions` | `session_id` is a required UUID4 primary key | Generic source ID, source type and locator, UTC start and end, FPS when known, frame dimensions, checkpoint, pipeline version and archive directory |
| `events` | `id` is the local integer primary key. `event_id` is a required unique UUID4. `session_id` references `sessions` and is indexed. | Class, confidence, first and last frame, representative frame, detection and track counts, normalized XYXY box, timestamps, evidence paths, location, severity, model, checkpoint and pipeline version |

Each session can contain many events. Evidence and crop paths are relative to the session's
`archive_dir`. SQLite also retains the complete JSON event in `payload_json` for round-trip
verification. JSONL, CSV and SQLite share the same event UUID. The optional session event
number is only for human reference.

`EventStore.save_session()` writes the session and its events in one transaction.
An identical retry with the same UUIDs keeps the existing rows. Reusing a UUID
with different data raises an error and rolls back the transaction. A new capture session
gets a new UUID and can create new events even when its boxes resemble an earlier session.

The source ID and source type are independent of file names. A later phone, USB camera
or RTSP adapter can supply its own locator and publish events under a capture UUID.
In the Stage 7.1 replay, both geographic coordinates are SQL `NULL`, the source is
`unavailable`, and severity is `Unknown`. Cross-session physical-defect matching still
requires a way to establish identity across views.
