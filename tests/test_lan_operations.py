from uuid import uuid4

from fastapi.testclient import TestClient

from roadguard.lan.api import create_app
from roadguard.lan.store import stamp

TOKEN = "local-validation-key-only-123456"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
UPLOAD = HEADERS | {
    "X-Video-Extension": ".mp4",
    "X-RoadGuard-Origin": "mobile_camera",
}


class Sender:
    mirror = False

    def __init__(self):
        self.sent = []

    def send(self, token, event, *, test=False):
        self.sent.append((token, event, test))


def test_upload_events_and_priority_push(tmp_path):
    event_id, session_id = str(uuid4()), str(uuid4())

    def detector(input_path, output):
        assert input_path.read_bytes() == b"video"
        evidence = output / "evidence" / f"{event_id}.jpg"
        evidence.parent.mkdir(parents=True)
        evidence.write_bytes(b"evidence")
        return [
            {
                "event_id": event_id,
                "session_id": session_id,
                "recorded_at": stamp(),
                "captured_at": None,
                "class_id": "D20",
                "class_name": "alligator_crack",
                "confidence": 0.99,
                "first_seen_frame": 0,
                "last_seen_frame": 2,
                "representative_frame": 1,
                "detection_count": 3,
                "source_id": "mobile clip",
                "evidence_frame_path": evidence.relative_to(output).as_posix(),
            }
        ]

    sender = Sender()
    with TestClient(create_app(tmp_path, TOKEN, detector, sender)) as client:
        assert client.get("/events").status_code == 401
        assert client.post("/jobs", headers=UPLOAD, content=b"").status_code == 400
        assert client.post("/jobs", headers=UPLOAD, content=b"video").status_code == 202
        event = client.get(f"/events/{event_id}", headers=HEADERS).json()
        assert event["priority_status"] == "normal"
        assert event["source_type"] == "mobile_camera"
        assert event["captured_at"] is None
        assert event["severity"] == "Unknown"
        assert event["latitude"] is None
        assert client.get(f"/events/{event_id}/evidence").status_code == 401
        evidence = client.get(f"/events/{event_id}/evidence", headers=HEADERS)
        assert evidence.content == b"evidence"
        assert not list(tmp_path.glob("jobs/*/mobile_video.*"))
        installation = f"/installations/{uuid4()}"
        body = {"token": "example-registration-token-for-test-only"}
        assert client.put(installation, headers=HEADERS, json=body).status_code == 200
        event_url = f"/events/{event_id}"
        assert (
            client.patch(event_url, headers=HEADERS, json={"status": "reviewing"}).status_code
            == 200
        )
        assert not sender.sent
        for _ in range(2):
            change = client.patch(event_url, headers=HEADERS, json={"priority_status": "important"})
            assert change.status_code == 200
        assert len(sender.sent) == 1
        assert sender.sent[0][1]["event_id"] == event_id
        assert client.post(f"{installation}/test", headers=HEADERS).json()["accepted_by_fcm"]
        assert sender.sent[1][2]
        assert sender.sent[1][1]["event_id"] != event_id
        assert client.post(f"{installation}/test", headers=HEADERS).status_code == 429
        assert client.delete(installation, headers=HEADERS).json() == {"registered": False}


def test_truncated_upload_does_not_block_other_video(tmp_path):
    def detector(_input, _output):
        return []

    with TestClient(create_app(tmp_path, TOKEN, detector)) as client:
        unknown = UPLOAD | {"X-Video-Extension": "../mp4"}
        assert client.post("/jobs", headers=unknown, content=b"video").status_code == 415
        assert client.post("/jobs", headers=UPLOAD, content=b"").status_code == 400
        job_id = client.post("/jobs", headers=UPLOAD, content=b"video").json()["id"]
        assert client.get(f"/jobs/{job_id}", headers=HEADERS).json()["state"] == "completed"
