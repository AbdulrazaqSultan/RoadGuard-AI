import base64
import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from roadguard.lan.api import create_app
from roadguard.lan.location import event_location, read_trace
from roadguard.lan.store import stamp

TOKEN = "location-regression-credential-12345"
UPLOAD = {
    "Authorization": f"Bearer {TOKEN}",
    "X-Video-Extension": ".mp4",
    "X-RoadGuard-Origin": "mobile_camera",
}


def encoded(fixes):
    return base64.urlsafe_b64encode(json.dumps(fixes).encode()).decode()


def test_only_fresh_precise_fixes_match_the_event_timestamp():
    fixes = read_trace(encoded([[0, 15.1, 44.1, 90], [3000, 15.2, 44.2, 8]]))
    assert event_location(0, fixes) == {}
    assert event_location(2.3, fixes)["location_time_offset_ms"] == 700
    assert event_location(2.3, fixes)["latitude"] == 15.2
    assert event_location(4.6, fixes) == {}
    assert event_location(None, fixes) == {}
    for bad in (
        encoded([[1000, 0, 0, 1], [1000, 0, 0, 1]]),
        encoded([[500, 0, 181, 6]]),
        encoded([[500, True, 45, 6]]),
        encoded([[500, 0, 0, float("nan")]]),
        "*" * 10,
    ):
        with pytest.raises(ValueError, match="invalid_location_trace"):
            read_trace(bad)


def test_camera_ingestion_and_address_preserve_existing_identity(tmp_path):
    session = str(uuid4())

    def fixture(_video, _output):
        return [
            {
                "event_id": str(uuid4()),
                "session_id": session,
                "recorded_at": stamp(),
                "captured_at": None,
                "video_timestamp_s": timestamp,
                "class_id": "D20",
                "class_name": "alligator_crack",
                "confidence": 0.77,
                "first_seen_frame": 0,
                "last_seen_frame": 3,
                "representative_frame": 1,
                "detection_count": 3,
                "source_id": "validation clip",
            }
            for timestamp in (1.0, 6.0, 12.0)
        ]

    with TestClient(create_app(tmp_path, TOKEN, fixture)) as client:
        trace = {"X-RoadGuard-Location": encoded([[1000, 15.1, 44.1, 8], [6000, 15.2, 44.2, 32]])}
        gallery = UPLOAD | trace | {"X-RoadGuard-Origin": "mobile_gallery"}
        assert client.post("/jobs", headers=gallery, content=b"clip").status_code == 422
        assert client.post("/jobs", headers=UPLOAD | trace, content=b"clip").status_code == 202
        first, second, third = reversed(client.get("/events", headers=UPLOAD).json())
        assert first["latitude"] == 15.1 and second["latitude"] == 15.2
        assert first["captured_at"] is None and first["location_source"] == "camera_device"
        assert third["location_available"] is False and third["latitude"] is None
        address = {
            "address_region": "ذمار",
            "address_street": "الدائري الغربي",
            "address_landmark": "مثال تحقق",
        }
        a = client.patch(f"/events/{first['event_id']}/address", headers=UPLOAD, json=address)
        assert a.status_code == 200 and a.json()["address_street"] == "الدائري الغربي"
        assert a.json()["updated_at"] == first["updated_at"]
        b = client.patch(f"/events/{second['event_id']}/address", headers=UPLOAD, json=address)
        assert b.status_code == 200 and b.json()["address_region"] == "ذمار"
        assert "address_street" not in b.json() and "address_landmark" not in b.json()
        c = client.patch(f"/events/{third['event_id']}/address", headers=UPLOAD, json=address)
        assert c.status_code == 409
