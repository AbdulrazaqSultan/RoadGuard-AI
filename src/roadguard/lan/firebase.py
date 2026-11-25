"""Server-side Firebase delivery and optional Firestore mirroring."""

import json
import re
from datetime import UTC, datetime
from urllib import error, parse, request

from roadguard.lan.store import stamp


class FirebaseDeliveryError(RuntimeError):
    """A provider error suitable for reporting without exposing tokens."""

    def __init__(self, status: int, code: str) -> None:
        self.status, self.code = status, code
        super().__init__(f"Firebase {status}: {code}")


class FirebasePublisher:
    def __init__(self, project: str, *, mirror: bool = False) -> None:
        if not re.fullmatch(r"[a-z][a-z\d-]{4,28}[a-z\d]", project):
            raise ValueError("Set the actual Firebase project ID.")
        import google.auth
        from google.auth.transport.requests import Request

        self.credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        self.transport = Request()
        self.project = project
        self.mirror = mirror

    def _call(self, url: str, method: str, payload: dict) -> None:
        headers = {"Content-Type": "application/json"}
        try:
            self.credentials.before_request(self.transport, method, url, headers)
            call = request.Request(url, json.dumps(payload).encode(), headers, method=method)
            with request.urlopen(call, timeout=15):
                pass
        except error.HTTPError as exc:
            try:
                response = json.loads(exc.read(16_384) or b"{}").get("error", {})
            except (UnicodeError, ValueError, AttributeError):
                response = {}
            detail = next(
                (
                    item.get("errorCode")
                    for item in response.get("details", [])
                    if item.get("errorCode")
                ),
                response.get("status", "REJECTED"),
            )
            if not re.fullmatch(r"[A-Z_]+", detail):
                detail = "REJECTED"
            raise FirebaseDeliveryError(exc.code, detail) from None
        except Exception:
            raise FirebaseDeliveryError(503, "CONNECTION_OR_CREDENTIALS") from None

    def send(self, token: str, event: dict, *, test: bool = False) -> None:
        priority = event["priority_status"]
        if priority not in {"important", "critical"}:
            raise ValueError("Only important or critical events can trigger notifications.")
        priority_label = "حرجة" if priority == "critical" else "مهمة"
        title = "إشعار RoadGuardAI تجريبي" if test else f"أولوية {priority_label} لمتابعة عيب طريق"
        body = (
            "وصل تنبيه الاختبار من الكمبيوتر"
            if test
            else (f"{event['class_id']} • {event['event_id'][:8].upper()}")
        )
        data = {
            "event_id": event["event_id"],
            "class_id": event["class_id"],
            "priority_status": priority,
            "kind": "test" if test else "event",
        }
        if event.get("captured_at"):
            data["captured_at"] = event["captured_at"]
        self._call(
            f"https://fcm.googleapis.com/v1/projects/{self.project}/messages:send",
            "POST",
            {
                "message": {
                    "token": token,
                    "notification": {"title": title, "body": body},
                    "data": data,
                    "android": {
                        "priority": "HIGH",
                        "ttl": "120s",
                        "notification": {"channel_id": "roadguard_priority"},
                    },
                }
            },
        )

    def publish(self, event: dict, fields: tuple[str, ...] | None = None) -> None:
        if not self.mirror:
            return
        document = event if fields is None else {key: event[key] for key in fields}
        body = {
            "fields": {
                key: firestore_value(key, value)
                for key, value in document.items()
                if key != "evidence_available"
            }
        }
        mask = ""
        if fields is not None:
            mask = "?" + parse.urlencode(
                [
                    ("currentDocument.exists", "true"),
                    *(("updateMask.fieldPaths", key) for key in fields),
                ]
            )
        self._call(
            f"https://firestore.googleapis.com/v1/projects/{self.project}"
            f"/databases/(default)/documents/road_defect_events/{event['event_id']}{mask}",
            "PATCH",
            body,
        )


def firestore_value(key: str, value: object) -> dict:
    if value is None:
        return {"nullValue": None}
    if key in {"captured_at", "created_at", "updated_at"}:
        return {"timestampValue": datetime.fromisoformat(str(value)).astimezone(UTC).isoformat()}
    if isinstance(value, bool):
        return {"booleanValue": value}
    if isinstance(value, int):
        return {"integerValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    return {"stringValue": str(value)}


def probe_event(event_id: str) -> dict:
    return {
        "event_id": event_id,
        "class_id": "TEST",
        "priority_status": "important",
        "captured_at": stamp(),
    }
