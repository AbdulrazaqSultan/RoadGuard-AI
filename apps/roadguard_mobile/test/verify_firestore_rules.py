"""Verify the RoadGuard web roles against disposable local Firebase emulators."""

import json
from uuid import uuid4

import requests

PROJECT = "demo-roadguard-admin"
AUTH = "http://127.0.0.1:9099/identitytoolkit.googleapis.com/v1"
DATABASE = f"projects/{PROJECT}/databases/(default)"
FIRESTORE = f"http://127.0.0.1:8080/v1/{DATABASE}"
COLLECTION = f"{FIRESTORE}/documents/road_defect_events"


def call(method: str, url: str, expected: int, *, token=None, body=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = requests.request(method, url, headers=headers, json=body, timeout=10)
    if response.status_code != expected:
        raise AssertionError(f"{method} {url} returned {response.status_code}; expected {expected}")
    return response.json()


def client(role: str | None) -> str:
    credentials = {
        "email": f"{uuid4().hex}@example.test",
        "password": uuid4().hex,
        "returnSecureToken": True,
    }
    account = call("POST", f"{AUTH}/accounts:signUp?key=test", 200, body=credentials)
    if role is None:
        return account["idToken"]
    call(
        "POST",
        f"{AUTH}/projects/{PROJECT}/accounts:update",
        200,
        token="owner",
        body={
            "localId": account["localId"],
            "customAttributes": json.dumps({"roadguard_role": role}),
        },
    )
    return call("POST", f"{AUTH}/accounts:signInWithPassword?key=test", 200, body=credentials)[
        "idToken"
    ]


def main():
    event_id = str(uuid4())
    document = f"{COLLECTION}/{event_id}"
    fields = {
        key: {"stringValue": value}
        for key, value in {
            "event_id": event_id,
            "status": "new",
            "priority_status": "normal",
        }.items()
    }
    call("PATCH", document, 200, token="owner", body={"fields": fields})
    try:
        call("GET", document, 403)
        member, viewer, editor, admin = [
            client(role) for role in (None, "viewer", "editor", "admin")
        ]
        call("GET", document, 403, token=member)
        for token in (viewer, editor, admin):
            call("GET", document, 200, token=token)

        def update(token, field, value, expected):
            call(
                "POST",
                f"{FIRESTORE}/documents:commit",
                expected,
                token=token,
                body={
                    "writes": [
                        {
                            "update": {
                                "name": f"{DATABASE}/documents/road_defect_events/{event_id}",
                                "fields": {field: value},
                            },
                            "updateMask": {"fieldPaths": [field]},
                            "updateTransforms": [
                                {"fieldPath": "updated_at", "setToServerValue": "REQUEST_TIME"}
                            ],
                        }
                    ]
                },
            )

        update(viewer, "status", {"stringValue": "confirmed"}, 403)
        update(editor, "status", {"stringValue": "reviewing"}, 200)
        update(admin, "priority_status", {"stringValue": "important"}, 200)
        update(admin, "status", {"stringValue": "invalid"}, 403)
        update(admin, "latitude", {"doubleValue": 15}, 403)
        call("PATCH", f"{COLLECTION}/{uuid4()}", 403, token=admin, body={"fields": fields})
        call("DELETE", document, 403, token=admin)
        print(
            "Firestore emulator passed: guest, unassigned, viewer, editor, admin, "
            "field limits, create and delete."
        )
    finally:
        call("DELETE", document, 200, token="owner")


if __name__ == "__main__":
    main()
