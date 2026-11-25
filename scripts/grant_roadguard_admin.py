"""Grant the web dashboard role to an existing Firebase Authentication user."""

import argparse
import json
import re

import google.auth
from google.auth.exceptions import GoogleAuthError
from google.auth.transport.requests import AuthorizedSession
from requests.exceptions import RequestException


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="Firebase project ID")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--email", help="Existing Firebase Authentication email")
    target.add_argument("--uid", help="Existing Firebase Authentication UID")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9-]{4,28}[a-z0-9]", args.project):
        parser.error("invalid Firebase project ID")

    endpoint = f"https://identitytoolkit.googleapis.com/v1/projects/{args.project}/accounts"
    selector = {"email": [args.email]} if args.email else {"localId": [args.uid]}
    try:
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
            quota_project_id=args.project,
        )
        with AuthorizedSession(credentials) as session:
            users = post(session, endpoint, "lookup", selector).get("users", [])
            if len(users) != 1 or users[0].get("disabled"):
                raise SystemExit("Expected one existing, enabled Firebase Authentication account.")
            user = users[0]
            claims = json.loads(user.get("customAttributes") or "{}")
            if claims.get("roadguard_role") != "admin":
                claims["roadguard_role"] = "admin"
                post(
                    session,
                    endpoint,
                    "update",
                    {"localId": user["localId"], "customAttributes": json.dumps(claims)},
                )
    except (GoogleAuthError, RequestException) as error:
        raise SystemExit(
            f"Firebase Authentication request failed: {type(error).__name__}"
        ) from None
    print(
        f"RoadGuard admin role confirmed in {args.project}. Sign in again to refresh the ID token."
    )


def post(session: AuthorizedSession, endpoint: str, action: str, body: dict) -> dict:
    response = session.post(f"{endpoint}:{action}", json=body, timeout=20)
    if not response.ok:
        raise SystemExit(f"Firebase accounts:{action} failed (HTTP {response.status_code}).")
    return response.json()


if __name__ == "__main__":
    main()
