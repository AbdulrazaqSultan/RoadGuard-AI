"""Serve the Android operations API on the computer running RoadGuardAI."""

import argparse
import ipaddress
import os
from pathlib import Path

import uvicorn

from roadguard.lan.api import VideoProcessor, create_app
from roadguard.lan.firebase import FirebasePublisher

PRIVATE = tuple(
    ipaddress.ip_network(value) for value in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Computer's private IPv4 address.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--firebase-project", help="Use Google ADC to send FCM")
    parser.add_argument("--mirror-firestore", action="store_true")
    parser.add_argument("--tls-cert", type=Path)
    parser.add_argument("--tls-key", type=Path)
    args = parser.parse_args()
    address = ipaddress.IPv4Address(args.host)
    if not address.is_loopback and not any(address in network for network in PRIVATE):
        parser.error("Bind only to loopback or the PC's private IPv4 LAN address.")
    if args.mirror_firestore and not args.firebase_project:
        parser.error("--mirror-firestore requires --firebase-project.")
    if bool(args.tls_cert) != bool(args.tls_key):
        parser.error("Provide both TLS options or neither.")
    key = os.environ.get("ROADGUARD_LAN_TOKEN", "")
    if len(key) < 24 or not key.isascii():
        parser.error("Set ROADGUARD_LAN_TOKEN to a random value of at least 24 ASCII characters.")
    root = Path(__file__).resolve().parents[1]
    publisher = (
        FirebasePublisher(args.firebase_project, mirror=args.mirror_firestore)
        if args.firebase_project
        else None
    )
    app = create_app(
        root / "data/runtime/lan_ops", key, VideoProcessor(root, args.device), publisher
    )
    uvicorn.run(
        app,
        host=str(address),
        port=args.port,
        workers=1,
        access_log=False,
        ssl_certfile=args.tls_cert,
        ssl_keyfile=args.tls_key,
    )


if __name__ == "__main__":
    main()
