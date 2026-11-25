"""Stable experiment identities and collision-free attempt directories."""

import hashlib
import json
import re
from pathlib import Path


def reserve_run(parent: Path, name: str, identity: dict) -> tuple[str, Path]:
    """Reserve an independent attempt for a reproducible configuration identity."""
    if re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,47}", name) is None:
        raise ValueError(
            "Use 1–48 letters, digits, underscores or hyphens for the experiment name."
        )
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode())
    experiment_id = f"{name}-{digest.hexdigest()[:12]}"
    parent.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while True:
        directory = parent / f"{experiment_id}-{attempt:03d}"
        try:
            directory.mkdir()
            return experiment_id, directory
        except FileExistsError:
            attempt += 1
