"""Project path discovery and atomic report writes."""

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def project_root(start: Path | None = None) -> Path:
    """Locate the repository from a script, the repository root, or a notebook."""
    directory = (start or Path.cwd()).resolve()
    for candidate in (directory, *directory.parents):
        if (candidate / "configs" / "experiment.yaml").is_file():
            return candidate
    raise FileNotFoundError(f"Could not locate the RoadGuard repository from {directory}")


def write_text(path: Path, content: str) -> None:
    """Publish a complete UTF-8 file via an atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as temporary:
        temp_path = Path(temporary.name)
        try:
            temporary.write(content)
        except BaseException:
            temporary.close()
            temp_path.unlink(missing_ok=True)
            raise
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def write_json(path: Path, data: Any) -> None:
    """Write a machine-readable report without nonstandard NaN values."""
    write_text(path, json.dumps(data, indent=2, allow_nan=False) + "\n")
