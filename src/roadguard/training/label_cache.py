"""Place Ultralytics label caches outside the frozen dataset."""

from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from threading import RLock
from typing import Any
from unittest.mock import patch

_PATCH_LOCK = RLock()


@contextmanager
def isolated_label_cache(
    snapshot: Path,
    outputs: Path,
    *,
    splits: tuple[str, ...] = ("train", "val"),
) -> Iterator[Path]:
    """Redirect the pinned backend's split caches without changing its validation logic."""
    from filelock import FileLock
    from ultralytics.data.dataset import YOLODataset

    if outputs.resolve().is_relative_to(snapshot.resolve()):
        raise ValueError("The label cache must be outside the dataset snapshot.")
    directory = outputs / version("ultralytics") / snapshot.name
    directory.mkdir(parents=True, exist_ok=True)
    expected = {snapshot / "labels" / f"{split}.cache" for split in splits}
    with _PATCH_LOCK:
        load_or_scan = YOLODataset._load_or_scan_cache

        def redirect(dataset: Any, cache_path: Path, cache_hash: str) -> tuple[dict, bool]:
            if cache_path not in expected:
                raise ValueError(
                    f"Unexpected dataset cache path for the selected splits: {cache_path}"
                )
            target = directory / cache_path.name
            with FileLock(str(target.with_suffix(".lock"))):
                return load_or_scan(dataset, target, cache_hash)

        with patch.object(YOLODataset, "_load_or_scan_cache", redirect):
            yield directory
