"""Acquire verified, resumable dataset archives through the Hugging Face Hub client."""

import hashlib
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from roadguard.data.unified_release import ARCHIVES, REPOSITORY, REVISION, SIDECARS
from roadguard.data.validation import sha256

LOGGER = logging.getLogger(__name__)


def _verify(path: Path, size: int, digest: str, *, lfs: bool) -> None:
    if path.stat().st_size != size:
        raise ValueError(f"Wrong byte count for {path}. Move the invalid file aside and retry.")
    if lfs:
        actual = sha256(path)
    else:
        content = path.read_bytes()
        actual = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    if actual != digest:
        raise ValueError(f"Checksum mismatch for {path}. Move the invalid file aside and retry.")


def acquire(
    root: Path, *, offline: bool = False, workers: int = 2, use_ranges: bool = False,
) -> None:
    """Verify existing files and resume missing payloads at the pinned revision."""
    from filelock import FileLock, Timeout

    if workers < 1:
        raise ValueError("Use at least one download worker.")
    root.mkdir(parents=True, exist_ok=True)
    inventory = SIDECARS | ARCHIVES
    lock = FileLock(str(root / ".acquire.lock"), timeout=0)
    try:
        lock.acquire()
    except Timeout as error:
        raise RuntimeError(f"Another process is acquiring data under {root}") from error
    try:
        missing = []
        for relative, (size, digest) in inventory.items():
            path = root / relative
            if path.is_file():
                _verify(path, size, digest, lfs=relative in ARCHIVES)
                LOGGER.info("Verified existing %s (%s bytes)", relative, f"{size:,}")
            else:
                missing.append(relative)

        if offline and missing:
            raise FileNotFoundError(f"Missing local files: {', '.join(missing)}")
        if missing:
            required = sum(inventory[relative][0] for relative in missing)
            cache = root / ".cache" / ("ranges" if use_ranges else "huggingface")
            partial = sum(file.stat().st_size for file in cache.rglob("*.incomplete"))
            if shutil.disk_usage(root).free < max(0, required - partial):
                raise OSError(f"Insufficient free space for the {required:,} missing payload bytes.")
            logging.getLogger("httpx").setLevel(logging.WARNING)
            from huggingface_hub import hf_hub_download

            def download(relative: str) -> None:
                LOGGER.info("Fetching %s (%s bytes)", relative, f"{inventory[relative][0]:,}")
                hf_hub_download(
                    REPOSITORY, relative, repo_type="dataset", revision=REVISION,
                    local_dir=root, token=False, etag_timeout=30,
                )
                _verify(root / relative, *inventory[relative], lfs=relative in ARCHIVES)
                LOGGER.info("Verified %s", relative)

            try:
                if use_ranges:
                    from roadguard.data.range_download import download_ranges

                    for relative in missing:
                        if relative in ARCHIVES:
                            download_ranges(root, relative, *inventory[relative], workers=workers)
                        else:
                            download(relative)
                else:
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        list(executor.map(download, missing))
            except Exception as error:
                detail = str(error) if use_ranges and isinstance(error, OSError) else (
                    type(error).__name__
                )
                raise RuntimeError(
                    f"Hub download interrupted: {detail}. "
                    "Run the same command again to resume."
                ) from error
    finally:
        lock.release()

    LOGGER.info(
        "Acquired %d verified files, total %s bytes, resumed payload %s bytes, revision %s",
        len(inventory), f"{sum(size for size, _ in inventory.values()):,}",
        f"{sum(inventory[name][0] for name in missing):,}", REVISION,
    )
