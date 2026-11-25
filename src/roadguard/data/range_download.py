"""Sequentially verify archives assembled from resumable Hugging Face byte ranges."""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from roadguard.data.unified_release import REPOSITORY, REVISION
from roadguard.data.validation import sha256

LOGGER = logging.getLogger(__name__)
CHUNK_BYTES = 8 * 1024 * 1024
RETRIES = 4


def _retrieve(client: Any, url: str, part: Path, start: int, end: int, size: int) -> None:
    byte_range = f"bytes={start}-{end}"
    with client.stream(
        "GET", url, params={"download": "true", "range": f"{start}-{end}"},
        headers={"Range": byte_range, "Accept-Encoding": "identity"},
    ) as response:
        if response.status_code != 206:
            raise OSError(f"Range request returned HTTP {response.status_code}")
        if response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
            raise OSError("Unexpected Content-Range from Hub")
        received = 0
        with part.open("r+b", buffering=0) as output:
            output.seek(start)
            for chunk in response.iter_raw(chunk_size=1024 * 1024):
                received += len(chunk)
                if received > end - start + 1:
                    raise OSError("Range exceeded the expected length")
                if output.write(chunk) != len(chunk):
                    raise OSError("Partial write to archive buffer")
            if received != end - start + 1:
                raise OSError("Incomplete archive range")
            os.fsync(output.fileno())


def download_ranges(
    root: Path, relative: str, size: int, checksum: str, *, workers: int,
) -> None:
    """Resume at completed byte ranges using the same public Hub resolve URL."""
    import httpx
    from huggingface_hub import hf_hub_url

    output = root / relative
    cache = root / ".cache" / "ranges" / checksum
    cache.mkdir(parents=True, exist_ok=True)
    part = cache / "payload.incomplete"
    if not part.exists():
        with part.open("xb") as stream:
            stream.truncate(size)
    elif part.stat().st_size != size:
        raise ValueError(f"Unexpected partial archive size: {part}")

    ranges = [(start, min(start + CHUNK_BYTES, size) - 1) for start in range(0, size, CHUNK_BYTES)]
    missing = [(start, end) for start, end in ranges if not (cache / f"{start}.done").is_file()]
    complete = size - sum(end - start + 1 for start, end in missing)
    started = time.monotonic()
    last_log = started
    url = hf_hub_url(REPOSITORY, relative, repo_type="dataset", revision=REVISION)
    LOGGER.info("Resuming %s at %s / %s bytes", relative, f"{complete:,}", f"{size:,}")

    with httpx.Client(follow_redirects=True, timeout=60) as client:
        def fetch(start: int, end: int) -> int:
            for attempt in range(RETRIES):
                try:
                    _retrieve(client, url, part, start, end, size)
                except (httpx.HTTPError, OSError) as error:
                    if attempt == RETRIES - 1:
                        detail = str(error) if isinstance(error, OSError) else type(error).__name__
                        raise OSError(f"Range {start}-{end} failed: {detail}") from error
                    time.sleep(3 * 2**attempt)
                else:
                    (cache / f"{start}.done").touch()
                    return end - start + 1
            raise AssertionError("Unreachable range retry state")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            tasks = [executor.submit(fetch, start, end) for start, end in missing]
            failures = []
            consecutive = 0
            try:
                for task in as_completed(tasks):
                    try:
                        complete += task.result()
                    except OSError as error:
                        failures.append(str(error))
                        LOGGER.warning("%s: %s", relative, error)
                        consecutive += 1
                        if consecutive >= max(4, workers):
                            raise OSError("Repeated range failures; preserved completed slices.") from error
                        continue
                    consecutive = 0
                    now = time.monotonic()
                    if complete == size or now - last_log >= 10:
                        LOGGER.info(
                            "%s: %s / %s bytes (%.1f%%)",
                            relative, f"{complete:,}", f"{size:,}", 100 * complete / size,
                        )
                        last_log = now
            except BaseException:
                for task in tasks:
                    task.cancel()
                raise
            if failures:
                raise OSError(
                    f"{len(failures)} range(s) incomplete in {relative}. "
                    f"Last failure: {failures[-1]}"
                )

    if sha256(part) != checksum:
        raise ValueError(f"Archive checksum failed. Preserve the partial file for review: {part}")
    output.parent.mkdir(parents=True, exist_ok=True)
    part.rename(output)
