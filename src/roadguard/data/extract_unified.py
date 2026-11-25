"""Safely unpack a verified dataset release for local annotation inspection."""

import logging
import shutil
import tarfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from tempfile import TemporaryDirectory

from roadguard.data.acquisition import acquire
from roadguard.data.unified import discover_layout, read_class_config
from roadguard.data.unified_release import ARCHIVES, REPOSITORY, REVISION
from roadguard.utils.paths import write_json

LOGGER = logging.getLogger(__name__)


def extract_archive(archive: Path, destination: Path) -> tuple[int, int]:
    """Copy regular files only and reject unsafe or duplicate member paths."""
    files = size = 0
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            path = PurePosixPath(member.name)
            invalid = path.is_absolute() or any(
                part == ".." or any(char in part for char in '\\:<>"|?*')
                or part.endswith((" ", ".")) or PureWindowsPath(part).is_reserved()
                for part in path.parts
            )
            if invalid or not (member.isdir() or member.isfile()):
                raise ValueError(f"Unsupported member {member.name!r} in {archive.name}.")
            target = destination.joinpath(*path.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            contents = stream.extractfile(member)
            if contents is None:
                raise ValueError(f"Unreadable member {member.name!r} in {archive.name}.")
            with contents, target.open("xb") as output:
                shutil.copyfileobj(contents, output, length=1024 * 1024)
            files += 1
            size += member.size
    return files, size


def extract_release(root: Path, *, verify_archives: bool = True) -> Path:
    """Verify pinned payloads, detect the layout and publish a complete local extract."""
    from filelock import FileLock

    if verify_archives:
        acquire(root, offline=True)
    read_class_config(root / "rdd_merged.yaml")
    with FileLock(str(root / ".extract.lock"), timeout=0):
        output = root / "extracted"
        if output.is_dir():
            if not (output / "acquisition.json").is_file():
                raise ValueError(f"Unrecognized existing extract: {output}")
            discover_layout(root)
            LOGGER.info("Reusing local extract at %s", output)
            return output

        with TemporaryDirectory(prefix=".extract-", dir=root) as staging:
            extracted = Path(staging) / "extracted"
            extracted.mkdir()
            counts = {}
            for archive in ARCHIVES:
                LOGGER.info("Extracting %s", archive)
                files, size = extract_archive(root / archive, extracted)
                counts[archive] = {"files": files, "bytes": size}
                LOGGER.info("Extracted %d files from %s (%s bytes)", files, archive, f"{size:,}")
            write_json(extracted / "acquisition.json", {
                "repository": REPOSITORY, "revision": REVISION, "archives": counts,
            })
            discover_layout(root)
            extracted.rename(output)
        return output
