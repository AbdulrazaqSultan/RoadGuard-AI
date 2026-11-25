"""Optionally fetch the official RDD2022 archive without extracting it."""

import argparse
import logging
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

from roadguard.cli import run
from roadguard.data.validation import validate_dataset
from roadguard.utils.paths import project_root

ARCHIVE_URL = (
    "https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/RDD2022.zip"
)
LOGGER = logging.getLogger(__name__)


def verify_archive(path: Path) -> None:
    """Require a complete ZIP containing labeled images and annotations."""
    try:
        with ZipFile(path) as archive:
            names = archive.namelist()
            if not any("/annotations/xmls/" in name and name.endswith(".xml") for name in names):
                raise ValueError(f"No labeled RDD2022 files in {path}.")
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"CRC mismatch in {bad_member}: {path}.")
    except BadZipFile as error:
        raise ValueError(f"Invalid RDD2022 ZIP: {path}.") from error


def main() -> None:
    """Reuse a validated extract or archive before attempting a download."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    root = project_root()
    dataset = args.dataset_root or root / "data" / "raw" / "rdd2022"
    archive = args.archive or root / "data" / "raw" / "RDD2022.zip"
    if dataset.is_dir() and any(dataset.iterdir()):
        report = validate_dataset(dataset, "voc")
        if report.valid:
            LOGGER.info("Existing RDD2022 extract passed validation: %s", dataset)
            return
        raise ValueError(f"Existing extract needs review: {dataset}. Run prepare_dataset.py.")
    if archive.exists():
        verify_archive(archive)
        LOGGER.info("Existing RDD2022 archive passed validation: %s", archive)
        return
    archive.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading the official RDD2022 archive to %s", archive)
    with NamedTemporaryFile(dir=archive.parent, suffix=".part", delete=False) as output:
        temporary = Path(output.name)
    try:
        with urlopen(ARCHIVE_URL, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        verify_archive(temporary)
        # Windows rename fails if a different process already published this archive.
        if archive.exists():
            verify_archive(archive)
        else:
            temporary.rename(archive)
    finally:
        temporary.unlink(missing_ok=True)
    LOGGER.info("Archive verified. Extract labeled country folders into %s", dataset)


if __name__ == "__main__":
    run(main)
