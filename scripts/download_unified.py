"""Download and verify the pinned Unified Road Defect Dataset release."""

import argparse
import os
from pathlib import Path

from roadguard.cli import run
from roadguard.data.acquisition import acquire
from roadguard.utils.paths import project_root


def main() -> None:
    """Choose a local acquisition directory and download or verify its files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--ranges", action="store_true", help="Use resumable HTTP byte ranges")
    args = parser.parse_args()
    root = (args.root or project_root() / "data" / "raw" / "unified").resolve()
    os.environ.setdefault("HF_XET_CACHE", str(root / ".cache" / "xet"))
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    acquire(root, offline=args.offline, workers=args.workers, use_ranges=args.ranges)


if __name__ == "__main__":
    run(main)
