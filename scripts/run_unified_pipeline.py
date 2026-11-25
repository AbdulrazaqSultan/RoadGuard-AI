"""Acquire, extract, validate and prepare the Unified dataset in one run."""

import logging
import os
import time
from pathlib import Path

from roadguard.cli import config_parser, configured, run
from roadguard.data.acquisition import acquire
from roadguard.data.extract_unified import extract_release
from roadguard.data.workflow import prepare


def main() -> None:
    """Run Stage 2 through validation and publish only if the checks pass."""
    parser = config_parser(__doc__)
    parser.set_defaults(config=Path("configs/data/unified.yaml"))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--ranges", action="store_true")
    parser.add_argument("--attempts", type=int, default=1)
    args = parser.parse_args()
    root, config = configured(args)
    if config["data"]["source_format"] != "unified":
        raise ValueError("This pipeline requires data.source_format=unified.")
    if args.attempts < 1:
        raise ValueError("Use at least one acquisition attempt.")
    raw = (root / config["data"]["dataset_root"]).resolve()
    os.environ.setdefault("HF_XET_CACHE", str(raw / ".cache" / "xet"))
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    for attempt in range(args.attempts):
        try:
            acquire(raw, offline=args.offline, workers=args.workers, use_ranges=args.ranges)
        except RuntimeError:
            if args.offline or attempt + 1 == args.attempts:
                raise
            logging.getLogger(__name__).warning("Retrying the resumable download in 30 seconds.")
            time.sleep(30)
        else:
            break
    extract_release(raw, verify_archives=False)
    print(prepare(root, config))


if __name__ == "__main__":
    run(main)
