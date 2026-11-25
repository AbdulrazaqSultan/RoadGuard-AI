"""Validate Stage 7.1 on both archived independent road-video runs."""

import argparse
from pathlib import Path

from roadguard.cli import run
from roadguard.config import load_config
from roadguard.utils.paths import project_root
from roadguard.vehicle.stage7_1 import finalize_stage7_1, replay_and_persist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalize", type=Path, help="Run quality gates on an existing replay.")
    args = parser.parse_args()
    root = project_root()
    if args.finalize:
        print(finalize_stage7_1(root, args.finalize.resolve()))
    else:
        config = load_config(
            root,
            profile=root / "configs/vehicle/stage7_1.yaml",
            extra_defaults={"vehicle": "vehicle/default.yaml"},
        )
        print(replay_and_persist(root, config))


if __name__ == "__main__":
    run(main)
