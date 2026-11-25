"""Run the one frozen Stage 6 test pass or finalize its saved artifacts."""

import argparse
from pathlib import Path

from roadguard.cli import run
from roadguard.evaluation.stage6 import evaluate_test_once
from roadguard.utils.paths import project_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalize", type=Path, help="Report from the existing capture only.")
    args = parser.parse_args()
    root = project_root()
    if args.finalize is None:
        print(evaluate_test_once(root))
    else:
        from roadguard.evaluation.stage6_report import finalize_test

        print(finalize_test(root, args.finalize.resolve()))


if __name__ == "__main__":
    run(main)
