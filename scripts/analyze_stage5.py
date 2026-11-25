"""Capture and review errors from Stage 4 best.pt on the frozen validation split."""

import argparse
from pathlib import Path

from roadguard.cli import run
from roadguard.evaluation.stage5 import capture_validation
from roadguard.utils.paths import project_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--analyze", type=Path, help="Analyze an existing validation capture")
    mode.add_argument("--finalize", type=Path, help="Check and document an analyzed run")
    args = parser.parse_args()
    root = project_root()
    if args.analyze or args.finalize:
        from roadguard.evaluation.stage5_review import analyze_capture, finalize_analysis

        if args.analyze:
            print(analyze_capture(root, args.analyze.resolve()))
        else:
            print(finalize_analysis(root, args.finalize.resolve()))
    else:
        print(capture_validation(root))


if __name__ == "__main__":
    run(main)
