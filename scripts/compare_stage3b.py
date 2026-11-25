"""Read the four validation archives and optionally save the screening recommendation."""

import argparse
import json
import os

from roadguard.cli import run
from roadguard.evaluation.screening import save_screening_comparison, screening_comparison
from roadguard.utils.paths import project_root


def main() -> None:
    """Review only saved metrics and checkpoint architecture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recommend", choices=("YOLO11n", "YOLO11s", "YOLO26n", "YOLO26s"))
    parser.add_argument("--rationale")
    parser.add_argument("--longer-training-justified", action="store_true")
    args = parser.parse_args()
    if args.recommend and not args.rationale:
        parser.error("--recommend requires an evidence-based --rationale")
    os.environ.setdefault("YOLO_OFFLINE", "true")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    root = project_root()
    report = screening_comparison(root)
    if args.recommend:
        output = save_screening_comparison(
            root,
            report,
            {
                "model": args.recommend,
                "rationale": args.rationale,
                "longer_training_justified": args.longer_training_justified,
                "longer_training_started": False,
            },
        )
        print(f"Saved Stage 3B comparison to {output}")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run(main)
