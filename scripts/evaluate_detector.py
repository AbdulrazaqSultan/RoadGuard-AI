"""Evaluate the best checkpoint of a completed RoadGuard experiment."""

import argparse
from pathlib import Path

from roadguard.cli import run
from roadguard.evaluation.evaluator import evaluate


def main() -> None:
    """Evaluate on validation by default or on the explicitly selected test split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Completed experiment directory")
    parser.add_argument("--dataset", type=Path, required=True, help="Original prepared snapshot")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    print(evaluate(args.run, args.dataset, args.split, batch_size=args.batch_size))


if __name__ == "__main__":
    run(main)
