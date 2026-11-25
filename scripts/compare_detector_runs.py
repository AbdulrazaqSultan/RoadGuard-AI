"""Save a matched comparison of two completed validation-selected runs."""

import argparse
from pathlib import Path

from roadguard.cli import run
from roadguard.evaluation.comparison import compare_validation_runs
from roadguard.utils.paths import project_root


def main() -> None:
    """Compare validated checkpoints without opening any dataset images."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or project_root() / "artifacts/experiments/stage3_comparison.json"
    report = compare_validation_runs(args.baseline, args.challenger, output)
    print(f"Validation selected {report['selected_run']}. Report saved to {output}")


if __name__ == "__main__":
    run(main)
