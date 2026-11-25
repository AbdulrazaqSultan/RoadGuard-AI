"""Prepare visual cases or score completed event-identity review annotations."""

import argparse
from pathlib import Path

from roadguard.cli import run
from roadguard.utils.paths import project_root
from roadguard.vehicle.ground_truth_review import finish_review, prepare_review


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalize", type=Path, help="Stage 7.2 review archive")
    parser.add_argument("--adjudications", type=Path, help="Manually reviewed CSV")
    parser.add_argument("--t48-notes", type=Path, help="Manual T48 assessment JSON")
    args = parser.parse_args()
    root = project_root()
    if args.finalize:
        if args.adjudications is None or args.t48_notes is None:
            parser.error("--finalize requires --adjudications and --t48-notes")
        print(finish_review(root, args.finalize.resolve(), args.adjudications, args.t48_notes))
    else:
        print(prepare_review(root))


if __name__ == "__main__":
    run(main)
