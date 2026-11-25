"""Prepare a validation-selected YOLO11 training run on CUDA."""

from pathlib import Path

from roadguard.cli import config_parser, configured, run
from roadguard.training.trainer import train


def main() -> None:
    """Launch a new run or continue an explicitly selected interrupted run."""
    parser = config_parser(__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Prepared snapshot directory")
    parser.add_argument("--resume", type=Path, help="Interrupted experiment directory")
    args = parser.parse_args()
    root, config = configured(args)
    print(train(root, config, args.dataset, resume=args.resume))


if __name__ == "__main__":
    run(main)
