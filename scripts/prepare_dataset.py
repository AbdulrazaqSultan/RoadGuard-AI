"""Validate source data and materialize a deterministic YOLO snapshot."""

from roadguard.cli import config_parser, configured, run
from roadguard.data.workflow import prepare


def main() -> None:
    """Validate source data and optionally perform preparation."""
    parser = config_parser(__doc__)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root, config = configured(args)
    prepare(root, config, validate_only=args.validate_only)


if __name__ == "__main__":
    run(main)
