"""Create training-set statistics, charts and annotated samples."""

from roadguard.cli import config_parser, configured, run
from roadguard.data.workflow import analyze


def main() -> None:
    """Run reusable EDA code with the configured dataset and output paths."""
    args = config_parser(__doc__).parse_args()
    root, config = configured(args)
    analyze(root, config)


if __name__ == "__main__":
    run(main)
