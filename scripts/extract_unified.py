"""Verify the pinned dataset archives and extract them for offline validation."""

from roadguard.cli import config_parser, configured, run
from roadguard.data.extract_unified import extract_release


def main() -> None:
    """Extract the configured Unified release."""
    parser = config_parser(__doc__)
    args = parser.parse_args()
    root, config = configured(args)
    print(extract_release(root / config["data"]["dataset_root"]))


if __name__ == "__main__":
    run(main)
