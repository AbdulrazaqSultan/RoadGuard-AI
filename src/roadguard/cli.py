"""Shared script arguments and consistent command error handling."""

import argparse
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import yaml

from roadguard.config import Config, load_config
from roadguard.utils.logging import configure_logging
from roadguard.utils.paths import project_root


def config_parser(description: str) -> argparse.ArgumentParser:
    """Create a parser with shared configuration options."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, help="YAML profile layered on the defaults")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    return parser


def configured(args: argparse.Namespace) -> tuple[Path, Config]:
    """Resolve the repository root and requested settings."""
    root = project_root()
    return root, load_config(root, profile=args.config, overrides=args.overrides)


@contextmanager
def command_errors() -> Iterator[None]:
    """Render expected operational failures as a structured command error."""
    try:
        yield
    except (ValueError, OSError, RuntimeError, ImportError, yaml.YAMLError) as error:
        logging.getLogger("roadguard").error("%s", error)
        raise SystemExit(1) from error


def run(main: Callable[[], None]) -> None:
    """Set up logging and dispatch one command."""
    configure_logging()
    with command_errors():
        main()
