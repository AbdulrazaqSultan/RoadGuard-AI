"""Load repository YAML defaults with checked, layered overrides."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from roadguard.utils.paths import project_root

DEFAULT_CONFIGS = {
    "data": "data/default.yaml",
    "model": "models/default.yaml",
    "training": "training/default.yaml",
    "experiment": "experiment.yaml",
}
Config = dict[str, Any]


def _mapping(path: Path) -> Config:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def _merge(target: Config, changes: Config, prefix: str = "") -> None:
    for key, value in changes.items():
        address = f"{prefix}{key}"
        if key not in target:
            raise ValueError(f"Unknown setting: {address}")
        default = target[key]
        if isinstance(default, dict) and isinstance(value, dict):
            _merge(default, value, f"{address}.")
            continue
        if key in {"weights", "groups_csv"}:
            valid_type = value is None or isinstance(value, str)
        elif type(default) is float:
            valid_type = type(value) in (int, float)
        else:
            valid_type = type(value) is type(default) and not isinstance(value, dict)
        if not valid_type:
            raise ValueError(f"Invalid value for {address}: {value!r}")
        target[key] = value


def load_config(
    root: Path | None = None,
    *,
    profile: Path | None = None,
    overrides: Iterable[str] = (),
    extra_defaults: dict[str, str] | None = None,
) -> Config:
    """Apply defaults, an optional profile, then dotted KEY=YAML_VALUE arguments."""
    root = root or project_root()
    defaults = DEFAULT_CONFIGS | (extra_defaults or {})
    config = {key: _mapping(root / "configs" / path) for key, path in defaults.items()}
    if profile is not None:
        _merge(config, _mapping(profile))
    for override in overrides:
        key, separator, value = override.partition("=")
        parts = key.split(".")
        if not separator or len(parts) < 2 or not all(parts):
            raise ValueError(f"Expected section.setting=value, received {override!r}")
        change = yaml.safe_load(value)
        for part in reversed(parts):
            change = {part: change}
        _merge(config, change)
    return config
