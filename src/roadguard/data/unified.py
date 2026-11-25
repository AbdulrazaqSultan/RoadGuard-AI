"""Validate publisher classes, discover extracted partitions and preserve provenance."""

import re
from pathlib import Path, PurePosixPath

import yaml

from roadguard.data.dataset import CLASS_NAMES, RDD_CLASSES, Provenance

PUBLISHER_SPLITS = ("train", "val")
COUNTRIES = {
    "united_states": "United States",
    "czech": "Czech Republic",
    "japan": "Japan",
    "norway": "Norway",
    "india": "India",
    "china": "China",
    "spain": "Spain",
}
SOURCE_NAMES = {"rdd": "RDD2022", "uav": "UAV-PDD2023", "rdv": "RoadDamageVision"}


def infer_provenance(stem: str) -> Provenance:
    """Recognize publisher prefixes, explicit country tags and parent-image suffixes."""
    prefix, _, name = stem.casefold().partition("_")
    if prefix not in SOURCE_NAMES:
        return Provenance()

    country = next(
        (country for tag, country in COUNTRIES.items() if name.startswith(f"{tag}_")), None
    )
    view = None
    group = None
    if prefix == "uav":
        # This source's publisher identifies its capture country and platform.
        country, view = "China", "aerial"
        tile = re.fullmatch(r"(.+\d)_(?:top|bottom)_(?:left|right)", name)
        if tile:
            group = f"{prefix}:{tile[1]}"
    elif prefix == "rdv":
        view = "aerial"
        # Roboflow gives renditions of one original distinct .rf. hashes.
        rendition = re.fullmatch(r"(.+?)(?:_jpg|_jpeg|_png)?\.rf\.[0-9a-f]{16,}", name)
        if rendition:
            group = f"{prefix}:{rendition[1]}"
    elif name.startswith("china_drone_"):
        view = "aerial"
    elif name.startswith("china_motorbike_"):
        view = "motorbike"
    return Provenance(SOURCE_NAMES[prefix], country, view, group)


def read_class_config(path: Path) -> dict:
    """Accept the exact semantic class order and reject contradictory CRDDC tags."""
    with path.open(encoding="utf-8-sig") as stream:
        settings = yaml.safe_load(stream)
    names = settings.get("names") if isinstance(settings, dict) else None
    if isinstance(names, list):
        names = dict(enumerate(names))
    if (
        not isinstance(names, dict)
        or any(type(key) is not int for key in names)
        or set(names) != set(CLASS_NAMES)
    ):
        raise ValueError(f"{path.name} must define numeric class IDs 0 through 3.")
    if settings.get("nc", len(CLASS_NAMES)) != len(CLASS_NAMES):
        raise ValueError(f"{path.name} advertises a different class count.")
    codes = {class_id: code.lower() for code, class_id in RDD_CLASSES.items()}
    for class_id, expected in CLASS_NAMES.items():
        raw = names[class_id]
        alias = re.fullmatch(r"([a-z _-]+?)(?:\s*\((d\d+)\))?", str(raw).strip().lower())
        if alias is None:
            raise ValueError(f"Invalid class name for ID {class_id}: {raw!r}")
        label = re.sub(r"[ _-]+", "_", alias[1].strip())
        if label != expected or (alias[2] is not None and alias[2] != codes[class_id]):
            raise ValueError(f"Class ID {class_id}: expected {expected}, found {raw!r}.")
    return settings


def discover_layout(root: Path) -> tuple[Path, list[tuple[Path, Path, str]]]:
    """Locate the publisher descriptor and match its relative paths to the extracted tree."""
    descriptors = sorted(root.rglob("rdd_merged.yaml"))
    if len(descriptors) != 1:
        raise ValueError(f"Expected one rdd_merged.yaml under {root}; found {len(descriptors)}.")
    descriptor = descriptors[0]
    config = read_class_config(descriptor)
    relatives = {}
    for split in PUBLISHER_SPLITS:
        value = config.get(split)
        if not isinstance(value, str):
            raise ValueError(f"Missing relative publisher {split} image directory.")
        parts = PurePosixPath(value).parts
        if len(parts) < 2 or parts[0] != "images" or any(part in {".", ".."} for part in parts):
            raise ValueError(f"Unsupported publisher {split} directory: {value!r}")
        relatives[split] = Path(*parts)

    inventory = [
        folder for folder in root.rglob("images")
        if folder.is_dir() and any((folder.parent / relative).is_dir()
                                   for relative in relatives.values())
    ]
    if len(inventory) != 1:
        raise ValueError(f"Expected one extracted images tree under {root}; found {len(inventory)}.")
    base = inventory[0].parent
    return descriptor, [
        (base / relative, base / "labels" / Path(*relative.parts[1:]), split)
        for split, relative in relatives.items()
    ]
