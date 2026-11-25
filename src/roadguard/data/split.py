"""Deterministic held-out splits with optional route or capture-group isolation."""

import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from roadguard.data.dataset import SPLITS, ImageRecord


def read_groups(path: Path, records: list[ImageRecord]) -> dict[str, str]:
    """Read a complete source-relative image to capture group mapping."""
    groups = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["image", "group"]:
            raise ValueError("Groups CSV must have image,group headers.")
        for row in reader:
            image = (row["image"] or "").strip().replace("\\", "/")
            group = (row["group"] or "").strip()
            if not image or not group or image in groups or None in row:
                raise ValueError(f"Empty or duplicate image/group at CSV line {reader.line_num}.")
            groups[image] = group
    expected = {record.relative_image.as_posix() for record in records}
    if set(groups) != expected:
        raise ValueError(
            f"Groups CSV must cover every image exactly once. "
            f"Missing: {len(expected - groups.keys())}, unexpected: {len(groups.keys() - expected)}."
        )
    return groups


def _validate_policy(ratios: dict[str, float], seed: int) -> None:
    if set(ratios) != set(SPLITS) or not all(0 < value < 1 for value in ratios.values()):
        raise ValueError("Use positive train, val and test ratios below 1.")
    if not math.isclose(sum(ratios.values()), 1):
        raise ValueError("Split ratios must sum to 1.")
    if not 0 <= seed < 2**32:
        raise ValueError("Seed must be between 0 and 2**32 - 1.")


def assign_splits(
    records: list[ImageRecord],
    ratios: dict[str, float],
    seed: int,
    groups: dict[str, str] | None = None,
) -> dict[str, list[ImageRecord]]:
    """Preserve provided splits or assign reproducibly, keeping each group intact."""
    _validate_policy(ratios, seed)
    records = sorted(records, key=lambda record: record.relative_image.as_posix())
    result: dict[str, list[ImageRecord]] = {name: [] for name in SPLITS}
    units: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        key = record.relative_image.as_posix()
        units[groups[key] if groups is not None else key].append(record)
    if any(record.split is not None for record in records):
        for unit in units.values():
            split = unit[0].split
            if split not in result or any(record.split != split for record in unit):
                raise ValueError("Supplied splits are mixed or a capture group spans multiple splits.")
            result[split].extend(unit)
        if not all(result.values()):
            raise ValueError("Supplied train, validation and test splits must all be nonempty.")
        return result

    if len(units) < len(SPLITS):
        raise ValueError("At least three images or independent capture groups are required.")
    ordered = list(units.values())
    random.Random(seed).shuffle(ordered)
    ordered.sort(key=len, reverse=True)
    targets = {name: len(records) * ratios[name] for name in SPLITS}
    for index, unit in enumerate(ordered):
        empty = [name for name in SPLITS if not result[name]]
        choices = empty if len(empty) == len(ordered) - index else SPLITS
        split = max(choices, key=lambda name: targets[name] - len(result[name]))
        result[split].extend(unit)
    return result


def _strata(unit: list[ImageRecord]) -> Counter[tuple[str, ...]]:
    counts: Counter[tuple[str, ...]] = Counter()
    for row in unit:
        domain = row.provenance
        counts[("source", domain.source)] += 1
        if domain.country or domain.view:
            counts[("domain", domain.source, domain.country or "", domain.view or "")] += 1
        for class_id in {box.class_id for box in row.boxes}:
            counts[("class", str(class_id))] += 1
            counts[("source_class", domain.source, str(class_id))] += 1
    return counts


def assign_source_splits(
    records: list[ImageRecord],
    ratios: dict[str, float],
    seed: int,
    groups: dict[str, str] | None = None,
) -> dict[str, list[ImageRecord]]:
    """Repartition the full upstream pool, balancing domains and class coverage by group."""
    _validate_policy(ratios, seed)
    if len({row.pixel_hash for row in records}) != len(records):
        raise ValueError("Review exact pixel duplicates before repartitioning.")
    units: dict[tuple[str, str], list[ImageRecord]] = defaultdict(list)
    for row in sorted(records, key=lambda record: record.relative_image.as_posix()):
        path = row.relative_image.as_posix()
        if groups is not None:
            key = ("group", groups[path])
        elif row.provenance.capture_group:
            key = ("group", row.provenance.capture_group)
        else:
            key = ("image", path)
        units[key].append(row)
    if groups is not None:
        parents = {}
        for row in records:
            parent = row.provenance.capture_group
            if parent is None:
                continue
            group = groups[row.relative_image.as_posix()]
            if parents.setdefault(parent, group) != group:
                raise ValueError(f"Groups CSV separates related images from {parent}.")
    if len(units) < len(SPLITS):
        raise ValueError("At least three independent source groups or images are required.")

    ordered = [(rows, _strata(rows)) for rows in units.values()]
    random.Random(seed).shuffle(ordered)
    totals: Counter[tuple[str, ...]] = Counter()
    for _, counts in ordered:
        totals.update(counts)
    ordered.sort(key=lambda unit: (
        min((totals[key] for key in unit[1] if key[0] == "class"), default=len(records)),
        -len(unit[0]),
    ))
    result: dict[str, list[ImageRecord]] = {split: [] for split in SPLITS}
    seen: dict[str, Counter[tuple[str, ...]]] = {split: Counter() for split in SPLITS}
    quotas = {split: len(records) * ratios[split] for split in SPLITS}
    for index, (rows, counts) in enumerate(ordered):
        empty = [split for split in SPLITS if not result[split]]
        choices = empty if len(empty) == len(ordered) - index else list(SPLITS)
        fitting = [s for s in choices if len(result[s]) + len(rows) <= math.ceil(quotas[s])]

        scores = {}
        for split in fitting or choices:
            balance = 1 - len(result[split]) / quotas[split]
            diversity = sum(
                count * (1 - seen[split][key] / (totals[key] * ratios[split]))
                for key, count in counts.items()
            )
            scores[split] = balance + diversity / max(1, counts.total())

        selected = max(scores, key=scores.__getitem__)
        result[selected].extend(rows)
        seen[selected].update(counts)
    return result
