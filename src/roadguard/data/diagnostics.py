"""Descriptive statistics and review-only bounding-box diagnostics."""

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from statistics import mean, median

from roadguard.data.dataset import CLASS_NAMES, Box, ImageRecord

@dataclass(frozen=True)
class BoxMeasurements:
    """Area and shape in both normalized and pixel coordinates."""

    normalized_area: float
    pixel_area: float
    pixel_width: float
    pixel_height: float
    aspect_ratio: float


def measure(box: Box, width: int, height: int) -> BoxMeasurements:
    """Convert YOLO geometry to image-relative area and pixel dimensions."""
    pixel_width, pixel_height = width * box.width, height * box.height
    return BoxMeasurements(
        box.width * box.height, pixel_width * pixel_height,
        pixel_width, pixel_height, pixel_width / pixel_height,
    )


def describe(values: Iterable[float]) -> dict[str, float | int | None]:
    """Summarize a numeric distribution, including an empty distribution."""
    data = list(values)
    return {
        "count": len(data),
        "minimum": min(data) if data else None,
        "mean": mean(data) if data else None,
        "median": median(data) if data else None,
        "maximum": max(data) if data else None,
    }


def _class_counts(rows: list[ImageRecord]) -> dict:
    instances = Counter(box.class_id for row in rows for box in row.boxes)
    images = Counter(class_id for row in rows for class_id in {box.class_id for box in row.boxes})
    minimum = min(instances[class_id] for class_id in CLASS_NAMES)
    return {
        "images": len(rows),
        "objects": instances.total(),
        "negatives": sum(not row.boxes for row in rows),
        "instances_per_class": {label: instances[cid] for cid, label in CLASS_NAMES.items()},
        "images_per_class": {label: images[cid] for cid, label in CLASS_NAMES.items()},
        "instance_share": {
            label: instances[cid] / instances.total() if instances else 0
            for cid, label in CLASS_NAMES.items()
        },
        "image_share": {
            label: images[cid] / len(rows) if rows else 0 for cid, label in CLASS_NAMES.items()
        },
        "largest_to_smallest_class_ratio": max(instances.values()) / minimum if minimum else None,
    }


def _group(rows: list[ImageRecord], attr: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[getattr(row.provenance, attr) or "unknown"].append(row)
    return {key: _class_counts(value) for key, value in sorted(groups.items())}


def summarize_records(rows: list[ImageRecord], *, geometry: bool = True) -> dict:
    """Summarize class balance, source/country counts and optionally geometry."""
    result = {
        **_class_counts(rows),
        "by_source": _group(rows, "source"),
        "by_country": _group(rows, "country"),
        "by_view": _group(rows, "view"),
    }
    if not geometry:
        return result
    boxes = defaultdict(list)
    for row in rows:
        for box in row.boxes:
            boxes[box.class_id].append(measure(box, row.width, row.height))
    result.update({
        "box_geometry": _geometry([box for measurements in boxes.values() for box in measurements]),
        "box_geometry_per_class": {
            label: _geometry(boxes[cid]) for cid, label in CLASS_NAMES.items()
        },
        "image_width_px": describe(row.width for row in rows),
        "image_height_px": describe(row.height for row in rows),
        "image_aspect_ratio": describe(row.width / row.height for row in rows),
        "image_resolution": dict(Counter(f"{row.width}x{row.height}" for row in rows)),
        "objects_per_image": dict(sorted(Counter(len(row.boxes) for row in rows).items())),
        "multi_object_images": sum(len(row.boxes) > 1 for row in rows),
        "multi_class_images": sum(len({box.class_id for box in row.boxes}) > 1 for row in rows),
    })
    return result


def _geometry(boxes: list[BoxMeasurements]) -> dict:
    return {
        key: describe(getattr(box, key) for box in boxes)
        for key in BoxMeasurements.__dataclass_fields__
    }


@dataclass(frozen=True)
class DiagnosticThresholds:
    """Review criteria used to flag labeled objects for inspection."""

    tiny_area_fraction: float
    tiny_pixel_area: float
    large_area_fraction: float
    extreme_aspect_ratio: float

    def __post_init__(self) -> None:
        if not (
            0 < self.tiny_area_fraction < self.large_area_fraction <= 1
            and self.tiny_pixel_area > 0
            and self.extreme_aspect_ratio > 1
        ):
            raise ValueError("Invalid bounding-box diagnostic thresholds.")

    def flags(self, values: BoxMeasurements) -> list[str]:
        """List diagnostic flags for one otherwise valid box."""
        return [
            label for label, matched in (
                ("tiny_normalized_area", values.normalized_area < self.tiny_area_fraction),
                ("tiny_pixel_area", values.pixel_area < self.tiny_pixel_area),
                ("large_area", values.normalized_area >= self.large_area_fraction),
                ("elongated", elongation(values) >= self.extreme_aspect_ratio),
            ) if matched
        ]


def elongation(values: BoxMeasurements) -> float:
    """Return longer side divided by shorter side."""
    return max(values.aspect_ratio, 1 / values.aspect_ratio)


def flagged_boxes(
    rows: list[ImageRecord], thresholds: DiagnosticThresholds
) -> Iterator[tuple[ImageRecord, int, Box, BoxMeasurements, list[str]]]:
    """Yield each flagged annotation once while preserving all labels."""
    for row in rows:
        for index, box in enumerate(row.boxes, 1):
            values = measure(box, row.width, row.height)
            flags = thresholds.flags(values)
            if flags:
                yield row, index, box, values, flags
