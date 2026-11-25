"""Parse YOLO labels and convert Pascal VOC road damage boxes."""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

from roadguard.data.dataset import CLASS_NAMES, RDD_CLASSES, Box

# Allow floating-point rounding at the edge of an otherwise in-frame box.
EDGE_TOLERANCE = 1e-6


def _validate_box(box: Box) -> None:
    x, y, width, height = box.coordinates
    if box.class_id not in CLASS_NAMES:
        raise ValueError(f"class ID {box.class_id} is outside 0–{len(CLASS_NAMES) - 1}")
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in box.coordinates):
        raise ValueError("box values must be finite and normalized to [0, 1]")
    if width == 0 or height == 0:
        raise ValueError("zero-area bounding box")
    edges = (x - width / 2, y - height / 2, x + width / 2, y + height / 2)
    if min(edges) < -EDGE_TOLERANCE or max(edges) > 1 + EDGE_TOLERANCE:
        raise ValueError("box extends outside the image")


def read_yolo(path: Path) -> tuple[tuple[Box, ...], list[str]]:
    """Return valid boxes and line-specific errors from a UTF-8 label file."""
    boxes = []
    errors = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            class_id, x, y, width, height = line.split()
            box = Box(int(class_id), float(x), float(y), float(width), float(height))
            _validate_box(box)
        except ValueError as error:
            errors.append(f"Line {line_number}: {error}")
        else:
            boxes.append(box)
    return tuple(boxes), errors


def read_voc(
    path: Path, width: int, height: int
) -> tuple[tuple[Box, ...], list[str], set[str]]:
    """Map the four RDD codes and report errors and skipped non-target codes."""
    document = ET.parse(path).getroot()
    if document.tag != "annotation":
        raise ValueError("expected a Pascal VOC annotation element")
    xml_size = (int(document.findtext("size/width", "")), int(document.findtext("size/height", "")))
    if xml_size != (width, height):
        raise ValueError(f"XML size {xml_size} differs from image size {(width, height)}")
    boxes = []
    errors = []
    skipped = set()
    for index, item in enumerate(document.findall("object"), 1):
        try:
            code = item.findtext("name", "").strip().upper()
            if not code:
                raise ValueError("missing object class")
            if code not in RDD_CLASSES:
                skipped.add(code)
                continue
            left, top, right, bottom = (
                float(item.findtext(f"bndbox/{name}", ""))
                for name in ("xmin", "ymin", "xmax", "ymax")
            )
            box = Box(
                RDD_CLASSES[code],
                (left + right) / (2 * width),
                (top + bottom) / (2 * height),
                (right - left) / width,
                (bottom - top) / height,
            )
            _validate_box(box)
        except ValueError as error:
            errors.append(f"Object {index}: {error}")
        else:
            boxes.append(box)
    return tuple(boxes), errors, skipped
