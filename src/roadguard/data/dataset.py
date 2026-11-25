"""The four-class road damage taxonomy and portable dataset records."""

from dataclasses import dataclass
from pathlib import Path

CLASS_NAMES = {
    0: "longitudinal_crack",
    1: "transverse_crack",
    2: "alligator_crack",
    3: "pothole",
}
RDD_CLASSES = dict(zip(("D00", "D10", "D20", "D40"), CLASS_NAMES, strict=True))
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Box:
    """A YOLO box in normalized center and size coordinates."""

    class_id: int
    x: float
    y: float
    width: float
    height: float

    @property
    def coordinates(self) -> tuple[float, float, float, float]:
        """Return center x, center y, width and height."""
        return self.x, self.y, self.width, self.height

    def yolo_line(self) -> str:
        """Serialize the box without rounding away small positive widths."""
        return " ".join(map(str, (self.class_id, *self.coordinates)))


@dataclass(frozen=True)
class Provenance:
    """Source metadata with a group only when a common parent is identifiable."""

    source: str = "unknown"
    country: str | None = None
    view: str | None = None
    capture_group: str | None = None


@dataclass(frozen=True)
class ImageRecord:
    """One valid image and its annotations, including explicitly empty labels."""

    image: Path
    label: Path
    relative_image: Path
    split: str | None
    width: int
    height: int
    image_hash: str
    pixel_hash: str
    boxes: tuple[Box, ...]
    provenance: Provenance = Provenance()
