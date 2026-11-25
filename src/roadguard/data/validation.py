"""Inspect labeled datasets without changing source files."""

import hashlib
import logging
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml
from PIL import Image

from roadguard.data.annotations import read_voc, read_yolo
from roadguard.data.dataset import CLASS_NAMES, IMAGE_EXTENSIONS, SPLITS, ImageRecord, Provenance
from roadguard.data.unified import discover_layout, infer_provenance

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Issue:
    """A finding with a machine-readable code and affected files."""

    level: str
    code: str
    paths: tuple[str, ...]
    detail: str


@dataclass
class ValidationReport:
    """Records fit for analysis and all findings from inspection."""

    root: Path
    images_found: int = 0
    labels_found: int = 0
    layout: dict = field(default_factory=dict)
    records: list[ImageRecord] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """A dataset is usable when at least one pair exists and there are no errors."""
        return bool(self.records) and not any(issue.level == "error" for issue in self.issues)

    def add(
        self, code: str, paths: Iterable[Path], detail: str, *, level: str = "error"
    ) -> None:
        """Add a finding with paths relative to the input root."""
        self.issues.append(
            Issue(level, code, tuple(p.relative_to(self.root).as_posix() for p in paths), detail)
        )

    def summary(self) -> dict:
        """Return counts and file-level findings for the JSON report."""
        summary = {
            "root": str(self.root),
            "valid": self.valid,
            "images_found": self.images_found,
            "labels_found": self.labels_found,
            "valid_pairs": len(self.records),
            "images_without_annotations": sum(not record.boxes for record in self.records),
            "issue_counts": dict(Counter(issue.code for issue in self.issues)),
            "issues": [asdict(issue) for issue in self.issues],
        }
        if self.layout:
            summary["layout"] = self.layout
            summary["class_coverage"] = {
                split: {
                    CLASS_NAMES[class_id]: count
                    for class_id, count in sorted(Counter(
                        box.class_id for row in self.records if row.split == split
                        for box in row.boxes
                    ).items())
                }
                for split in self.layout["partitions"]
            }
        return summary


def sha256(path: Path) -> str:
    """Hash the original file bytes without loading the entire file into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _files(folder: Path, extensions: set[str]) -> dict[str, list[Path]]:
    entries: dict[str, list[Path]] = {}
    for file in sorted(folder.rglob("*")):
        if file.is_file() and file.suffix.lower() in extensions:
            key = file.relative_to(folder).with_suffix("").as_posix().casefold()
            entries.setdefault(key, []).append(file)
    return entries


def _image_info(path: Path) -> tuple[int, int, str]:
    with Image.open(path) as image:
        if image.format == "JPEG":
            with path.open("rb") as stream:
                stream.seek(-2, 2)
                if stream.read() != b"\xff\xd9":
                    raise ValueError("JPEG must end with a valid end-of-image marker")
        image.verify()
    with Image.open(path) as image:
        if image.getexif().get(274, 1) != 1:
            raise ValueError("normalize EXIF orientation and box coordinates before import")
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("multi-frame images are unsupported")
        rgb = image.convert("RGB")
        width, height = rgb.size
        digest = hashlib.sha256(f"{width}x{height}\0".encode())
        digest.update(rgb.tobytes())
        return width, height, digest.hexdigest()


def _partitions(root: Path, source_format: str) -> list[tuple[Path, Path, str | None]]:
    if source_format == "voc":
        # RDD2022 uses country (and sometimes acquisition platform) subdirectories.
        folders = [root] if root.name.casefold() == "train" else sorted(root.rglob("train"))
        return [(folder / "images", folder / "annotations" / "xmls", None)
                for folder in folders if folder.is_dir()]
    if source_format == "yolo":
        return [(root / "images", root / "labels", None)]
    if source_format == "yolo_split":
        return [(root / "images" / s, root / "labels" / s, s) for s in SPLITS]
    raise ValueError(f"Unknown dataset format: {source_format}")


def _inspect_partition(
    report: ValidationReport,
    images_dir: Path,
    labels_dir: Path,
    split: str | None,
    *,
    voc: bool,
    names: dict[str, tuple[Path, str | None]],
    pixels: dict[str, tuple[Path, str | None]],
) -> None:
    if not images_dir.is_dir() or not labels_dir.is_dir():
        report.add("missing_directory", (images_dir, labels_dir), "Expected images and labels.")
    images = _files(images_dir, IMAGE_EXTENSIONS)
    labels = _files(labels_dir, {".xml" if voc else ".txt"})
    report.images_found += sum(map(len, images.values()))
    report.labels_found += sum(map(len, labels.values()))
    if not images:
        report.add("no_images", (images_dir,), "No supported image files were found.")
    keys = sorted(images.keys() | labels.keys())
    for index, key in enumerate(keys, 1):
        if index % 2000 == 0:
            LOGGER.info("Validated %s / %s pairs in %s", f"{index:,}", f"{len(keys):,}", images_dir)
        image_paths = images.get(key, [])
        label_paths = labels.get(key, [])
        if not image_paths:
            report.add("missing_image", label_paths, "Label has no matching image.")
        if not label_paths:
            report.add("missing_label", image_paths, "Image has no matching label.")
        if len(label_paths) > 1:
            report.add("duplicate_label", label_paths, "Multiple labels match the same relative stem.")
        metadata = []
        for image in image_paths:
            _check_duplicate(report, names, image.stem.casefold(), image, split, "duplicate_name")
            try:
                width, height, digest = _image_info(image)
                image_hash = sha256(image)
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
                report.add("invalid_image", (image,), str(error))
                continue
            _check_duplicate(report, pixels, digest, image, split, "duplicate_image")
            metadata.append((image, width, height, digest, image_hash))

        if len(image_paths) != 1 or len(metadata) != 1 or len(label_paths) != 1:
            continue
        image, width, height, digest, image_hash = metadata[0]
        label = label_paths[0]
        try:
            if voc:
                boxes, errors, skipped = read_voc(label, width, height)
                if skipped:
                    report.add(
                        "non_target_classes", (label,), ", ".join(sorted(skipped)), level="warning"
                    )
            else:
                boxes, errors = read_yolo(label)
        except (ValueError, OSError, ET.ParseError) as error:
            report.add("invalid_annotation", (label,), str(error))
            continue
        if errors:
            for error in errors:
                report.add("invalid_annotation", (label,), error)
            continue
        if not boxes:
            code = "empty_annotation_file" if not voc else "no_target_annotations"
            report.add(code, (image, label), "Zero target boxes.", level="warning")
        report.records.append(
            ImageRecord(
                image, label, image.relative_to(report.root), split, width, height,
                image_hash, digest, boxes,
                infer_provenance(image.stem) if report.layout else Provenance(),
            )
        )


def _check_duplicate(
    report: ValidationReport,
    seen: dict[str, tuple[Path, str | None]],
    key: str,
    path: Path,
    split: str | None,
    code: str,
) -> None:
    if key in seen:
        original, original_split = seen[key]
        report.add(code, (original, path), "Repeated image stem or exact decoded pixels.")
        if original_split != split:
            report.add("split_leakage", (original, path), f"{original_split} and {split} overlap.")
    else:
        seen[key] = (path, split)


def validate_dataset(root: Path, source_format: str) -> ValidationReport:
    """Check pairing, complete image decoding, annotation geometry and split overlap."""
    root = root.resolve()
    report = ValidationReport(root)
    if source_format == "unified":
        try:
            descriptor, partitions = discover_layout(root)
        except (OSError, ValueError, yaml.YAMLError) as error:
            report.add("invalid_unified_layout", (root,), str(error))
            return report
        report.layout = {
            "descriptor": descriptor.relative_to(root).as_posix(),
            "partitions": {
                split: {"images": images.relative_to(root).as_posix(),
                        "labels": labels.relative_to(root).as_posix()}
                for images, labels, split in partitions
            },
        }
    else:
        partitions = _partitions(root, source_format)
    if not partitions:
        report.add("missing_layout", (root,), "Expected RDD2022 train/images and train/annotations/xmls.")
    names: dict[str, tuple[Path, str | None]] = {}
    pixels: dict[str, tuple[Path, str | None]] = {}
    for images, labels, split in partitions:
        _inspect_partition(report, images, labels, split, voc=source_format == "voc",
                           names=names, pixels=pixels)
    return report
