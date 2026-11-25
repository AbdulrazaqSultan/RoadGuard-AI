"""Render training-only plots and annotated inspection sheets."""

import random
from collections import defaultdict
from pathlib import Path

from roadguard.data.dataset import CLASS_NAMES, ImageRecord
from roadguard.data.diagnostics import (
    DiagnosticThresholds,
    elongation,
    flagged_boxes,
    measure,
)
from roadguard.data.statistics import sample_images


def _geometry_figure(rows: list[ImageRecord], statistics: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    boxes = [measure(box, row.width, row.height) for row in rows for box in row.boxes]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
    for axis, values, title, log in (
        (axes[0, 0], [row.width for row in rows], "Image width (px)", False),
        (axes[0, 1], [row.height for row in rows], "Image height (px)", False),
        (axes[0, 2], [row.width / row.height for row in rows], "Image width / height", False),
        (axes[1, 0], [box.normalized_area for box in boxes], "log10(box area fraction)", True),
        (axes[1, 1], [box.pixel_area for box in boxes], "log10(box area in px²)", True),
        (axes[1, 2], [box.aspect_ratio for box in boxes], "log10(box width / height)", True),
    ):
        if log:
            from math import log10

            values = [log10(value) for value in values]
        axis.hist(values, bins=35)
        axis.set_title(title)
        axis.set_ylabel("Count")
    fig.suptitle(f"Training geometry · {statistics['images']:,} images")
    fig.savefig(output / "geometry.png", dpi=150)
    plt.close(fig)


def _composition_figure(statistics: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for axis, key, title in (
        (axes[0, 0], "instances_per_class", "Annotations per class"),
        (axes[0, 1], "images_per_class", "Images per class"),
        (axes[1, 0], "objects_per_image", "Objects per image"),
    ):
        counts = statistics[key]
        axis.bar([str(key).replace("_", " ") for key in counts], list(counts.values()))
        axis.tick_params(axis="x", labelrotation=20)
        axis.set_title(title)
    sources = statistics["by_source"]
    axes[1, 1].barh(list(sources), [row["images"] for row in sources.values()])
    axes[1, 1].set_title("Training images per source")
    fig.savefig(output / "composition.png", dpi=150)
    plt.close(fig)


def _sheet(rows: list[ImageRecord], title: str, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, len(rows), figsize=(5.2 * len(rows), 4.6), squeeze=False)
    for axis, row in zip(axes[0], rows, strict=True):
        with Image.open(row.image) as image:
            image.thumbnail((1100, 1100))
            axis.imshow(image.convert("RGB"), extent=(0, row.width, row.height, 0))
        axis.set_axis_off()
        axis.set_title(
            f"{row.relative_image.stem[:58]}\n{row.provenance.source} · {len(row.boxes)} boxes",
            fontsize=8,
        )
        for box in row.boxes:
            x = (box.x - box.width / 2) * row.width
            y = (box.y - box.height / 2) * row.height
            color = colors[box.class_id % len(colors)]
            axis.add_patch(Rectangle(
                (x, y), box.width * row.width, box.height * row.height,
                fill=False, edgecolor=color, linewidth=1.5,
            ))
            axis.text(
                x, y, CLASS_NAMES[box.class_id].replace("_", " "), fontsize=6,
                va="bottom", color="white", bbox={"facecolor": color, "pad": 0.6, "alpha": 0.9},
                clip_on=True,
            )
    fig.suptitle(title.replace("_", " "))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_unified_figures(
    rows: list[ImageRecord], statistics: dict, output: Path, *,
    seed: int, count: int, thresholds: DiagnosticThresholds,
) -> list[Path]:
    """Save composition, geometry and representative sheets for training images."""
    output.mkdir(parents=True, exist_ok=True)
    _composition_figure(statistics, output)
    _geometry_figure(rows, statistics, output)
    paths = [output / "composition.png", output / "geometry.png"]
    ordered = sorted(rows, key=lambda row: row.relative_image)
    rng = random.Random(seed)
    categories = sample_images(ordered, count, seed)
    sources = defaultdict(list)
    for row in ordered:
        sources[row.provenance.source].append(row)
    for source, matches in sorted(sources.items()):
        categories[f"source_{source}"] = rng.sample(matches, min(count, len(matches)))
    multi = [row for row in ordered if len({box.class_id for box in row.boxes}) > 1]
    suspicious = {row.relative_image for row, *_ in flagged_boxes(ordered, thresholds)}
    flagged = [row for row in ordered if row.relative_image in suspicious]
    labeled = [row for row in ordered if row.boxes]
    categories.update({
        "multi_defect": rng.sample(multi, min(count, len(multi))),
        "most_objects": sorted(labeled, key=lambda row: -len(row.boxes))[:count],
        "smallest_normalized_boxes": sorted(labeled, key=lambda row: min(
            box.width * box.height for box in row.boxes
        ))[:count],
        "largest_normalized_boxes": sorted(labeled, key=lambda row: -max(
            box.width * box.height for box in row.boxes
        ))[:count],
        "extreme_aspect_ratios": sorted(labeled, key=lambda row: -max(
            elongation(measure(box, row.width, row.height)) for box in row.boxes
        ))[:count],
        "flagged_boxes": rng.sample(flagged, min(count, len(flagged))),
    })
    for name, matches in categories.items():
        if matches:
            path = output / f"{name}.png"
            _sheet(matches, name, path)
            paths.append(path)
    return paths
