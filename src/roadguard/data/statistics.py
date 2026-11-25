"""Machine-readable statistics computed from validated image-label pairs."""

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from roadguard.data.dataset import CLASS_NAMES, ImageRecord


@dataclass
class DatasetStatistics:
    """Aggregate counts and raw distributions backing the EDA charts."""

    total_images: int
    total_annotations: int
    annotations_per_class: dict[str, int]
    images_per_class: dict[str, int]
    annotation_share_per_class: dict[str, float]
    largest_to_smallest_class_ratio: float | None
    objects_per_image: dict[int, int]
    box_area_fraction: list[float]
    box_aspect_ratio: list[float]
    image_resolution: dict[str, int]
    empty_images: int
    multi_object_images: int
    multi_defect_images: int
    samples_per_class: dict[str, list[str]]


def sample_images(
    records: list[ImageRecord], count: int, seed: int
) -> dict[str, list[ImageRecord]]:
    """Choose reproducible representative images for every available class."""
    if count <= 0:
        raise ValueError("sample_images_per_class must be positive.")
    rng = random.Random(seed)
    ordered = sorted(records, key=lambda record: record.relative_image)
    samples = {}
    for class_id, name in CLASS_NAMES.items():
        matches = [record for record in ordered if any(box.class_id == class_id for box in record.boxes)]
        samples[name] = rng.sample(matches, min(len(matches), count))
    return samples


def summarize(
    records: list[ImageRecord], samples: dict[str, list[ImageRecord]]
) -> DatasetStatistics:
    """Summarize class imbalance, image sizes, objects per image and box shapes."""
    boxes = [box for record in records for box in record.boxes]
    annotation_counts = Counter(box.class_id for box in boxes)
    image_classes = [{box.class_id for box in record.boxes} for record in records]
    image_counts = Counter(class_id for classes in image_classes for class_id in classes)
    counts = {name: annotation_counts[class_id] for class_id, name in CLASS_NAMES.items()}
    object_counts = Counter(len(record.boxes) for record in records)
    minimum = min(counts.values())
    return DatasetStatistics(
        total_images=len(records),
        total_annotations=len(boxes),
        annotations_per_class=counts,
        images_per_class={name: image_counts[cid] for cid, name in CLASS_NAMES.items()},
        annotation_share_per_class={
            name: count / len(boxes) if boxes else 0.0 for name, count in counts.items()
        },
        largest_to_smallest_class_ratio=max(counts.values()) / minimum if minimum else None,
        objects_per_image=dict(sorted(object_counts.items())),
        box_area_fraction=[box.width * box.height for box in boxes],
        box_aspect_ratio=[
            (box.width * record.width) / (box.height * record.height)
            for record in records for box in record.boxes
        ],
        image_resolution=dict(Counter(f"{row.width}x{row.height}" for row in records)),
        empty_images=object_counts[0],
        multi_object_images=sum(count for size, count in object_counts.items() if size > 1),
        multi_defect_images=sum(len(classes) > 1 for classes in image_classes),
        samples_per_class={
            name: [record.relative_image.as_posix() for record in rows] for name, rows in samples.items()
        },
    )


def save_charts(
    statistics: DatasetStatistics, samples: dict[str, list[ImageRecord]], output: Path
) -> None:
    """Save distribution charts and annotated sample images using Matplotlib defaults."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    output.mkdir(parents=True, exist_ok=True)
    fig, panels = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for ax, counts, title in (
        (panels[0, 0], statistics.annotations_per_class, "Annotations per class"),
        (panels[0, 1], statistics.images_per_class, "Images per class"),
        (panels[0, 2], statistics.objects_per_image, "Objects per image"),
    ):
        ax.bar([str(key).replace("_", " ") for key in counts], counts.values())
        ax.tick_params(axis="x", rotation=25)
        ax.set_title(title)
    for ax, values, title in (
        (panels[1, 0], statistics.box_area_fraction, "Box area fraction"),
        (panels[1, 1], statistics.box_aspect_ratio, "Box width / height in pixels"),
    ):
        ax.hist(values, bins=30)
        ax.set_title(title)
    resolutions = Counter(statistics.image_resolution).most_common(10)
    panels[1, 2].barh([name for name, _ in resolutions], [n for _, n in resolutions])
    panels[1, 2].invert_yaxis()
    panels[1, 2].set_title("Most common image resolutions")
    fig.savefig(output / "distributions.png", dpi=150)
    plt.close(fig)

    for class_id, name in CLASS_NAMES.items():
        rows = samples[name]
        if not rows:
            (output / f"{name}.png").unlink(missing_ok=True)
            continue
        fig, axes = plt.subplots(1, len(rows), figsize=(4 * len(rows), 4), squeeze=False)
        for ax, row in zip(axes[0], rows, strict=True):
            with Image.open(row.image) as image:
                ax.imshow(image.convert("RGB"))
            ax.set_axis_off()
            ax.set_title(row.image.stem, fontsize=8, wrap=True)
            for box in row.boxes:
                if box.class_id == class_id:
                    ax.add_patch(Rectangle(
                        ((box.x - box.width / 2) * row.width, (box.y - box.height / 2) * row.height),
                        box.width * row.width,
                        box.height * row.height,
                        fill=False,
                        linewidth=2,
                    ))
        fig.suptitle(name.replace("_", " ").title())
        fig.savefig(output / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
