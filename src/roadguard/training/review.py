"""Archive a small, reproducible validation-only prediction and error sample."""

import logging
import random
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from roadguard.data.annotations import read_yolo
from roadguard.data.dataset import CLASS_NAMES, Box
from roadguard.evaluation.box_matching import Prediction, match_boxes
from roadguard.utils.paths import write_json

LOGGER = logging.getLogger(__name__)
SAMPLE_LIMIT = 16
MATCH_IOU = 0.5


def select_examples(manifest: dict, seed: int, count: int = SAMPLE_LIMIT) -> list[dict]:
    """Cover validation sources and classes where possible, then fill from a seeded sample."""
    rows = [entry for entry in manifest["files"] if entry["split"] == "val"]
    random.Random(seed).shuffle(rows)
    chosen = {}
    for field in ("source", "class_ids"):
        levels = {}
        for row in rows:
            values = row.get(field, []) if field == "class_ids" else [row.get(field, "unknown")]
            for value in values:
                levels.setdefault(value, row)
        for row in levels.values():
            chosen[row["image"]] = row
    for row in rows:
        if len(chosen) >= count:
            break
        chosen[row["image"]] = row
    return list(chosen.values())[:count]


def draw_comparison(
    image_path: Path,
    labels: tuple[Box, ...],
    predictions: list[Prediction],
    output: Path,
    title: str,
    *,
    focus: tuple[int | None, int | None] = (None, None),
) -> None:
    """Draw aligned annotation and prediction panels with optional highlighted boxes."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, panels = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    with Image.open(image_path) as image:
        image.thumbnail((1100, 1100))
        arr = image.convert("RGB")
    for axis, items, heading, selected in (
        (panels[0], [(box, None) for box in labels], "Ground truth", focus[0]),
        (panels[1], [(p.box, p.confidence) for p in predictions], "Predictions", focus[1]),
    ):
        axis.imshow(arr, extent=(0, 1, 1, 0), aspect=arr.height / arr.width)
        axis.set_axis_off()
        axis.set_title(heading)
        indices = sorted(range(len(items)), key=lambda index: index == selected)
        for index in indices:
            box, score = items[index]
            x, y = box.x - box.width / 2, box.y - box.height / 2
            color = colors[box.class_id % len(colors)]
            axis.add_patch(
                Rectangle(
                    (x, y),
                    box.width,
                    box.height,
                    linewidth=3 if index == selected else 2,
                    edgecolor=color,
                    fill=False,
                    alpha=1 if selected is None or index == selected else 0.35,
                )
            )
            label = CLASS_NAMES[box.class_id].replace("_", " ")
            if score is not None:
                label += f" {score:.2f}"
            if selected is not None and index != selected:
                continue
            axis.text(
                x,
                y,
                label,
                va="bottom",
                fontsize=7,
                color="white",
                bbox={"facecolor": color, "alpha": 0.9, "pad": 0.7},
                clip_on=True,
            )
    fig.suptitle(title)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def archive_review(
    model: Any,
    snapshot: Path,
    manifest: dict,
    run: Path,
    settings: dict,
) -> dict:
    """Use the loaded best detector to write sample images, error counts and predictions."""
    model_settings = settings["model"]
    training = settings["training"]
    confidence = model_settings["confidence_threshold"]
    selected = select_examples(manifest, training["seed"])
    if not selected:
        raise ValueError("Validation review requires at least one validation image.")
    directory = run / "validation_review"
    directory.mkdir(exist_ok=True)
    rows = []
    per_class: dict[str, Counter] = defaultdict(Counter)
    confusions: Counter[str] = Counter()
    total: Counter[str] = Counter()
    for index, entry in enumerate(selected, 1):
        image_path = snapshot / entry["image"]
        labels, errors = read_yolo(snapshot / entry["label"])
        if errors:
            raise ValueError(f"Prepared validation label has errors: {errors[0]}")
        result = model.predict(
            source=str(image_path),
            device=0,
            imgsz=training["image_size"],
            conf=confidence,
            iou=model_settings["iou_threshold"],
            verbose=False,
        )[0].boxes
        boxes = result.xywhn.cpu().tolist()
        predictions = [
            Prediction(Box(int(cid), *coords), float(score))
            for cid, coords, score in zip(
                result.cls.cpu().tolist(),
                boxes,
                result.conf.cpu().tolist(),
                strict=True,
            )
        ]
        outcome = match_boxes(labels, predictions, iou_threshold=MATCH_IOU)
        for key in ("true_positives", "false_positives", "false_negatives"):
            total[key] += outcome[key]
        for label, values in outcome["per_class"].items():
            per_class[label].update(values)
        confusions.update(outcome["class_confusions"])
        original = entry.get("source_path", entry.get("source", entry["image"]))
        plot = directory / f"{index:02d}.png"
        draw_comparison(image_path, labels, predictions, plot, Path(original).name)
        rows.append(
            {
                "image": entry["image"],
                "original_image": original,
                "source": entry.get("source", "unknown") if "source_path" in entry else "unknown",
                "figure": plot.relative_to(run).as_posix(),
                "labels": [asdict(label) for label in labels],
                "predictions": [asdict(prediction) for prediction in predictions],
                **outcome,
            }
        )
    review = {
        "scope": "validation_sample",
        "sample_count": len(rows),
        "confidence_threshold": confidence,
        "match_iou_threshold": MATCH_IOU,
        "totals": dict(total),
        "per_class": {label: dict(values) for label, values in per_class.items()},
        "class_confusions": dict(confusions),
        "samples": rows,
    }
    write_json(directory / "review.json", review)
    LOGGER.info("Archived %d validation prediction examples under %s", len(rows), directory)
    return review
