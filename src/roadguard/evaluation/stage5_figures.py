"""Render the validation diagnostics and reproducible visual review sample."""

import random
from collections import Counter
from pathlib import Path

from roadguard.evaluation.stage5 import CODE
from roadguard.evaluation.stage5_errors import ReviewedImage
from roadguard.training.review import draw_comparison


def plot_diagnostics(
    run: Path,
    classes: list[dict],
    confusion: dict,
    sizes: list[dict],
    confidence: list[dict] | None = None,
    *,
    partition: str = "validation",
) -> None:
    """Plot class performance, confusion, size and optional confidence statistics."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import PercentFormatter

    out = run / "figures"
    out.mkdir(exist_ok=True)

    def save(figure, name: str) -> None:
        figure.savefig(out / name, dpi=165, facecolor="white")
        plt.close(figure)

    codes = [row["class"] for row in classes]
    x = np.arange(len(codes))
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for offset, key, color, title in (
        (-0.23, "ap50_95", "#236c92", "AP50–95"),
        (0, "precision", "#59a29a", "Precision"),
        (0.23, "recall", "#c78c53", "Recall"),
    ):
        left.bar(x + offset, [row[key] for row in classes], 0.22, label=title, color=color)
    left.set(title=f"Ultralytics {partition} metrics", xticks=x, xticklabels=codes, ylim=(0, 1))
    left.yaxis.set_major_formatter(PercentFormatter(1))
    left.legend(frameon=False, loc="upper left")
    for offset, key, color, label in (
        (-0.16, "false_positives", "#ad453c", "False positives"),
        (0.16, "false_negatives", "#d59450", "False negatives"),
    ):
        bars = right.bar(x + offset, [row[key] for row in classes], 0.31, color=color, label=label)
        right.bar_label(bars, padding=2, fontsize=8)
    right.set(title="Errors at confidence ≥ 0.25", xticks=x, xticklabels=codes, ylabel="Boxes")
    right.margins(y=0.19)
    right.legend(frameon=False, loc="upper right")
    save(fig, "per_class_performance.png")

    names = list(confusion)
    matrix = np.array([[confusion[a][b] for b in names] for a in names])
    sums = matrix.sum(axis=1, keepdims=True)
    fractions = np.divide(matrix, sums, out=np.zeros_like(matrix, dtype=float), where=sums != 0)
    fig, axis = plt.subplots(figsize=(8.0, 6), layout="constrained")
    image = axis.imshow(fractions, cmap="Blues", vmin=0, vmax=1)
    for i, j in np.ndindex(fractions.shape):
        value = fractions[i, j]
        color = "white" if value > 0.55 else "#202733"
        axis.text(j, i, f"{matrix[i, j]:,}\n{value:.1%}", ha="center", va="center", color=color)
    axis.set(
        title=f"{partition.title()} confusion matrix at confidence ≥ 0.25",
        xlabel="Prediction",
        ylabel="Ground truth",
        xticks=range(len(names)),
        yticks=range(len(names)),
        xticklabels=names,
        yticklabels=names,
    )
    fig.colorbar(image, ax=axis, shrink=0.75, format=PercentFormatter(1), label="Share of row")
    save(fig, "confusion_matrix.png")

    all_sizes = [row for row in sizes if row["class"] == "All"]
    fig, axis = plt.subplots(figsize=(7.8, 4.4), layout="constrained")
    names = [f"{row['size'].title()}\nn={row['annotations']:,}" for row in all_sizes]
    rate = [row["detection_rate"] for row in all_sizes]
    fn_rate = [row["false_negative_rate"] for row in all_sizes]
    detected = axis.bar(names, rate, 0.51, label="Detected", color="#236c92")
    missed = axis.bar(names, fn_rate, 0.51, bottom=rate, label="Missed", color="#d59450")
    axis.bar_label(
        detected, labels=[f"{value:.1%}" for value in rate], label_type="center", color="white"
    )
    axis.bar_label(missed, labels=[f"{value:.1%}" for value in fn_rate], label_type="center")
    axis.set(title="Detection by relative bounding-box area", ylim=(0, 1.13))
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.legend(frameon=False, loc="upper center", ncol=2)
    save(fig, "object_size.png")

    if confidence is None:
        return
    fig, axis = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
    x = np.arange(len(confidence))
    names = [
        f"{row['from_inclusive']:.2f}–{row['to_exclusive_except_1']:.2f}" for row in confidence
    ]
    for offset, key, color, label in (
        (-0.18, "true_positive", "#236c92", "Correct"),
        (0.18, "false_positive", "#ad453c", "False positives"),
    ):
        axis.bar(x + offset, [row[key] for row in confidence], 0.35, label=label, color=color)
    axis.axvline(2.5, color="#525760", linestyle="--", linewidth=1, label="Operating threshold")
    axis.set(
        title="Score distribution at the 0.001 evaluation floor",
        xlabel="Confidence bin",
        ylabel="Predictions (log scale)",
        yscale="log",
        xticks=x,
        xticklabels=names,
    )
    axis.legend(frameon=False)
    save(fig, "confidence_distribution.png")


def rank_hard_examples(images: list[ReviewedImage]) -> list[dict]:
    """Rank affected scenes by 2×FN + FP, with explicit severity tie-breaks."""
    rows = []
    for image in images:
        active = [row for row in image.rows if row["above_operating_confidence"]]
        counts = Counter(row["status"] for row in active)
        fps = [row["confidence"] for row in active if row["status"] != "true_positive"]
        if not image.misses and not fps:
            continue
        rows.append(
            {
                "image": image.image,
                "source": image.source,
                "score": 2 * len(image.misses) + len(fps),
                "false_negatives": len(image.misses),
                "false_positives": len(fps),
                "highest_confidence_fp": max(fps, default=None),
                "largest_missed_relative_area": max(
                    (row["relative_area"] for row in image.misses), default=None
                ),
                "lowest_tp_iou": min(
                    (row["iou"] for row in active if row["status"] == "true_positive"), default=None
                ),
                "class_confusions": counts["class_confusion"],
                "localization_errors": counts["localization_error"],
                "distinct_annotation_classes": len({label.class_id for label in image.labels}),
                "operating_predictions": len(active),
            }
        )
    rows.sort(
        key=lambda row: (
            -row["score"],
            -(row["highest_confidence_fp"] or 0),
            -(row["largest_missed_relative_area"] or 0),
            -row["distinct_annotation_classes"],
            row["image"],
        )
    )
    return [{"rank": i, **row} for i, row in enumerate(rows, 1)]


def select_representative(rows: list[dict], score, class_key: str, limit: int = 8) -> list[dict]:
    """Prioritize severity, covering classes and distinct images before filling."""
    ordered = sorted(rows, key=lambda row: (-score(row), row["image"]))
    picked = {}
    for code in CODE.values():
        if len(picked) >= limit:
            break
        match = next(
            (row for row in ordered if row[class_key] == code and row["image"] not in picked),
            None,
        )
        if match is not None:
            picked[match["image"]] = match
    for row in ordered:
        if len(picked) >= limit:
            break
        picked.setdefault(row["image"], row)
    return list(picked.values())


def make_galleries(
    snapshot: Path,
    run: Path,
    images: list[ReviewedImage],
    ranking: list[dict],
) -> dict[str, list[dict]]:
    """Draw distinct validation examples for each requested diagnostic category."""
    active = [row for image in images for row in image.rows if row["above_operating_confidence"]]
    misses = [row for image in images for row in image.misses]
    tp = [row for row in active if row["status"] == "true_positive"]
    fp = [row for row in active if row["status"] != "true_positive"]
    candidates = {
        "true_positives": select_representative(
            tp, lambda row: row["iou"] * row["confidence"], "predicted_class"
        ),
        "false_positives": select_representative(
            fp, lambda row: row["confidence"], "predicted_class"
        ),
        "false_negatives": select_representative(
            misses, lambda row: row["relative_area"], "ground_truth_class"
        ),
        "localization_errors": select_representative(
            [row for row in fp if row["status"] == "localization_error"],
            lambda row: row["confidence"] * (1 - row["iou"]),
            "predicted_class",
        ),
        "class_confusions": select_representative(
            [row for row in fp if row["status"] == "class_confusion"],
            lambda row: row["confidence"] * row["iou"],
            "ground_truth_class",
        ),
        "difficult_scenes": ranking[:8],
    }
    return draw_gallery(snapshot, run, images, candidates)


def draw_gallery(
    snapshot: Path,
    run: Path,
    images: list[ReviewedImage],
    candidates: dict[str, list[dict]],
) -> dict[str, list[dict]]:
    """Render selected examples with the shared comparison and focus overlays."""
    lookup = {image.image: image for image in images}
    gallery = {}
    for category, examples in candidates.items():
        directory = run / "galleries" / category
        directory.mkdir(parents=True, exist_ok=True)
        gallery[category] = []
        for number, example in enumerate(examples, 1):
            image = lookup[example["image"]]
            operating = {
                row["prediction_index"]: image.predictions[row["prediction_index"]]
                for row in image.rows
                if row["above_operating_confidence"]
            }
            pred_index = example.get("prediction_index")
            label_index = example.get("label_index")
            focus = (
                label_index,
                list(operating).index(pred_index) if pred_index in operating else None,
            )
            tokens = [category.replace("_", " ").title(), image.source]
            if "ground_truth_class" in example and example["ground_truth_class"]:
                tokens.append(f"GT {example['ground_truth_class']}")
            if "confidence" in example:
                tokens.append(f"Pred {example['predicted_class']} · {example['confidence']:.2f}")
            if example.get("iou") is not None:
                tokens.append(f"IoU {example['iou']:.2f}")
            if category == "difficult_scenes":
                tokens.append(f"{example['false_negatives']} FN · {example['false_positives']} FP")
            figure = directory / f"{number:02d}_{Path(image.image).stem}.png"
            draw_comparison(
                snapshot / image.image,
                image.labels,
                list(operating.values()),
                figure,
                " | ".join(tokens),
                focus=focus,
            )
            gallery[category].append(example | {"figure": figure.relative_to(run).as_posix()})
    return gallery


def make_visual_sample(
    snapshot: Path, run: Path, fp: list[dict], fn: list[dict], *, seed: int
) -> list[dict]:
    """Draw seeded samples for human inspection, without assigning speculative causes."""
    from PIL import Image, ImageDraw, ImageOps

    directory = run / "visual_audit"
    directory.mkdir(exist_ok=True)
    randomizer = random.Random(seed)
    rows = []
    for code, errors, box_key in (
        ("FP", fp, "predicted_box_xywh"),
        ("FN", fn, "ground_truth_box_xywh"),
    ):
        sampled = randomizer.sample(errors, min(16, len(errors)))
        for start in range(0, len(sampled), 8):
            sheet = Image.new("RGB", (1440, 660), "#f1f4f6")
            for index, row in enumerate(sampled[start : start + 8]):
                number = start + index + 1
                sample_id = f"{code}{number:02d}"
                x, y, width, height = row[box_key]
                with Image.open(snapshot / row["image"]) as source:
                    w, h = source.size
                    bounds = (
                        max(0, int((x - max(0.18, width * 1.6)) * w)),
                        max(0, int((y - max(0.18, height * 1.6)) * h)),
                        min(w, int((x + max(0.18, width * 1.6)) * w + 1)),
                        min(h, int((y + max(0.18, height * 1.6)) * h + 1)),
                    )
                    crop = source.convert("RGB").crop(bounds)
                thumb = ImageOps.contain(crop, (346, 278), Image.Resampling.LANCZOS)
                scale_x, scale_y = thumb.width / crop.width, thumb.height / crop.height
                focused = (
                    ((x - width / 2) * w - bounds[0]) * scale_x,
                    ((y - height / 2) * h - bounds[1]) * scale_y,
                    ((x + width / 2) * w - bounds[0]) * scale_x,
                    ((y + height / 2) * h - bounds[1]) * scale_y,
                )
                ImageDraw.Draw(thumb).rectangle(focused, outline="#ffda31", width=2)
                left, top = index % 4 * 360, index // 4 * 330
                sheet.paste(thumb, (left + (360 - thumb.width) // 2, top + 29))
                label = row.get("predicted_class", row["ground_truth_class"])
                status = row["status"].replace("_", " ")
                heading = f"{sample_id}  {label}  {status}"
                if code == "FP":
                    heading += f"  {row['confidence']:.2f}"
                ImageDraw.Draw(sheet).text((left + 8, top + 8), heading, fill="#172230")
                sheet_name = f"{code.lower()}_{start // 8 + 1}.png"
                rows.append(row | {"sample_id": sample_id, "contact_sheet": sheet_name})
            sheet.save(directory / sheet_name)
    return rows
