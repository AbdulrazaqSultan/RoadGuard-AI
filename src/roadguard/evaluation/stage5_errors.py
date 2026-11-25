"""Classify validation detections at the archived operating point."""

from collections import Counter
from dataclasses import dataclass
from statistics import mean, median, quantiles

from roadguard.data.dataset import Box
from roadguard.evaluation.box_matching import Prediction, box_iou, match_indices
from roadguard.evaluation.stage5 import CODE, MATCH_IOU

LOCALIZATION_MIN_IOU = 0.1
BACKGROUND = "Background"
SIZE_BUCKETS = ("small", "medium", "large")


@dataclass(frozen=True)
class ReviewedImage:
    """Capture outputs plus deterministic diagnostic assignments for one image."""

    image: str
    source: str
    labels: tuple[Box, ...]
    predictions: list[Prediction]
    rows: list[dict]
    misses: list[dict]


def review_image(capture: dict, source: str, threshold: float) -> ReviewedImage:
    """Use class-aware TP matching, then pair residual confusion/localization errors."""
    labels = tuple(Box(cid, *box) for cid, *box in capture["labels"])
    predictions = [Prediction(Box(cid, *box), score) for cid, score, *box in capture["predictions"]]
    floor_matches = match_indices(labels, predictions, iou_threshold=MATCH_IOU)
    active = {i for i, pred in enumerate(predictions) if pred.confidence >= threshold}
    matched = {
        pred: (label, overlap, "true_positive")
        for pred, (label, overlap) in floor_matches.items()
        if pred in active
    }
    used = {label for label, _, _ in matched.values()}
    missed = set(range(len(labels))) - used
    overlaps = {
        pred_idx: [box_iou(pred.box, label) for label in labels]
        for pred_idx, pred in enumerate(predictions)
        if pred_idx not in floor_matches
    }
    candidates = []
    for pred_idx in active - matched.keys():
        box = predictions[pred_idx].box
        for label_idx in missed:
            same_class = box.class_id == labels[label_idx].class_id
            iou = overlaps[pred_idx][label_idx]
            if not same_class and iou >= MATCH_IOU:
                candidates.append((0, -iou, pred_idx, label_idx, "class_confusion"))
            elif same_class and LOCALIZATION_MIN_IOU <= iou < MATCH_IOU:
                candidates.append((1, -iou, pred_idx, label_idx, "localization_error"))

    paired_misses = {}
    for _, negative_iou, pred_idx, label_idx, status in sorted(candidates):
        if pred_idx not in matched and label_idx not in used:
            matched[pred_idx] = label_idx, -negative_iou, status
            paired_misses[label_idx] = pred_idx, -negative_iou, status
            used.add(label_idx)

    base = {"image": capture["image"], "source": source}
    rows = []
    for index, pred in enumerate(predictions):
        if index in matched:
            label_idx, iou, status = matched[index]
        elif index in floor_matches:
            label_idx, iou = floor_matches[index]
            status = "below_operating_threshold"
        else:
            nearby = overlaps[index]
            label_idx = max(range(len(nearby)), key=nearby.__getitem__, default=None)
            iou = nearby[label_idx] if label_idx is not None else 0.0
            if iou == 0:
                label_idx = None
            status = "unmatched" if index in active else "below_operating_threshold"
            if index in active:
                duplicate = any(
                    overlap >= MATCH_IOU and labels[j].class_id == pred.box.class_id
                    for j, overlap in enumerate(nearby)
                )
                if duplicate:
                    status = "duplicate_detection"
        label = labels[label_idx] if label_idx is not None else None
        rows.append(
            base
            | {
                "prediction_index": index,
                "predicted_class": CODE[pred.box.class_id],
                "confidence": pred.confidence,
                "predicted_box_xywh": pred.box.coordinates,
                "label_index": label_idx,
                "ground_truth_class": CODE[label.class_id] if label is not None else None,
                "ground_truth_box_xywh": label.coordinates if label is not None else None,
                "iou": iou if label is not None else None,
                "status": status,
                "is_paired": index in matched or index in floor_matches,
                "above_operating_confidence": index in active,
                "floor_true_positive": index in floor_matches,
            }
        )
    below = {
        label: (pred, iou) for pred, (label, iou) in floor_matches.items() if pred not in active
    }
    misses = []
    for index in sorted(missed):
        label = labels[index]
        pred_idx, iou, status = paired_misses.get(index, (None, None, "missed"))
        below_idx, below_iou = below.get(index, (None, None))
        misses.append(
            base
            | {
                "label_index": index,
                "ground_truth_class": CODE[label.class_id],
                "ground_truth_box_xywh": label.coordinates,
                "relative_area": label.width * label.height,
                "prediction_index": pred_idx,
                "iou": iou,
                "status": status,
                "below_threshold_prediction_index": below_idx,
                "below_threshold_confidence": (
                    predictions[below_idx].confidence if below_idx is not None else None
                ),
                "below_threshold_iou": below_iou,
            }
        )
    return ReviewedImage(capture["image"], source, labels, predictions, rows, misses)


def confusion_counts(images: list[ReviewedImage]) -> dict:
    """Use ground truth as rows and predicted class as columns, with a background bin."""
    names = [*CODE.values(), BACKGROUND]
    matrix = {actual: dict.fromkeys(names, 0) for actual in names}
    for image in images:
        for row in image.rows:
            if row["above_operating_confidence"]:
                matched = row["status"] in {"true_positive", "class_confusion"}
                actual = row["ground_truth_class"] if matched else BACKGROUND
                matrix[actual][row["predicted_class"]] += 1
        for row in image.misses:
            if row["status"] != "class_confusion":
                matrix[row["ground_truth_class"]][BACKGROUND] += 1
    return matrix


def size_bucket(area: float, definition: dict) -> str:
    """Assign relative box area using the inclusive validation cutoff definitions."""
    if area <= definition["small_max_inclusive"]:
        return "small"
    if area <= definition["medium_max_inclusive"]:
        return "medium"
    return "large"


def size_breakdown(
    images: list[ReviewedImage], *, definition: dict | None = None
) -> tuple[dict, list[dict]]:
    """Fit area terciles on validation or reuse previously frozen cutoffs."""
    if definition is None:
        areas = [label.width * label.height for image in images for label in image.labels]
        lower, upper = quantiles(areas, n=3, method="inclusive")
        definition = {
            "method": "inclusive empirical terciles of validation annotation relative box area",
            "small_max_inclusive": lower,
            "medium_max_inclusive": upper,
            "area": "normalized width × normalized height",
        }
    all_counts = Counter()
    miss_counts = Counter()
    for image in images:
        missed = {row["label_index"] for row in image.misses}
        for index, label in enumerate(image.labels):
            area = label.width * label.height
            bucket = size_bucket(area, definition)
            for key in ((bucket, "All"), (bucket, CODE[label.class_id])):
                all_counts[key] += 1
                miss_counts[key] += index in missed
    rows = []
    for bucket in SIZE_BUCKETS:
        for code in ("All", *CODE.values()):
            total, missed = all_counts[bucket, code], miss_counts[bucket, code]
            rows.append(
                {
                    "size": bucket,
                    "class": code,
                    "annotations": total,
                    "detected": total - missed,
                    "false_negatives": missed,
                    "detection_rate": (total - missed) / total if total else None,
                    "false_negative_rate": missed / total if total else None,
                }
            )
    return definition, rows


def confidence_breakdown(rows: list[dict], operating_confidence: float) -> tuple[dict, list[dict]]:
    """Summarize native scores at the fixed operating point and the broad evaluation floor."""
    from bisect import bisect_right

    bins = sorted({0.0, 0.05, 0.10, operating_confidence, 0.5, 0.75, 1.0})
    counts = Counter()
    series = {"true_positive": [], "false_positive": []}
    for row in rows:
        status = "true_positive" if row["floor_true_positive"] else "false_positive"
        confidence = row["confidence"]
        bucket = min(bisect_right(bins, confidence) - 1, len(bins) - 2)
        counts[status, bucket] += 1
        series[status].append(confidence)
    summary = {
        status: {
            "all_at_evaluation_floor": len(scores),
            "below_operating_confidence": sum(score < operating_confidence for score in scores),
            "at_operating_confidence": len(active),
            "operating_mean": mean(active) if active else None,
            "operating_median": median(active) if active else None,
            "operating_below_0_50": sum(score < 0.5 for score in active),
        }
        for status, scores in series.items()
        for active in ([score for score in scores if score >= operating_confidence],)
    }
    histogram = [
        {
            "from_inclusive": left,
            "to_exclusive_except_1": right,
            "true_positive": counts["true_positive", index],
            "false_positive": counts["false_positive", index],
        }
        for index, (left, right) in enumerate(zip(bins[:-1], bins[1:], strict=True))
    ]
    return summary, histogram
