"""Match boxes for sampled validation error review at a fixed IoU threshold."""

from collections import Counter
from dataclasses import dataclass

from roadguard.data.dataset import CLASS_NAMES, Box


@dataclass(frozen=True)
class Prediction:
    """A road damage prediction and its detector confidence."""

    box: Box
    confidence: float


def box_iou(left: Box, right: Box) -> float:
    """Intersection over union in normalized image coordinates."""
    dx = min(left.x + left.width / 2, right.x + right.width / 2) - max(
        left.x - left.width / 2, right.x - right.width / 2
    )
    dy = min(left.y + left.height / 2, right.y + right.height / 2) - max(
        left.y - left.height / 2, right.y - right.height / 2
    )
    intersection = max(0.0, dx) * max(0.0, dy)
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union > 0 else 0.0


def match_indices(
    labels: tuple[Box, ...],
    predictions: list[Prediction],
    *,
    iou_threshold: float,
) -> dict[int, tuple[int, float]]:
    """Map prediction indices to same-class labels and IoU in confidence order."""
    if not 0 < iou_threshold <= 1:
        raise ValueError("The IoU match threshold must be in (0, 1].")
    unmatched = set(range(len(labels)))
    matches = {}
    for index in sorted(range(len(predictions)), key=lambda i: -predictions[i].confidence):
        prediction = predictions[index]
        box = prediction.box
        candidates = [
            (box_iou(box, labels[i]), -i) for i in unmatched if labels[i].class_id == box.class_id
        ]
        overlap, negative_index = max(candidates, default=(0.0, 0))
        if overlap >= iou_threshold:
            unmatched.remove(-negative_index)
            matches[index] = (-negative_index, overlap)
    return matches


def match_boxes(
    labels: tuple[Box, ...],
    predictions: list[Prediction],
    *,
    iou_threshold: float,
) -> dict:
    """Summarize one image with confidence-ordered, class-aware one-to-one matches."""
    matches = match_indices(labels, predictions, iou_threshold=iou_threshold)
    unmatched = set(range(len(labels))) - {label for label, _ in matches.values()}
    correct = Counter(predictions[index].box.class_id for index in matches)
    # Keep the previous confidence ordering for the tie-break in confusion pairing.
    false_positives = [
        predictions[index].box
        for index in sorted(range(len(predictions)), key=lambda i: -predictions[i].confidence)
        if index not in matches
    ]
    false_negatives = [labels[i] for i in sorted(unmatched)]
    fp_count = Counter(box.class_id for box in false_positives)
    fn_count = Counter(box.class_id for box in false_negatives)

    overlaps = sorted(
        [
            (box_iou(label, box), label_idx, prediction_idx)
            for label_idx, label in enumerate(false_negatives)
            for prediction_idx, box in enumerate(false_positives)
            if label.class_id != box.class_id
        ],
        reverse=True,
    )
    used_labels: set[int] = set()
    used_predictions: set[int] = set()
    confusions: Counter[str] = Counter()
    for overlap, label_idx, prediction_idx in overlaps:
        if overlap < iou_threshold:
            break
        if label_idx in used_labels or prediction_idx in used_predictions:
            continue
        used_labels.add(label_idx)
        used_predictions.add(prediction_idx)
        expected = CLASS_NAMES[false_negatives[label_idx].class_id]
        inferred = CLASS_NAMES[false_positives[prediction_idx].class_id]
        confusions[f"{expected} → {inferred}"] += 1

    return {
        "true_positives": correct.total(),
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
        "per_class": {
            label: {
                "true_positives": correct[cid],
                "false_positives": fp_count[cid],
                "false_negatives": fn_count[cid],
            }
            for cid, label in CLASS_NAMES.items()
        },
        "class_confusions": dict(confusions),
    }
