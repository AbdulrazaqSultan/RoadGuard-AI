"""Serialize Ultralytics detection metrics with explicit absent-class values."""

from typing import Any

from roadguard.data.dataset import CLASS_NAMES


def detection_metrics(metrics: Any) -> dict[str, Any]:
    """Report AP plus precision, recall and F1 at the backend's operating point."""
    box = metrics.box
    per_class = dict.fromkeys(CLASS_NAMES.values())
    for position, class_id in enumerate(box.ap_class_index):
        precision, recall, ap50, ap = map(float, box.class_result(position))
        per_class[CLASS_NAMES[int(class_id)]] = {
            "precision": precision,
            "recall": recall,
            "f1": float(box.f1[position]),
            "ap50": ap50,
            "ap50_95": ap,
        }
    return {
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "macro_f1": float(box.f1.mean()) if len(box.f1) else 0.0,
        "per_class": per_class,
        "latency_ms_per_image": {key: float(value) for key, value in metrics.speed.items()},
    }
