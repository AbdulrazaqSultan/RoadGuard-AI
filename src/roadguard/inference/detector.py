"""Reusable inference entry point for a trained RoadGuard checkpoint."""

from pathlib import Path
from typing import Any

from roadguard.data.dataset import Box
from roadguard.evaluation.box_matching import Prediction
from roadguard.models.detection.yolo import load_detector
from roadguard.utils.device import require_cuda


class Detector:
    """Load a trained checkpoint once and run prediction on demand."""

    def __init__(self, checkpoint: Path, *, device: str = "cuda") -> None:
        if device not in {"cuda", "cpu"}:
            raise ValueError("Select the cuda or cpu inference device.")
        if device == "cuda":
            require_cuda()
        self.device: int | str = 0 if device == "cuda" else "cpu"
        self._model = load_detector(checkpoint, trained=True)

    @property
    def execution_device(self) -> str | None:
        """Return the device opened by the backend after its first prediction."""
        device = getattr(self._model.predictor, "device", None)
        return str(device) if device is not None else None

    def predict(self, image: Path, *, image_size: int, confidence: float, iou: float) -> Any:
        """Return Ultralytics results for one local image."""
        if not image.is_file():
            raise FileNotFoundError(image)
        return self._predict(str(image), image_size, confidence, iou)

    def _predict(self, image: Any, size: int, confidence: float, iou: float, **options: Any) -> Any:
        return self._model.predict(
            source=image,
            device=self.device,
            imgsz=size,
            conf=confidence,
            iou=iou,
            verbose=False,
            **options,
        )[0]

    def predict_frame(
        self, pixels: Any, *, image_size: int, confidence: float, iou: float
    ) -> list[Prediction]:
        """Return normalized road-defect boxes from one OpenCV BGR frame."""
        result = self._predict(
            pixels,
            image_size,
            confidence,
            iou,
            quantize=16 if self.device == 0 else None,
            max_det=300,
        ).boxes
        return [
            Prediction(Box(int(class_id), *box), float(score))
            for class_id, score, box in zip(
                result.cls.tolist(), result.conf.tolist(), result.xywhn.tolist(), strict=True
            )
        ]
