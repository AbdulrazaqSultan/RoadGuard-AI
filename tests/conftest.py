"""Synthetic source images with distinct pixels for data pipeline tests."""

from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def make_pair(tmp_path: Path) -> Callable[..., Path]:
    """Create an image and label in the requested partition."""

    def factory(number: int, label: str | None = "0 0.5 0.5 0.4 0.4", split: str = "") -> Path:
        image = tmp_path / "images" / split / f"image_{number}.png"
        annotation = tmp_path / "labels" / split / f"image_{number}.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 10), (number, 40, 80)).save(image)
        if label is not None:
            annotation.parent.mkdir(parents=True, exist_ok=True)
            annotation.write_text(label, encoding="utf-8")
        return image

    return factory
