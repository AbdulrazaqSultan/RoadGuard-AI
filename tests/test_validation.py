"""Regression coverage for label geometry, pairing and split leakage."""

import shutil
from pathlib import Path

import pytest
from PIL import Image

from roadguard.data.validation import validate_dataset

GOOD_BOX = "0 0.5 0.5 0.2 0.4"


@pytest.mark.parametrize(
    ("label", "message"),
    [
        ("4 0.5 0.5 0.2 0.4", "class ID"),
        ("1.0 0.5 0.5 0.2 0.4", "invalid literal"),
        ("0 0.5 nan 0.2 0.4", "finite"),
        ("0 0.5 0.5 0 0.4", "zero-area"),
        ("1 0.95 0.5 0.2 0.4", "outside the image"),
        ("0 0.5 0.5 0.2", "unpack"),
    ],
)
def test_invalid_yolo_label(tmp_path, make_pair, label, message):
    make_pair(1, label)
    report = validate_dataset(tmp_path, "yolo")
    assert not report.valid
    assert report.issues[0].code == "invalid_annotation"
    assert message in report.issues[0].detail


def test_empty_missing_and_corrupt_pairs(tmp_path, make_pair):
    make_pair(1, "")
    make_pair(2, None)
    make_pair(3).write_bytes(b"not an image")
    make_pair(4).unlink()
    report = validate_dataset(tmp_path, "yolo")
    assert report.summary()["images_without_annotations"] == 1
    assert set(report.summary()["issue_counts"]) == {
        "empty_annotation_file",
        "missing_label",
        "invalid_image",
        "missing_image",
    }


def test_duplicate_name_and_pixels_cross_splits(tmp_path, make_pair):
    image = make_pair(1, GOOD_BOX, "train")
    renamed = make_pair(2, GOOD_BOX, "val")
    shutil.copyfile(image, renamed)
    make_pair(1, GOOD_BOX, "test")
    report = validate_dataset(tmp_path, "yolo_split")
    assert not report.valid
    assert report.summary()["issue_counts"] == {
        "duplicate_image": 2,
        "split_leakage": 3,
        "duplicate_name": 1,
    }


def test_import_rdd_voc(tmp_path: Path):
    labeled = tmp_path / "India" / "train"
    images = labeled / "images"
    labels = labeled / "annotations" / "xmls"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.new("RGB", (100, 50)).save(images / "sample.jpg")
    (labels / "sample.xml").write_text(
        "<annotation><size><width>100</width><height>50</height></size>"
        "<object><name>D40</name><bndbox><xmin>10</xmin><xmax>30</xmax>"
        "<ymin>10</ymin><ymax>30</ymax></bndbox></object>"
        "<object><name>D44</name></object></annotation>",
        encoding="utf-8",
    )
    report = validate_dataset(tmp_path, "voc")
    assert report.valid
    assert report.records[0].boxes[0].class_id == 3
    assert report.records[0].boxes[0].coordinates == (0.2, 0.4, 0.2, 0.4)
    assert report.issues[0].code == "non_target_classes"
