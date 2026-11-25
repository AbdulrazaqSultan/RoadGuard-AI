"""Exercise saved experiment review without launching or loading a model."""

import json

from roadguard.data.dataset import Box
from roadguard.evaluation.box_matching import Prediction, match_boxes
from roadguard.training.archive import list_experiments, open_experiment
from roadguard.training.review import select_examples


def test_review_example_selection_is_validation_only():
    entries = [
        {"image": str(i), "split": split, "source": str(i % 3), "class_ids": [i % 4]}
        for i, split in enumerate(["train", "test"] + ["val"] * 15)
    ]
    first = select_examples({"files": entries}, 42, count=8)
    assert first == select_examples({"files": entries}, 42, count=8)
    assert len(first) == 8
    assert {entry["split"] for entry in first} == {"val"}
    assert len({entry["source"] for entry in first}) == 3


def test_one_to_one_matches_and_wrong_class_overlap():
    labels = (Box(0, 0.25, 0.25, 0.2, 0.2), Box(3, 0.7, 0.7, 0.1, 0.1))
    result = match_boxes(
        labels,
        [
            Prediction(labels[0], 0.9),
            Prediction(labels[0], 0.7),
            Prediction(Box(2, 0.7, 0.7, 0.1, 0.1), 0.8),
        ],
        iou_threshold=0.5,
    )
    assert result["true_positives"] == result["false_negatives"] == 1
    assert result["false_positives"] == 2
    assert result["class_confusions"] == {"pothole → alligator_crack": 1}


def test_reopen_saved_run_without_cuda(tmp_path):
    run = tmp_path / "trial"
    run.mkdir()
    (run / "metadata.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    (run / "results.csv").write_text(" epoch , time \n 1 , 0.4 \n", encoding="utf-8")
    assert list_experiments(tmp_path) == ["trial"]
    archived = open_experiment(run)
    assert archived.metadata == {"status": "completed"}
    assert archived.epoch_metrics == [{"epoch": "1", "time": "0.4"}]
    assert archived.plots("results.png") == []
