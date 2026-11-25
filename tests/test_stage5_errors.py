from roadguard.evaluation.stage5_errors import confusion_counts, review_image


def test_stage5_error_pairs_leave_counts_at_the_operating_point():
    result = review_image(
        {
            "image": "images/val/case.jpg",
            "labels": [
                [0, 0.2, 0.2, 0.2, 0.2],
                [1, 0.5, 0.5, 0.2, 0.2],
                [2, 0.8, 0.8, 0.2, 0.2],
                [3, 0.2, 0.8, 0.1, 0.1],
            ],
            "predictions": [
                [0, 0.90, 0.2, 0.2, 0.2, 0.2],
                [0, 0.80, 0.2, 0.2, 0.2, 0.2],
                [0, 0.70, 0.5, 0.5, 0.2, 0.2],
                [2, 0.60, 0.9, 0.8, 0.2, 0.2],
                [3, 0.20, 0.2, 0.8, 0.1, 0.1],
            ],
        },
        "validation fixture",
        0.25,
    )
    assert [row["status"] for row in result.rows] == [
        "true_positive",
        "duplicate_detection",
        "class_confusion",
        "localization_error",
        "below_operating_threshold",
    ]
    assert [row["status"] for row in result.misses] == [
        "class_confusion",
        "localization_error",
        "missed",
    ]
    assert result.misses[-1]["below_threshold_prediction_index"] == 4
    assert result.rows[4]["floor_true_positive"] is True
    matrix = confusion_counts([result])
    assert matrix["D10"]["D00"] == 1
    assert matrix["D20"]["Background"] == matrix["Background"]["D20"] == 1
    assert matrix["D40"]["Background"] == 1
