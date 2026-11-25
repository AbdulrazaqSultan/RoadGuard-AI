from roadguard.evaluation.stage5_errors import review_image, size_breakdown


def test_frozen_size_cutoffs_and_inclusive_boundaries_are_reused():
    definition = {
        "small_max_inclusive": 0.0625,
        "medium_max_inclusive": 0.25,
        "method": "recorded validation thresholds",
        "area": "relative box area",
    }
    image = review_image(
        {
            "image": "images/test/fixture.jpg",
            "labels": [
                [0, 0.25, 0.25, 0.25, 0.25],
                [1, 0.50, 0.50, 0.50, 0.50],
                [2, 0.50, 0.50, 0.75, 0.75],
            ],
            "predictions": [[2, 0.8, 0.50, 0.50, 0.75, 0.75]],
        },
        "fixture",
        0.25,
    )
    archived_definition, rows = size_breakdown([image], definition=definition)
    assert archived_definition == definition
    results = {row["size"]: row for row in rows if row["class"] == "All"}
    assert [results[key]["annotations"] for key in ("small", "medium", "large")] == [1, 1, 1]
    assert [results[key]["false_negative_rate"] for key in ("small", "medium", "large")] == [
        1.0,
        1.0,
        0.0,
    ]
