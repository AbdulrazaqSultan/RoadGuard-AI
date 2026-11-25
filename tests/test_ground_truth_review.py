import json

import pytest

from roadguard.vehicle.ground_truth_review import score_adjudications


def _decide(index, ids=(), status="CONFIRMED", *, spurious=False):
    return {
        "case_id": str(index),
        "review_status": status,
        "ground_truth_defect_ids": json.dumps(ids),
        "physical_class": "",
        "spurious_confirmed": str(spurious).lower(),
        "reviewer": "review fixture",
        "notes": "Recorded visual evidence for this fixture.",
    }


def test_scores_only_manually_supported_event_links():
    cases = [
        {"case_id": str(index), "event_id": f"event-{index}", "video": "video1"}
        for index in range(7)
    ]
    decisions = [
        _decide(0, ("GT001",)),
        _decide(1, ("GT002",)),
        _decide(2, ("GT002",)),
        _decide(3, ("GT003", "GT004")),
        _decide(4, spurious=True),
        _decide(5, status="AMBIGUOUS"),
        _decide(6, status="NOT_REVIEWABLE"),
    ]
    rows, metrics = score_adjudications(cases, decisions)
    assert [row["error_type"] for row in rows] == [
        "CORRECT",
        "FRAGMENTED",
        "FRAGMENTED",
        "OVER_MERGED",
        "SPURIOUS",
        "AMBIGUOUS",
        "AMBIGUOUS",
    ]
    assert metrics["event_identity_accuracy"] == pytest.approx(1 / 5)
    assert metrics["fragmentation_rate"] == pytest.approx(1 / 4)
    assert metrics["over_merge_rate"] == pytest.approx(1 / 5)
    assert metrics["ambiguous_events"] == 1
    assert metrics["not_reviewable_events"] == 1


@pytest.mark.parametrize("decision", [_decide(0), _decide(0, ("GT001",), status="AMBIGUOUS")])
def test_rejects_unsubstantiated_confirmation_or_identity_on_ambiguous_review(decision):
    cases = [{"case_id": "0", "event_id": "event-0", "video": "video1"}]
    with pytest.raises(ValueError, match="Invalid physical-defect decision"):
        score_adjudications(cases, [decision])


def test_no_confirmed_events_returns_undefined_rates():
    cases = [{"case_id": "0", "event_id": "event-0", "video": "video1"}]
    _, metrics = score_adjudications(cases, [_decide(0, status="AMBIGUOUS")])
    assert metrics["event_identity_accuracy"] is None
    assert metrics["over_merge_rate"] is None
    assert metrics["fragmentation_rate"] is None
