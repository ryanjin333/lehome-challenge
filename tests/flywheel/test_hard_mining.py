import pytest

from lehome.flywheel.hard_mining import FailureEvidence, rank_failures


def test_failures_rank_by_category_gap_stall_and_progress() -> None:
    failures = (
        FailureEvidence("pant", "pant_long", False, 0.1, 220, 400, True),
        FailureEvidence("shirt", "top_short", False, 0.7, 20, 400, False),
    )

    ranked = rank_failures(failures, category_success={"pant_long": 0.0, "top_short": 0.8})

    assert [item.episode_id for item in ranked] == ["pant", "shirt"]
    assert ranked[0].priority_reasons == ("category_gap", "low_progress", "stalled", "restorable")
    assert ranked[0].official_success is False
    assert ranked[0].official_return == 0.1
    assert ranked[0].diagnostics["max_progress"] == 0.1


def test_ranking_is_deterministic_and_rejects_missing_official_category_metric() -> None:
    failures = (
        FailureEvidence("b", "pant_long", False, 0.0, 0, 1, False),
        FailureEvidence("a", "pant_long", False, 0.0, 0, 1, False),
    )
    assert [item.episode_id for item in rank_failures(failures, category_success={"pant_long": 0.0})] == ["a", "b"]
    with pytest.raises(ValueError, match="category_success"):
        rank_failures(failures, category_success={})
