"""Pure asynchronous promotion ordering tests."""

from __future__ import annotations


def test_category_floor_beats_higher_overall_score() -> None:
    from lehome_train.groot.experiment_promotion import EvaluationScore, rank_key

    balanced = EvaluationScore("balanced", "a" * 64, (3, 3, 3, 2), 11, 0.2, 20.0, False)
    collapsed = EvaluationScore("collapsed", "b" * 64, (5, 5, 3, 0), 13, 0.8, 10.0, False)
    assert rank_key(balanced) > rank_key(collapsed)


def test_top_three_promotions_exclude_safety_failure() -> None:
    from lehome_train.groot.experiment_promotion import EvaluationScore, select_1k_promotions

    scores = [
        EvaluationScore(str(index), (str(index) * 64)[:64], (5, 5, 5, 5), 20 - index, 0.0, 1.0, index == 0)
        for index in range(4)
    ]
    assert [score.experiment_id for score in select_1k_promotions(scores)] == ["1", "2", "3"]
