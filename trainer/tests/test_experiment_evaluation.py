"""Strict all-category evaluation report tests."""

from __future__ import annotations


def test_evaluation_requires_four_categories() -> None:
    from lehome_train.groot.experiment_evaluation import build_experiment_evaluation

    categories = {name: {"successes": 4, "episodes": 5} for name in ("top_long", "top_short", "pant_long", "pant_short")}
    report = build_experiment_evaluation(experiment_id="a" * 64, checkpoint_receipt_sha256="b" * 64, matrix_sha256="c" * 64, policy_digest="d" * 64, categories=categories, episode_artifacts=[])
    assert report.overall_successes == 16


def test_evaluation_score_exposes_promotion_metrics() -> None:
    from lehome_train.groot.experiment_evaluation import build_experiment_evaluation, to_evaluation_score

    categories = {name: {"successes": 4, "episodes": 5} for name in ("top_long", "top_short", "pant_long", "pant_short")}
    report = build_experiment_evaluation(
        experiment_id="a" * 64, checkpoint_receipt_sha256="b" * 64,
        matrix_sha256="c" * 64, policy_digest="d" * 64,
        categories=categories, episode_artifacts=[], paired_improvement=0.25,
        gpu_seconds=32.5, infrastructure_retry_count=2, safety_failure=False,
        progress_metrics={"mean_terminal_progress": 0.75},
        recovery_metrics={"successful_recoveries": 3},
    )
    score = to_evaluation_score(report)
    assert score.overall_successes == 16
    assert score.paired_improvement == 0.25
    assert score.gpu_seconds == 32.5
