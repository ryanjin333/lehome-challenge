import pytest

from lehome.flywheel.hard_mining import FailureEvidence, rank_failures
from scripts.build_hard_state_recovery_matrix import _progress, build_matrix, main as build_matrix_main


def test_near_miss_outranks_dead_terminal_even_with_a_smaller_category_gap() -> None:
    failures = (
        FailureEvidence("pant", "pant_long", False, 0.1, 220, 400, True),
        FailureEvidence("shirt", "top_short", False, 0.7, 20, 400, True),
    )

    ranked = rank_failures(failures, category_success={"pant_long": 0.0, "top_short": 0.8})

    assert [item.episode_id for item in ranked] == ["shirt", "pant"]
    assert ranked[0].priority_reasons == ("category_gap", "high_progress", "short_stall", "restorable")
    assert ranked[0].eligible_for_recovery is True
    assert ranked[0].diagnostics["dead_terminal"] is False
    assert ranked[1].eligible_for_recovery is False
    assert ranked[1].diagnostics["dead_terminal"] is True
    assert ranked[0].official_success is False
    assert ranked[0].official_return is None
    assert ranked[0].diagnostics["max_progress"] == 0.7


def test_ranking_is_deterministic_and_rejects_missing_official_category_metric() -> None:
    failures = (
        FailureEvidence("b", "pant_long", False, 0.0, 0, 1, False),
        FailureEvidence("a", "pant_long", False, 0.0, 0, 1, False),
    )
    assert [item.episode_id for item in rank_failures(failures, category_success={"pant_long": 0.0})] == ["a", "b"]
    with pytest.raises(ValueError, match="category_success"):
        rank_failures(failures, category_success={})


def test_ranking_preserves_an_explicit_official_return_without_diagnostic_substitution() -> None:
    evidence = FailureEvidence("episode", "pant_long", False, 0.2, 0, 100, False, official_return=-4.5)

    ranked = rank_failures((evidence,), category_success={"pant_long": 0.5})

    assert ranked[0].official_return == -4.5
    assert ranked[0].diagnostics["max_progress"] == 0.2


def test_recovery_matrix_excludes_dead_terminals_and_records_near_miss_evidence() -> None:
    rows = [
        {
            "episode_id": "top-short-near",
            "episode_path": "/campaign/top-short-near/episode.json",
            "terminal_path": "/campaign/top-short-near/terminal.json",
            "category": "top_short",
            "garment": "Top_Short_Seen_0",
            "seed": 11,
            "max_progress": 0.7,
            "stalled_steps": 10,
            "length": 100,
        },
        {
            "episode_id": "pant-dead",
            "episode_path": "/campaign/pant-dead/episode.json",
            "terminal_path": "/campaign/pant-dead/terminal.json",
            "category": "pant_long",
            "garment": "Pant_Long_Seen_0",
            "seed": 12,
            "max_progress": 0.1,
            "stalled_steps": 90,
            "length": 100,
        },
    ]

    matrix = build_matrix(
        rows,
        category_success={"top_short": 0.25, "pant_long": 0.2},
        limit=40,
    )

    assert [row["source_episode_id"] for row in matrix] == ["top-short-near"]
    assert matrix[0]["selection_profile"] == "near_miss_v1"
    assert matrix[0]["selection_evidence"] == {
        "max_progress": 0.7,
        "stall_fraction": 0.1,
        "eligible_for_recovery": True,
    }


def test_matrix_builder_defaults_to_a_new_near_miss_artifact(tmp_path) -> None:
    campaign_root = tmp_path / "campaign"
    campaign_root.mkdir()

    assert build_matrix_main(["--campaign-root", str(campaign_root)]) == 0
    assert (campaign_root / "hard-state-nearmiss.json").read_text(encoding="utf-8") == "[]\n"


def test_progress_uses_peak_step_reward_and_counts_only_the_trailing_stall() -> None:
    annotations = [
        {"step": 0, "success": False, "reward": 0.10},
        {"step": 1, "success": False, "reward": 0.70},
        {"step": 2, "success": False, "reward": 0.68},
        {"step": 3, "success": False, "reward": 0.65},
    ]

    assert _progress({"outcome": "timeout"}, annotations) == (0.70, 2)
