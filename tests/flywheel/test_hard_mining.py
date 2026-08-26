import hashlib
import json

import pytest

from lehome.flywheel.hard_mining import FailureEvidence, rank_failures
from scripts.build_hard_state_recovery_matrix import (
    _progress,
    build_matrix,
    collect_failures,
    main as build_matrix_main,
)


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


def test_recovery_matrix_excludes_unverified_near_misses_and_dead_terminals() -> None:
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

    assert matrix == []


def test_matrix_builder_defaults_to_a_new_near_miss_artifact(tmp_path) -> None:
    campaign_root = tmp_path / "campaign"
    campaign_root.mkdir()

    assert build_matrix_main(["--campaign-root", str(campaign_root)]) == 0
    assert (campaign_root / "hard-state-nearmiss.json").read_text(encoding="utf-8") == "[]\n"


def test_matrix_builder_combines_failures_from_multiple_campaign_roots(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    raw = first / "worker" / "raw" / "episode-a"
    continuations = raw / "snapshots" / "continuations"
    continuations.mkdir(parents=True)
    second.mkdir()
    (raw / "episode.json").write_text(
        json.dumps({
            "accepted_success": False,
            "episode_id": "episode-a",
            "identity": {
                "category": "pant_long",
                "garment_name": "Pant_Long_Seen_4",
                "seed": 920_094,
            },
        }) + "\n",
        encoding="utf-8",
    )
    annotations = [
        {"step": 0, "success": False, "reward": 0.10},
        {"step": 16, "success": False, "reward": 0.52},
        {"step": 32, "success": False, "reward": 0.50},
        {"step": 35, "success": False, "reward": 0.30},
    ]
    (raw / "annotations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in annotations), encoding="utf-8"
    )
    (raw / "snapshots" / "terminal.json").write_text("{}\n", encoding="utf-8")
    for step in (16, 32):
        (continuations / f"{step:06d}.json").write_text(json.dumps({
            "schema_version": 3,
            "robot_position": [0.0] * 12,
            "robot_velocity": [0.0] * 12,
            "cloth_position": [[0.0, 0.0, 0.0]],
            "cloth_velocity": [[0.0, 0.0, 0.0]],
            "rng_state": {},
            "garment_name": "Pant_Long_Seen_4",
            "randomization": {"strategy": "canonical", "continuation_step": step},
            "scene_state": {},
            "cloth_state_authority": "usd_local_points_v1",
        }) + "\n", encoding="utf-8")
    output = tmp_path / "combined.json"

    assert build_matrix_main([
        "--campaign-root", str(first),
        "--campaign-root", str(second),
        "--output", str(output),
        "--category-success", "pant_long=0.0",
    ]) == 0

    assert len(json.loads(output.read_text(encoding="utf-8"))) == 1


def test_progress_uses_peak_step_reward_and_counts_only_the_trailing_stall() -> None:
    annotations = [
        {"step": 0, "success": False, "reward": 0.10},
        {"step": 1, "success": False, "reward": 0.70},
        {"step": 2, "success": False, "reward": 0.68},
        {"step": 3, "success": False, "reward": 0.65},
    ]

    assert _progress({"outcome": "timeout"}, annotations) == (0.70, 2)


def test_failure_audit_restores_nearest_snapshot_at_reward_drop_not_terminal(tmp_path) -> None:
    raw = tmp_path / "campaign" / "worker" / "raw" / "episode-a"
    continuations = raw / "snapshots" / "continuations"
    continuations.mkdir(parents=True)
    (raw / "episode.json").write_text(
        '{"accepted_success":false,"episode_id":"episode-a","identity":'
        '{"category":"top_short","garment_name":"Top_Short_Seen_0","seed":7}}\n',
        encoding="utf-8",
    )
    annotations = [
        {"step": 0, "success": False, "reward": 0.10},
        {"step": 16, "success": False, "reward": 0.52},
        {"step": 32, "success": False, "reward": 0.50},
        {"step": 35, "success": False, "reward": 0.33},
    ]
    (raw / "annotations.jsonl").write_text(
        "".join(f"{__import__('json').dumps(row)}\n" for row in annotations),
        encoding="utf-8",
    )
    (raw / "snapshots" / "terminal.json").write_text("{}\n", encoding="utf-8")
    for step in (16, 32):
        (continuations / f"{step:06d}.json").write_text(json.dumps({
            "schema_version": 3,
            "robot_position": [0.0] * 12,
            "robot_velocity": [0.0] * 12,
            "cloth_position": [[0.0, 0.0, 0.0]],
            "cloth_velocity": [[0.0, 0.0, 0.0]],
            "rng_state": {},
            "garment_name": "Top_Short_Seen_0",
            "randomization": {"strategy": "canonical", "continuation_step": step},
            "scene_state": {"garment_reset_pose": [0.0, 0.0, 0.67, 0.0, 0.0, 90.0]},
            "cloth_state_authority": "usd_local_points_v1",
        }) + "\n", encoding="utf-8")

    failures = collect_failures(tmp_path / "campaign")

    assert len(failures) == 1
    assert failures[0]["restore_snapshot"] == str(continuations / "000032.json")
    assert failures[0]["restore_snapshot_sha256"] == hashlib.sha256(
        (continuations / "000032.json").read_bytes()
    ).hexdigest()
    assert failures[0]["restore_snapshot_cloth_frame"] == "usd_local_points_v1"
    assert failures[0]["restore_snapshot_step"] == 32
    assert failures[0]["moment_of_ruin"] == {
        "signal": "dense_reward_proxy_no_success_head",
        "peak_progress": 0.52,
        "peak_step": 16,
        "detection_step": 35,
        "restore_step": 32,
        "progress_drop": pytest.approx(0.19),
        "drop_threshold": 0.12,
        "minimum_peak": 0.25,
    }


def test_recovery_matrix_binds_moment_of_ruin_snapshot_and_evidence() -> None:
    moment = {
        "signal": "dense_reward_proxy_no_success_head",
        "peak_progress": 0.7,
        "peak_step": 32,
        "detection_step": 47,
        "restore_step": 32,
        "progress_drop": 0.2,
        "drop_threshold": 0.12,
        "minimum_peak": 0.25,
    }
    rows = [{
        "episode_id": "near-miss",
        "episode_path": "/campaign/near-miss/episode.json",
        "terminal_path": "/campaign/near-miss/terminal.json",
        "restore_snapshot": "/campaign/near-miss/continuations/000032.json",
        "restore_snapshot_sha256": "a" * 64,
        "restore_snapshot_cloth_frame": "usd_local_points_v1",
        "restore_snapshot_step": 32,
        "moment_of_ruin": moment,
        "category": "top_short",
        "garment": "Top_Short_Seen_0",
        "seed": 11,
        "max_progress": 0.7,
        "stalled_steps": 10,
        "length": 100,
        "restorable": True,
    }]

    matrix = build_matrix(rows, category_success={"top_short": 0.25}, limit=1)

    assert matrix[0]["restore_snapshot"] == rows[0]["restore_snapshot"]
    assert matrix[0]["restore_snapshot_sha256"] == "a" * 64
    assert matrix[0]["restore_snapshot_cloth_frame"] == "usd_local_points_v1"
    assert matrix[0]["restore_snapshot_step"] == 32
    assert matrix[0]["replay_kind"] == "verified_hard_state_moment_of_ruin_v1"
    assert matrix[0]["parent_episode_id"] == "near-miss"
    assert matrix[0]["lineage_id"] == "near-miss"
    assert matrix[0]["selection_profile"] == "moment_of_ruin_reward_drop_v1"
    assert matrix[0]["selection_evidence"]["moment_of_ruin"] == moment


def test_authenticated_moment_of_ruin_is_not_rejected_by_the_later_terminal_stall() -> None:
    moment = {
        "signal": "dense_reward_proxy_no_success_head",
        "peak_progress": 0.379,
        "peak_step": 153,
        "detection_step": 169,
        "restore_step": 160,
        "progress_drop": 0.13,
        "drop_threshold": 0.12,
        "minimum_peak": 0.25,
    }
    rows = [{
        "episode_id": "early-collapse-long-tail",
        "episode_path": "/campaign/early-collapse-long-tail/episode.json",
        "terminal_path": "/campaign/early-collapse-long-tail/terminal.json",
        "restore_snapshot": "/campaign/early-collapse-long-tail/continuations/000160.json",
        "restore_snapshot_sha256": "b" * 64,
        "restore_snapshot_cloth_frame": "usd_local_points_v1",
        "restore_snapshot_step": 160,
        "moment_of_ruin": moment,
        "category": "pant_long",
        "garment": "Pant_Long_Seen_4",
        "seed": 920_032,
        "max_progress": 0.379,
        "stalled_steps": 446,
        "length": 600,
        "restorable": True,
    }]

    matrix = build_matrix(rows, category_success={"pant_long": 0.0}, limit=1)

    assert len(matrix) == 1
    assert matrix[0]["restore_snapshot_step"] == 160
    assert matrix[0]["selection_evidence"]["eligible_for_recovery"] is True
    assert matrix[0]["selection_evidence"]["terminal_near_miss_eligible"] is False


def test_failure_audit_rejects_a_cuda_cloth_restore_for_cpu_only_hard_state_collection(tmp_path) -> None:
    raw = tmp_path / "campaign" / "worker" / "raw" / "episode-a"
    continuations = raw / "snapshots" / "continuations"
    continuations.mkdir(parents=True)
    (raw / "episode.json").write_text(json.dumps({
        "accepted_success": False,
        "episode_id": "episode-a",
        "identity": {"category": "top_short", "garment_name": "Top_Short_Seen_0", "seed": 7},
    }) + "\n", encoding="utf-8")
    (raw / "annotations.jsonl").write_text(
        json.dumps({"step": 16, "success": False, "reward": 0.5}) + "\n"
        + json.dumps({"step": 32, "success": False, "reward": 0.2}) + "\n",
        encoding="utf-8",
    )
    (raw / "snapshots" / "terminal.json").write_text("{}\n", encoding="utf-8")
    (continuations / "000032.json").write_text(json.dumps({
        "schema_version": 2,
        "robot_position": [0.0] * 12,
        "robot_velocity": [0.0] * 12,
        "cloth_position": [[0.0, 0.0, 0.0]],
        "cloth_velocity": [[0.0, 0.0, 0.0]],
        "rng_state": {},
        "garment_name": "Top_Short_Seen_0",
        "randomization": {"strategy": "canonical", "continuation_step": 32},
        "scene_state": {"garment_reset_pose": [0.0, 0.0, 0.67, 0.0, 0.0, 90.0]},
        "cloth_state_authority": "physx_cloth_view_world_v1",
    }) + "\n", encoding="utf-8")

    failures = collect_failures(tmp_path / "campaign")

    assert len(failures) == 1
    assert failures[0]["restorable"] is False
    assert failures[0]["restore_snapshot"] is None
