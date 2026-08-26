"""The corrected controls keep ordinary source provenance with exact ratios."""
from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[2] / "scripts/generate_control_mixture_inputs.py"
    spec = importlib.util.spec_from_file_location("control_mixture_inputs", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_control_profiles_are_true_pure_bc_and_95_5_ordinary() -> None:
    module = _module()

    assert module.control_schedule("a") == {
        "bc_percent": 100,
        "ordinary_percent": 0,
        "batch64_quotas": {"bc": 64, "rollout": 0, "dagger": 0},
    }
    assert module.control_schedule("b") == {
        "bc_percent": 95,
        "ordinary_percent": 5,
        "batch64_quotas": {"bc": 61, "rollout": 3, "dagger": 0},
    }


def test_control_selection_keeps_authenticated_ordinary_windows_at_zero_quota() -> None:
    module = _module()
    ranges = [
        {"source_kind": "flywheel", "raw_episode_id": "rollout-1", "raw_frame_start": 0},
        {"source_kind": "organizer", "raw_episode_id": "1", "raw_frame_start": 16},
    ]

    assert module.control_selections(ranges) == [ranges[1], ranges[0]]


def test_control_lineage_uses_experiment_manifest_field_names() -> None:
    module = _module()
    selections = [
        {
            "source_kind": "organizer",
            "raw_episode_id": "1",
            "split": "train",
        },
        {
            "source_kind": "organizer",
            "raw_episode_id": "2",
            "split": "validation",
        },
    ]

    assert set(module.lineage_hashes(selections)) == {"train_sha256", "validation_sha256"}


@pytest.mark.parametrize(
    ("arm", "source_kind"),
    (("success-replay", "success_replay"), ("hard-state", "hard_state")),
)
def test_targeted_arms_are_distinct_90_10_sources(
    arm: str, source_kind: str,
) -> None:
    module = _module()

    assert module.control_schedule(arm) == {
        "bc_percent": 90,
        "ordinary_percent": 10,
        "batch64_quotas": {"bc": 58, "rollout": 6, "dagger": 0},
    }
    assert module.arm_source(arm) == source_kind


def test_targeted_sources_must_be_selected_explicitly() -> None:
    module = _module()

    assert module.selected_arms(None) == ("a", "b")
    assert module.selected_arms(["success-replay"]) == ("success-replay",)


def _episode(root: Path, episode_id: str, category: str) -> None:
    path = root / "raw" / episode_id
    path.mkdir(parents=True)
    (path / "episode.json").write_text(
        json.dumps({"identity": {"category": category}}), encoding="utf-8"
    )


def _window(episode_id: str, start: int) -> dict[str, object]:
    return {
        "source_kind": "flywheel",
        "source_manifest_sha256": "c" * 64,
        "source_episode_id": episode_id,
        "raw_episode_id": episode_id,
        "raw_frame_start": start,
        "raw_frame_stop": start + 16,
        "raw_frame_ids": [str(frame) for frame in range(start, start + 16)],
        "split": "train",
    }


def test_success_replay_selection_balances_authenticated_windows_by_category(
    tmp_path: Path,
) -> None:
    module = _module()
    organizer = {
        "source_kind": "organizer",
        "raw_episode_id": "0",
        "raw_frame_start": 0,
        "raw_frame_stop": 16,
        "split": "train",
    }
    categories = ("top_long", "top_short", "pant_long", "pant_short")
    ranges: list[dict[str, object]] = [organizer]
    for category_index, category in enumerate(categories):
        for episode_index in range(2):
            episode_id = f"{category}-{episode_index}"
            _episode(tmp_path, episode_id, category)
            for window_index in range(3 + category_index):
                ranges.append(_window(episode_id, window_index * 16))

    selected = module.targeted_selections(
        ranges,
        campaign_root=tmp_path,
        mode="success-replay",
        recovery_audit=None,
    )

    flywheel = [row for row in selected if row["source_kind"] == "flywheel"]
    by_category: dict[str, int] = {category: 0 for category in categories}
    selected_episodes: dict[str, set[str]] = {category: set() for category in categories}
    for row in flywheel:
        category = str(row["raw_episode_id"]).rsplit("-", 1)[0]
        by_category[category] += 1
        selected_episodes[category].add(str(row["raw_episode_id"]))
    assert by_category == {category: 6 for category in categories}
    assert selected_episodes == {category: {f"{category}-0", f"{category}-1"} for category in categories}
    assert selected[0] == organizer


def test_hard_state_selection_uses_only_audit_bound_recovery_windows_and_balances(
    tmp_path: Path,
) -> None:
    module = _module()
    categories = ("top_long", "top_short", "pant_long")
    ranges: list[dict[str, object]] = [
        {
            "source_kind": "organizer",
            "raw_episode_id": "0",
            "raw_frame_start": 0,
            "raw_frame_stop": 16,
            "split": "train",
        }
    ]
    selected_recoveries: list[dict[str, object]] = []
    for category_index, category in enumerate(categories):
        episode_id = f"{category}-recovery"
        _episode(tmp_path, episode_id, category)
        windows = [_window(episode_id, index * 16) for index in range(2 + category_index)]
        ranges.extend(windows)
        selected_recoveries.append(
            {
                "source_episode_id": episode_id,
                "category": category,
                "h16_ranges": [
                    {
                        "start": row["raw_frame_start"],
                        "stop": row["raw_frame_stop"],
                        "frame_ids": list(
                            range(int(row["raw_frame_start"]), int(row["raw_frame_stop"]))
                        ),
                    }
                    for row in windows
                ],
            }
        )
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "kind": "lehome_successful_recovery_audit",
                "continuation_contract": "authenticated_cloth_snapshot_at_fresh_h16_next_action_boundary_v2",
                "selected_recoveries": selected_recoveries,
            }
        ),
        encoding="utf-8",
    )

    selected = module.targeted_selections(
        ranges,
        campaign_root=tmp_path,
        mode="hard-state",
        recovery_audit=audit,
    )

    flywheel = [row for row in selected if row["source_kind"] == "flywheel"]
    assert len(flywheel) == 6
    assert {
        str(row["raw_episode_id"]).removesuffix("-recovery"): sum(
            other["raw_episode_id"] == row["raw_episode_id"] for other in flywheel
        )
        for row in flywheel
    } == {category: 2 for category in categories}


def test_hard_state_selection_materializes_audited_h16_between_plan_boundaries(
    tmp_path: Path,
) -> None:
    module = _module()
    organizer = {
        "source_kind": "organizer",
        "raw_episode_id": "0",
        "raw_frame_start": 0,
        "raw_frame_stop": 16,
        "split": "train",
    }
    ranges: list[dict[str, object]] = [organizer]
    recoveries: list[dict[str, object]] = []
    for category in ("top_long", "top_short", "pant_long"):
        episode_id = f"{category}-between"
        _episode(tmp_path, episode_id, category)
        annotations = tmp_path / "raw" / episode_id / "annotations.jsonl"
        annotations.write_text("{}\n" * 32, encoding="utf-8")
        ranges.extend([_window(episode_id, 0), _window(episode_id, 16)])
        recoveries.append(
            {
                "source_episode_id": episode_id,
                "category": category,
                "annotation_count": 32,
                "source_artifacts": {
                    "annotations_sha256": hashlib.sha256(annotations.read_bytes()).hexdigest()
                },
                "h16_ranges": [
                    {"start": 3, "stop": 19, "frame_ids": list(range(3, 19))}
                ],
            }
        )
    audit = tmp_path / "between-audit.json"
    audit.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "kind": "lehome_successful_recovery_audit",
                "continuation_contract": "authenticated_cloth_snapshot_at_fresh_h16_next_action_boundary_v2",
                "selected_recoveries": recoveries,
            }
        ),
        encoding="utf-8",
    )

    selected = module.targeted_selections(
        ranges,
        campaign_root=tmp_path,
        mode="hard-state",
        recovery_audit=audit,
    )

    flywheel = [row for row in selected if row["source_kind"] == "flywheel"]
    assert len(flywheel) == 3
    assert {row["raw_frame_start"] for row in flywheel} == {3}
    assert {row["raw_frame_stop"] for row in flywheel} == {19}


def test_targeted_arm_rebinds_rollout_bundle_to_the_actual_published_source() -> None:
    module = _module()
    experiment = {
        "bc_bundle": {
            "repository": "old-bc",
            "revision": "1" * 40,
            "prefix": "old-bc",
            "manifest_sha256": "1" * 64,
            "tree_sha256": "2" * 64,
        },
        "rollout_bundle": {
            "repository": "old-rollout",
            "revision": "2" * 40,
            "prefix": "rollouts/round-2",
            "manifest_sha256": "3" * 64,
            "tree_sha256": "4" * 64,
        },
    }
    plan = {
        "input_bindings": {
            "organizer_manifest_sha256": "a" * 64,
            "organizer_tree_sha256": "b" * 64,
            "campaign_receipt_sha256": "c" * 64,
            "campaign_tree_sha256": "d" * 64,
        }
    }
    publications = {
        "sources": [
            {
                "source_id": "organizer",
                "source_type": "bc",
                "repository": "ryanjin333/lehome-groot-n17-data",
                "revision": "e" * 40,
                "prefix": "bc/full",
            },
            {
                "source_id": "rollout",
                "source_type": "rollout",
                "repository": "ryanjin333/lehome-groot-n17-rollouts",
                "revision": "f" * 40,
                "prefix": "rollouts/round-3",
            },
        ]
    }

    module.bind_source_bundles(experiment, plan=plan, source_publications=publications)

    assert experiment["bc_bundle"] | {
        "repository": "ryanjin333/lehome-groot-n17-data",
        "revision": "e" * 40,
        "prefix": "bc/full",
        "manifest_sha256": "a" * 64,
        "tree_sha256": "b" * 64,
    } == experiment["bc_bundle"]
    assert experiment["rollout_bundle"] | {
        "repository": "ryanjin333/lehome-groot-n17-rollouts",
        "revision": "f" * 40,
        "prefix": "rollouts/round-3",
        "manifest_sha256": "c" * 64,
        "tree_sha256": "d" * 64,
    } == experiment["rollout_bundle"]
