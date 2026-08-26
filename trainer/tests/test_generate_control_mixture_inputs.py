"""The corrected controls keep ordinary source provenance with exact ratios."""
from __future__ import annotations

import importlib.util
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
