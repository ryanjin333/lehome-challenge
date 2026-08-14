from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_builder_allows_authenticated_train_episode_to_be_demoted_to_frozen_mixture_validation(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture_builder import validate_plan_windows

    organizer = tmp_path / "organizer"
    _write(organizer / "manifest.json", {
        "train_episode_ids": ["1"], "validation_episode_ids": ["2"],
    })
    plan = {"selected_frame_ranges": [{
        "source_kind": "organizer", "source_manifest_sha256": "a" * 64,
        "source_episode_id": "1", "raw_episode_id": "1", "raw_frame_start": 0,
        "raw_frame_stop": 16, "frame_start": 0, "frame_stop": 16, "raw_frame_ids": [str(i) for i in range(16)],
        "split": "validation",
    }]}

    windows = validate_plan_windows(plan, organizer_manifest=organizer / "manifest.json", accepted_rollouts={})

    assert windows[0]["split"] == "validation"


def test_builder_rejects_original_validation_episode_promoted_to_mixture_train(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture_builder import validate_plan_windows

    _write(tmp_path / "organizer.json", {"train_episode_ids": ["1"], "validation_episode_ids": ["2"]})
    plan = {"selected_frame_ranges": [{
        "source_kind": "organizer", "source_manifest_sha256": "a" * 64,
        "source_episode_id": "2", "raw_episode_id": "2", "raw_frame_start": 0,
        "raw_frame_stop": 16, "frame_start": 0, "frame_stop": 16, "raw_frame_ids": [str(i) for i in range(16)], "split": "train",
    }]}

    with pytest.raises(ValueError, match="split"):
        validate_plan_windows(plan, organizer_manifest=tmp_path / "organizer.json", accepted_rollouts={})


def test_builder_rejects_unaccepted_rollout_even_when_range_is_h16(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture_builder import validate_plan_windows

    _write(tmp_path / "organizer.json", {"train_episode_ids": ["1"], "validation_episode_ids": []})
    plan = {"selected_frame_ranges": [{
        "source_kind": "flywheel", "source_manifest_sha256": "b" * 64,
        "source_episode_id": "0", "raw_episode_id": "attempt-0", "raw_frame_start": 0,
        "raw_frame_stop": 16, "raw_frame_ids": [str(i) for i in range(16)],
        "split": "train",
    }]}

    with pytest.raises(ValueError, match="accepted"):
        validate_plan_windows(plan, organizer_manifest=tmp_path / "organizer.json", accepted_rollouts={})


def test_builder_collapses_exact_duplicate_source_ranges_before_runtime_scheduler_oversamples(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture_builder import validate_plan_windows

    _write(tmp_path / "organizer.json", {"train_episode_ids": ["1"], "validation_episode_ids": []})
    def selection(kind: str, source: str, raw: str, source_start: int) -> dict[str, object]:
        return {"source_kind": kind, "source_manifest_sha256": "a" * 64, "source_episode_id": source, "raw_episode_id": raw, "frame_start": source_start, "frame_stop": source_start + 16, "raw_frame_start": 0, "raw_frame_stop": 16, "raw_frame_ids": [str(i) for i in range(16)], "split": "train"}
    plan = {"selected_frame_ranges": [selection("organizer", "1", "1", 0), selection("organizer", "1", "1", 16), selection("flywheel", "0", "attempt", 0), selection("flywheel", "1", "attempt", 16)]}

    windows = validate_plan_windows(plan, organizer_manifest=tmp_path / "organizer.json", accepted_rollouts={"attempt": "attempt"})

    assert len(windows) == 2
