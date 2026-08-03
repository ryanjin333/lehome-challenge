from __future__ import annotations

import json
from pathlib import Path

import pytest

from lehome_train.flywheel.mix import build_mix_plan


def _dataset(path: Path, *, source: str, frames: int, grade: str = "A", holdout: bool = False) -> Path:
    path.mkdir()
    (path / "manifest.json").write_text(json.dumps({"flywheel_source": source, "frame_count": frames, "episode_ids": [f"{source}-0"]}), encoding="utf-8")
    (path / "provenance.json").write_text(json.dumps({"quality_grade": grade, "action_source": "expert", "release_stage": "public_unseen" if holdout else "seen"}), encoding="utf-8")
    return path


def test_mix_targets_seventy_thirty_by_training_frames(tmp_path: Path) -> None:
    plan = build_mix_plan(_dataset(tmp_path / "organizer", source="organizer", frames=700), _dataset(tmp_path / "new", source="flywheel", frames=300), seed=20260803)
    assert plan.organizer_training_frames == 700
    assert plan.flywheel_training_frames == 300
    assert plan.source_weights == {"organizer": 0.7, "flywheel": 0.3}
    assert plan.grade_weights == {"A": 1.0, "B": 0.5}
    assert plan.sha256 == build_mix_plan(tmp_path / "organizer", tmp_path / "new", seed=20260803).sha256


@pytest.mark.parametrize(("grade", "holdout", "message"), [("A", True, "holdout"), ("C", False, "grade")])
def test_mix_rejects_holdout_and_ineligible_grade(tmp_path: Path, grade: str, holdout: bool, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_mix_plan(_dataset(tmp_path / "org", source="organizer", frames=10), _dataset(tmp_path / "bad", source="flywheel", frames=10, grade=grade, holdout=holdout), seed=1)
