from __future__ import annotations

import json
from pathlib import Path

import pytest

from lehome_train.flywheel.materialize import materialize_episode


def _raw_episode(root: Path, *, grade: str = "A", holdout: bool = False) -> Path:
    root.mkdir(parents=True)
    frames = []
    for index in range(20):
        frames.append(
            {
                "step": index,
                "action_source": "policy" if index < 4 else "expert",
                "state": [float(index)] * 12,
                "action": [float(index + 1)] * 12,
                "cameras": {name: f"{name}-{index}.jpg" for name in ("top_rgb", "left_rgb", "right_rgb")},
            }
        )
    (root / "episode.json").write_text(json.dumps({
        "episode_id": "episode-1", "quality_grade": grade, "official_success": True,
        "garment_name": "public-unseen" if holdout else "organizer-shirt",
        "fps": 30, "frames": frames, "provenance": {"policy_revision": "a" * 40},
    }), encoding="utf-8")
    return root


def test_materializer_writes_three_camera_12d_expert_windows(tmp_path: Path) -> None:
    report = materialize_episode(_raw_episode(tmp_path / "raw"), tmp_path / "out")

    assert report.selected_observations == 1
    assert report.rejected_by_reason["policy"] == 4
    payload = json.loads((tmp_path / "out" / "episode.json").read_text(encoding="utf-8"))
    assert tuple(payload["camera_keys"]) == ("top_rgb", "left_rgb", "right_rgb")
    assert len(payload["frames"][0]["state"]) == 12
    assert len(payload["frames"][0]["action"]) == 12
    assert payload["frames"][0]["future_actions"][-1] == [20.0] * 12


@pytest.mark.parametrize(("grade", "holdout", "message"), [("C", False, "Grade C"), ("A", True, "holdout")])
def test_materializer_rejects_grade_c_and_holdout(tmp_path: Path, grade: str, holdout: bool, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        materialize_episode(_raw_episode(tmp_path / "raw", grade=grade, holdout=holdout), tmp_path / "out")
