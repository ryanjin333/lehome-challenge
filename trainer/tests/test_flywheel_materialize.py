from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from lehome.flywheel.artifacts import EpisodeArtifactWriter
from lehome_train.flywheel.materialize import materialize_episode


def _video(path: Path, frames: int = 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(("ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=red:s=2x2:r=30", "-frames:v", str(frames), "-pix_fmt", "yuv420p", str(path)), check=True)


def _raw_episode(root: Path, *, grade: str = "A", holdout: bool = False) -> Path:
    writer = EpisodeArtifactWriter(root, "episode-1")
    for index in range(20):
        writer.append_annotation({
            "step": index, "monotonic_ns": index, "wall_time_ns": index,
            "action_source": "policy" if index < 4 else "expert", "segment": 0 if index < 4 else 1,
            "state": [float(index)] * 12, "action": [float(index + 1)] * 12,
            "reward": 1.0, "success": True, "expert_sequence": index if index >= 4 else None,
            "expert_sample_age_ms": 1.0 if index >= 4 else None,
        })
    for camera in ("top_rgb", "left_rgb", "right_rgb"):
        _video(writer.staging / "videos" / f"{camera}.mp4")
    return writer.finalize({
        "mode": "dagger", "outcome": "success", "accepted_success": True, "trainable": True,
        "quality_grade": grade, "rejection_reasons": [],
        "identity": {"release_stage": "public_unseen" if holdout else "seen", "instruction": "fold the garment on the table", "policy_revision": "a" * 40},
    }, required_videos=("top_rgb.mp4", "left_rgb.mp4", "right_rgb.mp4"))


def test_materializer_verifies_real_artifact_and_writes_canonical_v2_layout(tmp_path: Path) -> None:
    report = materialize_episode(_raw_episode(tmp_path), tmp_path / "out")

    assert report.selected_observations == 1
    assert report.rejected_by_reason["policy"] == 4
    payload = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert payload["output_format"] == "groot_lerobot_v2.1_per_episode"
    assert payload["state_schema"]["dimension"] == payload["action_schema"]["dimension"] == 12
    assert (tmp_path / "out" / "data" / "chunk-000" / "episode_000000.parquet").is_file()
    assert (tmp_path / "out" / "videos" / "chunk-000" / "top_rgb" / "episode_000000.mp4").is_file()


@pytest.mark.parametrize(("grade", "holdout", "message"), [("C", False, "Grade C"), ("A", True, "holdout")])
def test_materializer_rejects_grade_c_and_holdout(tmp_path: Path, grade: str, holdout: bool, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        materialize_episode(_raw_episode(tmp_path, grade=grade, holdout=holdout), tmp_path / "out")


def test_materializer_rejects_tampered_or_unlisted_raw_artifact(tmp_path: Path) -> None:
    raw = _raw_episode(tmp_path)
    (raw / "surprise.txt").write_text("unlisted", encoding="utf-8")
    with pytest.raises(ValueError, match="unlisted"):
        materialize_episode(raw, tmp_path / "out")
