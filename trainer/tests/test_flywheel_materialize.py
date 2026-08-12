from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from lehome.flywheel.artifacts import EpisodeArtifactWriter
from lehome_train.flywheel.materialize import materialize_episode, materialize_rft_episode


def _video(path: Path, frames: int = 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(("ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=red:s=2x2:r=30", "-frames:v", str(frames), "-pix_fmt", "yuv420p", str(path)), check=True)


def _raw_episode(root: Path, *, grade: str = "A", holdout: bool = False, frames: int = 20) -> Path:
    writer = EpisodeArtifactWriter(root, "episode-1")
    for index in range(frames):
        writer.append_annotation({
            "step": index, "monotonic_ns": index, "wall_time_ns": index,
            "action_source": "policy" if index < 4 else "expert", "segment": 0 if index < 4 else 1,
            "state": [float(index)] * 12, "action": [float(index + 1)] * 12,
            "reward": 1.0, "success": True, "expert_sequence": index if index >= 4 else None,
            "expert_sample_age_ms": 1.0 if index >= 4 else None,
        })
    for camera in ("top_rgb", "left_rgb", "right_rgb"):
        _video(writer.staging / "videos" / f"{camera}.mp4", frames=frames)
    return writer.finalize({
        "mode": "dagger", "outcome": "success", "accepted_success": True, "trainable": True,
        "quality_grade": grade, "rejection_reasons": [],
        "identity": {"release_stage": "public_unseen" if holdout else "seen", "instruction": "fold the garment on the table", "policy_revision": "a" * 40, "code_revision": "b" * 40},
    }, required_videos=("top_rgb.mp4", "left_rgb.mp4", "right_rgb.mp4"))


def _raw_rft_episode(
    root: Path,
    *,
    episode_id: str = "episode-rft-1",
    holdout: bool = False,
    accepted_success: bool = True,
    outcome: str = "success",
    terminal_reason: str = "success",
    frames: int = 45,
    production_schema: bool = True,
    category: str = "top_long",
) -> Path:
    writer = EpisodeArtifactWriter(root, episode_id)
    for index in range(frames):
        writer.append_annotation({
            "step": index, "monotonic_ns": index, "wall_time_ns": index,
            "action_source": "policy", "segment": 0,
            "state": [float(index)] * 12, "action": [float(index + 1)] * 12,
            "reward": 1.0, "success": accepted_success,
            "policy_request_id": f"request-{index // 40}",
            "policy_chunk_offset": index % 40,
        })
    for camera in ("top_rgb", "left_rgb", "right_rgb"):
        _video(writer.staging / "videos" / f"{camera}.mp4", frames=frames)
    episode = {
        "outcome": outcome,
        "accepted_success": accepted_success, "terminal_reason": terminal_reason,
        "bc_target_count": 0,
        "provenance": {
            "execution_backend": "policy_server",
            "execution_mode": "policy_server",
            "parity_stage": "server_cpu",
            "policy_artifact_sha256": "c" * 64,
            "policy_device": "cuda:0",
            "simulator_device": "cpu",
        },
        "identity": {
            "category": category,
            "release_stage": "public_unseen" if holdout else "seen",
            "instruction": "fold the garment on the table",
            "policy_revision": "a" * 40,
            "code_revision": "b" * 40,
        },
    }
    if not production_schema:
        episode["mode"] = "autonomous"
        episode.pop("bc_target_count")
        episode.pop("provenance")
    return writer.finalize(
        episode,
        required_videos=("top_rgb.mp4", "left_rgb.mp4", "right_rgb.mp4"),
    )


def test_materializer_verifies_real_artifact_and_writes_canonical_v2_layout(tmp_path: Path) -> None:
    report = materialize_episode(_raw_episode(tmp_path), tmp_path / "out")

    assert report.selected_observations == 1
    assert report.rejected_by_reason["policy"] == 4
    payload = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert payload["output_format"] == "groot_lerobot_v2.1_per_episode"
    assert payload["state_schema"]["dimension"] == payload["action_schema"]["dimension"] == 12
    assert payload["frame_count"] == 16
    assert payload["future_actions"]["tail_convention"] == "one_complete_raw_window_per_episode"
    provenance = json.loads((tmp_path / "out" / "meta" / "materialization-provenance.json").read_text(encoding="utf-8"))
    assert provenance["selected_frame_ranges"] == [{"raw_episode_id": "episode-1", "frame_start": 4, "frame_stop": 20, "action_source": "expert"}]
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


def test_rft_materializer_accepts_only_verified_seen_policy_successes(tmp_path: Path) -> None:
    report = materialize_rft_episode(_raw_rft_episode(tmp_path), tmp_path / "out")

    assert report.selected_observations == 6
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["episode_count"] == 1
    assert manifest["frame_count"] == 45
    assert manifest["future_actions"] == {
        "horizon": 40,
        "loader_allow_padding": False,
        "materialized_windows": False,
        "tail_convention": "drop_incomplete_windows",
        "valid_window_counts": {"0": 6},
    }
    provenance = json.loads(
        (tmp_path / "out" / "meta" / "materialization-provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["training_method"] == "rejection_finetuning"
    assert provenance["selected_frame_ranges"] == [{
        "raw_episode_id": "episode-rft-1",
        "frame_start": 0,
        "frame_stop": 45,
        "action_source": "policy",
    }]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"holdout": True}, "holdout"),
        ({"accepted_success": False, "outcome": "failure", "terminal_reason": "timeout"}, "accepted autonomous success"),
    ],
)
def test_rft_materializer_rejects_holdout_and_failures(
    tmp_path: Path, kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        materialize_rft_episode(_raw_rft_episode(tmp_path, **kwargs), tmp_path / "out")
