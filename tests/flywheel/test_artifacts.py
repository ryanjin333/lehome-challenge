import json
from pathlib import Path

import pytest

from lehome.flywheel.artifacts import EpisodeArtifactWriter, verify_episode


def test_episode_is_visible_only_after_atomic_finalize(tmp_path: Path) -> None:
    writer = EpisodeArtifactWriter(tmp_path, "episode-001")
    writer.append_annotation({"step": 0, "action_source": "policy"})
    assert not (tmp_path / "raw" / "episode-001").exists()

    final = writer.finalize({"outcome": "timeout"})

    assert final == tmp_path / "raw" / "episode-001"
    assert verify_episode(final)["episode_id"] == "episode-001"
    assert not (tmp_path / ".pending" / "episode-001").exists()


def test_finalize_refuses_missing_or_empty_video(tmp_path: Path) -> None:
    writer = EpisodeArtifactWriter(tmp_path, "episode-002")
    writer.append_annotation({"step": 0})
    with pytest.raises(ValueError, match="video"):
        writer.finalize({"outcome": "success"}, required_videos=("top.mp4",))

    video = writer.staging / "videos" / "top.mp4"
    video.parent.mkdir()
    video.touch()
    with pytest.raises(ValueError, match="video"):
        writer.finalize({"outcome": "success"}, required_videos=("top.mp4",))


def test_finalize_rejects_path_traversal_symlinks_and_duplicate_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="path-safe"):
        EpisodeArtifactWriter(tmp_path, "../escape")

    writer = EpisodeArtifactWriter(tmp_path, "episode-003")
    writer.append_annotation({"step": 0})
    (writer.staging / "link").symlink_to(tmp_path / "target")
    with pytest.raises(ValueError, match="symlink"):
        writer.finalize({"outcome": "success"})

    first = EpisodeArtifactWriter(tmp_path, "episode-004")
    first.append_annotation({"step": 0})
    first.finalize({"outcome": "success"})
    with pytest.raises(ValueError, match="already exists"):
        EpisodeArtifactWriter(tmp_path, "episode-004")


def test_annotation_append_refuses_a_symlink_target(tmp_path: Path) -> None:
    writer = EpisodeArtifactWriter(tmp_path, "episode-symlink")
    target = tmp_path / "outside.jsonl"
    (writer.staging / "annotations.jsonl").symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        writer.append_annotation({"step": 0})
    assert not target.exists()


def test_verify_rejects_unlisted_files_bad_hash_and_duplicate_manifest_entries(tmp_path: Path) -> None:
    writer = EpisodeArtifactWriter(tmp_path, "episode-005")
    writer.append_annotation({"step": 0})
    final = writer.finalize({"outcome": "success"})

    (final / "unexpected.txt").write_text("not in manifest", encoding="utf-8")
    with pytest.raises(ValueError, match="unlisted"):
        verify_episode(final)
    (final / "unexpected.txt").unlink()

    manifest_path = final / "SHA256SUMS.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["episode.json"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        verify_episode(final)

    manifest_path.write_text(
        '{"annotations.jsonl":{"size":12,"sha256":"' + "0" * 64
        + '"},"annotations.jsonl":{"size":12,"sha256":"' + "0" * 64 + '"}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        verify_episode(final)
