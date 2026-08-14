from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from collections import Counter
from threading import Lock
import time

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lehome_train.data.convert import LEGACY_DATA_PATH, LEGACY_VIDEO_PATH, _modality_metadata, _validate_output_video
from lehome_train.data.inspect import artifact_identities
from lehome_train.data.validate import validate_prepared_dataset
from lehome_train.flywheel.mix import (
    ACTION_HORIZON,
    _PersistentLock,
    build_mix_plan,
    load_generation_receipt,
    materialize_mixed_snapshot,
    validate_mix_plan_payload,
    verify_generation,
)
from lehome_train.io import atomic_write_json, canonical_json_sha256
from test_flywheel_materialize import _raw_episode


REVISION = "a" * 40


def _video(path: Path, *, frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(("ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=red:s=2x2:r=30", "-frames:v", str(frames), "-pix_fmt", "yuv420p", str(path)), check=True)


def _prepared_source(
    root: Path,
    *,
    kind: str,
    grade: str | None = None,
    episodes: int = 2,
    frames: int = ACTION_HORIZON,
    episode_start: int = 0,
    release_stage: str = "seen",
    accepted_success: bool = True,
    action_source: str = "expert",
) -> Path:
    """Small real prepared-v2 fixture with parquet and all required MP4 streams."""

    for episode in range(episodes):
        episode_id = episode_start + episode
        base = episode * frames
        path = root / LEGACY_DATA_PATH.format(episode_chunk=episode_id // 1000, episode_index=episode_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({
            "observation.state": pa.array([[float(base + frame)] * 12 for frame in range(frames)], type=pa.list_(pa.float32(), 12)),
            "action": pa.array([[float(base + frame + 1)] * 12 for frame in range(frames)], type=pa.list_(pa.float32(), 12)),
            "timestamp": pa.array([frame / 30 for frame in range(frames)], type=pa.float32()),
            "frame_index": pa.array(range(frames), type=pa.int64()),
            "episode_index": pa.array([episode_id] * frames, type=pa.int64()),
            "index": pa.array(range(base, base + frames), type=pa.int64()),
            "task_index": pa.array([0] * frames, type=pa.int64()),
        }), path, compression="zstd")
        for camera in ("top_rgb", "left_rgb", "right_rgb"):
            _video(root / LEGACY_VIDEO_PATH.format(episode_chunk=episode_id // 1000, episode_index=episode_id, video_key=f"observation.images.{camera}"), frames=frames)
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    atomic_write_json(meta / "info.json", {
        "codebase_version": "v2.1", "robot_type": "dual_so101_follower", "total_episodes": episodes,
        "total_frames": episodes * frames, "total_tasks": 1, "total_videos": episodes * 3,
        "total_chunks": 1, "chunks_size": 1000, "fps": 30, "data_path": LEGACY_DATA_PATH,
        "video_path": LEGACY_VIDEO_PATH, "features": {f"observation.images.{camera}": {"dtype": "video", "shape": [480, 640, 3]} for camera in ("top_rgb", "left_rgb", "right_rgb")},
    })
    (meta / "episodes.jsonl").write_text("".join(json.dumps({"episode_index": episode_start + episode, "length": frames, "task_index": 0}) + "\n" for episode in range(episodes)), encoding="utf-8")
    (meta / "episodes_stats.jsonl").write_text("".join(json.dumps({"episode_index": episode_start + episode, "stats": {}}) + "\n" for episode in range(episodes)), encoding="utf-8")
    (meta / "tasks.jsonl").write_text('{"task":"fold the garment on the table","task_index":0}\n', encoding="utf-8")
    atomic_write_json(meta / "modality.json", _modality_metadata())
    if kind == "flywheel":
        assert grade is not None
        atomic_write_json(meta / "materialization-provenance.json", {
            "raw_episode_id": f"raw-{grade}", "raw_manifest_sha256": ("b" if grade == "A" else "c") * 64,
            "raw_manifest_verified": True, "quality_grade": grade, "selection_horizon": ACTION_HORIZON,
            "raw_identity": {"release_stage": release_stage, "instruction": "fold the garment on the table", "code_revision": REVISION},
            "accepted_success": accepted_success, "trainable": accepted_success, "outcome": "success" if accepted_success else "timeout",
            "rejected_by_reason": {"policy": 1, "hold": 1, "short_tail": 0},
            "selected_frame_ranges": [{"raw_episode_id": f"raw-{grade}", "frame_start": episode * ACTION_HORIZON, "frame_stop": (episode + 1) * ACTION_HORIZON, "action_source": action_source} for episode in range(episodes)],
        })
    artifacts = artifact_identities(root)
    manifest = {
        "schema_version": 1, "output_format": "groot_lerobot_v2.1_per_episode", "source_revision": REVISION,
        "output_artifacts": artifacts, "output_manifest_sha256": canonical_json_sha256(artifacts),
        "frame_count": episodes * frames, "episode_count": episodes, "fps": 30,
        "train_episode_ids": [str(episode_start + episode) for episode in range(episodes)], "validation_episode_ids": [],
        "fixed_language_instruction": "fold the garment on the table",
        "future_actions": {"horizon": ACTION_HORIZON, "loader_allow_padding": False},
        "camera_schema": [{"source_key": f"observation.images.{camera}", "target_modality": camera, "dtype": "video", "shape": [480, 640, 3]} for camera in ("top_rgb", "left_rgb", "right_rgb")],
    }
    atomic_write_json(root / "manifest.json", manifest)
    return root


def test_mix_materializes_real_ranges_with_exact_train_ratio_and_valid_stats(tmp_path: Path) -> None:
    organizer = _prepared_source(tmp_path / "organizer", kind="organizer")
    grade_a = _prepared_source(tmp_path / "grade-a", kind="flywheel", grade="A", episodes=1)
    grade_b = _prepared_source(tmp_path / "grade-b", kind="flywheel", grade="B", episodes=1)

    plan = build_mix_plan(organizer, [grade_a, grade_b], seed=20260803)
    assert build_mix_plan(organizer, [grade_a, grade_b], seed=20260803).to_dict() == plan.to_dict()
    result = materialize_mixed_snapshot(plan, organizer, [grade_a, grade_b], tmp_path / "mixed")

    assert result["validation"]["valid"] is True
    assert plan.organizer_training_frames * 3 == plan.flywheel_training_frames * 7
    assert plan.grade_weights == {"A": 1.0, "B": 0.5}
    assert {item.quality_grade for item in plan.selections if item.source_kind == "flywheel"} == {"A", "B"}
    assert Counter(item.quality_grade for item in plan.selections if item.split == "train" and item.source_kind == "flywheel") == {"A": 2, "B": 1}
    assert all(item.frame_stop - item.frame_start == ACTION_HORIZON for item in plan.selections)
    assert all(len(item.source_frame_ids) == ACTION_HORIZON for item in plan.selections)
    mixed = tmp_path / "mixed"
    manifest = json.loads((mixed / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["flywheel_mix_plan"]["sha256"] == plan.sha256
    assert json.loads((mixed / "meta" / "mix-selection.json").read_text(encoding="utf-8"))["sha256"] == plan.sha256
    assert manifest["train_episode_ids"]
    assert manifest["validation_episode_ids"]
    first = pq.read_table(mixed / LEGACY_DATA_PATH.format(episode_chunk=0, episode_index=0))
    assert first["frame_index"].to_pylist() == list(range(ACTION_HORIZON))
    assert first["episode_index"].to_pylist() == [0] * ACTION_HORIZON
    assert first["index"].to_pylist() == list(range(ACTION_HORIZON))
    all_indices = [
        index
        for episode_id in range(manifest["episode_count"])
        for index in pq.read_table(mixed / LEGACY_DATA_PATH.format(episode_chunk=0, episode_index=episode_id))["index"].to_pylist()
    ]
    assert all_indices == list(range(manifest["frame_count"]))
    for camera in ("top_rgb", "left_rgb", "right_rgb"):
        _validate_output_video(mixed / LEGACY_VIDEO_PATH.format(episode_chunk=0, episode_index=0, video_key=camera), expected_frame_count=ACTION_HORIZON, expected_fps=30)
    manifest["flywheel_mix_plan"]["selected_frame_ranges"][0]["source_frame_ids"][0] = "tampered"
    (mixed / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        validate_prepared_dataset(mixed)


def test_mix_resolves_canonical_source_camera_keys_and_rejects_ambiguity_escape_or_symlinks(tmp_path: Path) -> None:
    import lehome_train.flywheel.mix as mix
    organizer = _prepared_source(tmp_path / "organizer", kind="organizer")
    source = mix._prepared_source(organizer, kind="organizer")
    info = json.loads((organizer / "meta" / "info.json").read_text(encoding="utf-8"))
    assert mix._source_camera_keys(source, info) == {camera: f"observation.images.{camera}" for camera in ("top_rgb", "left_rgb", "right_rgb")}
    # Canonical aggregate-RFT schemas omit target_modality but retain the exact
    # observation.images suffix contract.
    for item in source.manifest["camera_schema"]:
        item.pop("target_modality")
    assert mix._source_camera_keys(source, info)["top_rgb"] == "observation.images.top_rgb"
    untyped_info = dict(info)
    untyped_info["features"] = {}
    with pytest.raises(ValueError, match="feature contract"):
        mix._source_camera_keys(source, untyped_info)
    malformed_info = dict(info)
    malformed_info["features"] = dict(info["features"])
    malformed_info["features"]["observation.images.top_rgb"] = {"dtype": "video", "shape": [1]}
    with pytest.raises(ValueError, match="canonical video"):
        mix._source_camera_keys(source, malformed_info)
    source.manifest["camera_schema"].append(dict(source.manifest["camera_schema"][0]))
    with pytest.raises(ValueError, match="missing or ambiguous"):
        mix._source_camera_keys(source, info)
    source.manifest["camera_schema"].pop()
    source.manifest["camera_schema"] = [item for item in source.manifest["camera_schema"] if item["source_key"] != "observation.images.left_rgb"]
    with pytest.raises(ValueError, match="missing or ambiguous"):
        mix._source_camera_keys(source, info)
    info["video_path"] = "../videos/{video_key}/episode_{episode_index:06d}.mp4"
    with pytest.raises(ValueError, match="escapes"):
        mix._source_video_path(source, info, episode=0, source_key="observation.images.top_rgb")
    info["video_path"] = LEGACY_VIDEO_PATH

    videos = organizer / "videos"
    outside = tmp_path / "outside-videos"
    videos.rename(outside)
    videos.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        mix._source_video_path(source, info, episode=0, source_key="observation.images.top_rgb")

    organizer = _prepared_source(tmp_path / "organizer-intermediate", kind="organizer")
    source = mix._prepared_source(organizer, kind="organizer")
    info = json.loads((organizer / "meta" / "info.json").read_text(encoding="utf-8"))
    chunk = organizer / "videos" / "chunk-000"
    outside_chunk = tmp_path / "outside-chunk"
    chunk.rename(outside_chunk)
    chunk.symlink_to(outside_chunk, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        mix._source_video_path(source, info, episode=0, source_key="observation.images.top_rgb")


def test_mix_parallel_video_materialization_is_bounded_and_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only independent video slices run concurrently; sealed output is stable."""

    organizer = _prepared_source(tmp_path / "organizer", kind="organizer")
    grade_a = _prepared_source(tmp_path / "grade-a", kind="flywheel", grade="A", episodes=1)
    grade_b = _prepared_source(tmp_path / "grade-b", kind="flywheel", grade="B", episodes=1)
    plan = build_mix_plan(organizer, [grade_a, grade_b], seed=20260813)
    active = 0
    maximum_active = 0
    lock = Lock()

    def copy_video(source: Path, destination: Path, *, start: int, stop: int) -> None:
        nonlocal active, maximum_active
        assert stop - start == ACTION_HORIZON
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            time.sleep(0.01)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr("lehome_train.flywheel.mix._copy_selected_video", copy_video)
    single = materialize_mixed_snapshot(
        plan, organizer, [grade_a, grade_b], tmp_path / "single", video_workers=1,
    )
    assert maximum_active == 1
    maximum_active = 0
    parallel = materialize_mixed_snapshot(
        plan, organizer, [grade_a, grade_b], tmp_path / "parallel", video_workers=2,
    )

    assert maximum_active == 2
    single_root = Path(single["path"])
    parallel_root = Path(parallel["path"])
    assert artifact_identities(single_root, exclude={"manifest.json"}) == artifact_identities(
        parallel_root, exclude={"manifest.json"},
    )
    assert json.loads((single_root / "manifest.json").read_text(encoding="utf-8")) == json.loads(
        (parallel_root / "manifest.json").read_text(encoding="utf-8"),
    )
    assert load_generation_receipt(single_root) == load_generation_receipt(parallel_root)


def test_mix_real_ffmpeg_video_materialization_is_identical_across_worker_counts(
    tmp_path: Path,
) -> None:
    """Real libx264 slices must retain the sealed output identity."""

    organizer = _prepared_source(tmp_path / "organizer", kind="organizer")
    grade_a = _prepared_source(tmp_path / "grade-a", kind="flywheel", grade="A", episodes=1)
    grade_b = _prepared_source(tmp_path / "grade-b", kind="flywheel", grade="B", episodes=1)
    plan = build_mix_plan(organizer, [grade_a, grade_b], seed=20260813)

    single = materialize_mixed_snapshot(
        plan, organizer, [grade_a, grade_b], tmp_path / "single", video_workers=1,
    )
    parallel = materialize_mixed_snapshot(
        plan, organizer, [grade_a, grade_b], tmp_path / "parallel", video_workers=2,
    )

    single_root = Path(single["path"])
    parallel_root = Path(parallel["path"])
    assert artifact_identities(single_root, exclude={"manifest.json"}) == artifact_identities(
        parallel_root, exclude={"manifest.json"},
    )
    assert json.loads((single_root / "manifest.json").read_text(encoding="utf-8")) == json.loads(
        (parallel_root / "manifest.json").read_text(encoding="utf-8"),
    )
    assert load_generation_receipt(single_root) == load_generation_receipt(parallel_root)


@pytest.mark.parametrize("video_workers", (0, 1.5, 9))
def test_mix_rejects_invalid_video_worker_caps(tmp_path: Path, video_workers: object) -> None:
    organizer = _prepared_source(tmp_path / "organizer", kind="organizer")
    flywheel = _prepared_source(tmp_path / "flywheel", kind="flywheel", grade="A")
    plan = build_mix_plan(organizer, flywheel, seed=20260813)

    with pytest.raises(ValueError, match="video_workers"):
        materialize_mixed_snapshot(
            plan, organizer, flywheel, tmp_path / "mixed", video_workers=video_workers,  # type: ignore[arg-type]
        )


def test_mix_video_failure_cleans_temporary_tree_without_generation_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    organizer = _prepared_source(tmp_path / "organizer", kind="organizer")
    flywheel = _prepared_source(tmp_path / "flywheel", kind="flywheel", grade="A")
    plan = build_mix_plan(organizer, flywheel, seed=20260813)
    destination = tmp_path / "failed"

    def fail_slice(source: Path, destination: Path, *, start: int, stop: int) -> None:
        raise RuntimeError("synthetic slice failure")

    monkeypatch.setattr("lehome_train.flywheel.mix._copy_selected_video", fail_slice)

    with pytest.raises(RuntimeError, match="synthetic slice failure"):
        materialize_mixed_snapshot(plan, organizer, flywheel, destination, video_workers=2)

    assert not destination.exists()
    assert not destination.with_name(destination.name + ".generation.json").exists()
    assert not list(tmp_path.glob(".failed.*.tmp"))


def test_mix_persistent_staging_retains_verified_work_after_a_slice_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A named staging root is the opt-in crash-resume boundary."""
    organizer = _prepared_source(tmp_path / "organizer", kind="organizer")
    flywheel = _prepared_source(tmp_path / "flywheel", kind="flywheel", grade="A")
    plan = build_mix_plan(organizer, flywheel, seed=20260813)
    staging = tmp_path / "resume"

    def fail_slice(source: Path, destination: Path, *, start: int, stop: int) -> None:
        raise RuntimeError("synthetic persistent slice failure")

    monkeypatch.setattr("lehome_train.flywheel.mix._copy_selected_video", fail_slice)
    with pytest.raises(RuntimeError, match="synthetic persistent slice failure"):
        materialize_mixed_snapshot(
            plan, organizer, flywheel, tmp_path / "failed-persistent",
            persistent_staging_root=staging,
        )
    assert (staging / "state.json").is_file()
    assert (staging / "work").is_dir()


def test_mix_persistent_restart_reuses_verified_parquet_and_matches_fresh_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    organizer = _prepared_source(tmp_path / "organizer", kind="organizer")
    flywheel = _prepared_source(tmp_path / "flywheel", kind="flywheel", grade="A")
    plan = build_mix_plan(organizer, flywheel, seed=20260813)
    staging = tmp_path / "resume"
    destination = tmp_path / "resumed"
    original_slice = __import__("lehome_train.flywheel.mix", fromlist=["_copy_selected_video"])._copy_selected_video
    original_write = pq.write_table
    writes = 0

    def fail_slice(source: Path, destination: Path, *, start: int, stop: int) -> None:
        raise RuntimeError("interrupted after parquet")

    monkeypatch.setattr("lehome_train.flywheel.mix._copy_selected_video", fail_slice)
    with pytest.raises(RuntimeError, match="interrupted after parquet"):
        materialize_mixed_snapshot(plan, organizer, flywheel, destination, persistent_staging_root=staging)

    def count_write(*args: object, **kwargs: object) -> None:
        nonlocal writes
        if len(args) > 1 and "data/chunk-" in str(args[1]):
            writes += 1
        original_write(*args, **kwargs)

    monkeypatch.setattr("lehome_train.flywheel.mix._copy_selected_video", original_slice)
    monkeypatch.setattr("lehome_train.flywheel.mix.pq.write_table", count_write)
    materialize_mixed_snapshot(plan, organizer, flywheel, destination, persistent_staging_root=staging, video_workers=1)
    assert 0 < writes < len(plan.selections)
    monkeypatch.setattr("lehome_train.flywheel.mix.pq.write_table", original_write)
    fresh = tmp_path / "fresh"
    materialize_mixed_snapshot(plan, organizer, flywheel, fresh, video_workers=4)
    assert artifact_identities(destination, exclude={"manifest.json"}) == artifact_identities(fresh, exclude={"manifest.json"})
    assert not staging.exists()


def test_mix_persistent_resume_rejects_tampered_state_and_unexpected_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    organizer = _prepared_source(tmp_path / "organizer", kind="organizer")
    flywheel = _prepared_source(tmp_path / "flywheel", kind="flywheel", grade="A")
    plan = build_mix_plan(organizer, flywheel, seed=20260813)
    staging = tmp_path / "resume"
    monkeypatch.setattr("lehome_train.flywheel.mix._copy_selected_video", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stop")))
    with pytest.raises(RuntimeError, match="stop"):
        materialize_mixed_snapshot(plan, organizer, flywheel, tmp_path / "destination", persistent_staging_root=staging)
    (staging / "unexpected").write_text("no", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected"):
        materialize_mixed_snapshot(plan, organizer, flywheel, tmp_path / "destination", persistent_staging_root=staging)


@pytest.mark.parametrize("node", ("symlink", "fifo"))
def test_mix_persistent_resume_rejects_symlink_and_special_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, node: str,
) -> None:
    organizer = _prepared_source(tmp_path / "organizer", kind="organizer")
    flywheel = _prepared_source(tmp_path / "flywheel", kind="flywheel", grade="A")
    plan = build_mix_plan(organizer, flywheel, seed=20260813)
    staging = tmp_path / "resume"
    monkeypatch.setattr("lehome_train.flywheel.mix._copy_selected_video", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stop")))
    with pytest.raises(RuntimeError, match="stop"):
        materialize_mixed_snapshot(plan, organizer, flywheel, tmp_path / "destination", persistent_staging_root=staging)
    bad = staging / "work" / "data" / "chunk-000" / "bad"
    bad.parent.mkdir(parents=True, exist_ok=True)
    if node == "symlink":
        bad.symlink_to(staging / "state.json")
    else:
        os.mkfifo(bad)
    with pytest.raises(ValueError, match="unexpected"):
        materialize_mixed_snapshot(plan, organizer, flywheel, tmp_path / "destination", persistent_staging_root=staging)


def test_persistent_lock_excludes_second_owner_and_releases_after_process_death(tmp_path: Path) -> None:
    lock_path = tmp_path / "lock"
    with _PersistentLock(lock_path):
        with pytest.raises(RuntimeError, match="locked"):
            with _PersistentLock(lock_path):
                pass
    # flock ownership is tied to the process descriptor, rather than an O_EXCL
    # sentinel: a killed process releases it for the next recovery attempt.
    child = subprocess.Popen(("python3", "-c", "import fcntl,time; f=open(__import__('sys').argv[1], 'a+b'); fcntl.flock(f, fcntl.LOCK_EX); time.sleep(60)", str(lock_path)))
    try:
        for _ in range(50):
            try:
                with _PersistentLock(lock_path):
                    pass
            except RuntimeError:
                break
            time.sleep(.01)
        else:
            pytest.fail("child did not acquire advisory lock")
        child.kill(); child.wait(timeout=5)
        with _PersistentLock(lock_path):
            pass
    finally:
        if child.poll() is None:
            child.kill(); child.wait(timeout=5)


def test_persistent_lock_rejects_symlink_and_fifo(tmp_path: Path) -> None:
    target = tmp_path / "target"; target.write_text("x", encoding="utf-8")
    link = tmp_path / "link"; link.symlink_to(target)
    with pytest.raises(ValueError, match="regular"):
        with _PersistentLock(link):
            pass
    fifo = tmp_path / "fifo"; os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular"):
        with _PersistentLock(fifo):
            pass


def test_mix_persistent_resume_rejects_state_plan_source_and_code_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    organizer = _prepared_source(tmp_path / "organizer", kind="organizer")
    flywheel = _prepared_source(tmp_path / "flywheel", kind="flywheel", grade="A")
    plan = build_mix_plan(organizer, flywheel, seed=20260813)
    staging = tmp_path / "resume"
    monkeypatch.setattr("lehome_train.flywheel.mix._copy_selected_video", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stop")))
    with pytest.raises(RuntimeError, match="stop"):
        materialize_mixed_snapshot(plan, organizer, flywheel, tmp_path / "destination", persistent_staging_root=staging)
    state = json.loads((staging / "state.json").read_text(encoding="utf-8"))
    state["materializer_sha256"] = "0" * 64
    (staging / "state.json").write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="plan, source, or code"):
        materialize_mixed_snapshot(plan, organizer, flywheel, tmp_path / "destination", persistent_staging_root=staging)


def test_mix_persistent_state_binds_destination_and_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    organizer = _prepared_source(tmp_path / "organizer", kind="organizer")
    flywheel = _prepared_source(tmp_path / "flywheel", kind="flywheel", grade="A")
    plan, staging = build_mix_plan(organizer, flywheel, seed=20260813), tmp_path / "resume"
    monkeypatch.setattr("lehome_train.flywheel.mix._copy_selected_video", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stop")))
    with pytest.raises(RuntimeError, match="stop"):
        materialize_mixed_snapshot(plan, organizer, flywheel, tmp_path / "one", persistent_staging_root=staging, persistent_source_evidence={"source": "one"})
    with pytest.raises(ValueError, match="plan, source, or code"):
        materialize_mixed_snapshot(plan, organizer, flywheel, tmp_path / "two", persistent_staging_root=staging, persistent_source_evidence={"source": "two"})


def test_mix_persistent_initialization_is_idempotent_after_prestate_crash(tmp_path: Path) -> None:
    organizer = _prepared_source(tmp_path / "organizer", kind="organizer")
    flywheel = _prepared_source(tmp_path / "flywheel", kind="flywheel", grade="A")
    plan, staging = build_mix_plan(organizer, flywheel, seed=20260813), tmp_path / "resume"
    (staging / "work").mkdir(parents=True); (staging / "receipts").mkdir()
    materialize_mixed_snapshot(plan, organizer, flywheel, tmp_path / "destination", persistent_staging_root=staging)
    assert not staging.exists()


def test_mix_persistent_stateless_root_rejects_unrelated_file_without_deleting_it(tmp_path: Path) -> None:
    organizer = _prepared_source(tmp_path / "organizer", kind="organizer")
    flywheel = _prepared_source(tmp_path / "flywheel", kind="flywheel", grade="A")
    plan, staging = build_mix_plan(organizer, flywheel, seed=20260813), tmp_path / "resume"
    staging.mkdir(); unrelated = staging / "do-not-delete"; unrelated.write_text("user", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected"):
        materialize_mixed_snapshot(plan, organizer, flywheel, tmp_path / "destination", persistent_staging_root=staging)
    assert unrelated.read_text(encoding="utf-8") == "user"


def test_mix_persistent_stateless_retry_removes_only_owned_state_atomic_temp(tmp_path: Path) -> None:
    organizer = _prepared_source(tmp_path / "organizer", kind="organizer")
    flywheel = _prepared_source(tmp_path / "flywheel", kind="flywheel", grade="A")
    plan, staging = build_mix_plan(organizer, flywheel, seed=20260813), tmp_path / "resume"
    staging.mkdir(); owned = staging / ".state.json.kill-window.tmp"; owned.write_text("partial", encoding="utf-8")
    materialize_mixed_snapshot(plan, organizer, flywheel, tmp_path / "destination", persistent_staging_root=staging)
    assert not owned.exists() and not staging.exists()


def test_mix_persistent_terminal_retry_accepts_only_exact_sealed_destination(tmp_path: Path) -> None:
    organizer = _prepared_source(tmp_path / "organizer", kind="organizer")
    flywheel = _prepared_source(tmp_path / "flywheel", kind="flywheel", grade="A")
    plan, staging, destination = build_mix_plan(organizer, flywheel, seed=20260813), tmp_path / "resume", tmp_path / "destination"
    materialize_mixed_snapshot(plan, organizer, flywheel, destination, persistent_staging_root=staging, persistent_source_evidence={"seed": 1})
    assert materialize_mixed_snapshot(plan, organizer, flywheel, destination, persistent_staging_root=staging, persistent_source_evidence={"seed": 1})["resumed_after_terminal_cleanup"] is True
    with pytest.raises(ValueError, match="terminal destination"):
        materialize_mixed_snapshot(plan, organizer, flywheel, destination, persistent_staging_root=tmp_path / "other", persistent_source_evidence={"seed": 2})


def test_mix_materializer_identity_binds_all_behavior_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    import lehome_train.data.convert as convert
    import lehome_train.data.inspect as inspect
    import lehome_train.data.mapping as mapping
    import lehome_train.data.split as split
    import lehome_train.groot.modality as modality
    import lehome_train.models as models
    from lehome_train.flywheel.mix import _mix_materializer_identity
    original = __import__("lehome_train.flywheel.mix", fromlist=["sha256_file"]).sha256_file
    baseline = _mix_materializer_identity()
    for module in (convert, inspect, mapping, split, modality, models):
        monkeypatch.setattr("lehome_train.flywheel.mix.sha256_file", lambda path, original=original, module=module: "f" * 64 if Path(path) == Path(module.__file__) else original(path))
        assert _mix_materializer_identity() != baseline
        monkeypatch.undo()


def test_validation_reservation_scales_with_bounded_state_for_23k_ranges() -> None:
    from lehome_train.flywheel.mix import _Chunk, _reserve_validation_chunks
    chunks = [
        _Chunk(
            source_kind="organizer" if index % 2 == 0 else "flywheel", source_root=Path("/source"),
            source_manifest_sha256=("a" if index % 2 == 0 else "b") * 64, source_revision="c" * 40,
            episode_id=str(index), start=0, stop=ACTION_HORIZON, frame_ids=tuple(str(frame) for frame in range(ACTION_HORIZON)),
            raw_manifest_sha256=("d" if index % 2 == 0 else "e") * 64, raw_episode_id=str(index),
            raw_frame_start=0, raw_frame_stop=ACTION_HORIZON, raw_frame_ids=tuple(str(frame) for frame in range(ACTION_HORIZON)), quality_grade=None,
        )
        for index in range(23_089)
    ]
    selected = _reserve_validation_chunks(chunks, 2_308, seed=20260813)
    assert len(selected) == 2_308
    selected_episodes = {item.raw_episode_id for item in selected}
    assert {item.source_kind for item in chunks if item.raw_episode_id not in selected_episodes} == {"organizer", "flywheel"}


def test_validation_reservation_keeps_both_kinds_trainable_when_balanced_holdout_exists() -> None:
    from lehome_train.flywheel.mix import _Chunk, _reserve_validation_chunks
    chunks = [
        _Chunk("organizer" if index < 2 else "flywheel", Path("/source"), ("a" if index < 2 else "b") * 64, "c" * 40,
               str(index), 0, ACTION_HORIZON, tuple(str(frame) for frame in range(ACTION_HORIZON)),
               ("d" if index < 2 else "e") * 64, str(index), 0, ACTION_HORIZON,
               tuple(str(frame) for frame in range(ACTION_HORIZON)), None)
        for index in range(4)
    ]
    selected = _reserve_validation_chunks(chunks, 2, seed=5)
    assert [item.source_kind for item in selected].count("organizer") == 1
    assert [item.source_kind for item in selected].count("flywheel") == 1


def test_mix_persistent_resume_regenerates_semantically_wrong_parquet_and_invalid_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    organizer = _prepared_source(tmp_path / "organizer", kind="organizer")
    flywheel = _prepared_source(tmp_path / "flywheel", kind="flywheel", grade="A")
    plan = build_mix_plan(organizer, flywheel, seed=20260813)
    staging, destination = tmp_path / "resume", tmp_path / "destination"
    monkeypatch.setattr("lehome_train.flywheel.mix._seal_mixed_work", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("after jobs")))
    with pytest.raises(RuntimeError, match="after jobs"):
        materialize_mixed_snapshot(plan, organizer, flywheel, destination, persistent_staging_root=staging)
    parquet = next((staging / "work" / "data").rglob("*.parquet"))
    video = next((staging / "work" / "videos").rglob("*.mp4"))
    table = pq.read_table(parquet).set_column(0, "observation.state", pa.array([[99.0] * 12] * ACTION_HORIZON, type=pa.list_(pa.float32(), 12)))
    pq.write_table(table, parquet, compression="zstd")
    video.write_bytes(b"not an mp4")
    # Model a fully written receipt for corrupt media; the semantic/ffprobe
    # validators, not merely receipt presence, decide whether it is reusable.
    from lehome_train.io import sha256_file
    for receipt in (staging / "receipts").glob("*.json"):
        body = json.loads(receipt.read_text(encoding="utf-8"))
        candidate = staging / "work" / body["relative_path"]
        if candidate in (parquet, video):
            body["sha256"], body["byte_size"] = sha256_file(candidate), candidate.stat().st_size
            receipt.write_text(json.dumps(body), encoding="utf-8")
    monkeypatch.undo()
    materialize_mixed_snapshot(plan, organizer, flywheel, destination, persistent_staging_root=staging)
    _validate_output_video(next(destination.rglob("*.mp4")), expected_frame_count=ACTION_HORIZON, expected_fps=30)
    assert pq.read_table(destination / parquet.relative_to(staging / "work"))["observation.state"][0].as_py()[0] != 99.0


def test_mix_persistent_resume_repairs_only_exact_post_promotion_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    organizer = _prepared_source(tmp_path / "organizer", kind="organizer")
    flywheel = _prepared_source(tmp_path / "flywheel", kind="flywheel", grade="A")
    plan = build_mix_plan(organizer, flywheel, seed=20260813)
    staging, destination = tmp_path / "resume", tmp_path / "destination"
    receipt = destination.with_name(destination.name + ".generation.json")
    original = atomic_write_json
    def fail_receipt(path: Path, value: object) -> None:
        if path == receipt:
            raise RuntimeError("lost after promotion")
        original(path, value)
    monkeypatch.setattr("lehome_train.flywheel.mix.atomic_write_json", fail_receipt)
    with pytest.raises(RuntimeError, match="lost after promotion"):
        materialize_mixed_snapshot(plan, organizer, flywheel, destination, persistent_staging_root=staging)
    monkeypatch.setattr("lehome_train.flywheel.mix.atomic_write_json", original)
    assert destination.is_dir() and staging.exists() and not receipt.exists()
    assert materialize_mixed_snapshot(plan, organizer, flywheel, destination, persistent_staging_root=staging)["resumed_after_promotion"] is True
    verify_generation(destination)


def test_mix_persistent_resume_rejects_sealed_destination_plan_or_evidence_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    organizer = _prepared_source(tmp_path / "organizer", kind="organizer")
    flywheel = _prepared_source(tmp_path / "flywheel", kind="flywheel", grade="A")
    plan = build_mix_plan(organizer, flywheel, seed=20260813)
    staging, destination = tmp_path / "resume", tmp_path / "destination"
    receipt = destination.with_name(destination.name + ".generation.json")
    original = atomic_write_json
    monkeypatch.setattr("lehome_train.flywheel.mix.atomic_write_json", lambda path, value: (_ for _ in ()).throw(RuntimeError("receipt interruption")) if path == receipt else original(path, value))
    with pytest.raises(RuntimeError, match="receipt interruption"):
        materialize_mixed_snapshot(plan, organizer, flywheel, destination, persistent_staging_root=staging, persistent_source_evidence={"seed": 1})
    monkeypatch.setattr("lehome_train.flywheel.mix.atomic_write_json", original)
    # A valid sealed receipt from a different evidence binding must not consume
    # this staging root, even though the selected files remain byte-valid.
    from lehome_train.flywheel.mix import _generation_receipt
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8")); manifest["persistent_source_evidence"] = {"seed": 2}; (destination / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    original(destination.with_name(destination.name + ".generation.json"), _generation_receipt(destination))
    with pytest.raises(ValueError, match="sealed destination"):
        materialize_mixed_snapshot(plan, organizer, flywheel, destination, persistent_staging_root=staging, persistent_source_evidence={"seed": 1})
    assert staging.exists()
    with pytest.raises(ValueError, match="plan, source, or code"):
        materialize_mixed_snapshot(build_mix_plan(organizer, flywheel, seed=20260814), organizer, flywheel, destination, persistent_staging_root=staging, persistent_source_evidence={"seed": 1})


def test_mix_persistent_resume_rejects_arbitrary_post_promotion_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    organizer = _prepared_source(tmp_path / "organizer", kind="organizer")
    flywheel = _prepared_source(tmp_path / "flywheel", kind="flywheel", grade="A")
    plan = build_mix_plan(organizer, flywheel, seed=20260813)
    staging, destination = tmp_path / "resume", tmp_path / "destination"
    monkeypatch.setattr("lehome_train.flywheel.mix._copy_selected_video", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stop")))
    with pytest.raises(RuntimeError, match="stop"):
        materialize_mixed_snapshot(plan, organizer, flywheel, destination, persistent_staging_root=staging)
    destination.mkdir(); (destination / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="destination"):
        materialize_mixed_snapshot(plan, organizer, flywheel, destination, persistent_staging_root=staging)


def test_mix_persistent_resume_discards_stale_postprocessing_before_reseal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    organizer = _prepared_source(tmp_path / "organizer", kind="organizer")
    flywheel = _prepared_source(tmp_path / "flywheel", kind="flywheel", grade="A")
    plan = build_mix_plan(organizer, flywheel, seed=20260813)
    staging, destination = tmp_path / "resume", tmp_path / "destination"
    monkeypatch.setattr("lehome_train.flywheel.mix._seal_mixed_work", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("after jobs")))
    with pytest.raises(RuntimeError, match="after jobs"):
        materialize_mixed_snapshot(plan, organizer, flywheel, destination, persistent_staging_root=staging)
    stale = staging / "work" / "meta" / "validation.json"
    stale.parent.mkdir(parents=True); stale.write_text("stale", encoding="utf-8")
    monkeypatch.undo()
    materialize_mixed_snapshot(plan, organizer, flywheel, destination, persistent_staging_root=staging)
    assert not (destination / "meta" / "validation.json").exists()
    verify_generation(destination)


def test_mix_receipt_failure_removes_just_promoted_destination_and_partial_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    organizer = _prepared_source(tmp_path / "organizer", kind="organizer")
    flywheel = _prepared_source(tmp_path / "flywheel", kind="flywheel", grade="A")
    plan = build_mix_plan(organizer, flywheel, seed=20260813)
    destination = tmp_path / "receipt-failed"
    receipt = destination.with_name(destination.name + ".generation.json")
    original_atomic_write_json = atomic_write_json

    def fail_receipt_write(path: Path, value: object) -> None:
        if path == receipt:
            path.write_text("partial receipt", encoding="utf-8")
            raise RuntimeError("synthetic receipt write failure")
        original_atomic_write_json(path, value)

    monkeypatch.setattr("lehome_train.flywheel.mix.atomic_write_json", fail_receipt_write)

    with pytest.raises(RuntimeError, match="synthetic receipt write failure"):
        materialize_mixed_snapshot(plan, organizer, flywheel, destination)

    assert not destination.exists()
    assert not receipt.exists()
    assert not list(tmp_path.glob(".receipt-failed.*.tmp"))


def test_generation_receipt_binds_exact_70_30_mix_and_artifacts(tmp_path: Path) -> None:
    organizer = _prepared_source(tmp_path / "organizer", kind="organizer")
    grade_a = _prepared_source(tmp_path / "grade-a", kind="flywheel", grade="A", episodes=1)
    grade_b = _prepared_source(tmp_path / "grade-b", kind="flywheel", grade="B", episodes=1)
    plan = build_mix_plan(organizer, [grade_a, grade_b], seed=20260812)
    result = materialize_mixed_snapshot(plan, organizer, [grade_a, grade_b], tmp_path / "generation")

    receipt = load_generation_receipt(result["path"])

    assert receipt["organizer_training_frames"] * 3 == receipt["rft_training_frames"] * 7
    assert receipt["sealed"] is True
    assert len(receipt["output_manifest_sha256"]) == 64
    verify_generation(result["path"])


def test_mix_accepts_canonical_autonomous_rft_snapshot_not_expert_provenance(
    tmp_path: Path,
) -> None:
    """The released RFT aggregate uses policy trajectories and rft-selection."""
    organizer = _prepared_source(tmp_path / "organizer", kind="organizer", episodes=2)
    rft = _prepared_source(tmp_path / "rft", kind="flywheel", grade="A", episodes=2, action_source="policy")
    provenance = rft / "meta" / "materialization-provenance.json"
    provenance.unlink()
    manifest = json.loads((rft / "manifest.json").read_text(encoding="utf-8"))
    manifest.update({
        "source_format": "verified_flywheel_rft_release",
        "source_repository": "ryanjin333/lehome-groot-n17-data",
        "source_revision": "e6cd1c182514c15271c805d03a646e7a4f95b17c",
        "source_release_id": "b" * 64,
    })
    for camera in manifest["camera_schema"]:
        camera.pop("target_modality", None)
    # Rebuild the listed artifact set after replacing provenance with the
    # canonical aggregate selection artifact.
    atomic_write_json(rft / "meta" / "rft-selection.json", {
        "schema_version": 1,
        "source_repository": "ryanjin333/lehome-groot-n17-data",
        "source_revision": "e6cd1c182514c15271c805d03a646e7a4f95b17c",
        "release_id": "b" * 64,
        "action_horizon": 16,
        "excluded_public_unseen": 0,
        "excluded_failed": 0,
        "episodes": [
            {"episode_index": index, "raw_episode_id": f"rft-{index}", "raw_manifest_sha256": ("c" if index == 0 else "d") * 64, "frame_count": 16, "valid_window_count": 1, "category": "top_long"}
            for index in range(2)
        ],
    })
    artifacts = artifact_identities(rft, exclude={"manifest.json"})
    manifest["output_artifacts"] = artifacts
    manifest["output_manifest_sha256"] = canonical_json_sha256(artifacts)
    atomic_write_json(rft / "manifest.json", manifest)

    plan = build_mix_plan(organizer, rft, seed=5)
    materialize_mixed_snapshot(plan, organizer, rft, tmp_path / "mixed-rft")

    assert plan.organizer_training_frames * 3 == plan.flywheel_training_frames * 7
    assert {item.source_kind for item in plan.selections} == {"organizer", "flywheel"}


def test_mix_materializes_nonzero_canonical_rft_chunk_with_exact_raw_lineage(tmp_path: Path) -> None:
    """A selected policy chunk is only one bounded slice of its raw trajectory."""

    import lehome_train.flywheel.mix as mix

    organizer = _prepared_source(tmp_path / "organizer", kind="organizer", episodes=2)
    rft = _prepared_source(
        tmp_path / "rft",
        kind="flywheel",
        grade="A",
        episodes=2,
        frames=32,
        episode_start=70,
        action_source="policy",
    )
    (rft / "meta" / "materialization-provenance.json").unlink()
    manifest = json.loads((rft / "manifest.json").read_text(encoding="utf-8"))
    manifest.update({
        "source_format": "verified_flywheel_rft_release",
        "source_repository": "ryanjin333/lehome-groot-n17-data",
        "source_revision": "e6cd1c182514c15271c805d03a646e7a4f95b17c",
        "source_release_id": "b" * 64,
    })
    for camera in manifest["camera_schema"]:
        camera.pop("target_modality", None)
    atomic_write_json(rft / "meta" / "rft-selection.json", {
        "schema_version": 1,
        "source_repository": "ryanjin333/lehome-groot-n17-data",
        "source_revision": "e6cd1c182514c15271c805d03a646e7a4f95b17c",
        "release_id": "b" * 64,
        "action_horizon": 16,
        "excluded_public_unseen": 0,
        "excluded_failed": 0,
        "episodes": [
            {"episode_index": 70 + index, "raw_episode_id": f"rft-{70 + index}", "raw_manifest_sha256": ("c" if index == 0 else "d") * 64, "frame_count": 326, "valid_window_count": 311, "category": "top_long"}
            for index in range(2)
        ],
    })
    artifacts = artifact_identities(rft, exclude={"manifest.json"})
    manifest["output_artifacts"] = artifacts
    manifest["output_manifest_sha256"] = canonical_json_sha256(artifacts)
    atomic_write_json(rft / "manifest.json", manifest)

    plan = build_mix_plan(organizer, rft, seed=5)
    selected = next(item for item in plan.selections if item.source_kind == "flywheel" and item.frame_start == ACTION_HORIZON)
    assert selected.source_episode_id in {"70", "71"}
    assert (selected.raw_frame_start, selected.raw_frame_stop, selected.raw_frame_ids) == (
        ACTION_HORIZON,
        2 * ACTION_HORIZON,
        tuple(str(index) for index in range(ACTION_HORIZON, 2 * ACTION_HORIZON)),
    )

    materialize_mixed_snapshot(plan, organizer, rft, tmp_path / "mixed")

    payload = plan.to_dict()
    tampered = next(
        item for item in payload["selected_frame_ranges"]
        if item["source_kind"] == "flywheel" and item["frame_start"] == ACTION_HORIZON
    )
    tampered.update({
        "raw_frame_start": 2 * ACTION_HORIZON,
        "raw_frame_stop": 3 * ACTION_HORIZON,
        "raw_frame_ids": [str(index) for index in range(2 * ACTION_HORIZON, 3 * ACTION_HORIZON)],
    })
    payload["sha256"] = canonical_json_sha256({key: value for key, value in payload.items() if key != "sha256"})
    with pytest.raises(ValueError, match="raw lineage no longer matches"):
        materialize_mixed_snapshot(mix._plan_from_payload(payload), organizer, rft, tmp_path / "tampered")


def test_generation_changes_after_seal_are_rejected(tmp_path: Path) -> None:
    organizer = _prepared_source(tmp_path / "organizer", kind="organizer")
    grade_a = _prepared_source(tmp_path / "grade-a", kind="flywheel", grade="A", episodes=1)
    grade_b = _prepared_source(tmp_path / "grade-b", kind="flywheel", grade="B", episodes=1)
    plan = build_mix_plan(organizer, [grade_a, grade_b], seed=20260812)
    result = materialize_mixed_snapshot(plan, organizer, [grade_a, grade_b], tmp_path / "generation")
    manifest = Path(result["path"]) / "manifest.json"
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="sealed generation"):
        verify_generation(result["path"])


def test_mix_reserves_validation_source_ranges_before_training_oversampling(tmp_path: Path) -> None:
    """A held-out frame range must never be recycled by train oversampling."""

    organizer = _prepared_source(tmp_path / "organizer", kind="organizer", episodes=2)
    grade_a = _prepared_source(tmp_path / "grade-a", kind="flywheel", grade="A", episodes=1)
    grade_b = _prepared_source(tmp_path / "grade-b", kind="flywheel", grade="B", episodes=1)

    # This exact small pool previously selected organizer episode 1 for both
    # validation and repeated training slots.
    plan = build_mix_plan(organizer, [grade_a, grade_b], seed=1)
    train_source_frames = {
        (item.source_manifest_sha256, item.source_episode_id, frame_id)
        for item in plan.selections
        if item.split == "train"
        for frame_id in item.source_frame_ids
    }
    validation_source_frames = {
        (item.source_manifest_sha256, item.source_episode_id, frame_id)
        for item in plan.selections
        if item.split == "validation"
        for frame_id in item.source_frame_ids
    }

    assert train_source_frames.isdisjoint(validation_source_frames)
    # The reduced source pools still support deterministic oversampling.
    assert len([item for item in plan.selections if item.split == "train" and item.source_kind == "organizer"]) == 7
    assert len([item for item in plan.selections if item.split == "train" and item.source_kind == "flywheel"]) == 3


def test_mix_rejects_hash_valid_plan_with_cross_split_source_frame_leakage(tmp_path: Path) -> None:
    organizer = _prepared_source(tmp_path / "organizer", kind="organizer", episodes=2)
    grade_a = _prepared_source(tmp_path / "grade-a", kind="flywheel", grade="A", episodes=1)
    grade_b = _prepared_source(tmp_path / "grade-b", kind="flywheel", grade="B", episodes=1)
    payload = build_mix_plan(organizer, [grade_a, grade_b], seed=1).to_dict()
    train_range = next(item for item in payload["selected_frame_ranges"] if item["split"] == "train")
    validation_range = next(item for item in payload["selected_frame_ranges"] if item["split"] == "validation")
    validation_range.update({key: value for key, value in train_range.items() if key not in {"destination_episode_id", "split"}})
    payload["sha256"] = canonical_json_sha256({key: value for key, value in payload.items() if key != "sha256"})

    with pytest.raises(ValueError, match="source frames overlap"):
        validate_mix_plan_payload(payload)


def test_mix_rejects_hash_valid_plan_that_splits_raw_lineage_with_new_local_ids(tmp_path: Path) -> None:
    organizer = _prepared_source(tmp_path / "organizer", kind="organizer", episodes=2)
    grade_a = _prepared_source(tmp_path / "grade-a", kind="flywheel", grade="A", episodes=1)
    grade_b = _prepared_source(tmp_path / "grade-b", kind="flywheel", grade="B", episodes=1)
    payload = build_mix_plan(organizer, [grade_a, grade_b], seed=1).to_dict()
    train_range = next(item for item in payload["selected_frame_ranges"] if item["split"] == "train")
    validation_range = next(item for item in payload["selected_frame_ranges"] if item["split"] == "validation")
    validation_range.update({
        key: train_range[key]
        for key in ("raw_manifest_sha256", "raw_episode_id", "raw_frame_start", "raw_frame_stop", "raw_frame_ids")
    })
    payload["sha256"] = canonical_json_sha256({key: value for key, value in payload.items() if key != "sha256"})

    with pytest.raises(ValueError, match="raw frames overlap"):
        validate_mix_plan_payload(payload)


def test_mix_rejects_a_pool_that_cannot_supply_disjoint_validation_ranges(tmp_path: Path) -> None:
    organizer = _prepared_source(tmp_path / "organizer", kind="organizer", episodes=1)
    flywheel = _prepared_source(tmp_path / "flywheel", kind="flywheel", grade="A", episodes=1)

    # Training needs both source kinds for 70/30, so holding out either lone
    # range would make the prior behavior leak it back through oversampling.
    with pytest.raises(ValueError, match="too few distinct lineage episodes"):
        build_mix_plan(organizer, flywheel, seed=1)


def test_mix_rejects_tampered_plan_and_source_without_leaving_output(tmp_path: Path) -> None:
    organizer = _prepared_source(tmp_path / "organizer", kind="organizer")
    flywheel = _prepared_source(tmp_path / "flywheel", kind="flywheel", grade="A")
    plan = build_mix_plan(organizer, flywheel, seed=7)
    payload = plan.to_dict()
    payload["selected_frame_ranges"][0]["source_frame_ids"][0] = "tampered"
    with pytest.raises(ValueError, match="hash"):
        validate_mix_plan_payload(payload)
    parquet = flywheel / LEGACY_DATA_PATH.format(episode_chunk=0, episode_index=0)
    table = pq.read_table(parquet)
    pq.write_table(table.set_column(table.schema.get_field_index("action"), "action", pa.array([[999.0] * 12] * ACTION_HORIZON, type=table["action"].type)), parquet, compression="zstd")
    destination = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="artifact hash"):
        materialize_mixed_snapshot(plan, organizer, flywheel, destination)
    assert not destination.exists()


@pytest.mark.parametrize(
    ("grade", "release_stage", "accepted_success", "action_source", "message"),
    [
        ("C", "seen", True, "expert", "grade"),
        ("A", "public_unseen", True, "expert", "holdout"),
        ("A", "seen", False, "expert", "failed"),
        ("A", "seen", True, "hold", "non-expert"),
    ],
)
def test_mix_rejects_ineligible_flywheel_contracts(tmp_path: Path, grade: str, release_stage: str, accepted_success: bool, action_source: str, message: str) -> None:
    organizer = _prepared_source(tmp_path / "organizer", kind="organizer")
    flywheel = _prepared_source(tmp_path / "bad", kind="flywheel", grade=grade, episodes=1, release_stage=release_stage, accepted_success=accepted_success, action_source=action_source)

    with pytest.raises(ValueError, match=message):
        build_mix_plan(organizer, flywheel, seed=1)


def test_mix_rejects_historical_task_1_materialized_contract_without_canonical_features(tmp_path: Path) -> None:
    organizer = _prepared_source(tmp_path / "organizer", kind="organizer")
    raw_a = _raw_episode(tmp_path / "raw-a", grade="A")
    raw_b = _raw_episode(tmp_path / "raw-b", grade="B")
    from lehome_train.flywheel.materialize import materialize_episode

    grade_a = tmp_path / "task1-a"
    grade_b = tmp_path / "task1-b"
    materialize_episode(raw_a, grade_a)
    materialize_episode(raw_b, grade_b)

    plan = build_mix_plan(organizer, [grade_a, grade_b], seed=13)
    with pytest.raises(ValueError, match="feature contract"):
        materialize_mixed_snapshot(plan, organizer, [grade_a, grade_b], tmp_path / "mixed")


def test_mix_keeps_overlapping_real_materializer_windows_in_one_split(tmp_path: Path) -> None:
    """Adjacent Task-1 windows share raw frames even though local IDs differ."""

    organizer = _prepared_source(tmp_path / "organizer", kind="organizer", episodes=2)
    from lehome_train.flywheel.materialize import materialize_episode

    materialized: list[Path] = []
    for grade in ("A", "B"):
        output = tmp_path / f"task1-{grade.lower()}"
        materialize_episode(_raw_episode(tmp_path / f"raw-{grade.lower()}", grade=grade, frames=21), output)
        materialized.append(output)

    first_ranges = json.loads((materialized[0] / "meta" / "materialization-provenance.json").read_text(encoding="utf-8"))["selected_frame_ranges"]
    assert [(item["frame_start"], item["frame_stop"]) for item in first_ranges] == [(4, 20), (5, 21)]

    plan = build_mix_plan(organizer, materialized, seed=1)
    splits_by_raw_episode: dict[tuple[str, str], set[str]] = {}
    for item in plan.selections:
        if item.source_kind != "flywheel":
            continue
        key = (item.raw_manifest_sha256, item.raw_episode_id)
        splits_by_raw_episode.setdefault(key, set()).add(item.split)

    assert all(len(splits) == 1 for splits in splits_by_raw_episode.values())
