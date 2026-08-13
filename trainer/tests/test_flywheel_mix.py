from __future__ import annotations

import json
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


def _prepared_source(root: Path, *, kind: str, grade: str | None = None, episodes: int = 2, release_stage: str = "seen", accepted_success: bool = True, action_source: str = "expert") -> Path:
    """Small real prepared-v2 fixture with parquet and all required MP4 streams."""

    for episode in range(episodes):
        base = episode * ACTION_HORIZON
        path = root / LEGACY_DATA_PATH.format(episode_chunk=0, episode_index=episode)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({
            "observation.state": pa.array([[float(base + frame)] * 12 for frame in range(ACTION_HORIZON)], type=pa.list_(pa.float32(), 12)),
            "action": pa.array([[float(base + frame + 1)] * 12 for frame in range(ACTION_HORIZON)], type=pa.list_(pa.float32(), 12)),
            "timestamp": pa.array([frame / 30 for frame in range(ACTION_HORIZON)], type=pa.float32()),
            "frame_index": pa.array(range(ACTION_HORIZON), type=pa.int64()),
            "episode_index": pa.array([episode] * ACTION_HORIZON, type=pa.int64()),
            "index": pa.array(range(base, base + ACTION_HORIZON), type=pa.int64()),
            "task_index": pa.array([0] * ACTION_HORIZON, type=pa.int64()),
        }), path, compression="zstd")
        for camera in ("top_rgb", "left_rgb", "right_rgb"):
            _video(root / LEGACY_VIDEO_PATH.format(episode_chunk=0, episode_index=episode, video_key=camera), frames=ACTION_HORIZON)
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    atomic_write_json(meta / "info.json", {
        "codebase_version": "v2.1", "robot_type": "dual_so101_follower", "total_episodes": episodes,
        "total_frames": episodes * ACTION_HORIZON, "total_tasks": 1, "total_videos": episodes * 3,
        "total_chunks": 1, "chunks_size": 1000, "fps": 30, "data_path": LEGACY_DATA_PATH,
        "video_path": LEGACY_VIDEO_PATH, "features": {},
    })
    (meta / "episodes.jsonl").write_text("".join(json.dumps({"episode_index": episode, "length": ACTION_HORIZON, "task_index": 0}) + "\n" for episode in range(episodes)), encoding="utf-8")
    (meta / "episodes_stats.jsonl").write_text("".join(json.dumps({"episode_index": episode, "stats": {}}) + "\n" for episode in range(episodes)), encoding="utf-8")
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
        "frame_count": episodes * ACTION_HORIZON, "episode_count": episodes, "fps": 30,
        "train_episode_ids": [str(episode) for episode in range(episodes)], "validation_episode_ids": [],
        "fixed_language_instruction": "fold the garment on the table",
        "future_actions": {"horizon": ACTION_HORIZON, "loader_allow_padding": False},
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
            {"episode_index": index, "raw_episode_id": f"rft-{index}", "raw_manifest_sha256": ("c" if index == 0 else "d") * 64, "frame_count": 32, "valid_window_count": 17, "category": "top_long"}
            for index in range(2)
        ],
    })
    artifacts = artifact_identities(rft, exclude={"manifest.json"})
    manifest["output_artifacts"] = artifacts
    manifest["output_manifest_sha256"] = canonical_json_sha256(artifacts)
    atomic_write_json(rft / "manifest.json", manifest)

    plan = build_mix_plan(organizer, rft, seed=5)

    assert plan.organizer_training_frames * 3 == plan.flywheel_training_frames * 7
    assert {item.source_kind for item in plan.selections} == {"organizer", "flywheel"}


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


def test_mix_consumes_the_real_task_1_materialized_contract(tmp_path: Path) -> None:
    organizer = _prepared_source(tmp_path / "organizer", kind="organizer")
    raw_a = _raw_episode(tmp_path / "raw-a", grade="A")
    raw_b = _raw_episode(tmp_path / "raw-b", grade="B")
    from lehome_train.flywheel.materialize import materialize_episode

    grade_a = tmp_path / "task1-a"
    grade_b = tmp_path / "task1-b"
    materialize_episode(raw_a, grade_a)
    materialize_episode(raw_b, grade_b)

    plan = build_mix_plan(organizer, [grade_a, grade_b], seed=13)
    result = materialize_mixed_snapshot(plan, organizer, [grade_a, grade_b], tmp_path / "mixed")

    assert result["validation"]["valid"] is True
    assert set(plan.raw_manifest_hashes) == {
        json.loads((grade_a / "meta" / "materialization-provenance.json").read_text(encoding="utf-8"))["raw_manifest_sha256"],
        json.loads((grade_b / "meta" / "materialization-provenance.json").read_text(encoding="utf-8"))["raw_manifest_sha256"],
    }


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
