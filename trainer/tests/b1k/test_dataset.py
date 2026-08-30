from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from lehome_train.io import canonical_json_sha256

from lehome_train.b1k.dataset import MaterializedTrainingManifest, RGB_CAMERA_KEYS, TrainingManifest, build_training_manifest, materialize_training_manifest, validate_training_manifest


def _metadata(root: Path) -> None:
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "annotations").mkdir(exist_ok=True)
    info = {
        "total_tasks": 100,
        "total_episodes": 20_000,
        "robot_type": "R1Pro",
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            **{key: {"dtype": "video"} for key in RGB_CAMERA_KEYS},
            "observation.state": {"dtype": "float32", "shape": [61]},
            "action": {"dtype": "float32", "shape": [23]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"task_index": index, "task": f"task {index}"}
                for index in range(100)
            ]
        ),
        root / "meta" / "tasks.parquet",
    )
    (root / "meta" / "tasks.jsonl").write_text(
        "".join(
            json.dumps({"task_index": index, "task_name": f"task {index}", "task": f"task {index}"}) + "\n"
            for index in range(100)
        ),
        encoding="utf-8",
    )
    (root / "meta" / "stats.json").write_text("{}", encoding="utf-8")
    (root / "annotations" / "skill_summary.csv").write_text("task\n", encoding="utf-8")
    (root / "annotations" / "skill_type_summary.csv").write_text("type\n", encoding="utf-8")
    episodes = []
    for task_index in range(100):
        for offset in range(200):
            episode_index = task_index * 200 + offset
            record: dict[str, object] = {
                "episode_index": episode_index,
                "task_index": task_index,
                "demo_index_within_task": offset,
                "raw_episode_id": episode_index + 10,
                "task_instance_id": f"instance-{task_index:04d}",
                "tasks": [f"task {task_index}"],
                "annotation_path": f"annotations/task-{task_index:04d}/episode_{episode_index + 10:08d}.json",
                "data/chunk_index": episode_index // 1_000,
                "data/file_index": episode_index // 100,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }
            for camera_key in RGB_CAMERA_KEYS:
                record[f"videos/{camera_key}/chunk_index"] = episode_index // 1_000
                record[f"videos/{camera_key}/file_index"] = episode_index // 100
            episodes.append(record)
    pq.write_table(
        pa.Table.from_pylist(episodes),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )


@pytest.fixture
def lerobot_v3_metadata(tmp_path: Path) -> Path:
    _metadata(tmp_path)
    return tmp_path


def _rewrite_episodes(root: Path, mutate) -> None:
    path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    records = pq.read_table(path).to_pylist()
    mutate(records)
    pq.write_table(pa.Table.from_pylist(records), path)


def test_manifest_counts_tasks_from_metadata_not_storage_chunks(lerobot_v3_metadata: Path) -> None:
    manifest = build_training_manifest(
        lerobot_v3_metadata,
        repository="behavior-1k/2026-challenge-demos",
        revision="4f50b44796641a4d526a19d9aeadc8aa51e2f2c2",
    )

    payload = manifest.to_dict()
    assert len(payload["tasks"]) == 100
    assert len(payload["episodes"]) == 20_000
    assert {task["demonstrations"] for task in payload["tasks"]} == {200}
    assert payload["source"] == {
        "repository": "behavior-1k/2026-challenge-demos",
        "revision": "4f50b44796641a4d526a19d9aeadc8aa51e2f2c2",
    }
    assert payload["episodes"][0]["task_index"] == 0
    assert payload["episodes"][0]["video_paths"] == {
        key: f"videos/{key}/chunk-000/file-000.mp4" for key in RGB_CAMERA_KEYS
    }
    assert "meta/info.json" in payload["required_files"]
    assert "meta/tasks.parquet" in payload["required_files"]
    assert "meta/tasks.jsonl" in payload["required_files"]
    assert "meta/stats.json" in payload["required_files"]
    assert "meta/episodes/chunk-000/file-000.parquet" in payload["required_files"]
    assert "annotations/skill_summary.csv" in payload["required_files"]
    assert "annotations/skill_type_summary.csv" in payload["required_files"]
    assert "annotations/task-0000/episode_00000010.json" in payload["required_files"]
    assert len(payload["required_files"]) == len(set(payload["required_files"]))
    assert all(item["byte_size"] is None and item["sha256"] is None for item in payload["artifacts"])
    assert payload["fingerprint"] == build_training_manifest(
        lerobot_v3_metadata,
        repository="behavior-1k/2026-challenge-demos",
        revision="4f50b44796641a4d526a19d9aeadc8aa51e2f2c2",
    ).fingerprint


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda records: records.pop(), "task 99.*199.*200"),
        (lambda records: records.__setitem__(1, dict(records[1], episode_index=0)), "duplicate episode_index"),
        (lambda records: records.__setitem__(0, dict(records[0], tasks=[])), "exactly one task"),
        (lambda records: records.__setitem__(0, dict(records[0], tasks=["task 0", "task 1"])), "exactly one task"),
        (lambda records: records[0].pop(f"videos/{RGB_CAMERA_KEYS[0]}/file_index"), "missing camera asset"),
        (lambda records: records.__setitem__(0, dict(records[0], annotation_path="")), "annotation_path"),
        (lambda records: records.__setitem__(0, dict(records[0], annotation_path="../outside.json")), "annotation_path"),
    ],
)
def test_manifest_rejects_invalid_episode_metadata(
    lerobot_v3_metadata: Path, mutation, message: str
) -> None:
    _rewrite_episodes(lerobot_v3_metadata, mutation)
    with pytest.raises(ValueError, match=message):
        build_training_manifest(lerobot_v3_metadata, repository="behavior-1k/2026-challenge-demos", revision="4f50b44796641a4d526a19d9aeadc8aa51e2f2c2")


def test_manifest_rejects_task_count_and_camera_allowlist_drift(lerobot_v3_metadata: Path) -> None:
    tasks_path = lerobot_v3_metadata / "meta" / "tasks.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"task_index": 0, "task": "task 0"}]),
        tasks_path,
    )
    with pytest.raises(ValueError, match="exactly 100"):
        build_training_manifest(lerobot_v3_metadata, repository="behavior-1k/2026-challenge-demos", revision="4f50b44796641a4d526a19d9aeadc8aa51e2f2c2")

    _metadata(lerobot_v3_metadata)
    info_path = lerobot_v3_metadata / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"]["observation.rgb.unapproved"] = {"dtype": "video"}
    info_path.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError, match="camera allowlist"):
        build_training_manifest(lerobot_v3_metadata, repository="behavior-1k/2026-challenge-demos", revision="4f50b44796641a4d526a19d9aeadc8aa51e2f2c2")


def test_manifest_excludes_official_depth_paths(lerobot_v3_metadata: Path) -> None:
    manifest = build_training_manifest(
        lerobot_v3_metadata,
        repository="behavior-1k/2026-challenge-demos",
        revision="4f50b44796641a4d526a19d9aeadc8aa51e2f2c2",
    )
    assert all("depth" not in path for path in manifest.to_dict()["required_files"])


def test_manifest_accepts_official_depth_declarations_but_excludes_them_from_training_selection(lerobot_v3_metadata: Path) -> None:
    info_path = lerobot_v3_metadata / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    for camera in ("left", "right", "top"):
        info["features"][f"observation.depth_linear.{camera}"] = {"dtype": "video"}
    info_path.write_text(json.dumps(info), encoding="utf-8")
    manifest = build_training_manifest(
        lerobot_v3_metadata,
        repository="behavior-1k/2026-challenge-demos",
        revision="4f50b44796641a4d526a19d9aeadc8aa51e2f2c2",
    )
    assert all("depth" not in path for path in manifest.required_files)


def test_manifest_rejects_unofficial_depth_declarations(lerobot_v3_metadata: Path) -> None:
    info_path = lerobot_v3_metadata / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"]["observation.depth_linear.unapproved"] = {"dtype": "video"}
    info_path.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError, match="depth"):
        build_training_manifest(lerobot_v3_metadata, repository="behavior-1k/2026-challenge-demos", revision="4f50b44796641a4d526a19d9aeadc8aa51e2f2c2")


def test_materialized_manifest_serializes_the_exact_fingerprinted_payload() -> None:
    payload = {
        "schema_version": 1,
        "selection_fingerprint": "a" * 64,
        "artifacts": [{"path": "meta/stats.json", "byte_size": 2, "sha256": "b" * 64}],
        "feature_schema": {"action": {"dtype": "float32", "shape": [23]}},
    }
    manifest = MaterializedTrainingManifest(
        selection_fingerprint=payload["selection_fingerprint"], artifacts=tuple(payload["artifacts"]),
        feature_schema=payload["feature_schema"], fingerprint=canonical_json_sha256(payload),
    )
    assert manifest.to_dict() == {**payload, "fingerprint": canonical_json_sha256(payload)}


def test_manifest_records_known_artifact_identity_without_requiring_payload(lerobot_v3_metadata: Path) -> None:
    manifest = build_training_manifest(
        lerobot_v3_metadata, repository="behavior-1k/2026-challenge-demos",
        revision="4f50b44796641a4d526a19d9aeadc8aa51e2f2c2",
        artifact_metadata={"data/chunk-000/file-000.parquet": {"byte_size": 12, "sha256": "a" * 64}},
    )
    assert {item["path"]: item for item in manifest.to_dict()["artifacts"]}["data/chunk-000/file-000.parquet"] == {
        "path": "data/chunk-000/file-000.parquet", "byte_size": 12, "sha256": "a" * 64,
    }


def test_manifest_rejects_demo_index_outside_exact_200_range(lerobot_v3_metadata: Path) -> None:
    _rewrite_episodes(lerobot_v3_metadata, lambda records: records[0].__setitem__("demo_index_within_task", 200))
    with pytest.raises(ValueError, match="demo_index_within_task"):
        build_training_manifest(lerobot_v3_metadata, repository="behavior-1k/2026-challenge-demos", revision="4f50b44796641a4d526a19d9aeadc8aa51e2f2c2")


def test_materialize_requires_real_regular_files_and_hashes_schema(lerobot_v3_metadata: Path) -> None:
    payload = {
        "schema_version": 1, "source": {"repository": "behavior-1k/2026-challenge-demos", "revision": "4f50b44796641a4d526a19d9aeadc8aa51e2f2c2"},
        "tasks": [], "episodes": [], "required_files": ["meta/info.json"], "artifacts": [{"path": "meta/info.json", "byte_size": None, "sha256": None}],
    }
    selection = TrainingManifest(
        source={"repository": "behavior-1k/2026-challenge-demos", "revision": "4f50b44796641a4d526a19d9aeadc8aa51e2f2c2"},
        tasks=(), episodes=(), required_files=("meta/info.json",), artifacts=({"path": "meta/info.json", "byte_size": None, "sha256": None},), fingerprint=canonical_json_sha256(payload),
    )
    with pytest.raises(ValueError, match="100 tasks"):
        materialize_training_manifest(lerobot_v3_metadata, selection)


def test_materialize_rejects_selected_checksum_mismatch(lerobot_v3_metadata: Path) -> None:
    payload = {"schema_version": 1, "source": {"repository": "behavior-1k/2026-challenge-demos", "revision": "4f50b44796641a4d526a19d9aeadc8aa51e2f2c2"}, "tasks": [], "episodes": [], "required_files": ["meta/info.json"], "artifacts": [{"path": "meta/info.json", "byte_size": 1, "sha256": "a" * 64}]}
    selection = TrainingManifest(source=payload["source"], tasks=(), episodes=(), required_files=("meta/info.json",), artifacts=({"path": "meta/info.json", "byte_size": 1, "sha256": "a" * 64},), fingerprint=canonical_json_sha256(payload))
    with pytest.raises(ValueError, match="100 tasks"):
        materialize_training_manifest(lerobot_v3_metadata, selection)


def test_selection_validation_rejects_duplicate_or_missing_task_indices_before_download() -> None:
    tasks = tuple({"task_index": 0, "task_name": "task 0", "task": "task 0", "demonstrations": 200} for _ in range(100))
    episodes = tuple({"episode_index": index, "task_index": 0, "demo_index_within_task": index % 200} for index in range(20_000))
    payload = {
        "schema_version": 1,
        "source": {"repository": "behavior-1k/2026-challenge-demos", "revision": "4f50b44796641a4d526a19d9aeadc8aa51e2f2c2"},
        "tasks": list(tasks), "episodes": list(episodes), "required_files": [], "artifacts": [],
    }
    selection = TrainingManifest(
        source=payload["source"], tasks=tasks, episodes=episodes, required_files=(), artifacts=(),
        fingerprint=canonical_json_sha256(payload),
    )
    with pytest.raises(ValueError, match="task indices"):
        validate_training_manifest(selection)
