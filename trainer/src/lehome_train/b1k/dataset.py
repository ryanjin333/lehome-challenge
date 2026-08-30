"""Metadata-first manifest builder for the immutable Behavior 1K v3 source."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq
import pyarrow as pa

from lehome_train.constants import BEHAVIOR_1K_DATASET_REPOSITORY, BEHAVIOR_1K_DATASET_REVISION
from lehome_train.data.inspect import format_v3_path, load_v3_episode_records, read_json_object
from lehome_train.io import canonical_json_sha256, sha256_file
from lehome_train.b1k.snapshot_integrity import ValidatedArtifact


RGB_CAMERA_KEYS = (
    "observation.rgb.left_realsense_link_camera_0",
    "observation.rgb.right_realsense_link_camera_0",
    "observation.rgb.zed_link_camera_0",
)
_OFFICIAL_DEPTH_FEATURE_KEYS = (
    "observation.depth_linear.left",
    "observation.depth_linear.right",
    "observation.depth_linear.top",
)
_PARQUET_TASK_FIELDS = {"task_index", "task"}
_JSONL_TASK_FIELDS = {"task_index", "task_name", "task"}
_ANNOTATION_SUMMARIES = (
    "annotations/skill_summary.csv",
    "annotations/skill_type_summary.csv",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _string(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _task_instance_id(value: object, label: str) -> str | int:
    if type(value) is int:
        return value
    if type(value) is str and value:
        return value
    raise ValueError(f"{label} must be a non-empty string or integer")


def _relative_path(root: Path, value: object, label: str) -> str:
    relative = _string(value, label)
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
        raise ValueError(f"{label} must be a safe relative path")
    try:
        (root / path).resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} must be a safe relative path") from error
    return relative


def _read_tasks(path: Path) -> list[dict[str, object]]:
    try:
        rows = pq.read_table(path).to_pylist()
    except Exception as error:
        raise ValueError("invalid required tasks.parquet") from error
    if not all(isinstance(row, dict) and set(row) == _PARQUET_TASK_FIELDS for row in rows):
        raise ValueError("tasks.parquet rows have an incompatible schema")
    return [dict(row) for row in rows]


def _read_tasks_jsonl(path: Path) -> list[dict[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid required tasks.jsonl") from error
    if not all(isinstance(row, dict) and set(row) == _JSONL_TASK_FIELDS for row in rows):
        raise ValueError("tasks.jsonl rows have an incompatible schema")
    return [dict(row) for row in rows]


def _parquet_task_index(rows: list[dict[str, object]]) -> dict[int, str]:
    indexed: dict[int, str] = {}
    for row in rows:
        index = _integer(row["task_index"], "task_index")
        task = _string(row["task"], "task")
        if index in indexed:
            raise ValueError("duplicate task_index")
        indexed[index] = task
    if set(indexed) != set(range(100)):
        raise ValueError("task metadata must contain exactly 100 task indices 0 through 99")
    if len(set(indexed.values())) != 100:
        raise ValueError("task metadata contains duplicate task identities")
    return indexed


def _jsonl_task_index(rows: list[dict[str, object]]) -> dict[int, dict[str, str]]:
    indexed: dict[int, dict[str, str]] = {}
    for row in rows:
        index = _integer(row["task_index"], "task_index")
        if index in indexed:
            raise ValueError("duplicate task_index")
        indexed[index] = {
            "task_name": _string(row["task_name"], "task_name"),
            "task": _string(row["task"], "task"),
        }
    if set(indexed) != set(range(100)):
        raise ValueError("task metadata must contain exactly 100 task indices 0 through 99")
    return indexed


def _validate_info(root: Path, allowed_camera_keys: tuple[str, ...]) -> dict[str, Any]:
    info = read_json_object(root / "meta" / "info.json")
    if info.get("total_tasks") != 100 or info.get("total_episodes") != 20_000:
        raise ValueError("info must declare exactly 100 tasks and 20,000 episodes")
    if info.get("robot_type") != "R1Pro":
        raise ValueError("info robot_type must be R1Pro")
    if not isinstance(info.get("data_path"), str) or not isinstance(info.get("video_path"), str):
        raise ValueError("info must provide v3 data and video path templates")
    features = info.get("features")
    if not isinstance(features, Mapping):
        raise ValueError("info features must be an object")
    depth_keys = {key for key in features if "depth" in key}
    if depth_keys and depth_keys != set(_OFFICIAL_DEPTH_FEATURE_KEYS):
        raise ValueError("depth feature declarations differ from the pinned official schema")
    for key in depth_keys:
        if not isinstance(features[key], Mapping) or features[key].get("dtype") != "video":
            raise ValueError("official depth feature declarations must be video features")
    for key, shape in (("observation.state", [61]), ("action", [23])):
        feature = features.get(key)
        if (
            not isinstance(feature, Mapping)
            or feature.get("shape") != shape
            or feature.get("dtype") != "float32"
        ):
            raise ValueError(f"info {key} must have shape {shape}")
    rgb_keys = {key for key in features if key.startswith("observation.rgb.")}
    if rgb_keys != set(allowed_camera_keys):
        raise ValueError("camera allowlist differs from the approved RGB camera keys")
    for key in allowed_camera_keys:
        feature = features[key]
        if not isinstance(feature, Mapping) or feature.get("dtype") != "video":
            raise ValueError(f"approved RGB camera {key} must be a video feature")
    return info


@dataclass(frozen=True, slots=True)
class TrainingManifest:
    source: dict[str, str]
    tasks: tuple[dict[str, object], ...]
    episodes: tuple[dict[str, object], ...]
    required_files: tuple[str, ...]
    artifacts: tuple[dict[str, object], ...]
    fingerprint: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "source": self.source,
            "tasks": list(self.tasks),
            "episodes": list(self.episodes),
            "required_files": list(self.required_files),
            "artifacts": list(self.artifacts),
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class MaterializedTrainingManifest:
    selection_fingerprint: str
    artifacts: tuple[dict[str, object], ...]
    feature_schema: dict[str, object]
    fingerprint: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "selection_fingerprint": self.selection_fingerprint,
            "artifacts": list(self.artifacts),
            "feature_schema": self.feature_schema,
            "fingerprint": self.fingerprint,
        }


def validate_training_manifest(selection: TrainingManifest) -> None:
    """Reject forged or unsafe selection payloads before touching downloaded data."""

    if not isinstance(selection, TrainingManifest):
        raise ValueError("selection manifest has an incompatible type")
    payload = selection.to_dict()
    fingerprint = payload.pop("fingerprint")
    if fingerprint != canonical_json_sha256(payload):
        raise ValueError("selection manifest fingerprint does not match its payload")
    if selection.source != {"repository": BEHAVIOR_1K_DATASET_REPOSITORY, "revision": BEHAVIOR_1K_DATASET_REVISION}:
        raise ValueError("selection manifest source is not pinned")
    if len(selection.tasks) != 100 or len(selection.episodes) != 20_000:
        raise ValueError("selection manifest must contain 100 tasks and 20,000 episodes")
    task_indices: set[int] = set()
    task_names: set[str] = set()
    for task in selection.tasks:
        if not isinstance(task, Mapping):
            raise ValueError("selection manifest task schema is invalid")
        task_index = _integer(task.get("task_index"), "task_index")
        task_name = _string(task.get("task"), "task")
        if task.get("demonstrations") != 200:
            raise ValueError("selection manifest task must declare exactly 200 demonstrations")
        task_indices.add(task_index)
        task_names.add(task_name)
    if task_indices != set(range(100)) or len(task_names) != 100:
        raise ValueError("selection manifest task indices must be unique and exactly 0 through 99")
    episode_indices: set[int] = set()
    demos_by_task: dict[int, set[int]] = {index: set() for index in range(100)}
    for episode in selection.episodes:
        if not isinstance(episode, Mapping):
            raise ValueError("selection manifest episode schema is invalid")
        episode_index = _integer(episode.get("episode_index"), "episode_index")
        task_index = _integer(episode.get("task_index"), "task_index")
        demo_index = _integer(episode.get("demo_index_within_task"), "demo_index_within_task")
        if task_index not in task_indices or demo_index not in range(200):
            raise ValueError("selection manifest episode task or demonstration index is invalid")
        if episode_index in episode_indices or demo_index in demos_by_task[task_index]:
            raise ValueError("selection manifest episode or demonstration indices are duplicated")
        episode_indices.add(episode_index)
        demos_by_task[task_index].add(demo_index)
    if episode_indices != set(range(20_000)) or any(demos != set(range(200)) for demos in demos_by_task.values()):
        raise ValueError("selection manifest must contain exactly 200 unique demonstrations per task")
    if len(selection.required_files) != len(set(selection.required_files)):
        raise ValueError("selection manifest required files are duplicated")
    if any("depth" in path or _relative_path(Path("/selection"), path, "required_file") != path for path in selection.required_files):
        raise ValueError("selection manifest contains an unsafe or depth artifact")
    if len(selection.artifacts) != len(selection.required_files) or {item.get("path") for item in selection.artifacts} != set(selection.required_files):
        raise ValueError("selection manifest artifacts are not a required-files bijection")
    for item in selection.artifacts:
        if set(item) != {"path", "byte_size", "sha256"}:
            raise ValueError("selection manifest artifact schema is invalid")
        size, checksum = item["byte_size"], item["sha256"]
        if (size is None) != (checksum is None) or (size is not None and (type(size) is not int or size < 0 or type(checksum) is not str or not _SHA256.fullmatch(checksum))):
            raise ValueError("selection manifest artifact identity is invalid")


def materialize_training_manifest(
    root: str | Path,
    selection: TrainingManifest,
    *,
    validated_artifacts: Sequence[ValidatedArtifact] | None = None,
) -> MaterializedTrainingManifest:
    """Verify a previously selected manifest after every artifact is downloaded."""

    if not isinstance(selection, TrainingManifest):
        raise ValueError("materialization requires a training selection manifest")
    validate_training_manifest(selection)
    source = Path(root)
    known = {
        str(item["path"]): {"byte_size": item["byte_size"], "sha256": item["sha256"]}
        for item in selection.artifacts
        if item["byte_size"] is not None
    }
    validated = None
    if validated_artifacts is not None:
        validated = {}
        for artifact in validated_artifacts:
            if artifact.path in validated:
                raise ValueError("validated artifacts are duplicated")
            validated[artifact.path] = artifact
    rebuilt = build_training_manifest(source, repository=selection.source["repository"], revision=selection.source["revision"], artifact_metadata=known)
    if rebuilt.to_dict() != selection.to_dict():
        raise ValueError("selection manifest does not match on-disk metadata")
    expected = {str(item.get("path")): item for item in selection.artifacts if isinstance(item, Mapping)}
    artifacts: list[dict[str, object]] = []
    for relative in selection.required_files:
        path = source / relative
        try:
            path.resolve().relative_to(source.resolve())
        except ValueError as error:
            raise ValueError(f"selected artifact is missing or unsafe: {relative}") from error
        cursor = source
        for component in Path(relative).parts:
            cursor = cursor / component
            if cursor.is_symlink():
                raise ValueError(f"selected artifact is missing or unsafe: {relative}")
        if not path.is_file():
            raise ValueError(f"selected artifact is missing or unsafe: {relative}")
        artifact = None if validated is None else validated.get(relative)
        if artifact is not None:
            observed = {"path": relative, "byte_size": artifact.byte_size, "sha256": artifact.sha256}
            if path.stat().st_size != artifact.byte_size:
                raise ValueError(f"artifact identity mismatch: {relative}")
        elif validated is not None:
            raise ValueError(f"selected artifact is absent from the validated snapshot: {relative}")
        else:
            observed = {"path": relative, "byte_size": path.stat().st_size, "sha256": sha256_file(path)}
        prior = expected.get(relative)
        if prior is not None and (prior.get("byte_size") is not None or prior.get("sha256") is not None) and (prior.get("byte_size") != observed["byte_size"] or prior.get("sha256") != observed["sha256"]):
            raise ValueError(f"artifact identity mismatch: {relative}")
        artifacts.append(observed)
    _validate_data_parquet_schemas(source, selection)
    info = _validate_info(source, RGB_CAMERA_KEYS)
    features = info["features"]
    schema = {str(key): value for key, value in sorted(features.items()) if "depth" not in key}
    payload = {"schema_version": 1, "selection_fingerprint": selection.fingerprint, "artifacts": artifacts, "feature_schema": schema}
    return MaterializedTrainingManifest(
        selection_fingerprint=selection.fingerprint, artifacts=tuple(artifacts), feature_schema=schema,
        fingerprint=canonical_json_sha256(payload),
    )


def _validate_data_parquet_schemas(root: Path, selection: TrainingManifest) -> None:
    """Check every selected data shard carries direct 61D/23D float32 vectors."""

    for relative in sorted({str(episode["data_path"]) for episode in selection.episodes}):
        try:
            schema = pq.read_schema(root / relative)
            state = schema.field("observation.state").type
            action = schema.field("action").type
        except Exception as error:
            raise ValueError(f"invalid selected data parquet schema: {relative}") from error
        for label, field_type, width in (("observation.state", state, 61), ("action", action, 23)):
            if not (pa.types.is_fixed_size_list(field_type) and field_type.list_size == width and pa.types.is_float32(field_type.value_type)):
                raise ValueError(f"selected data parquet {label} must be float32[{width}]: {relative}")


def build_training_manifest(
    root: str | Path,
    *,
    repository: str,
    revision: str,
    allowed_camera_keys: tuple[str, ...] = RGB_CAMERA_KEYS,
    artifact_metadata: Mapping[str, Mapping[str, object]] | None = None,
) -> TrainingManifest:
    """Build the complete, sorted 100-by-200 selection without downloading data."""

    if repository != BEHAVIOR_1K_DATASET_REPOSITORY or revision != BEHAVIOR_1K_DATASET_REVISION:
        raise ValueError("manifest source must use the pinned Behavior 1K repository and commit")
    if type(allowed_camera_keys) is not tuple or allowed_camera_keys != RGB_CAMERA_KEYS:
        raise ValueError("camera allowlist must be exactly the approved RGB cameras")
    source = Path(root)
    info = _validate_info(source, allowed_camera_keys)
    required_metadata = (
        "meta/info.json", "meta/stats.json", "meta/tasks.parquet", "meta/tasks.jsonl",
    )
    for relative in required_metadata + _ANNOTATION_SUMMARIES:
        if not (source / relative).is_file():
            raise ValueError(f"missing required metadata file: {relative}")
    parquet_tasks = _parquet_task_index(_read_tasks(source / "meta" / "tasks.parquet"))
    jsonl_tasks = _jsonl_task_index(_read_tasks_jsonl(source / "meta" / "tasks.jsonl"))
    if {index: values["task_name"] for index, values in jsonl_tasks.items()} != parquet_tasks:
        raise ValueError("tasks.parquet and tasks.jsonl disagree")

    records = load_v3_episode_records(source)
    required_files = set(required_metadata + _ANNOTATION_SUMMARIES)
    episodes: list[dict[str, object]] = []
    seen_episodes: set[int] = set()
    seen_annotations: set[str] = set()
    demos_by_task: dict[int, set[int]] = {index: set() for index in range(100)}
    episodes_by_task: dict[int, list[int]] = {index: [] for index in range(100)}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("episode metadata rows must be objects")
        episode_index = _integer(record.get("episode_index"), "episode_index")
        if episode_index in seen_episodes:
            raise ValueError("duplicate episode_index")
        seen_episodes.add(episode_index)
        task_index = _integer(record.get("task_index"), f"episode {episode_index} task_index")
        if task_index not in parquet_tasks:
            raise ValueError(f"episode {episode_index} references an unknown task_index")
        task_values = record.get("tasks")
        if not isinstance(task_values, list) or len(task_values) != 1 or type(task_values[0]) is not str:
            raise ValueError(f"episode {episode_index} must contain exactly one task annotation")
        task = parquet_tasks[task_index]
        if task_values[0] != task:
            raise ValueError(f"episode {episode_index} task annotation disagrees with task metadata")
        demo_index = _integer(record.get("demo_index_within_task"), f"episode {episode_index} demo_index_within_task")
        if demo_index not in range(200):
            raise ValueError(f"episode {episode_index} demo_index_within_task must be in 0 through 199")
        if demo_index in demos_by_task[task_index]:
            raise ValueError(f"task {task_index} has duplicate demo_index_within_task")
        demos_by_task[task_index].add(demo_index)
        _integer(record.get("raw_episode_id"), f"episode {episode_index} raw_episode_id")
        task_instance_id = _task_instance_id(
            record.get("task_instance_id"), f"episode {episode_index} task_instance_id"
        )
        annotation_path = _relative_path(source, record.get("annotation_path"), "annotation_path")
        if not annotation_path.startswith(f"annotations/task-{task_index:04d}/") or not annotation_path.endswith(".json"):
            raise ValueError(f"episode {episode_index} annotation_path disagrees with task_index")
        if annotation_path in seen_annotations:
            raise ValueError("duplicate annotation_path")
        seen_annotations.add(annotation_path)
        data_path = format_v3_path(
            source, str(info["data_path"]),
            chunk_index=_integer(record.get("data/chunk_index"), f"episode {episode_index} data chunk"),
            file_index=_integer(record.get("data/file_index"), f"episode {episode_index} data file"),
        ).relative_to(source).as_posix()
        meta_path = format_v3_path(
            source, "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            chunk_index=_integer(record.get("meta/episodes/chunk_index"), f"episode {episode_index} meta chunk"),
            file_index=_integer(record.get("meta/episodes/file_index"), f"episode {episode_index} meta file"),
        ).relative_to(source).as_posix()
        if not (source / meta_path).is_file():
            raise ValueError(f"missing required metadata file: {meta_path}")
        video_paths: dict[str, str] = {}
        for camera_key in allowed_camera_keys:
            video_paths[camera_key] = format_v3_path(
                source, str(info["video_path"]),
                chunk_index=_integer(record.get(f"videos/{camera_key}/chunk_index"), "missing camera asset"),
                file_index=_integer(record.get(f"videos/{camera_key}/file_index"), "missing camera asset"),
                video_key=camera_key,
            ).relative_to(source).as_posix()
        required_files.update((annotation_path, data_path, meta_path, *video_paths.values()))
        episodes_by_task[task_index].append(episode_index)
        episodes.append({
            "episode_index": episode_index,
            "task_index": task_index,
            "demo_index_within_task": demo_index,
            "raw_episode_id": record["raw_episode_id"],
            "task_instance_id": task_instance_id,
            "annotation_path": annotation_path,
            "data_path": data_path,
            "video_paths": video_paths,
        })
    tasks: list[dict[str, object]] = []
    for task_index in range(100):
        if len(episodes_by_task[task_index]) != 200 or len(demos_by_task[task_index]) != 200:
            raise ValueError(f"task {task_index} has {len(episodes_by_task[task_index])} demonstrations; expected 200")
        tasks.append({"task_index": task_index, **jsonl_tasks[task_index], "demonstrations": 200})
    if set(seen_episodes) != set(range(20_000)):
        raise ValueError("episode indices must be exactly 0 through 19,999")
    episodes.sort(key=lambda item: (int(item["task_index"]), int(item["episode_index"])))
    artifacts: list[dict[str, object]] = []
    metadata = {} if artifact_metadata is None else artifact_metadata
    for path in sorted(required_files):
        identity = metadata.get(path)
        if identity is None:
            artifacts.append({"path": path, "byte_size": None, "sha256": None})
            continue
        if not isinstance(identity, Mapping) or set(identity) != {"byte_size", "sha256"}:
            raise ValueError("artifact metadata must contain exactly byte_size and sha256")
        byte_size = identity["byte_size"]
        checksum = identity["sha256"]
        if type(byte_size) is not int or byte_size < 0 or type(checksum) is not str or not _SHA256.fullmatch(checksum):
            raise ValueError("artifact metadata has an invalid byte_size or sha256")
        artifacts.append({"path": path, "byte_size": byte_size, "sha256": checksum})
    if any(path not in required_files for path in metadata):
        raise ValueError("artifact metadata references a non-required file")
    payload = {
        "schema_version": 1,
        "source": {"repository": repository, "revision": revision},
        "tasks": tasks,
        "episodes": episodes,
        "required_files": sorted(required_files),
        "artifacts": artifacts,
    }
    return TrainingManifest(
        source=payload["source"], tasks=tuple(tasks), episodes=tuple(episodes),
        required_files=tuple(payload["required_files"]), fingerprint=canonical_json_sha256(payload),
        artifacts=tuple(artifacts),
    )
