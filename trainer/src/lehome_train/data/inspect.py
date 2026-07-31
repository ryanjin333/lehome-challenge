"""Fail-closed inspection of the organizer's sharded LeRobot v3 dataset."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow.parquet as pq

from lehome_train.data.mapping import CAMERA_KEYS, JOINT_NAMES, expected_mapping
from lehome_train.io import atomic_write_json, canonical_json_sha256, sha256_file


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid required JSON file: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def load_v3_episode_records(source: Path) -> list[dict[str, Any]]:
    """Load and stably order v3 episode metadata across all metadata shards."""

    paths = sorted((source / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise ValueError("no v3 episode metadata shards found")
    records: list[dict[str, Any]] = []
    for path in paths:
        try:
            records.extend(pq.read_table(path).to_pylist())
        except Exception as error:
            raise ValueError(f"invalid v3 episode metadata shard: {path}") from error
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("v3 episode metadata rows must be objects")
    records.sort(key=lambda record: int(record.get("episode_index", -1)))
    return records


def artifact_identities(
    root: Path,
    *,
    exclude: set[str] | None = None,
) -> list[dict[str, object]]:
    """Hash every regular file below ``root`` in stable relative-path order."""

    excluded = exclude or set()
    artifacts: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"dataset artifacts must not be symlinks: {path}")
        if not path.is_file():
            continue
        relative_path = path.relative_to(root).as_posix()
        if relative_path in excluded:
            continue
        artifacts.append(
            {
                "relative_path": relative_path,
                "sha256": sha256_file(path),
                "byte_size": path.stat().st_size,
            }
        )
    return artifacts


def format_v3_path(
    root: Path,
    pattern: str,
    *,
    chunk_index: int,
    file_index: int,
    video_key: str | None = None,
) -> Path:
    """Format one checked v3 shard path and prevent root escape."""

    try:
        relative = pattern.format(
            chunk_index=chunk_index,
            file_index=file_index,
            video_key=video_key,
        )
    except (KeyError, ValueError) as error:
        raise ValueError("invalid LeRobot v3 path pattern") from error
    path = root / relative
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("LeRobot v3 path pattern escapes the dataset root") from error
    return path


def _feature_error(
    errors: list[str],
    features: Mapping[str, object],
    key: str,
    kind: str,
) -> Mapping[str, Any] | None:
    value = features.get(key)
    if not isinstance(value, Mapping):
        errors.append(f"missing {kind} feature: {key}")
        return None
    return value


def _validate_vector_feature(
    errors: list[str],
    feature: Mapping[str, Any] | None,
    kind: str,
) -> None:
    if feature is None:
        return
    if feature.get("dtype") != "float32":
        errors.append(f"{kind} dtype must be float32")
    if feature.get("shape") != [12]:
        errors.append(f"{kind} shape must be [12]")
    if feature.get("names") != list(JOINT_NAMES):
        errors.append(f"{kind} joint order does not match checked 12D order")


def _record_integer(
    record: Mapping[str, Any],
    key: str,
    episode_id: int,
    errors: list[str],
) -> int | None:
    value = record.get(key)
    if type(value) is not int:
        errors.append(f"episode {episode_id} metadata {key} must be an integer")
        return None
    return value


def _scan_episode_slice(
    table_values: Mapping[str, list[Any]],
    *,
    start: int,
    length: int,
    episode_id: int,
    fps: float,
    errors: list[str],
) -> None:
    values = {
        key: column[start : start + length]
        for key, column in table_values.items()
    }
    if values["frame_index"] != list(range(length)):
        errors.append(f"episode {episode_id} frame_index alignment drift")
    if values["episode_index"] != [episode_id] * length:
        errors.append(f"episode {episode_id} episode_index alignment drift")
    for kind, rows in (
        ("state", values["observation.state"]),
        ("action", values["action"]),
    ):
        for frame_index, row in enumerate(rows):
            if not isinstance(row, list) or len(row) != 12:
                errors.append(
                    f"episode {episode_id} {kind} dimension drift at frame {frame_index}"
                )
                break
            if not all(
                type(value) in (int, float) and math.isfinite(float(value))
                for value in row
            ):
                errors.append(
                    f"episode {episode_id} non-finite {kind} at frame {frame_index}"
                )
                break
    tolerance = max(1e-5, 1e-4 / fps)
    for frame_index, timestamp in enumerate(values["timestamp"]):
        if (
            type(timestamp) not in (int, float)
            or not math.isfinite(float(timestamp))
            or abs(float(timestamp) - frame_index / fps) > tolerance
        ):
            errors.append(f"episode {episode_id} timestamp/frame alignment drift")
            break


def _group_records(
    records: Iterable[dict[str, Any]],
    chunk_key: str,
    file_key: str,
    errors: list[str],
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        episode_id = record.get("episode_index")
        chunk_index = record.get(chunk_key)
        file_index = record.get(file_key)
        if type(episode_id) is not int:
            errors.append("episode IDs must be unique integers")
            continue
        if type(chunk_index) is not int or type(file_index) is not int:
            errors.append(
                f"episode {episode_id} is missing integer {chunk_key}/{file_key}"
            )
            continue
        grouped[(chunk_index, file_index)].append(record)
    return grouped


def _scan_data_shards(
    source: Path,
    info: Mapping[str, Any],
    records: list[dict[str, Any]],
    fps: float,
    errors: list[str],
) -> int:
    pattern = info.get("data_path")
    if not isinstance(pattern, str):
        errors.append("info data_path must be a string")
        return 0
    grouped = _group_records(
        records,
        "data/chunk_index",
        "data/file_index",
        errors,
    )
    frame_count = 0
    for (chunk_index, file_index), file_records in sorted(grouped.items()):
        try:
            path = format_v3_path(
                source,
                pattern,
                chunk_index=chunk_index,
                file_index=file_index,
            )
        except ValueError as error:
            errors.append(str(error))
            continue
        if not path.is_file():
            errors.append(f"missing v3 data shard: {path.relative_to(source)}")
            continue
        try:
            table = pq.read_table(
                path,
                columns=[
                    "observation.state",
                    "action",
                    "timestamp",
                    "frame_index",
                    "episode_index",
                ],
            )
        except Exception as error:
            errors.append(f"could not read v3 data shard {path.name}: {error}")
            continue
        ordered = sorted(
            file_records,
            key=lambda record: int(record.get("dataset_from_index", -1)),
        )
        file_offset = ordered[0].get("dataset_from_index")
        if type(file_offset) is not int:
            errors.append(f"data shard {path.name} has invalid dataset offsets")
            continue
        expected_local_start = 0
        values = table.to_pydict()
        for record in ordered:
            episode_id = int(record["episode_index"])
            dataset_from = _record_integer(
                record, "dataset_from_index", episode_id, errors
            )
            dataset_to = _record_integer(
                record, "dataset_to_index", episode_id, errors
            )
            metadata_length = _record_integer(record, "length", episode_id, errors)
            if dataset_from is None or dataset_to is None or metadata_length is None:
                continue
            start = dataset_from - file_offset
            length = dataset_to - dataset_from
            if (
                start != expected_local_start
                or length <= 0
                or length != metadata_length
                or start + length > table.num_rows
            ):
                errors.append(f"episode {episode_id} has invalid v3 data slice")
                continue
            _scan_episode_slice(
                values,
                start=start,
                length=length,
                episode_id=episode_id,
                fps=fps,
                errors=errors,
            )
            frame_count += length
            expected_local_start += length
        if expected_local_start != table.num_rows:
            errors.append(f"v3 data shard {path.name} has unassigned frames")
    return frame_count


def _validate_video_shards(
    source: Path,
    info: Mapping[str, Any],
    records: list[dict[str, Any]],
    fps: float,
    errors: list[str],
) -> None:
    pattern = info.get("video_path")
    if not isinstance(pattern, str):
        errors.append("info video_path must be a string")
        return
    for camera_key in CAMERA_KEYS:
        chunk_key = f"videos/{camera_key}/chunk_index"
        file_key = f"videos/{camera_key}/file_index"
        grouped = _group_records(records, chunk_key, file_key, errors)
        for (chunk_index, file_index), file_records in sorted(grouped.items()):
            try:
                path = format_v3_path(
                    source,
                    pattern,
                    chunk_index=chunk_index,
                    file_index=file_index,
                    video_key=camera_key,
                )
            except ValueError as error:
                errors.append(str(error))
                continue
            if not path.is_file():
                errors.append(
                    f"missing camera video shard: {path.relative_to(source)}"
                )
            last_end = 0.0

            def video_start(item: Mapping[str, Any]) -> float:
                value = item.get(f"videos/{camera_key}/from_timestamp")
                return float(value) if type(value) in (int, float) else math.inf

            for record in sorted(
                file_records,
                key=video_start,
            ):
                episode_id = int(record["episode_index"])
                start = record.get(f"videos/{camera_key}/from_timestamp")
                end = record.get(f"videos/{camera_key}/to_timestamp")
                length = record.get("length")
                if (
                    type(start) not in (int, float)
                    or type(end) not in (int, float)
                    or type(length) is not int
                    or not math.isfinite(float(start))
                    or not math.isfinite(float(end))
                    or abs(float(start) - last_end) > 1e-6
                    or abs((float(end) - float(start)) - length / fps) > 1e-5
                ):
                    errors.append(f"episode {episode_id} has invalid camera time slice")
                    continue
                last_end = float(end)


def inspect_dataset(
    source_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    """Inspect v3 schema and payload shards without guessing a checked mapping."""

    source = Path(source_path)
    if not source.is_dir():
        raise FileNotFoundError(f"source dataset does not exist: {source}")
    info = read_json_object(source / "meta" / "info.json")
    records = load_v3_episode_records(source)
    errors: list[str] = []
    if info.get("codebase_version") != "v3.0":
        errors.append("source codebase_version must be v3.0")
    if not (source / "meta" / "tasks.parquet").is_file():
        errors.append("missing v3 tasks.parquet")

    features_value = info.get("features")
    if not isinstance(features_value, Mapping):
        raise ValueError("info features must be an object")
    features: Mapping[str, object] = features_value
    state = _feature_error(errors, features, "observation.state", "state")
    action = _feature_error(errors, features, "action", "action")
    _validate_vector_feature(errors, state, "state")
    _validate_vector_feature(errors, action, "action")

    observed_cameras = sorted(
        key
        for key, value in features.items()
        if isinstance(value, Mapping) and value.get("dtype") == "video"
    )
    for key in sorted(set(CAMERA_KEYS) - set(observed_cameras)):
        errors.append(f"missing camera: {key}")
    for key in sorted(set(observed_cameras) - set(CAMERA_KEYS)):
        errors.append(f"extra camera: {key}")
    for key in CAMERA_KEYS:
        feature = features.get(key)
        if isinstance(feature, Mapping) and feature.get("shape") != [480, 640, 3]:
            errors.append(f"camera shape must be [480, 640, 3]: {key}")

    fps_value = info.get("fps")
    if (
        type(fps_value) not in (int, float)
        or not math.isfinite(float(fps_value))
        or fps_value <= 0
    ):
        errors.append("dataset FPS must be finite and positive")
        fps = 0.0
    else:
        fps = float(fps_value)
    for key in CAMERA_KEYS:
        feature = features.get(key)
        if isinstance(feature, Mapping):
            video_info = feature.get("info")
            camera_fps = (
                video_info.get("video.fps")
                if isinstance(video_info, Mapping)
                else None
            )
            if (
                type(camera_fps) not in (int, float)
                or abs(float(camera_fps) - fps) > 1e-9
            ):
                errors.append(f"inconsistent episode FPS for camera: {key}")

    episode_ids: list[str] = []
    seen_ids: set[int] = set()
    for record in records:
        episode_id = record.get("episode_index")
        if type(episode_id) is not int or episode_id in seen_ids:
            errors.append("episode IDs must be unique integers")
            continue
        seen_ids.add(episode_id)
        episode_ids.append(str(episode_id))
    episode_ids.sort(key=lambda value: int(value))
    frame_count = (
        _scan_data_shards(source, info, records, fps, errors) if fps > 0 else 0
    )
    if fps > 0:
        _validate_video_shards(source, info, records, fps, errors)
    if info.get("total_episodes") != len(records):
        errors.append("info total_episodes does not match episode metadata")
    if info.get("total_frames") != frame_count:
        errors.append("info total_frames does not match episode payloads")

    artifacts = artifact_identities(source)
    report: dict[str, object] = {
        "dataset_name": source.name,
        "source_format": "lerobot_v3_sharded",
        "source_manifest_sha256": canonical_json_sha256(artifacts),
        "source_artifacts": artifacts,
        "episode_ids": episode_ids,
        "episode_count": len(records),
        "frame_count": frame_count,
        "fps": fps,
        "observed_schema": {
            "state": {
                "source_key": "observation.state",
                "dtype": state.get("dtype") if state else None,
                "shape": state.get("shape") if state else None,
                "names": state.get("names") if state else None,
            },
            "action": {
                "source_key": "action",
                "dtype": action.get("dtype") if action else None,
                "shape": action.get("shape") if action else None,
                "names": action.get("names") if action else None,
                "storage": "absolute",
            },
            "cameras": [
                {
                    "source_key": key,
                    "dtype": features[key].get("dtype"),
                    "shape": features[key].get("shape"),
                }
                for key in observed_cameras
                if isinstance(features[key], Mapping)
            ],
        },
        "validation_errors": errors,
        "valid": not errors,
        "proposed_mapping": expected_mapping() if not errors else None,
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(destination, report)
    return report
