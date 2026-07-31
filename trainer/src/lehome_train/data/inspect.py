"""Fail-closed inspection of the organizer's LeRobot dataset."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

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


def _read_object(path: Path) -> dict[str, Any]:
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


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        values = [
            json.loads(line, object_pairs_hook=_strict_object)
            for line in lines
            if line.strip()
        ]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid required JSON lines file: {path}") from error
    if not all(isinstance(value, dict) for value in values):
        raise ValueError(f"JSON lines entries must be objects: {path}")
    return values


def artifact_identities(root: Path, *, exclude: set[str] | None = None) -> list[dict[str, object]]:
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
    names = feature.get("names")
    if names != list(JOINT_NAMES):
        errors.append(f"{kind} joint order does not match checked 12D order")


def _episode_path(info: Mapping[str, Any], source: Path, episode_id: int) -> Path:
    pattern = info.get("data_path")
    if not isinstance(pattern, str):
        raise ValueError("info data_path must be a string")
    path = source / pattern.format(
        episode_chunk=episode_id // int(info.get("chunks_size", 1000)),
        episode_index=episode_id,
    )
    try:
        path.resolve().relative_to(source.resolve())
    except ValueError as error:
        raise ValueError("info data_path escapes the dataset root") from error
    return path


def _scan_episode(
    source: Path,
    info: Mapping[str, Any],
    episode: Mapping[str, Any],
    fps: float,
    errors: list[str],
) -> int:
    episode_id = episode.get("episode_index")
    length = episode.get("length")
    if type(episode_id) is not int or type(length) is not int or length < 0:
        errors.append("episode metadata must contain nonnegative integer IDs and lengths")
        return 0
    parquet_path = _episode_path(info, source, episode_id)
    if not parquet_path.is_file():
        errors.append(f"missing episode parquet: {parquet_path.relative_to(source)}")
        return 0
    try:
        table = pq.read_table(
            parquet_path,
            columns=[
                "observation.state",
                "action",
                "timestamp",
                "frame_index",
                "episode_index",
            ],
        )
    except Exception as error:
        errors.append(f"could not read episode {episode_id} parquet: {error}")
        return 0
    if table.num_rows != length:
        errors.append(f"episode {episode_id} length does not match parquet rows")
    values = table.to_pydict()
    expected_frames = list(range(table.num_rows))
    if values["frame_index"] != expected_frames:
        errors.append(f"episode {episode_id} frame_index alignment drift")
    if values["episode_index"] != [episode_id] * table.num_rows:
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
    timestamps = values["timestamp"]
    tolerance = max(1e-5, 1e-4 / fps)
    for frame_index, timestamp in enumerate(timestamps):
        if (
            type(timestamp) not in (int, float)
            or not math.isfinite(float(timestamp))
            or abs(float(timestamp) - frame_index / fps) > tolerance
        ):
            errors.append(f"episode {episode_id} timestamp/frame alignment drift")
            break
    return table.num_rows


def inspect_dataset(
    source_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    """Inspect schema and payloads while returning a non-authoritative proposal."""

    source = Path(source_path)
    if not source.is_dir():
        raise FileNotFoundError(f"source dataset does not exist: {source}")
    info = _read_object(source / "meta" / "info.json")
    episodes = _read_json_lines(source / "meta" / "episodes.jsonl")
    errors: list[str] = []
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
    missing_cameras = sorted(set(CAMERA_KEYS) - set(observed_cameras))
    extra_cameras = sorted(set(observed_cameras) - set(CAMERA_KEYS))
    for key in missing_cameras:
        errors.append(f"missing camera: {key}")
    for key in extra_cameras:
        errors.append(f"extra camera: {key}")
    for key in CAMERA_KEYS:
        feature = features.get(key)
        if isinstance(feature, Mapping):
            if feature.get("shape") != [480, 640, 3]:
                errors.append(f"camera shape must be [480, 640, 3]: {key}")

    fps_value = info.get("fps")
    if type(fps_value) not in (int, float) or not math.isfinite(float(fps_value)) or fps_value <= 0:
        errors.append("dataset FPS must be finite and positive")
        fps = 0.0
    else:
        fps = float(fps_value)
    for key in CAMERA_KEYS:
        feature = features.get(key)
        if isinstance(feature, Mapping):
            video_info = feature.get("info")
            camera_fps = video_info.get("video.fps") if isinstance(video_info, Mapping) else None
            if type(camera_fps) not in (int, float) or abs(float(camera_fps) - fps) > 1e-9:
                errors.append(f"inconsistent episode FPS for camera: {key}")

    episode_ids: list[str] = []
    frame_count = 0
    seen_ids: set[int] = set()
    for episode in episodes:
        episode_id = episode.get("episode_index")
        if type(episode_id) is not int or episode_id in seen_ids:
            errors.append("episode IDs must be unique integers")
            continue
        seen_ids.add(episode_id)
        episode_ids.append(str(episode_id))
        if fps > 0:
            frame_count += _scan_episode(source, info, episode, fps, errors)
    episode_ids.sort(key=lambda value: int(value))
    if info.get("total_episodes") != len(episodes):
        errors.append("info total_episodes does not match episode metadata")
    if info.get("total_frames") != frame_count:
        errors.append("info total_frames does not match episode payloads")

    artifacts = artifact_identities(source)
    report: dict[str, object] = {
        "dataset_name": source.name,
        "source_manifest_sha256": canonical_json_sha256(artifacts),
        "source_artifacts": artifacts,
        "episode_ids": episode_ids,
        "episode_count": len(episodes),
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
