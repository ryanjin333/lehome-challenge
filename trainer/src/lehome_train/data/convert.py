"""Deterministic local LeRobot repackaging for pinned GR00T N1.7."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from lehome_train.constants import ISAAC_GROOT_REVISION
from lehome_train.data.inspect import artifact_identities, inspect_dataset
from lehome_train.data.mapping import (
    ACTION_HORIZON,
    FIXED_INSTRUCTION,
    JOINT_NAMES,
    load_checked_mapping,
)
from lehome_train.data.split import split_episode_ids
from lehome_train.io import atomic_write_json, canonical_json_bytes, canonical_json_sha256


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(value, dict) for value in values):
        raise ValueError(f"JSON lines entries must be objects: {path}")
    return values


def _write_json_lines(path: Path, values: list[Mapping[str, object]]) -> None:
    payload = b"\n".join(canonical_json_bytes(value) for value in values) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _format_path(
    root: Path,
    pattern: str,
    *,
    episode_id: int,
    chunks_size: int,
    video_key: str | None = None,
) -> Path:
    path = root / pattern.format(
        episode_chunk=episode_id // chunks_size,
        episode_index=episode_id,
        video_key=video_key,
    )
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("LeRobot path pattern escapes the dataset root") from error
    return path


def _modality_metadata() -> dict[str, object]:
    groups = {
        "left_arm": {"start": 0, "end": 5},
        "left_gripper": {"start": 5, "end": 6},
        "right_arm": {"start": 6, "end": 11},
        "right_gripper": {"start": 11, "end": 12},
    }
    return {
        "video": {
            camera: {"original_key": f"observation.images.{camera}"}
            for camera in ("top_rgb", "left_rgb", "right_rgb")
        },
        "state": {
            key: {**bounds, "original_key": "observation.state"}
            for key, bounds in groups.items()
        },
        "action": {
            key: {**bounds, "original_key": "action"}
            for key, bounds in groups.items()
        },
        "annotation": {
            "human.task_description": {"original_key": "task_index"}
        },
    }


def _validate_commit(value: str) -> None:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("converter_commit must be a full lowercase Git commit")


def _copy_episode(
    source: Path,
    output: Path,
    info: Mapping[str, Any],
    episode_id: int,
    camera_keys: list[str],
) -> None:
    chunks_size = int(info["chunks_size"])
    data_pattern = str(info["data_path"])
    source_parquet = _format_path(
        source,
        data_pattern,
        episode_id=episode_id,
        chunks_size=chunks_size,
    )
    output_parquet = _format_path(
        output,
        data_pattern,
        episode_id=episode_id,
        chunks_size=chunks_size,
    )
    table = pq.read_table(source_parquet)
    task_index = table.schema.get_field_index("task_index")
    if task_index < 0:
        raise ValueError(f"episode {episode_id} has no task_index column")
    table = table.set_column(
        task_index,
        "task_index",
        pa.array([0] * table.num_rows, type=pa.int64()),
    )
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_parquet, compression="zstd")

    video_pattern = info.get("video_path")
    if not isinstance(video_pattern, str):
        raise ValueError("info video_path must be a string")
    for camera_key in camera_keys:
        source_video = _format_path(
            source,
            video_pattern,
            episode_id=episode_id,
            chunks_size=chunks_size,
            video_key=camera_key,
        )
        if not source_video.is_file():
            raise ValueError(
                f"missing camera payload for episode {episode_id}: {camera_key}"
            )
        output_video = _format_path(
            output,
            video_pattern,
            episode_id=episode_id,
            chunks_size=chunks_size,
            video_key=camera_key,
        )
        output_video.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_video, output_video)


def convert_dataset(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    mapping_path: str | Path | None,
    converter_commit: str,
    split_seed: int = 42,
    validation_fraction: float = 0.1,
) -> dict[str, object]:
    """Validate and repackage the complete local dataset without remote writes."""

    mapping = load_checked_mapping(mapping_path)
    _validate_commit(converter_commit)
    source = Path(source_path)
    destination = Path(destination_path)
    if destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("conversion destination must not be inside the source dataset")
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite existing conversion destination: {destination}"
        )
    inspection = inspect_dataset(source)
    if not inspection["valid"]:
        errors = inspection["validation_errors"]
        raise ValueError(f"source schema validation failed: {errors[0]}")
    if inspection["proposed_mapping"] != mapping:
        raise ValueError("checked mapping does not match observed source schema")

    split = split_episode_ids(
        inspection["episode_ids"],
        seed=split_seed,
        validation_fraction=validation_fraction,
    )
    info = _read_object(source / "meta" / "info.json")
    episodes = _read_json_lines(source / "meta" / "episodes.jsonl")
    camera_keys = [camera["source_key"] for camera in mapping["cameras"]]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
    )
    try:
        for episode in episodes:
            _copy_episode(
                source,
                temporary,
                info,
                int(episode["episode_index"]),
                camera_keys,
            )
        output_info = dict(info)
        output_info["total_tasks"] = 1
        output_info["splits"] = {
            "train": list(split.train),
            "validation": list(split.validation),
        }
        (temporary / "meta").mkdir(parents=True, exist_ok=True)
        atomic_write_json(temporary / "meta" / "info.json", output_info)
        output_episodes = [
            {
                **episode,
                "tasks": [FIXED_INSTRUCTION],
            }
            for episode in sorted(episodes, key=lambda item: int(item["episode_index"]))
        ]
        _write_json_lines(temporary / "meta" / "episodes.jsonl", output_episodes)
        _write_json_lines(
            temporary / "meta" / "tasks.jsonl",
            [{"task_index": 0, "task": FIXED_INSTRUCTION}],
        )
        atomic_write_json(temporary / "meta" / "modality.json", _modality_metadata())
        atomic_write_json(temporary / "meta" / "lehome_mapping.json", mapping)

        valid_window_counts = {
            str(episode["episode_index"]): max(
                0, int(episode["length"]) - ACTION_HORIZON + 1
            )
            for episode in sorted(episodes, key=lambda item: int(item["episode_index"]))
        }
        output_artifacts = artifact_identities(temporary)
        manifest: dict[str, object] = {
            "schema_version": 1,
            "source_dataset": source.name,
            "source_manifest_sha256": inspection["source_manifest_sha256"],
            "source_artifacts": inspection["source_artifacts"],
            "output_artifacts": output_artifacts,
            "output_manifest_sha256": canonical_json_sha256(output_artifacts),
            "mapping_sha256": canonical_json_sha256(mapping),
            "converter_commit": converter_commit,
            "pinned_groot_revision": ISAAC_GROOT_REVISION,
            "fps": inspection["fps"],
            "frame_count": inspection["frame_count"],
            "episode_count": inspection["episode_count"],
            "split_seed": split_seed,
            "validation_fraction": validation_fraction,
            "train_episode_ids": list(split.train),
            "validation_episode_ids": list(split.validation),
            "camera_schema": [
                {
                    **camera,
                    "dtype": "video",
                    "shape": [480, 640, 3],
                }
                for camera in mapping["cameras"]
            ],
            "state_schema": {
                "source_key": "observation.state",
                "dimension": 12,
                "names": list(JOINT_NAMES),
            },
            "action_schema": mapping["action"],
            "fixed_language_instruction": FIXED_INSTRUCTION,
            "future_actions": {
                "horizon": ACTION_HORIZON,
                "loader_allow_padding": False,
                "materialized_windows": False,
                "tail_convention": "drop_incomplete_windows",
                "valid_action_mask": "implicit_all_true_for_emitted_windows",
                "valid_window_counts": valid_window_counts,
            },
        }
        atomic_write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
