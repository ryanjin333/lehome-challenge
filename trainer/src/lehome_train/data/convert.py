"""Deterministic LeRobot v3-to-v2 repackaging for pinned GR00T N1.7."""

from __future__ import annotations

from collections import defaultdict
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Iterable, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from lehome_train.constants import ISAAC_GROOT_REVISION
from lehome_train.data.inspect import (
    artifact_identities,
    format_v3_path,
    inspect_dataset,
    load_v3_episode_records,
    read_json_object,
)
from lehome_train.data.mapping import (
    ACTION_HORIZON,
    FIXED_INSTRUCTION,
    JOINT_NAMES,
    load_checked_mapping,
)
from lehome_train.data.split import split_episode_ids
from lehome_train.io import (
    atomic_write_json,
    canonical_json_bytes,
    canonical_json_sha256,
)


LEGACY_DATA_PATH = (
    "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
)
LEGACY_VIDEO_PATH = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/"
    "episode_{episode_index:06d}.mp4"
)


def _write_json_lines(path: Path, values: Iterable[Mapping[str, object]]) -> None:
    payload = b"\n".join(canonical_json_bytes(value) for value in values) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _validate_hex_revision(value: str, field_name: str) -> None:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a full lowercase immutable commit")


def _validate_provenance(
    source_repository: str,
    source_revision: str,
    converter_commit: str,
    converter_container_digest: str,
) -> None:
    if not source_repository or source_repository.strip() != source_repository:
        raise ValueError("source_repository must be a nonempty immutable repository ID")
    _validate_hex_revision(source_revision, "source_revision")
    _validate_hex_revision(converter_commit, "converter_commit")
    if (
        len(converter_container_digest) != 71
        or not converter_container_digest.startswith("sha256:")
        or any(
            character not in "0123456789abcdef"
            for character in converter_container_digest[7:]
        )
    ):
        raise ValueError("converter container digest must be sha256:<64 lowercase hex>")


def _group_data_records(
    records: Iterable[dict[str, Any]],
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                int(record["data/chunk_index"]),
                int(record["data/file_index"]),
            )
        ].append(record)
    return grouped


def _load_episode_tables(
    source: Path,
    info: Mapping[str, Any],
    records: list[dict[str, Any]],
) -> dict[int, pa.Table]:
    """Apply pinned v3 global-index offsets to reconstruct episode tables."""

    pattern = info["data_path"]
    if not isinstance(pattern, str):
        raise ValueError("info data_path must be a string")
    tables: dict[int, pa.Table] = {}
    for (chunk_index, file_index), file_records in sorted(
        _group_data_records(records).items()
    ):
        source_path = format_v3_path(
            source,
            pattern,
            chunk_index=chunk_index,
            file_index=file_index,
        )
        table = pq.read_table(source_path)
        ordered = sorted(
            file_records,
            key=lambda record: int(record["dataset_from_index"]),
        )
        file_offset = int(ordered[0]["dataset_from_index"])
        for record in ordered:
            episode_id = int(record["episode_index"])
            start = int(record["dataset_from_index"]) - file_offset
            stop = int(record["dataset_to_index"]) - file_offset
            tables[episode_id] = table.slice(start, stop - start)
    return tables


def _legacy_path(
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
        raise ValueError("legacy output path escapes conversion root") from error
    return path


def _write_episode_data(
    output: Path,
    info: Mapping[str, Any],
    records: list[dict[str, Any]],
    tables: Mapping[int, pa.Table],
) -> None:
    chunks_size = int(info["chunks_size"])
    for record in sorted(records, key=lambda item: int(item["episode_index"])):
        episode_id = int(record["episode_index"])
        table = tables[episode_id]
        task_index = table.schema.get_field_index("task_index")
        if task_index < 0:
            raise ValueError(f"episode {episode_id} has no task_index column")
        table = table.set_column(
            task_index,
            "task_index",
            pa.array([0] * table.num_rows, type=pa.int64()),
        )
        destination = _legacy_path(
            output,
            LEGACY_DATA_PATH,
            episode_id=episode_id,
            chunks_size=chunks_size,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, destination, compression="zstd")


def _extract_video_segment(
    source: Path,
    destination: Path,
    *,
    start: float,
    end: float,
) -> None:
    if not source.is_file() or source.suffix.lower() != ".mp4":
        raise ValueError(f"invalid source MP4 shard: {source}")
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or start >= end:
        raise ValueError("invalid v3 video timestamp slice")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.6f}",
        "-i",
        str(source),
        "-t",
        f"{end - start:.6f}",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "1",
        "-y",
        str(destination),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            timeout=300,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError("ffmpeg is required for LeRobot v3 video slicing") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"ffmpeg timed out slicing {source}") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() if error.stderr else "unknown ffmpeg error"
        raise RuntimeError(f"ffmpeg failed slicing {source}: {detail}") from error


def _write_episode_videos(
    source: Path,
    output: Path,
    info: Mapping[str, Any],
    records: list[dict[str, Any]],
    camera_keys: list[str],
) -> None:
    source_pattern = info["video_path"]
    if not isinstance(source_pattern, str):
        raise ValueError("info video_path must be a string")
    chunks_size = int(info["chunks_size"])
    for camera_key in camera_keys:
        for record in sorted(records, key=lambda item: int(item["episode_index"])):
            episode_id = int(record["episode_index"])
            source_video = format_v3_path(
                source,
                source_pattern,
                chunk_index=int(record[f"videos/{camera_key}/chunk_index"]),
                file_index=int(record[f"videos/{camera_key}/file_index"]),
                video_key=camera_key,
            )
            destination = _legacy_path(
                output,
                LEGACY_VIDEO_PATH,
                episode_id=episode_id,
                chunks_size=chunks_size,
                video_key=camera_key,
            )
            _extract_video_segment(
                source_video,
                destination,
                start=float(record[f"videos/{camera_key}/from_timestamp"]),
                end=float(record[f"videos/{camera_key}/to_timestamp"]),
            )


def _v2_info(
    source_info: Mapping[str, Any],
    *,
    episode_count: int,
    camera_count: int,
) -> dict[str, Any]:
    info = dict(source_info)
    info["codebase_version"] = "v2.1"
    info["data_path"] = LEGACY_DATA_PATH
    info["video_path"] = LEGACY_VIDEO_PATH
    info.pop("data_files_size_in_mb", None)
    info.pop("video_files_size_in_mb", None)
    features = {
        key: dict(value) if isinstance(value, Mapping) else value
        for key, value in info["features"].items()
    }
    for feature in features.values():
        if isinstance(feature, dict) and feature.get("dtype") != "video":
            feature.pop("fps", None)
    info["features"] = features
    chunks_size = int(info["chunks_size"])
    info["total_chunks"] = math.ceil(episode_count / chunks_size)
    info["total_videos"] = episode_count * camera_count
    info["total_tasks"] = 1
    return info


def _v2_episode_metadata(record: Mapping[str, Any]) -> dict[str, object]:
    episode = {
        key: value
        for key, value in record.items()
        if not key.startswith(("data/", "videos/", "stats/", "meta/"))
        and key not in {"dataset_from_index", "dataset_to_index"}
    }
    episode["episode_index"] = int(record["episode_index"])
    episode["length"] = int(record["dataset_to_index"]) - int(
        record["dataset_from_index"]
    )
    episode["tasks"] = [FIXED_INSTRUCTION]
    return episode


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


def convert_dataset(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    mapping_path: str | Path | None,
    source_repository: str,
    source_revision: str,
    converter_commit: str,
    converter_container_digest: str,
    split_seed: int = 42,
    validation_fraction: float = 0.1,
) -> dict[str, object]:
    """Validate and convert a complete local v3 snapshot without remote writes."""

    mapping = load_checked_mapping(mapping_path)
    _validate_provenance(
        source_repository,
        source_revision,
        converter_commit,
        converter_container_digest,
    )
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
    info = read_json_object(source / "meta" / "info.json")
    records = load_v3_episode_records(source)
    camera_keys = [camera["source_key"] for camera in mapping["cameras"]]
    episode_tables = _load_episode_tables(source, info, records)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
    )
    try:
        _write_episode_data(temporary, info, records, episode_tables)
        _write_episode_videos(source, temporary, info, records, camera_keys)
        (temporary / "meta").mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            temporary / "meta" / "info.json",
            _v2_info(
                info,
                episode_count=len(records),
                camera_count=len(camera_keys),
            ),
        )
        sorted_records = sorted(records, key=lambda item: int(item["episode_index"]))
        _write_json_lines(
            temporary / "meta" / "episodes.jsonl",
            (_v2_episode_metadata(record) for record in sorted_records),
        )
        _write_json_lines(
            temporary / "meta" / "episodes_stats.jsonl",
            (
                {"episode_index": int(record["episode_index"]), "stats": {}}
                for record in sorted_records
            ),
        )
        _write_json_lines(
            temporary / "meta" / "tasks.jsonl",
            [{"task_index": 0, "task": FIXED_INSTRUCTION}],
        )
        atomic_write_json(temporary / "meta" / "modality.json", _modality_metadata())
        atomic_write_json(temporary / "meta" / "lehome_mapping.json", mapping)
        source_stats = source / "meta" / "stats.json"
        if source_stats.is_file():
            shutil.copyfile(source_stats, temporary / "meta" / "stats.json")

        valid_window_counts = {
            str(record["episode_index"]): max(
                0,
                int(record["dataset_to_index"])
                - int(record["dataset_from_index"])
                - ACTION_HORIZON
                + 1,
            )
            for record in sorted_records
        }
        output_artifacts = artifact_identities(temporary)
        manifest: dict[str, object] = {
            "schema_version": 1,
            "source_dataset": source.name,
            "source_format": "lerobot_v3_sharded",
            "output_format": "groot_lerobot_v2.1_per_episode",
            "source_repository": source_repository,
            "source_revision": source_revision,
            "source_manifest_sha256": inspection["source_manifest_sha256"],
            "source_artifacts": inspection["source_artifacts"],
            "output_artifacts": output_artifacts,
            "output_manifest_sha256": canonical_json_sha256(output_artifacts),
            "mapping_sha256": canonical_json_sha256(mapping),
            "converter_commit": converter_commit,
            "converter_container_digest": converter_container_digest,
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
