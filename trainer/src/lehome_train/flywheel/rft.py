"""Build one immutable GR00T RFT snapshot from verified rollout artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import shutil
import tempfile
from typing import Iterable, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from lehome_train.data.convert import LEGACY_DATA_PATH, LEGACY_VIDEO_PATH, _modality_metadata
from lehome_train.data.inspect import artifact_identities
from lehome_train.data.mapping import FIXED_INSTRUCTION, JOINT_NAMES
from lehome_train.data.split import split_episode_ids
from lehome_train.flywheel.materialize import (
    CAMERA_KEYS,
    RFT_ACTION_HORIZON,
    _validate_output_video,
    _verify_raw,
    _is_autonomous_policy_success,
    materialize_rft_episode,
)
from lehome_train.io import atomic_write_json, canonical_json_bytes, canonical_json_sha256


_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _write_lines(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def _dataset_features() -> dict[str, object]:
    features: dict[str, object] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [12],
            "names": list(JOINT_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": [12],
            "names": list(JOINT_NAMES),
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for camera in CAMERA_KEYS:
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.channels": 3,
                "video.fps": 30,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    return features


def _require_identity(
    repository: str, revision: str, release_id: str
) -> None:
    if not repository or any(character.isspace() for character in repository):
        raise ValueError("RFT source repository is invalid")
    if not _REVISION.fullmatch(revision):
        raise ValueError("RFT source revision must be immutable")
    if not _SHA256.fullmatch(release_id):
        raise ValueError("RFT release ID must be a SHA-256")


def materialize_rft_snapshot(
    raw_episode_roots: Iterable[str | Path],
    destination: str | Path,
    *,
    source_repository: str,
    source_revision: str,
    release_id: str,
    split_seed: int,
    validation_fraction: float,
) -> dict[str, object]:
    """Select seen autonomous successes and build one efficient 40-step dataset."""

    _require_identity(source_repository, source_revision, release_id)
    if type(split_seed) is not int:
        raise ValueError("RFT split seed must be an integer")
    if (
        type(validation_fraction) not in (int, float)
        or not math.isfinite(float(validation_fraction))
        or not 0 < float(validation_fraction) < 1
    ):
        raise ValueError("RFT validation fraction must preserve a holdout")
    roots = tuple(sorted((Path(root) for root in raw_episode_roots), key=str))
    if len({root.resolve() for root in roots}) != len(roots):
        raise ValueError("RFT raw episode roots must be unique")
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("refusing to overwrite RFT snapshot destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
    )
    intermediates = temporary / ".materialized"
    intermediates.mkdir()
    excluded_public_unseen = 0
    excluded_failed = 0
    selected: list[dict[str, object]] = []
    global_index = 0
    try:
        for root in roots:
            raw = _verify_raw(root)
            identity = raw.get("identity")
            if not isinstance(identity, Mapping):
                raise ValueError("RFT raw episode lacks immutable identity")
            stage = identity.get("release_stage")
            if stage == "public_unseen":
                excluded_public_unseen += 1
                continue
            if stage != "seen":
                raise ValueError("RFT raw episode release stage is unsupported")
            if not _is_autonomous_policy_success(raw):
                excluded_failed += 1
                continue
            numeric_id = len(selected)
            prepared = intermediates / f"episode-{numeric_id:06d}"
            report = materialize_rft_episode(root, prepared)
            source_data = prepared / LEGACY_DATA_PATH.format(
                episode_chunk=0, episode_index=0
            )
            table = pq.read_table(source_data)
            frame_count = table.num_rows
            output_data = temporary / LEGACY_DATA_PATH.format(
                episode_chunk=numeric_id // 1000, episode_index=numeric_id
            )
            output_data.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                pa.table({
                    "observation.state": table["observation.state"],
                    "action": table["action"],
                    "timestamp": pa.array(
                        [index / 30 for index in range(frame_count)], type=pa.float32()
                    ),
                    "frame_index": pa.array(range(frame_count), type=pa.int64()),
                    "episode_index": pa.array([numeric_id] * frame_count, type=pa.int64()),
                    "index": pa.array(
                        range(global_index, global_index + frame_count), type=pa.int64()
                    ),
                    "task_index": pa.array([0] * frame_count, type=pa.int64()),
                }),
                output_data,
                compression="zstd",
            )
            for camera in CAMERA_KEYS:
                source_video = prepared / LEGACY_VIDEO_PATH.format(
                    episode_chunk=0, episode_index=0, video_key=camera
                )
                output_video = temporary / LEGACY_VIDEO_PATH.format(
                    episode_chunk=numeric_id // 1000,
                    episode_index=numeric_id,
                    video_key=f"observation.images.{camera}",
                )
                output_video.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_video, output_video)
                _validate_output_video(
                    output_video, expected_frame_count=frame_count, expected_fps=30.0
                )
            selected.append({
                "episode_index": numeric_id,
                "raw_episode_id": report.episode_id,
                "raw_manifest_sha256": json.loads(
                    (prepared / "meta" / "materialization-provenance.json").read_text(
                        encoding="utf-8"
                    )
                )["raw_manifest_sha256"],
                "frame_count": frame_count,
                "valid_window_count": report.selected_observations,
            })
            global_index += frame_count
        if len(selected) < 2:
            raise ValueError("RFT snapshot requires at least two seen successes")
        shutil.rmtree(intermediates)
        episode_ids = tuple(str(item["episode_index"]) for item in selected)
        split = split_episode_ids(
            episode_ids,
            seed=split_seed,
            validation_fraction=float(validation_fraction),
        )
        meta = temporary / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        atomic_write_json(meta / "info.json", {
            "codebase_version": "v2.1",
            "robot_type": "dual_so101_follower",
            "total_episodes": len(selected),
            "total_frames": global_index,
            "total_tasks": 1,
            "total_videos": len(selected) * len(CAMERA_KEYS),
            "total_chunks": math.ceil(len(selected) / 1000),
            "chunks_size": 1000,
            "fps": 30,
            "data_path": LEGACY_DATA_PATH,
            "video_path": LEGACY_VIDEO_PATH,
            "features": _dataset_features(),
        })
        _write_lines(meta / "episodes.jsonl", (
            {
                "episode_index": item["episode_index"],
                "length": item["frame_count"],
                "task_index": 0,
                "tasks": [FIXED_INSTRUCTION],
            }
            for item in selected
        ))
        _write_lines(meta / "episodes_stats.jsonl", (
            {"episode_index": item["episode_index"], "stats": {}}
            for item in selected
        ))
        _write_lines(meta / "tasks.jsonl", [{"task_index": 0, "task": FIXED_INSTRUCTION}])
        atomic_write_json(meta / "modality.json", _modality_metadata())
        atomic_write_json(meta / "rft-selection.json", {
            "schema_version": 1,
            "source_repository": source_repository,
            "source_revision": source_revision,
            "release_id": release_id,
            "action_horizon": RFT_ACTION_HORIZON,
            "excluded_public_unseen": excluded_public_unseen,
            "excluded_failed": excluded_failed,
            "episodes": selected,
        })
        output_artifacts = artifact_identities(temporary)
        manifest: dict[str, object] = {
            "schema_version": 1,
            "source_format": "verified_flywheel_rft_release",
            "output_format": "groot_lerobot_v2.1_per_episode",
            "source_repository": source_repository,
            "source_revision": source_revision,
            "source_release_id": release_id,
            "output_artifacts": output_artifacts,
            "output_manifest_sha256": canonical_json_sha256(output_artifacts),
            "fps": 30,
            "frame_count": global_index,
            "episode_count": len(selected),
            "split_seed": split_seed,
            "validation_fraction": float(validation_fraction),
            "train_episode_ids": list(split.train),
            "validation_episode_ids": list(split.validation),
            "camera_schema": [
                {
                    "source_key": f"observation.images.{camera}",
                    "dtype": "video",
                    "shape": [480, 640, 3],
                }
                for camera in CAMERA_KEYS
            ],
            "state_schema": {
                "source_key": "observation.state",
                "dimension": 12,
                "names": list(JOINT_NAMES),
            },
            "action_schema": {
                "source_key": "action",
                "dimension": 12,
                "names": list(JOINT_NAMES),
                "storage": "absolute",
            },
            "fixed_language_instruction": FIXED_INSTRUCTION,
            "future_actions": {
                "horizon": RFT_ACTION_HORIZON,
                "loader_allow_padding": False,
                "materialized_windows": False,
                "tail_convention": "drop_incomplete_windows",
                "valid_window_counts": {
                    str(item["episode_index"]): item["valid_window_count"]
                    for item in selected
                },
            },
            "statistics": {
                "status": "pending_final_rft_train_only",
                "files": [],
            },
        }
        atomic_write_json(temporary / "manifest.json", manifest)
        from lehome_train.data.stats import write_train_statistics
        from lehome_train.data.validate import validate_prepared_dataset

        statistics = write_train_statistics(temporary)
        validation = validate_prepared_dataset(temporary)
        temporary.replace(destination)
        return {
            "path": str(destination),
            "accepted_seen_successes": len(selected),
            "excluded_public_unseen": excluded_public_unseen,
            "excluded_failed": excluded_failed,
            "statistics": statistics,
            "validation": validation,
        }
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


__all__ = ["materialize_rft_snapshot"]
