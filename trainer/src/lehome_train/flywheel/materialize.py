"""Verified raw-recorder artifacts to canonical per-episode LeRobot v2 data."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from lehome_train.data.convert import LEGACY_DATA_PATH, LEGACY_VIDEO_PATH, _validate_output_video
from lehome_train.data.mapping import FIXED_INSTRUCTION, JOINT_NAMES
from lehome_train.io import atomic_write_json, canonical_json_bytes, canonical_json_sha256


ACTION_HORIZON = 16
CAMERA_KEYS = ("top_rgb", "left_rgb", "right_rgb")


@dataclass(frozen=True, slots=True)
class MaterializationReport:
    episode_id: str
    selected_observations: int
    rejected_by_reason: dict[str, int]
    output_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": 1, "episode_id": self.episode_id,
                "selected_observations": self.selected_observations,
                "rejected_by_reason": dict(sorted(self.rejected_by_reason.items())),
                "output_sha256": self.output_sha256}


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("raw annotation contains a duplicate JSON field")
        result[key] = value
    return result


def _verify_raw(root: Path) -> dict[str, object]:
    """Use the collection-side terminal checksum verifier before decoding."""
    try:
        from lehome.flywheel.artifacts import verify_episode
    except ImportError as error:
        raise RuntimeError("the collection flywheel artifact verifier is required") from error
    raw = verify_episode(root)
    if not isinstance(raw, dict):
        raise ValueError("raw episode verifier returned malformed metadata")
    if not (root / "annotations.jsonl").is_file():
        raise ValueError("verified raw episode is missing annotations.jsonl")
    for camera in CAMERA_KEYS:
        if not (root / "videos" / f"{camera}.mp4").is_file():
            raise ValueError(f"verified raw episode is missing {camera} video")
    return raw


def _annotations(path: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            value = json.loads(line, object_pairs_hook=_strict_object)
        except json.JSONDecodeError:
            raise ValueError(f"raw annotation {number} is malformed") from None
        if not isinstance(value, dict):
            raise ValueError(f"raw annotation {number} must be an object")
        rows.append(value)
    if not rows:
        raise ValueError("verified raw episode has no annotations")
    return tuple(rows)


def _vector(value: object, *, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 12:
        raise ValueError(f"{label} must be a finite 12D vector")
    result: list[float] = []
    for item in value:
        if type(item) not in (int, float) or not float("-inf") < float(item) < float("inf"):
            raise ValueError(f"{label} must be a finite 12D vector")
        result.append(float(item))
    return result


def _eligible_windows(rows: tuple[dict[str, object], ...]) -> tuple[list[dict[str, object]], Counter[str]]:
    rejected: Counter[str] = Counter()
    for row in rows:
        source = row.get("action_source")
        if source not in {"policy", "expert", "hold"}:
            raise ValueError("raw annotation has an invalid action source")
        _vector(row.get("state"), label="raw state")
        _vector(row.get("action"), label="raw action")
        if source != "expert":
            rejected[str(source)] += 1
    selected: list[dict[str, object]] = []
    for index, observation in enumerate(rows):
        if observation["action_source"] != "expert":
            continue
        window = rows[index:index + ACTION_HORIZON]
        if len(window) != ACTION_HORIZON:
            rejected["short_tail"] += 1
            continue
        first_segment = observation.get("segment")
        valid = True
        for offset, candidate in enumerate(window):
            if (candidate.get("action_source") != "expert" or candidate.get("segment") != first_segment
                    or candidate.get("step") != observation.get("step", index) + offset):
                valid = False
                break
            age = candidate.get("expert_sample_age_ms")
            if type(age) not in (int, float) or not 0 <= float(age) < float("inf"):
                rejected["stale_expert"] += 1
                valid = False
                break
        if not valid:
            continue
        selected.append({"state": _vector(observation["state"], label="state"),
                         "action": _vector(observation["action"], label="action"),
                         "step": int(observation.get("step", index)),
                         "future_actions": [_vector(item["action"], label="future action") for item in window]})
    return selected, rejected


def _write_lines(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\n".join(canonical_json_bytes(value) for value in values) + b"\n")


def _copy_selected_video(source: Path, destination: Path, *, steps: list[int]) -> None:
    selection = "+".join(f"eq(n\\,{step})" for step in steps)
    command = ("ffmpeg", "-y", "-loglevel", "error", "-i", str(source), "-vf",
               f"select='{selection}',setpts=N/(30*TB)", "-an", "-c:v", "libx264",
               "-pix_fmt", "yuv420p", "-r", "30", "-g", "30", str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"could not materialize verified camera video: {source.name}") from error
    _validate_output_video(destination, expected_frame_count=len(steps), expected_fps=30.0)


def materialize_episode(raw_root: str | Path, output_root: str | Path) -> MaterializationReport:
    """Materialize only accepted Grade A/B expert windows from a terminal artifact.

    This consumes the recorder's real checksum manifest plus annotations; any
    non-expert, holdout, failed, stale, or incomplete target remains diagnostic.
    """
    raw_root = Path(raw_root)
    output = Path(output_root)
    if output.exists():
        raise FileExistsError("refusing to overwrite materialized episode")
    raw = _verify_raw(raw_root)
    if raw.get("quality_grade") == "C":
        raise ValueError("Grade C episodes cannot enter training")
    if raw.get("quality_grade") not in {"A", "B"} or raw.get("accepted_success") is not True or raw.get("trainable") is not True or raw.get("outcome") != "success":
        raise ValueError("raw episode is not an accepted successful expert episode")
    identity = raw.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("raw episode lacks immutable identity")
    if identity.get("release_stage") == "public_unseen":
        raise ValueError("evaluation holdout cannot enter training")
    if identity.get("instruction") != FIXED_INSTRUCTION:
        raise ValueError("raw episode has an incompatible task instruction")
    rows = _annotations(raw_root / "annotations.jsonl")
    selected, rejected = _eligible_windows(rows)
    if not selected:
        raise ValueError("accepted episode contains no complete expert windows")
    output.mkdir(parents=True)
    try:
        data_path = output / LEGACY_DATA_PATH.format(episode_chunk=0, episode_index=0)
        data_path.parent.mkdir(parents=True)
        pq.write_table(pa.table({"observation.state": pa.array([item["state"] for item in selected], type=pa.list_(pa.float32(), 12)),
                                 "action": pa.array([item["action"] for item in selected], type=pa.list_(pa.float32(), 12)),
                                 "timestamp": pa.array([index / 30 for index in range(len(selected))], type=pa.float32()),
                                 "frame_index": pa.array(range(len(selected)), type=pa.int64()),
                                 "episode_index": pa.array([0] * len(selected), type=pa.int64()),
                                 "index": pa.array(range(len(selected)), type=pa.int64()),
                                 "task_index": pa.array([0] * len(selected), type=pa.int64())}), data_path, compression="zstd")
        steps = [item["step"] for item in selected]
        for camera in CAMERA_KEYS:
            _copy_selected_video(raw_root / "videos" / f"{camera}.mp4", output / LEGACY_VIDEO_PATH.format(episode_chunk=0, episode_index=0, video_key=camera), steps=steps)
        meta = output / "meta"
        meta.mkdir()
        atomic_write_json(meta / "info.json", {"codebase_version": "v2.1", "robot_type": "dual_so101_follower", "total_episodes": 1, "total_frames": len(selected), "total_tasks": 1, "total_videos": 3, "total_chunks": 1, "chunks_size": 1000, "fps": 30, "data_path": LEGACY_DATA_PATH, "video_path": LEGACY_VIDEO_PATH, "features": {}})
        _write_lines(meta / "episodes.jsonl", [{"episode_index": 0, "length": len(selected), "task_index": 0}])
        _write_lines(meta / "episodes_stats.jsonl", [{"episode_index": 0, "stats": {}}])
        _write_lines(meta / "tasks.jsonl", [{"task_index": 0, "task": FIXED_INSTRUCTION}])
        atomic_write_json(meta / "materialization-provenance.json", {"raw_episode_id": raw["episode_id"], "raw_identity": dict(identity), "raw_manifest_verified": True, "selection_horizon": ACTION_HORIZON, "selected_steps": steps})
        manifest = {"schema_version": 1, "output_format": "groot_lerobot_v2.1_per_episode", "source_format": "flywheel_raw_terminal_artifact", "fps": 30, "episode_count": 1, "frame_count": len(selected), "train_episode_ids": ["0"], "validation_episode_ids": [], "fixed_language_instruction": FIXED_INSTRUCTION, "camera_schema": [{"source_key": f"observation.images.{camera}", "dtype": "video", "shape": [480, 640, 3]} for camera in CAMERA_KEYS], "state_schema": {"source_key": "observation.state", "dimension": 12, "names": list(JOINT_NAMES)}, "action_schema": {"source_key": "action", "dimension": 12, "names": list(JOINT_NAMES), "storage": "absolute"}, "future_actions": {"horizon": ACTION_HORIZON, "loader_allow_padding": False, "materialized_windows": True, "tail_convention": "drop_incomplete_windows", "valid_window_counts": {"0": len(selected)}}, "statistics": {"status": "pending_final_mixed_train_only", "files": []}}
        atomic_write_json(output / "manifest.json", manifest)
        digest = canonical_json_sha256(manifest)
        report = MaterializationReport(str(raw["episode_id"]), len(selected), dict(rejected), digest)
        atomic_write_json(output / "selection-report.json", report.to_dict())
        return report
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise
