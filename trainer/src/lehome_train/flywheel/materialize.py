"""Verified raw-recorder artifacts to canonical per-episode LeRobot v2 data."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from lehome_train.data.convert import LEGACY_DATA_PATH, LEGACY_VIDEO_PATH, _validate_output_video
from lehome_train.data.inspect import artifact_identities
from lehome_train.data.mapping import FIXED_INSTRUCTION, JOINT_NAMES
from lehome_train.io import atomic_write_json, canonical_json_bytes, canonical_json_sha256
from lehome_train.io import sha256_file


ACTION_HORIZON = 16
RFT_ACTION_HORIZON = 16
CAMERA_KEYS = ("top_rgb", "left_rgb", "right_rgb")
_CUDA_DEVICE = re.compile(r"cuda:[0-9]+")


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


def _is_autonomous_policy_success(raw: Mapping[str, object]) -> bool:
    """Accept either the legacy autonomous marker or the closed release schema."""
    if raw.get("accepted_success") is not True or raw.get("outcome") != "success":
        return False
    mode = raw.get("mode")
    if mode is not None:
        return mode == "autonomous" and raw.get("terminal_reason") == "success"
    # Persistent collection evaluates the official success predicate on the
    # last frame.  A trajectory may therefore be accepted at its horizon even
    # though the worker's lifecycle reason remains ``horizon``.
    if raw.get("terminal_reason") not in {"success", "horizon"}:
        return False
    provenance = raw.get("provenance")
    if not isinstance(provenance, Mapping):
        return False
    artifact_sha256 = provenance.get("policy_artifact_sha256")
    policy_device = provenance.get("policy_device")
    parity_stage = provenance.get("parity_stage")
    simulator_device = provenance.get("simulator_device")
    canonical_policy = isinstance(policy_device, str) and _CUDA_DEVICE.fullmatch(policy_device) is not None
    persistent_cuda = (
        isinstance(simulator_device, str)
        and _CUDA_DEVICE.fullmatch(simulator_device) is not None
        and policy_device == simulator_device
    )
    return (
        type(raw.get("bc_target_count")) is int
        and raw.get("bc_target_count") == 0
        and provenance.get("execution_backend") == "policy_server"
        and provenance.get("execution_mode") == "policy_server"
        and parity_stage in {"server_cpu", "persistent_collection"}
        and (
            (parity_stage == "server_cpu" and simulator_device == "cpu")
            or (
                parity_stage == "persistent_collection"
                and (simulator_device == "cpu" or persistent_cuda)
            )
        )
        and canonical_policy
        and isinstance(artifact_sha256, str)
        and len(artifact_sha256) == 64
        and all(character in "0123456789abcdef" for character in artifact_sha256)
    )


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


def _eligible_windows(rows: tuple[dict[str, object], ...]) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Adapt annotations once, then defer selection/reporting to core flywheel code."""
    from lehome.flywheel.export import build_selection_report, select_expert_windows
    from lehome.flywheel.models import ActionSource, EpisodeFrame

    frames: list[EpisodeFrame] = []
    by_step: dict[int, dict[str, object]] = {}
    for index, row in enumerate(rows):
        try:
            step = row.get("step", index)
            if type(step) is not int:
                raise ValueError
            frame = EpisodeFrame(step=step, monotonic_ns=int(row["monotonic_ns"]), wall_time_ns=int(row["wall_time_ns"]),
                                 state=tuple(_vector(row.get("state"), label="raw state")),
                                 action=tuple(_vector(row.get("action"), label="raw action")),
                                 action_source=ActionSource(row["action_source"]), reward=float(row["reward"]),
                                 success=bool(row["success"]), segment=int(row["segment"]),
                                 policy_request_id=row.get("policy_request_id"), policy_chunk_offset=row.get("policy_chunk_offset"),
                                 expert_sequence=row.get("expert_sequence"), expert_sample_age_ms=row.get("expert_sample_age_ms"))
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"raw annotation {index + 1} violates the shared frame contract") from None
        frames.append(frame)
        by_step[step] = row
    windows = select_expert_windows(frames, horizon=ACTION_HORIZON, accepted_success=True)
    report = build_selection_report(frames, horizon=ACTION_HORIZON, accepted_success=True)
    selected = [
        {
            "step": window.observation_step,
            "source_steps": list(range(window.observation_step, window.observation_step + ACTION_HORIZON)),
            "states": [_vector(by_step[step]["state"], label="state") for step in range(window.observation_step, window.observation_step + ACTION_HORIZON)],
            "actions": [list(action) for action in window.future_actions],
        }
        for window in windows
    ]
    return selected, report.as_dict()


def _eligible_policy_windows(
    rows: tuple[dict[str, object], ...],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Select complete contiguous policy-action windows for rejection fine-tuning."""
    parsed: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        try:
            step = row.get("step", index)
            if type(step) is not int:
                raise ValueError
            action_source = row["action_source"]
            if action_source != "policy":
                raise ValueError
            segment = row["segment"]
            if type(segment) is not int:
                raise ValueError
            policy_request_id = row["policy_request_id"]
            policy_chunk_offset = row["policy_chunk_offset"]
            if not isinstance(policy_request_id, str) or not policy_request_id:
                raise ValueError
            if type(policy_chunk_offset) is not int or policy_chunk_offset < 0:
                raise ValueError
            parsed.append({
                "step": step,
                "segment": segment,
                "state": _vector(row.get("state"), label="raw state"),
                "action": _vector(row.get("action"), label="raw action"),
            })
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                f"raw annotation {index + 1} violates the autonomous policy frame contract"
            ) from None

    selected: list[dict[str, object]] = []
    rejected = {"incomplete_tail": min(len(parsed), RFT_ACTION_HORIZON - 1), "discontinuity": 0}
    for start in range(max(0, len(parsed) - RFT_ACTION_HORIZON + 1)):
        window = parsed[start : start + RFT_ACTION_HORIZON]
        first_step = window[0]["step"]
        if any(
            item["step"] != first_step + offset or item["segment"] != window[0]["segment"]
            for offset, item in enumerate(window)
        ):
            rejected["discontinuity"] += 1
            continue
        selected.append({
            "step": first_step,
            "source_steps": [int(item["step"]) for item in window],
            "states": [item["state"] for item in window],
            "actions": [item["action"] for item in window],
        })
    return selected, rejected


def _write_lines(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\n".join(canonical_json_bytes(value) for value in values) + b"\n")


def _copy_selected_video(source: Path, destination: Path, *, steps: list[int]) -> None:
    selection = "+".join(f"eq(n\\,{step})" for step in steps)
    command = ("ffmpeg", "-y", "-loglevel", "error", "-i", str(source), "-vf",
               f"select='{selection}',setpts=N/(30*TB)", "-an", "-c:v", "libx264",
               "-pix_fmt", "yuv420p", "-r", "30", "-g", "30", "-threads", "1", str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"could not materialize verified camera video: {source.name}") from error
    _validate_output_video(destination, expected_frame_count=len(steps), expected_fps=30.0)


def _materialize_selection(
    *,
    raw_root: Path,
    output: Path,
    raw: Mapping[str, object],
    identity: Mapping[str, object],
    selected: list[dict[str, object]],
    rejected: dict[str, int],
    action_source: str,
    training_method: str,
) -> MaterializationReport:
    output.mkdir(parents=True)
    try:
        global_index = 0
        for episode_index, item in enumerate(selected):
            data_path = output / LEGACY_DATA_PATH.format(episode_chunk=0, episode_index=episode_index)
            data_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table({"observation.state": pa.array(item["states"], type=pa.list_(pa.float32(), 12)),
                                     "action": pa.array(item["actions"], type=pa.list_(pa.float32(), 12)),
                                     "timestamp": pa.array([index / 30 for index in range(ACTION_HORIZON)], type=pa.float32()),
                                     "frame_index": pa.array(range(ACTION_HORIZON), type=pa.int64()),
                                     "episode_index": pa.array([episode_index] * ACTION_HORIZON, type=pa.int64()),
                                     "index": pa.array(range(global_index, global_index + ACTION_HORIZON), type=pa.int64()),
                                     "task_index": pa.array([0] * ACTION_HORIZON, type=pa.int64())}), data_path, compression="zstd")
            for camera in CAMERA_KEYS:
                _copy_selected_video(raw_root / "videos" / f"{camera}.mp4", output / LEGACY_VIDEO_PATH.format(episode_chunk=0, episode_index=episode_index, video_key=camera), steps=item["source_steps"])
            global_index += ACTION_HORIZON
        meta = output / "meta"
        meta.mkdir()
        atomic_write_json(meta / "info.json", {"codebase_version": "v2.1", "robot_type": "dual_so101_follower", "total_episodes": len(selected), "total_frames": global_index, "total_tasks": 1, "total_videos": len(selected) * 3, "total_chunks": 1, "chunks_size": 1000, "fps": 30, "data_path": LEGACY_DATA_PATH, "video_path": LEGACY_VIDEO_PATH, "features": {}})
        _write_lines(meta / "episodes.jsonl", [{"episode_index": index, "length": ACTION_HORIZON, "task_index": 0} for index in range(len(selected))])
        _write_lines(meta / "episodes_stats.jsonl", [{"episode_index": index, "stats": {}} for index in range(len(selected))])
        _write_lines(meta / "tasks.jsonl", [{"task_index": 0, "task": FIXED_INSTRUCTION}])
        provenance = {
            "raw_episode_id": raw["episode_id"],
            "raw_manifest_sha256": sha256_file(raw_root / "SHA256SUMS.json"),
            "raw_identity": dict(identity),
            "raw_manifest_verified": True,
            "accepted_success": raw["accepted_success"],
            "outcome": raw["outcome"],
            "training_method": training_method,
            "selection_horizon": ACTION_HORIZON,
            "rejected_by_reason": rejected,
            "selected_frame_ranges": [{"raw_episode_id": raw["episode_id"], "frame_start": item["source_steps"][0], "frame_stop": item["source_steps"][-1] + 1, "action_source": action_source} for item in selected],
        }
        if "quality_grade" in raw:
            provenance["quality_grade"] = raw["quality_grade"]
        if "trainable" in raw:
            provenance["trainable"] = raw["trainable"]
        if "terminal_reason" in raw:
            provenance["terminal_reason"] = raw["terminal_reason"]
        atomic_write_json(meta / "materialization-provenance.json", provenance)
        output_artifacts = artifact_identities(output)
        manifest = {"schema_version": 1, "output_format": "groot_lerobot_v2.1_per_episode", "source_format": "flywheel_raw_terminal_artifact", "fps": 30, "episode_count": len(selected), "frame_count": global_index, "train_episode_ids": [str(index) for index in range(len(selected))], "validation_episode_ids": [], "fixed_language_instruction": FIXED_INSTRUCTION, "camera_schema": [{"source_key": f"observation.images.{camera}", "dtype": "video", "shape": [480, 640, 3]} for camera in CAMERA_KEYS], "state_schema": {"source_key": "observation.state", "dimension": 12, "names": list(JOINT_NAMES)}, "action_schema": {"source_key": "action", "dimension": 12, "names": list(JOINT_NAMES), "storage": "absolute"}, "future_actions": {"horizon": ACTION_HORIZON, "loader_allow_padding": False, "materialized_windows": True, "tail_convention": "one_complete_raw_window_per_episode", "valid_window_counts": {str(index): 1 for index in range(len(selected))}}, "output_artifacts": output_artifacts, "output_manifest_sha256": canonical_json_sha256(output_artifacts), "statistics": {"status": "pending_final_mixed_train_only", "files": []}}
        atomic_write_json(output / "manifest.json", manifest)
        digest = canonical_json_sha256(manifest)
        report = MaterializationReport(str(raw["episode_id"]), len(selected), dict(rejected), digest)
        atomic_write_json(output / "selection-report.json", report.to_dict())
        return report
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise


def _materialize_rft_trajectory(
    *,
    raw_root: Path,
    output: Path,
    raw: Mapping[str, object],
    identity: Mapping[str, object],
    rows: tuple[dict[str, object], ...],
    valid_window_count: int,
    rejected: dict[str, int],
) -> MaterializationReport:
    """Store one policy-success trajectory once; GR00T forms its windows."""
    states = [_vector(row.get("state"), label="raw state") for row in rows]
    actions = [_vector(row.get("action"), label="raw action") for row in rows]
    frame_count = len(rows)
    output.mkdir(parents=True)
    try:
        data_path = output / LEGACY_DATA_PATH.format(episode_chunk=0, episode_index=0)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table({
                "observation.state": pa.array(states, type=pa.list_(pa.float32(), 12)),
                "action": pa.array(actions, type=pa.list_(pa.float32(), 12)),
                "timestamp": pa.array([index / 30 for index in range(frame_count)], type=pa.float32()),
                "frame_index": pa.array(range(frame_count), type=pa.int64()),
                "episode_index": pa.array([0] * frame_count, type=pa.int64()),
                "index": pa.array(range(frame_count), type=pa.int64()),
                "task_index": pa.array([0] * frame_count, type=pa.int64()),
            }),
            data_path,
            compression="zstd",
        )
        for camera in CAMERA_KEYS:
            source = raw_root / "videos" / f"{camera}.mp4"
            destination = output / LEGACY_VIDEO_PATH.format(
                episode_chunk=0, episode_index=0, video_key=camera
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            _validate_output_video(
                destination, expected_frame_count=frame_count, expected_fps=30.0
            )
        meta = output / "meta"
        meta.mkdir()
        atomic_write_json(meta / "info.json", {
            "codebase_version": "v2.1", "robot_type": "dual_so101_follower",
            "total_episodes": 1, "total_frames": frame_count, "total_tasks": 1,
            "total_videos": 3, "total_chunks": 1, "chunks_size": 1000, "fps": 30,
            "data_path": LEGACY_DATA_PATH, "video_path": LEGACY_VIDEO_PATH, "features": {},
        })
        _write_lines(meta / "episodes.jsonl", [{"episode_index": 0, "length": frame_count, "task_index": 0}])
        _write_lines(meta / "episodes_stats.jsonl", [{"episode_index": 0, "stats": {}}])
        _write_lines(meta / "tasks.jsonl", [{"task_index": 0, "task": FIXED_INSTRUCTION}])
        atomic_write_json(meta / "materialization-provenance.json", {
            "raw_episode_id": raw["episode_id"],
            "raw_manifest_sha256": sha256_file(raw_root / "SHA256SUMS.json"),
            "raw_identity": dict(identity),
            "raw_manifest_verified": True,
            "accepted_success": True,
            "outcome": "success",
            "terminal_reason": "success",
            "training_method": "rejection_finetuning",
            "selection_horizon": RFT_ACTION_HORIZON,
            "valid_observation_count": valid_window_count,
            "rejected_by_reason": rejected,
            "selected_frame_ranges": [{
                "raw_episode_id": raw["episode_id"],
                "frame_start": 0,
                "frame_stop": frame_count,
                "action_source": "policy",
            }],
        })
        output_artifacts = artifact_identities(output)
        manifest = {
            "schema_version": 1,
            "output_format": "groot_lerobot_v2.1_per_episode",
            "source_format": "flywheel_raw_terminal_artifact",
            "fps": 30,
            "episode_count": 1,
            "frame_count": frame_count,
            "train_episode_ids": ["0"],
            "validation_episode_ids": [],
            "fixed_language_instruction": FIXED_INSTRUCTION,
            "camera_schema": [{"source_key": f"observation.images.{camera}", "dtype": "video", "shape": [480, 640, 3]} for camera in CAMERA_KEYS],
            "state_schema": {"source_key": "observation.state", "dimension": 12, "names": list(JOINT_NAMES)},
            "action_schema": {"source_key": "action", "dimension": 12, "names": list(JOINT_NAMES), "storage": "absolute"},
            "future_actions": {
                "horizon": RFT_ACTION_HORIZON,
                "loader_allow_padding": False,
                "materialized_windows": False,
                "tail_convention": "drop_incomplete_windows",
                "valid_window_counts": {"0": valid_window_count},
            },
            "output_artifacts": output_artifacts,
            "output_manifest_sha256": canonical_json_sha256(output_artifacts),
            "statistics": {"status": "pending_final_rft_snapshot_train_only", "files": []},
        }
        atomic_write_json(output / "manifest.json", manifest)
        digest = canonical_json_sha256(manifest)
        report = MaterializationReport(
            str(raw["episode_id"]), valid_window_count, dict(rejected), digest
        )
        atomic_write_json(output / "selection-report.json", report.to_dict())
        return report
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise


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
        global_index = 0
        for episode_index, item in enumerate(selected):
            data_path = output / LEGACY_DATA_PATH.format(episode_chunk=0, episode_index=episode_index)
            data_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table({"observation.state": pa.array(item["states"], type=pa.list_(pa.float32(), 12)),
                                     "action": pa.array(item["actions"], type=pa.list_(pa.float32(), 12)),
                                     "timestamp": pa.array([index / 30 for index in range(ACTION_HORIZON)], type=pa.float32()),
                                     "frame_index": pa.array(range(ACTION_HORIZON), type=pa.int64()),
                                     "episode_index": pa.array([episode_index] * ACTION_HORIZON, type=pa.int64()),
                                     "index": pa.array(range(global_index, global_index + ACTION_HORIZON), type=pa.int64()),
                                     "task_index": pa.array([0] * ACTION_HORIZON, type=pa.int64())}), data_path, compression="zstd")
            for camera in CAMERA_KEYS:
                _copy_selected_video(raw_root / "videos" / f"{camera}.mp4", output / LEGACY_VIDEO_PATH.format(episode_chunk=0, episode_index=episode_index, video_key=camera), steps=item["source_steps"])
            global_index += ACTION_HORIZON
        meta = output / "meta"
        meta.mkdir()
        atomic_write_json(meta / "info.json", {"codebase_version": "v2.1", "robot_type": "dual_so101_follower", "total_episodes": len(selected), "total_frames": global_index, "total_tasks": 1, "total_videos": len(selected) * 3, "total_chunks": 1, "chunks_size": 1000, "fps": 30, "data_path": LEGACY_DATA_PATH, "video_path": LEGACY_VIDEO_PATH, "features": {}})
        _write_lines(meta / "episodes.jsonl", [{"episode_index": index, "length": ACTION_HORIZON, "task_index": 0} for index in range(len(selected))])
        _write_lines(meta / "episodes_stats.jsonl", [{"episode_index": index, "stats": {}} for index in range(len(selected))])
        _write_lines(meta / "tasks.jsonl", [{"task_index": 0, "task": FIXED_INSTRUCTION}])
        atomic_write_json(meta / "materialization-provenance.json", {"raw_episode_id": raw["episode_id"], "raw_manifest_sha256": sha256_file(raw_root / "SHA256SUMS.json"), "quality_grade": raw["quality_grade"], "raw_identity": dict(identity), "raw_manifest_verified": True, "accepted_success": raw["accepted_success"], "trainable": raw["trainable"], "outcome": raw["outcome"], "selection_horizon": ACTION_HORIZON, "rejected_by_reason": rejected, "selected_frame_ranges": [{"raw_episode_id": raw["episode_id"], "frame_start": item["source_steps"][0], "frame_stop": item["source_steps"][-1] + 1, "action_source": "expert"} for item in selected]})
        output_artifacts = artifact_identities(output)
        manifest = {"schema_version": 1, "output_format": "groot_lerobot_v2.1_per_episode", "source_format": "flywheel_raw_terminal_artifact", "fps": 30, "episode_count": len(selected), "frame_count": global_index, "train_episode_ids": [str(index) for index in range(len(selected))], "validation_episode_ids": [], "fixed_language_instruction": FIXED_INSTRUCTION, "camera_schema": [{"source_key": f"observation.images.{camera}", "dtype": "video", "shape": [480, 640, 3]} for camera in CAMERA_KEYS], "state_schema": {"source_key": "observation.state", "dimension": 12, "names": list(JOINT_NAMES)}, "action_schema": {"source_key": "action", "dimension": 12, "names": list(JOINT_NAMES), "storage": "absolute"}, "future_actions": {"horizon": ACTION_HORIZON, "loader_allow_padding": False, "materialized_windows": True, "tail_convention": "one_complete_raw_window_per_episode", "valid_window_counts": {str(index): 1 for index in range(len(selected))}}, "output_artifacts": output_artifacts, "output_manifest_sha256": canonical_json_sha256(output_artifacts), "statistics": {"status": "pending_final_mixed_train_only", "files": []}}
        atomic_write_json(output / "manifest.json", manifest)
        digest = canonical_json_sha256(manifest)
        report = MaterializationReport(str(raw["episode_id"]), len(selected), dict(rejected), digest)
        atomic_write_json(output / "selection-report.json", report.to_dict())
        return report
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise


def materialize_rft_episode(raw_root: str | Path, output_root: str | Path) -> MaterializationReport:
    """Materialize a verified seen-scenario autonomous success for RFT.

    RFT deliberately learns the successful policy trajectory. This contract is
    separate from :func:`materialize_episode`, whose DAgger path accepts only
    expert-action windows.
    """
    raw_root = Path(raw_root)
    output = Path(output_root)
    if output.exists():
        raise FileExistsError("refusing to overwrite materialized episode")
    raw = _verify_raw(raw_root)
    if not _is_autonomous_policy_success(raw):
        raise ValueError("raw episode is not an accepted autonomous success")
    identity = raw.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("raw episode lacks immutable identity")
    if identity.get("release_stage") == "public_unseen":
        raise ValueError("evaluation holdout cannot enter training")
    if identity.get("release_stage") != "seen":
        raise ValueError("RFT training requires an explicitly seen release stage")
    if identity.get("instruction") != FIXED_INSTRUCTION:
        raise ValueError("raw episode has an incompatible task instruction")
    rows = _annotations(raw_root / "annotations.jsonl")
    selected, rejected = _eligible_policy_windows(rows)
    if not selected:
        raise ValueError("accepted episode contains no complete policy windows")
    if any(row.get("step", index) != index for index, row in enumerate(rows)):
        raise ValueError("RFT policy trajectory must align every step with its video frame")
    if len({row.get("segment") for row in rows}) != 1:
        raise ValueError("RFT policy trajectory must remain in one contiguous segment")
    return _materialize_rft_trajectory(
        raw_root=raw_root,
        output=output,
        raw=raw,
        identity=identity,
        rows=rows,
        valid_window_count=len(selected),
        rejected=rejected,
    )
