"""Isaac-facing recorder that writes immutable autonomous-policy evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import subprocess
import time
from typing import Iterable, Mapping
from uuid import uuid4

import numpy as np

from .artifacts import EpisodeArtifactWriter, atomic_write_json
from .export import ExpertWindow, SelectionReport, build_selection_report, select_expert_windows
from .fidelity import validate_fidelity
from .models import ActionSource, EpisodeFrame, EpisodeIdentity, EpisodeOutcome, QualityGrade
from .snapshots import Snapshot, canonical_reset_hash


CANONICAL_CAMERA_NAMES = ("top_rgb", "left_rgb", "right_rgb")
CANONICAL_VIDEO_FILENAMES = tuple(f"{camera}.mp4" for camera in CANONICAL_CAMERA_NAMES)


def finite_vector(value: object, *, size: int, name: str) -> tuple[float, ...]:
    values = np.asarray(value, dtype=np.float32)
    if values.shape != (size,) or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return tuple(float(item) for item in values)


class _VideoSink:
    def __init__(self) -> None:
        self._frames: dict[str, list[np.ndarray]] = {camera: [] for camera in CANONICAL_CAMERA_NAMES}

    def append(self, observation: Mapping[str, object]) -> None:
        for camera in CANONICAL_CAMERA_NAMES:
            key = f"observation.images.{camera}"
            frame = np.asarray(observation[key])
            if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
                raise ValueError(f"{key} must be an HWC uint8 RGB frame")
            self._frames[camera].append(np.array(frame, copy=True))

    def count(self, camera: str) -> int:
        return len(self._frames[camera])

    def encode(self, root: Path, *, fps: int = 30) -> tuple[str, ...]:
        encoded: list[str] = []
        for camera in CANONICAL_CAMERA_NAMES:
            frames = self._frames[camera]
            if not frames:
                raise ValueError(f"{camera} has no video frames")
            height, width, channels = frames[0].shape
            if channels != 3 or any(frame.shape != (height, width, 3) for frame in frames):
                raise ValueError(f"{camera} video frames must share one RGB shape")
            output = root / "videos" / f"{camera}.mp4"
            output.parent.mkdir(parents=True, exist_ok=True)
            command = (
                "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                "-video_size", f"{width}x{height}", "-framerate", str(fps), "-i", "-",
                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output),
            )
            payload = b"".join(frame.tobytes() for frame in frames)
            result = subprocess.run(command, input=payload, capture_output=True, check=False)
            if result.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
                detail = result.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"failed to encode {camera} H.264 video: {detail}")
            encoded.append(f"{camera}.mp4")
        return tuple(encoded)


@dataclass(frozen=True, slots=True)
class RecordedEpisode:
    path: Path
    episode: dict[str, object]
    annotations: tuple[dict[str, object], ...]
    expert_windows: tuple[ExpertWindow, ...] = ()
    selection_report: SelectionReport | None = None


def _validate_provenance(provenance: Mapping[str, object] | None) -> dict[str, object]:
    result = dict(provenance or {})
    digest = result.get("policy_artifact_sha256")
    image = result.get("image_identity")
    if result and (not isinstance(digest, str) or len(digest) != 64 or not isinstance(image, str) or not image):
        raise ValueError("recorder provenance requires artifact SHA-256 and image identity")
    if any("secret" in key.lower() or "token" in key.lower() or "env" in key.lower() for key in result):
        raise ValueError("recorder provenance must not contain secrets or raw environment")
    return result


def _validate_visible_contact(value: Mapping[str, object]) -> dict[str, object]:
    """Require recorded simulator geometry evidence, never inferred arm motion."""
    contact = dict(value)
    if set(contact) != {"observed", "source", "minimum_distance_m"}:
        raise ValueError("visible contact evidence has unsupported fields")
    if not isinstance(contact["observed"], bool):
        raise ValueError("visible contact observed flag must be boolean")
    if contact["source"] != "simulator_particle_to_gripper_distance":
        raise ValueError("visible contact must use simulator particle-to-gripper evidence")
    distance = contact["minimum_distance_m"]
    if not isinstance(distance, (int, float)) or not math.isfinite(distance) or distance < 0:
        raise ValueError("visible contact minimum distance must be finite and non-negative")
    contact["minimum_distance_m"] = float(distance)
    return contact


def _identity_payload(identity: EpisodeIdentity) -> dict[str, object]:
    return {
        "episode_id": identity.episode_id,
        "policy_repo": identity.policy_repo, "policy_revision": identity.policy_revision,
        "policy_step": identity.policy_step, "code_revision": identity.code_revision,
        "asset_revision": identity.asset_revision, "simulator_version": identity.simulator_version,
        "garment_name": identity.garment_name, "category": identity.category,
        "release_stage": identity.release_stage, "seed": identity.seed,
        "instruction": identity.instruction, "strategy": identity.strategy,
    }


def _validated_simple_curriculum_fidelity(fidelity: Mapping[str, object] | None) -> dict[str, bool]:
    try:
        return validate_fidelity(fidelity)
    except ValueError as error:
        raise ValueError("simple curriculum fidelity evidence is invalid") from error


class MixedSourceRecorder:
    """Atomic raw recorder for policy, expert, and hold action-source segments."""

    def __init__(
        self,
        run_root: Path,
        *,
        identity: EpisodeIdentity | None = None,
        policy_revision: str | None = None,
        episode_id: str | None = None,
        provenance: Mapping[str, object] | None = None,
        mode: str,
        horizon: int = 16,
        max_expert_sample_age_ms: float | None = None,
        require_identity: bool = True,
        simple_curriculum_collection: bool = False,
    ) -> None:
        if mode not in {"autonomous", "practice", "expert", "dagger"}:
            raise ValueError("recorder mode is unsupported")
        if require_identity and identity is None:
            raise ValueError("mixed-source recorder requires a validated episode identity")
        policy_revision = identity.policy_revision if identity is not None else policy_revision
        if policy_revision is None:
            raise ValueError("policy_revision is required")
        if len(policy_revision) != 40 or any(char not in "0123456789abcdef" for char in policy_revision):
            raise ValueError("policy_revision must be a pinned 40-character revision")
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("expert selection horizon must be positive")
        if max_expert_sample_age_ms is not None and (
            not isinstance(max_expert_sample_age_ms, (int, float)) or max_expert_sample_age_ms < 0
        ):
            raise ValueError("maximum expert sample age must be non-negative")
        self.writer = EpisodeArtifactWriter(run_root, episode_id or (identity.episode_id if identity is not None else f"episode-{uuid4().hex}"))
        self.policy_revision = policy_revision
        self.mode = mode
        self.horizon = horizon
        self.max_expert_sample_age_ms = max_expert_sample_age_ms
        if identity is not None and identity.policy_revision != policy_revision:
            raise ValueError("recorder identity policy revision does not match")
        if identity is not None and identity.episode_id != self.writer.episode_id:
            raise ValueError("recorder identity episode ID does not match writer")
        self.identity = identity
        self.provenance = _validate_provenance(provenance)
        self.video_sink = _VideoSink()
        self.step = 0
        self._annotations: list[dict[str, object]] = []
        self._frames: list[EpisodeFrame] = []
        self._snapshots: set[str] = set()
        self._continuation_snapshot_steps: set[int] = set()
        self._reset_hash: str | None = None
        self._expert_started = False
        self._finished = False
        if type(simple_curriculum_collection) is not bool:
            raise ValueError("simple_curriculum_collection must be a boolean")
        self._simple_curriculum_collection = simple_curriculum_collection

    def record_step(
        self,
        observation: Mapping[str, object],
        action: object,
        *,
        reward: float,
        success: bool,
        action_source: ActionSource,
        segment: int,
        policy_request_id: str | None = None,
        policy_chunk_offset: int | None = None,
        expert_sequence: int | None = None,
        expert_sample_age_ms: float | None = None,
    ) -> None:
        if self._finished:
            raise ValueError("recorder has already finished")
        if not isinstance(action_source, ActionSource):
            raise ValueError("action_source must be an ActionSource")
        if action_source is ActionSource.POLICY:
            if not policy_request_id or not isinstance(policy_chunk_offset, int) or policy_chunk_offset < 0:
                raise ValueError("policy source requires request ID and chunk offset")
            if expert_sequence is not None or expert_sample_age_ms is not None:
                raise ValueError("policy source forbids expert provenance")
            if self.mode == "dagger" and self._expert_started:
                raise ValueError("DAgger policy source cannot resume after expert takeover")
        elif action_source is ActionSource.EXPERT:
            if not isinstance(expert_sequence, int) or expert_sequence < 0 or not isinstance(expert_sample_age_ms, (int, float)) or expert_sample_age_ms < 0 or not math.isfinite(expert_sample_age_ms):
                raise ValueError("expert source requires sequence and finite sample age")
            if policy_request_id is not None or policy_chunk_offset is not None:
                raise ValueError("expert source forbids policy provenance")
            if self.mode == "dagger" and "takeover" not in self._snapshots:
                raise ValueError("DAgger expert source requires a takeover snapshot")
            self._expert_started = True
        else:
            if any(value is not None for value in (policy_request_id, policy_chunk_offset, expert_sequence, expert_sample_age_ms)):
                raise ValueError("hold source forbids policy and expert provenance")
        if not isinstance(reward, (int, float)) or not math.isfinite(reward):
            raise ValueError("reward must be finite")
        frame = EpisodeFrame(
            step=self.step,
            monotonic_ns=time.monotonic_ns(),
            wall_time_ns=time.time_ns(),
            state=finite_vector(observation["observation.state"], size=12, name="state"),
            action=finite_vector(action, size=12, name="action"),
            action_source=action_source,
            reward=float(reward),
            success=bool(success),
            segment=segment,
            policy_request_id=policy_request_id,
            policy_chunk_offset=policy_chunk_offset,
            expert_sequence=expert_sequence,
            expert_sample_age_ms=float(expert_sample_age_ms) if expert_sample_age_ms is not None else None,
        )
        if frame.step != len(self._frames):
            raise ValueError("recorder frame step continuity is invalid")
        annotation = asdict(frame)
        annotation["action_source"] = frame.action_source.value
        if self.identity is not None:
            annotation.update({
                "category": self.identity.category,
                "garment_name": self.identity.garment_name,
                "seed": self.identity.seed,
            })
        self.writer.append_annotation(annotation)
        self._annotations.append(annotation)
        self._frames.append(frame)
        self.video_sink.append(observation)
        self.step += 1

    def record_snapshot(self, name: str, snapshot: Snapshot) -> None:
        if self._finished or name not in {"reset", "takeover", "terminal"}:
            raise ValueError("snapshot name must be reset, takeover, or terminal before finalization")
        if not isinstance(snapshot, Snapshot):
            raise ValueError("recorder snapshots must use the validated Snapshot schema")
        payload = snapshot.to_dict()
        directory = self.writer.staging / "snapshots"
        directory.mkdir(exist_ok=True)
        atomic_write_json(directory / f"{name}.json", payload)
        self._snapshots.add(name)
        if name == "reset":
            self._reset_hash = canonical_reset_hash(snapshot)

    def record_continuation_snapshot(self, step: int, snapshot: Snapshot) -> None:
        """Persist a full simulator state immediately before an H=16 action.

        ``step`` is the next annotation/action index.  This makes a recovery
        source directly restorable at a fresh policy-request boundary without
        trying to reconstruct hidden simulator state through open-loop replay.
        """

        if self._finished or type(step) is not int or step <= 0 or step != self.step or step % self.horizon:
            raise ValueError("continuation snapshot must be the current positive H16 boundary before finalization")
        if not isinstance(snapshot, Snapshot):
            raise ValueError("recorder snapshots must use the validated Snapshot schema")
        directory = self.writer.staging / "snapshots" / "continuations"
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{step:06d}.json"
        if destination.exists():
            raise ValueError("continuation snapshot already exists for this H16 boundary")
        atomic_write_json(destination, snapshot.to_dict())
        self._continuation_snapshot_steps.add(step)

    def _discard_continuation_snapshots_at_or_after_first_success(self) -> None:
        first_success = next((frame.step for frame in self._frames if frame.success), None)
        if first_success is None:
            raise ValueError("accepted autonomous episode has no successful annotation")
        directory = self.writer.staging / "snapshots" / "continuations"
        for step in tuple(self._continuation_snapshot_steps):
            if step < first_success:
                continue
            (directory / f"{step:06d}.json").unlink()
            self._continuation_snapshot_steps.remove(step)

    def _encode_videos(self) -> tuple[str, ...]:
        if any(self.video_sink.count(camera) != self.step for camera in CANONICAL_CAMERA_NAMES):
            raise ValueError("video frame count does not match annotation count")
        encoded = self.video_sink.encode(self.writer.staging)
        if encoded != CANONICAL_VIDEO_FILENAMES:
            raise ValueError("encoder did not produce every canonical autonomous video")
        return CANONICAL_VIDEO_FILENAMES

    def _base_episode(self) -> dict[str, object]:
        episode: dict[str, object] = {"policy_revision": self.policy_revision}
        if self.identity is not None:
            episode["identity"] = _identity_payload(self.identity)
        if self.provenance:
            episode["provenance"] = self.provenance
        if self._reset_hash is not None:
            episode["reset_hash"] = self._reset_hash
        return episode

    def finish(self, *, outcome: EpisodeOutcome, controls: Iterable[str]) -> RecordedEpisode:
        """Finalize raw evidence and expose only in-memory eligible expert windows."""
        if self._finished:
            raise ValueError("recorder has already finished")
        if not isinstance(outcome, EpisodeOutcome):
            raise ValueError("recorder finish requires a validated EpisodeOutcome")
        controls = tuple(controls)
        if not all(isinstance(control, str) and control for control in controls):
            raise ValueError("recorder controls must be non-empty strings")
        self._finished = True
        diagnostic_reasons: list[str] = []
        candidate = outcome.accepted and outcome.outcome == "success" and outcome.quality_grade in {QualityGrade.A, QualityGrade.B} and not outcome.rejection_reasons and self.mode != "practice"
        if not candidate:
            diagnostic_reasons.append("outcome_not_trainable")
        if candidate:
            for required in ("reset", "terminal"):
                if required not in self._snapshots:
                    diagnostic_reasons.append(f"missing_{required}_snapshot")
            if self.mode == "dagger" and "takeover" not in self._snapshots:
                diagnostic_reasons.append("missing_takeover_snapshot")
        required_videos: tuple[str, ...] = ()
        if candidate and not diagnostic_reasons:
            try:
                required_videos = self._encode_videos()
            except Exception as error:
                diagnostic_reasons.append("video_encoding_failed")
                recorder_error = str(error)
            else:
                recorder_error = None
        else:
            recorder_error = None
        trainable = candidate and not diagnostic_reasons
        windows = select_expert_windows(
            self._frames,
            horizon=self.horizon,
            accepted_success=trainable,
            max_expert_sample_age_ms=self.max_expert_sample_age_ms,
        )
        report = build_selection_report(
            self._frames,
            horizon=self.horizon,
            accepted_success=trainable,
            max_expert_sample_age_ms=self.max_expert_sample_age_ms,
        )
        episode = self._base_episode() | {
            "mode": self.mode,
            "outcome": outcome.outcome,
            "accepted_success": trainable,
            "quality_grade": outcome.quality_grade.value,
            "rejection_reasons": [reason.value for reason in outcome.rejection_reasons],
            "operator_disposition": outcome.operator_disposition,
            "controls": list(controls),
            "trainable": trainable,
            "bc_target_count": len(windows),
            "selection_report": report.as_dict(),
        }
        if diagnostic_reasons:
            episode["diagnostic"] = True
            episode["diagnostic_reasons"] = diagnostic_reasons
        if recorder_error is not None:
            episode["recorder_error"] = recorder_error
        path = self.writer.finalize(episode, required_videos=required_videos)
        return RecordedEpisode(path=path, episode=episode | {"episode_id": path.name}, annotations=tuple(self._annotations), expert_windows=windows, selection_report=report)

    def finish_autonomous(
        self,
        *,
        reason: str,
        accepted_success: bool,
        visible_contact: Mapping[str, object] | None = None,
        fidelity: Mapping[str, object] | None = None,
    ) -> RecordedEpisode:
        """Compatibility finalizer preserving autonomous diagnostic semantics."""
        if self._finished:
            raise ValueError("recorder has already finished")
        if not reason:
            raise ValueError("terminal reason is required")
        if accepted_success:
            self._discard_continuation_snapshots_at_or_after_first_success()
        self._finished = True
        episode = self._base_episode() | {
            "mode": self.mode,
            "terminal_reason": reason,
            "accepted_success": bool(accepted_success),
            "outcome": "success" if accepted_success else "timeout",
            "bc_target_count": 0,
        }
        if visible_contact is not None:
            episode["visible_contact"] = _validate_visible_contact(visible_contact)
        if self._simple_curriculum_collection:
            verified_fidelity = _validated_simple_curriculum_fidelity(fidelity)
            episode["fidelity"] = verified_fidelity
            episode["safety_failure"] = verified_fidelity["safety_failure"]
        required_videos = self._encode_videos()
        path = self.writer.finalize(episode, required_videos=required_videos)
        return RecordedEpisode(path=path, episode=episode | {"episode_id": path.name}, annotations=tuple(self._annotations))


class AutonomousRecorder:
    """Compatibility facade delegating atomic capture to :class:`MixedSourceRecorder`."""

    def __init__(self, run_root: Path, *, policy_revision: str, episode_id: str | None = None, identity: EpisodeIdentity | None = None, provenance: Mapping[str, object] | None = None, simple_curriculum_collection: bool = False) -> None:
        self._recorder = MixedSourceRecorder(
            run_root,
            policy_revision=policy_revision,
            episode_id=episode_id,
            identity=identity,
            provenance=provenance,
            mode="autonomous",
            require_identity=False,
            simple_curriculum_collection=simple_curriculum_collection,
        )
        self.writer = self._recorder.writer
        self.policy_revision = self._recorder.policy_revision
        self.identity = self._recorder.identity
        self.provenance = self._recorder.provenance
        self.video_sink = self._recorder.video_sink

    @property
    def step(self) -> int:
        return self._recorder.step

    @classmethod
    def for_test(cls, run_root: Path, *, policy_revision: str, simple_curriculum_collection: bool = False) -> "AutonomousRecorder":
        return cls(run_root, policy_revision=policy_revision, episode_id="episode-test", simple_curriculum_collection=simple_curriculum_collection)

    def record_step(self, observation: Mapping[str, object], action: object, *, reward: float, success: bool, request_id: str, chunk_offset: int) -> None:
        self._recorder.record_step(
            observation, action, reward=reward, success=success, action_source=ActionSource.POLICY,
            segment=0, policy_request_id=request_id, policy_chunk_offset=chunk_offset,
        )

    def record_snapshot(self, name: str, snapshot: Snapshot) -> None:
        if name not in {"reset", "terminal"}:
            raise ValueError("snapshot name must be reset or terminal before finalization")
        self._recorder.record_snapshot(name, snapshot)

    def record_continuation_snapshot(self, step: int, snapshot: Snapshot) -> None:
        self._recorder.record_continuation_snapshot(step, snapshot)

    def finish(
        self,
        *,
        reason: str,
        accepted_success: bool,
        visible_contact: Mapping[str, object] | None = None,
        fidelity: Mapping[str, object] | None = None,
    ) -> RecordedEpisode:
        return self._recorder.finish_autonomous(
            reason=reason,
            accepted_success=accepted_success,
            visible_contact=visible_contact,
            fidelity=fidelity,
        )


__all__ = [
    "AutonomousRecorder",
    "CANONICAL_CAMERA_NAMES",
    "CANONICAL_VIDEO_FILENAMES",
    "MixedSourceRecorder",
    "RecordedEpisode",
    "finite_vector",
]
