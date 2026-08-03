"""Isaac-facing recorder that writes immutable autonomous-policy evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import subprocess
import time
from typing import Mapping
from uuid import uuid4

import numpy as np

from .artifacts import EpisodeArtifactWriter
from .models import ActionSource, EpisodeFrame


_CAMERAS = ("top_rgb", "left_rgb", "right_rgb")


def finite_vector(value: object, *, size: int, name: str) -> tuple[float, ...]:
    values = np.asarray(value, dtype=np.float32)
    if values.shape != (size,) or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return tuple(float(item) for item in values)


class _VideoSink:
    def __init__(self) -> None:
        self._frames: dict[str, list[np.ndarray]] = {camera: [] for camera in _CAMERAS}

    def append(self, observation: Mapping[str, object]) -> None:
        for camera in _CAMERAS:
            key = f"observation.images.{camera}"
            frame = np.asarray(observation[key])
            if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
                raise ValueError(f"{key} must be an HWC uint8 RGB frame")
            self._frames[camera].append(np.array(frame, copy=True))

    def count(self, camera: str) -> int:
        return len(self._frames[camera])

    def encode(self, root: Path, *, fps: int = 30) -> tuple[str, ...]:
        encoded: list[str] = []
        for camera in _CAMERAS:
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


class AutonomousRecorder:
    """Record exactly the actions handed to one Isaac environment's ``step``."""

    def __init__(self, run_root: Path, *, policy_revision: str, episode_id: str | None = None) -> None:
        if len(policy_revision) != 40 or any(char not in "0123456789abcdef" for char in policy_revision):
            raise ValueError("policy_revision must be a pinned 40-character revision")
        self.writer = EpisodeArtifactWriter(run_root, episode_id or f"episode-{uuid4().hex}")
        self.policy_revision = policy_revision
        self.video_sink = _VideoSink()
        self.step = 0
        self._annotations: list[dict[str, object]] = []
        self._finished = False

    @classmethod
    def for_test(cls, run_root: Path, *, policy_revision: str) -> "AutonomousRecorder":
        return cls(run_root, policy_revision=policy_revision, episode_id="episode-test")

    def record_step(
        self,
        observation: Mapping[str, object],
        action: object,
        *,
        reward: float,
        success: bool,
        request_id: str,
        chunk_offset: int,
    ) -> None:
        if self._finished:
            raise ValueError("recorder has already finished")
        if not request_id or not isinstance(chunk_offset, int) or chunk_offset < 0:
            raise ValueError("policy request provenance is invalid")
        if not isinstance(reward, (int, float)) or not math.isfinite(reward):
            raise ValueError("reward must be finite")
        frame = EpisodeFrame(
            step=self.step,
            monotonic_ns=time.monotonic_ns(),
            wall_time_ns=time.time_ns(),
            state=finite_vector(observation["observation.state"], size=12, name="state"),
            action=finite_vector(action, size=12, name="action"),
            action_source=ActionSource.POLICY,
            reward=float(reward),
            success=bool(success),
            segment=0,
            policy_request_id=request_id,
            policy_chunk_offset=chunk_offset,
        )
        annotation = asdict(frame)
        annotation["action_source"] = frame.action_source.value
        self.writer.append_annotation(annotation)
        self._annotations.append(annotation)
        self.video_sink.append(observation)
        self.step += 1

    def finish(self, *, reason: str, accepted_success: bool) -> RecordedEpisode:
        if self._finished:
            raise ValueError("recorder has already finished")
        if not reason:
            raise ValueError("terminal reason is required")
        self._finished = True
        episode: dict[str, object] = {
            "policy_revision": self.policy_revision,
            "terminal_reason": reason,
            "accepted_success": bool(accepted_success),
            "outcome": "success" if accepted_success else "timeout",
            # Autonomous policy data is diagnostic evidence, never an initial BC target.
            "bc_target_count": 0,
        }
        required_videos: tuple[str, ...] = ()
        try:
            if any(self.video_sink.count(camera) != self.step for camera in _CAMERAS):
                raise ValueError("video frame count does not match annotation count")
            required_videos = self.video_sink.encode(self.writer.staging)
        except Exception as error:
            # Preserve a terminal diagnostic episode even when a local encoder fails.
            episode["outcome"] = "error"
            episode["accepted_success"] = False
            episode["recorder_error"] = str(error)
        path = self.writer.finalize(episode, required_videos=required_videos)
        return RecordedEpisode(path=path, episode=episode | {"episode_id": path.name}, annotations=tuple(self._annotations))


__all__ = ["AutonomousRecorder", "RecordedEpisode", "finite_vector"]
