"""GR00T N1.7 adapter for the LeHome Isaac Sim evaluation interface.

The LeHome simulator exposes one flat 12-D joint observation and three HWC
camera frames.  GR00T's policy API uses nested modality dictionaries, with
one array per state/action group and an explicit batch/time dimension.  This
module is the only conversion boundary between those two contracts.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base_policy import BasePolicy
from .registry import PolicyRegistry


_INSTRUCTION = "fold the garment on the table"
_CAMERAS = ("top_rgb", "left_rgb", "right_rgb")
_STATE_GROUPS = (
    ("left_arm", slice(0, 5)),
    ("left_gripper", slice(5, 6)),
    ("right_arm", slice(6, 11)),
    ("right_gripper", slice(11, 12)),
)
_ACTION_GROUPS = tuple(name for name, _ in _STATE_GROUPS)


def _as_frame(value: Any, *, key: str) -> np.ndarray:
    frame = np.asarray(value)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"{key} must be an HWC RGB frame")
    if frame.dtype != np.uint8:
        raise ValueError(f"{key} must have dtype uint8, got {frame.dtype}")
    return frame


def _as_state(value: Any) -> np.ndarray:
    state = np.asarray(value)
    if state.shape != (12,):
        raise ValueError(f"observation.state must have shape (12,), got {state.shape}")
    state = state.astype(np.float32, copy=False)
    if not np.isfinite(state).all():
        raise ValueError("observation.state contains a non-finite value")
    return state


def build_groot_observation(
    observation: Mapping[str, Any],
    *,
    instruction: str = _INSTRUCTION,
) -> dict[str, Any]:
    """Convert one LeHome observation to the strict GR00T policy format."""

    if instruction != _INSTRUCTION:
        raise ValueError("LeHome GR00T training uses one fixed instruction")
    try:
        state = _as_state(observation["observation.state"])
        frames = {
            camera: _as_frame(observation[f"observation.images.{camera}"], key=camera)
            for camera in _CAMERAS
        }
    except KeyError as error:
        raise ValueError(f"missing LeHome observation key: {error.args[0]}") from None

    return {
        "video": {
            camera: frame[None, None, ...] for camera, frame in frames.items()
        },
        "state": {
            name: state[indices][None, None, ...]
            for name, indices in _STATE_GROUPS
        },
        "language": {"annotation.human.task_description": [[instruction]]},
    }


def flatten_groot_action(action: Mapping[str, Any]) -> np.ndarray:
    """Take the first predicted action step in the checked 12-D joint order."""

    parts: list[np.ndarray] = []
    for key in _ACTION_GROUPS:
        if key not in action:
            raise ValueError(f"GR00T action is missing {key}")
        values = np.asarray(action[key])
        if values.ndim != 3 or values.shape[0] != 1 or values.shape[1] < 1:
            raise ValueError(
                f"GR00T action {key} must have shape (1,T,D), got {values.shape}"
            )
        part = np.asarray(values[0, 0], dtype=np.float32).reshape(-1)
        expected = 5 if key.endswith("_arm") else 1
        if part.size != expected:
            raise ValueError(
                f"GR00T action {key} must have dimension {expected}, got {part.size}"
            )
        parts.append(part)
    result = np.concatenate(parts).astype(np.float32, copy=False)
    if result.shape != (12,) or not np.isfinite(result).all():
        raise ValueError("GR00T action is not a finite 12-D vector")
    return result


@PolicyRegistry.register("groot")
class GrootPolicy(BasePolicy):
    """Strict GR00T N1.7 policy wrapper used by ``scripts.eval``."""

    def __init__(
        self,
        *,
        model_path: str,
        device: str = "cuda",
        task_description: str = _INSTRUCTION,
        **_: Any,
    ) -> None:
        super().__init__()
        if not model_path:
            raise ValueError("model_path is required for the GR00T policy")
        if task_description != _INSTRUCTION:
            raise ValueError("task_description differs from the checked GR00T contract")
        try:
            from gr00t.data.embodiment_tags import EmbodimentTag
            from gr00t.policy import Gr00tPolicy as OfficialGr00tPolicy
        except ImportError as error:
            raise RuntimeError(
                "the pinned Isaac-GR00T runtime is required for --policy_type groot"
            ) from error
        runtime_device = "cuda:0" if device == "cuda" else device
        self._policy = OfficialGr00tPolicy(
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            model_path=model_path,
            device=runtime_device,
            strict=True,
        )

    def reset(self) -> None:
        self._policy.reset()

    def select_action(self, observation: Mapping[str, Any]) -> np.ndarray:
        groot_observation = build_groot_observation(observation)
        action, _ = self._policy.get_action(groot_observation)
        return flatten_groot_action(action)


__all__ = [
    "GrootPolicy",
    "build_groot_observation",
    "flatten_groot_action",
]
