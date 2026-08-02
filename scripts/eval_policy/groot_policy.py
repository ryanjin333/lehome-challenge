"""GR00T N1.7 adapter for the LeHome Isaac Sim evaluation interface.

The LeHome simulator exposes one flat 12-D joint observation and three HWC
camera frames.  GR00T's policy API uses nested modality dictionaries, with
one array per state/action group and an explicit batch/time dimension.  This
module is the only conversion boundary between those two contracts.
"""

from __future__ import annotations

from collections import deque
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
_ACTION_KEYS = tuple(f"action.{name}" for name in _ACTION_GROUPS)
_ACTION_DIMENSION = 12


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


def flatten_groot_action_chunk(action: Mapping[str, Any]) -> np.ndarray:
    """Flatten a GR00T action chunk to ``(horizon, 12)`` joint order.

    ``Gr00tPolicy.get_action`` returns one batch of denormalized, absolute
    actions with shape ``(1, T, D)`` for each modality group.  The LeHome
    environment consumes one 12-D target per simulator step, so the batch
    dimension is removed while the complete action horizon is retained.
    """

    parts: list[np.ndarray] = []
    horizon: int | None = None
    for group, key in zip(_ACTION_GROUPS, _ACTION_KEYS, strict=True):
        # ``parse_action_gr00t`` and ``Gr00tPolicy.get_action`` expose the
        # public action namespace as ``action.<group>``.  Accepting the bare
        # group as a compatibility fallback keeps this boundary usable with
        # older pinned GR00T builds, while still preferring the checked API.
        actual_key = key if key in action else group
        if actual_key not in action:
            raise ValueError(f"GR00T action is missing {key}")
        values = np.asarray(action[actual_key], dtype=np.float32)
        if values.ndim == 3:
            if values.shape[0] != 1:
                raise ValueError(
                    f"GR00T action {actual_key} must have batch size 1, got {values.shape}"
                )
            values = values[0]
        if values.ndim != 2 or values.shape[0] < 1:
            raise ValueError(
                f"GR00T action {actual_key} must have shape (1,T,D) or (T,D), got {values.shape}"
            )
        if horizon is None:
            horizon = values.shape[0]
        elif values.shape[0] != horizon:
            raise ValueError("GR00T action groups have different horizons")
        expected = 5 if group.endswith("_arm") else 1
        if values.shape[1] != expected:
            raise ValueError(
                f"GR00T action {actual_key} must have dimension {expected}, got {values.shape[1]}"
            )
        parts.append(values)
    result = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    if result.shape[1] != _ACTION_DIMENSION or not np.isfinite(result).all():
        raise ValueError("GR00T action chunk is not finite 12-D joint data")
    return result


def flatten_groot_action(action: Mapping[str, Any]) -> np.ndarray:
    """Take the first predicted action step in the checked 12-D joint order."""

    return flatten_groot_action_chunk(action)[0]


class ActionChunkQueue:
    """Small FIFO for consuming GR00T's action horizon between inferences."""

    def __init__(self) -> None:
        self._pending: deque[np.ndarray] = deque()

    def extend(self, chunk: np.ndarray) -> None:
        values = np.asarray(chunk, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != _ACTION_DIMENSION:
            raise ValueError(
                f"action chunk must have shape (T,{_ACTION_DIMENSION}), got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("action chunk contains a non-finite value")
        self._pending.extend(np.array(row, dtype=np.float32, copy=True) for row in values)

    def pop(self) -> np.ndarray | None:
        return self._pending.popleft() if self._pending else None

    def clear(self) -> None:
        self._pending.clear()


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
        self._action_queue = ActionChunkQueue()

    def reset(self) -> None:
        self._action_queue.clear()
        self._policy.reset()

    def select_action(self, observation: Mapping[str, Any]) -> np.ndarray:
        queued_action = self._action_queue.pop()
        if queued_action is not None:
            return queued_action
        groot_observation = build_groot_observation(observation)
        action, _ = self._policy.get_action(groot_observation)
        action_chunk = flatten_groot_action_chunk(action)
        self._action_queue.extend(action_chunk[1:])
        return action_chunk[0]


__all__ = [
    "GrootPolicy",
    "ActionChunkQueue",
    "build_groot_observation",
    "flatten_groot_action_chunk",
    "flatten_groot_action",
]
