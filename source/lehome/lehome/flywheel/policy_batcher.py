"""Pure, bounded batching for the session-aware GR00T rollout gateway.

The transport owns admission and response routing.  This module only groups
already-admitted inference requests, invokes one model call, and binds each
split action chunk back to its exact request.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import sha256
from time import monotonic_ns
from typing import Any, Callable, Mapping

import numpy as np

from .policy_protocol import ACTION_HORIZON, PolicyRequest, PolicyResponse


_ACTION_GROUPS = ("left_arm", "left_gripper", "right_arm", "right_gripper")
_ACTION_DIMENSIONS = (5, 1, 5, 1)


@dataclass(frozen=True, slots=True)
class BatchResult:
    """A response that must be routed to ``request``'s original socket peer."""

    request: PolicyRequest
    response: PolicyResponse


@dataclass(frozen=True, slots=True)
class DiscardedInference:
    """An admitted request dropped before a model invocation."""

    request: PolicyRequest
    reason: str


class BatchFlush(list[BatchResult]):
    """Batch responses plus auditable pre-inference discards."""

    def __init__(
        self,
        results: list[BatchResult] | None = None,
        *,
        discarded: list[DiscardedInference] | None = None,
    ) -> None:
        super().__init__(results or [])
        self.discarded = tuple(discarded or [])


@dataclass(frozen=True, slots=True)
class BatchReceipt:
    """Immutable evidence for one successful physical model invocation."""

    policy_sha256: str
    seed_identity: int
    identities: tuple[tuple[str, int, str], ...]
    batch_occupancy: int
    max_batch_size: int
    batch_wait_ns: int
    model_latency_ns: int
    returned_action_chunk_sha256: tuple[str, ...]

    def to_wire(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "policy_sha256": self.policy_sha256,
            "seed_identity": self.seed_identity,
            "identities": [
                {"session_id": session_id, "episode_generation": generation, "request_id": request_id}
                for session_id, generation, request_id in self.identities
            ],
            "batch_occupancy": self.batch_occupancy,
            "max_batch_size": self.max_batch_size,
            "batch_wait_ns": self.batch_wait_ns,
            "model_latency_ns": self.model_latency_ns,
            "returned_action_chunk_sha256": list(self.returned_action_chunk_sha256),
        }


@dataclass(frozen=True, slots=True)
class _QueuedInference:
    request: PolicyRequest
    received_ns: int


def _collate(values: list[Any], *, path: str = "observation") -> Any:
    """Concatenate individually batched, structurally identical GR00T inputs."""

    first = values[0]
    if isinstance(first, Mapping):
        keys = set(first)
        if any(not isinstance(value, Mapping) or set(value) != keys for value in values[1:]):
            raise ValueError(f"{path} mappings do not have identical keys")
        return {
            key: _collate([value[key] for value in values], path=f"{path}.{key}")
            for key in first
        }
    if isinstance(first, np.ndarray):
        if any(not isinstance(value, np.ndarray) for value in values):
            raise ValueError(f"{path} mixes ndarray and non-ndarray values")
        if first.ndim < 1 or first.shape[0] != 1:
            raise ValueError(f"{path} must have an individual batch dimension of one")
        if any(value.dtype != first.dtype or value.shape != first.shape for value in values[1:]):
            raise ValueError(f"{path} ndarrays do not have identical dtype and shape")
        if first.dtype.hasobject or not np.isfinite(first).all():
            raise ValueError(f"{path} ndarray is unsafe or non-finite")
        if any(value.dtype.hasobject or not np.isfinite(value).all() for value in values[1:]):
            raise ValueError(f"{path} ndarray is unsafe or non-finite")
        return np.concatenate(values, axis=0)
    if isinstance(first, list):
        if any(not isinstance(value, list) or len(value) != 1 for value in values):
            raise ValueError(f"{path} must contain one language item per request")
        return [value[0] for value in values]
    raise ValueError(f"{path} must be a nested mapping, ndarray, or one-item list")


def collate_observations(observations: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate strict single-item GR00T observations along their batch axis."""

    if not observations or any(not isinstance(observation, Mapping) for observation in observations):
        raise ValueError("batch must contain observations")
    for observation in observations:
        if set(observation) != {"video", "state", "language"}:
            raise ValueError("observation keys do not match the GR00T contract")
        video, state, language = observation["video"], observation["state"], observation["language"]
        if not isinstance(video, Mapping) or set(video) != {"top_rgb", "left_rgb", "right_rgb"}:
            raise ValueError("video keys do not match the GR00T contract")
        if not isinstance(state, Mapping) or set(state) != set(_ACTION_GROUPS):
            raise ValueError("state keys do not match the GR00T contract")
        if not isinstance(language, Mapping) or set(language) != {"annotation.human.task_description"}:
            raise ValueError("language keys do not match the GR00T contract")
        for camera, frame in video.items():
            if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 5 or frame.shape[:2] != (1, 1) or frame.shape[-1] != 3:
                raise ValueError(f"video.{camera} has invalid dtype or shape")
        for group, dimension in zip(_ACTION_GROUPS, _ACTION_DIMENSIONS, strict=True):
            values = state[group]
            if not isinstance(values, np.ndarray) or values.dtype != np.float32 or values.shape != (1, 1, dimension):
                raise ValueError(f"state.{group} has invalid dtype or shape")
        items = language["annotation.human.task_description"]
        if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], list):
            raise ValueError("language task description has invalid shape")
    return _collate(list(observations))


def _split_actions(action: object, *, batch_size: int) -> list[np.ndarray]:
    if not isinstance(action, Mapping):
        raise ValueError("model action must be a mapping")
    canonical = tuple(f"action.{group}" for group in _ACTION_GROUPS)
    keys = tuple(action)
    if set(keys) == set(_ACTION_GROUPS):
        actual_keys = _ACTION_GROUPS
    elif set(keys) == set(canonical):
        actual_keys = canonical
    else:
        raise ValueError("model action keys do not match the GR00T contract")
    parts: list[np.ndarray] = []
    for key, dimension in zip(actual_keys, _ACTION_DIMENSIONS, strict=True):
        values = action[key]
        if not isinstance(values, np.ndarray) or values.dtype != np.float32:
            raise ValueError(f"model action {key} must be float32 ndarray")
        if values.shape != (batch_size, ACTION_HORIZON, dimension) or not np.isfinite(values).all():
            raise ValueError(f"model action {key} has invalid shape or values")
        parts.append(values)
    flattened = np.concatenate(parts, axis=2)
    return [flattened[index].copy() for index in range(batch_size)]


class PolicyBatcher:
    """Bound inference calls to at most four independent session requests."""

    def __init__(
        self,
        model: Any,
        *,
        policy_sha256: str,
        max_batch_size: int = 4,
        batch_window_ns: int = 5_000_000,
        seed_identity: int = 0,
        clock_ns: Callable[[], int] = monotonic_ns,
    ) -> None:
        if not isinstance(policy_sha256, str) or len(policy_sha256) != 64:
            raise ValueError("policy_sha256 must be a SHA-256 hex digest")
        if not 1 <= max_batch_size <= 4:
            raise ValueError("max_batch_size must be in 1..4")
        if batch_window_ns <= 0:
            raise ValueError("batch_window_ns must be positive")
        if not isinstance(seed_identity, int) or isinstance(seed_identity, bool) or not 0 <= seed_identity < 2**32:
            raise ValueError("seed_identity must be in 0..2^32-1")
        if not callable(getattr(model, "get_action", None)):
            raise ValueError("model must expose get_action")
        self._model = model
        self._policy_sha256 = policy_sha256
        self._max_batch_size = max_batch_size
        self._batch_window_ns = batch_window_ns
        self._seed_identity = seed_identity
        self._clock_ns = clock_ns
        self._pending: deque[_QueuedInference] = deque()
        self._cancelled: set[tuple[str, int, str]] = set()
        self._receipts: deque[BatchReceipt] = deque()
        self.model_calls = 0

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def enqueue(self, request: PolicyRequest, *, received_ns: int) -> None:
        if request.operation != "infer":
            raise ValueError("only inference requests may enter the policy batcher")
        if request.policy_sha256 != self._policy_sha256:
            raise ValueError("request policy digest differs from batcher policy")
        self._pending.append(_QueuedInference(request=request, received_ns=received_ns))

    def cancel(self, request: PolicyRequest) -> bool:
        """Mark an inference stale and report whether it remains queued."""

        identity = (request.session_id, request.episode_generation, request.request_id)
        queued = any(
            (item.request.session_id, item.request.episode_generation, item.request.request_id) == identity
            for item in self._pending
        )
        self._cancelled.add(identity)
        return queued

    def drain_receipts(self) -> list[dict[str, object]]:
        receipts = [receipt.to_wire() for receipt in self._receipts]
        self._receipts.clear()
        return receipts

    def _is_due(self, now_ns: int) -> bool:
        if len(self._pending) >= self._max_batch_size:
            return True
        oldest = self._pending[0]
        return (
            now_ns >= oldest.received_ns + self._batch_window_ns
            or any(now_ns + self._batch_window_ns >= item.request.deadline_ns for item in self._pending)
        )

    def flush(
        self,
        *,
        now_ns: int,
        is_live: Callable[[PolicyRequest, int], bool] | None = None,
    ) -> BatchFlush:
        """Process one due batch and return exact request/response bindings."""

        if not self._pending or not self._is_due(now_ns):
            return BatchFlush()
        candidates: list[_QueuedInference] = []
        discarded: list[DiscardedInference] = []
        while self._pending and len(candidates) < self._max_batch_size:
            candidate = self._pending.popleft()
            identity = (candidate.request.session_id, candidate.request.episode_generation, candidate.request.request_id)
            if identity in self._cancelled:
                discarded.append(DiscardedInference(candidate.request, "cancelled"))
                continue
            if now_ns >= candidate.request.deadline_ns:
                discarded.append(DiscardedInference(candidate.request, "expired"))
                continue
            if is_live is not None and not is_live(candidate.request, now_ns):
                discarded.append(DiscardedInference(candidate.request, "stale"))
                continue
            candidates.append(candidate)
        if not candidates:
            return BatchFlush(discarded=discarded)
        requests = [candidate.request for candidate in candidates]
        try:
            observation = collate_observations([request.observation or {} for request in requests])
        except ValueError:
            return BatchFlush(
                [BatchResult(request, PolicyResponse.error(request, error_code="invalid_observation")) for request in requests],
                discarded=discarded,
            )
        try:
            inference_started_ns = self._clock_ns()
            self.model_calls += 1
            returned = self._model.get_action(observation)
            action = returned[0] if isinstance(returned, tuple) else returned
            chunks = _split_actions(action, batch_size=len(requests))
        except (TypeError, ValueError, KeyError, IndexError):
            return BatchFlush(
                [BatchResult(request, PolicyResponse.error(request, error_code="invalid_model_action")) for request in requests],
                discarded=discarded,
            )
        inference_finished_ns = self._clock_ns()
        self._receipts.append(
            BatchReceipt(
                policy_sha256=self._policy_sha256,
                seed_identity=self._seed_identity,
                identities=tuple(
                    (request.session_id, request.episode_generation, request.request_id)
                    for request in requests
                ),
                batch_occupancy=len(requests),
                max_batch_size=self._max_batch_size,
                batch_wait_ns=max(0, inference_started_ns - min(item.received_ns for item in candidates)),
                model_latency_ns=max(0, inference_finished_ns - inference_started_ns),
                returned_action_chunk_sha256=tuple(sha256(chunk.tobytes()).hexdigest() for chunk in chunks),
            )
        )
        return BatchFlush(
            [
                BatchResult(
                    request=request,
                    response=PolicyResponse.ok(request, action_chunk=chunk.tobytes(), action_horizon=ACTION_HORIZON),
                )
                for request, chunk in zip(requests, chunks, strict=True)
            ],
            discarded=discarded,
        )


__all__ = [
    "BatchFlush",
    "BatchReceipt",
    "BatchResult",
    "DiscardedInference",
    "PolicyBatcher",
    "collate_observations",
]
