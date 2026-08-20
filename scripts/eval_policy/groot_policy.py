"""GR00T N1.7 adapter for the LeHome Isaac Sim evaluation interface.

The LeHome simulator exposes one flat 12-D joint observation and three HWC
camera frames.  GR00T's policy API uses nested modality dictionaries, with
one array per state/action group and an explicit batch/time dimension.  This
module is the only conversion boundary between those two contracts.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from io import BytesIO
import json
import os
from pathlib import Path
from stat import S_ISREG
from time import perf_counter_ns, time_ns
from typing import Any, Mapping
from uuid import uuid4

import numpy as np

from lehome.flywheel.policy_protocol import (
    ACTION_HORIZON as _SESSION_ACTION_HORIZON,
    PolicyRequest,
    PolicyResponse,
    SessionStateError,
    pack_envelope,
    unpack_envelope,
    validate_response_for_request,
)

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
_ACTION_HORIZON = 16
_NDARRAY_MARKER = "__ndarray_class__"


def _msgpack():
    """Import the pinned wire dependency only for the server-backed policy."""

    try:
        import msgpack
    except ImportError as error:
        raise RuntimeError("groot_server requires pinned msgpack and pyzmq") from error
    return msgpack


def _encode_policy_server_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        stream = BytesIO()
        np.save(stream, value, allow_pickle=False)
        return {_NDARRAY_MARKER: True, "as_npy": stream.getvalue()}
    raise TypeError(f"cannot encode {type(value)!r} for GR00T policy server")


def _decode_policy_server_value(value: Any) -> Any:
    if not isinstance(value, dict) or value.get(_NDARRAY_MARKER) is not True:
        return value
    if set(value) != {_NDARRAY_MARKER, "as_npy"} or not isinstance(value["as_npy"], bytes):
        raise ValueError("policy server ndarray envelope is malformed")
    try:
        decoded = np.load(BytesIO(value["as_npy"]), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError("policy server ndarray payload is invalid") from error
    if not isinstance(decoded, np.ndarray) or decoded.dtype.hasobject:
        raise ValueError("policy server ndarray payload is unsafe")
    return decoded


def pack_policy_server_message(value: Any) -> bytes:
    """Match NVIDIA's PolicyServer msgpack/``.npy`` ndarray wire format."""

    return _msgpack().packb(value, default=_encode_policy_server_value, use_bin_type=True)


def unpack_policy_server_message(value: bytes) -> Any:
    """Decode a PolicyServer message without allowing object deserialization."""

    try:
        return _msgpack().unpackb(value, raw=False, object_hook=_decode_policy_server_value)
    except (TypeError, ValueError) as error:
        raise ValueError("policy server response is not valid msgpack") from error


class PolicyServerClient:
    """Finite-timeout, REQ-safe client for NVIDIA's loopback PolicyServer."""

    def __init__(
        self,
        endpoint: str,
        api_token: str,
        timeout_seconds: float,
        *,
        socket_factory: Any | None = None,
    ) -> None:
        if not endpoint.startswith("tcp://127.0.0.1:"):
            raise ValueError("policy server endpoint must use loopback TCP")
        if not api_token:
            raise ValueError("policy server API token is required")
        if timeout_seconds <= 0:
            raise ValueError("policy server request timeout must be positive")
        self._endpoint = endpoint
        self._api_token = api_token
        self._timeout_milliseconds = max(1, round(timeout_seconds * 1000))
        self._socket_factory = socket_factory or self._default_socket_factory
        self._socket: Any | None = None

    def _default_socket_factory(self) -> Any:
        try:
            import zmq
        except ImportError as error:
            raise RuntimeError("groot_server requires pinned msgpack and pyzmq") from error
        return zmq.Context.instance().socket(zmq.REQ)

    def _new_socket(self) -> Any:
        try:
            import zmq
        except ImportError as error:
            raise RuntimeError("groot_server requires pinned msgpack and pyzmq") from error
        socket = self._socket_factory()
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDTIMEO, self._timeout_milliseconds)
        socket.setsockopt(zmq.RCVTIMEO, self._timeout_milliseconds)
        socket.connect(self._endpoint)
        self._socket = socket
        return socket

    def _discard_socket(self) -> None:
        socket, self._socket = self._socket, None
        if socket is not None:
            try:
                socket.close(linger=0)
            except TypeError:  # narrow compatibility for minimal test doubles
                socket.close()

    def close(self) -> None:
        self._discard_socket()

    def request(self, endpoint: str, data: Any) -> Any:
        """Retry once only after destroying the poisoned REQ socket."""

        if not endpoint or not isinstance(endpoint, str):
            raise ValueError("policy server endpoint name is required")
        message = {"endpoint": endpoint, "data": data, "api_token": self._api_token}
        for attempt in range(2):
            socket = self._socket or self._new_socket()
            try:
                socket.send(pack_policy_server_message(message))
                response = unpack_policy_server_message(socket.recv())
                if isinstance(response, dict) and "error" in response:
                    if set(response) != {"error"} or not isinstance(response["error"], str):
                        raise ValueError("policy server error envelope is malformed")
                    raise RuntimeError(f"policy server error: {response['error']}")
                return response
            except RuntimeError:
                self._discard_socket()
                raise
            except Exception as error:
                self._discard_socket()
                if attempt:
                    raise RuntimeError("policy server request failed after socket reset") from error
        raise AssertionError("unreachable")

    def ping(self) -> None:
        if self.request("ping", {}) != {"status": "ok", "message": "Server is running"}:
            raise ValueError("policy server ping response is invalid")

    def reset(self) -> None:
        if self.request("reset", {"options": None}) != {}:
            raise ValueError("policy server reset response is invalid")

    def get_action(self, observation: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        response = self.request("get_action", {"observation": observation, "options": None})
        if not isinstance(response, list) or len(response) != 2:
            raise ValueError("policy server action response must be [action, info]")
        action, info = response
        if not isinstance(action, Mapping) or not isinstance(info, Mapping):
            raise ValueError("policy server action response has invalid members")
        return action, info


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
    if horizon != _ACTION_HORIZON:
        raise ValueError(
            f"GR00T action horizon must be {_ACTION_HORIZON}, got {horizon}"
        )
    result = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    if result.shape[1] != _ACTION_DIMENSION or not np.isfinite(result).all():
        raise ValueError("GR00T action chunk is not finite 12-D joint data")
    return result


def flatten_groot_action(action: Mapping[str, Any]) -> np.ndarray:
    """Take the first predicted action step in the checked 12-D joint order."""

    return flatten_groot_action_chunk(action)[0]


def validate_policy_server_action_chunk(action: Mapping[str, Any]) -> np.ndarray:
    """Reject server responses that drift from the exact GR00T action wire contract."""

    # ``run_groot_policy_server`` binds NVIDIA's raw ``Gr00tPolicy`` directly,
    # which returns bare modality keys.  Only ``Gr00tSimPolicyWrapper`` adds
    # the ``action.`` prefix, and that wrapper expects a different observation
    # contract than the nested request sent by this client.
    if set(action) != set(_ACTION_GROUPS):
        raise ValueError("policy server action keys must exactly match the GR00T contract")
    for group in _ACTION_GROUPS:
        values = action.get(group)
        expected_dimension = 5 if group.endswith("_arm") else 1
        if not isinstance(values, np.ndarray):
            raise ValueError(f"policy server action {group} must be an ndarray")
        if values.dtype != np.float32 or values.shape != (1, _ACTION_HORIZON, expected_dimension):
            raise ValueError(f"policy server action {group} has invalid dtype or shape")
    return flatten_groot_action_chunk(action)


class ActionChunkQueue:
    """Small FIFO for consuming GR00T's action horizon between inferences."""

    def __init__(self) -> None:
        self._pending: deque[QueuedAction] = deque()

    def extend(self, chunk: np.ndarray, *, request_id: str = "legacy") -> None:
        values = np.asarray(chunk, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != _ACTION_DIMENSION:
            raise ValueError(
                f"action chunk must have shape (T,{_ACTION_DIMENSION}), got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("action chunk contains a non-finite value")
        if not request_id:
            raise ValueError("action request ID must be non-empty")
        self._pending.extend(
            QueuedAction(np.array(row, dtype=np.float32, copy=True), request_id, offset)
            for offset, row in enumerate(values)
        )

    def pop(self) -> np.ndarray | None:
        queued = self.pop_with_provenance()
        return None if queued is None else queued.value

    def pop_with_provenance(self) -> "QueuedAction | None":
        return self._pending.popleft() if self._pending else None

    def pop_with_provenance_required(self) -> "QueuedAction":
        queued = self.pop_with_provenance()
        if queued is None:
            raise RuntimeError("action queue unexpectedly emptied")
        return queued

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def clear(self) -> None:
        self._pending.clear()


@dataclass(frozen=True, slots=True)
class QueuedAction:
    value: np.ndarray
    request_id: str
    chunk_offset: int


class SessionPolicyClient:
    """Appliance-only DEALER client for the session-aware rollout gateway.

    It is intentionally not a ``PolicyServerClient`` subclass: its msgpack
    envelopes and reset semantics are incompatible with NVIDIA's synchronous
    REP server used by the legacy evaluation path.
    """

    def __init__(
        self,
        endpoint: str,
        policy_sha256: str,
        timeout_seconds: float,
        *,
        session_id: str | None = None,
        socket_factory: Any | None = None,
        request_transport: Any | None = None,
        now_ns: Any = time_ns,
    ) -> None:
        if not endpoint.startswith("tcp://127.0.0.1:"):
            raise ValueError("session policy gateway endpoint must use loopback TCP")
        if timeout_seconds <= 0:
            raise ValueError("session policy gateway timeout must be positive")
        if not isinstance(policy_sha256, str) or len(policy_sha256) != 64:
            raise ValueError("session policy gateway requires a policy SHA-256")
        self._endpoint = endpoint
        self._policy_sha256 = policy_sha256
        self._timeout_ns = max(1, round(timeout_seconds * 1_000_000_000))
        self._timeout_milliseconds = max(1, round(timeout_seconds * 1_000))
        self._session_id = session_id or uuid4().hex
        self._socket_factory = socket_factory or self._default_socket_factory
        self._request_transport = request_transport
        self._now_ns = now_ns
        self._socket: Any | None = None
        self._action_queue = ActionChunkQueue()
        self._episode_generation = 0
        self._request_sequence = 0
        self._session_ready = False
        self._session_started = False

    def _default_socket_factory(self) -> Any:
        try:
            import zmq
        except ImportError as error:
            raise RuntimeError("session gateway requires pinned pyzmq") from error
        return zmq.Context.instance().socket(zmq.DEALER)

    def _new_socket(self) -> Any:
        try:
            import zmq
        except ImportError as error:
            raise RuntimeError("session gateway requires pinned pyzmq") from error
        socket = self._socket_factory()
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDTIMEO, self._timeout_milliseconds)
        socket.setsockopt(zmq.RCVTIMEO, self._timeout_milliseconds)
        socket.setsockopt(zmq.IDENTITY, f"lehome-policy-{self._session_id}".encode("ascii"))
        socket.connect(self._endpoint)
        self._socket = socket
        return socket

    def _discard_socket(self) -> None:
        socket, self._socket = self._socket, None
        if socket is not None:
            try:
                socket.close(linger=0)
            except TypeError:
                socket.close()

    def close(self) -> None:
        self._discard_socket()

    def _deadline_ns(self) -> int:
        return self._now_ns() + self._timeout_ns

    def _next_request_id(self) -> str:
        self._request_sequence += 1
        return f"{self._session_id}:{self._request_sequence:020d}"

    def _exchange(self, request: PolicyRequest) -> PolicyResponse:
        try:
            payload = pack_envelope(request)
            if self._request_transport is not None:
                received = self._request_transport(payload)
            else:
                socket = self._socket or self._new_socket()
                socket.send(payload)
                received = socket.recv()
            response = unpack_envelope(received)
            if not isinstance(response, PolicyResponse):
                raise ValueError("session gateway returned a request instead of a response")
            return response
        except Exception:
            # A DEALER socket may retain a late reply after a receive timeout.
            # It cannot be reused because the next request could consume that
            # reply and bind it to the wrong request identity.
            if self._request_transport is None:
                self._discard_socket()
            raise

    def _send_reset(self, *, advance_generation: bool) -> None:
        if advance_generation:
            self._episode_generation += 1
            self._session_started = True
        request = PolicyRequest.reset(
            session_id=self._session_id,
            episode_generation=self._episode_generation,
            request_id=self._next_request_id(),
            policy_sha256=self._policy_sha256,
            deadline_ns=self._deadline_ns(),
        )
        response = self._exchange(request)
        validate_response_for_request(response, request, now_ns=self._now_ns())
        self._session_ready = True

    def _ensure_session(self) -> None:
        if self._session_ready:
            return
        self._send_reset(advance_generation=not self._session_started)

    def reset(self) -> None:
        """Begin a new episode generation and discard every cached action."""

        self._action_queue.clear()
        self._session_ready = False
        self._send_reset(advance_generation=True)

    @property
    def episode_generation(self) -> int:
        """The gateway generation acknowledged by the most recent reset."""

        return self._episode_generation

    @property
    def action_horizon(self) -> int:
        """Action chunks are kept locally at the protocol's fixed H=16."""

        return _SESSION_ACTION_HORIZON

    def cancel(self, request_id: str) -> None:
        """Tell the gateway that an outstanding inference must not be routed."""

        self._ensure_session()
        request = PolicyRequest.cancel(
            session_id=self._session_id,
            episode_generation=self._episode_generation,
            request_id=self._next_request_id(),
            policy_sha256=self._policy_sha256,
            deadline_ns=self._deadline_ns(),
            cancelled_request_id=request_id,
        )
        response = self._exchange(request)
        validate_response_for_request(response, request, now_ns=self._now_ns())

    def _gateway_observation(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Send the GR00T video/state/language contract, not raw Isaac keys."""

        if "observation.state" in observation and all(
            f"observation.images.{camera}" in observation for camera in _CAMERAS
        ):
            return build_groot_observation(observation)
        return dict(observation)

    def _request_action_chunk(self, observation: Mapping[str, Any]) -> tuple[np.ndarray, str]:
        self._ensure_session()
        request = PolicyRequest.infer(
            session_id=self._session_id,
            episode_generation=self._episode_generation,
            request_id=self._next_request_id(),
            policy_sha256=self._policy_sha256,
            deadline_ns=self._deadline_ns(),
            observation=self._gateway_observation(observation),
        )
        try:
            response = self._exchange(request)
            validate_response_for_request(response, request, now_ns=self._now_ns())
        except SessionStateError:
            # The only recoverable protocol rejection is a restarted gateway
            # that forgot this otherwise-live session.  Replay a reset without
            # advancing the worker's episode, then preserve the request ID.
            if response.error_code != "unknown_session":
                raise
            self._discard_socket()
            self._session_ready = False
            self._ensure_session()
            response = self._exchange(request)
            validate_response_for_request(response, request, now_ns=self._now_ns())
        if response.action_chunk is None:
            raise ValueError("inference response did not include an action chunk")
        expected_bytes = _SESSION_ACTION_HORIZON * _ACTION_DIMENSION * np.dtype(np.float32).itemsize
        if len(response.action_chunk) != expected_bytes:
            raise ValueError("session gateway action chunk has invalid byte length")
        return (
            np.frombuffer(response.action_chunk, dtype=np.float32).reshape(
                _SESSION_ACTION_HORIZON, _ACTION_DIMENSION
            ).copy(),
            request.request_id,
        )

    def select_action(self, observation: Mapping[str, Any]) -> np.ndarray:
        return self.select_action_with_provenance(observation).value

    def select_action_with_provenance(self, observation: Mapping[str, Any]) -> QueuedAction:
        queued_action = self._action_queue.pop_with_provenance()
        if queued_action is not None:
            return queued_action
        chunk, request_id = self._request_action_chunk(observation)
        self._action_queue.extend(chunk, request_id=request_id)
        return self._action_queue.pop_with_provenance_required()


def _append_policy_telemetry(*, request_id: str, latency_seconds: float, queue_depth_after_enqueue: int) -> None:
    """Append one strict record to the campaign-provisioned telemetry file."""
    raw_path = os.environ.get("LEHOME_FLYWHEEL_POLICY_TELEMETRY_PATH")
    if not raw_path:
        return
    path = Path(raw_path)
    flags = os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as error:
        raise RuntimeError("policy telemetry path is unsafe or unavailable") from error
    try:
        if not S_ISREG(os.fstat(fd).st_mode):
            raise RuntimeError("policy telemetry path is not a regular file")
        payload = json.dumps(
            {
                "request_id": request_id,
                "latency_seconds": latency_seconds,
                "queue_depth_after_enqueue": queue_depth_after_enqueue,
            },
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        written = os.write(fd, payload)
        if written != len(payload):
            raise RuntimeError("policy telemetry record was not fully written")
    finally:
        os.close(fd)


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
        return self.select_action_with_provenance(observation).value

    def select_action_with_provenance(self, observation: Mapping[str, Any]) -> QueuedAction:
        queued_action = self._action_queue.pop_with_provenance()
        if queued_action is not None:
            return queued_action
        request_id = uuid4().hex
        groot_observation = build_groot_observation(observation)
        inference_started = perf_counter_ns()
        action, _ = self._policy.get_action(groot_observation)
        inference_finished = perf_counter_ns()
        action_chunk = flatten_groot_action_chunk(action)
        self._action_queue.extend(action_chunk, request_id=request_id)
        _append_policy_telemetry(
            request_id=request_id,
            latency_seconds=(inference_finished - inference_started) / 1_000_000_000,
            queue_depth_after_enqueue=self._action_queue.pending_count,
        )
        return self._action_queue.pop_with_provenance_required()


@PolicyRegistry.register("groot_server")
class GrootServerPolicy(BasePolicy):
    """GR00T policy adapter whose incompatible runtime remains out of Isaac."""

    def __init__(
        self,
        *,
        policy_server_endpoint: str,
        policy_server_token_env: str,
        policy_server_request_timeout: float,
        task_description: str = _INSTRUCTION,
        **_: Any,
    ) -> None:
        super().__init__()
        if task_description != _INSTRUCTION:
            raise ValueError("task_description differs from the checked GR00T contract")
        if not policy_server_token_env:
            raise ValueError("policy server token environment variable is required")
        token = os.environ.get(policy_server_token_env, "")
        self._client = PolicyServerClient(
            policy_server_endpoint,
            token,
            policy_server_request_timeout,
        )
        self._action_queue = ActionChunkQueue()

    def reset(self) -> None:
        self._action_queue.clear()
        self._client.reset()

    def select_action(self, observation: Mapping[str, Any]) -> np.ndarray:
        return self.select_action_with_provenance(observation).value

    def select_action_with_provenance(self, observation: Mapping[str, Any]) -> QueuedAction:
        queued_action = self._action_queue.pop_with_provenance()
        if queued_action is not None:
            return queued_action
        request_id = uuid4().hex
        inference_started = perf_counter_ns()
        action, _info = self._client.get_action(build_groot_observation(observation))
        inference_finished = perf_counter_ns()
        action_chunk = validate_policy_server_action_chunk(action)
        self._action_queue.extend(action_chunk, request_id=request_id)
        _append_policy_telemetry(
            request_id=request_id,
            latency_seconds=(inference_finished - inference_started) / 1_000_000_000,
            queue_depth_after_enqueue=self._action_queue.pending_count,
        )
        return self._action_queue.pop_with_provenance_required()


__all__ = [
    "GrootPolicy",
    "GrootServerPolicy",
    "PolicyServerClient",
    "SessionPolicyClient",
    "ActionChunkQueue",
    "QueuedAction",
    "build_groot_observation",
    "flatten_groot_action_chunk",
    "flatten_groot_action",
    "validate_policy_server_action_chunk",
    "pack_policy_server_message",
    "unpack_policy_server_message",
]
