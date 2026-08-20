"""Versioned, session-aware envelopes for the rollout policy gateway.

This is deliberately separate from NVIDIA's synchronous ``PolicyServer``
wire protocol.  The rollout appliance uses it between persistent workers and
the one batched gateway; legacy evaluation keeps its existing REP client.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from typing import Any, Mapping

import msgpack
import numpy as np


SCHEMA_VERSION = 1
ACTION_HORIZON = 16
REQUIRED_ENVELOPE_FIELDS = (
    "schema_version",
    "session_id",
    "episode_generation",
    "request_id",
    "policy_sha256",
    "deadline_ns",
)
_NDARRAY_MARKER = "__ndarray_class__"


class PolicyProtocolError(ValueError):
    """Base class for invalid or rejected session-gateway traffic."""


class DuplicateRequestError(PolicyProtocolError):
    """A request ID has already been used in this session generation."""


class ExpiredRequestError(PolicyProtocolError):
    """The request or matching response arrived after its deadline."""


class StaleResponseError(PolicyProtocolError):
    """A response does not bind to the request currently awaiting it."""


class SessionStateError(PolicyProtocolError):
    """A request targets a missing, wrong, or cancelled session generation."""


class PolicyDigestError(PolicyProtocolError):
    """A request does not target the gateway's loaded policy."""


def _require_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_sha256(value: object, *, field: str = "policy_sha256") -> str:
    value = _require_string(value, field=field)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _require_generation(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("episode_generation must be a non-negative integer")
    return value


def _require_deadline(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("deadline_ns must be a positive integer")
    return value


def _canonical_value(value: Any) -> Any:
    """Return JSON-safe data with mapping keys ordered recursively."""

    if isinstance(value, Mapping):
        return {str(key): _canonical_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, tuple):
        return [_canonical_value(item) for item in value]
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    if isinstance(value, bytes):
        return {"__bytes_sha256__": sha256(value).hexdigest(), "length": len(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"envelope canonicalization does not support {type(value)!r}")


def _encode_msgpack_value(value: Any) -> Any:
    """Encode ndarray payloads with NumPy's non-pickle ``.npy`` format."""

    if not isinstance(value, np.ndarray):
        raise TypeError(f"cannot encode {type(value)!r} in a policy envelope")
    if value.dtype.hasobject:
        raise ValueError("policy envelope ndarray payload must not contain Python objects")
    stream = BytesIO()
    np.save(stream, value, allow_pickle=False)
    return {_NDARRAY_MARKER: True, "as_npy": stream.getvalue()}


def _decode_msgpack_value(value: Any) -> Any:
    if not isinstance(value, dict) or value.get(_NDARRAY_MARKER) is not True:
        return value
    if set(value) != {_NDARRAY_MARKER, "as_npy"} or not isinstance(value["as_npy"], bytes):
        raise ValueError("policy envelope ndarray payload is malformed")
    try:
        decoded = np.load(BytesIO(value["as_npy"]), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError("policy envelope ndarray payload is invalid") from error
    if not isinstance(decoded, np.ndarray) or decoded.dtype.hasobject:
        raise ValueError("policy envelope ndarray payload is unsafe")
    return decoded


@dataclass(frozen=True, slots=True)
class PolicyRequest:
    """A request addressed to one session and one episode generation."""

    operation: str
    session_id: str
    episode_generation: int
    request_id: str
    policy_sha256: str
    deadline_ns: int
    observation: Mapping[str, Any] | None = None
    cancelled_request_id: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        if self.operation not in {"infer", "reset", "cancel"}:
            raise ValueError("operation must be infer, reset, or cancel")
        _require_string(self.session_id, field="session_id")
        _require_generation(self.episode_generation)
        _require_string(self.request_id, field="request_id")
        _require_sha256(self.policy_sha256)
        _require_deadline(self.deadline_ns)
        if self.operation == "infer":
            if not isinstance(self.observation, Mapping):
                raise ValueError("infer request observation must be a mapping")
            if self.cancelled_request_id is not None:
                raise ValueError("infer request must not name a cancelled request")
        elif self.operation == "cancel":
            _require_string(self.cancelled_request_id, field="cancelled_request_id")
            if self.observation is not None:
                raise ValueError("cancel request must not contain an observation")
        elif self.observation is not None or self.cancelled_request_id is not None:
            raise ValueError("reset request must not contain a payload")

    @classmethod
    def infer(cls, **kwargs: Any) -> "PolicyRequest":
        return cls(operation="infer", **kwargs)

    @classmethod
    def reset(cls, **kwargs: Any) -> "PolicyRequest":
        return cls(operation="reset", **kwargs)

    @classmethod
    def cancel(cls, **kwargs: Any) -> "PolicyRequest":
        return cls(operation="cancel", **kwargs)

    def to_wire(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "message_type": "request",
            "operation": self.operation,
            **_identity_wire(self),
        }
        if self.operation == "infer":
            result["observation"] = dict(self.observation or {})
        if self.operation == "cancel":
            result["cancelled_request_id"] = self.cancelled_request_id
        return result

    def canonical_metadata_bytes(self) -> bytes:
        return _canonical_bytes(self.to_wire())


@dataclass(frozen=True, slots=True)
class PolicyResponse:
    """The only response envelope accepted by a session policy client."""

    session_id: str
    episode_generation: int
    request_id: str
    policy_sha256: str
    deadline_ns: int
    status: str
    action_chunk: bytes | None = None
    action_horizon: int | None = None
    action_chunk_sha256: str | None = None
    error_code: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        _require_string(self.session_id, field="session_id")
        _require_generation(self.episode_generation)
        _require_string(self.request_id, field="request_id")
        _require_sha256(self.policy_sha256)
        _require_deadline(self.deadline_ns)
        if self.status not in {"ok", "error"}:
            raise ValueError("response status must be ok or error")
        if self.status == "ok":
            if self.action_chunk is not None:
                if not isinstance(self.action_chunk, bytes):
                    raise ValueError("action_chunk must be bytes")
                if self.action_horizon != ACTION_HORIZON:
                    raise ValueError(f"action_horizon must be {ACTION_HORIZON}")
                if self.action_chunk_sha256 != sha256(self.action_chunk).hexdigest():
                    raise ValueError("action_chunk_sha256 does not match action_chunk")
            elif self.action_horizon is not None or self.action_chunk_sha256 is not None:
                raise ValueError("empty OK response must not contain action metadata")
            if self.error_code is not None:
                raise ValueError("OK response must not contain error_code")
        elif self.action_chunk is not None or self.action_horizon is not None or self.action_chunk_sha256 is not None:
            raise ValueError("error response must not contain an action")
        else:
            _require_string(self.error_code, field="error_code")

    @classmethod
    def ok(
        cls,
        request: PolicyRequest,
        *,
        action_chunk: bytes | None = None,
        action_horizon: int | None = None,
    ) -> "PolicyResponse":
        return cls(
            **_identity_wire(request),
            status="ok",
            action_chunk=action_chunk,
            action_horizon=action_horizon if action_chunk is not None else None,
            action_chunk_sha256=sha256(action_chunk).hexdigest() if action_chunk is not None else None,
        )

    @classmethod
    def error(cls, request: PolicyRequest, *, error_code: str) -> "PolicyResponse":
        return cls(**_identity_wire(request), status="error", error_code=error_code)

    def to_wire(self) -> dict[str, Any]:
        return {
            "message_type": "response",
            **_identity_wire(self),
            "status": self.status,
            "action_chunk": self.action_chunk,
            "action_horizon": self.action_horizon,
            "action_chunk_sha256": self.action_chunk_sha256,
            "error_code": self.error_code,
        }

    def canonical_metadata_bytes(self) -> bytes:
        return _canonical_bytes(self.to_wire())


def _identity_wire(envelope: PolicyRequest | PolicyResponse) -> dict[str, Any]:
    return {
        "schema_version": envelope.schema_version,
        "session_id": envelope.session_id,
        "episode_generation": envelope.episode_generation,
        "request_id": envelope.request_id,
        "policy_sha256": envelope.policy_sha256,
        "deadline_ns": envelope.deadline_ns,
    }


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _canonical_value(value), allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def envelope_from_wire(value: object) -> PolicyRequest | PolicyResponse:
    if not isinstance(value, Mapping):
        raise ValueError("policy envelope must be a mapping")
    missing = set(REQUIRED_ENVELOPE_FIELDS) - set(value)
    if missing:
        raise ValueError(f"policy envelope is missing {sorted(missing)[0]}")
    message_type = value.get("message_type")
    common = {key: value[key] for key in REQUIRED_ENVELOPE_FIELDS}
    if message_type == "request":
        operation = value.get("operation")
        expected = set(REQUIRED_ENVELOPE_FIELDS) | {"message_type", "operation"}
        if operation == "infer":
            expected.add("observation")
        elif operation == "cancel":
            expected.add("cancelled_request_id")
        if set(value) != expected:
            raise ValueError("request envelope has unknown or missing fields")
        return PolicyRequest(
            operation=operation,
            observation=value.get("observation"),
            cancelled_request_id=value.get("cancelled_request_id"),
            **common,
        )
    if message_type == "response":
        expected = set(REQUIRED_ENVELOPE_FIELDS) | {
            "message_type", "status", "action_chunk", "action_horizon", "action_chunk_sha256", "error_code"
        }
        if set(value) != expected:
            raise ValueError("response envelope has unknown or missing fields")
        return PolicyResponse(
            status=value.get("status"),
            action_chunk=value.get("action_chunk"),
            action_horizon=value.get("action_horizon"),
            action_chunk_sha256=value.get("action_chunk_sha256"),
            error_code=value.get("error_code"),
            **common,
        )
    raise ValueError("policy envelope message_type must be request or response")


def pack_envelope(envelope: PolicyRequest | PolicyResponse) -> bytes:
    """Serialize a checked protocol envelope with the pinned msgpack version."""

    return msgpack.packb(envelope.to_wire(), default=_encode_msgpack_value, use_bin_type=True)


def unpack_envelope(payload: bytes) -> PolicyRequest | PolicyResponse:
    try:
        value = msgpack.unpackb(
            payload, raw=False, strict_map_key=False, object_hook=_decode_msgpack_value
        )
    except ValueError as error:
        if str(error).startswith("policy envelope ndarray payload"):
            raise
        raise ValueError("policy envelope is not valid msgpack") from error
    except (TypeError, msgpack.ExtraData) as error:
        raise ValueError("policy envelope is not valid msgpack") from error
    return envelope_from_wire(value)


def validate_response_for_request(
    response: PolicyResponse, request: PolicyRequest, *, now_ns: int
) -> PolicyResponse:
    """Fail closed before a late or cross-episode action reaches a worker."""

    if now_ns >= request.deadline_ns:
        raise ExpiredRequestError("policy response arrived after request deadline")
    request_identity = _identity_wire(request)
    response_identity = _identity_wire(response)
    if request_identity != response_identity:
        raise StaleResponseError("policy response does not match its request identity")
    if response.status == "error":
        raise SessionStateError(f"policy gateway rejected request: {response.error_code}")
    return response


class SessionRequestGuard:
    """Gateway-side pure session admission state, safe to rebuild after restart."""

    def __init__(self, *, policy_sha256: str) -> None:
        self._policy_sha256 = _require_sha256(policy_sha256)
        self._generations: dict[str, int] = {}
        self._seen_request_ids: dict[tuple[str, int], set[str]] = {}
        self._cancelled: set[tuple[str, int, str]] = set()

    def accept(self, request: PolicyRequest, *, now_ns: int) -> None:
        if now_ns >= request.deadline_ns:
            raise ExpiredRequestError("policy request deadline has elapsed")
        if request.policy_sha256 != self._policy_sha256:
            raise PolicyDigestError("policy request targets a different policy digest")
        key = (request.session_id, request.episode_generation)
        current_generation = self._generations.get(request.session_id)
        if request.operation == "reset":
            if current_generation is not None and request.episode_generation < current_generation:
                raise SessionStateError("reset must advance episode_generation")
            seen = self._seen_request_ids.get(key)
            if seen is not None and request.request_id in seen:
                # An identical reset identity can be retried after the
                # gateway accepted it and the DEALER reply was lost.
                if current_generation == request.episode_generation:
                    return
                raise DuplicateRequestError("policy request_id was already used for this session generation")
            if current_generation == request.episode_generation:
                # A reset can be accepted by the gateway just before its reply
                # is lost.  A new-ID retry for that exact generation is an
                # idempotent reattachment, not a second episode reset.
                if seen is None:
                    raise SessionStateError("active session generation has no reset state")
                seen.add(request.request_id)
                return
            self._generations[request.session_id] = request.episode_generation
            for prior_key in tuple(self._seen_request_ids):
                if prior_key[0] == request.session_id and prior_key != key:
                    del self._seen_request_ids[prior_key]
            self._seen_request_ids[key] = {request.request_id}
            self._cancelled = {
                identity for identity in self._cancelled if identity[0] != request.session_id
            }
            return
        if current_generation != request.episode_generation:
            raise SessionStateError("request episode_generation is not the active generation")
        seen = self._seen_request_ids.get(key)
        if seen is None:
            raise SessionStateError("active session generation has no reset state")
        if request.request_id in seen:
            raise DuplicateRequestError("policy request_id was already used for this session generation")
        seen.add(request.request_id)
        if request.operation == "cancel":
            self._cancelled.add((request.session_id, request.episode_generation, request.cancelled_request_id or ""))

    def is_cancelled(self, request: PolicyRequest) -> bool:
        return (request.session_id, request.episode_generation, request.request_id) in self._cancelled

    def is_request_live(self, request: PolicyRequest, *, now_ns: int) -> bool:
        """Whether a previously admitted inference can still be routed safely."""

        key = (request.session_id, request.episode_generation)
        return (
            request.operation == "infer"
            and request.policy_sha256 == self._policy_sha256
            and now_ns < request.deadline_ns
            and self._generations.get(request.session_id) == request.episode_generation
            and request.request_id in self._seen_request_ids.get(key, set())
            and not self.is_cancelled(request)
        )


__all__ = [
    "ACTION_HORIZON",
    "SCHEMA_VERSION",
    "REQUIRED_ENVELOPE_FIELDS",
    "PolicyRequest",
    "PolicyResponse",
    "PolicyProtocolError",
    "DuplicateRequestError",
    "ExpiredRequestError",
    "StaleResponseError",
    "SessionStateError",
    "PolicyDigestError",
    "SessionRequestGuard",
    "envelope_from_wire",
    "pack_envelope",
    "unpack_envelope",
    "validate_response_for_request",
]
