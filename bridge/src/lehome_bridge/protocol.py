"""Versioned, authenticated wire messages for the physical leader bridge."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import struct
from typing import Any, Iterable, Mapping


PROTOCOL_VERSION = 3
MAX_FRAME_BYTES = 64 * 1024
SIGNATURE_BYTES = hashlib.sha256().digest_size
_SHA256_HEX_LENGTH = 64


def _finite_12(values: Iterable[object]) -> tuple[float, ...]:
    sample = tuple(values)
    if len(sample) != 12 or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in sample):
        raise ValueError("positions must be finite 12D values")
    return tuple(float(value) for value in sample)


def _normalized_12(values: Iterable[object]) -> tuple[float, ...]:
    sample = _finite_12(values)
    for index, value in enumerate(sample):
        lower = 0.0 if index % 6 == 5 else -100.0
        if value < lower or value > 100.0:
            raise ValueError("normalized positions must be within SO101 normalization limits")
    return sample


def _sha256(value: str, name: str) -> str:
    if len(value) != _SHA256_HEX_LENGTH or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _motor_limits(values: Iterable[Iterable[object]], name: str) -> tuple[tuple[float, float], ...]:
    pairs = tuple(tuple(pair) for pair in values)
    if len(pairs) != 6:
        raise ValueError(f"{name} motor limits must contain six ordered (min,max) pairs")
    limits: list[tuple[float, float]] = []
    for pair in pairs:
        if len(pair) != 2 or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in pair):
            raise ValueError(f"{name} motor limits must contain finite (min,max) pairs")
        lower, upper = float(pair[0]), float(pair[1])
        if lower >= upper:
            raise ValueError(f"{name} motor limits require min < max")
        limits.append((lower, upper))
    return tuple(limits)


@dataclass(frozen=True, slots=True)
class BridgeMessage:
    """Authenticated session messages; the secret is never a field."""

    kind: str
    session_nonce: str
    sequence: int
    monotonic_ns: int | None = None
    positions: tuple[float, ...] | None = None
    raw_positions: tuple[float, ...] | None = None
    rtt_ns: int | None = None
    rtt_age_ns: int | None = None
    sample_sequence: int | None = None
    probe_nonce: str | None = None
    left_serial: str | None = None
    right_serial: str | None = None
    left_calibration_sha256: str | None = None
    right_calibration_sha256: str | None = None
    left_motor_limits: tuple[tuple[float, float], ...] | None = None
    right_motor_limits: tuple[tuple[float, float], ...] | None = None
    hz: int | None = None
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("unsupported bridge protocol version")
        if self.kind not in {"handshake", "sample", "ping", "ack"}:
            raise ValueError("unsupported bridge message kind")
        if not isinstance(self.session_nonce, str) or not self.session_nonce:
            raise ValueError("bridge session nonce is required")
        if not isinstance(self.sequence, int) or self.sequence < 0:
            raise ValueError("bridge sequence must be a non-negative integer")
        if self.kind == "handshake":
            if self.sequence != 0:
                raise ValueError("handshake sequence must be zero")
            values = (self.left_serial, self.right_serial)
            if not all(isinstance(value, str) and value for value in values):
                raise ValueError("handshake requires non-empty serial identities")
            if self.left_serial == self.right_serial:
                raise ValueError("leader serial identities must be distinct")
            _sha256(self.left_calibration_sha256 or "", "left calibration hash")
            _sha256(self.right_calibration_sha256 or "", "right calibration hash")
            if self.left_motor_limits is None or self.right_motor_limits is None:
                raise ValueError("handshake motor limits are required")
            object.__setattr__(self, "left_motor_limits", _motor_limits(self.left_motor_limits, "left"))
            object.__setattr__(self, "right_motor_limits", _motor_limits(self.right_motor_limits, "right"))
            if not isinstance(self.hz, int) or self.hz <= 0:
                raise ValueError("handshake hz must be a positive integer")
            if any(value is not None for value in (self.positions, self.raw_positions, self.monotonic_ns, self.rtt_ns, self.rtt_age_ns, self.sample_sequence, self.probe_nonce)):
                raise ValueError("handshake must not include a sample")
        elif self.kind == "sample":
            if self.sequence == 0:
                raise ValueError("sample sequence must follow the handshake")
            if not isinstance(self.monotonic_ns, int) or self.monotonic_ns < 0:
                raise ValueError("sample monotonic_ns must be a non-negative integer")
            if self.positions is None or self.raw_positions is None:
                raise ValueError("sample positions are required")
            if not isinstance(self.rtt_ns, int) or self.rtt_ns <= 0:
                raise ValueError("sample requires a positive client-measured rtt_ns")
            if not isinstance(self.rtt_age_ns, int) or self.rtt_age_ns < 0:
                raise ValueError("sample requires a non-negative rtt_age_ns")
            if not isinstance(self.sample_sequence, int) or self.sample_sequence <= 0:
                raise ValueError("sample requires a positive leader sample sequence")
            object.__setattr__(self, "positions", _normalized_12(self.positions))
            object.__setattr__(self, "raw_positions", _finite_12(self.raw_positions))
            if any(value is not None for value in (self.left_serial, self.right_serial, self.left_calibration_sha256, self.right_calibration_sha256, self.left_motor_limits, self.right_motor_limits, self.hz, self.probe_nonce)):
                raise ValueError("sample must not repeat handshake identity or motor limits")
        else:
            if self.sequence == 0 or not isinstance(self.probe_nonce, str) or not self.probe_nonce:
                raise ValueError("probe messages require a positive sequence and nonce")
            if any(value is not None for value in (self.monotonic_ns, self.positions, self.raw_positions, self.rtt_ns, self.rtt_age_ns, self.sample_sequence, self.left_serial, self.right_serial, self.left_calibration_sha256, self.right_calibration_sha256, self.left_motor_limits, self.right_motor_limits, self.hz)):
                raise ValueError("probe messages must contain only their nonce")

    @classmethod
    def handshake(
        cls,
        *,
        session_nonce: str,
        sequence: int,
        left_serial: str,
        right_serial: str,
        left_calibration_sha256: str,
        right_calibration_sha256: str,
        left_motor_limits: Iterable[Iterable[object]],
        right_motor_limits: Iterable[Iterable[object]],
        hz: int,
    ) -> "BridgeMessage":
        return cls(
            kind="handshake",
            session_nonce=session_nonce,
            sequence=sequence,
            left_serial=left_serial,
            right_serial=right_serial,
            left_calibration_sha256=left_calibration_sha256,
            right_calibration_sha256=right_calibration_sha256,
            left_motor_limits=_motor_limits(left_motor_limits, "left"),
            right_motor_limits=_motor_limits(right_motor_limits, "right"),
            hz=hz,
        )

    @classmethod
    def sample(
        cls, session_nonce: str, sequence: int, monotonic_ns: int, positions: Iterable[object], *, raw_positions: Iterable[object], sample_sequence: int, rtt_ns: int, rtt_age_ns: int
    ) -> "BridgeMessage":
        return cls(
            kind="sample",
            session_nonce=session_nonce,
            sequence=sequence,
            monotonic_ns=monotonic_ns,
            positions=_normalized_12(positions),
            raw_positions=_finite_12(raw_positions),
            sample_sequence=sample_sequence,
            rtt_ns=rtt_ns,
            rtt_age_ns=rtt_age_ns,
        )

    @classmethod
    def ping(cls, session_nonce: str, sequence: int, probe_nonce: str) -> "BridgeMessage":
        return cls(kind="ping", session_nonce=session_nonce, sequence=sequence, probe_nonce=probe_nonce)

    @classmethod
    def ack(cls, session_nonce: str, sequence: int, probe_nonce: str) -> "BridgeMessage":
        return cls(kind="ack", session_nonce=session_nonce, sequence=sequence, probe_nonce=probe_nonce)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "kind": self.kind,
            "protocol_version": self.protocol_version,
            "sequence": self.sequence,
            "session_nonce": self.session_nonce,
        }
        if self.kind == "handshake":
            result.update(
                hz=self.hz,
                left_calibration_sha256=self.left_calibration_sha256,
                left_serial=self.left_serial,
                left_motor_limits=[list(pair) for pair in self.left_motor_limits or ()],
                right_calibration_sha256=self.right_calibration_sha256,
                right_serial=self.right_serial,
                right_motor_limits=[list(pair) for pair in self.right_motor_limits or ()],
            )
        elif self.kind == "sample":
            result.update(monotonic_ns=self.monotonic_ns, positions=list(self.positions or ()), raw_positions=list(self.raw_positions or ()), sample_sequence=self.sample_sequence, rtt_ns=self.rtt_ns, rtt_age_ns=self.rtt_age_ns)
        else:
            result.update(probe_nonce=self.probe_nonce)
        return result

    @classmethod
    def from_json(cls, payload: bytes) -> "BridgeMessage":
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("bridge message payload is not valid canonical JSON") from error
        if not isinstance(decoded, Mapping):
            raise ValueError("bridge message payload must be an object")
        required = {"kind", "protocol_version", "sequence", "session_nonce"}
        if not required.issubset(decoded):
            raise ValueError("bridge message is missing required fields")
        if decoded["protocol_version"] != PROTOCOL_VERSION:
            raise ValueError("unsupported bridge protocol version")
        if decoded["kind"] == "handshake":
            expected = required | {"left_serial", "right_serial", "left_calibration_sha256", "right_calibration_sha256", "left_motor_limits", "right_motor_limits", "hz"}
            if set(decoded) != expected:
                raise ValueError("invalid handshake fields")
            return cls(**decoded)
        if decoded["kind"] == "sample":
            expected = required | {"monotonic_ns", "positions", "raw_positions", "sample_sequence", "rtt_ns", "rtt_age_ns"}
            if set(decoded) != expected:
                raise ValueError("invalid sample fields")
            return cls.sample(decoded["session_nonce"], decoded["sequence"], decoded["monotonic_ns"], decoded["positions"], raw_positions=decoded["raw_positions"], sample_sequence=decoded["sample_sequence"], rtt_ns=decoded["rtt_ns"], rtt_age_ns=decoded["rtt_age_ns"])
        if decoded["kind"] in {"ping", "ack"}:
            expected = required | {"probe_nonce"}
            if set(decoded) != expected:
                raise ValueError("invalid probe fields")
            return cls(kind=decoded["kind"], session_nonce=decoded["session_nonce"], sequence=decoded["sequence"], probe_nonce=decoded["probe_nonce"])
        raise ValueError("unsupported bridge message kind")


def _canonical_payload(message: BridgeMessage) -> bytes:
    return json.dumps(message.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _check_secret(secret: bytes) -> None:
    if not isinstance(secret, bytes) or len(secret) < 32:
        raise ValueError("bridge secret must be at least 32 bytes")


def encode_message(message: BridgeMessage, *, secret: bytes) -> bytes:
    """Return a bounded length-prefixed HMAC-SHA256 frame."""
    _check_secret(secret)
    payload = _canonical_payload(message)
    signature = hmac.digest(secret, payload, "sha256")
    body = signature + payload
    if len(body) > MAX_FRAME_BYTES:
        raise ValueError("bridge frame exceeds the 64 KiB maximum")
    return struct.pack("!I", len(body)) + body


def split_frame(wire: bytes) -> tuple[bytes, bytes]:
    if len(wire) < 4:
        raise ValueError("bridge frame is truncated")
    size = struct.unpack("!I", wire[:4])[0]
    if size > MAX_FRAME_BYTES:
        raise ValueError("bridge frame exceeds the 64 KiB maximum")
    if size < SIGNATURE_BYTES or len(wire) != size + 4:
        raise ValueError("bridge frame length is invalid")
    return wire[4 : 4 + SIGNATURE_BYTES], wire[4 + SIGNATURE_BYTES :]


def decode_message(wire: bytes, *, secret: bytes) -> BridgeMessage:
    """Authenticate one frame without applying a direction-specific sequence rule."""
    _check_secret(secret)
    signature, payload = split_frame(wire)
    expected = hmac.digest(secret, payload, "sha256")
    if not hmac.compare_digest(signature, expected):
        raise ValueError("bridge message authentication failed")
    return BridgeMessage.from_json(payload)


class MessageVerifier:
    """Authenticates a single bridge session and enforces strict sequencing."""

    def __init__(self, *, secret: bytes, expected_nonce: str) -> None:
        _check_secret(secret)
        if not expected_nonce:
            raise ValueError("expected bridge session nonce is required")
        self.secret = secret
        self.expected_nonce = expected_nonce
        self.next_sequence = 0

    def verify(self, wire: bytes) -> BridgeMessage:
        message = decode_message(wire, secret=self.secret)
        if message.session_nonce != self.expected_nonce:
            raise ValueError("bridge session nonce mismatch")
        if message.sequence != self.next_sequence:
            raise ValueError("bridge sequence is stale, duplicate, or reordered")
        if self.next_sequence == 0 and message.kind != "handshake":
            raise ValueError("bridge handshake is required before samples")
        if self.next_sequence > 0 and message.kind not in {"sample", "ping"}:
            raise ValueError("bridge client may send only samples or probes after the handshake")
        self.next_sequence += 1
        return message
