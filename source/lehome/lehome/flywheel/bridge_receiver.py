"""Fail-closed loopback receiver for authenticated Mac leader samples."""

from __future__ import annotations

from dataclasses import dataclass
import math
import socket
import struct
import threading
import time
from typing import Callable, Iterable, Protocol


LOOPBACK_HOST = "127.0.0.1"
_MAX_FRAME_BYTES = 64 * 1024


def _finite_12(values: Iterable[object], *, field: str) -> tuple[float, ...]:
    result = tuple(values)
    if len(result) != 12 or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in result):
        raise ValueError(f"{field} must contain 12 finite values")
    return tuple(float(value) for value in result)


def _motor_limits(values: Iterable[Iterable[object]], *, field: str) -> tuple[tuple[float, float], ...]:
    pairs = tuple(tuple(pair) for pair in values)
    if len(pairs) != 6:
        raise ValueError(f"{field} must contain six ordered (min,max) pairs")
    result: list[tuple[float, float]] = []
    for pair in pairs:
        if len(pair) != 2 or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in pair):
            raise ValueError(f"{field} must contain finite (min,max) pairs")
        lower, upper = float(pair[0]), float(pair[1])
        if lower >= upper:
            raise ValueError(f"{field} requires min < max")
        result.append((lower, upper))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class Handshake:
    session_nonce: str
    sequence: int
    left_serial: str
    right_serial: str
    left_calibration_sha256: str
    right_calibration_sha256: str
    left_motor_limits: tuple[tuple[float, float], ...]
    right_motor_limits: tuple[tuple[float, float], ...]
    hz: int

    def __post_init__(self) -> None:
        if not self.session_nonce or self.sequence != 0:
            raise ValueError("bridge handshake must have a nonce and sequence zero")
        if not self.left_serial or not self.right_serial or self.left_serial == self.right_serial:
            raise ValueError("bridge handshake requires distinct serial identities")
        if not isinstance(self.hz, int) or self.hz <= 0:
            raise ValueError("bridge handshake sampling rate is invalid")
        for value in (self.left_calibration_sha256, self.right_calibration_sha256):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError("bridge handshake calibration hash is invalid")
        object.__setattr__(self, "left_motor_limits", _motor_limits(self.left_motor_limits, field="left motor limits"))
        object.__setattr__(self, "right_motor_limits", _motor_limits(self.right_motor_limits, field="right motor limits"))


@dataclass(frozen=True, slots=True)
class LeaderSampleFrame:
    session_nonce: str
    sequence: int
    sent_monotonic_ns: int
    positions: tuple[float, ...]
    raw_positions: tuple[float, ...]
    rtt_ns: int
    rtt_age_ns: int

    def __post_init__(self) -> None:
        if not self.session_nonce or not isinstance(self.sequence, int) or self.sequence <= 0:
            raise ValueError("bridge sample sequence is invalid")
        if not isinstance(self.sent_monotonic_ns, int) or self.sent_monotonic_ns < 0:
            raise ValueError("bridge sample timestamp is invalid")
        if not isinstance(self.rtt_ns, int) or self.rtt_ns <= 0:
            raise ValueError("bridge sample RTT is invalid")
        if not isinstance(self.rtt_age_ns, int) or self.rtt_age_ns < 0:
            raise ValueError("bridge sample RTT age is invalid")
        object.__setattr__(self, "positions", _finite_12(self.positions, field="bridge normalized positions"))
        object.__setattr__(self, "raw_positions", _finite_12(self.raw_positions, field="bridge raw positions"))


@dataclass(frozen=True, slots=True)
class RecordedExpertSample:
    sequence: int
    sent_monotonic_ns: int
    received_monotonic_ns: int
    raw_positions: tuple[float, ...]
    normalized_positions: tuple[float, ...]
    converted_command: tuple[float, ...]
    jitter_ms: float


@dataclass(frozen=True, slots=True)
class ExpertCommand:
    command: tuple[float, ...]
    eligible: bool
    reason: str | None
    sequence: int | None = None
    sample_age_ms: float | None = None


def _canonical_so101_converter(positions: tuple[float, ...]) -> tuple[float, ...]:
    """Use the existing follower conversion boundary only on the Isaac host."""
    try:  # Imported only in an actual remote collection session, never test import.
        import numpy as np
        from lehome.utils.robot_utils import convert_lerobot_action_to_leisaac
    except ImportError as error:  # pragma: no cover - host integration failure
        raise RuntimeError("canonical SO101 action conversion is unavailable") from error
    left = convert_lerobot_action_to_leisaac(np.asarray([positions[:6]], dtype=np.float32))[0, :6]
    right = convert_lerobot_action_to_leisaac(np.asarray([positions[6:]], dtype=np.float32))[0, :6]
    return _finite_12((*left, *right), field="converted expert command")


class BridgeReceiver:
    """Owns health state, raw/converted recording, holds, and explicit resync."""

    def __init__(
        self,
        *,
        max_age_ms: float = 80.0,
        max_jitter_ms: float = 30.0,
        max_rtt_ms: float = 80.0,
        max_rtt_age_ms: float = 1_000.0,
        converter: Callable[[tuple[float, ...]], Iterable[object]] | None = None,
        expected_calibrations: tuple[str, str] | None = None,
        expected_motor_limits: tuple[Iterable[Iterable[object]], Iterable[Iterable[object]]] | None = None,
    ) -> None:
        if not math.isfinite(max_age_ms) or max_age_ms <= 0:
            raise ValueError("max_age_ms must be positive and finite")
        if not math.isfinite(max_jitter_ms) or max_jitter_ms < 0:
            raise ValueError("max_jitter_ms must be non-negative and finite")
        if not math.isfinite(max_rtt_ms) or max_rtt_ms <= 0:
            raise ValueError("max_rtt_ms must be positive and finite")
        if not math.isfinite(max_rtt_age_ms) or max_rtt_age_ms < 0:
            raise ValueError("max_rtt_age_ms must be non-negative and finite")
        self.max_age_ms = max_age_ms
        self.max_jitter_ms = max_jitter_ms
        self.max_rtt_ms = max_rtt_ms
        self.max_rtt_age_ms = max_rtt_age_ms
        self.converter = converter or _canonical_so101_converter
        self.expected_calibrations = expected_calibrations
        self.expected_motor_limits = (
            None
            if expected_motor_limits is None
            else (
                _motor_limits(expected_motor_limits[0], field="expected left motor limits"),
                _motor_limits(expected_motor_limits[1], field="expected right motor limits"),
            )
        )
        self.handshake: Handshake | None = None
        self.last_sample: RecordedExpertSample | None = None
        self.last_safe_command: tuple[float, ...] = (0.0,) * 12
        self.records: list[RecordedExpertSample] = []
        self.requires_resync = False
        self.disconnected = False
        self.jitter_ms = 0.0
        self.arrival_jitter_ms = 0.0
        self.sender_sync_jitter_ms = 0.0
        self.rtt_ms: float | None = None
        self.rtt_age_ms: float | None = None
        self._rtt_healthy = False

    def accept_handshake(self, handshake: Handshake) -> None:
        if self.handshake is not None:
            raise ValueError("bridge handshake was already accepted")
        if self.expected_calibrations is not None and (
            handshake.left_calibration_sha256,
            handshake.right_calibration_sha256,
        ) != self.expected_calibrations:
            raise ValueError("bridge calibration is incompatible with this collection session")
        if self.expected_motor_limits is not None and (
            handshake.left_motor_limits,
            handshake.right_motor_limits,
        ) != self.expected_motor_limits:
            raise ValueError("bridge motor limits are incompatible with this collection session")
        self.handshake = handshake
        self.disconnected = False

    def accept_sample(
        self, sample: LeaderSampleFrame, *, received_monotonic_ns: int | None = None
    ) -> RecordedExpertSample:
        if self.handshake is None:
            raise ValueError("bridge handshake is required before samples")
        if sample.session_nonce != self.handshake.session_nonce:
            raise ValueError("bridge session nonce mismatch")
        expected_sequence = 1 if self.last_sample is None else self.last_sample.sequence + 1
        if sample.sequence != expected_sequence:
            self.requires_resync = True
            raise ValueError("bridge sequence is stale, duplicate, or reordered")
        received = time.monotonic_ns() if received_monotonic_ns is None else received_monotonic_ns
        if not isinstance(received, int) or received < 0:
            raise ValueError("received monotonic timestamp is invalid")
        raw_limits = self.handshake.left_motor_limits + self.handshake.right_motor_limits
        if any(value < lower or value > upper for value, (lower, upper) in zip(sample.raw_positions, raw_limits)):
            self.requires_resync = True
            raise ValueError("raw leader position is outside the handshake-advertised motor limits")
        self.rtt_ms = sample.rtt_ns / 1_000_000
        self.rtt_age_ms = sample.rtt_age_ns / 1_000_000
        self._rtt_healthy = self.rtt_ms <= self.max_rtt_ms and self.rtt_age_ms <= self.max_rtt_age_ms
        if not self._rtt_healthy:
            self.requires_resync = True
        if self.last_sample is not None:
            expected_period_ns = int(1_000_000_000 / self.handshake.hz)
            received_period_ns = received - self.last_sample.received_monotonic_ns
            sender_period_ns = sample.sent_monotonic_ns - self.last_sample.sent_monotonic_ns
            # The two hosts' monotonic clock origins are unrelated.  Only
            # durations measured on each individual host are comparable.
            self.arrival_jitter_ms = abs(received_period_ns - expected_period_ns) / 1_000_000
            self.sender_sync_jitter_ms = abs(received_period_ns - sender_period_ns) / 1_000_000
            self.jitter_ms = max(self.arrival_jitter_ms, self.sender_sync_jitter_ms)
            if sender_period_ns <= 0:
                self.jitter_ms = self.max_jitter_ms + 1.0
            if self.jitter_ms > self.max_jitter_ms:
                self.requires_resync = True
        converted = _finite_12(self.converter(sample.positions), field="converted expert command")
        record = RecordedExpertSample(
            sequence=sample.sequence,
            sent_monotonic_ns=sample.sent_monotonic_ns,
            received_monotonic_ns=received,
            raw_positions=sample.raw_positions,
            normalized_positions=sample.positions,
            converted_command=converted,
            jitter_ms=self.jitter_ms,
        )
        self.last_sample = record
        self.records.append(record)
        if not self.requires_resync and not self.disconnected:
            self.last_safe_command = converted
        return record

    def close_connection(self) -> None:
        self.disconnected = True
        self.requires_resync = True

    def resync(self, *, now_ns: int | None = None) -> None:
        """Explicit operator action after a fresh healthy sample has arrived."""
        command = self.current(now_ns=now_ns)
        if command.reason == "rtt_exceeded":
            raise ValueError("cannot resync until a fresh RTT measurement is healthy")
        if command.reason != "resync_required":
            raise ValueError("cannot resync unless current bridge state is resync_required")
        self.requires_resync = False
        self.jitter_ms = 0.0
        self.arrival_jitter_ms = 0.0
        self.sender_sync_jitter_ms = 0.0
        self.last_safe_command = self.last_sample.converted_command

    def current(self, *, now_ns: int | None = None) -> ExpertCommand:
        now = time.monotonic_ns() if now_ns is None else now_ns
        if self.disconnected:
            return ExpertCommand(self.last_safe_command, False, "disconnected")
        if self.last_sample is None:
            return ExpertCommand(self.last_safe_command, False, "no_sample")
        age_ms = (now - self.last_sample.received_monotonic_ns) / 1_000_000
        if age_ms > self.max_age_ms:
            self.requires_resync = True
            return ExpertCommand(self.last_safe_command, False, "stale_sample", self.last_sample.sequence, age_ms)
        if not self._rtt_healthy:
            return ExpertCommand(self.last_safe_command, False, "rtt_exceeded", self.last_sample.sequence, age_ms)
        if self.jitter_ms > self.max_jitter_ms:
            self.requires_resync = True
            return ExpertCommand(self.last_safe_command, False, "jitter_exceeded", self.last_sample.sequence, age_ms)
        if self.requires_resync:
            return ExpertCommand(self.last_safe_command, False, "resync_required", self.last_sample.sequence, age_ms)
        return ExpertCommand(self.last_sample.converted_command, True, None, self.last_sample.sequence, age_ms)


class _Verifier(Protocol):
    def verify(self, wire: bytes): ...


class LoopbackBridgeServer:
    """One-client TCP wrapper. SSH exposes it; it never binds a public address."""

    def __init__(
        self,
        *,
        secret: bytes,
        session_nonce: str,
        host: str = LOOPBACK_HOST,
        port: int = 18080,
        receiver: BridgeReceiver | None = None,
    ) -> None:
        if host != LOOPBACK_HOST:
            raise ValueError("bridge receiver must bind exactly to loopback")
        if not session_nonce or len(secret) < 32:
            raise ValueError("bridge server requires a session nonce and 32-byte secret")
        self.secret = secret
        self.session_nonce = session_nonce
        self.host = host
        self.port = port
        self.receiver = receiver or BridgeReceiver()
        self._listener: socket.socket | None = None
        self._client: socket.socket | None = None
        self._client_active = False
        self._cancel = threading.Event()
        self._state_lock = threading.Lock()
        self.failure: str | None = None

    def start(self) -> None:
        with self._state_lock:
            if self._listener is not None:
                raise RuntimeError("bridge server is already listening")
            if self._cancel.is_set():
                raise RuntimeError("bridge server was already cancelled")
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # A Mac may connect only after its operator has completed the SSH
        # tunnel.  Poll solely for explicit cancellation; there is no startup
        # timeout that can silently kill the only accept thread.
        listener.settimeout(0.1)
        listener.bind((LOOPBACK_HOST, self.port))
        listener.listen(1)
        with self._state_lock:
            if self._cancel.is_set():
                listener.close()
                raise RuntimeError("bridge server was cancelled while starting")
            self._listener = listener

    def close(self) -> None:
        self._cancel.set()
        with self._state_lock:
            listener, client = self._listener, self._client
            self._listener = None
            self._client = None
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        if client is not None:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except (AttributeError, OSError):
                pass
            try:
                client.close()
            except OSError:
                pass
        self.receiver.close_connection()

    @staticmethod
    def _read_exact(client: socket.socket, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = client.recv(remaining)
            if not chunk:
                raise ConnectionError("bridge client disconnected")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _verifier(self) -> _Verifier:
        try:
            from lehome_bridge.protocol import MessageVerifier
        except ImportError as error:  # pragma: no cover - deployment packaging guard
            raise RuntimeError("install lehome-bridge on the receiver host") from error
        return MessageVerifier(secret=self.secret, expected_nonce=self.session_nonce)

    def serve_one_client(self) -> None:
        with self._state_lock:
            listener = self._listener
            if listener is None:
                if self._cancel.is_set():
                    return
                raise RuntimeError("bridge server has not been started")
            if self._client_active:
                raise RuntimeError("bridge server accepts exactly one client")
            self._client_active = True
        while not self._cancel.is_set():
            try:
                client, _ = listener.accept()
                break
            except socket.timeout:
                continue
            except OSError:
                if self._cancel.is_set():
                    return
                self.failure = "listener_accept_failed"
                self.receiver.close_connection()
                raise
        else:
            return
        with self._state_lock:
            if self._cancel.is_set():
                cancelled = True
            else:
                self._client = client
                cancelled = False
        if cancelled:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except (AttributeError, OSError):
                pass
            try:
                client.close()
            except OSError:
                pass
            return
        try:
            verifier = self._verifier()
            client.settimeout(1.0)
            while not self._cancel.is_set():
                size = struct.unpack("!I", self._read_exact(client, 4))[0]
                if size > _MAX_FRAME_BYTES:
                    raise ValueError("bridge frame exceeds the 64 KiB maximum")
                message = verifier.verify(struct.pack("!I", size) + self._read_exact(client, size))
                if message.kind == "handshake":
                    self.receiver.accept_handshake(
                        Handshake(
                            message.session_nonce,
                            message.sequence,
                            message.left_serial,
                            message.right_serial,
                            message.left_calibration_sha256,
                            message.right_calibration_sha256,
                            message.left_motor_limits,
                            message.right_motor_limits,
                            message.hz,
                        )
                    )
                elif message.kind == "ping":
                    try:
                        from lehome_bridge.protocol import BridgeMessage, encode_message
                    except ImportError as error:  # pragma: no cover - deployment packaging guard
                        raise RuntimeError("install lehome-bridge on the receiver host") from error
                    client.sendall(
                        encode_message(
                            BridgeMessage.ack(message.session_nonce, message.sequence, message.probe_nonce),
                            secret=self.secret,
                        )
                    )
                else:
                    self.receiver.accept_sample(
                        LeaderSampleFrame(
                            message.session_nonce,
                            message.sample_sequence,
                            message.monotonic_ns,
                            message.positions,
                            message.raw_positions,
                            message.rtt_ns,
                            message.rtt_age_ns,
                        )
                    )
        except (ConnectionError, OSError, ValueError):
            if self._cancel.is_set():
                return
            self.failure = "bridge_transport_failed"
            self.receiver.close_connection()
            raise
        finally:
            try:
                client.close()
            except OSError:
                pass
            with self._state_lock:
                if self._client is client:
                    self._client = None
