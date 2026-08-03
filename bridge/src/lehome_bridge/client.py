"""Fixed-rate authenticated client for an already-forwarded loopback socket."""

from __future__ import annotations

from pathlib import Path
import socket
import stat
import time
from typing import Protocol

from .leaders import DualLeaderReader, LeaderSample
from .protocol import BridgeMessage, encode_message


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18080


def read_secret_file(path: Path) -> bytes:
    """Read a secret only from an owner-private regular file; never log its path."""
    secret_path = Path(path)
    status = secret_path.stat()
    if not stat.S_ISREG(status.st_mode) or stat.S_IMODE(status.st_mode) != 0o600:
        raise ValueError("bridge secret file must be a regular mode-0600 file")
    secret = secret_path.read_bytes()
    if len(secret) < 32:
        raise ValueError("bridge secret must contain at least 32 bytes")
    return secret


class _Socket(Protocol):
    def sendall(self, data: bytes) -> None: ...
    def close(self) -> None: ...


class BridgeConnection:
    """One authenticated session. It only knows loopback by default."""

    def __init__(
        self,
        reader: DualLeaderReader,
        *,
        secret: bytes,
        session_nonce: str,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        hz: int = 30,
        sock: _Socket | None = None,
    ) -> None:
        if not session_nonce:
            raise ValueError("bridge session nonce is required")
        if not isinstance(hz, int) or hz <= 0:
            raise ValueError("bridge hz must be positive")
        self.reader = reader
        self.secret = secret
        self.session_nonce = session_nonce
        self.host = host
        self.port = port
        self.hz = hz
        self.sock = sock
        self.sequence = 0
        self.stop_requested = False

    def connect(self) -> None:
        if self.sock is None:
            self.sock = socket.create_connection((self.host, self.port), timeout=3.0)
        self._send_handshake()

    def _send(self, message: BridgeMessage) -> None:
        if self.sock is None:
            raise RuntimeError("bridge connection is not open")
        self.sock.sendall(encode_message(message, secret=self.secret))

    def _send_handshake(self) -> None:
        self._send(
            BridgeMessage.handshake(
                session_nonce=self.session_nonce,
                sequence=0,
                left_serial=self.reader.left_bus.serial_identity,
                right_serial=self.reader.right_bus.serial_identity,
                left_calibration_sha256=self.reader.left_calibration.sha256,
                right_calibration_sha256=self.reader.right_calibration.sha256,
                left_motor_limits=self.reader.left_motor_limits,
                right_motor_limits=self.reader.right_motor_limits,
                hz=self.hz,
            )
        )
        self.sequence = 1

    def send_sample(self, sample: LeaderSample) -> None:
        if self.sequence == 0:
            raise RuntimeError("bridge handshake must be sent before samples")
        self._send(BridgeMessage.sample(self.session_nonce, self.sequence, sample.monotonic_ns, sample.positions))
        self.sequence += 1

    def request_stop(self) -> None:
        self.stop_requested = True

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None


def sleep_until_monotonic_ns(deadline_ns: int) -> None:
    remaining = deadline_ns - time.monotonic_ns()
    if remaining > 0:
        time.sleep(remaining / 1_000_000_000)


def stream(reader: DualLeaderReader, connection: BridgeConnection, *, hz: int = 30) -> None:
    """Read and send at a monotonic-deadline rate without serial side effects here."""
    if not isinstance(hz, int) or hz <= 0:
        raise ValueError("bridge hz must be positive")
    period_ns = int(1_000_000_000 / hz)
    deadline = time.monotonic_ns()
    while not connection.stop_requested:
        connection.send_sample(reader.read())
        deadline += period_ns
        sleep_until_monotonic_ns(deadline)
