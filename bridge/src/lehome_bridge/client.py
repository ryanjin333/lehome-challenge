"""Fixed-rate authenticated client for an already-forwarded loopback socket."""

from __future__ import annotations

import os
from pathlib import Path
import socket
import stat
import threading
import time
from secrets import token_urlsafe
from typing import Protocol

from .leaders import DualLeaderReader, LeaderSample
from .protocol import BridgeMessage, decode_message, encode_message


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18080
# Refresh before the receiver's one-second RTT freshness budget expires, while
# tolerating an otherwise healthy near-80 ms tunnel RTT.
RTT_REFRESH_INTERVAL_NS = 750_000_000


def _require_private_secret_parent(path: Path) -> None:
    """Require the directory contract that makes name-based final unlink safe."""
    status = Path(path).parent.stat()
    if (
        not stat.S_ISDIR(status.st_mode)
        or status.st_uid != os.geteuid()
        or stat.S_IMODE(status.st_mode) & (stat.S_IWGRP | stat.S_IWOTH)
    ):
        raise ValueError("bridge secret parent directory must be owner-private")


def read_secret_file(path: Path) -> bytes:
    """Read a secret only from an owner-private regular file; never log its path."""
    secret, _ = read_secret_file_with_identity(path)
    return secret


def read_secret_file_with_identity(path: Path) -> tuple[bytes, tuple[int, int]]:
    """Read one inode without following a replacement or a symlink."""
    secret_path = Path(path)
    _require_private_secret_parent(secret_path)
    status = secret_path.lstat()
    if not stat.S_ISREG(status.st_mode) or stat.S_IMODE(status.st_mode) != 0o600:
        raise ValueError("bridge secret file must be a regular mode-0600 file")
    descriptor = os.open(secret_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        identity = (opened.st_dev, opened.st_ino)
        if identity != (status.st_dev, status.st_ino) or not stat.S_ISREG(opened.st_mode) or stat.S_IMODE(opened.st_mode) != 0o600:
            raise ValueError("bridge secret file changed while it was being opened")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        secret = b"".join(chunks)
    finally:
        os.close(descriptor)
    if len(secret) < 32:
        try:
            remove_secret_file(secret_path, identity=identity)
        except RuntimeError as cleanup_error:
            raise cleanup_error from ValueError("bridge secret must contain at least 32 bytes")
        raise ValueError("bridge secret must contain at least 32 bytes")
    return secret, identity


def remove_secret_file(path: Path, *, identity: tuple[int, int]) -> None:
    """Zero and unlink precisely the one session secret this process opened."""
    secret_path = Path(path)
    _require_private_secret_parent(secret_path)
    try:
        status = secret_path.lstat()
    except FileNotFoundError:
        return
    if (
        not stat.S_ISREG(status.st_mode)
        or stat.S_IMODE(status.st_mode) != 0o600
        or (status.st_dev, status.st_ino) != identity
    ):
        raise RuntimeError("refusing to remove a bridge secret path not opened by this session")
    descriptor = os.open(secret_path, os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != identity:
            raise RuntimeError("refusing to overwrite a replaced bridge secret")
        remaining = opened.st_size
        while remaining:
            written = os.write(descriptor, b"\0" * min(remaining, 64 * 1024))
            if written <= 0:
                raise RuntimeError("failed to overwrite the bridge session secret")
            remaining -= written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    final_status = secret_path.lstat()
    if (final_status.st_dev, final_status.st_ino) != identity:
        raise RuntimeError("refusing to remove a replaced bridge secret")
    secret_path.unlink()


class _Socket(Protocol):
    def sendall(self, data: bytes) -> None: ...
    def recv(self, size: int) -> bytes: ...
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
        self.sample_sequence = 1
        self.stop_requested = False
        self._last_rtt_ns: int | None = None
        self._rtt_measured_monotonic_ns: int | None = None
        self._send_lock = threading.Lock()
        self._rtt_lock = threading.Lock()
        self._probe_lock = threading.Lock()
        self._rtt_probe_thread: threading.Thread | None = None
        self._rtt_probe_error: BaseException | None = None

    def connect(self) -> None:
        if self.sock is None:
            self.sock = socket.create_connection((self.host, self.port), timeout=3.0)
        self._send_handshake()

    def _send_unlocked(self, message: BridgeMessage) -> None:
        if self.sock is None:
            raise RuntimeError("bridge connection is not open")
        self.sock.sendall(encode_message(message, secret=self.secret))

    def _send(self, message: BridgeMessage) -> None:
        with self._send_lock:
            self._send_unlocked(message)

    def _send_handshake(self) -> None:
        with self._send_lock:
            self._send_unlocked(
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

    def _read_exact(self, size: int) -> bytes:
        if self.sock is None:
            raise RuntimeError("bridge connection is not open")
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = self.sock.recv(remaining)
            if not chunk:
                raise ConnectionError("bridge receiver disconnected during RTT probe")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def refresh_rtt(self) -> int:
        """Measure a nonce-bound RTT entirely on the client monotonic clock."""
        if self.sequence == 0:
            raise RuntimeError("bridge handshake must be sent before RTT probes")
        probe_nonce = token_urlsafe(24)
        with self._send_lock:
            sequence = self.sequence
            started = time.monotonic_ns()
            self._send_unlocked(BridgeMessage.ping(self.session_nonce, sequence, probe_nonce))
            self.sequence += 1
        size = int.from_bytes(self._read_exact(4), "big")
        if size > 64 * 1024:
            raise ValueError("bridge RTT acknowledgement exceeds the maximum frame size")
        body = self._read_exact(size)
        acknowledgement = decode_message(size.to_bytes(4, "big") + body, secret=self.secret)
        finished = time.monotonic_ns()
        if (
            acknowledgement.kind != "ack"
            or acknowledgement.session_nonce != self.session_nonce
            or acknowledgement.sequence != sequence
            or acknowledgement.probe_nonce != probe_nonce
        ):
            raise ValueError("bridge RTT acknowledgement does not match the probe")
        with self._rtt_lock:
            self._last_rtt_ns = finished - started
            self._rtt_measured_monotonic_ns = finished
            return self._last_rtt_ns

    def _refresh_rtt_in_background(self) -> None:
        try:
            self.refresh_rtt()
        except BaseException as error:
            # Sampling checks this before constructing the next signed command;
            # a failed probe therefore holds rather than silently using a stale
            # health result.
            with self._rtt_lock:
                self._rtt_probe_error = error

    def start_rtt_refresh(self) -> bool:
        """Start at most one probe without delaying 30 Hz sample transmission."""
        with self._probe_lock:
            if self._rtt_probe_thread is not None and self._rtt_probe_thread.is_alive():
                return False
            worker = threading.Thread(target=self._refresh_rtt_in_background, name="lehome-bridge-rtt", daemon=True)
            self._rtt_probe_thread = worker
            worker.start()
            return True

    def _current_rtt(self) -> tuple[int, int]:
        with self._rtt_lock:
            if self._rtt_probe_error is not None:
                raise RuntimeError("bridge RTT refresh failed; refusing to stream stale health") from self._rtt_probe_error
            if self._last_rtt_ns is None or self._rtt_measured_monotonic_ns is None:
                raise RuntimeError("bridge RTT must be measured before samples")
            return self._last_rtt_ns, self._rtt_measured_monotonic_ns

    def send_sample(self, sample: LeaderSample) -> None:
        if self.sequence == 0:
            raise RuntimeError("bridge handshake must be sent before samples")
        rtt_ns, measured_ns = self._current_rtt()
        rtt_age_ns = time.monotonic_ns() - measured_ns
        with self._send_lock:
            sequence = self.sequence
            sample_sequence = self.sample_sequence
            self._send_unlocked(
                BridgeMessage.sample(
                    self.session_nonce,
                    sequence,
                    sample.monotonic_ns,
                    sample.positions,
                    raw_positions=sample.raw_positions,
                    sample_sequence=sample_sequence,
                    rtt_ns=rtt_ns,
                    rtt_age_ns=rtt_age_ns,
                )
            )
            self.sequence += 1
            self.sample_sequence += 1

    def request_stop(self) -> None:
        self.stop_requested = True

    def close(self) -> None:
        self.request_stop()
        if self.sock is not None:
            self.sock.close()
            self.sock = None
        with self._probe_lock:
            probe = self._rtt_probe_thread
        if probe is not None and probe is not threading.current_thread():
            probe.join(timeout=1.0)


def sleep_until_monotonic_ns(deadline_ns: int) -> None:
    remaining = deadline_ns - time.monotonic_ns()
    if remaining > 0:
        time.sleep(remaining / 1_000_000_000)


def stream(reader: DualLeaderReader, connection: BridgeConnection, *, hz: int = 30) -> None:
    """Read and send at a monotonic-deadline rate without serial side effects here."""
    if not isinstance(hz, int) or hz <= 0:
        raise ValueError("bridge hz must be positive")
    period_ns = int(1_000_000_000 / hz)
    if connection._rtt_measured_monotonic_ns is None:
        # Initial health is established before the cadence starts.  Subsequent
        # probes are independent so a normal WAN RTT cannot create a fake
        # serial/jitter fault.
        connection.refresh_rtt()
    deadline = time.monotonic_ns()
    while not connection.stop_requested:
        _, measured_ns = connection._current_rtt()
        if time.monotonic_ns() - measured_ns >= RTT_REFRESH_INTERVAL_NS:
            connection.start_rtt_refresh()
        connection.send_sample(reader.read())
        deadline += period_ns
        completed_at = time.monotonic_ns()
        if completed_at > deadline:
            missed_periods = (completed_at - deadline) // period_ns + 1
            deadline += missed_periods * period_ns
        sleep_until_monotonic_ns(deadline)
