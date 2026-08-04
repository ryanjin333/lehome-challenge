from __future__ import annotations

import json
from pathlib import Path

import pytest

from lehome_bridge.client import BridgeConnection, read_secret_file
from lehome_bridge.leaders import JOINTS, DualLeaderReader
from lehome_bridge.protocol import BridgeMessage, MessageVerifier, encode_message


class FakeBus:
    def __init__(self, *, serial: str, positions: dict[str, float]) -> None:
        self.serial_identity = serial
        self.positions = positions
        self.reads = 0

    def sync_read(self, register: str) -> dict[str, float]:
        assert register == "Present_Position"
        self.reads += 1
        return self.positions


def calibration(root: Path, name: str) -> Path:
    path = root / f"{name}.json"
    path.write_text(
        json.dumps(
            {
                joint: {
                    "id": index + 1,
                    "drive_mode": 0,
                    "homing_offset": 0,
                    "range_min": -1000,
                    "range_max": 1000,
                }
                for index, joint in enumerate(JOINTS)
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def test_dual_reader_returns_left_then_right_joint_order(tmp_path: Path) -> None:
    left = FakeBus(serial="L1", positions={name: index for index, name in enumerate(JOINTS)})
    right = FakeBus(serial="R1", positions={name: index + 10 for index, name in enumerate(JOINTS)})
    reader = DualLeaderReader(
        left,
        right,
        left_calibration=calibration(tmp_path, "left"),
        right_calibration=calibration(tmp_path, "right"),
    )
    sample = reader.read()
    assert sample.positions == (0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15)
    assert sample.left_serial == "L1"
    assert sample.right_serial == "R1"
    assert reader.left_motor_limits == ((-1000.0, 1000.0),) * 6
    assert reader.right_motor_limits == ((-1000.0, 1000.0),) * 6
    assert left.reads == right.reads == 1


def test_reader_rejects_same_bus_or_invalid_calibration(tmp_path: Path) -> None:
    bus = FakeBus(serial="same", positions={name: 0 for name in JOINTS})
    with pytest.raises(ValueError, match="distinct"):
        DualLeaderReader(
            bus,
            bus,
            left_calibration=calibration(tmp_path, "left"),
            right_calibration=calibration(tmp_path, "right"),
        )
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="calibration"):
        DualLeaderReader(
            FakeBus(serial="left", positions={name: 0 for name in JOINTS}),
            FakeBus(serial="right", positions={name: 0 for name in JOINTS}),
            left_calibration=invalid,
            right_calibration=calibration(tmp_path, "right-2"),
        )


def test_secret_file_must_be_private_and_at_least_32_bytes(tmp_path: Path) -> None:
    secret = tmp_path / "secret"
    secret.write_bytes(b"x" * 32)
    secret.chmod(0o600)
    assert read_secret_file(secret) == b"x" * 32
    secret.chmod(0o644)
    with pytest.raises(ValueError, match="0600"):
        read_secret_file(secret)


def test_connection_sends_exact_calibration_derived_motor_limits(tmp_path: Path) -> None:
    class FakeSocket:
        def __init__(self) -> None:
            self.frames: list[bytes] = []

        def sendall(self, data: bytes) -> None:
            self.frames.append(data)

        def close(self) -> None:
            pass

    reader = DualLeaderReader(
        FakeBus(serial="left", positions={name: 0 for name in JOINTS}),
        FakeBus(serial="right", positions={name: 0 for name in JOINTS}),
        left_calibration=calibration(tmp_path, "left"),
        right_calibration=calibration(tmp_path, "right"),
    )
    socket = FakeSocket()
    BridgeConnection(reader, secret=b"x" * 32, session_nonce="n", sock=socket).connect()
    handshake = MessageVerifier(secret=b"x" * 32, expected_nonce="n").verify(socket.frames[0])
    assert handshake.left_motor_limits == reader.left_motor_limits
    assert handshake.right_motor_limits == reader.right_motor_limits


def test_connection_attaches_a_nonce_bound_client_clock_rtt_to_samples(tmp_path: Path, monkeypatch) -> None:
    class FakeSocket:
        def __init__(self) -> None:
            self.frames: list[bytes] = []
            self.response = bytearray(
                encode_message(BridgeMessage.ack("n", 1, "probe"), secret=b"x" * 32)
            )

        def sendall(self, data: bytes) -> None:
            self.frames.append(data)

        def recv(self, size: int) -> bytes:
            chunk = bytes(self.response[:size])
            del self.response[:size]
            return chunk

        def close(self) -> None:
            pass

    reader = DualLeaderReader(
        FakeBus(serial="left", positions={name: 0 for name in JOINTS}),
        FakeBus(serial="right", positions={name: 0 for name in JOINTS}),
        left_calibration=calibration(tmp_path, "left"),
        right_calibration=calibration(tmp_path, "right"),
    )
    ticks = iter((1_000, 1_250, 1_300, 1_350))
    monkeypatch.setattr("lehome_bridge.client.token_urlsafe", lambda _: "probe")
    monkeypatch.setattr("lehome_bridge.client.time.monotonic_ns", lambda: next(ticks))
    socket = FakeSocket()
    connection = BridgeConnection(reader, secret=b"x" * 32, session_nonce="n", sock=socket)

    connection.connect()
    assert connection.refresh_rtt() == 250
    connection.send_sample(reader.read())

    verifier = MessageVerifier(secret=b"x" * 32, expected_nonce="n")
    assert verifier.verify(socket.frames[0]).kind == "handshake"
    assert verifier.verify(socket.frames[1]).kind == "ping"
    sample = verifier.verify(socket.frames[2])
    assert sample.rtt_ns == 250
    assert sample.rtt_age_ns == 100
