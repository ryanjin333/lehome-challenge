from __future__ import annotations

import json
from pathlib import Path
import time

import pytest

from lehome_bridge.client import BridgeConnection, read_secret_file, stream
from lehome_bridge.leaders import JOINTS, DualLeaderReader
from lehome_bridge.protocol import BridgeMessage, MessageVerifier, encode_message


class FakeBus:
    def __init__(self, *, serial: str, positions: dict[str, float]) -> None:
        self.serial_identity = serial
        self.positions = positions
        self.reads = 0

    def sync_read(self, register: str, *, normalize: bool = True) -> dict[str, float]:
        assert register == "Present_Position"
        assert normalize is False
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
    assert sample.positions == pytest.approx((0.0, 0.1, 0.2, 0.3, 0.4, 50.25, 1.0, 1.1, 1.2, 1.3, 1.4, 50.75))
    assert sample.raw_positions == tuple(range(6)) + tuple(range(10, 16))
    assert sample.left_serial == "L1"
    assert sample.right_serial == "R1"
    assert reader.left_motor_limits == ((-1000.0, 1000.0),) * 6
    assert reader.right_motor_limits == ((-1000.0, 1000.0),) * 6
    assert left.reads == right.reads == 1


def test_reader_signs_one_raw_so101_read_and_the_matching_calibration_normalization(tmp_path: Path) -> None:
    """The raw bounds and normalized commands must describe the same instant."""
    left_calibration = tmp_path / "left.json"
    right_calibration = tmp_path / "right.json"
    values = {
        "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 0, "range_min": 480, "range_max": 3560},
        "shoulder_lift": {"id": 2, "drive_mode": 1, "homing_offset": 0, "range_min": 630, "range_max": 3450},
        "elbow_flex": {"id": 3, "drive_mode": 0, "homing_offset": 0, "range_min": 390, "range_max": 3710},
        "wrist_flex": {"id": 4, "drive_mode": 1, "homing_offset": 0, "range_min": 550, "range_max": 3510},
        "wrist_roll": {"id": 5, "drive_mode": 0, "homing_offset": 0, "range_min": 470, "range_max": 3620},
        "gripper": {"id": 6, "drive_mode": 1, "homing_offset": 0, "range_min": 830, "range_max": 3150},
    }
    left_calibration.write_text(json.dumps(values, sort_keys=True), encoding="utf-8")
    right_calibration.write_text(json.dumps(values, sort_keys=True), encoding="utf-8")
    raw = {name: values[name]["range_min"] + (values[name]["range_max"] - values[name]["range_min"]) / 4 for name in JOINTS}
    reader = DualLeaderReader(
        FakeBus(serial="left", positions=raw),
        FakeBus(serial="right", positions=raw),
        left_calibration=left_calibration,
        right_calibration=right_calibration,
    )

    sample = reader.read()

    assert sample.raw_positions == tuple(raw[name] for name in JOINTS) * 2
    assert sample.positions == pytest.approx((-50.0, 50.0, -50.0, 50.0, -50.0, 75.0) * 2)


def test_reader_rejects_raw_counts_outside_the_so101_calibration_before_normalizing(tmp_path: Path) -> None:
    reader = DualLeaderReader(
        FakeBus(serial="left", positions={name: 1001 for name in JOINTS}),
        FakeBus(serial="right", positions={name: 0 for name in JOINTS}),
        left_calibration=calibration(tmp_path, "left"),
        right_calibration=calibration(tmp_path, "right"),
    )

    with pytest.raises(ValueError, match="raw.*limits"):
        reader.read()


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


def test_near_80ms_rtt_refresh_does_not_block_30hz_samples(tmp_path: Path, monkeypatch) -> None:
    class DelayedAckSocket:
        def __init__(self) -> None:
            self.frames: list[tuple[str, int]] = []
            self.response = bytearray()
            self.delay_s = 0.0
            self._delay_pending = False

        def sendall(self, data: bytes) -> None:
            # A verifier cannot be reused because this test intentionally sees
            # handshake, samples, and probes from concurrent refreshes.
            from lehome_bridge.protocol import decode_message

            decoded = decode_message(data, secret=b"x" * 32)
            self.frames.append((decoded.kind, time.monotonic_ns()))
            if decoded.kind == "ping":
                self.response.extend(encode_message(BridgeMessage.ack("n", decoded.sequence, decoded.probe_nonce), secret=b"x" * 32))
                self._delay_pending = True

        def recv(self, size: int) -> bytes:
            if self._delay_pending:
                time.sleep(self.delay_s)
                self._delay_pending = False
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
    socket = DelayedAckSocket()
    connection = BridgeConnection(reader, secret=b"x" * 32, session_nonce="n", sock=socket)
    connection.connect()
    connection.refresh_rtt()
    socket.delay_s = 0.075
    monkeypatch.setattr("lehome_bridge.client.RTT_REFRESH_INTERVAL_NS", 1)

    class FourSamples:
        def __init__(self) -> None:
            self.count = 0

        def read(self):
            self.count += 1
            sample = reader.read()
            if self.count == 4:
                connection.request_stop()
            return sample

    try:
        stream(FourSamples(), connection, hz=30)
    finally:
        connection.close()

    sample_times = [timestamp for kind, timestamp in socket.frames if kind == "sample"]
    assert len(sample_times) == 4
    assert max(later - earlier for earlier, later in zip(sample_times, sample_times[1:])) < 60_000_000
