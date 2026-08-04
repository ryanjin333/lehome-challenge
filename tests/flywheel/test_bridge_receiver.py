from __future__ import annotations

import socket
import threading
import time
from types import SimpleNamespace

import pytest

from lehome.flywheel.bridge_receiver import BridgeReceiver, Handshake, LeaderSampleFrame, LoopbackBridgeServer
from lehome_bridge.client import BridgeConnection
from lehome_bridge.leaders import LeaderSample


def valid_handshake() -> Handshake:
    return Handshake(
        session_nonce="nonce",
        sequence=0,
        left_serial="left",
        right_serial="right",
        left_calibration_sha256="a" * 64,
        right_calibration_sha256="b" * 64,
        left_motor_limits=((0.0, 1.0),) * 6,
        right_motor_limits=((0.0, 1.0),) * 6,
        hz=30,
    )


def valid_sample(
    sequence: int,
    timestamp: int | None = None,
    *,
    rtt_ns: int = 1_000_000,
    rtt_age_ns: int = 0,
) -> LeaderSampleFrame:
    return LeaderSampleFrame("nonce", sequence, sequence * 33_333_333 if timestamp is None else timestamp, (0.0,) * 12, rtt_ns, rtt_age_ns)


def test_receiver_holds_after_stale_sample_and_requires_explicit_resync() -> None:
    receiver = BridgeReceiver(max_age_ms=80.0, max_jitter_ms=30.0, converter=lambda values: values)
    receiver.accept_handshake(valid_handshake())
    receiver.accept_sample(valid_sample(1), received_monotonic_ns=1_000_000_000)
    assert receiver.current(now_ns=1_010_000_000).eligible is True
    held = receiver.current(now_ns=1_100_000_000)
    assert held.eligible is False
    assert held.reason == "stale_sample"
    assert held.command == receiver.last_safe_command
    receiver.accept_sample(valid_sample(2), received_monotonic_ns=1_101_000_000)
    receiver.accept_sample(valid_sample(3), received_monotonic_ns=1_134_333_333)
    assert receiver.current(now_ns=1_135_000_000).reason == "resync_required"
    receiver.resync()
    assert receiver.current(now_ns=1_135_000_000).eligible is True


def test_receiver_rejects_jitter_disconnect_and_bad_session_sequence() -> None:
    receiver = BridgeReceiver(max_age_ms=100.0, max_jitter_ms=1.0, converter=lambda values: values)
    receiver.accept_handshake(valid_handshake())
    receiver.accept_sample(valid_sample(1), received_monotonic_ns=1_000_000_000)
    receiver.accept_sample(valid_sample(2), received_monotonic_ns=1_100_000_000)
    assert receiver.current(now_ns=1_100_000_001).reason == "jitter_exceeded"
    receiver.close_connection()
    assert receiver.current(now_ns=1_100_000_001).reason == "disconnected"
    with pytest.raises(ValueError, match="handshake"):
        BridgeReceiver(converter=lambda values: values).accept_sample(valid_sample(1))


def test_loopback_server_refuses_public_bind() -> None:
    with pytest.raises(ValueError, match="loopback"):
        LoopbackBridgeServer(secret=b"x" * 32, session_nonce="nonce", host="0.0.0.0")
    server = LoopbackBridgeServer(secret=b"x" * 32, session_nonce="nonce")
    assert server.host == "127.0.0.1"


def test_receiver_rejects_a_handshake_with_unexpected_motor_limits() -> None:
    receiver = BridgeReceiver(
        converter=lambda values: values,
        expected_motor_limits=(((0.0, 2.0),) * 6, ((0.0, 1.0),) * 6),
    )
    with pytest.raises(ValueError, match="motor limits"):
        receiver.accept_handshake(valid_handshake())


def test_receiver_rejects_raw_positions_outside_the_accepted_leader_limits() -> None:
    converted: list[tuple[float, ...]] = []
    receiver = BridgeReceiver(converter=lambda values: converted.append(values) or values)
    receiver.accept_handshake(valid_handshake())

    with pytest.raises(ValueError, match="raw leader position"):
        receiver.accept_sample(
            LeaderSampleFrame("nonce", 1, 100, (1.01,) + (0.0,) * 11, 1_000_000, 0),
            received_monotonic_ns=1_000_000_000,
        )

    assert converted == []
    assert receiver.last_sample is None
    assert receiver.requires_resync is True


def test_receiver_uses_only_same_clock_deltas_and_rejects_buffered_streams() -> None:
    receiver = BridgeReceiver(max_jitter_ms=30.0, converter=lambda values: values)
    receiver.accept_handshake(valid_handshake())
    # The absolute sender and receiver clock origins are intentionally unrelated.
    receiver.accept_sample(valid_sample(1, timestamp=8_000_000_000_000), received_monotonic_ns=1_000_000_000)
    receiver.accept_sample(valid_sample(2, timestamp=8_000_033_333_333), received_monotonic_ns=1_033_333_333)
    assert receiver.current(now_ns=1_034_000_000).eligible is True

    # A sender-side one-second gap replayed at the expected receiver cadence is
    # a buffered stream, not a healthy live leader sample.
    receiver.accept_sample(valid_sample(3, timestamp=8_001_033_333_333), received_monotonic_ns=1_066_666_666)
    assert receiver.current(now_ns=1_067_000_000).reason == "jitter_exceeded"


def test_receiver_requires_a_fresh_client_measured_rtt_before_expert_control() -> None:
    receiver = BridgeReceiver(max_rtt_ms=10.0, max_rtt_age_ms=20.0, converter=lambda values: values)
    receiver.accept_handshake(valid_handshake())
    receiver.accept_sample(
        valid_sample(1, rtt_ns=11_000_000),
        received_monotonic_ns=1_000_000_000,
    )

    assert receiver.current(now_ns=1_001_000_000).reason == "rtt_exceeded"

    receiver.accept_sample(
        valid_sample(2, timestamp=33_333_334, rtt_ns=1_000_000, rtt_age_ns=21_000_000),
        received_monotonic_ns=1_033_333_333,
    )
    assert receiver.current(now_ns=1_034_000_000).reason == "rtt_exceeded"

    with pytest.raises(ValueError, match="RTT"):
        receiver.resync()
    assert receiver.requires_resync is True


def test_loopback_bridge_keeps_leader_sample_sequence_contiguous_across_rtt_probes() -> None:
    class Reader:
        left_bus = SimpleNamespace(serial_identity="left")
        right_bus = SimpleNamespace(serial_identity="right")
        left_calibration = SimpleNamespace(sha256="a" * 64)
        right_calibration = SimpleNamespace(sha256="b" * 64)
        left_motor_limits = ((0.0, 1.0),) * 6
        right_motor_limits = ((0.0, 1.0),) * 6

    receiver = BridgeReceiver(max_jitter_ms=1_000.0, converter=lambda values: values)
    server = LoopbackBridgeServer(secret=b"x" * 32, session_nonce="nonce", port=0, receiver=receiver)
    server.start()
    assert server._listener is not None
    port = server._listener.getsockname()[1]
    def serve() -> None:
        try:
            server.serve_one_client()
        except ConnectionError:
            pass

    thread = threading.Thread(target=serve)
    thread.start()
    connection = BridgeConnection(Reader(), secret=b"x" * 32, session_nonce="nonce", port=port)
    try:
        connection.connect()
        connection.refresh_rtt()
        connection.send_sample(LeaderSample(time.monotonic_ns(), (0.0,) * 12, "left", "right"))
        connection.refresh_rtt()
        connection.send_sample(LeaderSample(time.monotonic_ns(), (0.0,) * 12, "left", "right"))
    finally:
        connection.close()
        thread.join(timeout=2.0)
        server.close()

    assert [sample.sequence for sample in receiver.records] == [1, 2]
