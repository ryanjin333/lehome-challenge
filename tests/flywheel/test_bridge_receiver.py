from __future__ import annotations

import socket

import pytest

from lehome.flywheel.bridge_receiver import BridgeReceiver, Handshake, LeaderSampleFrame, LoopbackBridgeServer


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


def valid_sample(sequence: int, timestamp: int = 1) -> LeaderSampleFrame:
    return LeaderSampleFrame("nonce", sequence, timestamp, (0.0,) * 12)


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
