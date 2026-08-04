from __future__ import annotations

import struct
import json

import pytest

from lehome_bridge.protocol import BridgeMessage, MessageVerifier, encode_message


LIMITS = tuple((float(index), float(index + 1)) for index in range(6))


def handshake() -> BridgeMessage:
    return BridgeMessage.handshake(
        session_nonce="nonce-1",
        sequence=0,
        left_serial="left-001",
        right_serial="right-002",
        left_calibration_sha256="a" * 64,
        right_calibration_sha256="b" * 64,
        left_motor_limits=LIMITS,
        right_motor_limits=LIMITS,
        hz=30,
    )


def test_signed_message_round_trip_and_replay_rejection() -> None:
    secret = b"x" * 32
    verifier = MessageVerifier(secret=secret, expected_nonce="nonce-1")
    wire = encode_message(handshake(), secret=secret)
    assert verifier.verify(wire).sequence == 0
    with pytest.raises(ValueError, match="sequence"):
        verifier.verify(wire)


def test_tampered_message_fails_authentication() -> None:
    wire = bytearray(
        encode_message(BridgeMessage.sample("n", 1, 10, [0.0] * 12, raw_positions=[1000.0] * 12, sample_sequence=1, rtt_ns=1, rtt_age_ns=0), secret=b"k" * 32)
    )
    wire[-1] ^= 1
    with pytest.raises(ValueError, match="authentication"):
        MessageVerifier(secret=b"k" * 32, expected_nonce="n").verify(bytes(wire))


def test_protocol_rejects_bad_identity_nonfinite_samples_and_oversize_frames() -> None:
    with pytest.raises(ValueError, match="distinct"):
        BridgeMessage.handshake(
            session_nonce="n",
            sequence=0,
            left_serial="same",
            right_serial="same",
            left_calibration_sha256="a" * 64,
            right_calibration_sha256="b" * 64,
            left_motor_limits=LIMITS,
            right_motor_limits=LIMITS,
            hz=30,
        )
    with pytest.raises(ValueError, match="finite 12D"):
        BridgeMessage.sample("n", 1, 10, [float("nan")] * 12, raw_positions=[1000.0] * 12, sample_sequence=1, rtt_ns=1, rtt_age_ns=0)
    with pytest.raises(ValueError, match="maximum"):
        MessageVerifier(secret=b"k" * 32, expected_nonce="n").verify(
            struct.pack("!I", 65_537) + b"x" * 65_537
        )


def test_signed_sample_rejects_a_stale_protocol_declaration() -> None:
    sample = BridgeMessage.sample("n", 1, 10, [0.0] * 12, raw_positions=[1000.0] * 12, sample_sequence=1, rtt_ns=1, rtt_age_ns=0)
    payload = sample.to_dict()
    payload["protocol_version"] = 2
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError, match="unsupported"):
        BridgeMessage.from_json(canonical)

    payload = sample.to_dict()
    del payload["raw_positions"]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError, match="invalid sample fields"):
        BridgeMessage.from_json(canonical)


def test_handshake_round_trips_ordered_motor_limits_and_samples_cannot_repeat_them() -> None:
    message = handshake()
    restored = BridgeMessage.from_json(message_to_payload(message))
    assert restored.left_motor_limits == LIMITS
    assert restored.right_motor_limits == LIMITS
    with pytest.raises(ValueError, match="motor limits"):
        BridgeMessage(
            kind="sample",
            session_nonce="n",
            sequence=1,
            monotonic_ns=1,
            positions=(0.0,) * 12,
            raw_positions=(1000.0,) * 12,
            sample_sequence=1,
            rtt_ns=1,
            rtt_age_ns=0,
            left_motor_limits=LIMITS,
        )
    with pytest.raises(ValueError, match="motor limits"):
        BridgeMessage.handshake(
            session_nonce="n",
            sequence=0,
            left_serial="left",
            right_serial="right",
            left_calibration_sha256="a" * 64,
            right_calibration_sha256="b" * 64,
            left_motor_limits=((1.0, 1.0),) * 6,
            right_motor_limits=LIMITS,
            hz=30,
        )


def test_ping_ack_echoes_the_probe_nonce_for_client_clock_rtt_measurement() -> None:
    ping = BridgeMessage.ping("nonce-1", 1, "probe-1")
    ack = BridgeMessage.ack("nonce-1", 1, "probe-1")

    assert BridgeMessage.from_json(message_to_payload(ping)) == ping
    assert BridgeMessage.from_json(message_to_payload(ack)) == ack


def test_sample_authenticates_raw_encoder_counts_separately_from_normalized_positions() -> None:
    sample = BridgeMessage.sample(
        "nonce-1", 1, 10, [-50.0, 50.0, -50.0, 50.0, -50.0, 75.0] * 2,
        raw_positions=[1250.0, 1335.0, 1220.0, 1290.0, 1257.5, 1410.0] * 2,
        sample_sequence=1,
        rtt_ns=1,
        rtt_age_ns=0,
    )

    restored = BridgeMessage.from_json(message_to_payload(sample))

    assert restored.positions == sample.positions
    assert restored.raw_positions == sample.raw_positions


def message_to_payload(message: BridgeMessage) -> bytes:
    return json.dumps(message.to_dict(), sort_keys=True, separators=(",", ":")).encode()
