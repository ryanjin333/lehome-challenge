from __future__ import annotations

import struct
import json

import pytest

from lehome_bridge.protocol import BridgeMessage, MessageVerifier, encode_message


def handshake() -> BridgeMessage:
    return BridgeMessage.handshake(
        session_nonce="nonce-1",
        sequence=0,
        left_serial="left-001",
        right_serial="right-002",
        left_calibration_sha256="a" * 64,
        right_calibration_sha256="b" * 64,
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
        encode_message(BridgeMessage.sample("n", 1, 10, [0.0] * 12), secret=b"k" * 32)
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
            hz=30,
        )
    with pytest.raises(ValueError, match="finite 12D"):
        BridgeMessage.sample("n", 1, 10, [float("nan")] * 12)
    with pytest.raises(ValueError, match="maximum"):
        MessageVerifier(secret=b"k" * 32, expected_nonce="n").verify(
            struct.pack("!I", 65_537) + b"x" * 65_537
        )


def test_signed_sample_rejects_a_non_v1_protocol_declaration() -> None:
    sample = BridgeMessage.sample("n", 1, 10, [0.0] * 12)
    payload = sample.to_dict()
    payload["protocol_version"] = 2
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError, match="unsupported"):
        BridgeMessage.from_json(canonical)
