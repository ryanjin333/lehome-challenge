from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import msgpack


def _request(protocol, *, request_id: str = "request-1", generation: int = 4, deadline_ns: int = 2_000):
    return protocol.PolicyRequest.infer(
        session_id="worker-0",
        episode_generation=generation,
        request_id=request_id,
        policy_sha256="a" * 64,
        deadline_ns=deadline_ns,
        observation={"state": [0.0] * 12, "language": "fold"},
    )


def test_canonical_typed_envelopes_round_trip_and_bind_all_identities() -> None:
    from lehome.flywheel import policy_protocol as protocol

    request = _request(protocol)
    wire = request.to_wire()

    assert set(protocol.REQUIRED_ENVELOPE_FIELDS) <= set(wire)
    assert protocol.unpack_envelope(protocol.pack_envelope(request)) == request
    assert request.canonical_metadata_bytes() == protocol.PolicyRequest.infer(
        session_id="worker-0",
        episode_generation=4,
        request_id="request-1",
        policy_sha256="a" * 64,
        deadline_ns=2_000,
        observation={"language": "fold", "state": [0.0] * 12},
    ).canonical_metadata_bytes()

    with pytest.raises(ValueError, match="deadline_ns"):
        protocol.envelope_from_wire({key: value for key, value in wire.items() if key != "deadline_ns"})


def test_msgpack_round_trips_real_numpy_observation_without_pickle() -> None:
    from lehome.flywheel import policy_protocol as protocol

    request = protocol.PolicyRequest.infer(
        session_id="worker-0",
        episode_generation=4,
        request_id="request-1",
        policy_sha256="a" * 64,
        deadline_ns=2_000,
        observation={
            "video": {
                camera: np.full((2, 3, 3), index, dtype=np.uint8)
                for index, camera in enumerate(("top_rgb", "left_rgb", "right_rgb"))
            },
            "state": np.arange(12, dtype=np.float32),
            "language": {"annotation.human.task_description": [["fold the garment on the table"]]},
        },
    )

    decoded = protocol.unpack_envelope(protocol.pack_envelope(request))

    assert isinstance(decoded, protocol.PolicyRequest)
    assert decoded.observation is not None
    np.testing.assert_array_equal(decoded.observation["state"], request.observation["state"])
    for camera in ("top_rgb", "left_rgb", "right_rgb"):
        frame = decoded.observation["video"][camera]
        assert frame.dtype == np.uint8
        assert frame.shape == (2, 3, 3)
        np.testing.assert_array_equal(frame, request.observation["video"][camera])

    unsafe = request.to_wire()
    unsafe["observation"] = {"state": {"__ndarray_class__": True, "as_npy": b"not-an-npy"}}
    with pytest.raises(ValueError, match="ndarray payload"):
        protocol.unpack_envelope(msgpack.packb(unsafe, use_bin_type=True))


def test_guard_rejects_expired_duplicate_and_cancelled_requests_and_recovers_after_gateway_restart() -> None:
    from lehome.flywheel import policy_protocol as protocol

    guard = protocol.SessionRequestGuard(policy_sha256="a" * 64)
    reset = protocol.PolicyRequest.reset(
        session_id="worker-0",
        episode_generation=4,
        request_id="reset-4",
        policy_sha256="a" * 64,
        deadline_ns=2_000,
    )
    guard.accept(reset, now_ns=1_000)
    request = _request(protocol)
    guard.accept(request, now_ns=1_000)
    with pytest.raises(protocol.DuplicateRequestError):
        guard.accept(request, now_ns=1_000)

    cancel = protocol.PolicyRequest.cancel(
        session_id="worker-0",
        episode_generation=4,
        request_id="cancel-1",
        policy_sha256="a" * 64,
        deadline_ns=2_000,
        cancelled_request_id=request.request_id,
    )
    guard.accept(cancel, now_ns=1_000)
    assert guard.is_cancelled(request) is True
    with pytest.raises(protocol.ExpiredRequestError):
        guard.accept(_request(protocol, request_id="late", deadline_ns=1_000), now_ns=1_000)

    restarted_gateway = protocol.SessionRequestGuard(policy_sha256="a" * 64)
    restarted_gateway.accept(reset, now_ns=1_000)
    restarted_gateway.accept(_request(protocol, request_id="after-restart"), now_ns=1_000)


def test_guard_prunes_old_generation_and_exposes_only_live_uncancelled_inference() -> None:
    from lehome.flywheel import policy_protocol as protocol

    guard = protocol.SessionRequestGuard(policy_sha256="a" * 64)
    reset_one = protocol.PolicyRequest.reset(
        session_id="worker-0", episode_generation=1, request_id="reset-1",
        policy_sha256="a" * 64, deadline_ns=2_000,
    )
    old_request = _request(protocol, request_id="old-request", generation=1)
    guard.accept(reset_one, now_ns=1_000)
    guard.accept(old_request, now_ns=1_000)
    assert guard.is_request_live(old_request, now_ns=1_000) is True

    reset_two = protocol.PolicyRequest.reset(
        session_id="worker-0", episode_generation=2, request_id="reset-2",
        policy_sha256="a" * 64, deadline_ns=2_000,
    )
    guard.accept(reset_two, now_ns=1_000)
    assert guard.is_request_live(old_request, now_ns=1_000) is False
    assert set(guard._seen_request_ids) == {("worker-0", 2)}

    current_request = _request(protocol, request_id="current-request", generation=2)
    guard.accept(current_request, now_ns=1_000)
    assert guard.is_request_live(current_request, now_ns=1_000) is True
    cancel = protocol.PolicyRequest.cancel(
        session_id="worker-0", episode_generation=2, request_id="cancel-current",
        policy_sha256="a" * 64, deadline_ns=2_000,
        cancelled_request_id=current_request.request_id,
    )
    guard.accept(cancel, now_ns=1_000)
    assert guard.is_request_live(current_request, now_ns=1_000) is False

    expiring_request = _request(protocol, request_id="expiring-request", generation=2, deadline_ns=1_001)
    guard.accept(expiring_request, now_ns=1_000)
    assert guard.is_request_live(expiring_request, now_ns=1_001) is False


def test_guard_rejects_unknown_or_stale_generations_without_allocating_state() -> None:
    from lehome.flywheel import policy_protocol as protocol

    guard = protocol.SessionRequestGuard(policy_sha256="a" * 64)
    unknown = protocol.PolicyRequest.infer(
        session_id="unknown-worker", episode_generation=1, request_id="unknown-request",
        policy_sha256="a" * 64, deadline_ns=2_000, observation={"state": [0.0] * 12},
    )
    with pytest.raises(protocol.SessionStateError):
        guard.accept(unknown, now_ns=1_000)
    assert guard._seen_request_ids == {}

    reset = protocol.PolicyRequest.reset(
        session_id="worker-0", episode_generation=2, request_id="reset-2",
        policy_sha256="a" * 64, deadline_ns=2_000,
    )
    guard.accept(reset, now_ns=1_000)
    stale = _request(protocol, request_id="stale-request", generation=1)
    with pytest.raises(protocol.SessionStateError):
        guard.accept(stale, now_ns=1_000)
    assert set(guard._seen_request_ids) == {("worker-0", 2)}


def test_response_validation_rejects_stale_generation_and_deadline() -> None:
    from lehome.flywheel import policy_protocol as protocol

    request = _request(protocol)
    response = protocol.PolicyResponse.ok(request, action_chunk=b"chunk", action_horizon=16)
    assert protocol.validate_response_for_request(response, request, now_ns=1_000) == response

    with pytest.raises(protocol.StaleResponseError):
        protocol.validate_response_for_request(
            replace(response, episode_generation=request.episode_generation - 1), request, now_ns=1_000
        )
    with pytest.raises(protocol.ExpiredRequestError):
        protocol.validate_response_for_request(response, request, now_ns=request.deadline_ns)


def test_guard_treats_identical_reset_identity_as_idempotent_reattachment() -> None:
    from lehome.flywheel import policy_protocol as protocol

    guard = protocol.SessionRequestGuard(policy_sha256="a" * 64)
    reset = protocol.PolicyRequest.reset(
        session_id="worker-0",
        episode_generation=1,
        request_id="reset-1",
        policy_sha256="a" * 64,
        deadline_ns=2_000,
    )
    guard.accept(reset, now_ns=1_000)
    guard.accept(reset, now_ns=1_000)
    infer = protocol.PolicyRequest.infer(
        session_id="worker-0",
        episode_generation=1,
        request_id="infer-1",
        policy_sha256="a" * 64,
        deadline_ns=2_000,
        observation={"state": [0.0] * 12},
    )
    guard.accept(infer, now_ns=1_000)
    assert guard.is_request_live(infer, now_ns=1_000) is True
