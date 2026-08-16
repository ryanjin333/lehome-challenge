from __future__ import annotations

import numpy as np
import pytest


def _request(protocol, *, request_id: str, deadline_ns: int = 10_000, value: int = 0):
    return protocol.PolicyRequest.infer(
        session_id=f"worker-{request_id}",
        episode_generation=1,
        request_id=request_id,
        policy_sha256="a" * 64,
        deadline_ns=deadline_ns,
        observation={
            "video": {camera: np.full((1, 1, 2, 3, 3), value, dtype=np.uint8) for camera in ("top_rgb", "left_rgb", "right_rgb")},
            "state": {
                "left_arm": np.full((1, 1, 5), value, dtype=np.float32),
                "left_gripper": np.full((1, 1, 1), value, dtype=np.float32),
                "right_arm": np.full((1, 1, 5), value, dtype=np.float32),
                "right_gripper": np.full((1, 1, 1), value, dtype=np.float32),
            },
            "language": {"annotation.human.task_description": [["fold the garment on the table"]]},
        },
    )


class _Model:
    def __init__(self) -> None:
        self.observations = []

    def get_action(self, observation):
        self.observations.append(observation)
        batch = observation["state"]["left_arm"].shape[0]
        return {
            "left_arm": np.stack([np.full((16, 5), index, dtype=np.float32) for index in range(batch)]),
            "left_gripper": np.stack([np.full((16, 1), index, dtype=np.float32) for index in range(batch)]),
            "right_arm": np.stack([np.full((16, 5), index, dtype=np.float32) for index in range(batch)]),
            "right_gripper": np.stack([np.full((16, 1), index, dtype=np.float32) for index in range(batch)]),
        }, {"ignored": True}


def test_batcher_collates_four_nested_observations_once_and_preserves_request_order() -> None:
    from lehome.flywheel import policy_protocol as protocol
    from lehome.flywheel.policy_batcher import PolicyBatcher

    model = _Model()
    batcher = PolicyBatcher(model, policy_sha256="a" * 64, batch_window_ns=100)
    requests = [_request(protocol, request_id=f"request-{index}", value=index) for index in range(4)]
    for request in requests:
        batcher.enqueue(request, received_ns=1_000)

    results = batcher.flush(now_ns=1_000)

    assert [result.request.request_id for result in results] == [request.request_id for request in requests]
    assert [result.response.status for result in results] == ["ok"] * 4
    assert len(model.observations) == 1
    assert model.observations[0]["video"]["top_rgb"].shape == (4, 1, 2, 3, 3)
    assert model.observations[0]["state"]["right_arm"].shape == (4, 1, 5)
    assert model.observations[0]["language"]["annotation.human.task_description"] == [
        ["fold the garment on the table"],
    ] * 4
    for index, result in enumerate(results):
        action = np.frombuffer(result.response.action_chunk, dtype=np.float32).reshape(16, 12)
        np.testing.assert_array_equal(action, np.full((16, 12), index, dtype=np.float32))


def test_batcher_flushes_partial_batch_at_window_or_deadline_and_ignores_cancelled_request() -> None:
    from lehome.flywheel import policy_protocol as protocol
    from lehome.flywheel.policy_batcher import PolicyBatcher

    model = _Model()
    batcher = PolicyBatcher(model, policy_sha256="a" * 64, batch_window_ns=100)
    early = _request(protocol, request_id="early", deadline_ns=1_050)
    cancelled = _request(protocol, request_id="cancelled")
    batcher.enqueue(early, received_ns=1_000)
    batcher.enqueue(cancelled, received_ns=1_000)
    batcher.cancel(cancelled)

    results = batcher.flush(now_ns=1_000)

    assert [result.request.request_id for result in results] == ["early"]
    assert len(model.observations) == 1


def test_batcher_rejects_malformed_observation_without_calling_model() -> None:
    from lehome.flywheel import policy_protocol as protocol
    from lehome.flywheel.policy_batcher import PolicyBatcher

    model = _Model()
    batcher = PolicyBatcher(model, policy_sha256="a" * 64, batch_window_ns=100)
    malformed = _request(protocol, request_id="bad")
    malformed = protocol.PolicyRequest.infer(
        session_id=malformed.session_id,
        episode_generation=malformed.episode_generation,
        request_id=malformed.request_id,
        policy_sha256=malformed.policy_sha256,
        deadline_ns=malformed.deadline_ns,
        observation={"video": {"top_rgb": np.zeros((1, 2, 3), dtype=np.uint8)}},
    )
    batcher.enqueue(malformed, received_ns=1_000)

    results = batcher.flush(now_ns=1_100)

    assert [(result.response.status, result.response.error_code) for result in results] == [
        ("error", "invalid_observation")
    ]
    assert model.observations == []


def test_batcher_reports_pre_inference_expiry_as_a_discarded_request() -> None:
    from lehome.flywheel import policy_protocol as protocol
    from lehome.flywheel.policy_batcher import PolicyBatcher

    batcher = PolicyBatcher(_Model(), policy_sha256="a" * 64, batch_window_ns=100)
    request = _request(protocol, request_id="expired", deadline_ns=1_050)
    batcher.enqueue(request, received_ns=1_000)

    flushed = batcher.flush(now_ns=1_050)

    assert flushed == []
    assert [(item.request.request_id, item.reason) for item in flushed.discarded] == [
        ("expired", "expired")
    ]
