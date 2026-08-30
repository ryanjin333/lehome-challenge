from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from scripts.serve_official_docker_policy_bridge import (
    BridgeProtocolError,
    OfficialDockerPolicyBridge,
    decode_official_observation,
)


def _image_envelope(value: np.ndarray) -> dict[str, object]:
    return {
        "base64": base64.b64encode(value.tobytes()).decode("ascii"),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def _payload() -> dict[str, object]:
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    depth = np.arange(2 * 3, dtype=np.uint16).reshape(2, 3)
    return {
        "action": [0.0] * 12,
        "observation.state": [float(index) for index in range(12)],
        "observation.images.top_rgb": _image_envelope(image),
        "observation.images.left_rgb": _image_envelope(image + 1),
        "observation.images.right_rgb": _image_envelope(image + 2),
        "observation.top_depth": _image_envelope(depth),
    }


class _Client:
    def __init__(self) -> None:
        self.reset_count = 0
        self.observation = None

    def reset(self) -> None:
        self.reset_count += 1

    def get_action(self, observation):
        self.observation = observation
        return {
            "left_arm": np.zeros((1, 16, 5), dtype=np.float32),
            "left_gripper": np.zeros((1, 16, 1), dtype=np.float32),
            "right_arm": np.zeros((1, 16, 5), dtype=np.float32),
            "right_gripper": np.zeros((1, 16, 1), dtype=np.float32),
        }, {}


def _build_observation(observation):
    state = observation["observation.state"]
    return {
        "video": {
            name: observation[f"observation.images.{name}"][None, None, ...]
            for name in ("top_rgb", "left_rgb", "right_rgb")
        },
        "state": {"all": state[None, None, ...]},
        "language": {"annotation.human.task_description": [["fold the garment on the table"]]},
    }


def _validate_action(action):
    return np.concatenate(
        [action[name][0] for name in ("left_arm", "left_gripper", "right_arm", "right_gripper")],
        axis=1,
    )


def _bridge(client):
    return OfficialDockerPolicyBridge(
        client,
        observation_builder=_build_observation,
        action_validator=_validate_action,
    )


def test_bridge_decodes_official_base64_observations_and_returns_exact_action_chunk() -> None:
    client = _Client()
    bridge = _bridge(client)

    assert bridge.reset({}) == {"status": "ok"}
    response = bridge.infer(_payload())

    assert client.reset_count == 1
    assert set(client.observation) == {"video", "state", "language"}
    assert np.asarray(client.observation["video"]["top_rgb"]).shape == (1, 1, 2, 3, 3)
    assert list(response) == ["actions"]
    actions = np.asarray(response["actions"], dtype=np.float32)
    assert actions.shape == (16, 12)
    assert np.isfinite(actions).all()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update({"garment_category": "top_long"}),
        lambda payload: payload["observation.images.top_rgb"].update({"extra": True}),
        lambda payload: payload["observation.images.top_rgb"].update({"dtype": "object"}),
        lambda payload: payload["observation.images.top_rgb"].update({"shape": [2, 3, 4]}),
        lambda payload: payload.update({"observation.state": [0.0] * 11}),
    ],
)
def test_bridge_rejects_labels_and_malformed_ndarray_envelopes(mutation) -> None:
    payload = _payload()
    mutation(payload)

    with pytest.raises(BridgeProtocolError):
        decode_official_observation(payload)


def test_bridge_rejects_nonfinite_or_wrong_action_chunks() -> None:
    client = _Client()
    bridge = _bridge(client)
    client.get_action = lambda _observation: (
        {
            "left_arm": np.full((1, 16, 5), np.nan, dtype=np.float32),
            "left_gripper": np.zeros((1, 16, 1), dtype=np.float32),
            "right_arm": np.zeros((1, 16, 5), dtype=np.float32),
            "right_gripper": np.zeros((1, 16, 1), dtype=np.float32),
        },
        {},
    )

    with pytest.raises(BridgeProtocolError, match="finite 16x12"):
        bridge.infer(_payload())


def test_bridge_json_decoder_rejects_duplicate_keys() -> None:
    with pytest.raises(BridgeProtocolError, match="duplicate"):
        OfficialDockerPolicyBridge.decode_json(
            b'{"observation.state":[],"observation.state":[]}'
        )


def test_bridge_json_response_never_contains_policy_metadata() -> None:
    response = _bridge(_Client()).infer(_payload())
    raw = json.dumps(response, allow_nan=False)
    assert "garment" not in raw
    assert "category" not in raw


def test_bridge_accepts_exact_official_action_field_but_does_not_forward_it_to_policy() -> None:
    client = _Client()
    bridge = _bridge(client)
    payload = _payload()
    payload["action"] = [float(index) for index in range(12)]

    decoded = decode_official_observation(payload)
    assert decoded["action"].shape == (12,)
    bridge.infer(payload)
    assert "action" not in client.observation
