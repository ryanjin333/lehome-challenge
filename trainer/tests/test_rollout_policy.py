from __future__ import annotations

import importlib
import json
import pickle
from pathlib import Path
import sys
import types

import numpy as np
import pytest


def _load_policy_module():
    """Load the adapter without importing Isaac/LeRobot dependencies."""

    package = types.ModuleType("scripts.eval_policy")
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "scripts" / "eval_policy")]
    sys.modules.setdefault("scripts", types.ModuleType("scripts"))
    sys.modules["scripts.eval_policy"] = package
    for name in ("scripts.eval_policy.base_policy", "scripts.eval_policy.registry"):
        sys.modules.pop(name, None)
    sys.modules.pop("scripts.eval_policy.groot_policy", None)
    return importlib.import_module("scripts.eval_policy.groot_policy")


def _chunk(batch: int, horizon: int, dimension: int, start: float) -> np.ndarray:
    values = np.arange(batch * horizon * dimension, dtype=np.float32) + start
    return values.reshape(batch, horizon, dimension)


def test_policy_loader_uses_the_repository_adapter_path():
    module = _load_policy_module()

    assert Path(module.__file__).resolve().parent == Path(__file__).resolve().parents[2] / "scripts" / "eval_policy"


def test_flatten_groot_action_chunk_preserves_all_horizon_steps_and_group_order():
    module = _load_policy_module()
    action = {
        "left_arm": _chunk(1, 16, 5, 0),
        "left_gripper": _chunk(1, 16, 1, 10),
        "right_arm": _chunk(1, 16, 5, 20),
        "right_gripper": _chunk(1, 16, 1, 30),
    }

    flattened = module.flatten_groot_action_chunk(action)

    assert flattened.shape == (16, 12)
    np.testing.assert_array_equal(
        flattened[0],
        np.array([0, 1, 2, 3, 4, 10, 20, 21, 22, 23, 24, 30], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        flattened[1],
        np.array([5, 6, 7, 8, 9, 11, 25, 26, 27, 28, 29, 31], dtype=np.float32),
    )


def test_action_chunk_queue_returns_each_step_then_refills():
    module = _load_policy_module()
    queue = module.ActionChunkQueue()
    chunk = np.arange(24, dtype=np.float32).reshape(2, 12)

    queue.extend(chunk)

    np.testing.assert_array_equal(queue.pop(), chunk[0])
    np.testing.assert_array_equal(queue.pop(), chunk[1])
    assert queue.pop() is None


def test_action_chunk_queue_reports_pending_actions():
    module = _load_policy_module()
    queue = module.ActionChunkQueue()
    chunk = np.arange(24, dtype=np.float32).reshape(2, 12)

    assert queue.pending_count == 0
    queue.extend(chunk)
    assert queue.pending_count == 2
    queue.pop()
    assert queue.pending_count == 1
    queue.clear()
    assert queue.pending_count == 0

    queue.extend(chunk)
    queue.clear()
    assert queue.pop() is None


def test_action_queue_reports_request_and_offset():
    module = _load_policy_module()
    queue = module.ActionChunkQueue()
    queue.extend(np.zeros((16, 12), dtype=np.float32), request_id="req-7")

    item = queue.pop_with_provenance()

    assert item.request_id == "req-7"
    assert item.chunk_offset == 0


def test_policy_records_only_cache_miss_inference_latency_and_queue_depth(monkeypatch, tmp_path):
    module = _load_policy_module()
    telemetry_path = tmp_path / "policy-telemetry.jsonl"
    telemetry_path.touch()
    monkeypatch.setenv("LEHOME_FLYWHEEL_POLICY_TELEMETRY_PATH", str(telemetry_path))
    ticks = iter((1_000_000_000, 1_250_000_000))
    monkeypatch.setattr(module, "perf_counter_ns", lambda: next(ticks))

    class FakeOfficialPolicy:
        calls = 0

        def get_action(self, _observation):
            self.calls += 1
            return {
                "action.left_arm": _chunk(1, 16, 5, 0),
                "action.left_gripper": _chunk(1, 16, 1, 10),
                "action.right_arm": _chunk(1, 16, 5, 20),
                "action.right_gripper": _chunk(1, 16, 1, 30),
            }, None

    policy = module.GrootPolicy.__new__(module.GrootPolicy)
    policy._policy = FakeOfficialPolicy()
    policy._action_queue = module.ActionChunkQueue()
    observation = {
        "observation.state": np.zeros(12, dtype=np.float32),
        **{f"observation.images.{camera}": np.zeros((2, 2, 3), dtype=np.uint8) for camera in ("top_rgb", "left_rgb", "right_rgb")},
    }

    first = policy.select_action_with_provenance(observation)
    second = policy.select_action_with_provenance(observation)

    records = [json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines()]
    assert first.request_id == second.request_id
    assert policy._policy.calls == 1
    assert records == [{
        "request_id": first.request_id,
        "latency_seconds": 0.25,
        "queue_depth_after_enqueue": 16,
    }]

def test_flatten_groot_action_chunk_rejects_non_contract_horizon():
    module = _load_policy_module()
    action = {
        "action.left_arm": _chunk(1, 2, 5, 0),
        "action.left_gripper": _chunk(1, 2, 1, 10),
        "action.right_arm": _chunk(1, 2, 5, 20),
        "action.right_gripper": _chunk(1, 2, 1, 30),
    }

    try:
        module.flatten_groot_action_chunk(action)
    except ValueError as error:
        assert "horizon" in str(error)
    else:
        raise AssertionError("a non-16-step action chunk must be rejected")


def _install_wire_fakes(monkeypatch):
    """Provide msgpack/zmq fakes without requiring the Isaac runtime."""

    msgpack = types.ModuleType("msgpack")
    def packb(value, default=None, **_kwargs):
        def visit(item):
            if isinstance(item, dict):
                return {key: visit(value) for key, value in item.items()}
            if isinstance(item, list):
                return [visit(entry) for entry in item]
            if isinstance(item, np.ndarray):
                return default(item)
            return item
        return pickle.dumps(visit(value))
    msgpack.packb = packb
    def unpackb(value, object_hook=None, **_kwargs):
        def visit(item):
            if isinstance(item, dict):
                item = {key: visit(value) for key, value in item.items()}
                return object_hook(item) if object_hook else item
            if isinstance(item, list):
                return [visit(entry) for entry in item]
            return item
        return visit(pickle.loads(value))
    msgpack.unpackb = unpackb
    zmq = types.ModuleType("zmq")
    zmq.REQ = 1
    zmq.LINGER = 2
    zmq.SNDTIMEO = 3
    zmq.RCVTIMEO = 4
    zmq.Again = TimeoutError
    monkeypatch.setitem(sys.modules, "msgpack", msgpack)
    monkeypatch.setitem(sys.modules, "zmq", zmq)


def test_policy_server_wire_round_trip_encodes_numpy_as_npy_without_pickle(monkeypatch):
    _install_wire_fakes(monkeypatch)
    module = _load_policy_module()
    observation = {"state": np.arange(12, dtype=np.float32)}

    packed = module.pack_policy_server_message(observation)
    encoded = pickle.loads(packed)
    assert encoded["state"]["__ndarray_class__"] is True
    assert isinstance(encoded["state"]["as_npy"], bytes)

    decoded = module.unpack_policy_server_message(packed)
    assert decoded["state"].dtype == np.float32
    np.testing.assert_array_equal(decoded["state"], observation["state"])


def test_policy_server_client_recreates_req_socket_after_timeout_before_retry(monkeypatch):
    _install_wire_fakes(monkeypatch)
    module = _load_policy_module()

    class Socket:
        def __init__(self, outcome):
            self.outcome = outcome
            self.closed = False
            self.connected = []

        def setsockopt(self, *_args): pass
        def connect(self, endpoint): self.connected.append(endpoint)
        def send(self, _payload):
            if self.outcome == "timeout":
                raise TimeoutError("timed out")
        def recv(self): return module.pack_policy_server_message({"status": "ok", "message": "Server is running"})
        def close(self, *_args): self.closed = True

    sockets = [Socket("timeout"), Socket("ok")]
    client = module.PolicyServerClient(
        "tcp://127.0.0.1:5000", "token", 1.0,
        socket_factory=lambda: sockets.pop(0),
    )
    assert client.ping() is None
    assert sockets == []


def test_policy_server_client_sends_official_action_and_reset_envelopes(monkeypatch):
    _install_wire_fakes(monkeypatch)
    module = _load_policy_module()

    class Socket:
        def __init__(self): self.sent = []
        def setsockopt(self, *_args): pass
        def connect(self, _endpoint): pass
        def send(self, payload): self.sent.append(module.unpack_policy_server_message(payload))
        def recv(self): return module.pack_policy_server_message({})
        def close(self, *_args, **_kwargs): pass

    socket = Socket()
    client = module.PolicyServerClient("tcp://127.0.0.1:5000", "bound-token", 1.0, socket_factory=lambda: socket)
    client.reset()
    assert socket.sent == [{"endpoint": "reset", "data": {"options": None}, "api_token": "bound-token"}]


def test_server_policy_validates_reset_and_action_response_contract(monkeypatch):
    _install_wire_fakes(monkeypatch)
    module = _load_policy_module()

    action = {
        "left_arm": _chunk(1, 16, 5, 0),
        "left_gripper": _chunk(1, 16, 1, 10),
        "right_arm": _chunk(1, 16, 5, 20),
        "right_gripper": _chunk(1, 16, 1, 30),
    }

    class Client:
        def reset(self):
            assert self.request("reset", {"options": None}) == {}

        def get_action(self, data):
            response = self.request("get_action", data)
            assert isinstance(response, list) and len(response) == 2
            return response[0], response[1]

        def request(self, endpoint, data):
            if endpoint == "reset":
                assert data == {"options": None}
            else:
                assert endpoint == "get_action"
                assert isinstance(data, dict)
            return {} if endpoint == "reset" else [action, {"server": "ok"}]

    policy = module.GrootServerPolicy.__new__(module.GrootServerPolicy)
    policy._client = Client()
    policy._action_queue = module.ActionChunkQueue()
    policy.reset()
    returned = policy.select_action_with_provenance({
        "observation.state": np.zeros(12, dtype=np.float32),
        **{f"observation.images.{camera}": np.zeros((2, 2, 3), dtype=np.uint8) for camera in ("top_rgb", "left_rgb", "right_rgb")},
    })
    assert returned.value.shape == (12,)
    assert returned.chunk_offset == 0

    malformed = dict(action)
    malformed["left_arm"] = malformed["left_arm"].astype(np.float64)
    with pytest.raises(ValueError, match="dtype or shape"):
        module.validate_policy_server_action_chunk(malformed)
    malformed = dict(action)
    malformed["action.unexpected"] = np.zeros((1, 16, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="exactly"):
        module.validate_policy_server_action_chunk(malformed)

    policy._client = types.SimpleNamespace(get_action=lambda *_args: (_ for _ in ()).throw(RuntimeError("policy server error: invalid token")))
    policy._action_queue.clear()
    with pytest.raises(RuntimeError, match="server error"):
        policy.select_action_with_provenance({
            "observation.state": np.zeros(12, dtype=np.float32),
            **{f"observation.images.{camera}": np.zeros((2, 2, 3), dtype=np.uint8) for camera in ("top_rgb", "left_rgb", "right_rgb")},
        })
