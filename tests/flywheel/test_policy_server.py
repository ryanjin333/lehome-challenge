from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys
import types

import numpy as np
import pytest

_REPOSITORY_ROOT = Path(__file__).parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

import scripts.run_groot_policy_server as policy_server


def _load_groot_policy_module():
    package = types.ModuleType("scripts.eval_policy")
    package.__path__ = [str(Path(__file__).parents[2] / "scripts" / "eval_policy")]
    sys.modules["scripts.eval_policy"] = package
    for name in ("scripts.eval_policy.base_policy", "scripts.eval_policy.registry"):
        sys.modules.pop(name, None)
    sys.modules.pop("scripts.eval_policy.groot_policy", None)
    return importlib.import_module("scripts.eval_policy.groot_policy")


@pytest.fixture(scope="module")
def validate_policy_server_action_chunk():
    monkeypatch = pytest.MonkeyPatch()
    package = types.ModuleType("scripts.eval_policy")
    package.__path__ = [str(Path(__file__).parents[2] / "scripts" / "eval_policy")]
    monkeypatch.setitem(sys.modules, "scripts.eval_policy", package)
    registry = importlib.import_module("scripts.eval_policy.registry")
    registry.PolicyRegistry._registry.pop("groot", None)
    registry.PolicyRegistry._registry.pop("groot_server", None)
    monkeypatch.delitem(sys.modules, "scripts.eval_policy.groot_policy", raising=False)
    module = importlib.import_module("scripts.eval_policy.groot_policy")
    try:
        yield module.validate_policy_server_action_chunk
    finally:
        module.PolicyRegistry._registry.pop("groot", None)
        module.PolicyRegistry._registry.pop("groot_server", None)
        monkeypatch.undo()


def test_policy_runtime_seed_covers_python_numpy_and_every_visible_cuda_device(monkeypatch) -> None:
    events: list[tuple[str, int]] = []
    numpy = types.ModuleType("numpy")
    numpy.random = types.SimpleNamespace(seed=lambda value: events.append(("numpy", value)))
    torch = types.ModuleType("torch")
    torch.manual_seed = lambda value: events.append(("torch", value))
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        manual_seed_all=lambda value: events.append(("cuda", value)),
    )
    monkeypatch.setitem(sys.modules, "numpy", numpy)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(policy_server.random, "seed", lambda value: events.append(("python", value)))

    policy_server.seed_policy_runtime(42)

    assert events == [("python", 42), ("numpy", 42), ("torch", 42), ("cuda", 42)]


def test_policy_server_action_chunk_requires_the_pinned_16_step_horizon(
    validate_policy_server_action_chunk,
) -> None:
    dimensions = {
        "left_arm": 5,
        "left_gripper": 1,
        "right_arm": 5,
        "right_gripper": 1,
    }
    action = {
        group: np.zeros((1, 16, dimension), dtype=np.float32)
        for group, dimension in dimensions.items()
    }

    chunk = validate_policy_server_action_chunk(action)

    assert chunk.shape == (16, 12)
    with pytest.raises(ValueError, match="dtype or shape"):
        validate_policy_server_action_chunk(
            {
                group: np.zeros((1, 40, dimension), dtype=np.float32)
                for group, dimension in dimensions.items()
            }
        )


def test_policy_server_uses_pinned_run_lifecycle_without_context_manager(monkeypatch, tmp_path) -> None:
    events: list[object] = []

    class FakePolicy:
        def __init__(self, **kwargs) -> None:
            events.append(("policy", kwargs))

    class FakeServer:
        def __init__(self, policy, *, host: str, port: int, api_token: str) -> None:
            events.append(("server", policy, host, port, api_token))

        def run(self) -> None:
            events.append("run")

    gr00t = types.ModuleType("gr00t")
    gr00t.__path__ = []
    data = types.ModuleType("gr00t.data")
    data.__path__ = []
    embodiment_tags = types.ModuleType("gr00t.data.embodiment_tags")
    embodiment_tags.EmbodimentTag = types.SimpleNamespace(NEW_EMBODIMENT="new_embodiment")
    policy = types.ModuleType("gr00t.policy")
    policy.__path__ = []
    policy.Gr00tPolicy = FakePolicy
    server_client = types.ModuleType("gr00t.policy.server_client")
    server_client.PolicyServer = FakeServer
    for name, module in {
        "gr00t": gr00t,
        "gr00t.data": data,
        "gr00t.data.embodiment_tags": embodiment_tags,
        "gr00t.policy": policy,
        "gr00t.policy.server_client": server_client,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setattr(policy_server, "unblock_termination_signals", lambda: events.append("unblock"))
    monkeypatch.setattr(policy_server, "seed_policy_runtime", lambda seed: events.append(("seed", seed)))
    monkeypatch.setenv("POLICY_SERVER_TOKEN", "t" * 32)
    model_path = tmp_path / "model"
    model_path.mkdir()

    assert policy_server.run(
        argparse.Namespace(
            model_path=model_path,
            host="127.0.0.1",
            port=5555,
            api_token_env="POLICY_SERVER_TOKEN",
            device="cuda:0",
            seed=42,
        )
    ) == 0
    assert events[0] == "unblock"
    assert events[1] == ("seed", 42)
    assert events[2] == (
        "policy",
        {
            "embodiment_tag": "new_embodiment",
            "model_path": str(model_path),
            "device": "cuda:0",
            "strict": True,
        },
    )
    assert events[3][0] == "server"
    assert isinstance(events[3][1], FakePolicy)
    assert events[3][2:] == ("127.0.0.1", 5555, "t" * 32)
    assert events[4:] == ["run"]


def test_session_client_caches_one_horizon_16_chunk_locally_and_reset_clears_it() -> None:
    from lehome.flywheel import policy_protocol as protocol

    module = _load_groot_policy_module()
    requests = []
    guard = protocol.SessionRequestGuard(policy_sha256="a" * 64)
    chunk = np.arange(16 * 12, dtype=np.float32).reshape(16, 12)

    def transport(payload: bytes) -> bytes:
        request = protocol.unpack_envelope(payload)
        assert isinstance(request, protocol.PolicyRequest)
        guard.accept(request, now_ns=1_000)
        requests.append(request)
        if request.operation == "infer":
            return protocol.pack_envelope(
                protocol.PolicyResponse.ok(
                    request, action_chunk=chunk.tobytes(), action_horizon=16
                )
            )
        return protocol.pack_envelope(protocol.PolicyResponse.ok(request))

    client = module.SessionPolicyClient(
        "tcp://127.0.0.1:5500",
        "a" * 64,
        1.0,
        session_id="worker-0",
        request_transport=transport,
        now_ns=lambda: 1_000,
    )
    observation = {"state": [0.0] * 12}

    returned = [client.select_action_with_provenance(observation) for _ in range(16)]
    assert [item.chunk_offset for item in returned] == list(range(16))
    assert len([request for request in requests if request.operation == "infer"]) == 1

    client.reset()
    after_reset = client.select_action_with_provenance(observation)
    assert after_reset.chunk_offset == 0
    assert len([request for request in requests if request.operation == "infer"]) == 2
    assert len([request for request in requests if request.operation == "reset"]) == 2
    assert [request.request_id for request in requests] == [
        f"worker-0:{index:020d}" for index in range(1, 5)
    ]


def test_session_client_converts_lehome_observation_before_infer() -> None:
    from lehome.flywheel import policy_protocol as protocol

    module = _load_groot_policy_module()
    chunk = np.zeros((16, 12), dtype=np.float32)
    requests = []

    def transport(payload: bytes) -> bytes:
        request = protocol.unpack_envelope(payload)
        assert isinstance(request, protocol.PolicyRequest)
        requests.append(request)
        if request.operation == "infer":
            return protocol.pack_envelope(
                protocol.PolicyResponse.ok(
                    request, action_chunk=chunk.tobytes(), action_horizon=16
                )
            )
        return protocol.pack_envelope(protocol.PolicyResponse.ok(request))

    client = module.SessionPolicyClient(
        "tcp://127.0.0.1:5500",
        "a" * 64,
        1.0,
        session_id="worker-0",
        request_transport=transport,
        now_ns=lambda: 1_000,
    )
    observation = {
        "observation.state": np.zeros(12, dtype=np.float32),
        "observation.images.top_rgb": np.zeros((4, 6, 3), dtype=np.uint8),
        "observation.images.left_rgb": np.zeros((4, 6, 3), dtype=np.uint8),
        "observation.images.right_rgb": np.zeros((4, 6, 3), dtype=np.uint8),
    }
    action = client.select_action_with_provenance(observation)
    assert action.value.shape == (12,)
    infer = next(request for request in requests if request.operation == "infer")
    assert set(infer.observation) == {"video", "state", "language"}
    assert set(infer.observation["video"]) == {"top_rgb", "left_rgb", "right_rgb"}
    assert infer.observation["video"]["top_rgb"].shape == (1, 1, 4, 6, 3)


def test_session_client_rejects_stale_or_expired_responses_and_replays_reset_after_gateway_restart() -> None:
    from lehome.flywheel import policy_protocol as protocol

    module = _load_groot_policy_module()
    chunk = np.zeros((16, 12), dtype=np.float32).tobytes()
    initial_gateway = protocol.SessionRequestGuard(policy_sha256="a" * 64)
    restarted_gateway = protocol.SessionRequestGuard(policy_sha256="a" * 64)
    requests = []
    restarted = False

    def transport(payload: bytes) -> bytes:
        nonlocal restarted
        request = protocol.unpack_envelope(payload)
        assert isinstance(request, protocol.PolicyRequest)
        requests.append(request)
        gateway = restarted_gateway if restarted else initial_gateway
        try:
            gateway.accept(request, now_ns=1_000)
        except protocol.SessionStateError:
            return protocol.pack_envelope(protocol.PolicyResponse.error(request, error_code="unknown_session"))
        if request.operation == "reset":
            return protocol.pack_envelope(protocol.PolicyResponse.ok(request))
        if not restarted:
            restarted = True
            return protocol.pack_envelope(protocol.PolicyResponse.error(request, error_code="unknown_session"))
        return protocol.pack_envelope(
            protocol.PolicyResponse.ok(request, action_chunk=chunk, action_horizon=16)
        )

    client = module.SessionPolicyClient(
        "tcp://127.0.0.1:5500", "a" * 64, 1.0,
        session_id="worker-0", request_transport=transport, now_ns=lambda: 1_000,
    )
    action = client.select_action_with_provenance({"state": [0.0] * 12})
    assert action.value.shape == (12,)
    assert [request.operation for request in requests] == ["reset", "infer", "reset", "infer"]
    assert requests[1].request_id == requests[3].request_id

    stale_request = protocol.PolicyRequest.infer(
        session_id="worker-1", episode_generation=1, request_id="infer-1",
        policy_sha256="a" * 64, deadline_ns=2_000, observation={"state": [0.0] * 12},
    )
    stale_response = protocol.PolicyResponse.ok(stale_request, action_chunk=chunk, action_horizon=16)
    with pytest.raises(protocol.StaleResponseError):
        protocol.validate_response_for_request(
            stale_response,
            protocol.PolicyRequest.infer(
                session_id="worker-1", episode_generation=2, request_id="infer-1",
                policy_sha256="a" * 64, deadline_ns=2_000, observation={"state": [0.0] * 12},
            ),
            now_ns=1_000,
        )
    with pytest.raises(protocol.ExpiredRequestError):
        protocol.validate_response_for_request(stale_response, stale_request, now_ns=2_000)


def test_session_client_recovers_when_reset_was_accepted_but_its_reply_timed_out() -> None:
    from lehome.flywheel import policy_protocol as protocol

    module = _load_groot_policy_module()
    guard = protocol.SessionRequestGuard(policy_sha256="a" * 64)
    requests = []
    reset_reply_lost = False
    chunk = np.zeros((16, 12), dtype=np.float32).tobytes()

    def transport(payload: bytes) -> bytes:
        nonlocal reset_reply_lost
        request = protocol.unpack_envelope(payload)
        assert isinstance(request, protocol.PolicyRequest)
        guard.accept(request, now_ns=1_000)
        requests.append(request)
        if request.operation == "reset" and not reset_reply_lost:
            reset_reply_lost = True
            raise TimeoutError("reply lost after gateway accepted reset")
        if request.operation == "infer":
            return protocol.pack_envelope(
                protocol.PolicyResponse.ok(request, action_chunk=chunk, action_horizon=16)
            )
        return protocol.pack_envelope(protocol.PolicyResponse.ok(request))

    client = module.SessionPolicyClient(
        "tcp://127.0.0.1:5500", "a" * 64, 1.0,
        session_id="worker-0", request_transport=transport, now_ns=lambda: 1_000,
    )
    with pytest.raises(TimeoutError, match="reply lost"):
        client.reset()

    action = client.select_action_with_provenance({"state": [0.0] * 12})
    assert action.value.shape == (12,)
    assert [(request.operation, request.episode_generation) for request in requests] == [
        ("reset", 1), ("reset", 1), ("infer", 1),
    ]


def test_session_client_discards_dealer_socket_after_timeout_before_late_reply_can_poison_retry(monkeypatch) -> None:
    from lehome.flywheel import policy_protocol as protocol

    module = _load_groot_policy_module()
    fake_zmq = types.SimpleNamespace(DEALER=1, LINGER=2, SNDTIMEO=3, RCVTIMEO=4, IDENTITY=5)
    monkeypatch.setitem(sys.modules, "zmq", fake_zmq)
    chunk = np.zeros((16, 12), dtype=np.float32).tobytes()

    class Socket:
        def __init__(self, *, timeout_first_inference: bool) -> None:
            self.timeout_first_inference = timeout_first_inference
            self.sent = []
            self.pending = []
            self.closed = False

        def setsockopt(self, *_args) -> None:
            pass

        def connect(self, _endpoint) -> None:
            pass

        def send(self, payload: bytes) -> None:
            request = protocol.unpack_envelope(payload)
            assert isinstance(request, protocol.PolicyRequest)
            self.sent.append(request)
            if request.operation == "infer":
                self.pending.append(
                    protocol.pack_envelope(
                        protocol.PolicyResponse.ok(request, action_chunk=chunk, action_horizon=16)
                    )
                )
            else:
                self.pending.append(protocol.pack_envelope(protocol.PolicyResponse.ok(request)))

        def recv(self) -> bytes:
            if self.timeout_first_inference and self.sent[-1].operation == "infer":
                self.timeout_first_inference = False
                raise TimeoutError("first response timed out but remains queued")
            return self.pending.pop(0)

        def close(self, *_args, **_kwargs) -> None:
            self.closed = True

    poisoned = Socket(timeout_first_inference=True)
    replacement = Socket(timeout_first_inference=False)
    sockets = [poisoned, replacement]
    client = module.SessionPolicyClient(
        "tcp://127.0.0.1:5500", "a" * 64, 1.0,
        session_id="worker-0", socket_factory=lambda: sockets.pop(0), now_ns=lambda: 1_000,
    )

    with pytest.raises(TimeoutError, match="remains queued"):
        client.select_action_with_provenance({"state": [0.0] * 12})
    next_action = client.select_action_with_provenance({"state": [0.0] * 12})

    assert poisoned.closed is True
    assert len(poisoned.pending) == 1  # The late first reply was never read as the retry response.
    assert next_action.value.shape == (12,)
    assert [request.operation for request in replacement.sent] == ["infer"]
