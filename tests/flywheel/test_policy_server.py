from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys
import types

import numpy as np
import pytest

import scripts.run_groot_policy_server as policy_server


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
