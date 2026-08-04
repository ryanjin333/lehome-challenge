from __future__ import annotations

import argparse
import sys
import types

import scripts.run_groot_policy_server as policy_server


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
    monkeypatch.setenv("POLICY_SERVER_TOKEN", "t" * 32)
    model_path = tmp_path / "model"
    model_path.mkdir()

    assert policy_server.run(
        argparse.Namespace(
            model_path=model_path,
            host="127.0.0.1",
            port=5555,
            api_token_env="POLICY_SERVER_TOKEN",
        )
    ) == 0
    assert events[0] == "unblock"
    assert events[1] == (
        "policy",
        {
            "embodiment_tag": "new_embodiment",
            "model_path": str(model_path),
            "device": "cuda:0",
            "strict": True,
        },
    )
    assert events[2][0] == "server"
    assert isinstance(events[2][1], FakePolicy)
    assert events[2][2:] == ("127.0.0.1", 5555, "t" * 32)
    assert events[3:] == ["run"]
