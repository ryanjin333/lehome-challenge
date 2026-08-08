from __future__ import annotations

import importlib
from pathlib import Path
import sys
import types

import numpy as np
import pytest


@pytest.fixture(scope="module")
def validator():
    monkeypatch = pytest.MonkeyPatch()
    package = types.ModuleType("scripts.eval_policy")
    package.__path__ = [str(Path(__file__).parents[2] / "scripts" / "eval_policy")]
    monkeypatch.setitem(sys.modules, "scripts.eval_policy", package)
    monkeypatch.delitem(sys.modules, "scripts.eval_policy.groot_policy", raising=False)
    module = importlib.import_module("scripts.eval_policy.groot_policy")
    try:
        yield module.validate_policy_server_action_chunk
    finally:
        module.PolicyRegistry._registry.pop("groot", None)
        module.PolicyRegistry._registry.pop("groot_server", None)
        monkeypatch.undo()


def _official_server_action(*, horizon: int = 16) -> dict[str, np.ndarray]:
    return {
        "left_arm": np.zeros((1, horizon, 5), dtype=np.float32),
        "left_gripper": np.zeros((1, horizon, 1), dtype=np.float32),
        "right_arm": np.zeros((1, horizon, 5), dtype=np.float32),
        "right_gripper": np.zeros((1, horizon, 1), dtype=np.float32),
    }


def test_policy_server_accepts_the_raw_official_policy_action_namespace(validator) -> None:
    chunk = validator(_official_server_action())

    assert chunk.shape == (16, 12)
    assert chunk.dtype == np.float32


def test_policy_server_rejects_the_sim_wrapper_action_namespace(validator) -> None:
    prefixed = {f"action.{key}": value for key, value in _official_server_action().items()}

    with pytest.raises(ValueError, match="keys.*contract"):
        validator(prefixed)


def test_policy_server_rejects_a_different_decoded_horizon(validator) -> None:
    with pytest.raises(ValueError, match="dtype or shape"):
        validator(_official_server_action(horizon=40))
