from __future__ import annotations

import importlib
import sys
import types

import numpy as np


def _load_policy_module():
    """Load the adapter without importing Isaac/LeRobot dependencies."""

    package = types.ModuleType("scripts.eval_policy")
    package.__path__ = ["scripts/eval_policy"]
    sys.modules.setdefault("scripts", types.ModuleType("scripts"))
    sys.modules["scripts.eval_policy"] = package
    for name in ("scripts.eval_policy.base_policy", "scripts.eval_policy.registry"):
        sys.modules.pop(name, None)
    sys.modules.pop("scripts.eval_policy.groot_policy", None)
    return importlib.import_module("scripts.eval_policy.groot_policy")


def _chunk(batch: int, horizon: int, dimension: int, start: float) -> np.ndarray:
    values = np.arange(batch * horizon * dimension, dtype=np.float32) + start
    return values.reshape(batch, horizon, dimension)


def test_flatten_groot_action_chunk_preserves_all_horizon_steps_and_group_order():
    module = _load_policy_module()
    action = {
        "action.left_arm": _chunk(1, 2, 5, 0),
        "action.left_gripper": _chunk(1, 2, 1, 10),
        "action.right_arm": _chunk(1, 2, 5, 20),
        "action.right_gripper": _chunk(1, 2, 1, 30),
    }

    flattened = module.flatten_groot_action_chunk(action)

    assert flattened.shape == (2, 12)
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

    queue.extend(chunk)
    queue.clear()
    assert queue.pop() is None
