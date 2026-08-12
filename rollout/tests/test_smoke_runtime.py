from __future__ import annotations

import pytest

from b1k_rollout import smoke_runtime


def test_runtime_smoke_counts_only_real_local_policy_action_tensors() -> None:
    class Action:
        def numel(self) -> int:
            return 7
    assert smoke_runtime._action_dimension(Action()) == 7
    assert smoke_runtime._action_dimension({"robot": Action()}) == 7
    assert smoke_runtime._action_dimension({}) == 0
    assert smoke_runtime._action_dimension("forged") == 0


def test_runtime_smoke_accepts_one_executed_nonterminal_evaluator_step() -> None:
    assert smoke_runtime._evaluator_outcome(False, False) == "advanced"
    assert smoke_runtime._evaluator_outcome(True, False) == "terminal"
    assert smoke_runtime._evaluator_outcome(False, True) == "terminal"


@pytest.mark.parametrize("value", ["b1k-bootstrap-ok/escape", "b1k-bootstrap-" + "a" * 32 + "-smoke-model", "b1k-bootstrap-" + "A" * 32 + "-success-fixture", "wrong"])
def test_runtime_smoke_rejects_unsafe_remote_probe_prefixes(value: str) -> None:
    with pytest.raises(RuntimeError):
        smoke_runtime._prefix(value)
