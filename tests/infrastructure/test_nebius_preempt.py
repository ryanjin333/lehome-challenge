"""Bounded 60-second preemption shutdown contract with injected clocks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

GUEST_DIR = Path(__file__).resolve().parents[2] / "infrastructure" / "nebius" / "guest"
sys.path.insert(0, str(GUEST_DIR))

from lehome_preempt import (  # noqa: E402
    PREEMPTION_BUDGET_SECONDS,
    CommandResult,
    PreemptionError,
    PreemptionHooks,
    handle_preemption,
)


class FakeClock:
    def __init__(self, times):
        self._times = list(times)

    def __call__(self):
        return self._times.pop(0)


def ok_hook(name):
    return lambda: {name: "done"}


def make_hooks(**overrides):
    base = dict(
        stop_leases=ok_hook("stop_leases"),
        mark_interrupted=ok_hook("mark_interrupted"),
        flush_ledgers=ok_hook("flush_ledgers"),
        close_terminal_artifacts=ok_hook("close_terminal_artifacts"),
    )
    base.update(overrides)
    return PreemptionHooks(**base)


def test_full_shutdown_sequence_within_budget(tmp_path):
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    times = [0.0] + [1.0] * 30 + [5.0] * 10
    result = handle_preemption(
        receipts_dir=receipts,
        hooks=make_hooks(),
        role="rollout",
        run_id="run-1",
        clock=FakeClock(times),
    )
    assert result.completed_steps == (
        "stop_leases", "mark_interrupted", "flush_ledgers", "close_terminal_artifacts",
    )
    assert not result.checkpoint_requested
    receipt = json.loads(result.receipt_path.read_text())
    assert receipt["kind"] == "preemption_shutdown"
    assert receipt["role"] == "rollout"
    assert receipt["completed_steps"] == list(result.completed_steps)
    assert receipt["errors"] == []
    assert list(receipts.iterdir()) == [result.receipt_path], "exactly one receipt, no temp files"


def test_training_checkpoint_requested_only_when_budget_remains(tmp_path):
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    checkpoint_calls = []

    def request_checkpoint(remaining):
        checkpoint_calls.append(remaining)
        return {"checkpoint": "step-1500", "remaining": remaining}

    times = [0.0] + [1.0] * 30
    result = handle_preemption(
        receipts_dir=receipts,
        hooks=make_hooks(request_training_checkpoint=request_checkpoint),
        role="training",
        run_id="run-1",
        clock=FakeClock(times),
    )
    assert result.checkpoint_requested
    assert "training_checkpoint" in result.completed_steps
    assert checkpoint_calls and checkpoint_calls[0] > 0


def test_budget_exhaustion_skips_later_steps_but_writes_receipt(tmp_path):
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    # Budget expires immediately after the first step.
    times = [0.0, 1.0, 61.0, 61.0, 61.0, 61.0, 61.0]
    result = handle_preemption(
        receipts_dir=receipts,
        hooks=make_hooks(),
        role="rollout",
        run_id="run-1",
        clock=FakeClock(times),
        budget_seconds=PREEMPTION_BUDGET_SECONDS,
    )
    assert result.completed_steps == ("stop_leases",)
    receipt = json.loads(result.receipt_path.read_text())
    assert "skipped, budget exhausted" in receipt["errors"][0]


def test_hub_sync_is_deadline_bounded(tmp_path):
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    sync_commands = []

    def fake_sync(command):
        sync_commands.append(tuple(command))
        return CommandResult(0, "uploaded", "")

    times = [0.0] + [1.0] * 30
    result = handle_preemption(
        receipts_dir=receipts,
        hooks=make_hooks(bounded_hub_sync=fake_sync),
        role="rollout",
        run_id="run-1",
        clock=FakeClock(times),
        hub_sync_command=["hf-sync", "--workspace", "/mnt/lehome"],
    )
    assert "bounded_hub_sync" in result.completed_steps
    assert result.hub_sync["exit_code"] == 0
    assert sync_commands[0][:2] == ("hf-sync", "--workspace")
    assert sync_commands[0][-2] == "--deadline-seconds"
    assert float(sync_commands[0][-1]) <= PREEMPTION_BUDGET_SECONDS


def test_hook_failure_recorded_and_shutdown_continues(tmp_path):
    receipts = tmp_path / "receipts"
    receipts.mkdir()

    def broken_flush():
        raise OSError("disk gone")

    times = [0.0] + [1.0] * 30
    result = handle_preemption(
        receipts_dir=receipts,
        hooks=make_hooks(flush_ledgers=broken_flush),
        role="rollout",
        run_id="run-1",
        clock=FakeClock(times),
    )
    assert "flush_ledgers" not in result.completed_steps
    assert "close_terminal_artifacts" in result.completed_steps
    assert any("flush_ledgers: disk gone" in error for error in result.receipt["errors"])
    assert result.receipt_path.exists(), "receipt must be written even after hook failure"


def test_receipt_not_overwritten_on_second_preemption(tmp_path):
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    times = [0.0] + [1.0] * 30
    first = handle_preemption(
        receipts_dir=receipts, hooks=make_hooks(), role="rollout", run_id="run-1", clock=FakeClock(list(times)),
    )
    second = handle_preemption(
        receipts_dir=receipts, hooks=make_hooks(), role="rollout", run_id="run-1", clock=FakeClock(list(times)),
    )
    assert first.receipt_path != second.receipt_path
    assert first.receipt_path.exists() and second.receipt_path.exists()


def test_rejects_unknown_role_and_bad_budget(tmp_path):
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    times = [0.0] * 20
    with pytest.raises(PreemptionError, match="role"):
        handle_preemption(
            receipts_dir=receipts, hooks=make_hooks(), role="eval", run_id="r", clock=FakeClock(list(times)),
        )
    with pytest.raises(PreemptionError, match="budget_seconds"):
        handle_preemption(
            receipts_dir=receipts, hooks=make_hooks(), role="rollout", run_id="r",
            clock=FakeClock(list(times)), budget_seconds=0,
        )
    with pytest.raises(PreemptionError, match="receipts directory"):
        handle_preemption(
            receipts_dir=tmp_path / "missing", hooks=make_hooks(), role="rollout", run_id="r",
            clock=FakeClock(list(times)),
        )


def test_checkpoint_hook_ignored_for_rollout_role(tmp_path):
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    calls = []
    times = [0.0] + [1.0] * 30
    result = handle_preemption(
        receipts_dir=receipts,
        hooks=make_hooks(request_training_checkpoint=lambda remaining: calls.append(remaining) or {}),
        role="rollout",
        run_id="run-1",
        clock=FakeClock(times),
    )
    assert not result.checkpoint_requested
    assert calls == []
