"""Bounded 60-second preemption shutdown contract with injected clocks."""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import pytest

GUEST_DIR = Path(__file__).resolve().parents[2] / "infrastructure" / "nebius" / "guest"
SOURCE_DIR = Path(__file__).resolve().parents[2] / "source" / "lehome"
sys.path.insert(0, str(GUEST_DIR))
sys.path.insert(0, str(SOURCE_DIR))

import lehome_preempt  # noqa: E402
from lehome_preempt import (  # noqa: E402
    PREEMPTION_BUDGET_SECONDS,
    CommandResult,
    PreemptionError,
    PreemptionHooks,
    RolloutPreemptionContext,
    build_rollout_preemption_hooks,
    handle_preemption,
    main,
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


def test_training_preempt_main_refuses_a_failed_process_stop_without_receipt(tmp_path):
    receipts = tmp_path / "receipts"
    with pytest.raises(PreemptionError, match="trainer stop was not confirmed"):
        main([
            "--role", "training", "--run-id", "run-1", "--receipts-dir", str(receipts),
            "--training-stop-status", "failed",
        ])
    assert not receipts.exists()


def test_training_preempt_main_records_confirmed_stop_and_explicit_non_applicable_steps(tmp_path):
    receipts = tmp_path / "receipts"
    assert main([
        "--role", "training", "--run-id", "run-1", "--receipts-dir", str(receipts),
        "--training-stop-status", "stopped",
    ]) == 0
    receipt = json.loads((receipts / "preemption-receipt.json").read_text(encoding="utf-8"))
    assert receipt["completed_steps"] == [
        "stop_leases", "mark_interrupted", "flush_ledgers", "close_terminal_artifacts",
    ]
    assert receipt["steps"]["stop_leases"] == {"verified": True, "training_process": "stopped"}
    assert receipt["steps"]["mark_interrupted"] == {"applicable": False, "role": "training"}
    assert receipt["steps"]["flush_ledgers"] == {"applicable": False, "role": "training"}
    assert receipt["steps"]["close_terminal_artifacts"] == {"applicable": False, "role": "training"}
    assert receipt["errors"] == []


def test_rollout_preemption_hooks_pause_retry_flush_and_finalize_real_ledger(tmp_path):
    from lehome.flywheel.task_ledger import TaskLedger

    run_root = tmp_path / "workspace" / "campaign"
    run_root.mkdir(parents=True)
    matrix = [
        {"attempt_id": "attempt-a", "garment": "Top_Long_Seen_0", "seed": 1},
        {"attempt_id": "attempt-b", "garment": "Top_Short_Seen_0", "seed": 2},
    ]
    matrix_path = run_root / "matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    matrix_sha256 = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    database = run_root / "ledger.sqlite3"
    ledger = TaskLedger(database, attempt_matrix=matrix, max_attempts=2, target_accepted=2)
    try:
        first = ledger.lease_next("worker-1", lease_duration_ns=30_000_000_000)
        second = ledger.lease_next("worker-2", lease_duration_ns=30_000_000_000)
        assert first is not None and second is not None
    finally:
        ledger.close()

    finalized = []
    context = RolloutPreemptionContext(
        run_id="run-1", run_root=run_root, database=database, attempt_matrix=matrix_path,
        attempt_matrix_sha256=matrix_sha256, max_attempts=2, target_accepted=2,
    )
    hooks = build_rollout_preemption_hooks(
        context,
        finalizer=lambda **kwargs: finalized.append(kwargs) or 0,
    )
    stopped = hooks.stop_leases()
    interrupted = hooks.mark_interrupted()
    flushed = hooks.flush_ledgers()
    closed = hooks.close_terminal_artifacts()

    assert stopped == {"verified": True, "campaign_paused": True, "active_leases_interrupted": 2}
    assert interrupted == {"verified": True, "retryable_preemption_attempts": 2}
    assert flushed["verified"] is True and flushed["wal_checkpoint_busy"] == 0
    assert closed["verified"] is True and closed["finalized"] == 0
    assert finalized and finalized[0]["database"] == database

    reopened = TaskLedger(database, attempt_matrix=matrix, max_attempts=2, target_accepted=2)
    try:
        assert reopened.is_stopped
        assert reopened.status(first.attempt.attempt_id) == "retryable"
        assert reopened.status(second.attempt.attempt_id) == "retryable"
        events = reopened.events()
        assert sum(event.event_type == "campaign_paused" for event in events) == 1
        assert sum(event.event_type == "interrupted" for event in events) == 2
    finally:
        reopened.close()


def test_rollout_preemption_reuses_controlled_materialization_rows_without_changing_attempt_ids(tmp_path):
    from lehome.flywheel.task_ledger import TaskLedger

    run_root = tmp_path / "workspace" / "campaign"; run_root.mkdir(parents=True)
    reset, annotations, continuation = run_root / "reset.json", run_root / "annotations.jsonl", run_root / "continuation.json"
    reset.write_text("{}", encoding="utf-8"); annotations.write_text("", encoding="utf-8"); continuation.write_text("{}", encoding="utf-8")
    caps = {"pant_long": 4, "top_long": 1, "top_short": 3, "pant_short": 0}
    categories = ["pant_long"] * 4 + ["top_long"] + ["top_short"] * 3
    rows = [
        {
            "attempt_id": f"controlled-{index}", "trial_id": f"controlled-{index}",
            "category": category, "category_acceptance_cap": caps[category],
            "strategy": "canonical", "recovery_kind": "controlled_success_recovery_snapshot_v2",
            "controlled_matrix_sha256": "a" * 64, "perturbation_seed": 71_000 + index,
            "perturbation_fingerprint": f"{index + 100:064x}",
            "source_state_perturbation_fingerprint": f"{index + 200:064x}",
            "source_seed": 50110, "source_continuation_state": [0.0] * 12,
            "source_state_fingerprint": f"{index + 300:064x}", "source_round_id": "round",
            "source_episode_id": f"episode-{index}", "source_episode_digest": f"{index + 400:064x}",
            "source_reset_sha256": "a" * 64, "source_annotations_sha256": "b" * 64,
            "source_continuation_snapshot_sha256": "c" * 64, "prefix_stop": 16,
            "source_first_success_step": 19, "source_reset": str(reset),
            "source_annotations": str(annotations), "source_continuation_snapshot": str(continuation),
        }
        for index, category in enumerate(categories)
    ]
    matrix_path = run_root / "materialization.json"
    matrix_path.write_text(json.dumps({"schema_version": 2, "kind": "controlled_success_recovery_materialization_v2", "matrix_sha256": "a" * 64, "target_accepted": 8, "category_acceptance_caps": caps, "rows": rows}), encoding="utf-8")
    database = run_root / "ledger.sqlite3"
    ledger = TaskLedger(database, attempt_matrix=rows, max_attempts=8, target_accepted=8)
    original_ids = [attempt.attempt_id for attempt in ledger.attempts()]; ledger.close()
    context = RolloutPreemptionContext(run_id="controlled", run_root=run_root, database=database, attempt_matrix=matrix_path, attempt_matrix_sha256=hashlib.sha256(matrix_path.read_bytes()).hexdigest(), max_attempts=8, target_accepted=8)
    hooks = build_rollout_preemption_hooks(context, finalizer=lambda **_kwargs: 0)
    hooks.stop_leases()
    reopened = TaskLedger(database, attempt_matrix=rows, max_attempts=8, target_accepted=8)
    try:
        assert [attempt.attempt_id for attempt in reopened.attempts()] == original_ids
    finally:
        reopened.close()


def test_controlled_smoke_preemption_passes_explicit_finalizer_mode_only_for_exact_1_1(tmp_path):
    from lehome.flywheel.task_ledger import TaskLedger

    root = tmp_path / "workspace" / "smoke"; root.mkdir(parents=True)
    matrix = [{"attempt_id": "smoke", "garment": "Top_Long_Seen_0", "seed": 1}]
    matrix_path = root / "smoke.json"; matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    database = root / "ledger.sqlite3"
    TaskLedger(database, attempt_matrix=matrix, max_attempts=1, target_accepted=1).close()
    context = RolloutPreemptionContext(
        run_id="smoke", run_root=root, database=database, attempt_matrix=matrix_path,
        attempt_matrix_sha256=hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
        max_attempts=1, target_accepted=1, controlled_recovery_smoke=True,
        controlled_recovery_smoke_run_id="a" * 32,
        controlled_recovery_smoke_matrix_sha256="b" * 64,
        controlled_recovery_smoke_materialization_sha256="c" * 64,
        controlled_recovery_smoke_row_index=0,
    )
    calls = []
    hooks = build_rollout_preemption_hooks(context, finalizer=lambda **kwargs: calls.append(kwargs) or 0)
    assert hooks.close_terminal_artifacts()["verified"] is True
    assert calls[0]["controlled_recovery_smoke"] is True
    with pytest.raises(PreemptionError, match="exactly 1/1"):
        RolloutPreemptionContext(
            run_id="bad", run_root=root, database=database, attempt_matrix=matrix_path,
            attempt_matrix_sha256=hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
            max_attempts=2, target_accepted=1, controlled_recovery_smoke=True,
            controlled_recovery_smoke_run_id="a" * 32,
            controlled_recovery_smoke_matrix_sha256="b" * 64,
            controlled_recovery_smoke_materialization_sha256="c" * 64,
            controlled_recovery_smoke_row_index=0,
        )


def test_rollout_preempt_main_refuses_missing_context_without_receipt(tmp_path):
    receipts = tmp_path / "receipts"
    with pytest.raises(PreemptionError, match="rollout preemption context"):
        main([
            "--role", "rollout", "--run-id", "run-1", "--receipts-dir", str(receipts),
            "--training-stop-status", "not-applicable",
            "--rollout-context", str(tmp_path / "missing-context.json"),
            "--workspace-root", str(tmp_path / "workspace"),
        ])
    assert not receipts.exists()


def _write_active_rollout_context(tmp_path: Path, *, run_id: str) -> tuple[Path, Path]:
    """Create the root-authored minimum context needed before hook binding."""

    workspace = tmp_path / "workspace"
    run_root = workspace / "campaign"
    run_root.mkdir(parents=True)
    database = run_root / "ledger.sqlite3"
    database.write_bytes(b"ledger-placeholder")
    matrix = run_root / "attempt-matrix.json"
    matrix.write_text("[]\n", encoding="utf-8")
    context = workspace / "rollout-preemption.json"
    context.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "lehome_rollout_preemption_context",
                "active": True,
                "run_id": run_id,
                "run_root": str(run_root),
                "database": str(database),
                "attempt_matrix": str(matrix),
                "attempt_matrix_sha256": hashlib.sha256(matrix.read_bytes()).hexdigest(),
                "max_attempts": 1,
                "target_accepted": 1,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return workspace, context


def test_rollout_preempt_main_derives_receipt_run_id_from_root_authored_context(
    tmp_path, monkeypatch,
):
    workspace, context = _write_active_rollout_context(tmp_path, run_id="active-campaign")
    receipts = tmp_path / "receipts"
    monkeypatch.setattr(lehome_preempt, "build_rollout_preemption_hooks", lambda _context: make_hooks())

    assert main([
        "--role", "rollout", "--receipts-dir", str(receipts),
        "--training-stop-status", "not-applicable",
        "--rollout-context", str(context), "--workspace-root", str(workspace),
    ]) == 0

    receipt = json.loads((receipts / "preemption-receipt.json").read_text(encoding="utf-8"))
    assert receipt["run_id"] == "active-campaign"


def test_rollout_preempt_main_refuses_an_explicit_run_id_that_conflicts_with_context(
    tmp_path, monkeypatch,
):
    workspace, context = _write_active_rollout_context(tmp_path, run_id="active-campaign")
    monkeypatch.setattr(lehome_preempt, "build_rollout_preemption_hooks", lambda _context: make_hooks())

    with pytest.raises(PreemptionError, match="does not match"):
        main([
            "--role", "rollout", "--run-id", "stale-runtime-env", "--receipts-dir", str(tmp_path / "receipts"),
            "--training-stop-status", "not-applicable",
            "--rollout-context", str(context), "--workspace-root", str(workspace),
        ])
