"""Bounded 60-second preemption shutdown for Nebius preemptible guests.

Nebius delivers SIGTERM with a documented ~60 second window before loss.  The
handler never pretends to suspend opaque Isaac or trainer process memory; it
executes a fixed, receipt-producing sequence:

1. stop issuing new training work or rollout leases;
2. mark in-flight rollout attempts retryable and flush ledgers;
3. atomically close only already-terminal artifacts;
4. request a bounded training checkpoint only when the trainer confirms one
   can be completed inside the remaining budget;
5. persist a lifecycle receipt on the shared disk;
6. run a deadline-bounded Hugging Face sync of small critical receipts;
7. exit without deleting local durable state.

Every dependency is injected: clocks, hooks, and subprocesses are all
testable without root, GPUs, or network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Callable, Mapping, Sequence
from uuid import uuid4

from lehome_workspace import write_manifest_atomic


PREEMPTION_BUDGET_SECONDS = 60
RECEIPT_NAME = "preemption-receipt.json"


class PreemptionError(RuntimeError):
    """A shutdown step failed; the receipt still records what completed."""


@dataclass(frozen=True, slots=True)
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str


CommandRunner = Callable[[Sequence[str]], CommandResult]
ClockSeconds = Callable[[], float]


@dataclass(slots=True)
class PreemptionHooks:
    """Ordered shutdown callbacks; each returns a small JSON-safe summary."""

    stop_leases: Callable[[], Mapping[str, object]]
    mark_interrupted: Callable[[], Mapping[str, object]]
    flush_ledgers: Callable[[], Mapping[str, object]]
    close_terminal_artifacts: Callable[[], Mapping[str, object]]
    request_training_checkpoint: Callable[[float], Mapping[str, object]] | None = None
    bounded_hub_sync: CommandRunner | None = None


@dataclass(frozen=True, slots=True)
class PreemptionResult:
    receipt_path: Path
    receipt: Mapping[str, object]
    completed_steps: tuple[str, ...]
    checkpoint_requested: bool
    hub_sync: Mapping[str, object]


def _directory_fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remaining(deadline_seconds: float, clock: ClockSeconds) -> float:
    return deadline_seconds - clock()


def handle_preemption(
    *,
    receipts_dir: Path,
    hooks: PreemptionHooks,
    role: str,
    run_id: str,
    clock: ClockSeconds,
    budget_seconds: float = PREEMPTION_BUDGET_SECONDS,
    hub_sync_command: Sequence[str] | None = None,
) -> PreemptionResult:
    """Run the bounded shutdown sequence and persist its receipt.

    The receipt is written even when later steps run out of budget; it is the
    evidence a replacement VM uses to decide what to retry.  Local durable
    state is never deleted here.
    """
    if role not in {"training", "rollout"}:
        raise PreemptionError(f"role must be training or rollout, got {role!r}")
    if not isinstance(budget_seconds, (int, float)) or isinstance(budget_seconds, bool) or budget_seconds <= 0:
        raise PreemptionError("budget_seconds must be positive")
    if receipts_dir.is_symlink() or not receipts_dir.is_dir():
        raise PreemptionError(f"receipts directory missing or unsafe: {receipts_dir}")

    started = clock()
    deadline = started + budget_seconds
    completed: list[str] = []
    steps: dict[str, Mapping[str, object]] = {}
    checkpoint_requested = False
    hub_sync_summary: Mapping[str, object] = {"attempted": False}
    errors: list[str] = []

    def _execute(name: str, action: Callable[[], Mapping[str, object]]) -> bool:
        if _remaining(deadline, clock) <= 0:
            errors.append(f"{name}: skipped, budget exhausted")
            return False
        try:
            summary = action()
        except Exception as error:  # noqa: BLE001 - shutdown must keep going
            errors.append(f"{name}: {error}")
            return False
        if not isinstance(summary, Mapping):
            raise PreemptionError(f"hook {name} must return a mapping summary")
        steps[name] = summary
        completed.append(name)
        return True

    _execute("stop_leases", hooks.stop_leases)
    _execute("mark_interrupted", hooks.mark_interrupted)
    _execute("flush_ledgers", hooks.flush_ledgers)
    _execute("close_terminal_artifacts", hooks.close_terminal_artifacts)

    if hooks.request_training_checkpoint is not None and role == "training":
        remaining = _remaining(deadline, clock)
        if remaining > 0:
            try:
                summary = hooks.request_training_checkpoint(remaining)
                checkpoint_requested = True
                steps["training_checkpoint"] = summary
                completed.append("training_checkpoint")
            except Exception as error:  # noqa: BLE001
                errors.append(f"training_checkpoint: {error}")

    if hooks.bounded_hub_sync is not None and hub_sync_command is not None:
        remaining = _remaining(deadline, clock)
        if remaining > 0:
            try:
                result = hooks.bounded_hub_sync(list(hub_sync_command) + ["--deadline-seconds", f"{remaining:.3f}"])
                hub_sync_summary = {
                    "attempted": True,
                    "exit_code": result.exit_code,
                    "stderr_tail": result.stderr.strip()[-200:],
                }
                if result.exit_code == 0:
                    completed.append("bounded_hub_sync")
            except Exception as error:  # noqa: BLE001
                errors.append(f"bounded_hub_sync: {error}")
                hub_sync_summary = {"attempted": True, "error": str(error)}

    receipt = {
        "schema_version": 1,
        "kind": "preemption_shutdown",
        "role": role,
        "run_id": run_id,
        "started_at_epoch_seconds": started,
        "budget_seconds": budget_seconds,
        "completed_steps": list(completed),
        "checkpoint_requested": checkpoint_requested,
        "hub_sync": hub_sync_summary,
        "steps": steps,
        "errors": errors,
        "finished_at_epoch_seconds": clock(),
    }
    receipt_path = receipts_dir / RECEIPT_NAME
    if receipt_path.exists() or receipt_path.is_symlink():
        receipt_path = receipts_dir / f"preemption-receipt-{uuid4().hex}.json"
    write_manifest_atomic(receipt_path, receipt)
    _directory_fsync(receipts_dir)
    return PreemptionResult(
        receipt_path=receipt_path,
        receipt=receipt,
        completed_steps=tuple(completed),
        checkpoint_requested=checkpoint_requested,
        hub_sync=hub_sync_summary,
    )
