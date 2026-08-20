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
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
from typing import Callable, Mapping, Sequence
from uuid import uuid4

from lehome_workspace import write_manifest_atomic


PREEMPTION_BUDGET_SECONDS = 60
RECEIPT_NAME = "preemption-receipt.json"
ROLLOUT_CONTEXT_NAME = "rollout-preemption.json"
_LOWERCASE_SHA256 = __import__("re").compile(r"^[0-9a-f]{64}$")


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


@dataclass(frozen=True, slots=True)
class RolloutPreemptionContext:
    """Exact active campaign inputs needed to close a preempted rollout."""

    run_id: str
    run_root: Path
    database: Path
    attempt_matrix: Path
    attempt_matrix_sha256: str
    max_attempts: int
    target_accepted: int
    controlled_recovery_smoke: bool = False
    controlled_recovery_smoke_run_id: str | None = None
    controlled_recovery_smoke_matrix_sha256: str | None = None
    controlled_recovery_smoke_materialization_sha256: str | None = None
    controlled_recovery_smoke_row_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise PreemptionError("rollout preemption context run_id is invalid")
        for label in ("run_root", "database", "attempt_matrix"):
            value = getattr(self, label)
            if not isinstance(value, Path) or not value.is_absolute():
                raise PreemptionError(f"rollout preemption context {label} must be absolute")
        if not _LOWERCASE_SHA256.fullmatch(self.attempt_matrix_sha256):
            raise PreemptionError("rollout preemption context matrix digest is invalid")
        if type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 400:
            raise PreemptionError("rollout preemption context max_attempts must be in 1..400")
        if type(self.target_accepted) is not int or not 1 <= self.target_accepted <= 150:
            raise PreemptionError("rollout preemption context target_accepted must be in 1..150")
        if self.controlled_recovery_smoke:
            if self.max_attempts != 1 or self.target_accepted != 1:
                raise PreemptionError("controlled smoke preemption context must be exactly 1/1")
            if (not isinstance(self.controlled_recovery_smoke_run_id, str)
                    or __import__("re").fullmatch(r"[0-9a-f]{32}", self.controlled_recovery_smoke_run_id) is None
                    or not isinstance(self.controlled_recovery_smoke_row_index, int)
                    or self.controlled_recovery_smoke_row_index < 0):
                raise PreemptionError("controlled smoke preemption identity is invalid")
            for value in (self.controlled_recovery_smoke_matrix_sha256, self.controlled_recovery_smoke_materialization_sha256):
                if not isinstance(value, str) or _LOWERCASE_SHA256.fullmatch(value) is None:
                    raise PreemptionError("controlled smoke preemption hashes are invalid")
        elif any(value is not None for value in (
            self.controlled_recovery_smoke_run_id, self.controlled_recovery_smoke_matrix_sha256,
            self.controlled_recovery_smoke_materialization_sha256, self.controlled_recovery_smoke_row_index,
        )):
            raise PreemptionError("normal rollout preemption context must not carry smoke identity")


def _regular_file(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError:
        raise PreemptionError(f"{label} is missing") from None
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o022:
        raise PreemptionError(f"{label} must be a non-writable-by-others regular file")


def load_rollout_preemption_context(
    path: Path, *, workspace_root: Path,
) -> RolloutPreemptionContext:
    """Load the root-authored active-campaign pointer without guessing a ledger."""

    _regular_file(path, "rollout preemption context")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise PreemptionError("rollout preemption context is unreadable") from None
    expected = {
        "schema_version", "kind", "active", "run_id", "run_root", "database",
        "attempt_matrix", "attempt_matrix_sha256", "max_attempts", "target_accepted",
    }
    smoke_fields = {
        "controlled_recovery_smoke", "controlled_recovery_smoke_run_id",
        "controlled_recovery_smoke_matrix_sha256", "controlled_recovery_smoke_materialization_sha256",
        "controlled_recovery_smoke_row_index",
    }
    if isinstance(payload, dict) and payload.get("controlled_recovery_smoke") is True:
        expected |= smoke_fields
    if (
        not isinstance(payload, dict)
        or set(payload) != expected
        or payload.get("schema_version") != 1
        or payload.get("kind") != "lehome_rollout_preemption_context"
        or payload.get("active") is not True
    ):
        raise PreemptionError("rollout preemption context has an incompatible schema or is inactive")
    try:
        context = RolloutPreemptionContext(
            run_id=payload["run_id"],
            run_root=Path(payload["run_root"]),
            database=Path(payload["database"]),
            attempt_matrix=Path(payload["attempt_matrix"]),
            attempt_matrix_sha256=payload["attempt_matrix_sha256"],
            max_attempts=payload["max_attempts"],
            target_accepted=payload["target_accepted"],
            controlled_recovery_smoke=payload.get("controlled_recovery_smoke", False),
            controlled_recovery_smoke_run_id=payload.get("controlled_recovery_smoke_run_id"),
            controlled_recovery_smoke_matrix_sha256=payload.get("controlled_recovery_smoke_matrix_sha256"),
            controlled_recovery_smoke_materialization_sha256=payload.get("controlled_recovery_smoke_materialization_sha256"),
            controlled_recovery_smoke_row_index=payload.get("controlled_recovery_smoke_row_index"),
        )
    except (KeyError, TypeError):
        raise PreemptionError("rollout preemption context is malformed") from None
    workspace = workspace_root.resolve(strict=True)
    run_root = context.run_root.resolve(strict=True)
    if workspace_root.is_symlink() or context.run_root.is_symlink() or not run_root.is_relative_to(workspace):
        raise PreemptionError("rollout preemption run root is outside the shared workspace")
    if context.database != context.run_root / "ledger.sqlite3":
        raise PreemptionError("rollout preemption database is not the campaign ledger")
    _regular_file(context.database, "rollout campaign ledger")
    _regular_file(context.attempt_matrix, "rollout attempt matrix")
    if hashlib.sha256(context.attempt_matrix.read_bytes()).hexdigest() != context.attempt_matrix_sha256:
        raise PreemptionError("rollout attempt matrix SHA-256 mismatch")
    return context


def _attempt_matrix(path: Path) -> list[Mapping[str, object]]:
    try:
        from lehome.flywheel.recovery_collection import load_attempt_matrix

        return load_attempt_matrix(path)
    except (ImportError, ValueError) as error:
        raise PreemptionError(f"rollout attempt matrix is invalid: {error}") from None


def _checkpoint_and_fsync(database: Path) -> Mapping[str, object]:
    connection = sqlite3.connect(database, isolation_level=None, timeout=5.0)
    try:
        connection.execute("PRAGMA busy_timeout = 5000")
        busy, log_frames, checkpointed_frames = connection.execute("PRAGMA wal_checkpoint(FULL)").fetchone()
    finally:
        connection.close()
    if busy:
        raise PreemptionError("rollout ledger WAL checkpoint remained busy")
    fsynced: list[str] = []
    for candidate in (database, Path(f"{database}-wal"), Path(f"{database}-shm")):
        if not candidate.exists():
            continue
        _regular_file(candidate, "rollout ledger file")
        descriptor = os.open(candidate, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        fsynced.append(candidate.name)
    _directory_fsync(database.parent)
    return {
        "verified": True,
        "wal_checkpoint_busy": busy,
        "wal_log_frames": log_frames,
        "wal_checkpointed_frames": checkpointed_frames,
        "fsynced": fsynced,
    }


def build_rollout_preemption_hooks(
    context: RolloutPreemptionContext,
    *,
    finalizer: Callable[..., int] | None = None,
) -> PreemptionHooks:
    """Bind real ledger/finalizer hooks for one explicit active campaign."""

    from lehome.flywheel.task_ledger import TaskLedger

    matrix = _attempt_matrix(context.attempt_matrix)

    def open_ledger() -> TaskLedger:
        return TaskLedger(
            context.database,
            attempt_matrix=matrix,
            max_attempts=context.max_attempts,
            target_accepted=context.target_accepted,
        )

    def stop_leases() -> Mapping[str, object]:
        ledger = open_ledger()
        try:
            active_before = sum(ledger.status(attempt.attempt_id) == "leased" for attempt in ledger.attempts())
            ledger.pause_for_preemption("nebius-preemption")
            return {
                "verified": ledger.is_stopped,
                "campaign_paused": ledger.is_stopped and not ledger.is_terminal,
                "active_leases_interrupted": active_before,
            }
        finally:
            ledger.close()

    def mark_interrupted() -> Mapping[str, object]:
        ledger = open_ledger()
        try:
            if not ledger.is_stopped or ledger.is_terminal:
                raise PreemptionError("rollout campaign did not enter resumable paused state")
            if any(ledger.status(attempt.attempt_id) == "leased" for attempt in ledger.attempts()):
                raise PreemptionError("rollout campaign retains an active lease after pause")
            retryable = sum(
                event.event_type == "retryable" and event.payload.get("reason") == "preemption"
                for event in ledger.events()
            )
            return {"verified": True, "retryable_preemption_attempts": retryable}
        finally:
            ledger.close()

    def flush_ledgers() -> Mapping[str, object]:
        return _checkpoint_and_fsync(context.database)

    def close_terminal_artifacts() -> Mapping[str, object]:
        selected_finalizer = finalizer
        if selected_finalizer is None:
            from scripts.run_groot_artifact_sync import run_finalizer_once
            selected_finalizer = run_finalizer_once
        finalized = selected_finalizer(
            database=context.database,
            attempt_matrix=context.attempt_matrix,
            run_root=context.run_root,
            max_pending_items=16,
            max_pending_bytes=16 * 2**30,
            max_attempts=context.max_attempts,
            target_accepted=context.target_accepted,
            controlled_recovery_smoke=context.controlled_recovery_smoke,
        )
        _checkpoint_and_fsync(context.database)
        return {"verified": True, "finalized": finalized}

    return PreemptionHooks(
        stop_leases=stop_leases,
        mark_interrupted=mark_interrupted,
        flush_ledgers=flush_ledgers,
        close_terminal_artifacts=close_terminal_artifacts,
    )


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



def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Write a bounded LeHome preemption receipt.")
    parser.add_argument("--role", required=True, choices=("training", "rollout"))
    # A rollout's durable context is the authority for its identity. A
    # runtime.env file can outlive an old campaign on the same shared disk,
    # while training has no equivalent root-authored context.
    parser.add_argument("--run-id")
    parser.add_argument("--receipts-dir", required=True)
    parser.add_argument(
        "--training-stop-status", required=True,
        choices=("stopped", "failed", "not-applicable"),
    )
    parser.add_argument(
        "--rollout-context", type=Path,
        default=Path("/run/lehome/rollout-preemption.json"),
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("/mnt/lehome"))
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.role == "training":
        if not args.run_id:
            raise PreemptionError("training preemption requires a run_id")
        if args.training_stop_status != "stopped":
            raise PreemptionError("trainer stop was not confirmed; refusing preemption receipt")
    rollout_context = None
    if args.role == "rollout":
        rollout_context = load_rollout_preemption_context(
            args.rollout_context, workspace_root=args.workspace_root,
        )
        if args.run_id is not None and rollout_context.run_id != args.run_id:
            raise PreemptionError("rollout preemption context run_id does not match the active VM")
        run_id = rollout_context.run_id
    else:
        run_id = args.run_id
    receipts = Path(args.receipts_dir)
    receipts.mkdir(parents=True, exist_ok=True)
    if rollout_context is not None:
        hooks = build_rollout_preemption_hooks(rollout_context)
    else:
        not_applicable = lambda: {"applicable": False, "role": "training"}
        hooks = PreemptionHooks(
            stop_leases=lambda: {"verified": True, "training_process": "stopped"},
            mark_interrupted=not_applicable,
            flush_ledgers=not_applicable,
            close_terminal_artifacts=not_applicable,
        )
    result = handle_preemption(
        receipts_dir=receipts,
        hooks=hooks,
        role=args.role,
        run_id=run_id,
        clock=time.time,
    )
    required = {"stop_leases", "mark_interrupted", "flush_ledgers", "close_terminal_artifacts"}
    if result.receipt.get("errors") or not required.issubset(result.completed_steps):
        raise PreemptionError("preemption shutdown was incomplete; inspect the durable receipt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
