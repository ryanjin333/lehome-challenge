"""Bounded background finalization for raw terminal rollout episodes.

Workers write raw terminal episodes locally and immediately request new work.
This CPU/disk queue validates, hashes, classifies, and relocates those
episodes without ever blocking a simulator worker:

- an episode counts toward the accepted target only after validation passes;
- closed terminal artifacts are never rewritten; success moves the directory
  atomically into ``accepted/`` while failures stay under ``attempts/``;
- backpressure refuses new handoffs when pending items or bytes exceed
  configured bounds; it never drops evidence to preserve throughput.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
from time import monotonic
from typing import Any

from lehome.flywheel.artifacts import atomic_write_json, build_sha256_manifest


RECEIPT_NAME = "worker-receipt.json"
MANIFEST_NAME = "SHA256SUMS.json"


class QueueFullError(RuntimeError):
    """Backpressure: the finalizer cannot accept another handoff yet."""


@dataclass(frozen=True, slots=True)
class FinalizationResult:
    attempt_id: str
    outcome: str
    reason: str
    accepted_dir: Path | None
    terminal_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class _PendingItem:
    worker_id: str
    attempt_id: str
    lease_id: str
    output_dir: Path
    byte_size: int


def _require_identifier(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _directory_size(root: Path) -> int:
    total = 0
    for current, _directories, file_names in os.walk(root, followlinks=False):
        for file_name in file_names:
            path = Path(current) / file_name
            if path.is_symlink():
                raise ValueError("raw episode must not contain symlinks")
            total += path.stat().st_size
    return total


def _validate_raw_episode(output_dir: Path, attempt_id: str) -> str | None:
    """Return a rejection reason, or None when the raw episode is sound."""
    if output_dir.is_symlink() or not output_dir.is_dir():
        return "output directory missing or unsafe"
    receipt_path = output_dir / RECEIPT_NAME
    if receipt_path.is_symlink() or not receipt_path.is_file():
        return "worker receipt missing"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return f"worker receipt unreadable: {error}"
    if not isinstance(receipt, dict):
        return "worker receipt must be a JSON object"
    if receipt.get("schema_version") != 1:
        return "worker receipt schema_version must be 1"
    if receipt.get("attempt_id") != attempt_id:
        return "worker receipt attempt_id does not match the ledger attempt"
    video_root = output_dir / "videos"
    if video_root.is_symlink() or not video_root.is_dir():
        return "videos directory missing"
    for current, directory_names, file_names in os.walk(output_dir, followlinks=False):
        for directory_name in directory_names:
            if (Path(current) / directory_name).is_symlink():
                return "raw episode contains a symlinked directory"
        for file_name in file_names:
            path = Path(current) / file_name
            if path.is_symlink():
                return "raw episode contains a symlinked file"
            if path.stat().st_size == 0:
                return f"raw episode contains an empty file: {path.relative_to(output_dir)}"
    return None


def _strict_receipt_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    receipt: dict[str, object] = {}
    for key, value in pairs:
        if key in receipt:
            raise ValueError("duplicate worker receipt field")
        receipt[key] = value
    return receipt


def _reject_receipt_constant(value: str) -> object:
    raise ValueError(f"non-finite worker receipt constant: {value}")


def _episode_succeeded(
    output_dir: Path, *, worker_id: str, attempt_id: str, lease_id: str,
) -> tuple[bool | None, str]:
    try:
        receipt = json.loads(
            (output_dir / RECEIPT_NAME).read_text(encoding="utf-8"),
            object_pairs_hook=_strict_receipt_object,
            parse_constant=_reject_receipt_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        return None, f"worker receipt is malformed: {error}"
    if not isinstance(receipt, dict):
        return None, "worker receipt must be an object"
    if (
        type(receipt.get("schema_version")) is not int
        or receipt["schema_version"] != 1
        or receipt.get("attempt_id") != attempt_id
        or receipt.get("worker_id") != worker_id
        or receipt.get("lease_id") != lease_id
    ):
        return None, "worker receipt does not bind the leased identity"
    outcome = receipt.get("outcome")
    if not isinstance(outcome, dict):
        return None, "worker receipt outcome missing"
    success = outcome.get("success")
    if type(success) is not bool:
        return None, "worker receipt outcome success must be a boolean"
    if success:
        return True, ""
    return False, "episode outcome is not a success"


class ArtifactFinalizationQueue:
    """FIFO finalizer with item and byte backpressure bounds.

    ``run_root`` is the appliance-local raw episode root containing
    ``attempts/`` and ``accepted/``.  All queue state is in memory; the
    durable record of every decision lives in the SQLite ledger and the
    relocated accepted artifacts.
    """

    def __init__(
        self,
        *,
        run_root: Path,
        ledger: Any,
        max_pending_items: int,
        max_pending_bytes: int,
        evaluation_only: bool = False,
        evaluation_terminal_root: Path | None = None,
    ) -> None:
        if not isinstance(max_pending_items, int) or isinstance(max_pending_items, bool) or max_pending_items < 1:
            raise ValueError("max_pending_items must be a positive integer")
        if not isinstance(max_pending_bytes, int) or isinstance(max_pending_bytes, bool) or max_pending_bytes < 1:
            raise ValueError("max_pending_bytes must be a positive integer")
        self._run_root = Path(run_root).resolve()
        if type(evaluation_only) is not bool:
            raise ValueError("evaluation_only must be a boolean")
        self._evaluation_terminal_root: Path | None = None
        if evaluation_only and evaluation_terminal_root is None:
            evaluation_terminal_root = self._run_root / "evaluation-terminal"
        if not evaluation_only and evaluation_terminal_root is not None:
            raise ValueError("evaluation terminal root requires evaluation-only mode")
        if evaluation_terminal_root is not None:
            requested = Path(evaluation_terminal_root)
            expected = self._run_root / "evaluation-terminal"
            if requested.is_symlink() or requested.resolve() != expected:
                raise ValueError("evaluation terminal root must be the canonical run-root child")
            self._evaluation_terminal_root = expected
        self._ledger = ledger
        self._max_pending_items = max_pending_items
        self._max_pending_bytes = max_pending_bytes
        self._pending: list[_PendingItem] = []
        self._pending_attempts: set[str] = set()
        self._pending_bytes = 0

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def pending_bytes(self) -> int:
        return self._pending_bytes

    def enqueue(self, worker_id: str, attempt_id: str, lease_id: str, output_dir: Path) -> None:
        """Accept one raw terminal handoff subject to backpressure bounds."""
        worker_id = _require_identifier(worker_id, field="worker_id")
        attempt_id = _require_identifier(attempt_id, field="attempt_id")
        lease_id = _require_identifier(lease_id, field="lease_id")
        resolved = Path(output_dir).resolve()
        if not resolved.is_relative_to(self._run_root):
            raise ValueError("raw episode output must live inside the run root")
        if attempt_id in self._pending_attempts:
            raise ValueError(f"attempt {attempt_id} is already pending finalization")
        if len(self._pending) >= self._max_pending_items:
            raise QueueFullError("pending finalization items exceed the configured bound")
        byte_size = _directory_size(resolved)
        if self._pending_bytes + byte_size > self._max_pending_bytes:
            raise QueueFullError("pending finalization bytes exceed the configured bound")
        self._pending.append(_PendingItem(worker_id, attempt_id, lease_id, resolved, byte_size))
        self._pending_attempts.add(attempt_id)
        self._pending_bytes += byte_size

    def finalize_next(self) -> FinalizationResult | None:
        """Validate and settle the oldest pending episode, if any."""
        if not self._pending:
            return None
        item = self._pending.pop(0)
        self._pending_attempts.discard(item.attempt_id)
        self._pending_bytes -= item.byte_size

        validation_reason = _validate_raw_episode(item.output_dir, item.attempt_id)
        if validation_reason is not None:
            self._ledger.validate_terminal(item.attempt_id, "infrastructure_abort")
            return FinalizationResult(
                item.attempt_id, "infrastructure_abort", validation_reason, None,
            )

        succeeded, success_reason = _episode_succeeded(
            item.output_dir,
            worker_id=item.worker_id,
            attempt_id=item.attempt_id,
            lease_id=item.lease_id,
        )
        if succeeded is None:
            self._ledger.validate_terminal(item.attempt_id, "infrastructure_abort")
            return FinalizationResult(
                item.attempt_id, "infrastructure_abort", success_reason, None,
            )
        if self._evaluation_terminal_root is not None:
            terminal_dir = self._publish_evaluation_terminal(item)
            outcome = "accepted" if succeeded else "rejected"
            self._ledger.validate_terminal(
                item.attempt_id, outcome, artifact_id=str(terminal_dir),
            )
            return FinalizationResult(
                item.attempt_id,
                outcome,
                "" if succeeded else success_reason,
                None,
                terminal_dir,
            )
        reason: str | None = None if succeeded else success_reason
        if reason is None:
            # Multiple appliance finalizer processes may observe terminal
            # handoffs at once. Serialize the cap check, move, and append-only
            # ledger transition across them so no losing success is moved to
            # accepted before the category/global quota is known to be open.
            with self._admission_lock():
                if self._controlled_category_full(item.attempt_id):
                    reason = "controlled recovery category acceptance cap already reached"
                elif self._ledger.accepted_count() >= self._ledger.target_accepted():
                    reason = "accepted episode target already reached"
                else:
                    accepted_dir = self._accept(item)
                    self._ledger.validate_terminal(item.attempt_id, "accepted", artifact_id=str(accepted_dir))
                    return FinalizationResult(item.attempt_id, "accepted", "", accepted_dir)

        self._ledger.validate_terminal(item.attempt_id, "rejected")
        return FinalizationResult(item.attempt_id, "rejected", reason, None)

    def _controlled_category_full(self, attempt_id: str) -> bool:
        assignment = next(
            (attempt.assignment for attempt in self._ledger.attempts() if attempt.attempt_id == attempt_id),
            None,
        )
        if not isinstance(assignment, dict) or assignment.get("recovery_kind") != "controlled_success_recovery_snapshot_v3":
            return False
        category = assignment.get("category")
        cap = assignment.get("category_acceptance_cap")
        if not isinstance(category, str) or type(cap) is not int or cap < 0:
            raise ValueError("controlled recovery assignment has an invalid category cap")
        accepted = 0
        for attempt in self._ledger.attempts():
            candidate = attempt.assignment
            if (candidate.get("recovery_kind") == "controlled_success_recovery_snapshot_v3"
                    and candidate.get("category") == category
                    and self._ledger.status(attempt.attempt_id) == "accepted"):
                accepted += 1
        return accepted >= cap

    @contextmanager
    def _admission_lock(self):
        self._run_root.mkdir(parents=True, exist_ok=True)
        if self._run_root.is_symlink() or not self._run_root.is_dir():
            raise ValueError("run root must be a real directory")
        path = self._run_root / ".finalizer-admission.lock"
        if path.is_symlink():
            raise ValueError("finalizer admission lock is unsafe")
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def drain(self, *, deadline_seconds: float) -> int:
        """Finalize every pending episode within the shutdown deadline."""
        finalized = 0
        deadline = monotonic() + deadline_seconds
        while self._pending:
            if monotonic() > deadline:
                break
            self.finalize_next()
            finalized += 1
        return finalized

    def _accept(self, item: _PendingItem) -> Path:
        accepted_root = self._run_root / "accepted"
        accepted_root.mkdir(parents=True, exist_ok=True)
        if accepted_root.is_symlink():
            raise ValueError("accepted root must not be a symlink")
        destination = accepted_root / item.attempt_id
        if destination.exists() or destination.is_symlink():
            raise ValueError(f"accepted artifact already exists: {item.attempt_id}")
        os.replace(item.output_dir, destination)
        manifest = build_sha256_manifest(destination)
        atomic_write_json(destination / MANIFEST_NAME, manifest)
        return destination

    def _publish_evaluation_terminal(self, item: _PendingItem) -> Path:
        """Retain every valid evaluation outcome without training admission."""
        if self._evaluation_terminal_root is None:
            raise RuntimeError("evaluation terminal publication is not enabled")
        root = self._evaluation_terminal_root
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink() or not root.is_dir():
            raise ValueError("evaluation terminal root must be a real directory")
        destination = root / item.attempt_id
        if destination.exists() or destination.is_symlink():
            raise ValueError(f"evaluation terminal artifact already exists: {item.attempt_id}")
        os.replace(item.output_dir, destination)
        manifest = build_sha256_manifest(destination)
        atomic_write_json(destination / MANIFEST_NAME, manifest)
        return destination


__all__ = ["ArtifactFinalizationQueue", "FinalizationResult", "QueueFullError"]
