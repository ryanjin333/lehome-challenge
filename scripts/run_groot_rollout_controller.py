"""Run the local JSON-lines transport for the persistent rollout ledger.

Workers may use a socket/RPC adapter in production, but the operation boundary
is intentionally small and every operation below mutates or reads ``TaskLedger``
rather than maintaining a second in-memory scheduler.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

from lehome.flywheel.task_ledger import Lease, TaskLedger


_MAX_SQLITE_INTEGER = 2**63 - 1


def _lease_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("lease seconds must be a finite positive number") from error
    nanoseconds = seconds * 1_000_000_000
    if not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(nanoseconds) or not 1 <= nanoseconds <= _MAX_SQLITE_INTEGER:
        raise argparse.ArgumentTypeError("lease seconds must convert to a positive SQLite-safe nanosecond duration")
    return seconds


class LocalRolloutController:
    """Transport-neutral dispatcher for lease, heartbeat, terminal, and stop."""

    def __init__(self, ledger: TaskLedger, *, lease_duration_ns: int) -> None:
        self._ledger = ledger
        self._lease_duration_ns = lease_duration_ns

    def handle(self, request: Mapping[str, object]) -> dict[str, object]:
        operation = request.get("operation")
        if operation == "lease":
            lease = self._ledger.lease_next(self._string(request, "worker_id"), lease_duration_ns=self._lease_duration_ns)
            return {"status": "unavailable"} if lease is None else self._lease_response(lease)
        if operation == "heartbeat":
            lease = self._ledger.heartbeat(
                self._string(request, "worker_id"), self._string(request, "attempt_id"), self._string(request, "lease_id"),
                lease_duration_ns=self._lease_duration_ns,
            )
            return self._lease_response(lease)
        if operation == "terminal":
            status = self._ledger.record_terminal(
                self._string(request, "worker_id"), self._string(request, "attempt_id"), self._string(request, "lease_id"),
                self._string(request, "raw_artifact_id"),
            )
            return {"status": status}
        if operation == "stop":
            self._ledger.stop_for_preemption(self._string(request, "reason"))
            return {"status": "stopped"}
        if operation == "resume":
            self._ledger.resume_after_preemption(self._string(request, "reason"))
            return {"status": "resumed"}
        raise ValueError("operation must be lease, heartbeat, terminal, stop, or resume")

    @staticmethod
    def _string(request: Mapping[str, object], field: str) -> str:
        value = request.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field} must be a non-empty string")
        return value

    @staticmethod
    def _lease_response(lease: Lease) -> dict[str, object]:
        return {
            "status": "leased",
            "lease_id": lease.lease_id,
            "worker_id": lease.worker_id,
            "expires_at_ns": lease.expires_at_ns,
            "attempt": {
                "attempt_id": lease.attempt.attempt_id,
                "schedule_index": lease.attempt.schedule_index,
                "assignment": dict(lease.attempt.assignment),
            },
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--attempt-matrix", type=Path, required=True, help="immutable JSON array of assignment objects")
    parser.add_argument("--lease-seconds", type=_lease_seconds, default=300.0)
    parser.add_argument("--max-attempts", type=int, default=400)
    parser.add_argument("--target-accepted", type=int, default=150)
    return parser


def _load_matrix(path: Path) -> list[Mapping[str, object]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("attempt matrix must be a regular JSON file")
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("attempt matrix is not valid JSON") from error
    if not isinstance(decoded, list) or not all(isinstance(item, dict) for item in decoded):
        raise ValueError("attempt matrix must be a JSON array of objects")
    return decoded


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lease_duration_ns = int(args.lease_seconds * 1_000_000_000)
    ledger = TaskLedger(
        args.database, attempt_matrix=_load_matrix(args.attempt_matrix), max_attempts=args.max_attempts,
        target_accepted=args.target_accepted,
    )
    controller = LocalRolloutController(ledger, lease_duration_ns=lease_duration_ns)
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("request must be a JSON object")
                response: dict[str, Any] = controller.handle(request)
            except (ValueError, TypeError) as error:
                response = {"status": "error", "error": str(error)}
            print(json.dumps(response, sort_keys=True, separators=(",", ":")), flush=True)
    finally:
        ledger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
