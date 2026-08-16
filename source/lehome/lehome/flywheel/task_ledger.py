"""Append-only SQLite lease ledger for persistent rollout workers.

The ledger deliberately stores the immutable rollout schedule separately from
the worker that executes it.  A retry therefore produces more events against
the same attempt ID; it never mutates an old lease or invents a new schedule
entry.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from threading import RLock
from time import time_ns
from typing import Any, Callable, Iterator, Mapping


MAX_CAMPAIGN_ATTEMPTS = 400
MAX_ACCEPTED_EPISODES = 150
MAX_SQLITE_INTEGER = 2**63 - 1
_TERMINAL_STATUSES = frozenset({"accepted", "rejected", "infrastructure_abort"})
_LEASED_STATES = frozenset({"leased"})
_STATE_EVENTS = frozenset({
    "leased", "lease_expired", "retryable", "interrupted", "terminal_pending_validation",
    "accepted", "rejected", "infrastructure_abort",
})


@dataclass(frozen=True, slots=True)
class Attempt:
    """One immutable schedule entry, independent of its executing worker."""

    attempt_id: str
    schedule_index: int
    assignment: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Lease:
    """A time-bounded right for a worker to execute one immutable attempt."""

    attempt: Attempt
    lease_id: str
    worker_id: str
    expires_at_ns: int


@dataclass(frozen=True, slots=True)
class LedgerEvent:
    event_id: int
    at_ns: int
    event_type: str
    attempt_id: str | None
    lease_id: str | None
    worker_id: str | None
    payload: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _AttemptState:
    status: str
    lease: Lease | None


def _canonical_json(value: object) -> str:
    """Encode only deterministic, JSON-native schedule and event values."""

    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("ledger values must be canonical JSON") from error


def _decode_json(value: str) -> object:
    return json.loads(value)


def _require_identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


class TaskLedger:
    """SQLite WAL ledger with append-only transition records.

    ``clock_ns`` is injected so expiry, heartbeat, and preemption behavior can
    be tested without wall-clock sleeps.  Every mutation uses ``BEGIN
    IMMEDIATE``: lease selection and issuance remain one serialized operation
    even when controller requests arrive concurrently.
    """

    def __init__(
        self,
        database: Path | str,
        *,
        attempt_matrix: list[Mapping[str, object]] | tuple[Mapping[str, object], ...],
        max_attempts: int = MAX_CAMPAIGN_ATTEMPTS,
        target_accepted: int = MAX_ACCEPTED_EPISODES,
        clock_ns: Callable[[], int] = time_ns,
    ) -> None:
        if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or not 1 <= max_attempts <= MAX_CAMPAIGN_ATTEMPTS:
            raise ValueError(f"max_attempts must be in 1..{MAX_CAMPAIGN_ATTEMPTS}")
        if not isinstance(target_accepted, int) or isinstance(target_accepted, bool) or not 1 <= target_accepted <= MAX_ACCEPTED_EPISODES:
            raise ValueError(f"target_accepted must be in 1..{MAX_ACCEPTED_EPISODES}")
        if len(attempt_matrix) > max_attempts:
            raise ValueError(f"attempt matrix exceeds max_attempts ({max_attempts}, never above {MAX_CAMPAIGN_ATTEMPTS})")
        self._clock_ns = clock_ns
        self._database = Path(database).resolve()
        self._lock = RLock()
        self._connection = sqlite3.connect(self._database, isolation_level=None, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._journal_mode = str(self._connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]).lower()
        if self._journal_mode != "wal":
            self._connection.close()
            raise RuntimeError(f"SQLite WAL mode is required, got {self._journal_mode!r}")
        self._connection.execute("PRAGMA synchronous = FULL")
        self._assert_schema_is_new_or_complete()
        self._create_schema()
        self._matrix = self._normalize_matrix(attempt_matrix)
        self._matrix_sha256 = sha256(_canonical_json([attempt.assignment for attempt in self._matrix]).encode("utf-8")).hexdigest()
        self._initialize_or_verify(max_attempts=max_attempts, target_accepted=target_accepted)

    @property
    def is_stopped(self) -> bool:
        """Whether dispatch is currently paused or terminally ended."""

        with self._lock:
            return self._campaign_state() != "active"

    @property
    def is_terminal(self) -> bool:
        """A terminally ended campaign cannot be reopened after preemption."""

        with self._lock:
            return self._campaign_state() == "ended"

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def journal_mode(self) -> str:
        with self._lock:
            return self._journal_mode

    def audit_connection(self) -> sqlite3.Connection:
        """Open a separate OS-enforced read-only SQLite connection for audits."""

        with self._lock:
            connection = sqlite3.connect(f"{self._database.as_uri()}?mode=ro", uri=True, isolation_level=None)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only = ON")
            return connection

    def attempts(self) -> tuple[Attempt, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT attempt_id, schedule_index, assignment_json FROM attempts ORDER BY schedule_index"
            ).fetchall()
            return tuple(
                Attempt(row["attempt_id"], row["schedule_index"], _decode_json(row["assignment_json"]))
                for row in rows
            )

    def events(self, attempt_id: str | None = None) -> tuple[LedgerEvent, ...]:
        with self._lock:
            if attempt_id is None:
                rows = self._connection.execute(
                    "SELECT event_id, at_ns, event_type, attempt_id, lease_id, worker_id, payload_json FROM events ORDER BY event_id"
                ).fetchall()
            else:
                rows = self._connection.execute(
                    "SELECT event_id, at_ns, event_type, attempt_id, lease_id, worker_id, payload_json "
                    "FROM events WHERE attempt_id = ? ORDER BY event_id", (attempt_id,)
                ).fetchall()
            return tuple(self._event_from_row(row) for row in rows)

    def status(self, attempt_id: str) -> str:
        with self._lock:
            return self._state_for_attempt(attempt_id).status

    def lease_next(self, worker_id: str, *, lease_duration_ns: int) -> Lease | None:
        """Give a free worker the earliest retryable/pending schedule entry."""

        worker_id = _require_identifier(worker_id, field="worker_id")
        self._require_duration(lease_duration_ns)
        with self._write():
            now_ns = self._now()
            expires_at_ns = self._expires_at(now_ns, lease_duration_ns)
            if self.is_stopped:
                return None
            self._expire_leases(now_ns)
            existing = self._active_lease_for_worker(worker_id)
            if existing is not None:
                return existing
            if self._accepted_count() >= self._target_accepted() or self._issued_lease_count() >= self._max_attempts():
                return None
            for attempt in self.attempts():
                state = self._state_for_attempt(attempt.attempt_id)
                if state.status not in {"retryable", "pending"}:
                    continue
                sequence = self._lease_sequence(attempt.attempt_id) + 1
                lease_id = sha256(f"{attempt.attempt_id}:lease:{sequence}".encode("ascii")).hexdigest()
                lease = Lease(attempt, lease_id, worker_id, expires_at_ns)
                self._append_event(
                    "leased", attempt_id=attempt.attempt_id, lease_id=lease_id, worker_id=worker_id,
                    payload={"expires_at_ns": expires_at_ns, "lease_sequence": sequence}, at_ns=now_ns,
                )
                return lease
            return None

    def heartbeat(self, worker_id: str, attempt_id: str, lease_id: str, *, lease_duration_ns: int) -> Lease:
        worker_id = _require_identifier(worker_id, field="worker_id")
        _require_identifier(attempt_id, field="attempt_id")
        _require_identifier(lease_id, field="lease_id")
        self._require_duration(lease_duration_ns)
        with self._write():
            now_ns = self._now()
            expires_at_ns = self._expires_at(now_ns, lease_duration_ns)
            self._expire_leases(now_ns)
            state = self._state_for_attempt(attempt_id)
            lease = self._require_active_lease(state, worker_id, lease_id)
            renewed = Lease(lease.attempt, lease.lease_id, lease.worker_id, expires_at_ns)
            self._append_event(
                "heartbeat", attempt_id=attempt_id, lease_id=lease_id, worker_id=worker_id,
                payload={"expires_at_ns": renewed.expires_at_ns}, at_ns=now_ns,
            )
            return renewed

    def record_terminal(self, worker_id: str, attempt_id: str, lease_id: str, raw_artifact_id: str) -> str:
        """Close worker execution before asynchronous validation begins."""

        worker_id = _require_identifier(worker_id, field="worker_id")
        raw_artifact_id = _require_identifier(raw_artifact_id, field="raw_artifact_id")
        with self._write():
            now_ns = self._now()
            self._expire_leases(now_ns)
            state = self._state_for_attempt(attempt_id)
            if state.status in {"terminal_pending_validation", *_TERMINAL_STATUSES}:
                matching = [
                    event for event in self.events(attempt_id)
                    if event.event_type == "terminal_pending_validation"
                    and event.lease_id == lease_id
                    and event.worker_id == worker_id
                    and event.payload.get("raw_artifact_id") == raw_artifact_id
                ]
                if matching:
                    return state.status
                raise ValueError("attempt already has a terminal result")
            self._require_active_lease(state, worker_id, lease_id)
            existing_raw = self._connection.execute(
                "SELECT attempt_id FROM events WHERE event_type = 'terminal_pending_validation' "
                "AND json_extract(payload_json, '$.raw_artifact_id') = ? LIMIT 1", (raw_artifact_id,)
            ).fetchone()
            if existing_raw is not None:
                raise ValueError("raw artifact is already attached to another terminal attempt")
            self._append_event(
                "terminal_pending_validation", attempt_id=attempt_id, lease_id=lease_id, worker_id=worker_id,
                payload={"raw_artifact_id": raw_artifact_id}, at_ns=now_ns,
            )
            return "terminal_pending_validation"

    def validate_terminal(self, attempt_id: str, outcome: str, *, artifact_id: str | None = None) -> str:
        """Record one immutable validator outcome after a raw terminal close."""

        _require_identifier(attempt_id, field="attempt_id")
        if outcome not in _TERMINAL_STATUSES:
            raise ValueError("outcome must be accepted, rejected, or infrastructure_abort")
        if outcome == "accepted":
            artifact_id = _require_identifier(artifact_id, field="artifact_id")
        elif artifact_id is not None:
            artifact_id = _require_identifier(artifact_id, field="artifact_id")
        with self._write():
            state = self._state_for_attempt(attempt_id)
            payload = {"artifact_id": artifact_id} if artifact_id is not None else {}
            if state.status == outcome:
                event = self.events(attempt_id)[-1]
                if event.payload == payload:
                    return outcome
                raise ValueError("terminal outcome already recorded with different artifact")
            if state.status != "terminal_pending_validation":
                raise ValueError("attempt is not pending terminal validation")
            if outcome == "accepted":
                if self._accepted_count() >= self._target_accepted():
                    raise ValueError("accepted episode target already reached")
                existing = self._connection.execute(
                    "SELECT attempt_id FROM events WHERE event_type = 'accepted' "
                    "AND json_extract(payload_json, '$.artifact_id') = ? LIMIT 1", (artifact_id,)
                ).fetchone()
                if existing is not None:
                    raise ValueError("artifact is already accepted")
            self._append_event(outcome, attempt_id=attempt_id, payload=payload, at_ns=self._now())
            return outcome

    def stop_for_preemption(self, reason: str) -> None:
        """Compatibility alias for a resumable preemption pause."""

        self.pause_for_preemption(reason)

    def pause_for_preemption(self, reason: str) -> None:
        """Pause dispatch and explicitly preserve every in-flight retry decision."""

        reason = _require_identifier(reason, field="reason")
        with self._write():
            state = self._campaign_state()
            if state == "paused":
                return
            if state == "ended":
                raise ValueError("terminal campaign cannot be paused")
            now_ns = self._now()
            self._append_event(
                "campaign_paused", payload={"reason": reason, "epoch": self._campaign_epoch() + 1}, at_ns=now_ns,
            )
            for attempt in self.attempts():
                state = self._state_for_attempt(attempt.attempt_id)
                if state.status not in _LEASED_STATES or state.lease is None:
                    continue
                self._append_event(
                    "interrupted", attempt_id=attempt.attempt_id, lease_id=state.lease.lease_id,
                    worker_id=state.lease.worker_id, payload={"reason": reason}, at_ns=now_ns,
                )
                self._append_event(
                    "retryable", attempt_id=attempt.attempt_id, lease_id=state.lease.lease_id,
                    worker_id=state.lease.worker_id, payload={"reason": "preemption"}, at_ns=now_ns,
                )

    def resume_after_preemption(self, reason: str) -> None:
        """Append a new controller epoch so a replacement VM can issue retries."""

        reason = _require_identifier(reason, field="reason")
        with self._write():
            state = self._campaign_state()
            if state == "ended":
                raise ValueError("terminal campaign cannot be resumed")
            if state != "paused":
                raise ValueError("campaign is not paused")
            self._append_event(
                "campaign_resumed", payload={"reason": reason, "epoch": self._campaign_epoch()}, at_ns=self._now(),
            )

    def end_campaign(self, reason: str) -> None:
        """Record a true terminal end, distinct from a resumable preemption pause."""

        reason = _require_identifier(reason, field="reason")
        with self._write():
            if self._campaign_state() == "ended":
                return
            self._append_event("campaign_ended", payload={"reason": reason}, at_ns=self._now())

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS attempts (
                attempt_id TEXT PRIMARY KEY,
                schedule_index INTEGER NOT NULL UNIQUE,
                assignment_json TEXT NOT NULL
            );
            CREATE TRIGGER IF NOT EXISTS metadata_never_update
            BEFORE UPDATE ON metadata BEGIN SELECT RAISE(ABORT, 'campaign metadata is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS metadata_never_delete
            BEFORE DELETE ON metadata BEGIN SELECT RAISE(ABORT, 'campaign metadata is immutable'); END;
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                at_ns INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                attempt_id TEXT REFERENCES attempts(attempt_id),
                lease_id TEXT,
                worker_id TEXT,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS events_attempt_event ON events(attempt_id, event_id);
            CREATE TRIGGER IF NOT EXISTS events_never_update
            BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
            CREATE TRIGGER IF NOT EXISTS events_never_delete
            BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
            CREATE TRIGGER IF NOT EXISTS attempts_never_update
            BEFORE UPDATE ON attempts BEGIN SELECT RAISE(ABORT, 'attempts are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS attempts_never_delete
            BEFORE DELETE ON attempts BEGIN SELECT RAISE(ABORT, 'attempts are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS attempts_never_insert
            BEFORE INSERT ON attempts
            WHEN EXISTS (SELECT 1 FROM metadata WHERE key = 'matrix_bootstrap_complete' AND value = 'true')
            BEGIN SELECT RAISE(ABORT, 'attempts are immutable'); END;
            """
        )

    def _assert_schema_is_new_or_complete(self) -> None:
        rows = self._connection.execute(
            "SELECT type, name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
        ).fetchall()
        if not rows:
            return
        tables = {row["name"] for row in rows if row["type"] == "table"}
        triggers = {row["name"] for row in rows if row["type"] == "trigger"}
        required_tables = {"metadata", "attempts", "events"}
        required_triggers = {
            "metadata_never_update", "metadata_never_delete", "events_never_update", "events_never_delete",
            "attempts_never_update", "attempts_never_delete", "attempts_never_insert",
        }
        if tables != required_tables or triggers != required_triggers:
            raise ValueError("existing rollout ledger schema is corrupt or incompatible")
        expected_columns = {
            "metadata": (("key", "TEXT", 0, 1), ("value", "TEXT", 1, 0)),
            "attempts": (("attempt_id", "TEXT", 0, 1), ("schedule_index", "INTEGER", 1, 0), ("assignment_json", "TEXT", 1, 0)),
            "events": (
                ("event_id", "INTEGER", 0, 1), ("at_ns", "INTEGER", 1, 0), ("event_type", "TEXT", 1, 0),
                ("attempt_id", "TEXT", 0, 0), ("lease_id", "TEXT", 0, 0), ("worker_id", "TEXT", 0, 0),
                ("payload_json", "TEXT", 1, 0),
            ),
        }
        for table, expected in expected_columns.items():
            actual = tuple(
                (row["name"], str(row["type"]).upper(), row["notnull"], row["pk"])
                for row in self._connection.execute(f"PRAGMA table_info({table})").fetchall()
            )
            if actual != expected:
                raise ValueError("existing rollout ledger schema has incompatible columns")
        unique_schedule_indexes = [
            row["name"] for row in self._connection.execute("PRAGMA index_list(attempts)").fetchall()
            if row["unique"] and [column["name"] for column in self._connection.execute(f"PRAGMA index_info({row['name']})").fetchall()] == ["schedule_index"]
        ]
        if len(unique_schedule_indexes) != 1:
            raise ValueError("existing rollout ledger schema lacks unique schedule indexes")
        event_indexes = {
            row["name"] for row in self._connection.execute("PRAGMA index_list(events)").fetchall()
        }
        if event_indexes != {"events_attempt_event"}:
            raise ValueError("existing rollout ledger schema has incompatible event indexes")
        foreign_keys = self._connection.execute("PRAGMA foreign_key_list(events)").fetchall()
        if len(foreign_keys) != 1 or (foreign_keys[0]["table"], foreign_keys[0]["from"], foreign_keys[0]["to"]) != (
            "attempts", "attempt_id", "attempt_id",
        ):
            raise ValueError("existing rollout ledger schema has incompatible event foreign keys")
        trigger_rules = {
            "metadata_never_update": ("update", "metadata", "campaign metadata is immutable"),
            "metadata_never_delete": ("delete", "metadata", "campaign metadata is immutable"),
            "events_never_update": ("update", "events", "events are append-only"),
            "events_never_delete": ("delete", "events", "events are append-only"),
            "attempts_never_update": ("update", "attempts", "attempts are immutable"),
            "attempts_never_delete": ("delete", "attempts", "attempts are immutable"),
        }
        trigger_sql = {
            row["name"]: "".join(str(row["sql"]).lower().split())
            for row in self._connection.execute("SELECT name, sql FROM sqlite_master WHERE type = 'trigger'").fetchall()
        }
        for name, (operation, table, message) in trigger_rules.items():
            expected = "".join(f"before {operation} on {table} begin select raise(abort, '{message}'); end".split())
            if expected not in trigger_sql[name]:
                raise ValueError("existing rollout ledger schema has incompatible append-only trigger semantics")
        bootstrap_insert_rule = "".join(
            "before insert on attempts when exists (select 1 from metadata where key = "
            "'matrix_bootstrap_complete' and value = 'true') begin select raise(abort, 'attempts are immutable'); end".split()
        )
        if bootstrap_insert_rule not in trigger_sql["attempts_never_insert"]:
            raise ValueError("existing rollout ledger schema has incompatible bootstrap trigger semantics")

    def _normalize_matrix(self, matrix: list[Mapping[str, object]] | tuple[Mapping[str, object], ...]) -> tuple[Attempt, ...]:
        if not isinstance(matrix, (list, tuple)) or not matrix:
            raise ValueError("attempt_matrix must contain at least one assignment")
        attempts: list[Attempt] = []
        for index, assignment in enumerate(matrix):
            if not isinstance(assignment, Mapping) or not assignment:
                raise ValueError("each attempt assignment must be a non-empty mapping")
            assignment_data = json.loads(_canonical_json(dict(assignment)))
            attempt_id = sha256(_canonical_json({"schedule_index": index, "assignment": assignment_data}).encode("utf-8")).hexdigest()
            attempts.append(Attempt(attempt_id, index, assignment_data))
        return tuple(attempts)

    def _initialize_or_verify(self, *, max_attempts: int, target_accepted: int) -> None:
        with self._write():
            rows = self._connection.execute("SELECT key, value FROM metadata").fetchall()
            metadata = {row["key"]: row["value"] for row in rows}
            expected = {
                "matrix_sha256": self._matrix_sha256,
                "max_attempts": str(max_attempts),
                "target_accepted": str(target_accepted),
                "matrix_bootstrap_complete": "true",
            }
            if not metadata:
                self._connection.executemany(
                    "INSERT INTO metadata(key, value) VALUES (?, ?)",
                    tuple((key, value) for key, value in expected.items() if key != "matrix_bootstrap_complete"),
                )
                self._connection.executemany(
                    "INSERT INTO attempts(attempt_id, schedule_index, assignment_json) VALUES (?, ?, ?)",
                    [(attempt.attempt_id, attempt.schedule_index, _canonical_json(attempt.assignment)) for attempt in self._matrix],
                )
                self._connection.execute(
                    "INSERT INTO metadata(key, value) VALUES ('matrix_bootstrap_complete', 'true')"
                )
                return
            if metadata != expected:
                raise ValueError("attempt matrix or campaign limits are immutable")
            stored = self.attempts()
            if stored != self._matrix:
                raise ValueError("stored attempt matrix is immutable and differs from supplied matrix")

    @contextmanager
    def _write(self) -> Iterator[None]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def _now(self) -> int:
        now_ns = self._clock_ns()
        if not isinstance(now_ns, int) or isinstance(now_ns, bool) or not 0 <= now_ns <= MAX_SQLITE_INTEGER:
            raise ValueError("clock_ns must return a SQLite-safe non-negative integer")
        return now_ns

    @staticmethod
    def _require_duration(duration_ns: int) -> None:
        if not isinstance(duration_ns, int) or isinstance(duration_ns, bool) or duration_ns <= 0:
            raise ValueError("lease_duration_ns must be a positive integer")

    @staticmethod
    def _expires_at(now_ns: int, duration_ns: int) -> int:
        if duration_ns > MAX_SQLITE_INTEGER - now_ns:
            raise ValueError("lease_duration_ns exceeds a SQLite-safe expiry at the current clock")
        return now_ns + duration_ns

    def _event_from_row(self, row: sqlite3.Row) -> LedgerEvent:
        payload = _decode_json(row["payload_json"])
        assert isinstance(payload, dict)
        return LedgerEvent(
            row["event_id"], row["at_ns"], row["event_type"], row["attempt_id"], row["lease_id"], row["worker_id"], payload,
        )

    def _attempt(self, attempt_id: str) -> Attempt:
        row = self._connection.execute(
            "SELECT attempt_id, schedule_index, assignment_json FROM attempts WHERE attempt_id = ?", (attempt_id,)
        ).fetchone()
        if row is None:
            raise ValueError("unknown attempt_id")
        return Attempt(row["attempt_id"], row["schedule_index"], _decode_json(row["assignment_json"]))

    def _state_for_attempt(self, attempt_id: str) -> _AttemptState:
        attempt = self._attempt(attempt_id)
        status = "pending"
        lease: Lease | None = None
        for event in self.events(attempt_id):
            if event.event_type == "leased":
                expires_at_ns = event.payload.get("expires_at_ns")
                assert isinstance(expires_at_ns, int)
                status = "leased"
                lease = Lease(attempt, event.lease_id or "", event.worker_id or "", expires_at_ns)
            elif event.event_type == "heartbeat" and status == "leased" and lease is not None and event.lease_id == lease.lease_id:
                expires_at_ns = event.payload.get("expires_at_ns")
                assert isinstance(expires_at_ns, int)
                lease = Lease(attempt, lease.lease_id, lease.worker_id, expires_at_ns)
            elif event.event_type in _STATE_EVENTS:
                status = event.event_type
                lease = None
        return _AttemptState(status, lease)

    def _require_active_lease(self, state: _AttemptState, worker_id: str, lease_id: str) -> Lease:
        if state.status != "leased" or state.lease is None:
            raise ValueError("attempt has no active lease")
        if state.lease.worker_id != worker_id:
            raise ValueError("lease belongs to a different worker")
        if state.lease.lease_id != lease_id:
            raise ValueError("lease_id does not match active lease")
        return state.lease

    def _expire_leases(self, now_ns: int) -> None:
        for attempt in self.attempts():
            state = self._state_for_attempt(attempt.attempt_id)
            if state.status != "leased" or state.lease is None or now_ns < state.lease.expires_at_ns:
                continue
            self._append_event(
                "lease_expired", attempt_id=attempt.attempt_id, lease_id=state.lease.lease_id,
                worker_id=state.lease.worker_id, payload={"expires_at_ns": state.lease.expires_at_ns}, at_ns=now_ns,
            )
            self._append_event(
                "retryable", attempt_id=attempt.attempt_id, lease_id=state.lease.lease_id,
                worker_id=state.lease.worker_id, payload={"reason": "lease_expired"}, at_ns=now_ns,
            )

    def _append_event(
        self, event_type: str, *, attempt_id: str | None = None, lease_id: str | None = None,
        worker_id: str | None = None, payload: Mapping[str, object] | None = None, at_ns: int,
    ) -> None:
        self._connection.execute(
            "INSERT INTO events(at_ns, event_type, attempt_id, lease_id, worker_id, payload_json) VALUES (?, ?, ?, ?, ?, ?)",
            (at_ns, event_type, attempt_id, lease_id, worker_id, _canonical_json(dict(payload or {}))),
        )

    def _lease_sequence(self, attempt_id: str) -> int:
        return int(self._connection.execute(
            "SELECT COUNT(*) FROM events WHERE attempt_id = ? AND event_type = 'leased'", (attempt_id,)
        ).fetchone()[0])

    def _issued_lease_count(self) -> int:
        return int(self._connection.execute("SELECT COUNT(*) FROM events WHERE event_type = 'leased'").fetchone()[0])

    def _accepted_count(self) -> int:
        return int(self._connection.execute("SELECT COUNT(*) FROM events WHERE event_type = 'accepted'").fetchone()[0])

    def _max_attempts(self) -> int:
        return int(self._connection.execute("SELECT value FROM metadata WHERE key = 'max_attempts'").fetchone()[0])

    def _target_accepted(self) -> int:
        return int(self._connection.execute("SELECT value FROM metadata WHERE key = 'target_accepted'").fetchone()[0])

    def _active_lease_for_worker(self, worker_id: str) -> Lease | None:
        for attempt in self.attempts():
            state = self._state_for_attempt(attempt.attempt_id)
            if state.status == "leased" and state.lease is not None and state.lease.worker_id == worker_id:
                return state.lease
        return None

    def _campaign_state(self) -> str:
        row = self._connection.execute(
            "SELECT event_type FROM events WHERE event_type IN ('campaign_paused', 'campaign_resumed', 'campaign_ended') "
            "ORDER BY event_id DESC LIMIT 1"
        ).fetchone()
        if row is None or row["event_type"] == "campaign_resumed":
            return "active"
        if row["event_type"] == "campaign_paused":
            return "paused"
        return "ended"

    def _campaign_epoch(self) -> int:
        return int(self._connection.execute(
            "SELECT COUNT(*) FROM events WHERE event_type = 'campaign_paused'"
        ).fetchone()[0])
