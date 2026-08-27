from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest


def _matrix(size: int = 4) -> list[dict[str, object]]:
    return [
        {
            "garment": f"Top_Long_Seen_{index}",
            "seed": 100 + index,
            "difficulty": "seen" if index % 2 == 0 else "hard_state",
        }
        for index in range(size)
    ]


@pytest.fixture
def clock() -> list[int]:
    return [1_000]


@pytest.fixture
def ledger(tmp_path, clock):
    from lehome.flywheel.task_ledger import TaskLedger

    result = TaskLedger(
        tmp_path / "rollouts.sqlite3",
        attempt_matrix=_matrix(),
        max_attempts=4,
        target_accepted=2,
        clock_ns=lambda: clock[0],
    )
    yield result
    result.close()


def test_creates_deterministic_immutable_attempt_matrix_in_wal_mode(tmp_path) -> None:
    from lehome.flywheel.task_ledger import TaskLedger

    first = TaskLedger(tmp_path / "first.sqlite3", attempt_matrix=_matrix(), clock_ns=lambda: 1)
    second = TaskLedger(tmp_path / "second.sqlite3", attempt_matrix=_matrix(), clock_ns=lambda: 1)
    try:
        assert [attempt.attempt_id for attempt in first.attempts()] == [
            attempt.attempt_id for attempt in second.attempts()
        ]
        assert [attempt.schedule_index for attempt in first.attempts()] == [0, 1, 2, 3]
        assert first.journal_mode() == "wal"
        with pytest.raises(ValueError, match="immutable"):
            TaskLedger(tmp_path / "first.sqlite3", attempt_matrix=list(reversed(_matrix())), clock_ns=lambda: 1)
    finally:
        first.close()
        second.close()


def test_terminal_outcome_completion_counts_policy_rejections_but_not_infrastructure_retries(tmp_path) -> None:
    from lehome.flywheel.task_ledger import TaskLedger

    rows = _matrix(4)
    ledger = TaskLedger(
        tmp_path / "terminal-outcomes.sqlite3", attempt_matrix=rows,
        max_attempts=5, target_accepted=3, completion_metric="terminal_outcomes", clock_ns=lambda: 1,
    )
    try:
        first = ledger.lease_next("worker-a", lease_duration_ns=10**18)
        assert first is not None
        assert ledger.record_interrupted("worker-a", first.attempt.attempt_id, first.lease_id, "isaac_restart") == "retryable"
        retry = ledger.lease_next("worker-b", lease_duration_ns=100)
        assert retry is not None
        assert retry.attempt == first.attempt
        assert retry.attempt.assignment == rows[0]
        assert ledger.completion_count == 0

        assert ledger.reject_attempt("worker-b", retry.attempt.attempt_id, retry.lease_id, reason="policy_failure") == "rejected"
        second = ledger.lease_next("worker-c", lease_duration_ns=100)
        assert second is not None
        assert ledger.record_infrastructure_abort("worker-c", second.attempt.attempt_id, second.lease_id, reason="isaac_timeout") == "infrastructure_abort"
        assert ledger.completion_count == 1
        third = ledger.lease_next("worker-d", lease_duration_ns=100)
        assert third is not None
        assert ledger.record_terminal("worker-d", third.attempt.attempt_id, third.lease_id, "raw-third") == "terminal_pending_validation"
        assert ledger.validate_terminal(third.attempt.attempt_id, "accepted", artifact_id="accepted-third") == "accepted"
        fourth = ledger.lease_next("worker-e", lease_duration_ns=100)
        assert fourth is not None
        assert ledger.reject_attempt("worker-e", fourth.attempt.attempt_id, fourth.lease_id, reason="policy_failure") == "rejected"

        assert ledger.accepted_count() == 1
        assert ledger.terminal_outcome_count() == 3
        assert ledger.completion_count == 3
        assert ledger.is_terminal
        assert ledger.lease_next("worker-e", lease_duration_ns=100) is None
    finally:
        ledger.close()


def test_completion_metric_and_all_immutable_metadata_reject_mismatched_reopen(tmp_path) -> None:
    from lehome.flywheel.task_ledger import TaskLedger

    database = tmp_path / "immutable-metadata.sqlite3"
    first = TaskLedger(
        database, attempt_matrix=_matrix(3), max_attempts=4, target_accepted=3,
        completion_metric="terminal_outcomes", clock_ns=lambda: 1,
    )
    first.close()
    for kwargs in (
        {"completion_metric": "accepted_successes"},
        {"max_attempts": 3},
        {"target_accepted": 2},
    ):
        options = {"max_attempts": 4, "target_accepted": 3, "completion_metric": "terminal_outcomes"}
        options.update(kwargs)
        with pytest.raises(ValueError, match="immutable"):
            TaskLedger(database, attempt_matrix=_matrix(3), clock_ns=lambda: 1, **options)


def test_legacy_accepted_success_completion_cap_is_unchanged(tmp_path) -> None:
    from lehome.flywheel.task_ledger import MAX_ACCEPTED_EPISODES, TaskLedger

    with pytest.raises(ValueError, match=str(MAX_ACCEPTED_EPISODES)):
        TaskLedger(tmp_path / "legacy-cap.sqlite3", attempt_matrix=_matrix(1), target_accepted=151)


def test_legacy_metadata_backfills_only_accepted_successes_metric(tmp_path) -> None:
    from lehome.flywheel.task_ledger import TaskLedger

    database = tmp_path / "legacy.sqlite3"
    ledger = TaskLedger(database, attempt_matrix=_matrix(1), max_attempts=1, target_accepted=1)
    ledger.close()
    connection = sqlite3.connect(database)
    connection.execute("DROP TRIGGER metadata_never_delete")
    connection.execute("DELETE FROM metadata WHERE key = 'completion_metric'")
    connection.execute("CREATE TRIGGER metadata_never_delete BEFORE DELETE ON metadata BEGIN SELECT RAISE(ABORT, 'campaign metadata is immutable'); END;")
    connection.commit()
    connection.close()

    reopened = TaskLedger(database, attempt_matrix=_matrix(1), max_attempts=1, target_accepted=1)
    try:
        assert reopened.completion_metric() == "accepted_successes"
    finally:
        reopened.close()
    with pytest.raises(ValueError, match="immutable"):
        TaskLedger(
            database, attempt_matrix=_matrix(1), max_attempts=1, target_accepted=1,
            completion_metric="terminal_outcomes",
        )


def test_exact_terminal_handoff_with_malformed_artifact_retries_the_same_assignment(tmp_path) -> None:
    from lehome.flywheel.task_ledger import TaskLedger

    ledger = TaskLedger(
        tmp_path / "exact-retry.sqlite3", attempt_matrix=_matrix(1), max_attempts=2,
        target_accepted=1, completion_metric="terminal_outcomes",
    )
    try:
        first = ledger.lease_next("worker-a", lease_duration_ns=10**18)
        assert first is not None
        assert ledger.record_terminal("worker-a", first.attempt.attempt_id, first.lease_id, "raw-a") == "terminal_pending_validation"
        assert ledger.retry_terminal_infrastructure(first.attempt.attempt_id, reason="malformed_artifact") == "retryable"
        retry = ledger.lease_next("worker-b", lease_duration_ns=10**18)
        assert retry is not None and retry.attempt == first.attempt
        assert ledger.completion_count == 0
        assert len([event for event in ledger.events() if event.event_type == "leased"]) == 2
    finally:
        ledger.close()


def test_exact_fidelity_abort_never_retries_or_counts_as_a_terminal_outcome(tmp_path) -> None:
    from lehome.flywheel.task_ledger import TaskLedger

    ledger = TaskLedger(
        tmp_path / "fidelity-stop.sqlite3", attempt_matrix=_matrix(1), max_attempts=2,
        target_accepted=1, completion_metric="terminal_outcomes", clock_ns=lambda: 1,
    )
    try:
        lease = ledger.lease_next("worker", lease_duration_ns=100)
        assert lease is not None
        assert ledger.record_fidelity_abort(
            "worker", lease.attempt.attempt_id, lease.lease_id, session_id="session", generation=1,
            fidelity_code="missing_cloth",
            fidelity={"missing_cloth": True, "cloth_flight": False, "nonfinite_cloth_state": False,
                      "safety_failure": False, "monitor_active": True, "monitor_observed": True},
            runtime={"simulation_device": "cpu", "cloth_device": "cpu", "renderer_device": "cuda:0",
                     "camera_device": "cuda:0", "policy_device": "cuda:0"},
        ) == "infrastructure_abort"
        assert ledger.completion_count == 0
        assert ledger.lease_next("retry-worker", lease_duration_ns=100) is None
    finally:
        ledger.close()


@pytest.mark.parametrize("target", [100, 300])
def test_terminal_outcome_partitions_complete_on_a_mix_of_successes_and_policy_failures(tmp_path, target: int) -> None:
    from lehome.flywheel.task_ledger import TaskLedger

    ledger = TaskLedger(
        tmp_path / f"terminal-{target}.sqlite3", attempt_matrix=_matrix(target), max_attempts=target,
        target_accepted=target, completion_metric="terminal_outcomes", clock_ns=lambda: 1,
    )
    try:
        for index in range(target):
            lease = ledger.lease_next(f"worker-{index}", lease_duration_ns=100)
            assert lease is not None
            if index % 2:
                assert ledger.reject_attempt(f"worker-{index}", lease.attempt.attempt_id, lease.lease_id, reason="policy_failure") == "rejected"
            else:
                assert ledger.record_terminal(f"worker-{index}", lease.attempt.attempt_id, lease.lease_id, f"raw-{index}") == "terminal_pending_validation"
                assert ledger.validate_terminal(lease.attempt.attempt_id, "accepted", artifact_id=f"accepted-{index}") == "accepted"
        assert ledger.completion_count == target
        assert ledger.terminal_outcome_count() == target
        assert ledger.is_terminal
    finally:
        ledger.close()


def test_lease_immediately_assigns_next_pending_attempt_without_wave_barrier(ledger) -> None:
    first = ledger.lease_next("worker-a", lease_duration_ns=100)
    second = ledger.lease_next("worker-b", lease_duration_ns=100)

    assert first is not None and second is not None
    assert first.attempt.schedule_index == 0
    assert second.attempt.schedule_index == 1
    assert first.lease_id != second.lease_id
    assert ledger.status(first.attempt.attempt_id) == "leased"


def test_duplicate_request_from_a_worker_returns_its_existing_unexpired_lease(ledger) -> None:
    first = ledger.lease_next("worker-a", lease_duration_ns=100)
    assert first is not None

    duplicate = ledger.lease_next("worker-a", lease_duration_ns=100)

    assert duplicate == first
    assert ledger.events(first.attempt.attempt_id) == ledger.events()
    assert len([event for event in ledger.events() if event.event_type == "leased"]) == 1


def test_heartbeat_renews_matching_worker_lease_with_injected_clock(ledger, clock) -> None:
    lease = ledger.lease_next("worker-a", lease_duration_ns=100)
    assert lease is not None
    clock[0] = 1_050

    renewed = ledger.heartbeat("worker-a", lease.attempt.attempt_id, lease.lease_id, lease_duration_ns=100)

    assert renewed.expires_at_ns == 1_150
    assert ledger.events(lease.attempt.attempt_id)[-1].event_type == "heartbeat"
    with pytest.raises(ValueError, match="worker"):
        ledger.heartbeat("worker-b", lease.attempt.attempt_id, lease.lease_id, lease_duration_ns=100)


def test_record_fidelity_abort_writes_typed_terminal_payload(ledger) -> None:
    lease = ledger.lease_next("worker-a", lease_duration_ns=100)
    assert lease is not None
    fidelity = {
        "missing_cloth": True, "cloth_flight": False,
        "nonfinite_cloth_state": False, "safety_failure": False,
        "monitor_active": True, "monitor_observed": True,
    }
    assert ledger.record_fidelity_abort(
        "worker-a", lease.attempt.attempt_id, lease.lease_id,
        session_id="session-a", generation=3, fidelity_code="missing_cloth", fidelity=fidelity,
        runtime={"simulation_device": "cpu", "cloth_device": "cpu", "renderer_device": "cuda:0", "camera_device": "cuda:0", "policy_device": "cuda:0"},
    ) == "infrastructure_abort"

    event = ledger.events(lease.attempt.attempt_id)[-1]
    assert event.event_type == "infrastructure_abort"
    assert event.payload == {
        "failure_class": "fidelity", "fidelity_code": "missing_cloth", "fidelity": fidelity,
        "lease_id": lease.lease_id, "worker_id": "worker-a",
        "session_id": "session-a", "generation": 3,
        "runtime": {"simulation_device": "cpu", "cloth_device": "cpu", "renderer_device": "cuda:0", "camera_device": "cuda:0", "policy_device": "cuda:0"},
    }


@pytest.mark.parametrize("field", ["monitor_active", "monitor_observed"])
def test_record_fidelity_abort_refuses_unobserved_monitors_without_writing(ledger, field: str) -> None:
    lease = ledger.lease_next("worker-a", lease_duration_ns=100)
    assert lease is not None
    fidelity = {
        "missing_cloth": True, "cloth_flight": False,
        "nonfinite_cloth_state": False, "safety_failure": False,
        "monitor_active": True, "monitor_observed": True,
    }
    fidelity[field] = False
    before = ledger.events(lease.attempt.attempt_id)

    with pytest.raises(ValueError, match="fidelity"):
        ledger.record_fidelity_abort(
            "worker-a", lease.attempt.attempt_id, lease.lease_id,
            session_id="session-a", generation=1, fidelity_code="missing_cloth", fidelity=fidelity,
            runtime={"simulation_device": "cpu", "cloth_device": "cpu", "renderer_device": "cuda:0", "camera_device": "cuda:0", "policy_device": "cuda:0"},
        )

    assert ledger.events(lease.attempt.attempt_id) == before

def test_lease_and_heartbeat_reject_expiry_past_sqlite_signed_integer_without_writing_events(tmp_path) -> None:
    from lehome.flywheel.task_ledger import TaskLedger

    sqlite_max = 2**63 - 1
    clock = [sqlite_max - 5]
    ledger = TaskLedger(tmp_path / "clock-bound.sqlite3", attempt_matrix=_matrix(), clock_ns=lambda: clock[0])
    try:
        with pytest.raises(ValueError, match="SQLite-safe"):
            ledger.lease_next("worker-a", lease_duration_ns=6)
        assert ledger.events() == ()

        lease = ledger.lease_next("worker-a", lease_duration_ns=1)
        assert lease is not None
        assert lease.expires_at_ns == sqlite_max - 4
        event_count = len(ledger.events())
        clock[0] = sqlite_max - 1
        with pytest.raises(ValueError, match="SQLite-safe"):
            ledger.heartbeat("worker-a", lease.attempt.attempt_id, lease.lease_id, lease_duration_ns=2)
        assert len(ledger.events()) == event_count
    finally:
        ledger.close()


def test_expired_lease_appends_retry_history_and_releases_same_immutable_attempt(ledger, clock) -> None:
    first = ledger.lease_next("worker-a", lease_duration_ns=10)
    assert first is not None
    clock[0] = 1_010

    retry = ledger.lease_next("worker-b", lease_duration_ns=10)

    assert retry is not None
    assert retry.attempt.attempt_id == first.attempt.attempt_id
    assert retry.lease_id != first.lease_id
    assert [event.event_type for event in ledger.events(first.attempt.attempt_id)] == [
        "leased", "lease_expired", "retryable", "leased"
    ]


def test_terminal_and_validation_are_idempotent_and_do_not_duplicate_accepted_artifact(ledger) -> None:
    lease = ledger.lease_next("worker-a", lease_duration_ns=100)
    assert lease is not None

    assert ledger.record_terminal("worker-a", lease.attempt.attempt_id, lease.lease_id, "raw-1") == "terminal_pending_validation"
    assert ledger.record_terminal("worker-a", lease.attempt.attempt_id, lease.lease_id, "raw-1") == "terminal_pending_validation"
    assert ledger.validate_terminal(lease.attempt.attempt_id, "accepted", artifact_id="artifact-1") == "accepted"
    assert ledger.validate_terminal(lease.attempt.attempt_id, "accepted", artifact_id="artifact-1") == "accepted"
    assert ledger.status(lease.attempt.attempt_id) == "accepted"
    assert [event.event_type for event in ledger.events(lease.attempt.attempt_id)] == [
        "leased", "terminal_pending_validation", "accepted"
    ]
    other = ledger.lease_next("worker-b", lease_duration_ns=100)
    assert other is not None
    ledger.record_terminal("worker-b", other.attempt.attempt_id, other.lease_id, "raw-2")
    with pytest.raises(ValueError, match="already accepted"):
        ledger.validate_terminal(other.attempt.attempt_id, "accepted", artifact_id="artifact-1")


def test_raw_artifact_id_cannot_be_reused_by_a_different_terminal_attempt(ledger) -> None:
    first = ledger.lease_next("worker-a", lease_duration_ns=100)
    second = ledger.lease_next("worker-b", lease_duration_ns=100)
    assert first is not None and second is not None
    ledger.record_terminal("worker-a", first.attempt.attempt_id, first.lease_id, "raw-shared")

    with pytest.raises(ValueError, match="raw artifact"):
        ledger.record_terminal("worker-b", second.attempt.attempt_id, second.lease_id, "raw-shared")


def test_campaign_stops_leasing_after_target_or_maximum_attempts(tmp_path) -> None:
    from lehome.flywheel.task_ledger import TaskLedger

    by_target = TaskLedger(tmp_path / "target.sqlite3", attempt_matrix=_matrix(3), max_attempts=3, target_accepted=1, clock_ns=lambda: 1)
    try:
        lease = by_target.lease_next("worker-a", lease_duration_ns=10)
        assert lease is not None
        by_target.record_terminal("worker-a", lease.attempt.attempt_id, lease.lease_id, "raw")
        by_target.validate_terminal(lease.attempt.attempt_id, "accepted", artifact_id="accepted")
        assert by_target.lease_next("worker-b", lease_duration_ns=10) is None
    finally:
        by_target.close()

    by_cap = TaskLedger(tmp_path / "cap.sqlite3", attempt_matrix=_matrix(2), max_attempts=2, target_accepted=2, clock_ns=lambda: 1)
    try:
        assert by_cap.lease_next("worker-a", lease_duration_ns=10) is not None
        assert by_cap.lease_next("worker-b", lease_duration_ns=10) is not None
        assert by_cap.lease_next("worker-c", lease_duration_ns=10) is None
    finally:
        by_cap.close()
    with pytest.raises(ValueError, match="400"):
        TaskLedger(tmp_path / "too-many.sqlite3", attempt_matrix=_matrix(401), clock_ns=lambda: 1)


def test_lease_next_can_bind_a_fresh_worker_to_one_garment(tmp_path) -> None:
    from lehome.flywheel.task_ledger import TaskLedger

    matrix = [
        {"attempt_id": "top-0-a", "garment_name": "Top_Long_Seen_0", "seed": 1},
        {"attempt_id": "top-1-a", "garment_name": "Top_Long_Seen_1", "seed": 2},
        {"attempt_id": "top-0-b", "garment_name": "Top_Long_Seen_0", "seed": 3},
    ]
    ledger = TaskLedger(
        tmp_path / "garment-affinity.sqlite3",
        attempt_matrix=matrix,
        max_attempts=3,
        target_accepted=3,
        clock_ns=lambda: 1,
    )
    try:
        first = ledger.lease_next(
            "garment-0-worker",
            lease_duration_ns=10,
            assignment_filter={"garment_name": "Top_Long_Seen_0"},
        )
        second = ledger.lease_next(
            "garment-1-worker",
            lease_duration_ns=10,
            assignment_filter={"garment_name": "Top_Long_Seen_1"},
        )

        assert first is not None and first.attempt.assignment["attempt_id"] == "top-0-a"
        assert second is not None and second.attempt.assignment["attempt_id"] == "top-1-a"
    finally:
        ledger.close()


def test_validation_cannot_overshoot_the_accepted_episode_target(tmp_path) -> None:
    from lehome.flywheel.task_ledger import TaskLedger

    target = TaskLedger(tmp_path / "target-cap.sqlite3", attempt_matrix=_matrix(2), target_accepted=1, clock_ns=lambda: 1)
    try:
        first = target.lease_next("worker-a", lease_duration_ns=10)
        second = target.lease_next("worker-b", lease_duration_ns=10)
        assert first is not None and second is not None
        target.record_terminal("worker-a", first.attempt.attempt_id, first.lease_id, "raw-1")
        target.record_terminal("worker-b", second.attempt.attempt_id, second.lease_id, "raw-2")
        target.validate_terminal(first.attempt.attempt_id, "accepted", artifact_id="accepted-1")

        with pytest.raises(ValueError, match="target"):
            target.validate_terminal(second.attempt.attempt_id, "accepted", artifact_id="accepted-2")
    finally:
        target.close()


def test_preemption_pauses_new_leases_then_resume_retries_the_same_attempt(ledger) -> None:
    lease = ledger.lease_next("worker-a", lease_duration_ns=100)
    assert lease is not None

    ledger.stop_for_preemption("spot-instance-interrupted")

    assert ledger.is_stopped is True
    assert ledger.lease_next("worker-b", lease_duration_ns=100) is None
    assert [event.event_type for event in ledger.events(lease.attempt.attempt_id)] == [
        "leased", "interrupted", "retryable"
    ]
    assert ledger.status(lease.attempt.attempt_id) == "retryable"
    ledger.resume_after_preemption("replacement-vm-ready")

    retry = ledger.lease_next("worker-b", lease_duration_ns=100)

    assert retry is not None
    assert retry.attempt.attempt_id == lease.attempt.attempt_id
    assert retry.lease_id != lease.lease_id
    assert ledger.is_stopped is False
    assert [event.event_type for event in ledger.events() if event.attempt_id is None] == [
        "campaign_paused", "campaign_resumed"
    ]
    with pytest.raises(ValueError, match="not paused"):
        ledger.resume_after_preemption("duplicate-replacement")
    assert [event.event_type for event in ledger.events() if event.attempt_id is None] == [
        "campaign_paused", "campaign_resumed"
    ]


def test_transitions_use_begin_immediate_and_events_are_append_only(ledger) -> None:
    statements: list[str] = []
    ledger._connection.set_trace_callback(statements.append)
    lease = ledger.lease_next("worker-a", lease_duration_ns=10)
    assert lease is not None
    assert "BEGIN IMMEDIATE" in statements
    audit = ledger.audit_connection()
    try:
        event_rows = audit.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        with pytest.raises(sqlite3.OperationalError, match="readonly|read-only"):
            audit.execute("DELETE FROM events")
        with pytest.raises(sqlite3.OperationalError, match="readonly|read-only"):
            audit.execute("UPDATE attempts SET schedule_index = 99")
    finally:
        audit.close()
    ledger.heartbeat("worker-a", lease.attempt.attempt_id, lease.lease_id, lease_duration_ns=10)
    audit = ledger.audit_connection()
    try:
        assert audit.execute("SELECT COUNT(*) FROM events").fetchone()[0] == event_rows + 1
    finally:
        audit.close()


def test_concurrent_controller_calls_issue_distinct_leases_without_sqlite_thread_errors(ledger) -> None:
    with ThreadPoolExecutor(max_workers=2) as executor:
        leases = list(executor.map(lambda worker: ledger.lease_next(worker, lease_duration_ns=100), ("worker-a", "worker-b")))

    assert all(lease is not None for lease in leases)
    assert {lease.attempt.attempt_id for lease in leases if lease is not None} == {
        attempt.attempt_id for attempt in ledger.attempts()[:2]
    }


def test_refuses_to_initialize_when_sqlite_does_not_enter_wal_mode(tmp_path, monkeypatch) -> None:
    from lehome.flywheel import task_ledger

    actual_connect = task_ledger.sqlite3.connect

    class _Cursor:
        def fetchone(self):
            return ("delete",)

    class _Connection:
        def __init__(self, actual) -> None:
            self.actual = actual

        def execute(self, statement, parameters=()):
            if statement == "PRAGMA journal_mode = WAL":
                return _Cursor()
            return self.actual.execute(statement, parameters)

        def __getattr__(self, name):
            return getattr(self.actual, name)

    connections = []

    def connect(*args, **kwargs):
        connection = _Connection(actual_connect(*args, **kwargs))
        connections.append(connection)
        return connection

    monkeypatch.setattr(task_ledger.sqlite3, "connect", connect)
    with pytest.raises(RuntimeError, match="WAL"):
        task_ledger.TaskLedger(tmp_path / "not-wal.sqlite3", attempt_matrix=_matrix(), clock_ns=lambda: 1)
    connections[0].close()


def test_refuses_to_initialize_a_partially_corrupt_existing_ledger_schema(tmp_path) -> None:
    from lehome.flywheel.task_ledger import TaskLedger

    database = tmp_path / "corrupt.sqlite3"
    connection = sqlite3.connect(database)
    try:
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(ValueError, match="corrupt|schema"):
        TaskLedger(database, attempt_matrix=_matrix(), clock_ns=lambda: 1)


def test_refuses_existing_schema_with_a_named_but_no_op_append_only_trigger(tmp_path) -> None:
    from lehome.flywheel.task_ledger import TaskLedger

    database = tmp_path / "no-op-trigger.sqlite3"
    ledger = TaskLedger(database, attempt_matrix=_matrix(), clock_ns=lambda: 1)
    ledger.close()
    connection = sqlite3.connect(database)
    try:
        connection.execute("DROP TRIGGER events_never_update")
        connection.execute("CREATE TRIGGER events_never_update BEFORE UPDATE ON events BEGIN SELECT 1; END")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(ValueError, match="corrupt|schema"):
        TaskLedger(database, attempt_matrix=_matrix(), clock_ns=lambda: 1)


def test_attempt_table_update_and_delete_triggers_block_a_writable_external_connection(tmp_path) -> None:
    from lehome.flywheel.task_ledger import TaskLedger

    database = tmp_path / "trigger-enforced.sqlite3"
    ledger = TaskLedger(database, attempt_matrix=_matrix(), clock_ns=lambda: 1)
    try:
        attempt_id = ledger.attempts()[0].attempt_id
        writer = sqlite3.connect(database)
        try:
            with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                writer.execute("UPDATE attempts SET schedule_index = 99 WHERE attempt_id = ?", (attempt_id,))
            with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                writer.execute("DELETE FROM attempts WHERE attempt_id = ?", (attempt_id,))
        finally:
            writer.close()
    finally:
        ledger.close()


def test_attempt_table_insert_trigger_blocks_external_schedule_injection_after_bootstrap(tmp_path) -> None:
    from lehome.flywheel.task_ledger import TaskLedger

    database = tmp_path / "insert-trigger-enforced.sqlite3"
    ledger = TaskLedger(database, attempt_matrix=_matrix(), clock_ns=lambda: 1)
    try:
        baseline = ledger.attempts()
        writer = sqlite3.connect(database)
        try:
            with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                writer.execute(
                    "INSERT INTO attempts(attempt_id, schedule_index, assignment_json) VALUES (?, ?, ?)",
                    ("f" * 64, 99, '{"garment":"injected","seed":999}'),
                )
        finally:
            writer.close()
        assert ledger.attempts() == baseline
        lease = ledger.lease_next("worker-a", lease_duration_ns=10)
        assert lease is not None
        assert lease.attempt.attempt_id != "f" * 64
    finally:
        ledger.close()


def test_reject_attempt_does_not_remain_retryable(ledger) -> None:
    lease = ledger.lease_next("worker-a", lease_duration_ns=100)
    assert lease is not None
    assert ledger.reject_attempt("worker-a", lease.attempt.attempt_id, lease.lease_id, reason="ValueError") == "rejected"
    assert ledger.status(lease.attempt.attempt_id) == "rejected"
    next_lease = ledger.lease_next("worker-b", lease_duration_ns=100)
    assert next_lease is not None
    assert next_lease.attempt.attempt_id != lease.attempt.attempt_id
    # rejected attempt itself must not be leased again
    statuses = [ledger.status(attempt.attempt_id) for attempt in ledger.attempts() if attempt.attempt_id == lease.attempt.attempt_id]
    assert statuses == ["rejected"]


def test_preparation_timeout_is_an_atomic_distinct_infrastructure_abort_across_reopen(tmp_path) -> None:
    from lehome.flywheel.task_ledger import TaskLedger

    database = tmp_path / "infrastructure-abort.sqlite3"
    first = TaskLedger(database, attempt_matrix=_matrix(2), max_attempts=2, target_accepted=1, clock_ns=lambda: 1)
    try:
        lease = first.lease_next("worker-a", lease_duration_ns=100)
        assert lease is not None
        assert first.record_infrastructure_abort(
            "worker-a", lease.attempt.attempt_id, lease.lease_id, reason="preparation_timeout",
        ) == "infrastructure_abort"
        assert first.status(lease.attempt.attempt_id) == "infrastructure_abort"
        assert [event.event_type for event in first.events(lease.attempt.attempt_id)] == [
            "leased", "interrupted", "terminal_pending_validation", "infrastructure_abort",
        ]
        audit = first.audit_connection()
        try:
            rows = audit.execute(
                "SELECT event_type, payload_json FROM events WHERE attempt_id = ? ORDER BY event_id",
                (lease.attempt.attempt_id,),
            ).fetchall()
            assert [row["event_type"] for row in rows][-1] == "infrastructure_abort"
            assert "preparation_timeout" in rows[-1]["payload_json"]
        finally:
            audit.close()
    finally:
        first.close()

    reopened = TaskLedger(database, attempt_matrix=_matrix(2), max_attempts=2, target_accepted=1, clock_ns=lambda: 1)
    try:
        assert reopened.status(lease.attempt.attempt_id) == "infrastructure_abort"
        next_lease = reopened.lease_next("worker-b", lease_duration_ns=100)
        assert next_lease is not None
        assert next_lease.attempt.attempt_id != lease.attempt.attempt_id
    finally:
        reopened.close()


def test_an_active_broken_first_attempt_cannot_retry_storm_or_starve_later_rows(tmp_path) -> None:
    from lehome.flywheel.task_ledger import TaskLedger

    ledger = TaskLedger(
        tmp_path / "broken-first.sqlite3", attempt_matrix=_matrix(2), max_attempts=2, target_accepted=2,
        clock_ns=lambda: 1,
    )
    try:
        broken = ledger.lease_next("worker-a", lease_duration_ns=100)
        assert broken is not None
        # A worker that re-raises a deterministic execution failure leaves its
        # lease active for bounded supervisor handling; it must not append a
        # retryable transition that repeatedly wins earliest-row selection.
        later = ledger.lease_next("worker-b", lease_duration_ns=100)
        assert later is not None
        assert later.attempt.schedule_index == 1
        assert [event.event_type for event in ledger.events(broken.attempt.attempt_id)] == ["leased"]
        assert len([event for event in ledger.events() if event.event_type == "leased"]) == 2
    finally:
        ledger.close()
