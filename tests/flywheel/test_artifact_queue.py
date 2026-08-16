"""Bounded finalization queue: validation, ledger outcomes, backpressure."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lehome.flywheel.artifact_queue import (
    ArtifactFinalizationQueue,
    QueueFullError,
)
from lehome.flywheel.task_ledger import TaskLedger


ATTEMPT_MATRIX = (
    {"episode": "episode-001", "category": "top-long"},
    {"episode": "episode-002", "category": "top-long"},
    {"episode": "episode-003", "category": "pant-short"},
)
LEASE_NS = 10**18
WORKERS = ("worker-1", "worker-2", "worker-3")


@pytest.fixture()
def ledger(tmp_path):
    database = tmp_path / "ledger.db"
    task_ledger = TaskLedger(database, attempt_matrix=ATTEMPT_MATRIX, max_attempts=10, target_accepted=3)
    yield task_ledger
    task_ledger.close()


def handoff_terminal(
    ledger: TaskLedger, run_root: Path, *, attempt_index: int, success: bool = True, corrupt: bool = False,
) -> tuple[str, str, str, Path]:
    """Simulate one worker: lease, write a raw terminal episode, close terminal.

    The ledger always leases the lowest pending schedule index, so earlier
    indexes are consumed by dedicated filler workers first.
    """
    for filler_index in range(attempt_index):
        filler_attempt = ledger.attempts()[filler_index]
        if ledger.status(filler_attempt.attempt_id) != "pending":
            continue
        filler = ledger.lease_next(f"filler-{filler_index}", lease_duration_ns=LEASE_NS)
        assert filler is not None and filler.attempt.schedule_index == filler_index
    worker_id = WORKERS[attempt_index]
    lease = ledger.lease_next(worker_id, lease_duration_ns=LEASE_NS)
    assert lease is not None
    assert lease.attempt.schedule_index == attempt_index
    attempt_id = lease.attempt.attempt_id

    output_dir = run_root / "attempts" / attempt_id
    videos = output_dir / "videos"
    videos.mkdir(parents=True)
    (videos / "top.mp4").write_bytes(b"\x00fakevideo")
    receipt = {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "lease_id": lease.lease_id,
        "worker_id": worker_id,
        "outcome": {"success": success, "metrics": [{"success": success}]},
    }
    (output_dir / "worker-receipt.json").write_text(json.dumps(receipt, sort_keys=True))
    if corrupt:
        (videos / "top.mp4").write_bytes(b"")  # empty file fails validation
    ledger.record_terminal(worker_id, attempt_id, lease.lease_id, str(output_dir))
    return worker_id, attempt_id, lease.lease_id, output_dir


def _queue(tmp_path, ledger, **overrides):
    defaults = dict(run_root=tmp_path / "run", ledger=ledger, max_pending_items=4, max_pending_bytes=1 << 30)
    defaults.update(overrides)
    return ArtifactFinalizationQueue(**defaults)


def test_success_episode_is_validated_then_accepted(tmp_path, ledger):
    run_root = tmp_path / "run"
    queue = _queue(tmp_path, ledger)
    worker_id, attempt_id, lease_id, output_dir = handoff_terminal(ledger, run_root, attempt_index=0)

    queue.enqueue(worker_id, attempt_id, lease_id, output_dir)
    result = queue.finalize_next()

    assert result is not None
    assert result.outcome == "accepted"
    assert ledger.status(attempt_id) == "accepted"
    accepted = run_root / "accepted" / attempt_id
    assert (accepted / "worker-receipt.json").is_file()
    assert (accepted / "SHA256SUMS.json").is_file()


def test_failed_episode_is_rejected(tmp_path, ledger):
    run_root = tmp_path / "run"
    queue = _queue(tmp_path, ledger)
    worker_id, attempt_id, lease_id, output_dir = handoff_terminal(ledger, run_root, attempt_index=1, success=False)

    queue.enqueue(worker_id, attempt_id, lease_id, output_dir)
    result = queue.finalize_next()

    assert result.outcome == "rejected"
    assert ledger.status(attempt_id) == "rejected"
    assert result.reason
    assert not (run_root / "accepted" / attempt_id).exists()


def test_empty_video_file_is_rejected(tmp_path, ledger):
    run_root = tmp_path / "run"
    queue = _queue(tmp_path, ledger)
    worker_id, attempt_id, lease_id, output_dir = handoff_terminal(ledger, run_root, attempt_index=2, corrupt=True)

    queue.enqueue(worker_id, attempt_id, lease_id, output_dir)
    result = queue.finalize_next()

    assert result.outcome == "rejected"
    assert ledger.status(attempt_id) == "rejected"


def test_backpressure_refuses_items_beyond_bounds(tmp_path, ledger):
    run_root = tmp_path / "run"
    queue = _queue(tmp_path, ledger, max_pending_items=1)
    first = handoff_terminal(ledger, run_root, attempt_index=0)
    second = handoff_terminal(ledger, run_root, attempt_index=1)

    queue.enqueue(*first[:3], first[3])
    with pytest.raises(QueueFullError):
        queue.enqueue(*second[:3], second[3])

    queue.finalize_next()
    queue.enqueue(*second[:3], second[3])
    assert queue.finalize_next().attempt_id == second[1]


def test_byte_bound_backpressure(tmp_path, ledger):
    run_root = tmp_path / "run"
    queue = _queue(tmp_path, ledger, max_pending_bytes=1)
    first = handoff_terminal(ledger, run_root, attempt_index=0)
    with pytest.raises(QueueFullError):
        queue.enqueue(*first[:3], first[3])


def test_drain_finalizes_all_within_deadline(tmp_path, ledger):
    run_root = tmp_path / "run"
    queue = _queue(tmp_path, ledger, max_pending_items=8)
    attempt_ids = []
    for index in (0, 1):
        handoff = handoff_terminal(ledger, run_root, attempt_index=index)
        attempt_ids.append(handoff[1])
        queue.enqueue(*handoff[:3], handoff[3])

    drained = queue.drain(deadline_seconds=5.0)

    assert drained == 2
    for attempt_id in attempt_ids:
        assert ledger.status(attempt_id) == "accepted"


def test_duplicate_enqueue_for_same_attempt_is_refused(tmp_path, ledger):
    run_root = tmp_path / "run"
    queue = _queue(tmp_path, ledger)
    handoff = handoff_terminal(ledger, run_root, attempt_index=0)
    queue.enqueue(*handoff[:3], handoff[3])
    with pytest.raises(ValueError, match="already pending"):
        queue.enqueue(*handoff[:3], handoff[3])


def test_enqueue_rejects_artifact_outside_run_root(tmp_path, ledger):
    run_root = tmp_path / "run"
    queue = _queue(tmp_path, ledger)
    handoff = handoff_terminal(ledger, run_root, attempt_index=0)
    with pytest.raises(ValueError, match="run root"):
        queue.enqueue(*handoff[:3], tmp_path)
