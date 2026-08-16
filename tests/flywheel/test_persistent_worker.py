from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import threading
import time
import types


@dataclass
class Attempt:
    attempt_id: str
    assignment: dict[str, object]


@dataclass
class Lease:
    attempt: Attempt
    lease_id: str


class FakeController:
    def __init__(self, leases: list[Lease]) -> None:
        self._leases = list(leases)
        self.completed: list[tuple[str, str, str, str]] = []
        self.interrupted: list[tuple[str, str, str, str]] = []
        self.heartbeats: list[tuple[str, str, str]] = []

    def lease_next(self, worker_id: str):
        return self._leases.pop(0) if self._leases else None

    def record_terminal(self, worker_id: str, attempt_id: str, lease_id: str, raw_artifact_id: str):
        self.completed.append((worker_id, attempt_id, lease_id, raw_artifact_id))

    def record_interrupted(self, worker_id: str, attempt_id: str, lease_id: str, reason: str) -> None:
        self.interrupted.append((worker_id, attempt_id, lease_id, reason))

    def heartbeat(self, worker_id: str, attempt_id: str, lease_id: str) -> None:
        self.heartbeats.append((worker_id, attempt_id, lease_id))


class FakePolicy:
    action_horizon = 16

    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1


class FakeSession:
    def __init__(self) -> None:
        self.prepared: list[tuple[str, int, int]] = []
        self.runs: list[str] = []
        self.closed = False
        self.runtime_receipt = {
            "simulation_device": "cpu",
            "cloth_device": "cpu",
            "renderer_device": "cuda:0",
            "camera_device": "cuda:0",
            "cloth_backend": "usd",
            "cloth_readback": {"positions": 1, "velocities": 1},
            "visible_contact_canary": {"observed": False},
            "policy_device": "cuda:0",
        }

    def prepare_episode(self, *, garment_name: str, seed: int, episode_generation: int, reset_policy: bool = True) -> None:
        assert reset_policy is False
        self.prepared.append((garment_name, seed, episode_generation))

    def run_episode(self, *, assignment, attempt_output_dir, policy, cancellation_event):
        self.runs.append(str(assignment["attempt_id"]))
        return {"success": True, "output_dir": str(attempt_output_dir)}

    def close(self) -> None:
        self.closed = True


def test_worker_reuses_one_cpu_cloth_simulator_and_immediately_leases_next_attempt(tmp_path) -> None:
    from lehome.flywheel.persistent_worker import PersistentRolloutWorker

    controller = FakeController([
        Lease(Attempt("attempt-a", {"garment": "Top_Long_Seen_0", "seed": 11}), "lease-a"),
        Lease(Attempt("attempt-b", {"garment": "Top_Long_Seen_1", "seed": 12}), "lease-b"),
    ])
    sessions: list[FakeSession] = []
    policy = FakePolicy()

    worker = PersistentRolloutWorker(
        worker_id="worker-0", session_id="session-0", controller=controller,
        simulator_factory=lambda: sessions.append(FakeSession()) or sessions[-1], policy=policy,
        output_root=tmp_path, renderer_device="cuda:0", policy_device="cuda:0",
    )

    receipts = worker.run()

    assert len(sessions) == 1
    assert sessions[0].prepared == [
        ("Top_Long_Seen_0", 11, 1), ("Top_Long_Seen_1", 12, 2),
    ]
    assert sessions[0].runs == ["attempt-a", "attempt-b"]
    assert policy.resets == 2
    assert [item[1] for item in controller.completed] == ["attempt-a", "attempt-b"]
    assert [receipt["output_dir"] for receipt in receipts] == [
        str(tmp_path / "worker-0" / "session-0" / "attempt-a" / "lease-a" / "generation-1"),
        str(tmp_path / "worker-0" / "session-0" / "attempt-b" / "lease-b" / "generation-2"),
    ]
    assert all(receipt["cloth_device"] == "cpu" for receipt in receipts)
    assert all(receipt["action_horizon"] == 16 for receipt in receipts)
    assert sessions[0].closed is True


def test_worker_records_an_interruption_then_stops_leasing_for_shutdown(tmp_path) -> None:
    from lehome.flywheel.persistent_worker import PersistentRolloutWorker

    lease = Lease(Attempt("attempt-a", {"garment": "Top_Long_Seen_0", "seed": 11}), "lease-a")
    controller = FakeController([lease, lease])
    session = FakeSession()

    class RetrySession(FakeSession):
        def run_episode(self, *, assignment, attempt_output_dir, policy, cancellation_event):
            self.runs.append(str(assignment["attempt_id"]))
            if len(self.runs) == 1:
                raise InterruptedError("spot instance warning")
            return {"success": True}

    retry_session = RetrySession()
    worker = PersistentRolloutWorker(
        worker_id="worker-0", session_id="session-0", controller=controller,
        simulator_factory=lambda: retry_session, policy=FakePolicy(), output_root=tmp_path,
        renderer_device="cuda:0", policy_device="cuda:0",
    )

    receipts = worker.run()

    assert retry_session.runs == ["attempt-a"]
    assert len(controller.interrupted) == 1
    assert controller.interrupted[0][:3] == ("worker-0", "attempt-a", "lease-a")
    assert receipts == []
    assert len(controller._leases) == 1


def test_worker_refuses_a_restarted_worker_output_collision(tmp_path) -> None:
    from lehome.flywheel.persistent_worker import PersistentRolloutWorker
    import pytest

    lease = Lease(Attempt("attempt-a", {"garment": "Top_Long_Seen_0", "seed": 11}), "lease-a")
    path = tmp_path / "worker-0" / "session-0" / "attempt-a" / "lease-a" / "generation-1"
    path.mkdir(parents=True)
    worker = PersistentRolloutWorker(
        worker_id="worker-0", session_id="session-0", controller=FakeController([lease]),
        simulator_factory=FakeSession, policy=FakePolicy(), output_root=tmp_path,
        renderer_device="cuda:0", policy_device="cuda:0",
    )

    with pytest.raises(ValueError, match="output directory"):
        worker.run()


def test_worker_refuses_gpu_cloth_before_requesting_a_lease(tmp_path) -> None:
    from lehome.flywheel.persistent_worker import PersistentRolloutWorker

    controller = FakeController([])
    session = FakeSession()
    session.runtime_receipt["cloth_device"] = "cuda:0"
    worker = PersistentRolloutWorker(
        worker_id="worker-0", session_id="session-0", controller=controller,
        simulator_factory=lambda: session, policy=FakePolicy(), output_root=tmp_path,
        renderer_device="cuda:0", policy_device="cuda:0",
    )

    import pytest
    with pytest.raises(ValueError, match="CPU cloth"):
        worker.run()


def test_worker_rejects_missing_or_mismatched_actual_policy_device(tmp_path) -> None:
    from lehome.flywheel.persistent_worker import PersistentRolloutWorker
    import pytest

    session = FakeSession()
    session.runtime_receipt["policy_device"] = "cuda:1"
    worker = PersistentRolloutWorker(
        worker_id="worker-0", session_id="session-0", controller=FakeController([]),
        simulator_factory=lambda: session, policy=FakePolicy(), output_root=tmp_path,
        renderer_device="cuda:0", policy_device="cuda:0",
    )

    with pytest.raises(ValueError, match="policy device"):
        worker.run()


def test_interruption_is_an_append_only_ledger_retry_and_releases_the_next_lease(tmp_path) -> None:
    from lehome.flywheel.persistent_worker import PersistentRolloutWorker
    from lehome.flywheel.task_ledger import TaskLedger

    ledger = TaskLedger(
        tmp_path / "ledger.sqlite3",
        attempt_matrix=[{"garment": "Top_Long_Seen_0", "seed": 11}],
        max_attempts=2, target_accepted=1, clock_ns=lambda: 1,
    )

    class LedgerController:
        def lease_next(self, worker_id):
            return ledger.lease_next(worker_id, lease_duration_ns=100)

        def record_terminal(self, worker_id, attempt_id, lease_id, raw_artifact_id):
            return ledger.record_terminal(worker_id, attempt_id, lease_id, raw_artifact_id)

        def record_interrupted(self, worker_id, attempt_id, lease_id, reason):
            return ledger.record_interrupted(worker_id, attempt_id, lease_id, reason)

    class InterruptOnce(FakeSession):
        def run_episode(self, *, assignment, attempt_output_dir, policy, cancellation_event):
            self.runs.append(str(assignment["attempt_id"]))
            if len(self.runs) == 1:
                raise InterruptedError("interrupted")
            return {"success": True}

    try:
        worker = PersistentRolloutWorker(
            worker_id="worker-0", session_id="session-0", controller=LedgerController(),
            simulator_factory=InterruptOnce, policy=FakePolicy(), output_root=tmp_path / "output",
            renderer_device="cuda:0", policy_device="cuda:0",
        )
        assert worker.run() == []
        replacement = PersistentRolloutWorker(
            worker_id="worker-0", session_id="replacement-session", controller=LedgerController(),
            simulator_factory=FakeSession, policy=FakePolicy(), output_root=tmp_path / "output",
            renderer_device="cuda:0", policy_device="cuda:0",
        )
        receipts = replacement.run()
        events = ledger.events(receipts[0]["attempt_id"])
        assert [event.event_type for event in events] == [
            "leased", "interrupted", "retryable", "leased", "terminal_pending_validation",
        ]
        assert events[0].lease_id != events[3].lease_id
    finally:
        ledger.close()


def test_h16_local_cache_policy_is_reset_once_before_an_episode(tmp_path) -> None:
    from lehome.flywheel.persistent_worker import PersistentRolloutWorker

    class CachedPolicy(FakePolicy):
        def __init__(self) -> None:
            super().__init__()
            self.pending = []
            self.inference_requests = 0

        def reset(self) -> None:
            super().reset()
            self.pending.clear()

        def select_action(self, _observation):
            if not self.pending:
                self.inference_requests += 1
                self.pending.extend(range(16))
            return self.pending.pop(0)

    class ActionSession(FakeSession):
        def run_episode(self, *, assignment, attempt_output_dir, policy, cancellation_event):
            for _ in range(17):
                policy.select_action({})
            return {"success": True}

    policy = CachedPolicy()
    worker = PersistentRolloutWorker(
        worker_id="worker-0", session_id="session-0",
        controller=FakeController([Lease(Attempt("attempt-a", {"garment": "shirt", "seed": 1}), "lease-a")]),
        simulator_factory=ActionSession, policy=policy, output_root=tmp_path,
        renderer_device="cuda:0", policy_device="cuda:0",
    )

    worker.run()

    assert policy.resets == 1
    assert policy.inference_requests == 2


def test_worker_receipt_uses_the_policy_clients_actual_episode_generation(tmp_path) -> None:
    from lehome.flywheel.persistent_worker import PersistentRolloutWorker

    class GeneratedPolicy(FakePolicy):
        def __init__(self) -> None:
            super().__init__()
            self.episode_generation = 40

        def reset(self) -> None:
            super().reset()
            self.episode_generation += 1

    policy = GeneratedPolicy()
    worker = PersistentRolloutWorker(
        worker_id="worker-0", session_id="session-0",
        controller=FakeController([Lease(Attempt("attempt-a", {"garment": "shirt", "seed": 1}), "lease-a")]),
        simulator_factory=FakeSession, policy=policy, output_root=tmp_path,
        renderer_device="cuda:0", policy_device="cuda:0",
    )

    receipts = worker.run()

    assert receipts[0]["episode_generation"] == policy.episode_generation == 41


def test_launcher_validates_the_renderer_before_forcing_cpu_cloth(monkeypatch) -> None:
    repository = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repository))
    from scripts.run_groot_persistent_worker import prepare_cpu_cloth_launch

    args = types.SimpleNamespace(device="cuda:9", renderer_device="cuda:2", policy_device="cuda:3")
    environment: dict[str, str] = {}

    assert prepare_cpu_cloth_launch(args, environ=environment) == "2"
    assert args.device == "cpu"
    assert args.camera_device == "cuda:2"
    assert environment == {"LEHOME_FLYWHEEL_WORKER_GPU": "2"}


def test_worker_heartbeats_while_an_episode_blocks_and_stops_the_timer(tmp_path) -> None:
    from lehome.flywheel.persistent_worker import PersistentRolloutWorker

    started = threading.Event()
    release = threading.Event()

    class BlockingSession(FakeSession):
        def run_episode(self, *, assignment, attempt_output_dir, policy, cancellation_event):
            started.set()
            assert release.wait(timeout=1.0)
            return {"success": True}

    controller = FakeController([Lease(Attempt("attempt-a", {"garment": "shirt", "seed": 1}), "lease-a")])
    worker = PersistentRolloutWorker(
        worker_id="worker-0", session_id="session-0", controller=controller,
        simulator_factory=BlockingSession, policy=FakePolicy(), output_root=tmp_path,
        renderer_device="cuda:0", policy_device="cuda:0", heartbeat_interval_seconds=0.01,
    )
    thread = threading.Thread(target=worker.run)
    thread.start()
    assert started.wait(timeout=1.0)
    deadline = time.monotonic() + 1.0
    while not controller.heartbeats and time.monotonic() < deadline:
        time.sleep(0.01)
    release.set()
    thread.join(timeout=1.0)

    assert controller.heartbeats == [("worker-0", "attempt-a", "lease-a")]
    assert not thread.is_alive()
