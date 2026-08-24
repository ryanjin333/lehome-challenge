from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import threading
import time
import types

import pytest


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
            "simulation_device": "cuda:0",
            "cloth_device": "cuda:0",
            "renderer_device": "cuda:0",
            "camera_device": "cuda:0",
            "cloth_backend": "physx_cloth_view",
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


class SourceController(FakeController):
    def __init__(self, leases: list[Lease], statuses: dict[str, list[str]]) -> None:
        super().__init__(leases)
        self._statuses = {attempt_id: list(values) for attempt_id, values in statuses.items()}
        self.status_calls: list[str] = []
        self.infrastructure_aborts: list[tuple[str, str, str, str]] = []

    def status(self, attempt_id: str) -> str:
        self.status_calls.append(attempt_id)
        values = self._statuses[attempt_id]
        return values.pop(0) if len(values) > 1 else values[0]

    def record_infrastructure_abort(self, worker_id: str, attempt_id: str, lease_id: str, *, reason: str) -> str:
        self.infrastructure_aborts.append((worker_id, attempt_id, lease_id, reason))
        return "infrastructure_abort"


def _source_assignment(seed: int) -> dict[str, object]:
    return {
        "snapshot_source_bootstrap": True,
        "category": "top_long",
        "garment": "Top_Long_Seen_0",
        "seed": seed,
        "source_seed": seed,
    }


def test_source_discovery_waits_for_clean_rejection_then_reuses_the_same_session(tmp_path) -> None:
    from lehome.flywheel.persistent_worker import PersistentRolloutWorker

    first = Lease(Attempt("attempt-a", _source_assignment(11)), "lease-a")
    second = Lease(Attempt("attempt-b", _source_assignment(12)), "lease-b")
    controller = SourceController([first, second], {
        "attempt-a": ["terminal_pending_validation", "rejected"],
        "attempt-b": ["accepted"],
    })
    session = FakeSession()
    worker = PersistentRolloutWorker(
        worker_id="worker-0", session_id="session-0", controller=controller,
        simulator_factory=lambda: session, policy=FakePolicy(), output_root=tmp_path,
        renderer_device="cuda:0", policy_device="cuda:0",
    )

    receipts = worker.run()

    assert [receipt["attempt_id"] for receipt in receipts] == ["attempt-a", "attempt-b"]
    assert session.runs == ["attempt-a", "attempt-b"]
    assert controller.status_calls == ["attempt-a", "attempt-a", "attempt-b"]


def test_policy_action_safety_rejection_durably_rejects_source_lease_and_continues(
    tmp_path,
) -> None:
    from lehome.flywheel.persistent_worker import (
        InfrastructureInvalidAttemptError,
        PersistentRolloutWorker,
        PolicyActionSafetyRejectionError,
    )

    first = Lease(Attempt("attempt-a", _source_assignment(11)), "lease-a")
    second = Lease(Attempt("attempt-b", _source_assignment(12)), "lease-b")

    class RejectingSourceController(SourceController):
        def __init__(self) -> None:
            super().__init__([first, second], {"attempt-b": ["accepted"]})
            self.rejected: list[tuple[str, str, str, str]] = []

        def reject_attempt(self, worker_id, attempt_id, lease_id, *, reason):
            self.rejected.append((worker_id, attempt_id, lease_id, reason))
            return "rejected"

    class PolicyRejectedSession(FakeSession):
        def run_episode(self, *, assignment, **kwargs):
            self.runs.append(str(assignment["attempt_id"]))
            if assignment["attempt_id"] == "attempt-a":
                raise PolicyActionSafetyRejectionError(
                    "policy_action_outside_live_joint_limits"
                )
            return {"success": True, "output_dir": str(kwargs["attempt_output_dir"])}

    controller = RejectingSourceController()
    session = PolicyRejectedSession()
    worker = PersistentRolloutWorker(
        worker_id="worker-0", session_id="session-0", controller=controller,
        simulator_factory=lambda: session, policy=FakePolicy(), output_root=tmp_path,
        renderer_device="cuda:0", policy_device="cuda:0",
    )

    receipts = worker.run()

    assert not issubclass(
        PolicyActionSafetyRejectionError, InfrastructureInvalidAttemptError
    )
    assert controller.rejected == [
        ("worker-0", "attempt-a", "lease-a", "policy_action_outside_live_joint_limits")
    ]
    assert controller.infrastructure_aborts == []
    assert session.runs == ["attempt-a", "attempt-b"]
    assert [receipt["attempt_id"] for receipt in receipts] == ["attempt-b"]
    assert controller.status_calls == ["attempt-b"]


def test_source_discovery_runtime_infrastructure_abort_stops_before_the_second_lease(tmp_path) -> None:
    from lehome.flywheel.persistent_worker import PersistentRolloutWorker

    first = Lease(Attempt("attempt-a", _source_assignment(11)), "lease-a")
    second = Lease(Attempt("attempt-b", _source_assignment(12)), "lease-b")
    controller = SourceController([first, second], {})
    class InvalidAfterEpisode(FakeSession):
        def run_episode(self, **kwargs):
            result = super().run_episode(**kwargs)
            self.runtime_receipt["cloth_device"] = "cpu"
            return result

    session = InvalidAfterEpisode()
    worker = PersistentRolloutWorker(
        worker_id="worker-0", session_id="session-0", controller=controller,
        simulator_factory=lambda: session, policy=FakePolicy(), output_root=tmp_path,
        renderer_device="cuda:0", policy_device="cuda:0",
    )

    with pytest.raises(RuntimeError, match="source discovery"):
        worker.run()

    assert controller.infrastructure_aborts == [("worker-0", "attempt-a", "lease-a", "runtime_evidence_invalid")]
    assert controller._leases == [second]


def test_source_discovery_snapshot_failure_is_an_infrastructure_abort_before_the_second_lease(tmp_path) -> None:
    from lehome.flywheel.persistent_worker import PersistentRolloutWorker

    first = Lease(Attempt("attempt-a", _source_assignment(11)), "lease-a")
    second = Lease(Attempt("attempt-b", _source_assignment(12)), "lease-b")
    controller = SourceController([first, second], {"attempt-b": ["accepted"]})

    class SnapshotFailure(FakeSession):
        def run_episode(self, *, assignment, **kwargs):
            self.runs.append(str(assignment["attempt_id"]))
            raise ValueError("snapshot evidence is missing")

    session = SnapshotFailure()
    worker = PersistentRolloutWorker(
        worker_id="worker-0", session_id="session-0", controller=controller,
        simulator_factory=lambda: session, policy=FakePolicy(), output_root=tmp_path,
        renderer_device="cuda:0", policy_device="cuda:0",
    )

    with pytest.raises(RuntimeError, match="source discovery"):
        worker.run()

    assert session.runs == ["attempt-a"]
    assert controller.infrastructure_aborts == [
        ("worker-0", "attempt-a", "lease-a", "source_snapshot_evidence_invalid"),
    ]
    assert controller._leases == [second]


def test_source_discovery_real_finalizer_abort_stops_before_the_second_lease(tmp_path) -> None:
    from lehome.flywheel.artifact_queue import ArtifactFinalizationQueue
    from lehome.flywheel.persistent_worker import PersistentRolloutWorker
    from lehome.flywheel.task_ledger import TaskLedger

    ledger = TaskLedger(
        tmp_path / "ledger.sqlite3",
        attempt_matrix=[_source_assignment(11), _source_assignment(12)],
        max_attempts=2,
        target_accepted=2,
    )
    queue = ArtifactFinalizationQueue(
        run_root=tmp_path / "output", ledger=ledger,
        max_pending_items=2, max_pending_bytes=1 << 30,
    )

    class FinalizingController:
        lease_calls = 0

        def lease_next(self, worker_id: str):
            self.lease_calls += 1
            return ledger.lease_next(worker_id, lease_duration_ns=10**18)

        def record_terminal(self, worker_id, attempt_id, lease_id, raw_artifact_id):
            ledger.record_terminal(worker_id, attempt_id, lease_id, raw_artifact_id)
            queue.enqueue(worker_id, attempt_id, lease_id, Path(raw_artifact_id))
            assert queue.finalize_next() is not None

        def status(self, attempt_id: str) -> str:
            return ledger.status(attempt_id)

        def heartbeat(self, worker_id, attempt_id, lease_id):
            return ledger.heartbeat(worker_id, attempt_id, lease_id, lease_duration_ns=10**18)

        def record_infrastructure_abort(self, worker_id, attempt_id, lease_id, *, reason: str):
            return ledger.record_infrastructure_abort(worker_id, attempt_id, lease_id, reason=reason)

    controller = FinalizingController()
    worker = PersistentRolloutWorker(
        worker_id="worker-0", session_id="session-0", controller=controller,
        simulator_factory=FakeSession, policy=FakePolicy(), output_root=tmp_path / "output",
        renderer_device="cuda:0", policy_device="cuda:0",
    )
    try:
        with pytest.raises(RuntimeError, match="infrastructure abort"):
            worker.run()
        first_attempt = ledger.attempts()[0].attempt_id
        assert ledger.status(first_attempt) == "infrastructure_abort"
        assert controller.lease_calls == 1
    finally:
        ledger.close()


@pytest.mark.parametrize("terminal", ["infrastructure_abort", "leased"])
def test_source_discovery_refuses_to_lease_after_nonfinal_or_infrastructure_finalization(tmp_path, terminal: str) -> None:
    from lehome.flywheel.persistent_worker import PersistentRolloutWorker

    first = Lease(Attempt("attempt-a", _source_assignment(11)), "lease-a")
    second = Lease(Attempt("attempt-b", _source_assignment(12)), "lease-b")
    controller = SourceController([first, second], {"attempt-a": [terminal]})
    worker = PersistentRolloutWorker(
        worker_id="worker-0", session_id="session-0", controller=controller,
        simulator_factory=FakeSession, policy=FakePolicy(), output_root=tmp_path,
        renderer_device="cuda:0", policy_device="cuda:0",
    )

    with pytest.raises(RuntimeError, match="source discovery"):
        worker.run()

    assert controller._leases == [second]


def test_source_discovery_finalization_timeout_stops_before_the_second_lease(tmp_path) -> None:
    from lehome.flywheel.persistent_worker import PersistentRolloutWorker

    first = Lease(Attempt("attempt-a", _source_assignment(11)), "lease-a")
    second = Lease(Attempt("attempt-b", _source_assignment(12)), "lease-b")
    controller = SourceController([first, second], {"attempt-a": ["terminal_pending_validation"]})
    worker = PersistentRolloutWorker(
        worker_id="worker-0", session_id="session-0", controller=controller,
        simulator_factory=FakeSession, policy=FakePolicy(), output_root=tmp_path,
        renderer_device="cuda:0", policy_device="cuda:0", source_finalization_timeout_seconds=0.001,
    )

    with pytest.raises(RuntimeError, match="timeout"):
        worker.run()

    assert controller._leases == [second]


def test_worker_reuses_one_cuda_cloth_simulator_and_immediately_leases_next_attempt(tmp_path) -> None:
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
    assert all(receipt["cloth_device"] == "cuda:0" for receipt in receipts)
    assert all(receipt["action_horizon"] == 16 for receipt in receipts)
    assert sessions[0].closed is True


def test_worker_reuses_authenticated_source_seed_for_controlled_recovery_reset(tmp_path) -> None:
    """A perturbation seed must never reseed the source-prefix replay."""

    from lehome.flywheel.persistent_worker import PersistentRolloutWorker

    session = FakeSession()
    controller = FakeController([
        Lease(Attempt("attempt-controlled", {
            "garment": "Pant_Long_Seen_4", "seed": 71_000,
                "source_seed": 50_110, "recovery_kind": "controlled_success_recovery_snapshot_v3",
        }), "lease-controlled"),
    ])
    worker = PersistentRolloutWorker(
        worker_id="worker-0", session_id="session-0", controller=controller,
        simulator_factory=lambda: session, policy=FakePolicy(), output_root=tmp_path,
        renderer_device="cuda:0", policy_device="cuda:0",
    )

    receipts = worker.run()

    assert session.prepared == [("Pant_Long_Seen_4", 50_110, 1)]
    assert receipts[0]["seed"] == 71_000


def test_worker_reports_simulator_factory_failure_before_slow_kit_shutdown(tmp_path, capsys) -> None:
    from lehome.flywheel.persistent_worker import PersistentRolloutWorker

    def fail_factory():
        raise TypeError("live constructor mismatch")

    worker = PersistentRolloutWorker(
        worker_id="worker-0", session_id="session-0", controller=FakeController([]),
        simulator_factory=fail_factory, policy=FakePolicy(), output_root=tmp_path,
        renderer_device="cuda:0", policy_device="cuda:0",
    )

    try:
        worker.run()
    except TypeError as error:
        assert str(error) == "live constructor mismatch"
    else:
        raise AssertionError("simulator factory failure was not propagated")

    assert "simulator factory failed: TypeError: live constructor mismatch" in capsys.readouterr().out


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


def test_worker_refuses_a_cloth_device_that_does_not_match_its_assignment(tmp_path) -> None:
    from lehome.flywheel.persistent_worker import PersistentRolloutWorker

    controller = FakeController([])
    session = FakeSession()
    session.runtime_receipt["cloth_device"] = "cpu"
    worker = PersistentRolloutWorker(
        worker_id="worker-0", session_id="session-0", controller=controller,
        simulator_factory=lambda: session, policy=FakePolicy(), output_root=tmp_path,
        renderer_device="cuda:0", policy_device="cuda:0",
    )

    import pytest
    with pytest.raises(ValueError, match="assigned simulator device"):
        worker.run()


def test_cpu_source_worker_accepts_usd_local_receipt_but_cuda_rejects_it(tmp_path) -> None:
    from lehome.flywheel.persistent_worker import PersistentRolloutWorker

    cpu_session = FakeSession()
    cpu_session.runtime_receipt.update({
        "simulation_device": "cpu",
        "cloth_device": "cpu",
        "cloth_backend": "usd_local_points_v1",
    })
    cpu_worker = PersistentRolloutWorker(
        worker_id="worker-cpu", session_id="session-cpu", controller=FakeController([]),
        simulator_factory=lambda: cpu_session, policy=FakePolicy(), output_root=tmp_path,
        renderer_device="cuda:0", policy_device="cuda:0", simulator_device="cpu",
    )

    assert cpu_worker._validate_runtime_receipt(cpu_session)["cloth_backend"] == "usd_local_points_v1"

    cuda_session = FakeSession()
    cuda_session.runtime_receipt["cloth_backend"] = "usd_local_points_v1"
    cuda_worker = PersistentRolloutWorker(
        worker_id="worker-cuda", session_id="session-cuda", controller=FakeController([]),
        simulator_factory=lambda: cuda_session, policy=FakePolicy(), output_root=tmp_path,
        renderer_device="cuda:0", policy_device="cuda:0",
    )
    with pytest.raises(ValueError, match="PhysX cloth backend"):
        cuda_worker._validate_runtime_receipt(cuda_session)


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


def test_launcher_binds_cloth_physics_to_the_renderer_cuda_device(monkeypatch) -> None:
    repository = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repository))
    from scripts.run_groot_persistent_worker import prepare_persistent_cloth_launch

    args = types.SimpleNamespace(
        device="cuda:9", simulator_device="cuda:2",
        renderer_device="cuda:2", policy_device="cuda:2",
    )
    environment: dict[str, str] = {}

    assert prepare_persistent_cloth_launch(args, environ=environment) == "2"
    assert args.device == "cuda:2"
    assert args.camera_device == "cuda:2"
    assert environment == {"LEHOME_FLYWHEEL_WORKER_GPU": "2"}


def test_parsed_default_simulator_inherits_a_nonzero_renderer_gpu(monkeypatch, tmp_path) -> None:
    repository = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repository))
    from scripts.run_groot_persistent_worker import build_parser, prepare_persistent_cloth_launch

    utils_package = types.ModuleType("scripts.utils")
    utils_package.__path__ = []
    parser_module = types.ModuleType("scripts.utils.parser")
    parser_module.setup_eval_parser = lambda: __import__("argparse").ArgumentParser()
    monkeypatch.setitem(sys.modules, "scripts.utils", utils_package)
    monkeypatch.setitem(sys.modules, "scripts.utils.parser", parser_module)

    args = build_parser().parse_args([
        "--database", str(tmp_path / "ledger.sqlite3"),
        "--attempt-matrix", str(tmp_path / "matrix.json"),
        "--worker-id", "worker-2",
        "--session-id", "session-2",
        "--output-root", str(tmp_path / "output"),
        "--renderer-device", "cuda:2",
        "--policy-device", "cuda:2",
        "--policy-gateway-endpoint", "tcp://127.0.0.1:15555",
        "--policy-sha256", "a" * 64,
        "--policy-ready-file", str(tmp_path / "ready.json"),
        "--initial-garment", "Top_Short_Seen_2",
    ])

    environment: dict[str, str] = {}
    assert prepare_persistent_cloth_launch(args, environ=environment) == "2"
    assert args.device == "cuda:2"
    assert args.camera_device == "cuda:2"
    assert environment == {"LEHOME_FLYWHEEL_WORKER_GPU": "2"}


def test_launcher_preserves_cpu_cloth_for_the_source_bootstrap_diagnostic(monkeypatch) -> None:
    repository = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repository))
    from scripts.run_groot_persistent_worker import prepare_persistent_cloth_launch

    args = types.SimpleNamespace(
        device="cuda:9", simulator_device="cpu",
        renderer_device="cuda:2", policy_device="cuda:2",
    )
    environment: dict[str, str] = {}

    assert prepare_persistent_cloth_launch(args, environ=environment) == "2"
    assert args.device == "cpu"
    assert args.camera_device == "cuda:2"
    assert environment == {"LEHOME_FLYWHEEL_WORKER_GPU": "2"}


def test_launcher_rejects_a_non_cuda_renderer_or_policy() -> None:
    from scripts.run_groot_persistent_worker import prepare_persistent_cloth_launch

    with pytest.raises(ValueError, match="renderer and policy devices"):
        prepare_persistent_cloth_launch(types.SimpleNamespace(
            device="cpu", simulator_device="cpu", renderer_device="cpu", policy_device="cuda:0",
        ), environ={})


def test_launcher_rejects_a_policy_on_a_different_physical_gpu() -> None:
    from scripts.run_groot_persistent_worker import prepare_persistent_cloth_launch

    with pytest.raises(ValueError, match="same physical CUDA device"):
        prepare_persistent_cloth_launch(types.SimpleNamespace(
            device="cuda:0", simulator_device="cuda:0", renderer_device="cuda:0", policy_device="cuda:1",
        ), environ={})


def test_runtime_rejects_cpu_cloth_for_a_non_source_matrix_before_opening_the_ledger(tmp_path) -> None:
    from scripts.run_groot_persistent_worker import run

    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps([{
        "attempt_id": "ordinary", "garment": "Top_Short_Seen_2", "seed": 50066,
    }]), encoding="utf-8")
    opened_ledger: list[bool] = []

    def ledger_factory(*_args, **_kwargs):
        opened_ledger.append(True)
        raise AssertionError("non-source CPU matrix must fail before the ledger opens")

    args = types.SimpleNamespace(
        device="cpu", renderer_device="cuda:0", policy_device="cuda:0",
        lease_seconds=30.0, preparation_timeout_seconds=30.0,
        attempt_matrix=matrix, database=tmp_path / "ledger.sqlite3",
        max_attempts=1, target_accepted=1,
    )

    with pytest.raises(ValueError, match="CPU cloth is reserved"):
        run(args, ledger_factory=ledger_factory)
    assert opened_ledger == []


def test_runtime_rejects_a_cpu_source_descriptor_with_mismatched_seed_before_ledger(tmp_path) -> None:
    from scripts.run_groot_persistent_worker import run

    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps([{
        "attempt_id": "source", "category": "top_short", "garment": "Top_Short_Seen_2",
        "seed": 50066, "source_seed": 50067, "snapshot_source_bootstrap": True,
    }]), encoding="utf-8")
    opened_ledger: list[bool] = []

    def ledger_factory(*_args, **_kwargs):
        opened_ledger.append(True)
        raise AssertionError("invalid CPU source descriptor must fail before the ledger opens")

    args = types.SimpleNamespace(
        device="cpu", renderer_device="cuda:0", policy_device="cuda:0",
        lease_seconds=30.0, preparation_timeout_seconds=30.0,
        attempt_matrix=matrix, database=tmp_path / "ledger.sqlite3",
        max_attempts=1, target_accepted=1,
    )

    with pytest.raises(ValueError, match="CPU cloth is reserved"):
        run(args, ledger_factory=ledger_factory)
    assert opened_ledger == []


def test_runtime_admits_bounded_multirow_cpu_source_discovery_before_ledger(tmp_path) -> None:
    from scripts.run_groot_persistent_worker import run

    rows = [_source_assignment(50_110 + index) for index in range(3)]
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps(rows), encoding="utf-8")
    opened: list[object] = []

    def ledger_factory(*args, **kwargs):
        opened.append((args, kwargs))
        raise RuntimeError("ledger reached")

    args = types.SimpleNamespace(
        device="cpu", renderer_device="cuda:0", policy_device="cuda:0",
        lease_seconds=30.0, preparation_timeout_seconds=30.0,
        attempt_matrix=matrix, database=tmp_path / "ledger.sqlite3",
        max_attempts=3, target_accepted=2,
    )

    with pytest.raises(RuntimeError, match="ledger reached"):
        run(args, ledger_factory=ledger_factory)

    assert len(opened) == 1


@pytest.mark.parametrize("mutate", [
    lambda rows: rows.__setitem__(1, {**rows[1], "category": "top_short"}),
    lambda rows: rows.__setitem__(1, {**rows[1], "source_seed": True}),
    lambda rows: rows.__setitem__(1, {**rows[1], "recovery_kind": "controlled_success_recovery_snapshot_v3"}),
])
def test_runtime_rejects_mixed_or_tampered_cpu_source_discovery_before_ledger(tmp_path, mutate) -> None:
    from scripts.run_groot_persistent_worker import run

    rows = [_source_assignment(50_110 + index) for index in range(3)]
    mutate(rows)
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps(rows), encoding="utf-8")
    opened: list[bool] = []

    def ledger_factory(*_args, **_kwargs):
        opened.append(True)
        raise AssertionError("invalid CPU source discovery must fail before the ledger opens")

    args = types.SimpleNamespace(
        device="cpu", renderer_device="cuda:0", policy_device="cuda:0",
        lease_seconds=30.0, preparation_timeout_seconds=30.0,
        attempt_matrix=matrix, database=tmp_path / "ledger.sqlite3",
        max_attempts=3, target_accepted=2,
    )

    with pytest.raises(ValueError):
        run(args, ledger_factory=ledger_factory)
    assert opened == []


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

    assert controller.heartbeats[0] == ("worker-0", "attempt-a", "lease-a")
    assert set(controller.heartbeats) == {("worker-0", "attempt-a", "lease-a")}
    assert not thread.is_alive()


def test_worker_leases_before_visible_contact_canary(tmp_path) -> None:
    from lehome.flywheel.persistent_worker import PersistentRolloutWorker

    class DeferredContactSession(FakeSession):
        def __init__(self) -> None:
            super().__init__()
            self.runtime_receipt = dict(self.runtime_receipt)
            self.runtime_receipt.pop("visible_contact_canary", None)

        def run_episode(self, *, assignment, attempt_output_dir, policy, cancellation_event):
            self.runtime_receipt["visible_contact_canary"] = {"observed": False}
            return super().run_episode(
                assignment=assignment, attempt_output_dir=attempt_output_dir,
                policy=policy, cancellation_event=cancellation_event,
            )

    controller = FakeController([
        Lease(Attempt("attempt-a", {"garment": "Top_Long_Seen_0", "seed": 11, "category": "top_long"}), "lease-a"),
    ])
    session = DeferredContactSession()
    worker = PersistentRolloutWorker(
        worker_id="worker-0", session_id="session-0", controller=controller,
        simulator_factory=lambda: session, policy=FakePolicy(), output_root=tmp_path,
        renderer_device="cuda:0", policy_device="cuda:0",
    )

    receipts = worker.run()

    assert [item["attempt_id"] for item in receipts] == ["attempt-a"]
    assert session.runs == ["attempt-a"]


def test_worker_leases_without_startup_cloth_readback(tmp_path) -> None:
    from lehome.flywheel.persistent_worker import PersistentRolloutWorker

    class NoReadbackSession(FakeSession):
        def __init__(self) -> None:
            super().__init__()
            self.runtime_receipt = dict(self.runtime_receipt)
            self.runtime_receipt.pop("cloth_readback", None)
            self.runtime_receipt.pop("visible_contact_canary", None)

        def run_episode(self, *, assignment, attempt_output_dir, policy, cancellation_event):
            self.runtime_receipt["cloth_readback"] = {"positions": 1, "velocities": 1}
            self.runtime_receipt["visible_contact_canary"] = {"observed": False}
            return super().run_episode(
                assignment=assignment, attempt_output_dir=attempt_output_dir,
                policy=policy, cancellation_event=cancellation_event,
            )

    controller = FakeController([
        Lease(Attempt("attempt-a", {"garment": "Top_Long_Seen_0", "seed": 11, "category": "top_long"}), "lease-a"),
    ])
    session = NoReadbackSession()
    worker = PersistentRolloutWorker(
        worker_id="worker-0", session_id="session-0", controller=controller,
        simulator_factory=lambda: session, policy=FakePolicy(), output_root=tmp_path,
        renderer_device="cuda:0", policy_device="cuda:0",
    )
    receipts = worker.run()
    assert [item["attempt_id"] for item in receipts] == ["attempt-a"]
    assert session.runs == ["attempt-a"]


def test_worker_infrastructure_aborts_an_episode_without_post_reset_cloth_evidence(tmp_path) -> None:
    from lehome.flywheel.persistent_worker import PersistentRolloutWorker

    class MissingPostResetEvidence(FakeSession):
        def __init__(self) -> None:
            super().__init__()
            self.runtime_receipt = dict(self.runtime_receipt)
            self.runtime_receipt.pop("cloth_readback", None)
            self.runtime_receipt.pop("visible_contact_canary", None)

    class Controller(FakeController):
        def __init__(self, leases):
            super().__init__(leases)
            self.aborted = []

        def record_infrastructure_abort(self, worker_id, attempt_id, lease_id, *, reason):
            self.aborted.append((worker_id, attempt_id, lease_id, reason))
            return "infrastructure_abort"

    controller = Controller([
        Lease(Attempt("attempt-a", {"garment": "Top_Long_Seen_0", "seed": 11}), "lease-a"),
    ])
    worker = PersistentRolloutWorker(
        worker_id="worker-0", session_id="session-0", controller=controller,
        simulator_factory=MissingPostResetEvidence, policy=FakePolicy(), output_root=tmp_path,
        renderer_device="cuda:0", policy_device="cuda:0",
    )

    assert worker.run() == []
    assert controller.completed == []
    assert controller.interrupted == []
    assert controller.aborted == [("worker-0", "attempt-a", "lease-a", "runtime_evidence_invalid")]
    assert list(tmp_path.rglob("worker-receipt.json")) == []


def test_worker_classifies_cloth_numerical_divergence_as_infrastructure_invalid(tmp_path) -> None:
    from lehome.flywheel.persistent_worker import (
        PersistentRolloutWorker,
        SimulatorNumericalDivergenceError,
    )

    class DivergentSession(FakeSession):
        def run_episode(self, *, assignment, attempt_output_dir, policy, cancellation_event):
            self.runs.append(str(assignment["attempt_id"]))
            raise SimulatorNumericalDivergenceError(
                "simulator_numerical_divergence: cloth physical-health admission failed"
            )

    class Controller(FakeController):
        def __init__(self, leases):
            super().__init__(leases)
            self.aborted = []

        def record_infrastructure_abort(self, worker_id, attempt_id, lease_id, *, reason):
            self.aborted.append((worker_id, attempt_id, lease_id, reason))
            return "infrastructure_abort"

    controller = Controller([
        Lease(Attempt("attempt-a", {"garment": "Top_Long_Seen_0", "seed": 11}), "lease-a"),
    ])
    worker = PersistentRolloutWorker(
        worker_id="worker-0", session_id="session-0", controller=controller,
        simulator_factory=DivergentSession, policy=FakePolicy(), output_root=tmp_path,
        renderer_device="cuda:0", policy_device="cuda:0",
    )

    assert worker.run() == []
    assert controller.completed == []
    assert controller.aborted == [
        ("worker-0", "attempt-a", "lease-a", "simulator_numerical_divergence")
    ]


def test_worker_continues_after_a_single_episode_runtime_error(tmp_path) -> None:
    from lehome.flywheel.persistent_worker import PersistentRolloutWorker

    class FailThenOk(FakeSession):
        def run_episode(self, *, assignment, attempt_output_dir, policy, cancellation_event):
            self.runs.append(str(assignment["attempt_id"]))
            if assignment["attempt_id"] == "attempt-a":
                raise RuntimeError("flywheel garment displayColor readback mismatch")
            return {"success": True}

    controller = FakeController([
        Lease(Attempt("attempt-a", {"garment": "Pant_Long_Seen_0", "seed": 151, "category": "pant_long"}), "lease-a"),
        Lease(Attempt("attempt-b", {"garment": "Top_Long_Seen_0", "seed": 11, "category": "top_long"}), "lease-b"),
    ])
    session = FailThenOk()
    worker = PersistentRolloutWorker(
        worker_id="worker-0", session_id="session-0", controller=controller,
        simulator_factory=lambda: session, policy=FakePolicy(), output_root=tmp_path,
        renderer_device="cuda:0", policy_device="cuda:0",
    )
    with pytest.raises(RuntimeError, match="displayColor"):
        worker.run()
    assert session.runs == ["attempt-a"]
    assert controller.interrupted == []


def test_worker_rejects_a_restore_mismatch_instead_of_retrying_the_same_lease(tmp_path) -> None:
    from lehome.flywheel.persistent_worker import PersistentRolloutWorker

    class RestoreBoom(FakeSession):
        def run_episode(self, *, assignment, attempt_output_dir, policy, cancellation_event):
            self.runs.append(str(assignment["attempt_id"]))
            if assignment["attempt_id"] == "attempt-a":
                raise ValueError("snapshot garment does not match the active environment")
            return {"success": True}

    class Controller(FakeController):
        def __init__(self, leases):
            super().__init__(leases)
            self.rejected = []
        def reject_attempt(self, worker_id, attempt_id, lease_id, *, reason):
            self.rejected.append((worker_id, attempt_id, lease_id, reason))
            return "rejected"

    bad = Lease(Attempt("attempt-a", {"garment": "Pant_Long_Seen_5", "seed": 149, "restore_snapshot": "/tmp/term.json"}), "lease-a")
    nxt = Lease(Attempt("attempt-b", {"garment": "Pant_Long_Seen_7", "seed": 151}), "lease-b")
    controller = Controller([bad, nxt])
    session = RestoreBoom()
    worker = PersistentRolloutWorker(
        worker_id="worker-0", session_id="session-0", controller=controller,
        simulator_factory=lambda: session, policy=FakePolicy(), output_root=tmp_path,
        renderer_device="cuda:0", policy_device="cuda:0",
    )
    receipts = worker.run()
    assert session.runs == ["attempt-a", "attempt-b"]
    assert controller.rejected[0][:3] == ("worker-0", "attempt-a", "lease-a")
    assert [item["attempt_id"] for item in receipts] == ["attempt-b"]


def test_worker_heartbeats_while_preparing_an_episode(tmp_path) -> None:
    """A slow garment switch must retain its lease before inference begins."""

    from lehome.flywheel.persistent_worker import PersistentRolloutWorker

    started = threading.Event()
    release = threading.Event()

    class BlockingPreparationSession(FakeSession):
        def prepare_episode(self, *, garment_name, seed, episode_generation, reset_policy=True):
            started.set()
            assert release.wait(timeout=1.0)
            return super().prepare_episode(
                garment_name=garment_name, seed=seed, episode_generation=episode_generation,
                reset_policy=reset_policy,
            )

    controller = FakeController([Lease(Attempt("attempt-a", {"garment": "shirt", "seed": 1}), "lease-a")])
    worker = PersistentRolloutWorker(
        worker_id="worker-0", session_id="session-0", controller=controller,
        simulator_factory=BlockingPreparationSession, policy=FakePolicy(), output_root=tmp_path,
        renderer_device="cuda:0", policy_device="cuda:0", heartbeat_interval_seconds=0.01,
        preparation_timeout_seconds=1.0,
    )
    thread = threading.Thread(target=worker.run)
    thread.start()
    assert started.wait(timeout=1.0)
    deadline = time.monotonic() + 1.0
    while not controller.heartbeats and time.monotonic() < deadline:
        time.sleep(0.01)
    release.set()
    thread.join(timeout=1.0)

    assert controller.heartbeats
    assert set(controller.heartbeats) == {("worker-0", "attempt-a", "lease-a")}
    assert not thread.is_alive()


def test_preparation_timeout_rejects_then_requests_an_isolated_hard_exit(tmp_path, capsys) -> None:
    """A native preparation hang is terminal for its lease, never retried forever."""

    from lehome.flywheel.persistent_worker import PreparationTimeoutError, PersistentRolloutWorker

    started = threading.Event()
    release = threading.Event()
    hard_exit_called = threading.Event()
    exit_codes: list[int] = []

    class Controller(FakeController):
        def __init__(self, leases):
            super().__init__(leases)
            self.rejected = []

        def record_infrastructure_abort(self, worker_id, attempt_id, lease_id, *, reason):
            self.rejected.append((worker_id, attempt_id, lease_id, reason))
            return "infrastructure_abort"

    class BlockingPreparationSession(FakeSession):
        def prepare_episode(self, *, garment_name, seed, episode_generation, reset_policy=True):
            started.set()
            assert release.wait(timeout=1.0)
            return super().prepare_episode(
                garment_name=garment_name, seed=seed, episode_generation=episode_generation,
                reset_policy=reset_policy,
            )

    controller = Controller([Lease(Attempt("attempt-a", {"garment": "shirt", "seed": 1}), "lease-a")])
    worker = PersistentRolloutWorker(
        worker_id="worker-0", session_id="session-0", controller=controller,
        simulator_factory=BlockingPreparationSession, policy=FakePolicy(), output_root=tmp_path,
        renderer_device="cuda:0", policy_device="cuda:0", heartbeat_interval_seconds=0.01,
        preparation_timeout_seconds=0.03,
        hard_exit=lambda status: (exit_codes.append(status), hard_exit_called.set()),
    )
    errors: list[BaseException] = []

    def run_worker() -> None:
        try:
            worker.run()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=run_worker)
    thread.start()
    assert started.wait(timeout=1.0)
    assert hard_exit_called.wait(timeout=1.0)
    assert controller.rejected == [("worker-0", "attempt-a", "lease-a", "preparation_timeout")]
    assert exit_codes == [70]
    assert "preparation timeout; aborting infrastructure-invalid attempt" in capsys.readouterr().out

    # The injected test exit returns instead of terminating the process. Let
    # preparation return so the worker can surface the same timeout locally.
    release.set()
    thread.join(timeout=1.0)
    assert len(errors) == 1
    assert isinstance(errors[0], PreparationTimeoutError)
    assert not thread.is_alive()


def test_normal_preparation_cancels_its_watchdog(tmp_path) -> None:
    from lehome.flywheel.persistent_worker import PersistentRolloutWorker

    exit_codes: list[int] = []
    controller = FakeController([Lease(Attempt("attempt-a", {"garment": "shirt", "seed": 1}), "lease-a")])
    worker = PersistentRolloutWorker(
        worker_id="worker-0", session_id="session-0", controller=controller,
        simulator_factory=FakeSession, policy=FakePolicy(), output_root=tmp_path,
        renderer_device="cuda:0", policy_device="cuda:0", preparation_timeout_seconds=0.03,
        hard_exit=exit_codes.append,
    )

    assert [receipt["attempt_id"] for receipt in worker.run()] == ["attempt-a"]
    time.sleep(0.06)
    assert exit_codes == []


def test_worker_requires_a_strictly_positive_preparation_timeout(tmp_path) -> None:
    from lehome.flywheel.persistent_worker import PersistentRolloutWorker
    import pytest

    with pytest.raises(ValueError, match="preparation_timeout_seconds must be positive"):
        PersistentRolloutWorker(
            worker_id="worker-0", session_id="session-0", controller=FakeController([]),
            simulator_factory=FakeSession, policy=FakePolicy(), output_root=tmp_path,
            renderer_device="cuda:0", policy_device="cuda:0", preparation_timeout_seconds=0,
        )


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0])
def test_launcher_rejects_an_invalid_preparation_timeout_before_opening_the_ledger(monkeypatch, timeout) -> None:
    repository = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repository))
    from scripts.run_groot_persistent_worker import run

    opened_ledger = []

    def ledger_factory(*_args, **_kwargs):
        opened_ledger.append(True)
        raise AssertionError("invalid preparation timeout must not open the ledger")

    args = types.SimpleNamespace(device="cpu", lease_seconds=30.0, preparation_timeout_seconds=timeout)
    with pytest.raises(ValueError, match="preparation timeout seconds must be positive"):
        run(args, ledger_factory=ledger_factory)
    assert opened_ledger == []


def test_runtime_worker_loads_generated_controlled_materialization_shape(tmp_path) -> None:
    from scripts.run_groot_persistent_worker import _load_matrix

    reset, annotations, continuation = tmp_path / "reset.json", tmp_path / "annotations.jsonl", tmp_path / "continuation.json"
    reset.write_text("{}", encoding="utf-8"); annotations.write_text("", encoding="utf-8"); continuation.write_text("{}", encoding="utf-8")
    caps = {"pant_long": 4, "top_long": 1, "top_short": 3, "pant_short": 0}
    categories = ["pant_long"] * 4 + ["top_long"] + ["top_short"] * 3
    rows = [
        {
            "attempt_id": f"controlled-{index}", "trial_id": f"controlled-{index}",
            "category": category, "category_acceptance_cap": caps[category],
            "strategy": "canonical", "recovery_kind": "controlled_success_recovery_snapshot_v3",
            "controlled_matrix_sha256": "a" * 64, "perturbation_seed": 71_000 + index,
            "perturbation_fingerprint": f"{index + 100:064x}",
            "source_state_perturbation_fingerprint": f"{index + 200:064x}",
            "source_state_fingerprint": f"{index + 300:064x}", "source_round_id": "round",
                "source_episode_id": f"episode-{index}", "source_episode_digest": f"{index + 400:064x}",
                "source_seed": 50110, "source_continuation_state": [float(index)] * 12,
                "source_snapshot_schema_version": 2, "source_snapshot_authority": "physx_cloth_view_world_v1", "source_only_envelope": False,
            "source_immutable_revision": "a" * 40,
            "source_reset_sha256": "a" * 64, "source_annotations_sha256": "b" * 64,
            "source_continuation_snapshot_sha256": "c" * 64, "prefix_stop": 16,
            "source_first_success_step": 19, "source_reset": str(reset),
            "source_annotations": str(annotations), "source_continuation_snapshot": str(continuation),
        }
        for index, category in enumerate(categories)
    ]
    path = tmp_path / "materialization.json"
    path.write_text(json.dumps({"schema_version": 3, "kind": "controlled_success_recovery_materialization_v3", "matrix_sha256": "a" * 64, "target_accepted": 8, "category_acceptance_caps": caps, "rows": rows}), encoding="utf-8")
    assert _load_matrix(path) == rows


def test_failed_preparation_stops_its_heartbeat_before_the_next_lease(tmp_path) -> None:
    from lehome.flywheel.persistent_worker import PersistentRolloutWorker

    class FailsFirstPreparation(FakeSession):
        def prepare_episode(self, *, garment_name, seed, episode_generation, reset_policy=True):
            if garment_name == "bad-shirt":
                time.sleep(0.03)
                raise RuntimeError("garment preparation failed")
            return super().prepare_episode(
                garment_name=garment_name, seed=seed, episode_generation=episode_generation,
                reset_policy=reset_policy,
            )

    controller = FakeController([
        Lease(Attempt("attempt-a", {"garment": "bad-shirt", "seed": 1}), "lease-a"),
        Lease(Attempt("attempt-b", {"garment": "good-shirt", "seed": 2}), "lease-b"),
    ])
    worker = PersistentRolloutWorker(
        worker_id="worker-0", session_id="session-0", controller=controller,
        simulator_factory=FailsFirstPreparation, policy=FakePolicy(), output_root=tmp_path,
        renderer_device="cuda:0", policy_device="cuda:0", heartbeat_interval_seconds=0.01,
    )

    with pytest.raises(RuntimeError, match="garment preparation failed"):
        worker.run()
    old_heartbeat_count = controller.heartbeats.count(("worker-0", "attempt-a", "lease-a"))
    time.sleep(0.04)
    assert old_heartbeat_count >= 1
    assert controller.heartbeats.count(("worker-0", "attempt-a", "lease-a")) == old_heartbeat_count
