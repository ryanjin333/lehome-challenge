"""CPU-only capacity lifecycle contracts for the fixed three-VM sweep pool."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest


def _snapshot(
    *,
    ready_training: int = 0,
    leaseable_training: int | None = None,
    eval_ready: int = 0,
    leases: tuple[dict[str, object], ...] = (),
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "ready_training_count": ready_training,
        "leaseable_training_count": ready_training if leaseable_training is None else leaseable_training,
        "eval_ready_count": eval_ready,
        "active_leases": list(leases),
        "idle_stop_recommended": (ready_training if leaseable_training is None else leaseable_training) == 0 and not any(item["capability"] == "training" for item in leases),
    }


class _FakeController:
    def __init__(self, snapshots: list[dict[str, object]] | Exception) -> None:
        self.snapshots = snapshots
        self.calls = 0

    def capacity_snapshot(self) -> dict[str, object]:
        self.calls += 1
        if isinstance(self.snapshots, Exception):
            raise self.snapshots
        return self.snapshots[min(self.calls - 1, len(self.snapshots) - 1)]


class _FakeNebius:
    def __init__(self, states: dict[str, str]) -> None:
        self.states = dict(states)
        self.calls: list[tuple[str, str]] = []

    def instance_state(self, instance_id: str) -> str:
        self.calls.append(("state", instance_id))
        return self.states[instance_id]

    def start_instance(self, instance_id: str) -> None:
        self.calls.append(("start", instance_id))
        assert self.states[instance_id] == "STOPPED"
        self.states[instance_id] = "RUNNING"

    def stop_instance(self, instance_id: str) -> None:
        self.calls.append(("stop", instance_id))
        assert self.states[instance_id] == "RUNNING"
        self.states[instance_id] = "STOPPED"


def _gate(tmp_path: Path) -> tuple[Path, str]:
    gate = tmp_path / "deployment-gate.json"
    gate.write_bytes(b"verified-deployment-gate\n")
    gate.chmod(0o444)
    return gate, hashlib.sha256(gate.read_bytes()).hexdigest()


def _config(tmp_path: Path):
    from lehome_train.groot.experiment_capacity import (
        CapacityConfig,
        ManagedWorker,
        PRODUCTION_EVALUATOR_WORKER_ID,
    )

    gate, digest = _gate(tmp_path)
    return CapacityConfig(
        training_workers=(ManagedWorker("train-vm-a", "train-a"), ManagedWorker("train-vm-b", "train-b")),
        rollout_worker=ManagedWorker("rollout-vm", PRODUCTION_EVALUATOR_WORKER_ID),
        idle_seconds=600,
        operation_cap=3,
        deployment_gate_path=gate,
        deployment_gate_sha256=digest,
    )


def test_controller_snapshot_is_transactional_and_has_only_capacity_fields(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_controller import ExperimentController
    from test_experiment_controller import _job

    controller = ExperimentController(tmp_path / "controller.sqlite3")
    job = _job(tmp_path, "a")
    controller.add_jobs([job])
    lease = controller.lease_next("train-a", "training", 1, 100)
    assert lease is not None

    snapshot = controller.capacity_snapshot(now_ns=2)

    assert snapshot == {
        "schema_version": 1,
        "ready_training_count": 0,
        "leaseable_training_count": 0,
        "eval_ready_count": 0,
        "active_leases": [{"lease_id": lease.lease_id, "experiment_id": job.experiment_id, "worker_id": "train-a", "capability": "training", "expires_ns": 101}],
        "idle_stop_recommended": False,
    }


def test_capacity_does_not_start_a_trainer_when_the_production_sixty_second_lease_would_exceed_budget(
    tmp_path: Path,
) -> None:
    """Capacity demand must use the worker's real initial paid lease floor."""
    from lehome_train.groot.experiment_capacity import CapacityLifecycle
    from lehome_train.groot.experiment_controller import ExperimentController
    from test_experiment_controller import _job

    second = 1_000_000_000
    controller = ExperimentController(
        tmp_path / "controller.sqlite3",
        gpu_seconds_ceiling=59.0,
        spend_ceiling=59.0,
        estimated_gpu_seconds_per_step=0.001,
        gpu_price_per_second=1.0,
    )
    job = _job(tmp_path, "sixty-second-budget-floor")
    controller.add_jobs([job])

    snapshot = controller.capacity_snapshot(now_ns=0)
    assert snapshot["ready_training_count"] == 1
    assert snapshot["leaseable_training_count"] == 0

    nebius = _FakeNebius({"train-vm-a": "STOPPED", "train-vm-b": "STOPPED", "rollout-vm": "STOPPED"})
    receipt = CapacityLifecycle(_config(tmp_path), _FakeController([snapshot]), nebius).reconcile(now_ns=0)
    assert receipt.actions == ()

    # The production worker requests exactly this initial TTL.  Capacity must
    # agree with real admission rather than starting a VM for a job that the
    # controller immediately rejects.
    assert controller.lease_next("trainer", "training", 0, 60 * second) is None
    assert controller.state(job.experiment_id) == "BLOCKED_BUDGET"


def test_capacity_starts_only_exact_stopped_training_vm_for_ready_training(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_capacity import CapacityLifecycle

    nebius = _FakeNebius({"train-vm-a": "STOPPED", "train-vm-b": "STOPPED", "rollout-vm": "STOPPED"})
    receipt = CapacityLifecycle(_config(tmp_path), _FakeController([_snapshot(ready_training=1)]), nebius).reconcile(now_ns=1)

    assert receipt.actions == ("start:training",)
    assert nebius.calls == [("state", "train-vm-a"), ("state", "train-vm-b"), ("state", "rollout-vm"), ("start", "train-vm-a")]
    assert all(action not in {"create", "delete"} for action, _ in nebius.calls)


def test_capacity_starts_rollout_only_for_pending_evaluation_and_keeps_lanes_independent(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_capacity import CapacityLifecycle

    nebius = _FakeNebius({"train-vm-a": "STOPPED", "train-vm-b": "STOPPED", "rollout-vm": "STOPPED"})
    receipt = CapacityLifecycle(_config(tmp_path), _FakeController([_snapshot(eval_ready=1)]), nebius).reconcile(now_ns=1)

    assert receipt.actions == ("start:evaluation",)
    assert nebius.states == {"train-vm-a": "STOPPED", "train-vm-b": "STOPPED", "rollout-vm": "RUNNING"}


def test_capacity_treats_a_final_evaluation_lease_as_rollout_demand(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_capacity import (
        CapacityLifecycle,
        PRODUCTION_EVALUATOR_WORKER_ID,
    )

    final_lease = {
        "lease_id": "f" * 32,
        "experiment_id": "e" * 64,
        "worker_id": PRODUCTION_EVALUATOR_WORKER_ID,
        "capability": "final_evaluation",
        "expires_ns": 99,
    }
    nebius = _FakeNebius({"train-vm-a": "STOPPED", "train-vm-b": "STOPPED", "rollout-vm": "STOPPED"})

    receipt = CapacityLifecycle(
        _config(tmp_path), _FakeController([_snapshot(leases=(final_lease,))]), nebius,
    ).reconcile(now_ns=1)

    assert receipt.actions == ("start:evaluation",)
    assert nebius.states["rollout-vm"] == "RUNNING"


@pytest.mark.parametrize("capability", ("evaluation", "final_evaluation"))
def test_capacity_never_stops_the_production_evaluator_during_any_active_evaluation_lease(tmp_path: Path, capability: str) -> None:
    """Both evaluator lanes bind to the deployment-gated rollout identity."""
    from lehome_train.groot.experiment_capacity import (
        CapacityConfig,
        CapacityLifecycle,
        ManagedWorker,
        PRODUCTION_EVALUATOR_WORKER_ID,
    )

    gate, digest = _gate(tmp_path)
    config = CapacityConfig(
        training_workers=(ManagedWorker("train-vm-a", "train-a"), ManagedWorker("train-vm-b", "train-b")),
        rollout_worker=ManagedWorker("rollout-vm", PRODUCTION_EVALUATOR_WORKER_ID),
        idle_seconds=600,
        operation_cap=3,
        deployment_gate_path=gate,
        deployment_gate_sha256=digest,
    )
    active_lease = {
        "lease_id": "f" * 32,
        "experiment_id": "e" * 64,
        "worker_id": PRODUCTION_EVALUATOR_WORKER_ID,
        "capability": capability,
        "expires_ns": 900_000_000_000,
    }
    nebius = _FakeNebius({"train-vm-a": "STOPPED", "train-vm-b": "STOPPED", "rollout-vm": "RUNNING"})
    lifecycle = CapacityLifecycle(config, _FakeController([_snapshot(leases=(active_lease,))]), nebius)

    assert lifecycle.reconcile(now_ns=1).actions == ()
    assert lifecycle.reconcile(now_ns=601_000_000_001).actions == ()
    assert nebius.states["rollout-vm"] == "RUNNING"


def test_capacity_rejects_unknown_active_lease_capabilities_without_provider_calls(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_capacity import CapacityLifecycle

    unknown = {
        "lease_id": "f" * 32,
        "experiment_id": "e" * 64,
        "worker_id": "eval-a",
        "capability": "mystery",
        "expires_ns": 99,
    }
    nebius = _FakeNebius({"train-vm-a": "STOPPED", "train-vm-b": "STOPPED", "rollout-vm": "STOPPED"})

    receipt = CapacityLifecycle(
        _config(tmp_path), _FakeController([_snapshot(leases=(unknown,))]), nebius,
    ).reconcile(now_ns=1)

    assert receipt.status == "controller_unavailable"
    assert receipt.actions == ()
    assert nebius.calls == []


def test_capacity_does_not_mutate_when_controller_is_unavailable(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_capacity import CapacityLifecycle, ControllerUnavailable

    nebius = _FakeNebius({"train-vm-a": "STOPPED", "train-vm-b": "STOPPED", "rollout-vm": "STOPPED"})
    receipt = CapacityLifecycle(_config(tmp_path), _FakeController(ControllerUnavailable("offline")), nebius).reconcile(now_ns=1)

    assert receipt.status == "controller_unavailable"
    assert receipt.actions == ()
    assert nebius.calls == []


def test_capacity_never_exceeds_the_configured_operation_cap(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_capacity import (
        CapacityConfig,
        CapacityLifecycle,
        ManagedWorker,
        PRODUCTION_EVALUATOR_WORKER_ID,
    )

    gate, digest = _gate(tmp_path)
    config = CapacityConfig(
        training_workers=(ManagedWorker("train-vm-a", "train-a"), ManagedWorker("train-vm-b", "train-b")),
        rollout_worker=ManagedWorker("rollout-vm", PRODUCTION_EVALUATOR_WORKER_ID),
        idle_seconds=600,
        operation_cap=1,
        deployment_gate_path=gate,
        deployment_gate_sha256=digest,
    )
    nebius = _FakeNebius({"train-vm-a": "STOPPED", "train-vm-b": "STOPPED", "rollout-vm": "STOPPED"})
    receipt = CapacityLifecycle(config, _FakeController([_snapshot(ready_training=2, eval_ready=1)]), nebius).reconcile(now_ns=1)

    assert receipt.actions == ("start:training",)
    assert [call for call in nebius.calls if call[0] in {"start", "stop"}] == [("start", "train-vm-a")]


def test_capacity_never_stops_a_training_vm_with_its_active_lease(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_capacity import CapacityLifecycle

    nebius = _FakeNebius({"train-vm-a": "RUNNING", "train-vm-b": "RUNNING", "rollout-vm": "STOPPED"})
    lifecycle = CapacityLifecycle(_config(tmp_path), _FakeController([_snapshot(leases=({"lease_id": "a" * 32, "experiment_id": "b" * 64, "worker_id": "train-a", "capability": "training", "expires_ns": 99},))]), nebius)
    first = lifecycle.reconcile(now_ns=1)
    second = lifecycle.reconcile(now_ns=602_000_000_000)

    assert first.actions == () and second.actions == ("stop:training",)
    assert nebius.states["train-vm-a"] == "RUNNING"
    assert nebius.states["train-vm-b"] == "STOPPED"


def test_capacity_uses_backpressure_aware_leaseable_training_demand_for_idle_and_resume(tmp_path: Path) -> None:
    """Queued work must not keep paid trainers alive when it cannot lease.

    The controller deliberately pauses dependent continuations while more than
    two evaluations wait.  Its authenticated snapshot must distinguish those
    raw READY records from jobs a training worker may actually lease.  The
    lifecycle then starts the 600-second idle clock, stops the exact existing
    trainer VMs, and resumes them as soon as evaluation drains.
    """
    from lehome_train.groot.experiment_capacity import CapacityLifecycle

    def backpressured_snapshot() -> dict[str, object]:
        return _snapshot(ready_training=2, leaseable_training=0, eval_ready=3)

    def leaseable_snapshot() -> dict[str, object]:
        return _snapshot(ready_training=2, leaseable_training=2)

    nebius = _FakeNebius({"train-vm-a": "RUNNING", "train-vm-b": "RUNNING", "rollout-vm": "RUNNING"})
    lifecycle = CapacityLifecycle(
        _config(tmp_path),
        _FakeController([backpressured_snapshot(), backpressured_snapshot(), leaseable_snapshot()]),
        nebius,
    )

    assert lifecycle.reconcile(now_ns=1).actions == ()
    assert lifecycle.reconcile(now_ns=600_000_000_001).actions == ("stop:training", "stop:training")
    assert nebius.states == {"train-vm-a": "STOPPED", "train-vm-b": "STOPPED", "rollout-vm": "RUNNING"}
    assert lifecycle.reconcile(now_ns=600_000_000_002).actions == ("start:training", "start:training")
    assert nebius.states["train-vm-a"] == nebius.states["train-vm-b"] == "RUNNING"
    assert all(action not in {"create", "delete"} for action, _ in nebius.calls)


def test_capacity_backpressure_never_stops_an_existing_active_training_lease(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_capacity import CapacityLifecycle

    active = {
        "lease_id": "a" * 32,
        "experiment_id": "b" * 64,
        "worker_id": "train-a",
        "capability": "training",
        "expires_ns": 9_999_999_999_999,
    }
    snapshot = _snapshot(ready_training=2, leaseable_training=0, eval_ready=3, leases=(active,))
    nebius = _FakeNebius({"train-vm-a": "RUNNING", "train-vm-b": "STOPPED", "rollout-vm": "RUNNING"})
    lifecycle = CapacityLifecycle(_config(tmp_path), _FakeController([snapshot, snapshot]), nebius)

    assert lifecycle.reconcile(now_ns=1).actions == ()
    assert lifecycle.reconcile(now_ns=600_000_000_001).actions == ()
    assert nebius.states["train-vm-a"] == "RUNNING"


def test_capacity_stops_only_after_six_hundred_seconds_of_controller_confirmed_idle(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_capacity import CapacityLifecycle

    nebius = _FakeNebius({"train-vm-a": "RUNNING", "train-vm-b": "STOPPED", "rollout-vm": "RUNNING"})
    lifecycle = CapacityLifecycle(_config(tmp_path), _FakeController([_snapshot(), _snapshot(), _snapshot()]), nebius)

    assert lifecycle.reconcile(now_ns=1).actions == ()
    assert lifecycle.reconcile(now_ns=599_000_000_000).actions == ()
    assert lifecycle.reconcile(now_ns=600_000_000_001).actions == ("stop:training", "stop:evaluation")


def test_capacity_receipt_is_append_only_and_does_not_disclose_vm_identifiers(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_capacity import CapacityLifecycle

    nebius = _FakeNebius({"train-vm-a": "STOPPED", "train-vm-b": "STOPPED", "rollout-vm": "STOPPED"})
    lifecycle = CapacityLifecycle(_config(tmp_path), _FakeController([_snapshot(ready_training=1)]), nebius)
    first = lifecycle.reconcile(now_ns=1)
    second = lifecycle.reconcile(now_ns=2)
    receipt_log = tmp_path / "receipts.jsonl"
    lifecycle.append_receipt(receipt_log, first)
    lifecycle.append_receipt(receipt_log, second)

    lines = receipt_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "train-vm-a" not in receipt_log.read_text(encoding="utf-8")
    assert hashlib.sha256(b"train-vm-a").hexdigest() in lines[0]


def test_capacity_config_requires_exact_two_training_and_one_rollout_workers(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_capacity import load_capacity_config

    config = tmp_path / "capacity.json"
    config.write_text('{"schema_version":1}', encoding="utf-8")
    config.chmod(0o600)
    with pytest.raises(ValueError, match="capacity config"):
        load_capacity_config(config)


def test_capacity_config_rejects_a_nonproduction_evaluator_identity(tmp_path: Path) -> None:
    """The single rollout VM cannot be configured with an unrelated worker ID."""
    from lehome_train.groot.experiment_capacity import CapacityConfig, ManagedWorker

    gate, digest = _gate(tmp_path)
    with pytest.raises(ValueError, match="rollout worker identity"):
        CapacityConfig(
            training_workers=(ManagedWorker("train-vm-a", "train-a"), ManagedWorker("train-vm-b", "train-b")),
            rollout_worker=ManagedWorker("rollout-vm", "rollout-evaluator"),
            idle_seconds=600,
            operation_cap=3,
            deployment_gate_path=gate,
            deployment_gate_sha256=digest,
        )


def test_capacity_refuses_paid_start_when_the_deployment_gate_is_missing_or_tampered(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_capacity import (
        CapacityConfig,
        CapacityLifecycle,
        ManagedWorker,
        PRODUCTION_EVALUATOR_WORKER_ID,
    )

    gate = tmp_path / "deployment-gate.json"
    gate.write_bytes(b"verified-gate\n")
    gate.chmod(0o444)
    digest = hashlib.sha256(gate.read_bytes()).hexdigest()
    config = CapacityConfig(
        training_workers=(ManagedWorker("train-vm-a", "train-a"), ManagedWorker("train-vm-b", "train-b")),
        rollout_worker=ManagedWorker("rollout-vm", PRODUCTION_EVALUATOR_WORKER_ID),
        idle_seconds=600,
        operation_cap=3,
        deployment_gate_path=gate,
        deployment_gate_sha256=digest,
    )
    gate.chmod(0o644)
    gate.write_bytes(b"tampered\n")
    nebius = _FakeNebius({"train-vm-a": "STOPPED", "train-vm-b": "STOPPED", "rollout-vm": "STOPPED"})

    receipt = CapacityLifecycle(config, _FakeController([_snapshot(ready_training=2, eval_ready=1)]), nebius).reconcile(now_ns=1)

    assert receipt.status == "deployment_gate_unavailable"
    assert receipt.actions == ()
    assert nebius.calls == []


def test_http_capacity_controller_requires_and_uses_a_safe_private_ca(tmp_path, monkeypatch) -> None:
    from lehome_train.groot import experiment_capacity as module

    token = tmp_path / "controller-token"; token.write_text("secret\n"); token.chmod(0o600)
    with pytest.raises(ValueError, match="private CA"):
        module.HttpCapacityController("https://controller", token, None)

    ca = tmp_path / "controller-ca.crt"; ca.write_text("test CA\n"); ca.chmod(0o644)
    unsafe = tmp_path / "ca-link.crt"; unsafe.symlink_to(ca)
    with pytest.raises(ValueError, match="private CA"):
        module.HttpCapacityController("https://controller", token, unsafe)

    context = object()
    observed: dict[str, object] = {}
    monkeypatch.setattr(module.ssl, "create_default_context", lambda *, cafile: observed.update(cafile=cafile) or context)

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_args): return None
        def read(self): return b'{"schema_version":1}'

    def open_request(_request, *, timeout, context):
        observed.update(timeout=timeout, context=context)
        return Response()

    monkeypatch.setattr(module, "urlopen", open_request)
    controller = module.HttpCapacityController("https://controller", token, ca)
    assert controller.capacity_snapshot() == {"schema_version": 1}
    assert observed == {"cafile": str(ca), "timeout": 20, "context": context}
