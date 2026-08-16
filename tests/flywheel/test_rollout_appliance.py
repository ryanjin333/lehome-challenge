"""Four-worker rollout appliance: topology, ordering, affinity, shutdown."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "run_groot_rollout_appliance_under_test",
    _REPOSITORY_ROOT / "scripts" / "run_groot_rollout_appliance.py",
)
_module = importlib.util.module_from_spec(_spec)
import sys as _sys
_sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

APPLIANCE_SERVICES = _module.APPLIANCE_SERVICES
ApplianceConfig = _module.ApplianceConfig
ApplianceError = _module.ApplianceError
RolloutAppliance = _module.RolloutAppliance
build_process_topology = _module.build_process_topology


@pytest.fixture()
def config(tmp_path):
    return ApplianceConfig(
        workspace_root=tmp_path / "workspace",
        worker_count=4,
        vcpu_budget=24,
        policy_sha256="a" * 64,
        policy_gateway_endpoint="tcp://127.0.0.1:5555",
        renderer_device="cuda:0",
        policy_device="cuda:0",
    )


def test_topology_has_exactly_one_of_each_service_and_four_workers(config):
    topology = build_process_topology(config)
    counts = {}
    for spec in topology.services:
        counts[spec.service] = counts.get(spec.service, 0) + 1
    assert counts == {
        "policy_server": 1,
        "controller": 1,
        "worker": 4,
        "finalizer_pool": 1,
        "uploader": 1,
    }
    assert tuple(APPLIANCE_SERVICES) == (
        "workspace_admission", "policy_server", "controller",
        "finalizer_pool", "uploader", "worker",
    )


def test_worker_count_below_four_requires_explicit_debug_flag(tmp_path):
    debug_config = ApplianceConfig(
        workspace_root=tmp_path / "workspace", worker_count=2, vcpu_budget=24,
        policy_sha256="a" * 64, policy_gateway_endpoint="tcp://127.0.0.1:5555",
        renderer_device="cuda:0", policy_device="cuda:0",
    )
    with pytest.raises(ApplianceError, match="debug"):
        build_process_topology(debug_config)

    debug_config.debug_low_worker_count = True
    topology = build_process_topology(debug_config)
    assert sum(1 for spec in topology.services if spec.service == "worker") == 2
    assert topology.recorded_worker_count_override == 2


def test_worker_count_above_four_rejected(tmp_path):
    config = ApplianceConfig(
        workspace_root=tmp_path / "workspace", worker_count=5, vcpu_budget=24,
        policy_sha256="a" * 64, policy_gateway_endpoint="tcp://127.0.0.1:5555",
        renderer_device="cuda:0", policy_device="cuda:0",
    )
    with pytest.raises(ApplianceError, match="four"):
        build_process_topology(config)


def test_workers_get_distinct_output_roots(config):
    topology = build_process_topology(config)
    roots = [spec.output_root for spec in topology.services if spec.service == "worker"]
    assert len(set(roots)) == 4
    assert all(root.is_relative_to(config.workspace_root) for root in roots)


def test_cpu_allocation_fits_24_vcpus_without_oversubscription(config):
    topology = build_process_topology(config)
    all_cores: list[int] = []
    for spec in topology.services:
        assert spec.cpu_cores, f"{spec.service} must own explicit cores"
        all_cores.extend(spec.cpu_cores)
    assert len(all_cores) == len(set(all_cores)), "no core may be double-booked"
    assert max(all_cores) < 24
    worker_specs = [spec for spec in topology.services if spec.service == "worker"]
    for spec in worker_specs:
        assert len(spec.cpu_cores) == 5
    shared = [spec for spec in topology.services if spec.service != "worker"]
    assert sum(len(spec.cpu_cores) for spec in shared) == 4


def test_cpu_allocation_rejects_oversubscribed_budget(tmp_path):
    config = ApplianceConfig(
        workspace_root=tmp_path / "workspace", worker_count=4, vcpu_budget=20,
        policy_sha256="a" * 64, policy_gateway_endpoint="tcp://127.0.0.1:5555",
        renderer_device="cuda:0", policy_device="cuda:0",
    )
    with pytest.raises(ApplianceError, match="vcpu"):
        build_process_topology(config)


class FakeChild:
    def __init__(self, service: str, pid: int):
        self.service = service
        self.pid = pid
        self.terminated = False
        self.exit_code = 0
        self.signaled = False

    def poll(self):
        return self.exit_code if self.terminated else None

    def terminate(self):
        self.signaled = True
        self.terminated = True


class FakeSpawner:
    def __init__(self):
        self.spawned: list[tuple[str, tuple[str, ...], tuple[int, ...]]] = []
        self._pid = 100

    def spawn(self, service: str, command, cpu_cores):
        self._pid += 1
        self.spawned.append((service, tuple(command), tuple(cpu_cores)))
        return FakeChild(service, self._pid)


def test_start_order_admission_then_model_then_workers(config):
    spawner = FakeSpawner()
    appliance = RolloutAppliance(config, spawner=spawner)
    appliance.admit_workspace = lambda: {"active_role": "rollout"}  # fake admission
    appliance.wait_policy_ready = lambda: {"ready": True, "digest": "a" * 64}

    topology = appliance.start()

    order = [service for service, _command, _cores in spawner.spawned]
    assert order[0] == "policy_server"
    assert order.index("controller") < order.index("worker")
    assert order.index("policy_server") < order.index("worker")
    assert order.count("worker") == 4
    assert topology.started


def test_start_refuses_without_workspace_admission(config):
    spawner = FakeSpawner()
    appliance = RolloutAppliance(config, spawner=spawner)

    def deny():
        raise ApplianceError("workspace manifest role lease conflict")

    appliance.admit_workspace = deny
    with pytest.raises(ApplianceError, match="workspace"):
        appliance.start()
    assert spawner.spawned == []


def test_start_refuses_policy_digest_mismatch(config):
    spawner = FakeSpawner()
    appliance = RolloutAppliance(config, spawner=spawner)
    appliance.admit_workspace = lambda: {"active_role": "rollout"}
    appliance.wait_policy_ready = lambda: {"ready": True, "digest": "b" * 64}

    with pytest.raises(ApplianceError, match="digest"):
        appliance.start()


def test_child_failure_propagates_and_stops_appliance(config):
    spawner = FakeSpawner()
    appliance = RolloutAppliance(config, spawner=spawner, max_restarts=0)
    appliance.admit_workspace = lambda: {"active_role": "rollout"}
    appliance.wait_policy_ready = lambda: {"ready": True, "digest": "a" * 64}
    appliance.start()

    failed = appliance.children["worker-0"]
    failed.exit_code = 1
    failed.terminated = True

    with pytest.raises(ApplianceError, match="restart limit"):
        appliance.supervise_once()
    assert appliance.stopped
    for child in appliance.children.values():
        # Already-dead children are never re-signaled; live ones must be.
        assert child.signaled or child.terminated


def test_restart_within_limit_recovers(config):
    spawner = FakeSpawner()
    appliance = RolloutAppliance(config, spawner=spawner, max_restarts=1)
    appliance.admit_workspace = lambda: {"active_role": "rollout"}
    appliance.wait_policy_ready = lambda: {"ready": True, "digest": "a" * 64}
    appliance.start()

    failed = appliance.children["worker-0"]
    failed.exit_code = 1
    failed.terminated = True
    appliance.supervise_once()

    assert not appliance.stopped
    assert appliance.children["worker-0"].signaled is False
    worker_spawns = [s for s in spawner.spawned if s[0] == "worker"]
    assert len(worker_spawns) == 5


def test_shutdown_sends_sigterm_to_every_child(config):
    spawner = FakeSpawner()
    appliance = RolloutAppliance(config, spawner=spawner)
    appliance.admit_workspace = lambda: {"active_role": "rollout"}
    appliance.wait_policy_ready = lambda: {"ready": True, "digest": "a" * 64}
    appliance.start()

    appliance.shutdown(reason="sigterm")

    assert appliance.stopped
    for child in appliance.children.values():
        assert child.signaled
