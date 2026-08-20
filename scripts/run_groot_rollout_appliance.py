"""Supervise the complete four-worker LeHome rollout appliance.

Process topology on one preemptible RTX PRO 6000 (24 vCPU) machine:

    GPU:  one batched policy server (single loaded checkpoint)
    CPU:  one controller, one finalizer/writer pool, one HF uploader,
          and four persistent CPU-cloth Isaac workers (5 cores each)

The model server starts before any worker; no worker starts until shared-disk
admission and the policy readiness digest both verify.  Four workers is the
production default; any lower count requires an explicit debug flag and is
recorded in the startup receipt.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from dataclasses import replace as dataclasses_replace
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4


DEFAULT_POLICY_GATEWAY_ENDPOINT = "tcp://127.0.0.1:15555"
DEFAULT_POLICY_TIMEOUT_SECONDS = 180
APPLIANCE_SERVICES = (
    "workspace_admission",
    "policy_server",
    "controller",
    "finalizer_pool",
    "uploader",
    "worker",
)
PRODUCTION_WORKER_COUNT = 4
WORKER_CORES = 5
SHARED_SERVICE_CORES = 4


class ApplianceError(RuntimeError):
    """The appliance must not start or must stop on a contract violation."""


@dataclass(slots=True)
class ApplianceConfig:
    """Immutable appliance inputs; debug_low_worker_count is opt-in only."""

    workspace_root: Path
    worker_count: int
    vcpu_budget: int
    policy_sha256: str
    policy_gateway_endpoint: str
    renderer_device: str
    policy_device: str
    debug_low_worker_count: bool = False
    database: Path | None = None
    attempt_matrix: Path | None = None
    policy_ready_file: Path | None = None
    initial_garment: str = "Top_Long_Unseen_0"

    def __post_init__(self) -> None:
        if self.worker_count < 1:
            raise ApplianceError("worker_count must be positive")


@dataclass(frozen=True, slots=True)
class ServiceSpec:
    """One supervised process: identity, command, and pinned CPU cores."""

    name: str
    service: str
    command: tuple[str, ...]
    cpu_cores: tuple[int, ...]
    output_root: Path | None = None


@dataclass(frozen=True, slots=True)
class ProcessTopology:
    services: tuple[ServiceSpec, ...]
    recorded_worker_count_override: int | None
    started: bool = False


def build_process_topology(config: ApplianceConfig) -> ProcessTopology:
    """Plan the process layout and CPU affinity before anything spawns.

    Core budget: ``worker_count * 5 + 4`` must fit the vCPU budget.  With 24
    vCPUs and four workers that is exactly 24; this allocation is a starting
    configuration, not a measured throughput claim.
    """
    if config.worker_count > PRODUCTION_WORKER_COUNT:
        raise ApplianceError("the appliance never runs more than four workers")
    if config.worker_count < PRODUCTION_WORKER_COUNT and not config.debug_low_worker_count:
        raise ApplianceError(
            "four workers is the production default; a lower count requires the explicit debug flag"
        )
    required_cores = config.worker_count * WORKER_CORES + SHARED_SERVICE_CORES
    if required_cores > config.vcpu_budget:
        raise ApplianceError(
            f"appliance needs {required_cores} cores but the vcpu budget is {config.vcpu_budget}"
        )
    services: list[ServiceSpec] = []
    core = 0

    def allocate(count: int) -> tuple[int, ...]:
        nonlocal core
        cores = tuple(range(core, core + count))
        core += count
        return cores

    policy_port = int(config.policy_gateway_endpoint.rsplit(":", 1)[-1])
    services.append(ServiceSpec(
        name="policy-server",
        service="policy_server",
        command=(
            "run_groot_batched_policy_server.py",
            "--host", "127.0.0.1", "--port", str(policy_port),
            "--policy-sha256", config.policy_sha256,
            "--device", config.policy_device,
        ),
        cpu_cores=allocate(1),
    ))
    services.append(ServiceSpec(
        name="controller",
        service="controller",
        command=("run_groot_rollout_controller.py",),
        cpu_cores=allocate(1),
    ))
    services.append(ServiceSpec(
        name="finalizer-pool",
        service="finalizer_pool",
        command=("run_groot_artifact_sync.py", "--role", "finalizer"),
        cpu_cores=allocate(1),
    ))
    services.append(ServiceSpec(
        name="uploader",
        service="uploader",
        command=("run_groot_artifact_sync.py", "--role", "uploader"),
        cpu_cores=allocate(1),
    ))
    workspace = config.workspace_root
    database = config.database or (workspace / "rollouts" / "ledger.sqlite3")
    attempt_matrix = config.attempt_matrix or (workspace / "eval" / "matrices" / "unseen-80.json")
    policy_ready_file = config.policy_ready_file or (
        workspace / "eval" / "receipts" / "policy" / "ready.json"
    )
    for index in range(config.worker_count):
        worker_root = config.workspace_root / "rollouts" / "attempts" / f"worker-{index}"
        session_id = f"worker-{index}-{uuid4().hex}"
        services.append(ServiceSpec(
            name=f"worker-{index}",
            service="worker",
            command=(
                "run_groot_persistent_worker.py",
                "--worker-id", f"worker-{index}",
                "--session-id", session_id,
                "--policy-gateway-endpoint", config.policy_gateway_endpoint,
                "--policy-sha256", config.policy_sha256,
                "--renderer-device", config.renderer_device,
                "--policy-device", config.policy_device,
                "--output-root", str(worker_root),
                "--database", str(database),
                "--attempt-matrix", str(attempt_matrix),
                "--policy-ready-file", str(policy_ready_file),
                "--initial-garment", config.initial_garment,
                "--policy-timeout-seconds", str(DEFAULT_POLICY_TIMEOUT_SECONDS),
            ),
            cpu_cores=allocate(WORKER_CORES),
            output_root=worker_root,
        ))
    override = config.worker_count if config.worker_count != PRODUCTION_WORKER_COUNT else None
    return ProcessTopology(tuple(services), override)


class ChildProcess:
    """Minimal injected child handle mirroring subprocess.Popen semantics."""

    def poll(self) -> int | None: ...
    def terminate(self) -> None: ...


class Spawner:
    """Production spawner; tests inject a fake with the same interface."""

    def spawn(self, service: str, command: Sequence[str], cpu_cores: Sequence[int]) -> ChildProcess:
        raise NotImplementedError("production spawning happens on the rollout VM")


@dataclass(slots=True)
class RolloutAppliance:
    """Ordered startup, restart-limited supervision, coordinated shutdown."""

    config: ApplianceConfig
    spawner: Spawner
    max_restarts: int = 1
    topology: ProcessTopology | None = field(default=None, init=False)
    children: dict[str, Any] = field(default_factory=dict, init=False)
    restarts: dict[str, int] = field(default_factory=dict, init=False)
    stopped: bool = field(default=False, init=False)

    # Injection points for the two external gates; the real implementations
    # run shared-disk admission and poll the policy readiness JSON.
    admit_workspace: Callable[[], Mapping[str, object]] = None  # type: ignore[assignment]
    wait_policy_ready: Callable[[], Mapping[str, object]] = None  # type: ignore[assignment]

    def start(self) -> ProcessTopology:
        if self.admit_workspace is None:
            raise ApplianceError("workspace admission is required before startup")
        manifest = self.admit_workspace()
        if manifest.get("active_role") != "rollout":
            raise ApplianceError("workspace manifest does not lease the rollout role")
        topology = build_process_topology(self.config)
        self.topology = topology

        policy_specs = [spec for spec in topology.services if spec.service == "policy_server"]
        if len(policy_specs) != 1:
            raise ApplianceError("topology must contain exactly one policy server")
        policy_spec = policy_specs[0]
        self.children[policy_spec.name] = self.spawner.spawn(
            policy_spec.service, policy_spec.command, policy_spec.cpu_cores,
        )
        readiness = self.wait_policy_ready() if self.wait_policy_ready else {}
        if readiness.get("ready") is not True:
            self.shutdown(reason="policy not ready")
            raise ApplianceError("policy server did not become ready")
        if readiness.get("digest") != self.config.policy_sha256:
            self.shutdown(reason="policy digest mismatch")
            raise ApplianceError(
                f"policy readiness digest mismatch: {readiness.get('digest')!r} "
                f"versus expected {self.config.policy_sha256!r}"
            )

        for spec in topology.services:
            if spec.service == "policy_server":
                continue
            self.children[spec.name] = self.spawner.spawn(spec.service, spec.command, spec.cpu_cores)
        return dataclasses_replace(topology, started=True)

    def supervise_once(self) -> None:
        """One supervision pass: restart dead children within limits or stop."""
        if self.stopped or self.topology is None:
            return
        spec_by_name = {spec.name: spec for spec in self.topology.services}
        for name, child in list(self.children.items()):
            exit_code = child.poll()
            if exit_code is None:
                continue
            self.restarts[name] = self.restarts.get(name, 0) + 1
            if self.restarts[name] > self.max_restarts:
                self.shutdown(reason="restart limit exceeded")
                raise ApplianceError(f"{name} exceeded the restart limit ({self.max_restarts})")
            spec = spec_by_name[name]
            self.children[name] = self.spawner.spawn(spec.service, spec.command, spec.cpu_cores)

    def shutdown(self, reason: str) -> None:
        self.stopped = True
        for child in self.children.values():
            if child.poll() is None:
                child.terminate()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--worker-count", type=int, default=PRODUCTION_WORKER_COUNT)
    parser.add_argument("--vcpu-budget", type=int, default=24)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--policy-gateway-endpoint", default=DEFAULT_POLICY_GATEWAY_ENDPOINT)
    parser.add_argument("--renderer-device", default="cuda:0")
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--debug-low-worker-count", action="store_true")
    parser.add_argument("--receipt-file", type=Path, required=True)
    args = parser.parse_args(argv)

    config = ApplianceConfig(
        workspace_root=args.workspace_root,
        worker_count=args.worker_count,
        vcpu_budget=args.vcpu_budget,
        policy_sha256=args.policy_sha256,
        policy_gateway_endpoint=args.policy_gateway_endpoint,
        renderer_device=args.renderer_device,
        policy_device=args.policy_device,
        debug_low_worker_count=args.debug_low_worker_count,
    )
    topology = build_process_topology(config)
    receipt = {
        "schema_version": 1,
        "kind": "rollout_appliance_plan",
        "worker_count": config.worker_count,
        "worker_count_override_recorded": topology.recorded_worker_count_override,
        "services": [
            {"name": spec.name, "service": spec.service, "cpu_cores": list(spec.cpu_cores)}
            for spec in topology.services
        ],
    }
    args.receipt_file.parent.mkdir(parents=True, exist_ok=True)
    args.receipt_file.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
