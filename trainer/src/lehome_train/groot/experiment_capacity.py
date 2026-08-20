"""Fail-closed lifecycle decisions for the fixed asynchronous sweep pool.

This module deliberately knows nothing about credentials or VM creation.  A
caller supplies the three already-created instance identities and a tiny
Nebius adapter exposing only state, start, and stop operations.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import ssl
import stat
import subprocess
import time
from typing import Callable, Mapping, Protocol
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from lehome_train.groot.experiment_deployment_gate import PRODUCTION_EVALUATOR_WORKER_ID
from lehome_train.io import canonical_json_sha256


class ControllerUnavailable(RuntimeError):
    """The authenticated controller cannot provide a trustworthy snapshot."""


class CapacityController(Protocol):
    def capacity_snapshot(self) -> Mapping[str, object]: ...


class NebiusInstanceRunner(Protocol):
    def instance_state(self, instance_id: str) -> str: ...
    def start_instance(self, instance_id: str) -> None: ...
    def stop_instance(self, instance_id: str) -> None: ...


_COMPUTE_INSTANCE_ID_PREFIX = "computeinstance-"

def _require_compute_instance_id(instance_id: str) -> str:
    """Accept only a concrete Nebius Compute instance identity."""
    if (
        type(instance_id) is not str
        or not instance_id.startswith(_COMPUTE_INSTANCE_ID_PREFIX)
        or len(instance_id) <= len(_COMPUTE_INSTANCE_ID_PREFIX)
        or not instance_id[len(_COMPUTE_INSTANCE_ID_PREFIX) :].isalnum()
    ):
        raise ValueError("Nebius instance identity is invalid")
    return instance_id


class NebiusCliInstanceRunner:
    """Minimal production adapter for already-created Nebius instances only.

    The command surface is intentionally restricted to instance ``get``,
    ``start``, and ``stop``. It does not accept a shell command or expose any
    create/delete capability. Provider failures retry a bounded number of
    times; malformed or mismatched JSON fails closed immediately.
    """

    def __init__(
        self,
        *,
        subprocess_run: Callable[..., object] = subprocess.run,
        timeout_seconds: int = 30,
        max_attempts: int = 3,
        sleep: Callable[[float], None] = time.sleep,
        provider_config_file: str | Path | None = None,
        provider_config_owner_uid: int | None = None,
    ) -> None:
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 120:
            raise ValueError("Nebius CLI timeout is invalid")
        if type(max_attempts) is not int or not 1 <= max_attempts <= 3:
            raise ValueError("Nebius CLI attempt count is invalid")
        self._subprocess_run = subprocess_run
        self._timeout_seconds = timeout_seconds
        self._max_attempts = max_attempts
        self._sleep = sleep
        if provider_config_file is None:
            if provider_config_owner_uid is not None:
                raise ValueError("Nebius provider auth owner is invalid")
            self._provider_config_file: Path | None = None
            self._provider_config_owner_uid: int | None = None
        else:
            config = Path(provider_config_file)
            if not config.is_absolute() or type(provider_config_owner_uid) is not int or provider_config_owner_uid < 0:
                raise ValueError("Nebius provider auth config is invalid")
            self._provider_config_file = config
            self._provider_config_owner_uid = provider_config_owner_uid

    def _require_provider_auth_config(self) -> Path | None:
        """Fail closed instead of allowing a process-default CLI profile.

        Unit-level adapters may omit this binding.  The production entrypoint
        always supplies it and pins the expected root owner, so an environment
        variable, a home-directory profile, or a swapped config cannot become
        authority for a Compute mutation.
        """
        config = self._provider_config_file
        if config is None:
            return None
        try:
            details = config.stat()
            if (
                config.is_symlink()
                or not config.is_file()
                or details.st_uid != self._provider_config_owner_uid
                or stat.S_IMODE(details.st_mode) != 0o600
                or not 1 <= details.st_size <= 65536
            ):
                raise RuntimeError("Nebius provider authentication config is unsafe")
        except OSError as error:
            raise RuntimeError("Nebius provider authentication config is unsafe") from error
        return config

    def _command(self, operation: str, instance_id: str) -> tuple[str, ...]:
        if operation not in {"get", "start", "stop"}:
            raise ValueError("Nebius CLI operation is invalid")
        command = ["nebius"]
        config = self._require_provider_auth_config()
        if config is not None:
            command.extend(("--config", str(config)))
        command.extend((
            "compute",
            "instance",
            operation,
            "--id",
            _require_compute_instance_id(instance_id),
            "--format",
            "json",
            "--no-progress",
            "--timeout",
            f"{self._timeout_seconds}s",
            "--no-browser",
            "--no-check-update",
            "--retries",
            "1",
        ))
        return tuple(command)

    @staticmethod
    def _parse_response(payload: object, instance_id: str, *, require_state: bool) -> Mapping[str, object]:
        if not isinstance(payload, str):
            raise RuntimeError("Nebius CLI returned invalid JSON state")
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as error:
            raise RuntimeError("Nebius CLI returned invalid JSON state") from error
        if not isinstance(value, Mapping):
            raise RuntimeError("Nebius CLI returned invalid JSON state")
        metadata = value.get("metadata")
        if not isinstance(metadata, Mapping) or metadata.get("id") != instance_id:
            raise RuntimeError("Nebius CLI instance identity mismatch")
        status = value.get("status")
        state = status.get("state") if isinstance(status, Mapping) else None
        if require_state and state not in {"STOPPED", "RUNNING"}:
            raise RuntimeError("Nebius CLI returned invalid state")
        return value

    def _run(self, operation: str, instance_id: str, *, require_state: bool) -> Mapping[str, object]:
        command = self._command(operation, instance_id)
        last_error: BaseException | None = None
        for attempt in range(self._max_attempts):
            try:
                completed = self._subprocess_run(
                    command,
                    timeout=self._timeout_seconds + 5,
                    text=True,
                    capture_output=True,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                last_error = error
            else:
                if getattr(completed, "returncode", None) == 0:
                    return self._parse_response(getattr(completed, "stdout", None), instance_id, require_state=require_state)
                last_error = RuntimeError("Nebius CLI operation failed")
            if attempt + 1 < self._max_attempts:
                self._sleep(float(attempt + 1))
        raise RuntimeError("Nebius CLI operation failed") from last_error

    def instance_state(self, instance_id: str) -> str:
        result = self._run("get", instance_id, require_state=True)
        status = result["status"]
        assert isinstance(status, Mapping)  # validated in _parse_response
        return str(status["state"])

    def start_instance(self, instance_id: str) -> None:
        self._run("start", instance_id, require_state=False)

    def stop_instance(self, instance_id: str) -> None:
        self._run("stop", instance_id, require_state=False)


@dataclass(frozen=True, slots=True)
class ManagedWorker:
    instance_id: str
    worker_id: str

    def __post_init__(self) -> None:
        if not all(type(item) is str and item for item in (self.instance_id, self.worker_id)):
            raise ValueError("managed worker identity is invalid")


@dataclass(frozen=True, slots=True)
class CapacityConfig:
    training_workers: tuple[ManagedWorker, ManagedWorker]
    rollout_worker: ManagedWorker
    idle_seconds: int
    operation_cap: int
    deployment_gate_path: Path
    deployment_gate_sha256: str

    def __post_init__(self) -> None:
        workers = (*self.training_workers, self.rollout_worker)
        if len(self.training_workers) != 2 or self.idle_seconds < 600 or not 1 <= self.operation_cap <= 3:
            raise ValueError("capacity config is invalid")
        if len({item.instance_id for item in workers}) != 3 or len({item.worker_id for item in workers}) != 3:
            raise ValueError("capacity config has duplicate worker identities")
        if self.rollout_worker.worker_id != PRODUCTION_EVALUATOR_WORKER_ID:
            raise ValueError("capacity config rollout worker identity is invalid")
        if (
            not isinstance(self.deployment_gate_path, Path)
            or not self.deployment_gate_path.is_absolute()
            or type(self.deployment_gate_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", self.deployment_gate_sha256) is None
        ):
            raise ValueError("capacity deployment gate binding is invalid")


@dataclass(frozen=True, slots=True)
class CapacityReceipt:
    status: str
    now_ns: int
    snapshot_sha256: str | None
    actions: tuple[str, ...]
    action_instance_sha256: tuple[str, ...]

    def canonical(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "lehome_fixed_pool_capacity_receipt",
            "status": self.status,
            "now_ns": self.now_ns,
            "snapshot_sha256": self.snapshot_sha256,
            "actions": [
                {"action": action, "instance_sha256": identity}
                for action, identity in zip(self.actions, self.action_instance_sha256, strict=True)
            ],
        }


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_snapshot(value: Mapping[str, object]) -> dict[str, object]:
    snapshot = dict(value)
    required = {
        "schema_version",
        "ready_training_count",
        "leaseable_training_count",
        "eval_ready_count",
        "active_leases",
        "idle_stop_recommended",
    }
    if set(snapshot) != required or snapshot.get("schema_version") != 1:
        raise ControllerUnavailable("controller capacity snapshot is malformed")
    if any(
        type(snapshot[key]) is not int or int(snapshot[key]) < 0
        for key in ("ready_training_count", "leaseable_training_count", "eval_ready_count")
    ) or int(snapshot["leaseable_training_count"]) > int(snapshot["ready_training_count"]) or type(snapshot["idle_stop_recommended"]) is not bool:
        raise ControllerUnavailable("controller capacity snapshot is malformed")
    leases = snapshot["active_leases"]
    if not isinstance(leases, list):
        raise ControllerUnavailable("controller capacity snapshot is malformed")
    canonical_leases: list[dict[str, object]] = []
    seen: set[str] = set()
    for lease in leases:
        if not isinstance(lease, Mapping) or set(lease) != {"lease_id", "experiment_id", "worker_id", "capability", "expires_ns"}:
            raise ControllerUnavailable("controller capacity snapshot is malformed")
        if not all(type(lease[key]) is str and lease[key] for key in ("lease_id", "experiment_id", "worker_id")) or lease.get("capability") not in {"training", "evaluation", "final_evaluation"} or type(lease.get("expires_ns")) is not int or str(lease["lease_id"]) in seen:
            raise ControllerUnavailable("controller capacity snapshot is malformed")
        seen.add(str(lease["lease_id"]))
        canonical_leases.append(dict(lease))
    snapshot["active_leases"] = canonical_leases
    expected_idle = (
        int(snapshot["leaseable_training_count"]) == 0
        and not any(lease["capability"] == "training" for lease in canonical_leases)
    )
    if snapshot["idle_stop_recommended"] is not expected_idle:
        raise ControllerUnavailable("controller capacity snapshot is malformed")
    return snapshot


class CapacityLifecycle:
    """Make one bounded, idempotent decision from an authenticated snapshot."""

    def __init__(self, config: CapacityConfig, controller: CapacityController, nebius: NebiusInstanceRunner) -> None:
        self.config = config
        self.controller = controller
        self.nebius = nebius
        self._idle_since_ns: dict[str, int] = {}

    def _receipt(self, status: str, now_ns: int, snapshot: Mapping[str, object] | None, actions: list[tuple[str, ManagedWorker]]) -> CapacityReceipt:
        return CapacityReceipt(
            status=status,
            now_ns=now_ns,
            snapshot_sha256=None if snapshot is None else canonical_json_sha256(snapshot),
            actions=tuple(action for action, _ in actions),
            action_instance_sha256=tuple(_sha256(worker.instance_id) for _, worker in actions),
        )

    def reconcile(self, *, now_ns: int) -> CapacityReceipt:
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("capacity clock is invalid")
        gate = self.config.deployment_gate_path
        try:
            if (
                gate.is_symlink()
                or not gate.is_file()
                or stat.S_IMODE(gate.stat().st_mode) != 0o444
                or hashlib.sha256(gate.read_bytes()).hexdigest() != self.config.deployment_gate_sha256
            ):
                return self._receipt("deployment_gate_unavailable", now_ns, None, [])
        except OSError:
            return self._receipt("deployment_gate_unavailable", now_ns, None, [])
        try:
            snapshot = _validate_snapshot(self.controller.capacity_snapshot())
        except (ControllerUnavailable, TimeoutError, OSError):
            return self._receipt("controller_unavailable", now_ns, None, [])

        # Query all instances before any mutation.  A partial provider outage
        # must not turn into a partially applied capacity decision.
        workers = (*self.config.training_workers, self.config.rollout_worker)
        try:
            states = {worker.instance_id: self.nebius.instance_state(worker.instance_id) for worker in workers}
        except (TimeoutError, OSError, RuntimeError):
            return self._receipt("provider_unavailable", now_ns, snapshot, [])
        if any(state not in {"STOPPED", "RUNNING"} for state in states.values()):
            return self._receipt("provider_unavailable", now_ns, snapshot, [])

        leases = tuple(snapshot["active_leases"])
        active_training_workers = {str(item["worker_id"]) for item in leases if item["capability"] == "training"}
        active_evaluation_workers = {str(item["worker_id"]) for item in leases if item["capability"] in {"evaluation", "final_evaluation"}}
        leaseable_training = int(snapshot["leaseable_training_count"])
        ready_evaluation = int(snapshot["eval_ready_count"])
        actions: list[tuple[str, ManagedWorker]] = []

        def add(action: str, worker: ManagedWorker) -> bool:
            if len(actions) >= self.config.operation_cap:
                return False
            actions.append((action, worker))
            return True

        # Raw READY records are intentionally not a demand signal: the
        # controller can pause dependent continuations behind evaluation
        # backpressure or mark the remaining queue budget-inadmissible.  Its
        # transactional leaseable count is the only safe paid-start input.
        training_demand = min(2, leaseable_training + len(active_training_workers))
        running_training = sum(states[item.instance_id] == "RUNNING" for item in self.config.training_workers)
        for worker in self.config.training_workers:
            if running_training >= training_demand or states[worker.instance_id] != "STOPPED":
                continue
            if add("start:training", worker):
                running_training += 1

        rollout_demand = ready_evaluation > 0 or bool(active_evaluation_workers)
        if rollout_demand and states[self.config.rollout_worker.instance_id] == "STOPPED":
            add("start:evaluation", self.config.rollout_worker)

        # A worker process exiting is deliberately irrelevant here.  Only the
        # controller's current lease snapshot can make an instance idle.
        for worker in self.config.training_workers:
            if leaseable_training == 0 and worker.worker_id not in active_training_workers:
                self._idle_since_ns.setdefault(worker.instance_id, now_ns)
            else:
                self._idle_since_ns.pop(worker.instance_id, None)
        if ready_evaluation == 0 and self.config.rollout_worker.worker_id not in active_evaluation_workers:
            self._idle_since_ns.setdefault(self.config.rollout_worker.instance_id, now_ns)
        else:
            self._idle_since_ns.pop(self.config.rollout_worker.instance_id, None)

        managed_roles = [
            *((worker, "training") for worker in self.config.training_workers),
            (self.config.rollout_worker, "evaluation"),
        ]
        scheduled_instances = {selected.instance_id for _, selected in actions}
        for worker, role in managed_roles:
            if states[worker.instance_id] != "RUNNING" or worker.instance_id in scheduled_instances:
                continue
            idle_since = self._idle_since_ns.get(worker.instance_id)
            if idle_since is not None and now_ns - idle_since >= self.config.idle_seconds * 1_000_000_000:
                add("stop:" + role, worker)

        # The adapter intentionally exposes no create/delete surface.  Apply
        # only the deterministic bounded action list after all gates pass.
        for action, worker in actions:
            if action.startswith("start:"):
                self.nebius.start_instance(worker.instance_id)
            else:
                self.nebius.stop_instance(worker.instance_id)
        return self._receipt("ok", now_ns, snapshot, actions)

    @staticmethod
    def append_receipt(path: str | Path, receipt: CapacityReceipt) -> None:
        destination = Path(path)
        if not destination.is_absolute() or destination.is_symlink():
            raise ValueError("capacity receipt path is unsafe")
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if destination.exists() and (not destination.is_file() or stat.S_IMODE(destination.stat().st_mode) != 0o600):
            raise ValueError("capacity receipt log is unsafe")
        payload = json.dumps(receipt.canonical(), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii") + b"\n"
        descriptor = os.open(destination, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            with os.fdopen(descriptor, "ab", closefd=True) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            raise


def load_capacity_config(path: str | Path) -> CapacityConfig:
    source = Path(path)
    if not source.is_absolute() or source.is_symlink() or not source.is_file() or stat.S_IMODE(source.stat().st_mode) != 0o600:
        raise ValueError("capacity config is unsafe")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("capacity config is invalid") from error
    if not isinstance(value, Mapping) or set(value) != {"schema_version", "training_workers", "rollout_worker", "idle_seconds", "operation_cap", "deployment_gate_path", "deployment_gate_sha256"} or value.get("schema_version") != 1 or not isinstance(value.get("training_workers"), list) or len(value["training_workers"]) != 2 or not isinstance(value.get("rollout_worker"), Mapping):
        raise ValueError("capacity config is invalid")
    def worker(raw: object) -> ManagedWorker:
        if not isinstance(raw, Mapping) or set(raw) != {"instance_id", "worker_id"}:
            raise ValueError("capacity config is invalid")
        return ManagedWorker(raw["instance_id"], raw["worker_id"])
    try:
        return CapacityConfig(tuple(worker(item) for item in value["training_workers"]), worker(value["rollout_worker"]), value["idle_seconds"], value["operation_cap"], Path(value["deployment_gate_path"]), value["deployment_gate_sha256"])
    except (TypeError, ValueError) as error:
        raise ValueError("capacity config is invalid") from error


def load_root_owned_capacity_config(path: str | Path) -> CapacityConfig:
    """Load the operator-owned VM identities; users cannot substitute them."""
    source = Path(path)
    if source.is_symlink() or not source.is_file() or source.stat().st_uid != 0:
        raise ValueError("capacity config must be root-owned")
    config = load_capacity_config(source)
    gate = config.deployment_gate_path
    if gate.is_symlink() or not gate.is_file() or gate.stat().st_uid != 0 or stat.S_IMODE(gate.stat().st_mode) != 0o444:
        raise ValueError("capacity deployment gate must be root-owned and immutable")
    from lehome_train.groot.experiment_deployment_gate import load_deployment_gate

    deployment = load_deployment_gate(gate, config.deployment_gate_sha256)
    if (
        tuple(worker.instance_id for worker in config.training_workers) != deployment.training_instance_ids
        or config.rollout_worker.instance_id != deployment.rollout_instance_id
    ):
        raise ValueError("capacity workers do not match the immutable deployment gate")
    for worker in (*config.training_workers, config.rollout_worker):
        _require_compute_instance_id(worker.instance_id)
    return config


class HttpCapacityController:
    """Authenticated read-only controller client for the lifecycle daemon."""

    def __init__(self, url: str, token_file: str | Path, ca_file: str | Path | None) -> None:
        from lehome_train.groot.experiment_service import load_bearer_token

        if not url.startswith("https://") and not url.startswith("http://127.0.0.1:"):
            raise ValueError("capacity controller URL must use TLS or loopback")
        if url.startswith("https://") and ca_file is None:
            raise ValueError("capacity controller HTTPS requires a private CA file")
        self.tls_context: ssl.SSLContext | None = None
        if ca_file is not None:
            private_ca = Path(ca_file)
            if (
                not private_ca.is_absolute()
                or private_ca.is_symlink()
                or not private_ca.is_file()
                or stat.S_IMODE(private_ca.stat().st_mode) & 0o022
            ):
                raise ValueError("capacity controller private CA file is unsafe")
            self.tls_context = ssl.create_default_context(cafile=str(private_ca))
        self.url = url.rstrip("/")
        self.token = load_bearer_token(token_file)

    def capacity_snapshot(self) -> Mapping[str, object]:
        request = Request(self.url + "/capacity", headers={"Authorization": "Bearer " + self.token}, method="GET")
        try:
            with urlopen(request, timeout=20, context=self.tls_context) as response:
                value = json.loads(response.read())
        except (HTTPError, TimeoutError, OSError, json.JSONDecodeError) as error:
            raise ControllerUnavailable("controller capacity endpoint unavailable") from error
        if not isinstance(value, Mapping):
            raise ControllerUnavailable("controller capacity endpoint returned invalid JSON")
        return value
