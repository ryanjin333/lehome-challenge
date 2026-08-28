"""Persistent, dependency-injected rollout worker orchestration.

This module deliberately contains no Isaac or GR00T imports.  The launcher
constructs the long-lived simulator and policy objects after Isaac's
``AppLauncher`` is active; keeping the orchestration layer pure makes its
lease/retry and identity boundaries testable on a controller-only host.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
from threading import Event, Thread, Timer
from time import monotonic, sleep
import traceback
from typing import Any, Callable, Mapping, Protocol
from uuid import uuid4

from .fidelity import (
    FIDELITY_CODES,
    ClothFidelityError,
    fidelity_receipt,
    validate_fidelity,
    validate_fidelity_diagnostic,
)


ACTION_HORIZON = 16
_SAFE_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")
_PREPARATION_TIMEOUT_REASON = "preparation_timeout"
_PREPARATION_TIMEOUT_EXIT_STATUS = 70
_SOURCE_FINALIZATION_POLL_SECONDS = 1.0
_DEFAULT_SOURCE_FINALIZATION_TIMEOUT_SECONDS = 300.0
POLICY_ACTION_SAFETY_REJECTION_REASON = "policy_action_outside_live_joint_limits"


class PreparationTimeoutError(RuntimeError):
    """Raised only when an injected test hard-exit returns to Python."""


class InfrastructureInvalidAttemptError(ValueError):
    """An episode cannot be used because runtime evidence is invalid."""


class PolicyActionSafetyRejectionError(ValueError):
    """A raw policy target exceeded the live joint-limit safety tolerance."""


class SimulatorNumericalDivergenceError(InfrastructureInvalidAttemptError):
    """Live simulator state became nonphysical and must not enter the ledger."""


class FidelityFailureError(InfrastructureInvalidAttemptError):
    """A typed pre-frame physical or safety fidelity failure."""

    def __init__(
        self, fidelity_code: str, fidelity: Mapping[str, object], *,
        diagnostic: Mapping[str, object] | None = None,
    ) -> None:
        if fidelity_code not in FIDELITY_CODES:
            raise ValueError("fidelity failure evidence is invalid")
        try:
            validated = validate_fidelity(fidelity, code=fidelity_code)
        except ValueError as error:
            raise ValueError("fidelity failure evidence is invalid") from error
        self.fidelity_code = fidelity_code
        self.fidelity = validated
        try:
            self.diagnostic = (
                validate_fidelity_diagnostic(diagnostic)
                if diagnostic is not None else None
            )
        except ValueError as error:
            raise ValueError("fidelity failure diagnostic is invalid") from error
        super().__init__(fidelity_code)


class _LeaseController(Protocol):
    def lease_next(self, worker_id: str) -> Any | None: ...

    def record_terminal(self, worker_id: str, attempt_id: str, lease_id: str, raw_artifact_id: str) -> Any: ...

    def heartbeat(self, worker_id: str, attempt_id: str, lease_id: str) -> Any: ...

    def reject_attempt(self, worker_id: str, attempt_id: str, lease_id: str, *, reason: str) -> str: ...

    def record_infrastructure_abort(self, worker_id: str, attempt_id: str, lease_id: str, *, reason: str) -> str: ...

    def status(self, attempt_id: str) -> str: ...


class _EpisodeSession(Protocol):
    runtime_receipt: Mapping[str, object]

    def prepare_episode(self, *, garment_name: str, seed: int, episode_generation: int) -> None: ...

    def run_episode(self, *, assignment: Mapping[str, object], attempt_output_dir: Path, policy: Any, cancellation_event: Event) -> Mapping[str, object]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class WorkerIdentity:
    """Stable process identity shared by every episode of one worker."""

    worker_id: str
    session_id: str
    renderer_device: str
    policy_device: str

    def __post_init__(self) -> None:
        for field in ("worker_id", "session_id", "renderer_device", "policy_device"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field):
                raise ValueError(f"{field} must be a non-empty string")
        if re.fullmatch(r"cuda:[0-9]+", self.renderer_device) is None:
            raise ValueError("renderer_device must be a canonical CUDA device")
        if self.policy_device != self.renderer_device:
            raise ValueError("persistent worker requires policy and renderer on the same physical CUDA device")


def _assignment_for(lease: Any) -> tuple[str, Mapping[str, object]]:
    attempt = getattr(lease, "attempt", None)
    attempt_id = getattr(attempt, "attempt_id", None)
    assignment = getattr(attempt, "assignment", None)
    if not isinstance(attempt_id, str) or not attempt_id:
        raise ValueError("lease must contain a non-empty immutable attempt ID")
    if not isinstance(assignment, Mapping):
        raise ValueError("lease must contain an immutable assignment mapping")
    return attempt_id, assignment


def _assignment_value(assignment: Mapping[str, object], *names: str) -> object:
    for name in names:
        if name in assignment:
            return assignment[name]
    raise ValueError(f"attempt assignment requires one of: {', '.join(names)}")


def _is_source_discovery_assignment(assignment: Mapping[str, object]) -> bool:
    """The existing source-bootstrap marker is the worker-side discovery contract."""

    return assignment.get("snapshot_source_bootstrap") is True


def _write_receipt(path: Path, receipt: Mapping[str, object]) -> None:
    """Write one small terminal receipt without ever reusing an output path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ValueError("refusing to overwrite a persistent worker receipt")
    payload = (json.dumps(dict(receipt), sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise ValueError("refusing to overwrite a persistent worker receipt") from error
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _safe_path_component(value: str, *, field: str) -> str:
    if not _SAFE_PATH_COMPONENT.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"{field} must be a path-safe identifier")
    return value


class PersistentRolloutWorker:
    """Run sequential immutable leases using exactly one simulator session.

    The controller adapter is intentionally tiny: production can use
    :class:`TaskLedger` directly, while worker-process tests inject an in-memory
    adapter.  An interrupted attempt is reported if the adapter supports it;
    the next lease remains controller-owned, so a replacement worker can retry
    the same immutable attempt identity.
    """

    def __init__(
        self,
        *,
        worker_id: str,
        session_id: str,
        controller: _LeaseController,
        simulator_factory: Callable[[], _EpisodeSession],
        policy: Any,
        output_root: Path | str,
        renderer_device: str,
        policy_device: str,
        simulator_device: str | None = None,
        heartbeat_interval_seconds: float = 30.0,
        preparation_timeout_seconds: float = 180.0,
        source_finalization_timeout_seconds: float = _DEFAULT_SOURCE_FINALIZATION_TIMEOUT_SECONDS,
        simple_curriculum_collection: bool = False,
        fidelity_diagnostic: bool = False,
        hard_exit: Callable[[int], None] = os._exit,
    ) -> None:
        self.identity = WorkerIdentity(worker_id, session_id, renderer_device, policy_device)
        self._simulator_device = renderer_device if simulator_device is None else simulator_device
        if self._simulator_device != "cpu" and self._simulator_device != renderer_device:
            raise ValueError("simulator_device must be cpu or the assigned renderer device")
        self._controller = controller
        self._simulator_factory = simulator_factory
        self._policy = policy
        self._output_root = Path(output_root)
        self._episode_generation = 0
        if type(simple_curriculum_collection) is not bool:
            raise ValueError("simple_curriculum_collection must be a boolean")
        if type(fidelity_diagnostic) is not bool:
            raise ValueError("fidelity_diagnostic must be a boolean")
        if simple_curriculum_collection and fidelity_diagnostic:
            raise ValueError("fidelity diagnostic and simple curriculum modes are mutually exclusive")
        self._simple_curriculum_collection = simple_curriculum_collection
        self._fidelity_diagnostic = fidelity_diagnostic
        if heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be positive")
        if (
            isinstance(preparation_timeout_seconds, bool)
            or not isinstance(preparation_timeout_seconds, (int, float))
            or not math.isfinite(preparation_timeout_seconds)
            or preparation_timeout_seconds <= 0
        ):
            raise ValueError("preparation_timeout_seconds must be positive")
        if (
            isinstance(source_finalization_timeout_seconds, bool)
            or not isinstance(source_finalization_timeout_seconds, (int, float))
            or not math.isfinite(source_finalization_timeout_seconds)
            or source_finalization_timeout_seconds <= 0
        ):
            raise ValueError("source_finalization_timeout_seconds must be positive")
        if not callable(hard_exit):
            raise ValueError("hard_exit must be callable")
        self._heartbeat_interval_seconds = heartbeat_interval_seconds
        self._preparation_timeout_seconds = float(preparation_timeout_seconds)
        self._source_finalization_timeout_seconds = float(source_finalization_timeout_seconds)
        self._hard_exit = hard_exit

    @property
    def episode_generation(self) -> int:
        return self._episode_generation

    def _validate_runtime_receipt(self, session: _EpisodeSession, *, require_contact: bool = True) -> Mapping[str, object]:
        raw = getattr(session, "runtime_receipt", None)
        receipt = raw() if callable(raw) else raw
        if not isinstance(receipt, Mapping):
            raise ValueError("persistent worker session must expose a runtime receipt")
        if (
            receipt.get("simulation_device") != self._simulator_device
            or receipt.get("cloth_device") != self._simulator_device
        ):
            raise ValueError("persistent rollout cloth device does not match the assigned simulator device")
        expected_backend = "usd_local_points_v1" if self._simulator_device == "cpu" else "physx_cloth_view"
        if receipt.get("cloth_backend") != expected_backend:
            expected_name = "USD-local CPU" if self._simulator_device == "cpu" else "live PhysX"
            raise ValueError(f"persistent rollout requires the {expected_name} cloth backend")
        if require_contact:
            if not isinstance(receipt.get("cloth_readback"), Mapping):
                raise ValueError("persistent rollout requires observed live cloth readback")
            contact = receipt.get("visible_contact_canary")
            if not isinstance(contact, Mapping) or not isinstance(contact.get("observed"), bool):
                raise ValueError("persistent rollout requires visible-contact canary evidence")
        if receipt.get("renderer_device") != self.identity.renderer_device:
            raise ValueError("runtime receipt renderer device does not match worker identity")
        if receipt.get("camera_device", self.identity.renderer_device) != self.identity.renderer_device:
            raise ValueError("runtime receipt camera device does not match worker identity")
        if receipt.get("policy_device") != self.identity.policy_device:
            raise ValueError("runtime receipt policy device does not match worker identity")
        return dict(receipt)

    def _action_horizon(self) -> int:
        value = getattr(self._policy, "action_horizon", ACTION_HORIZON)
        if value != ACTION_HORIZON:
            raise ValueError("persistent worker requires a local action horizon of 16")
        return ACTION_HORIZON

    def _record_interruption(self, *, attempt_id: str, lease_id: str, reason: str) -> None:
        record = getattr(self._controller, "record_interrupted", None)
        if callable(record):
            record(self.identity.worker_id, attempt_id, lease_id, reason)

    def _episode_output_dir(self, *, attempt_id: str, lease_id: str, generation: int) -> Path:
        """Bind every raw path to the worker, session, immutable lease, and generation."""

        components = (
            _safe_path_component(self.identity.worker_id, field="worker_id"),
            _safe_path_component(self.identity.session_id, field="session_id"),
            _safe_path_component(attempt_id, field="attempt_id"),
            _safe_path_component(lease_id, field="lease_id"),
            f"generation-{generation}",
        )
        root = self._output_root.absolute()
        for ancestor in (root, *root.parents):
            if ancestor.exists() and ancestor.is_symlink():
                raise ValueError("persistent worker output ancestry must not contain symlinks")
        root.mkdir(parents=True, exist_ok=True)
        resolved_root = root.resolve(strict=True)
        path = root.joinpath(*components)
        if not path.resolve(strict=False).is_relative_to(resolved_root):
            raise ValueError("persistent worker output path escapes its root")
        if path.exists() or path.is_symlink():
            raise ValueError("persistent worker output directory already exists")
        return path

    def _prepare_and_run_with_heartbeat(
        self,
        session: _EpisodeSession,
        *,
        garment_name: str,
        seed: int,
        generation: int,
        assignment: Mapping[str, object],
        output_dir: Path,
        attempt_id: str,
        lease_id: str,
    ) -> Mapping[str, object]:
        """Keep an active lease through reset and irreversibly quarantine a hung reset.

        Isaac/Omniverse preparation can block inside native USD code.  The
        watchdog therefore records a terminal rejection before calling
        ``os._exit`` in production; normal Python unwinding is not dependable
        after that boundary.  Tests inject a returning exit hook and observe
        :class:`PreparationTimeoutError` once the artificial preparation call
        is released.
        """

        stop = Event()
        failures: list[BaseException] = []
        preparation_timed_out = Event()

        def heartbeat() -> None:
            while not stop.wait(self._heartbeat_interval_seconds):
                try:
                    self._controller.heartbeat(self.identity.worker_id, attempt_id, lease_id)
                except BaseException as error:
                    failures.append(error)
                    stop.set()

        def expire_preparation() -> None:
            preparation_timed_out.set()
            try:
                abort = getattr(self._controller, "record_infrastructure_abort", None)
                if not callable(abort):
                    raise RuntimeError("controller does not support durable infrastructure abort")
                abort(
                    self.identity.worker_id,
                    attempt_id,
                    lease_id,
                    reason=_PREPARATION_TIMEOUT_REASON,
                )
                print(
                    "persistent worker: preparation timeout; aborting infrastructure-invalid attempt "
                    f"attempt_id={attempt_id} lease_id={lease_id} reason={_PREPARATION_TIMEOUT_REASON}",
                    flush=True,
                )
            except BaseException as error:
                # We still terminate the isolated worker: allowing a native
                # hang to retain this simulator process is worse than relying
                # on the controller's lease expiry/recovery path.
                print(
                    "persistent worker: preparation timeout infrastructure-abort failed: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
            self._hard_exit(_PREPARATION_TIMEOUT_EXIT_STATUS)

        # Start the lease-renewal loop before entering prepare_episode, not
        # merely after its first slow garment switch.  The first renewal keeps
        # the same cadence as episode execution and avoids extending a lease
        # before the worker has actually reached its preparation boundary.
        thread = Thread(target=heartbeat, name=f"lease-heartbeat-{self.identity.worker_id}", daemon=True)
        thread.start()
        watchdog = Timer(self._preparation_timeout_seconds, expire_preparation)
        watchdog.daemon = True
        try:
            watchdog.start()
            try:
                session.prepare_episode(
                    garment_name=garment_name,
                    seed=seed,
                    episode_generation=generation,
                    reset_policy=False,
                )
            finally:
                watchdog.cancel()
                # ``Timer.cancel`` cannot prevent a callback that has just
                # started.  Joining proves a normal preparation has no late
                # timeout thread left that could reject a completed lease.
                watchdog.join()
            if preparation_timed_out.is_set():
                raise PreparationTimeoutError(
                    f"preparation exceeded {self._preparation_timeout_seconds:g}s "
                    f"({_PREPARATION_TIMEOUT_REASON})"
                )
            scoped_assignment = dict(assignment)
            scoped_assignment["simple_curriculum_collection"] = (
                self._simple_curriculum_collection or self._fidelity_diagnostic
            )
            outcome = session.run_episode(
                assignment=scoped_assignment, attempt_output_dir=output_dir, policy=self._policy,
                cancellation_event=stop,
            )
        finally:
            stop.set()
            thread.join(timeout=self._heartbeat_interval_seconds + 1.0)
        if thread.is_alive():
            raise RuntimeError("lease heartbeat thread did not stop")
        if failures:
            raise InterruptedError("lease heartbeat failed") from failures[0]
        return outcome

    def _reset_policy_generation(self) -> int:
        reset = getattr(self._policy, "reset", None)
        if not callable(reset):
            raise ValueError("persistent worker policy must expose reset()")
        reset()
        reported = getattr(self._policy, "episode_generation", None)
        if reported is None:
            self._episode_generation += 1
            return self._episode_generation
        if not isinstance(reported, int) or reported < 1:
            raise ValueError("policy reset did not report a positive episode generation")
        if reported <= self._episode_generation:
            raise ValueError("policy episode generation did not advance on reset")
        self._episode_generation = reported
        return reported

    def _wait_for_terminal_finalization(
        self, *, attempt_id: str, allow_retryable: bool,
    ) -> str:
        """Require finalization before a gated worker can lease again."""

        deadline = monotonic() + self._source_finalization_timeout_seconds
        context = "exact partition" if allow_retryable else "source discovery"
        while True:
            status = self._controller.status(attempt_id)
            if status in {"accepted", "rejected"}:
                return status
            if status in {"retryable", "leased"} and allow_retryable:
                return status
            if status == "infrastructure_abort":
                raise RuntimeError(f"{context} finalization infrastructure abort")
            if status != "terminal_pending_validation":
                raise RuntimeError(f"{context} finalization reached unexpected state: {status!r}")
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise RuntimeError(f"{context} finalization timeout")
            sleep(min(_SOURCE_FINALIZATION_POLL_SECONDS, remaining))

    def run(self, *, max_episodes: int | None = None) -> list[dict[str, object]]:
        """Drain immediately available leases, retaining one simulator instance."""

        if max_episodes is not None and (not isinstance(max_episodes, int) or max_episodes < 0):
            raise ValueError("max_episodes must be a non-negative integer or None")
        try:
            session = self._simulator_factory()
        except BaseException as error:
            # Isaac/Kit shutdown can take minutes. Emit the actionable cause
            # before the outer launcher enters that shutdown path.
            print(
                f"persistent worker: simulator factory failed: {type(error).__name__}: {error}",
                flush=True,
            )
            raise
        print("persistent worker: simulator session ready", flush=True)
        # Device/backend identity can be checked before the first reset.
        # Visible-contact geometry is only reliable after Isaac has reset the
        # garment, so do not block the first lease on that canary.
        runtime_receipt = self._validate_runtime_receipt(session, require_contact=False)
        print("persistent worker: runtime receipt validated", flush=True)
        receipts: list[dict[str, object]] = []
        try:
            while max_episodes is None or len(receipts) < max_episodes:
                lease = self._controller.lease_next(self.identity.worker_id)
                print(f"persistent worker: lease_next -> {lease!r}", flush=True)
                if lease is None:
                    break
                attempt_id, assignment = _assignment_for(lease)
                lease_id = getattr(lease, "lease_id", None)
                if not isinstance(lease_id, str) or not lease_id:
                    raise ValueError("lease must contain a non-empty lease ID")
                garment_name = _assignment_value(assignment, "garment", "garment_name")
                seed = _assignment_value(assignment, "seed")
                if not isinstance(garment_name, str) or not garment_name:
                    raise ValueError("attempt garment must be a non-empty string")
                if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
                    raise ValueError("attempt seed must be a non-negative integer")
                environment_seed = seed
                if assignment.get("recovery_kind") == "controlled_success_recovery_snapshot_v3":
                    source_seed = assignment.get("source_seed")
                    if type(source_seed) is not int or source_seed < 0:
                        raise ValueError("controlled recovery requires a non-negative authenticated source seed")
                    # The assignment seed identifies the bounded perturbation;
                    # The source seed rebuilds the matching scene before its
                    # authenticated H16 physical snapshot is restored.
                    environment_seed = source_seed

                try:
                    generation = self._reset_policy_generation()
                    output_dir = self._episode_output_dir(
                        attempt_id=attempt_id, lease_id=lease_id, generation=generation,
                    )
                    outcome = self._prepare_and_run_with_heartbeat(
                        session,
                        garment_name=garment_name,
                        seed=environment_seed,
                        generation=generation,
                        assignment={**dict(assignment), "attempt_id": attempt_id},
                        output_dir=output_dir,
                        attempt_id=attempt_id,
                        lease_id=lease_id,
                    )
                    if not isinstance(outcome, Mapping):
                        raise ValueError("episode session must return a mapping receipt")
                    # Startup only establishes CUDA device identity.  The
                    # physical-cloth readback and visible-contact canary must
                    # be freshly observable after reset and episode execution
                    # before this lease can become terminal.
                    try:
                        runtime_receipt = self._validate_runtime_receipt(session, require_contact=True)
                    except ValueError as error:
                        raise InfrastructureInvalidAttemptError(str(error)) from error
                except BaseException as error:
                    print(f"persistent worker: episode failed: {type(error).__name__}: {error}", flush=True)
                    traceback.print_exc()
                    # Evaluation translates reset-time physical faults, but
                    # later preparation, readback, snapshot, and runtime
                    # paths can still emit the producer's typed receipt.
                    # Normalize it here before any generic error policy.
                    if isinstance(error, ClothFidelityError):
                        error = (
                            FidelityFailureError(
                                error.code, error.fidelity, diagnostic=error.diagnostic,
                            )
                            if self._simple_curriculum_collection or self._fidelity_diagnostic
                            else SimulatorNumericalDivergenceError(str(error))
                        )
                    if isinstance(error, PreparationTimeoutError):
                        # In production the watchdog has already hard-exited.
                        # This path exists solely when an injected test exit
                        # hook returns; never append a contradictory retry.
                        raise
                    if isinstance(error, PolicyActionSafetyRejectionError):
                        if self._simple_curriculum_collection or self._fidelity_diagnostic:
                            abort = getattr(self._controller, "record_fidelity_abort", None)
                            if not callable(abort):
                                raise RuntimeError("controller does not support durable fidelity abort") from error
                            abort(
                                self.identity.worker_id, attempt_id, lease_id,
                                session_id=self.identity.session_id, generation=generation,
                                fidelity_code="safety_failure",
                                fidelity=fidelity_receipt(
                                    missing_cloth=False, cloth_flight=False,
                                    nonfinite_cloth_state=False, safety_failure=True,
                                    monitor_active=True, monitor_observed=True,
                                ),
                                runtime={
                                    key: runtime_receipt[key]
                                    for key in ("simulation_device", "cloth_device", "renderer_device", "camera_device", "policy_device")
                                },
                            )
                            if _is_source_discovery_assignment(assignment):
                                raise RuntimeError("source discovery fidelity abort") from error
                            mode = "fidelity diagnostic" if self._fidelity_diagnostic else "simple curriculum campaign"
                            raise RuntimeError(f"{mode} fidelity abort") from error
                        reject = getattr(self._controller, "reject_attempt", None)
                        if not callable(reject):
                            raise RuntimeError(
                                "controller does not support durable policy safety rejection"
                            ) from error
                        reject(
                            self.identity.worker_id,
                            attempt_id,
                            lease_id,
                            reason=POLICY_ACTION_SAFETY_REJECTION_REASON,
                        )
                        continue
                    visual_replay = assignment.get("strategy") == "visual_only"
                    if isinstance(error, FidelityFailureError) and (
                        self._simple_curriculum_collection
                        or self._fidelity_diagnostic
                        or visual_replay
                    ):
                        abort = getattr(self._controller, "record_fidelity_abort", None)
                        if not callable(abort):
                            raise RuntimeError("controller does not support durable fidelity abort") from error
                        abort_kwargs = {
                            "session_id": self.identity.session_id,
                            "generation": generation,
                            "fidelity_code": error.fidelity_code,
                            "fidelity": error.fidelity,
                            "runtime": {
                                key: runtime_receipt[key]
                                for key in ("simulation_device", "cloth_device", "renderer_device", "camera_device", "policy_device")
                            },
                        }
                        if error.diagnostic is not None:
                            abort_kwargs["diagnostic"] = error.diagnostic
                        abort_status = abort(
                            self.identity.worker_id,
                            attempt_id,
                            lease_id,
                            **abort_kwargs,
                        )
                        if abort_status == "retryable":
                            raise RuntimeError(
                                "fidelity-invalid attempt was requeued; requesting clean worker restart"
                            ) from error
                        if _is_source_discovery_assignment(assignment):
                            raise RuntimeError("source discovery fidelity abort") from error
                        if visual_replay:
                            raise RuntimeError("visual replay fidelity abort") from error
                        mode = "fidelity diagnostic" if self._fidelity_diagnostic else "simple curriculum campaign"
                        raise RuntimeError(f"{mode} fidelity abort") from error
                    if isinstance(error, InfrastructureInvalidAttemptError):
                        abort = getattr(self._controller, "record_infrastructure_abort", None)
                        if not callable(abort):
                            raise RuntimeError("controller does not support durable infrastructure abort") from error
                        abort_status = abort(
                            self.identity.worker_id,
                            attempt_id,
                            lease_id,
                            reason=(
                                "simulator_numerical_divergence"
                                if isinstance(error, SimulatorNumericalDivergenceError)
                                else "runtime_evidence_invalid"
                            ),
                        )
                        if abort_status == "retryable":
                            raise RuntimeError(
                                "infrastructure-invalid attempt was requeued; requesting clean worker restart"
                            ) from error
                        if _is_source_discovery_assignment(assignment):
                            raise RuntimeError("source discovery infrastructure abort") from error
                        if self._fidelity_diagnostic:
                            raise RuntimeError("fidelity diagnostic infrastructure abort") from error
                        continue
                    restore_failed = "snapshot" in str(error).lower() or "restore" in str(error).lower()
                    if restore_failed:
                        if _is_source_discovery_assignment(assignment):
                            abort = getattr(self._controller, "record_infrastructure_abort", None)
                            if not callable(abort):
                                raise RuntimeError("controller does not support durable infrastructure abort") from error
                            abort(
                                self.identity.worker_id,
                                attempt_id,
                                lease_id,
                                reason="source_snapshot_evidence_invalid",
                            )
                            raise RuntimeError("source discovery snapshot evidence is invalid") from error
                        reject = getattr(self._controller, "reject_attempt", None)
                        if callable(reject):
                            reject(self.identity.worker_id, attempt_id, lease_id, reason=type(error).__name__)
                        else:
                            self._record_interruption(attempt_id=attempt_id, lease_id=lease_id, reason=type(error).__name__)
                        continue
                    if isinstance(error, InterruptedError):
                        # Explicit preemption/cancellation is the one normal
                        # retryable path: release once and let a replacement
                        # worker continue the immutable attempt later.
                        self._record_interruption(
                            attempt_id=attempt_id, lease_id=lease_id, reason=type(error).__name__,
                        )
                        break
                    # Ordinary deterministic errors are not retryable from
                    # this process.  Let the bounded shell supervisor decide
                    # whether to restart; do not append retryable ledger
                    # events that repeatedly select the earliest bad row.
                    raise

                receipt: dict[str, object] = {
                    "schema_version": 1,
                    "attempt_id": attempt_id,
                    "lease_id": lease_id,
                    "worker_id": self.identity.worker_id,
                    "session_id": self.identity.session_id,
                    "seed": seed,
                    **({"source_seed": environment_seed} if assignment.get("recovery_kind") == "controlled_success_recovery_snapshot_v3" else {}),
                    "garment": garment_name,
                    "episode_generation": generation,
                    "output_dir": str(output_dir),
                    "action_horizon": self._action_horizon(),
                    "simulation_device": runtime_receipt["simulation_device"],
                    "cloth_device": runtime_receipt["cloth_device"],
                    "renderer_device": runtime_receipt["renderer_device"],
                    "camera_device": runtime_receipt["camera_device"],
                    "policy_device": self.identity.policy_device,
                    "runtime": dict(runtime_receipt),
                    "outcome": dict(outcome),
                }
                _write_receipt(output_dir / "worker-receipt.json", receipt)
                self._controller.record_terminal(
                    self.identity.worker_id, attempt_id, lease_id, str(output_dir)
                )
                if (
                    _is_source_discovery_assignment(assignment)
                    or self._simple_curriculum_collection
                    or self._fidelity_diagnostic
                ):
                    finalization_status = self._wait_for_terminal_finalization(
                        attempt_id=attempt_id,
                        allow_retryable=self._simple_curriculum_collection,
                    )
                    if finalization_status in {"retryable", "leased"}:
                        continue
                receipts.append(receipt)
        finally:
            session.close()
        return receipts


__all__ = [
    "ACTION_HORIZON", "InfrastructureInvalidAttemptError",
    "POLICY_ACTION_SAFETY_REJECTION_REASON", "PolicyActionSafetyRejectionError",
    "PreparationTimeoutError",
    "PersistentRolloutWorker", "WorkerIdentity",
    "SimulatorNumericalDivergenceError",
]
