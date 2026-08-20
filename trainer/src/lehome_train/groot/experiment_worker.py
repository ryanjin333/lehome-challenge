"""Fail-closed training worker orchestration over the narrow controller API."""

from __future__ import annotations

import time
import inspect
import subprocess
import threading
from typing import Any, Callable, Mapping, Protocol

from lehome_train.groot.experiment_publication import bind_checkpoint_publication


class PreemptionRequested(RuntimeError):
    pass


class ControllerUnavailable(RuntimeError):
    pass


class HubTransportError(RuntimeError):
    pass


class CapacityUnavailable(RuntimeError):
    """A provider capacity/preemption response; safe to retry the exact lease."""
    pass


class InfrastructureFailure(ValueError):
    """A deterministic local identity or filesystem failure."""
    pass


class ControllerProtocolError(InfrastructureFailure):
    """The controller answered, but rejected a malformed worker request."""
    pass


class ManifestMismatch(InfrastructureFailure):
    pass


class LineageMismatch(InfrastructureFailure):
    pass


class UnsafePath(InfrastructureFailure):
    pass


def is_retryable_transport(error: BaseException) -> bool:
    """Keep provider/controller/Hub outages separate from bad immutable inputs."""
    if isinstance(error, (PreemptionRequested, ControllerUnavailable, HubTransportError, CapacityUnavailable)):
        return True
    chain: list[BaseException] = [error]
    while chain:
        current = chain.pop()
        name = type(current).__name__.lower()
        module = type(current).__module__.lower()
        if any(token in name for token in ("timeout", "connection", "transport", "preempt", "capacity", "ratelimit")):
            return True
        if "huggingface_hub" in module and any(token in name for token in ("http", "hub", "entrynotfound")):
            return True
        for nested in (current.__cause__, current.__context__):
            if isinstance(nested, BaseException):
                chain.append(nested)
    return False


class LeaseHeartbeatGuard:
    """Keep an active controller lease alive while a blocking task is running."""

    def __init__(self, controller: Any, lease: Any, *, interval_seconds: float = 20.0, lease_ns: int = 60_000_000_000) -> None:
        if not 0 < interval_seconds <= 20 or lease_ns != 60_000_000_000:
            raise ValueError("heartbeat guard must renew a 60-second lease every 20 seconds or less")
        self.controller, self.lease, self.interval_seconds, self.lease_ns = controller, lease, interval_seconds, lease_ns
        self.cancelled = threading.Event()
        self._stopped = threading.Event()
        self._failed: BaseException | None = None
        self._thread: threading.Thread | None = None

    def _beat(self) -> None:
        try:
            self.controller.heartbeat(self.lease, time.time_ns(), self.lease_ns)
        except BaseException as error:
            self._failed = error
            self.cancelled.set()

    def _run(self) -> None:
        while not self._stopped.wait(self.interval_seconds):
            self._beat()
            if self.cancelled.is_set():
                return

    def __enter__(self) -> "LeaseHeartbeatGuard":
        self._beat()
        if self.cancelled.is_set():
            raise ControllerUnavailable("controller heartbeat failed before task start") from self._failed
        self._thread = threading.Thread(target=self._run, name="lehome-lease-heartbeat", daemon=True)
        self._thread.start()
        return self

    def assert_owned(self) -> None:
        if self.cancelled.is_set():
            raise ControllerUnavailable("controller lease was lost during blocking task") from self._failed

    def __exit__(self, *_: object) -> None:
        self._stopped.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds + 1)


def run_with_cancellation(call: Callable[..., Any], argument: object, cancellation: threading.Event, *, parent_publication: Mapping[str, object] | None = None) -> Any:
    """Pass optional safe execution coordinates only when an adapter declares them."""
    try:
        parameters = inspect.signature(call).parameters
    except (TypeError, ValueError):
        parameters = {}
    kwargs: dict[str, object] = {}
    if "cancellation" in parameters:
        kwargs["cancellation"] = cancellation
    if parent_publication is not None and "parent_publication" in parameters:
        kwargs["parent_publication"] = parent_publication
    return call(argument, **kwargs)


def run_subprocess_cancellable(argv: list[str], *, env: Mapping[str, str], cancellation: threading.Event, poll_seconds: float = 0.2) -> None:
    """Terminate a guest controller promptly when its controller lease is lost."""
    process = subprocess.Popen(argv, env=dict(env))
    try:
        while process.poll() is None:
            if cancellation.wait(poll_seconds):
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                raise ControllerUnavailable("controller lease lost; guest process terminated")
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, argv)
    finally:
        if process.poll() is None:
            process.terminate()


class ExperimentWorker:
    def __init__(self, controller: Any, *, worker_id: str, runner: Any | None = None, idle_timeout_seconds: int = 600, heartbeat_interval_seconds: float = 20.0, identity_preflight: Callable[[object], None] | None = None) -> None:
        if not worker_id or idle_timeout_seconds < 0 or not 0 < heartbeat_interval_seconds <= 20:
            raise ValueError("worker configuration is invalid")
        self.controller, self.worker_id, self.runner, self.idle_timeout_seconds, self.heartbeat_interval_seconds, self.identity_preflight = controller, worker_id, runner, idle_timeout_seconds, heartbeat_interval_seconds, identity_preflight
        self._pending_publications: dict[str, tuple[Any, str, Mapping[str, object]]] = {}

    def _persist_pending_publication(self, lease: Any, receipt: str, publication: Mapping[str, object]) -> None:
        self._pending_publications[lease.experiment_id] = (lease, receipt, publication)
        persist = getattr(self.runner, "persist_pending_publication", None)
        if callable(persist):
            persist(lease, receipt, publication)

    def _clear_pending_publication(self, experiment_id: str) -> None:
        self._pending_publications.pop(experiment_id, None)
        clear = getattr(self.runner, "clear_pending_publication", None)
        if callable(clear):
            clear(experiment_id)

    def _load_pending_publications(self) -> tuple[tuple[Any, str, Mapping[str, object]], ...]:
        pending = dict(self._pending_publications)
        reader = getattr(self.runner, "pending_publications", None)
        if callable(reader):
            for item in reader():
                if not isinstance(item, tuple) or len(item) != 3:
                    raise ManifestMismatch("pending publication handoff is malformed")
                lease, receipt, publication = item
                if type(receipt) is not str or not isinstance(publication, Mapping):
                    raise ManifestMismatch("pending publication handoff is malformed")
                pending[getattr(lease, "experiment_id", "")] = (lease, receipt, publication)
        return tuple(pending.values())

    def _settle_terminal_receipt(self, lease: Any, receipt: str) -> object:
        """Use the controller's idempotent receipt transition when available."""
        reconcile = getattr(self.controller, "reconcile_terminal_receipt", None)
        if callable(reconcile):
            return reconcile(lease, receipt, time.time_ns())
        # Narrow compatibility for direct unit-test adapters.  Production HTTP
        # clients implement the canonical reconciliation method below.
        return self.controller.complete(lease, receipt, time.time_ns())

    def _reconcile_pending_publications(self) -> bool:
        """Settle durable terminal receipts before publishing or leasing work.

        A response-lost completion may already have deleted the live lease, so
        publication is never retried until the exact immutable receipt has
        reconciled.  On any transport ambiguity we leave the handoff durable
        and return ``False``; the caller must not lease and rerun GPU work.
        """
        for lease, receipt, publication in self._load_pending_publications():
            if not getattr(lease, "experiment_id", None):
                raise ManifestMismatch("pending publication has no experiment identity")
            if not all(type(getattr(lease, field, None)) is str and getattr(lease, field) for field in ("lease_id", "worker_id")):
                # A legacy handoff can prove bytes, but without the original
                # lease identity it cannot prove that a never-arrived complete
                # request still belongs to this worker.  Leave it durable and
                # do not lease a GPU retry.
                return False
            bind_checkpoint_publication(lease.job, receipt, publication)
            try:
                state = self._settle_terminal_receipt(lease, receipt)
                # A response can be lost after the controller has both verified
                # this exact publication and completed evaluation.  The durable
                # handoff then proves recovery, not authority to regress the
                # job.  Only PUBLISHING still requires a publication mutation.
                if state is None or state == "PUBLISHING":
                    self.controller.publication_verified(lease.experiment_id, dict(publication), time.time_ns())
            except BaseException as error:
                if is_retryable_transport(error) or isinstance(error, ControllerProtocolError):
                    return False
                raise ManifestMismatch("terminal receipt reconciliation failed") from error
            self._clear_pending_publication(lease.experiment_id)
        return True

    def run(self, *, max_jobs: int | None = None) -> int:
        completed = 0
        idle_since = time.monotonic()
        while max_jobs is None or completed < max_jobs:
            if not self._reconcile_pending_publications():
                return completed
            lease = self.controller.lease_next(self.worker_id, "training", now_ns=time.time_ns(), lease_ns=60_000_000_000)
            if lease is None:
                if time.monotonic() - idle_since >= self.idle_timeout_seconds:
                    return completed
                time.sleep(min(0.1, max(0.0, self.idle_timeout_seconds)))
                continue
            idle_since = time.monotonic()
            publication_pending = False
            try:
                if self.runner is None:
                    raise ManifestMismatch("no production runner configured")
                if self.identity_preflight is not None:
                    # This executes before the first heartbeat or guest launch:
                    # a mismatched immutable trainer cannot consume GPU time.
                    self.identity_preflight(lease.job)
                with LeaseHeartbeatGuard(self.controller, lease, interval_seconds=self.heartbeat_interval_seconds) as heartbeat:
                    result = run_with_cancellation(self.runner.run, lease.job, heartbeat.cancelled, parent_publication=getattr(lease, "parent_publication", None))
                    heartbeat.assert_owned()
                if not isinstance(result, dict) or set(result) != {"terminal_receipt_sha256", "publication"}:
                    raise ManifestMismatch("runner must return terminal receipt and publication")
                receipt, publication = result["terminal_receipt_sha256"], result["publication"]
                if type(receipt) is not str or not isinstance(publication, dict):
                    raise ManifestMismatch("runner result is malformed")
                bind_checkpoint_publication(lease.job, receipt, publication)
                self._persist_pending_publication(lease, receipt, publication)
                publication_pending = True
                heartbeat.assert_owned()
                state = self._settle_terminal_receipt(lease, receipt)
                if state is None or state == "PUBLISHING":
                    self.controller.publication_verified(lease.experiment_id, publication, time.time_ns())
                self._clear_pending_publication(lease.experiment_id)
                completed += 1
            except BaseException as error:
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                # Once an immutable terminal receipt is persisted, never turn a
                # post-complete transport outage into a second training run.
                if publication_pending:
                    if is_retryable_transport(error) or isinstance(error, ControllerProtocolError):
                        return completed
                    if isinstance(error, ValueError):
                        raise ManifestMismatch("terminal receipt reconciliation failed") from error
                if is_retryable_transport(error):
                    self.controller.retryable(lease, type(error).__name__, time.time_ns())
                else:
                    self.controller.block_infrastructure(lease, type(error).__name__, time.time_ns())
        return completed
