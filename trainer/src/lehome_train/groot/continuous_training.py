"""Completion-gated immutable local snapshots for continuous GR00T training."""
from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import Future
from queue import Queue
import os
import shutil
from pathlib import Path
import tempfile
from time import time
import threading
from typing import Callable, Mapping

from lehome_train.groot.production_adapters import _verified_checkpoint_state_at
from lehome_train.groot.local_recovery import (
    AttestationCancelled,
    HF_RECOVERY_STEPS,
    LOCAL_CHECKPOINT_STEPS,
    attest_local_checkpoint,
    record_immutable_publication,
    validate_immutable_publication_admission,
    validate_local_recovery_admission,
)
from lehome_train.io import sha256_file


@dataclass(frozen=True, slots=True)
class CompletedCheckpoint:
    optimizer_step: int
    source_sha256: str
    snapshot_root: Path
    observed_at_unix: int


class ProviderInterrupted(RuntimeError):
    """A verified provider-side interruption that may resume from publication."""


class SnapshotCancelled(RuntimeError):
    """A preemption requested while copying a checkpoint publication snapshot."""


class _PublicationCancelled:
    """Internal no-op result for daemon work stopped before Hub side effects."""


class PreemptionController:
    """Signal-safe state only; tests invoke ``request`` without touching OS signals."""

    def __init__(self) -> None:
        self.requested = False
        self.signal_number: int | None = None
        self.finalized_step: int | None = None

    def request(self, signal_number: int | None = None) -> None:
        self.requested = True
        self.signal_number = signal_number

    def handler(self, signal_number: int, _frame: object) -> None:
        self.request(signal_number)

    def status(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "signal_number": self.signal_number,
            "finalized_step": self.finalized_step,
        }


class _DaemonSerialPublisher:
    """One serial publisher whose blocked transport cannot delay process exit.

    Hub upload calls are intentionally run by a daemon thread: after a provider
    preemption the supervisor must return within its deadline even if an
    in-flight HTTP call cannot be interrupted by Python.  ``Future`` retains
    the normal-path result/error contract, while cancellation prevents queued
    boundaries from starting after preemption.
    """

    def __init__(self, *, cancel_requested: Callable[[], bool] | None = None) -> None:
        self._queue: Queue[tuple[Future[object], Callable[[], object]] | None] = Queue()
        self._futures: list[Future[object]] = []
        self._cancel_requested = cancel_requested
        self.thread = threading.Thread(
            target=self._run, name="checkpoint-publisher", daemon=True,
        )
        self.thread.start()

    def submit(self, work: Callable[[], object]) -> Future[object]:
        future: Future[object] = Future()
        self._futures.append(future)
        self._queue.put((future, work))
        return future

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            future, work = item
            if not future.set_running_or_notify_cancel():
                continue
            if self._cancel_requested is not None and self._cancel_requested():
                future.set_result(_PublicationCancelled())
                continue
            try:
                future.set_result(work())
            except BaseException as error:
                future.set_exception(error)

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        if cancel_futures:
            for future in self._futures:
                future.cancel()
        self._queue.put(None)
        if wait:
            self.thread.join()


def _has_unresolved_submitted_publication(
    submitted: Mapping[int, object], resolved: Mapping[int, object],
) -> bool:
    """Keep the observer alive until every submitted boundary was consumed."""
    return any(future is not None and step not in resolved for step, future in submitted.items())


def _raise_if_cancelled(cancel_requested: Callable[[], bool] | None) -> None:
    if cancel_requested is not None and cancel_requested():
        raise SnapshotCancelled("checkpoint snapshot cancelled")


def _tree_sha256(root: Path, *, cancel_requested: Callable[[], bool] | None = None) -> str:
    import hashlib
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        _raise_if_cancelled(cancel_requested)
        if path.is_symlink() or not path.is_file():
            if path.is_symlink():
                raise ValueError("complete checkpoint contains a symlink")
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        if cancel_requested is None:
            digest.update(sha256_file(path).encode())
        else:
            file_digest = hashlib.sha256()
            with path.open("rb") as stream:
                while True:
                    _raise_if_cancelled(cancel_requested)
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    file_digest.update(chunk)
            digest.update(file_digest.hexdigest().encode())
    return digest.hexdigest()


def snapshot_checkpoint(
    checkpoint: str | Path, *, optimizer_step: int,
    cancel_requested: Callable[[], bool] | None = None,
) -> CompletedCheckpoint:
    """Verify an official completion marker then take a no-hardlink copy."""
    source = Path(checkpoint)
    _raise_if_cancelled(cancel_requested)
    try:
        _verified_checkpoint_state_at(source, optimizer_step)
    except ValueError as error:
        raise ValueError("complete checkpoint is required before packaging") from error
    destination = source.parent / f".{source.name}.snapshot-{optimizer_step}"
    if destination.exists():
        # A resumed observer may encounter its prior immutable copy.  Accept it
        # only when it remains a completed checkpoint of the same boundary.
        _raise_if_cancelled(cancel_requested)
        _verified_checkpoint_state_at(destination, optimizer_step)
        return CompletedCheckpoint(optimizer_step, _tree_sha256(destination, cancel_requested=cancel_requested), destination, int(time()))
    temporary = Path(tempfile.mkdtemp(
        prefix=f"{destination.name}.", suffix=".incomplete", dir=source.parent,
    ))

    def copy_file(source_path: str, destination_path: str, *, follow_symlinks: bool = True) -> str:
        _raise_if_cancelled(cancel_requested)
        with open(source_path, "rb") as read_stream, open(destination_path, "wb") as write_stream:
            while True:
                _raise_if_cancelled(cancel_requested)
                chunk = read_stream.read(1024 * 1024)
                if not chunk:
                    break
                write_stream.write(chunk)
            write_stream.flush()
            os.fsync(write_stream.fileno())
        shutil.copystat(source_path, destination_path, follow_symlinks=follow_symlinks)
        return destination_path

    try:
        shutil.copytree(source, temporary, dirs_exist_ok=True, symlinks=False, copy_function=copy_file)
        _raise_if_cancelled(cancel_requested)
        _verified_checkpoint_state_at(temporary, optimizer_step)
        digest = _tree_sha256(temporary, cancel_requested=cancel_requested)
        _raise_if_cancelled(cancel_requested)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return CompletedCheckpoint(optimizer_step, digest, destination, int(time()))


def run_continuous_supervisor(
    *,
    run_root: Path,
    launch: Callable[[], object],
    package: Callable[[CompletedCheckpoint], object],
    publish: Callable[[object], bool],
    wait: Callable[[], None] | None = None,
    already_published: tuple[int, ...] = (),
    local_recovery_root: Path | None = None,
    local_identity: Mapping[str, object] | None = None,
    preemption: PreemptionController | None = None,
    initial_immutable_publication: Mapping[str, object] | None = None,
    initial_immutable_anchor: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], ...]:
    """Launch once, observe official save completion, publish in one worker.

    The launch thread never receives publisher credentials.  Publication uses
    an immutable snapshot and a serial daemon publisher; callers may use a
    blocking wait hook in production or a deterministic no-op hook in tests.
    """
    if (local_recovery_root is None) != (local_identity is None):
        raise ValueError("continuous supervisor local recovery root and identity must be paired")
    if local_recovery_root is not None and local_identity is not None:
        validate_local_recovery_admission(
            metadata_root=local_recovery_root, identity=local_identity,
        )
    if already_published not in ((), (1000,)):
        raise ValueError("continuous supervisor already-published recovery admission is invalid")
    if already_published == ():
        if initial_immutable_publication is not None or initial_immutable_anchor is not None:
            raise ValueError("continuous supervisor immutable admission requires already-published 1000")
    else:
        if not isinstance(initial_immutable_publication, Mapping) or not isinstance(initial_immutable_anchor, Mapping):
            raise ValueError("continuous supervisor immutable admission requires the exact 1000 publication and anchor")
        validate_immutable_publication_admission(
            publication=initial_immutable_publication,
            anchor=initial_immutable_anchor,
            optimizer_step=1000,
        )

    launch_error: list[BaseException] = []
    finished = threading.Event()

    def train() -> None:
        try:
            if preemption is not None and preemption.requested:
                return
            launch()
        except BaseException as error:
            launch_error.append(error)
        finally:
            finished.set()

    training_thread = threading.Thread(target=train, daemon=preemption is not None)
    training_thread.start()
    submitted: dict[int, object] = {step: None for step in already_published}
    local_attested: set[int] = set()
    local_checkpoints: dict[int, object] = {}
    last_publication: Mapping[str, object] | None = initial_immutable_publication
    last_anchor: Mapping[str, object] | None = initial_immutable_anchor
    resolved: dict[int, object] = {}

    def resolve_publication(step: int, future: object) -> object | None:
        nonlocal last_publication, last_anchor
        if step in resolved:
            return resolved[step]
        done = getattr(future, "done", None)
        if not callable(done) or not done():
            return None
        result = future.result()  # type: ignore[union-attr]
        resolved[step] = result
        if (
            step == 1000 and isinstance(result, Mapping)
            and result.get("optimizer_step") == 1000
            and result.get("readback_verified") is True
        ):
            anchor = result.get("runtime_checkpoint_anchor")
            if (
                isinstance(anchor, Mapping)
                and type(anchor.get("immutable_anchor_revision")) is str
                and type(anchor.get("anchor_sha256")) is str
            ):
                last_publication = {
                    key: value for key, value in result.items()
                    if key != "runtime_checkpoint_anchor"
                }
                last_anchor = {
                    "immutable_anchor_revision": anchor["immutable_anchor_revision"],
                    "anchor_sha256": anchor["anchor_sha256"],
                }
        return result

    def attest_available_local_boundaries() -> None:
        """Attest every complete trainer boundary independently of Hub timing."""
        if local_recovery_root is None or local_identity is None:
            return
        for step in LOCAL_CHECKPOINT_STEPS:
            if preemption is not None and preemption.requested:
                raise AttestationCancelled("local checkpoint attestation cancelled")
            if step in local_attested:
                continue
            checkpoint = run_root / f"checkpoint-{step}"
            if not checkpoint.is_dir():
                continue
            try:
                _verified_checkpoint_state_at(checkpoint, step)
            except ValueError:
                continue
            local_checkpoints[step] = attest_local_checkpoint(
                checkpoint=checkpoint, metadata_root=local_recovery_root,
                optimizer_step=step, identity=local_identity,
                cancel_requested=None if preemption is None else lambda: preemption.requested,
            )
            local_attested.add(step)

    def finalize_preemption() -> tuple[dict[str, object], ...]:
        """Return completed publications without starting any further tree work."""
        assert preemption is not None and preemption.requested
        preemption.finalized_step = max(local_attested) if local_attested else None
        immutable: list[dict[str, object]] = []
        for step in HF_RECOVERY_STEPS:
            future = submitted.get(step)
            if future is None:
                continue
            receipt = resolve_publication(step, future)
            if receipt is True:
                immutable.append({"optimizer_step": step, "readback_verified": True})
            elif isinstance(receipt, dict) and receipt.get("readback_verified") is True and receipt.get("optimizer_step") == step:
                immutable.append(receipt)
                # SIGTERM forbids fresh tree work.  The main loop attests
                # completed local boundaries before observing the flag;
                # journal only that already-verified receipt here.
                local_checkpoint = local_checkpoints.get(step)
                candidate_anchor = receipt.get("runtime_checkpoint_anchor")
                if local_recovery_root is not None and local_checkpoint is not None and isinstance(candidate_anchor, Mapping):
                    record_immutable_publication(
                        metadata_root=local_recovery_root,
                        checkpoint=local_checkpoint,  # type: ignore[arg-type]
                        publication={key: value for key, value in receipt.items() if key != "runtime_checkpoint_anchor"},
                        anchor={
                            "immutable_anchor_revision": candidate_anchor.get("immutable_anchor_revision"),
                            "anchor_sha256": candidate_anchor.get("anchor_sha256"),
                        },
                    )
        for future in submitted.values():
            if future is not None:
                future.cancel()  # type: ignore[union-attr]
        return tuple(immutable)

    finished_polls = 0
    def publication_cancelled() -> bool:
        return preemption is not None and preemption.requested

    publisher = _DaemonSerialPublisher(cancel_requested=publication_cancelled)
    try:
        while not finished.is_set() or len(submitted) < len(HF_RECOVERY_STEPS):
            for published_step, future in submitted.items():
                if future is not None:
                    resolve_publication(published_step, future)
            if preemption is not None and preemption.requested:
                preemption.finalized_step = max(local_attested) if local_attested else None
                break
            try:
                attest_available_local_boundaries()
            except AttestationCancelled:
                if preemption is None or not preemption.requested:
                    raise
                preemption.finalized_step = max(local_attested) if local_attested else None
                break
            for step in HF_RECOVERY_STEPS:
                if step in submitted:
                    continue
                # 2K is meaningful only after the immutable 1K artifact and
                # its anchor have both read back successfully.  A resume may
                # supply that already-authenticated chain as the initial state.
                if step == 2000 and (last_publication is None or last_anchor is None):
                    continue
                checkpoint = run_root / f"checkpoint-{step}"
                if checkpoint.is_dir():
                    try:
                        snapshot = snapshot_checkpoint(
                            checkpoint, optimizer_step=step,
                            cancel_requested=None if preemption is None else lambda: preemption.requested,
                        )
                    except SnapshotCancelled:
                        if preemption is None or not preemption.requested:
                            raise
                        preemption.finalized_step = max(local_attested) if local_attested else None
                        break
                    except ValueError:
                        # The upstream save directory becomes visible before
                        # trainer_state.json is atomically complete.  Observe
                        # again; never package an incomplete boundary.
                        continue
                    def publish_snapshot(item: CompletedCheckpoint = snapshot) -> object:
                        # The daemon observes the same gate before dequeuing,
                        # and this second pair of checks closes the race after
                        # dequeue but before packaging or Hub side effects.
                        if publication_cancelled():
                            return _PublicationCancelled()
                        packaged = package(item)
                        if publication_cancelled():
                            return _PublicationCancelled()
                        return publish(packaged)

                    submitted[step] = publisher.submit(publish_snapshot)
            if preemption is not None and preemption.requested:
                preemption.finalized_step = max(local_attested) if local_attested else None
                break
            # Preserve already published checkpoints on interruption; the
            # terminal caller converts this into an authenticated resume state.
            if launch_error and finished.is_set():
                break
            if wait is None:
                if finished.is_set():
                    finished_polls += 1
                    pending_publication = _has_unresolved_submitted_publication(submitted, resolved)
                    # A complete trainer can expose 2K before the 1K Hub
                    # readback returns.  Keep observing that pending future:
                    # it is the only safe event that unlocks 2K submission.
                    if len(submitted) == len(HF_RECOVERY_STEPS) or (
                        finished_polls >= 20 and not pending_publication
                    ):
                        break
                threading.Event().wait(0.1)
            else:
                wait()
        if preemption is not None and preemption.requested:
            return finalize_preemption()
        immutable: list[dict[str, object]] = []
        for step in HF_RECOVERY_STEPS:
            if step not in submitted:
                continue
            future = submitted[step]
            if future is None:
                continue
            receipt = resolve_publication(step, future)
            while receipt is None:
                if preemption is not None and preemption.requested:
                    return finalize_preemption()
                # Do not call Future.result() here: a stuck Hub call must be
                # interruptible by SIGTERM even during normal collection.
                threading.Event().wait(0.05)
                receipt = resolve_publication(step, future)
            assert receipt is not None
            if preemption is not None and preemption.requested:
                return finalize_preemption()
            try:
                attest_available_local_boundaries()
            except AttestationCancelled:
                if preemption is None or not preemption.requested:
                    raise
                return finalize_preemption()
            if receipt is True:
                immutable.append({"optimizer_step": step, "readback_verified": True})
            elif isinstance(receipt, dict) and receipt.get("readback_verified") is True and receipt.get("optimizer_step") == step:
                immutable.append(receipt)
                if step in HF_RECOVERY_STEPS and local_recovery_root is not None:
                    local_checkpoint = local_checkpoints.get(step)
                    candidate_anchor = receipt.get("runtime_checkpoint_anchor")
                    if local_checkpoint is not None and isinstance(candidate_anchor, Mapping):
                        record_immutable_publication(
                            metadata_root=local_recovery_root,
                            checkpoint=local_checkpoint,  # type: ignore[arg-type]
                            publication={key: value for key, value in receipt.items() if key != "runtime_checkpoint_anchor"},
                            anchor={
                                "immutable_anchor_revision": candidate_anchor.get("immutable_anchor_revision"),
                                "anchor_sha256": candidate_anchor.get("anchor_sha256"),
                            },
                        )
                if step == 1000:
                    candidate_anchor = receipt.get("runtime_checkpoint_anchor")
                    if isinstance(candidate_anchor, Mapping):
                        last_publication = receipt
                        last_anchor = candidate_anchor
    finally:
        preempted = preemption is not None and preemption.requested
        publisher.shutdown(wait=not preempted, cancel_futures=preempted)
    if preemption is not None and preemption.requested:
        return tuple(immutable)
    if launch_error:
        training_thread.join()
        error = launch_error[0]
        # Only an external interruption preserves a resumable terminal.  A
        # trainer/config/data error is evidence against continuation and must
        # remain a hard failure rather than being mistaken for a provider stop.
        if not isinstance(error, (KeyboardInterrupt, ProviderInterrupted)):
            raise error
        return tuple(immutable)
    training_thread.join()
    return tuple(immutable)
