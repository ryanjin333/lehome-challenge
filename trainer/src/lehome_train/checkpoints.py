"""Checkpoint identity, retention, disk reserve, and asynchronous uploads."""

from __future__ import annotations

from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, replace
import json
import math
import re
from pathlib import Path
from queue import Queue
from threading import Thread
from time import monotonic, sleep
from typing import Callable, Iterable, Mapping, Protocol

from lehome_train.io import atomic_write_json
from lehome_train.models import CheckpointRecord, model_from_mapping


GIBIBYTE = 1024**3
MINIMUM_UNALLOCATED_DISK_BYTES = 20 * GIBIBYTE
MAX_UPLOAD_ATTEMPTS = 5
_INITIAL_RETRY_DELAY_SECONDS = 0.25
_MAX_RETRY_DELAY_SECONDS = 1.0
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHECKPOINT_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class CheckpointDescriptor:
    """A resumable record plus exact normalization-processor identity."""

    record: CheckpointRecord
    normalization_sha256: str
    schedule_sha256: str
    locally_verified: bool

    def __post_init__(self) -> None:
        if not isinstance(self.record, CheckpointRecord):
            raise TypeError("checkpoint descriptor requires a CheckpointRecord")
        if (
            type(self.normalization_sha256) is not str
            or not _SHA256.fullmatch(self.normalization_sha256)
        ):
            raise ValueError("normalization SHA-256 is invalid")
        if type(self.schedule_sha256) is not str or not _SHA256.fullmatch(
            self.schedule_sha256
        ):
            raise ValueError("schedule SHA-256 is invalid")
        if self.record.schedule_sha256 != self.schedule_sha256:
            raise ValueError("checkpoint record and descriptor schedule identities differ")
        if type(self.locally_verified) is not bool:
            raise ValueError("locally verified flag must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "normalization_sha256": self.normalization_sha256,
            "schedule_sha256": self.schedule_sha256,
            "locally_verified": self.locally_verified,
            "record": self.record.to_dict(),
        }


def write_checkpoint_descriptor(
    destination: str | Path,
    checkpoint: CheckpointDescriptor,
) -> None:
    """Atomically record all compatibility-critical checkpoint identity."""

    if not isinstance(checkpoint, CheckpointDescriptor):
        raise TypeError("checkpoint must be a CheckpointDescriptor")
    atomic_write_json(destination, checkpoint.to_dict())


def load_checkpoint_descriptor(source: str | Path) -> CheckpointDescriptor:
    """Load a strict checkpoint descriptor, rejecting schema drift."""

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("checkpoint descriptor is malformed")
            result[key] = value
        return result

    try:
        decoded = json.loads(
            Path(source).read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("checkpoint descriptor is malformed") from error
    if not isinstance(decoded, dict) or set(decoded) != {
        "schema_version",
        "normalization_sha256",
        "schedule_sha256",
        "locally_verified",
        "record",
    }:
        raise ValueError("checkpoint descriptor is malformed")
    if decoded["schema_version"] != _CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint descriptor has an unsupported schema")
    if not isinstance(decoded["record"], Mapping):
        raise ValueError("checkpoint descriptor record is malformed")
    return CheckpointDescriptor(
        record=model_from_mapping(CheckpointRecord, decoded["record"]),
        normalization_sha256=decoded["normalization_sha256"],
        schedule_sha256=decoded["schedule_sha256"],
        locally_verified=decoded["locally_verified"],
    )


def validate_checkpoint_identity(
    checkpoint: CheckpointDescriptor,
    *,
    experiment_id: str,
    experiment_config_sha256: str,
    dataset_manifest_sha256: str,
    normalization_sha256: str,
    schedule_sha256: str,
    physical_batch_size: int,
    maximum_optimizer_step: int,
    checkpoint_interval_steps: int,
    require_remote_verification: bool,
) -> CheckpointDescriptor:
    """Fail closed on any incompatible or impossible checkpoint provenance."""

    if not isinstance(checkpoint, CheckpointDescriptor):
        raise TypeError("checkpoint must be a CheckpointDescriptor")
    record = checkpoint.record
    if record.experiment_id != experiment_id:
        raise ValueError("checkpoint experiment identity is incompatible")
    if record.experiment_config_sha256 != experiment_config_sha256:
        raise ValueError("checkpoint experiment config identity is incompatible")
    if record.dataset_manifest_sha256 != dataset_manifest_sha256:
        raise ValueError("checkpoint dataset manifest identity is incompatible")
    if checkpoint.normalization_sha256 != normalization_sha256:
        raise ValueError("checkpoint normalization identity is incompatible")
    if checkpoint.schedule_sha256 != schedule_sha256:
        raise ValueError("checkpoint schedule identity is incompatible")
    if type(physical_batch_size) is not int or physical_batch_size <= 0:
        raise ValueError("physical batch size must be a positive integer")
    if type(maximum_optimizer_step) is not int or maximum_optimizer_step <= 0:
        raise ValueError("maximum optimizer step must be a positive integer")
    if type(checkpoint_interval_steps) is not int or checkpoint_interval_steps <= 0:
        raise ValueError("checkpoint interval must be a positive integer")
    if not record.resumable:
        raise ValueError("checkpoint is not resumable")
    if not checkpoint.locally_verified:
        raise ValueError("checkpoint is not locally verified")
    if require_remote_verification and not record.remotely_verified:
        raise ValueError("checkpoint is not remotely verified")
    if record.optimizer_step > maximum_optimizer_step:
        raise ValueError("checkpoint exceeds the training schedule")
    if record.optimizer_step % checkpoint_interval_steps:
        raise ValueError("checkpoint is not on a checkpoint boundary")
    if record.sample_presentations != record.optimizer_step * physical_batch_size:
        raise ValueError("checkpoint sample presentations are incompatible")
    return checkpoint


def require_compatible_checkpoint(
    checkpoint: CheckpointDescriptor,
    *,
    experiment_id: str,
    experiment_config_sha256: str,
    dataset_manifest_sha256: str,
    normalization_sha256: str,
    schedule_sha256: str,
    physical_batch_size: int,
    maximum_optimizer_step: int,
    checkpoint_interval_steps: int,
) -> CheckpointDescriptor:
    """Require a remotely verified checkpoint safe for restart."""

    return validate_checkpoint_identity(
        checkpoint,
        experiment_id=experiment_id,
        experiment_config_sha256=experiment_config_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
        normalization_sha256=normalization_sha256,
        schedule_sha256=schedule_sha256,
        physical_batch_size=physical_batch_size,
        maximum_optimizer_step=maximum_optimizer_step,
        checkpoint_interval_steps=checkpoint_interval_steps,
        require_remote_verification=True,
    )


def latest_verified_resumable(
    checkpoints: Iterable[CheckpointDescriptor],
) -> CheckpointDescriptor | None:
    verified = tuple(
        checkpoint
        for checkpoint in checkpoints
        if checkpoint.record.resumable and checkpoint.record.remotely_verified
    )
    return max(verified, key=lambda item: item.record.optimizer_step, default=None)


def prunable_checkpoints(
    checkpoints: Iterable[CheckpointDescriptor],
    *,
    keep_newest: int,
) -> tuple[CheckpointDescriptor, ...]:
    """Select old checkpoints while always protecting the newest verified resume."""

    if type(keep_newest) is not int or keep_newest < 0:
        raise ValueError("keep_newest must be a nonnegative integer")
    observed = tuple(checkpoints)
    if not all(isinstance(item, CheckpointDescriptor) for item in observed):
        raise TypeError("all checkpoints must be CheckpointDescriptor values")
    if len({item.record.optimizer_step for item in observed}) != len(observed):
        raise ValueError("checkpoint optimizer steps must be unique")
    ordered = sorted(observed, key=lambda item: item.record.optimizer_step)
    protected = set(ordered[-keep_newest:] if keep_newest else ())
    verified = latest_verified_resumable(ordered)
    if verified is not None:
        protected.add(verified)
    return tuple(
        item
        for item in ordered
        if item not in protected
        and item.locally_verified
        and item.record.remotely_verified
    )


def can_continue_without_upload(
    writable_free_bytes: int,
    complete_checkpoint_bytes: int,
) -> bool:
    """Require room for two additional full checkpoints plus twenty GiB."""

    if type(writable_free_bytes) is not int or writable_free_bytes < 0:
        raise ValueError("writable free bytes must be a nonnegative integer")
    if type(complete_checkpoint_bytes) is not int or complete_checkpoint_bytes <= 0:
        raise ValueError("complete checkpoint bytes must be a positive integer")
    return writable_free_bytes >= (
        2 * complete_checkpoint_bytes + MINIMUM_UNALLOCATED_DISK_BYTES
    )


def can_begin_checkpoint_chunk(
    writable_free_bytes: int,
    estimated_checkpoint_bytes: int,
) -> bool:
    """Reserve the imminent checkpoint, two more complete copies, and 20 GiB."""

    if type(writable_free_bytes) is not int or writable_free_bytes < 0:
        raise ValueError("writable free bytes must be a nonnegative integer")
    if type(estimated_checkpoint_bytes) is not int or estimated_checkpoint_bytes <= 0:
        raise ValueError("estimated checkpoint bytes must be a positive integer")
    return writable_free_bytes >= (
        3 * estimated_checkpoint_bytes + MINIMUM_UNALLOCATED_DISK_BYTES
    )


class CheckpointUploader(Protocol):
    """Transport that must enforce the supplied timeout on each attempt."""

    def __call__(
        self,
        checkpoint: CheckpointDescriptor,
        *,
        timeout_seconds: float,
    ) -> bool: ...


def _retry_delays(max_attempts: int) -> tuple[float, ...]:
    return tuple(
        min(
            _INITIAL_RETRY_DELAY_SECONDS * (2**attempt),
            _MAX_RETRY_DELAY_SECONDS,
        )
        for attempt in range(max_attempts - 1)
    )


def _upload_deadline_budget(
    *,
    max_attempts: int,
    attempt_timeout_seconds: float,
) -> float:
    return max_attempts * attempt_timeout_seconds + sum(_retry_delays(max_attempts))


def retry_checkpoint_upload(
    checkpoint: CheckpointDescriptor,
    *,
    uploader: CheckpointUploader,
    max_attempts: int = MAX_UPLOAD_ATTEMPTS,
    attempt_timeout_seconds: float = 30.0,
    sleeper: Callable[[float], None] = sleep,
) -> bool:
    """Attempt one verified upload at most five times with bounded backoff."""

    if not isinstance(checkpoint, CheckpointDescriptor):
        raise TypeError("checkpoint must be a CheckpointDescriptor")
    if type(max_attempts) is not int or not 1 <= max_attempts <= MAX_UPLOAD_ATTEMPTS:
        raise ValueError("max_attempts must be between one and five")
    if (
        type(attempt_timeout_seconds) not in (int, float)
        or not math.isfinite(float(attempt_timeout_seconds))
        or attempt_timeout_seconds <= 0
    ):
        raise ValueError("attempt timeout seconds must be finite and positive")
    delays = _retry_delays(max_attempts)
    for attempt in range(max_attempts):
        try:
            uploaded = uploader(
                checkpoint,
                timeout_seconds=float(attempt_timeout_seconds),
            )
        except Exception:
            uploaded = False
        if type(uploaded) is not bool:
            raise TypeError("checkpoint uploader must return a boolean")
        if uploaded:
            return True
        if attempt + 1 < max_attempts:
            sleeper(delays[attempt])
    return False


class AsyncCheckpointUploads:
    """Single-worker upload queue that never owns checkpoint deletion."""

    def __init__(
        self,
        *,
        uploader: CheckpointUploader,
        attempt_timeout_seconds: float = 30.0,
        max_attempts: int = MAX_UPLOAD_ATTEMPTS,
        sleeper: Callable[[float], None] = sleep,
    ) -> None:
        if type(max_attempts) is not int or not 1 <= max_attempts <= MAX_UPLOAD_ATTEMPTS:
            raise ValueError("max_attempts must be between one and five")
        if (
            type(attempt_timeout_seconds) not in (int, float)
            or not math.isfinite(float(attempt_timeout_seconds))
            or attempt_timeout_seconds <= 0
        ):
            raise ValueError("attempt timeout seconds must be finite and positive")
        self._uploader = uploader
        self._attempt_timeout_seconds = float(attempt_timeout_seconds)
        self._max_attempts = max_attempts
        self._deadline_budget = _upload_deadline_budget(
            max_attempts=max_attempts,
            attempt_timeout_seconds=attempt_timeout_seconds,
        )
        if not math.isfinite(self._deadline_budget) or self._deadline_budget <= 0:
            raise ValueError("upload deadline budget must be finite and positive")
        self._sleeper = sleeper
        self._pending: dict[Future[bool], CheckpointDescriptor] = {}
        self._completed: list[CheckpointDescriptor] = []
        self._failed: list[CheckpointDescriptor] = []
        self._submitted_steps: set[int] = set()
        self._closed = False
        self._deadline_expired = False
        self._work_queue: Queue[
            tuple[Future[bool], CheckpointDescriptor] | None
        ] = Queue()
        self._worker = Thread(
            target=self._run_worker,
            name="checkpoint-upload",
            daemon=True,
        )
        self._worker.start()

    def _run_worker(self) -> None:
        while True:
            work = self._work_queue.get()
            try:
                if work is None:
                    return
                future, checkpoint = work
                if not future.set_running_or_notify_cancel():
                    continue
                try:
                    uploaded = retry_checkpoint_upload(
                        checkpoint,
                        uploader=self._uploader,
                        max_attempts=self._max_attempts,
                        attempt_timeout_seconds=self._attempt_timeout_seconds,
                        sleeper=self._sleeper,
                    )
                except BaseException as error:
                    future.set_exception(error)
                else:
                    future.set_result(uploaded)
            finally:
                self._work_queue.task_done()

    @property
    def failed(self) -> tuple[CheckpointDescriptor, ...]:
        return tuple(self._failed)

    @property
    def has_pending(self) -> bool:
        return bool(self._pending)

    def submit(self, checkpoint: CheckpointDescriptor) -> None:
        if self._closed:
            raise RuntimeError("checkpoint upload queue is closed")
        step = checkpoint.record.optimizer_step
        if step in self._submitted_steps:
            raise ValueError("checkpoint upload was already submitted")
        self._submitted_steps.add(step)
        future: Future[bool] = Future()
        self._pending[future] = checkpoint
        self._work_queue.put((future, checkpoint))

    def poll(self) -> tuple[CheckpointDescriptor, ...]:
        """Harvest completed work without waiting for active uploads."""

        newly_completed: list[CheckpointDescriptor] = []
        for future, checkpoint in tuple(self._pending.items()):
            if not future.done():
                continue
            del self._pending[future]
            if future.result():
                verified = replace(
                    checkpoint,
                    record=replace(checkpoint.record, remotely_verified=True),
                )
                self._completed.append(verified)
                newly_completed.append(verified)
            else:
                self._failed.append(checkpoint)
        return tuple(newly_completed)

    def finish(self) -> tuple[CheckpointDescriptor, ...]:
        """Wait for bounded workers and return every verified descriptor."""

        pending = tuple(self._pending)
        deadline = monotonic() + self._deadline_budget * len(pending)
        for future in pending:
            remaining = deadline - monotonic()
            try:
                future.result(timeout=max(0.0, remaining))
            except FutureTimeoutError:
                self._deadline_expired = True
                for pending_future, checkpoint in tuple(self._pending.items()):
                    pending_future.cancel()
                    self._failed.append(checkpoint)
                    del self._pending[pending_future]
                break
        self.poll()
        return tuple(self._completed)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.finish()
        finally:
            self._closed = True
            self._work_queue.put(None)
            if not self._deadline_expired:
                self._worker.join()

    def __enter__(self) -> "AsyncCheckpointUploads":
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()
