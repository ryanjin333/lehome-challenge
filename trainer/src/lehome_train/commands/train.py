"""Fixed-exposure, restart-safe full training orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Callable, Literal, Protocol

from lehome_train.batch_select import has_required_headroom
from lehome_train.checkpoints import (
    AsyncCheckpointUploads,
    CheckpointDescriptor,
    can_continue_without_upload,
    require_compatible_checkpoint,
    validate_checkpoint_identity,
)
from lehome_train.io import atomic_write_json, canonical_json_sha256
from lehome_train.models import ExperimentConfig, SmokeResult
from lehome_train.schedule import (
    DEFAULT_PEAK_LEARNING_RATE,
    ExposureSchedule,
    TOTAL_SAMPLE_PRESENTATIONS,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class NonFiniteTrainingLoss(RuntimeError):
    """Training stopped before checkpointing a non-finite model state."""


@dataclass(frozen=True, slots=True)
class TrainingChunkRequest:
    """Immutable canonical schedule slice supplied to the training adapter."""

    schedule_sha256: str
    start_optimizer_step: int
    end_optimizer_step: int
    start_sample_presentations: int
    end_sample_presentations: int
    total_optimizer_steps: int
    physical_batch_size: int
    warmup_optimizer_steps: int
    warmup_fraction: float
    base_learning_rate: float
    peak_learning_rate: float
    start_learning_rate_multiplier: float
    end_learning_rate_multiplier: float
    start_learning_rate: float
    end_learning_rate: float
    resume_checkpoint: CheckpointDescriptor | None

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.schedule_sha256):
            raise ValueError("training schedule SHA-256 is invalid")
        for field_name in (
            "start_optimizer_step",
            "start_sample_presentations",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be nonnegative")
        for field_name in (
            "end_optimizer_step",
            "end_sample_presentations",
            "total_optimizer_steps",
            "physical_batch_size",
            "warmup_optimizer_steps",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.end_optimizer_step <= self.start_optimizer_step:
            raise ValueError("training chunk end step must follow its start")
        if self.end_optimizer_step > self.total_optimizer_steps:
            raise ValueError("training chunk exceeds the canonical schedule")
        if self.end_sample_presentations <= self.start_sample_presentations:
            raise ValueError("training chunk end presentations must follow its start")
        if self.start_sample_presentations != (
            self.start_optimizer_step * self.physical_batch_size
        ) or self.end_sample_presentations != (
            self.end_optimizer_step * self.physical_batch_size
        ):
            raise ValueError("training chunk presentations do not match its step range")
        if not 0 < self.warmup_optimizer_steps < self.total_optimizer_steps:
            raise ValueError("warmup optimizer steps must be within the schedule")
        for field_name in (
            "warmup_fraction",
            "base_learning_rate",
            "peak_learning_rate",
            "start_learning_rate_multiplier",
            "end_learning_rate_multiplier",
            "start_learning_rate",
            "end_learning_rate",
        ):
            value = getattr(self, field_name)
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite")
        if not 0 < self.warmup_fraction < 1:
            raise ValueError("warmup fraction must be strictly between zero and one")
        if self.base_learning_rate != 0.0 or self.peak_learning_rate <= 0:
            raise ValueError("training learning-rate bounds are invalid")
        if not 0 <= self.start_learning_rate_multiplier <= 1:
            raise ValueError("start learning-rate multiplier is invalid")
        if not 0 <= self.end_learning_rate_multiplier <= 1:
            raise ValueError("end learning-rate multiplier is invalid")
        if self.resume_checkpoint is not None and not isinstance(
            self.resume_checkpoint,
            CheckpointDescriptor,
        ):
            raise TypeError("resume checkpoint must be a CheckpointDescriptor")


@dataclass(frozen=True, slots=True)
class TrainingChunkReceipt:
    """Progress evidence returned by an injected persistent trainer."""

    schedule_sha256: str
    start_optimizer_step: int
    end_optimizer_step: int
    sample_presentations: int
    physical_batch_size: int
    finite_loss: bool

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.schedule_sha256):
            raise ValueError("training receipt schedule SHA-256 is invalid")
        if type(self.start_optimizer_step) is not int or self.start_optimizer_step < 0:
            raise ValueError("training chunk start step must be nonnegative")
        if (
            type(self.end_optimizer_step) is not int
            or self.end_optimizer_step <= self.start_optimizer_step
        ):
            raise ValueError("training chunk end step must follow its start")
        if type(self.sample_presentations) is not int or self.sample_presentations <= 0:
            raise ValueError("training chunk presentations must be positive")
        if type(self.physical_batch_size) is not int or self.physical_batch_size <= 0:
            raise ValueError("training chunk physical batch must be positive")
        if type(self.finite_loss) is not bool:
            raise ValueError("training chunk finite-loss flag must be boolean")


class TrainingRunner(Protocol):
    def __call__(self, request: TrainingChunkRequest) -> TrainingChunkReceipt: ...


class Checkpointer(Protocol):
    def __call__(
        self,
        *,
        optimizer_step: int,
        sample_presentations: int,
    ) -> CheckpointDescriptor: ...


@dataclass(frozen=True, slots=True)
class TrainingRunResult:
    """Machine-readable terminal state without any provider mutation handle."""

    status: Literal["completed", "paused_disk_reserve"]
    experiment_id: str
    experiment_config_sha256: str
    schedule_sha256: str
    physical_batch_size: int
    final_optimizer_step: int
    sample_presentations: int
    checkpoints: tuple[CheckpointDescriptor, ...]
    failed_upload_optimizer_steps: tuple[int, ...]
    provider_hourly_price: float | None
    instance_start_time: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "status": self.status,
            "experiment_id": self.experiment_id,
            "experiment_config_sha256": self.experiment_config_sha256,
            "schedule_sha256": self.schedule_sha256,
            "physical_batch_size": self.physical_batch_size,
            "final_optimizer_step": self.final_optimizer_step,
            "sample_presentations": self.sample_presentations,
            "checkpoints": [checkpoint.to_dict() for checkpoint in self.checkpoints],
            "failed_upload_optimizer_steps": list(self.failed_upload_optimizer_steps),
            "provider_hourly_price": self.provider_hourly_price,
            "instance_start_time": self.instance_start_time,
        }


def _validate_selected_smoke(
    selected_smoke: SmokeResult | None,
    *,
    experiment_config: ExperimentConfig,
    experiment_config_sha256: str,
) -> SmokeResult:
    if not isinstance(selected_smoke, SmokeResult):
        raise ValueError("a selected smoke result is required")
    if (
        not selected_smoke.stable
        or not selected_smoke.finite_loss
        or selected_smoke.failure_reason is not None
        or selected_smoke.gradient_accumulation_steps != 1
        or selected_smoke.optimizer_steps != 100
    ):
        raise ValueError("selected smoke result is not a stable fixed-contract attempt")
    if selected_smoke.physical_batch_size != experiment_config.physical_batch_size:
        raise ValueError("selected physical batch does not match experiment config")
    if selected_smoke.experiment_config_sha256 != experiment_config_sha256:
        raise ValueError("selected smoke experiment config identity is incompatible")
    if selected_smoke.dataset_manifest_sha256 != experiment_config.dataset_manifest_sha256:
        raise ValueError("selected smoke dataset manifest identity is incompatible")
    if not has_required_headroom(selected_smoke):
        raise ValueError("selected smoke result does not satisfy physical VRAM headroom")
    return selected_smoke


def _validate_chunk(
    receipt: object,
    *,
    request: TrainingChunkRequest,
) -> TrainingChunkReceipt:
    if not isinstance(receipt, TrainingChunkReceipt):
        raise TypeError("training runner must return TrainingChunkReceipt")
    if receipt.schedule_sha256 != request.schedule_sha256:
        raise ValueError("training runner receipt schedule identity is incompatible")
    if receipt.start_optimizer_step != request.start_optimizer_step:
        raise ValueError("training chunk start step is not monotonic")
    if receipt.end_optimizer_step != request.end_optimizer_step:
        raise ValueError("training chunk did not reach the checkpoint boundary")
    if receipt.physical_batch_size != request.physical_batch_size:
        raise ValueError("training chunk changed the selected physical batch")
    expected_presentations = (
        request.end_sample_presentations - request.start_sample_presentations
    )
    if receipt.sample_presentations != expected_presentations:
        raise ValueError("training chunk presentation count is incompatible")
    if not receipt.finite_loss:
        raise NonFiniteTrainingLoss("training aborted immediately after non-finite loss")
    return receipt


def _chunk_request(
    schedule: ExposureSchedule,
    *,
    start_optimizer_step: int,
    end_optimizer_step: int,
    resume_checkpoint: CheckpointDescriptor | None,
) -> TrainingChunkRequest:
    start_multiplier = schedule.learning_rate_multiplier(start_optimizer_step)
    end_multiplier = schedule.learning_rate_multiplier(end_optimizer_step)
    return TrainingChunkRequest(
        schedule_sha256=schedule.sha256,
        start_optimizer_step=start_optimizer_step,
        end_optimizer_step=end_optimizer_step,
        start_sample_presentations=(
            start_optimizer_step * schedule.physical_batch_size
        ),
        end_sample_presentations=end_optimizer_step * schedule.physical_batch_size,
        total_optimizer_steps=schedule.total_optimizer_steps,
        physical_batch_size=schedule.physical_batch_size,
        warmup_optimizer_steps=schedule.warmup_optimizer_steps,
        warmup_fraction=float(schedule.warmup_fraction),
        base_learning_rate=0.0,
        peak_learning_rate=DEFAULT_PEAK_LEARNING_RATE,
        start_learning_rate_multiplier=start_multiplier,
        end_learning_rate_multiplier=end_multiplier,
        start_learning_rate=schedule.learning_rate(start_optimizer_step),
        end_learning_rate=schedule.learning_rate(end_optimizer_step),
        resume_checkpoint=resume_checkpoint,
    )


def _replace_verified(
    checkpoints: list[CheckpointDescriptor],
    verified: tuple[CheckpointDescriptor, ...],
) -> None:
    by_step = {item.record.optimizer_step: item for item in verified}
    for index, checkpoint in enumerate(checkpoints):
        replacement = by_step.get(checkpoint.record.optimizer_step)
        if replacement is not None:
            checkpoints[index] = replacement


def _validate_provider_metadata(
    provider_hourly_price: float | None,
    instance_start_time: str | None,
) -> None:
    if provider_hourly_price is not None and (
        type(provider_hourly_price) not in (int, float)
        or not math.isfinite(float(provider_hourly_price))
        or provider_hourly_price < 0
    ):
        raise ValueError("provider hourly price must be a finite nonnegative number")
    if instance_start_time is not None and (
        type(instance_start_time) is not str or not instance_start_time.strip()
    ):
        raise ValueError("instance start time must be a non-empty string")


def run_fixed_exposure_training(
    *,
    experiment_config: ExperimentConfig,
    selected_smoke: SmokeResult | None,
    normalization_sha256: str,
    runner: TrainingRunner,
    checkpointer: Checkpointer,
    uploader: Callable[[CheckpointDescriptor], bool],
    disk_probe: Callable[[], int],
    complete_checkpoint_bytes: int,
    resume_checkpoint: CheckpointDescriptor | None = None,
    upload_sleeper: Callable[[float], None] | None = None,
    provider_hourly_price: float | None = None,
    instance_start_time: str | None = None,
    status_path: str | Path | None = None,
) -> TrainingRunResult:
    """Train exactly 768,000 presentations with safe checkpoint boundaries.

    Uploads run on a bounded single-worker queue.  This function records cost
    facts but deliberately accepts no provider deletion callback or rental ID.
    """

    if not isinstance(experiment_config, ExperimentConfig):
        raise TypeError("experiment_config must be an ExperimentConfig")
    if experiment_config.sample_presentations != TOTAL_SAMPLE_PRESENTATIONS:
        raise ValueError("full training requires exactly 768000 sample presentations")
    if experiment_config.gradient_accumulation_steps != 1:
        raise ValueError("full training requires gradient accumulation exactly 1")
    if not _SHA256.fullmatch(normalization_sha256):
        raise ValueError("normalization SHA-256 is invalid")
    _validate_provider_metadata(provider_hourly_price, instance_start_time)

    schedule = ExposureSchedule(
        physical_batch_size=experiment_config.physical_batch_size,
        sample_presentations=experiment_config.sample_presentations,
    )
    config_sha256 = canonical_json_sha256(experiment_config)
    smoke = _validate_selected_smoke(
        selected_smoke,
        experiment_config=experiment_config,
        experiment_config_sha256=config_sha256,
    )
    # Validate storage inputs before any paid training work begins.
    can_continue_without_upload(0, complete_checkpoint_bytes)

    checkpoints: list[CheckpointDescriptor] = []
    start_step = 0
    first_resume = resume_checkpoint
    if resume_checkpoint is not None:
        require_compatible_checkpoint(
            resume_checkpoint,
            experiment_id=smoke.experiment_id,
            experiment_config_sha256=config_sha256,
            dataset_manifest_sha256=experiment_config.dataset_manifest_sha256,
            normalization_sha256=normalization_sha256,
            physical_batch_size=experiment_config.physical_batch_size,
            maximum_optimizer_step=schedule.total_optimizer_steps,
            checkpoint_interval_steps=schedule.checkpoint_interval_steps,
        )
        checkpoints.append(resume_checkpoint)
        start_step = resume_checkpoint.record.optimizer_step

    status: Literal["completed", "paused_disk_reserve"] = "completed"
    observed_checkpoint_bytes = complete_checkpoint_bytes
    sleeper = upload_sleeper if upload_sleeper is not None else __import__("time").sleep
    with AsyncCheckpointUploads(uploader=uploader, sleeper=sleeper) as uploads:
        for boundary_step in schedule.checkpoint_steps:
            if boundary_step <= start_step:
                continue

            _replace_verified(checkpoints, uploads.poll())
            if uploads.failed or uploads.has_pending:
                writable_free_bytes = disk_probe()
                if not can_continue_without_upload(
                    writable_free_bytes,
                    observed_checkpoint_bytes,
                ):
                    status = "paused_disk_reserve"
                    break

            request = _chunk_request(
                schedule,
                start_optimizer_step=start_step,
                end_optimizer_step=boundary_step,
                resume_checkpoint=first_resume,
            )
            receipt = runner(request)
            first_resume = None
            _validate_chunk(
                receipt,
                request=request,
            )
            sample_presentations = boundary_step * experiment_config.physical_batch_size
            checkpoint = checkpointer(
                optimizer_step=boundary_step,
                sample_presentations=sample_presentations,
            )
            validate_checkpoint_identity(
                checkpoint,
                experiment_id=smoke.experiment_id,
                experiment_config_sha256=config_sha256,
                dataset_manifest_sha256=experiment_config.dataset_manifest_sha256,
                normalization_sha256=normalization_sha256,
                physical_batch_size=experiment_config.physical_batch_size,
                maximum_optimizer_step=schedule.total_optimizer_steps,
                checkpoint_interval_steps=schedule.checkpoint_interval_steps,
                require_remote_verification=False,
            )
            checkpoints.append(checkpoint)
            observed_checkpoint_bytes = max(
                observed_checkpoint_bytes,
                checkpoint.record.artifact.byte_size,
            )
            uploads.submit(checkpoint)
            start_step = boundary_step

        _replace_verified(checkpoints, uploads.finish())
        failed_upload_steps = tuple(
            sorted(item.record.optimizer_step for item in uploads.failed)
        )

    result = TrainingRunResult(
        status=status,
        experiment_id=smoke.experiment_id,
        experiment_config_sha256=config_sha256,
        schedule_sha256=schedule.sha256,
        physical_batch_size=experiment_config.physical_batch_size,
        final_optimizer_step=start_step,
        sample_presentations=start_step * experiment_config.physical_batch_size,
        checkpoints=tuple(checkpoints),
        failed_upload_optimizer_steps=failed_upload_steps,
        provider_hourly_price=(
            None if provider_hourly_price is None else float(provider_hourly_price)
        ),
        instance_start_time=instance_start_time,
    )
    if status_path is not None:
        atomic_write_json(status_path, result.to_dict())
    return result
