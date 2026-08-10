"""Sequential fixed-contract physical-batch smoke-test orchestration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from threading import Event, Thread
from typing import Callable, Literal, Protocol

from lehome_train.batch_select import (
    NoStableBatchError,
    batch_candidates,
    fallback_candidates,
    has_required_headroom,
    select_largest_stable_batch,
)
from lehome_train.groot.config import FineTuneLaunchConfig
from lehome_train.models import SmokeResult, StrictModel
from lehome_train.telemetry import (
    SampledOperationError,
    TelemetrySample,
    TelemetrySampler,
    TelemetrySummary,
    sample_operation,
    summarize_failure_telemetry,
    summarize_telemetry,
)


SMOKE_OPTIMIZER_STEPS = 100
FailureReason = Literal["cuda_oom", "non_finite_loss"]


class SmokeRunnerFailure(RuntimeError):
    """A typed runner failure that is safe to normalize into an attempt."""

    def __init__(
        self,
        *,
        failure_reason: Literal["cuda_oom"],
        optimizer_steps: int = 0,
        initialization_seconds: float = 0.0,
        warmup_seconds: float = 0.0,
        telemetry_samples: tuple[TelemetrySample, ...] = (),
    ) -> None:
        super().__init__(failure_reason)
        if failure_reason != "cuda_oom":
            raise ValueError("raised smoke runner failures must be proven CUDA OOM")
        self.failure_reason = failure_reason
        self.optimizer_steps = optimizer_steps
        self.initialization_seconds = initialization_seconds
        self.warmup_seconds = warmup_seconds
        self.telemetry_samples = telemetry_samples

    @classmethod
    def cuda_oom(
        cls,
        *,
        optimizer_steps: int = 0,
        initialization_seconds: float = 0.0,
        warmup_seconds: float = 0.0,
        telemetry_samples: tuple[TelemetrySample, ...] = (),
    ) -> "SmokeRunnerFailure":
        return cls(
            failure_reason="cuda_oom",
            optimizer_steps=optimizer_steps,
            initialization_seconds=initialization_seconds,
            warmup_seconds=warmup_seconds,
            telemetry_samples=telemetry_samples,
        )

    def to_receipt(
        self,
        *,
        sampled_telemetry: tuple[TelemetrySample, ...] = (),
    ) -> "SmokeAttemptReceipt":
        return SmokeAttemptReceipt(
            optimizer_steps=self.optimizer_steps,
            gradient_accumulation_steps=1,
            finite_loss=True,
            initialization_seconds=self.initialization_seconds,
            warmup_seconds=self.warmup_seconds,
            steady_state_seconds=0.0,
            steady_state_optimizer_steps=0,
            telemetry_samples=self.telemetry_samples or sampled_telemetry,
            failure_reason=self.failure_reason,
        )


@dataclass(frozen=True, slots=True)
class SmokeAttemptReceipt:
    """Runner evidence used to validate one official training process."""

    optimizer_steps: int
    gradient_accumulation_steps: int
    finite_loss: bool
    initialization_seconds: float
    warmup_seconds: float
    steady_state_seconds: float
    steady_state_optimizer_steps: int
    telemetry_samples: tuple[TelemetrySample, ...]
    failure_reason: FailureReason | None = None
    per_device_telemetry_samples: tuple[tuple[TelemetrySample, ...], ...] = ()

    def __post_init__(self) -> None:
        if type(self.optimizer_steps) is not int or self.optimizer_steps < 0:
            raise ValueError("smoke optimizer steps must be nonnegative")
        if type(self.gradient_accumulation_steps) is not int or self.gradient_accumulation_steps <= 0:
            raise ValueError("smoke gradient accumulation must be positive")
        if type(self.finite_loss) is not bool:
            raise ValueError("smoke finite loss must be boolean")
        if type(self.steady_state_optimizer_steps) is not int or self.steady_state_optimizer_steps < 0:
            raise ValueError("steady-state optimizer steps must be nonnegative")
        if self.steady_state_optimizer_steps > self.optimizer_steps:
            raise ValueError("steady-state optimizer steps exceed total progress")
        if not isinstance(self.telemetry_samples, tuple) or not all(
            isinstance(sample, TelemetrySample) for sample in self.telemetry_samples
        ):
            raise ValueError("smoke telemetry samples must be a tuple of TelemetrySample")
        if not isinstance(self.per_device_telemetry_samples, tuple) or not all(
            isinstance(samples, tuple)
            and samples
            and all(isinstance(sample, TelemetrySample) for sample in samples)
            for samples in self.per_device_telemetry_samples
        ):
            raise ValueError("per-device smoke telemetry must contain non-empty sample tuples")
        if self.failure_reason not in (None, "cuda_oom", "non_finite_loss"):
            raise ValueError("smoke failure reason is unsupported")
        if (not self.finite_loss) != (self.failure_reason == "non_finite_loss"):
            raise ValueError("non-finite loss and smoke failure reason must agree")


@dataclass(frozen=True, slots=True)
class SmokeAttempt(StrictModel):
    """Complete machine-readable record for one sequential launch."""

    attempt_experiment_id: str
    result: SmokeResult
    telemetry: TelemetrySummary
    completed_optimizer_steps: int
    per_device_telemetry: tuple[TelemetrySummary, ...] = ()

    def __post_init__(self) -> None:
        StrictModel.__post_init__(self)
        if not self.attempt_experiment_id:
            raise ValueError("smoke attempt experiment identity must be non-empty")
        if self.completed_optimizer_steps < 0:
            raise ValueError("completed smoke optimizer steps must be nonnegative")
        if self.completed_optimizer_steps > self.result.optimizer_steps:
            raise ValueError("completed smoke optimizer steps exceed configured budget")


@dataclass(frozen=True, slots=True)
class SmokeReport(StrictModel):
    """All attempts and the largest selectable physical batch, if any."""

    selected_batch_size: int | None
    attempts: tuple[SmokeAttempt, ...]

    def __post_init__(self) -> None:
        StrictModel.__post_init__(self)
        if self.selected_batch_size is not None and self.selected_batch_size <= 0:
            raise ValueError("selected smoke batch must be positive")
        attempted_batches = tuple(
            attempt.result.physical_batch_size for attempt in self.attempts
        )
        if len(attempted_batches) != len(set(attempted_batches)):
            raise ValueError("smoke batches must not be attempted more than once")
        if self.selected_batch_size is not None and self.selected_batch_size not in attempted_batches:
            raise ValueError("selected smoke batch was not attempted")


class SmokeRunner(Protocol):
    """Injected synchronous adapter around the pinned official launcher."""

    def __call__(self, config: FineTuneLaunchConfig) -> SmokeAttemptReceipt: ...


SamplerFactory = Callable[[], TelemetrySampler]


class MultiDeviceTelemetrySampler(Protocol):
    def sample_all(self) -> tuple[TelemetrySample, ...]: ...

    def close(self) -> None: ...


def _attempt_config(base: FineTuneLaunchConfig, batch_size: int) -> FineTuneLaunchConfig:
    """Change only batch identity and the fixed smoke step budget."""

    if base.num_gpus == 4:
        if batch_size != 1:
            raise ValueError("four-GPU smoke profile requires per-device batch 1")
        return replace(
            base,
            experiment_name=f"{base.experiment_name}-distributed-smoke",
            max_steps=SMOKE_OPTIMIZER_STEPS,
            save_steps=SMOKE_OPTIMIZER_STEPS,
        )
    return replace(
        base,
        experiment_name=f"{base.experiment_name}-batch-{batch_size}",
        physical_batch_size=batch_size,
        global_batch_size=batch_size,
        gradient_accumulation_steps=1,
        max_steps=SMOKE_OPTIMIZER_STEPS,
    )


def _run_with_optional_sampler(
    runner: SmokeRunner,
    config: FineTuneLaunchConfig,
    sampler_factory: SamplerFactory | None,
) -> SmokeAttemptReceipt:
    if sampler_factory is None:
        try:
            receipt = runner(config)
        except SmokeRunnerFailure as failure:
            receipt = failure.to_receipt()
        if not isinstance(receipt, SmokeAttemptReceipt):
            raise TypeError("smoke runner must return SmokeAttemptReceipt")
        return receipt

    sampler = sampler_factory()
    if not hasattr(sampler, "sample"):
        raise TypeError("smoke sampler factory must return a TelemetrySampler")
    if hasattr(sampler, "sample_all"):
        return _run_with_multi_device_sampler(runner, config, sampler)
    try:
        try:
            receipt, boundary_samples = sample_operation(
                lambda: runner(config),
                sampler=sampler,
            )
        except SampledOperationError as sampled_error:
            if not isinstance(sampled_error.original_error, SmokeRunnerFailure):
                raise sampled_error.original_error from sampled_error
            receipt = sampled_error.original_error.to_receipt(
                sampled_telemetry=sampled_error.samples,
            )
            boundary_samples = sampled_error.samples
    finally:
        close = getattr(sampler, "close", None)
        if callable(close):
            close()
    if not isinstance(receipt, SmokeAttemptReceipt):
        raise TypeError("smoke runner must return SmokeAttemptReceipt")
    if receipt.telemetry_samples:
        return receipt
    return replace(receipt, telemetry_samples=boundary_samples)


def _run_with_multi_device_sampler(
    runner: SmokeRunner,
    config: FineTuneLaunchConfig,
    sampler: MultiDeviceTelemetrySampler,
) -> SmokeAttemptReceipt:
    """Capture aligned evidence for every distributed device during the run."""

    first = sampler.sample_all()
    samples = [[sample] for sample in first]
    stopped = Event()
    errors: list[BaseException] = []

    def poll() -> None:
        while not stopped.wait(0.1):
            try:
                observed = sampler.sample_all()
                if len(observed) != len(samples):
                    raise ValueError("distributed telemetry device count drifted")
                for destination, sample in zip(samples, observed):
                    destination.append(sample)
            except BaseException as error:
                errors.append(error)
                stopped.set()

    polling = Thread(target=poll, name="lehome-distributed-smoke-telemetry", daemon=True)
    polling.start()
    try:
        receipt = runner(config)
    except SmokeRunnerFailure as failure:
        receipt = failure.to_receipt()
    finally:
        stopped.set()
        polling.join()
        try:
            final = sampler.sample_all()
            if len(final) != len(samples):
                raise ValueError("distributed telemetry device count drifted")
            for destination, sample in zip(samples, final):
                destination.append(sample)
        except BaseException as error:
            errors.append(error)
        sampler.close()
    if errors:
        raise RuntimeError("distributed telemetry sampling failed during smoke launch") from errors[0]
    if not isinstance(receipt, SmokeAttemptReceipt):
        raise TypeError("smoke runner must return SmokeAttemptReceipt")
    evidence = tuple(tuple(device_samples) for device_samples in samples)
    return replace(
        receipt,
        telemetry_samples=receipt.telemetry_samples or evidence[0],
        per_device_telemetry_samples=evidence,
    )


def _validated_attempt(
    *,
    config: FineTuneLaunchConfig,
    receipt: SmokeAttemptReceipt,
    physical_vram_bytes: int,
    experiment_config_sha256: str,
    dataset_manifest_sha256: str,
    experiment_id: str,
) -> SmokeAttempt:
    if receipt.gradient_accumulation_steps != 1:
        raise ValueError("smoke gradient accumulation must remain exactly 1")
    if receipt.optimizer_steps != SMOKE_OPTIMIZER_STEPS and receipt.failure_reason is None:
        raise ValueError("successful smoke attempts must run exactly 100 optimizer steps")
    if not receipt.telemetry_samples:
        raise ValueError("every smoke attempt requires telemetry samples")
    if config.num_gpus == 4 and len(receipt.per_device_telemetry_samples) != 4:
        raise ValueError("four-GPU smoke requires telemetry evidence from every device")
    if receipt.failure_reason is None:
        telemetry = summarize_telemetry(
            receipt.telemetry_samples,
            initialization_seconds=receipt.initialization_seconds,
            warmup_seconds=receipt.warmup_seconds,
            steady_state_seconds=receipt.steady_state_seconds,
            steady_state_optimizer_steps=receipt.steady_state_optimizer_steps,
            physical_batch_size=config.global_batch_size,
        )
    else:
        telemetry = summarize_failure_telemetry(
            receipt.telemetry_samples,
            initialization_seconds=receipt.initialization_seconds,
            warmup_seconds=receipt.warmup_seconds,
        )
    if telemetry.physical_total_vram_bytes != physical_vram_bytes:
        raise ValueError("NVML total does not match expected physical VRAM")
    per_device_telemetry: tuple[TelemetrySummary, ...] = ()
    if receipt.per_device_telemetry_samples:
        summarizer = (
            summarize_failure_telemetry
            if receipt.failure_reason is not None
            else summarize_telemetry
        )
        summaries: list[TelemetrySummary] = []
        for samples in receipt.per_device_telemetry_samples:
            if receipt.failure_reason is None:
                summaries.append(
                    summarizer(
                        samples,
                        initialization_seconds=receipt.initialization_seconds,
                        warmup_seconds=receipt.warmup_seconds,
                        steady_state_seconds=receipt.steady_state_seconds,
                        steady_state_optimizer_steps=receipt.steady_state_optimizer_steps,
                        physical_batch_size=config.global_batch_size,
                    )
                )
            else:
                summaries.append(
                    summarizer(
                        samples,
                        initialization_seconds=receipt.initialization_seconds,
                        warmup_seconds=receipt.warmup_seconds,
                    )
                )
        per_device_telemetry = tuple(summaries)
        if min(item.physical_total_vram_bytes for item in per_device_telemetry) != physical_vram_bytes:
            raise ValueError("per-device NVML totals do not match the limiting visible GPU")
    distributed_headroom = all(
        item.minimum_steady_state_free_vram_bytes is not None
        and item.minimum_steady_state_free_vram_bytes * 100
        >= item.physical_total_vram_bytes * 10
        for item in per_device_telemetry
    )
    stable = (
        receipt.optimizer_steps == SMOKE_OPTIMIZER_STEPS
        and receipt.gradient_accumulation_steps == 1
        and receipt.finite_loss
        and receipt.failure_reason is None
        and (config.num_gpus != 4 or distributed_headroom)
    )
    result = SmokeResult(
        experiment_id=experiment_id,
        experiment_config_sha256=experiment_config_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
        physical_batch_size=config.global_batch_size,
        gradient_accumulation_steps=receipt.gradient_accumulation_steps,
        optimizer_steps=SMOKE_OPTIMIZER_STEPS,
        stable=stable,
        finite_loss=receipt.finite_loss,
        physical_vram_bytes=telemetry.physical_total_vram_bytes,
        peak_reserved_vram_bytes=telemetry.peak_reserved_vram_bytes,
        minimum_steady_state_free_vram_bytes=(
            telemetry.minimum_steady_state_free_vram_bytes
        ),
        steady_steps_per_second=telemetry.steady_steps_per_second,
        samples_per_second=telemetry.samples_per_second,
        failure_reason=receipt.failure_reason,
    )
    return SmokeAttempt(
        attempt_experiment_id=config.experiment_name,
        result=result,
        telemetry=telemetry,
        completed_optimizer_steps=receipt.optimizer_steps,
        per_device_telemetry=per_device_telemetry,
    )


def run_smoke_tests(
    *,
    base_config: FineTuneLaunchConfig,
    physical_vram_bytes: int,
    experiment_config_sha256: str,
    dataset_manifest_sha256: str,
    runner: SmokeRunner,
    sampler_factory: SamplerFactory | None = None,
) -> SmokeReport:
    """Run all approved candidates synchronously and stop at memory boundaries."""

    primary = (1,) if base_config.num_gpus == 4 else batch_candidates(physical_vram_bytes)
    first_candidate = primary[0]
    pending = list(primary)
    fallback_mode = False
    attempts: list[SmokeAttempt] = []

    while pending:
        batch_size = pending.pop(0)
        config = _attempt_config(base_config, batch_size)
        receipt = _run_with_optional_sampler(runner, config, sampler_factory)
        attempt = _validated_attempt(
            config=config,
            receipt=receipt,
            physical_vram_bytes=physical_vram_bytes,
            experiment_config_sha256=experiment_config_sha256,
            dataset_manifest_sha256=dataset_manifest_sha256,
            experiment_id=base_config.experiment_name,
        )
        attempts.append(attempt)
        memory_failure = attempt.result.failure_reason == "cuda_oom"
        headroom_failure = (
            attempt.result.stable and not has_required_headroom(attempt.result)
        )
        memory_boundary = memory_failure or headroom_failure

        if fallback_mode:
            if attempt.result.stable and not memory_boundary:
                break
            continue
        if memory_boundary:
            if batch_size == first_candidate:
                pending = list(fallback_candidates(first_candidate))
                fallback_mode = True
                continue
            break

    try:
        selected = select_largest_stable_batch(
            attempt.result for attempt in attempts
        )
    except NoStableBatchError:
        selected = None
    return SmokeReport(selected_batch_size=selected, attempts=tuple(attempts))
