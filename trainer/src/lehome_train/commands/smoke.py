"""Sequential fixed-contract physical-batch smoke-test orchestration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Protocol

from lehome_train.batch_select import (
    batch_candidates,
    fallback_candidates,
    has_required_free_vram,
    has_required_headroom,
)
from lehome_train.groot.config import FineTuneLaunchConfig
from lehome_train.models import SmokeResult, StrictModel
from lehome_train.telemetry import (
    TelemetrySample,
    TelemetrySampler,
    TelemetrySummary,
    sample_operation,
    summarize_telemetry,
)


SMOKE_OPTIMIZER_STEPS = 100


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
    error_code: str | None = None
    memory_failure: bool = False

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
        if self.error_code is not None and (
            type(self.error_code) is not str or not self.error_code.strip()
        ):
            raise ValueError("smoke error code must be non-empty when present")
        if type(self.memory_failure) is not bool:
            raise ValueError("smoke memory failure flag must be boolean")
        if self.memory_failure and self.error_code is None:
            raise ValueError("proven memory failure requires an error code")


@dataclass(frozen=True, slots=True)
class SmokeAttempt(StrictModel):
    """Complete machine-readable record for one sequential launch."""

    result: SmokeResult
    telemetry: TelemetrySummary
    completed_optimizer_steps: int
    memory_failure: bool

    def __post_init__(self) -> None:
        StrictModel.__post_init__(self)
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


def _attempt_config(base: FineTuneLaunchConfig, batch_size: int) -> FineTuneLaunchConfig:
    """Change only batch identity and the fixed smoke step budget."""

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
        receipt = runner(config)
        if not isinstance(receipt, SmokeAttemptReceipt):
            raise TypeError("smoke runner must return SmokeAttemptReceipt")
        return receipt

    sampler = sampler_factory()
    if not hasattr(sampler, "sample"):
        raise TypeError("smoke sampler factory must return a TelemetrySampler")
    try:
        receipt, boundary_samples = sample_operation(
            lambda: runner(config),
            sampler=sampler,
        )
    finally:
        close = getattr(sampler, "close", None)
        if callable(close):
            close()
    if not isinstance(receipt, SmokeAttemptReceipt):
        raise TypeError("smoke runner must return SmokeAttemptReceipt")
    if receipt.telemetry_samples:
        return receipt
    return replace(receipt, telemetry_samples=boundary_samples)


def _validated_attempt(
    *,
    config: FineTuneLaunchConfig,
    receipt: SmokeAttemptReceipt,
    physical_vram_bytes: int,
    experiment_config_sha256: str,
    dataset_manifest_sha256: str,
) -> SmokeAttempt:
    if receipt.gradient_accumulation_steps != 1:
        raise ValueError("smoke gradient accumulation must remain exactly 1")
    if receipt.optimizer_steps != SMOKE_OPTIMIZER_STEPS and not receipt.memory_failure:
        raise ValueError("successful smoke attempts must run exactly 100 optimizer steps")
    if not receipt.telemetry_samples:
        raise ValueError("every smoke attempt requires telemetry samples")
    telemetry = summarize_telemetry(
        receipt.telemetry_samples,
        initialization_seconds=receipt.initialization_seconds,
        warmup_seconds=receipt.warmup_seconds,
        steady_state_seconds=receipt.steady_state_seconds,
        steady_state_optimizer_steps=receipt.steady_state_optimizer_steps,
        physical_batch_size=config.physical_batch_size,
    )
    error_code = receipt.error_code
    if not receipt.finite_loss and error_code is None:
        error_code = "non_finite_loss"
    stable = (
        receipt.optimizer_steps == SMOKE_OPTIMIZER_STEPS
        and receipt.gradient_accumulation_steps == 1
        and receipt.finite_loss
        and error_code is None
        and not receipt.memory_failure
    )
    result = SmokeResult(
        experiment_id=config.experiment_name,
        experiment_config_sha256=experiment_config_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
        physical_batch_size=config.physical_batch_size,
        gradient_accumulation_steps=receipt.gradient_accumulation_steps,
        optimizer_steps=SMOKE_OPTIMIZER_STEPS,
        stable=stable,
        finite_loss=receipt.finite_loss,
        physical_vram_bytes=physical_vram_bytes,
        peak_reserved_vram_bytes=telemetry.peak_reserved_vram_bytes,
        steady_steps_per_second=telemetry.steady_steps_per_second,
        samples_per_second=telemetry.samples_per_second,
        error_code=error_code,
    )
    return SmokeAttempt(
        result=result,
        telemetry=telemetry,
        completed_optimizer_steps=receipt.optimizer_steps,
        memory_failure=receipt.memory_failure,
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

    primary = batch_candidates(physical_vram_bytes)
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
        )
        attempts.append(attempt)
        headroom_failure = not has_required_free_vram(
            physical_vram_bytes,
            attempt.telemetry.minimum_free_vram_bytes,
        )
        memory_boundary = attempt.memory_failure or headroom_failure

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
        selected = max(
            attempt.result.physical_batch_size
            for attempt in attempts
            if attempt.result.stable
            and attempt.result.finite_loss
            and has_required_headroom(attempt.result)
            and has_required_free_vram(
                physical_vram_bytes,
                attempt.telemetry.minimum_free_vram_bytes,
            )
        )
    except ValueError:
        selected = None
    return SmokeReport(selected_batch_size=selected, attempts=tuple(attempts))
