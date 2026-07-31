from __future__ import annotations

from time import sleep

import pytest

from lehome_train.batch_select import GIBIBYTE
from lehome_train.commands.smoke import (
    SMOKE_OPTIMIZER_STEPS,
    SmokeAttemptReceipt,
    run_smoke_tests,
)
from lehome_train.constants import MODEL_REVISION
from lehome_train.groot.config import FineTuneLaunchConfig
from lehome_train.telemetry import TelemetrySample, sample_operation, summarize_telemetry


DATASET_REVISION = "d" * 40
SHA_A = "a" * 64
SHA_B = "b" * 64


def _config(**overrides: object) -> FineTuneLaunchConfig:
    values: dict[str, object] = {
        "base_model_path": "/models/groot",
        "base_model_revision": MODEL_REVISION,
        "dataset_path": "/data/prepared",
        "dataset_revision": DATASET_REVISION,
        "modality_config_path": "/config/modality.py",
        "output_dir": "/output",
        "experiment_name": "smoke",
        "physical_batch_size": 1,
        "max_steps": 1,
        "save_steps": 100,
        "warmup_ratio": 0.05,
    }
    values.update(overrides)
    return FineTuneLaunchConfig(**values)


def _samples(
    total_gib: int,
    reserved_gib: float,
    *,
    free_gib: float | None = None,
) -> tuple[TelemetrySample, ...]:
    observed_free_gib = total_gib - reserved_gib if free_gib is None else free_gib
    return (
        TelemetrySample(0.0, 4 * GIBIBYTE, 5 * GIBIBYTE, (total_gib - 5) * GIBIBYTE, 20.0, 100.0, 45.0, 2 * GIBIBYTE),
        TelemetrySample(
            10.0,
            int((reserved_gib - 1) * GIBIBYTE),
            int(reserved_gib * GIBIBYTE),
            int(observed_free_gib * GIBIBYTE),
            90.0,
            350.0,
            75.0,
            3 * GIBIBYTE,
        ),
    )


def _receipt(
    batch: int,
    *,
    total_gib: int = 96,
    reserved_gib: float = 70,
    optimizer_steps: int = 100,
    accumulation: int = 1,
    finite_loss: bool = True,
    error_code: str | None = None,
    memory_failure: bool = False,
    free_gib: float | None = None,
) -> SmokeAttemptReceipt:
    return SmokeAttemptReceipt(
        optimizer_steps=optimizer_steps,
        gradient_accumulation_steps=accumulation,
        finite_loss=finite_loss,
        initialization_seconds=3.0,
        warmup_seconds=2.0,
        steady_state_seconds=10.0,
        steady_state_optimizer_steps=min(80, optimizer_steps),
        telemetry_samples=_samples(total_gib, reserved_gib, free_gib=free_gib),
        error_code=error_code,
        memory_failure=memory_failure,
    )


def test_smoke_launches_tier_candidates_sequentially_with_fixed_contract() -> None:
    calls: list[FineTuneLaunchConfig] = []

    def runner(config: FineTuneLaunchConfig) -> SmokeAttemptReceipt:
        assert calls == [] or calls[-1].physical_batch_size < config.physical_batch_size
        calls.append(config)
        return _receipt(config.physical_batch_size)

    report = run_smoke_tests(
        base_config=_config(),
        physical_vram_bytes=96 * GIBIBYTE,
        experiment_config_sha256=SHA_A,
        dataset_manifest_sha256=SHA_B,
        runner=runner,
    )

    assert [config.physical_batch_size for config in calls] == [16, 32, 64]
    assert all(config.max_steps == SMOKE_OPTIMIZER_STEPS == 100 for config in calls)
    assert all(config.global_batch_size == config.physical_batch_size for config in calls)
    assert all(config.gradient_accumulation_steps == 1 for config in calls)
    assert all(config.action_horizon == 16 for config in calls)
    assert report.selected_batch_size == 64
    assert [attempt.result.physical_batch_size for attempt in report.attempts] == [16, 32, 64]


def test_memory_failure_stops_larger_batches_and_preserves_complete_attempt_record() -> None:
    calls: list[int] = []

    def runner(config: FineTuneLaunchConfig) -> SmokeAttemptReceipt:
        calls.append(config.physical_batch_size)
        if config.physical_batch_size == 32:
            return _receipt(
                32,
                reserved_gib=95,
                optimizer_steps=17,
                error_code="cuda_oom",
                memory_failure=True,
            )
        return _receipt(config.physical_batch_size, reserved_gib=70)

    report = run_smoke_tests(
        base_config=_config(),
        physical_vram_bytes=96 * GIBIBYTE,
        experiment_config_sha256=SHA_A,
        dataset_manifest_sha256=SHA_B,
        runner=runner,
    )

    assert calls == [16, 32]
    assert report.selected_batch_size == 16
    assert len(report.attempts) == 2
    failed = report.attempts[-1]
    assert failed.result.stable is False
    assert failed.result.error_code == "cuda_oom"
    assert failed.completed_optimizer_steps == 17
    assert failed.telemetry.peak_allocated_vram_bytes > 0
    assert failed.telemetry.peak_reserved_vram_bytes == 95 * GIBIBYTE
    assert failed.telemetry.peak_gpu_utilization_percent == 90.0
    assert failed.telemetry.peak_power_watts == 350.0
    assert failed.telemetry.peak_temperature_celsius == 75.0
    assert failed.telemetry.peak_host_memory_bytes == 3 * GIBIBYTE
    assert failed.telemetry.initialization_seconds == 3.0
    assert failed.telemetry.warmup_seconds == 2.0


def test_first_candidate_headroom_failure_falls_back_until_largest_smaller_passes() -> None:
    calls: list[int] = []

    def runner(config: FineTuneLaunchConfig) -> SmokeAttemptReceipt:
        batch = config.physical_batch_size
        calls.append(batch)
        return _receipt(batch, reserved_gib=92 if batch == 16 else 80)

    report = run_smoke_tests(
        base_config=_config(),
        physical_vram_bytes=96 * GIBIBYTE,
        experiment_config_sha256=SHA_A,
        dataset_manifest_sha256=SHA_B,
        runner=runner,
    )

    assert calls == [16, 8]
    assert report.selected_batch_size == 8


def test_headroom_uses_nvml_physical_free_memory_not_only_torch_reserved_memory() -> None:
    calls: list[int] = []

    def runner(config: FineTuneLaunchConfig) -> SmokeAttemptReceipt:
        batch = config.physical_batch_size
        calls.append(batch)
        return _receipt(
            batch,
            reserved_gib=70,
            free_gib=5 if batch == 16 else 20,
        )

    report = run_smoke_tests(
        base_config=_config(),
        physical_vram_bytes=96 * GIBIBYTE,
        experiment_config_sha256=SHA_A,
        dataset_manifest_sha256=SHA_B,
        runner=runner,
    )

    assert calls == [16, 8]
    assert report.selected_batch_size == 8


@pytest.mark.parametrize(
    ("receipt", "message"),
    [
        (_receipt(16, optimizer_steps=99), "exactly 100 optimizer steps"),
        (_receipt(16, accumulation=2), "gradient accumulation"),
    ],
)
def test_smoke_rejects_receipts_that_change_comparison_contract(
    receipt: SmokeAttemptReceipt,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        run_smoke_tests(
            base_config=_config(),
            physical_vram_bytes=96 * GIBIBYTE,
            experiment_config_sha256=SHA_A,
            dataset_manifest_sha256=SHA_B,
            runner=lambda _config: receipt,
        )


def test_non_finite_loss_is_recorded_as_unstable_and_never_selected() -> None:
    report = run_smoke_tests(
        base_config=_config(),
        physical_vram_bytes=40 * GIBIBYTE,
        experiment_config_sha256=SHA_A,
        dataset_manifest_sha256=SHA_B,
        runner=lambda config: _receipt(
            config.physical_batch_size,
            total_gib=40,
            reserved_gib=30,
            finite_loss=False,
        ),
    )

    assert report.selected_batch_size is None
    assert all(attempt.result.finite_loss is False for attempt in report.attempts)
    assert all(attempt.result.stable is False for attempt in report.attempts)
    assert all(attempt.result.error_code == "non_finite_loss" for attempt in report.attempts)


def test_telemetry_separates_warmup_from_steady_state_throughput() -> None:
    summary = summarize_telemetry(
        _samples(96, 80),
        initialization_seconds=4.0,
        warmup_seconds=6.0,
        steady_state_seconds=20.0,
        steady_state_optimizer_steps=50,
        physical_batch_size=16,
    )

    assert summary.initialization_seconds == 4.0
    assert summary.warmup_seconds == 6.0
    assert summary.steady_state_seconds == 20.0
    assert summary.steady_steps_per_second == 2.5
    assert summary.samples_per_second == 40.0


def test_headroom_telemetry_excludes_initialization_and_warmup_samples() -> None:
    samples = (
        TelemetrySample(0.0, 1, 2, 1 * GIBIBYTE, 1.0, 1.0, 1.0, 1),
        TelemetrySample(10.0, 3, 4, 20 * GIBIBYTE, 1.0, 1.0, 1.0, 1),
    )

    summary = summarize_telemetry(
        samples,
        initialization_seconds=4.0,
        warmup_seconds=6.0,
        steady_state_seconds=10.0,
        steady_state_optimizer_steps=50,
        physical_batch_size=16,
    )

    assert summary.minimum_steady_state_free_vram_bytes == 20 * GIBIBYTE


def test_telemetry_samples_during_a_blocking_sequential_launch() -> None:
    class Sampler:
        def __init__(self) -> None:
            self.calls = 0

        def sample(self) -> TelemetrySample:
            self.calls += 1
            return TelemetrySample(
                float(self.calls),
                self.calls,
                self.calls,
                96 * GIBIBYTE - self.calls,
                1.0,
                1.0,
                1.0,
                1,
            )

    sampler = Sampler()
    result, samples = sample_operation(
        lambda: (sleep(0.02), "complete")[1],
        sampler=sampler,
        sample_interval_seconds=0.001,
    )

    assert result == "complete"
    assert len(samples) >= 3
    assert samples == tuple(sorted(samples, key=lambda sample: sample.timestamp_seconds))
