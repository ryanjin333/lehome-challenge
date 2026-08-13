from __future__ import annotations

from time import sleep

import pytest

from lehome_train.batch_select import GIBIBYTE, NoStableBatchError
from lehome_train.commands.smoke import (
    SMOKE_OPTIMIZER_STEPS,
    SmokeAttemptReceipt,
    SmokeRunnerFailure,
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
        TelemetrySample(
            0.0,
            4 * GIBIBYTE,
            5 * GIBIBYTE,
            (total_gib - 5) * GIBIBYTE,
            20.0,
            100.0,
            45.0,
            2 * GIBIBYTE,
            total_gib * GIBIBYTE,
            "GPU-test",
        ),
        TelemetrySample(
            10.0,
            int((reserved_gib - 1) * GIBIBYTE),
            int(reserved_gib * GIBIBYTE),
            int(observed_free_gib * GIBIBYTE),
            90.0,
            350.0,
            75.0,
            3 * GIBIBYTE,
            total_gib * GIBIBYTE,
            "GPU-test",
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
    failure_reason: str | None = None,
    free_gib: float | None = None,
    telemetry_samples: tuple[TelemetrySample, ...] | None = None,
    initialization_seconds: float = 3.0,
    warmup_seconds: float = 2.0,
    steady_state_seconds: float = 10.0,
) -> SmokeAttemptReceipt:
    resolved_failure_reason = (
        "non_finite_loss"
        if not finite_loss and failure_reason is None
        else failure_reason
    )
    return SmokeAttemptReceipt(
        optimizer_steps=optimizer_steps,
        gradient_accumulation_steps=accumulation,
        finite_loss=finite_loss,
        initialization_seconds=initialization_seconds,
        warmup_seconds=warmup_seconds,
        steady_state_seconds=steady_state_seconds,
        steady_state_optimizer_steps=min(80, optimizer_steps),
        telemetry_samples=(
            _samples(total_gib, reserved_gib, free_gib=free_gib)
            if telemetry_samples is None
            else telemetry_samples
        ),
        failure_reason=resolved_failure_reason,
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
    assert {attempt.result.experiment_id for attempt in report.attempts} == {"smoke"}
    assert [attempt.attempt_experiment_id for attempt in report.attempts] == [
        "smoke-batch-16",
        "smoke-batch-32",
        "smoke-batch-64",
    ]
    assert all(config.max_steps == SMOKE_OPTIMIZER_STEPS == 100 for config in calls)
    assert all(config.global_batch_size == config.physical_batch_size for config in calls)
    assert all(config.gradient_accumulation_steps == 1 for config in calls)
    assert all(config.model_action_chunk_capacity == 40 for config in calls)
    assert all(config.training_action_horizon == 16 for config in calls)
    assert report.selected_batch_size == 64
    assert [attempt.result.physical_batch_size for attempt in report.attempts] == [16, 32, 64]


def test_four_gpu_smoke_runs_only_the_100_step_per_device_one_profile() -> None:
    calls: list[FineTuneLaunchConfig] = []

    def runner(config: FineTuneLaunchConfig) -> SmokeAttemptReceipt:
        calls.append(config)
        return __import__("dataclasses").replace(
            _receipt(1, total_gib=24, reserved_gib=20),
            per_device_telemetry_samples=tuple(
                _samples(24, 20) for _ in range(4)
            ),
        )

    report = run_smoke_tests(
        base_config=_config(
            physical_batch_size=1,
            global_batch_size=4,
            num_gpus=4,
        ),
        physical_vram_bytes=24 * GIBIBYTE,
        experiment_config_sha256=SHA_A,
        dataset_manifest_sha256=SHA_B,
        runner=runner,
    )

    assert len(calls) == 1
    assert calls[0].physical_batch_size == 1
    assert calls[0].global_batch_size == 4
    assert calls[0].gradient_accumulation_steps == 1
    assert calls[0].max_steps == SMOKE_OPTIMIZER_STEPS == 100
    assert report.selected_batch_size == 4
    assert report.attempts[0].result.physical_batch_size == 4


def test_four_gpu_smoke_requires_headroom_evidence_from_every_visible_device() -> None:
    class MultiSampler:
        def __init__(self) -> None:
            self.calls = 0
            self.closed = False

        def sample(self) -> TelemetrySample:
            return self.sample_all()[0]

        def sample_all(self) -> tuple[TelemetrySample, ...]:
            timestamp = float(self.calls * 10)
            self.calls += 1
            return tuple(
                TelemetrySample(
                    timestamp,
                    4 * GIBIBYTE,
                    20 * GIBIBYTE,
                    free * GIBIBYTE,
                    80.0,
                    300.0,
                    70.0,
                    2 * GIBIBYTE,
                    24 * GIBIBYTE,
                    f"GPU-{index}",
                )
                for index, free in enumerate((3, 3, 3, 2))
            )

        def close(self) -> None:
            self.closed = True

    sampler = MultiSampler()

    def runner(_config: FineTuneLaunchConfig) -> SmokeAttemptReceipt:
        return __import__("dataclasses").replace(
            _receipt(1, total_gib=24, reserved_gib=20), telemetry_samples=()
        )

    report = run_smoke_tests(
        base_config=_config(
            physical_batch_size=1,
            global_batch_size=4,
            num_gpus=4,
        ),
        physical_vram_bytes=24 * GIBIBYTE,
        experiment_config_sha256=SHA_A,
        dataset_manifest_sha256=SHA_B,
        runner=runner,
        sampler_factory=lambda: sampler,
    )

    assert sampler.closed is True
    assert report.selected_batch_size is None
    assert report.attempts[0].result.stable is False
    assert len(report.attempts[0].per_device_telemetry) == 4
    assert report.attempts[0].per_device_telemetry[-1].minimum_steady_state_free_vram_bytes == 2 * GIBIBYTE


def test_memory_failure_stops_larger_batches_and_preserves_complete_attempt_record() -> None:
    calls: list[int] = []

    def runner(config: FineTuneLaunchConfig) -> SmokeAttemptReceipt:
        calls.append(config.physical_batch_size)
        if config.physical_batch_size == 32:
            return _receipt(
                32,
                reserved_gib=95,
                optimizer_steps=17,
                failure_reason="cuda_oom",
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
    assert failed.result.failure_reason == "cuda_oom"
    assert failed.completed_optimizer_steps == 17
    assert failed.telemetry.peak_allocated_vram_bytes > 0
    assert failed.telemetry.peak_reserved_vram_bytes == 95 * GIBIBYTE
    assert failed.telemetry.peak_gpu_utilization_percent == 90.0
    assert failed.telemetry.peak_power_watts == 350.0
    assert failed.telemetry.peak_temperature_celsius == 75.0
    assert failed.telemetry.peak_host_memory_bytes == 3 * GIBIBYTE
    assert failed.telemetry.initialization_seconds == 3.0
    assert failed.telemetry.warmup_seconds == 2.0


def test_oom_before_steady_state_is_recorded_and_falls_back() -> None:
    calls: list[int] = []
    early_sample = _samples(96, 5)[0]

    def runner(config: FineTuneLaunchConfig) -> SmokeAttemptReceipt:
        batch = config.physical_batch_size
        calls.append(batch)
        if batch == 16:
            return _receipt(
                batch,
                optimizer_steps=0,
                failure_reason="cuda_oom",
                telemetry_samples=(early_sample,),
                initialization_seconds=20.0,
                warmup_seconds=0.0,
                steady_state_seconds=0.0,
            )
        return _receipt(batch, reserved_gib=70)

    report = run_smoke_tests(
        base_config=_config(),
        physical_vram_bytes=96 * GIBIBYTE,
        experiment_config_sha256=SHA_A,
        dataset_manifest_sha256=SHA_B,
        runner=runner,
    )

    assert calls == [16, 8]
    assert report.selected_batch_size == 8
    failure = report.attempts[0]
    assert failure.result.failure_reason == "cuda_oom"
    assert failure.result.minimum_steady_state_free_vram_bytes is None
    assert failure.result.steady_steps_per_second == 0.0
    assert failure.telemetry.peak_reserved_vram_bytes == 5 * GIBIBYTE


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
    assert all(
        attempt.result.failure_reason == "non_finite_loss"
        for attempt in report.attempts
    )


def test_typed_raised_oom_keeps_samples_and_unknown_exception_aborts() -> None:
    calls: list[int] = []

    class Sampler:
        def __init__(self) -> None:
            self.calls = 0

        def sample(self) -> TelemetrySample:
            self.calls += 1
            return TelemetrySample(
                float(self.calls - 1),
                self.calls,
                self.calls,
                90 * GIBIBYTE,
                1.0,
                1.0,
                1.0,
                1,
                96 * GIBIBYTE,
                "GPU-test",
            )

    def runner(config: FineTuneLaunchConfig) -> SmokeAttemptReceipt:
        batch = config.physical_batch_size
        calls.append(batch)
        if batch == 16:
            raise SmokeRunnerFailure.cuda_oom(initialization_seconds=1.0)
        return _receipt(batch)

    report = run_smoke_tests(
        base_config=_config(),
        physical_vram_bytes=96 * GIBIBYTE,
        experiment_config_sha256=SHA_A,
        dataset_manifest_sha256=SHA_B,
        runner=runner,
        sampler_factory=Sampler,
    )

    assert calls == [16, 8]
    assert report.selected_batch_size == 8
    assert report.attempts[0].result.failure_reason == "cuda_oom"
    assert report.attempts[0].telemetry.physical_total_vram_bytes == 96 * GIBIBYTE

    with pytest.raises(RuntimeError, match="unknown runner failure"):
        run_smoke_tests(
            base_config=_config(),
            physical_vram_bytes=96 * GIBIBYTE,
            experiment_config_sha256=SHA_A,
            dataset_manifest_sha256=SHA_B,
            runner=lambda _config: (_ for _ in ()).throw(
                RuntimeError("unknown runner failure")
            ),
            sampler_factory=Sampler,
        )


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
        TelemetrySample(
            0.0, 1, 2, 1 * GIBIBYTE, 1.0, 1.0, 1.0, 1,
            96 * GIBIBYTE, "GPU-test",
        ),
        TelemetrySample(
            10.0, 3, 4, 20 * GIBIBYTE, 1.0, 1.0, 1.0, 1,
            96 * GIBIBYTE, "GPU-test",
        ),
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


def test_telemetry_rejects_total_free_and_device_identity_drift() -> None:
    with pytest.raises(ValueError, match="free VRAM.*total"):
        TelemetrySample(
            0.0, 1, 2, 97 * GIBIBYTE, 1.0, 1.0, 1.0, 1,
            96 * GIBIBYTE, "GPU-test",
        )

    total_drift = (
        TelemetrySample(
            0.0, 1, 2, 40 * GIBIBYTE, 1.0, 1.0, 1.0, 1,
            96 * GIBIBYTE, "GPU-test",
        ),
        TelemetrySample(
            1.0, 1, 2, 40 * GIBIBYTE, 1.0, 1.0, 1.0, 1,
            64 * GIBIBYTE, "GPU-test",
        ),
    )
    with pytest.raises(ValueError, match="total.*drift"):
        summarize_telemetry(
            total_drift,
            initialization_seconds=0.0,
            warmup_seconds=0.0,
            steady_state_seconds=1.0,
            steady_state_optimizer_steps=1,
            physical_batch_size=1,
        )

    device_drift = (
        total_drift[0],
        TelemetrySample(
            1.0, 1, 2, 40 * GIBIBYTE, 1.0, 1.0, 1.0, 1,
            96 * GIBIBYTE, "GPU-other",
        ),
    )
    with pytest.raises(ValueError, match="device.*drift"):
        summarize_telemetry(
            device_drift,
            initialization_seconds=0.0,
            warmup_seconds=0.0,
            steady_state_seconds=1.0,
            steady_state_optimizer_steps=1,
            physical_batch_size=1,
        )


def test_smoke_rejects_preflight_and_nvml_total_mismatch() -> None:
    with pytest.raises(ValueError, match="expected physical VRAM"):
        run_smoke_tests(
            base_config=_config(),
            physical_vram_bytes=96 * GIBIBYTE,
            experiment_config_sha256=SHA_A,
            dataset_manifest_sha256=SHA_B,
            runner=lambda config: _receipt(
                config.physical_batch_size,
                total_gib=64,
                reserved_gib=40,
            ),
        )


def test_smoke_uses_canonical_batch_selector(monkeypatch: pytest.MonkeyPatch) -> None:
    selected: list[tuple[int, ...]] = []

    def selector(results: object) -> int:
        batches = tuple(result.physical_batch_size for result in results)  # type: ignore[union-attr]
        selected.append(batches)
        return max(batches)

    monkeypatch.setattr("lehome_train.commands.smoke.select_largest_stable_batch", selector)
    report = run_smoke_tests(
        base_config=_config(),
        physical_vram_bytes=96 * GIBIBYTE,
        experiment_config_sha256=SHA_A,
        dataset_manifest_sha256=SHA_B,
        runner=lambda config: _receipt(config.physical_batch_size),
    )

    assert selected == [(16, 32, 64)]
    assert report.selected_batch_size == 64


def test_no_stable_batch_is_none_but_unrelated_value_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lehome_train.commands.smoke.select_largest_stable_batch",
        lambda _results: (_ for _ in ()).throw(NoStableBatchError("typed")),
    )
    report = run_smoke_tests(
        base_config=_config(),
        physical_vram_bytes=96 * GIBIBYTE,
        experiment_config_sha256=SHA_A,
        dataset_manifest_sha256=SHA_B,
        runner=lambda config: _receipt(config.physical_batch_size),
    )
    assert report.selected_batch_size is None

    monkeypatch.setattr(
        "lehome_train.commands.smoke.select_largest_stable_batch",
        lambda _results: (_ for _ in ()).throw(
            ValueError("no stable batch text from unrelated validation")
        ),
    )
    with pytest.raises(ValueError, match="unrelated validation"):
        run_smoke_tests(
            base_config=_config(),
            physical_vram_bytes=96 * GIBIBYTE,
            experiment_config_sha256=SHA_A,
            dataset_manifest_sha256=SHA_B,
            runner=lambda config: _receipt(config.physical_batch_size),
        )


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
                96 * GIBIBYTE,
                "GPU-test",
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
