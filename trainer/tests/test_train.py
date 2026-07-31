from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import sleep

import pytest

from lehome_train.checkpoints import GIBIBYTE, CheckpointDescriptor
from lehome_train.commands.train import (
    NonFiniteTrainingLoss,
    TrainingChunkRequest,
    TrainingChunkReceipt,
    run_fixed_exposure_training,
)
from lehome_train.io import canonical_json_sha256
from lehome_train.models import (
    ArtifactIdentity,
    CheckpointRecord,
    ExperimentConfig,
    SmokeResult,
)


SHA_B = "b" * 64
SHA_C = "c" * 64
COMMIT = "d" * 40
DIGEST = f"sha256:{'e' * 64}"


def _config(batch: int = 64) -> ExperimentConfig:
    return ExperimentConfig(
        repository_commit=COMMIT,
        container_digest=DIGEST,
        model_repository="nvidia/GR00T-N1.7-3B",
        model_revision=COMMIT,
        dataset_repository="private/prepared",
        dataset_revision=COMMIT,
        dataset_manifest_sha256=SHA_B,
        physical_batch_size=batch,
        gradient_accumulation_steps=1,
        sample_presentations=768_000,
        action_horizon=16,
        tune_language_backbone=False,
        tune_visual_backbone=False,
    )


def _smoke(config: ExperimentConfig) -> SmokeResult:
    return SmokeResult(
        experiment_id="experiment-001",
        experiment_config_sha256=canonical_json_sha256(config),
        dataset_manifest_sha256=config.dataset_manifest_sha256,
        physical_batch_size=config.physical_batch_size,
        gradient_accumulation_steps=1,
        optimizer_steps=100,
        stable=True,
        finite_loss=True,
        physical_vram_bytes=96 * GIBIBYTE,
        peak_reserved_vram_bytes=70 * GIBIBYTE,
        minimum_steady_state_free_vram_bytes=20 * GIBIBYTE,
        steady_steps_per_second=1.0,
        samples_per_second=float(config.physical_batch_size),
        failure_reason=None,
    )


def _receipt(
    request: TrainingChunkRequest,
    *,
    finite: bool = True,
    schedule_sha256: str | None = None,
) -> TrainingChunkReceipt:
    return TrainingChunkReceipt(
        schedule_sha256=(
            request.schedule_sha256 if schedule_sha256 is None else schedule_sha256
        ),
        start_optimizer_step=request.start_optimizer_step,
        end_optimizer_step=request.end_optimizer_step,
        sample_presentations=(
            request.end_sample_presentations - request.start_sample_presentations
        ),
        physical_batch_size=request.physical_batch_size,
        finite_loss=finite,
    )


def _checkpointer(config: ExperimentConfig, *, byte_size: int = GIBIBYTE):
    config_sha = canonical_json_sha256(config)

    def checkpoint(*, optimizer_step: int, sample_presentations: int) -> CheckpointDescriptor:
        return CheckpointDescriptor(
            record=CheckpointRecord(
                experiment_id="experiment-001",
                optimizer_step=optimizer_step,
                sample_presentations=sample_presentations,
                experiment_config_sha256=config_sha,
                dataset_manifest_sha256=SHA_B,
                artifact=ArtifactIdentity(
                    relative_path=f"checkpoints/step-{optimizer_step}.tar.zst",
                    sha256=f"{optimizer_step:064x}",
                    byte_size=byte_size,
                ),
                resumable=True,
                remotely_verified=False,
            ),
            normalization_sha256=SHA_C,
        )

    return checkpoint


def test_training_requires_selected_compatible_smoke_result() -> None:
    config = _config()

    with pytest.raises(ValueError, match="selected smoke result"):
        run_fixed_exposure_training(
            experiment_config=config,
            selected_smoke=None,
            normalization_sha256=SHA_C,
            runner=lambda **_kwargs: pytest.fail("runner must not start"),
            checkpointer=_checkpointer(config),
            uploader=lambda _checkpoint: True,
            disk_probe=lambda: 100 * GIBIBYTE,
            complete_checkpoint_bytes=GIBIBYTE,
        )

    with pytest.raises(ValueError, match="selected physical batch"):
        run_fixed_exposure_training(
            experiment_config=config,
            selected_smoke=replace(_smoke(config), physical_batch_size=32),
            normalization_sha256=SHA_C,
            runner=lambda **_kwargs: pytest.fail("runner must not start"),
            checkpointer=_checkpointer(config),
            uploader=lambda _checkpoint: True,
            disk_probe=lambda: 100 * GIBIBYTE,
            complete_checkpoint_bytes=GIBIBYTE,
        )

    low_headroom = replace(
        _smoke(config),
        minimum_steady_state_free_vram_bytes=9 * GIBIBYTE,
    )
    with pytest.raises(ValueError, match="headroom"):
        run_fixed_exposure_training(
            experiment_config=config,
            selected_smoke=low_headroom,
            normalization_sha256=SHA_C,
            runner=lambda **_kwargs: pytest.fail("runner must not start"),
            checkpointer=_checkpointer(config),
            uploader=lambda _checkpoint: True,
            disk_probe=lambda: 100 * GIBIBYTE,
            complete_checkpoint_bytes=GIBIBYTE,
        )


def test_training_runs_exact_exposure_and_twelve_checkpoint_boundaries() -> None:
    config = _config(batch=64)
    calls: list[TrainingChunkRequest] = []

    def runner(request: TrainingChunkRequest) -> TrainingChunkReceipt:
        calls.append(request)
        return _receipt(request)

    result = run_fixed_exposure_training(
        experiment_config=config,
        selected_smoke=_smoke(config),
        normalization_sha256=SHA_C,
        runner=runner,
        checkpointer=_checkpointer(config),
        uploader=lambda _checkpoint: True,
        disk_probe=lambda: 100 * GIBIBYTE,
        complete_checkpoint_bytes=GIBIBYTE,
    )

    assert result.status == "completed"
    assert result.final_optimizer_step == 12_000
    assert result.sample_presentations == 768_000
    assert len(calls) == len(result.checkpoints) == 12
    assert all(checkpoint.record.remotely_verified for checkpoint in result.checkpoints)
    first = calls[0]
    assert first.start_optimizer_step == 0
    assert first.end_optimizer_step == 1_000
    assert first.start_sample_presentations == 0
    assert first.end_sample_presentations == 64_000
    assert first.total_optimizer_steps == 12_000
    assert first.warmup_optimizer_steps == 600
    assert first.base_learning_rate == 0.0
    assert first.peak_learning_rate == pytest.approx(1e-4)
    assert first.start_learning_rate_multiplier == 0.0
    assert first.end_learning_rate_multiplier < 1.0
    assert first.start_learning_rate == 0.0
    assert first.end_learning_rate == pytest.approx(
        first.peak_learning_rate * first.end_learning_rate_multiplier
    )
    assert len({request.schedule_sha256 for request in calls}) == 1


def test_training_aborts_immediately_on_non_finite_loss() -> None:
    config = _config()
    checkpoint_calls: list[int] = []

    with pytest.raises(NonFiniteTrainingLoss, match="non-finite"):
        run_fixed_exposure_training(
            experiment_config=config,
            selected_smoke=_smoke(config),
            normalization_sha256=SHA_C,
            runner=lambda request: _receipt(request, finite=False),
            checkpointer=lambda **values: checkpoint_calls.append(
                int(values["optimizer_step"])
            ),
            uploader=lambda _checkpoint: True,
            disk_probe=lambda: 100 * GIBIBYTE,
            complete_checkpoint_bytes=GIBIBYTE,
        )

    assert checkpoint_calls == []


def test_training_resumes_only_after_verified_compatible_checkpoint() -> None:
    config = _config()
    resume = replace(
        _checkpointer(config)(optimizer_step=2_000, sample_presentations=128_000),
        record=replace(
            _checkpointer(config)(optimizer_step=2_000, sample_presentations=128_000).record,
            remotely_verified=True,
        ),
    )
    requests: list[TrainingChunkRequest] = []

    result = run_fixed_exposure_training(
        experiment_config=config,
        selected_smoke=_smoke(config),
        normalization_sha256=SHA_C,
        resume_checkpoint=resume,
        runner=lambda request: requests.append(request) or _receipt(request),
        checkpointer=_checkpointer(config),
        uploader=lambda _checkpoint: True,
        disk_probe=lambda: 100 * GIBIBYTE,
        complete_checkpoint_bytes=GIBIBYTE,
    )

    first = requests[0]
    assert first.start_optimizer_step == 2_000
    assert first.start_sample_presentations == 128_000
    assert first.start_learning_rate_multiplier > 0.0
    assert first.start_learning_rate_multiplier < 1.0
    assert first.start_learning_rate == pytest.approx(
        first.peak_learning_rate * first.start_learning_rate_multiplier
    )
    assert first.resume_checkpoint == resume
    assert result.final_optimizer_step == 12_000
    assert len(result.checkpoints) == 11


def test_training_rejects_runner_receipt_with_mismatched_schedule_identity() -> None:
    config = _config()

    with pytest.raises(ValueError, match="schedule identity"):
        run_fixed_exposure_training(
            experiment_config=config,
            selected_smoke=_smoke(config),
            normalization_sha256=SHA_C,
            runner=lambda request: _receipt(request, schedule_sha256="f" * 64),
            checkpointer=_checkpointer(config),
            uploader=lambda _checkpoint: True,
            disk_probe=lambda: 100 * GIBIBYTE,
            complete_checkpoint_bytes=GIBIBYTE,
        )


def test_failed_upload_pauses_before_boundary_when_disk_reserve_is_insufficient() -> None:
    config = _config()
    runner_calls: list[int] = []

    def runner(request: TrainingChunkRequest) -> TrainingChunkReceipt:
        sleep(0.01)
        runner_calls.append(request.end_optimizer_step)
        return _receipt(request)

    result = run_fixed_exposure_training(
        experiment_config=config,
        selected_smoke=_smoke(config),
        normalization_sha256=SHA_C,
        runner=runner,
        checkpointer=_checkpointer(config),
        uploader=lambda _checkpoint: False,
        disk_probe=lambda: 22 * GIBIBYTE - 1,
        complete_checkpoint_bytes=GIBIBYTE,
        upload_sleeper=lambda _delay: None,
    )

    assert result.status == "paused_disk_reserve"
    assert runner_calls == [1_000]
    assert result.failed_upload_optimizer_steps == (1_000,)


def test_disk_reserve_uses_largest_observed_complete_checkpoint() -> None:
    config = _config()
    runner_calls: list[int] = []

    def runner(request: TrainingChunkRequest) -> TrainingChunkReceipt:
        sleep(0.01)
        runner_calls.append(request.end_optimizer_step)
        return _receipt(request)

    result = run_fixed_exposure_training(
        experiment_config=config,
        selected_smoke=_smoke(config),
        normalization_sha256=SHA_C,
        runner=runner,
        checkpointer=_checkpointer(config, byte_size=5 * GIBIBYTE),
        uploader=lambda _checkpoint: False,
        disk_probe=lambda: 25 * GIBIBYTE,
        complete_checkpoint_bytes=GIBIBYTE,
        upload_sleeper=lambda _delay: None,
    )

    assert result.status == "paused_disk_reserve"
    assert runner_calls == [1_000]


def test_training_records_optional_provider_metadata_without_rental_actions(tmp_path: Path) -> None:
    config = _config()
    status_path = tmp_path / "training-status.json"

    result = run_fixed_exposure_training(
        experiment_config=config,
        selected_smoke=_smoke(config),
        normalization_sha256=SHA_C,
        runner=lambda request: _receipt(request),
        checkpointer=_checkpointer(config),
        uploader=lambda _checkpoint: True,
        disk_probe=lambda: 100 * GIBIBYTE,
        complete_checkpoint_bytes=GIBIBYTE,
        provider_hourly_price=1.75,
        instance_start_time="2026-07-31T12:00:00Z",
        status_path=status_path,
    )

    assert result.provider_hourly_price == 1.75
    assert result.instance_start_time == "2026-07-31T12:00:00Z"
    assert '"provider_hourly_price":1.75' in status_path.read_text(encoding="utf-8")
