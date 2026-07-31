from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from lehome_train.checkpoints import CheckpointDescriptor
from lehome_train.commands.report import build_training_report, write_training_report
from lehome_train.io import canonical_json_sha256
from lehome_train.models import (
    ArtifactIdentity,
    CheckpointRecord,
    ExperimentConfig,
    SmokeResult,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
REPOSITORY_COMMIT = "d" * 40
GROOT_REVISION = "e" * 40
BASE_REVISION = "f" * 40
DATASET_REVISION = "1" * 40
IMAGE_DIGEST = f"sha256:{'2' * 64}"


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        repository_commit=REPOSITORY_COMMIT,
        container_digest=IMAGE_DIGEST,
        model_repository="nvidia/GR00T-N1.7-3B",
        model_revision=BASE_REVISION,
        dataset_repository="ryanjin333/lehome-groot-n17-data",
        dataset_revision=DATASET_REVISION,
        dataset_manifest_sha256=SHA_A,
        physical_batch_size=64,
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
        physical_batch_size=64,
        gradient_accumulation_steps=1,
        optimizer_steps=100,
        stable=True,
        finite_loss=True,
        physical_vram_bytes=96 * 1024**3,
        peak_reserved_vram_bytes=70 * 1024**3,
        minimum_steady_state_free_vram_bytes=20 * 1024**3,
        steady_steps_per_second=1.25,
        samples_per_second=80.0,
        failure_reason=None,
    )


def _checkpoint(config: ExperimentConfig, step: int) -> CheckpointDescriptor:
    return CheckpointDescriptor(
        record=CheckpointRecord(
            experiment_id="experiment-001",
            optimizer_step=step,
            sample_presentations=step * config.physical_batch_size,
            experiment_config_sha256=canonical_json_sha256(config),
            dataset_manifest_sha256=config.dataset_manifest_sha256,
            schedule_sha256=SHA_B,
            artifact=ArtifactIdentity(
                relative_path=f"checkpoints/step-{step}.tar.zst",
                sha256=f"{step:064x}",
                byte_size=1234,
            ),
            resumable=True,
            remotely_verified=True,
        ),
        normalization_sha256=SHA_C,
        schedule_sha256=SHA_B,
        locally_verified=True,
    )


def test_report_contains_complete_exact_provenance_and_checkpoint_hashes() -> None:
    config = _config()
    smoke = _smoke(config)
    checkpoints = (_checkpoint(config, 1_000), _checkpoint(config, 2_000))

    report = build_training_report(
        experiment_config=config,
        isaac_groot_revision=GROOT_REVISION,
        smoke_result=smoke,
        checkpoints=checkpoints,
        instance_started_at="2026-07-31T10:00:00Z",
        generated_at="2026-07-31T12:30:00Z",
        provider_hourly_price=1.20,
    )
    payload = report.to_dict()

    assert payload["container_image_digest"] == IMAGE_DIGEST
    assert payload["repository_commit"] == REPOSITORY_COMMIT
    assert payload["isaac_groot_revision"] == GROOT_REVISION
    assert payload["base_model"] == {
        "repository": "nvidia/GR00T-N1.7-3B",
        "revision": BASE_REVISION,
    }
    assert payload["prepared_dataset"] == {
        "repository": "ryanjin333/lehome-groot-n17-data",
        "revision": DATASET_REVISION,
        "manifest_sha256": SHA_A,
    }
    assert payload["resolved_training_config"] == config.to_dict()
    assert payload["resolved_training_config_sha256"] == canonical_json_sha256(config)
    assert payload["smoke_metrics"] == smoke.to_dict()
    assert payload["checkpoints"] == [
        {
            "artifact": checkpoint.record.artifact.to_dict(),
            "dataset_manifest_sha256": checkpoint.record.dataset_manifest_sha256,
            "experiment_config_sha256": checkpoint.record.experiment_config_sha256,
            "experiment_id": checkpoint.record.experiment_id,
            "locally_verified": checkpoint.locally_verified,
            "normalization_sha256": checkpoint.normalization_sha256,
            "optimizer_step": checkpoint.record.optimizer_step,
            "remotely_verified": checkpoint.record.remotely_verified,
            "resumable": checkpoint.record.resumable,
            "retention_state": "retained_locally",
            "retained_locally": checkpoint.locally_verified,
            "sample_presentations": checkpoint.record.sample_presentations,
            "schedule_sha256": checkpoint.schedule_sha256,
        }
        for checkpoint in checkpoints
    ]
    assert set(payload["checkpoints"][0]) == {
        "artifact",
        "dataset_manifest_sha256",
        "experiment_config_sha256",
        "experiment_id",
        "locally_verified",
        "normalization_sha256",
        "optimizer_step",
        "remotely_verified",
        "resumable",
        "retention_state",
        "retained_locally",
        "sample_presentations",
        "schedule_sha256",
    }
    assert payload["runtime_seconds"] == 9_000.0
    assert payload["provider_hourly_price"] == 1.20
    assert payload["cost"] == pytest.approx(3.0)


def test_report_rejects_incomplete_or_incompatible_provenance() -> None:
    config = _config()
    smoke = _smoke(config)
    checkpoint = _checkpoint(config, 1_000)

    with pytest.raises(ValueError, match="immutable.*dataset revision"):
        build_training_report(
            experiment_config=replace(config, dataset_revision="latest"),
            isaac_groot_revision=GROOT_REVISION,
            smoke_result=smoke,
            checkpoints=(checkpoint,),
            instance_started_at="2026-07-31T10:00:00Z",
            generated_at="2026-07-31T12:30:00Z",
            provider_hourly_price=1.20,
        )

    with pytest.raises(ValueError, match="checkpoint.*config"):
        build_training_report(
            experiment_config=config,
            isaac_groot_revision=GROOT_REVISION,
            smoke_result=smoke,
            checkpoints=(
                replace(
                    checkpoint,
                    record=replace(
                        checkpoint.record,
                        experiment_config_sha256="9" * 64,
                    ),
                ),
            ),
            instance_started_at="2026-07-31T10:00:00Z",
            generated_at="2026-07-31T12:30:00Z",
            provider_hourly_price=1.20,
        )

    with pytest.raises(ValueError, match="normalization"):
        build_training_report(
            experiment_config=config,
            isaac_groot_revision=GROOT_REVISION,
            smoke_result=smoke,
            checkpoints=(
                checkpoint,
                replace(_checkpoint(config, 2_000), normalization_sha256="8" * 64),
            ),
            instance_started_at="2026-07-31T10:00:00Z",
            generated_at="2026-07-31T12:30:00Z",
            provider_hourly_price=1.20,
        )

    with pytest.raises(ValueError, match="schedule"):
        build_training_report(
            experiment_config=config,
            isaac_groot_revision=GROOT_REVISION,
            smoke_result=smoke,
            checkpoints=(
                checkpoint,
                replace(
                    _checkpoint(config, 2_000),
                    record=replace(
                        _checkpoint(config, 2_000).record,
                        schedule_sha256="7" * 64,
                    ),
                    schedule_sha256="7" * 64,
                ),
            ),
            instance_started_at="2026-07-31T10:00:00Z",
            generated_at="2026-07-31T12:30:00Z",
            provider_hourly_price=1.20,
        )


def test_report_preserves_pruned_checkpoint_retention_state() -> None:
    config = _config()
    pruned = replace(
        _checkpoint(config, 1_000),
        locally_verified=False,
        record=replace(_checkpoint(config, 1_000).record, remotely_verified=True),
    )

    report = build_training_report(
        experiment_config=config,
        isaac_groot_revision=GROOT_REVISION,
        smoke_result=_smoke(config),
        checkpoints=(pruned,),
        instance_started_at="2026-07-31T10:00:00Z",
        generated_at="2026-07-31T12:30:00Z",
        provider_hourly_price=1.20,
    )

    assert report.to_dict()["checkpoints"][0]["retained_locally"] is False
    assert report.to_dict()["checkpoints"][0]["locally_verified"] is False
    assert (
        report.to_dict()["checkpoints"][0]["retention_state"]
        == "pruned_after_remote_verification"
    )


def test_report_calculates_runtime_and_cost_and_redacts_no_secret(
    tmp_path: Path,
) -> None:
    config = _config()
    token = "hf_" + "s" * 40

    report = build_training_report(
        experiment_config=config,
        isaac_groot_revision=GROOT_REVISION,
        smoke_result=_smoke(config),
        checkpoints=(_checkpoint(config, 1_000),),
        instance_started_at="2026-07-31T10:15:00+00:00",
        generated_at="2026-07-31T11:45:30+00:00",
        provider_hourly_price=2.40,
    )
    destination = tmp_path / "reports" / "training-report.json"
    destination.parent.mkdir()
    write_training_report(destination, report)

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["runtime_seconds"] == 5_430.0
    assert payload["cost"] == pytest.approx(3.62)
    assert token not in destination.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="secret") as error:
        build_training_report(
            experiment_config=replace(config, model_repository=token),
            isaac_groot_revision=GROOT_REVISION,
            smoke_result=_smoke(config),
            checkpoints=(_checkpoint(config, 1_000),),
            instance_started_at="2026-07-31T10:00:00Z",
            generated_at="2026-07-31T11:00:00Z",
            provider_hourly_price=1.0,
        )
    assert token not in str(error.value)
