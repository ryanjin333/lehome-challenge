from __future__ import annotations

import json
from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

import lehome_train.commands.report as report_module
import lehome_train.report_evidence as report_evidence
from lehome_train.checkpoints import CheckpointDescriptor
from lehome_train.commands.report import (
    build_training_report,
    write_training_report,
)
from lehome_train.commands.sync import SyncResult
from lehome_train.constants import DEFAULT_MODEL_REPO
from lehome_train.io import canonical_json_sha256
from lehome_train.models import (
    ArtifactIdentity,
    CheckpointRecord,
    ExperimentConfig,
    SmokeResult,
    SyncEntry,
    SyncManifest,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
REPOSITORY_COMMIT = "d" * 40
GROOT_REVISION = "e" * 40
BASE_REVISION = "f" * 40
DATASET_REVISION = "1" * 40
IMAGE_DIGEST = f"sha256:{'2' * 64}"


def _config(*, gradient_accumulation_steps: int = 1) -> ExperimentConfig:
    return ExperimentConfig(
        repository_commit=REPOSITORY_COMMIT,
        container_digest=IMAGE_DIGEST,
        model_repository="nvidia/GR00T-N1.7-3B",
        model_revision=BASE_REVISION,
        dataset_repository="ryanjin333/lehome-groot-n17-data",
        dataset_revision=DATASET_REVISION,
        dataset_manifest_sha256=SHA_A,
        physical_batch_size=64,
        gradient_accumulation_steps=gradient_accumulation_steps,
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
        gradient_accumulation_steps=config.gradient_accumulation_steps,
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
    content = f"checkpoint-{step}".encode("utf-8")
    return CheckpointDescriptor(
        record=CheckpointRecord(
            experiment_id="experiment-001",
            optimizer_step=step,
            sample_presentations=(
                step
                * config.physical_batch_size
                * config.gradient_accumulation_steps
            ),
            experiment_config_sha256=canonical_json_sha256(config),
            dataset_manifest_sha256=config.dataset_manifest_sha256,
            schedule_sha256=SHA_B,
            artifact=ArtifactIdentity(
                relative_path=f"checkpoints/step-{step}.tar.zst",
                sha256=hashlib.sha256(content).hexdigest(),
                byte_size=len(content),
            ),
            resumable=True,
            remotely_verified=True,
        ),
        normalization_sha256=SHA_C,
        schedule_sha256=SHA_B,
        locally_verified=True,
    )


def _write_local_checkpoints(
    root: Path,
    checkpoints: tuple[CheckpointDescriptor, ...],
) -> Path:
    root.mkdir()
    for checkpoint in checkpoints:
        artifact = root / checkpoint.record.artifact.relative_path
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(
            f"checkpoint-{checkpoint.record.optimizer_step}".encode("utf-8")
        )
    return root


def _sync_evidence(
    config: ExperimentConfig,
    checkpoints: tuple[CheckpointDescriptor, ...],
) -> SyncResult:
    entries = tuple(
        SyncEntry(
            relative_path=checkpoint.record.artifact.relative_path,
            sha256=checkpoint.record.artifact.sha256,
            byte_size=checkpoint.record.artifact.byte_size,
            remotely_verified=True,
        )
        for checkpoint in checkpoints
    )
    return SyncResult(
        repository=DEFAULT_MODEL_REPO,
        immutable_revision="3" * 40,
        manifest=SyncManifest(
            experiment_id="experiment-001",
            experiment_config_sha256=canonical_json_sha256(config),
            remote_prefix="experiments/experiment-001/" + "4" * 64,
            entries=entries,
        ),
        disposable=True,
    )


def test_report_contains_complete_exact_provenance_and_checkpoint_hashes(
    tmp_path: Path,
) -> None:
    config = _config()
    smoke = _smoke(config)
    checkpoints = (_checkpoint(config, 1_000), _checkpoint(config, 2_000))
    local_root = _write_local_checkpoints(tmp_path / "experiment", checkpoints)
    sync_evidence = _sync_evidence(config, checkpoints)

    report = build_training_report(
        experiment_config=config,
        isaac_groot_revision=GROOT_REVISION,
        smoke_result=smoke,
        checkpoints=checkpoints,
        local_artifact_root=local_root,
        sync_evidence=sync_evidence,
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
            "controller_reported_locally_verified": checkpoint.locally_verified,
            "controller_reported_remotely_verified": checkpoint.record.remotely_verified,
            "deletion_receipt": None,
            "local_evidence_level": "verified_bytes",
            "locally_verified": True,
            "normalization_sha256": checkpoint.normalization_sha256,
            "optimizer_step": checkpoint.record.optimizer_step,
            "remote_evidence_level": "immutable_sync_readback",
            "remotely_verified": True,
            "resumable": checkpoint.record.resumable,
            "retention_evidence_level": "verified_bytes",
            "retention_state": "retained_locally_verified",
            "retained_locally": True,
            "sample_presentations": checkpoint.record.sample_presentations,
            "schedule_sha256": checkpoint.schedule_sha256,
        }
        for checkpoint in checkpoints
    ]
    assert set(payload["checkpoints"][0]) == {
        "artifact",
        "controller_reported_locally_verified",
        "controller_reported_remotely_verified",
        "dataset_manifest_sha256",
        "deletion_receipt",
        "experiment_config_sha256",
        "experiment_id",
        "local_evidence_level",
        "locally_verified",
        "normalization_sha256",
        "optimizer_step",
        "remote_evidence_level",
        "remotely_verified",
        "resumable",
        "retention_evidence_level",
        "retention_state",
        "retained_locally",
        "sample_presentations",
        "schedule_sha256",
    }
    assert payload["sync_snapshot_disposable"] is True
    assert payload["shutdown_disposable"] is False
    assert "artifacts_disposable" not in payload
    assert payload["sync_evidence"] == {
        "immutable_revision": "3" * 40,
        "remote_prefix": sync_evidence.manifest.remote_prefix,
        "repository": DEFAULT_MODEL_REPO,
    }
    assert payload["runtime_seconds"] == 9_000.0
    assert payload["provider_hourly_price"] == 1.20
    assert payload["cost"] == pytest.approx(3.0)


def test_report_rejects_incomplete_or_incompatible_provenance(tmp_path: Path) -> None:
    config = _config()
    smoke = _smoke(config)
    checkpoint = _checkpoint(config, 1_000)
    local_root = _write_local_checkpoints(tmp_path / "experiment", (checkpoint,))

    with pytest.raises(ValueError, match="immutable.*dataset revision"):
        build_training_report(
            experiment_config=replace(config, dataset_revision="latest"),
            isaac_groot_revision=GROOT_REVISION,
            smoke_result=smoke,
            checkpoints=(checkpoint,),
            local_artifact_root=local_root,
            sync_evidence=None,
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
            local_artifact_root=local_root,
            sync_evidence=None,
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
            local_artifact_root=local_root,
            sync_evidence=None,
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
            local_artifact_root=local_root,
            sync_evidence=None,
            instance_started_at="2026-07-31T10:00:00Z",
            generated_at="2026-07-31T12:30:00Z",
            provider_hourly_price=1.20,
        )


def test_report_labels_controller_reported_pruning_without_receipt(
    tmp_path: Path,
) -> None:
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
        local_artifact_root=tmp_path,
        sync_evidence=_sync_evidence(config, (pruned,)),
        instance_started_at="2026-07-31T10:00:00Z",
        generated_at="2026-07-31T12:30:00Z",
        provider_hourly_price=1.20,
    )

    checkpoint = report.to_dict()["checkpoints"][0]
    assert checkpoint["retained_locally"] is None
    assert checkpoint["locally_verified"] is False
    assert checkpoint["retention_state"] == "controller_reported_pruned"
    assert checkpoint["retention_evidence_level"] == "reported_only"
    assert checkpoint["remotely_verified"] is True


def test_report_accepts_matching_pruning_receipt(tmp_path: Path) -> None:
    config = _config()
    pruned = replace(_checkpoint(config, 1_000), locally_verified=False)
    receipt = report_evidence.CheckpointPruningReceipt(
        experiment_id="experiment-001",
        experiment_config_sha256=canonical_json_sha256(config),
        artifact=pruned.record.artifact,
        immutable_revision="3" * 40,
        deleted_at="2026-07-31T11:00:00-07:00",
    )

    report = build_training_report(
        experiment_config=config,
        isaac_groot_revision=GROOT_REVISION,
        smoke_result=_smoke(config),
        checkpoints=(pruned,),
        local_artifact_root=tmp_path,
        sync_evidence=_sync_evidence(config, (pruned,)),
        pruning_receipts=(receipt,),
        instance_started_at="2026-07-31T10:00:00Z",
        generated_at="2026-07-31T19:00:00Z",
        provider_hourly_price=1.20,
    )

    checkpoint = report.to_dict()["checkpoints"][0]
    assert checkpoint["retained_locally"] is False
    assert checkpoint["retention_state"] == "pruned_with_receipt"
    assert checkpoint["retention_evidence_level"] == "deletion_receipt"


def test_report_calculates_runtime_and_cost_and_redacts_no_secret(
    tmp_path: Path,
) -> None:
    config = _config()
    token = "hf_" + "s" * 40
    checkpoint = _checkpoint(config, 1_000)
    local_root = _write_local_checkpoints(tmp_path / "experiment", (checkpoint,))

    report = build_training_report(
        experiment_config=config,
        isaac_groot_revision=GROOT_REVISION,
        smoke_result=_smoke(config),
        checkpoints=(checkpoint,),
        local_artifact_root=local_root,
        sync_evidence=None,
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
    assert payload["instance_started_at"] == "2026-07-31T10:15:00Z"
    assert payload["generated_at"] == "2026-07-31T11:45:30Z"
    assert token not in destination.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="secret") as error:
        build_training_report(
            experiment_config=replace(config, model_repository=token),
            isaac_groot_revision=GROOT_REVISION,
            smoke_result=_smoke(config),
            checkpoints=(checkpoint,),
            local_artifact_root=local_root,
            sync_evidence=None,
            instance_started_at="2026-07-31T10:00:00Z",
            generated_at="2026-07-31T11:00:00Z",
            provider_hourly_price=1.0,
        )
    assert token not in str(error.value)


def test_report_uses_effective_batch_for_checkpoint_sample_accounting(
    tmp_path: Path,
) -> None:
    config = _config(gradient_accumulation_steps=4)
    checkpoint = _checkpoint(config, 1_000)
    local_root = _write_local_checkpoints(tmp_path / "experiment", (checkpoint,))

    report = build_training_report(
        experiment_config=config,
        isaac_groot_revision=GROOT_REVISION,
        smoke_result=_smoke(config),
        checkpoints=(checkpoint,),
        local_artifact_root=local_root,
        sync_evidence=None,
        instance_started_at="2026-07-31T10:00:00Z",
        generated_at="2026-07-31T11:00:00Z",
        provider_hourly_price=1.0,
    )
    assert report.checkpoints[0]["sample_presentations"] == 256_000

    mismatched = replace(
        checkpoint,
        record=replace(checkpoint.record, sample_presentations=64_000),
    )
    with pytest.raises(ValueError, match="sample presentations"):
        build_training_report(
            experiment_config=config,
            isaac_groot_revision=GROOT_REVISION,
            smoke_result=_smoke(config),
            checkpoints=(mismatched,),
            local_artifact_root=local_root,
            sync_evidence=None,
            instance_started_at="2026-07-31T10:00:00Z",
            generated_at="2026-07-31T11:00:00Z",
            provider_hourly_price=1.0,
        )


def test_report_requires_bytes_for_local_claim_and_sync_for_remote_claim(
    tmp_path: Path,
) -> None:
    config = _config()
    checkpoint = _checkpoint(config, 1_000)

    with pytest.raises(ValueError, match="local checkpoint artifact"):
        build_training_report(
            experiment_config=config,
            isaac_groot_revision=GROOT_REVISION,
            smoke_result=_smoke(config),
            checkpoints=(checkpoint,),
            local_artifact_root=tmp_path,
            sync_evidence=None,
            instance_started_at="2026-07-31T10:00:00Z",
            generated_at="2026-07-31T11:00:00Z",
            provider_hourly_price=1.0,
        )

    reported_only = replace(checkpoint, locally_verified=False)
    report = build_training_report(
        experiment_config=config,
        isaac_groot_revision=GROOT_REVISION,
        smoke_result=_smoke(config),
        checkpoints=(reported_only,),
        local_artifact_root=tmp_path,
        sync_evidence=None,
        instance_started_at="2026-07-31T10:00:00Z",
        generated_at="2026-07-31T11:00:00Z",
        provider_hourly_price=1.0,
    )
    evidence = report.to_dict()["checkpoints"][0]
    assert evidence["locally_verified"] is False
    assert evidence["remotely_verified"] is False
    assert evidence["remote_evidence_level"] == "descriptor_reported_only"
    assert report.to_dict()["sync_snapshot_disposable"] is False
    assert report.to_dict()["shutdown_disposable"] is False


def test_report_rejects_local_checkpoint_byte_mismatch(tmp_path: Path) -> None:
    config = _config()
    checkpoint = _checkpoint(config, 1_000)
    artifact = tmp_path / checkpoint.record.artifact.relative_path
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"stale or corrupt checkpoint")

    with pytest.raises(ValueError, match="hash or size"):
        build_training_report(
            experiment_config=config,
            isaac_groot_revision=GROOT_REVISION,
            smoke_result=_smoke(config),
            checkpoints=(checkpoint,),
            local_artifact_root=tmp_path,
            sync_evidence=None,
            instance_started_at="2026-07-31T10:00:00Z",
            generated_at="2026-07-31T11:00:00Z",
            provider_hourly_price=1.0,
        )


def test_equivalent_timestamp_offsets_serialize_identically(tmp_path: Path) -> None:
    config = _config()
    checkpoint = _checkpoint(config, 1_000)
    local_root = _write_local_checkpoints(tmp_path / "experiment", (checkpoint,))
    common = {
        "experiment_config": config,
        "isaac_groot_revision": GROOT_REVISION,
        "smoke_result": _smoke(config),
        "checkpoints": (checkpoint,),
        "local_artifact_root": local_root,
        "sync_evidence": None,
        "provider_hourly_price": 1.0,
    }

    utc = build_training_report(
        **common,
        instance_started_at="2026-07-31T17:00:00Z",
        generated_at="2026-07-31T18:00:00Z",
    )
    offset = build_training_report(
        **common,
        instance_started_at="2026-07-31T10:00:00-07:00",
        generated_at="2026-07-31T11:00:00-07:00",
    )

    assert utc.instance_started_at == offset.instance_started_at == "2026-07-31T17:00:00Z"
    assert utc.generated_at == offset.generated_at == "2026-07-31T18:00:00Z"


def test_pruning_receipt_file_round_trip_is_strict_and_canonical(tmp_path: Path) -> None:
    config = _config()
    receipt = report_evidence.CheckpointPruningReceipt(
        experiment_id="experiment-001",
        experiment_config_sha256=canonical_json_sha256(config),
        artifact=_checkpoint(config, 1_000).record.artifact,
        immutable_revision="3" * 40,
        deleted_at="2026-07-31T11:00:00-07:00",
    )
    path = tmp_path / "receipt.json"

    report_evidence.write_checkpoint_pruning_receipt(path, receipt)
    loaded = report_evidence.load_checkpoint_pruning_receipt(path)

    assert loaded == receipt
    assert loaded.deleted_at == "2026-07-31T18:00:00Z"


def test_report_module_does_not_reexport_pruning_receipt_helpers() -> None:
    assert not hasattr(report_module, "CheckpointPruningReceipt")
    assert not hasattr(report_module, "load_checkpoint_pruning_receipt")
    assert not hasattr(report_module, "write_checkpoint_pruning_receipt")
