from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from lehome_train.io import parse_json
from lehome_train.models import (
    ArtifactIdentity,
    CameraMapping,
    CheckpointRecord,
    ExperimentConfig,
    JointMapping,
    MemorizationResult,
    PreparedDatasetProvenance,
    SmokeResult,
    SourceInspection,
    SyncEntry,
    SyncManifest,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
COMMIT = "c" * 40
DIGEST = f"sha256:{'d' * 64}"


def source_inspection() -> SourceInspection:
    return SourceInspection(
        source_repository="organizer/four-types",
        source_revision=COMMIT,
        source_manifest_sha256=SHA_A,
        episode_ids=("episode-0001",),
        camera_mappings=(
            CameraMapping(
                source_key="observation.images.front",
                target_modality="video.front",
            ),
        ),
        joint_mappings=(
            JointMapping(
                source_index=0,
                source_name="left_shoulder_pan",
                target_index=0,
                target_name="left_shoulder_pan",
                arm="left",
                action_mode="relative",
                is_gripper=False,
            ),
            JointMapping(
                source_index=5,
                source_name="left_gripper",
                target_index=5,
                target_name="left_gripper",
                arm="left",
                action_mode="absolute",
                is_gripper=True,
            ),
        ),
        fps=30.0,
        frame_count=120,
        episode_count=1,
    )


def test_models_are_frozen_and_detach_mutable_collection_aliases() -> None:
    episode_ids = ["episode-0001"]
    inspection = SourceInspection(
        **{
            **source_inspection().to_dict(),
            "episode_ids": episode_ids,
            "camera_mappings": list(source_inspection().camera_mappings),
            "joint_mappings": list(source_inspection().joint_mappings),
        }
    )

    episode_ids.append("episode-0002")

    assert inspection.episode_ids == ("episode-0001",)
    assert isinstance(inspection.camera_mappings, tuple)
    with pytest.raises(FrozenInstanceError):
        inspection.frame_count = 121


def test_strict_model_parser_rejects_unknown_fields() -> None:
    payload = source_inspection().to_dict()
    payload["access_token"] = "not-allowed"

    with pytest.raises(ValueError, match="unknown field"):
        parse_json(SourceInspection, payload)


def test_strict_model_parser_rejects_missing_and_wrong_typed_fields() -> None:
    missing = source_inspection().to_dict()
    del missing["source_revision"]
    wrong_type = source_inspection().to_dict()
    wrong_type["frame_count"] = "120"

    with pytest.raises(ValueError, match="missing field"):
        parse_json(SourceInspection, missing)
    with pytest.raises(ValueError, match="frame_count"):
        parse_json(SourceInspection, wrong_type)


def test_all_artifact_contracts_have_typed_identity_and_provenance() -> None:
    artifact = ArtifactIdentity(
        relative_path="checkpoints/step-1000/model.safetensors",
        sha256=SHA_A,
        byte_size=42,
    )
    inspection = source_inspection()
    provenance = PreparedDatasetProvenance(
        dataset_repository="ryanjin333/lehome-groot-n17-data",
        dataset_revision="lehome-groot-n17-v1",
        source_repository=inspection.source_repository,
        source_revision=inspection.source_revision,
        source_manifest_sha256=inspection.source_manifest_sha256,
        converter_commit=COMMIT,
        container_digest=DIGEST,
        train_episode_ids=("episode-0001",),
        validation_episode_ids=("episode-0002",),
        camera_mappings=inspection.camera_mappings,
        joint_mappings=inspection.joint_mappings,
        fps=30.0,
        frame_count=240,
        episode_count=2,
        source_artifacts=(
            ArtifactIdentity(
                relative_path="source/episode-0001.parquet",
                sha256=SHA_B,
                byte_size=24,
            ),
        ),
        artifacts=(artifact,),
    )
    config = ExperimentConfig(
        repository_commit=COMMIT,
        container_digest=DIGEST,
        model_repository="nvidia/GR00T-N1.7-3B",
        model_revision=COMMIT,
        dataset_repository=provenance.dataset_repository,
        dataset_revision=provenance.dataset_revision,
        dataset_manifest_sha256=SHA_A,
        physical_batch_size=16,
        gradient_accumulation_steps=1,
        sample_presentations=768_000,
        action_horizon=16,
        tune_language_backbone=False,
        tune_visual_backbone=False,
    )
    checkpoint = CheckpointRecord(
        experiment_id="experiment-001",
        optimizer_step=1_000,
        sample_presentations=16_000,
        experiment_config_sha256=SHA_A,
        dataset_manifest_sha256=SHA_B,
        schedule_sha256=SHA_A,
        artifact=artifact,
        resumable=True,
        remotely_verified=False,
    )
    smoke = SmokeResult(
        experiment_id=checkpoint.experiment_id,
        experiment_config_sha256=SHA_A,
        dataset_manifest_sha256=SHA_B,
        physical_batch_size=16,
        gradient_accumulation_steps=1,
        optimizer_steps=100,
        stable=True,
        finite_loss=True,
        physical_vram_bytes=96_000_000_000,
        peak_reserved_vram_bytes=80_000_000_000,
        minimum_steady_state_free_vram_bytes=16_000_000_000,
        steady_steps_per_second=1.5,
        samples_per_second=24.0,
        failure_reason=None,
    )
    memorization = MemorizationResult(
        experiment_id=checkpoint.experiment_id,
        experiment_config_sha256=SHA_A,
        dataset_manifest_sha256=SHA_B,
        episode_id="episode-0001",
        initialized_normalized_mse=1.0,
        final_normalized_mse=0.08,
        initialized_dimension_mse=(1.0, 1.0),
        final_dimension_mse=(0.08, 0.09),
        sample_presentations=10_000,
        offline_gate_passed=True,
        promotable=False,
        pending_gate="simulator_expert_replay",
    )
    sync_manifest = SyncManifest(
        experiment_id=checkpoint.experiment_id,
        experiment_config_sha256=SHA_A,
        remote_prefix="experiments/experiment-001/" + SHA_B,
        entries=(
            SyncEntry(
                relative_path=artifact.relative_path,
                sha256=artifact.sha256,
                byte_size=artifact.byte_size,
            ),
        ),
    )

    assert config.action_horizon == 16
    assert checkpoint.artifact.sha256 == SHA_A
    assert checkpoint.schedule_sha256 == SHA_A
    assert provenance.source_artifacts[0].sha256 == SHA_B
    assert smoke.dataset_manifest_sha256 == SHA_B
    assert smoke.failure_reason is None
    invalid_smoke = smoke.to_dict()
    invalid_smoke["minimum_steady_state_free_vram_bytes"] = (
        smoke.physical_vram_bytes + 1
    )
    with pytest.raises(ValueError, match="free VRAM.*physical"):
        SmokeResult(**invalid_smoke)
    assert memorization.experiment_config_sha256 == SHA_A
    assert memorization.promotable is False
    assert sync_manifest.entries[0].relative_path == artifact.relative_path
    assert sync_manifest.remote_prefix.endswith(SHA_B)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_manifest_sha256", "short"),
        ("fps", float("nan")),
        ("frame_count", -1),
    ],
)
def test_models_reject_invalid_identity_or_measurement(
    field: str,
    value: object,
) -> None:
    values = source_inspection().to_dict()
    values[field] = value

    with pytest.raises(ValueError, match=field):
        SourceInspection(**values)


@pytest.mark.parametrize(
    "relative_path",
    [
        "",
        "/absolute/artifact.json",
        "../outside.json",
        "reports/../../outside.json",
        ".hidden",
        "reports/.private/result.json",
        "reports/./result.json",
        "reports//result.json",
        "reports/result.json/",
    ],
)
@pytest.mark.parametrize("model_type", [ArtifactIdentity, SyncEntry])
def test_parsed_artifact_records_reject_noncanonical_relative_paths(
    model_type: type[ArtifactIdentity] | type[SyncEntry],
    relative_path: str,
) -> None:
    payload = {
        "relative_path": relative_path,
        "sha256": SHA_A,
        "byte_size": 1,
    }
    if model_type is SyncEntry:
        payload["remotely_verified"] = False

    with pytest.raises(ValueError, match="relative_path"):
        parse_json(model_type, payload)


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("initialized_dimension_mse", (-0.01, 1.0)),
        ("final_dimension_mse", (0.1, -0.01)),
    ],
)
def test_memorization_result_rejects_negative_dimension_mse(
    field: str,
    values: tuple[float, ...],
) -> None:
    payload = MemorizationResult(
        experiment_id="experiment-001",
        experiment_config_sha256=SHA_A,
        dataset_manifest_sha256=SHA_B,
        episode_id="episode-0001",
        initialized_normalized_mse=1.0,
        final_normalized_mse=0.08,
        initialized_dimension_mse=(1.0, 1.0),
        final_dimension_mse=(0.08, 0.09),
        sample_presentations=10_000,
        offline_gate_passed=True,
        promotable=False,
        pending_gate="simulator_expert_replay",
    ).to_dict()
    payload[field] = list(values)

    with pytest.raises(ValueError, match=field):
        parse_json(MemorizationResult, payload)
