from __future__ import annotations

import pytest

from lehome_train.constants import MODEL_REVISION
from lehome_train.flywheel.augmentation import augmentation_profile
from lehome_train.groot.config import FineTuneLaunchConfig
from lehome_train.io import canonical_json_sha256


REVISION = "a" * 40
PARENT_REVISION = "b" * 40
PARENT_DIGEST = "c" * 64


def config(**overrides: object) -> FineTuneLaunchConfig:
    values: dict[str, object] = {
        "base_model_path": "nvidia/GR00T-N1.7-3B",
        "base_model_revision": MODEL_REVISION,
        "dataset_path": "/prepared/lehome-groot-n17-v1",
        "dataset_revision": REVISION,
        "modality_config_path": "/prepared/lehome-groot-n17-v1/meta/lehome_groot_modality.py",
        "output_dir": "/output/baseline",
        "experiment_name": "lehome-groot-baseline",
        "physical_batch_size": 64,
        "max_steps": 12_000,
        "save_steps": 1_000,
        "warmup_ratio": 0.05,
    }
    values.update(overrides)
    return FineTuneLaunchConfig(**values)


def canonical_holdout_receipt() -> dict[str, object]:
    receipt: dict[str, object] = {
        "schema_version": 1,
        "canonical_holdout_id": "lehome-canonical-holdout-v1",
        "dataset_revision": "a" * 40,
        "policy_revision": "b" * 40,
        "evaluation_manifest_sha256": "c" * 64,
        "mild_profile_sha256": augmentation_profile("mild").sha256,
        "metric_name": "success_rate",
        "metric_direction": "higher_is_better",
        "baseline_metric": 0.80,
        "candidate_metric": 0.79,
        "max_allowed_regression": 0.02,
        "non_regression_passed": True,
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    return receipt


def test_config_separates_40_step_model_capacity_from_16_step_training_target() -> None:
    resolved = config()

    assert resolved.num_gpus == 1
    assert resolved.global_batch_size == 64
    assert resolved.gradient_accumulation_steps == 1
    assert resolved.model_action_chunk_capacity == 40
    assert resolved.training_action_horizon == 16
    assert resolved.tune_llm is False
    assert resolved.tune_visual is False
    assert resolved.tune_projector is True
    assert resolved.tune_diffusion_model is True
    assert resolved.base_model_revision == MODEL_REVISION
    assert resolved.identity()["lr_scheduler_type"] == "cosine"
    assert resolved.identity()["decay_semantics"] == "cosine_remainder_after_warmup"
    assert resolved.identity()["augmentation_profile"] == "none"
    assert resolved.identity()["augmentation_profile_sha256"] == augmentation_profile("none").sha256


def test_config_records_four_gpu_global_batch_math_and_presentations() -> None:
    distributed = config(
        physical_batch_size=1,
        num_gpus=4,
        global_batch_size=4,
    )

    assert distributed.num_gpus == 4
    assert distributed.physical_batch_size == 1
    assert distributed.gradient_accumulation_steps == 1
    assert distributed.global_batch_size == 4
    assert distributed.sample_presentations_for_optimizer_steps(100) == 400
    assert distributed.identity()["global_batch_size"] == 4


def test_runtime_resume_cursor_is_derived_only_from_an_authenticated_checkpoint() -> None:
    initial = config(runtime_mixture_manifest="/runtime/m.json", runtime_window_index="/runtime/w.json", runtime_mounts_descriptor="/runtime/d.json")
    assert initial.runtime_resume_offset_for_global_step(3) == 192
    with pytest.raises(ValueError, match="authenticated checkpoint"):
        config(runtime_mixture_manifest="/runtime/m.json", runtime_window_index="/runtime/w.json", runtime_mounts_descriptor="/runtime/d.json", runtime_resume_sample_offset=64)
    with pytest.raises(ValueError, match="global_step"):
        initial.runtime_resume_offset_for_global_step(-1)


def test_runtime_checkpoint_cursor_is_not_an_immutable_launch_identity_field() -> None:
    immutable = config(
        runtime_mixture_manifest="/runtime/m.json",
        runtime_window_index="/runtime/w.json",
        runtime_mounts_descriptor="/runtime/d.json",
    )

    assert "runtime_resume_global_step" not in immutable.identity()


def test_runtime_resume_cursor_must_match_the_bound_checkpoint_step() -> None:
    with pytest.raises(ValueError, match="authenticated checkpoint"):
        config(
            runtime_mixture_manifest="/runtime/m.json",
            runtime_window_index="/runtime/w.json",
            runtime_mounts_descriptor="/runtime/d.json",
            runtime_resume_sample_offset=128,
            runtime_resume_global_step=1,
        )


def test_rft_config_binds_step_12000_parent_capacity_to_a_16_step_training_target() -> None:
    resolved = config(
        base_model_path="/cache/models/lehome/policies/step-12000",
        max_steps=2_000,
        save_steps=1_000,
        parent_checkpoint_repository="ryanjin333/lehome-groot-n17-models",
        parent_checkpoint_revision=PARENT_REVISION,
        parent_checkpoint_subpath="policies/step-12000",
        parent_checkpoint_artifact_sha256=PARENT_DIGEST,
    )

    assert resolved.model_action_chunk_capacity == 40
    assert resolved.training_action_horizon == 16
    assert resolved.max_steps == 2_000
    assert resolved.save_steps == 1_000
    assert resolved.identity()["parent_checkpoint_artifact_sha256"] == PARENT_DIGEST


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_action_chunk_capacity", 16, "model action chunk capacity"),
        ("training_action_horizon", 40, "training action horizon"),
        ("training_action_horizon", 8, "training action horizon"),
    ],
)
def test_config_refuses_dual_or_noncanonical_horizon_contracts(
    field: str, value: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        config(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("global_batch_size", 1),
        ("global_batch_size", 8),
        ("physical_batch_size", 2),
    ],
)
def test_four_gpu_config_refuses_incompatible_batch_math(field: str, value: int) -> None:
    values = {
        "physical_batch_size": 1,
        "num_gpus": 4,
        "global_batch_size": 4,
    }
    values[field] = value
    with pytest.raises(ValueError, match="global batch|per-device batch"):
        config(**values)


def test_config_records_augmentation_profile_hash_and_strict_gate_receipt() -> None:
    reference = config(
        augmentation_profile="nvidia_reference",
        augmentation_receipt=canonical_holdout_receipt(),
    )

    assert reference.identity()["augmentation_profile"] == "nvidia_reference"
    assert reference.identity()["augmentation_profile_sha256"] == augmentation_profile(
        "nvidia_reference", receipt=canonical_holdout_receipt()
    ).sha256
    assert reference.identity()["augmentation_receipt"] == canonical_holdout_receipt()


def test_config_identity_changes_when_augmentation_profile_changes() -> None:
    assert config().identity() != config(augmentation_profile="mild").identity()


def test_config_rejects_nvidia_reference_without_a_valid_receipt() -> None:
    with pytest.raises(ValueError, match="canonical-holdout receipt"):
        config(augmentation_profile="nvidia_reference")

    with pytest.raises(ValueError, match="secret"):
        config(
            augmentation_profile="nvidia_reference",
            augmentation_receipt={**canonical_holdout_receipt(), "token": "nope"},
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("num_gpus", 0, "exactly one GPU"),
        ("num_gpus", 2, "exactly one GPU"),
        ("global_batch_size", 32, "physical batch"),
        ("gradient_accumulation_steps", 2, "gradient accumulation"),
        ("warmup_ratio", 0.0, "warmup_ratio"),
        ("experiment_name", "../outside", "experiment_name"),
        ("experiment_name", "/outside", "experiment_name"),
        ("base_model_revision", "main", "pinned"),
        ("dataset_revision", "lehome-groot-n17-v1", "pinned"),
        ("tune_llm", True, "tune_llm"),
        ("tune_visual", True, "tune_visual"),
        ("tune_projector", False, "tune_projector"),
        ("tune_diffusion_model", False, "tune_diffusion_model"),
    ],
)
def test_config_refuses_incompatible_or_unpinned_inputs(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        config(**{field: value})
