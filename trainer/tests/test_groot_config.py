from __future__ import annotations

import pytest

from lehome_train.constants import MODEL_REVISION
from lehome_train.flywheel.augmentation import augmentation_profile
from lehome_train.groot.config import FineTuneLaunchConfig
from lehome_train.io import canonical_json_sha256


REVISION = "a" * 40


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


def test_config_enforces_single_gpu_fixed_batch_horizon_and_freezing() -> None:
    resolved = config()

    assert resolved.num_gpus == 1
    assert resolved.global_batch_size == 64
    assert resolved.gradient_accumulation_steps == 1
    assert resolved.action_horizon == 16
    assert resolved.tune_llm is False
    assert resolved.tune_visual is False
    assert resolved.tune_projector is True
    assert resolved.tune_diffusion_model is True
    assert resolved.base_model_revision == MODEL_REVISION
    assert resolved.identity()["lr_scheduler_type"] == "cosine"
    assert resolved.identity()["decay_semantics"] == "cosine_remainder_after_warmup"
    assert resolved.identity()["augmentation_profile"] == "none"
    assert resolved.identity()["augmentation_profile_sha256"] == augmentation_profile("none").sha256


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
        ("action_horizon", 8, "action horizon"),
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
