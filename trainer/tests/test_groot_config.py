from __future__ import annotations

import pytest

from lehome_train.constants import MODEL_REVISION
from lehome_train.groot.config import FineTuneLaunchConfig


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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("num_gpus", 0, "exactly one GPU"),
        ("num_gpus", 2, "exactly one GPU"),
        ("global_batch_size", 32, "physical batch"),
        ("gradient_accumulation_steps", 2, "gradient accumulation"),
        ("action_horizon", 8, "action horizon"),
        ("warmup_ratio", 0.0, "warmup_ratio"),
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
