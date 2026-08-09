from __future__ import annotations

import json

import pytest

from lehome_train.constants import (
    BEHAVIOR_1K_CHECKPOINT_BUCKET,
    BEHAVIOR_1K_FINAL_MODEL_REPOSITORY,
    ISAAC_GROOT_REPOSITORY,
    ISAAC_GROOT_REVISION,
    MODEL_REVISION,
)
from lehome_train.b1k.contracts import RunContract


BASE = {
    "HF_TOKEN": "not-a-real-token",
    "HF_DATASET_REPO": "behavior-1k/2026-challenge-demos",
    "HF_MODEL_REPO": BEHAVIOR_1K_FINAL_MODEL_REPOSITORY,
    "HF_CHECKPOINT_BUCKET": BEHAVIOR_1K_CHECKPOINT_BUCKET,
    "DATASET_REVISION": "4f50b44796641a4d526a19d9aeadc8aa51e2f2c2",
    "GROOT_REVISION": ISAAC_GROOT_REVISION,
    "CONTAINER_DIGEST": "sha256:" + "a" * 64,
    "RUN_ID": "b1k-20260803-001",
    "CYCLE_ID": "cycle-000",
    "TRAIN_STEPS": "15000",
    "SAVE_STEPS": "1000",
    "CHECKPOINT_KEEP": "2",
    "RESUME_POLICY": "auto",
    "AUTO_DESTROY": "0",
    "BASE_MODEL_REVISION": MODEL_REVISION,
    "TASK_MANIFEST_SHA256": "b" * 64,
    "MODALITY_SHA256": "ca3a2e406472650bee9439ed81a3f4a1531b6fe689cdc1b348d0f260a208e641",
    "STATS_SHA256": "c" * 64,
    "LAUNCH_PLAN_ID": "b1k-gpu1-effective-batch256",
    "WORLD_SIZE": "1",
    "LEARNING_RATE": "0.0001",
    "EXPERIMENT_NAME": "b1k-20260803-001",
    "LAUNCH_ARGUMENTS_SHA256": "d" * 64,
    "COSMOS_REPOSITORY": "nvidia/Cosmos-Reason2-2B",
    "COSMOS_REVISION": "9ce19a195e423419c349abfc86fd07178b230561",
}


def test_contract_is_secret_free_and_exact() -> None:
    contract = RunContract.from_environment(BASE)

    assert contract.to_dict() == {
        "auto_destroy": False,
        "checkpoint_keep": 2,
        "container_digest": "sha256:" + "a" * 64,
        "cycle_id": "cycle-000",
        "dataset_repo": "behavior-1k/2026-challenge-demos",
        "dataset_revision": "4f50b44796641a4d526a19d9aeadc8aa51e2f2c2",
        "groot_repository": ISAAC_GROOT_REPOSITORY,
        "base_model_revision": MODEL_REVISION,
        "task_manifest_sha256": "b" * 64,
        "modality_sha256": "ca3a2e406472650bee9439ed81a3f4a1531b6fe689cdc1b348d0f260a208e641",
        "stats_sha256": "c" * 64,
        "launch_plan_id": "b1k-gpu1-effective-batch256",
        "world_size": 1,
        "learning_rate": 0.0001,
        "experiment_name": "b1k-20260803-001",
        "launch_arguments_sha256": "d" * 64,
        "physical_batch_size": 64,
        "global_batch_size": 64,
        "gradient_accumulation_steps": 4,
        "effective_global_batch_size": 256,
        "weight_decay": 1e-5,
        "warmup_ratio": 0.05,
        "cosmos_repository": "nvidia/Cosmos-Reason2-2B",
        "cosmos_revision": "9ce19a195e423419c349abfc86fd07178b230561",
        "groot_revision": ISAAC_GROOT_REVISION,
        "manifest_path": None,
        "model_repo": BEHAVIOR_1K_FINAL_MODEL_REPOSITORY,
        "parent_cycle_id": None,
        "checkpoint_bucket": BEHAVIOR_1K_CHECKPOINT_BUCKET,
        "resume_policy": "auto",
        "run_id": "b1k-20260803-001",
        "save_steps": 1_000,
        "train_steps": 15_000,
    }
    serialized = json.dumps(contract.to_dict())
    assert "token" not in serialized.lower()
    assert "not-a-real-token" not in serialized


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("HF_TOKEN", "", "HF_TOKEN"),
        ("DATASET_REVISION", "v3.0", "DATASET_REVISION"),
        ("DATASET_REVISION", "main", "DATASET_REVISION"),
        ("GROOT_REVISION", "a" * 40, "GROOT_REVISION"),
        ("CONTAINER_DIGEST", "sha256:" + "a" * 63, "CONTAINER_DIGEST"),
        ("RUN_ID", "unsafe/path", "RUN_ID"),
        ("CYCLE_ID", "Cycle-000", "CYCLE_ID"),
        ("TRAIN_STEPS", "14999", "15,000"),
        ("SAVE_STEPS", "999", "1,000"),
        ("CHECKPOINT_KEEP", "1", "exactly 2"),
        ("RESUME_POLICY", "best-effort", "RESUME_POLICY"),
        ("AUTO_DESTROY", "yes", "AUTO_DESTROY"),
        ("CHECKPOINT_KEEP", "3", "exactly 2"),
        ("BASE_MODEL_REVISION", "a" * 40, "BASE_MODEL_REVISION"),
        ("TASK_MANIFEST_SHA256", "not-a-hash", "TASK_MANIFEST_SHA256"),
        ("MODALITY_SHA256", "a" * 64, "MODALITY_SHA256"),
        ("WORLD_SIZE", "5", "WORLD_SIZE"),
    ],
)
def test_contract_rejects_invalid_initial_run_values(
    key: str, value: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        RunContract.from_environment(dict(BASE, **{key: value}))


def test_contract_uses_safe_defaults_without_mutating_environment() -> None:
    contract = RunContract.from_environment(
        {key: value for key, value in BASE.items() if key not in {"CHECKPOINT_KEEP", "RESUME_POLICY", "AUTO_DESTROY"}}
    )

    assert contract.checkpoint_keep == 2
    assert contract.resume_policy == "auto"
    assert contract.auto_destroy is False


def test_contract_rejects_auto_destroy_enabled() -> None:
    values = dict(BASE)
    values["AUTO_DESTROY"] = "1"
    with pytest.raises(ValueError, match="AUTO_DESTROY"): RunContract.from_environment(values)


@pytest.mark.parametrize(
    ("key", "legacy_target"),
    [
        ("HF_MODEL_REPO", "ryanjin333/lehome-groot-n17-models"),
        ("HF_CHECKPOINT_BUCKET", "ryanjin333/lehome-groot-n17-checkpoints"),
    ],
)
def test_contract_rejects_every_legacy_lehome_publication_target(
    key: str, legacy_target: str
) -> None:
    with pytest.raises(ValueError, match=key):
        RunContract.from_environment(dict(BASE, **{key: legacy_target}))
