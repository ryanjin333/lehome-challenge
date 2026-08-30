from __future__ import annotations

import pytest

from b1k_rollout.contracts import RolloutContract
from b1k_rollout.identity import (
    BEHAVIOR_REVISION,
    DATASET_REPO,
    GROOT_REVISION,
    MODEL_REPO,
)


def _contract_values(**overrides: str) -> dict[str, str]:
    values = {
        "behavior_revision": BEHAVIOR_REVISION,
        "groot_revision": GROOT_REVISION,
        "model_repository": MODEL_REPO,
        "model_commit": "a" * 40,
        "dataset_repository": DATASET_REPO,
        "image_digest": "sha256:" + "b" * 64,
        "run_id": "run-001",
        "cycle_id": "cycle-001",
        "campaign_id": "campaign-001",
        "evaluator_mode": "train",
        "task_manifest_sha256": "c" * 64,
        "checkpoint_artifact_sha256": "d" * 64,
        "auto_destroy": "0",
    }
    values.update(overrides)
    return values


def test_contract_is_frozen_secret_free_and_content_addressed() -> None:
    contract = RolloutContract.from_mapping(_contract_values())

    assert contract.to_dict() == _contract_values()
    assert contract.identity == "513be039cad5c67f636bbaa52a9ae97241e02d13c771e0e96f828c7fafedd938"
    assert contract.identity != RolloutContract.from_mapping(
        _contract_values(run_id="run-002")
    ).identity
    with pytest.raises(AttributeError):
        contract.run_id = "different"  # type: ignore[misc]


@pytest.mark.parametrize(
    "token",
    [
        "ghp_" + "a" * 36,
        "gho_" + "a" * 36,
        "ghu_" + "a" * 36,
        "ghs_" + "a" * 36,
        "ghr_" + "a" * 76,
    ],
)
def test_contract_rejects_github_token_shapes_in_run_id(token: str) -> None:
    with pytest.raises(ValueError, match="credential"):
        RolloutContract.from_mapping(_contract_values(run_id=token))


def test_contract_allows_a_benign_hf_like_run_id() -> None:
    contract = RolloutContract.from_mapping(_contract_values(run_id="run-hf_models"))

    assert contract.run_id == "run-hf_models"


def test_contract_rejects_a_jwt_embedded_in_an_otherwise_valid_cycle_id() -> None:
    token = "eyJ" + "a" * 12 + "." + "b" * 12 + "." + "c" * 12

    with pytest.raises(ValueError, match="credential"):
        RolloutContract.from_mapping(_contract_values(cycle_id=token))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_commit", "main", "immutable commit"),
        ("image_digest", "latest", "image digest"),
        ("evaluator_mode", "hidden_test", "evaluator mode"),
        ("auto_destroy", "1", "AUTO_DESTROY"),
        ("model_repository", "ryanjin333/lehome-groot-n17-models", "LeHome"),
        ("dataset_repository", "ryanjin333/lehome-groot-n17-rollouts", "LeHome"),
    ],
)
def test_contract_rejects_mutable_or_forbidden_values(
    field: str, value: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        RolloutContract.from_mapping(_contract_values(**{field: value}))


@pytest.mark.parametrize("field", ["run_id", "cycle_id", "campaign_id", "task_manifest_sha256", "checkpoint_artifact_sha256"])
def test_contract_requires_each_campaign_identity_field(field: str) -> None:
    values = _contract_values()
    del values[field]
    with pytest.raises(ValueError, match=field):
        RolloutContract.from_mapping(values)


@pytest.mark.parametrize("forbidden_field", ["tasks", "task_names", "hf_token", "authorization"])
def test_contract_rejects_arbitrary_task_lists_and_credentials(
    forbidden_field: str,
) -> None:
    values: dict[str, object] = _contract_values()
    values[forbidden_field] = ["arbitrary-task"] if forbidden_field == "tasks" else "secret"
    with pytest.raises(ValueError, match="(task lists|credential|unknown)"):
        RolloutContract.from_mapping(values)
