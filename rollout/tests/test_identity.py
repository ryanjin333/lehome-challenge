from __future__ import annotations

import pytest

from b1k_rollout.identity import (
    BEHAVIOR_REVISION,
    DATASET_REPO,
    GROOT_REVISION,
    MODEL_REPO,
    canonical_json_sha256,
    reject_credential_material,
    require_immutable_commit,
)


def test_pinned_b1k_identities_are_the_approved_private_targets() -> None:
    assert BEHAVIOR_REVISION == "26f2c7ef7b9cf96bd0414f81e1e751e493762779"
    assert GROOT_REVISION == "ace36d935b376fbf25cd56371e23877b95407c40"
    assert MODEL_REPO == "ryanjin333/behavior1k-groot-n17-models"
    assert DATASET_REPO == "ryanjin333/behavior1k-groot-n17-rollouts"


def test_canonical_json_hash_is_independent_of_mapping_order() -> None:
    assert canonical_json_sha256({"b": [2, 3], "a": 1}) == canonical_json_sha256(
        {"a": 1, "b": [2, 3]}
    )


@pytest.mark.parametrize("revision", ["main", "v1.0", "a" * 39, "g" * 40])
def test_immutable_commit_rejects_mutable_or_malformed_refs(revision: str) -> None:
    with pytest.raises(ValueError, match="immutable commit"):
        require_immutable_commit(revision, label="model commit")


@pytest.mark.parametrize(
    "value",
    [
        {"hf_token": "hf_super_secret"},
        {"nested": {"Authorization": "Bearer secret"}},
        {"nested": ["api_key=super-secret"]},
    ],
)
def test_credential_material_is_rejected_recursively(value: object) -> None:
    with pytest.raises(ValueError, match="credential"):
        reject_credential_material(value)


@pytest.mark.parametrize(
    "token",
    [
        "campaign-ghp_" + "a" * 36,
        "gho_" + "a" * 36,
        "ghu_" + "a" * 36,
        "ghs_" + "a" * 36,
        "ghr_" + "a" * 76,
        "github_pat_" + "a" * 82,
        "glpat-" + "a" * 20,
        "AKIA" + "A" * 16,
        "xoxb-12345678901-12345678901-" + "a" * 24,
        "dckr_pat_" + "a" * 24,
        "hf_" + "a" * 34,
        "eyJ" + "a" * 12 + "." + "b" * 12 + "." + "c" * 12,
        "-----BEGIN PRIVATE KEY-----\nprivate\n-----END PRIVATE KEY-----",
    ],
)
def test_common_credentials_are_rejected_when_nested_in_contract_data(token: str) -> None:
    with pytest.raises(ValueError, match="credential"):
        reject_credential_material({"nested": [{"allowed_value": token}]})


def test_exact_github_classic_token_shape_is_rejected_standalone() -> None:
    with pytest.raises(ValueError, match="credential"):
        reject_credential_material("ghp_" + "a" * 36)


@pytest.mark.parametrize("value", ["run-hf_models", "hf_" + "a" * 33, "hf_" + "a" * 35])
def test_benign_or_near_match_hugging_face_values_are_not_credentials(value: str) -> None:
    reject_credential_material(value)
