from dataclasses import FrozenInstanceError, replace

import pytest

from lehome.flywheel.models import (
    ActionSource,
    EpisodeFrame,
    EpisodeIdentity,
    EpisodeOutcome,
    QualityGrade,
    RandomizationRecord,
    RejectionReason,
)


def identity() -> EpisodeIdentity:
    return EpisodeIdentity(
        episode_id="01JTEST0000000000000000000",
        policy_repo="ryanjin333/lehome-groot-n17-policy",
        policy_revision="a" * 40,
        policy_step=12000,
        code_revision="b" * 40,
        asset_revision="c" * 40,
        simulator_version="5.1.0",
        garment_name="Pant_Long_Seen_0",
        category="pant_long",
        release_stage="seen",
        seed=42,
        instruction="fold the garment on the table",
        strategy="canonical",
    )


def test_episode_identity_requires_pinned_artifacts_and_is_immutable() -> None:
    value = identity()
    assert value.policy_step == 12000
    with pytest.raises(ValueError, match="40-character"):
        replace(value, policy_revision="main")
    with pytest.raises(FrozenInstanceError):
        value.seed = 43  # type: ignore[misc]


@pytest.mark.parametrize("strategy", ("mild_geometry", "strong_geometry"))
def test_episode_identity_accepts_stable_geometry_collection_profiles(strategy: str) -> None:
    assert replace(identity(), strategy=strategy).strategy == strategy


@pytest.mark.parametrize("field,value", [("category", "dress"), ("release_stage", "private"), ("strategy", "random")])
def test_episode_identity_rejects_unknown_contract_values(field: str, value: str) -> None:
    with pytest.raises(ValueError):
        replace(identity(), **{field: value})


def test_frame_rejects_nonfinite_or_wrong_dimension_actions() -> None:
    with pytest.raises(ValueError, match="12 finite"):
        EpisodeFrame(
            step=0,
            monotonic_ns=1,
            wall_time_ns=2,
            state=(0.0,) * 12,
            action=(0.0,) * 11,
            action_source=ActionSource.EXPERT,
            reward=0.0,
            success=False,
            segment=1,
        )


def test_frame_rejects_mutable_vector_values() -> None:
    with pytest.raises(ValueError, match="tuples"):
        EpisodeFrame(
            step=0,
            monotonic_ns=1,
            wall_time_ns=2,
            state=[0.0] * 12,  # type: ignore[arg-type]
            action=(0.0,) * 12,
            action_source=ActionSource.EXPERT,
            reward=0.0,
            success=False,
            segment=1,
        )
    with pytest.raises(ValueError, match="12 finite"):
        EpisodeFrame(
            step=0,
            monotonic_ns=1,
            wall_time_ns=2,
            state=(0.0,) * 12,
            action=(0.0,) * 12,
            action_source=ActionSource.EXPERT,
            reward=float("nan"),
            success=False,
            segment=1,
        )


def test_outcome_and_randomization_keep_quality_and_rejections_explicit() -> None:
    outcome = EpisodeOutcome(
        outcome="success",
        accepted=True,
        quality_grade=QualityGrade.A,
        rejection_reasons=(),
        operator_disposition="accept",
    )
    assert outcome.accepted
    with pytest.raises(ValueError, match="accepted"):
        replace(outcome, outcome="timeout")
    record = RandomizationRecord(strategy="mild", values={"light_intensity": 1.1})
    assert record.values == {"light_intensity": 1.1}
    with pytest.raises(ValueError, match="canonical"):
        RandomizationRecord(strategy="canonical", values={"light_intensity": 1.1})
    assert RejectionReason.STALE_EXPERT.value == "stale_expert"
