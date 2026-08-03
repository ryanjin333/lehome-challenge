from __future__ import annotations

import pytest

from lehome_train.flywheel.augmentation import (
    build_sample_sheet_report,
    color_jitter_cli,
    augmentation_profile,
)
from lehome_train.io import canonical_json_sha256


def canonical_holdout_receipt() -> dict[str, object]:
    """Explicit fixture evidence; production code must never mint this gate."""

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


def rehashed(receipt: dict[str, object], **changes: object) -> dict[str, object]:
    amended = {**receipt, **changes}
    amended.pop("receipt_sha256", None)
    amended["receipt_sha256"] = canonical_json_sha256(amended)
    return amended


def test_mild_profile_matches_checked_nvidia_cli_contract() -> None:
    profile = augmentation_profile("mild")

    assert profile.color_jitter == {
        "brightness": 0.20,
        "contrast": 0.20,
        "saturation": 0.20,
        "hue": 0.05,
    }
    assert profile.sha256 == augmentation_profile("mild").sha256
    assert color_jitter_cli(profile) == (
        "--color-jitter-params",
        "brightness",
        "0.2",
        "contrast",
        "0.2",
        "saturation",
        "0.2",
        "hue",
        "0.05",
    )


def test_none_profile_has_no_color_jitter_tokens() -> None:
    assert color_jitter_cli(augmentation_profile("none")) == ()


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown augmentation profile"):
        augmentation_profile("strong")


def test_nvidia_reference_requires_a_passing_canonical_holdout_receipt() -> None:
    with pytest.raises(ValueError, match="canonical-holdout receipt"):
        augmentation_profile("nvidia_reference")

    receipt = canonical_holdout_receipt()
    assert augmentation_profile("nvidia_reference", receipt=receipt).color_jitter == {
        "brightness": 0.30,
        "contrast": 0.40,
        "saturation": 0.50,
        "hue": 0.08,
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda receipt: {**receipt, "passed": False},
        lambda receipt: {**receipt, "regression_detected": True},
        lambda receipt: {**receipt, "candidate_profile_sha256": "0" * 64},
        lambda receipt: {**receipt, "HF_TOKEN": "must-not-persist"},
        lambda receipt: {**receipt, "environment": {"PATH": "/bin"}},
    ],
)
def test_nvidia_reference_rejects_tampered_or_secret_bearing_receipts(mutate) -> None:
    with pytest.raises(ValueError):
        augmentation_profile("nvidia_reference", receipt=mutate(canonical_holdout_receipt()))


@pytest.mark.parametrize(
    "receipt",
    [
        lambda value: rehashed(value, candidate_metric=0.70),
        lambda value: rehashed(value, max_allowed_regression=-0.01),
        lambda value: rehashed(value, non_regression_passed=False),
        lambda value: rehashed(value, dataset_revision="A" * 40),
        lambda value: rehashed(value, policy_revision="not-a-pinned-revision"),
        lambda value: rehashed(value, evaluation_manifest_sha256="C" * 64),
    ],
)
def test_nvidia_reference_rejects_invalid_semantic_evidence_after_rehash(receipt) -> None:
    with pytest.raises(ValueError):
        augmentation_profile("nvidia_reference", receipt=receipt(canonical_holdout_receipt()))


def test_offline_sample_sheet_report_selects_exactly_32_fixed_three_camera_frames() -> None:
    frames = tuple(
        {"episode_id": "episode-0001", "frame_index": index} for index in range(32)
    )
    first = build_sample_sheet_report("mild", seed=20260803, frames=frames)
    second = build_sample_sheet_report("mild", seed=20260803, frames=frames)

    assert first == second
    assert first["profile_sha256"] == augmentation_profile("mild").sha256
    assert first["camera_keys"] == ["top_rgb", "left_rgb", "right_rgb"]
    assert first["frame_count"] == 32
    assert first["render_status"] == "pending_accepted_trainer_image"


@pytest.mark.parametrize(
    "frames",
    [
        tuple({"episode_id": "episode-0001", "frame_index": index} for index in range(31)),
        tuple({"episode_id": "episode-0001", "frame_index": 0} for _ in range(32)),
        tuple(
            {"episode_id": "episode-0001", "frame_index": index, "token": "nope"}
            for index in range(32)
        ),
    ],
)
def test_offline_sample_sheet_report_rejects_invalid_or_secret_bearing_selection(frames) -> None:
    with pytest.raises(ValueError):
        build_sample_sheet_report("mild", seed=20260803, frames=frames)
