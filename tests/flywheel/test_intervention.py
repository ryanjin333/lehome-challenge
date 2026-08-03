from __future__ import annotations

import pytest

from lehome.flywheel.intervention import InterventionController, TransitionError
from lehome.flywheel.quality import AttemptStats, QualityThresholds, grade_attempt


def thresholds() -> QualityThresholds:
    return QualityThresholds(
        dataset_revision="a" * 40,
        dataset_sha256="b" * 64,
        clean_velocity_p95=1.0,
        clean_acceleration_p95=1.0,
        clean_jitter_p95=1.0,
        max_velocity_p95=2.0,
        max_acceleration_p95=2.0,
        max_jitter_p95=2.0,
        allowed_stale_samples=0,
        allowed_unsafe_commands=0,
    )


def test_dagger_takeover_is_one_way_and_clears_policy_queue() -> None:
    controller = InterventionController(mode="dagger", sync_tolerance_rad=0.08)
    controller.start_policy()
    controller.request_takeover()
    with pytest.raises(TransitionError, match="synchronization"):
        controller.accept_expert(current_robot=(0.0,) * 12, leader_command=(0.2,) * 12)
    controller.accept_expert(current_robot=(0.0,) * 12, leader_command=(0.01,) * 12)
    assert controller.state == "expert"
    assert controller.policy_queue_clear_requested is True
    assert controller.action_source == "expert"
    with pytest.raises(TransitionError, match="one-way"):
        controller.start_policy()


def test_practice_never_exports_and_manual_accept_cannot_override_rejection() -> None:
    practice = InterventionController(mode="practice")
    practice.start_expert()
    assert practice.accept(grade_attempt(AttemptStats(official_success=True), thresholds())) is False
    assert practice.state == "diagnostic"

    expert = InterventionController(mode="expert")
    expert.start_expert()
    rejected = grade_attempt(AttemptStats(official_success=False), thresholds())
    assert expert.accept(rejected) is False
    assert expert.state == "diagnostic"
