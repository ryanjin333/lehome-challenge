from __future__ import annotations

import pytest

from lehome_train.schedule import (
    CHECKPOINT_SAMPLE_PRESENTATIONS,
    TOTAL_SAMPLE_PRESENTATIONS,
    ExposureSchedule,
    optimizer_steps_for_presentations,
)


@pytest.mark.parametrize(
    ("batch_size", "expected_steps"),
    [(64, 12_000), (32, 24_000), (16, 48_000), (8, 96_000)],
)
def test_fixed_exposure_derives_optimizer_steps(
    batch_size: int,
    expected_steps: int,
) -> None:
    schedule = ExposureSchedule(physical_batch_size=batch_size)

    assert schedule.total_optimizer_steps == expected_steps
    assert schedule.total_optimizer_steps * batch_size == TOTAL_SAMPLE_PRESENTATIONS
    assert schedule.checkpoint_interval_steps == CHECKPOINT_SAMPLE_PRESENTATIONS // batch_size
    assert schedule.checkpoint_steps[-1] == expected_steps
    assert len(schedule.checkpoint_steps) == 12


def test_optimizer_step_derivation_refuses_fractional_steps() -> None:
    with pytest.raises(ValueError, match="integral optimizer steps"):
        optimizer_steps_for_presentations(TOTAL_SAMPLE_PRESENTATIONS, 7)

    with pytest.raises(ValueError, match="integral optimizer steps"):
        ExposureSchedule(physical_batch_size=7)


def test_learning_rate_shape_is_fractional_across_selected_batch_sizes() -> None:
    batch_64 = ExposureSchedule(physical_batch_size=64, warmup_fraction=0.1)
    batch_8 = ExposureSchedule(physical_batch_size=8, warmup_fraction=0.1)

    assert batch_64.learning_rate_multiplier(0) == 0.0
    assert batch_64.learning_rate_multiplier(1_200) == pytest.approx(1.0)
    assert batch_8.learning_rate_multiplier(9_600) == pytest.approx(1.0)
    assert batch_64.learning_rate_multiplier(6_000) == pytest.approx(
        batch_8.learning_rate_multiplier(48_000)
    )
    assert batch_64.learning_rate_multiplier(12_000) == pytest.approx(0.0)


def test_schedule_rejects_non_fractional_warmup_or_non_integral_checkpoint_interval() -> None:
    with pytest.raises(ValueError, match="warmup_fraction"):
        ExposureSchedule(physical_batch_size=64, warmup_fraction=1.0)
    with pytest.raises(ValueError, match="checkpoint interval"):
        ExposureSchedule(
            physical_batch_size=64,
            checkpoint_sample_presentations=64_001,
        )
