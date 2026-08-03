from __future__ import annotations

from lehome.flywheel.randomization import sample_randomization


def test_canonical_changes_nothing_and_mild_is_reproducible() -> None:
    canonical = sample_randomization("canonical", seed=99)
    assert canonical.values == {}

    first = sample_randomization("mild", seed=99)
    second = sample_randomization("mild", seed=99)
    assert first == second
    assert 0.85 <= first.values["light_intensity_scale"] <= 1.15
    assert abs(first.values["camera_translation_m"][0]) <= 0.01


def test_strong_stays_inside_physical_bounds() -> None:
    result = sample_randomization("strong", seed=123)
    assert abs(result.values["garment_yaw_deg"]) <= 15.0
    assert abs(result.values["robot_base_translation_m"][1]) <= 0.02
