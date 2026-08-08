from __future__ import annotations

import pytest

from lehome.flywheel.randomization import (
    read_or_author_garment_display_color,
    sample_randomization,
    validate_material_receipt,
)


class FakeUsdAttribute:
    def __init__(self, value=None, *, valid=True, writable=True) -> None:
        self.value = value
        self.valid = valid
        self.writable = writable

    def IsValid(self):
        return self.valid

    def Get(self):
        return self.value

    def Set(self, value):
        if not self.writable:
            return False
        self.value = value
        return True


def test_canonical_changes_nothing_and_mild_is_reproducible() -> None:
    canonical = sample_randomization("canonical", seed=99)
    assert canonical.values == {}

    first = sample_randomization("mild", seed=99)
    second = sample_randomization("mild", seed=99)
    assert first == second
    assert 0.85 <= first.values["light_intensity_scale"] <= 1.15
    assert abs(first.values["camera_translation_m"][0]) <= 0.01


def test_missing_garment_display_color_is_authored_to_a_deterministic_baseline() -> None:
    attribute = FakeUsdAttribute()

    assert read_or_author_garment_display_color(attribute) == [[1.0, 1.0, 1.0]]
    assert attribute.value == [(1.0, 1.0, 1.0)]


@pytest.mark.parametrize(
    "attribute",
    (
        FakeUsdAttribute(valid=False),
        FakeUsdAttribute(writable=False),
        FakeUsdAttribute([]),
    ),
)
def test_garment_display_color_baseline_fails_closed_when_not_readable(attribute) -> None:
    with pytest.raises(RuntimeError, match="displayColor"):
        read_or_author_garment_display_color(attribute)


def test_strong_stays_inside_physical_bounds() -> None:
    result = sample_randomization("strong", seed=123)
    assert abs(result.values["garment_yaw_deg"]) <= 15.0
    assert abs(result.values["robot_base_translation_m"][1]) <= 0.02
    assert 1 <= result.values["table_texture_id"] <= 100
    assert all(0.65 <= value <= 1.0 for value in result.values["garment_display_color"])


def test_material_receipt_requires_exact_usd_readback() -> None:
    sampled = dict(sample_randomization("mild", seed=3).values)
    receipt = {**sampled, "table_texture_path": "/assets/1.png", "table_shader_input": "file"}
    validate_material_receipt(sampled, receipt)
    for key, message in (("table_texture_path", "asset"), ("table_shader_input", "input"), ("garment_display_color", "displayColor")):
        broken = dict(receipt); broken[key] = None
        with pytest.raises(RuntimeError, match=message): validate_material_receipt(sampled, broken)
    broken = dict(receipt); broken["table_texture_id"] += 1
    with pytest.raises(RuntimeError, match="table shader"): validate_material_receipt(sampled, broken)
    broken = dict(receipt); broken["garment_display_color"] = (0.0, 0.0, 0.0)
    with pytest.raises(RuntimeError, match="displayColor"): validate_material_receipt(sampled, broken)
