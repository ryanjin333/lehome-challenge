from __future__ import annotations

import numpy as np
import pytest
import lehome.flywheel.randomization as randomization

from lehome.flywheel.randomization import (
    read_or_author_garment_display_color,
    sample_randomization,
    validate_material_receipt,
    validate_randomization_receipt,
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


def test_geometry_profile_is_reproducible_without_unstable_material_fields() -> None:
    first = sample_randomization("mild_geometry", seed=139)
    second = sample_randomization("mild_geometry", seed=139)

    assert first == second
    assert set(first.values) == {
        "light_intensity_scale",
        "camera_translation_m",
        "garment_yaw_deg",
        "robot_base_translation_m",
    }
    assert "table_texture_id" not in first.values
    assert "garment_display_color" not in first.values

    receipt = dict(first.values)
    validate_randomization_receipt(dict(first.values), receipt)


def test_visual_only_profile_is_deterministic_and_strictly_physics_invariant() -> None:
    first = sample_randomization("visual_only", seed=140)
    second = sample_randomization("visual_only", seed=140)

    assert first == second
    assert set(first.values) == set(randomization.VISUAL_ONLY_FIELDS)
    assert set(first.values).isdisjoint(randomization.PHYSICS_AFFECTING_FIELDS)
    assert {
        "garment_yaw_deg", "garment_pose", "garment_scale",
        "robot_base_translation_m", "cloth_geometry", "cloth_material",
        "cloth_dynamics", "cloth_friction", "cloth_stiffness",
        "cloth_damping", "solver_iterations", "joint_limits",
    }.isdisjoint(first.values)

    for field in first.values:
        missing = dict(first.values)
        missing.pop(field)
        with pytest.raises(RuntimeError, match="fields"):
            validate_randomization_receipt(dict(first.values), missing)
    validate_randomization_receipt(dict(first.values), {
        **dict(first.values),
        "table_texture_path": "/assets/1.png",
        "table_shader_input": "file",
    })


def test_visual_only_orchestration_applies_only_visual_mutations_before_verification() -> None:
    orchestrate = getattr(randomization, "orchestrate_visual_only_replay", None)
    apply_visual = getattr(randomization, "apply_visual_mutations", None)
    assert callable(orchestrate)
    assert callable(apply_visual)

    events: list[str] = []
    sampled = dict(sample_randomization("visual_only", seed=140).values)

    class Adapter:
        def set_table_texture(self, texture_id: int) -> tuple[str, str]:
            assert texture_id == sampled["table_texture_id"]
            events.append("texture")
            return f"/assets/{texture_id}.png", "file"

        def set_garment_display_color(self, color: tuple[float, float, float]) -> tuple[float, float, float]:
            assert color == sampled["garment_display_color"]
            events.append("color")
            return color

        def scale_light_intensity(self, scale: float) -> float:
            assert scale == sampled["light_intensity_scale"]
            events.append("light")
            return scale

        def translate_cameras(self, translation: tuple[float, float, float]) -> tuple[float, float, float]:
            assert translation == sampled["camera_translation_m"]
            events.append("camera")
            return translation

        def __getattr__(self, name):
            raise AssertionError(f"visual replay reached a physical mutation: {name}")

    receipt = orchestrate(
        sampled,
        capture_state=lambda: events.append("capture") or {"cloth": "exact"},
        apply_visual_mutations=lambda: apply_visual(sampled, adapter=Adapter()),
        verify_state=lambda state: events.append(f"verify:{state['cloth']}"),
    )

    assert receipt["table_texture_id"] == sampled["table_texture_id"]
    assert events == ["capture", "texture", "color", "light", "camera", "verify:exact"]


def test_visual_only_drift_raises_typed_fidelity_failure_with_bounded_readback() -> None:
    from lehome.flywheel.persistent_worker import FidelityFailureError
    verify_state = getattr(randomization, "verify_visual_replay_state", None)
    assert callable(verify_state)
    pre_randomization = (
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        np.array([0.0, 0.0, 0.67, 0.0, 0.0, 90.0], dtype=np.float32),
    )
    observed = (
        np.array([[0.25, 0.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 0.5, 0.0]], dtype=np.float32),
        np.array([0.0, 0.0, 0.67, 0.0, 0.0, 92.0], dtype=np.float32),
    )

    with pytest.raises(FidelityFailureError) as error:
        verify_state(pre_randomization, observed)
    assert error.value.fidelity_code == "safety_failure"
    assert error.value.diagnostic == {
        "stage": "reset_write_readback",
        "write_readback": {
            "max_position_delta_m": 0.25,
            "max_velocity_delta_mps": 0.5,
        },
        "visual_replay": {
            "max_cloth_position_delta_m": 0.25,
            "max_cloth_velocity_delta_mps": 0.5,
            "max_garment_translation_delta_m": 0.0,
            "max_garment_rotation_delta_deg": 2.0,
        },
    }


def test_geometry_receipt_still_fails_closed_on_missing_or_extra_readback() -> None:
    sampled = dict(sample_randomization("strong_geometry", seed=149).values)
    missing = dict(sampled)
    missing.pop("garment_yaw_deg")
    with pytest.raises(RuntimeError, match="fields"):
        validate_randomization_receipt(sampled, missing)

    extra = {**sampled, "table_texture_path": "/assets/1.png"}
    with pytest.raises(RuntimeError, match="fields"):
        validate_randomization_receipt(sampled, extra)


def test_randomization_field_profile_controls_material_application() -> None:
    assert hasattr(randomization, "randomization_materials_enabled")
    geometry = dict(sample_randomization("mild_geometry", seed=139).values)
    full = dict(sample_randomization("mild", seed=139).values)

    assert randomization.randomization_materials_enabled(geometry) is False
    assert randomization.randomization_materials_enabled(full) is True

    partial_material = {**geometry, "table_texture_id": 7}
    with pytest.raises(RuntimeError, match="fields"):
        randomization.randomization_materials_enabled(partial_material)


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
