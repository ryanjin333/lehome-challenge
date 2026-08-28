"""Deterministic, explicitly bounded rollout-domain randomization samples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import RandomizationRecord


@dataclass(frozen=True, slots=True)
class RandomizationBounds:
    light: tuple[float, float]
    camera_m: float
    garment_yaw_deg: float
    base_m: float


BOUNDS = {
    "mild": RandomizationBounds(light=(0.85, 1.15), camera_m=0.01, garment_yaw_deg=5.0, base_m=0.005),
    "strong": RandomizationBounds(light=(0.65, 1.35), camera_m=0.02, garment_yaw_deg=15.0, base_m=0.02),
}

GEOMETRY_FIELDS = frozenset({
    "light_intensity_scale",
    "camera_translation_m",
    "garment_yaw_deg",
    "robot_base_translation_m",
})
MATERIAL_FIELDS = frozenset({"table_texture_id", "garment_display_color"})
FULL_RANDOMIZATION_FIELDS = GEOMETRY_FIELDS | MATERIAL_FIELDS
GEOMETRY_STRATEGIES = frozenset({"mild_geometry", "strong_geometry"})
VISUAL_ONLY_FIELDS = frozenset({
    "light_intensity_scale",
    "camera_translation_m",
    "table_texture_id",
    "garment_display_color",
})
# Keep the replay contract explicit: no field in a visual-only sample may alter
# cloth state, garment/robot pose, solver behavior, or the action envelope.
PHYSICS_AFFECTING_FIELDS = frozenset({
    "garment_yaw_deg",
    "garment_pose",
    "garment_translation_m",
    "garment_scale",
    "robot_base_translation_m",
    "cloth_geometry",
    "cloth_material",
    "cloth_dynamics",
    "cloth_friction",
    "cloth_stiffness",
    "cloth_damping",
    "solver_iterations",
    "joint_limits",
})


def read_or_author_garment_display_color(attribute) -> list[list[float]]:
    """Return a restorable color, authoring USD's conventional white fallback.

    Challenge garment meshes may declare ``primvars:displayColor`` without an
    authored value.  USD then returns ``None``, which is not snapshot-able.
    Author the neutral white fallback once so canonical and randomized trials
    share a deterministic, readable baseline.
    """
    if attribute is None or not attribute.IsValid():
        raise RuntimeError("flywheel garment displayColor is missing")
    value = attribute.Get()
    if value is None:
        if not attribute.Set([(1.0, 1.0, 1.0)]):
            raise RuntimeError("flywheel garment displayColor cannot be authored")
        value = attribute.Get()
    try:
        normalized = [[float(channel) for channel in color] for color in value]
    except (TypeError, ValueError) as error:
        raise RuntimeError("flywheel garment displayColor is unreadable") from error
    if not normalized or any(len(color) != 3 for color in normalized):
        raise RuntimeError("flywheel garment displayColor is unreadable")
    return normalized


def sample_randomization(strategy: str, *, seed: int) -> RandomizationRecord:
    """Sample a reproducible record; canonical evaluation deliberately has no values."""
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("randomization seed must be a non-negative integer")
    if strategy == "canonical":
        return RandomizationRecord(strategy, {})
    if strategy == "visual_only":
        bounds = BOUNDS["mild"]
        rng = np.random.default_rng(seed)
        return RandomizationRecord(strategy, {
            "light_intensity_scale": float(rng.uniform(*bounds.light)),
            "camera_translation_m": tuple(
                float(value)
                for value in rng.uniform(-bounds.camera_m, bounds.camera_m, 3)
            ),
            "table_texture_id": int(rng.integers(1, 101)),
            "garment_display_color": tuple(
                float(value) for value in rng.uniform(0.65, 1.0, 3)
            ),
        })
    bounds_name = strategy.removesuffix("_geometry")
    try:
        bounds = BOUNDS[bounds_name]
    except KeyError as error:
        raise ValueError(f"unsupported randomization strategy: {strategy}") from error
    rng = np.random.default_rng(seed)
    values: dict[str, object] = {
        "light_intensity_scale": float(rng.uniform(*bounds.light)),
        "camera_translation_m": tuple(float(value) for value in rng.uniform(-bounds.camera_m, bounds.camera_m, 3)),
        "garment_yaw_deg": float(rng.uniform(-bounds.garment_yaw_deg, bounds.garment_yaw_deg)),
        "robot_base_translation_m": tuple(float(value) for value in rng.uniform(-bounds.base_m, bounds.base_m, 3)),
    }
    if strategy not in GEOMETRY_STRATEGIES:
        values.update({
            "table_texture_id": int(rng.integers(1, 101)),
            "garment_display_color": tuple(float(value) for value in rng.uniform(0.65, 1.0, 3)),
        })
    return RandomizationRecord(strategy, values)


def validate_material_receipt(sampled: dict[str, object], receipt: dict[str, object]) -> None:
    """Fail closed unless USD asset/color readback proves sampled material values."""
    if not receipt.get("table_texture_path"):
        raise RuntimeError("flywheel table texture asset is missing")
    if receipt.get("table_shader_input") is None:
        raise RuntimeError("flywheel table shader input is missing")
    if receipt.get("garment_display_color") is None:
        raise RuntimeError("flywheel garment displayColor is missing")
    if receipt.get("table_texture_id") != sampled.get("table_texture_id"):
        raise RuntimeError("flywheel table shader readback mismatch")
    if tuple(receipt["garment_display_color"]) != tuple(sampled.get("garment_display_color", ())):
        raise RuntimeError("flywheel garment displayColor readback mismatch")


def randomization_materials_enabled(sampled: dict[str, object]) -> bool:
    """Validate one supported field profile and report whether materials are included."""

    sampled_fields = set(sampled)
    if sampled_fields == set(GEOMETRY_FIELDS):
        return False
    if sampled_fields == set(FULL_RANDOMIZATION_FIELDS) or sampled_fields == set(VISUAL_ONLY_FIELDS):
        return True
    raise RuntimeError("flywheel randomization sample fields are unsupported")


def validate_randomization_receipt(sampled: dict[str, object], receipt: dict[str, object]) -> None:
    """Require every sampled value and only the defined USD proof metadata."""
    sampled_fields = set(sampled)
    materials_enabled = False if not sampled else randomization_materials_enabled(sampled)
    extras = {"table_texture_path", "table_shader_input"} if materials_enabled else set()
    if set(receipt) - set(sampled) - extras or set(sampled) - set(receipt):
        raise RuntimeError("flywheel randomization receipt fields do not match sample")
    for key, expected in sampled.items():
        actual = receipt[key]
        if isinstance(expected, (tuple, list)):
            if not np.allclose(actual, expected, atol=1e-5): raise RuntimeError("flywheel randomization readback mismatch")
        elif isinstance(expected, float):
            if not np.isclose(actual, expected, atol=1e-5): raise RuntimeError("flywheel randomization readback mismatch")
        elif actual != expected: raise RuntimeError("flywheel randomization readback mismatch")
    if materials_enabled:
        validate_material_receipt(sampled, receipt)


__all__ = [
    "BOUNDS",
    "FULL_RANDOMIZATION_FIELDS",
    "GEOMETRY_FIELDS",
    "GEOMETRY_STRATEGIES",
    "MATERIAL_FIELDS",
    "PHYSICS_AFFECTING_FIELDS",
    "RandomizationBounds",
    "VISUAL_ONLY_FIELDS",
    "read_or_author_garment_display_color",
    "randomization_materials_enabled",
    "sample_randomization",
    "validate_material_receipt",
    "validate_randomization_receipt",
]
