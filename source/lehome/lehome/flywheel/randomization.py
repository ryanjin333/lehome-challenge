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


def sample_randomization(strategy: str, *, seed: int) -> RandomizationRecord:
    """Sample a reproducible record; canonical evaluation deliberately has no values."""
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("randomization seed must be a non-negative integer")
    if strategy == "canonical":
        return RandomizationRecord(strategy, {})
    try:
        bounds = BOUNDS[strategy]
    except KeyError as error:
        raise ValueError(f"unsupported randomization strategy: {strategy}") from error
    rng = np.random.default_rng(seed)
    values: dict[str, object] = {
        "light_intensity_scale": float(rng.uniform(*bounds.light)),
        "camera_translation_m": tuple(float(value) for value in rng.uniform(-bounds.camera_m, bounds.camera_m, 3)),
        "garment_yaw_deg": float(rng.uniform(-bounds.garment_yaw_deg, bounds.garment_yaw_deg)),
        "robot_base_translation_m": tuple(float(value) for value in rng.uniform(-bounds.base_m, bounds.base_m, 3)),
        "table_texture_id": int(rng.integers(1, 101)),
        "garment_display_color": tuple(float(value) for value in rng.uniform(0.65, 1.0, 3)),
    }
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


def validate_randomization_receipt(sampled: dict[str, object], receipt: dict[str, object]) -> None:
    """Require every sampled value and only the defined USD proof metadata."""
    extras = {"table_texture_path", "table_shader_input"}
    if set(receipt) - set(sampled) - extras or set(sampled) - set(receipt):
        raise RuntimeError("flywheel randomization receipt fields do not match sample")
    for key, expected in sampled.items():
        actual = receipt[key]
        if isinstance(expected, (tuple, list)):
            if not np.allclose(actual, expected, atol=1e-5): raise RuntimeError("flywheel randomization readback mismatch")
        elif isinstance(expected, float):
            if not np.isclose(actual, expected, atol=1e-5): raise RuntimeError("flywheel randomization readback mismatch")
        elif actual != expected: raise RuntimeError("flywheel randomization readback mismatch")
    if sampled:
        validate_material_receipt(sampled, receipt)


__all__ = ["BOUNDS", "RandomizationBounds", "sample_randomization"]
