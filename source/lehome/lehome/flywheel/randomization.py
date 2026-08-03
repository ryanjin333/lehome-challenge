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
    }
    return RandomizationRecord(strategy, values)


__all__ = ["BOUNDS", "RandomizationBounds", "sample_randomization"]
