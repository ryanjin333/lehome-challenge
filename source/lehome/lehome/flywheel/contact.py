"""Simulator-geometry evidence for visible robot-to-garment contact."""

from __future__ import annotations

import math

import numpy as np


# The SO-101 gripper frame is about 9.8 cm from the rigid gripper-link origin.
# Stay inside that physical extent so link-origin proximity remains a
# conservative contact witness instead of requiring a cloth particle to enter
# the rigid mesh itself.
_SO101_GRIPPER_LINK_CONTACT_RADIUS_M = 0.08


def visible_contact_from_simulator_geometry(
    cloth_positions: object,
    gripper_positions: object,
    *,
    threshold_m: float = _SO101_GRIPPER_LINK_CONTACT_RADIUS_M,
) -> dict[str, object]:
    """Measure particle-to-gripper proximity from Isaac readback, never joint motion."""
    if not isinstance(threshold_m, (int, float)) or not math.isfinite(threshold_m) or threshold_m <= 0:
        raise ValueError("visible contact threshold must be finite and positive")
    cloth = np.asarray(cloth_positions, dtype=np.float64)
    grippers = np.asarray(gripper_positions, dtype=np.float64)
    if cloth.ndim != 2 or cloth.shape[1:] != (3,) or grippers.ndim != 2 or grippers.shape[1:] != (3,):
        raise ValueError("visible contact requires finite N-by-3 simulator geometry")
    if not len(cloth) or not len(grippers) or not np.isfinite(cloth).all() or not np.isfinite(grippers).all():
        raise ValueError("visible contact requires non-empty finite simulator geometry")
    minimum_distance = float(np.sqrt(np.min(np.sum((cloth[:, None, :] - grippers[None, :, :]) ** 2, axis=-1))))
    return {
        "observed": minimum_distance <= threshold_m,
        "source": "simulator_particle_to_gripper_distance",
        "minimum_distance_m": minimum_distance,
    }
