"""Fail-closed pose adapter for the pinned IsaacLab tiled-camera API."""

from __future__ import annotations

import numpy as np


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def read_camera_world_pose(camera):
    """Read OpenGL/USD world poses from a camera's initialized prim view."""
    view = getattr(camera, "_view", None)
    if (
        not hasattr(camera, "set_world_poses")
        or view is None
        or not hasattr(view, "get_world_poses")
    ):
        raise RuntimeError("Isaac camera does not expose a restorable world pose")
    positions, orientations = view.get_world_poses()
    position_array = _as_numpy(positions)
    orientation_array = _as_numpy(orientations)
    if (
        position_array.ndim != 2
        or position_array.shape[-1] != 3
        or orientation_array.ndim != 2
        or orientation_array.shape[-1] != 4
        or position_array.shape[0] != orientation_array.shape[0]
        or not np.all(np.isfinite(position_array))
        or not np.all(np.isfinite(orientation_array))
    ):
        raise RuntimeError("Isaac camera world pose is unreadable")
    return positions, orientations


def write_camera_world_pose(camera, positions, orientations):
    """Set and verify OpenGL/USD world poses through IsaacLab's public setter."""
    read_camera_world_pose(camera)
    camera.set_world_poses(positions, orientations, convention="opengl")
    actual_positions, actual_orientations = read_camera_world_pose(camera)
    expected_positions = _as_numpy(positions)
    expected_orientations = _as_numpy(orientations)
    actual_position_array = _as_numpy(actual_positions)
    actual_orientation_array = _as_numpy(actual_orientations)
    direct_quaternion = np.all(
        np.isclose(actual_orientation_array, expected_orientations, atol=1e-5),
        axis=-1,
    )
    negated_quaternion = np.all(
        np.isclose(actual_orientation_array, -expected_orientations, atol=1e-5),
        axis=-1,
    )
    if not np.allclose(actual_position_array, expected_positions, atol=1e-5) or not np.all(
        direct_quaternion | negated_quaternion
    ):
        raise RuntimeError("Isaac camera world pose readback did not match request")
    return actual_positions, actual_orientations


__all__ = ["read_camera_world_pose", "write_camera_world_pose"]
