from __future__ import annotations

import numpy as np
import pytest

from lehome.flywheel.isaac_camera import read_camera_world_pose, write_camera_world_pose


class FakeCameraView:
    def __init__(self) -> None:
        self.positions = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
        self.orientations = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    def get_world_poses(self):
        return self.positions.copy(), self.orientations.copy()


class FakeTiledCamera:
    def __init__(self) -> None:
        self._view = FakeCameraView()
        self.convention = None

    def set_world_poses(self, positions, orientations, *, convention):
        self._view.positions = np.asarray(positions, dtype=np.float32)
        # Exercise sign-invariant quaternion readback.
        self._view.orientations = -np.asarray(orientations, dtype=np.float32)
        self.convention = convention


def test_tiled_camera_pose_uses_the_pinned_view_api_and_opengl_convention() -> None:
    camera = FakeTiledCamera()
    positions, orientations = read_camera_world_pose(camera)
    requested_positions = positions + np.asarray([[0.1, 0.0, -0.1]], dtype=np.float32)

    actual_positions, actual_orientations = write_camera_world_pose(
        camera,
        requested_positions,
        orientations,
    )

    assert camera.convention == "opengl"
    assert np.allclose(actual_positions, requested_positions)
    assert np.allclose(actual_orientations, -orientations)


@pytest.mark.parametrize("camera", (object(), type("NoGetter", (), {"_view": object(), "set_world_poses": lambda *args, **kwargs: None})()))
def test_tiled_camera_pose_fails_closed_without_a_restorable_view(camera) -> None:
    with pytest.raises(RuntimeError, match="camera.*pose"):
        read_camera_world_pose(camera)


def test_tiled_camera_pose_write_fails_closed_without_a_restorable_view() -> None:
    with pytest.raises(RuntimeError, match="camera.*pose"):
        write_camera_world_pose(
            object(),
            np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
            np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        )
