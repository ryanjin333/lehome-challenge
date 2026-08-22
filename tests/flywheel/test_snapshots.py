from __future__ import annotations

import numpy as np

from lehome.flywheel.snapshots import Snapshot, canonical_reset_hash, capture_snapshot, restore_snapshot


class FakeAdapter:
    robot_position = np.arange(12, dtype=np.float32)
    robot_velocity = np.zeros(12, dtype=np.float32)
    cloth_position = np.arange(30, dtype=np.float32).reshape(10, 3)
    cloth_velocity = np.ones((10, 3), dtype=np.float32)
    rng_state = {"seed": 42, "counter": 7}
    garment_name = "Pant_Long_Seen_0"


class SceneStateAdapter(FakeAdapter):
    def __init__(self) -> None:
        self.robot_position = np.arange(12, dtype=np.float32)
        self.robot_velocity = np.zeros(12, dtype=np.float32)
        self.cloth_position = np.arange(30, dtype=np.float32).reshape(10, 3)
        self.cloth_velocity = np.ones((10, 3), dtype=np.float32)
        self.rng_state = {"seed": 42, "counter": 7}
        self.garment_name = "Pant_Long_Seen_0"
        self.scene_state = {
            "camera_world_poses": [
                {"position": [1.0, 2.0, 3.0], "orientation": [1.0, 0.0, 0.0, 0.0]},
            ] * 3,
            "robot_root_poses": [
                {"position": [4.0, 5.0, 6.0], "orientation": [0.0, 1.0, 0.0, 0.0]},
            ] * 2,
            "light_intensity": 1200.0,
            "light_color": [0.75, 0.75, 0.75],
            "table_texture_path": "/assets/1.png",
            "table_shader_input": "file",
            "garment_display_color": [[0.8, 0.7, 0.6]],
            "garment_reset_pose": [0.0] * 6,
        }

    def flywheel_capture_state(self):
        return {
            "robot_position": self.robot_position,
            "robot_velocity": self.robot_velocity,
            "cloth_position": self.cloth_position,
            "cloth_velocity": self.cloth_velocity,
            "rng_state": self.rng_state,
            "garment_name": self.garment_name,
            "scene_state": self.scene_state,
        }

    def flywheel_restore_state(self, snapshot):
        self.robot_position = np.asarray(snapshot.robot_position, dtype=np.float32)
        self.robot_velocity = np.asarray(snapshot.robot_velocity, dtype=np.float32)
        self.cloth_position = np.asarray(snapshot.cloth_position, dtype=np.float32)
        self.cloth_velocity = np.asarray(snapshot.cloth_velocity, dtype=np.float32)
        self.rng_state = snapshot.rng_state
        self.scene_state = snapshot.scene_state


class PhysicsClothAdapter(SceneStateAdapter):
    def flywheel_capture_state(self):
        return {
            **super().flywheel_capture_state(),
            "cloth_state_authority": "physx_cloth_view_world_v1",
        }


def test_snapshot_round_trip_restores_every_state_group() -> None:
    env = FakeAdapter()
    snapshot = capture_snapshot(env, randomization={"strategy": "canonical"})
    assert isinstance(snapshot, Snapshot)

    env.robot_position[:] = -1
    env.cloth_position[:] = -1
    env.rng_state = {"seed": 0}
    restore_snapshot(env, snapshot)

    assert env.robot_position.tolist() == list(range(12))
    assert env.cloth_position.shape == (10, 3)
    assert env.rng_state == {"seed": 42, "counter": 7}
    assert snapshot.garment_name == "Pant_Long_Seen_0"


def test_snapshot_round_trip_restores_mutated_scene_state() -> None:
    env = SceneStateAdapter()
    snapshot = capture_snapshot(env, randomization={"strategy": "mild"})
    env.scene_state = {"camera_positions": [[99.0, 99.0, 99.0]]}
    restore_snapshot(env, snapshot)
    assert env.scene_state == {
        "camera_world_poses": [
            {"position": [1.0, 2.0, 3.0], "orientation": [1.0, 0.0, 0.0, 0.0]},
        ] * 3,
        "robot_root_poses": [
            {"position": [4.0, 5.0, 6.0], "orientation": [0.0, 1.0, 0.0, 0.0]},
        ] * 2,
        "light_intensity": 1200.0,
        "light_color": [0.75, 0.75, 0.75],
        "table_texture_path": "/assets/1.png",
        "table_shader_input": "file",
        "garment_display_color": [[0.8, 0.7, 0.6]],
        "garment_reset_pose": [0.0] * 6,
    }


def test_canonical_reset_hash_is_stable_and_changes_with_the_reset_state() -> None:
    env = FakeAdapter()
    first = capture_snapshot(env, randomization={"strategy": "canonical"})
    assert canonical_reset_hash(first) == canonical_reset_hash(first)

    env.cloth_position[0, 0] += 0.01
    changed = capture_snapshot(env, randomization={"strategy": "canonical"})
    assert canonical_reset_hash(changed) != canonical_reset_hash(first)


def test_physics_cloth_capture_emits_a_versioned_authority_contract() -> None:
    snapshot = capture_snapshot(
        PhysicsClothAdapter(), randomization={"strategy": "canonical"}
    )

    assert snapshot.schema_version == 2
    assert snapshot.cloth_state_authority == "physx_cloth_view_world_v1"
    assert snapshot.to_dict()["cloth_state_authority"] == "physx_cloth_view_world_v1"


def test_version_two_snapshot_rejects_nonphysics_cloth_authority() -> None:
    import pytest

    with pytest.raises(ValueError, match="cloth state authority"):
        Snapshot(
            schema_version=2,
            robot_position=(0.0,) * 12,
            robot_velocity=(0.0,) * 12,
            cloth_position=((0.0, 0.0, 0.0),),
            cloth_velocity=((0.0, 0.0, 0.0),),
            rng_state={},
            garment_name="Pant_Long_Seen_0",
            randomization={"strategy": "canonical"},
            cloth_state_authority="usd_authored_points_v1",
        )


def test_version_three_snapshot_marks_legacy_usd_local_points_explicitly() -> None:
    from lehome.flywheel.snapshots import LEGACY_USD_LOCAL_CLOTH_AUTHORITY

    snapshot = Snapshot(
        schema_version=3,
        robot_position=(0.0,) * 12,
        robot_velocity=(0.0,) * 12,
        cloth_position=((0.0, 0.0, 0.0),),
        cloth_velocity=((0.0, 0.0, 0.0),),
        rng_state={},
        garment_name="Pant_Short_Seen_5",
        randomization={"strategy": "canonical"},
        scene_state={"garment_reset_pose": [0.0, 0.0, 0.67, 0.0, 0.0, 90.0]},
        cloth_state_authority=LEGACY_USD_LOCAL_CLOTH_AUTHORITY,
    )

    assert snapshot.to_dict()["cloth_state_authority"] == "usd_local_points_v1"
