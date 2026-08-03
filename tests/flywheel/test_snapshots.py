from __future__ import annotations

import numpy as np

from lehome.flywheel.snapshots import Snapshot, capture_snapshot, restore_snapshot


class FakeAdapter:
    robot_position = np.arange(12, dtype=np.float32)
    robot_velocity = np.zeros(12, dtype=np.float32)
    cloth_position = np.arange(30, dtype=np.float32).reshape(10, 3)
    cloth_velocity = np.ones((10, 3), dtype=np.float32)
    rng_state = {"seed": 42, "counter": 7}
    garment_name = "Pant_Long_Seen_0"


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
