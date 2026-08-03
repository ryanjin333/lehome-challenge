"""Pure schema and adapters for restorable garment simulator snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Mapping, Protocol

import numpy as np


class SnapshotAdapter(Protocol):
    def flywheel_capture_state(self) -> Mapping[str, object]: ...

    def flywheel_restore_state(self, snapshot: "Snapshot") -> None: ...


def _json_round_trip(value: object) -> dict[str, object]:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    restored = json.loads(payload)
    if not isinstance(restored, dict):
        raise ValueError("snapshot mapping must be JSON object")
    return restored


def _finite_tuple(value: object, *, name: str) -> tuple[float, ...]:
    values = np.asarray(value, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite vector")
    return tuple(float(item) for item in values)


def _finite_xyz(value: object, *, name: str) -> tuple[tuple[float, float, float], ...]:
    values = np.asarray(value, dtype=np.float32)
    if values.ndim != 2 or values.shape[1:] != (3,) or not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite N-by-3 particle data")
    return tuple(tuple(float(item) for item in row) for row in values)


@dataclass(frozen=True, slots=True)
class Snapshot:
    schema_version: int
    robot_position: tuple[float, ...]
    robot_velocity: tuple[float, ...]
    cloth_position: tuple[tuple[float, float, float], ...]
    cloth_velocity: tuple[tuple[float, float, float], ...]
    rng_state: dict[str, object]
    garment_name: str
    randomization: dict[str, object]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported snapshot schema version")
        if len(self.robot_position) != 12 or len(self.robot_velocity) != 12:
            raise ValueError("snapshot robot vectors must be 12-D")
        if len(self.cloth_position) != len(self.cloth_velocity) or not self.cloth_position:
            raise ValueError("snapshot cloth position and velocity must be aligned")
        if not self.garment_name:
            raise ValueError("snapshot garment name is required")
        values = (
            *self.robot_position,
            *self.robot_velocity,
            *(value for point in self.cloth_position for value in point),
            *(value for point in self.cloth_velocity for value in point),
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("snapshot values must be finite")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "robot_position": list(self.robot_position),
            "robot_velocity": list(self.robot_velocity),
            "cloth_position": [list(point) for point in self.cloth_position],
            "cloth_velocity": [list(point) for point in self.cloth_velocity],
            "rng_state": _json_round_trip(self.rng_state),
            "garment_name": self.garment_name,
            "randomization": _json_round_trip(self.randomization),
        }


def _read(adapter: object, name: str) -> object:
    getter = getattr(adapter, f"get_{name}", None)
    return getter() if getter is not None else getattr(adapter, name)


def capture_snapshot(adapter: SnapshotAdapter | object, *, randomization: Mapping[str, object]) -> Snapshot:
    """Capture a JSON-safe schema without importing Isaac in pure-test callers."""
    captured = getattr(adapter, "flywheel_capture_state", None)
    state = captured() if captured is not None else {
        "robot_position": _read(adapter, "robot_position"),
        "robot_velocity": _read(adapter, "robot_velocity"),
        "cloth_position": _read(adapter, "cloth_position"),
        "cloth_velocity": _read(adapter, "cloth_velocity"),
        "rng_state": _read(adapter, "rng_state"),
        "garment_name": _read(adapter, "garment_name"),
    }
    return Snapshot(
        schema_version=1,
        robot_position=_finite_tuple(state["robot_position"], name="robot_position"),
        robot_velocity=_finite_tuple(state["robot_velocity"], name="robot_velocity"),
        cloth_position=_finite_xyz(state["cloth_position"], name="cloth_position"),
        cloth_velocity=_finite_xyz(state["cloth_velocity"], name="cloth_velocity"),
        rng_state=_json_round_trip(state["rng_state"]),
        garment_name=str(state["garment_name"]),
        randomization=_json_round_trip(dict(randomization)),
    )


def restore_snapshot(adapter: SnapshotAdapter | object, snapshot: Snapshot) -> None:
    """Restore exactly the state groups captured by :func:`capture_snapshot`."""
    if not isinstance(snapshot, Snapshot):
        raise ValueError("snapshot must be a Snapshot")
    restore = getattr(adapter, "flywheel_restore_state", None)
    if restore is not None:
        restore(snapshot)
        return
    adapter.robot_position = np.asarray(snapshot.robot_position, dtype=np.float32)
    adapter.robot_velocity = np.asarray(snapshot.robot_velocity, dtype=np.float32)
    adapter.cloth_position = np.asarray(snapshot.cloth_position, dtype=np.float32)
    adapter.cloth_velocity = np.asarray(snapshot.cloth_velocity, dtype=np.float32)
    adapter.rng_state = _json_round_trip(snapshot.rng_state)


__all__ = ["Snapshot", "SnapshotAdapter", "capture_snapshot", "restore_snapshot"]
