"""Pure schema and adapters for restorable garment simulator snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Mapping, Protocol

import numpy as np


PHYSX_CLOTH_STATE_AUTHORITY = "physx_cloth_view_world_v1"
LEGACY_USD_LOCAL_CLOTH_AUTHORITY = "usd_local_points_v1"


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
    # Exact USD/camera/root properties mutated by flywheel randomization.
    # Optional for pre-randomization snapshots produced by older callers.
    scene_state: dict[str, object] = field(default_factory=dict)
    # Version 2 snapshots are captured from the live PhysX cloth view in world
    # coordinates. Version 1 remains the byte-level legacy input. Version 3 is
    # an in-memory upgrade that labels authenticated legacy CPU USD-local
    # points before a backend-aware restore; neither legacy form is admitted by
    # controlled-recovery v3.
    cloth_state_authority: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version not in (1, 2, 3):
            raise ValueError("unsupported snapshot schema version")
        if self.schema_version == 1 and self.cloth_state_authority is not None:
            raise ValueError("snapshot v1 cannot declare a cloth state authority")
        if self.schema_version == 2 and self.cloth_state_authority != PHYSX_CLOTH_STATE_AUTHORITY:
            raise ValueError("snapshot v2 requires the live PhysX cloth state authority")
        if self.schema_version == 3 and self.cloth_state_authority != LEGACY_USD_LOCAL_CLOTH_AUTHORITY:
            raise ValueError("snapshot v3 requires the legacy USD-local cloth state authority")
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
        payload = {
            "schema_version": self.schema_version,
            "robot_position": list(self.robot_position),
            "robot_velocity": list(self.robot_velocity),
            "cloth_position": [list(point) for point in self.cloth_position],
            "cloth_velocity": [list(point) for point in self.cloth_velocity],
            "rng_state": _json_round_trip(self.rng_state),
            "garment_name": self.garment_name,
            "randomization": _json_round_trip(self.randomization),
            "scene_state": _json_round_trip(self.scene_state),
        }
        if self.schema_version in (2, 3):
            payload["cloth_state_authority"] = self.cloth_state_authority
        return payload


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
    authority = state.get("cloth_state_authority")
    if authority is None:
        schema_version = 1
    elif authority == PHYSX_CLOTH_STATE_AUTHORITY:
        schema_version = 2
    elif authority == LEGACY_USD_LOCAL_CLOTH_AUTHORITY:
        schema_version = 3
    else:
        raise ValueError("snapshot capture has an unsupported cloth state authority")
    return Snapshot(
        schema_version=schema_version,
        robot_position=_finite_tuple(state["robot_position"], name="robot_position"),
        robot_velocity=_finite_tuple(state["robot_velocity"], name="robot_velocity"),
        cloth_position=_finite_xyz(state["cloth_position"], name="cloth_position"),
        cloth_velocity=_finite_xyz(state["cloth_velocity"], name="cloth_velocity"),
        rng_state=_json_round_trip(state["rng_state"]),
        garment_name=str(state["garment_name"]),
        randomization=_json_round_trip(dict(randomization)),
        scene_state=_json_round_trip(state.get("scene_state", {})),
        cloth_state_authority=None if authority is None else str(authority),
    )


def canonical_reset_hash(snapshot: Snapshot) -> str:
    """Hash the complete reset state after simulator readback in one stable form."""
    if not isinstance(snapshot, Snapshot):
        raise ValueError("reset hash requires a validated Snapshot")
    payload = json.dumps(snapshot.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
    if snapshot.scene_state:
        adapter.scene_state = _json_round_trip(snapshot.scene_state)


__all__ = [
    "LEGACY_USD_LOCAL_CLOTH_AUTHORITY", "PHYSX_CLOTH_STATE_AUTHORITY",
    "Snapshot", "SnapshotAdapter", "canonical_reset_hash", "capture_snapshot", "restore_snapshot",
]
