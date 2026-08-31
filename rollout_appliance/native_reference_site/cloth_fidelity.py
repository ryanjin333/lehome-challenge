"""Read-only cloth-fidelity monitor for the pinned native evaluator."""

from __future__ import annotations

import atexit
import hashlib
import json
import math
import os
from pathlib import Path
from types import MethodType
from typing import Any, Mapping, Sequence

import numpy as np


ZERO_SHA256 = "0" * 64
_HEALTH_KEYS = {
    "healthy",
    "sample_count",
    "max_position_m",
    "max_extent_m",
    "max_velocity_mps",
    "max_position_limit_m",
    "max_extent_limit_m",
    "max_velocity_limit_mps",
    "missing_cloth",
    "cloth_flight",
    "nonfinite_cloth_state",
}


class ClothFidelityInvalid(RuntimeError):
    def __init__(self, code: str, health: Mapping[str, object]):
        super().__init__(f"{code}: measured cloth fidelity is invalid")
        self.code = code
        self.health = dict(health)


def _config_get(value: object, key: str, default: object) -> object:
    getter = getattr(value, "get", None)
    return getter(key, default) if callable(getter) else getattr(value, key, default)


def _failure(code: str, *, reason: str) -> ClothFidelityInvalid:
    health = {
        "healthy": False,
        "sample_count": 0,
        "max_position_m": 0.0,
        "max_extent_m": 0.0,
        "max_velocity_mps": 0.0,
        "max_position_limit_m": 0.0,
        "max_extent_limit_m": 0.0,
        "max_velocity_limit_mps": 0.0,
        "missing_cloth": code == "missing_cloth",
        "cloth_flight": code == "cloth_flight",
        "nonfinite_cloth_state": code == "nonfinite_cloth_state",
        "reason": reason,
    }
    return ClothFidelityInvalid(code, health)


def measure_cloth_health(env: object) -> dict[str, object]:
    """Apply the repository's existing physical-health admission equations."""
    cloth = getattr(env, "object", None)
    if cloth is None:
        raise _failure("missing_cloth", reason="cloth object is missing")
    prim = getattr(cloth, "_prim", None)
    get_attribute = getattr(prim, "GetAttribute", None)
    if not callable(get_attribute):
        raise _failure("missing_cloth", reason="cloth prim is unavailable")
    position_attribute = get_attribute("points")
    velocity_attribute = get_attribute("velocities")
    if position_attribute is None or velocity_attribute is None:
        raise _failure("missing_cloth", reason="cloth points or velocities are missing")
    try:
        positions = np.asarray(position_attribute.Get(), dtype=np.float64)
        velocities = np.asarray(velocity_attribute.Get(), dtype=np.float64)
    except (AttributeError, TypeError, ValueError):
        raise _failure("nonfinite_cloth_state", reason="cloth state is malformed") from None
    if (
        positions.ndim != 2
        or positions.shape[1:] != (3,)
        or velocities.shape != positions.shape
        or positions.shape[0] == 0
    ):
        raise _failure("nonfinite_cloth_state", reason="cloth state is not aligned Nx3")
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        raise _failure("nonfinite_cloth_state", reason="cloth state is nonfinite")

    garment = getattr(env, "garment_config", None)
    objects = _config_get(getattr(env, "particle_config", None), "objects", None)
    common = _config_get(objects, "common", {})
    particle_system = _config_get(objects, "particle_system", {})
    try:
        scale = np.asarray(
            _config_get(garment, "scale", _config_get(common, "scale", [1.0] * 3)),
            dtype=np.float64,
        )
        reset_range = np.asarray(
            _config_get(
                garment,
                "soft_reset_pos_range",
                _config_get(common, "soft_reset_pos_range", [0.0] * 6),
            ),
            dtype=np.float64,
        )
        configured_max_velocity = float(_config_get(particle_system, "max_velocity", 5.0))
    except (TypeError, ValueError):
        raise _failure("nonfinite_cloth_state", reason="cloth health configuration is malformed") from None
    if (
        scale.shape != (3,)
        or reset_range.shape != (6,)
        or not np.isfinite(scale).all()
        or not np.isfinite(reset_range).all()
        or not math.isfinite(configured_max_velocity)
        or configured_max_velocity <= 0.0
    ):
        raise _failure("nonfinite_cloth_state", reason="cloth health configuration is invalid")

    position_limit = float(np.max(np.abs(reset_range)) + 2.0 * np.max(np.abs(scale)))
    extent_limit = float(4.0 * np.max(np.abs(scale)))
    velocity_limit = float(configured_max_velocity * 0.95)
    max_position = float(np.max(np.abs(positions)))
    max_extent = float(np.max(np.ptp(positions, axis=0)))
    max_velocity = float(np.max(np.linalg.norm(velocities, axis=1)))
    flight = (
        max_position > position_limit
        or max_extent > extent_limit
        or max_velocity > velocity_limit
    )
    health = {
        "healthy": not flight,
        "sample_count": int(positions.shape[0]),
        "max_position_m": max_position,
        "max_extent_m": max_extent,
        "max_velocity_mps": max_velocity,
        "max_position_limit_m": position_limit,
        "max_extent_limit_m": extent_limit,
        "max_velocity_limit_mps": velocity_limit,
        "missing_cloth": False,
        "cloth_flight": flight,
        "nonfinite_cloth_state": False,
    }
    if flight:
        raise ClothFidelityInvalid("cloth_flight", health)
    return health


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")


class _EvidenceWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        if self.path.exists() or self.path.is_symlink():
            raise RuntimeError("cloth fidelity evidence path must be new")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        self.previous = ZERO_SHA256
        self.pending: list[bytes] = []

    def _append_record(self, event: dict[str, object]) -> None:
        event["previous_event_sha256"] = self.previous
        event_sha = hashlib.sha256(_canonical(event)).hexdigest()
        event["event_sha256"] = event_sha
        self.pending.append(_canonical(event))
        self.previous = event_sha

    def append(
        self,
        *,
        garment: str,
        reset_sequence: int,
        stage: str,
        step_index: int,
        health: Mapping[str, object],
    ) -> None:
        self._append_record({
            "schema_version": 1,
            "kind": "lehome_native_cloth_fidelity_event_v1",
            "garment": garment,
            "reset_sequence": reset_sequence,
            "stage": stage,
            "step_index": step_index,
            "health": dict(health),
        })

    def append_terminal(
        self,
        *,
        garment: str,
        reset_sequence: int,
        step_index: int,
        status: str,
    ) -> None:
        if status not in {"healthy", "invalid"}:
            raise ValueError("cloth fidelity terminal status is invalid")
        self._append_record({
            "schema_version": 1,
            "kind": "lehome_native_cloth_fidelity_terminal_v1",
            "garment": garment,
            "reset_sequence": reset_sequence,
            "step_index": step_index,
            "status": status,
        })

    def flush(self, *, reason: str) -> None:
        if not self.pending:
            return
        count = len(self.pending)
        with self.path.open("ab") as stream:
            stream.write(b"".join(self.pending))
            stream.flush()
            os.fsync(stream.fileno())
        self.pending.clear()
        print(
            "LEHOME_CLOTH_FIDELITY_FLUSH "
            + json.dumps(
                {"reason": reason, "event_count": count, "last_event_sha256": self.previous},
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )


def install_cloth_fidelity_monitor_on_env(env: object, evidence_path: Path) -> None:
    """Observe the native environment without changing action or score behavior."""
    writer = _EvidenceWriter(Path(evidence_path))
    original_reset = env.reset
    original_step = env.step
    original_success = env._get_success
    original_close = getattr(env, "close", None)
    state = {
        "reset_sequence": 0,
        "step_index": 0,
        "garment": "",
        "episode_active": False,
        "closed": False,
    }

    def finalize_episode(*, status: str, reason: str) -> None:
        if not state["episode_active"]:
            return
        writer.append_terminal(
            garment=str(state["garment"]),
            reset_sequence=int(state["reset_sequence"]),
            step_index=int(state["step_index"]),
            status=status,
        )
        writer.flush(reason=reason)
        state["episode_active"] = False

    def close_monitor(*, reason: str) -> None:
        if state["closed"]:
            return
        finalize_episode(status="healthy", reason=reason)
        state["closed"] = True

    def process_exit() -> None:
        close_monitor(reason="process_exit")

    atexit.register(process_exit)

    def reset(self: object, *args: object, **kwargs: object):
        finalize_episode(status="healthy", reason="next_reset")
        result = original_reset(*args, **kwargs)
        state["reset_sequence"] += 1
        state["step_index"] = 0
        state["garment"] = str(getattr(getattr(self, "cfg", None), "garment_name", ""))
        return result

    def observe(self: object, stage: str) -> None:
        try:
            health = measure_cloth_health(self)
        except ClothFidelityInvalid as error:
            writer.append(
                garment=str(state["garment"]),
                reset_sequence=int(state["reset_sequence"]),
                stage=stage,
                step_index=int(state["step_index"]),
                health=error.health,
            )
            state["episode_active"] = True
            finalize_episode(status="invalid", reason=f"{stage}_invalid")
            raise
        writer.append(
            garment=str(state["garment"]),
            reset_sequence=int(state["reset_sequence"]),
            stage=stage,
            step_index=int(state["step_index"]),
            health=health,
        )
        state["episode_active"] = True

    def step(self: object, action: object):
        result = original_step(action)
        state["step_index"] += 1
        observe(self, "post_step")
        return result

    def get_success(self: object):
        observe(self, "pre_score")
        return original_success()

    def close(self: object, *args: object, **kwargs: object):
        close_monitor(reason="explicit_close")
        atexit.unregister(process_exit)
        if callable(original_close):
            return original_close(*args, **kwargs)
        return None

    env.reset = MethodType(reset, env)
    env.step = MethodType(step, env)
    env._get_success = MethodType(get_success, env)
    env.close = MethodType(close, env)


def validate_cloth_fidelity_evidence(
    path: Path, *, expected_episodes: Sequence[tuple[str, int]]
) -> dict[str, object]:
    evidence = Path(path)
    if evidence.is_symlink() or not evidence.is_file():
        raise ValueError("cloth fidelity evidence is unavailable")
    previous = ZERO_SHA256
    events: list[dict[str, object]] = []
    for raw_line in evidence.read_bytes().splitlines(keepends=True):
        try:
            event = json.loads(raw_line)
        except (UnicodeError, json.JSONDecodeError):
            raise ValueError("cloth fidelity evidence is invalid JSON") from None
        if not isinstance(event, dict) or event.get("previous_event_sha256") != previous:
            raise ValueError("cloth fidelity evidence hash chain is invalid")
        event_sha = event.pop("event_sha256", None)
        if not isinstance(event_sha, str) or hashlib.sha256(_canonical(event)).hexdigest() != event_sha:
            raise ValueError("cloth fidelity evidence hash chain is invalid")
        event["event_sha256"] = event_sha
        previous = event_sha
        common_invalid = (
            event.get("schema_version") != 1
            or type(event.get("reset_sequence")) is not int
            or type(event.get("step_index")) is not int
            or not isinstance(event.get("garment"), str)
        )
        if event.get("kind") == "lehome_native_cloth_fidelity_event_v1":
            health = event.get("health")
            schema_invalid = (
                set(event) != {
                    "schema_version", "kind", "garment", "reset_sequence", "stage",
                    "step_index", "health", "previous_event_sha256", "event_sha256",
                }
                or event.get("stage") not in {"post_step", "pre_score"}
                or not isinstance(health, Mapping)
                or not _HEALTH_KEYS <= set(health)
            )
        elif event.get("kind") == "lehome_native_cloth_fidelity_terminal_v1":
            schema_invalid = (
                set(event) != {
                    "schema_version", "kind", "garment", "reset_sequence",
                    "step_index", "status", "previous_event_sha256", "event_sha256",
                }
                or event.get("status") not in {"healthy", "invalid"}
            )
        else:
            schema_invalid = True
        if common_invalid or schema_invalid:
            raise ValueError("cloth fidelity evidence schema is invalid")
        events.append(event)
    if not events:
        raise ValueError("cloth fidelity evidence is empty")

    active: list[tuple[str, int]] = []
    for key in dict.fromkeys((str(row["garment"]), int(row["reset_sequence"])) for row in events):
        selected = [row for row in events if (row["garment"], row["reset_sequence"]) == key]
        observations = [
            row for row in selected
            if row["kind"] == "lehome_native_cloth_fidelity_event_v1"
        ]
        terminals = [
            row for row in selected
            if row["kind"] == "lehome_native_cloth_fidelity_terminal_v1"
        ]
        if not observations or len(terminals) != 1 or selected[-1] is not terminals[0]:
            raise ValueError("cloth fidelity episode lacks a terminal sentinel")
        invalid_observed = any(
            any(bool(row["health"].get(flag)) for flag in (
                "missing_cloth", "cloth_flight", "nonfinite_cloth_state"
            ))
            for row in observations
        )
        terminal_status = terminals[0]["status"]
        if terminal_status == "healthy":
            if invalid_observed or not any(row["stage"] == "post_step" for row in observations):
                raise ValueError("cloth fidelity healthy terminal is inconsistent")
            if not any(row["stage"] == "pre_score" for row in observations):
                raise ValueError("cloth fidelity episode lacks pre-score evidence")
        elif not invalid_observed:
            raise ValueError("cloth fidelity invalid terminal lacks invalid evidence")
        active.append(key)
    ordinal: dict[str, int] = {}
    observed: list[tuple[str, int]] = []
    for garment, _reset in active:
        ordinal[garment] = ordinal.get(garment, 0) + 1
        observed.append((garment, ordinal[garment]))
    if observed != list(expected_episodes):
        raise ValueError("cloth fidelity episode provenance drift")
    invalid_episodes = 0
    for key in active:
        if any(
            row.get("kind") == "lehome_native_cloth_fidelity_event_v1"
            and any(bool(row["health"].get(flag)) for flag in ("missing_cloth", "cloth_flight", "nonfinite_cloth_state"))
            for row in events
            if (row["garment"], row["reset_sequence"]) == key
        ):
            invalid_episodes += 1
    return {
        "measured_episode_count": len(active),
        "fidelity_invalid_count": invalid_episodes,
        "event_count": len(events),
        "first_event_sha256": events[0]["event_sha256"],
        "last_event_sha256": events[-1]["event_sha256"],
        "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
    }
