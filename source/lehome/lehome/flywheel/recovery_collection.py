"""Verified bootstrap for controlled autonomous success-recovery rollouts.

The source action prefix is provenance, never data: it deterministically puts
the simulator at an audited intermediate state.  The recorder is deliberately
created only after that prefix and its bounded perturbation have completed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
import stat
from typing import Any, Mapping, Sequence

from lehome.flywheel.snapshots import Snapshot


RECOVERY_KIND = "controlled_success_recovery_v1"
_SHA256_LENGTH = 64
_LOWERCASE_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_CLOTH_DISPLACEMENT_M = 0.01
_MAX_CLOTH_VELOCITY_MPS = 0.05
_MAX_GRIPPER_OFFSET_RAD = 0.08
_REPLAY_FIDELITY_TOLERANCE_RAD = 0.005


@dataclass(frozen=True, slots=True)
class ControlledRecovery:
    reset_payload: Mapping[str, object]
    prefix_actions: tuple[tuple[float, ...], ...]
    teacher_actions: tuple[tuple[float, ...], ...]
    continuation_state: tuple[float, ...]
    perturbation_profile: Mapping[str, float]
    perturbation_seed: int
    provenance: Mapping[str, object]


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_json_file(value: object, expected: object, *, field: str) -> tuple[Path, object]:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an absolute regular file")
    path = Path(value)
    if not path.is_absolute() or path.is_symlink():
        raise ValueError(f"{field} must be an absolute regular file")
    try:
        if not stat.S_ISREG(path.stat().st_mode):
            raise ValueError(f"{field} must be an absolute regular file")
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"{field} must be an absolute regular file") from error
    if not isinstance(expected, str) or len(expected) != _SHA256_LENGTH or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError(f"{field} requires a lowercase SHA-256")
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"{field} SHA-256 mismatch")
    try:
        return path, json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} must contain JSON") from error


def _annotations(path: Path, expected: object) -> tuple[tuple[tuple[float, ...], ...], tuple[bool, ...]]:
    if path.is_symlink() or not path.is_absolute() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("source annotations must be an absolute regular file")
    if not isinstance(expected, str) or _sha256(path) != expected:
        raise ValueError("source annotations SHA-256 mismatch")
    rows: list[Mapping[str, object]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if not isinstance(item, Mapping):
                raise ValueError("source annotations row is malformed")
            rows.append(item)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("source annotations are malformed") from error
    actions: list[tuple[float, ...]] = []
    successes: list[bool] = []
    for index, row in enumerate(rows):
        if row.get("step") != index or not isinstance(row.get("action"), list):
            raise ValueError("source annotations must have ordered recorded actions")
        action = tuple(float(value) for value in row["action"])
        if len(action) != 12:
            raise ValueError("source action must be 12-D")
        if not all(math.isfinite(value) for value in action):
            raise ValueError("source action must be finite")
        actions.append(action)
        successes.append(row.get("success") is True)
    return tuple(actions), tuple(successes)


def _continuation_state(value: object, *, category: object, garment: object, fingerprint: object) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != 12:
        raise ValueError("controlled recovery continuation state must be a finite 12-D vector")
    if not isinstance(category, str) or not category or not isinstance(garment, str) or not garment:
        raise ValueError("controlled recovery continuation state identity is invalid")
    state = tuple(float(item) for item in value)
    if any(type(item) not in (int, float) or not math.isfinite(number) for item, number in zip(value, state, strict=True)):
        raise ValueError("controlled recovery continuation state must be a finite 12-D vector")
    rounded = ["0.000000" if number == 0.0 else format(number, ".6f") for number in state]
    actual = hashlib.sha256(json.dumps(
        {"category": category, "garment": garment, "state_rounding": "fixed_6dp", "state": rounded},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    if not isinstance(fingerprint, str) or fingerprint != actual:
        raise ValueError("controlled recovery continuation state fingerprint mismatch")
    return state


def _replay_fidelity(env: object, expected: Sequence[float]) -> Mapping[str, object]:
    from lehome.flywheel.snapshots import capture_snapshot

    observed = tuple(capture_snapshot(
        env, randomization={"strategy": "canonical", "recovery_kind": RECOVERY_KIND},
    ).robot_position)
    if len(observed) != 12 or not all(math.isfinite(value) for value in observed):
        raise ValueError("controlled recovery replay fidelity state is invalid")
    max_error = max(abs(actual - target) for actual, target in zip(observed, expected, strict=True))
    result = {
        "verified": True,
        "tolerance_rad": _REPLAY_FIDELITY_TOLERANCE_RAD,
        "max_abs_error_rad": max_error,
        "expected_state_sha256": hashlib.sha256(_canonical_bytes(list(expected))).hexdigest(),
        "observed_state_sha256": hashlib.sha256(_canonical_bytes(list(observed))).hexdigest(),
    }
    if max_error > _REPLAY_FIDELITY_TOLERANCE_RAD:
        raise ValueError("controlled recovery replay fidelity exceeds fixed robot-position tolerance")
    return result


def _teacher_success(env: object) -> bool:
    checker = getattr(env, "_get_success", None)
    if not callable(checker):
        raise ValueError("controlled recovery teacher probe requires a live success checker")
    result = checker()
    try:
        return bool(result.item()) if hasattr(result, "item") else bool(result)
    except (TypeError, ValueError) as error:
        raise ValueError("controlled recovery teacher probe success checker is invalid") from error


def _profile(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError("perturbation profile is required")
    expected = {"cloth_displacement_m", "cloth_velocity_mps", "gripper_offset_rad"}
    if set(value) != expected:
        raise ValueError("perturbation profile fields are invalid")
    result = {key: float(value[key]) for key in expected}
    if not all(math.isfinite(item) and item >= 0 for item in result.values()):
        raise ValueError("perturbation profile must be finite and non-negative")
    if (result["cloth_displacement_m"] > _MAX_CLOTH_DISPLACEMENT_M
            or result["cloth_velocity_mps"] > _MAX_CLOTH_VELOCITY_MPS
            or result["gripper_offset_rad"] > _MAX_GRIPPER_OFFSET_RAD):
        raise ValueError("perturbation profile exceeds a bounded recovery perturbation")
    return result


def load_controlled_recovery(assignment: Mapping[str, object]) -> ControlledRecovery:
    """Verify all immutable bootstrap inputs before the environment mutates."""

    if assignment.get("recovery_kind") != RECOVERY_KIND:
        raise ValueError("assignment is not a controlled success recovery")
    reset_path, reset = _verified_json_file(
        assignment.get("source_reset"), assignment.get("source_reset_sha256"), field="source reset"
    )
    if not isinstance(reset, Mapping):
        raise ValueError("source reset must contain a JSON object")
    annotations_value = assignment.get("source_annotations")
    if not isinstance(annotations_value, str):
        raise ValueError("source annotations must be an absolute regular file")
    actions, successes = _annotations(Path(annotations_value), assignment.get("source_annotations_sha256"))
    stop = assignment.get("prefix_stop")
    if type(stop) is not int or not 1 <= stop < len(actions):
        raise ValueError("prefix stop must select a strict nonempty action prefix")
    first_success = assignment.get("source_first_success_step")
    if type(first_success) is not int or not stop < first_success < len(actions):
        raise ValueError("prefix stop must precede the first recorded success")
    if not successes[first_success] or any(successes[:first_success]):
        raise ValueError("source first recorded success does not match authenticated annotations")
    prefix = actions[:stop]
    expected_prefix = assignment.get("action_prefix_sha256")
    if not isinstance(expected_prefix, str) or hashlib.sha256(_canonical_bytes([list(action) for action in prefix])).hexdigest() != expected_prefix:
        raise ValueError("action prefix SHA-256 mismatch")
    seed = assignment.get("perturbation_seed")
    if type(seed) is not int or seed < 0:
        raise ValueError("perturbation seed must be a non-negative integer")
    profile = _profile(assignment.get("perturbation_profile"))
    required = ("source_round_id", "source_episode_id", "source_episode_digest", "source_immutable_revision", "source_state_fingerprint", "perturbation_fingerprint", "source_state_perturbation_fingerprint")
    if any(not isinstance(assignment.get(key), str) or not assignment[key] for key in required):
        raise ValueError("controlled recovery source lineage is incomplete")
    continuation = _continuation_state(
        assignment.get("source_continuation_state"), category=assignment.get("category"),
        garment=assignment.get("garment"), fingerprint=assignment.get("source_state_fingerprint"),
    )
    teacher_probe = assignment.get("controlled_smoke_teacher_probe", False)
    if type(teacher_probe) is not bool:
        raise ValueError("controlled recovery teacher probe flag is invalid")
    if teacher_probe and assignment.get("controlled_smoke") is not True:
        raise ValueError("controlled recovery teacher probe is smoke-only")
    provenance = {key: assignment[key] for key in assignment if key.startswith("source_") or key in {"action_prefix_sha256", "prefix_stop", "perturbation_profile", "perturbation_seed", "perturbation_fingerprint", "recovery_kind", "controlled_smoke_teacher_probe"}}
    provenance.update({"source_reset": str(reset_path), "source_annotations": str(Path(annotations_value))})
    return ControlledRecovery(reset, prefix, actions[stop:first_success + 1] if teacher_probe else (), continuation, profile, seed, provenance)


def load_attempt_matrix(path_value: str | Path) -> list[Mapping[str, object]]:
    """Load legacy rows or the one canonical controlled materialization form.

    This is intentionally shared by the worker and preemption hooks so both
    reconstruct the same immutable TaskLedger attempt identities.
    """

    path = Path(path_value)
    if path.is_symlink() or not path.is_file():
        raise ValueError("attempt matrix must be a regular JSON file")
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("attempt matrix must be valid JSON") from error
    if isinstance(decoded, list):
        if not all(isinstance(row, Mapping) for row in decoded):
            raise ValueError("attempt matrix must be a JSON array of assignments")
        return [dict(row) for row in decoded]
    envelope = {"schema_version", "kind", "matrix_sha256", "target_accepted", "category_acceptance_caps", "rows"}
    if not isinstance(decoded, Mapping) or set(decoded) != envelope:
        raise ValueError("attempt matrix must be a JSON array or controlled materialization")
    matrix_sha256 = decoded.get("matrix_sha256")
    rows, target, caps = decoded.get("rows"), decoded.get("target_accepted"), decoded.get("category_acceptance_caps")
    if decoded.get("schema_version") != 1 or decoded.get("kind") != "controlled_success_recovery_materialization_v1":
        raise ValueError("controlled materialization has an incompatible schema")
    if not isinstance(matrix_sha256, str) or _LOWERCASE_SHA256.fullmatch(matrix_sha256) is None:
        raise ValueError("controlled materialization matrix hash is invalid")
    expected_caps = {"pant_long": 4, "top_long": 1, "top_short": 3, "pant_short": 0}
    if target != 8 or caps != expected_caps:
        raise ValueError("controlled materialization has an invalid acceptance contract")
    if not isinstance(rows, list) or not 8 <= len(rows) <= 96 or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("controlled materialization rows are invalid")
    hydrated: list[Mapping[str, object]] = []
    categories: dict[str, int] = {category: 0 for category in expected_caps}
    seen_attempt_ids: set[str] = set()
    seen_trial_ids: set[str] = set()
    seen_seeds: set[int] = set()
    seen_perturbations: set[str] = set()
    seen_source_perturbations: set[str] = set()
    for row in rows:
        if row.get("recovery_kind") != RECOVERY_KIND or row.get("controlled_matrix_sha256") != matrix_sha256:
            raise ValueError("controlled materialization matrix hash does not bind every row")
        category = row.get("category")
        if category not in {"pant_long", "top_long", "top_short"} or row.get("category_acceptance_cap") != expected_caps[category]:
            raise ValueError("controlled materialization row category or cap is invalid")
        categories[category] += 1
        if row.get("strategy") != "canonical":
            raise ValueError("controlled materialization row strategy is invalid")
        attempt_id, trial_id, seed = row.get("attempt_id"), row.get("trial_id"), row.get("perturbation_seed")
        if not isinstance(attempt_id, str) or not attempt_id or attempt_id in seen_attempt_ids or not isinstance(trial_id, str) or not trial_id or trial_id in seen_trial_ids:
            raise ValueError("controlled materialization attempt or trial IDs must be unique")
        if type(seed) is not int or seed < 0 or seed in seen_seeds:
            raise ValueError("controlled materialization perturbation seeds must be unique non-negative integers")
        seen_attempt_ids.add(attempt_id); seen_trial_ids.add(trial_id); seen_seeds.add(seed)
        for field, seen in (("perturbation_fingerprint", seen_perturbations), ("source_state_perturbation_fingerprint", seen_source_perturbations)):
            value = row.get(field)
            if not isinstance(value, str) or _LOWERCASE_SHA256.fullmatch(value) is None or value in seen:
                raise ValueError("controlled materialization fingerprints must be unique lowercase SHA-256 values")
            seen.add(value)
        for field in ("source_episode_digest", "source_reset_sha256", "source_annotations_sha256", "action_prefix_sha256", "source_state_fingerprint"):
            if not isinstance(row.get(field), str) or _LOWERCASE_SHA256.fullmatch(str(row[field])) is None:
                raise ValueError("controlled materialization source identity is invalid")
        continuation = row.get("source_continuation_state")
        if (not isinstance(continuation, list) or len(continuation) != 12
                or any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in continuation)):
            raise ValueError("controlled materialization continuation state is invalid")
        if not isinstance(row.get("source_round_id"), str) or not row["source_round_id"] or not isinstance(row.get("source_episode_id"), str) or not row["source_episode_id"]:
            raise ValueError("controlled materialization source identity is invalid")
        if type(row.get("prefix_stop")) is not int or type(row.get("source_first_success_step")) is not int or not 0 < row["prefix_stop"] < row["source_first_success_step"]:
            raise ValueError("controlled materialization source prefix is invalid")
        for field in ("source_reset", "source_annotations"):
            value = row.get(field)
            candidate = Path(value) if isinstance(value, str) else None
            if candidate is None or not candidate.is_absolute() or candidate.is_symlink() or not candidate.is_file():
                raise ValueError("controlled materialization source path is unsafe")
        hydrated.append(dict(row))
    if any(categories[category] < cap for category, cap in expected_caps.items() if cap):
        raise ValueError("controlled materialization schedule cannot reach its category acceptance caps")
    return hydrated


def replay_action_prefix(env: object, actions: Sequence[Sequence[float]]) -> None:
    """Open-loop replay of finite, exact 12D recorded actions."""

    normalized: list[list[float]] = []
    for raw in actions:
        action = [float(value) for value in raw]
        if len(action) != 12:
            raise ValueError("source action must be 12-D")
        if not all(math.isfinite(value) for value in action):
            raise ValueError("source action must be finite")
        normalized.append(action)
    step = getattr(env, "step", None)
    if not callable(step):
        raise ValueError("environment does not support stepping")
    for action in normalized:
        # Lightweight test adapters receive the original list representation;
        # Isaac accepts the same nested numeric action after evaluation converts it.
        if hasattr(env, "device"):
            try:
                import torch
                step(torch.tensor([action], dtype=torch.float32, device=getattr(env, "device")))
                continue
            except ImportError:  # pragma: no cover - Isaac always brings torch.
                pass
        step(action)


def _snapshot(value: Mapping[str, object]) -> Snapshot:
    return Snapshot(
        schema_version=int(value["schema_version"]), robot_position=tuple(value["robot_position"]),
        robot_velocity=tuple(value["robot_velocity"]),
        cloth_position=tuple(tuple(point) for point in value["cloth_position"]),
        cloth_velocity=tuple(tuple(point) for point in value["cloth_velocity"]),
        rng_state=dict(value["rng_state"]), garment_name=str(value["garment_name"]),
        randomization=dict(value.get("randomization") or {}), scene_state=dict(value.get("scene_state") or {}),
    )


def apply_controlled_perturbation(snapshot: Mapping[str, object] | Snapshot, profile: Mapping[str, object], seed: int) -> Snapshot:
    """Return a deterministic, bounded cloth/gripper-only intermediate state."""

    source = snapshot if isinstance(snapshot, Snapshot) else _snapshot(snapshot)
    values = _profile(profile)
    if type(seed) is not int or seed < 0:
        raise ValueError("perturbation seed must be a non-negative integer")
    rng = random.Random(seed)
    displaced = tuple(tuple(point[axis] + rng.uniform(-values["cloth_displacement_m"], values["cloth_displacement_m"]) for axis in range(3)) for point in source.cloth_position)
    velocities = tuple(tuple(point[axis] + rng.uniform(-values["cloth_velocity_mps"], values["cloth_velocity_mps"]) for axis in range(3)) for point in source.cloth_velocity)
    joints = list(source.robot_position)
    for index in (5, 11):
        joints[index] += rng.uniform(-values["gripper_offset_rad"], values["gripper_offset_rad"])
    return Snapshot(source.schema_version, tuple(joints), source.robot_velocity, displaced, velocities, source.rng_state, source.garment_name, source.randomization, source.scene_state)


def bootstrap_controlled_recovery(env: object, assignment: Mapping[str, object]) -> Mapping[str, object]:
    """Restore, prefix replay, perturb, restore/read back, then expose lineage."""

    from lehome.flywheel.snapshots import capture_snapshot, restore_snapshot

    recovery = load_controlled_recovery(assignment)
    restore_snapshot(env, _snapshot(recovery.reset_payload))
    replay_action_prefix(env, recovery.prefix_actions)
    checks = [_replay_fidelity(env, recovery.continuation_state)]
    teacher_provenance: Mapping[str, object] | None = None
    if recovery.teacher_actions:
        replay_action_prefix(env, recovery.teacher_actions)
        if not _teacher_success(env):
            raise ValueError("controlled recovery teacher probe did not reproduce source success")
        restore_snapshot(env, _snapshot(recovery.reset_payload))
        replay_action_prefix(env, recovery.prefix_actions)
        checks.append(_replay_fidelity(env, recovery.continuation_state))
        teacher_provenance = {"enabled": True, "verified": True, "replayed_action_count": len(recovery.teacher_actions)}
    intermediate = capture_snapshot(env, randomization={"strategy": "canonical", "recovery_kind": RECOVERY_KIND})
    perturbed = apply_controlled_perturbation(intermediate, recovery.perturbation_profile, recovery.perturbation_seed)
    restore_snapshot(env, perturbed)
    # Capture forces an adapter readback when available and is the state the
    # autonomous recorder must treat as its reset snapshot.
    capture_snapshot(env, randomization={"strategy": "canonical", "recovery_kind": RECOVERY_KIND})
    provenance = dict(recovery.provenance)
    provenance.update({"replay_fidelity": checks[-1], "replay_fidelity_checks": checks})
    if teacher_provenance is not None:
        provenance["teacher_probe"] = teacher_provenance
    return provenance


__all__ = ["RECOVERY_KIND", "ControlledRecovery", "apply_controlled_perturbation", "bootstrap_controlled_recovery", "load_attempt_matrix", "load_controlled_recovery", "replay_action_prefix"]
