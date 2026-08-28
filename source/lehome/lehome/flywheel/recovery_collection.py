"""Verified bootstrap for controlled autonomous success-recovery rollouts.

CUDA restores a checksum-authenticated physical H=16 continuation directly.
CPU reconstructs an authenticated USD-local source boundary from its reset and
recorded prefix because a USD snapshot cannot restore hidden solver state.
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
from typing import Any, Callable, Mapping, Sequence

from lehome.flywheel.snapshots import (
    LEGACY_USD_LOCAL_CLOTH_AUTHORITY, PHYSX_CLOTH_STATE_AUTHORITY, Snapshot,
)


RECOVERY_KIND = "controlled_success_recovery_snapshot_v3"
VERIFIED_SUCCESS_REPLAY_KINDS = frozenset(
    {"verified_success_reset_v1", "verified_success_early_snapshot_v1"}
)
VERIFIED_HARD_STATE_KINDS = frozenset({"verified_hard_state_moment_of_ruin_v1"})
_SHA256_LENGTH = 64
_LOWERCASE_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANONICAL_SEEN_GARMENTS = {
    "top_long": re.compile(r"^Top_Long_Seen_[0-9]+$"),
    "top_short": re.compile(r"^Top_Short_Seen_[0-9]+$"),
    "pant_long": re.compile(r"^Pant_Long_Seen_[0-9]+$"),
    "pant_short": re.compile(r"^Pant_Short_Seen_[0-9]+$"),
}
_MAX_CLOTH_DISPLACEMENT_M = 0.01
_MAX_CLOTH_VELOCITY_MPS = 0.05
_MAX_GRIPPER_OFFSET_RAD = 0.08
_REPLAY_FIDELITY_TOLERANCE_RAD = 0.005
_ROBOT_VELOCITY_FIDELITY_TOLERANCE_RADPS = 0.005
_CLOTH_POSITION_FIDELITY_TOLERANCE_M = 1e-5
_CLOTH_VELOCITY_FIDELITY_TOLERANCE_MPS = 1e-5


@dataclass(frozen=True, slots=True)
class ControlledRecovery:
    reset_snapshot: Snapshot
    prefix_actions: tuple[tuple[float, ...], ...]
    continuation_snapshot: Snapshot
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


def _strict_absolute_regular_file(value: object, *, field: str) -> Path:
    """Reject symlinks at the leaf *and every existing ancestor*.

    ``Path.resolve`` alone is insufficient: it accepts an ancestor symlink and
    would let a post-validation replacement redirect an immutable bootstrap
    input.  All source evidence paths are expected to be absolute files under
    a real materialization root.
    """

    if not isinstance(value, (str, Path)):
        raise ValueError(f"{field} must be an absolute regular file")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{field} must be an absolute regular file")
    current = path
    while True:
        try:
            metadata = current.lstat()
        except OSError as error:
            raise ValueError(f"{field} must be an absolute regular file") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{field} must be an absolute regular file")
        if current.parent == current:
            break
        current = current.parent
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"{field} must be an absolute regular file")
    return path


def _verified_json_file(value: object, expected: object, *, field: str) -> tuple[Path, object]:
    path = _strict_absolute_regular_file(value, field=field)
    try:
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
    path = _strict_absolute_regular_file(str(path), field="source annotations")
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


def _source_snapshot_contract(
    reset_snapshot: Snapshot, continuation_snapshot: Snapshot,
) -> tuple[int, str]:
    """Require one exact frame authority across the authenticated source pair."""

    contract = (
        continuation_snapshot.schema_version,
        continuation_snapshot.cloth_state_authority,
    )
    if contract not in {
        (2, PHYSX_CLOTH_STATE_AUTHORITY),
        (3, LEGACY_USD_LOCAL_CLOTH_AUTHORITY),
    } or (reset_snapshot.schema_version, reset_snapshot.cloth_state_authority) != contract:
        raise ValueError("controlled recovery source snapshots have an invalid or mixed cloth authority")
    return contract


def _projection_provenance(
    env: object, recovery: ControlledRecovery, projected: Snapshot,
) -> Mapping[str, object]:
    """Bind a legacy source to the exact CUDA PhysX readback it produced."""

    receipt = getattr(env, "_flywheel_legacy_projection_receipt", None)
    if not isinstance(receipt, Mapping):
        raise ValueError("controlled recovery legacy USD projection receipt is missing")
    required = {
        "source_snapshot_authority", "weld_map_identity",
        "welded_vertices_remap_to_orig_sha256", "welded_vertices_remap_to_weld_sha256",
    }
    if set(receipt) != required or receipt.get("source_snapshot_authority") != LEGACY_USD_LOCAL_CLOTH_AUTHORITY:
        raise ValueError("controlled recovery legacy USD projection receipt is malformed")
    if any(not isinstance(receipt[field], str) or _LOWERCASE_SHA256.fullmatch(receipt[field]) is None for field in required - {"source_snapshot_authority"}):
        raise ValueError("controlled recovery legacy USD projection receipt is malformed")
    state = {
        "robot_position": list(projected.robot_position),
        "robot_velocity": list(projected.robot_velocity),
        "cloth_position": [list(point) for point in projected.cloth_position],
        "cloth_velocity": [list(point) for point in projected.cloth_velocity],
        "cloth_state_authority": projected.cloth_state_authority,
    }
    source_sha = recovery.provenance.get("source_continuation_snapshot_sha256")
    if not isinstance(source_sha, str) or _LOWERCASE_SHA256.fullmatch(source_sha) is None:
        raise ValueError("controlled recovery legacy source snapshot SHA-256 is invalid")
    return {
        "source_snapshot_sha256": source_sha,
        "source_snapshot_authority": LEGACY_USD_LOCAL_CLOTH_AUTHORITY,
        **dict(receipt),
        "projected_schema_version": 2,
        "projected_cloth_state_authority": PHYSX_CLOTH_STATE_AUTHORITY,
        "projected_physical_state_sha256": hashlib.sha256(_canonical_bytes(state)).hexdigest(),
    }


def _runtime_cloth_contract(env: object) -> tuple[int, str]:
    device = str(getattr(env, "device", "cuda:0"))
    if device == "cpu":
        return 3, LEGACY_USD_LOCAL_CLOTH_AUTHORITY
    if re.fullmatch(r"cuda:[0-9]+", device):
        return 2, PHYSX_CLOTH_STATE_AUTHORITY
    raise ValueError("controlled recovery runtime device has no supported cloth authority")


def _replay_fidelity(env: object, expected: Snapshot) -> Mapping[str, object]:
    from lehome.flywheel.snapshots import capture_snapshot

    observed = capture_snapshot(
        env, randomization={"strategy": "canonical", "recovery_kind": RECOVERY_KIND},
    )
    contract = _runtime_cloth_contract(env)
    if ((expected.schema_version, expected.cloth_state_authority) != contract
            or (observed.schema_version, observed.cloth_state_authority) != contract):
        raise ValueError("controlled recovery replay fidelity lacks the expected cloth authority")
    observed_robot = tuple(observed.robot_position)
    observed_robot_velocity = tuple(observed.robot_velocity)
    if (len(observed_robot) != 12 or len(observed_robot_velocity) != 12
            or not all(math.isfinite(value) for value in (*observed_robot, *observed_robot_velocity))):
        raise ValueError("controlled recovery replay fidelity state is invalid")
    robot_error = max(abs(actual - target) for actual, target in zip(observed_robot, expected.robot_position, strict=True))
    robot_velocity_error = max(
        abs(actual - target)
        for actual, target in zip(observed_robot_velocity, expected.robot_velocity, strict=True)
    )
    if len(observed.cloth_position) != len(expected.cloth_position) or len(observed.cloth_velocity) != len(expected.cloth_velocity):
        raise ValueError("controlled recovery replay fidelity cloth shape is invalid")
    cloth_position_error = max(
        abs(actual - target)
        for actual_point, target_point in zip(observed.cloth_position, expected.cloth_position, strict=True)
        for actual, target in zip(actual_point, target_point, strict=True)
    )
    cloth_velocity_error = max(
        abs(actual - target)
        for actual_point, target_point in zip(observed.cloth_velocity, expected.cloth_velocity, strict=True)
        for actual, target in zip(actual_point, target_point, strict=True)
    )
    def physical_state(snapshot: Snapshot) -> dict[str, object]:
        return {
            "robot_position": list(snapshot.robot_position),
            "robot_velocity": list(snapshot.robot_velocity),
            "cloth_position": [list(point) for point in snapshot.cloth_position],
            "cloth_velocity": [list(point) for point in snapshot.cloth_velocity],
            "cloth_state_authority": snapshot.cloth_state_authority,
        }

    result = {
        "verified": True,
        "tolerance_rad": _REPLAY_FIDELITY_TOLERANCE_RAD,
        "max_abs_error_rad": robot_error,
        "robot_velocity_tolerance_radps": _ROBOT_VELOCITY_FIDELITY_TOLERANCE_RADPS,
        "max_abs_robot_velocity_error_radps": robot_velocity_error,
        "cloth_position_tolerance_m": _CLOTH_POSITION_FIDELITY_TOLERANCE_M,
        "cloth_velocity_tolerance_mps": _CLOTH_VELOCITY_FIDELITY_TOLERANCE_MPS,
        "max_abs_cloth_position_error_m": cloth_position_error,
        "max_abs_cloth_velocity_error_mps": cloth_velocity_error,
        "expected_state_sha256": hashlib.sha256(_canonical_bytes(physical_state(expected))).hexdigest(),
        "observed_state_sha256": hashlib.sha256(_canonical_bytes(physical_state(observed))).hexdigest(),
        "cloth_state_authority": observed.cloth_state_authority,
    }
    if robot_error > _REPLAY_FIDELITY_TOLERANCE_RAD:
        raise ValueError(
            "controlled recovery replay fidelity exceeds fixed robot-position tolerance "
            f"(observed={robot_error:.9g}, limit={_REPLAY_FIDELITY_TOLERANCE_RAD:.9g})"
        )
    if robot_velocity_error > _ROBOT_VELOCITY_FIDELITY_TOLERANCE_RADPS:
        raise ValueError(
            "controlled recovery replay fidelity exceeds fixed robot-velocity tolerance "
            f"(observed={robot_velocity_error:.9g}, limit={_ROBOT_VELOCITY_FIDELITY_TOLERANCE_RADPS:.9g})"
        )
    if cloth_position_error > _CLOTH_POSITION_FIDELITY_TOLERANCE_M:
        raise ValueError(
            "controlled recovery replay fidelity exceeds fixed cloth-position tolerance "
            f"(observed={cloth_position_error:.9g}, limit={_CLOTH_POSITION_FIDELITY_TOLERANCE_M:.9g})"
        )
    if cloth_velocity_error > _CLOTH_VELOCITY_FIDELITY_TOLERANCE_MPS:
        raise ValueError(
            "controlled recovery replay fidelity exceeds fixed cloth-velocity tolerance "
            f"(observed={cloth_velocity_error:.9g}, limit={_CLOTH_VELOCITY_FIDELITY_TOLERANCE_MPS:.9g})"
        )
    return result


def _teacher_success(env: object) -> bool:
    checker = getattr(env, "flywheel_check_success_unthrottled", None)
    if not callable(checker):
        raise ValueError("controlled recovery teacher probe requires an unthrottled live success checker")
    result = checker()
    if type(result) is not bool:
        raise ValueError("controlled recovery teacher probe success checker is invalid")
    return result


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
    _, reset_payload = _verified_json_file(
        assignment.get("source_reset"), assignment.get("source_reset_sha256"), field="source reset snapshot"
    )
    if not isinstance(reset_payload, Mapping):
        raise ValueError("source reset snapshot must contain a JSON object")
    reset_snapshot = _snapshot(reset_payload)
    snapshot_path, snapshot_payload = _verified_json_file(
        assignment.get("source_continuation_snapshot"), assignment.get("source_continuation_snapshot_sha256"), field="source continuation snapshot"
    )
    if not isinstance(snapshot_payload, Mapping):
        raise ValueError("source continuation snapshot must contain a JSON object")
    continuation_snapshot = _snapshot(snapshot_payload)
    source_snapshot_schema_version, source_snapshot_authority = _source_snapshot_contract(
        reset_snapshot, continuation_snapshot,
    )
    if (
        assignment.get("source_snapshot_schema_version") != source_snapshot_schema_version
        or assignment.get("source_snapshot_authority") != source_snapshot_authority
    ):
        raise ValueError("controlled recovery source snapshot authority is not bound by the assignment")
    if type(assignment.get("source_only_envelope")) is not bool or (
        source_snapshot_schema_version == 3 and assignment["source_only_envelope"] is not True
    ):
        raise ValueError("controlled recovery source envelope does not authorize this snapshot authority")
    annotations_value = assignment.get("source_annotations")
    if not isinstance(annotations_value, str):
        raise ValueError("source annotations must be an absolute regular file")
    actions, successes = _annotations(Path(annotations_value), assignment.get("source_annotations_sha256"))
    stop = assignment.get("prefix_stop")
    if type(stop) is not int or not 0 < stop < len(actions) or stop % 16:
        raise ValueError("continuation boundary must be a strict positive H16 action index")
    _validate_continuation_snapshot_boundary(
        continuation_snapshot=continuation_snapshot, reset_snapshot=reset_snapshot, step=stop,
    )
    first_success = assignment.get("source_first_success_step")
    if type(first_success) is not int or not stop < first_success < len(actions):
        raise ValueError("continuation boundary must precede the first recorded success")
    if not successes[first_success] or any(successes[:first_success]):
        raise ValueError("source first recorded success does not match authenticated annotations")
    seed = assignment.get("perturbation_seed")
    if type(seed) is not int or seed < 0:
        raise ValueError("perturbation seed must be a non-negative integer")
    profile = _profile(assignment.get("perturbation_profile"))
    required = ("source_round_id", "source_episode_id", "source_episode_digest", "source_immutable_revision", "source_state_fingerprint", "perturbation_fingerprint", "source_state_perturbation_fingerprint")
    if any(not isinstance(assignment.get(key), str) or not assignment[key] for key in required):
        raise ValueError("controlled recovery source lineage is incomplete")
    if type(assignment.get("source_seed")) is not int or assignment["source_seed"] < 0:
        raise ValueError("controlled recovery source reset seed is invalid")
    continuation = _continuation_state(
        assignment.get("source_continuation_state"), category=assignment.get("category"),
        garment=assignment.get("garment"), fingerprint=assignment.get("source_state_fingerprint"),
    )
    if (continuation_snapshot.garment_name != assignment.get("garment")
            or list(continuation_snapshot.robot_position) != list(continuation)):
        raise ValueError("source continuation snapshot does not match authenticated annotation state")
    teacher_probe = assignment.get("controlled_smoke_teacher_probe", False)
    if type(teacher_probe) is not bool:
        raise ValueError("controlled recovery teacher probe flag is invalid")
    if teacher_probe and assignment.get("controlled_smoke") is not True:
        raise ValueError("controlled recovery teacher probe is smoke-only")
    provenance = {key: assignment[key] for key in assignment if key.startswith("source_") or key in {"prefix_stop", "perturbation_profile", "perturbation_seed", "perturbation_fingerprint", "recovery_kind", "controlled_smoke_teacher_probe"}}
    provenance.update({"source_continuation_snapshot": str(snapshot_path), "source_annotations": str(Path(annotations_value))})
    return ControlledRecovery(
        reset_snapshot,
        actions[:stop],
        continuation_snapshot,
        actions[stop:first_success + 1] if teacher_probe else (),
        continuation,
        profile,
        seed,
        provenance,
    )


def _validate_v2_smoke_descriptor(row: Mapping[str, object]) -> None:
    """Admit only the immutable one-row descriptor produced by the smoke wrapper."""

    if row.get("controlled_smoke") is not True:
        raise ValueError("v2 controlled list requires an explicit controlled smoke descriptor")
    run_id = row.get("controlled_smoke_run_id")
    matrix_sha256 = row.get("controlled_smoke_matrix_sha256")
    materialization_sha256 = row.get("controlled_smoke_materialization_sha256")
    row_index = row.get("controlled_smoke_row_index")
    if (not isinstance(run_id, str) or re.fullmatch(r"[0-9a-f]{32}", run_id) is None
            or not isinstance(matrix_sha256, str) or _LOWERCASE_SHA256.fullmatch(matrix_sha256) is None
            or not isinstance(materialization_sha256, str) or _LOWERCASE_SHA256.fullmatch(materialization_sha256) is None
            or type(row_index) is not int or row_index < 0
            or row.get("controlled_matrix_sha256") != matrix_sha256):
        raise ValueError("controlled smoke descriptor lineage is invalid")
    identity = hashlib.sha256(f"{run_id}:{matrix_sha256}:{materialization_sha256}".encode("ascii")).hexdigest()[:20]
    if row.get("controlled_smoke_identity") != identity:
        raise ValueError("controlled smoke descriptor identity is invalid")
    zero, teacher = row.get("controlled_smoke_zero_perturbation"), row.get("controlled_smoke_teacher_probe")
    if type(zero) is not bool or type(teacher) is not bool:
        raise ValueError("controlled smoke descriptor mode is invalid")
    mode = (
        "zero_perturbation_teacher_continuation_probe_v1" if zero else "teacher_continuation_probe_v1"
    ) if teacher else ("zero_perturbation_control_v1" if zero else "bounded_perturbation_v1")
    mode_identity = hashlib.sha256(f"{identity}:{mode}".encode("ascii")).hexdigest()[:20]
    if row.get("controlled_smoke_perturbation_mode") != mode or row.get("controlled_smoke_mode_identity") != mode_identity:
        raise ValueError("controlled smoke descriptor mode identity is invalid")


def _strict_json_value_from_text(text: str, *, field: str) -> object:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{field} has duplicate JSON fields")
            result[key] = value
        return result

    def reject_constant(_: str) -> object:
        raise ValueError("non-finite JSON constant")

    try:
        return json.loads(
            text, object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{field} must be strict JSON") from error


def _strict_json_object(path: Path, *, field: str) -> dict[str, object]:
    try:
        payload = _strict_json_value_from_text(path.read_text(encoding="utf-8"), field=field)
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(f"{field} must be a strict JSON object") from error
    if type(payload) is not dict:
        raise ValueError(f"{field} must be a strict JSON object")
    return payload


def _strict_snapshot(payload: Mapping[str, object]) -> Snapshot:
    required = {
        "schema_version", "robot_position", "robot_velocity", "cloth_position", "cloth_velocity",
        "rng_state", "garment_name", "randomization", "cloth_state_authority",
    }
    if set(payload) not in (required, required | {"scene_state"}):
        raise ValueError("continuation snapshot has an incompatible schema")

    def vector(value: object, *, name: str) -> tuple[float, ...]:
        if not isinstance(value, list) or len(value) != 12 or any(type(item) not in (int, float) or not math.isfinite(float(item)) for item in value):
            raise ValueError(f"continuation snapshot {name} must be finite 12-D")
        return tuple(float(item) for item in value)

    def cloth(value: object, *, name: str) -> tuple[tuple[float, float, float], ...]:
        if not isinstance(value, list) or not value:
            raise ValueError(f"continuation snapshot {name} must be finite N-by-3")
        rows: list[tuple[float, float, float]] = []
        for point in value:
            if not isinstance(point, list) or len(point) != 3 or any(type(item) not in (int, float) or not math.isfinite(float(item)) for item in point):
                raise ValueError(f"continuation snapshot {name} must be finite N-by-3")
            rows.append(tuple(float(item) for item in point))
        return tuple(rows)

    if (type(payload.get("schema_version")) is not int or not isinstance(payload.get("rng_state"), dict)
            or not isinstance(payload.get("garment_name"), str) or not payload["garment_name"]
            or not isinstance(payload.get("randomization"), dict)
            or not isinstance(payload.get("scene_state", {}), dict)):
        raise ValueError("continuation snapshot has an incompatible schema")
    try:
        return Snapshot(
            schema_version=payload["schema_version"],
            robot_position=vector(payload["robot_position"], name="robot_position"),
            robot_velocity=vector(payload["robot_velocity"], name="robot_velocity"),
            cloth_position=cloth(payload["cloth_position"], name="cloth_position"),
            cloth_velocity=cloth(payload["cloth_velocity"], name="cloth_velocity"),
            rng_state=dict(payload["rng_state"]), garment_name=payload["garment_name"],
            randomization=dict(payload["randomization"]), scene_state=dict(payload.get("scene_state", {})),
            cloth_state_authority=str(payload["cloth_state_authority"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("continuation snapshot has an incompatible schema") from error


def _validate_continuation_snapshot_boundary(
    *, continuation_snapshot: Snapshot, reset_snapshot: Snapshot, step: int,
) -> None:
    """Bind the physical snapshot to the exact next-action H16 boundary.

    Policy observations are intentionally separate evidence: the legacy garment
    environment exposes a one-step-lagged joint tensor, while Snapshot capture
    reads the live articulation.  Restore fidelity must therefore use the
    manifest-authenticated physical state, not overwrite it with the lagged
    policy observation.
    """

    randomization = dict(continuation_snapshot.randomization)
    continuation_step = randomization.pop("continuation_step", None)
    if type(continuation_step) is not int or continuation_step != step:
        raise ValueError("continuation snapshot does not bind the exact H16 next-action boundary")
    if randomization != dict(reset_snapshot.randomization):
        raise ValueError("continuation snapshot randomization does not match the authenticated reset")


def _snapshot_source_descriptor_rows(descriptor_path: str | Path) -> list[dict[str, object]]:
    descriptor_file = _strict_absolute_regular_file(descriptor_path, field="snapshot source descriptor")
    try:
        rows = _strict_json_value_from_text(
            descriptor_file.read_text(encoding="utf-8"), field="snapshot source descriptor"
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError("snapshot source descriptor is malformed") from error
    if not isinstance(rows, list) or not all(type(row) is dict for row in rows):
        raise ValueError("snapshot source descriptor must contain JSON object rows")
    return [dict(row) for row in rows]


def _validate_snapshot_source_row(
    row: Mapping[str, object], *, allow_legacy_replay: bool,
    require_canonical_restore: bool = True,
) -> dict[str, object]:
    """Validate source row lineage without accepting unrelated recovery inputs."""

    row = dict(row)
    category, garment = row.get("category"), row.get("garment")
    seed, source_seed = row.get("seed"), row.get("source_seed")
    if (
        row.get("snapshot_source_bootstrap") is not True
        or row.get("recovery_kind") is not None
        or category not in {"top_long", "top_short", "pant_long", "pant_short"}
        or not isinstance(garment, str) or not garment
        or type(seed) is not int or seed < 0
        or type(source_seed) is not int or source_seed < 0 or source_seed != seed
    ):
        raise ValueError("snapshot source descriptor lineage is invalid")

    replay_kind = row.get("replay_kind")
    restore_keys = {
        "restore_snapshot", "restore_snapshot_sha256", "restore_snapshot_cloth_frame",
        "restore_snapshot_step", "parent_episode_id", "lineage_id",
    }
    if replay_kind is None:
        if any(key in row for key in restore_keys):
            raise ValueError("ordinary snapshot source descriptor cannot carry replay state")
        return row
    if not allow_legacy_replay:
        raise ValueError("snapshot source discovery descriptor must be ordinary autonomous collection")
    if replay_kind not in VERIFIED_SUCCESS_REPLAY_KINDS:
        raise ValueError("snapshot source descriptor replay kind is invalid")
    if row.get("parent_episode_id") != row.get("lineage_id") or not isinstance(
        row.get("parent_episode_id"), str
    ) or not row["parent_episode_id"]:
        raise ValueError("snapshot source descriptor replay lineage is invalid")
    cloth_frame = row.get("restore_snapshot_cloth_frame")
    if (
        cloth_frame != LEGACY_USD_LOCAL_CLOTH_AUTHORITY
        if replay_kind == "verified_success_reset_v1"
        else cloth_frame not in {
            LEGACY_USD_LOCAL_CLOTH_AUTHORITY, PHYSX_CLOTH_STATE_AUTHORITY,
        }
    ):
        raise ValueError("snapshot source descriptor cloth frame is invalid")
    restore = _strict_absolute_regular_file(
        row.get("restore_snapshot"), field="snapshot source restore snapshot"
    )
    expected = row.get("restore_snapshot_sha256")
    if not isinstance(expected, str) or _LOWERCASE_SHA256.fullmatch(expected) is None:
        raise ValueError("snapshot source restore snapshot SHA-256 is invalid")
    if _sha256(restore) != expected:
        raise ValueError("snapshot source restore snapshot SHA-256 mismatch")
    payload = _strict_json_object(restore, field="snapshot source restore snapshot")
    required = {
        "schema_version", "robot_position", "robot_velocity", "cloth_position",
        "cloth_velocity", "rng_state", "garment_name", "randomization", "scene_state",
    }
    if replay_kind == "verified_success_early_snapshot_v1":
        required.add("cloth_state_authority")
    schema_version = payload.get("schema_version")
    expected_schema = (
        1
        if replay_kind == "verified_success_reset_v1"
        else 3 if cloth_frame == LEGACY_USD_LOCAL_CLOTH_AUTHORITY else 2
    )
    restore_step = row.get("restore_snapshot_step")
    if (
        set(payload) != required
        or schema_version != expected_schema
        or (
            replay_kind == "verified_success_early_snapshot_v1"
            and (
                restore_step != 16
                or not isinstance(payload.get("randomization"), Mapping)
                or payload["randomization"].get("continuation_step") != restore_step
                or payload.get("cloth_state_authority") != cloth_frame
            )
        )
        or (replay_kind == "verified_success_reset_v1" and "restore_snapshot_step" in row)
    ):
        raise ValueError("snapshot source restore snapshot has an incompatible schema")

    def vector(value: object, *, size: int, name: str) -> tuple[float, ...]:
        if (
            not isinstance(value, list) or len(value) != size
            or any(type(item) not in (int, float) or not math.isfinite(float(item)) for item in value)
        ):
            raise ValueError(f"snapshot source restore snapshot {name} is invalid")
        return tuple(float(item) for item in value)

    def cloth(value: object, *, name: str) -> tuple[tuple[float, float, float], ...]:
        if not isinstance(value, list) or not value:
            raise ValueError(f"snapshot source restore snapshot {name} is invalid")
        rows: list[tuple[float, float, float]] = []
        for point in value:
            if (
                not isinstance(point, list) or len(point) != 3
                or any(type(item) not in (int, float) or not math.isfinite(float(item)) for item in point)
            ):
                raise ValueError(f"snapshot source restore snapshot {name} is invalid")
            rows.append(tuple(float(item) for item in point))
        return tuple(rows)

    try:
        snapshot = Snapshot(
            schema_version=int(schema_version),
            robot_position=vector(payload["robot_position"], size=12, name="robot_position"),
            robot_velocity=vector(payload["robot_velocity"], size=12, name="robot_velocity"),
            cloth_position=cloth(payload["cloth_position"], name="cloth_position"),
            cloth_velocity=cloth(payload["cloth_velocity"], name="cloth_velocity"),
            rng_state=dict(payload["rng_state"]), garment_name=str(payload["garment_name"]),
            randomization=dict(payload["randomization"]), scene_state=dict(payload["scene_state"]),
            cloth_state_authority=payload.get("cloth_state_authority"),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("snapshot source restore snapshot has an incompatible schema") from error
    pose = snapshot.scene_state.get("garment_reset_pose")
    if (
        snapshot.garment_name != garment
        or (
            snapshot.randomization.get("strategy") != "canonical"
            if require_canonical_restore
            else snapshot.randomization.get("strategy")
            not in {"canonical", "mild_geometry", "strong_geometry"}
        )
        or not isinstance(pose, list) or len(pose) != 6
        or any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in pose)
    ):
        raise ValueError("snapshot source restore snapshot cloth frame is invalid")
    return row


def validate_snapshot_source_descriptor(descriptor_path: str | Path) -> dict[str, object]:
    """Validate one legacy-compatible source assignment before simulator start."""

    rows = _snapshot_source_descriptor_rows(descriptor_path)
    if len(rows) != 1:
        raise ValueError("snapshot source descriptor must contain exactly one row")
    return _validate_snapshot_source_row(rows[0], allow_legacy_replay=True)


def validate_success_replay_descriptor(
    descriptor_path: str | Path | Mapping[str, object],
) -> list[dict[str, object]]:
    """Validate one bounded, exact-cap CPU success-replay campaign or row."""

    rows = (
        [dict(descriptor_path)]
        if isinstance(descriptor_path, Mapping)
        else _snapshot_source_descriptor_rows(descriptor_path)
    )
    if not 1 <= len(rows) <= 400:
        raise ValueError("success replay descriptor must contain 1..400 rows")
    allowed_fields = {
        "attempt_id", "trial_id", "garment", "garment_name", "category",
        "release_stage", "difficulty", "seed", "strategy", "restore_snapshot",
        "restore_snapshot_sha256", "restore_snapshot_cloth_frame",
        "parent_episode_id", "lineage_id", "replay_kind",
        "category_acceptance_cap",
    }
    fresh_provenance_fields = {
        "source_episode_sha256", "source_episode_root", "source_episode_path", "source_reset_sha256", "source_annotations_sha256",
        "source_continuation_snapshot_sha256", "source_state_fingerprint",
        "source_report_sha256", "source_matrix_sha256", "source_receipt_sha256", "source_receipt_path",
        "source_remote_prefix", "source_immutable_revision", "source_round_id", "source_run_id",
        "source_report_path", "source_matrix_path",
    }
    attempt_ids: set[str] = set()
    seeds: set[int] = set()
    category_counts: dict[str, int] = {}
    category_caps: dict[str, int] = {}
    fresh_rows = True
    for row in rows:
        is_fresh = fresh_provenance_fields <= set(row)
        expected_fields = (
            allowed_fields | {"restore_snapshot_step"}
            if row.get("replay_kind") == "verified_success_early_snapshot_v1"
            else allowed_fields
        )
        if is_fresh:
            expected_fields |= fresh_provenance_fields
        if set(row) != expected_fields:
            raise ValueError("success replay descriptor row fields are invalid")
        category = row.get("category")
        cap = row.get("category_acceptance_cap")
        if (
            row.get("attempt_id") != row.get("trial_id")
            or not isinstance(row.get("attempt_id"), str)
            or not row["attempt_id"]
            or row["attempt_id"] in attempt_ids
            or row.get("release_stage") != "seen"
            or row.get("garment_name") != row.get("garment")
            or type(cap) is not int
            or not 0 <= cap <= 150
        ):
            raise ValueError("success replay descriptor identity or cap is invalid")
        if is_fresh:
            if (
                row.get("strategy") != "visual_only"
                or cap != 50
                or any(
                    not isinstance(row[field], str)
                    or _LOWERCASE_SHA256.fullmatch(row[field]) is None
                    for field in fresh_provenance_fields
                    - {
                        "source_episode_root", "source_episode_path", "source_report_path", "source_matrix_path", "source_receipt_path",
                        "source_remote_prefix", "source_immutable_revision", "source_round_id", "source_run_id",
                    }
                )
                or not isinstance(row.get("source_round_id"), str)
                or re.fullmatch(r"fresh-12k-[a-z0-9-]{1,112}", row["source_round_id"]) is None
                or not isinstance(row.get("source_run_id"), str)
                or re.fullmatch(r"fresh-run-[a-z0-9-]{1,112}", row["source_run_id"]) is None
                or not isinstance(row.get("source_report_path"), str)
                or not Path(row["source_report_path"]).is_absolute()
                or not isinstance(row.get("source_matrix_path"), str)
                or not Path(row["source_matrix_path"]).is_absolute()
                or not isinstance(row.get("source_episode_path"), str)
                or not Path(row["source_episode_path"]).is_absolute()
                or not isinstance(row.get("source_episode_root"), str)
                or not Path(row["source_episode_root"]).is_absolute()
                or not isinstance(row.get("source_receipt_path"), str)
                or not Path(row["source_receipt_path"]).is_absolute()
                or row.get("source_remote_prefix")
                != f"rollout-rounds/{row['source_round_id']}/{row['parent_episode_id']}"
                or not isinstance(row.get("source_immutable_revision"), str)
                or re.fullmatch(r"[0-9a-f]{40}", row["source_immutable_revision"]) is None
            ):
                raise ValueError("fresh visual-only replay provenance is invalid")
        fresh_rows = fresh_rows and is_fresh
        if category in category_caps and category_caps[str(category)] != cap:
            raise ValueError("success replay category acceptance cap is inconsistent")
        category_caps[str(category)] = cap
        category_counts[str(category)] = category_counts.get(str(category), 0) + 1
        seed = row.get("seed")
        if type(seed) is not int or seed in seeds:
            raise ValueError("success replay descriptor seeds must be unique")
        verification_row = dict(row)
        verification_row["snapshot_source_bootstrap"] = True
        verification_row["source_seed"] = seed
        _validate_snapshot_source_row(
            verification_row,
            allow_legacy_replay=True,
            require_canonical_restore=is_fresh,
        )
        attempt_ids.add(str(row["attempt_id"]))
        seeds.add(seed)
    if any(category_caps[category] > count for category, count in category_counts.items()):
        raise ValueError("success replay category acceptance cap exceeds its attempts")
    if fresh_rows:
        if (
            len(rows) > 400
            or set(category_caps) != set(_CANONICAL_SEEN_GARMENTS)
            or any(category_caps[category] != 50 for category in _CANONICAL_SEEN_GARMENTS)
            or any(category_counts[category] > 100 for category in _CANONICAL_SEEN_GARMENTS)
            or sum(category_caps.values()) != 200
        ):
            raise ValueError("fresh visual-only replay caps must be the exact bounded 200 tuple")
    elif not 1 <= sum(category_caps.values()) <= 150:
        raise ValueError("success replay total acceptance cap must be in 1..150")
    return rows


def validate_hard_state_descriptor(
    descriptor_path: str | Path,
) -> list[dict[str, object]]:
    """Validate one bounded CPU-only moment-of-ruin recovery campaign."""

    rows = _snapshot_source_descriptor_rows(descriptor_path)
    if not 1 <= len(rows) <= 400:
        raise ValueError("hard-state descriptor must contain 1..400 rows")
    allowed_fields = {
        "attempt_id", "trial_id", "garment", "garment_name", "category",
        "release_stage", "difficulty", "seed", "strategy", "restore_snapshot",
        "restore_snapshot_sha256", "restore_snapshot_cloth_frame",
        "restore_snapshot_step", "parent_episode_id", "lineage_id",
        "source_episode_id", "source_episode_path", "replay_kind",
        "category_acceptance_cap", "rank_score", "priority_reasons",
        "selection_profile", "selection_evidence",
    }
    required_snapshot_fields = {
        "schema_version", "robot_position", "robot_velocity", "cloth_position",
        "cloth_velocity", "rng_state", "garment_name", "randomization",
        "scene_state", "cloth_state_authority",
    }
    attempt_ids: set[str] = set()
    seeds: set[int] = set()
    category_counts: dict[str, int] = {}
    category_caps: dict[str, int] = {}
    for row in rows:
        if set(row) != allowed_fields:
            raise ValueError("hard-state descriptor row fields are invalid")
        category = row.get("category")
        garment = row.get("garment")
        attempt_id = row.get("attempt_id")
        seed = row.get("seed")
        cap = row.get("category_acceptance_cap")
        parent = row.get("parent_episode_id")
        rank_score = row.get("rank_score")
        if (
            category not in _CANONICAL_SEEN_GARMENTS
            or not isinstance(garment, str)
            or _CANONICAL_SEEN_GARMENTS[str(category)].fullmatch(garment) is None
            or row.get("garment_name") != garment
            or row.get("trial_id") != attempt_id
            or not isinstance(attempt_id, str) or not attempt_id or attempt_id in attempt_ids
            or type(seed) is not int or seed < 0 or seed in seeds
            or row.get("release_stage") != "seen"
            or row.get("difficulty") != "hard_state"
            or row.get("strategy") != "canonical"
            or row.get("replay_kind") not in VERIFIED_HARD_STATE_KINDS
            or not isinstance(parent, str) or not parent
            or row.get("lineage_id") != parent
            or row.get("source_episode_id") != parent
            or not isinstance(row.get("source_episode_path"), str)
            or not str(row["source_episode_path"]).startswith("/")
            or type(cap) is not int or not 0 <= cap <= 150
            or type(rank_score) not in (int, float) or not math.isfinite(float(rank_score))
            or not isinstance(row.get("priority_reasons"), list)
            or any(not isinstance(reason, str) or not reason for reason in row["priority_reasons"])
            or row.get("selection_profile") != "moment_of_ruin_reward_drop_v1"
            or not isinstance(row.get("selection_evidence"), Mapping)
        ):
            raise ValueError("hard-state descriptor identity or evidence is invalid")
        if category in category_caps and category_caps[str(category)] != cap:
            raise ValueError("hard-state category acceptance cap is inconsistent")

        restore = _strict_absolute_regular_file(
            row.get("restore_snapshot"), field="hard-state restore snapshot"
        )
        expected = row.get("restore_snapshot_sha256")
        if not isinstance(expected, str) or _LOWERCASE_SHA256.fullmatch(expected) is None:
            raise ValueError("hard-state restore snapshot SHA-256 is invalid")
        if _sha256(restore) != expected:
            raise ValueError("hard-state restore snapshot SHA-256 mismatch")
        payload = _strict_json_object(restore, field="hard-state restore snapshot")
        restore_step = row.get("restore_snapshot_step")
        randomization = payload.get("randomization")
        scene_state = payload.get("scene_state")
        pose = scene_state.get("garment_reset_pose") if isinstance(scene_state, Mapping) else None
        moment = row["selection_evidence"].get("moment_of_ruin")
        if (
            set(payload) != required_snapshot_fields
            or payload.get("schema_version") != 3
            or row.get("restore_snapshot_cloth_frame") != LEGACY_USD_LOCAL_CLOTH_AUTHORITY
            or payload.get("cloth_state_authority") != LEGACY_USD_LOCAL_CLOTH_AUTHORITY
            or payload.get("garment_name") != garment
            or type(restore_step) is not int or restore_step <= 0 or restore_step % 16
            or not isinstance(randomization, Mapping)
            or randomization.get("strategy") != "canonical"
            or randomization.get("continuation_step") != restore_step
            or not isinstance(pose, list) or len(pose) != 6
            or any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in pose)
            or not isinstance(moment, Mapping)
            or moment.get("restore_step") != restore_step
        ):
            raise ValueError("hard-state restore snapshot has an incompatible CPU schema")
        for field, size in (("robot_position", 12), ("robot_velocity", 12)):
            values = payload.get(field)
            if (
                not isinstance(values, list) or len(values) != size
                or any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in values)
            ):
                raise ValueError(f"hard-state restore snapshot {field} is invalid")
        for field in ("cloth_position", "cloth_velocity"):
            values = payload.get(field)
            if (
                not isinstance(values, list) or not values
                or any(
                    not isinstance(point, list) or len(point) != 3
                    or any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in point)
                    for point in values
                )
            ):
                raise ValueError(f"hard-state restore snapshot {field} is invalid")
        attempt_ids.add(attempt_id)
        seeds.add(seed)
        category_counts[str(category)] = category_counts.get(str(category), 0) + 1
        category_caps[str(category)] = cap
    if any(category_caps[category] > category_counts[category] for category in category_caps):
        raise ValueError("hard-state category acceptance cap exceeds its attempts")
    if not 1 <= sum(category_caps.values()) <= 150:
        raise ValueError("hard-state total acceptance cap must be in 1..150")
    return rows


def validate_snapshot_source_discovery_descriptor(
    descriptor_path: str | Path,
) -> list[dict[str, object]]:
    """Validate a bounded ordinary source-discovery descriptor.

    Historical replay and controlled-recovery fields are excluded: this creates
    only fresh, autonomous schema-v2 sources. Mixed categories are allowed so a
    production four-worker wave can fill exact category deficits in one run.
    """

    rows = _snapshot_source_descriptor_rows(descriptor_path)
    if not 1 <= len(rows) <= 400:
        raise ValueError("snapshot source discovery descriptor must contain 1..400 rows")
    validated = [_validate_snapshot_source_row(row, allow_legacy_replay=False) for row in rows]
    seeds: set[int] = set()
    allowed_fields = {
        "snapshot_source_bootstrap", "snapshot_source_descriptor_sha256",
        "category", "garment", "garment_name", "seed", "source_seed",
    }
    for row in validated:
        category = row["category"]
        garment_name = row.get("garment_name")
        if (garment_name is not None and garment_name != row["garment"]):
            raise ValueError("snapshot source discovery descriptor garment identity is inconsistent")
        if _CANONICAL_SEEN_GARMENTS[str(category)].fullmatch(str(row["garment"])) is None:
            raise ValueError("snapshot source discovery descriptor garment identity is not a canonical seen garment")
        seed = row["seed"]
        if seed in seeds:
            raise ValueError("snapshot source discovery descriptor seeds must be unique")
        seeds.add(seed)
        if any(key not in allowed_fields for key in row):
            raise ValueError("snapshot source discovery descriptor must be ordinary autonomous collection")
    return validated


def validate_snapshot_source_bootstrap_evidence(
    *, accepted_root: str | Path, descriptor_path: str | Path | None = None,
    descriptor_row: Mapping[str, object] | None = None,
) -> tuple[int, ...]:
    """Validate one ordinary autonomous source before a source-only envelope.

    The bootstrap wrapper calls this shared collection-side gate after receipt
    readback.  It intentionally refuses partial snapshot evidence: a later
    recovery audit may only select a full H=16 physical boundary that is
    manifest-authenticated and tied back to the descriptor/episode/annotation
    identity.
    """

    if (descriptor_path is None) == (descriptor_row is None):
        raise ValueError("snapshot source evidence requires exactly one descriptor identity")
    if descriptor_path is not None:
        descriptor_row = validate_snapshot_source_descriptor(descriptor_path)
    else:
        assert descriptor_row is not None
        descriptor_row = _validate_snapshot_source_row(descriptor_row, allow_legacy_replay=True)
    category, garment, seed = descriptor_row.get("category"), descriptor_row.get("garment"), descriptor_row.get("seed")
    accepted = Path(accepted_root)
    if not accepted.is_absolute() or accepted.is_symlink() or not accepted.is_dir():
        raise ValueError("snapshot source accepted root is unsafe")
    raw = accepted / "raw" / accepted.name
    manifest_path = _strict_absolute_regular_file(raw / "SHA256SUMS.json", field="snapshot source checksum manifest")
    _strict_absolute_regular_file(raw / "episode.json", field="snapshot source episode")
    annotations_path = _strict_absolute_regular_file(raw / "annotations.jsonl", field="snapshot source annotations")
    reset_path = _strict_absolute_regular_file(raw / "snapshots" / "reset.json", field="snapshot source reset")
    try:
        from lehome.flywheel.artifacts import verify_episode_manifest
        episode, manifest = verify_episode_manifest(raw)
    except (ImportError, ValueError) as error:
        raise ValueError("snapshot source checksum manifest does not authenticate the accepted episode") from error
    del manifest_path, manifest
    if not isinstance(episode, Mapping):
        raise ValueError("snapshot source episode is malformed")
    identity = episode.get("identity")
    if (episode.get("episode_id") != accepted.name or episode.get("mode") != "autonomous"
            or episode.get("accepted_success") is not True or episode.get("outcome") != "success"
            or episode.get("terminal_reason") != "success" or episode.get("bc_target_count") != 0
            or not isinstance(identity, Mapping) or identity.get("category") != category
            or identity.get("garment_name") != garment or identity.get("seed") != seed):
        raise ValueError("snapshot source episode is not an accepted autonomous descriptor-bound success")
    reset_payload = _strict_json_object(reset_path, field="snapshot source reset")
    reset = _strict_snapshot(reset_payload)
    if reset.garment_name != garment:
        raise ValueError("snapshot source reset garment does not match the descriptor")
    annotation_rows: list[dict[str, object]] = []
    try:
        for line in annotations_path.read_text(encoding="utf-8").splitlines():
            value = _strict_json_value_from_text(line, field="snapshot source annotation")
            if type(value) is not dict:
                raise ValueError("snapshot source annotation must be a strict JSON object")
            annotation_rows.append(value)
    except (OSError, UnicodeError) as error:
        raise ValueError("snapshot source annotations are unreadable") from error
    if not annotation_rows:
        raise ValueError("snapshot source annotations are empty")
    request_ids: set[str] = set()
    for index, row in enumerate(annotation_rows):
        state, action, success = row.get("state"), row.get("action"), row.get("success")
        if (row.get("step") != index or row.get("action_source") != "policy"
                or not isinstance(row.get("policy_request_id"), str) or not row["policy_request_id"]
                or type(row.get("policy_chunk_offset")) is not int or row["policy_chunk_offset"] != index % 16
                or not isinstance(state, list) or len(state) != 12
                or not isinstance(action, list) or len(action) != 12
                or type(success) is not bool or type(row.get("reward")) not in (int, float)
                or not math.isfinite(float(row["reward"]))
                or any(type(item) not in (int, float) or not math.isfinite(float(item)) for item in (*state, *action))
                or row.get("category") != category or row.get("garment_name") != garment or row.get("seed") != seed):
            raise ValueError("snapshot source annotations do not satisfy the autonomous H16 trace contract")
        if index % 16 == 0:
            if row["policy_request_id"] in request_ids:
                raise ValueError("snapshot source annotations reuse a policy request")
            request_ids.add(row["policy_request_id"])
        elif row["policy_request_id"] != annotation_rows[index - 1]["policy_request_id"]:
            raise ValueError("snapshot source annotations do not preserve a 16-row policy chunk")
    first_success = next((index for index, row in enumerate(annotation_rows) if row["success"]), None)
    if first_success is None:
        raise ValueError("snapshot source annotations lack an official first success")
    if any(not row["success"] for row in annotation_rows[first_success:]):
        raise ValueError("snapshot source annotations do not latch success through terminal")
    directory = raw / "snapshots" / "continuations"
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("snapshot source has no continuation snapshots")
    continuations = sorted(directory.iterdir())
    if not continuations:
        raise ValueError("snapshot source has no continuation snapshots")
    steps: list[int] = []
    for path in continuations:
        if path.is_dir() or path.suffix != ".json":
            raise ValueError("snapshot source continuation directory contains an unexpected path")
        strict_path = _strict_absolute_regular_file(path, field="snapshot source continuation")
        match = re.fullmatch(r"([0-9]{6})\.json", strict_path.name)
        if match is None:
            raise ValueError("snapshot source continuation filename is not a six-digit H16 boundary")
        step = int(match.group(1))
        if step <= 0 or step % 16 or step >= first_success or step >= len(annotation_rows):
            raise ValueError("snapshot source continuation boundary is invalid")
        snapshot_payload = _strict_json_object(strict_path, field="snapshot source continuation")
        snapshot = _strict_snapshot(snapshot_payload)
        _source_snapshot_contract(reset, snapshot)
        row = annotation_rows[step]
        state = row.get("state")
        if (row.get("step") != step or row.get("action_source") != "policy" or row.get("policy_chunk_offset") != 0
                or row.get("category") != category or row.get("garment_name") != garment or row.get("seed") != seed
                or not isinstance(state, list) or len(state) != 12
                or any(type(item) not in (int, float) or not math.isfinite(float(item)) for item in state)
                or snapshot.garment_name != garment):
            raise ValueError("snapshot source continuation does not match the authenticated H16 annotation")
        _validate_continuation_snapshot_boundary(
            continuation_snapshot=snapshot, reset_snapshot=reset, step=step,
        )
        steps.append(step)
    return tuple(steps)


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
        v2_rows = [row for row in decoded if row.get("recovery_kind") == RECOVERY_KIND]
        if v2_rows:
            if len(decoded) != 1 or len(v2_rows) != 1:
                raise ValueError("v2 controlled smoke list must contain exactly one row")
            _validate_v2_smoke_descriptor(v2_rows[0])
            return [dict(v2_rows[0])]
        for row in decoded:
            capped_verified_replay = (
                row.get("replay_kind") in VERIFIED_SUCCESS_REPLAY_KINDS | VERIFIED_HARD_STATE_KINDS
                and "category_acceptance_cap" in row
            )
            controlled = (
                "recovery_kind" in row
                or any(str(key).startswith("source_continuation_") for key in row)
                or ("category_acceptance_cap" in row and not capped_verified_replay)
            )
            if controlled and row.get("recovery_kind") != RECOVERY_KIND:
                raise ValueError("legacy or incompatible controlled recovery list is forbidden")
        return [dict(row) for row in decoded]
    envelope = {"schema_version", "kind", "matrix_sha256", "target_accepted", "category_acceptance_caps", "rows"}
    if not isinstance(decoded, Mapping) or set(decoded) != envelope:
        raise ValueError("attempt matrix must be a JSON array or controlled materialization")
    matrix_sha256 = decoded.get("matrix_sha256")
    rows, target, caps = decoded.get("rows"), decoded.get("target_accepted"), decoded.get("category_acceptance_caps")
    if decoded.get("schema_version") != 3 or decoded.get("kind") != "controlled_success_recovery_materialization_v3":
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
        for field in ("source_episode_digest", "source_reset_sha256", "source_annotations_sha256", "source_continuation_snapshot_sha256", "source_state_fingerprint"):
            if not isinstance(row.get(field), str) or _LOWERCASE_SHA256.fullmatch(str(row[field])) is None:
                raise ValueError("controlled materialization source identity is invalid")
        if (
            row.get("source_snapshot_schema_version") not in (2, 3)
            or row.get("source_snapshot_authority") not in {
                PHYSX_CLOTH_STATE_AUTHORITY,
                LEGACY_USD_LOCAL_CLOTH_AUTHORITY,
            }
            or (row["source_snapshot_schema_version"] == 2 and row["source_snapshot_authority"] != PHYSX_CLOTH_STATE_AUTHORITY)
            or (row["source_snapshot_schema_version"] == 3 and row["source_snapshot_authority"] != LEGACY_USD_LOCAL_CLOTH_AUTHORITY)
        ):
            raise ValueError("controlled materialization source snapshot authority is invalid")
        if type(row.get("source_only_envelope")) is not bool or (
            row["source_snapshot_schema_version"] == 3 and row["source_only_envelope"] is not True
        ):
            raise ValueError("controlled materialization source envelope is invalid")
        if type(row.get("source_seed")) is not int or row["source_seed"] < 0:
            raise ValueError("controlled materialization source reset seed is invalid")
        continuation = row.get("source_continuation_state")
        if (not isinstance(continuation, list) or len(continuation) != 12
                or any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in continuation)):
            raise ValueError("controlled materialization continuation state is invalid")
        if not isinstance(row.get("source_round_id"), str) or not row["source_round_id"] or not isinstance(row.get("source_episode_id"), str) or not row["source_episode_id"]:
            raise ValueError("controlled materialization source identity is invalid")
        if (type(row.get("prefix_stop")) is not int or type(row.get("source_first_success_step")) is not int
                or not 0 < row["prefix_stop"] < row["source_first_success_step"] or row["prefix_stop"] % 16):
            raise ValueError("controlled materialization continuation boundary is invalid")
        for field in ("source_reset", "source_annotations", "source_continuation_snapshot"):
            value = row.get(field)
            try:
                _strict_absolute_regular_file(value, field="controlled materialization source path")
            except ValueError as error:
                raise ValueError("controlled materialization source path is unsafe") from error
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
        cloth_state_authority=value.get("cloth_state_authority"),
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
    return Snapshot(source.schema_version, tuple(joints), source.robot_velocity, displaced, velocities, source.rng_state, source.garment_name, source.randomization, source.scene_state, source.cloth_state_authority)


def bootstrap_controlled_recovery(
    env: object,
    assignment: Mapping[str, object],
    *,
    reset_callback: Callable[[], None] | None = None,
) -> Mapping[str, object]:
    """Authenticate a source H16 boundary, teacher-check, then perturb."""

    from lehome.flywheel.snapshots import capture_snapshot, restore_snapshot

    recovery = load_controlled_recovery(assignment)
    runtime_contract = _runtime_cloth_contract(env)
    if runtime_contract[0] == 3:
        if not callable(reset_callback):
            raise ValueError("controlled recovery CPU reset-prefix reconstruction requires a deterministic reset callback")
        restore_snapshot(env, recovery.reset_snapshot)
        reset_checks = [_replay_fidelity(env, recovery.reset_snapshot)]
        replay_action_prefix(env, recovery.prefix_actions)
        h16_checks = [_replay_fidelity(env, recovery.continuation_snapshot)]
        teacher_provenance: Mapping[str, object] | None = None
        if recovery.teacher_actions:
            replay_action_prefix(env, recovery.teacher_actions)
            if not _teacher_success(env):
                raise ValueError("controlled recovery teacher probe did not reproduce source success")
            teacher_provenance = {
                "enabled": True,
                "verified": True,
                "replayed_action_count": len(recovery.teacher_actions),
            }
        reset_callback()
        restore_snapshot(env, recovery.reset_snapshot)
        reset_checks.append(_replay_fidelity(env, recovery.reset_snapshot))
        replay_action_prefix(env, recovery.prefix_actions)
        h16_checks.append(_replay_fidelity(env, recovery.continuation_snapshot))
        intermediate = capture_snapshot(
            env, randomization={"strategy": "canonical", "recovery_kind": RECOVERY_KIND},
        )
        perturbed = apply_controlled_perturbation(
            intermediate, recovery.perturbation_profile, recovery.perturbation_seed,
        )
        restore_snapshot(env, perturbed)
        capture_snapshot(
            env, randomization={"strategy": "canonical", "recovery_kind": RECOVERY_KIND},
        )
        provenance = dict(recovery.provenance)
        provenance.update({
            "replay_fidelity": h16_checks[-1],
            "replay_fidelity_checks": [
                reset_checks[0], h16_checks[0], reset_checks[1], h16_checks[1],
            ],
            "cpu_reset_prefix_reconstruction": {
                "verified": True,
                "reset_fidelity_checks": reset_checks,
                "h16_fidelity_checks": h16_checks,
                "prefix_replayed_action_count": len(recovery.prefix_actions),
                "authenticated_reset_snapshot_restore": {
                    "count": 2,
                    "path": recovery.provenance["source_reset"],
                },
            },
        })
        if teacher_provenance is not None:
            provenance["teacher_probe"] = teacher_provenance
        return provenance

    restore_snapshot(env, recovery.continuation_snapshot)
    projected = capture_snapshot(
        env, randomization={"strategy": "canonical", "recovery_kind": RECOVERY_KIND},
    )
    if (projected.schema_version, projected.cloth_state_authority) != runtime_contract:
        raise ValueError("controlled recovery restore did not read back the expected cloth authority")
    legacy_projection = (
        _projection_provenance(env, recovery, projected)
        if recovery.continuation_snapshot.schema_version == 3 and runtime_contract[0] == 2
        else None
    )
    replay_target = (
        projected
        if recovery.continuation_snapshot.schema_version == 3
        else recovery.continuation_snapshot
    )
    checks = [_replay_fidelity(env, replay_target)]
    if legacy_projection is not None:
        restore_snapshot(env, recovery.continuation_snapshot)
        repeated = capture_snapshot(
            env, randomization={"strategy": "canonical", "recovery_kind": RECOVERY_KIND},
        )
        if _projection_provenance(env, recovery, repeated) != legacy_projection:
            raise ValueError("controlled recovery legacy USD projection is not idempotent")
        checks.append(_replay_fidelity(env, replay_target))
    teacher_provenance: Mapping[str, object] | None = None
    if recovery.teacher_actions:
        replay_action_prefix(env, recovery.teacher_actions)
        if not _teacher_success(env):
            raise ValueError("controlled recovery teacher probe did not reproduce source success")
        restore_snapshot(env, recovery.continuation_snapshot)
        checks.append(_replay_fidelity(env, replay_target))
        teacher_provenance = {"enabled": True, "verified": True, "replayed_action_count": len(recovery.teacher_actions)}
    intermediate = capture_snapshot(env, randomization={"strategy": "canonical", "recovery_kind": RECOVERY_KIND})
    perturbed = apply_controlled_perturbation(intermediate, recovery.perturbation_profile, recovery.perturbation_seed)
    restore_snapshot(env, perturbed)
    # Capture forces an adapter readback when available and is the state the
    # autonomous recorder must treat as its reset snapshot.
    capture_snapshot(env, randomization={"strategy": "canonical", "recovery_kind": RECOVERY_KIND})
    provenance = dict(recovery.provenance)
    provenance.update({"replay_fidelity": checks[-1], "replay_fidelity_checks": checks})
    if legacy_projection is not None:
        provenance["legacy_usd_projection"] = legacy_projection
    if teacher_provenance is not None:
        provenance["teacher_probe"] = teacher_provenance
    return provenance


__all__ = ["RECOVERY_KIND", "ControlledRecovery", "apply_controlled_perturbation", "bootstrap_controlled_recovery", "load_attempt_matrix", "load_controlled_recovery", "replay_action_prefix", "validate_hard_state_descriptor"]
