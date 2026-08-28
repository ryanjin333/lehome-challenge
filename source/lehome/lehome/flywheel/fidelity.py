"""Immutable first-100 fidelity receipt contract."""

from __future__ import annotations

import math
from typing import Mapping


CLOTH_FIDELITY_CODES = (
    "missing_cloth", "cloth_flight", "nonfinite_cloth_state",
)
FIDELITY_CODES = frozenset(CLOTH_FIDELITY_CODES) | {"safety_failure"}
FIDELITY_FIELDS = FIDELITY_CODES | {"monitor_active", "monitor_observed"}
FIDELITY_DIAGNOSTIC_STAGES = frozenset({
    "initialization_write_readback",
    "cached_reset_velocity",
    "reset_write_readback",
    "post_stabilization",
    "policy_step",
})
_PHYSICAL_METRICS = (
    "max_position_m", "max_extent_m", "max_velocity_mps",
)
_POLICY_JOINT_NAMES = frozenset({
    "left_shoulder_pan", "left_shoulder_lift", "left_elbow_flex",
    "left_wrist_flex", "left_wrist_roll", "left_gripper",
    "right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
    "right_wrist_flex", "right_wrist_roll", "right_gripper",
})
_MAX_DIAGNOSTIC_STEP_INDEX = 1_000_000
_POLICY_INTEGER_FIELDS = frozenset({
    "policy_action_dimension", "policy_action_nonfinite_count",
    "policy_action_outside_live_joint_limit_count",
    "policy_action_steps_outside_live_joint_limits",
    "policy_action_max_outside_live_joint_limit_count",
    "policy_action_total_steps",
})
_POLICY_MAP_INTEGER_FIELDS = frozenset({
    "policy_action_outside_live_joint_limit_step_counts",
})
_POLICY_MAP_FLOAT_FIELDS = frozenset({
    "policy_action_max_limit_violation_rad",
    "policy_action_max_target_to_live_joint_position_delta_rad",
})
_POLICY_ACTION_FIELDS = (
    {"policy_action_limits_available", "policy_action_joint_diagnostics"}
    | _POLICY_INTEGER_FIELDS | _POLICY_MAP_INTEGER_FIELDS | _POLICY_MAP_FLOAT_FIELDS
)


def _finite_float(value: object, *, field: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise ValueError(f"fidelity diagnostic {field} must be a finite non-negative float")
    return value


def _bounded_integer(value: object, *, field: str) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_DIAGNOSTIC_STEP_INDEX:
        raise ValueError(f"fidelity diagnostic {field} must be a bounded non-negative integer")
    return value


def _validate_joint_scalar_map(value: object, *, field: str, integer: bool) -> dict[str, object]:
    if not isinstance(value, Mapping) or len(value) > len(_POLICY_JOINT_NAMES):
        raise ValueError(f"fidelity diagnostic {field} must be a bounded mapping")
    if any(name not in _POLICY_JOINT_NAMES for name in value):
        raise ValueError(f"fidelity diagnostic {field} has an unknown joint")
    return {
        name: (
            _bounded_integer(raw, field=field)
            if integer else _finite_float(raw, field=field)
        )
        for name, raw in sorted(value.items())
    }


def _validate_policy_action(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or not value or not set(value) <= _POLICY_ACTION_FIELDS:
        raise ValueError("fidelity diagnostic policy_action has invalid fields")
    result: dict[str, object] = {}
    for field, raw in value.items():
        if field == "policy_action_limits_available":
            if type(raw) is not bool:
                raise ValueError("fidelity diagnostic policy action availability must be boolean")
            result[field] = raw
        elif field in _POLICY_INTEGER_FIELDS:
            result[field] = _bounded_integer(raw, field=field)
        elif field in _POLICY_MAP_INTEGER_FIELDS:
            result[field] = _validate_joint_scalar_map(raw, field=field, integer=True)
        elif field in _POLICY_MAP_FLOAT_FIELDS:
            result[field] = _validate_joint_scalar_map(raw, field=field, integer=False)
        elif field == "policy_action_joint_diagnostics":
            if not isinstance(raw, Mapping) or len(raw) > len(_POLICY_JOINT_NAMES):
                raise ValueError("fidelity diagnostic joint diagnostics must be bounded")
            if any(name not in _POLICY_JOINT_NAMES for name in raw):
                raise ValueError("fidelity diagnostic joint diagnostics has an unknown joint")
            joints: dict[str, object] = {}
            expected = {
                "target_finite", "outside_live_joint_limit", "limit_violation_rad",
                "target_to_live_joint_position_delta_rad",
            }
            for name, joint in sorted(raw.items()):
                if not isinstance(joint, Mapping) or set(joint) != expected:
                    raise ValueError("fidelity diagnostic joint entry has invalid fields")
                if type(joint["target_finite"]) is not bool or type(joint["outside_live_joint_limit"]) is not bool:
                    raise ValueError("fidelity diagnostic joint flags must be boolean")
                joints[name] = {
                    "target_finite": joint["target_finite"],
                    "outside_live_joint_limit": joint["outside_live_joint_limit"],
                    "limit_violation_rad": _finite_float(
                        joint["limit_violation_rad"], field="limit_violation_rad",
                    ),
                    "target_to_live_joint_position_delta_rad": _finite_float(
                        joint["target_to_live_joint_position_delta_rad"],
                        field="target_to_live_joint_position_delta_rad",
                    ),
                }
            result[field] = joints
    return {field: result[field] for field in sorted(result)}


def validate_fidelity_diagnostic(diagnostic: object) -> dict[str, object]:
    """Validate the small closed schema allowed in durable fidelity events."""

    allowed = {"stage", "step_index", "physical_health", "write_readback", "cached_reset_velocity", "policy_action"}
    if not isinstance(diagnostic, Mapping) or not set(diagnostic) <= allowed:
        raise ValueError("fidelity diagnostic has invalid fields")
    stage = diagnostic.get("stage")
    if stage not in FIDELITY_DIAGNOSTIC_STAGES:
        raise ValueError("fidelity diagnostic stage is invalid")
    if stage == "policy_step":
        if "step_index" not in diagnostic:
            raise ValueError("fidelity diagnostic policy step requires an index")
    elif "step_index" in diagnostic:
        raise ValueError("fidelity diagnostic step index is invalid for this stage")
    if "write_readback" in diagnostic and stage not in {
        "initialization_write_readback", "reset_write_readback",
    }:
        raise ValueError("fidelity diagnostic write readback stage is invalid")
    if "cached_reset_velocity" in diagnostic and stage != "cached_reset_velocity":
        raise ValueError("fidelity diagnostic cached reset stage is invalid")
    if "policy_action" in diagnostic and stage != "policy_step":
        raise ValueError("fidelity diagnostic policy action stage is invalid")

    result: dict[str, object] = {"stage": stage}
    if "step_index" in diagnostic:
        result["step_index"] = _bounded_integer(diagnostic["step_index"], field="step_index")
    if "physical_health" in diagnostic:
        physical = diagnostic["physical_health"]
        expected = {
            "max_position_m", "max_extent_m", "max_velocity_mps",
            "max_position_limit_m", "max_extent_limit_m", "max_velocity_limit_mps",
            "exceeded_metrics",
        }
        if not isinstance(physical, Mapping) or set(physical) != expected:
            raise ValueError("fidelity diagnostic physical health has invalid fields")
        exceeded = physical["exceeded_metrics"]
        if (
            not isinstance(exceeded, (list, tuple)) or not 1 <= len(exceeded) <= len(_PHYSICAL_METRICS)
            or len(set(exceeded)) != len(exceeded)
            or any(metric not in _PHYSICAL_METRICS for metric in exceeded)
        ):
            raise ValueError("fidelity diagnostic exceeded metrics are invalid")
        result["physical_health"] = {
            field: _finite_float(physical[field], field=field)
            for field in (
                "max_position_m", "max_extent_m", "max_velocity_mps",
                "max_position_limit_m", "max_extent_limit_m", "max_velocity_limit_mps",
            )
        } | {"exceeded_metrics": list(exceeded)}
    if "write_readback" in diagnostic:
        readback = diagnostic["write_readback"]
        expected = {"max_position_delta_m", "max_velocity_delta_mps"}
        if not isinstance(readback, Mapping) or set(readback) != expected:
            raise ValueError("fidelity diagnostic write readback has invalid fields")
        result["write_readback"] = {
            field: _finite_float(readback[field], field=field) for field in sorted(expected)
        }
    if "cached_reset_velocity" in diagnostic:
        cached = diagnostic["cached_reset_velocity"]
        expected = {"max_velocity_mps", "max_velocity_limit_mps"}
        if not isinstance(cached, Mapping) or set(cached) != expected:
            raise ValueError("fidelity diagnostic cached reset velocity has invalid fields")
        result["cached_reset_velocity"] = {
            field: _finite_float(cached[field], field=field) for field in sorted(expected)
        }
    if "policy_action" in diagnostic:
        result["policy_action"] = _validate_policy_action(diagnostic["policy_action"])
    return result


def fidelity_receipt(
    *,
    missing_cloth: bool,
    cloth_flight: bool,
    nonfinite_cloth_state: bool,
    safety_failure: bool,
    monitor_active: bool,
    monitor_observed: bool,
) -> dict[str, bool]:
    """Build the sole exact, monitored six-field fidelity receipt."""

    return validate_fidelity({
        "missing_cloth": missing_cloth,
        "cloth_flight": cloth_flight,
        "nonfinite_cloth_state": nonfinite_cloth_state,
        "safety_failure": safety_failure,
        "monitor_active": monitor_active,
        "monitor_observed": monitor_observed,
    })


def validate_fidelity(
    fidelity: object,
    *,
    code: str | None = None,
    require_monitors: bool = True,
) -> dict[str, bool]:
    """Return the exact six-field receipt or reject unauthenticated evidence."""

    if not isinstance(fidelity, Mapping) or set(fidelity) != FIDELITY_FIELDS:
        raise ValueError("fidelity receipt is incomplete")
    if any(type(fidelity[field]) is not bool for field in FIDELITY_FIELDS):
        raise ValueError("fidelity receipt must contain booleans")
    if code is not None and (code not in FIDELITY_CODES or fidelity[code] is not True):
        raise ValueError("fidelity code is invalid")
    if require_monitors and (
        fidelity["monitor_active"] is not True
        or fidelity["monitor_observed"] is not True
    ):
        raise ValueError("fidelity monitors must be active and observed")
    return {field: fidelity[field] for field in sorted(FIDELITY_FIELDS)}


class ClothFidelityError(RuntimeError):
    """A measured physical-cloth failure with its immutable receipt."""

    def __init__(
        self, code: str, fidelity: Mapping[str, object], *,
        detail: str | None = None, diagnostic: Mapping[str, object] | None = None,
    ) -> None:
        if code not in CLOTH_FIDELITY_CODES:
            raise ValueError("cloth fidelity code is invalid")
        self.code = code
        self.fidelity = validate_fidelity(fidelity, code=code)
        self.diagnostic = (
            validate_fidelity_diagnostic(diagnostic) if diagnostic is not None else None
        )
        super().__init__(detail or code)
