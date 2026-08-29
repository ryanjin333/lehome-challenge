"""Isolated raw-mesh public96 success checker for the N1.7 evaluator only."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Iterable, Sequence


RAW_CHECKER_OVERLAY_ID = "lehome_groot_n17_public96_raw_checker_v1"


class RawCheckerError(ValueError):
    """The public raw scoring inputs cannot be proven safe or complete."""


def overlay_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _raw_points(particle_object: Any, indices: Iterable[object]) -> list[tuple[float, float, float]]:
    try:
        mesh_tuple = particle_object.get_current_mesh_points()
    except Exception as error:  # the public96 boundary has no transformed fallback
        raise RawCheckerError("raw mesh tuple is unavailable") from error
    if not isinstance(mesh_tuple, tuple) or len(mesh_tuple) != 4:
        raise RawCheckerError("raw mesh tuple must contain transformed and raw mesh points")
    raw_mesh_points = mesh_tuple[1]
    try:
        points = list(raw_mesh_points)
    except TypeError as error:
        raise RawCheckerError("raw mesh points are not iterable") from error
    normalized: list[tuple[float, float, float]] = []
    for point in points:
        try:
            xyz = tuple(float(value) for value in point)
        except (TypeError, ValueError) as error:
            raise RawCheckerError("raw mesh point is not a numeric 3-vector") from error
        if len(xyz) != 3 or not all(math.isfinite(value) for value in xyz):
            raise RawCheckerError("raw mesh points must be finite 3-vectors")
        normalized.append(xyz)
    resolved: list[int] = []
    for index in indices:
        if type(index) is not int or not 0 <= index < len(normalized):
            raise RawCheckerError("public success checkpoint indices are invalid for raw mesh points")
        resolved.append(index)
    if not resolved:
        raise RawCheckerError("public success checker has no checkpoint indices")
    return [normalized[index] for index in resolved]


def _centimeters(points: Sequence[tuple[float, float, float]]) -> list[list[float]]:
    return [[coordinate * 100.0 for coordinate in point] for point in points]


def _distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(first, second, strict=True)))


def _raw_thresholds(particle_object: Any) -> list[float]:
    values = getattr(particle_object, "success_distance", None)
    if not isinstance(values, (list, tuple)) or not values:
        raise RawCheckerError("public raw success distances are unavailable")
    try:
        thresholds = [float(value) for value in values]
    except (TypeError, ValueError) as error:
        raise RawCheckerError("public raw success distances are invalid") from error
    if not all(math.isfinite(value) and value >= 0 for value in thresholds):
        raise RawCheckerError("public raw success distances must be finite and non-negative")
    return thresholds


def _check(points: Sequence[Sequence[float]], garment_type: str, thresholds: Sequence[float]) -> bool:
    if garment_type in {"top-long-sleeve", "top-short-sleeve"}:
        if len(thresholds) != 5 or len(points) < 6:
            raise RawCheckerError("top public success checker inputs are incomplete")
        return (
            _distance(points[0], points[4]) <= thresholds[0]
            and _distance(points[2], points[3]) <= thresholds[1]
            and _distance(points[1], points[5]) <= thresholds[2]
            and _distance(points[0], points[1]) >= thresholds[3]
            and _distance(points[4], points[5]) >= thresholds[4]
        )
    if garment_type == "long-pant":
        if len(thresholds) != 4 or len(points) < 6:
            raise RawCheckerError("long-pant public success checker inputs are incomplete")
        return (
            _distance(points[0], points[4]) <= thresholds[0]
            and _distance(points[0], points[2]) >= thresholds[1]
            and _distance(points[1], points[3]) >= thresholds[2]
            and _distance(points[1], points[5]) <= thresholds[3]
        )
    if garment_type == "short-pant":
        if len(thresholds) != 4 or len(points) < 6:
            raise RawCheckerError("short-pant public success checker inputs are incomplete")
        return (
            _distance(points[0], points[1]) <= thresholds[0]
            and _distance(points[4], points[5]) <= thresholds[1]
            and _distance(points[0], points[4]) >= thresholds[2]
            and _distance(points[1], points[5]) >= thresholds[3]
        )
    raise RawCheckerError(f"unknown public garment type: {garment_type}")


def raw_success_checker(particle_object: Any, garment_type: str) -> dict[str, object]:
    """Score only the raw mesh tuple's second value and unscaled thresholds."""
    points = _centimeters(_raw_points(particle_object, getattr(particle_object, "check_points", ())))
    thresholds = _raw_thresholds(particle_object)
    return {
        "success": _check(points, garment_type, thresholds),
        "garment_type": garment_type,
        "thresholds": thresholds,
        "details": {},
        "metadata_valid": True,
        "mesh_source": "raw_mesh_points",
        "threshold_scale": 1.0,
        "overlay_id": RAW_CHECKER_OVERLAY_ID,
        "overlay_sha256": overlay_sha256(),
    }


def install_raw_checker_overlay() -> dict[str, str]:
    """Patch only this process before the task module imports the checker."""
    from lehome.utils import success_checker_chanllege as checker

    def raw_position(particle_object: Any, index_list: Iterable[object]):
        points = _raw_points(particle_object, index_list)
        return _centimeters(points)

    checker.get_object_particle_position = raw_position
    checker.success_checker_garment_fold_unthrottled = raw_success_checker
    checker.success_checker_garment_fold = raw_success_checker
    return {"overlay_id": RAW_CHECKER_OVERLAY_ID, "overlay_sha256": overlay_sha256()}
