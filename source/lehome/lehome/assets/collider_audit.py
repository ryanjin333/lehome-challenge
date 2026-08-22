"""Fail-closed USD evidence for unsupported dynamic triangle-mesh colliders.

PhysX rejects triangle-mesh collision shapes on non-kinematic dynamic rigid
bodies.  The value ``none`` is the USD/PhysX approximation spelling for that
triangle-mesh mode.  This module deliberately audits the composed stage; it
does not mutate collision approximation or simulation settings.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


_SAFE_DYNAMIC_MESH_APPROXIMATIONS = frozenset({
    "boundingcube",
    "boundingsphere",
    "convexdecomposition",
    "convexhull",
    "sdf",
    "spherefill",
})


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _prim_path(prim: Any) -> str:
    path = _field(prim, "path")
    if path is None:
        get_path = getattr(prim, "GetPath", None)
        path = get_path() if callable(get_path) else None
    return str(path) if path is not None else "<unavailable>"


def _prim_type(prim: Any) -> str:
    type_name = _field(prim, "type_name")
    if type_name is None:
        get_type_name = getattr(prim, "GetTypeName", None)
        type_name = get_type_name() if callable(get_type_name) else None
    return str(type_name) if type_name is not None else "<unavailable>"


def _parent(prim: Any) -> Any:
    parent = _field(prim, "parent")
    if parent is not None:
        return parent
    get_parent = getattr(prim, "GetParent", None)
    return get_parent() if callable(get_parent) else None


def _rigid_body_for(prim: Any) -> Any:
    current = _parent(prim)
    while current is not None:
        if _field(current, "rigid_body", False):
            return current
        current = _parent(current)
    return None


def audit_dynamic_mesh_colliders(prims: Iterable[Any]) -> dict[str, object]:
    """Report, without modifying, dynamic triangle mesh collision shapes.

    ``prims`` is intentionally a small normalized interface so the evidence
    logic is CPU-testable without Isaac: mesh records expose ``collision``,
    ``approximation``, ``parent``, and their ancestor exposes ``rigid_body``
    and ``kinematic``.  :func:`audit_usd_stage` adapts real USD prims to it.
    """

    offending: list[dict[str, object]] = []
    for prim in prims:
        if _prim_type(prim) != "Mesh" or not bool(_field(prim, "collision", False)):
            continue
        rigid_body = _rigid_body_for(prim)
        if rigid_body is None or bool(_field(rigid_body, "kinematic", False)):
            continue
        approximation = _field(prim, "approximation")
        normalized_approximation = (
            str(approximation).strip().lower() if approximation is not None else "<unreadable>"
        )
        if normalized_approximation in _SAFE_DYNAMIC_MESH_APPROXIMATIONS:
            continue
        offending.append(
            {
                "usd_prim": _prim_path(prim),
                "prim_type": _prim_type(prim),
                "approximation": normalized_approximation,
                "rigid_body_prim": _prim_path(rigid_body),
                "rigid_body_kinematic": False,
            }
        )
    return {
        "healthy": not offending,
        **(
            {"reason": "unsupported_dynamic_triangle_mesh_collider"}
            if offending
            else {}
        ),
        "metric_name": "dynamic_triangle_mesh_collider_count",
        "metric_value": len(offending),
        "metric_limit": 0,
        "offending_colliders": offending,
    }


def audit_usd_stage(stage: Any) -> dict[str, object]:
    """Audit a composed USD stage and return stable prim-level evidence."""

    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError:
        return {
            "healthy": False,
            "reason": "collider_static_audit_unavailable",
            "metric_name": "dynamic_triangle_mesh_collider_count",
            "metric_value": "unavailable",
            "metric_limit": 0,
            "offending_colliders": [],
        }
    if stage is None:
        return {
            "healthy": False,
            "reason": "collider_static_audit_unavailable",
            "metric_name": "dynamic_triangle_mesh_collider_count",
            "metric_value": "unavailable",
            "metric_limit": 0,
            "offending_colliders": [],
        }

    def _nearest_api_ancestor(prim, api):
        current = prim
        while current:
            if current.HasAPI(api):
                return current
            current = current.GetParent()
        return None

    normalized_prims: list[dict[str, object]] = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        # CollisionAPI and MeshCollisionAPI author the mesh collider itself;
        # they are not inherited from an arbitrary ancestor Xform.
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        collision_enabled_attribute = prim.GetAttribute("physics:collisionEnabled")
        if (
            collision_enabled_attribute
            and collision_enabled_attribute.Get() is False
        ):
            continue
        rigid_body = _nearest_api_ancestor(prim, UsdPhysics.RigidBodyAPI)
        if not rigid_body:
            continue
        rigid_body_enabled_attribute = rigid_body.GetAttribute(
            "physics:rigidBodyEnabled"
        )
        if (
            rigid_body_enabled_attribute
            and rigid_body_enabled_attribute.Get() is False
        ):
            continue
        kinematic_attribute = rigid_body.GetAttribute("physics:kinematicEnabled")
        kinematic = bool(kinematic_attribute.Get()) if kinematic_attribute else False
        approximation_attribute = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr()
        approximation = approximation_attribute.Get() if approximation_attribute else None
        normalized_prims.append(
            {
                "path": str(prim.GetPath()),
                "type_name": str(prim.GetTypeName()),
                "collision": True,
                "approximation": approximation,
                "parent": {
                    "path": str(rigid_body.GetPath()),
                    "type_name": str(rigid_body.GetTypeName()),
                    "rigid_body": True,
                    "kinematic": kinematic,
                },
            }
        )
    return audit_dynamic_mesh_colliders(normalized_prims)


def audit_current_usd_stage() -> dict[str, object]:
    """Audit the active Kit stage, failing closed if it cannot be inspected."""

    try:
        import omni.usd
    except ImportError:
        return {
            "healthy": False,
            "reason": "collider_static_audit_unavailable",
            "metric_name": "dynamic_triangle_mesh_collider_count",
            "metric_value": "unavailable",
            "metric_limit": 0,
            "offending_colliders": [],
        }
    return audit_usd_stage(omni.usd.get_context().get_stage())


def audit_usd_file(usd_path: str) -> dict[str, object]:
    """Open one USD file for a CPU/static verifier without launching a sim."""

    try:
        from pxr import Usd
    except ImportError:
        return {
            "healthy": False,
            "reason": "collider_static_audit_unavailable",
            "metric_name": "dynamic_triangle_mesh_collider_count",
            "metric_value": "unavailable",
            "metric_limit": 0,
            "offending_colliders": [],
        }
    stage = Usd.Stage.Open(usd_path)
    result = audit_usd_stage(stage)
    return {"usd_path": usd_path, **result}
