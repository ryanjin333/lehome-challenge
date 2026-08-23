from __future__ import annotations

import ast
import importlib
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import types
from collections.abc import Mapping

import numpy as np


def _collider_audit_module():
    source_path = (
        Path(__file__).resolve().parents[1]
        / "source/lehome/lehome/assets/collider_audit.py"
    )
    spec = importlib.util.spec_from_file_location("collider_audit_test", source_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dynamic_mesh_collider_audit_reports_exact_prim_metric_and_limit() -> None:
    audit = _collider_audit_module()
    link = {
        "path": "/World/Robot/link",
        "type_name": "Xform",
        "rigid_body": True,
        "kinematic": False,
    }
    mesh = {
        "path": "/World/Robot/link/visual_mesh",
        "type_name": "Mesh",
        "collision": True,
        "approximation": "none",
        "parent": link,
    }
    sdf_mesh = {
        "path": "/World/Robot/link/collision_mesh",
        "type_name": "Mesh",
        "collision": True,
        "approximation": "sdf",
        "parent": link,
    }

    result = audit.audit_dynamic_mesh_colliders([mesh, sdf_mesh])

    assert result == {
        "healthy": False,
        "reason": "unsupported_dynamic_collider_geometry",
        "metric_name": "unsupported_dynamic_collider_count",
        "metric_value": 1,
        "metric_limit": 0,
        "offending_colliders": [
            {
                "usd_prim": "/World/Robot/link/visual_mesh",
                "prim_type": "Mesh",
                "approximation": "none",
                "rigid_body_prim": "/World/Robot/link",
                "rigid_body_kinematic": False,
            }
        ],
    }


def test_dynamic_mesh_collider_audit_fails_closed_when_approximation_is_unreadable() -> None:
    audit = _collider_audit_module()
    result = audit.audit_dynamic_mesh_colliders([
        {
            "path": "/World/Robot/link/unreadable_mesh",
            "type_name": "Mesh",
            "collision": True,
            "parent": {
                "path": "/World/Robot/link",
                "type_name": "Xform",
                "rigid_body": True,
                "kinematic": False,
            },
        }
    ])

    assert result["healthy"] is False
    assert result["metric_value"] == 1
    assert result["offending_colliders"][0]["approximation"] == "<unreadable>"


def test_dynamic_mesh_collider_audit_fails_closed_for_unknown_approximations() -> None:
    audit = _collider_audit_module()
    rigid_body = {
        "path": "/World/Robot/link",
        "type_name": "Xform",
        "rigid_body": True,
        "kinematic": False,
    }
    result = audit.audit_dynamic_mesh_colliders([
        {
            "path": "/World/Robot/link/unknown",
            "type_name": "Mesh",
            "collision": True,
            "approximation": "unexpectedApproximation",
            "parent": rigid_body,
        },
        {
            "path": "/World/Robot/link/mistyped",
            "type_name": "Mesh",
            "collision": True,
            "approximation": False,
            "parent": rigid_body,
        },
    ])

    assert result["healthy"] is False
    assert result["metric_value"] == 2
    assert [item["approximation"] for item in result["offending_colliders"]] == [
        "unexpectedapproximation",
        "false",
    ]


def test_mesh_simplification_is_an_unsupported_dynamic_triangle_mesh() -> None:
    audit = _collider_audit_module()
    result = audit.audit_dynamic_mesh_colliders([
        {
            "path": "/World/Robot/link/simplified_mesh",
            "type_name": "Mesh",
            "collision": True,
            "approximation": "meshSimplification",
            "parent": {
                "path": "/World/Robot/link",
                "type_name": "Xform",
                "rigid_body": True,
                "kinematic": False,
            },
        }
    ])

    assert result["healthy"] is False
    assert result["metric_value"] == 1
    assert result["offending_colliders"][0]["approximation"] == "meshsimplification"


def test_dynamic_plane_collider_is_rejected_even_without_mesh_schema() -> None:
    audit = _collider_audit_module()
    rigid_body = {
        "path": "/World/Scene/GroundPlane",
        "type_name": "Xform",
        "rigid_body": True,
        "kinematic": False,
    }

    result = audit.audit_dynamic_mesh_colliders([
        {
            "path": "/World/Scene/GroundPlane/CollisionPlane",
            "type_name": "Plane",
            "collision": True,
            "parent": rigid_body,
        }
    ])

    assert result["healthy"] is False
    assert result["metric_value"] == 1
    assert result["offending_colliders"] == [
        {
            "usd_prim": "/World/Scene/GroundPlane/CollisionPlane",
            "prim_type": "Plane",
            "approximation": "not_applicable",
            "rigid_body_prim": "/World/Scene/GroundPlane",
            "rigid_body_kinematic": False,
        }
    ]


def test_dynamic_collision_xform_uses_authored_mesh_approximation() -> None:
    audit = _collider_audit_module()
    rigid_body = {
        "path": "/World/Robot/base",
        "type_name": "Xform",
        "rigid_body": True,
        "kinematic": False,
    }

    result = audit.audit_dynamic_mesh_colliders([
        {
            "path": "/World/Robot/base/collisions",
            "type_name": "Xform",
            "collision": True,
            "mesh_collision": True,
            "approximation": "convexDecomposition",
            "parent": rigid_body,
        }
    ])

    assert result["healthy"] is True
    assert result["metric_value"] == 0


def test_room_spawn_disables_authored_dynamic_ground_plane_before_simulation() -> None:
    scene_source_path = (
        Path(__file__).resolve().parents[1]
        / "source/lehome/lehome/assets/scenes/bedroom.py"
    )
    scene_tree = ast.parse(scene_source_path.read_text(encoding="utf-8"))
    scene_cfg = next(
        node.value
        for node in scene_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "MARBLE_BEDROOM_CFG"
            for target in node.targets
        )
    )
    assert isinstance(scene_cfg, ast.Call)
    spawn_cfg = next(
        keyword.value for keyword in scene_cfg.keywords if keyword.arg == "spawn"
    )
    assert isinstance(spawn_cfg, ast.Call)
    rigid_props = next(
        keyword.value for keyword in spawn_cfg.keywords if keyword.arg == "rigid_props"
    )
    assert isinstance(rigid_props, ast.Call)
    assert isinstance(rigid_props.func, ast.Attribute)
    assert rigid_props.func.attr == "RigidBodyPropertiesCfg"
    rigid_enabled = next(
        keyword.value
        for keyword in rigid_props.keywords
        if keyword.arg == "rigid_body_enabled"
    )
    assert isinstance(rigid_enabled, ast.Constant)
    assert rigid_enabled.value is False

    env_source_path = (
        Path(__file__).resolve().parents[1]
        / "source/lehome/lehome/tasks/bedroom/garment_bi_v2.py"
    )
    tree = ast.parse(env_source_path.read_text(encoding="utf-8"))
    setup = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_setup_scene"
    )
    room_cfg = next(
        node
        for node in ast.walk(setup)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "MARBLE_BEDROOM_CFG"
        and node.attr == "spawn"
    )
    assert room_cfg is not None


def test_usd_collider_audit_uses_instance_proxies_and_mesh_collision_api(monkeypatch) -> None:
    audit = _collider_audit_module()

    class Attribute:
        def __init__(self, value):
            self.value = value

        def Get(self):
            return self.value

    class Prim:
        def __init__(self, path, type_name, parent=None, *, collision=False, rigid_body=False):
            self.path = path
            self.type_name = type_name
            self.parent = parent
            self.collision = collision
            self.rigid_body = rigid_body

        def IsA(self, schema):
            return schema == "Mesh" and self.type_name == "Mesh"

        def HasAPI(self, api):
            return (api == "CollisionAPI" and self.collision) or (
                api == "RigidBodyAPI" and self.rigid_body
            )

        def GetParent(self):
            return self.parent

        def GetAttribute(self, name):
            return Attribute(False) if name == "physics:kinematicEnabled" else None

        def GetPath(self):
            return self.path

        def GetTypeName(self):
            return self.type_name

        def __bool__(self):
            return True

    rigid_body = Prim("/so101_new_calib/base", "Xform", rigid_body=True)
    collision_xform = Prim(
        "/so101_new_calib/base/collisions",
        "Xform",
        parent=rigid_body,
    )
    instance_proxy_mesh = Prim(
        "/so101_new_calib/base/collisions/mesh_0",
        "Mesh",
        parent=collision_xform,
        collision=True,
    )

    class Stage:
        def Traverse(self):
            raise AssertionError("audit must traverse instance proxies")

    class MeshCollisionAPI:
        def __init__(self, prim):
            self.prim = prim

        def GetApproximationAttr(self):
            assert self.prim is instance_proxy_mesh
            return Attribute("convexDecomposition")

    pxr = types.ModuleType("pxr")
    pxr.Usd = types.SimpleNamespace(
        PrimRange=types.SimpleNamespace(
            Stage=lambda stage, traversal: [instance_proxy_mesh]
        ),
        TraverseInstanceProxies=lambda: "instance-proxies",
    )
    pxr.UsdGeom = types.SimpleNamespace(Mesh="Mesh")
    pxr.UsdPhysics = types.SimpleNamespace(
        CollisionAPI="CollisionAPI",
        RigidBodyAPI="RigidBodyAPI",
        MeshCollisionAPI=MeshCollisionAPI,
    )
    monkeypatch.setitem(sys.modules, "pxr", pxr)

    assert audit.audit_usd_stage(Stage()) == {
        "healthy": True,
        "metric_name": "unsupported_dynamic_collider_count",
        "metric_value": 0,
        "metric_limit": 0,
        "offending_colliders": [],
    }


def test_usd_collider_audit_finds_rigid_body_below_collision_ancestor(monkeypatch) -> None:
    audit = _collider_audit_module()

    class Attribute:
        def __init__(self, value):
            self.value = value

        def Get(self):
            return self.value

    class Prim:
        def __init__(self, path, type_name, parent=None, *, collision=False, rigid_body=False):
            self.path = path
            self.type_name = type_name
            self.parent = parent
            self.collision = collision
            self.rigid_body = rigid_body

        def IsA(self, schema):
            return schema == "Mesh" and self.type_name == "Mesh"

        def HasAPI(self, api):
            return (api == "CollisionAPI" and self.collision) or (
                api == "RigidBodyAPI" and self.rigid_body
            )

        def GetParent(self):
            return self.parent

        def GetAttribute(self, name):
            return Attribute(False) if name == "physics:kinematicEnabled" else None

        def GetPath(self):
            return self.path

        def GetTypeName(self):
            return self.type_name

        def __bool__(self):
            return True

    collision_root = Prim("/World/collisions", "Xform")
    rigid_body = Prim(
        "/World/collisions/dynamic_link",
        "Xform",
        parent=collision_root,
        rigid_body=True,
    )
    mesh = Prim(
        "/World/collisions/dynamic_link/mesh",
        "Mesh",
        parent=rigid_body,
        collision=True,
    )

    class MeshCollisionAPI:
        def __init__(self, prim):
            assert prim is mesh

        def GetApproximationAttr(self):
            return Attribute("none")

    pxr = types.ModuleType("pxr")
    pxr.Usd = types.SimpleNamespace(
        PrimRange=types.SimpleNamespace(Stage=lambda stage, traversal: [mesh]),
        TraverseInstanceProxies=lambda: "instance-proxies",
    )
    pxr.UsdGeom = types.SimpleNamespace(Mesh="Mesh")
    pxr.UsdPhysics = types.SimpleNamespace(
        CollisionAPI="CollisionAPI",
        RigidBodyAPI="RigidBodyAPI",
        MeshCollisionAPI=MeshCollisionAPI,
    )
    monkeypatch.setitem(sys.modules, "pxr", pxr)

    result = audit.audit_usd_stage(object())

    assert result["healthy"] is False
    assert result["metric_value"] == 1
    assert result["offending_colliders"][0]["usd_prim"] == str(mesh.GetPath())
    assert result["offending_colliders"][0]["rigid_body_prim"] == str(
        rigid_body.GetPath()
    )


def test_usd_collider_audit_normalizes_dynamic_plane_geometry(monkeypatch) -> None:
    audit = _collider_audit_module()

    class Attribute:
        def __init__(self, value):
            self.value = value

        def Get(self):
            return self.value

    class Prim:
        def __init__(self, path, type_name, parent=None, *, collision=False, rigid=False):
            self.path = path
            self.type_name = type_name
            self.parent = parent
            self.collision = collision
            self.rigid = rigid

        def IsA(self, schema):
            return schema == "Mesh" and self.type_name == "Mesh"

        def HasAPI(self, api):
            return (api == "CollisionAPI" and self.collision) or (
                api == "RigidBodyAPI" and self.rigid
            )

        def GetParent(self):
            return self.parent

        def GetAttribute(self, name):
            values = {
                "physics:collisionEnabled": True,
                "physics:rigidBodyEnabled": True,
                "physics:kinematicEnabled": False,
            }
            return Attribute(values[name]) if name in values else None

        def GetPath(self):
            return self.path

        def GetTypeName(self):
            return self.type_name

        def __bool__(self):
            return True

    rigid = Prim("/World/Scene/GroundPlane", "Xform", rigid=True)
    plane = Prim(
        "/World/Scene/GroundPlane/CollisionPlane",
        "Plane",
        parent=rigid,
        collision=True,
    )

    class MeshCollisionAPI:
        def __init__(self, prim):
            raise AssertionError("Plane geometry must not be adapted as a mesh")

    pxr = types.ModuleType("pxr")
    pxr.Usd = types.SimpleNamespace(
        PrimRange=types.SimpleNamespace(Stage=lambda stage, traversal: [plane]),
        TraverseInstanceProxies=lambda: "instance-proxies",
    )
    pxr.UsdGeom = types.SimpleNamespace(Mesh="Mesh")
    pxr.UsdPhysics = types.SimpleNamespace(
        CollisionAPI="CollisionAPI",
        RigidBodyAPI="RigidBodyAPI",
        MeshCollisionAPI=MeshCollisionAPI,
    )
    monkeypatch.setitem(sys.modules, "pxr", pxr)

    result = audit.audit_usd_stage(object())

    assert result["healthy"] is False
    assert result["metric_value"] == 1
    assert result["offending_colliders"][0]["prim_type"] == "Plane"


def test_usd_collider_audit_skips_parent_only_and_disabled_physics(monkeypatch) -> None:
    audit = _collider_audit_module()

    class Attribute:
        def __init__(self, value):
            self.value = value

        def Get(self):
            return self.value

    class Prim:
        def __init__(
            self,
            path,
            type_name,
            parent=None,
            *,
            collision=False,
            rigid_body=False,
            collision_enabled=True,
            rigid_body_enabled=True,
        ):
            self.path = path
            self.type_name = type_name
            self.parent = parent
            self.collision = collision
            self.rigid_body = rigid_body
            self.collision_enabled = collision_enabled
            self.rigid_body_enabled = rigid_body_enabled

        def IsA(self, schema):
            return schema == "Mesh" and self.type_name == "Mesh"

        def HasAPI(self, api):
            return (api == "CollisionAPI" and self.collision) or (
                api == "RigidBodyAPI" and self.rigid_body
            )

        def GetParent(self):
            return self.parent

        def GetAttribute(self, name):
            values = {
                "physics:collisionEnabled": self.collision_enabled,
                "physics:rigidBodyEnabled": self.rigid_body_enabled,
                "physics:kinematicEnabled": False,
            }
            return Attribute(values[name]) if name in values else None

        def GetPath(self):
            return self.path

        def GetTypeName(self):
            return self.type_name

        def __bool__(self):
            return True

    active_body = Prim("/World/active", "Xform", rigid_body=True)
    disabled_body = Prim(
        "/World/disabled-body",
        "Xform",
        rigid_body=True,
        rigid_body_enabled=False,
    )
    parent_collision = Prim(
        "/World/active/parent-collision",
        "Xform",
        parent=active_body,
        collision=True,
    )
    parent_only_mesh = Prim(
        "/World/active/parent-collision/mesh",
        "Mesh",
        parent=parent_collision,
    )
    disabled_collision_mesh = Prim(
        "/World/active/disabled-collision",
        "Mesh",
        parent=active_body,
        collision=True,
        collision_enabled=False,
    )
    disabled_body_mesh = Prim(
        "/World/disabled-body/mesh",
        "Mesh",
        parent=disabled_body,
        collision=True,
    )
    active_mesh = Prim(
        "/World/active/mesh",
        "Mesh",
        parent=active_body,
        collision=True,
    )
    meshes = [
        parent_only_mesh,
        disabled_collision_mesh,
        disabled_body_mesh,
        active_mesh,
    ]

    class MeshCollisionAPI:
        def __init__(self, prim):
            self.prim = prim

        def GetApproximationAttr(self):
            return Attribute("none")

    pxr = types.ModuleType("pxr")
    pxr.Usd = types.SimpleNamespace(
        PrimRange=types.SimpleNamespace(Stage=lambda stage, traversal: meshes),
        TraverseInstanceProxies=lambda: "instance-proxies",
    )
    pxr.UsdGeom = types.SimpleNamespace(Mesh="Mesh")
    pxr.UsdPhysics = types.SimpleNamespace(
        CollisionAPI="CollisionAPI",
        RigidBodyAPI="RigidBodyAPI",
        MeshCollisionAPI=MeshCollisionAPI,
    )
    monkeypatch.setitem(sys.modules, "pxr", pxr)

    result = audit.audit_usd_stage(object())

    assert result["metric_value"] == 1
    assert [item["usd_prim"] for item in result["offending_colliders"]] == [
        "/World/active/mesh"
    ]


def test_collider_health_and_admission_gate_fail_closed_with_live_evidence() -> None:
    env_source_path = (
        Path(__file__).resolve().parents[1]
        / "source/lehome/lehome/tasks/bedroom/garment_bi_v2.py"
    )
    env_tree = ast.parse(env_source_path.read_text(encoding="utf-8"))
    methods = {
        node.name: node
        for node in ast.walk(env_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"flywheel_collider_health", "flywheel_cloth_physical_health"}
    }
    env_module = ast.Module(
        body=[methods["flywheel_collider_health"], methods["flywheel_cloth_physical_health"]],
        type_ignores=[],
    )
    ast.fix_missing_locations(env_module)
    collision_health = {
        "healthy": False,
        "reason": "unsupported_dynamic_collider_geometry",
        "metric_name": "unsupported_dynamic_collider_count",
        "metric_value": 1,
        "metric_limit": 0,
        "offending_colliders": [
            {
                "usd_prim": "/World/Robot/link/visual_mesh",
                "prim_type": "Mesh",
                "approximation": "none",
            }
        ],
    }
    namespace = {"np": np, "audit_current_usd_stage": lambda: collision_health}
    exec(compile(env_module, str(env_source_path), "exec"), namespace)
    env = types.SimpleNamespace(_flywheel_collider_health=None)
    env.flywheel_collider_health = types.MethodType(namespace["flywheel_collider_health"], env)
    health = namespace["flywheel_cloth_physical_health"](env)
    assert health is collision_health

    evaluation_source_path = Path(__file__).resolve().parents[1] / "scripts/utils/evaluation.py"
    evaluation_tree = ast.parse(evaluation_source_path.read_text(encoding="utf-8"))
    gate = next(
        node
        for node in ast.walk(evaluation_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_require_flywheel_cloth_health"
    )
    gate_module = ast.Module(body=[gate], type_ignores=[])
    ast.fix_missing_locations(gate_module)
    error_type = type("SimulatorNumericalDivergenceError", (RuntimeError,), {})
    gate_namespace = {
        "Any": object,
        "Mapping": Mapping,
        "SimulatorNumericalDivergenceError": error_type,
        "_FLYWHEEL_POLICY_ACTION_JOINT_NAMES": (
            "left_shoulder_pan", "left_shoulder_lift", "left_elbow_flex",
            "left_wrist_flex", "left_wrist_roll", "left_gripper",
            "right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
            "right_wrist_flex", "right_wrist_roll", "right_gripper",
        ),
    }
    exec(compile(gate_module, str(evaluation_source_path), "exec"), gate_namespace)

    with __import__("pytest").raises(
        error_type,
        match=(
            r"unsupported_dynamic_collider_count=1 limit=0; .*"
            r"usd_prim=/World/Robot/link/visual_mesh prim_type=Mesh approximation=none"
        ),
    ):
        gate_namespace["_require_flywheel_cloth_health"](
            types.SimpleNamespace(flywheel_cloth_physical_health=lambda: health)
        )


def test_collider_health_converts_a_stage_readback_error_to_fail_closed_evidence() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "source/lehome/lehome/tasks/bedroom/garment_bi_v2.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "flywheel_collider_health"
    )
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "audit_current_usd_stage": lambda: (_ for _ in ()).throw(
            RuntimeError("stage readback failed")
        )
    }
    exec(compile(module, str(source_path), "exec"), namespace)
    env = types.SimpleNamespace(_flywheel_collider_health=None)

    health = namespace["flywheel_collider_health"](env)

    assert health == {
        "healthy": False,
        "reason": "collider_static_audit_unavailable",
        "metric_name": "unsupported_dynamic_collider_count",
        "metric_value": "unavailable",
        "metric_limit": 0,
        "offending_colliders": [],
    }


def test_garment_recreation_invalidates_the_composed_stage_collider_audit() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "source/lehome/lehome/tasks/bedroom/garment_bi_v2.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_create_garment_object"
    )
    method_source = ast.get_source_segment(source, method)
    assert method_source is not None
    assert "self._flywheel_collider_health = None" in method_source


def test_cpu_visible_contact_uses_live_usd_local_particles_without_constructing_a_physx_view() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "source/lehome/lehome/tasks/bedroom/garment_bi_v2.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "flywheel_visible_garment_contact"
    )
    method_source = ast.get_source_segment(source, method)
    assert method_source is not None
    assert 'str(self.device).lower() == "cpu"' in method_source
    assert "self._flywheel_legacy_cpu_cloth_state()" in method_source
    assert "self._flywheel_legacy_local_to_world(" in method_source
    assert "get_current_mesh_points" not in method_source


def test_cpu_cloth_backend_never_constructs_a_physx_view() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "source/lehome/lehome/tasks/bedroom/garment_bi_v2.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    method = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_flywheel_cloth_backend"
    )
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {}
    exec(compile(module, str(source_path), "exec"), namespace)
    env = types.SimpleNamespace(
        device="cpu",
        _flywheel_physics_cloth_view=lambda: (_ for _ in ()).throw(
            AssertionError("CPU backend must not create a PhysX cloth view")
        ),
    )

    assert namespace["_flywheel_cloth_backend"](env) == "usd_local_points_v1"


def test_cpu_initialize_obs_bypasses_the_external_physx_initializer() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "source/lehome/lehome/tasks/bedroom/garment_bi_v2.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    method = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "initialize_obs"
    )
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {}
    exec(compile(module, str(source_path), "exec"), namespace)
    initialized: list[str] = []
    env = types.SimpleNamespace(
        device="cpu",
        object=types.SimpleNamespace(initialize=lambda: (_ for _ in ()).throw(
            AssertionError("CPU initialize must not construct a PhysX cloth view")
        )),
        _flywheel_initialize_legacy_cpu_garment=lambda: initialized.append("usd-local"),
    )

    namespace["initialize_obs"](env)

    assert initialized == ["usd-local"]


def test_cpu_scene_pose_write_never_calls_physx_particle_restore(monkeypatch) -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "source/lehome/lehome/tasks/bedroom/garment_bi_v2.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    method = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "set_all_pose"
    )
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np}
    exec(compile(module, str(source_path), "exec"), namespace)

    isaacsim = types.ModuleType("isaacsim")
    core = types.ModuleType("isaacsim.core")
    utils = types.ModuleType("isaacsim.core.utils")
    rotations = types.ModuleType("isaacsim.core.utils.rotations")
    rotations.euler_angles_to_quat = lambda value, **_kwargs: np.asarray(
        [1.0, *np.asarray(value, dtype=np.float32)], dtype=np.float32
    )
    isaacsim.core, core.utils, utils.rotations = core, utils, rotations
    for name, value in {
        "isaacsim": isaacsim,
        "isaacsim.core": core,
        "isaacsim.core.utils": utils,
        "isaacsim.core.utils.rotations": rotations,
    }.items():
        monkeypatch.setitem(sys.modules, name, value)

    writes: list[tuple[np.ndarray, np.ndarray]] = []
    garment = types.SimpleNamespace(
        set_all_pose=lambda _pose: (_ for _ in ()).throw(
            AssertionError("CPU scene pose must not invoke the PhysX reset path")
        ),
        set_world_pose=lambda position, orientation: writes.append(
            (np.asarray(position), np.asarray(orientation))
        ),
    )
    env = types.SimpleNamespace(device="cpu", object=garment)
    pose = np.asarray([0.1, -0.2, 0.7, 10.0, 20.0, 30.0], dtype=np.float32)

    namespace["set_all_pose"](env, {"Garment": pose})

    assert len(writes) == 1
    np.testing.assert_array_equal(writes[0][0], pose[:3])
    np.testing.assert_array_equal(writes[0][1], np.asarray([1.0, 10.0, 20.0, 30.0]))
    np.testing.assert_array_equal(garment.reset_pose, pose)


def test_cpu_reset_restores_live_usd_state_without_physx(monkeypatch) -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "source/lehome/lehome/tasks/bedroom/garment_bi_v2.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    method = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_flywheel_reset_legacy_cpu_garment"
    )
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np}
    exec(compile(module, str(source_path), "exec"), namespace)
    isaacsim = types.ModuleType("isaacsim")
    core = types.ModuleType("isaacsim.core")
    utils = types.ModuleType("isaacsim.core.utils")
    rotations = types.ModuleType("isaacsim.core.utils.rotations")
    rotations.euler_angles_to_quat = lambda _value, **_kwargs: np.asarray([1.0, 0.0, 0.0, 0.0])
    isaacsim.core, core.utils, utils.rotations = core, utils, rotations
    for name, value in {
        "isaacsim": isaacsim,
        "isaacsim.core": core,
        "isaacsim.core.utils": utils,
        "isaacsim.core.utils.rotations": rotations,
    }.items():
        monkeypatch.setitem(sys.modules, name, value)
    state = {
        "positions": np.zeros((1, 3), dtype=np.float32),
        "velocities": np.zeros((1, 3), dtype=np.float32),
    }
    class Attribute:
        def __init__(self, key): self.key = key
        def Set(self, value): state[self.key] = np.asarray(value, dtype=np.float32)
    pose_writes: list[tuple[np.ndarray, np.ndarray]] = []
    env = types.SimpleNamespace(
        _flywheel_legacy_cpu_reset_state=(
            np.asarray([[0.2, 0.1, 0.7]], dtype=np.float32), np.zeros((1, 3), dtype=np.float32)
        ),
        garment_rng=np.random.RandomState(7),
        object=types.SimpleNamespace(
            _get_config_value=lambda _key, _source: ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "config"),
            set_world_pose=lambda position, orientation: pose_writes.append((position, orientation)),
            _ensure_physics_cloth_view=lambda: (_ for _ in ()).throw(
                AssertionError("CPU reset must not construct a PhysX cloth view")
            ),
        ),
        _flywheel_cloth_arrays=lambda positions, velocities: (np.asarray(positions), np.asarray(velocities)),
        _flywheel_legacy_cpu_cloth_attributes=lambda: (Attribute("positions"), Attribute("velocities")),
        _flywheel_legacy_usd_vec3f_array=lambda values: values,
        _flywheel_legacy_cpu_cloth_state=lambda: (state["positions"], state["velocities"]),
    )

    namespace["_flywheel_reset_legacy_cpu_garment"](env)

    assert pose_writes and np.allclose(state["positions"], [[0.2, 0.1, 0.7]])


def test_cpu_visible_contact_transforms_live_usd_local_points_without_physx(monkeypatch) -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "source/lehome/lehome/tasks/bedroom/garment_bi_v2.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    method = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "flywheel_visible_garment_contact"
    )
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np}
    exec(compile(module, str(source_path), "exec"), namespace)

    isaacsim = types.ModuleType("isaacsim")
    core = types.ModuleType("isaacsim.core")
    utils = types.ModuleType("isaacsim.core.utils")
    rotations = types.ModuleType("isaacsim.core.utils.rotations")
    rotations.quat_to_rot_matrix = lambda _quat: np.eye(3, dtype=np.float32)
    isaacsim.core, core.utils, utils.rotations = core, utils, rotations
    monkeypatch.setitem(sys.modules, "isaacsim", isaacsim)
    monkeypatch.setitem(sys.modules, "isaacsim.core", core)
    monkeypatch.setitem(sys.modules, "isaacsim.core.utils", utils)
    monkeypatch.setitem(sys.modules, "isaacsim.core.utils.rotations", rotations)
    contact = types.ModuleType("lehome.flywheel.contact")
    contact.visible_contact_from_simulator_geometry = lambda particles, grippers: {
        "observed": bool(np.allclose(particles, grippers)),
        "particles": particles.tolist(),
    }
    monkeypatch.setitem(sys.modules, "lehome.flywheel.contact", contact)

    env = types.SimpleNamespace(
        device="cpu",
        object=types.SimpleNamespace(
            get_world_pose=lambda: (np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 0.0, 0.0, 0.0])),
            get_world_scale=lambda: np.ones(3),
        ),
        left_arm=types.SimpleNamespace(
            body_names=["left_gripper"],
            data=types.SimpleNamespace(body_pos_w=np.asarray([[[1.0, 2.0, 3.0]]])),
        ),
        right_arm=types.SimpleNamespace(
            body_names=["right_gripper"],
            data=types.SimpleNamespace(body_pos_w=np.asarray([[[1.0, 2.0, 3.0]]])),
        ),
        _flywheel_legacy_cpu_cloth_state=lambda: (
            np.zeros((1, 3), dtype=np.float32), np.zeros((1, 3), dtype=np.float32)
        ),
        _flywheel_physics_cloth_state=lambda: (_ for _ in ()).throw(
            AssertionError("CPU contact must not create a PhysX cloth view")
        ),
        _flywheel_legacy_local_to_world=lambda points, velocities, position, _rotation, _scale: (
            points + position, velocities
        ),
    )

    evidence = namespace["flywheel_visible_garment_contact"](env)

    assert evidence == {"observed": True, "particles": [[1.0, 2.0, 3.0]]}


def test_garment_initializes_the_physx_cloth_view_on_cpu_and_cuda() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "source/lehome/lehome/assets/object/Garment.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "from pxr import Vt" in source
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "initialize"
    )
    method_source = ast.get_source_segment(source, method)
    assert method_source is not None
    assert 'if "cuda" in self._device' not in method_source
    assert "self._ensure_physics_cloth_view()" in method_source


def test_garment_rebinds_a_stale_physx_cloth_view_after_simulation_start() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "source/lehome/lehome/assets/object/Garment.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_ensure_physics_cloth_view"
    )
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)

    current_view = object()

    class SimulationManager:
        @staticmethod
        def get_physics_sim_view():
            return current_view

    namespace = {"SimulationManager": SimulationManager}
    exec(compile(module, str(source_path), "exec"), namespace)

    class View:
        def __init__(self) -> None:
            self.valid = False
            self.initialized_with = None

        def is_physics_handle_valid(self):
            return self.valid

        def initialize(self, value):
            self.initialized_with = value
            self.valid = True

        def get_world_positions(self):
            return []

        def get_velocities(self):
            return []

        def set_world_positions(self, _value):
            return None

        def set_velocities(self, _value):
            return None

    holder = types.SimpleNamespace(_cloth_prim_view=View(), physics_sim_view=None)
    rebound = namespace["_ensure_physics_cloth_view"](holder)

    assert rebound is holder._cloth_prim_view
    assert rebound.initialized_with is current_view
    assert holder.physics_sim_view is current_view


def test_persistent_worker_reports_runtime_preflight_failure_before_kit_close() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_groot_persistent_worker.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
    )
    method_source = ast.get_source_segment(source, method)
    assert method_source is not None
    assert "run() failed before kit close" in method_source
    assert method_source.index("run() failed before kit close") < method_source.index("closing kit")


def test_garment_reset_uses_the_authoritative_physx_view_on_cpu_and_cuda() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "source/lehome/lehome/assets/object/Garment.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    methods = {
        node.name: ast.get_source_segment(source, node)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"_get_initial_info", "reset", "set_all_pose", "_restore_world_particle_state"}
    }
    assert "get_world_positions" in methods["_get_initial_info"]
    assert "get_velocities" in methods["_get_initial_info"]
    assert "self._device" not in methods["_get_initial_info"]
    assert "_restore_world_particle_state" in methods["reset"]
    assert "_restore_world_particle_state" in methods["set_all_pose"]
    assert "set_world_positions" in methods["_restore_world_particle_state"]
    assert "set_velocities" in methods["_restore_world_particle_state"]
    assert "_ensure_physics_cloth_view" in methods["_restore_world_particle_state"]
    assert 'if self._device == "cpu"' not in methods["reset"]


def test_garment_world_particle_transform_applies_the_root_pose_delta() -> None:
    """Live PhysX particles are world-space, so a reset must move that world state."""

    source_path = (
        Path(__file__).resolve().parents[1]
        / "source/lehome/lehome/assets/object/Garment.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_transform_world_particle_positions"
    )
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    simulator_error = type("SimulatorNumericalDivergenceError", (ValueError,), {})
    namespace = {
        "np": np,
        "quat_to_rot_matrix": lambda quaternion: np.asarray(quaternion, dtype=np.float64).reshape(3, 3),
        "SimulatorNumericalDivergenceError": simulator_error,
    }
    exec(compile(module, str(source_path), "exec"), namespace)

    initial_points = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    initial_position = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    initial_rotation = np.eye(3, dtype=np.float32)
    target_position = np.asarray([2.0, 3.0, 0.0], dtype=np.float32)
    target_rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )

    actual = namespace["_transform_world_particle_positions"](
        initial_points, initial_position, initial_rotation, target_position, target_rotation
    )

    np.testing.assert_allclose(actual, np.asarray([[2.0, 4.0, 0.0], [1.0, 3.0, 0.0]], dtype=np.float32))

    with __import__("pytest").raises(simulator_error):
        namespace["_transform_world_particle_positions"](
            np.asarray([1.0, 2.0], dtype=np.float32),
            initial_position,
            initial_rotation,
            target_position,
            target_rotation,
        )


def test_garment_reset_delegates_to_frame_consistent_physx_restore_with_zero_velocity() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "source/lehome/lehome/assets/object/Garment.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    methods = {
        node.name: ast.get_source_segment(source, node)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"reset", "set_all_pose", "_restore_world_particle_state"}
    }
    assert "self._restore_world_particle_state(" in methods["reset"]
    assert "self._restore_world_particle_state(" in methods["set_all_pose"]
    assert "self.set_world_pose(" in methods["_restore_world_particle_state"]
    assert methods["_restore_world_particle_state"].index("self.set_world_pose(") < methods[
        "_restore_world_particle_state"
    ].index("cloth.set_world_positions(")
    assert "zeros_like" in methods["_restore_world_particle_state"]
    assert "readback mismatch" in methods["_restore_world_particle_state"]
    assert "SimulatorNumericalDivergenceError" in methods["_restore_world_particle_state"]


def test_garment_restore_keeps_legacy_cpu_replay_separate_from_controlled_physx_restore() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "source/lehome/lehome/tasks/bedroom/garment_bi_v2.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "flywheel_restore_state"
    )
    method_source = ast.get_source_segment(source, method)
    assert method_source is not None
    assert "schema_version == 1" in method_source
    assert "_flywheel_legacy_cpu_cloth_attributes" in method_source
    assert "schema_version == 2" in method_source
    assert 'cloth_state_authority", None) == "physx_cloth_view_world_v1"' in method_source


def test_cpu_source_capture_reads_live_usd_local_points_and_velocities() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "source/lehome/lehome/tasks/bedroom/garment_bi_v2.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "flywheel_capture_state"
    )
    method_source = ast.get_source_segment(source, method)
    assert method_source is not None
    assert 'str(self.device).lower() == "cpu"' in method_source
    assert "self._flywheel_legacy_cpu_cloth_state()" in method_source
    assert 'cloth_state_authority = "usd_local_points_v1"' in method_source


def test_cpu_source_reset_and_h16_capture_use_live_usd_local_state_without_physx() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "source/lehome/lehome/tasks/bedroom/garment_bi_v2.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    method = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "flywheel_capture_state"
    )
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np}
    exec(compile(module, str(source_path), "exec"), namespace)
    samples = iter((
        (np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32), np.zeros((1, 3), dtype=np.float32)),
        (np.asarray([[0.4, 0.5, 0.6]], dtype=np.float32), np.ones((1, 3), dtype=np.float32)),
    ))
    arm = types.SimpleNamespace(data=types.SimpleNamespace(
        joint_pos=np.zeros((1, 6), dtype=np.float32), joint_vel=np.zeros((1, 6), dtype=np.float32),
    ))
    env = types.SimpleNamespace(
        device="cpu", object=object(), left_arm=arm, right_arm=arm,
        cfg=types.SimpleNamespace(garment_name="Top_Long_Seen_0"),
        garment_rng=types.SimpleNamespace(
            get_state=lambda: ("MT19937", np.asarray([1, 2], dtype=np.uint32), 3, 0, 0.0)
        ),
        _flywheel_legacy_cpu_cloth_state=lambda: next(samples),
        _flywheel_physics_cloth_state=lambda: (_ for _ in ()).throw(
            AssertionError("CPU capture must not create a PhysX cloth view")
        ),
        _flywheel_capture_scene_state=lambda: {"scene": "live"},
    )

    reset, h16 = namespace["flywheel_capture_state"](env), namespace["flywheel_capture_state"](env)

    assert reset["cloth_state_authority"] == h16["cloth_state_authority"] == "usd_local_points_v1"
    assert np.allclose(reset["cloth_position"], [[0.1, 0.2, 0.3]])
    assert np.allclose(h16["cloth_position"], [[0.4, 0.5, 0.6]])


def test_legacy_cpu_cloth_snapshot_is_transformed_from_local_to_world_for_cuda() -> None:
    """Historical CPU snapshots store USD-local points, not PhysX world points."""

    source_path = (
        Path(__file__).resolve().parents[1]
        / "source/lehome/lehome/tasks/bedroom/garment_bi_v2.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_flywheel_legacy_local_to_world"
    ]
    assert len(matches) == 1, "legacy CPU-to-CUDA restore must have one explicit frame conversion"
    module = ast.Module(body=matches, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np}
    exec(compile(module, str(source_path), "exec"), namespace)

    local_positions = np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
    local_velocities = np.asarray([[0.0, 1.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    root_position = np.asarray([10.0, 20.0, 30.0], dtype=np.float32)
    root_scale = np.asarray([0.4, 0.5, 1.0], dtype=np.float32)
    root_rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )

    positions, velocities = namespace["_flywheel_legacy_local_to_world"](
        local_positions, local_velocities, root_position, root_rotation, root_scale
    )

    np.testing.assert_allclose(
        positions,
        np.asarray([[10.0, 20.4, 30.0], [9.0, 20.0, 30.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        velocities,
        np.asarray([[-0.5, 0.0, 0.0], [0.0, 0.8, 0.0]], dtype=np.float32),
    )


def test_legacy_cpu_cloth_transform_accepts_a_cuda_tensor_backed_world_scale() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "source/lehome/lehome/tasks/bedroom/garment_bi_v2.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_flywheel_legacy_local_to_world"
    )
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np}
    exec(compile(module, str(source_path), "exec"), namespace)

    class CudaTensorLike:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float32)

        def __array__(self, *args, **kwargs):
            raise TypeError("can't convert cuda tensor to numpy")

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    positions, velocities = namespace["_flywheel_legacy_local_to_world"](
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        np.eye(3, dtype=np.float32),
        CudaTensorLike([0.4, 0.5, 1.0]),
    )

    np.testing.assert_allclose(positions, [[0.4, 0.0, 0.0]], atol=1e-6)
    np.testing.assert_allclose(velocities, [[0.4, 0.0, 0.0]], atol=1e-6)


def _legacy_topology_projector():
    source_path = (
        Path(__file__).resolve().parents[1]
        / "source/lehome/lehome/tasks/bedroom/garment_bi_v2.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_flywheel_project_legacy_usd_to_physx"
    )
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    simulator_error = type("SimulatorNumericalDivergenceError", (ValueError,), {})
    namespace = {"np": np, "SimulatorNumericalDivergenceError": simulator_error}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["_flywheel_project_legacy_usd_to_physx"], simulator_error


def _physx_weld_map_reader():
    source_path = (
        Path(__file__).resolve().parents[1]
        / "source/lehome/lehome/tasks/bedroom/garment_bi_v2.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_flywheel_physx_weld_maps"
    )
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    simulator_error = type("SimulatorNumericalDivergenceError", (ValueError,), {})
    namespace = {"SimulatorNumericalDivergenceError": simulator_error}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["_flywheel_physx_weld_maps"], simulator_error


def test_physx_weld_map_reader_uses_active_cloth_prim_and_fails_closed() -> None:
    read_maps, simulator_error = _physx_weld_map_reader()

    class Attribute:
        def __init__(self, value):
            self.value = value

        def IsValid(self):
            return True

        def Get(self):
            return self.value

    class Prim:
        def __init__(self, attrs):
            self.attrs = attrs

        def GetAttribute(self, name):
            return self.attrs.get(name)

    class Environment:
        def __init__(self, attrs):
            self.object = type("Object", (), {"_prim": Prim(attrs)})()

    attrs = {
        "physxParticle:weldedVerticesRemapToOrig": Attribute([2, 0, 1]),
        "physxParticle:weldedVerticesRemapToWeld": Attribute([1, 2, 0]),
    }
    assert read_maps(Environment(attrs)) == ([2, 0, 1], [1, 2, 0])

    with __import__("pytest").raises(
        simulator_error,
        match=r"missing physxParticle:weldedVerticesRemapToWeld",
    ):
        read_maps(Environment({
            "physxParticle:weldedVerticesRemapToOrig": Attribute([0]),
        }))


def test_legacy_usd_snapshot_is_projected_into_live_physx_particle_order() -> None:
    project, _ = _legacy_topology_projector()
    asset = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    source_position = np.asarray(
        [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [10.0, 0.0, 0.0], [12.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    source_velocity = np.asarray(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    # The native cooked-map order is deliberately not authored USD order.
    live_rest = np.asarray(
        [[20.0, 0.0, 0.0], [30.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    welded_to_orig = np.asarray([3, 0, 1], dtype=np.int32)
    orig_to_weld = np.asarray([1, 2, 1, 0], dtype=np.int32)

    positions, velocities = project(
        source_position, source_velocity, asset, live_rest,
        welded_to_orig, orig_to_weld,
    )

    np.testing.assert_array_equal(positions[:, 0], [12.0, 10.0, 11.0])
    np.testing.assert_array_equal(velocities[:, 0], [3.0, 1.0, 2.0])


def test_legacy_usd_projection_rejects_inconsistent_duplicate_seam_state() -> None:
    project, simulator_error = _legacy_topology_projector()
    asset = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    positions = asset.copy()
    positions[2, 0] = 0.25

    with __import__("pytest").raises(
        simulator_error,
        match=(
            r"duplicate seam state is inconsistent: representative_index=0 "
            r"duplicate_index=2 position_max_abs_delta=0\.25 "
            r"velocity_max_abs_delta=0"
        ),
    ):
        project(
            positions, np.zeros_like(positions), asset, asset[:2],
            np.asarray([0, 1], dtype=np.int32), np.asarray([0, 1, 0], dtype=np.int32),
        )


def test_legacy_usd_projection_classifies_cardinality_mismatch_as_simulator_divergence() -> None:
    project, simulator_error = _legacy_topology_projector()
    asset = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    with __import__("pytest").raises(
        simulator_error,
        match=r"weldedVerticesRemapToOrig size mismatch: live_physx_count=1 remap_to_orig_count=2",
    ):
        project(
            asset, np.zeros_like(asset), asset, asset[:1],
            np.asarray([0, 1], dtype=np.int32), np.asarray([0, 0], dtype=np.int32),
        )


def test_legacy_usd_projection_uses_weld_maps_when_live_rest_has_deformed() -> None:
    project, simulator_error = _legacy_topology_projector()
    asset = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    live_rest = np.asarray([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]], dtype=np.float32)

    positions, velocities = project(
        asset, np.zeros_like(asset), asset, live_rest,
        np.asarray([1, 0], dtype=np.int32), np.asarray([1, 0], dtype=np.int32),
    )

    np.testing.assert_array_equal(positions[:, 0], [1.0, 0.0])
    np.testing.assert_array_equal(velocities, np.zeros((2, 3), dtype=np.float32))


def test_legacy_usd_projection_rejects_inconsistent_weld_maps() -> None:
    project, simulator_error = _legacy_topology_projector()
    asset = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)

    with __import__("pytest").raises(
        simulator_error,
        match=r"weld map inverse representative mismatch: welded_index=0 original_index=0 mapped_welded_index=1",
    ):
        project(
            asset, np.zeros_like(asset), asset, asset,
            np.asarray([0, 1], dtype=np.int32), np.asarray([1, 0], dtype=np.int32),
        )


def test_legacy_usd_projection_rejects_weld_map_that_groups_distinct_authored_vertices() -> None:
    project, simulator_error = _legacy_topology_projector()
    asset = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32
    )

    with __import__("pytest").raises(
        simulator_error,
        match=(
            r"weld map groups distinct authored vertices: original_index=1 "
            r"representative_index=0 asset_max_abs_delta=1"
        ),
    ):
        project(
            np.zeros_like(asset), np.zeros_like(asset), asset, asset[:2],
            np.asarray([0, 2], dtype=np.int32), np.asarray([0, 0, 1], dtype=np.int32),
        )


def test_legacy_usd_projection_rejects_missing_or_non_integer_weld_maps() -> None:
    project, simulator_error = _legacy_topology_projector()
    asset = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)

    with __import__("pytest").raises(
        simulator_error,
        match=r"weldedVerticesRemapToOrig must be a one-dimensional integer array",
    ):
        project(asset, np.zeros_like(asset), asset, asset, None, np.asarray([0, 1]))
    with __import__("pytest").raises(
        simulator_error,
        match=r"weldedVerticesRemapToWeld contains out-of-range index: index=1 value=2 live_physx_count=2",
    ):
        project(
            asset, np.zeros_like(asset), asset, asset,
            np.asarray([0, 1], dtype=np.int32), np.asarray([0, 2], dtype=np.int32),
        )


def test_legacy_usd_projection_handles_exact_pant_short_weld_shape_and_nonidentity_order() -> None:
    project, _ = _legacy_topology_projector()
    authored_count = 10_221
    welded_count = 10_033
    asset = np.zeros((authored_count, 3), dtype=np.float32)
    source_positions = np.zeros_like(asset)
    source_velocities = np.zeros_like(asset)
    welded_to_orig = np.concatenate((
        np.arange(8, 10_025, dtype=np.int32),
        np.arange(0, 8, dtype=np.int32),
        np.arange(10_213, 10_221, dtype=np.int32),
    ))
    assert welded_to_orig.shape == (welded_count,)
    orig_to_weld = np.zeros(authored_count, dtype=np.int32)
    orig_to_weld[welded_to_orig] = np.arange(welded_count, dtype=np.int32)
    # The 188 welded-away seam rows share particle zero's state exactly.
    source_positions[welded_to_orig, 0] = welded_to_orig
    source_velocities[welded_to_orig, 0] = welded_to_orig
    source_positions[10_025:10_213] = source_positions[welded_to_orig[0]]
    source_velocities[10_025:10_213] = source_velocities[welded_to_orig[0]]
    positions, velocities = project(
        source_positions, source_velocities, asset,
        np.full((welded_count, 3), 42.0, dtype=np.float32),
        welded_to_orig, orig_to_weld,
    )

    np.testing.assert_array_equal(positions[:, 0], welded_to_orig)
    np.testing.assert_array_equal(velocities[:, 0], welded_to_orig)


def test_legacy_cuda_restore_uses_scene_pose_frame_conversion_before_physx_write() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "source/lehome/lehome/tasks/bedroom/garment_bi_v2.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "flywheel_restore_state"
    )
    method_source = ast.get_source_segment(source, method)
    assert method_source is not None
    assert "_flywheel_legacy_local_to_world" in method_source
    assert "_flywheel_project_legacy_usd_to_physx" in method_source
    assert "garment_reset_pose" in method_source
    assert "get_world_scale" in method_source
    assert method_source.index("_flywheel_project_legacy_usd_to_physx") < method_source.index(
        "_flywheel_legacy_local_to_world"
    )
    assert method_source.index("_flywheel_legacy_local_to_world") < method_source.index(
        "cloth.set_world_positions"
    )


def test_authenticated_world_cloth_is_rigidly_rebased_across_randomized_garment_pose() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "source/lehome/lehome/tasks/bedroom/garment_bi_v2.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_flywheel_rebase_world_cloth"
    )
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np}
    exec(compile(module, str(source_path), "exec"), namespace)

    positions, velocities = namespace["_flywheel_rebase_world_cloth"](
        np.asarray([[11.0, 20.0, 30.0]], dtype=np.float32),
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        np.asarray([10.0, 20.0, 30.0], dtype=np.float32),
        np.eye(3, dtype=np.float32),
        np.asarray([10.0, 20.0, 30.0], dtype=np.float32),
        np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
    )

    np.testing.assert_allclose(positions, [[10.0, 21.0, 30.0]], atol=1e-6)
    np.testing.assert_allclose(velocities, [[0.0, 1.0, 0.0]], atol=1e-6)


def test_randomization_rewrites_authenticated_cloth_after_every_pose_mutation() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "source/lehome/lehome/tasks/bedroom/garment_bi_v2.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    restore_method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "flywheel_restore_state"
    )
    randomize_method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "apply_flywheel_randomization"
    )
    restore_source = ast.get_source_segment(source, restore_method)
    randomize_source = ast.get_source_segment(source, randomize_method)
    assert restore_source is not None and randomize_source is not None
    assert "_flywheel_preserved_restore_for_randomization" in restore_source
    assert "_flywheel_preserved_restore_for_randomization" in randomize_source
    assert "_flywheel_rebase_world_cloth" in randomize_source
    assert randomize_source.index("self.set_all_pose") < randomize_source.index(
        "cloth.set_world_positions"
    )
    assert randomize_source.index("cloth.set_world_positions") < randomize_source.index(
        "_flywheel_randomization_receipt = receipt"
    )


def _cloth_evidence(env) -> None:
    if not hasattr(env, "renderer_device"):
        env.renderer_device = "cuda:0"
    if not hasattr(env, "camera_device"):
        env.camera_device = env.renderer_device
    env._flywheel_cloth_backend = lambda: "physx_cloth_view"
    env._flywheel_physics_cloth_state = lambda: ([0.0], [0.0])
    env.flywheel_visible_garment_contact = lambda: {"observed": False}


def _cpu_cloth_evidence(env) -> None:
    if not hasattr(env, "renderer_device"):
        env.renderer_device = "cuda:0"
    if not hasattr(env, "camera_device"):
        env.camera_device = env.renderer_device
    env._flywheel_cloth_backend = lambda: "usd_local_points_v1"
    env._flywheel_legacy_cpu_cloth_state = lambda: (
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        np.zeros((1, 3), dtype=np.float32),
    )
    env._flywheel_physics_cloth_state = lambda: (_ for _ in ()).throw(
        AssertionError("CPU source evidence must not construct a PhysX cloth view")
    )
    env.flywheel_visible_garment_contact = lambda: {"observed": False}


def _evaluation(monkeypatch):
    """Load the session boundary without installing Isaac or torch."""
    repository = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repository))
    package = types.ModuleType("scripts.utils")
    package.__path__ = [str(repository / "scripts" / "utils")]
    modules = {
        "scripts.utils": package,
        "gymnasium": types.SimpleNamespace(make=lambda *_args, **_kwargs: None),
        "torch": types.SimpleNamespace(),
        "isaaclab": types.ModuleType("isaaclab"),
        "isaaclab.envs": types.ModuleType("isaaclab.envs"),
        "isaaclab_tasks": types.ModuleType("isaaclab_tasks"),
        "isaaclab_tasks.utils": types.ModuleType("isaaclab_tasks.utils"),
        "scripts.eval_policy": types.ModuleType("scripts.eval_policy"),
        "scripts.eval_policy.base_policy": types.ModuleType("scripts.eval_policy.base_policy"),
        "scripts.utils.eval_utils": types.ModuleType("scripts.utils.eval_utils"),
        "scripts.utils.common": types.ModuleType("scripts.utils.common"),
        "lehome.utils.record": types.ModuleType("lehome.utils.record"),
        "lehome.utils.logger": types.ModuleType("lehome.utils.logger"),
        "lehome.flywheel": types.ModuleType("lehome.flywheel"),
        "lehome.flywheel.persistent_worker": types.ModuleType("lehome.flywheel.persistent_worker"),
    }
    modules["isaaclab.envs"].DirectRLEnv = object
    modules["isaaclab_tasks.utils"].parse_env_cfg = lambda *_args, **_kwargs: None
    modules["scripts.eval_policy"].PolicyRegistry = object
    modules["scripts.eval_policy.base_policy"].BasePolicy = object
    modules["scripts.utils.eval_utils"].convert_ee_pose_to_joints = lambda *_args, **_kwargs: None
    modules["scripts.utils.eval_utils"].save_videos_from_observations = lambda *_args, **_kwargs: None
    modules["scripts.utils.eval_utils"].calculate_and_print_metrics = lambda *_args, **_kwargs: None
    modules["scripts.utils.common"].stabilize_garment_after_reset = lambda *_args, **_kwargs: None
    modules["lehome.utils.record"].RateLimiter = object
    modules["lehome.utils.record"].get_next_experiment_path_with_gap = lambda *_args, **_kwargs: None
    modules["lehome.utils.record"].append_episode_initial_pose = lambda *_args, **_kwargs: None
    modules["lehome.utils.logger"].get_logger = lambda *_args, **_kwargs: types.SimpleNamespace(info=lambda *_args: None)
    modules["lehome.flywheel.persistent_worker"].SimulatorNumericalDivergenceError = type(
        "SimulatorNumericalDivergenceError", (ValueError,), {}
    )
    modules["lehome.flywheel"].__path__ = [
        str(repository / "source" / "lehome" / "lehome" / "flywheel")
    ]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "scripts.utils.evaluation", raising=False)
    return importlib.import_module("scripts.utils.evaluation")


def test_evaluation_session_switches_garments_without_recreating_a_switchable_environment(monkeypatch) -> None:
    evaluation = _evaluation(monkeypatch)

    calls: list[object] = []

    class Environment:
        cfg = types.SimpleNamespace(garment_name="Top_Long_Seen_0", garment_version="Release", seed=None, random_seed=None)

        def switch_garment(self, name, stage):
            calls.append(("switch", name, stage))
            self.cfg.garment_name = name
            self.cfg.garment_version = stage

        def set_seed(self, seed):
            calls.append(("seed", seed))

        def close(self):
            calls.append("close")

    class Policy:
        def reset(self):
            calls.append("reset")

    args = types.SimpleNamespace(seed=0, num_episodes=1)
    env = Environment()
    session = evaluation.EvaluationSession(args, env=env, policy=Policy(), env_cfg=env.cfg)
    session.prepare_episode(garment_name="Top_Long_Seen_1", garment_stage="Release", seed=42, episode_generation=1)

    assert calls == [("seed", 42), ("switch", "Top_Long_Seen_1", "Release"), "reset"]
    assert env.cfg.seed == 42
    assert env.cfg.random_seed == 42


def test_garment_environment_set_seed_rebinds_the_active_object_rng() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "source/lehome/lehome/tasks/bedroom/garment_bi_v2.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "set_seed"
    )
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np}
    exec(compile(module, str(source_path), "exec"), namespace)

    first_object = types.SimpleNamespace(rng=None)
    second_object = types.SimpleNamespace(rng=None)
    first = types.SimpleNamespace(cfg=types.SimpleNamespace(seed=None, random_seed=None, use_random_seed=True), object=first_object)
    second = types.SimpleNamespace(cfg=types.SimpleNamespace(seed=None, random_seed=None, use_random_seed=True), object=second_object)

    namespace["set_seed"](first, 50_110)
    namespace["set_seed"](second, 50_110)

    assert first.cfg.seed == second.cfg.seed == 50_110
    assert first.cfg.random_seed == second.cfg.random_seed == 50_110
    assert first.cfg.use_random_seed is False
    assert second.cfg.use_random_seed is False
    assert first.object.rng is first.garment_rng
    assert second.object.rng is second.garment_rng
    assert first.garment_rng.uniform() == second.garment_rng.uniform()


def test_cloth_physical_health_rejects_finite_but_astronomical_state() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "source/lehome/lehome/tasks/bedroom/garment_bi_v2.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "flywheel_cloth_physical_health"
    )
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np}
    exec(compile(module, str(source_path), "exec"), namespace)

    normal = types.SimpleNamespace(
        flywheel_collider_health=lambda: {"healthy": True},
        _flywheel_physics_cloth_state=lambda: (
            np.asarray([[0.2, 0.1, 0.7], [0.3, 0.1, 0.7]], dtype=np.float32),
            np.zeros((2, 3), dtype=np.float32),
        )
    )
    divergent = types.SimpleNamespace(
        flywheel_collider_health=lambda: {"healthy": True},
        _flywheel_physics_cloth_state=lambda: (
            np.asarray([[1_000_000.0, 0.0, 0.0], [0.0, 1_000_000.0, 0.0]], dtype=np.float32),
            np.asarray([[1_000_000.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
        )
    )

    assert namespace["flywheel_cloth_physical_health"](normal)["healthy"] is True
    health = namespace["flywheel_cloth_physical_health"](divergent)
    assert health["healthy"] is False
    assert health["reason"] == "simulator_numerical_divergence"


def test_cpu_cloth_health_uses_live_usd_local_state_without_physx() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "source/lehome/lehome/tasks/bedroom/garment_bi_v2.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    method = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "flywheel_cloth_physical_health"
    )
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np}
    exec(compile(module, str(source_path), "exec"), namespace)
    env = types.SimpleNamespace(
        device="cpu",
        flywheel_collider_health=lambda: {"healthy": True},
        _flywheel_legacy_cpu_cloth_state=lambda: (
            np.asarray([[0.2, 0.1, 0.7], [0.3, 0.1, 0.7]], dtype=np.float32),
            np.zeros((2, 3), dtype=np.float32),
        ),
        _flywheel_physics_cloth_state=lambda: (_ for _ in ()).throw(
            AssertionError("CPU health must not create a PhysX cloth view")
        ),
    )

    assert namespace["flywheel_cloth_physical_health"](env)["healthy"] is True


def test_cloth_physical_health_reports_every_exceeded_metric_to_the_admission_gate() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "source/lehome/lehome/tasks/bedroom/garment_bi_v2.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "flywheel_cloth_physical_health"
    )
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np}
    exec(compile(module, str(source_path), "exec"), namespace)
    env = types.SimpleNamespace(
        flywheel_collider_health=lambda: {"healthy": True},
        _flywheel_physics_cloth_state=lambda: (
            np.asarray([[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]], dtype=np.float32),
            np.asarray([[6.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
        ),
    )

    health = namespace["flywheel_cloth_physical_health"](env)

    assert health["healthy"] is False
    assert health["exceeded_metrics"] == [
        {"metric_name": "max_position_m", "metric_value": 10.0, "metric_limit": 2.0},
        {"metric_name": "max_extent_m", "metric_value": 20.0, "metric_limit": 4.0},
        {"metric_name": "max_velocity_mps", "metric_value": 6.0, "metric_limit": 5.0001},
    ]

    evaluation_source_path = Path(__file__).resolve().parents[1] / "scripts/utils/evaluation.py"
    evaluation_tree = ast.parse(evaluation_source_path.read_text(encoding="utf-8"))
    gate = next(
        node
        for node in ast.walk(evaluation_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_require_flywheel_cloth_health"
    )
    gate_module = ast.Module(body=[gate], type_ignores=[])
    ast.fix_missing_locations(gate_module)
    error_type = type("SimulatorNumericalDivergenceError", (RuntimeError,), {})
    gate_namespace = {
        "Any": object,
        "Mapping": Mapping,
        "SimulatorNumericalDivergenceError": error_type,
        "_FLYWHEEL_POLICY_ACTION_JOINT_NAMES": (
            "left_shoulder_pan", "left_shoulder_lift", "left_elbow_flex",
            "left_wrist_flex", "left_wrist_roll", "left_gripper",
            "right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
            "right_wrist_flex", "right_wrist_roll", "right_gripper",
        ),
    }
    exec(compile(gate_module, str(evaluation_source_path), "exec"), gate_namespace)

    with __import__("pytest").raises(
        error_type,
        match=(
            r"max_position_m=10.0 limit=2.0; max_extent_m=20.0 limit=4.0; "
            r"max_velocity_mps=6.0 limit=5.0001"
        ),
    ):
        gate_namespace["_require_flywheel_cloth_health"](
            types.SimpleNamespace(flywheel_cloth_physical_health=lambda: health)
        )


def test_cloth_physical_health_classifies_invalid_physx_readback_as_divergence() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "source/lehome/lehome/tasks/bedroom/garment_bi_v2.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "flywheel_cloth_physical_health"
    )
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np}
    exec(compile(module, str(source_path), "exec"), namespace)

    invalid = types.SimpleNamespace(
        flywheel_collider_health=lambda: {"healthy": True},
        _flywheel_physics_cloth_state=lambda: (_ for _ in ()).throw(
            RuntimeError("garment PhysX cloth positions and velocities must be finite")
        )
    )

    health = namespace["flywheel_cloth_physical_health"](invalid)
    assert health == {
        "healthy": False,
        "reason": "simulator_numerical_divergence",
        "metric_name": "cloth_state_readback",
        "metric_value": "garment PhysX cloth positions and velocities must be finite",
        "metric_limit": "finite_aligned_nx3",
    }


def test_physx_cloth_readback_diagnoses_each_bounded_failure_class() -> None:
    """A source-canary abort must identify the failed PhysX admission, not coordinates."""

    source_path = (
        Path(__file__).resolve().parents[1]
        / "source/lehome/lehome/tasks/bedroom/garment_bi_v2.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    methods = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"_flywheel_cloth_arrays", "_flywheel_physics_cloth_state"}
    }
    module = ast.Module(
        body=[methods["_flywheel_cloth_arrays"], methods["_flywheel_physics_cloth_state"]],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"np": np}
    exec(compile(module, str(source_path), "exec"), namespace)

    class Cloth:
        def __init__(self, positions, velocities) -> None:
            self.positions = positions
            self.velocities = velocities

        def get_world_positions(self):
            if isinstance(self.positions, BaseException):
                raise self.positions
            return self.positions

        def get_velocities(self):
            if isinstance(self.velocities, BaseException):
                raise self.velocities
            return self.velocities

    def state(cloth, *, initial_count=2, root_position=(0.0, 0.0, 0.0)):
        object_state = types.SimpleNamespace(
            initial_points_positions=np.zeros((1, initial_count, 3), dtype=np.float32),
            get_world_pose=lambda: (
                np.asarray(root_position, dtype=np.float32),
                np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            ),
        )
        return types.SimpleNamespace(
            object=object_state,
            _flywheel_physics_cloth_view=lambda: cloth,
            _flywheel_cloth_arrays=namespace["_flywheel_cloth_arrays"],
        )

    cases = (
        (
            state(Cloth(np.asarray([[np.nan, 0.0, 0.0]], dtype=np.float32), np.zeros((1, 3), dtype=np.float32)), initial_count=1),
            "positions_nonfinite_count=1 velocities_nonfinite_count=0",
        ),
        (
            state(Cloth(np.zeros((1, 2, 3), dtype=np.float32), np.zeros((1, 3), dtype=np.float32))),
            "positions_shape=(2, 3) velocities_shape=(1, 3)",
        ),
        (
            state(Cloth(np.zeros((1, 3, 3), dtype=np.float32), np.zeros((1, 3, 3), dtype=np.float32))),
            "live_particle_count=3 initial_particle_count=2",
        ),
        (
            state(Cloth(RuntimeError("Kit getter failed"), np.zeros((1, 2, 3), dtype=np.float32))),
            "readback API failure",
        ),
        (
            state(Cloth(np.zeros((1, 2, 3), dtype=np.float32), np.zeros((1, 2, 3), dtype=np.float32)), root_position=(0.0, 0.0)),
            "root_position_shape=(2,)",
        ),
    )
    for env, diagnostic in cases:
        with __import__("pytest").raises(RuntimeError, match=__import__("re").escape(diagnostic)):
            namespace["_flywheel_physics_cloth_state"](env)


def test_flywheel_evaluation_checks_physical_health_before_success_or_recording() -> None:
    source_path = Path(__file__).resolve().parents[1] / "scripts/utils/evaluation.py"
    source = source_path.read_text(encoding="utf-8")
    method = ast.parse(source)
    loop = next(
        node
        for node in ast.walk(method)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run_evaluation_loop"
    )
    loop_source = ast.get_source_segment(source, loop)
    assert loop_source is not None
    health_index = loop_source.index("_require_flywheel_cloth_health(env)")
    assert health_index > loop_source.index("stabilize_garment_after_reset(env, args)")
    assert health_index < loop_source.index("observation_dict = env._get_observations()")
    assert health_index < loop_source.index("env._get_success()")
    assert health_index < loop_source.index("recorder.record_step(")
    assert health_index < loop_source.index("recorder.record_continuation_snapshot(")


def test_evaluation_session_refuses_a_switchable_environment_without_deterministic_seed_binding(monkeypatch) -> None:
    evaluation = _evaluation(monkeypatch)
    env = types.SimpleNamespace(
        cfg=types.SimpleNamespace(garment_name="Top_Long_Seen_0", garment_version="Release"),
        switch_garment=lambda *_args: None,
    )
    session = evaluation.EvaluationSession(
        types.SimpleNamespace(), env=env, policy=object(), env_cfg=env.cfg,
        require_deterministic_seed=True,
    )

    with __import__("pytest").raises(ValueError, match="set_seed"):
        session.prepare_episode(
            garment_name="Top_Long_Seen_1", seed=50_110, episode_generation=1
        )


def test_evaluation_session_preserves_explicit_random_seed_mode(monkeypatch) -> None:
    evaluation = _evaluation(monkeypatch)
    calls = []
    env = types.SimpleNamespace(
        cfg=types.SimpleNamespace(
            garment_name="Top_Long_Seen_0", garment_version="Release",
            use_random_seed=True,
        ),
        set_seed=lambda seed: calls.append(seed),
    )
    session = evaluation.EvaluationSession(
        types.SimpleNamespace(use_random_seed=True),
        env=env,
        policy=object(),
        env_cfg=env.cfg,
    )

    session.prepare_episode(
        garment_name="Top_Long_Seen_0", seed=42, episode_generation=1
    )

    assert calls == []
    assert env.cfg.use_random_seed is True


def test_cloth_physical_health_uses_garment_specific_scale_and_reset_overrides() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "source/lehome/lehome/tasks/bedroom/garment_bi_v2.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "flywheel_cloth_physical_health"
    )
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np}
    exec(compile(module, str(source_path), "exec"), namespace)

    env = types.SimpleNamespace(
        flywheel_collider_health=lambda: {"healthy": True},
        _flywheel_physics_cloth_state=lambda: (
            np.asarray([[2.1, 0.0, 0.7], [2.2, 0.0, 0.7]], dtype=np.float32),
            np.zeros((2, 3), dtype=np.float32),
        ),
        particle_config={
            "objects": {
                "common": {
                    "scale": [0.1, 0.1, 0.1],
                    "soft_reset_pos_range": [0.0] * 6,
                },
                "particle_system": {"max_velocity": 5.0},
            }
        },
        garment_config={
            "scale": [0.5, 0.5, 0.5],
            "soft_reset_pos_range": [2.0, 0.0, 0.7, 2.0, 0.0, 0.7],
        },
    )

    assert namespace["flywheel_cloth_physical_health"](env)["healthy"] is True


def test_cuda_cloth_runtime_receipt_requires_cuda_environment_device(monkeypatch) -> None:
    evaluation = _evaluation(monkeypatch)

    env = types.SimpleNamespace(device="cuda:0", cfg=types.SimpleNamespace())
    _cloth_evidence(env)
    args = types.SimpleNamespace(device="cuda:0", renderer_device="cuda:0", camera_device="cuda:0")
    session = evaluation.EvaluationSession(args, env=env, policy=object(), env_cfg=env.cfg)

    assert session.runtime_receipt["simulation_device"] == "cuda:0"
    assert session.runtime_receipt["cloth_device"] == "cuda:0"
    assert session.runtime_receipt["renderer_device"] == "cuda:0"


def test_source_bootstrap_runtime_receipt_preserves_cpu_cloth_with_cuda_rendering(monkeypatch) -> None:
    evaluation = _evaluation(monkeypatch)

    env = types.SimpleNamespace(device="cpu", cfg=types.SimpleNamespace())
    _cpu_cloth_evidence(env)
    args = types.SimpleNamespace(device="cpu", renderer_device="cuda:0", camera_device="cuda:0")
    policy = types.SimpleNamespace(runtime_device="cuda:0")
    session = evaluation.EvaluationSession(args, env=env, policy=policy, env_cfg=env.cfg)
    session._include_live_runtime_evidence = True

    assert session.runtime_receipt["simulation_device"] == "cpu"
    assert session.runtime_receipt["cloth_device"] == "cpu"
    assert session.runtime_receipt["cloth_backend"] == "usd_local_points_v1"
    assert session.runtime_receipt["cloth_readback"] == {"positions": 1, "velocities": 1}
    assert session.runtime_receipt["renderer_device"] == "cuda:0"
    assert session.runtime_receipt["camera_device"] == "cuda:0"
    assert session.runtime_receipt["policy_device"] == "cuda:0"


def test_cuda_runtime_receipt_rejects_usd_local_backend_mismatch(monkeypatch) -> None:
    evaluation = _evaluation(monkeypatch)
    env = types.SimpleNamespace(device="cuda:0", cfg=types.SimpleNamespace())
    _cpu_cloth_evidence(env)
    args = types.SimpleNamespace(device="cuda:0", renderer_device="cuda:0", camera_device="cuda:0")

    session = evaluation.EvaluationSession(args, env=env, policy=object(), env_cfg=env.cfg)

    with __import__("pytest").raises(ValueError, match="PhysX cloth backend"):
        session.runtime_receipt


def test_cuda_cloth_runtime_receipt_uses_observed_renderer_and_camera_devices(monkeypatch) -> None:
    evaluation = _evaluation(monkeypatch)

    env = types.SimpleNamespace(
        device="cuda:2", renderer_device="cuda:2", camera_device="cuda:2", cfg=types.SimpleNamespace(),
    )
    _cloth_evidence(env)
    args = types.SimpleNamespace(device="cuda:2", renderer_device="cuda:0", camera_device="cuda:0")
    session = evaluation.EvaluationSession(args, env=env, policy=object(), env_cfg=env.cfg)

    assert session.runtime_receipt["renderer_device"] == "cuda:2"
    assert session.runtime_receipt["camera_device"] == "cuda:2"


def test_evaluation_session_does_not_reset_a_policy_that_the_worker_already_reset(monkeypatch, tmp_path) -> None:
    evaluation = _evaluation(monkeypatch)
    reset_calls: list[str] = []

    class Policy:
        def reset(self):
            reset_calls.append("reset")

    env = types.SimpleNamespace(device="cpu", cfg=types.SimpleNamespace(garment_name="shirt", garment_version="Release"))
    env.close = lambda: None
    env.set_seed = lambda _seed: None
    args = types.SimpleNamespace(task="task", device="cpu", renderer_device="cuda:0", camera_device="cuda:0")
    session = evaluation.EvaluationSession(args, env=env, policy=Policy(), env_cfg=env.cfg)
    session.prepare_episode(garment_name="shirt", seed=42, episode_generation=1, reset_policy=False)

    assert reset_calls == []


def test_evaluation_session_keeps_legacy_writers_unless_given_an_explicit_attempt_output(monkeypatch, tmp_path) -> None:
    evaluation = _evaluation(monkeypatch)
    captured = []
    monkeypatch.setattr(evaluation, "run_evaluation_loop", lambda **kwargs: captured.append(kwargs["args"]) or [])
    env = types.SimpleNamespace(device="cpu", cfg=types.SimpleNamespace())
    args = types.SimpleNamespace(
        task="task", device="cpu", renderer_device="cuda:0", camera_device="cuda:0",
        video_dir="legacy-videos", eval_dataset_path="legacy-dataset",
    )
    session = evaluation.EvaluationSession(args, env=env, policy=object(), env_cfg=env.cfg)

    session.run_episode(assignment={"garment": "shirt"}, policy=object())
    session.run_episode(
        assignment={"garment": "shirt"}, policy=object(), attempt_output_dir=tmp_path / "attempt",
    )

    assert (captured[0].video_dir, captured[0].eval_dataset_path) == ("legacy-videos", "legacy-dataset")
    assert (captured[1].video_dir, captured[1].eval_dataset_path) == (
        str(tmp_path / "attempt" / "videos"), str(tmp_path / "attempt" / "dataset"),
    )


def test_evaluation_session_explicit_reset_flag_preserves_legacy_per_episode_resets(monkeypatch) -> None:
    evaluation = _evaluation(monkeypatch)
    reset_flags = []
    monkeypatch.setattr(evaluation, "run_evaluation_loop", lambda **kwargs: reset_flags.append(kwargs["reset_policy"]) or [])
    env = types.SimpleNamespace(device="cpu", cfg=types.SimpleNamespace())
    args = types.SimpleNamespace(task="task", video_dir="video", eval_dataset_path="dataset")
    session = evaluation.EvaluationSession(args, env=env, policy=object(), env_cfg=env.cfg)

    session.run_episode(assignment={"garment": "shirt"}, policy=object(), reset_policy=True)
    session.run_episode(assignment={"garment": "shirt"}, policy=object(), reset_policy=False)

    assert reset_flags == [True, False]


def test_persistent_attempt_authors_an_official_flywheel_manifest(monkeypatch, tmp_path) -> None:
    evaluation = _evaluation(monkeypatch)
    captured = []
    monkeypatch.setattr(evaluation, "run_evaluation_loop", lambda **kwargs: captured.append(kwargs["args"]) or [])
    env = types.SimpleNamespace(device="cuda:0", cfg=types.SimpleNamespace())
    args = types.SimpleNamespace(
        task="task", device="cuda:0", renderer_device="cuda:0", camera_device="cuda:0",
        policy_device="cuda:0", video_dir="legacy-videos", eval_dataset_path="legacy-dataset",
        flywheel_manifest=None,
    )
    session = evaluation.EvaluationSession(args, env=env, policy=object(), env_cfg=env.cfg)
    attempt = tmp_path / "attempt"
    session.run_episode(
        assignment={
            "garment": "Top_Long_Seen_0", "seed": 107, "category": "top_long",
            "release_stage": "seen", "difficulty": "seen", "attempt_id": "top-long-seen-0-seed-107",
        },
        policy=object(),
        attempt_output_dir=attempt,
    )
    manifest_path = Path(captured[0].flywheel_manifest)
    assert manifest_path == attempt / "flywheel-manifest.json"
    payload = __import__("json").loads(manifest_path.read_text())
    assert payload["episode_id"] == "top-long-seen-0-seed-107"
    assert payload["identity"]["garment_name"] == "Top_Long_Seen_0"
    assert payload["identity"]["strategy"] == "canonical"
    assert payload["policy_revision"] == "30ac1a84da67b099e115ad147bcd61e9d60046d3"
    assert payload["image_identity"].startswith("sha256:")
    assert payload["simulator_device"] == "cuda:0"


def test_source_bootstrap_manifest_marks_the_cpu_policy_server_parity_stage(monkeypatch, tmp_path) -> None:
    evaluation = _evaluation(monkeypatch)
    captured = []
    monkeypatch.setattr(evaluation, "run_evaluation_loop", lambda **kwargs: captured.append(kwargs["args"]) or [])
    env = types.SimpleNamespace(device="cpu", cfg=types.SimpleNamespace())
    args = types.SimpleNamespace(
        task="task", device="cpu", renderer_device="cuda:0", camera_device="cuda:0",
        policy_device="cuda:0", video_dir="legacy-videos", eval_dataset_path="legacy-dataset",
        flywheel_manifest=None,
    )
    session = evaluation.EvaluationSession(args, env=env, policy=object(), env_cfg=env.cfg)
    attempt = tmp_path / "attempt"

    session.run_episode(
        assignment={
            "garment": "Top_Short_Seen_2", "seed": 50066, "category": "top_short",
            "release_stage": "seen", "difficulty": "seen", "attempt_id": "source-cpu",
        },
        policy=object(),
        attempt_output_dir=attempt,
    )

    payload = __import__("json").loads(Path(captured[0].flywheel_manifest).read_text())
    assert payload["simulator_device"] == "cpu"
    assert payload["policy_device"] == "cuda:0"
    assert payload["parity_stage"] == "server_cpu"


def test_persistent_attempt_records_the_explicit_served_policy_identity(monkeypatch, tmp_path) -> None:
    evaluation = _evaluation(monkeypatch)
    captured = []
    monkeypatch.setattr(evaluation, "run_evaluation_loop", lambda **kwargs: captured.append(kwargs["args"]) or [])
    env = types.SimpleNamespace(device="cuda:0", cfg=types.SimpleNamespace())
    args = types.SimpleNamespace(
        task="task", device="cuda:0", renderer_device="cuda:0", camera_device="cuda:0",
        policy_device="cuda:0", video_dir="legacy-videos", eval_dataset_path="legacy-dataset",
        flywheel_manifest=None,
        policy_repo="ryanjin333/lehome-groot-n17-models",
        policy_revision="e" * 40,
        policy_step=2000,
        policy_artifact_sha256="7" * 64,
    )
    session = evaluation.EvaluationSession(args, env=env, policy=object(), env_cfg=env.cfg)
    attempt = tmp_path / "attempt"

    session.run_episode(
        assignment={
            "garment": "Top_Long_Unseen_0", "seed": 601, "category": "top_long",
            "release_stage": "public_unseen", "attempt_id": "top-long-public-unseen-0-seed-601",
        },
        policy=object(),
        attempt_output_dir=attempt,
    )

    payload = __import__("json").loads(Path(captured[0].flywheel_manifest).read_text())
    assert payload["policy_revision"] == "e" * 40
    assert payload["identity"]["policy_repo"] == "ryanjin333/lehome-groot-n17-models"
    assert payload["identity"]["policy_revision"] == "e" * 40
    assert payload["identity"]["policy_step"] == 2000
    assert payload["policy_artifact_sha256"] == "7" * 64


def test_persistent_randomized_assignment_authors_geometry_only_strategy(monkeypatch, tmp_path) -> None:
    evaluation = _evaluation(monkeypatch)
    captured = []
    monkeypatch.setattr(evaluation, "run_evaluation_loop", lambda **kwargs: captured.append(kwargs["args"]) or [])
    env = types.SimpleNamespace(device="cuda:0", cfg=types.SimpleNamespace())
    args = types.SimpleNamespace(
        task="task", device="cuda:0", renderer_device="cuda:0", camera_device="cuda:0",
        policy_device="cuda:0", video_dir="legacy-videos", eval_dataset_path="legacy-dataset",
        flywheel_manifest=None,
    )
    session = evaluation.EvaluationSession(args, env=env, policy=object(), env_cfg=env.cfg)
    attempt = tmp_path / "attempt"
    session.run_episode(
        assignment={
            "garment": "Top_Short_Seen_0", "seed": 139, "category": "top_short",
            "release_stage": "seen", "difficulty": "randomized",
            "attempt_id": "top-short-seen-0-mild-geometry-seed-139",
        },
        policy=object(),
        attempt_output_dir=attempt,
    )

    payload = __import__("json").loads(Path(captured[0].flywheel_manifest).read_text())
    assert payload["strategy"] == "mild_geometry"
    assert payload["identity"]["strategy"] == "mild_geometry"


def test_persistent_assignment_preserves_explicit_geometry_strategy(monkeypatch, tmp_path) -> None:
    evaluation = _evaluation(monkeypatch)
    assignment = {
        "garment": "Top_Short_Seen_1", "seed": 149, "category": "top_short",
        "release_stage": "seen", "difficulty": "randomized", "strategy": "strong_geometry",
        "attempt_id": "top-short-seen-1-strong-geometry-seed-149",
    }
    args = types.SimpleNamespace(device="cuda:0", policy_device="cuda:0")

    path = evaluation._write_persistent_flywheel_manifest(tmp_path / "attempt", assignment, args)
    payload = __import__("json").loads(path.read_text())

    assert payload["strategy"] == "strong_geometry"
    assert payload["identity"]["strategy"] == "strong_geometry"


def test_persistent_assignment_rejects_unstable_material_randomization(monkeypatch, tmp_path) -> None:
    evaluation = _evaluation(monkeypatch)
    assignment = {
        "garment": "Top_Short_Seen_1", "seed": 149, "category": "top_short",
        "release_stage": "seen", "difficulty": "randomized", "strategy": "strong",
        "attempt_id": "top-short-seen-1-strong-seed-149",
    }

    with __import__("pytest").raises(ValueError, match="geometry-only"):
        evaluation._write_persistent_flywheel_manifest(
            tmp_path / "attempt", assignment, types.SimpleNamespace(policy_device="cuda:0")
        )


def test_persistent_manifest_hands_controlled_recovery_identity_to_validator(monkeypatch, tmp_path) -> None:
    evaluation = _evaluation(monkeypatch)
    from lehome.flywheel.recovery_collection import load_controlled_recovery

    reset = tmp_path / "source-reset.json"
    continuation_snapshot = tmp_path / "source-continuation.json"
    annotations = tmp_path / "source-annotations.jsonl"
    reset.write_text(json.dumps({"schema_version": 2, "robot_position": [0.0] * 12, "robot_velocity": [0.0] * 12, "cloth_position": [[0.0, 0.0, 0.0]], "cloth_velocity": [[0.0, 0.0, 0.0]], "rng_state": {}, "garment_name": "Top_Long_Seen_0", "randomization": {"strategy": "canonical"}, "scene_state": {}, "cloth_state_authority": "physx_cloth_view_world_v1"}), encoding="utf-8")
    continuation_snapshot.write_text(json.dumps({"schema_version": 2, "robot_position": [0.0] * 12, "robot_velocity": [0.0] * 12, "cloth_position": [[0.0, 0.0, 0.0]], "cloth_velocity": [[0.0, 0.0, 0.0]], "rng_state": {}, "garment_name": "Top_Long_Seen_0", "randomization": {"strategy": "canonical", "continuation_step": 16}, "scene_state": {}, "cloth_state_authority": "physx_cloth_view_world_v1"}), encoding="utf-8")
    annotations.write_text(
        "".join(
                json.dumps({"step": step, "action": [float(step)] * 12, "success": step == 19}) + "\n"
                for step in range(20)
        ),
        encoding="utf-8",
    )
    category = "top_long"
    garment = "Top_Long_Seen_0"
    continuation_state = [0.0] * 12
    state_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "category": category,
                "garment": garment,
                "state_rounding": "fixed_6dp",
                "state": ["0.000000"] * 12,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assignment = {
        "recovery_kind": "controlled_success_recovery_snapshot_v3",
        "attempt_id": "controlled-top-long-0",
        "category": category,
        "garment": garment,
        "release_stage": "seen",
        "seed": 7,
        "strategy": "canonical",
        "source_reset": str(reset),
        "source_reset_sha256": hashlib.sha256(reset.read_bytes()).hexdigest(),
        "source_continuation_snapshot": str(continuation_snapshot),
        "source_continuation_snapshot_sha256": hashlib.sha256(continuation_snapshot.read_bytes()).hexdigest(),
        "source_annotations": str(annotations),
        "source_annotations_sha256": hashlib.sha256(annotations.read_bytes()).hexdigest(),
        "prefix_stop": 16,
        "source_first_success_step": 19,
        "perturbation_profile": {
            "cloth_displacement_m": 0.002,
            "cloth_velocity_mps": 0.01,
            "gripper_offset_rad": 0.02,
        },
        "perturbation_seed": 7,
        "source_round_id": "round",
        "source_episode_id": "episode",
        "source_episode_digest": "a" * 64,
        "source_immutable_revision": "b" * 40,
        "source_seed": 50110,
        "source_continuation_state": continuation_state,
        "source_snapshot_schema_version": 2,
        "source_snapshot_authority": "physx_cloth_view_world_v1",
        "source_only_envelope": False,
        "source_state_fingerprint": state_fingerprint,
        "perturbation_fingerprint": "d" * 64,
        "source_state_perturbation_fingerprint": "e" * 64,
    }

    manifest_path = evaluation._write_persistent_flywheel_manifest(
        tmp_path / "attempt", assignment, types.SimpleNamespace(device="cuda:0", policy_device="cuda:0")
    )
    envelope = json.loads(manifest_path.read_text(encoding="utf-8"))["controlled_recovery"]

    recovery = load_controlled_recovery(envelope)

    assert envelope["category"] == category
    assert envelope["garment"] == garment
    assert recovery.continuation_state == tuple(continuation_state)


def test_cuda_cloth_runtime_receipt_skips_contact_canary_until_requested(monkeypatch) -> None:
    evaluation = _evaluation(monkeypatch)
    calls = {"canary": 0}

    env = types.SimpleNamespace(device="cuda:0", cfg=types.SimpleNamespace())
    _cloth_evidence(env)
    def boom():
        calls["canary"] += 1
        raise RuntimeError("contact should not run at startup")
    env.flywheel_visible_garment_contact = boom
    args = types.SimpleNamespace(device="cuda:0", renderer_device="cuda:0", camera_device="cuda:0")
    session = evaluation.EvaluationSession(args, env=env, policy=object(), env_cfg=env.cfg)

    receipt = session.runtime_receipt
    assert "visible_contact_canary" not in receipt
    assert calls["canary"] == 0
    session._include_contact_canary = True
    try:
        session.runtime_receipt
    except RuntimeError as error:
        assert "contact should not run at startup" in str(error)
    else:
        raise AssertionError("expected contact canary to run only when requested")
    assert calls["canary"] == 1


def test_persistent_episode_enables_fresh_cloth_and_contact_runtime_evidence(monkeypatch, tmp_path) -> None:
    evaluation = _evaluation(monkeypatch)
    monkeypatch.setattr(
        evaluation,
        "run_evaluation_loop",
        lambda **_kwargs: [{"return": 0.0, "length": 1, "success": False}],
    )

    env = types.SimpleNamespace(
        device="cuda:0",
        renderer_device="cuda:0",
        camera_device="cuda:0",
        cfg=types.SimpleNamespace(garment_name="Top_Short_Seen_9", garment_version="Release"),
    )
    _cloth_evidence(env)
    args = types.SimpleNamespace(
        task="task",
        device="cuda:0",
        renderer_device="cuda:0",
        camera_device="cuda:0",
        video_dir="video",
        eval_dataset_path="dataset",
        flywheel_manifest=None,
    )
    session = evaluation.EvaluationSession(args, env=env, policy=object(), env_cfg=env.cfg)

    session.run_episode(
        assignment={"garment": "Top_Short_Seen_9"},
        attempt_output_dir=tmp_path / "attempt",
        policy=object(),
        cancellation_event=None,
    )

    receipt = session.runtime_receipt
    assert receipt["cloth_readback"] == {"positions": 1, "velocities": 1}
    assert receipt["visible_contact_canary"] == {"observed": False}




def test_run_episode_passes_hard_state_restore_into_evaluation_loop(monkeypatch, tmp_path) -> None:
    evaluation = _evaluation(monkeypatch)
    captured = []
    monkeypatch.setattr(evaluation, "run_evaluation_loop", lambda **kwargs: captured.append(kwargs["args"]) or [])
    env = types.SimpleNamespace(device="cpu", cfg=types.SimpleNamespace())
    args = types.SimpleNamespace(task="task", device="cpu", video_dir="v", eval_dataset_path="d")
    session = evaluation.EvaluationSession(args, env=env, policy=object(), env_cfg=env.cfg)
    session.run_episode(
        assignment={"garment": "Pant_Long_Seen_8", "restore_snapshot": "/tmp/terminal.json"},
        policy=object(),
    )
    assert captured[0].restore_snapshot == "/tmp/terminal.json"
    assert getattr(session, "_pending_restore_snapshot", None) in (None, "/tmp/terminal.json")


def test_verified_success_replay_binds_checked_snapshot_and_lineage_to_manifest(monkeypatch, tmp_path) -> None:
    evaluation = _evaluation(monkeypatch)
    captured = []
    monkeypatch.setattr(evaluation, "run_evaluation_loop", lambda **kwargs: captured.append(kwargs["args"]) or [])
    snapshot = tmp_path / "reset.json"
    snapshot.write_text(json.dumps({
        "schema_version": 1,
        "garment_name": "Top_Long_Seen_0",
        "scene_state": {"garment_reset_pose": [0.0, 0.0, 0.67, 0.0, 0.0, 90.0]},
    }), encoding="utf-8")
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    env = types.SimpleNamespace(device="cuda:0", cfg=types.SimpleNamespace())
    args = types.SimpleNamespace(
        task="task", device="cuda:0", renderer_device="cuda:0", camera_device="cuda:0",
        policy_device="cuda:0", video_dir="video", eval_dataset_path="dataset", flywheel_manifest=None,
    )
    session = evaluation.EvaluationSession(args, env=env, policy=object(), env_cfg=env.cfg)

    attempt = tmp_path / "attempt"
    session.run_episode(
        assignment={
            "attempt_id": "replay-top-long-000", "garment": "Top_Long_Seen_0", "category": "top_long",
            "release_stage": "seen", "seed": 50_000, "strategy": "mild_geometry",
            "replay_kind": "verified_success_reset_v1", "restore_snapshot": str(snapshot),
            "restore_snapshot_sha256": digest, "parent_episode_id": "original-top-long-1",
            "lineage_id": "original-top-long-1",
            "restore_snapshot_cloth_frame": "usd_local_points_v1",
        },
        policy=object(),
        attempt_output_dir=attempt,
    )

    expected_restore = json.loads(snapshot.read_text(encoding="utf-8"))
    expected_restore.update({
        "schema_version": 3,
        "cloth_state_authority": "usd_local_points_v1",
    })
    assert captured[0].restore_snapshot == expected_restore
    manifest = json.loads((attempt / "flywheel-manifest.json").read_text(encoding="utf-8"))
    assert manifest["restore_snapshot"] == str(snapshot)
    assert manifest["restore_snapshot_sha256"] == digest
    assert manifest["parent_episode_id"] == "original-top-long-1"
    assert manifest["lineage_id"] == "original-top-long-1"
    assert manifest["replay_kind"] == "verified_success_reset_v1"
    assert manifest["restore_snapshot_cloth_frame"] == "usd_local_points_v1"


def test_verified_success_replay_rejects_an_ambiguous_legacy_cloth_frame(monkeypatch, tmp_path) -> None:
    evaluation = _evaluation(monkeypatch)
    snapshot = tmp_path / "reset.json"
    snapshot.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()

    with __import__("pytest").raises(ValueError, match="cloth frame"):
        evaluation._verified_restore_assignment({
            "replay_kind": "verified_success_reset_v1",
            "restore_snapshot": str(snapshot),
            "restore_snapshot_sha256": digest,
            "parent_episode_id": "episode-1",
            "lineage_id": "episode-1",
        })


def test_verified_success_replay_rejects_a_tampered_snapshot_before_evaluation(monkeypatch, tmp_path) -> None:
    evaluation = _evaluation(monkeypatch)
    invoked = []
    monkeypatch.setattr(evaluation, "run_evaluation_loop", lambda **_kwargs: invoked.append(True) or [])
    snapshot = tmp_path / "reset.json"
    snapshot.write_text("{}", encoding="utf-8")
    env = types.SimpleNamespace(device="cpu", cfg=types.SimpleNamespace())
    args = types.SimpleNamespace(task="task", device="cpu", video_dir="video", eval_dataset_path="dataset")
    session = evaluation.EvaluationSession(args, env=env, policy=object(), env_cfg=env.cfg)

    with __import__("pytest").raises(ValueError, match="SHA-256"):
        session.run_episode(
            assignment={
                "garment": "Top_Long_Seen_0", "replay_kind": "verified_success_reset_v1",
                "restore_snapshot": str(snapshot), "restore_snapshot_sha256": "0" * 64,
            },
            policy=object(),
        )
    assert invoked == []


def test_policy_action_diagnostics_count_only_nonfinite_and_live_limit_violations() -> None:
    source_path = Path(__file__).resolve().parents[1] / "scripts/utils/evaluation.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    helper = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_flywheel_policy_action_limit_diagnostics"
    )
    joint_names = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "_FLYWHEEL_POLICY_ACTION_JOINT_NAMES"
            for target in node.targets
        )
    )
    module = ast.Module(body=[joint_names, helper], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"Any": object, "np": np}
    exec(compile(module, str(source_path), "exec"), namespace)

    class TensorLike:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float32)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    limits = np.repeat(
        np.asarray([[[-1.0, 1.0]]], dtype=np.float32), repeats=6, axis=1
    )
    env = types.SimpleNamespace(
        left_arm=types.SimpleNamespace(
            data=types.SimpleNamespace(
                soft_joint_pos_limits=TensorLike(limits),
                joint_pos=TensorLike(np.zeros((1, 6), dtype=np.float32)),
            )
        ),
        right_arm=types.SimpleNamespace(
            data=types.SimpleNamespace(
                soft_joint_pos_limits=TensorLike(limits),
                joint_pos=TensorLike(np.zeros((1, 6), dtype=np.float32)),
            )
        ),
    )
    action = TensorLike(
        [[-1.5, -1.0, 0.0, 1.0, 1.5, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    )

    diagnostics = namespace["_flywheel_policy_action_limit_diagnostics"](env, action)
    assert {
        key: diagnostics[key]
        for key in (
            "policy_action_limits_available",
            "policy_action_dimension",
            "policy_action_nonfinite_count",
            "policy_action_outside_live_joint_limit_count",
        )
    } == {
        "policy_action_limits_available": True,
        "policy_action_dimension": 12,
        "policy_action_nonfinite_count": 1,
        "policy_action_outside_live_joint_limit_count": 2,
    }
    assert diagnostics["policy_action_joint_diagnostics"]["left_gripper"] == {
        "target_finite": False,
        "outside_live_joint_limit": False,
        "limit_violation_rad": 0.0,
        "target_to_live_joint_position_delta_rad": 0.0,
    }


def test_policy_action_diagnostics_label_every_joint_with_limit_and_live_delta() -> None:
    source_path = Path(__file__).resolve().parents[1] / "scripts/utils/evaluation.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    helper = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_flywheel_policy_action_limit_diagnostics"
    )
    joint_names = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "_FLYWHEEL_POLICY_ACTION_JOINT_NAMES"
            for target in node.targets
        )
    )
    module = ast.Module(body=[joint_names, helper], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"Any": object, "np": np}
    exec(compile(module, str(source_path), "exec"), namespace)

    class TensorLike:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float32)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    limits = np.repeat(
        np.asarray([[[-1.0, 1.0]]], dtype=np.float32), repeats=6, axis=1
    )
    env = types.SimpleNamespace(
        left_arm=types.SimpleNamespace(
            data=types.SimpleNamespace(
                soft_joint_pos_limits=TensorLike(limits),
                joint_pos=TensorLike([[0.2, -0.2, 0.0, 0.3, -0.4, 0.5]]),
            )
        ),
        right_arm=types.SimpleNamespace(
            data=types.SimpleNamespace(
                soft_joint_pos_limits=TensorLike(limits),
                joint_pos=TensorLike([[-0.5, 0.4, -0.3, 0.2, -0.1, 0.0]]),
            )
        ),
    )
    action = TensorLike(
        [[-1.25, -0.2, 0.0, 0.3, 1.5, 0.5, -0.5, 0.4, -0.3, 0.2, -0.1, 0.0]]
    )

    diagnostics = namespace["_flywheel_policy_action_limit_diagnostics"](env, action)

    assert diagnostics["policy_action_joint_diagnostics"] == {
        "left_shoulder_pan": {
            "target_finite": True,
            "outside_live_joint_limit": True,
            "limit_violation_rad": 0.25,
            "target_to_live_joint_position_delta_rad": 1.45000005,
        },
        "left_shoulder_lift": {
            "target_finite": True,
            "outside_live_joint_limit": False,
            "limit_violation_rad": 0.0,
            "target_to_live_joint_position_delta_rad": 0.0,
        },
        "left_elbow_flex": {
            "target_finite": True,
            "outside_live_joint_limit": False,
            "limit_violation_rad": 0.0,
            "target_to_live_joint_position_delta_rad": 0.0,
        },
        "left_wrist_flex": {
            "target_finite": True,
            "outside_live_joint_limit": False,
            "limit_violation_rad": 0.0,
            "target_to_live_joint_position_delta_rad": 0.0,
        },
        "left_wrist_roll": {
            "target_finite": True,
            "outside_live_joint_limit": True,
            "limit_violation_rad": 0.5,
            "target_to_live_joint_position_delta_rad": 1.89999998,
        },
        "left_gripper": {
            "target_finite": True,
            "outside_live_joint_limit": False,
            "limit_violation_rad": 0.0,
            "target_to_live_joint_position_delta_rad": 0.0,
        },
        "right_shoulder_pan": {
            "target_finite": True,
            "outside_live_joint_limit": False,
            "limit_violation_rad": 0.0,
            "target_to_live_joint_position_delta_rad": 0.0,
        },
        "right_shoulder_lift": {
            "target_finite": True,
            "outside_live_joint_limit": False,
            "limit_violation_rad": 0.0,
            "target_to_live_joint_position_delta_rad": 0.0,
        },
        "right_elbow_flex": {
            "target_finite": True,
            "outside_live_joint_limit": False,
            "limit_violation_rad": 0.0,
            "target_to_live_joint_position_delta_rad": 0.0,
        },
        "right_wrist_flex": {
            "target_finite": True,
            "outside_live_joint_limit": False,
            "limit_violation_rad": 0.0,
            "target_to_live_joint_position_delta_rad": 0.0,
        },
        "right_wrist_roll": {
            "target_finite": True,
            "outside_live_joint_limit": False,
            "limit_violation_rad": 0.0,
            "target_to_live_joint_position_delta_rad": 0.0,
        },
        "right_gripper": {
            "target_finite": True,
            "outside_live_joint_limit": False,
            "limit_violation_rad": 0.0,
            "target_to_live_joint_position_delta_rad": 0.0,
        },
    }


def test_recorder_flywheel_step_projects_raw_policy_targets_and_records_the_applied_action(
    monkeypatch, tmp_path
) -> None:
    """The physical target is bounded, while raw policy-limit evidence remains intact."""
    evaluation = _evaluation(monkeypatch)

    assert hasattr(evaluation, "_project_flywheel_policy_action_to_live_limits")

    class FakeTensor:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float32)

        def float(self):
            return self

        def to(self, _device):
            return self

        def unsqueeze(self, axis):
            return FakeTensor(np.expand_dims(self.values, axis))

        def squeeze(self, axis=None):
            return self.values.squeeze(axis)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

        def clone(self):
            return FakeTensor(self.values.copy())

        def __setitem__(self, key, value):
            self.values[key] = value

    class FakeBool:
        def __init__(self, value):
            self.value = bool(value)

        def item(self):
            return self.value

    evaluation.torch = types.SimpleNamespace(
        Tensor=FakeTensor,
        from_numpy=lambda values: FakeTensor(values),
        tensor=lambda value: FakeBool(value),
    )
    evaluation.RateLimiter = lambda _step_hz: None
    manifest = {
        "_path": tmp_path / "flywheel-manifest.json",
        "strategy": "canonical",
        "seed": 1,
        "policy_revision": "revision",
        "policy_artifact_sha256": "artifact",
        "image_identity": "image",
        "execution_mode": "policy_server",
        "execution_backend": "policy_server",
        "simulator_device": "cpu",
    }
    monkeypatch.setattr(evaluation, "_load_flywheel_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        evaluation,
        "_flywheel_identity",
        lambda _manifest: types.SimpleNamespace(garment_name="Top_Long_Seen_0"),
    )
    monkeypatch.setattr(evaluation, "_validate_active_flywheel_garment", lambda *_args: None)

    recorded_actions = []

    class Recorder:
        def __init__(self, *_args, **_kwargs):
            self.step = 0

        def record_snapshot(self, *_args, **_kwargs):
            pass

        def record_step(self, _observation, action, **_kwargs):
            self.step += 1
            recorded_actions.append(np.asarray(action, dtype=np.float32).copy())

        def record_continuation_snapshot(self, *_args, **_kwargs):
            pass

        def finish(self, **_kwargs):
            pass

    recorder_module = types.ModuleType("lehome.flywheel.isaac_recorder")
    recorder_module.AutonomousRecorder = Recorder
    randomization_module = types.ModuleType("lehome.flywheel.randomization")
    randomization_module.sample_randomization = lambda *_args, **_kwargs: types.SimpleNamespace(values={})
    randomization_module.validate_randomization_receipt = lambda *_args, **_kwargs: None
    snapshots_module = types.ModuleType("lehome.flywheel.snapshots")
    snapshots_module.capture_snapshot = lambda *_args, **_kwargs: {"snapshot": True}
    monkeypatch.setitem(sys.modules, "lehome.flywheel.isaac_recorder", recorder_module)
    monkeypatch.setitem(sys.modules, "lehome.flywheel.randomization", randomization_module)
    monkeypatch.setitem(sys.modules, "lehome.flywheel.snapshots", snapshots_module)

    limits = np.repeat(
        np.asarray([[[-1.0, 1.0]]], dtype=np.float32), repeats=6, axis=1
    )
    stepped_actions = []
    env = types.SimpleNamespace(
        left_arm=types.SimpleNamespace(
            data=types.SimpleNamespace(
                soft_joint_pos_limits=limits,
                joint_pos=np.zeros((1, 6), dtype=np.float32),
            )
        ),
        right_arm=types.SimpleNamespace(
            data=types.SimpleNamespace(
                soft_joint_pos_limits=limits,
                joint_pos=np.zeros((1, 6), dtype=np.float32),
            )
        ),
        reset=lambda: None,
        apply_flywheel_randomization=lambda _sampled: {},
        flywheel_cloth_physical_health=lambda: {"healthy": True},
        flywheel_visible_garment_contact=lambda: {
            "observed": True,
            "minimum_distance_m": 0.01,
        },
        _get_observations=lambda: {"observation.state": np.zeros(12, dtype=np.float32)},
        _get_success=lambda: FakeBool(False),
        _get_rewards=lambda: 0.0,
    )
    env.step = lambda action: stepped_actions.append(action.detach().cpu().numpy().copy())

    raw_diagnostics = []
    original_diagnostics = evaluation._flywheel_policy_action_limit_diagnostics

    def capture_raw_diagnostics(step_env, action):
        diagnostics = original_diagnostics(step_env, action)
        raw_diagnostics.append(diagnostics)
        return diagnostics

    monkeypatch.setattr(
        evaluation, "_flywheel_policy_action_limit_diagnostics", capture_raw_diagnostics
    )
    policy = types.SimpleNamespace(
        reset=lambda: None,
        select_action_with_provenance=lambda _observation: types.SimpleNamespace(
            value=np.asarray(
                [0.25, 0.0, 1.25, 0.0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 0.0, 0.0],
                dtype=np.float32,
            ),
            request_id="request-1",
            chunk_offset=0,
        ),
    )
    args = types.SimpleNamespace(
        save_datasets=False,
        flywheel_manifest="enabled",
        seed=1,
        num_episodes=1,
        step_hz=30,
        max_steps=1,
        use_ee_pose=False,
        device="cpu",
        save_video=False,
    )

    evaluation.run_evaluation_loop(env, policy, args, garment_name="Top_Long_Seen_0")

    assert raw_diagnostics[0]["policy_action_outside_live_joint_limit_count"] == 2
    assert raw_diagnostics[0]["policy_action_joint_diagnostics"]["left_elbow_flex"] == {
        "target_finite": True,
        "outside_live_joint_limit": True,
        "limit_violation_rad": 0.25,
        "target_to_live_joint_position_delta_rad": 1.25,
    }
    assert raw_diagnostics[0]["policy_action_joint_diagnostics"]["right_elbow_flex"] == {
        "target_finite": True,
        "outside_live_joint_limit": True,
        "limit_violation_rad": 0.5,
        "target_to_live_joint_position_delta_rad": 1.5,
    }
    expected = np.asarray(
        [[0.25, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(stepped_actions, [expected])
    np.testing.assert_array_equal(recorded_actions, [expected.squeeze(0)])


def test_recorder_flywheel_step_fails_closed_before_env_step_when_live_limits_are_malformed(
    monkeypatch, tmp_path
) -> None:
    evaluation = _evaluation(monkeypatch)

    assert hasattr(evaluation, "_project_flywheel_policy_action_to_live_limits")

    class FakeTensor:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float32)

        def float(self):
            return self

        def to(self, _device):
            return self

        def unsqueeze(self, axis):
            return FakeTensor(np.expand_dims(self.values, axis))

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class FakeBool:
        def __init__(self, value):
            self.value = bool(value)

        def item(self):
            return self.value

    evaluation.torch = types.SimpleNamespace(
        Tensor=FakeTensor,
        from_numpy=lambda values: FakeTensor(values),
        tensor=lambda value: FakeBool(value),
    )
    evaluation.RateLimiter = lambda _step_hz: None
    manifest = {
        "_path": tmp_path / "flywheel-manifest.json",
        "strategy": "canonical",
        "seed": 1,
        "policy_revision": "revision",
        "policy_artifact_sha256": "artifact",
        "image_identity": "image",
        "execution_mode": "policy_server",
        "execution_backend": "policy_server",
        "simulator_device": "cpu",
    }
    monkeypatch.setattr(evaluation, "_load_flywheel_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        evaluation,
        "_flywheel_identity",
        lambda _manifest: types.SimpleNamespace(garment_name="Top_Long_Seen_0"),
    )
    monkeypatch.setattr(evaluation, "_validate_active_flywheel_garment", lambda *_args: None)
    recorder_module = types.ModuleType("lehome.flywheel.isaac_recorder")
    recorder_module.AutonomousRecorder = lambda *_args, **_kwargs: types.SimpleNamespace(
        record_snapshot=lambda *_args, **_kwargs: None,
    )
    randomization_module = types.ModuleType("lehome.flywheel.randomization")
    randomization_module.sample_randomization = lambda *_args, **_kwargs: types.SimpleNamespace(values={})
    randomization_module.validate_randomization_receipt = lambda *_args, **_kwargs: None
    snapshots_module = types.ModuleType("lehome.flywheel.snapshots")
    snapshots_module.capture_snapshot = lambda *_args, **_kwargs: {"snapshot": True}
    monkeypatch.setitem(sys.modules, "lehome.flywheel.isaac_recorder", recorder_module)
    monkeypatch.setitem(sys.modules, "lehome.flywheel.randomization", randomization_module)
    monkeypatch.setitem(sys.modules, "lehome.flywheel.snapshots", snapshots_module)

    stepped_actions = []
    env = types.SimpleNamespace(
        left_arm=types.SimpleNamespace(
            data=types.SimpleNamespace(
                soft_joint_pos_limits=np.zeros((1, 6, 3), dtype=np.float32),
                joint_pos=np.zeros((1, 6), dtype=np.float32),
            )
        ),
        right_arm=types.SimpleNamespace(
            data=types.SimpleNamespace(
                soft_joint_pos_limits=np.zeros((1, 6, 2), dtype=np.float32),
                joint_pos=np.zeros((1, 6), dtype=np.float32),
            )
        ),
        reset=lambda: None,
        apply_flywheel_randomization=lambda _sampled: {},
        flywheel_cloth_physical_health=lambda: {"healthy": True},
        _get_observations=lambda: {"observation.state": np.zeros(12, dtype=np.float32)},
    )
    env.step = lambda action: stepped_actions.append(action)
    policy = types.SimpleNamespace(
        reset=lambda: None,
        select_action_with_provenance=lambda _observation: types.SimpleNamespace(
            value=np.zeros(12, dtype=np.float32), request_id="request-1", chunk_offset=0,
        ),
    )
    args = types.SimpleNamespace(
        save_datasets=False,
        flywheel_manifest="enabled",
        seed=1,
        num_episodes=1,
        step_hz=30,
        max_steps=1,
        use_ee_pose=False,
        device="cpu",
        save_video=False,
    )

    with __import__("pytest").raises(
        evaluation.SimulatorNumericalDivergenceError, match="live soft joint limits"
    ):
        evaluation.run_evaluation_loop(env, policy, args, garment_name="Top_Long_Seen_0")

    assert stepped_actions == []


def test_policy_action_diagnostics_use_one_canonical_semantic_joint_order() -> None:
    source_path = Path(__file__).resolve().parents[1] / "scripts/utils/evaluation.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    constant = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "_FLYWHEEL_POLICY_ACTION_JOINT_NAMES"
            for target in node.targets
        )
    )
    assert ast.literal_eval(constant.value) == (
        "left_shoulder_pan",
        "left_shoulder_lift",
        "left_elbow_flex",
        "left_wrist_flex",
        "left_wrist_roll",
        "left_gripper",
        "right_shoulder_pan",
        "right_shoulder_lift",
        "right_elbow_flex",
        "right_wrist_flex",
        "right_wrist_roll",
        "right_gripper",
    )
    for function_name in (
        "_flywheel_policy_action_limit_diagnostics",
        "_require_flywheel_cloth_health",
    ):
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        )
        assert "_FLYWHEEL_POLICY_ACTION_JOINT_NAMES" in ast.get_source_segment(source, function)


def test_policy_action_diagnostics_fail_closed_for_nonfinite_live_position() -> None:
    source_path = Path(__file__).resolve().parents[1] / "scripts/utils/evaluation.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    helper = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_flywheel_policy_action_limit_diagnostics"
    )
    joint_names = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "_FLYWHEEL_POLICY_ACTION_JOINT_NAMES"
            for target in node.targets
        )
    )
    module = ast.Module(body=[joint_names, helper], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"Any": object, "np": np}
    exec(compile(module, str(source_path), "exec"), namespace)

    env = types.SimpleNamespace(
        left_arm=types.SimpleNamespace(
            data=types.SimpleNamespace(
                soft_joint_pos_limits=np.zeros((1, 6, 2), dtype=np.float32),
                joint_pos=np.zeros((1, 6), dtype=np.float32),
            )
        ),
        right_arm=types.SimpleNamespace(
            data=types.SimpleNamespace(
                soft_joint_pos_limits=np.zeros((1, 6, 2), dtype=np.float32),
                joint_pos=np.asarray([[0.0, 0.0, 0.0, np.nan, 0.0, 0.0]], dtype=np.float32),
            )
        ),
    )

    assert namespace["_flywheel_policy_action_limit_diagnostics"](
        env, np.zeros((1, 12), dtype=np.float32)
    ) == {"policy_action_limits_available": False}


def test_nonfinite_policy_target_keeps_all_cumulative_joint_keys_and_summary() -> None:
    source_path = Path(__file__).resolve().parents[1] / "scripts/utils/evaluation.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    constant = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "_FLYWHEEL_POLICY_ACTION_JOINT_NAMES"
            for target in node.targets
        )
    )
    helper = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_flywheel_policy_action_limit_diagnostics"
    )
    gate = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_require_flywheel_cloth_health"
    )
    module = ast.Module(body=[constant, helper, gate], type_ignores=[])
    ast.fix_missing_locations(module)
    error_type = type("SimulatorNumericalDivergenceError", (RuntimeError,), {})
    namespace = {
        "Any": object,
        "Mapping": Mapping,
        "SimulatorNumericalDivergenceError": error_type,
        "np": np,
    }
    exec(compile(module, str(source_path), "exec"), namespace)

    limits = np.repeat(
        np.asarray([[[-1.0, 1.0]]], dtype=np.float32), repeats=6, axis=1
    )
    env = types.SimpleNamespace(
        left_arm=types.SimpleNamespace(
            data=types.SimpleNamespace(
                soft_joint_pos_limits=limits,
                joint_pos=np.zeros((1, 6), dtype=np.float32),
            )
        ),
        right_arm=types.SimpleNamespace(
            data=types.SimpleNamespace(
                soft_joint_pos_limits=limits,
                joint_pos=np.zeros((1, 6), dtype=np.float32),
            )
        ),
    )
    step = namespace["_flywheel_policy_action_limit_diagnostics"](
        env,
        np.asarray(
            [[0.0, 0.0, 0.0, 0.0, 0.0, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
    )
    joints = step["policy_action_joint_diagnostics"]
    assert joints["left_gripper"]["target_finite"] is False
    assert step["policy_action_nonfinite_count"] == 1
    cumulative_counts = {joint_name: 0 for joint_name in joints}
    cumulative_violation = {
        joint_name: 0.0 for joint_name in joints
    }
    cumulative_delta = {joint_name: 0.0 for joint_name in joints}

    with __import__("pytest").raises(error_type) as raised:
        namespace["_require_flywheel_cloth_health"](
            types.SimpleNamespace(
                flywheel_cloth_physical_health=lambda: {
                    "healthy": False,
                    "reason": "simulator_numerical_divergence",
                }
            ),
            policy_action_diagnostics={
                **step,
                "policy_action_total_steps": 1,
                "policy_action_outside_live_joint_limit_step_counts": cumulative_counts,
                "policy_action_max_limit_violation_rad": cumulative_violation,
                "policy_action_max_target_to_live_joint_position_delta_rad": cumulative_delta,
            },
        )

    message = str(raised.value)
    for joint_name in joints:
        assert f"{joint_name}(outside_steps=0,max_violation_rad=0.0," in message

    loop = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run_evaluation_loop"
    )
    loop_source = ast.get_source_segment(source, loop)
    assert loop_source is not None
    assert loop_source.index("policy_action_outside_live_joint_limit_step_counts.setdefault") < loop_source.index(
        "target_finite is not True"
    )


def test_cloth_failure_includes_bounded_policy_action_history_without_coordinates() -> None:
    source_path = Path(__file__).resolve().parents[1] / "scripts/utils/evaluation.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    gate = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_require_flywheel_cloth_health"
    )
    module = ast.Module(body=[gate], type_ignores=[])
    ast.fix_missing_locations(module)
    error_type = type("SimulatorNumericalDivergenceError", (RuntimeError,), {})
    namespace = {
        "Any": object,
        "Mapping": Mapping,
        "SimulatorNumericalDivergenceError": error_type,
        "_FLYWHEEL_POLICY_ACTION_JOINT_NAMES": (
            "left_shoulder_pan", "left_shoulder_lift", "left_elbow_flex",
            "left_wrist_flex", "left_wrist_roll", "left_gripper",
            "right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
            "right_wrist_flex", "right_wrist_roll", "right_gripper",
        ),
    }
    exec(compile(module, str(source_path), "exec"), namespace)
    diagnostics = {
        "policy_action_limits_available": True,
        "policy_action_dimension": 12,
        "policy_action_nonfinite_count": 0,
        "policy_action_outside_live_joint_limit_count": 3,
        "policy_action_steps_outside_live_joint_limits": 2,
        "policy_action_max_outside_live_joint_limit_count": 4,
        "policy_action_total_steps": 7,
        "policy_action_outside_live_joint_limit_step_counts": {
            "left_shoulder_pan": 2,
            "left_shoulder_lift": 0,
            "left_elbow_flex": 0,
            "left_wrist_flex": 0,
            "left_wrist_roll": 1,
            "left_gripper": 0,
            "right_shoulder_pan": 0,
            "right_shoulder_lift": 0,
            "right_elbow_flex": 0,
            "right_wrist_flex": 0,
            "right_wrist_roll": 0,
            "right_gripper": 0,
        },
        "policy_action_max_limit_violation_rad": {
            "left_shoulder_pan": 0.25,
            "left_shoulder_lift": 0.0,
            "left_elbow_flex": 0.0,
            "left_wrist_flex": 0.0,
            "left_wrist_roll": 0.5,
            "left_gripper": 0.0,
            "right_shoulder_pan": 0.0,
            "right_shoulder_lift": 0.0,
            "right_elbow_flex": 0.0,
            "right_wrist_flex": 0.0,
            "right_wrist_roll": 0.0,
            "right_gripper": 0.0,
        },
        "policy_action_max_target_to_live_joint_position_delta_rad": {
            "left_shoulder_pan": 1.45,
            "left_shoulder_lift": 0.0,
            "left_elbow_flex": 0.0,
            "left_wrist_flex": 0.0,
            "left_wrist_roll": 1.9,
            "left_gripper": 0.0,
            "right_shoulder_pan": 0.0,
            "right_shoulder_lift": 0.0,
            "right_elbow_flex": 0.0,
            "right_wrist_flex": 0.0,
            "right_wrist_roll": 0.0,
            "right_gripper": 0.0,
        },
    }

    with __import__("pytest").raises(
        error_type,
        match=(
            r"policy_action_nonfinite_count=0; "
            r"policy_action_outside_live_joint_limit_count=3; "
            r"policy_action_steps_outside_live_joint_limits=2; "
            r"policy_action_max_outside_live_joint_limit_count=4; "
            r"policy_action_total_steps=7; "
            r"policy_action_joint_summary="
            r"left_shoulder_pan\(outside_steps=2,max_violation_rad=0.25,"
            r"max_target_to_live_joint_position_delta_rad=1.45\),"
            r"left_shoulder_lift\(outside_steps=0,max_violation_rad=0.0,"
            r"max_target_to_live_joint_position_delta_rad=0.0\),"
            r"left_elbow_flex\(outside_steps=0,max_violation_rad=0.0,"
            r"max_target_to_live_joint_position_delta_rad=0.0\),"
            r"left_wrist_flex\(outside_steps=0,max_violation_rad=0.0,"
            r"max_target_to_live_joint_position_delta_rad=0.0\),"
            r"left_wrist_roll\(outside_steps=1,max_violation_rad=0.5,"
            r"max_target_to_live_joint_position_delta_rad=1.9\),"
            r"left_gripper\(outside_steps=0,max_violation_rad=0.0,"
            r"max_target_to_live_joint_position_delta_rad=0.0\),"
            r"right_shoulder_pan\(outside_steps=0,max_violation_rad=0.0,"
            r"max_target_to_live_joint_position_delta_rad=0.0\),"
            r"right_shoulder_lift\(outside_steps=0,max_violation_rad=0.0,"
            r"max_target_to_live_joint_position_delta_rad=0.0\),"
            r"right_elbow_flex\(outside_steps=0,max_violation_rad=0.0,"
            r"max_target_to_live_joint_position_delta_rad=0.0\),"
            r"right_wrist_flex\(outside_steps=0,max_violation_rad=0.0,"
            r"max_target_to_live_joint_position_delta_rad=0.0\),"
            r"right_wrist_roll\(outside_steps=0,max_violation_rad=0.0,"
            r"max_target_to_live_joint_position_delta_rad=0.0\),"
            r"right_gripper\(outside_steps=0,max_violation_rad=0.0,"
            r"max_target_to_live_joint_position_delta_rad=0.0\)"
        ),
    ) as raised:
        namespace["_require_flywheel_cloth_health"](
            types.SimpleNamespace(
                flywheel_cloth_physical_health=lambda: {
                    "healthy": False,
                    "reason": "simulator_numerical_divergence",
                }
            ),
            policy_action_diagnostics=diagnostics,
        )

    assert "[" not in str(raised.value)


def test_cloth_failure_includes_bounded_action_projection_history() -> None:
    source_path = Path(__file__).resolve().parents[1] / "scripts/utils/evaluation.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    gate = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_require_flywheel_cloth_health"
    )
    module = ast.Module(body=[gate], type_ignores=[])
    ast.fix_missing_locations(module)
    error_type = type("SimulatorNumericalDivergenceError", (RuntimeError,), {})
    joint_names = (
        "left_shoulder_pan", "left_shoulder_lift", "left_elbow_flex",
        "left_wrist_flex", "left_wrist_roll", "left_gripper",
        "right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
        "right_wrist_flex", "right_wrist_roll", "right_gripper",
    )
    namespace = {
        "Any": object,
        "Mapping": Mapping,
        "SimulatorNumericalDivergenceError": error_type,
        "_FLYWHEEL_POLICY_ACTION_JOINT_NAMES": joint_names,
    }
    exec(compile(module, str(source_path), "exec"), namespace)

    with __import__("pytest").raises(error_type) as raised:
        namespace["_require_flywheel_cloth_health"](
            types.SimpleNamespace(
                flywheel_cloth_physical_health=lambda: {
                    "healthy": False,
                    "reason": "simulator_numerical_divergence",
                }
            ),
            policy_action_diagnostics={
                "policy_action_steps_projected": 2,
                "policy_action_max_simultaneous_projected_joint_count": 2,
                "policy_action_projected_joint_step_counts": {
                    name: 2 if name == "left_elbow_flex" else 0
                    for name in joint_names
                },
                "policy_action_max_abs_projection_rad": {
                    name: 0.25 if name == "left_elbow_flex" else 0.0
                    for name in joint_names
                },
            },
        )

    message = str(raised.value)
    assert "policy_action_steps_projected=2" in message
    assert "policy_action_max_simultaneous_projected_joint_count=2" in message
    assert (
        "left_elbow_flex(projected_steps=2,max_abs_projection_rad=0.25)"
        in message
    )
    assert "[" not in message


def test_policy_action_diagnostics_are_sampled_before_each_step_and_bound_to_health_failure() -> None:
    source_path = Path(__file__).resolve().parents[1] / "scripts/utils/evaluation.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    loop = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run_evaluation_loop"
    )
    loop_source = ast.get_source_segment(source, loop)
    assert loop_source is not None
    diagnostic_index = loop_source.index("_flywheel_policy_action_limit_diagnostics(env, action)")
    step_index = loop_source.index("env.step(action)")
    health_index = loop_source.index(
        "_require_flywheel_cloth_health(\n                    env, policy_action_diagnostics=policy_action_diagnostics\n                )"
    )
    assert diagnostic_index < step_index < health_index


def test_policy_action_diagnostics_accumulate_exact_per_joint_cumulative_fields() -> None:
    source_path = Path(__file__).resolve().parents[1] / "scripts/utils/evaluation.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    loop = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run_evaluation_loop"
    )
    loop_source = ast.get_source_segment(source, loop)
    assert loop_source is not None

    for field in (
        "policy_action_total_steps",
        "policy_action_outside_live_joint_limit_step_counts",
        "policy_action_max_limit_violation_rad",
        "policy_action_max_target_to_live_joint_position_delta_rad",
    ):
        assert field in loop_source


def test_policy_action_diagnostics_keep_prior_cumulative_evidence_after_unavailable_step() -> None:
    source_path = Path(__file__).resolve().parents[1] / "scripts/utils/evaluation.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    loop = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run_evaluation_loop"
    )
    unavailable_branch = next(
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and "policy_action_limits_available" in ast.unparse(node.test)
        and node.orelse
    )
    unavailable_source = "\n".join(
        ast.get_source_segment(source, statement) or ""
        for statement in unavailable_branch.orelse
    )

    for field in (
        "policy_action_total_steps",
        "policy_action_outside_live_joint_limit_step_counts",
        "policy_action_max_limit_violation_rad",
        "policy_action_max_target_to_live_joint_position_delta_rad",
    ):
        assert field in unavailable_source


def test_cloth_failure_reports_when_live_policy_limits_are_unavailable() -> None:
    source_path = Path(__file__).resolve().parents[1] / "scripts/utils/evaluation.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    gate = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_require_flywheel_cloth_health"
    )
    module = ast.Module(body=[gate], type_ignores=[])
    ast.fix_missing_locations(module)
    error_type = type("SimulatorNumericalDivergenceError", (RuntimeError,), {})
    namespace = {
        "Any": object,
        "Mapping": Mapping,
        "SimulatorNumericalDivergenceError": error_type,
        "_FLYWHEEL_POLICY_ACTION_JOINT_NAMES": (
            "left_shoulder_pan", "left_shoulder_lift", "left_elbow_flex",
            "left_wrist_flex", "left_wrist_roll", "left_gripper",
            "right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
            "right_wrist_flex", "right_wrist_roll", "right_gripper",
        ),
    }
    exec(compile(module, str(source_path), "exec"), namespace)

    with __import__("pytest").raises(
        error_type, match=r"policy_action_limits_available=False"
    ):
        namespace["_require_flywheel_cloth_health"](
            types.SimpleNamespace(
                flywheel_cloth_physical_health=lambda: {
                    "healthy": False,
                    "reason": "simulator_numerical_divergence",
                }
            ),
            policy_action_diagnostics={"policy_action_limits_available": False},
        )
