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
        "reason": "unsupported_dynamic_triangle_mesh_collider",
        "metric_name": "dynamic_triangle_mesh_collider_count",
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
        "metric_name": "dynamic_triangle_mesh_collider_count",
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
        "reason": "unsupported_dynamic_triangle_mesh_collider",
        "metric_name": "dynamic_triangle_mesh_collider_count",
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
    }
    exec(compile(gate_module, str(evaluation_source_path), "exec"), gate_namespace)

    with __import__("pytest").raises(
        error_type,
        match=(
            r"dynamic_triangle_mesh_collider_count=1 limit=0; .*"
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
        "metric_name": "dynamic_triangle_mesh_collider_count",
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


def test_cpu_visible_contact_uses_the_same_authoritative_physx_readback_as_runtime_evidence() -> None:
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
    assert "self._flywheel_physics_cloth_state()" in method_source
    assert "self._flywheel_cpu_cloth_state()" not in method_source
    assert "get_current_mesh_points" not in method_source


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


def _cloth_evidence(env) -> None:
    if not hasattr(env, "renderer_device"):
        env.renderer_device = "cuda:0"
    if not hasattr(env, "camera_device"):
        env.camera_device = env.renderer_device
    env._flywheel_cloth_backend = lambda: "physx_cloth_view"
    env._flywheel_physics_cloth_state = lambda: ([0.0], [0.0])
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
    }


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
    _cloth_evidence(env)
    args = types.SimpleNamespace(device="cpu", renderer_device="cuda:0", camera_device="cuda:0")
    policy = types.SimpleNamespace(runtime_device="cuda:0")
    session = evaluation.EvaluationSession(args, env=env, policy=policy, env_cfg=env.cfg)

    assert session.runtime_receipt["simulation_device"] == "cpu"
    assert session.runtime_receipt["cloth_device"] == "cpu"
    assert session.runtime_receipt["renderer_device"] == "cuda:0"
    assert session.runtime_receipt["camera_device"] == "cuda:0"
    assert session.runtime_receipt["policy_device"] == "cuda:0"


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
    snapshot.write_text(json.dumps({"schema_version": 1, "garment_name": "Top_Long_Seen_0"}), encoding="utf-8")
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
        },
        policy=object(),
        attempt_output_dir=attempt,
    )

    assert captured[0].restore_snapshot == json.loads(snapshot.read_text(encoding="utf-8"))
    manifest = json.loads((attempt / "flywheel-manifest.json").read_text(encoding="utf-8"))
    assert manifest["restore_snapshot"] == str(snapshot)
    assert manifest["restore_snapshot_sha256"] == digest
    assert manifest["parent_episode_id"] == "original-top-long-1"
    assert manifest["lineage_id"] == "original-top-long-1"
    assert manifest["replay_kind"] == "verified_success_reset_v1"


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
