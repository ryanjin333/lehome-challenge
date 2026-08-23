from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


def _snapshot() -> dict[str, object]:
    return {
        "schema_version": 2, "robot_position": [0.0] * 12,
        "robot_velocity": [0.0] * 12, "cloth_position": [[0.0, 0.0, 0.0]],
        "cloth_velocity": [[0.0, 0.0, 0.0]], "rng_state": {},
        "garment_name": "Top_Long_Seen_0", "randomization": {"strategy": "canonical"},
        "scene_state": {}, "cloth_state_authority": "physx_cloth_view_world_v1",
    }


def _state_fingerprint(*, category: str, garment: str, state: list[float]) -> str:
    rounded = ["0.000000" if value == 0.0 else format(value, ".6f") for value in state]
    return hashlib.sha256(json.dumps({"category": category, "garment": garment, "state_rounding": "fixed_6dp", "state": rounded}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _assignment(tmp_path, *, state: list[float] | None = None, teacher: bool = False) -> dict[str, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    reset = tmp_path / "reset.json"; continuation = tmp_path / "continuation.json"; annotations = tmp_path / "annotations.jsonl"
    source_state = state or [0.0] * 12
    source_snapshot = _snapshot() | {"robot_position": source_state, "randomization": {"strategy": "canonical", "continuation_step": 16}}
    reset.write_text(json.dumps(_snapshot()), encoding="utf-8")
    continuation.write_text(json.dumps(source_snapshot), encoding="utf-8")
    annotations.write_text("".join(json.dumps({"step": step, "action": [float(step)] * 12, "success": step == 19}) + "\n" for step in range(20)), encoding="utf-8")
    category, garment = "top_long", "Top_Long_Seen_0"
    return {
        "recovery_kind": "controlled_success_recovery_snapshot_v3", "source_reset": str(reset), "source_reset_sha256": hashlib.sha256(reset.read_bytes()).hexdigest(),
        "source_continuation_snapshot": str(continuation), "source_continuation_snapshot_sha256": hashlib.sha256(continuation.read_bytes()).hexdigest(),
        "source_annotations": str(annotations), "source_annotations_sha256": hashlib.sha256(annotations.read_bytes()).hexdigest(), "prefix_stop": 16, "source_first_success_step": 19,
        "perturbation_profile": {"cloth_displacement_m": 0.002, "cloth_velocity_mps": 0.01, "gripper_offset_rad": 0.02}, "perturbation_seed": 7,
        "source_round_id": "round", "source_episode_id": "episode", "source_episode_digest": "a" * 64, "source_immutable_revision": "b" * 40,
        "category": category, "garment": garment, "source_seed": 50110, "source_continuation_state": source_state,
        "source_snapshot_schema_version": 2, "source_snapshot_authority": "physx_cloth_view_world_v1", "source_only_envelope": False,
        "source_state_fingerprint": _state_fingerprint(category=category, garment=garment, state=source_state), "perturbation_fingerprint": "d" * 64, "source_state_perturbation_fingerprint": "e" * 64,
        **({"controlled_smoke": True, "controlled_smoke_teacher_probe": True} if teacher else {}),
    }


def _smoke_descriptor_row(tmp_path: Path) -> dict[str, object]:
    row = _assignment(tmp_path)
    run_id, matrix_sha256, materialization_sha256 = "a" * 32, "b" * 64, "c" * 64
    identity = hashlib.sha256(f"{run_id}:{matrix_sha256}:{materialization_sha256}".encode("ascii")).hexdigest()[:20]
    mode = "bounded_perturbation_v1"
    row.update({
        "attempt_id": "smoke-attempt", "trial_id": "smoke-trial", "strategy": "canonical",
        "category_acceptance_cap": 1, "controlled_matrix_sha256": matrix_sha256,
        "controlled_smoke": True, "controlled_smoke_run_id": run_id,
        "controlled_smoke_row_index": 0, "controlled_smoke_identity": identity,
        "controlled_smoke_mode_identity": hashlib.sha256(f"{identity}:{mode}".encode("ascii")).hexdigest()[:20],
        "controlled_smoke_perturbation_mode": mode, "controlled_smoke_zero_perturbation": False,
        "controlled_smoke_teacher_probe": False,
        "controlled_smoke_matrix_sha256": matrix_sha256,
        "controlled_smoke_materialization_sha256": materialization_sha256,
    })
    return row


def test_controlled_replay_is_exact_and_recorder_state_is_after_bootstrap() -> None:
    from lehome.flywheel.recovery_collection import (
        apply_controlled_perturbation, replay_action_prefix,
    )

    class Env:
        def __init__(self) -> None:
            self.actions: list[object] = []

        def step(self, action: object) -> None:
            self.actions.append(action)

    env = Env()
    actions = [[float(step)] * 12 for step in range(3)]
    replay_action_prefix(env, actions)
    assert env.actions == actions

    perturbed = apply_controlled_perturbation(
        _snapshot(), {"cloth_displacement_m": 0.002, "cloth_velocity_mps": 0.01, "gripper_offset_rad": 0.02}, 7,
    )
    assert perturbed.robot_position != _snapshot()["robot_position"]
    assert perturbed.cloth_position != _snapshot()["cloth_position"]


def test_snapshot_source_descriptor_authenticates_an_explicit_legacy_cpu_restore(tmp_path: Path) -> None:
    from lehome.flywheel.recovery_collection import validate_snapshot_source_descriptor

    reset = tmp_path / "historical-reset.json"
    reset.write_text(json.dumps({
        "schema_version": 1,
        "robot_position": [0.0] * 12,
        "robot_velocity": [0.0] * 12,
        "cloth_position": [[0.0, 0.0, 0.0]],
        "cloth_velocity": [[0.0, 0.0, 0.0]],
        "rng_state": {},
        "garment_name": "Pant_Short_Seen_5",
        "randomization": {"strategy": "canonical"},
        "scene_state": {"garment_reset_pose": [0.0, 0.0, 0.67, 0.0, 0.0, 90.0]},
    }), encoding="utf-8")
    row = {
        "snapshot_source_bootstrap": True,
        "category": "pant_short",
        "garment": "Pant_Short_Seen_5",
        "seed": 131,
        "source_seed": 131,
        "replay_kind": "verified_success_reset_v1",
        "restore_snapshot": str(reset),
        "restore_snapshot_sha256": hashlib.sha256(reset.read_bytes()).hexdigest(),
        "restore_snapshot_cloth_frame": "usd_local_points_v1",
        "parent_episode_id": "parent-success",
        "lineage_id": "parent-success",
    }
    descriptor = tmp_path / "descriptor.json"
    descriptor.write_text(json.dumps([row]), encoding="utf-8")

    validated = validate_snapshot_source_descriptor(descriptor)

    assert validated["restore_snapshot_cloth_frame"] == "usd_local_points_v1"
    row.pop("restore_snapshot_cloth_frame")
    descriptor.write_text(json.dumps([row]), encoding="utf-8")
    with pytest.raises(ValueError, match="cloth frame"):
        validate_snapshot_source_descriptor(descriptor)


def test_snapshot_source_discovery_descriptor_admits_bounded_same_category_ordinary_rows(tmp_path: Path) -> None:
    from lehome.flywheel.recovery_collection import validate_snapshot_source_discovery_descriptor

    descriptor = tmp_path / "discovery.json"
    rows = [
        {
            "snapshot_source_bootstrap": True,
            "category": "top_long",
            "garment": "Top_Long_Seen_0",
            "seed": 107 + index,
            "source_seed": 107 + index,
        }
        for index in range(3)
    ]
    descriptor.write_text(json.dumps(rows), encoding="utf-8")

    assert validate_snapshot_source_discovery_descriptor(descriptor) == rows


def test_snapshot_source_discovery_descriptor_rejects_mixed_or_controlled_rows(tmp_path: Path) -> None:
    from lehome.flywheel.recovery_collection import validate_snapshot_source_discovery_descriptor

    descriptor = tmp_path / "discovery.json"
    descriptor.write_text(json.dumps([
        {"snapshot_source_bootstrap": True, "category": "top_long", "garment": "Top_Long_Seen_0", "seed": 107, "source_seed": 107},
        {"snapshot_source_bootstrap": True, "category": "top_short", "garment": "Top_Short_Seen_0", "seed": 108, "source_seed": 108, "controlled_smoke": True},
    ]), encoding="utf-8")

    with pytest.raises(ValueError, match="discovery descriptor"):
        validate_snapshot_source_discovery_descriptor(descriptor)


def test_snapshot_source_discovery_descriptor_rejects_hidden_recovery_state(tmp_path: Path) -> None:
    from lehome.flywheel.recovery_collection import validate_snapshot_source_discovery_descriptor

    descriptor = tmp_path / "discovery.json"
    descriptor.write_text(json.dumps([{
        "snapshot_source_bootstrap": True,
        "category": "top_long",
        "garment": "Top_Long_Seen_0",
        "seed": 107,
        "source_seed": 107,
        "source_continuation_state": [0.0] * 12,
    }]), encoding="utf-8")

    with pytest.raises(ValueError, match="ordinary autonomous"):
        validate_snapshot_source_discovery_descriptor(descriptor)


def test_snapshot_source_discovery_descriptor_rejects_an_inconsistent_garment_alias(tmp_path: Path) -> None:
    from lehome.flywheel.recovery_collection import validate_snapshot_source_discovery_descriptor

    descriptor = tmp_path / "discovery.json"
    descriptor.write_text(json.dumps([{
        "snapshot_source_bootstrap": True,
        "category": "top_long",
        "garment": "Top_Long_Seen_0",
        "garment_name": "Top_Long_Seen_1",
        "seed": 107,
        "source_seed": 107,
    }]), encoding="utf-8")

    with pytest.raises(ValueError, match="garment identity"):
        validate_snapshot_source_discovery_descriptor(descriptor)


@pytest.mark.parametrize(("seed", "source_seed"), [(1, True), (1, 1.0)])
def test_snapshot_source_discovery_descriptor_requires_an_exact_integer_source_seed(
    tmp_path: Path, seed: int, source_seed: object,
) -> None:
    from lehome.flywheel.recovery_collection import validate_snapshot_source_discovery_descriptor

    descriptor = tmp_path / "discovery.json"
    descriptor.write_text(json.dumps([{
        "snapshot_source_bootstrap": True,
        "category": "top_long",
        "garment": "Top_Long_Seen_0",
        "seed": seed,
        "source_seed": source_seed,
    }]), encoding="utf-8")

    with pytest.raises(ValueError, match="lineage"):
        validate_snapshot_source_discovery_descriptor(descriptor)


@pytest.mark.parametrize(
    ("category", "garment"),
    [
        ("top_long", "Top_Short_Seen_0"),
        ("top_short", "Top_Long_Seen_0"),
        ("pant_long", "Pant_Short_Seen_0"),
        ("pant_short", "Pant_Long_Seen_0"),
        ("top_short", "Top_Short_Unseen_0"),
        ("pant_long", "../Pant_Long_Seen_0"),
        ("pant_short", "Pant_Short_Seen_-1"),
    ],
)
def test_snapshot_source_discovery_descriptor_requires_a_canonical_seen_garment(
    tmp_path: Path, category: str, garment: str,
) -> None:
    from lehome.flywheel.recovery_collection import validate_snapshot_source_discovery_descriptor

    descriptor = tmp_path / "discovery.json"
    descriptor.write_text(json.dumps([{
        "snapshot_source_bootstrap": True,
        "category": category,
        "garment": garment,
        "seed": 107,
        "source_seed": 107,
    }]), encoding="utf-8")

    with pytest.raises(ValueError, match="garment identity"):
        validate_snapshot_source_discovery_descriptor(descriptor)


def test_controlled_recovery_rejects_invalid_actions_and_perturbation_bounds() -> None:
    from lehome.flywheel.recovery_collection import (
        apply_controlled_perturbation, replay_action_prefix,
    )

    with pytest.raises(ValueError, match="12-D"):
        replay_action_prefix(object(), [[0.0] * 11])
    with pytest.raises(ValueError, match="finite"):
        replay_action_prefix(object(), [[float("nan")] * 12])
    with pytest.raises(ValueError, match="bound"):
        apply_controlled_perturbation(_snapshot(), {"cloth_displacement_m": 1.0, "cloth_velocity_mps": 0.0, "gripper_offset_rad": 0.0}, 1)


def test_verified_controlled_lineage_rejects_tampered_and_unsafe_files(tmp_path) -> None:
    from lehome.flywheel.recovery_collection import load_controlled_recovery

    assignment = _assignment(tmp_path)
    recovery = load_controlled_recovery(assignment)
    assert recovery.continuation_snapshot.robot_position == (0.0,) * 12
    annotations = Path(assignment["source_annotations"])
    annotations.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_controlled_recovery(assignment)
    annotations.unlink()
    annotations.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="absolute regular file"):
        load_controlled_recovery(assignment)


def test_controlled_runtime_binds_the_physical_snapshot_to_the_exact_h16_boundary_and_reset(tmp_path) -> None:
    from lehome.flywheel.recovery_collection import load_controlled_recovery

    wrong_step = _assignment(tmp_path / "wrong-step")
    continuation = Path(wrong_step["source_continuation_snapshot"])
    payload = json.loads(continuation.read_text(encoding="utf-8"))
    payload["randomization"]["continuation_step"] = 32
    continuation.write_text(json.dumps(payload), encoding="utf-8")
    wrong_step["source_continuation_snapshot_sha256"] = hashlib.sha256(continuation.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="H16 next-action boundary"):
        load_controlled_recovery(wrong_step)

    wrong_reset = _assignment(tmp_path / "wrong-reset")
    reset = Path(wrong_reset["source_reset"])
    payload = json.loads(reset.read_text(encoding="utf-8"))
    payload["randomization"] = {"strategy": "geometry"}
    reset.write_text(json.dumps(payload), encoding="utf-8")
    wrong_reset["source_reset_sha256"] = hashlib.sha256(reset.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="randomization"):
        load_controlled_recovery(wrong_reset)


def test_production_bootstrap_restores_prefixes_then_perturbs_before_policy_continuation(tmp_path) -> None:
    from lehome.flywheel.recovery_collection import bootstrap_controlled_recovery

    assignment = _assignment(tmp_path)

    class Env:
        def __init__(self) -> None:
            self.state = _snapshot(); self.actions: list[list[float]] = []
        def flywheel_restore_state(self, snapshot) -> None:
            self.state = snapshot.to_dict()
        def flywheel_capture_state(self):
            return self.state
        def step(self, action) -> None:
            self.actions.append(action)

    env = Env()
    provenance = bootstrap_controlled_recovery(env, assignment)
    assert env.actions == []
    assert env.state["cloth_position"] != _snapshot()["cloth_position"]
    assert provenance["source_episode_id"] == "episode"
    fidelity = provenance["replay_fidelity"]
    assert fidelity["expected_state_sha256"] == fidelity["observed_state_sha256"]


def test_controlled_recovery_fails_closed_for_missing_or_tampered_continuation_state_before_mutation(tmp_path) -> None:
    from lehome.flywheel.recovery_collection import load_controlled_recovery

    missing = _assignment(tmp_path / "missing")
    missing.pop("source_continuation_state")
    with pytest.raises(ValueError, match="continuation state"):
        load_controlled_recovery(missing)
    tampered = _assignment(tmp_path / "tampered")
    tampered["source_continuation_state"] = [0.1] * 12
    with pytest.raises(ValueError, match="fingerprint"):
        load_controlled_recovery(tampered)


def test_controlled_recovery_replay_fidelity_accepts_tolerance_and_rejects_drift_before_perturbation(tmp_path) -> None:
    from lehome.flywheel.recovery_collection import bootstrap_controlled_recovery

    class Env:
        def __init__(self, drift: float) -> None:
            self.state = _snapshot(); self.actions: list[list[float]] = []; self.drift = drift
        def flywheel_restore_state(self, snapshot) -> None: self.state = snapshot.to_dict()
        def step(self, action) -> None: self.actions.append(action)
        def flywheel_capture_state(self):
            observed = dict(self.state); observed["robot_position"] = [self.drift] * 12
            return observed

    accepted = Env(0.0049)
    provenance = bootstrap_controlled_recovery(accepted, _assignment(tmp_path / "accepted"))
    check = provenance["replay_fidelity"]
    assert check["verified"] is True and check["tolerance_rad"] == 0.005 and check["max_abs_error_rad"] == pytest.approx(0.0049)
    assert len(check["expected_state_sha256"]) == len(check["observed_state_sha256"]) == 64
    rejected = Env(0.0051)
    with pytest.raises(ValueError, match="replay fidelity"):
        bootstrap_controlled_recovery(rejected, _assignment(tmp_path / "rejected"))
    assert rejected.state["cloth_position"] == _snapshot()["cloth_position"]


def test_controlled_recovery_replay_fidelity_rejects_robot_velocity_drift_before_perturbation(tmp_path) -> None:
    from lehome.flywheel.recovery_collection import bootstrap_controlled_recovery

    class Env:
        def __init__(self) -> None:
            self.state = _snapshot()

        def flywheel_restore_state(self, snapshot) -> None:
            self.state = snapshot.to_dict()

        def flywheel_capture_state(self):
            observed = dict(self.state)
            observed["robot_velocity"] = [0.0051] * 12
            return observed

    env = Env()
    with pytest.raises(ValueError, match="robot-velocity tolerance"):
        bootstrap_controlled_recovery(env, _assignment(tmp_path / "robot-velocity"))
    assert env.state["cloth_position"] == _snapshot()["cloth_position"]


@pytest.mark.parametrize(
    ("field", "drift", "message"),
    [
        ("cloth_position", 1.1e-5, "cloth-position tolerance"),
        ("cloth_velocity", 1.1e-5, "cloth-velocity tolerance"),
    ],
)
def test_controlled_recovery_rejects_live_cloth_drift_before_teacher_replay(
    tmp_path, field: str, drift: float, message: str,
) -> None:
    from lehome.flywheel.recovery_collection import bootstrap_controlled_recovery

    class Env:
        def __init__(self) -> None:
            self.state = _snapshot()
            self.actions: list[list[float]] = []

        def flywheel_restore_state(self, snapshot) -> None:
            self.state = snapshot.to_dict()

        def flywheel_capture_state(self):
            observed = json.loads(json.dumps(self.state))
            observed[field][0][0] += drift
            return observed

        def step(self, action) -> None:
            self.actions.append(action)

    env = Env()
    with pytest.raises(ValueError, match=message):
        bootstrap_controlled_recovery(
            env, _assignment(tmp_path / field, teacher=True)
        )
    assert env.actions == []


def test_controlled_recovery_rejects_legacy_usd_snapshots_before_mutation(tmp_path) -> None:
    from lehome.flywheel.recovery_collection import load_controlled_recovery

    assignment = _assignment(tmp_path)
    for field in ("source_reset", "source_continuation_snapshot"):
        path = Path(assignment[field])
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["schema_version"] = 1
        payload.pop("cloth_state_authority")
        path.write_text(json.dumps(payload), encoding="utf-8")
        assignment[f"{field}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="invalid or mixed cloth authority"):
        load_controlled_recovery(assignment)


def test_controlled_recovery_accepts_checksum_bound_usd_local_v3_sources(tmp_path) -> None:
    from lehome.flywheel.recovery_collection import load_controlled_recovery

    assignment = _assignment(tmp_path)
    for field in ("source_reset", "source_continuation_snapshot"):
        path = Path(assignment[field])
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["schema_version"] = 3
        payload["cloth_state_authority"] = "usd_local_points_v1"
        path.write_text(json.dumps(payload), encoding="utf-8")
        assignment[f"{field}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    assignment["source_snapshot_schema_version"] = 3
    assignment["source_snapshot_authority"] = "usd_local_points_v1"

    with pytest.raises(ValueError, match="source envelope"):
        load_controlled_recovery(assignment)
    assignment["source_only_envelope"] = True

    recovery = load_controlled_recovery(assignment)

    assert recovery.continuation_snapshot.schema_version == 3
    assert recovery.provenance["source_snapshot_authority"] == "usd_local_points_v1"


def test_cuda_legacy_projection_binds_the_projected_readback_and_is_idempotent(tmp_path) -> None:
    from lehome.flywheel.recovery_collection import bootstrap_controlled_recovery

    assignment = _assignment(tmp_path, teacher=True)
    for field in ("source_reset", "source_continuation_snapshot"):
        path = Path(assignment[field])
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["schema_version"] = 3
        payload["cloth_state_authority"] = "usd_local_points_v1"
        path.write_text(json.dumps(payload), encoding="utf-8")
        assignment[f"{field}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    assignment["source_snapshot_schema_version"] = 3
    assignment["source_snapshot_authority"] = "usd_local_points_v1"
    assignment["source_only_envelope"] = True

    class Env:
        def __init__(self) -> None:
            self.state = _snapshot()
            self.legacy_restores = 0

        def flywheel_restore_state(self, snapshot) -> None:
            if snapshot.schema_version == 3:
                self.legacy_restores += 1
                self._flywheel_legacy_projection_receipt = {
                    "source_snapshot_authority": "usd_local_points_v1",
                    "weld_map_identity": "a" * 64,
                    "welded_vertices_remap_to_orig_sha256": "b" * 64,
                    "welded_vertices_remap_to_weld_sha256": "c" * 64,
                }
                self.state = _snapshot() | {
                    "robot_position": list(snapshot.robot_position),
                    "robot_velocity": list(snapshot.robot_velocity),
                }
            else:
                self.state = snapshot.to_dict()

        def flywheel_capture_state(self):
            return self.state

        def step(self, _action) -> None:
            return None

        def _get_success(self) -> bool:
            return True

    env = Env()
    provenance = bootstrap_controlled_recovery(env, assignment)

    projected = provenance["legacy_usd_projection"]
    assert env.legacy_restores == 3
    assert projected["source_snapshot_sha256"] == assignment["source_continuation_snapshot_sha256"]
    assert projected["source_snapshot_authority"] == "usd_local_points_v1"
    assert projected["projected_cloth_state_authority"] == "physx_cloth_view_world_v1"
    assert projected["projected_physical_state_sha256"] == provenance["replay_fidelity"]["expected_state_sha256"]


def test_smoke_teacher_probe_requires_success_then_reconstructs_the_verified_boundary(tmp_path) -> None:
    from lehome.flywheel.recovery_collection import bootstrap_controlled_recovery

    class Env:
        def __init__(self, teacher_success: bool) -> None:
            self.state = _snapshot(); self.actions: list[list[float]] = []; self.teacher_success = teacher_success; self.success_checks = 0
        def flywheel_restore_state(self, snapshot) -> None: self.state = snapshot.to_dict()
        def flywheel_capture_state(self): return self.state
        def step(self, action) -> None:
            self.actions.append(action); self.state["robot_position"] = [0.0] * 12
        def _get_success(self): self.success_checks += 1; return self.teacher_success

    env = Env(True)
    provenance = bootstrap_controlled_recovery(env, _assignment(tmp_path / "teacher", teacher=True))
    assert env.actions == [[16.0] * 12, [17.0] * 12, [18.0] * 12, [19.0] * 12]
    assert provenance["teacher_probe"]["verified"] is True
    assert len(provenance["replay_fidelity_checks"]) == 2
    with pytest.raises(ValueError, match="teacher probe"):
        bootstrap_controlled_recovery(Env(False), _assignment(tmp_path / "teacher-fail", teacher=True))


def test_teacher_probe_rejects_a_source_that_does_not_mark_its_declared_first_success(tmp_path) -> None:
    from lehome.flywheel.recovery_collection import load_controlled_recovery

    assignment = _assignment(tmp_path / "source-success", teacher=True)
    annotations = Path(assignment["source_annotations"])
    rows = [json.loads(line) for line in annotations.read_text(encoding="utf-8").splitlines()]
    rows[19]["success"] = False
    annotations.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assignment["source_annotations_sha256"] = hashlib.sha256(annotations.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="first recorded success"):
        load_controlled_recovery(assignment)


def test_single_materialization_loader_returns_hydrated_rows_and_rejects_identity_tampering(tmp_path) -> None:
    from lehome.flywheel.recovery_collection import load_attempt_matrix

    reset, annotations, continuation = tmp_path / "reset.json", tmp_path / "annotations.jsonl", tmp_path / "continuation.json"
    reset.write_text("{}", encoding="utf-8"); annotations.write_text("", encoding="utf-8"); continuation.write_text("{}", encoding="utf-8")
    categories = ["pant_long"] * 4 + ["top_long"] + ["top_short"] * 3
    caps = {"pant_long": 4, "top_long": 1, "top_short": 3, "pant_short": 0}
    rows = [{"attempt_id": f"controlled-{index}", "trial_id": f"controlled-{index}", "category": category, "category_acceptance_cap": caps[category], "strategy": "canonical", "recovery_kind": "controlled_success_recovery_snapshot_v3", "controlled_matrix_sha256": "a" * 64, "perturbation_seed": index, "perturbation_fingerprint": f"{index + 100:064x}", "source_state_perturbation_fingerprint": f"{index + 200:064x}", "source_seed": 50110, "source_continuation_state": [0.0] * 12, "source_snapshot_schema_version": 2, "source_snapshot_authority": "physx_cloth_view_world_v1", "source_only_envelope": False, "source_state_fingerprint": f"{index + 300:064x}", "source_round_id": "round", "source_episode_id": f"episode-{index}", "source_episode_digest": f"{index + 400:064x}", "source_reset_sha256": "a" * 64, "source_annotations_sha256": "b" * 64, "source_continuation_snapshot_sha256": "c" * 64, "prefix_stop": 16, "source_first_success_step": 19, "source_reset": str(reset), "source_annotations": str(annotations), "source_continuation_snapshot": str(continuation)} for index, category in enumerate(categories)]
    descriptor = tmp_path / "materialization.json"
    payload = {"schema_version": 3, "kind": "controlled_success_recovery_materialization_v3", "matrix_sha256": "a" * 64, "target_accepted": 8, "category_acceptance_caps": caps, "rows": rows}
    descriptor.write_text(json.dumps(payload), encoding="utf-8")
    assert load_attempt_matrix(descriptor) == rows
    linked = tmp_path / "linked"; linked.symlink_to(tmp_path, target_is_directory=True)
    rows[0]["source_continuation_snapshot"] = str(linked / "continuation.json")
    descriptor.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="source path is unsafe"):
        load_attempt_matrix(descriptor)
    rows[0]["source_continuation_snapshot"] = str(continuation)
    rows[0]["controlled_matrix_sha256"] = "b" * 64
    descriptor.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="matrix hash"):
        load_attempt_matrix(descriptor)


def test_materialization_loader_rejects_unreachable_or_noncanonical_controlled_schedule(tmp_path) -> None:
    from lehome.flywheel.recovery_collection import load_attempt_matrix

    reset, annotations, continuation = tmp_path / "reset.json", tmp_path / "annotations.jsonl", tmp_path / "continuation.json"
    reset.write_text("{}", encoding="utf-8"); annotations.write_text("", encoding="utf-8"); continuation.write_text("{}", encoding="utf-8")
    rows = [
        {
            "attempt_id": f"attempt-{index}", "trial_id": f"trial-{index}",
            "category": "pant_long", "category_acceptance_cap": 4,
            "strategy": "canonical", "recovery_kind": "controlled_success_recovery_snapshot_v3",
            "controlled_matrix_sha256": "a" * 64, "perturbation_seed": index,
            "perturbation_fingerprint": f"{index + 100:064x}",
            "source_state_perturbation_fingerprint": f"{index + 200:064x}",
                "source_seed": 50110, "source_continuation_state": [0.0] * 12, "source_snapshot_schema_version": 2, "source_snapshot_authority": "physx_cloth_view_world_v1", "source_only_envelope": False, "source_state_fingerprint": f"{index + 300:064x}", "source_round_id": "round",
            "source_episode_id": f"episode-{index}", "source_episode_digest": f"{index + 400:064x}",
            "source_reset_sha256": "a" * 64, "source_annotations_sha256": "b" * 64,
            "source_continuation_snapshot_sha256": "c" * 64, "prefix_stop": 16,
            "source_first_success_step": 19, "source_reset": str(reset),
            "source_annotations": str(annotations), "source_continuation_snapshot": str(continuation),
        }
        for index in range(8)
    ]
    path = tmp_path / "bad-materialization.json"
    path.write_text(json.dumps({"schema_version": 3, "kind": "controlled_success_recovery_materialization_v3", "matrix_sha256": "a" * 64, "target_accepted": 8, "category_acceptance_caps": {"pant_long": 4, "top_long": 1, "top_short": 3, "pant_short": 0}, "rows": rows}), encoding="utf-8")
    with pytest.raises(ValueError, match="reachable|category"):
        load_attempt_matrix(path)


def test_loader_rejects_legacy_controlled_list_but_keeps_ordinary_lists(tmp_path) -> None:
    from lehome.flywheel.recovery_collection import load_attempt_matrix

    legacy = tmp_path / "legacy-controlled.json"
    legacy.write_text(json.dumps([{"recovery_kind": "controlled_success_recovery_v1", "prefix_stop": 16}]), encoding="utf-8")
    with pytest.raises(ValueError, match="legacy or incompatible"):
        load_attempt_matrix(legacy)
    ordinary = tmp_path / "ordinary.json"
    ordinary.write_text(json.dumps([{"garment": "Top_Long_Seen_0", "seed": 7}]), encoding="utf-8")
    assert load_attempt_matrix(ordinary) == [{"garment": "Top_Long_Seen_0", "seed": 7}]


def test_loader_admits_only_the_exact_single_v2_smoke_descriptor_list(tmp_path: Path) -> None:
    from lehome.flywheel.recovery_collection import load_attempt_matrix

    arbitrary = tmp_path / "arbitrary-v2.json"
    arbitrary.write_text(json.dumps([_assignment(tmp_path / "arbitrary")]), encoding="utf-8")
    with pytest.raises(ValueError, match="controlled smoke"):
        load_attempt_matrix(arbitrary)
    row = _smoke_descriptor_row(tmp_path / "smoke")
    multiple = tmp_path / "multiple-v2.json"
    multiple.write_text(json.dumps([row, row]), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        load_attempt_matrix(multiple)
    exact = tmp_path / "exact-smoke.json"
    exact.write_text(json.dumps([row]), encoding="utf-8")
    assert load_attempt_matrix(exact) == [row]


def test_controlled_recovery_rejects_parent_symlink_inputs_before_mutation(tmp_path) -> None:
    from lehome.flywheel.recovery_collection import load_controlled_recovery

    real = tmp_path / "real"; real.mkdir()
    assignment = _assignment(real)
    parent = tmp_path / "linked"; parent.symlink_to(real, target_is_directory=True)
    assignment["source_continuation_snapshot"] = str(parent / "continuation.json")
    with pytest.raises(ValueError, match="absolute regular file"):
        load_controlled_recovery(assignment)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "schema_version": 3,
                "kind": "controlled_success_recovery_materialization_v3",
                "matrix_sha256": "a" * 64,
                "target_accepted": 8,
                "category_acceptance_caps": {"pant_long": 4, "top_long": 1, "top_short": 3, "pant_short": 0},
                "rows": [{}],
            },
            "rows are invalid",
        ),
        (
            {
                "schema_version": 1,
                "kind": "controlled_success_recovery_materialization_v1",
                "matrix_sha256": "a" * 64,
                "target_accepted": 8,
                "category_acceptance_caps": {"pant_long": 4, "top_long": 1, "top_short": 3, "pant_short": 0},
                "rows": [],
                "unexpected": True,
            },
            "array or controlled materialization",
        ),
    ],
)
def test_materialization_loader_rejects_one_row_and_malformed_envelopes(tmp_path, payload, message) -> None:
    from lehome.flywheel.recovery_collection import load_attempt_matrix

    descriptor = tmp_path / "invalid-materialization.json"
    descriptor.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_attempt_matrix(descriptor)
