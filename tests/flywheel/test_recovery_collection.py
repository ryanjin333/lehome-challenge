from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


def _snapshot() -> dict[str, object]:
    return {
        "schema_version": 1, "robot_position": [0.0] * 12,
        "robot_velocity": [0.0] * 12, "cloth_position": [[0.0, 0.0, 0.0]],
        "cloth_velocity": [[0.0, 0.0, 0.0]], "rng_state": {},
        "garment_name": "Top_Long_Seen_0",
    }


def _state_fingerprint(*, category: str, garment: str, state: list[float]) -> str:
    rounded = ["0.000000" if value == 0.0 else format(value, ".6f") for value in state]
    return hashlib.sha256(json.dumps({"category": category, "garment": garment, "state_rounding": "fixed_6dp", "state": rounded}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _assignment(tmp_path, *, state: list[float] | None = None, teacher: bool = False) -> dict[str, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    reset = tmp_path / "reset.json"; annotations = tmp_path / "annotations.jsonl"
    reset.write_text(json.dumps(_snapshot()), encoding="utf-8")
    annotations.write_text("".join(json.dumps({"step": step, "action": [float(step)] * 12, "success": step == 3}) + "\n" for step in range(4)), encoding="utf-8")
    source_state = state or [0.0] * 12
    category, garment = "top_long", "Top_Long_Seen_0"
    return {
        "recovery_kind": "controlled_success_recovery_v1", "source_reset": str(reset), "source_reset_sha256": hashlib.sha256(reset.read_bytes()).hexdigest(),
        "source_annotations": str(annotations), "source_annotations_sha256": hashlib.sha256(annotations.read_bytes()).hexdigest(), "prefix_stop": 2, "source_first_success_step": 3,
        "action_prefix_sha256": hashlib.sha256((json.dumps([[0.0] * 12, [1.0] * 12], sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest(),
        "perturbation_profile": {"cloth_displacement_m": 0.002, "cloth_velocity_mps": 0.01, "gripper_offset_rad": 0.02}, "perturbation_seed": 7,
        "source_round_id": "round", "source_episode_id": "episode", "source_episode_digest": "a" * 64, "source_immutable_revision": "b" * 40,
        "category": category, "garment": garment, "source_continuation_state": source_state,
        "source_state_fingerprint": _state_fingerprint(category=category, garment=garment, state=source_state), "perturbation_fingerprint": "d" * 64, "source_state_perturbation_fingerprint": "e" * 64,
        **({"controlled_smoke": True, "controlled_smoke_teacher_probe": True} if teacher else {}),
    }


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

    reset = tmp_path / "reset.json"
    annotations = tmp_path / "annotations.jsonl"
    reset.write_text(json.dumps(_snapshot()), encoding="utf-8")
    annotations.write_text("".join(json.dumps({"step": step, "action": [float(step)] * 12, "success": step == 3}) + "\n" for step in range(4)), encoding="utf-8")
    assignment = _assignment(tmp_path)
    recovery = load_controlled_recovery(assignment)
    assert recovery.prefix_actions == ((0.0,) * 12, (1.0,) * 12)
    annotations.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_controlled_recovery(assignment)
    annotations.unlink()
    annotations.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="absolute regular file"):
        load_controlled_recovery(assignment)


def test_production_bootstrap_restores_prefixes_then_perturbs_before_policy_continuation(tmp_path) -> None:
    from lehome.flywheel.recovery_collection import bootstrap_controlled_recovery

    reset = tmp_path / "reset.json"
    annotations = tmp_path / "annotations.jsonl"
    reset.write_text(json.dumps(_snapshot()), encoding="utf-8")
    annotations.write_text("".join(json.dumps({"step": step, "action": [float(step)] * 12, "success": step == 3}) + "\n" for step in range(4)), encoding="utf-8")
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
    assert env.actions == [[0.0] * 12, [1.0] * 12]
    assert env.state["cloth_position"] != _snapshot()["cloth_position"]
    assert provenance["source_episode_id"] == "episode"


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
        def flywheel_capture_state(self): return self.state
        def step(self, action) -> None:
            self.actions.append(action)
            self.state["robot_position"] = [self.drift] * 12

    accepted = Env(0.0049)
    provenance = bootstrap_controlled_recovery(accepted, _assignment(tmp_path / "accepted"))
    check = provenance["replay_fidelity"]
    assert check["verified"] is True and check["tolerance_rad"] == 0.005 and check["max_abs_error_rad"] == pytest.approx(0.0049)
    assert len(check["expected_state_sha256"]) == len(check["observed_state_sha256"]) == 64
    rejected = Env(0.0051)
    with pytest.raises(ValueError, match="replay fidelity"):
        bootstrap_controlled_recovery(rejected, _assignment(tmp_path / "rejected"))
    assert rejected.state["cloth_position"] == _snapshot()["cloth_position"]


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
    assert env.actions == [[0.0] * 12, [1.0] * 12, [2.0] * 12, [3.0] * 12, [0.0] * 12, [1.0] * 12]
    assert provenance["teacher_probe"]["verified"] is True
    assert len(provenance["replay_fidelity_checks"]) == 2
    with pytest.raises(ValueError, match="teacher probe"):
        bootstrap_controlled_recovery(Env(False), _assignment(tmp_path / "teacher-fail", teacher=True))


def test_teacher_probe_rejects_a_source_that_does_not_mark_its_declared_first_success(tmp_path) -> None:
    from lehome.flywheel.recovery_collection import load_controlled_recovery

    assignment = _assignment(tmp_path / "source-success", teacher=True)
    annotations = Path(assignment["source_annotations"])
    rows = [json.loads(line) for line in annotations.read_text(encoding="utf-8").splitlines()]
    rows[3]["success"] = False
    annotations.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assignment["source_annotations_sha256"] = hashlib.sha256(annotations.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="first recorded success"):
        load_controlled_recovery(assignment)


def test_single_materialization_loader_returns_hydrated_rows_and_rejects_identity_tampering(tmp_path) -> None:
    from lehome.flywheel.recovery_collection import load_attempt_matrix

    reset, annotations = tmp_path / "reset.json", tmp_path / "annotations.jsonl"
    reset.write_text("{}", encoding="utf-8"); annotations.write_text("", encoding="utf-8")
    categories = ["pant_long"] * 4 + ["top_long"] + ["top_short"] * 3
    caps = {"pant_long": 4, "top_long": 1, "top_short": 3, "pant_short": 0}
    rows = [{"attempt_id": f"controlled-{index}", "trial_id": f"controlled-{index}", "category": category, "category_acceptance_cap": caps[category], "strategy": "canonical", "recovery_kind": "controlled_success_recovery_v1", "controlled_matrix_sha256": "a" * 64, "perturbation_seed": index, "perturbation_fingerprint": f"{index + 100:064x}", "source_state_perturbation_fingerprint": f"{index + 200:064x}", "source_continuation_state": [0.0] * 12, "source_state_fingerprint": f"{index + 300:064x}", "source_round_id": "round", "source_episode_id": f"episode-{index}", "source_episode_digest": f"{index + 400:064x}", "source_reset_sha256": "a" * 64, "source_annotations_sha256": "b" * 64, "action_prefix_sha256": "c" * 64, "prefix_stop": 1, "source_first_success_step": 2, "source_reset": str(reset), "source_annotations": str(annotations)} for index, category in enumerate(categories)]
    descriptor = tmp_path / "materialization.json"
    payload = {"schema_version": 1, "kind": "controlled_success_recovery_materialization_v1", "matrix_sha256": "a" * 64, "target_accepted": 8, "category_acceptance_caps": caps, "rows": rows}
    descriptor.write_text(json.dumps(payload), encoding="utf-8")
    assert load_attempt_matrix(descriptor) == rows
    rows[0]["controlled_matrix_sha256"] = "b" * 64
    descriptor.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="matrix hash"):
        load_attempt_matrix(descriptor)


def test_materialization_loader_rejects_unreachable_or_noncanonical_controlled_schedule(tmp_path) -> None:
    from lehome.flywheel.recovery_collection import load_attempt_matrix

    reset, annotations = tmp_path / "reset.json", tmp_path / "annotations.jsonl"
    reset.write_text("{}", encoding="utf-8"); annotations.write_text("", encoding="utf-8")
    rows = [
        {
            "attempt_id": f"attempt-{index}", "trial_id": f"trial-{index}",
            "category": "pant_long", "category_acceptance_cap": 4,
            "strategy": "canonical", "recovery_kind": "controlled_success_recovery_v1",
            "controlled_matrix_sha256": "a" * 64, "perturbation_seed": index,
            "perturbation_fingerprint": f"{index + 100:064x}",
            "source_state_perturbation_fingerprint": f"{index + 200:064x}",
                "source_continuation_state": [0.0] * 12, "source_state_fingerprint": f"{index + 300:064x}", "source_round_id": "round",
            "source_episode_id": f"episode-{index}", "source_episode_digest": f"{index + 400:064x}",
            "source_reset_sha256": "a" * 64, "source_annotations_sha256": "b" * 64,
            "action_prefix_sha256": "c" * 64, "prefix_stop": 1,
            "source_first_success_step": 2, "source_reset": str(reset),
            "source_annotations": str(annotations),
        }
        for index in range(8)
    ]
    path = tmp_path / "bad-materialization.json"
    path.write_text(json.dumps({"schema_version": 1, "kind": "controlled_success_recovery_materialization_v1", "matrix_sha256": "a" * 64, "target_accepted": 8, "category_acceptance_caps": {"pant_long": 4, "top_long": 1, "top_short": 3, "pant_short": 0}, "rows": rows}), encoding="utf-8")
    with pytest.raises(ValueError, match="reachable|category"):
        load_attempt_matrix(path)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "schema_version": 1,
                "kind": "controlled_success_recovery_materialization_v1",
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
