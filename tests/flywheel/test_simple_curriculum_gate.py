from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_simple_curriculum_gate.py"


def _module():
    spec = importlib.util.spec_from_file_location("simple_curriculum_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _report(*, valid_outcomes: int = 100, invalid: int = 0, successes: int = 5,
            runtime_identities: tuple[str, ...] = ("a" * 64,), fidelity: dict[str, bool] | None = None) -> dict[str, object]:
    return {
        "valid_outcomes": valid_outcomes,
        "infrastructure_invalid_executions": invalid,
        "execution_count": valid_outcomes + invalid,
        "official_successes": successes,
        "runtime_identities": list(runtime_identities),
        "gate_trials": [{"fidelity": fidelity or {"missing_cloth": False, "cloth_flight": False, "nonfinite_cloth_state": False, "safety_failure": False}}],
        "safety_failure": False,
    }


def test_gate_continues_at_exactly_one_hundred_outcomes_and_five_successes() -> None:
    gate = _module()

    decision = gate.evaluate_gate(_report())

    assert decision.decision == "continue"
    assert decision.reason == "passed"
    assert decision.as_dict() == {"decision": "continue", "reason": "passed"}


def test_decision_model_refuses_unapproved_decision_reason_pairs() -> None:
    gate = _module()

    with pytest.raises(ValueError, match="invalid"):
        gate.GateDecision("continue", "invalid_ratio")


@pytest.mark.parametrize("field", ["missing_cloth", "cloth_flight", "nonfinite_cloth_state", "safety_failure"])
def test_gate_stops_for_every_episode_fidelity_failure(field: str) -> None:
    gate = _module()
    fidelity = {"missing_cloth": False, "cloth_flight": False, "nonfinite_cloth_state": False, "safety_failure": False}
    fidelity[field] = True

    decision = gate.evaluate_gate(_report(fidelity=fidelity))

    assert decision.as_dict() == {"decision": "fidelity_stop", "reason": "episode_fidelity"}


def test_gate_stops_for_authenticated_aggregate_safety_failure() -> None:
    gate = _module()
    report = _report(); report["safety_failure"] = True

    assert gate.evaluate_gate(report).as_dict() == {"decision": "fidelity_stop", "reason": "episode_fidelity"}


def test_gate_stops_fidelity_first_for_typed_preframe_failure() -> None:
    gate = _module()
    report = _report(valid_outcomes=0)
    report["gate_fidelity_failures"] = [{
        "fidelity_code": "missing_cloth",
        "fidelity": {
            "missing_cloth": True, "cloth_flight": False,
            "nonfinite_cloth_state": False, "safety_failure": False,
            "monitor_active": True, "monitor_observed": True,
        },
    }]

    assert gate.evaluate_gate(report).as_dict() == {
        "decision": "fidelity_stop", "reason": "episode_fidelity",
    }


@pytest.mark.parametrize(
    ("invalid", "decision", "reason"),
    [(0, "continue", "passed"), (2, "continue", "passed"), (3, "infrastructure_stop", "invalid_ratio")],
)
def test_gate_invalid_ratio_boundary_is_strictly_greater_than_two_percent(invalid: int, decision: str, reason: str) -> None:
    gate = _module()

    result = gate.evaluate_gate(_report(invalid=invalid))

    assert result.as_dict() == {"decision": decision, "reason": reason}


@pytest.mark.parametrize(("successes", "decision", "reason"), [(0, "insufficient_source_stop", "official_success_floor"), (4, "insufficient_source_stop", "official_success_floor"), (5, "continue", "passed")])
def test_gate_requires_five_official_successes(successes: int, decision: str, reason: str) -> None:
    gate = _module()

    assert gate.evaluate_gate(_report(successes=successes)).as_dict() == {"decision": decision, "reason": reason}


def test_gate_stops_for_mixed_runtime_identities() -> None:
    gate = _module()

    assert gate.evaluate_gate(_report(runtime_identities=("a" * 64, "b" * 64))).as_dict() == {
        "decision": "fidelity_stop", "reason": "mixed_runtime_identity",
    }


@pytest.mark.parametrize("valid_outcomes", [99, 101])
def test_gate_requires_exactly_one_hundred_valid_outcomes(valid_outcomes: int) -> None:
    gate = _module()

    assert gate.evaluate_gate(_report(valid_outcomes=valid_outcomes)).as_dict() == {
        "decision": "infrastructure_stop", "reason": "valid_outcome_count",
    }


POLICY = {"policy_repo": "owner/policy", "policy_revision": "e" * 40, "policy_step": 12000, "policy_artifact_sha256": "b" * 64}


def _catalog() -> dict[str, list[str]]:
    return {
        "top_long": [f"Top_Long_Seen_{index}" for index in range(10)],
        "top_short": [f"Top_Short_Seen_{index}" for index in range(10)],
        "pant_long": [f"Pant_Long_Seen_{index}" for index in range(10)],
        "pant_short": [f"Pant_Short_Seen_{index}" for index in range(10)],
    }


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _report_digest(value: object) -> str:
    body = dict(value)
    body.pop("report_sha256", None)
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _matrix() -> list[dict[str, object]]:
    from lehome.flywheel.simple_curriculum import build_calibration_rows

    return build_calibration_rows(_catalog(), seed_base=900)[:100]


def _authenticated_report(rows: list[dict[str, object]], *, invalid: int = 0, successes: int = 5) -> dict[str, object]:
    provenance = {
        **POLICY, "image_identity": "sha256:" + "d" * 64, "simulator_device": "cpu", "cloth_device": "cpu",
        "renderer_device": "cuda:0", "camera_device": "cuda:0", "policy_device": "cuda:0",
    }
    runtime = hashlib.sha256(json.dumps({
        **provenance, "code_revision": "c" * 40, "asset_revision": "a" * 40, "simulator_version": "5.1.0.0",
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    report: dict[str, object] = {
        "schema_version": 1, "kind": "lehome_simple_curriculum_first100_report_v1",
        "campaign_kind": "simple_curriculum_source_v1", "logical_stage": "calibration_head",
        "matrix_sha256": _digest(rows), "identity": POLICY,
        "valid_outcomes": 100, "infrastructure_invalid_executions": invalid,
        "execution_count": 100 + invalid, "official_successes": successes,
        "safety_failure": False, "gate_fidelity_failures": [],
        "runtime_identities": [runtime], "fresh_assignment_ids": sorted(str(row["attempt_id"]) for row in rows),
        "trials": [],
        "gate_trials": [
            {
                "assignment_id": row["attempt_id"], "trial_id": row["trial_id"],
                "terminal_event": "accepted" if index < successes else "rejected",
                "official_success": index < successes,
                "identity": {**POLICY, "code_revision": "c" * 40, "asset_revision": "a" * 40, "simulator_version": "5.1.0.0"},
                "provenance": provenance,
                "fidelity": {"missing_cloth": False, "cloth_flight": False, "nonfinite_cloth_state": False, "safety_failure": False},
            }
            for index, row in enumerate(rows)
        ],
    }
    report["report_sha256"] = _report_digest(report)
    return report


def _write_inputs(tmp_path: Path, report: dict[str, object], rows: list[dict[str, object]]) -> tuple[Path, Path, Path, Path, Path]:
    report_path, matrix_path, policy_path, catalog_path, output = (tmp_path / name for name in ("report.json", "matrix.json", "policy.json", "catalog.json", "gate.json"))
    report_path.write_bytes(_canonical(report)); matrix_path.write_bytes(_canonical(rows))
    policy_path.write_bytes(_canonical(POLICY)); catalog_path.write_bytes(_canonical(_catalog()))
    return report_path, matrix_path, policy_path, catalog_path, output


def _run_gate(report: Path, matrix: Path, policy: Path, catalog: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--report", str(report), "--matrix", str(matrix), "--trusted-policy-identity", str(policy), "--approved-garment-catalog", str(catalog), "--output", str(output)],
        cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "source/lehome")}, text=True, capture_output=True, check=False,
    )


def test_cli_authenticates_the_exact_calibration_head_and_writes_deterministic_immutable_receipt(tmp_path: Path) -> None:
    rows = _matrix(); report = _authenticated_report(rows)
    first = tmp_path / "one"; first.mkdir()
    report_path, matrix_path, policy_path, catalog_path, output = _write_inputs(first, report, rows)
    result = _run_gate(report_path, matrix_path, policy_path, catalog_path, output)

    assert result.returncode == 0, result.stderr
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["kind"] == "lehome_simple_curriculum_first100_gate_receipt_v1"
    assert receipt["decision"] == "continue" and receipt["reason"] == "passed"
    assert receipt["report_sha256"] == hashlib.sha256(report_path.read_bytes()).hexdigest()
    assert receipt["matrix_sha256"] == hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    duplicate = tmp_path / "two"; duplicate.mkdir()
    second = _write_inputs(duplicate, report, rows)
    assert _run_gate(*second).returncode == 0
    assert output.read_bytes() == second[-1].read_bytes()


def test_cli_authenticates_typed_preframe_fidelity_failure_before_outcome_count(tmp_path: Path) -> None:
    rows = _matrix(); report = _authenticated_report(rows)
    report["valid_outcomes"] = 99
    report["execution_count"] = 100
    report["infrastructure_invalid_executions"] = 1
    report["fresh_assignment_ids"] = report["fresh_assignment_ids"][:-1]
    report["gate_trials"] = report["gate_trials"][:-1]
    report["runtime_identities"] = [report["runtime_identities"][0]]
    report["gate_fidelity_failures"] = [{
        "assignment_id": rows[-1]["attempt_id"], "ledger_id": "a" * 64,
        "lease_id": "b" * 64, "worker_id": "worker-0", "session_id": "session-0", "generation": 1,
        "fidelity_code": "missing_cloth",
        "fidelity": {"missing_cloth": True, "cloth_flight": False, "nonfinite_cloth_state": False, "safety_failure": False, "monitor_active": True, "monitor_observed": True},
        "runtime": {"simulation_device": "cpu", "cloth_device": "cpu", "renderer_device": "cuda:0", "camera_device": "cuda:0", "policy_device": "cuda:0"},
    }]
    report["report_sha256"] = _report_digest(report)
    paths = _write_inputs(tmp_path, report, rows)

    result = _run_gate(*paths)

    assert result.returncode == 0, result.stderr
    assert json.loads(paths[-1].read_text(encoding="utf-8"))["decision"] == "fidelity_stop"


def test_cli_rejects_untrusted_typed_preframe_fidelity_runtime(tmp_path: Path) -> None:
    rows = _matrix(); report = _authenticated_report(rows)
    report["gate_fidelity_failures"] = [{
        "assignment_id": rows[-1]["attempt_id"], "ledger_id": "a" * 64,
        "lease_id": "b" * 64, "worker_id": "worker-0", "session_id": "session-0", "generation": 1,
        "fidelity_code": "missing_cloth",
        "fidelity": {"missing_cloth": True, "cloth_flight": False, "nonfinite_cloth_state": False, "safety_failure": False, "monitor_active": True, "monitor_observed": True},
        "runtime": {"simulation_device": "cpu", "cloth_device": "cpu", "renderer_device": "cuda:bogus", "camera_device": "cuda:0", "policy_device": "cuda:0"},
    }]
    report["report_sha256"] = _report_digest(report)
    paths = _write_inputs(tmp_path, report, rows)

    assert _run_gate(*paths).returncode != 0

@pytest.mark.parametrize("mutator", [
    lambda report, rows: report.update(fresh_assignment_ids=report["fresh_assignment_ids"][:-1]),
    lambda report, rows: report.update(fresh_assignment_ids=[*report["fresh_assignment_ids"], "extra"]),
    lambda report, rows: report.update(fresh_assignment_ids=[*report["fresh_assignment_ids"][:-1], report["fresh_assignment_ids"][0]]),
    lambda report, rows: report["gate_trials"].__setitem__(0, {**report["gate_trials"][0], "assignment_id": "wrong"}),
])
def test_cli_rejects_missing_extra_duplicate_or_mismatched_assignment_identities(tmp_path: Path, mutator) -> None:
    rows = _matrix(); report = _authenticated_report(rows); mutator(report, rows); report["report_sha256"] = _report_digest(report)
    paths = _write_inputs(tmp_path, report, rows)

    result = _run_gate(*paths)

    assert result.returncode != 0
    assert not paths[-1].exists()


@pytest.mark.parametrize("mutation", ["policy", "catalog", "provenance", "campaign", "matrix", "report_sha"])
def test_cli_fails_closed_on_untrusted_or_wrong_identity_inputs(tmp_path: Path, mutation: str) -> None:
    rows = _matrix(); report = _authenticated_report(rows); paths = _write_inputs(tmp_path, report, rows)
    if mutation == "policy":
        paths[2].write_bytes(_canonical({**POLICY, "policy_artifact_sha256": "c" * 64}))
    elif mutation == "catalog":
        catalog = _catalog(); catalog["top_long"] = list(reversed(catalog["top_long"])); paths[3].write_bytes(_canonical(catalog))
    elif mutation == "provenance":
        report["gate_trials"][0]["provenance"]["camera_device"] = "cpu"; report["report_sha256"] = _report_digest(report); paths[0].write_bytes(_canonical(report))
    elif mutation == "campaign":
        report["campaign_kind"] = "wrong"; report["report_sha256"] = _report_digest(report); paths[0].write_bytes(_canonical(report))
    elif mutation == "matrix":
        report["matrix_sha256"] = "0" * 64; report["report_sha256"] = _report_digest(report); paths[0].write_bytes(_canonical(report))
    else:
        report["official_successes"] = 99; paths[0].write_bytes(_canonical(report))

    assert _run_gate(*paths).returncode != 0
    assert not paths[-1].exists()


@pytest.mark.parametrize("mutation", ["logical_stage", "cuda_bogus"])
def test_cli_rejects_wrong_stage_and_noncanonical_cuda_provenance(tmp_path: Path, mutation: str) -> None:
    rows = _matrix(); report = _authenticated_report(rows); paths = _write_inputs(tmp_path, report, rows)
    if mutation == "logical_stage":
        report["logical_stage"] = "curriculum"
    else:
        report["gate_trials"][0]["provenance"]["policy_device"] = "cuda:bogus"
    report["report_sha256"] = _report_digest(report); paths[0].write_bytes(_canonical(report))

    assert _run_gate(*paths).returncode != 0


def test_cli_rejects_aggregate_safety_that_disagrees_with_gate_trials(tmp_path: Path) -> None:
    rows = _matrix(); report = _authenticated_report(rows); paths = _write_inputs(tmp_path, report, rows)
    report["safety_failure"] = True; report["report_sha256"] = _report_digest(report); paths[0].write_bytes(_canonical(report))

    assert _run_gate(*paths).returncode != 0


@pytest.mark.parametrize("field", ["code_revision", "asset_revision", "simulator_version", "image_identity"])
@pytest.mark.parametrize("value", ["", " ", True, 1, "not-a-digest"])
def test_gate_rejects_noncanonical_runtime_identity_fields(tmp_path: Path, field: str, value: object) -> None:
    rows = _matrix(); report = _authenticated_report(rows); paths = _write_inputs(tmp_path, report, rows)
    target = report["gate_trials"][0]["identity"] if field in {"code_revision", "asset_revision", "simulator_version"} else report["gate_trials"][0]["provenance"]
    target[field] = value
    report["report_sha256"] = _report_digest(report); paths[0].write_bytes(_canonical(report))

    assert _run_gate(*paths).returncode != 0


@pytest.mark.parametrize("raw", ['{"x":1,"x":2}', '{"x":NaN}'])
def test_cli_rejects_duplicate_key_and_nonfinite_json(tmp_path: Path, raw: str) -> None:
    rows = _matrix(); report = _authenticated_report(rows); paths = _write_inputs(tmp_path, report, rows)
    paths[0].write_text(raw, encoding="utf-8")

    assert _run_gate(*paths).returncode != 0
    assert not paths[-1].exists()


def test_cli_refuses_existing_symlink_alias_and_race_outputs(tmp_path: Path) -> None:
    rows = _matrix(); report = _authenticated_report(rows); paths = _write_inputs(tmp_path, report, rows)
    paths[-1].write_text("already", encoding="utf-8")
    assert _run_gate(*paths).returncode != 0
    paths[-1].unlink(); target = tmp_path / "target.json"; target.write_text("target", encoding="utf-8"); paths[-1].symlink_to(target)
    assert _run_gate(*paths).returncode != 0
    paths[-1].unlink(); alias = tmp_path / "alias"; alias.symlink_to(tmp_path, target_is_directory=True)
    assert _run_gate(paths[0], paths[1], paths[2], paths[3], alias / "gate.json").returncode != 0

    gate = _module()
    def competitor() -> None:
        paths[-1].write_bytes(b"competitor\n")
    with pytest.raises(FileExistsError):
        gate._atomic_write(paths[-1], b"ours\n", before_publish=competitor)
    assert paths[-1].read_bytes() == b"competitor\n"
