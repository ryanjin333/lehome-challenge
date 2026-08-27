from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "summarize_groot_persistent_evaluation.py"


def _module():
    spec = importlib.util.spec_from_file_location("persistent_summary_first100", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_report_preserves_legacy_fields_and_adds_first_hundred_gate_metrics(tmp_path: Path) -> None:
    module = _module()
    report = module._augment_first_hundred_metrics({
        "episodes": 100,
        "official_successes": 5,
        "gate_trials": [
            {
                "assignment_id": f"attempt-{index}",
                "terminal_event": "accepted" if index < 5 else "rejected",
                "identity": {"code_revision": "c" * 40, "asset_revision": "a" * 40, "simulator_version": "5.1.0.0"},
                "provenance": {
                    "policy_repo": "owner/policy", "policy_revision": "e" * 40,
                    "policy_step": 12000, "policy_artifact_sha256": "b" * 64,
                    "image_identity": "sha256:" + "d" * 64,
                    "simulator_device": "cpu", "cloth_device": "cpu",
                    "renderer_device": "cuda:0", "camera_device": "cuda:0", "policy_device": "cuda:0",
                },
                "fidelity": {"missing_cloth": False, "cloth_flight": False, "nonfinite_cloth_state": False, "safety_failure": False},
            }
            for index in range(100)
        ],
        "infrastructure_invalid_executions": 2,
    })

    assert report["episodes"] == 100
    assert report["official_successes"] == 5
    assert report["valid_outcomes"] == 100
    assert report["infrastructure_invalid_executions"] == 2
    assert report["execution_count"] == 102
    assert report["fresh_assignment_ids"] == sorted(f"attempt-{index}" for index in range(100))
    assert len(report["runtime_identities"]) == 1
    assert len(report["runtime_identities"][0]) == 64


POLICY = {
    "policy_repo": "owner/policy", "policy_revision": "e" * 40,
    "policy_step": 12000, "policy_artifact_sha256": "b" * 64,
}


def _catalog() -> dict[str, list[str]]:
    return {
        "top_long": [f"Top_Long_Seen_{index}" for index in range(10)],
        "top_short": [f"Top_Short_Seen_{index}" for index in range(10)],
        "pant_long": [f"Pant_Long_Seen_{index}" for index in range(10)],
        "pant_short": [f"Pant_Short_Seen_{index}" for index in range(10)],
    }


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _simple_campaign(tmp_path: Path, *, missing_receipt: int | None = None, malformed_receipt: int | None = None, retry_then_valid: int | None = None, contradictory_safety: int | None = None, episode_mutator=None, receipt_mutator=None) -> tuple[Path, Path, list[dict[str, object]], dict[str, str]]:
    from lehome.flywheel.simple_curriculum import build_calibration_rows
    from lehome.flywheel.task_ledger import TaskLedger

    rows = build_calibration_rows(_catalog(), seed_base=900)[:100]
    root = tmp_path / "campaign"; root.mkdir()
    matrix = tmp_path / "matrix.json"; matrix.write_bytes(_canonical(rows))
    ledger = TaskLedger(root / "ledger.sqlite3", attempt_matrix=rows, max_attempts=101 if retry_then_valid is not None else 100, target_accepted=100)
    ledger_ids: dict[str, str] = {}
    for index, row in enumerate(rows):
        lease = ledger.lease_next("worker", lease_duration_ns=1_000_000_000)
        assert lease is not None
        if index == retry_then_valid:
            ledger.record_interrupted("worker", lease.attempt.attempt_id, lease.lease_id, "test_retry")
            lease = ledger.lease_next("worker", lease_duration_ns=1_000_000_000)
            assert lease is not None
        ledger_id = lease.attempt.attempt_id
        output = root / "worker" / ledger_id / "generation-1"; raw = output / "raw" / ledger_id; raw.mkdir(parents=True)
        ledger_ids[str(row["attempt_id"])] = ledger_id
        ledger.record_terminal("worker", ledger_id, lease.lease_id, str(output))
        ledger.validate_terminal(ledger_id, "accepted" if index < 5 else "rejected", artifact_id=str(output) if index < 5 else None)
        episode = {
            "episode_id": ledger_id,
            "identity": {
                **POLICY, "episode_id": ledger_id, "code_revision": "c" * 40,
                "asset_revision": "a" * 40, "simulator_version": "5.1.0.0",
                "garment_name": row["garment_name"], "category": row["category"],
                "release_stage": row["release_stage"], "seed": row["seed"],
            },
            "provenance": {"policy_artifact_sha256": POLICY["policy_artifact_sha256"], "simulator_device": "cpu", "policy_device": "cuda:0", "image_identity": "sha256:" + "d" * 64},
            "outcome": "success" if index < 5 else "timeout", "accepted_success": index < 5,
            "safety_failure": index == contradictory_safety,
            "fidelity": {"missing_cloth": False, "cloth_flight": False, "nonfinite_cloth_state": False, "safety_failure": False},
        }
        if episode_mutator is not None:
            episode_mutator(episode, index)
        (raw / "episode.json").write_bytes(_canonical(episode))
        receipt = {
            "schema_version": 1, "attempt_id": ledger_id, "lease_id": lease.lease_id,
            "worker_id": "worker", "session_id": "session", "seed": row["seed"], "garment": row["garment_name"],
            "episode_generation": 1, "output_dir": str(output), "action_horizon": 250,
            "outcome": {"success": index < 5, "metrics": []}, "simulation_device": "cpu", "cloth_device": "cpu",
            "renderer_device": "cuda:0", "camera_device": "cuda:0", "policy_device": "cuda:0",
            "runtime": {"simulation_device": "cpu", "cloth_device": "cpu", "renderer_device": "cuda:0", "camera_device": "cuda:0", "policy_device": "cuda:0"},
        }
        if receipt_mutator is not None:
            receipt_mutator(receipt, index)
        receipt_path = output / "worker-receipt.json"
        if index != missing_receipt:
            receipt_path.write_bytes(b"{broken" if index == malformed_receipt else _canonical(receipt))
    ledger.close()
    return root, matrix, rows, ledger_ids


def test_simple_summary_uses_external_matrix_assignment_ids_and_passes_gate_directly(tmp_path: Path) -> None:
    summary = _module()
    root, matrix, rows, ledger_ids = _simple_campaign(tmp_path)

    report = summary.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        candidate_key="original_baseline", **POLICY,
    )

    assert set(report["trials"][0]) == {
        "schedule_index", "trial_id", "attempt_id", "category", "garment", "seed", "official_success",
        "terminal_event", "episode_sha256", "worker_receipt_sha256",
    }
    assert report["gate_trials"][0]["assignment_id"] == rows[0]["attempt_id"]
    assert report["gate_trials"][0]["assignment_id"] != ledger_ids[str(rows[0]["attempt_id"])]
    assert type(report["gate_trials"][0]["official_success"]) is bool
    assert report["gate_trials"][0]["identity"]["policy_artifact_sha256"] == POLICY["policy_artifact_sha256"]

    gate_spec = importlib.util.spec_from_file_location("gate_for_producer_test", ROOT / "scripts" / "check_simple_curriculum_gate.py")
    assert gate_spec and gate_spec.loader
    gate = importlib.util.module_from_spec(gate_spec); sys.modules[gate_spec.name] = gate; gate_spec.loader.exec_module(gate)
    receipt = gate.build_gate_receipt(
        report, report_bytes=_canonical(report), matrix=rows, matrix_bytes=matrix.read_bytes(),
        trusted_policy=POLICY, policy_bytes=_canonical(POLICY), catalog=_catalog(), catalog_bytes=_canonical(_catalog()),
    )
    assert receipt["decision"] == "continue"


@pytest.mark.parametrize("kind", ["missing", "malformed"])
def test_simple_summary_counts_incomplete_receipts_as_invalid_executions(tmp_path: Path, kind: str) -> None:
    summary = _module()
    root, matrix, _rows, _ledger_ids = _simple_campaign(
        tmp_path, missing_receipt=3 if kind == "missing" else None, malformed_receipt=3 if kind == "malformed" else None,
    )

    report = summary.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        candidate_key="original_baseline", **POLICY,
    )

    assert report["valid_outcomes"] == 99
    assert report["infrastructure_invalid_executions"] == 1
    assert report["execution_count"] == 100


def test_simple_summary_counts_retry_then_valid_as_one_invalid_execution(tmp_path: Path) -> None:
    summary = _module()
    root, matrix, _rows, _ledger_ids = _simple_campaign(tmp_path, retry_then_valid=3)

    report = summary.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        candidate_key="original_baseline", **POLICY,
    )

    assert report["valid_outcomes"] == 100
    assert report["infrastructure_invalid_executions"] == 1
    assert report["execution_count"] == 101


def test_simple_summary_does_not_allow_aggregate_safety_to_disagree_with_gate_fidelity(tmp_path: Path) -> None:
    summary = _module()
    root, matrix, _rows, _ledger_ids = _simple_campaign(tmp_path, contradictory_safety=3)

    report = summary.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        candidate_key="original_baseline", **POLICY,
    )

    assert report["valid_outcomes"] == 99
    assert report["safety_failure"] is False


@pytest.mark.parametrize("evidence_name,identity_key", [("worker-receipt.json", "attempt_id"), ("episode.json", "episode_id")])
@pytest.mark.parametrize("kind", ["malformed", "unsafe", "duplicate", "unbound"])
def test_simple_summary_counts_each_stray_evidence_file_once(tmp_path: Path, evidence_name: str, identity_key: str, kind: str) -> None:
    summary = _module()
    root, matrix, _rows, ledger_ids = _simple_campaign(tmp_path)
    stray = root / "stray"; stray.mkdir()
    if kind == "malformed":
        (stray / evidence_name).write_bytes(b"{broken")
    elif kind == "unsafe":
        (stray / evidence_name).symlink_to(next(root.rglob(evidence_name)))
    elif kind == "duplicate":
        original = next(root.rglob(evidence_name)); duplicate = json.loads(original.read_text())
        assert duplicate[identity_key] in ledger_ids.values()
        (stray / evidence_name).write_bytes(_canonical(duplicate))
    else:
        evidence = json.loads(next(root.rglob(evidence_name)).read_text()); evidence[identity_key] = "unbound-ledger-id"
        (stray / evidence_name).write_bytes(_canonical(evidence))

    report = summary.build_report(campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(), candidate_key="original_baseline", **POLICY)

    assert report["valid_outcomes"] == 100
    assert report["infrastructure_invalid_executions"] == 1
    assert report["execution_count"] == 101


@pytest.mark.parametrize("field", ["lease_id", "worker_id", "session_id", "episode_generation", "output_dir", "runtime"])
def test_simple_summary_rejects_receipt_not_bound_to_ledger_terminal_artifact_or_runtime(tmp_path: Path, field: str) -> None:
    summary = _module()

    def mutate(receipt, index):
        if index == 3:
            receipt[field] = {**receipt["runtime"], "camera_device": "cuda:1"} if field == "runtime" else "mismatch"
    root, matrix, _rows, _ledger_ids = _simple_campaign(tmp_path, receipt_mutator=mutate)

    report = summary.build_report(campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(), candidate_key="original_baseline", **POLICY)

    assert report["valid_outcomes"] == 99
    assert report["infrastructure_invalid_executions"] == 1


@pytest.mark.parametrize("field", ["code_revision", "asset_revision", "simulator_version", "image_identity"])
@pytest.mark.parametrize("value", ["", " ", True, 1, "not-a-digest"])
def test_simple_summary_rejects_noncanonical_runtime_identity_values(tmp_path: Path, field: str, value: object) -> None:
    summary = _module()

    def mutate(episode, index):
        if index == 3:
            target = episode["provenance"] if field == "image_identity" else episode["identity"]
            target[field] = value
    root, matrix, _rows, _ledger_ids = _simple_campaign(tmp_path, episode_mutator=mutate)

    report = summary.build_report(campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(), candidate_key="original_baseline", **POLICY)

    assert report["valid_outcomes"] == 99
    assert report["infrastructure_invalid_executions"] == 1
