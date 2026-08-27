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
                "fidelity": {"missing_cloth": False, "cloth_flight": False, "nonfinite_cloth_state": False, "safety_failure": False, "monitor_active": True, "monitor_observed": True},
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


def _simple_campaign(tmp_path: Path, *, missing_receipt: int | None = None, malformed_receipt: int | None = None, retry_then_valid: int | None = None, contradictory_safety: int | None = None, all_success: bool = False, evaluation_terminal: bool = True, episode_mutator=None, receipt_mutator=None, session_for_index=None) -> tuple[Path, Path, list[dict[str, object]], dict[str, str]]:
    from lehome.flywheel.artifact_queue import ArtifactFinalizationQueue
    from lehome.flywheel.simple_curriculum import build_calibration_rows
    from lehome.flywheel.task_ledger import TaskLedger

    rows = build_calibration_rows(_catalog(), seed_base=900)[:100]
    root = tmp_path / "campaign"; root.mkdir()
    matrix = tmp_path / "matrix.json"; matrix.write_bytes(_canonical(rows))
    ledger = TaskLedger(root / "ledger.sqlite3", attempt_matrix=rows, max_attempts=101 if retry_then_valid is not None else 100, target_accepted=100)
    finalizer = ArtifactFinalizationQueue(
        run_root=root, ledger=ledger, max_pending_items=1, max_pending_bytes=1 << 30,
        evaluation_only=evaluation_terminal,
    )
    ledger_ids: dict[str, str] = {}
    for index, row in enumerate(rows):
        lease = ledger.lease_next("worker", lease_duration_ns=1_000_000_000)
        assert lease is not None
        if index == retry_then_valid:
            ledger.record_interrupted("worker", lease.attempt.attempt_id, lease.lease_id, "test_retry")
            lease = ledger.lease_next("worker", lease_duration_ns=1_000_000_000)
            assert lease is not None
        ledger_id = lease.attempt.attempt_id
        session_id = session_for_index(index) if session_for_index is not None else "session"
        output = root / "worker" / session_id / ledger_id / lease.lease_id / f"generation-{index + 1}"
        raw = output / "raw" / ledger_id; raw.mkdir(parents=True)
        videos = output / "videos"; videos.mkdir(); (videos / "top.mp4").write_bytes(b"video")
        ledger_ids[str(row["attempt_id"])] = ledger_id
        episode = {
            "episode_id": ledger_id,
            "identity": {
                **POLICY, "episode_id": ledger_id, "code_revision": "c" * 40,
                "asset_revision": "a" * 40, "simulator_version": "5.1.0.0",
                "garment_name": row["garment_name"], "category": row["category"],
                "release_stage": row["release_stage"], "seed": row["seed"],
            },
            "provenance": {"policy_artifact_sha256": POLICY["policy_artifact_sha256"], "simulator_device": "cpu", "policy_device": "cuda:0", "image_identity": "sha256:" + "d" * 64},
            "outcome": "success" if all_success or index < 5 else "timeout", "accepted_success": all_success or index < 5,
            "safety_failure": index == contradictory_safety,
            "fidelity": {"missing_cloth": False, "cloth_flight": False, "nonfinite_cloth_state": False, "safety_failure": False, "monitor_active": True, "monitor_observed": True},
        }
        if episode_mutator is not None:
            episode_mutator(episode, index)
        (raw / "episode.json").write_bytes(_canonical(episode))
        receipt = {
            "schema_version": 1, "attempt_id": ledger_id, "lease_id": lease.lease_id,
            "worker_id": "worker", "session_id": session_id, "seed": row["seed"], "garment": row["garment_name"],
            "episode_generation": index + 1, "output_dir": str(output), "action_horizon": 250,
            "outcome": {"success": all_success or index < 5, "metrics": []}, "simulation_device": "cpu", "cloth_device": "cpu",
            "renderer_device": "cuda:0", "camera_device": "cuda:0", "policy_device": "cuda:0",
            "runtime": {"simulation_device": "cpu", "cloth_device": "cpu", "renderer_device": "cuda:0", "camera_device": "cuda:0", "policy_device": "cuda:0"},
        }
        if receipt_mutator is not None:
            receipt_mutator(receipt, index)
        receipt_path = output / "worker-receipt.json"
        if index != missing_receipt:
            receipt_path.write_bytes(b"{broken" if index == malformed_receipt else _canonical(receipt))
        ledger.record_terminal("worker", ledger_id, lease.lease_id, str(output))
        finalizer.enqueue("worker", ledger_id, lease.lease_id, output)
        result = finalizer.finalize_next()
        assert result is not None
        if index == 0:
            assert result.outcome == "accepted"
        if index == 5 and not all_success:
            assert result.outcome == "rejected"
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


def test_simple_summary_authenticates_accepted_finalizer_destination(tmp_path: Path) -> None:
    summary = _module()
    root, matrix, _rows, _ledger_ids = _simple_campaign(
        tmp_path, all_success=True, evaluation_terminal=False,
    )

    report = summary.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        candidate_key="original_baseline", **POLICY,
    )

    assert report["valid_outcomes"] == 100
    assert report["infrastructure_invalid_executions"] == 0


def test_simple_summary_admits_worker_restart_with_a_fresh_session(tmp_path: Path) -> None:
    summary = _module()
    root, matrix, _rows, _ledger_ids = _simple_campaign(
        tmp_path, session_for_index=lambda index: "session-after-restart" if index >= 50 else "session-before-restart",
    )

    report = summary.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        candidate_key="original_baseline", **POLICY,
    )

    assert report["valid_outcomes"] == 100
    assert report["infrastructure_invalid_executions"] == 0


@pytest.mark.parametrize("fidelity", [
    {"missing_cloth": False, "cloth_flight": False, "nonfinite_cloth_state": False, "safety_failure": False, "monitor_active": False, "monitor_observed": True},
    {"missing_cloth": False, "cloth_flight": False, "nonfinite_cloth_state": False, "safety_failure": False, "monitor_active": True, "monitor_observed": False},
    {"missing_cloth": False, "cloth_flight": False, "nonfinite_cloth_state": False, "safety_failure": False, "monitor_active": True},
])
def test_simple_summary_rejects_incomplete_or_unobserved_terminal_fidelity(tmp_path: Path, fidelity: dict[str, bool]) -> None:
    summary = _module()

    def mutate(episode, index):
        if index == 3:
            episode["fidelity"] = fidelity
    root, matrix, _rows, _ledger_ids = _simple_campaign(tmp_path, episode_mutator=mutate)

    report = summary.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        candidate_key="original_baseline", **POLICY,
    )

    assert report["valid_outcomes"] == 99
    assert report["infrastructure_invalid_executions"] == 1


def test_simple_summary_rejects_forged_finalized_artifact_destination(tmp_path: Path) -> None:
    summary = _module()
    root, matrix, rows, ledger_ids = _simple_campaign(tmp_path)
    ledger_id = ledger_ids[str(rows[3]["attempt_id"])]
    destination = root / "evaluation-terminal" / ledger_id
    destination.rename(root / "evaluation-terminal" / f"forged-{ledger_id}")

    report = summary.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        candidate_key="original_baseline", **POLICY,
    )

    assert report["valid_outcomes"] == 99
    assert report["infrastructure_invalid_executions"] == 1


def test_simple_summary_rejects_receipt_raw_output_path_not_bound_to_worker_lease(tmp_path: Path) -> None:
    summary = _module()
    root, matrix, rows, ledger_ids = _simple_campaign(tmp_path)
    ledger_id = ledger_ids[str(rows[3]["attempt_id"])]
    receipt_path = root / "evaluation-terminal" / ledger_id / "worker-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["output_dir"] = str(root / "worker" / "session" / ledger_id / "forged-lease" / "generation-4")
    receipt_path.write_bytes(_canonical(receipt))

    report = summary.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        candidate_key="original_baseline", **POLICY,
    )

    assert report["valid_outcomes"] == 99
    assert report["infrastructure_invalid_executions"] == 1


def test_simple_summary_rejects_stale_receipt_copied_into_another_finalized_artifact(tmp_path: Path) -> None:
    summary = _module()
    root, matrix, rows, ledger_ids = _simple_campaign(tmp_path)
    target = ledger_ids[str(rows[3]["attempt_id"])]
    source = ledger_ids[str(rows[4]["attempt_id"])]
    target_receipt = root / "evaluation-terminal" / target / "worker-receipt.json"
    target_receipt.write_bytes((root / "evaluation-terminal" / source / "worker-receipt.json").read_bytes())

    report = summary.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        candidate_key="original_baseline", **POLICY,
    )

    assert report["valid_outcomes"] < 100
    assert report["infrastructure_invalid_executions"] >= 1


def test_simple_summary_rejects_semantically_valid_finalized_episode_with_stale_manifest(tmp_path: Path) -> None:
    summary = _module()
    root, matrix, rows, ledger_ids = _simple_campaign(tmp_path)
    ledger_id = ledger_ids[str(rows[3]["attempt_id"])]
    episode_path = root / "evaluation-terminal" / ledger_id / "raw" / ledger_id / "episode.json"
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    episode["audit_note"] = "tampered-after-finalization"
    episode_path.write_bytes(_canonical(episode))

    report = summary.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        candidate_key="original_baseline", **POLICY,
    )

    assert report["valid_outcomes"] == 99
    assert report["infrastructure_invalid_executions"] == 1


@pytest.mark.parametrize("case", [
    "missing", "malformed", "duplicate", "nonfinite", "noncanonical_hash", "traversal",
    "missing_entry", "extra_file", "symlink",
])
def test_simple_summary_rejects_invalid_finalized_checksum_manifest(tmp_path: Path, case: str) -> None:
    summary = _module()
    root, matrix, rows, ledger_ids = _simple_campaign(tmp_path)
    ledger_id = ledger_ids[str(rows[3]["attempt_id"])]
    destination = root / "evaluation-terminal" / ledger_id
    manifest_path = destination / "SHA256SUMS.json"
    if case == "missing":
        manifest_path.unlink()
    elif case == "malformed":
        manifest_path.write_text("{broken", encoding="utf-8")
    elif case == "duplicate":
        manifest_path.write_text('{"worker-receipt.json":{},"worker-receipt.json":{}}', encoding="utf-8")
    elif case == "nonfinite":
        manifest_path.write_text('{"worker-receipt.json":{"sha256":NaN,"size":1}}', encoding="utf-8")
    elif case == "noncanonical_hash":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["worker-receipt.json"]["sha256"] = "A" * 64
        manifest_path.write_bytes(_canonical(manifest))
    elif case == "traversal":
        manifest_path.write_text('{"../worker-receipt.json":{"sha256":"' + "a" * 64 + '","size":1}}', encoding="utf-8")
    elif case == "missing_entry":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.pop("worker-receipt.json")
        manifest_path.write_bytes(_canonical(manifest))
    elif case == "extra_file":
        (destination / "extra-evidence.bin").write_bytes(b"unexpected")
    else:
        (destination / "unsafe-evidence").symlink_to(destination / "worker-receipt.json")

    report = summary.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        candidate_key="original_baseline", **POLICY,
    )

    assert report["valid_outcomes"] == 99
    assert report["infrastructure_invalid_executions"] == 1


def test_simple_summary_rejects_symlinked_finalized_artifact_ancestor(tmp_path: Path) -> None:
    summary = _module()
    root, matrix, rows, ledger_ids = _simple_campaign(tmp_path)
    ledger_id = ledger_ids[str(rows[3]["attempt_id"])]
    destination = root / "evaluation-terminal" / ledger_id
    outside = tmp_path / "outside" / ledger_id; outside.parent.mkdir()
    destination.rename(outside)
    destination.symlink_to(outside, target_is_directory=True)

    report = summary.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        candidate_key="original_baseline", **POLICY,
    )

    assert report["valid_outcomes"] == 99
    assert report["infrastructure_invalid_executions"] == 1


@pytest.mark.parametrize("fault", ["duplicate", "nonfinite"])
def test_simple_summary_rejects_non_strict_finalized_episode_json(tmp_path: Path, fault: str) -> None:
    summary = _module()
    root, matrix, rows, ledger_ids = _simple_campaign(tmp_path)
    ledger_id = ledger_ids[str(rows[3]["attempt_id"])]
    path = root / "evaluation-terminal" / ledger_id / "raw" / ledger_id / "episode.json"
    episode = json.loads(path.read_text())
    if fault == "duplicate":
        episode.pop("accepted_success")
        path.write_text(json.dumps(episode, sort_keys=True)[:-1] + ',"accepted_success":false,"accepted_success":true}')
    else:
        path.write_text(json.dumps(episode, sort_keys=True)[:-1] + ',"probe":NaN}')

    report = summary.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        candidate_key="original_baseline", **POLICY,
    )

    assert report["valid_outcomes"] == 99
    assert report["infrastructure_invalid_executions"] == 1
    assert report["execution_count"] == 100


def test_simple_summary_deduplicates_malformed_finalized_episode_execution(tmp_path: Path) -> None:
    summary = _module()
    root, matrix, rows, ledger_ids = _simple_campaign(tmp_path)
    ledger_id = ledger_ids[str(rows[3]["attempt_id"])]
    path = root / "evaluation-terminal" / ledger_id / "raw" / ledger_id / "episode.json"
    path.write_bytes(b"{broken")

    report = summary.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        candidate_key="original_baseline", **POLICY,
    )

    assert report["valid_outcomes"] == 99
    assert report["infrastructure_invalid_executions"] == 1
    assert report["execution_count"] == 100


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


def test_simple_summary_counts_retry_and_later_invalid_terminal_as_distinct_executions(tmp_path: Path) -> None:
    summary = _module()

    def mutate(receipt, index):
        if index == 3:
            receipt["runtime"] = {**receipt["runtime"], "camera_device": "cuda:1"}
    root, matrix, _rows, _ledger_ids = _simple_campaign(
        tmp_path, retry_then_valid=3, receipt_mutator=mutate,
    )

    report = summary.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        candidate_key="original_baseline", **POLICY,
    )

    assert report["valid_outcomes"] == 99
    assert report["infrastructure_invalid_executions"] == 2
    assert report["execution_count"] == 101


def test_simple_summary_counts_stray_evidence_with_retried_ledger_id_separately(tmp_path: Path) -> None:
    summary = _module()
    root, matrix, rows, ledger_ids = _simple_campaign(tmp_path, retry_then_valid=3)
    ledger_id = ledger_ids[str(rows[3]["attempt_id"])]
    stray = root / f"stray-{ledger_id}"; stray.mkdir()
    (stray / "episode.json").write_bytes(b"{broken")

    report = summary.build_report(
        campaign_root=root, matrix_path=matrix, matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
        candidate_key="original_baseline", **POLICY,
    )

    assert report["valid_outcomes"] == 100
    assert report["infrastructure_invalid_executions"] == 2
    assert report["execution_count"] == 102


def test_simple_summary_counts_untraversed_symlink_directory_as_unsafe_evidence(tmp_path: Path) -> None:
    summary = _module()
    root, matrix, _rows, _ledger_ids = _simple_campaign(tmp_path)
    outside = tmp_path / "outside"; outside.mkdir()
    (outside / "episode.json").write_text("{}")
    (root / "stray-evidence").symlink_to(outside, target_is_directory=True)

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
