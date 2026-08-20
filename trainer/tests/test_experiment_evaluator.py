"""CLI and final-evaluation queue contracts for the evaluator appliance."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import types

import pytest


def _evaluator_module() -> object:
    path = Path(__file__).parents[2] / "scripts" / "run_lehome_experiment_evaluator.py"
    spec = importlib.util.spec_from_file_location("lehome_experiment_evaluator_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluator_cli_passes_the_exact_manifest_set_digest_to_its_controller_client(tmp_path, monkeypatch) -> None:
    """A worker must never turn its campaign identity into the old zero digest."""
    module = _evaluator_module()
    categories = ("top_long", "top_short", "pant_long", "pant_short")
    rows = [
        {"trial_id": f"{category}-{offset}", "category": category, "garment_name": f"{category}-garment-{offset}", "release_stage": "public_unseen", "seed": offset}
        for category in categories for offset in range(5)
    ]
    matrix = tmp_path / "matrix.json"; matrix.write_text(json.dumps(rows, sort_keys=True, separators=(",", ":")) + "\n")
    token = tmp_path / "controller-token"; token.write_text("secret\n"); token.chmod(0o600)
    campaign_root = tmp_path / "campaign"; campaign_root.mkdir()
    manifest = "a" * 64
    observed: dict[str, object] = {}

    ca = tmp_path / "controller-ca.crt"; ca.write_text("test CA\n")

    class FakeClient:
        def __init__(self, url: str, token_file: Path, manifest_set_sha256: str, ca_file: Path) -> None:
            observed.update(url=url, token_file=token_file, manifest_set_sha256=manifest_set_sha256, ca_file=ca_file)

    monkeypatch.setitem(sys.modules, "run_lehome_experiment_worker", types.SimpleNamespace(HttpControllerClient=FakeClient))
    monkeypatch.setattr(module, "run_evaluation_loop", lambda controller, adapter, **kwargs: observed.update(controller=controller, kwargs=kwargs) or 0)

    assert module.main([
        "--controller-url", "http://127.0.0.1:8080",
        "--controller-ca-file", str(ca),
        "--matrix", str(matrix), "--matrix-sha256", hashlib.sha256(matrix.read_bytes()).hexdigest(),
        "--manifest-set-sha256", manifest,
        "--token-file", str(token), "--campaign-root", str(campaign_root),
    ]) == 0
    assert observed["manifest_set_sha256"] == manifest
    assert observed["ca_file"] == ca
    assert observed["kwargs"]["matrix_sha256"] == hashlib.sha256(matrix.read_bytes()).hexdigest()


def test_evaluator_rejects_an_envelope_before_constructing_the_controller_client(tmp_path, monkeypatch) -> None:
    module = _evaluator_module()
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps({"schema_version": 1, "training_holdouts": [], "trials": []}))
    token = tmp_path / "controller-token"; token.write_text("secret\n"); token.chmod(0o600)
    campaign_root = tmp_path / "campaign"; campaign_root.mkdir()
    ca = tmp_path / "controller-ca.crt"; ca.write_text("test CA\n")
    constructed: list[bool] = []
    monkeypatch.setitem(sys.modules, "run_lehome_experiment_worker", types.SimpleNamespace(HttpControllerClient=lambda *_args: constructed.append(True)))
    with pytest.raises(ValueError, match="exactly 20"):
        module.main([
            "--controller-url", "http://127.0.0.1:8080",
            "--controller-ca-file", str(ca),
            "--matrix", str(matrix), "--matrix-sha256", hashlib.sha256(matrix.read_bytes()).hexdigest(),
            "--manifest-set-sha256", "a" * 64,
            "--token-file", str(token), "--campaign-root", str(campaign_root),
        ])
    assert constructed == []


def test_controller_uses_a_distinct_final_unseen80_queue_and_winner_submission(tmp_path) -> None:
    """Final finalists cannot be consumed through the promotion /evaluation lane."""
    from lehome_train.groot.experiment_manifest import APPROVED_ORIGINAL_12K_CHECKPOINT
    from test_experiment_controller import _controller_authorized_two_k_finalist

    controller, _initial, _seed, _one_k, job = _controller_authorized_two_k_finalist(tmp_path)

    assert controller.enqueue_finalists([job.experiment_id], matrix_sha256="f" * 64, now_ns=100) == 1
    final = controller.lease_next("final-evaluator", "final_evaluation", now_ns=101, lease_ns=100)
    assert final and final.capability == "final_evaluation" and final.experiment_id == job.experiment_id
    assert controller.lease_next("promotion-evaluator", "evaluation", now_ns=102, lease_ns=100) is None

    from test_experiment_winner import _report

    report = _report(
        candidate=job.experiment_id,
        experiment_id=job.experiment_id,
        receipt="c" * 64,
        policy="d" * 64,
        matrix="f" * 64,
    )
    controller.submit_final_evaluation(final, report, now_ns=103)
    assert controller.final_evaluation_state(job.experiment_id) == "COMPLETED"
    decision = controller.final_winner_decision(
        baseline_report=_report(candidate="original-12k", policy=APPROVED_ORIGINAL_12K_CHECKPOINT["artifact_sha256"], matrix="f" * 64),
        matrix_sha256="f" * 64,
        now_ns=104,
    )
    assert decision["decision"] == "winner"
    assert decision["candidate_id"] == job.experiment_id


def test_final_evaluator_loop_leases_and_submits_only_through_the_final_lane() -> None:
    """The final worker must not accidentally call the promotion endpoint."""
    module = _evaluator_module()
    observed: list[str] = []

    class Lease:
        experiment_id = "a" * 64
        lease_id = "lease"
        worker_id = "final-worker"
        publication = {"receipt_sha256": "b" * 64}
        evaluation_matrix_sha256 = "c" * 64

    class Controller:
        def lease_next(self, worker_id, capability, **kwargs):
            observed.append("lease:" + worker_id + ":" + capability)
            return Lease() if len(observed) == 1 else None

        def submit_final_evaluation(self, lease, report, now_ns):
            observed.append("final-submit")

        def heartbeat(self, lease, now_ns, lease_ns):
            return None

    class Adapter:
        def run(self, lease, matrix, matrix_sha256, workers, *, cancellation=None):
            return {"experiment_id": lease.experiment_id, "matrix_sha256": matrix_sha256}

    assert module.run_evaluation_loop(
        Controller(), Adapter(), matrix="/matrix.json", matrix_sha256="c" * 64,
        max_jobs=1, idle_timeout_seconds=0, poll_seconds=0.1,
        evaluation_capability="final_evaluation",
    ) == 1
    assert observed == ["lease:lehome-experiment-evaluator:final_evaluation", "final-submit"]


def test_promotion_evaluator_leases_with_the_immutable_rollout_identity() -> None:
    """The evaluator identity must match the deployment-gated capacity worker."""
    module = _evaluator_module()
    observed: list[str] = []

    class Lease:
        experiment_id = "a" * 64
        lease_id = "lease"
        worker_id = "lehome-experiment-evaluator"
        evaluation_matrix_sha256 = "c" * 64

    class Controller:
        def lease_next(self, worker_id, capability, **kwargs):
            observed.append("lease:" + worker_id + ":" + capability)
            return Lease() if len(observed) == 1 else None

        def submit_evaluation(self, _lease, _report, _now_ns):
            observed.append("promotion-submit")

        def heartbeat(self, _lease, _now_ns, _lease_ns):
            return None

    class Adapter:
        def run(self, lease, matrix, matrix_sha256, workers, *, cancellation=None):
            return {"experiment_id": lease.experiment_id, "matrix_sha256": matrix_sha256}

    assert module.run_evaluation_loop(
        Controller(), Adapter(), matrix="/matrix.json", matrix_sha256="c" * 64,
        max_jobs=1, idle_timeout_seconds=0, poll_seconds=0.1,
    ) == 1
    assert observed == ["lease:lehome-experiment-evaluator:evaluation", "promotion-submit"]


def test_evaluator_rejects_a_leased_matrix_mismatch_before_starting_the_campaign() -> None:
    """Frozen bytes and the controller lease must agree before paid rollout work."""
    module = _evaluator_module()
    observed: list[str] = []

    class Lease:
        experiment_id = "a" * 64
        lease_id = "lease"
        worker_id = "rollout-evaluator"
        evaluation_matrix_sha256 = "b" * 64

    class Controller:
        def lease_next(self, *_args, **_kwargs):
            return Lease() if not observed else None

        def heartbeat(self, lease, *_args):
            return lease

        def block_infrastructure(self, _lease, reason, _now_ns):
            observed.append(reason)

    class Adapter:
        def run(self, *_args, **_kwargs):
            observed.append("campaign-started")
            raise AssertionError("campaign must not start")

    assert module.run_evaluation_loop(
        Controller(), Adapter(), matrix="/matrix.json", matrix_sha256="c" * 64,
        max_jobs=1, idle_timeout_seconds=0, poll_seconds=0.1,
    ) == 0
    assert observed == ["evaluation_matrix_mismatch"]


def test_finalist_seen_regression_handoff_binds_path_experiment_receipt_and_evidence(tmp_path) -> None:
    module = _evaluator_module()
    experiment_id = "a" * 64
    receipt = "b" * 64
    evidence = {
        "schema_version": 1,
        "kind": "lehome_experiment_seen_regression_evidence",
        "candidate_checkpoint_receipt_sha256": receipt,
        "major_seen_regression": False,
        "readback_verified": True,
        "sealed": True,
        "report_sha256": "c" * 64,
    }
    evidence_bytes = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("ascii")
    descriptor = {
        "schema_version": 1,
        "kind": "lehome_finalist_seen_regression_handoff",
        "experiment_id": experiment_id,
        "checkpoint_receipt_sha256": receipt,
        "evidence": evidence,
        "evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
    }
    root = tmp_path / "handoffs"
    finalist = root / experiment_id
    finalist.mkdir(parents=True)
    path = finalist / f"{receipt}.json"
    path.write_text(json.dumps(descriptor, sort_keys=True, separators=(",", ":")), encoding="ascii")
    path.chmod(0o444)
    root.chmod(0o555)

    assert module.load_finalist_seen_regression_handoff(root, experiment_id, receipt) == evidence

    root.chmod(0o755)
    try:
        module.load_finalist_seen_regression_handoff(root, experiment_id, receipt)
    except ValueError as error:
        assert "root" in str(error)
    else:
        raise AssertionError("a mutable final handoff root was accepted")
    root.chmod(0o555)
    descriptor["experiment_id"] = "d" * 64
    path.chmod(0o600)
    path.write_text(json.dumps(descriptor, sort_keys=True, separators=(",", ":")), encoding="ascii")
    path.chmod(0o444)
    try:
        module.load_finalist_seen_regression_handoff(root, experiment_id, receipt)
    except ValueError as error:
        assert "handoff" in str(error)
    else:
        raise AssertionError("a different finalist's seen-regression receipt was accepted")


def test_baseline_policy_must_match_the_jobs_pinned_original_parent_before_gpu_campaign(tmp_path) -> None:
    module = _evaluator_module()
    matrix = tmp_path / "matrix.json"
    matrix.write_text('[{"trial_id":"trial-1"}]\n', encoding="utf-8")
    calls: list[object] = []
    expected = "a" * 64
    adapter = module.PersistentFourWorkerAdapter(
        campaign_root=tmp_path / "campaigns",
        runner=lambda *args, **kwargs: calls.append((args, kwargs)),
        summarizer=lambda **kwargs: tmp_path / "unused.json",
        baseline_policy={
            "repository": "owner/models",
            "immutable_revision": "b" * 40,
            "target_step": 12000,
            "artifact_sha256": "c" * 64,
        },
    )
    job = type("Job", (), {
        "evaluation": type("Evaluation", (), {"policy_digest": expected})(),
    })()
    lease = type("Lease", (), {
        "experiment_id": "d" * 64,
        "job": job,
        "publication": {
            "repository": "owner/models",
            "immutable_revision": "e" * 40,
            "target_step": 500,
            "artifact_sha256": "f" * 64,
            "receipt_sha256": "1" * 64,
            "readback_verified": True,
        },
    })()
    with pytest.raises(ValueError, match="pinned original parent"):
        adapter.run(lease, str(matrix), "2" * 64, 4)
    assert calls == []
