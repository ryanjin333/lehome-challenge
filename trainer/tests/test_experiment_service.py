"""Controller HTTP service request validation test."""

from __future__ import annotations

import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest


def _request(url: str, token: str, path: str, body: dict[str, object]) -> tuple[int, dict[str, object]]:
    request = Request(url + path, data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"}, method="POST")
    with urlopen(request) as response:
        return response.status, json.loads(response.read())


def _get(url: str, token: str, path: str) -> tuple[int, dict[str, object]]:
    request = Request(url + path, headers={"Authorization": "Bearer " + token}, method="GET")
    with urlopen(request) as response:
        return response.status, json.loads(response.read())


def test_service_rejects_wildcard_without_proxy() -> None:
    from lehome_train.groot.experiment_service import validate_bind_address

    try:
        validate_bind_address("0.0.0.0", allow_tls_proxy=False)
    except ValueError:
        return
    raise AssertionError("unsafe wildcard bind accepted")


def test_service_returns_full_idempotent_lease_and_rejects_malformed_requests(tmp_path) -> None:
    from lehome_train.groot.experiment_controller import ExperimentController
    from lehome_train.groot.experiment_service import ExperimentService
    from test_experiment_controller import _job

    token = tmp_path / "token"; token.write_text("secret\n"); token.chmod(0o600)
    controller = ExperimentController(tmp_path / "controller.sqlite3")
    job = _job(tmp_path, "a"); controller.add_jobs([job])
    service = ExperimentService(("127.0.0.1", 0), controller, token)
    thread = threading.Thread(target=service.serve_forever, daemon=True); thread.start()
    url = "http://127.0.0.1:" + str(service.server_port)
    try:
        status, first = _request(url, "secret", "/lease", {"worker_id": "t1", "capability": "training", "now_ns": 1, "lease_ns": 100})
        assert status == 200 and first["job"]["experiment_id"] == job.experiment_id
        _, repeated = _request(url, "secret", "/lease", {"worker_id": "t1", "capability": "training", "now_ns": 2, "lease_ns": 100})
        assert repeated["lease_id"] == first["lease_id"]
        try:
            _request(url, "secret", "/complete", {"lease_id": first["lease_id"]})
        except HTTPError as error:
            assert error.code == 400
        else:
            raise AssertionError("malformed completion was accepted")
    finally:
        service.shutdown(); service.server_close(); thread.join()


def test_service_atomic_training_publication_evaluation_lifecycle(tmp_path) -> None:
    from lehome_train.groot.experiment_controller import ExperimentController
    from lehome_train.groot.experiment_service import ExperimentService
    from test_experiment_controller import _job, _publication, _report

    token = tmp_path / "token"; token.write_text("secret\n"); token.chmod(0o600)
    controller = ExperimentController(tmp_path / "controller.sqlite3")
    job = _job(tmp_path, "a"); controller.add_jobs([job])
    service = ExperimentService(("127.0.0.1", 0), controller, token)
    thread = threading.Thread(target=service.serve_forever, daemon=True); thread.start()
    url = "http://127.0.0.1:" + str(service.server_port)
    try:
        _, training = _request(url, "secret", "/lease", {"worker_id": "t", "capability": "training", "now_ns": 1, "lease_ns": 100})
        _, complete = _request(url, "secret", "/complete", {"lease_id": training["lease_id"], "experiment_id": job.experiment_id, "worker_id": "t", "receipt_sha256": "c" * 64, "now_ns": 2})
        assert complete["status"] == "publishing"
        # The first request may have committed while its response was lost.
        # Retrying the same immutable handoff cannot require the deleted lease.
        _, replay = _request(url, "secret", "/complete", {"lease_id": training["lease_id"], "experiment_id": job.experiment_id, "worker_id": "t", "receipt_sha256": "c" * 64, "now_ns": 99})
        assert replay["status"] == "publishing"
        with pytest.raises(HTTPError) as mismatch:
            _request(url, "secret", "/complete", {"lease_id": training["lease_id"], "experiment_id": job.experiment_id, "worker_id": "t", "receipt_sha256": "d" * 64, "now_ns": 100})
        assert mismatch.value.code == 400
        _, publication = _request(url, "secret", "/publication", {"experiment_id": job.experiment_id, "publication": _publication(job), "now_ns": 3})
        assert publication["status"] == "eval_ready"
        _, evaluation = _request(url, "secret", "/lease", {"worker_id": "e", "capability": "evaluation", "now_ns": 4, "lease_ns": 100})
        assert evaluation["publication"]["receipt_sha256"] == "c" * 64
        _, terminal = _request(url, "secret", "/evaluation", {"lease_id": evaluation["lease_id"], "experiment_id": job.experiment_id, "worker_id": "e", "report": _report(job), "now_ns": 5})
        assert terminal["status"] == "completed"
    finally:
        service.shutdown(); service.server_close(); thread.join()


def test_service_reconciles_expired_exact_completion_during_grace_and_rejects_stale_ordinary_calls(tmp_path) -> None:
    """The HTTP path preserves the controller-owned terminal handoff race fix."""
    from lehome_train.groot.experiment_controller import ExperimentController
    from lehome_train.groot.experiment_service import ExperimentService
    from test_experiment_controller import _job

    token = tmp_path / "token"; token.write_text("secret\n"); token.chmod(0o600)
    controller = ExperimentController(tmp_path / "controller.sqlite3", gpu_price_per_second=0.5)
    job = _job(tmp_path, "expired-service"); controller.add_jobs([job])
    service = ExperimentService(("127.0.0.1", 0), controller, token)
    thread = threading.Thread(target=service.serve_forever, daemon=True); thread.start()
    url = "http://127.0.0.1:" + str(service.server_port)
    try:
        _, lease = _request(url, "secret", "/lease", {"worker_id": "trainer", "capability": "training", "now_ns": 0, "lease_ns": 10})
        terminal = {"lease_id": lease["lease_id"], "experiment_id": job.experiment_id, "worker_id": "trainer", "now_ns": 11}
        # This is the first receipt delivery: the lease has expired, but the
        # controller admits exactly this original identity during its bounded
        # terminal handoff rather than allowing another trainer to rerun it.
        _, complete = _request(url, "secret", "/complete", {**terminal, "receipt_sha256": "c" * 64})
        assert complete == {"status": "publishing"}
        assert controller.state(job.experiment_id) == "PUBLISHING"
        settled = controller.budget_usage()
        assert settled == (500, 0.000000011, 0.0000000055)
        with pytest.raises(HTTPError) as repeat:
            _request(url, "secret", "/block", {**terminal, "reason": "late worker"})
        assert repeat.value.code == 400
        assert controller.budget_usage() == settled
    finally:
        service.shutdown(); service.server_close(); thread.join()


def test_service_exposes_authenticated_exact_capacity_snapshot(tmp_path) -> None:
    from lehome_train.groot.experiment_controller import ExperimentController
    from lehome_train.groot.experiment_service import ExperimentService
    from test_experiment_controller import _job

    token = tmp_path / "token"; token.write_text("secret\n"); token.chmod(0o600)
    controller = ExperimentController(tmp_path / "controller.sqlite3")
    controller.add_jobs([_job(tmp_path, "a")])
    service = ExperimentService(("127.0.0.1", 0), controller, token)
    thread = threading.Thread(target=service.serve_forever, daemon=True); thread.start()
    url = "http://127.0.0.1:" + str(service.server_port)
    try:
        _, snapshot = _get(url, "secret", "/capacity")
        assert set(snapshot) == {
            "schema_version",
            "ready_training_count",
            "leaseable_training_count",
            "eval_ready_count",
            "active_leases",
            "idle_stop_recommended",
        }
        assert snapshot["ready_training_count"] == 1
        assert snapshot["leaseable_training_count"] == 1
        try:
            _get(url, "wrong", "/capacity")
        except HTTPError as error:
            assert error.code == 401
        else:
            raise AssertionError("unauthenticated capacity snapshot was accepted")
    finally:
        service.shutdown(); service.server_close(); thread.join()


def test_service_authenticates_dedicated_awr_materialization_admission(tmp_path) -> None:
    from lehome_train.groot.experiment_ablation import bind_weighted_runtime_request_set
    from lehome_train.groot.experiment_controller import ExperimentController
    from lehome_train.groot.experiment_service import ExperimentService
    from test_experiment_ablation import _job, _pending, _receipt, _source

    token = tmp_path / "token"; token.write_text("secret\n"); token.chmod(0o600)
    parent = _job(); pending = _pending(parent); source = _source()
    admission = bind_weighted_runtime_request_set(parent, pending, weighted_runtime_request_set=source, materialization_receipt=_receipt(parent, pending, source))
    controller = ExperimentController(tmp_path / "controller.sqlite3")
    controller.add_jobs([admission.job])
    service = ExperimentService(("127.0.0.1", 0), controller, token)
    thread = threading.Thread(target=service.serve_forever, daemon=True); thread.start()
    url = "http://127.0.0.1:" + str(service.server_port)
    try:
        _, result = _request(url, "secret", "/awr-admission", {"experiment_id": admission.job.experiment_id, "receipt": dict(admission.receipt), "now_ns": 1})
        assert result == {"status": "ready", "receipt_sha256": admission.receipt_sha256}
        _, lease = _request(url, "secret", "/lease", {"worker_id": "trainer", "capability": "training", "now_ns": 2, "lease_ns": 100})
        assert lease["experiment_id"] == admission.job.experiment_id
    finally:
        service.shutdown(); service.server_close(); thread.join()


def test_service_keeps_final_unseen80_queue_and_winner_endpoint_outside_promotion_evaluation(tmp_path) -> None:
    from lehome_train.groot.experiment_manifest import APPROVED_ORIGINAL_12K_CHECKPOINT
    from lehome_train.groot.experiment_service import ExperimentService
    from test_experiment_controller import _controller_authorized_two_k_finalist
    from test_experiment_winner import _report

    token = tmp_path / "token"; token.write_text("secret\n"); token.chmod(0o600)
    controller, _initial, _seed, _one_k, job = _controller_authorized_two_k_finalist(tmp_path / "finalist")
    service = ExperimentService(("127.0.0.1", 0), controller, token)
    thread = threading.Thread(target=service.serve_forever, daemon=True); thread.start()
    url = "http://127.0.0.1:" + str(service.server_port)
    try:
        _, queue = _request(url, "secret", "/finalists", {"experiment_ids": [job.experiment_id], "matrix_sha256": "f" * 64, "now_ns": 100})
        assert queue == {"enqueued": 1}
        _, final = _request(url, "secret", "/lease", {"worker_id": "final-evaluator", "capability": "final_evaluation", "now_ns": 101, "lease_ns": 100})
        assert final["capability"] == "final_evaluation"
        report = _report(candidate=job.experiment_id, experiment_id=job.experiment_id, receipt="c" * 64, policy="d" * 64, matrix="f" * 64)
        try:
            _request(url, "secret", "/evaluation", {"lease_id": final["lease_id"], "experiment_id": job.experiment_id, "worker_id": "final-evaluator", "report": report, "now_ns": 102})
        except HTTPError as error:
            assert error.code == 400
        else:
            raise AssertionError("promotion endpoint accepted a final-evaluation lease")
        _, terminal = _request(url, "secret", "/final-evaluation", {"lease_id": final["lease_id"], "experiment_id": job.experiment_id, "worker_id": "final-evaluator", "report": report, "now_ns": 103})
        assert terminal == {"status": "completed"}
        baseline = _report(candidate="original-12k", policy=APPROVED_ORIGINAL_12K_CHECKPOINT["artifact_sha256"], matrix="f" * 64)
        _, decision = _request(url, "secret", "/final-winner", {"baseline_report": baseline, "matrix_sha256": "f" * 64, "now_ns": 104})
        assert decision["decision"] == "winner" and decision["candidate_id"] == job.experiment_id
        with pytest.raises(HTTPError) as error:
            _request(url, "secret", "/final-winner", {"baseline_report": baseline, "original_12k_checkpoint_digest": "0" * 64, "matrix_sha256": "f" * 64, "now_ns": 105})
        assert error.value.code == 400
    finally:
        service.shutdown(); service.server_close(); thread.join()
