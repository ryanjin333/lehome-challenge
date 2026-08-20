"""Worker no-wave-barrier lifecycle test."""

from __future__ import annotations

import pytest


def test_worker_stops_after_idle_timeout() -> None:
    from lehome_train.groot.experiment_worker import ExperimentWorker

    class Controller:
        def lease_next(self, *args, **kwargs):
            return None

    assert ExperimentWorker(Controller(), worker_id="worker", idle_timeout_seconds=0).run(max_jobs=1) == 0


def test_worker_blocks_mismatched_baked_trainer_identity_before_runner_launch(tmp_path) -> None:
    """A paid lease with a wrong image identity cannot enter the guest runner."""
    from lehome_train.groot.experiment_worker import ExperimentWorker
    from test_experiment_controller import _job

    job = _job(tmp_path, "image-mismatch")
    lease = type("Lease", (), {"lease_id": "lease", "experiment_id": job.experiment_id, "worker_id": "w", "job": job})()
    observed: list[str] = []

    class Controller:
        offered = True

        def lease_next(self, *_args, **_kwargs):
            if self.offered:
                self.offered = False
                return lease
            return None

        def block_infrastructure(self, _lease, reason, *_args):
            observed.append(reason)

        def retryable(self, *_args):
            raise AssertionError("identity mismatch is not retryable")

    class Runner:
        def run(self, *_args, **_kwargs):
            raise AssertionError("mismatched image job reached paid guest launch")

    def reject(_job):
        raise ValueError("training image identity does not match deployment gate")

    assert ExperimentWorker(
        Controller(), worker_id="w", runner=Runner(), identity_preflight=reject, idle_timeout_seconds=0,
    ).run(max_jobs=1) == 0
    assert observed == ["ValueError"]


def test_worker_cli_runs_leases_with_injected_transport_and_runtime(tmp_path) -> None:
    import importlib.util
    from pathlib import Path
    from test_experiment_job import _document
    from lehome_train.groot.experiment_job import dump_experiment_job, experiment_identity

    script = Path(__file__).resolve().parents[2] / "scripts/run_lehome_experiment_worker.py"
    spec = importlib.util.spec_from_file_location("worker_cli", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    def job(name):
        document = _document()
        document["arm"] = name
        document["trainer"] = {
            "image_id": "computeimage-training1",
            "oci_digest": "sha256:" + "a" * 64,
            "code_revision": "b" * 40,
        }
        document["publication"]["prefix"] = "experiments/" + name
        document["experiment_id"] = experiment_identity(document)
        return dump_experiment_job(tmp_path / (name + ".json"), document)

    gate = {
        "schema_version": 1,
        "kind": "lehome_experiment_pool_deployment_gate",
        "training_oci_digest": "sha256:" + "a" * 64,
        "training_code_revision": "b" * 40,
        "controller": {"instance_id": "computeinstance-controller1", "image_id": "computeimage-controller1", "image_status": "READY", "readback_verified": True},
        "training_workers": [
            {"slot": 1, "worker_id": "lehome-experiment-training-1", "instance_id": "computeinstance-training1", "image_id": "computeimage-training1", "image_status": "READY", "readback_verified": True},
            {"slot": 2, "worker_id": "lehome-experiment-training-2", "instance_id": "computeinstance-training2", "image_id": "computeimage-training1", "image_status": "READY", "readback_verified": True},
        ],
        "rollout_worker": {"worker_id": "lehome-experiment-evaluator", "instance_id": "computeinstance-rollout1", "image_id": "computeimage-rollout1", "image_status": "READY", "readback_verified": True},
        "teacher_probe": {"kind": "zero_perturbation_teacher_continuation_probe_v1", "attempt_id": "a" * 64, "round_id": "controlled-recovery-smoke-test-unsealed-staging", "matrix_sha256": "b" * 64, "materialization_sha256": "c" * 64, "episode_sha256": "d" * 64, "sync_receipt_sha256": "e" * 64, "immutable_revision": "f" * 40, "accepted": True, "readback_verified": True, "strict_seal_present": False},
    }
    import hashlib
    import json
    gate_path = tmp_path / "deployment-gate.json"
    gate_bytes = json.dumps(gate, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    gate_path.write_bytes(gate_bytes); gate_path.chmod(0o444)
    image_manifest = tmp_path / "training-image-manifest.json"
    image_manifest.write_text(json.dumps({"schema_version": 1, "kind": "vla-training-base-image", "oci_image": "ghcr.io/owner/trainer@sha256:" + "a" * 64, "oci_digest": "sha256:" + "a" * 64, "trainer_code_revision": "b" * 40}, sort_keys=True, separators=(",", ":")) + "\n")
    image_manifest.chmod(0o444)
    first, second = job("a"), job("b")
    class Client:
        def __init__(self): self.jobs = [first, second]; self.completed = []; self.published = []
        def lease_next(self, *_args, **_kwargs):
            if not self.jobs: return None
            job = self.jobs.pop(0)
            return type("Lease", (), {"lease_id": job.experiment_id, "experiment_id": job.experiment_id, "worker_id": "w", "job": job})()
        def complete(self, lease, receipt, _now): self.completed.append((lease.experiment_id, receipt))
        def heartbeat(self, lease, _now, lease_ns): return lease
        def publication_verified(self, experiment_id, publication, _now): self.published.append((experiment_id, publication))
        def retryable(self, *_args): raise AssertionError("unexpected retry")
        def block_infrastructure(self, *_args): raise AssertionError("unexpected block")
    class Runtime:
        def run(self, job):
            return {"terminal_receipt_sha256": "d" * 64, "publication": {"schema_version": 1, "experiment_id": job.experiment_id, "job_digest": job.experiment_id, "target_step": job.training.target_step, "repository": job.publication.checkpoint_repository, "immutable_revision": "a" * 40, "remote_prefix": job.publication.prefix + "/step-" + str(job.training.target_step), "artifact_sha256": "b" * 64, "receipt_sha256": "d" * 64, "readback_verified": True}}
    arguments = ["--controller-url", "https://controller", "--controller-ca-file", str(tmp_path / "ca.crt"), "--worker-id", "w", "--manifest-set-sha256", "a" * 64, "--cache-root", str(tmp_path), "--output-root", str(tmp_path), "--controller-token-file", str(tmp_path / "token"), "--hf-token-file", str(tmp_path / "hf"), "--deployment-gate", str(gate_path), "--deployment-gate-sha256", hashlib.sha256(gate_bytes).hexdigest(), "--training-image-manifest", str(image_manifest)]
    client = Client()
    assert module.main(arguments + ["--max-jobs", "2"], controller_factory=lambda *_: client, runtime_factory=lambda *_: Runtime()) == 2
    assert [item[0] for item in client.completed] == [first.experiment_id, second.experiment_id]
    assert [item[0] for item in client.published] == [first.experiment_id, second.experiment_id]

    rejected = job("wrong-code")
    wrong = dict(rejected.raw)
    wrong["trainer"] = {**wrong["trainer"], "code_revision": "c" * 40}
    rejected = dump_experiment_job(tmp_path / "wrong-code.json", wrong)

    class StopAfterBlock(BaseException):
        pass

    class RejectedClient(Client):
        def __init__(self):
            self.jobs = [rejected]; self.completed = []; self.published = []; self.blocked = []
        def block_infrastructure(self, _lease, reason, _now):
            self.blocked.append(reason)
            raise StopAfterBlock

    class RejectedRuntime:
        def run(self, *_args, **_kwargs):
            raise AssertionError("wrong code revision reached guest training launch")

    rejected_client = RejectedClient()
    with pytest.raises(StopAfterBlock):
        module.main(arguments + ["--max-jobs", "1"], controller_factory=lambda *_: rejected_client, runtime_factory=lambda *_: RejectedRuntime())
    assert rejected_client.blocked == ["ValueError"]


def test_worker_never_completes_after_heartbeat_loses_lease(tmp_path, monkeypatch) -> None:
    from lehome_train.groot.experiment_worker import ExperimentWorker
    from test_experiment_controller import _job

    job = _job(tmp_path, "a")
    lease = type("Lease", (), {"lease_id": "lease", "experiment_id": job.experiment_id, "worker_id": "w", "job": job})()
    class Controller:
        completed = []
        retried = []
        offered = True
        def lease_next(self, *_args, **_kwargs):
            if self.offered:
                self.offered = False
                return lease
            return None
        def heartbeat(self, *_args, **_kwargs):
            raise OSError("controller unavailable")
        def complete(self, *_args): self.completed.append(True)
        def retryable(self, *_args): self.retried.append(True)
        def block_infrastructure(self, *_args): raise AssertionError("wrong terminal state")
    class Runner:
        def run(self, _job, **_kwargs):
            __import__("time").sleep(0.03)
            return {"terminal_receipt_sha256": "d" * 64, "publication": {"schema_version": 1, "experiment_id": job.experiment_id, "job_digest": job.experiment_id, "target_step": job.training.target_step, "repository": job.publication.checkpoint_repository, "immutable_revision": "a" * 40, "remote_prefix": job.publication.prefix + "/step-" + str(job.training.target_step), "artifact_sha256": "b" * 64, "receipt_sha256": "d" * 64, "readback_verified": True}}
    controller = Controller()
    worker = ExperimentWorker(controller, worker_id="w", runner=Runner(), heartbeat_interval_seconds=0.01, idle_timeout_seconds=0)
    assert worker.run(max_jobs=1) == 0
    assert controller.completed == [] and controller.retried == [True]


def test_worker_reconciles_a_publishing_receipt_after_controller_transport_loss(tmp_path) -> None:
    """A completed terminal receipt must not be stranded in PUBLISHING."""
    from lehome_train.groot.experiment_worker import ControllerUnavailable, ExperimentWorker
    from test_experiment_controller import _job

    job = _job(tmp_path, "publishing")
    lease = type("Lease", (), {"lease_id": "lease", "experiment_id": job.experiment_id, "worker_id": "w", "job": job})()
    publication = {
        "schema_version": 1, "experiment_id": job.experiment_id, "job_digest": job.experiment_id,
        "target_step": job.training.target_step, "repository": job.publication.checkpoint_repository,
        "immutable_revision": "a" * 40, "remote_prefix": job.publication.prefix + "/step-500",
        "artifact_sha256": "b" * 64, "receipt_sha256": "d" * 64, "readback_verified": True,
    }

    class Controller:
        def __init__(self):
            self.offered = True
            self.completed = []
            self.published = []
            self.retried = []
            self.publication_failures = 1

        def lease_next(self, *_args, **_kwargs):
            if self.offered:
                self.offered = False
                return lease
            return None

        def heartbeat(self, lease, *_args):
            return lease

        def reconcile_terminal_receipt(self, incoming, receipt, *_args):
            if not self.completed:
                self.completed.append((incoming.experiment_id, receipt))
            assert self.completed == [(incoming.experiment_id, receipt)]
            return "PUBLISHING"

        def publication_verified(self, experiment_id, envelope, *_args):
            if self.publication_failures:
                self.publication_failures -= 1
                raise ControllerUnavailable("temporary controller outage")
            self.published.append((experiment_id, envelope))

        def retryable(self, *_args):
            self.retried.append(True)

        def block_infrastructure(self, *_args):
            raise AssertionError("publication transport outage is not a deterministic block")

    class Runner:
        def __init__(self): self.pending = []
        def run(self, _job, **_kwargs):
            return {"terminal_receipt_sha256": "d" * 64, "publication": publication}
        def persist_pending_publication(self, incoming, receipt, envelope):
            self.pending.append((incoming, receipt, envelope))
        def pending_publications(self):
            return tuple(self.pending)
        def clear_pending_publication(self, experiment_id):
            self.pending[:] = [item for item in self.pending if item[0].experiment_id != experiment_id]

    controller, runner = Controller(), Runner()
    worker = ExperimentWorker(controller, worker_id="w", runner=runner, idle_timeout_seconds=0)
    assert worker.run(max_jobs=1) == 0
    assert controller.completed == [(job.experiment_id, "d" * 64)]
    assert controller.retried == []
    assert runner.pending

    # A fresh worker loop owns no active training lease, but it can safely
    # publish the already read-back receipt once transport returns.
    assert worker.run(max_jobs=1) == 0
    assert [item[0] for item in controller.published] == [job.experiment_id]
    assert runner.pending == []


@pytest.mark.parametrize("committed_before_response", (False, True))
def test_worker_restart_reconciles_durable_terminal_handoff_without_rerunning_gpu_work(
    tmp_path, committed_before_response: bool,
) -> None:
    """Both sides of a lost completion response settle before any new lease."""
    from lehome_train.groot.experiment_worker import ControllerUnavailable, ExperimentWorker
    from test_experiment_controller import _job

    job = _job(tmp_path, "completion-" + str(committed_before_response))
    lease = type("Lease", (), {
        "lease_id": "lease", "experiment_id": job.experiment_id, "worker_id": "w",
        "capability": "training", "job": job,
    })()
    publication = {
        "schema_version": 1, "experiment_id": job.experiment_id, "job_digest": job.experiment_id,
        "target_step": job.training.target_step, "repository": job.publication.checkpoint_repository,
        "immutable_revision": "a" * 40, "remote_prefix": job.publication.prefix + "/step-500",
        "artifact_sha256": "b" * 64, "receipt_sha256": "d" * 64, "readback_verified": True,
    }

    class Controller:
        def __init__(self):
            self.offered, self.online, self.committed = True, False, committed_before_response
            self.reconciled: list[tuple[str, str]] = []
            self.published: list[str] = []

        def lease_next(self, *_args, **_kwargs):
            if self.offered:
                self.offered = False
                return lease
            return None

        def heartbeat(self, incoming, *_args): return incoming

        def reconcile_terminal_receipt(self, incoming, receipt, *_args):
            self.reconciled.append((incoming.experiment_id, receipt))
            if not self.online:
                # Model either a request that did not reach the controller or a
                # committed request whose response was lost.
                if committed_before_response:
                    self.committed = True
                raise ControllerUnavailable("completion response lost")
            assert incoming.lease_id == lease.lease_id and incoming.worker_id == "w"
            self.committed = True
            return "PUBLISHING"

        def publication_verified(self, experiment_id, _publication, *_args):
            assert self.committed
            self.published.append(experiment_id)

        def retryable(self, *_args):
            raise AssertionError("a durable terminal receipt must not be retried as GPU work")

        def block_infrastructure(self, *_args):
            raise AssertionError("a transport ambiguity must not block the immutable job")

    class Runner:
        def __init__(self): self.pending = []; self.calls = 0
        def run(self, _job, **_kwargs):
            self.calls += 1
            return {"terminal_receipt_sha256": "d" * 64, "publication": publication}
        def persist_pending_publication(self, incoming, receipt, envelope):
            self.pending[:] = [(incoming, receipt, envelope)]
        def pending_publications(self): return tuple(self.pending)
        def clear_pending_publication(self, experiment_id):
            self.pending[:] = [item for item in self.pending if item[0].experiment_id != experiment_id]

    controller, runner = Controller(), Runner()
    first = ExperimentWorker(controller, worker_id="w", runner=runner, idle_timeout_seconds=0)
    assert first.run(max_jobs=1) == 0
    assert runner.calls == 1 and runner.pending

    controller.online = True
    restarted = ExperimentWorker(controller, worker_id="w", runner=runner, idle_timeout_seconds=0)
    assert restarted.run(max_jobs=1) == 0
    assert runner.calls == 1
    assert controller.published == [job.experiment_id]
    assert runner.pending == []


def test_worker_restart_clears_a_published_handoff_after_evaluation_without_regression(tmp_path) -> None:
    """A stale durable handoff is not authority to reopen completed evaluation."""
    from lehome_train.groot.experiment_controller import ExperimentController
    from lehome_train.groot.experiment_worker import ExperimentWorker
    from test_experiment_controller import _job, _publication, _report

    controller = ExperimentController(tmp_path / "controller.sqlite3")
    job = _job(tmp_path, "completed-publication-handoff")
    controller.add_jobs([job])
    training = controller.lease_next("trainer", "training", 1, 100)
    assert training is not None
    controller.complete(training, "c" * 64, 2)
    publication = _publication(job)
    controller.publication_verified(job.experiment_id, publication, 3)
    evaluation = controller.lease_next("evaluator", "evaluation", 4, 100)
    assert evaluation is not None
    controller.submit_evaluation(evaluation, _report(job), 5)
    assert controller.state(job.experiment_id) == "COMPLETED"

    class Runner:
        def __init__(self):
            self.pending = [(training, "c" * 64, publication)]
            self.calls = 0

        def run(self, *_args, **_kwargs):
            self.calls += 1
            raise AssertionError("a completed durable handoff must not rerun training")

        def pending_publications(self):
            return tuple(self.pending)

        def clear_pending_publication(self, experiment_id):
            self.pending[:] = [item for item in self.pending if item[0].experiment_id != experiment_id]

    runner = Runner()
    events_before = controller._connection.execute(
        "SELECT COUNT(*) FROM events WHERE experiment_id=?", (job.experiment_id,),
    ).fetchone()[0]
    leases_before = controller._connection.execute("SELECT COUNT(*) FROM leases").fetchone()[0]
    promotions_before = controller._connection.execute(
        "SELECT COUNT(*) FROM promotion_candidates WHERE parent_experiment_id=?", (job.experiment_id,),
    ).fetchone()[0]
    assert ExperimentWorker(controller, worker_id="trainer", runner=runner, idle_timeout_seconds=0).run(max_jobs=1) == 0
    assert controller.state(job.experiment_id) == "COMPLETED"
    assert runner.calls == 0 and runner.pending == []
    assert controller._connection.execute(
        "SELECT COUNT(*) FROM events WHERE experiment_id=?", (job.experiment_id,),
    ).fetchone()[0] == events_before
    assert controller._connection.execute("SELECT COUNT(*) FROM leases").fetchone()[0] == leases_before
    assert controller._connection.execute(
        "SELECT COUNT(*) FROM promotion_candidates WHERE parent_experiment_id=?", (job.experiment_id,),
    ).fetchone()[0] == promotions_before
    assert controller.lease_next("evaluator-replay", "evaluation", 6, 100) is None


def test_runtime_runner_persists_only_verified_same_job_publication_handoffs(tmp_path) -> None:
    """A restart may reconcile a terminal receipt, never arbitrary partial bytes."""
    import importlib.util
    import json
    import shutil
    from pathlib import Path
    from test_experiment_runtime_request import _runtime_job_and_bundle

    script = Path(__file__).resolve().parents[2] / "scripts/run_lehome_experiment_worker.py"
    spec = importlib.util.spec_from_file_location("worker_handoff", script); assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    job, bundle = _runtime_job_and_bundle(tmp_path)
    token = tmp_path / "hf-token"; token.write_text("hf_abcdefghijklmnopqrst\n"); token.chmod(0o600)

    class Hydrator:
        def hydrate(self, _source, destination):
            shutil.copytree(bundle, destination, dirs_exist_ok=True)
            return destination

    guest_calls = []
    def guest(_argv, *, env, check):
        guest_calls.append(True)
        runtime = {}
        for line in Path(env["LEHOME_RUNTIME_ENV"]).read_text().splitlines():
            key, value = line.split("=", 1); runtime[key] = value.strip("'")
        output = Path(runtime["LEHOME_OUTPUT_ROOT"]) / "result.json"
        output.write_text(json.dumps({"immutable_checkpoint_publications": [{
            "optimizer_step": 500, "repository": job.publication.checkpoint_repository,
            "immutable_revision": "a" * 40, "remote_prefix": job.publication.prefix + "/step-500",
            "relative_path": "checkpoints/step-500.tar", "artifact_sha256": "d" * 64,
            "artifact_byte_size": 123, "descriptor_relative_path": "checkpoints/step-500.json",
            "descriptor_sha256": "e" * 64, "descriptor_byte_size": 45, "readback_verified": True,
        }]}, sort_keys=True, separators=(",", ":")))

    runner = module.ProductionRuntimeExperimentRunner(tmp_path / "cache", tmp_path / "output", token, hydrator=Hydrator(), training_script=Path("/fake/lehome-training.sh"), process_runner=guest)
    result = runner.run(job)
    # A retry after a controller outage reuses the verified terminal output;
    # it must not spend a second GPU training run for the same immutable job.
    assert runner.run(job) == result
    assert guest_calls == [True]
    lease = type("Lease", (), {
        "lease_id": "durable-lease", "experiment_id": job.experiment_id,
        "worker_id": "durable-worker", "capability": "training", "job": job,
    })()
    runner.persist_pending_publication(lease, result["terminal_receipt_sha256"], result["publication"])
    rehydrated = module.ProductionRuntimeExperimentRunner(tmp_path / "cache", tmp_path / "output", token)
    pending = rehydrated.pending_publications()
    assert len(pending) == 1 and pending[0][0].job.experiment_id == job.experiment_id
    assert pending[0][0].lease_id == "durable-lease"
    assert pending[0][0].worker_id == "durable-worker"
    assert pending[0][0].capability == "training"

    (tmp_path / "output" / "jobs" / job.experiment_id / "output" / "result.json").write_text("tampered")
    try:
        rehydrated.pending_publications()
    except ValueError as error:
        assert "handoff" in str(error)
    else:
        raise AssertionError("tampered terminal result was eligible for reconciliation")


def test_http_controller_client_types_transport_and_protocol_failures(tmp_path, monkeypatch) -> None:
    import importlib.util
    from pathlib import Path
    from urllib.error import HTTPError, URLError

    script = Path(__file__).resolve().parents[2] / "scripts/run_lehome_experiment_worker.py"
    spec = importlib.util.spec_from_file_location("worker_transport", script); assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    token = tmp_path / "controller-token"; token.write_text("secret\n"); token.chmod(0o600)
    ca = tmp_path / "controller-ca.crt"; ca.write_text("test CA\n"); ca.chmod(0o644)
    context = object()
    monkeypatch.setattr(module.ssl, "create_default_context", lambda *, cafile: context if cafile == str(ca) else None)
    client = module.HttpControllerClient("https://controller", token, "a" * 64, ca)
    monkeypatch.setattr(module, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(URLError("offline")))
    from lehome_train.groot.experiment_worker import ControllerProtocolError, ControllerUnavailable
    try:
        client._post("/lease", {})
    except ControllerUnavailable:
        pass
    else:
        raise AssertionError("controller transport outage was not typed retryable")
    monkeypatch.setattr(module, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(HTTPError("https://controller", 400, "bad", {}, None)))
    try:
        client._post("/lease", {})
    except ControllerProtocolError:
        pass
    else:
        raise AssertionError("controller protocol error was not typed deterministic")


def test_http_controller_client_replays_exact_terminal_receipt_identity(tmp_path, monkeypatch) -> None:
    import importlib.util
    import json
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "scripts/run_lehome_experiment_worker.py"
    spec = importlib.util.spec_from_file_location("worker_completion_client", script); assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    token = tmp_path / "controller-token"; token.write_text("secret\n"); token.chmod(0o600)
    observed: dict[str, object] = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_args): return None
        def read(self): return b'{"status":"publishing"}'

    def open_request(request, **_kwargs):
        observed["path"] = request.full_url
        observed["body"] = json.loads(request.data)
        return Response()

    monkeypatch.setattr(module, "urlopen", open_request)
    client = module.HttpControllerClient("http://127.0.0.1:8080", token, "a" * 64, None)
    lease = type("Lease", (), {"lease_id": "lease", "experiment_id": "b" * 64, "worker_id": "worker"})()
    assert client.reconcile_terminal_receipt(lease, "c" * 64, 7) == "PUBLISHING"
    assert observed == {
        "path": "http://127.0.0.1:8080/complete",
        "body": {
            "lease_id": "lease", "experiment_id": "b" * 64, "worker_id": "worker",
            "receipt_sha256": "c" * 64, "now_ns": 7,
        },
    }


def test_http_controller_client_requires_and_uses_a_safe_private_ca(tmp_path, monkeypatch) -> None:
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "scripts/run_lehome_experiment_worker.py"
    spec = importlib.util.spec_from_file_location("worker_private_tls", script); assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    token = tmp_path / "controller-token"; token.write_text("secret\n"); token.chmod(0o600)

    with pytest.raises(ValueError, match="private CA"):
        module.HttpControllerClient("https://controller", token, "a" * 64, None)

    ca = tmp_path / "controller-ca.crt"; ca.write_text("test CA\n"); ca.chmod(0o644)
    symlink = tmp_path / "ca-link.crt"; symlink.symlink_to(ca)
    with pytest.raises(ValueError, match="private CA"):
        module.HttpControllerClient("https://controller", token, "a" * 64, symlink)

    context = object()
    observed: dict[str, object] = {}
    monkeypatch.setattr(module.ssl, "create_default_context", lambda *, cafile: observed.update(cafile=cafile) or context)

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_args): return None
        def read(self): return b'{}'

    def open_request(_request, *, timeout, context):
        observed.update(timeout=timeout, context=context)
        return Response()

    monkeypatch.setattr(module, "urlopen", open_request)
    client = module.HttpControllerClient("https://controller", token, "a" * 64, ca)
    assert client._post("/lease", {}) == {}
    assert observed == {"cafile": str(ca), "timeout": 20, "context": context}


def test_worker_passes_exact_parent_publication_to_a_capable_runtime_runner(tmp_path) -> None:
    from lehome_train.groot.experiment_worker import ExperimentWorker
    from test_experiment_controller import _job

    job = _job(tmp_path, "child")
    parent_publication = {"schema_version": 2, "artifact_sha256": "a" * 64}
    lease = type("Lease", (), {"lease_id": "lease", "experiment_id": job.experiment_id, "worker_id": "w", "job": job, "parent_publication": parent_publication})()
    publication = {
        "schema_version": 1, "experiment_id": job.experiment_id, "job_digest": job.experiment_id,
        "target_step": job.training.target_step, "repository": job.publication.checkpoint_repository,
        "immutable_revision": "a" * 40, "remote_prefix": job.publication.prefix + "/step-500",
        "artifact_sha256": "b" * 64, "receipt_sha256": "d" * 64, "readback_verified": True,
    }

    class Controller:
        offered = True
        def lease_next(self, *_args, **_kwargs):
            if self.offered: self.offered = False; return lease
            return None
        def heartbeat(self, value, *_args): return value
        def complete(self, *_args): return None
        def publication_verified(self, *_args): return None
        def retryable(self, *_args): raise AssertionError("unexpected retry")
        def block_infrastructure(self, *_args): raise AssertionError("unexpected block")

    class Runner:
        observed = None
        def run(self, _job, *, parent_publication=None):
            self.observed = parent_publication
            return {"terminal_receipt_sha256": "d" * 64, "publication": publication}

    runner = Runner()
    assert ExperimentWorker(Controller(), worker_id="w", runner=runner, idle_timeout_seconds=0).run(max_jobs=1) == 1
    assert runner.observed == parent_publication
