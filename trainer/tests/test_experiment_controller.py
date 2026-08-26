"""Single-writer asynchronous lease regression tests."""

from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path

import pytest

from test_experiment_job import _document
from lehome_train.groot.experiment_job import dump_experiment_job, experiment_identity, load_experiment_job


def _job(tmp_path: Path, name: str):

    document = _document()
    document["arm"] = name
    document["publication"]["prefix"] = "experiments/" + name  # type: ignore[index]
    document["experiment_id"] = experiment_identity(document)
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return load_experiment_job(path)


def _recovery_job_and_receipt(tmp_path: Path, name: str = "recovery-d"):
    from lehome_train.io import canonical_json_sha256

    source = {
        "repository": "owner/data",
        "revision": "b" * 40,
        "prefix": "recovery",
        "manifest_sha256": "a" * 64,
        "tree_sha256": "a" * 64,
    }
    receipt = {
        "schema_version": 1,
        "kind": "verified_recovery_dependency",
        "source": source,
        "readback_verified": True,
        "trajectories": {
            category: [category + str(index) for index in range(5)]
            for category in ("top_long", "top_short", "pant_long", "pant_short")
        },
    }
    document = _document()
    document["arm"] = name
    document["publication"]["prefix"] = "experiments/" + name
    document["data_sources"] = [{"kind": "recovery", **source}]
    document["mixture"] = {
        "bc_percent": 95,
        "added_percent": 5,
        "batch64_quotas": {"bc": 61, "rollout": 3, "dagger": 0},
        "sampling_strategy": "unweighted",
    }
    document["dependencies"] = [canonical_json_sha256(receipt)]
    return dump_experiment_job(tmp_path / (name + ".json"), document), receipt


def test_two_workers_lease_without_wave_barrier(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_controller import ExperimentController

    controller = ExperimentController(tmp_path / "controller.sqlite3")
    controller.add_jobs([_job(tmp_path, name) for name in ("a", "b", "c")])
    first = controller.lease_next("train-a", "training", now_ns=1, lease_ns=100)
    second = controller.lease_next("train-b", "training", now_ns=1, lease_ns=100)
    assert first is not None and second is not None and first.experiment_id != second.experiment_id
    controller.complete(first, "c" * 64, now_ns=2)
    assert controller.lease_next("train-a", "training", now_ns=3, lease_ns=100) is not None


def test_promote_requires_verified_parent_and_admits_child_immediately(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_controller import ExperimentController
    from lehome_train.groot.experiment_job import experiment_identity, load_experiment_job
    controller = ExperimentController(tmp_path / "controller.sqlite3")
    parent, peer = _job(tmp_path, "parent"), _job(tmp_path, "peer"); controller.add_jobs([parent, peer])
    lease = controller.lease_next("train-a", "training", 1, 100); assert lease
    controller.complete(lease, "c" * 64, 2); controller.publication_verified(parent.experiment_id, _publication(parent), 3)
    evaluation = controller.lease_next("eval", "evaluation", 4, 100); assert evaluation
    controller.submit_evaluation(evaluation, _report(parent), 4)
    peer_lease = controller.lease_next("train-peer", "training", 5, 100); assert peer_lease
    controller.complete(peer_lease, "c" * 64, 6); controller.publication_verified(peer.experiment_id, _publication(peer), 7)
    peer_eval = controller.lease_next("eval", "evaluation", 8, 100); assert peer_eval
    peer_report = _report(peer); peer_report["promotion_metrics"]["paired_improvement"] = -1.0; _rehash_report(peer_report)
    controller.submit_evaluation(peer_eval, peer_report, 9)
    # The first closure now leases the independent seed-check pair before a
    # 1K child can be selected; it cannot promote on a single lucky seed.
    child_lease = controller.lease_next("train-b", "training", 6, 100)
    assert child_lease and child_lease.job.admission["kind"] == "seed_repeat"


@pytest.mark.parametrize(
    "third_one_k_terminal",
    ("completed", "safety_rejected", "infrastructure"),
    ids=("safe", "safety-rejected", "infrastructure-blocked"),
)
def test_two_k_waits_for_closed_six_result_field_and_all_three_one_k_results(
    tmp_path: Path, third_one_k_terminal: str
) -> None:
    """Early 1K completions cannot turn a partial initial field into a 2K final."""
    from lehome_train.groot.experiment_controller import ExperimentController

    controller = ExperimentController(tmp_path / "controller.sqlite3")
    initial = {_name: _job(tmp_path, _name) for _name in "abcdefg"}
    controller.add_jobs(list(initial.values()))
    now_ns = 1

    def finish_lease(lease, *, terminal: str = "completed"):
        nonlocal now_ns
        now_ns += 1
        controller.complete(lease, "c" * 64, now_ns)
        now_ns += 1
        publication = _publication(lease.job)
        publication["artifact_sha256"] = lease.experiment_id
        controller.publication_verified(lease.experiment_id, publication, now_ns)
        now_ns += 1
        evaluation = controller.lease_next("evaluator", "evaluation", now_ns, 100)
        assert evaluation is not None and evaluation.experiment_id == lease.experiment_id
        now_ns += 1
        if terminal == "infrastructure":
            controller.block_infrastructure(evaluation, "simulator", now_ns)
        else:
            report = _report(lease.job)
            report["policy_digest"] = lease.experiment_id
            if terminal == "safety_rejected":
                report["promotion_metrics"]["safety_failure"] = True
            _rehash_report(report)
            controller.submit_evaluation(evaluation, report, now_ns)
        now_ns += 1
        return lease

    def finish_training(
        expected_id: str | None = None,
        expected_step: int | None = None,
        *,
        terminal: str = "completed",
    ):
        lease = controller.lease_next("trainer", "training", now_ns, 100)
        assert lease is not None
        if expected_id is not None:
            assert lease.experiment_id == expected_id
        if expected_step is not None:
            assert lease.job.training.target_step == expected_step
        return finish_lease(lease, terminal=terminal)

    # The first two results lease the fixed independent seed pair before any
    # 1K continuation.  The repeat jobs take priority over unstarted 500s so
    # the ranking check is asynchronous rather than a seven-arm wave.
    finish_training(initial["a"].experiment_id, 500)
    finish_training(initial["b"].experiment_id, 500)
    first_seed = finish_training(expected_step=500)
    second_seed = finish_training(expected_step=500)
    assert first_seed.job.admission["kind"] == second_seed.job.admission["kind"] == "seed_repeat"
    first_one_k = finish_training(expected_step=1000)
    finish_training(initial["c"].experiment_id, 500)
    finish_training(initial["d"].experiment_id, 500)
    second_one_k = finish_training(expected_step=1000)
    assert first_one_k.job.training.target_step == second_one_k.job.training.target_step == 1000

    # Four initial results and two early continuations are not a closed field.
    assert controller._connection.execute(
        "SELECT COUNT(*) FROM promotion_candidates WHERE kind='step_2000'"
    ).fetchone()[0] == 0
    fifth_initial = controller.lease_next("trainer", "training", now_ns, 100)
    assert fifth_initial is not None
    assert fifth_initial.experiment_id == initial["e"].experiment_id
    assert fifth_initial.job.training.target_step == 500
    finish_lease(fifth_initial)

    # Complete the fifth and sixth valid initial results.  The sixth result
    # admits the third 1K continuation, but still cannot create a 2K until it
    # itself has a terminal evaluated result.
    finish_training(initial["f"].experiment_id, 500)
    assert controller._connection.execute(
        "SELECT COUNT(*) FROM promotion_candidates WHERE kind='step_2000'"
    ).fetchone()[0] == 0
    third_one_k = finish_training(expected_step=1000, terminal=third_one_k_terminal)
    if third_one_k_terminal == "safety_rejected":
        assert controller.state(third_one_k.experiment_id) == "REJECTED"
    elif third_one_k_terminal == "infrastructure":
        assert controller.state(third_one_k.experiment_id) == "BLOCKED_INFRA"

    two_k = controller.lease_next("trainer", "training", now_ns, 100)
    assert two_k is not None and two_k.job.training.target_step == 2000
    assert third_one_k.job.training.target_step == 1000
    if third_one_k_terminal != "completed":
        parents = {
            str(row[0])
            for row in controller._connection.execute(
                "SELECT parent_experiment_id FROM promotion_candidates WHERE kind='step_2000'"
            ).fetchall()
        }
        assert parents == {first_one_k.experiment_id, second_one_k.experiment_id}


def test_evaluation_retry_is_releasable(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_controller import ExperimentController
    controller = ExperimentController(tmp_path / "controller.sqlite3")
    job = _job(tmp_path, "a"); controller.add_jobs([job])
    train = controller.lease_next("t", "training", 1, 10); assert train
    controller.complete(train, "c" * 64, 2)
    controller.publication_verified(job.experiment_id, _publication(job), 3)
    lease = controller.lease_next("e", "evaluation", 4, 10); assert lease and lease.publication
    controller.retryable(lease, "network", 5)
    assert controller.lease_next("e2", "evaluation", 6, 10).experiment_id == job.experiment_id


def test_unpaired_evaluation_waits_for_baseline_without_promoting_or_safety_rejection(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_controller import ExperimentController

    controller = ExperimentController(tmp_path / "controller.sqlite3")
    job = _job(tmp_path, "unpaired")
    controller.add_jobs([job])
    train = controller.lease_next("train", "training", 1, 100)
    assert train
    controller.complete(train, "c" * 64, 2)
    controller.publication_verified(job.experiment_id, _publication(job), 3)
    evaluation = controller.lease_next("eval", "evaluation", 4, 100)
    assert evaluation
    report = _report(job)
    report["promotion_metrics"]["pairing"] = {"status": "baseline_evaluation_required"}
    _rehash_report(report)
    controller.submit_evaluation(evaluation, report, 5)
    assert controller.state(job.experiment_id) == "EVAL_WAITING_BASELINE"
    assert controller.pending_promotions(job.experiment_id) == ()


def _publication(job, receipt: str = "c" * 64):
    return {
        "schema_version": 1,
        "experiment_id": job.experiment_id,
        "job_digest": job.experiment_id,
        "target_step": job.training.target_step,
        "repository": job.publication.checkpoint_repository,
        "immutable_revision": "a" * 40,
        "remote_prefix": job.publication.prefix + "/step-" + str(job.training.target_step),
        "artifact_sha256": "d" * 64,
        "receipt_sha256": receipt,
        "readback_verified": True,
    }


def _report(job):
    categories = {name: {"successes": 4, "episodes": 5} for name in ("top_long", "top_short", "pant_long", "pant_short")}
    artifacts = []
    for index, category in enumerate(categories):
        for offset in range(5):
            artifacts.append({"schedule_index": len(artifacts), "trial_id": f"{category}-{offset}", "attempt_id": hashlib.sha256(f"attempt-{category}-{offset}".encode()).hexdigest(), "category": category, "garment": category + "-garment", "seed": index * 10 + offset, "official_success": int(offset < 4), "terminal_event": "accepted" if offset < 4 else "rejected", "episode_sha256": hashlib.sha256(f"episode-{category}-{offset}".encode()).hexdigest(), "worker_receipt_sha256": hashlib.sha256(f"worker-{category}-{offset}".encode()).hexdigest()})
    report = {
        "schema_version": 1,
        "experiment_id": job.experiment_id,
        "checkpoint_receipt_sha256": "c" * 64,
        "matrix_sha256": job.evaluation.matrix_sha256,
        "policy_digest": "d" * 64,
        "categories": categories,
        "episode_artifacts": artifacts,
        "promotion_metrics": {"overall_successes": 16, "overall_episodes": 20, "overall_success_rate": 0.8, "safety_failure": False, "paired_improvement": 0.0, "gpu_seconds": 0.0, "infrastructure_retry_count": 0, "progress": {"observed_episodes": 20, "mean_terminal_progress": 0.5}, "recovery": {"recovery_attempts": 0, "successful_recoveries": 0}, "pairing": {"status": "available", "baseline_report_sha256": "f" * 64, "paired_trials": 20, "candidate_wins": 10, "baseline_wins": 10, "ties": 0, "paired_improvement": 0.0, "progress_improvement": 0.0, "recovery_improvement": 0.0}},
        "provenance": {"trainer": dict(job.trainer), "runtime": {"code_revision": "b" * 40, "asset_revision": "b" * 40, "simulator_version": "Isaac", "image_identity": "rollout-image"}, "data_sources": [dict(source.__dict__) if hasattr(source, "__dict__") else {"kind": source.kind, "repository": source.repository, "revision": source.revision, "prefix": source.prefix, "manifest_sha256": source.manifest_sha256, "tree_sha256": source.tree_sha256} for source in job.data_sources]},
        "strict_seal": False,
        "evidence_report_sha256": "e" * 64,
    }
    report["report_sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()
    return report


def _rehash_report(report):
    # Keep the paired aggregate and per-trial pairing attestation coherent in
    # the negative-score fixtures below; production validation rejects a
    # rehashed report that changes only one of the two fields.
    metrics = report.get("promotion_metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get("pairing"), dict):
        pairing = metrics["pairing"]
        value = metrics.get("paired_improvement")
        if value == -1.0:
            pairing.update({"candidate_wins": 0, "baseline_wins": 20, "ties": 0, "paired_improvement": -1.0})
    report["report_sha256"] = hashlib.sha256(json.dumps({key: value for key, value in report.items() if key != "report_sha256"}, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()


def _controller_authorized_two_k_finalist(
    tmp_path: Path,
    *,
    terminal: str = "completed",
    controller_kwargs: dict[str, object] | None = None,
):
    """Build the exact promotion lineage which is eligible for unseen-80.

    This deliberately uses the controller's candidate/materialization path,
    rather than fabricating a job that merely looks like a 2K continuation.
    """
    from lehome_train.groot.experiment_controller import ExperimentController

    if terminal not in {"completed", "safety_rejected", "infrastructure", "unevaluated"}:
        raise ValueError("test terminal is invalid")
    controller = ExperimentController(tmp_path / "controller.sqlite3", **(controller_kwargs or {}))
    initial = _job(tmp_path, "promotion-parent")
    controller.add_jobs([initial])

    def publish_and_evaluate(job, now_ns: int, *, safety_failure: bool = False) -> int:
        training = controller.lease_next("trainer-" + str(now_ns), "training", now_ns, 100)
        assert training and training.experiment_id == job.experiment_id
        controller.complete(training, "c" * 64, now_ns + 1)
        controller.publication_verified(job.experiment_id, _publication(job), now_ns + 2)
        evaluation = controller.lease_next("evaluator-" + str(now_ns), "evaluation", now_ns + 3, 100)
        assert evaluation and evaluation.experiment_id == job.experiment_id
        report = _report(job)
        if safety_failure:
            report["promotion_metrics"]["safety_failure"] = True
            _rehash_report(report)
        controller.submit_evaluation(evaluation, report, now_ns + 4)
        return now_ns + 5

    next_ns = publish_and_evaluate(initial, 1)
    with controller._transaction():
        controller._candidate(initial.experiment_id, "seed_repeat", next_ns)
        controller._candidate(initial.experiment_id, "step_1000", next_ns)
        controller._materialize_pending_candidates(next_ns)
    children = {
        controller._job(str(row[0])).admission["kind"]: controller._job(str(row[0]))
        for row in controller._connection.execute(
            "SELECT experiment_id FROM promotion_children WHERE parent_experiment_id=?",
            (initial.experiment_id,),
        )
    }
    seed, one_k = children["seed_repeat"], children["continuation"]
    # The controller prioritizes the higher-rung continuation over the seed
    # repeat once both are ready.
    next_ns = publish_and_evaluate(one_k, next_ns)
    next_ns = publish_and_evaluate(seed, next_ns)
    with controller._transaction():
        controller._candidate(one_k.experiment_id, "step_2000", next_ns)
        controller._materialize_pending_candidates(next_ns)
    two_k_id = controller._connection.execute(
        "SELECT experiment_id FROM promotion_children WHERE parent_experiment_id=?",
        (one_k.experiment_id,),
    ).fetchone()[0]
    two_k = controller._job(str(two_k_id))

    training = controller.lease_next("trainer-two-k", "training", next_ns + 1, 100)
    assert training and training.experiment_id == two_k.experiment_id
    controller.complete(training, "c" * 64, next_ns + 2)
    controller.publication_verified(two_k.experiment_id, _publication(two_k), next_ns + 3)
    if terminal == "unevaluated":
        return controller, initial, seed, one_k, two_k
    evaluation = controller.lease_next("evaluator-two-k", "evaluation", next_ns + 4, 100)
    assert evaluation and evaluation.experiment_id == two_k.experiment_id
    if terminal == "infrastructure":
        controller.block_infrastructure(evaluation, "simulator", next_ns + 5)
    else:
        report = _report(two_k)
        if terminal == "safety_rejected":
            report["promotion_metrics"]["safety_failure"] = True
            _rehash_report(report)
        controller.submit_evaluation(evaluation, report, next_ns + 5)
    return controller, initial, seed, one_k, two_k


def test_finalist_queue_admits_only_a_safe_controller_generated_two_k_continuation(tmp_path: Path) -> None:
    """Initial, seed, and 1K jobs are not final candidates, even if published."""
    controller, initial, seed, one_k, two_k = _controller_authorized_two_k_finalist(tmp_path)

    for job in (initial, seed, one_k):
        with pytest.raises(ValueError, match="controller"):
            controller.enqueue_finalists([job.experiment_id], matrix_sha256="f" * 64, now_ns=100)
    assert controller.enqueue_finalists([two_k.experiment_id], matrix_sha256="f" * 64, now_ns=101) == 1
    assert controller.final_evaluation_state(two_k.experiment_id) == "READY"


@pytest.mark.parametrize("terminal", ("unevaluated", "safety_rejected", "infrastructure"))
def test_finalist_queue_rejects_nonterminal_or_unsafe_two_k_continuations(tmp_path: Path, terminal: str) -> None:
    controller, _initial, _seed, _one_k, two_k = _controller_authorized_two_k_finalist(tmp_path, terminal=terminal)

    with pytest.raises(ValueError, match="controller"):
        controller.enqueue_finalists([two_k.experiment_id], matrix_sha256="f" * 64, now_ns=100)


def test_finalist_queue_rejects_a_completed_initial_two_k_job_without_promotion_lineage(tmp_path: Path) -> None:
    """A target-step field is not authority to enter final evaluation."""
    from lehome_train.groot.experiment_controller import ExperimentController
    from lehome_train.groot.experiment_job import dump_experiment_job

    document = _document()
    document["arm"] = "forged-two-k"
    document["training"]["target_step"] = 2000
    document["publication"]["prefix"] = "experiments/forged-two-k"
    initial_two_k = dump_experiment_job(tmp_path / "forged-two-k.json", document)
    controller = ExperimentController(tmp_path / "forged.sqlite3")
    controller.add_jobs([initial_two_k])
    training = controller.lease_next("trainer", "training", 1, 100)
    assert training
    controller.complete(training, "c" * 64, 2)
    controller.publication_verified(initial_two_k.experiment_id, _publication(initial_two_k), 3)
    evaluation = controller.lease_next("evaluator", "evaluation", 4, 100)
    assert evaluation
    controller.submit_evaluation(evaluation, _report(initial_two_k), 5)

    with pytest.raises(ValueError, match="controller"):
        controller.enqueue_finalists([initial_two_k.experiment_id], matrix_sha256="f" * 64, now_ns=6)


def test_finalist_queue_requires_the_complete_controller_selected_primary_and_tied_set(tmp_path: Path) -> None:
    """Final unseen-80 cannot be operator-cherry-picked after 2K admission."""
    from lehome_train.groot.experiment_controller import ExperimentController

    controller = ExperimentController(tmp_path / "controller.sqlite3")
    roots = [_job(tmp_path, name) for name in ("primary-root", "tied-root")]
    controller.add_jobs(roots)

    def publish_and_evaluate(lease, now_ns: int) -> None:
        controller.complete(lease, "c" * 64, now_ns)
        controller.publication_verified(lease.experiment_id, _publication(lease.job), now_ns + 1)
        evaluation = controller.lease_next("evaluator-" + str(now_ns), "evaluation", now_ns + 2, 100)
        assert evaluation is not None and evaluation.experiment_id == lease.experiment_id
        controller.submit_evaluation(evaluation, _report(lease.job), now_ns + 3)

    initial_leases = [controller.lease_next("root-" + str(index), "training", 1, 100) for index in range(2)]
    assert all(initial_leases)
    for index, lease in enumerate(initial_leases, start=1):
        assert lease is not None
        publish_and_evaluate(lease, index * 10)

    # Construct the two controller-generated 1K branches explicitly.  The
    # independent seed guard is covered separately; this fixture isolates the
    # final selector's primary+tied transaction.
    with controller._transaction():
        for root in roots:
            controller._candidate(root.experiment_id, "step_1000", 30)
        controller._materialize_pending_candidates(30)
    one_k = {
        controller._job(str(experiment_id)).admission["source_experiment_id"]: controller._job(str(experiment_id))
        for (experiment_id,) in controller._connection.execute("SELECT experiment_id FROM promotion_children")
        if controller._job(str(experiment_id)).training.target_step == 1000
    }
    for index in range(1, 3):
        lease = controller.lease_next("one-k-" + str(index), "training", 40 + index, 100)
        assert lease is not None and lease.job.training.target_step == 1000
        publish_and_evaluate(lease, 50 + index * 10)

    with controller._transaction():
        for index, root in enumerate(roots):
            controller._candidate(one_k[root.experiment_id].experiment_id, "step_2000", 80, tied_runner=index == 1)
        controller._materialize_pending_candidates(80)
    two_k = {
        controller._job(str(experiment_id)).admission["source_experiment_id"]: controller._job(str(experiment_id))
        for (experiment_id,) in controller._connection.execute("SELECT experiment_id FROM promotion_children")
        if controller._job(str(experiment_id)).training.target_step == 2000
    }
    for index in range(1, 3):
        lease = controller.lease_next("two-k-" + str(index), "training", 90 + index, 100)
        assert lease is not None and lease.job.training.target_step == 2000
        publish_and_evaluate(lease, 100 + index * 10)

    primary = two_k[one_k[roots[0].experiment_id].experiment_id]
    tied = two_k[one_k[roots[1].experiment_id].experiment_id]
    expected = [primary.experiment_id, tied.experiment_id]
    assert list(controller._controller_selected_finalist_ids()) == expected
    with pytest.raises(ValueError, match="exact controller-selected"):
        controller.enqueue_finalists([primary.experiment_id], matrix_sha256="f" * 64, now_ns=200)
    with pytest.raises(ValueError, match="exact controller-selected"):
        controller.enqueue_finalists([*expected, roots[0].experiment_id], matrix_sha256="f" * 64, now_ns=201)

    # A stale/crashed partial insert must fail closed: never append the tied
    # runner later and silently turn an operator retry into cherry-picking.
    controller._connection.execute(
        "INSERT INTO final_evaluations(experiment_id,matrix_sha256,state,report,received_ns) VALUES(?,?, 'READY',NULL,NULL)",
        (primary.experiment_id, "f" * 64),
    )
    controller._connection.commit()
    with pytest.raises(ValueError, match="partially enqueued"):
        controller.enqueue_finalists(expected, matrix_sha256="f" * 64, now_ns=202)
    assert controller.final_evaluation_state(tied.experiment_id) is None


def test_final_winner_waits_for_the_complete_controller_selected_set_on_one_matrix(tmp_path: Path) -> None:
    """A first unseen-80 finisher cannot become the sweep winner by itself."""
    from lehome_train.groot.experiment_controller import ExperimentController
    from lehome_train.groot.experiment_manifest import APPROVED_ORIGINAL_12K_CHECKPOINT
    from test_experiment_winner import _report as final_report

    controller = ExperimentController(tmp_path / "controller.sqlite3")
    primary, tied = _job(tmp_path, "final-primary"), _job(tmp_path, "final-tied")
    controller.add_jobs([primary, tied])
    primary_parent, tied_parent = "1" * 64, "2" * 64
    matrix = "f" * 64
    baseline = final_report(
        candidate="original-12k",
        policy=APPROVED_ORIGINAL_12K_CHECKPOINT["artifact_sha256"],
        matrix=matrix,
    )
    primary_report = final_report(
        candidate=primary.experiment_id,
        experiment_id=primary.experiment_id,
        matrix=matrix,
    )
    tied_report = final_report(
        candidate=tied.experiment_id,
        experiment_id=tied.experiment_id,
        matrix=matrix,
    )
    with controller._transaction():
        controller._connection.execute(
            "INSERT INTO promotion_candidates VALUES(?,?, 'ADMITTED', ?,0)",
            (primary_parent, "step_2000", 1),
        )
        controller._connection.execute(
            "INSERT INTO promotion_candidates VALUES(?,?, 'ADMITTED', ?,1)",
            (tied_parent, "step_2000", 1),
        )
        controller._connection.execute(
            "INSERT INTO promotion_children VALUES(?,?,?)", (primary.experiment_id, primary_parent, 0)
        )
        controller._connection.execute(
            "INSERT INTO promotion_children VALUES(?,?,?)", (tied.experiment_id, tied_parent, 1)
        )
        controller._connection.execute(
            "INSERT INTO final_evaluations VALUES(?,?,?,?,?)",
            (primary.experiment_id, matrix, "COMPLETED", json.dumps(primary_report, sort_keys=True, separators=(",", ":")), 1),
        )
        # A completed report on another matrix does not complete the exact
        # final comparison set.
        controller._connection.execute(
            "INSERT INTO final_evaluations VALUES(?,?,?,?,?)",
            (tied.experiment_id, "e" * 64, "COMPLETED", json.dumps(tied_report, sort_keys=True, separators=(",", ":")), 1),
        )

    assert controller.final_winner_decision(
        baseline_report=baseline, matrix_sha256=matrix, now_ns=2,
    ) == {"decision": "finalists_pending"}

    with controller._transaction():
        controller._connection.execute(
            "UPDATE final_evaluations SET matrix_sha256=? WHERE experiment_id=?",
            (matrix, tied.experiment_id),
        )
    assert controller.final_winner_decision(
        baseline_report=baseline, matrix_sha256=matrix, now_ns=3,
    )["decision"] == "winner"


def test_evaluation_submission_consumes_exact_lease_and_frees_next_evaluation(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_controller import ExperimentController

    controller = ExperimentController(tmp_path / "controller.sqlite3")
    first, second = _job(tmp_path, "a"), _job(tmp_path, "b")
    controller.add_jobs([first, second])
    for job, worker in ((first, "t1"), (second, "t2")):
        training = controller.lease_next(worker, "training", 1, 100)
        assert training
        controller.complete(training, "c" * 64, 2)
        controller.publication_verified(job.experiment_id, _publication(job), 3)
    evaluation = controller.lease_next("eval", "evaluation", 4, 100)
    assert evaluation and evaluation.publication is not None
    controller.submit_evaluation(evaluation, _report(first), 5)
    next_evaluation = controller.lease_next("eval", "evaluation", 6, 100)
    assert next_evaluation and next_evaluation.experiment_id == second.experiment_id


def test_publication_and_evaluation_reject_wrong_binding(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_controller import ExperimentController

    controller = ExperimentController(tmp_path / "controller.sqlite3")
    job = _job(tmp_path, "a")
    controller.add_jobs([job])
    training = controller.lease_next("t", "training", 1, 100)
    assert training
    controller.complete(training, "c" * 64, 2)
    wrong = _publication(job, "e" * 64)
    try:
        controller.publication_verified(job.experiment_id, wrong, 3)
    except ValueError as error:
        assert "receipt" in str(error)
    else:
        raise AssertionError("wrong terminal publication receipt accepted")


def test_exact_publication_replay_after_evaluation_is_state_monotonic(tmp_path: Path) -> None:
    """A lost publication response cannot reopen an already evaluated job."""
    from lehome_train.groot.experiment_controller import ExperimentController

    controller = ExperimentController(tmp_path / "controller.sqlite3")
    job = _job(tmp_path, "publication-replay")
    controller.add_jobs([job])
    training = controller.lease_next("trainer", "training", 1, 100)
    assert training is not None
    controller.complete(training, "c" * 64, 2)
    publication = _publication(job)
    assert controller.publication_verified(job.experiment_id, publication, 3) == "EVAL_READY"
    evaluation = controller.lease_next("evaluator", "evaluation", 4, 100)
    assert evaluation is not None
    controller.submit_evaluation(evaluation, _report(job), 5)
    assert controller.state(job.experiment_id) == "COMPLETED"

    events_before = controller._connection.execute(
        "SELECT COUNT(*) FROM events WHERE experiment_id=?", (job.experiment_id,),
    ).fetchone()[0]
    candidates_before = controller._connection.execute(
        "SELECT COUNT(*) FROM promotion_candidates WHERE parent_experiment_id=?", (job.experiment_id,),
    ).fetchone()[0]
    assert controller.publication_verified(job.experiment_id, dict(publication), 6) == "COMPLETED"
    assert controller.state(job.experiment_id) == "COMPLETED"
    assert controller._connection.execute(
        "SELECT COUNT(*) FROM events WHERE experiment_id=?", (job.experiment_id,),
    ).fetchone()[0] == events_before
    assert controller._connection.execute(
        "SELECT COUNT(*) FROM promotion_candidates WHERE parent_experiment_id=?", (job.experiment_id,),
    ).fetchone()[0] == candidates_before
    assert controller.lease_next("evaluator-replay", "evaluation", 7, 100) is None

    mismatched = dict(publication)
    mismatched["artifact_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="publication replay mismatch"):
        controller.publication_verified(job.experiment_id, mismatched, 8)


def test_recovery_receipt_unblocks_only_verified_thresholds(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_controller import ExperimentController
    from test_experiment_job import _document
    from lehome_train.io import canonical_json_sha256

    receipt = {"schema_version": 1, "kind": "verified_recovery_dependency", "source": {"repository": "owner/data", "revision": "b" * 40, "prefix": "recovery", "manifest_sha256": "a" * 64, "tree_sha256": "a" * 64}, "readback_verified": True, "trajectories": {name: [name + str(i) for i in range(5)] for name in ("top_long", "top_short", "pant_long", "pant_short")}}
    receipt_digest = canonical_json_sha256(receipt)

    def make(arm, percent):
        doc = _document(); doc["arm"] = arm; doc["data_sources"] = [{"kind": "recovery", "repository": "owner/data", "revision": "b" * 40, "prefix": "recovery", "manifest_sha256": "a" * 64, "tree_sha256": "a" * 64}]; doc["mixture"] = {"bc_percent": 100 - percent, "added_percent": percent, "batch64_quotas": {"bc": 64 - round(64 * percent / 100), "rollout": round(64 * percent / 100), "dagger": 0}, "sampling_strategy": "unweighted"}; doc["dependencies"] = [receipt_digest]; return dump_experiment_job(tmp_path / (arm + ".json"), doc)
    jobs = [make(arm, percent) for arm, percent in (("d", 5), ("e", 10), ("f", 15), ("g", 20))]
    controller = ExperimentController(tmp_path / "controller.sqlite3"); controller.add_jobs(jobs)
    assert controller.satisfy_dependency(receipt, 1) == 3
    assert controller.state(jobs[0].experiment_id) == "READY" and controller.state(jobs[-1].experiment_id) == "BLOCKED_DATA"


def test_unavailable_recovery_admission_keeps_valid_dependency_blocked_but_leases_ordinary_work(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_controller import ExperimentController

    recovery, receipt = _recovery_job_and_receipt(tmp_path)
    ordinary = _job(tmp_path, "a")
    controller = ExperimentController(
        tmp_path / "controller.sqlite3",
        recovery_collection_admitted=False,
    )
    controller.add_jobs([ordinary, recovery])

    assert controller.satisfy_dependency(receipt, 1) == 0
    assert controller.state(recovery.experiment_id) == "BLOCKED_DATA"
    lease = controller.lease_next("ordinary", "training", 2, 100)
    assert lease is not None and lease.experiment_id == ordinary.experiment_id


def test_unavailable_recovery_admission_demotes_persisted_ready_job_on_restart(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_controller import ExperimentController

    database = tmp_path / "controller.sqlite3"
    recovery, receipt = _recovery_job_and_receipt(tmp_path)
    admitted = ExperimentController(database, recovery_collection_admitted=True)
    admitted.add_jobs([recovery])
    assert admitted.satisfy_dependency(receipt, 1) == 1
    assert admitted.state(recovery.experiment_id) == "READY"
    admitted.close()

    restarted = ExperimentController(database, recovery_collection_admitted=False)
    restarted.add_jobs([recovery])
    assert restarted.state(recovery.experiment_id) == "BLOCKED_DATA"


def test_unavailable_recovery_admission_revokes_stale_admitted_recovery_continuation(
    tmp_path: Path,
) -> None:
    """A historical D--G promotion cannot occupy the ordinary continuation field."""
    from lehome_train.groot.experiment_controller import ExperimentController

    database = tmp_path / "stale-admitted-recovery.sqlite3"
    recovery, receipt = _recovery_job_and_receipt(tmp_path, "stale-admitted-recovery")
    admitted = ExperimentController(database, recovery_collection_admitted=True)
    admitted.add_jobs([recovery])
    assert admitted.satisfy_dependency(receipt, 1) == 1
    lease = admitted.lease_next("recovery-worker", "training", 2, 100)
    assert lease is not None
    admitted.complete(lease, "c" * 64, 3)
    admitted.publication_verified(recovery.experiment_id, _publication(recovery), 4)
    with admitted._transaction():
        admitted._connection.execute(
            "UPDATE jobs SET state='COMPLETED' WHERE experiment_id=?", (recovery.experiment_id,)
        )
        admitted._connection.execute(
            "INSERT INTO evaluations VALUES(?,?,?)",
            (recovery.experiment_id, json.dumps(_report(recovery), sort_keys=True, separators=(",", ":")), 5),
        )
        admitted._candidate(recovery.experiment_id, "step_1000", 6)
        admitted._materialize_pending_candidates(6)
    assert admitted._connection.execute(
        "SELECT state FROM promotion_candidates WHERE parent_experiment_id=? AND kind='step_1000'",
        (recovery.experiment_id,),
    ).fetchone()[0] == "ADMITTED"
    admitted.close()

    restarted = ExperimentController(database, recovery_collection_admitted=False)
    restarted.add_jobs([recovery])
    assert restarted._connection.execute(
        "SELECT state FROM promotion_candidates WHERE parent_experiment_id=? AND kind='step_1000'",
        (recovery.experiment_id,),
    ).fetchone()[0] == "REVOKED"
    assert restarted._candidate_count("step_1000") == 0
    assert restarted._candidate_parents("step_1000") == set()


def test_unavailable_recovery_admission_excludes_historical_terminal_recovery_scores(
    tmp_path: Path,
) -> None:
    """Historical D--G reports cannot rank ahead of ordinary A/B/C evidence."""
    from lehome_train.groot.experiment_controller import ExperimentController

    ordinary = _job(tmp_path, "ordinary-score")
    recovery, _receipt = _recovery_job_and_receipt(tmp_path, "recovery-score")
    controller = ExperimentController(
        tmp_path / "historical-recovery-score.sqlite3",
        recovery_collection_admitted=False,
    )
    controller.add_jobs([ordinary, recovery])
    with controller._transaction():
        for job in (ordinary, recovery):
            controller._connection.execute(
                "UPDATE jobs SET state='COMPLETED' WHERE experiment_id=?", (job.experiment_id,)
            )
            controller._connection.execute(
                "INSERT INTO evaluations VALUES(?,?,?)",
                (job.experiment_id, json.dumps(_report(job), sort_keys=True, separators=(",", ":")), 1),
            )

    assert [score.experiment_id for score in controller._rung_scores(500)] == [ordinary.experiment_id]
    assert [score.experiment_id for score in controller._ranked_initial_500()] == [ordinary.experiment_id]


def test_unavailable_recovery_admission_closes_the_three_arm_ordinary_field_through_two_k(
    tmp_path: Path,
) -> None:
    """A/B/C continue through 2K even when clean recovery admission is absent."""
    from lehome_train.groot.experiment_controller import ExperimentController

    controller = ExperimentController(
        tmp_path / "ordinary-only.sqlite3",
        recovery_collection_admitted=False,
    )
    initial = [_job(tmp_path, name) for name in ("a", "b", "c")]
    controller.add_jobs(initial)
    now_ns = 1

    def train_and_publish(*, expected_step: int, expected_kind: str | None = None):
        nonlocal now_ns
        lease = controller.lease_next("trainer-" + str(now_ns), "training", now_ns, 100)
        assert lease is not None and lease.job is not None
        assert lease.job.training.target_step == expected_step
        if expected_kind is not None:
            assert lease.job.admission["kind"] == expected_kind
        controller.complete(lease, "c" * 64, now_ns + 1)
        controller.publication_verified(lease.experiment_id, _publication(lease.job), now_ns + 2)
        now_ns += 3
        return lease.job

    def evaluate(expected_id: str):
        nonlocal now_ns
        lease = controller.lease_next("evaluator-" + str(now_ns), "evaluation", now_ns, 100)
        assert lease is not None and lease.experiment_id == expected_id and lease.job is not None
        controller.submit_evaluation(lease, _report(lease.job), now_ns + 1)
        now_ns += 2

    first = train_and_publish(expected_step=500)
    evaluate(first.experiment_id)
    second = train_and_publish(expected_step=500)
    evaluate(second.experiment_id)
    first_seed = train_and_publish(expected_step=500, expected_kind="seed_repeat")
    evaluate(first_seed.experiment_id)
    second_seed = train_and_publish(expected_step=500, expected_kind="seed_repeat")
    evaluate(second_seed.experiment_id)

    one_k = train_and_publish(expected_step=1000, expected_kind="continuation")
    third = train_and_publish(expected_step=500)
    assert third.experiment_id == initial[2].experiment_id
    evaluate(one_k.experiment_id)
    evaluate(third.experiment_id)

    two_k = controller.lease_next("trainer-two-k", "training", now_ns, 100)
    assert two_k is not None and two_k.job is not None
    assert two_k.job.training.target_step == 2000


def test_unavailable_recovery_admission_cleans_stale_blocked_lease_and_budget_hold(tmp_path: Path) -> None:
    """A blocked recovery row never retains authority or blocks A/B/C budget."""
    from lehome_train.groot.experiment_controller import ExperimentController

    recovery, _receipt = _recovery_job_and_receipt(tmp_path)
    ordinary = _job(tmp_path, "a")
    controller = ExperimentController(
        tmp_path / "controller.sqlite3",
        gradient_step_ceiling=500,
        recovery_collection_admitted=False,
    )
    controller.add_jobs([recovery, ordinary])
    stale_lease_id = "stale-recovery-lease"
    with controller._transaction():
        controller._connection.execute(
            "INSERT INTO leases VALUES(?,?,?,?,?)",
            (stale_lease_id, recovery.experiment_id, "stale-worker", "training", 100),
        )
        controller._connection.execute(
            "INSERT INTO budget_reservations VALUES(?,?,?,?,?,?)",
            (stale_lease_id, recovery.experiment_id, 500, 0.0, 0.0, 0),
        )

    with pytest.raises(ValueError, match="invalid lease"):
        controller.lease_for(
            stale_lease_id,
            recovery.experiment_id,
            "stale-worker",
            now_ns=0,
        )
    assert controller._connection.execute(
        "SELECT COUNT(*) FROM leases WHERE experiment_id=?", (recovery.experiment_id,)
    ).fetchone()[0] == 0
    assert controller._connection.execute(
        "SELECT COUNT(*) FROM budget_reservations WHERE experiment_id=?", (recovery.experiment_id,)
    ).fetchone()[0] == 0
    ordinary_lease = controller.lease_next("ordinary-worker", "training", 1, 100)
    assert ordinary_lease is not None and ordinary_lease.experiment_id == ordinary.experiment_id


def test_unavailable_recovery_restart_settles_elapsed_gpu_budget_before_releasing_hold(
    tmp_path: Path,
) -> None:
    """Restart cleanup accounts for paid recovery time before A/B/C can reuse it."""
    from lehome_train.groot.experiment_controller import ExperimentController

    database = tmp_path / "elapsed-recovery.sqlite3"
    recovery, receipt = _recovery_job_and_receipt(tmp_path, "elapsed-recovery")
    started_ns = time.time_ns() - 2_000_000_000
    admitted = ExperimentController(
        database,
        gpu_seconds_ceiling=3.0,
        spend_ceiling=3.0,
        estimated_gpu_seconds_per_step=0.001,
        gpu_price_per_second=1.0,
        recovery_collection_admitted=True,
    )
    admitted.add_jobs([recovery])
    assert admitted.satisfy_dependency(receipt, started_ns) == 1
    lease = admitted.lease_next("recovery-worker", "training", started_ns, 1_000_000_000)
    assert lease is not None
    admitted.close()

    restarted = ExperimentController(
        database,
        gpu_seconds_ceiling=3.0,
        spend_ceiling=3.0,
        estimated_gpu_seconds_per_step=0.001,
        gpu_price_per_second=1.0,
        recovery_collection_admitted=False,
    )
    restarted.add_jobs([recovery])
    gradient_steps, gpu_seconds, spend = restarted.budget_usage()
    assert gradient_steps == 0
    assert gpu_seconds >= 1.0
    assert spend >= 1.0


def test_recovery_lease_defense_demotes_a_stale_ready_job_when_admission_is_unavailable(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_controller import ExperimentController

    recovery, receipt = _recovery_job_and_receipt(tmp_path)
    controller = ExperimentController(
        tmp_path / "controller.sqlite3",
        recovery_collection_admitted=False,
    )
    controller.add_jobs([recovery])
    assert controller.satisfy_dependency(receipt, 1) == 0
    # Model a stale persisted state from a prior accepted-gate controller.
    controller._connection.execute(
        "UPDATE jobs SET state='READY' WHERE experiment_id=?",
        (recovery.experiment_id,),
    )
    controller._connection.commit()

    assert controller.lease_next("trainer", "training", 2, 100) is None
    assert controller.state(recovery.experiment_id) == "BLOCKED_DATA"


@pytest.mark.parametrize("phase", ("training_leased", "evaluation_ready", "evaluation_leased"))
def test_unavailable_recovery_gate_revokes_every_stale_nonterminal_phase_on_restart(
    tmp_path: Path,
    phase: str,
) -> None:
    from lehome_train.groot.experiment_controller import ExperimentController

    database = tmp_path / (phase + ".sqlite3")
    recovery, receipt = _recovery_job_and_receipt(tmp_path, phase)
    admitted = ExperimentController(database, recovery_collection_admitted=True)
    admitted.add_jobs([recovery])
    assert admitted.satisfy_dependency(receipt, 1) == 1
    training = admitted.lease_next("trainer", "training", 2, 100)
    assert training is not None
    stale_lease = training
    if phase != "training_leased":
        admitted.complete(training, "c" * 64, 3)
        assert admitted.publication_verified(recovery.experiment_id, _publication(recovery), 4) == "EVAL_READY"
        if phase == "evaluation_leased":
            evaluation = admitted.lease_next("evaluator", "evaluation", 5, 100)
            assert evaluation is not None
            stale_lease = evaluation
    admitted.close()

    restarted = ExperimentController(database, recovery_collection_admitted=False)
    restarted.add_jobs([recovery])
    assert restarted.state(recovery.experiment_id) == "BLOCKED_DATA"
    assert restarted._connection.execute(
        "SELECT COUNT(*) FROM leases WHERE experiment_id=?", (recovery.experiment_id,)
    ).fetchone()[0] == 0
    assert restarted._connection.execute(
        "SELECT COUNT(*) FROM budget_reservations WHERE experiment_id=?", (recovery.experiment_id,)
    ).fetchone()[0] == 0
    with pytest.raises(ValueError, match="invalid lease|does not belong"):
        restarted.lease_for(
            stale_lease.lease_id,
            recovery.experiment_id,
            stale_lease.worker_id,
            now_ns=6,
        )
    with pytest.raises(ValueError, match="lease does not belong"):
        restarted.heartbeat(stale_lease.worker_id, stale_lease.lease_id, 6, 100)


def test_unavailable_recovery_gate_rejects_stale_publication_transition(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_controller import ExperimentController

    database = tmp_path / "publishing.sqlite3"
    recovery, receipt = _recovery_job_and_receipt(tmp_path, "publishing")
    admitted = ExperimentController(database, recovery_collection_admitted=True)
    admitted.add_jobs([recovery])
    assert admitted.satisfy_dependency(receipt, 1) == 1
    training = admitted.lease_next("trainer", "training", 2, 100)
    assert training is not None
    admitted.complete(training, "c" * 64, 3)
    assert admitted.state(recovery.experiment_id) == "PUBLISHING"
    admitted.close()

    restarted = ExperimentController(database, recovery_collection_admitted=False)
    restarted.add_jobs([recovery])
    assert restarted.state(recovery.experiment_id) == "BLOCKED_DATA"
    with pytest.raises(ValueError, match="recovery publication is not admitted"):
        restarted.publication_verified(recovery.experiment_id, _publication(recovery), 4)


def test_unavailable_recovery_admission_rejects_completed_recovery_final_winner_on_restart(
    tmp_path: Path,
) -> None:
    """Completed recovery finalists cannot become a winner after gate loss."""
    from lehome_train.groot.experiment_controller import ExperimentController
    from lehome_train.groot.experiment_manifest import APPROVED_ORIGINAL_12K_CHECKPOINT
    from test_experiment_winner import _report as final_report

    database = tmp_path / "completed-finalist.sqlite3"
    recovery, _receipt = _recovery_job_and_receipt(tmp_path, "completed-finalist")
    matrix = "f" * 64
    baseline = final_report(
        candidate="original-12k",
        policy=APPROVED_ORIGINAL_12K_CHECKPOINT["artifact_sha256"],
        matrix=matrix,
    )
    report = final_report(
        candidate=recovery.experiment_id,
        experiment_id=recovery.experiment_id,
        matrix=matrix,
    )
    admitted = ExperimentController(database, recovery_collection_admitted=True)
    admitted.add_jobs([recovery])
    with admitted._transaction():
        admitted._connection.execute(
            "UPDATE jobs SET state='COMPLETED' WHERE experiment_id=?", (recovery.experiment_id,)
        )
        admitted._connection.execute(
            "INSERT INTO promotion_candidates VALUES(?,?, 'ADMITTED', ?,0)",
            (recovery.experiment_id, "step_2000", 1),
        )
        admitted._connection.execute(
            "INSERT INTO promotion_children VALUES(?,?,?)",
            (recovery.experiment_id, recovery.experiment_id, 0),
        )
        admitted._connection.execute(
            "INSERT INTO final_evaluations VALUES(?,?,?,?,?)",
            (
                recovery.experiment_id,
                matrix,
                "COMPLETED",
                json.dumps(report, sort_keys=True, separators=(",", ":")),
                1,
            ),
        )
    admitted.close()

    restarted = ExperimentController(database, recovery_collection_admitted=False)
    restarted.add_jobs([recovery])
    with pytest.raises(ValueError, match="recovery final winner is not admitted"):
        restarted.final_winner_decision(
            baseline_report=baseline,
            matrix_sha256=matrix,
            now_ns=2,
        )


def test_unavailable_recovery_restart_revokes_pending_recovery_promotion_but_materializes_ordinary(
    tmp_path: Path,
) -> None:
    """A recovery promotion must not roll back ordinary restart reconciliation."""
    from lehome_train.groot.experiment_controller import ExperimentController

    database = tmp_path / "mixed-promotions.sqlite3"
    ordinary = _job(tmp_path, "ordinary-parent")
    recovery, receipt = _recovery_job_and_receipt(tmp_path, "recovery-parent")
    admitted = ExperimentController(database, recovery_collection_admitted=True)
    admitted.add_jobs([ordinary, recovery])
    assert admitted.satisfy_dependency(receipt, 1) == 1
    first = admitted.lease_next("ordinary-worker", "training", 2, 100)
    second = admitted.lease_next("recovery-worker", "training", 2, 100)
    assert first is not None and second is not None
    leases = {first.experiment_id: first, second.experiment_id: second}
    assert set(leases) == {ordinary.experiment_id, recovery.experiment_id}
    for parent in (ordinary, recovery):
        admitted.complete(leases[parent.experiment_id], "c" * 64, 3)
        admitted.publication_verified(parent.experiment_id, _publication(parent), 4)
        with admitted._transaction():
            admitted._connection.execute(
                "UPDATE jobs SET state='COMPLETED' WHERE experiment_id=?", (parent.experiment_id,)
            )
            admitted._connection.execute(
                "INSERT INTO evaluations VALUES(?,?,?)",
                (parent.experiment_id, json.dumps({"verified": True}), 5),
            )
            admitted._candidate(parent.experiment_id, "step_1000", 6)
    admitted.close()

    restarted = ExperimentController(database, recovery_collection_admitted=False)
    restarted.add_jobs([ordinary, recovery])
    restarted.reconcile_pending_candidates(7)

    ordinary_child = restarted._connection.execute(
        "SELECT experiment_id FROM promotion_children WHERE parent_experiment_id=?",
        (ordinary.experiment_id,),
    ).fetchone()
    assert ordinary_child is not None
    assert restarted._connection.execute(
        "SELECT state FROM promotion_candidates WHERE parent_experiment_id=? AND kind='step_1000'",
        (ordinary.experiment_id,),
    ).fetchone()[0] == "ADMITTED"
    assert restarted._connection.execute(
        "SELECT state FROM promotion_candidates WHERE parent_experiment_id=? AND kind='step_1000'",
        (recovery.experiment_id,),
    ).fetchone()[0] == "REVOKED"
    assert restarted.state(recovery.experiment_id) == "COMPLETED"


def test_recovery_dependency_requires_its_declared_immutable_receipt_digest(tmp_path: Path) -> None:
    """A matching source alone cannot unlock an arm bound to another receipt."""
    from lehome_train.groot.experiment_controller import ExperimentController
    from lehome_train.groot.experiment_job import dump_experiment_job
    from lehome_train.io import canonical_json_sha256
    from test_experiment_job import _document

    accepted = {"schema_version": 1, "kind": "verified_recovery_dependency", "source": {"repository": "owner/data", "revision": "b" * 40, "prefix": "recovery", "manifest_sha256": "a" * 64, "tree_sha256": "a" * 64}, "readback_verified": True, "trajectories": {name: [name + str(i) for i in range(5)] for name in ("top_long", "top_short", "pant_long", "pant_short")}}
    unrelated = json.loads(json.dumps(accepted))
    unrelated["trajectories"]["top_short"][0] = "another-valid-trajectory"
    document = _document()
    document["arm"] = "recovery-digest-bound"
    document["data_sources"] = [{"kind": "recovery", **accepted["source"]}]
    document["mixture"] = {"bc_percent": 95, "added_percent": 5, "batch64_quotas": {"bc": 61, "rollout": 3, "dagger": 0}, "sampling_strategy": "unweighted"}
    document["dependencies"] = [canonical_json_sha256(accepted)]
    job = dump_experiment_job(tmp_path / "recovery-digest-bound.json", document)
    controller = ExperimentController(tmp_path / "controller.sqlite3")
    controller.add_jobs([job])

    assert controller.satisfy_dependency(unrelated, 1) == 0
    assert controller.state(job.experiment_id) == "BLOCKED_DATA"
    assert controller._connection.execute("SELECT COUNT(*) FROM dependency_receipts").fetchone()[0] == 0
    assert controller.satisfy_dependency(accepted, 2) == 1
    assert controller.state(job.experiment_id) == "READY"


def test_verified_recovery_lineage_promotes_500_to_seed_repeat_one_k_and_two_k(tmp_path: Path) -> None:
    """Promoted recovery rungs retain the source receipt as well as their parent."""
    from lehome_train.groot.experiment_controller import ExperimentController
    from lehome_train.groot.experiment_job import dump_experiment_job
    from lehome_train.io import canonical_json_sha256

    receipt = {
        "schema_version": 1,
        "kind": "verified_recovery_dependency",
        "source": {
            "repository": "owner/data",
            "revision": "b" * 40,
            "prefix": "recovery/v1",
            "manifest_sha256": "a" * 64,
            "tree_sha256": "d" * 64,
        },
        "readback_verified": True,
        "trajectories": {
            category: [category + str(index) for index in range(5)]
            for category in ("top_long", "top_short", "pant_long", "pant_short")
        },
    }
    receipt_digest = canonical_json_sha256(receipt)
    document = _document()
    document["arm"] = "recovery-parent"
    document["data_sources"] = [{"kind": "recovery", **receipt["source"]}]
    document["mixture"] = {
        "bc_percent": 95,
        "added_percent": 5,
        "batch64_quotas": {"bc": 61, "rollout": 3, "dagger": 0},
        "sampling_strategy": "unweighted",
    }
    document["dependencies"] = [receipt_digest]
    parent = dump_experiment_job(tmp_path / "recovery-parent.json", document)
    database = tmp_path / "recovery-lineage.sqlite3"
    controller = ExperimentController(database)
    controller.add_jobs([parent])
    assert controller.satisfy_dependency(receipt, 1) == 1
    assert controller.state(parent.experiment_id) == "READY"

    parent_lease = controller.lease_next("parent", "training", 2, 100)
    assert parent_lease and parent_lease.experiment_id == parent.experiment_id
    controller.complete(parent_lease, "c" * 64, 3)
    controller.publication_verified(parent.experiment_id, _publication(parent), 4)
    controller._connection.execute("UPDATE jobs SET state='COMPLETED' WHERE experiment_id=?", (parent.experiment_id,))
    controller._connection.execute(
        "INSERT INTO evaluations VALUES(?,?,?)",
        (parent.experiment_id, json.dumps({"verified": True}), 5),
    )
    for kind in ("seed_repeat", "step_1000"):
        controller._connection.execute(
            "INSERT INTO promotion_candidates VALUES(?,?, 'PENDING', ?,0)",
            (parent.experiment_id, kind, 6),
        )
    controller._connection.commit()

    # A child that drops the source receipt must not consume an otherwise
    # valid admission merely because it still names the completed parent.
    injected_document = json.loads(json.dumps(dict(controller._generated_child(parent.experiment_id, "step_1000").raw)))
    injected_document["dependencies"] = [parent.experiment_id]
    injected = dump_experiment_job(tmp_path / "injected-recovery-child.json", injected_document)
    with pytest.raises(ValueError, match="controller-generated"):
        controller.promote(parent.experiment_id, injected, 7)

    controller.reconcile_pending_candidates(7)

    children = {
        controller._job(str(experiment_id)).admission["kind"]: controller._job(str(experiment_id))
        for (experiment_id,) in controller._connection.execute(
            "SELECT experiment_id FROM promotion_children WHERE parent_experiment_id=?",
            (parent.experiment_id,),
        )
    }
    seed_repeat, one_k = children["seed_repeat"], children["continuation"]
    for child in (seed_repeat, one_k):
        assert set(child.dependencies) == {parent.experiment_id, receipt_digest}
        assert controller.state(child.experiment_id) == "READY"
        persisted = {
            str(row[0]) for row in controller._connection.execute(
                "SELECT dependency FROM dependencies WHERE experiment_id=?", (child.experiment_id,)
            )
        }
        assert persisted == set(child.dependencies)

    first = controller.lease_next("seed", "training", 8, 100)
    second = controller.lease_next("one-k", "training", 8, 100)
    assert first and second
    seed_lease = first if first.experiment_id == seed_repeat.experiment_id else second
    one_k_lease = first if first.experiment_id == one_k.experiment_id else second
    assert seed_lease.experiment_id == seed_repeat.experiment_id
    assert one_k_lease.experiment_id == one_k.experiment_id
    controller.complete(seed_lease, "c" * 64, 9)
    controller.complete(one_k_lease, "c" * 64, 9)
    controller.publication_verified(one_k.experiment_id, _publication(one_k), 10)
    controller._connection.execute("UPDATE jobs SET state='COMPLETED' WHERE experiment_id=?", (one_k.experiment_id,))
    controller._connection.execute(
        "INSERT INTO evaluations VALUES(?,?,?)",
        (one_k.experiment_id, json.dumps({"verified": True}), 11),
    )
    controller._connection.execute(
        "INSERT INTO promotion_candidates VALUES(?,?, 'PENDING', ?,0)",
        (one_k.experiment_id, "step_2000", 12),
    )
    controller._connection.commit()
    controller.reconcile_pending_candidates(13)
    two_k_id = controller._connection.execute(
        "SELECT experiment_id FROM promotion_children WHERE parent_experiment_id=?", (one_k.experiment_id,)
    ).fetchone()[0]
    two_k = controller._job(str(two_k_id))
    assert two_k.training.target_step == 2000
    assert set(two_k.dependencies) == {one_k.experiment_id, receipt_digest}
    assert controller.state(two_k.experiment_id) == "READY"

    controller.close()
    restarted = ExperimentController(database)
    restarted.reconcile_pending_candidates(14)
    assert restarted.state(two_k.experiment_id) == "READY"
    lease = restarted.lease_next("two-k", "training", 15, 100)
    assert lease and lease.experiment_id == two_k.experiment_id


def test_promotion_requires_child_to_bind_parent_readback_checkpoint_and_receipt(tmp_path: Path) -> None:
    """A continuation cannot quietly resume from a mutable or unrelated checkpoint."""
    from lehome_train.groot.experiment_controller import ExperimentController
    from lehome_train.groot.experiment_job import dump_experiment_job

    controller = ExperimentController(tmp_path / "controller.sqlite3")
    parent, peer = _job(tmp_path, "parent"), _job(tmp_path, "peer")
    controller.add_jobs([parent, peer])
    train = controller.lease_next("train", "training", 1, 100)
    assert train
    controller.complete(train, "c" * 64, 2)
    controller.publication_verified(parent.experiment_id, _publication(parent), 3)
    evaluate = controller.lease_next("eval", "evaluation", 4, 100)
    assert evaluate
    controller.submit_evaluation(evaluate, _report(parent), 5)
    peer_lease = controller.lease_next("peer", "training", 6, 100); assert peer_lease
    controller.complete(peer_lease, "c" * 64, 7); controller.publication_verified(peer.experiment_id, _publication(peer), 8)
    peer_eval = controller.lease_next("eval", "evaluation", 9, 100); assert peer_eval
    peer_report = _report(peer); peer_report["promotion_metrics"]["paired_improvement"] = -1.0; _rehash_report(peer_report)
    controller.submit_evaluation(peer_eval, peer_report, 10)

    assert controller.pending_promotions(parent.experiment_id) == ()
    assert controller.state(parent.experiment_id) == "PROMOTED"


def test_continuation_promotion_preserves_the_original_baseline_policy(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_controller import ExperimentController
    from lehome_train.groot.experiment_job import dump_experiment_job

    controller = ExperimentController(tmp_path / "controller.sqlite3")
    parent = _job(tmp_path, "baseline-parent")
    controller.add_jobs([parent])
    training = controller.lease_next("trainer", "training", 1, 100)
    assert training
    controller.complete(training, "c" * 64, 2)
    controller.publication_verified(parent.experiment_id, _publication(parent), 3)
    controller._connection.execute(
        "UPDATE jobs SET state='COMPLETED' WHERE experiment_id=?", (parent.experiment_id,)
    )
    controller._connection.execute(
        "INSERT INTO evaluations VALUES(?,?,?)",
        (parent.experiment_id, json.dumps(_report(parent), sort_keys=True, separators=(",", ":")), 4),
    )
    controller._connection.execute(
        "INSERT INTO promotion_candidates VALUES(?,?, 'PENDING', ?,0)",
        (parent.experiment_id, "step_1000", 5),
    )
    controller._connection.commit()

    child = controller._generated_child(parent.experiment_id, "step_1000")
    document = json.loads(json.dumps(dict(child.raw)))
    document["evaluation"]["policy_digest"] = "e" * 64
    document.pop("experiment_id")
    tampered = dump_experiment_job(tmp_path / "tampered-continuation.json", document)

    try:
        controller.promote(parent.experiment_id, tampered, 6)
    except ValueError as error:
        assert "baseline" in str(error)
    else:
        raise AssertionError("continuation changed the original baseline policy")


def test_500_to_1k_budget_delta_uses_recorded_parent_not_staging_prefix(tmp_path: Path) -> None:
    """Production checkpoint prefixes are content locations, not step cursors."""
    from lehome_train.groot.experiment_controller import ExperimentController

    database = tmp_path / "controller.sqlite3"
    controller = ExperimentController(database, gradient_step_ceiling=1500)
    parent, peer = _job(tmp_path, "parent-staged"), _job(tmp_path, "peer-staged")
    controller.add_jobs([parent, peer])
    for job, worker, now_ns in ((parent, "parent", 1), (peer, "peer", 10)):
        lease = controller.lease_next(worker, "training", now_ns, 100)
        assert lease and lease.experiment_id == job.experiment_id
        controller.complete(lease, "c" * 64, now_ns + 1)
        publication = _publication(job)
        publication["remote_prefix"] = job.publication.prefix + "/production-checkpoint-staging/policy-current"
        controller.publication_verified(job.experiment_id, publication, now_ns + 2)
        evaluation = controller.lease_next("evaluator", "evaluation", now_ns + 3, 100)
        assert evaluation and evaluation.experiment_id == job.experiment_id
        report = _report(job)
        if job is peer:
            report["promotion_metrics"]["paired_improvement"] = -1.0
            _rehash_report(report)
        controller.submit_evaluation(evaluation, report, now_ns + 4)

    # This is a budget-lineage fixture, not a seed-ranking test: create the
    # already controller-approved continuation directly after its verified
    # parent evidence is present.
    with controller._transaction():
        controller._candidate(parent.experiment_id, "step_1000", 16)
        controller._materialize_pending_candidates(16)

    children = [
        controller._job(str(row[0]))
        for row in controller._connection.execute("SELECT experiment_id FROM promotion_children").fetchall()
        if controller._job(str(row[0])).training.target_step == 1000
    ]
    assert len(children) == 1
    controller.close()

    restarted = ExperimentController(database, gradient_step_ceiling=1500)
    first = restarted.lease_next("continuation", "training", 20, 100)
    assert first and first.experiment_id == children[0].experiment_id
    restarted.retryable(first, "preempted", 21)
    restarted.close()
    restarted = ExperimentController(database, gradient_step_ceiling=1500)
    retry = restarted.lease_next("continuation", "training", 22, 100)
    assert retry and retry.experiment_id == children[0].experiment_id
    restarted.complete(retry, "c" * 64, 23)
    assert restarted.budget_usage()[0] == 1500


def test_1k_to_2k_budget_delta_uses_recorded_parent_not_staging_prefix(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_controller import ExperimentController
    from lehome_train.groot.experiment_job import dump_experiment_job
    from test_experiment_job import _document

    document = _document()
    document["arm"] = "one-k-staged"
    document["training"]["target_step"] = 1000
    document["publication"]["prefix"] = "experiments/one-k-staged"
    parent = dump_experiment_job(tmp_path / "one-k-staged.json", document)
    database = tmp_path / "controller.sqlite3"
    controller = ExperimentController(database, gradient_step_ceiling=2000)
    controller.add_jobs([parent])
    lease = controller.lease_next("parent", "training", 1, 100)
    assert lease
    controller.complete(lease, "c" * 64, 2)
    publication = _publication(parent)
    publication["remote_prefix"] = parent.publication.prefix + "/production-checkpoint-staging/immutable-parent"
    controller.publication_verified(parent.experiment_id, publication, 3)
    evaluation = controller.lease_next("evaluator", "evaluation", 4, 100)
    assert evaluation
    controller.submit_evaluation(evaluation, _report(parent), 5)
    with controller._transaction():
        controller._candidate(parent.experiment_id, "step_2000", 6)
        controller._materialize_pending_candidates(6)
    child_id = str(controller._connection.execute(
        "SELECT experiment_id FROM promotion_children WHERE parent_experiment_id=?",
        (parent.experiment_id,),
    ).fetchone()[0])
    controller.close()

    restarted = ExperimentController(database, gradient_step_ceiling=2000)
    child = restarted.lease_next("continuation", "training", 7, 100)
    assert child and child.experiment_id == child_id
    restarted.complete(child, "c" * 64, 8)
    assert restarted.budget_usage()[0] == 2000


def test_promoted_budget_delta_fails_closed_when_parent_publication_step_drifts(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_controller import ExperimentController

    controller = ExperimentController(tmp_path / "controller.sqlite3")
    parent, peer = _job(tmp_path, "parent-step-drift"), _job(tmp_path, "peer-step-drift")
    controller.add_jobs([parent, peer])
    for job, worker, now_ns in ((parent, "parent", 1), (peer, "peer", 10)):
        lease = controller.lease_next(worker, "training", now_ns, 100)
        assert lease
        controller.complete(lease, "c" * 64, now_ns + 1)
        controller.publication_verified(job.experiment_id, _publication(job), now_ns + 2)
        evaluation = controller.lease_next("evaluator", "evaluation", now_ns + 3, 100)
        assert evaluation
        report = _report(job)
        if job is peer:
            report["promotion_metrics"]["paired_improvement"] = -1.0
            _rehash_report(report)
        controller.submit_evaluation(evaluation, report, now_ns + 4)
    with controller._transaction():
        controller._candidate(parent.experiment_id, "step_1000", 16)
        controller._materialize_pending_candidates(16)
    child_id, parent_id = controller._connection.execute(
        "SELECT experiment_id,parent_experiment_id FROM promotion_children "
        "JOIN jobs USING(experiment_id) WHERE json_extract(jobs.canonical,'$.training.target_step')=1000"
    ).fetchone()
    encoded = controller._connection.execute("SELECT publication FROM artifacts WHERE experiment_id=?", (parent_id,)).fetchone()[0]
    drifted = json.loads(encoded)
    drifted["target_step"] = 1000
    controller._connection.execute("UPDATE artifacts SET publication=? WHERE experiment_id=?", (json.dumps(drifted), parent_id))
    controller._connection.commit()

    try:
        controller.lease_next("continuation", "training", 30, 100)
    except RuntimeError as error:
        assert "parent" in str(error) and "step" in str(error)
    else:
        raise AssertionError(f"drifted parent publication admitted promoted child {child_id}")


def test_gradient_budget_is_reserved_before_a_training_lease(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_controller import ExperimentController

    controller = ExperimentController(tmp_path / "controller.sqlite3", gradient_step_ceiling=500)
    one, two = _job(tmp_path, "one"), _job(tmp_path, "two")
    controller.add_jobs([one, two])
    assert controller.lease_next("train-one", "training", 1, 100) is not None
    assert controller.lease_next("train-two", "training", 1, 100) is None
    assert controller.state(two.experiment_id) == "BLOCKED_BUDGET"


def test_tied_runner_8k_campaign_ceiling_is_independent_of_lease_order(tmp_path: Path) -> None:
    """The primary finalist must still run when the tied runner leases first."""
    from lehome_train.groot.experiment_controller import ExperimentController
    from lehome_train.groot.experiment_job import load_experiment_job

    def finalist(name: str):
        document = _document()
        document["arm"] = name
        document["training"]["target_step"] = 1000
        document["publication"]["prefix"] = "experiments/" + name
        document["experiment_id"] = experiment_identity(document)
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        return load_experiment_job(path)

    tied, primary = finalist("tied-first"), finalist("primary-second")
    controller = ExperimentController(
        tmp_path / "controller.sqlite3",
        gradient_step_ceiling=7000,
        tied_runner_gradient_step_ceiling=8000,
    )
    controller.add_jobs([tied, primary])
    controller._connection.execute(
        "INSERT INTO budget_usage VALUES(?,?,?,?)", ("prior-campaign", 6000, 0.0, 0.0)
    )
    controller._connection.execute(
        "INSERT INTO promotion_candidates VALUES(?,?, 'ADMITTED', ?,1)",
        (tied.experiment_id, "step_2000", 1),
    )
    controller._connection.execute(
        "INSERT INTO promotion_children VALUES(?,?,?)", (tied.experiment_id, tied.experiment_id, 1)
    )
    controller._connection.execute(
        "INSERT INTO promotion_children VALUES(?,?,?)", (primary.experiment_id, "parent-primary", 0)
    )
    controller._connection.commit()

    first = controller.lease_next("trainer-tied", "training", 1, 100)
    second = controller.lease_next("trainer-primary", "training", 2, 100)

    assert first is not None and first.experiment_id == tied.experiment_id
    assert second is not None and second.experiment_id == primary.experiment_id


def test_revoked_recovery_tied_runner_does_not_extend_any_gradient_budget_ceiling(
    tmp_path: Path,
) -> None:
    """A stale D--G tie cannot spend the extra 1K reserved for a live finalist."""
    from lehome_train.groot.experiment_controller import ExperimentController

    ordinary = _job(tmp_path, "ordinary-budget")
    recovery, _receipt = _recovery_job_and_receipt(tmp_path, "recovery-budget")
    controller = ExperimentController(
        tmp_path / "revoked-tied.sqlite3",
        gradient_step_ceiling=7000,
        tied_runner_gradient_step_ceiling=8000,
        recovery_collection_admitted=False,
    )
    controller.add_jobs([ordinary, recovery])
    with controller._transaction():
        controller._connection.execute(
            "INSERT INTO promotion_candidates VALUES(?,?, 'REVOKED', ?,1)",
            (recovery.experiment_id, "step_2000", 1),
        )
        controller._connection.execute(
            "INSERT INTO promotion_children VALUES(?,?,?)",
            (recovery.experiment_id, recovery.experiment_id, 1),
        )
        controller._connection.execute(
            "INSERT INTO budget_usage VALUES(?,?,?,?)", ("prior-budget", 7000, 0.0, 0.0)
        )

    assert controller._leaseable_training_count() == 0
    assert controller.lease_next("ordinary-worker", "training", 1, 100) is None

    with controller._transaction():
        controller._connection.execute("DELETE FROM budget_usage")
        controller._connection.execute(
            "INSERT INTO budget_usage VALUES(?,?,?,?)", ("prior-budget", 7000, 0.0, 0.0)
        )
        controller._connection.execute(
            "INSERT INTO budget_reservations VALUES(?,?,?,?,?,?)",
            ("completion-grace", ordinary.experiment_id, 500, 0.0, 0.0, 0),
        )
        with pytest.raises(RuntimeError, match="exceeds the campaign gradient budget"):
            controller._consume_completion_grace_gradient(
                "completion-grace", ordinary.experiment_id, 1,
            )


def test_preemption_charges_elapsed_gpu_time_once_without_consuming_gradient_twice(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_controller import ExperimentController

    controller = ExperimentController(tmp_path / "controller.sqlite3", estimated_gpu_seconds_per_step=1.0, gpu_price_per_second=0.5)
    job = _job(tmp_path, "preempted")
    controller.add_jobs([job])
    first = controller.lease_next("train", "training", 0, 10_000_000_000)
    assert first
    controller.retryable(first, "preempted", 2_000_000_000)
    second = controller.lease_next("train", "training", 3_000_000_000, 10_000_000_000)
    assert second
    controller.complete(second, "c" * 64, 5_000_000_000)
    assert controller.budget_usage() == (500, 4.0, 2.0)


def test_terminal_receipt_reconciliation_is_idempotent_and_rejects_a_mismatch(tmp_path: Path) -> None:
    """A lost completion response can be replayed without live lease authority."""
    from lehome_train.groot.experiment_controller import ExperimentController

    controller = ExperimentController(tmp_path / "controller.sqlite3")
    job = _job(tmp_path, "completion-reconciliation")
    controller.add_jobs([job])
    lease = controller.lease_next("trainer", "training", 0, 10)
    assert lease

    assert controller.reconcile_terminal_receipt(lease, "c" * 64, 1) == "PUBLISHING"
    # The initial transition removed the lease.  A response-lost retry is
    # therefore accepted from the immutable terminal receipt, not rejected as
    # a stale lease.
    assert controller.reconcile_terminal_receipt(lease, "c" * 64, 100) == "PUBLISHING"
    assert controller.state(job.experiment_id) == "PUBLISHING"
    assert controller._connection.execute(
        "SELECT receipt_sha256 FROM artifacts WHERE experiment_id=?", (job.experiment_id,)
    ).fetchone() == ("c" * 64,)

    with pytest.raises(ValueError, match="receipt mismatch"):
        controller.reconcile_terminal_receipt(lease, "d" * 64, 101)
    stolen = type(lease)(
        lease.lease_id, lease.experiment_id, "other-worker", lease.capability,
        lease.expires_ns, lease.job, lease.publication, lease.parent_publication,
        lease.evaluation_matrix_sha256,
    )
    with pytest.raises(ValueError, match="receipt ownership mismatch"):
        controller.reconcile_terminal_receipt(stolen, "c" * 64, 102)


@pytest.mark.parametrize("transition", ("retryable", "block_infrastructure"))
def test_expired_training_lease_rejects_ordinary_terminal_transitions_during_completion_grace(
    tmp_path: Path, transition: str
) -> None:
    """Only the exact durable completion receipt may cross the grace boundary."""
    from lehome_train.groot.experiment_controller import ExperimentController

    controller = ExperimentController(tmp_path / "controller.sqlite3", gpu_price_per_second=0.5)
    job = _job(tmp_path, "expired-training-" + transition)
    controller.add_jobs([job])
    stale = controller.lease_next("stale", "training", 0, 10)
    assert stale

    with pytest.raises(ValueError, match="invalid lease"):
        if transition == "retryable":
            controller.retryable(stale, "preempted", 11)
        else:
            controller.block_infrastructure(stale, "host lost", 11)

    assert controller.state(job.experiment_id) == "COMPLETION_GRACE"
    assert controller.budget_usage() == (0, 0.000000011, 0.0000000055)
    assert controller._connection.execute(
        "SELECT COUNT(*) FROM artifacts WHERE experiment_id=?", (job.experiment_id,)
    ).fetchone()[0] == 0

    assert controller.lease_next("replacement", "training", 12, 10) is None
    with pytest.raises(ValueError, match="invalid lease"):
        controller.retryable(stale, "late retry", 13)
    assert controller.budget_usage() == (0, 0.000000011, 0.0000000055)


def test_expired_training_completion_grace_reconciles_exact_receipt_before_any_retry(
    tmp_path: Path,
) -> None:
    """A never-arrived /complete cannot start a duplicate training attempt."""
    from lehome_train.groot.experiment_controller import (
        TERMINAL_RECEIPT_GRACE_NS,
        ExperimentController,
    )

    controller = ExperimentController(tmp_path / "controller.sqlite3", gpu_price_per_second=0.5)
    job = _job(tmp_path, "completion-grace-reconcile")
    controller.add_jobs([job])
    original = controller.lease_next("original", "training", 0, 10)
    assert original

    # A later request first commits expiry into the controller-owned handoff;
    # no other trainer can lease it while the original durable receipt is in
    # flight or its response was lost.
    assert controller.lease_next("replacement", "training", 11, 10) is None
    assert controller.state(job.experiment_id) == "COMPLETION_GRACE"
    handoff = controller._connection.execute(
        "SELECT lease_id,worker_id,attempt,grace_deadline_ns FROM terminal_handoffs WHERE experiment_id=?",
        (job.experiment_id,),
    ).fetchone()
    assert handoff is not None
    assert handoff[0] == original.lease_id and handoff[1] == "original"
    assert int(handoff[2]) >= 1
    assert int(handoff[3]) == 11 + TERMINAL_RECEIPT_GRACE_NS
    settled = controller.budget_usage()

    wrong_owner = type(original)(
        original.lease_id, original.experiment_id, "other-worker", original.capability,
        original.expires_ns, original.job, original.publication, original.parent_publication,
        original.evaluation_matrix_sha256,
    )
    with pytest.raises(ValueError, match="ownership mismatch"):
        controller.reconcile_terminal_receipt(wrong_owner, "c" * 64, 12)
    assert controller.state(job.experiment_id) == "COMPLETION_GRACE"

    assert controller.reconcile_terminal_receipt(original, "c" * 64, 13) == "PUBLISHING"
    assert controller.state(job.experiment_id) == "PUBLISHING"
    assert controller.lease_next("replacement", "training", 13, 10) is None
    assert controller._connection.execute(
        "SELECT COUNT(*) FROM terminal_handoffs WHERE experiment_id=?", (job.experiment_id,)
    ).fetchone()[0] == 0
    # Expiry settles elapsed GPU time once.  The durable receipt then consumes
    # the gradient reservation that grace kept on behalf of this exact run.
    assert controller.budget_usage() == (settled[0] + 500, settled[1], settled[2])


@pytest.mark.parametrize("target_step,expected_delta", ((500, 500), (1000, 500), (2000, 1000)))
def test_completion_grace_charges_exact_lineage_delta_once_on_success(
    tmp_path: Path, target_step: int, expected_delta: int,
) -> None:
    """A late receipt pays the same root/continuation delta as a live finish."""
    from lehome_train.groot.experiment_controller import ExperimentController

    controller = ExperimentController(
        tmp_path / f"grace-{target_step}.sqlite3",
        gradient_step_ceiling=10_000,
        tied_runner_gradient_step_ceiling=10_000,
        gpu_price_per_second=0.5,
    )
    parent = _job(tmp_path, f"grace-parent-{target_step}")
    controller.add_jobs([parent])

    def admit_child(source, kind: str, now_ns: int):
        lease = controller.lease_next("parent-" + kind, "training", now_ns, 10)
        assert lease and lease.experiment_id == source.experiment_id
        controller.complete(lease, "c" * 64, now_ns + 1)
        controller.publication_verified(source.experiment_id, _publication(source), now_ns + 2)
        controller._connection.execute(
            "UPDATE jobs SET state='COMPLETED' WHERE experiment_id=?", (source.experiment_id,)
        )
        controller._connection.execute(
            "INSERT INTO evaluations VALUES(?,?,?)",
            (source.experiment_id, json.dumps(_report(source), sort_keys=True, separators=(",", ":")), now_ns + 3),
        )
        controller._connection.commit()
        with controller._transaction():
            controller._candidate(source.experiment_id, kind, now_ns + 4)
            controller._materialize_pending_candidates(now_ns + 4)
        return controller._job(str(controller._connection.execute(
            "SELECT experiment_id FROM promotion_children WHERE parent_experiment_id=?",
            (source.experiment_id,),
        ).fetchone()[0]))

    job = parent
    now_ns = 0
    if target_step >= 1000:
        job = admit_child(parent, "step_1000", now_ns)
        now_ns = 100
    if target_step == 2000:
        job = admit_child(job, "step_2000", now_ns)
        now_ns = 200

    lease = controller.lease_next("late", "training", now_ns, 10)
    assert lease and lease.experiment_id == job.experiment_id
    before_expiry = controller.budget_usage()
    controller.capacity_snapshot(now_ns=now_ns + 11)
    settled = controller.budget_usage()
    assert settled == (before_expiry[0], before_expiry[1] + 0.000000011, before_expiry[2] + 0.0000000055)

    assert controller.reconcile_terminal_receipt(lease, "c" * 64, now_ns + 12) == "PUBLISHING"
    charged = controller.budget_usage()
    assert charged == (settled[0] + expected_delta, settled[1], settled[2])
    assert controller.reconcile_terminal_receipt(lease, "c" * 64, now_ns + 13) == "PUBLISHING"
    assert controller.budget_usage() == charged


def test_completion_grace_survives_controller_restart_and_remains_nonleaseable(
    tmp_path: Path,
) -> None:
    from lehome_train.groot.experiment_controller import ExperimentController

    database = tmp_path / "controller.sqlite3"
    controller = ExperimentController(database)
    job = _job(tmp_path, "completion-grace-restart")
    controller.add_jobs([job])
    original = controller.lease_next("original", "training", 0, 10)
    assert original
    snapshot = controller.capacity_snapshot(now_ns=11)
    assert snapshot["ready_training_count"] == 0
    assert snapshot["leaseable_training_count"] == 0
    controller.close()

    restarted = ExperimentController(database)
    assert restarted.lease_next("replacement", "training", 12, 10) is None
    assert restarted.reconcile_terminal_receipt(original, "c" * 64, 13) == "PUBLISHING"
    assert restarted.state(job.experiment_id) == "PUBLISHING"
    settled = restarted.budget_usage()
    assert settled == (500, 0.000000011, 0.0)
    assert restarted.reconcile_terminal_receipt(original, "c" * 64, 14) == "PUBLISHING"
    assert restarted.budget_usage() == settled


def test_expired_training_completion_grace_releases_once_then_rejects_old_receipt(
    tmp_path: Path,
) -> None:
    from lehome_train.groot.experiment_controller import (
        TERMINAL_RECEIPT_GRACE_NS,
        ExperimentController,
    )

    controller = ExperimentController(tmp_path / "controller.sqlite3", gpu_price_per_second=0.5)
    job = _job(tmp_path, "completion-grace-timeout")
    controller.add_jobs([job])
    original = controller.lease_next("original", "training", 0, 10)
    assert original
    assert controller.lease_next("other", "training", 11, 10) is None
    settled = controller.budget_usage()

    replacement = controller.lease_next(
        "replacement", "training", 11 + TERMINAL_RECEIPT_GRACE_NS + 1, 10,
    )
    assert replacement and replacement.experiment_id == job.experiment_id
    assert controller.state(job.experiment_id) == "LEASED"
    # The expired attempt's reservation was settled before the grace started;
    # there is only the replacement's active reservation now, never a second
    # settlement of the original lease.
    assert controller._connection.execute(
        "SELECT COUNT(*) FROM budget_reservations WHERE experiment_id=?", (job.experiment_id,)
    ).fetchone()[0] == 1
    # An empty grace never consumes its retained gradient hold or double-counts
    # the original attempt's already-settled wall-clock usage.
    assert controller.budget_usage() == settled
    with pytest.raises(ValueError, match="invalid lease"):
        controller.reconcile_terminal_receipt(original, "c" * 64, 12 + TERMINAL_RECEIPT_GRACE_NS)
    # The stale receipt neither settles nor mutates the replacement reservation.
    assert controller.lease_for(
        replacement.lease_id, job.experiment_id, "replacement", now_ns=12 + TERMINAL_RECEIPT_GRACE_NS
    ).lease_id == replacement.lease_id


def test_completion_grace_retains_its_gradient_reservation_under_campaign_ceiling(
    tmp_path: Path,
) -> None:
    """Other work cannot borrow a late receipt's still-authenticated budget."""
    from lehome_train.groot.experiment_controller import ExperimentController

    controller = ExperimentController(tmp_path / "controller.sqlite3", gradient_step_ceiling=500)
    original, other = _job(tmp_path, "grace-ceiling-original"), _job(tmp_path, "grace-ceiling-other")
    controller.add_jobs([original, other])
    lease = controller.lease_next("original", "training", 0, 10)
    assert lease and lease.experiment_id == original.experiment_id

    controller.capacity_snapshot(now_ns=11)
    assert controller.state(original.experiment_id) == "COMPLETION_GRACE"
    # The residual reservation has no wall-clock cost but still protects the
    # exact 500 gradients that the original run actually performed.
    held = controller._connection.execute(
        "SELECT gradient_steps,gpu_seconds,spend FROM budget_reservations WHERE lease_id=?",
        (lease.lease_id,),
    ).fetchone()
    assert held == (500, 0.0, 0.0)
    assert controller.lease_next("other", "training", 12, 10) is None
    assert controller.state(other.experiment_id) == "BLOCKED_BUDGET"

    assert controller.reconcile_terminal_receipt(lease, "c" * 64, 13) == "PUBLISHING"
    assert controller.budget_usage()[0] == 500


def test_completion_grace_fails_closed_if_a_legacy_ledger_already_exceeds_ceiling(
    tmp_path: Path,
) -> None:
    """A late result cannot publish without a valid accounted campaign budget."""
    from lehome_train.groot.experiment_controller import ExperimentController

    controller = ExperimentController(tmp_path / "controller.sqlite3", gradient_step_ceiling=500)
    job = _job(tmp_path, "grace-corrupt-ledger")
    controller.add_jobs([job])
    lease = controller.lease_next("trainer", "training", 0, 10)
    assert lease
    controller.capacity_snapshot(now_ns=11)
    controller._connection.execute(
        "INSERT INTO budget_usage VALUES(?,?,?,?)", ("legacy-overrun", 1, 0.0, 0.0)
    )
    controller._connection.commit()

    with pytest.raises(RuntimeError, match="campaign gradient budget"):
        controller.reconcile_terminal_receipt(lease, "c" * 64, 12)
    assert controller.state(job.experiment_id) == "COMPLETION_GRACE"
    assert controller._connection.execute(
        "SELECT COUNT(*) FROM artifacts WHERE experiment_id=?", (job.experiment_id,)
    ).fetchone()[0] == 0


def test_expired_heartbeat_and_lease_lookup_reconcile_before_renewal(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_controller import ExperimentController

    controller = ExperimentController(tmp_path / "controller.sqlite3")
    job = _job(tmp_path, "expired-heartbeat")
    controller.add_jobs([job])
    lease = controller.lease_next("trainer", "training", 0, 10)
    assert lease

    with pytest.raises(ValueError, match="lease does not belong"):
        controller.heartbeat("trainer", lease.lease_id, 11, 10)
    with pytest.raises(ValueError, match="invalid lease"):
        controller.lease_for(lease.lease_id, job.experiment_id, "trainer", now_ns=11)
    assert controller.state(job.experiment_id) == "COMPLETION_GRACE"


def test_expired_evaluation_and_final_evaluation_cannot_complete(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_controller import ExperimentController
    from test_experiment_winner import _report as _final_report

    controller = ExperimentController(tmp_path / "evaluation", gpu_price_per_second=0.5)
    job = _job(tmp_path, "expired-evaluation")
    controller.add_jobs([job])
    training = controller.lease_next("trainer", "training", 0, 100)
    assert training
    controller.complete(training, "c" * 64, 1)
    controller.publication_verified(job.experiment_id, _publication(job), 2)
    evaluation = controller.lease_next("evaluator", "evaluation", 3, 10)
    assert evaluation
    with pytest.raises(ValueError, match="invalid lease"):
        controller.submit_evaluation(evaluation, _report(job), 14)
    assert controller.state(job.experiment_id) == "EVAL_RETRYABLE"
    assert controller._connection.execute(
        "SELECT COUNT(*) FROM evaluations WHERE experiment_id=?", (job.experiment_id,)
    ).fetchone()[0] == 0

    final_controller, _initial, _seed, _one_k, finalist = _controller_authorized_two_k_finalist(
        tmp_path / "final", controller_kwargs={"gpu_price_per_second": 0.5}
    )
    final_controller.enqueue_finalists([finalist.experiment_id], matrix_sha256="f" * 64, now_ns=0)
    final = final_controller.lease_next("final-evaluator", "final_evaluation", 1, 10)
    assert final
    report = _final_report(
        candidate=finalist.experiment_id,
        experiment_id=finalist.experiment_id,
        receipt="c" * 64,
        policy="d" * 64,
        matrix="f" * 64,
    )
    with pytest.raises(ValueError, match="invalid lease"):
        final_controller.submit_final_evaluation(final, report, 12)
    assert final_controller.final_evaluation_state(finalist.experiment_id) == "RETRYABLE"


def test_training_lease_reserves_its_initial_wall_clock_gpu_time(tmp_path: Path) -> None:
    """A short step estimate cannot admit a longer paid training lease."""
    from lehome_train.groot.experiment_controller import ExperimentController

    second = 1_000_000_000
    controller = ExperimentController(
        tmp_path / "controller.sqlite3",
        gpu_seconds_ceiling=1.0,
        spend_ceiling=1.0,
        estimated_gpu_seconds_per_step=0.001,
        gpu_price_per_second=1.0,
    )
    job = _job(tmp_path, "training-initial-wall-clock")
    controller.add_jobs([job])

    assert controller.lease_next("trainer", "training", 0, 2 * second) is None
    assert controller.state(job.experiment_id) == "BLOCKED_BUDGET"


def test_training_heartbeat_extends_wall_clock_budget_or_cancels_at_ceiling(tmp_path: Path) -> None:
    """Training cannot remain leased after a heartbeat would overrun paid caps."""
    from lehome_train.groot.experiment_controller import ExperimentController

    second = 1_000_000_000
    controller = ExperimentController(
        tmp_path / "controller.sqlite3",
        gpu_seconds_ceiling=3.0,
        spend_ceiling=3.0,
        estimated_gpu_seconds_per_step=0.001,
        gpu_price_per_second=1.0,
    )
    job = _job(tmp_path, "training-heartbeat")
    controller.add_jobs([job])
    training = controller.lease_next("trainer", "training", 0, 2 * second)
    assert training

    extended = controller.heartbeat("trainer", training.lease_id, second, 2 * second)
    assert extended.expires_ns == 3 * second
    try:
        controller.heartbeat("trainer", training.lease_id, 2 * second, 2 * second)
    except RuntimeError as error:
        assert "training heartbeat" in str(error) and "budget" in str(error)
    else:
        raise AssertionError("over-budget training heartbeat kept a paid lease alive")
    assert controller.state(job.experiment_id) == "BLOCKED_BUDGET"
    assert controller.budget_usage() == (0, 2.0, 2.0)
    try:
        controller.lease_for(training.lease_id, job.experiment_id, "trainer", now_ns=2 * second)
    except ValueError:
        pass
    else:
        raise AssertionError("budget-blocked training lease remained live")


def test_evaluation_retries_reserve_and_settle_the_shared_gpu_spend_ceiling(tmp_path: Path) -> None:
    """Every rollout-GPU attempt counts, while evaluation never consumes gradients."""
    from lehome_train.groot.experiment_controller import ExperimentController

    second = 1_000_000_000
    controller = ExperimentController(
        tmp_path / "controller.sqlite3",
        gpu_seconds_ceiling=10.0,
        spend_ceiling=5.0,
        estimated_gpu_seconds_per_step=0.001,
        gpu_price_per_second=0.5,
    )
    job = _job(tmp_path, "evaluation-budget")
    controller.add_jobs([job])
    training = controller.lease_next("trainer", "training", 0, second)
    assert training
    controller.complete(training, "c" * 64, 0)
    controller.publication_verified(job.experiment_id, _publication(job), 0)

    first = controller.lease_next("evaluator", "evaluation", 0, 6 * second)
    assert first
    controller.retryable(first, "preempted", 2 * second)
    second_attempt = controller.lease_next("evaluator", "evaluation", 2 * second, 6 * second)
    assert second_attempt
    controller.retryable(second_attempt, "preempted", 6 * second)

    # Six seconds were actually consumed. A fresh six-second reservation would
    # exceed both the ten-GPU-second and five-dollar campaign ceilings.
    assert controller.lease_next("evaluator", "evaluation", 6 * second, 6 * second) is None
    assert controller.state(job.experiment_id) == "EVAL_BLOCKED_BUDGET"
    assert controller.budget_usage() == (500, 6.0, 3.0)


def test_evaluation_terminal_paths_settle_once_and_heartbeat_cannot_overreserve(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_controller import ExperimentController

    second = 1_000_000_000
    controller = ExperimentController(
        tmp_path / "controller.sqlite3",
        gpu_seconds_ceiling=3.0,
        spend_ceiling=3.0,
        estimated_gpu_seconds_per_step=0.001,
        gpu_price_per_second=1.0,
    )
    job = _job(tmp_path, "evaluation-heartbeat")
    controller.add_jobs([job])
    training = controller.lease_next("trainer", "training", 0, second)
    assert training
    controller.complete(training, "c" * 64, 0)
    controller.publication_verified(job.experiment_id, _publication(job), 0)
    evaluation = controller.lease_next("evaluator", "evaluation", 0, 2 * second)
    assert evaluation

    controller.heartbeat("evaluator", evaluation.lease_id, second, 2 * second)
    try:
        controller.heartbeat("evaluator", evaluation.lease_id, 2 * second, 2 * second)
    except RuntimeError as error:
        assert "budget" in str(error)
    else:
        raise AssertionError("evaluation heartbeat exceeded the reserved campaign budget")
    controller.block_infrastructure(evaluation, "simulator", 2 * second)
    assert controller.budget_usage() == (500, 2.0, 2.0)
    try:
        controller.block_infrastructure(evaluation, "duplicate", 2 * second)
    except ValueError:
        pass
    else:
        raise AssertionError("a terminal evaluation lease was settled twice")
    assert controller.budget_usage() == (500, 2.0, 2.0)


def test_evaluation_expiry_and_final_evaluation_retry_charge_elapsed_gpu_time(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_controller import ExperimentController

    second = 1_000_000_000
    controller = ExperimentController(tmp_path / "controller.sqlite3", gpu_price_per_second=0.25)
    job = _job(tmp_path, "evaluation-expiry")
    controller.add_jobs([job])
    training = controller.lease_next("trainer", "training", 0, second)
    assert training
    controller.complete(training, "c" * 64, 0)
    controller.publication_verified(job.experiment_id, _publication(job), 0)

    evaluation = controller.lease_next("evaluator", "evaluation", 0, second)
    assert evaluation
    controller.capacity_snapshot(now_ns=2 * second)
    controller.capacity_snapshot(now_ns=3 * second)
    assert controller.budget_usage() == (500, 2.0, 0.5)

    final_controller, _initial, _seed, _one_k, finalist = _controller_authorized_two_k_finalist(
        tmp_path / "finalist",
        controller_kwargs={"gpu_price_per_second": 0.25},
    )
    before_final = final_controller.budget_usage()
    assert final_controller.enqueue_finalists([finalist.experiment_id], matrix_sha256="f" * 64, now_ns=3 * second) == 1
    final = final_controller.lease_next("evaluator", "final_evaluation", 3 * second, 4 * second)
    assert final
    final_controller.retryable(final, "preempted", 5 * second)
    after_final = final_controller.budget_usage()
    assert after_final[0] == before_final[0]
    assert after_final[1] - before_final[1] == 2.0
    assert after_final[2] - before_final[2] == 0.5


def test_successful_evaluation_settles_elapsed_gpu_time_without_new_gradients(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_controller import ExperimentController

    second = 1_000_000_000
    controller = ExperimentController(tmp_path / "controller.sqlite3", gpu_price_per_second=0.25)
    job = _job(tmp_path, "evaluation-success")
    controller.add_jobs([job])
    training = controller.lease_next("trainer", "training", 0, second)
    assert training
    controller.complete(training, "c" * 64, 0)
    controller.publication_verified(job.experiment_id, _publication(job), 0)
    evaluation = controller.lease_next("evaluator", "evaluation", 0, 4 * second)
    assert evaluation
    controller.submit_evaluation(evaluation, _report(job), 3 * second)
    assert controller.budget_usage() == (500, 3.0, 0.75)


def test_successful_final_evaluation_settles_the_same_campaign_spend_ledger(tmp_path: Path) -> None:
    from test_experiment_winner import _report as _final_report

    second = 1_000_000_000
    controller, _initial, _seed, _one_k, job = _controller_authorized_two_k_finalist(
        tmp_path,
        controller_kwargs={"gpu_price_per_second": 0.5},
    )
    before_final = controller.budget_usage()
    controller.enqueue_finalists([job.experiment_id], matrix_sha256="f" * 64, now_ns=0)
    final = controller.lease_next("evaluator", "final_evaluation", 0, 4 * second)
    assert final
    report = _final_report(
        candidate=job.experiment_id,
        experiment_id=job.experiment_id,
        receipt="c" * 64,
        policy="d" * 64,
        matrix="f" * 64,
    )
    controller.submit_final_evaluation(final, report, 3 * second)
    after_final = controller.budget_usage()
    assert after_final[0] == before_final[0]
    assert after_final[1] - before_final[1] == 3.0
    assert after_final[2] - before_final[2] == 1.5


def test_evaluation_reservation_survives_controller_restart_and_expires_once(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_controller import ExperimentController

    second = 1_000_000_000
    database = tmp_path / "controller.sqlite3"
    controller = ExperimentController(database, gpu_price_per_second=0.5)
    job = _job(tmp_path, "evaluation-restart")
    controller.add_jobs([job])
    training = controller.lease_next("trainer", "training", 0, second)
    assert training
    controller.complete(training, "c" * 64, 0)
    controller.publication_verified(job.experiment_id, _publication(job), 0)
    assert controller.lease_next("evaluator", "evaluation", 0, second)
    controller._connection.close()

    restarted = ExperimentController(database, gpu_price_per_second=0.5)
    restarted.capacity_snapshot(now_ns=2 * second)
    assert restarted.state(job.experiment_id) == "EVAL_RETRYABLE"
    assert restarted.budget_usage() == (500, 2.0, 1.0)
    restarted.capacity_snapshot(now_ns=3 * second)
    assert restarted.budget_usage() == (500, 2.0, 1.0)


def test_safety_is_rejected_from_promotion(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_controller import ExperimentController

    controller = ExperimentController(tmp_path / "controller.sqlite3")
    safety = _job(tmp_path, "safety")
    controller.add_jobs([safety])
    train = controller.lease_next("train-safety", "training", 1, 100)
    assert train
    controller.complete(train, "c" * 64, 2)
    controller.publication_verified(safety.experiment_id, _publication(safety), 3)
    evaluation = controller.lease_next("eval", "evaluation", 4, 100)
    assert evaluation
    report = _report(safety); report["promotion_metrics"]["safety_failure"] = True
    report["report_sha256"] = hashlib.sha256(json.dumps({key: value for key, value in report.items() if key != "report_sha256"}, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()
    controller.submit_evaluation(evaluation, report, 5)
    assert controller.state(safety.experiment_id) == "REJECTED"


def test_seed_repeat_reuses_the_original_12k_parent_with_a_new_seed(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_controller import ExperimentController

    controller = ExperimentController(tmp_path / "controller.sqlite3")
    parent, peer = _job(tmp_path, "parent"), _job(tmp_path, "peer")
    controller.add_jobs([parent, peer])
    train = controller.lease_next("train", "training", 1, 100); assert train
    controller.complete(train, "c" * 64, 2); controller.publication_verified(parent.experiment_id, _publication(parent), 3)
    eval_lease = controller.lease_next("eval", "evaluation", 4, 100); assert eval_lease
    controller.submit_evaluation(eval_lease, _report(parent), 5)
    peer_lease = controller.lease_next("peer", "training", 6, 100); assert peer_lease
    controller.complete(peer_lease, "c" * 64, 7); controller.publication_verified(peer.experiment_id, _publication(peer), 8)
    peer_eval = controller.lease_next("eval", "evaluation", 9, 100); assert peer_eval
    peer_report = _report(peer); peer_report["promotion_metrics"]["paired_improvement"] = -1.0; _rehash_report(peer_report)
    controller.submit_evaluation(peer_eval, peer_report, 10)

    assert controller.state(parent.experiment_id) == "PROMOTED"
    controller.lease_next("continuation", "training", 7, 100)
    repeat_lease = controller.lease_next("seed-trainer", "training", 7, 100)
    assert repeat_lease and repeat_lease.job.admission["kind"] == "seed_repeat"
    assert repeat_lease.parent_publication is None
    assert dict(repeat_lease.job.parent_checkpoint) == dict(parent.parent_checkpoint)


@pytest.mark.parametrize(
    ("repeat_a", "repeat_b", "expected_sources"),
    (
        (-1.0, 0.0, {"a", "b"}),
        (0.0, -1.0, {"a"}),
    ),
    ids=("out-of-order-ranking-reversal-retains-both", "stable-ranking-keeps-primary-only"),
)
def test_seed_repeats_gate_one_k_admission_with_a_deterministic_ranking_check(
    tmp_path: Path,
    repeat_a: float,
    repeat_b: float,
    expected_sources: set[str],
) -> None:
    """The repeat pair is independent evidence, never a decorative side run.

    Initial ``a`` outranks ``b``.  The repeat reports arrive in the reverse
    order.  A flipped repeat ranking retains both source configurations for
    1K; a stable ranking admits only the original leader.  In either case the
    controller must wait for *both* repeat reports before materializing 1K.
    """
    from lehome_train.groot.experiment_controller import ExperimentController

    controller = ExperimentController(tmp_path / "controller.sqlite3")
    initial = {name: _job(tmp_path, name) for name in ("a", "b")}
    controller.add_jobs(list(initial.values()))

    def publish_and_score(lease, now_ns: int, improvement: float) -> None:
        controller.complete(lease, "c" * 64, now_ns)
        controller.publication_verified(lease.experiment_id, _publication(lease.job), now_ns + 1)
        evaluation = controller.lease_next("evaluator-" + str(now_ns), "evaluation", now_ns + 2, 100)
        assert evaluation is not None and evaluation.experiment_id == lease.experiment_id
        report = _report(lease.job)
        report["promotion_metrics"]["paired_improvement"] = improvement
        _rehash_report(report)
        controller.submit_evaluation(evaluation, report, now_ns + 3)

    first = controller.lease_next("initial-a", "training", 1, 100)
    second = controller.lease_next("initial-b", "training", 1, 100)
    assert first is not None and second is not None
    by_id = {lease.experiment_id: lease for lease in (first, second)}
    publish_and_score(by_id[initial["a"].experiment_id], 10, 0.0)
    publish_and_score(by_id[initial["b"].experiment_id], 20, -1.0)

    # Two second seeds, and no premature 1K continuation, are required once
    # the first ranking closure identifies the comparison pair.
    children = [
        controller._job(str(row[0]))
        for row in controller._connection.execute("SELECT experiment_id FROM promotion_children")
    ]
    repeats = {child.admission["source_experiment_id"]: child for child in children if child.admission["kind"] == "seed_repeat"}
    assert set(repeats) == {initial["a"].experiment_id, initial["b"].experiment_id}
    assert not [child for child in children if child.training.target_step == 1000]

    # Lease both repeat jobs but report B first.  One repeat result is not a
    # ranking decision and must not create a 1K child.
    repeat_first = controller.lease_next("repeat-a", "training", 30, 100)
    repeat_second = controller.lease_next("repeat-b", "training", 30, 100)
    assert repeat_first is not None and repeat_second is not None
    repeat_leases = {lease.job.admission["source_experiment_id"]: lease for lease in (repeat_first, repeat_second)}
    publish_and_score(repeat_leases[initial["b"].experiment_id], 40, repeat_b)
    assert not [
        controller._job(str(row[0]))
        for row in controller._connection.execute("SELECT experiment_id FROM promotion_children")
        if controller._job(str(row[0])).training.target_step == 1000
    ]
    publish_and_score(repeat_leases[initial["a"].experiment_id], 50, repeat_a)

    one_k_sources = {
        controller._job(str(experiment_id)).admission["source_experiment_id"]
        for (experiment_id,) in controller._connection.execute("SELECT experiment_id FROM promotion_children")
        if controller._job(str(experiment_id)).training.target_step == 1000
    }
    assert one_k_sources == {initial[name].experiment_id for name in expected_sources}


def test_last_safety_rejected_seed_repeat_releases_unrelated_safe_one_k_candidates(
    tmp_path: Path,
) -> None:
    """A failed seed check cannot freeze unrelated arms behind the pair."""
    from lehome_train.groot.experiment_controller import ExperimentController

    controller = ExperimentController(tmp_path / "controller.sqlite3")
    initial = {name: _job(tmp_path, name) for name in ("a", "b", "c", "d")}
    controller.add_jobs(list(initial.values()))

    def publish(lease, now_ns: int) -> None:
        controller.complete(lease, "c" * 64, now_ns)
        controller.publication_verified(lease.experiment_id, _publication(lease.job), now_ns + 1)

    def evaluate(lease, now_ns: int, *, safety_failure: bool = False) -> None:
        report = _report(lease.job)
        if safety_failure:
            report["promotion_metrics"]["safety_failure"] = True
        _rehash_report(report)
        controller.submit_evaluation(lease, report, now_ns)

    first = controller.lease_next("initial-a", "training", 1, 100)
    second = controller.lease_next("initial-b", "training", 1, 100)
    assert first is not None and second is not None
    publish(first, 2)
    evaluate(controller.lease_next("evaluator", "evaluation", 4, 100), 5)
    publish(second, 6)
    evaluate(controller.lease_next("evaluator", "evaluation", 8, 100), 9)

    repeats = [
        controller.lease_next("repeat-" + str(index), "training", 10, 100)
        for index in (1, 2)
    ]
    assert all(repeats)
    for index, repeat in enumerate(repeats, start=1):
        assert repeat is not None and repeat.job.admission["kind"] == "seed_repeat"
        # Keep seed outputs in PUBLISHING until unrelated initial reports
        # land; evaluator priority must not accidentally make this a wave.
        controller.complete(repeat, "c" * 64, 10 + index * 2)

    # Publish two unrelated safe arms while the seed pair waits for evaluator
    # evidence.  They must remain eligible if the final repeat is unsafe.
    for index, name in enumerate(("c", "d"), start=1):
        training = controller.lease_next("initial-" + name, "training", 20 + index, 100)
        assert training is not None and training.experiment_id == initial[name].experiment_id
        publish(training, 30 + index * 2)
    for now_ns in (40, 50):
        evaluation = controller.lease_next("evaluator", "evaluation", now_ns, 100)
        assert evaluation is not None and evaluation.experiment_id in {initial["c"].experiment_id, initial["d"].experiment_id}
        evaluate(evaluation, now_ns + 1)

    # The first repeat is safe; the second is the last terminal event and is
    # safety-rejected.  It must still reconcile the seed gate.
    assert repeats[0] is not None and repeats[1] is not None
    controller.publication_verified(repeats[0].experiment_id, _publication(repeats[0].job), 59)
    first_repeat_evaluation = controller.lease_next("evaluator", "evaluation", 60, 100)
    assert first_repeat_evaluation is not None and first_repeat_evaluation.job.admission["kind"] == "seed_repeat"
    evaluate(first_repeat_evaluation, 61)
    controller.publication_verified(repeats[1].experiment_id, _publication(repeats[1].job), 69)
    last_repeat_evaluation = controller.lease_next("evaluator", "evaluation", 70, 100)
    assert last_repeat_evaluation is not None and last_repeat_evaluation.job.admission["kind"] == "seed_repeat"
    evaluate(last_repeat_evaluation, 71, safety_failure=True)
    assert controller.state(last_repeat_evaluation.experiment_id) == "REJECTED"

    one_k_sources = {
        controller._job(str(experiment_id)).admission["source_experiment_id"]
        for (experiment_id,) in controller._connection.execute("SELECT experiment_id FROM promotion_children")
        if controller._job(str(experiment_id)).training.target_step == 1000
    }
    assert one_k_sources == {initial["c"].experiment_id, initial["d"].experiment_id}


def test_evaluation_and_automatic_child_materialization_are_one_transaction(tmp_path: Path) -> None:
    """A materializer exception leaves the evaluator lease unconsumed.

    This is the crash/error equivalent of the controller's restart repair: no
    committed evaluation may leave only a PENDING child behind for an operator
    to discover later.  Retrying the exact live lease after the fault performs
    the whole transition once.
    """
    from lehome_train.groot.experiment_controller import ExperimentController

    controller = ExperimentController(tmp_path / "controller.sqlite3")
    parent, peer = _job(tmp_path, "parent"), _job(tmp_path, "peer")
    controller.add_jobs([parent, peer])
    first = controller.lease_next("train-a", "training", 1, 100)
    assert first
    controller.complete(first, "c" * 64, 2)
    controller.publication_verified(first.experiment_id, _publication(parent), 3)
    first_eval = controller.lease_next("eval", "evaluation", 4, 100)
    assert first_eval
    controller.submit_evaluation(first_eval, _report(parent), 5)
    second = controller.lease_next("train-b", "training", 6, 100)
    assert second and second.experiment_id == peer.experiment_id
    controller.complete(second, "c" * 64, 7)
    controller.publication_verified(peer.experiment_id, _publication(peer), 8)
    second_eval = controller.lease_next("eval", "evaluation", 9, 100)
    assert second_eval

    original_promote = controller.promote

    def fail_materialization(*_args, **_kwargs):
        raise RuntimeError("injected materializer failure")

    controller.promote = fail_materialization  # type: ignore[method-assign]
    try:
        controller.submit_evaluation(second_eval, _report(peer), 10)
    except RuntimeError as error:
        assert str(error) == "injected materializer failure"
    else:
        raise AssertionError("materializer failure consumed the evaluator lease")
    assert controller.state(peer.experiment_id) == "LEASED"
    assert controller.pending_promotions(parent.experiment_id) == ()
    assert controller.pending_promotions(peer.experiment_id) == ()
    assert controller.lease_for(second_eval.lease_id, peer.experiment_id, "eval", now_ns=10).lease_id == second_eval.lease_id

    controller.promote = original_promote  # type: ignore[method-assign]
    controller.submit_evaluation(second_eval, _report(peer), 11)
    assert controller.state(parent.experiment_id) == "PROMOTED"
