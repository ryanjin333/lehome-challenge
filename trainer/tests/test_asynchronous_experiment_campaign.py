"""CPU-only campaign invariant: two trainers never create a third training lane."""
from pathlib import Path
import json
import hashlib
from test_experiment_controller import _job


def _continuation(tmp_path: Path, parent, *, name: str, target_step: int):
    from lehome_train.groot.experiment_job import dump_experiment_job
    from test_experiment_job import _document

    document = _document()
    document["arm"] = name
    document["training"]["target_step"] = target_step  # type: ignore[index]
    document["publication"]["prefix"] = "experiments/" + name  # type: ignore[index]
    document["dependencies"] = [parent.experiment_id]
    document["admission"] = {"kind": "continuation", "source_experiment_id": parent.experiment_id}
    document["parent_checkpoint"] = {
        "repository": parent.publication.checkpoint_repository,
        "revision": "a" * 40,
        "subpath": parent.publication.prefix + "/step-" + str(parent.training.target_step),
        "artifact_sha256": "d" * 64,
        "receipt_sha256": "c" * 64,
    }
    return dump_experiment_job(tmp_path / (name + ".json"), document)

def test_campaign_leases_only_two_training_jobs_at_once(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_controller import ExperimentController
    controller = ExperimentController(tmp_path / "campaign.sqlite3", max_gpu_leases=3)
    controller.add_jobs([_job(tmp_path, name) for name in ("a", "b", "c", "d")])
    leases = [controller.lease_next("train-" + str(index), "training", index, 100) for index in range(3)]
    assert sum(lease is not None for lease in leases) == 2


def test_out_of_order_500_results_admit_a_1k_child_without_waiting_for_the_other_trainer(tmp_path: Path) -> None:
    """ASHA makes the next rung available as soon as its own result is verified."""
    from lehome_train.groot.experiment_controller import ExperimentController
    from test_experiment_controller import _publication, _report

    controller = ExperimentController(tmp_path / "campaign.sqlite3")
    first, second, third = (_job(tmp_path, name) for name in ("a", "b", "c"))
    controller.add_jobs([first, second, third])
    first_lease = controller.lease_next("trainer-1", "training", 1, 100)
    second_lease = controller.lease_next("trainer-2", "training", 1, 100)
    assert first_lease and second_lease

    # Whichever job finishes first gets evaluated and creates an admission while
    # the other trainer is still running.  No global seven-arm wave exists.
    completed = second if second_lease.experiment_id == second.experiment_id else first
    completed_lease = second_lease if completed is second else first_lease
    controller.complete(completed_lease, "c" * 64, 2)
    controller.publication_verified(completed.experiment_id, _publication(completed), 3)
    evaluation = controller.lease_next("evaluator", "evaluation", 4, 100)
    assert evaluation and evaluation.experiment_id == completed.experiment_id
    controller.submit_evaluation(evaluation, _report(completed), 5)
    assert controller.pending_promotions(completed.experiment_id) == ()
    assert controller.state(first_lease.experiment_id if completed_lease is second_lease else second_lease.experiment_id) == "LEASED"

    next_lease = controller.lease_next("trainer-2", "training", 7, 100)
    assert next_lease and next_lease.experiment_id == third.experiment_id


def _terminal_receipt(controller, lease, now_ns: int) -> str:
    """Complete one leased run with a distinct immutable policy artifact."""
    assert lease and lease.job
    digest = hashlib.sha256(lease.experiment_id.encode("ascii")).hexdigest()
    controller.complete(lease, "c" * 64, now_ns)
    return digest


def _publish(controller, job, digest: str, now_ns: int) -> None:
    """Readback verification is deliberately a separate controller transition."""
    from test_experiment_controller import _publication

    publication = _publication(job)
    publication["artifact_sha256"] = digest
    controller.publication_verified(job.experiment_id, publication, now_ns)


def _published(controller, lease, now_ns: int) -> str:
    digest = _terminal_receipt(controller, lease, now_ns)
    _publish(controller, lease.job, digest, now_ns + 1)
    return digest


def _scored_report(job, digest: str, paired_improvement: float):
    """Keep the strict pair witness coherent while controlling only rank."""
    from test_experiment_controller import _rehash_report, _report

    report = _report(job)
    report["policy_digest"] = digest
    delta = int(round(paired_improvement * 20))
    assert delta % 2 == 0 and -20 <= delta <= 20
    pairing = report["promotion_metrics"]["pairing"]
    pairing.update(
        {
            "candidate_wins": (20 + delta) // 2,
            "baseline_wins": (20 - delta) // 2,
            "ties": 0,
            "paired_improvement": paired_improvement,
        }
    )
    report["promotion_metrics"]["paired_improvement"] = paired_improvement
    _rehash_report(report)
    return report


def _one_k_child_parents(controller) -> list[str]:
    rows = controller._connection.execute(  # controller-owned durable state
        "SELECT pc.experiment_id,pc.parent_experiment_id FROM promotion_children pc "
        "JOIN jobs j ON j.experiment_id=pc.experiment_id ORDER BY j.created_order"
    ).fetchall()
    return [
        str(parent_id)
        for experiment_id, parent_id in rows
        if controller._job(str(experiment_id)).training.target_step == 1000
    ]


def _child_ids_at_rung(controller, target_step: int) -> list[str]:
    return [
        str(experiment_id)
        for (experiment_id,) in controller._connection.execute("SELECT experiment_id FROM promotion_children ORDER BY experiment_id").fetchall()
        if controller._job(str(experiment_id)).training.target_step == target_step
    ]


def test_seven_initial_results_autonomously_materialize_ranked_children_and_close_2k_rung(tmp_path: Path) -> None:
    """A restart-safe ASHA campaign does not retain completion-order rankings.

    Seven 500 jobs finish before their single evaluator reports them.  The
    reports are deliberately out of score order: each 2/4/6 local closure
    admits one additional highest *unpromoted* 1K parent, while a stronger
    seventh result cannot rewrite an immutable earlier admission.  This is a
    CPU-only proof that no operator has to call ``promote``.
    """
    from lehome_train.groot.experiment_controller import ExperimentController

    controller = ExperimentController(tmp_path / "campaign.sqlite3")
    jobs = {name: _job(tmp_path, name) for name in "abcdefg"}
    # The manifest order controls evaluator FIFO; workers complete b..g while
    # a is still leased, which makes the report order intentionally out of
    # training completion order.
    initial = [jobs[name] for name in ("b", "a", "c", "d", "e", "f", "g")]
    controller.add_jobs(initial)
    first = controller.lease_next("trainer-a", "training", 1, 10_000)
    second = controller.lease_next("trainer-b", "training", 1, 10_000)
    assert first and second and first.experiment_id == jobs["b"].experiment_id and second.experiment_id == jobs["a"].experiment_id
    initial_digests = {"b": _published(controller, first, 2)}
    now = 10
    for name in ("c", "d", "e", "f", "g"):
        lease = controller.lease_next("trainer-a", "training", now, 10_000)
        assert lease and lease.experiment_id == jobs[name].experiment_id
        # Keep the future evaluations in PUBLISHING until after the first
        # local closure.  That is a real backpressure condition, not a hidden
        # scheduling bypass: evaluator backlog >2 correctly pauses trainers.
        initial_digests[name] = _terminal_receipt(controller, lease, now + 1)
        now += 10
    initial_digests["a"] = _published(controller, second, now + 1)

    # The same eval worker reports the immutable 500 parents in manifest FIFO.
    # At 2 scores a wins; at 4 a late stronger c receives the next slot; at 6
    # e receives the third.  g is stronger still but cannot mutate admissions.
    paired = {"b": -0.6, "a": -0.4, "c": 0.8, "d": 0.6, "e": 1.0, "f": 0.4, "g": 0.9}
    report_time = 100
    for index, name in enumerate(("b", "a", "c", "d", "e", "f", "g"), start=1):
        if name not in {"b", "a"}:
            _publish(controller, jobs[name], initial_digests[name], report_time - 1)
        evaluation = controller.lease_next("evaluator", "evaluation", report_time, 10_000)
        assert evaluation and evaluation.experiment_id == jobs[name].experiment_id
        controller.submit_evaluation(evaluation, _scored_report(jobs[name], initial_digests[name], paired[name]), report_time + 1)
        report_time += 10
        if index == 2:
            # The first pair launches two independent seeds before one 1K
            # continuation is admitted.  Their reports preserve A's lead, so
            # the stable seed gate releases only A without waiting for C-G.
            repeats = [
                controller.lease_next("seed-trainer-" + str(seed_index), "training", report_time, 10_000)
                for seed_index in (1, 2)
            ]
            assert all(repeats)
            repeat_digests = {}
            for seed in repeats:
                assert seed and seed.job and seed.job.admission["kind"] == "seed_repeat"
                repeat_digests[seed.job.admission["source_experiment_id"]] = _published(controller, seed, report_time + 1)
            for source_id, seed_digest in repeat_digests.items():
                repeat = next(seed for seed in repeats if seed and seed.job and seed.job.admission["source_experiment_id"] == source_id)
                evaluation = controller.lease_next("seed-evaluator-" + source_id[:6], "evaluation", report_time + 2, 10_000)
                assert evaluation and evaluation.experiment_id == repeat.experiment_id
                source_name = "a" if source_id == jobs["a"].experiment_id else "b"
                controller.submit_evaluation(evaluation, _scored_report(repeat.job, seed_digest, paired[source_name]), report_time + 3)
                report_time += 4
            assert _one_k_child_parents(controller) == [jobs["a"].experiment_id]
            one_k = controller.lease_next("trainer-a", "training", report_time, 10_000)
            assert one_k and one_k.job and one_k.job.training.target_step == 1000
            assert one_k.parent_publication and one_k.parent_publication["artifact_sha256"] == initial_digests["a"]
        elif index == 4:
            assert _one_k_child_parents(controller) == [jobs["a"].experiment_id, jobs["c"].experiment_id]
        elif index == 6:
            assert _one_k_child_parents(controller) == [jobs["a"].experiment_id, jobs["c"].experiment_id, jobs["e"].experiment_id]
        elif index == 7:
            assert _one_k_child_parents(controller) == [jobs["a"].experiment_id, jobs["c"].experiment_id, jobs["e"].experiment_id]

    # Materialization is part of the same transaction as the evaluation.  A
    # restart can re-run reconciliation but cannot create a duplicate child.
    children_before_restart = _child_ids_at_rung(controller, 1000)
    controller.close()
    controller = ExperimentController(tmp_path / "campaign.sqlite3")
    controller.add_jobs(initial)
    controller.reconcile_pending_candidates(report_time + 1)
    assert _child_ids_at_rung(controller, 1000) == children_before_restart

    # a's continuation was leased at the first closure; c/e now use the one
    # free training lane beside the deliberately independent seed-repeat.
    one_k_digests: dict[str, str] = {}
    one_k_digests["a"] = _published(controller, one_k, report_time + 2)
    for expected in ("c", "e"):
        lease = controller.lease_next("trainer-a", "training", report_time + 3, 10_000)
        assert lease and lease.job and lease.job.training.target_step == 1000
        assert lease.job.admission["source_experiment_id"] == jobs[expected].experiment_id
        one_k_digests[expected] = _published(controller, lease, report_time + 4)

    # No 2K candidate appears until every controller-selected 1K run is both
    # terminal and evaluated.  The first two reports must leave the 2K rung
    # empty; the third autonomously creates the exact-rule finalist(s).
    one_k_jobs = {
        controller._job(experiment_id).admission["source_experiment_id"]: controller._job(experiment_id)
        for experiment_id in _child_ids_at_rung(controller, 1000)
    }
    for index, name in enumerate(("a", "c", "e"), start=1):
        child = one_k_jobs[jobs[name].experiment_id]
        evaluation = controller.lease_next("evaluator", "evaluation", report_time + 10 + index, 10_000)
        assert evaluation and evaluation.experiment_id == child.experiment_id
        controller.submit_evaluation(evaluation, _scored_report(child, one_k_digests[name], 0.0), report_time + 20 + index)
        if index < 3:
            assert _child_ids_at_rung(controller, 2000) == []
    finalists = _child_ids_at_rung(controller, 2000)
    assert finalists
    final_lease = controller.lease_next("trainer-a", "training", report_time + 30, 10_000)
    assert final_lease and final_lease.job and final_lease.job.training.target_step == 2000
