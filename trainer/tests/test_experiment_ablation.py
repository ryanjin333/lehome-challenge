"""Matched AWR-style weighted-replay ablation tests."""

from __future__ import annotations

from copy import deepcopy

import pytest


def _publish_final(report):
    from lehome_train.groot.experiment_winner import publish_final_unseen80_report

    class Hub:
        payload: bytes | None = None

        def upload_bytes(self, repository: str, path: str, payload: bytes) -> None:
            assert repository == "owner/final-reports" and path == "finals/awr-parent.json"
            self.payload = payload

        def read_bytes(self, repository: str, path: str) -> bytes:
            assert repository == "owner/final-reports" and path == "finals/awr-parent.json"
            assert self.payload is not None
            return self.payload

    return publish_final_unseen80_report(report, transport=Hub(), repository="owner/final-reports", path="finals/awr-parent.json")


def _job():
    from lehome_train.groot.experiment_job import _parse, experiment_identity

    document: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": "",
        "arm": "recovery-d",
        "parent_checkpoint": {"repository": "owner/models", "revision": "b" * 40, "subpath": "policies/step-12000", "artifact_sha256": "a" * 64},
        "trainer": {"image_id": "image", "oci_digest": "sha256:" + "a" * 64, "code_revision": "b" * 40},
        "data_sources": [
            {"kind": "bc", "repository": "owner/data", "revision": "b" * 40, "prefix": "bc/full", "manifest_sha256": "a" * 64, "tree_sha256": "b" * 64},
            {"kind": "recovery", "repository": "owner/data", "revision": "b" * 40, "prefix": "recovery/v1", "manifest_sha256": "c" * 64, "tree_sha256": "d" * 64},
            {"kind": "runtime_request_set", "repository": "owner/data", "revision": "b" * 40, "prefix": "requests/unweighted", "manifest_sha256": "e" * 64, "tree_sha256": "f" * 64},
        ],
        "mixture": {"bc_percent": 95, "added_percent": 5, "batch64_quotas": {"bc": 61, "rollout": 3, "dagger": 0}, "sampling_strategy": "unweighted"},
        "training": {"action_horizon": 16, "batch_size": 64, "seed": 17, "target_step": 500, "save_steps": 500},
        "evaluation": {"matrix_id": "unseen20", "matrix_sha256": "1" * 64, "policy_digest": "a" * 64},
        "publication": {"checkpoint_repository": "owner/checkpoints", "result_repository": "owner/results", "prefix": "experiments/recovery-d"},
        "dependencies": ["3" * 64],
    }
    document["experiment_id"] = experiment_identity(document)
    return _parse(document)


def _winning_report(job):
    from lehome_train.groot.experiment_winner import seal_final_unseen80_report

    categories = {"top_long": 12, "top_short": 12, "pant_long": 12, "pant_short": 20}
    episodes = []
    for category, count in categories.items():
        for index in range(20):
            episodes.append({"trial_id": f"{category}-{index}", "category": category, "official_success": int(index < count), "artifact_sha256": "5" * 64, "receipt_sha256": "6" * 64, "readback_verified": True, "sealed": True})
    publication = {
        "schema_version": 2,
        "experiment_id": job.experiment_id,
        "job_digest": job.experiment_id,
        "target_step": job.training.target_step,
        "repository": "owner/checkpoints",
        "immutable_revision": "b" * 40,
        "remote_prefix": "experiments/recovery-d/step-500",
        "artifact_sha256": "7" * 64,
        "receipt_sha256": "8" * 64,
        "readback_verified": True,
        "relative_path": "checkpoint.tar",
        "artifact_byte_size": 1,
        "descriptor_relative_path": "checkpoint.json",
        "descriptor_sha256": "9" * 64,
        "descriptor_byte_size": 1,
    }
    return _publish_final(seal_final_unseen80_report({
        "schema_version": 2,
        "kind": "lehome_experiment_final_unseen80",
        "candidate_id": job.experiment_id,
        "experiment_id": job.experiment_id,
        "checkpoint_receipt_sha256": "8" * 64,
        "checkpoint_publication": publication,
        "matrix_sha256": "1" * 64,
        "policy_digest": "7" * 64,
        "categories": {category: {"successes": count, "episodes": 20} for category, count in categories.items()},
        "overall_successes": 56,
        "episode_artifacts": episodes,
        "safety_failure": False,
        "major_seen_regression": False,
    }))


def _pending(job):
    from lehome_train.groot.awr_weighting import AwrReplayConfig
    from lehome_train.groot.experiment_ablation import build_awr_style_ablation

    return build_awr_style_ablation(
        job,
        winning_unweighted_report=_winning_report(job),
        progress_evidence_sha256="4" * 64,
        progress_evidence_receipt=_evidence_receipt(),
        replay_config=AwrReplayConfig(temperature=0.75, minimum=0.5, maximum=3.0),
    )


def _evidence_receipt() -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "lehome_awr_progress_evidence_receipt",
        "evidence_sha256": "4" * 64,
        "mixture_id": "a" * 64,
        "mixture_manifest_sha256": "b" * 64,
        "authenticated_principal_sha256": "c" * 64,
        "readback_receipt_sha256": "d" * 64,
        "readback_verified": True,
    }


def _source() -> dict[str, str]:
    return {"kind": "runtime_request_set", "repository": "owner/data", "revision": "b" * 40, "prefix": "requests/awr-style", "manifest_sha256": "a" * 64, "tree_sha256": "c" * 64}


def _receipt(job, pending, source):
    from lehome_train.groot.experiment_ablation import matched_training_sha256, pending_admission_sha256
    from lehome_train.groot.experiment_runtime_request import runtime_profile_sha256
    from lehome_train.groot.experiment_job import _parse, experiment_identity

    child = deepcopy(dict(job.raw))
    child["data_sources"][2] = source
    child["mixture"]["sampling_strategy"] = "awr_style_weighted_replay"
    child["publication"]["prefix"] += "-awr-style"
    child["admission"] = {
        "kind": "awr_style_weighted_replay",
        "pending_admission_sha256": pending_admission_sha256(pending),
        "matched_training_sha256": matched_training_sha256(job),
        "progress_evidence_sha256": "4" * 64,
        "progress_evidence_receipt_sha256": __import__("lehome_train.groot.awr_weighting", fromlist=["authenticated_progress_evidence_receipt_sha256"]).authenticated_progress_evidence_receipt_sha256(_evidence_receipt()),
        "progress_evidence_mixture_id": "a" * 64,
        "progress_evidence_mixture_manifest_sha256": "b" * 64,
        "awr_replay_config_sha256": pending["awr_replay_config_sha256"],
        "winning_unweighted_experiment_id": job.experiment_id,
        "winning_unweighted_report_sha256": pending["winning_unweighted_report_sha256"],
        "winning_unweighted_seal_sha256": pending["winning_unweighted_seal_sha256"],
    }
    child.pop("experiment_id")
    child["experiment_id"] = experiment_identity(child)
    parsed = _parse(child)
    return {
        "schema_version": 1,
        "kind": "lehome_awr_style_weighted_request_set_receipt",
        "pending_admission_sha256": pending_admission_sha256(pending),
        "weighted_runtime_request_set": source,
        "child_runtime_profile_sha256": runtime_profile_sha256(parsed),
        "matched_training_sha256": matched_training_sha256(job),
        "readback_receipt_sha256": "d" * 64,
        "authenticated_principal_sha256": "e" * 64,
        "progress_evidence_sha256": "4" * 64,
        "progress_evidence_receipt_sha256": child["admission"]["progress_evidence_receipt_sha256"],
        "progress_evidence_mixture_id": "a" * 64,
        "progress_evidence_mixture_manifest_sha256": "b" * 64,
        "awr_replay_config_sha256": pending["awr_replay_config_sha256"],
        "winning_unweighted_experiment_id": job.experiment_id,
        "winning_unweighted_report_sha256": pending["winning_unweighted_report_sha256"],
        "winning_unweighted_seal_sha256": pending["winning_unweighted_seal_sha256"],
        "readback_verified": True,
    }


def test_ablation_is_pending_until_a_distinct_weighted_request_set_has_authenticated_readback() -> None:
    from lehome_train.groot.experiment_ablation import bind_weighted_runtime_request_set, pending_admission_sha256
    from lehome_train.groot.experiment_job import _parse

    job = _job()
    pending = _pending(job)
    assert pending["kind"] == "lehome_awr_style_pending_materialization"
    assert pending["controller_admission_contract"]["lease_state_before_receipt"] == "PENDING_MATERIALIZATION"  # type: ignore[index]
    assert len(pending_admission_sha256(pending)) == 64
    with pytest.raises(ValueError):
        _parse(pending)

    source = _source()
    admission = bind_weighted_runtime_request_set(job, pending, weighted_runtime_request_set=source, materialization_receipt=_receipt(job, pending, source))
    assert admission.job.mixture.sampling_strategy == "awr_style_weighted_replay"
    assert admission.job.parent_checkpoint == job.parent_checkpoint
    assert admission.job.training == job.training
    assert admission.job.evaluation == job.evaluation
    assert admission.job.publication.prefix.endswith("-awr-style")


def test_ablation_rejects_same_bundle_and_tampered_or_unrunnable_request_binding() -> None:
    from lehome_train.groot.experiment_ablation import bind_weighted_runtime_request_set

    job = _job()
    pending = _pending(job)
    reused = dict(job.raw["data_sources"][2])
    with pytest.raises(ValueError, match="may not reuse"):
        bind_weighted_runtime_request_set(job, pending, weighted_runtime_request_set=reused, materialization_receipt={})

    source = _source()
    receipt = _receipt(job, pending, source)
    receipt["child_runtime_profile_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not bind"):
        bind_weighted_runtime_request_set(job, pending, weighted_runtime_request_set=source, materialization_receipt=receipt)


def test_ablation_requires_verified_winning_unweighted_recovery_parent() -> None:
    from lehome_train.groot.awr_weighting import AwrReplayConfig
    from lehome_train.groot.experiment_ablation import build_awr_style_ablation

    job = _job()
    report = _winning_report(job)
    report["safety_failure"] = True
    from lehome_train.groot.experiment_winner import seal_final_unseen80_report
    report = _publish_final(seal_final_unseen80_report(report))
    with pytest.raises(ValueError, match="verified winning"):
        build_awr_style_ablation(job, winning_unweighted_report=report, progress_evidence_sha256="4" * 64, progress_evidence_receipt=_evidence_receipt(), replay_config=AwrReplayConfig(temperature=1, minimum=0.5, maximum=2))


def test_cpu_e2e_winning_report_to_weighted_materialization_receipt_to_leaseable_job(tmp_path) -> None:
    """No GPU: receipt admission is the only path out of pending materialization."""
    from lehome_train.groot.experiment_ablation import bind_weighted_runtime_request_set
    from lehome_train.groot.experiment_controller import ExperimentController

    parent = _job()
    pending = _pending(parent)
    source = _source()
    admission = bind_weighted_runtime_request_set(
        parent,
        pending,
        weighted_runtime_request_set=source,
        materialization_receipt=_receipt(parent, pending, source),
    )
    controller = ExperimentController(tmp_path / "controller.sqlite3")
    controller.add_jobs([admission.job])
    assert controller.state(admission.job.experiment_id) == "PENDING_MATERIALIZATION"
    assert controller.lease_next("trainer", "training", now_ns=1, lease_ns=100) is None

    # Generic recovery proof cannot accidentally unlock the weighted ablation.
    recovery = {
        "schema_version": 1,
        "kind": "verified_recovery_dependency",
        "source": {"repository": "owner/data", "revision": "b" * 40, "prefix": "recovery/v1", "manifest_sha256": "c" * 64, "tree_sha256": "d" * 64},
        "readback_verified": True,
        "trajectories": {category: [f"{category}-{index}" for index in range(5)] for category in ("top_long", "top_short", "pant_long", "pant_short")},
    }
    assert controller.satisfy_dependency(recovery, 2) == 0
    assert controller.state(admission.job.experiment_id) == "PENDING_MATERIALIZATION"

    assert controller.satisfy_awr_style_admission(admission.job.experiment_id, admission.receipt, 3) == admission.receipt_sha256
    assert controller.state(admission.job.experiment_id) == "READY"
    lease = controller.lease_next("trainer", "training", now_ns=4, lease_ns=100)
    assert lease and lease.experiment_id == admission.job.experiment_id
