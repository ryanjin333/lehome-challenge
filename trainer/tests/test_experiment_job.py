"""Immutable job contract regression tests."""

from __future__ import annotations

import json
from pathlib import Path


SHA = "a" * 64
REV = "b" * 40


def _document() -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_id": "",
        "arm": "a",
        "parent_checkpoint": {"repository": "owner/models", "revision": REV, "subpath": "policies/step-12000", "artifact_sha256": SHA},
        "trainer": {"image_id": "image", "oci_digest": "sha256:" + SHA, "code_revision": REV},
        "data_sources": [{"kind": "bc", "repository": "owner/data", "revision": REV, "prefix": "bc/full", "manifest_sha256": SHA, "tree_sha256": SHA}],
        "mixture": {"bc_percent": 100, "added_percent": 0, "batch64_quotas": {"bc": 64, "rollout": 0, "dagger": 0}, "sampling_strategy": "unweighted"},
        "training": {"action_horizon": 16, "batch_size": 64, "seed": 1, "target_step": 500, "save_steps": 500},
        "evaluation": {"matrix_id": "unseen20", "matrix_sha256": SHA, "policy_digest": SHA},
        "publication": {"checkpoint_repository": "owner/checkpoints", "result_repository": "owner/results", "prefix": "experiments/a"},
        "dependencies": [],
    }


def test_job_id_hashes_canonical_document_except_declared_id(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_job import experiment_identity, load_experiment_job

    document = _document()
    document["experiment_id"] = experiment_identity(document)
    path = tmp_path / "job.json"
    path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    assert load_experiment_job(path).experiment_id == document["experiment_id"]


def test_job_rejects_secrets_and_unsafe_publication_prefix(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_job import experiment_identity, load_experiment_job

    document = _document()
    document["publication"] = {"checkpoint_repository": "owner/checkpoints", "result_repository": "owner/results", "prefix": "../unsafe"}
    document["experiment_id"] = experiment_identity(document)
    path = tmp_path / "job.json"
    path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    try:
        load_experiment_job(path)
    except ValueError as error:
        assert "prefix" in str(error)
    else:
        raise AssertionError("unsafe manifest was accepted")


def test_initial_and_seed_repeat_jobs_bind_baseline_policy_to_original_parent(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_job import experiment_identity, load_experiment_job

    for kind in ("initial", "seed_repeat"):
        document = _document()
        document["evaluation"]["policy_digest"] = "c" * 64
        if kind == "seed_repeat":
            document["admission"] = {"kind": kind, "source_experiment_id": "d" * 64}
        document["experiment_id"] = experiment_identity(document)
        path = tmp_path / f"{kind}.json"
        path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")), encoding="utf-8")

        try:
            load_experiment_job(path)
        except ValueError as error:
            assert "baseline" in str(error) and "parent" in str(error)
        else:
            raise AssertionError(f"{kind} job accepted a different baseline policy")
