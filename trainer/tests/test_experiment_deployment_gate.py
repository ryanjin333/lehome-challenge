"""Immutable paid-capacity admission gate for the three-GPU sweep."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest


def _document() -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "lehome_experiment_pool_deployment_gate",
        "training_oci_digest": "sha256:" + "a" * 64,
        "training_code_revision": "b" * 40,
        "controller": {
            "instance_id": "computeinstance-controller1",
            "image_id": "computeimage-controller1",
            "image_status": "READY",
            "readback_verified": True,
        },
        "training_workers": [
            {
                "slot": 1,
                "worker_id": "lehome-experiment-training-1",
                "instance_id": "computeinstance-training1",
                "image_id": "computeimage-training1",
                "image_status": "READY",
                "readback_verified": True,
            },
            {
                "slot": 2,
                "worker_id": "lehome-experiment-training-2",
                "instance_id": "computeinstance-training2",
                "image_id": "computeimage-training1",
                "image_status": "READY",
                "readback_verified": True,
            },
        ],
        "rollout_worker": {
            "worker_id": "lehome-experiment-evaluator",
            "instance_id": "computeinstance-rollout1",
            "image_id": "computeimage-rollout1",
            "image_status": "READY",
            "readback_verified": True,
        },
        "teacher_probe": {
            "kind": "zero_perturbation_teacher_continuation_probe_v1",
            "attempt_id": "a" * 64,
            "round_id": "controlled-recovery-smoke-final12-unsealed-staging",
            "matrix_sha256": "b" * 64,
            "materialization_sha256": "c" * 64,
            "episode_sha256": "d" * 64,
            "sync_receipt_sha256": "e" * 64,
            "immutable_revision": "f" * 40,
            "accepted": True,
            "readback_verified": True,
            "strict_seal_present": False,
        },
    }


def _write(path: Path, document: dict[str, object]) -> str:
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    path.write_bytes(payload)
    path.chmod(0o444)
    return hashlib.sha256(payload).hexdigest()


def _training_image_manifest(path: Path, *, digest: str = "a" * 64) -> None:
    path.write_bytes(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "vla-training-base-image",
                "oci_image": "ghcr.io/owner/trainer@sha256:" + digest,
                "oci_digest": "sha256:" + digest,
                "trainer_code_revision": "b" * 40,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    path.chmod(0o444)


def test_deployment_gate_binds_exact_images_instances_and_accepted_teacher_probe(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_deployment_gate import load_deployment_gate

    path = tmp_path / "deployment-gate.json"
    digest = _write(path, _document())
    gate = load_deployment_gate(path, digest)

    assert gate.sha256 == digest
    assert gate.controller_instance_id == "computeinstance-controller1"
    assert gate.training_instance_ids == ("computeinstance-training1", "computeinstance-training2")
    assert gate.rollout_instance_id == "computeinstance-rollout1"
    assert gate.training_oci_digest == "sha256:" + "a" * 64
    assert gate.training_code_revision == "b" * 40
    assert gate.teacher_probe_receipt_sha256 == "e" * 64


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["teacher_probe"].update(accepted=False),
        lambda value: value["teacher_probe"].update(readback_verified=False),
        lambda value: value["teacher_probe"].update(strict_seal_present=True),
        lambda value: value["teacher_probe"].update(kind="bounded_perturbation_v1"),
        lambda value: value["training_workers"][1].update(slot=1),
        lambda value: value["training_workers"][1].update(instance_id="computeinstance-training1"),
        lambda value: value["rollout_worker"].update(image_status="CREATING"),
        lambda value: value["rollout_worker"].update(image_id="computeimage-training1"),
        lambda value: value["controller"].update(readback_verified=False),
        lambda value: value.pop("training_oci_digest"),
        lambda value: value.pop("training_code_revision"),
    ),
)
def test_deployment_gate_rejects_unverified_or_nonteacher_admission(tmp_path: Path, mutation) -> None:
    from lehome_train.groot.experiment_deployment_gate import load_deployment_gate

    document = deepcopy(_document())
    mutation(document)
    path = tmp_path / "deployment-gate.json"
    digest = _write(path, document)
    with pytest.raises(ValueError, match="deployment gate"):
        load_deployment_gate(path, digest)


def test_deployment_gate_rejects_byte_or_digest_tampering(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_deployment_gate import load_deployment_gate

    path = tmp_path / "deployment-gate.json"
    digest = _write(path, _document())
    with pytest.raises(ValueError, match="SHA-256"):
        load_deployment_gate(path, "0" * 64)
    path.chmod(0o644)
    with pytest.raises(ValueError, match="immutable"):
        load_deployment_gate(path, digest)


def test_paid_training_identity_requires_the_gate_image_and_baked_oci_manifest(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_deployment_gate import (
        bind_training_job_identity,
        load_deployment_gate,
        load_training_image_manifest,
    )
    from test_experiment_job import _document as job_document_template
    from lehome_train.groot.experiment_job import dump_experiment_job

    gate_path = tmp_path / "deployment-gate.json"
    gate_digest = _write(gate_path, _document())
    manifest_path = tmp_path / "training-image-manifest.json"
    _training_image_manifest(manifest_path)
    job_document = job_document_template()
    job_document["trainer"] = {
        "image_id": "computeimage-training1",
        "oci_digest": "sha256:" + "a" * 64,
        "code_revision": "b" * 40,
    }
    job = dump_experiment_job(tmp_path / "job.json", job_document)

    bind_training_job_identity(
        job,
        load_deployment_gate(gate_path, gate_digest),
        load_training_image_manifest(manifest_path),
    )

    for field, value in (
        ("image_id", "computeimage-unapproved"),
        ("oci_digest", "sha256:" + "c" * 64),
        ("code_revision", "c" * 40),
    ):
        mismatched = job_document_template()
        mismatched["trainer"] = dict(job_document["trainer"])
        mismatched["trainer"][field] = value
        bad_job = dump_experiment_job(tmp_path / (field + ".json"), mismatched)
        with pytest.raises(ValueError, match="training (image identity|OCI identity|code revision)"):
            bind_training_job_identity(
                bad_job,
                load_deployment_gate(gate_path, gate_digest),
                load_training_image_manifest(manifest_path),
            )

    mismatched_baked_manifest = tmp_path / "mismatched-training-image-manifest.json"
    _training_image_manifest(mismatched_baked_manifest, digest="d" * 64)
    with pytest.raises(ValueError, match="baked training OCI identity"):
        bind_training_job_identity(
            job,
            load_deployment_gate(gate_path, gate_digest),
            load_training_image_manifest(mismatched_baked_manifest),
        )

    mismatched_baked_code_manifest = tmp_path / "mismatched-training-code-manifest.json"
    _training_image_manifest(mismatched_baked_code_manifest)
    mismatched_baked_code_manifest.chmod(0o644)
    baked_document = json.loads(mismatched_baked_code_manifest.read_text(encoding="utf-8"))
    baked_document["trainer_code_revision"] = "c" * 40
    mismatched_baked_code_manifest.write_text(
        json.dumps(baked_document, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    mismatched_baked_code_manifest.chmod(0o444)
    with pytest.raises(ValueError, match="baked training code revision"):
        bind_training_job_identity(
            job,
            load_deployment_gate(gate_path, gate_digest),
            load_training_image_manifest(mismatched_baked_code_manifest),
        )
