"""Canonical publication receipts gate evaluation leases."""

from __future__ import annotations


def test_publication_envelope_requires_exact_immutable_checkpoint_identity() -> None:
    from lehome_train.groot.experiment_publication import parse_checkpoint_publication

    raw = {
        "schema_version": 1,
        "experiment_id": "a" * 64,
        "job_digest": "a" * 64,
        "target_step": 500,
        "repository": "owner/checkpoints",
        "immutable_revision": "b" * 40,
        "remote_prefix": "experiments/a/step-500",
        "artifact_sha256": "c" * 64,
        "receipt_sha256": "d" * 64,
        "readback_verified": True,
    }
    receipt = parse_checkpoint_publication(raw)
    assert receipt.canonical["immutable_revision"] == "b" * 40

    raw["target_step"] = 0
    try:
        parse_checkpoint_publication(raw)
    except ValueError as error:
        assert "target step" in str(error)
    else:
        raise AssertionError("zero-step pseudo publication was accepted")


def test_publication_v2_carries_exact_archive_and_descriptor_readback_bindings() -> None:
    from lehome_train.groot.experiment_publication import parse_checkpoint_publication

    raw = {
        "schema_version": 2,
        "experiment_id": "a" * 64,
        "job_digest": "a" * 64,
        "target_step": 500,
        "repository": "owner/checkpoints",
        "immutable_revision": "b" * 40,
        "remote_prefix": "experiments/a/step-500",
        "relative_path": "checkpoints/step-500.tar",
        "artifact_sha256": "c" * 64,
        "artifact_byte_size": 123,
        "descriptor_relative_path": "checkpoints/step-500.json",
        "descriptor_sha256": "d" * 64,
        "descriptor_byte_size": 45,
        "receipt_sha256": "e" * 64,
        "readback_verified": True,
    }
    publication = parse_checkpoint_publication(raw)
    assert publication.relative_path == "checkpoints/step-500.tar"
    assert publication.descriptor_sha256 == "d" * 64

    raw["relative_path"] = "../escape.tar"
    try:
        parse_checkpoint_publication(raw)
    except ValueError as error:
        assert "relative path" in str(error)
    else:
        raise AssertionError("unsafe archive path was accepted")
