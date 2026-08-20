"""Canonical, read-back checkpoint publications for asynchronous experiments.

This deliberately contains only non-secret remote object identity.  A training
worker may report that a terminal receipt exists, but an evaluator is never
leased until this receipt has been parsed and bound to the immutable job.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import Mapping


_SHA = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_FIELDS = {
    "schema_version",
    "experiment_id",
    "job_digest",
    "target_step",
    "repository",
    "immutable_revision",
    "remote_prefix",
    "artifact_sha256",
    "receipt_sha256",
    "readback_verified",
}
_V2_FIELDS = _FIELDS | {
    "relative_path",
    "artifact_byte_size",
    "descriptor_relative_path",
    "descriptor_sha256",
    "descriptor_byte_size",
}


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise ValueError(f"publication {label} must be SHA-256")
    return value


def _repository(value: object) -> str:
    if type(value) is not str or not value or value.startswith("/") or " " in value or value.count("/") != 1:
        raise ValueError("publication repository is invalid")
    return value


def _prefix(value: object) -> str:
    if type(value) is not str or not value or value.startswith("/") or "\\" in value:
        raise ValueError("publication remote prefix is invalid")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("publication remote prefix is invalid")
    return value


def _relative_path(value: object, label: str) -> str:
    if type(value) is not str or not value or value.startswith("/") or "\\" in value:
        raise ValueError(f"publication {label} relative path is invalid")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"publication {label} relative path is invalid")
    return value


def _positive_size(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"publication {label} byte size is invalid")
    return value


@dataclass(frozen=True, slots=True)
class CheckpointPublication:
    experiment_id: str
    job_digest: str
    target_step: int
    repository: str
    immutable_revision: str
    remote_prefix: str
    artifact_sha256: str
    receipt_sha256: str
    canonical: Mapping[str, object]
    relative_path: str | None = None
    artifact_byte_size: int | None = None
    descriptor_relative_path: str | None = None
    descriptor_sha256: str | None = None
    descriptor_byte_size: int | None = None


def parse_checkpoint_publication(value: Mapping[str, object]) -> CheckpointPublication:
    """Parse one exact, read-back verified immutable publication envelope."""
    document = dict(value)
    schema_version = document.get("schema_version")
    if (schema_version == 1 and set(document) != _FIELDS) or (schema_version == 2 and set(document) != _V2_FIELDS) or schema_version not in {1, 2}:
        raise ValueError("publication envelope schema is invalid")
    if document.get("readback_verified") is not True:
        raise ValueError("publication requires readback verification")
    target_step = document.get("target_step")
    if type(target_step) is not int or target_step not in {500, 1000, 2000}:
        raise ValueError("publication target step is invalid")
    revision = document.get("immutable_revision")
    if type(revision) is not str or _REVISION.fullmatch(revision) is None:
        raise ValueError("publication immutable revision is invalid")
    canonical_data: dict[str, object] = {
        "schema_version": schema_version,
        "experiment_id": _sha(document.get("experiment_id"), "experiment ID"),
        "job_digest": _sha(document.get("job_digest"), "job digest"),
        "target_step": target_step,
        "repository": _repository(document.get("repository")),
        "immutable_revision": revision,
        "remote_prefix": _prefix(document.get("remote_prefix")),
        "artifact_sha256": _sha(document.get("artifact_sha256"), "artifact digest"),
        "receipt_sha256": _sha(document.get("receipt_sha256"), "receipt"),
        "readback_verified": True,
    }
    if schema_version == 2:
        canonical_data.update({
            "relative_path": _relative_path(document.get("relative_path"), "artifact"),
            "artifact_byte_size": _positive_size(document.get("artifact_byte_size"), "artifact"),
            "descriptor_relative_path": _relative_path(document.get("descriptor_relative_path"), "descriptor"),
            "descriptor_sha256": _sha(document.get("descriptor_sha256"), "descriptor"),
            "descriptor_byte_size": _positive_size(document.get("descriptor_byte_size"), "descriptor"),
        })
    canonical = MappingProxyType(canonical_data)
    return CheckpointPublication(
        experiment_id=canonical["experiment_id"],
        job_digest=canonical["job_digest"],
        target_step=target_step,
        repository=canonical["repository"],
        immutable_revision=revision,
        remote_prefix=canonical["remote_prefix"],
        artifact_sha256=canonical["artifact_sha256"],
        receipt_sha256=canonical["receipt_sha256"],
        canonical=canonical,
        relative_path=canonical.get("relative_path") if isinstance(canonical.get("relative_path"), str) else None,
        artifact_byte_size=canonical.get("artifact_byte_size") if type(canonical.get("artifact_byte_size")) is int else None,
        descriptor_relative_path=canonical.get("descriptor_relative_path") if isinstance(canonical.get("descriptor_relative_path"), str) else None,
        descriptor_sha256=canonical.get("descriptor_sha256") if isinstance(canonical.get("descriptor_sha256"), str) else None,
        descriptor_byte_size=canonical.get("descriptor_byte_size") if type(canonical.get("descriptor_byte_size")) is int else None,
    )


def bind_checkpoint_publication(job: object, terminal_receipt_sha256: str, value: Mapping[str, object]) -> CheckpointPublication:
    """Bind an already-read-back receipt to one immutable training job."""
    publication = parse_checkpoint_publication(value)
    experiment_id = getattr(job, "experiment_id", None)
    training = getattr(job, "training", None)
    destination = getattr(job, "publication", None)
    if publication.experiment_id != experiment_id or publication.job_digest != experiment_id or publication.target_step != getattr(training, "target_step", None):
        raise ValueError("publication envelope does not bind job")
    if publication.repository != getattr(destination, "checkpoint_repository", None) or not publication.remote_prefix.startswith(str(getattr(destination, "prefix", "")) + "/"):
        raise ValueError("publication envelope does not bind destination")
    if publication.receipt_sha256 != terminal_receipt_sha256:
        raise ValueError("publication receipt mismatch")
    return publication
