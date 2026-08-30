"""Evidence verification and normalization for auditable training reports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Mapping

from lehome_train.checkpoints import CheckpointDescriptor
from lehome_train.commands.sync import SyncResult
from lehome_train.constants import DEFAULT_MODEL_REPO
from lehome_train.io import atomic_write_json
from lehome_train.models import ArtifactIdentity, SyncEntry, model_from_mapping


_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def parse_timestamp(value: str, *, label: str) -> datetime:
    """Parse one explicit timezone-aware ISO-8601 timestamp."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be an explicit timezone-aware timestamp")
    try:
        parsed = datetime.fromisoformat(
            value[:-1] + "+00:00" if value.endswith("Z") else value
        )
    except ValueError:
        raise ValueError(f"{label} must be an explicit timezone-aware timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must be an explicit timezone-aware timestamp")
    return parsed


def canonical_timestamp(value: datetime) -> str:
    """Serialize one timestamp as canonical UTC ISO-8601 with a Z suffix."""

    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def require_immutable_revision(value: str, *, label: str) -> None:
    """Require one full lowercase Git commit revision."""

    if not isinstance(value, str) or not _COMMIT.fullmatch(value):
        raise ValueError(f"{label} must be an immutable 40-character commit revision")


@dataclass(frozen=True, slots=True)
class CheckpointPruningReceipt:
    """Controller deletion evidence tied to one remotely verified artifact."""

    experiment_id: str
    experiment_config_sha256: str
    artifact: ArtifactIdentity
    immutable_revision: str
    deleted_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_id, str) or not self.experiment_id:
            raise ValueError("pruning receipt experiment identity is invalid")
        if (
            not isinstance(self.experiment_config_sha256, str)
            or not _SHA256.fullmatch(self.experiment_config_sha256)
        ):
            raise ValueError("pruning receipt config identity is invalid")
        if not isinstance(self.artifact, ArtifactIdentity):
            raise TypeError("pruning receipt artifact must be an ArtifactIdentity")
        require_immutable_revision(
            self.immutable_revision,
            label="pruning receipt revision",
        )
        deleted = parse_timestamp(
            self.deleted_at,
            label="pruning receipt deletion time",
        )
        object.__setattr__(self, "deleted_at", canonical_timestamp(deleted))

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "experiment_config_sha256": self.experiment_config_sha256,
            "artifact": self.artifact.to_dict(),
            "immutable_revision": self.immutable_revision,
            "deleted_at": self.deleted_at,
        }


def write_checkpoint_pruning_receipt(
    destination: str | Path,
    receipt: CheckpointPruningReceipt,
) -> None:
    """Atomically persist one strict deletion receipt."""

    if not isinstance(receipt, CheckpointPruningReceipt):
        raise TypeError("receipt must be a CheckpointPruningReceipt")
    atomic_write_json(destination, {"schema_version": 1, **receipt.to_dict()})


def load_checkpoint_pruning_receipt(
    source: str | Path,
) -> CheckpointPruningReceipt:
    """Load a pruning receipt while rejecting duplicate and unknown fields."""

    def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("pruning receipt contains a duplicate field")
            value[key] = item
        return value

    try:
        decoded = json.loads(
            Path(source).read_text(encoding="utf-8"),
            object_pairs_hook=strict_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("pruning receipt is unavailable or malformed") from None
    fields = {
        "schema_version",
        "experiment_id",
        "experiment_config_sha256",
        "artifact",
        "immutable_revision",
        "deleted_at",
    }
    if not isinstance(decoded, Mapping) or set(decoded) != fields:
        raise ValueError("pruning receipt has an incompatible schema")
    if decoded["schema_version"] != 1 or not isinstance(decoded["artifact"], Mapping):
        raise ValueError("pruning receipt has an incompatible schema")
    return CheckpointPruningReceipt(
        experiment_id=decoded["experiment_id"],
        experiment_config_sha256=decoded["experiment_config_sha256"],
        artifact=model_from_mapping(ArtifactIdentity, decoded["artifact"]),
        immutable_revision=decoded["immutable_revision"],
        deleted_at=decoded["deleted_at"],
    )


def _local_artifact_matches(root: Path, artifact: ArtifactIdentity) -> bool:
    candidate = root.joinpath(*artifact.relative_path.split("/"))
    current = root
    for component in artifact.relative_path.split("/")[:-1]:
        current = current / component
        if current.is_symlink():
            raise ValueError("local checkpoint artifact path contains a symlink")
    try:
        path_metadata = os.lstat(candidate)
    except FileNotFoundError:
        return False
    except OSError:
        raise ValueError("local checkpoint artifact could not be inspected safely") from None
    if not stat.S_ISREG(path_metadata.st_mode):
        raise ValueError("local checkpoint artifact must be a regular file")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(candidate, flags)
    except OSError:
        raise ValueError("local checkpoint artifact could not be opened safely") from None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("local checkpoint artifact must be a regular file")
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    finally:
        os.close(descriptor)
    if metadata.st_size != artifact.byte_size or digest.hexdigest() != artifact.sha256:
        raise ValueError("local checkpoint artifact hash or size differs from descriptor")
    return True


def _validate_sync_evidence(
    evidence: SyncResult | None,
    *,
    experiment_id: str,
    config_sha256: str,
) -> tuple[dict[str, SyncEntry], dict[str, object] | None, bool]:
    if evidence is None:
        return {}, None, False
    if not isinstance(evidence, SyncResult):
        raise TypeError("sync_evidence must be a SyncResult or None")
    require_immutable_revision(
        evidence.immutable_revision,
        label="sync evidence revision",
    )
    if evidence.repository != DEFAULT_MODEL_REPO:
        raise ValueError("sync evidence repository is incompatible")
    if evidence.manifest.experiment_id != experiment_id:
        raise ValueError("sync evidence experiment identity is incompatible")
    if evidence.manifest.experiment_config_sha256 != config_sha256:
        raise ValueError("sync evidence experiment config identity is incompatible")
    if type(evidence.disposable) is not bool:
        raise ValueError("sync evidence disposable state is invalid")
    if evidence.disposable and (
        not evidence.manifest.entries
        or not all(entry.remotely_verified for entry in evidence.manifest.entries)
    ):
        raise ValueError("sync evidence disposable state contradicts its manifest")
    return (
        {entry.relative_path: entry for entry in evidence.manifest.entries},
        {
            "repository": evidence.repository,
            "immutable_revision": evidence.immutable_revision,
            "remote_prefix": evidence.manifest.remote_prefix,
        },
        evidence.disposable,
    )


def normalize_checkpoint_evidence(
    checkpoints: tuple[CheckpointDescriptor, ...],
    *,
    experiment_id: str,
    config_sha256: str,
    dataset_manifest_sha256: str,
    effective_batch_size: int,
    local_artifact_root: Path,
    sync_evidence: SyncResult | None,
    pruning_receipts: tuple[CheckpointPruningReceipt, ...],
) -> tuple[tuple[dict[str, object], ...], dict[str, object] | None, bool]:
    """Normalize checkpoint claims against local, remote, and deletion evidence."""

    if not checkpoints:
        raise ValueError("training report requires at least one checkpoint")
    remote_entries, sync_summary, sync_disposable = _validate_sync_evidence(
        sync_evidence,
        experiment_id=experiment_id,
        config_sha256=config_sha256,
    )
    sync_revision = None if sync_evidence is None else sync_evidence.immutable_revision
    receipt_by_path: dict[str, CheckpointPruningReceipt] = {}
    for receipt in pruning_receipts:
        if not isinstance(receipt, CheckpointPruningReceipt):
            raise TypeError("pruning receipts must be CheckpointPruningReceipt values")
        path = receipt.artifact.relative_path
        if path in receipt_by_path:
            raise ValueError("pruning receipts contain duplicate artifact paths")
        receipt_by_path[path] = receipt

    normalized: list[dict[str, object]] = []
    expected_normalization_sha256: str | None = None
    expected_schedule_sha256: str | None = None
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, CheckpointDescriptor):
            raise TypeError("training report checkpoints must be CheckpointDescriptor values")
        record = checkpoint.record
        if record.experiment_id != experiment_id:
            raise ValueError("checkpoint experiment identity differs from smoke result")
        if record.experiment_config_sha256 != config_sha256:
            raise ValueError("checkpoint experiment config identity is incompatible")
        if record.dataset_manifest_sha256 != dataset_manifest_sha256:
            raise ValueError("checkpoint prepared dataset identity is incompatible")
        if expected_normalization_sha256 is None:
            expected_normalization_sha256 = checkpoint.normalization_sha256
        elif checkpoint.normalization_sha256 != expected_normalization_sha256:
            raise ValueError("checkpoint normalization identities are incompatible")
        if expected_schedule_sha256 is None:
            expected_schedule_sha256 = checkpoint.schedule_sha256
        elif checkpoint.schedule_sha256 != expected_schedule_sha256:
            raise ValueError("checkpoint schedule identities are incompatible")
        if record.sample_presentations != record.optimizer_step * effective_batch_size:
            raise ValueError("checkpoint sample presentations are incompatible")

        local_verified = _local_artifact_matches(local_artifact_root, record.artifact)
        if checkpoint.locally_verified and not local_verified:
            raise ValueError("controller-reported local checkpoint artifact is missing")
        remote_entry = remote_entries.get(record.artifact.relative_path)
        remote_verified = bool(
            remote_entry is not None
            and remote_entry.remotely_verified
            and remote_entry.sha256 == record.artifact.sha256
            and remote_entry.byte_size == record.artifact.byte_size
        )
        receipt = receipt_by_path.pop(record.artifact.relative_path, None)
        if receipt is not None:
            if local_verified:
                raise ValueError("pruning receipt contradicts retained local artifact bytes")
            if receipt.experiment_id != experiment_id:
                raise ValueError("pruning receipt experiment identity is incompatible")
            if receipt.experiment_config_sha256 != config_sha256:
                raise ValueError("pruning receipt config identity is incompatible")
            if receipt.artifact != record.artifact:
                raise ValueError("pruning receipt artifact identity is incompatible")
            if sync_revision is None or receipt.immutable_revision != sync_revision:
                raise ValueError("pruning receipt immutable revision is incompatible")
            if not remote_verified:
                raise ValueError("pruning receipt lacks matching immutable sync evidence")

        if local_verified:
            retained_locally: bool | None = True
            retention_state = "retained_locally_verified"
            retention_evidence_level = "verified_bytes"
        elif receipt is not None:
            retained_locally = False
            retention_state = "pruned_with_receipt"
            retention_evidence_level = "deletion_receipt"
        else:
            retained_locally = None
            retention_state = (
                "controller_reported_pruned"
                if not checkpoint.locally_verified
                else "unverified"
            )
            retention_evidence_level = "reported_only"
        normalized.append(
            {
                "artifact": record.artifact.to_dict(),
                "controller_reported_locally_verified": checkpoint.locally_verified,
                "controller_reported_remotely_verified": record.remotely_verified,
                "dataset_manifest_sha256": record.dataset_manifest_sha256,
                "deletion_receipt": None if receipt is None else receipt.to_dict(),
                "experiment_config_sha256": record.experiment_config_sha256,
                "experiment_id": record.experiment_id,
                "local_evidence_level": (
                    "verified_bytes" if local_verified else "unverified"
                ),
                "locally_verified": local_verified,
                "normalization_sha256": checkpoint.normalization_sha256,
                "optimizer_step": record.optimizer_step,
                "remote_evidence_level": (
                    "immutable_sync_readback"
                    if remote_verified
                    else (
                        "descriptor_reported_only"
                        if record.remotely_verified
                        else "unverified"
                    )
                ),
                "remotely_verified": remote_verified,
                "resumable": record.resumable,
                "retention_evidence_level": retention_evidence_level,
                "retention_state": retention_state,
                "retained_locally": retained_locally,
                "sample_presentations": record.sample_presentations,
                "schedule_sha256": checkpoint.schedule_sha256,
            }
        )

    if receipt_by_path:
        raise ValueError("pruning receipt does not identify a reported checkpoint")
    paths = tuple(item["artifact"]["relative_path"] for item in normalized)
    if len(paths) != len(set(paths)):
        raise ValueError("training report checkpoint artifact paths must be unique")
    optimizer_steps = tuple(item["optimizer_step"] for item in normalized)
    if len(optimizer_steps) != len(set(optimizer_steps)):
        raise ValueError("training report checkpoint optimizer steps must be unique")
    checkpoints_remotely_verified = all(
        checkpoint["remotely_verified"] is True for checkpoint in normalized
    )
    return (
        tuple(normalized),
        sync_summary,
        sync_disposable and checkpoints_remotely_verified,
    )
