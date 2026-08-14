"""Immutable checkpoint lifecycle for the no-recut runtime-mixture campaign.

The provider lifecycle deliberately lives elsewhere.  This module only turns a
completed optimizer boundary into an authenticated immutable publication and
then derives safe resume/disposal evidence from those publications.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Mapping, Protocol

from lehome_train.checkpoints import CheckpointDescriptor, load_checkpoint_descriptor
from lehome_train.constants import DEFAULT_MODEL_REPO
from lehome_train.io import canonical_json_sha256, sha256_file


_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_STEPS = (1000, 2000)
_PROVIDER_LOSS_KINDS = {"instance_absent", "preempted", "provider_unreachable"}


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _revision(value: object, label: str) -> str:
    if type(value) is not str or _REVISION.fullmatch(value) is None:
        raise ValueError(f"{label} must name an immutable revision")
    return value


def _relative(value: object, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value.startswith("/")
        or "\\" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError(f"{label} must be a safe relative path")
    return value


@dataclass(frozen=True, slots=True)
class RuntimeMixtureTrainingIdentity:
    """The complete fixed identity needed to resume runtime-mixture training."""

    mixture_id: str
    deployment_receipt_sha256: str
    # (source ID, immutable revision, authorized prefix, source-tree SHA-256)
    source_revisions: tuple[tuple[str, str, str, str], ...]
    schedule_seed: int
    code_bundle_sha256: str
    code_bundle_revision: str
    oci_image: str
    parent_step12000_artifact_sha256: str
    physical_batch_size: int = 64
    action_horizon: int = 16

    def __post_init__(self) -> None:
        _sha(self.mixture_id, "mixture_id")
        _sha(self.deployment_receipt_sha256, "deployment_receipt_sha256")
        _sha(self.code_bundle_sha256, "code_bundle_sha256")
        _revision(self.code_bundle_revision, "code_bundle_revision")
        _sha(self.parent_step12000_artifact_sha256, "parent_step12000_artifact_sha256")
        if type(self.schedule_seed) is not int or self.schedule_seed < 0:
            raise ValueError("schedule_seed must be a nonnegative integer")
        if type(self.oci_image) is not str or not self.oci_image.startswith("sha256:"):
            raise ValueError("oci_image must be an OCI digest")
        _sha(self.oci_image.removeprefix("sha256:"), "oci_image digest")
        if self.physical_batch_size != 64 or self.action_horizon != 16:
            raise ValueError("runtime mixture identity requires batch64 and h16")
        expected = {"organizer", "rollout"}
        if len(self.source_revisions) != 2 or {row[0] for row in self.source_revisions} != expected:
            raise ValueError("runtime mixture identity requires exact organizer and rollout sources")
        for source_id, revision, prefix, tree_sha256 in self.source_revisions:
            if type(source_id) is not str:
                raise ValueError("runtime source identity is malformed")
            _revision(revision, "runtime source revision")
            _relative(prefix, "runtime source prefix")
            _sha(tree_sha256, "runtime source tree hash")
            if (source_id == "organizer" and prefix != "bc/full") or (
                source_id == "rollout" and re.fullmatch(r"rollouts/round-[1-9][0-9]*", prefix) is None
            ):
                raise ValueError("runtime source identity has an unauthorized prefix")

    def to_dict(self) -> dict[str, object]:
        return {
            "mixture_id": self.mixture_id,
            "deployment_receipt_sha256": self.deployment_receipt_sha256,
            "source_revisions": [
                {"source_id": source_id, "immutable_revision": revision, "prefix": prefix, "tree_sha256": tree_sha}
                for source_id, revision, prefix, tree_sha in self.source_revisions
            ],
            "schedule_seed": self.schedule_seed,
            "code_bundle_sha256": self.code_bundle_sha256,
            "code_bundle_revision": self.code_bundle_revision,
            "oci_image": self.oci_image,
            "parent_step12000_artifact_sha256": self.parent_step12000_artifact_sha256,
            "physical_batch_size": self.physical_batch_size,
            "action_horizon": self.action_horizon,
        }

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.to_dict())


class RuntimeCheckpointHub(Protocol):
    def list_tree(self, *, repository: str, revision: str) -> object: ...
    def download_files(self, *, repository: str, revision: str, destination: Path, relative_paths: tuple[str, ...], remote_prefix: str) -> object: ...


@dataclass(frozen=True, slots=True)
class RuntimeMixtureResumeIdentity:
    identity_sha256: str
    optimizer_step: int
    global_sample_offset: int
    physical_batch_size: int
    checkpoint_archive: Path
    checkpoint_descriptor: Path

    def dataset_kwargs(self) -> dict[str, int]:
        return {
            "global_sample_offset": self.global_sample_offset,
            "expected_global_step": self.optimizer_step,
            "global_batch_size": self.physical_batch_size,
        }


def hub_checkpoint_publisher(uploader: object, *, timeout_seconds: float = 30.0) -> Callable[..., dict[str, object]]:
    """Adapt the proven ``HubCheckpointUploader.publish_receipt`` stable API."""
    method = getattr(uploader, "publish_receipt", None)
    if not callable(method):
        raise ValueError("checkpoint uploader has no publish_receipt method")

    def publish(*, checkpoint: CheckpointDescriptor, artifact_root: Path, identity: RuntimeMixtureTrainingIdentity) -> dict[str, object]:
        del artifact_root, identity
        value = method(checkpoint, timeout_seconds=timeout_seconds)
        if not isinstance(value, dict):
            raise ValueError("checkpoint uploader returned an invalid receipt")
        return value

    return publish


def _publication_fields(value: Mapping[str, object], *, step: int) -> dict[str, object]:
    required = {
        "optimizer_step", "repository", "immutable_revision", "remote_prefix", "relative_path",
        "artifact_sha256", "artifact_byte_size", "descriptor_relative_path", "descriptor_sha256",
        "descriptor_byte_size", "readback_verified",
    }
    if set(value) != required or value.get("optimizer_step") != step or value.get("repository") != DEFAULT_MODEL_REPO:
        raise ValueError("checkpoint publication has an incompatible schema")
    if value.get("readback_verified") is not True:
        raise ValueError("checkpoint publication lacks uploader readback verification")
    _revision(value.get("immutable_revision"), "checkpoint immutable revision")
    _relative(value.get("remote_prefix"), "checkpoint remote prefix")
    _relative(value.get("relative_path"), "checkpoint archive path")
    _relative(value.get("descriptor_relative_path"), "checkpoint descriptor path")
    _sha(value.get("artifact_sha256"), "checkpoint archive hash")
    _sha(value.get("descriptor_sha256"), "checkpoint descriptor hash")
    if any(type(value[field]) is not int or value[field] <= 0 for field in ("artifact_byte_size", "descriptor_byte_size")):
        raise ValueError("checkpoint publication byte sizes are invalid")
    return dict(value)


def _tree_paths(value: object) -> set[str]:
    result: set[str] = set()
    for entry in value if isinstance(value, (tuple, list)) else ():
        path = entry if isinstance(entry, str) else getattr(entry, "relative_path", None)
        entry_type = "file" if isinstance(entry, str) else getattr(entry, "entry_type", None)
        if type(path) is str and entry_type == "file":
            result.add(path)
    return result


def _verify_publication_readback(
    *, publication: Mapping[str, object], identity: RuntimeMixtureTrainingIdentity,
    hub: RuntimeCheckpointHub, destination: Path,
) -> CheckpointDescriptor:
    step = publication.get("optimizer_step")
    if step not in _STEPS:
        raise ValueError("runtime checkpoint publication has an unsupported step")
    raw = _publication_fields(
        {key: publication[key] for key in (
            "optimizer_step", "repository", "immutable_revision", "remote_prefix", "relative_path",
            "artifact_sha256", "artifact_byte_size", "descriptor_relative_path", "descriptor_sha256",
            "descriptor_byte_size", "readback_verified",
        ) if key in publication},
        step=step,
    )
    if destination.exists() or destination.is_symlink():
        raise ValueError("checkpoint readback destination must be absent")
    archive_path, descriptor_path = str(raw["relative_path"]), str(raw["descriptor_relative_path"])
    prefix = str(raw["remote_prefix"])
    tree = _tree_paths(hub.list_tree(repository=DEFAULT_MODEL_REPO, revision=str(raw["immutable_revision"])))
    if {f"{prefix}/{archive_path}", f"{prefix}/{descriptor_path}"} - tree:
        raise ValueError("checkpoint immutable tree lacks archive or descriptor")
    destination.mkdir(parents=True)
    try:
        hub.download_files(
            repository=DEFAULT_MODEL_REPO, revision=str(raw["immutable_revision"]), destination=destination,
            relative_paths=(archive_path, descriptor_path), remote_prefix=prefix,
        )
        archive, descriptor_file = destination / archive_path, destination / descriptor_path
        if (
            archive.is_symlink() or not archive.is_file() or archive.stat().st_size != raw["artifact_byte_size"]
            or sha256_file(archive) != raw["artifact_sha256"]
            or descriptor_file.is_symlink() or not descriptor_file.is_file() or descriptor_file.stat().st_size != raw["descriptor_byte_size"]
            or sha256_file(descriptor_file) != raw["descriptor_sha256"]
        ):
            raise ValueError("checkpoint immutable publication readback mismatch")
        descriptor = load_checkpoint_descriptor(descriptor_file)
        record = descriptor.record
        if (
            record.optimizer_step != step or record.dataset_manifest_sha256 != identity.mixture_id
            or record.sample_presentations != step * identity.physical_batch_size
            or record.artifact.relative_path != archive_path or record.artifact.sha256 != raw["artifact_sha256"]
            or record.artifact.byte_size != raw["artifact_byte_size"] or not record.resumable
        ):
            raise ValueError("checkpoint descriptor is not bound to runtime mixture identity")
        return descriptor
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def verify_runtime_mixture_publication(
    *, publication: Mapping[str, object], identity: RuntimeMixtureTrainingIdentity,
    hub: RuntimeCheckpointHub, destination: Path,
) -> CheckpointDescriptor:
    """Freshly tree-list and byte-readback a published runtime checkpoint."""
    if publication.get("kind") != "runtime_mixture_checkpoint_publication" or publication.get("schema_version") != 1:
        raise ValueError("runtime checkpoint publication kind is invalid")
    if publication.get("identity") != identity.to_dict() or publication.get("identity_sha256") != identity.sha256:
        raise ValueError("runtime checkpoint publication identity is incompatible")
    cursor = publication.get("runtime_cursor")
    step = publication.get("optimizer_step")
    if (
        not isinstance(cursor, Mapping) or step not in _STEPS
        or cursor != {"optimizer_step": step, "global_sample_offset": step * identity.physical_batch_size,
                      "physical_batch_size": identity.physical_batch_size, "action_horizon": identity.action_horizon}
        or publication.get("fresh_tree_readback_verified") is not True
    ):
        raise ValueError("runtime checkpoint cursor is incompatible")
    return _verify_publication_readback(publication=publication, identity=identity, hub=hub, destination=destination)


def publish_runtime_mixture_checkpoint(
    *, identity: RuntimeMixtureTrainingIdentity, checkpoint: CheckpointDescriptor, artifact_root: Path,
    publisher: Callable[..., dict[str, object]], hub: RuntimeCheckpointHub,
) -> dict[str, object]:
    """Publish one 1K/2K observer package and bind it to the fixed campaign."""
    step = checkpoint.record.optimizer_step
    if step not in _STEPS:
        raise ValueError("runtime checkpoints are only published at optimizer 1000 and 2000")
    archive = artifact_root / checkpoint.record.artifact.relative_path
    descriptor_path = artifact_root / f"checkpoints/step-{step}.json"
    if (
        checkpoint.record.dataset_manifest_sha256 != identity.mixture_id
        or checkpoint.record.sample_presentations != step * identity.physical_batch_size
        or checkpoint.record.artifact.sha256 != sha256_file(archive)
        or checkpoint.record.artifact.byte_size != archive.stat().st_size
        or load_checkpoint_descriptor(descriptor_path) != checkpoint
    ):
        raise ValueError("local runtime checkpoint package is incompatible")
    raw = _publication_fields(publisher(checkpoint=checkpoint, artifact_root=artifact_root, identity=identity), step=step)
    if raw["relative_path"] != checkpoint.record.artifact.relative_path:
        raise ValueError("publisher changed the checkpoint archive path")
    publication = {
        "schema_version": 1, "kind": "runtime_mixture_checkpoint_publication", **raw,
        "identity": identity.to_dict(), "identity_sha256": identity.sha256,
        "runtime_cursor": {"optimizer_step": step, "global_sample_offset": step * identity.physical_batch_size,
                           "physical_batch_size": identity.physical_batch_size, "action_horizon": identity.action_horizon},
        "fresh_tree_readback_verified": True,
    }
    scratch = Path(tempfile.mkdtemp(prefix="runtime-checkpoint-readback-", dir=artifact_root.parent))
    shutil.rmtree(scratch)
    _verify_publication_readback(publication=publication, identity=identity, hub=hub, destination=scratch)
    shutil.rmtree(scratch, ignore_errors=True)
    return publication


def _publications(
    publications: tuple[Mapping[str, object], ...], *, identity: RuntimeMixtureTrainingIdentity,
) -> tuple[dict[str, object], ...]:
    result = tuple(dict(item) for item in publications)
    steps = tuple(item.get("optimizer_step") for item in result)
    if len(result) != len(set(steps)) or any(step not in _STEPS for step in steps):
        raise ValueError("runtime checkpoint publications must be unique 1K/2K boundaries")
    for item in result:
        if item.get("kind") != "runtime_mixture_checkpoint_publication" or item.get("identity") != identity.to_dict() or item.get("identity_sha256") != identity.sha256:
            raise ValueError("runtime checkpoint publication is not bound to the training identity")
        step = item["optimizer_step"]
        expected = {"optimizer_step": step, "global_sample_offset": step * 64, "physical_batch_size": 64, "action_horizon": 16}
        if item.get("runtime_cursor") != expected or item.get("fresh_tree_readback_verified") is not True:
            raise ValueError("runtime checkpoint publication cursor is not authenticated")
    return result


def runtime_mixture_completion_terminal(
    *, identity: RuntimeMixtureTrainingIdentity, instance_id: str, publications: tuple[Mapping[str, object], ...],
) -> dict[str, object]:
    checked = _publications(publications, identity=identity)
    if {item["optimizer_step"] for item in checked} != set(_STEPS):
        raise ValueError("runtime mixture completion requires exact optimizer 1000 and 2000 publications")
    return _terminal(kind="runtime_mixture_training_complete", status="complete", identity=identity, instance_id=instance_id, publications=checked, provider_loss=None, disposable=True)


def provider_interruption_terminal(
    *, identity: RuntimeMixtureTrainingIdentity, instance_id: str, publications: tuple[Mapping[str, object], ...], provider_loss: Mapping[str, object],
) -> dict[str, object]:
    checked = _publications(publications, identity=identity)
    if not checked or set(provider_loss) != {"kind", "evidence_sha256"} or provider_loss.get("kind") not in _PROVIDER_LOSS_KINDS:
        raise ValueError("verified provider loss is required for an interruption terminal")
    _sha(provider_loss.get("evidence_sha256"), "provider loss evidence")
    return _terminal(kind="runtime_mixture_provider_interrupted", status="provider_interrupted", identity=identity, instance_id=instance_id, publications=checked, provider_loss=dict(provider_loss), disposable=False)


def _terminal(*, kind: str, status: str, identity: RuntimeMixtureTrainingIdentity, instance_id: str, publications: tuple[dict[str, object], ...], provider_loss: dict[str, object] | None, disposable: bool) -> dict[str, object]:
    if type(instance_id) is not str or not instance_id:
        raise ValueError("instance_id is required")
    step = max(int(item["optimizer_step"]) for item in publications)
    return {
        "schema_version": 1, "kind": kind, "status": status, "instance_id": instance_id,
        "identity": identity.to_dict(), "identity_sha256": identity.sha256,
        "immutable_checkpoint_publications": list(sorted(publications, key=lambda item: int(item["optimizer_step"]))),
        "resumable_checkpoint_step": step,
        "runtime_cursor": {"optimizer_step": step, "global_sample_offset": step * 64, "physical_batch_size": 64, "action_horizon": 16},
        "provider_loss": provider_loss, "disposable": disposable,
    }


def _terminal_publications(terminal: Mapping[str, object], *, identity: RuntimeMixtureTrainingIdentity, allow_complete: bool) -> tuple[dict[str, object], ...]:
    required = {
        "schema_version", "kind", "status", "instance_id", "identity", "identity_sha256",
        "immutable_checkpoint_publications", "resumable_checkpoint_step", "runtime_cursor",
        "provider_loss", "disposable",
    }
    if set(terminal) != required:
        raise ValueError("runtime terminal has an incompatible receipt schema")
    kind = terminal.get("kind")
    expected_kind = {"runtime_mixture_provider_interrupted", "runtime_mixture_training_complete"} if allow_complete else {"runtime_mixture_provider_interrupted"}
    if kind not in expected_kind or terminal.get("schema_version") != 1 or terminal.get("identity") != identity.to_dict() or terminal.get("identity_sha256") != identity.sha256:
        raise ValueError("runtime terminal identity is incompatible")
    if kind == "runtime_mixture_provider_interrupted":
        loss = terminal.get("provider_loss")
        if (
            not isinstance(loss, Mapping) or set(loss) != {"kind", "evidence_sha256"}
            or loss.get("kind") not in _PROVIDER_LOSS_KINDS
        ):
            raise ValueError("runtime terminal lacks authenticated provider loss")
        _sha(loss.get("evidence_sha256"), "provider loss evidence")
    raw = terminal.get("immutable_checkpoint_publications")
    if not isinstance(raw, list):
        raise ValueError("runtime terminal publications are missing")
    checked = _publications(tuple(raw), identity=identity)
    step = max(int(item["optimizer_step"]) for item in checked) if checked else None
    expected_cursor = None if step is None else {"optimizer_step": step, "global_sample_offset": step * 64, "physical_batch_size": 64, "action_horizon": 16}
    if terminal.get("resumable_checkpoint_step") != step or terminal.get("runtime_cursor") != expected_cursor:
        raise ValueError("runtime terminal has a double or stale cursor")
    return checked


def discover_runtime_mixture_resume(
    *, terminal: Mapping[str, object], identity: RuntimeMixtureTrainingIdentity, hub: RuntimeCheckpointHub, destination: Path,
) -> RuntimeMixtureResumeIdentity:
    """Discover and hydrate only an authenticated provider-interruption resume."""
    publications = _terminal_publications(terminal, identity=identity, allow_complete=False)
    latest = max(publications, key=lambda item: int(item["optimizer_step"]))
    _verify_publication_readback(publication=latest, identity=identity, hub=hub, destination=destination)
    step = int(latest["optimizer_step"])
    return RuntimeMixtureResumeIdentity(identity.sha256, step, step * 64, 64, destination / str(latest["relative_path"]), destination / str(latest["descriptor_relative_path"]))


def authorize_runtime_mixture_disposal(
    *, instance_id: str, terminal: Mapping[str, object], identity: RuntimeMixtureTrainingIdentity, hub: RuntimeCheckpointHub,
) -> dict[str, object]:
    """Return evidence authorizing disposal; never performs a provider action."""
    publications = _terminal_publications(terminal, identity=identity, allow_complete=True)
    if (
        terminal.get("kind") != "runtime_mixture_training_complete" or terminal.get("status") != "complete"
        or terminal.get("disposable") is not True or terminal.get("instance_id") != instance_id
        or {item["optimizer_step"] for item in publications} != set(_STEPS)
    ):
        raise ValueError("runtime mixture disposal requires the exact completed instance and both publications")
    for publication in publications:
        temporary = Path(tempfile.mkdtemp(prefix="runtime-disposal-readback-"))
        shutil.rmtree(temporary)
        try:
            verify_runtime_mixture_publication(publication=publication, identity=identity, hub=hub, destination=temporary)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
    return {"schema_version": 1, "kind": "runtime_mixture_disposal_authorization", "instance_id": instance_id, "identity_sha256": identity.sha256, "publication_steps": [1000, 2000], "both_publication_readbacks_verified": True}
