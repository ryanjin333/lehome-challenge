"""Hydrate and verify one immutable runtime request-set bundle.

The experiment worker never executes untrusted remote shell text.  It accepts
only a pinned, file-by-file verified request set and writes its own private
runtime environment for the existing guest training controller.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import tempfile
from types import MappingProxyType
from typing import Any, Mapping, Protocol

from lehome_train.checkpoints import load_checkpoint_descriptor
from lehome_train.groot.experiment_job import ArtifactBinding, ExperimentJob
from lehome_train.groot.experiment_publication import (
    bind_checkpoint_publication,
    parse_checkpoint_publication,
)
from lehome_train.io import canonical_json_sha256, sha256_file


_ENV_KEYS = {
    "LEHOME_EXPERIMENT_MANIFEST", "LEHOME_EXPERIMENT_MANIFEST_SHA256",
    "LEHOME_CODE_BUNDLE", "LEHOME_CODE_BUNDLE_SHA256", "LEHOME_CODE_REVISION",
    "LEHOME_RUNTIME_HYDRATE_REQUEST", "LEHOME_RUNTIME_HYDRATE_REQUEST_SHA256",
    "LEHOME_RUNTIME_PILOT_REQUEST", "LEHOME_RUNTIME_PILOT_REQUEST_SHA256",
    "LEHOME_RUNTIME_WARMUP_REQUEST", "LEHOME_RUNTIME_WARMUP_REQUEST_SHA256",
    "LEHOME_RUNTIME_TRAIN_REQUEST", "LEHOME_RUNTIME_TRAIN_REQUEST_SHA256",
}


class RequestSetHydrator(Protocol):
    def hydrate(self, source: ArtifactBinding, destination: Path) -> Path: ...


class ParentCheckpointHub(Protocol):
    """The deliberately small read-only Hub boundary for a promoted parent."""

    def list_tree(self, *, repository: str, revision: str) -> object: ...

    def download_files(
        self,
        *,
        repository: str,
        revision: str,
        destination: Path,
        relative_paths: tuple[str, ...],
        remote_prefix: str,
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class PromotedParentMaterialization:
    """A verified immutable parent archive and its extracted policy directory."""

    archive: Path
    descriptor: Path
    checkpoint_path: Path
    cursor: Mapping[str, int]
    publication: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class SweepRuntimeRequest:
    """The per-job canonical bytes consumed by the existing guest command."""

    launch_config: Path
    experiment_config: Path
    train_request: Path
    binding: Path
    result_output: Path


def _safe_relative(value: object, label: str) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise ValueError(f"{label} is unsafe")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} is unsafe")
    return value


def _strict_json(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is malformed") from error
    if not isinstance(value, dict) or json.dumps(value, sort_keys=True, separators=(",", ":")).encode() != raw:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _runtime_source(job: ExperimentJob) -> ArtifactBinding:
    sources = [source for source in job.data_sources if source.kind == "runtime_request_set"]
    if len(sources) != 1:
        raise ValueError("production experiment requires exactly one runtime_request_set source")
    return sources[0]


def runtime_profile_document(job: ExperimentJob) -> dict[str, object]:
    """Non-cyclic job projection which a request set can bind before job finalization."""
    raw = dict(job.raw)
    raw.pop("experiment_id", None)
    raw["data_sources"] = [
        value for value in raw["data_sources"]
        if isinstance(value, dict) and value.get("kind") != "runtime_request_set"
    ]
    return raw


def runtime_profile_sha256(job: ExperimentJob) -> str:
    return canonical_json_sha256(runtime_profile_document(job))


def runtime_compatibility_profile_document(job: ExperimentJob) -> dict[str, object]:
    """Return the immutable portion reusable by a promoted sweep child.

    This deliberately is *not* a pruned copy of ``job.raw``.  Controller
    generated descendants change orchestration coordinates (arm label,
    dependency/admission, seed, target rung, promoted parent, output prefix,
    and the policy being evaluated), while retaining the exact request set
    selected for their parent.  Binding any of those mutable coordinates here
    would make every otherwise-valid seed repeat or promoted continuation fail
    before hydration.

    The projection therefore binds only the immutable training contract: the
    schema/trainer identity, ordinary data sources, mixture, h16/b64/save
    cadence, evaluation matrix identity, and publication repositories.  The
    dynamically supplied overlay separately binds the child-specific parent,
    target, seed, policy digest, and prefix after controller admission.
    """
    return {
        "schema_version": 1,
        "kind": "lehome_runtime_compatibility_profile",
        "trainer": dict(job.trainer),
        "data_sources": [
            {
                "kind": source.kind,
                "repository": source.repository,
                "revision": source.revision,
                "prefix": source.prefix,
                "manifest_sha256": source.manifest_sha256,
                "tree_sha256": source.tree_sha256,
            }
            for source in job.data_sources
            if source.kind != "runtime_request_set"
        ],
        "mixture": {
            "bc_percent": job.mixture.bc_percent,
            "added_percent": job.mixture.added_percent,
            "batch64_quotas": dict(job.mixture.batch64_quotas),
            "sampling_strategy": job.mixture.sampling_strategy,
        },
        "training": {
            "action_horizon": job.training.action_horizon,
            "batch_size": job.training.batch_size,
            "save_steps": job.training.save_steps,
        },
        "evaluation": {
            "matrix_id": job.evaluation.matrix_id,
            "matrix_sha256": job.evaluation.matrix_sha256,
        },
        "publication": {
            "checkpoint_repository": job.publication.checkpoint_repository,
            "result_repository": job.publication.result_repository,
        },
    }


def runtime_compatibility_profile_sha256(job: ExperimentJob) -> str:
    return canonical_json_sha256(runtime_compatibility_profile_document(job))


def build_sweep_train_overlay(
    job: ExperimentJob,
    *, workspace: Path,
    base_train_request: Path,
    compatibility_profile_sha256: str,
    parent_publication: Mapping[str, object] | None = None,
) -> Path:
    """Write the local job-specific overlay, without pretending it is hydrated.

    The guest/runtime currently does not consume this document.  Keeping that
    distinction explicit prevents a promoted job from executing with a parent
    archive that has not first been fresh-readback hydrated.  A subsequent
    checkpoint-hydrator slice may consume this exact contract.
    """
    workspace = Path(workspace)
    base_train_request = Path(base_train_request)
    if (
        not workspace.is_absolute() or workspace.is_symlink() or not workspace.is_dir()
        or base_train_request.is_symlink() or not base_train_request.is_file()
    ):
        raise ValueError("sweep train overlay workspace is unsafe")
    try:
        relative = base_train_request.relative_to(workspace).as_posix()
    except ValueError as error:
        raise ValueError("sweep train overlay base request escapes workspace") from error
    _safe_relative(relative, "sweep train overlay base request")
    if type(compatibility_profile_sha256) is not str or len(compatibility_profile_sha256) != 64:
        raise ValueError("sweep train overlay compatibility profile hash is invalid")
    if compatibility_profile_sha256 != runtime_compatibility_profile_sha256(job):
        raise ValueError("sweep train overlay does not match its reusable compatibility profile")
    checked_parent_publication = _promoted_parent_publication(job, parent_publication)
    target = workspace / "prepared" / "sweep-train-overlay.json"
    document = {
        "schema_version": 1,
        "kind": "lehome_sweep_train_request_overlay",
        "base_train_request": relative,
        "base_train_request_sha256": sha256_file(base_train_request),
        "compatibility_profile_sha256": compatibility_profile_sha256,
        "experiment_id": job.experiment_id,
        "job_digest": job.experiment_id,
        "parent_checkpoint": dict(job.parent_checkpoint),
        "parent_publication": checked_parent_publication,
        "trainer": dict(job.trainer),
        "mixture": {
            "bc_percent": job.mixture.bc_percent,
            "added_percent": job.mixture.added_percent,
            "batch64_quotas": dict(job.mixture.batch64_quotas),
            "sampling_strategy": job.mixture.sampling_strategy,
        },
        "training": {
            "seed": job.training.seed,
            "target_step": job.training.target_step,
            "save_steps": job.training.save_steps,
        },
        "evaluation": {
            "matrix_id": job.evaluation.matrix_id,
            "matrix_sha256": job.evaluation.matrix_sha256,
            "policy_digest": job.evaluation.policy_digest,
        },
        "publication": {
            "checkpoint_repository": job.publication.checkpoint_repository,
            "result_repository": job.publication.result_repository,
            "prefix": job.publication.prefix,
        },
    }
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != payload:
            raise ValueError("sweep train overlay conflicts with immutable prior bytes")
        return target
    target.write_bytes(payload)
    target.chmod(0o600)
    return target


def _canonical_object(path: Path, label: str) -> dict[str, object]:
    value = _strict_json(path, label)
    return dict(value)


def _write_private_canonical(path: Path, value: Mapping[str, object]) -> None:
    payload = json.dumps(dict(value), sort_keys=True, separators=(",", ":")).encode()
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise ValueError("dynamic sweep request conflicts with immutable prior bytes")
        return
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("dynamic sweep request destination is unsafe")
    path.write_bytes(payload)
    path.chmod(0o600)


def build_sweep_runtime_request(
    job: ExperimentJob,
    *,
    workspace: Path,
    base_train_request: Path,
    base_launch_config: Path,
    base_experiment_config: Path,
    overlay: Path,
    promoted_parent: PromotedParentMaterialization | None,
) -> SweepRuntimeRequest:
    """Derive per-job config/experiment/train envelopes with canonical bytes.

    The immutable request-set supplies only common, compatibility-bound input
    files.  Every field that must vary by experiment is generated locally from
    the authenticated job plus overlay; no mutable text is accepted from a
    controller response or a shell environment.
    """
    workspace = Path(workspace)
    paths = (base_train_request, base_launch_config, base_experiment_config, overlay)
    if not workspace.is_absolute() or workspace.is_symlink() or not workspace.is_dir():
        raise ValueError("sweep runtime workspace is unsafe")
    for path in paths:
        if Path(path).is_symlink() or not Path(path).is_file():
            raise ValueError("sweep runtime base request is unsafe")
        try:
            Path(path).relative_to(workspace)
        except ValueError as error:
            raise ValueError("sweep runtime base request escapes workspace") from error
    overlay_value = _canonical_object(Path(overlay), "sweep train overlay")
    if (
        overlay_value.get("experiment_id") != job.experiment_id
        or overlay_value.get("job_digest") != job.experiment_id
        or overlay_value.get("training", {}).get("target_step") != job.training.target_step
        or overlay_value.get("base_train_request_sha256") != sha256_file(base_train_request)
    ):
        raise ValueError("sweep train overlay does not bind the current job")
    if job.training.target_step == 500:
        if promoted_parent is not None or overlay_value.get("parent_publication") is not None:
            raise ValueError("root 500-step sweep request cannot use a promoted parent")
        parent_path = None
        parent_policy_sha = job.parent_checkpoint["artifact_sha256"]
    else:
        if promoted_parent is None:
            raise ValueError("promoted sweep request requires a hydrated parent")
        verify_promoted_parent_hydration(
            job,
            publication=promoted_parent.publication,
            archive=promoted_parent.archive,
            descriptor_path=promoted_parent.descriptor,
        )
        try:
            parent_relative = promoted_parent.checkpoint_path.relative_to(workspace / "cache")
        except ValueError as error:
            raise ValueError("promoted sweep checkpoint is outside the cache mount") from error
        from lehome_train.groot.checkpoint_identity import policy_artifact_sha256
        parent_path = "/cache/" + parent_relative.as_posix()
        parent_policy_sha = policy_artifact_sha256(promoted_parent.checkpoint_path)
    base_launch = _canonical_object(Path(base_launch_config), "sweep base launch config")
    base_experiment = _canonical_object(Path(base_experiment_config), "sweep base experiment config")
    base_train = _canonical_object(Path(base_train_request), "sweep base train request")
    if set(base_train) != {"schema_version", "command", "arguments"} or base_train.get("schema_version") != 1 or base_train.get("command") != "runtime-mixture-train" or not isinstance(base_train.get("arguments"), dict):
        raise ValueError("sweep base train request is not a runtime-mixture command")
    generated = workspace / "prepared" / "sweep" / job.experiment_id
    launch_path = generated / "launch.json"
    experiment_path = generated / "experiment.json"
    train_path = generated / "runtime-train.json"
    binding_path = generated / "request-binding.json"
    launch = dict(base_launch)
    launch.update({
        "experiment_name": job.experiment_id,
        "max_steps": job.training.target_step,
        "save_steps": 500,
        "output_dir": f"/output/sweep/{job.experiment_id}",
        "parent_checkpoint_repository": job.parent_checkpoint["repository"],
        "parent_checkpoint_revision": job.parent_checkpoint["revision"],
        "parent_checkpoint_subpath": job.parent_checkpoint["subpath"],
        "parent_checkpoint_artifact_sha256": parent_policy_sha,
    })
    if parent_path is not None:
        launch["base_model_path"] = parent_path
    # This is decoded to SweepRuntimeProfile by ProductionRuntime.  It is not
    # accepted by the legacy final-2K branch.
    launch["runtime_sweep_profile"] = {
        "schema_version": 1,
        "kind": "lehome_sweep_runtime_profile",
        "mixture_weights": {
            "bc": job.mixture.bc_percent,
            "rollout": job.mixture.added_percent,
            "dagger": 0,
        },
        "training": {
            "action_horizon": 16,
            "global_batch_size": 64,
            "target_step": job.training.target_step,
            "save_steps": 500,
            "terminal_publish": True,
        },
    }
    experiment = dict(base_experiment)
    experiment.update({
        "repository_commit": job.trainer["code_revision"],
        "container_digest": job.trainer["oci_digest"],
        "sample_presentations": job.training.target_step * 64,
        "physical_batch_size": 64,
        "gradient_accumulation_steps": 1,
        "action_horizon": 16,
    })
    arguments = dict(base_train["arguments"])
    arguments.update({
        "launch_config": "/prepared/sweep/%s/launch.json" % job.experiment_id,
        "experiment_config": "/prepared/sweep/%s/experiment.json" % job.experiment_id,
        "result_output": "/output/sweep/%s/runtime-train-result.json" % job.experiment_id,
        "status_output": "/output/sweep/%s/runtime-train-status.json" % job.experiment_id,
        # Sweep continuations consume their full parent archive through the
        # authenticated runtime environment and stage it privately below the
        # child output root immediately before Trainer.train.  These legacy
        # final-2K resume fields stay absent so the two resume protocols cannot
        # be confused or mixed.
        "runtime_resume_archive": None,
        "runtime_resume_descriptor": None,
        "runtime_resume_cursor": None,
        "runtime_resume_anchor": None,
        "runtime_resume_publication": None,
    })
    train = {"schema_version": 1, "command": "runtime-mixture-train", "arguments": arguments}
    _write_private_canonical(launch_path, launch)
    _write_private_canonical(experiment_path, experiment)
    _write_private_canonical(train_path, train)
    binding: dict[str, object] = {
        "schema_version": 1,
        "kind": "lehome_sweep_runtime_request_binding",
        "experiment_id": job.experiment_id,
        "overlay_sha256": sha256_file(overlay),
        "launch_config_sha256": sha256_file(launch_path),
        "experiment_config_sha256": sha256_file(experiment_path),
        "train_request_sha256": sha256_file(train_path),
        "target_step": job.training.target_step,
        "parent_publication": None if promoted_parent is None else dict(promoted_parent.publication),
        "parent_checkpoint_path": parent_path,
        "parent_cursor": None if promoted_parent is None else dict(promoted_parent.cursor),
    }
    _write_private_canonical(binding_path, binding)
    return SweepRuntimeRequest(
        launch_path, experiment_path, train_path, binding_path,
        workspace / "output" / "sweep" / job.experiment_id / "runtime-train-result.json",
    )


def _promoted_parent_publication(
    job: object, publication: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Validate the immutable publication reference before local hydration."""
    parent = getattr(job, "parent_checkpoint", None)
    training = getattr(job, "training", None)
    target_step = getattr(training, "target_step", None)
    if not isinstance(parent, Mapping) or target_step not in (500, 1000, 2000):
        raise ValueError("sweep job parent or target is invalid")
    if target_step == 500:
        if publication is not None:
            raise ValueError("root 500-step sweep job must not name a parent publication")
        return None
    if not isinstance(publication, Mapping):
        raise ValueError("promoted parent publication is not the exact prior rung")
    try:
        parsed = parse_checkpoint_publication(publication)
    except ValueError as error:
        raise ValueError("promoted parent publication is not canonical schema-v2") from error
    step = parsed.target_step
    if (
        publication.get("schema_version") != 2
        or parsed.repository != parent.get("repository")
        or parsed.immutable_revision != parent.get("revision")
        or parsed.remote_prefix != parent.get("subpath")
        or parsed.artifact_sha256 != parent.get("artifact_sha256")
        or parsed.receipt_sha256 != parent.get("receipt_sha256")
        or step not in (500, 1000)
        or step >= target_step
    ):
        raise ValueError("promoted parent publication is not the exact prior rung")
    if (
        parsed.relative_path is None
        or parsed.artifact_byte_size is None
        or parsed.descriptor_relative_path is None
        or parsed.descriptor_sha256 is None
        or parsed.descriptor_byte_size is None
    ):
        raise ValueError("promoted parent publication lacks immutable archive metadata")
    return dict(parsed.canonical)


def verify_promoted_parent_hydration(
    job: object,
    *,
    publication: Mapping[str, object],
    archive: Path | None,
    descriptor_path: Path | None,
) -> None:
    """Require actual read-back checkpoint bytes before a promoted launch.

    This is intentionally a validation boundary, not a downloader.  The
    separate parent-checkpoint hydrator must obtain these two files from the
    exact immutable publication and call this before it writes a runnable
    train-request overlay.
    """
    parent = getattr(job, "parent_checkpoint", None)
    training = getattr(job, "training", None)
    target_step = getattr(training, "target_step", None)
    if not isinstance(parent, Mapping) or target_step not in (1000, 2000):
        raise ValueError("promoted parent hydration requires a 1K or 2K child job")
    canonical_publication = _promoted_parent_publication(job, publication)
    if archive is None or descriptor_path is None:
        raise ValueError("promoted parent archive and descriptor must be hydrated before launch")
    archive, descriptor_path = Path(archive), Path(descriptor_path)
    if (
        archive.is_symlink() or descriptor_path.is_symlink()
        or not archive.is_file() or not descriptor_path.is_file()
        or sha256_file(archive) != canonical_publication["artifact_sha256"]
        or archive.stat().st_size != canonical_publication["artifact_byte_size"]
        or sha256_file(descriptor_path) != canonical_publication["descriptor_sha256"]
        or descriptor_path.stat().st_size != canonical_publication["descriptor_byte_size"]
        or parent.get("artifact_sha256") != canonical_publication["artifact_sha256"]
    ):
        raise ValueError("promoted parent hydration does not match immutable publication bytes")
    descriptor = load_checkpoint_descriptor(descriptor_path)
    record = descriptor.record
    if (
        record.optimizer_step != canonical_publication["target_step"]
        or not record.resumable
        or record.artifact.sha256 != canonical_publication["artifact_sha256"]
        or record.artifact.byte_size != canonical_publication["artifact_byte_size"]
    ):
        raise ValueError("promoted parent descriptor does not bind hydrated archive")


def _strict_archive_member_path(name: str) -> tuple[str, ...]:
    path = PurePosixPath(name)
    # TarInfo.name may retain a trailing slash, so inspect its literal pieces
    # instead of relying on PurePosixPath normalisation alone.
    if (
        path.is_absolute()
        or "\\" in name
        or any(piece in {"", ".", ".."} for piece in name.split("/"))
        or not path.parts
    ):
        raise ValueError("promoted parent archive contains an unsafe member")
    return path.parts


def _extract_promoted_policy(
    archive: Path,
    *,
    optimizer_step: int,
    destination: Path,
) -> Path:
    """Extract exactly one checkpoint tree without trusting tar paths.

    Published sweep artifacts include the run root and a `checkpoint-N`
    directory.  A continuation needs the latter as a model directory, not an
    arbitrary archive member nor an archive-mounted path.  The extraction is
    intentionally standalone so a malformed archive fails before the trainer
    sees any gradients.
    """
    if archive.is_symlink() or not archive.is_file() or destination.exists():
        raise ValueError("promoted parent extraction destination is unsafe")
    checkpoint_name = f"checkpoint-{optimizer_step}"
    try:
        opened = tarfile.open(archive, "r")
    except (OSError, tarfile.TarError) as error:
        raise ValueError("promoted parent archive is unreadable") from error
    with opened:
        members = opened.getmembers()
        if not members:
            raise ValueError("promoted parent archive is empty")
        selected: list[tuple[tarfile.TarInfo, tuple[str, ...]]] = []
        roots: set[str] = set()
        seen: set[tuple[str, ...]] = set()
        for member in members:
            parts = _strict_archive_member_path(member.name)
            if (
                parts in seen
                or member.issym()
                or member.islnk()
                or not (member.isdir() or member.isfile())
            ):
                raise ValueError("promoted parent archive contains an unsafe member")
            seen.add(parts)
            roots.add(parts[0])
            if len(parts) >= 2 and parts[1] == checkpoint_name:
                selected.append((member, parts[2:]))
        if len(roots) != 1 or not selected:
            raise ValueError("promoted parent archive lacks its exact checkpoint")
        # The checkpoint root itself must be an explicit directory.  This
        # prevents a tar with only files under an implicit path from becoming
        # a surprising model root.
        if not any(not relative and member.isdir() for member, relative in selected):
            raise ValueError("promoted parent archive checkpoint root is missing")
        destination.mkdir(mode=0o700)
        try:
            for member, relative in selected:
                if not relative:
                    continue
                target = destination.joinpath(*relative)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=False, mode=0o700)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                source = opened.extractfile(member)
                if source is None:
                    raise ValueError("promoted parent archive member has no payload")
                with source, target.open("xb") as stream:
                    shutil.copyfileobj(source, stream)
                if target.stat().st_size != member.size or target.is_symlink():
                    raise ValueError("promoted parent archive extraction changed bytes")
        except BaseException:
            shutil.rmtree(destination, ignore_errors=True)
            raise
    if (destination / "model.safetensors").is_symlink():
        raise ValueError("promoted parent checkpoint model is unsafe")
    return destination


def _hub_tree_paths(tree: object) -> set[str]:
    """Normalise the two supported transport tree representations."""
    if not isinstance(tree, (list, tuple)):
        raise ValueError("promoted parent immutable tree is unavailable")
    paths: set[str] = set()
    for entry in tree:
        relative = getattr(entry, "relative_path", None)
        entry_type = getattr(entry, "entry_type", "file")
        if not isinstance(relative, str) or entry_type != "file":
            continue
        paths.add(relative)
    return paths


def hydrate_promoted_parent(
    job: ExperimentJob,
    *,
    publication: Mapping[str, object],
    cache_root: Path,
    hub: ParentCheckpointHub,
) -> PromotedParentMaterialization | None:
    """Fresh-list and atomically materialize the exact promoted parent.

    Root 500-step jobs deliberately return ``None``: their original 12K
    parent is hydrated by the immutable request set.  1K/2K jobs never fall
    back to that parent; a missing or mismatched prior-rung archive aborts
    before the guest controller is launched.
    """
    if job.training.target_step == 500:
        if publication:
            raise ValueError("root 500-step sweep job cannot hydrate a promoted parent")
        return None
    publication = _promoted_parent_publication(job, publication)
    cache_root = Path(cache_root)
    if not cache_root.is_absolute() or cache_root.is_symlink():
        raise ValueError("promoted parent cache root is unsafe")
    cache_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if cache_root.is_symlink() or not cache_root.is_dir():
        raise ValueError("promoted parent cache root is unsafe")
    artifact_sha = str(publication["artifact_sha256"])
    descriptor_sha = str(publication["descriptor_sha256"])
    final = cache_root / f"{artifact_sha}-{descriptor_sha}"
    archive_name = Path(str(publication["relative_path"])).name
    descriptor_name = Path(str(publication["descriptor_relative_path"])).name

    def verify_final(root: Path) -> PromotedParentMaterialization:
        archive = root / "archive" / archive_name
        descriptor = root / "descriptor" / descriptor_name
        checkpoint = root / "checkpoint"
        verify_promoted_parent_hydration(
            job, publication=publication, archive=archive, descriptor_path=descriptor,
        )
        if checkpoint.is_symlink() or not checkpoint.is_dir():
            raise ValueError("promoted parent checkpoint materialization is missing")
        # Re-extracting is deliberately avoided: the evidence records the
        # immutable archive/descriptor and exact checkpoint path.  Require a
        # model layout now so cached corruption cannot reach launch.
        from lehome_train.groot.checkpoint_identity import policy_artifact_sha256
        policy_artifact_sha256(checkpoint)
        step = int(publication["target_step"])
        return PromotedParentMaterialization(
            archive=archive,
            descriptor=descriptor,
            checkpoint_path=checkpoint,
            cursor=MappingProxyType({
                "optimizer_step": step,
                "global_sample_offset": step * 64,
                "physical_batch_size": 64,
                "action_horizon": 16,
            }),
            publication=MappingProxyType(dict(publication)),
        )

    if final.exists():
        if final.is_symlink() or not final.is_dir():
            raise ValueError("promoted parent cached materialization is unsafe")
        return verify_final(final)
    prefix = str(publication["remote_prefix"])
    archive_relative = _safe_relative(str(publication["relative_path"]), "promoted parent archive path")
    descriptor_relative = _safe_relative(str(publication["descriptor_relative_path"]), "promoted parent descriptor path")
    remote_archive = f"{prefix}/{archive_relative}"
    remote_descriptor = f"{prefix}/{descriptor_relative}"
    tree = _hub_tree_paths(hub.list_tree(
        repository=str(publication["repository"]),
        revision=str(publication["immutable_revision"]),
    ))
    if remote_archive not in tree or remote_descriptor not in tree:
        raise ValueError("promoted parent immutable tree lacks archive or descriptor")
    stage = Path(tempfile.mkdtemp(prefix=".promoted-parent-", dir=cache_root))
    try:
        hub.download_files(
            repository=str(publication["repository"]),
            revision=str(publication["immutable_revision"]),
            destination=stage,
            relative_paths=(archive_relative, descriptor_relative),
            remote_prefix=prefix,
        )
        archive = stage / archive_relative
        descriptor = stage / descriptor_relative
        verify_promoted_parent_hydration(
            job, publication=publication, archive=archive, descriptor_path=descriptor,
        )
        staged_final = stage / "materialized"
        archive_target = staged_final / "archive" / archive_name
        descriptor_target = staged_final / "descriptor" / descriptor_name
        archive_target.parent.mkdir(parents=True, mode=0o700)
        descriptor_target.parent.mkdir(parents=True, mode=0o700)
        shutil.copyfile(archive, archive_target, follow_symlinks=False)
        shutil.copyfile(descriptor, descriptor_target, follow_symlinks=False)
        _extract_promoted_policy(
            archive_target,
            optimizer_step=int(publication["target_step"]),
            destination=staged_final / "checkpoint",
        )
        # Validate the final tree before rename, then atomically publish it.
        verify_final(staged_final)
        try:
            os.replace(staged_final, final)
        except FileExistsError:
            pass
        return verify_final(final)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _tree_sha(entries: list[dict[str, object]]) -> str:
    return canonical_json_sha256(entries)


@dataclass(frozen=True, slots=True)
class RuntimeRequestSet:
    root: Path
    environment: Mapping[str, str]
    result_output: str
    compatibility_profile_sha256: str | None = None


def _verify(root: Path, job: ExperimentJob, source: ArtifactBinding) -> RuntimeRequestSet:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("hydrated request set is unsafe")
    manifest_path = root / "bundle-manifest.json"
    if sha256_file(manifest_path) != source.manifest_sha256:
        raise ValueError("runtime request-set manifest SHA-256 mismatch")
    manifest = _strict_json(manifest_path, "runtime request-set manifest")
    v1_fields = {"schema_version", "kind", "runtime_profile", "runtime_profile_sha256", "environment", "result_output", "files", "tree_sha256"}
    v2_fields = {"schema_version", "kind", "runtime_compatibility_profile", "runtime_compatibility_profile_sha256", "environment", "result_output", "files", "tree_sha256"}
    is_v2 = set(manifest) == v2_fields
    expected_profile = (
        runtime_compatibility_profile_document(job)
        if is_v2 else runtime_profile_document(job)
    )
    expected_profile_sha256 = (
        runtime_compatibility_profile_sha256(job)
        if is_v2 else runtime_profile_sha256(job)
    )
    profile_key = "runtime_compatibility_profile" if is_v2 else "runtime_profile"
    profile_sha_key = profile_key + "_sha256"
    if (
        set(manifest) not in (v1_fields, v2_fields)
        or manifest.get("schema_version") != 1
        or manifest.get("kind") != "lehome_runtime_request_set"
        or manifest.get("tree_sha256") != source.tree_sha256
        or manifest.get(profile_key) != expected_profile
        or manifest.get(profile_sha_key) != expected_profile_sha256
    ):
        raise ValueError("runtime request-set manifest is invalid")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("runtime request-set manifest files are invalid")
    expected: dict[str, dict[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "byte_size", "sha256"}:
            raise ValueError("runtime request-set entry is invalid")
        path = _safe_relative(entry["path"], "runtime request-set path")
        if path in expected or type(entry["byte_size"]) is not int or entry["byte_size"] < 0 or type(entry["sha256"]) is not str or len(entry["sha256"]) != 64:
            raise ValueError("runtime request-set entry is invalid")
        expected[path] = dict(entry)
    actual: dict[str, Path] = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError("runtime request-set contains a symlink")
        if path.is_file() and relative != "bundle-manifest.json":
            actual[relative] = path
        elif not path.is_dir() and not path.is_file():
            raise ValueError("runtime request-set contains an unsafe file type")
    if set(actual) != set(expected):
        raise ValueError("runtime request-set file listing mismatch")
    normalized: list[dict[str, object]] = []
    for relative, path in sorted(actual.items()):
        entry = expected[relative]
        if path.stat().st_size != entry["byte_size"] or sha256_file(path) != entry["sha256"]:
            raise ValueError("runtime request-set file digest mismatch")
        normalized.append({"path": relative, "byte_size": entry["byte_size"], "sha256": entry["sha256"]})
    if _tree_sha(normalized) != source.tree_sha256:
        raise ValueError("runtime request-set tree SHA-256 mismatch")
    environment = manifest.get("environment")
    if not isinstance(environment, dict) or set(environment) != _ENV_KEYS:
        raise ValueError("runtime request-set environment is invalid")
    for key, value in environment.items():
        if key.endswith("SHA256"):
            if type(value) is not str or len(value) != 64:
                raise ValueError("runtime request-set descriptor hash is invalid")
        elif key == "LEHOME_CODE_REVISION":
            if type(value) is not str or len(value) != 40:
                raise ValueError("runtime request-set descriptor code revision is invalid")
        else:
            relative = _safe_relative(value, "runtime environment path")
            if relative not in actual:
                raise ValueError("runtime environment path is not in request set")
    result_output = _safe_relative(manifest.get("result_output"), "runtime result output")
    if not result_output.startswith("output/"):
        raise ValueError("runtime result output must be beneath output")
    return RuntimeRequestSet(
        root, MappingProxyType({key: str(value) for key, value in environment.items()}),
        result_output, expected_profile_sha256 if is_v2 else None,
    )


def materialize_request_set(job: ExperimentJob, cache_root: Path, *, hydrator: RequestSetHydrator) -> RuntimeRequestSet:
    """Hydrate a pinned request set into a cache and atomically expose it."""
    source = _runtime_source(job)
    cache = Path(cache_root)
    if not cache.is_absolute() or cache.is_symlink():
        raise ValueError("runtime request-set cache root is unsafe")
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    final = cache / source.tree_sha256
    if final.exists():
        return _verify(final, job, source)
    stage = Path(tempfile.mkdtemp(prefix=".request-set-", dir=cache))
    try:
        hydrated = Path(hydrator.hydrate(source, stage))
        if hydrated != stage:
            raise ValueError("hydrator did not return its staged request-set root")
        verified = _verify(stage, job, source)
        try:
            os.replace(stage, final)
        except FileExistsError:
            shutil.rmtree(stage)
        return _verify(final, job, source)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


class HuggingFaceRequestSetHydrator:
    """Production transport boundary; network calls live only behind this type."""

    def __init__(self, token_file: Path, *, downloader: Any | None = None) -> None:
        self.token_file, self.downloader = Path(token_file), downloader

    def hydrate(self, source: ArtifactBinding, destination: Path) -> Path:
        if self.token_file.is_symlink() or not self.token_file.is_file() or stat.S_IMODE(self.token_file.stat().st_mode) != 0o600:
            raise ValueError("Hugging Face token file is unsafe")
        token = self.token_file.read_text(encoding="utf-8").strip()
        if not token:
            raise ValueError("Hugging Face token is empty")
        if self.downloader is None:
            from huggingface_hub import snapshot_download
            downloader = snapshot_download
        else:
            downloader = self.downloader
        downloaded = Path(downloader(repo_id=source.repository, revision=source.revision, token=token, allow_patterns=[source.prefix + "/**"], local_dir=str(destination / "download"), local_dir_use_symlinks=False))
        payload = downloaded / source.prefix
        if payload.is_symlink() or not payload.is_dir():
            raise ValueError("Hugging Face request-set prefix is missing")
        for child in payload.iterdir():
            shutil.move(str(child), destination / child.name)
        shutil.rmtree(destination / "download", ignore_errors=True)
        return destination


class HuggingFacePromotedParentHub:
    """Read-only pinned-Hub adapter used only after a controller lease."""

    def __init__(self, token_file: Path, *, transport: object | None = None) -> None:
        self.token_file = Path(token_file)
        self.transport = transport

    def _token(self) -> str:
        if (
            self.token_file.is_symlink()
            or not self.token_file.is_file()
            or stat.S_IMODE(self.token_file.stat().st_mode) != 0o600
        ):
            raise ValueError("Hugging Face token file is unsafe")
        token = self.token_file.read_text(encoding="utf-8").strip()
        if not token:
            raise ValueError("Hugging Face token is empty")
        return token

    def _transport(self) -> object:
        if self.transport is not None:
            return self.transport
        from lehome_train.hub import HuggingFaceHubTransport
        return HuggingFaceHubTransport(timeout_seconds=21600.0)

    def list_tree(self, *, repository: str, revision: str) -> object:
        transport = self._transport()
        return transport.list_tree(repository=repository, revision=revision, token=self._token())

    def download_files(
        self,
        *,
        repository: str,
        revision: str,
        destination: Path,
        relative_paths: tuple[str, ...],
        remote_prefix: str,
    ) -> object:
        from lehome_train.hub import download_files
        return download_files(
            transport=self._transport(),
            repository=repository,
            revision=revision,
            destination=destination,
            relative_paths=relative_paths,
            remote_prefix=remote_prefix,
            environ={"HF_TOKEN": self._token()},
            max_attempts=3,
        )


def copy_request_set_to_workspace(request_set: RuntimeRequestSet, workspace_root: Path) -> Path:
    workspace = Path(workspace_root)
    if workspace.exists() or workspace.is_symlink() or not workspace.is_absolute():
        raise ValueError("runtime workspace target is unsafe")
    stage = workspace.with_name(workspace.name + ".staging")
    if stage.exists() or stage.is_symlink():
        raise ValueError("runtime workspace staging target is unsafe")
    shutil.copytree(request_set.root, stage, symlinks=False)
    for path in stage.rglob("*"):
        if path.is_symlink():
            shutil.rmtree(stage, ignore_errors=True)
            raise ValueError("runtime workspace contains a symlink")
    for name in ("prepared", "cache", "output", "private"):
        (stage / name).mkdir(exist_ok=True, mode=0o700)
    os.replace(stage, workspace)
    return workspace


def build_runtime_environment(
    job: ExperimentJob,
    request_set: RuntimeRequestSet,
    workspace: Path,
    *,
    hf_token_file: Path,
    sweep_overlay: Path | None = None,
    promoted_parent: PromotedParentMaterialization | None = None,
    sweep_runtime_request: SweepRuntimeRequest | None = None,
) -> tuple[Path, Path]:
    """Generate a private env file from checked descriptor values only."""
    if hf_token_file.is_symlink() or not hf_token_file.is_file() or stat.S_IMODE(hf_token_file.stat().st_mode) != 0o600:
        raise ValueError("runtime Hugging Face token file is unsafe")
    environment = request_set.environment
    result_relative = request_set.result_output
    env: dict[str, str] = {
        "LEHOME_ROLE": "training",
        "LEHOME_RUN_ID": job.experiment_id,
        "LEHOME_WORKSPACE_MOUNT": str(workspace),
        "LEHOME_PREPARED_ROOT": str(workspace / "prepared"),
        "LEHOME_CACHE_ROOT": str(workspace / "cache"),
        "LEHOME_OUTPUT_ROOT": str(workspace / "output"),
        "LEHOME_HF_TOKEN_FILE": str(hf_token_file),
        "LEHOME_EXPERIMENT_JOB_DIGEST": job.experiment_id,
        "LEHOME_CODE_REVISION": job.trainer["code_revision"],
    }
    for key, value in environment.items():
        if key == "LEHOME_CODE_REVISION":
            if value != job.trainer["code_revision"]:
                raise ValueError("runtime descriptor code revision does not bind job")
            continue
        if key.endswith("SHA256"):
            env[key] = str(value)
        else:
            env[key] = str(workspace / str(value))
    if sweep_overlay is not None:
        sweep_overlay = Path(sweep_overlay)
        try:
            sweep_relative = sweep_overlay.relative_to(workspace).as_posix()
        except ValueError as error:
            raise ValueError("sweep train overlay escapes workspace") from error
        _safe_relative(sweep_relative, "sweep train overlay")
        if sweep_overlay.is_symlink() or not sweep_overlay.is_file():
            raise ValueError("sweep train overlay is missing or unsafe")
        env["LEHOME_SWEEP_TRAIN_OVERLAY"] = str(sweep_overlay)
        env["LEHOME_SWEEP_TRAIN_OVERLAY_SHA256"] = sha256_file(sweep_overlay)
        if sweep_runtime_request is None:
            raise ValueError("sweep train overlay requires dynamic runtime request bytes")
        for path, key in (
            (sweep_runtime_request.train_request, "LEHOME_SWEEP_TRAIN_REQUEST"),
            (sweep_runtime_request.binding, "LEHOME_SWEEP_RUNTIME_BINDING"),
        ):
            try:
                path.relative_to(workspace)
            except ValueError as error:
                raise ValueError("sweep runtime request escapes workspace") from error
            if path.is_symlink() or not path.is_file():
                raise ValueError("sweep runtime request is missing or unsafe")
            env[key] = str(path)
            env[key + "_SHA256"] = sha256_file(path)
        if job.training.target_step == 500:
            if promoted_parent is not None:
                raise ValueError("root sweep job cannot inject a promoted parent")
        else:
            if promoted_parent is None:
                raise ValueError("promoted sweep job has no hydrated parent")
            for path, label in (
                (promoted_parent.archive, "archive"),
                (promoted_parent.descriptor, "descriptor"),
                (promoted_parent.checkpoint_path, "checkpoint"),
            ):
                try:
                    path.relative_to(workspace)
                except ValueError as error:
                    raise ValueError(f"promoted parent {label} escapes workspace") from error
            env["LEHOME_SWEEP_PARENT_ARCHIVE"] = str(promoted_parent.archive)
            env["LEHOME_SWEEP_PARENT_DESCRIPTOR"] = str(promoted_parent.descriptor)
            env["LEHOME_SWEEP_PARENT_CHECKPOINT"] = str(promoted_parent.checkpoint_path)
    descriptor = {
        "schema_version": 1,
        "kind": "lehome_runtime_request_set_descriptor",
        "job_digest": job.experiment_id,
        "target_step": job.training.target_step,
        "sweep_overlay_sha256": None if sweep_overlay is None else sha256_file(sweep_overlay),
        "environment": dict(environment),
        "result_output": result_relative,
    }
    descriptor_path = workspace / "private" / "runtime-request-descriptor.json"
    descriptor_path.write_bytes(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode())
    descriptor_path.chmod(0o600)
    env_path = workspace / "private" / "runtime.env"
    # Values are validated paths/hashes/identities, then quoted by Python;
    # remote descriptor text is never sourced as arbitrary shell content.
    import shlex
    env_path.write_text("".join(f"{key}={shlex.quote(value)}\n" for key, value in sorted(env.items())), encoding="utf-8")
    env_path.chmod(0o600)
    result = workspace / result_relative
    if sweep_runtime_request is not None:
        result = sweep_runtime_request.result_output
    return env_path, result


def publication_from_result(job: ExperimentJob, result_output: Path) -> dict[str, object]:
    payload = _strict_json(result_output, "runtime train result")
    publications = payload.get("immutable_checkpoint_publications")
    if not isinstance(publications, list):
        raise ValueError("runtime train result has no immutable publications")
    matches = [value for value in publications if isinstance(value, dict) and value.get("optimizer_step") == job.training.target_step]
    if len(matches) != 1:
        raise ValueError("runtime train result lacks the target-step publication")
    value = matches[0]
    required = {
        "repository", "immutable_revision", "remote_prefix", "relative_path",
        "artifact_sha256", "artifact_byte_size", "descriptor_relative_path",
        "descriptor_sha256", "descriptor_byte_size", "readback_verified",
    }
    if not required.issubset(value) or value.get("readback_verified") is not True:
        raise ValueError("runtime target publication is not read-back verified")
    envelope = {
        "schema_version": 2,
        "experiment_id": job.experiment_id,
        "job_digest": job.experiment_id,
        "target_step": job.training.target_step,
        "repository": value["repository"],
        "immutable_revision": value["immutable_revision"],
        "remote_prefix": value["remote_prefix"],
        "relative_path": value["relative_path"],
        "artifact_sha256": value["artifact_sha256"],
        "artifact_byte_size": value["artifact_byte_size"],
        "descriptor_relative_path": value["descriptor_relative_path"],
        "descriptor_sha256": value["descriptor_sha256"],
        "descriptor_byte_size": value["descriptor_byte_size"],
        "receipt_sha256": sha256_file(result_output),
        "readback_verified": True,
    }
    return dict(bind_checkpoint_publication(job, envelope["receipt_sha256"], envelope).canonical)
