"""Pre-rent, secret-safe fixed-path checks for Behavior 1K."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Mapping, Protocol
from uuid import uuid4

from lehome_train.constants import BEHAVIOR_1K_CHECKPOINT_BUCKET, BEHAVIOR_1K_DATASET_REPOSITORY, BEHAVIOR_1K_DATASET_REVISION, BEHAVIOR_1K_FINAL_MODEL_REPOSITORY, COSMOS_REPOSITORY, COSMOS_REVISION, MODEL_REVISION
from lehome_train.io import atomic_write_json, canonical_json_sha256, sha256_file
from lehome_train.b1k.models import derive_groot_config
from lehome_train.b1k.training import SUPPORTED_GPU_COUNTS
from lehome_train.b1k.bucket_protocol import BucketHelperClient, BucketNotFound
from lehome_train.b1k.snapshot_integrity import RemoteManifest, SnapshotValidation, allow_patterns_sha256, build_remote_manifest, build_snapshot_receipt, read_snapshot_json, validate_local_snapshot, validate_snapshot_receipt, verify_artifact_stat_invariants
from lehome_train.b1k.snapshot_state import bound_destination, destination_lock, fsync_directory, open_staged_destination, validate_destination_binding
from lehome_train.groot.model_snapshot import BASE_MODEL_REPOSITORY


_GPU_NAMES = {
    "NVIDIA RTX PRO 6000 Blackwell Server", "NVIDIA RTX PRO 6000 Blackwell Workstation",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition", "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
}
_MINIMUM_DISK = 1_500_000_000_000
_RGB = (
    "observation.rgb.left_realsense_link_camera_0",
    "observation.rgb.right_realsense_link_camera_0",
    "observation.rgb.zed_link_camera_0",
)
_SMOKE_LOCAL_ROOT = Path("/workspace/checkpoints")
_SMOKE_BYTES = b"b1k-remote-access-smoke-v1\n"
_SMOKE_CLEANUP_ATTEMPTS = 3
_SMOKE_CLEANUP_INTERVAL_SECONDS = 0.2
_SNAPSHOT_RECEIPT = ".b1k-snapshot-receipt.json"
_SNAPSHOT_INTENT = ".b1k-snapshot-intent.json"


class ModelSmokeObjectNotFound(ValueError):
    """The exact temporary model-repository smoke object is absent."""


@dataclass(frozen=True, slots=True)
class WorkspacePaths:
    root: Path
    dataset: Path
    groot_upstream: Path
    cosmos: Path
    derived_model: Path
    output: Path
    checkpoints: Path
    logs: Path
    final: Path

    @classmethod
    def from_root(cls, root: str | Path, *, run_id: str) -> "WorkspacePaths":
        root = Path(root)
        if root != Path("/workspace") or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", run_id): raise ValueError("B1K workspace paths are fixed")
        return cls(root, root / "data/b1k", root / "models/groot-upstream", root / "models/cosmos", root / "models/groot", root / "outputs" / run_id, root / "checkpoints", root / "logs", root / "final")

    def create(self) -> None:
        # Snapshot destinations must not be pre-created: their existence is the
        # completed-cache signal and is always receipt-validated.
        for path in (self.root, self.dataset.parent, self.groot_upstream.parent, self.cosmos.parent, self.output, self.checkpoints, self.logs, self.final):
            path.mkdir(parents=True, exist_ok=True)


def require_hardware(nvidia_smi_output: str, *, free_bytes: int) -> None:
    if type(free_bytes) is not int or free_bytes < _MINIMUM_DISK: raise ValueError("B1K requires at least 1.5 TB free disk")
    devices: list[int] = []
    for line in nvidia_smi_output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2 or fields[0] not in _GPU_NAMES or not fields[1].isdigit(): raise ValueError("nvidia-smi CSV output is invalid")
        devices.append(int(fields[1]))
    if len(devices) not in SUPPORTED_GPU_COUNTS or any(memory < 98_304 for memory in devices): raise ValueError("B1K requires one to four RTX PRO 6000 Blackwell GPUs with at least 96 GiB")


def read_hf_token(path: str | Path = "/workspace/.cache/huggingface/token") -> str:
    token_path = Path(path)
    try: stat = token_path.lstat()
    except OSError as error: raise ValueError("Hugging Face token file is missing") from error
    if token_path.is_symlink() or not token_path.is_file() or stat.st_uid != os.getuid() or stat.st_mode & 0o7777 != 0o600: raise ValueError("Hugging Face token file is unsafe")
    token = token_path.read_text(encoding="utf-8").strip()
    if not token: raise ValueError("Hugging Face token file is empty")
    return token


def dataset_snapshot_patterns() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the only allowed metadata and full-dataset Hub patterns."""
    metadata = ("meta/**", "annotations/skill_summary.csv", "annotations/skill_type_summary.csv")
    return metadata, ("meta/**", "data/**", "annotations/**", *(f"videos/{camera}/**" for camera in _RGB))


def _staging_path(destination: Path) -> Path:
    return destination.with_name("." + destination.name + ".incomplete")


def _manifest_payload(value: object) -> object:
    return value.to_dict() if hasattr(value, "to_dict") else value


def _manifest_hash(value: object) -> str:
    payload = _manifest_payload(value)
    if isinstance(payload, Mapping) and type(payload.get("fingerprint")) is str:
        unhashed = dict(payload)
        fingerprint = unhashed.pop("fingerprint")
        if fingerprint != canonical_json_sha256(unhashed):
            raise ValueError("manifest fingerprint does not match its canonical payload")
        return fingerprint
    return canonical_json_sha256(payload)


def _allow_patterns_fingerprint(allow_patterns: tuple[str, ...] | None) -> str:
    return allow_patterns_sha256(allow_patterns)


def _read_snapshot_json(path: Path, label: str) -> dict[str, Any]:
    return read_snapshot_json(path, label)


def _snapshot_identity(repository: str, revision: str, allow_patterns: tuple[str, ...] | None) -> dict[str, object]:
    return {
        "repository": repository,
        "revision": revision,
        "allow_patterns_sha256": _allow_patterns_fingerprint(allow_patterns),
    }


def _validate_snapshot_receipt(
    destination: Path,
    *,
    repository: str,
    revision: str,
    allow_patterns: tuple[str, ...] | None,
    remote_manifest: RemoteManifest,
    manifest_hashes: Mapping[str, str] | None = None,
    validation: SnapshotValidation | None = None,
    local_derived_paths: tuple[str, ...] = (),
) -> tuple[dict[str, Any], SnapshotValidation]:
    """Return a receipt backed by one authoritative, reusable local hash table."""

    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("completed snapshot directory is unsafe")
    receipt_path = destination / _SNAPSHOT_RECEIPT
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("completed snapshot receipt is missing")
    table = validation or validate_local_snapshot(destination, remote_manifest, local_derived_paths=local_derived_paths)
    receipt = validate_snapshot_receipt(
        receipt_path,
        repository=repository,
        revision=revision,
        allow_patterns=allow_patterns,
        remote_manifest=remote_manifest,
        validation=table,
        manifest_hashes=manifest_hashes,
    )
    verify_artifact_stat_invariants(destination, table.artifacts, local_derived_paths=local_derived_paths)
    return receipt, table


def _open_snapshot_staging(destination: Path, *, repository: str, revision: str, allow_patterns: tuple[str, ...] | None) -> tuple[Path, bool]:
    """Open a resumable deterministic stage, never trusting an unreceipted final."""

    identity = _snapshot_identity(repository, revision, allow_patterns)
    return open_staged_destination(
        destination,
        intent_name=_SNAPSHOT_INTENT,
        identity=identity,
        read_intent=lambda path: _read_snapshot_json(path, "snapshot incomplete staging receipt"),
        label="snapshot",
    )


def _promote_snapshot(
    staging: Path,
    destination: Path,
    *,
    repository: str,
    revision: str,
    allow_patterns: tuple[str, ...] | None,
    remote_manifest: RemoteManifest,
    manifest_hashes: Mapping[str, str] | None = None,
    validation: SnapshotValidation | None = None,
    local_derived_paths: tuple[str, ...] = (),
    binding_destination: Path | None = None,
) -> tuple[Path, SnapshotValidation]:
    def validate_binding() -> None:
        if binding_destination is not None:
            validate_destination_binding(binding_destination)

    validate_binding()
    if destination.exists() or destination.is_symlink():
        raise ValueError("completed snapshot destination already exists")
    identity = _snapshot_identity(repository, revision, allow_patterns)
    intent = _read_snapshot_json(staging / _SNAPSHOT_INTENT, "snapshot incomplete staging receipt")
    if intent != identity:
        raise ValueError("snapshot incomplete staging receipt does not match")
    validation = validation or validate_local_snapshot(staging, remote_manifest, local_derived_paths=local_derived_paths)
    receipt = build_snapshot_receipt(
        repository=repository,
        revision=revision,
        allow_patterns=allow_patterns,
        remote_manifest=remote_manifest,
        artifacts=validation.artifacts,
        manifest_hashes={} if manifest_hashes is None else manifest_hashes,
    )
    validate_binding()
    atomic_write_json(staging / _SNAPSHOT_RECEIPT, receipt)
    validate_binding()
    _validate_snapshot_receipt(staging, repository=repository, revision=revision, allow_patterns=allow_patterns, remote_manifest=remote_manifest, manifest_hashes=manifest_hashes, validation=validation, local_derived_paths=local_derived_paths)
    validate_binding()
    os.replace(staging, destination)
    validate_binding()
    fsync_directory(destination.parent)
    validate_binding()
    _validate_snapshot_receipt(destination, repository=repository, revision=revision, allow_patterns=allow_patterns, remote_manifest=remote_manifest, manifest_hashes=manifest_hashes, validation=validation, local_derived_paths=local_derived_paths)
    return destination, validation


def _remote_manifest(hub: "HubAccess", *, repository: str, revision: str, allow_patterns: tuple[str, ...] | None, token: str) -> RemoteManifest:
    query = getattr(hub, "remote_manifest", None)
    if not callable(query):
        raise ValueError("authoritative Hub remote manifest query is unavailable")
    manifest = query(repository, revision=revision, allow_patterns=allow_patterns, token=token)
    if not isinstance(manifest, RemoteManifest):
        raise ValueError("authoritative Hub remote manifest is invalid")
    if manifest.repository != repository or manifest.revision != revision or manifest.allow_patterns_sha256 != _allow_patterns_fingerprint(allow_patterns):
        raise ValueError("authoritative Hub remote manifest identity does not match")
    return manifest


def _ensure_model_snapshot(*, hub: "HubAccess", repository: str, revision: str, destination: Path, token: str, validation_cache: dict[Path, SnapshotValidation] | None = None) -> Path:
    with destination_lock(destination):
        local_dir, completed = _open_snapshot_staging(destination, repository=repository, revision=revision, allow_patterns=None)
        if completed:
            manifest = _remote_manifest(hub, repository=repository, revision=revision, allow_patterns=None, token=token)
            if validation_cache is None or destination not in validation_cache:
                _, validation = _validate_snapshot_receipt(destination, repository=repository, revision=revision, allow_patterns=None, remote_manifest=manifest)
                if validation_cache is not None: validation_cache[destination] = validation
            return local_dir
        hub.snapshot_download(repository, revision=revision, local_dir=bound_destination(destination, local_dir), allow_patterns=None, token=token)
        validate_destination_binding(destination)
        manifest = _remote_manifest(hub, repository=repository, revision=revision, allow_patterns=None, token=token)
        validate_destination_binding(destination)
        _, validation = _promote_snapshot(bound_destination(destination, local_dir), bound_destination(destination), repository=repository, revision=revision, allow_patterns=None, remote_manifest=manifest, binding_destination=destination)
        validate_destination_binding(destination)
        if validation_cache is not None: validation_cache[destination] = validation
        return destination


def _cosmos_snapshot_identity(destination: Path, *, hub: "HubAccess", token: str, validation: SnapshotValidation | None = None) -> dict[str, str]:
    """Bind derivation to the revalidated immutable Cosmos snapshot receipt."""

    manifest = _remote_manifest(hub, repository=COSMOS_REPOSITORY, revision=COSMOS_REVISION, allow_patterns=None, token=token)
    receipt, _ = _validate_snapshot_receipt(
        destination,
        repository=COSMOS_REPOSITORY,
        revision=COSMOS_REVISION,
        allow_patterns=None,
        remote_manifest=manifest,
        validation=validation,
    )
    return {
        "repository": COSMOS_REPOSITORY,
        "revision": COSMOS_REVISION,
        "receipt_sha256": canonical_json_sha256(receipt),
        "artifacts_sha256": canonical_json_sha256(receipt["validated_artifacts"]),
    }


@dataclass(slots=True)
class BootstrapWorkflow:
    """Injectable Hub work: token is passed only as an argument, never retained."""
    snapshot: Callable[[str, str, Path, tuple[str, ...], str], object]
    remote_manifest: Callable[[str, str, tuple[str, ...] | None, str], RemoteManifest]
    build_selection: Callable[[Path], object]
    materialize: Callable[..., object]

    def dataset(self, *, local_dir: Path, token: str) -> None:
        if not token or not local_dir.is_absolute(): raise ValueError("dataset bootstrap inputs are invalid")
        with destination_lock(local_dir):
            self._dataset_locked(local_dir=local_dir, token=token)

    def _dataset_locked(self, *, local_dir: Path, token: str) -> None:
        metadata, full = dataset_snapshot_patterns()
        snapshot_root, completed = _open_snapshot_staging(
            local_dir,
            repository=BEHAVIOR_1K_DATASET_REPOSITORY,
            revision=BEHAVIOR_1K_DATASET_REVISION,
            allow_patterns=full,
        )
        if completed:
            manifest = self.remote_manifest(BEHAVIOR_1K_DATASET_REPOSITORY, BEHAVIOR_1K_DATASET_REVISION, full, token)
            _, validation = _validate_snapshot_receipt(
                local_dir,
                repository=BEHAVIOR_1K_DATASET_REPOSITORY,
                revision=BEHAVIOR_1K_DATASET_REVISION,
                allow_patterns=full,
                remote_manifest=manifest,
            )
            selection = self.build_selection(snapshot_root)
            materialized = _materialize_snapshot(self.materialize, snapshot_root, validation)
            _validate_snapshot_receipt(
                local_dir,
                repository=BEHAVIOR_1K_DATASET_REPOSITORY,
                revision=BEHAVIOR_1K_DATASET_REVISION,
                allow_patterns=full,
                remote_manifest=manifest,
                manifest_hashes={"selection": _manifest_hash(selection), "materialized": _manifest_hash(materialized)},
                validation=validation,
            )
            return
        self.snapshot(BEHAVIOR_1K_DATASET_REPOSITORY, BEHAVIOR_1K_DATASET_REVISION, bound_destination(local_dir, snapshot_root), metadata, token)
        validate_destination_binding(local_dir)
        selection = self.build_selection(snapshot_root)
        self.snapshot(BEHAVIOR_1K_DATASET_REPOSITORY, BEHAVIOR_1K_DATASET_REVISION, bound_destination(local_dir, snapshot_root), full, token)
        validate_destination_binding(local_dir)
        if _manifest_hash(self.build_selection(snapshot_root)) != _manifest_hash(selection):
            raise ValueError("dataset selection changed after payload download")
        manifest = self.remote_manifest(BEHAVIOR_1K_DATASET_REPOSITORY, BEHAVIOR_1K_DATASET_REVISION, full, token)
        validation = validate_local_snapshot(snapshot_root, manifest)
        materialized = _materialize_snapshot(self.materialize, snapshot_root, validation)
        validate_destination_binding(local_dir)
        _promote_snapshot(
            bound_destination(local_dir, snapshot_root),
            bound_destination(local_dir),
            repository=BEHAVIOR_1K_DATASET_REPOSITORY,
            revision=BEHAVIOR_1K_DATASET_REVISION,
            allow_patterns=full,
            remote_manifest=manifest,
            manifest_hashes={"selection": _manifest_hash(selection), "materialized": _manifest_hash(materialized)},
            validation=validation,
            binding_destination=local_dir,
        )
        validate_destination_binding(local_dir)


class HubAccess(Protocol):
    def repo_info(self, repository: str, *, revision: str | None, token: str) -> object: ...
    def bucket_info(self, bucket: str, *, token: str) -> object: ...
    def create_bucket(self, bucket: str, *, private: bool, token: str) -> object: ...
    def snapshot_download(self, repository: str, *, revision: str, local_dir: Path, allow_patterns: tuple[str, ...] | None, token: str) -> object: ...
    def remote_manifest(self, repository: str, *, revision: str, allow_patterns: tuple[str, ...] | None, token: str) -> RemoteManifest: ...
    def upload_model_file(self, repository: str, key: str, data: bytes, *, token: str) -> None: ...
    def download_model_file(self, repository: str, key: str, *, token: str) -> bytes: ...
    def delete_model_file(self, repository: str, key: str, *, token: str) -> None: ...
    def list_model_files(self, repository: str, prefix: str, *, token: str) -> tuple[str, ...]: ...
    def upload_bucket_file(self, bucket: str, source: Path, key: str, *, token: str) -> None: ...
    def download_bucket_file(self, bucket: str, key: str, destination: Path, *, token: str) -> None: ...
    def delete_bucket_file(self, bucket: str, key: str, *, token: str) -> None: ...
    def list_bucket_files(self, bucket: str, prefix: str, *, token: str) -> tuple[str, ...]: ...


class HfHubAdapter:
    """Concrete Hub 0.36 transport; token stays in method-local arguments."""
    def repo_info(self, repository: str, *, revision: str | None, token: str) -> object:
        from huggingface_hub import HfApi
        return HfApi().repo_info(repo_id=repository, repo_type="dataset" if repository == BEHAVIOR_1K_DATASET_REPOSITORY else "model", revision=revision, token=token)
    def bucket_info(self, bucket: str, *, token: str) -> object: raise RuntimeError("bucket access requires BucketHelperAccess")
    def create_bucket(self, bucket: str, *, private: bool, token: str) -> object: raise RuntimeError("bucket access requires BucketHelperAccess")
    def snapshot_download(self, repository: str, *, revision: str, local_dir: Path, allow_patterns: tuple[str, ...] | None, token: str) -> object:
        from huggingface_hub import snapshot_download
        return snapshot_download(repo_id=repository, repo_type="dataset" if repository == BEHAVIOR_1K_DATASET_REPOSITORY else "model", revision=revision, local_dir=str(local_dir), allow_patterns=list(allow_patterns) if allow_patterns else None, token=token)
    def remote_manifest(self, repository: str, *, revision: str, allow_patterns: tuple[str, ...] | None, token: str) -> RemoteManifest:
        from huggingface_hub import HfApi
        info = self.repo_info(repository, revision=revision, token=token)
        if not _exact_revision(info, revision):
            raise ValueError("Hub remote manifest revision drifted from the requested pin")
        api = HfApi()
        return build_remote_manifest(
            repository=repository,
            revision=revision,
            resolved_revision=revision,
            entries=api.list_repo_tree(
                repository,
                recursive=True,
                revision=revision,
                repo_type="dataset" if repository == BEHAVIOR_1K_DATASET_REPOSITORY else "model",
                token=token,
            ),
            allow_patterns=allow_patterns,
        )
    def upload_model_file(self, repository: str, key: str, data: bytes, *, token: str) -> None:
        from huggingface_hub import HfApi
        HfApi().upload_file(path_or_fileobj=data, path_in_repo=key, repo_id=repository, repo_type="model", token=token)
    def download_model_file(self, repository: str, key: str, *, token: str) -> bytes:
        from huggingface_hub import HfApi
        return Path(HfApi().hf_hub_download(repository, key, repo_type="model", force_download=True, token=token)).read_bytes()
    def delete_model_file(self, repository: str, key: str, *, token: str) -> None:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import EntryNotFoundError
        try:
            HfApi().delete_file(key, repository, repo_type="model", token=token)
        except EntryNotFoundError as error:
            raise ModelSmokeObjectNotFound("exact smoke object was not found") from error
    def list_model_files(self, repository: str, prefix: str, *, token: str) -> tuple[str, ...]:
        from huggingface_hub import HfApi
        return tuple(item.path for item in HfApi().list_repo_tree(repository, prefix, recursive=True, repo_type="model", token=token))


class BucketHelperAccess:
    def __init__(self, client: BucketHelperClient) -> None: self.client = client
    def bucket_info(self, bucket: str, *, token: str) -> object:
        result = self.client.request("info", {"bucket_id": bucket})
        return {"private": result["private"]}
    def create_bucket(self, bucket: str, *, private: bool, token: str) -> object:
        result = self.client.request("ensure", {"bucket_id": bucket, "create": True})
        return {"private": result["private"]}
    def upload_bucket_file(self, bucket: str, source: Path, key: str, *, token: str) -> None:
        self.client.request("upload", {"bucket_id": bucket, "local_path": str(source), "remote_path": key})
    def download_bucket_file(self, bucket: str, key: str, destination: Path, *, token: str) -> None:
        self.client.request("download", {"bucket_id": bucket, "remote_path": key, "local_path": str(destination)})
    def delete_bucket_file(self, bucket: str, key: str, *, token: str) -> None:
        self.client.request("delete", {"bucket_id": bucket, "paths": [key]})
    def list_bucket_files(self, bucket: str, prefix: str, *, token: str) -> tuple[str, ...]:
        return tuple(item["path"] for item in self.client.request("list", {"bucket_id": bucket, "prefix": prefix})["files"])


class ProductionHubAccess(HfHubAdapter):
    """Compose Hub 0.36 snapshots with the isolated 1.24 bucket helper."""
    def __init__(self, bucket_client: BucketHelperClient) -> None: self.bucket = BucketHelperAccess(bucket_client)
    def bucket_info(self, bucket: str, *, token: str) -> object: return self.bucket.bucket_info(bucket, token=token)
    def create_bucket(self, bucket: str, *, private: bool, token: str) -> object: return self.bucket.create_bucket(bucket, private=private, token=token)
    def upload_bucket_file(self, bucket: str, source: Path, key: str, *, token: str) -> None: self.bucket.upload_bucket_file(bucket, source, key, token=token)
    def download_bucket_file(self, bucket: str, key: str, destination: Path, *, token: str) -> None: self.bucket.download_bucket_file(bucket, key, destination, token=token)
    def delete_bucket_file(self, bucket: str, key: str, *, token: str) -> None: self.bucket.delete_bucket_file(bucket, key, token=token)
    def list_bucket_files(self, bucket: str, prefix: str, *, token: str) -> tuple[str, ...]: return self.bucket.list_bucket_files(bucket, prefix, token=token)


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    dataset: Path
    groot_upstream: Path
    cosmos: Path
    derived_model: Path
    selection_manifest: Path
    materialized_manifest: Path
    modality_sha256: str
    stats_sha256: str
    model_derivation: Path

    def offline_environment(self) -> dict[str, str]:
        return {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "offline",
            "WANDB_DIR": "/workspace/logs/wandb",
        }


def _private(info: object) -> bool: return bool(getattr(info, "private", False) if not isinstance(info, dict) else info.get("private"))


def _exact_revision(info: object, revision: str) -> bool:
    observed = getattr(info, "sha", None) if not isinstance(info, dict) else info.get("sha")
    return observed == revision


def _model_smoke_absent(*, hub: HubAccess, key: str, token: str) -> None:
    prefix = key.rsplit("/", 1)[0] + "/"
    for attempt in range(_SMOKE_CLEANUP_ATTEMPTS):
        if key not in hub.list_model_files(BEHAVIOR_1K_FINAL_MODEL_REPOSITORY, prefix, token=token): return
        if attempt + 1 < _SMOKE_CLEANUP_ATTEMPTS: time.sleep(_SMOKE_CLEANUP_INTERVAL_SECONDS)
    raise ValueError("model repository smoke cleanup did not remove its exact key")


def _bucket_smoke_absent(*, hub: HubAccess, key: str, token: str) -> None:
    prefix = key.rsplit("/", 1)[0] + "/"
    for attempt in range(_SMOKE_CLEANUP_ATTEMPTS):
        if key not in hub.list_bucket_files(BEHAVIOR_1K_CHECKPOINT_BUCKET, prefix, token=token): return
        if attempt + 1 < _SMOKE_CLEANUP_ATTEMPTS: time.sleep(_SMOKE_CLEANUP_INTERVAL_SECONDS)
    raise ValueError("checkpoint bucket smoke cleanup did not remove its exact key")


def _model_smoke_probe(*, hub: HubAccess, token: str) -> None:
    key = f"smoke/{uuid4().hex}/model.bin"
    upload_attempted = False
    try:
        upload_attempted = True
        hub.upload_model_file(BEHAVIOR_1K_FINAL_MODEL_REPOSITORY, key, _SMOKE_BYTES, token=token)
        if hub.download_model_file(BEHAVIOR_1K_FINAL_MODEL_REPOSITORY, key, token=token) != _SMOKE_BYTES:
            raise ValueError("model repository smoke readback did not match")
    finally:
        if upload_attempted:
            try:
                hub.delete_model_file(BEHAVIOR_1K_FINAL_MODEL_REPOSITORY, key, token=token)
            except ModelSmokeObjectNotFound:
                pass
            _model_smoke_absent(hub=hub, key=key, token=token)


def _bucket_smoke_probe(*, hub: HubAccess, token: str) -> None:
    key = f"smoke/{uuid4().hex}/checkpoint.bin"
    with tempfile.TemporaryDirectory(prefix="b1k-smoke-", dir=_SMOKE_LOCAL_ROOT) as temporary:
        source = Path(temporary) / "source"; destination = Path(temporary) / "readback"
        source.write_bytes(_SMOKE_BYTES)
        upload_attempted = False
        try:
            upload_attempted = True
            hub.upload_bucket_file(BEHAVIOR_1K_CHECKPOINT_BUCKET, source, key, token=token)
            hub.download_bucket_file(BEHAVIOR_1K_CHECKPOINT_BUCKET, key, destination, token=token)
            if destination.read_bytes() != _SMOKE_BYTES:
                raise ValueError("checkpoint bucket smoke readback did not match")
        finally:
            if upload_attempted:
                try:
                    hub.delete_bucket_file(BEHAVIOR_1K_CHECKPOINT_BUCKET, key, token=token)
                except BucketNotFound:
                    pass
                _bucket_smoke_absent(hub=hub, key=key, token=token)


def preflight_remote_access(token: str, create_bucket_flag: str, *, hub: HubAccess) -> None:
    """Verify every required remote capability before any download."""
    if not token or create_bucket_flag not in {"0", "1"}: raise ValueError("invalid remote access preflight")
    for repository, revision in ((BEHAVIOR_1K_DATASET_REPOSITORY, BEHAVIOR_1K_DATASET_REVISION), (BASE_MODEL_REPOSITORY, MODEL_REVISION), (COSMOS_REPOSITORY, COSMOS_REVISION)):
        if not _exact_revision(hub.repo_info(repository, revision=revision, token=token), revision):
            raise ValueError("repository did not resolve to its pinned revision")
    if not _private(hub.repo_info(BEHAVIOR_1K_FINAL_MODEL_REPOSITORY, revision=None, token=token)): raise ValueError("model repository must be private")
    try: bucket = hub.bucket_info(BEHAVIOR_1K_CHECKPOINT_BUCKET, token=token)
    except BucketNotFound:
        if create_bucket_flag != "1": raise ValueError("checkpoint bucket is missing")
        bucket = hub.create_bucket(BEHAVIOR_1K_CHECKPOINT_BUCKET, private=True, token=token)
    if not _private(bucket): raise ValueError("checkpoint bucket must be private")
    _model_smoke_probe(hub=hub, token=token)
    _bucket_smoke_probe(hub=hub, token=token)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_name("." + path.name + ".incomplete")
    payload = value.to_dict() if hasattr(value, "to_dict") else value
    temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":"))); temporary.replace(path)


def write_redacted_status(path: Path, *, phase: str, result: BootstrapResult | None = None) -> None:
    payload: dict[str, object] = {"phase": phase}
    if result is not None: payload["result"] = {"dataset": str(result.dataset), "modality_sha256": result.modality_sha256, "stats_sha256": result.stats_sha256}
    _atomic_json(path, payload)


def _materialize_snapshot(materialize: Callable[..., object], root: Path, validation: SnapshotValidation) -> object:
    return materialize(root, validated_artifacts=validation.artifacts)


def _validated_sha256(validation: SnapshotValidation, relative_path: str) -> str:
    for artifact in validation.artifacts:
        if artifact.path == relative_path:
            return artifact.sha256
    raise ValueError(f"remote snapshot does not contain required artifact: {relative_path}")


def bootstrap_remote(*, paths: WorkspacePaths, token: str, create_bucket_flag: str, hub: HubAccess, build_selection: Callable[[Path], object], materialize: Callable[..., object], deploy_modality: Callable[[Path], Path], stats_path: Path) -> BootstrapResult:
    """Serialize the dataset lifecycle before inspecting reusable state."""

    with destination_lock(paths.dataset):
        return _bootstrap_remote_locked(
            paths=paths,
            token=token,
            create_bucket_flag=create_bucket_flag,
            hub=hub,
            build_selection=build_selection,
            materialize=materialize,
            deploy_modality=deploy_modality,
            stats_path=stats_path,
        )


def _bootstrap_remote_locked(*, paths: WorkspacePaths, token: str, create_bucket_flag: str, hub: HubAccess, build_selection: Callable[[Path], object], materialize: Callable[..., object], deploy_modality: Callable[[Path], Path], stats_path: Path) -> BootstrapResult:
    """Bootstrap only receipt-validated snapshots, using sibling resumable stages.

    Hub's local ``.cache`` stays in the stage throughout retries, so its own
    resumable transfer state survives process interruption. A final path is
    exposed only after an authoritative remote-manifest/local-payload bijection.
    """

    expected_modality = paths.dataset / "meta/modality.json"
    expected_stats = paths.dataset / "meta/stats.json"
    if Path(stats_path) != expected_stats:
        raise ValueError("stats artifact must use the loader path")
    preflight_remote_access(token, create_bucket_flag, hub=hub)
    metadata, full = dataset_snapshot_patterns()
    dataset_local, dataset_completed = _open_snapshot_staging(
        paths.dataset,
        repository=BEHAVIOR_1K_DATASET_REPOSITORY,
        revision=BEHAVIOR_1K_DATASET_REVISION,
        allow_patterns=full,
    )
    model_validations: dict[Path, SnapshotValidation] = {}
    dataset_validation: SnapshotValidation | None = None
    dataset_manifest: RemoteManifest | None = None
    if not dataset_completed:
        hub.snapshot_download(BEHAVIOR_1K_DATASET_REPOSITORY, revision=BEHAVIOR_1K_DATASET_REVISION, local_dir=bound_destination(paths.dataset, dataset_local), allow_patterns=metadata, token=token)
        validate_destination_binding(paths.dataset)
    if dataset_completed:
        dataset_manifest = _remote_manifest(hub, repository=BEHAVIOR_1K_DATASET_REPOSITORY, revision=BEHAVIOR_1K_DATASET_REVISION, allow_patterns=full, token=token)
        _, dataset_validation = _validate_snapshot_receipt(
            paths.dataset,
            repository=BEHAVIOR_1K_DATASET_REPOSITORY,
            revision=BEHAVIOR_1K_DATASET_REVISION,
            allow_patterns=full,
            remote_manifest=dataset_manifest,
            local_derived_paths=("meta/modality.json",),
        )
        selection = build_selection(dataset_local)
        selection_hash = _manifest_hash(selection)
        materialized = _materialize_snapshot(materialize, dataset_local, dataset_validation)
        materialized_hash = _manifest_hash(materialized)
        _validate_snapshot_receipt(
            paths.dataset,
            repository=BEHAVIOR_1K_DATASET_REPOSITORY,
            revision=BEHAVIOR_1K_DATASET_REVISION,
            allow_patterns=full,
            remote_manifest=dataset_manifest,
            manifest_hashes={"selection": selection_hash, "materialized": materialized_hash},
            validation=dataset_validation,
            local_derived_paths=("meta/modality.json",),
        )
    else:
        selection = build_selection(dataset_local)
        selection_hash = _manifest_hash(selection)
        with ThreadPoolExecutor(max_workers=3) as pool:
            data = pool.submit(hub.snapshot_download, BEHAVIOR_1K_DATASET_REPOSITORY, revision=BEHAVIOR_1K_DATASET_REVISION, local_dir=bound_destination(paths.dataset, dataset_local), allow_patterns=full, token=token)
            base = pool.submit(_ensure_model_snapshot, hub=hub, repository=BASE_MODEL_REPOSITORY, revision=MODEL_REVISION, destination=paths.groot_upstream, token=token, validation_cache=model_validations)
            cosmos = pool.submit(_ensure_model_snapshot, hub=hub, repository=COSMOS_REPOSITORY, revision=COSMOS_REVISION, destination=paths.cosmos, token=token, validation_cache=model_validations)
            data.result(); validate_destination_binding(paths.dataset); base.result(); cosmos.result()
        confirmed_selection = build_selection(dataset_local)
        if _manifest_hash(confirmed_selection) != selection_hash:
            raise ValueError("dataset selection changed after payload download")
        dataset_manifest = _remote_manifest(hub, repository=BEHAVIOR_1K_DATASET_REPOSITORY, revision=BEHAVIOR_1K_DATASET_REVISION, allow_patterns=full, token=token)
        dataset_validation = validate_local_snapshot(dataset_local, dataset_manifest, local_derived_paths=("meta/modality.json",))
        materialized = _materialize_snapshot(materialize, dataset_local, dataset_validation)
        materialized_hash = _manifest_hash(materialized)
        staged_modality = deploy_modality(dataset_local)
        expected_staged_modality = dataset_local / "meta/modality.json"
        if staged_modality != expected_staged_modality or staged_modality.is_symlink() or not staged_modality.is_file():
            raise ValueError("modality artifact must use the loader path")
        modality_hash = sha256_file(staged_modality)
        if modality_hash != "ca3a2e406472650bee9439ed81a3f4a1531b6fe689cdc1b348d0f260a208e641":
            raise ValueError("R1Pro modality hash is invalid")
        staged_stats = dataset_local / "meta/stats.json"
        if staged_stats.is_symlink() or not staged_stats.is_file():
            raise ValueError("stats artifact must use the loader path")
        stats_hash = _validated_sha256(dataset_validation, "meta/stats.json")
        validate_destination_binding(paths.dataset)
        _promote_snapshot(
            bound_destination(paths.dataset, dataset_local),
            bound_destination(paths.dataset),
            repository=BEHAVIOR_1K_DATASET_REPOSITORY,
            revision=BEHAVIOR_1K_DATASET_REVISION,
            allow_patterns=full,
            remote_manifest=dataset_manifest,
            manifest_hashes={"selection": selection_hash, "materialized": materialized_hash},
            validation=dataset_validation,
            local_derived_paths=("meta/modality.json",),
            binding_destination=paths.dataset,
        )
        validate_destination_binding(paths.dataset)
    selection_path = paths.output / "selection-manifest.json"; _atomic_json(selection_path, selection)
    materialized_path = paths.output / "materialized-manifest.json"; _atomic_json(materialized_path, materialized)
    if dataset_completed:
        modality = expected_modality
        if modality.is_symlink() or not modality.is_file():
            raise ValueError("modality artifact must use the loader path")
        modality_hash = sha256_file(modality)
        if modality_hash != "ca3a2e406472650bee9439ed81a3f4a1531b6fe689cdc1b348d0f260a208e641":
            raise ValueError("R1Pro modality hash is invalid")
        if expected_stats.is_symlink() or not expected_stats.is_file():
            raise ValueError("stats artifact must use the loader path")
        if dataset_validation is None:
            raise ValueError("cached dataset validation is missing")
        stats_hash = _validated_sha256(dataset_validation, "meta/stats.json")
    with ThreadPoolExecutor(max_workers=3) as pool:
        # If a previous interruption happened before model promotion, resume
        # only the matching sibling stages.  Fully receipted models are reused.
        base = pool.submit(_ensure_model_snapshot, hub=hub, repository=BASE_MODEL_REPOSITORY, revision=MODEL_REVISION, destination=paths.groot_upstream, token=token, validation_cache=model_validations)
        cosmos = pool.submit(_ensure_model_snapshot, hub=hub, repository=COSMOS_REPOSITORY, revision=COSMOS_REVISION, destination=paths.cosmos, token=token, validation_cache=model_validations)
        base.result(); cosmos.result()
    derivation = derive_groot_config(
        paths.groot_upstream,
        paths.derived_model,
        cosmos_identity=_cosmos_snapshot_identity(paths.cosmos, hub=hub, token=token, validation=model_validations.get(paths.cosmos)),
        upstream_validation=None if paths.groot_upstream not in model_validations else model_validations[paths.groot_upstream].artifacts,
    ); derivation_path = paths.output / "model-derivation.json"; _atomic_json(derivation_path, derivation)
    result = BootstrapResult(paths.dataset, paths.groot_upstream, paths.cosmos, paths.derived_model, selection_path, materialized_path, modality_hash, stats_hash, derivation_path)
    write_redacted_status(paths.output / "bootstrap-status.json", phase="complete", result=result)
    return result
