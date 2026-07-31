"""Immutable publication and retrieval of validated prepared datasets."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping

from lehome_train.constants import DEFAULT_DATA_REPO
from lehome_train.data.validate import REQUIRED_VALIDATION_ARTIFACTS
from lehome_train.hub import (
    HubTransport,
    download_files,
    require_access,
    upload_files,
)
from lehome_train.io import canonical_json_sha256, sha256_file
from lehome_train.models import ArtifactIdentity, SyncEntry, model_from_mapping
from lehome_train.redaction import generate_upload_allowlist


_CONTROL_PATHS = ("manifest.json", "meta/prepared_hashes.json")
_MINIMUM_STAGING_RESERVE_BYTES = 64 * 1024**2
_MAXIMUM_STAGING_RESERVE_BYTES = 1024**3
_STAGING_RESERVE_FRACTION_DENOMINATOR = 20


@dataclass(frozen=True, slots=True)
class PublishedDataset:
    """Immutable identity returned after remote readback verification."""

    repository: str
    revision: str
    dataset_manifest_sha256: str
    entries: tuple[SyncEntry, ...]


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("dataset metadata contains a duplicate field")
        value[key] = item
    return value


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("prepared dataset metadata is unavailable or malformed") from None
    if not isinstance(value, dict):
        raise ValueError("prepared dataset metadata must be an object")
    return value


def _recorded_output_artifacts(manifest: Mapping[str, Any]) -> tuple[ArtifactIdentity, ...]:
    encoded = manifest.get("output_artifacts")
    if not isinstance(encoded, list) or not encoded:
        raise ValueError("prepared dataset has no hashed output artifact allowlist")
    if manifest.get("output_manifest_sha256") != canonical_json_sha256(encoded):
        raise ValueError("prepared dataset output artifact allowlist hash is invalid")
    try:
        artifacts = tuple(
            model_from_mapping(ArtifactIdentity, item)
            for item in encoded
            if isinstance(item, Mapping)
        )
    except (TypeError, ValueError):
        raise ValueError("prepared dataset output artifact allowlist is invalid") from None
    paths = tuple(artifact.relative_path for artifact in artifacts)
    if len(artifacts) != len(encoded) or len(set(paths)) != len(paths):
        raise ValueError("prepared dataset output artifact allowlist is invalid")
    return artifacts


def _recorded_validation_hashes(
    dataset: Path,
    manifest: Mapping[str, Any],
) -> dict[str, str]:
    prepared_hashes = _read_object(dataset / "meta" / "prepared_hashes.json")
    if set(prepared_hashes) != {"schema_version", "artifacts"}:
        raise ValueError("prepared validation hash allowlist has an invalid schema")
    if prepared_hashes["schema_version"] != 1:
        raise ValueError("prepared validation hash allowlist has an invalid version")
    hashes = prepared_hashes["artifacts"]
    if not isinstance(hashes, Mapping) or set(hashes) != set(REQUIRED_VALIDATION_ARTIFACTS):
        raise ValueError("prepared validation hash allowlist is incomplete")
    if not all(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
        for value in hashes.values()
    ):
        raise ValueError("prepared validation hash allowlist contains an invalid hash")

    statistics = manifest.get("statistics")
    if not isinstance(statistics, Mapping) or statistics.get("status") != "computed_task_4_train_only":
        raise ValueError("prepared dataset statistics are not complete")
    files = statistics.get("files")
    if not isinstance(files, list):
        raise ValueError("prepared dataset statistics allowlist is invalid")
    expected_statistics = {
        path: hashes[path]
        for path in REQUIRED_VALIDATION_ARTIFACTS
        if path != "meta/validation_report.json"
    }
    recorded_statistics: dict[str, object] = {}
    for entry in files:
        if not isinstance(entry, Mapping) or set(entry) != {"relative_path", "sha256"}:
            raise ValueError("prepared dataset statistics allowlist is invalid")
        path = entry["relative_path"]
        if not isinstance(path, str) or path in recorded_statistics:
            raise ValueError("prepared dataset statistics allowlist is invalid")
        recorded_statistics[path] = entry["sha256"]
    if recorded_statistics != expected_statistics:
        raise ValueError("prepared dataset statistics hashes differ from validation hashes")

    report = _read_object(dataset / "meta" / "validation_report.json")
    manifest_sha256 = sha256_file(dataset / "manifest.json")
    if report.get("valid") is not True:
        raise ValueError("prepared dataset does not have a successful validation report")
    if report.get("dataset_manifest_sha256") != manifest_sha256:
        raise ValueError("prepared dataset changed after validation")
    return dict(hashes)


def _allowlisted_paths_from_control_files(dataset: Path) -> tuple[str, ...]:
    manifest = _read_object(dataset / "manifest.json")
    output = _recorded_output_artifacts(manifest)
    prepared_hashes = _read_object(dataset / "meta" / "prepared_hashes.json")
    hashes = prepared_hashes.get("artifacts")
    if (
        set(prepared_hashes) != {"schema_version", "artifacts"}
        or prepared_hashes.get("schema_version") != 1
        or not isinstance(hashes, Mapping)
        or set(hashes) != set(REQUIRED_VALIDATION_ARTIFACTS)
    ):
        raise ValueError("prepared validation hash allowlist is incomplete")
    return tuple(
        sorted(
            {
                *(artifact.relative_path for artifact in output),
                *hashes.keys(),
                *_CONTROL_PATHS,
            }
        )
    )


def _validated_entries(dataset: Path) -> tuple[SyncEntry, ...]:
    if not dataset.is_dir() or dataset.is_symlink():
        raise ValueError("prepared dataset directory is unavailable")
    manifest = _read_object(dataset / "manifest.json")
    output = _recorded_output_artifacts(manifest)
    validation_hashes = _recorded_validation_hashes(dataset, manifest)
    expected_hashes = {
        artifact.relative_path: artifact.sha256
        for artifact in output
    }
    expected_sizes = {
        artifact.relative_path: artifact.byte_size
        for artifact in output
    }
    for path, digest in validation_hashes.items():
        existing = expected_hashes.get(path)
        if existing is not None and existing != digest:
            raise ValueError("prepared dataset contains conflicting artifact hashes")
        expected_hashes[path] = digest
    paths = tuple(sorted(set(expected_hashes) | set(_CONTROL_PATHS)))
    actual = generate_upload_allowlist(dataset, paths)
    for entry in actual:
        expected_digest = expected_hashes.get(entry.relative_path)
        if expected_digest is not None and entry.sha256 != expected_digest:
            raise ValueError("prepared dataset contains a dirty hashed artifact")
        expected_size = expected_sizes.get(entry.relative_path)
        if expected_size is not None and entry.byte_size != expected_size:
            raise ValueError("prepared dataset contains a dirty sized artifact")
    return actual


def _verify_entries(root: Path, entries: tuple[SyncEntry, ...]) -> None:
    observed = generate_upload_allowlist(
        root,
        tuple(entry.relative_path for entry in entries),
    )
    if observed != entries:
        raise ValueError("remote prepared dataset hash verification failed")


def _verify_complete_tree(root: Path, relative_paths: tuple[str, ...]) -> None:
    expected_files = set(relative_paths)
    expected_directories = {
        "/".join(parts[:index])
        for relative_path in relative_paths
        for parts in (relative_path.split("/"),)
        for index in range(1, len(parts))
    }
    observed_files: set[str] = set()
    pending = [(root, "")]
    while pending:
        directory, prefix = pending.pop()
        for entry in os.scandir(directory):
            relative = f"{prefix}/{entry.name}" if prefix else entry.name
            if entry.is_symlink():
                raise ValueError("remote prepared dataset contains an unexpected symlink")
            if entry.is_dir(follow_symlinks=False):
                if relative not in expected_directories:
                    raise ValueError("remote prepared dataset contains an unexpected directory")
                pending.append((Path(entry.path), relative))
            elif entry.is_file(follow_symlinks=False):
                observed_files.add(relative)
            else:
                raise ValueError("remote prepared dataset contains an unexpected path type")
    if observed_files != expected_files:
        raise ValueError("remote prepared dataset contains an unexpected file set")


def _stage_entries(
    source: Path,
    entries: tuple[SyncEntry, ...],
    *,
    staging_root: Path,
) -> Path:
    staging = Path(
        tempfile.mkdtemp(
            prefix="lehome-dataset-upload-",
            dir=staging_root,
        )
    )
    try:
        for entry in entries:
            destination = staging / entry.relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / entry.relative_path, destination)
        _verify_entries(staging, entries)
        return staging
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _free_space_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def _required_staging_bytes(entries: tuple[SyncEntry, ...]) -> int:
    payload_bytes = sum(entry.byte_size for entry in entries)
    reserve_bytes = min(
        _MAXIMUM_STAGING_RESERVE_BYTES,
        max(
            _MINIMUM_STAGING_RESERVE_BYTES,
            payload_bytes // _STAGING_RESERVE_FRACTION_DENOMINATOR,
        ),
    )
    return payload_bytes + reserve_bytes


def _require_staging_capacity(
    staging_root: Path,
    entries: tuple[SyncEntry, ...],
    free_space_probe: Callable[[Path], int],
    *,
    phase: str,
) -> None:
    available_bytes = free_space_probe(staging_root)
    if type(available_bytes) is not int or available_bytes < 0:
        raise ValueError(
            f"dataset {phase} staging free-space probe returned an invalid value"
        )
    required_bytes = _required_staging_bytes(entries)
    if available_bytes < required_bytes:
        raise ValueError(
            f"dataset {phase} staging filesystem has insufficient space "
            f"(requires {required_bytes} bytes, has {available_bytes} bytes)"
        )


def publish_prepared_dataset(
    dataset_path: str | os.PathLike[str],
    *,
    repository: str,
    revision: str,
    transport: HubTransport,
    environ: Mapping[str, str] | None = None,
    max_attempts: int = 3,
    staging_root: str | os.PathLike[str] | None = None,
    free_space_probe: Callable[[Path], int] = _free_space_bytes,
) -> PublishedDataset:
    """Publish a verified snapshot staged beside the dataset by default.

    ``staging_root`` can select another large filesystem. The existing directory
    must have room for every allowlisted byte plus a bounded safety reserve.
    """

    if repository != DEFAULT_DATA_REPO:
        raise ValueError(f"prepared datasets may only be published to {DEFAULT_DATA_REPO}")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("dataset publication revision must be explicit")
    dataset = Path(dataset_path)
    entries = _validated_entries(dataset)
    manifest_sha256 = next(
        entry.sha256
        for entry in entries
        if entry.relative_path == "manifest.json"
    )
    require_access(
        transport=transport,
        repository=repository,
        read=True,
        write=True,
        environ=environ,
    )
    resolved_staging_root = (
        dataset.parent
        if staging_root is None
        else Path(staging_root)
    )
    if not resolved_staging_root.is_dir() or resolved_staging_root.is_symlink():
        raise ValueError("dataset staging root must be an existing regular directory")
    _require_staging_capacity(
        resolved_staging_root,
        entries,
        free_space_probe,
        phase="upload",
    )
    staging = _stage_entries(
        dataset,
        entries,
        staging_root=resolved_staging_root,
    )
    try:
        immutable_revision = upload_files(
            transport=transport,
            repository=repository,
            revision=revision,
            source=staging,
            entries=entries,
            environ=environ,
            max_attempts=max_attempts,
        )
    finally:
        shutil.rmtree(staging)
    _require_staging_capacity(
        resolved_staging_root,
        entries,
        free_space_probe,
        phase="readback",
    )
    readback = Path(
        tempfile.mkdtemp(
            prefix="lehome-dataset-readback-",
            dir=resolved_staging_root,
        )
    )
    try:
        download_files(
            transport=transport,
            repository=repository,
            revision=immutable_revision,
            destination=readback,
            relative_paths=tuple(entry.relative_path for entry in entries),
            environ=environ,
            max_attempts=max_attempts,
        )
        _verify_entries(readback, entries)
    finally:
        shutil.rmtree(readback)
    return PublishedDataset(
        repository=repository,
        revision=immutable_revision,
        dataset_manifest_sha256=manifest_sha256,
        entries=entries,
    )


def download_prepared_dataset(
    destination_path: str | os.PathLike[str],
    *,
    repository: str,
    revision: str,
    expected_manifest_sha256: str,
    transport: HubTransport,
    environ: Mapping[str, str] | None = None,
    max_attempts: int = 3,
) -> Path:
    """Retrieve one immutable dataset and expose it only after full verification."""

    if repository != DEFAULT_DATA_REPO:
        raise ValueError(f"prepared datasets may only be downloaded from {DEFAULT_DATA_REPO}")
    if (
        not isinstance(expected_manifest_sha256, str)
        or len(expected_manifest_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_manifest_sha256
        )
    ):
        raise ValueError("expected dataset manifest SHA-256 is invalid")
    destination = Path(destination_path)
    if destination.exists():
        raise FileExistsError("refusing to overwrite a prepared dataset destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    require_access(
        transport=transport,
        repository=repository,
        read=True,
        write=False,
        environ=environ,
    )
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            suffix=".incomplete",
            dir=destination.parent,
        )
    )
    try:
        download_files(
            transport=transport,
            repository=repository,
            revision=revision,
            destination=temporary,
            relative_paths=_CONTROL_PATHS,
            environ=environ,
            max_attempts=max_attempts,
        )
        if sha256_file(temporary / "manifest.json") != expected_manifest_sha256:
            raise ValueError("remote prepared dataset manifest hash verification failed")
        relative_paths = _allowlisted_paths_from_control_files(temporary)
        download_files(
            transport=transport,
            repository=repository,
            revision=revision,
            destination=temporary,
            relative_paths=relative_paths,
            environ=environ,
            max_attempts=max_attempts,
        )
        entries = _validated_entries(temporary)
        if tuple(entry.relative_path for entry in entries) != relative_paths:
            raise ValueError("remote prepared dataset allowlist verification failed")
        _verify_complete_tree(temporary, relative_paths)
        os.replace(temporary, destination)
        return destination
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
