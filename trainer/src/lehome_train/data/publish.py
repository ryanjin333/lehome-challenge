"""Immutable publication and retrieval of validated prepared datasets."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

from lehome_train.constants import DEFAULT_DATA_REPO
from lehome_train.hub import (
    HubTransport,
    download_files,
    require_access,
    upload_files,
)
from lehome_train.io import canonical_json_sha256, sha256_file
from lehome_train.models import ArtifactIdentity, SyncEntry, model_from_mapping
from lehome_train.redaction import generate_upload_allowlist


_VALIDATION_ARTIFACTS = (
    "meta/lehome_groot_modality.py",
    "meta/relative_stats.json",
    "meta/stats.json",
    "meta/validation_report.json",
)
_CONTROL_PATHS = ("manifest.json", "meta/prepared_hashes.json")


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
    if not isinstance(hashes, Mapping) or set(hashes) != set(_VALIDATION_ARTIFACTS):
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
        for path in _VALIDATION_ARTIFACTS
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
        or set(hashes) != set(_VALIDATION_ARTIFACTS)
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


def publish_prepared_dataset(
    dataset_path: str | os.PathLike[str],
    *,
    repository: str,
    revision: str,
    transport: HubTransport,
    environ: Mapping[str, str] | None = None,
    max_attempts: int = 3,
) -> PublishedDataset:
    """Publish one validated allowlist and verify it from the resolved commit."""

    if repository != DEFAULT_DATA_REPO:
        raise ValueError(f"prepared datasets may only be published to {DEFAULT_DATA_REPO}")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("dataset publication revision must be explicit")
    dataset = Path(dataset_path)
    entries = _validated_entries(dataset)
    require_access(
        transport=transport,
        repository=repository,
        read=True,
        write=True,
        environ=environ,
    )
    immutable_revision = upload_files(
        transport=transport,
        repository=repository,
        revision=revision,
        source=dataset,
        entries=entries,
        environ=environ,
        max_attempts=max_attempts,
    )
    readback = Path(tempfile.mkdtemp(prefix="lehome-dataset-readback-"))
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
        shutil.rmtree(readback, ignore_errors=True)
    return PublishedDataset(
        repository=repository,
        revision=immutable_revision,
        dataset_manifest_sha256=sha256_file(dataset / "manifest.json"),
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
        os.replace(temporary, destination)
        return destination
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
