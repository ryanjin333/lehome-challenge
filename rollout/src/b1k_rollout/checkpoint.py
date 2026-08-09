"""Exact, fail-closed materialization for a final B1K policy release.

The model repository can contain evidence, Hub metadata, and several final
releases.  A rollout consumes exactly one immutable release: its selected
``final-manifest.json`` and the files it names below ``checkpoint/``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import uuid
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Protocol

from b1k_rollout.controller import CheckpointReceipt
from b1k_rollout.identity import MODEL_REPO, canonical_json_bytes, require_immutable_commit, require_sha256


_FINAL_MANIFEST_PATH = "final-manifest.json"
_MARKER_NAME = ".b1k-final-policy.json"
_MARKER_SCHEMA_VERSION = 1
_COPY_BUFFER_SIZE = 1024 * 1024
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024


class MaterializationError(RuntimeError):
    """The selected immutable final policy cannot be safely used."""


class ModelHub(Protocol):
    """Streaming file boundary for a single immutable model-repository commit."""

    def open_file(self, repository: str, *, revision: str, path: str) -> BinaryIO: ...


@dataclass(frozen=True, slots=True)
class _FileSpec:
    remote_path: str
    local_path: PurePosixPath
    byte_size: int
    sha256: str

    def marker_value(self) -> dict[str, object]:
        return {"byte_size": self.byte_size, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class _FinalManifest:
    run_id: str
    sha256: str
    files: tuple[_FileSpec, ...]


class FinalPolicyMaterializer:
    """Implement the controller's immutable ``CheckpointSource`` protocol.

    ``artifact_sha256`` is deliberately the SHA-256 of the manifest bytes, not
    a synthetic local-directory hash.  The marker binds the materialized tree
    to the private repository, immutable commit, authenticated manifest bytes,
    and exact selected file set.
    """

    def __init__(self, *, hub: ModelHub, expected_manifest_sha256: str) -> None:
        self._hub = hub
        try:
            self._expected_manifest_sha256 = require_sha256(
                expected_manifest_sha256, label="checkpoint artifact"
            )
        except ValueError as error:
            raise ValueError("checkpoint artifact must be a SHA-256 hash") from error

    def download(self, *, repository: str, revision: str, destination: Path) -> CheckpointReceipt:
        """Materialize the selected checkpoint tree atomically, or fail closed."""

        repository, revision, destination = self._validate_request(repository, revision, destination)
        if destination.exists() or destination.is_symlink():
            self._verify_existing(destination, repository=repository, revision=revision)
            return self._receipt(revision, destination)

        self._discard_incomplete_staging(destination)
        manifest = self._fetch_manifest(repository=repository, revision=revision)
        stage = destination.parent / f".{destination.name}.incomplete-{uuid.uuid4().hex}"
        try:
            os.mkdir(stage, 0o700)
            for spec in manifest.files:
                self._download_checked_file(stage, repository=repository, revision=revision, spec=spec)
            self._write_marker(stage, repository=repository, revision=revision, manifest=manifest)
            marker = self._load_marker(stage)
            self._verify_marker_identity(marker, repository=repository, revision=revision)
            self._verify_tree(stage, marker, expected_manifest=manifest)
            os.replace(stage, destination)
        except MaterializationError:
            self._discard_stage(stage)
            raise
        except OSError as error:
            self._discard_stage(stage)
            raise MaterializationError("immutable checkpoint materialization failed") from error
        return CheckpointReceipt(revision, manifest.sha256, destination.resolve())

    def readback(self, *, repository: str, revision: str, destination: Path) -> CheckpointReceipt:
        """Freshly authenticate the exact manifest and verify every local byte."""

        repository, revision, destination = self._validate_request(repository, revision, destination)
        manifest = self._fetch_manifest(repository=repository, revision=revision)
        marker = self._load_marker(destination)
        self._verify_marker_identity(marker, repository=repository, revision=revision)
        self._verify_tree(destination, marker, expected_manifest=manifest)
        return CheckpointReceipt(revision, manifest.sha256, destination.resolve())

    def _validate_request(self, repository: str, revision: str, destination: Path) -> tuple[str, str, Path]:
        if repository != MODEL_REPO:
            raise MaterializationError("final policy repository is invalid")
        try:
            revision = require_immutable_commit(revision, label="model commit")
        except ValueError as error:
            raise MaterializationError("final policy revision must be immutable") from error
        destination = Path(destination)
        if not destination.is_absolute() or destination.name in ("", ".", ".."):
            raise MaterializationError("checkpoint destination must be an absolute directory path")
        parent = destination.parent
        _ensure_directory_without_symlinks(parent)
        try:
            metadata = parent.lstat()
        except OSError as error:
            raise MaterializationError("checkpoint destination parent is unavailable") from error
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise MaterializationError("checkpoint destination parent is unsafe")
        return repository, revision, destination

    def _fetch_manifest(self, *, repository: str, revision: str) -> _FinalManifest:
        content = self._read_remote_bytes(repository=repository, revision=revision, path=_FINAL_MANIFEST_PATH)
        actual_sha256 = hashlib.sha256(content).hexdigest()
        if actual_sha256 != self._expected_manifest_sha256:
            raise MaterializationError("final manifest does not match the checkpoint artifact contract")
        return _parse_final_manifest(content, sha256=actual_sha256)

    def _read_remote_bytes(self, *, repository: str, revision: str, path: str) -> bytes:
        try:
            with closing(self._hub.open_file(repository, revision=revision, path=path)) as reader:
                chunks: list[bytes] = []
                total = 0
                while chunk := reader.read(_COPY_BUFFER_SIZE):
                    if not isinstance(chunk, bytes):
                        raise MaterializationError("immutable model download returned invalid bytes")
                    total += len(chunk)
                    if total > _MAX_MANIFEST_BYTES:
                        raise MaterializationError("final manifest exceeds the maximum safe size")
                    chunks.append(chunk)
        except MaterializationError:
            raise
        except Exception as error:
            raise MaterializationError("immutable final manifest download failed") from error
        return b"".join(chunks)

    def _download_checked_file(
        self, stage: Path, *, repository: str, revision: str, spec: _FileSpec
    ) -> None:
        descriptor = _open_destination_descriptor(stage, spec.local_path)
        total = 0
        digest = hashlib.sha256()
        try:
            with closing(self._hub.open_file(repository, revision=revision, path=spec.remote_path)) as reader:
                with os.fdopen(descriptor, "wb", closefd=True) as writer:
                    descriptor = -1
                    while chunk := reader.read(_COPY_BUFFER_SIZE):
                        if not isinstance(chunk, bytes):
                            raise MaterializationError("immutable model download returned invalid bytes")
                        total += len(chunk)
                        if total > spec.byte_size:
                            raise MaterializationError("checkpoint file byte size does not match final manifest")
                        digest.update(chunk)
                        writer.write(chunk)
                    writer.flush()
                    os.fsync(writer.fileno())
        except MaterializationError:
            raise
        except Exception as error:
            raise MaterializationError("immutable checkpoint file download failed") from error
        finally:
            if descriptor != -1:
                os.close(descriptor)
        if total != spec.byte_size:
            raise MaterializationError("checkpoint file byte size does not match final manifest")
        if digest.hexdigest() != spec.sha256:
            raise MaterializationError("checkpoint file SHA-256 does not match final manifest")

    def _write_marker(self, stage: Path, *, repository: str, revision: str, manifest: _FinalManifest) -> None:
        value = {
            "schema_version": _MARKER_SCHEMA_VERSION,
            "repository": repository,
            "revision": revision,
            "manifest_sha256": manifest.sha256,
            "run_id": manifest.run_id,
            "files": {spec.remote_path: spec.marker_value() for spec in manifest.files},
        }
        _write_bytes_safely(stage, PurePosixPath(_MARKER_NAME), canonical_json_bytes(value))

    def _verify_existing(self, destination: Path, *, repository: str, revision: str) -> None:
        marker = self._load_marker(destination)
        self._verify_marker_identity(marker, repository=repository, revision=revision)
        self._verify_tree(destination, marker, expected_manifest=None)

    def _load_marker(self, destination: Path) -> Mapping[str, object]:
        if destination.is_symlink() or not destination.is_dir():
            raise MaterializationError("checkpoint destination is invalid")
        marker = destination / _MARKER_NAME
        try:
            metadata = marker.lstat()
            if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise MaterializationError("checkpoint marker is invalid")
            raw = marker.read_bytes()
        except MaterializationError:
            raise
        except OSError as error:
            raise MaterializationError("checkpoint marker is unavailable") from error
        try:
            value = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_object)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise MaterializationError("checkpoint marker is invalid") from error
        if not isinstance(value, Mapping) or set(value) != {
            "schema_version", "repository", "revision", "manifest_sha256", "run_id", "files"
        }:
            raise MaterializationError("checkpoint marker is invalid")
        if value.get("schema_version") != _MARKER_SCHEMA_VERSION:
            raise MaterializationError("checkpoint marker is invalid")
        if not isinstance(value.get("repository"), str) or not isinstance(value.get("revision"), str):
            raise MaterializationError("checkpoint marker is invalid")
        if not isinstance(value.get("run_id"), str) or not value["run_id"]:
            raise MaterializationError("checkpoint marker is invalid")
        try:
            require_immutable_commit(value["revision"], label="marker revision")
            require_sha256(value.get("manifest_sha256"), label="marker manifest")
            files = _parse_file_mapping(value.get("files"), require_checkpoint=True)
        except (TypeError, ValueError) as error:
            raise MaterializationError("checkpoint marker is invalid") from error
        if not files:
            raise MaterializationError("checkpoint marker is invalid")
        return value

    def _verify_marker_identity(self, marker: Mapping[str, object], *, repository: str, revision: str) -> None:
        if (
            marker.get("repository") != repository
            or marker.get("revision") != revision
            or marker.get("manifest_sha256") != self._expected_manifest_sha256
        ):
            raise MaterializationError("checkpoint marker does not match the selected immutable release")

    def _verify_tree(
        self,
        destination: Path,
        marker: Mapping[str, object],
        *,
        expected_manifest: _FinalManifest | None,
    ) -> None:
        if destination.is_symlink() or not destination.is_dir():
            raise MaterializationError("checkpoint destination is invalid")
        marker_files = _parse_file_mapping(marker["files"], require_checkpoint=True)
        if expected_manifest is not None:
            self._verify_marker_identity(marker, repository=MODEL_REPO, revision=marker["revision"])
            if marker.get("run_id") != expected_manifest.run_id or marker.get("manifest_sha256") != expected_manifest.sha256:
                raise MaterializationError("checkpoint marker does not match final manifest")
            expected_files = {spec.remote_path: spec for spec in expected_manifest.files}
            if marker_files != expected_files:
                raise MaterializationError("checkpoint marker file set does not match final manifest")
        expected_local = {spec.local_path.as_posix(): spec for spec in marker_files.values()}
        expected_directories = {
            parent.as_posix()
            for spec in marker_files.values()
            for parent in spec.local_path.parents
            if parent != PurePosixPath(".")
        }
        actual_local: set[str] = set()
        try:
            for path in destination.rglob("*"):
                relative = path.relative_to(destination).as_posix()
                metadata = path.lstat()
                if stat.S_ISLNK(metadata.st_mode):
                    raise MaterializationError("checkpoint tree must not contain symlinks")
                if stat.S_ISDIR(metadata.st_mode):
                    if relative not in expected_directories:
                        raise MaterializationError("checkpoint tree contains an unexpected directory")
                    continue
                if not stat.S_ISREG(metadata.st_mode):
                    raise MaterializationError("checkpoint tree contains an unsupported entry")
                if relative == _MARKER_NAME:
                    continue
                actual_local.add(relative)
                spec = expected_local.get(relative)
                if spec is None:
                    raise MaterializationError("checkpoint tree contains an unexpected file")
                if metadata.st_size != spec.byte_size:
                    raise MaterializationError("checkpoint file byte size does not match final manifest")
                if _file_sha256(path) != spec.sha256:
                    raise MaterializationError("checkpoint file SHA-256 does not match final manifest")
        except MaterializationError:
            raise
        except OSError as error:
            raise MaterializationError("checkpoint tree is unreadable") from error
        if actual_local != set(expected_local):
            raise MaterializationError("checkpoint tree is missing a manifest file")

    def _receipt(self, revision: str, destination: Path) -> CheckpointReceipt:
        return CheckpointReceipt(revision, self._expected_manifest_sha256, destination.resolve())

    def _discard_incomplete_staging(self, destination: Path) -> None:
        prefix = f".{destination.name}.incomplete-"
        try:
            candidates = tuple(destination.parent.iterdir())
        except OSError as error:
            raise MaterializationError("checkpoint staging directory cannot be inspected") from error
        for candidate in candidates:
            if candidate.name.startswith(prefix):
                self._discard_stage(candidate)

    @staticmethod
    def _discard_stage(stage: Path) -> None:
        try:
            metadata = stage.lstat()
        except FileNotFoundError:
            return
        except OSError as error:
            raise MaterializationError("checkpoint staging directory cannot be inspected") from error
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise MaterializationError("checkpoint staging path is unsafe")
        try:
            shutil.rmtree(stage)
        except OSError as error:
            raise MaterializationError("incomplete checkpoint staging cleanup failed") from error


def _parse_final_manifest(raw: bytes, *, sha256: str) -> _FinalManifest:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise MaterializationError("final manifest is invalid") from error
    if not isinstance(value, Mapping) or set(value) != {"schema_version", "run_id", "files"}:
        raise MaterializationError("final manifest has unknown or missing keys")
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise MaterializationError("final manifest schema version is invalid")
    run_id = value.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise MaterializationError("final manifest run ID is invalid")
    try:
        files = _parse_file_mapping(value.get("files"), require_checkpoint=True, allow_noncheckpoint=True)
    except (TypeError, ValueError) as error:
        raise MaterializationError("final manifest files are invalid") from error
    selected = tuple(sorted(files.values(), key=lambda spec: spec.remote_path))
    if not selected:
        raise MaterializationError("final manifest must select nonempty checkpoint files")
    return _FinalManifest(run_id=run_id, sha256=sha256, files=selected)


def _parse_file_mapping(
    value: object, *, require_checkpoint: bool, allow_noncheckpoint: bool = False
) -> dict[str, _FileSpec]:
    if not isinstance(value, Mapping):
        raise TypeError("files must be an object")
    result: dict[str, _FileSpec] = {}
    local_paths: set[PurePosixPath] = set()
    normalized_paths: set[str] = set()
    for remote_path, descriptor in value.items():
        if not isinstance(remote_path, str):
            raise TypeError("manifest path must be a string")
        normalized_path = _normalized_manifest_path(remote_path)
        if normalized_path in normalized_paths:
            raise ValueError("manifest paths must not duplicate after normalization")
        normalized_paths.add(normalized_path)
        if not isinstance(descriptor, Mapping) or set(descriptor) != {"byte_size", "sha256"}:
            raise ValueError("manifest file descriptor has unknown or missing keys")
        byte_size = descriptor.get("byte_size")
        if type(byte_size) is not int or byte_size < 0:
            raise ValueError("manifest byte size is invalid")
        try:
            sha256 = require_sha256(descriptor.get("sha256"), label="manifest file")
        except ValueError as error:
            raise ValueError("manifest file SHA-256 is invalid") from error
        if remote_path.startswith("checkpoint/"):
            local_path = _checkpoint_relative_path(remote_path)
            if local_path in local_paths:
                raise ValueError("manifest paths must not duplicate after normalization")
            local_paths.add(local_path)
            result[remote_path] = _FileSpec(
                remote_path=remote_path, local_path=local_path, byte_size=byte_size, sha256=sha256
            )
        elif not allow_noncheckpoint:
            raise ValueError("manifest must select only checkpoint paths")
    if require_checkpoint and not result:
        raise ValueError("manifest checkpoint files are empty")
    return result


def _checkpoint_relative_path(remote_path: str) -> PurePosixPath:
    if not remote_path.startswith("checkpoint/"):
        raise ValueError("manifest must select only checkpoint paths")
    return PurePosixPath(*_normalized_manifest_path(remote_path).removeprefix("checkpoint/").split("/"))


def _normalized_manifest_path(remote_path: str) -> str:
    if "\\" in remote_path or "\x00" in remote_path or remote_path.endswith("/"):
        raise ValueError("manifest path is invalid")
    parts = remote_path.split("/")
    if not remote_path or any(part in ("", ".", "..") for part in parts):
        raise ValueError("manifest path is invalid")
    path = PurePosixPath(*parts)
    if path.is_absolute() or path.parts != tuple(parts):
        raise ValueError("manifest path is invalid")
    return path.as_posix()


def _reject_duplicate_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON object has duplicate keys")
        result[key] = value
    return result


def _open_destination_descriptor(root: Path, relative: PurePosixPath) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(root, flags)
    try:
        for component in relative.parts[:-1]:
            try:
                os.mkdir(component, 0o700, dir_fd=descriptor)
            except FileExistsError:
                pass
            child = os.open(component, flags, dir_fd=descriptor)
            previous = descriptor
            descriptor = child
            os.close(previous)
        file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            file_flags |= os.O_NOFOLLOW
        result = os.open(relative.name, file_flags, 0o600, dir_fd=descriptor)
    except Exception:
        os.close(descriptor)
        raise
    os.close(descriptor)
    return result


def _ensure_directory_without_symlinks(path: Path) -> None:
    """Create an absolute directory one descriptor-checked component at a time."""

    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for component in path.parts[1:]:
            try:
                os.mkdir(component, 0o700, dir_fd=descriptor)
            except FileExistsError:
                pass
            child = os.open(component, flags, dir_fd=descriptor)
            previous = descriptor
            descriptor = child
            os.close(previous)
    except OSError as error:
        raise MaterializationError("checkpoint destination parent is unsafe") from error
    finally:
        os.close(descriptor)


def _write_bytes_safely(root: Path, relative: PurePosixPath, content: bytes) -> None:
    descriptor = _open_destination_descriptor(root, relative)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as writer:
            descriptor = -1
            writer.write(content)
            writer.flush()
            os.fsync(writer.fileno())
    finally:
        if descriptor != -1:
            os.close(descriptor)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as reader:
            while chunk := reader.read(_COPY_BUFFER_SIZE):
                digest.update(chunk)
    except OSError as error:
        raise MaterializationError("checkpoint tree is unreadable") from error
    return digest.hexdigest()
