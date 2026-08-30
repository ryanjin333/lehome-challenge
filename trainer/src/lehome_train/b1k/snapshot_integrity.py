"""Pinned-Hub manifests and single-pass local snapshot verification."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from fnmatch import fnmatch
from typing import Any, Iterable, Mapping, Sequence

from lehome_train.io import canonical_json_sha256


_RECEIPT_SCHEMA_VERSION = 2
_CHUNK_SIZE = 1024 * 1024
_CONTROL_PATHS = frozenset({".b1k-snapshot-receipt.json", ".b1k-snapshot-intent.json"})


@dataclass(frozen=True, slots=True)
class RemoteManifestEntry:
    path: str
    byte_size: int
    identity_kind: str
    identity: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "byte_size": self.byte_size,
            "identity_kind": self.identity_kind,
            "identity": self.identity,
        }


@dataclass(frozen=True, slots=True)
class RemoteManifest:
    repository: str
    revision: str
    allow_patterns_sha256: str
    entries: tuple[RemoteManifestEntry, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class ValidatedArtifact:
    path: str
    byte_size: int
    sha256: str
    git_blob_sha1: str
    device: int
    inode: int
    mtime_ns: int

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
            "git_blob_sha1": self.git_blob_sha1,
        }


@dataclass(frozen=True, slots=True)
class SnapshotValidation:
    artifacts: tuple[ValidatedArtifact, ...]
    hash_passes: int


def allow_patterns_sha256(allow_patterns: tuple[str, ...] | None) -> str:
    return canonical_json_sha256({"allow_patterns": None if allow_patterns is None else list(allow_patterns)})


def _safe_path(value: object) -> str:
    if type(value) is not str or not value or value.startswith(("/", "\\")) or "\\" in value:
        raise ValueError("remote manifest path is unsafe")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("remote manifest path is unsafe")
    return value


def _exact_hex(value: object, length: int, label: str) -> str:
    if type(value) is not str or len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"remote manifest {label} is invalid")
    return value


def _entry_value(entry: object, name: str, missing: object = None) -> object:
    if isinstance(entry, Mapping):
        return entry.get(name, missing)
    return getattr(entry, name, missing)


def _is_directory_entry(entry: object) -> bool:
    tree_id = _entry_value(entry, "tree_id", _entry_value(entry, "oid", None) if type(entry).__name__ == "RepoFolder" else None)
    if tree_id is None:
        return False
    if _entry_value(entry, "size", None) is not None or _entry_value(entry, "blob_id", None) is not None or _entry_value(entry, "lfs", None) is not None:
        raise ValueError("remote manifest directory entry is malformed")
    _safe_path(_entry_value(entry, "path", None))
    return True


def _lfs_value(lfs: object, name: str) -> object:
    if isinstance(lfs, Mapping):
        return lfs.get(name)
    return getattr(lfs, name, None)


def _normalize_remote_entry(entry: object) -> RemoteManifestEntry:
    path = _safe_path(_entry_value(entry, "path", None))
    size = _entry_value(entry, "size", None)
    if type(size) is not int or size < 0:
        raise ValueError("remote manifest byte size is invalid")
    lfs = _entry_value(entry, "lfs", None)
    blob_id = _entry_value(entry, "blob_id", _entry_value(entry, "oid", None))
    if lfs is None:
        return RemoteManifestEntry(path, size, "git_blob_sha1", _exact_hex(blob_id, 40, "Git blob identity"))
    lfs_size = _lfs_value(lfs, "size")
    lfs_sha256 = _lfs_value(lfs, "sha256")
    if type(lfs_size) is not int or lfs_size != size:
        raise ValueError("remote manifest LFS size is invalid")
    return RemoteManifestEntry(path, size, "sha256", _exact_hex(lfs_sha256, 64, "LFS SHA-256"))


def _matches_allow_patterns(paths: Sequence[str], allow_patterns: tuple[str, ...] | None) -> set[str]:
    if allow_patterns is None:
        return set(paths)
    # This is the exact 0.36.2 ``filter_repo_objects`` allow-only branch:
    # a trailing slash gains ``*`` and matching delegates to ``fnmatch``.
    patterns = tuple(pattern + "*" if pattern.endswith("/") else pattern for pattern in allow_patterns)
    return {path for path in paths if any(fnmatch(path, pattern) for pattern in patterns)}


def build_remote_manifest(
    *,
    repository: str,
    revision: str,
    resolved_revision: str,
    entries: Iterable[object],
    allow_patterns: tuple[str, ...] | None,
) -> RemoteManifest:
    """Normalize an exact, pinned Hub tree into an immutable payload contract."""

    if type(repository) is not str or not repository or type(revision) is not str or not revision:
        raise ValueError("remote manifest identity is invalid")
    if resolved_revision != revision:
        raise ValueError("remote manifest revision drifted from the requested pin")
    normalized: list[RemoteManifestEntry] = []
    seen: set[str] = set()
    for raw in entries:
        if _is_directory_entry(raw):
            continue
        entry = _normalize_remote_entry(raw)
        if entry.path in seen:
            raise ValueError("remote manifest has duplicate paths")
        seen.add(entry.path)
        normalized.append(entry)
    allowed = _matches_allow_patterns([entry.path for entry in normalized], allow_patterns)
    selected = tuple(sorted((entry for entry in normalized if entry.path in allowed), key=lambda entry: entry.path))
    if not selected:
        raise ValueError("remote manifest selected payload set is empty")
    payload = {
        "repository": repository,
        "revision": revision,
        "allow_patterns_sha256": allow_patterns_sha256(allow_patterns),
        "entries": [entry.to_dict() for entry in selected],
    }
    return RemoteManifest(repository, revision, payload["allow_patterns_sha256"], selected, canonical_json_sha256(payload))


def _hash_file_once(path: Path) -> ValidatedArtifact:
    before = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(before.st_mode):
        raise ValueError("snapshot contains an unsafe file")
    sha256 = hashlib.sha256()
    blob = hashlib.sha1()
    blob.update(f"blob {before.st_size}\0".encode("ascii"))
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(_CHUNK_SIZE):
                sha256.update(chunk)
                blob.update(chunk)
            after = os.fstat(stream.fileno())
    except OSError as error:
        raise ValueError("snapshot payload cannot be read safely") from error
    if not stat.S_ISREG(after.st_mode) or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("snapshot payload changed while it was hashed")
    return ValidatedArtifact(path="", byte_size=before.st_size, sha256=sha256.hexdigest(), git_blob_sha1=blob.hexdigest(), device=before.st_dev, inode=before.st_ino, mtime_ns=before.st_mtime_ns)


def _walk_payload_paths(root: Path, *, local_derived_paths: frozenset[str]) -> tuple[str, ...]:
    root_stat = root.lstat()
    if root.is_symlink() or not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError("snapshot directory is unsafe")
    paths: list[str] = []
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directories:
            child = current_path / name
            relative = child.relative_to(root).as_posix()
            if child.is_symlink():
                raise ValueError("snapshot contains an unsafe symlink")
            if current_path == root and name == ".cache":
                continue
            if not stat.S_ISDIR(child.lstat().st_mode):
                raise ValueError("snapshot contains an unsafe directory")
            if relative in local_derived_paths:
                raise ValueError("snapshot local-derived artifact is a directory")
        directories[:] = [name for name in directories if not (current_path == root and name == ".cache")]
        for name in files:
            child = current_path / name
            relative = child.relative_to(root).as_posix()
            if child.is_symlink():
                raise ValueError("snapshot contains an unsafe symlink")
            if relative in local_derived_paths:
                continue
            if not stat.S_ISREG(child.lstat().st_mode):
                raise ValueError("snapshot contains an unsafe file")
            paths.append(relative)
    return tuple(sorted(paths))


def validate_local_snapshot(
    root: str | Path,
    remote_manifest: RemoteManifest,
    *,
    local_derived_paths: Iterable[str] = (),
) -> SnapshotValidation:
    """Hash every payload exactly once and enforce a remote/local bijection."""

    root = Path(root)
    derived = _CONTROL_PATHS | frozenset(_safe_path(path) for path in local_derived_paths)
    local_paths = _walk_payload_paths(root, local_derived_paths=derived)
    expected = {entry.path: entry for entry in remote_manifest.entries}
    observed = set(local_paths)
    missing = sorted(set(expected) - observed)
    extra = sorted(observed - set(expected))
    if missing:
        raise ValueError(f"snapshot is missing remote payloads: {missing[0]}")
    if extra:
        raise ValueError(f"snapshot has extra local payloads: {extra[0]}")
    artifacts: list[ValidatedArtifact] = []
    for relative in local_paths:
        hashed = _hash_file_once(root / relative)
        artifact = ValidatedArtifact(relative, hashed.byte_size, hashed.sha256, hashed.git_blob_sha1, hashed.device, hashed.inode, hashed.mtime_ns)
        remote = expected[relative]
        if artifact.byte_size != remote.byte_size:
            raise ValueError("snapshot payload size does not match the remote manifest")
        identity = artifact.sha256 if remote.identity_kind == "sha256" else artifact.git_blob_sha1
        if identity != remote.identity:
            raise ValueError("snapshot payload identity does not match the remote manifest")
        artifacts.append(artifact)
    return SnapshotValidation(tuple(artifacts), hash_passes=1)


def verify_artifact_stat_invariants(
    root: str | Path,
    artifacts: Sequence[ValidatedArtifact],
    *,
    local_derived_paths: Iterable[str] = (),
) -> None:
    """Recheck the already-hashed table without reopening any payload bytes."""

    root = Path(root)
    derived = _CONTROL_PATHS | frozenset(_safe_path(path) for path in local_derived_paths)
    paths = _walk_payload_paths(root, local_derived_paths=derived)
    expected = {artifact.path: artifact for artifact in artifacts}
    if set(paths) != set(expected):
        raise ValueError("snapshot payload paths changed after validation")
    for relative in paths:
        observed = (root / relative).lstat()
        artifact = expected[relative]
        if not stat.S_ISREG(observed.st_mode) or (observed.st_size, observed.st_dev, observed.st_ino, observed.st_mtime_ns) != (artifact.byte_size, artifact.device, artifact.inode, artifact.mtime_ns):
            raise ValueError("snapshot payload stat invariants changed after validation")


def build_snapshot_receipt(
    *,
    repository: str,
    revision: str,
    allow_patterns: tuple[str, ...] | None,
    remote_manifest: RemoteManifest,
    artifacts: Sequence[ValidatedArtifact],
    manifest_hashes: Mapping[str, str],
) -> dict[str, object]:
    if remote_manifest.repository != repository or remote_manifest.revision != revision or remote_manifest.allow_patterns_sha256 != allow_patterns_sha256(allow_patterns):
        raise ValueError("remote manifest does not match snapshot receipt identity")
    if not all(type(key) is str and type(value) is str for key, value in manifest_hashes.items()):
        raise ValueError("snapshot receipt manifest hashes are invalid")
    return {
        "schema_version": _RECEIPT_SCHEMA_VERSION,
        "repository": repository,
        "revision": revision,
        "allow_patterns_sha256": allow_patterns_sha256(allow_patterns),
        "remote_manifest_sha256": remote_manifest.sha256,
        "hash_passes": 1,
        "manifest_hashes": dict(manifest_hashes),
        "validated_artifacts": [artifact.to_dict() for artifact in artifacts],
    }


def read_snapshot_json(path: str | Path, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} has duplicate JSON keys")
            result[key] = value
        return result

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates, parse_constant=lambda _: (_ for _ in ()).throw(ValueError(f"{label} has nonfinite JSON")))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} is invalid")
    return value


def validate_snapshot_receipt(
    receipt_path: str | Path,
    *,
    repository: str,
    revision: str,
    allow_patterns: tuple[str, ...] | None,
    remote_manifest: RemoteManifest,
    validation: SnapshotValidation,
    manifest_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    receipt = read_snapshot_json(receipt_path, "completed snapshot receipt")
    expected = build_snapshot_receipt(
        repository=repository,
        revision=revision,
        allow_patterns=allow_patterns,
        remote_manifest=remote_manifest,
        artifacts=validation.artifacts,
        manifest_hashes=receipt.get("manifest_hashes") if manifest_hashes is None and isinstance(receipt.get("manifest_hashes"), Mapping) else {} if manifest_hashes is None else manifest_hashes,
    )
    if set(receipt) != set(expected) or any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("completed snapshot receipt does not match the remote manifest and validated artifacts")
    return receipt
