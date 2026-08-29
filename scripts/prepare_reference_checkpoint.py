#!/usr/bin/env python3
"""Create or verify an offline, inference-only view of the pinned GR00T checkpoint."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
from typing import Mapping


MANIFEST_FILENAME = "reference-checkpoint-view.json"

_CONFIG_FILENAME = "config.json"
_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_REPOSITORY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
_ARTIFACT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_REMOVED_FIELDS: dict[str, object] = {
    "num_decay_steps": 4000,
    "decay_lr_ratio": 0.1,
}
_MANIFEST_KEYS = {
    "schema_version",
    "source_repository",
    "source_revision",
    "source_pretrained_model",
    "expected_source_config_sha256",
    "actual_source_config_sha256",
    "adapted_config_sha256",
    "removed_fields",
    "linked_artifacts",
}
_ARTIFACT_KEYS = {"relative_name", "size", "sha256", "absolute_target"}


def _canonical_json(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        raise ValueError("document cannot be represented as canonical JSON") from None
    return (text + "\n").encode("ascii")


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key: {key}")
        result[key] = value
    return result


def _parse_json_object(payload: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {constant}"),
            ),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError):
        raise ValueError(f"{label} is not strict JSON") from None
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return value


def _normalize_source(path: Path | str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise ValueError("source pretrained_model directory must not be a symlink")
    try:
        normalized = candidate.resolve(strict=True)
        metadata = normalized.lstat()
    except OSError:
        raise ValueError("source pretrained_model directory is unavailable") from None
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("source pretrained_model path must be a directory")
    return normalized


def _normalize_destination(path: Path | str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        parent = candidate.parent.resolve(strict=True)
        parent_metadata = parent.lstat()
    except OSError:
        raise ValueError("destination parent is unavailable") from None
    if not stat.S_ISDIR(parent_metadata.st_mode):
        raise ValueError("destination parent must be a directory")
    destination = parent / candidate.name
    try:
        destination.lstat()
    except FileNotFoundError:
        return destination
    except OSError:
        raise ValueError("destination cannot be inspected safely") from None
    raise ValueError("destination already exists")


def _validate_trust_inputs(
    expected_source_config_sha256: str,
    source_repository: str,
    source_revision: str,
) -> None:
    if _SHA256.fullmatch(expected_source_config_sha256) is None:
        raise ValueError("expected source config SHA-256 must be 64 lowercase hex characters")
    if _REPOSITORY.fullmatch(source_repository) is None:
        raise ValueError("source repository must be an owner/repository identifier")
    if _REVISION.fullmatch(source_revision) is None:
        raise ValueError("source revision must be an immutable 40-character commit digest")


def _validate_expected_artifacts(
    expected_artifacts: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    if type(expected_artifacts) is not dict or not expected_artifacts:
        raise ValueError("expected artifacts must be a nonempty JSON object")
    normalized: dict[str, dict[str, object]] = {}
    for name in sorted(expected_artifacts):
        value = expected_artifacts[name]
        if (
            type(name) is not str
            or _ARTIFACT_NAME.fullmatch(name) is None
            or name in {_CONFIG_FILENAME, MANIFEST_FILENAME}
        ):
            raise ValueError("expected artifacts contain an unsafe name")
        if type(value) is not dict or set(value) != {"size", "sha256"}:
            raise ValueError(f"expected artifacts entry {name} has an invalid schema")
        size = value.get("size")
        digest = value.get("sha256")
        if type(size) is not int or size < 0:
            raise ValueError(f"expected artifacts entry {name} has an invalid size")
        if type(digest) is not str or _SHA256.fullmatch(digest) is None:
            raise ValueError(f"expected artifacts entry {name} has an invalid SHA-256")
        normalized[name] = {"size": size, "sha256": digest}
    return normalized


def _parse_expected_artifacts_json(value: str) -> dict[str, dict[str, object]]:
    try:
        payload = value.encode("ascii")
    except UnicodeError:
        raise ValueError("expected artifacts JSON must be canonical ASCII JSON") from None
    document = _parse_json_object(payload, "expected artifacts JSON")
    normalized = _validate_expected_artifacts(document)
    canonical = _canonical_json(normalized).decode("ascii").removesuffix("\n")
    if value != canonical:
        raise ValueError("expected artifacts JSON must be canonical strict JSON")
    return normalized


def _regular_file_metadata(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError:
        raise ValueError(f"{label} is unavailable") from None
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file and not a symlink")
    return metadata


_FILE_STABILITY_FIELDS = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
_DIRECTORY_STABILITY_FIELDS = ("st_dev", "st_ino", "st_mode", "st_mtime_ns", "st_ctime_ns")


def _same_metadata(left: os.stat_result, right: os.stat_result, fields: tuple[str, ...]) -> bool:
    return all(getattr(left, field) == getattr(right, field) for field in fields)


@dataclass
class _OpenRegularSnapshot:
    path: Path
    label: str
    descriptor: int
    initial_metadata: os.stat_result
    size: int
    sha256: str

    def read_payload(self) -> bytes:
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(self.descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        if len(payload) != self.size or hashlib.sha256(payload).hexdigest() != self.sha256:
            raise ValueError(f"{self.label} changed while reading")
        return payload

    def revalidate(self) -> None:
        try:
            opened = os.fstat(self.descriptor)
            current = self.path.lstat()
        except OSError:
            raise ValueError(f"{self.label} changed during verification") from None
        if (
            not _same_metadata(self.initial_metadata, opened, _FILE_STABILITY_FIELDS)
            or not _same_metadata(self.initial_metadata, current, _FILE_STABILITY_FIELDS)
            or not stat.S_ISREG(current.st_mode)
        ):
            raise ValueError(f"{self.label} changed during verification")

    def close(self) -> None:
        os.close(self.descriptor)


@dataclass
class _OpenDirectorySnapshot:
    path: Path
    label: str
    descriptor: int
    initial_metadata: os.stat_result
    initial_entries: set[str]

    def revalidate(self) -> None:
        try:
            opened = os.fstat(self.descriptor)
            current = self.path.lstat()
            entries = {entry.name for entry in self.path.iterdir()}
        except OSError:
            raise ValueError(f"{self.label} changed during verification") from None
        if (
            not _same_metadata(self.initial_metadata, opened, _DIRECTORY_STABILITY_FIELDS)
            or not _same_metadata(self.initial_metadata, current, _DIRECTORY_STABILITY_FIELDS)
            or entries != self.initial_entries
        ):
            raise ValueError(f"{self.label} changed during verification")

    def close(self) -> None:
        os.close(self.descriptor)


@dataclass(frozen=True)
class _SymlinkSnapshot:
    path: Path
    label: str
    initial_metadata: os.stat_result
    target: str

    def revalidate(self) -> None:
        try:
            current = self.path.lstat()
            target = os.readlink(self.path)
        except OSError:
            raise ValueError(f"{self.label} symlink changed during verification") from None
        if (
            not _same_metadata(self.initial_metadata, current, _FILE_STABILITY_FIELDS)
            or not stat.S_ISLNK(current.st_mode)
            or target != self.target
        ):
            raise ValueError(f"{self.label} symlink changed during verification")


def _open_regular_snapshot(path: Path, label: str) -> _OpenRegularSnapshot:
    before = _regular_file_metadata(path, label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise ValueError(f"{label} cannot be opened safely") from None
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(f"{label} changed while opening")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after_read = os.fstat(descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    try:
        after = _regular_file_metadata(path, label)
    except BaseException:
        os.close(descriptor)
        raise
    if (
        not _same_metadata(before, after_read, _FILE_STABILITY_FIELDS)
        or not _same_metadata(after_read, after, _FILE_STABILITY_FIELDS)
    ):
        os.close(descriptor)
        raise ValueError(f"{label} changed after hashing")
    return _OpenRegularSnapshot(path, label, descriptor, before, after_read.st_size, digest.hexdigest())


def _open_directory_snapshot(path: Path, label: str) -> _OpenDirectorySnapshot:
    descriptor: int | None = None
    try:
        before = path.lstat()
        if not stat.S_ISDIR(before.st_mode) or path.is_symlink():
            raise ValueError(f"{label} must be a directory and not a symlink")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        opened = os.fstat(descriptor)
        entries = {entry.name for entry in path.iterdir()}
        after = path.lstat()
    except ValueError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError:
        if descriptor is not None:
            os.close(descriptor)
        raise ValueError(f"{label} cannot be opened safely") from None
    if (
        not _same_metadata(before, opened, _DIRECTORY_STABILITY_FIELDS)
        or not _same_metadata(opened, after, _DIRECTORY_STABILITY_FIELDS)
    ):
        os.close(descriptor)
        raise ValueError(f"{label} changed while opening")
    return _OpenDirectorySnapshot(path, label, descriptor, before, entries)


def _hash_regular_file(path: Path, label: str) -> tuple[int, str]:
    snapshot = _open_regular_snapshot(path, label)
    try:
        snapshot.revalidate()
        return snapshot.size, snapshot.sha256
    finally:
        snapshot.close()


def _read_regular_file(path: Path, label: str) -> tuple[bytes, str]:
    snapshot = _open_regular_snapshot(path, label)
    try:
        payload = snapshot.read_payload()
        snapshot.revalidate()
        return payload, snapshot.sha256
    finally:
        snapshot.close()


def _read_and_sanitize_source_config(
    source: Path,
    expected_sha256: str,
) -> tuple[dict[str, object], dict[str, object], str]:
    payload, actual_sha256 = _read_regular_file(source / _CONFIG_FILENAME, "source config.json")
    return _sanitize_source_config_payload(payload, expected_sha256, actual_sha256)


def _sanitize_source_config_payload(
    payload: bytes,
    expected_sha256: str,
    actual_sha256: str,
) -> tuple[dict[str, object], dict[str, object], str]:
    if actual_sha256 != expected_sha256:
        raise ValueError("source config.json SHA-256 mismatch")
    config = _parse_json_object(payload, "source config.json")
    if config.get("type") != "groot" or type(config.get("type")) is not str:
        raise ValueError("source config.json type must be groot")
    for key, expected_value in _REMOVED_FIELDS.items():
        if key not in config:
            raise ValueError(f"source config.json is missing {key}")
        actual_value = config[key]
        if type(actual_value) is not type(expected_value) or actual_value != expected_value:
            raise ValueError(f"source config.json {key} has an unexpected value")
    adapted = {key: value for key, value in config.items() if key not in _REMOVED_FIELDS}
    if set(adapted) != set(config) - set(_REMOVED_FIELDS):
        raise ValueError("adapted config key preservation failed")
    if any(adapted[key] != config[key] for key in adapted):
        raise ValueError("adapted config semantic preservation failed")
    return config, adapted, actual_sha256


def _source_artifact_paths(
    source: Path,
    expected_artifacts: Mapping[str, Mapping[str, object]],
) -> list[Path]:
    try:
        entries = sorted(source.iterdir(), key=lambda path: path.name)
    except OSError:
        raise ValueError("source pretrained_model directory is unreadable") from None
    artifacts: list[Path] = []
    for path in entries:
        name = path.name
        if name == _CONFIG_FILENAME:
            continue
        if name == MANIFEST_FILENAME or _ARTIFACT_NAME.fullmatch(name) is None:
            raise ValueError(f"unsafe artifact name: {name!r}")
        _regular_file_metadata(path, f"source artifact {name}")
        artifacts.append(path)
    if not artifacts:
        raise ValueError("source pretrained_model contains no checkpoint artifacts")
    if {path.name for path in artifacts} != set(expected_artifacts):
        raise ValueError("source entries do not exactly match expected artifacts trust input")
    return artifacts


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_directory_exclusively(temporary: Path, destination: Path) -> None:
    """Atomically rename a directory while refusing an existing destination."""
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(temporary)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin" and hasattr(library, "renamex_np"):
        result = library.renamex_np(source_bytes, destination_bytes, 0x00000004)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        result = library.renameat2(-100, source_bytes, -100, destination_bytes, 1)  # RENAME_NOREPLACE
    else:
        raise RuntimeError("atomic no-replace directory publication is unsupported on this platform")
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise ValueError("destination already exists")
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _remove_exact_published_destination(
    destination: Path,
    published_metadata: os.stat_result,
) -> None:
    try:
        current = destination.lstat()
    except OSError:
        raise RuntimeError("published destination disappeared before rollback") from None
    if (
        not stat.S_ISDIR(current.st_mode)
        or (current.st_dev, current.st_ino) != (published_metadata.st_dev, published_metadata.st_ino)
    ):
        raise RuntimeError("published destination identity changed; refusing unsafe rollback")
    quarantine = Path(tempfile.mkdtemp(prefix=".reference-checkpoint-rollback-", dir=destination.parent))
    quarantine.rmdir()
    _publish_directory_exclusively(destination, quarantine)
    moved = quarantine.lstat()
    if (moved.st_dev, moved.st_ino) != (published_metadata.st_dev, published_metadata.st_ino):
        raise RuntimeError("published destination identity changed during rollback")
    try:
        shutil.rmtree(quarantine)
    finally:
        _fsync_directory(destination.parent)


def _manifest_artifacts(
    source: Path,
    artifact_paths: list[Path],
    expected_artifacts: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for artifact in artifact_paths:
        size, digest = _hash_regular_file(artifact, f"source artifact {artifact.name}")
        expected = expected_artifacts[artifact.name]
        if size != expected["size"] or digest != expected["sha256"]:
            raise ValueError(f"source artifact {artifact.name} does not match expected artifacts trust input")
        target = artifact.resolve(strict=True)
        if target.parent != source:
            raise ValueError(f"source artifact {artifact.name} resolves outside source directory")
        rows.append(
            {
                "relative_name": artifact.name,
                "size": expected["size"],
                "sha256": expected["sha256"],
                "absolute_target": str(target),
            },
        )
    return rows


def _validate_manifest_schema(
    manifest: dict[str, object],
    expected_artifacts: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    if (
        set(manifest) != _MANIFEST_KEYS
        or type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != _SCHEMA_VERSION
    ):
        raise ValueError("manifest schema is invalid")
    if type(manifest.get("source_repository")) is not str or _REPOSITORY.fullmatch(manifest["source_repository"]) is None:
        raise ValueError("manifest repository is invalid")
    if type(manifest.get("source_revision")) is not str or _REVISION.fullmatch(manifest["source_revision"]) is None:
        raise ValueError("manifest revision is invalid")
    for key in ("expected_source_config_sha256", "actual_source_config_sha256", "adapted_config_sha256"):
        if type(manifest.get(key)) is not str or _SHA256.fullmatch(manifest[key]) is None:
            raise ValueError(f"manifest {key} is invalid")
    if type(manifest.get("source_pretrained_model")) is not str or not Path(manifest["source_pretrained_model"]).is_absolute():
        raise ValueError("manifest source pretrained_model path is invalid")
    removed_fields = manifest.get("removed_fields")
    if (
        type(removed_fields) is not dict
        or set(removed_fields) != set(_REMOVED_FIELDS)
        or any(
            type(removed_fields[key]) is not type(expected_value)
            or removed_fields[key] != expected_value
            for key, expected_value in _REMOVED_FIELDS.items()
        )
    ):
        raise ValueError("manifest removed fields are invalid")
    rows = manifest.get("linked_artifacts")
    if type(rows) is not list or not rows:
        raise ValueError("manifest linked artifacts are invalid")
    checked: list[dict[str, object]] = []
    previous_name: str | None = None
    for value in rows:
        if type(value) is not dict or set(value) != _ARTIFACT_KEYS:
            raise ValueError("manifest linked artifact schema is invalid")
        name = value.get("relative_name")
        size = value.get("size")
        digest = value.get("sha256")
        target = value.get("absolute_target")
        if type(name) is not str or _ARTIFACT_NAME.fullmatch(name) is None or name in {_CONFIG_FILENAME, MANIFEST_FILENAME}:
            raise ValueError("manifest linked artifact name is invalid")
        if previous_name is not None and name <= previous_name:
            raise ValueError("manifest linked artifacts must be unique and sorted")
        if type(size) is not int or size < 0:
            raise ValueError("manifest linked artifact size is invalid")
        if type(digest) is not str or _SHA256.fullmatch(digest) is None:
            raise ValueError("manifest linked artifact SHA-256 is invalid")
        if type(target) is not str or not Path(target).is_absolute():
            raise ValueError("manifest linked artifact target is invalid")
        previous_name = name
        checked.append(value)
    if {row["relative_name"] for row in checked} != set(expected_artifacts):
        raise ValueError("manifest linked artifacts do not match expected artifacts trust input")
    for row in checked:
        expected = expected_artifacts[row["relative_name"]]
        if row["size"] != expected["size"] or row["sha256"] != expected["sha256"]:
            raise ValueError(
                f"manifest linked artifact {row['relative_name']} does not match expected artifacts trust input",
            )
    return checked


def verify_reference_checkpoint(
    *,
    source_pretrained_model: Path | str,
    destination_view: Path | str,
    expected_source_config_sha256: str,
    source_repository: str,
    source_revision: str,
    expected_artifacts: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Independently verify a compatibility view against explicit trust inputs."""
    _validate_trust_inputs(expected_source_config_sha256, source_repository, source_revision)
    normalized_expected_artifacts = _validate_expected_artifacts(expected_artifacts)
    source = _normalize_source(source_pretrained_model)
    destination_candidate = Path(destination_view)
    if destination_candidate.is_symlink():
        raise ValueError("destination view must not be a symlink")
    try:
        destination = destination_candidate.resolve(strict=True)
        destination_metadata = destination.lstat()
    except OSError:
        raise ValueError("destination view is unavailable") from None
    if not stat.S_ISDIR(destination_metadata.st_mode):
        raise ValueError("destination view must be a directory")
    source_directory = _open_directory_snapshot(source, "source pretrained_model directory")
    view_directory: _OpenDirectorySnapshot | None = None
    regular_snapshots: list[_OpenRegularSnapshot] = []
    symlink_snapshots: list[_SymlinkSnapshot] = []
    try:
        view_directory = _open_directory_snapshot(destination, "destination view directory")
        expected_source_names = {_CONFIG_FILENAME, *normalized_expected_artifacts}
        expected_view_names = {_CONFIG_FILENAME, MANIFEST_FILENAME, *normalized_expected_artifacts}
        if source_directory.initial_entries != expected_source_names:
            raise ValueError("source entries do not exactly match expected artifacts trust input")
        if view_directory.initial_entries != expected_view_names:
            raise ValueError("destination view entries do not match manifest")

        manifest_snapshot = _open_regular_snapshot(destination / MANIFEST_FILENAME, "view manifest")
        regular_snapshots.append(manifest_snapshot)
        manifest_payload = manifest_snapshot.read_payload()
        manifest = _parse_json_object(manifest_payload, "view manifest")
        if manifest_payload != _canonical_json(manifest):
            raise ValueError("view manifest is not canonical JSON")
        artifact_rows = _validate_manifest_schema(manifest, normalized_expected_artifacts)
        if manifest["source_repository"] != source_repository:
            raise ValueError("manifest source repository does not match trust input")
        if manifest["source_revision"] != source_revision:
            raise ValueError("manifest source revision does not match trust input")
        if manifest["source_pretrained_model"] != str(source):
            raise ValueError("manifest source pretrained_model does not match trust input")
        if manifest["expected_source_config_sha256"] != expected_source_config_sha256:
            raise ValueError("manifest expected source config SHA-256 does not match trust input")

        source_config_snapshot = _open_regular_snapshot(source / _CONFIG_FILENAME, "source config.json")
        regular_snapshots.append(source_config_snapshot)
        source_config, expected_adapted, actual_source_digest = _sanitize_source_config_payload(
            source_config_snapshot.read_payload(),
            expected_source_config_sha256,
            source_config_snapshot.sha256,
        )
        if manifest["actual_source_config_sha256"] != actual_source_digest:
            raise ValueError("manifest actual source config SHA-256 is invalid")

        adapted_snapshot = _open_regular_snapshot(destination / _CONFIG_FILENAME, "adapted config.json")
        regular_snapshots.append(adapted_snapshot)
        adapted_payload = adapted_snapshot.read_payload()
        if adapted_payload != _canonical_json(expected_adapted):
            raise ValueError("adapted config is not the deterministic canonical sanitization")
        if adapted_snapshot.sha256 != manifest["adapted_config_sha256"]:
            raise ValueError("adapted config SHA-256 does not match manifest")
        adapted = _parse_json_object(adapted_payload, "adapted config.json")
        if adapted != expected_adapted or set(adapted) != set(source_config) - set(_REMOVED_FIELDS):
            raise ValueError("adapted config semantics do not exactly preserve source config")

        for row in artifact_rows:
            name = row["relative_name"]
            source_artifact = source / name
            try:
                expected_target = str(source_artifact.resolve(strict=True))
            except OSError:
                raise ValueError(f"source artifact {name} is unavailable") from None
            if row["absolute_target"] != expected_target or Path(expected_target).parent != source:
                raise ValueError(f"linked artifact {name} target is invalid")
            artifact_snapshot = _open_regular_snapshot(source_artifact, f"source artifact {name}")
            regular_snapshots.append(artifact_snapshot)
            trusted = normalized_expected_artifacts[name]
            if artifact_snapshot.size != trusted["size"] or artifact_snapshot.sha256 != trusted["sha256"]:
                raise ValueError(f"source artifact {name} does not match expected artifacts trust input")
            link = destination / name
            try:
                link_metadata = link.lstat()
                link_target = os.readlink(link)
            except OSError:
                raise ValueError(f"linked artifact {name} is unavailable") from None
            if not stat.S_ISLNK(link_metadata.st_mode):
                raise ValueError(f"linked artifact {name} is not a symlink")
            if not os.path.isabs(link_target) or link_target != expected_target:
                raise ValueError(f"linked artifact {name} target mismatch")
            symlink_snapshots.append(
                _SymlinkSnapshot(link, f"linked artifact {name}", link_metadata, link_target),
            )

        for symlink_snapshot in symlink_snapshots:
            symlink_snapshot.revalidate()
        for regular_snapshot in regular_snapshots:
            regular_snapshot.revalidate()
        view_directory.revalidate()
        source_directory.revalidate()
        return manifest
    finally:
        for regular_snapshot in reversed(regular_snapshots):
            regular_snapshot.close()
        if view_directory is not None:
            view_directory.close()
        source_directory.close()


def prepare_reference_checkpoint(
    *,
    source_pretrained_model: Path | str,
    destination_view: Path | str,
    expected_source_config_sha256: str,
    source_repository: str,
    source_revision: str,
    expected_artifacts: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Atomically create a provenance-bound view without copying checkpoint artifacts."""
    _validate_trust_inputs(expected_source_config_sha256, source_repository, source_revision)
    normalized_expected_artifacts = _validate_expected_artifacts(expected_artifacts)
    source = _normalize_source(source_pretrained_model)
    destination = _normalize_destination(destination_view)
    _, adapted, actual_source_digest = _read_and_sanitize_source_config(
        source,
        expected_source_config_sha256,
    )
    artifact_paths = _source_artifact_paths(source, normalized_expected_artifacts)
    artifact_rows = _manifest_artifacts(source, artifact_paths, normalized_expected_artifacts)
    adapted_payload = _canonical_json(adapted)
    manifest: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "source_repository": source_repository,
        "source_revision": source_revision,
        "source_pretrained_model": str(source),
        "expected_source_config_sha256": expected_source_config_sha256,
        "actual_source_config_sha256": actual_source_digest,
        "adapted_config_sha256": hashlib.sha256(adapted_payload).hexdigest(),
        "removed_fields": dict(_REMOVED_FIELDS),
        "linked_artifacts": artifact_rows,
    }

    temporary = Path(tempfile.mkdtemp(prefix=".reference-checkpoint-view-", dir=destination.parent))
    published_metadata: os.stat_result | None = None
    try:
        _write_exclusive(temporary / _CONFIG_FILENAME, adapted_payload)
        for row in artifact_rows:
            (temporary / row["relative_name"]).symlink_to(row["absolute_target"])
        _write_exclusive(temporary / MANIFEST_FILENAME, _canonical_json(manifest))
        _fsync_directory(temporary)
        temporary_metadata = temporary.lstat()
        _publish_directory_exclusively(temporary, destination)
        published_metadata = temporary_metadata
        _fsync_directory(destination.parent)
        return verify_reference_checkpoint(
            source_pretrained_model=source,
            destination_view=destination,
            expected_source_config_sha256=expected_source_config_sha256,
            source_repository=source_repository,
            source_revision=source_revision,
            expected_artifacts=normalized_expected_artifacts,
        )
    except BaseException as error:
        if published_metadata is not None:
            try:
                _remove_exact_published_destination(destination, published_metadata)
            except BaseException as rollback_error:
                error.add_note(f"publication rollback failed: {rollback_error}")
        raise
    finally:
        if published_metadata is None:
            shutil.rmtree(temporary, ignore_errors=True)


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-pretrained-model", type=Path, required=True)
    parser.add_argument("--destination-view", type=Path, required=True)
    parser.add_argument("--expected-source-config-sha256", required=True)
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument(
        "--expected-artifacts-json",
        required=True,
        help="canonical inline JSON mapping artifact names to trusted size and sha256 values",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    create = subparsers.add_parser("create", help="create the offline inference-only view atomically")
    verify = subparsers.add_parser("verify", help="verify an existing view and all source artifacts")
    _add_shared_arguments(create)
    _add_shared_arguments(verify)
    args = parser.parse_args(argv)
    expected_artifacts = _parse_expected_artifacts_json(args.expected_artifacts_json)
    operation = prepare_reference_checkpoint if args.mode == "create" else verify_reference_checkpoint
    manifest = operation(
        source_pretrained_model=args.source_pretrained_model,
        destination_view=args.destination_view,
        expected_source_config_sha256=args.expected_source_config_sha256,
        source_repository=args.source_repository,
        source_revision=args.source_revision,
        expected_artifacts=expected_artifacts,
    )
    sys.stdout.buffer.write(_canonical_json(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
