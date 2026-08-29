#!/usr/bin/env python3
"""Create or verify an offline, inference-only view of the pinned GR00T checkpoint."""

from __future__ import annotations

import argparse
import ctypes
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


def _regular_file_metadata(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError:
        raise ValueError(f"{label} is unavailable") from None
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file and not a symlink")
    return metadata


def _inspect_regular_file(
    path: Path,
    label: str,
    *,
    capture_payload: bool,
) -> tuple[int, str, bytes | None]:
    before = _regular_file_metadata(path, label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise ValueError(f"{label} cannot be opened safely") from None
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if capture_payload else None
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(f"{label} changed while opening")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after_read = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = _regular_file_metadata(path, label)
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after_read, field) for field in stable_fields):
        raise ValueError(f"{label} changed while hashing")
    if any(getattr(after_read, field) != getattr(after, field) for field in stable_fields):
        raise ValueError(f"{label} changed after hashing")
    payload = b"".join(chunks) if chunks is not None else None
    return after_read.st_size, digest.hexdigest(), payload


def _hash_regular_file(path: Path, label: str) -> tuple[int, str]:
    size, digest, _ = _inspect_regular_file(path, label, capture_payload=False)
    return size, digest


def _read_regular_file(path: Path, label: str) -> tuple[bytes, str]:
    size, digest, payload = _inspect_regular_file(path, label, capture_payload=True)
    if payload is None or len(payload) != size:
        raise ValueError(f"{label} could not be captured safely")
    return payload, digest


def _read_and_sanitize_source_config(
    source: Path,
    expected_sha256: str,
) -> tuple[dict[str, object], dict[str, object], str]:
    payload, actual_sha256 = _read_regular_file(source / _CONFIG_FILENAME, "source config.json")
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


def _source_artifact_paths(source: Path) -> list[Path]:
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


def _manifest_artifacts(source: Path, artifact_paths: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for artifact in artifact_paths:
        size, digest = _hash_regular_file(artifact, f"source artifact {artifact.name}")
        target = artifact.resolve(strict=True)
        if target.parent != source:
            raise ValueError(f"source artifact {artifact.name} resolves outside source directory")
        rows.append(
            {
                "relative_name": artifact.name,
                "size": size,
                "sha256": digest,
                "absolute_target": str(target),
            },
        )
    return rows


def _validate_manifest_schema(manifest: dict[str, object]) -> list[dict[str, object]]:
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
    return checked


def verify_reference_checkpoint(
    *,
    source_pretrained_model: Path | str,
    destination_view: Path | str,
    expected_source_config_sha256: str,
    source_repository: str,
    source_revision: str,
) -> dict[str, object]:
    """Independently verify a compatibility view against explicit trust inputs."""
    _validate_trust_inputs(expected_source_config_sha256, source_repository, source_revision)
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

    manifest_path = destination / MANIFEST_FILENAME
    manifest_payload, _ = _read_regular_file(manifest_path, "view manifest")
    manifest = _parse_json_object(manifest_payload, "view manifest")
    if manifest_payload != _canonical_json(manifest):
        raise ValueError("view manifest is not canonical JSON")
    artifact_rows = _validate_manifest_schema(manifest)
    if manifest["source_repository"] != source_repository:
        raise ValueError("manifest source repository does not match trust input")
    if manifest["source_revision"] != source_revision:
        raise ValueError("manifest source revision does not match trust input")
    if manifest["source_pretrained_model"] != str(source):
        raise ValueError("manifest source pretrained_model does not match trust input")
    if manifest["expected_source_config_sha256"] != expected_source_config_sha256:
        raise ValueError("manifest expected source config SHA-256 does not match trust input")

    source_config, expected_adapted, actual_source_digest = _read_and_sanitize_source_config(
        source,
        expected_source_config_sha256,
    )
    if manifest["actual_source_config_sha256"] != actual_source_digest:
        raise ValueError("manifest actual source config SHA-256 is invalid")

    adapted_payload, adapted_digest = _read_regular_file(destination / _CONFIG_FILENAME, "adapted config.json")
    if adapted_payload != _canonical_json(expected_adapted):
        raise ValueError("adapted config is not the deterministic canonical sanitization")
    if adapted_digest != manifest["adapted_config_sha256"]:
        raise ValueError("adapted config SHA-256 does not match manifest")
    adapted = _parse_json_object(adapted_payload, "adapted config.json")
    if adapted != expected_adapted:
        raise ValueError("adapted config semantics do not match source config")
    if set(adapted) != set(source_config) - set(_REMOVED_FIELDS):
        raise ValueError("adapted config keys do not exactly preserve source config")

    expected_source_names = {_CONFIG_FILENAME}
    expected_view_names = {_CONFIG_FILENAME, MANIFEST_FILENAME}
    for row in artifact_rows:
        name = row["relative_name"]
        expected_source_names.add(name)
        expected_view_names.add(name)
        source_artifact = source / name
        expected_target = str(source_artifact.resolve(strict=True))
        if row["absolute_target"] != expected_target or Path(expected_target).parent != source:
            raise ValueError(f"linked artifact {name} target is invalid")
        size, digest = _hash_regular_file(source_artifact, f"source artifact {name}")
        if size != row["size"]:
            raise ValueError(f"linked artifact {name} size mismatch")
        if digest != row["sha256"]:
            raise ValueError(f"linked artifact {name} SHA-256 mismatch")
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

    try:
        actual_source_names = {entry.name for entry in source.iterdir()}
        actual_view_names = {entry.name for entry in destination.iterdir()}
    except OSError:
        raise ValueError("checkpoint directories cannot be enumerated safely") from None
    if actual_source_names != expected_source_names:
        raise ValueError("source pretrained_model entries do not match manifest")
    if actual_view_names != expected_view_names:
        raise ValueError("destination view entries do not match manifest")
    return manifest


def prepare_reference_checkpoint(
    *,
    source_pretrained_model: Path | str,
    destination_view: Path | str,
    expected_source_config_sha256: str,
    source_repository: str,
    source_revision: str,
) -> dict[str, object]:
    """Atomically create a provenance-bound view without copying checkpoint artifacts."""
    _validate_trust_inputs(expected_source_config_sha256, source_repository, source_revision)
    source = _normalize_source(source_pretrained_model)
    destination = _normalize_destination(destination_view)
    _, adapted, actual_source_digest = _read_and_sanitize_source_config(
        source,
        expected_source_config_sha256,
    )
    artifact_paths = _source_artifact_paths(source)
    artifact_rows = _manifest_artifacts(source, artifact_paths)
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
    published = False
    try:
        _write_exclusive(temporary / _CONFIG_FILENAME, adapted_payload)
        for row in artifact_rows:
            (temporary / row["relative_name"]).symlink_to(row["absolute_target"])
        _write_exclusive(temporary / MANIFEST_FILENAME, _canonical_json(manifest))
        _fsync_directory(temporary)
        verify_reference_checkpoint(
            source_pretrained_model=source,
            destination_view=temporary,
            expected_source_config_sha256=expected_source_config_sha256,
            source_repository=source_repository,
            source_revision=source_revision,
        )
        _publish_directory_exclusively(temporary, destination)
        published = True
        _fsync_directory(destination.parent)
    finally:
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)
    return verify_reference_checkpoint(
        source_pretrained_model=source,
        destination_view=destination,
        expected_source_config_sha256=expected_source_config_sha256,
        source_repository=source_repository,
        source_revision=source_revision,
    )


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-pretrained-model", type=Path, required=True)
    parser.add_argument("--destination-view", type=Path, required=True)
    parser.add_argument("--expected-source-config-sha256", required=True)
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--source-revision", required=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    create = subparsers.add_parser("create", help="create the offline inference-only view atomically")
    verify = subparsers.add_parser("verify", help="verify an existing view and all source artifacts")
    _add_shared_arguments(create)
    _add_shared_arguments(verify)
    args = parser.parse_args(argv)
    operation = prepare_reference_checkpoint if args.mode == "create" else verify_reference_checkpoint
    manifest = operation(
        source_pretrained_model=args.source_pretrained_model,
        destination_view=args.destination_view,
        expected_source_config_sha256=args.expected_source_config_sha256,
        source_repository=args.source_repository,
        source_revision=args.source_revision,
    )
    sys.stdout.buffer.write(_canonical_json(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
