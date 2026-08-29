"""Publish a closed public96 result only after local and public readback proof.

The immutable Hub prefix is ``public96/results/<matrix16>-<result16>`` where
each component is the first 16 lowercase hex characters of the corresponding
SHA-256.  It is intentionally content-addressed: a matching tree can resume;
any other existing tree is a collision.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import stat
import sys
import tempfile
import time
from typing import Mapping, Sequence

from scripts import eval_groot_n17_public96 as evaluator
from scripts.publish_simple_curriculum_collection import (
    CollectionPublicationError,
    HuggingFacePublicDatasetTransport,
    _retry as _canonical_retry,
    _scan_publication_descriptor_and_content_fd as _canonical_scan_descriptor,
)


_HEX = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,511}$")
_MANIFEST = "SHA256SUMS.json"
_REPOSITORY = "ryanjin333/lehome-groot-n17-rollouts"
_RFC3339_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class Public96PublicationError(RuntimeError):
    """Public96 evidence is unsafe, incomplete, or not durably public."""


@dataclass(frozen=True)
class PublicationEntry:
    relative_path: str
    sha256: str
    byte_size: int


@dataclass(frozen=True)
class _ArtifactSnapshot:
    contents: bytes
    sha256: str
    byte_size: int


@dataclass(frozen=True)
class Public96PublicationResult:
    repository: str
    ref: str
    remote_prefix: str
    immutable_revision: str
    matrix_sha256: str
    result_sha256: str
    verifier_receipt_sha256: str
    manifest_sha256: str
    entries: tuple[PublicationEntry, ...]
    entry_count: int
    tree_sha256: str


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_relative(value: object) -> str:
    if not isinstance(value, str) or not _SAFE.fullmatch(value):
        raise Public96PublicationError("publication path is not canonical")
    parts = PurePosixPath(value).parts
    if not parts or any(part in {"", ".", ".."} for part in parts) or PurePosixPath(value).is_absolute():
        raise Public96PublicationError("publication path is not canonical")
    return value


def _absolute_without_resolution(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _safe_root(path: Path) -> Path:
    root = _absolute_without_resolution(path)
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except OSError as error:
            raise Public96PublicationError("closed run root is unavailable") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise Public96PublicationError("closed run root contains a symlink")
    if not root.is_dir():
        raise Public96PublicationError("closed run root is unavailable")
    return root


def _open_entry(root: Path, relative: str) -> tuple[int, os.stat_result]:
    root_fd = -1
    current_fd = -1
    try:
        root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
        current_fd = root_fd
        for index, part in enumerate(PurePosixPath(relative).parts):
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            if index + 1 != len(PurePosixPath(relative).parts):
                flags |= getattr(os, "O_DIRECTORY", 0)
            next_fd = os.open(part, flags, dir_fd=current_fd)
            os.close(current_fd); current_fd = next_fd
        metadata = os.fstat(current_fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise Public96PublicationError("publication source contains an unsafe path")
        return current_fd, metadata
    except Public96PublicationError:
        if current_fd >= 0: os.close(current_fd)
        raise
    except OSError as error:
        if current_fd >= 0: os.close(current_fd)
        raise Public96PublicationError("publication source contains an unsafe path") from error


def _read_descriptor(root: Path, relative: str) -> tuple[bytes, str, int]:
    fd, before = _open_entry(root, _safe_relative(relative))
    try:
        try:
            _canonical_scan_descriptor(fd, relative)
        except CollectionPublicationError as error:
            raise Public96PublicationError("publication source violates the canonical credential policy") from error
        with os.fdopen(os.dup(fd), "rb", closefd=True) as handle:
            contents = handle.read()
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size) != (after.st_dev, after.st_ino, after.st_size):
            raise Public96PublicationError("publication source changed while being inspected")
        if not contents:
            raise Public96PublicationError("publication source file is empty")
        return contents, _digest_bytes(contents), len(contents)
    finally:
        os.close(fd)


def _json_descriptor(root: Path, relative: str, label: str) -> tuple[dict[str, object], _ArtifactSnapshot]:
    contents, digest, size = _read_descriptor(root, relative)
    try:
        value = json.loads(contents.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise Public96PublicationError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise Public96PublicationError(f"{label} must be an object")
    return value, _ArtifactSnapshot(contents, digest, size)


def load_token(token_file: Path) -> str:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if type(no_follow) is not int or not no_follow:
        raise Public96PublicationError("HF token file is unavailable")
    try:
        descriptor = os.open(token_file, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | no_follow)
    except OSError:
        raise Public96PublicationError("HF token file is unavailable") from None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_mode & 0o077 or before.st_uid != os.geteuid() or not 1 <= before.st_size <= 64 * 1024:
            raise Public96PublicationError("HF token file must be owner-only and regular")
        with os.fdopen(os.dup(descriptor), "rb", closefd=True) as handle:
            raw = handle.read((64 * 1024) + 1)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size) != (after.st_dev, after.st_ino, after.st_size) or len(raw) != before.st_size:
            raise Public96PublicationError("HF token file changed while being read")
        token = raw.decode("utf-8").strip()
    except UnicodeError:
        raise Public96PublicationError("HF token file is unreadable") from None
    finally:
        os.close(descriptor)
    if not token or any(character.isspace() for character in token):
        raise Public96PublicationError("HF token is unavailable")
    return token


def _artifact_matches(root: Path, descriptor: object, expected: str, label: str) -> _ArtifactSnapshot:
    if not isinstance(descriptor, Mapping) or set(descriptor) != {"relative_path", "sha256"}:
        raise Public96PublicationError(f"{label} descriptor is invalid")
    if descriptor.get("relative_path") != expected or not isinstance(descriptor.get("sha256"), str):
        raise Public96PublicationError(f"{label} descriptor does not bind the expected artifact")
    contents, actual, size = _read_descriptor(root, expected)
    if descriptor["sha256"] != actual:
        raise Public96PublicationError(f"{label} descriptor digest mismatch")
    return _ArtifactSnapshot(contents, actual, size)


def _verify_readiness_snapshot(root: Path, result: Mapping[str, object], snapshot: _ArtifactSnapshot) -> None:
    """Rebind readiness to a result-descriptor-verified stage receipt."""
    episodes = result.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise Public96PublicationError("result episode artifacts are invalid")
    artifacts = episodes[0].get("artifacts") if isinstance(episodes[0], Mapping) else None
    receipt_descriptor = artifacts.get("receipt") if isinstance(artifacts, Mapping) else None
    if not isinstance(receipt_descriptor, Mapping) or not isinstance(receipt_descriptor.get("relative_path"), str):
        raise Public96PublicationError("result stage artifact descriptor is invalid")
    relative = str(receipt_descriptor["relative_path"])
    receipt_snapshot = _entry_from_descriptor(root, relative, receipt_descriptor)
    try:
        stage_receipt = json.loads(receipt_snapshot.contents.decode("utf-8"))
        readiness = stage_receipt["policy_server_readiness"]["artifact"]
    except (UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise Public96PublicationError("stage receipt readiness binding is invalid") from error
    if not isinstance(readiness, Mapping) or readiness.get("relative_path") != "policy-server-readiness.json" or readiness.get("sha256") != snapshot.sha256:
        raise Public96PublicationError("stage receipt readiness binding changed")


def _verify_verifier_receipt(root: Path, result: Mapping[str, object], result_sha256: str, matrix_sha256: str) -> tuple[str, dict[str, _ArtifactSnapshot]]:
    receipt, receipt_snapshot = _json_descriptor(root, "verifier-receipt.json", "verifier receipt")
    required = {"kind", "result", "policy_server_log", "summary", "matrix_sha256", "checkpoint", "raw_checker_overlay", "publication"}
    if set(receipt) != required or receipt.get("kind") != "lehome_groot_n17_public96_verifier_receipt_v1":
        raise Public96PublicationError("verifier receipt schema is invalid")
    result_snapshot = _artifact_matches(root, receipt.get("result"), "result.json", "verifier receipt result")
    if result_snapshot.sha256 != result_sha256:
        raise Public96PublicationError("verifier receipt result digest mismatch")
    log_snapshot = _artifact_matches(root, receipt.get("policy_server_log"), "policy-server.log", "verifier receipt policy log")
    for field in ("summary", "matrix_sha256", "checkpoint", "raw_checker_overlay", "publication"):
        if receipt.get(field) != result.get(field):
            raise Public96PublicationError("verifier receipt does not bind the current result")
    if receipt["matrix_sha256"] != matrix_sha256:
        raise Public96PublicationError("verifier receipt matrix digest mismatch")
    readiness_contents, readiness_sha256, readiness_size = _read_descriptor(root, "policy-server-readiness.json")
    readiness_snapshot = _ArtifactSnapshot(readiness_contents, readiness_sha256, readiness_size)
    _verify_readiness_snapshot(root, result, readiness_snapshot)
    return receipt_snapshot.sha256, {
        "result.json": result_snapshot,
        "verifier-receipt.json": receipt_snapshot,
        "policy-server-readiness.json": readiness_snapshot,
        "policy-server.log": log_snapshot,
    }


def _entry_from_descriptor(root: Path, relative: str, descriptor: object) -> _ArtifactSnapshot:
    if not isinstance(descriptor, Mapping) or set(descriptor) != {"relative_path", "sha256"} or descriptor.get("relative_path") != relative:
        raise Public96PublicationError("result artifact descriptor is invalid")
    contents, digest, size = _read_descriptor(root, relative)
    if descriptor.get("sha256") != digest:
        raise Public96PublicationError("result artifact descriptor digest mismatch")
    return _ArtifactSnapshot(contents, digest, size)


def _collect_entries(root: Path, result: Mapping[str, object], root_snapshots: Mapping[str, _ArtifactSnapshot]) -> tuple[PublicationEntry, ...]:
    entries: dict[str, PublicationEntry] = {}

    def add(entry: PublicationEntry) -> None:
        old = entries.setdefault(entry.relative_path, entry)
        if old != entry:
            raise Public96PublicationError("duplicate result artifact paths disagree")

    for relative in ("result.json", "verifier-receipt.json", "policy-server-readiness.json", "policy-server.log"):
        expected = root_snapshots.get(relative)
        if expected is None:
            raise Public96PublicationError("verified root artifact snapshot is missing")
        _, digest, size = _read_descriptor(root, relative)
        if (digest, size) != (expected.sha256, expected.byte_size):
            raise Public96PublicationError("verified root artifact changed before staging")
        add(PublicationEntry(relative, expected.sha256, expected.byte_size))
    episodes = result.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 96:
        raise Public96PublicationError("result does not contain exactly 96 episodes")
    stage_logs: set[str] = set(); stage_receipts: set[str] = set(); videos: set[str] = set()
    for episode in episodes:
        if not isinstance(episode, Mapping) or not isinstance(episode.get("artifacts"), Mapping):
            raise Public96PublicationError("result episode artifacts are invalid")
        artifacts = episode["artifacts"]
        for key, seen in (("log", stage_logs), ("receipt", stage_receipts)):
            descriptor = artifacts.get(key)
            if not isinstance(descriptor, Mapping) or not isinstance(descriptor.get("relative_path"), str):
                raise Public96PublicationError("result stage artifact descriptor is invalid")
            relative = str(descriptor["relative_path"]); seen.add(relative); snapshot = _entry_from_descriptor(root, relative, descriptor); add(PublicationEntry(relative, snapshot.sha256, snapshot.byte_size))
        videos_map = artifacts.get("videos")
        if not isinstance(videos_map, Mapping) or set(videos_map) != {"top_rgb", "left_rgb", "right_rgb"}:
            raise Public96PublicationError("result video artifact descriptors are invalid")
        for descriptor in videos_map.values():
            if not isinstance(descriptor, Mapping) or not isinstance(descriptor.get("relative_path"), str):
                raise Public96PublicationError("result video artifact descriptor is invalid")
            relative = str(descriptor["relative_path"]); videos.add(relative); snapshot = _entry_from_descriptor(root, relative, descriptor); add(PublicationEntry(relative, snapshot.sha256, snapshot.byte_size))
    if len(stage_logs) != 48 or len(stage_receipts) != 48 or len(videos) != 288 or len(entries) != 388:
        raise Public96PublicationError("public96 allowlist does not contain exactly 48 stages and 288 videos")
    return tuple(sorted(entries.values(), key=lambda item: item.relative_path))


def _entry_digest(entries: Sequence[PublicationEntry]) -> str:
    return _digest_bytes(_canonical([{"relative_path": entry.relative_path, "sha256": entry.sha256, "byte_size": entry.byte_size} for entry in entries]))


def _stage(root: Path, entries: Sequence[PublicationEntry], root_snapshots: Mapping[str, _ArtifactSnapshot]) -> tuple[Path, tuple[PublicationEntry, ...], str]:
    staging = Path(tempfile.mkdtemp(prefix="lehome-public96-stage-", dir=root.parent))
    try:
        for entry in entries:
            snapshot = root_snapshots.get(entry.relative_path)
            if snapshot is None:
                contents, digest, size = _read_descriptor(root, entry.relative_path)
            else:
                contents, digest, size = snapshot.contents, snapshot.sha256, snapshot.byte_size
            if (digest, size) != (entry.sha256, entry.byte_size):
                raise Public96PublicationError("publication source changed after validation")
            target = staging / entry.relative_path; target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(contents); handle.flush(); os.fsync(handle.fileno())
            staged_contents, staged_digest, staged_size = _read_descriptor(staging, entry.relative_path)
            if (staged_contents, staged_digest, staged_size) != (contents, entry.sha256, entry.byte_size):
                raise Public96PublicationError("publication staging bytes changed")
        manifest = _canonical([{"path": entry.relative_path, "sha256": entry.sha256, "byte_size": entry.byte_size} for entry in entries])
        manifest_path = staging / _MANIFEST
        with manifest_path.open("xb") as handle:
            handle.write(manifest); handle.flush(); os.fsync(handle.fileno())
        _, manifest_digest, manifest_size = _read_descriptor(staging, _MANIFEST)
        if (manifest_digest, manifest_size) != (_digest_bytes(manifest), len(manifest)):
            raise Public96PublicationError("publication manifest staging bytes changed")
        manifest_entry = PublicationEntry(_MANIFEST, manifest_digest, manifest_size)
        return staging, tuple((*entries, manifest_entry)), manifest_entry.sha256
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _tree_files(tree: Sequence[object], prefix: str) -> set[str]:
    observed: set[str] = set(); count = 0; base = prefix + "/"
    for item in tree:
        path, entry_type = getattr(item, "relative_path", None), getattr(item, "entry_type", None)
        if not isinstance(path, str) or not isinstance(entry_type, str) or not path.startswith(base):
            raise Public96PublicationError("remote tree is malformed")
        if entry_type == "file":
            relative = path.removeprefix(base)
            if relative in observed: raise Public96PublicationError("remote tree contains duplicate paths")
            observed.add(relative); count += 1
        elif entry_type != "directory":
            raise Public96PublicationError("remote tree contains an unsafe entry")
    if count != len(observed): raise Public96PublicationError("remote tree contains duplicate paths")
    return observed


def _transport_retry(operation, *, label: str, deadline: float, monotonic) -> object:
    """Use the repository's bounded Hub retry/deadline policy verbatim."""
    try:
        return _canonical_retry(
            operation, label=label, max_attempts=3, deadline=deadline,
            monotonic=monotonic, deadline_label="public96 publication",
        )
    except CollectionPublicationError as error:
        raise Public96PublicationError(f"{label} failed") from error


def _verify_download(
    transport, *, repository: str, revision: str, prefix: str, entries: Sequence[PublicationEntry],
    token: str | None, staging: Path, deadline: float, monotonic,
) -> None:
    destination = Path(tempfile.mkdtemp(prefix="lehome-public96-readback-", dir=staging.parent))
    try:
        observed = _transport_retry(
            lambda budget: transport.download_files(
                repository=repository, revision=revision, destination=destination,
                relative_paths=tuple(entry.relative_path for entry in entries), token=token,
                remote_prefix=prefix, timeout_seconds=budget,
            ),
            label="anonymous readback" if token is None else "authenticated readback",
            deadline=deadline, monotonic=monotonic,
        )
        if observed != revision:
            raise Public96PublicationError("readback did not bind the immutable revision")
        for entry in entries:
            contents, digest, size = _read_descriptor(destination, entry.relative_path)
            if (digest, size) != (entry.sha256, entry.byte_size):
                raise Public96PublicationError("authenticated readback bytes mismatch" if token is not None else "anonymous readback bytes mismatch")
    finally:
        shutil.rmtree(destination, ignore_errors=True)


def _open_receipt_parent(path: Path) -> tuple[str, int]:
    """Pin a non-symlink receipt parent before inspecting or creating a file."""
    absolute = _absolute_without_resolution(path)
    parent = absolute.parent
    current = Path(parent.anchor)
    for part in parent.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except OSError as error:
            raise Public96PublicationError("receipt parent is unavailable") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise Public96PublicationError("receipt parent contains a symlink")
    try:
        descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise Public96PublicationError("receipt parent is unavailable") from error
    if not path.name or path.name in {".", ".."}:
        os.close(descriptor)
        raise Public96PublicationError("receipt path is invalid")
    return path.name, descriptor


def _read_existing_receipt(path: Path) -> bytes | None:
    name, parent = _open_receipt_parent(path)
    try:
        try:
            descriptor = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent)
        except FileNotFoundError:
            return None
        except OSError as error:
            raise Public96PublicationError("publication receipt already exists or is unsafe") from error
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise Public96PublicationError("publication receipt already exists or is unsafe")
            with os.fdopen(os.dup(descriptor), "rb", closefd=True) as handle:
                contents = handle.read()
            after = os.fstat(descriptor)
            if (before.st_dev, before.st_ino, before.st_size) != (after.st_dev, after.st_ino, after.st_size):
                raise Public96PublicationError("publication receipt already exists or is unsafe")
            return contents
        finally:
            os.close(descriptor)
    finally:
        os.close(parent)


def _write_new_receipt(path: Path, contents: bytes) -> None:
    name, parent = _open_receipt_parent(path)
    temporary = f".{name}.{secrets.token_hex(16)}.tmp"
    descriptor = -1
    try:
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=parent)
            total = 0
            while total < len(contents):
                total += os.write(descriptor, contents[total:])
            os.fsync(descriptor)
        except OSError as error:
            raise Public96PublicationError("publication receipt could not be durably written") from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        try:
            # link(2) is an atomic no-replace promotion in the same pinned
            # directory; rename would permit replacing another writer's receipt.
            os.link(temporary, name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
        except FileExistsError as error:
            raise Public96PublicationError("publication receipt already exists") from error
        except OSError as error:
            raise Public96PublicationError("publication receipt could not be atomically promoted") from error
        finally:
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass
        try:
            os.fsync(parent)
        except OSError as error:
            raise Public96PublicationError("publication receipt directory fsync failed") from error
    finally:
        os.close(parent)


def _write_or_validate_receipt(path: Path, payload: Mapping[str, object]) -> None:
    required = {
        "schema_version", "kind", "repository", "mutable_ref", "remote_prefix", "immutable_revision",
        "matrix_sha256", "result_sha256", "verifier_receipt_sha256", "manifest_sha256", "entry_count",
        "tree_sha256", "authenticated_readback_verified", "anonymous_readback_verified", "published_at_utc",
    }

    def valid(value: object) -> bool:
        if not isinstance(value, Mapping) or set(value) != required:
            return False
        if value.get("schema_version") != 1 or type(value.get("schema_version")) is not int or value.get("kind") != "lehome_groot_n17_public96_publication_receipt_v1":
            return False
        if not all(isinstance(value.get(field), str) and value[field] for field in ("repository", "mutable_ref", "remote_prefix")):
            return False
        if not re.fullmatch(r"public96/results/[0-9a-f]{16}-[0-9a-f]{16}", str(value.get("remote_prefix"))):
            return False
        if not isinstance(value.get("immutable_revision"), str) or _COMMIT.fullmatch(value["immutable_revision"]) is None:
            return False
        if any(not isinstance(value.get(field), str) or _HEX.fullmatch(value[field]) is None for field in ("matrix_sha256", "result_sha256", "verifier_receipt_sha256", "manifest_sha256", "tree_sha256")):
            return False
        if type(value.get("entry_count")) is not int or value["entry_count"] < 1:
            return False
        if value.get("authenticated_readback_verified") is not True or value.get("anonymous_readback_verified") is not True:
            return False
        timestamp = value.get("published_at_utc")
        if not isinstance(timestamp, str) or _RFC3339_UTC.fullmatch(timestamp) is None:
            return False
        try:
            datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            return False
        return True

    if not valid(payload):
        raise Public96PublicationError("publication receipt payload is invalid")
    existing_contents = _read_existing_receipt(path)
    if existing_contents is not None:
        try: existing = json.loads(existing_contents.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error: raise Public96PublicationError("publication receipt already exists or is unsafe") from error
        stable = {key: value for key, value in payload.items() if key != "published_at_utc"}
        if not valid(existing) or {key: existing.get(key) for key in stable} != stable:
            raise Public96PublicationError("publication receipt already exists and differs")
        return
    _write_new_receipt(path, (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def publish_public96(
    run_root: Path, *, matrix: Path, matrix_sha256_path: Path, token: str, transport,
    repository: str = _REPOSITORY, ref: str = "main", receipt_output: Path | None = None,
    timeout_seconds: float = 300.0, monotonic=time.monotonic,
) -> Public96PublicationResult:
    """Verify a closed run, publish only its exact evidence, then read it twice."""
    if not isinstance(token, str) or not token or any(character.isspace() for character in token):
        raise Public96PublicationError("HF token is unavailable")
    if not isinstance(repository, str) or "/" not in repository or not repository.strip() or not isinstance(ref, str) or not ref or _COMMIT.fullmatch(ref):
        raise Public96PublicationError("publication repository or mutable ref is invalid")
    if type(timeout_seconds) not in (int, float) or not 0 < timeout_seconds <= 900 or not callable(monotonic):
        raise Public96PublicationError("publication timeout is invalid")
    root = _safe_root(Path(run_root))
    result, result_snapshot = _json_descriptor(root, "result.json", "result")
    result_sha256 = result_snapshot.sha256
    try:
        stages = evaluator.load_frozen_matrix(matrix, matrix_sha256_path)
        matrix_digest = evaluator._matrix_digest(matrix, matrix_sha256_path)
        summary = evaluator.verify_result(result, stages=stages, matrix_sha256=matrix_digest, output_root=root)
    except (OSError, ValueError, evaluator.Public96ContractError) as error:
        raise Public96PublicationError("result verifier rejected the closed run") from error
    if summary != result.get("summary"):
        raise Public96PublicationError("result verifier summary mismatch")
    verifier_receipt_sha256, root_snapshots = _verify_verifier_receipt(root, result, result_sha256, matrix_digest)
    raw_entries = _collect_entries(root, result, root_snapshots)
    staging, entries, manifest_sha256 = _stage(root, raw_entries, root_snapshots)
    prefix = f"public96/results/{matrix_digest[:16]}-{result_sha256[:16]}"
    deadline = monotonic() + float(timeout_seconds)

    def resolve_head() -> str:
        head = _transport_retry(
            lambda budget: transport.resolve_approved_ref(repository=repository, ref=ref, token=token, timeout_seconds=budget),
            label="publication ref resolution", deadline=deadline, monotonic=monotonic,
        )
        if not isinstance(head, str) or not _COMMIT.fullmatch(head):
            raise Public96PublicationError("publication ref did not resolve to an immutable revision")
        return head

    def list_prefix(revision: str):
        return _tree_files(_transport_retry(
            lambda budget: transport.list_tree(repository=repository, revision=revision, token=token, remote_prefix=prefix, timeout_seconds=budget),
            label="immutable prefix listing", deadline=deadline, monotonic=monotonic,
        ), prefix)

    try:
        try:
            head = resolve_head()
            expected = {entry.relative_path for entry in entries}
            existing = list_prefix(head)
            if existing and existing != expected: raise Public96PublicationError("immutable public96 prefix collision")
            if existing:
                revision = head
                try: _verify_download(transport, repository=repository, revision=revision, prefix=prefix, entries=entries, token=token, staging=staging, deadline=deadline, monotonic=monotonic)
                except Public96PublicationError as error: raise Public96PublicationError("immutable public96 prefix collision") from error
            else:
                try:
                    revision = _canonical_retry(
                        lambda budget: transport.upload_files(
                            repository=repository, revision=ref, source=staging, entries=entries, token=token,
                            remote_prefix=prefix, parent_commit=head, timeout_seconds=budget,
                        ),
                        label="public96 upload", max_attempts=3, deadline=deadline, monotonic=monotonic,
                        deadline_label="public96 publication", reconcile_ambiguous=True,
                    )
                except CollectionPublicationError:
                    # A disconnected write can have committed.  Reconcile the
                    # current mutable head to this exact content-addressed
                    # prefix; never issue another blind mutable upload.
                    revision = resolve_head()
                    if list_prefix(revision) != expected:
                        raise Public96PublicationError("ambiguous upload did not reconcile to the exact immutable tree") from None
                    try:
                        _verify_download(transport, repository=repository, revision=revision, prefix=prefix, entries=entries, token=token, staging=staging, deadline=deadline, monotonic=monotonic)
                    except Public96PublicationError as error:
                        raise Public96PublicationError("ambiguous upload did not reconcile to exact authenticated bytes") from error
                if not isinstance(revision, str) or not _COMMIT.fullmatch(revision): raise Public96PublicationError("upload did not return an immutable revision")
            remote = list_prefix(revision)
            if remote != expected: raise Public96PublicationError("immutable remote tree does not match the staged allowlist")
            _verify_download(transport, repository=repository, revision=revision, prefix=prefix, entries=entries, token=token, staging=staging, deadline=deadline, monotonic=monotonic)
            _verify_download(transport, repository=repository, revision=revision, prefix=prefix, entries=entries, token=None, staging=staging, deadline=deadline, monotonic=monotonic)
        except Public96PublicationError:
            raise
        except Exception as error:
            raise Public96PublicationError("Hub transport/readback failed") from error
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    tree_sha256 = _entry_digest(entries)
    result_value = Public96PublicationResult(repository, ref, prefix, revision, matrix_digest, result_sha256, verifier_receipt_sha256, manifest_sha256, entries, len(entries), tree_sha256)
    receipt = receipt_output or root / "public96-publication-receipt.json"
    _write_or_validate_receipt(Path(receipt), {"schema_version": 1, "kind": "lehome_groot_n17_public96_publication_receipt_v1", "repository": repository, "mutable_ref": ref, "remote_prefix": prefix, "immutable_revision": revision, "matrix_sha256": matrix_digest, "result_sha256": result_sha256, "verifier_receipt_sha256": verifier_receipt_sha256, "manifest_sha256": manifest_sha256, "entry_count": len(entries), "tree_sha256": tree_sha256, "authenticated_readback_verified": True, "anonymous_readback_verified": True, "published_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")})
    return result_value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True); parser.add_argument("--matrix", type=Path, required=True); parser.add_argument("--matrix-sha256", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True); parser.add_argument("--repository", default=_REPOSITORY); parser.add_argument("--ref", default="main"); parser.add_argument("--receipt-output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = publish_public96(args.run_root, matrix=args.matrix, matrix_sha256_path=args.matrix_sha256, token=load_token(args.token_file), repository=args.repository, ref=args.ref, receipt_output=args.receipt_output, transport=HuggingFacePublicDatasetTransport())
    except Public96PublicationError as error:
        print(f"public96 publication failed: {error}", file=sys.stderr); return 2
    except Exception:
        print("public96 publication failed: transport or dependency error", file=sys.stderr); return 2
    print(json.dumps({"immutable_revision": result.immutable_revision, "remote_prefix": result.remote_prefix, "entry_count": result.entry_count}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
