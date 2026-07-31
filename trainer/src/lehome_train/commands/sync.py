"""Closed-allowlist model artifact synchronization with immutable readback."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable, Iterable, Mapping

from lehome_train.constants import DEFAULT_MODEL_REPO
from lehome_train.hub import HubTransport, download_files, require_access, upload_files
from lehome_train.io import atomic_write_json
from lehome_train.models import SyncEntry, SyncManifest
from lehome_train.redaction import ArtifactRejected, generate_upload_allowlist


_ROOT_ARTIFACTS = ("provenance.json", "resolved-config.json")
_GROUPS = ("checkpoints", "logs", "reports")
_MANIFEST_NAME = "sync-manifest.json"
_MINIMUM_STAGING_RESERVE_BYTES = 64 * 1024**2
_MAXIMUM_STAGING_RESERVE_BYTES = 1024**3
_STAGING_RESERVE_FRACTION_DENOMINATOR = 20


@dataclass(frozen=True, slots=True)
class SyncResult:
    """Remote commit plus the per-artifact hash verification disposal gate."""

    repository: str
    immutable_revision: str
    manifest: SyncManifest
    disposable: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "repository": self.repository,
            "immutable_revision": self.immutable_revision,
            "manifest": self.manifest.to_dict(),
            "disposable": self.disposable,
        }


def _walk_group(root: Path, group: str) -> tuple[str, ...]:
    directory = root / group
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"sync requires the {group} artifact group")
    paths: list[str] = []
    pending = [directory]
    while pending:
        current = pending.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                relative = Path(entry.path).relative_to(root).as_posix()
                if entry.is_symlink():
                    paths.append(relative)
                elif entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    if group != "reports" or relative.endswith(".json"):
                        paths.append(relative)
                else:
                    paths.append(relative)
    if not paths:
        raise ValueError(f"sync requires the {group} artifact group")
    return tuple(paths)


def _generated_paths(experiment_root: Path) -> tuple[str, ...]:
    paths: list[str] = []
    for relative_path in _ROOT_ARTIFACTS:
        path = experiment_root / relative_path
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"sync requires {relative_path}")
        paths.append(relative_path)
    for group in _GROUPS:
        paths.extend(_walk_group(experiment_root, group))
    return tuple(sorted(paths))


def generate_sync_manifest(
    experiment_root: str | os.PathLike[str],
    *,
    experiment_id: str,
    experiment_config_sha256: str,
    relative_paths: Iterable[str | os.PathLike[str]] | None = None,
) -> SyncManifest:
    """Generate and atomically record the only artifact upload allowlist."""

    root = Path(experiment_root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("experiment root must be an existing regular directory")
    paths = (
        _generated_paths(root)
        if relative_paths is None
        else tuple(os.fspath(path) for path in relative_paths)
    )
    if _MANIFEST_NAME in paths:
        raise ValueError("sync manifest cannot include its own mutable path")
    try:
        entries = generate_upload_allowlist(root, paths)
    except ArtifactRejected as error:
        raise ValueError(str(error)) from None
    manifest = SyncManifest(
        experiment_id=experiment_id,
        experiment_config_sha256=experiment_config_sha256,
        entries=entries,
    )
    atomic_write_json(root / _MANIFEST_NAME, manifest.to_dict())
    return manifest


def _verified_remote_entries(
    readback: Path,
    entries: tuple[SyncEntry, ...],
) -> tuple[SyncEntry, ...]:
    verified: list[SyncEntry] = []
    for expected in entries:
        matches = False
        try:
            observed = generate_upload_allowlist(readback, (expected.relative_path,))
            matches = (
                len(observed) == 1
                and observed[0].sha256 == expected.sha256
                and observed[0].byte_size == expected.byte_size
            )
        except (ArtifactRejected, OSError, ValueError):
            matches = False
        verified.append(replace(expected, remotely_verified=matches))
    return tuple(verified)


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
    available = free_space_probe(staging_root)
    if type(available) is not int or available < 0:
        raise ValueError(f"sync {phase} free-space probe returned an invalid value")
    required = _required_staging_bytes(entries)
    if available < required:
        raise ValueError(
            f"sync {phase} staging filesystem has insufficient space "
            f"(requires {required} bytes, has {available} bytes)"
        )


def _complete_tree_matches(root: Path, entries: tuple[SyncEntry, ...]) -> bool:
    expected_files = {entry.relative_path for entry in entries}
    expected_directories = {
        "/".join(parts[:index])
        for relative_path in expected_files
        for parts in (relative_path.split("/"),)
        for index in range(1, len(parts))
    }
    observed_files: set[str] = set()
    pending = [(root, "")]
    try:
        while pending:
            directory, prefix = pending.pop()
            for entry in os.scandir(directory):
                relative = f"{prefix}/{entry.name}" if prefix else entry.name
                if entry.is_symlink():
                    return False
                if entry.is_dir(follow_symlinks=False):
                    if relative not in expected_directories:
                        return False
                    pending.append((Path(entry.path), relative))
                elif entry.is_file(follow_symlinks=False):
                    observed_files.add(relative)
                else:
                    return False
    except OSError:
        return False
    return observed_files == expected_files


def _stage_entries(
    source: Path,
    entries: tuple[SyncEntry, ...],
    *,
    staging_root: Path,
) -> Path:
    staging = Path(
        tempfile.mkdtemp(prefix="lehome-model-upload-", dir=staging_root)
    )
    try:
        for entry in entries:
            destination = staging / entry.relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / entry.relative_path, destination)
        observed = generate_upload_allowlist(
            staging,
            tuple(entry.relative_path for entry in entries),
        )
        if observed != entries or not _complete_tree_matches(staging, entries):
            raise ValueError("staged sync artifact hashes changed during copy")
        return staging
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def sync_experiment(
    experiment_root: str | os.PathLike[str],
    *,
    experiment_id: str,
    experiment_config_sha256: str,
    repository: str,
    revision: str,
    transport: HubTransport,
    staging_root: str | os.PathLike[str],
    environ: Mapping[str, str] | None = None,
    max_attempts: int = 5,
    free_space_probe: Callable[[Path], int] = _free_space_bytes,
) -> SyncResult:
    """Upload generated artifacts and derive disposal only from hash readback."""

    if repository != DEFAULT_MODEL_REPO:
        raise ValueError(f"training artifacts may only be synchronized to {DEFAULT_MODEL_REPO}")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("model artifact upload revision must be explicit")
    root = Path(experiment_root)
    resolved_staging_root = Path(staging_root)
    if not resolved_staging_root.is_dir() or resolved_staging_root.is_symlink():
        raise ValueError("sync staging root must be an existing regular directory")
    require_access(
        transport=transport,
        repository=repository,
        read=True,
        write=True,
        environ=environ,
    )
    manifest = generate_sync_manifest(
        root,
        experiment_id=experiment_id,
        experiment_config_sha256=experiment_config_sha256,
    )
    _require_staging_capacity(
        resolved_staging_root,
        manifest.entries,
        free_space_probe,
        phase="upload",
    )
    staging = _stage_entries(
        root,
        manifest.entries,
        staging_root=resolved_staging_root,
    )
    try:
        immutable_revision = upload_files(
            transport=transport,
            repository=repository,
            revision=revision,
            source=staging,
            entries=manifest.entries,
            environ=environ,
            max_attempts=max_attempts,
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    _require_staging_capacity(
        resolved_staging_root,
        manifest.entries,
        free_space_probe,
        phase="readback",
    )
    readback = Path(
        tempfile.mkdtemp(
            prefix="lehome-model-readback-",
            dir=resolved_staging_root,
        )
    )
    try:
        download_files(
            transport=transport,
            repository=repository,
            revision=immutable_revision,
            destination=readback,
            relative_paths=tuple(entry.relative_path for entry in manifest.entries),
            environ=environ,
            max_attempts=max_attempts,
        )
        if _complete_tree_matches(readback, manifest.entries):
            verified_entries = _verified_remote_entries(readback, manifest.entries)
        else:
            verified_entries = tuple(
                replace(entry, remotely_verified=False) for entry in manifest.entries
            )
    finally:
        shutil.rmtree(readback, ignore_errors=True)

    verified_manifest = SyncManifest(
        experiment_id=manifest.experiment_id,
        experiment_config_sha256=manifest.experiment_config_sha256,
        entries=verified_entries,
    )
    atomic_write_json(root / _MANIFEST_NAME, verified_manifest.to_dict())
    disposable = bool(verified_entries) and all(
        entry.remotely_verified for entry in verified_entries
    )
    return SyncResult(
        repository=repository,
        immutable_revision=immutable_revision,
        manifest=verified_manifest,
        disposable=disposable,
    )
