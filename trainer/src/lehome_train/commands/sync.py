"""Closed-allowlist model artifact synchronization with immutable readback."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping

from lehome_train.constants import DEFAULT_MODEL_REPO
from lehome_train.hub import HubTransport, download_files, require_access, upload_files
from lehome_train.io import atomic_write_json
from lehome_train.models import SyncEntry, SyncManifest
from lehome_train.redaction import ArtifactRejected, generate_upload_allowlist


_ROOT_ARTIFACTS = ("provenance.json", "resolved-config.json")
_GROUPS = ("checkpoints", "logs", "reports")
_MANIFEST_NAME = "sync-manifest.json"


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


def sync_experiment(
    experiment_root: str | os.PathLike[str],
    *,
    experiment_id: str,
    experiment_config_sha256: str,
    repository: str,
    revision: str,
    transport: HubTransport,
    environ: Mapping[str, str] | None = None,
    max_attempts: int = 5,
) -> SyncResult:
    """Upload generated artifacts and derive disposal only from hash readback."""

    if repository != DEFAULT_MODEL_REPO:
        raise ValueError(f"training artifacts may only be synchronized to {DEFAULT_MODEL_REPO}")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("model artifact upload revision must be explicit")
    root = Path(experiment_root)
    manifest = generate_sync_manifest(
        root,
        experiment_id=experiment_id,
        experiment_config_sha256=experiment_config_sha256,
    )
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
        source=root,
        entries=manifest.entries,
        environ=environ,
        max_attempts=max_attempts,
    )
    with tempfile.TemporaryDirectory(prefix="lehome-model-readback-") as temporary:
        readback = Path(temporary)
        download_files(
            transport=transport,
            repository=repository,
            revision=immutable_revision,
            destination=readback,
            relative_paths=tuple(entry.relative_path for entry in manifest.entries),
            environ=environ,
            max_attempts=max_attempts,
        )
        verified_entries = _verified_remote_entries(readback, manifest.entries)

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
