"""Immutable experiment snapshot hydration for fresh training machines."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

from lehome_train.commands.sync import SyncResult
from lehome_train.constants import DEFAULT_MODEL_REPO
from lehome_train.hub import HubTransport, download_files, require_access
from lehome_train.io import sha256_file


def _required_bytes(sync_result: SyncResult) -> int:
    payload = sum(entry.byte_size for entry in sync_result.manifest.entries)
    return payload + max(64 * 1024**2, min(1024**3, payload // 20))


def _verify_tree(root: Path, sync_result: SyncResult) -> None:
    expected = {
        entry.relative_path: (entry.sha256, entry.byte_size)
        for entry in sync_result.manifest.entries
    }
    observed: set[str] = set()
    pending = [root]
    while pending:
        directory = pending.pop()
        for entry in os.scandir(directory):
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            if entry.is_symlink():
                raise ValueError("restored snapshot verification found a symlink")
            if entry.is_dir(follow_symlinks=False):
                pending.append(path)
            elif entry.is_file(follow_symlinks=False):
                observed.add(relative)
            else:
                raise ValueError("restored snapshot verification found a special path")
    if observed != set(expected):
        raise ValueError("restored snapshot verification found an incompatible tree")
    for relative, (digest, byte_size) in expected.items():
        path = root / relative
        if path.stat().st_size != byte_size or sha256_file(path) != digest:
            raise ValueError("restored snapshot verification found changed bytes")


def restore_experiment_snapshot(
    destination_path: str | os.PathLike[str],
    *,
    sync_result: SyncResult,
    transport: HubTransport,
    staging_root: str | os.PathLike[str],
    environ: Mapping[str, str] | None = None,
    max_attempts: int = 3,
) -> Path:
    """Download one exact synced tree and atomically expose it after verification."""

    if not isinstance(sync_result, SyncResult):
        raise TypeError("restore requires a strict sync result")
    if (
        sync_result.repository != DEFAULT_MODEL_REPO
        or not sync_result.disposable
        or not sync_result.manifest.entries
        or not all(entry.remotely_verified for entry in sync_result.manifest.entries)
    ):
        raise ValueError("restore requires a fully verified approved sync result")
    destination = Path(destination_path)
    if destination.exists():
        raise FileExistsError("refusing to overwrite a restore destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(staging_root)
    if not staging.is_dir() or staging.is_symlink():
        raise ValueError("restore staging root must be an existing regular directory")
    if os.stat(staging).st_dev != os.stat(destination.parent).st_dev:
        raise ValueError("restore staging and destination must share one filesystem")
    if shutil.disk_usage(staging).free < _required_bytes(sync_result):
        raise ValueError("restore staging filesystem has insufficient capacity")
    require_access(
        transport=transport,
        repository=sync_result.repository,
        read=True,
        write=False,
        environ=environ,
    )
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            suffix=".incomplete",
            dir=staging,
        )
    )
    try:
        relative_paths = tuple(
            entry.relative_path for entry in sync_result.manifest.entries
        )
        download_files(
            transport=transport,
            repository=sync_result.repository,
            revision=sync_result.immutable_revision,
            destination=temporary,
            relative_paths=relative_paths,
            remote_prefix=sync_result.manifest.remote_prefix,
            environ=environ,
            max_attempts=max_attempts,
        )
        _verify_tree(temporary, sync_result)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination
