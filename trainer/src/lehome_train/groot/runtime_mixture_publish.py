"""Injected, private-only publication for runtime-mixture source and final bytes.

All network activity is behind ``HubTransport``; callers supply the process
``HF_TOKEN`` and tests supply an in-memory transport.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from lehome_train.groot.runtime_mixture import APPROVED_MIXTURE_REPOSITORY, source_tree_sha256
from lehome_train.hub import HubTransport, require_access, upload_files, download_files, list_repository_tree
from lehome_train.io import atomic_write_json, sha256_file
from lehome_train.models import SyncEntry


def _entries(root: Path) -> tuple[SyncEntry, ...]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("publication root is unavailable")
    result: list[SyncEntry] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("publication tree contains a symlink")
        if path.is_file():
            result.append(SyncEntry(path.relative_to(root).as_posix(), sha256_file(path), path.stat().st_size))
    if not result:
        raise ValueError("publication tree is empty")
    return tuple(result)


def publish_source(*, root: str | Path, source_type: str, round_id: str | None, revision: str, receipt_path: str | Path, transport: HubTransport) -> dict[str, object]:
    """Publish one complete mounted source tree and verify a fresh readback."""
    if source_type == "bc":
        prefix = "bc/full"
    elif source_type == "rollout" and isinstance(round_id, str) and round_id and "/" not in round_id:
        prefix = f"rollouts/round-{round_id}"
    else:
        raise ValueError("source publication type or round ID is invalid")
    if not isinstance(revision, str) or not revision:
        raise ValueError("source publication revision target is required")
    local = Path(root)
    entries = _entries(local)
    require_access(transport=transport, repository=APPROVED_MIXTURE_REPOSITORY, read=True, write=True)
    revision = upload_files(transport=transport, repository=APPROVED_MIXTURE_REPOSITORY, revision=revision, source=local, entries=entries, remote_prefix=prefix, max_attempts=1)
    tree = list_repository_tree(transport=transport, repository=APPROVED_MIXTURE_REPOSITORY, revision=revision, max_attempts=1)
    expected = {f"{prefix}/{entry.relative_path}" for entry in entries}
    observed = {entry.relative_path for entry in tree if entry.entry_type == "file" and entry.relative_path.startswith(prefix + "/")}
    if observed != expected:
        raise ValueError("source publication remote tree differs from the complete local source")
    with tempfile.TemporaryDirectory(prefix="lehome-runtime-readback-") as temporary:
        readback = Path(temporary)
        download_files(transport=transport, repository=APPROVED_MIXTURE_REPOSITORY, revision=revision, destination=readback, relative_paths=tuple(item.relative_path for item in entries), remote_prefix=prefix, max_attempts=1)
        if _entries(readback) != entries:
            raise ValueError("source publication readback hash or size mismatch")
    receipt = {"repository": APPROVED_MIXTURE_REPOSITORY, "immutable_revision": revision, "remote_prefix": prefix, "source_tree_sha256": source_tree_sha256(local), "fresh_readback_verified": True, "tree_listing_verified": True}
    target = Path(receipt_path)
    if target.exists() or target.is_symlink():
        raise FileExistsError("source publication receipt destination is immutable")
    atomic_write_json(target, receipt)
    return receipt
