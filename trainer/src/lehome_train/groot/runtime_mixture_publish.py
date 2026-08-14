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
from lehome_train.io import atomic_write_json, canonical_json_sha256, sha256_file
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


def publish_pending_mixture(*, pending_root: str | Path, revision: str, receipt_path: str | Path, transport: HubTransport) -> dict[str, object]:
    """Upload only a builder's pending bytes under its deterministic prefix."""
    root = Path(pending_root)
    pending = __import__("json").loads((root / "publication-pending.json").read_text(encoding="utf-8"))
    if not isinstance(pending, dict) or pending.get("repository") != APPROVED_MIXTURE_REPOSITORY or pending.get("publication_pending") is not True or not isinstance(pending.get("mixture_id"), str):
        raise ValueError("mixture publication pending artifact is invalid")
    prefix = f"mixtures/{pending['mixture_id']}"
    if pending.get("prefix") != prefix:
        raise ValueError("mixture publication prefix drift")
    entries = _entries(root)
    require_access(transport=transport, repository=APPROVED_MIXTURE_REPOSITORY, read=True, write=True)
    immutable = upload_files(transport=transport, repository=APPROVED_MIXTURE_REPOSITORY, revision=revision, source=root, entries=entries, remote_prefix=prefix, max_attempts=1)
    with tempfile.TemporaryDirectory(prefix="lehome-mixture-readback-") as temporary:
        readback = Path(temporary)
        download_files(transport=transport, repository=APPROVED_MIXTURE_REPOSITORY, revision=immutable, destination=readback, relative_paths=tuple(item.relative_path for item in entries), remote_prefix=prefix, max_attempts=1)
        if _entries(readback) != entries:
            raise ValueError("mixture publication readback hash or size mismatch")
    receipt = {"repository": APPROVED_MIXTURE_REPOSITORY, "immutable_revision": immutable, "remote_prefix": prefix, "mixture_id": pending["mixture_id"], "fresh_readback_verified": True, "tree_listing_verified": True}
    target = Path(receipt_path)
    if target.exists() or target.is_symlink():
        raise FileExistsError("mixture publication receipt destination is immutable")
    atomic_write_json(target, receipt)
    return receipt


def finalize_pending_mixture(*, pending_root: str | Path, publication_receipt: str | Path, destination: str | Path, source_mounts: dict[str, str]) -> Path:
    """Turn verified pending bytes into the schema consumed by RuntimeMixtureDataset."""
    root, output = Path(pending_root), Path(destination)
    pending = __import__("json").loads((root / "publication-pending.json").read_text(encoding="utf-8"))
    receipt = __import__("json").loads(Path(publication_receipt).read_text(encoding="utf-8"))
    if output.exists() or pending.get("mixture_id") != receipt.get("mixture_id") or receipt.get("repository") != APPROVED_MIXTURE_REPOSITORY or receipt.get("remote_prefix") != pending.get("prefix") or receipt.get("fresh_readback_verified") is not True or receipt.get("tree_listing_verified") is not True:
        raise ValueError("mixture finalization receipt or destination is invalid")
    windows = __import__("json").loads((root / "windows.json").read_text(encoding="utf-8"))["windows"]
    output.mkdir(parents=True)
    shutil.copy2(root / "mixture-normalization.json", output / "mixture-normalization.json")
    manifest = {"schema_version": 2, "kind": "lehome_runtime_mixture", "repository": APPROVED_MIXTURE_REPOSITORY, "revision": receipt["immutable_revision"], "safe_prefix": pending["prefix"], "mixture_id": pending["mixture_id"], "sources": pending["sources"], "camera_schema": ["observation.images.top_rgb", "observation.images.left_rgb", "observation.images.right_rgb"], "image_shape": [480, 640, 3], "state_schema": {"dimension": 12, "storage": "absolute"}, "action_schema": {"dimension": 12, "storage": "absolute"}, "fps": 30, "action_horizon": 16, "instruction": "fold the garment on the table", "schedule_seed": 17, "cycle_size": 10, "mixture_normalization": {"path": "mixture-normalization.json", "sha256": sha256_file(output / "mixture-normalization.json"), "byte_size": (output / "mixture-normalization.json").stat().st_size}, "window_index": {"path": "windows.json", "sha256": "", "byte_size": 0}}
    index = {"schema_version": 2, "manifest_sha256": canonical_json_sha256(manifest), "windows": windows}
    atomic_write_json(output / "windows.json", index)
    manifest["window_index"] = {"path": "windows.json", "sha256": sha256_file(output / "windows.json"), "byte_size": (output / "windows.json").stat().st_size}
    atomic_write_json(output / "mixture.json", manifest)
    atomic_write_json(output / "mounts.json", {"schema_version": 2, "repository": manifest["repository"], "revision": manifest["revision"], "safe_prefix": manifest["safe_prefix"], "release_receipt_path": str(Path(publication_receipt).resolve()), "release_receipt_sha256": sha256_file(Path(publication_receipt)), "mounts": [{"source_id": source["source_id"], "root": source_mounts[source["source_id"]], "source_tree_sha256": source["source_tree_sha256"], "artifact_receipt_sha256": source["artifact_receipt_sha256"]} for source in pending["sources"]]})
    return output
