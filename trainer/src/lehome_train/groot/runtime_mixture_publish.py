"""Injected, private-only publication for runtime-mixture source and final bytes.

All network activity is behind ``HubTransport``; callers supply the process
``HF_TOKEN`` and tests supply an in-memory transport.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from lehome_train.groot.runtime_mixture import (
    ACTION_HORIZON,
    APPROVED_MIXTURE_REPOSITORY,
    CAMERAS,
    FPS,
    INSTRUCTION,
    load_runtime_contract,
    pending_mixture_id,
)
from lehome_train.hub import HubTransport, require_access, upload_files, download_files, list_repository_tree
from lehome_train.io import atomic_write_json, canonical_json_sha256, sha256_file
from lehome_train.models import SyncEntry
from lehome_train.redaction import generate_upload_allowlist


def _entries(root: Path) -> tuple[SyncEntry, ...]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("publication root is unavailable")
    paths: list[str] = []
    pending = [root]
    while pending:
        current = pending.pop()
        with os.scandir(current) as children:
            for child in children:
                relative = Path(child.path).relative_to(root).as_posix()
                if child.is_symlink():
                    raise ValueError("publication tree contains a symlink")
                if child.is_dir(follow_symlinks=False):
                    pending.append(Path(child.path))
                elif child.is_file(follow_symlinks=False):
                    paths.append(relative)
                else:
                    raise ValueError("publication tree contains an unsupported path type")
    if not paths:
        raise ValueError("publication tree is empty")
    return generate_upload_allowlist(root, tuple(sorted(paths)))


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def _preflight_receipt_destination(
    *, root: Path, receipt_path: str | Path, label: str, disallow: tuple[Path, ...] = ()
) -> Path:
    target = Path(receipt_path)
    if not target.is_absolute() or target.exists() or target.is_symlink():
        raise FileExistsError(f"{label} receipt destination must be an absent absolute path")
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise ValueError(f"{label} receipt destination parent is unavailable")
    if _within(target, root) or any(target.resolve(strict=False) == item.resolve(strict=False) for item in disallow):
        raise ValueError(f"{label} receipt destination must be external and non-overlapping")
    return target


def _load_exact(path: Path, *, keys: set[str], label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is malformed") from error
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} has an incompatible schema")
    return value


def _tree_matches(tree: tuple[object, ...], *, prefix: str, entries: tuple[SyncEntry, ...]) -> bool:
    expected = {f"{prefix}/{entry.relative_path}" for entry in entries}
    directories = {
        f"{prefix}/" + "/".join(parts[:index])
        for entry in entries
        for parts in (entry.relative_path.split("/"),)
        for index in range(1, len(parts))
    }
    observed: set[str] = set()
    for entry in tree:
        path, entry_type = getattr(entry, "relative_path", None), getattr(entry, "entry_type", None)
        if not isinstance(path, str) or not path.startswith(prefix + "/"):
            continue
        if entry_type == "file" and path in expected:
            observed.add(path)
        elif entry_type == "directory" and path in directories:
            continue
        else:
            return False
    return observed == expected


_PENDING_KEYS = {"schema_version", "kind", "repository", "mixture_id", "prefix", "sources", "normalization_sha256", "windows_sha256", "publication_pending"}


def _pending(root: Path) -> dict[str, object]:
    pending = _load_exact(root / "publication-pending.json", keys=_PENDING_KEYS, label="mixture publication pending artifact")
    mixture_id = pending.get("mixture_id")
    if (
        pending.get("schema_version") != 1
        or pending.get("kind") != "runtime_mixture_publication_pending"
        or pending.get("repository") != APPROVED_MIXTURE_REPOSITORY
        or pending.get("publication_pending") is not True
        or not isinstance(mixture_id, str)
        or pending_mixture_id(pending) != mixture_id
        or pending.get("prefix") != f"mixtures/{mixture_id}"
    ):
        raise ValueError("mixture publication pending artifact content address is invalid")
    return pending


def publish_source(*, root: str | Path, source_type: str, round_id: str | None, revision: str, receipt_path: str | Path, transport: HubTransport) -> dict[str, object]:
    """Publish one complete mounted source tree and verify a fresh readback."""
    if source_type == "bc" and round_id is None:
        prefix = "bc/full"
    elif source_type == "rollout" and isinstance(round_id, str) and re.fullmatch(r"[1-9][0-9]*", round_id):
        prefix = f"rollouts/round-{round_id}"
    else:
        raise ValueError("source publication type or round ID is invalid")
    if not isinstance(revision, str) or not revision:
        raise ValueError("source publication revision target is required")
    local = Path(root)
    target = _preflight_receipt_destination(root=local, receipt_path=receipt_path, label="source publication")
    entries = _entries(local)
    require_access(transport=transport, repository=APPROVED_MIXTURE_REPOSITORY, read=True, write=True)
    revision = upload_files(transport=transport, repository=APPROVED_MIXTURE_REPOSITORY, revision=revision, source=local, entries=entries, remote_prefix=prefix, max_attempts=1)
    tree = list_repository_tree(transport=transport, repository=APPROVED_MIXTURE_REPOSITORY, revision=revision, max_attempts=1)
    if not _tree_matches(tree, prefix=prefix, entries=entries):
        raise ValueError("source publication remote tree differs from the complete local source")
    with tempfile.TemporaryDirectory(prefix="lehome-runtime-readback-") as temporary:
        readback = Path(temporary)
        download_files(transport=transport, repository=APPROVED_MIXTURE_REPOSITORY, revision=revision, destination=readback, relative_paths=tuple(item.relative_path for item in entries), remote_prefix=prefix, max_attempts=1)
        if _entries(readback) != entries:
            raise ValueError("source publication readback hash or size mismatch")
    receipt = {"repository": APPROVED_MIXTURE_REPOSITORY, "immutable_revision": revision, "remote_prefix": prefix, "fresh_readback_verified": True, "tree_listing_verified": True}
    atomic_write_json(target, receipt)
    return receipt


def publish_pending_mixture(*, pending_root: str | Path, revision: str, receipt_path: str | Path, transport: HubTransport) -> dict[str, object]:
    """Upload only a builder's pending bytes under its deterministic prefix."""
    root = Path(pending_root)
    target = _preflight_receipt_destination(root=root, receipt_path=receipt_path, label="mixture publication")
    pending = _pending(root)
    prefix = f"mixtures/{pending['mixture_id']}"
    if pending.get("prefix") != prefix:
        raise ValueError("mixture publication prefix drift")
    entries = _entries(root)
    require_access(transport=transport, repository=APPROVED_MIXTURE_REPOSITORY, read=True, write=True)
    immutable = upload_files(transport=transport, repository=APPROVED_MIXTURE_REPOSITORY, revision=revision, source=root, entries=entries, remote_prefix=prefix, max_attempts=1)
    tree = list_repository_tree(transport=transport, repository=APPROVED_MIXTURE_REPOSITORY, revision=immutable, max_attempts=1)
    if not _tree_matches(tree, prefix=prefix, entries=entries):
        raise ValueError("mixture publication remote tree differs from the complete pending artifact")
    with tempfile.TemporaryDirectory(prefix="lehome-mixture-readback-") as temporary:
        readback = Path(temporary)
        download_files(transport=transport, repository=APPROVED_MIXTURE_REPOSITORY, revision=immutable, destination=readback, relative_paths=tuple(item.relative_path for item in entries), remote_prefix=prefix, max_attempts=1)
        if _entries(readback) != entries:
            raise ValueError("mixture publication readback hash or size mismatch")
    receipt = {"repository": APPROVED_MIXTURE_REPOSITORY, "immutable_revision": immutable, "remote_prefix": prefix, "mixture_id": pending["mixture_id"], "fresh_readback_verified": True, "tree_listing_verified": True}
    atomic_write_json(target, receipt)
    return receipt


def _entry_records(entries: tuple[SyncEntry, ...]) -> list[dict[str, object]]:
    return [
        {"relative_path": entry.relative_path, "sha256": entry.sha256, "byte_size": entry.byte_size}
        for entry in entries
    ]


def finalize_pending_mixture(
    *,
    pending_root: str | Path,
    publication_receipt: str | Path,
    destination: str | Path,
    deployment_receipt_path: str | Path,
    source_mounts: dict[str, str],
    revision: str,
    transport: HubTransport,
) -> Path:
    """Publish and materialize final schema-2 bytes without a revision hash cycle.

    The builder-pending revision remains immutable audit evidence.  This stage
    copies those bytes, replaces only the runtime index with the schema-2
    index, publishes the complete final tree, and records its actual revision
    externally.  ``mixture.json`` deliberately omits that revision: the
    deployment receipt binds it to exact final bytes after readback.
    """
    root, output = Path(pending_root), Path(destination)
    pending = _pending(root)
    release = Path(publication_receipt)
    receipt = _load_exact(release, keys={"repository", "immutable_revision", "remote_prefix", "mixture_id", "fresh_readback_verified", "tree_listing_verified"}, label="mixture publication receipt")
    if (
        output.exists()
        or pending.get("schema_version") != 1
        or pending.get("kind") != "runtime_mixture_publication_pending"
        or pending.get("repository") != APPROVED_MIXTURE_REPOSITORY
        or pending.get("publication_pending") is not True
        or not isinstance(pending.get("mixture_id"), str)
        or pending.get("mixture_id") != receipt.get("mixture_id")
        or receipt.get("repository") != APPROVED_MIXTURE_REPOSITORY
        or receipt.get("remote_prefix") != pending.get("prefix")
        or type(receipt.get("immutable_revision")) is not str
        or re.fullmatch(r"[0-9a-f]{40}", str(receipt.get("immutable_revision"))) is None
        or receipt.get("fresh_readback_verified") is not True
        or receipt.get("tree_listing_verified") is not True
    ):
        raise ValueError("mixture finalization receipt or destination is invalid")
    normalization = root / "mixture-normalization.json"
    windows_file = root / "windows.json"
    if sha256_file(normalization) != pending["normalization_sha256"] or sha256_file(windows_file) != pending["windows_sha256"]:
        raise ValueError("mixture finalization pending bytes drift")
    windows_value = _load_exact(windows_file, keys={"schema_version", "windows"}, label="pending window index")
    if windows_value["schema_version"] != 3 or not isinstance(windows_value["windows"], list):
        raise ValueError("pending window index is invalid")
    windows = windows_value["windows"]
    if (
        not isinstance(pending["sources"], list)
        or set(source_mounts) != {item.get("source_id") for item in pending["sources"] if isinstance(item, dict)}
        or not all(type(root) is str and Path(root).is_absolute() for root in source_mounts.values())
        or not isinstance(revision, str)
        or not revision
    ):
        raise ValueError("mixture finalization source mounts are incomplete")
    deployment = _preflight_receipt_destination(root=root, receipt_path=deployment_receipt_path, label="runtime deployment", disallow=(release,))
    if output.exists() or _within(deployment, output):
        raise FileExistsError("runtime finalization destinations are immutable")
    staging = output.parent / f".{output.name}.{pending['mixture_id'][:12]}.tmp"
    if staging.exists():
        raise FileExistsError("runtime finalization staging already exists")
    try:
        shutil.copytree(root, staging)
        shutil.copy2(normalization, staging / "mixture-normalization.json")
        manifest = {"schema_version": 2, "kind": "lehome_runtime_mixture", "repository": APPROVED_MIXTURE_REPOSITORY, "safe_prefix": pending["prefix"], "mixture_id": pending["mixture_id"], "sources": pending["sources"], "camera_schema": list(CAMERAS), "image_shape": [480, 640, 3], "state_schema": {"dimension": 12, "storage": "absolute"}, "action_schema": {"dimension": 12, "storage": "absolute"}, "fps": FPS, "action_horizon": ACTION_HORIZON, "instruction": INSTRUCTION, "schedule_seed": 17, "cycle_size": 10, "mixture_normalization": {"path": "mixture-normalization.json", "sha256": sha256_file(staging / "mixture-normalization.json"), "byte_size": (staging / "mixture-normalization.json").stat().st_size}, "window_index": {"path": "windows.json", "sha256": "", "byte_size": 0}}
        index = {"schema_version": 2, "manifest_sha256": canonical_json_sha256(manifest), "windows": windows}
        atomic_write_json(staging / "windows.json", index)
        manifest["window_index"] = {"path": "windows.json", "sha256": sha256_file(staging / "windows.json"), "byte_size": (staging / "windows.json").stat().st_size}
        atomic_write_json(staging / "mixture.json", manifest)
        entries = _entries(staging)
        require_access(transport=transport, repository=APPROVED_MIXTURE_REPOSITORY, read=True, write=True)
        immutable = upload_files(transport=transport, repository=APPROVED_MIXTURE_REPOSITORY, revision=revision, source=staging, entries=entries, remote_prefix=str(pending["prefix"]), max_attempts=1)
        tree = list_repository_tree(transport=transport, repository=APPROVED_MIXTURE_REPOSITORY, revision=immutable, max_attempts=1)
        if not _tree_matches(tree, prefix=str(pending["prefix"]), entries=entries):
            raise ValueError("runtime deployment remote tree differs from finalized bytes")
        with tempfile.TemporaryDirectory(prefix="lehome-runtime-final-readback-") as temporary:
            readback = Path(temporary)
            download_files(transport=transport, repository=APPROVED_MIXTURE_REPOSITORY, revision=immutable, destination=readback, relative_paths=tuple(entry.relative_path for entry in entries), remote_prefix=str(pending["prefix"]), max_attempts=1)
            if _entries(readback) != entries:
                raise ValueError("runtime deployment readback hash or size mismatch")
        deployment_value = {"repository": APPROVED_MIXTURE_REPOSITORY, "immutable_revision": immutable, "remote_prefix": pending["prefix"], "mixture_id": pending["mixture_id"], "pending_receipt_sha256": sha256_file(release), "artifact_entries": _entry_records(entries), "fresh_readback_verified": True, "tree_listing_verified": True}
        atomic_write_json(deployment, deployment_value)
        atomic_write_json(staging / "mounts.json", {"schema_version": 2, "repository": manifest["repository"], "safe_prefix": manifest["safe_prefix"], "deployment_receipt_path": str(deployment.resolve()), "deployment_receipt_sha256": sha256_file(deployment), "mounts": [{"source_id": source["source_id"], "root": source_mounts[source["source_id"]], "source_tree_sha256": source["source_tree_sha256"], "artifact_receipt_sha256": source["artifact_receipt_sha256"]} for source in pending["sources"]]})
        load_runtime_contract(staging / "mixture.json", staging / "mounts.json")
        os.replace(staging, output)
        return output
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def publish_source_from_request(path: str | Path, *, transport: HubTransport) -> dict[str, object]:
    """Run one source publication from the exact offline-reviewable envelope."""

    request = _load_exact(Path(path), keys={"schema_version", "command", "arguments"}, label="runtime source publication request")
    arguments = request["arguments"]
    expected = {"root", "source_type", "round_id", "revision", "receipt_path"}
    if (
        request["schema_version"] != 1
        or request["command"] != "publish-runtime-source"
        or not isinstance(arguments, dict)
        or set(arguments) != expected
        or any(type(arguments[key]) is not str or not arguments[key] for key in expected - {"round_id"})
        or (arguments["round_id"] is not None and (type(arguments["round_id"]) is not str or not arguments["round_id"]))
    ):
        raise ValueError("runtime source publication request has an incompatible schema")
    return publish_source(**arguments, transport=transport)


def publish_pending_mixture_from_request(path: str | Path, *, transport: HubTransport) -> dict[str, object]:
    """Run pending publication from the exact offline-reviewable envelope."""

    request = _load_exact(Path(path), keys={"schema_version", "command", "arguments"}, label="runtime mixture publication request")
    arguments = request["arguments"]
    expected = {"pending_root", "revision", "receipt_path"}
    if (
        request["schema_version"] != 1
        or request["command"] != "publish-runtime-mixture"
        or not isinstance(arguments, dict)
        or set(arguments) != expected
        or any(type(arguments[key]) is not str or not arguments[key] for key in expected)
    ):
        raise ValueError("runtime mixture publication request has an incompatible schema")
    return publish_pending_mixture(**arguments, transport=transport)


def finalize_pending_mixture_from_request(path: str | Path, *, transport: HubTransport) -> Path:
    """Run final deployment from the exact offline-reviewable envelope."""

    request = _load_exact(Path(path), keys={"schema_version", "command", "arguments"}, label="runtime mixture finalization request")
    arguments = request["arguments"]
    expected = {"pending_root", "publication_receipt", "destination", "deployment_receipt_path", "source_mounts", "revision"}
    if (
        request["schema_version"] != 1
        or request["command"] != "finalize-runtime-mixture"
        or not isinstance(arguments, dict)
        or set(arguments) != expected
        or any(type(arguments[key]) is not str or not arguments[key] for key in expected - {"source_mounts"})
        or not isinstance(arguments["source_mounts"], dict)
        or any(type(key) is not str or type(value) is not str for key, value in arguments["source_mounts"].items())
    ):
        raise ValueError("runtime mixture finalization request has an incompatible schema")
    return finalize_pending_mixture(**arguments, transport=transport)
