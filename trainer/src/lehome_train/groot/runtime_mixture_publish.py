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

from lehome_train.constants import DEFAULT_DATA_REPO, DEFAULT_ROLLOUT_REPO
from lehome_train.groot.runtime_mixture import (
    ACTION_HORIZON,
    APPROVED_MIXTURE_REPOSITORY,
    CAMERAS,
    FPS,
    INSTRUCTION,
    load_runtime_contract,
    pending_mixture_id,
    source_tree_sha256,
)
from lehome_train.groot.experiment_manifest import batch64_quotas
from lehome_train.hub import (
    HubTransport,
    download_files,
    list_repository_tree,
    require_access,
    resolve_approved_ref,
    upload_files,
    upload_large_folder,
)
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


def _stage_large_source(
    *, root: Path, staging_root: str | Path, prefix: str, entries: tuple[SyncEntry, ...],
    disallow: tuple[Path, ...],
) -> tuple[Path, tuple[SyncEntry, ...]]:
    """Materialize only approved source files below the target remote prefix."""

    staging = Path(staging_root)
    if (
        not staging.is_absolute()
        or staging.is_symlink()
        or (staging.exists() and not staging.is_dir())
        or staging.parent.is_symlink()
        or not staging.parent.is_dir()
        or _within(staging, root)
        or _within(root, staging)
        or any(_within(item, staging) or _within(staging, item) for item in disallow)
    ):
        raise ValueError("large source upload staging root must be an external safe directory")
    if not staging.exists():
        staging.mkdir()

    staged = staging
    for component in prefix.split("/"):
        staged /= component
        if staged.exists():
            if staged.is_symlink() or not staged.is_dir():
                raise ValueError("large source upload staging hierarchy is unsafe")
        else:
            staged.mkdir()

    staged_entries = tuple(
        SyncEntry(f"{prefix}/{entry.relative_path}", entry.sha256, entry.byte_size)
        for entry in entries
    )
    if any(staged.iterdir()):
        if _entries(staged) != entries:
            raise ValueError("large source upload staging tree differs from the exact allowlist")
        return staging, staged_entries

    for entry in entries:
        source = root / entry.relative_path
        target = staged / entry.relative_path
        if source.is_symlink() or not source.is_file():
            raise ValueError("large source upload allowlist file is unavailable")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target, follow_symlinks=False)
        except OSError:
            shutil.copyfile(source, target, follow_symlinks=False)
        if target.is_symlink() or not target.is_file() or SyncEntry(
            entry.relative_path, sha256_file(target), target.stat(follow_symlinks=False).st_size,
        ) != entry:
            raise ValueError("large source upload staging copy differs from the exact allowlist")
    if _entries(staged) != entries:
        raise ValueError("large source upload staging tree differs from the exact allowlist")
    return staging, staged_entries


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
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_strict_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is malformed") from error
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} has an incompatible schema")
    return value


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON field")
        value[key] = item
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
_BOUND_PENDING_KEYS = _PENDING_KEYS | {"experiment_manifest_sha256", "mixture_weights", "source_quotas"}
_SOURCE_READBACK_KEYS = {
    "repository", "immutable_revision", "remote_prefix", "fresh_readback_verified",
    "tree_listing_verified",
}
_SOURCE_UPLOAD_JOURNAL_KEYS = {
    "schema_version", "kind", "repository", "immutable_revision", "remote_prefix",
    "source_type", "round_id", "artifact_entries", "tree_listing_verified",
    "readback_pending",
}
_SOURCE_READBACK_BATCH_SIZE = 800
_LARGE_SOURCE_UPLOAD_WORKERS = 4
_DEPLOYMENT_KEYS = {
    "repository", "immutable_revision", "remote_prefix", "mixture_id",
    "pending_receipt_sha256", "artifact_entries", "fresh_readback_verified",
    "tree_listing_verified",
}
_BOUND_DEPLOYMENT_KEYS = _DEPLOYMENT_KEYS | {"experiment_manifest_sha256", "mixture_weights", "source_quotas"}


def _pending(root: Path) -> dict[str, object]:
    path = root / "publication-pending.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("mixture publication pending artifact is missing or unsafe")
    try:
        pending = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_strict_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("mixture publication pending artifact is malformed") from error
    if not isinstance(pending, dict):
        raise ValueError("mixture publication pending artifact has an incompatible schema")
    keys = _BOUND_PENDING_KEYS if type(pending.get("schema_version")) is int and pending.get("schema_version") == 2 else _PENDING_KEYS
    if set(pending) != keys:
        raise ValueError("mixture publication pending artifact has an incompatible schema")
    mixture_id = pending.get("mixture_id")
    if (
        type(pending.get("schema_version")) is not int
        or pending.get("schema_version") not in {1, 2}
        or pending.get("kind") != "runtime_mixture_publication_pending"
        or pending.get("repository") != APPROVED_MIXTURE_REPOSITORY
        or pending.get("publication_pending") is not True
        or not isinstance(mixture_id, str)
        or pending_mixture_id(pending) != mixture_id
        or pending.get("prefix") != f"mixtures/{mixture_id}"
    ):
        raise ValueError("mixture publication pending artifact content address is invalid")
    if pending["schema_version"] == 2:
        weights, quotas = pending["mixture_weights"], pending["source_quotas"]
        if (
            not isinstance(pending["experiment_manifest_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", pending["experiment_manifest_sha256"]) is None
            or not isinstance(weights, dict) or not isinstance(quotas, dict)
            or set(weights) != {"bc", "rollout", "dagger"} or set(quotas) != {"bc", "rollout", "dagger"}
            or any(type(weights[kind]) is not int or type(quotas[kind]) is not int for kind in ("bc", "rollout", "dagger"))
            or weights["bc"] <= 0 or weights["rollout"] <= 0 or weights["dagger"] != 0 or weights["bc"] + weights["rollout"] != 100
        ):
            raise ValueError("manifest-bound pending mixture weights are invalid")
        expected = batch64_quotas(weights)
        if not isinstance(pending["sources"], list) or any(
            not isinstance(item, dict) or type(item.get("quota")) is not int
            for item in pending["sources"]
        ):
            raise ValueError("manifest-bound pending source quotas are invalid")
        actual = {kind: sum(item["quota"] for item in pending["sources"] if item.get("source_type") == kind) for kind in expected}
        if quotas != expected or actual != expected:
            raise ValueError("manifest-bound pending mixture quotas drift")
    return pending


def _immutable_revision(value: object, *, label: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError(f"{label} must name an immutable revision")
    return value


def _hydration_entries(value: object, *, label: str) -> tuple[SyncEntry, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must contain an exact artifact tree")
    entries: list[SyncEntry] = []
    for record in value:
        if not isinstance(record, dict) or set(record) != {"relative_path", "sha256", "byte_size"}:
            raise ValueError(f"{label} artifact is malformed")
        try:
            entries.append(SyncEntry(record["relative_path"], record["sha256"], record["byte_size"]))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} artifact is malformed") from error
    if len({entry.relative_path for entry in entries}) != len(entries):
        raise ValueError(f"{label} has duplicate artifact paths")
    return tuple(sorted(entries, key=lambda entry: entry.relative_path))


def _source_upload_journal(path: Path) -> tuple[dict[str, object], tuple[SyncEntry, ...]]:
    journal = _load_exact(path, keys=_SOURCE_UPLOAD_JOURNAL_KEYS, label="runtime source upload journal")
    entries = _hydration_entries(journal.get("artifact_entries"), label="runtime source upload journal")
    prefix = journal.get("remote_prefix")
    source_type = journal.get("source_type")
    round_id = journal.get("round_id")
    expected_prefix = (
        "bc/full" if source_type == "bc" and round_id is None
        else f"rollouts/round-{round_id}" if source_type == "rollout" and type(round_id) is str and re.fullmatch(r"[1-9][0-9]*", round_id)
        else None
    )
    if (
        journal.get("schema_version") != 1
        or journal.get("kind") != "runtime_source_upload_journal"
        or journal.get("repository") != _source_repository(str(source_type))
        or journal.get("tree_listing_verified") is not True
        or journal.get("readback_pending") is not True
        or prefix != expected_prefix
    ):
        raise ValueError("runtime source upload journal is not an exact immutable source binding")
    _immutable_revision(journal.get("immutable_revision"), label="runtime source upload journal")
    return journal, entries


def _preflight_readback_root(
    *, root: Path, readback_root: str | Path, journal: Path, receipt: Path,
) -> Path:
    target = Path(readback_root)
    if not target.is_absolute() or target.is_symlink() or target.parent.is_symlink() or not target.parent.is_dir():
        raise ValueError("runtime source readback root must be an absolute safe path")
    if target.exists() and not target.is_dir():
        raise ValueError("runtime source readback root must be a directory when present")
    if (
        _within(target, root) or _within(journal, target) or _within(receipt, target)
        or target.resolve(strict=False) == journal.resolve(strict=False)
        or target.resolve(strict=False) == receipt.resolve(strict=False)
    ):
        raise ValueError("runtime source readback root must be external and non-overlapping")
    return target


def _verified_readback_entries(*, root: Path, entries: tuple[SyncEntry, ...]) -> tuple[SyncEntry, ...]:
    """Reject an unsafe or unexpected stable readback; retain exact bytes only."""

    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ValueError("runtime source readback root is unavailable or unsafe")
    if not root.exists():
        return ()
    expected = {entry.relative_path: entry for entry in entries}
    expected_directories = {
        "/".join(entry.relative_path.split("/")[:depth])
        for entry in entries
        for depth in range(1, len(entry.relative_path.split("/")))
    }
    present: list[SyncEntry] = []
    pending = [root]
    while pending:
        current = pending.pop()
        with os.scandir(current) as children:
            for child in children:
                relative = Path(child.path).relative_to(root).as_posix()
                if child.is_symlink():
                    raise ValueError("runtime source readback contains a symlink")
                if child.is_dir(follow_symlinks=False):
                    if relative not in expected_directories:
                        raise ValueError("runtime source readback contains an unexpected directory")
                    pending.append(Path(child.path))
                    continue
                if not child.is_file(follow_symlinks=False):
                    raise ValueError("runtime source readback contains an unsupported path type")
                expected_entry = expected.get(relative)
                if expected_entry is None:
                    raise ValueError("runtime source readback contains an unexpected file")
                actual = SyncEntry(relative, sha256_file(Path(child.path)), child.stat(follow_symlinks=False).st_size)
                if actual != expected_entry:
                    raise ValueError("runtime source readback bytes are tampered or incomplete")
                present.append(actual)
    return tuple(sorted(present, key=lambda entry: entry.relative_path))


def _source_prefix(*, source_type: str, round_id: str | None) -> str:
    if source_type == "bc" and round_id is None:
        return "bc/full"
    if source_type == "rollout" and isinstance(round_id, str) and re.fullmatch(r"[1-9][0-9]*", round_id):
        return f"rollouts/round-{round_id}"
    raise ValueError("source publication type or round ID is invalid")


def _source_repository(source_type: str) -> str:
    if source_type == "bc":
        return DEFAULT_DATA_REPO
    if source_type == "rollout":
        return DEFAULT_ROLLOUT_REPO
    raise ValueError("source publication type is invalid")


def _write_source_upload_journal(
    *, path: Path, revision: str, prefix: str, source_type: str,
    round_id: str | None, entries: tuple[SyncEntry, ...],
) -> dict[str, object]:
    journal = {
        "schema_version": 1, "kind": "runtime_source_upload_journal",
        "repository": _source_repository(source_type), "immutable_revision": revision,
        "remote_prefix": prefix, "source_type": source_type, "round_id": round_id,
        "artifact_entries": _entry_records(entries), "tree_listing_verified": True,
        "readback_pending": True,
    }
    atomic_write_json(path, journal)
    return journal


def _source_readback(path: Path, *, source_type: str) -> dict[str, object]:
    receipt = _load_exact(path, keys=_SOURCE_READBACK_KEYS, label="runtime source readback receipt")
    prefix = receipt.get("remote_prefix")
    if (
        receipt.get("repository") != _source_repository(source_type)
        or receipt.get("fresh_readback_verified") is not True
        or receipt.get("tree_listing_verified") is not True
        or (source_type == "bc" and prefix != "bc/full")
        or (source_type == "rollout" and (type(prefix) is not str or re.fullmatch(r"rollouts/round-[1-9][0-9]*", prefix) is None))
    ):
        raise ValueError("runtime source readback receipt is not an authenticated campaign source")
    _immutable_revision(receipt.get("immutable_revision"), label="runtime source readback receipt")
    return receipt


def _remote_files_under_prefix(
    *, transport: HubTransport, repository: str, revision: str, prefix: str,
) -> tuple[str, ...]:
    tree = list_repository_tree(
        transport=transport, repository=repository, revision=revision, remote_prefix=prefix,
        max_attempts=1,
    )
    base = prefix + "/"
    files: set[str] = set()
    directories: set[str] = set()
    for entry in tree:
        if not entry.relative_path.startswith(base):
            continue
        relative = entry.relative_path.removeprefix(base)
        if entry.entry_type == "file":
            files.add(relative)
        elif entry.entry_type == "directory":
            directories.add(relative)
        else:
            raise ValueError("runtime hydration remote tree contains a symlink or special entry")
    if not files or any(
        directory not in {"/".join(path.split("/")[:depth]) for path in files for depth in range(1, len(path.split("/")))}
        for directory in directories
    ):
        raise ValueError("runtime hydration remote tree is incomplete or unexpected")
    return tuple(sorted(files))


def _copy_receipt(source: Path, destination: Path, *, expected_sha256: str) -> None:
    if source.is_symlink() or not source.is_file() or sha256_file(source) != expected_sha256:
        raise ValueError("runtime source readback receipt drift")
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file() or sha256_file(destination) != expected_sha256:
            raise ValueError("runtime hydration receipt destination is immutable")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if destination.is_symlink() or sha256_file(destination) != expected_sha256:
        raise ValueError("runtime source readback receipt copy mismatch")


def _resume_hydration_tree(
    root: Path, *, expected_entries: tuple[SyncEntry, ...] | None = None,
    expected_paths: tuple[str, ...] | None = None, label: str,
) -> None:
    """Permit only authenticated completed files left by an interrupted download."""
    if (expected_entries is None) == (expected_paths is None):
        raise ValueError("runtime hydration resume contract is invalid")
    allowed = (
        {entry.relative_path: entry for entry in expected_entries}
        if expected_entries is not None else {path: None for path in expected_paths or ()}
    )
    if not root.exists():
        root.mkdir(parents=True)
        return
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{label} is not a safe resumable directory")
    pending = [root]
    while pending:
        current = pending.pop()
        with os.scandir(current) as children:
            for child in children:
                path = Path(child.path)
                relative = path.relative_to(root).as_posix()
                if child.is_symlink():
                    raise ValueError(f"{label} contains a symlink")
                if child.is_dir(follow_symlinks=False):
                    pending.append(path)
                    continue
                if not child.is_file(follow_symlinks=False) or relative not in allowed:
                    raise ValueError(f"{label} contains an unexpected entry")
                entry = allowed[relative]
                if entry is not None and (
                    sha256_file(path) != entry.sha256 or path.stat(follow_symlinks=False).st_size != entry.byte_size
                ):
                    raise ValueError(f"{label} contains a changed partial artifact")


def hydrate_runtime_mixture_from_request(path: str | Path, *, transport: HubTransport) -> dict[str, object]:
    """Hydrate only the receipt-bound immutable runtime mixture on an x86 host.

    The immutable manifest keeps authoring receipt paths.  This writes a local
    mount descriptor that replaces those *locations* while requiring identical
    receipt bytes, avoiding an immutable-manifest rewrite or hash cycle.
    """
    request = _load_exact(Path(path), keys={"schema_version", "command", "arguments"}, label="runtime mixture hydration request")
    arguments = request["arguments"]
    expected = {"deployment_receipt", "source_readback_receipts", "destination", "mounts_descriptor"}
    if (
        type(request.get("schema_version")) is not int
        or request.get("schema_version") != 1
        or request.get("command") != "hydrate-runtime-mixture"
        or not isinstance(arguments, dict)
        or set(arguments) != expected
        or any(type(arguments[key]) is not str or not arguments[key] for key in expected - {"source_readback_receipts"})
        or not isinstance(arguments.get("source_readback_receipts"), dict)
        or any(type(key) is not str or type(value) is not str or not value for key, value in arguments["source_readback_receipts"].items())
    ):
        raise ValueError("runtime mixture hydration request has an incompatible schema")
    deployment_path = Path(arguments["deployment_receipt"])
    try:
        deployment_probe = json.loads(deployment_path.read_text(encoding="utf-8"), object_pairs_hook=_strict_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("runtime deployment receipt is malformed") from error
    deployment = _load_exact(
        deployment_path,
        keys=_BOUND_DEPLOYMENT_KEYS if isinstance(deployment_probe, dict) and "experiment_manifest_sha256" in deployment_probe else _DEPLOYMENT_KEYS,
        label="runtime deployment receipt",
    )
    if (
        deployment.get("repository") != APPROVED_MIXTURE_REPOSITORY
        or deployment.get("fresh_readback_verified") is not True
        or deployment.get("tree_listing_verified") is not True
        or type(deployment.get("remote_prefix")) is not str
        or type(deployment.get("mixture_id")) is not str
        or deployment["remote_prefix"] != f"mixtures/{deployment['mixture_id']}"
    ):
        raise ValueError("runtime deployment receipt is not authenticated")
    revision = _immutable_revision(deployment.get("immutable_revision"), label="runtime deployment receipt")
    entries = _hydration_entries(deployment.get("artifact_entries"), label="runtime deployment receipt")
    destination, mounts_path = Path(arguments["destination"]), Path(arguments["mounts_descriptor"])
    if (
        not destination.is_absolute() or not mounts_path.is_absolute()
        or mounts_path.exists() or destination.is_symlink() or mounts_path.is_symlink()
        or mounts_path.parent != destination
        or deployment_path.is_symlink() or not deployment_path.is_file()
    ):
        raise ValueError("runtime hydration destinations must be safe absolute paths beside the mixture")
    require_access(transport=transport, repository=APPROVED_MIXTURE_REPOSITORY, read=True, write=False)
    prefix = str(deployment["remote_prefix"])
    if _remote_files_under_prefix(transport=transport, repository=APPROVED_MIXTURE_REPOSITORY, revision=revision, prefix=prefix) != tuple(entry.relative_path for entry in entries):
        raise ValueError("runtime deployment remote tree differs from the authenticated receipt")
    _resume_hydration_tree(destination, expected_entries=entries, label="runtime hydration destination")
    try:
        download_files(
            transport=transport, repository=APPROVED_MIXTURE_REPOSITORY, revision=revision,
            destination=destination, relative_paths=tuple(entry.relative_path for entry in entries),
            remote_prefix=prefix, max_attempts=1,
        )
        if _entries(destination) != entries:
            raise ValueError("runtime deployment hydration bytes differ from the authenticated receipt")
        legacy_manifest_keys = {
            "schema_version", "kind", "repository", "safe_prefix", "mixture_id", "sources",
            "camera_schema", "image_shape", "state_schema", "action_schema", "fps", "action_horizon",
            "instruction", "schedule_seed", "cycle_size", "mixture_normalization", "window_index",
        }
        manifest_probe = json.loads((destination / "mixture.json").read_text(encoding="utf-8"), object_pairs_hook=_strict_pairs)
        manifest = _load_exact(destination / "mixture.json", keys=legacy_manifest_keys | {"experiment_manifest_sha256", "mixture_weights", "source_quotas"} if isinstance(manifest_probe, dict) and type(manifest_probe.get("schema_version")) is int and manifest_probe.get("schema_version") == 3 else legacy_manifest_keys, label="hydrated runtime mixture manifest")
        if manifest.get("schema_version") == 3:
            if any(deployment.get(key) != manifest.get(key) for key in ("experiment_manifest_sha256", "mixture_weights", "source_quotas")):
                raise ValueError("hydrated runtime experiment binding differs from deployment receipt")
        elif set(deployment) != _DEPLOYMENT_KEYS:
            raise ValueError("legacy hydrated runtime cannot use a manifest-bound deployment receipt")
        sources = manifest.get("sources")
        receipt_paths = arguments["source_readback_receipts"]
        if not isinstance(sources, list) or not sources:
            raise ValueError("hydrated runtime mixture lacks sources")
        if {source.get("source_id") for source in sources if isinstance(source, dict)} != set(receipt_paths):
            raise ValueError("runtime source receipts do not exactly cover immutable mixture sources")
        roots: list[dict[str, object]] = []
        receipt_root = destination.parent / "receipts"
        for source in sources:
            if not isinstance(source, dict) or type(source.get("source_id")) is not str or source.get("source_type") not in {"bc", "rollout"} or not isinstance(source.get("publication"), dict):
                raise ValueError("hydrated runtime source is malformed")
            source_id = source["source_id"]
            publication = source["publication"]
            receipt_path = Path(receipt_paths[source_id])
            receipt = _source_readback(receipt_path, source_type=source["source_type"])
            if (
                publication.get("repository") != receipt["repository"]
                or publication.get("revision") != receipt["immutable_revision"]
                or publication.get("prefix") != receipt["remote_prefix"]
                or publication.get("readback_receipt_sha256") != sha256_file(receipt_path)
            ):
                raise ValueError("runtime source receipt does not bind the immutable manifest publication")
            source_root = destination.parent / "sources" / source_id
            source_repository = str(receipt["repository"])
            source_files = _remote_files_under_prefix(
                transport=transport, repository=source_repository,
                revision=str(receipt["immutable_revision"]), prefix=str(receipt["remote_prefix"]),
            )
            _resume_hydration_tree(
                source_root, expected_paths=source_files, label="runtime hydration source destination",
            )
            download_files(
                transport=transport, repository=source_repository,
                revision=str(receipt["immutable_revision"]), destination=source_root,
                relative_paths=source_files, remote_prefix=str(receipt["remote_prefix"]), max_attempts=1,
            )
            if source_tree_sha256(source_root) != source.get("source_tree_sha256"):
                raise ValueError("runtime source hydration tree differs from immutable manifest")
            local_receipt = receipt_root / f"{source_id}-readback.json"
            _copy_receipt(receipt_path, local_receipt, expected_sha256=str(publication["readback_receipt_sha256"]))
            roots.append({
                "source_id": source_id, "root": str(source_root),
                "source_tree_sha256": source["source_tree_sha256"],
                "artifact_receipt_sha256": source["artifact_receipt_sha256"],
                "source_readback_receipt_path": str(local_receipt),
                "source_readback_receipt_sha256": sha256_file(local_receipt),
            })
        atomic_write_json(mounts_path, {
            "schema_version": 2, "repository": APPROVED_MIXTURE_REPOSITORY,
            "safe_prefix": prefix, "deployment_receipt_path": str(deployment_path),
            "deployment_receipt_sha256": sha256_file(deployment_path), "mounts": roots,
        })
        load_runtime_contract(destination / "mixture.json", mounts_path)
        result = {
            "schema_version": 1, "kind": "runtime_mixture_hydration",
            "repository": APPROVED_MIXTURE_REPOSITORY, "immutable_revision": revision,
            "remote_prefix": prefix, "mixture_id": deployment["mixture_id"],
            "artifact_tree_sha256": canonical_json_sha256(_entry_records(entries)),
            "mounts_descriptor": str(mounts_path), "fresh_readback_verified": True,
        }
        if manifest.get("schema_version") == 3:
            result.update({key: manifest[key] for key in ("experiment_manifest_sha256", "mixture_weights", "source_quotas")})
        return result
    except BaseException:
        # A bounded Hub retry can resume exact regular files at these stable
        # targets.  Preserve them; any drift is rejected on the next entry.
        raise


def adopt_uploaded_runtime_source(
    *, root: str | Path, source_type: str, round_id: str | None,
    immutable_revision: str, upload_journal_path: str | Path, transport: HubTransport,
) -> dict[str, object]:
    """Bind a known immutable remote source without uploading or readback."""

    local = Path(root)
    prefix = _source_prefix(source_type=source_type, round_id=round_id)
    repository = _source_repository(source_type)
    revision = _immutable_revision(immutable_revision, label="runtime source adoption revision")
    journal = _preflight_receipt_destination(
        root=local, receipt_path=upload_journal_path, label="source upload journal",
    )
    entries = _entries(local)
    require_access(transport=transport, repository=repository, read=True, write=False)
    tree = list_repository_tree(
        transport=transport, repository=repository, revision=revision,
        remote_prefix=prefix, max_attempts=1,
    )
    if not _tree_matches(tree, prefix=prefix, entries=entries):
        raise ValueError("adopted runtime source remote tree differs from the complete local source")
    return _write_source_upload_journal(
        path=journal, revision=revision, prefix=prefix, source_type=source_type,
        round_id=round_id, entries=entries,
    )


def verify_uploaded_runtime_source(
    *,
    root: str | Path,
    upload_journal_path: str | Path,
    readback_root: str | Path,
    receipt_path: str | Path,
    transport: HubTransport,
) -> dict[str, object]:
    """Strictly resume or complete immutable source readback without uploading."""

    local = Path(root)
    journal_path = Path(upload_journal_path)
    receipt = _preflight_receipt_destination(
        root=local, receipt_path=receipt_path, label="runtime source readback", disallow=(journal_path,),
    )
    readback = _preflight_readback_root(root=local, readback_root=readback_root, journal=journal_path, receipt=receipt)
    journal, expected_entries = _source_upload_journal(journal_path)
    local_entries = _entries(local)
    if local_entries != expected_entries:
        raise ValueError("runtime source local tree differs from the immutable upload journal")
    revision = _immutable_revision(journal["immutable_revision"], label="runtime source upload journal")
    prefix = str(journal["remote_prefix"])
    repository = str(journal["repository"])
    require_access(transport=transport, repository=repository, read=True, write=False)
    tree = list_repository_tree(
        transport=transport, repository=repository, revision=revision,
        remote_prefix=prefix, max_attempts=1,
    )
    if not _tree_matches(tree, prefix=prefix, entries=expected_entries):
        raise ValueError("runtime source remote tree differs from the immutable upload journal")
    readback.mkdir(exist_ok=True)
    verified = _verified_readback_entries(root=readback, entries=expected_entries)
    missing = tuple(entry.relative_path for entry in expected_entries if entry not in set(verified))
    for start in range(0, len(missing), _SOURCE_READBACK_BATCH_SIZE):
        batch = missing[start:start + _SOURCE_READBACK_BATCH_SIZE]
        download_files(
            transport=transport, repository=repository, revision=revision,
            destination=readback, relative_paths=batch, remote_prefix=prefix, max_attempts=3,
        )
        verified = _verified_readback_entries(root=readback, entries=expected_entries)
    if verified != expected_entries:
        raise ValueError("runtime source readback does not contain the complete exact tree")
    result = {
        "repository": repository, "immutable_revision": revision,
        "remote_prefix": prefix, "fresh_readback_verified": True,
        "tree_listing_verified": True,
    }
    atomic_write_json(receipt, result)
    return result


def publish_source(
    *, root: str | Path, source_type: str, round_id: str | None, revision: str,
    receipt_path: str | Path, transport: HubTransport,
    upload_journal_path: str | Path | None = None,
    readback_root: str | Path | None = None,
    large_upload: bool = False,
    large_upload_staging_root: str | Path | None = None,
) -> dict[str, object]:
    """Upload one source, persist its immutable journal, then verify fresh bytes.

    The default locations keep small legacy callers compatible while exposing a
    stable resumable path for large source trees through the explicit fields.
    """
    prefix = _source_prefix(source_type=source_type, round_id=round_id)
    repository = _source_repository(source_type)
    if not isinstance(revision, str) or not revision:
        raise ValueError("source publication revision target is required")
    local = Path(root)
    target = _preflight_receipt_destination(root=local, receipt_path=receipt_path, label="source publication")
    journal_path = Path(upload_journal_path) if upload_journal_path is not None else target.with_name(f"{target.stem}-upload-journal.json")
    journal = _preflight_receipt_destination(
        root=local, receipt_path=journal_path, label="source upload journal", disallow=(target,),
    )
    stable_readback = (
        Path(readback_root) if readback_root is not None
        else target.parent / f"{target.stem}-readback"
    )
    _preflight_readback_root(root=local, readback_root=stable_readback, journal=journal, receipt=target)
    entries = _entries(local)
    if type(large_upload) is not bool:
        raise ValueError("large source upload flag must be boolean")
    if large_upload != (large_upload_staging_root is not None):
        raise ValueError("large source upload requires one external staging root")
    require_access(transport=transport, repository=repository, read=True, write=True)
    if large_upload:
        staging, staged_entries = _stage_large_source(
            root=local, staging_root=large_upload_staging_root, prefix=prefix, entries=entries,
            disallow=(target, journal, stable_readback),
        )
        upload_large_folder(
            transport=transport, repository=repository, revision=revision,
            source=staging, entries=staged_entries, remote_prefix=prefix,
            max_workers=_LARGE_SOURCE_UPLOAD_WORKERS,
        )
        revision = resolve_approved_ref(
            transport=transport, repository=repository, ref=revision,
        )
    else:
        revision = upload_files(
            transport=transport, repository=repository, revision=revision,
            source=local, entries=entries, remote_prefix=prefix, max_attempts=1,
        )
    tree = list_repository_tree(
        transport=transport, repository=repository, revision=revision,
        remote_prefix=prefix, max_attempts=1,
    )
    if not _tree_matches(tree, prefix=prefix, entries=entries):
        raise ValueError("source publication remote tree differs from the complete local source")
    _write_source_upload_journal(
        path=journal, revision=revision, prefix=prefix, source_type=source_type,
        round_id=round_id, entries=entries,
    )
    return verify_uploaded_runtime_source(
        root=local, upload_journal_path=journal, readback_root=stable_readback,
        receipt_path=target, transport=transport,
    )


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
    tree = list_repository_tree(
        transport=transport, repository=APPROVED_MIXTURE_REPOSITORY, revision=immutable,
        remote_prefix=prefix, max_attempts=1,
    )
    if not _tree_matches(tree, prefix=prefix, entries=entries):
        raise ValueError("mixture publication remote tree differs from the complete pending artifact")
    with tempfile.TemporaryDirectory(prefix="lehome-mixture-readback-") as temporary:
        readback = Path(temporary)
        download_files(transport=transport, repository=APPROVED_MIXTURE_REPOSITORY, revision=immutable, destination=readback, relative_paths=tuple(item.relative_path for item in entries), remote_prefix=prefix, max_attempts=1)
        if _entries(readback) != entries:
            raise ValueError("mixture publication readback hash or size mismatch")
    receipt = {"repository": APPROVED_MIXTURE_REPOSITORY, "immutable_revision": immutable, "remote_prefix": prefix, "mixture_id": pending["mixture_id"], "fresh_readback_verified": True, "tree_listing_verified": True}
    if pending["schema_version"] == 2:
        receipt.update({key: pending[key] for key in ("experiment_manifest_sha256", "mixture_weights", "source_quotas")})
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
    receipt_keys = {"repository", "immutable_revision", "remote_prefix", "mixture_id", "fresh_readback_verified", "tree_listing_verified"}
    if pending.get("schema_version") == 2:
        receipt_keys |= {"experiment_manifest_sha256", "mixture_weights", "source_quotas"}
    receipt = _load_exact(release, keys=receipt_keys, label="mixture publication receipt")
    if (
        output.exists()
        or pending.get("schema_version") not in {1, 2}
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
    if pending["schema_version"] == 2 and any(receipt[key] != pending[key] for key in ("experiment_manifest_sha256", "mixture_weights", "source_quotas")):
        raise ValueError("mixture finalization experiment binding drift")
    normalization = root / "mixture-normalization.json"
    windows_file = root / "windows.json"
    if sha256_file(normalization) != pending["normalization_sha256"] or sha256_file(windows_file) != pending["windows_sha256"]:
        raise ValueError("mixture finalization pending bytes drift")
    windows_value = _load_exact(windows_file, keys={"schema_version", "windows"}, label="pending window index")
    if type(windows_value["schema_version"]) is not int or windows_value["schema_version"] != 3 or not isinstance(windows_value["windows"], list):
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
        manifest = {"schema_version": 3 if pending["schema_version"] == 2 else 2, "kind": "lehome_runtime_mixture", "repository": APPROVED_MIXTURE_REPOSITORY, "safe_prefix": pending["prefix"], "mixture_id": pending["mixture_id"], "sources": pending["sources"], "camera_schema": list(CAMERAS), "image_shape": [480, 640, 3], "state_schema": {"dimension": 12, "storage": "absolute"}, "action_schema": {"dimension": 12, "storage": "absolute"}, "fps": FPS, "action_horizon": ACTION_HORIZON, "instruction": INSTRUCTION, "schedule_seed": 17, "cycle_size": sum(source["quota"] for source in pending["sources"]), "mixture_normalization": {"path": "mixture-normalization.json", "sha256": sha256_file(staging / "mixture-normalization.json"), "byte_size": (staging / "mixture-normalization.json").stat().st_size}, "window_index": {"path": "windows.json", "sha256": "", "byte_size": 0}}
        if pending["schema_version"] == 2:
            manifest.update({key: pending[key] for key in ("experiment_manifest_sha256", "mixture_weights", "source_quotas")})
        index = {"schema_version": 2, "manifest_sha256": canonical_json_sha256(manifest), "windows": windows}
        atomic_write_json(staging / "windows.json", index)
        manifest["window_index"] = {"path": "windows.json", "sha256": sha256_file(staging / "windows.json"), "byte_size": (staging / "windows.json").stat().st_size}
        atomic_write_json(staging / "mixture.json", manifest)
        entries = _entries(staging)
        require_access(transport=transport, repository=APPROVED_MIXTURE_REPOSITORY, read=True, write=True)
        immutable = upload_files(transport=transport, repository=APPROVED_MIXTURE_REPOSITORY, revision=revision, source=staging, entries=entries, remote_prefix=str(pending["prefix"]), max_attempts=1)
        tree = list_repository_tree(
            transport=transport, repository=APPROVED_MIXTURE_REPOSITORY, revision=immutable,
            remote_prefix=str(pending["prefix"]), max_attempts=1,
        )
        if not _tree_matches(tree, prefix=str(pending["prefix"]), entries=entries):
            raise ValueError("runtime deployment remote tree differs from finalized bytes")
        with tempfile.TemporaryDirectory(prefix="lehome-runtime-final-readback-") as temporary:
            readback = Path(temporary)
            download_files(transport=transport, repository=APPROVED_MIXTURE_REPOSITORY, revision=immutable, destination=readback, relative_paths=tuple(entry.relative_path for entry in entries), remote_prefix=str(pending["prefix"]), max_attempts=1)
            if _entries(readback) != entries:
                raise ValueError("runtime deployment readback hash or size mismatch")
        deployment_value = {"repository": APPROVED_MIXTURE_REPOSITORY, "immutable_revision": immutable, "remote_prefix": pending["prefix"], "mixture_id": pending["mixture_id"], "pending_receipt_sha256": sha256_file(release), "artifact_entries": _entry_records(entries), "fresh_readback_verified": True, "tree_listing_verified": True}
        if pending["schema_version"] == 2:
            deployment_value.update({key: pending[key] for key in ("experiment_manifest_sha256", "mixture_weights", "source_quotas")})
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
    legacy = {"root", "source_type", "round_id", "revision", "receipt_path"}
    resumable = legacy | {"upload_journal_path", "readback_root"}
    large = resumable | {"large_upload", "large_upload_staging_root"}
    if (
        request["schema_version"] != 1
        or request["command"] != "publish-runtime-source"
        or not isinstance(arguments, dict)
        or frozenset(arguments) not in {frozenset(legacy), frozenset(resumable), frozenset(large)}
        or any(type(arguments[key]) is not str or not arguments[key] for key in set(arguments) - {"round_id", "large_upload"})
        or (arguments["round_id"] is not None and (type(arguments["round_id"]) is not str or not arguments["round_id"]))
        or (set(arguments) == large and arguments["large_upload"] is not True)
    ):
        raise ValueError("runtime source publication request has an incompatible schema")
    return publish_source(**arguments, transport=transport)


def verify_uploaded_runtime_source_from_request(path: str | Path, *, transport: HubTransport) -> dict[str, object]:
    """Run no-upload source verification from one exact reviewable envelope."""

    request = _load_exact(Path(path), keys={"schema_version", "command", "arguments"}, label="runtime source verification request")
    arguments = request["arguments"]
    expected = {"root", "upload_journal_path", "readback_root", "receipt_path"}
    if (
        request["schema_version"] != 1
        or request["command"] != "verify-uploaded-runtime-source"
        or not isinstance(arguments, dict)
        or set(arguments) != expected
        or any(type(arguments[key]) is not str or not arguments[key] for key in expected)
    ):
        raise ValueError("runtime source verification request has an incompatible schema")
    return verify_uploaded_runtime_source(**arguments, transport=transport)


def adopt_uploaded_runtime_source_from_request(path: str | Path, *, transport: HubTransport) -> dict[str, object]:
    """Create an upload journal for an exact pre-existing immutable source."""

    request = _load_exact(Path(path), keys={"schema_version", "command", "arguments"}, label="runtime source adoption request")
    arguments = request["arguments"]
    expected = {"root", "source_type", "round_id", "immutable_revision", "upload_journal_path"}
    if (
        request["schema_version"] != 1
        or request["command"] != "adopt-uploaded-runtime-source"
        or not isinstance(arguments, dict)
        or set(arguments) != expected
        or any(type(arguments[key]) is not str or not arguments[key] for key in expected - {"round_id"})
        or (arguments["round_id"] is not None and (type(arguments["round_id"]) is not str or not arguments["round_id"]))
    ):
        raise ValueError("runtime source adoption request has an incompatible schema")
    return adopt_uploaded_runtime_source(**arguments, transport=transport)


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
