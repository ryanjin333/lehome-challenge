"""Publish the inference subset of a completed GR00T checkpoint.

Full optimizer checkpoints remain the authoritative preemption recovery state on
the protected workspace disk.  This module deliberately publishes only the
closed set of files needed to load the policy for evaluation or rollout.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
from typing import Mapping, Sequence

from lehome_train.constants import DEFAULT_MODEL_REPO
from lehome_train.hub import (
    HubTransport,
    HuggingFaceHubTransport,
    download_files,
    list_repository_tree,
    require_access,
    resolve_approved_ref,
    upload_large_folder,
)
from lehome_train.io import atomic_write_json, canonical_json_sha256, sha256_file
from lehome_train.models import SyncEntry


_EXPERIMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHARD = re.compile(r"^model-([0-9]{5})-of-([0-9]{5})\.safetensors$")
_REQUIRED_ROOT_FILES = (
    "config.json",
    "processor_config.json",
    "embodiment_id.json",
    "statistics.json",
    "model.safetensors.index.json",
)
_REQUIRED_EXPERIMENT_FILES = (
    "experiment_cfg/config.yaml",
    "experiment_cfg/conf.yaml",
    "experiment_cfg/dataset_statistics.json",
    "experiment_cfg/final_model_config.json",
    "experiment_cfg/final_processor_config.json",
)
_MANIFEST_NAME = "deployable-manifest.json"


def _strict_json(path: Path) -> Mapping[str, object]:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("checkpoint JSON contains a duplicate field")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=object_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("checkpoint JSON is unavailable or malformed") from None
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint JSON must contain an object")
    return value


def _regular(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} must be a nonempty regular file")


def _source_paths(checkpoint: Path, optimizer_step: int) -> tuple[str, ...]:
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError("checkpoint root must be a regular directory")
    if checkpoint.name != f"checkpoint-{optimizer_step}" or optimizer_step != 2000:
        raise ValueError("deployable publication requires the completed step-2000 checkpoint")
    for relative in _REQUIRED_ROOT_FILES + _REQUIRED_EXPERIMENT_FILES:
        _regular(checkpoint / relative, relative)

    index = _strict_json(checkpoint / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError("checkpoint model index has no weight map")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in weight_map.items()):
        raise ValueError("checkpoint model index is malformed")
    shard_names = tuple(sorted(set(str(value) for value in weight_map.values())))
    parsed = [(_SHARD.fullmatch(name), name) for name in shard_names]
    if any(match is None for match, _ in parsed):
        raise ValueError("checkpoint model index contains an unsafe shard name")
    totals = {int(match.group(2)) for match, _ in parsed if match is not None}
    if len(totals) != 1:
        raise ValueError("checkpoint model index mixes shard totals")
    total = totals.pop()
    expected = tuple(f"model-{number:05d}-of-{total:05d}.safetensors" for number in range(1, total + 1))
    if shard_names != expected:
        raise ValueError("checkpoint model index is incomplete")
    observed_models = tuple(sorted(path.name for path in checkpoint.glob("*.safetensors")))
    if observed_models != expected:
        raise ValueError("checkpoint contains mixed, missing, or unindexed model files")
    for shard in expected:
        _regular(checkpoint / shard, shard)
    return tuple(sorted(_REQUIRED_ROOT_FILES + _REQUIRED_EXPERIMENT_FILES + expected))


def _entry(root: Path, relative: str) -> SyncEntry:
    path = root / relative
    _regular(path, relative)
    return SyncEntry(relative, sha256_file(path), path.stat().st_size, False)


@dataclass(frozen=True, slots=True)
class DeployableCheckpoint:
    experiment_id: str
    optimizer_step: int
    stage_root: Path
    payload_root: Path
    remote_prefix: str
    entries: tuple[SyncEntry, ...]
    sha256: str
    byte_size: int


def _verify_bundle(bundle: DeployableCheckpoint) -> None:
    if bundle.stage_root.is_symlink() or not bundle.stage_root.is_dir():
        raise ValueError("deployable staging root is unavailable")
    if bundle.payload_root != bundle.stage_root / bundle.remote_prefix:
        raise ValueError("deployable payload root is not bound to its remote prefix")
    expected = {entry.relative_path for entry in bundle.entries}
    observed = {
        path.relative_to(bundle.payload_root).as_posix()
        for path in bundle.payload_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if observed != expected or any(path.is_symlink() for path in bundle.payload_root.rglob("*")):
        raise ValueError("deployable staging tree differs from its allowlist")
    for entry in bundle.entries:
        path = bundle.payload_root / entry.relative_path
        if path.stat().st_size != entry.byte_size or sha256_file(path) != entry.sha256:
            raise ValueError("deployable staging bytes changed after construction")
    if sha256_file(bundle.payload_root / _MANIFEST_NAME) != bundle.sha256:
        raise ValueError("deployable manifest identity changed")


def build_deployable_checkpoint(
    checkpoint: str | os.PathLike[str],
    *,
    staging_root: str | os.PathLike[str],
    experiment_id: str,
    optimizer_step: int,
) -> DeployableCheckpoint:
    """Build an immutable hard-link tree containing no optimizer recovery state."""

    if not isinstance(experiment_id, str) or _EXPERIMENT.fullmatch(experiment_id) is None:
        raise ValueError("experiment ID is not a safe path component")
    source = Path(checkpoint)
    relative_paths = _source_paths(source, optimizer_step)
    source_entries = tuple(_entry(source, relative) for relative in relative_paths)
    identity = {
        "schema_version": 1,
        "kind": "lehome_groot_deployable_checkpoint",
        "experiment_id": experiment_id,
        "optimizer_step": optimizer_step,
        "source_checkpoint": source.name,
        "entries": [entry.to_dict() for entry in source_entries],
    }
    bundle_id = canonical_json_sha256(identity)
    remote_prefix = f"policies/{experiment_id}/step-{optimizer_step}/{bundle_id}"
    parent = Path(staging_root)
    if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
        raise ValueError("deployable staging parent is unsafe")
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / f"deployable-{bundle_id}"
    incomplete = parent / f".deployable-{bundle_id}.incomplete"
    if final.exists() or incomplete.exists():
        raise ValueError("deployable staging identity already exists")
    payload = incomplete / remote_prefix
    payload.mkdir(parents=True)
    try:
        for entry in source_entries:
            target = payload / entry.relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source / entry.relative_path, target, follow_symlinks=False)
            except OSError:
                raise ValueError("deployable staging requires same-filesystem hard links") from None
        atomic_write_json(payload / _MANIFEST_NAME, identity | {"remote_prefix": remote_prefix})
        entries = tuple(sorted(source_entries + (_entry(payload, _MANIFEST_NAME),), key=lambda item: item.relative_path))
        manifest_sha256 = sha256_file(payload / _MANIFEST_NAME)
        os.replace(incomplete, final)
        bundle = DeployableCheckpoint(
            experiment_id=experiment_id,
            optimizer_step=optimizer_step,
            stage_root=final,
            payload_root=final / remote_prefix,
            remote_prefix=remote_prefix,
            entries=entries,
            sha256=manifest_sha256,
            byte_size=sum(entry.byte_size for entry in entries),
        )
        _verify_bundle(bundle)
        return bundle
    except BaseException:
        shutil.rmtree(incomplete, ignore_errors=True)
        raise


def publish_deployable_checkpoint(
    bundle: DeployableCheckpoint,
    *,
    repository: str,
    revision: str,
    token: str,
    transport: HubTransport,
    readback_root: str | os.PathLike[str],
    receipt_path: str | os.PathLike[str],
) -> dict[str, object]:
    """Upload one private deployable tree and byte-read it from its immutable commit."""

    if repository != DEFAULT_MODEL_REPO or revision != "main":
        raise ValueError("deployable publication target is not approved")
    if not isinstance(token, str) or not token or any(character.isspace() for character in token):
        raise ValueError("deployable publisher token is invalid")
    _verify_bundle(bundle)
    environ = {"HF_TOKEN": token}
    require_access(transport=transport, repository=repository, read=True, write=True, environ=environ)
    upload_entries = tuple(
        SyncEntry(
            f"{bundle.remote_prefix}/{entry.relative_path}",
            entry.sha256,
            entry.byte_size,
            False,
        )
        for entry in bundle.entries
    )
    upload_large_folder(
        transport=transport,
        repository=repository,
        revision=revision,
        source=bundle.stage_root,
        entries=upload_entries,
        remote_prefix=bundle.remote_prefix,
        environ=environ,
        max_workers=4,
    )
    immutable_revision = resolve_approved_ref(
        transport=transport,
        repository=repository,
        ref=revision,
        environ=environ,
    )
    tree = list_repository_tree(
        transport=transport,
        repository=repository,
        revision=immutable_revision,
        remote_prefix=bundle.remote_prefix,
        environ=environ,
        max_attempts=3,
    )
    prefix = bundle.remote_prefix + "/"
    observed = {
        entry.relative_path[len(prefix) :]
        for entry in tree
        if entry.entry_type == "file" and entry.relative_path.startswith(prefix)
    }
    expected = {entry.relative_path for entry in bundle.entries}
    if observed != expected:
        raise ValueError("immutable Hub tree differs from deployable allowlist")

    readback = Path(readback_root)
    if readback.exists() or readback.is_symlink():
        raise ValueError("deployable readback root must be absent")
    try:
        download_files(
            transport=transport,
            repository=repository,
            revision=immutable_revision,
            destination=readback,
            relative_paths=tuple(entry.relative_path for entry in bundle.entries),
            remote_prefix=bundle.remote_prefix,
            environ=environ,
            max_attempts=3,
        )
        for entry in bundle.entries:
            path = readback / entry.relative_path
            if path.is_symlink() or not path.is_file() or path.stat().st_size != entry.byte_size or sha256_file(path) != entry.sha256:
                raise ValueError("deployable Hub readback differs from local identity")
        receipt = {
            "schema_version": 1,
            "kind": "lehome_groot_deployable_checkpoint_publication",
            "repository": repository,
            "private_repository": True,
            "immutable_revision": immutable_revision,
            "remote_prefix": bundle.remote_prefix,
            "experiment_id": bundle.experiment_id,
            "optimizer_step": bundle.optimizer_step,
            "bundle_sha256": bundle.sha256,
            "bundle_byte_size": bundle.byte_size,
            "entries": [entry.to_dict() | {"remotely_verified": True} for entry in bundle.entries],
            "readback_verified": True,
        }
        receipt_destination = Path(receipt_path)
        receipt_parent = receipt_destination.parent
        if receipt_parent.is_symlink() or (receipt_parent.exists() and not receipt_parent.is_dir()):
            raise ValueError("deployable receipt parent is unsafe")
        receipt_parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(receipt_destination, receipt)
        return receipt
    finally:
        shutil.rmtree(readback, ignore_errors=True)


def _token(path: Path) -> str:
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o077:
        raise ValueError("publisher token file must be a private regular file")
    token = path.read_text(encoding="utf-8").strip()
    if not token or any(character.isspace() for character in token):
        raise ValueError("publisher token file is invalid")
    return token


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--optimizer-step", type=int, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--readback-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    bundle = build_deployable_checkpoint(
        args.checkpoint,
        staging_root=args.staging_root,
        experiment_id=args.experiment_id,
        optimizer_step=args.optimizer_step,
    )
    publish_deployable_checkpoint(
        bundle,
        repository=DEFAULT_MODEL_REPO,
        revision="main",
        token=_token(args.token_file),
        transport=HuggingFaceHubTransport(timeout_seconds=120.0),
        readback_root=args.readback_root,
        receipt_path=args.receipt,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
