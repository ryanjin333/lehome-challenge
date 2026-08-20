#!/usr/bin/env python3
"""Bind sealed success-replay data to one immutable runtime mixture for AWR."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from typing import Mapping, Sequence

from lehome_train.constants import DEFAULT_ROLLOUT_REPO
from lehome_train.groot.runtime_mixture import (
    _manifest_digest_binding,
    _parse_manifest,
    _parse_window,
)
from lehome_train.io import canonical_json_bytes, canonical_json_sha256, sha256_file


_CATEGORIES = frozenset({"top_long", "top_short", "pant_long", "pant_short"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$")


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _reject_nonfinite(_: str) -> object:
    raise ValueError("non-finite JSON value")


def _load_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is invalid") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} is invalid")
    return value


def _exact(value: Mapping[str, object], keys: set[str], *, label: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{label} has unknown or missing fields")


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a SHA-256")
    return value


def _identifier(value: object, *, label: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} is unsafe")
    return value


def _assert_no_symlink_components(path: Path, *, label: str) -> None:
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            if stat.S_ISLNK(current.lstat().st_mode):
                raise ValueError(f"{label} is unsafe")
        except FileNotFoundError:
            break
        except OSError as error:
            raise ValueError(f"{label} is unavailable") from error


def _real_directory(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    _assert_no_symlink_components(path, label=label)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be a real directory")
    return path.resolve(strict=True)


def _real_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    _assert_no_symlink_components(path, label=label)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    return path.resolve(strict=True)


def _ordered_path_inputs(
    value: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    *,
    label: str,
) -> tuple[str | os.PathLike[str], ...]:
    if isinstance(value, (str, os.PathLike)):
        return (value,)
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise ValueError(f"{label} must be one path or an ordered path list")
    paths = tuple(value)
    if not paths or any(not isinstance(path, (str, os.PathLike)) for path in paths):
        raise ValueError(f"{label} must be a nonempty ordered path list")
    return paths


def _safe_child(root: Path, relative: str, *, label: str) -> Path:
    path = PurePosixPath(relative)
    if not relative or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} is unsafe")
    current = root
    for part in path.parts:
        current /= part
        try:
            mode = current.lstat().st_mode
        except OSError as error:
            raise ValueError(f"{label} is unavailable") from error
        if stat.S_ISLNK(mode):
            raise ValueError(f"{label} is unsafe")
    if not stat.S_ISREG(current.lstat().st_mode):
        raise ValueError(f"{label} is not a regular file")
    return current


def _walk_regular_files(root: Path, *, label: str) -> dict[str, Path]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{label} is unsafe")
    files: dict[str, Path] = {}
    for current, directories, names in os.walk(root, followlinks=False):
        base = Path(current)
        for directory in directories:
            if (base / directory).is_symlink():
                raise ValueError(f"{label} contains a symlink")
        for name in names:
            path = base / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"{label} contains an unsafe file")
            relative = path.relative_to(root).as_posix()
            if relative in files:
                raise ValueError(f"{label} contains duplicate file identity")
            files[relative] = path
    return files


def _directory_entries(root: Path, *, label: str) -> dict[str, Path]:
    try:
        entries = sorted(root.iterdir(), key=lambda entry: entry.name)
    except OSError as error:
        raise ValueError(f"{label} is unavailable") from error
    result = {entry.name: entry for entry in entries}
    if len(result) != len(entries):
        raise ValueError(f"{label} contains duplicate path identity")
    return result


def _reject_overlapping_source_paths(
    *,
    roots: Sequence[Path],
    files: Sequence[Path],
    output: Path,
) -> None:
    for index, root in enumerate(roots):
        for other in roots[index + 1:]:
            if root in other.parents or other in root.parents:
                raise ValueError("ordered success replay source roots overlap")
        if any(root in file.parents for file in files) or root in output.parents:
            raise ValueError("ordered success replay source path overlaps another input or output")


def _verify_artifact_manifest(root: Path) -> dict[str, Path]:
    files = _walk_regular_files(root, label="accepted replay artifact")
    manifest_path = files.get("SHA256SUMS.json")
    if manifest_path is None:
        raise ValueError("accepted replay artifact lacks checksum manifest")
    manifest = _load_json(manifest_path, label="accepted replay checksum manifest")
    checked: dict[str, Path] = {}
    for relative, record in manifest.items():
        if not isinstance(relative, str) or not isinstance(record, Mapping):
            raise ValueError("accepted replay checksum manifest is malformed")
        _exact(record, {"sha256", "size"}, label="accepted replay checksum entry")
        expected_hash = _sha256(record["sha256"], label="accepted replay checksum")
        if type(record["size"]) is not int or record["size"] < 0:
            raise ValueError("accepted replay checksum size is invalid")
        file = _safe_child(root, relative, label="accepted replay checksum target")
        if file.stat().st_size != record["size"] or sha256_file(file) != expected_hash:
            raise ValueError("accepted replay artifact checksum mismatch")
        checked[relative] = file
    actual = set(files) - {"SHA256SUMS.json"}
    if set(checked) != actual:
        raise ValueError("accepted replay artifact has unsealed or missing files")
    return checked


def _episode_digest(files: Mapping[str, Path]) -> str:
    entries = [
        {"relative_path": relative, "sha256": sha256_file(path), "byte_size": path.stat().st_size}
        for relative, path in sorted(files.items())
    ]
    return hashlib.sha256(canonical_json_bytes(entries)).hexdigest()


def _load_round_seal(path: Path) -> tuple[dict[str, object], dict[str, str], dict[str, str]]:
    seal = _load_json(path, label="success replay round seal")
    _exact(
        seal,
        {
            "schema_version", "kind", "repository", "round_id", "episode_count",
            "episode_sha256s", "immutable_revisions", "readback_verified", "seal_sha256",
        },
        label="success replay round seal",
    )
    if seal["schema_version"] != 2 or seal["kind"] != "rollout_round_seal" or seal["readback_verified"] is not True:
        raise ValueError("success replay round seal schema is invalid")
    if seal["repository"] != DEFAULT_ROLLOUT_REPO:
        raise ValueError("success replay round seal repository is invalid")
    round_id = _identifier(seal["round_id"], label="success replay round ID")
    if type(seal["episode_count"]) is not int or seal["episode_count"] < 1:
        raise ValueError("success replay round seal episode count is invalid")
    episode_hashes, revisions = seal["episode_sha256s"], seal["immutable_revisions"]
    if not isinstance(episode_hashes, dict) or not isinstance(revisions, dict):
        raise ValueError("success replay round seal maps are invalid")
    if set(episode_hashes) != set(revisions) or len(episode_hashes) != seal["episode_count"]:
        raise ValueError("success replay round seal episode coverage is invalid")
    hashes: dict[str, str] = {}
    immutable_revisions: dict[str, str] = {}
    for episode_id in episode_hashes:
        _identifier(episode_id, label="sealed replay episode ID")
        hashes[episode_id] = _sha256(episode_hashes[episode_id], label="sealed replay episode hash")
        revision = revisions[episode_id]
        if type(revision) is not str or _REVISION.fullmatch(revision) is None:
            raise ValueError("sealed replay immutable revision is invalid")
        immutable_revisions[episode_id] = revision
    claimed = _sha256(seal["seal_sha256"], label="success replay seal hash")
    seal_body = {
        "round_id": seal["round_id"],
        "repository": seal["repository"],
        "episode_sha256s": seal["episode_sha256s"],
        "immutable_revisions": seal["immutable_revisions"],
    }
    if canonical_json_sha256(seal_body) != claimed:
        raise ValueError("success replay round seal digest mismatch")
    return seal, hashes, immutable_revisions


def _validate_receipt(
    path: Path,
    *,
    episode_id: str,
    episode_sha256: str,
    immutable_revision: str,
    seal: Mapping[str, object],
) -> None:
    receipt = _load_json(path, label="success replay Hub receipt")
    _exact(
        receipt,
        {
            "schema_version", "attempt_id", "repository", "round_id", "remote_prefix",
            "publication_ref", "immutable_revision", "entry_count", "episode_sha256", "readback_verified",
        },
        label="success replay Hub receipt",
    )
    if (
        receipt["schema_version"] != 1
        or receipt["attempt_id"] != episode_id
        or receipt["repository"] != seal["repository"]
        or receipt["round_id"] != seal["round_id"]
        or receipt["remote_prefix"] != f"rollout-rounds/{seal['round_id']}/{episode_id}"
        or receipt["immutable_revision"] != immutable_revision
        or receipt["episode_sha256"] != episode_sha256
        or receipt["readback_verified"] is not True
        or type(receipt["publication_ref"]) is not str
        or not receipt["publication_ref"]
        or type(receipt["entry_count"]) is not int
        or receipt["entry_count"] < 1
    ):
        raise ValueError("success replay Hub receipt does not authenticate sealed episode")


def _progress_from_annotations(path: Path) -> tuple[float, int]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError("replay annotations are unavailable") from error
    if not lines:
        raise ValueError("replay annotations are empty")
    rewards: list[float] = []
    first_success: int | None = None
    for expected_step, line in enumerate(lines):
        try:
            record = json.loads(line, object_pairs_hook=_strict_pairs, parse_constant=_reject_nonfinite)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError("replay annotation is invalid") from error
        if not isinstance(record, dict):
            raise ValueError("replay annotation is invalid")
        step, reward, success = record.get("step"), record.get("reward"), record.get("success")
        if type(step) is not int or step != expected_step:
            raise ValueError("replay annotation steps are not contiguous")
        if type(reward) not in (int, float) or not math.isfinite(float(reward)):
            raise ValueError("replay annotation reward is non-finite")
        if type(success) is not bool:
            raise ValueError("replay annotation success is invalid")
        rewards.append(float(reward))
        if success and first_success is None:
            first_success = step
    if first_success is None:
        raise ValueError("accepted replay episode has no success progress")
    return sum(rewards) / len(rewards), first_success


def _validate_accepted_episode(root: Path, *, episode_id: str) -> tuple[str, float, int, str]:
    files = _verify_artifact_manifest(root)
    raw_prefix = f"raw/{episode_id}/"
    episode_path = files.get(raw_prefix + "episode.json")
    annotations_path = files.get(raw_prefix + "annotations.jsonl")
    if episode_path is None or annotations_path is None:
        raise ValueError("accepted replay artifact has no raw episode provenance")
    episode = _load_json(episode_path, label="accepted replay episode")
    identity = episode.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("accepted replay episode identity is invalid")
    if (
        episode.get("episode_id") != episode_id
        or episode.get("accepted_success") is not True
        or episode.get("outcome") != "success"
        or identity.get("episode_id") != episode_id
        or identity.get("release_stage") != "seen"
        or identity.get("category") not in _CATEGORIES
    ):
        raise ValueError("accepted replay episode is not a seen success")
    mean_reward, first_success = _progress_from_annotations(annotations_path)
    return str(identity["category"]), mean_reward, first_success, _episode_digest(files)


def _load_runtime_windows(path: Path) -> tuple[object, dict[str, str]]:
    manifest = _parse_manifest(path)
    index_path = _safe_child(path.parent, manifest.window_index_path, label="runtime mixture window index")
    if index_path.stat().st_size != manifest.window_index_byte_size or sha256_file(index_path) != manifest.window_index_sha256:
        raise ValueError("runtime mixture window index binding mismatch")
    index = _load_json(index_path, label="runtime mixture window index")
    _exact(index, {"schema_version", "manifest_sha256", "windows"}, label="runtime mixture window index")
    if index["schema_version"] != 2 or index["manifest_sha256"] != _manifest_digest_binding(manifest.raw) or not isinstance(index["windows"], list):
        raise ValueError("runtime mixture window index immutable binding mismatch")
    windows = tuple(_parse_window(item) for item in index["windows"])
    source_types = {source.source_id: source.source_type for source in manifest.sources}
    lineages: dict[str, set[str]] = defaultdict(set)
    selected: dict[str, str] = {}
    lineage_episodes: dict[str, str] = {}
    window_ids: set[str] = set()
    train_locators: dict[str, tuple[str, str]] = {}
    train_path_owners: dict[str, str] = {}
    for window in windows:
        if window.window_id in window_ids:
            raise ValueError("runtime mixture window ID is duplicated")
        window_ids.add(window.window_id)
        if source_types.get(window.source_id) != window.source_type:
            raise ValueError("runtime mixture window source binding mismatch")
        lineages[window.lineage_id].add(window.split)
        if window.source_type == "rollout" and window.split == "train":
            prior = selected.setdefault(window.source_episode_id, window.lineage_id)
            if prior != window.lineage_id:
                raise ValueError("runtime mixture rollout lineage is ambiguous")
            prior_episode = lineage_episodes.setdefault(window.lineage_id, window.source_episode_id)
            if prior_episode != window.source_episode_id:
                raise ValueError("runtime mixture rollout lineage joins distinct episodes")
            locator = (
                str(window.source_locator["attempt_root"]),
                str(window.source_locator["attempt_manifest_path"]),
            )
            prior_locator = train_locators.setdefault(window.source_episode_id, locator)
            if prior_locator != locator:
                raise ValueError("runtime mixture rollout episode has ambiguous source paths")
            for source_path in locator:
                prior_owner = train_path_owners.setdefault(source_path, window.source_episode_id)
                if prior_owner != window.source_episode_id:
                    raise ValueError("runtime mixture rollout source path joins distinct episodes")
    if any(len(splits) > 1 for splits in lineages.values()):
        raise ValueError("runtime mixture has train-validation lineage overlap")
    if not selected:
        raise ValueError("runtime mixture has no rollout training episodes")
    return manifest, selected


def _category_scores(metrics: Mapping[str, tuple[str, float, int]]) -> dict[str, float]:
    groups: dict[str, list[tuple[str, float, int]]] = defaultdict(list)
    for episode_id, (category, mean_reward, first_success) in metrics.items():
        groups[category].append((episode_id, mean_reward, first_success))
    scores: dict[str, float] = {}
    for entries in groups.values():
        # Earlier terminal success is the primary progress signal.  Mean reward
        # resolves actual trajectory quality; equal measurements receive equal
        # normalized ranks, not an arbitrary episode-ID preference.
        ordered = sorted(entries, key=lambda row: (row[2], -row[1], row[0]))
        size = len(ordered)
        position = 0
        while position < size:
            key = (ordered[position][2], ordered[position][1])
            end = position + 1
            while end < size and (ordered[end][2], ordered[end][1]) == key:
                end += 1
            percentile = 0.5 if size == 1 else ((position + end - 1) / 2.0) / (size - 1)
            # Higher quality receives a larger finite AWR replay score.
            score = round(1.0 - 2.0 * percentile, 12)
            for index in range(position, end):
                scores[ordered[index][0]] = score
            position = end
    return scores


def _preflight_output(path: Path) -> tuple[Path, Path]:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise FileExistsError("AWR evidence output must be an absent absolute path")
    parent = path.parent
    _assert_no_symlink_components(parent, label="AWR evidence output parent")
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("AWR evidence output parent is unsafe")
    sidecar = Path(str(path) + ".sha256")
    if sidecar.exists() or sidecar.is_symlink():
        raise FileExistsError("AWR evidence sidecar must be absent")
    return parent.resolve(strict=True), sidecar


def _stage_payload(parent: Path, destination: Path, payload: bytes) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _fsync_directory(parent: Path) -> None:
    directory = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _write_absent_pair(
    parent: Path,
    *,
    output: Path,
    payload: bytes,
    sidecar: Path,
    sidecar_payload: bytes,
) -> None:
    """Publish the immutable evidence pair, rolling back a partial commit."""

    output_temporary = _stage_payload(parent, output, payload)
    sidecar_temporary: Path | None = None
    published: list[Path] = []
    try:
        sidecar_temporary = _stage_payload(parent, sidecar, sidecar_payload)
        os.link(output_temporary, output)
        published.append(output)
        os.link(sidecar_temporary, sidecar)
        published.append(sidecar)
        _fsync_directory(parent)
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        _fsync_directory(parent)
        raise
    finally:
        output_temporary.unlink(missing_ok=True)
        if sidecar_temporary is not None:
            sidecar_temporary.unlink(missing_ok=True)


def build_awr_progress_evidence(
    *,
    mixture_manifest: str | Path,
    accepted_root: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    sync_receipts_root: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    round_seal: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    output: str | Path,
) -> dict[str, object]:
    """Build one immutable train-only evidence document or publish nothing.

    The three source arguments accept either their historical single path or
    equally sized ordered path lists.  List position is part of the immutable
    source-round lineage.
    """
    manifest_path = _real_file(mixture_manifest, label="runtime mixture manifest")
    accepted_inputs = _ordered_path_inputs(accepted_root, label="accepted success replay roots")
    receipt_inputs = _ordered_path_inputs(sync_receipts_root, label="success replay receipt roots")
    seal_inputs = _ordered_path_inputs(round_seal, label="success replay round seals")
    if len(accepted_inputs) != len(receipt_inputs) or len(accepted_inputs) != len(seal_inputs):
        raise ValueError("ordered success replay source lists must have equal lengths")
    accepted_roots = tuple(
        _real_directory(value, label=f"accepted success replay root {index}")
        for index, value in enumerate(accepted_inputs)
    )
    receipt_roots = tuple(
        _real_directory(value, label=f"success replay receipt root {index}")
        for index, value in enumerate(receipt_inputs)
    )
    seal_paths = tuple(
        _real_file(value, label=f"success replay round seal {index}")
        for index, value in enumerate(seal_inputs)
    )
    source_paths = (*accepted_roots, *receipt_roots, *seal_paths)
    if len(set(source_paths)) != len(source_paths):
        raise ValueError("ordered success replay sources contain a path collision")
    destination = Path(output)
    parent, sidecar = _preflight_output(destination)
    canonical_destination = parent / destination.name
    _reject_overlapping_source_paths(
        roots=(*accepted_roots, *receipt_roots),
        files=(manifest_path, *seal_paths),
        output=canonical_destination,
    )

    manifest, selected = _load_runtime_windows(manifest_path)
    metrics: dict[str, tuple[str, float, int]] = {}
    receipt_paths: dict[str, Path] = {}
    provenance_paths: dict[str, str] = {}
    source_episode_order: list[str] = []
    source_rounds: list[dict[str, object]] = []
    round_ids: set[str] = set()
    logical_paths: set[str] = set()
    for source_index, (accepted, receipts, seal_path) in enumerate(
        zip(accepted_roots, receipt_roots, seal_paths, strict=True)
    ):
        seal, sealed_hashes, revisions = _load_round_seal(seal_path)
        round_id = str(seal["round_id"])
        if round_id in round_ids:
            raise ValueError("ordered success replay sources contain a round ID collision")
        round_ids.add(round_id)
        accepted_entries = _directory_entries(
            accepted, label="accepted success replay root",
        )
        if (
            set(accepted_entries) != set(sealed_hashes)
            or any(entry.is_symlink() or not entry.is_dir() for entry in accepted_entries.values())
        ):
            raise ValueError("accepted success replay root has missing, extra, or unsafe episodes")
        receipt_entries = _directory_entries(
            receipts, label="success replay receipt root",
        )
        expected_receipts = {f"{episode_id}.sync.json" for episode_id in sealed_hashes}
        if (
            set(receipt_entries) != expected_receipts
            or any(entry.is_symlink() or not entry.is_file() for entry in receipt_entries.values())
        ):
            raise ValueError("success replay receipt root has missing, extra, or unsafe receipts")
        seal_sha256 = str(seal["seal_sha256"])
        source_rounds.append({
            "source_index": source_index,
            "round_id": round_id,
            "repository": str(seal["repository"]),
            "seal_sha256": seal_sha256,
            "episode_count": len(sealed_hashes),
        })
        source_prefix = f"source-rounds/{source_index:04d}-{round_id}-{seal_sha256}"
        for episode_id in sorted(sealed_hashes):
            if episode_id in metrics:
                raise ValueError("ordered success replay sources contain an episode ID collision")
            category, mean_reward, first_success, digest = _validate_accepted_episode(
                accepted_entries[episode_id], episode_id=episode_id,
            )
            if digest != sealed_hashes[episode_id]:
                raise ValueError("accepted success replay digest does not match round seal")
            receipt_path = receipt_entries[f"{episode_id}.sync.json"]
            _validate_receipt(
                receipt_path,
                episode_id=episode_id,
                episode_sha256=sealed_hashes[episode_id],
                immutable_revision=revisions[episode_id],
                seal=seal,
            )
            provenance_path = f"{source_prefix}/hf-sync-receipts/{episode_id}.sync.json"
            if provenance_path in logical_paths:
                raise ValueError("ordered success replay sources contain a provenance path collision")
            logical_paths.add(provenance_path)
            metrics[episode_id] = (category, mean_reward, first_success)
            receipt_paths[episode_id] = receipt_path
            provenance_paths[episode_id] = provenance_path
            source_episode_order.append(episode_id)
    if set(selected) != set(metrics):
        raise ValueError("runtime mixture rollout train episodes do not exactly cover sealed replay sources")

    scores = _category_scores({episode_id: metrics[episode_id] for episode_id in selected})
    evidence = {
        "schema_version": 1,
        "kind": "lehome_awr_progress_evidence",
        "mixture_id": manifest.mixture_id,
        "mixture_manifest_sha256": canonical_json_sha256(manifest.raw),
        "episodes": [
            {
                "episode_id": episode_id,
                "lineage_id": selected[episode_id],
                "split": "train",
                "score_kind": "progress",
                "score": scores[episode_id],
                "provenance_path": provenance_paths[episode_id],
                "provenance_sha256": sha256_file(receipt_paths[episode_id]),
            }
            for episode_id in source_episode_order
        ],
    }
    payload = canonical_json_bytes(evidence)
    digest = hashlib.sha256(payload).hexdigest()
    _write_absent_pair(
        parent,
        output=destination,
        payload=payload,
        sidecar=sidecar,
        sidecar_payload=(digest + "\n").encode("ascii"),
    )
    return {
        "evidence_path": str(destination),
        "evidence_sha256": digest,
        "mixture_id": manifest.mixture_id,
        "mixture_manifest_sha256": evidence["mixture_manifest_sha256"],
        "episode_count": len(evidence["episodes"]),
        "source_rounds": source_rounds,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixture-manifest", type=Path, required=True)
    repeated_help = "repeat once per source round; occurrence order is immutable lineage"
    parser.add_argument("--accepted-root", type=Path, action="append", required=True, help=repeated_help)
    parser.add_argument("--sync-receipts-root", type=Path, action="append", required=True, help=repeated_help)
    parser.add_argument("--round-seal", type=Path, action="append", required=True, help=repeated_help)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build_awr_progress_evidence(
        mixture_manifest=args.mixture_manifest,
        accepted_root=args.accepted_root,
        sync_receipts_root=args.sync_receipts_root,
        round_seal=args.round_seal,
        output=args.output,
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
