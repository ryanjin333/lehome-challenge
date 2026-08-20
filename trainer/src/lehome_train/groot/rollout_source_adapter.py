"""Create one immutable runtime rollout source from sealed rollout rounds.

The rollout appliance publishes individual terminal episodes under a round
specific Hub prefix.  The runtime trainer, by contrast, mounts one canonical
``rollouts/round-N`` source.  This adapter is the narrow bridge between those
two contracts.  It never accepts an unsealed episode and it keeps a local
origin record for every preserved runtime episode identity.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from lehome_train.constants import DEFAULT_ROLLOUT_REPO
from lehome_train.io import canonical_json_bytes, canonical_json_sha256, sha256_file


_ROUND = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_CATEGORIES = frozenset({"top_long", "top_short", "pant_long", "pant_short"})
_MAX_SELECTED = 150
_HELD_OUT_GARMENTS = frozenset({
    "Top_Long_Unseen_1", "Top_Short_Unseen_1",
    "Pant_Long_Unseen_1", "Pant_Short_Unseen_1",
})


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _load(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is malformed") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _safe_relative(value: object, label: str) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise ValueError(f"{label} is unsafe")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} is unsafe")
    return value


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} is invalid")
    return value


def _regular_tree_hash(root: Path) -> str:
    """Hash a complete local input tree and reject symlinks/special files."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("rollout source root is missing or unsafe")
    rows: list[dict[str, object]] = []
    pending = [root]
    while pending:
        current = pending.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                path = Path(entry.path)
                relative = path.relative_to(root).as_posix()
                if entry.is_symlink():
                    raise ValueError("rollout source tree contains a symlink")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif entry.is_file(follow_symlinks=False):
                    rows.append({
                        "path": relative,
                        "sha256": sha256_file(path),
                        "byte_size": path.stat(follow_symlinks=False).st_size,
                    })
                else:
                    raise ValueError("rollout source tree contains an unsupported path")
    return canonical_json_sha256(sorted(rows, key=lambda row: str(row["path"])))


def _sync_digest(root: Path) -> str:
    """Match the appliance HubSyncDaemon's artifact digest exactly."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("sealed raw episode is missing or unsafe")
    entries: list[dict[str, object]] = []
    pending = [root]
    while pending:
        current = pending.pop()
        with os.scandir(current) as children:
            for child in children:
                path = Path(child.path)
                if child.is_symlink():
                    raise ValueError("sealed raw episode contains a symlink")
                if child.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif child.is_file(follow_symlinks=False):
                    relative = path.relative_to(root).as_posix()
                    if relative != "SHA256SUMS.json":
                        entries.append({
                            "relative_path": relative,
                            "sha256": sha256_file(path),
                            "byte_size": path.stat(follow_symlinks=False).st_size,
                        })
                else:
                    raise ValueError("sealed raw episode contains an unsupported path")
    if not entries:
        raise ValueError("sealed raw episode has no publishable entries")
    return hashlib.sha256(
        json.dumps(sorted(entries, key=lambda row: str(row["relative_path"])), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write(path: Path, value: object) -> None:
    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _copy_regular_tree(source: Path, destination: Path, *, exclude: set[str]) -> None:
    """Copy byte-for-byte without retaining mutable hardlinks to source rounds."""
    if source.is_symlink() or not source.is_dir():
        raise ValueError("sealed raw episode is missing or unsafe")
    destination.mkdir(parents=True)
    for current, directories, files in os.walk(source, followlinks=False):
        current_path = Path(current)
        for directory in directories:
            if (current_path / directory).is_symlink():
                raise ValueError("sealed raw episode contains a symlink")
        for filename in files:
            item = current_path / filename
            if item.is_symlink() or not item.is_file():
                raise ValueError("sealed raw episode contains an unsafe file")
            relative = item.relative_to(source).as_posix()
            if relative in exclude:
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item, target, follow_symlinks=False)
            if target.is_symlink() or sha256_file(target) != sha256_file(item):
                raise ValueError("canonical rollout copy hash mismatch")


def _seal(path: Path) -> dict[str, Any]:
    document = _load(path, "rollout round seal")
    required = {
        "schema_version", "kind", "round_id", "repository", "episode_count",
        "episode_sha256s", "immutable_revisions", "readback_verified", "seal_sha256",
    }
    if set(document) != required or document.get("schema_version") != 2 or document.get("kind") != "rollout_round_seal":
        raise ValueError("rollout round seal schema is incompatible")
    round_id = document.get("round_id")
    if type(round_id) is not str or _ROUND.fullmatch(round_id) is None:
        raise ValueError("rollout round seal ID is invalid")
    if document.get("repository") != DEFAULT_ROLLOUT_REPO or document.get("readback_verified") is not True:
        raise ValueError("rollout round seal lacks approved readback")
    digests, revisions = document.get("episode_sha256s"), document.get("immutable_revisions")
    if not isinstance(digests, dict) or not isinstance(revisions, dict) or set(digests) != set(revisions):
        raise ValueError("rollout round seal episode lineage is invalid")
    if type(document.get("episode_count")) is not int or document["episode_count"] != len(digests) or not 1 <= len(digests) <= _MAX_SELECTED:
        raise ValueError("rollout round seal episode count is outside the accepted cap")
    if any(
        not isinstance(attempt, str) or not attempt or "/" in attempt or "\\" in attempt or attempt in {".", ".."}
        or _SHA256.fullmatch(digest) is None or _REVISION.fullmatch(revisions[attempt]) is None
        for attempt, digest in digests.items()
    ):
        raise ValueError("rollout round seal episode identity is invalid")
    body = {
        "round_id": round_id, "repository": DEFAULT_ROLLOUT_REPO,
        "episode_sha256s": digests, "immutable_revisions": revisions,
    }
    if document.get("seal_sha256") != canonical_json_sha256(body):
        raise ValueError("rollout round seal hash mismatch")
    return document


def _sync_receipt(path: Path, *, round_id: str, attempt_id: str, digest: str, revision: str) -> dict[str, Any]:
    document = _load(path, "rollout sync receipt")
    required = {
        "schema_version", "attempt_id", "repository", "round_id", "remote_prefix",
        "publication_ref", "immutable_revision", "entry_count", "episode_sha256", "readback_verified",
    }
    if set(document) != required or document.get("schema_version") != 1:
        raise ValueError("rollout sync receipt schema is incompatible")
    if (
        document.get("attempt_id") != attempt_id
        or document.get("repository") != DEFAULT_ROLLOUT_REPO
        or document.get("round_id") != round_id
        or document.get("remote_prefix") != f"rollout-rounds/{round_id}/{attempt_id}"
        or document.get("episode_sha256") != digest
        or document.get("immutable_revision") != revision
        or document.get("readback_verified") is not True
        or type(document.get("publication_ref")) is not str
        or type(document.get("entry_count")) is not int
        or document["entry_count"] <= 0
    ):
        raise ValueError("rollout sync receipt does not prove immutable readback")
    return document


def _accepted_episode(
    *, root: Path, receipts_root: Path, seal: Mapping[str, Any], attempt_id: str,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    # The appliance seals and uploads the complete accepted episode package,
    # not only its nested training payload.  The accepted root therefore has
    # the live shape ``accepted/<attempt>/raw/<attempt>``.  Authenticate the
    # package against the Hub receipt/seal, then return only the raw subtree
    # that the runtime mixture consumes.
    package = root / attempt_id
    raw = package / "raw" / attempt_id
    if package.is_symlink() or not package.is_dir() or raw.is_symlink() or not raw.is_dir():
        raise ValueError("sealed episode raw root is missing or unsafe")
    try:
        from lehome.flywheel.artifacts import verify_episode_manifest
        from lehome_train.flywheel.materialize import _is_autonomous_policy_success
    except ImportError as error:
        raise RuntimeError("canonical rollout artifact verification is unavailable") from error
    try:
        episode, _ = verify_episode_manifest(raw)
    except ValueError as error:
        raise ValueError("sealed episode manifest verification failed") from error
    identity = episode.get("identity")
    if (
        episode.get("episode_id") != attempt_id
        or not isinstance(identity, Mapping)
        or identity.get("release_stage") != "seen"
        or identity.get("category") not in _CATEGORIES
        or type(identity.get("garment_name")) is not str
        or not identity["garment_name"]
        or not _is_autonomous_policy_success(episode)
    ):
        raise ValueError("sealed episode is not an accepted seen autonomous success")
    digest = str(seal["episode_sha256s"][attempt_id])
    revision = str(seal["immutable_revisions"][attempt_id])
    receipt_path = receipts_root / f"{attempt_id}.sync.json"
    receipt = _sync_receipt(
        receipt_path, round_id=str(seal["round_id"]), attempt_id=attempt_id,
        digest=digest, revision=revision,
    )
    if _sync_digest(package) != digest:
        raise ValueError("sealed episode digest does not match local accepted package")
    return dict(episode), raw, receipt


def _canonical_episode_id(round_id: str, attempt_id: str) -> str:
    """Preserve the sealed episode key used by AWR evidence and Hub receipts.

    Attempt IDs are globally unique schedule identifiers.  The caller rejects
    cross-round collisions instead of renaming them: a renamed key would sever
    the direct seal-to-AWR evidence binding.
    """
    del round_id
    return attempt_id


def _origin_sync_digest(
    root: Path,
    origin_manifest: Mapping[str, object],
    package_manifest: Mapping[str, object],
    *,
    attempt_id: str,
) -> str:
    """Recreate the appliance package digest without retaining duplicate videos.

    The top-level package checksum manifest enumerates every byte uploaded to
    the Hub, including success videos outside the nested training payload.  Its
    canonical rows are seal-bound by the resulting digest.  We retain that
    manifest and verify every raw member against the byte-for-byte copied raw
    tree; non-training package members remain authenticated by their seal-bound
    hashes without being copied into the runtime source.
    """
    if not origin_manifest or "episode.json" not in origin_manifest:
        raise ValueError("derived rollout origin manifest is incomplete")
    origin_entries: dict[str, tuple[str, int]] = {}
    for relative, record in origin_manifest.items():
        safe = _safe_relative(relative, "derived rollout origin manifest path")
        if safe == "SHA256SUMS.json" or not isinstance(record, Mapping):
            raise ValueError("derived rollout origin manifest is malformed")
        expected_sha, expected_size = _sha(record.get("sha256"), "derived rollout origin hash"), record.get("size")
        if type(expected_size) is not int or expected_size < 0 or set(record) != {"sha256", "size"}:
            raise ValueError("derived rollout origin manifest entry is malformed")
        target = root / ("origin-episode.json" if safe == "episode.json" else safe)
        if target.is_symlink() or not target.is_file() or target.stat().st_size != expected_size or sha256_file(target) != expected_sha:
            raise ValueError("derived rollout origin bytes drift")
        origin_entries[safe] = (expected_sha, expected_size)

    if not package_manifest or not {"flywheel-manifest.json", "worker-receipt.json"} <= set(package_manifest):
        raise ValueError("derived rollout package manifest is incomplete")
    prefix = f"raw/{attempt_id}/"
    expected_raw_paths = {prefix + "SHA256SUMS.json", *(prefix + relative for relative in origin_entries)}
    actual_raw_paths = {relative for relative in package_manifest if isinstance(relative, str) and relative.startswith(prefix)}
    if actual_raw_paths != expected_raw_paths:
        raise ValueError("derived rollout package manifest raw coverage is incomplete")

    entries: list[dict[str, object]] = []
    for relative, record in package_manifest.items():
        safe = _safe_relative(relative, "derived rollout package manifest path")
        if safe == "SHA256SUMS.json" or not isinstance(record, Mapping):
            raise ValueError("derived rollout package manifest is malformed")
        expected_sha = _sha(record.get("sha256"), "derived rollout package hash")
        expected_size = record.get("size")
        if type(expected_size) is not int or expected_size < 0 or set(record) != {"sha256", "size"}:
            raise ValueError("derived rollout package manifest entry is malformed")
        if safe.startswith(prefix):
            nested = safe[len(prefix):]
            target = root / (
                "origin-episode.json" if nested == "episode.json"
                else "origin-sha256s.json" if nested == "SHA256SUMS.json"
                else nested
            )
            if target.is_symlink() or not target.is_file() or target.stat().st_size != expected_size or sha256_file(target) != expected_sha:
                raise ValueError("derived rollout package raw bytes drift")
        entries.append({"relative_path": safe, "sha256": expected_sha, "byte_size": expected_size})
    return hashlib.sha256(
        json.dumps(sorted(entries, key=lambda row: str(row["relative_path"])), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_derived_rollout_source(
    root: str | Path, *, selected: Mapping[str, str], campaign_receipt: Mapping[str, object],
) -> None:
    """Reauthenticate an adapter output before runtime window authorization.

    This is intentionally callable by ``runtime_mixture_builder`` on every
    build.  A derived source must therefore retain its original seal/readback
    evidence, not merely claim that an earlier adapter invocation succeeded.
    """
    source = Path(root)
    lineage = _load(source / "source-lineage.json", "derived rollout source lineage")
    if set(lineage) != {"schema_version", "kind", "rounds", "episodes"} or lineage.get("schema_version") != 1 or lineage.get("kind") != "lehome_runtime_rollout_source_lineage" or not isinstance(lineage.get("rounds"), list) or not isinstance(lineage.get("episodes"), list):
        raise ValueError("derived rollout source lineage schema is incompatible")
    round_records: dict[str, Mapping[str, object]] = {}
    for record in lineage["rounds"]:
        if not isinstance(record, Mapping) or set(record) != {"round_id", "repository", "seal_path", "seal_sha256", "episode_count"}:
            raise ValueError("derived rollout source round lineage is malformed")
        round_id = record.get("round_id")
        if type(round_id) is not str or _ROUND.fullmatch(round_id) is None or round_id in round_records or record.get("repository") != DEFAULT_ROLLOUT_REPO or type(record.get("episode_count")) is not int:
            raise ValueError("derived rollout source round identity is invalid")
        seal_path = source / _safe_relative(record.get("seal_path"), "derived rollout source seal path")
        if sha256_file(seal_path) != _sha(record.get("seal_sha256"), "derived rollout source seal hash"):
            raise ValueError("derived rollout source seal hash drift")
        seal = _seal(seal_path)
        if seal["round_id"] != round_id or seal["episode_count"] != record["episode_count"]:
            raise ValueError("derived rollout source seal identity drift")
        round_records[round_id] = {**record, "seal": seal}
    if not round_records:
        raise ValueError("derived rollout source has no sealed source rounds")
    episodes: dict[str, Mapping[str, object]] = {}
    expected_fields = {
        "canonical_episode_id", "source_round_id", "source_attempt_id", "source_episode_sha256",
        "source_immutable_revision", "source_seal_path", "source_seal_sha256",
        "source_sync_receipt_path", "source_sync_receipt_sha256", "origin_episode_sha256",
        "origin_checksum_manifest_sha256", "origin_package_manifest_sha256",
    }
    for record in lineage["episodes"]:
        if not isinstance(record, Mapping) or set(record) != expected_fields:
            raise ValueError("derived rollout source episode lineage is malformed")
        canonical_id, round_id, attempt_id = record.get("canonical_episode_id"), record.get("source_round_id"), record.get("source_attempt_id")
        if (
            type(canonical_id) is not str or canonical_id not in selected or canonical_id in episodes
            or type(round_id) is not str or round_id not in round_records
            or type(attempt_id) is not str or not attempt_id
            or canonical_id != _canonical_episode_id(round_id, attempt_id)
        ):
            raise ValueError("derived rollout source episode identity is invalid")
        round_record = round_records[round_id]
        seal = round_record["seal"]
        if (
            seal["episode_sha256s"].get(attempt_id) != record.get("source_episode_sha256")
            or seal["immutable_revisions"].get(attempt_id) != record.get("source_immutable_revision")
            or record.get("source_seal_path") != round_record["seal_path"]
            or record.get("source_seal_sha256") != round_record["seal_sha256"]
        ):
            raise ValueError("derived rollout source episode seal lineage drift")
        receipt_path = source / _safe_relative(record.get("source_sync_receipt_path"), "derived rollout source sync receipt path")
        if sha256_file(receipt_path) != _sha(record.get("source_sync_receipt_sha256"), "derived rollout source sync receipt hash"):
            raise ValueError("derived rollout source sync receipt hash drift")
        _sync_receipt(
            receipt_path, round_id=round_id, attempt_id=attempt_id,
            digest=str(record["source_episode_sha256"]), revision=str(record["source_immutable_revision"]),
        )
        raw = source / "raw" / canonical_id
        if raw.is_symlink() or not raw.is_dir():
            raise ValueError("derived rollout raw root is missing or unsafe")
        if (
            sha256_file(raw / "origin-episode.json") != _sha(record.get("origin_episode_sha256"), "derived rollout origin episode hash")
            or sha256_file(raw / "origin-sha256s.json") != _sha(record.get("origin_checksum_manifest_sha256"), "derived rollout origin manifest hash")
            or sha256_file(raw / "origin-package-sha256s.json") != _sha(record.get("origin_package_manifest_sha256"), "derived rollout package manifest hash")
        ):
            raise ValueError("derived rollout origin control hash drift")
        origin_manifest = _load(raw / "origin-sha256s.json", "derived rollout origin checksum manifest")
        package_manifest = _load(raw / "origin-package-sha256s.json", "derived rollout package checksum manifest")
        if _origin_sync_digest(raw, origin_manifest, package_manifest, attempt_id=attempt_id) != record["source_episode_sha256"]:
            raise ValueError("derived rollout origin does not match sealed Hub digest")
        episodes[canonical_id] = record
    if set(episodes) != set(selected):
        raise ValueError("derived rollout source lineage does not exactly cover selected successes")


def _validated_garment_index(
    *, organizer_root: Path, organizer_manifest: Path, path: Path, expected_sha256: str,
) -> dict[str, str]:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != _sha(expected_sha256, "BC garment index hash"):
        raise ValueError("BC garment index is missing or hash-drifted")
    try:
        path.resolve(strict=True).relative_to(organizer_root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise ValueError("BC garment index escapes organizer root") from error
    document = _load(path, "BC garment index")
    if set(document) != {"schema_version", "kind", "episodes"} or document.get("schema_version") != 1 or document.get("kind") != "lehome_bc_garment_index" or not isinstance(document.get("episodes"), list) or canonical_json_bytes(document) != path.read_bytes():
        raise ValueError("BC garment index schema is incompatible")
    organizer = _load(organizer_manifest, "organizer manifest")
    expected = organizer.get("train_episode_ids", []) + organizer.get("validation_episode_ids", [])
    if not isinstance(expected, list) or any(type(item) is not str or not item for item in expected) or len(expected) != len(set(expected)):
        raise ValueError("organizer split ledger is malformed")
    result: dict[str, str] = {}
    for row in document["episodes"]:
        if not isinstance(row, Mapping) or set(row) != {"episode_id", "garment_name"}:
            raise ValueError("BC garment index row is malformed")
        episode_id, garment = row.get("episode_id"), row.get("garment_name")
        if type(episode_id) is not str or type(garment) is not str or not garment or episode_id in result:
            raise ValueError("BC garment index identity is malformed")
        if garment in _HELD_OUT_GARMENTS:
            raise ValueError("BC garment index selects a held-out evaluation garment")
        result[episode_id] = garment
    if set(result) != set(expected):
        raise ValueError("BC garment index does not exactly cover organizer splits")
    return result


def _h16_parquet_ranges(organizer_root: Path, episode_id: str) -> list[tuple[int, int]]:
    if not episode_id.isdigit():
        raise ValueError("BC episode ID is not compatible with the pinned parquet layout")
    episode = int(episode_id)
    path = organizer_root / f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet"
    if path.is_symlink() or not path.is_file():
        raise ValueError("BC parquet artifact is missing or unsafe")
    try:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        if not {"observation.state", "action"} <= set(parquet.schema_arrow.names):
            raise ValueError("BC parquet lacks required state/action columns")
        row_count = parquet.metadata.num_rows
        if parquet.num_row_groups < 1 or row_count < 16:
            return []
        sample = parquet.read_row_group(0, columns=["observation.state", "action"]).slice(0, 1)
        state, action = sample["observation.state"].to_pylist()[0], sample["action"].to_pylist()[0]
    except ImportError as error:
        raise RuntimeError("runtime plan generation requires the pinned parquet runtime") from error
    for value, label in ((state, "BC state"), (action, "BC action")):
        if not isinstance(value, list) or len(value) != 12 or any(type(item) not in (int, float) or not math.isfinite(float(item)) for item in value):
            raise ValueError(f"{label} metadata is not a finite 12D vector")
    return [(start, start + 16) for start in range(0, row_count - 15, 16)]


def _h16_rollout_ranges(raw: Path) -> list[tuple[int, int]]:
    annotations = raw / "annotations.jsonl"
    if annotations.is_symlink() or not annotations.is_file():
        raise ValueError("rollout annotations are missing or unsafe")
    try:
        lines = annotations.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError("rollout annotations are unavailable") from error
    for line in lines:
        try:
            row = json.loads(line, object_pairs_hook=_pairs)
        except json.JSONDecodeError as error:
            raise ValueError("rollout annotation is malformed") from error
        if not isinstance(row, Mapping):
            raise ValueError("rollout annotation is malformed")
        for field in ("state", "action"):
            vector = row.get(field)
            if not isinstance(vector, list) or len(vector) != 12 or any(type(item) not in (int, float) or not math.isfinite(float(item)) for item in vector):
                raise ValueError("rollout annotation lacks finite 12D state/action metadata")
    return [(start, start + 16) for start in range(0, len(lines) - 15, 16)]


def build_runtime_plan(
    *, organizer_root: str | Path, campaign_root: str | Path,
    garment_index_path: str | Path, garment_index_sha256: str,
    experiment_config_path: str | Path, destination: str | Path,
) -> dict[str, object]:
    """Generate the deterministic h16 plan state consumed by the runtime builder.

    Runtime scheduling, not this plan, enforces the batch-64 45/19 source
    cadence.  The plan keeps every available non-overlapping accepted rollout
    h16 range so AWR evidence can cover the complete sealed set, and keeps BC
    train and validation episode lineages separate.
    """
    organizer, campaign, output = Path(organizer_root), Path(campaign_root), Path(destination)
    if (
        organizer.is_symlink() or not organizer.is_dir() or campaign.is_symlink() or not campaign.is_dir()
        or not output.is_absolute() or output.exists() or output.is_symlink() or output.parent.is_symlink() or not output.parent.is_dir()
    ):
        raise ValueError("runtime plan roots or immutable destination are unsafe")
    organizer_manifest, campaign_receipt = organizer / "manifest.json", campaign / "campaign-receipt.json"
    selected_path, lineage_path = campaign / "selected-150.json", campaign / "source-lineage.json"
    config = Path(experiment_config_path)
    from lehome_train.groot.experiment_manifest import load_runtime_profile

    weights, _ = load_runtime_profile(config)
    garments = _validated_garment_index(
        organizer_root=organizer, organizer_manifest=organizer_manifest,
        path=Path(garment_index_path), expected_sha256=garment_index_sha256,
    )
    # Reuse the exact runtime admission checks rather than accepting a lookalike
    # selected ledger or a stale adapter directory.
    from lehome_train.groot.runtime_mixture_builder import (
        validate_selected_bindings, validate_selected_raw_roots,
    )

    receipt = _load(campaign_receipt, "campaign receipt")
    selected = validate_selected_bindings(_load(selected_path, "selected rollout bindings"), receipt)
    validate_derived_rollout_source(campaign, selected=selected, campaign_receipt=receipt)
    validate_selected_raw_roots(campaign, selected, receipt, held_out_garments=tuple(_HELD_OUT_GARMENTS))
    organizer_value = _load(organizer_manifest, "organizer manifest")
    train_ids, validation_ids = organizer_value.get("train_episode_ids"), organizer_value.get("validation_episode_ids")
    if not isinstance(train_ids, list) or not isinstance(validation_ids, list) or any(type(item) is not str for item in train_ids + validation_ids) or set(train_ids) & set(validation_ids):
        raise ValueError("organizer train/validation split is malformed")
    selections: list[dict[str, object]] = []
    organizer_hash, campaign_hash = sha256_file(organizer_manifest), sha256_file(campaign_receipt)
    for split, ids in (("train", sorted(train_ids)), ("validation", sorted(validation_ids))):
        for episode_id in ids:
            if episode_id not in garments:
                raise ValueError("BC selection lacks a garment attestation")
            for start, stop in _h16_parquet_ranges(organizer, episode_id):
                selections.append({
                    "source_kind": "organizer", "source_manifest_sha256": organizer_hash,
                    "source_episode_id": episode_id, "raw_episode_id": episode_id,
                    "frame_start": start, "frame_stop": stop,
                    "raw_frame_start": start, "raw_frame_stop": stop,
                    "raw_frame_ids": [str(frame) for frame in range(start, stop)], "split": split,
                })
    for episode_id in sorted(selected):
        raw = campaign / "raw" / episode_id
        for start, stop in _h16_rollout_ranges(raw):
            selections.append({
                "source_kind": "flywheel", "source_manifest_sha256": campaign_hash,
                "source_episode_id": episode_id, "raw_episode_id": episode_id,
                "raw_frame_start": start, "raw_frame_stop": stop,
                "raw_frame_ids": [str(frame) for frame in range(start, stop)], "split": "train",
            })
    if not any(row["source_kind"] == "organizer" and row["split"] == "train" for row in selections):
        raise ValueError("BC train split has no complete h16 state/action range")
    if not any(row["source_kind"] == "organizer" and row["split"] == "validation" for row in selections):
        raise ValueError("BC validation split has no complete h16 state/action range")
    if not any(row["source_kind"] == "flywheel" for row in selections):
        raise ValueError("sealed rollout source has no complete h16 state/action range")
    from lehome_train.groot.runtime_mixture import source_tree_sha256

    bindings = {
        "organizer_manifest_sha256": organizer_hash,
        "organizer_tree_sha256": source_tree_sha256(organizer),
        "campaign_receipt_sha256": campaign_hash,
        "campaign_tree_sha256": source_tree_sha256(campaign),
        "selected_bindings_sha256": sha256_file(selected_path),
        "source_lineage_sha256": sha256_file(lineage_path),
        "garment_index_sha256": sha256_file(Path(garment_index_path)),
        "experiment_config_sha256": sha256_file(config),
        "runtime_schedule": {
            "bc": weights["bc"], "rollout": weights["rollout"],
            "batch_size": 64, "action_horizon": 16,
        },
    }
    plan: dict[str, object] = {
        "schema_version": 1, "kind": "runtime_mixture_plan",
        "input_bindings": bindings, "selected_frame_ranges": selections,
    }
    plan["sha256"] = canonical_json_sha256(plan)
    state = {
        "schema_version": 1, "kind": "runtime_mixture_plan_state",
        "plan": plan, "plan_sha256": plan["sha256"],
    }
    _write(output, state)
    return {
        "schema_version": 1, "kind": "runtime_mixture_plan_receipt",
        "plan_path": str(output), "plan_sha256": str(plan["sha256"]),
        "selection_count": len(selections), "input_bindings": bindings,
    }


def build_rollout_source(
    *, rounds: Sequence[Mapping[str, object]], destination: str | Path, runtime_round: int,
) -> dict[str, object]:
    """Derive one immutable rollout source from 1..N sealed rounds.

    Each round ``root`` is its appliance ``accepted`` directory, whose children
    are complete seal-bound episode packages.  The destination is
    intentionally absent-only.  It contains the canonical raw artifacts
    consumed by runtime training plus copied seals and receipts that bind every
    generated episode back to its original Hub readback.
    """
    if not isinstance(rounds, Sequence) or isinstance(rounds, (str, bytes)) or not 1 <= len(rounds) <= 16:
        raise ValueError("rollout source requires one to sixteen sealed rounds")
    if type(runtime_round) is not int or runtime_round < 1:
        raise ValueError("runtime rollout round must be a positive integer")
    output = Path(destination)
    if not output.is_absolute() or output.exists() or output.is_symlink() or output.parent.is_symlink() or not output.parent.is_dir():
        raise FileExistsError("rollout source destination must be an absent path below a real directory")

    sources: list[dict[str, Any]] = []
    for descriptor in rounds:
        if not isinstance(descriptor, Mapping) or set(descriptor) != {"root", "receipts_root", "seal_path"}:
            raise ValueError("rollout source round descriptor is malformed")
        values = {key: descriptor[key] for key in descriptor}
        if any(type(value) is not str or not value or not Path(value).is_absolute() for value in values.values()):
            raise ValueError("rollout source round paths must be absolute")
        root, receipts, seal_path = (Path(values[key]) for key in ("root", "receipts_root", "seal_path"))
        if root.is_symlink() or not root.is_dir() or receipts.is_symlink() or not receipts.is_dir():
            raise ValueError("rollout source round root is missing or unsafe")
        seal = _seal(seal_path)
        sources.append({
            "root": root, "receipts": receipts, "seal_path": seal_path, "seal": seal,
            "root_identity": _regular_tree_hash(root),
            "receipts_identity": _regular_tree_hash(receipts),
            "seal_identity": sha256_file(seal_path),
        })
    if len({str(source["seal"]["round_id"]) for source in sources}) != len(sources):
        raise ValueError("rollout source rounds contain duplicate round IDs")

    rows: list[dict[str, Any]] = []
    for source in sources:
        seal = source["seal"]
        for attempt_id in sorted(seal["episode_sha256s"]):
            episode, raw, sync = _accepted_episode(
                root=source["root"], receipts_root=source["receipts"], seal=seal, attempt_id=attempt_id,
            )
            canonical_id = _canonical_episode_id(str(seal["round_id"]), attempt_id)
            rows.append({
                "canonical_id": canonical_id, "attempt_id": attempt_id, "episode": episode,
                "raw": raw, "sync": sync, "source": source,
            })
    if not 1 <= len(rows) <= _MAX_SELECTED or len({row["canonical_id"] for row in rows}) != len(rows):
        raise ValueError("combined rollout source count is outside cap or has a cross-round episode-ID collision")
    rows.sort(key=lambda row: str(row["canonical_id"]))

    staging = output.parent / f".{output.name}.adapter.tmp"
    if staging.exists() or staging.is_symlink():
        raise FileExistsError("rollout source staging already exists")
    try:
        staging.mkdir()
        selected_rows: list[dict[str, str]] = []
        lineage_episodes: list[dict[str, object]] = []
        copied_rounds: list[dict[str, object]] = []
        for source in sorted(sources, key=lambda item: str(item["seal"]["round_id"])):
            seal = source["seal"]
            round_id = str(seal["round_id"])
            seal_target = staging / "source-seals" / f"{round_id}.json"
            seal_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source["seal_path"], seal_target)
            if sha256_file(seal_target) != source["seal_identity"]:
                raise ValueError("copied rollout round seal hash mismatch")
            copied_rounds.append({
                "round_id": round_id, "repository": DEFAULT_ROLLOUT_REPO,
                "seal_path": seal_target.relative_to(staging).as_posix(),
                "seal_sha256": source["seal_identity"], "episode_count": seal["episode_count"],
            })
        for row in rows:
            canonical_id, raw, episode, sync = row["canonical_id"], row["raw"], row["episode"], row["sync"]
            target = staging / "raw" / canonical_id
            _copy_regular_tree(raw, target, exclude={"episode.json", "SHA256SUMS.json"})
            # Retain the exact origin control bytes.  These sidecars allow a
            # later verifier to reconstruct and authenticate the original Hub
            # digest without changing the seal-bound episode identity.
            shutil.copyfile(raw / "episode.json", target / "origin-episode.json")
            shutil.copyfile(raw / "SHA256SUMS.json", target / "origin-sha256s.json")
            package_manifest = raw.parent.parent / "SHA256SUMS.json"
            if package_manifest.is_symlink() or not package_manifest.is_file():
                raise ValueError("sealed accepted package checksum manifest is missing or unsafe")
            shutil.copyfile(package_manifest, target / "origin-package-sha256s.json")
            canonical_episode = dict(episode)
            canonical_episode["episode_id"] = canonical_id
            _write(target / "episode.json", canonical_episode)
            try:
                from lehome.flywheel.artifacts import build_sha256_manifest
            except ImportError as error:
                raise RuntimeError("canonical rollout artifact writer is unavailable") from error
            _write(target / "SHA256SUMS.json", build_sha256_manifest(target))
            selected_rows.append({
                "attempt_id": canonical_id, "episode_id": canonical_id,
                "episode_manifest_sha256": sha256_file(target / "SHA256SUMS.json"),
            })
            source = row["source"]
            seal = source["seal"]
            receipt_target = staging / "source-sync-receipts" / f"{canonical_id}.json"
            source_receipt = source["receipts"] / f"{row['attempt_id']}.sync.json"
            receipt_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_receipt, receipt_target)
            if sha256_file(receipt_target) != sha256_file(source_receipt):
                raise ValueError("copied rollout sync receipt hash mismatch")
            lineage_episodes.append({
                "canonical_episode_id": canonical_id,
                "source_round_id": seal["round_id"], "source_attempt_id": row["attempt_id"],
                "source_episode_sha256": seal["episode_sha256s"][row["attempt_id"]],
                "source_immutable_revision": seal["immutable_revisions"][row["attempt_id"]],
                "source_seal_path": f"source-seals/{seal['round_id']}.json",
                "source_seal_sha256": source["seal_identity"],
                "source_sync_receipt_path": receipt_target.relative_to(staging).as_posix(),
                "source_sync_receipt_sha256": sha256_file(receipt_target),
                "origin_episode_sha256": sha256_file(raw / "episode.json"),
                "origin_checksum_manifest_sha256": sha256_file(raw / "SHA256SUMS.json"),
                "origin_package_manifest_sha256": sha256_file(package_manifest),
            })
        selected_rows.sort(key=lambda item: item["attempt_id"])
        selected = {
            "schema_version": 2, "selected_count": len(selected_rows), "max_selected_count": _MAX_SELECTED,
            "selected_bindings": selected_rows,
        }
        selected["selection_sha256"] = canonical_json_sha256(selected)
        _write(staging / "selected-150.json", selected)
        campaign = {"schema_version": 2, "kind": "runtime_rollout_source_campaign", "attempt_receipts": [
            {
                "attempt_id": row["attempt_id"], "episode_id": row["episode_id"],
                "category": next(item["episode"]["identity"]["category"] for item in rows if item["canonical_id"] == row["attempt_id"]),
                "accepted_success": True, "release_stage": "seen", "outcome": "success",
            }
            for row in selected_rows
        ]}
        _write(staging / "campaign-receipt.json", campaign)
        lineage = {
            "schema_version": 1, "kind": "lehome_runtime_rollout_source_lineage",
            "rounds": copied_rounds, "episodes": sorted(lineage_episodes, key=lambda item: str(item["canonical_episode_id"])),
        }
        _write(staging / "source-lineage.json", lineage)
        for source in sources:
            if (
                _regular_tree_hash(source["root"]) != source["root_identity"]
                or _regular_tree_hash(source["receipts"]) != source["receipts_identity"]
                or sha256_file(source["seal_path"]) != source["seal_identity"]
            ):
                raise ValueError("sealed rollout source mutated during adapter build")
        os.replace(staging, output)
        return {
            "schema_version": 1, "kind": "runtime_rollout_source_adapter_receipt",
            "destination": str(output), "selected_count": len(selected_rows),
            "runtime_round": runtime_round, "runtime_prefix": f"rollouts/round-{runtime_round}",
            "selected_bindings_sha256": sha256_file(output / "selected-150.json"),
            "campaign_receipt_sha256": sha256_file(output / "campaign-receipt.json"),
            "source_lineage_sha256": sha256_file(output / "source-lineage.json"),
        }
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_from_request(path: str | Path) -> dict[str, object]:
    """Run the adapter from a strict, shared-disk friendly JSON request."""
    request = _load(Path(path), "rollout source adapter request")
    if set(request) != {"schema_version", "command", "arguments"} or request.get("schema_version") != 1 or request.get("command") != "build-rollout-source" or not isinstance(request.get("arguments"), dict):
        raise ValueError("rollout source adapter request has an incompatible schema")
    arguments = request["arguments"]
    if set(arguments) != {"rounds", "destination", "runtime_round"} or not isinstance(arguments["rounds"], list) or type(arguments["destination"]) is not str or type(arguments["runtime_round"]) is not int:
        raise ValueError("rollout source adapter request arguments are incomplete or unknown")
    return build_rollout_source(
        rounds=arguments["rounds"], destination=arguments["destination"],
        runtime_round=arguments["runtime_round"],
    )


def build_runtime_plan_from_request(path: str | Path) -> dict[str, object]:
    """Run plan generation from one strict shared-disk request envelope."""
    request = _load(Path(path), "runtime plan request")
    if set(request) != {"schema_version", "command", "arguments"} or request.get("schema_version") != 1 or request.get("command") != "build-runtime-plan" or not isinstance(request.get("arguments"), dict):
        raise ValueError("runtime plan request has an incompatible schema")
    arguments = request["arguments"]
    expected = {
        "organizer_root", "campaign_root", "garment_index_path", "garment_index_sha256",
        "experiment_config_path", "destination",
    }
    if set(arguments) != expected or any(type(arguments[key]) is not str or not arguments[key] for key in expected):
        raise ValueError("runtime plan request arguments are incomplete or unknown")
    return build_runtime_plan(**arguments)
