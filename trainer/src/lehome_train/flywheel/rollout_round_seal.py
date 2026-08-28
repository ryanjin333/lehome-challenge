"""Dependency-light seal for readback-verified rollout rounds."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable, Mapping

from lehome_train.constants import DEFAULT_ROLLOUT_REPO
from lehome_train.io import atomic_write_json, canonical_json_sha256


_ROUND_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_EPISODE_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_IMMUTABLE_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
MAX_ROLL_ROUND_EPISODES = 150


class RolloutRoundSealError(RuntimeError):
    """An immutable rollout round cannot be sealed from incomplete evidence."""


@dataclass(frozen=True, slots=True)
class RolloutRoundSeal:
    """The durable seal that permits calling a rollout round complete."""

    round_id: str
    episode_count: int
    repository: str
    episode_sha256s: Mapping[str, str]
    immutable_revisions: Mapping[str, str]
    seal_receipt_path: Path


def _load_sync_receipt(receipts_root: Path, attempt_id: str, round_id: str) -> dict[str, object]:
    if not attempt_id or "/" in attempt_id or attempt_id in {".", ".."}:
        raise RolloutRoundSealError(f"attempt_id is not path-safe: {attempt_id!r}")
    path = receipts_root / f"{attempt_id}.sync.json"
    if path.is_symlink() or not path.is_file():
        raise RolloutRoundSealError(f"sync receipt missing for accepted episode {attempt_id!r}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RolloutRoundSealError(f"sync receipt unreadable for {attempt_id!r}: {error}") from error
    if not isinstance(payload, dict):
        raise RolloutRoundSealError(f"sync receipt for {attempt_id!r} must be a JSON object")
    if payload.get("schema_version") != 1:
        raise RolloutRoundSealError(f"sync receipt schema_version for {attempt_id!r} must be 1")
    if payload.get("round_id") != round_id:
        raise RolloutRoundSealError(f"sync receipt round_id for {attempt_id!r} does not match the sealed round")
    if payload.get("readback_verified") is not True:
        raise RolloutRoundSealError(f"sync receipt readback for {attempt_id!r} is not verified")
    episode_sha256 = payload.get("episode_sha256")
    if not isinstance(episode_sha256, str) or not _EPISODE_SHA256_PATTERN.fullmatch(episode_sha256):
        raise RolloutRoundSealError(f"sync receipt episode_sha256 for {attempt_id!r} is invalid")
    if payload.get("attempt_id") != attempt_id:
        raise RolloutRoundSealError(f"sync receipt attempt_id for {attempt_id!r} does not match")
    if payload.get("repository") != DEFAULT_ROLLOUT_REPO:
        raise RolloutRoundSealError(f"sync receipt repository for {attempt_id!r} is not approved")
    immutable_revision = payload.get("immutable_revision")
    if not isinstance(immutable_revision, str) or not _IMMUTABLE_REVISION_PATTERN.fullmatch(immutable_revision):
        raise RolloutRoundSealError(f"sync receipt immutable revision for {attempt_id!r} is invalid")
    if payload.get("remote_prefix") != f"rollout-rounds/{round_id}/{attempt_id}":
        raise RolloutRoundSealError(f"sync receipt prefix for {attempt_id!r} does not match")
    return payload


def seal_rollout_round(
    *, receipts_root: str | Path, round_id: str, attempt_ids: Iterable[str],
    seal_receipt_path: str | Path,
) -> RolloutRoundSeal:
    """Seal only a bounded set of accepted, readback-proven episodes."""

    receipts_root_path = Path(receipts_root).resolve()
    if receipts_root_path.is_symlink() or not receipts_root_path.is_dir():
        raise RolloutRoundSealError("receipts root must be a real directory")
    if not isinstance(round_id, str) or not _ROUND_ID_PATTERN.fullmatch(round_id):
        raise RolloutRoundSealError(f"round_id must be path-safe lowercase, got {round_id!r}")
    ids = tuple(attempt_ids)
    if not ids:
        raise RolloutRoundSealError("a rollout round must seal at least one accepted episode")
    if len(ids) != len(set(ids)):
        raise RolloutRoundSealError("attempt_ids contains duplicates")
    if len(ids) > MAX_ROLL_ROUND_EPISODES:
        raise RolloutRoundSealError(f"a rollout round cannot exceed {MAX_ROLL_ROUND_EPISODES} accepted episodes")
    episode_sha256s: dict[str, str] = {}
    immutable_revisions: dict[str, str] = {}
    for attempt_id in ids:
        payload = _load_sync_receipt(receipts_root_path, attempt_id, round_id)
        episode_sha256s[attempt_id] = str(payload["episode_sha256"])
        immutable_revisions[attempt_id] = str(payload["immutable_revision"])
    seal_path = Path(seal_receipt_path).resolve()
    if seal_path.exists() or seal_path.is_symlink():
        raise RolloutRoundSealError("rollout round seal receipt already exists")
    seal_path.parent.mkdir(parents=True, exist_ok=True)
    seal_body = {
        "round_id": round_id, "repository": DEFAULT_ROLLOUT_REPO,
        "episode_sha256s": episode_sha256s, "immutable_revisions": immutable_revisions,
    }
    atomic_write_json(seal_path, {
        "schema_version": 2, "kind": "rollout_round_seal", "round_id": round_id,
        "repository": DEFAULT_ROLLOUT_REPO, "episode_count": len(ids),
        "episode_sha256s": episode_sha256s, "immutable_revisions": immutable_revisions,
        "readback_verified": True, "seal_sha256": canonical_json_sha256(seal_body),
    })
    return RolloutRoundSeal(
        round_id, len(ids), DEFAULT_ROLLOUT_REPO, episode_sha256s, immutable_revisions, seal_path,
    )
