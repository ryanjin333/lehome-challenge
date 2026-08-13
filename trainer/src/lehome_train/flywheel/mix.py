"""Deterministically freeze and materialize safe flywheel training mixtures.

The unit of selection is a concrete contiguous 16-frame source range.  This is
deliberately more restrictive than selecting episode IDs: a v2 loader may form
an action horizon only inside one emitted range, never across a filtered raw
segment or an oversampling boundary.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from lehome_train.data.convert import (
    LEGACY_DATA_PATH,
    LEGACY_VIDEO_PATH,
    _modality_metadata,
)
from lehome_train.data.mapping import ACTION_HORIZON, FIXED_INSTRUCTION, JOINT_NAMES
from lehome_train.data.inspect import artifact_identities
from lehome_train.data.split import split_episode_ids
from lehome_train.io import atomic_write_json, canonical_json_bytes, canonical_json_sha256, sha256_file


GRADE_WEIGHTS = {"A": 1.0, "B": 0.5}
SOURCE_WEIGHTS = {"organizer": 0.7, "flywheel": 0.3}
_SPLIT_FRACTION = 0.1
_CAMERAS = ("top_rgb", "left_rgb", "right_rgb")
_DEFAULT_VIDEO_WORKERS = 4
_MAX_VIDEO_WORKERS = 8


@dataclass(frozen=True, slots=True)
class FrameSelection:
    """One physically copyable source range and its frozen output identity."""

    source_kind: str
    source_manifest_sha256: str
    source_episode_id: str
    frame_start: int
    frame_stop: int
    source_frame_ids: tuple[str, ...]
    raw_manifest_sha256: str
    raw_episode_id: str
    raw_frame_start: int
    raw_frame_stop: int
    raw_frame_ids: tuple[str, ...]
    destination_episode_id: str
    split: str
    quality_grade: str | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "source_kind": self.source_kind,
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_episode_id": self.source_episode_id,
            "frame_start": self.frame_start,
            "frame_stop": self.frame_stop,
            "source_frame_ids": list(self.source_frame_ids),
            "raw_manifest_sha256": self.raw_manifest_sha256,
            "raw_episode_id": self.raw_episode_id,
            "raw_frame_start": self.raw_frame_start,
            "raw_frame_stop": self.raw_frame_stop,
            "raw_frame_ids": list(self.raw_frame_ids),
            "destination_episode_id": self.destination_episode_id,
            "split": self.split,
        }
        if self.quality_grade is not None:
            result["quality_grade"] = self.quality_grade
        return result


@dataclass(frozen=True, slots=True)
class MixPlan:
    seed: int
    split_seed: int
    validation_fraction: float
    organizer_training_frames: int
    flywheel_training_frames: int
    source_weights: dict[str, float]
    grade_weights: dict[str, float]
    selections: tuple[FrameSelection, ...]
    source_revisions: dict[str, str]
    raw_manifest_hashes: tuple[str, ...]
    rejected_by_reason: dict[str, int]
    sha256: str

    @property
    def organizer_episode_ids(self) -> tuple[str, ...]:
        return tuple(
            item.source_episode_id
            for item in self.selections
            if item.split == "train" and item.source_kind == "organizer"
        )

    @property
    def flywheel_episode_ids(self) -> tuple[str, ...]:
        return tuple(
            item.source_episode_id
            for item in self.selections
            if item.split == "train" and item.source_kind == "flywheel"
        )

    def body(self) -> dict[str, object]:
        return {
            "schema_version": 3,
            "seed": self.seed,
            "split_seed": self.split_seed,
            "validation_fraction": self.validation_fraction,
            "organizer_training_frames": self.organizer_training_frames,
            "flywheel_training_frames": self.flywheel_training_frames,
            "source_weights": dict(sorted(self.source_weights.items())),
            "grade_weights": dict(sorted(self.grade_weights.items())),
            "selected_frame_ranges": [item.to_dict() for item in self.selections],
            "source_revisions": dict(sorted(self.source_revisions.items())),
            "raw_manifest_hashes": list(self.raw_manifest_hashes),
            "rejected_by_reason": dict(sorted(self.rejected_by_reason.items())),
        }

    def to_dict(self) -> dict[str, object]:
        return self.body() | {"sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class _Chunk:
    source_kind: str
    source_root: Path
    source_manifest_sha256: str
    source_revision: str
    episode_id: str
    start: int
    stop: int
    frame_ids: tuple[str, ...]
    raw_manifest_sha256: str
    raw_episode_id: str
    raw_frame_start: int
    raw_frame_stop: int
    raw_frame_ids: tuple[str, ...]
    quality_grade: str | None


@dataclass(frozen=True, slots=True)
class _PreparedSource:
    root: Path
    kind: str
    manifest: Mapping[str, Any]
    manifest_sha256: str
    source_revision: str
    quality_grade: str | None
    raw_manifest_sha256: str | None
    raw_lineage_by_episode: Mapping[str, tuple[str, int, int, tuple[str, ...]]]
    rejection_counts: Mapping[str, int]


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("mix JSON contains duplicate fields")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"mix metadata unavailable: {path}") from error
    if not isinstance(value, dict):
        raise ValueError("mix metadata must be an object")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _revision(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be an immutable revision")
    return value


def _verify_artifacts(root: Path, manifest: Mapping[str, Any]) -> None:
    artifacts = manifest.get("output_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("prepared mix source has no output artifact manifest")
    if manifest.get("output_manifest_sha256") != canonical_json_sha256(artifacts):
        raise ValueError("prepared mix source output manifest hash is invalid")
    seen: set[str] = set()
    for item in artifacts:
        if not isinstance(item, Mapping) or set(item) != {"relative_path", "sha256", "byte_size"}:
            raise ValueError("prepared mix source artifact schema is invalid")
        relative = item["relative_path"]
        digest = item["sha256"]
        size = item["byte_size"]
        if not isinstance(relative, str) or not relative or relative.startswith("/") or ".." in Path(relative).parts:
            raise ValueError("prepared mix source artifact path is invalid")
        _sha256(digest, "prepared mix source artifact hash")
        if type(size) is not int or size < 0:
            raise ValueError("prepared mix source artifact size is invalid")
        path = root / relative
        if relative in seen or path.is_symlink() or not path.is_file():
            raise ValueError("prepared mix source artifact is missing or duplicated")
        if path.stat().st_size != size or sha256_file(path) != digest:
            raise ValueError("prepared mix source artifact hash mismatch")
        seen.add(relative)


def _materialized_provenance(
    root: Path,
    *,
    materialized_episode_ids: Sequence[str],
) -> tuple[str, str, Mapping[str, int], Mapping[str, tuple[str, int, int, tuple[str, ...]]]]:
    provenance = _read_json(root / "meta" / "materialization-provenance.json")
    if provenance.get("raw_manifest_verified") is not True:
        raise ValueError("flywheel source raw artifact was not checksum verified")
    if provenance.get("quality_grade") not in GRADE_WEIGHTS:
        raise ValueError("flywheel source grade must be A or B")
    if provenance.get("accepted_success") is not True or provenance.get("trainable") is not True or provenance.get("outcome") != "success":
        raise ValueError("failed flywheel source cannot enter mix")
    identity = provenance.get("raw_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("flywheel source has no raw identity")
    if identity.get("release_stage") == "public_unseen":
        raise ValueError("evaluation holdout cannot enter mix")
    if identity.get("instruction") != FIXED_INSTRUCTION:
        raise ValueError("flywheel source has an incompatible task instruction")
    if provenance.get("selection_horizon") != ACTION_HORIZON:
        raise ValueError("flywheel source action horizon is incompatible")
    ranges = provenance.get("selected_frame_ranges")
    if not isinstance(ranges, list) or not ranges:
        raise ValueError("flywheel source lacks immutable expert frame ranges")
    if len(ranges) != len(materialized_episode_ids):
        raise ValueError("flywheel provenance must map one raw range to every materialized episode")
    raw_hash = _sha256(provenance.get("raw_manifest_sha256"), "raw manifest hash")
    lineage_by_episode: dict[str, tuple[str, int, int, tuple[str, ...]]] = {}
    for episode_id, item in zip(sorted(materialized_episode_ids, key=int), ranges, strict=True):
        if not isinstance(item, Mapping) or item.get("action_source") != "expert":
            raise ValueError("flywheel source contains non-expert targets")
        raw_episode_id = item.get("raw_episode_id")
        start, stop = item.get("frame_start"), item.get("frame_stop")
        if not isinstance(raw_episode_id, str) or not raw_episode_id or type(start) is not int or type(stop) is not int or start < 0 or stop - start != ACTION_HORIZON:
            raise ValueError("flywheel provenance range must be an integer action-horizon raw range")
        lineage_by_episode[episode_id] = (
            raw_episode_id,
            start,
            stop,
            tuple(str(frame_id) for frame_id in range(start, stop)),
        )
    revision = _revision(identity.get("code_revision"), "flywheel code revision")
    counts = provenance.get("rejected_by_reason")
    if not isinstance(counts, Mapping) or any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError("flywheel source rejection counts are invalid")
    return raw_hash, revision, counts, lineage_by_episode  # type: ignore[return-value]


def _rft_snapshot_provenance(
    root: Path,
    *,
    manifest: Mapping[str, Any],
    materialized_episode_ids: Sequence[str],
) -> tuple[str, str, Mapping[str, int], Mapping[str, tuple[str, int, int, tuple[str, ...]]], str]:
    """Adapt the canonical aggregate autonomous-RFT snapshot to mix lineage."""
    if manifest.get("source_format") != "verified_flywheel_rft_release":
        raise ValueError("flywheel source lacks a supported materialization contract")
    selection = _read_json(root / "meta" / "rft-selection.json")
    repository, revision, release_id = (
        selection.get("source_repository"),
        selection.get("source_revision"),
        selection.get("release_id"),
    )
    if (
        repository != manifest.get("source_repository")
        or revision != manifest.get("source_revision")
        or release_id != manifest.get("source_release_id")
    ):
        raise ValueError("RFT selection identity does not match snapshot manifest")
    revision = _revision(revision, "RFT source revision")
    _sha256(release_id, "RFT source release ID")
    if selection.get("action_horizon") != ACTION_HORIZON:
        raise ValueError("RFT selection action horizon is incompatible")
    excluded = (selection.get("excluded_public_unseen"), selection.get("excluded_failed"))
    if any(type(value) is not int or value < 0 for value in excluded):
        raise ValueError("RFT selection exclusion counts are invalid")
    episodes = selection.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("RFT selection has no accepted autonomous episodes")
    lineage: dict[str, tuple[str, int, int, tuple[str, ...]]] = {}
    hashes: list[str] = []
    categories: list[str] = []
    by_index: dict[str, Mapping[str, Any]] = {}
    for item in episodes:
        if not isinstance(item, Mapping) or type(item.get("episode_index")) is not int:
            raise ValueError("RFT selection episode is malformed")
        episode_id = str(item["episode_index"])
        if episode_id in by_index:
            raise ValueError("RFT selection episode is duplicated")
        raw_id, raw_hash, frame_count, category = (
            item.get("raw_episode_id"), item.get("raw_manifest_sha256"),
            item.get("frame_count"), item.get("category", "unknown"),
        )
        if not isinstance(raw_id, str) or not raw_id or type(frame_count) is not int or frame_count < ACTION_HORIZON:
            raise ValueError("RFT selection episode has invalid policy trajectory identity")
        _sha256(raw_hash, "RFT raw manifest hash")
        if not isinstance(category, str) or not category:
            raise ValueError("RFT selection category is invalid")
        by_index[episode_id] = item
        hashes.append(raw_hash)
        categories.append(category)
    for episode_id in materialized_episode_ids:
        item = by_index.get(episode_id)
        if item is None:
            raise ValueError("RFT selection does not bind every materialized episode")
        frame_count = item["frame_count"]
        assert isinstance(frame_count, int)
        lineage[episode_id] = (
            str(item["raw_episode_id"]), 0, frame_count,
            tuple(str(index) for index in range(frame_count)),
        )
    # A/B are DAgger-only quality labels. Aggregate RFT remains policy-only;
    # equal treatment preserves source ratio without inventing grades.
    counts = {"rft_policy_success": len(materialized_episode_ids)}
    raw_binding = canonical_json_sha256({"release_id": release_id, "raw_manifest_hashes": sorted(hashes)})
    return raw_binding, revision, counts, lineage, "A"


def _prepared_source(root_value: str | Path, *, kind: str) -> _PreparedSource:
    root = Path(root_value)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("prepared mix source must be a real directory")
    manifest = _read_json(root / "manifest.json")
    if manifest.get("output_format") != "groot_lerobot_v2.1_per_episode":
        raise ValueError("mix inputs must be canonical prepared-v2 datasets")
    if manifest.get("fixed_language_instruction") != FIXED_INSTRUCTION:
        raise ValueError("prepared mix source has an incompatible instruction")
    if manifest.get("future_actions", {}).get("horizon") != ACTION_HORIZON:
        raise ValueError("prepared mix source has an incompatible action horizon")
    _verify_artifacts(root, manifest)
    manifest_sha = sha256_file(root / "manifest.json")
    episode_ids = manifest.get("train_episode_ids")
    if not isinstance(episode_ids, list) or not episode_ids or not all(isinstance(value, str) for value in episode_ids) or len(set(episode_ids)) != len(episode_ids):
        raise ValueError("prepared mix source has no unique train episodes")
    if kind == "organizer":
        return _PreparedSource(
            root=root,
            kind=kind,
            manifest=manifest,
            manifest_sha256=manifest_sha,
            source_revision=_revision(manifest.get("source_revision"), "organizer source revision"),
            quality_grade=None,
            raw_manifest_sha256=None,
            raw_lineage_by_episode={},
            rejection_counts={},
        )
    if manifest.get("source_format") == "verified_flywheel_rft_release":
        raw_hash, revision, counts, lineage_by_episode, grade = _rft_snapshot_provenance(
            root, manifest=manifest, materialized_episode_ids=episode_ids
        )
    else:
        raw_hash, revision, counts, lineage_by_episode = _materialized_provenance(root, materialized_episode_ids=episode_ids)
        grade = str(_read_json(root / "meta" / "materialization-provenance.json")["quality_grade"])
    return _PreparedSource(
        root=root,
        kind=kind,
        manifest=manifest,
        manifest_sha256=manifest_sha,
        source_revision=revision,
        quality_grade=grade,
        raw_manifest_sha256=raw_hash,
        raw_lineage_by_episode=lineage_by_episode,
        rejection_counts=counts,
    )


def _source_chunks(source: _PreparedSource) -> list[_Chunk]:
    ids = source.manifest.get("train_episode_ids")
    if not isinstance(ids, list) or not ids or not all(isinstance(value, str) for value in ids):
        raise ValueError("prepared mix source has no train episodes")
    if len(set(ids)) != len(ids):
        raise ValueError("prepared mix source train episodes are duplicated")
    info = _read_json(source.root / "meta" / "info.json")
    pattern, chunks_size = info.get("data_path"), info.get("chunks_size")
    if not isinstance(pattern, str) or type(chunks_size) is not int or chunks_size <= 0:
        raise ValueError("prepared mix source has invalid v2 metadata")
    chunks: list[_Chunk] = []
    for episode_id in sorted(ids, key=int):
        numeric = int(episode_id)
        path = source.root / pattern.format(episode_chunk=numeric // chunks_size, episode_index=numeric)
        if path.is_symlink() or not path.is_file():
            raise ValueError("prepared mix source is missing selected parquet")
        table = pq.read_table(path, columns=["observation.state", "action", "frame_index", "episode_index", "index"])
        if table.num_rows < ACTION_HORIZON:
            continue
        states, actions = table["observation.state"].to_pylist(), table["action"].to_pylist()
        frames = table["frame_index"].to_pylist()
        episodes = table["episode_index"].to_pylist()
        global_ids = table["index"].to_pylist()
        if any(len(value) != 12 for value in states + actions):
            raise ValueError("prepared mix source has non-12D vectors")
        if frames != list(range(table.num_rows)) or episodes != [numeric] * table.num_rows:
            raise ValueError("prepared mix source does not have canonical frame boundaries")
        if len(set(global_ids)) != len(global_ids):
            raise ValueError("prepared mix source has duplicate global frame IDs")
        for start in range(0, table.num_rows - ACTION_HORIZON + 1, ACTION_HORIZON):
            stop = start + ACTION_HORIZON
            if source.kind == "organizer":
                raw_manifest_sha256 = source.manifest_sha256
                raw_episode_id = episode_id
                raw_frame_start = frames[start]
                raw_frame_stop = frames[stop - 1] + 1
                raw_frame_ids = tuple(str(value) for value in frames[start:stop])
            else:
                try:
                    raw_episode_id, lineage_start, lineage_stop, lineage_ids = source.raw_lineage_by_episode[episode_id]
                except KeyError:
                    raise ValueError("flywheel provenance is missing a materialized episode lineage") from None
                raw_manifest_sha256 = source.raw_manifest_sha256
                if raw_manifest_sha256 is None:
                    raise ValueError("flywheel source is missing raw manifest lineage")
                if lineage_stop - lineage_start != len(lineage_ids) or stop > len(lineage_ids):
                    raise ValueError("flywheel provenance does not cover selected policy frames")
                raw_frame_start = lineage_start + start
                raw_frame_stop = lineage_start + stop
                raw_frame_ids = lineage_ids[start:stop]
            chunks.append(
                _Chunk(
                    source_kind=source.kind,
                    source_root=source.root,
                    source_manifest_sha256=source.manifest_sha256,
                    source_revision=source.source_revision,
                    episode_id=episode_id,
                    start=start,
                    stop=stop,
                    frame_ids=tuple(str(value) for value in global_ids[start:stop]),
                    raw_manifest_sha256=raw_manifest_sha256,
                    raw_episode_id=raw_episode_id,
                    raw_frame_start=raw_frame_start,
                    raw_frame_stop=raw_frame_stop,
                    raw_frame_ids=raw_frame_ids,
                    quality_grade=source.quality_grade,
                )
            )
    if not chunks:
        raise ValueError("prepared mix source has no complete action-horizon ranges")
    return chunks


def _cycle(items: Sequence[_Chunk], count: int, *, seed: int) -> list[_Chunk]:
    ordered = list(items)
    random.Random(seed).shuffle(ordered)
    return [ordered[index % len(ordered)] for index in range(count)]


def _weighted_flywheel_cycle(items: Sequence[_Chunk], count: int, *, seed: int) -> list[_Chunk]:
    weighted: list[_Chunk] = []
    for item in items:
        weighted.extend([item] * (2 if item.quality_grade == "A" else 1))
    return _cycle(weighted, count, seed=seed)


def _source_frame_keys(item: _Chunk | FrameSelection) -> set[tuple[str, str, str]]:
    """Return the immutable identity of every selected source frame."""

    if isinstance(item, _Chunk):
        episode_id, frame_ids = item.episode_id, item.frame_ids
    else:
        episode_id, frame_ids = item.source_episode_id, item.source_frame_ids
    return {
        (item.source_manifest_sha256, episode_id, frame_id)
        for frame_id in frame_ids
    }


def _raw_episode_key(item: _Chunk | FrameSelection) -> tuple[str, str]:
    return item.raw_manifest_sha256, item.raw_episode_id


def _raw_frame_keys(item: _Chunk | FrameSelection) -> set[tuple[str, str, str]]:
    return {
        (item.raw_manifest_sha256, item.raw_episode_id, frame_id)
        for frame_id in item.raw_frame_ids
    }


def _reserve_validation_chunks(items: Sequence[_Chunk], count: int, *, seed: int) -> list[_Chunk]:
    """Reserve whole immutable lineage episodes while leaving both kinds trainable."""

    if count <= 0:
        return []
    grouped: dict[tuple[str, str], list[_Chunk]] = {}
    for item in items:
        grouped.setdefault(_raw_episode_key(item), []).append(item)
    groups = list(grouped.values())
    if any(len({item.source_kind for item in group}) != 1 for group in groups):
        raise ValueError("one immutable lineage episode cannot span organizer and flywheel sources")
    groups_by_kind = {kind: sum(group[0].source_kind == kind for group in groups) for kind in SOURCE_WEIGHTS}
    if any(groups_by_kind[kind] < 1 for kind in SOURCE_WEIGHTS):
        raise ValueError("mix has no lineage episode for a required training kind")
    random.Random(seed).shuffle(groups)
    # Dynamic programming picks an exact validation-slot total without consuming
    # the final lineage episode of either required training kind.  A range group
    # is indivisible: all overlapping Task-1 windows inherit one split.
    states: dict[tuple[int, int, int], tuple[int, ...]] = {(0, 0, 0): ()}
    for index, group in enumerate(groups):
        kind = group[0].source_kind
        size = len(group)
        for (frames, organizer_groups, flywheel_groups), selected in tuple(states.items()):
            selected_by_kind = organizer_groups if kind == "organizer" else flywheel_groups
            if frames + size > count or selected_by_kind >= groups_by_kind[kind] - 1:
                continue
            next_key = (
                frames + size,
                organizer_groups + (kind == "organizer"),
                flywheel_groups + (kind == "flywheel"),
            )
            if next_key not in states:
                states[next_key] = selected + (index,)
    matches = [selected for (frames, _, _), selected in states.items() if frames == count]
    if not matches:
        raise ValueError("mix has too few distinct lineage episodes for an unsplit validation holdout")
    return [item for index in matches[0] for item in groups[index]]


def _require_cross_split_source_frame_disjointness(selections: Sequence[FrameSelection]) -> None:
    """Reject plans whose train and validation source frames intersect."""

    source_frames = {"train": set(), "validation": set()}
    for item in selections:
        item_frames = _source_frame_keys(item)
        if len(item_frames) != ACTION_HORIZON:
            raise ValueError("flywheel mix plan source range has duplicate frame IDs")
        source_frames[item.split].update(item_frames)
    if source_frames["train"] & source_frames["validation"]:
        raise ValueError("flywheel mix plan train and validation source frames overlap")


def _require_cross_split_raw_lineage_isolation(selections: Sequence[FrameSelection]) -> None:
    """Reject plans that split a raw episode or reuse one raw frame across splits."""

    raw_frames = {"train": set(), "validation": set()}
    raw_episodes = {"train": set(), "validation": set()}
    for item in selections:
        item_frames = _raw_frame_keys(item)
        if len(item_frames) != ACTION_HORIZON:
            raise ValueError("flywheel mix plan raw lineage range has duplicate frame IDs")
        raw_frames[item.split].update(item_frames)
        raw_episodes[item.split].add(_raw_episode_key(item))
    if raw_frames["train"] & raw_frames["validation"]:
        raise ValueError("flywheel mix plan train and validation raw frames overlap")
    if raw_episodes["train"] & raw_episodes["validation"]:
        raise ValueError("flywheel mix plan splits one immutable raw lineage episode across train and validation")


def _total_with_exact_train_slots(train_slots: int, *, split_seed: int) -> tuple[int, set[str]]:
    total = train_slots
    while True:
        split = split_episode_ids(
            tuple(str(index) for index in range(total)),
            seed=split_seed,
            validation_fraction=_SPLIT_FRACTION,
        )
        if len(split.train) == train_slots and split.validation:
            return total, set(split.train)
        total += 1


def _plan_from_payload(payload: Mapping[str, Any]) -> MixPlan:
    if set(payload) != {
        "schema_version", "seed", "split_seed", "validation_fraction",
        "organizer_training_frames", "flywheel_training_frames", "source_weights",
        "grade_weights", "selected_frame_ranges", "source_revisions", "raw_manifest_hashes",
        "rejected_by_reason", "sha256",
    }:
        raise ValueError("flywheel mix plan schema is invalid")
    if payload["schema_version"] != 3 or type(payload["seed"]) is not int or type(payload["split_seed"]) is not int:
        raise ValueError("flywheel mix plan identity is invalid")
    if payload["validation_fraction"] != _SPLIT_FRACTION:
        raise ValueError("flywheel mix plan validation fraction is invalid")
    if payload["source_weights"] != SOURCE_WEIGHTS or payload["grade_weights"] != GRADE_WEIGHTS:
        raise ValueError("flywheel mix plan weights are invalid")
    selections_value = payload["selected_frame_ranges"]
    if not isinstance(selections_value, list) or not selections_value:
        raise ValueError("flywheel mix plan has no selected frame ranges")
    selections: list[FrameSelection] = []
    destinations: set[str] = set()
    for item in selections_value:
        if not isinstance(item, Mapping):
            raise ValueError("flywheel mix plan range is invalid")
        required = {
            "source_kind", "source_manifest_sha256", "source_episode_id", "frame_start", "frame_stop", "source_frame_ids",
            "raw_manifest_sha256", "raw_episode_id", "raw_frame_start", "raw_frame_stop", "raw_frame_ids",
            "destination_episode_id", "split",
        }
        if set(item) not in (required, required | {"quality_grade"}):
            raise ValueError("flywheel mix plan range schema is invalid")
        if item.get("source_kind") not in SOURCE_WEIGHTS or item.get("split") not in {"train", "validation"}:
            raise ValueError("flywheel mix plan range role is invalid")
        start, stop = item.get("frame_start"), item.get("frame_stop")
        frame_ids = item.get("source_frame_ids")
        raw_start, raw_stop = item.get("raw_frame_start"), item.get("raw_frame_stop")
        raw_frame_ids = item.get("raw_frame_ids")
        destination = item.get("destination_episode_id")
        if type(start) is not int or type(stop) is not int or stop - start != ACTION_HORIZON or not isinstance(frame_ids, list) or len(frame_ids) != ACTION_HORIZON or not all(isinstance(value, str) for value in frame_ids) or not isinstance(destination, str) or destination in destinations:
            raise ValueError("flywheel mix plan range is not an action-horizon boundary")
        if type(raw_start) is not int or type(raw_stop) is not int or raw_start < 0 or raw_stop - raw_start != ACTION_HORIZON or not isinstance(raw_frame_ids, list) or raw_frame_ids != [str(frame_id) for frame_id in range(raw_start, raw_stop)]:
            raise ValueError("flywheel mix plan raw lineage is not an action-horizon boundary")
        raw_episode_id = item.get("raw_episode_id")
        if not isinstance(raw_episode_id, str) or not raw_episode_id:
            raise ValueError("flywheel mix plan raw lineage episode is invalid")
        grade = item.get("quality_grade")
        if item["source_kind"] == "flywheel" and grade not in GRADE_WEIGHTS:
            raise ValueError("flywheel mix plan range has an invalid grade")
        if item["source_kind"] == "organizer" and grade is not None:
            raise ValueError("organizer mix range must not have a grade")
        destinations.add(destination)
        selections.append(FrameSelection(
            str(item["source_kind"]), _sha256(item["source_manifest_sha256"], "mix source manifest hash"), str(item["source_episode_id"]),
            start, stop, tuple(frame_ids), _sha256(item["raw_manifest_sha256"], "mix raw manifest hash"), raw_episode_id,
            raw_start, raw_stop, tuple(raw_frame_ids), destination, str(item["split"]), str(grade) if grade is not None else None,
        ))
    source_revisions = payload["source_revisions"]
    raw_hashes = payload["raw_manifest_hashes"]
    rejected = payload["rejected_by_reason"]
    if not isinstance(source_revisions, Mapping) or not all(
        isinstance(key, str) for key in source_revisions
    ):
        raise ValueError("flywheel mix plan source revisions are invalid")
    for value in source_revisions.values():
        _revision(value, "mix source revision")
    if not isinstance(raw_hashes, list):
        raise ValueError("flywheel mix plan raw manifest hashes are invalid")
    for value in raw_hashes:
        _sha256(value, "mix raw manifest hash")
    if not isinstance(rejected, Mapping) or any(not isinstance(key, str) or type(value) is not int or value < 0 for key, value in rejected.items()):
        raise ValueError("flywheel mix plan rejection counts are invalid")
    plan = MixPlan(
        seed=payload["seed"], split_seed=payload["split_seed"], validation_fraction=payload["validation_fraction"],
        organizer_training_frames=payload["organizer_training_frames"], flywheel_training_frames=payload["flywheel_training_frames"],
        source_weights=dict(payload["source_weights"]), grade_weights=dict(payload["grade_weights"]),
        selections=tuple(selections), source_revisions=dict(source_revisions), raw_manifest_hashes=tuple(raw_hashes),
        rejected_by_reason=dict(rejected), sha256=_sha256(payload["sha256"], "mix plan hash"),
    )
    if any(type(value) is not int or value < 0 for value in (plan.organizer_training_frames, plan.flywheel_training_frames)):
        raise ValueError("flywheel mix plan training frame counts are invalid")
    return plan


def validate_mix_plan_payload(payload: Mapping[str, Any]) -> MixPlan:
    """Decode and authenticate a JSON plan before any statistics are read."""

    plan = _plan_from_payload(payload)
    verify_mix_plan(plan)
    train = [item for item in plan.selections if item.split == "train"]
    validation = [item for item in plan.selections if item.split == "validation"]
    if not train or not validation:
        raise ValueError("flywheel mix plan must preserve train and validation ranges")
    organizer = sum(ACTION_HORIZON for item in train if item.source_kind == "organizer")
    flywheel = sum(ACTION_HORIZON for item in train if item.source_kind == "flywheel")
    if (organizer, flywheel) != (plan.organizer_training_frames, plan.flywheel_training_frames):
        raise ValueError("flywheel mix plan training frame counts differ from selections")
    if organizer * 3 != flywheel * 7:
        raise ValueError("flywheel mix plan is not an exact 70/30 training-frame mixture")
    _require_cross_split_source_frame_disjointness(plan.selections)
    _require_cross_split_raw_lineage_isolation(plan.selections)
    return plan


def build_mix_plan(
    organizer: str | Path,
    flywheel: str | Path | Sequence[str | Path],
    *,
    seed: int,
    organizer_fraction: float = 0.70,
    split_seed: int | None = None,
) -> MixPlan:
    """Freeze copyable source ranges with exact post-split 70/30 train frames."""

    if type(seed) is not int or organizer_fraction != SOURCE_WEIGHTS["organizer"]:
        raise ValueError("mix requires organizer fraction 0.70")
    if split_seed is None:
        split_seed = seed
    if type(split_seed) is not int:
        raise ValueError("mix split seed must be an integer")
    flywheel_roots = [flywheel] if isinstance(flywheel, (str, Path)) else list(flywheel)
    if not flywheel_roots:
        raise ValueError("mix requires at least one flywheel materialized-v2 source")
    organizer_source = _prepared_source(organizer, kind="organizer")
    flywheel_sources = [_prepared_source(root, kind="flywheel") for root in flywheel_roots]
    if len({source.manifest_sha256 for source in flywheel_sources}) != len(flywheel_sources):
        raise ValueError("mix flywheel sources must have distinct immutable manifests")
    organizer_chunks = _source_chunks(organizer_source)
    flywheel_chunks = [chunk for source in flywheel_sources for chunk in _source_chunks(source)]
    required_slots = max(
        math.ceil(len(organizer_chunks) / SOURCE_WEIGHTS["organizer"]),
        math.ceil(len(flywheel_chunks) / SOURCE_WEIGHTS["flywheel"]),
    )
    train_slots = math.ceil(required_slots / 10) * 10
    organizer_slots = train_slots * 7 // 10
    flywheel_slots = train_slots - organizer_slots
    total_slots, train_ids = _total_with_exact_train_slots(train_slots, split_seed=split_seed)
    validation_chunks = _reserve_validation_chunks(
        organizer_chunks + flywheel_chunks,
        total_slots - train_slots,
        seed=seed ^ 0xA55A,
    )
    reserved_raw_episodes = {_raw_episode_key(item) for item in validation_chunks}
    training_organizer_chunks = [
        item for item in organizer_chunks
        if _raw_episode_key(item) not in reserved_raw_episodes
    ]
    training_flywheel_chunks = [
        item for item in flywheel_chunks
        if _raw_episode_key(item) not in reserved_raw_episodes
    ]
    train_organizer = iter(_cycle(training_organizer_chunks, organizer_slots, seed=seed))
    train_flywheel = iter(_weighted_flywheel_cycle(training_flywheel_chunks, flywheel_slots, seed=seed ^ 0x5A17))
    validation = iter(validation_chunks)
    selections: list[FrameSelection] = []
    for index in range(total_slots):
        destination = str(index)
        if destination in train_ids:
            # Stable alternation means neither source order nor filesystem order
            # biases the train split; the terminal counts remain exact.
            train_position = sum(item.split == "train" for item in selections)
            chunk = next(train_organizer) if train_position * 3 % 10 < 7 else next(train_flywheel)
            split = "train"
        else:
            chunk, split = next(validation), "validation"
        selections.append(FrameSelection(
            chunk.source_kind, chunk.source_manifest_sha256, chunk.episode_id, chunk.start, chunk.stop, chunk.frame_ids,
            chunk.raw_manifest_sha256, chunk.raw_episode_id, chunk.raw_frame_start, chunk.raw_frame_stop, chunk.raw_frame_ids,
            destination, split, chunk.quality_grade,
        ))
    source_revisions = {
        f"{source.kind}:{source.manifest_sha256}": source.source_revision
        for source in (organizer_source, *flywheel_sources)
    }
    raw_hashes = tuple(sorted(source.raw_manifest_sha256 for source in flywheel_sources if source.raw_manifest_sha256 is not None))
    rejected: dict[str, int] = {}
    for source in flywheel_sources:
        for reason, count in source.rejection_counts.items():
            rejected[reason] = rejected.get(reason, 0) + count
    provisional = MixPlan(seed, split_seed, _SPLIT_FRACTION, organizer_slots * ACTION_HORIZON, flywheel_slots * ACTION_HORIZON, dict(SOURCE_WEIGHTS), dict(GRADE_WEIGHTS), tuple(selections), source_revisions, raw_hashes, rejected, "")
    plan = MixPlan(
        seed, split_seed, _SPLIT_FRACTION, organizer_slots * ACTION_HORIZON,
        flywheel_slots * ACTION_HORIZON, dict(SOURCE_WEIGHTS), dict(GRADE_WEIGHTS),
        tuple(selections), source_revisions, raw_hashes, rejected,
        canonical_json_sha256(provisional.body()),
    )
    validate_mix_plan_payload(plan.to_dict())
    return plan


def verify_mix_plan(plan: MixPlan) -> None:
    if canonical_json_sha256(plan.body()) != plan.sha256:
        raise ValueError("frozen mix plan hash is invalid")


def _copy_selected_video(source: Path, destination: Path, *, start: int, stop: int) -> None:
    from lehome_train.flywheel.materialize import _copy_selected_video as copy_video

    copy_video(source, destination, steps=list(range(start, stop)))


class _BoundedVideoSlicer:
    """Run independently-addressed video slices with a bounded work queue."""

    def __init__(self, workers: int) -> None:
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mix-video")
        self._workers = workers
        self._pending: dict[Future[None], int] = {}
        self._next_index = 0
        self._closed = False

    def submit(self, source: Path, destination: Path, *, start: int, stop: int) -> None:
        self._require_open()
        if len(self._pending) == self._workers:
            self._wait_for_batch()
        future = self._executor.submit(
            _copy_selected_video, source, destination, start=start, stop=stop,
        )
        self._pending[future] = self._next_index
        self._next_index += 1

    def finish(self) -> None:
        self._require_open()
        while self._pending:
            self._wait_for_batch()
        self._executor.shutdown(wait=True)
        self._closed = True

    def cancel(self) -> None:
        if self._closed:
            return
        for future in self._pending:
            future.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._pending.clear()
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("bounded video slicer is already closed")

    def _wait_for_batch(self) -> None:
        completed, _ = wait(self._pending, return_when=FIRST_COMPLETED)
        failures: list[tuple[int, BaseException]] = []
        for future in sorted(completed, key=self._pending.__getitem__):
            index = self._pending[future]
            try:
                future.result()
            except BaseException as error:
                failures.append((index, error))
        if failures:
            for future in self._pending:
                if future not in completed:
                    future.cancel()
            self._executor.shutdown(wait=True, cancel_futures=True)
            for future in sorted(self._pending, key=self._pending.__getitem__):
                if future in completed or future.cancelled():
                    continue
                try:
                    future.result()
                except BaseException as error:
                    failures.append((self._pending[future], error))
            self._pending.clear()
            self._closed = True
            raise min(failures, key=lambda item: item[0])[1]
        wait(self._pending)
        for future in sorted(self._pending, key=self._pending.__getitem__):
            index = self._pending.pop(future)
            try:
                future.result()
            except BaseException as error:
                failures.append((index, error))
        if failures:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._closed = True
            raise min(failures, key=lambda item: item[0])[1]


def _source_by_hash(sources: Iterable[_PreparedSource]) -> dict[str, _PreparedSource]:
    ordered = tuple(sources)
    result = {source.manifest_sha256: source for source in ordered}
    if len(result) != len(ordered):
        raise ValueError("mix sources have duplicate immutable manifest hashes")
    return result


def _expected_raw_lineage(
    source: _PreparedSource,
    selection: FrameSelection,
) -> tuple[str, str, int, int, tuple[str, ...]]:
    if source.kind == "organizer":
        start, stop = selection.frame_start, selection.frame_stop
        return source.manifest_sha256, selection.source_episode_id, start, stop, tuple(str(frame_id) for frame_id in range(start, stop))
    try:
        raw_episode_id, start, stop, frame_ids = source.raw_lineage_by_episode[selection.source_episode_id]
    except KeyError:
        raise ValueError("mix source provenance is missing the planned materialized episode") from None
    if source.raw_manifest_sha256 is None:
        raise ValueError("mix source provenance is missing the planned raw manifest")
    return source.raw_manifest_sha256, raw_episode_id, start, stop, frame_ids


def _write_lines(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


_GENERATION_RECEIPT_SUFFIX = ".generation.json"


def _generation_receipt_path(root: Path) -> Path:
    return root.with_name(root.name + _GENERATION_RECEIPT_SUFFIX)


def _generation_receipt(root: Path) -> dict[str, object]:
    manifest = _read_json(root / "manifest.json")
    plan = validate_mix_plan_payload(manifest.get("flywheel_mix_plan", {}))
    statistics = manifest.get("statistics")
    if not isinstance(statistics, Mapping) or not isinstance(statistics.get("files"), list):
        raise ValueError("sealed generation statistics are invalid")
    receipt: dict[str, object] = {
        "schema_version": 1,
        "sealed": True,
        "source_revisions": dict(sorted(plan.source_revisions.items())),
        "mix_plan_sha256": plan.sha256,
        "organizer_training_frames": plan.organizer_training_frames,
        "rft_training_frames": plan.flywheel_training_frames,
        "split_seed": plan.split_seed,
        "raw_manifest_hashes": list(plan.raw_manifest_hashes),
        "dataset_manifest_sha256": sha256_file(root / "manifest.json"),
        "output_manifest_sha256": manifest.get("output_manifest_sha256"),
        "statistics_sha256": canonical_json_sha256(statistics),
    }
    source_evidence = manifest.get("persistent_source_evidence")
    if source_evidence is not None:
        if not isinstance(source_evidence, Mapping):
            raise ValueError("sealed generation persistent source evidence is invalid")
        receipt["persistent_source_evidence"] = dict(source_evidence)
    return receipt


def load_generation_receipt(root_value: str | Path) -> dict[str, object]:
    """Load the sibling immutable receipt for one materialized generation."""

    root = Path(root_value)
    receipt = _read_json(_generation_receipt_path(root))
    expected = _generation_receipt(root)
    if receipt != expected:
        raise ValueError("sealed generation receipt is invalid")
    return receipt


def verify_generation(root_value: str | Path) -> dict[str, object]:
    """Rehash a sealed generation and reject changed or extra dataset files."""

    root = Path(root_value)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("sealed generation root is unavailable")
    receipt = load_generation_receipt(root)
    manifest = _read_json(root / "manifest.json")
    artifacts = manifest.get("output_artifacts")
    if not isinstance(artifacts, list) or manifest.get("output_manifest_sha256") != canonical_json_sha256(artifacts):
        raise ValueError("sealed generation artifact manifest is invalid")
    listed = {item.get("relative_path") for item in artifacts if isinstance(item, Mapping)}
    # ``manifest.json`` owns the artifact list and is deliberately not a member
    # of it; including it would make the manifest self-referential.
    actual = {
        item["relative_path"]
        for item in artifact_identities(root, exclude={"manifest.json"})
    }
    if listed != actual:
        raise ValueError("sealed generation files changed after seal")
    _verify_artifacts(root, manifest)
    if receipt["sealed"] is not True:
        raise ValueError("sealed generation receipt is not sealed")
    return receipt


def materialize_mixed_snapshot(
    plan: MixPlan,
    organizer: str | Path,
    flywheel: str | Path | Sequence[str | Path],
    destination: str | Path,
    *,
    persistent_source_evidence: Mapping[str, object] | None = None,
    video_workers: int = _DEFAULT_VIDEO_WORKERS,
) -> dict[str, object]:
    """Atomically copy a frozen plan into one canonical prepared-v2 snapshot."""

    if type(video_workers) is not int or not 1 <= video_workers <= _MAX_VIDEO_WORKERS:
        raise ValueError(f"video_workers must be an integer from 1 to {_MAX_VIDEO_WORKERS}")
    verify_mix_plan(plan)
    validate_mix_plan_payload(plan.to_dict())
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("refusing to overwrite mixed snapshot destination")
    flywheel_roots = [flywheel] if isinstance(flywheel, (str, Path)) else list(flywheel)
    organizer_source = _prepared_source(organizer, kind="organizer")
    flywheel_sources = [_prepared_source(root, kind="flywheel") for root in flywheel_roots]
    sources = _source_by_hash((organizer_source, *flywheel_sources))
    expected_revisions = {
        f"{source.kind}:{source.manifest_sha256}": source.source_revision
        for source in (organizer_source, *flywheel_sources)
    }
    if plan.source_revisions != expected_revisions:
        raise ValueError("mix plan source revisions no longer match materialized inputs")
    for selection in plan.selections:
        source = sources.get(selection.source_manifest_sha256)
        if source is None or source.kind != selection.source_kind:
            raise ValueError("mix plan references an unavailable source manifest")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent))
    video_slicer = _BoundedVideoSlicer(video_workers)
    promoted = False
    try:
        global_index = 0
        episode_rows: list[dict[str, object]] = []
        for selection in sorted(plan.selections, key=lambda item: int(item.destination_episode_id)):
            source = sources[selection.source_manifest_sha256]
            if (selection.raw_manifest_sha256, selection.raw_episode_id, selection.raw_frame_start, selection.raw_frame_stop, selection.raw_frame_ids) != _expected_raw_lineage(source, selection):
                raise ValueError("mix plan raw lineage no longer matches materialized inputs")
            source_info = _read_json(source.root / "meta" / "info.json")
            source_pattern = source_info["data_path"]
            source_chunk_size = source_info["chunks_size"]
            numeric_source = int(selection.source_episode_id)
            source_data = source.root / str(source_pattern).format(episode_chunk=numeric_source // int(source_chunk_size), episode_index=numeric_source)
            table = pq.read_table(source_data).slice(selection.frame_start, ACTION_HORIZON)
            if table.num_rows != ACTION_HORIZON or tuple(str(value) for value in table["index"].to_pylist()) != selection.source_frame_ids:
                raise ValueError("mix source range changed after plan freeze")
            numeric_destination = int(selection.destination_episode_id)
            output_data = temporary / LEGACY_DATA_PATH.format(episode_chunk=numeric_destination // 1000, episode_index=numeric_destination)
            output_data.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table({
                "observation.state": table["observation.state"],
                "action": table["action"],
                "timestamp": pa.array([index / 30 for index in range(ACTION_HORIZON)], type=pa.float32()),
                "frame_index": pa.array(range(ACTION_HORIZON), type=pa.int64()),
                "episode_index": pa.array([numeric_destination] * ACTION_HORIZON, type=pa.int64()),
                "index": pa.array(range(global_index, global_index + ACTION_HORIZON), type=pa.int64()),
                "task_index": pa.array([0] * ACTION_HORIZON, type=pa.int64()),
            }), output_data, compression="zstd")
            for camera in _CAMERAS:
                source_video = source.root / LEGACY_VIDEO_PATH.format(episode_chunk=numeric_source // int(source_chunk_size), episode_index=numeric_source, video_key=camera)
                output_video = temporary / LEGACY_VIDEO_PATH.format(episode_chunk=numeric_destination // 1000, episode_index=numeric_destination, video_key=camera)
                video_slicer.submit(source_video, output_video, start=selection.frame_start, stop=selection.frame_stop)
            episode_rows.append({"episode_index": numeric_destination, "length": ACTION_HORIZON, "task_index": 0, "tasks": [FIXED_INSTRUCTION]})
            global_index += ACTION_HORIZON
        video_slicer.finish()
        meta = temporary / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        info = {
            "codebase_version": "v2.1", "robot_type": "dual_so101_follower", "total_episodes": len(episode_rows),
            "total_frames": global_index, "total_tasks": 1, "total_videos": len(episode_rows) * len(_CAMERAS),
            "total_chunks": math.ceil(len(episode_rows) / 1000), "chunks_size": 1000, "fps": 30,
            "data_path": LEGACY_DATA_PATH, "video_path": LEGACY_VIDEO_PATH,
            "features": {camera: {"dtype": "video", "shape": [480, 640, 3], "info": {"video.fps": 30}} for camera in _CAMERAS},
        }
        atomic_write_json(meta / "info.json", info)
        _write_lines(meta / "episodes.jsonl", episode_rows)
        _write_lines(meta / "episodes_stats.jsonl", ({"episode_index": row["episode_index"], "stats": {}} for row in episode_rows))
        _write_lines(meta / "tasks.jsonl", [{"task_index": 0, "task": FIXED_INSTRUCTION}])
        atomic_write_json(meta / "modality.json", _modality_metadata())
        atomic_write_json(meta / "mix-selection.json", plan.to_dict())
        split = split_episode_ids(tuple(str(row["episode_index"]) for row in episode_rows), seed=plan.split_seed, validation_fraction=plan.validation_fraction)
        planned_train = tuple(item.destination_episode_id for item in plan.selections if item.split == "train")
        planned_validation = tuple(item.destination_episode_id for item in plan.selections if item.split == "validation")
        if split.train != planned_train or split.validation != planned_validation:
            raise ValueError("mix plan split assignments differ from deterministic split")
        output_artifacts = artifact_identities(temporary)
        manifest: dict[str, object] = {
            "schema_version": 2, "source_format": "prepared-v2-frame-range-mix", "output_format": "groot_lerobot_v2.1_per_episode",
            "source_revisions": plan.source_revisions, "raw_manifest_hashes": list(plan.raw_manifest_hashes),
            "output_artifacts": output_artifacts, "output_manifest_sha256": canonical_json_sha256(output_artifacts),
            "fps": 30, "frame_count": global_index, "episode_count": len(episode_rows), "split_seed": plan.split_seed,
            "validation_fraction": plan.validation_fraction, "train_episode_ids": list(split.train), "validation_episode_ids": list(split.validation),
            "camera_schema": [{"source_key": f"observation.images.{camera}", "dtype": "video", "shape": [480, 640, 3]} for camera in _CAMERAS],
            "state_schema": {"source_key": "observation.state", "dimension": 12, "names": list(JOINT_NAMES)},
            "action_schema": {"source_key": "action", "dimension": 12, "names": list(JOINT_NAMES), "storage": "absolute"},
            "fixed_language_instruction": FIXED_INSTRUCTION,
            "future_actions": {"horizon": ACTION_HORIZON, "loader_allow_padding": False, "materialized_windows": True, "tail_convention": "one_complete_source_range_per_episode", "valid_window_counts": {str(row["episode_index"]): 1 for row in episode_rows}},
            "flywheel_mix_plan": plan.to_dict(), "statistics": {"status": "pending_final_mixed_train_only", "files": []},
        }
        if persistent_source_evidence is not None:
            if not isinstance(persistent_source_evidence, Mapping):
                raise ValueError("persistent source evidence must be an object")
            manifest["persistent_source_evidence"] = dict(persistent_source_evidence)
        atomic_write_json(temporary / "manifest.json", manifest)
        from lehome_train.data.stats import write_train_statistics
        from lehome_train.data.validate import validate_prepared_dataset

        statistics = write_train_statistics(temporary)
        validation = validate_prepared_dataset(temporary)
        # Statistics and validation write their own evidence after the first
        # source-artifact validation.  Seal the final closed file set only once
        # those writes are complete; ``manifest.json`` remains excluded to
        # avoid a self-referential artifact digest.
        final_manifest = _read_json(temporary / "manifest.json")
        final_artifacts = artifact_identities(temporary, exclude={"manifest.json"})
        final_manifest["output_artifacts"] = final_artifacts
        final_manifest["output_manifest_sha256"] = canonical_json_sha256(final_artifacts)
        atomic_write_json(temporary / "manifest.json", final_manifest)
        temporary.replace(destination)
        promoted = True
        atomic_write_json(_generation_receipt_path(destination), _generation_receipt(destination))
        return {"path": str(destination), "mix_plan_sha256": plan.sha256, "statistics": statistics, "validation": validation}
    except BaseException:
        video_slicer.cancel()
        shutil.rmtree(temporary, ignore_errors=True)
        if promoted:
            shutil.rmtree(destination, ignore_errors=True)
            _generation_receipt_path(destination).unlink(missing_ok=True)
        raise
