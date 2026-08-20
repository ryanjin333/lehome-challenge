"""Fail-closed, provenance-bound audit for successful policy recoveries.

The audit deliberately does not infer visual failure modes.  It admits only
sealed autonomous successes and identifies recoveries solely from finite,
recorded reward and policy annotations.  The emitted selection is a stable
handoff contract: a later runtime adapter can consume its h=16 ranges and
lineage without re-running this detector.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from lehome_train.groot.rollout_source_adapter import _accepted_episode, _seal
from lehome_train.io import canonical_json_bytes, canonical_json_sha256, sha256_file


_CATEGORIES = ("top_long", "top_short", "pant_long", "pant_short")
_HORIZON = 16


@dataclass(frozen=True)
class RecoveryThresholds:
    """Finite detector thresholds; reward comparisons use ``improvement_epsilon``."""

    nontrivial_progress: float = 0.15
    minimum_drawdown: float = 0.05
    minimum_stall_steps: int = 16
    minimum_recovery_gain: float = 0.10
    improvement_epsilon: float = 1e-6


@dataclass(frozen=True)
class _Event:
    kind: str
    adverse_start: int
    trough: int | None
    confirmation: int
    peak: int
    strength: float
    recovery_gain: float


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field in annotations")
        result[key] = value
    return result


def _reject_constant(_: str) -> object:
    raise ValueError("annotation JSON contains a non-finite number")


def _absolute_directory(path: str | Path, label: str) -> Path:
    target = Path(path)
    if not target.is_absolute() or target.is_symlink() or not target.is_dir():
        raise ValueError(f"{label} must be an absolute, non-symlink directory")
    if target.resolve(strict=True) != target:
        raise ValueError(f"{label} contains a symlink")
    return target


def _absolute_file(path: str | Path, label: str) -> Path:
    target = Path(path)
    if not target.is_absolute() or target.is_symlink() or not target.is_file():
        raise ValueError(f"{label} must be an absolute, non-symlink regular file")
    if target.resolve(strict=True) != target:
        raise ValueError(f"{label} contains a symlink")
    return target


def _output_path(path: str | Path) -> Path:
    target = Path(path)
    if not target.is_absolute() or target.exists() or target.is_symlink():
        raise FileExistsError("audit output must be an absent absolute non-symlink path")
    parent = target.parent
    if not parent.is_dir() or parent.is_symlink() or parent.resolve(strict=True) != parent:
        raise ValueError("audit output parent must be a real directory")
    if target.with_name(target.name + ".sha256").exists():
        raise FileExistsError("audit output checksum path already exists")
    return target


def _overlaps(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
        return True
    except ValueError:
        try:
            right.relative_to(left)
            return True
        except ValueError:
            return False


def _validate_thresholds(thresholds: RecoveryThresholds, horizon: int, minimum: int) -> None:
    if horizon != _HORIZON:
        raise ValueError("successful recovery audit horizon is fixed at 16")
    if type(minimum) is not int or minimum <= 0:
        raise ValueError("per-category minimum must be a positive integer")
    if type(thresholds.minimum_stall_steps) is not int or thresholds.minimum_stall_steps <= 0:
        raise ValueError("minimum stall steps must be a positive integer")
    values = (
        thresholds.nontrivial_progress, thresholds.minimum_drawdown,
        thresholds.minimum_recovery_gain, thresholds.improvement_epsilon,
    )
    if any(type(value) not in (int, float) or not math.isfinite(float(value)) or float(value) < 0 for value in values):
        raise ValueError("recovery thresholds must be finite and non-negative")
    if thresholds.nontrivial_progress <= thresholds.improvement_epsilon:
        raise ValueError("nontrivial progress must exceed improvement epsilon")
    if thresholds.minimum_drawdown <= thresholds.improvement_epsilon:
        raise ValueError("minimum drawdown must exceed improvement epsilon")
    if thresholds.minimum_recovery_gain <= thresholds.improvement_epsilon:
        raise ValueError("minimum recovery gain must exceed improvement epsilon")


def _annotations(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("sealed annotations are missing or unsafe")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError("sealed annotations are unreadable") from error
    if not lines:
        raise ValueError("sealed annotations are empty")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            value = json.loads(line, object_pairs_hook=_strict_pairs, parse_constant=_reject_constant)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError("sealed annotation is malformed") from error
        if not isinstance(value, dict):
            raise ValueError("sealed annotation must be an object")
        required = {
            "step", "reward", "success", "state", "action", "action_source",
            "policy_request_id", "policy_chunk_offset",
        }
        if not required <= set(value):
            if not {"policy_request_id", "policy_chunk_offset"} <= set(value):
                raise ValueError("annotation policy chunk provenance is missing")
            raise ValueError("sealed annotation is missing recovery fields")
        if type(value["step"]) is not int or value["step"] != index:
            raise ValueError("annotation steps must be zero-based contiguous integers")
        if type(value["reward"]) not in (int, float) or not math.isfinite(float(value["reward"])):
            raise ValueError("annotation reward must be finite")
        if type(value["success"]) is not bool or value["action_source"] != "policy":
            raise ValueError("annotations must record boolean success and policy actions")
        request_id, chunk_offset = value["policy_request_id"], value["policy_chunk_offset"]
        if (not isinstance(request_id, str) or not request_id.strip()
                or type(chunk_offset) is not int or not 0 <= chunk_offset < _HORIZON):
            raise ValueError("annotation policy chunk provenance is malformed")
        for field in ("state", "action"):
            vector = value[field]
            if not isinstance(vector, list) or len(vector) != 12 or any(
                type(number) not in (int, float) or not math.isfinite(float(number)) for number in vector
            ):
                raise ValueError(f"annotation {field} must be a finite 12-D vector")
        rows.append(value)
    _validate_policy_chunk_trace(rows)
    official = next((index for index, row in enumerate(rows) if row["success"]), None)
    if official is None or not rows[-1]["success"] or not all(row["success"] for row in rows[official:]):
        raise ValueError("annotation success must remain latched through the terminal record")
    return rows


def _validate_policy_chunk_trace(rows: Sequence[Mapping[str, object]]) -> None:
    """Authenticate the exact H=16 action-cache trace, not only individual fields."""

    request_ids: set[str] = set()
    active_request: str | None = None
    for index, row in enumerate(rows):
        request_id, offset = row["policy_request_id"], row["policy_chunk_offset"]
        expected_offset = index % _HORIZON
        if offset != expected_offset:
            raise ValueError("annotation policy chunk trace has an invalid offset transition")
        if expected_offset == 0:
            if request_id in request_ids:
                raise ValueError("annotation policy chunk trace reuses a request ID")
            request_ids.add(str(request_id))
            active_request = str(request_id)
        elif request_id != active_request:
            raise ValueError("annotation policy chunk trace changes request before offset 15")


def _continuation_start(
    *, rows: Sequence[Mapping[str, object]], event: _Event, category: str, garment: str,
) -> dict[str, object] | None:
    """Bind a reset-safe policy continuation at an authenticated chunk boundary.

    Replaying the source prefix reconstructs physical state but not GR00T's
    action-cache/session history.  Only an action with offset zero is therefore
    evidence that a new policy request can begin from the selected state.
    """

    for index in range(event.adverse_start, event.confirmation):
        row = rows[index]
        if row["policy_chunk_offset"] != 0:
            continue
        state = list(row["state"])
        return {
            "annotation_index": index,
            "step": row["step"],
            "policy_request_id": row["policy_request_id"],
            "policy_chunk_offset": 0,
            "state": state,
            "state_fingerprint": _fingerprint(category=category, garment=garment, state=state),
        }
    return None


def _drawdown_events(rewards: list[float], official: int, thresholds: RecoveryThresholds) -> list[_Event]:
    candidates: list[_Event] = []
    for peak in range(official):
        if rewards[peak] < thresholds.nontrivial_progress:
            continue
        for trough in range(peak + 1, official):
            depth = rewards[peak] - rewards[trough]
            if depth + thresholds.improvement_epsilon < thresholds.minimum_drawdown:
                continue
            confirmation = next((
                position for position in range(trough + 1, official + 1)
                if (
                    rewards[position] + thresholds.improvement_epsilon >= rewards[peak]
                    and rewards[position] - rewards[trough] + thresholds.improvement_epsilon
                    >= thresholds.minimum_recovery_gain
                )
            ), None)
            if confirmation is not None:
                candidates.append(_Event(
                    kind="reward_drawdown", adverse_start=trough, trough=trough,
                    confirmation=confirmation, peak=peak, strength=depth,
                    recovery_gain=rewards[confirmation] - rewards[trough],
                ))
    return candidates


def _event(rows: list[dict[str, Any]], thresholds: RecoveryThresholds) -> _Event | None:
    official = next(index for index, row in enumerate(rows) if row["success"])
    rewards = [float(row["reward"]) for row in rows]
    # The current production annotation schema has no reward-freshness field.
    # Cached reward plateaus are therefore indistinguishable from true stalls,
    # so stall-then-recovery admission is deliberately disabled until a sealed
    # freshness signal exists.  Drawdown evidence remains independently sound.
    candidates = _drawdown_events(rewards, official, thresholds)
    if not candidates:
        return None
    # Strongest evidence first; ties are fully resolved by event position and kind.
    return min(candidates, key=lambda item: (
        -item.strength, -item.recovery_gain, item.adverse_start, item.confirmation, item.peak, item.kind,
    ))


def _fingerprint(*, category: str, garment: str, state: list[object]) -> str:
    # Six fixed decimal places make the state identity portable across JSON float
    # spellings; normalized negative zero prevents two encodings of zero.
    rounded = []
    for value in state:
        number = float(value)
        rounded.append("0.000000" if number == 0.0 else format(number, ".6f"))
    return canonical_json_sha256({"category": category, "garment": garment, "state_rounding": "fixed_6dp", "state": rounded})


def _h16_ranges(*, steps: Sequence[int], interval_start: int, interval_stop: int) -> list[dict[str, object]]:
    length = len(steps)
    if length < _HORIZON:
        return []
    ranges: list[dict[str, object]] = []
    # Anchor at the corrective onset, except when that would leave no complete
    # window before episode end.  This deterministic backward shift preserves
    # an h=16 corrective window rather than emitting a forbidden short tail.
    start = min(interval_start, length - _HORIZON)
    while start <= interval_stop:
        stop = start + _HORIZON
        if stop <= length and start <= interval_stop and stop - 1 >= interval_start:
            ranges.append({"start": start, "stop": stop, "frame_ids": list(steps[start:stop])})
        start += _HORIZON
    return ranges


def _lineage_id(record: Mapping[str, object]) -> str:
    return canonical_json_sha256(record)


def _atomic_pair(path: Path, document: Mapping[str, object]) -> None:
    checksum_path = path.with_name(path.name + ".sha256")
    payload = canonical_json_bytes(document)
    checksum = hashlib.sha256(payload).hexdigest().encode("ascii") + b"\n"
    temporary: list[Path] = []
    linked: list[Path] = []
    try:
        for destination, body in ((path, payload), (checksum_path, checksum)):
            descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
            staged = Path(name)
            temporary.append(staged)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
        # The main JSON is the commit marker.  Publish its checksum first so a
        # crash cannot make a visible audit appear without its sidecar.
        for destination, staged in zip((checksum_path, path), reversed(temporary), strict=True):
            os.link(staged, destination)
            linked.append(destination)
        flags = os.O_RDONLY | (os.O_DIRECTORY if hasattr(os, "O_DIRECTORY") else 0)
        descriptor = os.open(path.parent, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        for destination in reversed(linked):
            destination.unlink(missing_ok=True)
        raise
    finally:
        for staged in temporary:
            staged.unlink(missing_ok=True)


def audit_successful_recoveries(
    *,
    accepted_roots: Sequence[str | Path],
    receipt_roots: Sequence[str | Path],
    round_seal_paths: Sequence[str | Path],
    output_path: str | Path,
    thresholds: RecoveryThresholds = RecoveryThresholds(),
    horizon: int = _HORIZON,
    per_category_minimum: int = 5,
) -> dict[str, object]:
    """Audit sealed policy-success rounds and atomically emit recovery-only h=16 selections."""

    _validate_thresholds(thresholds, horizon, per_category_minimum)
    if not accepted_roots or not (len(accepted_roots) == len(receipt_roots) == len(round_seal_paths)):
        raise ValueError("accepted roots, receipt roots, and round seals must be non-empty and same-length")
    output = _output_path(output_path)
    accepted = tuple(_absolute_directory(path, "accepted root") for path in accepted_roots)
    receipts = tuple(_absolute_directory(path, "HF receipt root") for path in receipt_roots)
    seals = tuple(_absolute_file(path, "round seal") for path in round_seal_paths)
    inputs = (*accepted, *receipts, *seals)
    if any(_overlaps(output, source) or _overlaps(output.with_name(output.name + ".sha256"), source) for source in inputs):
        raise ValueError("audit output must not overlap an input source")

    round_records: list[dict[str, object]] = []
    admitted: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    seen_rounds: set[str] = set()
    seen_episodes: set[str] = set()
    for ordinal, (root, receipt_root, seal_path) in enumerate(zip(accepted, receipts, seals, strict=True)):
        seal = _seal(seal_path)
        round_id = str(seal["round_id"])
        if round_id in seen_rounds:
            raise ValueError("round seal ID collision")
        seen_rounds.add(round_id)
        round_records.append({
            "ordinal": ordinal, "round_id": round_id, "seal_file_name": seal_path.name,
            "seal_file_sha256": sha256_file(seal_path), "seal_sha256": seal["seal_sha256"],
            "repository": seal["repository"], "episode_count": seal["episode_count"],
        })
        for attempt_id in sorted(seal["episode_sha256s"]):
            if attempt_id in seen_episodes:
                raise ValueError("cross-round episode-ID collision")
            seen_episodes.add(attempt_id)
            episode, raw, receipt = _accepted_episode(
                root=root, receipts_root=receipt_root, seal=seal, attempt_id=attempt_id,
            )
            identity = episode["identity"]
            category, garment = str(identity["category"]), str(identity["garment_name"])
            rows = _annotations(raw / "annotations.jsonl")
            package = root / attempt_id
            admitted_record: dict[str, object] = {
                "source_round_id": round_id, "source_round_ordinal": ordinal,
                "source_episode_id": attempt_id, "source_immutable_revision": seal["immutable_revisions"][attempt_id],
                "source_episode_digest": seal["episode_sha256s"][attempt_id],
                "source_receipt_file_name": f"{attempt_id}.sync.json",
                "source_receipt_sha256": sha256_file(receipt_root / f"{attempt_id}.sync.json"),
                "source_receipt_remote_prefix": receipt["remote_prefix"],
                "source_receipt_publication_ref": receipt["publication_ref"],
                "source_artifacts": {
                    # Hub packages are authenticated by the seal-bound sync
                    # digest.  They do not carry a top-level SHA256SUMS file;
                    # retain hashes of their actual package metadata instead.
                    "package_sync_digest": seal["episode_sha256s"][attempt_id],
                    "flywheel_manifest_sha256": sha256_file(package / "flywheel-manifest.json"),
                    "worker_receipt_sha256": sha256_file(package / "worker-receipt.json"),
                    "raw_checksum_manifest_sha256": sha256_file(raw / "SHA256SUMS.json"),
                    "episode_manifest_sha256": sha256_file(raw / "episode.json"),
                    "annotations_sha256": sha256_file(raw / "annotations.jsonl"),
                },
                "category": category, "garment": garment, "release_stage": identity["release_stage"],
                "annotation_count": len(rows), "official_success_step": next(
                    row["step"] for row in rows if row["success"]
                ),
                "receipt_immutable_revision": receipt["immutable_revision"],
            }
            admitted.append(admitted_record)
            found = _event(rows, thresholds)
            if found is None:
                exclusions.append({"source_round_id": round_id, "source_episode_id": attempt_id, "reason": "no_meaningful_recovery"})
                continue
            event = {
                "kind": found.kind, "adverse_start": found.adverse_start,
                "adverse_start_step": rows[found.adverse_start]["step"], "trough": found.trough,
                "trough_step": None if found.trough is None else rows[found.trough]["step"],
                "peak": found.peak, "peak_step": rows[found.peak]["step"],
                "recovery_confirmation": found.confirmation,
                "recovery_confirmation_step": rows[found.confirmation]["step"],
                "strength": found.strength, "recovery_gain": found.recovery_gain,
            }
            continuation = _continuation_start(
                rows=rows, event=found, category=category, garment=garment,
            )
            if continuation is None:
                exclusions.append({
                    "source_round_id": round_id, "source_episode_id": attempt_id,
                    "reason": "no_fresh_policy_boundary_before_recovery_confirmation",
                })
                continue
            lineage = _lineage_id({
                "round_id": round_id, "episode_id": attempt_id,
                "episode_digest": seal["episode_sha256s"][attempt_id],
                "immutable_revision": seal["immutable_revisions"][attempt_id],
                "start_step": continuation["step"], "stop_step": rows[found.confirmation]["step"],
                "policy_request_id": continuation["policy_request_id"],
            })
            ranges = _h16_ranges(
                steps=[int(row["step"]) for row in rows], interval_start=found.adverse_start,
                interval_stop=found.confirmation,
            )
            if not ranges:
                exclusions.append({
                    "source_round_id": round_id, "source_episode_id": attempt_id,
                    "reason": "no_full_h16_corrective_window",
                })
                continue
            candidates.append({
                **admitted_record, "fingerprint": continuation["state_fingerprint"], "lineage_id": lineage,
                "recovery_event": event,
                "continuation_start": continuation,
                "h16_ranges": ranges,
            })

    winners: dict[str, dict[str, object]] = {}
    duplicates: list[dict[str, object]] = []
    selected: list[dict[str, object]] = []
    for candidate in sorted(candidates, key=lambda row: (int(row["source_round_ordinal"]), str(row["source_episode_id"]))):
        fingerprint = str(candidate["fingerprint"])
        winner = winners.get(fingerprint)
        if winner is not None:
            duplicates.append({
                "duplicate_episode_id": candidate["source_episode_id"], "winner_episode_id": winner["source_episode_id"],
                "fingerprint": fingerprint, "reason": "duplicate_continuation_start_state",
            })
            continue
        winners[fingerprint] = candidate
        selected.append(candidate)
    counts = {category: sum(row["category"] == category for row in selected) for category in _CATEGORIES}
    shortfalls = {category: max(0, per_category_minimum - count) for category, count in counts.items()}
    document: dict[str, object] = {
        "schema_version": 2, "kind": "lehome_successful_recovery_audit", "horizon": _HORIZON,
        "thresholds": asdict(thresholds), "per_category_minimum": per_category_minimum,
        "rounds": round_records, "admitted_episodes": admitted,
        "selected_recoveries": selected, "duplicates": duplicates, "exclusions": exclusions,
        "per_category_counts": counts, "shortfalls": shortfalls, "ready": not any(shortfalls.values()),
        "fingerprint_normalization": "category+garment+fresh_policy_continuation_state fixed_6dp canonical JSON SHA-256",
        "detector_mode": "reward_drawdown_only_no_reward_freshness_annotations",
    }
    document["semantic_sha256"] = canonical_json_sha256(document)
    _atomic_pair(output, document)
    return {
        "output_path": str(output), "output_sha256": hashlib.sha256(canonical_json_bytes(document)).hexdigest(),
        "ready": document["ready"], "per_category_counts": counts, "shortfalls": shortfalls,
        "selected_count": len(selected),
    }
