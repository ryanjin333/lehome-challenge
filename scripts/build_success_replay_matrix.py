#!/usr/bin/env python3
"""Build a balanced replay matrix from checksum-verified 12K successes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import stat
import tempfile
from typing import Mapping, Sequence


CATEGORIES = ("top_long", "top_short", "pant_long", "pant_short")
PARENT_POLICY_REPO = "ryanjin333/lehome-groot-n17-models"
PARENT_REVISION = "30ac1a84da67b099e115ad147bcd61e9d60046d3"
PARENT_ASSET_REVISION = "bea65fd960ad5a1bb3bd3fa77164b28001c08ef9"
PARENT_ARTIFACT_SHA256 = (
    "3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CUDA_DEVICE = re.compile(r"^cuda:[0-9]+$")


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _load_json(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_strict_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is malformed") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} is malformed")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_file(
    *, episode_root: Path, relative: str, checksums: Mapping[str, object]
) -> tuple[Path, str]:
    record = checksums.get(relative)
    if (
        not isinstance(record, Mapping)
        or set(record) != {"sha256", "size"}
        or type(record.get("sha256")) is not str
        or _SHA256.fullmatch(str(record["sha256"])) is None
        or type(record.get("size")) is not int
        or int(record["size"]) < 0
    ):
        raise ValueError("accepted episode checksum manifest is incomplete")
    path = episode_root
    for component in Path(relative).parts:
        if component in {"", ".", ".."}:
            raise ValueError("accepted episode checksum target is missing or unsafe")
        path /= component
        try:
            mode = path.lstat().st_mode
        except OSError as error:
            raise ValueError("accepted episode checksum target is missing or unsafe") from error
        if stat.S_ISLNK(mode):
            raise ValueError("accepted episode checksum target is missing or unsafe")
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("accepted episode checksum target is missing or unsafe")
    if path.stat().st_size != record["size"] or _sha256(path) != record["sha256"]:
        raise ValueError("accepted episode checksum mismatch")
    return path, str(record["sha256"])


def _episode_artifact_sha256(episode_root: Path) -> str:
    """Match the Hub sync digest over the immutable accepted artifact tree."""

    entries: list[dict[str, object]] = []
    for path in sorted(episode_root.rglob("*"), key=lambda candidate: candidate.as_posix()):
        if path.is_symlink():
            raise ValueError("accepted episode artifact contains a symlink")
        if not path.is_file() or path.name == "SHA256SUMS.json":
            continue
        relative = path.relative_to(episode_root).as_posix()
        entries.append(
            {
                "relative_path": relative,
                "sha256": _sha256(path),
                "byte_size": path.stat().st_size,
            }
        )
    if not entries:
        raise ValueError("accepted episode artifact is empty")
    return hashlib.sha256(_canonical_bytes(entries)).hexdigest()


def _state_fingerprint(*, category: str, garment: str, continuation: Mapping[str, object]) -> str:
    state = continuation.get("robot_position")
    if (
        not isinstance(state, list)
        or len(state) != 12
        or any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in state)
    ):
        raise ValueError("accepted continuation state is invalid")
    rounded = ["0.000000" if float(value) == 0.0 else format(float(value), ".6f") for value in state]
    return hashlib.sha256(
        json.dumps(
            {
                "category": category,
                "garment": garment,
                "state_rounding": "fixed_6dp",
                "state": rounded,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _successes(
    accepted_roots: Sequence[Path], *, require_every_category: bool = True,
    require_cpu: bool = False, require_annotations: bool = False,
) -> dict[str, list[dict[str, object]]]:
    if not accepted_roots:
        raise ValueError("at least one accepted root is required")
    grouped: dict[str, list[dict[str, object]]] = {category: [] for category in CATEGORIES}
    seen_attempt_ids: set[str] = set()
    seen_roots: list[Path] = []
    for accepted_root in accepted_roots:
        if not accepted_root.is_absolute() or accepted_root.is_symlink() or not accepted_root.is_dir():
            raise ValueError("accepted root must be a real absolute directory")
        resolved_root = accepted_root.resolve(strict=True)
        if any(
            resolved_root == prior
            or resolved_root.is_relative_to(prior)
            or prior.is_relative_to(resolved_root)
            for prior in seen_roots
        ):
            raise ValueError("accepted roots overlap")
        seen_roots.append(resolved_root)
        for episode_root in sorted(accepted_root.iterdir(), key=lambda path: path.name):
            if episode_root.is_symlink():
                raise ValueError("accepted root contains a symlink")
            if not episode_root.is_dir():
                continue
            attempt_id = episode_root.name
            if (
                not attempt_id
                or "/" in attempt_id
                or "\\" in attempt_id
                or attempt_id in seen_attempt_ids
            ):
                raise ValueError("accepted episode ID is unsafe or duplicated")
            seen_attempt_ids.add(attempt_id)
            checksums = _load_json(
                episode_root / "SHA256SUMS.json", label="accepted episode checksum manifest"
            )
            episode_relative = f"raw/{attempt_id}/episode.json"
            reset_relative = f"raw/{attempt_id}/snapshots/reset.json"
            continuation_relative = f"raw/{attempt_id}/snapshots/continuations/000016.json"
            episode_path, _ = _verified_file(
                episode_root=episode_root, relative=episode_relative, checksums=checksums
            )
            reset_path, _ = _verified_file(
                episode_root=episode_root, relative=reset_relative, checksums=checksums
            )
            continuation_path, continuation_sha256 = _verified_file(
                episode_root=episode_root, relative=continuation_relative, checksums=checksums
            )
            annotations_path: Path | None = None
            annotations_sha256: str | None = None
            if require_annotations:
                annotations_path, annotations_sha256 = _verified_file(
                    episode_root=episode_root,
                    relative=f"raw/{attempt_id}/annotations.jsonl",
                    checksums=checksums,
                )
            episode = _load_json(episode_path, label="accepted episode")
            reset = _load_json(reset_path, label="accepted reset snapshot")
            continuation = _load_json(
                continuation_path, label="accepted early continuation snapshot"
            )
            identity = episode.get("identity")
            provenance = episode.get("provenance")
            if not isinstance(identity, Mapping) or not isinstance(provenance, Mapping):
                raise ValueError("accepted episode identity is malformed")
            category = identity.get("category")
            garment = identity.get("garment_name")
            simulator_device = provenance.get("simulator_device")
            cloth_frame = (
                "usd_local_points_v1"
                if simulator_device == "cpu"
                else "physx_cloth_view_world_v1"
            )
            expected_schema = 3 if simulator_device == "cpu" else 2
            if (
                episode.get("episode_id") != attempt_id
                or identity.get("episode_id") != attempt_id
                or episode.get("accepted_success") is not True
                or episode.get("outcome") != "success"
                or category not in CATEGORIES
                or type(garment) is not str
                or not garment
                or identity.get("release_stage") != "seen"
                or identity.get("policy_repo") != PARENT_POLICY_REPO
                or identity.get("policy_revision") != PARENT_REVISION
                or identity.get("policy_step") != 12_000
                or identity.get("asset_revision") != PARENT_ASSET_REVISION
                or provenance.get("policy_artifact_sha256") != PARENT_ARTIFACT_SHA256
                or not (
                    simulator_device == "cpu"
                    or (isinstance(simulator_device, str) and _CUDA_DEVICE.fullmatch(simulator_device))
                )
                or (require_cpu and simulator_device != "cpu")
                or reset.get("garment_name") != garment
                or reset.get("schema_version") != expected_schema
                or reset.get("cloth_state_authority") != cloth_frame
            ):
                raise ValueError("accepted episode is not a verified 12K seen-garment success")
            if (
                continuation.get("schema_version") != expected_schema
                or continuation.get("cloth_state_authority") != cloth_frame
                or continuation.get("garment_name") != garment
                or not isinstance(continuation.get("randomization"), Mapping)
                or continuation["randomization"].get("continuation_step") != 16
            ):
                raise ValueError("accepted early continuation snapshot is incompatible")
            grouped[str(category)].append(
                {
                    "parent_episode_id": attempt_id,
                    "garment": garment,
                    "restore_snapshot": str(continuation_path),
                    "restore_snapshot_sha256": continuation_sha256,
                    "restore_snapshot_cloth_frame": cloth_frame,
                    "restore_snapshot_step": 16,
                    "accepted_root": accepted_root,
                    "episode_root": episode_root,
                    "episode_sha256": _episode_artifact_sha256(episode_root)
                    if require_annotations else None,
                    "reset_sha256": _sha256(reset_path) if require_annotations else None,
                    "annotations_sha256": annotations_sha256,
                    "continuation_state_fingerprint": _state_fingerprint(
                        category=str(category), garment=garment, continuation=continuation,
                    ) if require_annotations else None,
                }
            )
    if require_every_category and any(not grouped[category] for category in CATEGORIES):
        raise ValueError("verified successes are required for every category")
    return grouped


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _fresh_source_parents(
    *, accepted_roots: Sequence[Path], source_reports: Sequence[Path],
    source_matrices: Sequence[Path],
) -> tuple[dict[str, list[dict[str, object]]], dict[str, object]]:
    """Authenticate fresh source reports, matrices, receipts, and local artifacts.

    This is intentionally separate from the legacy accepted-root path: fresh
    visual replay may only be built from a fully bound source campaign.
    """

    if not source_reports or not source_matrices:
        raise ValueError("fresh visual-only replay requires source reports and source matrices")
    matrices: dict[str, tuple[Path, dict[str, Mapping[str, object]]]] = {}
    matrix_attempts: set[str] = set()
    for path in source_matrices:
        # Matrices are immutable JSON arrays, unlike reports.
        if path.is_symlink() or not path.is_file():
            raise ValueError("fresh source matrix is missing or unsafe")
        try:
            rows = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_strict_pairs)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("fresh source matrix is malformed") from error
        if not isinstance(rows, list) or not rows:
            raise ValueError("fresh source matrix is malformed")
        digest = _sha256(path)
        if digest in matrices:
            raise ValueError("fresh source matrix is duplicated")
        by_attempt: dict[str, Mapping[str, object]] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("fresh source matrix row is malformed")
            attempt_id = row.get("attempt_id")
            if (
                not isinstance(attempt_id, str) or not attempt_id
                or attempt_id in matrix_attempts
                or row.get("trial_id") != attempt_id
                or row.get("category") not in CATEGORIES
                or not isinstance(row.get("garment_name"), str) or not row["garment_name"]
                or row.get("release_stage") != "seen"
            ):
                raise ValueError("fresh source matrix identity is invalid")
            matrix_attempts.add(attempt_id)
            by_attempt[attempt_id] = row
        matrices[digest] = (path, by_attempt)

    report_trials: dict[str, tuple[Mapping[str, object], Mapping[str, object], str]] = {}
    report_context: dict[str, object] = {}
    for path in source_reports:
        report = _load_json(path, label="fresh source report")
        report_sha256 = _sha256(path)
        body = dict(report)
        declared = body.pop("report_sha256", None)
        if (
            set(report) != {
                "schema_version", "kind", "campaign_kind", "round_id", "matrix_sha256",
                "identity", "trials", "safety_failure", "report_sha256",
            }
            or declared != hashlib.sha256(_canonical_bytes(body)).hexdigest()
            or report.get("schema_version") != 1
            or report.get("kind") != "lehome_fresh_12k_success_source_report_v1"
            or report.get("campaign_kind") != "fresh_12k_success_source_v1"
            or not isinstance(report.get("round_id"), str)
            or re.fullmatch(r"fresh-12k-[a-z0-9-]{1,112}", report["round_id"]) is None
            or report.get("safety_failure") is not False
            or not isinstance(report.get("identity"), Mapping)
            or report["identity"] != {
                "policy_repo": PARENT_POLICY_REPO,
                "policy_revision": PARENT_REVISION,
                "policy_step": 12_000,
                "policy_artifact_sha256": PARENT_ARTIFACT_SHA256,
            }
            or not isinstance(report.get("matrix_sha256"), str)
            or report["matrix_sha256"] not in matrices
            or not isinstance(report.get("trials"), list)
        ):
            raise ValueError("fresh source report is not authenticated for the exact 12K policy")
        matrix_rows = matrices[report["matrix_sha256"]][1]
        for trial in report["trials"]:
            if not isinstance(trial, Mapping):
                raise ValueError("fresh source report trial is malformed")
            attempt_id = trial.get("attempt_id")
            matrix_row = matrix_rows.get(attempt_id) if isinstance(attempt_id, str) else None
            if (
                attempt_id in report_trials
                or matrix_row is None
                or trial.get("category") != matrix_row.get("category")
                or trial.get("garment_name") != matrix_row.get("garment_name")
                or type(trial.get("accepted_success")) is not bool
                or trial.get("outcome") not in {"success", "failure"}
                or trial.get("simulator_device") != "cpu"
                or trial.get("cloth_device") != "cpu"
                or any(trial.get(field) is not False for field in (
                    "safety_failure", "numerical_failure", "cloth_failure",
                ))
                or not isinstance(trial.get("artifact_sha256"), str)
                or _SHA256.fullmatch(trial["artifact_sha256"]) is None
                or not isinstance(trial.get("hub_sync_receipt_sha256"), str)
                or _SHA256.fullmatch(trial["hub_sync_receipt_sha256"]) is None
                or trial.get("remote_prefix") != f"rollout-rounds/{report['round_id']}/{attempt_id}"
            ):
                raise ValueError("fresh source report trial is not an eligible CPU source")
            report_trials[attempt_id] = (trial, report, report_sha256)
            report_context[attempt_id] = {
                "source_report_sha256": report_sha256,
                "source_matrix_sha256": report["matrix_sha256"],
                "source_round_id": report["round_id"],
            }
    if set(report_trials) != matrix_attempts:
        raise ValueError("every fresh source must appear in exactly one report and source matrix")

    grouped = _successes(
        accepted_roots, require_every_category=False, require_cpu=True, require_annotations=True
    )
    parents_by_attempt = {
        str(parent["parent_episode_id"]): parent
        for category in CATEGORIES for parent in grouped[category]
    }
    if not set(parents_by_attempt).issubset(report_trials):
        raise ValueError("a success not present in the authenticated fresh source report is rejected")

    fresh_rates: dict[tuple[str, str], tuple[int, int]] = {}
    for trial, _, _ in report_trials.values():
        key = (str(trial["category"]), str(trial["garment_name"]))
        successes, total = fresh_rates.get(key, (0, 0))
        fresh_rates[key] = (
            successes + int(trial.get("accepted_success") is True and trial.get("outcome") == "success"),
            total + 1,
        )

    for attempt_id, parent in parents_by_attempt.items():
        trial, report, report_sha256 = report_trials[attempt_id]
        if (
            trial.get("accepted_success") is not True
            or trial.get("outcome") != "success"
            or trial.get("category") not in CATEGORIES
            or trial.get("category") != next(
                category for category in CATEGORIES if parent in grouped[category]
            )
            or trial.get("garment_name") != parent["garment"]
            or trial.get("artifact_sha256") != parent["episode_sha256"]
        ):
            raise ValueError("fresh source success identity is not authenticated")
        receipt_path = Path(parent["accepted_root"]).parent / "hf-sync-receipts" / f"{attempt_id}.sync.json"
        receipt = _load_json(receipt_path, label="fresh source Hub sync receipt")
        if (
            _sha256(receipt_path) != trial.get("hub_sync_receipt_sha256")
            or receipt.get("readback_verified") is not True
            or receipt.get("attempt_id") != attempt_id
            or receipt.get("round_id") != report["round_id"]
            or receipt.get("remote_prefix") != trial.get("remote_prefix")
            or receipt.get("episode_sha256") != parent["episode_sha256"]
            or not isinstance(receipt.get("immutable_revision"), str)
            or re.fullmatch(r"[0-9a-f]{40}", receipt["immutable_revision"]) is None
        ):
            raise ValueError("fresh source Hub receipt does not bind the immutable artifact")
        parent.update(
            {
                **report_context[attempt_id],
                "source_receipt_sha256": _sha256(receipt_path),
                "source_remote_prefix": receipt["remote_prefix"],
                "source_immutable_revision": receipt["immutable_revision"],
                "fresh_success_rate": fresh_rates[(str(trial["category"]), str(trial["garment_name"]))][0]
                / fresh_rates[(str(trial["category"]), str(trial["garment_name"]))][1],
            }
        )
    return grouped, report_context


def _write_absent(path: Path, payload: bytes) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise FileExistsError("output must be an absent absolute path")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("output parent is unsafe")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _preflight_outputs(path: Path) -> None:
    """Reject either immutable output before publishing either one."""

    receipt = Path(str(path) + ".sha256")
    for candidate in (path, receipt):
        if not candidate.is_absolute() or candidate.exists() or candidate.is_symlink():
            raise FileExistsError("output must be an absent absolute path")
    if path.parent.is_symlink() or (path.parent.exists() and not path.parent.is_dir()):
        raise ValueError("output parent is unsafe")


def build_success_replay_matrix(
    *,
    output: str | Path,
    accepted_root: str | Path | None = None,
    accepted_roots: Sequence[str | Path] | None = None,
    attempts_per_category: int = 50,
    attempts_by_category: Mapping[str, int] | None = None,
    acceptance_caps: Mapping[str, int] | None = None,
    seed_base: int = 50_000,
    source_reports: Sequence[str | Path] | None = None,
    source_matrices: Sequence[str | Path] | None = None,
    strategy: str | None = None,
    attempt_cap_per_category: int | None = None,
    acceptance_cap_per_category: int | None = None,
    max_attempts: int | None = None,
    target_accepted: int | None = None,
    rng_seed: int | None = None,
) -> dict[str, object]:
    fresh_inputs = (
        source_reports, source_matrices, strategy, attempt_cap_per_category,
        acceptance_cap_per_category, max_attempts, target_accepted, rng_seed,
    )
    fresh_mode = any(value is not None for value in fresh_inputs)
    if fresh_mode:
        if (
            source_reports is None or source_matrices is None or strategy != "visual_only"
            or attempt_cap_per_category != 100 or acceptance_cap_per_category != 50
            or max_attempts != 400 or target_accepted != 200
            or type(rng_seed) is not int or rng_seed < 0
            or attempts_by_category is not None or acceptance_caps is not None
        ):
            raise ValueError("fresh replay requires the exact bounded visual-only 12K tuple")
        if attempts_per_category != 50 or seed_base != 50_000:
            raise ValueError("fresh replay cannot combine legacy builder controls")
    if attempts_by_category is None:
        if type(attempts_per_category) is not int or not 1 <= attempts_per_category <= 100:
            raise ValueError("attempts_per_category must be between 1 and 100")
        attempt_counts = {category: attempts_per_category for category in CATEGORIES}
    else:
        if set(attempts_by_category) != set(CATEGORIES):
            raise ValueError("attempts_by_category must contain every canonical category")
        attempt_counts = dict(attempts_by_category)
        if any(type(value) is not int or not 0 <= value <= 400 for value in attempt_counts.values()):
            raise ValueError("category attempt counts must be integers between 0 and 400")
        if not 1 <= sum(attempt_counts.values()) <= 400:
            raise ValueError("total category attempt count must be between 1 and 400")
    normalized_caps: dict[str, int] | None = None
    if acceptance_caps is not None:
        if set(acceptance_caps) != set(CATEGORIES):
            raise ValueError("acceptance_caps must contain every canonical category")
        normalized_caps = dict(acceptance_caps)
        if any(
            type(normalized_caps[category]) is not int
            or normalized_caps[category] < 0
            or normalized_caps[category] > attempt_counts[category]
            for category in CATEGORIES
        ):
            raise ValueError("each acceptance cap must be between zero and its category attempt count")
        if not 1 <= sum(normalized_caps.values()) <= 150:
            raise ValueError("total acceptance cap must be between 1 and 150")
    if type(seed_base) is not int or seed_base < 0:
        raise ValueError("seed_base must be nonnegative")
    if (accepted_root is None) == (accepted_roots is None):
        raise ValueError("provide exactly one accepted root input form")
    roots = (
        (Path(accepted_root),)
        if accepted_root is not None
        else tuple(Path(root) for root in accepted_roots or ())
    )
    if fresh_mode:
        grouped, _ = _fresh_source_parents(
            accepted_roots=roots,
            source_reports=tuple(Path(path) for path in source_reports or ()),
            source_matrices=tuple(Path(path) for path in source_matrices or ()),
        )
        attempt_counts = {category: 100 if grouped[category] else 0 for category in CATEGORIES}
        normalized_caps = {category: 50 for category in CATEGORIES}
    else:
        grouped = _successes(roots)
    rows: list[dict[str, object]] = []
    seed_offset = 0
    generator = random.Random(rng_seed) if fresh_mode else None
    shortages: list[dict[str, str]] = []
    for category_index, category in enumerate(CATEGORIES):
        parents = grouped[category]
        if fresh_mode and not parents:
            shortages.append({"category": category, "reason": "no_eligible_source"})
            continue
        for index in range(attempt_counts[category]):
            if fresh_mode:
                # First select a garment by its observed fresh success rate,
                # then one source state uniformly within that garment.
                by_garment: dict[str, list[dict[str, object]]] = {}
                for parent in parents:
                    by_garment.setdefault(str(parent["garment"]), []).append(parent)
                garments = sorted(by_garment)
                weights = [
                    max(1.0 - float(by_garment[garment][0]["fresh_success_rate"]), 0.01)
                    for garment in garments
                ]
                parent = generator.choices(garments, weights=weights, k=1)[0]  # type: ignore[union-attr]
                parent = generator.choice(by_garment[parent])  # type: ignore[union-attr]
                row_strategy = "visual_only"
                seed = int(rng_seed) + seed_offset + index
            else:
                parent = parents[index % len(parents)]
                row_strategy = "mild_geometry" if index % 5 < 3 else "strong_geometry"
                seed = seed_base + seed_offset + index
            replay_identity = {
                "schema_version": 1,
                "parent_episode_id": parent["parent_episode_id"],
                "category": category,
                "strategy": row_strategy,
                "seed": seed,
            }
            suffix = hashlib.sha256(_canonical_bytes(replay_identity)).hexdigest()[:16]
            attempt_id = f"replay-{category.replace('_', '-')}-{index:03d}-{suffix}"
            row: dict[str, object] = {
                    "attempt_id": attempt_id,
                    "trial_id": attempt_id,
                    "garment": parent["garment"],
                    "garment_name": parent["garment"],
                    "category": category,
                    "release_stage": "seen",
                    "difficulty": "randomized",
                    "seed": seed,
                    "strategy": row_strategy,
                    "restore_snapshot": parent["restore_snapshot"],
                    "restore_snapshot_sha256": parent["restore_snapshot_sha256"],
                    "restore_snapshot_cloth_frame": parent["restore_snapshot_cloth_frame"],
                    "restore_snapshot_step": parent["restore_snapshot_step"],
                    "parent_episode_id": parent["parent_episode_id"],
                    "lineage_id": parent["parent_episode_id"],
                    "replay_kind": "verified_success_early_snapshot_v1",
                }
            if normalized_caps is not None:
                row["category_acceptance_cap"] = normalized_caps[category]
            if fresh_mode:
                row.update(
                    {
                        "source_episode_sha256": parent["episode_sha256"],
                        "source_reset_sha256": parent["reset_sha256"],
                        "source_annotations_sha256": parent["annotations_sha256"],
                        "source_continuation_snapshot_sha256": parent["restore_snapshot_sha256"],
                        "source_state_fingerprint": parent["continuation_state_fingerprint"],
                        "source_report_sha256": parent["source_report_sha256"],
                        "source_matrix_sha256": parent["source_matrix_sha256"],
                        "source_receipt_sha256": parent["source_receipt_sha256"],
                        "source_remote_prefix": parent["source_remote_prefix"],
                        "source_immutable_revision": parent["source_immutable_revision"],
                        "source_round_id": parent["source_round_id"],
                    }
                )
            rows.append(row)
        seed_offset += attempt_counts[category]
    payload = _canonical_bytes(rows)
    destination = Path(output)
    _preflight_outputs(destination)
    _write_absent(destination, payload)
    digest = hashlib.sha256(payload).hexdigest()
    _write_absent(Path(str(destination) + ".sha256"), (digest + "\n").encode("ascii"))
    receipt: dict[str, object] = {
        "matrix_path": str(destination),
        "matrix_sha256": digest,
        "attempt_count": len(rows),
    }
    if attempts_by_category is None:
        receipt["attempts_per_category"] = attempts_per_category
    else:
        receipt["attempts_by_category"] = attempt_counts
    if normalized_caps is not None:
        receipt["acceptance_caps"] = normalized_caps
    if fresh_mode:
        receipt.update(
            {
                "strategy": "visual_only",
                "attempt_cap_per_category": attempt_cap_per_category,
                "acceptance_cap_per_category": acceptance_cap_per_category,
                "max_attempts": max_attempts,
                "target_accepted": target_accepted,
                "rng_seed": rng_seed,
                "shortages": shortages,
            }
        )
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempts-per-category", type=int, default=50)
    parser.add_argument("--attempts-by-category-json")
    parser.add_argument("--acceptance-caps-json")
    parser.add_argument("--seed-base", type=int, default=50_000)
    parser.add_argument("--source-report", type=Path, action="append")
    parser.add_argument("--source-matrix", type=Path, action="append")
    parser.add_argument("--strategy")
    parser.add_argument("--attempt-cap-per-category", type=int)
    parser.add_argument("--acceptance-cap-per-category", type=int)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--target-accepted", type=int)
    parser.add_argument("--rng-seed", type=int)
    args = parser.parse_args(argv)
    attempts_by_category = (
        json.loads(args.attempts_by_category_json)
        if args.attempts_by_category_json is not None
        else None
    )
    acceptance_caps = (
        json.loads(args.acceptance_caps_json)
        if args.acceptance_caps_json is not None
        else None
    )
    print(
        json.dumps(
            build_success_replay_matrix(
                accepted_roots=tuple(args.accepted_root),
                output=args.output,
                attempts_per_category=args.attempts_per_category,
                attempts_by_category=attempts_by_category,
                acceptance_caps=acceptance_caps,
                seed_base=args.seed_base,
                source_reports=tuple(args.source_report) if args.source_report else None,
                source_matrices=tuple(args.source_matrix) if args.source_matrix else None,
                strategy=args.strategy,
                attempt_cap_per_category=args.attempt_cap_per_category,
                acceptance_cap_per_category=args.acceptance_cap_per_category,
                max_attempts=args.max_attempts,
                target_accepted=args.target_accepted,
                rng_seed=args.rng_seed,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
