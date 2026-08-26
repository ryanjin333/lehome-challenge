#!/usr/bin/env python3
"""Build a balanced replay matrix from checksum-verified 12K successes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Mapping


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


def _successes(accepted_root: Path) -> dict[str, list[dict[str, object]]]:
    if not accepted_root.is_absolute() or accepted_root.is_symlink() or not accepted_root.is_dir():
        raise ValueError("accepted root must be a real absolute directory")
    grouped: dict[str, list[dict[str, object]]] = {category: [] for category in CATEGORIES}
    for episode_root in sorted(accepted_root.iterdir(), key=lambda path: path.name):
        if episode_root.is_symlink():
            raise ValueError("accepted root contains a symlink")
        if not episode_root.is_dir():
            continue
        attempt_id = episode_root.name
        if not attempt_id or "/" in attempt_id or "\\" in attempt_id:
            raise ValueError("accepted episode ID is unsafe")
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
            or reset.get("garment_name") != garment
            or reset.get("schema_version") != 1
        ):
            raise ValueError("accepted episode is not a verified 12K seen-garment success")
        cloth_frame = (
            "usd_local_points_v1"
            if simulator_device == "cpu"
            else "physx_cloth_view_world_v1"
        )
        expected_schema = 3 if simulator_device == "cpu" else 2
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
            }
        )
    if any(not grouped[category] for category in CATEGORIES):
        raise ValueError("verified successes are required for every category")
    return grouped


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


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
    accepted_root: str | Path,
    output: str | Path,
    attempts_per_category: int = 50,
    attempts_by_category: Mapping[str, int] | None = None,
    acceptance_caps: Mapping[str, int] | None = None,
    seed_base: int = 50_000,
) -> dict[str, object]:
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
    grouped = _successes(Path(accepted_root))
    rows: list[dict[str, object]] = []
    seed_offset = 0
    for category_index, category in enumerate(CATEGORIES):
        parents = grouped[category]
        for index in range(attempt_counts[category]):
            parent = parents[index % len(parents)]
            strategy = "mild_geometry" if index % 5 < 3 else "strong_geometry"
            seed = seed_base + seed_offset + index
            replay_identity = {
                "schema_version": 1,
                "parent_episode_id": parent["parent_episode_id"],
                "category": category,
                "strategy": strategy,
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
                    "strategy": strategy,
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
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempts-per-category", type=int, default=50)
    parser.add_argument("--attempts-by-category-json")
    parser.add_argument("--acceptance-caps-json")
    parser.add_argument("--seed-base", type=int, default=50_000)
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
                accepted_root=args.accepted_root,
                output=args.output,
                attempts_per_category=args.attempts_per_category,
                attempts_by_category=attempts_by_category,
                acceptance_caps=acceptance_caps,
                seed_base=args.seed_base,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
