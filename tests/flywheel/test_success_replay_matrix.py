from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


CATEGORIES = ("top_long", "top_short", "pant_long", "pant_short")


def _write_success(root: Path, *, attempt_id: str, category: str, garment: str) -> None:
    episode_root = root / attempt_id
    raw = episode_root / "raw" / attempt_id
    snapshots = raw / "snapshots"
    snapshots.mkdir(parents=True)
    episode = {
        "episode_id": attempt_id,
        "accepted_success": True,
        "outcome": "success",
        "identity": {
            "episode_id": attempt_id,
            "policy_repo": "ryanjin333/lehome-groot-n17-models",
            "category": category,
            "garment_name": garment,
            "release_stage": "seen",
            "policy_revision": "30ac1a84da67b099e115ad147bcd61e9d60046d3",
            "policy_step": 12000,
        },
        "provenance": {
            "policy_artifact_sha256": "3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06"
        },
    }
    reset = {
        "schema_version": 1,
        "garment_name": garment,
        "robot_position": [0.0] * 12,
        "robot_velocity": [0.0] * 12,
        "cloth_position": [[0.0, 0.0, 0.0]],
        "cloth_velocity": [[0.0, 0.0, 0.0]],
        "rng_state": {},
    }
    episode_path = raw / "episode.json"
    reset_path = snapshots / "reset.json"
    episode_path.write_text(json.dumps(episode), encoding="utf-8")
    reset_path.write_text(json.dumps(reset), encoding="utf-8")
    checksums = {}
    for path in (episode_path, reset_path):
        relative = path.relative_to(episode_root).as_posix()
        checksums[relative] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
        }
    (episode_root / "SHA256SUMS.json").write_text(
        json.dumps(checksums), encoding="utf-8"
    )


def test_builder_creates_balanced_lineage_bound_success_replays(tmp_path: Path) -> None:
    from scripts.build_success_replay_matrix import build_success_replay_matrix

    accepted = tmp_path / "accepted"
    for index, category in enumerate(CATEGORIES):
        _write_success(
            accepted,
            attempt_id=f"parent-{category}",
            category=category,
            garment=f"{category.title().replace('_', '_')}_Seen_{index}",
        )
    matrix_path = tmp_path / "matrix.json"

    receipt = build_success_replay_matrix(
        accepted_root=accepted,
        output=matrix_path,
        attempts_per_category=5,
        seed_base=50_000,
    )

    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    assert len(matrix) == 20
    assert len({row["attempt_id"] for row in matrix}) == 20
    assert {category: sum(row["category"] == category for row in matrix) for category in CATEGORIES} == {
        category: 5 for category in CATEGORIES
    }
    assert all(row["parent_episode_id"] == row["lineage_id"] for row in matrix)
    assert all(row["replay_kind"] == "verified_success_reset_v1" for row in matrix)
    assert all(Path(row["restore_snapshot"]).is_absolute() for row in matrix)
    assert all(len(row["restore_snapshot_sha256"]) == 64 for row in matrix)
    assert {row["strategy"] for row in matrix} == {"mild_geometry", "strong_geometry"}
    assert receipt["matrix_sha256"] == hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    assert (tmp_path / "matrix.json.sha256").read_text(encoding="ascii") == receipt["matrix_sha256"] + "\n"


def test_builder_is_deterministic_and_rejects_tampered_or_incomplete_sources(
    tmp_path: Path,
) -> None:
    from scripts.build_success_replay_matrix import build_success_replay_matrix

    accepted = tmp_path / "accepted"
    for index, category in enumerate(CATEGORIES):
        _write_success(
            accepted,
            attempt_id=f"parent-{category}",
            category=category,
            garment=f"Garment_{index}",
        )
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    build_success_replay_matrix(
        accepted_root=accepted, output=first, attempts_per_category=3, seed_base=9_000
    )
    build_success_replay_matrix(
        accepted_root=accepted, output=second, attempts_per_category=3, seed_base=9_000
    )
    assert first.read_bytes() == second.read_bytes()

    reset = next((accepted / "parent-top_short").rglob("reset.json"))
    reset_bytes = reset.read_bytes()
    reset.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        build_success_replay_matrix(
            accepted_root=accepted,
            output=tmp_path / "tampered.json",
            attempts_per_category=1,
        )
    reset.write_bytes(reset_bytes)

    for path in (accepted / "parent-pant_short").rglob("*"):
        if path.is_file():
            path.unlink()
    with pytest.raises(ValueError, match="missing or unsafe|every category"):
        build_success_replay_matrix(
            accepted_root=accepted,
            output=tmp_path / "incomplete.json",
            attempts_per_category=1,
        )


def test_builder_rejects_symlinked_sources_and_never_partially_clobbers_outputs(
    tmp_path: Path,
) -> None:
    from scripts.build_success_replay_matrix import build_success_replay_matrix

    accepted = tmp_path / "accepted"
    for index, category in enumerate(CATEGORIES):
        _write_success(
            accepted,
            attempt_id=f"parent-{category}",
            category=category,
            garment=f"Garment_{index}",
        )

    raw = accepted / "parent-top_long" / "raw"
    raw_target = tmp_path / "raw-target"
    raw.rename(raw_target)
    raw.symlink_to(raw_target, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe"):
        build_success_replay_matrix(
            accepted_root=accepted,
            output=tmp_path / "unsafe.json",
            attempts_per_category=1,
        )

    raw.unlink()
    raw_target.rename(raw)
    output = tmp_path / "matrix.json"
    receipt = tmp_path / "matrix.json.sha256"
    receipt.write_text("already exists\n", encoding="ascii")
    with pytest.raises(FileExistsError, match="absent"):
        build_success_replay_matrix(
            accepted_root=accepted,
            output=output,
            attempts_per_category=1,
        )
    assert not output.exists()
    assert receipt.read_text(encoding="ascii") == "already exists\n"


def test_builder_rejects_a_success_not_from_the_pinned_original_12k_repo(
    tmp_path: Path,
) -> None:
    from scripts.build_success_replay_matrix import build_success_replay_matrix

    accepted = tmp_path / "accepted"
    for index, category in enumerate(CATEGORIES):
        _write_success(
            accepted,
            attempt_id=f"parent-{category}",
            category=category,
            garment=f"Garment_{index}",
        )
    episode_root = accepted / "parent-top_short"
    episode_path = episode_root / "raw" / "parent-top_short" / "episode.json"
    payload = json.loads(episode_path.read_text(encoding="utf-8"))
    payload["identity"]["policy_repo"] = "regressed-2k"
    episode_path.write_text(json.dumps(payload), encoding="utf-8")
    checksums = json.loads((episode_root / "SHA256SUMS.json").read_text(encoding="utf-8"))
    checksums["raw/parent-top_short/episode.json"] = {
        "sha256": hashlib.sha256(episode_path.read_bytes()).hexdigest(),
        "size": episode_path.stat().st_size,
    }
    (episode_root / "SHA256SUMS.json").write_text(json.dumps(checksums), encoding="utf-8")

    with pytest.raises(ValueError, match="verified 12K"):
        build_success_replay_matrix(
            accepted_root=accepted,
            output=tmp_path / "matrix.json",
            attempts_per_category=1,
        )
