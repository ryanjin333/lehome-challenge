from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


CATEGORIES = ("top_long", "top_short", "pant_long", "pant_short")


def _write_success(
    root: Path,
    *,
    attempt_id: str,
    category: str,
    garment: str,
    simulator_device: str = "cpu",
) -> None:
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
            "asset_revision": "bea65fd960ad5a1bb3bd3fa77164b28001c08ef9",
            "campaign_round_id": "fresh-12k-source-20260827",
            "campaign_run_id": "fresh-run-20260827-a",
        },
        "randomization": {"strategy": "canonical"},
        "provenance": {
            "policy_artifact_sha256": "3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06",
            "simulator_device": simulator_device,
            "cloth_device": simulator_device,
            "renderer_device": "cuda:0",
            "camera_device": "cuda:0",
            "policy_device": "cuda:0",
        },
    }
    cloth_state_authority = (
        "usd_local_points_v1"
        if simulator_device == "cpu"
        else "physx_cloth_view_world_v1"
    )
    snapshot_schema_version = 3 if simulator_device == "cpu" else 2
    reset = {
        "schema_version": snapshot_schema_version,
        "cloth_state_authority": cloth_state_authority,
        "garment_name": garment,
        "robot_position": [0.0] * 12,
        "robot_velocity": [0.0] * 12,
        "cloth_position": [[0.0, 0.0, 0.0]],
        "cloth_velocity": [[0.0, 0.0, 0.0]],
        "rng_state": {},
        "randomization": {"strategy": "canonical"},
        "scene_state": {"garment_reset_pose": [0.0, 0.0, 0.67, 0.0, 0.0, 90.0]},
    }
    episode_path = raw / "episode.json"
    reset_path = snapshots / "reset.json"
    continuation_path = snapshots / "continuations" / "000016.json"
    continuation_path.parent.mkdir()
    episode_path.write_text(json.dumps(episode), encoding="utf-8")
    reset_path.write_text(json.dumps(reset), encoding="utf-8")
    continuation = {
        **reset,
        "randomization": {**reset["randomization"], "continuation_step": 16},
    }
    continuation_path.write_text(json.dumps(continuation), encoding="utf-8")
    annotations_path = raw / "annotations.jsonl"
    annotations_path.write_text(
        "".join(
            json.dumps({"step": step, "action": [0.0] * 12, "success": step == 19}) + "\n"
            for step in range(20)
        ),
        encoding="utf-8",
    )
    checksums = {}
    for path in (episode_path, reset_path, continuation_path, annotations_path):
        relative = path.relative_to(episode_root).as_posix()
        checksums[relative] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
        }
    (episode_root / "SHA256SUMS.json").write_text(
        json.dumps(checksums), encoding="utf-8"
    )


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _report_digest(report: dict[str, object]) -> str:
    body = dict(report)
    body.pop("report_sha256", None)
    return hashlib.sha256(_canonical(body)).hexdigest()


def _artifact_digest(episode_root: Path) -> str:
    entries = []
    for path in sorted(episode_root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.json":
            entries.append(
                {
                    "relative_path": path.relative_to(episode_root).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "byte_size": path.stat().st_size,
                }
            )
    return hashlib.sha256(_canonical(entries)).hexdigest()


def _refresh_checksum(episode_root: Path, relative: str) -> None:
    path = episode_root / relative
    checksums = json.loads((episode_root / "SHA256SUMS.json").read_text(encoding="utf-8"))
    checksums[relative] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size": path.stat().st_size}
    (episode_root / "SHA256SUMS.json").write_bytes(_canonical(checksums))


def _write_fresh_sources(accepted: Path) -> tuple[Path, Path]:
    """Write the authenticated source-evidence shape consumed by fresh mode."""

    matrix_rows = []
    receipt_root = accepted.parent / "hf-sync-receipts"
    receipt_root.mkdir()
    source_trials = []
    for episode_root in sorted(accepted.iterdir()):
        attempt_id = episode_root.name
        episode = json.loads(
            (episode_root / "raw" / attempt_id / "episode.json").read_text(encoding="utf-8")
        )
        identity = episode["identity"]
        matrix_rows.append(
            {
                "attempt_id": attempt_id,
                "trial_id": attempt_id,
                "category": identity["category"],
                "garment_name": identity["garment_name"],
                "release_stage": "seen",
                "strategy": "canonical",
                "campaign_kind": "fresh_12k_success_source_v1",
                "logical_stage": "fresh_success_source",
                "campaign_round_id": identity["campaign_round_id"],
                "campaign_run_id": identity["campaign_run_id"],
            }
        )
    matrix_path = accepted.parent / "fresh-source-matrix.json"
    matrix_path.write_bytes(_canonical(matrix_rows))
    matrix_sha256 = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    for episode_root in sorted(accepted.iterdir()):
        attempt_id = episode_root.name
        episode = json.loads(
            (episode_root / "raw" / attempt_id / "episode.json").read_text(encoding="utf-8")
        )
        identity = episode["identity"]
        artifact_sha256 = _artifact_digest(episode_root)
        receipt = {
            "schema_version": 1,
            "attempt_id": attempt_id,
            "repository": "ryanjin333/lehome-groot-n17-rollouts",
            "round_id": "fresh-12k-source-20260827",
            "run_id": "fresh-run-20260827-a",
            "remote_prefix": f"rollout-rounds/fresh-12k-source-20260827/{attempt_id}",
            "publication_ref": "main",
            "immutable_revision": "a" * 40,
            "entry_count": len(list(episode_root.rglob("*"))),
            "episode_sha256": artifact_sha256,
            "readback_verified": True,
        }
        receipt_path = receipt_root / f"{attempt_id}.sync.json"
        receipt_path.write_bytes(_canonical(receipt))
        source_trials.append(
            {
                "attempt_id": attempt_id,
                "category": identity["category"],
                "garment_name": identity["garment_name"],
                "accepted_success": True,
                "official_success": True,
                "outcome": "success",
                "simulator_device": "cpu",
                "cloth_device": "cpu",
                "renderer_device": "cuda:0",
                "camera_device": "cuda:0",
                "policy_device": "cuda:0",
                "safety_failure": False,
                "numerical_failure": False,
                "cloth_failure": False,
                "artifact_sha256": artifact_sha256,
                "hub_sync_receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                "remote_prefix": receipt["remote_prefix"],
                "campaign_round_id": "fresh-12k-source-20260827",
                "campaign_run_id": "fresh-run-20260827-a",
            }
        )
    report: dict[str, object] = {
        "schema_version": 1,
        "kind": "lehome_fresh_12k_success_source_report_v1",
        "campaign_kind": "fresh_12k_success_source_v1",
        "logical_stage": "fresh_success_source",
        "round_id": "fresh-12k-source-20260827",
        "run_id": "fresh-run-20260827-a",
        "matrix_sha256": matrix_sha256,
        "identity": {
            "policy_repo": "ryanjin333/lehome-groot-n17-models",
            "policy_revision": "30ac1a84da67b099e115ad147bcd61e9d60046d3",
            "policy_step": 12000,
            "policy_artifact_sha256": "3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06",
        },
        "trials": source_trials,
        "safety_failure": False,
    }
    report["report_sha256"] = _report_digest(report)
    report_path = accepted.parent / "fresh-source-report.json"
    report_path.write_bytes(_canonical(report))
    return report_path, matrix_path


def _append_authenticated_policy_failures(
    report_path: Path, matrix_path: Path, *, category: str, garment: str, count: int,
) -> None:
    """Add valid policy failures: they affect rate weighting but have no parent artifacts."""

    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for index in range(count):
        attempt_id = f"failed-{category}-{index}"
        matrix.append(
            {
                "attempt_id": attempt_id, "trial_id": attempt_id, "category": category,
                "garment_name": garment, "release_stage": "seen", "strategy": "canonical",
                "campaign_kind": "fresh_12k_success_source_v1", "logical_stage": "fresh_success_source",
                "campaign_round_id": report["round_id"], "campaign_run_id": report["run_id"],
            }
        )
        report["trials"].append(
            {
                "attempt_id": attempt_id, "category": category, "garment_name": garment,
                "accepted_success": False, "official_success": False, "outcome": "failure",
                "simulator_device": "cpu", "cloth_device": "cpu", "renderer_device": "cuda:0",
                "camera_device": "cuda:0", "policy_device": "cuda:0",
                "safety_failure": False, "numerical_failure": False, "cloth_failure": False,
                "remote_prefix": f"rollout-rounds/{report['round_id']}/{attempt_id}",
                "campaign_round_id": report["round_id"], "campaign_run_id": report["run_id"],
            }
        )
    matrix_path.write_bytes(_canonical(matrix))
    report["matrix_sha256"] = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    report["report_sha256"] = _report_digest(report)
    report_path.write_bytes(_canonical(report))


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
    assert all(row["replay_kind"] == "verified_success_early_snapshot_v1" for row in matrix)
    assert all(row["restore_snapshot_step"] == 16 for row in matrix)
    assert all(row["restore_snapshot"].endswith("/snapshots/continuations/000016.json") for row in matrix)
    assert all(row["restore_snapshot_cloth_frame"] == "usd_local_points_v1" for row in matrix)
    assert all(Path(row["restore_snapshot"]).is_absolute() for row in matrix)
    assert all(len(row["restore_snapshot_sha256"]) == 64 for row in matrix)
    assert {row["strategy"] for row in matrix} == {"mild_geometry", "strong_geometry"}
    assert receipt["matrix_sha256"] == hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    assert (tmp_path / "matrix.json.sha256").read_text(encoding="ascii") == receipt["matrix_sha256"] + "\n"


def test_builder_combines_verified_successes_from_multiple_campaign_roots(tmp_path: Path) -> None:
    from scripts.build_success_replay_matrix import build_success_replay_matrix

    first = tmp_path / "first" / "accepted"
    second = tmp_path / "second" / "accepted"
    for index, category in enumerate(CATEGORIES):
        _write_success(
            first if index % 2 == 0 else second,
            attempt_id=f"parent-{category}",
            category=category,
            garment=f"Garment_{index}",
        )

    matrix_path = tmp_path / "matrix.json"
    build_success_replay_matrix(
        accepted_roots=(first, second),
        output=matrix_path,
        attempts_per_category=1,
    )

    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    assert len(matrix) == 4
    assert {row["category"] for row in matrix} == set(CATEGORIES)


def test_builder_supports_exact_category_attempts_and_acceptance_caps(tmp_path: Path) -> None:
    from lehome.flywheel.recovery_collection import (
        load_attempt_matrix,
        validate_success_replay_descriptor,
    )
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
    attempts = {"top_long": 12, "top_short": 4, "pant_long": 24, "pant_short": 0}
    caps = {"top_long": 7, "top_short": 3, "pant_long": 9, "pant_short": 0}

    receipt = build_success_replay_matrix(
        accepted_root=accepted,
        output=matrix_path,
        attempts_by_category=attempts,
        acceptance_caps=caps,
        seed_base=70_000,
    )

    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    assert len(matrix) == 40
    assert {
        category: sum(row["category"] == category for row in matrix)
        for category in CATEGORIES
    } == attempts
    assert all(row["category_acceptance_cap"] == caps[row["category"]] for row in matrix)
    assert receipt["attempts_by_category"] == attempts
    assert receipt["acceptance_caps"] == caps
    assert validate_success_replay_descriptor(matrix_path) == matrix
    assert load_attempt_matrix(matrix_path) == matrix


def test_builder_rejects_caps_without_attempts(tmp_path: Path) -> None:
    from scripts.build_success_replay_matrix import build_success_replay_matrix

    accepted = tmp_path / "accepted"
    for index, category in enumerate(CATEGORIES):
        _write_success(
            accepted,
            attempt_id=f"parent-{category}",
            category=category,
            garment=f"{category.title().replace('_', '_')}_Seen_{index}",
        )

    with pytest.raises(ValueError, match="acceptance cap"):
        build_success_replay_matrix(
            accepted_root=accepted,
            output=tmp_path / "matrix.json",
            attempts_by_category={"top_long": 0, "top_short": 1, "pant_long": 1, "pant_short": 1},
            acceptance_caps={"top_long": 1, "top_short": 1, "pant_long": 1, "pant_short": 1},
        )


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


def test_builder_rejects_a_legacy_reset_from_a_different_asset_revision(tmp_path: Path) -> None:
    from scripts.build_success_replay_matrix import build_success_replay_matrix

    accepted = tmp_path / "accepted"
    for index, category in enumerate(CATEGORIES):
        _write_success(
            accepted,
            attempt_id=f"parent-{category}",
            category=category,
            garment=f"Garment_{index}",
        )
    episode_root = accepted / "parent-pant_short"
    episode_path = episode_root / "raw" / "parent-pant_short" / "episode.json"
    payload = json.loads(episode_path.read_text(encoding="utf-8"))
    payload["identity"]["asset_revision"] = "0" * 40
    episode_path.write_text(json.dumps(payload), encoding="utf-8")
    checksums = json.loads((episode_root / "SHA256SUMS.json").read_text(encoding="utf-8"))
    checksums["raw/parent-pant_short/episode.json"] = {
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


def test_builder_accepts_successes_from_any_cuda_worker_index(tmp_path: Path) -> None:
    from scripts.build_success_replay_matrix import build_success_replay_matrix

    accepted = tmp_path / "accepted"
    for index, category in enumerate(CATEGORIES):
        _write_success(
            accepted,
            attempt_id=f"parent-{category}",
            category=category,
            garment=f"Garment_{index}",
            simulator_device=f"cuda:{index}",
        )

    matrix_path = tmp_path / "matrix.json"
    build_success_replay_matrix(
        accepted_root=accepted,
        output=matrix_path,
        attempts_per_category=1,
    )

    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    assert len(matrix) == 4
    assert all(row["restore_snapshot_cloth_frame"] == "physx_cloth_view_world_v1" for row in matrix)


def test_fresh_visual_only_mode_binds_authenticated_cpu_sources_and_is_deterministic(
    tmp_path: Path,
) -> None:
    from lehome.flywheel.recovery_collection import validate_success_replay_descriptor
    from scripts.build_success_replay_matrix import build_success_replay_matrix

    accepted = tmp_path / "accepted"
    for index, category in enumerate(CATEGORIES):
        _write_success(
            accepted,
            attempt_id=f"fresh-{category}",
            category=category,
            garment=f"{category.title().replace('_', '_')}_Seen_{index}",
        )
    report, source_matrix = _write_fresh_sources(accepted)
    first, second = tmp_path / "first.json", tmp_path / "second.json"

    receipt = build_success_replay_matrix(
        accepted_root=accepted,
        output=first,
        source_reports=(report,),
        source_matrices=(source_matrix,),
        strategy="visual_only",
        attempt_cap_per_category=100,
        acceptance_cap_per_category=50,
        max_attempts=400,
        target_accepted=200,
        rng_seed=20260827400,
    )
    build_success_replay_matrix(
        accepted_root=accepted,
        output=second,
        source_reports=(report,),
        source_matrices=(source_matrix,),
        strategy="visual_only",
        attempt_cap_per_category=100,
        acceptance_cap_per_category=50,
        max_attempts=400,
        target_accepted=200,
        rng_seed=20260827400,
    )

    rows = json.loads(first.read_text(encoding="utf-8"))
    assert first.read_bytes() == second.read_bytes()
    assert len(rows) == 400
    assert receipt["shortages"] == []
    assert {row["strategy"] for row in rows} == {"visual_only"}
    assert {row["category_acceptance_cap"] for row in rows} == {50}
    assert {row["category"] for row in rows} == set(CATEGORIES)
    assert validate_success_replay_descriptor(first) == rows
    assert all(row["restore_snapshot_step"] == 16 for row in rows)
    for row in rows:
        assert set(
            (
                "source_episode_sha256",
                "source_reset_sha256",
                "source_annotations_sha256",
                "source_continuation_snapshot_sha256",
                "source_state_fingerprint",
                "source_report_sha256",
                "source_matrix_sha256",
                "source_receipt_sha256",
            )
        ) <= set(row)


def test_fresh_visual_only_sampling_uses_all_authenticated_policy_outcomes_for_garment_weights(
    tmp_path: Path,
) -> None:
    from scripts.build_success_replay_matrix import _fresh_source_parents, build_success_replay_matrix

    accepted = tmp_path / "accepted"
    _write_success(accepted, attempt_id="fresh-top-long-low", category="top_long", garment="Top_Long_Seen_0")
    _write_success(accepted, attempt_id="fresh-top-long-high", category="top_long", garment="Top_Long_Seen_1")
    for index, category in enumerate(CATEGORIES[1:], start=2):
        _write_success(accepted, attempt_id=f"fresh-{category}", category=category, garment=f"{category.title().replace('_', '_')}_Seen_{index}")
    report, source_matrix = _write_fresh_sources(accepted)
    _append_authenticated_policy_failures(
        report, source_matrix, category="top_long", garment="Top_Long_Seen_0", count=3,
    )

    grouped, _ = _fresh_source_parents(
        accepted_roots=(accepted,), source_reports=(report,), source_matrices=(source_matrix,),
    )
    rates = {str(parent["garment"]): float(parent["fresh_success_rate"]) for parent in grouped["top_long"]}
    assert rates == {"Top_Long_Seen_0": 0.25, "Top_Long_Seen_1": 1.0}
    assert {garment: max(1 - rate, 0.01) for garment, rate in rates.items()} == {
        "Top_Long_Seen_0": 0.75, "Top_Long_Seen_1": 0.01,
    }

    first, second = tmp_path / "first.json", tmp_path / "second.json"
    for output in (first, second):
        build_success_replay_matrix(
            accepted_root=accepted, output=output, source_reports=(report,), source_matrices=(source_matrix,),
            strategy="visual_only", attempt_cap_per_category=100, acceptance_cap_per_category=50,
            max_attempts=400, target_accepted=200, rng_seed=20260827400,
        )
    rows = json.loads(first.read_text(encoding="utf-8"))
    top_long_parents = [row["parent_episode_id"] for row in rows if row["category"] == "top_long"]
    assert first.read_bytes() == second.read_bytes()
    assert top_long_parents.count("fresh-top-long-low") == 99
    assert top_long_parents.count("fresh-top-long-high") == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "old-round",
        "wrong-matrix",
        "not-success",
        "missing-receipt",
        "receipt-mismatch",
        "wrong-policy",
        "wrong-campaign-kind",
        "wrong-logical-stage",
        "cuda-cloth",
        "missing-step-16",
        "noncanonical-parent",
        "noncanonical-step-16",
        "old-episode-relabel",
        "safety",
        "numerical",
        "cloth",
        "mixed-garment",
        "unreported-success",
    ],
)
def test_fresh_visual_only_mode_rejects_every_untrusted_source_boundary(
    tmp_path: Path, mutation: str
) -> None:
    from scripts.build_success_replay_matrix import build_success_replay_matrix

    accepted = tmp_path / "accepted"
    for index, category in enumerate(CATEGORIES):
        _write_success(
            accepted,
            attempt_id=f"fresh-{category}",
            category=category,
            garment=f"{category.title().replace('_', '_')}_Seen_{index}",
        )
    report_path, matrix_path = _write_fresh_sources(accepted)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    trial = report["trials"][0]
    episode_root = accepted / trial["attempt_id"]
    episode_path = episode_root / "raw" / trial["attempt_id"] / "episode.json"

    if mutation == "old-round":
        report["round_id"] = "success-replay-12k-round-1"
    elif mutation == "wrong-matrix":
        report["matrix_sha256"] = "0" * 64
    elif mutation == "not-success":
        trial["accepted_success"] = False
    elif mutation == "missing-receipt":
        (accepted.parent / "hf-sync-receipts" / f"{trial['attempt_id']}.sync.json").unlink()
    elif mutation == "receipt-mismatch":
        receipt = accepted.parent / "hf-sync-receipts" / f"{trial['attempt_id']}.sync.json"
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        payload["episode_sha256"] = "0" * 64
        receipt.write_bytes(_canonical(payload))
    elif mutation == "wrong-policy":
        report["identity"]["policy_revision"] = "0" * 40
    elif mutation == "wrong-campaign-kind":
        report["campaign_kind"] = "other_campaign_v1"
    elif mutation == "wrong-logical-stage":
        report["logical_stage"] = "other_stage"
    elif mutation == "cuda-cloth":
        trial["simulator_device"] = "cuda:0"
    elif mutation == "missing-step-16":
        continuation = episode_root / "raw" / trial["attempt_id"] / "snapshots" / "continuations" / "000016.json"
        continuation.unlink()
    elif mutation == "noncanonical-parent":
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        episode["randomization"] = {"strategy": "mild_geometry"}
        episode_path.write_bytes(_canonical(episode))
        _refresh_checksum(episode_root, f"raw/{trial['attempt_id']}/episode.json")
    elif mutation == "noncanonical-step-16":
        continuation = episode_root / "raw" / trial["attempt_id"] / "snapshots" / "continuations" / "000016.json"
        payload = json.loads(continuation.read_text(encoding="utf-8"))
        payload["randomization"] = {"strategy": "strong_geometry", "continuation_step": 16}
        continuation.write_bytes(_canonical(payload))
        _refresh_checksum(episode_root, f"raw/{trial['attempt_id']}/snapshots/continuations/000016.json")
    elif mutation == "old-episode-relabel":
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        episode["identity"]["campaign_round_id"] = "success-replay-12k-round-1"
        episode_path.write_bytes(_canonical(episode))
        _refresh_checksum(episode_root, f"raw/{trial['attempt_id']}/episode.json")
    elif mutation in {"safety", "numerical", "cloth"}:
        trial[f"{mutation}_failure"] = True
    elif mutation == "mixed-garment":
        trial["garment_name"] = "Top_Long_Seen_999"
    elif mutation == "unreported-success":
        report["trials"] = report["trials"][1:]
    report["report_sha256"] = _report_digest(report)
    report_path.write_bytes(_canonical(report))

    with pytest.raises(ValueError):
        build_success_replay_matrix(
            accepted_root=accepted,
            output=tmp_path / "matrix.json",
            source_reports=(report_path,),
            source_matrices=(matrix_path,),
            strategy="visual_only",
            attempt_cap_per_category=100,
            acceptance_cap_per_category=50,
            max_attempts=400,
            target_accepted=200,
            rng_seed=20260827400,
        )


def test_fresh_visual_only_mode_emits_a_category_shortage_without_borrowing_source(
    tmp_path: Path,
) -> None:
    from scripts.build_success_replay_matrix import build_success_replay_matrix

    accepted = tmp_path / "accepted"
    for index, category in enumerate(CATEGORIES[:-1]):
        _write_success(
            accepted,
            attempt_id=f"fresh-{category}",
            category=category,
            garment=f"{category.title().replace('_', '_')}_Seen_{index}",
        )
    report, source_matrix = _write_fresh_sources(accepted)
    output = tmp_path / "matrix.json"
    receipt = build_success_replay_matrix(
        accepted_root=accepted,
        output=output,
        source_reports=(report,),
        source_matrices=(source_matrix,),
        strategy="visual_only",
        attempt_cap_per_category=100,
        acceptance_cap_per_category=50,
        max_attempts=400,
        target_accepted=200,
        rng_seed=20260827400,
    )

    rows = json.loads(output.read_text(encoding="utf-8"))
    assert len(rows) == 300
    assert all(row["category"] != "pant_short" for row in rows)
    assert receipt["shortages"] == [{"category": "pant_short", "reason": "no_eligible_source"}]


def test_fresh_visual_only_mode_binds_each_parent_to_its_own_report_matrix_pair(tmp_path: Path) -> None:
    from scripts.build_success_replay_matrix import build_success_replay_matrix

    first, second = tmp_path / "one" / "accepted", tmp_path / "two" / "accepted"
    for index, category in enumerate(CATEGORIES):
        root = first if index < 2 else second
        _write_success(root, attempt_id=f"fresh-{category}", category=category, garment=f"{category.title().replace('_', '_')}_Seen_{index}")
    first_report, first_matrix = _write_fresh_sources(first)
    second_report, second_matrix = _write_fresh_sources(second)

    output = tmp_path / "matrix.json"
    build_success_replay_matrix(
        accepted_roots=(first, second), output=output,
        source_reports=(first_report, second_report), source_matrices=(first_matrix, second_matrix),
        strategy="visual_only", attempt_cap_per_category=100, acceptance_cap_per_category=50,
        max_attempts=400, target_accepted=200, rng_seed=20260827400,
    )

    rows = json.loads(output.read_text(encoding="utf-8"))
    assert {row["source_report_path"] for row in rows} == {str(first_report), str(second_report)}
    assert {row["source_matrix_path"] for row in rows} == {str(first_matrix), str(second_matrix)}
