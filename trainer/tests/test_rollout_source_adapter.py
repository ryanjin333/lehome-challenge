from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")


def _episode_digest(root: Path) -> str:
    """Match the rollout uploader's digest (all files except SHA256SUMS)."""
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.relative_to(root).as_posix() != "SHA256SUMS.json":
            payload = path.read_bytes()
            rows.append({
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "byte_size": len(payload),
            })
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _source_round(tmp_path: Path, *, round_id: str, attempt_id: str, category: str) -> tuple[Path, Path, Path]:
    from lehome.flywheel.artifacts import build_sha256_manifest
    from lehome_train.io import canonical_json_sha256

    root = tmp_path / round_id / "accepted"
    package = root / attempt_id
    raw = package / "raw" / attempt_id
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "annotations.jsonl").write_text(
        "".join(json.dumps({"state": [0.0] * 12, "action": [0.0] * 12}) + "\n" for _ in range(16)),
        encoding="utf-8",
    )
    _write(raw / "episode.json", {
        "episode_id": attempt_id,
        "accepted_success": True,
        "outcome": "success",
        "terminal_reason": "success",
        "bc_target_count": 0,
        "provenance": {
            "execution_backend": "policy_server",
            "execution_mode": "policy_server",
            "parity_stage": "persistent_collection",
            "policy_artifact_sha256": "c" * 64,
            "policy_device": "cuda:0",
            "simulator_device": "cpu",
        },
        "identity": {
            "release_stage": "seen", "category": category,
            "garment_name": f"{category}-seen",
        },
    })
    _write(raw / "SHA256SUMS.json", build_sha256_manifest(raw))
    _write(package / "flywheel-manifest.json", {"attempt_id": attempt_id})
    _write(package / "worker-receipt.json", {"attempt_id": attempt_id})
    _write(package / "SHA256SUMS.json", build_sha256_manifest(package))
    digest = _episode_digest(package)
    revision = "a" * 40
    receipts = tmp_path / round_id / "hf-sync-receipts"
    _write(receipts / f"{attempt_id}.sync.json", {
        "schema_version": 1, "attempt_id": attempt_id,
        "repository": "ryanjin333/lehome-groot-n17-rollouts", "round_id": round_id,
        "remote_prefix": f"rollout-rounds/{round_id}/{attempt_id}",
        "publication_ref": "main", "immutable_revision": revision,
        "entry_count": 2, "episode_sha256": digest, "readback_verified": True,
    })
    body = {
        "round_id": round_id,
        "repository": "ryanjin333/lehome-groot-n17-rollouts",
        "episode_sha256s": {attempt_id: digest},
        "immutable_revisions": {attempt_id: revision},
    }
    seal = tmp_path / round_id / "seal.json"
    _write(seal, {
        "schema_version": 2, "kind": "rollout_round_seal", **body,
        "episode_count": 1, "readback_verified": True,
        "seal_sha256": canonical_json_sha256(body),
    })
    return root, receipts, seal


def test_adapter_combines_two_sealed_rounds_with_original_ids_and_a_bounded_selection(
    tmp_path: Path,
) -> None:
    from lehome_train.groot.rollout_source_adapter import (
        build_rollout_source,
        validate_derived_rollout_source,
    )
    from lehome_train.io import canonical_json_sha256, sha256_file

    fresh, fresh_receipts, fresh_seal = _source_round(
        tmp_path, round_id="balanced-round", attempt_id="fresh-attempt", category="top_short",
    )
    replay, replay_receipts, replay_seal = _source_round(
        tmp_path, round_id="replay-round", attempt_id="replay-attempt", category="pant_short",
    )
    destination = tmp_path / "round-1"

    receipt = build_rollout_source(
        rounds=(
            {"root": str(fresh), "receipts_root": str(fresh_receipts), "seal_path": str(fresh_seal)},
            {"root": str(replay), "receipts_root": str(replay_receipts), "seal_path": str(replay_seal)},
        ),
        destination=destination, runtime_round=2,
    )

    selected = json.loads((destination / "selected-150.json").read_text(encoding="utf-8"))
    rows = selected["selected_bindings"]
    assert selected["schema_version"] == 2
    assert selected["selected_count"] == 2
    assert selected["max_selected_count"] == 150
    assert len({row["attempt_id"] for row in rows}) == 2
    assert {row["attempt_id"] for row in rows} == {"fresh-attempt", "replay-attempt"}
    assert selected["selection_sha256"] == canonical_json_sha256({
        "schema_version": 2, "selected_count": 2, "max_selected_count": 150,
        "selected_bindings": rows,
    })
    assert receipt["selected_count"] == 2
    assert receipt["runtime_prefix"] == "rollouts/round-2"

    lineage = json.loads((destination / "source-lineage.json").read_text(encoding="utf-8"))
    assert {row["source_round_id"] for row in lineage["episodes"]} == {"balanced-round", "replay-round"}
    for row in rows:
        raw = destination / "raw" / row["attempt_id"]
        episode = json.loads((raw / "episode.json").read_text(encoding="utf-8"))
        assert episode["episode_id"] == row["episode_id"] == row["attempt_id"]
        assert sha256_file(raw / "SHA256SUMS.json") == row["episode_manifest_sha256"]
    validate_derived_rollout_source(
        destination,
        selected={row["attempt_id"]: row["episode_manifest_sha256"] for row in rows},
        campaign_receipt=json.loads((destination / "campaign-receipt.json").read_text(encoding="utf-8")),
    )
    from lehome_train.groot.runtime_mixture_builder import validate_selected_raw_roots

    validate_selected_raw_roots(
        destination,
        {row["attempt_id"]: row["episode_manifest_sha256"] for row in rows},
        json.loads((destination / "campaign-receipt.json").read_text(encoding="utf-8")),
    )


def test_adapter_rejects_cross_round_episode_id_collision_before_writing(tmp_path: Path) -> None:
    from lehome_train.groot.rollout_source_adapter import build_rollout_source

    fresh, fresh_receipts, fresh_seal = _source_round(
        tmp_path, round_id="balanced-round", attempt_id="same-attempt", category="top_short",
    )
    replay, replay_receipts, replay_seal = _source_round(
        tmp_path, round_id="replay-round", attempt_id="same-attempt", category="pant_short",
    )
    destination = tmp_path / "round-1"

    with pytest.raises(ValueError, match="collision"):
        build_rollout_source(
            rounds=(
                {"root": str(fresh), "receipts_root": str(fresh_receipts), "seal_path": str(fresh_seal)},
                {"root": str(replay), "receipts_root": str(replay_receipts), "seal_path": str(replay_seal)},
            ),
            destination=destination, runtime_round=2,
        )
    assert not destination.exists()


@pytest.mark.parametrize("mutation", ["seal_digest", "receipt_digest", "symlink"])
def test_adapter_fails_closed_before_writing_destination_on_source_provenance_drift(
    tmp_path: Path, mutation: str,
) -> None:
    from lehome_train.groot.rollout_source_adapter import build_rollout_source

    root, receipts, seal = _source_round(
        tmp_path, round_id="balanced-round", attempt_id="attempt", category="top_short",
    )
    if mutation == "seal_digest":
        document = json.loads(seal.read_text(encoding="utf-8"))
        document["seal_sha256"] = "0" * 64
        _write(seal, document)
    elif mutation == "receipt_digest":
        receipt = receipts / "attempt.sync.json"
        document = json.loads(receipt.read_text(encoding="utf-8"))
        document["episode_sha256"] = "0" * 64
        _write(receipt, document)
    else:
        (root / "attempt" / "raw" / "attempt" / "annotations.jsonl").unlink()
        (root / "attempt" / "raw" / "attempt" / "annotations.jsonl").symlink_to("episode.json")

    destination = tmp_path / "round-1"
    with pytest.raises(ValueError, match="seal|readback|digest|unsafe|manifest|symlink"):
        build_rollout_source(
            rounds=({"root": str(root), "receipts_root": str(receipts), "seal_path": str(seal)},),
            destination=destination, runtime_round=2,
        )
    assert not destination.exists()


def test_derived_source_revalidation_rejects_post_adapter_origin_mutation(tmp_path: Path) -> None:
    from lehome_train.groot.rollout_source_adapter import (
        build_rollout_source,
        validate_derived_rollout_source,
    )

    root, receipts, seal = _source_round(
        tmp_path, round_id="balanced-round", attempt_id="attempt", category="top_short",
    )
    destination = tmp_path / "round-1"
    build_rollout_source(
        rounds=({"root": str(root), "receipts_root": str(receipts), "seal_path": str(seal)},),
        destination=destination, runtime_round=2,
    )
    selected = json.loads((destination / "selected-150.json").read_text(encoding="utf-8"))
    bindings = {row["attempt_id"]: row["episode_manifest_sha256"] for row in selected["selected_bindings"]}
    (destination / "raw" / next(iter(bindings)) / "origin-episode.json").write_text("tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="origin|digest|hash"):
        validate_derived_rollout_source(
            destination, selected=bindings,
            campaign_receipt=json.loads((destination / "campaign-receipt.json").read_text(encoding="utf-8")),
        )


def test_plan_generator_emits_h16_train_validation_lineage_and_binds_every_input(
    tmp_path: Path,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from lehome_train.groot.rollout_source_adapter import (
        build_rollout_source,
        build_runtime_plan,
    )
    from lehome_train.io import canonical_json_bytes, canonical_json_sha256, sha256_file

    fresh, fresh_receipts, fresh_seal = _source_round(
        tmp_path, round_id="balanced-round", attempt_id="fresh-attempt", category="top_short",
    )
    replay, replay_receipts, replay_seal = _source_round(
        tmp_path, round_id="replay-round", attempt_id="replay-attempt", category="pant_short",
    )
    campaign = tmp_path / "round-1"
    build_rollout_source(
        rounds=(
            {"root": str(fresh), "receipts_root": str(fresh_receipts), "seal_path": str(fresh_seal)},
            {"root": str(replay), "receipts_root": str(replay_receipts), "seal_path": str(replay_seal)},
        ), destination=campaign, runtime_round=2,
    )
    organizer = tmp_path / "bc"
    _write(organizer / "manifest.json", {"train_episode_ids": ["0"], "validation_episode_ids": ["1"]})
    for episode_id in (0, 1):
        data = organizer / f"data/chunk-000/episode_{episode_id:06d}.parquet"
        data.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({
            "observation.state": [[float(episode_id)] * 12 for _ in range(16)],
            "action": [[float(episode_id)] * 12 for _ in range(16)],
        }), data)
    index = organizer / "garment-index.json"
    index.write_bytes(canonical_json_bytes({
        "schema_version": 1, "kind": "lehome_bc_garment_index",
        "episodes": [
            {"episode_id": "0", "garment_name": "Top_Short_Seen_0"},
            {"episode_id": "1", "garment_name": "Pant_Short_Seen_0"},
        ],
    }))
    config = tmp_path / "experiment-config.json"
    profile = {
        "schema_version": 1,
        "kind": "lehome_runtime_profile",
        "mixture_weights": {"bc": 70, "rollout": 30, "dagger": 0},
        "training": {
            "action_horizon": 16, "global_batch_size": 64, "physical_batch_size": 64,
            "max_steps": 2000, "local_checkpoint_steps": [500, 1000, 1500, 2000],
            "hf_checkpoint_steps": [1000, 2000], "loader_candidates": [0, 4, 8, 12, 16],
        },
        "held_out_garments": [
            "Top_Long_Unseen_1", "Top_Short_Unseen_1",
            "Pant_Long_Unseen_1", "Pant_Short_Unseen_1",
        ],
    }
    config.write_bytes(canonical_json_bytes(profile))
    destination = tmp_path / "runtime-plan.json"

    receipt = build_runtime_plan(
        organizer_root=organizer, campaign_root=campaign,
        garment_index_path=index, garment_index_sha256=sha256_file(index),
        experiment_config_path=config, destination=destination,
    )

    state = json.loads(destination.read_text(encoding="utf-8"))
    plan = state["plan"]
    assert state["plan_sha256"] == plan["sha256"] == canonical_json_sha256({key: value for key, value in plan.items() if key != "sha256"})
    assert receipt["plan_sha256"] == state["plan_sha256"]
    assert plan["input_bindings"]["organizer_manifest_sha256"] == sha256_file(organizer / "manifest.json")
    assert plan["input_bindings"]["campaign_receipt_sha256"] == sha256_file(campaign / "campaign-receipt.json")
    assert plan["input_bindings"]["experiment_config_sha256"] == sha256_file(config)
    selections = plan["selected_frame_ranges"]
    assert {row["source_kind"] for row in selections} == {"organizer", "flywheel"}
    assert {row["split"] for row in selections} == {"train", "validation"}
    assert {row["raw_episode_id"] for row in selections if row["source_kind"] == "flywheel"} == {"fresh-attempt", "replay-attempt"}
    assert all(row["raw_frame_stop"] - row["raw_frame_start"] == 16 for row in selections)
    from types import MappingProxyType, SimpleNamespace
    from lehome_train.groot.runtime_mixture_builder import validate_generated_runtime_plan_bindings

    validate_generated_runtime_plan_bindings(
        plan, organizer_root=organizer, campaign_root=campaign,
        organizer_manifest=organizer / "manifest.json",
        campaign_receipt=campaign / "campaign-receipt.json",
        selected_bindings=campaign / "selected-150.json", garment_index=index,
        experiment=SimpleNamespace(
            weights={"bc": 70, "rollout": 30, "dagger": 0},
            held_out_garments=tuple(profile["held_out_garments"]),
            raw=MappingProxyType({"training": profile["training"]}),
        ),
    )

    profile["mixture_weights"] = {"bc": 80, "rollout": 20, "dagger": 0}
    changed_config = tmp_path / "changed-profile.json"
    changed_config.write_bytes(canonical_json_bytes(profile))
    from lehome_train.groot.experiment_manifest import load_runtime_profile

    weights, quotas = load_runtime_profile(changed_config)
    assert dict(weights) == {"bc": 80, "rollout": 20, "dagger": 0}
    assert dict(quotas) == {"bc": 51, "rollout": 13, "dagger": 0}


def test_generated_plan_rejects_an_experiment_profile_mismatch(tmp_path: Path) -> None:
    from types import MappingProxyType, SimpleNamespace

    from lehome_train.groot.runtime_mixture_builder import validate_generated_runtime_plan_bindings

    digest = "0" * 64
    plan = {
        "schema_version": 1,
        "kind": "runtime_mixture_plan",
        "input_bindings": {
            "organizer_manifest_sha256": digest,
            "organizer_tree_sha256": digest,
            "campaign_receipt_sha256": digest,
            "campaign_tree_sha256": digest,
            "selected_bindings_sha256": digest,
            "source_lineage_sha256": digest,
            "garment_index_sha256": digest,
            "experiment_config_sha256": "1" * 64,
            "runtime_schedule": {"bc": 70, "rollout": 30, "batch_size": 64, "action_horizon": 16},
        },
        "selected_frame_ranges": [],
        "sha256": digest,
    }
    organizer, campaign = tmp_path / "bc", tmp_path / "rollout"
    organizer.mkdir()
    campaign.mkdir()
    files = {
        "organizer-manifest": tmp_path / "organizer-manifest.json",
        "campaign-receipt": tmp_path / "campaign-receipt.json",
        "selected": tmp_path / "selected.json",
        "garment-index": tmp_path / "garment-index.json",
        "source-lineage": campaign / "source-lineage.json",
    }
    for path in files.values():
        path.write_text("{}", encoding="utf-8")
    # The validator reaches the profile check only after authenticating the
    # concrete input digests, so bind those values to this tiny fixture.
    from lehome_train.groot.runtime_mixture import source_tree_sha256
    from lehome_train.io import sha256_file

    plan["input_bindings"].update({
        "organizer_manifest_sha256": sha256_file(files["organizer-manifest"]),
        "organizer_tree_sha256": source_tree_sha256(organizer),
        "campaign_receipt_sha256": sha256_file(files["campaign-receipt"]),
        "campaign_tree_sha256": source_tree_sha256(campaign),
        "selected_bindings_sha256": sha256_file(files["selected"]),
        "source_lineage_sha256": sha256_file(files["source-lineage"]),
        "garment_index_sha256": sha256_file(files["garment-index"]),
    })
    experiment = SimpleNamespace(
        weights={"bc": 70, "rollout": 30, "dagger": 0},
        held_out_garments=(
            "Top_Long_Unseen_1", "Top_Short_Unseen_1",
            "Pant_Long_Unseen_1", "Pant_Short_Unseen_1",
        ),
        raw=MappingProxyType({"training": {
            "action_horizon": 16, "global_batch_size": 64, "physical_batch_size": 64,
            "max_steps": 2000, "local_checkpoint_steps": [500, 1000, 1500, 2000],
            "hf_checkpoint_steps": [1000, 2000], "loader_candidates": [0, 4, 8, 12, 16],
        }}),
    )
    with pytest.raises(ValueError, match="profile drift"):
        validate_generated_runtime_plan_bindings(
            plan, organizer_root=organizer, campaign_root=campaign,
            organizer_manifest=files["organizer-manifest"], campaign_receipt=files["campaign-receipt"],
            selected_bindings=files["selected"], garment_index=files["garment-index"],
            experiment=experiment,
        )
