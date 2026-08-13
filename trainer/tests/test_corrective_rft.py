from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

from lehome.flywheel.artifacts import build_sha256_manifest
from lehome_train.flywheel.corrective import (
    APPROVED_PARENT_ARTIFACT_SHA256,
    APPROVED_PARENT_REPOSITORY,
    APPROVED_PARENT_REVISION,
    CATEGORY_SUCCESS_FLOORS,
    CorrectiveCampaignPolicy,
    assess_corrective_campaign,
    build_corrective_campaign_receipt,
    build_corrective_publication_plan,
    build_corrective_selection_bundle,
    bind_corrective_episode_artifacts,
    build_corrective_selection_bundle,
    select_corrective_successes,
    verify_corrective_selection_bundle,
)
from lehome_train.flywheel import rft
from lehome_train.flywheel.rft import materialize_verified_corrective_rft_snapshot


def test_corrective_next_wave_categories_are_effort_aware_with_bounded_starvation() -> None:
    # Live corrective counts: TL 13/7, TS 19/9, PL 13/2, PS 19/17.  The
    # smoothed effort estimate must give the hard pant-long category two slots
    # without permanently excluding any unresolved category.
    assert __import__("lehome_train.flywheel.corrective", fromlist=["_next_wave_categories"])._next_wave_categories({
        "top_long": 7, "top_short": 9, "pant_long": 2, "pant_short": 17,
    }, {
        "top_long": 13, "top_short": 19, "pant_long": 13, "pant_short": 19,
    }) == ("pant_long", "top_short", "pant_long", "top_long")

    allocator = __import__("lehome_train.flywheel.corrective", fromlist=["_next_wave_categories"])
    # Two full waves scheduled elsewhere make pant-short's lower count stale;
    # it is reserved one slot even though its immediate effort is lower.
    bounded = allocator._next_wave_categories(
        {"top_long": 29, "top_short": 44, "pant_long": 29, "pant_short": 44},
        {"top_long": 56, "top_short": 56, "pant_long": 56, "pant_short": 40},
    )
    assert "pant_short" in bounded and len(bounded) == 4


def test_corrective_campaign_requires_distinct_seen_success_floors_and_prioritizes_short_garments() -> None:
    policy = CorrectiveCampaignPolicy(
        max_attempts=24,
        max_hourly_cost_usd=2.0,
        category_success_floors={
            "top_long": 1,
            "top_short": 2,
            "pant_long": 1,
            "pant_short": 2,
        },
        unique_success_floor=6,
    )

    report = assess_corrective_campaign(
        [
            {"episode_id": "top-long-1", "category": "top_long", "release_stage": "seen", "accepted_success": True},
            {"episode_id": "top-short-1", "category": "top_short", "release_stage": "seen", "accepted_success": True},
            {"episode_id": "pant-long-1", "category": "pant_long", "release_stage": "seen", "accepted_success": True},
            {"episode_id": "pant-short-1", "category": "pant_short", "release_stage": "seen", "accepted_success": True},
        ],
        policy=policy,
        attempted_episodes=4,
        offered_hourly_cost_usd=1.0,
        rental_kind="on-demand",
    )

    assert report.launch_allowed is True
    assert report.collection_complete is False
    assert report.priority_categories == ("top_short", "pant_short")
    assert report.category_successes == {
        "pant_long": 1,
        "pant_short": 1,
        "top_long": 1,
        "top_short": 1,
    }
    assert report.missing_successes == {"pant_short": 1, "top_short": 1}


def test_corrective_campaign_rejects_unseen_and_duplicate_success_evidence() -> None:
    policy = CorrectiveCampaignPolicy(
        max_attempts=24,
        max_hourly_cost_usd=2.0,
        category_success_floors={category: 1 for category in CATEGORY_SUCCESS_FLOORS},
        unique_success_floor=4,
    )

    with pytest.raises(ValueError, match="unseen"):
        assess_corrective_campaign(
            [
                {"episode_id": "top-long-1", "category": "top_long", "release_stage": "seen", "accepted_success": True},
                {"episode_id": "top-short-1", "category": "top_short", "release_stage": "seen", "accepted_success": True},
                {"episode_id": "pant-long-1", "category": "pant_long", "release_stage": "seen", "accepted_success": True},
                {"episode_id": "pant-short-1", "category": "pant_short", "release_stage": "public_unseen", "accepted_success": True},
            ],
            policy=policy,
            attempted_episodes=4,
            offered_hourly_cost_usd=1.0,
            rental_kind="on-demand",
        )

    with pytest.raises(ValueError, match="distinct"):
        assess_corrective_campaign(
            [
                {"episode_id": "top-long-1", "category": "top_long", "release_stage": "seen", "accepted_success": True},
                {"episode_id": "top-short-1", "category": "top_short", "release_stage": "seen", "accepted_success": True},
                {"episode_id": "pant-long-1", "category": "pant_long", "release_stage": "seen", "accepted_success": True},
                {"episode_id": "pant-long-1", "category": "pant_long", "release_stage": "seen", "accepted_success": True},
                {"episode_id": "pant-short-1", "category": "pant_short", "release_stage": "seen", "accepted_success": True},
            ],
            policy=policy,
            attempted_episodes=5,
            offered_hourly_cost_usd=1.0,
            rental_kind="on-demand",
        )


@pytest.mark.parametrize(
    ("attempts", "cost", "rental_kind", "message"),
    [
        (25, 1.0, "on-demand", "attempt"),
        (1, 2.01, "on-demand", "cost"),
        (1, 1.0, "interruptible", "on-demand"),
    ],
)
def test_corrective_campaign_stops_before_an_unbounded_or_non_on_demand_launch(
    attempts: int,
    cost: float,
    rental_kind: str,
    message: str,
) -> None:
    policy = CorrectiveCampaignPolicy(
        max_attempts=24,
        max_hourly_cost_usd=2.0,
        category_success_floors={category: 1 for category in CATEGORY_SUCCESS_FLOORS},
        unique_success_floor=4,
    )

    with pytest.raises(ValueError, match=message):
        assess_corrective_campaign(
            (),
            policy=policy,
            attempted_episodes=attempts,
            offered_hourly_cost_usd=cost,
            rental_kind=rental_kind,
        )


def test_default_corrective_policy_binds_the_next_campaign_objective() -> None:
    policy = CorrectiveCampaignPolicy()

    assert policy.max_attempts == 400
    assert policy.max_hourly_cost_usd == 2.0
    assert policy.unique_success_floor == 150
    assert policy.category_success_floors == {
        "top_long": 30,
        "top_short": 45,
        "pant_long": 30,
        "pant_short": 45,
    }


def _attempt(
    attempt_id: str,
    category: str,
    worker_slot: int,
    *,
    accepted_success: bool = True,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "wave_index": 0,
        "worker_slot": worker_slot,
        "episode_id": f"episode-{attempt_id}",
        "category": category,
        "release_stage": "seen",
        "outcome": "success" if accepted_success else "failure",
        "accepted_success": accepted_success,
        "reset_sha256": "a" * 63 + str(worker_slot),
        "randomization_sha256": "b" * 63 + str(worker_slot),
        "hard_state_sha256": "c" * 63 + str(worker_slot),
        "parent_checkpoint_repository": APPROVED_PARENT_REPOSITORY,
        "parent_checkpoint_revision": APPROVED_PARENT_REVISION,
        "parent_checkpoint_artifact_sha256": APPROVED_PARENT_ARTIFACT_SHA256,
        "parent_checkpoint_step": 12000,
        "code_revision": "f" * 40,
        "asset_revision": "0" * 40,
        "image_identity": "sha256:" + "1" * 64,
        "simulator_version": "5.1.0.0",
        "provider": {
            "rental_kind": "on-demand",
            "instance_hourly_cost_usd": 0.8,
            "account_hourly_total_usd": 0.8002,
            "offer_id": 40705900,
            "gpu_name": "RTX 3090",
            "num_gpus": 4,
        },
    }


def _verified_trainable_artifacts(tmp_path, selected):
    artifacts: dict[str, dict[str, object]] = {}
    for item in selected:
        root = tmp_path / str(item["episode_id"])
        root.mkdir()
        (root / "episode.json").write_text(json.dumps({
            "episode_id": item["episode_id"],
            "accepted_success": True,
            "outcome": "success",
            "terminal_reason": "success",
            "mode": "autonomous",
            "identity": {"release_stage": "seen"},
        }), encoding="utf-8")
        (root / "SHA256SUMS.json").write_text(json.dumps(
            build_sha256_manifest(root), sort_keys=True, separators=(",", ":")
        ), encoding="utf-8")
        artifacts[str(item["attempt_id"])] = {
            "episode_id": item["episode_id"],
            "release_stage": "seen",
            "root": str(root),
            "episode_manifest_sha256": __import__("hashlib").sha256(
                (root / "SHA256SUMS.json").read_bytes()
            ).hexdigest(),
        }
    return artifacts


def test_corrective_ledger_derives_attempt_count_and_requires_four_worker_waves() -> None:
    receipt = build_corrective_campaign_receipt(
        [
            _attempt("0001", "top_short", 0),
            _attempt("0002", "pant_short", 1),
            _attempt("0003", "top_short", 2, accepted_success=False),
            _attempt("0004", "pant_short", 3, accepted_success=False),
        ]
    )

    assert receipt["attempt_count"] == 4
    assert receipt["next_wave_categories"] == [
        "top_short", "pant_short", "top_long", "pant_long"
    ]
    broken = [_attempt("0001", "top_short", 0)]
    with pytest.raises(ValueError, match="four-worker"):
        build_corrective_campaign_receipt(broken)


def test_corrective_ledger_rejects_wrong_gpu_topology_shared_spend_and_duplicate_episodes() -> None:
    attempts = [
        _attempt("0001", "top_short", 0),
        _attempt("0002", "pant_short", 1),
        _attempt("0003", "top_long", 2),
        _attempt("0004", "pant_long", 3),
    ]
    attempts[0]["provider"] = {**attempts[0]["provider"], "num_gpus": 1}
    with pytest.raises(ValueError, match="4x3090"):
        build_corrective_campaign_receipt(attempts)

    attempts[0]["provider"] = {
        **attempts[1]["provider"], "account_hourly_total_usd": 2.01
    }
    with pytest.raises(ValueError, match="shared \\$2/hr"):
        build_corrective_campaign_receipt(attempts)

    attempts[0]["provider"] = dict(attempts[1]["provider"])
    attempts[1]["episode_id"] = attempts[0]["episode_id"]
    with pytest.raises(ValueError, match="episode IDs"):
        build_corrective_campaign_receipt(attempts)


def test_corrective_selection_uses_distinct_state_fingerprints_and_exact_floors() -> None:
    successes = [
        _attempt(f"{index:04d}", category, index % 4)
        for index, category in enumerate(
            [*("top_long",) * 31, *("top_short",) * 46, *("pant_long",) * 31, *("pant_short",) * 45],
            start=1,
        )
    ]
    attempts = [
        *successes,
        *[
            _attempt(f"failure-{index}", "top_short", 0, accepted_success=False)
            for index in range(3)
        ],
    ]
    for index, attempt in enumerate(attempts):
        attempt["wave_index"] = index // 4
        attempt["worker_slot"] = index % 4
        attempt["reset_sha256"] = f"{index:064x}"
        attempt["randomization_sha256"] = f"{index + 200:064x}"
        attempt["hard_state_sha256"] = f"{index + 400:064x}"
    selected = select_corrective_successes(attempts)

    assert len(selected) == 150
    assert {category: sum(item["category"] == category for item in selected) for category in CATEGORY_SUCCESS_FLOORS} == CATEGORY_SUCCESS_FLOORS
    duplicate = [*attempts]
    duplicate[1] = {
        **duplicate[1],
        "reset_sha256": duplicate[0]["reset_sha256"],
        "randomization_sha256": duplicate[0]["randomization_sha256"],
        "hard_state_sha256": duplicate[0]["hard_state_sha256"],
    }
    with pytest.raises(ValueError, match="state fingerprint"):
        select_corrective_successes(duplicate)


def test_corrective_selection_excludes_verified_horizon_episodes_before_exact_floors(
    tmp_path,
) -> None:
    """Receipt-successes are insufficient when the verified raw terminal is not success."""
    attempts = []
    for index, category in enumerate([
        *("top_long",) * 30,
        *("top_short",) * 47,
        *("pant_long",) * 30,
        *("pant_short",) * 45,
    ]):
        attempt = _attempt(f"{index:04d}", category, index % 4)
        attempt["wave_index"] = index // 4
        attempt["reset_sha256"] = f"{index:064x}"
        attempt["randomization_sha256"] = f"{index + 200:064x}"
        attempt["hard_state_sha256"] = f"{index + 400:064x}"
        attempts.append(attempt)

    horizon_attempt_ids = {"0030", "0031"}
    artifacts: dict[str, dict[str, object]] = {}
    for attempt in attempts:
        root = tmp_path / str(attempt["episode_id"])
        root.mkdir()
        terminal_reason = "horizon" if attempt["attempt_id"] in horizon_attempt_ids else "success"
        (root / "episode.json").write_text(json.dumps({
            "episode_id": attempt["episode_id"],
            "accepted_success": True,
            "outcome": "success",
            "terminal_reason": terminal_reason,
            "mode": "autonomous",
            "identity": {"release_stage": "seen"},
        }), encoding="utf-8")
        (root / "SHA256SUMS.json").write_text(json.dumps(
            build_sha256_manifest(root), sort_keys=True, separators=(",", ":")
        ), encoding="utf-8")
        artifacts[str(attempt["attempt_id"])] = {
            "episode_id": attempt["episode_id"],
            "release_stage": "seen",
            "root": str(root),
            "episode_manifest_sha256": __import__("hashlib").sha256(
                (root / "SHA256SUMS.json").read_bytes()
            ).hexdigest(),
        }

    bundle = build_corrective_selection_bundle(attempts, artifacts)

    assert len(bundle.bindings) == 150
    assert len(verify_corrective_selection_bundle(bundle)) == 150
    assert {item.attempt_id for item in bundle.bindings}.isdisjoint(horizon_attempt_ids)
    assert {
        category: sum(item["category"] == category for item in bundle.selected_attempt_receipts)
        for category in CATEGORY_SUCCESS_FLOORS
    } == CATEGORY_SUCCESS_FLOORS


def test_typed_corrective_chain_binds_selected_artifacts_and_rejects_stale_manifest(tmp_path) -> None:
    attempts = []
    categories = [
        *("top_long",) * 30,
        *("top_short",) * 45,
        *("pant_long",) * 30,
        *("pant_short",) * 45,
        *("top_short",) * 2,
    ]
    for index, category in enumerate(categories):
        attempt = _attempt(f"{index:04d}", category, index % 4)
        attempt["wave_index"] = index // 4
        attempt["reset_sha256"] = f"{index:064x}"
        attempt["randomization_sha256"] = f"{index + 200:064x}"
        attempt["hard_state_sha256"] = f"{index + 400:064x}"
        attempts.append(attempt)
    selected = select_corrective_successes(attempts)
    artifacts = _verified_trainable_artifacts(tmp_path, attempts)

    bundle = build_corrective_selection_bundle(attempts, artifacts)
    bindings = verify_corrective_selection_bundle(bundle)
    plan = build_corrective_publication_plan(bundle)

    assert len(bindings) == len(plan["selected_attempts"]) == 150
    assert plan["repository_private"] is True
    assert plan["disposable"] is False
    assert plan["required_verification"] == ["immutable_revision", "tree_listing", "fresh_readback"]
    stale = dict(artifacts)
    stale[selected[0]["attempt_id"]] = {
        **stale[selected[0]["attempt_id"]],
        "episode_manifest_sha256": artifacts[selected[1]["attempt_id"]]["episode_manifest_sha256"],
    }
    with pytest.raises(ValueError, match="manifests"):
        bind_corrective_episode_artifacts(selected, stale)


def test_selection_bundle_rejects_a_forged_binding_even_when_its_hash_is_recomputed(tmp_path) -> None:
    attempts = []
    for index, category in enumerate([
        *("top_long",) * 30, *("top_short",) * 45,
        *("pant_long",) * 30, *("pant_short",) * 45, *("top_short",) * 2,
    ]):
        attempt = _attempt(f"{index:04d}", category, index % 4)
        attempt["wave_index"] = index // 4
        attempt["reset_sha256"] = f"{index:064x}"
        attempt["randomization_sha256"] = f"{index + 200:064x}"
        attempt["hard_state_sha256"] = f"{index + 400:064x}"
        attempts.append(attempt)
    selected = select_corrective_successes(attempts)
    artifacts = _verified_trainable_artifacts(tmp_path, attempts)
    bundle = build_corrective_selection_bundle(attempts, artifacts)
    forged_bindings = (
        replace(bundle.bindings[0], episode_id="forged-episode"), *bundle.bindings[1:]
    )
    forged_body = {
        "schema_version": 1, "kind": "corrective_rft_selection",
        "campaign_receipt_sha256": bundle.campaign_receipt["receipt_sha256"],
        "selected_attempt_receipts": [dict(item) for item in bundle.selected_attempt_receipts],
        "bindings": [
            {"attempt_id": item.attempt_id, "episode_id": item.episode_id, "root": item.root, "episode_manifest_sha256": item.episode_manifest_sha256}
            for item in forged_bindings
        ],
    }
    forged = replace(bundle, bindings=forged_bindings, selection_sha256=hashlib.sha256(
        json.dumps(forged_body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest())
    with pytest.raises(ValueError, match="selected receipt"):
        verify_corrective_selection_bundle(forged)


def test_corrective_campaign_allows_distinct_verified_provider_facts_per_wave() -> None:
    attempts = [
        _attempt(f"{index:04d}", "top_short", index % 4)
        for index in range(8)
    ]
    for index, attempt in enumerate(attempts):
        attempt["wave_index"] = index // 4
        attempt["reset_sha256"] = f"{index:064x}"
        attempt["randomization_sha256"] = f"{index + 200:064x}"
        attempt["hard_state_sha256"] = f"{index + 400:064x}"
        if attempt["wave_index"] == 1:
            attempt["provider"] = {**attempt["provider"], "offer_id": 40705901, "instance_hourly_cost_usd": 0.9, "account_hourly_total_usd": 0.9002}

    receipt = build_corrective_campaign_receipt(attempts)

    assert receipt["provider_by_wave"]["0"]["offer_id"] == 40705900
    assert receipt["provider_by_wave"]["1"]["offer_id"] == 40705901


def test_verified_corrective_materialization_hands_exact_bindings_to_pre_hash_snapshot(
    tmp_path,
    monkeypatch,
) -> None:
    attempts = []
    for index, category in enumerate(
        [
            *("top_long",) * 30,
            *("top_short",) * 45,
            *("pant_long",) * 30,
            *("pant_short",) * 45,
            *("top_short",) * 2,
        ]
    ):
        attempt = _attempt(f"{index:04d}", category, index % 4)
        attempt["wave_index"] = index // 4
        attempt["reset_sha256"] = f"{index:064x}"
        attempt["randomization_sha256"] = f"{index + 200:064x}"
        attempt["hard_state_sha256"] = f"{index + 400:064x}"
        attempts.append(attempt)
    selected = select_corrective_successes(attempts)
    artifacts = _verified_trainable_artifacts(tmp_path, attempts)
    episode_by_root = {item["root"]: str(item["episode_id"]) for item in artifacts.values()}
    bundle = build_corrective_selection_bundle(attempts, artifacts)
    received: dict[str, object] = {}
    monkeypatch.setattr(
        rft,
        "_verify_raw",
        lambda root: {
            "episode_id": episode_by_root[str(root)],
            "identity": {"release_stage": "seen"},
        },
    )
    monkeypatch.setattr(
        rft,
        "materialize_rft_snapshot",
        lambda roots, destination, **kwargs: received.update(
            {"roots": tuple(roots), "destination": destination, **kwargs}
        ) or {"path": str(destination), "accepted_seen_successes": 150},
    )

    result = materialize_verified_corrective_rft_snapshot(
        bundle,
        tmp_path / "snapshot",
        source_repository="ryanjin333/lehome-groot-n17-data",
        source_revision="a" * 40,
        release_id="b" * 64,
        split_seed=1,
        validation_fraction=0.2,
    )

    assert result["corrective_campaign"]["campaign_receipt_sha256"] == bundle.campaign_receipt["receipt_sha256"]
    assert len(received["roots"]) == 150
    assert len(received["corrective_campaign"]["selected_bindings"]) == 150
    stale = next(tmp_path.iterdir()) / "SHA256SUMS.json"
    stale.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest is stale"):
        materialize_verified_corrective_rft_snapshot(
            bundle,
            tmp_path / "snapshot-two",
            source_repository="ryanjin333/lehome-groot-n17-data",
            source_revision="a" * 40,
            release_id="b" * 64,
            split_seed=1,
            validation_fraction=0.2,
        )


def test_verified_corrective_materialization_rejects_a_final_snapshot_with_not_150_episodes(
    tmp_path,
    monkeypatch,
) -> None:
    attempts = []
    for index, category in enumerate([
        *("top_long",) * 30, *("top_short",) * 45,
        *("pant_long",) * 30, *("pant_short",) * 45, *("top_short",) * 2,
    ]):
        attempt = _attempt(f"{index:04d}", category, index % 4)
        attempt["wave_index"] = index // 4
        attempt["reset_sha256"] = f"{index:064x}"
        attempt["randomization_sha256"] = f"{index + 200:064x}"
        attempt["hard_state_sha256"] = f"{index + 400:064x}"
        attempts.append(attempt)
    artifacts = _verified_trainable_artifacts(tmp_path, attempts)
    bundle = build_corrective_selection_bundle(attempts, artifacts)
    episode_by_root = {item.root: item.episode_id for item in bundle.bindings}
    monkeypatch.setattr(
        rft, "_verify_raw",
        lambda root: {"episode_id": episode_by_root[str(root)], "identity": {"release_stage": "seen"}},
    )
    monkeypatch.setattr(rft, "materialize_rft_snapshot", lambda *args, **kwargs: {"accepted_seen_successes": 149})

    with pytest.raises(ValueError, match="exactly 150 episodes"):
        materialize_verified_corrective_rft_snapshot(
            bundle, tmp_path / "snapshot", source_repository="ryanjin333/lehome-groot-n17-data",
            source_revision="a" * 40, release_id="b" * 64, split_seed=1, validation_fraction=0.2,
        )
