from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import types

import pytest


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _selected_150_document() -> tuple[dict[str, object], dict[str, object]]:
    """Production-shaped selected index and campaign ledger, sans raw artifacts."""
    from lehome_train.io import canonical_json_sha256

    categories = ("top_long", "top_short", "pant_long", "pant_short")
    rows = [
        {
            "attempt_id": f"attempt-{index:03d}",
            "episode_id": f"attempt-{index:03d}",
            # Historical name retained in the frozen index.  Its value binds
            # SHA256SUMS.json, not episode.json.
            "episode_manifest_sha256": f"{index:064x}",
        }
        for index in range(150)
    ]
    document = {
        "schema_version": 1,
        "selection_sha256": canonical_json_sha256({"schema_version": 1, "selected_bindings": rows}),
        "selected_bindings": rows,
    }
    campaign = {
        "attempt_receipts": [
            {
                "attempt_id": row["attempt_id"],
                "episode_id": row["episode_id"],
                "category": categories[index % len(categories)],
                "accepted_success": True,
                "release_stage": "seen",
                "outcome": "success",
            }
            for index, row in enumerate(rows)
        ]
    }
    return document, campaign


def _raw_selected_campaign(
    tmp_path: Path, *, campaign_attempts: int = 150,
) -> tuple[Path, dict[str, object], dict[str, object]]:
    """Make selected raw roots plus optional rejected campaign-attempt roots."""
    from lehome.flywheel.artifacts import build_sha256_manifest
    from lehome_train.io import canonical_json_sha256

    campaign = tmp_path / "campaign"
    categories = ("top_long", "top_short", "pant_long", "pant_short")
    rows: list[dict[str, str]] = []
    receipts: list[dict[str, object]] = []
    if campaign_attempts < 150:
        raise ValueError("campaign fixture must retain all selected attempts")
    for index in range(campaign_attempts):
        attempt_id = f"attempt-{index:03d}"
        category = categories[index % len(categories)]
        raw = campaign / "raw" / attempt_id
        selected = index < 150
        if selected:
            _write(raw / "episode.json", {
                "episode_id": attempt_id,
                "accepted_success": True,
                "outcome": "success",
                "terminal_reason": "success",
                "mode": "autonomous",
                "identity": {"release_stage": "seen", "category": category, "garment_name": f"{category}-seen-{index}"},
            })
            _write(raw / "SHA256SUMS.json", build_sha256_manifest(raw))
            rows.append({
                "attempt_id": attempt_id,
                "episode_id": attempt_id,
                "episode_manifest_sha256": hashlib.sha256((raw / "SHA256SUMS.json").read_bytes()).hexdigest(),
            })
        else:
            # Rejected attempts are preserved as campaign evidence, but cannot
            # be normalized because they are not in selected-150.
            (raw / "rejected-attempt.txt").parent.mkdir(parents=True, exist_ok=True)
            (raw / "rejected-attempt.txt").write_text("not selected", encoding="utf-8")
        receipts.append({
            "attempt_id": attempt_id,
            "episode_id": attempt_id,
            "category": category,
            "accepted_success": selected,
            "release_stage": "seen",
            "outcome": "success" if selected else "failure",
        })
    document = {
        "schema_version": 1,
        "selection_sha256": canonical_json_sha256({"schema_version": 1, "selected_bindings": rows}),
        "selected_bindings": rows,
    }
    return campaign, document, {"attempt_receipts": receipts}


def _experiment_with_empty_bc_garment_index(organizer: Path, campaign: Path) -> dict[str, object]:
    from test_experiment_manifest import _manifest
    from lehome_train.io import canonical_json_bytes, sha256_file
    from lehome_train.groot.runtime_mixture import source_tree_sha256

    index = organizer / "garment-index.json"
    index.write_bytes(canonical_json_bytes({"schema_version": 1, "kind": "lehome_bc_garment_index", "episodes": []}))
    experiment = _manifest()
    experiment["bc_bundle"]["garment_index_sha256"] = sha256_file(index)  # type: ignore[index]
    experiment["bc_bundle"]["tree_sha256"] = source_tree_sha256(organizer)  # type: ignore[index]
    experiment["bc_bundle"]["manifest_sha256"] = sha256_file(organizer / "manifest.json")  # type: ignore[index]
    experiment["rollout_bundle"]["tree_sha256"] = source_tree_sha256(campaign)  # type: ignore[index]
    experiment["rollout_bundle"]["manifest_sha256"] = sha256_file(campaign / "campaign-receipt.json")  # type: ignore[index]
    return experiment


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "heldout", "tamper", "unknown"])
def test_authenticated_bc_garment_index_fails_closed_on_coverage_and_identity_drift(
    tmp_path: Path, mutation: str,
) -> None:
    from lehome_train.groot.runtime_mixture_builder import load_authenticated_bc_garment_index
    from lehome_train.io import canonical_json_bytes, sha256_file

    organizer = tmp_path / "organizer"
    manifest = organizer / "manifest.json"
    _write(manifest, {"train_episode_ids": ["episode-0"], "validation_episode_ids": ["episode-1"]})
    index = organizer / "garment-index.json"
    value: dict[str, object] = {"schema_version": 1, "kind": "lehome_bc_garment_index", "episodes": [
        {"episode_id": "episode-0", "garment_name": "Top_Long_Seen_1"},
        {"episode_id": "episode-1", "garment_name": "Pant_Long_Seen_1"},
    ]}
    if mutation == "missing":
        value["episodes"] = value["episodes"][:-1]  # type: ignore[index]
    elif mutation == "extra":
        value["episodes"].append({"episode_id": "extra", "garment_name": "Top_Long_Seen_2"})  # type: ignore[index]
    elif mutation == "duplicate":
        value["episodes"].append({"episode_id": "episode-0", "garment_name": "Top_Long_Seen_2"})  # type: ignore[index]
    elif mutation == "heldout":
        value["episodes"][1]["garment_name"] = "Pant_Short_Unseen_1"  # type: ignore[index]
    elif mutation == "unknown":
        value["unexpected"] = True
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_bytes(canonical_json_bytes(value))
    expected = sha256_file(index)
    if mutation == "tamper":
        index.write_text(index.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(ValueError):
        load_authenticated_bc_garment_index(
            organizer, manifest, "garment-index.json", expected,
            held_out_garments=("Top_Long_Unseen_1", "Top_Short_Unseen_1", "Pant_Long_Unseen_1", "Pant_Short_Unseen_1"),
        )


def test_authenticated_bc_garment_index_returns_complete_nonheld_mapping(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture_builder import load_authenticated_bc_garment_index
    from lehome_train.io import canonical_json_bytes, sha256_file

    organizer = tmp_path / "organizer"
    manifest = organizer / "manifest.json"
    _write(manifest, {"train_episode_ids": ["episode-0"], "validation_episode_ids": ["episode-1"]})
    index = organizer / "garment-index.json"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_bytes(canonical_json_bytes({"schema_version": 1, "kind": "lehome_bc_garment_index", "episodes": [
        {"episode_id": "episode-0", "garment_name": "Top_Long_Seen_1"},
        {"episode_id": "episode-1", "garment_name": "Pant_Long_Seen_1"},
    ]}))

    assert load_authenticated_bc_garment_index(
        organizer, manifest, "garment-index.json", sha256_file(index),
        held_out_garments=("Top_Long_Unseen_1", "Top_Short_Unseen_1", "Pant_Long_Unseen_1", "Pant_Short_Unseen_1"),
    ) == {"episode-0": "Top_Long_Seen_1", "episode-1": "Pant_Long_Seen_1"}


def test_source_mutation_during_normalization_blocks_pending_destination(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture_builder import (
        immutable_source_identities,
        require_unchanged_source_identities,
    )

    organizer, campaign = tmp_path / "organizer", tmp_path / "campaign"
    _write(organizer / "manifest.json", {"train_episode_ids": [], "validation_episode_ids": []})
    _write(organizer / "garment-index.json", {})
    _write(campaign / "campaign-receipt.json", {"attempt_receipts": []})
    identities = immutable_source_identities(
        organizer, campaign, organizer / "manifest.json", campaign / "campaign-receipt.json",
    )
    # This write represents a source root being modified by a concurrent actor
    # while normalization is reading already authenticated inputs.
    (organizer / "changed-during-normalization.bin").write_bytes(b"changed")

    with pytest.raises(ValueError, match="changed during mixture generation"):
        require_unchanged_source_identities(
            identities, organizer_root=organizer, campaign_root=campaign,
            organizer_manifest=organizer / "manifest.json", campaign_receipt=campaign / "campaign-receipt.json",
        )
    assert not (tmp_path / "mixture").exists()


def test_builder_rejects_root_mutation_during_normalization_before_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lehome_train.groot.runtime_mixture import source_tree_sha256
    from lehome_train.groot.runtime_mixture_builder import build_runtime_mixture
    from lehome_train.io import canonical_json_sha256, sha256_file

    organizer, campaign = tmp_path / "organizer", tmp_path / "campaign"
    _write(organizer / "manifest.json", {"train_episode_ids": [], "validation_episode_ids": []})
    _write(organizer / "garment-index.json", {})
    _write(campaign / "campaign-receipt.json", {"attempt_receipts": []})
    plan = {"selected_frame_ranges": []}
    plan["sha256"] = canonical_json_sha256(plan)
    _write(tmp_path / "plan.json", {"plan": plan, "plan_sha256": plan["sha256"]})
    _write(tmp_path / "selected-150.json", {})
    _write(tmp_path / "source-publications.json", {})
    _write(tmp_path / "experiment.json", {})
    bundle = types.SimpleNamespace(
        repository="ryanjin333/lehome-groot-n17-data", revision="b" * 40, prefix="bc/full",
        tree_sha256=source_tree_sha256(organizer), manifest_sha256=sha256_file(organizer / "manifest.json"),
        garment_index_path="garment-index.json", garment_index_sha256="a" * 64,
    )
    rollout = types.SimpleNamespace(
        repository="ryanjin333/lehome-groot-n17-data", revision="c" * 40, prefix="rollouts/round-1",
        tree_sha256=source_tree_sha256(campaign), manifest_sha256=sha256_file(campaign / "campaign-receipt.json"),
    )
    experiment = types.SimpleNamespace(
        bc_bundle=bundle, rollout_bundle=rollout, held_out_garments=(),
        mixture_manifest_sha256=plan["sha256"], quotas={"bc": 7, "rollout": 3}, weights={"bc": 70, "rollout": 30, "dagger": 0},
        identity_sha256="d" * 64, train_lineage_sha256=canonical_json_sha256({"split": "train", "lineage_ids": ["bc:episode-0"]}),
        validation_lineage_sha256=canonical_json_sha256({"split": "validation", "lineage_ids": []}),
    )
    monkeypatch.setattr("lehome_train.groot.runtime_mixture_builder.load_experiment_manifest", lambda _path: experiment)
    monkeypatch.setattr("lehome_train.groot.runtime_mixture_builder.load_authenticated_bc_garment_index", lambda *_args, **_kwargs: {})
    monkeypatch.setattr("lehome_train.groot.runtime_mixture_builder.validate_selected_bindings", lambda *_args: {"attempt-000": "a" * 64})
    monkeypatch.setattr("lehome_train.groot.runtime_mixture_builder.validate_selected_raw_roots", lambda *_args, **_kwargs: {})
    monkeypatch.setattr("lehome_train.groot.runtime_mixture_builder.validate_plan_windows", lambda *_args, **_kwargs: [{"source_kind": "organizer", "raw_episode_id": "episode-0", "raw_frame_start": 0, "raw_frame_stop": 16, "split": "train"}])
    monkeypatch.setattr("lehome_train.groot.runtime_mixture_builder._source_publications", lambda _path: {"organizer": {"repository": bundle.repository, "revision": bundle.revision, "prefix": bundle.prefix, "readback_receipt_path": "/tmp/bc.json", "readback_receipt_sha256": "a" * 64}, "rollout": {"repository": rollout.repository, "revision": rollout.revision, "prefix": rollout.prefix, "readback_receipt_path": "/tmp/rollout.json", "readback_receipt_sha256": "a" * 64}})
    monkeypatch.setattr("lehome_train.groot.runtime_mixture_builder._normalization_statistics", lambda *_args, **_kwargs: (organizer / "mutated-during-normalization.bin").write_bytes(b"drift") or {})

    with pytest.raises(ValueError, match="changed during mixture generation"):
        build_runtime_mixture(
            organizer_root=organizer, campaign_root=campaign,
            source_publications=tmp_path / "source-publications.json", selected_bindings=tmp_path / "selected-150.json",
            plan_state=tmp_path / "plan.json", destination=tmp_path / "mixture", experiment_manifest=tmp_path / "experiment.json",
        )
    assert not (tmp_path / "mixture").exists()


@pytest.mark.parametrize("garment", [
    "Top_Long_Unseen_1", "Top_Short_Unseen_1", "Pant_Long_Unseen_1", "Pant_Short_Unseen_1",
])
def test_authenticated_rollout_garment_identity_rejects_each_held_out_name(
    tmp_path: Path, garment: str,
) -> None:
    from lehome.flywheel.artifacts import build_sha256_manifest
    from lehome_train.io import canonical_json_sha256
    from lehome_train.groot.runtime_mixture_builder import validate_selected_bindings, validate_selected_raw_roots

    campaign, document, receipt = _raw_selected_campaign(tmp_path)
    raw = campaign / "raw" / "attempt-000"
    episode = json.loads((raw / "episode.json").read_text(encoding="utf-8"))
    episode["identity"]["garment_name"] = garment
    _write(raw / "episode.json", episode)
    (raw / "SHA256SUMS.json").unlink()
    _write(raw / "SHA256SUMS.json", build_sha256_manifest(raw))
    document["selected_bindings"][0]["episode_manifest_sha256"] = hashlib.sha256((raw / "SHA256SUMS.json").read_bytes()).hexdigest()
    document["selection_sha256"] = canonical_json_sha256({"schema_version": 1, "selected_bindings": document["selected_bindings"]})

    with pytest.raises(ValueError, match="held-out garment"):
        validate_selected_raw_roots(
            campaign, validate_selected_bindings(document, receipt), receipt,
            held_out_garments=(garment,),
        )


def test_authenticated_rollout_garment_identity_keeps_nonheld_seen_control(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture_builder import validate_selected_bindings, validate_selected_raw_roots

    campaign, document, receipt = _raw_selected_campaign(tmp_path)

    garments = validate_selected_raw_roots(campaign, validate_selected_bindings(document, receipt), receipt)

    assert garments["attempt-000"] == "top_long-seen-0"


def test_builder_allows_authenticated_train_episode_to_be_demoted_to_frozen_mixture_validation(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture_builder import validate_plan_windows

    organizer = tmp_path / "organizer"
    _write(organizer / "manifest.json", {
        "train_episode_ids": ["1"], "validation_episode_ids": ["2"],
    })
    plan = {"selected_frame_ranges": [{
        "source_kind": "organizer", "source_manifest_sha256": "a" * 64,
        "source_episode_id": "1", "raw_episode_id": "1", "raw_frame_start": 0,
        "raw_frame_stop": 16, "frame_start": 0, "frame_stop": 16, "raw_frame_ids": [str(i) for i in range(16)],
        "split": "validation",
    }]}

    windows = validate_plan_windows(plan, organizer_manifest=organizer / "manifest.json", accepted_rollouts={})

    assert windows[0]["split"] == "validation"


def test_builder_rejects_original_validation_episode_promoted_to_mixture_train(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture_builder import validate_plan_windows

    _write(tmp_path / "organizer.json", {"train_episode_ids": ["1"], "validation_episode_ids": ["2"]})
    plan = {"selected_frame_ranges": [{
        "source_kind": "organizer", "source_manifest_sha256": "a" * 64,
        "source_episode_id": "2", "raw_episode_id": "2", "raw_frame_start": 0,
        "raw_frame_stop": 16, "frame_start": 0, "frame_stop": 16, "raw_frame_ids": [str(i) for i in range(16)], "split": "train",
    }]}

    with pytest.raises(ValueError, match="split"):
        validate_plan_windows(plan, organizer_manifest=tmp_path / "organizer.json", accepted_rollouts={})


def test_builder_rejects_unaccepted_rollout_even_when_range_is_h16(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture_builder import validate_plan_windows

    _write(tmp_path / "organizer.json", {"train_episode_ids": ["1"], "validation_episode_ids": []})
    plan = {"selected_frame_ranges": [{
        "source_kind": "flywheel", "source_manifest_sha256": "b" * 64,
        "source_episode_id": "0", "raw_episode_id": "attempt-0", "raw_frame_start": 0,
        "raw_frame_stop": 16, "raw_frame_ids": [str(i) for i in range(16)],
        "split": "train",
    }]}

    with pytest.raises(ValueError, match="accepted"):
        validate_plan_windows(plan, organizer_manifest=tmp_path / "organizer.json", accepted_rollouts={})


def test_builder_collapses_exact_duplicate_source_ranges_before_runtime_scheduler_oversamples(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture_builder import validate_plan_windows

    _write(tmp_path / "organizer.json", {"train_episode_ids": ["1"], "validation_episode_ids": []})
    def selection(kind: str, source: str, raw: str, source_start: int) -> dict[str, object]:
        return {"source_kind": kind, "source_manifest_sha256": "a" * 64, "source_episode_id": source, "raw_episode_id": raw, "frame_start": source_start, "frame_stop": source_start + 16, "raw_frame_start": 0, "raw_frame_stop": 16, "raw_frame_ids": [str(i) for i in range(16)], "split": "train"}
    plan = {"selected_frame_ranges": [selection("organizer", "1", "1", 0), selection("organizer", "1", "1", 16), selection("flywheel", "0", "attempt", 0), selection("flywheel", "1", "attempt", 16)]}

    windows = validate_plan_windows(plan, organizer_manifest=tmp_path / "organizer.json", accepted_rollouts={"attempt": "attempt"})

    assert len(windows) == 2


def test_selected_150_requires_canonical_hash_exact_rows_and_production_ledger_membership() -> None:
    from lehome_train.groot.runtime_mixture_builder import validate_selected_bindings

    document, campaign = _selected_150_document()
    rows = document["selected_bindings"]

    assert validate_selected_bindings(document, campaign) == {row["attempt_id"]: row["episode_manifest_sha256"] for row in rows}  # type: ignore[index]
    document["selected_bindings"][0]["episode_id"] = "wrong"
    with pytest.raises(ValueError, match="selected|identity|binding"):
        validate_selected_bindings(document, campaign)


@pytest.mark.parametrize("mutation", ["selection_sha256", "count", "accepted", "stage", "outcome"])
def test_selected_150_rejects_hash_count_or_campaign_acceptance_drift(mutation: str) -> None:
    from lehome_train.groot.runtime_mixture_builder import validate_selected_bindings

    document, campaign = _selected_150_document()
    if mutation == "selection_sha256":
        document["selection_sha256"] = "0" * 64
    elif mutation == "count":
        document["selected_bindings"] = document["selected_bindings"][:-1]
    elif mutation == "accepted":
        campaign["attempt_receipts"][0]["accepted_success"] = False
    elif mutation == "stage":
        campaign["attempt_receipts"][0]["release_stage"] = "public_unseen"
    else:
        campaign["attempt_receipts"][0]["outcome"] = "failure"

    with pytest.raises(ValueError, match="selected-150|accepted|binding|ledger"):
        validate_selected_bindings(document, campaign)


def test_selected_150_rejects_legacy_opaque_hash_but_derives_a_new_canonical_artifact_without_mutation() -> None:
    from lehome_train.groot.runtime_mixture_builder import validate_selected_bindings
    from lehome_train.io import canonical_json_sha256

    derived, campaign = _selected_150_document()
    rows = derived["selected_bindings"]
    original_rows = json.loads(json.dumps(rows))
    legacy = {"schema_version": 1, "selection_sha256": "a" * 64, "selected_bindings": rows}
    with pytest.raises(ValueError, match="canonical"):
        validate_selected_bindings(legacy, campaign)
    derived = {"schema_version": 1, "selection_sha256": canonical_json_sha256({"schema_version": 1, "selected_bindings": rows}), "selected_bindings": rows}
    assert validate_selected_bindings(derived, campaign)["attempt-000"] == "0" * 64
    assert rows == original_rows


def test_selected_150_raw_roots_use_the_legacy_field_as_a_checksum_manifest_binding(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture_builder import (
        validate_selected_bindings,
        validate_selected_raw_roots,
    )

    campaign, document, receipt = _raw_selected_campaign(tmp_path)
    raw = campaign / "raw" / "attempt-000"
    assert document["selected_bindings"][0]["episode_manifest_sha256"] == hashlib.sha256((raw / "SHA256SUMS.json").read_bytes()).hexdigest()
    assert document["selected_bindings"][0]["episode_manifest_sha256"] != hashlib.sha256((raw / "episode.json").read_bytes()).hexdigest()

    validate_selected_raw_roots(campaign, validate_selected_bindings(document, receipt), receipt)


def test_selected_150_allows_the_production_332_attempt_campaign_tree(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture_builder import (
        validate_selected_bindings,
        validate_selected_raw_roots,
    )

    campaign, document, receipt = _raw_selected_campaign(tmp_path, campaign_attempts=332)

    assert len(receipt["attempt_receipts"]) == 332
    assert len(list((campaign / "raw").iterdir())) == 332
    validate_selected_raw_roots(campaign, validate_selected_bindings(document, receipt), receipt)


@pytest.mark.parametrize("mutation", ["unledgered", "symlink"])
def test_selected_150_rejects_unledgered_or_symlinked_campaign_raw_entries(
    tmp_path: Path, mutation: str,
) -> None:
    from lehome_train.groot.runtime_mixture_builder import (
        validate_selected_bindings,
        validate_selected_raw_roots,
    )

    campaign, document, receipt = _raw_selected_campaign(tmp_path, campaign_attempts=332)
    raw = campaign / "raw"
    if mutation == "unledgered":
        (raw / "unledgered-attempt").mkdir()
    else:
        (raw / "attempt-151" / "rejected-attempt.txt").unlink()
        (raw / "attempt-151").rmdir()
        (raw / "attempt-151").symlink_to(raw / "attempt-150", target_is_directory=True)

    with pytest.raises(ValueError, match="unledgered|unsafe"):
        validate_selected_raw_roots(campaign, validate_selected_bindings(document, receipt), receipt)


@pytest.mark.parametrize("mutation", ["checksum", "unused_selected", "raw_stage", "raw_outcome"])
def test_selected_150_raw_roots_reject_checksum_or_acceptance_tampering(
    tmp_path: Path, mutation: str,
) -> None:
    from lehome_train.groot.runtime_mixture_builder import (
        validate_selected_bindings,
        validate_selected_raw_roots,
    )
    from lehome_train.io import canonical_json_sha256

    campaign, document, receipt = _raw_selected_campaign(tmp_path)
    index = 149 if mutation == "unused_selected" else 0
    raw = campaign / "raw" / f"attempt-{index:03d}"
    if mutation == "checksum":
        _write(raw / "SHA256SUMS.json", {"episode.json": {"sha256": "0" * 64, "size": 0}})
    else:
        episode = json.loads((raw / "episode.json").read_text(encoding="utf-8"))
        if mutation == "raw_stage":
            episode["identity"]["release_stage"] = "public_unseen"
        elif mutation == "raw_outcome":
            episode["outcome"] = "failure"
        else:
            episode["identity"]["category"] = "wrong"
        _write(raw / "episode.json", episode)
        episode_bytes = (raw / "episode.json").read_bytes()
        _write(raw / "SHA256SUMS.json", {
            "episode.json": {"sha256": hashlib.sha256(episode_bytes).hexdigest(), "size": len(episode_bytes)},
        })
        document["selected_bindings"][index]["episode_manifest_sha256"] = hashlib.sha256((raw / "SHA256SUMS.json").read_bytes()).hexdigest()
        document["selection_sha256"] = canonical_json_sha256({
            "schema_version": 1,
            "selected_bindings": document["selected_bindings"],
        })

    with pytest.raises(ValueError, match="raw|checksum|manifest|accepted|identity"):
        validate_selected_raw_roots(campaign, validate_selected_bindings(document, receipt), receipt)


def test_builder_rejects_an_unused_selected_raw_tamper_before_window_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lehome_train.groot.runtime_mixture_builder import build_runtime_mixture
    from lehome_train.io import canonical_json_sha256

    campaign, document, receipt = _raw_selected_campaign(tmp_path)
    _write(campaign / "campaign-receipt.json", receipt)
    (campaign / "raw" / "attempt-149" / "SHA256SUMS.json").write_text("tampered", encoding="utf-8")
    selected = tmp_path / "selected-150.json"
    _write(selected, document)
    _write(tmp_path / "source-publications.json", {})
    organizer = tmp_path / "organizer"
    _write(organizer / "manifest.json", {"train_episode_ids": [], "validation_episode_ids": []})
    plan = {"selected_frame_ranges": []}
    plan["sha256"] = canonical_json_sha256(plan)
    state = tmp_path / "plan.json"
    _write(state, {"plan": plan, "plan_sha256": plan["sha256"]})
    experiment = tmp_path / "experiment.json"
    _write(experiment, _experiment_with_empty_bc_garment_index(organizer, campaign))
    monkeypatch.setattr(
        "lehome_train.groot.runtime_mixture_builder.validate_plan_windows",
        lambda *_args, **_kwargs: pytest.fail("raw validation must precede window selection"),
    )

    with pytest.raises(ValueError, match="raw rollout checksum-manifest binding drift"):
        build_runtime_mixture(
            organizer_root=organizer,
            campaign_root=campaign,
            source_publications=tmp_path / "source-publications.json",
            selected_bindings=selected,
            plan_state=state,
            destination=tmp_path / "mixture",
            experiment_manifest=experiment,
        )


def test_builder_rejects_unused_selected_held_out_garment_before_window_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lehome.flywheel.artifacts import build_sha256_manifest
    from lehome_train.groot.runtime_mixture_builder import build_runtime_mixture
    from lehome_train.io import canonical_json_sha256

    campaign, document, receipt = _raw_selected_campaign(tmp_path)
    _write(campaign / "campaign-receipt.json", receipt)
    raw = campaign / "raw" / "attempt-149"
    episode = json.loads((raw / "episode.json").read_text(encoding="utf-8"))
    episode["identity"]["garment_name"] = "Pant_Short_Unseen_1"
    _write(raw / "episode.json", episode)
    (raw / "SHA256SUMS.json").unlink()
    _write(raw / "SHA256SUMS.json", build_sha256_manifest(raw))
    document["selected_bindings"][149]["episode_manifest_sha256"] = hashlib.sha256((raw / "SHA256SUMS.json").read_bytes()).hexdigest()
    document["selection_sha256"] = canonical_json_sha256({"schema_version": 1, "selected_bindings": document["selected_bindings"]})
    selected = tmp_path / "selected-150.json"
    _write(selected, document)
    _write(tmp_path / "source-publications.json", {})
    organizer = tmp_path / "organizer"
    _write(organizer / "manifest.json", {"train_episode_ids": [], "validation_episode_ids": []})
    plan = {"selected_frame_ranges": []}
    plan["sha256"] = canonical_json_sha256(plan)
    state = tmp_path / "plan.json"
    _write(state, {"plan": plan, "plan_sha256": plan["sha256"]})
    experiment = tmp_path / "experiment.json"
    _write(experiment, _experiment_with_empty_bc_garment_index(organizer, campaign))
    monkeypatch.setattr(
        "lehome_train.groot.runtime_mixture_builder.validate_plan_windows",
        lambda *_args, **_kwargs: pytest.fail("all selected raw roots must be validated before window selection"),
    )

    with pytest.raises(ValueError, match="held-out garment"):
        build_runtime_mixture(
            organizer_root=organizer,
            campaign_root=campaign,
            source_publications=tmp_path / "source-publications.json",
            selected_bindings=selected,
            plan_state=state,
            destination=tmp_path / "mixture",
            experiment_manifest=experiment,
        )


def test_loader_pilot_requires_the_canonical_x86_worker_sweep_but_rejects_caller_throughput_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This contract fails before parquet construction; make its dependency
    # optional on the macOS test lane.
    monkeypatch.setitem(sys.modules, "pyarrow", types.ModuleType("pyarrow"))
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", types.ModuleType("pyarrow.parquet"))
    from lehome_train.groot.runtime_mixture_builder import pilot_from_request

    request = tmp_path / "pilot.json"
    _write(request, {
        "schema_version": 1,
        "command": "pilot-runtime-mixture",
        "arguments": {
            "mixture_manifest": "/runtime/mixture.json",
            "mounts_descriptor": "/runtime/mounts.json",
            "sample_count": 100,
            "worker_counts": [0, 4, 8, 16, 24],
            "timeout_seconds": 60,
            "authenticated_evidence": {"mixture_id": "a" * 64},
            "gpu_starvation_floor_samples_per_second": 1.0,
        },
    })

    with pytest.raises(ValueError, match="canonical"):
        pilot_from_request(request)


def test_loader_pilot_converts_one_bc_and_one_rollout_payload_through_pinned_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CPU pilot validates the pinned VLAStepData message surface before timing."""
    monkeypatch.setitem(sys.modules, "pyarrow", types.ModuleType("pyarrow"))
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", types.ModuleType("pyarrow.parquet"))
    import lehome_train.groot.runtime_mixture as mixture

    class Window:
        def __init__(self, source_type: str, window_id: str) -> None:
            self.source_type = source_type
            self.window_id = window_id

    bc, rollout = Window("bc", "bc-h16"), Window("rollout", "rollout-h16")
    contract = types.SimpleNamespace(training_windows=(bc, rollout))
    decoded = {"bc-h16": {"source": "bc"}, "rollout-h16": {"source": "rollout"}}
    calls: list[object] = []

    class Loader:
        cache_cap = 2

        def __init__(self, _contract: object) -> None:
            pass

        def load(self, window: Window) -> object:
            return decoded[window.window_id]

    class Dataset:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

    class DataLoader:
        def __init__(self, _dataset: object, **_kwargs: object) -> None:
            self._remaining = 100

        def __iter__(self):
            return self

        def __next__(self) -> object:
            if self._remaining == 0:
                raise StopIteration
            self._remaining -= 1
            return object()

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_initialized=lambda: False)
    torch_utils = types.ModuleType("torch.utils")
    torch_utils_data = types.ModuleType("torch.utils.data")
    torch_utils_data.DataLoader = DataLoader
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.utils", torch_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.data", torch_utils_data)
    monkeypatch.setattr(mixture, "load_runtime_contract", lambda *_args: contract)
    monkeypatch.setattr(mixture, "RangeSourceLoader", Loader)
    monkeypatch.setattr(mixture, "RuntimeMixtureDataset", Dataset)
    monkeypatch.setattr(mixture, "pinned_processor_messages", lambda payload: calls.append(payload) or [{"content": payload}])

    request = tmp_path / "pilot.json"
    _write(request, {
        "schema_version": 1,
        "command": "pilot-runtime-mixture",
        "arguments": {
            "mixture_manifest": "/runtime/mixture.json",
            "mounts_descriptor": "/runtime/mounts.json",
            "sample_count": 100,
            "worker_counts": [0, 4, 8, 16, 24],
            "timeout_seconds": 60,
            "authenticated_evidence": {"mixture_id": "a" * 64},
        },
    })

    from lehome_train.groot.runtime_mixture_builder import pilot_from_request

    receipt = pilot_from_request(request)

    assert calls == [decoded["bc-h16"], decoded["rollout-h16"]]
    assert receipt["schema_version"] == 4
    assert receipt["model_loaded"] is False
    assert receipt["authenticated_evidence"] == {"mixture_id": "a" * 64}
