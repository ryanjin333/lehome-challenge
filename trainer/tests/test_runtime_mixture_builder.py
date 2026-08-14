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
                "identity": {"release_stage": "seen", "category": category},
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
    organizer = tmp_path / "organizer"
    _write(organizer / "manifest.json", {"train_episode_ids": [], "validation_episode_ids": []})
    plan = {"selected_frame_ranges": []}
    plan["sha256"] = canonical_json_sha256(plan)
    state = tmp_path / "plan.json"
    _write(state, {"plan": plan, "plan_sha256": plan["sha256"]})
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
