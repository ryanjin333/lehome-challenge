from __future__ import annotations

import json
from pathlib import Path

import pytest


def _identity() -> dict[str, object]:
    return {
        "experiment_manifest_sha256": "a" * 64,
        "parent_checkpoint_artifact_sha256": "b" * 64,
        "runtime_mixture_id": "c" * 64,
        "trainer_code_sha256": "d" * 64,
        "trainer_code_revision": "e" * 40,
    }


def _official(root: Path, step: int) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"weights-" + str(step).encode())
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer-" + str(step).encode())
    (checkpoint / "scheduler.pt").write_bytes(b"scheduler-" + str(step).encode())
    (checkpoint / "rng_state.pth").write_bytes(b"rng-" + str(step).encode())
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "log_history": [{"step": step, "loss": 0.25}]}),
        encoding="utf-8",
    )
    return checkpoint


def _official_sharded(root: Path, step: int) -> Path:
    checkpoint = _official(root, step)
    (checkpoint / "model.safetensors").unlink()
    (checkpoint / "model-00001-of-00002.safetensors").write_bytes(b"first-shard")
    (checkpoint / "model-00002-of-00002.safetensors").write_bytes(b"second-shard")
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({
            "metadata": {"total_size": 22},
            "weight_map": {
                "model.layers.0.weight": "model-00001-of-00002.safetensors",
                "model.layers.1.weight": "model-00002-of-00002.safetensors",
            },
        }),
        encoding="utf-8",
    )
    return checkpoint


def test_local_recovery_identity_binds_optional_awr_transform(tmp_path: Path) -> None:
    from lehome_train.groot.local_recovery import (
        attest_local_checkpoint,
        discover_local_recovery,
    )

    identity = _identity() | {
        "awr_evidence_sha256": "f" * 64,
        "awr_config_sha256": "1" * 64,
    }
    local = attest_local_checkpoint(
        checkpoint=_official(tmp_path / "run", 500),
        metadata_root=tmp_path / "shared",
        optimizer_step=500,
        identity=identity,
    )

    assert discover_local_recovery(
        metadata_root=tmp_path / "shared", identity=identity
    ) == local
    with pytest.raises(ValueError, match="identity"):
        discover_local_recovery(
            metadata_root=tmp_path / "shared",
            identity=identity | {"awr_config_sha256": "2" * 64},
        )


def test_attestation_accepts_a_complete_indexed_safetensors_checkpoint(tmp_path: Path) -> None:
    from lehome_train.groot.local_recovery import attest_local_checkpoint

    checkpoint = _official_sharded(tmp_path / "run", 500)
    local = attest_local_checkpoint(
        checkpoint=checkpoint, metadata_root=tmp_path / "shared", optimizer_step=500,
        identity=_identity(),
    )

    receipt = json.loads(local.receipt_path.read_text(encoding="utf-8"))
    assert [entry["path"] for entry in receipt["checkpoint_tree"]] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "optimizer.pt",
        "rng_state.pth",
        "scheduler.pt",
        "trainer_state.json",
    ]


def test_attestation_syncs_validated_checkpoint_files_and_directories_before_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lehome_train.groot.local_recovery as recovery

    checkpoint = _official(tmp_path / "run", 500)
    nested = checkpoint / "global_step500"
    nested.mkdir()
    (nested / "trainer.bin").write_bytes(b"state")
    calls: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        recovery, "_fsync_regular_file",
        lambda path: calls.append(("file", path)), raising=False,
    )
    monkeypatch.setattr(recovery, "_fsync_dir", lambda path: calls.append(("dir", path)))

    local = recovery.attest_local_checkpoint(
        checkpoint=checkpoint, metadata_root=tmp_path / "shared", optimizer_step=500,
        identity=_identity(),
    )

    checkpoint_calls = [call for call in calls if call[1].is_relative_to(checkpoint)]
    expected_files = sorted(path for path in checkpoint.rglob("*") if path.is_file())
    assert checkpoint_calls == [
        *(("file", path) for path in expected_files),
    ] + [
        ("dir", nested), ("dir", checkpoint),
    ]
    assert calls.index(("dir", tmp_path / "shared")) > calls.index(("dir", checkpoint))
    assert local.receipt_path.with_suffix(".COMPLETE").is_file()


def test_attestation_fsync_failure_never_writes_a_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lehome_train.groot.local_recovery as recovery

    checkpoint, metadata = _official(tmp_path / "run", 500), tmp_path / "shared"
    monkeypatch.setattr(
        recovery, "_fsync_regular_file",
        lambda _path: (_ for _ in ()).throw(OSError("checkpoint fsync failed")),
        raising=False,
    )

    with pytest.raises(OSError, match="checkpoint fsync failed"):
        recovery.attest_local_checkpoint(
            checkpoint=checkpoint, metadata_root=metadata, optimizer_step=500,
            identity=_identity(),
        )

    assert not list(metadata.glob("checkpoint-500.*"))


def test_marker_repair_fsync_failure_does_not_write_the_completion_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lehome_train.groot.local_recovery as recovery

    checkpoint, metadata = _official(tmp_path / "run", 500), tmp_path / "shared"
    receipt = recovery.attest_local_checkpoint(
        checkpoint=checkpoint, metadata_root=metadata, optimizer_step=500, identity=_identity(),
    )
    marker = receipt.receipt_path.with_suffix(".COMPLETE")
    marker.unlink()
    monkeypatch.setattr(
        recovery, "_fsync_regular_file",
        lambda _path: (_ for _ in ()).throw(OSError("repair fsync failed")),
    )

    with pytest.raises(OSError, match="repair fsync failed"):
        recovery.discover_local_recovery(metadata_root=metadata, identity=_identity())

    assert not marker.exists()


def test_attestation_rejects_trainer_state_only_checkpoint_without_creating_a_receipt(tmp_path: Path) -> None:
    from lehome_train.groot.local_recovery import attest_local_checkpoint

    checkpoint = tmp_path / "run" / "checkpoint-500"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 500, "log_history": [{"step": 500, "loss": .25}]}),
        encoding="utf-8",
    )
    metadata = tmp_path / "shared"

    with pytest.raises(ValueError, match="safetensors model weights"):
        attest_local_checkpoint(
            checkpoint=checkpoint, metadata_root=metadata, optimizer_step=500, identity=_identity(),
        )

    assert not list(metadata.glob("checkpoint-500.*"))


def test_attestation_leaves_official_checkpoint_untouched_and_streams_tree_receipt(tmp_path: Path) -> None:
    from lehome_train.groot.local_recovery import attest_local_checkpoint

    checkpoint = _official(tmp_path / "run", 500)
    original = sorted(item.relative_to(checkpoint).as_posix() for item in checkpoint.rglob("*"))
    local = attest_local_checkpoint(
        checkpoint=checkpoint, metadata_root=tmp_path / "shared-recovery",
        optimizer_step=500, identity=_identity(),
    )

    assert local.optimizer_step == 500
    assert sorted(item.relative_to(checkpoint).as_posix() for item in checkpoint.rglob("*")) == original
    receipt = json.loads(local.receipt_path.read_text())
    assert receipt["global_sample_offset"] == 32_000
    assert [item["path"] for item in receipt["checkpoint_tree"]] == ["model.safetensors", "optimizer.pt", "rng_state.pth", "scheduler.pt", "trainer_state.json"]
    assert all(type(item["size"]) is int and len(item["sha256"]) == 64 for item in receipt["checkpoint_tree"])
    assert (local.receipt_path.with_suffix(".COMPLETE")).is_file()


def test_discovery_repairs_a_valid_checkpoint_receipt_missing_only_its_marker(tmp_path: Path) -> None:
    from lehome_train.groot.local_recovery import attest_local_checkpoint, discover_local_recovery

    checkpoint = _official(tmp_path / "run", 500)
    metadata = tmp_path / "shared"
    receipt = attest_local_checkpoint(
        checkpoint=checkpoint, metadata_root=metadata, optimizer_step=500, identity=_identity(),
    )
    receipt.receipt_path.with_suffix(".COMPLETE").unlink()

    recovered = discover_local_recovery(metadata_root=metadata, identity=_identity())
    assert recovered is not None and recovered.receipt_sha256 == receipt.receipt_sha256
    assert receipt.receipt_path.with_suffix(".COMPLETE").is_file()


def test_discovery_rejects_a_malformed_markerless_checkpoint_receipt(tmp_path: Path) -> None:
    from lehome_train.groot.local_recovery import attest_local_checkpoint, discover_local_recovery

    checkpoint = _official(tmp_path / "run", 500)
    metadata = tmp_path / "shared"
    receipt = attest_local_checkpoint(
        checkpoint=checkpoint, metadata_root=metadata, optimizer_step=500, identity=_identity(),
    )
    receipt.receipt_path.with_suffix(".COMPLETE").unlink()
    payload = json.loads(receipt.receipt_path.read_text(encoding="utf-8"))
    payload["global_sample_offset"] = 1
    receipt.receipt_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="identity or cursor"):
        discover_local_recovery(metadata_root=metadata, identity=_identity())
    assert not receipt.receipt_path.with_suffix(".COMPLETE").exists()


def test_discovery_prefers_newest_local_and_fails_closed_for_partial_or_tree_drift(tmp_path: Path) -> None:
    from lehome_train.groot.local_recovery import attest_local_checkpoint, discover_local_recovery

    root, metadata = tmp_path / "run", tmp_path / "shared-recovery"
    five = _official(root, 500)
    fifteen = _official(root, 1500)
    attest_local_checkpoint(checkpoint=five, metadata_root=metadata, optimizer_step=500, identity=_identity())
    latest = attest_local_checkpoint(
        checkpoint=fifteen, metadata_root=metadata, optimizer_step=1500, identity=_identity(),
    )
    assert discover_local_recovery(metadata_root=metadata, identity=_identity()) == latest
    (fifteen / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="tree"):
        discover_local_recovery(metadata_root=metadata, identity=_identity())

    # A markerless sidecar is repairable only after exact validation; malformed
    # JSON remains fail-closed rather than becoming a recovery candidate.
    (fifteen / "model.safetensors").write_bytes(b"weights-1500")
    (metadata / "checkpoint-2000.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="identity or cursor"):
        discover_local_recovery(metadata_root=metadata, identity=_identity())


def test_local_post_one_k_receipts_are_independent_of_delayed_hf_lineage(tmp_path: Path) -> None:
    from lehome_train.groot.local_recovery import attest_local_checkpoint, discover_local_recovery

    checkpoint = _official(tmp_path / "run", 1500)
    local = attest_local_checkpoint(
        checkpoint=checkpoint, metadata_root=tmp_path / "shared", optimizer_step=1500,
        identity=_identity(),
    )
    discovered = discover_local_recovery(metadata_root=tmp_path / "shared", identity=_identity())
    assert discovered == local
    assert discovered.last_immutable_publication is None
    assert discovered.last_immutable_anchor is None
    receipt = json.loads(local.receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == 3
    assert "last_immutable_publication" not in receipt
    assert "last_immutable_anchor" not in receipt


def test_attestation_cancels_during_tree_hash_without_leaving_a_receipt(
    tmp_path: Path,
) -> None:
    import lehome_train.groot.local_recovery as recovery

    checkpoint, metadata = _official(tmp_path / "run", 500), tmp_path / "shared"
    calls = 0

    def cancel_during_hash() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 4

    with pytest.raises(recovery.AttestationCancelled, match="cancelled"):
        recovery.attest_local_checkpoint(
            checkpoint=checkpoint, metadata_root=metadata, optimizer_step=500,
            identity=_identity(), cancel_requested=cancel_during_hash,
        )
    assert not list(metadata.glob("checkpoint-500.*"))


def test_existing_receipt_revalidation_cancellation_never_repairs_its_missing_marker(tmp_path: Path) -> None:
    import lehome_train.groot.local_recovery as recovery

    metadata = tmp_path / "shared"
    checkpoint = _official(tmp_path / "run", 500)
    receipt = recovery.attest_local_checkpoint(
        checkpoint=checkpoint, metadata_root=metadata, optimizer_step=500, identity=_identity(),
    )
    marker = receipt.receipt_path.with_suffix(".COMPLETE")
    marker.unlink()
    checks = 0

    def cancel_during_revalidation() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 4

    with pytest.raises(recovery.AttestationCancelled, match="cancelled"):
        recovery._read_receipt(
            receipt.receipt_path, identity=_identity(), cancel_requested=cancel_during_revalidation,
        )
    assert not marker.exists()


def test_local_2000_is_not_terminal_until_its_immutable_publication_journal_is_complete(tmp_path: Path) -> None:
    from lehome_train.groot.local_recovery import (
        attest_local_checkpoint,
        discover_local_recovery,
        record_immutable_publication,
    )

    checkpoint = _official(tmp_path / "run", 2000)
    metadata = tmp_path / "shared"
    local = attest_local_checkpoint(
        checkpoint=checkpoint, metadata_root=metadata, optimizer_step=2000, identity=_identity(),
    )
    assert discover_local_recovery(metadata_root=metadata, identity=_identity()).terminal_immutable_publication is None

    publication = {"optimizer_step": 2000, "readback_verified": True, "immutable_revision": "1" * 40}
    anchor = {"immutable_anchor_revision": "1" * 40, "anchor_sha256": "2" * 64}
    record_immutable_publication(
        metadata_root=metadata, checkpoint=local, publication=publication, anchor=anchor,
    )
    discovered = discover_local_recovery(metadata_root=metadata, identity=_identity())
    assert discovered.terminal_immutable_publication == publication
    assert discovered.terminal_immutable_anchor == anchor


def test_discovery_treats_valid_markerless_two_k_publication_journal_as_nonterminal(tmp_path: Path) -> None:
    from lehome_train.groot.local_recovery import (
        attest_local_checkpoint,
        discover_local_recovery,
        record_immutable_publication,
    )

    metadata = tmp_path / "shared"
    local = attest_local_checkpoint(
        checkpoint=_official(tmp_path / "run", 2000), metadata_root=metadata,
        optimizer_step=2000, identity=_identity(),
    )
    publication = {"optimizer_step": 2000, "readback_verified": True, "immutable_revision": "1" * 40}
    anchor = {"immutable_anchor_revision": "1" * 40, "anchor_sha256": "2" * 64}
    record_immutable_publication(
        metadata_root=metadata, checkpoint=local, publication=publication, anchor=anchor,
    )
    marker = metadata / "publication-2000.COMPLETE"
    marker.unlink()

    recovered = discover_local_recovery(metadata_root=metadata, identity=_identity())
    assert recovered is not None
    assert recovered.terminal_immutable_publication is None
    assert not marker.exists()


def test_discovery_does_not_authorize_a_tampered_markerless_publication_journal(tmp_path: Path) -> None:
    from lehome_train.groot.local_recovery import (
        attest_local_checkpoint,
        discover_local_recovery,
        record_immutable_publication,
    )

    metadata = tmp_path / "shared"
    local = attest_local_checkpoint(
        checkpoint=_official(tmp_path / "run", 2000), metadata_root=metadata,
        optimizer_step=2000, identity=_identity(),
    )
    record_immutable_publication(
        metadata_root=metadata, checkpoint=local,
        publication={"optimizer_step": 2000, "readback_verified": True, "immutable_revision": "1" * 40},
        anchor={"immutable_anchor_revision": "1" * 40, "anchor_sha256": "2" * 64},
    )
    journal = metadata / "publication-2000.json"
    journal.with_suffix(".COMPLETE").unlink()
    payload = json.loads(journal.read_text(encoding="utf-8"))
    payload["checkpoint_receipt_sha256"] = "0" * 64
    journal.write_text(json.dumps(payload), encoding="utf-8")

    recovered = discover_local_recovery(metadata_root=metadata, identity=_identity())
    assert recovered is not None
    assert recovered.terminal_immutable_publication is None
    assert not journal.with_suffix(".COMPLETE").exists()


def test_local_1000_recovers_its_later_immutable_publication_anchor_without_rewriting_receipt(tmp_path: Path) -> None:
    from lehome_train.groot.local_recovery import (
        attest_local_checkpoint,
        discover_local_recovery,
        record_immutable_publication,
    )

    checkpoint = _official(tmp_path / "run", 1000)
    metadata = tmp_path / "shared"
    local = attest_local_checkpoint(
        checkpoint=checkpoint, metadata_root=metadata, optimizer_step=1000,
        identity=_identity(),
    )
    publication = {"optimizer_step": 1000, "readback_verified": True, "immutable_revision": "1" * 40}
    anchor = {"immutable_anchor_revision": "1" * 40, "anchor_sha256": "2" * 64}
    record_immutable_publication(
        metadata_root=metadata, checkpoint=local, publication=publication, anchor=anchor,
    )

    recovered = discover_local_recovery(metadata_root=metadata, identity=_identity())
    assert recovered.receipt_sha256 == local.receipt_sha256
    assert recovered.last_immutable_publication == publication
    assert recovered.last_immutable_anchor == anchor

    # A publication sidecar cannot stand in for the completed local checkpoint
    # it claims to bind; that visible orphan is a fail-closed recovery state.
    (metadata / "checkpoint-1000.json").unlink()
    with pytest.raises(ValueError, match="publication journal"):
        discover_local_recovery(metadata_root=metadata, identity=_identity())


def test_safe_legacy_v2_receipt_remains_readable_during_journal_migration(tmp_path: Path) -> None:
    from lehome_train.groot.local_recovery import attest_local_checkpoint, discover_local_recovery
    from lehome_train.io import canonical_json_sha256

    checkpoint = _official(tmp_path / "run", 1500)
    metadata = tmp_path / "shared"
    current = attest_local_checkpoint(
        checkpoint=checkpoint, metadata_root=metadata, optimizer_step=1500, identity=_identity(),
    )
    legacy = json.loads(current.receipt_path.read_text(encoding="utf-8"))
    legacy["schema_version"] = 2
    legacy["last_immutable_publication"] = {"optimizer_step": 1000, "immutable_revision": "f" * 40}
    legacy["last_immutable_anchor"] = {"immutable_anchor_revision": "f" * 40, "anchor_sha256": "0" * 64}
    current.receipt_path.write_text(json.dumps(legacy), encoding="utf-8")
    current.receipt_path.with_suffix(".COMPLETE").write_text(
        canonical_json_sha256(legacy) + "\n", encoding="ascii",
    )

    recovered = discover_local_recovery(metadata_root=metadata, identity=_identity())
    assert recovered is not None
    assert recovered.last_immutable_publication == legacy["last_immutable_publication"]


@pytest.mark.parametrize("field,value", [("schema_version", True), ("optimizer_step", 500.0), ("global_sample_offset", True), ("physical_batch_size", 64.0), ("action_horizon", True)])
def test_local_receipt_rejects_noncanonical_integer_boundary_types(tmp_path: Path, field: str, value: object) -> None:
    from lehome_train.groot.local_recovery import attest_local_checkpoint, discover_local_recovery

    checkpoint = _official(tmp_path / "run", 500)
    metadata = tmp_path / "shared"
    receipt = attest_local_checkpoint(checkpoint=checkpoint, metadata_root=metadata, optimizer_step=500, identity=_identity())
    payload = json.loads(receipt.receipt_path.read_text())
    payload[field] = value
    receipt.receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity or cursor"):
        discover_local_recovery(metadata_root=metadata, identity=_identity())
