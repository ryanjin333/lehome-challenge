from __future__ import annotations

import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pytest

from lehome_train.b1k import models
from lehome_train.b1k import snapshot_state
from lehome_train.b1k.models import derive_groot_config
from lehome_train.constants import COSMOS_REPOSITORY, COSMOS_REVISION


def _cosmos_identity(receipt: str = "c" * 64) -> dict[str, str]:
    return {
        "repository": COSMOS_REPOSITORY,
        "revision": COSMOS_REVISION,
        "receipt_sha256": receipt,
        "artifacts_sha256": "d" * 64,
    }


def test_derived_groot_model_reuses_weights_and_only_patches_cosmos_config(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"; upstream.mkdir(); (upstream / "weights.safetensors").write_bytes(b"weights")
    (upstream / "experiment_cfg").mkdir(); (upstream / "experiment_cfg" / "nested.json").write_text('{"kept":true}')
    (upstream / "config.json").write_text(json.dumps({"model_name": "old", "other": 1}))
    derived = tmp_path / "derived"
    receipt = derive_groot_config(upstream, derived, cosmos_path="/workspace/models/cosmos", cosmos_identity=_cosmos_identity())
    assert json.loads((derived / "config.json").read_text()) == {"model_name": "/workspace/models/cosmos", "other": 1}
    assert (derived / "weights.safetensors").read_bytes() == b"weights"
    assert (derived / "experiment_cfg" / "nested.json").read_text() == '{"kept":true}'
    assert receipt["upstream_config_sha256"] != receipt["derived_config_sha256"]
    assert derive_groot_config(upstream, derived, cosmos_path="/workspace/models/cosmos", cosmos_identity=_cosmos_identity()) == receipt


def test_derived_groot_model_fails_closed_when_hardlinking_is_cross_filesystem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = tmp_path / "upstream"; upstream.mkdir(); (upstream / "config.json").write_text('{"model_name":"upstream"}')
    (upstream / "weights.bin").write_bytes(b"weights")
    monkeypatch.setattr(models.os, "link", lambda _source, _target: (_ for _ in ()).throw(OSError(__import__("errno").EXDEV, "cross-device")))
    with pytest.raises(ValueError, match="hardlinks"):
        derive_groot_config(upstream, tmp_path / "derived", cosmos_identity=_cosmos_identity())


def test_derived_groot_model_fails_closed_for_an_invalid_completed_tree(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"; upstream.mkdir()
    (upstream / "config.json").write_text('{"model_name":"upstream"}')
    (upstream / "nested").mkdir(); (upstream / "nested" / "weights.safetensors").write_bytes(b"complete")
    derived = tmp_path / "derived"; derived.mkdir()
    (derived / "config.json").write_text('{"model_name":"/workspace/models/cosmos"}')
    with pytest.raises(ValueError, match="receipt"):
        derive_groot_config(upstream, derived, cosmos_path="/workspace/models/cosmos", cosmos_identity=_cosmos_identity())
    assert (derived / "config.json").exists()


def test_derived_groot_model_reuses_only_a_hash_valid_receipted_tree(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"; upstream.mkdir()
    (upstream / "config.json").write_text('{"model_name":"upstream"}')
    (upstream / "weights.safetensors").write_bytes(b"complete")
    derived = tmp_path / "derived"
    receipt = derive_groot_config(upstream, derived, cosmos_identity=_cosmos_identity())
    assert (derived / ".b1k-derived-receipt.json").is_file()
    assert derive_groot_config(upstream, derived, cosmos_identity=_cosmos_identity()) == receipt
    (derived / "weights.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="receipt"):
        derive_groot_config(upstream, derived, cosmos_identity=_cosmos_identity())


def test_derived_groot_model_rejects_a_tampered_config_even_when_model_name_is_unchanged(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"; upstream.mkdir(); (upstream / "config.json").write_text('{"model_name":"upstream"}')
    (upstream / "weights.bin").write_bytes(b"weights")
    derived = tmp_path / "derived"; derive_groot_config(upstream, derived, cosmos_identity=_cosmos_identity())
    (derived / "config.json").write_text('{"model_name":"/workspace/models/cosmos","extra":true}')
    with pytest.raises(ValueError, match="receipt"):
        derive_groot_config(upstream, derived, cosmos_identity=_cosmos_identity())


def test_derived_groot_model_rejects_unsafe_incomplete_or_upstream_symlinks(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"; upstream.mkdir()
    (upstream / "config.json").write_text('{"model_name":"upstream"}')
    (upstream / "outside").write_bytes(b"unsafe")
    (upstream / "weights.safetensors").symlink_to(upstream / "outside")
    with pytest.raises(ValueError, match="unsafe"):
        derive_groot_config(upstream, tmp_path / "derived", cosmos_identity=_cosmos_identity())

    safe = tmp_path / "safe"; safe.mkdir(); (safe / "config.json").write_text('{"model_name":"upstream"}')
    (safe / "weights.safetensors").write_bytes(b"complete")
    staging = tmp_path / ".derived.incomplete"; staging.mkdir(); (staging / "foreign").write_bytes(b"foreign")
    with pytest.raises(ValueError, match="incomplete"):
        derive_groot_config(safe, tmp_path / "derived", cosmos_identity=_cosmos_identity())


def test_derived_groot_model_resumes_a_matching_partial_sibling_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = tmp_path / "upstream"; upstream.mkdir(); (upstream / "config.json").write_text('{"model_name":"upstream"}')
    (upstream / "first.bin").write_bytes(b"first"); (upstream / "second.bin").write_bytes(b"second")
    derived = tmp_path / "derived"
    original_link = models.os.link
    attempts = 0

    def interrupted_link(source: Path, target: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise OSError("interrupted")
        original_link(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(models.os, "link", interrupted_link)
        with pytest.raises(OSError, match="interrupted"):
            derive_groot_config(upstream, derived, cosmos_identity=_cosmos_identity())
    staging = tmp_path / ".derived.incomplete"
    assert (staging / ".b1k-derived-intent.json").is_file()
    assert derive_groot_config(upstream, derived, cosmos_identity=_cosmos_identity())["upstream_config_sha256"]
    assert (derived / "first.bin").read_bytes() == b"first"
    assert (derived / "second.bin").read_bytes() == b"second"


@pytest.mark.parametrize("mutation", ["intent", "extra", "symlink"])
def test_derived_groot_model_rejects_mismatched_or_unsafe_partial_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    upstream = tmp_path / "upstream"; upstream.mkdir(); (upstream / "config.json").write_text('{"model_name":"upstream"}')
    (upstream / "weights.bin").write_bytes(b"weights")
    derived = tmp_path / "derived"
    with monkeypatch.context() as patch:
        patch.setattr(models.os, "link", lambda _source, _target: (_ for _ in ()).throw(OSError("interrupted")))
        with pytest.raises(OSError, match="interrupted"):
            derive_groot_config(upstream, derived, cosmos_identity=_cosmos_identity())
    staging = tmp_path / ".derived.incomplete"
    if mutation == "intent":
        value = json.loads((staging / ".b1k-derived-intent.json").read_text()); value["cosmos_path"] = "/wrong"
        (staging / ".b1k-derived-intent.json").write_text(json.dumps(value))
    elif mutation == "extra":
        (staging / "foreign.bin").write_bytes(b"foreign")
    else:
        (staging / "link.bin").symlink_to(upstream / "weights.bin")
    with pytest.raises(ValueError, match="incomplete"):
        derive_groot_config(upstream, derived, cosmos_identity=_cosmos_identity())


def test_derived_groot_model_reuse_is_bound_to_the_validated_cosmos_snapshot_identity(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"; upstream.mkdir(); (upstream / "config.json").write_text('{"model_name":"upstream"}')
    (upstream / "weights.bin").write_bytes(b"weights")
    derived = tmp_path / "derived"
    derive_groot_config(upstream, derived, cosmos_identity=_cosmos_identity())
    with pytest.raises(ValueError, match="receipt"):
        derive_groot_config(upstream, derived, cosmos_identity=_cosmos_identity("e" * 64))


def test_derived_groot_model_rejects_a_staging_intent_left_in_a_completed_tree(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"; upstream.mkdir(); (upstream / "config.json").write_text('{"model_name":"upstream"}')
    (upstream / "weights.bin").write_bytes(b"weights")
    derived = tmp_path / "derived"
    derive_groot_config(upstream, derived, cosmos_identity=_cosmos_identity())
    (derived / ".b1k-derived-intent.json").write_text("{}")
    with pytest.raises(ValueError, match="receipt"):
        derive_groot_config(upstream, derived, cosmos_identity=_cosmos_identity())


def test_derived_model_serializes_same_destination_and_holds_lock_through_post_promotion_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = tmp_path / "upstream"; upstream.mkdir(); (upstream / "config.json").write_text('{"model_name":"upstream"}')
    (upstream / "weights.bin").write_bytes(b"weights")
    derived = tmp_path / "derived"
    original_validate = models._valid_derived_receipt
    post_promotion_seen = False

    def check_lock(*args: object, **kwargs: object):
        nonlocal post_promotion_seen
        value = original_validate(*args, **kwargs)
        if args[1] == derived and derived.exists():
            held = getattr(snapshot_state._held_locks, "paths", set())
            assert str((tmp_path / ".derived.lock").absolute()) in held
            post_promotion_seen = True
        return value

    monkeypatch.setattr(models, "_valid_derived_receipt", check_lock)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: derive_groot_config(upstream, derived, cosmos_identity=_cosmos_identity()), range(2)))
    assert results[0] == results[1]
    assert post_promotion_seen
    assert not (tmp_path / ".derived.incomplete").exists()


def test_derived_config_is_atomic_and_stage_is_synced_after_intent_removal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = tmp_path / "upstream"; upstream.mkdir(); (upstream / "config.json").write_text('{"model_name":"upstream"}')
    (upstream / "weights.bin").write_bytes(b"weights")
    derived = tmp_path / "derived"
    writes: list[Path] = []; syncs: list[Path] = []
    original_write, original_sync = models.atomic_write_json, models.fsync_directory

    def record_write(path: Path, value: object) -> None:
        writes.append(path); original_write(path, value)

    def record_sync(path: Path) -> None:
        syncs.append(path); original_sync(path)

    monkeypatch.setattr(models, "atomic_write_json", record_write)
    monkeypatch.setattr(models, "fsync_directory", record_sync)
    derive_groot_config(upstream, derived, cosmos_identity=_cosmos_identity())
    assert any(path.name == "config.json" for path in writes)
    assert any(path.name == ".derived.incomplete" for path in syncs)


def test_derived_parent_swap_immediately_before_promotion_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = tmp_path / "upstream"; upstream.mkdir(); (upstream / "config.json").write_text('{"model_name":"upstream"}')
    (upstream / "weights.bin").write_bytes(b"weights")
    parent = tmp_path / "parent"; parent.mkdir(); derived = parent / "derived"
    original_sync = models.fsync_directory
    swapped = False

    def swap_before_rename(path: Path) -> None:
        nonlocal swapped
        original_sync(path)
        if path.name == ".derived.incomplete" and not swapped:
            swapped = True
            parent.rename(tmp_path / "moved-parent"); parent.mkdir()

    monkeypatch.setattr(models, "fsync_directory", swap_before_rename)
    with pytest.raises(ValueError, match="parent"):
        derive_groot_config(upstream, derived, cosmos_identity=_cosmos_identity())
    assert swapped and not derived.exists()
