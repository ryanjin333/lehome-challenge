from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from lehome_train.b1k.snapshot_integrity import (
    build_remote_manifest,
    build_snapshot_receipt,
    read_snapshot_json,
    validate_local_snapshot,
    validate_snapshot_receipt,
    verify_artifact_stat_invariants,
)


_REPOSITORY = "owner/repository"
_REVISION = "1" * 40


def _git_blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _entry(path: str, data: bytes, *, lfs: bool = False) -> dict[str, object]:
    return {
        "path": path,
        "size": len(data),
        "blob_id": _git_blob(data),
        "lfs": {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()} if lfs else None,
    }


def _manifest(*entries: dict[str, object]):
    return build_remote_manifest(
        repository=_REPOSITORY,
        revision=_REVISION,
        resolved_revision=_REVISION,
        entries=entries,
        allow_patterns=("meta/**", "data/**"),
    )


def test_remote_manifest_uses_git_blob_identity_for_regular_files(tmp_path: Path) -> None:
    data = b"git object\n"
    manifest = _manifest(_entry("meta/info.json", data))
    (tmp_path / "meta").mkdir(); (tmp_path / "meta/info.json").write_bytes(data)

    validation = validate_local_snapshot(tmp_path, manifest)

    assert manifest.entries[0].identity_kind == "git_blob_sha1"
    assert manifest.entries[0].identity == _git_blob(data)
    assert validation.hash_passes == 1
    assert validation.artifacts[0].git_blob_sha1 == _git_blob(data)


def test_remote_manifest_rejects_an_empty_selected_payload_set() -> None:
    with pytest.raises(ValueError, match="empty"):
        _manifest(_entry("outside-allowlist.bin", b"payload"))


def test_remote_manifest_uses_sha256_identity_for_lfs_files(tmp_path: Path) -> None:
    data = b"lfs payload\n"
    manifest = _manifest(_entry("data/chunk.bin", data, lfs=True))
    (tmp_path / "data").mkdir(); (tmp_path / "data/chunk.bin").write_bytes(data)

    validation = validate_local_snapshot(tmp_path, manifest)

    assert manifest.entries[0].identity_kind == "sha256"
    assert validation.artifacts[0].sha256 == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize(
    "entries",
    [
        (_entry("meta/../escape.json", b"x"),),
        (_entry("meta/one.json", b"x"), _entry("meta/one.json", b"x")),
        ({"path": "meta", "tree_id": "f" * 40, "size": 1, "blob_id": "a" * 40},),
        ({"path": "meta/unknown.json", "size": 1, "blob_id": "a" * 40, "lfs": {}},),
    ],
)
def test_remote_manifest_rejects_unsafe_duplicate_directory_and_ambiguous_entries(entries: tuple[dict[str, object], ...]) -> None:
    with pytest.raises(ValueError, match="manifest"):
        _manifest(*entries)


@pytest.mark.parametrize("actual", [b"", b"partial"])
def test_authoritative_manifest_rejects_successful_looking_empty_or_partial_download(tmp_path: Path, actual: bytes) -> None:
    expected = b"complete Cosmos payload"
    manifest = _manifest(_entry("data/cosmos.bin", expected, lfs=True))
    (tmp_path / "data").mkdir(); (tmp_path / "data/cosmos.bin").write_bytes(actual)

    with pytest.raises(ValueError, match="identity|size"):
        validate_local_snapshot(tmp_path, manifest)


def test_authoritative_manifest_rejects_missing_extra_and_bad_hash(tmp_path: Path) -> None:
    expected = b"expected"
    manifest = _manifest(_entry("data/payload.bin", expected, lfs=True))
    (tmp_path / "data").mkdir(); (tmp_path / "data/payload.bin").write_bytes(b"wrong!!!")
    (tmp_path / "data/extra.bin").write_bytes(b"extra")

    with pytest.raises(ValueError, match="extra"):
        validate_local_snapshot(tmp_path, manifest)
    (tmp_path / "data/extra.bin").unlink()
    with pytest.raises(ValueError, match="identity"):
        validate_local_snapshot(tmp_path, manifest)
    (tmp_path / "data/payload.bin").unlink()
    with pytest.raises(ValueError, match="missing"):
        validate_local_snapshot(tmp_path, manifest)


def test_local_validation_excludes_only_root_hf_transfer_cache_and_rejects_links(tmp_path: Path) -> None:
    data = b"payload"
    manifest = _manifest(_entry("meta/info.json", data))
    (tmp_path / "meta").mkdir(); (tmp_path / "meta/info.json").write_bytes(data)
    (tmp_path / ".cache").mkdir(); (tmp_path / ".cache/state").write_text("transfer")
    validation = validate_local_snapshot(tmp_path, manifest)
    assert [artifact.path for artifact in validation.artifacts] == ["meta/info.json"]
    (tmp_path / "meta/link").symlink_to("info.json")
    with pytest.raises(ValueError, match="symlink"):
        validate_local_snapshot(tmp_path, manifest)


def test_receipt_rejects_duplicate_keys_and_reuses_one_hashed_artifact_table(tmp_path: Path) -> None:
    data = b"one pass only"
    manifest = _manifest(_entry("meta/info.json", data))
    (tmp_path / "meta").mkdir(); (tmp_path / "meta/info.json").write_bytes(data)
    validation = validate_local_snapshot(tmp_path, manifest)
    receipt = build_snapshot_receipt(
        repository=_REPOSITORY,
        revision=_REVISION,
        allow_patterns=("meta/**", "data/**"),
        remote_manifest=manifest,
        artifacts=validation.artifacts,
        manifest_hashes={"selection": "a" * 64},
    )
    receipt_path = tmp_path / ".b1k-snapshot-receipt.json"
    receipt_path.write_text(json.dumps(receipt))

    reused = validate_snapshot_receipt(
        receipt_path,
        repository=_REPOSITORY,
        revision=_REVISION,
        allow_patterns=("meta/**", "data/**"),
        remote_manifest=manifest,
        validation=validation,
    )
    assert reused["remote_manifest_sha256"] == manifest.sha256
    assert reused["hash_passes"] == 1
    verify_artifact_stat_invariants(tmp_path, validation.artifacts)

    receipt_path.write_text('{"schema_version":2,"schema_version":2}')
    with pytest.raises(ValueError, match="duplicate"):
        read_snapshot_json(receipt_path, "receipt")


def test_cached_validation_hashes_each_payload_once_per_cycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = b"payload"
    manifest = _manifest(_entry("meta/info.json", data))
    (tmp_path / "meta").mkdir(); (tmp_path / "meta/info.json").write_bytes(data)
    import lehome_train.b1k.snapshot_integrity as integrity

    calls = 0
    real = integrity._hash_file_once

    def counted(path: Path):
        nonlocal calls
        calls += 1
        return real(path)

    monkeypatch.setattr(integrity, "_hash_file_once", counted)
    validation = validate_local_snapshot(tmp_path, manifest)
    receipt = build_snapshot_receipt(
        repository=_REPOSITORY, revision=_REVISION, allow_patterns=("meta/**", "data/**"),
        remote_manifest=manifest, artifacts=validation.artifacts, manifest_hashes={},
    )
    receipt_path = tmp_path / ".b1k-snapshot-receipt.json"; receipt_path.write_text(json.dumps(receipt))
    validate_snapshot_receipt(
        receipt_path, repository=_REPOSITORY, revision=_REVISION, allow_patterns=("meta/**", "data/**"),
        remote_manifest=manifest, validation=validation,
    )

    assert calls == 1
    assert validation.hash_passes == 1


def test_hub_adapter_uses_recursive_pinned_tree_metadata_for_the_authoritative_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    from lehome_train.b1k.bootstrap import HfHubAdapter

    data = b"weights"
    calls: list[dict[str, object]] = []

    class Api:
        def repo_info(self, **kwargs: object) -> object:
            calls.append(kwargs)
            return {"sha": _REVISION}

        def list_repo_tree(self, *args: object, **kwargs: object) -> object:
            calls.append({"repo_id": args[0], **kwargs})
            return [SimpleNamespace(path="weights.bin", size=len(data), blob_id=_git_blob(data), lfs=None)]

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=Api))
    manifest = HfHubAdapter().remote_manifest(_REPOSITORY, revision=_REVISION, allow_patterns=None, token="memory-token")

    assert manifest.entries[0].identity == _git_blob(data)
    assert calls[1] == {
        "repo_id": _REPOSITORY, "recursive": True, "revision": _REVISION,
        "repo_type": "model", "token": "memory-token",
    }
