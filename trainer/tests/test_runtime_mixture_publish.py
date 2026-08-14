from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from lehome_train.hub import HubAccess, HubTreeEntry
from lehome_train.io import sha256_file


TOKEN = "hf_fake_process_token_only"
REPOSITORY = "ryanjin333/lehome-groot-n17-data"
REVISION = "a" * 40


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")


class MemoryTransport:
    """Literal fake private Hub transport; it never opens a network socket."""

    def __init__(self, *, fault: str | None = None) -> None:
        self.fault = fault
        self.remote: dict[str, bytes] = {}
        self.calls: list[tuple[str, str, str | None]] = []

    def check_access(self, *, repository: str, token: str) -> HubAccess:
        assert repository == REPOSITORY
        assert token == TOKEN
        return HubAccess(can_read=True, can_write=True, private_repository=True)

    def upload_files(self, *, repository: str, revision: str, source: Path, entries, token: str, remote_prefix: str | None = None) -> str:
        assert repository == REPOSITORY and token == TOKEN and remote_prefix is not None
        self.calls.append(("upload", revision, remote_prefix))
        if self.fault == "upload":
            raise OSError("hf_token_looks_real_but_is_fake")
        uploaded = list(entries)
        if self.fault == "partial-upload":
            uploaded.pop()
        self.remote.update({
            f"{remote_prefix}/{entry.relative_path}": (source / entry.relative_path).read_bytes()
            for entry in uploaded
        })
        if self.fault == "extra-remote-file":
            self.remote[f"{remote_prefix}/unexpected.bin"] = b"unexpected"
        return REVISION

    def list_tree(self, *, repository: str, revision: str, token: str) -> tuple[HubTreeEntry, ...]:
        assert repository == REPOSITORY and token == TOKEN
        self.calls.append(("tree", revision, None))
        if self.fault == "list":
            raise OSError("tree failed")
        return tuple(HubTreeEntry(path, "file") for path in sorted(self.remote))

    def download_files(self, *, repository: str, revision: str, destination: Path, relative_paths, token: str, remote_prefix: str | None = None) -> str:
        assert repository == REPOSITORY and token == TOKEN and remote_prefix is not None
        self.calls.append(("download", revision, remote_prefix))
        if self.fault == "download":
            raise OSError("download failed")
        for relative in relative_paths:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = self.remote[f"{remote_prefix}/{relative}"]
            target.write_bytes(b"changed" if self.fault == "changed-readback" else payload)
        return "b" * 40 if self.fault == "wrong-readback-revision" else revision


@pytest.fixture(autouse=True)
def _token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", TOKEN)


def _source(root: Path) -> None:
    (root / "nested").mkdir(parents=True)
    (root / "manifest.json").write_bytes(b'{"schema":1}')
    (root / "nested" / "payload.bin").write_bytes(b"immutable")


def test_source_publisher_proves_complete_remote_tree_and_fresh_bytes_without_mutating_source(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture import source_tree_sha256
    from lehome_train.groot.runtime_mixture_publish import publish_source

    source, receipt_path = tmp_path / "bc", tmp_path / "receipts" / "bc.json"
    _source(source)
    receipt_path.parent.mkdir()
    before = source_tree_sha256(source)
    receipt = publish_source(root=source, source_type="bc", round_id=None, revision="draft", receipt_path=receipt_path, transport=MemoryTransport())

    assert receipt == {
        "repository": REPOSITORY, "immutable_revision": REVISION,
        "remote_prefix": "bc/full", "fresh_readback_verified": True,
        "tree_listing_verified": True,
    }
    assert source_tree_sha256(source) == before
    assert receipt_path.is_file() and not (source / "bc.json").exists()


def test_hydrator_recreates_exact_remote_trees_and_rewrites_only_local_receipt_mounts(
    tmp_path: Path,
) -> None:
    from lehome_train.groot.runtime_mixture import load_runtime_contract
    from lehome_train.groot.runtime_mixture_publish import hydrate_runtime_mixture_from_request
    from test_runtime_mixture import _contract

    manifest, _index, _mounts = _contract(tmp_path / "authoring")
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    deployment = manifest.parent / "release-receipt.json"
    transport = MemoryTransport()
    for artifact in json.loads(deployment.read_text(encoding="utf-8"))["artifact_entries"]:
        relative = artifact["relative_path"]
        transport.remote[f"mixtures/{'d' * 64}/{relative}"] = (manifest.parent / relative).read_bytes()
    receipt_paths: dict[str, str] = {}
    for source in manifest_value["sources"]:
        root = manifest.parent / ("bc" if source["source_id"] == "bc" else "round-1")
        prefix = source["publication"]["prefix"]
        for file in root.rglob("*"):
            if file.is_file():
                transport.remote[f"{prefix}/{file.relative_to(root).as_posix()}"] = file.read_bytes()
        receipt_paths[source["source_id"]] = source["publication"]["readback_receipt_path"]
    request = tmp_path / "hydrate.json"
    destination = tmp_path / "hydrated" / "mixture"
    mounts = destination / "mounts.json"
    _write(request, {
        "schema_version": 1,
        "command": "hydrate-runtime-mixture",
        "arguments": {
            "deployment_receipt": str(deployment),
            "source_readback_receipts": receipt_paths,
            "destination": str(destination),
            "mounts_descriptor": str(mounts),
        },
    })

    result = hydrate_runtime_mixture_from_request(request, transport=transport)

    assert result["immutable_revision"] == "a" * 40
    assert load_runtime_contract(destination / "mixture.json", mounts).mounts["bc"] == destination.parent / "sources" / "bc"
    assert all("/authoring/" not in entry["source_readback_receipt_path"] for entry in json.loads(mounts.read_text())["mounts"])


@pytest.mark.parametrize("fault", ["partial-upload", "extra-remote-file", "changed-readback", "download"])
def test_source_publisher_rejects_incomplete_or_changed_remote_content(tmp_path: Path, fault: str) -> None:
    from lehome_train.groot.runtime_mixture_publish import publish_source

    source = tmp_path / "source"
    _source(source)
    with pytest.raises((RuntimeError, ValueError), match="upload|tree|readback|download|remote"):
        publish_source(root=source, source_type="rollout", round_id="12", revision="draft", receipt_path=tmp_path / "receipt.json", transport=MemoryTransport(fault=fault))
    assert not (tmp_path / "receipt.json").exists()


def test_source_publisher_never_leaks_token_shaped_transport_errors(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture_publish import publish_source

    source = tmp_path / "source"
    _source(source)
    with pytest.raises(RuntimeError) as error:
        publish_source(root=source, source_type="bc", round_id=None, revision="draft", receipt_path=tmp_path / "receipt.json", transport=MemoryTransport(fault="upload"))
    assert "hf_token_looks_real_but_is_fake" not in str(error.value)


@pytest.mark.parametrize("relative", ["credentials.json", "nested/token.txt"])
def test_publisher_uses_central_redaction_before_any_transport_call(tmp_path: Path, relative: str) -> None:
    from lehome_train.groot.runtime_mixture_publish import publish_source

    source = tmp_path / "source"
    _source(source)
    target = source / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("hf_" + "a" * 40 if relative.endswith(".txt") else "{}", encoding="utf-8")
    transport = MemoryTransport()
    with pytest.raises(ValueError, match="credential|token|access"):
        publish_source(root=source, source_type="bc", round_id=None, revision="draft", receipt_path=tmp_path / "receipt.json", transport=transport)
    assert transport.calls == []


def test_source_receipt_destination_is_preflighted_external_and_absent_before_access(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture_publish import publish_source

    source = tmp_path / "source"
    _source(source)
    transport = MemoryTransport()
    with pytest.raises((FileExistsError, ValueError), match="receipt|external|immutable"):
        publish_source(root=source, source_type="bc", round_id=None, revision="draft", receipt_path=source / "receipt.json", transport=transport)
    assert transport.calls == []


def test_existing_or_overlapping_pending_receipt_destination_is_rejected_before_access(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture_publish import publish_pending_mixture

    pending, _source_mounts = _pending_from_runtime_contract(tmp_path)
    transport = MemoryTransport()
    target = pending / "receipt.json"
    with pytest.raises((FileExistsError, ValueError), match="receipt|external|immutable"):
        publish_pending_mixture(pending_root=pending, revision="draft", receipt_path=target, transport=transport)
    assert transport.calls == []


def test_tree_match_rejects_link_or_special_entries_under_the_exact_prefix() -> None:
    from lehome_train.groot.runtime_mixture_publish import _tree_matches
    from lehome_train.models import SyncEntry

    entries = (SyncEntry("payload.json", "a" * 64, 1),)
    assert not _tree_matches((HubTreeEntry("bc/full/payload.json", "file"), HubTreeEntry("bc/full/link", "symlink")), prefix="bc/full", entries=entries)
    assert not _tree_matches((HubTreeEntry("bc/full/payload.json", "file"), HubTreeEntry("bc/full/device", "special")), prefix="bc/full", entries=entries)


def _pending_from_runtime_contract(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    from lehome_train.groot.runtime_mixture import pending_mixture_id
    from test_runtime_mixture import _contract

    manifest_path, index_path, mounts_path = _contract(tmp_path / "contract")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    mounts = json.loads(mounts_path.read_text(encoding="utf-8"))
    pending = tmp_path / "pending"
    pending.mkdir()
    normalization = manifest_path.parent / "mixture-normalization.json"
    shutil.copy2(normalization, pending / normalization.name)
    _write(pending / "windows.json", {"schema_version": 3, "windows": index["windows"]})
    pending_value = {
        "schema_version": 1, "kind": "runtime_mixture_publication_pending",
        "repository": REPOSITORY, "sources": manifest["sources"],
        "normalization_sha256": sha256_file(pending / normalization.name),
        "windows_sha256": sha256_file(pending / "windows.json"),
        "publication_pending": True,
    }
    mixture_id = pending_mixture_id(pending_value)
    _write(pending / "publication-pending.json", {
        **pending_value, "mixture_id": mixture_id, "prefix": f"mixtures/{mixture_id}",
    })
    return pending, {entry["source_id"]: entry["root"] for entry in mounts["mounts"]}


def test_pending_publisher_and_finalizer_emit_loadable_contract_with_no_hash_cycle(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture_publish import finalize_pending_mixture, publish_pending_mixture

    pending, source_mounts = _pending_from_runtime_contract(tmp_path)
    publication_receipt = tmp_path / "receipts" / "mixture.json"
    publication_receipt.parent.mkdir()
    transport = MemoryTransport()
    receipt = publish_pending_mixture(pending_root=pending, revision="draft", receipt_path=publication_receipt, transport=transport)
    deployment_receipt = tmp_path / "receipts" / "deployment.json"
    output = finalize_pending_mixture(pending_root=pending, publication_receipt=publication_receipt, destination=tmp_path / "final", deployment_receipt_path=deployment_receipt, source_mounts=source_mounts, revision="final-draft", transport=transport)

    assert receipt["immutable_revision"] == REVISION
    assert (output / "mixture.json").is_file()
    assert (output / "windows.json").is_file()
    assert (output / "mounts.json").is_file()
    prefix = f"mixtures/{receipt['mixture_id']}/"
    expected = {
        f"{prefix}{path.relative_to(output).as_posix()}": path.read_bytes()
        for path in output.rglob("*")
        if path.is_file() and path.name != "mounts.json"
    }
    assert transport.remote == expected


def test_pending_publisher_recomputes_its_content_address_before_access(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture_publish import publish_pending_mixture

    pending, _source_mounts = _pending_from_runtime_contract(tmp_path)
    payload = json.loads((pending / "publication-pending.json").read_text(encoding="utf-8"))
    payload["mixture_id"] = "f" * 64
    payload["prefix"] = "mixtures/" + "f" * 64
    _write(pending / "publication-pending.json", payload)
    transport = MemoryTransport()
    with pytest.raises(ValueError, match="content|mixture|prefix"):
        publish_pending_mixture(pending_root=pending, revision="draft", receipt_path=tmp_path / "receipt.json", transport=transport)
    assert transport.calls == []


def test_finalizer_recomputes_pending_content_address_before_access(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture_publish import finalize_pending_mixture, publish_pending_mixture

    pending, source_mounts = _pending_from_runtime_contract(tmp_path)
    receipt = tmp_path / "pending-receipt.json"
    publish_pending_mixture(pending_root=pending, revision="draft", receipt_path=receipt, transport=MemoryTransport())
    payload = json.loads((pending / "publication-pending.json").read_text(encoding="utf-8"))
    payload["mixture_id"] = "f" * 64
    payload["prefix"] = "mixtures/" + "f" * 64
    _write(pending / "publication-pending.json", payload)
    transport = MemoryTransport()
    with pytest.raises(ValueError, match="content|mixture|prefix"):
        finalize_pending_mixture(pending_root=pending, publication_receipt=receipt, destination=tmp_path / "final", deployment_receipt_path=tmp_path / "deployment.json", source_mounts=source_mounts, revision="final-draft", transport=transport)
    assert transport.calls == []


@pytest.mark.parametrize("mutation", ["pending", "receipt"])
def test_finalizer_rejects_tampered_pending_bytes_or_release_receipt(tmp_path: Path, mutation: str) -> None:
    from lehome_train.groot.runtime_mixture_publish import finalize_pending_mixture, publish_pending_mixture

    pending, source_mounts = _pending_from_runtime_contract(tmp_path)
    receipt = tmp_path / "receipt.json"
    publish_pending_mixture(pending_root=pending, revision="draft", receipt_path=receipt, transport=MemoryTransport())
    target = pending / "windows.json" if mutation == "pending" else receipt
    value = json.loads(target.read_text(encoding="utf-8"))
    value["tampered"] = True
    _write(target, value)
    with pytest.raises(ValueError, match="drift|schema|invalid"):
        finalize_pending_mixture(pending_root=pending, publication_receipt=receipt, destination=tmp_path / "final", deployment_receipt_path=tmp_path / "deployment.json", source_mounts=source_mounts, revision="final-draft", transport=MemoryTransport())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repository", "other/private"),
        ("remote_prefix", "mixtures/" + "e" * 64),
        ("immutable_revision", "main"),
    ],
)
def test_finalizer_rejects_wrong_pending_receipt_repository_prefix_or_revision(tmp_path: Path, field: str, value: str) -> None:
    from lehome_train.groot.runtime_mixture_publish import finalize_pending_mixture, publish_pending_mixture

    pending, source_mounts = _pending_from_runtime_contract(tmp_path)
    receipt = tmp_path / "receipt.json"
    publish_pending_mixture(pending_root=pending, revision="draft", receipt_path=receipt, transport=MemoryTransport())
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload[field] = value
    _write(receipt, payload)
    with pytest.raises(ValueError, match="receipt|invalid"):
        finalize_pending_mixture(pending_root=pending, publication_receipt=receipt, destination=tmp_path / "final", deployment_receipt_path=tmp_path / "deployment.json", source_mounts=source_mounts, revision="final-draft", transport=MemoryTransport())


def test_publish_request_envelopes_are_exact_and_use_injected_transport(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture_publish import publish_source_from_request

    source = tmp_path / "source"
    _source(source)
    request = tmp_path / "request.json"
    _write(request, {"schema_version": 1, "command": "publish-runtime-source", "arguments": {"root": str(source), "source_type": "bc", "round_id": None, "revision": "draft", "receipt_path": str(tmp_path / "receipt.json")}})
    result = publish_source_from_request(request, transport=MemoryTransport())
    assert result["remote_prefix"] == "bc/full"
    _write(request, {"schema_version": 1, "command": "publish-runtime-source", "arguments": {"root": str(source), "source_type": "bc", "round_id": None, "revision": "draft", "receipt_path": str(tmp_path / "other.json"), "repository": REPOSITORY}})
    with pytest.raises(ValueError, match="incompatible|unknown"):
        publish_source_from_request(request, transport=MemoryTransport())


def test_pending_and_finalization_request_envelopes_use_the_same_fake_transport(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture_publish import (
        finalize_pending_mixture_from_request,
        publish_pending_mixture_from_request,
    )

    pending, source_mounts = _pending_from_runtime_contract(tmp_path)
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    transport = MemoryTransport()
    publish_request = tmp_path / "publish.json"
    pending_receipt = receipts / "pending.json"
    _write(publish_request, {"schema_version": 1, "command": "publish-runtime-mixture", "arguments": {"pending_root": str(pending), "revision": "pending-draft", "receipt_path": str(pending_receipt)}})
    assert publish_pending_mixture_from_request(publish_request, transport=transport)["immutable_revision"] == REVISION
    final_request = tmp_path / "final.json"
    destination = tmp_path / "final"
    _write(final_request, {"schema_version": 1, "command": "finalize-runtime-mixture", "arguments": {"pending_root": str(pending), "publication_receipt": str(pending_receipt), "destination": str(destination), "deployment_receipt_path": str(receipts / "deployment.json"), "source_mounts": source_mounts, "revision": "final-draft"}})
    assert finalize_pending_mixture_from_request(final_request, transport=transport) == destination
    bad = json.loads(final_request.read_text(encoding="utf-8"))
    bad["arguments"]["repository"] = REPOSITORY
    _write(final_request, bad)
    with pytest.raises(ValueError, match="incompatible|schema"):
        finalize_pending_mixture_from_request(final_request, transport=transport)
