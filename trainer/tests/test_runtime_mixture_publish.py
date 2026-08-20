from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from lehome_train.hub import HubAccess, HubRateLimitError, HubTransientError, HubTreeEntry
from lehome_train.io import sha256_file


TOKEN = "hf_fake_process_token_only"
REPOSITORY = "ryanjin333/lehome-groot-n17-data"
ROLLOUT_REPOSITORY = "ryanjin333/lehome-groot-n17-rollouts"
MIXTURE_REPOSITORY = ROLLOUT_REPOSITORY
REVISION = "a" * 40


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")


class MemoryTransport:
    """Literal fake private Hub transport; it never opens a network socket."""

    def __init__(
        self, *, fault: str | None = None, expected_revision: str | None = None,
        expected_repository: str = REPOSITORY,
    ) -> None:
        self.fault = fault
        self.expected_revision = expected_revision
        self.expected_repositories = {
            expected_repository,
        } if expected_repository != REPOSITORY else {REPOSITORY, ROLLOUT_REPOSITORY}
        self.remote: dict[str, bytes] = {}
        self.calls: list[tuple[str, str, str | None]] = []
        self.repository_calls: list[tuple[str, str, str | None]] = []
        self.large_uploads: list[dict[str, object]] = []
        self.branch_head = REVISION
        self.downloaded_files = 0
        self.download_batches: list[tuple[str, ...]] = []

    def check_access(self, *, repository: str, token: str) -> HubAccess:
        assert repository in self.expected_repositories
        assert token == TOKEN
        return HubAccess(can_read=True, can_write=True, private_repository=True)

    def upload_files(self, *, repository: str, revision: str, source: Path, entries, token: str, remote_prefix: str | None = None) -> str:
        assert repository in self.expected_repositories and token == TOKEN and remote_prefix is not None
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

    def upload_large_folder(
        self, *, repository: str, revision: str, source: Path, entries, token: str,
        remote_prefix: str, max_workers: int,
    ) -> None:
        """Provider-like resumable large upload, deliberately without a prefix API."""

        assert repository in self.expected_repositories and token == TOKEN
        assert 1 <= max_workers <= 8
        staged = tuple(entries)
        assert remote_prefix == "rollouts/round-1"
        assert staged and all(entry.relative_path.startswith(remote_prefix + "/") for entry in staged)
        assert all(not entry.relative_path.startswith(".cache/") for entry in staged)
        self.large_uploads.append({
            "revision": revision,
            "source": source,
            "entries": staged,
            "cache_present": (source / ".cache").exists(),
            "payloads": {entry.relative_path: (source / entry.relative_path).read_bytes() for entry in staged},
        })
        if self.fault == "large-interrupt":
            self.remote.update({
                entry.relative_path: (source / entry.relative_path).read_bytes()
                for entry in staged[:1]
            })
            cache = source / ".cache" / "huggingface"
            cache.mkdir(parents=True, exist_ok=True)
            (cache / "resume-state").write_bytes(b"resume")
            raise HubTransientError("resumable upload interrupted")
        if self.fault != "large-stale-head":
            self.remote.update({
                entry.relative_path: (source / entry.relative_path).read_bytes()
                for entry in staged
            })

    def resolve_approved_ref(self, *, repository: str, ref: str, token: str) -> str:
        assert repository in self.expected_repositories and token == TOKEN
        self.calls.append(("resolve", ref, None))
        return self.branch_head

    def list_tree(
        self, *, repository: str, revision: str, token: str,
        remote_prefix: str | None = None,
    ) -> tuple[HubTreeEntry, ...]:
        assert repository in self.expected_repositories and token == TOKEN
        if self.expected_revision is not None:
            assert revision == self.expected_revision
        self.calls.append(("tree", revision, remote_prefix))
        self.repository_calls.append(("tree", repository, remote_prefix))
        if self.fault == "list":
            raise OSError("tree failed")
        return tuple(HubTreeEntry(path, "file") for path in sorted(self.remote))

    def download_files(self, *, repository: str, revision: str, destination: Path, relative_paths, token: str, remote_prefix: str | None = None) -> str:
        assert repository in self.expected_repositories and token == TOKEN and remote_prefix is not None
        self.calls.append(("download", revision, remote_prefix))
        self.repository_calls.append(("download", repository, remote_prefix))
        self.download_batches.append(tuple(relative_paths))
        if self.fault == "download":
            raise OSError("download failed")
        for relative in relative_paths:
            if self.fault == "rate-limit-at-957" and self.downloaded_files >= 957:
                raise HubTransientError("rate limited")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = self.remote[f"{remote_prefix}/{relative}"]
            target.write_bytes(b"changed" if self.fault == "changed-readback" else payload)
            self.downloaded_files += 1
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
    transport = MemoryTransport()
    receipt = publish_source(root=source, source_type="bc", round_id=None, revision="draft", receipt_path=receipt_path, transport=transport)

    assert receipt == {
        "repository": REPOSITORY, "immutable_revision": REVISION,
        "remote_prefix": "bc/full", "fresh_readback_verified": True,
        "tree_listing_verified": True,
    }
    assert source_tree_sha256(source) == before
    assert receipt_path.is_file() and not (source / "bc.json").exists()
    assert ("tree", REVISION, "bc/full") in transport.calls


def test_rollout_source_publishes_to_the_private_rollout_repository(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture_publish import publish_source

    source = tmp_path / "rollout"
    _source(source)
    transport = MemoryTransport(expected_repository=ROLLOUT_REPOSITORY)

    receipt = publish_source(
        root=source,
        source_type="rollout",
        round_id="2",
        revision="draft",
        receipt_path=tmp_path / "round-2.json",
        transport=transport,
    )

    assert receipt["repository"] == ROLLOUT_REPOSITORY
    assert receipt["remote_prefix"] == "rollouts/round-2"


def test_large_source_stages_only_the_exact_prefixed_allowlist_then_binds_the_final_head(
    tmp_path: Path,
) -> None:
    from lehome_train.groot.runtime_mixture import source_tree_sha256
    from lehome_train.groot.runtime_mixture_publish import publish_source

    source = tmp_path / "source"
    _source(source)
    before = source_tree_sha256(source)
    staging = tmp_path / "external-resumable-state"
    transport = MemoryTransport()

    receipt = publish_source(
        root=source, source_type="rollout", round_id="1", revision="main",
        receipt_path=tmp_path / "receipt.json", upload_journal_path=tmp_path / "journal.json",
        readback_root=tmp_path / "readback", large_upload=True,
        large_upload_staging_root=staging, transport=transport,
    )

    assert receipt["immutable_revision"] == REVISION
    assert source_tree_sha256(source) == before
    assert not [call for call in transport.calls if call[0] == "upload"]
    assert transport.calls.count(("resolve", "main", None)) == 1
    assert len(transport.large_uploads) == 1
    upload = transport.large_uploads[0]
    assert upload["revision"] == "main" and upload["cache_present"] is False
    assert set(upload["payloads"]) == {
        "rollouts/round-1/manifest.json", "rollouts/round-1/nested/payload.bin",
    }
    assert not any(path.startswith(".cache/") for path in upload["payloads"])


def test_large_source_interruption_preserves_external_staging_for_retry_but_no_receipt(
    tmp_path: Path,
) -> None:
    from lehome_train.groot.runtime_mixture_publish import publish_source

    source = tmp_path / "source"
    _source(source)
    staging, journal, receipt = tmp_path / "external-state", tmp_path / "journal.json", tmp_path / "receipt.json"
    transport = MemoryTransport(fault="large-interrupt")
    arguments = {
        "root": source, "source_type": "rollout", "round_id": "1", "revision": "main",
        "receipt_path": receipt, "upload_journal_path": journal, "readback_root": tmp_path / "readback",
        "large_upload": True, "large_upload_staging_root": staging, "transport": transport,
    }

    with pytest.raises(RuntimeError, match="large"):
        publish_source(**arguments)

    assert (staging / ".cache" / "huggingface" / "resume-state").is_file()
    assert not journal.exists() and not receipt.exists()
    assert len(transport.large_uploads) == 1

    transport.fault = None
    result = publish_source(**arguments)

    assert result["fresh_readback_verified"] is True
    assert len(transport.large_uploads) == 2
    assert transport.large_uploads[1]["cache_present"] is True


def test_large_source_rejects_an_unchanged_stale_branch_head_before_journal_or_receipt(
    tmp_path: Path,
) -> None:
    from lehome_train.groot.runtime_mixture_publish import publish_source

    source = tmp_path / "source"
    _source(source)
    journal, receipt = tmp_path / "journal.json", tmp_path / "receipt.json"
    transport = MemoryTransport(fault="large-stale-head")

    with pytest.raises(ValueError, match="tree|remote"):
        publish_source(
            root=source, source_type="rollout", round_id="1", revision="main",
            receipt_path=receipt, upload_journal_path=journal, readback_root=tmp_path / "readback",
            large_upload=True, large_upload_staging_root=tmp_path / "external-state",
            transport=transport,
        )

    assert not journal.exists() and not receipt.exists()


@pytest.mark.parametrize("unsafe", ["extra-file", "symlink"])
def test_large_source_rejects_an_unsafe_or_nonexact_preexisting_staged_prefix(
    tmp_path: Path, unsafe: str,
) -> None:
    from lehome_train.groot.runtime_mixture_publish import publish_source

    source = tmp_path / "source"
    _source(source)
    staging = tmp_path / "external-state"
    prefix = staging / "rollouts" / "round-1"
    prefix.mkdir(parents=True)
    if unsafe == "extra-file":
        (prefix / "unexpected.bin").write_bytes(b"outside allowlist")
    else:
        prefix.rmdir()
        (staging / "rollouts").rmdir()
        other = tmp_path / "other"
        other.mkdir()
        prefix.parent.symlink_to(other, target_is_directory=True)
    journal, receipt = tmp_path / "journal.json", tmp_path / "receipt.json"
    transport = MemoryTransport()

    with pytest.raises(ValueError, match="staging|unsafe|allowlist"):
        publish_source(
            root=source, source_type="rollout", round_id="1", revision="main",
            receipt_path=receipt, upload_journal_path=journal, readback_root=tmp_path / "readback",
            large_upload=True, large_upload_staging_root=staging, transport=transport,
        )

    assert not transport.large_uploads and not journal.exists() and not receipt.exists()


def test_source_readback_resumes_fixed_safe_batches_after_rate_limit_without_reuploading(
    tmp_path: Path,
) -> None:
    from lehome_train.groot.runtime_mixture_publish import (
        publish_source,
        verify_uploaded_runtime_source,
    )

    source = tmp_path / "source"
    for index in range(4_007):
        path = source / "shards" / f"{index:04d}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{index}".encode("ascii"))
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    journal = receipts / "source-upload.json"
    receipt = receipts / "source-readback.json"
    readback = tmp_path / "stable-readback"
    transport = MemoryTransport(fault="rate-limit-at-957")

    with pytest.raises(RuntimeError, match="download"):
        publish_source(
            root=source,
            source_type="bc",
            round_id=None,
            revision="draft",
            receipt_path=receipt,
            upload_journal_path=journal,
            readback_root=readback,
            transport=transport,
        )

    assert journal.is_file()
    assert not receipt.exists()
    assert len([path for path in readback.rglob("*") if path.is_file()]) == 957
    assert len([call for call in transport.calls if call[0] == "upload"]) == 1
    assert max(map(len, transport.download_batches)) <= 800

    transport.fault = None
    result = verify_uploaded_runtime_source(
        root=source,
        upload_journal_path=journal,
        readback_root=readback,
        receipt_path=receipt,
        transport=transport,
    )

    assert result["fresh_readback_verified"] is True
    assert receipt.is_file()
    assert len([call for call in transport.calls if call[0] == "upload"]) == 1
    assert max(map(len, transport.download_batches)) <= 800
    assert len([path for path in readback.rglob("*") if path.is_file()]) == 4_007


@pytest.mark.parametrize("mutation", ["journal", "revision", "extra", "local-drift", "receipt-overlap"])
def test_source_readback_verifier_rejects_unbound_or_unsafe_resume_state(
    tmp_path: Path, mutation: str,
) -> None:
    from lehome_train.groot.runtime_mixture_publish import (
        publish_source,
        verify_uploaded_runtime_source,
    )

    source = tmp_path / "source"
    _source(source)
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    journal = receipts / "source-upload.json"
    receipt = receipts / "source-readback.json"
    readback = tmp_path / "stable-readback"
    transport = MemoryTransport(fault="download")
    with pytest.raises(RuntimeError, match="download"):
        publish_source(
            root=source, source_type="bc", round_id=None, revision="draft",
            receipt_path=receipt, upload_journal_path=journal,
            readback_root=readback, transport=transport,
        )
    transport.fault = None
    if mutation == "journal":
        payload = json.loads(journal.read_text(encoding="utf-8"))
        payload["readback_pending"] = False
        _write(journal, payload)
    elif mutation == "revision":
        payload = json.loads(journal.read_text(encoding="utf-8"))
        payload["immutable_revision"] = "b" * 40
        _write(journal, payload)
        transport.expected_revision = REVISION
    elif mutation == "extra":
        readback.mkdir(exist_ok=True)
        (readback / "unexpected.bin").write_bytes(b"unexpected")
    elif mutation == "local-drift":
        (source / "manifest.json").write_bytes(b'{"schema":2}')
    else:
        receipt = readback / "receipt.json"

    with pytest.raises((FileExistsError, RuntimeError, ValueError), match="journal|revision|readback|local|receipt|external|unsafe|tree"):
        verify_uploaded_runtime_source(
            root=source, upload_journal_path=journal,
            readback_root=readback, receipt_path=receipt, transport=transport,
        )
    assert not receipt.exists()
    assert not [call for call in transport.calls if call[0] == "upload"][1:]


def test_adopt_uploaded_source_writes_only_a_pending_journal_then_feeds_no_upload_verifier(
    tmp_path: Path,
) -> None:
    from lehome_train.groot.runtime_mixture_publish import (
        adopt_uploaded_runtime_source,
        verify_uploaded_runtime_source,
    )

    source = tmp_path / "source"
    _source(source)
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    journal = receipts / "adopted-upload.json"
    receipt = receipts / "readback.json"
    transport = MemoryTransport()
    for path in source.rglob("*"):
        if path.is_file():
            transport.remote[f"bc/full/{path.relative_to(source).as_posix()}"] = path.read_bytes()

    adopted = adopt_uploaded_runtime_source(
        root=source, source_type="bc", round_id=None, immutable_revision=REVISION,
        upload_journal_path=journal, transport=transport,
    )

    assert adopted["readback_pending"] is True
    assert journal.is_file() and not receipt.exists()
    assert not [call for call in transport.calls if call[0] == "upload"]
    verified = verify_uploaded_runtime_source(
        root=source, upload_journal_path=journal, readback_root=tmp_path / "stable-readback",
        receipt_path=receipt, transport=transport,
    )
    assert verified["fresh_readback_verified"] is True
    assert not [call for call in transport.calls if call[0] == "upload"]


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
    assert ("tree", REPOSITORY, "bc/full") in transport.repository_calls
    assert ("download", REPOSITORY, "bc/full") in transport.repository_calls
    assert ("tree", ROLLOUT_REPOSITORY, "rollouts/round-1") in transport.repository_calls
    assert ("download", ROLLOUT_REPOSITORY, "rollouts/round-1") in transport.repository_calls


@pytest.mark.parametrize("mismatch", [None, "experiment_manifest_sha256", "mixture_weights", "source_quotas"])
def test_hydrator_accepts_bound_schema_v3_and_rejects_any_deployment_binding_drift(
    tmp_path: Path, mismatch: str | None,
) -> None:
    from lehome_train.groot.runtime_mixture import _manifest_digest_binding, sha256_file
    from lehome_train.groot.runtime_mixture_publish import hydrate_runtime_mixture_from_request
    from test_runtime_mixture import _contract, _sha_path

    manifest, index, mounts = _contract(tmp_path / "authoring")
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    binding = {
        "experiment_manifest_sha256": "f" * 64,
        "mixture_weights": {"bc": 80, "rollout": 20, "dagger": 0},
        "source_quotas": {"bc": 51, "rollout": 13, "dagger": 0},
    }
    manifest_value.update({"schema_version": 3, "cycle_size": 64, **binding})
    manifest_value["sources"][0]["quota"] = 51
    manifest_value["sources"][1]["quota"] = 13
    index_value = json.loads(index.read_text(encoding="utf-8"))
    index_value["manifest_sha256"] = _manifest_digest_binding(manifest_value)
    _write(index, index_value)
    manifest_value["window_index"].update({"sha256": sha256_file(index), "byte_size": index.stat().st_size})
    _write(manifest, manifest_value)
    deployment = manifest.parent / "release-receipt.json"
    deployment_value = json.loads(deployment.read_text(encoding="utf-8"))
    deployment_value.update(binding)
    for artifact in deployment_value["artifact_entries"]:
        file = manifest.parent / artifact["relative_path"]
        artifact.update({"sha256": _sha_path(file), "byte_size": file.stat().st_size})
    if mismatch is not None:
        deployment_value[mismatch] = (
            "e" * 64 if mismatch == "experiment_manifest_sha256"
            else {"bc": 70, "rollout": 30, "dagger": 0} if mismatch == "mixture_weights"
            else {"bc": 45, "rollout": 19, "dagger": 0}
        )
    _write(deployment, deployment_value)
    transport = MemoryTransport()
    for artifact in deployment_value["artifact_entries"]:
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
    destination = tmp_path / "hydrated" / "mixture"
    request = tmp_path / "hydrate.json"
    _write(request, {"schema_version": 1, "command": "hydrate-runtime-mixture", "arguments": {
        "deployment_receipt": str(deployment), "source_readback_receipts": receipt_paths,
        "destination": str(destination), "mounts_descriptor": str(destination / "mounts.json"),
    }})

    if mismatch is None:
        result = hydrate_runtime_mixture_from_request(request, transport=transport)
        assert result["experiment_manifest_sha256"] == "f" * 64
        assert result["source_quotas"] == {"bc": 51, "rollout": 13, "dagger": 0}
    else:
        with pytest.raises(ValueError, match="experiment binding"):
            hydrate_runtime_mixture_from_request(request, transport=transport)


def test_hydrator_preserves_and_resumes_partial_stable_targets_after_a_rate_limit(
    tmp_path: Path,
) -> None:
    from lehome_train.groot.runtime_mixture_publish import hydrate_runtime_mixture_from_request
    from test_runtime_mixture import _contract

    manifest, _index, _mounts = _contract(tmp_path / "authoring")
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    deployment = manifest.parent / "release-receipt.json"

    class RateOnceTransport(MemoryTransport):
        def __init__(self) -> None:
            super().__init__()
            self.rate_limited = True

        def download_files(self, **kwargs: object) -> str:
            if self.rate_limited:
                self.rate_limited = False
                destination = kwargs["destination"]
                relative = kwargs["relative_paths"][0]
                prefix = kwargs["remote_prefix"]
                assert isinstance(destination, Path) and isinstance(relative, str) and isinstance(prefix, str)
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(self.remote[f"{prefix}/{relative}"])
                raise HubRateLimitError("rate limited")
            return super().download_files(**kwargs)

    transport = RateOnceTransport()
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
    _write(request, {
        "schema_version": 1, "command": "hydrate-runtime-mixture",
        "arguments": {
            "deployment_receipt": str(deployment), "source_readback_receipts": receipt_paths,
            "destination": str(destination), "mounts_descriptor": str(destination / "mounts.json"),
        },
    })

    with pytest.raises(RuntimeError, match="rate limited after 1 attempts"):
        hydrate_runtime_mixture_from_request(request, transport=transport)
    assert any(path.is_file() for path in destination.rglob("*"))
    assert not (destination / "mounts.json").exists()

    result = hydrate_runtime_mixture_from_request(request, transport=transport)
    assert result["kind"] == "runtime_mixture_hydration"
    assert (destination / "mounts.json").is_file()


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
        "repository": MIXTURE_REPOSITORY, "sources": manifest["sources"],
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


def test_manifest_bound_pending_artifact_binds_80_20_content_and_final_runtime_contract(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture import pending_mixture_id
    from lehome_train.groot.runtime_mixture_publish import finalize_pending_mixture, publish_pending_mixture

    pending, source_mounts = _pending_from_runtime_contract(tmp_path)
    value = json.loads((pending / "publication-pending.json").read_text(encoding="utf-8"))
    value.update({
        "schema_version": 2,
        "experiment_manifest_sha256": "f" * 64,
        "mixture_weights": {"bc": 80, "rollout": 20, "dagger": 0},
        "source_quotas": {"bc": 51, "rollout": 13, "dagger": 0},
    })
    value["sources"][0]["quota"] = 51
    value["sources"][1]["quota"] = 13
    value.pop("mixture_id")
    value.pop("prefix")
    mixture_id = pending_mixture_id(value)
    _write(pending / "publication-pending.json", {**value, "mixture_id": mixture_id, "prefix": f"mixtures/{mixture_id}"})

    transport = MemoryTransport()
    receipt_path = tmp_path / "receipts" / "pending.json"
    receipt_path.parent.mkdir()
    receipt = publish_pending_mixture(pending_root=pending, revision="draft", receipt_path=receipt_path, transport=transport)
    output = finalize_pending_mixture(
        pending_root=pending, publication_receipt=receipt_path, destination=tmp_path / "final",
        deployment_receipt_path=tmp_path / "receipts" / "deployment.json", source_mounts=source_mounts,
        revision="final-draft", transport=transport,
    )

    final = json.loads((output / "mixture.json").read_text(encoding="utf-8"))
    assert receipt["experiment_manifest_sha256"] == "f" * 64
    assert final["schema_version"] == 3
    assert final["mixture_weights"] == {"bc": 80, "rollout": 20, "dagger": 0}
    assert final["source_quotas"] == {"bc": 51, "rollout": 13, "dagger": 0}


@pytest.mark.parametrize("mutation", ["schema-float", "schema-bool", "quota-float", "quota-bool"])
def test_pending_rejects_noninteger_schema_or_quota_before_hub_access(
    tmp_path: Path, mutation: str,
) -> None:
    from lehome_train.groot.runtime_mixture import pending_mixture_id
    from lehome_train.groot.runtime_mixture_publish import publish_pending_mixture

    pending, _source_mounts = _pending_from_runtime_contract(tmp_path)
    value = json.loads((pending / "publication-pending.json").read_text(encoding="utf-8"))
    value.update({
        "schema_version": 2,
        "experiment_manifest_sha256": "f" * 64,
        "mixture_weights": {"bc": 80, "rollout": 20, "dagger": 0},
        "source_quotas": {"bc": 51, "rollout": 13, "dagger": 0},
    })
    value["sources"][0]["quota"] = 51
    value["sources"][1]["quota"] = 13
    if mutation == "schema-float":
        value["schema_version"] = 2.0
    elif mutation == "schema-bool":
        value["schema_version"] = True
    elif mutation == "quota-float":
        value["source_quotas"]["bc"] = 4.0
    else:
        value["source_quotas"]["bc"] = True
    value.pop("mixture_id")
    value.pop("prefix")
    mixture_id = "a" * 64 if mutation in {"schema-float", "schema-bool"} else pending_mixture_id(value)
    _write(pending / "publication-pending.json", {**value, "mixture_id": mixture_id, "prefix": f"mixtures/{mixture_id}"})
    transport = MemoryTransport()

    with pytest.raises(ValueError):
        publish_pending_mixture(
            pending_root=pending, revision="draft", receipt_path=tmp_path / "receipt.json", transport=transport,
        )

    assert transport.calls == []


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


def test_large_source_publish_request_requires_the_explicit_external_staging_contract(
    tmp_path: Path,
) -> None:
    from lehome_train.groot.runtime_mixture_publish import publish_source_from_request

    source = tmp_path / "source"
    _source(source)
    request = tmp_path / "request.json"
    _write(request, {
        "schema_version": 1, "command": "publish-runtime-source", "arguments": {
            "root": str(source), "source_type": "rollout", "round_id": "1", "revision": "main",
            "receipt_path": str(tmp_path / "receipt.json"),
            "upload_journal_path": str(tmp_path / "journal.json"),
            "readback_root": str(tmp_path / "readback"),
            "large_upload": True,
            "large_upload_staging_root": str(tmp_path / "external-state"),
        },
    })

    assert publish_source_from_request(request, transport=MemoryTransport())["immutable_revision"] == REVISION


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
