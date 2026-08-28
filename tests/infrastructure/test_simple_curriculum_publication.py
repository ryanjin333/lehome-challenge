"""Offline contract tests for immutable simple-curriculum publication."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import pytest
from requests import Response
from requests.exceptions import ConnectionError as RequestsConnectionError, HTTPError


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "publish_simple_curriculum_collection.py"
COMMIT = "a" * 40


class HubTransientError(RuntimeError):
    """Named exactly like the canonical transport's retryable error."""


def _module():
    spec = importlib.util.spec_from_file_location("simple_curriculum_publication", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Entry:
    def __init__(self, relative_path: str, entry_type: str = "file") -> None:
        self.relative_path = relative_path
        self.entry_type = entry_type


class FakePublicTransport:
    """Deterministic public Hub seam: no network or credential behavior."""

    def __init__(self) -> None:
        self.store: dict[str, dict[str, bytes]] = {}
        self.upload_calls = 0
        self.authenticated_downloads = 0
        self.anonymous_downloads = 0
        self.transient_uploads = 0
        self.fail_access = False
        self.fail_public_read = False
        self.parent_commits: list[str | None] = []

    def resolve_approved_ref(self, *, repository, ref, token):
        assert repository == "owner/public-dataset" and ref == "main" and token == "token"
        return COMMIT

    def list_tree(self, *, repository, revision, token, remote_prefix=None):
        assert repository == "owner/public-dataset" and revision == COMMIT
        bucket = self.store.get(str(remote_prefix), {})
        return tuple(_Entry(f"{remote_prefix}/{path}") for path in sorted(bucket))

    def upload_files(self, *, repository, revision, source, entries, token, remote_prefix=None, parent_commit=None):
        self.upload_calls += 1
        self.parent_commits.append(parent_commit)
        if self.transient_uploads:
            self.transient_uploads -= 1
            raise HubTransientError("simulated transport reset")
        if self.fail_access:
            raise PermissionError("bad token")
        bucket = self.store.setdefault(str(remote_prefix), {})
        for entry in entries:
            bucket[entry.relative_path] = (Path(source) / entry.relative_path).read_bytes()
        return COMMIT

    def download_files(self, *, repository, revision, destination, relative_paths, token, remote_prefix=None):
        assert repository == "owner/public-dataset" and revision == COMMIT
        if token is None:
            self.anonymous_downloads += 1
            if self.fail_public_read:
                raise PermissionError("repository is private")
        else:
            self.authenticated_downloads += 1
        bucket = self.store[str(remote_prefix)]
        for path in relative_paths:
            target = Path(destination) / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bucket[path])
        return COMMIT


def _bundle(module, tmp_path: Path):
    root = tmp_path / "bundle"
    (root / "manifests").mkdir(parents=True)
    (root / "seals").mkdir()
    (root / "manifests" / "matrix.json").write_bytes(b'{"matrix":true}\n')
    (root / "seals" / "final.json").write_bytes(b'{"seal":true}\n')
    return module.CollectionPublicationBundle(
        root=root,
        run_id="fresh-run-publication-test",
        repository="owner/public-dataset",
        revision="main",
        files=("manifests/matrix.json", "seals/final.json"),
    )


def test_collection_bundle_uses_immutable_layout_and_fresh_authenticated_and_anonymous_readbacks(tmp_path: Path) -> None:
    module = _module()
    transport = FakePublicTransport()

    result = module.publish_collection_bundle(_bundle(module, tmp_path), token="token", transport=transport)

    assert result.remote_prefix == "collection-rounds/fresh-run-publication-test"
    assert result.readback_verified is True
    assert result.public_readback_verified is True
    assert transport.upload_calls == 1
    assert transport.authenticated_downloads == 1
    assert transport.anonymous_downloads == 1
    assert {entry.relative_path for entry in result.entries} == {"manifests/matrix.json", "seals/final.json"}
    assert transport.parent_commits == [COMMIT]


def test_collection_bundle_identical_resume_still_performs_both_readbacks_without_reupload(tmp_path: Path) -> None:
    module = _module()
    transport = FakePublicTransport()
    bundle = _bundle(module, tmp_path)
    first = module.publish_collection_bundle(bundle, token="token", transport=transport)

    second = module.publish_collection_bundle(bundle, token="token", transport=transport)

    assert second.immutable_revision == first.immutable_revision
    assert transport.upload_calls == 1
    assert transport.authenticated_downloads == 2
    assert transport.anonymous_downloads == 2


def test_collection_bundle_rejects_existing_different_bytes_before_upload(tmp_path: Path) -> None:
    module = _module()
    transport = FakePublicTransport()
    bundle = _bundle(module, tmp_path)
    prefix = "collection-rounds/fresh-run-publication-test"
    transport.store[prefix] = {
        "manifests/matrix.json": b"different\n",
        "seals/final.json": (bundle.root / "seals/final.json").read_bytes(),
    }

    with pytest.raises(module.CollectionPublicationError, match="collision"):
        module.publish_collection_bundle(bundle, token="token", transport=transport)

    assert transport.upload_calls == 0


def test_collection_bundle_retries_only_classified_transient_transport_failures(tmp_path: Path) -> None:
    module = _module()
    transport = FakePublicTransport()
    transport.transient_uploads = 2

    result = module.publish_collection_bundle(_bundle(module, tmp_path), token="token", transport=transport, max_attempts=3)

    assert result.readback_verified is True
    assert transport.upload_calls == 3


def test_collection_bundle_retries_real_requests_connection_error(tmp_path: Path) -> None:
    module = _module()

    class FlakyTransport(FakePublicTransport):
        def upload_files(self, **kwargs):
            if self.upload_calls < 2:
                self.upload_calls += 1
                self.parent_commits.append(kwargs.get("parent_commit"))
                raise RequestsConnectionError("network reset")
            return super().upload_files(**kwargs)

    transport = FlakyTransport()
    assert module.publish_collection_bundle(_bundle(module, tmp_path), token="token", transport=transport).readback_verified
    assert transport.upload_calls == 3


def test_collection_bundle_uses_captured_head_as_compare_and_commit_parent(tmp_path: Path) -> None:
    """A concurrent branch writer must make this immutable publication fail closed."""
    module = _module()

    class ConcurrentWriter(FakePublicTransport):
        def upload_files(self, **kwargs):
            assert kwargs["parent_commit"] == COMMIT
            self.store[str(kwargs["remote_prefix"])] = {"manifests/other.json": b"other"}
            raise RuntimeError("parent commit changed")

    transport = ConcurrentWriter()
    with pytest.raises(module.CollectionPublicationError, match="collision"):
        module.publish_collection_bundle(_bundle(module, tmp_path), token="token", transport=transport)
    assert transport.upload_calls == 0


def test_collection_bundle_reconciles_lost_upload_response_only_when_bytes_match(tmp_path: Path) -> None:
    module = _module()

    class LostResponse(FakePublicTransport):
        def upload_files(self, **kwargs):
            self.upload_calls += 1
            self.parent_commits.append(kwargs.get("parent_commit"))
            bucket = self.store.setdefault(str(kwargs["remote_prefix"]), {})
            for entry in kwargs["entries"]:
                bucket[entry.relative_path] = (Path(kwargs["source"]) / entry.relative_path).read_bytes()
            raise RuntimeError("connection outcome unknown")

    transport = LostResponse()
    result = module.publish_collection_bundle(_bundle(module, tmp_path), token="token", transport=transport)
    assert result.readback_verified and result.public_readback_verified
    assert transport.upload_calls == 1


def test_huggingface_transport_treats_only_missing_prefix_entry_as_empty_tree(monkeypatch) -> None:
    """huggingface_hub 0.36 raises on a missing path rather than yielding []."""
    module = _module()

    class EntryNotFoundError(HTTPError):
        pass

    response = Response(); response.status_code = 404
    missing = EntryNotFoundError("Entry Not Found"); missing.response = response

    class Api:
        def list_repo_tree(self, **_kwargs):
            raise missing

    transport = module.HuggingFacePublicDatasetTransport()
    monkeypatch.setattr(transport, "_api", lambda _token: Api())
    assert transport.list_tree(
        repository="owner/public-dataset", revision=COMMIT, token="token",
        remote_prefix="collection-rounds/fresh-run-publication-test",
    ) == ()

    denied = EntryNotFoundError("Entry Not Found"); denied.response = Response(); denied.response.status_code = 401
    monkeypatch.setattr(transport, "_api", lambda _token: type("DeniedApi", (), {
        "list_repo_tree": staticmethod(lambda **_kwargs: (_ for _ in ()).throw(denied)),
    })())
    with pytest.raises(EntryNotFoundError):
        transport.list_tree(
            repository="owner/public-dataset", revision=COMMIT, token="token",
            remote_prefix="collection-rounds/fresh-run-publication-test",
        )


def test_collection_bundle_fails_authentication_without_retry_or_receipt(tmp_path: Path) -> None:
    module = _module()
    transport = FakePublicTransport()
    transport.fail_access = True

    with pytest.raises(module.CollectionPublicationError, match="upload"):
        module.publish_collection_bundle(_bundle(module, tmp_path), token="token", transport=transport)

    assert transport.upload_calls == 1


def test_collection_bundle_rejects_a_non_public_destination_after_authenticated_readback(tmp_path: Path) -> None:
    module = _module()
    transport = FakePublicTransport()
    transport.fail_public_read = True

    with pytest.raises(module.CollectionPublicationError, match="public readback"):
        module.publish_collection_bundle(_bundle(module, tmp_path), token="token", transport=transport)

    assert transport.upload_calls == 1
    assert transport.authenticated_downloads == 1
    assert transport.anonymous_downloads == 1


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _complete_campaign(root: Path, *, count: int = 1000, stopped: bool = True) -> None:
    attempts = [f"attempt-{index:04d}" for index in range(count)]
    _write_json(root / "reports/fresh-source-matrix.json", [{"attempt_id": item} for item in attempts])
    _write_json(root / "reports/fresh-source-report.json", {
        "round_id": "fresh-12k-publication-test", "run_id": "fresh-run-publication-test",
        "trials": [{"attempt_id": item, "accepted_success": index % 2 == 0} for index, item in enumerate(attempts)],
    })
    _write_json(root / "reports/fresh-terminal-artifacts.json", {
        "entries": [{"attempt_id": item} for item in attempts],
    })
    replay_receipts = {}
    for index in range(200):
        attempt = f"replay-{index:04d}"
        artifact = root / "replay/accepted" / attempt / "episode.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("{}", encoding="utf-8")
        receipt_path = root / "replay/hf-sync-receipts" / f"{attempt}.sync.json"
        _write_json(receipt_path, {"attempt_id": attempt, "readback_verified": True})
        replay_receipts[attempt] = {
            "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "episode_sha256": "c" * 64, "immutable_revision": "d" * 40,
        }
    _write_json(root / "replay/success-replay-readback-seal.json", {
        "kind": "lehome_success_replay_readback_seal_v1", "outcome": "complete",
        "accepted_attempt_ids": [f"replay-{index:04d}" for index in range(200)],
        "accepted_by_category": {"pant_long": 50, "pant_short": 50, "top_long": 50, "top_short": 50},
        "readback_receipts": replay_receipts,
        "readback_verified": True,
    })
    observation = {
        "schema_version": 1, "kind": "lehome_simple_curriculum_verified_gpu_stop_v1",
        "provider": "nebius_compute_api", "instance_id": "computeinstance-u00t6xfqhadrcmssa2",
        "state": "STOPPED", "verified": stopped, "observed_at_utc": "2026-08-28T00:00:00Z",
        "provider_response_sha256": "c" * 64,
    }
    observation_path = root / "stage-receipts/gpu-stop-observation.json"
    _write_json(observation_path, observation)
    _write_json(root / "stage-receipts/gpu-stop.json", {"output": {
        "terminal_outcome": "complete", "stop_status": "succeeded",
        "rollout_instance_id": observation["instance_id"], "verified_stopped": stopped,
        "stop_observation_sha256": hashlib.sha256(observation_path.read_bytes()).hexdigest(),
    }})


def test_final_seals_are_distinct_and_complete_requires_exact_fresh_replay_and_verified_stop(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path / "campaign"
    _complete_campaign(root)

    with pytest.raises(module.CollectionPublicationError, match="authoritative"):
        module.build_final_seal(
            root, run_id="fresh-run-publication-test", round_id="fresh-12k-publication-test",
            terminal_outcome="complete", rollout_instance_id="computeinstance-u00t6xfqhadrcmssa2",
        )

    observation_path = root / "stage-receipts/gpu-stop-observation.json"
    observation_sha = hashlib.sha256(observation_path.read_bytes()).hexdigest()
    _write_json(root / "stage-receipts/gpu-stop.json", {"output": {
        "terminal_outcome": "fidelity_stop", "stop_status": "succeeded",
        "rollout_instance_id": "computeinstance-u00t6xfqhadrcmssa2", "verified_stopped": True,
        "stop_observation_sha256": observation_sha,
    }})
    fidelity = module.build_final_seal(
        root, run_id="fresh-run-publication-test", round_id="fresh-12k-publication-test",
        terminal_outcome="fidelity_stop", rollout_instance_id="computeinstance-u00t6xfqhadrcmssa2",
    )
    assert fidelity["kind"] == "lehome_simple_curriculum_fidelity_infrastructure_stop_seal_v1"
    _write_json(root / "stage-receipts/gpu-stop.json", {"output": {
        "terminal_outcome": "replay_shortage", "stop_status": "succeeded",
        "rollout_instance_id": "computeinstance-u00t6xfqhadrcmssa2", "verified_stopped": True,
        "stop_observation_sha256": observation_sha,
    }})
    shortage = module.build_final_seal(
        root, run_id="fresh-run-publication-test", round_id="fresh-12k-publication-test",
        terminal_outcome="replay_shortage", rollout_instance_id="computeinstance-u00t6xfqhadrcmssa2",
    )
    assert shortage["kind"] == "lehome_simple_curriculum_insufficient_fresh_source_seal_v1"

    _complete_campaign(root, count=999)
    with pytest.raises(module.CollectionPublicationError, match="authoritative"):
        module.build_final_seal(
            root, run_id="fresh-run-publication-test", round_id="fresh-12k-publication-test",
            terminal_outcome="complete", rollout_instance_id="computeinstance-u00t6xfqhadrcmssa2",
        )


def test_complete_seal_rejects_id_lists_that_are_not_task6_authenticated_evidence(tmp_path: Path) -> None:
    """A minimal 1,000-row JSON fixture is not production completion proof."""
    module = _module()
    root = tmp_path / "campaign"
    _complete_campaign(root)

    with pytest.raises(module.CollectionPublicationError, match="authoritative"):
        module.build_final_seal(
            root, run_id="fresh-run-publication-test", round_id="fresh-12k-publication-test",
            terminal_outcome="complete", rollout_instance_id="computeinstance-u00t6xfqhadrcmssa2",
        )


def test_final_seal_rejects_a_stop_request_without_a_verified_stopped_observation(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path / "campaign"
    _complete_campaign(root, stopped=False)

    with pytest.raises(module.CollectionPublicationError, match="stop"):
        module.build_final_seal(
            root, run_id="fresh-run-publication-test", round_id="fresh-12k-publication-test",
            terminal_outcome="complete", rollout_instance_id="computeinstance-u00t6xfqhadrcmssa2",
        )


def test_high_level_publication_resume_does_not_add_its_local_receipts_to_the_remote_tree(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path / "campaign"
    _complete_campaign(root)
    observation_path = root / "stage-receipts/gpu-stop-observation.json"
    _write_json(root / "stage-receipts/gpu-stop.json", {"output": {
        "terminal_outcome": "fidelity_stop", "stop_status": "succeeded",
        "rollout_instance_id": "computeinstance-u00t6xfqhadrcmssa2", "verified_stopped": True,
        "stop_observation_sha256": hashlib.sha256(observation_path.read_bytes()).hexdigest(),
    }})
    transport = FakePublicTransport()
    kwargs = {
        "run_id": "fresh-run-publication-test", "round_id": "fresh-12k-publication-test",
        "terminal_outcome": "fidelity_stop", "rollout_instance_id": "computeinstance-u00t6xfqhadrcmssa2",
        "repository": "owner/public-dataset", "revision": "main", "token": "token", "transport": transport,
    }

    first, receipt, readback = module.publish_collection(root, **kwargs)
    second, _, _ = module.publish_collection(root, **kwargs)

    assert receipt.is_file() and readback.is_file()
    assert first.immutable_revision == second.immutable_revision
    assert transport.upload_calls == 1
