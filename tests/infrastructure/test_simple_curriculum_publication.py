"""Offline contract tests for immutable simple-curriculum publication."""

from __future__ import annotations

import contextlib
import importlib.util
import hashlib
import io
import json
from pathlib import Path
import shutil
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
        self.list_revisions: list[str] = []
        self.download_revisions: list[str] = []

    def resolve_approved_ref(self, *, repository, ref, token):
        assert repository == "owner/public-dataset" and ref == "main" and token == "token"
        return COMMIT

    def list_tree(self, *, repository, revision, token, remote_prefix=None):
        assert repository == "owner/public-dataset" and revision == COMMIT
        self.list_revisions.append(revision)
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
        self.download_revisions.append(revision)
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


PROVISIONAL_STAGES = (
    "calibration-matrix", "calibration-head", "first-100-gate",
    "calibration-tail", "calibration-report", "curriculum-matrix",
    "curriculum-a", "curriculum-b", "fresh-report", "replay-matrix",
    "success-replay",
)


def _provisional_campaign(
    module, tmp_path: Path, *, stages: tuple[str, ...], fresh: int = 0,
    replay: int = 0,
) -> tuple[Path, str, str]:
    """Build raw controller-shaped evidence for the real provisional publisher."""
    run_id, round_id = "fresh-run-provisional-matrix", "fresh-12k-provisional-matrix"
    root = tmp_path / run_id
    identity = {"policy": "12k", "simulator_device": "cpu"}
    predecessor = None
    for stage in stages:
        body = {
            "stage": stage, "predecessor_receipt_sha256": predecessor,
            "runtime_identity": identity, "output": {"stage": stage},
        }
        receipt = {**body, "receipt_sha256": hashlib.sha256(module._canonical(body)).hexdigest()}
        _write_json(root / "stage-receipts" / f"{stage}.json", receipt)
        predecessor = receipt["receipt_sha256"]
    entries = []
    for index in range(fresh):
        attempt = f"fresh-{index:04d}"
        artifact = root / "fresh" / "curriculum-a" / "evaluation-terminal" / attempt
        artifact.mkdir(parents=True, exist_ok=True)
        receipt = root / "fresh" / "curriculum-a" / "hf-sync-receipts" / f"{attempt}.sync.json"
        _write_json(receipt, {
            "attempt_id": attempt, "repository": "ryanjin333/lehome-groot-n17-rollouts",
            "remote_prefix": f"rollout-rounds/{round_id}/{attempt}",
            "immutable_revision": COMMIT, "episode_sha256": "e" * 64,
            "round_id": round_id, "run_id": run_id, "readback_verified": True,
        })
        entries.append({"attempt_id": attempt, "finalized_artifact_root": str(artifact)})
    _write_json(root / "reports" / "fresh-terminal-artifacts.json", {"entries": entries})
    replay_receipts = {}
    categories = ("pant_long", "pant_short", "top_long", "top_short")
    for index in range(replay):
        attempt = f"replay-{index:04d}"
        receipt = root / "replay" / "hf-sync-receipts" / f"{attempt}.sync.json"
        _write_json(receipt, {
            "attempt_id": attempt, "repository": "ryanjin333/lehome-groot-n17-rollouts",
            "remote_prefix": f"rollout-rounds/{round_id}-replay/{attempt}",
            "immutable_revision": COMMIT, "episode_sha256": "d" * 64,
            "round_id": f"{round_id}-replay", "run_id": run_id, "readback_verified": True,
        })
        replay_receipts[attempt] = {
            "receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
            "episode_sha256": "d" * 64, "immutable_revision": COMMIT,
            "category": categories[index % len(categories)],
        }
    _write_json(root / "replay" / "success-replay-readback-seal.json", {
        "readback_receipts": replay_receipts,
        "accepted_attempt_ids": list(replay_receipts),
    })
    return root, run_id, round_id


def _publish_provisional(
    module, monkeypatch, root: Path, run_id: str, round_id: str, outcome: str,
    transport: FakePublicTransport, **kwargs,
) -> dict[str, object]:
    # The deployed publisher has an exact mount boundary.  The acceptance
    # fixture supplies the same boundary deterministically without mounting a
    # host path or touching a provider.
    monkeypatch.setattr(module, "_provisional_campaign_root", lambda _run_id: root, raising=False)
    return module.publish_provisional_collection(
        root, run_id=run_id, round_id=round_id, terminal_outcome=outcome,
        rollout_instance_id="computeinstance-u00t6xfqhadrcmssa2",
        repository="owner/public-dataset", revision="main", token="token", transport=transport,
        **kwargs,
    )

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


def test_collection_entry_manifest_hashes_nonempty_descriptor_bytes(tmp_path: Path) -> None:
    module = _module()
    bundle = _bundle(module, tmp_path)

    entries = {entry.relative_path: entry for entry in module._collect_entries(bundle)}

    for relative in bundle.files:
        source = bundle.root / relative
        assert (entries[relative].sha256, entries[relative].byte_size) == (
            hashlib.sha256(source.read_bytes()).hexdigest(), source.stat().st_size,
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


@pytest.mark.parametrize("shape", ("root", "ancestor", "leaf", "escape"))
def test_collection_bundle_rejects_any_symlink_before_transport(shape: str, tmp_path: Path) -> None:
    """Publication must never traverse a symlinked raw bundle path."""
    module = _module()
    real = tmp_path / "real"
    bundle = _bundle(module, real)
    if shape == "root":
        alias = tmp_path / "bundle-alias"
        alias.symlink_to(bundle.root, target_is_directory=True)
        bundle = module.CollectionPublicationBundle(
            root=alias, run_id=bundle.run_id, repository=bundle.repository,
            revision=bundle.revision, files=bundle.files,
        )
    elif shape == "ancestor":
        outer = tmp_path / "outer"; outer.mkdir()
        alias = outer / "nested-alias"
        alias.symlink_to(real, target_is_directory=True)
        bundle = module.CollectionPublicationBundle(
            root=alias / "bundle", run_id=bundle.run_id, repository=bundle.repository,
            revision=bundle.revision, files=bundle.files,
        )
    else:
        target = tmp_path / "outside.json" if shape == "escape" else bundle.root / "manifests" / "target.json"
        target.write_bytes(b'{"outside":true}\n')
        leaf = bundle.root / "manifests" / ("escaped.json" if shape == "escape" else "matrix.json")
        if leaf.exists():
            leaf.unlink()
        leaf.symlink_to(target)
        files = ("manifests/escaped.json", "seals/final.json") if shape == "escape" else bundle.files
        bundle = module.CollectionPublicationBundle(
            root=bundle.root, run_id=bundle.run_id, repository=bundle.repository,
            revision=bundle.revision, files=files,
        )

    class NoTransport(FakePublicTransport):
        def resolve_approved_ref(self, **_kwargs):
            raise AssertionError("unsafe local bundle reached transport")

    with pytest.raises(module.CollectionPublicationError, match="unsafe|symlink"):
        module.publish_collection_bundle(bundle, token="token", transport=NoTransport())


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


def test_crash_window_reconciles_only_missing_readback_at_the_recorded_immutable_revision(tmp_path: Path) -> None:
    """A receipt-only crash never replays paid work or republishes mutable main."""
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
    _, receipt, readback = module.publish_collection(root, **kwargs)
    readback.unlink()  # Crash after receipt fsync, before the local readback receipt.
    transport.upload_calls = 0
    transport.authenticated_downloads = 0
    transport.anonymous_downloads = 0
    transport.download_revisions.clear()
    transport.list_revisions.clear()

    result, recovered = module.reconcile_collection_publication(
        root, run_id=kwargs["run_id"], round_id=kwargs["round_id"],
        terminal_outcome=kwargs["terminal_outcome"], token="token", transport=transport,
    )

    assert recovered == readback and recovered.is_file()
    assert result.immutable_revision == COMMIT
    assert transport.upload_calls == 0
    assert transport.authenticated_downloads == 1 and transport.anonymous_downloads == 1
    assert transport.list_revisions == [COMMIT]
    assert transport.download_revisions == [COMMIT, COMMIT]
    # A later controller restart adopts the durable pair instead of making
    # another network call or running any stage.
    module.reconcile_collection_publication(
        root, run_id=kwargs["run_id"], round_id=kwargs["round_id"],
        terminal_outcome=kwargs["terminal_outcome"], token="token", transport=transport,
    )
    assert transport.upload_calls == 0
    assert transport.download_revisions == [COMMIT, COMMIT]
    assert receipt.is_file()


@pytest.mark.parametrize("fault", ("malformed-receipt", "missing-remote", "public-denied"))
def test_crash_window_reconcile_fails_closed_for_bad_durable_or_remote_evidence(fault: str, tmp_path: Path) -> None:
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
    _, receipt, readback = module.publish_collection(root, **kwargs)
    readback.unlink()
    if fault == "malformed-receipt":
        receipt.write_text("{}\n", encoding="utf-8")
    elif fault == "missing-remote":
        remote = transport.store["collection-rounds/fresh-run-publication-test"]
        remote.pop(next(path for path in remote if path.startswith("seals/")))
    else:
        transport.fail_public_read = True
    transport.upload_calls = 0

    with pytest.raises(module.CollectionPublicationError):
        module.reconcile_collection_publication(
            root, run_id=kwargs["run_id"], round_id=kwargs["round_id"],
            terminal_outcome=kwargs["terminal_outcome"], token="token", transport=transport,
        )
    assert transport.upload_calls == 0
    assert not readback.exists()


def test_failure_staging_uses_a_fixed_evidence_allowlist_and_never_uploads_debug_files(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path / "campaign"
    _complete_campaign(root)
    observation = root / "stage-receipts/gpu-stop-observation.json"
    _write_json(root / "stage-receipts/gpu-stop.json", {"output": {
        "terminal_outcome": "fidelity_stop", "stop_status": "succeeded",
        "rollout_instance_id": "computeinstance-u00t6xfqhadrcmssa2", "verified_stopped": True,
        "stop_observation_sha256": hashlib.sha256(observation.read_bytes()).hexdigest(),
    }})
    seal = module.build_final_seal(
        root, run_id="fresh-run-publication-test", round_id="fresh-12k-publication-test",
        terminal_outcome="fidelity_stop", rollout_instance_id="computeinstance-u00t6xfqhadrcmssa2",
    )
    seal_path = root / "seals" / "fidelity.json"
    module._write_immutable_json(seal_path, seal)
    # Neither ordinary forgotten diagnostics nor credential-bearing debug
    # content belongs in a terminal failure bundle.
    (root / "reports/debug.json").write_text('{"why":"investigate"}', encoding="utf-8")
    (root / "reports/debug-token.json").write_text('{"token":"super-secret-value-123"}', encoding="utf-8")

    bundle = module._stage_collection_bundle(
        root, seal_path=seal_path, run_id="fresh-run-publication-test",
        repository="owner/public-dataset", revision="main",
    )
    try:
        assert "reports/debug.json" not in bundle.files
        assert "reports/debug-token.json" not in bundle.files
        assert "seals/fidelity.json" in bundle.files
        assert all("debug" not in path for path in bundle.files)
    finally:
        __import__("shutil").rmtree(bundle.root, ignore_errors=True)


def test_known_failure_evidence_with_a_secret_is_rejected_before_public_staging(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path / "campaign"
    _complete_campaign(root)
    observation = root / "stage-receipts/gpu-stop-observation.json"
    _write_json(root / "stage-receipts/gpu-stop.json", {"output": {
        "terminal_outcome": "fidelity_stop", "stop_status": "succeeded",
        "rollout_instance_id": "computeinstance-u00t6xfqhadrcmssa2", "verified_stopped": True,
        "stop_observation_sha256": hashlib.sha256(observation.read_bytes()).hexdigest(),
    }})
    _write_json(root / "reports/first-100-gate.json", {"token": "super-secret-value-123"})
    seal = module.build_final_seal(
        root, run_id="fresh-run-publication-test", round_id="fresh-12k-publication-test",
        terminal_outcome="fidelity_stop", rollout_instance_id="computeinstance-u00t6xfqhadrcmssa2",
    )
    seal_path = root / "seals" / "fidelity.json"
    module._write_immutable_json(seal_path, seal)

    with pytest.raises(module.CollectionPublicationError, match="credential-like content"):
        module._stage_collection_bundle(
            root, seal_path=seal_path, run_id="fresh-run-publication-test",
            repository="owner/public-dataset", revision="main",
        )


def test_complete_publication_stages_real_task6_recorder_finalizer_and_hub_sync_evidence(tmp_path: Path) -> None:
    """Exercise the accepted production evidence path, not a hand-written seal.

    This reuses the real persistent-worker/AutonomousRecorder/finalizer test
    seam for all 1,000 fresh terminal outcomes.  It then uses the real
    HubSyncDaemon receipts for both fresh and visual-only replay before the
    Task-7 publisher re-authenticates and public-readback publishes the tree.
    """
    module = _module()
    producer_spec = importlib.util.spec_from_file_location(
        "task7_real_producer_fixture", ROOT / "tests/infrastructure/test_groot_persistent_summary.py",
    )
    assert producer_spec and producer_spec.loader
    producer = importlib.util.module_from_spec(producer_spec)
    sys.modules[producer_spec.name] = producer
    producer_spec.loader.exec_module(producer)

    from lehome.flywheel.hub_sync import HubSyncDaemon
    from lehome.flywheel.simple_curriculum import build_calibration_rows
    from lehome.flywheel.task_ledger import TaskLedger

    controller = module._task6_controller_module()
    root = tmp_path / "campaign"
    run_id, round_id = "fresh-run-publication-real", "fresh-12k-publication-real"
    policy = {
        "policy_repo": "ryanjin333/lehome-groot-n17-models",
        "policy_revision": "30ac1a84da67b099e115ad147bcd61e9d60046d3",
        "policy_step": 12_000,
        "policy_artifact_sha256": "3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06",
    }
    runtime = {
        **policy,
        "rollout_image": "repo/rollout@sha256:" + "a" * 64,
        "trainer_image": "repo/trainer@sha256:" + "b" * 64,
        "simulator_device": "cpu", "cloth_device": "cpu", "policy_device": "cuda:0", "worker_count": 4,
    }
    config = controller.CollectionConfig(
        campaign_root=root, host_code_root=ROOT, run_id=run_id, round_id=round_id,
        max_wall_seconds=3600.0, max_spend_usd=99.0, paid=False, gpu_stop_command=None,
        runtime_identity=runtime,
    )
    catalog = producer._catalog()
    calibration = build_calibration_rows(catalog, seed_base=91_000)
    by_category = {category: [row for row in calibration if row["category"] == category] for category in catalog}
    # Arrange the first five in each physical partition to give the real
    # recorder/finalizer seam accepted source evidence for every category.
    calibration_rows = by_category["top_long"] + by_category["top_short"] + by_category["pant_long"] + by_category["pant_short"]
    curriculum_rows = [
        {
            **row,
            "attempt_id": f"curriculum-{index:04d}", "trial_id": f"curriculum-{index:04d}",
            "seed": int(row["seed"]) + 1_000_000, "source_seed": int(row["seed"]) + 1_000_000,
            "logical_stage": "curriculum",
        }
        for index, row in enumerate((
            by_category["pant_long"] * 3 + by_category["pant_short"] * 3
        )[:600])
    ]
    # Keep every fresh matrix row unique even though the curriculum's category
    # distribution is intentionally ordered for this producer-path fixture.
    assert len(calibration_rows) == 400 and len(curriculum_rows) == 600
    for parent_name, rows, partitions in (
        ("calibration", calibration_rows, (("calibration-head", 0, 100, 100, 150), ("calibration-tail", 100, 400, 300, 400))),
        ("curriculum", curriculum_rows, (("curriculum-a", 0, 300, 300, 400), ("curriculum-b", 300, 600, 300, 400))),
    ):
        logical = root / "matrices" / f"{parent_name}.json"
        logical.parent.mkdir(parents=True, exist_ok=True)
        logical.write_bytes(producer._canonical(rows))
        for partition, start, end, target, lease_budget in partitions:
            physical, _manifest, _details = controller.materialize_partition(
                parent_matrix=logical, parent_matrix_sha256=hashlib.sha256(logical.read_bytes()).hexdigest(),
                output_directory=root / "partitions", partition_id=partition, start=start, end=end,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                partition_root, matrix, partition_rows = producer._real_persistent_campaign(
                    tmp_path, rows=json.loads(physical.read_text(encoding="utf-8")),
                    campaign_root=root / "fresh" / partition, matrix_path=physical, policy=policy,
                    campaign_round_id=round_id, campaign_run_id=run_id, max_attempts=lease_budget,
                )
            # This controller state is intentionally retained locally but is
            # explicitly excluded from the public complete bundle.
            (partition_root / "rollout-preemption.json").write_text("{}\n", encoding="utf-8")
            report = producer._module().build_simple_partition_report(
                campaign_root=partition_root, matrix_path=matrix,
                matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(), **policy,
            )
            report_path = root / "reports" / "partitions" / f"{partition}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_bytes(producer._canonical(report))
            sync = HubSyncDaemon(
                repository="ryanjin333/lehome-groot-n17-rollouts", round_id=round_id, run_id=run_id,
                token="fixture-token", transport=producer._ReceiptTransport(),
                accepted_root=partition_root / "evaluation-terminal",
                receipts_root=partition_root / "hf-sync-receipts", readback_root=partition_root / "hf-readback",
                revision="main",
            )
            for trial in report["trials"]:
                attempt = str(trial["attempt_id"])
                sync.sync_episode(attempt, partition_root / "evaluation-terminal" / attempt)

    controller._build_fresh_source_report(config)
    fresh = controller.CommandRunner(config)._discover("fresh-report", {})
    assert set(fresh["artifacts"]) == {"report", "matrix", "terminal_artifact_manifest"}

    # The replay ledger has the exact 4x100/4x50 contract.  Replayed artifact
    # bytes come from recorder/finalizer-authenticated fresh terminals and are
    # then independently published by the real HubSync receipt producer.
    source_by_category: dict[str, Path] = {}
    for entry in json.loads((root / "reports/fresh-terminal-artifacts.json").read_text(encoding="utf-8"))["entries"]:
        artifact = Path(str(entry["finalized_artifact_root"]))
        episode = json.loads((artifact / "raw" / str(entry["attempt_id"]) / "episode.json").read_text(encoding="utf-8"))
        category = str(episode["identity"]["category"])
        if entry["terminal_event"] == "accepted" and category not in source_by_category:
            source_by_category[category] = artifact
    assert set(source_by_category) == {"top_long", "top_short", "pant_long", "pant_short"}
    replay_rows = [
        {
            "attempt_id": f"replay-{category}-{index:03d}", "trial_id": f"replay-{category}-{index:03d}",
            "category": category, "strategy": "visual_only", "category_acceptance_cap": 50,
        }
        for category in ("top_long", "top_short", "pant_long", "pant_short") for index in range(100)
    ]
    replay_path = root / "replay" / "replay.json"; replay_path.parent.mkdir(parents=True)
    replay_path.write_bytes(producer._canonical(replay_rows))
    (root / "replay/replay.json.sha256").write_text(hashlib.sha256(replay_path.read_bytes()).hexdigest() + "\n", encoding="ascii")
    replay_ledger = TaskLedger(root / "replay" / "ledger.sqlite3", attempt_matrix=replay_rows, max_attempts=400, target_accepted=200)
    (root / "replay" / "accepted").mkdir()
    replay_sync = HubSyncDaemon(
        repository="ryanjin333/lehome-groot-n17-rollouts", round_id=round_id + "-replay", run_id=run_id,
        token="fixture-token", transport=producer._ReceiptTransport(), accepted_root=root / "replay" / "accepted",
        receipts_root=root / "replay" / "hf-sync-receipts", readback_root=root / "replay" / "hf-readback", revision="main",
    )
    try:
        for index in range(400):
            lease = replay_ledger.lease_next("replay-worker", lease_duration_ns=10**15)
            assert lease is not None
            category = str(lease.attempt.assignment["category"])
            # Reject the first half in every category, then accept the second
            # half.  The real ledger stops dispatch at its 200-accepted cap,
            # so this order proves all 400 terminal rows settle first.
            if int(str(lease.attempt.assignment["attempt_id"]).rsplit("-", 1)[1]) >= 50:
                artifact = root / "replay" / "accepted" / lease.attempt.attempt_id
                shutil.copytree(source_by_category[category], artifact)
                replay_sync.sync_episode(lease.attempt.attempt_id, artifact)
                replay_ledger.record_terminal("replay-worker", lease.attempt.attempt_id, lease.lease_id, str(artifact))
                assert replay_ledger.validate_terminal(lease.attempt.attempt_id, "accepted", artifact_id=str(artifact)) == "accepted"
            else:
                replay_ledger.record_terminal(
                    "replay-worker", lease.attempt.attempt_id, lease.lease_id,
                    f"rejected-{lease.attempt.attempt_id}",
                )
                assert replay_ledger.validate_terminal(lease.attempt.attempt_id, "rejected") == "rejected"
    finally:
        replay_ledger.close()
    assert controller._discover_success_replay(config, matrix=replay_path, ledger=root / "replay" / "ledger.sqlite3")["result"] == "complete"

    observation = {
        "schema_version": 1, "kind": "lehome_simple_curriculum_verified_gpu_stop_v1",
        "provider": "nebius_compute_api", "instance_id": "computeinstance-u00t6xfqhadrcmssa2",
        "state": "STOPPED", "verified": True, "observed_at_utc": "2026-08-28T00:00:00Z",
        "provider_response_sha256": "c" * 64,
    }
    observation_path = root / "stage-receipts/gpu-stop-observation.json"
    _write_json(observation_path, observation)
    _write_json(root / "stage-receipts/gpu-stop.json", {"output": {
        "terminal_outcome": "complete", "stop_status": "succeeded",
        "rollout_instance_id": observation["instance_id"], "verified_stopped": True,
        "stop_observation_sha256": hashlib.sha256(observation_path.read_bytes()).hexdigest(),
    }})
    # A root runtime cache is real-but-private controller state; its presence
    # cannot be a completion requirement or accidentally public evidence.
    (root / "hf-cache").mkdir()
    (root / "hf-cache" / "runtime.json").write_text('{"cache":"private"}\n', encoding="utf-8")
    # Conversely an unfamiliar byte in a public evidence area must fail
    # closed.  That proves the complete bundle is an allowlist, not a walk.
    unreviewed = root / "reports" / "unreviewed-debug.json"
    unreviewed.write_text('{"why":"not reviewed"}\n', encoding="utf-8")
    with pytest.raises(module.CollectionPublicationError, match="unreviewed"):
        module.publish_collection(
            root, run_id=run_id, round_id=round_id, terminal_outcome="complete",
            rollout_instance_id=observation["instance_id"], repository="owner/public-dataset",
            revision="main", token="token", transport=FakePublicTransport(),
        )
    unreviewed.unlink()
    published, _receipt, _readback = module.publish_collection(
        root, run_id=run_id, round_id=round_id, terminal_outcome="complete",
        rollout_instance_id=observation["instance_id"], repository="owner/public-dataset",
        revision="main", token="token", transport=FakePublicTransport(),
    )
    assert published.public_readback_verified is True


def test_provisional_complete_publishes_the_exact_authoritative_pre_stop_matrix(monkeypatch, tmp_path: Path) -> None:
    module = _module()
    root, run_id, round_id = _provisional_campaign(
        module, tmp_path, stages=PROVISIONAL_STAGES, fresh=1000, replay=200,
    )
    transport = FakePublicTransport()
    monkeypatch.setattr(
        module, "_authenticate_complete_task6_evidence",
        lambda *_args, **_kwargs: (1000, 617, 400, 200, {
            "pant_long": 50, "pant_short": 50, "top_long": 50, "top_short": 50,
        }),
    )

    receipt = _publish_provisional(module, monkeypatch, root, run_id, round_id, "complete", transport)

    prefix = f"collection-rounds/{run_id}/manifests/provisional"
    assert receipt["remote_prefix"] == prefix
    assert receipt["immutable_revision"] == COMMIT
    assert receipt["entry_count"] == len(receipt["entries"])
    assert receipt["readback_verified"] is True and receipt["public_readback_verified"] is True
    payloads = transport.store[prefix]
    evidence = json.loads(payloads["manifests/provisional/evidence-manifest.json"])
    task6 = json.loads(payloads["manifests/provisional/task6-validation.json"])
    references = json.loads(payloads["manifests/provisional/hub-artifact-references.json"])
    assert evidence["reachable_stages"] == list(PROVISIONAL_STAGES)
    assert evidence["completion_claim"] == "none" and evidence["gpu_stop_verified"] is False
    assert evidence["terminal_chain_head"] == evidence["stage_receipts"][-1]["receipt_sha256"]
    assert task6 == {
        "schema_version": 1, "kind": "lehome_simple_curriculum_task6_validation_v1",
        "terminal_outcome": "complete", "result": "complete", "fresh_valid_outcomes": 1000,
        "fresh_official_successes": 617, "replay_attempts": 400,
        "replay_accepted_successes": 200,
        "replay_accepted_by_category": {"pant_long": 50, "pant_short": 50, "top_long": 50, "top_short": 50},
    }
    assert len(references["fresh"]) == 1000 and len(references["success_replay"]) == 200
    assert transport.authenticated_downloads == 1 and transport.anonymous_downloads == 1


def test_provisional_shortages_preserve_only_the_reachable_authoritative_evidence(monkeypatch, tmp_path: Path) -> None:
    module = _module()
    fresh_root, fresh_run, fresh_round = _provisional_campaign(
        module, tmp_path / "fresh", stages=PROVISIONAL_STAGES[:6], fresh=37, replay=0,
    )
    fresh_transport = FakePublicTransport()
    fresh = _publish_provisional(
        module, monkeypatch, fresh_root, fresh_run, fresh_round, "insufficient_source_stop", fresh_transport,
    )
    fresh_prefix = str(fresh["remote_prefix"])
    fresh_evidence = json.loads(fresh_transport.store[fresh_prefix]["manifests/provisional/evidence-manifest.json"])
    fresh_task6 = json.loads(fresh_transport.store[fresh_prefix]["manifests/provisional/task6-validation.json"])
    fresh_refs = json.loads(fresh_transport.store[fresh_prefix]["manifests/provisional/hub-artifact-references.json"])
    assert fresh_evidence["reachable_stages"] == list(PROVISIONAL_STAGES[:6])
    assert fresh_task6["result"] == "not_complete"
    assert fresh_task6["fresh_reference_count"] == 37 and fresh_task6["replay_accepted_reference_count"] == 0
    assert len(fresh_refs["fresh"]) == 37 and fresh_refs["success_replay"] == []

    replay_root, replay_run, replay_round = _provisional_campaign(
        module, tmp_path / "replay", stages=PROVISIONAL_STAGES, fresh=1000, replay=73,
    )
    replay_transport = FakePublicTransport()
    replay = _publish_provisional(
        module, monkeypatch, replay_root, replay_run, replay_round, "replay_shortage", replay_transport,
    )
    replay_payloads = replay_transport.store[str(replay["remote_prefix"])]
    replay_evidence = json.loads(replay_payloads["manifests/provisional/evidence-manifest.json"])
    replay_task6 = json.loads(replay_payloads["manifests/provisional/task6-validation.json"])
    replay_refs = json.loads(replay_payloads["manifests/provisional/hub-artifact-references.json"])
    assert replay_evidence["reachable_stages"] == list(PROVISIONAL_STAGES)
    assert replay_task6["fresh_reference_count"] == 1000
    assert replay_task6["replay_accepted_reference_count"] == 73
    assert len(replay_refs["success_replay"]) == 73

    fidelity_root, fidelity_run, fidelity_round = _provisional_campaign(
        module, tmp_path / "fidelity", stages=PROVISIONAL_STAGES[:3], fresh=4, replay=0,
    )
    fidelity_transport = FakePublicTransport()
    fidelity = _publish_provisional(
        module, monkeypatch, fidelity_root, fidelity_run, fidelity_round, "fidelity_stop", fidelity_transport,
    )
    fidelity_payloads = fidelity_transport.store[str(fidelity["remote_prefix"])]
    fidelity_evidence = json.loads(fidelity_payloads["manifests/provisional/evidence-manifest.json"])
    fidelity_refs = json.loads(fidelity_payloads["manifests/provisional/hub-artifact-references.json"])
    assert fidelity_evidence["reachable_stages"] == list(PROVISIONAL_STAGES[:3])
    assert len(fidelity_refs["fresh"]) == 4 and fidelity_refs["success_replay"] == []


@pytest.mark.parametrize("length", (0, 1, 3, 7, 11))
def test_provisional_infrastructure_stops_preserve_each_real_stage_prefix(monkeypatch, tmp_path: Path, length: int) -> None:
    module = _module()
    root, run_id, round_id = _provisional_campaign(
        module, tmp_path, stages=PROVISIONAL_STAGES[:length], fresh=2, replay=1,
    )
    transport = FakePublicTransport()

    receipt = _publish_provisional(module, monkeypatch, root, run_id, round_id, "infrastructure_stop", transport)

    payloads = transport.store[str(receipt["remote_prefix"])]
    evidence = json.loads(payloads["manifests/provisional/evidence-manifest.json"])
    task6 = json.loads(payloads["manifests/provisional/task6-validation.json"])
    references = json.loads(payloads["manifests/provisional/hub-artifact-references.json"])
    assert evidence["reachable_stages"] == list(PROVISIONAL_STAGES[:length])
    assert task6["fresh_reference_count"] == 2 and task6["replay_accepted_reference_count"] == 1
    assert len(references["fresh"]) == 2 and len(references["success_replay"]) == 1


@pytest.mark.parametrize("tamper", ("body", "predecessor", "task6-count", "fresh-reference", "replay-reference"))
def test_provisional_fails_closed_for_tampered_authoritative_evidence(monkeypatch, tmp_path: Path, tamper: str) -> None:
    module = _module()
    root, run_id, round_id = _provisional_campaign(
        module, tmp_path, stages=PROVISIONAL_STAGES, fresh=1000, replay=200,
    )
    monkeypatch.setattr(module, "_authenticate_complete_task6_evidence", lambda *_args, **_kwargs: (
        999 if tamper == "task6-count" else 1000, 1, 400, 200,
        {"pant_long": 50, "pant_short": 50, "top_long": 50, "top_short": 50},
    ))
    if tamper == "body":
        path = root / "stage-receipts/calibration-head.json"
        payload = json.loads(path.read_text(encoding="utf-8")); payload["output"] = {"altered": True}; _write_json(path, payload)
    elif tamper == "predecessor":
        path = root / "stage-receipts/calibration-head.json"
        payload = json.loads(path.read_text(encoding="utf-8")); payload["predecessor_receipt_sha256"] = "b" * 64; _write_json(path, payload)
    elif tamper == "fresh-reference":
        path = root / "fresh/curriculum-a/hf-sync-receipts/fresh-0000.sync.json"
        payload = json.loads(path.read_text(encoding="utf-8")); payload["immutable_revision"] = "main"; _write_json(path, payload)
    elif tamper == "replay-reference":
        path = root / "replay/success-replay-readback-seal.json"
        payload = json.loads(path.read_text(encoding="utf-8")); payload["readback_receipts"]["replay-0000"]["receipt_sha256"] = "bad"; _write_json(path, payload)

    with pytest.raises(module.CollectionPublicationError):
        _publish_provisional(module, monkeypatch, root, run_id, round_id, "complete", FakePublicTransport())


@pytest.mark.parametrize("fault", ("extra", "missing", "changed-readback", "returned-revision", "lost-response"))
def test_provisional_fails_closed_for_non_identical_or_unpinned_remote_state(monkeypatch, tmp_path: Path, fault: str) -> None:
    module = _module()
    root, run_id, round_id = _provisional_campaign(module, tmp_path, stages=(), fresh=0, replay=0)

    class FaultyTransport(FakePublicTransport):
        def upload_files(self, **kwargs):
            response = super().upload_files(**kwargs)
            bucket = self.store[str(kwargs["remote_prefix"])]
            if fault == "extra": bucket["manifests/provisional/extra.json"] = b"{}\n"
            if fault == "missing": bucket.pop(next(iter(bucket)))
            if fault == "changed-readback": bucket[next(iter(bucket))] = b"tampered\n"
            if fault == "lost-response": raise RuntimeError("connection outcome unknown")
            return "main" if fault == "returned-revision" else response

    transport = FaultyTransport()
    if fault == "lost-response":
        # The publisher may reconcile an ambiguous upload only when the
        # independently pinned remote tree is exact.
        assert _publish_provisional(module, monkeypatch, root, run_id, round_id, "infrastructure_stop", transport)["immutable_revision"] == COMMIT
    else:
        with pytest.raises(module.CollectionPublicationError):
            _publish_provisional(module, monkeypatch, root, run_id, round_id, "infrastructure_stop", transport)


@pytest.mark.parametrize("phase", ("upload", "authenticated-readback", "anonymous-readback", "retry"))
def test_provisional_deadline_bounds_upload_readback_and_retries(monkeypatch, tmp_path: Path, phase: str) -> None:
    module = _module()
    root, run_id, round_id = _provisional_campaign(module, tmp_path, stages=(), fresh=0, replay=0)
    now = [0.0]

    class ExpiringTransport(FakePublicTransport):
        def upload_files(self, **kwargs):
            if phase == "retry":
                now[0] = 1.1
                raise HubTransientError("temporary")
            result = super().upload_files(**kwargs)
            if phase == "upload": now[0] = 1.1
            return result

        def download_files(self, **kwargs):
            result = super().download_files(**kwargs)
            if phase == "authenticated-readback" and kwargs["token"] is not None: now[0] = 1.1
            if phase == "anonymous-readback" and kwargs["token"] is None: now[0] = 1.1
            return result

    with pytest.raises(module.CollectionPublicationError, match="timeout"):
        _publish_provisional(
            module, monkeypatch, root, run_id, round_id, "infrastructure_stop", ExpiringTransport(),
            timeout_seconds=1.0, monotonic=lambda: now[0],
        )
