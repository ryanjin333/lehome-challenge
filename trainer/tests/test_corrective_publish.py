from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

from lehome.flywheel.artifacts import build_sha256_manifest
from lehome_train.flywheel.corrective import (
    APPROVED_PARENT_ARTIFACT_SHA256,
    APPROVED_PARENT_REPOSITORY,
    APPROVED_PARENT_REVISION,
    build_corrective_publication_bundle,
    build_corrective_selection_bundle,
    select_corrective_successes,
)
from lehome_train.flywheel.publish import (
    CorrectiveCanaryAbortPublicationBundle,
    CorrectiveCanaryPublicationBundle,
    build_corrective_canary_abort_publication_bundle,
    build_corrective_canary_publication_bundle,
    publish_private_corrective_canary,
    publish_private_corrective_canary_abort,
    publish_verified_corrective_rft,
    verify_uploaded_corrective_rft,
)
from lehome_train.hub import HubAccess, HubTreeEntry
from lehome_train.io import canonical_json_bytes, canonical_json_sha256, sha256_file
from lehome_train.models import SyncEntry
from lehome_train.flywheel import publish as publish_module
from lehome_train.flywheel import publish_canary as canary_cli


REPOSITORY = Path(__file__).resolve().parents[2]
CAMPAIGN_SPEC = importlib.util.spec_from_file_location(
    "corrective_campaign_publish_test", REPOSITORY / "scripts" / "run_groot_corrective_campaign.py"
)
assert CAMPAIGN_SPEC is not None and CAMPAIGN_SPEC.loader is not None
corrective_campaign = importlib.util.module_from_spec(CAMPAIGN_SPEC)
sys.modules[CAMPAIGN_SPEC.name] = corrective_campaign
CAMPAIGN_SPEC.loader.exec_module(corrective_campaign)


TOKEN = "hf_corrective_process_token"


class FakeTransport:
    def __init__(self, *, fail_at: str | None = None) -> None:
        self.fail_at = fail_at
        self.remote: dict[str, bytes] = {}
        self.upload_prefix: str | None = None
        self.uploaded_paths: tuple[str, ...] = ()

    def check_access(self, *, repository: str, token: str) -> HubAccess:
        assert repository == "ryanjin333/lehome-groot-n17-data"
        assert token == TOKEN
        return HubAccess(can_read=True, can_write=True, private_repository=True)

    def upload_files(self, *, repository: str, revision: str, source: Path, entries, token: str, remote_prefix: str | None = None) -> str:
        if self.fail_at == "upload":
            raise OSError("upload unavailable")
        assert remote_prefix is not None
        self.upload_prefix = remote_prefix
        self.uploaded_paths = tuple(entry.relative_path for entry in entries)
        self.remote = {
            f"{remote_prefix}/{entry.relative_path}": (source / entry.relative_path).read_bytes()
            for entry in entries
        }
        return "a" * 40

    def list_tree(self, *, repository: str, revision: str, token: str):
        if self.fail_at == "list":
            raise OSError("list unavailable")
        return tuple(HubTreeEntry(path, "file") for path in sorted(self.remote))

    def download_files(self, *, repository: str, revision: str, destination: Path, relative_paths, token: str, remote_prefix: str | None = None) -> str:
        if self.fail_at == "download":
            raise OSError("download unavailable")
        assert remote_prefix == self.upload_prefix
        for relative_path in relative_paths:
            target = destination / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.remote[f"{remote_prefix}/{relative_path}"])
        return revision


def test_corrective_tree_match_accepts_expected_remote_directories() -> None:
    entries = (
        SyncEntry("root.json", "a" * 64, 2),
        SyncEntry("nested/payload.json", "b" * 64, 2),
    )
    tree = (
        HubTreeEntry("release", "directory"),
        HubTreeEntry("release/root.json", "file"),
        HubTreeEntry("release/nested", "directory"),
        HubTreeEntry("release/nested/payload.json", "file"),
    )
    assert publish_module._tree_matches(tree, "release", entries)


def _attempt(index: int, category: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "attempt_id": f"corrective-wave-{index // 4:06d}-worker-{index % 4}",
        "wave_index": index // 4,
        "worker_slot": index % 4,
        "episode_id": f"corrective-wave-{index // 4:06d}-worker-{index % 4}",
        "category": category,
        "release_stage": "seen",
        "outcome": "success",
        "accepted_success": True,
        "reset_sha256": f"{index:064x}",
        "randomization_sha256": f"{index + 200:064x}",
        "hard_state_sha256": f"{index + 400:064x}",
        "parent_checkpoint_repository": APPROVED_PARENT_REPOSITORY,
        "parent_checkpoint_revision": APPROVED_PARENT_REVISION,
        "parent_checkpoint_artifact_sha256": APPROVED_PARENT_ARTIFACT_SHA256,
        "parent_checkpoint_step": 12000,
        "code_revision": "f" * 40,
        "asset_revision": "0" * 40,
        "image_identity": "sha256:" + "1" * 64,
        "simulator_version": "5.1.0.0",
        "provider": {
            "rental_kind": "on-demand", "instance_hourly_cost_usd": 0.8,
            "account_hourly_total_usd": 0.8002, "offer_id": 40705900,
            "gpu_name": "RTX 3090", "num_gpus": 4,
        },
    }


def _publication_input(tmp_path: Path):
    categories = [
        *("top_long",) * 30, *("top_short",) * 45,
        *("pant_long",) * 30, *("pant_short",) * 45,
        *("top_short",) * 2,
    ]
    attempts = [_attempt(index, category) for index, category in enumerate(categories)]
    for attempt in attempts:
        wave_index = int(attempt["wave_index"])
        attempt["provider"] = {
            **attempt["provider"],
            "offer_id": 40705900 + wave_index,
        }
    artifacts: dict[str, dict[str, object]] = {}
    for attempt in attempts:
        root = tmp_path / "raw" / str(attempt["episode_id"])
        root.mkdir(parents=True)
        (root / "episode.json").write_bytes(canonical_json_bytes({
            "episode_id": attempt["episode_id"],
            "accepted_success": True,
            "outcome": "success",
            "terminal_reason": "success",
            "mode": "autonomous",
            "identity": {"release_stage": "seen"},
        }))
        (root / "annotations.jsonl").write_bytes(b"{}\n")
        (root / "SHA256SUMS.json").write_bytes(
            canonical_json_bytes(build_sha256_manifest(root))
        )
        policy = tmp_path / "policy" / f"{attempt['attempt_id']}.json"
        policy.parent.mkdir(exist_ok=True)
        policy.write_bytes(canonical_json_bytes({"attempt_id": attempt["attempt_id"], "policy": "receipt"}))
        artifacts[str(attempt["attempt_id"])] = {
            "episode_id": attempt["episode_id"], "release_stage": "seen", "root": str(root),
            "episode_manifest_sha256": sha256_file(root / "SHA256SUMS.json"),
            "attempt_receipt": attempt,
            "policy_receipt_path": str(policy), "policy_receipt_sha256": sha256_file(policy),
        }
    selection = build_corrective_selection_bundle(attempts, artifacts)
    wave_evidence: dict[int, dict[str, object]] = {}
    for wave_index in range(len(attempts) // 4):
        wave = tmp_path / "waves" / f"wave-{wave_index:06d}.json"
        wave.parent.mkdir(exist_ok=True)
        source_snapshot = tmp_path / "provider-snapshots" / f"wave-{wave_index:06d}.json"
        source_snapshot.parent.mkdir(exist_ok=True)
        source_snapshot.write_bytes(canonical_json_bytes({"wave_index": wave_index, "source": "provider"}))
        wave_attempts = attempts[wave_index * 4 : wave_index * 4 + 4]
        provider_evidence = {
            "schema_version": 1, "kind": "external_provider_offer_evidence", "evidence_id": f"evidence-{wave_index}",
            "queried_at_unix": 1, "expires_at_unix": 2, "source_snapshot_path": source_snapshot.name,
            "source_snapshot_sha256": sha256_file(source_snapshot), "source_response_sha256": "a" * 64,
            "rental_kind": "on-demand", "instance_hourly_cost_usd": 0.8, "account_hourly_total_usd": 0.8002,
            "offer_id": 40705900 + wave_index, "gpu_name": "RTX 3090", "num_gpus": 4,
        }
        wave.write_bytes(canonical_json_bytes({
            "schema_version": 1, "kind": "corrective_rft_wave", "wave_index": wave_index,
            "provider": dict(wave_attempts[0]["provider"]), "provider_evidence": provider_evidence,
            "attempts": [{"attempt_id": item["attempt_id"]} for item in wave_attempts],
        }))
        wave_evidence[wave_index] = {
            "wave_manifest_path": str(wave), "wave_manifest_sha256": sha256_file(wave),
            "provider_evidence": provider_evidence,
            "provider_snapshot_path": str(source_snapshot),
            "provider_snapshot_sha256": sha256_file(source_snapshot),
        }
    publication = build_corrective_publication_bundle(selection, artifacts, wave_evidence)
    assert len(publication.attempt_artifacts) == 152
    assert len(publication.selection.bindings) == 150
    snapshot = tmp_path / "snapshot"
    (snapshot / "meta").mkdir(parents=True)
    (snapshot / "data.bin").write_bytes(b"materialized snapshot")
    selected = select_corrective_successes(attempts)
    (snapshot / "manifest.json").write_bytes(canonical_json_bytes({
        "schema_version": 1, "source_format": "verified_flywheel_rft_release",
        "episode_count": 150,
        "future_actions": {"horizon": 16},
        "corrective_campaign": {"campaign_receipt_sha256": selection.campaign_receipt["receipt_sha256"]},
    }))
    (snapshot / "meta" / "rft-selection.json").write_bytes(canonical_json_bytes({
        "schema_version": 1,
        "action_horizon": 16,
        "episodes": [
            {"raw_episode_id": item.episode_id, "raw_manifest_sha256": item.episode_manifest_sha256}
            for item in selection.bindings
        ],
        "corrective_campaign": {
            "campaign_receipt_sha256": selection.campaign_receipt["receipt_sha256"],
            "selected_bindings": [
                {"attempt_id": item["attempt_id"], "episode_id": item["episode_id"], "episode_manifest_sha256": artifacts[str(item["attempt_id"])]["episode_manifest_sha256"]}
                for item in selected
            ],
        },
    }))
    return publication, snapshot


@pytest.mark.parametrize(
    ("path", "field", "value"),
    [
        ("manifest.json", "future_actions", None),
        ("manifest.json", "future_actions", {"horizon": 40}),
        ("meta/rft-selection.json", "action_horizon", None),
        ("meta/rft-selection.json", "action_horizon", 40),
    ],
)
def test_corrective_snapshot_requires_matching_16_step_horizons(
    tmp_path: Path, path: str, field: str, value: object
) -> None:
    publication, snapshot = _publication_input(tmp_path)
    current = json.loads((snapshot / path).read_text(encoding="utf-8"))
    if value is None:
        current.pop(field)
    else:
        current[field] = value
    (snapshot / path).write_bytes(canonical_json_bytes(current))

    with pytest.raises(ValueError, match="action horizon must be exactly 16"):
        publish_module._require_snapshot(publication, snapshot)


@pytest.mark.parametrize("mutation", ["manifest_count", "episodes_missing", "episode_identity"])
def test_corrective_snapshot_requires_exact_selected_payload(
    tmp_path: Path, mutation: str
) -> None:
    publication, snapshot = _publication_input(tmp_path)
    if mutation == "manifest_count":
        path = snapshot / "manifest.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["episode_count"] = 149
    else:
        path = snapshot / "meta" / "rft-selection.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if mutation == "episodes_missing":
            value["episodes"].pop()
        else:
            value["episodes"][0]["raw_episode_id"] = "forged"
    path.write_bytes(canonical_json_bytes(value))

    with pytest.raises(ValueError, match="exact selected episode payload"):
        publish_module._require_snapshot(publication, snapshot)


def _release_with_instances(tmp_path: Path):
    from lehome_train.flywheel.publish import build_corrective_release_publication_bundle

    publication, snapshot = _publication_input(tmp_path)
    receipts = {}
    for wave_index, wave in publication.wave_evidence.items():
        path = tmp_path / "instances" / f"wave-{wave_index:06d}.json"
        _write_json(path, {
            "schema_version": 1, "kind": "corrective_vast_instance", "instance_id": 10_000 + wave_index,
            "wave_index": wave_index, "host": "private.example", "port": 22,
            "provider_evidence_sha256": canonical_json_sha256(wave.provider_evidence),
        })
        receipts[wave_index] = path
    return build_corrective_release_publication_bundle(publication, receipts), snapshot


def test_release_allows_one_verified_lease_to_collect_multiple_waves(tmp_path: Path) -> None:
    from lehome_train.flywheel.publish import build_corrective_release_publication_bundle

    publication, _snapshot = _publication_input(tmp_path)
    receipts = {}
    shared_instance_id = 4242
    for wave_index, wave in publication.wave_evidence.items():
        path = tmp_path / "shared-instance" / f"wave-{wave_index:06d}.json"
        _write_json(path, {
            "schema_version": 1, "kind": "corrective_vast_instance", "instance_id": shared_instance_id,
            "wave_index": wave_index, "lease_wave_index": 0, "host": "private.example", "port": 22,
            "provider_evidence_sha256": canonical_json_sha256(wave.provider_evidence),
        })
        receipts[wave_index] = path
    bundle = build_corrective_release_publication_bundle(publication, receipts)
    assert set(bundle.instance_ids.values()) == {shared_instance_id}


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))
    return sha256_file(path)


def _canary_bundle(tmp_path: Path) -> CorrectiveCanaryPublicationBundle:
    publication, _snapshot = _publication_input(tmp_path)
    attempt = next(iter(publication.attempt_artifacts.values()))
    wave = publication.wave_evidence[attempt.attempt_receipt["wave_index"]]
    sync = tmp_path / "canary-sync"
    raw = sync / "raw" / attempt.attempt_id
    raw.parent.mkdir(parents=True)
    shutil.copytree(attempt.raw_episode_root, raw)
    baseline = {
        "parent_checkpoint_repository": APPROVED_PARENT_REPOSITORY,
        "parent_checkpoint_revision": APPROVED_PARENT_REVISION,
        "parent_checkpoint_artifact_sha256": APPROVED_PARENT_ARTIFACT_SHA256,
        "parent_checkpoint_step": 12000,
    }
    command = ["/controller", "run-trial", "--episode-id", attempt.attempt_id]
    scheduled = {
        "attempt_id": attempt.attempt_id, "episode_id": attempt.attempt_receipt["episode_id"],
        "wave_index": attempt.attempt_receipt["wave_index"], "worker_slot": attempt.attempt_receipt["worker_slot"],
        "command": command,
    }
    wave_path = tmp_path / "canary" / "wave.json"
    scheduled_wave = [
        {**scheduled, "attempt_id": f"corrective-wave-000000-worker-{slot}", "episode_id": f"corrective-wave-000000-worker-{slot}", "worker_slot": slot}
        for slot in range(4)
    ]
    wave_body = {
        "schema_version": 1, "kind": "corrective_rft_wave", "wave_index": 0,
        "baseline": baseline, "provider": dict(attempt.attempt_receipt["provider"]),
        "provider_evidence": dict(wave.provider_evidence), "attempts": scheduled_wave,
    }
    source_wave_sha = _write_json(wave_path, wave_body)
    canary_path = tmp_path / "canary" / "manifest.json"
    canary_sha = _write_json(canary_path, {
        "schema_version": 1, "kind": "corrective_rft_canary", "wave_index": 0,
        "source_wave_sha256": source_wave_sha, "baseline": baseline,
        "provider": dict(attempt.attempt_receipt["provider"]), "attempt": scheduled,
        "episode_count": 1, "external_lifecycle_required": True,
    })
    provider_path = tmp_path / "canary" / "provider-evidence.json"
    provider_sha = _write_json(provider_path, wave.provider_evidence)
    snapshot_path = Path(wave.provider_snapshot_path)
    instance_path = tmp_path / "canary" / "instance.json"
    instance_sha = _write_json(instance_path, {
        "schema_version": 1, "kind": "corrective_vast_instance", "instance_id": 123,
        "wave_index": 0, "host": "private.example", "port": 22,
        "provider_evidence_sha256": canonical_json_sha256(wave.provider_evidence),
    })
    policy_command = ["/groot-python", "policy-server", "--model-path", "/private/policies/step-12000"]
    policy = {
        "episode_id": attempt.attempt_receipt["episode_id"], "model_path": "/private/policies/step-12000", "command": policy_command,
        "checkpoint_revision": APPROVED_PARENT_REVISION,
        "checkpoint_digest": APPROVED_PARENT_ARTIFACT_SHA256,
        "code_revision": attempt.attempt_receipt["code_revision"],
        "image_identity": attempt.attempt_receipt["image_identity"],
    }
    policy_path = sync / f"policy-server-receipt-{attempt.attempt_id}.json"
    policy_path.write_bytes(canonical_json_bytes(policy))
    updated = type(attempt)(
        attempt.attempt_id, attempt.attempt_receipt, str(raw),
        attempt.episode_manifest_sha256, str(policy_path), sha256_file(policy_path),
    )
    hashes = {path.relative_to(sync).as_posix(): sha256_file(path) for path in sorted(sync.rglob("*")) if path.is_file()}
    terminal_path = tmp_path / "canary" / "terminal.json"
    terminal_sha = _write_json(terminal_path, {
        "schema_version": 1, "kind": "corrective_canary_terminal", "attempt_id": attempt.attempt_id,
        "instance_id": 123, "canary_manifest_sha256": canary_sha,
        "staged_bundle_sha256": "d" * 64, "transport_returncode": 0,
        "raw_manifest_sha256": updated.episode_manifest_sha256,
        "policy_receipt_sha256": updated.policy_receipt_sha256,
        "synced_evidence_sha256": canonical_json_sha256(hashes),
    })
    provisional = CorrectiveCanaryPublicationBundle(
        updated, str(canary_path), canary_sha, str(wave_path), source_wave_sha,
        str(provider_path), provider_sha, str(snapshot_path), sha256_file(snapshot_path),
        str(instance_path), instance_sha, str(terminal_path), terminal_sha, str(sync), "0" * 64,
    )
    body = {
        "schema_version": 1, "kind": "corrective_rft_canary_publication", "attempt_id": updated.attempt_id,
        "attempt_receipt_sha256": canonical_json_sha256(updated.attempt_receipt),
        "episode_manifest_sha256": updated.episode_manifest_sha256,
        "policy_receipt_sha256": updated.policy_receipt_sha256,
        "canary_manifest_sha256": canary_sha, "source_wave_manifest_sha256": source_wave_sha,
        "provider_evidence_sha256": provider_sha, "provider_snapshot_sha256": provisional.provider_snapshot_sha256,
        "instance_receipt_sha256": instance_sha, "terminal_receipt_sha256": terminal_sha,
        "synced_evidence_sha256": canonical_json_sha256(hashes),
    }
    return CorrectiveCanaryPublicationBundle(
        updated, str(canary_path), canary_sha, str(wave_path), source_wave_sha,
        str(provider_path), provider_sha, str(snapshot_path), sha256_file(snapshot_path),
        str(instance_path), instance_sha, str(terminal_path), terminal_sha, str(sync), canonical_json_sha256(body),
    )


def _abort_canary_bundle(tmp_path: Path) -> CorrectiveCanaryAbortPublicationBundle:
    success = _canary_bundle(tmp_path)
    sync = Path(success.synced_evidence_root)
    shutil.rmtree(sync / "raw")
    Path(success.attempt.policy_receipt_path).unlink()
    (sync / "canary.abort.log").write_text("startup hydration failed\n", encoding="utf-8")
    hashes = {
        path.relative_to(sync).as_posix(): sha256_file(path)
        for path in sorted(sync.rglob("*")) if path.is_file()
    }
    terminal_path = Path(success.terminal_receipt_path)
    terminal_sha = _write_json(terminal_path, {
        "schema_version": 1, "kind": "corrective_canary_abort",
        "attempt_id": success.attempt.attempt_id, "instance_id": 123,
        "canary_manifest_sha256": success.canary_manifest_sha256,
        "staged_bundle_sha256": "d" * 64, "transport_returncode": 1,
        "synced_evidence_sha256": canonical_json_sha256(hashes),
    })
    body = {
        "schema_version": 1, "kind": "corrective_rft_canary_abort_publication",
        "canary_manifest_sha256": success.canary_manifest_sha256,
        "source_wave_manifest_sha256": success.source_wave_manifest_sha256,
        "provider_evidence_sha256": success.provider_evidence_sha256,
        "provider_snapshot_sha256": success.provider_snapshot_sha256,
        "instance_receipt_sha256": success.instance_receipt_sha256,
        "abort_receipt_sha256": terminal_sha,
        "synced_evidence_sha256": canonical_json_sha256(hashes),
    }
    return CorrectiveCanaryAbortPublicationBundle(
        success.canary_manifest_path, success.canary_manifest_sha256,
        success.source_wave_manifest_path, success.source_wave_manifest_sha256,
        success.provider_evidence_path, success.provider_evidence_sha256,
        success.provider_snapshot_path, success.provider_snapshot_sha256,
        success.instance_receipt_path, success.instance_receipt_sha256,
        str(terminal_path), terminal_sha, str(sync), canonical_json_sha256(body),
    )


def test_corrective_publisher_uploads_all_attempt_artifacts_and_writes_disposal_only_after_readback(tmp_path: Path, monkeypatch) -> None:
    publication, snapshot = _release_with_instances(tmp_path)
    receipt = tmp_path / "disposal.json"
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    transport = FakeTransport()

    result = publish_verified_corrective_rft(
        publication, snapshot, revision="main", transport=transport, disposal_receipt=receipt,
    )

    assert result.disposable is True
    assert receipt.is_file()
    saved = json.loads(receipt.read_text(encoding="utf-8"))
    assert saved["disposable"] is True
    assert saved["immutable_revision"] == "a" * 40
    assert transport.upload_prefix == f"corrective-rft/{result.release_id}"
    assert "attempts/corrective-wave-000037-worker-3/attempt-receipt.json" in transport.uploaded_paths
    assert "attempts/corrective-wave-000037-worker-3/raw/SHA256SUMS.json" in transport.uploaded_paths
    assert "attempts/corrective-wave-000037-worker-3/policy-receipt.json" in transport.uploaded_paths
    assert "waves/wave-000037/manifest.json" in transport.uploaded_paths
    assert "waves/wave-000037/instance-receipt.json" in transport.uploaded_paths
    assert "waves/wave-000037/provider-evidence.json" in transport.uploaded_paths
    assert "waves/wave-000037/provider-source-snapshot.json" in transport.uploaded_paths
    assert "selected-150.json" in transport.uploaded_paths
    assert "materialized-snapshot/manifest.json" in transport.uploaded_paths
    assert json.loads(receipt.read_text(encoding="utf-8"))["instance_ids"]["37"] == 10037
    assert TOKEN not in receipt.read_text(encoding="utf-8")


@pytest.mark.parametrize("failure", ["upload", "list", "download"])
def test_corrective_publisher_failure_never_writes_disposal_receipt(tmp_path: Path, monkeypatch, failure: str) -> None:
    publication, snapshot = _release_with_instances(tmp_path)
    receipt = tmp_path / "disposal.json"
    monkeypatch.setenv("HF_TOKEN", TOKEN)

    with pytest.raises((RuntimeError, ValueError)):
        publish_verified_corrective_rft(
            publication, snapshot, revision="main", transport=FakeTransport(fail_at=failure), disposal_receipt=receipt,
        )

    assert not receipt.exists()


def test_corrective_publisher_resumes_immutable_readback_without_reupload(tmp_path: Path, monkeypatch) -> None:
    publication, snapshot = _release_with_instances(tmp_path)
    receipt = tmp_path / "disposal.json"
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    transport = FakeTransport(fail_at="download")
    with pytest.raises(RuntimeError):
        publish_verified_corrective_rft(
            publication, snapshot, revision="main", transport=transport, disposal_receipt=receipt,
        )
    uploaded = transport.uploaded_paths
    transport.fail_at = None

    result = verify_uploaded_corrective_rft(
        publication, snapshot, immutable_revision="a" * 40,
        transport=transport, disposal_receipt=receipt,
    )

    assert result.disposable is True
    assert transport.uploaded_paths == uploaded
    assert json.loads(receipt.read_text())["fresh_readback_verified"] is True


def test_publication_bundle_rejects_missing_or_unbound_attempt_artifacts(tmp_path: Path) -> None:
    publication, _snapshot = _publication_input(tmp_path)
    missing = dict(publication.attempt_artifacts)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="attempt artifact IDs"):
        build_corrective_publication_bundle(publication.selection, missing, publication.wave_evidence)


def test_publication_bundle_rejects_missing_per_wave_provider_evidence(tmp_path: Path) -> None:
    publication, _snapshot = _publication_input(tmp_path)
    evidence = dict(publication.wave_evidence)
    evidence.pop(0)
    with pytest.raises(ValueError, match="wave evidence"):
        build_corrective_publication_bundle(publication.selection, publication.attempt_artifacts, evidence)


def test_release_publication_rejects_cross_instance_provider_evidence(tmp_path: Path) -> None:
    publication, _snapshot = _publication_input(tmp_path)
    from lehome_train.flywheel.publish import build_corrective_release_publication_bundle

    receipts = {}
    for wave_index in publication.wave_evidence:
        path = tmp_path / "instances" / f"wave-{wave_index:06d}.json"
        _write_json(path, {
            "schema_version": 1, "kind": "corrective_vast_instance", "instance_id": 10_000 + wave_index,
            "wave_index": wave_index, "host": "private.example", "port": 22,
            "provider_evidence_sha256": "0" * 64,
        })
        receipts[wave_index] = path

    with pytest.raises(ValueError, match="instance provenance"):
        build_corrective_release_publication_bundle(publication, receipts)


def test_publication_bundle_accepts_the_real_campaign_wave_manifest_shape(tmp_path: Path) -> None:
    publication, _snapshot = _publication_input(tmp_path)
    wave = publication.wave_evidence[0]
    baseline = {
        "policy_path": "/policy", "policy_revision_file": "/policy/revision", "parent_checkpoint_repository": APPROVED_PARENT_REPOSITORY,
        "parent_checkpoint_revision": APPROVED_PARENT_REVISION, "parent_checkpoint_artifact_sha256": APPROVED_PARENT_ARTIFACT_SHA256,
        "code_revision": "f" * 40, "asset_revision": "0" * 40, "simulator_version": "5.1.0.0",
        "release_assets_root": "/assets", "image_identity": "sha256:" + "1" * 64, "groot_root": "/groot", "groot_revision": "a" * 40, "groot_python": "/python", "controller_python": "/controller-python",
    }
    produced = corrective_campaign._wave_manifest(
        tmp_path / "campaign", wave_index=0, categories=("top_long",) * 4,
        baseline=baseline, provider=publication.selection.campaign_receipt["provider_by_wave"]["0"], provider_evidence=wave.provider_evidence,
    )
    path = tmp_path / "actual-wave.json"
    path.write_bytes(canonical_json_bytes(produced))
    evidence = dict(publication.wave_evidence)
    evidence[0] = {
        "wave_manifest_path": str(path), "wave_manifest_sha256": sha256_file(path),
        "provider_evidence": wave.provider_evidence, "provider_snapshot_path": wave.provider_snapshot_path,
        "provider_snapshot_sha256": wave.provider_snapshot_sha256,
    }
    rebuilt = build_corrective_publication_bundle(publication.selection, publication.attempt_artifacts, evidence)
    assert rebuilt.wave_evidence[0].wave_manifest_sha256 == sha256_file(path)


def test_corrective_publisher_reverifies_every_raw_manifest_before_staging(tmp_path: Path, monkeypatch) -> None:
    publication, snapshot = _release_with_instances(tmp_path)
    receipt = tmp_path / "disposal.json"
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    first = next(iter(publication.corrective.attempt_artifacts.values()))
    (Path(first.raw_episode_root) / "annotations.jsonl").write_bytes(b'{"mutated":true}\n')

    with pytest.raises(ValueError, match="raw episode manifest"):
        publish_verified_corrective_rft(
            publication, snapshot, revision="main", transport=FakeTransport(), disposal_receipt=receipt,
        )

    assert not receipt.exists()


def test_private_one_episode_canary_is_read_back_before_its_disposal_receipt(tmp_path: Path, monkeypatch) -> None:
    bundle = _canary_bundle(tmp_path)
    receipt = tmp_path / "canary-disposal.json"
    monkeypatch.setenv("HF_TOKEN", TOKEN)

    result = publish_private_corrective_canary(
        bundle, revision="main", transport=FakeTransport(), disposal_receipt=receipt,
    )

    saved = json.loads(receipt.read_text(encoding="utf-8"))
    assert result.disposable is True
    assert saved == {
        "schema_version": 1, "kind": "corrective_rft_private_canary",
        "repository": "ryanjin333/lehome-groot-n17-data", "immutable_revision": "a" * 40,
        "remote_prefix": result.remote_prefix, "canary_type": "success_canary",
        "attempt_id": bundle.attempt.attempt_id, "episode_id": bundle.attempt.attempt_receipt["episode_id"],
        "instance_id": 123, "canary_sha256": bundle.canary_sha256, "entry_count": len(result.entries),
        "repository_private": True, "tree_listing_verified": True, "fresh_readback_verified": True,
        "training_admission": False, "disposable": True,
    }
    assert {"canary-manifest.json", "source-wave-manifest.json", "provider-evidence.json", "provider-source-snapshot.json", "instance-receipt.json", "terminal-receipt.json"} <= {entry.relative_path for entry in result.entries}


def test_private_canary_rejects_a_bare_attempt_artifact_without_lifecycle_provenance(tmp_path: Path, monkeypatch) -> None:
    publication, _snapshot = _publication_input(tmp_path)
    receipt = tmp_path / "forged-canary-disposal.json"
    monkeypatch.setenv("HF_TOKEN", TOKEN)

    with pytest.raises(ValueError, match="publication bundle"):
        publish_private_corrective_canary(
            next(iter(publication.attempt_artifacts.values())),
            revision="main", transport=FakeTransport(), disposal_receipt=receipt,
        )

    assert not receipt.exists()


def test_private_abort_canary_publishes_synced_evidence_without_raw_artifacts(tmp_path: Path, monkeypatch) -> None:
    bundle = _abort_canary_bundle(tmp_path)
    receipt = tmp_path / "canary-abort-disposal.json"
    monkeypatch.setenv("HF_TOKEN", TOKEN)

    result = publish_private_corrective_canary_abort(
        bundle, revision="main", transport=FakeTransport(), disposal_receipt=receipt,
    )

    saved = json.loads(receipt.read_text(encoding="utf-8"))
    assert result.disposable is True
    assert saved["kind"] == "corrective_rft_private_canary"
    assert saved["canary_type"] == "abort_canary"
    assert saved["instance_id"] == 123
    assert saved["training_admission"] is False
    assert saved["fresh_readback_verified"] is True


def test_private_task_failure_canary_publishes_raw_evidence_without_training_admission(tmp_path: Path, monkeypatch) -> None:
    success = _canary_bundle(tmp_path)
    sync = Path(success.synced_evidence_root)
    raw_hash = sha256_file(Path(success.attempt.raw_episode_root) / "SHA256SUMS.json")
    policy_hash = sha256_file(Path(success.attempt.policy_receipt_path))
    abort_path = Path(success.terminal_receipt_path)
    abort_body = json.loads(abort_path.read_text(encoding="utf-8"))
    abort_body.update({
        "kind": "corrective_canary_non_training_abort", "transport_returncode": 0,
        "non_training_admitted": False, "raw_manifest_sha256": raw_hash,
        "policy_receipt_sha256": policy_hash,
    })
    abort_body.pop("accepted_success", None)
    _write_json(abort_path, abort_body)
    bundle = build_corrective_canary_abort_publication_bundle(
        canary_manifest_path=success.canary_manifest_path, source_wave_manifest_path=success.source_wave_manifest_path,
        provider_evidence_path=success.provider_evidence_path, provider_snapshot_path=success.provider_snapshot_path,
        instance_receipt_path=success.instance_receipt_path, abort_receipt_path=abort_path,
        synced_evidence_root=success.synced_evidence_root,
    )
    receipt = tmp_path / "task-failure.json"
    monkeypatch.setenv("HF_TOKEN", TOKEN)

    publish_private_corrective_canary_abort(bundle, revision="main", transport=FakeTransport(), disposal_receipt=receipt)

    saved = json.loads(receipt.read_text(encoding="utf-8"))
    assert saved["canary_type"] == "task_failure_canary"
    assert saved["training_admission"] is False and saved["disposable"] is True


@pytest.mark.parametrize("failure", ["upload", "list", "download"])
def test_private_abort_canary_failure_never_writes_disposal_receipt(tmp_path: Path, monkeypatch, failure: str) -> None:
    bundle = _abort_canary_bundle(tmp_path)
    receipt = tmp_path / "abort-failed.json"
    monkeypatch.setenv("HF_TOKEN", TOKEN)

    with pytest.raises((RuntimeError, ValueError)):
        publish_private_corrective_canary_abort(
            bundle, revision="main", transport=FakeTransport(fail_at=failure), disposal_receipt=receipt,
        )

    assert not receipt.exists()


def test_private_abort_canary_rejects_empty_synced_evidence(tmp_path: Path, monkeypatch) -> None:
    bundle = _abort_canary_bundle(tmp_path)
    for path in Path(bundle.synced_evidence_root).iterdir():
        if path.is_file():
            path.unlink()
    monkeypatch.setenv("HF_TOKEN", TOKEN)

    with pytest.raises(ValueError, match="artifact root is empty"):
        publish_private_corrective_canary_abort(
            bundle, revision="main", transport=FakeTransport(), disposal_receipt=tmp_path / "abort-empty.json",
        )


def test_canary_bundle_builders_accept_lifecycle_success_and_early_abort_evidence(tmp_path: Path) -> None:
    success = _canary_bundle(tmp_path)
    rebuilt_success = build_corrective_canary_publication_bundle(
        attempt_receipt=success.attempt.attempt_receipt, raw_episode_root=success.attempt.raw_episode_root,
        policy_receipt_path=success.attempt.policy_receipt_path, canary_manifest_path=success.canary_manifest_path,
        source_wave_manifest_path=success.source_wave_manifest_path, provider_evidence_path=success.provider_evidence_path,
        provider_snapshot_path=success.provider_snapshot_path, instance_receipt_path=success.instance_receipt_path,
        terminal_receipt_path=success.terminal_receipt_path, synced_evidence_root=success.synced_evidence_root,
    )
    abort = _abort_canary_bundle(tmp_path / "abort")
    rebuilt_abort = build_corrective_canary_abort_publication_bundle(
        canary_manifest_path=abort.canary_manifest_path, source_wave_manifest_path=abort.source_wave_manifest_path,
        provider_evidence_path=abort.provider_evidence_path, provider_snapshot_path=abort.provider_snapshot_path,
        instance_receipt_path=abort.instance_receipt_path, abort_receipt_path=abort.abort_receipt_path,
        synced_evidence_root=abort.synced_evidence_root,
    )
    assert rebuilt_success.canary_sha256 == success.canary_sha256
    assert rebuilt_abort.abort_sha256 == abort.abort_sha256


def _cli_args(bundle, receipt: Path, *, kind: str):
    parser = canary_cli.build_parser()
    arguments = [
        "--kind", kind, "--revision", "main", "--disposal-receipt", str(receipt),
        "--canary-manifest", bundle.canary_manifest_path,
        "--source-wave-manifest", bundle.source_wave_manifest_path,
        "--provider-evidence", bundle.provider_evidence_path,
        "--provider-snapshot", bundle.provider_snapshot_path,
        "--instance-receipt", bundle.instance_receipt_path,
        "--terminal-receipt", bundle.terminal_receipt_path if kind == "success" else bundle.abort_receipt_path,
        "--synced-evidence-root", bundle.synced_evidence_root,
    ]
    if kind == "success":
        arguments += [
            "--attempt-receipt", str(Path(bundle.synced_evidence_root).parent / "attempt-receipt.json"),
            "--raw-episode-root", bundle.attempt.raw_episode_root,
            "--policy-receipt", bundle.attempt.policy_receipt_path,
        ]
        _write_json(Path(arguments[arguments.index("--attempt-receipt") + 1]), bundle.attempt.attempt_receipt)
    return parser.parse_args(arguments)


@pytest.mark.parametrize("kind", ["success", "abort"])
def test_literal_canary_publisher_cli_constructs_and_publishes_each_bundle_kind(tmp_path: Path, monkeypatch, kind: str) -> None:
    bundle = _canary_bundle(tmp_path) if kind == "success" else _abort_canary_bundle(tmp_path)
    receipt = tmp_path / f"{kind}.json"
    monkeypatch.setenv("HF_TOKEN", TOKEN)

    result = canary_cli.run(_cli_args(bundle, receipt, kind=kind), transport=FakeTransport())

    saved = json.loads(receipt.read_text(encoding="utf-8"))
    assert result["immutable_revision"] == "a" * 40
    assert saved["canary_type"] == f"{kind}_canary"
