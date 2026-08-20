"""Idempotent, readback-verified Hugging Face sync for accepted episodes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lehome.flywheel.hub_sync import HubSyncDaemon, HubSyncError
from lehome_train.hub import HubAccess, HubTreeEntry, HubTransientError
from lehome_train.constants import DEFAULT_ROLLOUT_REPO
from lehome_train.models import SyncEntry
from lehome_train.flywheel.publish import (
    RolloutRoundSealError,
    seal_rollout_round,
)


REPOSITORY = DEFAULT_ROLLOUT_REPO
ROUND_ID = "round-1"
PUBLICATION_REF = "rollout-round-1"
FIRST_COMMIT = "1" * 40


class FakeTransport:
    """In-memory HubTransport double with failure injection."""

    def __init__(self):
        self.store: dict[str, dict[str, bytes]] = {}
        self.upload_failures = 0
        self.uploads: list[tuple[str, tuple[str, ...]]] = []
        self.readbacks = 0
        self.readback_revisions: list[str] = []

    def check_access(self, *, repository, token):
        return HubAccess(can_read=True, can_write=True)

    def upload_files(self, *, repository, revision, source, entries, token, remote_prefix=None):
        self.uploads.append((remote_prefix, tuple(entry.relative_path for entry in entries)))
        if self.upload_failures > 0:
            self.upload_failures -= 1
            raise HubTransientError("simulated upload failure")
        bucket = self.store.setdefault(remote_prefix, {})
        for entry in entries:
            data = (Path(source) / entry.relative_path).read_bytes()
            bucket[entry.relative_path] = data
        return FIRST_COMMIT

    def upload_large_folder(self, **kwargs):
        raise AssertionError("episode sync must use upload_files")

    def download_files(self, *, repository, revision, destination, relative_paths, token, remote_prefix=None):
        self.readbacks += 1
        self.readback_revisions.append(revision)
        bucket = self.store.get(remote_prefix, {})
        for relative_path in relative_paths:
            target = Path(destination) / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bucket[relative_path])
        return revision

    def list_tree(self, *, repository, revision, token, remote_prefix=None):
        bucket = self.store.get(remote_prefix, {})
        entries = [HubTreeEntry(remote_prefix, "directory")]
        for relative_path in sorted(bucket):
            entries.append(HubTreeEntry(f"{remote_prefix}/{relative_path}", "file"))
        return tuple(entries)

    def resolve_approved_ref(self, *, repository, ref, token):
        return "0" * 40


def _make_accepted_episode(accepted_root: Path, attempt_id: str) -> Path:
    episode = accepted_root / attempt_id
    (episode / "videos").mkdir(parents=True)
    (episode / "videos" / "top.mp4").write_bytes(b"video-bytes")
    (episode / "worker-receipt.json").write_text(json.dumps({
        "schema_version": 1, "attempt_id": attempt_id,
        "lease_id": "lease-1", "worker_id": "worker-1",
        "outcome": {"success": True, "metrics": [{"success": True}]},
    }, sort_keys=True))
    manifest = {}
    for current, _dirs, names in __import__("os").walk(episode):
        for name in names:
            path = Path(current) / name
            relative = path.relative_to(episode).as_posix()
            manifest[relative] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size": path.stat().st_size}
    (episode / "SHA256SUMS.json").write_text(json.dumps(manifest, sort_keys=True))
    return episode


@pytest.fixture()
def daemon(tmp_path):
    accepted_root = tmp_path / "accepted"
    accepted_root.mkdir()
    transport = FakeTransport()
    sync = HubSyncDaemon(
        repository=REPOSITORY, round_id=ROUND_ID, token="token",
        transport=transport, accepted_root=accepted_root,
        receipts_root=tmp_path / "receipts", readback_root=tmp_path / "readback",
        revision=PUBLICATION_REF, max_attempts=3,
    )
    return sync, transport, accepted_root


def test_sync_uploads_immutable_path_and_verifies_readback(daemon):
    sync, transport, accepted_root = daemon
    episode = _make_accepted_episode(accepted_root, "attempt-1")

    receipt = sync.sync_episode("attempt-1", episode)

    assert receipt.attempt_id == "attempt-1"
    assert receipt.readback_verified
    prefix = f"rollout-rounds/{ROUND_ID}/attempt-1"
    assert transport.uploads == [(prefix, ("videos/top.mp4", "worker-receipt.json"))]
    assert transport.readbacks == 1
    assert transport.readback_revisions == [FIRST_COMMIT]
    receipt_file = sync.receipts_root / "attempt-1.sync.json"
    assert receipt_file.is_file()
    on_disk = json.loads(receipt_file.read_text())
    assert on_disk["remote_prefix"] == prefix
    assert on_disk["publication_ref"] == PUBLICATION_REF
    assert on_disk["immutable_revision"] == FIRST_COMMIT
    assert on_disk["readback_verified"] is True
    assert on_disk["repository"] == DEFAULT_ROLLOUT_REPO


def test_duplicate_sync_returns_existing_receipt_without_reupload(daemon):
    sync, transport, accepted_root = daemon
    episode = _make_accepted_episode(accepted_root, "attempt-1")
    first = sync.sync_episode("attempt-1", episode)
    upload_count = len(transport.uploads)

    second = sync.sync_episode("attempt-1", episode)

    assert second == first
    assert len(transport.uploads) == upload_count


def test_upload_retry_succeeds_after_transient_failures(daemon):
    sync, transport, accepted_root = daemon
    transport.upload_failures = 2
    episode = _make_accepted_episode(accepted_root, "attempt-2")

    receipt = sync.sync_episode("attempt-2", episode)

    assert receipt.readback_verified
    assert len(transport.uploads) == 3


def test_upload_retry_exhaustion_raises_without_receipt(daemon):
    sync, transport, accepted_root = daemon
    transport.upload_failures = 99
    episode = _make_accepted_episode(accepted_root, "attempt-3")

    with pytest.raises(HubSyncError, match="upload"):
        sync.sync_episode("attempt-3", episode)

    assert len(transport.uploads) == 3
    assert not (sync.receipts_root / "attempt-3.sync.json").exists()


def test_remote_hash_mismatch_fails_readback(daemon):
    sync, transport, accepted_root = daemon
    episode = _make_accepted_episode(accepted_root, "attempt-4")
    original_download = transport.download_files

    def corrupting_download(**kwargs):
        revision = original_download(**kwargs)
        for path in Path(kwargs["destination"]).rglob("*"):
            if path.is_file():
                path.write_bytes(path.read_bytes() + b"corruption")
        return revision

    transport.download_files = corrupting_download

    with pytest.raises(HubSyncError, match="readback"):
        sync.sync_episode("attempt-4", episode)


def test_missing_accepted_directory_rejected(daemon):
    sync, _transport, accepted_root = daemon
    with pytest.raises(HubSyncError, match="accepted episode"):
        sync.sync_episode("attempt-5", accepted_root / "attempt-5")


def test_episode_outside_accepted_root_rejected(daemon, tmp_path):
    sync, _transport, _accepted_root = daemon
    with pytest.raises(HubSyncError, match="accepted root"):
        sync.sync_episode("attempt-6", tmp_path)


def test_round_pending_reports_unsynced_episodes(daemon):
    sync, _transport, accepted_root = daemon
    _make_accepted_episode(accepted_root, "attempt-7")
    synced_episode = _make_accepted_episode(accepted_root, "attempt-8")
    sync.sync_episode("attempt-8", synced_episode)

    pending = sync.pending_for_round(("attempt-7", "attempt-8"))

    assert pending == ("attempt-7",)
    assert not sync.round_sealable(("attempt-7", "attempt-8"))
    assert sync.round_sealable(("attempt-8",))
def test_immutable_commit_is_rejected_as_publication_target(tmp_path):
    accepted_root = tmp_path / "accepted"
    accepted_root.mkdir()
    with pytest.raises(HubSyncError, match="mutable branch"):
        HubSyncDaemon(
            repository=REPOSITORY, round_id=ROUND_ID, token="token",
            transport=FakeTransport(), accepted_root=accepted_root,
            receipts_root=tmp_path / "receipts", readback_root=tmp_path / "readback",
            revision="0" * 40, max_attempts=3,
        )
def test_round_seal_requires_every_sync_receipt(daemon, tmp_path):
    sync, _transport, accepted_root = daemon
    synced = _make_accepted_episode(accepted_root, "attempt-9")
    receipt = sync.sync_episode("attempt-9", synced)

    seal = seal_rollout_round(
        receipts_root=sync.receipts_root,
        round_id=ROUND_ID,
        attempt_ids=("attempt-9",),
        seal_receipt_path=tmp_path / "round-1.seal.json",
    )
    assert seal.episode_count == 1
    assert seal.round_id == ROUND_ID
    on_disk = json.loads((tmp_path / "round-1.seal.json").read_text())
    assert on_disk["readback_verified"] is True
    assert on_disk["episode_sha256s"]["attempt-9"] == receipt.episode_sha256
    assert on_disk["immutable_revisions"]["attempt-9"] == receipt.immutable_revision
    assert on_disk["repository"] == DEFAULT_ROLLOUT_REPO

    with pytest.raises(RolloutRoundSealError, match="missing"):
        seal_rollout_round(
            receipts_root=sync.receipts_root,
            round_id=ROUND_ID,
            attempt_ids=("attempt-9", "attempt-never-synced"),
            seal_receipt_path=tmp_path / "round-1.bad.json",
        )


def test_round_seal_rejects_unverified_or_wrong_round_receipt(daemon, tmp_path):
    sync, _transport, accepted_root = daemon
    synced = _make_accepted_episode(accepted_root, "attempt-10")
    sync.sync_episode("attempt-10", synced)
    receipt_file = sync.receipts_root / "attempt-10.sync.json"
    payload = json.loads(receipt_file.read_text())
    payload["round_id"] = "round-other"
    receipt_file.write_text(json.dumps(payload, sort_keys=True))

    with pytest.raises(RolloutRoundSealError, match="round_id"):
        seal_rollout_round(
            receipts_root=sync.receipts_root,
            round_id=ROUND_ID,
            attempt_ids=("attempt-10",),
            seal_receipt_path=tmp_path / "round-1.bad2.json",
        )

    payload["round_id"] = ROUND_ID
    payload["readback_verified"] = False
    receipt_file.write_text(json.dumps(payload, sort_keys=True))
    with pytest.raises(RolloutRoundSealError, match="readback"):
        seal_rollout_round(
            receipts_root=sync.receipts_root,
            round_id=ROUND_ID,
            attempt_ids=("attempt-10",),
            seal_receipt_path=tmp_path / "round-1.bad3.json",
        )


def test_round_seal_rejects_receipt_without_exact_immutable_hub_binding(daemon, tmp_path):
    sync, _transport, accepted_root = daemon
    synced = _make_accepted_episode(accepted_root, "attempt-immutable")
    sync.sync_episode("attempt-immutable", synced)
    receipt_file = sync.receipts_root / "attempt-immutable.sync.json"
    original = json.loads(receipt_file.read_text())

    for field, value in (
        ("repository", "owner/other"),
        ("immutable_revision", "main"),
        ("remote_prefix", "rollout-rounds/round-1/other-attempt"),
        ("attempt_id", "other-attempt"),
    ):
        payload = dict(original)
        payload[field] = value
        receipt_file.write_text(json.dumps(payload, sort_keys=True))
        with pytest.raises(RolloutRoundSealError, match="repository|immutable|prefix|attempt_id"):
            seal_rollout_round(
                receipts_root=sync.receipts_root,
                round_id=ROUND_ID,
                attempt_ids=("attempt-immutable",),
                seal_receipt_path=tmp_path / f"bad-{field}.json",
            )


def test_round_seal_enforces_target_bounds_and_no_reseal(daemon, tmp_path):
    sync, _transport, accepted_root = daemon
    synced = _make_accepted_episode(accepted_root, "attempt-11")
    sync.sync_episode("attempt-11", synced)

    seal_path = tmp_path / "round-1.sealed.json"
    seal_rollout_round(
        receipts_root=sync.receipts_root, round_id=ROUND_ID,
        attempt_ids=("attempt-11",), seal_receipt_path=seal_path,
    )
    with pytest.raises(RolloutRoundSealError, match="already"):
        seal_rollout_round(
            receipts_root=sync.receipts_root, round_id=ROUND_ID,
            attempt_ids=("attempt-11",), seal_receipt_path=seal_path,
        )

    with pytest.raises(RolloutRoundSealError, match="at least one"):
        seal_rollout_round(
            receipts_root=sync.receipts_root, round_id=ROUND_ID,
            attempt_ids=(), seal_receipt_path=tmp_path / "empty.json",
        )

    too_many = tuple(f"attempt-{i}" for i in range(151))
    with pytest.raises(RolloutRoundSealError, match="150"):
        seal_rollout_round(
            receipts_root=sync.receipts_root, round_id=ROUND_ID,
            attempt_ids=too_many, seal_receipt_path=tmp_path / "many.json",
        )
