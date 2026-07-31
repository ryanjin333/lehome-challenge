from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from lehome_train.commands.restore import restore_experiment_snapshot
from lehome_train.commands.sync import SyncResult
from lehome_train.constants import DEFAULT_MODEL_REPO
from lehome_train.models import SyncEntry, SyncManifest


class FakeTransport:
    def __init__(self, remote: dict[str, bytes]) -> None:
        self.remote = remote
        self.downloads: list[tuple[str, str | None, tuple[str, ...]]] = []

    def check_access(self, *, repository: str, token: str):
        from lehome_train.hub import HubAccess

        return HubAccess(can_read=True, can_write=False)

    def download_files(
        self,
        *,
        repository: str,
        revision: str,
        destination: Path,
        relative_paths: tuple[str, ...],
        token: str,
        remote_prefix: str | None = None,
    ) -> str:
        self.downloads.append((revision, remote_prefix, relative_paths))
        for relative in relative_paths:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.remote[relative])
        return revision


def _sync_result(remote: dict[str, bytes]) -> SyncResult:
    entries = tuple(
        SyncEntry(
            relative_path=path,
            sha256=hashlib.sha256(content).hexdigest(),
            byte_size=len(content),
            remotely_verified=True,
        )
        for path, content in sorted(remote.items())
    )
    return SyncResult(
        repository=DEFAULT_MODEL_REPO,
        immutable_revision="a" * 40,
        manifest=SyncManifest(
            experiment_id="experiment-001",
            experiment_config_sha256="b" * 64,
            remote_prefix="experiments/experiment-001/" + "c" * 64,
            entries=entries,
        ),
        disposable=True,
    )


def test_restore_downloads_exact_immutable_snapshot_and_exposes_after_verification(
    tmp_path: Path,
) -> None:
    remote = {
        "checkpoints/step-1000.tar": b"checkpoint",
        "resolved-config.json": b"{}",
        "reports/training.json": b"{}",
    }
    transport = FakeTransport(remote)
    destination = tmp_path / "restored"

    restored = restore_experiment_snapshot(
        destination,
        sync_result=_sync_result(remote),
        transport=transport,
        staging_root=tmp_path,
        environ={"HF_TOKEN": "hf_explicit_restore_token"},
    )

    assert restored == destination
    assert {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    } == set(remote)
    assert transport.downloads == [
        (
            "a" * 40,
            "experiments/experiment-001/" + "c" * 64,
            tuple(sorted(remote)),
        )
    ]


def test_restore_rejects_hash_mismatch_without_partial_destination(tmp_path: Path) -> None:
    expected = {"checkpoints/step-1000.tar": b"expected"}
    transport = FakeTransport({"checkpoints/step-1000.tar": b"corrupt"})
    destination = tmp_path / "restored"

    with pytest.raises(ValueError, match="verification"):
        restore_experiment_snapshot(
            destination,
            sync_result=_sync_result(expected),
            transport=transport,
            staging_root=tmp_path,
            environ={"HF_TOKEN": "hf_explicit_restore_token"},
        )

    assert not destination.exists()


def test_restore_refuses_overwrite_and_unverified_sync_result(tmp_path: Path) -> None:
    remote = {"resolved-config.json": b"{}"}
    destination = tmp_path / "restored"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        restore_experiment_snapshot(
            destination,
            sync_result=_sync_result(remote),
            transport=FakeTransport(remote),
            staging_root=tmp_path,
            environ={"HF_TOKEN": "hf_explicit_restore_token"},
        )

    destination.rmdir()
    unverified = _sync_result(remote)
    object.__setattr__(unverified, "disposable", False)
    with pytest.raises(ValueError, match="verified"):
        restore_experiment_snapshot(
            destination,
            sync_result=unverified,
            transport=FakeTransport(remote),
            staging_root=tmp_path,
            environ={"HF_TOKEN": "hf_explicit_restore_token"},
        )
