from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

import pytest

from lehome_train.commands.sync import generate_sync_manifest, sync_experiment
from lehome_train.constants import DEFAULT_MODEL_REPO
from lehome_train.models import SyncEntry


CONFIG_SHA = "a" * 64
TOKEN = "hf_process_memory_only"


class FakeTransport:
    def __init__(
        self,
        *,
        corrupt_path: str | None = None,
        before_upload: Callable[[], None] | None = None,
        extra_remote_path: str | None = None,
        symlink_remote_path: str | None = None,
    ) -> None:
        self.corrupt_path = corrupt_path
        self.before_upload = before_upload
        self.extra_remote_path = extra_remote_path
        self.symlink_remote_path = symlink_remote_path
        self.uploaded: dict[str, bytes] = {}
        self.upload_source: Path | None = None
        self.upload_token: str | None = None
        self.download_token: str | None = None

    def check_access(self, *, repository: str, token: str):
        from lehome_train.hub import HubAccess

        assert repository == DEFAULT_MODEL_REPO
        return HubAccess(can_read=True, can_write=True, private_repository=True)

    def upload_files(
        self,
        *,
        repository: str,
        revision: str,
        source: Path,
        entries: tuple[SyncEntry, ...],
        token: str,
    ) -> str:
        self.upload_token = token
        self.upload_source = source
        if self.before_upload is not None:
            self.before_upload()
        self.uploaded = {
            entry.relative_path: (source / entry.relative_path).read_bytes()
            for entry in entries
        }
        return "b" * 40

    def download_files(
        self,
        *,
        repository: str,
        revision: str,
        destination: Path,
        relative_paths: tuple[str, ...],
        token: str,
    ) -> str:
        self.download_token = token
        for relative_path in relative_paths:
            target = destination / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.uploaded[relative_path])
        if self.corrupt_path is not None:
            (destination / self.corrupt_path).write_bytes(b"remote mismatch")
        if self.extra_remote_path is not None:
            extra = destination / self.extra_remote_path
            extra.parent.mkdir(parents=True, exist_ok=True)
            extra.write_bytes(b"unexpected remote artifact")
        if self.symlink_remote_path is not None:
            link = destination / self.symlink_remote_path
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(destination / relative_paths[0])
        return revision


def _experiment(root: Path) -> None:
    files = {
        "checkpoints/step-1000.tar.zst": b"checkpoint",
        "logs/train.log": b"redacted training log",
        "resolved-config.json": b'{"batch":64}',
        "provenance.json": b'{"commit":"immutable"}',
        "reports/training-report.json": b'{"cost":1.25}',
    }
    for relative_path, content in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _staging_root(experiment_root: Path) -> Path:
    staging = experiment_root.with_name(experiment_root.name + "-staging")
    staging.mkdir()
    return staging


def test_generated_sync_manifest_is_a_closed_complete_allowlist(tmp_path: Path) -> None:
    _experiment(tmp_path)
    (tmp_path / "scratch.txt").write_text("do not upload", encoding="utf-8")

    manifest = generate_sync_manifest(
        tmp_path,
        experiment_id="experiment-001",
        experiment_config_sha256=CONFIG_SHA,
    )

    assert tuple(entry.relative_path for entry in manifest.entries) == (
        "checkpoints/step-1000.tar.zst",
        "logs/train.log",
        "provenance.json",
        "reports/training-report.json",
        "resolved-config.json",
    )
    assert all(not entry.remotely_verified for entry in manifest.entries)
    assert (tmp_path / "sync-manifest.json").is_file()


def test_sync_marks_disposable_only_after_immutable_remote_hash_readback(
    tmp_path: Path,
) -> None:
    _experiment(tmp_path)
    transport = FakeTransport()
    staging_root = _staging_root(tmp_path)

    result = sync_experiment(
        tmp_path,
        experiment_id="experiment-001",
        experiment_config_sha256=CONFIG_SHA,
        repository=DEFAULT_MODEL_REPO,
        revision="experiment-001",
        transport=transport,
        staging_root=staging_root,
        environ={"HF_TOKEN": TOKEN},
    )

    assert result.immutable_revision == "b" * 40
    assert result.disposable is True
    assert all(entry.remotely_verified for entry in result.manifest.entries)
    assert transport.upload_token == TOKEN
    assert transport.download_token == TOKEN
    assert transport.upload_source != tmp_path
    assert list(staging_root.iterdir()) == []
    assert TOKEN not in (tmp_path / "sync-manifest.json").read_text(encoding="utf-8")


def test_sync_refuses_to_mark_unmatched_artifacts_disposable(tmp_path: Path) -> None:
    _experiment(tmp_path)
    transport = FakeTransport(corrupt_path="checkpoints/step-1000.tar.zst")
    staging_root = _staging_root(tmp_path)

    result = sync_experiment(
        tmp_path,
        experiment_id="experiment-001",
        experiment_config_sha256=CONFIG_SHA,
        repository=DEFAULT_MODEL_REPO,
        revision="experiment-001",
        transport=transport,
        staging_root=staging_root,
        environ={"HF_TOKEN": TOKEN},
    )

    by_path = {entry.relative_path: entry for entry in result.manifest.entries}
    assert result.disposable is False
    assert by_path["checkpoints/step-1000.tar.zst"].remotely_verified is False


def test_sync_uploads_a_staged_snapshot_when_live_source_mutates(tmp_path: Path) -> None:
    _experiment(tmp_path)
    checkpoint = tmp_path / "checkpoints" / "step-1000.tar.zst"
    staging_root = _staging_root(tmp_path)
    transport = FakeTransport(
        before_upload=lambda: checkpoint.write_bytes(b"mutated after scan")
    )

    result = sync_experiment(
        tmp_path,
        experiment_id="experiment-001",
        experiment_config_sha256=CONFIG_SHA,
        repository=DEFAULT_MODEL_REPO,
        revision="experiment-001",
        transport=transport,
        staging_root=staging_root,
        environ={"HF_TOKEN": TOKEN},
    )

    assert result.disposable is True
    assert transport.uploaded["checkpoints/step-1000.tar.zst"] == b"checkpoint"
    assert checkpoint.read_bytes() == b"mutated after scan"


@pytest.mark.parametrize("remote_kind", ["extra", "symlink"])
def test_sync_rejects_unexpected_remote_tree_entries(
    tmp_path: Path,
    remote_kind: str,
) -> None:
    _experiment(tmp_path)
    staging_root = _staging_root(tmp_path)
    transport = FakeTransport(
        extra_remote_path="reports/unexpected.json" if remote_kind == "extra" else None,
        symlink_remote_path="reports/unexpected.json" if remote_kind == "symlink" else None,
    )

    result = sync_experiment(
        tmp_path,
        experiment_id="experiment-001",
        experiment_config_sha256=CONFIG_SHA,
        repository=DEFAULT_MODEL_REPO,
        revision="experiment-001",
        transport=transport,
        staging_root=staging_root,
        environ={"HF_TOKEN": TOKEN},
    )

    assert result.disposable is False
    assert not any(entry.remotely_verified for entry in result.manifest.entries)
    assert list(staging_root.iterdir()) == []


def test_sync_requires_caller_selected_capacity_checked_staging_root(
    tmp_path: Path,
) -> None:
    _experiment(tmp_path)
    staging_root = _staging_root(tmp_path)

    with pytest.raises(ValueError, match="insufficient"):
        sync_experiment(
            tmp_path,
            experiment_id="experiment-001",
            experiment_config_sha256=CONFIG_SHA,
            repository=DEFAULT_MODEL_REPO,
            revision="experiment-001",
            transport=FakeTransport(),
            staging_root=staging_root,
            free_space_probe=lambda _path: 0,
            environ={"HF_TOKEN": TOKEN},
        )


@pytest.mark.parametrize(
    ("relative_path", "content"),
    [
        ("logs/.env", b"SAFE=value"),
        ("reports/result.json", ("hf_" + "s" * 40).encode()),
    ],
)
def test_sync_fails_closed_on_secret_bearing_artifacts(
    tmp_path: Path,
    relative_path: str,
    content: bytes,
) -> None:
    _experiment(tmp_path)
    target = tmp_path / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)

    with pytest.raises(ValueError, match="upload policy") as error:
        generate_sync_manifest(
            tmp_path,
            experiment_id="experiment-001",
            experiment_config_sha256=CONFIG_SHA,
        )

    assert content.decode() not in str(error.value)


def test_sync_rejects_symlinks_and_missing_required_artifact_groups(
    tmp_path: Path,
) -> None:
    _experiment(tmp_path)
    shutil.rmtree(tmp_path / "reports")

    with pytest.raises(ValueError, match="reports"):
        generate_sync_manifest(
            tmp_path,
            experiment_id="experiment-001",
            experiment_config_sha256=CONFIG_SHA,
        )

    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "training-report.json").symlink_to(
        tmp_path / "provenance.json"
    )
    with pytest.raises(ValueError, match="upload policy"):
        generate_sync_manifest(
            tmp_path,
            experiment_id="experiment-001",
            experiment_config_sha256=CONFIG_SHA,
        )
