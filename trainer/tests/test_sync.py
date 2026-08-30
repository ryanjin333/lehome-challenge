from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

import pytest

import lehome_train.hub as hub_module
import lehome_train.commands.sync as sync_module
from lehome_train.commands.sync import generate_sync_manifest, sync_experiment
from lehome_train.constants import DEFAULT_MODEL_REPO
from lehome_train.io import canonical_json_sha256
from lehome_train.models import SyncEntry


CONFIG_SHA = "a" * 64
TOKEN = "hf_process_memory_only"


class FakeTransport:
    def __init__(
        self,
        *,
        corrupt_path: str | None = None,
        before_upload: Callable[[], None] | None = None,
        extra_scope: str | None = None,
        symlink_under_prefix: bool = False,
        list_failure: bool = False,
    ) -> None:
        self.corrupt_path = corrupt_path
        self.before_upload = before_upload
        self.extra_scope = extra_scope
        self.symlink_under_prefix = symlink_under_prefix
        self.list_failure = list_failure
        self.uploaded: dict[str, bytes] = {}
        self.remote: dict[str, tuple[str, bytes]] = {}
        self.upload_source: Path | None = None
        self.upload_prefix: str | None = None
        self.download_prefix: str | None = None
        self.list_revision: str | None = None
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
        remote_prefix: str | None = None,
    ) -> str:
        self.upload_token = token
        self.upload_source = source
        self.upload_prefix = remote_prefix
        assert remote_prefix is not None
        if self.before_upload is not None:
            self.before_upload()
        self.uploaded = {
            entry.relative_path: (source / entry.relative_path).read_bytes()
            for entry in entries
        }
        for relative_path, content in self.uploaded.items():
            self.remote[f"{remote_prefix}/{relative_path}"] = ("file", content)
        if self.corrupt_path is not None:
            self.remote[f"{remote_prefix}/{self.corrupt_path}"] = (
                "file",
                b"remote mismatch",
            )
        if self.extra_scope == "inside":
            self.remote[f"{remote_prefix}/reports/unexpected.json"] = (
                "file",
                b"unexpected remote artifact",
            )
        elif self.extra_scope == "outside":
            self.remote["experiments/another-run/unrelated.bin"] = (
                "file",
                b"unrelated remote artifact",
            )
        if self.symlink_under_prefix:
            self.remote[f"{remote_prefix}/reports/unexpected.json"] = (
                "symlink",
                b"reports/training-report.json",
            )
        return "b" * 40

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
        self.download_token = token
        self.download_prefix = remote_prefix
        assert remote_prefix is not None
        for relative_path in relative_paths:
            target = destination / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.remote[f"{remote_prefix}/{relative_path}"][1])
        return revision

    def list_tree(
        self,
        *,
        repository: str,
        revision: str,
        token: str,
    ) -> tuple[hub_module.HubTreeEntry, ...]:
        self.list_revision = revision
        if self.list_failure:
            raise OSError("remote listing unavailable")
        return tuple(
            hub_module.HubTreeEntry(path, entry_type)
            for path, (entry_type, _content) in sorted(self.remote.items())
        )


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
    manifest_identity = {
        "schema_version": 1,
        "experiment_id": "experiment-001",
        "experiment_config_sha256": CONFIG_SHA,
        "entries": [entry.to_dict() for entry in manifest.entries],
    }
    assert manifest.remote_prefix == (
        "experiments/experiment-001/" + canonical_json_sha256(manifest_identity)
    )
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
    assert transport.upload_prefix == result.manifest.remote_prefix
    assert transport.download_prefix == result.manifest.remote_prefix
    assert transport.list_revision == result.immutable_revision
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
def test_sync_rejects_unexpected_entries_already_present_under_remote_prefix(
    tmp_path: Path,
    remote_kind: str,
) -> None:
    _experiment(tmp_path)
    staging_root = _staging_root(tmp_path)
    transport = FakeTransport(
        extra_scope="inside" if remote_kind == "extra" else None,
        symlink_under_prefix=remote_kind == "symlink",
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


def test_sync_allows_unrelated_files_outside_content_addressed_prefix(
    tmp_path: Path,
) -> None:
    _experiment(tmp_path)
    staging_root = _staging_root(tmp_path)
    transport = FakeTransport(extra_scope="outside")

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
    assert all(entry.remotely_verified for entry in result.manifest.entries)


def test_sync_fails_closed_when_immutable_tree_listing_fails(tmp_path: Path) -> None:
    _experiment(tmp_path)
    staging_root = _staging_root(tmp_path)
    transport = FakeTransport(list_failure=True)

    result = sync_experiment(
        tmp_path,
        experiment_id="experiment-001",
        experiment_config_sha256=CONFIG_SHA,
        repository=DEFAULT_MODEL_REPO,
        revision="experiment-001",
        transport=transport,
        staging_root=staging_root,
        environ={"HF_TOKEN": TOKEN},
        max_attempts=1,
    )

    assert result.immutable_revision == "b" * 40
    assert result.disposable is False
    assert not any(entry.remotely_verified for entry in result.manifest.entries)
    assert transport.download_prefix is None
    assert list(staging_root.iterdir()) == []


def test_sync_result_file_round_trip_preserves_immutable_evidence(
    tmp_path: Path,
) -> None:
    _experiment(tmp_path)
    staging_root = _staging_root(tmp_path)
    result = sync_experiment(
        tmp_path,
        experiment_id="experiment-001",
        experiment_config_sha256=CONFIG_SHA,
        repository=DEFAULT_MODEL_REPO,
        revision="experiment-001",
        transport=FakeTransport(),
        staging_root=staging_root,
        environ={"HF_TOKEN": TOKEN},
    )
    path = tmp_path / "reports" / "sync-result.json"

    sync_module.write_sync_result(path, result)
    loaded = sync_module.load_sync_result(path)

    assert loaded == result
    assert loaded.immutable_revision == "b" * 40


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
