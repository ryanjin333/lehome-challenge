from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import pytest

from lehome_train.constants import DEFAULT_DATA_REPO
from lehome_train.data.publish import (
    download_prepared_dataset,
    publish_prepared_dataset,
)
from lehome_train.hub import HubAccess
from lehome_train.io import atomic_write_json, canonical_json_sha256, sha256_file
from lehome_train.models import SyncEntry


class FakeHubTransport:
    def __init__(self) -> None:
        self.access = HubAccess(can_read=True, can_write=True)
        self.commit = "a" * 40
        self.remote: dict[str, bytes] = {}
        self.uploaded_entries: tuple[SyncEntry, ...] = ()
        self.download_revisions: list[str] = []
        self.corrupt_download_path: str | None = None
        self.before_upload: Callable[[], None] | None = None
        self.upload_sources: list[Path] = []
        self.download_destinations: list[Path] = []
        self.upload_source_exists_at_download: list[bool] = []
        self.write_unexpected_file = False
        self.write_unexpected_symlink = False

    def check_access(self, *, repository: str, token: str) -> HubAccess:
        return self.access

    def upload_files(
        self,
        *,
        repository: str,
        revision: str,
        source: Path,
        entries: tuple[SyncEntry, ...],
        token: str,
    ) -> str:
        self.upload_sources.append(source)
        if self.before_upload is not None:
            self.before_upload()
        self.uploaded_entries = entries
        self.remote = {
            entry.relative_path: (source / entry.relative_path).read_bytes()
            for entry in entries
        }
        return self.commit

    def download_files(
        self,
        *,
        repository: str,
        revision: str,
        destination: Path,
        relative_paths: tuple[str, ...],
        token: str,
    ) -> str:
        self.download_revisions.append(revision)
        self.download_destinations.append(destination)
        self.upload_source_exists_at_download.append(
            bool(self.upload_sources and self.upload_sources[-1].exists())
        )
        for relative_path in relative_paths:
            target = destination / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.remote[relative_path])
            if relative_path == self.corrupt_download_path:
                target.write_bytes(b"remote mismatch")
        if self.write_unexpected_file:
            (destination / "unexpected.bin").write_bytes(b"not allowlisted")
        if self.write_unexpected_symlink:
            unexpected_link = destination / "unexpected-link"
            if not unexpected_link.exists():
                os.symlink("manifest.json", unexpected_link)
        return revision


def _write_validated_dataset(root: Path) -> Path:
    dataset = root / "prepared"
    meta = dataset / "meta"
    meta.mkdir(parents=True)
    payload = dataset / "data" / "episode.bin"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"immutable episode")
    (meta / "stats.json").write_bytes(b'{"stats":"train-only"}')
    (meta / "relative_stats.json").write_bytes(b'{"stats":"relative"}')
    (meta / "lehome_groot_modality.py").write_bytes(b"MODALITY = 'joint-space'\n")

    output_artifacts = [
        {
            "relative_path": "data/episode.bin",
            "sha256": sha256_file(payload),
            "byte_size": payload.stat().st_size,
        }
    ]
    manifest = {
        "schema_version": 1,
        "output_artifacts": output_artifacts,
        "output_manifest_sha256": canonical_json_sha256(output_artifacts),
        "statistics": {
            "status": "computed_task_4_train_only",
            "files": [
                {
                    "relative_path": relative,
                    "sha256": sha256_file(dataset / relative),
                }
                for relative in (
                    "meta/lehome_groot_modality.py",
                    "meta/stats.json",
                    "meta/relative_stats.json",
                )
            ],
        },
    }
    atomic_write_json(dataset / "manifest.json", manifest)
    atomic_write_json(
        meta / "validation_report.json",
        {
            "schema_version": 1,
            "valid": True,
            "dataset_manifest_sha256": sha256_file(dataset / "manifest.json"),
        },
    )
    required = (
        "meta/lehome_groot_modality.py",
        "meta/relative_stats.json",
        "meta/stats.json",
        "meta/validation_report.json",
    )
    atomic_write_json(
        meta / "prepared_hashes.json",
        {
            "schema_version": 1,
            "artifacts": {
                relative: sha256_file(dataset / relative)
                for relative in required
            },
        },
    )
    return dataset


def test_publish_uses_only_the_validated_hash_allowlist_and_reads_back_commit(
    tmp_path: Path,
) -> None:
    dataset = _write_validated_dataset(tmp_path)
    (dataset / "unlisted-secret.txt").write_text("never upload me", encoding="utf-8")
    transport = FakeHubTransport()

    published = publish_prepared_dataset(
        dataset,
        repository=DEFAULT_DATA_REPO,
        revision="lehome-groot-n17-v1",
        transport=transport,
        environ={"HF_TOKEN": "hf_publish_process_token"},
    )

    expected_paths = {
        "data/episode.bin",
        "manifest.json",
        "meta/lehome_groot_modality.py",
        "meta/prepared_hashes.json",
        "meta/relative_stats.json",
        "meta/stats.json",
        "meta/validation_report.json",
    }
    assert {entry.relative_path for entry in transport.uploaded_entries} == expected_paths
    assert set(transport.remote) == expected_paths
    assert published.repository == DEFAULT_DATA_REPO
    assert published.revision == transport.commit
    assert published.dataset_manifest_sha256 == sha256_file(dataset / "manifest.json")
    assert transport.download_revisions == [transport.commit]


def test_publish_contract_includes_every_validator_required_artifact(
    tmp_path: Path,
) -> None:
    from lehome_train.data.validate import REQUIRED_VALIDATION_ARTIFACTS

    dataset = _write_validated_dataset(tmp_path)
    transport = FakeHubTransport()

    publish_prepared_dataset(
        dataset,
        repository=DEFAULT_DATA_REPO,
        revision="lehome-groot-n17-v1",
        transport=transport,
        environ={"HF_TOKEN": "hf_publish_process_token"},
    )

    published_paths = {
        entry.relative_path
        for entry in transport.uploaded_entries
    }
    assert set(REQUIRED_VALIDATION_ARTIFACTS) <= published_paths


def test_publish_checks_permission_before_creating_a_staging_tree(
    tmp_path: Path,
) -> None:
    dataset = _write_validated_dataset(tmp_path)
    transport = FakeHubTransport()
    transport.access = HubAccess(can_read=True, can_write=False)
    staging_root = tmp_path / "must-not-be-created"

    with pytest.raises(PermissionError, match="write"):
        publish_prepared_dataset(
            dataset,
            repository=DEFAULT_DATA_REPO,
            revision="lehome-groot-n17-v1",
            transport=transport,
            environ={"HF_TOKEN": "hf_publish_process_token"},
            staging_root=staging_root,
        )

    assert not staging_root.exists()
    assert transport.upload_sources == []


def test_publish_uses_the_caller_selected_staging_root(tmp_path: Path) -> None:
    dataset = _write_validated_dataset(tmp_path / "source")
    staging_root = tmp_path / "large-volume"
    staging_root.mkdir()
    transport = FakeHubTransport()

    publish_prepared_dataset(
        dataset,
        repository=DEFAULT_DATA_REPO,
        revision="lehome-groot-n17-v1",
        transport=transport,
        environ={"HF_TOKEN": "hf_publish_process_token"},
        staging_root=staging_root,
    )

    assert transport.upload_sources[0].parent == staging_root
    assert transport.upload_source_exists_at_download == [False]
    assert transport.download_destinations[0].parent == staging_root
    assert not tuple(staging_root.iterdir())


def test_publish_refuses_staging_filesystem_with_insufficient_free_space(
    tmp_path: Path,
) -> None:
    dataset = _write_validated_dataset(tmp_path / "source")
    staging_root = tmp_path / "small-volume"
    staging_root.mkdir()
    observed_roots: list[Path] = []
    transport = FakeHubTransport()
    payload_bytes = sum(
        path.stat().st_size
        for path in dataset.rglob("*")
        if path.is_file()
    )
    required_bytes = payload_bytes + (64 * 1024**2)

    with pytest.raises(ValueError, match="staging.*space") as error:
        publish_prepared_dataset(
            dataset,
            repository=DEFAULT_DATA_REPO,
            revision="lehome-groot-n17-v1",
            transport=transport,
            environ={"HF_TOKEN": "hf_publish_process_token"},
            staging_root=staging_root,
            free_space_probe=lambda path: (
                observed_roots.append(path)
                or required_bytes - 1
            ),
        )

    assert f"requires {required_bytes} bytes" in str(error.value)
    assert observed_roots == [staging_root]
    assert not tuple(staging_root.iterdir())
    assert transport.upload_sources == []


def test_publish_rechecks_space_after_upload_before_creating_readback(
    tmp_path: Path,
) -> None:
    dataset = _write_validated_dataset(tmp_path / "source")
    staging_root = tmp_path / "phase-volume"
    staging_root.mkdir()
    observed_roots: list[Path] = []
    transport = FakeHubTransport()

    def phase_free_space(path: Path) -> int:
        observed_roots.append(path)
        return 10**12 if len(observed_roots) == 1 else 0

    with pytest.raises(ValueError, match="readback.*space"):
        publish_prepared_dataset(
            dataset,
            repository=DEFAULT_DATA_REPO,
            revision="lehome-groot-n17-v1",
            transport=transport,
            environ={"HF_TOKEN": "hf_publish_process_token"},
            staging_root=staging_root,
            free_space_probe=phase_free_space,
        )

    assert observed_roots == [staging_root, staging_root]
    assert len(transport.upload_sources) == 1
    assert not transport.upload_sources[0].exists()
    assert transport.download_destinations == []
    assert not tuple(staging_root.iterdir())


def test_publish_refuses_dirty_hashed_payloads(tmp_path: Path) -> None:
    dataset = _write_validated_dataset(tmp_path)
    (dataset / "data" / "episode.bin").write_bytes(b"changed after validation")
    transport = FakeHubTransport()

    with pytest.raises(ValueError, match="dirty"):
        publish_prepared_dataset(
            dataset,
            repository=DEFAULT_DATA_REPO,
            revision="lehome-groot-n17-v1",
            transport=transport,
            environ={"HF_TOKEN": "hf_publish_process_token"},
        )

    assert transport.uploaded_entries == ()


def test_publish_refuses_unhashed_manifest_payloads(tmp_path: Path) -> None:
    dataset = _write_validated_dataset(tmp_path)
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_artifacts"][0].pop("sha256")
    manifest["output_manifest_sha256"] = canonical_json_sha256(
        manifest["output_artifacts"]
    )
    atomic_write_json(manifest_path, manifest)
    report_path = dataset / "meta" / "validation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["dataset_manifest_sha256"] = sha256_file(manifest_path)
    atomic_write_json(report_path, report)
    hashes_path = dataset / "meta" / "prepared_hashes.json"
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
    hashes["artifacts"]["meta/validation_report.json"] = sha256_file(report_path)
    atomic_write_json(hashes_path, hashes)

    with pytest.raises(ValueError, match="allowlist"):
        publish_prepared_dataset(
            dataset,
            repository=DEFAULT_DATA_REPO,
            revision="lehome-groot-n17-v1",
            transport=FakeHubTransport(),
            environ={"HF_TOKEN": "hf_publish_process_token"},
        )


def test_publish_fails_if_immutable_remote_readback_does_not_match(
    tmp_path: Path,
) -> None:
    dataset = _write_validated_dataset(tmp_path)
    transport = FakeHubTransport()
    transport.corrupt_download_path = "data/episode.bin"

    with pytest.raises(ValueError, match="remote.*hash"):
        publish_prepared_dataset(
            dataset,
            repository=DEFAULT_DATA_REPO,
            revision="lehome-groot-n17-v1",
            transport=transport,
            environ={"HF_TOKEN": "hf_publish_process_token"},
        )

    assert transport.download_revisions == [transport.commit]


def test_publish_uploads_a_stable_snapshot_if_original_changes_after_validation(
    tmp_path: Path,
) -> None:
    dataset = _write_validated_dataset(tmp_path)
    original_manifest_sha256 = sha256_file(dataset / "manifest.json")
    transport = FakeHubTransport()

    def mutate_original() -> None:
        (dataset / "data" / "episode.bin").write_bytes(b"concurrent live mutation")
        (dataset / "manifest.json").write_bytes(b"concurrent manifest mutation")

    transport.before_upload = mutate_original

    published = publish_prepared_dataset(
        dataset,
        repository=DEFAULT_DATA_REPO,
        revision="lehome-groot-n17-v1",
        transport=transport,
        environ={"HF_TOKEN": "hf_publish_process_token"},
    )

    assert transport.upload_sources != [dataset]
    assert transport.upload_sources[0].parent == dataset.parent
    assert transport.remote["data/episode.bin"] == b"immutable episode"
    assert sha256_file(dataset / "manifest.json") != original_manifest_sha256
    assert published.dataset_manifest_sha256 == original_manifest_sha256


def test_download_verifies_explicit_revision_before_atomically_completing(
    tmp_path: Path,
) -> None:
    source = _write_validated_dataset(tmp_path / "source")
    transport = FakeHubTransport()
    published = publish_prepared_dataset(
        source,
        repository=DEFAULT_DATA_REPO,
        revision="lehome-groot-n17-v1",
        transport=transport,
        environ={"HF_TOKEN": "hf_publish_process_token"},
    )
    destination = tmp_path / "downloaded"

    completed = download_prepared_dataset(
        destination,
        repository=DEFAULT_DATA_REPO,
        revision=published.revision,
        expected_manifest_sha256=published.dataset_manifest_sha256,
        transport=transport,
        environ={"HF_TOKEN": "hf_download_process_token"},
    )

    assert completed == destination
    assert sha256_file(destination / "manifest.json") == published.dataset_manifest_sha256
    assert {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    } == set(transport.remote)
    assert all(revision == published.revision for revision in transport.download_revisions)


def test_remote_hash_mismatch_leaves_dataset_incomplete_and_non_trainable(
    tmp_path: Path,
) -> None:
    source = _write_validated_dataset(tmp_path / "source")
    transport = FakeHubTransport()
    published = publish_prepared_dataset(
        source,
        repository=DEFAULT_DATA_REPO,
        revision="lehome-groot-n17-v1",
        transport=transport,
        environ={"HF_TOKEN": "hf_publish_process_token"},
    )
    transport.corrupt_download_path = "data/episode.bin"
    destination = tmp_path / "downloaded"

    with pytest.raises(ValueError, match="dirty|hash"):
        download_prepared_dataset(
            destination,
            repository=DEFAULT_DATA_REPO,
            revision=published.revision,
            expected_manifest_sha256=published.dataset_manifest_sha256,
            transport=transport,
            environ={"HF_TOKEN": "hf_download_process_token"},
        )

    assert not destination.exists()
    assert not tuple(tmp_path.glob(".downloaded.*.incomplete"))


def test_download_rejects_an_unexpected_remote_file_before_completion(
    tmp_path: Path,
) -> None:
    source = _write_validated_dataset(tmp_path / "source")
    transport = FakeHubTransport()
    published = publish_prepared_dataset(
        source,
        repository=DEFAULT_DATA_REPO,
        revision="lehome-groot-n17-v1",
        transport=transport,
        environ={"HF_TOKEN": "hf_publish_process_token"},
    )
    transport.write_unexpected_file = True
    destination = tmp_path / "downloaded"

    with pytest.raises(ValueError, match="unexpected"):
        download_prepared_dataset(
            destination,
            repository=DEFAULT_DATA_REPO,
            revision=published.revision,
            expected_manifest_sha256=published.dataset_manifest_sha256,
            transport=transport,
            environ={"HF_TOKEN": "hf_download_process_token"},
        )

    assert not destination.exists()


def test_download_rejects_a_remote_symlink_before_completion(
    tmp_path: Path,
) -> None:
    source = _write_validated_dataset(tmp_path / "source")
    transport = FakeHubTransport()
    published = publish_prepared_dataset(
        source,
        repository=DEFAULT_DATA_REPO,
        revision="lehome-groot-n17-v1",
        transport=transport,
        environ={"HF_TOKEN": "hf_publish_process_token"},
    )
    transport.write_unexpected_symlink = True
    destination = tmp_path / "downloaded"

    with pytest.raises(ValueError, match="symlink"):
        download_prepared_dataset(
            destination,
            repository=DEFAULT_DATA_REPO,
            revision=published.revision,
            expected_manifest_sha256=published.dataset_manifest_sha256,
            transport=transport,
            environ={"HF_TOKEN": "hf_download_process_token"},
        )

    assert not destination.exists()
