from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from b1k_rollout.checkpoint import FinalPolicyMaterializer, MaterializationError
from b1k_rollout.controller import CheckpointReceipt
from b1k_rollout.identity import MODEL_REPO


_COMMIT = "a" * 40


class FakeModelHub:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = dict(files)
        self.calls: list[tuple[str, str, str]] = []

    def open_file(self, repository: str, *, revision: str, path: str) -> io.BytesIO:
        self.calls.append((repository, revision, path))
        try:
            return io.BytesIO(self.files[path])
        except KeyError as error:
            raise OSError("requested immutable file is absent") from error


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest(*, files: dict[str, bytes], extra_files: dict[str, bytes] | None = None) -> bytes:
    entries = {
        path: {"byte_size": len(contents), "sha256": _sha256(contents)}
        for path, contents in files.items()
    }
    entries.update(
        {
            path: {"byte_size": len(contents), "sha256": _sha256(contents)}
            for path, contents in (extra_files or {}).items()
        }
    )
    return json.dumps({"schema_version": 1, "run_id": "run-001", "files": entries}, sort_keys=True).encode()


def _materializer(hub: FakeModelHub, manifest: bytes) -> FinalPolicyMaterializer:
    return FinalPolicyMaterializer(hub=hub, expected_manifest_sha256=_sha256(manifest))


def test_materializer_promotes_only_the_verified_checkpoint_subtree(tmp_path: Path) -> None:
    checkpoint_files = {
        "checkpoint/config.json": b'{"action_horizon":40}\n',
        "checkpoint/nested/weights.bin": b"immutable weights",
    }
    excluded_files = {
        "evidence/training.log": b"not part of the serving checkpoint",
        ".gitattributes": b"*.bin filter=lfs\n",
    }
    manifest = _manifest(files=checkpoint_files, extra_files=excluded_files)
    hub = FakeModelHub({"final-manifest.json": manifest, **checkpoint_files, **excluded_files})
    destination = tmp_path / "checkpoint"

    receipt = _materializer(hub, manifest).download(
        repository=MODEL_REPO, revision=_COMMIT, destination=destination
    )

    assert receipt == CheckpointReceipt(_COMMIT, _sha256(manifest), destination.resolve())
    assert (destination / "config.json").read_bytes() == checkpoint_files["checkpoint/config.json"]
    assert (destination / "nested" / "weights.bin").read_bytes() == checkpoint_files["checkpoint/nested/weights.bin"]
    assert not (destination / "evidence").exists()
    assert not (destination / ".gitattributes").exists()
    assert (destination / ".b1k-final-policy.json").is_file()
    assert [path for _, _, path in hub.calls] == [
        "final-manifest.json",
        "checkpoint/config.json",
        "checkpoint/nested/weights.bin",
    ]


def test_materializer_rejects_a_manifest_whose_bytes_do_not_match_the_contract(tmp_path: Path) -> None:
    contents = {"checkpoint/weights.bin": b"immutable weights"}
    manifest = _manifest(files=contents)
    hub = FakeModelHub({"final-manifest.json": manifest, **contents})
    materializer = FinalPolicyMaterializer(hub=hub, expected_manifest_sha256="b" * 64)

    with pytest.raises(MaterializationError, match="manifest.*contract"):
        materializer.download(repository=MODEL_REPO, revision=_COMMIT, destination=tmp_path / "checkpoint")

    assert not (tmp_path / "checkpoint").exists()
    assert hub.calls == [(MODEL_REPO, _COMMIT, "final-manifest.json")]


def test_materializer_redacts_an_unexpected_remote_manifest_failure(tmp_path: Path) -> None:
    class FailingHub:
        def open_file(self, repository: str, *, revision: str, path: str) -> io.BytesIO:
            raise RuntimeError("provider detail must not become an operator-facing error")

    with pytest.raises(MaterializationError, match="final manifest download failed"):
        FinalPolicyMaterializer(hub=FailingHub(), expected_manifest_sha256="a" * 64).download(
            repository=MODEL_REPO, revision=_COMMIT, destination=tmp_path / "checkpoint"
        )


@pytest.mark.parametrize(
    "manifest",
    [
        b'{"schema_version":1,"run_id":"run-001","files":{"checkpoint/../escape":{"byte_size":1,"sha256":"' + b"a" * 64 + b'"}}}',
        b'{"schema_version":1,"run_id":"run-001","files":{"checkpoint/weights.bin":{"byte_size":1,"sha256":"' + b"a" * 64 + b'","extra":true}}}',
        b'{"schema_version":1,"run_id":"run-001","files":{"checkpoint/weights.bin":{"byte_size":1,"sha256":"' + b"a" * 64 + b'"}},"unexpected":true}',
        b'{"schema_version":1,"run_id":"run-001","run_id":"run-002","files":{"checkpoint/weights.bin":{"byte_size":1,"sha256":"' + b"a" * 64 + b'"}}}',
    ],
)
def test_materializer_rejects_traversal_duplicate_and_unknown_manifest_shape(
    tmp_path: Path, manifest: bytes
) -> None:
    hub = FakeModelHub({"final-manifest.json": manifest})

    with pytest.raises(MaterializationError, match="manifest"):
        _materializer(hub, manifest).download(
            repository=MODEL_REPO, revision=_COMMIT, destination=tmp_path / "checkpoint"
        )

    assert not (tmp_path / "checkpoint").exists()


def test_materializer_fails_closed_on_a_checkpoint_file_size_or_hash_mismatch(tmp_path: Path) -> None:
    expected = b"immutable weights"
    manifest = _manifest(files={"checkpoint/weights.bin": expected})
    hub = FakeModelHub({"final-manifest.json": manifest, "checkpoint/weights.bin": b"tampered"})

    with pytest.raises(MaterializationError, match="byte size|SHA-256"):
        _materializer(hub, manifest).download(
            repository=MODEL_REPO, revision=_COMMIT, destination=tmp_path / "checkpoint"
        )

    assert not (tmp_path / "checkpoint").exists()
    assert not list(tmp_path.glob(".checkpoint.incomplete-*"))


def test_materializer_reuses_only_a_complete_marked_tree_and_readback_refetches_manifest(
    tmp_path: Path,
) -> None:
    files = {"checkpoint/weights.bin": b"immutable weights"}
    manifest = _manifest(files=files)
    hub = FakeModelHub({"final-manifest.json": manifest, **files})
    materializer = _materializer(hub, manifest)
    destination = tmp_path / "checkpoint"

    first = materializer.download(repository=MODEL_REPO, revision=_COMMIT, destination=destination)
    calls_after_first = list(hub.calls)
    resumed = materializer.download(repository=MODEL_REPO, revision=_COMMIT, destination=destination)
    readback = materializer.readback(repository=MODEL_REPO, revision=_COMMIT, destination=destination)

    assert resumed == first == readback
    assert hub.calls == calls_after_first + [(MODEL_REPO, _COMMIT, "final-manifest.json")]
    (destination / "unexpected.bin").write_bytes(b"not declared")
    with pytest.raises(MaterializationError, match="unexpected"):
        materializer.readback(repository=MODEL_REPO, revision=_COMMIT, destination=destination)


def test_materializer_rejects_an_empty_undeclared_directory_in_a_reused_tree(tmp_path: Path) -> None:
    files = {"checkpoint/nested/weights.bin": b"immutable weights"}
    manifest = _manifest(files=files)
    hub = FakeModelHub({"final-manifest.json": manifest, **files})
    materializer = _materializer(hub, manifest)
    destination = tmp_path / "checkpoint"
    materializer.download(repository=MODEL_REPO, revision=_COMMIT, destination=destination)
    (destination / "evidence").mkdir()

    with pytest.raises(MaterializationError, match="unexpected directory"):
        materializer.readback(repository=MODEL_REPO, revision=_COMMIT, destination=destination)


def test_materializer_rejects_a_symlink_in_an_existing_checkpoint_tree(tmp_path: Path) -> None:
    files = {"checkpoint/weights.bin": b"immutable weights"}
    manifest = _manifest(files=files)
    hub = FakeModelHub({"final-manifest.json": manifest, **files})
    materializer = _materializer(hub, manifest)
    destination = tmp_path / "checkpoint"
    materializer.download(repository=MODEL_REPO, revision=_COMMIT, destination=destination)
    (destination / "link").symlink_to(destination / "weights.bin")

    with pytest.raises(MaterializationError, match="symlink"):
        materializer.download(repository=MODEL_REPO, revision=_COMMIT, destination=destination)


def test_materializer_discards_only_its_incomplete_sibling_staging_directories(tmp_path: Path) -> None:
    files = {"checkpoint/weights.bin": b"immutable weights"}
    manifest = _manifest(files=files)
    hub = FakeModelHub({"final-manifest.json": manifest, **files})
    stale = tmp_path / ".checkpoint.incomplete-interrupted"
    stale.mkdir()
    (stale / "partial").write_bytes(b"partial")
    unrelated = tmp_path / ".other.incomplete-interrupted"
    unrelated.mkdir()

    _materializer(hub, manifest).download(
        repository=MODEL_REPO, revision=_COMMIT, destination=tmp_path / "checkpoint"
    )

    assert not stale.exists()
    assert unrelated.is_dir()


def test_materializer_creates_a_missing_non_symlink_destination_parent(tmp_path: Path) -> None:
    files = {"checkpoint/weights.bin": b"immutable weights"}
    manifest = _manifest(files=files)
    hub = FakeModelHub({"final-manifest.json": manifest, **files})
    destination = tmp_path / "campaign" / "checkpoint"

    receipt = _materializer(hub, manifest).download(
        repository=MODEL_REPO, revision=_COMMIT, destination=destination
    )

    assert receipt.local_path == destination
    assert (destination / "weights.bin").read_bytes() == b"immutable weights"
