from __future__ import annotations

from pathlib import Path

import pytest

from lehome_train.constants import MODEL_REVISION
from lehome_train.groot.model_snapshot import (
    BASE_MODEL_REPOSITORY,
    ModelSnapshotFile,
    download_base_model,
)


class FakeModelTransport:
    def __init__(self) -> None:
        self.remote = {
            "config.json": b"{}",
            "processor/config.json": b"{}",
            "weights/model.safetensors": b"weights",
        }
        self.tokens: list[str | None] = []

    def list_files(self, *, repository: str, revision: str, token: str | None):
        self.tokens.append(token)
        return MODEL_REVISION, tuple(
            ModelSnapshotFile(path, len(content))
            for path, content in sorted(self.remote.items())
        )

    def download_snapshot(
        self,
        *,
        repository: str,
        revision: str,
        destination: Path,
        token: str | None,
    ) -> str:
        self.tokens.append(token)
        for path, content in self.remote.items():
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        return MODEL_REVISION


def test_base_model_retrieve_exposes_exact_complete_pinned_snapshot(tmp_path: Path) -> None:
    transport = FakeModelTransport()
    destination = tmp_path / "cache" / "model"
    staging = tmp_path / "cache"
    staging.mkdir()

    restored = download_base_model(
        destination,
        repository=BASE_MODEL_REPOSITORY,
        revision=MODEL_REVISION,
        transport=transport,
        staging_root=staging,
        environ={"HF_TOKEN": "hf_explicit_model_token"},
        free_space_probe=lambda _path: 10 * 1024**3,
    )

    assert restored == destination
    assert transport.tokens == ["hf_explicit_model_token", "hf_explicit_model_token"]
    assert (destination / "lehome_model_snapshot.json").is_file()
    assert {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file() and path.name != "lehome_model_snapshot.json"
    } == set(transport.remote)


@pytest.mark.parametrize(
    ("repository", "revision"),
    [("other/model", MODEL_REVISION), (BASE_MODEL_REPOSITORY, "main")],
)
def test_base_model_retrieve_rejects_mutable_or_unapproved_identity_before_network(
    tmp_path: Path, repository: str, revision: str
) -> None:
    transport = FakeModelTransport()
    staging = tmp_path / "cache"
    staging.mkdir()

    with pytest.raises(ValueError, match="pinned"):
        download_base_model(
            staging / "model",
            repository=repository,
            revision=revision,
            transport=transport,
            staging_root=staging,
            environ={},
            free_space_probe=lambda _path: 10 * 1024**3,
        )

    assert transport.tokens == []


def test_base_model_retrieve_rejects_incomplete_snapshot_without_destination(
    tmp_path: Path,
) -> None:
    transport = FakeModelTransport()
    staging = tmp_path / "cache"
    staging.mkdir()
    original = transport.download_snapshot

    def incomplete(**kwargs: object) -> str:
        revision = original(**kwargs)
        (kwargs["destination"] / "config.json").unlink()
        return revision

    transport.download_snapshot = incomplete  # type: ignore[method-assign]
    destination = staging / "model"

    with pytest.raises(ValueError, match="complete"):
        download_base_model(
            destination,
            repository=BASE_MODEL_REPOSITORY,
            revision=MODEL_REVISION,
            transport=transport,
            staging_root=staging,
            environ={},
            free_space_probe=lambda _path: 10 * 1024**3,
        )

    assert not destination.exists()
