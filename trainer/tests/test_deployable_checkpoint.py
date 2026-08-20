from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import pytest

from lehome_train.groot.deployable_checkpoint import (
    build_deployable_checkpoint,
    publish_deployable_checkpoint,
)
from lehome_train.hub import HubAccess, HubTreeEntry


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _checkpoint(root: Path) -> Path:
    checkpoint = root / "checkpoint-2000"
    _write(checkpoint / "config.json", b"{}\n")
    _write(checkpoint / "processor_config.json", b"{}\n")
    _write(checkpoint / "embodiment_id.json", b"{}\n")
    _write(checkpoint / "statistics.json", b"{}\n")
    _write(checkpoint / "model-00001-of-00002.safetensors", b"model-one")
    _write(checkpoint / "model-00002-of-00002.safetensors", b"model-two")
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 18},
                "weight_map": {
                    "layer.0": "model-00001-of-00002.safetensors",
                    "layer.1": "model-00002-of-00002.safetensors",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    for name in (
        "config.yaml",
        "conf.yaml",
        "dataset_statistics.json",
        "final_model_config.json",
        "final_processor_config.json",
    ):
        _write(checkpoint / "experiment_cfg" / name, b"{}\n")
    # Full local recovery state must never enter the deployable publication.
    _write(checkpoint / "optimizer.pt", b"private-optimizer")
    _write(checkpoint / "scheduler.pt", b"private-scheduler")
    _write(checkpoint / "rng_state.pth", b"private-rng")
    _write(checkpoint / "trainer_state.json", b'{"global_step":2000}\n')
    _write(checkpoint / "training_args.bin", b"private-training-args")
    return checkpoint


def test_build_deployable_checkpoint_is_exact_model_only_hardlink_tree(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path / "source")

    bundle = build_deployable_checkpoint(
        source,
        staging_root=tmp_path / "staging",
        experiment_id="lehome-awr-2k-20260818-v1",
        optimizer_step=2000,
    )

    expected = {
        "config.json",
        "processor_config.json",
        "embodiment_id.json",
        "statistics.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "experiment_cfg/config.yaml",
        "experiment_cfg/conf.yaml",
        "experiment_cfg/dataset_statistics.json",
        "experiment_cfg/final_model_config.json",
        "experiment_cfg/final_processor_config.json",
        "deployable-manifest.json",
    }
    observed = {
        path.relative_to(bundle.payload_root).as_posix()
        for path in bundle.payload_root.rglob("*")
        if path.is_file()
    }
    assert observed == expected
    assert os.stat(source / "model-00001-of-00002.safetensors").st_ino == os.stat(
        bundle.payload_root / "model-00001-of-00002.safetensors"
    ).st_ino
    assert not any("optimizer" in entry.relative_path for entry in bundle.entries)
    assert not any("trainer_state" in entry.relative_path for entry in bundle.entries)
    assert bundle.byte_size == sum(entry.byte_size for entry in bundle.entries)


@pytest.mark.parametrize(
    "mutation",
    ("extra-shard", "monolithic-model", "symlink", "wrong-step"),
)
def test_build_deployable_checkpoint_rejects_ambiguous_or_unsafe_sources(
    tmp_path: Path, mutation: str
) -> None:
    source = _checkpoint(tmp_path / "source")
    optimizer_step = 2000
    if mutation == "extra-shard":
        _write(source / "model-00003-of-00003.safetensors", b"unindexed")
    elif mutation == "monolithic-model":
        _write(source / "model.safetensors", b"ambiguous")
    elif mutation == "symlink":
        (source / "statistics.json").unlink()
        (source / "statistics.json").symlink_to(source / "config.json")
    else:
        optimizer_step = 1000

    with pytest.raises(ValueError):
        build_deployable_checkpoint(
            source,
            staging_root=tmp_path / "staging",
            experiment_id="lehome-awr-2k-20260818-v1",
            optimizer_step=optimizer_step,
        )


class _MemoryHub:
    def __init__(self, remote: Path) -> None:
        self.remote = remote
        self.revision = "a" * 40

    def check_access(self, *, repository: str, token: str) -> HubAccess:
        assert token == "temporary-write-token"
        return HubAccess(can_read=True, can_write=True, private_repository=True)

    def upload_large_folder(
        self,
        *,
        repository: str,
        revision: str,
        source: Path,
        entries: tuple,
        remote_prefix: str,
        token: str,
        max_workers: int,
    ) -> None:
        del repository, revision, remote_prefix, token, max_workers
        for entry in entries:
            target = self.remote / entry.relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / entry.relative_path, target)

    def resolve_approved_ref(self, *, repository: str, ref: str, token: str) -> str:
        del repository, ref, token
        return self.revision

    def list_tree(self, *, repository: str, revision: str, token: str, remote_prefix=None):
        del repository, revision, token, remote_prefix
        return tuple(
            HubTreeEntry(path.relative_to(self.remote).as_posix(), "file")
            for path in self.remote.rglob("*")
            if path.is_file()
        )

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
        del repository, token
        assert remote_prefix is not None
        for relative in relative_paths:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.remote / remote_prefix / relative, target)
        return revision


def test_publish_deployable_checkpoint_requires_immutable_full_hash_readback(
    tmp_path: Path,
) -> None:
    source = _checkpoint(tmp_path / "source")
    bundle = build_deployable_checkpoint(
        source,
        staging_root=tmp_path / "staging",
        experiment_id="lehome-awr-2k-20260818-v1",
        optimizer_step=2000,
    )
    receipt_path = tmp_path / "receipts" / "deployable-2000.json"

    receipt = publish_deployable_checkpoint(
        bundle,
        repository="ryanjin333/lehome-groot-n17-models",
        revision="main",
        token="temporary-write-token",
        transport=_MemoryHub(tmp_path / "remote"),
        readback_root=tmp_path / "readback",
        receipt_path=receipt_path,
    )

    assert receipt["optimizer_step"] == 2000
    assert receipt["immutable_revision"] == "a" * 40
    assert receipt["private_repository"] is True
    assert receipt["readback_verified"] is True
    assert receipt["bundle_sha256"] == bundle.sha256
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt
    assert not (tmp_path / "readback").exists()
