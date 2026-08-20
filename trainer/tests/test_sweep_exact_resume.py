"""CPU evidence for authenticated promoted-rung continuation semantics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest


def _checkpoint(root: Path, step: int) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"promoted-policy")
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
    (checkpoint / "rng_state.pth").write_bytes(b"rng")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "log_history": [{"step": step, "loss": 0.1}]}),
        encoding="utf-8",
    )
    return checkpoint


def _config(tmp_path: Path, *, parent_checkpoint: Path, parent_sha: str, target_step: int):
    from lehome_train.constants import MODEL_REVISION
    from lehome_train.groot.config import FineTuneLaunchConfig
    from lehome_train.groot.experiment_manifest import SweepRuntimeProfile

    return FineTuneLaunchConfig(
        base_model_path=str(parent_checkpoint), base_model_revision=MODEL_REVISION,
        dataset_path="/prepared/runtime", dataset_revision="a" * 40,
        modality_config_path="/prepared/modality.py", output_dir=str(tmp_path / "output"),
        experiment_name="child", physical_batch_size=64, global_batch_size=64,
        max_steps=target_step, save_steps=500, warmup_ratio=0.05,
        parent_checkpoint_repository="owner/models", parent_checkpoint_revision="b" * 40,
        parent_checkpoint_subpath="experiments/parent", parent_checkpoint_artifact_sha256=parent_sha,
        runtime_mixture_manifest="/prepared/runtime/mixture.json",
        runtime_window_index="/prepared/runtime/windows.json",
        runtime_mounts_descriptor="/prepared/runtime/mounts.json",
        runtime_sweep_profile=SweepRuntimeProfile(
            weights={"bc": 95, "rollout": 5, "dagger": 0},
            quotas={"bc": 61, "rollout": 3, "dagger": 0}, target_step=target_step,
            save_steps=500, terminal_publish=True, action_horizon=16, global_batch_size=64,
        ),
    )


def _binding(*, step: int, archive: Path, descriptor: Path) -> dict[str, object]:
    publication = {
        "schema_version": 2, "experiment_id": "p" * 64, "job_digest": "p" * 64,
        "target_step": step, "repository": "owner/models", "immutable_revision": "a" * 40,
        "remote_prefix": "experiments/parent", "relative_path": "checkpoints/parent.tar",
        "artifact_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "artifact_byte_size": archive.stat().st_size,
        "descriptor_relative_path": "checkpoints/parent.json",
        "descriptor_sha256": hashlib.sha256(descriptor.read_bytes()).hexdigest(),
        "descriptor_byte_size": descriptor.stat().st_size,
        "receipt_sha256": "b" * 64, "readback_verified": True,
    }
    return {
        "parent_publication": publication,
        "parent_cursor": {
            "optimizer_step": step, "global_sample_offset": step * 64,
            "physical_batch_size": 64, "action_horizon": 16,
        },
    }


def test_promoted_sweep_stages_the_immutable_full_state_and_reuses_local_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lehome_train.groot.production_runtime as runtime
    from lehome_train.groot.checkpoint_identity import policy_artifact_sha256

    source_root = tmp_path / "source" / "parent-run"
    source = _checkpoint(source_root, 500)
    archive = tmp_path / "parent.tar"
    with tarfile.open(archive, "w") as bundle:
        bundle.add(source_root, arcname="parent-run")
    descriptor = tmp_path / "parent.json"
    descriptor.write_bytes(b"parent descriptor")
    cache_parent = tmp_path / "cache" / "parent"
    cache_parent.mkdir(parents=True)
    (cache_parent / "model.safetensors").write_bytes(b"promoted-policy")
    config = _config(
        tmp_path, parent_checkpoint=cache_parent,
        parent_sha=policy_artifact_sha256(cache_parent), target_step=1000,
    )
    binding = _binding(step=500, archive=archive, descriptor=descriptor)
    record = SimpleNamespace(
        optimizer_step=500, resumable=True,
        artifact=SimpleNamespace(
            sha256=binding["parent_publication"]["artifact_sha256"],
            byte_size=archive.stat().st_size,
        ),
    )
    monkeypatch.setattr(runtime, "load_checkpoint_descriptor", lambda _path: SimpleNamespace(record=record))
    monkeypatch.setattr(runtime, "_mounted_path", lambda value, *_args, **_kwargs: Path(value))
    monkeypatch.setenv("LEHOME_SWEEP_PARENT_ARCHIVE", str(archive))
    monkeypatch.setenv("LEHOME_SWEEP_PARENT_DESCRIPTOR", str(descriptor))
    monkeypatch.setenv("LEHOME_SWEEP_PARENT_CHECKPOINT", str(cache_parent))

    staged = runtime._stage_promoted_sweep_resume(config=config, binding=binding)

    assert staged.name == "checkpoint-500"
    assert (staged / "optimizer.pt").read_bytes() == b"optimizer"
    assert json.loads((staged / "trainer_state.json").read_text())["global_step"] == 500
    assert runtime._select_sweep_resume_checkpoint(config=config, binding=binding) == staged

    local = _checkpoint(Path(config.output_dir) / config.experiment_name, 1000)
    assert runtime._select_sweep_resume_checkpoint(config=config, binding=binding) == local


def test_promoted_sweep_rejects_model_only_reset_without_immutable_resume_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lehome_train.groot.production_runtime as runtime
    from lehome_train.groot.checkpoint_identity import policy_artifact_sha256

    parent = tmp_path / "cache" / "parent"
    parent.mkdir(parents=True)
    (parent / "model.safetensors").write_bytes(b"model-only")
    config = _config(
        tmp_path, parent_checkpoint=parent,
        parent_sha=policy_artifact_sha256(parent), target_step=1000,
    )
    # The exact publication shape is irrelevant here: a parent without the
    # archive+descriptor environment cannot reach the launcher at all.
    binding = {
        "parent_publication": {"target_step": 500},
        "parent_cursor": {
            "optimizer_step": 500, "global_sample_offset": 500 * 64,
            "physical_batch_size": 64, "action_horizon": 16,
        },
    }
    monkeypatch.delenv("LEHOME_SWEEP_PARENT_ARCHIVE", raising=False)
    monkeypatch.delenv("LEHOME_SWEEP_PARENT_DESCRIPTOR", raising=False)
    monkeypatch.delenv("LEHOME_SWEEP_PARENT_CHECKPOINT", raising=False)

    with pytest.raises(ValueError, match="full-state resume inputs"):
        runtime._select_sweep_resume_checkpoint(config=config, binding=binding)
