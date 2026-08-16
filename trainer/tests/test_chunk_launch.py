from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import lehome_train.groot.chunk_launch as chunk_launch
from lehome_train.groot.chunk_launch import StopAtOptimizerStep, _resume_value


def test_stop_callback_stops_only_at_requested_global_step() -> None:
    callback = StopAtOptimizerStep(1_000)
    control = SimpleNamespace(
        should_training_stop=False,
        should_save=False,
        should_log=False,
    )

    returned = callback.on_step_end(
        args=object(),
        state=SimpleNamespace(global_step=999),
        control=control,
    )
    assert returned.should_training_stop is False

    returned = callback.on_step_end(
        args=object(),
        state=SimpleNamespace(global_step=1_000),
        control=control,
    )
    assert returned.should_training_stop is True
    assert returned.should_save is True
    assert returned.should_log is True


def test_stop_callback_accepts_transformers_train_begin_hook() -> None:
    callback = StopAtOptimizerStep(1_000)
    control = SimpleNamespace()

    assert callback.on_train_begin(
        args=object(),
        state=SimpleNamespace(),
        control=control,
    ) is control


def test_fresh_run_disables_upstream_unconditional_resume(tmp_path) -> None:
    assert _resume_value(tmp_path) is False


def test_existing_checkpoint_preserves_real_resume(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-1000"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 1000}), encoding="utf-8"
    )

    assert _resume_value(tmp_path) is True


def test_distributed_resume_requires_complete_zero2_shards(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 100}), encoding="utf-8"
    )
    shard_root = checkpoint / "global_step100"
    shard_root.mkdir()
    (shard_root / "mp_rank_00_model_states.pt").write_bytes(b"model")
    for rank in range(4):
        (
            shard_root / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
        ).write_bytes(b"optimizer")

    assert _resume_value(tmp_path, num_gpus=4) is True
    (shard_root / "mp_rank_00_model_states.pt").unlink()
    with __import__("pytest").raises(ValueError, match="incomplete ZeRO-2"):
        _resume_value(tmp_path, num_gpus=4)


def test_distributed_rank_validation_requires_one_rank_per_visible_gpu() -> None:
    assert chunk_launch._validate_distributed_rank_device(
        world_size="4", local_rank="3", rank="3", device_count=4, num_gpus=4
    ) == 3

    with __import__("pytest").raises(ValueError, match="world size"):
        chunk_launch._validate_distributed_rank_device(
            world_size="3", local_rank="0", rank="0", device_count=4, num_gpus=4
        )
    with __import__("pytest").raises(ValueError, match="local rank"):
        chunk_launch._validate_distributed_rank_device(
            world_size="4", local_rank="4", rank="0", device_count=4, num_gpus=4
        )
    with __import__("pytest").raises(ValueError, match="visible CUDA GPUs"):
        chunk_launch._validate_distributed_rank_device(
            world_size="4", local_rank="0", rank="0", device_count=3, num_gpus=4
        )


def test_follower_rank_uses_private_metadata_root_then_restores_canonical_checkpoint_root(
    tmp_path: Path,
) -> None:
    arguments = [
        "--output-dir",
        str(tmp_path / "output"),
        "--experiment-name",
        "run",
    ]
    rewritten, canonical, staging = chunk_launch._rank_metadata_staging(
        arguments,
        num_gpus=4,
        environment={"WORLD_SIZE": "4", "LOCAL_RANK": "2", "RANK": "2"},
    )

    assert rewritten[rewritten.index("--output-dir") + 1] == str(
        tmp_path / "output" / ".lehome-rank-metadata-2"
    )
    assert canonical == tmp_path / "output" / "run"
    assert staging == tmp_path / "output" / ".lehome-rank-metadata-2"

    calls: list[str] = []
    trainer = SimpleNamespace(args=SimpleNamespace(output_dir="private/run"))
    chunk_launch._restore_canonical_trainer_output(
        trainer,
        canonical_run=canonical,
        synchronize=lambda: calls.append("barrier"),
    )
    assert calls == ["barrier"]
    assert trainer.args.output_dir == str(canonical)


def test_rank_zero_and_single_gpu_keep_canonical_metadata_arguments(tmp_path: Path) -> None:
    arguments = [
        "--output-dir",
        str(tmp_path / "output"),
        "--experiment-name",
        "run",
    ]
    rank_zero, canonical, staging = chunk_launch._rank_metadata_staging(
        arguments,
        num_gpus=4,
        environment={"WORLD_SIZE": "4", "LOCAL_RANK": "0", "RANK": "0"},
    )
    single, single_canonical, single_staging = chunk_launch._rank_metadata_staging(
        arguments, num_gpus=1, environment={}
    )

    assert rank_zero == arguments
    assert canonical == tmp_path / "output" / "run"
    assert staging is None
    assert single == arguments
    assert single_canonical is None
    assert single_staging is None


def test_metadata_staging_cleanup_runs_after_follower_error_path(tmp_path: Path) -> None:
    staging = tmp_path / ".lehome-rank-metadata-3"
    staging.mkdir()
    (staging / "processor.json").write_text("{}", encoding="utf-8")

    chunk_launch._cleanup_metadata_staging(staging)

    assert not staging.exists()


def test_zero_step_wrapper_initializes_and_saves_without_entering_training(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "run"
    original_calls: list[bool] = []
    saved: list[Path] = []

    class FakeTrainer:
        def __init__(self) -> None:
            self.args = SimpleNamespace(output_dir=str(output))

        def add_callback(self, _callback: object) -> None:
            raise AssertionError("zero-step initialization must not install a train callback")

        def train(self, *, resume_from_checkpoint: bool) -> None:
            original_calls.append(resume_from_checkpoint)
            checkpoint = output / "checkpoint-1"
            checkpoint.mkdir(parents=True)

        def save_model(self) -> None:
            output.mkdir(parents=True, exist_ok=True)
            model = output / "initialized-model.bin"
            model.write_bytes(b"initialized")
            saved.append(model)

    def fake_run_path(_entrypoint: str, *, run_name: str) -> None:
        assert run_name == "__main__"
        trainer = FakeTrainer()
        trainer.train(resume_from_checkpoint=True)
        trainer.save_model()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(Trainer=FakeTrainer))
    monkeypatch.setattr(chunk_launch.runpy, "run_path", fake_run_path)

    chunk_launch.main(
        [
            "--stop-after-step",
            "0",
            "--",
            str(tmp_path / "gr00t" / "experiment" / "launch_finetune.py"),
        ]
    )

    assert original_calls == []
    assert saved == [output / "initialized-model.bin"]
    assert not (output / "checkpoint-1").exists()


def test_chunk_runtime_wrapper_keeps_guarded_arguments_and_runs_entrypoint_in_process(
    tmp_path: Path, monkeypatch
) -> None:
    """The chunk guard must not bypass the DatasetFactory substitution."""
    captured: list[list[str]] = []

    class FakeTrainer:
        def add_callback(self, _callback: object) -> None:
            pass

        def train(self, **_kwargs: object) -> None:
            pass

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(Trainer=FakeTrainer))
    from lehome_train.groot import runtime_mixture_entrypoint

    monkeypatch.setattr(runtime_mixture_entrypoint, "main", lambda argv: captured.append(list(argv)) or 0)
    wrapper = [
        "-m", "lehome_train.groot.runtime_mixture_entrypoint",
        "--mixture-manifest", "/runtime/mixture.json",
        "--window-index", "/runtime/windows.json",
        "--mounts-descriptor", "/runtime/mounts.json",
        "--official-launch", str(tmp_path / "gr00t" / "experiment" / "launch_finetune.py"),
        "--", "--output-dir", "/output", "--experiment-name", "run", "--num-gpus", "1", "--global-batch-size", "64",
    ]

    chunk_launch.main(["--stop-after-step", "1", "--", *wrapper])

    separator = wrapper.index("--")
    assert captured == [[
        *wrapper[2:separator],
        "--resume-sample-offset", "0", "--resume-global-step", "0", "--global-batch-size", "64",
        "--", *wrapper[separator + 1:],
    ]]


def test_runtime_chunk_authenticates_checkpoint_step_before_resetting_dataset_seed(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "output" / "run"
    checkpoint = output / "checkpoint-10"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text('{"global_step":10}', encoding="utf-8")
    seen: list[object] = []

    class Dataset:
        seed = 17
        def reset_seed(self, _new_seed: int) -> None: seen.append("reset")

    class FakeTrainer:
        def __init__(self) -> None:
            self.args = SimpleNamespace(output_dir=str(output))
            self.train_dataset = Dataset()

        def add_callback(self, _callback: object) -> None: pass
        def train(self, **_kwargs: object) -> None: seen.append("train")

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(Trainer=FakeTrainer))
    from lehome_train.groot import runtime_mixture_entrypoint
    monkeypatch.setattr(runtime_mixture_entrypoint, "main", lambda _argv: FakeTrainer().train(resume_from_checkpoint=True) or 0)
    wrapper = [
        "-m", "lehome_train.groot.runtime_mixture_entrypoint",
        "--mixture-manifest", "/runtime/mixture.json", "--window-index", "/runtime/windows.json",
        "--mounts-descriptor", "/runtime/mounts.json",
        "--official-launch", str(tmp_path / "gr00t" / "experiment" / "launch_finetune.py"),
        "--", "--output-dir", str(tmp_path / "output"), "--experiment-name", "run", "--num-gpus", "1", "--global-batch-size", "64",
    ]

    chunk_launch.main(["--stop-after-step", "11", "--", *wrapper])

    assert seen == ["train"]


def test_runtime_checkpoint_binding_rejects_a_resume_path_outside_selected_safe_roots(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "untrusted" / "run" / "checkpoint-1000"
    outside.mkdir(parents=True)
    (outside / "trainer_state.json").write_text('{"global_step":1000}', encoding="utf-8")
    official = [
        "--output-dir", str(tmp_path / "output"), "--experiment-name", "run",
        "--num-gpus", "1", "--global-batch-size", "64",
        "--resume-from-checkpoint", str(outside),
    ]

    with pytest.raises(ValueError, match="authenticated runtime resume checkpoint"):
        chunk_launch._runtime_checkpoint_binding(["--"], official, num_gpus=1)


@pytest.mark.parametrize("kind", ("canonical", "staging", "output"))
def test_runtime_checkpoint_binding_rejects_symlinked_resume_ancestry(
    tmp_path: Path, kind: str,
) -> None:
    output = tmp_path / "output"
    if kind == "output":
        external_output = tmp_path / "external-output"
        checkpoint = external_output / "run" / "checkpoint-1000"
        checkpoint.mkdir(parents=True)
        os.symlink(external_output, output, target_is_directory=True)
        checkpoint = output / "run" / "checkpoint-1000"
    elif kind == "canonical":
        output.mkdir()
        external_run = tmp_path / "external-run"
        checkpoint = external_run / "checkpoint-1000"
        checkpoint.mkdir(parents=True)
        os.symlink(external_run, output / "run", target_is_directory=True)
        checkpoint = output / "run" / "checkpoint-1000"
    else:
        output.mkdir()
        external_stage = tmp_path / "external-stage"
        checkpoint = external_stage / "run" / "checkpoint-1000"
        checkpoint.mkdir(parents=True)
        os.symlink(
            external_stage, output / ".runtime-hf-resume-1000-deadbeefdeadbeef",
            target_is_directory=True,
        )
        checkpoint = output / ".runtime-hf-resume-1000-deadbeefdeadbeef" / "run" / "checkpoint-1000"
    (checkpoint / "trainer_state.json").write_text('{"global_step":1000}', encoding="utf-8")
    official = [
        "--output-dir", str(output), "--experiment-name", "run",
        "--num-gpus", "1", "--global-batch-size", "64",
        "--resume-from-checkpoint", str(checkpoint),
    ]

    with pytest.raises(ValueError, match="symlink"):
        chunk_launch._runtime_checkpoint_binding(["--"], official, num_gpus=1)


@pytest.mark.parametrize("experiment", (".", ".."))
def test_runtime_checkpoint_binding_rejects_non_component_experiment_direct_bypass(
    tmp_path: Path, experiment: str,
) -> None:
    output = tmp_path / "output"
    checkpoint = output / "checkpoint-1000"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text('{"global_step":1000}', encoding="utf-8")
    official = [
        "--output-dir", str(output), "--experiment-name", experiment,
        "--num-gpus", "1", "--global-batch-size", "64",
        "--resume-from-checkpoint", str(checkpoint),
    ]

    with pytest.raises(ValueError, match="unsafe checkpoint binding"):
        chunk_launch._runtime_checkpoint_binding(["--"], official, num_gpus=1)


def test_runtime_checkpoint_binding_rejects_dotdot_output_direct_bypass(tmp_path: Path) -> None:
    output = tmp_path / "output" / ".." / "external"
    checkpoint = tmp_path / "external" / "run" / "checkpoint-1000"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text('{"global_step":1000}', encoding="utf-8")
    official = [
        "--output-dir", str(output), "--experiment-name", "run",
        "--num-gpus", "1", "--global-batch-size", "64",
        "--resume-from-checkpoint", str(checkpoint),
    ]

    with pytest.raises(ValueError, match="unsafe checkpoint binding"):
        chunk_launch._runtime_checkpoint_binding(["--"], official, num_gpus=1)
