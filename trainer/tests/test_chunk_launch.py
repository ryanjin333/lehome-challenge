from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

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
