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


def test_fresh_run_disables_upstream_unconditional_resume(tmp_path) -> None:
    assert _resume_value(tmp_path) is False


def test_existing_checkpoint_preserves_real_resume(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-1000"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 1000}), encoding="utf-8"
    )

    assert _resume_value(tmp_path) is True


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
