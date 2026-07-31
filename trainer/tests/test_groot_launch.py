from __future__ import annotations

import math
from pathlib import Path

import pytest

from lehome_train.groot.config import FineTuneLaunchConfig
from lehome_train.groot.launch import build_launch, launch_finetune
from lehome_train.groot.metrics import parse_trainer_log_lines


REVISION = "a" * 40


def config(**overrides: object) -> FineTuneLaunchConfig:
    values: dict[str, object] = {
        "base_model_path": "nvidia/GR00T-N1.7-3B",
        "base_model_revision": "2fc962b973bccdd5d8ce4f67cc63b264d6886495",
        "dataset_path": "/prepared/lehome-groot-n17-v1",
        "dataset_revision": REVISION,
        "modality_config_path": "/prepared/lehome-groot-n17-v1/meta/lehome_groot_modality.py",
        "output_dir": "/output/baseline",
        "experiment_name": "lehome-groot-baseline",
        "physical_batch_size": 64,
        "max_steps": 12_000,
        "save_steps": 1_000,
        "warmup_ratio": 0.05,
        "decay_ratio": 0.95,
    }
    values.update(overrides)
    return FineTuneLaunchConfig(**values)


def test_build_launch_uses_only_pinned_official_entrypoint_and_redacts_token() -> None:
    launch = build_launch(
        config(),
        visible_devices="1",
        environment={"HF_TOKEN": "hf_abcdefghijklmnopqrstuvwxyz0123456789", "PATH": "/bin"},
        official_checkout="/opt/isaac-groot",
    )

    assert launch.command[:2] == ("python", "/opt/isaac-groot/gr00t/experiment/launch_finetune.py")
    assert "--base-model-path" in launch.command
    assert "--base-model-revision" not in launch.command
    assert "--global-batch-size" in launch.command
    assert "64" in launch.command
    assert "--gradient-accumulation-steps" in launch.command
    assert "--warmup-ratio" in launch.command
    assert "--tune-projector" in launch.command
    assert "--tune-diffusion-model" in launch.command
    assert "--no-tune-llm" in launch.command
    assert "--no-tune-visual" in launch.command
    assert launch.environment["CUDA_VISIBLE_DEVICES"] == "1"
    assert "HF_TOKEN" not in launch.environment
    assert "hf_" not in " ".join(launch.command)


@pytest.mark.parametrize("visible_devices", ["", "0,1", "0, 1", "NoDevFiles"])
def test_build_launch_refuses_zero_or_multiple_visible_gpus(visible_devices: str) -> None:
    with pytest.raises(ValueError, match="visible GPU"):
        build_launch(
            config(),
            visible_devices=visible_devices,
            environment={},
            official_checkout="/opt/isaac-groot",
        )


def test_launch_refuses_incompatible_existing_experiment_directory(tmp_path: Path) -> None:
    output = tmp_path / "experiment"
    output.mkdir()
    (output / "lehome_launch.json").write_text('{"dataset_revision":"' + ("b" * 40) + '"}', encoding="utf-8")

    with pytest.raises(ValueError, match="incompatible experiment"):
        launch_finetune(
            config(output_dir=str(output)),
            visible_devices="0",
            environment={},
            official_checkout="/opt/isaac-groot",
            runner=lambda *_args, **_kwargs: pytest.fail("runner must not execute"),
        )


def test_parser_returns_finite_structured_loss_throughput_and_checkpoint_timing() -> None:
    records = parse_trainer_log_lines(
        [
            "{'loss': 0.25, 'grad_norm': 1.3, 'learning_rate': 0.0001, 'epoch': 0.1}",
            "{'loss': 0.125, 'step': 12, 'steps_per_second': 3.5, 'samples_per_second': 224.0}",
            "Saving model checkpoint to /output/baseline/checkpoint-12",
            "{'loss': nan, 'step': 13}",
        ]
    )

    assert [record.kind for record in records] == ["loss", "loss", "checkpoint", "loss"]
    assert records[0].loss == 0.25
    assert records[1].optimizer_step == 12
    assert records[1].steps_per_second == 3.5
    assert records[1].samples_per_second == 224.0
    assert records[2].checkpoint_path == "/output/baseline/checkpoint-12"
    assert records[3].finite_loss is False
    assert math.isnan(records[3].loss or 0.0)
