from __future__ import annotations

import math
from pathlib import Path
import subprocess

import pytest

from lehome_train.groot.config import FineTuneLaunchConfig
from lehome_train.groot.launch import build_launch, launch_finetune
from lehome_train.groot.metrics import parse_trainer_log_lines
from lehome_train.constants import ISAAC_GROOT_REVISION


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
    }
    values.update(overrides)
    return FineTuneLaunchConfig(**values)


@pytest.fixture
def official_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    checkout = tmp_path / "isaac-groot"
    entrypoint = checkout / "gr00t" / "experiment" / "launch_finetune.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("# official launcher fixture\n", encoding="utf-8")
    monkeypatch.setattr(
        "lehome_train.groot.launch._checkout_head",
        lambda _path: ISAAC_GROOT_REVISION,
    )
    monkeypatch.setattr("lehome_train.groot.launch._checkout_is_clean", lambda _path: True)
    return checkout


def test_build_launch_uses_only_pinned_official_entrypoint_and_redacts_token(
    official_checkout: Path,
) -> None:
    launch = build_launch(
        config(),
        visible_devices="1",
        environment={"HF_TOKEN": "hf_abcdefghijklmnopqrstuvwxyz0123456789", "PATH": "/bin"},
        official_checkout=official_checkout,
    )

    assert launch.command[:2] == ("python", str(official_checkout / "gr00t" / "experiment" / "launch_finetune.py"))
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
def test_build_launch_refuses_zero_or_multiple_visible_gpus(
    visible_devices: str,
    official_checkout: Path,
) -> None:
    with pytest.raises(ValueError, match="visible GPU"):
        build_launch(
            config(),
            visible_devices=visible_devices,
            environment={},
            official_checkout=official_checkout,
        )


def test_launch_refuses_incompatible_existing_experiment_directory(
    tmp_path: Path,
    official_checkout: Path,
) -> None:
    output = tmp_path / "experiment"
    effective = output / "lehome-groot-baseline"
    effective.mkdir(parents=True)
    (effective / "lehome_launch.json").write_text('{"dataset_revision":"' + ("b" * 40) + '"}', encoding="utf-8")

    with pytest.raises(ValueError, match="incompatible experiment"):
        launch_finetune(
            config(output_dir=str(output)),
            visible_devices="0",
            environment={},
            official_checkout=official_checkout,
            runner=lambda *_args, **_kwargs: pytest.fail("runner must not execute"),
        )


def test_launch_identity_rejects_behavior_changing_settings_before_runner(
    tmp_path: Path,
    official_checkout: Path,
) -> None:
    output = tmp_path / "experiment"
    expected = config(output_dir=str(output))
    effective = output / expected.experiment_name
    effective.mkdir(parents=True)
    (effective / "lehome_launch.json").write_text(
        __import__("json").dumps(expected.identity()), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="incompatible experiment"):
        launch_finetune(
            config(output_dir=str(output), learning_rate=2e-4),
            visible_devices="0",
            environment={},
            official_checkout=official_checkout,
            runner=lambda *_args, **_kwargs: pytest.fail("runner must not execute"),
        )


def test_build_launch_refuses_missing_or_wrong_pinned_official_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="official GR00T entrypoint"):
        build_launch(
            config(), visible_devices="0", environment={}, official_checkout=tmp_path / "missing"
        )

    checkout = tmp_path / "wrong-revision"
    entrypoint = checkout / "gr00t" / "experiment" / "launch_finetune.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("# fixture\n", encoding="utf-8")
    monkeypatch.setattr("lehome_train.groot.launch._checkout_head", lambda _path: "b" * 40)
    with pytest.raises(ValueError, match="pinned Isaac-GR00T"):
        build_launch(config(), visible_devices="0", environment={}, official_checkout=checkout)


def test_build_launch_refuses_altered_or_untracked_official_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "real-git-checkout"
    entrypoint = checkout / "gr00t" / "experiment" / "launch_finetune.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("# committed launcher\n", encoding="utf-8")
    for arguments in (
        ("git", "init", str(checkout)),
        ("git", "-C", str(checkout), "config", "user.email", "test@example.invalid"),
        ("git", "-C", str(checkout), "config", "user.name", "Test"),
        ("git", "-C", str(checkout), "add", "."),
        ("git", "-C", str(checkout), "commit", "-m", "fixture"),
    ):
        subprocess.run(arguments, check=True, capture_output=True, text=True)
    monkeypatch.setattr(
        "lehome_train.groot.launch._checkout_head",
        lambda _path: ISAAC_GROOT_REVISION,
    )

    build_launch(config(), visible_devices="0", environment={}, official_checkout=checkout)
    entrypoint.write_text("# altered launcher\n", encoding="utf-8")

    with pytest.raises(ValueError, match="official GR00T checkout is not clean"):
        build_launch(config(), visible_devices="0", environment={}, official_checkout=checkout)

    subprocess.run(
        ("git", "-C", str(checkout), "checkout", "--", "gr00t/experiment/launch_finetune.py"),
        check=True,
        capture_output=True,
        text=True,
    )
    (checkout / "untracked.py").write_text("# untracked\n", encoding="utf-8")
    with pytest.raises(ValueError, match="official GR00T checkout is not clean"):
        build_launch(config(), visible_devices="0", environment={}, official_checkout=checkout)


def test_identity_is_written_to_effective_official_experiment_directory(
    tmp_path: Path,
    official_checkout: Path,
) -> None:
    output = tmp_path / "output-root"
    completed = launch_finetune(
        config(output_dir=str(output)),
        visible_devices="0",
        environment={},
        official_checkout=official_checkout,
        runner=lambda *_args, **_kwargs: __import__("subprocess").CompletedProcess([], 0),
    )

    assert completed.returncode == 0
    assert not (output / "lehome_launch.json").exists()
    assert (output / "lehome-groot-baseline" / "lehome_launch.json").is_file()


def test_parser_returns_finite_structured_loss_throughput_and_checkpoint_timing() -> None:
    records = parse_trainer_log_lines(
        [
            "{'loss': 0.25, 'grad_norm': 1.3, 'learning_rate': 0.0001, 'epoch': 0.1}",
            "{'loss': 0.125, 'step': 12, 'steps_per_second': 3.5, 'samples_per_second': 224.0}",
            "Saving model checkpoint to /output/baseline/checkpoint-12",
            "{'loss': nan, 'step': 13}",
        ],
        timestamps_seconds=[10.0, 11.0, 14.5, 15.0],
    )

    assert [record.kind for record in records] == ["loss", "loss", "checkpoint", "loss"]
    assert records[0].loss == 0.25
    assert records[1].optimizer_step == 12
    assert records[1].steps_per_second == 3.5
    assert records[1].samples_per_second == 224.0
    assert records[2].checkpoint_path == "/output/baseline/checkpoint-12"
    assert records[2].recorded_at_seconds == 14.5
    assert records[3].finite_loss is False
    assert math.isnan(records[3].loss or 0.0)
