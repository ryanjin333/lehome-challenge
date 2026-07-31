from __future__ import annotations

import json
from pathlib import Path

import pytest

import lehome_train.groot.production_adapters as adapters
from lehome_train.commands.train import TrainingChunkRequest
from lehome_train.constants import MODEL_REVISION
from lehome_train.groot.config import FineTuneLaunchConfig
from lehome_train.io import canonical_json_sha256, sha256_file
from lehome_train.models import ExperimentConfig
from lehome_train.offline_eval import OfflineEvaluation
from lehome_train.schedule import ExposureSchedule


COMMIT = "a" * 40
DATASET_REVISION = "b" * 40
DATASET_SHA256 = "c" * 64
NORMALIZATION_SHA256 = "d" * 64
IMAGE_DIGEST = "sha256:" + "e" * 64


def _config(tmp_path: Path, *, batch: int, max_steps: int, save_steps: int) -> FineTuneLaunchConfig:
    dataset = tmp_path / "prepared" / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    (dataset / "manifest.json").write_text(
        json.dumps(
            {
                "output_format": "groot_lerobot_v2.1_per_episode",
                "train_episode_ids": ["0"],
                "validation_episode_ids": ["1"],
            }
        ),
        encoding="utf-8",
    )
    model = tmp_path / "cache" / "model"
    model.mkdir(parents=True, exist_ok=True)
    modality = dataset / "meta" / "modality.py"
    modality.parent.mkdir(parents=True, exist_ok=True)
    modality.write_text("# fixture\n", encoding="utf-8")
    output = tmp_path / "output" / "experiment"
    output.mkdir(parents=True, exist_ok=True)
    return FineTuneLaunchConfig(
        base_model_path=str(model),
        base_model_revision=MODEL_REVISION,
        dataset_path=str(dataset),
        dataset_revision=DATASET_REVISION,
        modality_config_path=str(modality),
        output_dir=str(output),
        experiment_name="experiment-001",
        physical_batch_size=batch,
        max_steps=max_steps,
        save_steps=save_steps,
        warmup_ratio=0.05,
    )


def _experiment(batch: int) -> ExperimentConfig:
    return ExperimentConfig(
        repository_commit=COMMIT,
        container_digest=IMAGE_DIGEST,
        model_repository="nvidia/GR00T-N1.7-3B",
        model_revision=MODEL_REVISION,
        dataset_repository="ryanjin333/lehome-groot-n17-data",
        dataset_revision=DATASET_REVISION,
        dataset_manifest_sha256=DATASET_SHA256,
        physical_batch_size=batch,
        gradient_accumulation_steps=1,
        sample_presentations=768_000,
        action_horizon=16,
        tune_language_backbone=False,
        tune_visual_backbone=False,
    )


def _write_checkpoint(config: FineTuneLaunchConfig, step: int) -> Path:
    checkpoint = Path(config.output_dir) / config.experiment_name / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "weights.bin").write_bytes(f"weights-{step}".encode())
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": step,
                "log_history": [{"step": step, "loss": 0.25}],
            }
        ),
        encoding="utf-8",
    )
    return checkpoint


def _request(schedule: ExposureSchedule, start: int, end: int) -> TrainingChunkRequest:
    return TrainingChunkRequest(
        schedule_sha256=schedule.sha256,
        start_optimizer_step=start,
        end_optimizer_step=end,
        start_sample_presentations=start * schedule.physical_batch_size,
        end_sample_presentations=end * schedule.physical_batch_size,
        total_optimizer_steps=schedule.total_optimizer_steps,
        physical_batch_size=schedule.physical_batch_size,
        warmup_optimizer_steps=schedule.warmup_optimizer_steps,
        warmup_fraction=float(schedule.warmup_fraction),
        base_learning_rate=0.0,
        peak_learning_rate=1e-4,
        start_learning_rate_multiplier=schedule.learning_rate_multiplier(start),
        end_learning_rate_multiplier=schedule.learning_rate_multiplier(end),
        start_learning_rate=schedule.learning_rate(start),
        end_learning_rate=schedule.learning_rate(end),
        input_checkpoint=None,
        input_checkpoint_sha256=None,
    )


def test_training_session_runs_pinned_boundary_and_packages_verified_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schedule = ExposureSchedule(physical_batch_size=64)
    config = _config(
        tmp_path,
        batch=64,
        max_steps=schedule.total_optimizer_steps,
        save_steps=schedule.checkpoint_interval_steps,
    )
    launched: list[int] = []

    def fake_launch(_config: FineTuneLaunchConfig, **kwargs: object) -> None:
        step = kwargs["stop_after_optimizer_step"]
        assert isinstance(step, int)
        launched.append(step)
        _write_checkpoint(config, step)

    monkeypatch.setattr(adapters, "launch_finetune_to_step", fake_launch)
    session = adapters.GrootTrainingSession(
        config=config,
        experiment_config=_experiment(64),
        normalization_sha256=NORMALIZATION_SHA256,
        resume_checkpoint=None,
    )
    request = _request(schedule, 0, schedule.checkpoint_interval_steps)

    receipt = session.run_chunk(request)
    descriptor = session.package_checkpoint(
        optimizer_step=request.end_optimizer_step,
        sample_presentations=request.end_sample_presentations,
        schedule_sha256=schedule.sha256,
    )

    artifact = Path(config.output_dir) / descriptor.record.artifact.relative_path
    assert launched == [schedule.checkpoint_interval_steps]
    assert receipt.start_optimizer_step == 0
    assert receipt.end_optimizer_step == schedule.checkpoint_interval_steps
    assert receipt.finite_loss is True
    assert artifact.is_file()
    assert descriptor.record.artifact.sha256 == sha256_file(artifact)
    assert descriptor.record.artifact.byte_size == artifact.stat().st_size
    assert descriptor.record.experiment_config_sha256 == canonical_json_sha256(
        _experiment(64)
    )
    assert descriptor.locally_verified is True
    assert descriptor.record.remotely_verified is False


def test_training_session_rejects_nonfinite_or_unproven_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schedule = ExposureSchedule(physical_batch_size=64)
    config = _config(
        tmp_path,
        batch=64,
        max_steps=schedule.total_optimizer_steps,
        save_steps=schedule.checkpoint_interval_steps,
    )

    def fake_launch(_config: FineTuneLaunchConfig, **kwargs: object) -> None:
        checkpoint = _write_checkpoint(config, kwargs["stop_after_optimizer_step"])
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        state["log_history"] = [{"step": 1000, "loss": float("nan")}]
        (checkpoint / "trainer_state.json").write_text(json.dumps(state))

    monkeypatch.setattr(adapters, "launch_finetune_to_step", fake_launch)
    session = adapters.GrootTrainingSession(
        config=config,
        experiment_config=_experiment(64),
        normalization_sha256=NORMALIZATION_SHA256,
        resume_checkpoint=None,
    )

    with pytest.raises(ValueError, match="finite loss"):
        session.run_chunk(_request(schedule, 0, schedule.checkpoint_interval_steps))


def test_checkpoint_state_requires_loss_from_current_boundary(
    tmp_path: Path,
) -> None:
    schedule = ExposureSchedule(physical_batch_size=64)
    config = _config(
        tmp_path,
        batch=64,
        max_steps=schedule.total_optimizer_steps,
        save_steps=schedule.checkpoint_interval_steps,
    )
    checkpoint = _write_checkpoint(config, schedule.checkpoint_interval_steps)
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    state["log_history"] = [{"step": 1, "loss": 0.5}]
    (checkpoint / "trainer_state.json").write_text(json.dumps(state))

    with pytest.raises(ValueError, match="current loss"):
        adapters._verified_checkpoint_state(
            config, schedule.checkpoint_interval_steps
        )


def test_memorization_initializes_without_steps_then_evaluates_exact_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, batch=1, max_steps=10_000, save_steps=1_000)
    launched: list[int] = []
    evaluations: list[tuple[Path, Path, str]] = []

    def fake_launch(_config: FineTuneLaunchConfig, **kwargs: object) -> None:
        step = kwargs["stop_after_optimizer_step"]
        launched.append(step)
        run = Path(config.output_dir) / config.experiment_name
        run.mkdir(parents=True, exist_ok=True)
        if step:
            _write_checkpoint(config, step)
        else:
            (run / "config.json").write_text("{}")

    expected = OfflineEvaluation(1.0, (1.0,), 1, 1)

    def fake_evaluate(*, dataset_path: Path, model_path: Path, episode_id: str) -> OfflineEvaluation:
        evaluations.append((dataset_path, model_path, episode_id))
        return expected

    monkeypatch.setattr(adapters, "launch_finetune_to_step", fake_launch)
    monkeypatch.setattr(adapters, "evaluate_checkpoint_episode", fake_evaluate)
    session = adapters.GrootMemorizationSession(config=config)

    initial = session.evaluate(episode_id="0", sample_presentations=0)
    receipt = session.train_chunk(
        episode_id="0", optimizer_steps=500, physical_batch_size=1
    )
    trained = session.evaluate(episode_id="0", sample_presentations=500)

    assert launched == [0, 500]
    assert initial == trained == expected
    assert evaluations[0][1] == Path(config.output_dir) / config.experiment_name
    assert evaluations[1][1].name == "checkpoint-500"
    assert receipt.start_optimizer_step == 0
    assert receipt.end_optimizer_step == 500


def test_smoke_runner_records_only_proven_progress_and_no_fake_steady_timing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, batch=16, max_steps=100, save_steps=100)
    _write_checkpoint(config, 100)
    monkeypatch.setattr(
        adapters,
        "_run_smoke_process",
        lambda _config: (
            (
                "{'loss': 0.5, 'step': 10}",
                "{'loss': 0.25, 'step': 50}",
                "{'loss': 0.125, 'step': 100}",
            ),
            (2.0, 8.0, 18.0),
        ),
    )

    receipt = adapters.GrootSmokeRunner()(config)

    assert receipt.optimizer_steps == 100
    assert receipt.finite_loss is True
    assert receipt.initialization_seconds == 0
    assert receipt.warmup_seconds == 2.0
    assert receipt.steady_state_seconds == 16.0
    assert receipt.steady_state_optimizer_steps == 90
    assert receipt.telemetry_samples == ()


def test_training_session_requires_exact_predecessor_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schedule = ExposureSchedule(physical_batch_size=64)
    config = _config(
        tmp_path,
        batch=64,
        max_steps=schedule.total_optimizer_steps,
        save_steps=schedule.checkpoint_interval_steps,
    )
    _write_checkpoint(config, schedule.checkpoint_interval_steps)
    session = adapters.GrootTrainingSession(
        config=config,
        experiment_config=_experiment(64),
        normalization_sha256=NORMALIZATION_SHA256,
        resume_checkpoint=None,
    )
    launched = False

    def fake_launch(*_args: object, **_kwargs: object) -> None:
        nonlocal launched
        launched = True

    monkeypatch.setattr(adapters, "launch_finetune_to_step", fake_launch)
    request = _request(
        schedule,
        0,
        schedule.checkpoint_interval_steps * 2,
    )

    with pytest.raises(ValueError, match="predecessor"):
        session.run_chunk(request)
    assert launched is False


def test_training_session_accepts_already_completed_exact_target_without_relaunch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schedule = ExposureSchedule(physical_batch_size=64)
    config = _config(
        tmp_path,
        batch=64,
        max_steps=schedule.total_optimizer_steps,
        save_steps=schedule.checkpoint_interval_steps,
    )
    target = schedule.checkpoint_interval_steps
    _write_checkpoint(config, target)
    session = adapters.GrootTrainingSession(
        config=config,
        experiment_config=_experiment(64),
        normalization_sha256=NORMALIZATION_SHA256,
        resume_checkpoint=None,
    )
    monkeypatch.setattr(
        adapters,
        "launch_finetune_to_step",
        lambda *_args, **_kwargs: pytest.fail("must not relaunch completed target"),
    )

    receipt = session.run_chunk(_request(schedule, 0, target))

    assert receipt.end_optimizer_step == target
