from __future__ import annotations

import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

import lehome_train.groot.production_adapters as adapters
from lehome_train.checkpoints import CheckpointDescriptor, write_checkpoint_descriptor
from lehome_train.commands.train import TrainingChunkRequest
from lehome_train.constants import MODEL_REVISION
from lehome_train.groot.config import FineTuneLaunchConfig
from lehome_train.io import canonical_json_sha256, sha256_file
from lehome_train.models import ArtifactIdentity, CheckpointRecord, ExperimentConfig
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
    run_root = Path(config.output_dir) / config.experiment_name
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "lehome_launch.json").write_text(
        json.dumps(config.identity()), encoding="utf-8"
    )
    checkpoint = run_root / f"checkpoint-{step}"
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


def _write_zero2_shards(checkpoint: Path) -> None:
    shard_root = checkpoint / f"global_step{checkpoint.name.removeprefix('checkpoint-')}"
    shard_root.mkdir()
    (shard_root / "mp_rank_00_model_states.pt").write_bytes(b"model")
    for rank in range(4):
        (
            shard_root / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
        ).write_bytes(b"optimizer")


def test_visible_gpu_mapping_honors_reordered_numeric_and_uuid_cuda_devices() -> None:
    numeric = adapters.resolve_visible_gpu_devices(
        "3,1,0,2", expected_gpu_count=4
    )
    assert [device.cuda_device_index for device in numeric] == [0, 1, 2, 3]
    assert [device.nvml_device_index for device in numeric] == [3, 1, 0, 2]

    uuid = adapters.resolve_visible_gpu_devices(
        "GPU-c,GPU-a,GPU-d,GPU-b",
        expected_gpu_count=4,
        uuid_indices={"GPU-a": 0, "GPU-b": 1, "GPU-c": 2, "GPU-d": 3},
    )
    assert [device.nvml_device_index for device in uuid] == [2, 0, 3, 1]

    with pytest.raises(ValueError, match="MIG"):
        adapters.resolve_visible_gpu_devices("MIG-one,1,2,3", expected_gpu_count=4)


def test_nvml_probe_uses_resolved_physical_indices_not_cuda_logical_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int | None]] = []

    class FakeSampler:
        def __init__(self, *, device_index: int, nvml_device_index: int | None = None) -> None:
            calls.append((device_index, nvml_device_index))
            self.nvml_device_index = nvml_device_index

        def __enter__(self) -> "FakeSampler":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def sample(self) -> SimpleNamespace:
            assert self.nvml_device_index is not None
            return SimpleNamespace(
                physical_total_vram_bytes=(self.nvml_device_index + 20) * 1024**3,
                free_vram_bytes=3 * 1024**3,
            )

    monkeypatch.setattr(adapters, "NvmlTelemetrySampler", FakeSampler)
    observed = adapters.probe_visible_gpu_memory(
        expected_gpu_count=4, visible_devices="3,1,0,2"
    )

    assert calls == [(0, 3), (1, 1), (2, 0), (3, 2)]
    assert [item.total_bytes // 1024**3 for item in observed] == [23, 21, 20, 22]


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


def _resume_descriptor(
    config: FineTuneLaunchConfig,
    experiment: ExperimentConfig,
    schedule: ExposureSchedule,
    archive: Path,
    step: int,
    *,
    experiment_id: str = "experiment-001",
) -> CheckpointDescriptor:
    return CheckpointDescriptor(
        record=CheckpointRecord(
            experiment_id=experiment_id,
            optimizer_step=step,
            sample_presentations=step * schedule.physical_batch_size,
            experiment_config_sha256=canonical_json_sha256(experiment),
            dataset_manifest_sha256=experiment.dataset_manifest_sha256,
            schedule_sha256=schedule.sha256,
            artifact=ArtifactIdentity(
                archive.relative_to(config.output_dir).as_posix(),
                sha256_file(archive),
                archive.stat().st_size,
            ),
            resumable=True,
            remotely_verified=True,
        ),
        normalization_sha256=NORMALIZATION_SHA256,
        schedule_sha256=schedule.sha256,
        locally_verified=True,
    )


def test_resume_archive_is_verified_in_staging_then_atomically_exposed(
    tmp_path: Path,
) -> None:
    schedule = ExposureSchedule(physical_batch_size=64)
    config = _config(
        tmp_path,
        batch=64,
        max_steps=schedule.total_optimizer_steps,
        save_steps=schedule.checkpoint_interval_steps,
    )
    step = schedule.checkpoint_interval_steps
    source_root = tmp_path / "source" / config.experiment_name
    source_config = __import__("dataclasses").replace(
        config, output_dir=str(tmp_path / "source")
    )
    source = _write_checkpoint(source_config, step)
    (source_root / "lehome_launch.json").write_text(
        json.dumps(config.identity()), encoding="utf-8"
    )
    archive = Path(config.output_dir) / "checkpoints" / f"step-{step}.tar"
    archive.parent.mkdir()
    adapters._tar_checkpoint(source, archive, source_root, config.experiment_name)
    experiment = _experiment(64)

    adapters.GrootTrainingSession(
        config=config,
        experiment_config=experiment,
        normalization_sha256=NORMALIZATION_SHA256,
        resume_checkpoint=_resume_descriptor(config, experiment, schedule, archive, step),
    )

    restored = Path(config.output_dir) / config.experiment_name / f"checkpoint-{step}"
    assert (restored / "trainer_state.json").is_file()
    assert not list(restored.parent.glob(".*.incomplete-*"))


def test_incompatible_resume_descriptor_is_rejected_before_archive_extraction(
    tmp_path: Path,
) -> None:
    schedule = ExposureSchedule(physical_batch_size=64)
    config = _config(
        tmp_path,
        batch=64,
        max_steps=schedule.total_optimizer_steps,
        save_steps=schedule.checkpoint_interval_steps,
    )
    step = schedule.checkpoint_interval_steps
    archive = Path(config.output_dir) / "checkpoints" / f"step-{step}.tar"
    archive.parent.mkdir()
    with tarfile.open(archive, "w") as bundle:
        payload = tmp_path / "payload"
        payload.write_bytes(b"untrusted")
        bundle.add(payload, arcname=f"{config.experiment_name}/checkpoint-{step}/weights.bin")
    experiment = _experiment(64)

    with pytest.raises(ValueError, match="experiment identity"):
        adapters.GrootTrainingSession(
            config=config,
            experiment_config=experiment,
            normalization_sha256=NORMALIZATION_SHA256,
            resume_checkpoint=_resume_descriptor(
                config,
                experiment,
                schedule,
                archive,
                step,
                experiment_id="wrong-experiment",
            ),
        )

    assert not (Path(config.output_dir) / config.experiment_name / f"checkpoint-{step}").exists()


def test_resume_archive_rejects_preexisting_symlink_destination_without_escape(
    tmp_path: Path,
) -> None:
    schedule = ExposureSchedule(physical_batch_size=64)
    config = _config(
        tmp_path,
        batch=64,
        max_steps=schedule.total_optimizer_steps,
        save_steps=schedule.checkpoint_interval_steps,
    )
    step = schedule.checkpoint_interval_steps
    source = tmp_path / "source" / config.experiment_name / f"checkpoint-{step}"
    source.mkdir(parents=True)
    (source / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "log_history": [{"step": step, "loss": 0.5}]}),
        encoding="utf-8",
    )
    archive = Path(config.output_dir) / "checkpoints" / f"step-{step}.tar"
    archive.parent.mkdir()
    with tarfile.open(archive, "w") as bundle:
        bundle.add(
            source,
            arcname=f"{config.experiment_name}/checkpoint-{step}",
        )
    experiment = _experiment(64)
    outside = tmp_path / "outside"
    outside.mkdir()
    run_root = Path(config.output_dir) / config.experiment_name
    run_root.mkdir()
    (run_root / f"checkpoint-{step}").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="destination already exists"):
        adapters.GrootTrainingSession(
            config=config,
            experiment_config=experiment,
            normalization_sha256=NORMALIZATION_SHA256,
            resume_checkpoint=_resume_descriptor(config, experiment, schedule, archive, step),
        )

    assert list(outside.iterdir()) == []


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


def test_checkpoint_archive_restores_launch_identity_then_runs_next_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schedule = ExposureSchedule(
        physical_batch_size=1, checkpoint_sample_presentations=100
    )
    config = _config(tmp_path, batch=1, max_steps=200, save_steps=100)
    _write_checkpoint(config, 100)
    session = adapters.GrootTrainingSession(
        config=config,
        experiment_config=_experiment(1),
        normalization_sha256=NORMALIZATION_SHA256,
        resume_checkpoint=None,
    )
    session._progress = 100
    descriptor = session.package_checkpoint(
        optimizer_step=100,
        sample_presentations=100,
        schedule_sha256=schedule.sha256,
    )
    descriptor = __import__("dataclasses").replace(
        descriptor,
        record=__import__("dataclasses").replace(
            descriptor.record, remotely_verified=True
        ),
    )
    __import__("shutil").rmtree(Path(config.output_dir) / config.experiment_name)

    resumed = adapters.GrootTrainingSession(
        config=config,
        experiment_config=_experiment(1),
        normalization_sha256=NORMALIZATION_SHA256,
        resume_checkpoint=descriptor,
    )
    launched: list[int] = []

    def fake_launch(_config: FineTuneLaunchConfig, **kwargs: object) -> None:
        launched.append(kwargs["stop_after_optimizer_step"])
        _write_checkpoint(config, 200)

    monkeypatch.setattr(adapters, "launch_finetune_to_step", fake_launch)
    receipt = resumed.run_chunk(_request(schedule, 100, 200))

    assert (Path(config.output_dir) / config.experiment_name / "lehome_launch.json").is_file()
    assert launched == [200]
    assert receipt.start_optimizer_step == 100
    assert receipt.end_optimizer_step == 200


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


def test_four_gpu_checkpoint_requires_every_zero2_model_and_optimizer_shard(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, batch=1, max_steps=100, save_steps=100)
    config = __import__("dataclasses").replace(
        config, num_gpus=4, global_batch_size=4
    )
    checkpoint = _write_checkpoint(config, 100)
    _write_zero2_shards(checkpoint)
    adapters._verified_checkpoint_state(config, 100)

    shard_root = checkpoint / "global_step100"
    (shard_root / "bf16_zero_pp_rank_3_mp_rank_00_optim_states.pt").unlink()
    with pytest.raises(ValueError, match="shard layout"):
        adapters._verified_checkpoint_state(config, 100)
    (shard_root / "bf16_zero_pp_rank_3_mp_rank_00_optim_states.pt").write_bytes(
        b"optimizer"
    )
    (shard_root / "bf16_zero_pp_rank_4_mp_rank_00_optim_states.pt").write_bytes(
        b"extra"
    )
    with pytest.raises(ValueError, match="shard layout"):
        adapters._verified_checkpoint_state(config, 100)
    (shard_root / "bf16_zero_pp_rank_4_mp_rank_00_optim_states.pt").unlink()
    (shard_root / "mp_rank_00_model_states.pt").unlink()
    (shard_root / "mp_rank_00_model_states.pt").symlink_to(
        shard_root / "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt"
    )
    with pytest.raises(ValueError, match="regular files"):
        adapters._verified_checkpoint_state(config, 100)


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


def test_checkpoint_uploader_uses_explicit_parent_token_without_process_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uploader = adapters.HubCheckpointUploader(
        repository=adapters.DEFAULT_MODEL_REPO,
        revision="a" * 40,
        experiment_id="experiment-001",
        artifact_root=tmp_path,
        token="publisher-token",
    )
    assert uploader._hub_environ == {"HF_TOKEN": "publisher-token"}
    assert "HF_TOKEN" not in __import__("os").environ


def test_checkpoint_uploader_publishes_descriptor_with_archive_to_one_immutable_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "checkpoints" / "step-1000.tar"
    archive.parent.mkdir()
    archive.write_bytes(b"checkpoint archive")
    checkpoint = CheckpointDescriptor(
        record=CheckpointRecord(
            experiment_id="experiment-001",
            optimizer_step=1000,
            sample_presentations=64_000,
            experiment_config_sha256="a" * 64,
            dataset_manifest_sha256="b" * 64,
            schedule_sha256="c" * 64,
            artifact=ArtifactIdentity(
                "checkpoints/step-1000.tar", sha256_file(archive), archive.stat().st_size,
            ),
            resumable=True,
            remotely_verified=False,
        ),
        normalization_sha256="d" * 64,
        schedule_sha256="c" * 64,
        locally_verified=True,
    )
    descriptor = tmp_path / "checkpoints" / "step-1000.json"
    write_checkpoint_descriptor(descriptor, checkpoint)
    uploaded: list[object] = []

    monkeypatch.setattr(adapters, "HuggingFaceHubTransport", lambda **_kwargs: object())
    monkeypatch.setattr(adapters, "require_access", lambda **_kwargs: None)

    def fake_upload(**kwargs: object) -> str:
        assert kwargs["max_attempts"] == 3
        uploaded.extend(kwargs["entries"])
        return "e" * 40

    def fake_download(**kwargs: object) -> str:
        assert kwargs["max_attempts"] == 3
        destination = kwargs["destination"]
        assert isinstance(destination, Path)
        for relative in kwargs["relative_paths"]:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((tmp_path / relative).read_bytes())
        return "e" * 40

    monkeypatch.setattr(adapters, "upload_files", fake_upload)
    monkeypatch.setattr(adapters, "download_files", fake_download)
    publication = adapters.HubCheckpointUploader(
        repository=adapters.DEFAULT_MODEL_REPO,
        revision="main",
        experiment_id="experiment-001",
        artifact_root=tmp_path,
        token="publisher-token",
    ).publish_receipt(checkpoint, timeout_seconds=1)

    assert {entry.relative_path for entry in uploaded} == {
        "checkpoints/step-1000.tar", "checkpoints/step-1000.json",
    }
    assert publication["immutable_revision"] == "e" * 40
    assert publication["descriptor_relative_path"] == "checkpoints/step-1000.json"
    assert publication["descriptor_sha256"] == sha256_file(descriptor)
    assert publication["descriptor_byte_size"] == descriptor.stat().st_size
