"""Concrete, evidence-checking adapters for the pinned GR00T runtime."""

from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from time import monotonic
from typing import Mapping

from lehome_train.checkpoints import (
    CheckpointDescriptor,
    write_checkpoint_descriptor,
)
from lehome_train.commands.memorize import ChunkReceipt
from lehome_train.commands.smoke import SmokeAttemptReceipt, SmokeRunnerFailure
from lehome_train.commands.train import TrainingChunkReceipt, TrainingChunkRequest
from lehome_train.constants import DEFAULT_MODEL_REPO
from lehome_train.groot.config import FineTuneLaunchConfig
from lehome_train.groot.launch import launch_finetune, launch_finetune_to_step
from lehome_train.groot.metrics import parse_trainer_log_lines
from lehome_train.hub import (
    HuggingFaceHubTransport,
    download_files,
    require_access,
    upload_files,
)
from lehome_train.io import canonical_json_bytes, canonical_json_sha256, sha256_file
from lehome_train.models import (
    ArtifactIdentity,
    CheckpointRecord,
    ExperimentConfig,
    SyncEntry,
)
from lehome_train.offline_eval import OfflineEvaluation, evaluate_action_predictions


def probe_physical_vram_bytes() -> int:
    """Read total memory from NVML without accepting a configured claim."""

    from lehome_train.telemetry import NvmlTelemetrySampler

    with NvmlTelemetrySampler() as sampler:
        return sampler.sample().physical_total_vram_bytes


def _run_root(config: FineTuneLaunchConfig) -> Path:
    return Path(config.output_dir) / config.experiment_name


def _checkpoint_path(config: FineTuneLaunchConfig, optimizer_step: int) -> Path:
    return _run_root(config) / f"checkpoint-{optimizer_step}"


def _checkpoint_steps(config: FineTuneLaunchConfig) -> tuple[int, ...]:
    root = _run_root(config)
    if not root.is_dir():
        return ()
    steps: list[int] = []
    for path in root.glob("checkpoint-*"):
        suffix = path.name.removeprefix("checkpoint-")
        if path.is_dir() and not path.is_symlink() and suffix.isdigit():
            steps.append(int(suffix))
    return tuple(sorted(steps))


def _preflight_segment(
    config: FineTuneLaunchConfig,
    *,
    expected_predecessor_step: int,
    target_step: int,
) -> bool:
    """Require exact latest resume state; return true if target already completed."""

    steps = _checkpoint_steps(config)
    latest = steps[-1] if steps else 0
    if latest > target_step:
        raise ValueError("GR00T output is ahead of the requested controller boundary")
    if latest == target_step:
        _verified_checkpoint_state(config, target_step)
        return True
    if latest != expected_predecessor_step:
        raise ValueError("GR00T output is missing the exact predecessor checkpoint")
    if latest:
        _verified_checkpoint_state(config, latest)
    return False


def _verified_checkpoint_state(
    config: FineTuneLaunchConfig,
    optimizer_step: int,
) -> Mapping[str, object]:
    checkpoint = _checkpoint_path(config, optimizer_step)
    state_path = checkpoint / "trainer_state.json"
    if not checkpoint.is_dir() or checkpoint.is_symlink() or not state_path.is_file():
        raise ValueError("GR00T checkpoint boundary has no trainer-state evidence")
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("GR00T trainer-state evidence is malformed") from None
    if not isinstance(state, Mapping) or state.get("global_step") != optimizer_step:
        raise ValueError("GR00T trainer-state progress does not match the boundary")
    history = state.get("log_history")
    if not isinstance(history, list):
        raise ValueError("GR00T trainer-state has no loss evidence")
    loss_entries = [
        entry
        for entry in history
        if isinstance(entry, Mapping) and "loss" in entry
    ]
    if not loss_entries or any(
        type(entry["loss"]) not in (int, float)
        or not math.isfinite(float(entry["loss"]))
        for entry in loss_entries
    ):
        raise ValueError("GR00T checkpoint boundary does not prove finite loss")
    if not any(entry.get("step") == optimizer_step for entry in loss_entries):
        raise ValueError("GR00T checkpoint boundary has no current loss evidence")
    return state


def _launch_kwargs() -> dict[str, object]:
    return {
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "environment": os.environ,
        "official_checkout": os.environ.get("LEHOME_GROOT_ROOT", "/opt/isaac-groot"),
    }


class _CapturedLaunchFailure(RuntimeError):
    def __init__(self, output: str) -> None:
        super().__init__("captured GR00T launch failed")
        self.output = output


def _run_smoke_process(
    config: FineTuneLaunchConfig,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    lines: list[str] = []
    timestamps: list[float] = []

    def runner(
        command: tuple[str, ...], *, env: Mapping[str, str], check: bool
    ) -> subprocess.CompletedProcess[object]:
        started = monotonic()
        process = subprocess.Popen(
            command,
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line.rstrip("\n"))
            timestamps.append(monotonic() - started)
        returncode = process.wait()
        if check and returncode:
            raise _CapturedLaunchFailure("\n".join(lines))
        return subprocess.CompletedProcess(command, returncode)

    launch_finetune(config, runner=runner, **_launch_kwargs())
    return tuple(lines), tuple(timestamps)


def _single_episode_dataset(dataset: Path, episode_id: str, destination: Path) -> Path:
    """Create an image-local view that exposes exactly one training episode."""

    try:
        manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("prepared memorization manifest is malformed") from None
    train = manifest.get("train_episode_ids") if isinstance(manifest, Mapping) else None
    if not isinstance(train, list) or episode_id not in train:
        raise ValueError("memorization episode is absent from the prepared training split")
    if train == [episode_id]:
        return dataset
    source_identity = {
        "schema_version": 1,
        "source_manifest_sha256": sha256_file(dataset / "manifest.json"),
        "episode_id": episode_id,
    }
    if destination.exists():
        identity_path = destination / "lehome_single_episode.json"
        try:
            observed_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise ValueError("memorization episode view is incompatible") from None
        if (
            not destination.is_dir()
            or destination.is_symlink()
            or observed_identity != source_identity
        ):
            raise ValueError("memorization episode view is incompatible")
        try:
            episode_lines = [
                json.loads(line)
                for line in (destination / "meta" / "episodes.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            ]
            viewed_manifest = json.loads(
                (destination / "manifest.json").read_text(encoding="utf-8")
            )
            from lehome_train.data.stats import _data_path

            viewed_data = _data_path(destination, episode_id)
            source_data = _data_path(dataset, episode_id)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            raise ValueError("memorization episode view is incompatible") from None
        if (
            len(episode_lines) != 1
            or str(episode_lines[0].get("episode_index")) != episode_id
            or viewed_manifest.get("train_episode_ids") != [episode_id]
            or viewed_data.resolve() != source_data.resolve()
            or (dataset / "videos").exists()
            and (
                not (destination / "videos").is_symlink()
                or (destination / "videos").resolve() != (dataset / "videos").resolve()
            )
        ):
            raise ValueError("memorization episode view is incompatible")
        return destination

    from lehome_train.data.stats import _data_path

    temporary = destination.with_name(f".{destination.name}.incomplete")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        shutil.copytree(dataset / "meta", temporary / "meta")
        selected_data = _data_path(dataset, episode_id)
        target_data = temporary / selected_data.relative_to(dataset)
        target_data.parent.mkdir(parents=True, exist_ok=True)
        target_data.symlink_to(selected_data.resolve())
        videos = dataset / "videos"
        if videos.exists():
            (temporary / "videos").symlink_to(videos.resolve(), target_is_directory=True)
        filtered = []
        episodes_path = temporary / "meta" / "episodes.jsonl"
        for line in episodes_path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if str(item.get("episode_index")) == episode_id:
                filtered.append(item)
        if len(filtered) != 1:
            raise ValueError("memorization episode metadata is not unique")
        from lehome_train.io import atomic_write_json

        episodes_payload = b"".join(
            canonical_json_bytes(item) + b"\n" for item in filtered
        )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".episodes.jsonl.", suffix=".tmp", dir=episodes_path.parent
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(episodes_payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, episodes_path)
        copied_manifest = dict(manifest)
        copied_manifest["train_episode_ids"] = [episode_id]
        copied_manifest["validation_episode_ids"] = []
        atomic_write_json(temporary / "manifest.json", copied_manifest)
        atomic_write_json(temporary / "lehome_single_episode.json", source_identity)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def evaluate_checkpoint_episode(
    *,
    dataset_path: Path,
    model_path: Path,
    episode_id: str,
) -> OfflineEvaluation:
    """Run pinned open-loop policy replay and compute normalized 12D MSE."""

    try:
        import numpy as np
        from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
        from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.eval.open_loop_eval import parse_action_gr00t, parse_observation_gr00t
        from gr00t.policy import Gr00tPolicy
    except ImportError:
        raise RuntimeError("pinned GR00T offline evaluator is unavailable") from None

    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        model_path=str(model_path),
        device="cuda:0",
        strict=True,
    )
    loader = LeRobotEpisodeLoader(
        dataset_path=str(dataset_path),
        modality_configs=policy.get_modality_config(),
        video_backend="torchcodec",
        video_backend_kwargs=None,
    )
    trajectory = loader[0]
    action_keys = loader.modality_configs["action"].modality_keys
    modality = dict(loader.modality_configs)
    modality.pop("action")
    predicted: list[list[float]] = []
    expert: list[list[float]] = []
    horizon = len(loader.modality_configs["action"].delta_indices)
    if horizon <= 0:
        raise ValueError("pinned policy action horizon is empty")
    frame_count = len(trajectory)
    for start in range(0, frame_count, horizon):
        point = extract_step_data(
            trajectory,
            start,
            modality,
            EmbodimentTag.NEW_EMBODIMENT,
        )
        observation: dict[str, object] = {
            **{f"state.{key}": value for key, value in point.states.items()},
            **{f"video.{key}": np.array(value) for key, value in point.images.items()},
        }
        for language_key in loader.modality_configs["language"].modality_keys:
            observation[language_key] = point.text
        raw_action, _info = policy.get_action(
            parse_observation_gr00t(observation, loader.modality_configs)
        )
        action_chunk = parse_action_gr00t(raw_action)
        for offset in range(min(horizon, frame_count - start)):
            predicted.append(
                np.concatenate(
                    [
                        np.atleast_1d(action_chunk[f"action.{key}"][offset])
                        for key in action_keys
                    ]
                ).astype(float).tolist()
            )
    for frame in range(frame_count):
        expert.append(
            np.concatenate(
                [
                    np.atleast_1d(trajectory.iloc[frame][f"action.{key}"])
                    for key in action_keys
                ]
            ).astype(float).tolist()
        )
    try:
        statistics = json.loads(
            (dataset_path / "meta" / "stats.json").read_text(encoding="utf-8")
        )["action"]
        scale = statistics["std"]
        minimum = statistics["min"]
        maximum = statistics["max"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        raise ValueError("prepared action normalization statistics are malformed") from None
    frames = tuple(range(frame_count))
    return evaluate_action_predictions(
        predicted_actions=predicted,
        expert_actions=expert,
        normalization_scale=scale,
        action_min=minimum,
        action_max=maximum,
        prediction_frame_indices=frames,
        expert_frame_indices=frames,
    )


class GrootMemorizationSession:
    """Persistent one-episode launch/evaluation session."""

    def __init__(self, *, config: FineTuneLaunchConfig) -> None:
        self.config = config
        self._episode_id: str | None = None
        self._progress = 0
        self._initialized = False

    def _episode_config(self, episode_id: str) -> FineTuneLaunchConfig:
        if self._episode_id is not None and self._episode_id != episode_id:
            raise ValueError("memorization session cannot change episodes")
        self._episode_id = episode_id
        dataset = _single_episode_dataset(
            Path(self.config.dataset_path),
            episode_id,
            Path(self.config.output_dir) / f"memorize-episode-{episode_id}",
        )
        return replace(self.config, dataset_path=str(dataset))

    def _launch(self, episode_id: str, stop_step: int) -> FineTuneLaunchConfig:
        config = self._episode_config(episode_id)
        launch_finetune_to_step(
            config,
            stop_after_optimizer_step=stop_step,
            **_launch_kwargs(),
        )
        self._initialized = True
        return config

    def train_chunk(
        self, *, episode_id: str, optimizer_steps: int, physical_batch_size: int
    ) -> ChunkReceipt:
        start = self._progress
        end = start + optimizer_steps
        config = self._episode_config(episode_id)
        if not _preflight_segment(
            config,
            expected_predecessor_step=start,
            target_step=end,
        ):
            config = self._launch(episode_id, end)
        _verified_checkpoint_state(config, end)
        self._progress = end
        return ChunkReceipt(start, end, optimizer_steps, physical_batch_size, True)

    def evaluate(
        self, *, episode_id: str, sample_presentations: int
    ) -> OfflineEvaluation:
        if sample_presentations != self._progress:
            raise ValueError("memorization evaluation progress is not current")
        config = self._episode_config(episode_id)
        if not self._initialized:
            config = self._launch(episode_id, 0)
        model_path = (
            _run_root(config)
            if sample_presentations == 0
            else _checkpoint_path(config, sample_presentations)
        )
        return evaluate_checkpoint_episode(
            dataset_path=Path(config.dataset_path),
            model_path=model_path,
            episode_id=episode_id,
        )

    def verify_checkpoint(self, *, episode_id: str, sample_presentations: int) -> None:
        config = self._episode_config(episode_id)
        _verified_checkpoint_state(config, sample_presentations)


class GrootSmokeRunner:
    """One official smoke process with conservative timing evidence."""

    def __call__(self, config: FineTuneLaunchConfig) -> SmokeAttemptReceipt:
        try:
            lines, timestamps = _run_smoke_process(config)
        except _CapturedLaunchFailure as error:
            if "cuda" in error.output.casefold() and "out of memory" in error.output.casefold():
                raise SmokeRunnerFailure.cuda_oom() from None
            raise
        state = _verified_checkpoint_state(config, config.max_steps)
        metrics = parse_trainer_log_lines(lines, timestamps_seconds=timestamps)
        losses = [metric for metric in metrics if metric.kind == "loss"]
        history = state["log_history"]
        assert isinstance(history, list)
        history_steps = [
            item.get("step") for item in history if isinstance(item, Mapping) and "loss" in item
        ]
        aligned = []
        for index, metric in enumerate(losses):
            step = metric.optimizer_step
            if step is None and index < len(history_steps) and type(history_steps[index]) is int:
                step = history_steps[index]
            if step is not None and metric.recorded_at_seconds is not None:
                aligned.append((step, metric.recorded_at_seconds))
        warmup_step = max(1, round(config.max_steps * config.warmup_ratio))
        steady = [(step, timestamp) for step, timestamp in aligned if step >= warmup_step]
        if len(steady) < 2 or steady[-1][0] <= steady[0][0] or steady[-1][1] <= steady[0][1]:
            raise ValueError("GR00T smoke run did not prove a steady-state timing interval")
        steady_steps = steady[-1][0] - steady[0][0]
        warmup_seconds = steady[0][1]
        steady_seconds = steady[-1][1] - steady[0][1]
        return SmokeAttemptReceipt(
            optimizer_steps=config.max_steps,
            gradient_accumulation_steps=1,
            finite_loss=True,
            initialization_seconds=0.0,
            warmup_seconds=warmup_seconds,
            steady_state_seconds=steady_seconds,
            steady_state_optimizer_steps=steady_steps,
            telemetry_samples=(),
            failure_reason=None,
        )


def _tar_checkpoint(source: Path, destination: Path, arcname: str) -> None:
    if not source.is_dir() or source.is_symlink():
        raise ValueError("GR00T checkpoint source is unavailable")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.incomplete")
    try:
        with tarfile.open(temporary, "w") as archive:
            for path in (source, *sorted(source.rglob("*"))):
                if path.is_symlink() or (not path.is_dir() and not path.is_file()):
                    raise ValueError("GR00T checkpoint contains an unsupported path")
                relative = Path(arcname) / path.relative_to(source)
                info = archive.gettarinfo(str(path), arcname=relative.as_posix())
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                if path.is_file():
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)
                else:
                    archive.addfile(info)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _restore_checkpoint_archive(
    archive_path: Path,
    *,
    output_root: Path,
    expected_member_root: str,
) -> None:
    with tarfile.open(archive_path, "r") as archive:
        members = archive.getmembers()
        if not members:
            raise ValueError("resume checkpoint archive is empty")
        for member in members:
            path = Path(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or member.issym()
                or member.islnk()
                or not (member.isdir() or member.isfile())
                or path.parts[:1] != (expected_member_root,)
            ):
                raise ValueError("resume checkpoint archive has an unsafe member")
        archive.extractall(output_root, members=members)


class GrootTrainingSession:
    """Official resumable process plus local checkpoint evidence operations."""

    def __init__(
        self,
        *,
        config: FineTuneLaunchConfig,
        experiment_config: ExperimentConfig,
        normalization_sha256: str,
        resume_checkpoint: CheckpointDescriptor | None,
    ) -> None:
        self.config = config
        self.experiment_config = experiment_config
        self.normalization_sha256 = normalization_sha256
        self._progress = 0 if resume_checkpoint is None else resume_checkpoint.record.optimizer_step
        if resume_checkpoint is not None:
            artifact = Path(config.output_dir) / resume_checkpoint.record.artifact.relative_path
            if (
                not artifact.is_file()
                or artifact.stat().st_size != resume_checkpoint.record.artifact.byte_size
                or sha256_file(artifact) != resume_checkpoint.record.artifact.sha256
            ):
                raise ValueError("resume checkpoint archive failed local verification")
            expected = f"{config.experiment_name}/checkpoint-{self._progress}"
            if not _checkpoint_path(config, self._progress).is_dir():
                _restore_checkpoint_archive(
                    artifact,
                    output_root=Path(config.output_dir),
                    expected_member_root=expected,
                )

    def run_chunk(self, request: TrainingChunkRequest) -> TrainingChunkReceipt:
        if request.start_optimizer_step != self._progress:
            raise ValueError("GR00T training session progress is not monotonic")
        if not _preflight_segment(
            self.config,
            expected_predecessor_step=request.start_optimizer_step,
            target_step=request.end_optimizer_step,
        ):
            launch_finetune_to_step(
                self.config,
                stop_after_optimizer_step=request.end_optimizer_step,
                **_launch_kwargs(),
            )
        _verified_checkpoint_state(self.config, request.end_optimizer_step)
        self._progress = request.end_optimizer_step
        return TrainingChunkReceipt(
            schedule_sha256=request.schedule_sha256,
            input_checkpoint_sha256=request.input_checkpoint_sha256,
            start_optimizer_step=request.start_optimizer_step,
            end_optimizer_step=request.end_optimizer_step,
            sample_presentations=(
                request.end_sample_presentations - request.start_sample_presentations
            ),
            physical_batch_size=request.physical_batch_size,
            finite_loss=True,
        )

    def package_checkpoint(
        self, *, optimizer_step: int, sample_presentations: int, schedule_sha256: str
    ) -> CheckpointDescriptor:
        if optimizer_step != self._progress:
            raise ValueError("cannot package a checkpoint that is not current")
        source = _checkpoint_path(self.config, optimizer_step)
        _verified_checkpoint_state(self.config, optimizer_step)
        relative = f"checkpoints/step-{optimizer_step}.tar"
        artifact_path = Path(self.config.output_dir) / relative
        _tar_checkpoint(
            source,
            artifact_path,
            f"{self.config.experiment_name}/checkpoint-{optimizer_step}",
        )
        descriptor = CheckpointDescriptor(
            record=CheckpointRecord(
                experiment_id=self.config.experiment_name,
                optimizer_step=optimizer_step,
                sample_presentations=sample_presentations,
                experiment_config_sha256=canonical_json_sha256(self.experiment_config),
                dataset_manifest_sha256=self.experiment_config.dataset_manifest_sha256,
                schedule_sha256=schedule_sha256,
                artifact=ArtifactIdentity(
                    relative_path=relative,
                    sha256=sha256_file(artifact_path),
                    byte_size=artifact_path.stat().st_size,
                ),
                resumable=True,
                remotely_verified=False,
            ),
            normalization_sha256=self.normalization_sha256,
            schedule_sha256=schedule_sha256,
            locally_verified=True,
        )
        write_checkpoint_descriptor(
            Path(self.config.output_dir) / "checkpoints" / f"step-{optimizer_step}.json",
            descriptor,
        )
        return descriptor

    def disk_free_bytes(self) -> int:
        return shutil.disk_usage(Path(self.config.output_dir)).free

    def delete_checkpoint_archive(self, checkpoint: CheckpointDescriptor) -> None:
        artifact = Path(self.config.output_dir) / checkpoint.record.artifact.relative_path
        if not artifact.is_file() or sha256_file(artifact) != checkpoint.record.artifact.sha256:
            raise ValueError("refusing to delete a checkpoint archive with changed identity")
        artifact.unlink()


class HubCheckpointUploader:
    """Explicit-token upload/readback adapter used only by the parent process."""

    def __init__(
        self,
        *,
        repository: str,
        revision: str,
        experiment_id: str,
        artifact_root: str | os.PathLike[str] | None = None,
    ) -> None:
        if repository != DEFAULT_MODEL_REPO:
            raise ValueError("checkpoint repository is not approved")
        if not revision or not experiment_id:
            raise ValueError("checkpoint upload identity is incomplete")
        self.repository = repository
        self.revision = revision
        self.experiment_id = experiment_id
        self.artifact_root = None if artifact_root is None else Path(artifact_root)

    def __call__(
        self, checkpoint: CheckpointDescriptor, *, timeout_seconds: float
    ) -> bool:
        if self.artifact_root is None:
            raise ValueError("checkpoint uploader has no artifact root")
        artifact = self.artifact_root / checkpoint.record.artifact.relative_path
        if (
            not artifact.is_file()
            or artifact.stat().st_size != checkpoint.record.artifact.byte_size
            or sha256_file(artifact) != checkpoint.record.artifact.sha256
        ):
            raise ValueError("checkpoint upload artifact failed local verification")
        transport = HuggingFaceHubTransport(timeout_seconds=timeout_seconds)
        require_access(
            transport=transport,
            repository=self.repository,
            read=True,
            write=True,
        )
        entry = SyncEntry(
            relative_path=checkpoint.record.artifact.relative_path,
            sha256=checkpoint.record.artifact.sha256,
            byte_size=checkpoint.record.artifact.byte_size,
            remotely_verified=False,
        )
        remote_prefix = (
            f"checkpoint-staging/{self.experiment_id}/{entry.sha256}"
        )
        immutable_revision = upload_files(
            transport=transport,
            repository=self.repository,
            revision=self.revision,
            source=self.artifact_root,
            entries=(entry,),
            remote_prefix=remote_prefix,
            max_attempts=1,
        )
        readback = Path(
            tempfile.mkdtemp(prefix="checkpoint-readback-", dir=self.artifact_root)
        )
        try:
            download_files(
                transport=transport,
                repository=self.repository,
                revision=immutable_revision,
                destination=readback,
                relative_paths=(entry.relative_path,),
                remote_prefix=remote_prefix,
                max_attempts=1,
            )
            observed = readback / entry.relative_path
            return (
                observed.is_file()
                and observed.stat().st_size == entry.byte_size
                and sha256_file(observed) == entry.sha256
            )
        finally:
            shutil.rmtree(readback, ignore_errors=True)
