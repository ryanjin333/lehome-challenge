"""Production composition root for the pinned GR00T training controllers."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
from typing import Mapping

from lehome_train.checkpoints import load_checkpoint_descriptor
from lehome_train.commands.memorize import run_memorization
from lehome_train.commands.prepare import prepare_training_environment
from lehome_train.commands.smoke import run_smoke_tests
from lehome_train.commands.train import run_fixed_exposure_training
from lehome_train.constants import ISAAC_GROOT_REVISION
from lehome_train.constants import DEFAULT_MODEL_REPO
from lehome_train.data.normalization import normalization_identity
from lehome_train.groot.config import FineTuneLaunchConfig
from lehome_train.groot.launch import build_launch
from lehome_train.groot.launch import launch_finetune_to_step
from lehome_train.groot.production_adapters import (
    GrootMemorizationSession,
    GrootSmokeRunner,
    GrootTrainingSession,
    HubCheckpointUploader,
    probe_physical_vram_bytes,
)
from lehome_train.io import (
    atomic_write_json,
    canonical_json_bytes,
    canonical_json_sha256,
    load_json,
)
from lehome_train.models import ArtifactIdentity, ExperimentConfig, SmokeResult
from lehome_train.preflight import HubPermission, HubTarget, PREFLIGHT_STAGE_NAMES
from lehome_train.preflight import reject_secret_bearing_config
from lehome_train.schedule import (
    CHECKPOINT_SAMPLE_PRESENTATIONS,
    TOTAL_SAMPLE_PRESENTATIONS,
    optimizer_steps_for_presentations,
)
from lehome_train.telemetry import NvmlTelemetrySampler
from lehome_train.hub import HuggingFaceHubTransport
from lehome_train.io import sha256_file


_ALLOWED_ROOTS = (Path("/prepared"), Path("/output"), Path("/cache"))


def _exact(arguments: object, fields: set[str], command: str) -> Mapping[str, object]:
    if not isinstance(arguments, Mapping) or set(arguments) != fields:
        raise ValueError(f"production {command} request has an incompatible schema")
    if not all(type(key) is str for key in arguments):
        raise ValueError(f"production {command} request keys must be strings")
    reject_secret_bearing_config(dict(arguments))
    return arguments


def _mounted_path(
    value: object,
    label: str,
    *,
    must_exist: bool = False,
    regular_file: bool = False,
) -> Path:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be an absolute mounted path")
    path = Path(value)
    if ".." in path.parts or not path.is_absolute() or path.is_symlink():
        raise ValueError(f"{label} must be an absolute mounted path without aliases")
    resolved = path.resolve(strict=False)
    roots = tuple(root.resolve(strict=False) for root in _ALLOWED_ROOTS)
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise ValueError(f"{label} must stay beneath /cache, /prepared, or /output")
    if must_exist:
        if regular_file and not path.is_file():
            raise ValueError(f"{label} must be an existing regular file")
        if not regular_file and not path.exists():
            raise ValueError(f"{label} must exist")
    return path


def _load_config(path_value: object) -> FineTuneLaunchConfig:
    path = _mounted_path(
        path_value,
        "launch_config",
        must_exist=True,
        regular_file=True,
    )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("launch_config is malformed") from None
    if not isinstance(raw, dict):
        raise ValueError("launch_config root must be an object")
    reject_secret_bearing_config(raw)
    try:
        config = FineTuneLaunchConfig(**raw)
    except (TypeError, ValueError):
        raise ValueError("launch_config has an incompatible schema") from None
    for candidate in (
        config.base_model_path,
        config.dataset_path,
        config.modality_config_path,
        config.output_dir,
    ):
        _mounted_path(candidate, "launch_config path")
    return config


def _load_experiment(path_value: object) -> ExperimentConfig:
    path = _mounted_path(
        path_value,
        "experiment_config",
        must_exist=True,
        regular_file=True,
    )
    return load_json(ExperimentConfig, path)


def _load_smoke(path_value: object) -> SmokeResult:
    path = _mounted_path(
        path_value,
        "selected_smoke_result",
        must_exist=True,
        regular_file=True,
    )
    return load_json(SmokeResult, path)


def _optional_string(value: object, label: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a non-empty string or null")
    return value


def _finite_nonnegative_number(value: object, label: str) -> float | None:
    if value is None:
        return None
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a nonnegative number or null")
    number = float(value)
    if not __import__("math").isfinite(number) or number < 0:
        raise ValueError(f"{label} must be a nonnegative number or null")
    return number


def _sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _positive_integer(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _visible_device() -> str:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not value:
        raise ValueError("CUDA_VISIBLE_DEVICES must identify exactly one GPU")
    try:
        import torch
    except ImportError:
        raise RuntimeError("the pinned PyTorch runtime is unavailable") from None
    if torch.cuda.device_count() != 1:
        raise ValueError("exactly one CUDA GPU must be visible")
    return value


def _prepare_output(path_value: object, label: str) -> Path:
    path = _mounted_path(path_value, label)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise ValueError(f"{label} output parent cannot be created") from None
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError(f"{label} output parent is not a regular directory")
    probe = path.parent / f".{path.name}.write-probe"
    try:
        probe.write_bytes(b"")
        probe.unlink()
    except OSError:
        raise ValueError(f"{label} output parent is not writable") from None
    return path


def _prepare_outputs(request: Mapping[str, object], *names: str) -> dict[str, Path]:
    return {name: _prepare_output(request[name], name) for name in names}


def _write_result(path: Path, value: object) -> dict[str, object]:
    payload = value.to_dict() if hasattr(value, "to_dict") else value
    if not isinstance(payload, Mapping) or not all(type(key) is str for key in payload):
        raise TypeError("production controller returned an invalid result")
    detached = dict(payload)
    reject_secret_bearing_config(detached)
    atomic_write_json(path, detached)
    return detached


def _write_safe_json_artifact(path: Path, payload: Mapping[str, object]) -> None:
    """Write one uploadable artifact after applying the central secret policy."""

    detached = dict(payload)
    reject_secret_bearing_config(detached)
    if path.exists() and (not path.is_file() or path.is_symlink()):
        raise ValueError("production artifact destination must be a regular file")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError("production artifact parent must be a regular directory")
    atomic_write_json(path, detached)


def _write_immutable_json_artifact(path: Path, payload: Mapping[str, object]) -> None:
    """Create or verify an immutable root artifact without silent replacement."""

    detached = dict(payload)
    reject_secret_bearing_config(detached)
    if path.exists():
        if not path.is_file() or path.is_symlink():
            raise ValueError("production artifact destination must be a regular file")
        try:
            existing = path.read_bytes()
        except OSError:
            raise ValueError("production artifact destination is unreadable") from None
        if existing != canonical_json_bytes(detached):
            raise ValueError("production artifact identity is incompatible")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError("production artifact parent must be a regular directory")
    atomic_write_json(path, detached)


def _record_stage(root: Path, name: str, payload: Mapping[str, object]) -> tuple[ArtifactIdentity, ...]:
    directory = root / "stage-records"
    directory.mkdir(exist_ok=True)
    path = directory / f"{name}.json"
    atomic_write_json(path, dict(payload))
    return (
        ArtifactIdentity(
            relative_path=path.relative_to(root).as_posix(),
            sha256=sha256_file(path),
            byte_size=path.stat().st_size,
        ),
    )


def _network_measurement(path_value: object) -> dict[str, object]:
    path = _mounted_path(
        path_value,
        "network_measurement",
        must_exist=True,
        regular_file=True,
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("network measurement evidence is malformed") from None
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "downloaded_bytes",
        "duration_seconds",
    }:
        raise ValueError("network measurement evidence has an incompatible schema")
    downloaded = value["downloaded_bytes"]
    duration = value["duration_seconds"]
    if (
        value["schema_version"] != 1
        or type(downloaded) is not int
        or downloaded <= 0
        or type(duration) not in (int, float)
        or not math.isfinite(float(duration))
        or duration <= 0
    ):
        raise ValueError("network measurement evidence is invalid")
    gigabits_per_second = downloaded * 8 / float(duration) / 1_000_000_000
    if gigabits_per_second < 1.0:
        raise ValueError("network measurement is below 1 Gbps")
    return {
        "schema_version": 1,
        "downloaded_bytes": downloaded,
        "duration_seconds": float(duration),
        "gigabits_per_second": gigabits_per_second,
    }


class ProductionRuntime:
    """Checked image runtime that delegates policy to Tasks 8-10 controllers."""

    def prepare(self, arguments: dict[str, object]) -> dict[str, object]:
        fields = {
            "launch_config",
            "experiment_config",
            "model_snapshot_manifest",
            "dataset_snapshot_manifest",
            "network_measurement",
            "artifact_repository",
            "artifact_revision",
            "status_output",
        }
        request = _exact(arguments, fields, "prepare")
        outputs = _prepare_outputs(request, "status_output")
        config = _load_config(request["launch_config"])
        experiment = _load_experiment(request["experiment_config"])
        model_manifest = _mounted_path(
            request["model_snapshot_manifest"],
            "model_snapshot_manifest",
            must_exist=True,
            regular_file=True,
        )
        dataset_manifest = _mounted_path(
            request["dataset_snapshot_manifest"],
            "dataset_snapshot_manifest",
            must_exist=True,
            regular_file=True,
        )
        network = _network_measurement(request["network_measurement"])
        artifact_repository = request["artifact_repository"]
        artifact_revision = request["artifact_revision"]
        if (
            artifact_repository != DEFAULT_MODEL_REPO
            or type(artifact_revision) is not str
            or len(artifact_revision) != 40
            or any(character not in "0123456789abcdef" for character in artifact_revision)
        ):
            raise ValueError("prepare artifact destination identity is incompatible")
        normalization_sha256 = normalization_identity(config.dataset_path)
        visible_device = _visible_device()
        physical_vram = probe_physical_vram_bytes()
        output_root = Path(config.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        token = os.environ.get("HF_TOKEN")
        transport = HuggingFaceHubTransport(timeout_seconds=30.0)

        def permission_check(
            supplied_token: str, repository: str, _revision: str
        ) -> HubPermission:
            access = transport.check_access(
                repository=repository,
                token=supplied_token,
            )
            return HubPermission(
                can_upload=access.can_write,
                can_readback=access.can_read,
                private_repository=access.private_repository,
            )

        resolved_config = {
            **experiment.to_dict(),
            "artifact_repository": artifact_repository,
            "artifact_revision": artifact_revision,
            "launch_config": config.identity(),
            "normalization_sha256": normalization_sha256,
        }
        identity_artifacts = (
            ArtifactIdentity(
                "inputs/model-snapshot.json",
                sha256_file(model_manifest),
                model_manifest.stat().st_size,
            ),
            ArtifactIdentity(
                "inputs/dataset-snapshot.json",
                sha256_file(dataset_manifest),
                dataset_manifest.stat().st_size,
            ),
        )

        def image_runtime(root: Path) -> tuple[ArtifactIdentity, ...]:
            launch = build_launch(
                config,
                visible_devices=visible_device,
                environment=os.environ,
                official_checkout=os.environ.get("LEHOME_GROOT_ROOT", "/opt/isaac-groot"),
            )
            return _record_stage(root, "image_runtime_verification", {"command": list(launch.command)})

        def network_stage(root: Path) -> tuple[ArtifactIdentity, ...]:
            return _record_stage(root, "network_measurement", network)

        def model_stage(root: Path) -> tuple[ArtifactIdentity, ...]:
            return _record_stage(
                root,
                "model_download",
                {"repository": experiment.model_repository, "revision": experiment.model_revision},
            )

        def dataset_stage(root: Path) -> tuple[ArtifactIdentity, ...]:
            return _record_stage(
                root,
                "dataset_download",
                {"repository": experiment.dataset_repository, "revision": experiment.dataset_revision},
            )

        def validation_stage(root: Path) -> tuple[ArtifactIdentity, ...]:
            from lehome_train.data.validate import validate_prepared_dataset

            report = validate_prepared_dataset(
                config.dataset_path,
                groot_root=os.environ.get("LEHOME_GROOT_ROOT", "/opt/isaac-groot"),
            )
            if report.get("valid") is not True or report.get("dataset_manifest_sha256") != experiment.dataset_manifest_sha256:
                raise ValueError("prepared dataset validation evidence is incompatible")
            return _record_stage(root, "schema_hash_validation", report)

        def initialize_stage(root: Path) -> tuple[ArtifactIdentity, ...]:
            initialization = __import__("dataclasses").replace(
                config,
                output_dir=str(root),
            )
            launch_finetune_to_step(
                initialization,
                stop_after_optimizer_step=0,
                visible_devices=visible_device,
                environment=os.environ,
                official_checkout=os.environ.get("LEHOME_GROOT_ROOT", "/opt/isaac-groot"),
            )
            return _record_stage(root, "model_initialization", {"initialized": True})

        stage_operations = dict(
            zip(
                PREFLIGHT_STAGE_NAMES,
                (
                    image_runtime,
                    network_stage,
                    model_stage,
                    dataset_stage,
                    validation_stage,
                    initialize_stage,
                ),
                strict=True,
            )
        )
        result = prepare_training_environment(
            output_root=output_root,
            resolved_config=resolved_config,
            artifacts=identity_artifacts,
            visible_devices=visible_device,
            visible_vram_bytes=(physical_vram,),
            writable_free_bytes=shutil.disk_usage(output_root).free,
            token=token,
            hub_targets=(
                HubTarget(experiment.dataset_repository, experiment.dataset_revision),
                HubTarget(artifact_repository, artifact_revision),
            ),
            hub_permission_check=permission_check,
            stage_operations=stage_operations,
            model_snapshot_root=config.base_model_path,
            model_snapshot_manifest=model_manifest,
            dataset_snapshot_root=config.dataset_path,
            dataset_snapshot_manifest=dataset_manifest,
        )
        experiment_config_payload = experiment.to_dict()
        provenance = {
            "schema_version": 1,
            "experiment_id": config.experiment_name,
            "preflight_experiment_id": result.experiment.experiment_id,
            "experiment_config_sha256": canonical_json_sha256(experiment),
            "repository_commit": experiment.repository_commit,
            "container_digest": experiment.container_digest,
            "isaac_groot_revision": ISAAC_GROOT_REVISION,
            "model_repository": experiment.model_repository,
            "model_revision": experiment.model_revision,
            "model_snapshot_manifest_sha256": sha256_file(model_manifest),
            "dataset_repository": experiment.dataset_repository,
            "dataset_revision": experiment.dataset_revision,
            "dataset_manifest_sha256": experiment.dataset_manifest_sha256,
            "dataset_snapshot_manifest_sha256": sha256_file(dataset_manifest),
            "normalization_sha256": normalization_sha256,
            "artifact_repository": artifact_repository,
            "artifact_revision": artifact_revision,
        }
        _write_immutable_json_artifact(
            output_root / "resolved-config.json", experiment_config_payload
        )
        _write_immutable_json_artifact(output_root / "provenance.json", provenance)
        _write_safe_json_artifact(
            output_root / "logs" / "prepare.json",
            {
                "schema_version": 1,
                "event": "prepared",
                "experiment_id": config.experiment_name,
                "preflight_experiment_id": result.experiment.experiment_id,
                "normalization_sha256": normalization_sha256,
            },
        )
        payload = {
            "schema_version": 1,
            "status": "prepared",
            "isaac_groot_revision": ISAAC_GROOT_REVISION,
            "experiment_id": result.experiment.experiment_id,
            "experiment_root": str(result.experiment.root),
            "normalization_sha256": normalization_sha256,
            "completed_stages": list(result.completed_stages),
            "hardware": {
                "visible_device": result.hardware.visible_device,
                "vram_bytes": result.hardware.vram_bytes,
                "writable_free_bytes": result.hardware.writable_free_bytes,
            },
        }
        atomic_write_json(outputs["status_output"], payload)
        return payload

    def memorize(self, arguments: dict[str, object]) -> dict[str, object]:
        fields = {
            "launch_config",
            "experiment_config",
            "dataset_manifest_sha256",
            "requested_episode_id",
            "result_output",
            "status_output",
        }
        request = _exact(arguments, fields, "memorize")
        outputs = _prepare_outputs(request, "result_output", "status_output")
        config = _load_config(request["launch_config"])
        experiment = _load_experiment(request["experiment_config"])
        dataset_sha256 = _sha256(
            request["dataset_manifest_sha256"], "dataset_manifest_sha256"
        )
        requested_episode_id = _optional_string(
            request["requested_episode_id"], "requested_episode_id"
        )
        if config.physical_batch_size != 1 or config.max_steps != 10_000:
            raise ValueError("memorize requires physical batch 1 and exactly 10,000 steps")
        if experiment.physical_batch_size != 1:
            raise ValueError("memorize experiment config must use physical batch 1")
        if experiment.dataset_manifest_sha256 != dataset_sha256:
            raise ValueError("memorize dataset manifest identity is incompatible")
        session = GrootMemorizationSession(config=config)
        result = run_memorization(
            dataset_path=config.dataset_path,
            experiment_id=config.experiment_name,
            experiment_config_sha256=canonical_json_sha256(experiment),
            dataset_manifest_sha256=dataset_sha256,
            trainer=session.train_chunk,
            evaluator=session.evaluate,
            checkpointer=session.verify_checkpoint,
            requested_episode_id=requested_episode_id,
        )
        payload = _write_result(outputs["result_output"], result)
        atomic_write_json(outputs["status_output"], payload)
        return payload

    def smoke(self, arguments: dict[str, object]) -> dict[str, object]:
        fields = {
            "launch_config",
            "experiment_config",
            "report_output",
            "selected_result_output",
            "status_output",
        }
        request = _exact(arguments, fields, "smoke")
        outputs = _prepare_outputs(
            request,
            "report_output",
            "selected_result_output",
            "status_output",
        )
        config = _load_config(request["launch_config"])
        experiment = _load_experiment(request["experiment_config"])
        config_sha256 = canonical_json_sha256(experiment)
        runner = GrootSmokeRunner()
        report = run_smoke_tests(
            base_config=config,
            physical_vram_bytes=probe_physical_vram_bytes(),
            experiment_config_sha256=config_sha256,
            dataset_manifest_sha256=experiment.dataset_manifest_sha256,
            runner=runner,
            sampler_factory=NvmlTelemetrySampler,
        )
        report_payload = _write_result(outputs["report_output"], report)
        selected = next(
            (
                attempt.result
                for attempt in report.attempts
                if attempt.result.physical_batch_size == report.selected_batch_size
            ),
            None,
        )
        if selected is None:
            failure = {
                "schema_version": 1,
                "status": "no_stable_batch",
                "report": report_payload,
            }
            atomic_write_json(outputs["status_output"], failure)
            raise RuntimeError("smoke found no stable physical batch with required headroom")
        _write_result(outputs["selected_result_output"], selected)
        payload = {
            "schema_version": 1,
            "status": "smoke_completed",
            "selected_batch_size": report.selected_batch_size,
            "report_output": str(outputs["report_output"]),
            "selected_result_output": str(outputs["selected_result_output"]),
        }
        atomic_write_json(outputs["status_output"], payload)
        return payload

    def train(self, arguments: dict[str, object]) -> dict[str, object]:
        fields = {
            "launch_config",
            "experiment_config",
            "selected_smoke_result",
            "normalization_sha256",
            "estimated_checkpoint_bytes",
            "checkpoint_repository",
            "checkpoint_revision",
            "resume_checkpoint",
            "provider_hourly_price",
            "instance_start_time",
            "result_output",
            "status_output",
        }
        request = _exact(arguments, fields, "train")
        outputs = _prepare_outputs(request, "result_output", "status_output")
        config = _load_config(request["launch_config"])
        experiment = _load_experiment(request["experiment_config"])
        selected_smoke = _load_smoke(request["selected_smoke_result"])
        requested_normalization_sha256 = _sha256(
            request["normalization_sha256"], "normalization_sha256"
        )
        normalization_sha256 = normalization_identity(config.dataset_path)
        if requested_normalization_sha256 != normalization_sha256:
            raise ValueError("train normalization identity is incompatible")
        estimated_bytes = _positive_integer(
            request["estimated_checkpoint_bytes"], "estimated_checkpoint_bytes"
        )
        repository = request["checkpoint_repository"]
        revision = request["checkpoint_revision"]
        if type(repository) is not str or type(revision) is not str or not revision:
            raise ValueError("checkpoint upload destination is invalid")
        resume_path = request["resume_checkpoint"]
        if resume_path is not None:
            resume_path = _mounted_path(
                resume_path,
                "resume_checkpoint",
                must_exist=True,
                regular_file=True,
            )
        resume = None if resume_path is None else load_checkpoint_descriptor(resume_path)
        provider_price = _finite_nonnegative_number(
            request["provider_hourly_price"], "provider_hourly_price"
        )
        instance_start_time = _optional_string(
            request["instance_start_time"], "instance_start_time"
        )
        if experiment.sample_presentations != TOTAL_SAMPLE_PRESENTATIONS:
            raise ValueError("train requires exactly 768000 sample presentations")
        if config.physical_batch_size != experiment.physical_batch_size:
            raise ValueError("train launch and experiment physical batches differ")
        expected_steps = optimizer_steps_for_presentations(
            TOTAL_SAMPLE_PRESENTATIONS,
            experiment.physical_batch_size,
        )
        expected_save_steps = optimizer_steps_for_presentations(
            CHECKPOINT_SAMPLE_PRESENTATIONS,
            experiment.physical_batch_size,
        )
        if config.max_steps != expected_steps or config.save_steps != expected_save_steps:
            raise ValueError("train launch does not match the fixed exposure schedule")
        session = GrootTrainingSession(
            config=config,
            experiment_config=experiment,
            normalization_sha256=normalization_sha256,
            resume_checkpoint=resume,
        )
        uploader = HubCheckpointUploader(
            repository=repository,
            revision=revision,
            experiment_id=selected_smoke.experiment_id,
            artifact_root=config.output_dir,
        )
        result = run_fixed_exposure_training(
            experiment_config=experiment,
            selected_smoke=selected_smoke,
            normalization_sha256=normalization_sha256,
            runner=session.run_chunk,
            checkpointer=session.package_checkpoint,
            uploader=uploader,
            disk_probe=session.disk_free_bytes,
            estimated_checkpoint_bytes=estimated_bytes,
            checkpoint_deleter=session.delete_checkpoint_archive,
            resume_checkpoint=resume,
            provider_hourly_price=provider_price,
            instance_start_time=instance_start_time,
            status_path=outputs["status_output"],
        )
        payload = _write_result(outputs["result_output"], result)
        training_log = {
            "schema_version": 1,
            "event": "training_terminal",
            "experiment_id": selected_smoke.experiment_id,
            "experiment_config_sha256": canonical_json_sha256(experiment),
            "normalization_sha256": normalization_sha256,
            "status": payload.get("status"),
            "sample_presentations": payload.get("sample_presentations"),
        }
        _write_safe_json_artifact(
            Path(config.output_dir) / "logs" / "train.json", training_log
        )
        return payload


def create() -> ProductionRuntime:
    """Return the image's operational runtime; construction has no GPU side effects."""

    return ProductionRuntime()
