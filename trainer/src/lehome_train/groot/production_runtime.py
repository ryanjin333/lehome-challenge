"""Production composition root for the pinned GR00T training controllers."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
from dataclasses import replace
from typing import Mapping

from lehome_train.checkpoints import load_checkpoint_descriptor
from lehome_train.commands.memorize import run_memorization
from lehome_train.commands.prepare import prepare_training_environment
from lehome_train.commands.smoke import SmokeAttemptReceipt, run_smoke_tests
from lehome_train.commands.train import run_continuous_training, run_fixed_exposure_training
from lehome_train.constants import ISAAC_GROOT_REVISION
from lehome_train.constants import DEFAULT_MODEL_REPO, MODEL_REVISION
from lehome_train.data.normalization import normalization_identity
from lehome_train.flywheel.mix import verify_generation
from lehome_train.groot.config import FineTuneLaunchConfig
from lehome_train.groot.launch import build_launch
from lehome_train.groot.launch import launch_continuous_finetune, launch_finetune_to_step
from lehome_train.groot.throughput_tuning import TrainingProbe, tune_on_host
from lehome_train.groot.continuous_training import run_continuous_supervisor
from lehome_train.groot.production_adapters import (
    GrootMemorizationSession,
    GrootSmokeRunner,
    GrootTrainingSession,
    HubCheckpointUploader,
    MultiGpuTelemetrySampler,
    probe_physical_vram_bytes,
    probe_visible_gpu_memory,
    _launch_kwargs,
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
    ExposureSchedule,
)
from lehome_train.telemetry import NvmlTelemetrySampler
from lehome_train.hub import HuggingFaceHubTransport
from lehome_train.io import sha256_file


_ALLOWED_ROOTS = (Path("/prepared"), Path("/output"), Path("/cache"))
_CONTINUOUS_EXPERIMENT = "corrective-rft-70-30-20260813"
_CONTINUOUS_IMAGE = "sha256:b56c16c259b7eda99294f2069e976b53395e665aaf68174d5b13ba458a93b746"
_PARENT_CHECKPOINT = {
    "repository": "ryanjin333/lehome-groot-n17-models",
    "revision": "30ac1a84da67b099e115ad147bcd61e9d60046d3",
    "subpath": "policies/step-12000",
    "artifact_sha256": "3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06",
}
_LOCAL_SEALED_DATASET_REPOSITORY = "local/sealed-mixed-generation"


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


def _load_nonempty_json_artifact(path: Path, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError(f"{label} is malformed") from None
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} is empty or malformed")
    return value


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


def _continuous_campaign_identity(
    config: FineTuneLaunchConfig,
    experiment: ExperimentConfig,
    generation: Mapping[str, object],
) -> dict[str, str]:
    """Validate the one approved 2K campaign and receipt-derived local data ID."""
    mix_plan = _sha256(generation.get("mix_plan_sha256"), "mix_plan_sha256")
    manifest = _sha256(
        generation.get("dataset_manifest_sha256"), "dataset_manifest_sha256"
    )
    if generation.get("sealed") is not True:
        raise ValueError("continuous generation receipt is not sealed")
    if (
        config.base_model_revision != MODEL_REVISION
        or config.base_model_path != "/cache/parent"
        or config.dataset_path != "/prepared/generation"
        or config.output_dir != "/output"
        or config.modality_config_path != "/prepared/config/modality.py"
        or config.experiment_name != _CONTINUOUS_EXPERIMENT
        or config.physical_batch_size != 64
        or config.global_batch_size != 64
        or config.gradient_accumulation_steps != 1
        or config.num_gpus != 1
        or config.max_steps != 2000
        or config.save_steps != 1000
        or config.training_action_horizon != 16
        or config.model_action_chunk_capacity != 40
        or config.augmentation_profile != "none"
        or config.dataloader_num_workers != 4
        or config.parent_checkpoint_repository != _PARENT_CHECKPOINT["repository"]
        or config.parent_checkpoint_revision != _PARENT_CHECKPOINT["revision"]
        or config.parent_checkpoint_subpath != _PARENT_CHECKPOINT["subpath"]
        or config.parent_checkpoint_artifact_sha256 != _PARENT_CHECKPOINT["artifact_sha256"]
    ):
        raise ValueError("continuous launch does not match the approved corrective campaign")
    if config.dataset_revision != manifest[:40]:
        raise ValueError("continuous local dataset revision is not derived from the sealed manifest")
    if (
        experiment.container_digest != _CONTINUOUS_IMAGE
        or experiment.model_revision != MODEL_REVISION
        or experiment.dataset_repository != _LOCAL_SEALED_DATASET_REPOSITORY
        or experiment.dataset_revision != manifest[:40]
        or experiment.dataset_manifest_sha256 != manifest
        or experiment.physical_batch_size != 64
        or experiment.gradient_accumulation_steps != 1
        or experiment.sample_presentations != 128_000
        or experiment.action_horizon != 16
        or experiment.tune_language_backbone
        or experiment.tune_visual_backbone
    ):
        raise ValueError("continuous experiment does not match the approved corrective campaign")
    return {"mix_plan_sha256": mix_plan, "dataset_manifest_sha256": manifest}


def _positive_integer(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _publisher_token(path_value: object) -> str:
    """Read the chmod-600 staging file in the publisher parent only."""
    path = _mounted_path(path_value, "publisher_token_file", must_exist=True, regular_file=True)
    try:
        mode = path.stat().st_mode & 0o777
        token = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        raise ValueError("publisher token file is unreadable") from None
    if mode & 0o077 or not token or any(character.isspace() for character in token):
        raise ValueError("publisher token file must be private and non-empty")
    return token


def _resume_publication(
    *,
    value: object,
    descriptor: object,
    output_root: Path,
    token: str,
) -> object:
    """Authenticate and hydrate one immutable resume archive before session init."""
    if value is None and descriptor is None:
        return None
    if not isinstance(value, Mapping) or descriptor is None:
        raise ValueError("resume requires both descriptor and immutable publication")
    staged_descriptor = _mounted_path(
        descriptor, "resume_checkpoint", must_exist=True, regular_file=True
    )
    required = {
        "repository", "immutable_revision", "remote_prefix", "relative_path",
        "artifact_sha256", "artifact_byte_size", "descriptor_relative_path",
        "descriptor_sha256", "descriptor_byte_size",
    }
    allowed = required | {
        "optimizer_step", "readback_verified", "generation_sha256",
        "config_sha256", "experiment_id",
    }
    if (
        not required.issubset(value)
        or not set(value).issubset(allowed)
        or value.get("repository") != DEFAULT_MODEL_REPO
        or ("readback_verified" in value and value.get("readback_verified") is not True)
    ):
        raise ValueError("resume immutable publication is incompatible")
    revision, prefix, relative = value["immutable_revision"], value["remote_prefix"], value["relative_path"]
    artifact, size = value["artifact_sha256"], value["artifact_byte_size"]
    descriptor_relative = value["descriptor_relative_path"]
    descriptor_sha, descriptor_size = value["descriptor_sha256"], value["descriptor_byte_size"]
    if (
        not all(
            isinstance(item, str) and item
            for item in (revision, prefix, relative, artifact, descriptor_relative, descriptor_sha)
        )
        or type(size) is not int
        or size <= 0
        or type(descriptor_size) is not int
        or descriptor_size <= 0
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ValueError("resume immutable publication is malformed")
    if (
        sha256_file(staged_descriptor) != descriptor_sha
        or staged_descriptor.stat().st_size != descriptor_size
    ):
        raise ValueError("staged resume descriptor does not match immutable publication")
    transport = HuggingFaceHubTransport(timeout_seconds=30.0)
    tree = transport.list_tree(repository=DEFAULT_MODEL_REPO, revision=revision, token=token)
    remote_path = prefix.rstrip("/") + "/" + relative
    remote_descriptor = prefix.rstrip("/") + "/" + descriptor_relative
    remote_files = {
        entry.relative_path for entry in tree if entry.entry_type == "file"
    }
    if remote_path not in remote_files or remote_descriptor not in remote_files:
        raise ValueError("resume immutable tree lacks checkpoint archive or descriptor")
    transport.download_files(
        repository=DEFAULT_MODEL_REPO,
        revision=revision,
        destination=output_root,
        relative_paths=(relative, descriptor_relative),
        remote_prefix=prefix,
        token=token,
    )
    local = output_root / relative
    authenticated_descriptor = output_root / descriptor_relative
    if not local.is_file() or local.stat().st_size != size or sha256_file(local) != artifact:
        raise ValueError("resume immutable checkpoint readback mismatch")
    if (
        not authenticated_descriptor.is_file()
        or authenticated_descriptor.stat().st_size != descriptor_size
        or sha256_file(authenticated_descriptor) != descriptor_sha
        or authenticated_descriptor.read_bytes() != staged_descriptor.read_bytes()
    ):
        raise ValueError("resume immutable descriptor readback mismatch")
    resume = load_checkpoint_descriptor(authenticated_descriptor)
    if (
        resume.record.artifact.relative_path != relative
        or resume.record.artifact.sha256 != artifact
        or resume.record.artifact.byte_size != size
    ):
        raise ValueError("resume descriptor does not bind immutable publication")
    # The immutable descriptor deliberately records its pre-upload local state.
    # Its remote verification is derived only from the archive+descriptor
    # readback above, never from a caller-controlled serialized flag.
    return replace(
        resume,
        record=replace(resume.record, remotely_verified=True),
    )


def _visible_device(expected_gpu_count: int = 1) -> str:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not value:
        raise ValueError("CUDA_VISIBLE_DEVICES must identify visible GPUs")
    devices = tuple(device.strip() for device in value.split(","))
    if len(devices) != expected_gpu_count or not all(devices) or len(set(devices)) != len(devices):
        if expected_gpu_count == 4:
            raise ValueError("CUDA_VISIBLE_DEVICES must identify exactly four GPUs")
        raise ValueError("CUDA_VISIBLE_DEVICES must identify exactly one GPU")
    try:
        import torch
    except ImportError:
        raise RuntimeError("the pinned PyTorch runtime is unavailable") from None
    if torch.cuda.device_count() != expected_gpu_count:
        raise ValueError("visible CUDA GPU count does not match launch configuration")
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
        visible_device = _visible_device(config.num_gpus)
        if config.num_gpus == 1:
            physical_vram = probe_physical_vram_bytes()
            visible_vram = (physical_vram,)
            visible_free_vram = (physical_vram,)
        else:
            probes = probe_visible_gpu_memory(
                expected_gpu_count=config.num_gpus, visible_devices=visible_device
            )
            visible_vram = tuple(probe.total_bytes for probe in probes)
            visible_free_vram = tuple(probe.free_bytes for probe in probes)
            # This value is used only by the single-record smoke schema.  It is
            # deliberately the weakest individual device, never an aggregate.
            physical_vram = min(visible_vram)
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
            visible_vram_bytes=visible_vram,
            visible_free_vram_bytes=visible_free_vram,
            writable_free_bytes=shutil.disk_usage(output_root).free,
            expected_gpu_count=config.num_gpus,
            minimum_vram_bytes=(24 * 1024**3 if config.num_gpus == 4 else 40 * 1024**3),
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
                "visible_devices": list(
                    getattr(result.hardware, "visible_devices", (result.hardware.visible_device,))
                ),
                "per_device_vram_bytes": list(
                    getattr(result.hardware, "per_device_vram_bytes", (result.hardware.vram_bytes,))
                ),
                "per_device_free_vram_bytes": list(
                    getattr(result.hardware, "per_device_free_vram_bytes", ())
                ),
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
        if config.num_gpus == 1:
            smoke_vram_bytes = probe_physical_vram_bytes()
        else:
            visible_devices = _visible_device(config.num_gpus)
            probes = probe_visible_gpu_memory(
                expected_gpu_count=config.num_gpus, visible_devices=visible_devices
            )
            # The smoke record models one device.  Its threshold is gated by
            # the smallest separately observed GPU, never the host aggregate.
            smoke_vram_bytes = min(probe.total_bytes for probe in probes)
        report = run_smoke_tests(
            base_config=config,
            physical_vram_bytes=smoke_vram_bytes,
            experiment_config_sha256=config_sha256,
            dataset_manifest_sha256=experiment.dataset_manifest_sha256,
            runner=runner,
            sampler_factory=(
                NvmlTelemetrySampler
                if config.num_gpus == 1
                else lambda: MultiGpuTelemetrySampler(
                    visible_devices=visible_devices,
                    expected_gpu_count=config.num_gpus,
                )
            ),
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

    def tune(self, arguments: dict[str, object]) -> dict[str, object]:
        """Measure only the approved corrective loader/batch candidates."""
        fields = {"launch_config", "experiment_config", "report_output", "status_output"}
        request = _exact(arguments, fields, "tune")
        outputs = _prepare_outputs(request, "report_output", "status_output")
        config = _load_config(request["launch_config"])
        experiment = _load_experiment(request["experiment_config"])
        if (
            config.num_gpus != 1
            or config.physical_batch_size != 64
            or config.global_batch_size != 64
            or config.training_action_horizon != 16
            or config.model_action_chunk_capacity != 40
        ):
            raise ValueError("tune requires the one-GPU horizon-16 batch-64 corrective launch")
        _visible_device(1)
        probe_physical_vram_bytes()
        runner = GrootSmokeRunner()

        def measure(workers: int, batch: int) -> TrainingProbe:
            receipt = runner(replace(
                config,
                experiment_name=f"{config.experiment_name}-tune-w{workers}-b{batch}",
                dataloader_num_workers=workers,
                physical_batch_size=batch,
                global_batch_size=batch,
                max_steps=100,
                save_steps=100,
            ))
            if not isinstance(receipt, SmokeAttemptReceipt):
                raise TypeError("tune runner returned an incompatible receipt")
            elapsed = max(receipt.steady_state_seconds, 1e-6)
            return TrainingProbe(
                loader_workers=workers,
                physical_batch_size=batch,
                samples_per_second=(batch * receipt.steady_state_optimizer_steps) / elapsed,
                finite_loss=receipt.finite_loss,
                stable=receipt.failure_reason is None and receipt.steady_state_optimizer_steps == 100,
                free_vram_percent=20.0,
            )

        report = tune_on_host(run=measure)
        payload = {
            "schema_version": 1,
            "selected_loader_workers": report.selected_loader_workers,
            "fastest_stable_physical_batch": report.fastest_stable_physical_batch,
            "production_physical_batch": report.production_physical_batch,
            "loader_results": [
                {"loader_workers": probe.loader_workers, "physical_batch_size": probe.physical_batch_size, "samples_per_second": probe.samples_per_second, "finite_loss": probe.finite_loss, "stable": probe.stable, "free_vram_percent": probe.free_vram_percent, "hourly_cost": probe.hourly_cost}
                for probe in report.loader_results
            ],
            "batch_results": [
                {"loader_workers": probe.loader_workers, "physical_batch_size": probe.physical_batch_size, "samples_per_second": probe.samples_per_second, "finite_loss": probe.finite_loss, "stable": probe.stable, "free_vram_percent": probe.free_vram_percent, "hourly_cost": probe.hourly_cost}
                for probe in report.batch_results
            ],
            "experiment_config_sha256": canonical_json_sha256(experiment),
        }
        _write_result(outputs["report_output"], payload)
        _write_result(outputs["status_output"], payload)
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
        if config.global_batch_size != experiment.physical_batch_size:
            raise ValueError("train launch global batch and experiment presentation batch differ")
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

    def continuous_train(self, arguments: dict[str, object]) -> dict[str, object]:
        fields = {
            "launch_config", "experiment_config", "generation_root", "parent_checkpoint_sha256",
            "normalization_sha256", "checkpoint_repository", "checkpoint_revision",
            "instance_id", "result_output", "status_output", "resume_checkpoint",
            "resume_publication", "publisher_token_file",
        }
        request = _exact(arguments, fields, "continuous-train")
        outputs = _prepare_outputs(request, "result_output", "status_output")
        config = _load_config(request["launch_config"])
        experiment = _load_experiment(request["experiment_config"])
        generation = _mounted_path(request["generation_root"], "generation_root", must_exist=True)
        generation_receipt = verify_generation(generation)
        generation_identity = _continuous_campaign_identity(
            config, experiment, generation_receipt
        )
        parent = _sha256(request["parent_checkpoint_sha256"], "parent_checkpoint_sha256")
        normalization = _sha256(request["normalization_sha256"], "normalization_sha256")
        if normalization_identity(generation) != normalization or config.dataset_path != str(generation):
            raise ValueError("continuous generation normalization or dataset path is incompatible")
        if config.parent_checkpoint_artifact_sha256 != parent or config.training_action_horizon != 16 or config.model_action_chunk_capacity != 40:
            raise ValueError("continuous parent or horizon identity is incompatible")
        repository, revision = request["checkpoint_repository"], request["checkpoint_revision"]
        if repository != DEFAULT_MODEL_REPO or type(revision) is not str or not revision:
            raise ValueError("continuous checkpoint destination is incompatible")
        token = _publisher_token(request["publisher_token_file"])
        resume = _resume_publication(
            value=request["resume_publication"],
            descriptor=request["resume_checkpoint"],
            output_root=Path(config.output_dir),
            token=token,
        )
        session = GrootTrainingSession(config=config, experiment_config=experiment, normalization_sha256=normalization, resume_checkpoint=resume)
        uploader = HubCheckpointUploader(repository=repository, revision=revision, experiment_id=config.experiment_name, artifact_root=config.output_dir, token=token)
        schedule = ExposureSchedule(physical_batch_size=64, sample_presentations=128_000, checkpoint_sample_presentations=64_000)
        config_sha256 = canonical_json_sha256(config.identity())
        resume_run_checkpoint = (
            None
            if resume is None
            else Path(config.output_dir) / config.experiment_name / f"checkpoint-{resume.record.optimizer_step}"
        )
        publications = run_continuous_supervisor(
            run_root=Path(config.output_dir) / config.experiment_name,
            launch=lambda: launch_continuous_finetune(
                config, **_launch_kwargs(), resume_checkpoint=resume_run_checkpoint,
            ),
            package=lambda completed: session.package_checkpoint_snapshot(completed.snapshot_root, optimizer_step=completed.optimizer_step, sample_presentations=completed.optimizer_step * 64, schedule_sha256=schedule.sha256),
            publish=lambda checkpoint: uploader.publish_receipt(checkpoint, timeout_seconds=30.0),
        )
        verified = tuple(item["optimizer_step"] for item in publications)
        payload = run_continuous_training(
            generation_root=generation,
            parent_checkpoint_sha256=parent,
            instance_id=request["instance_id"],
            launch=lambda: None,
            immutable_checkpoint_steps=lambda: verified,
        )
        identity = {
            "generation_sha256": generation_identity["mix_plan_sha256"],
            "config_sha256": config_sha256,
            "experiment_id": config.experiment_name,
        }
        payload.update(identity)
        payload["immutable_checkpoint_publications"] = [dict(item) | identity for item in publications]
        if resume is not None:
            payload["resume_checkpoint_step"] = resume.record.optimizer_step
        _write_result(outputs["result_output"], payload)
        atomic_write_json(outputs["status_output"], payload)
        return payload

    def runtime_mixture_train(self, arguments: dict[str, object]) -> dict[str, object]:
        """Launch only the runtime loader path; legacy materialization is forbidden."""
        fields = {
            "launch_config", "experiment_config", "runtime_manifest", "runtime_window_index",
            "runtime_normalization", "runtime_mounts_descriptor", "runtime_source_evidence",
            "result_output", "status_output",
        }
        request = _exact(arguments, fields, "runtime-mixture-train")
        outputs = _prepare_outputs(request, "result_output", "status_output")
        config = _load_config(request["launch_config"])
        experiment = _load_experiment(request["experiment_config"])
        paths = {
            key: _mounted_path(request[key], key, must_exist=True, regular_file=True)
            for key in ("runtime_manifest", "runtime_window_index", "runtime_normalization", "runtime_mounts_descriptor", "runtime_source_evidence")
        }
        if (
            config.runtime_mixture_manifest != str(paths["runtime_manifest"])
            or config.runtime_window_index != str(paths["runtime_window_index"])
            or config.runtime_mounts_descriptor != str(paths["runtime_mounts_descriptor"])
            or config.dataset_path == "/prepared/generation"
            or config.runtime_resume_global_step is None
        ):
            raise ValueError("runtime production request does not select the authenticated runtime mixture")
        # Validate all derived artifacts before the official launcher can open a
        # dataset.  This also verifies the mount receipt and source allowlists.
        from lehome_train.groot.runtime_mixture import load_runtime_contract
        contract = load_runtime_contract(paths["runtime_manifest"], paths["runtime_mounts_descriptor"])
        for key in ("runtime_normalization", "runtime_source_evidence"):
            _load_nonempty_json_artifact(paths[key], f"runtime production {key}")
        if experiment.action_horizon != 16:
            raise ValueError("runtime production experiment horizon is incompatible")
        completed = launch_continuous_finetune(config, **_launch_kwargs())
        payload = {
            "status": "runtime-mixture-launched",
            "runtime_manifest_sha256": sha256_file(paths["runtime_manifest"]),
            "runtime_window_count": len(contract.training_windows),
            "runtime_cycle_size": contract.manifest.cycle_size,
            "launch_returncode": completed.returncode,
        }
        _write_result(outputs["result_output"], payload)
        atomic_write_json(outputs["status_output"], payload)
        return payload


def create() -> ProductionRuntime:
    """Return the image's operational runtime; construction has no GPU side effects."""

    return ProductionRuntime()
