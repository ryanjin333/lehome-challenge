"""Restart-safe orchestration of the paid-machine preparation gate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Callable, Mapping, Sequence

from lehome_train.experiment import (
    Experiment,
    create_or_resume_experiment,
    mark_stage_complete,
    mark_stage_failed,
    pending_stages,
)
from lehome_train.models import ArtifactIdentity
from lehome_train.preflight import (
    HardwareReport,
    HubPermission,
    HubTarget,
    PREFLIGHT_STAGE_NAMES,
    check_hardware,
    reject_secret_bearing_config,
    verify_hub_upload_readback_permission,
    verify_snapshot_manifest,
)


@dataclass(frozen=True, slots=True)
class PrepareResult:
    """Safe-to-train preparation facts, excluding any credential value."""

    experiment: Experiment
    hardware: HardwareReport
    completed_stages: tuple[str, ...]


def _config_revision(config: Mapping[str, object], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str):
        raise ValueError(f"resolved config lacks {key}")
    return value


def prepare_training_environment(
    *,
    output_root: str | Path,
    resolved_config: Mapping[str, object],
    artifacts: Sequence[ArtifactIdentity],
    visible_devices: str | None,
    visible_vram_bytes: Sequence[int],
    writable_free_bytes: int,
    token: str | None,
    hub_targets: Sequence[HubTarget],
    hub_permission_check: Callable[[str, str, str], HubPermission],
    stage_operations: Mapping[str, Callable[[Path], Sequence[ArtifactIdentity]]],
    model_snapshot_root: str | Path,
    model_snapshot_manifest: str | Path,
    dataset_snapshot_root: str | Path,
    dataset_snapshot_manifest: str | Path,
) -> PrepareResult:
    """Run only missing preparation stages after all no-cost gates pass.

    Tokens are deliberately passed only to the caller-provided permission
    callback. They are not placed in configuration, logs, status, or child
    process environments.
    """

    if tuple(stage_operations) != PREFLIGHT_STAGE_NAMES:
        raise ValueError("prepare operations must use the complete canonical stage order")
    reject_secret_bearing_config(dict(resolved_config))
    hardware = check_hardware(
        visible_devices=visible_devices,
        visible_vram_bytes=visible_vram_bytes,
        writable_free_bytes=writable_free_bytes,
    )
    model_revision = _config_revision(resolved_config, "model_revision")
    dataset_revision = _config_revision(resolved_config, "dataset_revision")
    model_snapshot = verify_snapshot_manifest(
        snapshot_root=model_snapshot_root,
        manifest_path=model_snapshot_manifest,
        expected_revision=model_revision,
        label="model",
    )
    dataset_snapshot = verify_snapshot_manifest(
        snapshot_root=dataset_snapshot_root,
        manifest_path=dataset_snapshot_manifest,
        expected_revision=dataset_revision,
        label="dataset",
    )
    if not hub_targets:
        raise ValueError("at least one configured private Hub target is required")
    configured_targets = {
        (_config_revision(resolved_config, "dataset_repository"), dataset_revision),
    }
    artifact_repository = resolved_config.get("artifact_repository")
    artifact_revision = resolved_config.get("artifact_revision")
    if artifact_repository is not None or artifact_revision is not None:
        configured_targets.add(
            (
                _config_revision(resolved_config, "artifact_repository"),
                _config_revision(resolved_config, "artifact_revision"),
            )
        )
    else:
        configured_targets.add(
            (_config_revision(resolved_config, "model_repository"), model_revision)
        )
    if any((target.repository, target.revision) not in configured_targets for target in hub_targets):
        raise ValueError("Hub permission target must match a configured repository and revision")
    for target in hub_targets:
        verify_hub_upload_readback_permission(
            token=token,
            target=target,
            permission_check=hub_permission_check,
        )
    identity_config = dict(resolved_config)
    identity_config["model_snapshot_manifest_sha256"] = model_snapshot.manifest_sha256
    identity_config["dataset_snapshot_manifest_sha256"] = dataset_snapshot.manifest_sha256
    experiment = create_or_resume_experiment(
        output_root,
        resolved_config=identity_config,
        artifacts=artifacts,
    )
    completed: list[str] = []
    for stage in pending_stages(experiment):
        started = monotonic()
        try:
            artifacts_for_stage = tuple(stage_operations[stage](experiment.root))
            mark_stage_complete(
                experiment,
                stage,
                duration_seconds=monotonic() - started,
                artifacts=artifacts_for_stage,
            )
        except Exception as error:
            mark_stage_failed(
                experiment,
                stage,
                duration_seconds=monotonic() - started,
                error=error,
            )
            raise RuntimeError(f"preflight stage {stage} failed safely") from None
        completed.append(stage)
    return PrepareResult(
        experiment=experiment,
        hardware=hardware,
        completed_stages=tuple(completed),
    )
