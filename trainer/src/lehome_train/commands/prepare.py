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
    pending_stages,
)
from lehome_train.models import ArtifactIdentity
from lehome_train.preflight import (
    HardwareReport,
    PREFLIGHT_STAGE_NAMES,
    check_hardware,
    verify_hub_write_permission,
    verify_immutable_revision,
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
    hub_permission_check: Callable[[str], bool],
    stage_operations: Mapping[str, Callable[[], None]],
    expected_model_revision: str | None = None,
    observed_model_revision: str | None = None,
    expected_dataset_revision: str | None = None,
    observed_dataset_revision: str | None = None,
) -> PrepareResult:
    """Run only missing preparation stages after all no-cost gates pass.

    Tokens are deliberately passed only to the caller-provided permission
    callback. They are not placed in configuration, logs, status, or child
    process environments.
    """

    if tuple(stage_operations) != PREFLIGHT_STAGE_NAMES:
        raise ValueError("prepare operations must use the complete canonical stage order")
    hardware = check_hardware(
        visible_devices=visible_devices,
        visible_vram_bytes=visible_vram_bytes,
        writable_free_bytes=writable_free_bytes,
    )
    model_revision = _config_revision(resolved_config, "model_revision")
    dataset_revision = _config_revision(resolved_config, "dataset_revision")
    verify_immutable_revision(
        expected_revision=(
            model_revision if expected_model_revision is None else expected_model_revision
        ),
        observed_revision=(
            model_revision if observed_model_revision is None else observed_model_revision
        ),
        label="model",
    )
    verify_immutable_revision(
        expected_revision=(
            dataset_revision if expected_dataset_revision is None else expected_dataset_revision
        ),
        observed_revision=(
            dataset_revision if observed_dataset_revision is None else observed_dataset_revision
        ),
        label="dataset",
    )
    verify_hub_write_permission(token=token, permission_check=hub_permission_check)
    experiment = create_or_resume_experiment(
        output_root,
        resolved_config=resolved_config,
        artifacts=artifacts,
    )
    completed: list[str] = []
    for stage in pending_stages(experiment):
        started = monotonic()
        stage_operations[stage]()
        mark_stage_complete(experiment, stage, duration_seconds=monotonic() - started)
        completed.append(stage)
    return PrepareResult(
        experiment=experiment,
        hardware=hardware,
        completed_stages=tuple(completed),
    )
