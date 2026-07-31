"""Complete, secret-safe provenance and cost reports for one training run."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import re
from pathlib import Path

from lehome_train.checkpoints import CheckpointDescriptor
from lehome_train.io import atomic_write_json, canonical_json_sha256
from lehome_train.models import ArtifactIdentity, ExperimentConfig, SmokeResult
from lehome_train.preflight import reject_secret_bearing_config


_COMMIT = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True, slots=True)
class TrainingReport:
    """Auditable identity, throughput, artifact, runtime, and cost facts."""

    experiment_id: str
    generated_at: str
    instance_started_at: str
    container_image_digest: str
    repository_commit: str
    isaac_groot_revision: str
    base_model_repository: str
    base_model_revision: str
    dataset_repository: str
    dataset_revision: str
    dataset_manifest_sha256: str
    resolved_training_config: dict[str, object]
    resolved_training_config_sha256: str
    smoke_metrics: dict[str, object]
    checkpoint_artifacts: tuple[ArtifactIdentity, ...]
    runtime_seconds: float
    provider_hourly_price: float
    cost: float

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "experiment_id": self.experiment_id,
            "generated_at": self.generated_at,
            "instance_started_at": self.instance_started_at,
            "container_image_digest": self.container_image_digest,
            "repository_commit": self.repository_commit,
            "isaac_groot_revision": self.isaac_groot_revision,
            "base_model": {
                "repository": self.base_model_repository,
                "revision": self.base_model_revision,
            },
            "prepared_dataset": {
                "repository": self.dataset_repository,
                "revision": self.dataset_revision,
                "manifest_sha256": self.dataset_manifest_sha256,
            },
            "resolved_training_config": dict(self.resolved_training_config),
            "resolved_training_config_sha256": self.resolved_training_config_sha256,
            "smoke_metrics": dict(self.smoke_metrics),
            "checkpoint_artifacts": [
                artifact.to_dict() for artifact in self.checkpoint_artifacts
            ],
            "runtime_seconds": self.runtime_seconds,
            "provider_hourly_price": self.provider_hourly_price,
            "cost": self.cost,
        }


def _parse_timestamp(value: str, *, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be an explicit timezone-aware timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError:
        raise ValueError(f"{label} must be an explicit timezone-aware timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must be an explicit timezone-aware timestamp")
    return parsed


def _require_immutable_revision(value: str, *, label: str) -> None:
    if not isinstance(value, str) or not _COMMIT.fullmatch(value):
        raise ValueError(f"{label} must be an immutable 40-character commit revision")


def _validate_checkpoints(
    checkpoints: tuple[CheckpointDescriptor, ...],
    *,
    experiment_id: str,
    config_sha256: str,
    dataset_manifest_sha256: str,
) -> tuple[ArtifactIdentity, ...]:
    if not checkpoints:
        raise ValueError("training report requires at least one checkpoint")
    artifacts: list[ArtifactIdentity] = []
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, CheckpointDescriptor):
            raise TypeError("training report checkpoints must be CheckpointDescriptor values")
        record = checkpoint.record
        if record.experiment_id != experiment_id:
            raise ValueError("checkpoint experiment identity differs from smoke result")
        if record.experiment_config_sha256 != config_sha256:
            raise ValueError("checkpoint experiment config identity is incompatible")
        if record.dataset_manifest_sha256 != dataset_manifest_sha256:
            raise ValueError("checkpoint prepared dataset identity is incompatible")
        if not checkpoint.locally_verified:
            raise ValueError("checkpoint artifact has not been locally hash verified")
        artifacts.append(record.artifact)
    paths = tuple(artifact.relative_path for artifact in artifacts)
    if len(paths) != len(set(paths)):
        raise ValueError("training report checkpoint artifact paths must be unique")
    return tuple(artifacts)


def build_training_report(
    *,
    experiment_config: ExperimentConfig,
    isaac_groot_revision: str,
    smoke_result: SmokeResult,
    checkpoints: tuple[CheckpointDescriptor, ...],
    instance_started_at: str,
    generated_at: str,
    provider_hourly_price: float,
) -> TrainingReport:
    """Build a complete report only from mutually compatible immutable inputs."""

    if not isinstance(experiment_config, ExperimentConfig):
        raise TypeError("experiment_config must be an ExperimentConfig")
    if not isinstance(smoke_result, SmokeResult):
        raise TypeError("smoke_result must be a SmokeResult")
    reject_secret_bearing_config(experiment_config.to_dict())
    _require_immutable_revision(isaac_groot_revision, label="Isaac GR00T revision")
    _require_immutable_revision(
        experiment_config.model_revision,
        label="base model revision",
    )
    _require_immutable_revision(
        experiment_config.dataset_revision,
        label="immutable prepared dataset revision",
    )
    if type(provider_hourly_price) not in (int, float) or not math.isfinite(
        float(provider_hourly_price)
    ) or provider_hourly_price < 0:
        raise ValueError("provider hourly price must be a finite nonnegative number")

    started = _parse_timestamp(instance_started_at, label="instance start time")
    finished = _parse_timestamp(generated_at, label="report generation time")
    runtime_seconds = (finished - started).total_seconds()
    if runtime_seconds < 0:
        raise ValueError("report generation time precedes instance start time")

    config_sha256 = canonical_json_sha256(experiment_config)
    if smoke_result.experiment_config_sha256 != config_sha256:
        raise ValueError("smoke experiment config identity is incompatible")
    if smoke_result.dataset_manifest_sha256 != experiment_config.dataset_manifest_sha256:
        raise ValueError("smoke prepared dataset identity is incompatible")
    if not smoke_result.stable or not smoke_result.finite_loss:
        raise ValueError("training report requires a stable finite smoke result")
    artifacts = _validate_checkpoints(
        checkpoints,
        experiment_id=smoke_result.experiment_id,
        config_sha256=config_sha256,
        dataset_manifest_sha256=experiment_config.dataset_manifest_sha256,
    )

    report = TrainingReport(
        experiment_id=smoke_result.experiment_id,
        generated_at=generated_at,
        instance_started_at=instance_started_at,
        container_image_digest=experiment_config.container_digest,
        repository_commit=experiment_config.repository_commit,
        isaac_groot_revision=isaac_groot_revision,
        base_model_repository=experiment_config.model_repository,
        base_model_revision=experiment_config.model_revision,
        dataset_repository=experiment_config.dataset_repository,
        dataset_revision=experiment_config.dataset_revision,
        dataset_manifest_sha256=experiment_config.dataset_manifest_sha256,
        resolved_training_config=experiment_config.to_dict(),
        resolved_training_config_sha256=config_sha256,
        smoke_metrics=smoke_result.to_dict(),
        checkpoint_artifacts=artifacts,
        runtime_seconds=runtime_seconds,
        provider_hourly_price=float(provider_hourly_price),
        cost=runtime_seconds / 3600.0 * float(provider_hourly_price),
    )
    reject_secret_bearing_config(report.to_dict())
    return report


def write_training_report(destination: str | Path, report: TrainingReport) -> None:
    """Atomically write a report after repeating the central secret policy."""

    if not isinstance(report, TrainingReport):
        raise TypeError("report must be a TrainingReport")
    payload = report.to_dict()
    reject_secret_bearing_config(payload)
    atomic_write_json(destination, payload)
