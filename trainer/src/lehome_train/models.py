"""Strict immutable contracts exchanged by portable trainer stages.

The contracts deliberately use named fields and nested records instead of
free-form dictionaries.  This makes provenance drift and schema additions
explicit failures rather than silently accepted input.
"""

from __future__ import annotations

import math
import re
import types
from dataclasses import dataclass, fields
from typing import Any, Literal, Mapping, Optional, TypeVar, Union, get_args, get_origin, get_type_hints


T = TypeVar("T", bound="StrictModel")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_CONTAINER_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _field_error(field_name: str, reason: str) -> ValueError:
    return ValueError(f"invalid field {field_name}: {reason}")


def _normalize_value(value: object, annotation: object, field_name: str) -> object:
    origin = get_origin(annotation)
    arguments = get_args(annotation)

    if origin is Literal:
        if value not in arguments:
            raise _field_error(field_name, "unsupported literal")
        return value

    if origin in (Union, types.UnionType):
        for option in arguments:
            try:
                return _normalize_value(value, option, field_name)
            except (TypeError, ValueError):
                continue
        raise _field_error(field_name, "does not match any supported type")

    if origin is tuple:
        if not isinstance(value, (tuple, list)):
            raise _field_error(field_name, "expected an array")
        if len(arguments) != 2 or arguments[1] is not Ellipsis:
            raise _field_error(field_name, "unsupported tuple contract")
        return tuple(
            _normalize_value(item, arguments[0], f"{field_name}[]")
            for item in value
        )

    if isinstance(annotation, type) and issubclass(annotation, StrictModel):
        if isinstance(value, annotation):
            return value
        if isinstance(value, Mapping):
            return model_from_mapping(annotation, value)
        raise _field_error(field_name, "expected an object")

    if annotation is str:
        if type(value) is not str:
            raise _field_error(field_name, "expected a string")
        return value
    if annotation is bool:
        if type(value) is not bool:
            raise _field_error(field_name, "expected a boolean")
        return value
    if annotation is int:
        if type(value) is not int:
            raise _field_error(field_name, "expected an integer")
        return value
    if annotation is float:
        if type(value) not in (int, float):
            raise _field_error(field_name, "expected a number")
        normalized = float(value)
        if not math.isfinite(normalized):
            raise _field_error(field_name, "must be finite")
        return normalized
    if annotation is type(None):
        if value is not None:
            raise _field_error(field_name, "expected null")
        return None

    raise _field_error(field_name, "unsupported schema annotation")


def _json_value(value: object) -> object:
    if isinstance(value, StrictModel):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class StrictModel:
    """Base for frozen, recursively validated JSON records."""

    def __post_init__(self) -> None:
        annotations = get_type_hints(type(self))
        for field in fields(self):
            normalized = _normalize_value(
                getattr(self, field.name),
                annotations[field.name],
                field.name,
            )
            object.__setattr__(self, field.name, normalized)

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-compatible representation."""

        return {
            field.name: _json_value(getattr(self, field.name))
            for field in fields(self)
        }


def model_from_mapping(model_type: type[T], value: Mapping[object, object]) -> T:
    """Construct ``model_type`` while rejecting missing and unknown fields."""

    if not isinstance(model_type, type) or not issubclass(model_type, StrictModel):
        raise TypeError("model_type must be a StrictModel subclass")
    if not all(type(key) is str for key in value):
        raise ValueError("model object contains a non-string field name")

    expected = {field.name for field in fields(model_type)}
    supplied = set(value)
    unknown = sorted(supplied - expected)
    missing = sorted(expected - supplied)
    if unknown:
        raise ValueError(f"unknown field: {unknown[0]}")
    if missing:
        raise ValueError(f"missing field: {missing[0]}")

    return model_type(**{name: value[name] for name in expected})


def _require_nonempty(value: str, field_name: str) -> None:
    if not value:
        raise _field_error(field_name, "must not be empty")


def validate_artifact_relative_path(
    value: str,
    field_name: str = "relative_path",
) -> str:
    """Validate one canonical POSIX experiment-relative artifact path."""

    _require_nonempty(value, field_name)
    if value.startswith("/"):
        raise _field_error(field_name, "must be relative")
    if "\\" in value or "\x00" in value:
        raise _field_error(field_name, "must use canonical POSIX components")

    components = value.split("/")
    if any(component in ("", ".") for component in components):
        raise _field_error(field_name, "must not contain path aliases")
    if ".." in components:
        raise _field_error(field_name, "must not traverse")
    if any(component.startswith(".") for component in components):
        raise _field_error(field_name, "must not contain dot components")
    return value


def _require_sha256(value: str, field_name: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise _field_error(field_name, "must be a lowercase SHA-256 hex digest")


def _require_commit(value: str, field_name: str) -> None:
    if not _COMMIT_RE.fullmatch(value):
        raise _field_error(field_name, "must be a full lowercase Git commit")


def _require_nonnegative(value: int, field_name: str) -> None:
    if value < 0:
        raise _field_error(field_name, "must be nonnegative")


def _require_positive(value: int | float, field_name: str) -> None:
    if value <= 0:
        raise _field_error(field_name, "must be positive")


@dataclass(frozen=True, slots=True)
class ArtifactIdentity(StrictModel):
    """Content identity for one experiment-relative artifact."""

    relative_path: str
    sha256: str
    byte_size: int

    def __post_init__(self) -> None:
        StrictModel.__post_init__(self)
        validate_artifact_relative_path(self.relative_path)
        _require_sha256(self.sha256, "sha256")
        _require_nonnegative(self.byte_size, "byte_size")


@dataclass(frozen=True, slots=True)
class CameraMapping(StrictModel):
    """Explicit organizer-camera to GR00T-modality mapping."""

    source_key: str
    target_modality: str

    def __post_init__(self) -> None:
        StrictModel.__post_init__(self)
        _require_nonempty(self.source_key, "source_key")
        _require_nonempty(self.target_modality, "target_modality")


@dataclass(frozen=True, slots=True)
class JointMapping(StrictModel):
    """One primary action dimension: relative arm joint or absolute gripper."""

    source_index: int
    source_name: str
    target_index: int
    target_name: str
    arm: Literal["left", "right"]
    action_mode: Literal["relative", "absolute"]
    is_gripper: bool

    def __post_init__(self) -> None:
        StrictModel.__post_init__(self)
        _require_nonnegative(self.source_index, "source_index")
        _require_nonnegative(self.target_index, "target_index")
        _require_nonempty(self.source_name, "source_name")
        _require_nonempty(self.target_name, "target_name")
        expected_mode = "absolute" if self.is_gripper else "relative"
        if self.action_mode != expected_mode:
            raise _field_error(
                "action_mode",
                "arm joints must be relative and grippers must be absolute",
            )


@dataclass(frozen=True, slots=True)
class SourceInspection(StrictModel):
    """Identity and resolved schema discovered in an organizer dataset."""

    source_repository: str
    source_revision: str
    source_manifest_sha256: str
    episode_ids: tuple[str, ...]
    camera_mappings: tuple[CameraMapping, ...]
    joint_mappings: tuple[JointMapping, ...]
    fps: float
    frame_count: int
    episode_count: int

    def __post_init__(self) -> None:
        StrictModel.__post_init__(self)
        _require_nonempty(self.source_repository, "source_repository")
        _require_nonempty(self.source_revision, "source_revision")
        _require_sha256(self.source_manifest_sha256, "source_manifest_sha256")
        _require_positive(self.fps, "fps")
        _require_nonnegative(self.frame_count, "frame_count")
        _require_nonnegative(self.episode_count, "episode_count")
        if self.episode_count != len(self.episode_ids):
            raise _field_error("episode_count", "must match episode_ids")
        if len(set(self.episode_ids)) != len(self.episode_ids):
            raise _field_error("episode_ids", "must be unique")


@dataclass(frozen=True, slots=True)
class PreparedDatasetProvenance(StrictModel):
    """Complete source, converter, split, schema, and output identity."""

    dataset_repository: str
    dataset_revision: str
    source_repository: str
    source_revision: str
    source_manifest_sha256: str
    converter_commit: str
    container_digest: str
    train_episode_ids: tuple[str, ...]
    validation_episode_ids: tuple[str, ...]
    camera_mappings: tuple[CameraMapping, ...]
    joint_mappings: tuple[JointMapping, ...]
    fps: float
    frame_count: int
    episode_count: int
    source_artifacts: tuple[ArtifactIdentity, ...]
    artifacts: tuple[ArtifactIdentity, ...]

    def __post_init__(self) -> None:
        StrictModel.__post_init__(self)
        for name in (
            "dataset_repository",
            "dataset_revision",
            "source_repository",
            "source_revision",
        ):
            _require_nonempty(getattr(self, name), name)
        _require_sha256(self.source_manifest_sha256, "source_manifest_sha256")
        _require_commit(self.converter_commit, "converter_commit")
        if not _CONTAINER_DIGEST_RE.fullmatch(self.container_digest):
            raise _field_error("container_digest", "must be a SHA-256 image digest")
        _require_positive(self.fps, "fps")
        _require_nonnegative(self.frame_count, "frame_count")
        _require_nonnegative(self.episode_count, "episode_count")
        all_episode_ids = self.train_episode_ids + self.validation_episode_ids
        if self.episode_count != len(all_episode_ids):
            raise _field_error("episode_count", "must match split episode IDs")
        if len(set(all_episode_ids)) != len(all_episode_ids):
            raise _field_error("train_episode_ids", "split episode IDs must be disjoint")


@dataclass(frozen=True, slots=True)
class ExperimentConfig(StrictModel):
    """Resolved immutable inputs controlling one training experiment."""

    repository_commit: str
    container_digest: str
    model_repository: str
    model_revision: str
    dataset_repository: str
    dataset_revision: str
    dataset_manifest_sha256: str
    physical_batch_size: int
    gradient_accumulation_steps: int
    sample_presentations: int
    action_horizon: int
    tune_language_backbone: bool
    tune_visual_backbone: bool

    def __post_init__(self) -> None:
        StrictModel.__post_init__(self)
        _require_commit(self.repository_commit, "repository_commit")
        if not _CONTAINER_DIGEST_RE.fullmatch(self.container_digest):
            raise _field_error("container_digest", "must be a SHA-256 image digest")
        for name in (
            "model_repository",
            "model_revision",
            "dataset_repository",
            "dataset_revision",
        ):
            _require_nonempty(getattr(self, name), name)
        _require_sha256(self.dataset_manifest_sha256, "dataset_manifest_sha256")
        for name in (
            "physical_batch_size",
            "gradient_accumulation_steps",
            "sample_presentations",
            "action_horizon",
        ):
            _require_positive(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class CheckpointRecord(StrictModel):
    """Resumable checkpoint identity tied to exact data and configuration."""

    experiment_id: str
    optimizer_step: int
    sample_presentations: int
    experiment_config_sha256: str
    dataset_manifest_sha256: str
    artifact: ArtifactIdentity
    resumable: bool
    remotely_verified: bool

    def __post_init__(self) -> None:
        StrictModel.__post_init__(self)
        _require_nonempty(self.experiment_id, "experiment_id")
        _require_nonnegative(self.optimizer_step, "optimizer_step")
        _require_nonnegative(self.sample_presentations, "sample_presentations")
        _require_sha256(self.experiment_config_sha256, "experiment_config_sha256")
        _require_sha256(self.dataset_manifest_sha256, "dataset_manifest_sha256")


@dataclass(frozen=True, slots=True)
class SmokeResult(StrictModel):
    """Result of one fixed-accumulation physical-batch smoke attempt."""

    experiment_id: str
    experiment_config_sha256: str
    dataset_manifest_sha256: str
    physical_batch_size: int
    gradient_accumulation_steps: int
    optimizer_steps: int
    stable: bool
    finite_loss: bool
    physical_vram_bytes: int
    peak_reserved_vram_bytes: int
    steady_steps_per_second: float
    samples_per_second: float
    error_code: Optional[str]

    def __post_init__(self) -> None:
        StrictModel.__post_init__(self)
        _require_nonempty(self.experiment_id, "experiment_id")
        _require_sha256(self.experiment_config_sha256, "experiment_config_sha256")
        _require_sha256(self.dataset_manifest_sha256, "dataset_manifest_sha256")
        _require_positive(self.physical_batch_size, "physical_batch_size")
        _require_positive(
            self.gradient_accumulation_steps,
            "gradient_accumulation_steps",
        )
        _require_positive(self.optimizer_steps, "optimizer_steps")
        _require_positive(self.physical_vram_bytes, "physical_vram_bytes")
        _require_nonnegative(
            self.peak_reserved_vram_bytes,
            "peak_reserved_vram_bytes",
        )
        _require_nonnegative_float(
            self.steady_steps_per_second,
            "steady_steps_per_second",
        )
        _require_nonnegative_float(self.samples_per_second, "samples_per_second")


def _require_nonnegative_float(value: float, field_name: str) -> None:
    if value < 0:
        raise _field_error(field_name, "must be nonnegative")


@dataclass(frozen=True, slots=True)
class MemorizationResult(StrictModel):
    """Offline one-episode diagnostic result; never a promotable checkpoint."""

    experiment_id: str
    experiment_config_sha256: str
    dataset_manifest_sha256: str
    episode_id: str
    initialized_normalized_mse: float
    final_normalized_mse: float
    initialized_dimension_mse: tuple[float, ...]
    final_dimension_mse: tuple[float, ...]
    sample_presentations: int
    offline_gate_passed: bool
    promotable: bool
    pending_gate: str

    def __post_init__(self) -> None:
        StrictModel.__post_init__(self)
        _require_nonempty(self.experiment_id, "experiment_id")
        _require_sha256(self.experiment_config_sha256, "experiment_config_sha256")
        _require_sha256(self.dataset_manifest_sha256, "dataset_manifest_sha256")
        _require_nonempty(self.episode_id, "episode_id")
        _require_nonnegative_float(
            self.initialized_normalized_mse,
            "initialized_normalized_mse",
        )
        _require_nonnegative_float(self.final_normalized_mse, "final_normalized_mse")
        if len(self.initialized_dimension_mse) != len(self.final_dimension_mse):
            raise _field_error(
                "final_dimension_mse",
                "must match initialized dimension count",
            )
        for field_name in (
            "initialized_dimension_mse",
            "final_dimension_mse",
        ):
            for value in getattr(self, field_name):
                _require_nonnegative_float(value, field_name)
        _require_nonnegative(self.sample_presentations, "sample_presentations")
        if self.promotable:
            raise _field_error("promotable", "memorization results are diagnostic")
        if self.pending_gate != "simulator_expert_replay":
            raise _field_error(
                "pending_gate",
                "must identify the remaining simulator gate",
            )


@dataclass(frozen=True, slots=True)
class SyncEntry(StrictModel):
    """One allowlisted artifact and its local/remote verification state."""

    relative_path: str
    sha256: str
    byte_size: int
    remotely_verified: bool = False

    def __post_init__(self) -> None:
        StrictModel.__post_init__(self)
        validate_artifact_relative_path(self.relative_path)
        _require_sha256(self.sha256, "sha256")
        _require_nonnegative(self.byte_size, "byte_size")


@dataclass(frozen=True, slots=True)
class SyncManifest(StrictModel):
    """Generated upload allowlist tied to one resolved experiment config."""

    experiment_id: str
    experiment_config_sha256: str
    entries: tuple[SyncEntry, ...]

    def __post_init__(self) -> None:
        StrictModel.__post_init__(self)
        _require_nonempty(self.experiment_id, "experiment_id")
        _require_sha256(self.experiment_config_sha256, "experiment_config_sha256")
        paths = tuple(entry.relative_path for entry in self.entries)
        if len(set(paths)) != len(paths):
            raise _field_error("entries", "relative paths must be unique")
