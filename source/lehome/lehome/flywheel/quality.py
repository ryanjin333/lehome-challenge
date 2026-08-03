"""Statistic-derived, fail-closed grading for physical expert attempts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re


_PINNED_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _finite_positive(value: float, field: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field} must be positive and finite")
    return float(value)


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    """Thresholds materialized from one pinned organizer-statistics revision."""

    dataset_revision: str
    dataset_sha256: str
    clean_velocity_p95: float
    clean_acceleration_p95: float
    clean_jitter_p95: float
    max_velocity_p95: float
    max_acceleration_p95: float
    max_jitter_p95: float
    allowed_stale_samples: int
    allowed_unsafe_commands: int

    def __post_init__(self) -> None:
        if not _PINNED_REVISION.fullmatch(self.dataset_revision):
            raise ValueError("quality thresholds require a pinned organizer dataset revision")
        if not _SHA256.fullmatch(self.dataset_sha256):
            raise ValueError("quality thresholds require an organizer dataset SHA-256")
        for field in (
            "clean_velocity_p95",
            "clean_acceleration_p95",
            "clean_jitter_p95",
            "max_velocity_p95",
            "max_acceleration_p95",
            "max_jitter_p95",
        ):
            object.__setattr__(self, field, _finite_positive(getattr(self, field), field))
        if self.max_velocity_p95 < self.clean_velocity_p95:
            raise ValueError("maximum velocity threshold must not be below the clean threshold")
        if self.max_acceleration_p95 < self.clean_acceleration_p95:
            raise ValueError("maximum acceleration threshold must not be below the clean threshold")
        if self.max_jitter_p95 < self.clean_jitter_p95:
            raise ValueError("maximum jitter threshold must not be below the clean threshold")
        for field in ("allowed_stale_samples", "allowed_unsafe_commands"):
            value = getattr(self, field)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class AttemptStats:
    official_success: bool
    hesitations: int = 0
    corrections: int = 0
    stale_samples: int = 0
    unsafe_commands: int = 0
    disconnected: bool = False
    manual_discarded: bool = False
    velocity_p95: float = 0.0
    acceleration_p95: float = 0.0
    jitter_p95: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.official_success, bool):
            raise ValueError("official_success must be boolean")
        for field in ("hesitations", "corrections", "stale_samples", "unsafe_commands"):
            value = getattr(self, field)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        for field in ("velocity_p95", "acceleration_p95", "jitter_p95"):
            value = getattr(self, field)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{field} must be non-negative and finite")

    def transport_rejections(self, thresholds: QualityThresholds) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.stale_samples > thresholds.allowed_stale_samples:
            reasons.append("stale_samples")
        if self.disconnected:
            reasons.append("disconnected")
        if self.jitter_p95 > thresholds.max_jitter_p95:
            reasons.append("jitter_exceeded")
        return tuple(reasons)

    def safety_rejections(self, thresholds: QualityThresholds) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.unsafe_commands > thresholds.allowed_unsafe_commands:
            reasons.append("unsafe_commands")
        if self.velocity_p95 > thresholds.max_velocity_p95:
            reasons.append("velocity_exceeded")
        if self.acceleration_p95 > thresholds.max_acceleration_p95:
            reasons.append("acceleration_exceeded")
        return tuple(reasons)


@dataclass(frozen=True, slots=True)
class QualityResult:
    grade: str
    reasons: tuple[str, ...]
    sampling_weight: float

    @property
    def trainable(self) -> bool:
        return self.grade in {"A", "B"}


def grade_attempt(stats: AttemptStats, thresholds: QualityThresholds) -> QualityResult:
    """Grade without an operator override path for failure, transport, or safety."""
    rejection = stats.transport_rejections(thresholds) + stats.safety_rejections(thresholds)
    if not stats.official_success:
        rejection += ("official_failure",)
    if stats.manual_discarded:
        rejection += ("operator_discarded",)
    if rejection:
        return QualityResult("C", tuple(dict.fromkeys(rejection)), 0.0)
    if (
        stats.hesitations
        or stats.corrections
        or stats.velocity_p95 > thresholds.clean_velocity_p95
        or stats.acceleration_p95 > thresholds.clean_acceleration_p95
        or stats.jitter_p95 > thresholds.clean_jitter_p95
    ):
        return QualityResult("B", ("successful_recovery",), 0.5)
    return QualityResult("A", ("clean_success",), 1.0)


def load_quality_thresholds(
    path: Path,
    *,
    expected_dataset_revision: str,
    expected_dataset_sha256: str,
) -> QualityThresholds:
    """Load only a complete threshold derivation for the explicitly pinned data."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ValueError("quality thresholds manifest is required")
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("quality thresholds manifest must be valid JSON") from error
    if not isinstance(document, dict) or set(document) != {"schema_version", "organizer_dataset", "derivation", "thresholds"}:
        raise ValueError("quality thresholds manifest has an invalid schema")
    if document["schema_version"] != 1:
        raise ValueError("quality thresholds manifest schema version is unsupported")
    dataset = document["organizer_dataset"]
    derivation = document["derivation"]
    thresholds = document["thresholds"]
    if not isinstance(dataset, dict) or set(dataset) != {"revision", "sha256"}:
        raise ValueError("quality thresholds manifest requires a pinned organizer dataset")
    if dataset["revision"] != expected_dataset_revision or dataset["sha256"] != expected_dataset_sha256:
        raise ValueError("quality thresholds manifest does not match the pinned organizer dataset")
    if not isinstance(derivation, dict) or set(derivation) != {"source_statistics_sha256", "sample_count", "quantiles"}:
        raise ValueError("quality thresholds manifest requires statistical derivation metadata")
    if not _SHA256.fullmatch(derivation["source_statistics_sha256"] if isinstance(derivation["source_statistics_sha256"], str) else ""):
        raise ValueError("quality thresholds manifest requires a statistics SHA-256")
    if not isinstance(derivation["sample_count"], int) or derivation["sample_count"] <= 0:
        raise ValueError("quality thresholds manifest requires a positive statistical sample count")
    quantiles = derivation["quantiles"]
    if not isinstance(quantiles, dict) or set(quantiles) != {"clean", "maximum"}:
        raise ValueError("quality thresholds manifest requires clean and maximum quantiles")
    clean, maximum = quantiles["clean"], quantiles["maximum"]
    if not all(isinstance(value, (int, float)) and 0 < value <= 1 for value in (clean, maximum)) or clean > maximum:
        raise ValueError("quality thresholds manifest quantiles are invalid")
    if not isinstance(thresholds, dict):
        raise ValueError("quality thresholds manifest thresholds are invalid")
    required = {
        "clean_velocity_p95",
        "clean_acceleration_p95",
        "clean_jitter_p95",
        "max_velocity_p95",
        "max_acceleration_p95",
        "max_jitter_p95",
        "allowed_stale_samples",
        "allowed_unsafe_commands",
    }
    if set(thresholds) != required:
        raise ValueError("quality thresholds manifest is incomplete")
    return QualityThresholds(dataset_revision=dataset["revision"], dataset_sha256=dataset["sha256"], **thresholds)
