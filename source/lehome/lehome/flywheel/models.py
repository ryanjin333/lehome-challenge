"""Immutable data contracts shared by flywheel collection and export."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
import re
from types import MappingProxyType
from typing import Mapping


PINNED = re.compile(r"^[0-9a-f]{40}$")
CATEGORIES = frozenset({"top_long", "top_short", "pant_long", "pant_short"})
RELEASE_STAGES = frozenset({"seen", "public_unseen"})
STRATEGIES = frozenset({"canonical", "mild", "strong"})


class ActionSource(StrEnum):
    POLICY = "policy"
    EXPERT = "expert"
    HOLD = "hold"


class QualityGrade(StrEnum):
    A = "A"
    B = "B"
    C = "C"


class RejectionReason(StrEnum):
    POLICY = "policy"
    HOLD = "hold"
    FAILED_EPISODE = "failed_episode"
    SHORT_TAIL = "short_tail"
    STALE_EXPERT = "stale_expert"
    HOLDOUT = "holdout"
    MISSING = "missing"
    CLIPPED = "clipped"
    UNSAFE = "unsafe"
    DISCONNECTED = "disconnected"
    OPERATOR_DISCARDED = "operator_discarded"


@dataclass(frozen=True, slots=True)
class EpisodeIdentity:
    episode_id: str
    policy_repo: str
    policy_revision: str
    policy_step: int
    code_revision: str
    asset_revision: str
    simulator_version: str
    garment_name: str
    category: str
    release_stage: str
    seed: int
    instruction: str
    strategy: str

    def __post_init__(self) -> None:
        for name in ("policy_revision", "code_revision", "asset_revision"):
            if not PINNED.fullmatch(getattr(self, name)):
                raise ValueError(f"{name} must be a pinned 40-character revision")
        if not self.episode_id or "/" in self.episode_id or "\\" in self.episode_id:
            raise ValueError("episode_id must be a non-empty path-safe identifier")
        if not self.policy_repo or not self.simulator_version or not self.garment_name:
            raise ValueError("episode provenance fields must be non-empty")
        if not isinstance(self.policy_step, int) or self.policy_step < 0:
            raise ValueError("policy_step must be a non-negative integer")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not self.instruction.strip():
            raise ValueError("instruction must be non-empty")
        if self.category not in CATEGORIES:
            raise ValueError("unsupported garment category")
        if self.release_stage not in RELEASE_STAGES:
            raise ValueError("unsupported release stage")
        if self.strategy not in STRATEGIES:
            raise ValueError("unsupported collection strategy")


@dataclass(frozen=True, slots=True)
class EpisodeFrame:
    step: int
    monotonic_ns: int
    wall_time_ns: int
    state: tuple[float, ...]
    action: tuple[float, ...]
    action_source: ActionSource
    reward: float
    success: bool
    segment: int
    policy_request_id: str | None = None
    policy_chunk_offset: int | None = None
    expert_sequence: int | None = None
    expert_sample_age_ms: float | None = None

    def __post_init__(self) -> None:
        if len(self.state) != 12 or len(self.action) != 12:
            raise ValueError("state and action must contain 12 finite values")
        try:
            values = (*self.state, *self.action, self.reward)
        except TypeError as error:
            raise ValueError("state and action must contain 12 finite values") from error
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            raise ValueError("state and action must contain 12 finite values")
        if not isinstance(self.action_source, ActionSource):
            raise ValueError("action_source must be an ActionSource")
        if not isinstance(self.step, int) or self.step < 0:
            raise ValueError("step must be a non-negative integer")
        if not isinstance(self.segment, int) or self.segment < 0:
            raise ValueError("segment must be a non-negative integer")
        if not isinstance(self.monotonic_ns, int) or self.monotonic_ns < 0:
            raise ValueError("monotonic_ns must be a non-negative integer")
        if not isinstance(self.wall_time_ns, int) or self.wall_time_ns < 0:
            raise ValueError("wall_time_ns must be a non-negative integer")
        if self.policy_chunk_offset is not None and self.policy_chunk_offset < 0:
            raise ValueError("policy_chunk_offset must be non-negative")
        if self.expert_sequence is not None and self.expert_sequence < 0:
            raise ValueError("expert_sequence must be non-negative")
        if self.expert_sample_age_ms is not None and (
            not math.isfinite(self.expert_sample_age_ms) or self.expert_sample_age_ms < 0
        ):
            raise ValueError("expert_sample_age_ms must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class EpisodeOutcome:
    outcome: str
    accepted: bool
    quality_grade: QualityGrade
    rejection_reasons: tuple[RejectionReason, ...]
    operator_disposition: str

    def __post_init__(self) -> None:
        if self.outcome not in {"success", "timeout", "error", "unsafe", "discarded"}:
            raise ValueError("unsupported episode outcome")
        if not isinstance(self.quality_grade, QualityGrade):
            raise ValueError("quality_grade must be a QualityGrade")
        if not self.operator_disposition:
            raise ValueError("operator_disposition must be non-empty")
        if any(not isinstance(reason, RejectionReason) for reason in self.rejection_reasons):
            raise ValueError("rejection_reasons must contain RejectionReason values")
        if len(set(self.rejection_reasons)) != len(self.rejection_reasons):
            raise ValueError("rejection_reasons must not contain duplicates")
        if self.accepted and (self.outcome != "success" or self.quality_grade is QualityGrade.C):
            raise ValueError("accepted episodes must be successful and quality grade A or B")
        if not self.accepted and not self.rejection_reasons:
            raise ValueError("rejected episodes require a rejection reason")


@dataclass(frozen=True, slots=True)
class RandomizationRecord:
    strategy: str
    values: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError("unsupported collection strategy")
        if not all(isinstance(key, str) and key for key in self.values):
            raise ValueError("randomization keys must be non-empty strings")
        if self.strategy == "canonical" and self.values:
            raise ValueError("canonical strategy must not include randomization values")
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))
