"""Deterministic diagnostic ranking for failed rollout replay candidates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class FailureEvidence:
    episode_id: str
    category: str
    official_success: bool
    max_progress: float
    stalled_steps: int
    length: int
    restorable: bool
    official_return: float | None = None

    def __post_init__(self) -> None:
        if not self.episode_id or not self.category:
            raise ValueError("failure evidence requires episode_id and category")
        if self.official_success:
            raise ValueError("hard mining accepts only official failures")
        if not math.isfinite(self.max_progress):
            raise ValueError("max_progress must be finite")
        if not isinstance(self.stalled_steps, int) or self.stalled_steps < 0:
            raise ValueError("stalled_steps must be a non-negative integer")
        if not isinstance(self.length, int) or self.length < 0:
            raise ValueError("length must be a non-negative integer")
        if self.official_return is not None and not math.isfinite(self.official_return):
            raise ValueError("official_return must be finite")

    @property
    def preserved_official_return(self) -> float | None:
        """Return the recorder's official reward exactly as supplied."""
        return self.official_return


@dataclass(frozen=True, slots=True)
class RankedFailure:
    episode_id: str
    category: str
    official_success: bool
    official_return: float | None
    score: float
    eligible_for_recovery: bool
    priority_reasons: tuple[str, ...]
    diagnostics: Mapping[str, float | int | bool]

    @classmethod
    def from_evidence(
        cls,
        evidence: FailureEvidence,
        score: float,
        *,
        category_gap: float,
        progress: float,
        stall_fraction: float,
        short_stall: float,
        dead_terminal: bool,
        eligible_for_recovery: bool,
    ) -> RankedFailure:
        reasons = tuple(
            reason
            for reason, present in (
                ("category_gap", category_gap > 0.0),
                ("high_progress", progress >= 0.5),
                ("short_stall", stall_fraction <= 0.25),
                ("restorable", evidence.restorable),
            )
            if present
        )
        return cls(
            episode_id=evidence.episode_id,
            category=evidence.category,
            official_success=evidence.official_success,
            official_return=evidence.preserved_official_return,
            score=score,
            eligible_for_recovery=eligible_for_recovery,
            priority_reasons=reasons,
            diagnostics=MappingProxyType(
                {
                    "max_progress": evidence.max_progress,
                    "stalled_steps": evidence.stalled_steps,
                    "length": evidence.length,
                    "restorable": evidence.restorable,
                    "category_gap": category_gap,
                    "progress": progress,
                    "stall_fraction": stall_fraction,
                    "short_stall": short_stall,
                    "dead_terminal": dead_terminal,
                    "eligible_for_recovery": eligible_for_recovery,
                }
            ),
        )


def _category_metric(category_success: Mapping[str, float], category: str) -> float:
    if category not in category_success:
        raise ValueError(f"category_success is missing {category}")
    value = category_success[category]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("category_success values must be finite numbers")
    if value < 0.0 or value > 1.0:
        raise ValueError("category_success values must be between zero and one")
    return float(value)


def rank_failures(
    failures: Sequence[FailureEvidence], *, category_success: Mapping[str, float]
) -> tuple[RankedFailure, ...]:
    """Score diagnostics while leaving the official reward and success untouched."""
    ranked: list[RankedFailure] = []
    for failure in failures:
        category_gap = 1.0 - _category_metric(category_success, failure.category)
        progress = min(max(failure.max_progress, 0.0), 1.0)
        stall_fraction = min(failure.stalled_steps / max(failure.length, 1), 1.0)
        short_stall = 1.0 - stall_fraction
        dead_terminal = progress < 0.25 or stall_fraction > 0.75
        eligible_for_recovery = failure.restorable and progress >= 0.35 and stall_fraction <= 0.5
        score = 4.0 * category_gap + 4.0 * progress + 2.0 * short_stall + float(failure.restorable)
        ranked.append(
            RankedFailure.from_evidence(
                failure,
                score,
                category_gap=category_gap,
                progress=progress,
                stall_fraction=stall_fraction,
                short_stall=short_stall,
                dead_terminal=dead_terminal,
                eligible_for_recovery=eligible_for_recovery,
            )
        )
    return tuple(
        sorted(
            ranked,
            key=lambda item: (
                not item.eligible_for_recovery,
                bool(item.diagnostics["dead_terminal"]),
                -item.score,
                item.episode_id,
            ),
        )
    )
