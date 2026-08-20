"""Pure ranking rules for the asynchronous successive-halving campaign."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True, slots=True)
class EvaluationScore:
    experiment_id: str
    policy_digest: str
    category_successes: tuple[int, int, int, int]
    overall_successes: int
    paired_improvement: float
    gpu_seconds: float
    safety_failure: bool

    def __post_init__(self) -> None:
        if len(self.category_successes) != 4 or any(type(value) is not int or value < 0 for value in self.category_successes):
            raise ValueError("score requires four category counts")
        if type(self.overall_successes) is not int or self.overall_successes < 0 or not math.isfinite(self.paired_improvement) or not math.isfinite(self.gpu_seconds) or self.gpu_seconds < 0:
            raise ValueError("score contains invalid metrics")


def rank_key(score: EvaluationScore) -> tuple[int, int, int, float, float]:
    """Higher is better; safety is always rejected before policy metrics."""
    return (0 if score.safety_failure else 1, min(score.category_successes), score.overall_successes, score.paired_improvement, -score.gpu_seconds)


def _unique(scores: Iterable[EvaluationScore]) -> list[EvaluationScore]:
    ordered = list(scores)
    if len({score.policy_digest for score in ordered}) != len(ordered):
        raise ValueError("duplicate policy digest")
    return ordered


def select_1k_promotions(scores: Iterable[EvaluationScore]) -> tuple[EvaluationScore, ...]:
    return tuple(sorted((score for score in _unique(scores) if not score.safety_failure), key=rank_key, reverse=True)[:3])


def select_seed_repeats(scores: Iterable[EvaluationScore]) -> tuple[EvaluationScore, ...]:
    return tuple(sorted((score for score in _unique(scores) if not score.safety_failure), key=rank_key, reverse=True)[:2])


def select_2k_finalists(scores: Iterable[EvaluationScore]) -> tuple[EvaluationScore, ...]:
    ordered = list(select_1k_promotions(scores))
    if len(ordered) < 2:
        return tuple(ordered[:1])
    first, second = ordered[:2]
    within_one = first.overall_successes - second.overall_successes <= 1
    tied_categories = min(first.category_successes) == min(second.category_successes)
    return (first, second) if within_one and tied_categories else (first,)
