"""Finite, evidence-based worker-capacity decisions for rollout campaigns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class CapacitySample:
    workers: int
    elapsed_seconds: float
    completed_trials: int
    failed_trials: int
    inference_vram_margin: float
    render_vram_margin: float
    host_ram_margin: float = 1.0

    def __post_init__(self) -> None:
        if self.workers <= 0 or self.elapsed_seconds <= 0 or self.completed_trials < 0 or self.failed_trials < 0:
            raise ValueError("capacity sample contains invalid counts or elapsed time")

    def rejection_reasons(self, *, minimum_ram: float, minimum_vram: float) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.failed_trials:
            reasons.append("trial_failure")
        if self.host_ram_margin < minimum_ram:
            reasons.append("host_ram_margin")
        if self.inference_vram_margin < minimum_vram:
            reasons.append("inference_vram_margin")
        if self.render_vram_margin < minimum_vram:
            reasons.append("render_vram_margin")
        return tuple(reasons)

    @property
    def aggregate_rate(self) -> float:
        """Completed trials per second across the whole worker group."""
        return self.completed_trials / self.elapsed_seconds


@dataclass(frozen=True, slots=True)
class CapacityDecision:
    accepted_workers: int
    rejected: dict[int, tuple[str, ...]]


def choose_worker_count(samples: Sequence[CapacitySample], *, minimum_gain: float = 0.15) -> CapacityDecision:
    """Accept only the finite 1/2/4/6(/8) sweep prefix with measured margins."""
    if minimum_gain < 0:
        raise ValueError("minimum throughput gain must be non-negative")
    accepted = 1
    rejected: dict[int, tuple[str, ...]] = {}
    prior_rate: float | None = None
    allowed = (1, 2, 4, 6, 8)
    for sample in sorted(samples, key=lambda value: value.workers):
        if sample.workers not in allowed:
            raise ValueError("capacity sweep only supports 1, 2, 4, 6, and 8 workers")
        if sample.workers == 8 and accepted != 6:
            rejected[8] = ("six_workers_not_accepted",)
            break
        reasons = list(sample.rejection_reasons(minimum_ram=0.20, minimum_vram=0.15))
        if prior_rate is not None and sample.aggregate_rate / prior_rate - 1.0 < minimum_gain:
            reasons.append("throughput_gain")
        if reasons:
            rejected[sample.workers] = tuple(dict.fromkeys(reasons))
            if sample.workers == 1:
                accepted = 0
            break
        accepted = sample.workers
        prior_rate = sample.aggregate_rate
    return CapacityDecision(accepted, rejected)


__all__ = ["CapacityDecision", "CapacitySample", "choose_worker_count"]
