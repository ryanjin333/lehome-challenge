"""Finite, evidence-based worker-capacity decisions for rollout campaigns."""

from __future__ import annotations

from dataclasses import dataclass
import math
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
    first_progress_workers: int | None = None
    stale_ipc_count: int = 0
    peak_host_ram_bytes: int | None = None
    peak_vram_bytes: int | None = None
    cpu_utilization: float | None = None
    run_queue: int | None = None
    inference_latency_seconds: float | None = None
    inference_queue_depth: int | None = None
    policy_evidence_failures: tuple[str, ...] = ()
    failure_classes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("workers", self.workers),
            ("completed trials", self.completed_trials),
            ("failed trials", self.failed_trials),
            ("first-progress workers", self.first_progress_workers),
            ("stale IPC count", self.stale_ipc_count),
            ("peak host RAM bytes", self.peak_host_ram_bytes),
            ("peak VRAM bytes", self.peak_vram_bytes),
            ("run queue", self.run_queue),
            ("inference queue depth", self.inference_queue_depth),
        ):
            if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
                raise ValueError(f"{name} must be an integer when observed")
        for name, value in (
            ("elapsed time", self.elapsed_seconds),
            ("inference VRAM margin", self.inference_vram_margin),
            ("render VRAM margin", self.render_vram_margin),
            ("host RAM margin", self.host_ram_margin),
            ("CPU utilization", self.cpu_utilization),
            ("inference latency", self.inference_latency_seconds),
        ):
            if value is not None and (isinstance(value, bool) or not math.isfinite(value)):
                raise ValueError(f"{name} must be finite when observed")
        if self.workers <= 0 or self.elapsed_seconds <= 0 or self.completed_trials < 0 or self.failed_trials < 0:
            raise ValueError("capacity sample contains invalid counts or elapsed time")
        if self.first_progress_workers is not None and not 0 <= self.first_progress_workers <= self.workers:
            raise ValueError("first-progress worker count is outside the launched group")
        if self.stale_ipc_count < 0:
            raise ValueError("stale IPC count must be non-negative")
        if self.run_queue is not None and self.run_queue < 0:
            raise ValueError("run queue must be non-negative when observed")
        for name, value in (("CPU utilization", self.cpu_utilization), ("inference latency", self.inference_latency_seconds)):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative when observed")
        if self.inference_queue_depth is not None and self.inference_queue_depth < 0:
            raise ValueError("inference queue depth must be non-negative when observed")

    def rejection_reasons(
        self,
        *,
        minimum_ram: float,
        minimum_vram: float,
        max_inference_latency_seconds: float,
        max_inference_queue_depth: int,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.failed_trials:
            reasons.append("trial_failure")
        if self.first_progress_workers is not None and self.first_progress_workers != self.workers:
            reasons.append("first_progress_missing")
        if self.stale_ipc_count:
            reasons.append("stale_ipc")
        # Capacity is not a hardware guess: a measured policy-service response
        # and queue observation are required before a wave can be accepted.
        if self.cpu_utilization is None:
            reasons.append("cpu_utilization_unavailable")
        if self.run_queue is None:
            reasons.append("run_queue_unavailable")
        if self.inference_latency_seconds is None:
            reasons.append("inference_latency_unavailable")
        if self.inference_queue_depth is None:
            reasons.append("inference_queue_depth_unavailable")
        reasons.extend(self.policy_evidence_failures)
        if self.host_ram_margin < minimum_ram:
            reasons.append("host_ram_margin")
        if self.inference_vram_margin < minimum_vram:
            reasons.append("inference_vram_margin")
        if self.render_vram_margin < minimum_vram:
            reasons.append("render_vram_margin")
        if self.inference_latency_seconds is not None and self.inference_latency_seconds > max_inference_latency_seconds:
            reasons.append("inference_latency_limit")
        if self.inference_queue_depth is not None and self.inference_queue_depth > max_inference_queue_depth:
            reasons.append("inference_queue_depth_limit")
        # A saturated CPU with more runnable work than simulator workers is
        # direct evidence of scheduling pressure, rather than a host-size guess.
        if self.cpu_utilization is not None and self.run_queue is not None and self.cpu_utilization >= 0.90 and self.run_queue > self.workers:
            reasons.append("cpu_runqueue")
        return tuple(reasons)

    @property
    def aggregate_rate(self) -> float:
        """Completed trials per second across the whole worker group."""
        return self.completed_trials / self.elapsed_seconds


@dataclass(frozen=True, slots=True)
class CapacityDecision:
    accepted_workers: int
    rejected: dict[int, tuple[str, ...]]


def choose_worker_count(
    samples: Sequence[CapacitySample],
    *,
    minimum_gain: float = 0.15,
    max_inference_latency_seconds: float = 0.5,
    max_inference_queue_depth: int = 16,
) -> CapacityDecision:
    """Accept only the finite 1/2/4/6(/8) sweep prefix with measured margins."""
    if isinstance(minimum_gain, bool) or minimum_gain < 0 or not math.isfinite(minimum_gain):
        raise ValueError("minimum throughput gain must be finite and non-negative")
    if (
        isinstance(max_inference_latency_seconds, bool)
        or max_inference_latency_seconds <= 0
        or not math.isfinite(max_inference_latency_seconds)
    ):
        raise ValueError("maximum inference latency must be finite and positive")
    if (
        not isinstance(max_inference_queue_depth, int)
        or isinstance(max_inference_queue_depth, bool)
        or max_inference_queue_depth <= 0
    ):
        raise ValueError("maximum inference queue depth must be a positive integer")
    accepted = 0
    rejected: dict[int, tuple[str, ...]] = {}
    prior_rate: float | None = None
    allowed = (1, 2, 4, 6, 8)
    for sample in sorted(samples, key=lambda value: value.workers):
        if sample.workers not in allowed:
            raise ValueError("capacity sweep only supports 1, 2, 4, 6, and 8 workers")
        if sample.workers == 8 and accepted != 6:
            rejected[8] = ("six_workers_not_accepted",)
            break
        reasons = list(
            sample.rejection_reasons(
                minimum_ram=0.20,
                minimum_vram=0.15,
                max_inference_latency_seconds=max_inference_latency_seconds,
                max_inference_queue_depth=max_inference_queue_depth,
            )
        )
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
