"""Deterministic, bounded host-throughput measurement for corrective training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

LOADER_CANDIDATES = (0, 4, 8, 12, 16)
STEADY_STEPS = 100


class NoStableTrainingCandidate(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TrainingProbe:
    loader_workers: int
    physical_batch_size: int
    samples_per_second: float
    finite_loss: bool
    stable: bool
    free_vram_percent: float
    hourly_cost: float = 0.0


@dataclass(frozen=True, slots=True)
class ThroughputTuningReport:
    loader_results: tuple[TrainingProbe, ...]
    batch_results: tuple[TrainingProbe, ...]
    selected_loader_workers: int
    fastest_stable_physical_batch: int
    production_physical_batch: int = 64


def select_candidate(results: Sequence[TrainingProbe]) -> TrainingProbe:
    admitted = [item for item in results if item.finite_loss and item.stable and item.free_vram_percent >= 10.0]
    if not admitted:
        raise NoStableTrainingCandidate("no stable candidate with 10% VRAM headroom")
    return max(admitted, key=lambda item: (item.samples_per_second, -item.hourly_cost, item.free_vram_percent))


def tune_training(*, loader_results: Sequence[TrainingProbe], batch_results: Sequence[TrainingProbe]) -> ThroughputTuningReport:
    loader = select_candidate(loader_results)
    if tuple(item.loader_workers for item in loader_results) != LOADER_CANDIDATES or any(item.physical_batch_size != 64 for item in loader_results):
        raise ValueError("tuning probes require the exact fixed batch-64 loader sweep")
    if batch_results:
        raise ValueError("this campaign forbids a physical-batch sweep")
    return ThroughputTuningReport(tuple(loader_results), (), loader.loader_workers, 64)


def tune_on_host(*, run: Callable[[int, int], TrainingProbe]) -> ThroughputTuningReport:
    loaders = [run(workers, 64) for workers in LOADER_CANDIDATES]
    return tune_training(loader_results=loaders, batch_results=())
