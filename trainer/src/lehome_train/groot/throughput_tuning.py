"""Deterministic, bounded host-throughput measurement for corrective training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

LOADER_CANDIDATES = (4, 8, 12)
BATCH_CANDIDATES = (64, 96, 128)
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
    batch = select_candidate(batch_results)
    if loader.physical_batch_size != 64 or batch.loader_workers != loader.loader_workers:
        raise ValueError("tuning probes do not follow loader-first batch-64 contract")
    return ThroughputTuningReport(tuple(loader_results), tuple(batch_results), loader.loader_workers, batch.physical_batch_size)


def tune_on_host(*, run: Callable[[int, int], TrainingProbe]) -> ThroughputTuningReport:
    loaders = [run(workers, 64) for workers in LOADER_CANDIDATES]
    selected = select_candidate(loaders)
    batches: list[TrainingProbe] = []
    for batch in BATCH_CANDIDATES:
        outcome = run(selected.loader_workers, batch)
        batches.append(outcome)
        if not outcome.finite_loss or not outcome.stable:
            break
    return tune_training(loader_results=loaders, batch_results=batches)
