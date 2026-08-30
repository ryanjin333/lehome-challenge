"""Deterministic physical-batch candidates and VRAM-headroom selection."""

from __future__ import annotations

from collections.abc import Iterable

from lehome_train.models import SmokeResult


GIBIBYTE = 1024**3
MINIMUM_VRAM_BYTES = 40 * GIBIBYTE
HIGH_VRAM_TIER_BYTES = 64 * GIBIBYTE
REQUIRED_FREE_VRAM_PERCENT = 10


class NoStableBatchError(ValueError):
    """No comparable smoke attempt satisfies stability and headroom gates."""


def batch_candidates(physical_vram_bytes: int) -> tuple[int, ...]:
    """Return the approved ascending candidate tier for one physical GPU."""

    if type(physical_vram_bytes) is not int:
        raise ValueError("physical VRAM bytes must be an integer")
    if physical_vram_bytes < MINIMUM_VRAM_BYTES:
        raise ValueError("at least 40 GiB physical VRAM is required")
    if physical_vram_bytes < HIGH_VRAM_TIER_BYTES:
        return (8, 16, 32)
    return (16, 32, 64)


def fallback_candidates(first_candidate: int) -> tuple[int, ...]:
    """Return approved descending powers of two below the first candidate."""

    if type(first_candidate) is not int or first_candidate <= 0:
        raise ValueError("first batch candidate must be a positive integer")
    return tuple(batch for batch in (8, 4, 2, 1) if batch < first_candidate)


def has_required_headroom(result: SmokeResult) -> bool:
    """Use integer arithmetic to require at least 10% physical VRAM free."""

    if not isinstance(result, SmokeResult):
        raise TypeError("headroom requires a SmokeResult")
    if result.minimum_steady_state_free_vram_bytes is None:
        return False
    return has_required_free_vram(
        result.physical_vram_bytes,
        result.minimum_steady_state_free_vram_bytes,
    )


def has_required_free_vram(
    physical_vram_bytes: int,
    observed_free_vram_bytes: int,
) -> bool:
    """Gate on NVML physical free memory, including non-PyTorch consumers."""

    if type(physical_vram_bytes) is not int or physical_vram_bytes <= 0:
        raise ValueError("physical VRAM bytes must be a positive integer")
    if type(observed_free_vram_bytes) is not int or observed_free_vram_bytes < 0:
        raise ValueError("observed free VRAM bytes must be a nonnegative integer")
    return observed_free_vram_bytes * 100 >= (
        physical_vram_bytes * REQUIRED_FREE_VRAM_PERCENT
    )


def select_largest_stable_batch(results: Iterable[SmokeResult]) -> int:
    """Select the largest comparable stable result satisfying headroom."""

    observed = tuple(results)
    if not all(isinstance(result, SmokeResult) for result in observed):
        raise TypeError("batch selection requires SmokeResult records")
    if any(result.gradient_accumulation_steps != 1 for result in observed):
        raise ValueError("gradient accumulation must remain exactly 1")
    comparable = tuple(
        result
        for result in observed
        if result.optimizer_steps == 100
        and result.stable
        and result.finite_loss
        and has_required_headroom(result)
    )
    if not comparable:
        raise NoStableBatchError(
            "no stable batch leaves at least 10% physical VRAM free"
        )
    return max(result.physical_batch_size for result in comparable)
