from __future__ import annotations

import pytest

from lehome_train.batch_select import (
    GIBIBYTE,
    batch_candidates,
    fallback_candidates,
    has_required_headroom,
    select_largest_stable_batch,
)
from lehome_train.models import SmokeResult


SHA_A = "a" * 64
SHA_B = "b" * 64


def _result(
    batch: int,
    *,
    total_gib: int = 96,
    reserved_gib: float = 80,
    minimum_free_gib: float = 16,
    stable: bool = True,
    finite_loss: bool = True,
    accumulation: int = 1,
) -> SmokeResult:
    return SmokeResult(
        experiment_id=f"smoke-batch-{batch}",
        experiment_config_sha256=SHA_A,
        dataset_manifest_sha256=SHA_B,
        physical_batch_size=batch,
        gradient_accumulation_steps=accumulation,
        optimizer_steps=100,
        stable=stable,
        finite_loss=finite_loss,
        physical_vram_bytes=total_gib * GIBIBYTE,
        peak_reserved_vram_bytes=int(reserved_gib * GIBIBYTE),
        minimum_steady_state_free_vram_bytes=int(minimum_free_gib * GIBIBYTE),
        steady_steps_per_second=1.0,
        samples_per_second=float(batch),
        error_code=None if stable else "cuda_oom",
    )


@pytest.mark.parametrize(
    ("vram_gib", "expected"),
    [
        (40, (8, 16, 32)),
        (63, (8, 16, 32)),
        (64, (16, 32, 64)),
        (96, (16, 32, 64)),
    ],
)
def test_batch_candidates_follow_physical_vram_tiers(
    vram_gib: int,
    expected: tuple[int, ...],
) -> None:
    assert batch_candidates(vram_gib * GIBIBYTE) == expected


def test_batch_candidates_reject_subminimum_or_non_integer_vram() -> None:
    with pytest.raises(ValueError, match="40 GiB"):
        batch_candidates(40 * GIBIBYTE - 1)
    with pytest.raises(ValueError, match="integer"):
        batch_candidates(float(64 * GIBIBYTE))  # type: ignore[arg-type]


def test_first_candidate_fallback_uses_smaller_powers_of_two_without_duplicates() -> None:
    assert fallback_candidates(8) == (4, 2, 1)
    assert fallback_candidates(16) == (8, 4, 2, 1)


def test_headroom_gate_requires_at_least_ten_percent_physical_vram_free() -> None:
    assert has_required_headroom(_result(16, total_gib=40, minimum_free_gib=4)) is True
    assert has_required_headroom(_result(16, total_gib=40, minimum_free_gib=3.9999)) is False


def test_selection_uses_nvml_free_memory_not_allocator_reserved_memory() -> None:
    allocator_looks_safe_but_physical_free_is_low = _result(
        32,
        reserved_gib=70,
        minimum_free_gib=5,
    )
    allocator_peak_is_high_but_steady_physical_free_passes = _result(
        16,
        reserved_gib=92,
        minimum_free_gib=10,
    )

    assert has_required_headroom(allocator_looks_safe_but_physical_free_is_low) is False
    assert has_required_headroom(allocator_peak_is_high_but_steady_physical_free_passes) is True
    assert select_largest_stable_batch(
        (
            allocator_peak_is_high_but_steady_physical_free_passes,
            allocator_looks_safe_but_physical_free_is_low,
        )
    ) == 16


def test_selection_returns_largest_stable_finite_fixed_accumulation_batch() -> None:
    results = (
        _result(16, reserved_gib=70, minimum_free_gib=26),
        _result(32, reserved_gib=82, minimum_free_gib=14),
        _result(64, reserved_gib=90, minimum_free_gib=6),
    )

    assert select_largest_stable_batch(results) == 32

    with pytest.raises(ValueError, match="gradient accumulation"):
        select_largest_stable_batch((_result(16, accumulation=2),))
    with pytest.raises(ValueError, match="no stable batch"):
        select_largest_stable_batch((_result(16, finite_loss=False),))
