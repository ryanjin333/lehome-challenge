from __future__ import annotations

from lehome.flywheel.capacity import CapacitySample, choose_worker_count


def test_capacity_stops_at_six_when_eight_lacks_gain() -> None:
    samples = (
        CapacitySample(4, 120.0, 4, 0, 0.30, 0.25),
        CapacitySample(6, 82.0, 6, 0, 0.24, 0.20),
        CapacitySample(8, 110.0, 8, 0, 0.17, 0.13),
    )
    decision = choose_worker_count(samples, minimum_gain=0.15)
    assert decision.accepted_workers == 6
    assert decision.rejected[8] == ("render_vram_margin", "throughput_gain")


def test_capacity_compares_aggregate_throughput_not_per_worker_rate() -> None:
    decision = choose_worker_count(
        (
            CapacitySample(1, 100.0, 1, 0, 0.30, 0.25),
            CapacitySample(2, 100.0, 2, 0, 0.30, 0.25),
        ),
        minimum_gain=0.15,
    )

    assert decision.accepted_workers == 2
    assert decision.rejected == {}


def test_capacity_rejects_the_host_when_the_one_worker_sample_fails() -> None:
    decision = choose_worker_count((CapacitySample(1, 100.0, 0, 1, 0.30, 0.25),))

    assert decision.accepted_workers == 0
    assert decision.rejected == {1: ("trial_failure",)}
