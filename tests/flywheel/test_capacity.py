from __future__ import annotations

from lehome.flywheel.capacity import CapacitySample, choose_worker_count


def test_capacity_stops_at_six_when_eight_lacks_gain() -> None:
    samples = (
        CapacitySample(4, 120.0, 4, 0, 0.30, 0.25),
        CapacitySample(6, 82.0, 6, 0, 0.24, 0.20),
        CapacitySample(8, 77.0, 8, 0, 0.17, 0.13),
    )
    decision = choose_worker_count(samples, minimum_gain=0.15)
    assert decision.accepted_workers == 6
    assert decision.rejected[8] == ("render_vram_margin", "throughput_gain")
