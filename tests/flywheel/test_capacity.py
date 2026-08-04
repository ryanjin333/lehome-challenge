from __future__ import annotations

import pytest

from lehome.flywheel.capacity import CapacitySample, choose_worker_count


def test_capacity_stops_at_six_when_eight_lacks_gain() -> None:
    samples = (
        CapacitySample(4, 120.0, 4, 0, 0.30, 0.25, cpu_utilization=0.2, run_queue=1, inference_latency_seconds=0.1, inference_queue_depth=0),
        CapacitySample(6, 82.0, 6, 0, 0.24, 0.20, cpu_utilization=0.2, run_queue=1, inference_latency_seconds=0.1, inference_queue_depth=0),
        CapacitySample(8, 110.0, 8, 0, 0.17, 0.13, cpu_utilization=0.2, run_queue=1, inference_latency_seconds=0.1, inference_queue_depth=0),
    )
    decision = choose_worker_count(samples, minimum_gain=0.15)
    assert decision.accepted_workers == 6
    assert decision.rejected[8] == ("render_vram_margin", "throughput_gain")


def test_capacity_compares_aggregate_throughput_not_per_worker_rate() -> None:
    decision = choose_worker_count(
        (
            CapacitySample(1, 100.0, 1, 0, 0.30, 0.25, cpu_utilization=0.2, run_queue=1, inference_latency_seconds=0.1, inference_queue_depth=0),
            CapacitySample(2, 100.0, 2, 0, 0.30, 0.25, cpu_utilization=0.2, run_queue=1, inference_latency_seconds=0.1, inference_queue_depth=0),
        ),
        minimum_gain=0.15,
    )

    assert decision.accepted_workers == 2
    assert decision.rejected == {}


def test_capacity_rejects_the_host_when_the_one_worker_sample_fails() -> None:
    decision = choose_worker_count((CapacitySample(1, 100.0, 0, 1, 0.30, 0.25, cpu_utilization=0.2, run_queue=1, inference_latency_seconds=0.1, inference_queue_depth=0),))

    assert decision.accepted_workers == 0
    assert decision.rejected == {1: ("trial_failure",)}


def test_capacity_rejects_missing_progress_and_stale_ipc_even_with_memory_headroom() -> None:
    decision = choose_worker_count((
        CapacitySample(1, 10.0, 1, 0, 0.30, 0.30, first_progress_workers=0, stale_ipc_count=1, cpu_utilization=0.2, run_queue=1, inference_latency_seconds=0.1, inference_queue_depth=0),
    ))

    assert decision.accepted_workers == 0
    assert decision.rejected[1] == ("first_progress_missing", "stale_ipc")


def test_capacity_rejects_unobserved_inference_and_cpu_telemetry() -> None:
    decision = choose_worker_count((CapacitySample(1, 10.0, 1, 0, 0.30, 0.30),))

    assert decision.accepted_workers == 0
    assert decision.rejected[1] == (
        "cpu_utilization_unavailable",
        "run_queue_unavailable",
        "inference_latency_unavailable",
        "inference_queue_depth_unavailable",
    )


def test_capacity_rejects_policy_inference_above_explicit_limits() -> None:
    decision = choose_worker_count((
        CapacitySample(1, 10.0, 1, 0, 0.30, 0.30, cpu_utilization=0.2, run_queue=1, inference_latency_seconds=0.51, inference_queue_depth=17),
    ))

    assert decision.accepted_workers == 0
    assert decision.rejected[1] == ("inference_latency_limit", "inference_queue_depth_limit")


def test_capacity_rejects_worker_policy_evidence_failures() -> None:
    decision = choose_worker_count((
        CapacitySample(
            1, 10.0, 1, 0, 0.30, 0.30,
            cpu_utilization=0.2, run_queue=1, inference_latency_seconds=0.1, inference_queue_depth=0,
            policy_evidence_failures=("policy_telemetry_malformed", "policy_telemetry_wrong_worker"),
        ),
    ))

    assert decision.accepted_workers == 0
    assert decision.rejected[1] == ("policy_telemetry_malformed", "policy_telemetry_wrong_worker")


@pytest.mark.parametrize("value", (True, float("nan"), float("inf"), float("-inf")), ids=("bool", "nan", "positive-infinity", "negative-infinity"))
@pytest.mark.parametrize(
    "field",
    (
        "elapsed_seconds",
        "inference_vram_margin",
        "render_vram_margin",
        "host_ram_margin",
        "cpu_utilization",
        "inference_latency_seconds",
    ),
)
def test_capacity_sample_rejects_non_finite_acceptance_floats(field: str, value: object) -> None:
    values = {
        "workers": 1,
        "elapsed_seconds": 10.0,
        "completed_trials": 1,
        "failed_trials": 0,
        "inference_vram_margin": 0.30,
        "render_vram_margin": 0.30,
        "host_ram_margin": 1.0,
        "cpu_utilization": 0.2,
        "run_queue": 1,
        "inference_latency_seconds": 0.1,
        "inference_queue_depth": 0,
    }
    values[field] = value

    with pytest.raises(ValueError, match="finite"):
        CapacitySample(**values)


@pytest.mark.parametrize("value", (True, 1.0, float("nan"), float("inf")), ids=("bool", "float", "nan", "infinity"))
@pytest.mark.parametrize(
    "field",
    (
        "workers",
        "completed_trials",
        "failed_trials",
        "first_progress_workers",
        "stale_ipc_count",
        "run_queue",
        "inference_queue_depth",
    ),
)
def test_capacity_sample_rejects_non_integer_acceptance_counts(field: str, value: object) -> None:
    values = {
        "workers": 1,
        "elapsed_seconds": 10.0,
        "completed_trials": 1,
        "failed_trials": 0,
        "inference_vram_margin": 0.30,
        "render_vram_margin": 0.30,
        "first_progress_workers": 1,
        "stale_ipc_count": 0,
        "cpu_utilization": 0.2,
        "run_queue": 1,
        "inference_latency_seconds": 0.1,
        "inference_queue_depth": 0,
    }
    values[field] = value

    with pytest.raises(ValueError, match="integer"):
        CapacitySample(**values)


@pytest.mark.parametrize("value", (True, 16.0, float("nan"), float("inf")), ids=("bool", "float", "nan", "infinity"))
def test_capacity_rejects_non_integer_maximum_inference_queue_depth(value: object) -> None:
    with pytest.raises(ValueError, match="queue depth"):
        choose_worker_count((), max_inference_queue_depth=value)


@pytest.mark.parametrize("value", (True, float("nan"), float("inf"), float("-inf")), ids=("bool", "nan", "positive-infinity", "negative-infinity"))
def test_capacity_rejects_non_finite_minimum_gain(value: object) -> None:
    with pytest.raises(ValueError, match="minimum throughput gain"):
        choose_worker_count((), minimum_gain=value)


def test_capacity_rejects_boolean_maximum_inference_latency() -> None:
    with pytest.raises(ValueError, match="maximum inference latency"):
        choose_worker_count((), max_inference_latency_seconds=True)


def test_capacity_rejects_an_empty_sweep() -> None:
    assert choose_worker_count(()).accepted_workers == 0
