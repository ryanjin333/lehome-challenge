"""Contract tests for the measured runtime-mixture GPU non-starvation gate."""

from __future__ import annotations

import pytest
import sys
import types


HASH = "a" * 64
REVISION = "b" * 40


def _cpu_pilot() -> dict[str, object]:
    return {
        "schema_version": 4,
        "kind": "runtime_mixture_loader_pilot",
        "model_loaded": False,
        "gpu_initialized": False,
        "processor_contract": "pinned_processor_integration_required",
        "representative": {"bc_window_id": "bc-1", "rollout_window_id": "rollout-1", "three_cameras": True, "action_horizon": 16},
        "canonical_worker_counts": [0, 4, 8, 16, 24],
        "worker_counts": [0, 4, 8, 16, 24],
        "sample_count_per_worker": 100,
        "canonical_completion": True,
        "loader_throughput": {
            str(workers): {"decoded_samples": 100, "samples_per_second": float(workers + 1)}
            for workers in (0, 4, 8, 16, 24)
        },
        "timing_rows": [],
        "authenticated_evidence": {"mixture_id": HASH},
        "cache_cap": 8,
        "native_x86_required": True,
        "timeout_seconds": 60.0,
    }


def _binding() -> dict[str, object]:
    return {
        "mixture": {
            "repository": "ryanjin333/lehome-groot-n17-data",
            "revision": REVISION,
            "mixture_id": HASH,
            "manifest_sha256": HASH,
            "window_index_sha256": "c" * 64,
            "normalization_sha256": "d" * 64,
            "source_revisions": {"bc": "e" * 40, "rollout": "f" * 40},
        },
        "deployment": {
            "oci_image_digest": "sha256:" + "1" * 64,
            "provider": "vast.ai",
            "capability_sha256": "2" * 64,
        },
        "code": {
            "repository_revision": "3" * 40,
            "bundle_sha256": "4" * 64,
            "isaac_groot_revision": "5" * 40,
        },
        "parent_checkpoint": {
            "repository": "ryanjin333/lehome-groot-n17-models",
            "revision": "6" * 40,
            "subpath": "policies/step-12000",
            "artifact_sha256": "7" * 64,
        },
        "physical_batch_size": 64,
        "action_horizon": 16,
    }


class _Adapter:
    def __init__(self, *, viable: set[int] = {4, 8}, oom: int | None = None) -> None:
        self.viable = viable
        self.oom = oom
        self.calls: list[tuple[int, int, int]] = []

    def runtime_state(self):
        from lehome_train.groot.runtime_mixture_warmup import RuntimeState

        return RuntimeState(torch_cuda_available=True, torch_cuda_initialized=True, model_loaded=True)

    def measure(self, *, worker_count: int, burn_in_steps: int, measured_steps: int):
        from lehome_train.groot.runtime_mixture_warmup import GpuWarmupMeasurement

        self.calls.append((worker_count, burn_in_steps, measured_steps))
        if worker_count == self.oom:
            return GpuWarmupMeasurement(
                decoded_samples=0, measured_steps=0, loader_wait_seconds=0.0,
                step_seconds=0.0, gpu_busy_seconds=0.0,
                gpu_utilization_percent=0.0, oom=True, error="CUDA out of memory",
            )
        viable = worker_count in self.viable
        return GpuWarmupMeasurement(
            decoded_samples=64 * (burn_in_steps + measured_steps),
            measured_steps=measured_steps,
            loader_wait_seconds=5.0 if viable else 20.0,
            step_seconds=100.0,
            gpu_busy_seconds=80.0 if viable else 50.0,
            gpu_utilization_percent=80.0 if viable else 50.0,
            oom=False,
            error=None,
        )


def _receipt(adapter: _Adapter | None = None) -> dict[str, object]:
    from lehome_train.groot.runtime_mixture_warmup import build_gpu_warmup_receipt

    return build_gpu_warmup_receipt(
        cpu_pilot=_cpu_pilot(), binding=_binding(), adapter=adapter or _Adapter()
    )


def test_gpu_warmup_selects_lowest_worker_that_meets_fixed_gate() -> None:
    adapter = _Adapter(viable={4, 8, 16})
    receipt = _receipt(adapter)

    assert receipt["selected_loader_workers"] == 4
    assert adapter.calls == [(0, 10, 50), (4, 10, 50), (8, 10, 50), (16, 10, 50), (24, 10, 50)]
    assert receipt["gate"]["max_loader_wait_fraction"] == 0.10


def test_gpu_warmup_selection_is_deterministic_not_fastest() -> None:
    first = _receipt(_Adapter(viable={8, 16}))
    second = _receipt(_Adapter(viable={8, 16}))

    assert first["selected_loader_workers"] == second["selected_loader_workers"] == 8
    assert first["candidates"] == second["candidates"]


def test_gpu_warmup_rejects_oom_even_when_other_metrics_claim_success() -> None:
    with pytest.raises(RuntimeError, match="no GPU warm-up worker count"):
        _receipt(_Adapter(viable={4}, oom=4))


def test_gpu_warmup_records_an_oom_candidate_but_can_select_another_worker() -> None:
    receipt = _receipt(_Adapter(viable={8}, oom=0))

    assert receipt["selected_loader_workers"] == 8
    assert receipt["candidates"][0]["oom"] is True
    assert receipt["candidates"][0]["loader_wait_fraction"] is None


def test_gpu_warmup_rejects_when_no_worker_meets_loader_and_gpu_gate() -> None:
    with pytest.raises(RuntimeError, match="no GPU warm-up worker count"):
        _receipt(_Adapter(viable=set()))


def test_warmup_receipt_rejects_tampering_and_mismatched_bound_identities() -> None:
    from lehome_train.groot.runtime_mixture_warmup import validate_gpu_warmup_receipt

    receipt = _receipt()
    tampered = dict(receipt)
    tampered["selected_loader_workers"] = 8
    with pytest.raises(ValueError, match="lowest"):
        validate_gpu_warmup_receipt(tampered, expected_binding=_binding())

    for field, value in (
        ("deployment", {**_binding()["deployment"], "provider": "other"}),
        ("parent_checkpoint", {**_binding()["parent_checkpoint"], "revision": "9" * 40}),
        ("mixture", {**_binding()["mixture"], "manifest_sha256": "9" * 64}),
    ):
        expected = _binding()
        expected[field] = value
        with pytest.raises(ValueError, match="binding"):
            validate_gpu_warmup_receipt(receipt, expected_binding=expected)


def test_warmup_receipt_does_not_allow_caller_threshold_injection() -> None:
    from lehome_train.groot.runtime_mixture_warmup import build_gpu_warmup_receipt

    with pytest.raises(TypeError):
        build_gpu_warmup_receipt(
            cpu_pilot=_cpu_pilot(), binding=_binding(), adapter=_Adapter(),
            max_loader_wait_fraction=1.0,
        )


def test_warmup_receipt_rejects_missing_actual_cuda_or_model_state() -> None:
    from lehome_train.groot.runtime_mixture_warmup import RuntimeState, validate_gpu_warmup_receipt

    receipt = _receipt()
    receipt = dict(receipt)
    receipt["runtime_state"] = RuntimeState(
        torch_cuda_available=True, torch_cuda_initialized=True, model_loaded=False
    ).to_dict()
    with pytest.raises(ValueError, match="runtime state"):
        validate_gpu_warmup_receipt(receipt, expected_binding=_binding())


def test_live_adapter_queries_torch_state_and_delegates_live_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lehome_train.groot.runtime_mixture_warmup import (
        GpuWarmupMeasurement,
        TorchRuntimeWarmupMetricsAdapter,
    )

    monkeypatch.setitem(
        sys.modules, "torch", types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True, is_initialized=lambda: True)
        )
    )
    expected = GpuWarmupMeasurement(64 * 60, 50, 1.0, 20.0, 18.0, 90.0, False, None)
    adapter = TorchRuntimeWarmupMetricsAdapter(
        model_loaded=lambda: True,
        measure_live=lambda **kwargs: expected,
    )

    assert adapter.runtime_state().to_dict() == {
        "torch_cuda_available": True, "torch_cuda_initialized": True, "model_loaded": True,
    }
    assert adapter.measure(worker_count=4, burn_in_steps=10, measured_steps=50) == expected
