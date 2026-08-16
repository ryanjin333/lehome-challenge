"""Contract tests for the measured runtime-mixture GPU non-starvation gate."""

from __future__ import annotations

import pytest
import sys
import types
import json
from pathlib import Path


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
        "canonical_worker_counts": [0, 4, 8, 12, 16],
        "worker_counts": [0, 4, 8, 12, 16],
        "sample_count_per_worker": 100,
        "canonical_completion": True,
        "loader_throughput": {
            str(workers): {"decoded_samples": 100, "samples_per_second": float(workers + 1)}
            for workers in (0, 4, 8, 12, 16)
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
            "experiment_manifest_sha256": "8" * 64,
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
    def __init__(
        self, *, viable: set[int] = {4, 8}, oom: int | None = None,
        samples_per_second: dict[int, float] | None = None,
    ) -> None:
        self.viable = viable
        self.oom = oom
        self.samples_per_second = samples_per_second or {}
        self.calls: list[tuple[int, int, int]] = []

    def runtime_state(self):
        from lehome_train.groot.runtime_mixture_warmup import RuntimeState

        return RuntimeState(
            torch_cuda_available=True, torch_cuda_initialized=True, model_loaded=True,
            hostname="gpu-host", host_architecture="x86_64", torch_version="2.7.0",
            cuda_version="12.8", gpu_device_name="NVIDIA RTX PRO 6000",
            gpu_uuid="GPU-1234", total_vram_bytes=96 * 1024**3,
        )

    def measure(self, *, worker_count: int, burn_in_steps: int, measured_steps: int):
        from lehome_train.groot.runtime_mixture_warmup import GpuWarmupMeasurement

        self.calls.append((worker_count, burn_in_steps, measured_steps))
        if worker_count == self.oom:
            return _live_measurement(
                decoded_samples=0, measured_steps=0, loader_wait_seconds=0.0,
                step_seconds=0.0, gpu_busy_seconds=0.0, gpu_utilization_percent=0.0,
                oom=True, error="CUDA out of memory", observed_batch_sizes=(),
                loss_min=None, loss_max=None, loss_final=None,
                samples_per_second=0.0, step_latency_p50_seconds=None,
                step_latency_p95_seconds=None,
                materialization_proof=None,
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
            observed_batch_sizes=(64,) * (burn_in_steps + measured_steps),
            loss_min=0.1,
            loss_max=0.2,
            loss_final=0.15,
            peak_memory_allocated_bytes=32 * 1024**3,
            peak_memory_reserved_bytes=40 * 1024**3,
            minimum_free_vram_bytes=20 * 1024**3,
            samples_per_second=self.samples_per_second.get(worker_count, 320.0),
            step_latency_p50_seconds=0.2,
            step_latency_p95_seconds=0.25,
            materialization_proof={
                "bc": {"source_type": "bc", "window_id": "bc-window", "action_horizon": 16, "camera_count": 3},
                "rollout": {"source_type": "rollout", "window_id": "rollout-window", "action_horizon": 16, "camera_count": 3},
            },
        )


def _live_measurement(**overrides: object):
    from lehome_train.groot.runtime_mixture_warmup import GpuWarmupMeasurement

    values: dict[str, object] = {
        "decoded_samples": 64 * 60, "measured_steps": 50,
        "loader_wait_seconds": 1.0, "step_seconds": 20.0,
        "gpu_busy_seconds": 18.0, "gpu_utilization_percent": 90.0,
        "oom": False, "error": None, "observed_batch_sizes": (64,) * 60,
        "loss_min": 0.1, "loss_max": 0.2, "loss_final": 0.15,
        "peak_memory_allocated_bytes": 32 * 1024**3,
        "peak_memory_reserved_bytes": 40 * 1024**3,
        "minimum_free_vram_bytes": 20 * 1024**3,
        "samples_per_second": 320.0, "step_latency_p50_seconds": 0.2,
        "step_latency_p95_seconds": 0.25,
        "materialization_proof": {
            "bc": {"source_type": "bc", "window_id": "bc-window", "action_horizon": 16, "camera_count": 3},
            "rollout": {"source_type": "rollout", "window_id": "rollout-window", "action_horizon": 16, "camera_count": 3},
        },
    }
    values.update(overrides)
    return GpuWarmupMeasurement(**values)  # type: ignore[arg-type]


def _receipt(adapter: _Adapter | None = None) -> dict[str, object]:
    from lehome_train.groot.runtime_mixture_warmup import build_gpu_warmup_receipt

    return build_gpu_warmup_receipt(
        binding=_binding(), adapter=adapter or _Adapter()
    )


def test_gpu_warmup_selects_lowest_worker_that_meets_fixed_gate() -> None:
    adapter = _Adapter(viable={4, 8, 16})
    receipt = _receipt(adapter)

    assert receipt["selected_loader_workers"] == 4
    assert adapter.calls == [(0, 10, 50), (4, 10, 50), (8, 10, 50), (12, 10, 50), (16, 10, 50)]
    assert receipt["gate"]["max_loader_wait_fraction"] == 0.10


def test_gpu_warmup_selects_the_fastest_stable_admitted_worker() -> None:
    receipt = _receipt(_Adapter(
        viable={0, 4, 8, 12, 16},
        samples_per_second={0: 100.0, 4: 200.0, 8: 300.0, 12: 400.0, 16: 250.0},
    ))

    assert receipt["selected_loader_workers"] == 12


def test_gpu_warmup_breaks_equal_throughput_ties_by_lower_worker_count() -> None:
    receipt = _receipt(_Adapter(
        viable={4, 12}, samples_per_second={4: 400.0, 12: 400.0},
    ))

    assert receipt["selected_loader_workers"] == 4


def test_gpu_warmup_receipt_requires_live_gpu_memory_loss_latency_and_materialization_evidence() -> None:
    from lehome_train.groot.runtime_mixture_warmup import validate_gpu_warmup_receipt

    receipt = _receipt()
    assert receipt["runtime_state"]["total_vram_bytes"] == 96 * 1024**3
    assert receipt["candidates"][0]["observed_batch_sizes"] == [64] * 60
    assert receipt["candidates"][0]["materialization_proof"]["bc"]["camera_count"] == 3

    for path, value in (
        (("runtime_state", "hostname"), ""),
        (("runtime_state", "host_architecture"), ""),
        (("runtime_state", "torch_version"), ""),
        (("runtime_state", "cuda_version"), ""),
        (("runtime_state", "gpu_device_name"), ""),
        (("runtime_state", "gpu_uuid"), ""),
        (("runtime_state", "total_vram_bytes"), 89 * 1024**3),
        (("candidates", 0, "minimum_free_vram_bytes"), 3 * 1024**3),
        (("candidates", 0, "observed_batch_sizes"), [64, 32]),
        (("candidates", 0, "loss_final"), float("inf")),
        (("candidates", 0, "peak_memory_allocated_bytes"), 41 * 1024**3),
        (("candidates", 0, "peak_memory_reserved_bytes"), 1 * 1024**3),
        (("candidates", 0, "samples_per_second"), 0.0),
        (("candidates", 0, "step_latency_p50_seconds"), float("inf")),
        (("candidates", 0, "step_latency_p95_seconds"), 0.1),
        (("candidates", 0, "materialization_proof"), {}),
        (("binding", "mixture", "experiment_manifest_sha256"), "9" * 64),
    ):
        tampered = json.loads(json.dumps(receipt))
        target: object = tampered
        for key in path[:-1]:
            target = target[key]  # type: ignore[index]
        target[path[-1]] = value  # type: ignore[index]
        with pytest.raises(ValueError):
            validate_gpu_warmup_receipt(tampered, expected_binding=_binding())


def test_gpu_warmup_receipt_rejects_float_and_bool_discrete_evidence() -> None:
    from lehome_train.groot.runtime_mixture_warmup import validate_gpu_warmup_receipt

    receipt = _receipt()
    for path, value in (
        (("schema_version",), 2.0),
        (("binding", "physical_batch_size"), 64.0),
        (("binding", "action_horizon"), 16.0),
        (("gate", "worker_counts"), [False, 4, 8, 12, 16]),
        (("gate", "burn_in_steps"), 10.0),
        (("gate", "min_total_vram_bytes"), float(90 * 1024**3)),
        (("gate", "min_free_vram_bytes"), float(4 * 1024**3)),
        (("runtime_state", "total_vram_bytes"), float(96 * 1024**3)),
        (("candidates", 0, "worker_count"), False),
        (("candidates", 0, "burn_in_steps"), 10.0),
        (("candidates", 0, "measured_steps"), 50.0),
        (("candidates", 0, "decoded_samples"), float(64 * 60)),
        (("candidates", 0, "peak_memory_allocated_bytes"), float(32 * 1024**3)),
        (("candidates", 0, "peak_memory_reserved_bytes"), float(40 * 1024**3)),
        (("candidates", 0, "minimum_free_vram_bytes"), float(20 * 1024**3)),
        (("candidates", 0, "materialization_proof", "bc", "action_horizon"), 16.0),
        (("candidates", 0, "materialization_proof", "rollout", "camera_count"), 3.0),
        (("selected_loader_workers",), 4.0),
    ):
        tampered = json.loads(json.dumps(receipt))
        target: object = tampered
        for key in path[:-1]:
            target = target[key]  # type: ignore[index]
        target[path[-1]] = value  # type: ignore[index]
        with pytest.raises(ValueError):
            validate_gpu_warmup_receipt(tampered, expected_binding=_binding())


def test_gpu_warmup_receipt_requires_only_the_direct_gpu_binding() -> None:
    from lehome_train.groot.runtime_mixture_warmup import build_gpu_warmup_receipt

    receipt = build_gpu_warmup_receipt(binding=_binding(), adapter=_Adapter())

    assert "cpu_pilot_sha256" not in receipt
    assert receipt["selected_loader_workers"] == 4


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
    assert receipt["candidates"][0]["materialization_proof"] is None


def test_gpu_warmup_rejects_when_no_worker_meets_loader_and_gpu_gate() -> None:
    with pytest.raises(RuntimeError, match="no GPU warm-up worker count"):
        _receipt(_Adapter(viable=set()))


def test_warmup_receipt_rejects_tampering_and_mismatched_bound_identities() -> None:
    from lehome_train.groot.runtime_mixture_warmup import validate_gpu_warmup_receipt

    receipt = _receipt()
    tampered = dict(receipt)
    tampered["selected_loader_workers"] = 8
    with pytest.raises(ValueError, match="fastest"):
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


def test_warmup_receipt_rejects_tampered_nonfastest_selected_worker() -> None:
    from lehome_train.groot.runtime_mixture_warmup import validate_gpu_warmup_receipt

    receipt = _receipt(_Adapter(
        viable={4, 12}, samples_per_second={4: 300.0, 12: 400.0},
    ))
    assert receipt["selected_loader_workers"] == 12
    receipt["selected_loader_workers"] = 4

    with pytest.raises(ValueError, match="fastest"):
        validate_gpu_warmup_receipt(receipt, expected_binding=_binding())


def test_warmup_receipt_does_not_allow_caller_threshold_injection() -> None:
    from lehome_train.groot.runtime_mixture_warmup import build_gpu_warmup_receipt

    with pytest.raises(TypeError):
        build_gpu_warmup_receipt(
            binding=_binding(), adapter=_Adapter(),
            max_loader_wait_fraction=1.0,
        )


def test_warmup_receipt_rejects_missing_actual_cuda_or_model_state() -> None:
    from lehome_train.groot.runtime_mixture_warmup import RuntimeState, validate_gpu_warmup_receipt

    receipt = _receipt()
    receipt = dict(receipt)
    receipt["runtime_state"] = RuntimeState(
        torch_cuda_available=True, torch_cuda_initialized=True, model_loaded=False,
        hostname="gpu-host", host_architecture="x86_64", torch_version="2.7.0",
        cuda_version="12.8", gpu_device_name="NVIDIA RTX PRO 6000",
        gpu_uuid="GPU-1234", total_vram_bytes=96 * 1024**3,
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
    expected = _live_measurement()
    adapter = TorchRuntimeWarmupMetricsAdapter(
        model_loaded=lambda: True,
        measure_live=lambda **kwargs: expected,
        runtime_state_live=_Adapter().runtime_state,
    )

    assert adapter.runtime_state().to_dict() == _Adapter().runtime_state().to_dict()
    assert adapter.measure(worker_count=4, burn_in_steps=10, measured_steps=50) == expected


def test_warmup_request_uses_production_factory_live_adapter_not_authored_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lehome_train.groot.runtime_mixture_warmup import (
        GpuWarmupMeasurement,
        TorchRuntimeWarmupMetricsAdapter,
        warmup_from_request,
    )

    calls: list[dict[str, object]] = []

    class ProductionFactory:
        def __getattr__(self, _name: str):
            return lambda *_args: {"status": "unused"}

        def runtime_gpu_warmup_adapter(self, arguments: dict[str, object]):
            calls.append(arguments)
            return TorchRuntimeWarmupMetricsAdapter(
                model_loaded=lambda: True,
                measure_live=lambda **_kwargs: _live_measurement(),
                runtime_state_live=_Adapter().runtime_state,
            )

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True, is_initialized=lambda: True)
    ))
    monkeypatch.setitem(sys.modules, "warmup_production", types.SimpleNamespace(
        create=lambda: ProductionFactory()
    ))
    monkeypatch.setenv("LEHOME_TRAIN_RUNTIME_FACTORY", "warmup_production:create")
    request = tmp_path / "runtime-warmup.json"
    request.write_text(json.dumps({
        "schema_version": 1,
        "command": "runtime-gpu-warmup",
        "arguments": {"binding": _binding()},
    }), encoding="utf-8")

    receipt = warmup_from_request(request)

    assert receipt["selected_loader_workers"] == 0
    assert calls == [{"binding": _binding()}]


def test_warmup_request_envelope_excludes_cpu_pilot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lehome_train.groot.runtime_mixture_warmup import (
        GpuWarmupMeasurement,
        TorchRuntimeWarmupMetricsAdapter,
        warmup_from_request,
    )

    class ProductionFactory:
        def __getattr__(self, _name: str):
            return lambda *_args: {"status": "unused"}

        def runtime_gpu_warmup_adapter(self, arguments: dict[str, object]):
            assert arguments == {"binding": _binding()}
            return TorchRuntimeWarmupMetricsAdapter(
                model_loaded=lambda: True,
                measure_live=lambda **_kwargs: _live_measurement(),
                runtime_state_live=_Adapter().runtime_state,
            )

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True, is_initialized=lambda: True)
    ))
    monkeypatch.setitem(sys.modules, "direct_gpu_production", types.SimpleNamespace(
        create=lambda: ProductionFactory()
    ))
    monkeypatch.setenv("LEHOME_TRAIN_RUNTIME_FACTORY", "direct_gpu_production:create")
    request = tmp_path / "runtime-warmup.json"
    request.write_text(json.dumps({
        "schema_version": 1,
        "command": "runtime-gpu-warmup",
        "arguments": {"binding": _binding()},
    }), encoding="utf-8")

    receipt = warmup_from_request(request)

    assert "cpu_pilot_sha256" not in receipt


def test_warmup_request_rejects_preauthored_candidate_rows_and_missing_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lehome_train.groot.runtime_mixture_warmup import warmup_from_request

    request = tmp_path / "runtime-warmup.json"
    request.write_text(json.dumps({
        "schema_version": 1,
        "command": "runtime-gpu-warmup",
        "arguments": {
            "binding": _binding(),
            "candidates": [{"accepted": True}],
        },
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete or unknown"):
        warmup_from_request(request)

    request.write_text(json.dumps({
        "schema_version": 1,
        "command": "runtime-gpu-warmup",
        "arguments": {"binding": _binding()},
    }), encoding="utf-8")
    monkeypatch.delenv("LEHOME_TRAIN_RUNTIME_FACTORY", raising=False)
    with pytest.raises(RuntimeError, match="no training runtime factory"):
        warmup_from_request(request)
