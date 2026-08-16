"""Fail-closed measured GPU loader warm-up receipts for runtime mixtures.

The CPU pilot characterizes decoding only.  This module deliberately makes the
paid admission decision from a real model-loaded CUDA measurement adapter.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Protocol

from lehome_train.io import canonical_json_sha256, sha256_file


WORKER_COUNTS = (0, 4, 8, 12, 16)
BURN_IN_STEPS = 10
MEASURED_STEPS = 50
PHYSICAL_BATCH_SIZE = 64
ACTION_HORIZON = 16
MAX_LOADER_WAIT_FRACTION = 0.10
MIN_GPU_BUSY_FRACTION = 0.70
MIN_GPU_UTILIZATION_PERCENT = 70.0
GIBIBYTE = 1024 ** 3
MIN_TOTAL_VRAM_BYTES = 90 * GIBIBYTE
MIN_FREE_VRAM_BYTES = 4 * GIBIBYTE
MIN_FREE_VRAM_FRACTION = 0.05
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True, slots=True)
class RuntimeState:
    """State read from the actual torch/model process, never caller claims."""

    torch_cuda_available: bool
    torch_cuda_initialized: bool
    model_loaded: bool
    hostname: str
    host_architecture: str
    torch_version: str
    cuda_version: str
    gpu_device_name: str
    gpu_uuid: str
    total_vram_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "torch_cuda_available": self.torch_cuda_available,
            "torch_cuda_initialized": self.torch_cuda_initialized,
            "model_loaded": self.model_loaded,
            "hostname": self.hostname,
            "host_architecture": self.host_architecture,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "gpu_device_name": self.gpu_device_name,
            "gpu_uuid": self.gpu_uuid,
            "total_vram_bytes": self.total_vram_bytes,
        }


@dataclass(frozen=True, slots=True)
class GpuWarmupMeasurement:
    """Actual per-worker results from one instrumented model training run."""

    decoded_samples: int
    measured_steps: int
    loader_wait_seconds: float
    step_seconds: float
    gpu_busy_seconds: float
    gpu_utilization_percent: float
    oom: bool
    error: str | None
    observed_batch_sizes: tuple[int, ...]
    loss_min: float | None
    loss_max: float | None
    loss_final: float | None
    peak_memory_allocated_bytes: int
    peak_memory_reserved_bytes: int
    minimum_free_vram_bytes: int
    samples_per_second: float
    step_latency_p50_seconds: float | None
    step_latency_p95_seconds: float | None
    materialization_proof: Mapping[str, object] | None


class GpuWarmupMetricsAdapter(Protocol):
    """Injects the only environment-specific portion of the warm-up.

    ``measure`` must run a loaded model and query loader timing plus GPU metrics
    from the live process/NVML; this contract intentionally has no synthetic
    defaults.
    """

    def runtime_state(self) -> RuntimeState: ...

    def measure(
        self, *, worker_count: int, burn_in_steps: int, measured_steps: int
    ) -> GpuWarmupMeasurement: ...


class TorchRuntimeWarmupMetricsAdapter:
    """Query torch in the loaded process and inject only live measurement code.

    The lifecycle supplies ``measure_live`` around the real DataLoader/model
    loop and NVML sampler; this adapter neither fabricates utilization nor
    assumes that a model was loaded merely because CUDA is importable.
    """

    def __init__(
        self,
        *,
        model_loaded: Callable[[], bool],
        measure_live: Callable[..., GpuWarmupMeasurement],
        runtime_state_live: Callable[[], RuntimeState],
    ) -> None:
        self._model_loaded = model_loaded
        self._measure_live = measure_live
        self._runtime_state_live = runtime_state_live

    def runtime_state(self) -> RuntimeState:
        result = self._runtime_state_live()
        if not isinstance(result, RuntimeState):
            raise TypeError("GPU warm-up live runtime-state probe returned an invalid receipt")
        if result.model_loaded != self._model_loaded():
            raise RuntimeError("GPU warm-up model-loaded probe drifted during runtime-state collection")
        return result

    def measure(
        self, *, worker_count: int, burn_in_steps: int, measured_steps: int
    ) -> GpuWarmupMeasurement:
        result = self._measure_live(
            worker_count=worker_count,
            burn_in_steps=burn_in_steps,
            measured_steps=measured_steps,
        )
        if not isinstance(result, GpuWarmupMeasurement):
            raise TypeError("GPU warm-up live measurement returned an invalid receipt")
        return result


def _exact(value: object, fields: set[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} has an incompatible schema")
    return value


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a SHA-256")
    return value


def _revision(value: object, label: str) -> str:
    if type(value) is not str or _REVISION.fullmatch(value) is None:
        raise ValueError(f"{label} must be an immutable revision")
    return value


def _finite(value: object, label: str, *, minimum: float = 0.0) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)) or float(value) < minimum:
        raise ValueError(f"{label} must be a finite number at least {minimum}")
    return float(value)


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer at least {minimum}")
    return value


def validate_cpu_pilot(cpu_pilot: Mapping[str, object]) -> str:
    """Accept the CPU pilot as characterization only, never an admission gate."""

    expected = {
        "schema_version", "kind", "canonical_worker_counts", "worker_counts",
        "sample_count_per_worker", "canonical_completion", "loader_throughput",
        "authenticated_evidence", "model_loaded", "gpu_initialized",
        "processor_contract", "representative", "timing_rows", "cache_cap",
        "native_x86_required", "timeout_seconds",
    }
    _exact(cpu_pilot, expected, "CPU runtime-mixture pilot")
    if (
        cpu_pilot["schema_version"] != 4
        or cpu_pilot["kind"] != "runtime_mixture_loader_pilot"
        or cpu_pilot["canonical_worker_counts"] != list(WORKER_COUNTS)
        or cpu_pilot["worker_counts"] != list(WORKER_COUNTS)
        or type(cpu_pilot["sample_count_per_worker"]) is not int
        or cpu_pilot["sample_count_per_worker"] < 100
        or cpu_pilot["canonical_completion"] is not True
        or cpu_pilot["model_loaded"] is not False
        or type(cpu_pilot["gpu_initialized"]) is not bool
        or cpu_pilot["processor_contract"] != "pinned_processor_integration_required"
        or not isinstance(cpu_pilot["representative"], Mapping)
        or not isinstance(cpu_pilot["timing_rows"], list)
        or type(cpu_pilot["cache_cap"]) is not int
        or cpu_pilot["cache_cap"] < 1
        or cpu_pilot["native_x86_required"] is not True
        or _finite(cpu_pilot["timeout_seconds"], "CPU runtime-mixture timeout", minimum=1.0) > 1800
        or not isinstance(cpu_pilot["authenticated_evidence"], Mapping)
    ):
        raise ValueError("CPU runtime-mixture pilot is not canonical characterization")
    throughput = cpu_pilot["loader_throughput"]
    if not isinstance(throughput, Mapping) or set(throughput) != {str(item) for item in WORKER_COUNTS}:
        raise ValueError("CPU runtime-mixture pilot lacks the exact worker sweep")
    for workers in WORKER_COUNTS:
        row = throughput[str(workers)]
        if not isinstance(row, Mapping) or row.get("decoded_samples") != cpu_pilot["sample_count_per_worker"]:
            raise ValueError("CPU runtime-mixture pilot did not decode every sample")
        _finite(row.get("samples_per_second"), "CPU runtime-mixture samples_per_second")
    return canonical_json_sha256(cpu_pilot)


def validate_warmup_binding(binding: Mapping[str, object]) -> dict[str, object]:
    """Validate immutable inputs that must match before a launch can proceed."""

    required = {
        "mixture", "deployment", "code", "parent_checkpoint",
        "physical_batch_size", "action_horizon",
    }
    _exact(binding, required, "GPU warm-up binding")
    mixture = _exact(
        binding["mixture"],
        {"repository", "revision", "mixture_id", "manifest_sha256", "window_index_sha256", "normalization_sha256", "experiment_manifest_sha256", "source_revisions"},
        "GPU warm-up mixture binding",
    )
    if type(mixture["repository"]) is not str or not mixture["repository"]:
        raise ValueError("GPU warm-up mixture repository is invalid")
    for name in ("revision",):
        _revision(mixture[name], f"GPU warm-up mixture {name}")
    for name in ("mixture_id", "manifest_sha256", "window_index_sha256", "normalization_sha256", "experiment_manifest_sha256"):
        _sha(mixture[name], f"GPU warm-up mixture {name}")
    sources = mixture["source_revisions"]
    if not isinstance(sources, Mapping) or not sources or not all(
        type(key) is str and key and type(value) is str and _REVISION.fullmatch(value)
        for key, value in sources.items()
    ):
        raise ValueError("GPU warm-up source revisions are invalid")
    deployment = _exact(binding["deployment"], {"oci_image_digest", "provider", "capability_sha256"}, "GPU warm-up deployment binding")
    if type(deployment["oci_image_digest"]) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", deployment["oci_image_digest"]) is None or type(deployment["provider"]) is not str or not deployment["provider"]:
        raise ValueError("GPU warm-up deployment identity is invalid")
    _sha(deployment["capability_sha256"], "GPU warm-up capability hash")
    code = _exact(binding["code"], {"repository_revision", "bundle_sha256", "isaac_groot_revision"}, "GPU warm-up code binding")
    _revision(code["repository_revision"], "GPU warm-up code revision")
    _sha(code["bundle_sha256"], "GPU warm-up code bundle hash")
    _revision(code["isaac_groot_revision"], "GPU warm-up Isaac GR00T revision")
    parent = _exact(binding["parent_checkpoint"], {"repository", "revision", "subpath", "artifact_sha256"}, "GPU warm-up parent checkpoint binding")
    if type(parent["repository"]) is not str or not parent["repository"] or parent["subpath"] != "policies/step-12000":
        raise ValueError("GPU warm-up parent checkpoint identity is invalid")
    _revision(parent["revision"], "GPU warm-up parent checkpoint revision")
    _sha(parent["artifact_sha256"], "GPU warm-up parent checkpoint hash")
    if (
        type(binding["physical_batch_size"]) is not int
        or type(binding["action_horizon"]) is not int
        or binding["physical_batch_size"] != PHYSICAL_BATCH_SIZE
        or binding["action_horizon"] != ACTION_HORIZON
    ):
        raise ValueError("GPU warm-up binding must use batch 64 and horizon 16")
    return {key: dict(value) if isinstance(value, Mapping) else value for key, value in binding.items()}


def bind_warmup_to_runtime_artifacts(
    *,
    binding: Mapping[str, object],
    manifest_path: str | Path,
    window_index_path: str | Path,
    normalization_path: str | Path,
    mounts_descriptor_path: str | Path,
) -> dict[str, object]:
    """Cross-check the receipt binding against the mounted immutable mixture.

    The deployment/code/parent identities arrive from the authenticated lifecycle
    artifact, while all mixture and source revisions are recovered from the
    exact mounted bytes consumed by the real loader.
    """

    checked = validate_warmup_binding(binding)
    from lehome_train.groot.runtime_mixture import load_runtime_contract

    manifest = Path(manifest_path)
    index = Path(window_index_path)
    normalization = Path(normalization_path)
    mounts = Path(mounts_descriptor_path)
    contract = load_runtime_contract(manifest, mounts)
    if contract.manifest.experiment_manifest_sha256 is None:
        raise ValueError("legacy runtime mixture is not admitted to this campaign")
    if (
        sha256_file(index) != contract.manifest.window_index_sha256
        or sha256_file(normalization) != contract.manifest.normalization_sha256
    ):
        raise ValueError("runtime warm-up binding has mismatched runtime artifacts")
    try:
        mount_document = json.loads(mounts.read_text(encoding="utf-8"))
        deployment_path = Path(mount_document["deployment_receipt_path"])
        deployment = json.loads(deployment_path.read_text(encoding="utf-8"))
        revision = deployment["immutable_revision"]
    except (KeyError, OSError, UnicodeError, json.JSONDecodeError, TypeError) as error:
        raise ValueError("runtime warm-up binding cannot recover mixture revision") from error
    actual_mixture = {
        "repository": contract.manifest.repository,
        "revision": revision,
        "mixture_id": contract.manifest.mixture_id,
        "manifest_sha256": sha256_file(manifest),
        "window_index_sha256": contract.manifest.window_index_sha256,
        "normalization_sha256": contract.manifest.normalization_sha256,
        "experiment_manifest_sha256": contract.manifest.experiment_manifest_sha256,
        "source_revisions": {
            source.source_id: source.publication["revision"]
            for source in contract.manifest.sources
        },
    }
    if checked["mixture"] != actual_mixture:
        raise ValueError("runtime warm-up binding does not match mounted mixture artifacts")
    return checked


def _materialization_proof(value: Mapping[str, object] | None) -> dict[str, object]:
    """Require h16/three-camera BC and rollout evidence from the live session."""

    if not isinstance(value, Mapping) or set(value) != {"bc", "rollout"}:
        raise ValueError("GPU warm-up lacks authenticated materialization proof")
    checked: dict[str, object] = {}
    for source_type in ("bc", "rollout"):
        item = value[source_type]
        if not isinstance(item, Mapping) or set(item) != {"source_type", "window_id", "action_horizon", "camera_count"}:
            raise ValueError("GPU warm-up materialization proof is invalid")
        if (
            item["source_type"] != source_type
            or type(item["window_id"]) is not str or not item["window_id"]
            or type(item["action_horizon"]) is not int or item["action_horizon"] != ACTION_HORIZON
            or type(item["camera_count"]) is not int or item["camera_count"] != 3
        ):
            raise ValueError("GPU warm-up materialization proof does not prove BC/rollout h16 cameras")
        checked[source_type] = dict(item)
    return checked


def _runtime_state(value: Mapping[str, object]) -> dict[str, object]:
    fields = {
        "torch_cuda_available", "torch_cuda_initialized", "model_loaded", "hostname",
        "host_architecture", "torch_version", "cuda_version", "gpu_device_name",
        "gpu_uuid", "total_vram_bytes",
    }
    state = _exact(value, fields, "GPU warm-up runtime state")
    if any(type(state[name]) is not bool for name in ("torch_cuda_available", "torch_cuda_initialized", "model_loaded")) or not all(state[name] for name in ("torch_cuda_available", "torch_cuda_initialized", "model_loaded")):
        raise ValueError("GPU warm-up runtime state does not prove CUDA and model loading")
    if any(type(state[name]) is not str or not state[name].strip() for name in ("hostname", "host_architecture", "torch_version", "cuda_version", "gpu_device_name", "gpu_uuid")):
        raise ValueError("GPU warm-up runtime provenance is incomplete")
    if type(state["total_vram_bytes"]) is not int or state["total_vram_bytes"] < MIN_TOTAL_VRAM_BYTES:
        raise ValueError("GPU warm-up runtime does not have the required 90 GiB VRAM")
    return dict(state)


def _candidate(
    *, worker_count: int, measurement: GpuWarmupMeasurement, total_vram_bytes: int
) -> dict[str, object]:
    if type(measurement.decoded_samples) is not int or measurement.decoded_samples < 0 or type(measurement.measured_steps) is not int or measurement.measured_steps < 0 or type(measurement.oom) is not bool or (measurement.error is not None and (type(measurement.error) is not str or not measurement.error)):
        raise ValueError("GPU warm-up adapter returned an invalid measurement")
    loader_wait = _finite(measurement.loader_wait_seconds, "GPU loader wait seconds")
    steps = _finite(measurement.step_seconds, "GPU step seconds")
    busy = _finite(measurement.gpu_busy_seconds, "GPU busy seconds")
    utilization = _finite(measurement.gpu_utilization_percent, "GPU utilization percent")
    samples_per_second = _finite(measurement.samples_per_second, "GPU warm-up samples per second")
    if samples_per_second <= 0.0 and not measurement.oom and measurement.error is None:
        raise ValueError("GPU warm-up samples per second must be positive")
    if type(measurement.peak_memory_allocated_bytes) is not int or measurement.peak_memory_allocated_bytes < 0 or type(measurement.peak_memory_reserved_bytes) is not int or measurement.peak_memory_reserved_bytes < 0 or type(measurement.minimum_free_vram_bytes) is not int or measurement.minimum_free_vram_bytes < 0:
        raise ValueError("GPU warm-up memory measurement is invalid")
    if measurement.peak_memory_allocated_bytes > measurement.peak_memory_reserved_bytes:
        raise ValueError("GPU warm-up allocated peak exceeds reserved peak")
    batches = list(measurement.observed_batch_sizes)
    if not all(type(batch) is int and batch > 0 for batch in batches):
        raise ValueError("GPU warm-up observed batch sizes are invalid")
    losses = (measurement.loss_min, measurement.loss_max, measurement.loss_final)
    loss_valid = all(type(loss) in (int, float) and math.isfinite(float(loss)) for loss in losses)
    if loss_valid and float(measurement.loss_min) > float(measurement.loss_max):
        raise ValueError("GPU warm-up loss summary is inverted")
    p50 = None if measurement.step_latency_p50_seconds is None else _finite(measurement.step_latency_p50_seconds, "GPU warm-up p50 step latency")
    p95 = None if measurement.step_latency_p95_seconds is None else _finite(measurement.step_latency_p95_seconds, "GPU warm-up p95 step latency")
    if p50 is not None and p95 is not None and p95 < p50:
        raise ValueError("GPU warm-up p95 step latency is below p50")
    proof = (
        None
        if measurement.oom or measurement.error is not None
        else _materialization_proof(measurement.materialization_proof)
    )
    if not measurement.oom and measurement.error is None and (
        batches != [PHYSICAL_BATCH_SIZE] * (BURN_IN_STEPS + MEASURED_STEPS)
        or not loss_valid or p50 is None or p95 is None
        or measurement.minimum_free_vram_bytes < MIN_FREE_VRAM_BYTES
        or measurement.minimum_free_vram_bytes / total_vram_bytes < MIN_FREE_VRAM_FRACTION
    ):
        raise ValueError("GPU warm-up successful candidate lacks required live admission evidence")
    if busy > steps:
        raise ValueError("GPU busy seconds exceed measured step seconds")
    wait_fraction = loader_wait / steps if steps else None
    busy_fraction = busy / steps if steps else None
    accepted = (
        not measurement.oom
        and measurement.error is None
        and measurement.measured_steps == MEASURED_STEPS
        and measurement.decoded_samples >= PHYSICAL_BATCH_SIZE * (BURN_IN_STEPS + MEASURED_STEPS)
        and batches == [PHYSICAL_BATCH_SIZE] * (BURN_IN_STEPS + MEASURED_STEPS)
        and loss_valid
        and p50 is not None and p95 is not None
        and measurement.minimum_free_vram_bytes >= MIN_FREE_VRAM_BYTES
        and measurement.minimum_free_vram_bytes / total_vram_bytes >= MIN_FREE_VRAM_FRACTION
        and wait_fraction is not None
        and busy_fraction is not None
        and wait_fraction <= MAX_LOADER_WAIT_FRACTION
        and busy_fraction >= MIN_GPU_BUSY_FRACTION
        and utilization >= MIN_GPU_UTILIZATION_PERCENT
    )
    return {
        "worker_count": worker_count,
        "burn_in_steps": BURN_IN_STEPS,
        "measured_steps": measurement.measured_steps,
        "decoded_samples": measurement.decoded_samples,
        "loader_wait_seconds": loader_wait,
        "step_seconds": steps,
        "loader_wait_fraction": wait_fraction,
        "gpu_busy_seconds": busy,
        "gpu_busy_fraction": busy_fraction,
        "gpu_utilization_percent": utilization,
        "observed_batch_sizes": batches,
        "loss_min": measurement.loss_min,
        "loss_max": measurement.loss_max,
        "loss_final": measurement.loss_final,
        "peak_memory_allocated_bytes": measurement.peak_memory_allocated_bytes,
        "peak_memory_reserved_bytes": measurement.peak_memory_reserved_bytes,
        "minimum_free_vram_bytes": measurement.minimum_free_vram_bytes,
        "samples_per_second": samples_per_second,
        "step_latency_p50_seconds": p50,
        "step_latency_p95_seconds": p95,
        "materialization_proof": proof,
        "oom": measurement.oom,
        "error": measurement.error,
        "accepted": accepted,
    }


def _gate() -> dict[str, object]:
    return {
        "worker_counts": list(WORKER_COUNTS),
        "burn_in_steps": BURN_IN_STEPS,
        "measured_steps": MEASURED_STEPS,
        "max_loader_wait_fraction": MAX_LOADER_WAIT_FRACTION,
        "min_gpu_busy_fraction": MIN_GPU_BUSY_FRACTION,
        "min_gpu_utilization_percent": MIN_GPU_UTILIZATION_PERCENT,
        "min_total_vram_bytes": MIN_TOTAL_VRAM_BYTES,
        "min_free_vram_bytes": MIN_FREE_VRAM_BYTES,
        "min_free_vram_fraction": MIN_FREE_VRAM_FRACTION,
    }


def _validate_gate(value: object) -> None:
    expected = _gate()
    gate = _exact(value, set(expected), "GPU warm-up receipt gate")
    workers = gate["worker_counts"]
    if not isinstance(workers, list) or any(type(worker) is not int for worker in workers):
        raise ValueError("GPU warm-up receipt worker gate is not integral")
    for name in ("burn_in_steps", "measured_steps", "min_total_vram_bytes", "min_free_vram_bytes"):
        _integer(gate[name], f"GPU warm-up receipt gate {name}")
    if gate != expected:
        raise ValueError("GPU warm-up receipt gate was caller-modified")


def _fastest_admitted_worker(candidates: list[Mapping[str, object]]) -> int | None:
    """Select highest measured stable throughput; ties use lower worker count."""

    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (float(item["samples_per_second"]), -int(item["worker_count"])),
    )["worker_count"]  # type: ignore[return-value]


def build_gpu_warmup_receipt(
    *, binding: Mapping[str, object], adapter: GpuWarmupMetricsAdapter
) -> dict[str, object]:
    """Measure every fixed candidate and return only a non-starving receipt."""

    checked_binding = validate_warmup_binding(binding)
    state = adapter.runtime_state()
    if not isinstance(state, RuntimeState):
        raise TypeError("GPU warm-up adapter must return RuntimeState")
    _runtime_state(state.to_dict())
    candidates = [
        _candidate(
            worker_count=workers,
            total_vram_bytes=state.total_vram_bytes,
            measurement=adapter.measure(
                worker_count=workers,
                burn_in_steps=BURN_IN_STEPS,
                measured_steps=MEASURED_STEPS,
            ),
        )
        for workers in WORKER_COUNTS
    ]
    selected = _fastest_admitted_worker([item for item in candidates if item["accepted"]])
    receipt = {
        "schema_version": 2,
        "kind": "runtime_mixture_gpu_warmup",
        "binding": checked_binding,
        "runtime_state": state.to_dict(),
        "gate": _gate(),
        "candidates": candidates,
        "selected_loader_workers": selected,
    }
    if selected is None:
        raise RuntimeError("no GPU warm-up worker count proved loader non-starvation")
    validate_gpu_warmup_receipt(receipt, expected_binding=checked_binding)
    return receipt


def warmup_from_request(path: str | Path) -> dict[str, object]:
    """Run one schema-checked GPU warm-up through the image production factory.

    The envelope carries only the immutable direct-GPU binding. Candidate
    metrics are accepted only from the loaded production factory's live
    model/DataLoader/NVML adapter.
    """

    from lehome_train.runtime import _request_arguments, load_runtime_adapter

    arguments = _request_arguments(
        path,
        command="runtime-gpu-warmup",
        expected_fields={"binding"},
    )
    binding = arguments["binding"]
    if not isinstance(binding, Mapping):
        raise ValueError("runtime GPU warm-up requires a binding object")
    production = load_runtime_adapter(None)
    adapter_factory = getattr(production, "runtime_gpu_warmup_adapter", None)
    if not callable(adapter_factory):
        raise RuntimeError("training runtime factory has no runtime GPU warm-up adapter")
    adapter = adapter_factory(dict(arguments))
    if not isinstance(adapter, TorchRuntimeWarmupMetricsAdapter):
        raise RuntimeError("training runtime factory returned no live Torch warm-up adapter")
    receipt = build_gpu_warmup_receipt(
        binding=binding,
        adapter=adapter,
    )
    validate_gpu_warmup_receipt(
        receipt,
        expected_binding=binding,
    )
    return receipt


def validate_gpu_warmup_receipt(
    receipt: Mapping[str, object], *, expected_binding: Mapping[str, object]
) -> int:
    """Verify that an existing receipt still proves the fixed safe gate."""

    required = {
        "schema_version", "kind", "binding", "runtime_state",
        "gate", "candidates", "selected_loader_workers",
    }
    _exact(receipt, required, "GPU warm-up receipt")
    if type(receipt["schema_version"]) is not int or receipt["schema_version"] != 2 or receipt["kind"] != "runtime_mixture_gpu_warmup":
        raise ValueError("GPU warm-up receipt has an incompatible schema")
    checked_binding = validate_warmup_binding(expected_binding)
    receipt_binding = validate_warmup_binding(receipt["binding"] if isinstance(receipt["binding"], Mapping) else {})
    if receipt_binding != checked_binding:
        raise ValueError("GPU warm-up receipt binding does not match this runtime")
    state = _runtime_state(receipt["runtime_state"] if isinstance(receipt["runtime_state"], Mapping) else {})
    _validate_gate(receipt["gate"])
    rows = receipt["candidates"]
    if not isinstance(rows, list) or len(rows) != len(WORKER_COUNTS):
        raise ValueError("GPU warm-up receipt candidates are incomplete")
    admitted: list[Mapping[str, object]] = []
    candidate_fields = {
        "worker_count", "burn_in_steps", "measured_steps", "decoded_samples",
        "loader_wait_seconds", "step_seconds", "loader_wait_fraction",
        "gpu_busy_seconds", "gpu_busy_fraction", "gpu_utilization_percent", "oom",
        "error", "observed_batch_sizes", "loss_min", "loss_max", "loss_final",
        "peak_memory_allocated_bytes", "peak_memory_reserved_bytes",
        "minimum_free_vram_bytes", "samples_per_second", "step_latency_p50_seconds",
        "step_latency_p95_seconds", "materialization_proof", "accepted",
    }
    for workers, row in zip(WORKER_COUNTS, rows, strict=True):
        item = _exact(row, candidate_fields, "GPU warm-up candidate")
        if (
            type(item["worker_count"]) is not int or item["worker_count"] != workers
            or type(item["burn_in_steps"]) is not int or item["burn_in_steps"] != BURN_IN_STEPS
            or type(item["measured_steps"]) is not int
            or type(item["decoded_samples"]) is not int or item["decoded_samples"] < 0
            or type(item["oom"]) is not bool
            or (item["error"] is not None and (type(item["error"]) is not str or not item["error"]))
            or type(item["accepted"]) is not bool
        ):
            raise ValueError("GPU warm-up candidate identity is invalid")
        wait = _finite(item["loader_wait_seconds"], "GPU warm-up loader wait seconds")
        steps = _finite(item["step_seconds"], "GPU warm-up step seconds")
        busy = _finite(item["gpu_busy_seconds"], "GPU warm-up busy seconds")
        utilization = _finite(item["gpu_utilization_percent"], "GPU warm-up utilization")
        samples_per_second = _finite(item["samples_per_second"], "GPU warm-up samples per second")
        if samples_per_second <= 0.0 and not item["oom"] and item["error"] is None:
            raise ValueError("GPU warm-up samples per second must be positive")
        if type(item["peak_memory_allocated_bytes"]) is not int or item["peak_memory_allocated_bytes"] < 0 or type(item["peak_memory_reserved_bytes"]) is not int or item["peak_memory_reserved_bytes"] < 0 or item["peak_memory_allocated_bytes"] > item["peak_memory_reserved_bytes"] or type(item["minimum_free_vram_bytes"]) is not int or item["minimum_free_vram_bytes"] < 0:
            raise ValueError("GPU warm-up candidate memory telemetry is invalid")
        batches = item["observed_batch_sizes"]
        if not isinstance(batches, list) or any(type(batch) is not int or batch <= 0 for batch in batches):
            raise ValueError("GPU warm-up candidate batch observations are invalid")
        losses = (item["loss_min"], item["loss_max"], item["loss_final"])
        loss_valid = all(type(loss) in (int, float) and math.isfinite(float(loss)) for loss in losses)
        if loss_valid and float(item["loss_min"]) > float(item["loss_max"]):
            raise ValueError("GPU warm-up candidate loss summary is invalid")
        p50 = None if item["step_latency_p50_seconds"] is None else _finite(item["step_latency_p50_seconds"], "GPU warm-up p50 step latency")
        p95 = None if item["step_latency_p95_seconds"] is None else _finite(item["step_latency_p95_seconds"], "GPU warm-up p95 step latency")
        if p50 is not None and p95 is not None and p95 < p50:
            raise ValueError("GPU warm-up candidate latency summary is invalid")
        if item["oom"] or item["error"] is not None:
            if item["materialization_proof"] is not None:
                raise ValueError("GPU warm-up failed candidate must not claim materialization proof")
        else:
            _materialization_proof(item["materialization_proof"] if isinstance(item["materialization_proof"], Mapping) else None)
        if not item["oom"] and item["error"] is None and (
            batches != [PHYSICAL_BATCH_SIZE] * (BURN_IN_STEPS + MEASURED_STEPS)
            or not loss_valid or p50 is None or p95 is None
            or item["minimum_free_vram_bytes"] < MIN_FREE_VRAM_BYTES
            or item["minimum_free_vram_bytes"] / state["total_vram_bytes"] < MIN_FREE_VRAM_FRACTION
        ):
            raise ValueError("GPU warm-up candidate lacks required live admission evidence")
        if busy > steps:
            raise ValueError("GPU warm-up candidate metric fraction drift")
        if steps == 0:
            if item["loader_wait_fraction"] is not None or item["gpu_busy_fraction"] is not None:
                raise ValueError("GPU warm-up candidate metric fraction drift")
            wait_fraction = busy_fraction = None
        else:
            wait_fraction = _finite(item["loader_wait_fraction"], "GPU warm-up loader wait fraction")
            busy_fraction = _finite(item["gpu_busy_fraction"], "GPU warm-up busy fraction")
            if not math.isclose(wait_fraction, wait / steps, rel_tol=0.0, abs_tol=1e-12) or not math.isclose(busy_fraction, busy / steps, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("GPU warm-up candidate metric fraction drift")
        accepted = (
            not item["oom"] and item["error"] is None and item["measured_steps"] == MEASURED_STEPS
            and item["decoded_samples"] >= PHYSICAL_BATCH_SIZE * (BURN_IN_STEPS + MEASURED_STEPS)
            and batches == [PHYSICAL_BATCH_SIZE] * (BURN_IN_STEPS + MEASURED_STEPS) and loss_valid
            and p50 is not None and p95 is not None
            and item["minimum_free_vram_bytes"] >= MIN_FREE_VRAM_BYTES
            and item["minimum_free_vram_bytes"] / state["total_vram_bytes"] >= MIN_FREE_VRAM_FRACTION
            and wait_fraction is not None and busy_fraction is not None
            and wait_fraction <= MAX_LOADER_WAIT_FRACTION and busy_fraction >= MIN_GPU_BUSY_FRACTION
            and utilization >= MIN_GPU_UTILIZATION_PERCENT
        )
        if item["accepted"] is not accepted:
            raise ValueError("GPU warm-up candidate acceptance was tampered")
        if accepted:
            admitted.append(item)
    selected = _fastest_admitted_worker(admitted)
    if (
        type(receipt["selected_loader_workers"]) is not int
        or selected is None
        or receipt["selected_loader_workers"] != selected
    ):
        raise ValueError("GPU warm-up receipt does not select the fastest admitted worker")
    return selected
