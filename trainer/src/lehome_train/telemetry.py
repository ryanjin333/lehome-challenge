"""Lazy, CPU-safe GPU/host sampling and deterministic smoke summaries."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from threading import Event, Thread
from time import monotonic
from typing import Callable, Protocol, TypeVar

from lehome_train.models import StrictModel


@dataclass(frozen=True, slots=True)
class TelemetrySample(StrictModel):
    """One timestamped sample; optional fields are unavailable measurements."""

    timestamp_seconds: float
    allocated_vram_bytes: int | None
    reserved_vram_bytes: int | None
    free_vram_bytes: int | None
    gpu_utilization_percent: float | None
    power_watts: float | None
    temperature_celsius: float | None
    host_memory_bytes: int | None

    def __post_init__(self) -> None:
        StrictModel.__post_init__(self)
        if self.timestamp_seconds < 0:
            raise ValueError("telemetry timestamp must be nonnegative")
        for name in (
            "allocated_vram_bytes",
            "reserved_vram_bytes",
            "free_vram_bytes",
            "host_memory_bytes",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"telemetry {name} must be nonnegative")
        for name in (
            "gpu_utilization_percent",
            "power_watts",
            "temperature_celsius",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"telemetry {name} must be nonnegative")
        if self.gpu_utilization_percent is not None and self.gpu_utilization_percent > 100:
            raise ValueError("telemetry GPU utilization must not exceed 100 percent")


@dataclass(frozen=True, slots=True)
class TelemetrySummary(StrictModel):
    """Peak resources plus timings that exclude initialization and warm-up."""

    initialization_seconds: float
    warmup_seconds: float
    steady_state_seconds: float
    peak_allocated_vram_bytes: int
    peak_reserved_vram_bytes: int
    minimum_free_vram_bytes: int
    peak_gpu_utilization_percent: float | None
    peak_power_watts: float | None
    peak_temperature_celsius: float | None
    peak_host_memory_bytes: int | None
    steady_steps_per_second: float
    samples_per_second: float

    def __post_init__(self) -> None:
        StrictModel.__post_init__(self)
        for name in (
            "initialization_seconds",
            "warmup_seconds",
            "steady_state_seconds",
            "steady_steps_per_second",
            "samples_per_second",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"telemetry {name} must be nonnegative")
        for name in (
            "peak_allocated_vram_bytes",
            "peak_reserved_vram_bytes",
            "minimum_free_vram_bytes",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"telemetry {name} must be nonnegative")


class TelemetrySampler(Protocol):
    """Injected sampler contract used by paid runs and CPU-only tests."""

    def sample(self) -> TelemetrySample: ...


def _peak(samples: tuple[TelemetrySample, ...], name: str) -> int | float | None:
    values = tuple(
        value
        for sample in samples
        if (value := getattr(sample, name)) is not None
    )
    return max(values) if values else None


def summarize_telemetry(
    samples: tuple[TelemetrySample, ...],
    *,
    initialization_seconds: float,
    warmup_seconds: float,
    steady_state_seconds: float,
    steady_state_optimizer_steps: int,
    physical_batch_size: int,
) -> TelemetrySummary:
    """Aggregate raw samples without counting setup/warm-up as throughput."""

    if not samples:
        raise ValueError("at least one telemetry sample is required")
    if any(
        later.timestamp_seconds < earlier.timestamp_seconds
        for earlier, later in zip(samples, samples[1:])
    ):
        raise ValueError("telemetry timestamps must be monotonic")
    for name, value in (
        ("initialization", initialization_seconds),
        ("warmup", warmup_seconds),
        ("steady-state", steady_state_seconds),
    ):
        if type(value) not in (int, float) or not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"{name} duration must be finite and nonnegative")
    if type(steady_state_optimizer_steps) is not int or steady_state_optimizer_steps < 0:
        raise ValueError("steady-state optimizer steps must be nonnegative")
    if type(physical_batch_size) is not int or physical_batch_size <= 0:
        raise ValueError("physical batch size must be positive")
    if steady_state_optimizer_steps and steady_state_seconds <= 0:
        raise ValueError("positive steady-state progress requires positive duration")

    allocated = _peak(samples, "allocated_vram_bytes")
    reserved = _peak(samples, "reserved_vram_bytes")
    free_values = tuple(
        sample.free_vram_bytes
        for sample in samples
        if sample.free_vram_bytes is not None
    )
    if allocated is None or reserved is None or not free_values:
        raise ValueError("allocated, reserved, and free VRAM telemetry are required")
    steps_per_second = (
        steady_state_optimizer_steps / float(steady_state_seconds)
        if steady_state_seconds > 0
        else 0.0
    )
    return TelemetrySummary(
        initialization_seconds=float(initialization_seconds),
        warmup_seconds=float(warmup_seconds),
        steady_state_seconds=float(steady_state_seconds),
        peak_allocated_vram_bytes=int(allocated),
        peak_reserved_vram_bytes=int(reserved),
        minimum_free_vram_bytes=min(free_values),
        peak_gpu_utilization_percent=_optional_float(_peak(samples, "gpu_utilization_percent")),
        peak_power_watts=_optional_float(_peak(samples, "power_watts")),
        peak_temperature_celsius=_optional_float(_peak(samples, "temperature_celsius")),
        peak_host_memory_bytes=_optional_int(_peak(samples, "host_memory_bytes")),
        steady_steps_per_second=steps_per_second,
        samples_per_second=steps_per_second * physical_batch_size,
    )


def _optional_float(value: int | float | None) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: int | float | None) -> int | None:
    return None if value is None else int(value)


def _host_memory_bytes() -> int | None:
    """Read current RSS without adding a required third-party dependency."""

    status = Path("/proc/self/status")
    if status.is_file():
        try:
            for line in status.read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            return None
    try:
        import resource

        maximum_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, OSError, ValueError):
        return None
    return int(maximum_rss if sys.platform == "darwin" else maximum_rss * 1024)


class NvmlTelemetrySampler:
    """NVML/PyTorch sampler whose GPU dependencies load only on construction."""

    def __init__(self, *, device_index: int = 0, clock: Callable[[], float] = monotonic) -> None:
        if type(device_index) is not int or device_index < 0:
            raise ValueError("NVML device index must be nonnegative")
        try:
            import pynvml  # type: ignore[import-not-found]
            import torch  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError("NVML telemetry requires pynvml and torch") from error
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except BaseException:
            pynvml.nvmlShutdown()
            raise
        self._pynvml = pynvml
        self._torch = torch
        self._handle = handle
        self._device_index = device_index
        self._clock = clock
        self._started_at = clock()
        self._closed = False

    def _optional_nvml(self, operation: Callable[[], object]) -> object | None:
        try:
            return operation()
        except self._pynvml.NVMLError:
            return None

    def sample(self) -> TelemetrySample:
        if self._closed:
            raise RuntimeError("NVML telemetry sampler is closed")
        memory = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        utilization = self._optional_nvml(
            lambda: self._pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu
        )
        power_milliwatts = self._optional_nvml(
            lambda: self._pynvml.nvmlDeviceGetPowerUsage(self._handle)
        )
        temperature = self._optional_nvml(
            lambda: self._pynvml.nvmlDeviceGetTemperature(
                self._handle,
                self._pynvml.NVML_TEMPERATURE_GPU,
            )
        )
        return TelemetrySample(
            timestamp_seconds=self._clock() - self._started_at,
            allocated_vram_bytes=int(self._torch.cuda.memory_allocated(self._device_index)),
            reserved_vram_bytes=int(self._torch.cuda.memory_reserved(self._device_index)),
            free_vram_bytes=int(memory.free),
            gpu_utilization_percent=_optional_float(utilization),
            power_watts=(
                None if power_milliwatts is None else float(power_milliwatts) / 1000.0
            ),
            temperature_celsius=_optional_float(temperature),
            host_memory_bytes=_host_memory_bytes(),
        )

    def close(self) -> None:
        if not self._closed:
            self._pynvml.nvmlShutdown()
            self._closed = True

    def __enter__(self) -> "NvmlTelemetrySampler":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


_Result = TypeVar("_Result")


def sample_operation(
    operation: Callable[[], _Result],
    *,
    sampler: TelemetrySampler,
    sample_interval_seconds: float = 0.1,
) -> tuple[_Result, tuple[TelemetrySample, ...]]:
    """Sample throughout one synchronous operation without parallel launches.

    Only telemetry polling runs in the helper thread; the expensive official
    training launches remain strictly sequential in the calling thread.
    """

    if (
        type(sample_interval_seconds) not in (int, float)
        or not math.isfinite(float(sample_interval_seconds))
        or sample_interval_seconds <= 0
    ):
        raise ValueError("telemetry sample interval must be finite and positive")
    samples = [sampler.sample()]
    sampling_errors: list[BaseException] = []
    stopped = Event()

    def poll() -> None:
        while not stopped.wait(float(sample_interval_seconds)):
            try:
                samples.append(sampler.sample())
            except BaseException as error:
                sampling_errors.append(error)
                stopped.set()

    polling = Thread(target=poll, name="lehome-smoke-telemetry", daemon=True)
    polling.start()
    try:
        result = operation()
    finally:
        stopped.set()
        polling.join()
    if sampling_errors:
        raise RuntimeError("telemetry sampling failed during smoke launch") from sampling_errors[0]
    samples.append(sampler.sample())
    return result, tuple(samples)
