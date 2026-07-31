"""CPU-safe fail-closed checks that run before a paid GR00T job starts."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from time import monotonic
from typing import Callable, Iterable


GIBIBYTE = 1024**3
MINIMUM_VRAM_BYTES = 40 * GIBIBYTE
MINIMUM_DISK_BYTES = 200 * GIBIBYTE
PREFLIGHT_STAGE_NAMES = (
    "image_runtime_verification",
    "network_measurement",
    "model_download",
    "dataset_download",
    "schema_hash_validation",
    "model_initialization",
)
_VISIBLE_GPU = re.compile(r"(?:[0-9]+|GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9-]+)")
_REVISION = re.compile(r"[0-9a-f]{40}")
_SECRET_CONFIG_KEYS = frozenset({"token", "password", "secret", "api_key", "access_key"})


@dataclass(frozen=True, slots=True)
class HardwareReport:
    """The minimum hardware facts that must be true before training."""

    visible_device: str
    vram_bytes: int
    writable_free_bytes: int


@dataclass(frozen=True, slots=True)
class PreflightStage:
    """One named, independently timed preflight operation."""

    name: str
    operation: Callable[[], None]


@dataclass(frozen=True, slots=True)
class PreflightStageResult:
    """A successful stage timing suitable for machine-readable status."""

    name: str
    duration_seconds: float


def _one_visible_gpu(value: str | None) -> str:
    if not isinstance(value, str):
        raise ValueError("exactly one visible GPU is required")
    candidate = value.strip()
    if not _VISIBLE_GPU.fullmatch(candidate):
        raise ValueError("exactly one visible GPU is required")
    return candidate


def check_hardware(
    *,
    visible_devices: str | None,
    visible_vram_bytes: Iterable[int],
    writable_free_bytes: int,
) -> HardwareReport:
    """Reject insufficient, multi-GPU, or non-writable paid environments.

    The caller obtains GPU facts from NVML at runtime; keeping this function
    dependency-free lets its safety rules run in CPU-only tests as well.
    """

    device = _one_visible_gpu(visible_devices)
    memories = tuple(visible_vram_bytes)
    if len(memories) != 1 or type(memories[0]) is not int or memories[0] < 0:
        raise ValueError("exactly one visible GPU is required")
    if memories[0] < MINIMUM_VRAM_BYTES:
        raise ValueError("at least 40 GiB VRAM is required")
    if type(writable_free_bytes) is not int or writable_free_bytes < MINIMUM_DISK_BYTES:
        raise ValueError("at least 200 GiB writable disk is required")
    return HardwareReport(
        visible_device=device,
        vram_bytes=memories[0],
        writable_free_bytes=writable_free_bytes,
    )


def verify_immutable_revision(
    *,
    expected_revision: str,
    observed_revision: str,
    label: str,
) -> None:
    """Bind a local snapshot to one full immutable commit revision."""

    if not isinstance(label, str) or not label.strip():
        raise ValueError("revision label must be non-empty")
    if not isinstance(expected_revision, str) or not _REVISION.fullmatch(expected_revision):
        raise ValueError(f"{label} revision must be a pinned 40-character revision")
    if not isinstance(observed_revision, str) or observed_revision != expected_revision:
        raise ValueError(f"{label} revision does not match its immutable expected revision")


def verify_hub_write_permission(
    *,
    token: str | None,
    permission_check: Callable[[str], bool],
) -> None:
    """Prove explicit credentials can upload without persisting or logging them."""

    if not isinstance(token, str) or not token.strip() or any(character.isspace() for character in token):
        raise ValueError("an explicit non-empty Hub token is required for upload permission")
    try:
        allowed = permission_check(token)
    except Exception as error:
        raise ValueError("Hub write permission check failed") from error
    if allowed is not True:
        raise ValueError("Hub write permission is required before paid training")


def reject_secret_bearing_config(value: object) -> None:
    """Reject resolved configuration fields that could persist credentials.

    Credentials are an execution-only input to a permission callback. They are
    never a provenance field, even when a caller accidentally tries to include
    one in a nested configuration object.
    """

    if isinstance(value, dict):
        for key, nested in value.items():
            if not isinstance(key, str):
                continue
            normalized = key.casefold()
            if normalized in _SECRET_CONFIG_KEYS or normalized.endswith("_token"):
                raise ValueError("resolved configuration must not contain a secret field")
            reject_secret_bearing_config(nested)
    elif isinstance(value, (tuple, list)):
        for nested in value:
            reject_secret_bearing_config(nested)


def run_timed_stages(stages: Iterable[PreflightStage]) -> tuple[PreflightStageResult, ...]:
    """Run all required preflight stages in their immutable order."""

    resolved = tuple(stages)
    if tuple(stage.name for stage in resolved) != PREFLIGHT_STAGE_NAMES:
        raise ValueError("preflight stages must use the complete canonical order")
    results: list[PreflightStageResult] = []
    for stage in resolved:
        started = monotonic()
        stage.operation()
        duration = monotonic() - started
        if not math.isfinite(duration) or duration < 0:
            raise RuntimeError("preflight stage duration is invalid")
        results.append(PreflightStageResult(stage.name, duration))
    return tuple(results)
