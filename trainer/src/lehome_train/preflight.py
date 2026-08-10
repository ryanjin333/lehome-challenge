"""CPU-safe fail-closed checks that run before a paid GR00T job starts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
from time import monotonic
from typing import Callable, Iterable, Mapping

from lehome_train.io import sha256_file
from lehome_train.models import ArtifactIdentity, validate_artifact_relative_path
from lehome_train.redaction import ACCESS_TOKEN_PATTERNS


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
    visible_devices: tuple[str, ...] = ()
    per_device_vram_bytes: tuple[int, ...] = ()
    per_device_free_vram_bytes: tuple[int, ...] = ()


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


@dataclass(frozen=True, slots=True)
class SnapshotArtifactIdentity:
    """A hashed snapshot file, including the Hub's root `.gitattributes`."""

    relative_path: str
    sha256: str
    byte_size: int

    def __post_init__(self) -> None:
        if type(self.relative_path) is not str:
            raise ValueError("snapshot artifact path is invalid")
        if self.relative_path != ".gitattributes":
            validate_artifact_relative_path(self.relative_path)
        if type(self.sha256) is not str or not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("snapshot artifact SHA-256 is invalid")
        if type(self.byte_size) is not int or self.byte_size < 0:
            raise ValueError("snapshot artifact byte size is invalid")


@dataclass(frozen=True, slots=True)
class SnapshotVerification:
    """Revision plus verified local files from one immutable snapshot manifest."""

    revision: str
    manifest_sha256: str
    artifacts: tuple[SnapshotArtifactIdentity, ...]


@dataclass(frozen=True, slots=True)
class HubTarget:
    """One immutable private repository/revision required by this run."""

    repository: str
    revision: str

    def __post_init__(self) -> None:
        if not isinstance(self.repository, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", self.repository):
            raise ValueError("Hub target repository must be an explicit owner/name")
        if not isinstance(self.revision, str) or not _REVISION.fullmatch(self.revision):
            raise ValueError("Hub target revision must be a pinned 40-character revision")


@dataclass(frozen=True, slots=True)
class HubPermission:
    """The permission contract required before paid checkpoints are created."""

    can_upload: bool
    can_readback: bool
    private_repository: bool


def _visible_gpus(value: str | None, *, expected_count: int) -> tuple[str, ...]:
    if not isinstance(value, str):
        raise ValueError("exactly one visible GPU is required")
    candidates = tuple(candidate.strip() for candidate in value.split(","))
    if (
        len(candidates) != expected_count
        or any(not _VISIBLE_GPU.fullmatch(candidate) for candidate in candidates)
        or len(set(candidates)) != len(candidates)
    ):
        if expected_count == 4:
            raise ValueError("exactly four visible GPUs are required")
        raise ValueError("exactly one visible GPU is required")
    return candidates


def check_hardware(
    *,
    visible_devices: str | None,
    visible_vram_bytes: Iterable[int],
    visible_free_vram_bytes: Iterable[int] | None = None,
    writable_free_bytes: int,
    expected_gpu_count: int = 1,
    minimum_vram_bytes: int = MINIMUM_VRAM_BYTES,
) -> HardwareReport:
    """Reject insufficient or mismatched paid environments per visible GPU.

    The caller obtains GPU facts from NVML at runtime; keeping this function
    dependency-free lets its safety rules run in CPU-only tests as well.
    """

    if expected_gpu_count not in {1, 4}:
        raise ValueError("hardware gate requires exactly one or four GPUs")
    if type(minimum_vram_bytes) is not int or minimum_vram_bytes <= 0:
        raise ValueError("per-device VRAM minimum must be positive")
    devices = _visible_gpus(visible_devices, expected_count=expected_gpu_count)
    memories = tuple(visible_vram_bytes)
    if len(memories) != expected_gpu_count or any(
        type(memory) is not int or memory < 0 for memory in memories
    ):
        raise ValueError("visible GPU memory probe does not match visible devices")
    if any(memory < minimum_vram_bytes for memory in memories):
        if expected_gpu_count == 1 and minimum_vram_bytes == MINIMUM_VRAM_BYTES:
            raise ValueError("at least 40 GiB VRAM is required")
        raise ValueError("every visible GPU must satisfy the per-device VRAM minimum")
    free_memories = (
        memories if visible_free_vram_bytes is None else tuple(visible_free_vram_bytes)
    )
    if len(free_memories) != expected_gpu_count or any(
        type(memory) is not int or memory < 0 for memory in free_memories
    ):
        raise ValueError("visible GPU free-memory probe does not match visible devices")
    if any(free > total for free, total in zip(free_memories, memories)):
        raise ValueError("visible GPU free memory exceeds its per-device total")
    if any(free * 100 < total * 10 for free, total in zip(free_memories, memories)):
        raise ValueError("every visible GPU must retain at least 10 percent headroom")
    if type(writable_free_bytes) is not int or writable_free_bytes < MINIMUM_DISK_BYTES:
        raise ValueError("at least 200 GiB writable disk is required")
    return HardwareReport(
        visible_device=devices[0] if len(devices) == 1 else ",".join(devices),
        vram_bytes=memories[0],
        writable_free_bytes=writable_free_bytes,
        visible_devices=devices,
        per_device_vram_bytes=memories,
        per_device_free_vram_bytes=free_memories,
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


def _safe_snapshot_artifact(
    snapshot_root: Path, artifact: SnapshotArtifactIdentity
) -> None:
    relative = artifact.relative_path
    candidate = snapshot_root / relative
    try:
        candidate.resolve().relative_to(snapshot_root.resolve())
    except ValueError as error:
        raise ValueError("snapshot artifact path escapes its root") from error
    if not candidate.is_file() or candidate.is_symlink():
        raise ValueError("snapshot artifact is not a regular file")
    if candidate.stat().st_size != artifact.byte_size or sha256_file(candidate) != artifact.sha256:
        raise ValueError("snapshot artifact hash or size mismatch")


def _snapshot_files(snapshot_root: Path) -> set[str]:
    observed: set[str] = set()
    pending = [snapshot_root]
    while pending:
        directory = pending.pop()
        for entry in os.scandir(directory):
            path = Path(entry.path)
            if entry.is_symlink():
                raise ValueError("snapshot contains a symlink")
            if entry.is_dir(follow_symlinks=False):
                pending.append(path)
            elif entry.is_file(follow_symlinks=False):
                observed.add(path.relative_to(snapshot_root).as_posix())
            else:
                raise ValueError("snapshot contains a special path")
    return observed


def verify_snapshot_manifest(
    *,
    snapshot_root: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    expected_revision: str,
    label: str,
) -> SnapshotVerification:
    """Resolve a revision only from a local manifest with verified files."""

    root = Path(snapshot_root)
    manifest = Path(manifest_path)
    if not root.is_dir() or root.is_symlink() or not manifest.is_file() or manifest.is_symlink():
        raise ValueError(f"{label} snapshot manifest is unavailable")
    try:
        manifest.resolve().relative_to(root.resolve())
        decoded = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} snapshot manifest is malformed") from None
    if not isinstance(decoded, Mapping) or set(decoded) != {"revision", "artifacts"}:
        raise ValueError(f"{label} snapshot manifest is malformed")
    observed = decoded["revision"]
    if not isinstance(observed, str):
        raise ValueError(f"{label} snapshot manifest is malformed")
    verify_immutable_revision(
        expected_revision=expected_revision,
        observed_revision=observed,
        label=label,
    )
    encoded_artifacts = decoded["artifacts"]
    if not isinstance(encoded_artifacts, list) or not encoded_artifacts:
        raise ValueError(f"{label} snapshot manifest has no artifacts")
    try:
        artifacts = tuple(
            SnapshotArtifactIdentity(
                relative_path=item["relative_path"],
                sha256=item["sha256"],
                byte_size=item["byte_size"],
            )
            for item in encoded_artifacts
            if isinstance(item, Mapping)
            and set(item) == {"relative_path", "sha256", "byte_size"}
        )
    except (TypeError, ValueError):
        raise ValueError(f"{label} snapshot manifest artifacts are invalid") from None
    if len(artifacts) != len(encoded_artifacts) or len({item.relative_path for item in artifacts}) != len(artifacts):
        raise ValueError(f"{label} snapshot manifest artifacts are invalid")
    for artifact in artifacts:
        _safe_snapshot_artifact(root, artifact)
    expected_files = {artifact.relative_path for artifact in artifacts}
    expected_files.add(manifest.relative_to(root).as_posix())
    if _snapshot_files(root) != expected_files:
        raise ValueError(f"{label} snapshot contains unverified files")
    return SnapshotVerification(
        revision=observed,
        manifest_sha256=sha256_file(manifest),
        artifacts=artifacts,
    )


def verify_hub_upload_readback_permission(
    *,
    token: str | None,
    target: HubTarget,
    permission_check: Callable[[str, str, str], HubPermission],
) -> None:
    """Prove scoped private upload and readback access without recording a token."""

    if not isinstance(token, str) or not token.strip() or any(character.isspace() for character in token):
        raise ValueError("an explicit non-empty Hub token is required for upload permission")
    try:
        permission = permission_check(token, target.repository, target.revision)
    except Exception:
        raise ValueError("Hub upload/readback permission check failed") from None
    if not isinstance(permission, HubPermission):
        raise ValueError("Hub upload/readback permission check returned an invalid contract")
    if not (permission.can_upload and permission.can_readback and permission.private_repository):
        raise ValueError("private Hub upload and readback permission is required before paid training")


def reject_secret_bearing_config(value: object) -> None:
    """Reject resolved configuration fields that could persist credentials.

    Credentials are an execution-only input to a permission callback. They are
    never a provenance field, even when a caller accidentally tries to include
    one in a nested configuration object.
    """

    if isinstance(value, str):
        if any(pattern.search(value) for pattern in ACCESS_TOKEN_PATTERNS):
            raise ValueError("resolved configuration must not contain a secret value")
    elif isinstance(value, dict):
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
