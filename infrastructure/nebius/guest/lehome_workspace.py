"""Shared-disk mount and role-lease contract for preemptible Nebius guests.

Both golden images install this module as an early oneshot service.  It owns
exactly three responsibilities:

1. admit the shared 500 GiB network SSD only through its verified filesystem
   UUID (a disk that already carries a filesystem is never reformatted);
2. mount it at ``/mnt/lehome``; and
3. maintain the atomic ``workspace-manifest.json`` role lease so only one
   active role (training or rollout) ever owns the disk at a time.

All subprocesses are injected so the contract is testable without root.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Callable, Mapping, Sequence
from uuid import uuid4


WORKSPACE_MOUNT = "/mnt/lehome"
MANIFEST_NAME = "workspace-manifest.json"
MANIFEST_SCHEMA_VERSION = 1
ROLES = ("training", "rollout")
_UUID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class WorkspaceError(RuntimeError):
    """Fail-closed admission: the VM must not boot onto a wrong disk state."""


@dataclass(frozen=True, slots=True)
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str


CommandRunner = Callable[[Sequence[str]], CommandResult]


@dataclass(frozen=True, slots=True)
class WorkspaceConfig:
    """Immutable boot-time inputs for one role on the shared disk."""

    device: str
    role: str
    run_id: str
    expected_uuid: str | None = None
    mount_point: str = WORKSPACE_MOUNT

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or not self.device.startswith("/dev/"):
            raise WorkspaceError("device must be an absolute /dev/ block path")
        if self.role not in ROLES:
            raise WorkspaceError(f"role must be one of {ROLES}, got {self.role!r}")
        if not isinstance(self.run_id, str) or not self.run_id:
            raise WorkspaceError("run_id must be a non-empty string")
        if self.expected_uuid is not None and not _UUID_PATTERN.match(self.expected_uuid.lower()):
            raise WorkspaceError("expected_uuid must be a canonical lowercase filesystem UUID")
        if not self.mount_point.startswith("/"):
            raise WorkspaceError("mount_point must be absolute")


def _directory_fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _run(runner: CommandRunner, command: Sequence[str]) -> CommandResult:
    result = runner(command)
    if not isinstance(result, CommandResult):
        raise WorkspaceError("command runner must return CommandResult")
    return result


def blkid_uuid(device: str, runner: CommandRunner) -> str | None:
    """Return the filesystem UUID, or None when the disk is blank.

    ``blkid`` exits 2 when no filesystem signature exists; any other non-zero
    exit is an environment fault and fails closed.
    """
    result = _run(runner, ["blkid", "-s", "UUID", "-o", "value", device])
    if result.exit_code == 0:
        uuid = result.stdout.strip().lower()
        if not _UUID_PATTERN.match(uuid):
            raise WorkspaceError(f"blkid returned an unparsable UUID: {uuid!r}")
        return uuid
    if result.exit_code == 2:
        return None
    raise WorkspaceError(f"blkid failed for {device}: exit {result.exit_code}: {result.stderr.strip()}")


def format_blank_disk(device: str, uuid: str, runner: CommandRunner) -> None:
    """Format a verified-blank disk with a pinned UUID; never touch data."""
    if not _UUID_PATTERN.match(uuid.lower()):
        raise WorkspaceError("format UUID must be canonical lowercase")
    if blkid_uuid(device, runner) is not None:
        raise WorkspaceError("refusing to format a disk that already carries a filesystem")
    result = _run(runner, ["mkfs.ext4", "-q", "-U", uuid.lower(), device])
    if result.exit_code != 0:
        raise WorkspaceError(f"mkfs.ext4 failed: exit {result.exit_code}: {result.stderr.strip()}")


def read_mount_table(text: str) -> dict[str, str]:
    """Parse /proc/mounts-shaped text into {device: mount_point}."""
    table: dict[str, str] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 2 or fields[0] in {"proc", "sysfs", "devtmpfs", "tmpfs", "none"}:
            continue
        table[fields[0]] = fields[1]
    return table


def mount_workspace(config: WorkspaceConfig, runner: CommandRunner, mount_table_text: str) -> None:
    """Mount the shared disk once; refuse conflicting existing mounts."""
    table = read_mount_table(mount_table_text)
    existing = table.get(config.device)
    if existing is not None:
        if existing == config.mount_point:
            return
        raise WorkspaceError(f"device {config.device} is already mounted at {existing}")
    for other_device, mount_point in table.items():
        if mount_point == config.mount_point:
            raise WorkspaceError(f"mount point {config.mount_point} is occupied by {other_device}")
    result = _run(runner, ["mkdir", "-p", config.mount_point])
    if result.exit_code != 0:
        raise WorkspaceError(f"mkdir failed for {config.mount_point}: {result.stderr.strip()}")
    result = _run(runner, ["mount", "-o", "noatime", config.device, config.mount_point])
    if result.exit_code != 0:
        raise WorkspaceError(f"mount failed: exit {result.exit_code}: {result.stderr.strip()}")


def manifest_path(config: WorkspaceConfig) -> Path:
    return Path(config.mount_point) / MANIFEST_NAME


def write_manifest_atomic(path: Path, manifest: Mapping[str, object]) -> None:
    """Canonical, durable manifest write that never exposes a partial file."""
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        _directory_fsync(path.parent)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def load_manifest(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise WorkspaceError(f"workspace manifest missing or unsafe at {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise WorkspaceError(f"workspace manifest unreadable: {error}") from error
    if not isinstance(manifest, dict):
        raise WorkspaceError("workspace manifest must be a JSON object")
    return manifest


def _validate_manifest(manifest: Mapping[str, object], observed_uuid: str) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise WorkspaceError(f"unsupported manifest schema_version: {manifest.get('schema_version')!r}")
    if manifest.get("disk_uuid") != observed_uuid:
        raise WorkspaceError("manifest disk_uuid does not match the mounted filesystem")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise WorkspaceError("manifest run_id must be a non-empty string")
    active_role = manifest.get("active_role")
    if active_role not in (None, *ROLES):
        raise WorkspaceError(f"manifest active_role is corrupt: {active_role!r}")


def initialize_manifest(config: WorkspaceConfig, observed_uuid: str, clock_ns: Callable[[], int]) -> dict[str, object]:
    """Create the manifest on a genuinely fresh workspace and take the lease."""
    path = manifest_path(config)
    if path.exists() or path.is_symlink():
        raise WorkspaceError("refusing to reinitialize an existing workspace manifest")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "disk_uuid": observed_uuid,
        "created_at_ns": clock_ns(),
        "run_id": config.run_id,
        "active_role": config.role,
        "role_lease_at_ns": clock_ns(),
        "last_clean_handoff_ns": None,
        "durable_manifest_hashes": {},
    }
    write_manifest_atomic(path, manifest)
    return manifest


def acquire_role(config: WorkspaceConfig, observed_uuid: str, clock_ns: Callable[[], int]) -> dict[str, object]:
    """Admit this boot: verify identity, then take or confirm the role lease.

    A replacement VM for the same role may re-acquire its own lease.  A boot
    for a different role fails closed until ``release_role`` records a clean
    handoff on the outgoing VM.
    """
    path = manifest_path(config)
    manifest = load_manifest(path)
    _validate_manifest(manifest, observed_uuid)
    if manifest.get("run_id") != config.run_id:
        raise WorkspaceError(
            f"run identity mismatch: manifest {manifest.get('run_id')!r} versus boot {config.run_id!r}"
        )
    active_role = manifest.get("active_role")
    if active_role is not None and active_role != config.role:
        raise WorkspaceError(
            f"role lease conflict: disk is leased to {active_role!r}; refusing boot for {config.role!r}"
        )
    updated = dict(manifest)
    updated["active_role"] = config.role
    updated["role_lease_at_ns"] = clock_ns()
    write_manifest_atomic(path, updated)
    return updated


def release_role(config: WorkspaceConfig, clock_ns: Callable[[], int]) -> dict[str, object]:
    """Record a clean handoff so the other role may acquire the disk."""
    path = manifest_path(config)
    manifest = load_manifest(path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise WorkspaceError("cannot release a role on an unsupported manifest schema")
    active_role = manifest.get("active_role")
    if active_role != config.role:
        raise WorkspaceError(f"cannot release role {config.role!r}: disk leased to {active_role!r}")
    updated = dict(manifest)
    updated["active_role"] = None
    updated["last_clean_handoff_ns"] = clock_ns()
    write_manifest_atomic(path, updated)
    return updated


def ensure_layout(config: WorkspaceConfig) -> tuple[Path, ...]:
    """Create the durable shared-disk directory layout if absent."""
    root = Path(config.mount_point)
    if root.is_symlink() or not root.is_dir():
        raise WorkspaceError(f"workspace mount point missing: {root}")
    directories = (
        "cache/huggingface", "cache/containers", "cache/video",
        "datasets/bundles", "datasets/bc/full", "datasets/rollouts/round-1", "datasets/manifests",
        "checkpoints/local", "checkpoints/published-receipts",
        "rollouts/attempts", "rollouts/accepted", "rollouts/upload-queue",
        "ledgers", "logs", "receipts",
    )
    created: list[Path] = []
    for relative in directories:
        target = root / relative
        target.mkdir(parents=True, exist_ok=True)
        created.append(target)
    return tuple(created)
