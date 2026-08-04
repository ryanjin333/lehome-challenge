"""Resumable, local-only supervisor for isolated GR00T rollout processes."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import time
from typing import Callable, Sequence
from uuid import uuid4

from lehome.flywheel.artifacts import verify_episode_manifest
from lehome.flywheel.capacity import CapacityDecision, CapacitySample, choose_worker_count
from lehome.flywheel.isaac_recorder import CANONICAL_VIDEO_FILENAMES
from lehome.flywheel.matrix import Trial, load_public_matrix, matrix_sha256
from lehome.flywheel.runtime_preflight import require_isaac_sim_5_1_runtime


@dataclass(frozen=True, slots=True)
class CampaignState:
    output_root: Path
    trial_ids: tuple[str, ...]


def _validate_trial_id(trial_id: str) -> None:
    if (
        not isinstance(trial_id, str)
        or trial_id in {"", ".", ".."}
        or "/" in trial_id
        or "\\" in trial_id
        or Path(trial_id).is_absolute()
        or Path(trial_id).name != trial_id
    ):
        raise ValueError("trial ID must be a non-empty path-safe identifier")


def _open_campaign_directory(parent_fd: int, name: str, *, create: bool) -> int | None:
    """Open one trusted campaign child directory without following a symlink."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        details = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        if not create:
            return None
        os.mkdir(name, dir_fd=parent_fd)
        details = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise ValueError(f"campaign {name} root is unsafe")
    try:
        return os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        raise ValueError(f"campaign {name} root is unsafe") from error


def _open_controller_lock(root_fd: int) -> int:
    """Create the fixed lock once, then open it no-follow for every controller."""
    flags = os.O_RDWR | os.O_NOFOLLOW
    while True:
        try:
            return os.open(".campaign.lock", flags | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=root_fd)
        except FileExistsError:
            try:
                return os.open(".campaign.lock", flags, dir_fd=root_fd)
            except FileNotFoundError:
                # Another controller won creation but has not made the entry
                # observable yet; retry rather than accepting an unchecked path.
                continue


@contextmanager
def _locked_campaign_storage(output_root: Path):
    """Serialize cooperating controllers and expose a no-follow output-root FD."""
    if not hasattr(os, "O_NOFOLLOW") or os.name != "posix":
        raise ValueError("campaign retry storage requires POSIX no-follow support")
    root = Path(output_root)
    if root.is_symlink():
        raise ValueError("campaign output root must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("campaign output root must be a directory")
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    lock_fd = -1
    try:
        lock_fd = _open_controller_lock(root_fd)
        if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
            raise ValueError("campaign controller lock is unsafe")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        yield root, root_fd
    except OSError as error:
        raise ValueError("campaign output storage is unsafe") from error
    finally:
        if lock_fd >= 0:
            os.close(lock_fd)
        os.close(root_fd)


def _open_trial_directory(parent_fd: int, trial_id: str) -> os.stat_result | None:
    try:
        details = os.stat(trial_id, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise ValueError("campaign trial path is unsafe")
    return details


def _is_completed_locked(root: Path, root_fd: int, trial_id: str) -> bool:
    for name in (".pending", "quarantine"):
        parent_fd = _open_campaign_directory(root_fd, name, create=False)
        if parent_fd is not None:
            os.close(parent_fd)
    raw_fd = _open_campaign_directory(root_fd, "raw", create=False)
    if raw_fd is None:
        return False
    try:
        before = _open_trial_directory(raw_fd, trial_id)
        if before is None:
            return False
        episode_dir = root / "raw" / trial_id
        try:
            episode, manifest = verify_episode_manifest(episode_dir)
        except ValueError:
            return False
        after = _open_trial_directory(raw_fd, trial_id)
        if after is None or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise ValueError("campaign raw trial changed during verification")
    finally:
        os.close(raw_fd)
    if not isinstance(episode.get("terminal_reason"), str) or not episode["terminal_reason"]:
        return False
    if episode.get("outcome") == "error" or episode.get("recorder_error"):
        return False
    expected_videos = {f"videos/{filename}" for filename in CANONICAL_VIDEO_FILENAMES}
    manifest_videos = {path for path in manifest if path.startswith("videos/")}
    return manifest_videos == expected_videos and all(manifest[path]["size"] > 0 for path in expected_videos)


def is_completed_trial(output_root: Path, trial_id: str) -> bool:
    """Accept only terminal, non-error artifacts with canonical video evidence."""
    _validate_trial_id(trial_id)
    with _locked_campaign_storage(output_root) as (root, root_fd):
        return _is_completed_locked(root, root_fd, trial_id)


def _prepare_retry_attempt(output_root: Path, trial_id: str) -> None:
    """Atomically quarantine an invalid prior attempt before retrying its ID."""
    _validate_trial_id(trial_id)
    with _locked_campaign_storage(output_root) as (root, root_fd):
        if _is_completed_locked(root, root_fd, trial_id):
            return
        parent_fds: list[tuple[str, int, os.stat_result]] = []
        try:
            for parent_name in (".pending", "raw"):
                parent_fd = _open_campaign_directory(root_fd, parent_name, create=True)
                try:
                    details = _open_trial_directory(parent_fd, trial_id)
                except BaseException:
                    os.close(parent_fd)
                    raise
                if details is not None:
                    parent_fds.append((parent_name.removeprefix("."), parent_fd, details))
                else:
                    os.close(parent_fd)
            quarantine_fd = _open_campaign_directory(root_fd, "quarantine", create=True)
            if not parent_fds:
                os.close(quarantine_fd)
                return
            try:
                attempt = 1
                while True:
                    attempt_name = f"{trial_id}.attempt-{attempt:03d}"
                    try:
                        os.mkdir(attempt_name, dir_fd=quarantine_fd)
                    except FileExistsError:
                        existing = os.stat(attempt_name, dir_fd=quarantine_fd, follow_symlinks=False)
                        if stat.S_ISLNK(existing.st_mode) or not stat.S_ISDIR(existing.st_mode):
                            raise ValueError("campaign quarantine attempt path is unsafe")
                        attempt += 1
                        continue
                    attempt_fd = _open_campaign_directory(quarantine_fd, attempt_name, create=False)
                    break
                try:
                    for name, parent_fd, before in parent_fds:
                        current = _open_trial_directory(parent_fd, trial_id)
                        if current is None or (before.st_dev, before.st_ino) != (current.st_dev, current.st_ino):
                            raise ValueError("campaign trial changed during retry preparation")
                        try:
                            os.stat(name, dir_fd=attempt_fd, follow_symlinks=False)
                        except FileNotFoundError:
                            pass
                        else:
                            raise ValueError("campaign quarantine destination collision")
                        os.rename(trial_id, name, src_dir_fd=parent_fd, dst_dir_fd=attempt_fd)
                        moved = os.stat(name, dir_fd=attempt_fd, follow_symlinks=False)
                        if stat.S_ISLNK(moved.st_mode) or (moved.st_dev, moved.st_ino) != (before.st_dev, before.st_ino):
                            raise ValueError("campaign trial changed during quarantine")
                finally:
                    os.close(attempt_fd)
            finally:
                os.close(quarantine_fd)
        finally:
            for _, parent_fd, _ in parent_fds:
                os.close(parent_fd)


def pending_trial_ids(state: CampaignState) -> tuple[str, ...]:
    """Resume unless the terminal artifact passes the canonical completion predicate."""
    return tuple(
        trial_id
        for trial_id in state.trial_ids
        if not is_completed_trial(state.output_root, trial_id)
    )


def _write_heartbeat(path: Path, *, worker_id: int, trial_id: str, state: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"worker_id": worker_id, "trial_id": trial_id, "state": state, "monotonic_ns": time.monotonic_ns()}, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _trial_command(
    args: argparse.Namespace,
    trial: Trial,
    *,
    device: str | None = None,
    policy_server_port: int | None = None,
    policy_server_log: Path | None = None,
) -> list[str]:
    command = [
        sys.executable, "-m", "scripts.run_groot_flywheel_trial", "--policy-path", str(args.policy_path),
        "--policy-revision-file", str(args.policy_revision_file), "--garment", trial.garment_name,
        "--policy-repo", args.policy_repo, "--policy-step", str(args.policy_step), "--code-revision", args.code_revision,
        "--asset-revision", args.asset_revision, "--simulator-version", args.simulator_version,
        "--release-assets-root", str(args.release_assets_root),
        "--category", trial.category, "--release-stage", trial.release_stage,
        "--policy-artifact-sha256", args.policy_artifact_sha256, "--image-identity", args.image_identity,
        "--seed", str(trial.seed), "--episode-id", trial.trial_id, "--output-root", str(args.output_root),
        "--strategy", args.strategy,
        "--max-steps", str(args.max_steps), "--device", device or getattr(args, "device", "cuda"),
    ]
    if (policy_server_port is None) != (policy_server_log is None):
        raise ValueError("policy server port and log must be assigned together")
    if policy_server_port is not None:
        command.extend((
            "--groot-root", str(args.groot_root), "--groot-revision", args.groot_revision,
            "--groot-python", str(args.groot_python), "--policy-server-port", str(policy_server_port),
            "--policy-server-readiness-timeout", str(args.policy_server_readiness_timeout),
            "--policy-server-request-timeout", str(args.policy_server_request_timeout),
            "--policy-server-termination-grace", str(args.policy_server_termination_grace),
            "--policy-server-log", str(policy_server_log),
        ))
    command.append("--headless")
    return command


def _attempt_log_paths(worker_root: Path, trial_id: str) -> tuple[Path, Path]:
    attempt = 1
    while True:
        worker_log = worker_root / f"{trial_id}.attempt-{attempt:03d}.log"
        policy_server_log = worker_root / f"{trial_id}.attempt-{attempt:03d}.policy-server.log"
        if all(not path.exists() and not path.is_symlink() for path in (worker_log, policy_server_log)):
            return worker_log, policy_server_log
        attempt += 1


def _attempt_log_path(worker_root: Path, trial_id: str) -> Path:
    """Return the normal worker log while reserving its paired server-log suffix."""
    return _attempt_log_paths(worker_root, trial_id)[0]


def _allocate_loopback_port() -> int:
    """Reserve a candidate loopback port long enough to prevent in-wave collisions.

    The trial rechecks the candidate immediately before binding the policy server,
    because a released ephemeral port cannot be held across an exec boundary.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _allocate_loopback_ports(workers: int) -> tuple[int, ...]:
    if workers <= 0:
        raise ValueError("worker count must be positive")
    ports: set[int] = set()
    max_attempts = workers * 16
    while len(ports) != workers and max_attempts:
        ports.add(_allocate_loopback_port())
        max_attempts -= 1
    if len(ports) != workers:
        raise ValueError("could not allocate unique loopback policy-server ports")
    return tuple(sorted(ports))


def _run_one_worker(args: argparse.Namespace, *, worker_id: int, trial: Trial) -> int:
    worker_root = args.output_root / "workers" / f"worker-{worker_id:02d}"
    heartbeat = worker_root / "heartbeat.json"
    log_path, policy_server_log = _attempt_log_paths(worker_root, trial.trial_id)
    policy_server_port = _allocate_loopback_ports(1)[0]
    _prepare_retry_attempt(args.output_root, trial.trial_id)
    _write_heartbeat(heartbeat, worker_id=worker_id, trial_id=trial.trial_id, state="started")
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            _trial_command(args, trial, policy_server_port=policy_server_port, policy_server_log=policy_server_log),
            stdout=log, stderr=subprocess.STDOUT, env=_worker_environment(args, _cuda_device_index(args.device)),
        )
        try:
            returncode = process.wait(timeout=args.worker_timeout_seconds)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=args.terminate_grace_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            _write_heartbeat(heartbeat, worker_id=worker_id, trial_id=trial.trial_id, state="timeout")
            return 124
    _write_heartbeat(heartbeat, worker_id=worker_id, trial_id=trial.trial_id, state="terminal")
    return returncode


def _validate_sweep(values: str) -> tuple[int, ...]:
    if values != "1,2,4":
        raise ValueError("four-GPU capacity sweep must be exactly 1,2,4")
    return (1, 2, 4)


def _positive_finite_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return seconds


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _resource_margins(gpu_indices: Sequence[int] | None = None) -> tuple[float, float, float]:
    """Read host and assigned-GPU free margins; unknown telemetry fails closed."""
    host_margin = 0.0
    try:
        entries = dict(
            line.replace(":", "").split()[:2]
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
            if ":" in line
        )
        host_margin = int(entries["MemAvailable"]) / int(entries["MemTotal"])
    except (KeyError, OSError, ValueError):
        pass
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
            timeout=5.0,
        )
        observed = {
            int(index): (int(free), int(total))
            for line in completed.stdout.splitlines()
            for index, free, total in [line.split(",")]
        }
        assigned = tuple(gpu_indices) if gpu_indices is not None else tuple(observed)
        if len(set(assigned)) != len(assigned) or any(index not in observed for index in assigned):
            raise ValueError("assigned GPU telemetry is unavailable")
        margins = [observed[index][0] / observed[index][1] for index in assigned]
        vram_margin = min(margins) if completed.returncode == 0 and margins else 0.0
    except (OSError, ValueError, ZeroDivisionError, subprocess.TimeoutExpired):
        vram_margin = 0.0
    # Kept as a compatibility return slot for callers still unpacking three
    # values. Campaign decisions consume this once as combined Isaac+policy
    # usage, never as independent renderer and inference evidence.
    return host_margin, vram_margin, vram_margin


def _cuda_device_index(device: str) -> int | None:
    if device == "cpu":
        return None
    if device == "cuda":
        return 0
    if device.startswith("cuda:") and device.removeprefix("cuda:").isdigit():
        return int(device.removeprefix("cuda:"))
    raise ValueError("rollout device must be cpu, cuda, or cuda:<non-negative-index>")


def _visible_gpu_indices() -> tuple[int, ...]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            check=False, capture_output=True, text=True, timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError("CUDA rollout isolation requires nvidia-smi GPU inventory") from error
    if result.returncode != 0:
        raise ValueError("CUDA rollout isolation requires nvidia-smi GPU inventory")
    try:
        indices = tuple(int(line.strip()) for line in result.stdout.splitlines() if line.strip())
    except ValueError as error:
        raise ValueError("CUDA rollout isolation received an invalid GPU inventory") from error
    if not indices or len(set(indices)) != len(indices) or any(index < 0 for index in indices):
        raise ValueError("CUDA rollout isolation received an invalid GPU inventory")
    return indices


def _worker_gpu_indices(args: argparse.Namespace, workers: int) -> tuple[int | None, ...]:
    """Assign one unique physical GPU to every concurrent Isaac+policy process."""
    if workers <= 0:
        raise ValueError("worker count must be positive")
    requested = _cuda_device_index(getattr(args, "device", "cuda"))
    if requested is None:
        if workers != 1:
            raise ValueError("CPU rollout workers cannot be oversubscribed")
        return (None,)
    available = _visible_gpu_indices()
    if requested not in available:
        raise ValueError("requested rollout CUDA device is not visible")
    ordered = (requested, *(index for index in available if index != requested))
    if workers > len(ordered):
        raise ValueError("unsupported GPU oversubscription: each rollout worker requires one isolated GPU")
    return ordered[:workers]


def _worker_environment(
    args: argparse.Namespace,
    gpu_index: int | None,
    *,
    policy_telemetry_path: Path | None = None,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("LEHOME_FLYWHEEL_POLICY_TELEMETRY_PATH", None)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment.pop("LEHOME_FLYWHEEL_WORKER_GPU", None)
    if gpu_index is not None:
        # The trial receives a physical cuda:N and clears visibility itself
        # before Isaac launches. Only its isolated GR00T-server child narrows
        # CUDA_VISIBLE_DEVICES to this recorded physical GPU.
        environment["LEHOME_FLYWHEEL_WORKER_GPU"] = str(gpu_index)
    if policy_telemetry_path is not None:
        environment["LEHOME_FLYWHEEL_POLICY_TELEMETRY_PATH"] = str(policy_telemetry_path)
    return environment


def _worker_root_name(worker_id: int) -> str:
    if not isinstance(worker_id, int) or isinstance(worker_id, bool) or worker_id <= 0:
        raise ValueError("worker ID must be a positive integer")
    return f"worker-{worker_id:02d}"


@dataclass(frozen=True, slots=True)
class _ProvisionedPolicyTelemetry:
    path: Path
    device: int
    inode: int

    def __fspath__(self) -> str:
        return os.fspath(self.path)

    def __getattr__(self, name: str):
        return getattr(self.path, name)


def _prepare_policy_telemetry_path(output_root: Path, *, worker_id: int) -> _ProvisionedPolicyTelemetry:
    """Provision one exclusive append-only telemetry file under its worker root."""
    root = Path(output_root)
    if root.is_symlink():
        raise ValueError("campaign output root is unsafe")
    root.mkdir(parents=True, exist_ok=True)
    try:
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        raise ValueError("campaign output root is unsafe") from error
    workers_fd = worker_fd = telemetry_fd = -1
    filename = f"policy-telemetry-{uuid4().hex}.jsonl"
    worker_name = _worker_root_name(worker_id)
    try:
        workers_fd = _open_campaign_directory(root_fd, "workers", create=True)
        assert workers_fd is not None
        worker_fd = _open_campaign_directory(workers_fd, worker_name, create=True)
        assert worker_fd is not None
        telemetry_fd = os.open(
            filename,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=worker_fd,
        )
        details = os.fstat(telemetry_fd)
        if not stat.S_ISREG(details.st_mode):
            raise ValueError("policy telemetry path is unsafe")
    except OSError as error:
        raise ValueError("policy telemetry path is unsafe") from error
    finally:
        if telemetry_fd >= 0:
            os.close(telemetry_fd)
        if worker_fd >= 0:
            os.close(worker_fd)
        if workers_fd >= 0:
            os.close(workers_fd)
        os.close(root_fd)
    return _ProvisionedPolicyTelemetry(
        root / "workers" / worker_name / filename,
        details.st_dev,
        details.st_ino,
    )


class _PolicyTelemetrySampler:
    """Tail pre-provisioned worker files and retain only strict, attributable records."""

    _MAX_BYTES = 1_048_576
    _REQUIRED_KEYS = {"request_id", "latency_seconds", "queue_depth_after_enqueue"}

    def __init__(self, paths: dict[int, _ProvisionedPolicyTelemetry], *, wave_started_ns: int) -> None:
        self._paths = dict(paths)
        self._wave_started_ns = wave_started_ns
        self._fatal_failures: dict[int, list[str]] = {}
        self._latencies: list[float] = []
        self._queue_depths: list[int] = []
        self._offsets: dict[int, tuple[int, int, int]] = {}
        self._partial_lines: dict[int, bytes] = {}
        self._observed_workers: set[int] = set()

    def _record_failure(self, worker_id: int, reason: str) -> None:
        failures = self._fatal_failures.setdefault(worker_id, [])
        if reason not in failures:
            failures.append(reason)

    def _failure_records(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {"worker_id": worker_id, "failure_class": reason}
            for worker_id in sorted(self._fatal_failures)
            for reason in self._fatal_failures[worker_id]
        )

    def _read_worker(
        self,
        worker_id: int,
        provisioned: _ProvisionedPolicyTelemetry,
        *,
        final: bool,
    ) -> tuple[tuple[float, int], ...] | None:
        path = provisioned.path
        if path.parent.name != _worker_root_name(worker_id) or path.parent.parent.name != "workers":
            self._record_failure(worker_id, "policy_telemetry_wrong_worker")
            return None
        try:
            before = os.stat(path, follow_symlinks=False)
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
                self._record_failure(worker_id, "policy_telemetry_unsafe_path")
                return None
            if before.st_mtime_ns < self._wave_started_ns:
                self._record_failure(worker_id, "policy_telemetry_stale")
                return None
            if before.st_size > self._MAX_BYTES:
                self._record_failure(worker_id, "policy_telemetry_malformed")
                return None
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        except OSError:
            self._record_failure(worker_id, "policy_telemetry_missing")
            return None
        try:
            after = os.fstat(fd)
            if (
                not stat.S_ISREG(after.st_mode)
                or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
                or (after.st_dev, after.st_ino) != (provisioned.device, provisioned.inode)
            ):
                self._record_failure(worker_id, "policy_telemetry_unsafe_path")
                return None
            previous = self._offsets.get(worker_id)
            identity = (after.st_dev, after.st_ino)
            if previous is None:
                offset = 0
            elif previous[:2] != identity or after.st_size < previous[2]:
                self._record_failure(worker_id, "policy_telemetry_unsafe_path")
                return None
            else:
                offset = previous[2]
            remaining = after.st_size - offset
            if remaining > self._MAX_BYTES:
                self._record_failure(worker_id, "policy_telemetry_malformed")
                return None
            os.lseek(fd, offset, os.SEEK_SET)
            chunks: list[bytes] = []
            while remaining:
                chunk = os.read(fd, remaining)
                if not chunk:
                    self._record_failure(worker_id, "policy_telemetry_unsafe_path")
                    return None
                chunks.append(chunk)
                remaining -= len(chunk)
            payload = b"".join(chunks)
            self._offsets[worker_id] = (*identity, after.st_size)
        finally:
            os.close(fd)
        payload = self._partial_lines.pop(worker_id, b"") + payload
        if not payload:
            return ()
        values: list[tuple[float, int]] = []
        try:
            lines = payload.split(b"\n")
            trailing = lines.pop()
            if trailing:
                if final:
                    raise ValueError
                self._partial_lines[worker_id] = trailing
            for line in lines:
                if not line:
                    raise ValueError
                record = json.loads(line.decode("utf-8"))
                if not isinstance(record, dict) or set(record) != self._REQUIRED_KEYS:
                    raise ValueError
                request_id = record["request_id"]
                latency = record["latency_seconds"]
                queue_depth = record["queue_depth_after_enqueue"]
                if (
                    not isinstance(request_id, str)
                    or not request_id
                    or not isinstance(latency, (int, float))
                    or isinstance(latency, bool)
                    or not math.isfinite(latency)
                    or latency < 0
                    or not isinstance(queue_depth, int)
                    or isinstance(queue_depth, bool)
                    or queue_depth < 0
                ):
                    raise ValueError
                values.append((float(latency), queue_depth))
        except (UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError):
            self._record_failure(worker_id, "policy_telemetry_malformed")
            return None
        return tuple(values)

    def sample(self, *, final: bool = False) -> dict[str, object]:
        for worker_id, provisioned in self._paths.items():
            values = self._read_worker(worker_id, provisioned, final=final)
            if values is None:
                continue
            if values:
                self._observed_workers.add(worker_id)
                self._latencies.extend(latency for latency, _ in values)
                self._queue_depths.extend(depth for _, depth in values)
            elif final and worker_id not in self._observed_workers:
                self._record_failure(worker_id, "policy_telemetry_missing")
        failures = self._failure_records()
        valid = not failures and len(self._observed_workers) == len(self._paths) and self._latencies and self._queue_depths
        return {
            "inference_latency_seconds": max(self._latencies) if valid else None,
            "inference_queue_depth": max(self._queue_depths) if valid else None,
            "policy_evidence_failures": tuple(record["failure_class"] for record in failures),
            "policy_evidence_records": failures,
        }


def _trial_has_first_progress(output_root: Path, trial_id: str) -> bool:
    """The recorder's first committed annotation is stronger than a launched PID."""
    for parent in (".pending", "raw"):
        annotations = output_root / parent / trial_id / "annotations.jsonl"
        try:
            details = annotations.stat(follow_symlinks=False)
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(details.st_mode):
            continue
        if details.st_size > 0:
            return True
    return False


def _capacity_telemetry(gpu_indices: Sequence[int] | None = None) -> dict[str, object]:
    """Take one bounded host/assigned-GPU sample; unknown fields stay explicit."""
    host_margin, combined_vram_margin, _ = _resource_margins(gpu_indices)
    sample: dict[str, object] = {
        "host_ram_margin": host_margin,
        "combined_vram_margin": combined_vram_margin,
        "peak_host_ram_bytes": None,
        "peak_vram_bytes": None,
        "cpu_utilization": None,
        "run_queue": None,
        "inference_latency_seconds": None,
        "inference_queue_depth": None,
    }
    try:
        entries = {
            line.split(":", 1)[0]: int(line.split()[1]) * 1024
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines() if ":" in line
        }
        sample["peak_host_ram_bytes"] = entries["MemTotal"] - entries["MemAvailable"]
    except (KeyError, OSError, ValueError, IndexError):
        pass
    try:
        run_queue = Path("/proc/loadavg").read_text(encoding="utf-8").split()[3].split("/", 1)[0]
        sample["run_queue"] = int(run_queue)
    except (OSError, ValueError, IndexError):
        pass
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits"],
            check=False, capture_output=True, text=True, timeout=5.0,
        )
        observed = {
            int(index): (int(free), int(total))
            for line in result.stdout.splitlines()
            for index, free, total in [line.split(",")]
        }
        assigned = tuple(gpu_indices) if gpu_indices is not None else tuple(observed)
        usage = [observed[index][1] - observed[index][0] for index in assigned]
        if result.returncode == 0 and usage:
            sample["peak_vram_bytes"] = max(usage) * 1024 * 1024
    except (KeyError, OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return sample


def _cpu_counters() -> tuple[int, int] | None:
    """Return aggregate and idle CPU jiffies for a delta-based utilization reading."""
    try:
        fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
        if fields[0] != "cpu" or len(fields) < 5:
            return None
        values = tuple(int(value) for value in fields[1:])
    except (OSError, ValueError, IndexError):
        return None
    return sum(values), values[3] + (values[4] if len(values) > 4 else 0)


class _CapacityTelemetrySampler:
    """Associate each /proc/stat delta with an execution-time telemetry sample."""

    def __init__(self, gpu_indices: Sequence[int] | None = None) -> None:
        self._previous_cpu = _cpu_counters()
        self._gpu_indices = tuple(gpu_indices) if gpu_indices is not None else None

    def sample(self) -> dict[str, object]:
        sample = _capacity_telemetry(self._gpu_indices)
        current_cpu = _cpu_counters()
        if self._previous_cpu is not None and current_cpu is not None:
            total_delta = current_cpu[0] - self._previous_cpu[0]
            idle_delta = current_cpu[1] - self._previous_cpu[1]
            if total_delta > 0 and 0 <= idle_delta <= total_delta:
                sample["cpu_utilization"] = (total_delta - idle_delta) / total_delta
        self._previous_cpu = current_cpu
        return sample


def _cleanup_partially_launched_workers(
    args: argparse.Namespace,
    processes: Sequence[tuple[int, Trial, subprocess.Popen[str], Path, object, Path]],
) -> list[BaseException]:
    """Bound a best-effort shutdown without hiding the launch failure that caused it."""
    errors: list[BaseException] = []
    to_reap = list(processes)
    pending: list[tuple[int, Trial, subprocess.Popen[str], Path, object, Path]] = []
    for record in processes:
        worker_id, trial, process, heartbeat, log, log_path = record
        try:
            if process.poll() is None:
                pending.append(record)
        except BaseException as error:
            errors.append(RuntimeError(f"worker {worker_id} could not be polled during launch cleanup: {error}"))
            pending.append(record)

    for worker_id, trial, process, heartbeat, log, log_path in pending:
        try:
            process.terminate()
        except BaseException as error:
            errors.append(RuntimeError(f"worker {worker_id} could not be terminated during launch cleanup: {error}"))
        try:
            _write_heartbeat(heartbeat, worker_id=worker_id, trial_id=trial.trial_id, state="timeout")
        except BaseException as error:
            errors.append(RuntimeError(f"worker {worker_id} heartbeat cleanup failed: {error}"))

    terminate_deadline = time.monotonic() + args.terminate_grace_seconds
    while pending and time.monotonic() < terminate_deadline:
        still_pending = []
        for record in pending:
            worker_id, trial, process, heartbeat, log, log_path = record
            try:
                if process.poll() is None:
                    still_pending.append(record)
            except BaseException as error:
                errors.append(RuntimeError(f"worker {worker_id} could not be polled during launch cleanup: {error}"))
                still_pending.append(record)
        pending = still_pending
        if pending:
            time.sleep(min(0.1, max(0.0, terminate_deadline - time.monotonic())))

    for worker_id, trial, process, heartbeat, log, log_path in pending:
        try:
            process.kill()
        except BaseException as error:
            errors.append(RuntimeError(f"worker {worker_id} could not be killed during launch cleanup: {error}"))

    reap_deadline = time.monotonic() + args.terminate_grace_seconds
    for worker_id, trial, process, heartbeat, log, log_path in to_reap:
        remaining = max(0.0, reap_deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except BaseException as error:
            errors.append(RuntimeError(f"worker {worker_id} could not be reaped during launch cleanup: {error}"))

    for worker_id, trial, process, heartbeat, log, log_path in processes:
        try:
            log.close()
        except BaseException as error:
            errors.append(RuntimeError(f"worker {worker_id} log could not be closed during launch cleanup: {error}"))
    return errors


def _report_launch_cleanup_failures(launch_error: BaseException, cleanup_errors: Sequence[BaseException]) -> None:
    """Keep the launch exception primary while making cleanup faults observable on Python 3.10+."""
    if not cleanup_errors:
        return
    detail = "; ".join(str(error) for error in cleanup_errors)
    if hasattr(launch_error, "add_note"):
        launch_error.add_note(f"Additional launch cleanup failures: {detail}")
    else:
        print(f"Additional launch cleanup failures: {detail}", file=sys.stderr)


def _failure_classes(log_path: Path, *, returncode: int, progressed: bool) -> tuple[str, ...]:
    classes: list[str] = []
    if returncode:
        classes.append("timeout" if returncode == 124 else "nonzero_exit")
    if not progressed:
        classes.append("no_first_progress")
    if returncode:
        try:
            payload = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            payload = ""
        for marker, label in (
            (r"stale\s+ipc", "stale_ipc"),
            (r"vulkan", "vulkan"),
            (r"cuda", "cuda"),
            (r"policy", "policy"),
            (r"encoder", "video_encoder"),
        ):
            if re.search(rf"(?im)^\s*(?:error|fatal|critical)\b[^\n]*{marker}\b", payload):
                classes.append(label)
    return tuple(dict.fromkeys(classes))


def _run_worker_group(
    args: argparse.Namespace,
    assignments: Sequence[tuple[int, Trial]],
    *,
    gpu_indices: Sequence[int | None] | None = None,
    collect_telemetry: bool = False,
):
    """Start workers together and apply one launch-relative deadline to all."""
    started = time.monotonic()
    wave_started_ns = time.time_ns()
    processes: list[tuple[int, Trial, subprocess.Popen[str], Path, object, Path]] = []
    if gpu_indices is not None and len(gpu_indices) != len(assignments):
        raise ValueError("worker GPU assignment does not match the launched worker group")
    telemetry_samples: list[dict[str, object]] = []
    telemetry_sampler = _CapacityTelemetrySampler(gpu_indices) if collect_telemetry else None
    policy_telemetry_paths: dict[int, _ProvisionedPolicyTelemetry] = {}
    first_progress: dict[int, float] = {}
    launch_log: object | None = None
    policy_server_ports = _allocate_loopback_ports(len(assignments))
    try:
        for index, (worker_id, trial) in enumerate(assignments):
            worker_root = args.output_root / "workers" / f"worker-{worker_id:02d}"
            heartbeat = worker_root / "heartbeat.json"
            log_path, policy_server_log = _attempt_log_paths(worker_root, trial.trial_id)
            _prepare_retry_attempt(args.output_root, trial.trial_id)
            _write_heartbeat(heartbeat, worker_id=worker_id, trial_id=trial.trial_id, state="started")
            launch_log = log_path.open("x", encoding="utf-8")
            physical_gpu = gpu_indices[index] if gpu_indices is not None else None
            policy_telemetry_path = (
                _prepare_policy_telemetry_path(args.output_root, worker_id=worker_id)
                if collect_telemetry
                else None
            )
            policy_server_port = policy_server_ports[index]
            process = subprocess.Popen(
                _trial_command(
                    args,
                    trial,
                    device=f"cuda:{physical_gpu}" if physical_gpu is not None else getattr(args, "device", "cuda:0"),
                    policy_server_port=policy_server_port,
                    policy_server_log=policy_server_log,
                ), stdout=launch_log, stderr=subprocess.STDOUT,
                env=_worker_environment(
                    args,
                    physical_gpu,
                    policy_telemetry_path=(policy_telemetry_path.path if policy_telemetry_path else None),
                ),
            )
            if policy_telemetry_path is not None:
                policy_telemetry_paths[worker_id] = policy_telemetry_path
            processes.append((worker_id, trial, process, heartbeat, launch_log, log_path))
            launch_log = None
    except BaseException as launch_error:
        cleanup_errors = _cleanup_partially_launched_workers(args, processes)
        if launch_log is not None:
            try:
                launch_log.close()
            except BaseException as error:
                cleanup_errors.append(RuntimeError(f"unlaunched worker log could not be closed during launch cleanup: {error}"))
        _report_launch_cleanup_failures(launch_error, cleanup_errors)
        raise
    returncodes: dict[int, int] = {}
    pending = list(processes)
    policy_telemetry_sampler = (
        _PolicyTelemetrySampler(policy_telemetry_paths, wave_started_ns=wave_started_ns)
        if collect_telemetry
        else None
    )
    if collect_telemetry:
        sample = telemetry_sampler.sample()
        sample.update(policy_telemetry_sampler.sample())
        telemetry_samples.append(sample)
    deadline = started + args.worker_timeout_seconds
    while pending and time.monotonic() < deadline:
        still_pending: list[tuple[int, Trial, subprocess.Popen[str], Path, object, Path]] = []
        for worker_id, trial, process, heartbeat, log, log_path in pending:
            if worker_id not in first_progress and _trial_has_first_progress(args.output_root, trial.trial_id):
                first_progress[worker_id] = time.monotonic() - started
            returncode = process.poll()
            if returncode is None:
                still_pending.append((worker_id, trial, process, heartbeat, log, log_path))
                continue
            returncodes[worker_id] = returncode
            _write_heartbeat(heartbeat, worker_id=worker_id, trial_id=trial.trial_id, state="terminal")
        pending = still_pending
        if collect_telemetry:
            sample = telemetry_sampler.sample()
            sample.update(policy_telemetry_sampler.sample())
            telemetry_samples.append(sample)
        if pending:
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    if pending:
        for worker_id, trial, process, heartbeat, log, log_path in pending:
            process.terminate()
            _write_heartbeat(heartbeat, worker_id=worker_id, trial_id=trial.trial_id, state="timeout")
        grace_deadline = time.monotonic() + args.terminate_grace_seconds
        while pending and time.monotonic() < grace_deadline:
            still_pending = []
            for worker_id, trial, process, heartbeat, log, log_path in pending:
                if process.poll() is None:
                    still_pending.append((worker_id, trial, process, heartbeat, log, log_path))
            pending = still_pending
            if pending:
                time.sleep(min(0.1, max(0.0, grace_deadline - time.monotonic())))
        for worker_id, trial, process, heartbeat, log, log_path in pending:
            process.kill()
        reap_deadline = time.monotonic() + args.terminate_grace_seconds
        for worker_id, trial, process, heartbeat, log, log_path in pending:
            remaining = reap_deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"worker {worker_id} did not exit after SIGKILL")
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(f"worker {worker_id} did not exit after SIGKILL") from error
        for worker_id, trial, process, heartbeat, log, log_path in processes:
            if worker_id not in returncodes:
                returncodes[worker_id] = 124

    completed = failed = 0
    worker_failures: list[dict[str, object]] = []
    for worker_id, trial, process, heartbeat, log, log_path in processes:
        returncode = returncodes[worker_id]
        if worker_id not in first_progress and _trial_has_first_progress(args.output_root, trial.trial_id):
            first_progress[worker_id] = time.monotonic() - started
        log.close()
        if returncode == 0 and is_completed_trial(args.output_root, trial.trial_id):
            completed += 1
        else:
            failed += 1
        worker_failures.append({
            "worker_id": worker_id,
            "trial_id": trial.trial_id,
            "classes": list(_failure_classes(log_path, returncode=returncode, progressed=worker_id in first_progress)),
        })
    elapsed = time.monotonic() - started
    if not collect_telemetry:
        return elapsed, completed, failed
    sample = telemetry_sampler.sample()
    sample.update(policy_telemetry_sampler.sample(final=True))
    telemetry_samples.append(sample)
    return elapsed, completed, failed, first_progress, telemetry_samples, tuple(worker_failures)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True, help="committed canonical public 280-trial JSON")
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--policy-revision-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--policy-repo", required=True)
    parser.add_argument("--policy-step", type=int, required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--asset-revision", required=True)
    parser.add_argument("--release-assets-root", type=Path, required=True)
    parser.add_argument("--simulator-version", required=True)
    parser.add_argument("--policy-artifact-sha256", required=True)
    parser.add_argument("--image-identity", required=True)
    parser.add_argument("--capacity-sweep")
    parser.add_argument("--strategy", choices=("canonical", "mild", "strong"), default="canonical")
    parser.add_argument("--device", default="cuda:0", help="physical Isaac GPU forwarded to every trial")
    parser.add_argument("--groot-root", type=Path, help="pinned materialized GR00T checkout for policy-server children")
    parser.add_argument("--groot-revision", help="pinned GR00T checkout revision for policy-server children")
    parser.add_argument("--groot-python", type=Path, help="Python 3.10 interpreter in the pinned GR00T environment")
    parser.add_argument("--policy-server-readiness-timeout", type=_positive_finite_seconds, default=30.0)
    parser.add_argument("--policy-server-request-timeout", type=_positive_finite_seconds, default=2.5)
    parser.add_argument("--policy-server-termination-grace", type=_positive_finite_seconds, default=5.0)
    parser.add_argument("--trials-per-worker", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--worker-timeout-seconds", type=_positive_finite_seconds, default=1800.0)
    parser.add_argument("--terminate-grace-seconds", type=_positive_finite_seconds, default=5.0)
    parser.add_argument("--max-inference-latency-seconds", type=_positive_finite_seconds, default=0.5)
    parser.add_argument("--max-inference-queue-depth", type=_positive_int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_campaign(
    args: argparse.Namespace,
    *,
    runtime_preflight: Callable[[], object] | None = None,
) -> dict[str, object]:
    if (
        args.trials_per_worker <= 0
        or args.worker_timeout_seconds <= 0
        or not math.isfinite(args.worker_timeout_seconds)
        or not math.isfinite(args.terminate_grace_seconds)
        or args.terminate_grace_seconds <= 0
        or not math.isfinite(args.max_inference_latency_seconds)
        or args.max_inference_latency_seconds <= 0
        or args.max_inference_queue_depth <= 0
    ):
        raise ValueError("worker counts and timeouts must be finite and positive")
    # One host decision applies to every campaign child.  It deliberately runs
    # before matrix/output processing, subprocess creation, or policy hydration.
    if not args.dry_run:
        (runtime_preflight or require_isaac_sim_5_1_runtime)()
    matrix = load_public_matrix(args.matrix)
    trials = matrix.trials
    if not args.dry_run:
        required_server_values = (args.groot_root, args.groot_revision, args.groot_python)
        if any(value is None for value in required_server_values):
            raise ValueError("campaign execution requires pinned GR00T policy-server arguments")
        if not re.fullmatch(r"[0-9a-f]{40}", args.groot_revision):
            raise ValueError("GR00T revision must be a pinned 40-character SHA")
        device_index = _cuda_device_index(args.device)
        if device_index is None or args.device != f"cuda:{device_index}":
            raise ValueError("campaign execution requires --device cuda:<physical GPU>")
    args.output_root.mkdir(parents=True, exist_ok=True)
    state = CampaignState(args.output_root, tuple(trial.trial_id for trial in trials))
    by_id = {trial.trial_id: trial for trial in trials}
    pending = pending_trial_ids(state)
    records: list[dict[str, object]] = []
    if not args.dry_run and not args.capacity_sweep:
        for worker_id, trial_id in enumerate(pending, start=1):
            if (worker_id - 1) >= args.trials_per_worker:
                break
            returncode = _run_one_worker(args, worker_id=worker_id, trial=by_id[trial_id])
            records.append({"worker_id": worker_id, "trial_id": trial_id, "returncode": returncode, "mode": "sequential"})
    else:
        records = [{"trial_id": trial_id, "command": _trial_command(args, by_id[trial_id])} for trial_id in pending]

    if args.capacity_sweep and not args.dry_run:
        # A four-GPU sweep is made exclusively of documented 1/2/4 waves.
        # Do not consume a hidden sequential pilot before its 1-worker wave.
        capacity_pending = list(pending)
    elif args.dry_run:
        pending_after = pending
    else:
        pending_after = pending_trial_ids(state)

    report: dict[str, object] = {
        "schema_version": 1,
        "matrix": {
            "schema_version": matrix.schema_version,
            "sha256": matrix_sha256(matrix),
            "trial_count": len(trials),
            "training_holdouts": list(matrix.training_holdouts),
        },
        "pending_before": list(pending),
        "workers": records,
        "completed_after": [],
        "paths": {
            "raw_episodes": str(args.output_root / "raw"),
            "worker_logs": str(args.output_root / "workers"),
            "capacity_report": str(args.output_root / "capacity-report.json"),
        },
    }
    if args.capacity_sweep and not args.dry_run:
        counts = _validate_sweep(args.capacity_sweep)
        samples: list[CapacitySample] = []
        capacity_records: list[dict[str, object]] = []
        for count in counts:
            assignments = tuple((index + 1, by_id[trial_id]) for index, trial_id in enumerate(capacity_pending[:count]))
            if len(assignments) != count:
                capacity_records.append({"workers": count, "status": "skipped", "reason": "insufficient_pending_trials"})
                break
            try:
                gpu_indices = _worker_gpu_indices(args, count)
            except ValueError as error:
                capacity_records.append({
                    "workers": count,
                    "trial_ids": [trial.trial_id for _, trial in assignments],
                    "status": "skipped",
                    "reason": "unsupported_gpu_oversubscription",
                    "detail": str(error),
                })
                break
            capacity_pending = capacity_pending[count:]
            result = _run_worker_group(args, assignments, gpu_indices=gpu_indices, collect_telemetry=True)
            elapsed, completed, failed = result[:3]
            if len(result) == 3:
                first_progress, telemetry_samples, worker_failures = {}, [], ()
            else:
                _, _, _, first_progress, telemetry_samples, worker_failures = result
            if telemetry_samples:
                ram_margin = min(float(item["host_ram_margin"]) for item in telemetry_samples)
                combined_vram_margin = min(
                    float(item["combined_vram_margin"] if "combined_vram_margin" in item else item["inference_vram_margin"])
                    for item in telemetry_samples
                )
                peak_ram = max((item["peak_host_ram_bytes"] for item in telemetry_samples if item["peak_host_ram_bytes"] is not None), default=None)
                peak_vram = max((item["peak_vram_bytes"] for item in telemetry_samples if item["peak_vram_bytes"] is not None), default=None)
                max_run_queue = max((item["run_queue"] for item in telemetry_samples if item["run_queue"] is not None), default=None)
                max_cpu_utilization = max((item["cpu_utilization"] for item in telemetry_samples if item["cpu_utilization"] is not None), default=None)
                max_inference_latency = max((item["inference_latency_seconds"] for item in telemetry_samples if item["inference_latency_seconds"] is not None), default=None)
                max_inference_queue_depth = max((item["inference_queue_depth"] for item in telemetry_samples if item["inference_queue_depth"] is not None), default=None)
                policy_records_by_key: dict[tuple[object, object], dict[str, object]] = {}
                for item in telemetry_samples:
                    for record in item.get("policy_evidence_records", ()):
                        policy_records_by_key.setdefault(
                            (record["worker_id"], record["failure_class"]),
                            record,
                        )
                policy_evidence_records = tuple(policy_records_by_key.values())
                policy_evidence_failures = tuple(
                    record["failure_class"] for record in policy_evidence_records
                )
            else:
                ram_margin, combined_vram_margin, _ = _resource_margins(gpu_indices)
                peak_ram = peak_vram = max_run_queue = max_cpu_utilization = max_inference_latency = max_inference_queue_depth = None
                policy_evidence_records = ({"worker_id": worker_id, "failure_class": "policy_telemetry_missing"} for worker_id, _ in assignments)
                policy_evidence_records = tuple(policy_evidence_records)
                policy_evidence_failures = tuple(
                    record["failure_class"] for record in policy_evidence_records
                )
            attributed_worker_failures: dict[int, dict[str, object]] = {
                worker_id: {"worker_id": worker_id, "trial_id": trial.trial_id, "classes": []}
                for worker_id, trial in assignments
            }
            for worker_failure in worker_failures:
                worker_id = worker_failure.get("worker_id")
                if worker_id not in attributed_worker_failures:
                    continue
                classes = attributed_worker_failures[worker_id]["classes"]
                assert isinstance(classes, list)
                for failure_class in worker_failure.get("classes", ()):
                    if failure_class not in classes:
                        classes.append(failure_class)
            for record in policy_evidence_records:
                worker_id = record["worker_id"]
                failure_class = record["failure_class"]
                worker_failure = attributed_worker_failures.get(worker_id)
                if worker_failure is None:
                    continue
                classes = worker_failure["classes"]
                assert isinstance(classes, list)
                if failure_class not in classes:
                    classes.append(failure_class)
            worker_failures = tuple(attributed_worker_failures.values())
            sample = CapacitySample(
                # Isaac and the colocated GR00T service share each assigned GPU;
                # use their observed headroom once, not as two fake resources.
                count, elapsed, completed, failed, combined_vram_margin, 1.0, ram_margin,
                first_progress_workers=len(first_progress) if first_progress or telemetry_samples else None,
                stale_ipc_count=sum(
                    1
                    for worker_failure in worker_failures
                    if "stale_ipc" in worker_failure.get("classes", ())
                ),
                peak_host_ram_bytes=peak_ram, peak_vram_bytes=peak_vram, cpu_utilization=max_cpu_utilization,
                run_queue=max_run_queue, inference_latency_seconds=max_inference_latency,
                inference_queue_depth=max_inference_queue_depth,
                policy_evidence_failures=policy_evidence_failures,
                failure_classes=tuple(
                    failure_class
                    for worker_failure in worker_failures
                    for failure_class in worker_failure.get("classes", ())
                ),
            )
            samples.append(sample)
            failure_counts: dict[str, int] = {}
            for worker_failure in worker_failures:
                for failure_class in worker_failure.get("classes", ()):
                    failure_counts[failure_class] = failure_counts.get(failure_class, 0) + 1
            capacity_records.append({
                "workers": count, "trial_ids": [trial.trial_id for _, trial in assignments], "gpu_indices": list(gpu_indices),
                "elapsed_seconds": elapsed, "completed_trials": completed, "failed_trials": failed,
                "first_progress_seconds": {str(worker): seconds for worker, seconds in first_progress.items()},
                "host_ram_margin": ram_margin, "combined_vram_margin": combined_vram_margin,
                "peak_host_ram_bytes": peak_ram,
                "peak_vram_bytes": peak_vram, "max_cpu_utilization": max_cpu_utilization, "max_run_queue": max_run_queue,
                "inference_latency_seconds": max_inference_latency,
                "inference_queue_depth": max_inference_queue_depth,
                "policy_evidence_failures": list(policy_evidence_failures),
                "policy_evidence_records": list(policy_evidence_records),
                "worker_failures": list(worker_failures),
                "failure_counts": failure_counts,
            })
            if choose_worker_count(
                samples,
                max_inference_latency_seconds=args.max_inference_latency_seconds,
                max_inference_queue_depth=args.max_inference_queue_depth,
            ).accepted_workers != count:
                break
        decision = choose_worker_count(
            samples,
            max_inference_latency_seconds=args.max_inference_latency_seconds,
            max_inference_queue_depth=args.max_inference_queue_depth,
        )
        if not samples:
            decision = CapacityDecision(0, {counts[0]: ("no_valid_capacity_sample",)})
        report["capacity"] = {
            "requested": list(counts),
            "max_inference_latency_seconds": args.max_inference_latency_seconds,
            "max_inference_queue_depth": args.max_inference_queue_depth,
            "samples": capacity_records,
            "accepted_workers": decision.accepted_workers,
            "rejected": decision.rejected,
        }
        pending_after = pending_trial_ids(state)
    elif args.capacity_sweep:
        report["capacity"] = {"requested": list(_validate_sweep(args.capacity_sweep)), "status": "dry_run_no_processes"}
    pending_after_set = set(pending_after)
    report["completed_after"] = [trial_id for trial_id in state.trial_ids if trial_id not in pending_after_set]
    capacity = report.get("capacity", {})
    capacity_samples = capacity.get("samples", []) if isinstance(capacity, dict) else []
    wave_trial_ids = [
        record["trial_ids"]
        for record in capacity_samples
        if isinstance(record, dict) and "trial_ids" in record and record.get("status") != "skipped"
    ]
    sequential_trial_ids = [
        record["trial_id"] for record in records
        if record.get("mode") == "sequential" and "trial_id" in record
    ]
    attempted = set(sequential_trial_ids)
    for wave in wave_trial_ids:
        attempted.update(wave)
    report["episode_accounting"] = {
        "sequential_trial_ids": sequential_trial_ids,
        "capacity_wave_trial_ids": wave_trial_ids,
        "attempt_count": len(sequential_trial_ids) + sum(len(wave) for wave in wave_trial_ids),
        "attempted_unique_trial_ids": sorted(attempted),
    }
    (args.output_root / "capacity-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_campaign(args)
    except ValueError as error:
        print(f"campaign validation error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CampaignState", "pending_trial_ids", "run_campaign"]
