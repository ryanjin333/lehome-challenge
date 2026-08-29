#!/usr/bin/env python3
"""Local-only terminal finalizer for the one approved simple-curriculum VM.

The paid controller only writes a compact handoff.  This separate process owns
the provider STOPPED observation and the subsequent zero-compute publication.
It has no create/start/delete operation.
"""
from __future__ import annotations

from hashlib import sha256
import hashlib
import argparse
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import tempfile
import time
import re
from typing import Mapping, Protocol


EXACT_INSTANCE_ID = "computeinstance-u00t6xfqhadrcmssa2"
EXACT_INSTANCE_NAME = "lehome-rollout"
PROTECTED_DISK_ID = "computedisk-u00pbe55crxy7jr56x"
EXACT_IMAGE_ID = "computeimage-u00zf6w3yf72gakhcy"
TERMINAL_OUTCOMES = frozenset({"complete", "replay_shortage", "fidelity_stop", "infrastructure_stop", "insufficient_source_stop", "infrastructure_stop_failure"})
_COLLECTION_STAGES = (
    "calibration-matrix", "calibration-head", "first-100-gate",
    "calibration-tail", "calibration-report", "curriculum-matrix",
    "curriculum-a", "curriculum-b", "fresh-report", "replay-matrix",
    "success-replay",
)
_GATE_STAGES = _COLLECTION_STAGES[:3]
_REPLAY_SHORTAGE_STAGE_SETS = (_COLLECTION_STAGES[:10], _COLLECTION_STAGES)
_STOP_OUTCOMES = frozenset({"infrastructure_stop", "infrastructure_stop_failure"})
_SSH_TARGET = re.compile(
    r"[A-Za-z_][A-Za-z0-9_.-]{0,63}@"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)(?:\.(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?))*"
)
_RUN_ID = re.compile(r"^fresh-run-[a-z0-9-]{1,112}$")
_ROUND_ID = re.compile(r"^fresh-12k-[a-z0-9-]{1,112}$")


class FinalizationError(RuntimeError):
    pass


class Provider(Protocol):
    def get(self, instance_id: str) -> Mapping[str, object]: ...
    def stop(self, instance_id: str) -> None: ...


class Publisher(Protocol):
    def publish(self, root: Path, *, handoff: Mapping[str, object], stop_observation: Mapping[str, object], seal: Mapping[str, object]) -> Mapping[str, object]: ...


def _checked_absolute(value: str, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts) or path.is_symlink():
        raise FinalizationError(f"{label} is unsafe")
    return path


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _digest(value: object) -> str:
    return sha256(_canonical(value)).hexdigest()


def _hex(value: object, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)


def _entry_digest(entries: object) -> str:
    """Match the reviewed publisher's immutable entry-manifest digest."""
    try:
        payload = [
            {"relative_path": item.relative_path, "sha256": item.sha256, "byte_size": item.byte_size}
            for item in entries  # type: ignore[union-attr]
        ]
    except (AttributeError, TypeError) as error:
        raise FinalizationError("finalization evidence manifest is malformed") from error
    return _digest(payload)


def _json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise FinalizationError(f"{label} is malformed") from error
    return _json_bytes_object(raw, label=label)


def _json_bytes_object(raw: bytes, *, label: str) -> dict[str, object]:
    def reject_duplicate_keys(pairs: list[tuple[object, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if not isinstance(key, str) or key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=reject_duplicate_keys)
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise FinalizationError(f"{label} is malformed") from error
    if not isinstance(value, dict):
        raise FinalizationError(f"{label} is malformed")
    return value


def _load_operator_token(path: Path) -> str:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise FinalizationError("operator HF token file is unavailable") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_uid != os.geteuid()
    ):
        raise FinalizationError("operator HF token file must be regular, owned, and mode 0600")
    try:
        token = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as error:
        raise FinalizationError("operator HF token file is unreadable") from error
    if not token or any(character.isspace() for character in token):
        raise FinalizationError("operator HF token file is empty or invalid")
    return token


def validate_finalization_receipt(
    raw: Mapping[str, object], *, handoff: Mapping[str, object], evidence_revision: str,
    evidence_bundle_sha256: str | None, seal: Mapping[str, object],
) -> dict[str, object]:
    payload = dict(raw)
    declared = payload.pop("receipt_sha256", None)
    required = {
        "schema_version", "kind", "run_id", "round_id", "evidence_revision",
        "evidence_bundle_sha256", "final_seal_sha256", "readback_verified",
        "public_readback_verified", "receipt_sha256",
    }
    if set(raw) != required or declared != _digest(payload):
        raise FinalizationError("final publication receipt is malformed")
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "lehome_simple_curriculum_operator_finalization_receipt_v1"
        or payload.get("run_id") != handoff.get("run_id")
        or payload.get("round_id") != handoff.get("round_id")
        or payload.get("evidence_revision") != evidence_revision
        or (evidence_bundle_sha256 is not None and payload.get("evidence_bundle_sha256") != evidence_bundle_sha256)
        or payload.get("final_seal_sha256") != seal.get("seal_sha256")
        or payload.get("readback_verified") is not True
        or payload.get("public_readback_verified") is not True
        or not _hex(payload.get("evidence_revision"), 40)
        or not _hex(payload.get("evidence_bundle_sha256"), 64)
        or not _hex(payload.get("final_seal_sha256"), 64)
    ):
        raise FinalizationError("final publication receipt does not bind immutable evidence")
    return dict(raw)


def validate_handoff(raw: Mapping[str, object]) -> dict[str, object]:
    payload = dict(raw); declared = payload.pop("handoff_sha256", None)
    required = {"schema_version", "kind", "run_id", "round_id", "instance_id", "terminal_outcome", "predecessor_receipt_sha256", "code_revision", "code_tree_sha256", "runtime_identity", "runtime_identity_sha256", "first_100_receipt_sha256", "evidence", "handoff_sha256"}
    v2_required = required | {"provisional_publication"}
    if (set(raw) != required and set(raw) != v2_required) or declared != _digest(payload):
        raise FinalizationError("operator handoff is malformed")
    legacy = set(raw) == required
    if (legacy and (payload["schema_version"] != 1 or payload["kind"] != "lehome_simple_curriculum_operator_stop_handoff_v1")) or (
        not legacy and (payload["schema_version"] != 2 or payload["kind"] != "lehome_simple_curriculum_operator_stop_handoff_v2")
    ):
        raise FinalizationError("operator handoff kind is invalid")
    if payload["instance_id"] != EXACT_INSTANCE_ID or not isinstance(payload["run_id"], str) or not isinstance(payload["round_id"], str) or _RUN_ID.fullmatch(payload["run_id"]) is None or _ROUND_ID.fullmatch(payload["round_id"]) is None:
        raise FinalizationError("operator handoff is not bound to the approved VM")
    if not legacy:
        provisional = payload.get("provisional_publication")
        if (
            not isinstance(provisional, Mapping)
            or set(provisional) != {"immutable_revision", "bundle_sha256", "manifest_sha256", "receipt_sha256", "repository", "remote_prefix"}
            or provisional.get("repository") != "ryanjin333/lehome-groot-n17-rollouts"
            or provisional.get("remote_prefix") != f"collection-rounds/{payload['run_id']}/manifests/provisional"
            or not _hex(provisional.get("immutable_revision"), 40) or not _hex(provisional.get("bundle_sha256"), 64)
            or not _hex(provisional.get("manifest_sha256"), 64) or not _hex(provisional.get("receipt_sha256"), 64)
        ):
            raise FinalizationError("operator handoff provisional binding is invalid")
    if payload["terminal_outcome"] not in TERMINAL_OUTCOMES:
        raise FinalizationError("operator handoff outcome is invalid")
    if not _hex(payload["code_revision"], 40) or not _hex(payload["code_tree_sha256"], 64):
        raise FinalizationError("operator handoff code identity is invalid")
    if not isinstance(payload["runtime_identity"], Mapping) or payload["runtime_identity_sha256"] != _digest(dict(payload["runtime_identity"])):
        raise FinalizationError("operator handoff runtime identity is invalid")
    if not isinstance(payload["evidence"], list):
        raise FinalizationError("operator handoff evidence is invalid")
    stages: list[str] = []
    receipts: list[str] = []
    for item in payload["evidence"]:
        if not isinstance(item, Mapping) or set(item) != {"stage", "receipt_sha256", "file_sha256"} or not isinstance(item["stage"], str) or not _hex(item["receipt_sha256"], 64) or not _hex(item["file_sha256"], 64):
            raise FinalizationError("operator handoff evidence is invalid")
        stages.append(item["stage"])
        receipts.append(item["receipt_sha256"])
    outcome = payload["terminal_outcome"]
    actual_stages = tuple(stages)
    if outcome == "complete":
        expected = _COLLECTION_STAGES
    elif outcome == "replay_shortage":
        expected = None
        if actual_stages not in _REPLAY_SHORTAGE_STAGE_SETS:
            raise FinalizationError("replay-shortage handoff is not at a reachable terminal boundary")
    elif outcome in {"fidelity_stop", "insufficient_source_stop"}:
        expected = _GATE_STAGES
    elif outcome in _STOP_OUTCOMES:
        expected = None
        if actual_stages not in tuple(_COLLECTION_STAGES[:index] for index in range(len(_COLLECTION_STAGES) + 1)):
            raise FinalizationError("stop handoff evidence is not a reachable stage prefix")
    else:  # guarded above; retained as an explicit fail-closed boundary.
        raise FinalizationError("operator handoff outcome is invalid")
    if expected is not None and actual_stages != expected:
        raise FinalizationError("terminal handoff does not have the exact required evidence stages")
    predecessor = payload["predecessor_receipt_sha256"]
    if actual_stages:
        if not _hex(predecessor, 64) or predecessor != receipts[-1]:
            raise FinalizationError("operator handoff predecessor is not bound to final evidence")
    elif predecessor is not None:
        raise FinalizationError("empty stop handoff must have an explicit null predecessor")
    first_gate = next((receipt for stage, receipt in zip(stages, receipts, strict=True) if stage == "first-100-gate"), None)
    if payload["first_100_receipt_sha256"] != first_gate:
        raise FinalizationError("operator handoff first-100 receipt is not bound to gate evidence")
    return dict(raw)


def _validate_instance(raw: Mapping[str, object]) -> dict[str, object]:
    value = dict(raw)
    # Nebius CLI/API's authoritative instance representation is nested. Keep
    # the old flat shape only for existing offline controller seams.
    if all(key in value for key in ("metadata", "status", "spec")):
        metadata, status, spec = value["metadata"], value["status"], value["spec"]
        if not isinstance(metadata, Mapping) or not isinstance(status, Mapping) or not isinstance(spec, Mapping):
            raise FinalizationError("provider response is not the exact protected rollout VM")
        disks = spec.get("secondary_disks")
        found = []
        if isinstance(disks, list):
            for disk in disks:
                existing = disk.get("existing_disk") if isinstance(disk, Mapping) else None
                # Secondary attachment must be exactly the one existing
                # protected disk; a managed/unknown extra is not safe.
                if not isinstance(disk, Mapping) or set(disk) - {"attach_mode", "existing_disk", "device_id"} or not isinstance(existing, Mapping) or set(existing) != {"id"}:
                    raise FinalizationError("provider response is not the exact protected rollout VM")
                found.append(existing.get("id"))
        boot = spec.get("boot_disk")
        managed = boot.get("managed_disk") if isinstance(boot, Mapping) else None
        boot_spec = managed.get("spec") if isinstance(managed, Mapping) else None
        image = boot_spec.get("source_image_id") if isinstance(boot_spec, Mapping) else None
        if metadata.get("id") != EXACT_INSTANCE_ID or metadata.get("name") != EXACT_INSTANCE_NAME or status.get("state") not in {"RUNNING", "STOPPING", "STOPPED"} or found != [PROTECTED_DISK_ID] or image != EXACT_IMAGE_ID:
            raise FinalizationError("provider response is not the exact protected rollout VM")
        return {"state": status["state"], "raw": value}
    raise FinalizationError("provider response is not the real nested Nebius instance shape")


def stop_exact_instance(provider: Provider, *, timeout_seconds: float) -> dict[str, object]:
    if not isinstance(timeout_seconds, (int, float)) or not 0 < timeout_seconds <= 600:
        raise FinalizationError("stop timeout is invalid")
    deadline = time.monotonic() + float(timeout_seconds)
    set_deadline = getattr(provider, "set_stop_deadline", None)
    if callable(set_deadline):
        set_deadline(deadline)
    try:
        first = _validate_instance(provider.get(EXACT_INSTANCE_ID))
        if first["state"] == "RUNNING":
            provider.stop(EXACT_INSTANCE_ID)
        while True:
            observed = _validate_instance(provider.get(EXACT_INSTANCE_ID))
            if observed["state"] == "STOPPED":
                body = {"schema_version": 1, "kind": "lehome_simple_curriculum_stopped_observation_v1", "instance_id": EXACT_INSTANCE_ID, "instance_name": EXACT_INSTANCE_NAME, "protected_disk_id": PROTECTED_DISK_ID, "state": "STOPPED", "provider_response_sha256": _digest(observed.get("raw", observed))}
                return {**body, "observation_sha256": _digest(body)}
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise FinalizationError("provider did not report STOPPED before timeout")
            time.sleep(min(0.25, remaining))
    finally:
        if callable(set_deadline):
            set_deadline(None)


def _seal(handoff: Mapping[str, object], stopped: Mapping[str, object]) -> dict[str, object]:
    outcome = handoff["terminal_outcome"]
    # The remote handoff must have a separately validated exact-cap report;
    # without it terminal claims remain intentionally non-complete.
    complete = outcome == "complete" and handoff.get("schema_version") == 2
    if outcome == "complete": claim = "collection_complete" if complete else "caps_unverified"
    elif outcome in {"replay_shortage", "insufficient_source_stop"}: claim = "insufficient_fresh_source"
    else: claim = "fidelity_infrastructure_stop"
    body = {"schema_version": 1, "kind": "lehome_simple_curriculum_final_seal_v1", "run_id": handoff["run_id"], "round_id": handoff["round_id"], "terminal_outcome": outcome, "completion_claim": claim, "gpu_stop_verified": True, "handoff_sha256": handoff["handoff_sha256"], "stopped_observation_sha256": stopped["observation_sha256"]}
    return {**body, "seal_sha256": _digest(body)}


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise FinalizationError("finalization staging path is unsafe")
    temporary = path.with_name(path.name + ".tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(_canonical(value)); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()


class SubprocessNebiusProvider:
    """Restricted exact-instance adapter; it cannot create, start, or delete."""
    def __init__(self, command: tuple[str, ...] = ("nebius",), *, request_timeout_seconds: float = 15.0) -> None:
        if not isinstance(request_timeout_seconds, (int, float)) or not 0 < request_timeout_seconds <= 60:
            raise ValueError("Nebius request timeout is invalid")
        self.command = command
        self.request_timeout_seconds = float(request_timeout_seconds)
        self._stop_deadline: float | None = None

    def set_stop_deadline(self, deadline: float | None) -> None:
        if deadline is not None and (not isinstance(deadline, (int, float)) or deadline <= time.monotonic()):
            raise FinalizationError("stop deadline is exhausted")
        self._stop_deadline = None if deadline is None else float(deadline)

    def _timeout_budget(self) -> tuple[float, str]:
        remaining = self.request_timeout_seconds
        if self._stop_deadline is not None:
            remaining = self._stop_deadline - time.monotonic()
        if remaining <= 0:
            raise FinalizationError("Nebius stop deadline expired")
        process_timeout = min(self.request_timeout_seconds, remaining)
        # CLI duration flags are deliberately no longer than the subprocess
        # wall-clock cap.  The process cap remains authoritative for fractions.
        duration = f"{max(1, int(process_timeout))}s"
        return process_timeout, duration

    def _json(self, arguments: tuple[str, ...]) -> Mapping[str, object]:
        process_timeout, duration = self._timeout_budget()
        command = self.command + arguments + (
            "--format", "json", "--no-browser", "--no-progress", "--no-check-update",
            "--auth-timeout", duration, "--per-retry-timeout", duration,
            "--timeout", duration, "--retries", "1",
        )
        try:
            result = subprocess.run(
                command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=False, timeout=process_timeout,
            )
        except subprocess.TimeoutExpired as error:
            raise FinalizationError("Nebius exact-instance command timed out") from error
        if result.returncode:
            raise FinalizationError("Nebius exact-instance command failed")
        try: value = json.loads(result.stdout)
        except json.JSONDecodeError as error: raise FinalizationError("Nebius exact-instance response is not JSON") from error
        if not isinstance(value, Mapping): raise FinalizationError("Nebius exact-instance response is invalid")
        return value
    def get(self, instance_id: str) -> Mapping[str, object]:
        if instance_id != EXACT_INSTANCE_ID: raise FinalizationError("refusing non-approved instance")
        return self._json(("compute", "instance", "get", instance_id))
    def stop(self, instance_id: str) -> None:
        if instance_id != EXACT_INSTANCE_ID: raise FinalizationError("refusing non-approved instance")
        self._json(("compute", "instance", "stop", instance_id))


def _owned_directory_fd(path: Path, *, create: bool) -> int:
    """Open one operator-owned directory without following its final component."""
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise FinalizationError("durable handoff directory is unsafe")
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        if not create:
            raise FinalizationError("durable handoff directory is unavailable")
        parent = path.parent
        try:
            parent_metadata = parent.lstat()
        except OSError as error:
            raise FinalizationError("durable handoff parent is unsafe") from error
        trusted_owner = parent_metadata.st_uid == os.geteuid()
        trusted_sticky_root = (
            parent_metadata.st_uid == 0
            and bool(parent_metadata.st_mode & stat.S_ISVTX)
        )
        if (not stat.S_ISDIR(parent_metadata.st_mode)
                or not (trusted_owner or trusted_sticky_root)
                or parent.is_symlink()):
            raise FinalizationError("durable handoff parent is unsafe")
        parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            parent_fd = os.open(parent, parent_flags)
        except OSError as error:
            raise FinalizationError("durable handoff parent is unsafe") from error
        try:
            os.mkdir(path.name, mode=0o700, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except FileExistsError:
            pass
        except OSError as error:
            raise FinalizationError("durable handoff directory cannot be created") from error
        finally:
            os.close(parent_fd)
        try:
            metadata = path.lstat()
        except OSError as error:
            raise FinalizationError("durable handoff directory is unsafe") from error
    except OSError as error:
        raise FinalizationError("durable handoff directory is unsafe") from error
    if (path.is_symlink() or not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o022):
        raise FinalizationError("durable handoff directory is unsafe")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise FinalizationError("durable handoff directory is unsafe") from error
    opened = os.fstat(descriptor)
    if (not stat.S_ISDIR(opened.st_mode) or opened.st_uid != os.geteuid()
            or (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino)):
        os.close(descriptor)
        raise FinalizationError("durable handoff directory changed while opening")
    return descriptor


def _persist_durable_bytes(destination: Path, payload: bytes) -> None:
    if not destination.is_absolute() or destination.name in {"", ".", ".."}:
        raise FinalizationError("handoff destination is unsafe")
    parent_fd = _owned_directory_fd(destination.parent, create=True)
    temporary_name = f".{destination.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    temporary_created = False
    try:
        try:
            os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FinalizationError("handoff destination is unsafe")
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary_name, flags, 0o600, dir_fd=parent_fd)
        temporary_created = True
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(
            temporary_name, destination.name,
            src_dir_fd=parent_fd, dst_dir_fd=parent_fd,
        )
        temporary_created = False
        os.fsync(parent_fd)
        persisted = os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        if (not stat.S_ISREG(persisted.st_mode) or persisted.st_uid != os.geteuid()
                or stat.S_IMODE(persisted.st_mode) != 0o600):
            raise FinalizationError("durable handoff file is unsafe")
    except FinalizationError:
        raise
    except OSError as error:
        raise FinalizationError("durable handoff persistence failed") from error
    finally:
        if temporary_created:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)


def _persist_durable_json(destination: Path, value: Mapping[str, object]) -> None:
    _persist_durable_bytes(destination, _canonical(value))


def _read_durable_handoff(path: Path) -> dict[str, object]:
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise FinalizationError("durable operator handoff is unsafe")
    parent_fd = _owned_directory_fd(path.parent, create=False)
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path.name, flags, dir_fd=parent_fd)
        except OSError as error:
            raise FinalizationError("durable operator handoff is unsafe") from error
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            metadata = os.fstat(handle.fileno())
            if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid()
                    or stat.S_IMODE(metadata.st_mode) != 0o600):
                raise FinalizationError("durable operator handoff is unsafe")
            payload = handle.read()
    finally:
        os.close(parent_fd)
    return _json_bytes_object(payload, label="durable operator handoff")


def fetch_remote_handoff(*, ssh_target: str, port: int, campaign_root: str, destination: Path) -> dict[str, object]:
    """Fetch only the compact handoff over noninteractive SSH into temp storage."""
    root = _checked_absolute(campaign_root, label="remote campaign root")
    if not isinstance(port, int) or not 1 <= port <= 65535 or _SSH_TARGET.fullmatch(ssh_target or "") is None:
        raise FinalizationError("SSH target or port is invalid")
    remote = str(root / "reports" / "operator-stop-handoff.json")
    command = ("ssh", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2", "-p", str(port), "--", ssh_target, "cat", "--", remote)
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=20)
    except subprocess.TimeoutExpired as error:
        raise FinalizationError("remote handoff fetch timed out") from error
    if result.returncode: raise FinalizationError("remote handoff fetch failed")
    value = _json_bytes_object(result.stdout, label="remote handoff")
    _persist_durable_bytes(destination, result.stdout)
    return value


def _durable_handoff_path(run_id: str) -> Path:
    if _RUN_ID.fullmatch(run_id) is None:
        raise FinalizationError("operator handoff run ID is invalid")
    return Path(tempfile.gettempdir()) / "lehome-simple-curriculum-finalizer" / f"{run_id}.handoff.json"


class HfFinalizerPublisher:
    """Compact two-phase public finalization using the reviewed transport."""
    repository = "ryanjin333/lehome-groot-n17-rollouts"
    def __init__(self, token_path: Path, *, module: object | None = None, transport: object | None = None) -> None:
        self.token_path = token_path; self._module = module; self._transport = transport

    def _verify_provisional(self, *, module: object, transport: object, root: Path, token: str, handoff: Mapping[str, object]) -> None:
        """Fetch only the pinned compact JSON bundle; never SSH campaign bytes."""
        provisional = handoff.get("provisional_publication")
        if not isinstance(provisional, Mapping):
            return  # v1 is retained solely for old offline receipt fixtures.
        revision = str(provisional["immutable_revision"])
        prefix = str(provisional["remote_prefix"])
        stages = tuple(str(item["stage"]) for item in handoff["evidence"])
        files = (
            "manifests/provisional/evidence-manifest.json",
            "manifests/provisional/task6-validation.json",
            "manifests/provisional/hub-artifact-references.json",
            *(f"manifests/provisional/stage-receipts/{stage}.json" for stage in stages),
        )
        base_prefix = f"collection-rounds/{handoff['run_id']}"
        try:
            observed = module._tree_files(
                transport.list_tree(repository=self.repository, revision=revision, token=token, remote_prefix=base_prefix),
                prefix=base_prefix,
            )
            if observed != set(files): raise FinalizationError("provisional tree is not exact")
            # Authenticated and anonymous reads use wholly separate trees.
            # Neither path may overwrite the other before their bytes and
            # returned immutable revision have independently been checked.
            with tempfile.TemporaryDirectory(prefix="lehome-provisional-auth-", dir=root) as auth_dir, tempfile.TemporaryDirectory(prefix="lehome-provisional-public-", dir=root) as public_dir:
                destinations = ((Path(auth_dir), token), (Path(public_dir), None))
                for destination, auth in destinations:
                    observed_revision = transport.download_files(repository=self.repository, revision=revision, destination=destination, relative_paths=files, token=auth, remote_prefix=base_prefix)
                    if observed_revision != revision:
                        raise FinalizationError("provisional readback did not return pinned revision")
                for relative in files:
                    authenticated = Path(auth_dir) / relative
                    anonymous = Path(public_dir) / relative
                    if not authenticated.is_file() or not anonymous.is_file() or authenticated.read_bytes() != anonymous.read_bytes():
                        raise FinalizationError("authenticated and anonymous provisional bytes differ")
                    local = root / relative; local.parent.mkdir(parents=True, exist_ok=True); local.write_bytes(anonymous.read_bytes())
            bundle = module.CollectionPublicationBundle(root=root, run_id=handoff["run_id"], repository=self.repository, revision="main", files=files)
            entries = module._collect_entries(bundle)
            if module._entry_digest(entries) != provisional["bundle_sha256"]:
                raise FinalizationError("provisional bundle bytes do not bind handoff")
            receipt_body = {
                "schema_version": 1,
                "kind": "lehome_simple_curriculum_provisional_publication_receipt_v1",
                "run_id": handoff["run_id"],
                "round_id": handoff["round_id"],
                "repository": self.repository,
                "remote_prefix": prefix,
                "immutable_revision": revision,
                "entry_count": len(entries),
                "entries": [
                    {
                        "relative_path": entry.relative_path,
                        "sha256": entry.sha256,
                        "byte_size": entry.byte_size,
                    }
                    for entry in entries
                ],
                "bundle_sha256": module._entry_digest(entries),
                "manifest_sha256": provisional["manifest_sha256"],
                "readback_verified": True,
                "public_readback_verified": True,
            }
            if _digest(receipt_body) != provisional["receipt_sha256"]:
                raise FinalizationError("provisional receipt does not bind verified bundle")
            manifest = _json_object(root / files[0], label="provisional evidence manifest")
            if hashlib.sha256(_canonical(manifest)).hexdigest() != provisional["manifest_sha256"]:
                raise FinalizationError("provisional manifest bytes do not bind handoff")
            manifest_fields = {
                "schema_version", "kind", "run_id", "round_id", "instance_id", "campaign_root",
                "terminal_outcome", "reachable_stages", "stage_receipts", "terminal_chain_head",
                "first_100_receipt_sha256", "task6_validation_sha256",
                "hub_artifact_references_sha256", "completion_claim", "gpu_stop_verified",
            }
            if (set(manifest) != manifest_fields
                    or manifest.get("schema_version") != 1
                    or manifest.get("kind") != "lehome_simple_curriculum_provisional_evidence_manifest_v1"
                    or manifest.get("run_id") != handoff.get("run_id") or manifest.get("round_id") != handoff.get("round_id")
                    or manifest.get("instance_id") != EXACT_INSTANCE_ID
                    or manifest.get("campaign_root") != f"/mnt/lehome/eval/{handoff['run_id']}"
                    or manifest.get("terminal_outcome") != handoff.get("terminal_outcome")
                    or manifest.get("reachable_stages") != list(stages)
                    or manifest.get("completion_claim") != "none"
                    or manifest.get("gpu_stop_verified") is not False):
                raise FinalizationError("provisional manifest identity is invalid")
            listed = manifest.get("stage_receipts")
            if not isinstance(listed, list) or len(listed) != len(stages):
                raise FinalizationError("provisional stage chain is invalid")
            predecessor: str | None = None
            identity: object | None = None
            handoff_evidence = handoff.get("evidence")
            if not isinstance(handoff_evidence, list) or len(handoff_evidence) != len(stages):
                raise FinalizationError("provisional stage chain is invalid")
            for index, item in enumerate(listed):
                if not isinstance(item, Mapping) or set(item) != {
                    "stage", "receipt_sha256", "file_sha256", "predecessor_receipt_sha256",
                }:
                    raise FinalizationError("provisional stage chain is invalid")
                handoff_item = handoff_evidence[index]
                if (not isinstance(handoff_item, Mapping)
                        or item.get("stage") != stages[index]
                        or item.get("stage") != handoff_item.get("stage")
                        or item.get("receipt_sha256") != handoff_item.get("receipt_sha256")
                        or item.get("file_sha256") != handoff_item.get("file_sha256")
                        or item.get("predecessor_receipt_sha256") != predecessor):
                    raise FinalizationError("provisional stage chain does not bind handoff")
                receipt = _json_object(root / f"manifests/provisional/stage-receipts/{item['stage']}.json", label="provisional receipt")
                unsigned = dict(receipt); claimed = unsigned.pop("receipt_sha256", None)
                if (claimed != _digest(unsigned) or claimed != item.get("receipt_sha256")
                        or receipt.get("stage") != item.get("stage")
                        or receipt.get("predecessor_receipt_sha256") != predecessor
                        or item.get("predecessor_receipt_sha256") != predecessor
                        or hashlib.sha256(_canonical(receipt)).hexdigest() != item.get("file_sha256")
                        or not isinstance(receipt.get("runtime_identity"), Mapping)):
                    raise FinalizationError("provisional receipt chain is invalid")
                if identity is None: identity = receipt["runtime_identity"]
                elif identity != receipt["runtime_identity"]: raise FinalizationError("provisional receipt identity diverged")
                if receipt["runtime_identity"] != handoff.get("runtime_identity"):
                    raise FinalizationError("provisional receipt identity does not bind handoff")
                predecessor = claimed
            observed_first = next(
                (
                    item.get("receipt_sha256")
                    for item in listed
                    if isinstance(item, Mapping) and item.get("stage") == "first-100-gate"
                ),
                None,
            )
            expected_first = handoff.get("first_100_receipt_sha256")
            if (manifest.get("terminal_chain_head") != predecessor
                    or predecessor != handoff.get("predecessor_receipt_sha256")
                    or manifest.get("first_100_receipt_sha256") != observed_first
                    or observed_first != expected_first):
                raise FinalizationError("provisional terminal chain does not bind handoff")
            task6 = _json_object(root / files[1], label="provisional Task-6 validation")
            references = _json_object(root / files[2], label="provisional Hub references")
            if (hashlib.sha256(_canonical(task6)).hexdigest() != manifest.get("task6_validation_sha256")
                    or hashlib.sha256(_canonical(references)).hexdigest() != manifest.get("hub_artifact_references_sha256")):
                raise FinalizationError("provisional validation hashes are invalid")
            outcome = handoff.get("terminal_outcome")
            common_task6 = {
                "schema_version", "kind", "terminal_outcome", "result",
            }
            if (task6.get("schema_version") != 1
                    or task6.get("kind") != "lehome_simple_curriculum_task6_validation_v1"
                    or task6.get("terminal_outcome") != outcome):
                raise FinalizationError("provisional Task-6 identity is invalid")
            if (set(references) != {"schema_version", "kind", "fresh", "success_replay"}
                    or references.get("schema_version") != 1
                    or references.get("kind") != "lehome_simple_curriculum_hub_artifact_references_v1"):
                raise FinalizationError("provisional Hub references are malformed")
            fresh, replay = references.get("fresh"), references.get("success_replay")
            if not isinstance(fresh, list) or not isinstance(replay, list):
                raise FinalizationError("provisional Hub references are malformed")

            identities: set[str] = set()
            prefixes: set[str] = set()
            reference_fields = {
                "attempt_id", "repository", "prefix", "immutable_revision",
                "episode_sha256", "local_sync_receipt_sha256",
            }
            for source, source_round, values in (
                ("fresh", str(handoff["round_id"]), fresh),
                ("success_replay", f"{handoff['round_id']}-replay", replay),
            ):
                for reference in values:
                    if not isinstance(reference, Mapping) or set(reference) != reference_fields:
                        raise FinalizationError("provisional Hub reference is invalid")
                    attempt_id = reference.get("attempt_id")
                    expected_prefix = f"rollout-rounds/{source_round}/{attempt_id}"
                    if (not isinstance(attempt_id, str)
                            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", attempt_id) is None
                            or attempt_id in identities or reference.get("prefix") in prefixes
                            or reference.get("repository") != self.repository
                            or reference.get("prefix") != expected_prefix
                            or not _hex(reference.get("immutable_revision"), 40)
                            or not _hex(reference.get("episode_sha256"), 64)
                            or not _hex(reference.get("local_sync_receipt_sha256"), 64)):
                        raise FinalizationError(f"provisional {source} Hub reference is invalid")
                    identities.add(attempt_id); prefixes.add(expected_prefix)

            if outcome == "complete":
                expected_counts = {"fresh_valid_outcomes": 1000, "replay_attempts": 400, "replay_accepted_successes": 200,
                                   "replay_accepted_by_category": {"pant_long": 50, "pant_short": 50, "top_long": 50, "top_short": 50}}
                complete_fields = common_task6 | {
                    "fresh_valid_outcomes", "fresh_official_successes", "replay_attempts",
                    "replay_accepted_successes", "replay_accepted_by_category",
                }
                if (set(task6) != complete_fields or task6.get("result") != "complete"
                        or type(task6.get("fresh_official_successes")) is not int
                        or not 0 <= task6["fresh_official_successes"] <= 1000
                        or any(task6.get(key) != value for key, value in expected_counts.items())):
                    raise FinalizationError("provisional Task-6 caps are invalid")
                if len(fresh) != 1000 or len(replay) != 200:
                    raise FinalizationError("provisional Hub-reference closure is invalid")
            else:
                noncomplete_fields = common_task6 | {
                    "fresh_reference_count", "replay_accepted_reference_count",
                }
                if (set(task6) != noncomplete_fields or task6.get("result") != "not_complete"
                        or type(task6.get("fresh_reference_count")) is not int
                        or type(task6.get("replay_accepted_reference_count")) is not int
                        or task6["fresh_reference_count"] < 0 or task6["replay_accepted_reference_count"] < 0
                        or task6.get("fresh_reference_count") != len(fresh)
                        or task6.get("replay_accepted_reference_count") != len(replay)):
                    raise FinalizationError("provisional non-complete evidence is invalid")
        except FinalizationError:
            raise
        except Exception as error:
            raise FinalizationError("provisional immutable readback failed") from error

    def _stage_and_validate_remote_receipt(
        self, *, module: object, transport: object, root: Path, prefix: str,
        revision: str, token: str, handoff: Mapping[str, object], seal: Mapping[str, object],
        evidence_files: tuple[str, ...],
    ) -> tuple[object, ...]:
        final_name = "reports/final-publication.json"
        try:
            transport.download_files(
                repository=self.repository, revision=revision, destination=root,
                relative_paths=(final_name,), token=token, remote_prefix=prefix,
            )
        except Exception as error:  # the finalizer owns a normalized public boundary
            raise FinalizationError("final publication receipt staging failed") from error
        receipt = _json_object(root / final_name, label="final publication receipt")
        expected_evidence_revision = receipt.get("evidence_revision")
        if not _hex(expected_evidence_revision, 40):
            raise FinalizationError("final publication receipt does not bind immutable evidence")
        # Validate the self-hash, schema, ID binding, seal binding, and
        # booleans before a descriptor collector is allowed to consume it.
        validate_finalization_receipt(
            receipt, handoff=handoff, evidence_revision=expected_evidence_revision,
            evidence_bundle_sha256=None, seal=seal,
        )
        try:
            entries = module._collect_entries(module.CollectionPublicationBundle(
                root=root, run_id=handoff["run_id"], repository=self.repository,
                revision="main", files=evidence_files + (final_name,),
            ))
        except Exception as error:
            raise FinalizationError("final publication evidence is unsafe") from error
        evidence_entries = tuple(entry for entry in entries if entry.relative_path in evidence_files)
        if len(evidence_entries) != len(evidence_files):
            raise FinalizationError("final publication evidence manifest is incomplete")
        validate_finalization_receipt(
            receipt, handoff=handoff, evidence_revision=expected_evidence_revision,
            evidence_bundle_sha256=_entry_digest(evidence_entries), seal=seal,
        )
        try:
            at_evidence_revision = module._tree_files(
                transport.list_tree(
                    repository=self.repository, revision=expected_evidence_revision,
                    token=token, remote_prefix=prefix,
                ),
                prefix=prefix,
            )
            if at_evidence_revision != set(evidence_files):
                raise FinalizationError("final publication evidence revision is not exact")
            evidence_bundle = module.CollectionPublicationBundle(
                root=root, run_id=handoff["run_id"], repository=self.repository,
                revision="main", files=evidence_files,
            )
            for auth in (token, None):
                module._verify_download(
                    transport=transport, bundle=evidence_bundle,
                    revision=expected_evidence_revision, prefix=prefix,
                    entries=evidence_entries, token=auth,
                )
            full_bundle = module.CollectionPublicationBundle(
                root=root, run_id=handoff["run_id"], repository=self.repository,
                revision="main", files=evidence_files + (final_name,),
            )
            for auth in (token, None):
                module._verify_download(
                    transport=transport, bundle=full_bundle, revision=revision,
                    prefix=prefix, entries=entries, token=auth,
                )
        except FinalizationError:
            raise
        except Exception as error:
            raise FinalizationError("final publication immutable readback failed") from error
        return tuple(entries)

    def publish(self, root: Path, *, handoff: Mapping[str, object], stop_observation: Mapping[str, object], seal: Mapping[str, object]) -> Mapping[str, object]:
        module = self._module
        if module is None:
            import importlib.util
            source = Path(__file__).with_name("publish_simple_curriculum_collection.py")
            spec = importlib.util.spec_from_file_location("simple_curriculum_publisher", source)
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
        _load_operator_token(self.token_path)
        try:
            token = module._load_token(self.token_path)
        except Exception as error:
            raise FinalizationError("operator HF token file is invalid") from error
        self._verify_provisional(module=module, transport=self._transport or module.HuggingFacePublicDatasetTransport(), root=root, token=token, handoff=handoff)
        _atomic_json(root / "reports" / "operator-stop-handoff.json", handoff)
        _atomic_json(root / "reports" / "stopped-observation.json", stop_observation)
        _atomic_json(root / "seals" / "final-seal.json", seal)
        files = ("reports/operator-stop-handoff.json", "reports/stopped-observation.json", "seals/final-seal.json")
        bundle = module.CollectionPublicationBundle(root=root, run_id=handoff["run_id"], repository=self.repository, revision="main", files=files)
        transport = self._transport or module.HuggingFacePublicDatasetTransport()
        # Provisional evidence occupies its own immutable subtree.  Promotion
        # gets a sibling subtree so a known-good provisional bundle can never
        # be overwritten or mistaken for the final marker.
        prefix = f"collection-rounds/{handoff['run_id']}/manifests/promotion" if handoff.get("schema_version") == 2 else f"collection-rounds/{handoff['run_id']}"
        head = transport.resolve_approved_ref(repository=self.repository, ref="main", token=token)
        existing = module._tree_files(transport.list_tree(repository=self.repository, revision=head, token=token, remote_prefix=prefix), prefix=prefix)
        final_name = "reports/final-publication.json"
        evidence_names = set(files)
        if final_name in existing:
            if set(existing) != evidence_names | {final_name}:
                raise FinalizationError("immutable finalization prefix collision")
            self._stage_and_validate_remote_receipt(
                module=module, transport=transport, root=root, prefix=prefix,
                revision=head, token=token, handoff=handoff, seal=seal,
                evidence_files=files,
            )
            return {"immutable_revision": head, "readback_verified": True, "public_readback_verified": True}
        if set(existing) - evidence_names:
            raise FinalizationError("immutable finalization prefix collision")
        evidence = module.publish_collection_bundle(bundle, token=token, transport=transport, remote_prefix=prefix)
        receipt_body = {"schema_version": 1, "kind": "lehome_simple_curriculum_operator_finalization_receipt_v1", "run_id": handoff["run_id"], "round_id": handoff["round_id"], "evidence_revision": evidence.immutable_revision, "evidence_bundle_sha256": evidence.bundle_sha256, "final_seal_sha256": seal["seal_sha256"], "readback_verified": True, "public_readback_verified": True}
        receipt = {**receipt_body, "receipt_sha256": _digest(receipt_body)}
        validate_finalization_receipt(
            receipt, handoff=handoff, evidence_revision=evidence.immutable_revision,
            evidence_bundle_sha256=evidence.bundle_sha256, seal=seal,
        )
        _atomic_json(root / "reports" / "final-publication.json", receipt)
        entry = module._collect_entries(module.CollectionPublicationBundle(root=root, run_id=handoff["run_id"], repository=self.repository, revision="main", files=("reports/final-publication.json",)))[0]
        head = transport.resolve_approved_ref(repository=self.repository, ref="main", token=token)
        present = module._tree_files(transport.list_tree(repository=self.repository, revision=head, token=token, remote_prefix=evidence.remote_prefix), prefix=evidence.remote_prefix)
        if final_name in present:
            if set(present) != evidence_names | {final_name}:
                raise FinalizationError("immutable finalization prefix collision")
            head = transport.resolve_approved_ref(repository=self.repository, ref="main", token=token)
            self._stage_and_validate_remote_receipt(
                module=module, transport=transport, root=root, prefix=evidence.remote_prefix,
                revision=head, token=token, handoff=handoff, seal=seal,
                evidence_files=files,
            )
            return {"immutable_revision": head, "readback_verified": True, "public_readback_verified": True}
        if set(present) != evidence_names:
            raise FinalizationError("immutable finalization prefix collision")
        try:
            revision = transport.upload_files(
                repository=self.repository, revision="main", source=root,
                entries=(entry,), token=token, remote_prefix=evidence.remote_prefix,
                parent_commit=head,
            )
        except Exception as _upload_error:
            # The server may have committed the one immutable receipt before
            # the response was lost.  Reconcile that exact tree now; never
            # issue a second mutable upload from this call.
            try:
                reconciled_head = transport.resolve_approved_ref(
                    repository=self.repository, ref="main", token=token,
                )
                reconciled = module._tree_files(
                    transport.list_tree(
                        repository=self.repository, revision=reconciled_head,
                        token=token, remote_prefix=evidence.remote_prefix,
                    ),
                    prefix=evidence.remote_prefix,
                )
                if set(reconciled) != evidence_names | {final_name}:
                    raise FinalizationError("ambiguous final receipt upload did not produce the exact tree")
                self._stage_and_validate_remote_receipt(
                    module=module, transport=transport, root=root,
                    prefix=evidence.remote_prefix, revision=reconciled_head,
                    token=token, handoff=handoff, seal=seal,
                    evidence_files=files,
                )
                return {
                    "immutable_revision": reconciled_head,
                    "readback_verified": True,
                    "public_readback_verified": True,
                }
            except FinalizationError:
                raise
            except Exception as reconcile_error:
                raise FinalizationError("ambiguous final receipt upload reconciliation failed") from reconcile_error
        all_entries = tuple(evidence.entries) + (entry,)
        for auth in (token, None): module._verify_download(transport=transport, bundle=bundle, revision=revision, prefix=evidence.remote_prefix, entries=all_entries, token=auth)
        return {"immutable_revision": revision, "readback_verified": True, "public_readback_verified": True}


def finalize_operator_handoff(handoff: Mapping[str, object], *, provider: Provider, publisher: Publisher, staging_parent: Path, stop_timeout_seconds: float) -> dict[str, object]:
    """Stop in a safety boundary, then publish only if all prior checks pass."""
    validation_error: Exception | None = None
    valid: dict[str, object] | None = None
    try:
        valid = validate_handoff(handoff)
    except Exception as error:  # stop still happens below
        validation_error = error
    try:
        stopped = stop_exact_instance(provider, timeout_seconds=stop_timeout_seconds)
    except Exception as error:
        raise FinalizationError("infrastructure_stop_failure") from error
    if validation_error is not None:
        raise FinalizationError("infrastructure_stop_failure") from validation_error
    assert valid is not None
    seal = _seal(valid, stopped)
    with tempfile.TemporaryDirectory(prefix="lehome-finalizer-", dir=staging_parent) as temporary:
        receipt = publisher.publish(Path(temporary), handoff=valid, stop_observation=stopped, seal=seal)
    if receipt.get("readback_verified") is not True or receipt.get("public_readback_verified") is not True:
        raise FinalizationError("publication readback failed after STOPPED")
    return {"result": "finalized", "stop_observation": stopped, "seal": seal, "publication": dict(receipt)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-target"); parser.add_argument("--ssh-port", type=int)
    parser.add_argument("--remote-campaign-root"); parser.add_argument("--run-id"); parser.add_argument("--round-id"); parser.add_argument("--hf-token-file", type=Path)
    parser.add_argument("--stop-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--emergency-stop-only", action="store_true")
    args = parser.parse_args(argv)
    provider = SubprocessNebiusProvider()
    if args.emergency_stop_only:
        try:
            stop_exact_instance(provider, timeout_seconds=args.stop_timeout_seconds)
        except (FinalizationError, OSError, subprocess.SubprocessError, ValueError) as error:
            print(json.dumps({"result": "infrastructure_stop_failure", "error": str(error)}, sort_keys=True))
            return 2
        print(json.dumps({"result": "emergency_stopped"}, sort_keys=True))
        return 0
    try:
        if any(value is None for value in (args.ssh_target, args.ssh_port, args.remote_campaign_root, args.run_id, args.round_id, args.hf_token_file)):
            raise FinalizationError("normal finalization requires complete operator metadata")
        durable_handoff = _durable_handoff_path(args.run_id)
        if durable_handoff.exists() or durable_handoff.is_symlink():
            handoff = _read_durable_handoff(durable_handoff)
        else:
            handoff = fetch_remote_handoff(
                ssh_target=args.ssh_target, port=args.ssh_port,
                campaign_root=args.remote_campaign_root, destination=durable_handoff,
            )
        if handoff.get("run_id") != args.run_id or handoff.get("round_id") != args.round_id:
            raise FinalizationError("remote handoff does not match explicit invocation IDs")
        result = finalize_operator_handoff(
            handoff, provider=provider, publisher=HfFinalizerPublisher(args.hf_token_file),
            staging_parent=durable_handoff.parent, stop_timeout_seconds=args.stop_timeout_seconds,
        )
        print(json.dumps({"result": result["result"], "immutable_revision": result["publication"]["immutable_revision"]}, sort_keys=True))
        return 0
    except (FinalizationError, OSError, subprocess.SubprocessError, ValueError) as error:
        # Handoff fetch/validation is inside the safety boundary too.  The
        # remote process may be dead or its disk evidence corrupt, but it is
        # never a reason to leave this exact billed VM running.
        try:
            stop_exact_instance(provider, timeout_seconds=args.stop_timeout_seconds)
        except FinalizationError:
            pass
        print(json.dumps({"result": "infrastructure_stop_failure", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__": raise SystemExit(main())
