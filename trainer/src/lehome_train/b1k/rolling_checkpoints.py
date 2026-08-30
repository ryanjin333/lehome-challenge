"""Fail-closed, compatibility-bound rolling checkpoints over an object backend."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
from typing import Mapping, Protocol
from uuid import uuid4

from lehome_train.b1k.training import SUPPORTED_GPU_COUNTS, approved_launch_plans
from lehome_train.io import atomic_write_json


_RUN = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")


class ResumePolicy(str, Enum):
    AUTO = "auto"
    NEVER = "never"
    REQUIRE = "require"


def _canonical(value: object) -> bytes:
    def finite(item: object) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("checkpoint JSON must be finite")
        if isinstance(item, dict):
            if not all(type(key) is str for key in item): raise ValueError("checkpoint JSON keys must be strings")
            for child in item.values(): finite(child)
        elif isinstance(item, (list, tuple)):
            for child in item: finite(child)
    finite(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _file_identity(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256(); size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block); digest.update(block)
    return size, digest.hexdigest()


def _fallback_temporary(*, directory: Path | None = None, suffix: str = "") -> Path:
    descriptor, name = tempfile.mkstemp(dir=directory, prefix="b1k-checkpoint-", suffix=suffix)
    os.close(descriptor)
    return Path(name)


def _json_object(raw: bytes) -> dict[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result: raise ValueError("duplicate checkpoint JSON key")
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("malformed checkpoint JSON") from error
    if not isinstance(value, dict): raise ValueError("checkpoint JSON must be an object")
    return value


@dataclass(frozen=True, slots=True)
class CheckpointCompatibility:
    materialized_dataset_fingerprint: str
    modality_sha256: str
    stats_sha256: str
    groot_revision: str
    base_model_revision: str
    cosmos_revision: str
    container_digest: str
    cycle_id: str
    world_size: int
    plan_identity: str
    physical_batch_size: int
    global_batch_size: int
    gradient_accumulation_steps: int
    effective_global_batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    launch_argv_sha256: str

    def __post_init__(self) -> None:
        for value in (self.materialized_dataset_fingerprint, self.modality_sha256, self.stats_sha256, self.launch_argv_sha256):
            if type(value) is not str or not _SHA256.fullmatch(value): raise ValueError("compatibility SHA-256 is invalid")
        for value in (self.groot_revision, self.base_model_revision, self.cosmos_revision):
            if type(value) is not str or not _REVISION.fullmatch(value): raise ValueError("compatibility revision is invalid")
        if type(self.container_digest) is not str or not re.fullmatch(r"sha256:[0-9a-f]{64}", self.container_digest): raise ValueError("container digest is invalid")
        if type(self.cycle_id) is not str or not _RUN.fullmatch(self.cycle_id): raise ValueError("compatibility cycle ID is invalid")
        if type(self.world_size) is not int or self.world_size not in SUPPORTED_GPU_COUNTS: raise ValueError("world size is invalid")
        plan = next((candidate for candidate in approved_launch_plans(num_gpus=self.world_size) if candidate.identity == self.plan_identity), None)
        if plan is None or (self.physical_batch_size, self.global_batch_size, self.gradient_accumulation_steps, self.effective_global_batch_size) != (plan.physical_batch_size, plan.global_batch_size, plan.gradient_accumulation_steps, plan.effective_global_batch_size):
            raise ValueError("compatibility launch plan is invalid")
        if self.effective_global_batch_size != self.physical_batch_size * self.world_size * self.gradient_accumulation_steps: raise ValueError("compatibility batch arithmetic is invalid")
        if (self.learning_rate, self.weight_decay, self.warmup_ratio) != (plan.learning_rate, plan.weight_decay, plan.warmup_ratio): raise ValueError("compatibility optimizer is invalid")

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: object) -> "CheckpointCompatibility":
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__): raise ValueError("checkpoint compatibility schema is invalid")
        return cls(**value)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class CheckpointDescriptor:
    schema_version: int
    run_id: str
    step: int
    artifact_sha256: str
    artifact_byte_size: int
    compatibility: CheckpointCompatibility
    descriptor_sha256: str

    @classmethod
    def create(cls, *, run_id: str, step: int, artifact: bytes | Path, compatibility: CheckpointCompatibility) -> "CheckpointDescriptor":
        size, digest = _file_identity(artifact) if isinstance(artifact, Path) else (len(artifact), hashlib.sha256(artifact).hexdigest())
        raw = {"schema_version": 1, "run_id": run_id, "step": step, "artifact_sha256": digest, "artifact_byte_size": size, "compatibility": compatibility.to_dict()}
        return cls(schema_version=1, run_id=run_id, step=step, artifact_sha256=digest, artifact_byte_size=size, compatibility=compatibility, descriptor_sha256=hashlib.sha256(_canonical(raw)).hexdigest())

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": self.schema_version, "run_id": self.run_id, "step": self.step, "artifact_sha256": self.artifact_sha256, "artifact_byte_size": self.artifact_byte_size, "compatibility": self.compatibility.to_dict(), "descriptor_sha256": self.descriptor_sha256}

    def validate(self) -> None:
        if self.schema_version != 1 or not _RUN.fullmatch(self.run_id) or type(self.step) is not int or self.step not in range(1000, 15001, 1000) or type(self.artifact_byte_size) is not int or self.artifact_byte_size <= 0 or not _SHA256.fullmatch(self.artifact_sha256) or not _SHA256.fullmatch(self.descriptor_sha256): raise ValueError("checkpoint descriptor is invalid")
        raw = self.to_dict(); observed = raw.pop("descriptor_sha256")
        if observed != hashlib.sha256(_canonical(raw)).hexdigest(): raise ValueError("checkpoint descriptor self-hash is invalid")

    @classmethod
    def from_dict(cls, value: object) -> "CheckpointDescriptor":
        fields = {"schema_version", "run_id", "step", "artifact_sha256", "artifact_byte_size", "compatibility", "descriptor_sha256"}
        if not isinstance(value, dict) or set(value) != fields: raise ValueError("checkpoint descriptor schema is invalid")
        descriptor = cls(schema_version=value["schema_version"], run_id=value["run_id"], step=value["step"], artifact_sha256=value["artifact_sha256"], artifact_byte_size=value["artifact_byte_size"], compatibility=CheckpointCompatibility.from_dict(value["compatibility"]), descriptor_sha256=value["descriptor_sha256"])
        descriptor.validate(); return descriptor

    @classmethod
    def from_bytes(cls, raw: bytes) -> "CheckpointDescriptor":
        return cls.from_dict(_json_object(raw))


def _required_members(root: Path, *, step: int, world_size: int) -> None:
    if root.is_symlink() or not root.is_dir() or any(path.is_symlink() or (not path.is_dir() and not path.is_file()) for path in root.rglob("*")):
        raise ValueError("checkpoint contains unsafe filesystem entries")
    required = {"trainer_state.json", "optimizer.pt", "scheduler.pt", "config.json"}
    names = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if not required <= names: raise ValueError("checkpoint lacks native resumability payload")
    for name in required:
        if (root / name).stat().st_size == 0: raise ValueError("checkpoint payload is empty")
    for name in ("trainer_state.json", "config.json"):
        if not isinstance(_json_object((root / name).read_bytes()), dict): raise ValueError("checkpoint JSON is invalid")
    state = _json_object((root / "trainer_state.json").read_bytes())
    if state.get("global_step") != step: raise ValueError("checkpoint trainer state step mismatch")
    direct_names = {name for name in ("model.safetensors", "pytorch_model.bin") if name in names}
    direct_model = bool(direct_names)
    index_model = "model.safetensors.index.json" in names
    if direct_model == index_model or len(direct_names) > 1: raise ValueError("checkpoint model payload is invalid")
    if direct_model and any((root / name).stat().st_size == 0 for name in direct_names): raise ValueError("checkpoint model payload is invalid")
    if index_model:
        index = _json_object((root / "model.safetensors.index.json").read_bytes()); weights = index.get("weight_map")
        if not isinstance(weights, dict) or not weights or not all(type(name) is str and type(shard) is str and shard in names and (root / shard).stat().st_size > 0 for name, shard in weights.items()): raise ValueError("checkpoint model index is invalid")
    rng = {f"rng_state_{rank}.pth" for rank in range(world_size)} if world_size > 1 else {"rng_state.pth"}
    observed_rng = {name for name in names if re.fullmatch(r"rng_state(?:_[0-9]+)?\.pth", name)}
    if observed_rng != rng or any((root / name).stat().st_size == 0 for name in rng): raise ValueError("checkpoint RNG payload is invalid")


def validate_native_checkpoint(root: Path, *, step: int, world_size: int) -> None:
    """Public shared validator for package, restore, and finalization."""
    _required_members(root, step=step, world_size=world_size)


def package_checkpoint(source: Path, destination: Path, *, step: int, world_size: int = 1) -> Path:
    root_name = f"checkpoint-{step}"
    if not source.is_dir() or source.name != root_name or destination.exists() or source.is_symlink(): raise ValueError("checkpoint package paths are invalid")
    _required_members(source, step=step, world_size=world_size)
    temporary = destination.with_name(f".{destination.name}.incomplete")
    if temporary.exists() or temporary.is_symlink(): raise ValueError("checkpoint package temporary path exists")
    try:
        with tarfile.open(temporary, "w") as archive:
            for path in sorted(source.rglob("*")):
                if path.is_symlink() or not path.is_file(): raise ValueError("checkpoint package contains unsafe entry")
                info = archive.gettarinfo(str(path), arcname=(Path(root_name) / path.relative_to(source)).as_posix()); info.uid = info.gid = 0; info.uname = info.gname = ""; info.mtime = 0
                with path.open("rb") as handle: archive.addfile(info, handle)
        descriptor = os.open(temporary, os.O_RDONLY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def safe_restore_tar(artifact: Path, output_dir: Path, *, step: int, world_size: int = 1) -> Path:
    target = output_dir / f"checkpoint-{step}"
    if target.exists() or target.is_symlink(): raise ValueError("existing restore destination")
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.is_symlink(): raise ValueError("unsafe restore destination")
    root = f"checkpoint-{step}/"
    with tarfile.open(artifact, "r") as archive:
        members = archive.getmembers(); seen: set[str] = set()
        for member in members:
            if member.name in seen or member.name.startswith("/") or not member.name.startswith(root) or ".." in PurePosixPath(member.name).parts or not member.isfile(): raise ValueError("unsafe checkpoint archive")
            seen.add(member.name)
        staging = Path(tempfile.mkdtemp(dir=output_dir, prefix=".restore-"))
        try:
            archive.extractall(staging, members=members)
            restored = staging / root.rstrip("/")
            _required_members(restored, step=step, world_size=world_size)
            restored.replace(target)
        finally: shutil.rmtree(staging, ignore_errors=True)
    return target


class BucketBackend(Protocol):
    def write_bytes(self, path: str, value: bytes) -> None: ...
    def read_bytes(self, path: str) -> bytes: ...
    def write_json(self, path: str, value: Mapping[str, object]) -> None: ...
    def read_json(self, path: str) -> dict[str, object]: ...
    def copy(self, source: str, destination: str) -> None: ...
    def list(self, prefix: str) -> tuple[str, ...]: ...
    def delete(self, paths: tuple[str, ...]) -> None: ...
    def upload_file(self, source: Path, destination: str) -> None: ...
    def download_file(self, source: str, destination: Path) -> None: ...
    def temporary_file(self, suffix: str = "") -> Path: ...


class HelperBucketBackend:
    def __init__(self, client: object, bucket_id: str, transport_root: Path = Path("/workspace/checkpoints/.transport")) -> None: self.client, self.bucket_id, self.transport_root = client, bucket_id, transport_root
    def _temporary(self) -> Path:
        self.transport_root.mkdir(parents=True, exist_ok=True); fd, name = tempfile.mkstemp(dir=self.transport_root, prefix="bucket-"); os.close(fd); return Path(name)
    def temporary_file(self, suffix: str = "") -> Path:
        temporary = self._temporary()
        target = temporary.with_name(temporary.name + suffix)
        temporary.replace(target)
        return target
    def write_bytes(self, path: str, value: bytes) -> None:
        local = self._temporary()
        try: local.write_bytes(value); self.client.request("upload", {"bucket_id": self.bucket_id, "local_path": str(local), "remote_path": path})
        finally: local.unlink(missing_ok=True)
    def read_bytes(self, path: str) -> bytes:
        local = self._temporary()
        try: self.client.request("download", {"bucket_id": self.bucket_id, "remote_path": path, "local_path": str(local)}); return local.read_bytes()
        finally: local.unlink(missing_ok=True)
    def write_json(self, path: str, value: Mapping[str, object]) -> None: self.write_bytes(path, _canonical(dict(value)))
    def read_json(self, path: str) -> dict[str, object]: return _json_object(self.read_bytes(path))
    def copy(self, source: str, destination: str) -> None: self.client.request("copy", {"bucket_id": self.bucket_id, "source": source, "destination": destination})
    def list(self, prefix: str) -> tuple[str, ...]: return tuple(item["path"] for item in self.client.request("list", {"bucket_id": self.bucket_id, "prefix": prefix})["files"])
    def delete(self, paths: tuple[str, ...]) -> None: self.client.request("delete", {"bucket_id": self.bucket_id, "paths": list(paths)})
    def upload_file(self, source: Path, destination: str) -> None:
        if source.is_symlink() or not source.is_file(): raise ValueError("unsafe artifact source")
        self.client.request("upload", {"bucket_id": self.bucket_id, "local_path": str(source), "remote_path": destination})
    def download_file(self, source: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.client.request("download", {"bucket_id": self.bucket_id, "remote_path": source, "local_path": str(destination)})


@dataclass(frozen=True, slots=True)
class _Verified:
    descriptor: CheckpointDescriptor
    artifact_path: str
    descriptor_path: str


class RollingCheckpointStore:
    def __init__(self, *, backend: BucketBackend, run_id: str, compatibility: CheckpointCompatibility) -> None:
        if not _RUN.fullmatch(run_id): raise ValueError("checkpoint run ID is invalid")
        self.backend, self.run_id, self.compatibility = backend, run_id, compatibility
    def _download_identity(self, remote: str) -> tuple[int, str]:
        local = self.backend.temporary_file(".tar") if hasattr(self.backend, "temporary_file") else _fallback_temporary(suffix=".tar")
        try:
            self.backend.download_file(remote, local)
            return _file_identity(local)
        finally: local.unlink(missing_ok=True)
    def _validated(self, *, compatibility: CheckpointCompatibility | None = None) -> tuple[_Verified, ...]:
        # Read every artifact before retention: a corrupt higher checkpoint must
        # never cause deletion of a compatible older one. Xet hash binding may
        # optimize this later, but cannot weaken this safety property.
        groups: dict[tuple[int, str], set[str]] = {}
        prefix = f"verified/{self.run_id}/"
        for path in self.backend.list(prefix):
            parts = path.split("/")
            if len(parts) != 5 or parts[:2] != ["verified", self.run_id] or not re.fullmatch(r"step-(?:[1-9][0-9]*)", parts[2]) or not _SHA256.fullmatch(parts[3]) or parts[4] not in {"artifact.tar", "descriptor.json"}: raise ValueError("corrupt verified checkpoint path")
            groups.setdefault((int(parts[2][5:]), parts[3]), set()).add(parts[4])
        output: list[_Verified] = []
        by_step: set[int] = set()
        for (step, digest), members in groups.items():
            if members != {"artifact.tar", "descriptor.json"} or step in by_step: raise ValueError("ambiguous or incomplete verified checkpoint")
            by_step.add(step); root = f"verified/{self.run_id}/step-{step}/{digest}/"; descriptor_path = root + "descriptor.json"; artifact_path = root + "artifact.tar"
            descriptor = CheckpointDescriptor.from_bytes(self.backend.read_bytes(descriptor_path))
            size, observed = self._download_identity(artifact_path)
            if descriptor.run_id != self.run_id or descriptor.step != step or descriptor.artifact_sha256 != digest or (compatibility is not None and descriptor.compatibility != compatibility):
                raise ValueError("incompatible or corrupt verified checkpoint")
            if descriptor.artifact_byte_size != size or observed != digest:
                raise ValueError("checkpoint artifact identity is corrupt")
            output.append(_Verified(descriptor, artifact_path, descriptor_path))
        return tuple(sorted(output, key=lambda item: item.descriptor.step))
    def _namespace_nonempty(self) -> bool:
        return any(self.backend.list(prefix) for prefix in (f"runs/{self.run_id}/", f"verified/{self.run_id}/", f"staging/{self.run_id}/"))
    def _validate_latest(self, verified: tuple[_Verified, ...]) -> None:
        path = f"runs/{self.run_id}/latest.json"
        if not self.backend.list(path):
            if verified: raise ValueError("malformed latest checkpoint")
            return
        latest = _json_object(self.backend.read_bytes(path))
        if set(latest) != {"cycle_id", "step", "prefix", "descriptor_sha256"} or type(latest["step"]) is not int or type(latest["prefix"]) is not str or type(latest["descriptor_sha256"]) is not str:
            raise ValueError("malformed latest checkpoint")
        if latest["cycle_id"] != self.compatibility.cycle_id:
            raise ValueError("incompatible checkpoint cycle")
        newest = verified[-1] if verified else None
        match = newest if newest is not None and newest.descriptor.step == latest["step"] and newest.descriptor_path.rsplit("/", 1)[0] == latest["prefix"] and newest.descriptor.descriptor_sha256 == latest["descriptor_sha256"] else None
        if match is None: raise ValueError("malformed latest checkpoint")
    def publish(self, artifact: Path, descriptor: CheckpointDescriptor) -> None:
        size, digest = _file_identity(artifact) if artifact.is_file() and not artifact.is_symlink() else (0, "")
        descriptor.validate()
        if descriptor.run_id != self.run_id or descriptor.compatibility != self.compatibility or descriptor.artifact_byte_size != size or descriptor.artifact_sha256 != digest: raise ValueError("checkpoint descriptor is incompatible with artifact")
        stage = f"staging/{self.run_id}/{descriptor.step}/{digest}"; verified = f"verified/{self.run_id}/step-{descriptor.step}/{digest}"; descriptor_bytes = _canonical(descriptor.to_dict())
        try:
            self.backend.upload_file(artifact, stage + "/artifact.tar"); self.backend.write_bytes(stage + "/descriptor.json", descriptor_bytes)
            if self._download_identity(stage + "/artifact.tar") != (size, digest) or self.backend.read_bytes(stage + "/descriptor.json") != descriptor_bytes: raise ValueError("checkpoint stage readback mismatch")
            self.backend.copy(stage + "/artifact.tar", verified + "/artifact.tar"); self.backend.copy(stage + "/descriptor.json", verified + "/descriptor.json")
            if self._download_identity(verified + "/artifact.tar") != (size, digest) or self.backend.read_bytes(verified + "/descriptor.json") != descriptor_bytes: raise ValueError("checkpoint promote readback mismatch")
            latest = {"cycle_id": descriptor.compatibility.cycle_id, "step": descriptor.step, "prefix": verified, "descriptor_sha256": descriptor.descriptor_sha256}; self.backend.write_json(f"runs/{self.run_id}/latest.json", latest)
            if self.backend.read_json(f"runs/{self.run_id}/latest.json") != latest: raise ValueError("checkpoint latest readback mismatch")
            self.backend.delete((stage + "/artifact.tar", stage + "/descriptor.json"))
            if self.backend.list(stage + "/"): raise ValueError("checkpoint staging delete was incomplete")
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as error: raise ValueError("checkpoint publication failed") from error
        self._retain_two()
    def verified_steps(self) -> tuple[int, ...]: return tuple(item.descriptor.step for item in self._validated(compatibility=self.compatibility))
    def _retain_two(self) -> None:
        verified = self._validated(compatibility=self.compatibility)
        expected_steps = tuple(item.descriptor.step for item in verified[-2:])
        for item in verified[:-2]:
            prefix = item.artifact_path.rsplit("/", 1)[0] + "/"
            self.backend.delete((item.artifact_path, item.descriptor_path))
            if self.backend.list(prefix): raise ValueError("checkpoint retention delete was incomplete")
        remaining = self._validated(compatibility=self.compatibility)
        if len(remaining) > 2 or tuple(item.descriptor.step for item in remaining) != expected_steps: raise ValueError("checkpoint retention postcondition failed")

    def ensure_newest_two(self) -> tuple[int, ...]:
        """Repair retention before a resumed run or immutable finalization."""

        self._retain_two()
        verified = self._validated(compatibility=self.compatibility)
        self._validate_latest(verified)
        if len(verified) > 2:
            raise ValueError("checkpoint retention postcondition failed")
        return tuple(item.descriptor.step for item in verified)

    def _local_checkpoint_matches(self, target: Path, descriptor: CheckpointDescriptor) -> bool:
        if target.is_symlink() or not target.is_dir():
            return False
        temporary = _fallback_temporary(directory=target.parent, suffix=".tar")
        temporary.unlink(missing_ok=True)
        try:
            package_checkpoint(target, temporary, step=descriptor.step, world_size=self.compatibility.world_size)
            return _file_identity(temporary) == (descriptor.artifact_byte_size, descriptor.artifact_sha256)
        except (OSError, ValueError, tarfile.TarError):
            return False
        finally:
            temporary.unlink(missing_ok=True)

    def _quarantine_local_checkpoint(self, target: Path) -> Path:
        quarantine = target.with_name(f".{target.name}.unverified-{uuid4().hex}")
        if quarantine.exists() or quarantine.is_symlink():
            raise ValueError("checkpoint quarantine destination exists")
        os.replace(target, quarantine)
        return quarantine

    def _quarantine_unselected_local_checkpoints(self, destination: Path, *, selected_name: str) -> None:
        """Remove every competing checkpoint name before upstream auto-resume scans it."""

        if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
            raise ValueError("existing restore destination")
        destination.mkdir(parents=True, exist_ok=True)
        quarantine_root = destination / ".unverified-checkpoints"
        if quarantine_root.is_symlink() or (quarantine_root.exists() and not quarantine_root.is_dir()):
            raise ValueError("checkpoint quarantine directory is unsafe")
        quarantine_root.mkdir(mode=0o700, exist_ok=True)
        if quarantine_root.is_symlink() or not quarantine_root.is_dir() or quarantine_root.stat().st_mode & 0o077:
            raise ValueError("checkpoint quarantine directory is unsafe")
        candidates = tuple(path for path in destination.iterdir() if path.name.startswith("checkpoint-") and path.name != selected_name)
        if not candidates:
            return
        for candidate in candidates:
            quarantine = quarantine_root / f"{candidate.name}-{uuid4().hex}"
            if quarantine.exists() or quarantine.is_symlink():
                raise ValueError("checkpoint quarantine destination exists")
            os.replace(candidate, quarantine)

    @staticmethod
    def _assert_only_selected_local_checkpoint(destination: Path, selected_name: str) -> None:
        remaining = tuple(path.name for path in destination.iterdir() if path.name.startswith("checkpoint-"))
        if remaining != (selected_name,):
            raise ValueError("resume output contains unselected checkpoints")
    def inspect_resume_compatibility(self, policy: ResumePolicy) -> CheckpointCompatibility | None:
        """Validate remote state without preselecting a launch-plan identity.

        The production adapter uses this once to map a retained checkpoint to
        exactly one approved plan.  It never treats an unrecognized nonempty
        namespace or mixed-plan retained history as an empty checkpoint set.
        """

        if policy is ResumePolicy.NEVER:
            if self._namespace_nonempty(): raise ValueError("resume policy never refuses nonempty remote state")
            return None
        staging = self.backend.list(f"staging/{self.run_id}/")
        if staging: raise ValueError("partial staging checkpoint prevents resume")
        verified = self._validated(); self._validate_latest(verified)
        if not verified:
            if self._namespace_nonempty():
                raise ValueError("nonempty remote checkpoint namespace lacks verified state")
            if policy is ResumePolicy.REQUIRE: raise ValueError("resume policy require found no checkpoint")
            return None
        compatibilities = {item.descriptor.compatibility for item in verified}
        if len(compatibilities) != 1:
            raise ValueError("ambiguous verified checkpoint compatibility")
        return verified[-1].descriptor.compatibility

    def resume(self, policy: ResumePolicy, destination: Path) -> Path | None:
        compatibility = self.inspect_resume_compatibility(policy)
        if compatibility is None:
            return None
        if compatibility != self.compatibility:
            raise ValueError("incompatible verified checkpoint")
        verified = self._validated(compatibility=self.compatibility)
        self._validate_latest(verified)
        self.ensure_newest_two()
        verified = self._validated(compatibility=self.compatibility)
        self._validate_latest(verified)
        newest = verified[-1]
        if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
            raise ValueError("existing restore destination")
        destination.parent.mkdir(parents=True, exist_ok=True); local = self.backend.temporary_file(".tar") if hasattr(self.backend, "temporary_file") else _fallback_temporary(directory=destination.parent, suffix=".tar")
        try:
            self.backend.download_file(newest.artifact_path, local)
            if _file_identity(local) != (newest.descriptor.artifact_byte_size, newest.descriptor.artifact_sha256): raise ValueError("checkpoint artifact changed after validation")
            target = destination / f"checkpoint-{newest.descriptor.step}"
            self._quarantine_unselected_local_checkpoints(destination, selected_name=target.name)
            if target.exists() or target.is_symlink():
                if self._local_checkpoint_matches(target, newest.descriptor):
                    self._assert_only_selected_local_checkpoint(destination, target.name)
                    return target
                quarantine = self._quarantine_local_checkpoint(target)
                try:
                    restored = safe_restore_tar(local, destination, step=newest.descriptor.step, world_size=self.compatibility.world_size)
                except BaseException:
                    if not target.exists() and not target.is_symlink():
                        os.replace(quarantine, target)
                    raise
                if quarantine.is_symlink() or quarantine.is_file():
                    quarantine.unlink(missing_ok=True)
                else:
                    shutil.rmtree(quarantine, ignore_errors=True)
                self._assert_only_selected_local_checkpoint(destination, target.name)
                return restored
            restored = safe_restore_tar(local, destination, step=newest.descriptor.step, world_size=self.compatibility.world_size)
            self._assert_only_selected_local_checkpoint(destination, target.name)
            return restored
        finally: local.unlink(missing_ok=True)


@dataclass(slots=True)
class LocalCheckpointPublisher:
    """Package and publish one watcher-confirmed native checkpoint atomically.

    The long-running watcher calls this only after observing an unchanged,
    complete native directory twice.  The transient tar is deleted after the
    store's remote stage/promote/readback succeeds; the receipt remains local
    evidence for finalization and interruption recovery.
    """

    store: RollingCheckpointStore
    checkpoint_root: Path
    receipts_root: Path

    def publish(self, step: int) -> Path:
        if type(step) is not int or step not in range(1_000, 15_001, 1_000):
            raise ValueError("checkpoint publish step is invalid")
        source = self.checkpoint_root / f"checkpoint-{step}"
        validate_native_checkpoint(source, step=step, world_size=self.store.compatibility.world_size)
        if self.receipts_root.is_symlink():
            raise ValueError("checkpoint receipt directory is unsafe")
        self.receipts_root.mkdir(parents=True, exist_ok=True)
        receipt = self.receipts_root / f"step-{step}.json"
        if receipt.exists() or receipt.is_symlink():
            raise ValueError("checkpoint publication receipt already exists")
        descriptor, temporary_name = tempfile.mkstemp(dir=self.receipts_root, prefix=f".step-{step}-", suffix=".tar")
        os.close(descriptor)
        artifact = Path(temporary_name)
        artifact.unlink()
        try:
            package_checkpoint(source, artifact, step=step, world_size=self.store.compatibility.world_size)
            checkpoint = CheckpointDescriptor.create(
                run_id=self.store.run_id,
                step=step,
                artifact=artifact,
                compatibility=self.store.compatibility,
            )
            self.store.publish(artifact, checkpoint)
            atomic_write_json(
                receipt,
                {
                    "descriptor_sha256": checkpoint.descriptor_sha256,
                    "cycle_id": self.store.compatibility.cycle_id,
                    "run_id": self.store.run_id,
                    "step": step,
                    "verified_steps": list(self.store.verified_steps()),
                },
            )
            return receipt
        finally:
            artifact.unlink(missing_ok=True)
