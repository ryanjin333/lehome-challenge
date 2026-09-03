"""Fail-closed, offline contract for the public LeHome GR00T N1.5 recipe.

This module validates identities and renders commands.  It deliberately owns
no cloud lifecycle and never executes training.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import subprocess
import tempfile
from types import MappingProxyType
from typing import Mapping
import zipfile
from base64 import urlsafe_b64encode


_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_TRAINING_CONTAINER_IMAGE_ID = (
    "sha256:bec2b688ca03145dd20c010aa32b761a386e3fed57bdc45c3df5d86f9afa15c7"
)
_TRAINING_CONTAINER_PYTHON = "/opt/lehome-challenge/.venv/bin/python"
_TRAINING_CONTAINER_PYTHONPATH = (
    "/flash/site-packages:"
    "/deps/peft-0.18.1-py3-none-any.whl"
)
_TRAINING_CONTAINER_LEROBOT_ROOT = "/flash/site-packages/lerobot"


class ReproductionError(RuntimeError):
    """An immutable identity or artifact gate failed."""


@dataclass(frozen=True)
class ReproductionContract:
    source_repository: str
    source_revision: str
    source_tree: str
    dependency_lock_sha256: str
    lerobot_wheel_sha256: str
    lerobot_package_file_count: int
    lerobot_package_tree_sha256: str
    base_model_metadata_count: int
    base_model_metadata_sha256: str
    dataset_metadata_count: int
    dataset_metadata_sha256: str
    base_model_repository: str
    base_model_revision: str
    dataset_repository: str
    dataset_revision: str
    trusted_source_files: Mapping[str, str]
    vm_id: str
    disk_id: str
    python_version: str
    lerobot_version: str
    training_command: tuple[str, ...]
    training: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "trusted_source_files",
            MappingProxyType(dict(self.trusted_source_files)),
        )
        object.__setattr__(self, "training", MappingProxyType(dict(self.training)))


CONTRACT = ReproductionContract(
    source_repository="theo-zhou/lehome-groot-submission-4",
    source_revision="d384fe00508acd96ab1c3c5dc265e08261f94b3b",
    source_tree="8bb4ff37d03762f8c4bc4bce5783e7d811991a3e",
    dependency_lock_sha256="d0e6e3cb472cea3d04b0bc2d79b9d929bf498a392d5c155fa635f413fa092313",
    lerobot_wheel_sha256="b08c1c15b2356bd4e658122deabfb9dacd2d7447de4a4327720991723d4edf2c",
    lerobot_package_file_count=289,
    lerobot_package_tree_sha256="db3b4e18b166d4bb7fb4354cec82a7fbd15bb24230f9d71269a017c774e0852f",
    base_model_metadata_count=13,
    base_model_metadata_sha256="b49d2e9f419064cbe31fcc877263f5a1af4ca1ec10acd723b3c325dc0d6fc70d",
    dataset_metadata_count=67,
    dataset_metadata_sha256="152e3b0e3da178fba9d29ddb1858df95a4c20fe8118aa36b57bde71b0ee25b9a",
    base_model_repository="nvidia/GR00T-N1.5-3B",
    base_model_revision="869830fc749c35f34771aa5209f923ac57e4564e",
    dataset_repository="lehome/dataset_challenge_merged",
    dataset_revision="17e8dee8fac294ffd21d250501d3b31bf8679042",
    trusted_source_files={
        "configs/train_groot.yaml": "eb0c82d4a9960a072e454389d82a618d81a79b789c2f19b1733dba4c629e9f75",
        "shs/train/train_groot.sh": "2a49d25a1bbde7a54e6027fcbd490cb0334132b0f628eccad69413e19a1481b5",
        "scripts/utils/evaluation.py": "9a9d9e28008405ead892fdf1d115cd83f3d2be7d806381dbc92486d2e6d966a7",
        "shs/harvest/harvest_groot_until_success_00.sh": "3ac3aefefe7eea057d3df6d336a958552d276efb8dad365557a20dccc211b034",
    },
    vm_id="computeinstance-u00t6xfqhadrcmssa2",
    disk_id="computedisk-u00pbe55crxy7jr56x",
    python_version="3.11",
    lerobot_version="0.4.3",
    training_command=(
        "lerobot-train",
        "--config_path=configs/train_groot.yaml",
        "--wandb.mode=offline",
    ),
    training={
        "batch_size": 64,
        "steps": 12000,
        "optimizer_lr": 2e-4,
        "optimizer_beta1": 0.95,
        "optimizer_beta2": 0.999,
        "optimizer_eps": 1e-8,
        "optimizer_weight_decay": 1e-5,
        "warmup_ratio": 0.05,
        "num_decay_steps": 12000,
        "decay_lr_ratio": 0.1,
        "use_bf16": True,
        "tune_llm": False,
        "tune_visual": False,
        "tune_projector": True,
        "tune_diffusion_model": True,
        "image_transforms": False,
        "state_normalization": "mean_std",
        "action_normalization": "mean_std",
        "policy_image_size": 224,
        "state_dimension": 12,
        "action_dimension": 12,
        "save_freq": 1500,
        "log_freq": 500,
    },
)


@dataclass(frozen=True)
class VerifiedInputs:
    checkout: Path
    base_model_root: Path
    hub_cache_root: Path
    dataset_root: Path
    source_receipt: Path
    resolved_snapshots_receipt: Path
    source_receipt_sha256: str
    resolved_snapshots_receipt_sha256: str
    source_tree: str
    base_model_metadata_sha256: str
    dataset_metadata_sha256: str


def _canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError):
        raise ReproductionError("receipt is not canonical strict JSON") from None


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def _validated_hub_siblings(siblings: object) -> list[dict[str, object]]:
    if not isinstance(siblings, list) or not siblings:
        raise ReproductionError("Hub sibling metadata is empty or invalid")
    validated: list[dict[str, object]] = []
    paths: set[str] = set()
    for candidate in siblings:
        if not isinstance(candidate, dict) or set(candidate) != {
            "path",
            "blob_id",
            "size",
            "lfs_sha256",
        }:
            raise ReproductionError("Hub sibling metadata schema is invalid")
        path = candidate["path"]
        blob_id = candidate["blob_id"]
        size = candidate["size"]
        lfs_sha256 = candidate["lfs_sha256"]
        if not isinstance(path, str):
            raise ReproductionError("Hub sibling metadata path is invalid")
        pure = PurePosixPath(path)
        if (
            pure.is_absolute()
            or not pure.parts
            or any(part in {"", ".", ".."} for part in pure.parts)
            or path in paths
            or not isinstance(blob_id, str)
            or _REVISION.fullmatch(blob_id) is None
            or type(size) is not int
            or size < 0
            or (lfs_sha256 is not None and (
                not isinstance(lfs_sha256, str) or _SHA256.fullmatch(lfs_sha256) is None
            ))
        ):
            raise ReproductionError("Hub sibling metadata identity is invalid")
        paths.add(path)
        validated.append(dict(candidate))
    return sorted(validated, key=lambda value: str(value["path"]))


def hub_metadata_sha256(siblings: object) -> str:
    """Hash authoritative Hub sibling metadata using the frozen canonical form."""

    rows = _validated_hub_siblings(siblings)
    payload = "".join(
        f"{row['path']}\t{row['blob_id']}\t{row['size']}\t{row['lfs_sha256'] or ''}\n"
        for row in rows
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _tree_identity(files: Mapping[str, bytes]) -> tuple[int, str]:
    payload = b"".join(
        relative.encode("utf-8")
        + b"\0"
        + _sha256_bytes(content).encode("ascii")
        + b"\n"
        for relative, content in sorted(files.items())
    )
    return len(files), _sha256_bytes(payload)


def wheel_lerobot_tree_identity(wheel: bytes) -> tuple[int, str]:
    """Return the canonical source-tree identity embedded under ``lerobot/``."""

    try:
        with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
            files: dict[str, bytes] = {}
            for info in archive.infolist():
                if info.is_dir() or not info.filename.startswith("lerobot/"):
                    continue
                relative = info.filename.removeprefix("lerobot/")
                pure = PurePosixPath(relative)
                file_type = (info.external_attr >> 16) & 0o170000
                if (
                    not relative
                    or pure.is_absolute()
                    or any(part in {"", ".", ".."} for part in pure.parts)
                    or relative in files
                    or file_type == stat.S_IFLNK
                ):
                    raise ReproductionError("LeRobot wheel package entry is unsafe")
                files[relative] = archive.read(info)
    except (OSError, zipfile.BadZipFile, KeyError):
        raise ReproductionError("LeRobot wheel is invalid") from None
    if not files:
        raise ReproductionError("LeRobot wheel contains no package tree")
    return _tree_identity(files)


_COMPATIBILITY_WHEEL_KIND = "lehome_lerobot_043_groot_scheduler_compatibility_v1"
_COMPATIBILITY_WHEEL_FIELDS = {
    "num_decay_steps": 10000,
    "decay_lr_ratio": 0.1,
}
_COMPATIBILITY_WHEEL_CONFIG = "lerobot/policies/groot/configuration_groot.py"
_COMPATIBILITY_WHEEL_SOURCE_FRAGMENT = """    warmup_ratio: float = 0.05
    use_bf16: bool = True
"""
_COMPATIBILITY_WHEEL_DERIVED_FRAGMENT = """    warmup_ratio: float = 0.05
    num_decay_steps: int = 10000
    decay_lr_ratio: float = 0.1
    use_bf16: bool = True
"""
_COMPATIBILITY_WHEEL_SCHEDULER_SOURCE_FRAGMENT = """            num_warmup_steps=int(10000 * self.warmup_ratio),  # 5% warmup by default
            num_decay_steps=10000,  # Adjust based on training steps
            peak_lr=self.optimizer_lr,
            decay_lr=self.optimizer_lr * 0.1,
"""
_COMPATIBILITY_WHEEL_SCHEDULER_DERIVED_FRAGMENT = """            num_warmup_steps=int(self.num_decay_steps * self.warmup_ratio),  # 5% warmup by default
            num_decay_steps=self.num_decay_steps,  # Adjust based on training steps
            peak_lr=self.optimizer_lr,
            decay_lr=self.optimizer_lr * self.decay_lr_ratio,
"""


def _compatibility_transformation() -> dict[str, object]:
    return {
        "kind": _COMPATIBILITY_WHEEL_KIND,
        "fields": dict(_COMPATIBILITY_WHEEL_FIELDS),
    }


def resolve_groot_scheduler_from_yaml(text: str) -> dict[str, object]:
    """Resolve exactly the four scheduler values from the pinned train YAML.

    The source YAML is already byte-verified before this helper is used.  This
    intentionally accepts only the scalar form used by the pinned public
    recipe; it is not a general YAML interpreter.
    """
    if not isinstance(text, str):
        raise ReproductionError("pinned training YAML is invalid")

    def scalar(name: str) -> str:
        matches = re.findall(
            rf"^[ \t]*{re.escape(name)}:[ \t]*([^#\r\n]+?)[ \t]*(?:#[^\r\n]*)?$",
            text,
            re.MULTILINE,
        )
        if len(matches) != 1:
            raise ReproductionError(f"pinned training YAML lacks one {name} value")
        return matches[0].strip()

    try:
        steps = int(scalar("steps"), 10)
        optimizer_lr = Decimal(scalar("optimizer_lr"))
        warmup_ratio = Decimal(scalar("warmup_ratio"))
        num_decay_steps = int(scalar("num_decay_steps"), 10)
        decay_lr_ratio = Decimal(scalar("decay_lr_ratio"))
    except (ValueError, InvalidOperation):
        raise ReproductionError("pinned training YAML scheduler values are invalid") from None
    if (
        steps != 12000
        or optimizer_lr != Decimal("2e-4")
        or warmup_ratio != Decimal("0.05")
        or num_decay_steps != 12000
        or decay_lr_ratio != Decimal("0.1")
    ):
        raise ReproductionError("pinned training YAML scheduler values do not match the public recipe")
    return {
        "num_warmup_steps": int(num_decay_steps * warmup_ratio),
        "num_decay_steps": num_decay_steps,
        "peak_lr": float(optimizer_lr),
        "decay_lr": float(optimizer_lr * decay_lr_ratio),
    }


def _safe_wheel_entries(wheel: bytes) -> dict[str, bytes]:
    """Read a wheel as a complete safe immutable archive map."""
    try:
        with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
            entries: dict[str, bytes] = {}
            for info in archive.infolist():
                name = info.filename
                pure = PurePosixPath(name)
                file_type = (info.external_attr >> 16) & 0o170000
                if (
                    info.is_dir()
                    or not name
                    or pure.is_absolute()
                    or any(part in {"", ".", ".."} for part in pure.parts)
                    or name in entries
                    or file_type == stat.S_IFLNK
                ):
                    raise ReproductionError("LeRobot wheel contains unsafe entries")
                entries[name] = archive.read(info)
    except (OSError, zipfile.BadZipFile, KeyError):
        raise ReproductionError("LeRobot wheel is invalid") from None
    if not entries:
        raise ReproductionError("LeRobot wheel contains no entries")
    return entries


def _wheel_metadata_identity(entries: Mapping[str, bytes]) -> tuple[str, str, str]:
    metadata_paths = sorted(
        path for path in entries if path.endswith(".dist-info/METADATA")
    )
    if len(metadata_paths) != 1:
        raise ReproductionError("LeRobot wheel distribution metadata is invalid")
    metadata_path = metadata_paths[0]
    try:
        metadata = entries[metadata_path].decode("utf-8")
    except UnicodeError:
        raise ReproductionError("LeRobot wheel distribution metadata is invalid") from None
    name = re.search(r"^Name: ([^\r\n]+)$", metadata, re.MULTILINE)
    version = re.search(r"^Version: ([^\r\n]+)$", metadata, re.MULTILINE)
    if name is None or version is None:
        raise ReproductionError("LeRobot wheel distribution metadata is invalid")
    dist_info = metadata_path.removesuffix("/METADATA")
    return dist_info, name.group(1), version.group(1)


def _derived_wheel_bytes(entries: Mapping[str, bytes]) -> bytes:
    """Apply the only accepted public N1.5 compatibility transformation."""
    mutable = dict(entries)
    source = mutable.get(_COMPATIBILITY_WHEEL_CONFIG)
    if source is None:
        raise ReproductionError("LeRobot wheel lacks GrootConfig source")
    try:
        text = source.decode("utf-8")
    except UnicodeError:
        raise ReproductionError("LeRobot GrootConfig source is invalid") from None
    if text.count(_COMPATIBILITY_WHEEL_SOURCE_FRAGMENT) != 1 or text.count(
        _COMPATIBILITY_WHEEL_SCHEDULER_SOURCE_FRAGMENT
    ) != 1:
        raise ReproductionError("LeRobot GrootConfig source does not match the audited 0.4.3 form")
    text = text.replace(
        _COMPATIBILITY_WHEEL_SOURCE_FRAGMENT,
        _COMPATIBILITY_WHEEL_DERIVED_FRAGMENT,
        1,
    ).replace(
        _COMPATIBILITY_WHEEL_SCHEDULER_SOURCE_FRAGMENT,
        _COMPATIBILITY_WHEEL_SCHEDULER_DERIVED_FRAGMENT,
        1,
    )
    mutable[_COMPATIBILITY_WHEEL_CONFIG] = text.encode("utf-8")
    dist_info, distribution, version = _wheel_metadata_identity(mutable)
    if distribution != "lerobot" or version != CONTRACT.lerobot_version:
        raise ReproductionError("LeRobot wheel distribution identity mismatch")
    record = f"{dist_info}/RECORD"
    if record not in mutable:
        raise ReproductionError("LeRobot wheel RECORD is unavailable")
    record_rows = []
    for name, payload in sorted(mutable.items()):
        if name == record:
            continue
        digest = urlsafe_b64encode(hashlib.sha256(payload).digest()).decode("ascii").rstrip("=")
        record_rows.append(f"{name},sha256={digest},{len(payload)}\n")
    mutable[record] = ("".join(record_rows) + f"{record},,\n").encode("utf-8")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in sorted(mutable.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, payload)
    return output.getvalue()


def _write_atomic_bytes(path: Path | str, payload: bytes, label: str) -> Path:
    destination = Path(path)
    if not destination.is_absolute() or destination.exists() or destination.is_symlink():
        raise ReproductionError(f"{label} path is unavailable or unsafe")
    parent = _regular_directory(destination.parent, f"{label} parent")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o444)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise ReproductionError(f"{label} already exists") from None
        parent_descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return destination.resolve(strict=True)


def _expected_compatible_lerobot_wheel(
    upstream_bytes: bytes, expected_upstream_sha256: str
) -> tuple[bytes, dict[str, object]]:
    """Regenerate the only admissible derived wheel from the pinned upstream bytes."""
    if _SHA256.fullmatch(expected_upstream_sha256) is None:
        raise ReproductionError("trusted upstream LeRobot wheel digest is invalid")
    if _sha256_bytes(upstream_bytes) != expected_upstream_sha256:
        raise ReproductionError("upstream LeRobot wheel digest mismatch")
    entries = _safe_wheel_entries(upstream_bytes)
    _, distribution, version = _wheel_metadata_identity(entries)
    if distribution != "lerobot" or version != CONTRACT.lerobot_version:
        raise ReproductionError("upstream LeRobot wheel distribution identity mismatch")
    upstream_count, upstream_tree = wheel_lerobot_tree_identity(upstream_bytes)
    derived_bytes = _derived_wheel_bytes(entries)
    derived_count, derived_tree = wheel_lerobot_tree_identity(derived_bytes)
    changed = {
        _COMPATIBILITY_WHEEL_CONFIG: {
            "before_sha256": _sha256_bytes(entries[_COMPATIBILITY_WHEEL_CONFIG]),
            "after_sha256": _sha256_bytes(
                _safe_wheel_entries(derived_bytes)[_COMPATIBILITY_WHEEL_CONFIG]
            ),
        }
    }
    return derived_bytes, {
        "schema_version": 1,
        "kind": "lehome_public_n15_lerobot_compatibility_wheel_v1",
        "upstream_wheel_sha256": expected_upstream_sha256,
        "upstream_package_file_count": upstream_count,
        "upstream_package_tree_sha256": upstream_tree,
        "transformation": _compatibility_transformation(),
        "changed_package_files": changed,
        "derived_wheel_sha256": _sha256_bytes(derived_bytes),
        "derived_package_file_count": derived_count,
        "derived_package_tree_sha256": derived_tree,
        "distribution": {"name": distribution, "version": version},
    }


def build_compatible_lerobot_wheel(
    *,
    upstream_wheel: Path | str,
    output_wheel: Path | str,
    receipt_output: Path | str,
    expected_upstream_sha256: str = CONTRACT.lerobot_wheel_sha256,
) -> dict[str, object]:
    """Build and seal the deterministic N1.5-compatible LeRobot 0.4.3 wheel."""
    upstream = _regular_file(Path(upstream_wheel), "upstream LeRobot wheel")
    derived_bytes, value = _expected_compatible_lerobot_wheel(
        upstream.read_bytes(), expected_upstream_sha256
    )
    output = _write_atomic_bytes(output_wheel, derived_bytes, "compatible LeRobot wheel")
    try:
        receipt = write_receipt(
            output=receipt_output,
            value=value,
            label="compatible LeRobot wheel receipt",
        )
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return {**value, "wheel_path": str(output), "receipt_path": receipt["path"]}


def compatibility_wheel_identity(
    *,
    wheel: Path | str,
    receipt: Path | str,
    upstream_wheel: Path | str,
    expected_upstream_sha256: str = CONTRACT.lerobot_wheel_sha256,
) -> dict[str, object]:
    """Fail closed unless a compatible installed artifact is exactly sealed."""
    wheel_path = _regular_file(Path(wheel), "compatible LeRobot wheel")
    upstream_path = _regular_file(Path(upstream_wheel), "upstream LeRobot wheel")
    _, _, value = _load_receipt(receipt, "compatible LeRobot wheel receipt")
    expected_bytes, expected = _expected_compatible_lerobot_wheel(
        upstream_path.read_bytes(), expected_upstream_sha256
    )
    if value != expected:
        raise ReproductionError("compatibility wheel receipt identity mismatch")
    wheel_bytes = wheel_path.read_bytes()
    if wheel_bytes != expected_bytes:
        raise ReproductionError("compatibility wheel differs from expected derived wheel")
    return dict(value)


def _installed_package_tree_identity(root: Path) -> tuple[int, str]:
    files: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        if path.is_symlink():
            raise ReproductionError(f"installed LeRobot tree contains unsafe symlink: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ReproductionError(f"installed LeRobot tree contains unsafe entry: {relative}")
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        files[relative] = path.read_bytes()
    if not files:
        raise ReproductionError("installed LeRobot package tree is empty")
    return _tree_identity(files)


def _regular_file(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ReproductionError(f"{label} path is unsafe")
    try:
        metadata = path.lstat()
    except OSError:
        raise ReproductionError(f"{label} is unavailable or unsafe") from None
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise ReproductionError(f"{label} is unavailable or unsafe")
    return path.resolve(strict=True)


def _regular_directory(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ReproductionError(f"{label} path is unsafe")
    try:
        metadata = path.lstat()
    except OSError:
        raise ReproductionError(f"{label} is unavailable or unsafe") from None
    if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise ReproductionError(f"{label} is unavailable or unsafe")
    return path.resolve(strict=True)


def _load_receipt(path: Path | str, label: str) -> tuple[Path, bytes, dict[str, object]]:
    safe = _regular_file(Path(path), label)
    try:
        payload = safe.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ReproductionError(f"{label} is invalid JSON") from None
    if not isinstance(value, dict) or payload != _canonical_bytes(value):
        raise ReproductionError(f"{label} is not canonical JSON")
    return safe, payload, value


def resolve_dataset_blobs_mount(
    *,
    resolved_snapshots_receipt: Path | str,
    expected_dataset_root: Path | str,
    protected_root: Path | str = "/mnt/lehome",
    contract: ReproductionContract = CONTRACT,
) -> Path:
    """Return the canonical Hub blobs directory needed by staged dataset links."""

    _, _, value = _load_receipt(resolved_snapshots_receipt, "resolved snapshots receipt")
    dataset = value.get("dataset")
    if not isinstance(dataset, dict):
        raise ReproductionError("resolved snapshots receipt lacks the dataset identity")
    if (
        dataset.get("repository") != contract.dataset_repository
        or dataset.get("revision") != contract.dataset_revision
    ):
        raise ReproductionError("resolved dataset identity mismatch")
    raw_root = Path(str(dataset.get("root", "")))
    raw_snapshot = Path(str(dataset.get("snapshot_root", "")))
    root = _regular_directory(raw_root, "staged dataset root")
    snapshot = _regular_directory(raw_snapshot, "dataset Hub snapshot root")
    if raw_root != root or raw_snapshot != snapshot:
        raise ReproductionError("resolved dataset paths must be canonical")
    expected = _regular_directory(Path(expected_dataset_root), "expected staged dataset root")
    if root != expected:
        raise ReproductionError("resolved dataset root mismatch")
    expected_repository = "datasets--" + contract.dataset_repository.replace("/", "--")
    if (
        snapshot.name != contract.dataset_revision
        or snapshot.parent.name != "snapshots"
        or snapshot.parent.parent.name != expected_repository
    ):
        raise ReproductionError("resolved dataset snapshot path mismatch")
    blobs = _regular_directory(snapshot.parent.parent / "blobs", "dataset Hub blobs root")
    protected = _regular_directory(Path(protected_root), "protected workspace root")
    try:
        relative = blobs.relative_to(protected)
    except ValueError:
        raise ReproductionError("dataset Hub blobs root escapes the protected workspace") from None
    if not relative.parts:
        raise ReproductionError("dataset Hub blobs root is not a strict protected child")
    text = str(blobs)
    if any(character in text for character in "\r\n,"):
        raise ReproductionError("dataset Hub blobs root is unsafe for a container bind")
    return blobs


def _safe_relative_file(root: Path, relative: str, label: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ReproductionError(f"{label} path is unsafe")
    path = root.joinpath(*pure.parts)
    safe = _regular_file(path, label)
    try:
        safe.relative_to(root)
    except ValueError:
        raise ReproductionError(f"{label} path is unsafe") from None
    return safe


def _validate_source(
    checkout: Path | str,
    receipt_path: Path | str,
    contract: ReproductionContract,
) -> tuple[Path, Path, str]:
    root = _regular_directory(Path(checkout), "source checkout")
    receipt, receipt_bytes, value = _load_receipt(receipt_path, "source receipt")
    expected_keys = {"schema_version", "kind", "repository", "revision", "tree", "files"}
    if set(value) != expected_keys:
        raise ReproductionError("source receipt schema mismatch")
    if (
        value["schema_version"] != 1
        or value["kind"] != "lehome_public_n15_source_v1"
        or value["repository"] != contract.source_repository
        or value["revision"] != contract.source_revision
        or value["tree"] != contract.source_tree
        or _REVISION.fullmatch(str(value["revision"])) is None
        or _REVISION.fullmatch(str(value["tree"])) is None
    ):
        raise ReproductionError("source receipt identity mismatch")
    files = value["files"]
    expected_files = dict(contract.trusted_source_files)
    if not isinstance(files, dict) or files != expected_files:
        raise ReproductionError("source receipt file digest mismatch")
    for relative, expected_digest in expected_files.items():
        if _SHA256.fullmatch(expected_digest) is None:
            raise ReproductionError("trusted source digest is invalid")
        path = _safe_relative_file(root, relative, f"source file {relative}")
        if _sha256_file(path) != expected_digest:
            raise ReproductionError(f"source file digest mismatch: {relative}")
    dependency_lock = _safe_relative_file(root, "uv.lock", "upstream dependency lock")
    if _sha256_file(dependency_lock) != contract.dependency_lock_sha256:
        raise ReproductionError("upstream dependency lock digest mismatch")
    try:
        dependency_lock_text = dependency_lock.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        raise ReproductionError("upstream dependency lock is unreadable") from None
    if f"sha256:{contract.lerobot_wheel_sha256}" not in dependency_lock_text:
        raise ReproductionError("upstream dependency lock lacks the trusted LeRobot wheel")
    git_directory = _regular_directory(root / ".git", "source Git metadata")
    del git_directory

    def git(*arguments: str) -> str:
        environment = os.environ.copy()
        environment["GIT_OPTIONAL_LOCKS"] = "0"
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            raise ReproductionError(f"source Git proof failed: {result.stderr.strip()}")
        return result.stdout.strip()

    head = git("rev-parse", "HEAD")
    if head != contract.source_revision or head != value["revision"]:
        raise ReproductionError("source Git HEAD mismatch")
    tree = git("rev-parse", "HEAD^{tree}")
    if tree != contract.source_tree or tree != value["tree"]:
        raise ReproductionError("source Git tree mismatch")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ReproductionError("source Git checkout is modified")
    return root, receipt, _sha256_bytes(receipt_bytes)


def _verify_hub_snapshot(
    root: Path,
    siblings: object,
    *,
    label: str,
    blobs_root: Path,
    expected_count: int,
    expected_metadata_sha256: str,
) -> list[dict[str, object]]:
    rows = _validated_hub_siblings(siblings)
    if len(rows) != expected_count or hub_metadata_sha256(rows) != expected_metadata_sha256:
        raise ReproductionError(f"{label} authoritative Hub metadata mismatch")
    blobs = _regular_directory(blobs_root, f"{label} repository blobs directory")
    actual_files: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        if path.is_symlink():
            link = Path(os.readlink(path))
            if link.is_absolute():
                raise ReproductionError(f"{label} content contains unsafe symlink: {relative}")
            direct_target = path.parent / link
            try:
                target_metadata = direct_target.lstat()
            except OSError:
                raise ReproductionError(
                    f"{label} content contains dangling symlink: {relative}"
                ) from None
            if stat.S_ISLNK(target_metadata.st_mode) or not stat.S_ISREG(target_metadata.st_mode):
                raise ReproductionError(
                    f"{label} content contains chained or non-file symlink: {relative}"
                )
            resolved_target = direct_target.resolve(strict=True)
            try:
                target_relative = resolved_target.relative_to(blobs)
            except ValueError:
                raise ReproductionError(
                    f"{label} content symlink escapes repository blobs: {relative}"
                ) from None
            cursor = blobs
            for part in target_relative.parts[:-1]:
                cursor = cursor / part
                if cursor.is_symlink() or not cursor.is_dir():
                    raise ReproductionError(
                        f"{label} content contains chained symlink: {relative}"
                    )
            actual_files[relative] = resolved_target
            continue
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ReproductionError(f"{label} content contains unsafe entry: {relative}")
        actual_files[relative] = path.resolve(strict=True)
    expected_paths = {str(row["path"]) for row in rows}
    if set(actual_files) != expected_paths:
        raise ReproductionError(f"{label} content file set mismatches authoritative metadata")
    for row in rows:
        relative = str(row["path"])
        path = actual_files[relative]
        payload = path.read_bytes()
        if len(payload) != row["size"]:
            raise ReproductionError(f"{label} content size mismatch: {relative}")
        lfs_sha256 = row["lfs_sha256"]
        if lfs_sha256 is None:
            if _git_blob_sha1(payload) != row["blob_id"]:
                raise ReproductionError(f"{label} Git blob mismatch: {relative}")
        elif _sha256_bytes(payload) != lfs_sha256:
            raise ReproductionError(f"{label} LFS content mismatch: {relative}")
    return rows


def _validate_snapshots(
    receipt_path: Path | str,
    vm_id: str,
    disk_id: str,
    checkout: Path,
    contract: ReproductionContract,
) -> tuple[Path, Path, Path, Path, str, str, str]:
    if vm_id != contract.vm_id or disk_id != contract.disk_id:
        raise ReproductionError("VM or disk is not accepted by the reproduction contract")
    receipt, receipt_bytes, value = _load_receipt(
        receipt_path,
        "resolved snapshots receipt",
    )
    if set(value) != {"schema_version", "kind", "base_model", "dataset", "vm_id", "disk_id"}:
        raise ReproductionError("resolved snapshots receipt schema mismatch")
    if (
        value["schema_version"] != 1
        or value["kind"] != "lehome_public_n15_resolved_snapshots_v1"
        or value["vm_id"] != vm_id
        or value["disk_id"] != disk_id
    ):
        raise ReproductionError("resolved snapshots receipt identity mismatch")

    def snapshot(
        candidate: object,
        *,
        label: str,
        repository: str,
        revision: str,
    ) -> Path:
        expected = {"repository", "revision", "root", "siblings"}
        if label == "dataset":
            expected.add("snapshot_root")
        if not isinstance(candidate, dict) or set(candidate) != expected:
            raise ReproductionError(f"{label} snapshot receipt mismatch")
        if candidate["repository"] != repository or candidate["revision"] != revision:
            raise ReproductionError(f"{label} snapshot identity mismatch")
        if _REVISION.fullmatch(str(candidate["revision"])) is None or not isinstance(candidate["root"], str):
            raise ReproductionError(f"{label} snapshot identity mismatch")
        return _regular_directory(Path(candidate["root"]), f"{label} snapshot root")

    model = snapshot(
        value["base_model"],
        label="base model",
        repository=contract.base_model_repository,
        revision=contract.base_model_revision,
    )
    dataset = snapshot(
        value["dataset"],
        label="dataset",
        repository=contract.dataset_repository,
        revision=contract.dataset_revision,
    )
    expected_model_directory = "models--" + contract.base_model_repository.replace("/", "--")
    if (
        model.name != contract.base_model_revision
        or model.parent.name != "snapshots"
        or model.parent.parent.name != expected_model_directory
    ):
        raise ReproductionError("base model staged path mismatch")
    hub_cache = _regular_directory(model.parents[2], "base model Hub cache")
    model_ref = _regular_file(model.parent.parent / "refs/main", "base model main ref")
    try:
        ref_revision = model_ref.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        raise ReproductionError("base model main ref is unreadable") from None
    if ref_revision != contract.base_model_revision:
        raise ReproductionError("base model main ref mismatch")
    model_receipt = value["base_model"]
    assert isinstance(model_receipt, dict)
    model_siblings = _verify_hub_snapshot(
        model,
        model_receipt["siblings"],
        label="base model",
        blobs_root=model.parent.parent / "blobs",
        expected_count=contract.base_model_metadata_count,
        expected_metadata_sha256=contract.base_model_metadata_sha256,
    )
    model_paths = {str(row["path"]) for row in model_siblings}
    if "config.json" not in model_paths or not any(
        relative.endswith(".safetensors") for relative in model_paths
    ):
        raise ReproductionError("base model content manifest lacks required model files")
    expected_dataset = checkout / "Datasets/example/four_types_merged"
    if dataset != expected_dataset:
        raise ReproductionError("dataset staged path mismatch")
    dataset_receipt = value["dataset"]
    assert isinstance(dataset_receipt, dict)
    snapshot_root = _regular_directory(
        Path(str(dataset_receipt["snapshot_root"])),
        "dataset Hub snapshot root",
    )
    expected_dataset_directory = "datasets--" + contract.dataset_repository.replace("/", "--")
    if (
        snapshot_root.name != contract.dataset_revision
        or snapshot_root.parent.name != "snapshots"
        or snapshot_root.parent.parent.name != expected_dataset_directory
    ):
        raise ReproductionError("dataset Hub snapshot staged path mismatch")
    dataset_siblings = _verify_hub_snapshot(
        snapshot_root,
        dataset_receipt["siblings"],
        label="dataset Hub snapshot",
        blobs_root=snapshot_root.parent.parent / "blobs",
        expected_count=contract.dataset_metadata_count,
        expected_metadata_sha256=contract.dataset_metadata_sha256,
    )
    _verify_hub_snapshot(
        dataset,
        dataset_siblings,
        label="staged dataset",
        blobs_root=snapshot_root.parent.parent / "blobs",
        expected_count=contract.dataset_metadata_count,
        expected_metadata_sha256=contract.dataset_metadata_sha256,
    )
    dataset_paths = {str(row["path"]) for row in dataset_siblings}
    if not any(path.startswith("meta/") for path in dataset_paths) or not any(
        path.startswith("data/") for path in dataset_paths
    ):
        raise ReproductionError("dataset content manifest lacks metadata or data")
    return (
        model,
        hub_cache,
        dataset,
        receipt,
        _sha256_bytes(receipt_bytes),
        contract.base_model_metadata_sha256,
        contract.dataset_metadata_sha256,
    )


def verify_inputs(
    *,
    checkout: Path | str,
    source_receipt: Path | str,
    resolved_snapshots_receipt: Path | str,
    vm_id: str,
    disk_id: str,
    contract: ReproductionContract = CONTRACT,
) -> VerifiedInputs:
    """Verify all identities needed to render the upstream training command."""

    root, source_path, source_sha256 = _validate_source(checkout, source_receipt, contract)
    (
        model,
        hub_cache,
        dataset,
        snapshots_path,
        snapshots_sha256,
        model_metadata_sha256,
        dataset_metadata_sha256,
    ) = _validate_snapshots(
        resolved_snapshots_receipt,
        vm_id,
        disk_id,
        root,
        contract,
    )
    return VerifiedInputs(
        checkout=root,
        base_model_root=model,
        hub_cache_root=hub_cache,
        dataset_root=dataset,
        source_receipt=source_path,
        resolved_snapshots_receipt=snapshots_path,
        source_receipt_sha256=source_sha256,
        resolved_snapshots_receipt_sha256=snapshots_sha256,
        source_tree=contract.source_tree,
        base_model_metadata_sha256=model_metadata_sha256,
        dataset_metadata_sha256=dataset_metadata_sha256,
    )


def _write_atomic_json(path: Path | str, value: object, label: str) -> tuple[Path, str]:
    destination = Path(path)
    if not destination.is_absolute():
        raise ReproductionError(f"{label} path is unsafe")
    if destination.exists() or destination.is_symlink():
        raise ReproductionError(f"{label} already exists")
    parent = _regular_directory(destination.parent, f"{label} parent")
    payload = _canonical_bytes(value)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o444)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise ReproductionError(f"{label} already exists") from None
        parent_descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return destination.resolve(strict=True), _sha256_bytes(payload)


def write_receipt(
    *,
    output: Path | str,
    value: Mapping[str, object],
    label: str,
) -> dict[str, object]:
    """Atomically persist a strict-JSON receipt without overwriting."""

    path, digest = _write_atomic_json(output, dict(value), label)
    return {"path": str(path), "sha256": digest}


def _contract_json(contract: ReproductionContract) -> dict[str, object]:
    return {
        "source_repository": contract.source_repository,
        "source_revision": contract.source_revision,
        "source_tree": contract.source_tree,
        "dependency_lock_sha256": contract.dependency_lock_sha256,
        "lerobot_wheel_sha256": contract.lerobot_wheel_sha256,
        "lerobot_package_file_count": contract.lerobot_package_file_count,
        "lerobot_package_tree_sha256": contract.lerobot_package_tree_sha256,
        "base_model_metadata_count": contract.base_model_metadata_count,
        "base_model_metadata_sha256": contract.base_model_metadata_sha256,
        "dataset_metadata_count": contract.dataset_metadata_count,
        "dataset_metadata_sha256": contract.dataset_metadata_sha256,
        "base_model_repository": contract.base_model_repository,
        "base_model_revision": contract.base_model_revision,
        "dataset_repository": contract.dataset_repository,
        "dataset_revision": contract.dataset_revision,
        "trusted_source_files": dict(contract.trusted_source_files),
        "vm_id": contract.vm_id,
        "disk_id": contract.disk_id,
        "python_version": contract.python_version,
        "lerobot_version": contract.lerobot_version,
        "training_command": list(contract.training_command),
        "training": dict(contract.training),
    }


def build_training_manifest(
    *,
    verified: VerifiedInputs,
    contract: ReproductionContract = CONTRACT,
) -> dict[str, object]:
    """Build the canonical execution manifest for later offline comparison."""

    return {
        "schema_version": 1,
        "kind": "lehome_public_n15_training_execution_v1",
        "contract": _contract_json(contract),
        "inputs": {
            "base_model_root": str(verified.base_model_root),
            "hub_cache_root": str(verified.hub_cache_root),
            "dataset_root": str(verified.dataset_root),
            "source_receipt": str(verified.source_receipt),
            "source_receipt_sha256": verified.source_receipt_sha256,
            "source_tree": verified.source_tree,
            "resolved_snapshots_receipt": str(verified.resolved_snapshots_receipt),
            "resolved_snapshots_receipt_sha256": verified.resolved_snapshots_receipt_sha256,
            "base_model_metadata_sha256": verified.base_model_metadata_sha256,
            "dataset_metadata_sha256": verified.dataset_metadata_sha256,
        },
        "execution": {
            "cwd": str(verified.checkout),
            "argv": list(contract.training_command),
            "shell_argv": shlex.join(contract.training_command),
            "container": {
                "image_id": _TRAINING_CONTAINER_IMAGE_ID,
                "python_executable": _TRAINING_CONTAINER_PYTHON,
                "pythonpath": _TRAINING_CONTAINER_PYTHONPATH,
            },
            "env": {
                "HF_HUB_CACHE": str(verified.hub_cache_root),
                "HF_HUB_OFFLINE": "1",
                "PYTHONPATH": _TRAINING_CONTAINER_PYTHONPATH,
            },
        },
    }


def render_training(
    *,
    verified: VerifiedInputs,
    output: Path | str,
    contract: ReproductionContract = CONTRACT,
) -> dict[str, object]:
    """Atomically render, but never execute, the exact upstream command."""

    manifest = build_training_manifest(verified=verified, contract=contract)
    manifest_path, digest = _write_atomic_json(output, manifest, "training manifest")
    return {
        "schema_version": 1,
        "kind": "lehome_public_n15_rendered_training_v1",
        "manifest": str(manifest_path),
        "manifest_sha256": digest,
        "argv": list(contract.training_command),
        "shell_argv": shlex.join(contract.training_command),
    }


def _artifact_files(root: Path) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative in {"training-identity.json", "training-publication.json"}:
            continue
        metadata = path.lstat()
        if path.is_symlink():
            if relative == "checkpoints/last" and os.readlink(path) == "012000":
                continue
            raise ReproductionError(f"training output contains unsafe symlink: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ReproductionError(f"training output contains unsafe entry: {relative}")
        artifacts[relative] = path.resolve(strict=True)
    return artifacts


def _checksum_entries(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError):
        raise ReproductionError("training checksums are unreadable") from None
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  (\S+)", line)
        if match is None:
            raise ReproductionError("training checksum manifest is invalid")
        digest, relative = match.groups()
        pure = PurePosixPath(relative)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts) or relative in entries:
            raise ReproductionError("training checksum path is unsafe or duplicated")
        entries[relative] = digest
    return entries


def verify_training_output(
    *,
    verified: VerifiedInputs,
    training_root: Path | str,
    contract: ReproductionContract = CONTRACT,
) -> dict[str, object]:
    """Verify a complete step-12,000 upstream training output offline."""

    root = _regular_directory(Path(training_root), "training output root")
    last = root / "checkpoints/last"
    if not last.is_symlink() or os.readlink(last) != "012000":
        raise ReproductionError("upstream last-checkpoint link is missing or invalid")
    artifacts = _artifact_files(root)
    checksum_relative = "checksums.sha256"
    checksum_path = artifacts.get(checksum_relative)
    if checksum_path is None:
        raise ReproductionError("training checksums are missing")
    expected_paths = set(artifacts) - {checksum_relative}
    checksum_entries = _checksum_entries(checksum_path)
    if set(checksum_entries) != expected_paths:
        raise ReproductionError("training checksum file set mismatch")
    for relative, expected in checksum_entries.items():
        if _sha256_file(artifacts[relative]) != expected:
            raise ReproductionError(f"training checksum mismatch: {relative}")

    checkpoint_relative = f"checkpoints/{int(contract.training['steps']):06d}"
    checkpoint_root = _regular_directory(root / checkpoint_relative, "step-12,000 checkpoint")
    checkpoint_files = {
        relative: path
        for relative, path in artifacts.items()
        if relative.startswith(checkpoint_relative + "/")
    }
    if not checkpoint_files:
        raise ReproductionError("step-12,000 checkpoint is empty")

    required_pretrained = {
        "config.json",
        "model.safetensors",
        "train_config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_2_groot_pack_inputs_v3.safetensors",
        "policy_postprocessor_step_0_groot_action_unpack_unnormalize_v1.safetensors",
    }
    pretrained_root = checkpoint_root / "pretrained_model"
    _regular_directory(pretrained_root, "upstream pretrained model")
    pretrained_names = {
        path.relative_to(pretrained_root).as_posix()
        for path in pretrained_root.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if not required_pretrained.issubset(pretrained_names):
        raise ReproductionError("upstream pretrained-model checkpoint structure is incomplete")
    model_weights = _regular_file(pretrained_root / "model.safetensors", "trained model weights")
    if model_weights.stat().st_size == 0:
        raise ReproductionError("trained model weights are empty")
    try:
        train_config = json.loads((pretrained_root / "train_config.json").read_text(encoding="utf-8"))
        policy_config = json.loads((pretrained_root / "config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ReproductionError("upstream checkpoint configuration is invalid") from None
    if (
        not isinstance(train_config, dict)
        or train_config.get("steps") != int(contract.training["steps"])
        or train_config.get("batch_size") != int(contract.training["batch_size"])
        or not isinstance(policy_config, dict)
        or policy_config.get("type") != "groot"
    ):
        raise ReproductionError("upstream checkpoint configuration does not prove the recipe")

    training_state_root = _regular_directory(
        checkpoint_root / "training_state",
        "upstream training state",
    )
    required_training_state = {
        "optimizer_param_groups.json",
        "optimizer_state.safetensors",
        "rng_state.safetensors",
        "scheduler_state.json",
        "training_step.json",
    }
    state_names = {
        path.name
        for path in training_state_root.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if not required_training_state.issubset(state_names):
        raise ReproductionError("upstream training-state structure is incomplete")
    try:
        step_receipt = json.loads(
            (training_state_root / "training_step.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ReproductionError("training-step evidence is invalid") from None
    if (
        not isinstance(step_receipt, dict)
        or set(step_receipt) != {"step"}
        or step_receipt.get("step") != int(contract.training["steps"])
    ):
        raise ReproductionError("training-step evidence does not prove step 12000")

    source_copy = artifacts.get("evidence/source-receipt.json")
    snapshots_copy = artifacts.get("evidence/resolved-snapshots-receipt.json")
    if source_copy is None or snapshots_copy is None:
        raise ReproductionError("training source or resolved-snapshot receipt is missing")
    if source_copy.read_bytes() != verified.source_receipt.read_bytes():
        raise ReproductionError("training source receipt mismatch")
    if snapshots_copy.read_bytes() != verified.resolved_snapshots_receipt.read_bytes():
        raise ReproductionError("training resolved-snapshot receipt mismatch")
    execution_path = artifacts.get("evidence/execution-manifest.json")
    if execution_path is None:
        raise ReproductionError("training execution manifest is missing")
    _, _, execution = _load_receipt(execution_path, "training execution manifest")
    if execution != build_training_manifest(verified=verified, contract=contract):
        raise ReproductionError("training execution manifest mismatch")
    lock = artifacts.get("evidence/uv.lock")
    if (
        lock is None
        or _sha256_file(lock) != contract.dependency_lock_sha256
        or lock.read_bytes() != (verified.checkout / "uv.lock").read_bytes()
    ):
        raise ReproductionError("training dependency lock mismatch")
    image_path = artifacts.get("evidence/runtime-image-receipt.json")
    container_runtime_path = artifacts.get(
        "evidence/training-container-runtime-receipt.json"
    )
    if image_path is None or container_runtime_path is None:
        raise ReproductionError("training container runtime evidence is missing")
    _, _, image_receipt = _load_receipt(image_path, "training runtime image receipt")
    if image_receipt != {
        "schema_version": 1,
        "kind": "lehome_public_n15_training_runtime_image_v1",
        "image_id": _TRAINING_CONTAINER_IMAGE_ID,
    }:
        raise ReproductionError("training runtime image identity mismatch")
    _, _, container_runtime = _load_receipt(
        container_runtime_path, "training container runtime receipt"
    )
    container_keys = {
        "schema_version",
        "kind",
        "image_id",
        "python_executable",
        "python_version",
        "pythonpath",
        "lerobot_origin",
        "peft_origin",
        "flash_attn_origin",
        "torch_version",
        "torch_cuda_version",
        "cuda_capability",
    }
    python_version = container_runtime.get("python_version")
    if (
        set(container_runtime) != container_keys
        or container_runtime.get("schema_version") != 1
        or container_runtime.get("kind")
        != "lehome_public_n15_training_container_runtime_v1"
        or container_runtime.get("image_id") != _TRAINING_CONTAINER_IMAGE_ID
        or container_runtime.get("python_executable") != _TRAINING_CONTAINER_PYTHON
        or container_runtime.get("pythonpath") != _TRAINING_CONTAINER_PYTHONPATH
        or container_runtime.get("lerobot_origin")
        != _TRAINING_CONTAINER_LEROBOT_ROOT + "/__init__.py"
        or container_runtime.get("peft_origin")
        != "/deps/peft-0.18.1-py3-none-any.whl/peft/__init__.py"
        or container_runtime.get("flash_attn_origin")
        != "/flash/site-packages/flash_attn/__init__.py"
        or container_runtime.get("torch_version") != "2.7.0+cu128"
        or container_runtime.get("torch_cuda_version") != "12.8"
        or container_runtime.get("cuda_capability") != [12, 0]
        or not isinstance(python_version, list)
        or len(python_version) != 3
        or any(type(part) is not int for part in python_version)
        or python_version[:2] != [3, 11]
    ):
        raise ReproductionError("training container runtime identity mismatch")
    runtime_path = artifacts.get("evidence/runtime-receipt.json")
    if runtime_path is None:
        raise ReproductionError("training runtime receipt is missing")
    _, _, runtime = _load_receipt(runtime_path, "training runtime receipt")
    if set(runtime) != {
        "schema_version",
        "kind",
        "python_executable",
        "upstream_lerobot_wheel_path",
        "upstream_lerobot_wheel_sha256",
        "compatibility_wheel_path",
        "compatibility_wheel_sha256",
        "compatibility_wheel_receipt_path",
        "compatibility_wheel_receipt_sha256",
        "lerobot_package_root",
        "dependency_lock_path",
        "dependency_lock_sha256",
        "scheduler",
    }:
        raise ReproductionError("training runtime receipt schema mismatch")
    if (
        runtime["schema_version"] != 1
        or runtime["kind"] != "lehome_public_n15_training_runtime_v1"
        or runtime["upstream_lerobot_wheel_sha256"] != contract.lerobot_wheel_sha256
        or runtime["dependency_lock_sha256"] != contract.dependency_lock_sha256
        or runtime["dependency_lock_path"] != str(lock)
        or not isinstance(runtime["python_executable"], str)
        or not isinstance(runtime["upstream_lerobot_wheel_path"], str)
        or not isinstance(runtime["compatibility_wheel_path"], str)
        or not isinstance(runtime["compatibility_wheel_sha256"], str)
        or not isinstance(runtime["compatibility_wheel_receipt_path"], str)
        or not isinstance(runtime["compatibility_wheel_receipt_sha256"], str)
        or not isinstance(runtime["lerobot_package_root"], str)
        or not isinstance(runtime["scheduler"], dict)
    ):
        raise ReproductionError("training runtime identity mismatch")
    python_executable = _regular_file(
        Path(runtime["python_executable"]),
        "training Python executable proof",
    )
    try:
        version_result = subprocess.run(
            [
                str(python_executable),
                "-I",
                "-c",
                "import json,sys; print(json.dumps(list(sys.version_info[:3])))",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise ReproductionError("training Python version probe failed") from None
    try:
        actual_python_version = json.loads(version_result.stdout)
    except (TypeError, json.JSONDecodeError):
        raise ReproductionError("training Python version probe is invalid") from None
    try:
        expected_python = [int(part) for part in contract.python_version.split(".")]
    except ValueError:
        raise ReproductionError("contract Python version is invalid") from None
    if (
        version_result.returncode != 0
        or not isinstance(actual_python_version, list)
        or len(actual_python_version) != 3
        or actual_python_version[: len(expected_python)] != expected_python
        or any(type(part) is not int for part in actual_python_version)
    ):
        raise ReproductionError(
            f"training Python interpreter is not Python {contract.python_version}"
        )

    upstream_wheel = _regular_file(
        Path(runtime["upstream_lerobot_wheel_path"]), "upstream training LeRobot wheel"
    )
    if upstream_wheel != artifacts.get("evidence/upstream/lerobot-0.4.3-py3-none-any.whl"):
        raise ReproductionError("upstream training LeRobot wheel is outside the sealed output")
    upstream_wheel_bytes = upstream_wheel.read_bytes()
    if _sha256_bytes(upstream_wheel_bytes) != contract.lerobot_wheel_sha256:
        raise ReproductionError("upstream training LeRobot wheel digest mismatch")
    upstream_count, upstream_tree = wheel_lerobot_tree_identity(upstream_wheel_bytes)
    if (
        upstream_count != contract.lerobot_package_file_count
        or upstream_tree != contract.lerobot_package_tree_sha256
    ):
        raise ReproductionError("upstream training LeRobot wheel package tree mismatch")
    wheel = _regular_file(Path(runtime["compatibility_wheel_path"]), "compatible training LeRobot wheel")
    compatibility_receipt = _regular_file(
        Path(runtime["compatibility_wheel_receipt_path"]),
        "compatible training LeRobot wheel receipt",
    )
    if (
        wheel != artifacts.get("evidence/compatibility/lerobot-0.4.3-py3-none-any.whl")
        or compatibility_receipt
        != artifacts.get("evidence/compatibility/lerobot-compatibility-receipt.json")
        or _sha256_file(compatibility_receipt) != runtime["compatibility_wheel_receipt_sha256"]
    ):
        raise ReproductionError("compatible training LeRobot wheel is outside the sealed output")
    compatibility = compatibility_wheel_identity(
        wheel=wheel,
        receipt=compatibility_receipt,
        upstream_wheel=upstream_wheel,
        expected_upstream_sha256=contract.lerobot_wheel_sha256,
    )
    if runtime["compatibility_wheel_sha256"] != compatibility["derived_wheel_sha256"]:
        raise ReproductionError("compatible training LeRobot wheel digest mismatch")
    config = _safe_relative_file(
        verified.checkout, "configs/train_groot.yaml", "pinned training config"
    )
    try:
        scheduler = resolve_groot_scheduler_from_yaml(config.read_text(encoding="utf-8"))
    except UnicodeError:
        raise ReproductionError("pinned training config is unreadable") from None
    if runtime["scheduler"] != scheduler:
        raise ReproductionError("training runtime scheduler differs from pinned public YAML")
    package_root = _regular_directory(
        Path(runtime["lerobot_package_root"]),
        "installed LeRobot package root",
    )
    package_count, package_tree_sha256 = _installed_package_tree_identity(package_root)
    if (
        package_count != compatibility["derived_package_file_count"]
        or package_tree_sha256 != compatibility["derived_package_tree_sha256"]
    ):
        raise ReproductionError("installed LeRobot package differs from the compatible wheel")
    log = artifacts.get("logs/train.log")
    if log is None or log.stat().st_size == 0:
        raise ReproductionError("training log is missing or empty")
    try:
        log_text = log.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        raise ReproductionError("training log is unreadable") from None
    if (
        "Checkpoint policy after step 12000" not in log_text
        or "End of training" not in log_text
    ):
        raise ReproductionError("training log lacks successful step-12000 completion evidence")

    return {
        "schema_version": 1,
        "kind": "lehome_public_n15_verified_training_output_v1",
        "training_root": str(root),
        "step": int(contract.training["steps"]),
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_files": {
            relative: _sha256_file(path)
            for relative, path in sorted(checkpoint_files.items())
        },
        "artifact_count": len(expected_paths),
        "checksums_sha256": _sha256_file(checksum_path),
        "source_receipt_sha256": verified.source_receipt_sha256,
        "resolved_snapshots_receipt_sha256": verified.resolved_snapshots_receipt_sha256,
    }
