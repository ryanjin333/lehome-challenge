#!/usr/bin/env python3
"""Fail-closed evidence contracts for the native GR00T N1.5 reference gate."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from typing import Mapping, Sequence
from urllib.request import urlopen
import zipfile


SOURCE_REPOSITORY = "theo-zhou/lehome-groot-submission-4"
SOURCE_REVISION = "d384fe00508acd96ab1c3c5dc265e08261f94b3b"
SOURCE_TREE_SHA256 = "eada9f80b0dda1428177fe4551efa8059fe85845d4db5b32bb673f88a50c6bb2"
LEROBOT_VERSION = "0.4.3"
POLICY_CLASS = "scripts.eval_policy.lerobot_policy.LeRobotPolicy"
TASK_DESCRIPTION = "fold the garment on the table"
SUCCESS_CHECKER = "pinned_raw_success_distance_second_mesh_points"
_HEX = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_VM = re.compile(r"^computeinstance-[a-z0-9]+$")
_DISK = re.compile(r"^computedisk-[a-z0-9]+$")
_IMAGE = re.compile(r"^[^\s]+@sha256:[0-9a-f]{64}$")
_EPISODE_LINE = re.compile(r"Episode\s+(\d+)/2:.*?Success=(True|False)")
_KNOWN_INVALID = re.compile(r"(?:traceback|non[- ]?finite|cloth[ _-]?flight|missing cloth|safety failure|cuda error)", re.IGNORECASE)
PUBLIC_CACHE_REPOSITORY = "ryanjin333/lehome-groot-n17-rollouts"
PROVIDER_SOURCE_IMAGE_ID = "computeimage-u00zf6w3yf72gakhcy"
CANONICAL_CACHE_MANIFEST_SHA256 = "c27b2be5d5f055fafb462294d242d42853ac6cace5a867e5b9d7a159421643de"
METADATA_REPOSITORY = SOURCE_REPOSITORY
METADATA_REVISION = SOURCE_REVISION
ASSETS_REPOSITORY = "lehome/asset_challenge"
ASSETS_REVISION = "bea65fd960ad5a1bb3bd3fa77164b28001c08ef9"
ASSETS_RUNTIME_ROOTS = ("objects", "robots", "scenes", "textures")
RUNTIME_IMAGE_REFERENCE = "lehome-rollout:build"
RUNTIME_IMAGE_ID = "sha256:bec2b688ca03145dd20c010aa32b761a386e3fed57bdc45c3df5d86f9afa15c7"
CHECKPOINT_CONFIG_SHA256 = "b7c385bc57456eae603e929b84defb7e991194aade2aad70785e21991e37614c"
LEROBOT_WHEEL_SHA256 = "b08c1c15b2356bd4e658122deabfb9dacd2d7447de4a4327720991723d4edf2c"
# Derived from the verified wheel's sorted lerobot/ regular files as
# ``relative_path + NUL + sha256(file_bytes) + LF``. Only regular .pyc files
# are excluded when applying the same manifest algorithm to an installation.
LEROBOT_PACKAGE_TREE_SHA256 = "db3b4e18b166d4bb7fb4354cec82a7fbd15bb24230f9d71269a017c774e0852f"
LEROBOT_PACKAGE_FILE_COUNT = 289
CHECKPOINT_COMPATIBILITY_FIELDS = {
    "decay_lr_ratio": 0.1,
    "num_decay_steps": 4000,
}
PEFT_WHEEL_PATH = Path(
    "/mnt/lehome/reference-native/dependencies/peft-0.18.1-py3-none-any.whl"
)
PEFT_WHEEL_FILENAME = "peft-0.18.1-py3-none-any.whl"
PEFT_WHEEL_URL = "https://files.pythonhosted.org/packages/b3/14/b4e3f574acf349ae6f61f9c000a77f97a3b315b4bb6ad03791e79ae4a568/peft-0.18.1-py3-none-any.whl"
PEFT_WHEEL_SHA256 = "0bf06847a3551e3019fc58c440cffc9a6b73e6e2962c95b52e224f77bbdb50f1"
PEFT_WHEEL_SIZE = 556960
PEFT_VERSION = "0.18.1"
FLASH_ATTENTION_WHEEL_PATH = Path(
    "/mnt/lehome/reference-native/dependencies/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
)
FLASH_ATTENTION_WHEEL_FILENAME = (
    "flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
)
FLASH_ATTENTION_WHEEL_URL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3%2Bcu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
)
FLASH_ATTENTION_WHEEL_SHA256 = "cd1a45ebfc1731a13e55ad68e0c9ad92390ddfffba306f9222be67c6d5a805af"
FLASH_ATTENTION_WHEEL_SIZE = 256027206
FLASH_ATTENTION_VERSION = "2.8.3"
FLASH_ATTENTION_WHEEL_TAG = "cp311-cp311-linux_x86_64"
FLASH_ATTENTION_INSTALL_ROOT = "/opt/lehome-challenge/.venv/lib/python3.11/site-packages/flash_attn"


class NativeReferenceGateError(ValueError):
    """Raised when evidence cannot prove a safe native-reference result."""


def _peft_wheel_members(wheel: Path) -> None:
    """Reject zip members that could make the read-only overlay ambiguous."""
    try:
        with zipfile.ZipFile(wheel) as archive:
            members = archive.infolist()
            names: set[str] = set()
            for member in members:
                name = member.filename
                path = PurePosixPath(name)
                mode = member.external_attr >> 16
                if (
                    not name
                    or name in names
                    or "\\" in name
                    or path.is_absolute()
                    or ".." in path.parts
                    or stat.S_ISLNK(mode)
                    or member.flag_bits & 0x1
                ):
                    raise NativeReferenceGateError("unsafe PEFT wheel member")
                names.add(name)
            metadata_name = f"peft-{PEFT_VERSION}.dist-info/METADATA"
            if "peft/__init__.py" not in names or metadata_name not in names:
                raise NativeReferenceGateError("PEFT wheel package metadata is incomplete")
            try:
                metadata = archive.read(metadata_name).decode("utf-8", errors="strict")
            except (KeyError, UnicodeError) as error:
                raise NativeReferenceGateError("PEFT wheel metadata is unreadable") from error
    except zipfile.BadZipFile as error:
        raise NativeReferenceGateError("PEFT wheel is not a valid ZIP archive") from error
    fields: dict[str, str] = {}
    for line in metadata.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields.setdefault(key, value.strip())
    if fields.get("Name") != "peft" or fields.get("Version") != PEFT_VERSION:
        raise NativeReferenceGateError("PEFT wheel metadata is not the exact official distribution")


def inspect_peft_overlay() -> dict[str, object]:
    """Validate and zero-episode-probe the fixed, zipimport-only PEFT overlay."""
    wheel = Path(PEFT_WHEEL_PATH)
    if not wheel.is_absolute() or ".." in wheel.parts:
        raise NativeReferenceGateError("PEFT wheel fixed path is invalid")
    if wheel.name != PEFT_WHEEL_FILENAME:
        raise NativeReferenceGateError("PEFT wheel filename is not exact")
    try:
        metadata = wheel.lstat()
    except OSError as error:
        raise NativeReferenceGateError("PEFT wheel is unavailable or unsafe") from error
    if wheel.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise NativeReferenceGateError("PEFT wheel is unavailable or unsafe")
    if metadata.st_size != PEFT_WHEEL_SIZE:
        raise NativeReferenceGateError("PEFT wheel size does not match the exact official wheel")
    raw = _read_regular_bytes(wheel, "PEFT wheel")
    if hashlib.sha256(raw).hexdigest() != PEFT_WHEEL_SHA256:
        raise NativeReferenceGateError("PEFT wheel digest does not match the exact official wheel")
    _peft_wheel_members(wheel)

    original_path = list(sys.path)
    previous_modules = {
        name: module for name, module in sys.modules.items() if name == "peft" or name.startswith("peft.")
    }
    for name in previous_modules:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(wheel))
    try:
        peft = importlib.import_module("peft")
        origin = getattr(peft, "__file__", None)
        expected_origin = f"{wheel.resolve(strict=True)}/peft/__init__.py"
        if origin != expected_origin:
            raise NativeReferenceGateError("PEFT import origin is not the verified wheel")
        if importlib.metadata.version("peft") != PEFT_VERSION:
            raise NativeReferenceGateError("PEFT import version is not the verified wheel")
        if not all(callable(getattr(peft, name, None)) for name in ("LoraConfig", "get_peft_model")):
            raise NativeReferenceGateError("PEFT import does not expose required symbols")
    except NativeReferenceGateError:
        raise
    except Exception as error:
        raise NativeReferenceGateError("PEFT overlay import probe failed") from error
    finally:
        sys.path[:] = original_path
        for name in tuple(sys.modules):
            if name == "peft" or name.startswith("peft."):
                sys.modules.pop(name, None)
        sys.modules.update(previous_modules)
    return {
        "schema_version": 1,
        "kind": "lehome_native_reference_peft_overlay_v1",
        "wheel_path": str(wheel.resolve(strict=True)),
        "wheel_filename": PEFT_WHEEL_FILENAME,
        "wheel_url": PEFT_WHEEL_URL,
        "wheel_size": PEFT_WHEEL_SIZE,
        "wheel_sha256": PEFT_WHEEL_SHA256,
        "distribution_name": "peft",
        "peft_version": PEFT_VERSION,
        "peft_origin": expected_origin,
        "required_symbols": ["LoraConfig", "get_peft_model"],
    }


def prepare_peft_overlay(receipt_path: Path) -> dict[str, object]:
    receipt = inspect_peft_overlay()
    _write_exclusive(Path(receipt_path), receipt)
    return receipt


def _flash_attention_wheel_members(wheel: Path) -> None:
    """Reject an unsafe or non-native exact FlashAttention wheel offline."""
    try:
        with zipfile.ZipFile(wheel) as archive:
            members = archive.infolist()
            names: set[str] = set()
            for member in members:
                name = member.filename
                path = PurePosixPath(name)
                mode = member.external_attr >> 16
                if (
                    not name
                    or name in names
                    or "\\" in name
                    or path.is_absolute()
                    or ".." in path.parts
                    or stat.S_ISLNK(mode)
                    or member.flag_bits & 0x1
                ):
                    raise NativeReferenceGateError("unsafe FlashAttention wheel member")
                names.add(name)
            metadata_name = f"flash_attn-{FLASH_ATTENTION_VERSION}.dist-info/METADATA"
            wheel_name = f"flash_attn-{FLASH_ATTENTION_VERSION}.dist-info/WHEEL"
            if (
                "flash_attn/__init__.py" not in names
                or metadata_name not in names
                or wheel_name not in names
            ):
                raise NativeReferenceGateError("FlashAttention wheel package metadata is incomplete")
            try:
                metadata = archive.read(metadata_name).decode("utf-8", errors="strict")
                wheel_metadata = archive.read(wheel_name).decode("utf-8", errors="strict")
            except (KeyError, UnicodeError) as error:
                raise NativeReferenceGateError("FlashAttention wheel metadata is unreadable") from error
    except zipfile.BadZipFile as error:
        raise NativeReferenceGateError("FlashAttention wheel is not a valid ZIP archive") from error
    fields: dict[str, str] = {}
    for line in metadata.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields.setdefault(key, value.strip())
    if fields.get("Name") != "flash_attn" or fields.get("Version") != FLASH_ATTENTION_VERSION:
        raise NativeReferenceGateError("FlashAttention wheel metadata is not the exact official distribution")
    wheel_fields: dict[str, list[str]] = {}
    for line in wheel_metadata.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            wheel_fields.setdefault(key, []).append(value.strip())
    if (
        wheel_fields.get("Root-Is-Purelib") != ["false"]
        or wheel_fields.get("Tag") != [FLASH_ATTENTION_WHEEL_TAG]
    ):
        raise NativeReferenceGateError("FlashAttention wheel platform tag is not exact")


def inspect_flash_attention_overlay() -> dict[str, object]:
    """Validate the fixed compiled wheel without importing it or touching CUDA."""
    wheel = Path(FLASH_ATTENTION_WHEEL_PATH)
    if not wheel.is_absolute() or ".." in wheel.parts:
        raise NativeReferenceGateError("FlashAttention wheel fixed path is invalid")
    if wheel.name != FLASH_ATTENTION_WHEEL_FILENAME:
        raise NativeReferenceGateError("FlashAttention wheel filename is not exact")
    try:
        metadata = wheel.lstat()
    except OSError as error:
        raise NativeReferenceGateError("FlashAttention wheel is unavailable or unsafe") from error
    if wheel.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise NativeReferenceGateError("FlashAttention wheel is unavailable or unsafe")
    if metadata.st_size != FLASH_ATTENTION_WHEEL_SIZE:
        raise NativeReferenceGateError("FlashAttention wheel size does not match the exact official wheel")
    raw = _read_regular_bytes(wheel, "FlashAttention wheel")
    if hashlib.sha256(raw).hexdigest() != FLASH_ATTENTION_WHEEL_SHA256:
        raise NativeReferenceGateError("FlashAttention wheel digest does not match the exact official wheel")
    _flash_attention_wheel_members(wheel)
    return {
        "schema_version": 1,
        "kind": "lehome_native_reference_flash_attention_overlay_v1",
        "wheel_path": str(wheel.resolve(strict=True)),
        "wheel_filename": FLASH_ATTENTION_WHEEL_FILENAME,
        "wheel_url": FLASH_ATTENTION_WHEEL_URL,
        "wheel_size": FLASH_ATTENTION_WHEEL_SIZE,
        "wheel_sha256": FLASH_ATTENTION_WHEEL_SHA256,
        "distribution_name": "flash_attn",
        "flash_attn_version": FLASH_ATTENTION_VERSION,
        "wheel_tag": FLASH_ATTENTION_WHEEL_TAG,
    }


def prepare_flash_attention_overlay(receipt_path: Path) -> dict[str, object]:
    receipt = inspect_flash_attention_overlay()
    _write_exclusive(Path(receipt_path), receipt)
    return receipt


def _canonical_lerobot_package_tree(root: Path) -> tuple[str, int]:
    """Hash every package file except inert bytecode/cache artifacts."""
    requested = Path(root)
    if requested.is_symlink():
        raise NativeReferenceGateError("installed LeRobot package tree root is unsafe")
    package_root = requested.resolve(strict=True)
    digest = hashlib.sha256()
    count = 0
    for path in sorted(package_root.rglob("*")):
        relative = path.relative_to(package_root)
        try:
            metadata = path.lstat()
        except OSError as error:
            raise NativeReferenceGateError("installed LeRobot package tree is unreadable") from error
        if path.is_symlink():
            raise NativeReferenceGateError("installed LeRobot package tree contains a symlink")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise NativeReferenceGateError("installed LeRobot package tree contains an unsafe entry")
        if relative.suffix == ".pyc":
            continue
        file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(
            relative.as_posix().encode("utf-8")
            + b"\0"
            + file_sha256.encode("ascii")
            + b"\n"
        )
        count += 1
    return digest.hexdigest(), count


def _validate_installed_lerobot_package() -> dict[str, object]:
    try:
        distribution = importlib.metadata.distribution("lerobot")
        version = distribution.version
        requested_init = Path(distribution.locate_file("lerobot/__init__.py"))
        if requested_init.is_symlink() or requested_init.parent.is_symlink():
            raise NativeReferenceGateError("installed LeRobot distribution path is unsafe")
        distribution_init = requested_init.resolve(strict=True)
    except Exception as error:
        raise NativeReferenceGateError("installed LeRobot distribution identity is unavailable") from error
    if version != LEROBOT_VERSION:
        raise NativeReferenceGateError("official LeRobot wheel version changed")
    package_root = distribution_init.parent
    observed_sha256, observed_count = _canonical_lerobot_package_tree(package_root)
    if (
        observed_sha256 != LEROBOT_PACKAGE_TREE_SHA256
        or observed_count != LEROBOT_PACKAGE_FILE_COUNT
    ):
        raise NativeReferenceGateError("installed LeRobot package tree does not match official wheel")
    try:
        lerobot = importlib.import_module("lerobot")
        imported_init = Path(lerobot.__file__).resolve(strict=True)
    except Exception as error:
        raise NativeReferenceGateError("installed LeRobot import identity is unavailable") from error
    if imported_init != distribution_init:
        raise NativeReferenceGateError("imported LeRobot package and distribution are incoherent")
    return {
        "installed_lerobot_package_root": str(package_root),
        "expected_lerobot_package_tree_sha256": LEROBOT_PACKAGE_TREE_SHA256,
        "expected_lerobot_package_file_count": LEROBOT_PACKAGE_FILE_COUNT,
        "installed_lerobot_package_tree_sha256": observed_sha256,
        "installed_lerobot_package_file_count": observed_count,
    }


@contextmanager
def _repository_package_imports():
    """Expose this checkout only while importing sibling operator modules.

    Direct script execution puts ``scripts/`` rather than the repository root
    on ``sys.path``, so absolute ``scripts.*`` imports otherwise fail.  Keep
    this bootstrap scoped to the three operator branches; the native runtime
    probe continues to construct and validate its own pinned import path.
    """
    repository_root = Path(__file__).resolve().parents[1]
    package_marker = repository_root / "scripts" / "__init__.py"
    if not package_marker.is_file() or package_marker.is_symlink():
        raise NativeReferenceGateError("repository scripts package is unavailable or unsafe")
    root = str(repository_root)
    inserted = root not in sys.path
    if inserted:
        sys.path.insert(0, root)
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(root)
            except ValueError:
                pass


@dataclass(frozen=True)
class NativePublicationEntry:
    relative_path: str
    sha256: str
    byte_size: int


def _canonical_bytes(value: object) -> bytes:
    try:
        return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as error:
        raise NativeReferenceGateError("document is not canonical JSON") from error


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _canonical_cache_section(
    manifest: Mapping[str, object], section: str
) -> tuple[dict[str, tuple[int, str]], tuple[str, ...]]:
    expected_top = {"schema_version", "kind", "metadata", "assets"}
    if (
        set(manifest) != expected_top
        or manifest.get("schema_version") != 1
        or manifest.get("kind") != "lehome_native_reference_canonical_cache_manifest_v1"
    ):
        raise NativeReferenceGateError("canonical cache manifest identity is invalid")
    document = _object(manifest.get(section), f"canonical {section} manifest")
    if section == "metadata":
        expected_identity = {
            "repository_type": "model",
            "repository": METADATA_REPOSITORY,
            "revision": METADATA_REVISION,
            "root": "dataset_meta",
        }
        expected_keys = {*expected_identity, "files"}
        roots: tuple[str, ...] = ()
    elif section == "assets":
        expected_identity = {
            "repository_type": "dataset",
            "repository": ASSETS_REPOSITORY,
            "revision": ASSETS_REVISION,
            "runtime_roots": list(ASSETS_RUNTIME_ROOTS),
        }
        expected_keys = {*expected_identity, "files"}
        roots = ASSETS_RUNTIME_ROOTS
    else:
        raise NativeReferenceGateError("canonical cache section is invalid")
    if set(document) != expected_keys or any(document.get(key) != value for key, value in expected_identity.items()):
        raise NativeReferenceGateError(f"canonical {section} provenance is invalid")
    rows = document.get("files")
    if not isinstance(rows, list) or not rows:
        raise NativeReferenceGateError(f"canonical {section} file manifest is empty")
    files: dict[str, tuple[int, str]] = {}
    for row_value in rows:
        row = _object(row_value, f"canonical {section} file")
        if set(row) != {"path", "size", "sha256"}:
            raise NativeReferenceGateError(f"canonical {section} file schema is invalid")
        path = _safe_path(row.get("path"), f"canonical {section} file path")
        size, digest = row.get("size"), row.get("sha256")
        if type(size) is not int or size < 0:
            raise NativeReferenceGateError(f"canonical {section} file size is invalid")
        _digest(digest, f"canonical {section} file")
        if path in files:
            raise NativeReferenceGateError(f"canonical {section} file path is duplicated")
        if roots and PurePosixPath(path).parts[0] not in roots:
            raise NativeReferenceGateError("canonical assets file escapes the runtime roots")
        files[path] = (size, str(digest))
    return files, roots


def validate_canonical_cache_tree(
    root: Path,
    *,
    section: str,
    manifest_path: Path,
) -> str:
    """Authenticate one offline cache tree against the committed public-revision manifest."""
    raw = _read_regular_bytes(Path(manifest_path), "canonical cache manifest")
    if hashlib.sha256(raw).hexdigest() != CANONICAL_CACHE_MANIFEST_SHA256:
        raise NativeReferenceGateError("canonical cache manifest digest is invalid")
    try:
        manifest = _object(json.loads(raw), "canonical cache manifest")
    except (UnicodeError, json.JSONDecodeError, NativeReferenceGateError) as error:
        raise NativeReferenceGateError("canonical cache manifest is invalid") from error
    expected, _ = _canonical_cache_section(manifest, section)
    cache_root = Path(root)
    try:
        root_metadata = cache_root.lstat()
    except OSError as error:
        raise NativeReferenceGateError(f"canonical {section} cache is unavailable") from error
    if cache_root.is_symlink() or not stat.S_ISDIR(root_metadata.st_mode):
        raise NativeReferenceGateError(f"canonical {section} cache root is unsafe")

    observed_files: dict[str, Path] = {}
    observed_directories: set[str] = set()
    for path in sorted(cache_root.rglob("*")):
        relative = path.relative_to(cache_root).as_posix()
        metadata = path.lstat()
        if path.is_symlink():
            raise NativeReferenceGateError(f"canonical {section} cache contains a symlink")
        if stat.S_ISDIR(metadata.st_mode):
            observed_directories.add(relative)
        elif stat.S_ISREG(metadata.st_mode):
            observed_files[relative] = path
        else:
            raise NativeReferenceGateError(f"canonical {section} cache contains an unsafe entry")
    if set(observed_files) != set(expected):
        raise NativeReferenceGateError(f"canonical {section} cache file set is not exact")
    expected_directories = {
        parent.as_posix()
        for relative in expected
        for parent in PurePosixPath(relative).parents
        if parent.as_posix() != "."
    }
    if observed_directories != expected_directories:
        raise NativeReferenceGateError(f"canonical {section} cache directory set is not exact")

    tree_digest = hashlib.sha256()
    for relative in sorted(expected):
        path = observed_files[relative]
        expected_size, expected_sha256 = expected[relative]
        if path.stat().st_size != expected_size:
            raise NativeReferenceGateError(f"canonical {section} cache size mismatch: {relative}")
        file_digest = hashlib.sha256()
        tree_digest.update(relative.encode("utf-8") + b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                file_digest.update(block)
                tree_digest.update(block)
        if file_digest.hexdigest() != expected_sha256:
            raise NativeReferenceGateError(f"canonical {section} cache digest mismatch: {relative}")
    return tree_digest.hexdigest()


def authenticate_canonical_caches(
    metadata_root: Path,
    assets_root: Path,
    *,
    manifest_path: Path,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "lehome_native_reference_canonical_cache_authentication_v1",
        "canonical_manifest_sha256": CANONICAL_CACHE_MANIFEST_SHA256,
        "metadata_repository": METADATA_REPOSITORY,
        "metadata_revision": METADATA_REVISION,
        "metadata_tree_sha256": validate_canonical_cache_tree(
            metadata_root, section="metadata", manifest_path=manifest_path
        ),
        "assets_repository": ASSETS_REPOSITORY,
        "assets_revision": ASSETS_REVISION,
        "assets_tree_sha256": validate_canonical_cache_tree(
            assets_root, section="assets", manifest_path=manifest_path
        ),
    }


def validate_runtime_asset_bindings(
    canonical_assets_root: Path,
    runtime_repository_root: Path,
) -> dict[str, object]:
    """Prove that both evaluator-visible asset paths are the same bind mounts."""
    canonical_root = Path(canonical_assets_root).resolve(strict=True)
    repository_root = Path(runtime_repository_root).resolve(strict=True)
    runtime_assets_root = repository_root / "Assets"
    bindings: list[dict[str, object]] = []
    for name in ASSETS_RUNTIME_ROOTS:
        canonical = canonical_root / name
        runtime = runtime_assets_root / name
        try:
            canonical_metadata = canonical.lstat()
            runtime_metadata = runtime.lstat()
        except OSError as error:
            raise NativeReferenceGateError(
                f"native runtime asset binding is unavailable: {name}"
            ) from error
        if (
            canonical.is_symlink()
            or runtime.is_symlink()
            or not stat.S_ISDIR(canonical_metadata.st_mode)
            or not stat.S_ISDIR(runtime_metadata.st_mode)
        ):
            raise NativeReferenceGateError(f"native runtime asset binding is unsafe: {name}")
        canonical_identity = (canonical_metadata.st_dev, canonical_metadata.st_ino)
        runtime_identity = (runtime_metadata.st_dev, runtime_metadata.st_ino)
        if canonical_identity != runtime_identity:
            raise NativeReferenceGateError(
                f"native runtime asset binding device/inode mismatch: {name}"
            )
        bindings.append(
            {
                "root": name,
                "device": canonical_metadata.st_dev,
                "inode": canonical_metadata.st_ino,
            }
        )
    return {
        "schema_version": 1,
        "kind": "lehome_native_reference_runtime_asset_bindings_v1",
        "canonical_assets_root": str(canonical_root),
        "runtime_assets_root": str(runtime_assets_root.resolve(strict=True)),
        "bindings": bindings,
    }


def prepare_runtime_asset_mountpoints(runtime_root: Path) -> dict[str, object]:
    """Create only ignored asset mountpoints in an exact clean staged revision."""
    requested_root = Path(runtime_root)
    if requested_root.is_symlink():
        raise NativeReferenceGateError("staged runtime root is a symlink")
    root = requested_root.resolve(strict=True)
    if _COMMIT.fullmatch(root.name) is None:
        raise NativeReferenceGateError("staged runtime root is not revision-addressed")

    def git(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                ["git", "-C", str(root), *arguments],
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise NativeReferenceGateError("staged runtime Git validation failed") from error
        if completed.returncode != 0:
            raise NativeReferenceGateError("staged runtime Git validation failed")
        return completed.stdout.strip()

    if Path(git("rev-parse", "--show-toplevel")).resolve(strict=True) != root:
        raise NativeReferenceGateError("staged runtime root is not the Git worktree root")
    if git("rev-parse", "HEAD") != root.name:
        raise NativeReferenceGateError("staged runtime revision does not match its path")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise NativeReferenceGateError("staged runtime checkout is not clean")
    assets = root / "Assets"
    marker = assets / ".gitignore"
    if git("ls-files", "--error-unmatch", "Assets/.gitignore") != "Assets/.gitignore":
        raise NativeReferenceGateError("staged runtime Assets marker is not tracked")
    try:
        assets_metadata = assets.lstat()
        marker_metadata = marker.lstat()
    except OSError as error:
        raise NativeReferenceGateError("staged runtime Assets root is incomplete") from error
    if (
        assets.is_symlink()
        or marker.is_symlink()
        or not stat.S_ISDIR(assets_metadata.st_mode)
        or not stat.S_ISREG(marker_metadata.st_mode)
    ):
        raise NativeReferenceGateError("staged runtime Assets root is unsafe")

    mountpoints: list[dict[str, object]] = []
    for name in ASSETS_RUNTIME_ROOTS:
        path = assets / name
        try:
            path.mkdir(mode=0o755)
        except FileExistsError:
            pass
        metadata = path.lstat()
        if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
            raise NativeReferenceGateError(f"staged runtime asset mountpoint is unsafe: {name}")
        mountpoints.append({"root": name, "path": str(path)})
    if git("status", "--porcelain", "--untracked-files=all"):
        raise NativeReferenceGateError("asset mountpoint preparation dirtied staged runtime")
    return {
        "schema_version": 1,
        "kind": "lehome_native_reference_runtime_mountpoints_v1",
        "runtime_root": str(root),
        "runtime_revision": root.name,
        "mountpoints": mountpoints,
    }


def prepare_checkpoint_compatibility(
    checkpoint_root: Path,
    sanitized_config_root: Path,
    receipt_path: Path,
) -> dict[str, object]:
    """Build one exclusive inference-only config view without changing the checkpoint."""
    checkpoint = Path(checkpoint_root).resolve(strict=True)
    config_path = checkpoint / "config.json"
    raw = _read_regular_bytes(config_path, "checkpoint config")
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    if raw_sha256 != CHECKPOINT_CONFIG_SHA256:
        raise NativeReferenceGateError("checkpoint compatibility config digest mismatch")
    try:
        values = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise NativeReferenceGateError("checkpoint compatibility config is invalid") from error
    if type(values) is not dict:
        raise NativeReferenceGateError("checkpoint compatibility config must be an object")
    if set(CHECKPOINT_COMPATIBILITY_FIELDS) - set(values):
        raise NativeReferenceGateError("checkpoint compatibility fields are incomplete")
    if (
        type(values["num_decay_steps"]) is not int
        or values["num_decay_steps"] != 4000
        or type(values["decay_lr_ratio"]) is not float
        or values["decay_lr_ratio"] != 0.1
    ):
        raise NativeReferenceGateError("checkpoint compatibility field values changed")

    package_identity = _validate_installed_lerobot_package()
    try:
        from lerobot.policies.groot.configuration_groot import GrootConfig
    except Exception as error:
        raise NativeReferenceGateError("official GrootConfig is unavailable") from error
    fields = getattr(GrootConfig, "__dataclass_fields__", None)
    if (
        GrootConfig.__name__ != "GrootConfig"
        or GrootConfig.__module__ != "lerobot.policies.groot.configuration_groot"
        or not isinstance(fields, dict)
        or any(name in fields for name in CHECKPOINT_COMPATIBILITY_FIELDS)
    ):
        raise NativeReferenceGateError("official GrootConfig field mismatch is not exact")
    try:
        config_module = importlib.import_module(GrootConfig.__module__)
        groot_origin = Path(config_module.__file__).resolve(strict=True)
    except Exception as error:
        raise NativeReferenceGateError("official LeRobot wheel identity is unavailable") from error
    sanitized_values = dict(values)
    removed = [
        {"key": key, "value": sanitized_values.pop(key)}
        for key in sorted(CHECKPOINT_COMPATIBILITY_FIELDS)
    ]
    if set(values) - set(sanitized_values) != set(CHECKPOINT_COMPATIBILITY_FIELDS):
        raise NativeReferenceGateError("checkpoint compatibility removed an unexpected field")
    sanitized_raw = _canonical_bytes(sanitized_values)
    sanitized_root = Path(sanitized_config_root)
    if sanitized_root.is_symlink() or sanitized_root.exists():
        raise NativeReferenceGateError("sanitized checkpoint config root already exists")
    try:
        sanitized_root.mkdir(mode=0o700)
        sanitized_root = sanitized_root.resolve(strict=True)
        descriptor = os.open(
            sanitized_root / "config.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(sanitized_raw)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise NativeReferenceGateError("sanitized checkpoint config could not be created") from error
    if config_path.read_bytes() != raw:
        raise NativeReferenceGateError("original checkpoint config changed during normalization")
    receipt = {
        "schema_version": 1,
        "kind": "lehome_native_reference_checkpoint_compatibility_v1",
        "checkpoint_root": str(checkpoint),
        "raw_config_sha256": raw_sha256,
        "sanitized_config_root": str(sanitized_root),
        "sanitized_config_sha256": hashlib.sha256(sanitized_raw).hexdigest(),
        "removed_fields": removed,
        "lerobot_distribution": "lerobot",
        "lerobot_version": LEROBOT_VERSION,
        "lerobot_wheel_filename": "lerobot-0.4.3-py3-none-any.whl",
        "lerobot_wheel_sha256": LEROBOT_WHEEL_SHA256,
        "groot_config_origin": str(groot_origin),
        "groot_config_missing_fields": sorted(CHECKPOINT_COMPATIBILITY_FIELDS),
        "rationale": "inference_only_remove_unsupported_training_scheduler_fields",
        "original_checkpoint_unchanged": True,
        **package_identity,
    }
    _write_exclusive(Path(receipt_path), receipt)
    return receipt


def validate_runtime_image_observation(document: object) -> dict[str, object]:
    receipt = _object(document, "runtime image observation")
    expected = {
        "schema_version",
        "kind",
        "runtime_image_reference",
        "runtime_image_id",
        "captured_unix_seconds",
        "docker_inspect_sha256",
    }
    if (
        set(receipt) != expected
        or receipt.get("schema_version") != 1
        or receipt.get("kind")
        != "lehome_native_reference_runtime_image_observation_v1"
        or receipt.get("runtime_image_reference") != RUNTIME_IMAGE_REFERENCE
        or receipt.get("runtime_image_id") != RUNTIME_IMAGE_ID
    ):
        raise NativeReferenceGateError("runtime image observation is not the approved local image")
    _digest(receipt.get("docker_inspect_sha256"), "Docker inspect response")
    captured = receipt.get("captured_unix_seconds")
    if type(captured) is not int or captured <= 0:
        raise NativeReferenceGateError("runtime image observation capture time is invalid")
    if not 0 <= time.time() - captured <= 900:
        raise NativeReferenceGateError("runtime image observation is not fresh")
    return receipt


def capture_runtime_image_observation() -> dict[str, object]:
    """Inspect the one approved local image without allowing a caller reference."""
    command = ["docker", "image", "inspect", "--", RUNTIME_IMAGE_REFERENCE]
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise NativeReferenceGateError("approved runtime image inspection failed") from error
    if completed.returncode != 0 or not completed.stdout:
        raise NativeReferenceGateError("approved runtime image inspection failed")
    try:
        rows = json.loads(completed.stdout)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise NativeReferenceGateError("approved runtime image inspection is invalid") from error
    if type(rows) is not list or len(rows) != 1 or type(rows[0]) is not dict:
        raise NativeReferenceGateError("approved runtime image inspection is invalid")
    row = rows[0]
    tags = row.get("RepoTags")
    if (
        row.get("Id") != RUNTIME_IMAGE_ID
        or type(tags) is not list
        or RUNTIME_IMAGE_REFERENCE not in tags
        or any(not isinstance(tag, str) for tag in tags)
    ):
        raise NativeReferenceGateError("approved runtime image ID/tag is invalid")
    return validate_runtime_image_observation(
        {
            "schema_version": 1,
            "kind": "lehome_native_reference_runtime_image_observation_v1",
            "runtime_image_reference": RUNTIME_IMAGE_REFERENCE,
            "runtime_image_id": RUNTIME_IMAGE_ID,
            "captured_unix_seconds": int(time.time()),
            "docker_inspect_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        }
    )


def capture_host_runtime(source_root: Path, isaaclab_root: Path) -> dict[str, object]:
    """Resolve imports with pinned source paths before caller-controlled paths."""
    root = Path(source_root).resolve(strict=True)
    source_package_root = (root / "source" / "lehome").resolve(strict=True)
    trusted_isaaclab_root = Path(isaaclab_root).resolve(strict=True)
    original_path = list(sys.path)
    caller_directory = Path.cwd().resolve()
    retained: list[str] = []
    for value in original_path:
        if not value:
            continue
        try:
            if Path(value).resolve() == caller_directory:
                continue
        except OSError:
            continue
        retained.append(value)
    sys.path[:] = [str(source_package_root), str(root), *retained]
    for name in ("scripts.eval", "scripts", "lehome", "lerobot", "torch", "isaaclab.app", "isaaclab"):
        sys.modules.pop(name, None)
    try:
        lerobot = importlib.import_module("lerobot")
        torch = importlib.import_module("torch")
        try:
            from isaaclab.app import AppLauncher
        except Exception as error:
            raise NativeReferenceGateError(
                "native reference cannot import isaaclab.app.AppLauncher"
            ) from error

        def pinned_origin(name: str) -> str:
            spec = importlib.util.find_spec(name)
            value = None if spec is None else spec.origin
            if not isinstance(value, str) or not value:
                raise NativeReferenceGateError(f"native reference cannot resolve {name} origin")
            path = Path(value).resolve(strict=True)
            try:
                path.relative_to(root)
            except ValueError as error:
                raise NativeReferenceGateError(
                    f"native reference {name} origin escapes pinned source"
                ) from error
            return str(path)

        lerobot_origin = getattr(lerobot, "__file__", None)
        if not isinstance(lerobot_origin, str) or not lerobot_origin:
            raise NativeReferenceGateError("native reference cannot resolve lerobot origin")
        isaaclab_app_spec = importlib.util.find_spec("isaaclab.app")
        isaaclab_app_value = None if isaaclab_app_spec is None else isaaclab_app_spec.origin
        if not isinstance(isaaclab_app_value, str) or not isaaclab_app_value:
            raise NativeReferenceGateError("native reference cannot resolve isaaclab.app origin")
        isaaclab_app_origin = Path(isaaclab_app_value).resolve(strict=True)
        try:
            isaaclab_app_origin.relative_to(trusted_isaaclab_root)
        except ValueError as error:
            raise NativeReferenceGateError(
                "native reference isaaclab.app origin escapes trusted IsaacLab source"
            ) from error
        if (
            AppLauncher.__name__ != "AppLauncher"
            or not AppLauncher.__module__.startswith("isaaclab.app")
        ):
            raise NativeReferenceGateError("native reference AppLauncher class identity is invalid")
        return {
            "schema_version": 1,
            "kind": "lehome_native_reference_host_runtime_v1",
            "source_root": str(root),
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": sys.version,
            "torch_version": str(torch.__version__),
            "lerobot_version": str(getattr(lerobot, "__version__", "")),
            "lerobot_origin": str(Path(lerobot_origin).resolve()),
            "scripts_eval_origin": pinned_origin("scripts.eval"),
            "lehome_origin": pinned_origin("lehome"),
            "isaaclab_app_origin": str(isaaclab_app_origin),
            "app_launcher_class": "isaaclab.app.AppLauncher",
        }
    finally:
        sys.path[:] = original_path


def fetch_public_cache_manifest(
    revision: str,
    path: str,
    *,
    downloader: object | None = None,
) -> tuple[dict[str, object], bytes]:
    """Read the exact cache manifest anonymously from its immutable HF path."""
    if _COMMIT.fullmatch(revision) is None:
        raise NativeReferenceGateError("cache manifest revision is not immutable")
    relative = _safe_path(path, "cache manifest path")
    if not relative.startswith("reference-checks/"):
        raise NativeReferenceGateError("cache manifest path is outside reference-checks")
    url = f"https://huggingface.co/datasets/{PUBLIC_CACHE_REPOSITORY}/resolve/{revision}/{relative}?download=true"
    try:
        if downloader is None:
            with urlopen(url, timeout=30) as response:
                raw = response.read()
        else:
            raw = downloader(url)  # type: ignore[operator]
    except Exception as error:
        raise NativeReferenceGateError("immutable cache manifest readback failed") from error
    if not isinstance(raw, bytes):
        raise NativeReferenceGateError("immutable cache manifest readback is not bytes")
    try:
        manifest = _object(json.loads(raw), "cache manifest")
    except (UnicodeError, json.JSONDecodeError, NativeReferenceGateError) as error:
        raise NativeReferenceGateError("immutable cache manifest is invalid") from error
    expected = {
        "schema_version", "kind", "source_repository", "path",
        "checkpoint_tree_sha256", "metadata_tree_sha256", "assets_tree_sha256",
    }
    if set(manifest) != expected or manifest.get("schema_version") != 2 or manifest.get("kind") != "lehome_native_reference_cache_trust_manifest_v2" or manifest.get("source_repository") != PUBLIC_CACHE_REPOSITORY or manifest.get("path") != relative:
        raise NativeReferenceGateError("immutable cache manifest identity is invalid")
    for key in ("checkpoint_tree_sha256", "metadata_tree_sha256", "assets_tree_sha256"):
        _digest(manifest.get(key), f"cache manifest {key}")
    return manifest, raw


def validate_public_cache_bindings(manifest: Mapping[str, object], *, checkpoint_tree_sha256: str, metadata_tree_sha256: str, assets_tree_sha256: str) -> None:
    expected = (checkpoint_tree_sha256, metadata_tree_sha256, assets_tree_sha256)
    if any(_HEX.fullmatch(value) is None for value in expected) or tuple(manifest.get(key) for key in ("checkpoint_tree_sha256", "metadata_tree_sha256", "assets_tree_sha256")) != expected:
        raise NativeReferenceGateError("immutable cache manifest does not bind observed caches")


def publish_cache_manifest(
    manifest_path: Path,
    *,
    token: str,
    transport: object,
) -> dict[str, object]:
    """Publish exactly one cache-trust manifest, then prove anonymous readback."""
    if not isinstance(token, str) or not token or any(part.isspace() for part in token):
        raise NativeReferenceGateError("HF token is unavailable")
    raw = _read_regular_bytes(Path(manifest_path), "cache manifest")
    try:
        manifest = _object(json.loads(raw), "cache manifest")
    except (UnicodeError, json.JSONDecodeError, NativeReferenceGateError) as error:
        raise NativeReferenceGateError("cache manifest is invalid") from error
    path = str(manifest.get("path", ""))
    # Validate via the same public-readback schema, without an invented revision.
    expected = {
        "schema_version", "kind", "source_repository", "path",
        "checkpoint_tree_sha256", "metadata_tree_sha256", "assets_tree_sha256",
    }
    if (
        set(manifest) != expected
        or manifest.get("schema_version") != 2
        or manifest.get("kind") != "lehome_native_reference_cache_trust_manifest_v2"
        or manifest.get("source_repository") != PUBLIC_CACHE_REPOSITORY
        or not re.fullmatch(r"reference-checks/native-cache-[a-z0-9][a-z0-9.-]{0,63}/cache-trust-manifest\.json", path)
    ):
        raise NativeReferenceGateError("cache manifest identity is invalid")
    for key in ("checkpoint_tree_sha256", "metadata_tree_sha256", "assets_tree_sha256"):
        _digest(manifest.get(key), f"cache manifest {key}")
    prefix = path.rsplit("/", 1)[0]
    entry = NativePublicationEntry("cache-trust-manifest.json", hashlib.sha256(raw).hexdigest(), len(raw))
    staging = Path(tempfile.mkdtemp(prefix="native-cache-manifest-", dir=Path(manifest_path).resolve().parent))
    try:
        _write_bytes_exclusive(staging / entry.relative_path, raw)
        head = transport.resolve_approved_ref(repository=PUBLIC_CACHE_REPOSITORY, ref="main", token=token)
        if not isinstance(head, str) or _COMMIT.fullmatch(head) is None:
            raise NativeReferenceGateError("cache manifest publication ref is not immutable")
        if tuple(transport.list_tree(repository=PUBLIC_CACHE_REPOSITORY, revision=head, token=token, remote_prefix=prefix)):
            raise NativeReferenceGateError("cache manifest publication prefix already exists")
        revision = transport.upload_files(
            repository=PUBLIC_CACHE_REPOSITORY, revision="main", source=staging,
            entries=(entry,), token=token, remote_prefix=prefix, parent_commit=head,
        )
        if not isinstance(revision, str) or _COMMIT.fullmatch(revision) is None:
            raise NativeReferenceGateError("cache manifest publication did not return an immutable revision")
        remote = tuple(transport.list_tree(repository=PUBLIC_CACHE_REPOSITORY, revision=revision, token=token, remote_prefix=prefix))
        files = {str(getattr(item, "relative_path", "")).removeprefix(prefix + "/") for item in remote if getattr(item, "entry_type", None) == "file"}
        if files != {entry.relative_path}:
            raise NativeReferenceGateError("cache manifest publication tree is not exact")
        destination = Path(tempfile.mkdtemp(prefix="native-cache-readback-", dir=staging.parent))
        try:
            observed = transport.download_files(repository=PUBLIC_CACHE_REPOSITORY, revision=revision, destination=destination, relative_paths=(entry.relative_path,), token=None, remote_prefix=prefix)
            if observed != revision or _read_regular_bytes(destination / entry.relative_path, "cache manifest readback") != raw:
                raise NativeReferenceGateError("anonymous cache manifest readback bytes mismatch")
        finally:
            shutil.rmtree(destination, ignore_errors=True)
    except NativeReferenceGateError:
        raise
    except Exception as error:
        raise NativeReferenceGateError("cache manifest publication/readback failed") from error
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {
        "schema_version": 1,
        "kind": "lehome_native_reference_cache_manifest_readback_v1",
        "repository": PUBLIC_CACHE_REPOSITORY,
        "immutable_revision": revision,
        "path": path,
        "manifest_sha256": entry.sha256,
        "readback_verified": True,
    }


def _collect_native_bundle_entries(root: Path) -> tuple[NativePublicationEntry, ...]:
    try:
        metadata = root.lstat()
    except OSError as error:
        raise NativeReferenceGateError("native publication bundle is unavailable") from error
    if root.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise NativeReferenceGateError("native publication bundle is unsafe")
    entries: list[NativePublicationEntry] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(root)
        if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
            raise NativeReferenceGateError("native publication bundle has an unsafe entry")
        descriptor = _artifact_from_file(root, relative)
        entries.append(NativePublicationEntry(descriptor["path"], descriptor["sha256"], descriptor["size"]))
    if not entries:
        raise NativeReferenceGateError("native publication bundle is empty")
    return tuple(entries)


def _native_manifest_digest(entries: Sequence[NativePublicationEntry]) -> str:
    return canonical_sha256([
        {"relative_path": item.relative_path, "sha256": item.sha256, "byte_size": item.byte_size}
        for item in entries
    ])


def publish_native_reference_bundle(
    bundle_root: Path,
    execution: Mapping[str, object],
    *,
    token: str,
    transport: object,
    now: object = time.time,
) -> dict[str, object]:
    """Publish and anonymously read back one closed, immutable native bundle."""
    if not isinstance(token, str) or not token or any(part.isspace() for part in token):
        raise NativeReferenceGateError("HF token is unavailable")
    execution_receipt = _object(execution, "execution receipt")
    if execution_receipt.get("status") != "oracle_matched_pending_finalization":
        raise NativeReferenceGateError("only an oracle-matched execution may be published")
    root = Path(bundle_root).resolve(strict=True)
    manifest_path = root / "bundle-manifest.json"
    if manifest_path.exists() or manifest_path.is_symlink():
        raise NativeReferenceGateError("native bundle manifest already exists")
    preliminary = _collect_native_bundle_entries(root)
    _write_bytes_exclusive(manifest_path, _canonical_bytes({
        "schema_version": 1,
        "kind": "lehome_native_reference_bundle_manifest_v1",
        "entries": [{"relative_path": item.relative_path, "sha256": item.sha256, "byte_size": item.byte_size} for item in preliminary],
    }))
    entries = _collect_native_bundle_entries(root)
    execution_sha = canonical_sha256(execution_receipt)
    prefix = f"reference-checks/native-{execution_sha[:16]}"
    try:
        head = transport.resolve_approved_ref(repository=PUBLIC_CACHE_REPOSITORY, ref="main", token=token)
        if not isinstance(head, str) or _COMMIT.fullmatch(head) is None:
            raise NativeReferenceGateError("publication ref is not immutable")
        existing = tuple(transport.list_tree(repository=PUBLIC_CACHE_REPOSITORY, revision=head, token=token, remote_prefix=prefix))
        if existing:
            raise NativeReferenceGateError("native publication prefix already exists")
        revision = transport.upload_files(repository=PUBLIC_CACHE_REPOSITORY, revision="main", source=root, entries=entries, token=token, remote_prefix=prefix, parent_commit=head)
        if not isinstance(revision, str) or _COMMIT.fullmatch(revision) is None:
            raise NativeReferenceGateError("native publication did not return an immutable revision")
        remote = tuple(transport.list_tree(repository=PUBLIC_CACHE_REPOSITORY, revision=revision, token=token, remote_prefix=prefix))
        expected = {item.relative_path for item in entries}
        actual = {
            str(getattr(item, "relative_path", "")).removeprefix(prefix + "/")
            for item in remote if getattr(item, "entry_type", None) == "file"
        }
        if actual != expected:
            raise NativeReferenceGateError("native publication tree does not match bundle")
        destination = Path(tempfile.mkdtemp(prefix="native-reference-readback-", dir=root.parent))
        try:
            observed = transport.download_files(repository=PUBLIC_CACHE_REPOSITORY, revision=revision, destination=destination, relative_paths=tuple(sorted(expected)), token=None, remote_prefix=prefix)
            if observed != revision:
                raise NativeReferenceGateError("anonymous native readback revision mismatch")
            for entry in entries:
                artifact = _artifact_from_file(destination, Path(entry.relative_path))
                if artifact["sha256"] != entry.sha256 or artifact["size"] != entry.byte_size:
                    raise NativeReferenceGateError("anonymous native readback bytes mismatch")
        finally:
            shutil.rmtree(destination, ignore_errors=True)
    except NativeReferenceGateError:
        raise
    except Exception as error:
        raise NativeReferenceGateError("native publication/readback failed") from error
    published = now()  # type: ignore[operator]
    if type(published) not in (int, float) or published <= 0:
        raise NativeReferenceGateError("publication clock is invalid")
    return {
        "schema_version": 2,
        "kind": "lehome_native_reference_hf_readback_v2",
        "execution_receipt_sha256": execution_sha,
        "repository": PUBLIC_CACHE_REPOSITORY,
        "remote_prefix": prefix,
        "immutable_revision": revision,
        "bundle_manifest_sha256": _native_manifest_digest(entries),
        "published_unix_seconds": int(published),
        "readback_verified": True,
    }


def capture_provider_observation(provider: object, *, expected_state: str) -> dict[str, object]:
    """Capture only an exact existing Nebius instance observation."""
    if expected_state not in {"RUNNING", "STOPPED"}:
        raise NativeReferenceGateError("provider evidence state is invalid")
    try:
        from scripts.finalize_simple_curriculum_collection import (
            EXACT_IMAGE_ID,
            EXACT_INSTANCE_ID,
            EXACT_INSTANCE_NAME,
            PROTECTED_DISK_ID,
            _validate_instance,
        )
        raw = provider.get(EXACT_INSTANCE_ID)  # type: ignore[attr-defined]
        observed = _validate_instance(raw)
    except Exception as error:
        raise NativeReferenceGateError("provider exact-instance readback failed") from error
    if observed.get("state") != expected_state:
        raise NativeReferenceGateError("provider observation has the wrong instance state")
    body = {
        "schema_version": 1,
        "kind": "lehome_native_reference_provider_observation_v1",
        "vm_id": EXACT_INSTANCE_ID,
        "vm_name": EXACT_INSTANCE_NAME,
        "disk_id": PROTECTED_DISK_ID,
        "provider_source_image_id": EXACT_IMAGE_ID,
        "state": expected_state,
        "captured_unix_seconds": int(time.time()),
        "provider_response_sha256": canonical_sha256(observed["raw"]),
    }
    return validate_provider_observation(body, expected_state=expected_state)


def validate_provider_observation(document: object, *, expected_state: str) -> dict[str, object]:
    if expected_state not in {"RUNNING", "STOPPED"}:
        raise NativeReferenceGateError("provider evidence state is invalid")
    receipt = _object(document, "provider observation")
    expected = {"schema_version", "kind", "vm_id", "vm_name", "disk_id", "provider_source_image_id", "state", "captured_unix_seconds", "provider_response_sha256"}
    if set(receipt) != expected or receipt.get("schema_version") != 1 or receipt.get("kind") != "lehome_native_reference_provider_observation_v1" or receipt.get("vm_id") != "computeinstance-u00t6xfqhadrcmssa2" or receipt.get("vm_name") != "lehome-rollout" or receipt.get("disk_id") != "computedisk-u00pbe55crxy7jr56x" or receipt.get("provider_source_image_id") != PROVIDER_SOURCE_IMAGE_ID or receipt.get("state") != expected_state:
        raise NativeReferenceGateError("provider observation is not the exact protected VM state")
    _digest(receipt.get("provider_response_sha256"), "provider response")
    captured = receipt.get("captured_unix_seconds")
    if type(captured) is not int or captured <= 0:
        raise NativeReferenceGateError("provider observation capture time is invalid")
    if expected_state == "RUNNING" and not 0 <= time.time() - captured <= 900:
        raise NativeReferenceGateError("provider RUNNING observation is not fresh")
    return receipt


def _object(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise NativeReferenceGateError(f"{label} must be an object")
    return dict(value)


def _safe_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise NativeReferenceGateError(f"{label} is missing")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.name in {"", "."}:
        raise NativeReferenceGateError(f"{label} is unsafe")
    return path.as_posix()


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise NativeReferenceGateError(f"{label} is not a SHA-256 digest")
    return value


def _artifact(value: object, label: str) -> dict[str, object]:
    artifact = _object(value, label)
    if set(artifact) != {"path", "size", "sha256"}:
        raise NativeReferenceGateError(f"{label} has an unexpected schema")
    if type(artifact["size"]) is not int or artifact["size"] <= 0:
        raise NativeReferenceGateError(f"{label} has an invalid size")
    return {"path": _safe_path(artifact["path"], label), "size": artifact["size"], "sha256": _digest(artifact["sha256"], label)}


def _artifact_from_file(root: Path, relative: Path) -> dict[str, object]:
    path = root / relative
    try:
        metadata = path.lstat()
    except OSError as error:
        raise NativeReferenceGateError(f"missing required artifact: {relative.as_posix()}") from error
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
        raise NativeReferenceGateError(f"required artifact is unsafe: {relative.as_posix()}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": relative.as_posix(), "size": metadata.st_size, "sha256": digest.hexdigest()}


def _read_regular_bytes(path: Path, label: str) -> bytes:
    """Read one nonempty regular file without accepting a symlink boundary."""
    try:
        metadata = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
            raise OSError("unsafe")
        return path.read_bytes()
    except OSError as error:
        raise NativeReferenceGateError(f"{label} is unavailable or unsafe") from error


def _verify_artifact(root: Path, value: object, label: str) -> dict[str, object]:
    artifact = _artifact(value, label)
    observed = _artifact_from_file(root, Path(*PurePosixPath(artifact["path"]).parts))
    if observed != artifact:
        raise NativeReferenceGateError(f"{label} changed after receipt creation")
    return artifact


def oracle_attempts() -> tuple[dict[str, object], ...]:
    rows = (("top_long", "Top_Long_Seen_0", (True, True)), ("top_short", "Top_Short_Seen_0", (True, True)), ("pant_long", "Pant_Long_Seen_0", (True, True)), ("pant_short", "Pant_Short_Seen_0", (False, True)))
    return tuple({"attempt_id": f"native-reference-{stage}-{episode}", "stage": stage, "category": category, "garment": garment, "episode": episode, "expected_success": expected[episode - 1]} for stage, (category, garment, expected) in enumerate(rows, start=1) for episode in (1, 2))


def _validate_identity(value: object) -> dict[str, object]:
    identity = _object(value, "identity")
    fixed = {"source_repository": SOURCE_REPOSITORY, "source_revision": SOURCE_REVISION, "source_tree_sha256": SOURCE_TREE_SHA256, "lerobot_version": LEROBOT_VERSION, "policy_class": POLICY_CLASS, "policy_device": "cuda:0", "simulator_device": "cpu", "task_description": TASK_DESCRIPTION, "action_horizon": 16, "action_dimension": 12, "success_checker": SUCCESS_CHECKER, "cuda_available": True, "runtime_image_reference": RUNTIME_IMAGE_REFERENCE, "runtime_image_id": RUNTIME_IMAGE_ID, "peft_wheel_sha256": PEFT_WHEEL_SHA256, "flash_attention_wheel_sha256": FLASH_ATTENTION_WHEEL_SHA256, "pynput_backend": "dummy"}
    variable = {"checkpoint_tree_sha256", "metadata_tree_sha256", "assets_tree_sha256", "cache_trust_manifest_sha256", "provider_running_receipt_sha256", "runtime_image_receipt_sha256", "checkpoint_compatibility_receipt_sha256", "peft_overlay_receipt_sha256", "flash_attention_overlay_receipt_sha256", "flash_attention_runtime_receipt_sha256", "pynput_backend_receipt_sha256", "provider_source_image_id", "cuda_runtime", "cuda_device_count", "vm_id", "disk_id", "source_root", "python_executable", "python_version", "torch_version", "lerobot_origin", "scripts_eval_origin", "lehome_origin"}
    if set(identity) != {*fixed, *variable}:
        raise NativeReferenceGateError("identity has an unexpected schema")
    for key, expected in fixed.items():
        if identity.get(key) != expected:
            label = (
                "LeRobot"
                if key == "lerobot_version"
                else "runtime image"
                if key in {"runtime_image_reference", "runtime_image_id"}
                else key
            )
            raise NativeReferenceGateError(f"identity {label} does not match the native contract")
    for key in ("checkpoint_tree_sha256", "metadata_tree_sha256", "assets_tree_sha256", "cache_trust_manifest_sha256", "provider_running_receipt_sha256", "runtime_image_receipt_sha256", "checkpoint_compatibility_receipt_sha256", "peft_overlay_receipt_sha256", "flash_attention_overlay_receipt_sha256", "flash_attention_runtime_receipt_sha256", "pynput_backend_receipt_sha256"):
        _digest(identity.get(key), f"identity {key}")
    if type(identity["cuda_device_count"]) is not int or identity["cuda_device_count"] < 1 or not isinstance(identity["cuda_runtime"], str) or not identity["cuda_runtime"]:
        raise NativeReferenceGateError("identity does not prove CUDA availability")
    if not isinstance(identity["vm_id"], str) or _VM.fullmatch(identity["vm_id"]) is None:
        raise NativeReferenceGateError("identity VM ID is invalid")
    if not isinstance(identity["disk_id"], str) or _DISK.fullmatch(identity["disk_id"]) is None:
        raise NativeReferenceGateError("identity disk ID is invalid")
    if identity["provider_source_image_id"] != PROVIDER_SOURCE_IMAGE_ID:
        raise NativeReferenceGateError("identity provider source image does not match the protected VM")
    source_root = _absolute_identity_path(identity.get("source_root"), "source root")
    _absolute_identity_path(identity.get("python_executable"), "Python executable")
    for key in ("python_version", "torch_version"):
        if not isinstance(identity.get(key), str) or not identity[key]:
            raise NativeReferenceGateError(f"identity {key} is missing")
    _absolute_identity_path(identity.get("lerobot_origin"), "LeRobot origin")
    for key in ("scripts_eval_origin", "lehome_origin"):
        origin = _absolute_identity_path(identity.get(key), key.replace("_", " "))
        if origin == source_root or not origin.startswith(source_root + "/"):
            raise NativeReferenceGateError(f"identity {key} is outside the pinned source origin")
    return identity


def _absolute_identity_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise NativeReferenceGateError(f"identity {label} is missing")
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or path.name in {"", "."}:
        raise NativeReferenceGateError(f"identity {label} is unsafe")
    return path.as_posix()


def _validate_checkpoint_compatibility_receipt(
    document: object, *, bundle_root: Path
) -> dict[str, object]:
    receipt = _object(document, "checkpoint compatibility receipt")
    expected_keys = {
        "schema_version", "kind", "checkpoint_root", "raw_config_sha256",
        "sanitized_config_root", "sanitized_config_sha256", "removed_fields",
        "lerobot_distribution", "lerobot_version", "lerobot_wheel_filename",
        "lerobot_wheel_sha256", "groot_config_origin", "groot_config_missing_fields",
        "rationale", "original_checkpoint_unchanged",
        "installed_lerobot_package_root", "expected_lerobot_package_tree_sha256",
        "expected_lerobot_package_file_count", "installed_lerobot_package_tree_sha256",
        "installed_lerobot_package_file_count",
    }
    fixed = {
        "schema_version": 1,
        "kind": "lehome_native_reference_checkpoint_compatibility_v1",
        "raw_config_sha256": CHECKPOINT_CONFIG_SHA256,
        "removed_fields": [
            {"key": key, "value": CHECKPOINT_COMPATIBILITY_FIELDS[key]}
            for key in sorted(CHECKPOINT_COMPATIBILITY_FIELDS)
        ],
        "lerobot_distribution": "lerobot",
        "lerobot_version": LEROBOT_VERSION,
        "lerobot_wheel_filename": "lerobot-0.4.3-py3-none-any.whl",
        "lerobot_wheel_sha256": LEROBOT_WHEEL_SHA256,
        "groot_config_missing_fields": sorted(CHECKPOINT_COMPATIBILITY_FIELDS),
        "rationale": "inference_only_remove_unsupported_training_scheduler_fields",
        "original_checkpoint_unchanged": True,
        "expected_lerobot_package_tree_sha256": LEROBOT_PACKAGE_TREE_SHA256,
        "expected_lerobot_package_file_count": LEROBOT_PACKAGE_FILE_COUNT,
        "installed_lerobot_package_tree_sha256": LEROBOT_PACKAGE_TREE_SHA256,
        "installed_lerobot_package_file_count": LEROBOT_PACKAGE_FILE_COUNT,
    }
    if set(receipt) != expected_keys or any(receipt.get(key) != value for key, value in fixed.items()):
        raise NativeReferenceGateError("checkpoint compatibility receipt contract is invalid")
    _absolute_identity_path(receipt.get("checkpoint_root"), "compatibility checkpoint root")
    _absolute_identity_path(receipt.get("groot_config_origin"), "compatibility GrootConfig origin")
    _absolute_identity_path(
        receipt.get("installed_lerobot_package_root"), "installed LeRobot package root"
    )
    _digest(receipt.get("sanitized_config_sha256"), "sanitized checkpoint config")
    sanitized_root = _absolute_identity_path(
        receipt.get("sanitized_config_root"), "sanitized config root"
    )
    expected_root = str((bundle_root / "checkpoint-config-view").resolve(strict=True))
    if sanitized_root != expected_root:
        raise NativeReferenceGateError("sanitized config root escapes the execution bundle")
    sanitized = _artifact_from_file(bundle_root, Path("checkpoint-config-view/config.json"))
    if sanitized["sha256"] != receipt["sanitized_config_sha256"]:
        raise NativeReferenceGateError("sanitized checkpoint config changed after receipt creation")
    return receipt


def _validate_peft_overlay_receipt(document: object) -> dict[str, object]:
    receipt = _object(document, "PEFT overlay receipt")
    expected = {
        "schema_version": 1,
        "kind": "lehome_native_reference_peft_overlay_v1",
        "wheel_path": str(PEFT_WHEEL_PATH),
        "wheel_filename": PEFT_WHEEL_FILENAME,
        "wheel_url": PEFT_WHEEL_URL,
        "wheel_size": PEFT_WHEEL_SIZE,
        "wheel_sha256": PEFT_WHEEL_SHA256,
        "distribution_name": "peft",
        "peft_version": PEFT_VERSION,
        "peft_origin": f"{PEFT_WHEEL_PATH}/peft/__init__.py",
        "required_symbols": ["LoraConfig", "get_peft_model"],
    }
    if set(receipt) != set(expected) or any(receipt.get(key) != value for key, value in expected.items()):
        raise NativeReferenceGateError("PEFT overlay receipt contract is invalid")
    return receipt


def _validate_flash_attention_overlay_receipt(document: object) -> dict[str, object]:
    receipt = _object(document, "FlashAttention overlay receipt")
    expected = {
        "schema_version": 1,
        "kind": "lehome_native_reference_flash_attention_overlay_v1",
        "wheel_path": str(FLASH_ATTENTION_WHEEL_PATH),
        "wheel_filename": FLASH_ATTENTION_WHEEL_FILENAME,
        "wheel_url": FLASH_ATTENTION_WHEEL_URL,
        "wheel_size": FLASH_ATTENTION_WHEEL_SIZE,
        "wheel_sha256": FLASH_ATTENTION_WHEEL_SHA256,
        "distribution_name": "flash_attn",
        "flash_attn_version": FLASH_ATTENTION_VERSION,
        "wheel_tag": FLASH_ATTENTION_WHEEL_TAG,
    }
    if set(receipt) != set(expected) or any(receipt.get(key) != value for key, value in expected.items()):
        raise NativeReferenceGateError("FlashAttention overlay receipt contract is invalid")
    return receipt


def _validate_flash_attention_runtime_receipt(document: object) -> dict[str, object]:
    receipt = _object(document, "FlashAttention runtime receipt")
    expected_keys = {
        "schema_version", "kind", "torch_version", "torch_cuda_version",
        "torch_cxx11_abi", "cuda_capability", "flash_attn_version",
        "flash_attn_origin", "kernel",
    }
    if set(receipt) != expected_keys or receipt.get("schema_version") != 1 or receipt.get("kind") != "lehome_native_reference_flash_attention_runtime_v1":
        raise NativeReferenceGateError("FlashAttention runtime receipt contract is invalid")
    if receipt.get("torch_version") != "2.7.0+cu128" or receipt.get("torch_cuda_version") != "12.8" or receipt.get("torch_cxx11_abi") is not True:
        raise NativeReferenceGateError("FlashAttention runtime receipt has an incompatible torch build")
    if receipt.get("cuda_capability") != [12, 0]:
        raise NativeReferenceGateError("FlashAttention runtime receipt has an incompatible CUDA capability")
    if receipt.get("flash_attn_version") != FLASH_ATTENTION_VERSION:
        raise NativeReferenceGateError("FlashAttention runtime receipt version is not exact")
    origin = _absolute_identity_path(receipt.get("flash_attn_origin"), "FlashAttention origin")
    if origin != f"{FLASH_ATTENTION_INSTALL_ROOT}/__init__.py":
        raise NativeReferenceGateError("FlashAttention runtime receipt origin is not the ephemeral overlay")
    kernel = _object(receipt.get("kernel"), "FlashAttention kernel probe")
    if set(kernel) != {"shape", "dtype", "finite"} or kernel.get("shape") != [1, 2, 4, 64] or kernel.get("dtype") != "float16" or kernel.get("finite") is not True:
        raise NativeReferenceGateError("FlashAttention runtime receipt does not prove a finite kernel")
    return receipt


def _validate_pynput_backend_receipt(document: object) -> dict[str, object]:
    receipt = _object(document, "pynput backend receipt")
    expected = {
        "schema_version": 1,
        "kind": "lehome_native_reference_pynput_backend_v1",
        "pynput_backend": "dummy",
        "keyboard_listener_module": "pynput.keyboard._base",
        "keyboard_key_module": "pynput.keyboard._base",
        "x11_modules_loaded": False,
        "keyboard_control_started": False,
    }
    if set(receipt) != set(expected) or any(receipt.get(key) != value for key, value in expected.items()):
        raise NativeReferenceGateError("pynput backend receipt contract is invalid")
    return receipt


def _validate_preflight(document: object, identity: Mapping[str, object]) -> dict[str, object]:
    preflight = _object(document, "native reference preflight")
    expected_keys = {
        "schema_version", "kind", "identity", "cuda_probe_sha256",
        "host_runtime_sha256", "runtime_image_receipt_sha256",
        "checkpoint_compatibility_receipt_sha256", "peft_overlay_receipt_sha256",
        "flash_attention_overlay_receipt_sha256",
        "flash_attention_runtime_receipt_sha256", "pynput_backend_receipt_sha256",
    }
    if (
        set(preflight) != expected_keys
        or preflight.get("schema_version") != 2
        or preflight.get("kind") != "lehome_native_reference_preflight_v2"
        or preflight.get("identity") != dict(identity)
    ):
        raise NativeReferenceGateError("native reference preflight contract is invalid")
    for key in (
        "cuda_probe_sha256", "host_runtime_sha256", "runtime_image_receipt_sha256",
        "checkpoint_compatibility_receipt_sha256", "peft_overlay_receipt_sha256",
        "flash_attention_overlay_receipt_sha256",
        "flash_attention_runtime_receipt_sha256", "pynput_backend_receipt_sha256",
    ):
        _digest(preflight.get(key), f"native reference preflight {key}")
    for key in (
        "runtime_image_receipt_sha256", "checkpoint_compatibility_receipt_sha256",
        "peft_overlay_receipt_sha256", "flash_attention_overlay_receipt_sha256",
        "flash_attention_runtime_receipt_sha256", "pynput_backend_receipt_sha256",
    ):
        if preflight[key] != identity[key]:
            raise NativeReferenceGateError("native reference preflight is not bound to identity")
    return preflight


def _validate_attempt(value: object, expected: Mapping[str, object], root: Path | None) -> dict[str, object]:
    attempt = _object(value, "attempt")
    if set(attempt) != {*expected, "success", "videos", "log", "receipt"}:
        raise NativeReferenceGateError("attempt has an unexpected schema")
    if any(attempt.get(key) != item for key, item in expected.items()) or type(attempt.get("success")) is not bool:
        raise NativeReferenceGateError("attempt sequence does not match the native oracle")
    if type(attempt["videos"]) is not list or len(attempt["videos"]) != 3:
        raise NativeReferenceGateError("attempt must enumerate exactly three RGB videos")
    parser = _verify_artifact if root is not None else lambda _root, item, label: _artifact(item, label)
    base = root if root is not None else Path(".")
    videos = [parser(base, item, "attempt video") for item in attempt["videos"]]
    expected_views = {"left", "right", "top"}
    observed_views = {str(video["path"]).rsplit("_", 2)[-2] for video in videos if str(video["path"]).endswith("_rgb.mp4")}
    if observed_views != expected_views:
        raise NativeReferenceGateError("attempt videos must be left/right/top RGB")
    return {**{key: attempt[key] for key in expected}, "success": attempt["success"], "videos": videos, "log": parser(base, attempt["log"], "attempt log"), "receipt": parser(base, attempt["receipt"], "attempt receipt")}


def verify_native_reference_result(document: object, *, bundle_root: Path | None = None) -> dict[str, object]:
    """Assess execution only. A local oracle match never claims final passage."""
    result = _object(document, "native reference result")
    if set(result) != {"schema_version", "kind", "identity", "attempts"} or result.get("schema_version") != 2 or result.get("kind") != "lehome_native_reference_execution_result_v2":
        raise NativeReferenceGateError("native reference result kind is invalid")
    identity = _validate_identity(result.get("identity"))
    supporting_artifacts: dict[str, object] = {}
    if bundle_root is not None:
        preflight = _artifact_from_file(bundle_root, Path("preflight.json"))
        cache = _artifact_from_file(bundle_root, Path("evidence/cache-trust-manifest.json"))
        running = _artifact_from_file(bundle_root, Path("evidence/provider-running-receipt.json"))
        runtime_image = _artifact_from_file(bundle_root, Path("evidence/runtime-image-receipt.json"))
        compatibility = _artifact_from_file(bundle_root, Path("evidence/checkpoint-compatibility-receipt.json"))
        peft_overlay = _artifact_from_file(bundle_root, Path("evidence/peft-overlay-receipt.json"))
        flash_overlay = _artifact_from_file(bundle_root, Path("evidence/flash-attention-overlay-receipt.json"))
        flash_runtime = _artifact_from_file(bundle_root, Path("evidence/flash-attention-runtime-receipt.json"))
        pynput_backend = _artifact_from_file(bundle_root, Path("evidence/pynput-backend-receipt.json"))
        _validate_checkpoint_compatibility_receipt(
            _read_json(bundle_root / "evidence/checkpoint-compatibility-receipt.json", "checkpoint compatibility receipt"),
            bundle_root=bundle_root,
        )
        _validate_peft_overlay_receipt(
            _read_json(bundle_root / "evidence/peft-overlay-receipt.json", "PEFT overlay receipt")
        )
        _validate_flash_attention_overlay_receipt(
            _read_json(bundle_root / "evidence/flash-attention-overlay-receipt.json", "FlashAttention overlay receipt")
        )
        _validate_flash_attention_runtime_receipt(
            _read_json(bundle_root / "evidence/flash-attention-runtime-receipt.json", "FlashAttention runtime receipt")
        )
        _validate_pynput_backend_receipt(
            _read_json(bundle_root / "evidence/pynput-backend-receipt.json", "pynput backend receipt")
        )
        _validate_preflight(
            _read_json(bundle_root / "preflight.json", "native reference preflight"), identity
        )
        if cache["sha256"] != identity["cache_trust_manifest_sha256"] or running["sha256"] != identity["provider_running_receipt_sha256"] or runtime_image["sha256"] != identity["runtime_image_receipt_sha256"] or compatibility["sha256"] != identity["checkpoint_compatibility_receipt_sha256"] or peft_overlay["sha256"] != identity["peft_overlay_receipt_sha256"] or flash_overlay["sha256"] != identity["flash_attention_overlay_receipt_sha256"] or flash_runtime["sha256"] != identity["flash_attention_runtime_receipt_sha256"] or pynput_backend["sha256"] != identity["pynput_backend_receipt_sha256"]:
            raise NativeReferenceGateError("execution bundle support evidence is not bound to identity")
        supporting_artifacts = {"preflight": preflight, "cache_trust_manifest": cache, "provider_running_receipt": running, "runtime_image_receipt": runtime_image, "checkpoint_compatibility_receipt": compatibility, "peft_overlay_receipt": peft_overlay, "flash_attention_overlay_receipt": flash_overlay, "flash_attention_runtime_receipt": flash_runtime, "pynput_backend_receipt": pynput_backend}
    rows = result.get("attempts")
    if type(rows) is not list or len(rows) not in {2, 8}:
        raise NativeReferenceGateError("native reference result must contain either the two-attempt admission stop or all eight attempts")
    oracle = oracle_attempts(); attempts = [_validate_attempt(row, oracle[index], bundle_root) for index, row in enumerate(rows)]
    first = [row["success"] for row in attempts[:2]]
    evidence = {row["attempt_id"]: canonical_sha256({"attempt_receipt": row["receipt"], "log": row["log"], "videos": row["videos"]}) for row in attempts}
    if first != [True, True]:
        if len(attempts) != 2: raise NativeReferenceGateError("failed Top_Long admission must fail fast before later attempts")
        return {"schema_version": 1, "kind": "lehome_native_reference_execution_receipt_v2", "status": "evaluator_compatibility_stop", "reason": "top_long_admission_failed", "attempt_count": 2, "successes": sum(first), "oracle_vector": [row["expected_success"] for row in oracle], "attempt_evidence_sha256": evidence, "supporting_artifacts": supporting_artifacts, "identity": identity, "result_sha256": canonical_sha256(result)}
    if len(attempts) != 8: raise NativeReferenceGateError("passing Top_Long admission requires all eight sequential attempts")
    observed, expected = [row["success"] for row in attempts], [row["expected_success"] for row in oracle]
    if observed != expected:
        return {"schema_version": 1, "kind": "lehome_native_reference_execution_receipt_v2", "status": "evaluator_compatibility_stop", "reason": "oracle_outcome_mismatch", "attempt_count": 8, "successes": sum(observed), "oracle_vector": expected, "observed_vector": observed, "attempt_evidence_sha256": evidence, "supporting_artifacts": supporting_artifacts, "identity": identity, "result_sha256": canonical_sha256(result)}
    return {"schema_version": 1, "kind": "lehome_native_reference_execution_receipt_v2", "status": "oracle_matched_pending_finalization", "attempt_count": 8, "successes": 7, "oracle_vector": expected, "attempt_evidence_sha256": evidence, "supporting_artifacts": supporting_artifacts, "identity": identity, "result_sha256": canonical_sha256(result)}


def _write_exclusive(path: Path, value: Mapping[str, object]) -> None:
    payload = _canonical_bytes(dict(value))
    try: descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as error: raise NativeReferenceGateError("native reference receipt already exists") from error
    with os.fdopen(descriptor, "wb") as stream: stream.write(payload); stream.flush(); os.fsync(stream.fileno())


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as error:
        raise NativeReferenceGateError("native reference receipt already exists") from error
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())


def _write_result(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(".partial")
    if temporary.exists() or temporary.is_symlink(): raise NativeReferenceGateError("native result temporary path is unsafe")
    temporary.write_bytes(_canonical_bytes(dict(value))); os.replace(temporary, path)


def compile_native_stage(bundle_root: Path, *, stage: int, category: str, garment: str, identity: Mapping[str, object]) -> dict[str, object]:
    """Compile a real public evaluator log and exact files into append-only evidence."""
    root = Path(bundle_root).resolve(strict=True); log_relative = Path("logs") / f"stage-{stage}.log"; log = _artifact_from_file(root, log_relative)
    text = (root / log_relative).read_text(encoding="utf-8", errors="strict")
    if _KNOWN_INVALID.search(text): raise NativeReferenceGateError("native evaluator log contains an infrastructure or fidelity failure")
    matches = _EPISODE_LINE.findall(text)
    if len(matches) != 2 or [item[0] for item in matches] != ["1", "2"]: raise NativeReferenceGateError("native evaluator log does not contain exactly two ordered episode outcomes")
    if stage not in {1, 2, 3, 4}: raise NativeReferenceGateError("native stage is invalid")
    expected_rows = oracle_attempts()[(stage - 1) * 2:stage * 2]
    if any(row["category"] != category or row["garment"] != garment for row in expected_rows): raise NativeReferenceGateError("native stage descriptor does not match the oracle")
    result_path = root / "result.json"
    if result_path.exists():
        metadata = result_path.lstat()
        if result_path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise NativeReferenceGateError("native stage result is unsafe")
        result = _object(json.loads(result_path.read_text(encoding="utf-8")), "native reference result")
    else:
        result = {"schema_version": 2, "kind": "lehome_native_reference_execution_result_v2", "identity": _validate_identity(identity), "attempts": []}
    existing = result.get("attempts")
    if type(existing) is not list or len(existing) != (stage - 1) * 2 or result.get("identity") != _validate_identity(identity):
        raise NativeReferenceGateError("native stage result is not an append-only oracle prefix")
    receipts = root / "receipts"; receipts.mkdir(mode=0o700, exist_ok=True)
    for row, (_, outcome) in zip(expected_rows, matches, strict=True):
        outcome_directory = "success" if outcome == "True" else "failure"
        video_root = root / "videos" / f"stage-{stage}"
        expected_paths = [
            video_root / outcome_directory / f"episode{int(row['episode']) - 1}_observation_images_{view}_rgb.mp4"
            for view in ("left", "right", "top")
        ]
        episode_files = sorted(video_root.glob(f"**/episode{int(row['episode']) - 1}_*_rgb.mp4"))
        if set(episode_files) != set(expected_paths):
            raise NativeReferenceGateError(f"native evaluator has missing, extra, or wrong-directory videos for {row['attempt_id']}")
        candidates = expected_paths
        videos = [_artifact_from_file(root, path.relative_to(root)) for path in candidates]; receipt_relative = Path("receipts") / f"{row['attempt_id']}.json"; receipt_path = root / receipt_relative
        body = {"schema_version": 1, "kind": "lehome_native_reference_attempt_receipt_v1", "attempt_id": row["attempt_id"], "stage": stage, "category": category, "garment": garment, "episode": row["episode"], "success": outcome == "True", "log": log, "videos": videos}
        _write_exclusive(receipt_path, body); result["attempts"].append({**row, "success": outcome == "True", "videos": videos, "log": log, "receipt": _artifact_from_file(root, receipt_relative)})
    _write_result(result_path, result); return result


def finalize_native_reference_gate(execution: object, fidelity: object, publication: object, stopped: object) -> dict[str, object]:
    execution = _object(execution, "execution receipt")
    if execution.get("kind") != "lehome_native_reference_execution_receipt_v2" or execution.get("status") != "oracle_matched_pending_finalization": raise NativeReferenceGateError("execution receipt is not an oracle-matched pending result")
    identity = _validate_identity(execution.get("identity")); execution_sha = canonical_sha256(execution); fidelity = _object(fidelity, "fidelity review")
    if set(fidelity) != {"schema_version", "kind", "execution_receipt_sha256", "review_method", "attempts"} or fidelity.get("schema_version") != 1 or fidelity.get("kind") != "lehome_native_reference_fidelity_review_v1" or fidelity.get("execution_receipt_sha256") != execution_sha or fidelity.get("review_method") != "manual_video_audit": raise NativeReferenceGateError("fidelity review is not bound to the execution receipt")
    reviewed, ids = fidelity.get("attempts"), [row["attempt_id"] for row in oracle_attempts()]
    if type(reviewed) is not list or [row.get("attempt_id") if isinstance(row, dict) else None for row in reviewed] != ids: raise NativeReferenceGateError("fidelity review does not cover every attempt")
    for row in reviewed:
        if set(row) != {"attempt_id", "cloth_present", "cloth_flight", "nonfinite", "safety_failure", "evidence_sha256"} or row.get("cloth_present") is not True or any(row.get(key) is not False for key in ("cloth_flight", "nonfinite", "safety_failure")): raise NativeReferenceGateError("fidelity review contains an invalid outcome")
        expected_evidence = execution.get("attempt_evidence_sha256")
        if not isinstance(expected_evidence, dict) or row.get("evidence_sha256") != expected_evidence.get(row.get("attempt_id")):
            raise NativeReferenceGateError("fidelity review evidence is not bound to execution artifacts")
    publication = _object(publication, "publication receipt")
    if set(publication) != {"schema_version", "kind", "execution_receipt_sha256", "repository", "remote_prefix", "immutable_revision", "bundle_manifest_sha256", "published_unix_seconds", "readback_verified"} or publication.get("schema_version") != 2 or publication.get("kind") != "lehome_native_reference_hf_readback_v2" or publication.get("execution_receipt_sha256") != execution_sha or publication.get("repository") != PUBLIC_CACHE_REPOSITORY or not isinstance(publication.get("remote_prefix"), str) or re.fullmatch(r"reference-checks/native-[0-9a-f]{16}", publication["remote_prefix"]) is None or publication.get("readback_verified") is not True or not isinstance(publication.get("immutable_revision"), str) or _COMMIT.fullmatch(publication["immutable_revision"]) is None or type(publication.get("published_unix_seconds")) is not int or publication["published_unix_seconds"] <= 0: raise NativeReferenceGateError("publication receipt is not a readback-verified immutable upload")
    _digest(publication.get("bundle_manifest_sha256"), "publication bundle manifest")
    stopped = _object(stopped, "stopped VM receipt")
    stopped = validate_provider_observation(stopped, expected_state="STOPPED")
    if stopped.get("vm_id") != identity["vm_id"] or stopped.get("disk_id") != identity["disk_id"] or stopped.get("provider_source_image_id") != identity["provider_source_image_id"]:
        raise NativeReferenceGateError("stopped VM receipt is not bound to the execution identity")
    captured = stopped["captured_unix_seconds"]
    published = publication["published_unix_seconds"]
    if captured < published:
        raise NativeReferenceGateError("stopped VM receipt is not post-publication")
    if not 0 <= time.time() - captured <= 900:
        raise NativeReferenceGateError("stopped VM receipt is not fresh")
    return {"schema_version": 1, "kind": "lehome_native_reference_gate_final_receipt_v1", "status": "passed", "execution_receipt_sha256": execution_sha, "fidelity_review_sha256": canonical_sha256(fidelity), "publication_receipt_sha256": canonical_sha256(publication), "stopped_vm_receipt_sha256": canonical_sha256(stopped), "identity": identity}


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        if path.is_symlink() or not path.is_file(): raise OSError("unsafe")
        return _object(json.loads(path.read_bytes()), label)
    except (OSError, UnicodeError, json.JSONDecodeError, NativeReferenceGateError) as error: raise NativeReferenceGateError(f"{label} is invalid") from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify-execution"); verify.add_argument("--result", type=Path, required=True); verify.add_argument("--bundle-root", type=Path, required=True); verify.add_argument("--receipt", type=Path, required=True)
    compile_stage = commands.add_parser("compile-stage"); compile_stage.add_argument("--bundle-root", type=Path, required=True); compile_stage.add_argument("--stage", type=int, required=True); compile_stage.add_argument("--category", required=True); compile_stage.add_argument("--garment", required=True); compile_stage.add_argument("--identity", type=Path, required=True)
    finalize = commands.add_parser("finalize"); finalize.add_argument("--execution", type=Path, required=True); finalize.add_argument("--fidelity", type=Path, required=True); finalize.add_argument("--publication", type=Path, required=True); finalize.add_argument("--stopped", type=Path, required=True); finalize.add_argument("--receipt", type=Path, required=True)
    provider = commands.add_parser("capture-provider"); provider.add_argument("--state", choices=("RUNNING", "STOPPED"), required=True); provider.add_argument("--receipt", type=Path, required=True)
    bind_provider = commands.add_parser("bind-provider-receipt"); bind_provider.add_argument("--state", choices=("RUNNING", "STOPPED"), required=True); bind_provider.add_argument("--input", type=Path, required=True); bind_provider.add_argument("--receipt", type=Path, required=True)
    runtime_image = commands.add_parser("capture-runtime-image"); runtime_image.add_argument("--receipt", type=Path, required=True)
    bind_runtime_image = commands.add_parser("bind-runtime-image-receipt"); bind_runtime_image.add_argument("--input", type=Path, required=True); bind_runtime_image.add_argument("--receipt", type=Path, required=True)
    asset_bindings = commands.add_parser("validate-asset-bindings"); asset_bindings.add_argument("--assets-root", type=Path, required=True); asset_bindings.add_argument("--runtime-repo-root", type=Path, required=True)
    prepare_mountpoints = commands.add_parser("prepare-runtime-mountpoints"); prepare_mountpoints.add_argument("--runtime-root", type=Path, required=True)
    compatibility = commands.add_parser("prepare-checkpoint-compatibility"); compatibility.add_argument("--checkpoint-root", type=Path, required=True); compatibility.add_argument("--sanitized-config-root", type=Path, required=True); compatibility.add_argument("--receipt", type=Path, required=True)
    peft_overlay = commands.add_parser("validate-peft-overlay")
    prepare_peft = commands.add_parser("prepare-peft-overlay"); prepare_peft.add_argument("--receipt", type=Path, required=True)
    flash_attention_overlay = commands.add_parser("validate-flash-attention-overlay")
    prepare_flash_attention = commands.add_parser("prepare-flash-attention-overlay"); prepare_flash_attention.add_argument("--receipt", type=Path, required=True)
    cache = commands.add_parser("fetch-cache-manifest"); cache.add_argument("--revision", required=True); cache.add_argument("--path", required=True); cache.add_argument("--receipt", type=Path); cache.add_argument("--checkpoint-tree-sha256"); cache.add_argument("--metadata-tree-sha256"); cache.add_argument("--assets-tree-sha256")
    publish = commands.add_parser("publish-bundle"); publish.add_argument("--bundle-root", type=Path, required=True); publish.add_argument("--execution", type=Path, required=True); publish.add_argument("--token-file", type=Path, required=True); publish.add_argument("--receipt", type=Path, required=True)
    publish_cache = commands.add_parser("publish-cache-manifest"); publish_cache.add_argument("--manifest", type=Path, required=True); publish_cache.add_argument("--token-file", type=Path, required=True); publish_cache.add_argument("--receipt", type=Path, required=True)
    cache_auth = commands.add_parser("authenticate-cache"); cache_auth.add_argument("--metadata-root", type=Path, required=True); cache_auth.add_argument("--assets-root", type=Path, required=True); cache_auth.add_argument("--manifest", type=Path, required=True)
    host_runtime = commands.add_parser("probe-host-runtime"); host_runtime.add_argument("--source-root", type=Path, required=True); host_runtime.add_argument("--isaaclab-root", type=Path, required=True); host_runtime.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "compile-stage": receipt = compile_native_stage(args.bundle_root, stage=args.stage, category=args.category, garment=args.garment, identity=_read_json(args.identity, "identity"))
        elif args.command == "verify-execution": receipt = verify_native_reference_result(_read_json(args.result, "result"), bundle_root=args.bundle_root); _write_exclusive(args.receipt, receipt)
        elif args.command == "finalize": receipt = finalize_native_reference_gate(_read_json(args.execution, "execution receipt"), _read_json(args.fidelity, "fidelity review"), _read_json(args.publication, "publication receipt"), _read_json(args.stopped, "stopped VM receipt")); _write_exclusive(args.receipt, receipt)
        elif args.command == "capture-provider":
            with _repository_package_imports():
                from scripts.finalize_simple_curriculum_collection import SubprocessNebiusProvider
                receipt = capture_provider_observation(SubprocessNebiusProvider(), expected_state=args.state)
            _write_exclusive(args.receipt, receipt)
        elif args.command == "bind-provider-receipt":
            raw = args.input.read_bytes(); receipt = validate_provider_observation(_read_json(args.input, "provider observation"), expected_state=args.state); _write_bytes_exclusive(args.receipt, raw)
        elif args.command == "capture-runtime-image":
            receipt = capture_runtime_image_observation(); _write_exclusive(args.receipt, receipt)
        elif args.command == "bind-runtime-image-receipt":
            raw = _read_regular_bytes(args.input, "runtime image observation")
            receipt = validate_runtime_image_observation(_read_json(args.input, "runtime image observation"))
            _write_bytes_exclusive(args.receipt, raw)
        elif args.command == "validate-asset-bindings":
            receipt = validate_runtime_asset_bindings(args.assets_root, args.runtime_repo_root)
        elif args.command == "prepare-runtime-mountpoints":
            receipt = prepare_runtime_asset_mountpoints(args.runtime_root)
        elif args.command == "prepare-checkpoint-compatibility":
            receipt = prepare_checkpoint_compatibility(args.checkpoint_root, args.sanitized_config_root, args.receipt)
        elif args.command == "validate-peft-overlay":
            receipt = inspect_peft_overlay()
        elif args.command == "prepare-peft-overlay":
            receipt = prepare_peft_overlay(args.receipt)
        elif args.command == "validate-flash-attention-overlay":
            receipt = inspect_flash_attention_overlay()
        elif args.command == "prepare-flash-attention-overlay":
            receipt = prepare_flash_attention_overlay(args.receipt)
        elif args.command == "publish-bundle":
            with _repository_package_imports():
                from scripts.publish_simple_curriculum_collection import HuggingFacePublicDatasetTransport, _load_token
                receipt = publish_native_reference_bundle(args.bundle_root, _read_json(args.execution, "execution receipt"), token=_load_token(args.token_file), transport=HuggingFacePublicDatasetTransport())
            _write_exclusive(args.receipt, receipt)
        elif args.command == "publish-cache-manifest":
            with _repository_package_imports():
                from scripts.publish_simple_curriculum_collection import HuggingFacePublicDatasetTransport, _load_token
                receipt = publish_cache_manifest(args.manifest, token=_load_token(args.token_file), transport=HuggingFacePublicDatasetTransport())
            _write_exclusive(args.receipt, receipt)
        elif args.command == "authenticate-cache":
            receipt = authenticate_canonical_caches(args.metadata_root, args.assets_root, manifest_path=args.manifest)
        elif args.command == "probe-host-runtime":
            receipt = capture_host_runtime(args.source_root, args.isaaclab_root); _write_exclusive(args.receipt, receipt)
        else:
            receipt, raw = fetch_public_cache_manifest(args.revision, args.path)
            bindings = (args.checkpoint_tree_sha256, args.metadata_tree_sha256, args.assets_tree_sha256)
            if any(value is not None for value in bindings):
                if any(value is None for value in bindings): raise NativeReferenceGateError("cache manifest bindings must be complete")
                validate_public_cache_bindings(receipt, checkpoint_tree_sha256=args.checkpoint_tree_sha256, metadata_tree_sha256=args.metadata_tree_sha256, assets_tree_sha256=args.assets_tree_sha256)
            if args.receipt is not None: _write_bytes_exclusive(args.receipt, raw)
    except NativeReferenceGateError as error:
        if argv is not None: raise SystemExit(str(error)) from None
        print(f"error: {error}", file=sys.stderr); return 2
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 3 if args.command == "verify-execution" and receipt.get("status") == "evaluator_compatibility_stop" else 0


if __name__ == "__main__": raise SystemExit(main())
