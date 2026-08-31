"""Fail-closed, offline contract for the public LeHome GR00T N1.5 recipe.

This module validates identities and renders commands.  It deliberately owns
no cloud lifecycle and never executes training.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import tempfile
from types import MappingProxyType
from typing import Mapping


_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")


class ReproductionError(RuntimeError):
    """An immutable identity or artifact gate failed."""


@dataclass(frozen=True)
class ReproductionContract:
    source_repository: str
    source_revision: str
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
    training_command=("lerobot-train", "--config_path=configs/train_groot.yaml"),
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
    expected_keys = {"schema_version", "kind", "repository", "revision", "files"}
    if set(value) != expected_keys:
        raise ReproductionError("source receipt schema mismatch")
    if (
        value["schema_version"] != 1
        or value["kind"] != "lehome_public_n15_source_v1"
        or value["repository"] != contract.source_repository
        or value["revision"] != contract.source_revision
        or _REVISION.fullmatch(str(value["revision"])) is None
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
    return root, receipt, _sha256_bytes(receipt_bytes)


def _validate_snapshots(
    receipt_path: Path | str,
    vm_id: str,
    disk_id: str,
    checkout: Path,
    contract: ReproductionContract,
) -> tuple[Path, Path, Path, Path, str]:
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
        if not isinstance(candidate, dict) or set(candidate) != {"repository", "revision", "root"}:
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
    expected_dataset = checkout / "Datasets/example/four_types_merged"
    if dataset != expected_dataset:
        raise ReproductionError("dataset staged path mismatch")
    return model, hub_cache, dataset, receipt, _sha256_bytes(receipt_bytes)


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
    model, hub_cache, dataset, snapshots_path, snapshots_sha256 = _validate_snapshots(
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


def render_training(
    *,
    verified: VerifiedInputs,
    output: Path | str,
    contract: ReproductionContract = CONTRACT,
) -> dict[str, object]:
    """Atomically render, but never execute, the exact upstream command."""

    manifest = {
        "schema_version": 1,
        "kind": "lehome_public_n15_training_execution_v1",
        "contract": _contract_json(contract),
        "inputs": {
            "base_model_root": str(verified.base_model_root),
            "hub_cache_root": str(verified.hub_cache_root),
            "dataset_root": str(verified.dataset_root),
            "source_receipt": str(verified.source_receipt),
            "source_receipt_sha256": verified.source_receipt_sha256,
            "resolved_snapshots_receipt": str(verified.resolved_snapshots_receipt),
            "resolved_snapshots_receipt_sha256": verified.resolved_snapshots_receipt_sha256,
        },
        "execution": {
            "cwd": str(verified.checkout),
            "argv": list(contract.training_command),
            "shell_argv": shlex.join(contract.training_command),
            "env": {
                "HF_HUB_CACHE": str(verified.hub_cache_root),
                "HF_HUB_OFFLINE": "1",
            },
        },
    }
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
        metadata = path.lstat()
        if path.is_symlink():
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

    source_copy = artifacts.get("evidence/source-receipt.json")
    snapshots_copy = artifacts.get("evidence/resolved-snapshots-receipt.json")
    if source_copy is None or snapshots_copy is None:
        raise ReproductionError("training source or resolved-snapshot receipt is missing")
    if source_copy.read_bytes() != verified.source_receipt.read_bytes():
        raise ReproductionError("training source receipt mismatch")
    if snapshots_copy.read_bytes() != verified.resolved_snapshots_receipt.read_bytes():
        raise ReproductionError("training resolved-snapshot receipt mismatch")
    log = artifacts.get("logs/train.log")
    if log is None or log.stat().st_size == 0:
        raise ReproductionError("training log is missing or empty")

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
