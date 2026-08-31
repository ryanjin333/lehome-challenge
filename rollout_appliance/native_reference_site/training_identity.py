"""Pure-stdlib validator for the exact Task 1 N1.5 training-output receipt."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Mapping


_SHA256 = re.compile(r"[0-9a-f]{64}")
_RECEIPT_KEYS = {
    "schema_version",
    "kind",
    "training_root",
    "step",
    "checkpoint_root",
    "checkpoint_files",
    "artifact_count",
    "checksums_sha256",
    "source_receipt_sha256",
    "resolved_snapshots_receipt_sha256",
}
_REQUIRED_PRETRAINED = {
    "config.json",
    "model.safetensors",
    "train_config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_2_groot_pack_inputs_v3.safetensors",
    "policy_postprocessor_step_0_groot_action_unpack_unnormalize_v1.safetensors",
}


class TrainingIdentityError(RuntimeError):
    """The Task 1 receipt or the artifact tree it authenticates is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError):
        raise TrainingIdentityError("training identity receipt is not canonical JSON") from None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise TrainingIdentityError(f"{label} is unavailable or unsafe")
    return path.resolve(strict=True)


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise TrainingIdentityError(f"{label} is unavailable or unsafe")
    return path.resolve(strict=True)


def _manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError):
        raise TrainingIdentityError("training checksum manifest is unreadable") from None
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  (\S+)", line)
        if match is None:
            raise TrainingIdentityError("training checksum manifest is invalid")
        digest, relative = match.groups()
        pure = PurePosixPath(relative)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts) or relative in entries:
            raise TrainingIdentityError("training checksum path is unsafe or duplicated")
        entries[relative] = digest
    if not entries:
        raise TrainingIdentityError("training checksum manifest is empty")
    return entries


def validate_training_identity_receipt(
    receipt_path: Path, *, expected_pretrained_root: Path | None = None
) -> dict[str, object]:
    """Validate the complete Task 1 schema and its checksum-manifest contract."""
    receipt_file = _regular(Path(receipt_path), "candidate training identity receipt")
    raw = receipt_file.read_bytes()
    try:
        receipt = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError):
        raise TrainingIdentityError("candidate training identity receipt is invalid") from None
    if not isinstance(receipt, dict) or set(receipt) != _RECEIPT_KEYS or raw != _canonical(receipt):
        raise TrainingIdentityError("candidate training identity receipt schema is invalid")
    if (
        type(receipt.get("schema_version")) is not int
        or receipt["schema_version"] != 1
        or receipt.get("kind") != "lehome_public_n15_verified_training_output_v1"
        or type(receipt.get("step")) is not int
        or receipt["step"] != 12000
        or type(receipt.get("artifact_count")) is not int
        or receipt["artifact_count"] < 1
        or any(
            _SHA256.fullmatch(str(receipt.get(key))) is None
            for key in (
                "checksums_sha256",
                "source_receipt_sha256",
                "resolved_snapshots_receipt_sha256",
            )
        )
    ):
        raise TrainingIdentityError("candidate training identity receipt values are invalid")
    try:
        training_root = _directory(Path(receipt["training_root"]), "training output root")
        checkpoint_root = _directory(Path(receipt["checkpoint_root"]), "step-12000 checkpoint root")
    except (TypeError, OSError):
        raise TrainingIdentityError("candidate training identity paths are invalid") from None
    if (
        receipt["training_root"] != str(training_root)
        or checkpoint_root != training_root / "checkpoints/012000"
        or receipt["checkpoint_root"] != str(checkpoint_root)
    ):
        raise TrainingIdentityError("candidate training identity path relationship drift")
    pretrained_root = _directory(checkpoint_root / "pretrained_model", "candidate pretrained root")
    if expected_pretrained_root is not None:
        expected = _directory(Path(expected_pretrained_root), "expected candidate pretrained root")
        if pretrained_root != expected:
            raise TrainingIdentityError("candidate pretrained root cross-receipt mismatch")
    last = training_root / "checkpoints/last"
    if not last.is_symlink() or os.readlink(last) != "012000":
        raise TrainingIdentityError("candidate last-checkpoint link is invalid")

    artifacts: dict[str, Path] = {}
    for path in sorted(training_root.rglob("*")):
        relative = path.relative_to(training_root).as_posix()
        metadata = path.lstat()
        if path.is_symlink():
            if relative == "checkpoints/last" and os.readlink(path) == "012000":
                continue
            raise TrainingIdentityError("candidate training output contains an unsafe symlink")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise TrainingIdentityError("candidate training output contains an unsafe entry")
        artifacts[relative] = path.resolve(strict=True)
    checksum_path = _regular(training_root / "checksums.sha256", "training checksum manifest")
    if _sha256_file(checksum_path) != receipt["checksums_sha256"]:
        raise TrainingIdentityError("training checksum manifest digest mismatch")
    expected_artifacts = set(artifacts) - {"checksums.sha256"}
    checksums = _manifest(checksum_path)
    if set(checksums) != expected_artifacts or receipt["artifact_count"] != len(checksums):
        raise TrainingIdentityError("training artifact count or checksum file set mismatch")
    for relative, digest in checksums.items():
        if _sha256_file(artifacts[relative]) != digest:
            raise TrainingIdentityError(f"training artifact checksum mismatch: {relative}")

    checkpoint_files = receipt.get("checkpoint_files")
    expected_checkpoint_files = {
        relative: checksums[relative]
        for relative in sorted(checksums)
        if relative.startswith("checkpoints/012000/")
    }
    if (
        not isinstance(checkpoint_files, dict)
        or not checkpoint_files
        or checkpoint_files != expected_checkpoint_files
        or any(_SHA256.fullmatch(str(value)) is None for value in checkpoint_files.values())
    ):
        raise TrainingIdentityError("candidate checkpoint file manifest is incomplete or invalid")
    pretrained_names = {
        path.relative_to(pretrained_root).as_posix()
        for path in pretrained_root.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if not _REQUIRED_PRETRAINED.issubset(pretrained_names):
        raise TrainingIdentityError("candidate pretrained checkpoint structure is incomplete")
    cross_receipt = {
        "source_receipt_sha256": "evidence/source-receipt.json",
        "resolved_snapshots_receipt_sha256": "evidence/resolved-snapshots-receipt.json",
    }
    if any(checksums.get(relative) != receipt[key] for key, relative in cross_receipt.items()):
        raise TrainingIdentityError("candidate source evidence cross-receipt mismatch")
    checkpoint_files_sha256 = hashlib.sha256(_canonical(checkpoint_files)).hexdigest()
    return {
        "schema_version": 1,
        "kind": receipt["kind"],
        "training_root": str(training_root),
        "step": 12000,
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_files_sha256": checkpoint_files_sha256,
        "checkpoint_file_count": len(checkpoint_files),
        "artifact_count": receipt["artifact_count"],
        "checksums_sha256": receipt["checksums_sha256"],
        "source_receipt_sha256": receipt["source_receipt_sha256"],
        "resolved_snapshots_receipt_sha256": receipt["resolved_snapshots_receipt_sha256"],
        "identity_receipt_sha256": hashlib.sha256(raw).hexdigest(),
    }
