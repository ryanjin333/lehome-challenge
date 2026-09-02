"""Pure-stdlib validator for the exact Task 1 N1.5 training-output receipt."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import subprocess
import sys
from typing import Mapping
import zipfile


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
_PEFT_OVERLAY_RECEIPT = {
    "schema_version": 1,
    "kind": "lehome_native_reference_peft_overlay_v1",
    "wheel_path": "/mnt/lehome/reference-native/dependencies/peft-0.18.1-py3-none-any.whl",
    "wheel_filename": "peft-0.18.1-py3-none-any.whl",
    "wheel_url": "https://files.pythonhosted.org/packages/b3/14/b4e3f574acf349ae6f61f9c000a77f97a3b315b4bb6ad03791e79ae4a568/peft-0.18.1-py3-none-any.whl",
    "wheel_size": 556960,
    "wheel_sha256": "0bf06847a3551e3019fc58c440cffc9a6b73e6e2962c95b52e224f77bbdb50f1",
    "distribution_name": "peft",
    "peft_version": "0.18.1",
    "peft_origin": "/mnt/lehome/reference-native/dependencies/peft-0.18.1-py3-none-any.whl/peft/__init__.py",
    "required_symbols": ["LoraConfig", "get_peft_model"],
}


class TrainingIdentityError(RuntimeError):
    """The Task 1 receipt or the artifact tree it authenticates is invalid."""


def contract_identity(contract: object) -> dict[str, object]:
    """Return the exact manifest form used by Task 1's execution receipt."""
    if isinstance(contract, Mapping):
        return dict(contract)
    keys = (
        "source_repository", "source_revision", "source_tree", "dependency_lock_sha256",
        "lerobot_wheel_sha256", "lerobot_package_file_count", "lerobot_package_tree_sha256",
        "base_model_metadata_count", "base_model_metadata_sha256", "dataset_metadata_count",
        "dataset_metadata_sha256", "base_model_repository", "base_model_revision",
        "dataset_repository", "dataset_revision", "trusted_source_files", "vm_id", "disk_id",
        "python_version", "lerobot_version", "training_command", "training",
    )
    try:
        value = {key: getattr(contract, key) for key in keys}
    except AttributeError:
        raise TrainingIdentityError("Task 1 contract is incomplete") from None
    value["trusted_source_files"] = dict(value["trusted_source_files"])
    value["training_command"] = list(value["training_command"])
    value["training"] = dict(value["training"])
    return value


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


def _json(path: Path, label: str) -> dict[str, object]:
    raw = _regular(path, label).read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError):
        raise TrainingIdentityError(f"{label} is invalid JSON") from None
    if not isinstance(value, dict) or raw != _canonical(value):
        raise TrainingIdentityError(f"{label} is not canonical JSON")
    return value


def _tree_identity(files: Mapping[str, bytes]) -> tuple[int, str]:
    payload = b"".join(
        relative.encode("utf-8")
        + b"\0"
        + hashlib.sha256(content).hexdigest().encode("ascii")
        + b"\n"
        for relative, content in sorted(files.items())
    )
    return len(files), hashlib.sha256(payload).hexdigest()


def _wheel_identity(payload: bytes) -> tuple[int, str, str]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            files: dict[str, bytes] = {}
            for info in archive.infolist():
                if info.is_dir() or not info.filename.startswith("lerobot/"):
                    continue
                relative = info.filename.removeprefix("lerobot/")
                pure = PurePosixPath(relative)
                file_type = (info.external_attr >> 16) & 0o170000
                if (
                    not relative or pure.is_absolute()
                    or any(part in {"", ".", ".."} for part in pure.parts)
                    or relative in files or file_type == stat.S_IFLNK
                ):
                    raise TrainingIdentityError("LeRobot wheel package entry is unsafe")
                files[relative] = archive.read(info)
            metadata = archive.read("lerobot-0.4.3.dist-info/METADATA").decode("utf-8")
    except (OSError, UnicodeError, zipfile.BadZipFile, KeyError):
        raise TrainingIdentityError("LeRobot wheel is invalid") from None
    if not files:
        raise TrainingIdentityError("LeRobot wheel contains no package tree")
    count, tree = _tree_identity(files)
    return count, tree, metadata


def _installed_tree(root: Path) -> tuple[int, str]:
    package = _directory(root, "installed LeRobot package root")
    files: dict[str, bytes] = {}
    for path in sorted(package.rglob("*")):
        relative = path.relative_to(package).as_posix()
        metadata = path.lstat()
        if path.is_symlink():
            raise TrainingIdentityError("installed LeRobot tree contains an unsafe symlink")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise TrainingIdentityError("installed LeRobot tree contains an unsafe entry")
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        files[relative] = path.read_bytes()
    if not files:
        raise TrainingIdentityError("installed LeRobot package tree is empty")
    return _tree_identity(files)


def _hub_metadata_identity(rows: object) -> tuple[int, str]:
    if not isinstance(rows, list) or not rows:
        raise TrainingIdentityError("Hub sibling metadata is empty or invalid")
    validated: list[dict[str, object]] = []
    paths: set[str] = set()
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "blob_id", "size", "lfs_sha256"}
            or not isinstance(row.get("path"), str)
            or PurePosixPath(row["path"]).is_absolute()
            or any(part in {"", ".", ".."} for part in PurePosixPath(row["path"]).parts)
            or row["path"] in paths
            or re.fullmatch(r"[0-9a-f]{40}", str(row.get("blob_id"))) is None
            or type(row.get("size")) is not int or row["size"] < 0
            or (row.get("lfs_sha256") is not None and _SHA256.fullmatch(str(row["lfs_sha256"])) is None)
        ):
            raise TrainingIdentityError("Hub sibling metadata is invalid")
        paths.add(row["path"])
        validated.append(row)
    payload = "".join(
        f"{row['path']}\t{row['blob_id']}\t{row['size']}\t{row['lfs_sha256'] or ''}\n"
        for row in sorted(validated, key=lambda value: str(value["path"]))
    ).encode("utf-8")
    return len(validated), hashlib.sha256(payload).hexdigest()


def validate_training_identity_receipt(
    receipt_path: Path,
    *,
    expected_contract: object,
    expected_pretrained_root: Path | None = None,
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

    contract = contract_identity(expected_contract)
    required_contract_keys = {
        "source_repository", "source_revision", "source_tree", "dependency_lock_sha256",
        "lerobot_wheel_sha256", "lerobot_package_file_count", "lerobot_package_tree_sha256",
        "base_model_metadata_count", "base_model_metadata_sha256", "dataset_metadata_count",
        "dataset_metadata_sha256", "base_model_repository", "base_model_revision",
        "dataset_repository", "dataset_revision", "trusted_source_files", "vm_id", "disk_id",
        "python_version", "lerobot_version", "training_command", "training",
    }
    if set(contract) != required_contract_keys:
        raise TrainingIdentityError("Task 1 contract schema mismatch")
    try:
        train_config = json.loads((pretrained_root / "train_config.json").read_text(encoding="utf-8"))
        policy_config = json.loads((pretrained_root / "config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise TrainingIdentityError("upstream checkpoint configuration is invalid") from None
    if (
        not isinstance(train_config, dict)
        or train_config.get("steps") != int(contract["training"]["steps"])
        or train_config.get("batch_size") != int(contract["training"]["batch_size"])
        or not isinstance(policy_config, dict) or policy_config.get("type") != "groot"
    ):
        raise TrainingIdentityError("upstream checkpoint configuration does not prove the recipe")
    training_state = _directory(checkpoint_root / "training_state", "upstream training state")
    required_state = {
        "optimizer_param_groups.json", "optimizer_state.safetensors", "rng_state.safetensors",
        "scheduler_state.json", "training_step.json",
    }
    state_names = {
        path.name for path in training_state.iterdir() if path.is_file() and not path.is_symlink()
    }
    if not required_state.issubset(state_names):
        raise TrainingIdentityError("upstream training-state structure is incomplete")
    step_receipt = _json(training_state / "training_step.json", "training-step evidence")
    if set(step_receipt) != {"step"} or step_receipt.get("step") != 12000:
        raise TrainingIdentityError("training-step evidence does not prove step 12000")

    source_copy = _regular(training_root / "evidence/source-receipt.json", "training source receipt")
    snapshots_copy = _regular(
        training_root / "evidence/resolved-snapshots-receipt.json",
        "training resolved-snapshot receipt",
    )
    if (
        _sha256_file(source_copy) != receipt["source_receipt_sha256"]
        or _sha256_file(snapshots_copy) != receipt["resolved_snapshots_receipt_sha256"]
    ):
        raise TrainingIdentityError("training source evidence receipt mismatch")
    source_evidence = _json(source_copy, "training source receipt")
    if (
        set(source_evidence) != {"schema_version", "kind", "repository", "revision", "tree", "files"}
        or source_evidence.get("schema_version") != 1
        or source_evidence.get("kind") != "lehome_public_n15_source_v1"
        or source_evidence.get("repository") != contract["source_repository"]
        or source_evidence.get("revision") != contract["source_revision"]
        or source_evidence.get("tree") != contract["source_tree"]
        or source_evidence.get("files") != contract["trusted_source_files"]
    ):
        raise TrainingIdentityError("training source evidence identity mismatch")
    snapshots_evidence = _json(snapshots_copy, "training resolved-snapshot receipt")
    if (
        set(snapshots_evidence) != {"schema_version", "kind", "base_model", "dataset", "vm_id", "disk_id"}
        or snapshots_evidence.get("schema_version") != 1
        or snapshots_evidence.get("kind") != "lehome_public_n15_resolved_snapshots_v1"
        or snapshots_evidence.get("vm_id") != contract["vm_id"]
        or snapshots_evidence.get("disk_id") != contract["disk_id"]
    ):
        raise TrainingIdentityError("training resolved-snapshot identity mismatch")
    for label, repository_key, revision_key, count_key, digest_key in (
        ("base_model", "base_model_repository", "base_model_revision", "base_model_metadata_count", "base_model_metadata_sha256"),
        ("dataset", "dataset_repository", "dataset_revision", "dataset_metadata_count", "dataset_metadata_sha256"),
    ):
        snapshot = snapshots_evidence.get(label)
        expected_keys = {"repository", "revision", "root", "siblings"} | (
            {"snapshot_root"} if label == "dataset" else set()
        )
        if (
            not isinstance(snapshot, Mapping) or set(snapshot) != expected_keys
            or snapshot.get("repository") != contract[repository_key]
            or snapshot.get("revision") != contract[revision_key]
            or not isinstance(snapshot.get("root"), str) or not Path(snapshot["root"]).is_absolute()
            or (label == "dataset" and (not isinstance(snapshot.get("snapshot_root"), str) or not Path(snapshot["snapshot_root"]).is_absolute()))
            or _hub_metadata_identity(snapshot.get("siblings"))
            != (contract[count_key], contract[digest_key])
        ):
            raise TrainingIdentityError(f"training {label} snapshot evidence mismatch")
    execution = _json(
        training_root / "evidence/execution-manifest.json", "training execution manifest"
    )
    if set(execution) != {"schema_version", "kind", "contract", "inputs", "execution"}:
        raise TrainingIdentityError("training execution manifest schema mismatch")
    inputs = execution.get("inputs")
    command = execution.get("execution")
    input_keys = {
        "base_model_root", "hub_cache_root", "dataset_root", "source_receipt",
        "source_receipt_sha256", "source_tree", "resolved_snapshots_receipt",
        "resolved_snapshots_receipt_sha256", "base_model_metadata_sha256",
        "dataset_metadata_sha256",
    }
    if (
        execution.get("schema_version") != 1
        or execution.get("kind") != "lehome_public_n15_training_execution_v1"
        or execution.get("contract") != contract
        or not isinstance(inputs, Mapping) or set(inputs) != input_keys
        or inputs.get("source_receipt_sha256") != receipt["source_receipt_sha256"]
        or inputs.get("resolved_snapshots_receipt_sha256")
        != receipt["resolved_snapshots_receipt_sha256"]
        or inputs.get("source_tree") != contract["source_tree"]
        or inputs.get("base_model_metadata_sha256") != contract["base_model_metadata_sha256"]
        or inputs.get("dataset_metadata_sha256") != contract["dataset_metadata_sha256"]
        or any(not isinstance(inputs.get(key), str) or not Path(inputs[key]).is_absolute()
               for key in ("base_model_root", "hub_cache_root", "dataset_root", "source_receipt", "resolved_snapshots_receipt"))
        or not isinstance(command, Mapping)
        or set(command) != {"cwd", "argv", "shell_argv", "env"}
        or not isinstance(command.get("cwd"), str) or not Path(command["cwd"]).is_absolute()
        or command.get("argv") != contract["training_command"]
        or command.get("shell_argv") != shlex.join(contract["training_command"])
        or command.get("env") != {
            "HF_HUB_CACHE": inputs["hub_cache_root"],
            "HF_HUB_OFFLINE": "1",
            "PYTHONPATH": _PEFT_OVERLAY_RECEIPT["wheel_path"],
        }
    ):
        raise TrainingIdentityError("training execution manifest mismatch")

    peft_overlay = _json(
        training_root / "evidence/peft-overlay-receipt.json", "training PEFT overlay receipt"
    )
    if peft_overlay != _PEFT_OVERLAY_RECEIPT:
        raise TrainingIdentityError("training PEFT overlay identity mismatch")

    lock = _regular(training_root / "evidence/uv.lock", "training dependency lock")
    if _sha256_file(lock) != contract["dependency_lock_sha256"]:
        raise TrainingIdentityError("training dependency lock mismatch")
    runtime = _json(training_root / "evidence/runtime-receipt.json", "training runtime receipt")
    runtime_keys = {
        "schema_version", "kind", "python_executable", "upstream_lerobot_wheel_path",
        "upstream_lerobot_wheel_sha256", "compatibility_wheel_path",
        "compatibility_wheel_sha256", "compatibility_wheel_receipt_path",
        "compatibility_wheel_receipt_sha256", "lerobot_package_root",
        "dependency_lock_path", "dependency_lock_sha256", "scheduler",
    }
    upstream_wheel = _regular(
        training_root / "evidence/upstream/lerobot-0.4.3-py3-none-any.whl",
        "upstream training LeRobot wheel",
    )
    wheel = _regular(
        training_root / "evidence/compatibility/lerobot-0.4.3-py3-none-any.whl",
        "compatible training LeRobot wheel",
    )
    compatibility_receipt_file = _regular(
        training_root / "evidence/compatibility/lerobot-compatibility-receipt.json",
        "training LeRobot compatibility receipt",
    )
    compatibility_receipt = _json(
        compatibility_receipt_file,
        "training LeRobot compatibility receipt",
    )
    if (
        set(runtime) != runtime_keys
        or runtime.get("schema_version") != 1
        or runtime.get("kind") != "lehome_public_n15_training_runtime_v1"
        or runtime.get("upstream_lerobot_wheel_path") != str(upstream_wheel)
        or runtime.get("upstream_lerobot_wheel_sha256") != contract["lerobot_wheel_sha256"]
        or runtime.get("compatibility_wheel_path") != str(wheel)
        or runtime.get("compatibility_wheel_receipt_path") != str(compatibility_receipt_file)
        or runtime.get("compatibility_wheel_receipt_sha256") != _sha256_file(compatibility_receipt_file)
        or runtime.get("dependency_lock_path") != str(lock)
        or runtime.get("dependency_lock_sha256") != contract["dependency_lock_sha256"]
        or not isinstance(runtime.get("python_executable"), str)
        or not isinstance(runtime.get("lerobot_package_root"), str)
        or runtime.get("scheduler") != {
            "num_warmup_steps": 600, "num_decay_steps": 12000,
            "peak_lr": 2e-4, "decay_lr": 2e-5,
        }
    ):
        raise TrainingIdentityError("training runtime identity mismatch")
    upstream_bytes = upstream_wheel.read_bytes()
    if hashlib.sha256(upstream_bytes).hexdigest() != contract["lerobot_wheel_sha256"]:
        raise TrainingIdentityError("upstream training LeRobot wheel digest mismatch")
    upstream_count, upstream_tree, upstream_metadata = _wheel_identity(upstream_bytes)
    if (
        upstream_count != contract["lerobot_package_file_count"]
        or upstream_tree != contract["lerobot_package_tree_sha256"]
        or "Name: lerobot\n" not in upstream_metadata
        or f"Version: {contract['lerobot_version']}\n" not in upstream_metadata
    ):
        raise TrainingIdentityError("upstream training LeRobot wheel identity mismatch")
    source_root = Path(__file__).resolve().parents[2] / "source/lehome"
    if not source_root.is_dir():
        raise TrainingIdentityError("strong LeRobot compatibility verifier is unavailable")
    sys.path.insert(0, str(source_root))
    try:
        from lehome.n15_reproduction import (  # type: ignore[import-not-found]
            ReproductionError,
            compatibility_wheel_identity,
        )

        compatibility_wheel_identity(
            wheel=wheel,
            receipt=compatibility_receipt_file,
            upstream_wheel=upstream_wheel,
            expected_upstream_sha256=str(contract["lerobot_wheel_sha256"]),
        )
    except (ImportError, ReproductionError) as error:
        raise TrainingIdentityError("strong LeRobot compatibility verification failed") from error
    compatibility_keys = {
        "schema_version", "kind", "upstream_wheel_sha256", "upstream_package_file_count",
        "upstream_package_tree_sha256", "transformation", "changed_package_files",
        "derived_wheel_sha256", "derived_package_file_count", "derived_package_tree_sha256",
        "distribution",
    }
    if (
        set(compatibility_receipt) != compatibility_keys
        or compatibility_receipt.get("schema_version") != 1
        or compatibility_receipt.get("kind") != "lehome_public_n15_lerobot_compatibility_wheel_v1"
        or compatibility_receipt.get("upstream_wheel_sha256") != contract["lerobot_wheel_sha256"]
        or compatibility_receipt.get("upstream_package_file_count") != upstream_count
        or compatibility_receipt.get("upstream_package_tree_sha256") != upstream_tree
        or compatibility_receipt.get("transformation") != {
            "kind": "lehome_lerobot_043_groot_scheduler_compatibility_v1",
            "fields": {"num_decay_steps": 10000, "decay_lr_ratio": 0.1},
        }
        or compatibility_receipt.get("distribution") != {"name": "lerobot", "version": contract["lerobot_version"]}
        or not isinstance(compatibility_receipt.get("derived_wheel_sha256"), str)
        or not isinstance(compatibility_receipt.get("derived_package_file_count"), int)
        or not isinstance(compatibility_receipt.get("derived_package_tree_sha256"), str)
        or runtime.get("compatibility_wheel_sha256") != compatibility_receipt.get("derived_wheel_sha256")
    ):
        raise TrainingIdentityError("training LeRobot compatibility identity mismatch")
    wheel_bytes = wheel.read_bytes()
    wheel_count, wheel_tree, metadata = _wheel_identity(wheel_bytes)
    if (
        hashlib.sha256(wheel_bytes).hexdigest() != compatibility_receipt["derived_wheel_sha256"]
        or wheel_count != compatibility_receipt["derived_package_file_count"]
        or wheel_tree != compatibility_receipt["derived_package_tree_sha256"]
        or "Name: lerobot\n" not in metadata
        or f"Version: {contract['lerobot_version']}\n" not in metadata
    ):
        raise TrainingIdentityError("compatible training LeRobot wheel identity mismatch")
    if _installed_tree(Path(runtime["lerobot_package_root"])) != (wheel_count, wheel_tree):
        raise TrainingIdentityError("installed LeRobot package differs from the compatible wheel")
    python = _regular(Path(runtime["python_executable"]), "training Python executable proof")
    try:
        probe = subprocess.run(
            [str(python), "-I", "-c", "import json,sys; print(json.dumps(list(sys.version_info[:3])))"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10, check=False,
        )
        version = json.loads(probe.stdout)
        expected_python = [int(part) for part in str(contract["python_version"]).split(".")]
    except (OSError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired):
        raise TrainingIdentityError("training Python version probe failed") from None
    if (
        probe.returncode != 0 or not isinstance(version, list) or len(version) != 3
        or version[: len(expected_python)] != expected_python
        or any(type(part) is not int for part in version)
    ):
        raise TrainingIdentityError("training Python interpreter version mismatch")
    log = _regular(training_root / "logs/train.log", "training log")
    try:
        log_text = log.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        raise TrainingIdentityError("training log is unreadable") from None
    if (
        not log_text
        or "Checkpoint policy after step 12000" not in log_text
        or "End of training" not in log_text
    ):
        raise TrainingIdentityError("training log lacks step-12000 completion evidence")
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
