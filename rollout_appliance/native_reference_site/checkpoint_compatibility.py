"""Narrow inference-only loader view for the pinned public GR00T checkpoint."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import stat
from typing import Any

try:
    from rollout_appliance.native_reference_site.training_identity import (
        TrainingIdentityError,
        validate_training_identity_receipt,
    )
except ModuleNotFoundError:  # Native site is loaded as a standalone PYTHONPATH entry.
    from training_identity import (  # type: ignore[no-redef]
        TrainingIdentityError,
        validate_training_identity_receipt,
    )


EXPECTED_CONFIG_SHA256 = "b7c385bc57456eae603e929b84defb7e991194aade2aad70785e21991e37614c"
LEROBOT_WHEEL_SHA256 = "b08c1c15b2356bd4e658122deabfb9dacd2d7447de4a4327720991723d4edf2c"
# Canonical manifest of the exact verified wheel's sorted lerobot/ regular
# files: relative path, NUL, SHA-256 of real bytes, LF. Regular .pyc files are
# the sole runtime exclusion.
LEROBOT_PACKAGE_TREE_SHA256 = "db3b4e18b166d4bb7fb4354cec82a7fbd15bb24230f9d71269a017c774e0852f"
LEROBOT_PACKAGE_FILE_COUNT = 289
REMOVED_FIELDS = (
    {"key": "decay_lr_ratio", "value": 0.1},
    {"key": "num_decay_steps", "value": 4000},
)
CANDIDATE_REMOVED_FIELDS = (
    {"key": "decay_lr_ratio", "value": 0.1},
    {"key": "num_decay_steps", "value": 12000},
)


def install_cpu_action_normalization_boundary(
    policy_class: type[Any], action_key: object
) -> None:
    """Keep the evaluator's dummy action with its CPU normalization statistics."""
    original_prepare = getattr(policy_class, "_prepare_for_preprocessor", None)
    if not callable(original_prepare):
        raise RuntimeError("official LeRobot policy preparation method is unavailable")

    def prepare_with_cpu_action(self: object, observation: object) -> dict[object, object]:
        transition = original_prepare(self, observation)
        if not isinstance(transition, dict) or action_key not in transition:
            raise RuntimeError("official LeRobot transition action is unavailable")
        action = transition[action_key]
        move_to_cpu = getattr(action, "cpu", None)
        if not callable(move_to_cpu):
            raise RuntimeError("official LeRobot transition action is not a tensor")
        cpu_action = move_to_cpu()
        if str(getattr(cpu_action, "device", "")) != "cpu":
            raise RuntimeError("official LeRobot transition action did not move to CPU")
        transition[action_key] = cpu_action
        return transition

    policy_class._prepare_for_preprocessor = prepare_with_cpu_action


def _regular_bytes(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is unavailable or unsafe")
    return path.read_bytes()


def _resolved_directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"{label} is unavailable or unsafe")
    return path.resolve(strict=True)


def _installed_lerobot_identity() -> tuple[Path, str, int]:
    try:
        distribution = importlib.metadata.distribution("lerobot")
        if distribution.version != "0.4.3":
            raise RuntimeError("installed LeRobot version mismatch")
        requested_init = Path(distribution.locate_file("lerobot/__init__.py"))
        if requested_init.is_symlink() or requested_init.parent.is_symlink():
            raise RuntimeError("installed LeRobot distribution path is unsafe")
        distribution_init = requested_init.resolve(strict=True)
    except Exception as error:
        raise RuntimeError("installed LeRobot distribution identity is unavailable") from error
    package_root = _resolved_directory(distribution_init.parent, "installed LeRobot package root")
    digest = hashlib.sha256()
    count = 0
    for path in sorted(package_root.rglob("*")):
        relative = path.relative_to(package_root)
        metadata = path.lstat()
        if path.is_symlink():
            raise RuntimeError("installed LeRobot package tree contains a symlink")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("installed LeRobot package tree contains an unsafe entry")
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
    lerobot = importlib.import_module("lerobot")
    if Path(lerobot.__file__).resolve(strict=True) != distribution_init:
        raise RuntimeError("imported LeRobot package and distribution are incoherent")
    return package_root, digest.hexdigest(), count


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")


def prepare_candidate_checkpoint_config_view(
    checkpoint_root: Path,
    training_identity_receipt: Path,
    sanitized_config_root: Path,
    receipt_path: Path,
    *,
    groot_config_origin: Path | None = None,
    expected_package_tree_sha256: str = LEROBOT_PACKAGE_TREE_SHA256,
    expected_package_file_count: int = LEROBOT_PACKAGE_FILE_COUNT,
) -> dict[str, object]:
    """Create an inference-only view for the exact verified 12K candidate."""
    checkpoint = _resolved_directory(Path(checkpoint_root), "checkpoint root")
    training_path = Path(training_identity_receipt)
    training_raw = _regular_bytes(training_path, "candidate training identity receipt")
    try:
        training_identity = validate_training_identity_receipt(
            training_path, expected_pretrained_root=checkpoint
        )
    except TrainingIdentityError as error:
        raise RuntimeError(f"candidate training identity receipt is invalid: {error}") from None
    raw = _regular_bytes(checkpoint / "config.json", "raw checkpoint config")
    raw_sha = hashlib.sha256(raw).hexdigest()
    try:
        values = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError):
        raise RuntimeError("candidate checkpoint config is invalid") from None
    if (
        not isinstance(values, dict)
        or values.get("num_decay_steps") != 12000
        or type(values.get("num_decay_steps")) is not int
        or values.get("decay_lr_ratio") != 0.1
        or type(values.get("decay_lr_ratio")) is not float
    ):
        raise RuntimeError("candidate checkpoint scheduler identity is invalid")
    sanitized_values = dict(values)
    removed = [
        {"key": key, "value": sanitized_values.pop(key)}
        for key in sorted(("decay_lr_ratio", "num_decay_steps"))
    ]
    if tuple(removed) != CANDIDATE_REMOVED_FIELDS:
        raise RuntimeError("candidate compatibility removed unexpected values")
    sanitized_raw = _canonical_bytes(sanitized_values)
    sanitized = Path(sanitized_config_root)
    if sanitized.exists() or sanitized.is_symlink():
        raise RuntimeError("sanitized checkpoint config root already exists")
    sanitized.mkdir(mode=0o700)
    sanitized = sanitized.resolve(strict=True)
    descriptor = os.open(sanitized / "config.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(sanitized_raw)
        stream.flush()
        os.fsync(stream.fileno())
    package_root, tree_sha, file_count = _installed_lerobot_identity()
    if tree_sha != expected_package_tree_sha256 or file_count != expected_package_file_count:
        raise RuntimeError("installed LeRobot package tree does not match official wheel")
    if groot_config_origin is None:
        from lerobot.policies.groot.configuration_groot import GrootConfig

        groot_config_origin = Path(
            __import__(GrootConfig.__module__, fromlist=["*"]).__file__
        ).resolve(strict=True)
    else:
        groot_config_origin = Path(groot_config_origin).resolve(strict=True)
    receipt = {
        "schema_version": 1,
        "kind": "lehome_native_candidate_checkpoint_compatibility_v1",
        "checkpoint_root": str(checkpoint),
        "raw_config_sha256": raw_sha,
        "sanitized_config_root": str(sanitized),
        "sanitized_config_sha256": hashlib.sha256(sanitized_raw).hexdigest(),
        "removed_fields": removed,
        "lerobot_distribution": "lerobot",
        "lerobot_version": "0.4.3",
        "lerobot_wheel_filename": "lerobot-0.4.3-py3-none-any.whl",
        "lerobot_wheel_sha256": LEROBOT_WHEEL_SHA256,
        "groot_config_origin": str(groot_config_origin),
        "groot_config_missing_fields": ["decay_lr_ratio", "num_decay_steps"],
        "rationale": "inference_only_remove_unsupported_training_scheduler_fields",
        "original_checkpoint_unchanged": True,
        "installed_lerobot_package_root": str(package_root),
        "expected_lerobot_package_tree_sha256": expected_package_tree_sha256,
        "expected_lerobot_package_file_count": expected_package_file_count,
        "installed_lerobot_package_tree_sha256": tree_sha,
        "installed_lerobot_package_file_count": file_count,
        "training_identity_receipt": str(training_path.resolve(strict=True)),
        "training_identity_receipt_sha256": hashlib.sha256(training_raw).hexdigest(),
        "training_identity": training_identity,
    }
    target = Path(receipt_path)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(_canonical_bytes(receipt))
        stream.flush()
        os.fsync(stream.fileno())
    if (checkpoint / "config.json").read_bytes() != raw:
        raise RuntimeError("original checkpoint config changed during normalization")
    return receipt


def install_checkpoint_config_view(
    checkpoint_root: Path,
    sanitized_config_root: Path,
    receipt_path: Path,
    *,
    expected_config_sha256: str | None = None,
    expected_package_tree_sha256: str = LEROBOT_PACKAGE_TREE_SHA256,
    expected_package_file_count: int = LEROBOT_PACKAGE_FILE_COUNT,
) -> None:
    """Patch one exact config load while preserving every downstream original path."""
    checkpoint = _resolved_directory(Path(checkpoint_root), "checkpoint root")
    sanitized = _resolved_directory(Path(sanitized_config_root), "sanitized config root")
    receipt_file = Path(receipt_path)
    receipt = json.loads(_regular_bytes(receipt_file, "compatibility receipt"))
    expected_keys = {
        "schema_version",
        "kind",
        "checkpoint_root",
        "raw_config_sha256",
        "sanitized_config_root",
        "sanitized_config_sha256",
        "removed_fields",
        "lerobot_distribution",
        "lerobot_version",
        "lerobot_wheel_filename",
        "lerobot_wheel_sha256",
        "groot_config_origin",
        "groot_config_missing_fields",
        "rationale",
        "original_checkpoint_unchanged",
        "installed_lerobot_package_root",
        "expected_lerobot_package_tree_sha256",
        "expected_lerobot_package_file_count",
        "installed_lerobot_package_tree_sha256",
        "installed_lerobot_package_file_count",
    }
    candidate = isinstance(receipt, dict) and receipt.get("kind") == "lehome_native_candidate_checkpoint_compatibility_v1"
    if candidate:
        expected_keys |= {
            "training_identity_receipt",
            "training_identity_receipt_sha256",
            "training_identity",
        }
    if not isinstance(receipt, dict) or set(receipt) != expected_keys:
        raise RuntimeError("compatibility receipt has an unexpected schema")
    raw = _regular_bytes(checkpoint / "config.json", "raw checkpoint config")
    sanitized_raw = _regular_bytes(sanitized / "config.json", "sanitized checkpoint config")
    package_root, installed_tree_sha256, installed_file_count = _installed_lerobot_identity()
    removed_fields = list(CANDIDATE_REMOVED_FIELDS if candidate else REMOVED_FIELDS)
    expected = {
        "schema_version": 1,
        "kind": (
            "lehome_native_candidate_checkpoint_compatibility_v1"
            if candidate
            else "lehome_native_reference_checkpoint_compatibility_v1"
        ),
        "checkpoint_root": str(checkpoint),
        "raw_config_sha256": hashlib.sha256(raw).hexdigest(),
        "sanitized_config_root": str(sanitized),
        "sanitized_config_sha256": hashlib.sha256(sanitized_raw).hexdigest(),
        "removed_fields": removed_fields,
        "lerobot_distribution": "lerobot",
        "lerobot_version": "0.4.3",
        "lerobot_wheel_filename": "lerobot-0.4.3-py3-none-any.whl",
        "lerobot_wheel_sha256": LEROBOT_WHEEL_SHA256,
        "groot_config_missing_fields": ["decay_lr_ratio", "num_decay_steps"],
        "rationale": "inference_only_remove_unsupported_training_scheduler_fields",
        "original_checkpoint_unchanged": True,
        "installed_lerobot_package_root": str(package_root),
        "expected_lerobot_package_tree_sha256": expected_package_tree_sha256,
        "expected_lerobot_package_file_count": expected_package_file_count,
        "installed_lerobot_package_tree_sha256": installed_tree_sha256,
        "installed_lerobot_package_file_count": installed_file_count,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError("compatibility receipt does not bind the exact config view")
    if candidate:
        training_path = Path(str(receipt["training_identity_receipt"]))
        training_raw = _regular_bytes(training_path, "candidate training identity receipt")
        try:
            training_identity = validate_training_identity_receipt(
                training_path, expected_pretrained_root=checkpoint
            )
        except TrainingIdentityError as error:
            raise RuntimeError(f"candidate training identity receipt is invalid: {error}") from None
        if (
            hashlib.sha256(training_raw).hexdigest()
            != receipt["training_identity_receipt_sha256"]
            or training_identity != receipt["training_identity"]
        ):
            raise RuntimeError("candidate training identity does not bind the compatibility view")
        approved_config_sha256 = str(receipt["raw_config_sha256"])
    else:
        approved_config_sha256 = expected_config_sha256 or EXPECTED_CONFIG_SHA256
    if receipt["raw_config_sha256"] != approved_config_sha256:
        raise RuntimeError("raw checkpoint config digest is not approved")
    if (
        installed_tree_sha256 != expected_package_tree_sha256
        or installed_file_count != expected_package_file_count
    ):
        raise RuntimeError("installed LeRobot package tree does not match official wheel")

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.groot.configuration_groot import GrootConfig

    fields = getattr(GrootConfig, "__dataclass_fields__", None)
    if not isinstance(fields, dict) or any(row["key"] in fields for row in removed_fields):
        raise RuntimeError("official GrootConfig field mismatch is not present")
    origin = Path(__import__(GrootConfig.__module__, fromlist=["*"]).__file__).resolve(strict=True)
    if receipt.get("groot_config_origin") != str(origin):
        raise RuntimeError("compatibility receipt GrootConfig origin mismatch")

    original_from_pretrained = PreTrainedConfig.from_pretrained
    consumed = False

    @classmethod
    def exact_from_pretrained(
        cls: type[Any], pretrained_name_or_path: object, *args: object, **kwargs: object
    ) -> object:
        nonlocal consumed
        try:
            requested = Path(pretrained_name_or_path).resolve(strict=True)  # type: ignore[arg-type]
        except (OSError, TypeError) as error:
            raise RuntimeError("checkpoint config load path is unsafe") from error
        if requested != checkpoint:
            raise RuntimeError("checkpoint config loader received an unexpected path")
        if consumed:
            raise RuntimeError("checkpoint config compatibility view was requested more than once")
        consumed = True
        try:
            return original_from_pretrained(sanitized, *args, **kwargs)
        except Exception as error:
            raise RuntimeError("sanitized checkpoint config still failed official parsing") from error

    PreTrainedConfig.from_pretrained = exact_from_pretrained
