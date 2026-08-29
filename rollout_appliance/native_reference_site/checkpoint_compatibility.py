"""Narrow inference-only loader view for the pinned public GR00T checkpoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_CONFIG_SHA256 = "b7c385bc57456eae603e929b84defb7e991194aade2aad70785e21991e37614c"
LEROBOT_WHEEL_SHA256 = "b08c1c15b2356bd4e658122deabfb9dacd2d7447de4a4327720991723d4edf2c"
REMOVED_FIELDS = (
    {"key": "decay_lr_ratio", "value": 0.1},
    {"key": "num_decay_steps", "value": 4000},
)


def _regular_bytes(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is unavailable or unsafe")
    return path.read_bytes()


def _resolved_directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"{label} is unavailable or unsafe")
    return path.resolve(strict=True)


def install_checkpoint_config_view(
    checkpoint_root: Path,
    sanitized_config_root: Path,
    receipt_path: Path,
    *,
    expected_config_sha256: str = EXPECTED_CONFIG_SHA256,
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
    }
    if not isinstance(receipt, dict) or set(receipt) != expected_keys:
        raise RuntimeError("compatibility receipt has an unexpected schema")
    raw = _regular_bytes(checkpoint / "config.json", "raw checkpoint config")
    sanitized_raw = _regular_bytes(sanitized / "config.json", "sanitized checkpoint config")
    expected = {
        "schema_version": 1,
        "kind": "lehome_native_reference_checkpoint_compatibility_v1",
        "checkpoint_root": str(checkpoint),
        "raw_config_sha256": hashlib.sha256(raw).hexdigest(),
        "sanitized_config_root": str(sanitized),
        "sanitized_config_sha256": hashlib.sha256(sanitized_raw).hexdigest(),
        "removed_fields": list(REMOVED_FIELDS),
        "lerobot_distribution": "lerobot",
        "lerobot_version": "0.4.3",
        "lerobot_wheel_filename": "lerobot-0.4.3-py3-none-any.whl",
        "lerobot_wheel_sha256": LEROBOT_WHEEL_SHA256,
        "groot_config_missing_fields": ["decay_lr_ratio", "num_decay_steps"],
        "rationale": "inference_only_remove_unsupported_training_scheduler_fields",
        "original_checkpoint_unchanged": True,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError("compatibility receipt does not bind the exact config view")
    if receipt["raw_config_sha256"] != expected_config_sha256:
        raise RuntimeError("raw checkpoint config digest is not approved")

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.groot.configuration_groot import GrootConfig

    fields = getattr(GrootConfig, "__dataclass_fields__", None)
    if not isinstance(fields, dict) or any(row["key"] in fields for row in REMOVED_FIELDS):
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
