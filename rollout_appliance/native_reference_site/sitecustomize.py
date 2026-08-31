"""Redirect pinned public-evaluator logging without editing pinned source."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def _fatal(message: str) -> None:
    print(f"native reference site error: {message}", file=sys.stderr, flush=True)
    os._exit(72)


raw_log_root = os.environ.get("LEHOME_NATIVE_REFERENCE_LOG_PROJECT_ROOT", "")
raw_source_root = os.environ.get("LEHOME_NATIVE_REFERENCE_SOURCE_ROOT", "")
if not raw_log_root or not raw_source_root:
    _fatal("log and source roots are required")
if ".." in Path(raw_log_root).parts or not Path(raw_log_root).is_absolute():
    _fatal("log project root is unsafe")
try:
    source_root = Path(raw_source_root).resolve(strict=True)
    requested_log_root = Path(raw_log_root)
    if requested_log_root.is_symlink():
        _fatal("log project root is a symlink")
    requested_log_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    log_root = requested_log_root.resolve(strict=True)
except OSError:
    _fatal("log project root is unavailable")
if log_root == source_root or source_root in log_root.parents:
    _fatal("log project root is inside pinned source")

try:
    import lehome.utils.logger as _lehome_logger
except Exception:
    _fatal("pinned LeHome logger cannot be imported")


def _external_project_root() -> Path:
    return log_root


_lehome_logger.get_project_root = _external_project_root


raw_cloth_fidelity_evidence = os.environ.get(
    "LEHOME_NATIVE_CLOTH_FIDELITY_EVIDENCE", ""
)
if raw_cloth_fidelity_evidence:
    fidelity_path = Path(raw_cloth_fidelity_evidence)
    if (
        not fidelity_path.is_absolute()
        or ".." in fidelity_path.parts
        or fidelity_path.exists()
        or fidelity_path.is_symlink()
    ):
        _fatal("cloth fidelity evidence path is unsafe or already exists")
    try:
        import gymnasium as _gymnasium
        from cloth_fidelity import install_cloth_fidelity_monitor_on_env

        _original_gym_make = _gymnasium.make
        _monitor_installed = False

        def _monitored_gym_make(*args: object, **kwargs: object):
            global _monitor_installed
            created = _original_gym_make(*args, **kwargs)
            if _monitor_installed:
                _fatal("focused evaluator created more than one environment")
            install_cloth_fidelity_monitor_on_env(created.unwrapped, fidelity_path)
            _monitor_installed = True
            return created

        _gymnasium.make = _monitored_gym_make
    except Exception as error:
        _fatal(f"cloth fidelity monitor installation failed: {error}")


raw_checkpoint_root = os.environ.get("LEHOME_NATIVE_REFERENCE_CHECKPOINT_ROOT", "")
raw_sanitized_root = os.environ.get("LEHOME_NATIVE_REFERENCE_SANITIZED_CONFIG_ROOT", "")
raw_compatibility_receipt = os.environ.get(
    "LEHOME_NATIVE_REFERENCE_CHECKPOINT_COMPATIBILITY_RECEIPT", ""
)
compatibility_values = (raw_checkpoint_root, raw_sanitized_root, raw_compatibility_receipt)
if any(compatibility_values) and not all(compatibility_values):
    _fatal("checkpoint compatibility environment is incomplete")
if all(compatibility_values):
    try:
        from checkpoint_compatibility import (
            install_checkpoint_config_view,
            install_cpu_action_normalization_boundary,
        )

        install_checkpoint_config_view(
            Path(raw_checkpoint_root),
            Path(raw_sanitized_root),
            Path(raw_compatibility_receipt),
        )
        from lerobot.processor.core import TransitionKey
        from scripts.eval_policy.lerobot_policy import LeRobotPolicy

        install_cpu_action_normalization_boundary(
            LeRobotPolicy, TransitionKey.ACTION
        )
    except Exception as error:
        _fatal(f"checkpoint compatibility loader failed: {error}")
