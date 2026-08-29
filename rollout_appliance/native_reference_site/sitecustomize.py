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
