"""Production image adapter for explicit pinned GR00T launch requests."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

from lehome_train.constants import ISAAC_GROOT_REVISION
from lehome_train.groot.config import FineTuneLaunchConfig
from lehome_train.groot.launch import build_launch, launch_finetune
from lehome_train.io import atomic_write_json
from lehome_train.preflight import reject_secret_bearing_config


_ALLOWED_ROOTS = (Path("/prepared"), Path("/output"), Path("/cache"))


def _exact(arguments: object, fields: set[str], command: str) -> Mapping[str, object]:
    if not isinstance(arguments, Mapping) or set(arguments) != fields:
        raise ValueError(f"production {command} request has an incompatible schema")
    if not all(type(key) is str for key in arguments):
        raise ValueError(f"production {command} request keys must be strings")
    reject_secret_bearing_config(dict(arguments))
    return arguments


def _mounted_path(value: object, label: str, *, must_exist: bool = False) -> Path:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be an absolute mounted path")
    path = Path(value)
    if ".." in path.parts or not path.is_absolute() or path.is_symlink():
        raise ValueError(f"{label} must be an absolute mounted path without aliases")
    resolved = path.resolve(strict=False)
    if not any(resolved == root or root in resolved.parents for root in _ALLOWED_ROOTS):
        raise ValueError(f"{label} must stay beneath /cache, /prepared, or /output")
    if must_exist and not path.is_file():
        raise ValueError(f"{label} must be an existing regular file")
    return path


def _load_config(path_value: object) -> FineTuneLaunchConfig:
    path = _mounted_path(path_value, "launch_config", must_exist=True)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("launch_config is malformed") from None
    if not isinstance(raw, dict):
        raise ValueError("launch_config root must be an object")
    reject_secret_bearing_config(raw)
    try:
        config = FineTuneLaunchConfig(**raw)
    except (TypeError, ValueError):
        raise ValueError("launch_config has an incompatible schema") from None
    for candidate in (
        config.base_model_path,
        config.dataset_path,
        config.modality_config_path,
        config.output_dir,
    ):
        _mounted_path(candidate, "launch_config path")
    return config


def _visible_device() -> str:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not value:
        raise ValueError("CUDA_VISIBLE_DEVICES must identify exactly one GPU")
    try:
        import torch
    except ImportError:
        raise RuntimeError("the pinned PyTorch runtime is unavailable") from None
    if torch.cuda.device_count() != 1:
        raise ValueError("exactly one CUDA GPU must be visible")
    return value


def _status(path_value: object, payload: dict[str, object]) -> dict[str, object]:
    path = _mounted_path(path_value, "status_output")
    atomic_write_json(path, payload)
    return payload


def _run(config: FineTuneLaunchConfig) -> None:
    launch_finetune(
        config,
        visible_devices=_visible_device(),
        environment=os.environ,
        official_checkout=os.environ.get("LEHOME_GROOT_ROOT", "/opt/isaac-groot"),
    )


class ProductionRuntime:
    """Checked operational adapter installed as the image's default factory."""

    def prepare(self, arguments: dict[str, object]) -> dict[str, object]:
        request = _exact(arguments, {"launch_config", "status_output"}, "prepare")
        config = _load_config(request["launch_config"])
        launch = build_launch(
            config,
            visible_devices=_visible_device(),
            environment=os.environ,
            official_checkout=os.environ.get("LEHOME_GROOT_ROOT", "/opt/isaac-groot"),
        )
        for source in (config.base_model_path, config.dataset_path, config.modality_config_path):
            if not Path(source).exists():
                raise ValueError("prepare requires pre-downloaded model and dataset snapshots")
        return _status(
            request["status_output"],
            {
                "schema_version": 1,
                "status": "prepared",
                "isaac_groot_revision": ISAAC_GROOT_REVISION,
                "command": list(launch.command),
            },
        )

    def memorize(self, arguments: dict[str, object]) -> dict[str, object]:
        request = _exact(arguments, {"launch_config", "status_output"}, "memorize")
        config = _load_config(request["launch_config"])
        if config.physical_batch_size != 1 or config.max_steps > 10_000:
            raise ValueError("memorize requires physical batch 1 and at most 10,000 steps")
        _run(config)
        return _status(
            request["status_output"],
            {"schema_version": 1, "status": "memorization_completed", "max_steps": config.max_steps},
        )

    def smoke(self, arguments: dict[str, object]) -> dict[str, object]:
        request = _exact(arguments, {"launch_configs", "status_output"}, "smoke")
        paths = request["launch_configs"]
        if not isinstance(paths, list) or len(paths) != 3:
            raise ValueError("smoke requires three sequential launch configs")
        configs = [_load_config(path) for path in paths]
        if [item.physical_batch_size for item in configs] != [16, 32, 64]:
            raise ValueError("smoke launch configs must use sequential batches 16, 32, and 64")
        if any(item.max_steps != 100 for item in configs):
            raise ValueError("smoke launch configs must run exactly 100 optimizer steps")
        for config in configs:
            _run(config)
        return _status(
            request["status_output"],
            {"schema_version": 1, "status": "smoke_completed", "batches": [16, 32, 64]},
        )

    def train(self, arguments: dict[str, object]) -> dict[str, object]:
        request = _exact(arguments, {"launch_config", "status_output"}, "train")
        config = _load_config(request["launch_config"])
        _run(config)
        return _status(
            request["status_output"],
            {
                "schema_version": 1,
                "status": "training_process_completed",
                "physical_batch_size": config.physical_batch_size,
                "max_steps": config.max_steps,
            },
        )


def create() -> ProductionRuntime:
    """Return the image's operational runtime; construction has no GPU side effects."""

    return ProductionRuntime()
