"""Lazy checked-JSON boundary for image-provided training runtimes."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Protocol

from lehome_train.checkpoints import load_checkpoint_descriptor
from lehome_train.io import canonical_json_bytes, load_json
from lehome_train.models import ExperimentConfig, SmokeResult
from lehome_train.preflight import reject_secret_bearing_config


RUNTIME_FACTORY_ENV = "LEHOME_TRAIN_RUNTIME_FACTORY"
_GPU_COMMANDS = frozenset({"prepare", "memorize", "smoke", "tune", "train", "continuous-train", "runtime-mixture-train"})


class CommandRuntime(Protocol):
    """GPU command adapter supplied by the accepted Task 12 image."""

    def prepare(self, request: dict[str, object]) -> Mapping[str, object]: ...

    def memorize(self, request: dict[str, object]) -> Mapping[str, object]: ...

    def smoke(self, request: dict[str, object]) -> Mapping[str, object]: ...

    def tune(self, request: dict[str, object]) -> Mapping[str, object]: ...

    def train(self, request: dict[str, object]) -> Mapping[str, object]: ...

    def continuous_train(self, request: dict[str, object]) -> Mapping[str, object]: ...

    def runtime_mixture_train(self, request: dict[str, object]) -> Mapping[str, object]: ...


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("command request contains a duplicate field")
        value[key] = item
    return value


def _read_request(path: str | os.PathLike[str]) -> dict[str, object]:
    request_path = Path(path)
    if not request_path.is_file() or request_path.is_symlink():
        raise ValueError("command request must be an existing regular JSON file")
    try:
        decoded = json.loads(
            request_path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError("command request numbers must be finite")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("command request is malformed") from None
    if not isinstance(decoded, dict):
        raise ValueError("command request root must be an object")
    reject_secret_bearing_config(decoded)
    canonical_json_bytes(decoded)
    return decoded


def read_runtime_request(
    path: str | os.PathLike[str],
    *,
    expected_command: str,
) -> dict[str, object]:
    """Load the strict GPU request envelope and return detached arguments."""

    if expected_command not in _GPU_COMMANDS:
        raise ValueError("runtime command is unsupported")
    decoded = _read_request(path)
    if set(decoded) != {"schema_version", "command", "arguments"}:
        raise ValueError("runtime request has an incompatible schema")
    if decoded["schema_version"] != 1 or decoded["command"] != expected_command:
        raise ValueError("runtime request command identity is incompatible")
    arguments = decoded["arguments"]
    if not isinstance(arguments, dict):
        raise ValueError("runtime request arguments must be an object")
    return dict(arguments)


def load_runtime_adapter(
    factory_spec: str | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> CommandRuntime:
    """Load an explicit `module:factory` without importing it during CLI help."""

    environment = os.environ if environ is None else environ
    resolved = factory_spec or environment.get(RUNTIME_FACTORY_ENV)
    if not isinstance(resolved, str) or not resolved.strip():
        raise RuntimeError(
            "no training runtime factory is configured; Task 12 must set "
            f"{RUNTIME_FACTORY_ENV}=module:factory or pass --runtime-factory"
        )
    module_name, separator, attribute_name = resolved.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("runtime factory must use the module:factory form")
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, attribute_name)
        adapter = factory()
    except Exception:
        raise RuntimeError("training runtime factory could not be loaded") from None
    for command in _GPU_COMMANDS:
        attribute = command.replace("-", "_")
        if not callable(getattr(adapter, attribute, None)):
            raise RuntimeError("training runtime factory returned an incomplete adapter")
    return adapter


def dispatch_runtime_request(
    command: str,
    request_path: str | os.PathLike[str],
    *,
    factory_spec: str | None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Dispatch one checked request to the image's actual command adapter."""

    arguments = read_runtime_request(request_path, expected_command=command)
    adapter = load_runtime_adapter(factory_spec, environ=environ)
    try:
        result = getattr(adapter, command.replace("-", "_"))(arguments)
    except Exception:
        raise RuntimeError(f"{command} runtime adapter failed") from None
    if not isinstance(result, Mapping) or not all(type(key) is str for key in result):
        raise RuntimeError(f"{command} runtime adapter returned an invalid result")
    detached = dict(result)
    reject_secret_bearing_config(detached)
    canonical_json_bytes(detached)
    return detached


def _request_arguments(
    path: str | os.PathLike[str],
    *,
    command: str,
    expected_fields: set[str],
) -> dict[str, object]:
    decoded = _read_request(path)
    if set(decoded) != {"schema_version", "command", "arguments"}:
        raise ValueError(f"{command} request has an incompatible schema")
    if decoded["schema_version"] != 1 or decoded["command"] != command:
        raise ValueError(f"{command} request command identity is incompatible")
    arguments = decoded["arguments"]
    if not isinstance(arguments, dict) or set(arguments) != expected_fields:
        raise ValueError(f"{command} request arguments are incomplete or unknown")
    return dict(arguments)


def _string(arguments: Mapping[str, object], key: str) -> str:
    value = arguments[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"request field {key} must be a non-empty string")
    return value


def _has_symlink_component(path: Path) -> bool:
    """Return whether any existing component aliases another filesystem path."""

    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            return True
    return False


def _require_external_sync_output(experiment_root: str, output: str) -> None:
    """Keep sync evidence outside the mutable tree without path aliases."""

    experiment_path = Path(experiment_root)
    output_path = Path(output)
    if (
        ".." in experiment_path.parts
        or ".." in output_path.parts
        or _has_symlink_component(experiment_path)
        or _has_symlink_component(output_path)
    ):
        raise ValueError(
            "sync output must be outside experiment root without path aliases"
        )
    resolved_experiment = experiment_path.resolve(strict=False)
    resolved_output = output_path.resolve(strict=False)
    if (
        resolved_output == resolved_experiment
        or resolved_experiment in resolved_output.parents
    ):
        raise ValueError("sync output must be outside experiment root")


def execute_report_request(path: str | os.PathLike[str]) -> dict[str, object]:
    """Run the fully local report command from a strict request envelope."""

    from lehome_train.commands.report import (
        build_training_report,
        write_training_report,
    )
    from lehome_train.commands.sync import load_sync_result
    from lehome_train.report_evidence import load_checkpoint_pruning_receipt

    fields = {
        "experiment_config",
        "isaac_groot_revision",
        "smoke_result",
        "checkpoint_descriptors",
        "local_artifact_root",
        "sync_result",
        "pruning_receipts",
        "instance_started_at",
        "generated_at",
        "provider_hourly_price",
        "output",
    }
    arguments = _request_arguments(path, command="report", expected_fields=fields)
    checkpoint_paths = arguments["checkpoint_descriptors"]
    if not isinstance(checkpoint_paths, list) or not checkpoint_paths or not all(
        isinstance(item, str) and item for item in checkpoint_paths
    ):
        raise ValueError("report checkpoint_descriptors must be a non-empty string array")
    receipt_paths = arguments["pruning_receipts"]
    if not isinstance(receipt_paths, list) or not all(
        isinstance(item, str) and item for item in receipt_paths
    ):
        raise ValueError("report pruning_receipts must be a string array")
    sync_result_path = arguments["sync_result"]
    if sync_result_path is not None and (
        not isinstance(sync_result_path, str) or not sync_result_path
    ):
        raise ValueError("report sync_result must be a non-empty string or null")
    price = arguments["provider_hourly_price"]
    if type(price) not in (int, float):
        raise ValueError("report provider_hourly_price must be a number")
    report = build_training_report(
        experiment_config=load_json(
            ExperimentConfig,
            _string(arguments, "experiment_config"),
        ),
        isaac_groot_revision=_string(arguments, "isaac_groot_revision"),
        smoke_result=load_json(SmokeResult, _string(arguments, "smoke_result")),
        checkpoints=tuple(load_checkpoint_descriptor(item) for item in checkpoint_paths),
        local_artifact_root=_string(arguments, "local_artifact_root"),
        sync_evidence=(
            None if sync_result_path is None else load_sync_result(sync_result_path)
        ),
        pruning_receipts=tuple(
            load_checkpoint_pruning_receipt(item) for item in receipt_paths
        ),
        instance_started_at=_string(arguments, "instance_started_at"),
        generated_at=_string(arguments, "generated_at"),
        provider_hourly_price=float(price),
    )
    write_training_report(_string(arguments, "output"), report)
    return report.to_dict()


def execute_sync_request(path: str | os.PathLike[str]) -> dict[str, object]:
    """Run real private-Hub synchronization from a strict request envelope."""

    fields = {
        "experiment_root",
        "experiment_id",
        "experiment_config_sha256",
        "repository",
        "revision",
        "staging_root",
        "timeout_seconds",
        "max_attempts",
        "output",
    }
    arguments = _request_arguments(path, command="sync", expected_fields=fields)
    timeout = arguments["timeout_seconds"]
    attempts = arguments["max_attempts"]
    if type(timeout) not in (int, float):
        raise ValueError("sync timeout_seconds must be a number")
    if type(attempts) is not int:
        raise ValueError("sync max_attempts must be an integer")
    experiment_root = _string(arguments, "experiment_root")
    output = _string(arguments, "output")
    _require_external_sync_output(experiment_root, output)

    from lehome_train.commands.sync import sync_experiment, write_sync_result
    from lehome_train.hub import HuggingFaceHubTransport

    result = sync_experiment(
        experiment_root,
        experiment_id=_string(arguments, "experiment_id"),
        experiment_config_sha256=_string(arguments, "experiment_config_sha256"),
        repository=_string(arguments, "repository"),
        revision=_string(arguments, "revision"),
        transport=HuggingFaceHubTransport(timeout_seconds=float(timeout)),
        staging_root=_string(arguments, "staging_root"),
        max_attempts=attempts,
    )
    write_sync_result(output, result)
    return result.to_dict()
