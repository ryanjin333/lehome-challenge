"""Crash-safe, immutable identity and status for one training experiment."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

from lehome_train.io import atomic_write_json, canonical_json_sha256
from lehome_train.models import ArtifactIdentity
from lehome_train.preflight import PREFLIGHT_STAGE_NAMES, reject_secret_bearing_config


_IDENTITY_FILENAME = "experiment.json"
_STATUS_FILENAME = "status.json"
_LOG_DIRECTORY = "logs"
_LOG_FILENAME = "prepare.log"


@dataclass(frozen=True, slots=True)
class Experiment:
    """One effective output directory, either fresh or safely resumed."""

    experiment_id: str
    root: Path
    resumed: bool


def _identity_payload(
    resolved_config: Mapping[str, object],
    artifacts: Sequence[ArtifactIdentity],
) -> dict[str, object]:
    if not isinstance(resolved_config, Mapping) or not resolved_config:
        raise ValueError("resolved experiment config must be a non-empty mapping")
    reject_secret_bearing_config(dict(resolved_config))
    paths = tuple(item.relative_path for item in artifacts)
    if not artifacts or len(set(paths)) != len(paths):
        raise ValueError("experiment artifacts must be non-empty and unique")
    return {
        "artifacts": [item.to_dict() for item in sorted(artifacts, key=lambda item: item.relative_path)],
        "resolved_config": dict(resolved_config),
    }


def experiment_id(
    resolved_config: Mapping[str, object],
    artifacts: Sequence[ArtifactIdentity],
) -> str:
    """Return the full canonical hash of all behavior-affecting inputs."""

    return canonical_json_sha256(_identity_payload(resolved_config, artifacts))


def _read_json_object(path: Path, *, description: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is malformed") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} is malformed")
    return value


def _default_status(experiment_id_value: str, identity_sha256: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_id": experiment_id_value,
        "identity_sha256": identity_sha256,
        "stages": {
            name: {"state": "pending", "duration_seconds": None}
            for name in PREFLIGHT_STAGE_NAMES
        },
    }


def _validate_status(status: Mapping[str, object], experiment: Experiment) -> dict[str, object]:
    if status.get("schema_version") != 1:
        raise ValueError("experiment status has an unsupported schema")
    if status.get("experiment_id") != experiment.experiment_id:
        raise ValueError("experiment status identity is incompatible")
    if status.get("identity_sha256") != experiment.experiment_id:
        raise ValueError("experiment status identity is incompatible")
    stages = status.get("stages")
    if not isinstance(stages, dict) or set(stages) != set(PREFLIGHT_STAGE_NAMES):
        raise ValueError("experiment status stage contract is incompatible")
    for stage in PREFLIGHT_STAGE_NAMES:
        record = stages[stage]
        if not isinstance(record, dict) or set(record) != {"state", "duration_seconds"}:
            raise ValueError("experiment status stage record is incompatible")
        if record["state"] not in {"pending", "complete"}:
            raise ValueError("experiment status stage record is incompatible")
        duration = record["duration_seconds"]
        if record["state"] == "pending" and duration is not None:
            raise ValueError("pending experiment stage must not have a duration")
        if record["state"] == "complete" and (
            type(duration) not in (int, float) or not math.isfinite(float(duration)) or duration < 0
        ):
            raise ValueError("completed experiment stage duration is invalid")
    return dict(status)


def _append_log(experiment: Experiment, message: str) -> None:
    if "\n" in message or "\r" in message:
        raise ValueError("experiment log messages must be one line")
    path = experiment.root / _LOG_DIRECTORY / _LOG_FILENAME
    with path.open("a", encoding="utf-8") as stream:
        stream.write(message + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _candidate_root(output_root: Path, base_id: str, counter: int) -> Path:
    return output_root / (base_id if counter == 0 else f"{base_id}-superseded-{counter:02d}")


def create_or_resume_experiment(
    output_root: str | os.PathLike[str],
    *,
    resolved_config: Mapping[str, object],
    artifacts: Sequence[ArtifactIdentity],
) -> Experiment:
    """Create an immutable output root or resume only an exact matching run.

    A malformed or mismatched collision is retained for debugging.  A fresh,
    explicitly superseded directory is selected instead of overwriting it.
    """

    root = Path(output_root)
    if root.exists() and (not root.is_dir() or root.is_symlink()):
        raise ValueError("experiment output root must be a real directory")
    root.mkdir(parents=True, exist_ok=True)
    if not os.access(root, os.W_OK):
        raise ValueError("experiment output root is not writable")
    payload = _identity_payload(resolved_config, artifacts)
    base_id = canonical_json_sha256(payload)
    for counter in range(1000):
        candidate = _candidate_root(root, base_id, counter)
        if not candidate.exists():
            candidate.mkdir()
            (candidate / _LOG_DIRECTORY).mkdir()
            experiment = Experiment(base_id, candidate, False)
            atomic_write_json(candidate / _IDENTITY_FILENAME, payload)
            atomic_write_json(candidate / _STATUS_FILENAME, _default_status(base_id, base_id))
            _append_log(experiment, "prepare: created immutable experiment identity")
            return experiment
        if not candidate.is_dir() or candidate.is_symlink():
            continue
        identity_path = candidate / _IDENTITY_FILENAME
        status_path = candidate / _STATUS_FILENAME
        if not identity_path.is_file() or not status_path.is_file():
            continue
        try:
            existing_payload = _read_json_object(identity_path, description="experiment identity")
        except ValueError:
            continue
        if existing_payload != payload:
            continue
        experiment = Experiment(base_id, candidate, True)
        try:
            _validate_status(
                _read_json_object(status_path, description="experiment status"),
                experiment,
            )
        except ValueError:
            # Retain an incompatible partial run verbatim; its identity may be
            # useful for diagnosis, but it must never be resumed or overwritten.
            continue
        log_directory = candidate / _LOG_DIRECTORY
        if not log_directory.is_dir() or log_directory.is_symlink():
            continue
        _append_log(experiment, "prepare: resumed matching immutable experiment identity")
        return experiment
    raise RuntimeError("could not allocate a safe superseded experiment directory")


def _load_status(experiment: Experiment) -> dict[str, object]:
    return _validate_status(
        _read_json_object(experiment.root / _STATUS_FILENAME, description="experiment status"),
        experiment,
    )


def pending_stages(experiment: Experiment) -> tuple[str, ...]:
    """Return only stages that were not fully completed in a compatible run."""

    status = _load_status(experiment)
    stages = status["stages"]
    assert isinstance(stages, dict)
    return tuple(name for name in PREFLIGHT_STAGE_NAMES if stages[name]["state"] == "pending")


def mark_stage_complete(
    experiment: Experiment,
    stage: str,
    *,
    duration_seconds: float,
) -> None:
    """Atomically mark one canonical stage complete without mutating identity."""

    if stage not in PREFLIGHT_STAGE_NAMES:
        raise ValueError("unknown preflight stage")
    if type(duration_seconds) not in (int, float) or not math.isfinite(float(duration_seconds)) or duration_seconds < 0:
        raise ValueError("preflight stage duration must be finite and nonnegative")
    status = _load_status(experiment)
    stages = status["stages"]
    assert isinstance(stages, dict)
    index = PREFLIGHT_STAGE_NAMES.index(stage)
    if any(stages[name]["state"] != "complete" for name in PREFLIGHT_STAGE_NAMES[:index]):
        raise ValueError("preflight stages must complete in canonical order")
    if stages[stage]["state"] == "complete":
        return
    stages[stage] = {"state": "complete", "duration_seconds": float(duration_seconds)}
    atomic_write_json(experiment.root / _STATUS_FILENAME, status)
    _append_log(experiment, f"prepare: {stage} complete in {float(duration_seconds):.6f}s")
