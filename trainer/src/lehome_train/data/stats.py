"""Train-split-only normalization statistics for prepared GR00T datasets."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping

import pyarrow.parquet as pq

from lehome_train.data.inspect import read_json_object
from lehome_train.groot.modality import write_runtime_modality_config
from lehome_train.io import atomic_write_json, canonical_json_bytes, sha256_file


_STAT_NAMES = ("mean", "std", "min", "max", "q01", "q99")
_VECTOR_KEYS = ("observation.state", "action")
_RELATIVE_GROUPS = (("left_arm", 0, 5), ("right_arm", 6, 11))
_ACTION_HORIZON = 16


def _write_json_lines(path: Path, values: Iterable[Mapping[str, object]]) -> None:
    payload = b"".join(canonical_json_bytes(value) + b"\n" for value in values)
    path.write_bytes(payload)


@dataclass(frozen=True, slots=True)
class StatisticsBundle:
    """JSON-ready train-only statistics and their computation provenance."""

    stats: dict[str, dict[str, list[float]]]
    relative_stats: dict[str, dict[str, list[list[float]]]]


def _numeric_episode_id(value: str) -> int:
    try:
        integer = int(value)
    except ValueError as error:
        raise ValueError(f"prepared episode ID must be an integer: {value!r}") from error
    if str(integer) != value:
        raise ValueError(f"prepared episode ID must be canonical: {value!r}")
    return integer


def _load_manifest(dataset: Path) -> dict[str, Any]:
    manifest = read_json_object(dataset / "manifest.json")
    if manifest.get("output_format") != "groot_lerobot_v2.1_per_episode":
        raise ValueError("prepared dataset has an unsupported output format")
    return manifest


def _require_frozen_flywheel_mix(manifest: Mapping[str, Any]) -> None:
    """Reject a final flywheel snapshot unless its selection plan is immutable."""
    plan = manifest.get("flywheel_mix_plan")
    if plan is None:
        return
    if not isinstance(plan, Mapping):
        raise ValueError("flywheel mix plan is malformed")
    # This runs before any Parquet statistics are read.  A digest alone is not
    # enough: the plan must also name complete 16-frame ranges, have an exact
    # post-split 70/30 train mix, and retain a nonempty offline holdout.
    from lehome_train.flywheel.mix import validate_mix_plan_payload

    validate_mix_plan_payload(plan)


def _train_ids(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    train = manifest.get("train_episode_ids")
    validation = manifest.get("validation_episode_ids")
    if not isinstance(train, list) or not all(isinstance(value, str) for value in train):
        raise ValueError("prepared manifest train_episode_ids must be string IDs")
    if not isinstance(validation, list) or not all(isinstance(value, str) for value in validation):
        raise ValueError("prepared manifest validation_episode_ids must be string IDs")
    if not train:
        raise ValueError("prepared dataset has no training episodes")
    if len(set(train)) != len(train) or len(set(validation)) != len(validation):
        raise ValueError("prepared manifest episode IDs must be unique")
    if set(train).intersection(validation):
        raise ValueError("prepared manifest train and validation IDs must be disjoint")
    return tuple(sorted(train, key=_numeric_episode_id))


def _reject_openpi_statistics(dataset: Path) -> None:
    matches = sorted(path for path in dataset.rglob("norm_stats.json") if path.is_file())
    if matches:
        raise ValueError("OpenPI norm_stats.json is forbidden in a GR00T prepared dataset")


def _data_path(dataset: Path, episode_id: str) -> Path:
    info = read_json_object(dataset / "meta" / "info.json")
    pattern = info.get("data_path")
    chunks_size = info.get("chunks_size")
    if not isinstance(pattern, str) or type(chunks_size) is not int or chunks_size <= 0:
        raise ValueError("prepared info metadata has an invalid v2 data path")
    numeric_id = _numeric_episode_id(episode_id)
    try:
        relative = pattern.format(
            episode_chunk=numeric_id // chunks_size,
            episode_index=numeric_id,
        )
    except (KeyError, ValueError) as error:
        raise ValueError("prepared v2 data path pattern is invalid") from error
    candidate = dataset / relative
    try:
        candidate.resolve().relative_to(dataset.resolve())
    except ValueError as error:
        raise ValueError("prepared v2 data path escapes dataset") from error
    if not candidate.is_file():
        raise ValueError(f"missing prepared episode data: {relative}")
    return candidate


def _validate_vector(value: object, *, feature: str, episode_id: str, frame: int) -> list[float]:
    if not isinstance(value, list) or len(value) != 12:
        raise ValueError(
            f"prepared {feature} must be 12D at episode {episode_id} frame {frame}"
        )
    normalized: list[float] = []
    for dimension, item in enumerate(value):
        if type(item) not in (int, float) or not math.isfinite(float(item)):
            raise ValueError(
                f"prepared {feature} has non-finite value at episode {episode_id} "
                f"frame {frame} dimension {dimension}"
            )
        normalized.append(float(item))
    return normalized


def _episode_values(dataset: Path, episode_id: str) -> tuple[list[list[float]], list[list[float]]]:
    table = pq.read_table(_data_path(dataset, episode_id), columns=list(_VECTOR_KEYS))
    state_rows = table["observation.state"].to_pylist()
    action_rows = table["action"].to_pylist()
    if not state_rows or len(state_rows) != len(action_rows):
        raise ValueError(f"prepared episode {episode_id} has inconsistent state/action frames")
    states = [
        _validate_vector(value, feature="state", episode_id=episode_id, frame=index)
        for index, value in enumerate(state_rows)
    ]
    actions = [
        _validate_vector(value, feature="action", episode_id=episode_id, frame=index)
        for index, value in enumerate(action_rows)
    ]
    return states, actions


def _quantile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _vector_statistics(values: Iterable[list[float]], dimension: int) -> dict[str, list[float]]:
    rows = list(values)
    if not rows:
        raise ValueError("cannot compute statistics from no values")
    if any(len(row) != dimension for row in rows):
        raise ValueError("statistics input has inconsistent dimensions")
    result = {name: [] for name in _STAT_NAMES}
    for index in range(dimension):
        column = sorted(row[index] for row in rows)
        mean = sum(column) / len(column)
        variance = sum((value - mean) ** 2 for value in column) / len(column)
        result["mean"].append(mean)
        result["std"].append(math.sqrt(variance))
        result["min"].append(column[0])
        result["max"].append(column[-1])
        result["q01"].append(_quantile(column, 0.01))
        result["q99"].append(_quantile(column, 0.99))
    return result


def _relative_statistics(
    episodes: Iterable[tuple[list[list[float]], list[list[float]]]],
) -> dict[str, dict[str, list[list[float]]]]:
    by_group = {
        name: [[] for _ in range(_ACTION_HORIZON)]
        for name, _, _ in _RELATIVE_GROUPS
    }
    for states, actions in episodes:
        for start in range(max(0, len(actions) - _ACTION_HORIZON + 1)):
            current = states[start]
            for offset in range(_ACTION_HORIZON):
                target = actions[start + offset]
                for name, begin, end in _RELATIVE_GROUPS:
                    by_group[name][offset].append(
                        [target[index] - current[index] for index in range(begin, end)]
                    )
    output: dict[str, dict[str, list[list[float]]]] = {}
    for name, begin, end in _RELATIVE_GROUPS:
        per_offset = [_vector_statistics(rows, end - begin) for rows in by_group[name]]
        output[name] = {
            stat: [item[stat] for item in per_offset]
            for stat in _STAT_NAMES
        }
    return output


def compute_reference_statistics(dataset_path: str | Path) -> StatisticsBundle:
    """Compute finite 12D stats from only manifest-listed training episodes."""

    dataset = Path(dataset_path)
    _reject_openpi_statistics(dataset)
    manifest = _load_manifest(dataset)
    _require_frozen_flywheel_mix(manifest)
    train_ids = _train_ids(manifest)
    episodes = [_episode_values(dataset, episode_id) for episode_id in train_ids]
    states = [row for episode, _ in episodes for row in episode]
    actions = [row for _, episode in episodes for row in episode]
    return StatisticsBundle(
        stats={
            "observation.state": _vector_statistics(states, 12),
            "action": _vector_statistics(actions, 12),
        },
        relative_stats=_relative_statistics(episodes),
    )


def _json_plain(value: object) -> object:
    if hasattr(value, "tolist"):
        return _json_plain(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_plain(item) for item in value]
    if type(value) in (int, float):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("pinned GR00T statistics contain non-finite values")
        return number
    return value


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve())


def _train_only_runtime_view(dataset: Path, train_ids: tuple[str, ...]) -> Path:
    """Create a disposable v2 view so pinned APIs cannot see validation rows."""

    view = Path(tempfile.mkdtemp(prefix="lehome-groot-train-stats-"))
    try:
        meta = view / "meta"
        meta.mkdir()
        info = read_json_object(dataset / "meta" / "info.json")
        info["total_episodes"] = len(train_ids)
        info["total_frames"] = sum(
            pq.ParquetFile(_data_path(dataset, episode_id)).metadata.num_rows
            for episode_id in train_ids
        )
        atomic_write_json(meta / "info.json", info)
        selected = set(train_ids)
        episodes = [
            json.loads(line)
            for line in (dataset / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
            if line and str(json.loads(line)["episode_index"]) in selected
        ]
        if len(episodes) != len(train_ids):
            raise ValueError("prepared episodes metadata does not match training split")
        _write_json_lines(meta / "episodes.jsonl", episodes)
        shutil.copy2(dataset / "meta" / "tasks.jsonl", meta / "tasks.jsonl")
        shutil.copy2(dataset / "meta" / "modality.json", meta / "modality.json")
        for episode_id in train_ids:
            path = _data_path(dataset, episode_id)
            destination = view / path.relative_to(dataset)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        _link(dataset / "videos", view / "videos")
        return view
    except BaseException:
        shutil.rmtree(view, ignore_errors=True)
        raise


def _import_pinned_stats(groot_root: Path):
    stats_file = groot_root / "gr00t" / "data" / "stats.py"
    if not stats_file.is_file():
        raise FileNotFoundError("pinned Isaac-GR00T stats.py was not found")
    root_text = str(groot_root.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    spec = importlib.util.spec_from_file_location("lehome_pinned_groot_stats", stats_file)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load pinned Isaac-GR00T statistics API")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _register_runtime_modality(path: Path) -> None:
    name = "lehome_runtime_modality_config"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load prepared GR00T modality configuration")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


def _compute_pinned_statistics(dataset: Path, groot_root: Path) -> StatisticsBundle:
    """Run the pinned GR00T statistics APIs against a train-only temporary view."""

    train_ids = _train_ids(_load_manifest(dataset))
    view = _train_only_runtime_view(dataset, train_ids)
    try:
        modality_path = write_runtime_modality_config(view / "meta" / "lehome_modality.py")
        stats_api = _import_pinned_stats(groot_root)
        _register_runtime_modality(modality_path)
        parquet_paths = [_data_path(view, episode_id) for episode_id in train_ids]
        stats = stats_api.calculate_dataset_statistics(
            parquet_paths, features=list(_VECTOR_KEYS)
        )
        atomic_write_json(view / "meta" / "stats.json", _json_plain(stats))
        from gr00t.data.embodiment_tags import EmbodimentTag

        relative = {
            key: stats_api.calculate_stats_for_key(
                view, EmbodimentTag.NEW_EMBODIMENT, key
            )
            for key, _, _ in _RELATIVE_GROUPS
        }
        return StatisticsBundle(
            stats=_json_plain(stats),  # type: ignore[arg-type]
            relative_stats=_json_plain(relative),  # type: ignore[arg-type]
        )
    finally:
        shutil.rmtree(view, ignore_errors=True)


def write_train_statistics(
    dataset_path: str | Path,
    *,
    groot_root: str | Path | None = None,
) -> dict[str, object]:
    """Write train-only stats atomically, using pinned APIs when supplied.

    Fixture tests intentionally use the pure-Python reference.  The container
    preparation path passes ``groot_root`` and therefore exercises the pinned
    Isaac-GR00T API on a temporary view containing only training episodes.
    """

    dataset = Path(dataset_path)
    _reject_openpi_statistics(dataset)
    manifest = _load_manifest(dataset)
    resolved_root = groot_root or os.environ.get("LEHOME_GROOT_ROOT")
    if resolved_root:
        bundle = _compute_pinned_statistics(dataset, Path(resolved_root))
        runtime = "pinned_gr00t"
    else:
        bundle = compute_reference_statistics(dataset)
        runtime = "python_reference"
    meta = dataset / "meta"
    if not meta.is_dir():
        raise FileNotFoundError("prepared dataset meta directory does not exist")
    modality_path = write_runtime_modality_config(meta / "lehome_groot_modality.py")
    stats_path = meta / "stats.json"
    relative_path = meta / "relative_stats.json"
    atomic_write_json(stats_path, bundle.stats)
    atomic_write_json(relative_path, bundle.relative_stats)
    result = {
        "runtime": runtime,
        "modality_config_sha256": sha256_file(modality_path),
        "stats_sha256": sha256_file(stats_path),
        "relative_stats_sha256": sha256_file(relative_path),
    }
    manifest["statistics"] = {
        "status": "computed_task_4_train_only",
        "runtime": runtime,
        "files": [
            {
                "relative_path": "meta/lehome_groot_modality.py",
                "sha256": result["modality_config_sha256"],
            },
            {"relative_path": "meta/stats.json", "sha256": result["stats_sha256"]},
            {
                "relative_path": "meta/relative_stats.json",
                "sha256": result["relative_stats_sha256"],
            },
        ],
    }
    atomic_write_json(dataset / "manifest.json", manifest)
    return result
