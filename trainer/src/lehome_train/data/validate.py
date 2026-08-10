"""Fail-closed prepared-dataset validation before GR00T training."""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

from lehome_train.data.inspect import read_json_object
from lehome_train.data.split import split_episode_ids
from lehome_train.data.stats import (
    _data_path,
    _reject_openpi_statistics,
    compute_reference_statistics,
)
from lehome_train.groot.modality import (
    modality_contract,
    runtime_modality_config_source,
    validate_runtime_modality_config,
)
from lehome_train.io import atomic_write_json, canonical_json_sha256, sha256_file
from lehome_train.models import validate_artifact_relative_path


_STAT_NAMES = ("mean", "std", "min", "max", "q01", "q99")
REQUIRED_VALIDATION_ARTIFACTS = (
    "meta/lehome_groot_modality.py",
    "meta/relative_stats.json",
    "meta/stats.json",
    "meta/validation_report.json",
)


def _finite_vector(value: object, size: int, label: str) -> None:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{label} must have exactly {size} dimensions")
    for item in value:
        if type(item) not in (int, float) or not math.isfinite(float(item)):
            raise ValueError(f"{label} must contain only finite numbers")


def _action_horizon(manifest: Mapping[str, Any]) -> int:
    future = manifest.get("future_actions")
    horizon = future.get("horizon") if isinstance(future, Mapping) else None
    if type(horizon) is not int or horizon not in {16, 40}:
        raise ValueError("prepared future actions must have horizon 16 or 40")
    return horizon


def _validate_statistics(dataset: Path, *, action_horizon: int) -> None:
    stats = read_json_object(dataset / "meta" / "stats.json")
    if set(stats) != {"observation.state", "action"}:
        raise ValueError("stats.json must contain exactly 12D state and action statistics")
    for feature, value in stats.items():
        if not isinstance(value, Mapping):
            raise ValueError(f"stats.json {feature} must be an object")
        if set(value) != set(_STAT_NAMES):
            raise ValueError(f"stats.json {feature} has an incomplete statistics schema")
        for stat in _STAT_NAMES:
            _finite_vector(value[stat], 12, f"stats.json {feature}.{stat}")

    relative_path = dataset / "meta" / "relative_stats.json"
    if not relative_path.is_file():
        raise ValueError("relative_stats.json is required for relative arm actions")
    relative = read_json_object(relative_path)
    if set(relative) != {"left_arm", "right_arm"}:
        raise ValueError("relative_stats.json must contain exactly both relative arm groups")
    for group, value in relative.items():
        if not isinstance(value, Mapping) or set(value) != set(_STAT_NAMES):
            raise ValueError(f"relative_stats.json {group} has an incomplete statistics schema")
        for stat in _STAT_NAMES:
            rows = value[stat]
            if not isinstance(rows, list) or len(rows) != action_horizon:
                raise ValueError(
                    f"relative_stats.json {group}.{stat} must have {action_horizon} horizons"
                )
            for index, row in enumerate(rows):
                _finite_vector(row, 5, f"relative_stats.json {group}.{stat}[{index}]")


def _validate_manifest(manifest: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    if manifest.get("fixed_language_instruction") != "fold the garment on the table":
        raise ValueError("prepared manifest has the wrong language instruction")
    action_schema = manifest.get("action_schema")
    state_schema = manifest.get("state_schema")
    if not isinstance(action_schema, Mapping) or not isinstance(state_schema, Mapping):
        raise ValueError("prepared manifest has no action/state schema")
    if action_schema.get("storage") != "absolute" or action_schema.get("dimension") != 12:
        raise ValueError("prepared actions must retain 12D absolute organizer targets")
    if state_schema.get("dimension") != 12:
        raise ValueError("prepared state must be 12D")
    future = manifest.get("future_actions")
    _action_horizon(manifest)
    if future.get("loader_allow_padding") is not False:
        raise ValueError("prepared loader must not pad action windows")
    train = manifest.get("train_episode_ids")
    validation = manifest.get("validation_episode_ids")
    if not isinstance(train, list) or not all(isinstance(value, str) for value in train):
        raise ValueError("prepared train episode IDs are invalid")
    if not isinstance(validation, list) or not all(isinstance(value, str) for value in validation):
        raise ValueError("prepared validation episode IDs are invalid")
    if not train:
        raise ValueError("prepared dataset has no training episodes")
    if not validation:
        raise ValueError("prepared validation episode IDs must not be empty")
    if len(set(train)) != len(train) or len(set(validation)) != len(validation):
        raise ValueError("prepared split episode IDs must be unique")
    if set(train).intersection(validation):
        raise ValueError("prepared train and validation episode IDs must be disjoint")
    return train, validation


def _validate_offline_split(
    dataset: Path,
    manifest: Mapping[str, Any],
    train: list[str],
    validation: list[str],
) -> None:
    """Verify the held-out split exists locally without handing it to Trainer."""

    episodes_path = dataset / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        raise ValueError("prepared episodes metadata is missing")
    observed: set[str] = set()
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        try:
            value = json.loads(line)
            episode_id = str(value["episode_index"])
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError("prepared episodes metadata is malformed") from error
        if episode_id in observed:
            raise ValueError("prepared episodes metadata contains duplicate IDs")
        observed.add(episode_id)
    expected = set(train) | set(validation)
    if observed != expected:
        raise ValueError("prepared episodes metadata does not exactly match offline split")
    split_seed = manifest.get("split_seed")
    validation_fraction = manifest.get("validation_fraction")
    if type(split_seed) is not int:
        raise ValueError("prepared split seed must be an integer")
    if type(validation_fraction) not in (int, float) or not math.isfinite(
        float(validation_fraction)
    ):
        raise ValueError("prepared validation fraction must be finite")
    if not 0 < float(validation_fraction) < 1:
        raise ValueError("prepared validation fraction must preserve a held-out split")
    deterministic = split_episode_ids(
        tuple(observed),
        seed=split_seed,
        validation_fraction=float(validation_fraction),
    )
    if tuple(train) != deterministic.train or tuple(validation) != deterministic.validation:
        raise ValueError("prepared split differs from its deterministic held-out split")
    for episode_id in sorted(expected, key=int):
        _data_path(dataset, episode_id)


def _artifact_path(dataset: Path, relative_path: object) -> tuple[str, Path]:
    if not isinstance(relative_path, str):
        raise ValueError("recorded artifact path must be a string")
    relative = validate_artifact_relative_path(relative_path)
    candidate = dataset / relative
    try:
        candidate.resolve().relative_to(dataset.resolve())
    except ValueError as error:
        raise ValueError("recorded artifact path escapes prepared dataset") from error
    if not candidate.is_file() or candidate.is_symlink():
        raise ValueError("recorded artifact must be a regular file")
    return relative, candidate


def _verify_artifact_entry(
    dataset: Path,
    entry: object,
    *,
    require_size: bool,
    context: str,
) -> str:
    if not isinstance(entry, Mapping):
        raise ValueError(f"{context} must be an object")
    expected_fields = {"relative_path", "sha256"}
    if require_size:
        expected_fields.add("byte_size")
    if set(entry) != expected_fields:
        raise ValueError(f"{context} has an invalid schema")
    relative, path = _artifact_path(dataset, entry.get("relative_path"))
    digest = entry.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{context} has an invalid hash")
    if sha256_file(path) != digest:
        raise ValueError(f"{context} hash mismatch: {relative}")
    if require_size:
        size = entry.get("byte_size")
        if type(size) is not int or size < 0 or path.stat().st_size != size:
            raise ValueError(f"{context} size mismatch: {relative}")
    return relative


def _verify_recorded_artifacts(dataset: Path, manifest: Mapping[str, Any]) -> None:
    artifacts = manifest.get("output_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("prepared manifest has no output artifacts")
    if manifest.get("output_manifest_sha256") != canonical_json_sha256(artifacts):
        raise ValueError("prepared manifest output artifact list hash mismatch")
    seen: set[str] = set()
    for entry in artifacts:
        relative = _verify_artifact_entry(
            dataset, entry, require_size=True, context="prepared artifact"
        )
        if relative in seen:
            raise ValueError("prepared manifest contains duplicate output artifacts")
        seen.add(relative)

    statistics = manifest.get("statistics")
    if not isinstance(statistics, Mapping) or statistics.get("status") != "computed_task_4_train_only":
        raise ValueError("prepared manifest has no completed train-only statistics")
    files = statistics.get("files")
    if not isinstance(files, list):
        raise ValueError("prepared manifest statistics files are invalid")
    expected_stats_paths = {
        "meta/lehome_groot_modality.py",
        "meta/stats.json",
        "meta/relative_stats.json",
    }
    recorded_stats_paths = {
        _verify_artifact_entry(
            dataset, entry, require_size=False, context="recorded statistics"
        )
        for entry in files
    }
    if recorded_stats_paths != expected_stats_paths or len(files) != len(recorded_stats_paths):
        raise ValueError("prepared manifest statistics files differ from the contract")


_PINNED_FLOAT32_REL_TOLERANCE = 5e-4
_PINNED_FLOAT32_ABS_TOLERANCE = 5e-6


def _compare_statistics(expected: object, actual: object, label: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(expected) != set(actual):
            raise ValueError(f"prepared statistics differ from verified training split at {label}")
        for key in expected:
            _compare_statistics(expected[key], actual[key], f"{label}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError(f"prepared statistics differ from verified training split at {label}")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _compare_statistics(left, right, f"{label}[{index}]")
        return
    if type(expected) not in (int, float) or type(actual) not in (int, float):
        raise ValueError(f"prepared statistics differ from verified training split at {label}")
    # The independent reference accumulates Python floats while the pinned
    # GR00T implementation explicitly casts inputs and reductions to float32.
    # The bounds cover the measured float32 accumulation drift while still
    # rejecting material changes to the persisted train-only statistics.
    if not math.isclose(
        float(expected),
        float(actual),
        rel_tol=_PINNED_FLOAT32_REL_TOLERANCE,
        abs_tol=_PINNED_FLOAT32_ABS_TOLERANCE,
    ):
        raise ValueError(f"prepared statistics differ from verified training split at {label}")


def _validate_statistics_match_train_split(dataset: Path) -> None:
    """Bind persisted normalizers to the just-verified train episode IDs."""

    reference = compute_reference_statistics(dataset)
    _compare_statistics(
        reference.stats,
        read_json_object(dataset / "meta" / "stats.json"),
        "stats",
    )
    _compare_statistics(
        reference.relative_stats,
        read_json_object(dataset / "meta" / "relative_stats.json"),
        "relative_stats",
    )


def _manifest_modality_sha256(manifest: Mapping[str, Any]) -> str:
    statistics = manifest.get("statistics")
    if not isinstance(statistics, Mapping):
        raise ValueError("prepared manifest has no train-only statistics record")
    files = statistics.get("files")
    if not isinstance(files, list):
        raise ValueError("prepared manifest statistics files are invalid")
    matches = [
        item
        for item in files
        if isinstance(item, Mapping)
        and item.get("relative_path") == "meta/lehome_groot_modality.py"
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("sha256"), str):
        raise ValueError("prepared manifest lacks the modality configuration hash")
    digest = matches[0]["sha256"]
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("prepared manifest modality configuration hash is invalid")
    return digest


def _validate_modality_metadata(dataset: Path) -> None:
    metadata = read_json_object(dataset / "meta" / "modality.json")
    expected_video = {"top_rgb", "left_rgb", "right_rgb"}
    video = metadata.get("video")
    if not isinstance(video, Mapping) or set(video) != expected_video:
        raise ValueError("prepared modality metadata has invalid video mappings")
    for key in sorted(expected_video):
        value = video[key]
        if not isinstance(value, Mapping) or value.get("original_key") != (
            f"observation.images.{key}"
        ):
            raise ValueError("prepared modality metadata has invalid video source key")

    annotation = metadata.get("annotation")
    if not isinstance(annotation, Mapping) or set(annotation) != {
        "human.task_description"
    }:
        raise ValueError("prepared modality metadata has invalid annotation mappings")
    language = annotation["human.task_description"]
    if not isinstance(language, Mapping) or language.get("original_key") != "task_index":
        raise ValueError("prepared modality metadata has invalid annotation source key")

    expected_groups = {
        "left_arm": (0, 5),
        "left_gripper": (5, 6),
        "right_arm": (6, 11),
        "right_gripper": (11, 12),
    }
    for modality in ("state", "action"):
        groups = metadata.get(modality)
        if not isinstance(groups, Mapping) or set(groups) != set(expected_groups):
            raise ValueError(f"prepared modality metadata has invalid {modality} groups")
        for group, (start, end) in expected_groups.items():
            value = groups[group]
            if not isinstance(value, Mapping) or value.get("start") != start or value.get("end") != end:
                raise ValueError(f"prepared modality metadata has invalid {modality} dimensions")
            if value.get("original_key") != ("observation.state" if modality == "state" else "action"):
                raise ValueError(f"prepared modality metadata has invalid {modality} source key")


def _validate_modality(dataset: Path, manifest: Mapping[str, Any]) -> None:
    path = dataset / "meta" / "lehome_groot_modality.py"
    if not path.is_file():
        raise ValueError("prepared GR00T modality configuration is missing")
    source = path.read_text(encoding="utf-8")
    action_horizon = _action_horizon(manifest)
    if source != runtime_modality_config_source(action_horizon=action_horizon):
        raise ValueError("prepared GR00T modality configuration is not canonical")
    if sha256_file(path) != _manifest_modality_sha256(manifest):
        raise ValueError("prepared GR00T modality configuration hash differs from manifest")
    _validate_modality_metadata(dataset)
    contract = modality_contract(action_horizon=action_horizon)
    if contract["state"]["dimension"] != 12 or contract["action"]["dimension"] != 12:
        raise AssertionError("internal modality contract is not 12D")


def _run_pinned_loader(
    dataset: Path, groot_root: Path, *, action_horizon: int
) -> None:
    """Consume one joint-space VLA sample via the pinned GR00T loader."""

    if not (groot_root / "gr00t" / "data" / "dataset" / "lerobot_episode_loader.py").is_file():
        raise FileNotFoundError("pinned Isaac-GR00T loader was not found")
    root_text = str(groot_root.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    modality_path = dataset / "meta" / "lehome_groot_modality.py"
    spec = importlib.util.spec_from_file_location("lehome_validate_modality", modality_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load prepared modality configuration")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag

    config = MODALITY_CONFIGS[EmbodimentTag.NEW_EMBODIMENT.value]
    validate_runtime_modality_config(config, action_horizon=action_horizon)
    loader = LeRobotEpisodeLoader(dataset, config)
    episode = loader[0]
    sample = extract_step_data(
        episode, 0, config, EmbodimentTag.NEW_EMBODIMENT, allow_padding=False
    )
    if set(sample.actions) != {"left_arm", "left_gripper", "right_arm", "right_gripper"}:
        raise ValueError("pinned loader did not emit the required action groups")


def validate_prepared_dataset(
    dataset_path: str | Path,
    *,
    groot_root: str | Path | None = None,
) -> dict[str, object]:
    """Validate prepared data and atomically persist report plus artifact hashes."""

    dataset = Path(dataset_path)
    _reject_openpi_statistics(dataset)
    manifest = read_json_object(dataset / "manifest.json")
    from lehome_train.data.stats import _require_frozen_flywheel_mix
    _require_frozen_flywheel_mix(manifest)
    action_horizon = _action_horizon(manifest)
    train, validation = _validate_manifest(manifest)
    _validate_offline_split(dataset, manifest, train, validation)
    _validate_statistics(dataset, action_horizon=action_horizon)
    _validate_modality(dataset, manifest)
    _verify_recorded_artifacts(dataset, manifest)
    _validate_statistics_match_train_split(dataset)
    resolved_root = groot_root or os.environ.get("LEHOME_GROOT_ROOT")
    loader_integration = "not_run_no_pinned_runtime"
    if resolved_root:
        _run_pinned_loader(
            dataset, Path(resolved_root), action_horizon=action_horizon
        )
        loader_integration = "pinned_loader_one_batch"
    report = {
        "schema_version": 1,
        "valid": True,
        "dataset_manifest_sha256": sha256_file(dataset / "manifest.json"),
        "train_episode_count": len(train),
        "validation_episode_count": len(validation),
        "trainer_validation_split": "offline_only",
        "loader_integration": loader_integration,
        "modality_contract_sha256": canonical_json_sha256(
            modality_contract(action_horizon=action_horizon)
        ),
    }
    report_path = dataset / "meta" / "validation_report.json"
    atomic_write_json(report_path, report)
    hashes = {
        "schema_version": 1,
        "artifacts": {
            relative: sha256_file(dataset / relative)
            for relative in REQUIRED_VALIDATION_ARTIFACTS
        },
    }
    atomic_write_json(dataset / "meta" / "prepared_hashes.json", hashes)
    return report
