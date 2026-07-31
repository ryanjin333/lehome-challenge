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
from lehome_train.data.stats import _data_path, _reject_openpi_statistics
from lehome_train.groot.modality import modality_contract
from lehome_train.io import atomic_write_json, canonical_json_sha256, sha256_file


_STAT_NAMES = ("mean", "std", "min", "max", "q01", "q99")
_REQUIRED_ARTIFACTS = (
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


def _validate_statistics(dataset: Path) -> None:
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
            if not isinstance(rows, list) or len(rows) != 16:
                raise ValueError(f"relative_stats.json {group}.{stat} must have 16 horizons")
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
    if not isinstance(future, Mapping) or future.get("horizon") != 16:
        raise ValueError("prepared future actions must have horizon 16")
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
    if set(train).intersection(validation):
        raise ValueError("prepared train and validation episode IDs must be disjoint")
    return train, validation


def _validate_offline_split(dataset: Path, train: list[str], validation: list[str]) -> None:
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
    for episode_id in sorted(expected, key=int):
        _data_path(dataset, episode_id)


def _validate_modality(dataset: Path) -> None:
    path = dataset / "meta" / "lehome_groot_modality.py"
    if not path.is_file():
        raise ValueError("prepared GR00T modality configuration is missing")
    source = path.read_text(encoding="utf-8")
    required = (
        '"top_rgb", "left_rgb", "right_rgb"',
        '"left_arm", "left_gripper", "right_arm", "right_gripper"',
        "list(range(16))",
        "ActionRepresentation.RELATIVE",
        "ActionRepresentation.ABSOLUTE",
    )
    if any(fragment not in source for fragment in required):
        raise ValueError("prepared GR00T modality configuration differs from the contract")
    contract = modality_contract()
    if contract["state"]["dimension"] != 12 or contract["action"]["dimension"] != 12:
        raise AssertionError("internal modality contract is not 12D")


def _run_pinned_loader(dataset: Path, groot_root: Path) -> None:
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
    train, validation = _validate_manifest(manifest)
    _validate_offline_split(dataset, train, validation)
    _validate_statistics(dataset)
    _validate_modality(dataset)
    resolved_root = groot_root or os.environ.get("LEHOME_GROOT_ROOT")
    loader_integration = "not_run_no_pinned_runtime"
    if resolved_root:
        _run_pinned_loader(dataset, Path(resolved_root))
        loader_integration = "pinned_loader_one_batch"
    report = {
        "schema_version": 1,
        "valid": True,
        "dataset_manifest_sha256": sha256_file(dataset / "manifest.json"),
        "train_episode_count": len(train),
        "validation_episode_count": len(validation),
        "trainer_validation_split": "offline_only",
        "loader_integration": loader_integration,
        "modality_contract_sha256": canonical_json_sha256(modality_contract()),
    }
    report_path = dataset / "meta" / "validation_report.json"
    atomic_write_json(report_path, report)
    hashes = {
        "schema_version": 1,
        "artifacts": {
            relative: sha256_file(dataset / relative)
            for relative in _REQUIRED_ARTIFACTS
        },
    }
    atomic_write_json(dataset / "meta" / "prepared_hashes.json", hashes)
    return report
