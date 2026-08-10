"""The sole joint-space GR00T modality contract for LeHome."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from lehome_train.data.mapping import ACTION_HORIZON, FIXED_INSTRUCTION


_GROUPS = (
    ("left_arm", 5, "relative"),
    ("left_gripper", 1, "absolute"),
    ("right_arm", 5, "relative"),
    ("right_gripper", 1, "absolute"),
)


def _action_horizon(value: int) -> int:
    if type(value) is not int or value not in {16, 40}:
        raise ValueError("action horizon must be 16 or 40")
    return value


def modality_contract(*, action_horizon: int = ACTION_HORIZON) -> dict[str, object]:
    """Return a detached, serializable exact LeHome GR00T contract."""

    action_horizon = _action_horizon(action_horizon)

    groups = [
        {"key": key, "dimension": dimension, "representation": representation}
        for key, dimension, representation in _GROUPS
    ]
    return {
        "schema_version": 1,
        "video": {
            "delta_indices": [0],
            "modality_keys": ["top_rgb", "left_rgb", "right_rgb"],
        },
        "state": {
            "delta_indices": [0],
            "modality_keys": [item["key"] for item in groups],
            "dimension": 12,
        },
        "action": {
            "delta_indices": list(range(action_horizon)),
            "modality_keys": [item["key"] for item in groups],
            "dimension": 12,
            "groups": groups,
        },
        "language": {
            "delta_indices": [0],
            "modality_keys": ["annotation.human.task_description"],
            "instruction": FIXED_INSTRUCTION,
        },
    }


def runtime_modality_config_source(*, action_horizon: int = ACTION_HORIZON) -> str:
    """Return a self-contained config consumed by pinned Isaac-GR00T.

    The prepared dataset stores absolute 12D targets.  The two arm groups are
    marked relative here, so the pinned loader performs that subtraction once;
    both grippers remain absolute.
    """

    action_horizon = _action_horizon(action_horizon)
    source = """from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS, register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

lehome_so101_config = {
    \"video\": ModalityConfig(delta_indices=[0], modality_keys=[\"top_rgb\", \"left_rgb\", \"right_rgb\"]),
    \"state\": ModalityConfig(delta_indices=[0], modality_keys=[\"left_arm\", \"left_gripper\", \"right_arm\", \"right_gripper\"]),
    \"action\": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=[\"left_arm\", \"left_gripper\", \"right_arm\", \"right_gripper\"],
        action_configs=[
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        ],
    ),
    \"language\": ModalityConfig(delta_indices=[0], modality_keys=[\"annotation.human.task_description\"]),
}

if EmbodimentTag.NEW_EMBODIMENT.value in MODALITY_CONFIGS:
    existing = MODALITY_CONFIGS[EmbodimentTag.NEW_EMBODIMENT.value]
    assert existing["video"].modality_keys == lehome_so101_config["video"].modality_keys
    assert existing["state"].modality_keys == lehome_so101_config["state"].modality_keys
    assert existing["action"].modality_keys == lehome_so101_config["action"].modality_keys
    assert existing["action"].delta_indices == lehome_so101_config["action"].delta_indices
else:
    register_modality_config(lehome_so101_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
"""
    return source.replace("delta_indices=list(range(16))", f"delta_indices=list(range({action_horizon}))")


def _enum_value(value: object) -> str:
    """Normalize GR00T enum values without importing GR00T in fixture tests."""

    raw = getattr(value, "value", value)
    if not isinstance(raw, str):
        raise ValueError("runtime GR00T modality enum has an invalid value")
    return raw


def _runtime_field(config: object, field_name: str) -> object:
    try:
        return getattr(config, field_name)
    except AttributeError as error:
        raise ValueError(f"runtime GR00T modality has no {field_name}") from error


def validate_runtime_modality_config(
    config: Mapping[str, Any], *, action_horizon: int = ACTION_HORIZON
) -> None:
    """Fail closed unless a registered GR00T config is exactly our contract."""

    if set(config) != {"video", "state", "action", "language"}:
        raise ValueError("runtime GR00T modality must have exactly four modalities")
    action_horizon = _action_horizon(action_horizon)
    expected_groups = ["left_arm", "left_gripper", "right_arm", "right_gripper"]
    expected = {
        "video": ([0], ["top_rgb", "left_rgb", "right_rgb"]),
        "state": ([0], expected_groups),
        "action": (list(range(action_horizon)), expected_groups),
        "language": ([0], ["annotation.human.task_description"]),
    }
    for name, (delta_indices, modality_keys) in expected.items():
        actual = config[name]
        if _runtime_field(actual, "delta_indices") != delta_indices:
            raise ValueError(f"runtime GR00T {name} delta indices differ from contract")
        if _runtime_field(actual, "modality_keys") != modality_keys:
            raise ValueError(f"runtime GR00T {name} keys differ from contract")
    action_configs = _runtime_field(config["action"], "action_configs")
    if not isinstance(action_configs, list) or len(action_configs) != 4:
        raise ValueError("runtime GR00T action config must contain four joint groups")
    expected_representations = ["relative", "absolute", "relative", "absolute"]
    expected_dimensions = [5, 1, 5, 1]
    for index, (action_config, representation, dimension) in enumerate(
        zip(action_configs, expected_representations, expected_dimensions, strict=True)
    ):
        if _enum_value(_runtime_field(action_config, "rep")) != representation:
            raise ValueError(f"runtime GR00T action group {index} has wrong representation")
        if _enum_value(_runtime_field(action_config, "type")) != "non_eef":
            raise ValueError(f"runtime GR00T action group {index} is not joint space")
        if _enum_value(_runtime_field(action_config, "format")) != "default":
            raise ValueError(f"runtime GR00T action group {index} has wrong format")
        if dimension <= 0:
            raise AssertionError("internal action group dimension is invalid")


def write_runtime_modality_config(
    destination: str | Path, *, action_horizon: int = ACTION_HORIZON
) -> Path:
    """Atomically write the exact custom-embodiment Python configuration."""

    path = Path(destination)
    if not path.parent.is_dir():
        raise FileNotFoundError("modality configuration parent does not exist")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(runtime_modality_config_source(action_horizon=action_horizon))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path
