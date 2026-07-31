"""The sole joint-space GR00T modality contract for LeHome."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

from lehome_train.data.mapping import ACTION_HORIZON, FIXED_INSTRUCTION


_GROUPS = (
    ("left_arm", 5, "relative"),
    ("left_gripper", 1, "absolute"),
    ("right_arm", 5, "relative"),
    ("right_gripper", 1, "absolute"),
)


def modality_contract() -> dict[str, object]:
    """Return a detached, serializable exact LeHome GR00T contract."""

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
            "delta_indices": list(range(ACTION_HORIZON)),
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


def runtime_modality_config_source() -> str:
    """Return a self-contained config consumed by pinned Isaac-GR00T.

    The prepared dataset stores absolute 12D targets.  The two arm groups are
    marked relative here, so the pinned loader performs that subtraction once;
    both grippers remain absolute.
    """

    return """from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS, register_modality_config
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


def write_runtime_modality_config(destination: str | Path) -> Path:
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
            stream.write(runtime_modality_config_source())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path
