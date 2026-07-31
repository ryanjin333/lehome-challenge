"""Checked organizer-to-GR00T mapping contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


JOINT_BASENAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
JOINT_NAMES = tuple(
    f"{side}_{joint}"
    for side in ("left", "right")
    for joint in JOINT_BASENAMES
)
CAMERA_KEYS = (
    "observation.images.top_rgb",
    "observation.images.left_rgb",
    "observation.images.right_rgb",
)
FIXED_INSTRUCTION = "fold the garment on the table"
ACTION_HORIZON = 16


def expected_mapping() -> dict[str, object]:
    """Return the sole accepted mapping for ``four_types_merged``."""

    return {
        "schema_version": 1,
        "source_dataset": "four_types_merged",
        "state": {
            "source_key": "observation.state",
            "target_key": "observation.state",
            "dimension": 12,
            "names": list(JOINT_NAMES),
        },
        "action": {
            "source_key": "action",
            "target_key": "action",
            "dimension": 12,
            "names": list(JOINT_NAMES),
            "storage": "absolute",
            "groot_transform": [
                {"group": "left_arm", "indices": list(range(0, 5)), "mode": "relative"},
                {"group": "left_gripper", "indices": [5], "mode": "absolute"},
                {"group": "right_arm", "indices": list(range(6, 11)), "mode": "relative"},
                {"group": "right_gripper", "indices": [11], "mode": "absolute"},
            ],
        },
        "cameras": [
            {"source_key": key, "target_modality": key.rsplit(".", 1)[-1]}
            for key in CAMERA_KEYS
        ],
        "language": {
            "instruction": FIXED_INSTRUCTION,
            "target_modality": "annotation.human.task_description",
        },
        "future_actions": {
            "horizon": ACTION_HORIZON,
            "loader_allow_padding": False,
            "materialized_windows": False,
            "tail_convention": "drop_incomplete_windows",
        },
    }


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate mapping field: {key}")
        result[key] = value
    return result


def load_checked_mapping(path: str | Path | None) -> dict[str, object]:
    """Load the checked mapping and reject every missing, extra, or changed field."""

    if path is None:
        raise ValueError("checked mapping JSON is required")
    mapping_path = Path(path)
    try:
        decoded = json.loads(
            mapping_path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("checked mapping JSON is invalid") from error
    if not isinstance(decoded, Mapping):
        raise ValueError("checked mapping JSON root must be an object")
    mapping: dict[str, Any] = dict(decoded)
    if mapping != expected_mapping():
        raise ValueError(
            "checked mapping JSON does not exactly match the four_types_merged contract"
        )
    return mapping
