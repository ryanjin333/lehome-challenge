from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from fixtures.source_dataset import CAMERA_KEYS, JOINT_NAMES, make_source_dataset
from lehome_train.data.convert import convert_dataset
from lehome_train.data.split import split_episode_ids


MAPPING_PATH = (
    Path(__file__).parents[1] / "config" / "lehome_four_types_mapping.json"
)
CONVERTER_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def test_episode_split_is_stable_seeded_and_never_splits_frames() -> None:
    first = split_episode_ids(["11", "3", "7", "2"], seed=91, validation_fraction=0.25)
    second = split_episode_ids(["7", "2", "11", "3"], seed=91, validation_fraction=0.25)

    assert first == second
    assert set(first.train).isdisjoint(first.validation)
    assert sorted(first.train + first.validation) == ["11", "2", "3", "7"]


def test_conversion_requires_checked_mapping(tmp_path: Path) -> None:
    source = make_source_dataset(tmp_path)

    with pytest.raises(ValueError, match="checked mapping JSON is required"):
        convert_dataset(
            source,
            tmp_path / "output",
            mapping_path=None,
            converter_commit=CONVERTER_COMMIT,
        )


def test_conversion_preserves_absolute_actions_and_builds_groot_layout(
    tmp_path: Path,
) -> None:
    source = make_source_dataset(tmp_path)
    source_action = pq.read_table(
        source / "data" / "chunk-000" / "episode_000003.parquet"
    )["action"].to_pylist()

    manifest = convert_dataset(
        source,
        tmp_path / "output",
        mapping_path=MAPPING_PATH,
        converter_commit=CONVERTER_COMMIT,
        split_seed=17,
        validation_fraction=1 / 3,
    )

    output = tmp_path / "output"
    converted = pq.read_table(
        output / "data" / "chunk-000" / "episode_000003.parquet"
    )
    assert converted["action"].to_pylist() == source_action
    assert converted["episode_index"].to_pylist() == [3] * 18
    assert converted["frame_index"].to_pylist() == list(range(18))
    assert converted["task_index"].to_pylist() == [0] * 18
    assert "eef" not in json.dumps(manifest).lower()
    assert manifest["action_schema"]["storage"] == "absolute"
    assert manifest["action_schema"]["dimension"] == 12
    assert manifest["action_schema"]["names"] == JOINT_NAMES
    assert manifest["fixed_language_instruction"] == "fold the garment on the table"
    assert manifest["future_actions"] == {
        "horizon": 16,
        "loader_allow_padding": False,
        "materialized_windows": False,
        "tail_convention": "drop_incomplete_windows",
        "valid_action_mask": "implicit_all_true_for_emitted_windows",
        "valid_window_counts": {"3": 3, "7": 3, "11": 3},
    }
    assert [camera["source_key"] for camera in manifest["camera_schema"]] == list(
        CAMERA_KEYS
    )
    modality = json.loads(
        (output / "meta" / "modality.json").read_text(encoding="utf-8")
    )
    assert modality["action"]["left_arm"] == {
        "end": 5,
        "original_key": "action",
        "start": 0,
    }
    assert modality["action"]["left_gripper"]["start"] == 5
    assert modality["action"]["right_arm"]["start"] == 6
    assert modality["action"]["right_gripper"]["start"] == 11
    assert json.loads(
        (output / "meta" / "tasks.jsonl").read_text(encoding="utf-8")
    )["task"] == "fold the garment on the table"


def test_conversion_is_deterministic_and_does_not_mutate_source(tmp_path: Path) -> None:
    source = make_source_dataset(tmp_path)
    before = hashlib.sha256(
        b"".join(path.read_bytes() for path in sorted(source.rglob("*")) if path.is_file())
    ).hexdigest()

    first = convert_dataset(
        source,
        tmp_path / "first",
        mapping_path=MAPPING_PATH,
        converter_commit=CONVERTER_COMMIT,
        split_seed=5,
    )
    second = convert_dataset(
        source,
        tmp_path / "second",
        mapping_path=MAPPING_PATH,
        converter_commit=CONVERTER_COMMIT,
        split_seed=5,
    )

    assert first == second
    assert (tmp_path / "first" / "manifest.json").read_bytes() == (
        tmp_path / "second" / "manifest.json"
    ).read_bytes()
    after = hashlib.sha256(
        b"".join(path.read_bytes() for path in sorted(source.rglob("*")) if path.is_file())
    ).hexdigest()
    assert after == before


def test_conversion_fails_closed_on_drift_or_incompatible_destination(
    tmp_path: Path,
) -> None:
    source = make_source_dataset(
        tmp_path,
        mutate_info=lambda info: info["features"]["action"].update({"shape": [11]}),
    )
    with pytest.raises(ValueError, match="source schema validation failed"):
        convert_dataset(
            source,
            tmp_path / "output",
            mapping_path=MAPPING_PATH,
            converter_commit=CONVERTER_COMMIT,
        )

    valid_source = make_source_dataset(tmp_path / "valid")
    destination = tmp_path / "completed"
    destination.mkdir()
    (destination / "manifest.json").write_text('{"incompatible":true}', encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        convert_dataset(
            valid_source,
            destination,
            mapping_path=MAPPING_PATH,
            converter_commit=CONVERTER_COMMIT,
        )


def test_conversion_refuses_to_write_inside_source_dataset(tmp_path: Path) -> None:
    source = make_source_dataset(tmp_path)

    with pytest.raises(ValueError, match="inside the source dataset"):
        convert_dataset(
            source,
            source / "prepared",
            mapping_path=MAPPING_PATH,
            converter_commit=CONVERTER_COMMIT,
        )

    assert not (source / "prepared").exists()
