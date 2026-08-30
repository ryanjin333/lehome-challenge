from __future__ import annotations

import json
from pathlib import Path

import pytest

from fixtures.source_dataset import CAMERA_KEYS, JOINT_NAMES, make_source_dataset
from lehome_train.data.inspect import inspect_dataset


def test_inspection_reports_observed_schema_and_proposed_mapping(tmp_path: Path) -> None:
    source = make_source_dataset(tmp_path)
    report_path = tmp_path / "inspection.json"

    report = inspect_dataset(source, output_path=report_path)

    assert report["valid"] is True
    assert report["source_format"] == "lerobot_v3_sharded"
    assert report["dataset_name"] == "four_types_merged"
    assert report["episode_ids"] == ["3", "7", "11"]
    assert report["frame_count"] == 54
    assert report["fps"] == 10.0
    assert report["observed_schema"]["state"]["names"] == JOINT_NAMES
    assert report["observed_schema"]["action"]["storage"] == "absolute"
    assert [item["source_key"] for item in report["proposed_mapping"]["cameras"]] == list(
        CAMERA_KEYS
    )
    assert report["observed_schema"]["cameras"][0]["shape"] == [480, 640, 3]
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert (
        source / "data" / "chunk-000" / "file-000.parquet"
    ).is_file()
    assert (
        source / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ).is_file()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda info: info["features"].pop("observation.images.left_rgb"),
            "missing camera",
        ),
        (
            lambda info: info["features"].update(
                {
                    "observation.images.extra_rgb": {
                        "dtype": "video",
                        "shape": [480, 640, 3],
                        "names": ["height", "width", "channels"],
                        "info": {"video.fps": 10.0},
                    }
                }
            ),
            "extra camera",
        ),
        (
            lambda info: info["features"]["action"].update({"shape": [11]}),
            "action shape",
        ),
        (
            lambda info: info["features"]["observation.state"].update(
                {"names": list(reversed(JOINT_NAMES))}
            ),
            "joint order",
        ),
    ],
)
def test_inspection_surfaces_schema_drift_without_guessing(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    source = make_source_dataset(tmp_path, mutate_info=mutation)

    report = inspect_dataset(source)

    assert report["valid"] is False
    assert any(message in error for error in report["validation_errors"])
    assert report["proposed_mapping"] is None


def test_inspection_rejects_nonfinite_values(tmp_path: Path) -> None:
    def mutate_rows(episode_id: int, rows: dict[str, list[object]]) -> None:
        if episode_id == 3:
            rows["action"][2][4] = float("nan")

    source = make_source_dataset(tmp_path, mutate_rows=mutate_rows)

    report = inspect_dataset(source)

    assert report["valid"] is False
    assert any("non-finite action" in error for error in report["validation_errors"])


@pytest.mark.parametrize(
    ("mutate_rows", "message"),
    [
        (
            lambda episode_id, rows: (
                rows["timestamp"].__setitem__(4, rows["timestamp"][4] + 0.02)
                if episode_id == 7
                else None
            ),
            "timestamp/frame alignment",
        ),
        (
            lambda episode_id, rows: (
                rows["frame_index"].__setitem__(4, 5) if episode_id == 7 else None
            ),
            "frame_index alignment",
        ),
    ],
)
def test_inspection_rejects_timing_and_frame_alignment_drift(
    tmp_path: Path,
    mutate_rows,
    message: str,
) -> None:
    source = make_source_dataset(tmp_path, mutate_rows=mutate_rows)

    report = inspect_dataset(source)

    assert report["valid"] is False
    assert any(message in error for error in report["validation_errors"])
