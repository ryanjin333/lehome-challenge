from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

import lehome_train.data.convert as converter
from fixtures.source_dataset import (
    CAMERA_KEYS,
    JOINT_NAMES,
    count_video_frames,
    count_video_keyframes,
    first_frame_dominant_channel,
    make_source_dataset,
    video_fps,
)
from lehome_train.data.convert import convert_dataset
from lehome_train.data.split import split_episode_ids


MAPPING_PATH = (
    Path(__file__).parents[1] / "config" / "lehome_four_types_mapping.json"
)
CONVERTER_COMMIT = "0123456789abcdef0123456789abcdef01234567"
SOURCE_REVISION = "89abcdef0123456789abcdef0123456789abcdef"
CONTAINER_DIGEST = "sha256:" + ("a" * 64)
SOURCE_REPOSITORY = "ryanjin333/four_types_merged"


def _convert(source: Path, destination: Path, **kwargs):
    return convert_dataset(
        source,
        destination,
        mapping_path=MAPPING_PATH,
        source_repository=SOURCE_REPOSITORY,
        source_revision=SOURCE_REVISION,
        converter_commit=CONVERTER_COMMIT,
        converter_container_digest=CONTAINER_DIGEST,
        **kwargs,
    )


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
            source_repository=SOURCE_REPOSITORY,
            source_revision=SOURCE_REVISION,
            converter_commit=CONVERTER_COMMIT,
            converter_container_digest=CONTAINER_DIGEST,
        )


def test_conversion_preserves_absolute_actions_and_builds_groot_layout(
    tmp_path: Path,
) -> None:
    source = make_source_dataset(tmp_path)
    source_action = pq.read_table(
        source / "data" / "chunk-000" / "file-000.parquet"
    )["action"].slice(18, 18).to_pylist()
    consolidated_video = (
        source
        / "videos"
        / "observation.images.top_rgb"
        / "chunk-000"
        / "file-000.mp4"
    )
    assert count_video_keyframes(consolidated_video) < count_video_frames(
        consolidated_video
    )

    manifest = _convert(
        source,
        tmp_path / "output",
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
    for episode_id in (3, 7, 11):
        for camera_key in CAMERA_KEYS:
            output_video = (
                output
                / "videos"
                / "chunk-000"
                / camera_key
                / f"episode_{episode_id:06d}.mp4"
            )
            assert count_video_frames(output_video) == 18
            assert video_fps(output_video) == 10.0
    assert first_frame_dominant_channel(
        output
        / "videos"
        / "chunk-000"
        / "observation.images.top_rgb"
        / "episode_000007.mp4"
    ) == "blue"
    assert first_frame_dominant_channel(
        output
        / "videos"
        / "chunk-000"
        / "observation.images.top_rgb"
        / "episode_000003.mp4"
    ) == "red"
    output_info = json.loads(
        (output / "meta" / "info.json").read_text(encoding="utf-8")
    )
    assert output_info["codebase_version"] == "v2.1"
    assert output_info["data_path"] == (
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    assert "data_files_size_in_mb" not in output_info
    assert "video_files_size_in_mb" not in output_info
    assert not (output / "meta" / "stats.json").exists()
    assert not (output / "meta" / "relative_stats.json").exists()
    assert manifest["statistics"] == {
        "status": "pending_task_4_train_only",
        "files": [],
    }
    assert "eef" not in json.dumps(
        {
            "action_schema": manifest["action_schema"],
            "state_schema": manifest["state_schema"],
        }
    ).lower()
    assert manifest["source_repository"] == SOURCE_REPOSITORY
    assert manifest["source_revision"] == SOURCE_REVISION
    assert manifest["converter_container_digest"] == CONTAINER_DIGEST
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

    first = _convert(
        source,
        tmp_path / "first",
        split_seed=5,
    )
    second = _convert(
        source,
        tmp_path / "second",
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


def test_conversion_validates_every_episode_camera_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_source_dataset(tmp_path)
    validated: list[Path] = []
    original = converter._validate_output_video

    def track_validation(
        path: Path,
        *,
        expected_frame_count: int,
        expected_fps: float,
    ) -> None:
        original(
            path,
            expected_frame_count=expected_frame_count,
            expected_fps=expected_fps,
        )
        validated.append(path)

    monkeypatch.setattr(converter, "_validate_output_video", track_validation)

    _convert(source, tmp_path / "output")

    assert len(validated) == 3 * len(CAMERA_KEYS)


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
            source_repository=SOURCE_REPOSITORY,
            source_revision=SOURCE_REVISION,
            converter_commit=CONVERTER_COMMIT,
            converter_container_digest=CONTAINER_DIGEST,
        )

    valid_source = make_source_dataset(tmp_path / "valid")
    destination = tmp_path / "completed"
    destination.mkdir()
    (destination / "manifest.json").write_text('{"incompatible":true}', encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _convert(
            valid_source,
            destination,
        )


def test_conversion_refuses_to_write_inside_source_dataset(tmp_path: Path) -> None:
    source = make_source_dataset(tmp_path)

    with pytest.raises(ValueError, match="inside the source dataset"):
        _convert(
            source,
            source / "prepared",
        )

    assert not (source / "prepared").exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_repository", "", "source_repository"),
        ("source_revision", "main", "source_revision"),
        ("converter_container_digest", "sha256:latest", "container digest"),
    ],
)
def test_conversion_requires_immutable_provenance_inputs(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    source = make_source_dataset(tmp_path)
    arguments = {
        "mapping_path": MAPPING_PATH,
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "converter_commit": CONVERTER_COMMIT,
        "converter_container_digest": CONTAINER_DIGEST,
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=message):
        convert_dataset(source, tmp_path / "output", **arguments)
