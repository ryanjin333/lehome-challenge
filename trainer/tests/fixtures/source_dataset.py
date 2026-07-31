from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Callable

import pyarrow as pa
import pyarrow.parquet as pq


JOINT_NAMES = [
    f"{side}_{joint}"
    for side in ("left", "right")
    for joint in (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    )
]
CAMERA_KEYS = (
    "observation.images.top_rgb",
    "observation.images.left_rgb",
    "observation.images.right_rgb",
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


@lru_cache(maxsize=None)
def _fixture_video(frame_counts: tuple[int, ...], fps: int) -> Path:
    frame_count = sum(frame_counts)
    counts_slug = "-".join(str(count) for count in frame_counts)
    path = (
        Path(tempfile.gettempdir())
        / f"lehome-v3-fixture-{counts_slug}-frames-{fps}-fps.mp4"
    )
    colors = ("blue", "red", "green", "yellow")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    for index, count in enumerate(frame_counts):
        command.extend(
            [
                "-f",
                "lavfi",
                "-i",
                (
                    f"color=c={colors[index % len(colors)]}:"
                    f"s=640x480:r={fps}:d={count / fps:.9f}"
                ),
            ]
        )
    if len(frame_counts) > 1:
        inputs = "".join(f"[{index}:v]" for index in range(len(frame_counts)))
        command.extend(
            [
                "-filter_complex",
                f"{inputs}concat=n={len(frame_counts)}:v=1:a=0[outv]",
                "-map",
                "[outv]",
            ]
        )
    command.extend(
        [
            "-frames:v",
            str(frame_count),
            "-c:v",
            "libx264",
            "-g",
            "30",
            "-sc_threshold",
            "0",
            "-threads",
            "1",
            "-pix_fmt",
            "yuv420p",
            "-fflags",
            "+bitexact",
            "-y",
            str(path),
        ]
    )
    subprocess.run(
        command,
        check=True,
        timeout=30,
    )
    return path


def count_video_frames(path: Path) -> int:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return int(completed.stdout.strip())


def video_fps(path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    numerator, denominator = completed.stdout.strip().split("/", maxsplit=1)
    return int(numerator) / int(denominator)


def count_video_keyframes(path: Path) -> int:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-skip_frame",
            "nokey",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=key_frame",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return len(completed.stdout.splitlines())


def first_frame_dominant_channel(path: Path) -> str:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-vf",
            "scale=1:1",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    red, green, blue = completed.stdout
    return max(("red", red), ("green", green), ("blue", blue), key=lambda item: item[1])[0]


def make_source_dataset(
    root: Path,
    *,
    episode_ids: tuple[int, ...] = (7, 3, 11),
    frames_per_episode: int = 18,
    fps: float = 10.0,
    mutate_info: Callable[[dict[str, object]], None] | None = None,
    mutate_rows: Callable[[int, dict[str, list[object]]], None] | None = None,
) -> Path:
    dataset = root / "four_types_merged"
    features: dict[str, object] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [12],
            "names": JOINT_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": [12],
            "names": JOINT_NAMES,
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for camera_key in CAMERA_KEYS:
        features[camera_key] = {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.channels": 3,
                "video.fps": fps,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    info: dict[str, object] = {
        "codebase_version": "v3.0",
        "robot_type": "dual_so101_follower",
        "total_episodes": len(episode_ids),
        "total_frames": len(episode_ids) * frames_per_episode,
        "total_tasks": 1,
        "total_videos": (
            len({index // 2 for index in range(len(episode_ids))})
            * len(CAMERA_KEYS)
        ),
        "total_chunks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": fps,
        "splits": {"train": f"0:{len(episode_ids)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": (
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        ),
        "features": features,
    }
    if mutate_info is not None:
        mutate_info(info)
    _write_json(dataset / "meta" / "info.json", info)
    statistic = {
        name: [0.0] * 12
        for name in ("mean", "std", "min", "max", "q01", "q99")
    }
    _write_json(
        dataset / "meta" / "stats.json",
        {
            "observation.state": statistic,
            "action": statistic,
        },
    )
    pq.write_table(
        pa.table(
            {
                "task": pa.array(["organizer task"], type=pa.string()),
                "task_index": pa.array([0], type=pa.int64()),
            }
        ),
        dataset / "meta" / "tasks.parquet",
    )

    # Two consolidated v3 shards: the first contains two episodes, the second one.
    file_assignments = tuple(index // 2 for index in range(len(episode_ids)))
    global_index = 0
    file_tables: dict[int, list[pa.Table]] = {
        file_index: [] for file_index in set(file_assignments)
    }
    episode_records: list[dict[str, object]] = []
    file_local_offsets = {file_index: 0 for file_index in file_tables}
    for episode_id, file_index in zip(episode_ids, file_assignments, strict=True):
        rows: dict[str, list[object]] = {
            "observation.state": [],
            "action": [],
            "timestamp": [],
            "frame_index": [],
            "episode_index": [],
            "index": [],
            "task_index": [],
        }
        dataset_from_index = global_index
        for frame_index in range(frames_per_episode):
            rows["observation.state"].append(
                [
                    float(episode_id * 100 + frame_index * 10 + dimension)
                    for dimension in range(12)
                ]
            )
            rows["action"].append(
                [
                    float(episode_id * 1000 + frame_index * 10 + dimension)
                    for dimension in range(12)
                ]
            )
            rows["timestamp"].append(frame_index / fps)
            rows["frame_index"].append(frame_index)
            rows["episode_index"].append(episode_id)
            rows["index"].append(global_index)
            rows["task_index"].append(0)
            global_index += 1
        if mutate_rows is not None:
            mutate_rows(episode_id, rows)
        file_tables[file_index].append(
            pa.table(
                {
                    "observation.state": pa.array(
                        rows["observation.state"], type=pa.list_(pa.float32(), 12)
                    ),
                    "action": pa.array(
                        rows["action"], type=pa.list_(pa.float32(), 12)
                    ),
                    "timestamp": pa.array(rows["timestamp"], type=pa.float32()),
                    "frame_index": pa.array(rows["frame_index"], type=pa.int64()),
                    "episode_index": pa.array(rows["episode_index"], type=pa.int64()),
                    "index": pa.array(rows["index"], type=pa.int64()),
                    "task_index": pa.array(rows["task_index"], type=pa.int64()),
                }
            )
        )
        local_start = file_local_offsets[file_index]
        local_end = local_start + frames_per_episode
        file_local_offsets[file_index] = local_end
        record: dict[str, object] = {
            "episode_index": episode_id,
            "tasks": ["organizer task"],
            "length": frames_per_episode,
            "dataset_from_index": dataset_from_index,
            "dataset_to_index": global_index,
            "data/chunk_index": 0,
            "data/file_index": file_index,
        }
        for camera_key in CAMERA_KEYS:
            record[f"videos/{camera_key}/chunk_index"] = 0
            record[f"videos/{camera_key}/file_index"] = file_index
            record[f"videos/{camera_key}/from_timestamp"] = local_start / fps
            record[f"videos/{camera_key}/to_timestamp"] = local_end / fps
        episode_records.append(record)

    for file_index, tables in file_tables.items():
        path = dataset / "data" / "chunk-000" / f"file-{file_index:03d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.concat_tables(tables), path, compression="zstd")
        source_video = _fixture_video(
            tuple(table.num_rows for table in tables),
            int(fps),
        )
        for camera_key in CAMERA_KEYS:
            video_path = (
                dataset
                / "videos"
                / camera_key
                / "chunk-000"
                / f"file-{file_index:03d}.mp4"
            )
            video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_video, video_path)

    episodes_path = dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(episode_records), episodes_path, compression="zstd")
    return dataset
