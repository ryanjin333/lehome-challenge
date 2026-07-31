from __future__ import annotations

import json
from pathlib import Path
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
                "video.codec": "fixture",
                "video.pix_fmt": "rgb24",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    info: dict[str, object] = {
        "codebase_version": "v2.1",
        "robot_type": "dual_so101_follower",
        "total_episodes": len(episode_ids),
        "total_frames": len(episode_ids) * frames_per_episode,
        "total_tasks": 1,
        "total_videos": len(episode_ids) * len(CAMERA_KEYS),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{len(episode_ids)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "features": features,
    }
    if mutate_info is not None:
        mutate_info(info)
    _write_json(dataset / "meta" / "info.json", info)
    _write_json(
        dataset / "meta" / "modality.json",
        {"source": "organizer fixture"},
    )
    (dataset / "meta" / "tasks.jsonl").write_text(
        '{"task":"organizer task","task_index":0}\n',
        encoding="utf-8",
    )

    episode_lines: list[str] = []
    global_index = 0
    for episode_id in episode_ids:
        rows: dict[str, list[object]] = {
            "observation.state": [],
            "action": [],
            "timestamp": [],
            "frame_index": [],
            "episode_index": [],
            "index": [],
            "task_index": [],
        }
        for frame_index in range(frames_per_episode):
            state = [
                float(episode_id * 100 + frame_index * 10 + dimension)
                for dimension in range(12)
            ]
            action = [
                float(episode_id * 1000 + frame_index * 10 + dimension)
                for dimension in range(12)
            ]
            rows["observation.state"].append(state)
            rows["action"].append(action)
            rows["timestamp"].append(frame_index / fps)
            rows["frame_index"].append(frame_index)
            rows["episode_index"].append(episode_id)
            rows["index"].append(global_index)
            rows["task_index"].append(0)
            global_index += 1
        if mutate_rows is not None:
            mutate_rows(episode_id, rows)
        table = pa.table(
            {
                "observation.state": pa.array(
                    rows["observation.state"], type=pa.list_(pa.float32(), 12)
                ),
                "action": pa.array(rows["action"], type=pa.list_(pa.float32(), 12)),
                "timestamp": pa.array(rows["timestamp"], type=pa.float32()),
                "frame_index": pa.array(rows["frame_index"], type=pa.int64()),
                "episode_index": pa.array(rows["episode_index"], type=pa.int64()),
                "index": pa.array(rows["index"], type=pa.int64()),
                "task_index": pa.array(rows["task_index"], type=pa.int64()),
            }
        )
        parquet_path = (
            dataset / "data" / "chunk-000" / f"episode_{episode_id:06d}.parquet"
        )
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, parquet_path, compression="zstd")
        for camera_key in CAMERA_KEYS:
            video_path = (
                dataset
                / "videos"
                / "chunk-000"
                / camera_key
                / f"episode_{episode_id:06d}.mp4"
            )
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(
                b"safe-fixture-rgb-stream:"
                + camera_key.encode()
                + b":"
                + str(episode_id).encode()
            )
        episode_lines.append(
            json.dumps(
                {
                    "episode_index": episode_id,
                    "tasks": ["organizer task"],
                    "length": frames_per_episode,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    (dataset / "meta" / "episodes.jsonl").write_text(
        "\n".join(episode_lines) + "\n",
        encoding="utf-8",
    )
    return dataset
