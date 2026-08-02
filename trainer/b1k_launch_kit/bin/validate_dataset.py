#!/usr/bin/env python3
"""Validate the complete RGB-only BEHAVIOR 2026 training dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable


CAMERAS = (
    "observation.rgb.zed_link_camera_0",
    "observation.rgb.left_realsense_link_camera_0",
    "observation.rgb.right_realsense_link_camera_0",
)
EXPECTED_TASKS = 100
EXPECTED_DEMOS_PER_TASK = 200
REQUIRED_META = ("info.json", "stats.json", "tasks.parquet")


class ValidationError(RuntimeError):
    """Raised when the local dataset does not satisfy the training contract."""


def parquet_row_count(path: Path) -> int:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise ValidationError(
            "PyArrow is required for production validation. Run this with the GR00T venv."
        ) from exc
    return parquet.ParquetFile(path).metadata.num_rows


def _require_nonempty_files(path: Path, description: str) -> None:
    if not path.is_dir() or not any(item.is_file() and item.stat().st_size for item in path.rglob("*")):
        raise ValidationError(f"Missing or empty {description}: {path}")


def validate_dataset(
    root: Path | str,
    row_counter: Callable[[Path], int] = parquet_row_count,
) -> dict[str, int]:
    root = Path(root)
    if not root.is_dir():
        raise ValidationError(f"Dataset root does not exist: {root}")

    _require_nonempty_files(root / "annotations", "annotations")
    for filename in REQUIRED_META:
        file_path = root / "meta" / filename
        if not file_path.is_file() or file_path.stat().st_size == 0:
            raise ValidationError(f"Missing required metadata: {file_path}")

    depth_dirs = list((root / "videos").glob("observation.depth_linear.*"))
    if depth_dirs:
        raise ValidationError(f"Depth data is present but excluded by the manifest: {depth_dirs[0]}")

    total_demos = 0
    for task_id in range(EXPECTED_TASKS):
        chunk = f"chunk-{task_id:03d}"
        _require_nonempty_files(root / "data" / chunk, f"data chunk {chunk}")
        for camera in CAMERAS:
            _require_nonempty_files(
                root / "videos" / camera / chunk,
                f"RGB stream {camera} for {chunk}",
            )

        episode_dir = root / "meta" / "episodes" / chunk
        parquet_files = sorted(episode_dir.rglob("*.parquet")) if episode_dir.is_dir() else []
        if not parquet_files:
            raise ValidationError(f"Missing episode metadata for {chunk}: {episode_dir}")
        chunk_demos = sum(row_counter(path) for path in parquet_files)
        if chunk_demos != EXPECTED_DEMOS_PER_TASK:
            raise ValidationError(
                f"{chunk} contains {chunk_demos} demonstrations; expected {EXPECTED_DEMOS_PER_TASK}"
            )
        total_demos += chunk_demos

    expected_total = EXPECTED_TASKS * EXPECTED_DEMOS_PER_TASK
    if total_demos != expected_total:
        raise ValidationError(f"Found {total_demos} demonstrations; expected {expected_total}")

    return {
        "tasks": EXPECTED_TASKS,
        "demonstrations": total_demos,
        "rgb_streams": len(CAMERAS),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()
    try:
        result = validate_dataset(args.dataset_root)
    except ValidationError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
