#!/usr/bin/env python3
"""Derive or verify the immutable B100 rollout manifest from pinned task data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from b1k_rollout.task_manifest import build_task_manifest, render_task_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-data",
        type=Path,
        required=True,
        help="Pinned docs/challenge/task_data.json source file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "task-manifest.json",
        help="Manifest path to write or compare (default: rollout/task-manifest.json).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail instead of writing when the generated file is absent or stale.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        expected = render_task_manifest(build_task_manifest(args.task_data))
    except ValueError as error:
        print(f"task manifest generation failed: {error}", file=sys.stderr)
        return 2

    try:
        actual = args.output.read_text(encoding="utf-8")
    except OSError:
        actual = None
    if args.check:
        if actual != expected:
            print("task manifest is stale; regenerate it from the pinned task data", file=sys.stderr)
            return 1
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(expected, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
