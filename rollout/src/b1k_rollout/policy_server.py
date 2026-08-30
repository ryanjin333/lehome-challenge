"""Launch the pinned upstream B1K R1Pro WebSocket policy server unchanged."""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys


_GROOT_PYTHON = os.environ.get("GROOT_PYTHON", "/opt/isaac-groot/.venv/bin/python")
_SERVER = "/opt/isaac-groot/scripts/b1k/serve_b1k.py"
_MODALITY = "/opt/isaac-groot/examples/b1k/r1pro.py"


def build_command(*, checkpoint: Path, host: str, port: int) -> tuple[str, ...]:
    if checkpoint.is_symlink() or not checkpoint.is_dir() or not any(checkpoint.iterdir()):
        raise ValueError("checkpoint must be a non-empty real directory")
    if host != "127.0.0.1" or not 1 <= port <= 65535:
        raise ValueError("policy server must bind authenticated loopback on a valid port")
    return (
        _GROOT_PYTHON, _SERVER,
        "--model-path", str(checkpoint),
        "--modality-config-path", _MODALITY,
        "--embodiment-tag", "NEW_EMBODIMENT",
        "--host", host,
        "--port", str(port),
    )


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="b1k-rollout-policy-server")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args(arguments)
    try:
        command = build_command(checkpoint=Path(args.checkpoint), host=args.host, port=args.port)
    except ValueError as error:
        parser.error(str(error))
    os.execv(command[0], list(command))
    return 127  # pragma: no cover - exec replaces the process


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
