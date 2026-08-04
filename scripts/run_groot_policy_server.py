"""Run NVIDIA's pinned GR00T PolicyServer on authenticated loopback only."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import sys


def unblock_termination_signals() -> None:
    """Undo the parent's launch-time signal block before loading GR00T."""

    if hasattr(signal, "pthread_sigmask"):
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--api-token-env", required=True)
    return parser


def run(args: argparse.Namespace) -> int:
    unblock_termination_signals()
    if args.host != "127.0.0.1":
        raise ValueError("GR00T policy server must bind only 127.0.0.1")
    if not 1 <= args.port <= 65535:
        raise ValueError("GR00T policy server port must be in 1..65535")
    if args.model_path.is_symlink() or not args.model_path.is_dir():
        raise ValueError("GR00T policy model path must be a materialized directory")
    api_token = os.environ.get(args.api_token_env, "")
    if len(api_token) < 32:
        raise ValueError("GR00T policy server API token is missing or too short")
    try:
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy import Gr00tPolicy
        from gr00t.policy.server_client import PolicyServer
    except ImportError as error:
        raise RuntimeError("pinned NVIDIA GR00T PolicyServer is unavailable") from error
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        model_path=str(args.model_path),
        device="cuda:0",
        strict=True,
    )
    # The pinned NVIDIA PolicyServer exposes construction and ``run()`` only.
    server = PolicyServer(policy, host=args.host, port=args.port, api_token=api_token)
    server.run()
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(build_parser().parse_args(argv))
    except (RuntimeError, ValueError) as error:
        # Deliberately omit the token and environment value from every error.
        print(f"policy server error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
