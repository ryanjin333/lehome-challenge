"""Dedicated, receipt-producing GR00T server for the public96 contract only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import threading

from scripts.eval_groot_n17_public96 import CHECKPOINT, await_authenticated_policy_server_ready, canonical_policy_artifact_sha256
from scripts.groot_n17_public96_raw_checker import RAW_CHECKER_OVERLAY_ID, overlay_sha256


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True); parser.add_argument("--host", required=True); parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--api-token-env", required=True); parser.add_argument("--device", choices=("cuda:0",), required=True); parser.add_argument("--seed", type=int, required=True); parser.add_argument("--readiness-receipt", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> int:
    if args.host != "127.0.0.1" or not 1 <= args.port <= 65535 or args.readiness_receipt.exists() or args.readiness_receipt.is_symlink(): raise ValueError("public96 policy server invocation is unsafe")
    artifact = canonical_policy_artifact_sha256(args.model_path)
    if artifact != CHECKPOINT["artifact_sha256"]: raise ValueError("public96 policy artifact SHA-256 mismatch")
    token = os.environ.get(args.api_token_env, "")
    if len(token) < 32: raise ValueError("public96 policy token is missing")
    if hasattr(signal, "pthread_sigmask"): signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy import Gr00tPolicy
    from gr00t.policy.server_client import PolicyServer
    from scripts.run_groot_policy_server import seed_policy_runtime
    seed_policy_runtime(args.seed)
    policy = Gr00tPolicy(embodiment_tag=EmbodimentTag.NEW_EMBODIMENT, model_path=str(args.model_path), device=args.device, strict=True)
    server = PolicyServer(policy, host=args.host, port=args.port, api_token=token)
    failures: list[BaseException] = []

    def serve() -> None:
        try:
            server.run()
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=serve, name="public96-policy-server", daemon=False)
    thread.start()
    try:
        await_authenticated_policy_server_ready(
            port=args.port, token=token, readiness_timeout=10.0, request_timeout=1.0,
        )
    except BaseException:
        if failures:
            raise RuntimeError("public96 policy server failed before authenticated readiness") from failures[0]
        raise
    receipt = {"kind": "lehome_groot_n17_public96_policy_server_readiness_v1", "artifact_sha256": artifact, "runtime_policy_sha256": CHECKPOINT["runtime_policy_sha256"], "model_path": str(args.model_path.resolve()), "device": args.device, "adapter": "nvidia_gr00t_policy_server_public96_v1", "raw_checker_overlay": {"id": RAW_CHECKER_OVERLAY_ID, "sha256": overlay_sha256()}}
    args.readiness_receipt.parent.mkdir(parents=True, exist_ok=True); args.readiness_receipt.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    thread.join()
    if failures:
        raise RuntimeError("public96 policy server exited") from failures[0]
    return 0


if __name__ == "__main__": raise SystemExit(run(_parser().parse_args()))
