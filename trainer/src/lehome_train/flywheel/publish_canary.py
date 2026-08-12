"""Explicit private corrective-canary publisher; HF_TOKEN stays process-local."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from lehome_train.flywheel.publish import (
    build_corrective_canary_abort_publication_bundle,
    build_corrective_canary_publication_bundle,
    publish_private_corrective_canary,
    publish_private_corrective_canary_abort,
)
from lehome_train.hub import HuggingFaceHubTransport
from lehome_train.hub import HubTransport


def _object(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError(f"{label} is malformed") from None
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("success", "abort"), required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--disposal-receipt", type=Path, required=True)
    parser.add_argument("--canary-manifest", type=Path, required=True)
    parser.add_argument("--source-wave-manifest", type=Path, required=True)
    parser.add_argument("--provider-evidence", type=Path, required=True)
    parser.add_argument("--provider-snapshot", type=Path, required=True)
    parser.add_argument("--instance-receipt", type=Path, required=True)
    parser.add_argument("--terminal-receipt", type=Path, required=True)
    parser.add_argument("--synced-evidence-root", type=Path, required=True)
    parser.add_argument("--attempt-receipt", type=Path)
    parser.add_argument("--raw-episode-root", type=Path)
    parser.add_argument("--policy-receipt", type=Path)
    return parser


def run(args: argparse.Namespace, *, transport: HubTransport) -> dict[str, str]:
    """Construct the requested typed bundle and publish it through one transport."""

    common = {
        "canary_manifest_path": args.canary_manifest,
        "source_wave_manifest_path": args.source_wave_manifest,
        "provider_evidence_path": args.provider_evidence,
        "provider_snapshot_path": args.provider_snapshot,
        "instance_receipt_path": args.instance_receipt,
        "synced_evidence_root": args.synced_evidence_root,
    }
    if args.kind == "success":
        if args.attempt_receipt is None or args.raw_episode_root is None or args.policy_receipt is None:
            raise ValueError("success canary requires attempt, raw episode, and policy receipts")
        bundle = build_corrective_canary_publication_bundle(
            attempt_receipt=_object(args.attempt_receipt, "attempt receipt"),
            raw_episode_root=args.raw_episode_root, policy_receipt_path=args.policy_receipt,
            terminal_receipt_path=args.terminal_receipt, **common,
        )
        result = publish_private_corrective_canary(
            bundle, revision=args.revision, transport=transport, disposal_receipt=args.disposal_receipt,
        )
    else:
        bundle = build_corrective_canary_abort_publication_bundle(
            abort_receipt_path=args.terminal_receipt, **common,
        )
        result = publish_private_corrective_canary_abort(
            bundle, revision=args.revision, transport=transport, disposal_receipt=args.disposal_receipt,
        )
    return {"release_id": result.release_id, "immutable_revision": result.immutable_revision}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args, transport=HuggingFaceHubTransport())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
