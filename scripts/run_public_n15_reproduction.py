#!/usr/bin/env python3
"""Verify and render the pinned public GR00T N1.5 recipe without executing it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "source/lehome"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from lehome.n15_reproduction import (  # noqa: E402
    CONTRACT,
    ReproductionContract,
    ReproductionError,
    render_training,
    verify_inputs,
    verify_training_output,
    write_receipt,
)


def _add_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--resolved-snapshots-receipt", type=Path, required=True)
    parser.add_argument("--vm-id", required=True)
    parser.add_argument("--disk-id", required=True)
    parser.add_argument("--output", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify-inputs", help="verify pinned source and snapshots")
    _add_inputs(verify)
    render = commands.add_parser("render-training", help="render the exact upstream training argv")
    _add_inputs(render)
    output = commands.add_parser(
        "verify-training-output",
        help="verify a complete step-12,000 training output",
    )
    _add_inputs(output)
    output.add_argument("--training-root", type=Path, required=True)
    return parser


def _verified(args: argparse.Namespace, contract: ReproductionContract):
    return verify_inputs(
        checkout=args.checkout,
        source_receipt=args.source_receipt,
        resolved_snapshots_receipt=args.resolved_snapshots_receipt,
        vm_id=args.vm_id,
        disk_id=args.disk_id,
        contract=contract,
    )


def _verified_inputs_receipt(verified, contract: ReproductionContract) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "lehome_public_n15_verified_inputs_v1",
        "checkout": str(verified.checkout),
        "base_model_root": str(verified.base_model_root),
        "hub_cache_root": str(verified.hub_cache_root),
        "dataset_root": str(verified.dataset_root),
        "source_receipt": str(verified.source_receipt),
        "source_receipt_sha256": verified.source_receipt_sha256,
        "resolved_snapshots_receipt": str(verified.resolved_snapshots_receipt),
        "resolved_snapshots_receipt_sha256": verified.resolved_snapshots_receipt_sha256,
        "vm_id": contract.vm_id,
        "disk_id": contract.disk_id,
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    contract: ReproductionContract = CONTRACT,
) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        verified = _verified(args, contract)
        if args.command == "verify-inputs":
            value = _verified_inputs_receipt(verified, contract)
            stored = write_receipt(
                output=args.output,
                value=value,
                label="verified inputs receipt",
            )
            result = {**value, **stored}
        elif args.command == "render-training":
            result = render_training(
                verified=verified,
                output=args.output,
                contract=contract,
            )
        elif args.command == "verify-training-output":
            value = verify_training_output(
                verified=verified,
                training_root=args.training_root,
                contract=contract,
            )
            stored = write_receipt(
                output=args.output,
                value=value,
                label="verified training output receipt",
            )
            result = {**value, **stored}
        else:  # pragma: no cover - argparse constrains this branch.
            parser.error("unsupported command")
    except (ReproductionError, OSError, ValueError) as error:
        print(f"public N1.5 reproduction gate failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
