#!/usr/bin/env python3
"""Verify and render the pinned public GR00T N1.5 recipe without executing it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence


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
    lifecycle = commands.add_parser(
        "lifecycle-plan",
        help="write an immutable, pre-paid N1.5 lifecycle plan",
    )
    for item in (lifecycle, commands.add_parser("verify-lifecycle-plan", help="verify a prior immutable lifecycle plan")):
        item.add_argument("--run-id", required=True)
        item.add_argument("--repository", required=True)
        item.add_argument("--budget-usd", type=float, required=True)
        item.add_argument("--estimated-cost-usd", type=float, required=True)
        item.add_argument("--output", type=Path, required=True)
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
        "source_tree": verified.source_tree,
        "resolved_snapshots_receipt": str(verified.resolved_snapshots_receipt),
        "resolved_snapshots_receipt_sha256": verified.resolved_snapshots_receipt_sha256,
        "base_model_metadata_sha256": verified.base_model_metadata_sha256,
        "dataset_metadata_sha256": verified.dataset_metadata_sha256,
        "vm_id": contract.vm_id,
        "disk_id": contract.disk_id,
    }


def _lifecycle_plan(args: argparse.Namespace, contract: ReproductionContract) -> dict[str, object]:
    """Return a pure, immutable admission record before the host may start a VM."""
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", args.run_id) is None:
        raise ReproductionError("lifecycle run id is invalid")
    if not (args.budget_usd > 0.0 and args.budget_usd <= 100.0):
        raise ReproductionError("budget must be positive and cannot exceed the $100 hard cap")
    if not (args.estimated_cost_usd >= 0.0 and args.estimated_cost_usd <= args.budget_usd):
        raise ReproductionError("estimated lifecycle cost exceeds the approved budget")
    if re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", args.repository) is None:
        raise ReproductionError("public lifecycle repository is invalid")
    return {
        "schema_version": 1,
        "kind": "lehome_public_n15_lifecycle_plan_v1",
        "run_id": args.run_id,
        "vm_id": contract.vm_id,
        "protected_disk_id": contract.disk_id,
        "provider_source_image_id": "computeimage-u00zf6w3yf72gakhcy",
        "repository": args.repository,
        "prefixes": {
            "training": f"n15-public/{args.run_id}/training",
            "focused": f"n15-public/{args.run_id}/focused",
            "harvest": f"n15-public/{args.run_id}/harvest",
        },
        "budget_usd": args.budget_usd,
        "estimated_cost_usd": args.estimated_cost_usd,
        "stages": [
            "verify_stopped",
            "start",
            "validate_runtime",
            "train",
            "train_publish_readback",
            "focused_gate",
            "focused_gate_publish_readback",
            "harvest",
            "harvest_publish_readback",
            "stop",
        ],
        "downstream_forbidden_on_failure": True,
    }


def _verify_lifecycle_plan(args: argparse.Namespace, contract: ReproductionContract) -> dict[str, object]:
    expected = _lifecycle_plan(args, contract)
    try:
        payload = args.output.read_bytes()
        actual = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReproductionError("lifecycle plan is unreadable") from error
    canonical = (
        json.dumps(actual, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii") if isinstance(actual, dict) else b""
    if not isinstance(actual, dict) or payload != canonical or actual != expected:
        raise ReproductionError("lifecycle plan is not the exact immutable approved plan")
    return actual


def main(
    argv: Sequence[str] | None = None,
    *,
    contract: ReproductionContract = CONTRACT,
) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "lifecycle-plan":
            value = _lifecycle_plan(args, contract)
            stored = write_receipt(
                output=args.output,
                value=value,
                label="public N1.5 lifecycle plan",
            )
            result = {**value, **stored}
        elif args.command == "verify-lifecycle-plan":
            result = _verify_lifecycle_plan(args, contract)
        else:
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
