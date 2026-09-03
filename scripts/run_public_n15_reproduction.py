#!/usr/bin/env python3
"""Verify and render the pinned public GR00T N1.5 recipe without executing it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import sys
import time
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "source/lehome"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from lehome.n15_reproduction import (  # noqa: E402
    CONTRACT,
    ReproductionContract,
    ReproductionError,
    build_compatible_lerobot_wheel,
    cleanup_resume_scratch,
    compatibility_wheel_identity,
    finalize_training_output,
    render_training,
    prepare_resume_scratch,
    verify_resume_checkpoint,
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
    finalization = commands.add_parser(
        "finalize-training-output",
        help="recoverably assemble and atomically publish a verified training output",
    )
    for name in (
        "checkout", "source-receipt", "resolved-snapshots-receipt",
        "vm-id", "disk-id", "training-root", "staging-root", "upstream-output",
    ):
        finalization.add_argument(
            f"--{name}",
            type=Path if name not in {"vm-id", "disk-id"} else str,
            required=True,
        )
    resume = commands.add_parser(
        "verify-resume-checkpoint",
        help="authenticate one explicit partial native LeRobot checkpoint",
    )
    _add_inputs(resume)
    resume.add_argument("--training-root", type=Path, required=True)
    resume.add_argument("--staging-root", type=Path, required=True)
    resume.add_argument("--upstream-output", type=Path, required=True)
    resume.add_argument("--resume-step", type=int, required=True)
    resume.add_argument("--attempt-id", required=True)
    for scratch_command, help_text in (
        ("prepare-resume-scratch", "replace safe stale scratch with one owned attempt tree"),
        ("cleanup-resume-scratch", "remove one owned resume attempt scratch tree"),
    ):
        scratch = commands.add_parser(scratch_command, help=help_text)
        scratch.add_argument("--staging-root", type=Path, required=True)
        scratch.add_argument("--attempt-id", required=True)
    compatibility = commands.add_parser(
        "build-compatible-wheel",
        help="build the sealed two-field LeRobot 0.4.3 compatibility wheel",
    )
    compatibility.add_argument("--upstream-wheel", type=Path, required=True)
    compatibility.add_argument("--wheel-output", type=Path, required=True)
    compatibility.add_argument("--receipt-output", type=Path, required=True)
    compatibility_verify = commands.add_parser(
        "verify-compatible-wheel", help="verify a sealed LeRobot compatibility wheel"
    )
    compatibility_verify.add_argument("--upstream-wheel", type=Path, required=True)
    compatibility_verify.add_argument("--wheel", type=Path, required=True)
    compatibility_verify.add_argument("--receipt", type=Path, required=True)
    lifecycle = commands.add_parser(
        "lifecycle-plan",
        help="write an immutable, pre-paid N1.5 lifecycle plan",
    )
    for item in (lifecycle, commands.add_parser("verify-lifecycle-plan", help="verify a prior immutable lifecycle plan")):
        item.add_argument("--run-id", required=True)
        item.add_argument("--repository", required=True)
        item.add_argument("--remote-pipeline-root", required=True)
        item.add_argument("--budget-usd", type=float, required=True)
        item.add_argument("--estimated-cost-usd", type=float, required=True)
        item.add_argument("--output", type=Path, required=True)
    rate = commands.add_parser(
        "provider-rate-admission",
        help="validate immutable exact-VM rate evidence against the paid task window",
    )
    rate.add_argument("--rate-receipt", type=Path, required=True)
    rate.add_argument("--lifecycle-plan", type=Path, required=True)
    rate.add_argument("--paid-deadline", type=Path, required=True)
    rate.add_argument("--budget-usd", type=float, required=True)
    rate.add_argument("--hourly-ceiling-usd", type=float, required=True)
    rate.add_argument("--output", type=Path, required=True)
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
    if not str(args.remote_pipeline_root).startswith("/") or ".." in str(args.remote_pipeline_root).split("/"):
        raise ReproductionError("remote lifecycle pipeline root is invalid")
    return {
        "schema_version": 1,
        "kind": "lehome_public_n15_lifecycle_plan_v1",
        "run_id": args.run_id,
        "vm_id": contract.vm_id,
        "protected_disk_id": contract.disk_id,
        "provider_source_image_id": "computeimage-u00zf6w3yf72gakhcy",
        "repository": args.repository,
        "remote_pipeline_root": args.remote_pipeline_root,
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


def _immutable_canonical_json(path: Path, label: str) -> tuple[dict[str, object], bytes]:
    try:
        metadata = path.lstat()
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReproductionError(f"{label} is unreadable") from error
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o444
        or not isinstance(value, dict)
    ):
        raise ReproductionError(f"{label} is not an immutable regular file")
    try:
        canonical = (
            json.dumps(
                value, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True, allow_nan=False,
            ) + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ReproductionError(f"{label} is not canonical strict JSON") from error
    if payload != canonical:
        raise ReproductionError(f"{label} is not canonical strict JSON")
    return value, payload


def _provider_rate_admission(
    args: argparse.Namespace, contract: ReproductionContract,
) -> dict[str, object]:
    """Validate staged exact-VM price evidence and the task-local paid window."""

    rate, rate_bytes = _immutable_canonical_json(args.rate_receipt, "provider rate receipt")
    plan, plan_bytes = _immutable_canonical_json(args.lifecycle_plan, "lifecycle plan")
    deadline, deadline_bytes = _immutable_canonical_json(args.paid_deadline, "paid deadline")
    expected_rate_keys = {
        "schema_version", "kind", "vm_id", "currency", "hourly_rate_usd",
        "source", "observed_unix_seconds", "valid_until_unix_seconds",
    }
    hourly_rate = rate.get("hourly_rate_usd")
    observed = rate.get("observed_unix_seconds")
    valid_until = rate.get("valid_until_unix_seconds")
    if (
        set(rate) != expected_rate_keys
        or (
            rate.get("schema_version"), rate.get("kind"), rate.get("vm_id"),
            rate.get("currency"), rate.get("source"),
        ) != (
            1, "lehome_public_n15_exact_vm_rate_v1", contract.vm_id, "USD",
            "operator-staged-nebius-exact-vm-price",
        )
        or isinstance(hourly_rate, bool)
        or not isinstance(hourly_rate, (int, float))
        or not (hourly_rate > 0.0)
        or type(observed) is not int
        or type(valid_until) is not int
        or not (0 < valid_until - observed <= 86400)
    ):
        raise ReproductionError("provider rate receipt is not exact pinned VM price evidence")
    now = int(time.time())
    if observed > now or now > valid_until:
        raise ReproductionError("provider rate receipt is expired or not yet valid")
    if not (args.hourly_ceiling_usd > 0.0 and hourly_rate <= args.hourly_ceiling_usd):
        raise ReproductionError("provider rate exceeds the code-owned hourly ceiling")
    if not (args.budget_usd > 0.0 and args.budget_usd <= 100.0):
        raise ReproductionError("task budget is invalid")
    started = deadline.get("started_unix_seconds")
    ends = deadline.get("deadline_unix_seconds")
    if (
        plan.get("vm_id") != contract.vm_id
        or plan.get("budget_usd") != args.budget_usd
        or deadline.get("kind") != "lehome_public_n15_paid_deadline_v1"
        or deadline.get("run_id") != plan.get("run_id")
        or deadline.get("lifecycle_plan_sha256") != hashlib.sha256(plan_bytes).hexdigest()
        or type(started) is not int
        or type(ends) is not int
        or ends != started + 86400
        or not (started <= now < ends)
    ):
        raise ReproductionError("paid task window is invalid for provider rate admission")
    paid_window_seconds = ends - started
    elapsed_paid_seconds = now - started
    maximum_cost = hourly_rate * paid_window_seconds / 3600.0
    elapsed_cost = hourly_rate * elapsed_paid_seconds / 3600.0
    if maximum_cost > args.budget_usd or elapsed_cost > args.budget_usd:
        raise ReproductionError("exact-VM task cost exceeds the approved budget")
    base = {
        "schema_version": 1,
        "kind": "lehome_public_n15_provider_rate_admission_v1",
        "vm_id": contract.vm_id,
        "rate_receipt_sha256": hashlib.sha256(rate_bytes).hexdigest(),
        "lifecycle_plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "paid_deadline_sha256": hashlib.sha256(deadline_bytes).hexdigest(),
        "hourly_rate_usd": hourly_rate,
        "hourly_ceiling_usd": args.hourly_ceiling_usd,
        "budget_usd": args.budget_usd,
        "paid_window_seconds": paid_window_seconds,
        "maximum_paid_window_cost_usd": maximum_cost,
    }
    if args.output.exists() or args.output.is_symlink():
        actual, _ = _immutable_canonical_json(args.output, "provider rate admission")
        admitted_at = actual.get("admitted_at_unix_seconds")
        first_elapsed = actual.get("elapsed_paid_seconds_at_first_admission")
        if (
            {key: actual.get(key) for key in base} != base
            or set(actual) != {
                *base, "admitted_at_unix_seconds",
                "elapsed_paid_seconds_at_first_admission",
            }
            or type(admitted_at) is not int
            or type(first_elapsed) is not int
            or admitted_at - started != first_elapsed
            or not (observed <= admitted_at <= now)
        ):
            raise ReproductionError(
                "provider rate admission is not the exact immutable task admission"
            )
        return actual
    return {
        **base,
        "admitted_at_unix_seconds": now,
        "elapsed_paid_seconds_at_first_admission": elapsed_paid_seconds,
    }


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
        elif args.command == "provider-rate-admission":
            value = _provider_rate_admission(args, contract)
            if args.output.exists():
                result = value
            else:
                stored = write_receipt(
                    output=args.output,
                    value=value,
                    label="provider rate admission",
                )
                result = {**value, **stored}
        elif args.command == "build-compatible-wheel":
            result = build_compatible_lerobot_wheel(
                upstream_wheel=args.upstream_wheel,
                output_wheel=args.wheel_output,
                receipt_output=args.receipt_output,
                expected_upstream_sha256=contract.lerobot_wheel_sha256,
            )
        elif args.command == "verify-compatible-wheel":
            result = compatibility_wheel_identity(
                wheel=args.wheel,
                receipt=args.receipt,
                upstream_wheel=args.upstream_wheel,
                expected_upstream_sha256=contract.lerobot_wheel_sha256,
            )
        elif args.command == "prepare-resume-scratch":
            path = prepare_resume_scratch(
                staging_root=args.staging_root, attempt_id=args.attempt_id
            )
            result = {"scratch_root": str(path)}
        elif args.command == "cleanup-resume-scratch":
            cleanup_resume_scratch(
                staging_root=args.staging_root, attempt_id=args.attempt_id
            )
            result = {"removed": True}
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
            elif args.command == "finalize-training-output":
                result = finalize_training_output(
                    verified=verified,
                    training_root=args.training_root,
                    staging_root=args.staging_root,
                    upstream_output=args.upstream_output,
                    contract=contract,
                )
            elif args.command == "verify-resume-checkpoint":
                value = verify_resume_checkpoint(
                    verified=verified,
                    training_root=args.training_root,
                    staging_root=args.staging_root,
                    upstream_output=args.upstream_output,
                    requested_step=args.resume_step,
                    attempt_id=args.attempt_id,
                    contract=contract,
                )
                stored = write_receipt(
                    output=args.output,
                    value=value,
                    label="public N1.5 resume lineage receipt",
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
