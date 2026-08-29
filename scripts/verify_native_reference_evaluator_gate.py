#!/usr/bin/env python3
"""Fail-closed verifier for the isolated GR00T N1.5 reference oracle.

This module deliberately validates a small, self-contained result document.  It
does not import Isaac, LeRobot, or any local N1.7 policy code, so it is safe to
run before and after a paid native-evaluator invocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Mapping, Sequence


SOURCE_REPOSITORY = "theo-zhou/lehome-groot-submission-4"
SOURCE_REVISION = "d384fe00508acd96ab1c3c5dc265e08261f94b3b"
LEROBOT_VERSION = "0.4.3"
POLICY_CLASS = "scripts.eval_policy.lerobot_policy.LeRobotPolicy"
TASK_DESCRIPTION = "fold the garment on the table"
SUCCESS_CHECKER = "pinned_raw_success_distance_second_mesh_points"
SOURCE_TREE_SHA256 = "eada9f80b0dda1428177fe4551efa8059fe85845d4db5b32bb673f88a50c6bb2"
_HEX = re.compile(r"^[0-9a-f]{64}$")


class NativeReferenceGateError(ValueError):
    """Raised when native-reference evidence is incomplete or incompatible."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as error:
        raise NativeReferenceGateError("result is not canonical JSON") from error


def _strict_object(payload: object, label: str) -> dict[str, object]:
    if type(payload) is not dict:
        raise NativeReferenceGateError(f"{label} must be an object")
    return dict(payload)


def _safe_artifact_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise NativeReferenceGateError(f"{label} is missing")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.name in {"", "."}:
        raise NativeReferenceGateError(f"{label} is unsafe")
    return value


def oracle_attempts() -> tuple[dict[str, object], ...]:
    """Return the immutable published compatibility oracle in execution order."""
    rows = (
        ("top_long", "Top_Long_Seen_0", True),
        ("top_short", "Top_Short_Seen_0", True),
        ("pant_long", "Pant_Long_Seen_0", True),
        ("pant_short", "Pant_Short_Seen_0", False),
    )
    attempts: list[dict[str, object]] = []
    for stage, (category, garment, expected_success) in enumerate(rows, start=1):
        for episode in (1, 2):
            attempts.append({
                "attempt_id": f"native-reference-{stage}-{episode}",
                "stage": stage,
                "category": category,
                "garment": garment,
                "episode": episode,
                "expected_success": expected_success if episode == 1 else (True if category == "pant_short" else expected_success),
            })
    return tuple(attempts)


def _validate_identity(value: object) -> dict[str, object]:
    identity = _strict_object(value, "identity")
    expected = {
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "lerobot_version": LEROBOT_VERSION,
        "policy_class": POLICY_CLASS,
        "policy_device": "cuda:0",
        "simulator_device": "cpu",
        "task_description": TASK_DESCRIPTION,
        "action_horizon": 16,
        "action_dimension": 12,
        "success_checker": SUCCESS_CHECKER,
    }
    allowed = {*expected, "source_tree_sha256", "checkpoint_tree_sha256", "metadata_tree_sha256", "assets_tree_sha256"}
    if set(identity) != allowed:
        raise NativeReferenceGateError("identity has an unexpected schema")
    for key, expected_value in expected.items():
        if identity.get(key) != expected_value:
            label = "LeRobot" if key == "lerobot_version" else key
            raise NativeReferenceGateError(f"identity {label} does not match the native contract")
    if identity.get("source_tree_sha256") != SOURCE_TREE_SHA256:
        raise NativeReferenceGateError("identity source tree digest does not match the pinned native contract")
    for key in ("source_tree_sha256", "checkpoint_tree_sha256", "metadata_tree_sha256", "assets_tree_sha256"):
        if not isinstance(identity.get(key), str) or _HEX.fullmatch(str(identity[key])) is None:
            raise NativeReferenceGateError(f"identity {key} is not a SHA-256 digest")
    return identity


def _validate_attempt(row: object, expected: Mapping[str, object]) -> dict[str, object]:
    attempt = _strict_object(row, "attempt")
    if set(attempt) != {*expected, "success", "invalid_reason", "video", "log", "receipt"}:
        raise NativeReferenceGateError("attempt has an unexpected schema")
    for key, expected_value in expected.items():
        if attempt.get(key) != expected_value:
            raise NativeReferenceGateError("attempt sequence does not match the native oracle")
    if type(attempt.get("success")) is not bool:
        raise NativeReferenceGateError("attempt success must be boolean")
    if attempt.get("invalid_reason") is not None:
        raise NativeReferenceGateError(f"invalid outcome: {attempt['invalid_reason']}")
    _safe_artifact_path(attempt["video"], "attempt video")
    _safe_artifact_path(attempt["log"], "attempt log")
    _safe_artifact_path(attempt["receipt"], "attempt receipt")
    return attempt


def verify_native_reference_result(document: object) -> dict[str, object]:
    """Validate native results and return an immutable pass/typed-stop receipt."""
    result = _strict_object(document, "native reference result")
    if set(result) != {"schema_version", "kind", "identity", "attempts"}:
        raise NativeReferenceGateError("native reference result has an unexpected schema")
    if result.get("schema_version") != 1 or result.get("kind") != "lehome_native_reference_evaluator_result_v1":
        raise NativeReferenceGateError("native reference result kind is invalid")
    identity = _validate_identity(result.get("identity"))
    rows = result.get("attempts")
    if type(rows) is not list or len(rows) not in {2, 8}:
        raise NativeReferenceGateError("native reference result must contain either the two-attempt admission stop or all eight attempts")
    oracle = oracle_attempts()
    attempts = [_validate_attempt(row, oracle[index]) for index, row in enumerate(rows)]
    first_stage = [bool(row["success"]) for row in attempts[:2]]
    if first_stage != [True, True]:
        if len(attempts) != 2:
            raise NativeReferenceGateError("failed Top_Long admission must fail fast before later attempts")
        return {
            "schema_version": 1,
            "kind": "lehome_native_reference_evaluator_gate_receipt_v1",
            "status": "evaluator_compatibility_stop",
            "reason": "top_long_admission_failed",
            "attempt_count": 2,
            "successes": sum(first_stage),
            "oracle_vector": [row["expected_success"] for row in oracle],
            "identity": identity,
            "result_sha256": hashlib.sha256(_canonical_bytes(result)).hexdigest(),
        }
    if len(attempts) != 8:
        raise NativeReferenceGateError("passing Top_Long admission requires all eight sequential attempts")
    observed = [bool(row["success"]) for row in attempts]
    expected = [bool(row["expected_success"]) for row in oracle]
    if observed != expected:
        return {
            "schema_version": 1,
            "kind": "lehome_native_reference_evaluator_gate_receipt_v1",
            "status": "evaluator_compatibility_stop",
            "reason": "oracle_outcome_mismatch",
            "attempt_count": 8,
            "successes": sum(observed),
            "oracle_vector": expected,
            "observed_vector": observed,
            "identity": identity,
            "result_sha256": hashlib.sha256(_canonical_bytes(result)).hexdigest(),
        }
    return {
        "schema_version": 1,
        "kind": "lehome_native_reference_evaluator_gate_receipt_v1",
        "status": "passed",
        "attempt_count": 8,
        "successes": 7,
        "oracle_vector": expected,
        "identity": identity,
        "result_sha256": hashlib.sha256(_canonical_bytes(result)).hexdigest(),
    }


def _read_regular_json(path: Path) -> dict[str, object]:
    try:
        metadata = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise OSError("unsafe")
        with path.open("rb") as handle:
            return _strict_object(json.loads(handle.read()), "native reference result")
    except (OSError, UnicodeError, json.JSONDecodeError, NativeReferenceGateError) as error:
        raise NativeReferenceGateError("native reference result file is invalid") from error


def _write_exclusive(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(dict(value))
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as error:
        raise NativeReferenceGateError("native reference receipt already exists") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        receipt = verify_native_reference_result(_read_regular_json(args.result))
        _write_exclusive(args.receipt, receipt)
    except NativeReferenceGateError as error:
        if argv is not None:
            raise SystemExit(str(error)) from None
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
