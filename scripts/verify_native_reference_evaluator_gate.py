#!/usr/bin/env python3
"""Fail-closed evidence contracts for the native GR00T N1.5 reference gate."""

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
SOURCE_TREE_SHA256 = "eada9f80b0dda1428177fe4551efa8059fe85845d4db5b32bb673f88a50c6bb2"
LEROBOT_VERSION = "0.4.3"
POLICY_CLASS = "scripts.eval_policy.lerobot_policy.LeRobotPolicy"
TASK_DESCRIPTION = "fold the garment on the table"
SUCCESS_CHECKER = "pinned_raw_success_distance_second_mesh_points"
_HEX = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_VM = re.compile(r"^computeinstance-[a-z0-9]+$")
_DISK = re.compile(r"^computedisk-[a-z0-9]+$")
_IMAGE = re.compile(r"^[^\s]+@sha256:[0-9a-f]{64}$")
_EPISODE_LINE = re.compile(r"Episode\s+(\d+)/2:.*?Success=(True|False)")
_KNOWN_INVALID = re.compile(r"(?:traceback|non[- ]?finite|cloth[ _-]?flight|missing cloth|safety failure|cuda error)", re.IGNORECASE)


class NativeReferenceGateError(ValueError):
    """Raised when evidence cannot prove a safe native-reference result."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as error:
        raise NativeReferenceGateError("document is not canonical JSON") from error


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _object(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise NativeReferenceGateError(f"{label} must be an object")
    return dict(value)


def _safe_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise NativeReferenceGateError(f"{label} is missing")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.name in {"", "."}:
        raise NativeReferenceGateError(f"{label} is unsafe")
    return path.as_posix()


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise NativeReferenceGateError(f"{label} is not a SHA-256 digest")
    return value


def _artifact(value: object, label: str) -> dict[str, object]:
    artifact = _object(value, label)
    if set(artifact) != {"path", "size", "sha256"}:
        raise NativeReferenceGateError(f"{label} has an unexpected schema")
    if type(artifact["size"]) is not int or artifact["size"] <= 0:
        raise NativeReferenceGateError(f"{label} has an invalid size")
    return {"path": _safe_path(artifact["path"], label), "size": artifact["size"], "sha256": _digest(artifact["sha256"], label)}


def _artifact_from_file(root: Path, relative: Path) -> dict[str, object]:
    path = root / relative
    try:
        metadata = path.lstat()
    except OSError as error:
        raise NativeReferenceGateError(f"missing required artifact: {relative.as_posix()}") from error
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
        raise NativeReferenceGateError(f"required artifact is unsafe: {relative.as_posix()}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": relative.as_posix(), "size": metadata.st_size, "sha256": digest.hexdigest()}


def _verify_artifact(root: Path, value: object, label: str) -> dict[str, object]:
    artifact = _artifact(value, label)
    observed = _artifact_from_file(root, Path(*PurePosixPath(artifact["path"]).parts))
    if observed != artifact:
        raise NativeReferenceGateError(f"{label} changed after receipt creation")
    return artifact


def oracle_attempts() -> tuple[dict[str, object], ...]:
    rows = (("top_long", "Top_Long_Seen_0", (True, True)), ("top_short", "Top_Short_Seen_0", (True, True)), ("pant_long", "Pant_Long_Seen_0", (True, True)), ("pant_short", "Pant_Short_Seen_0", (False, True)))
    return tuple({"attempt_id": f"native-reference-{stage}-{episode}", "stage": stage, "category": category, "garment": garment, "episode": episode, "expected_success": expected[episode - 1]} for stage, (category, garment, expected) in enumerate(rows, start=1) for episode in (1, 2))


def _validate_identity(value: object) -> dict[str, object]:
    identity = _object(value, "identity")
    fixed = {"source_repository": SOURCE_REPOSITORY, "source_revision": SOURCE_REVISION, "source_tree_sha256": SOURCE_TREE_SHA256, "lerobot_version": LEROBOT_VERSION, "policy_class": POLICY_CLASS, "policy_device": "cuda:0", "simulator_device": "cpu", "task_description": TASK_DESCRIPTION, "action_horizon": 16, "action_dimension": 12, "success_checker": SUCCESS_CHECKER, "cuda_available": True}
    variable = {"checkpoint_tree_sha256", "metadata_tree_sha256", "assets_tree_sha256", "cache_trust_manifest_sha256", "cuda_runtime", "cuda_device_count", "vm_id", "disk_id", "image"}
    if set(identity) != {*fixed, *variable}:
        raise NativeReferenceGateError("identity has an unexpected schema")
    for key, expected in fixed.items():
        if identity.get(key) != expected:
            label = "LeRobot" if key == "lerobot_version" else key
            raise NativeReferenceGateError(f"identity {label} does not match the native contract")
    for key in ("checkpoint_tree_sha256", "metadata_tree_sha256", "assets_tree_sha256", "cache_trust_manifest_sha256"):
        _digest(identity.get(key), f"identity {key}")
    if type(identity["cuda_device_count"]) is not int or identity["cuda_device_count"] < 1 or not isinstance(identity["cuda_runtime"], str) or not identity["cuda_runtime"]:
        raise NativeReferenceGateError("identity does not prove CUDA availability")
    if not isinstance(identity["vm_id"], str) or _VM.fullmatch(identity["vm_id"]) is None:
        raise NativeReferenceGateError("identity VM ID is invalid")
    if not isinstance(identity["disk_id"], str) or _DISK.fullmatch(identity["disk_id"]) is None:
        raise NativeReferenceGateError("identity disk ID is invalid")
    if not isinstance(identity["image"], str) or _IMAGE.fullmatch(identity["image"]) is None:
        raise NativeReferenceGateError("identity image is not digest pinned")
    return identity


def _validate_attempt(value: object, expected: Mapping[str, object], root: Path | None) -> dict[str, object]:
    attempt = _object(value, "attempt")
    if set(attempt) != {*expected, "success", "videos", "log", "receipt"}:
        raise NativeReferenceGateError("attempt has an unexpected schema")
    if any(attempt.get(key) != item for key, item in expected.items()) or type(attempt.get("success")) is not bool:
        raise NativeReferenceGateError("attempt sequence does not match the native oracle")
    if type(attempt["videos"]) is not list or not attempt["videos"]:
        raise NativeReferenceGateError("attempt must enumerate at least one video")
    parser = _verify_artifact if root is not None else lambda _root, item, label: _artifact(item, label)
    base = root if root is not None else Path(".")
    videos = [parser(base, item, "attempt video") for item in attempt["videos"]]
    return {**{key: attempt[key] for key in expected}, "success": attempt["success"], "videos": videos, "log": parser(base, attempt["log"], "attempt log"), "receipt": parser(base, attempt["receipt"], "attempt receipt")}


def verify_native_reference_result(document: object, *, bundle_root: Path | None = None) -> dict[str, object]:
    """Assess execution only. A local oracle match never claims final passage."""
    result = _object(document, "native reference result")
    if set(result) != {"schema_version", "kind", "identity", "attempts"} or result.get("schema_version") != 2 or result.get("kind") != "lehome_native_reference_execution_result_v2":
        raise NativeReferenceGateError("native reference result kind is invalid")
    identity = _validate_identity(result.get("identity"))
    rows = result.get("attempts")
    if type(rows) is not list or len(rows) not in {2, 8}:
        raise NativeReferenceGateError("native reference result must contain either the two-attempt admission stop or all eight attempts")
    oracle = oracle_attempts(); attempts = [_validate_attempt(row, oracle[index], bundle_root) for index, row in enumerate(rows)]
    first = [row["success"] for row in attempts[:2]]
    if first != [True, True]:
        if len(attempts) != 2: raise NativeReferenceGateError("failed Top_Long admission must fail fast before later attempts")
        return {"schema_version": 1, "kind": "lehome_native_reference_execution_receipt_v2", "status": "evaluator_compatibility_stop", "reason": "top_long_admission_failed", "attempt_count": 2, "successes": sum(first), "oracle_vector": [row["expected_success"] for row in oracle], "identity": identity, "result_sha256": canonical_sha256(result)}
    if len(attempts) != 8: raise NativeReferenceGateError("passing Top_Long admission requires all eight sequential attempts")
    observed, expected = [row["success"] for row in attempts], [row["expected_success"] for row in oracle]
    if observed != expected:
        return {"schema_version": 1, "kind": "lehome_native_reference_execution_receipt_v2", "status": "evaluator_compatibility_stop", "reason": "oracle_outcome_mismatch", "attempt_count": 8, "successes": sum(observed), "oracle_vector": expected, "observed_vector": observed, "identity": identity, "result_sha256": canonical_sha256(result)}
    return {"schema_version": 1, "kind": "lehome_native_reference_execution_receipt_v2", "status": "oracle_matched_pending_finalization", "attempt_count": 8, "successes": 7, "oracle_vector": expected, "identity": identity, "result_sha256": canonical_sha256(result)}


def _write_exclusive(path: Path, value: Mapping[str, object]) -> None:
    payload = _canonical_bytes(dict(value))
    try: descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as error: raise NativeReferenceGateError("native reference receipt already exists") from error
    with os.fdopen(descriptor, "wb") as stream: stream.write(payload); stream.flush(); os.fsync(stream.fileno())


def _write_result(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(".partial")
    if temporary.exists() or temporary.is_symlink(): raise NativeReferenceGateError("native result temporary path is unsafe")
    temporary.write_bytes(_canonical_bytes(dict(value))); os.replace(temporary, path)


def compile_native_stage(bundle_root: Path, *, stage: int, category: str, garment: str, identity: Mapping[str, object]) -> dict[str, object]:
    """Compile a real public evaluator log and exact files into append-only evidence."""
    root = Path(bundle_root).resolve(strict=True); log_relative = Path("logs") / f"stage-{stage}.log"; log = _artifact_from_file(root, log_relative)
    text = (root / log_relative).read_text(encoding="utf-8", errors="strict")
    if _KNOWN_INVALID.search(text): raise NativeReferenceGateError("native evaluator log contains an infrastructure or fidelity failure")
    matches = _EPISODE_LINE.findall(text)
    if len(matches) != 2 or [item[0] for item in matches] != ["1", "2"]: raise NativeReferenceGateError("native evaluator log does not contain exactly two ordered episode outcomes")
    if stage not in {1, 2, 3, 4}: raise NativeReferenceGateError("native stage is invalid")
    expected_rows = oracle_attempts()[(stage - 1) * 2:stage * 2]
    if any(row["category"] != category or row["garment"] != garment for row in expected_rows): raise NativeReferenceGateError("native stage descriptor does not match the oracle")
    result_path = root / "result.json"
    if result_path.exists():
        metadata = result_path.lstat()
        if result_path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise NativeReferenceGateError("native stage result is unsafe")
        result = _object(json.loads(result_path.read_text(encoding="utf-8")), "native reference result")
    else:
        result = {"schema_version": 2, "kind": "lehome_native_reference_execution_result_v2", "identity": _validate_identity(identity), "attempts": []}
    existing = result.get("attempts")
    if type(existing) is not list or len(existing) != (stage - 1) * 2 or result.get("identity") != _validate_identity(identity):
        raise NativeReferenceGateError("native stage result is not an append-only oracle prefix")
    receipts = root / "receipts"; receipts.mkdir(mode=0o700, exist_ok=True)
    for row, (_, outcome) in zip(expected_rows, matches, strict=True):
        candidates = sorted((root / "videos" / f"stage-{stage}").glob(f"**/episode{int(row['episode']) - 1}_*.mp4"))
        if not candidates: raise NativeReferenceGateError(f"native evaluator is missing videos for {row['attempt_id']}")
        videos = [_artifact_from_file(root, path.relative_to(root)) for path in candidates]; receipt_relative = Path("receipts") / f"{row['attempt_id']}.json"; receipt_path = root / receipt_relative
        body = {"schema_version": 1, "kind": "lehome_native_reference_attempt_receipt_v1", "attempt_id": row["attempt_id"], "stage": stage, "category": category, "garment": garment, "episode": row["episode"], "success": outcome == "True", "log": log, "videos": videos}
        _write_exclusive(receipt_path, body); result["attempts"].append({**row, "success": outcome == "True", "videos": videos, "log": log, "receipt": _artifact_from_file(root, receipt_relative)})
    _write_result(result_path, result); return result


def finalize_native_reference_gate(execution: object, fidelity: object, publication: object, stopped: object) -> dict[str, object]:
    execution = _object(execution, "execution receipt")
    if execution.get("kind") != "lehome_native_reference_execution_receipt_v2" or execution.get("status") != "oracle_matched_pending_finalization": raise NativeReferenceGateError("execution receipt is not an oracle-matched pending result")
    identity = _validate_identity(execution.get("identity")); execution_sha = canonical_sha256(execution); fidelity = _object(fidelity, "fidelity review")
    if set(fidelity) != {"schema_version", "kind", "execution_receipt_sha256", "review_method", "attempts"} or fidelity.get("schema_version") != 1 or fidelity.get("kind") != "lehome_native_reference_fidelity_review_v1" or fidelity.get("execution_receipt_sha256") != execution_sha or fidelity.get("review_method") != "manual_video_audit": raise NativeReferenceGateError("fidelity review is not bound to the execution receipt")
    reviewed, ids = fidelity.get("attempts"), [row["attempt_id"] for row in oracle_attempts()]
    if type(reviewed) is not list or [row.get("attempt_id") if isinstance(row, dict) else None for row in reviewed] != ids: raise NativeReferenceGateError("fidelity review does not cover every attempt")
    for row in reviewed:
        if set(row) != {"attempt_id", "cloth_present", "cloth_flight", "nonfinite", "safety_failure", "evidence_sha256"} or row.get("cloth_present") is not True or any(row.get(key) is not False for key in ("cloth_flight", "nonfinite", "safety_failure")): raise NativeReferenceGateError("fidelity review contains an invalid outcome")
        _digest(row.get("evidence_sha256"), "fidelity review evidence")
    publication = _object(publication, "publication receipt")
    if set(publication) != {"schema_version", "kind", "execution_receipt_sha256", "immutable_revision", "bundle_manifest_sha256", "readback_verified"} or publication.get("schema_version") != 1 or publication.get("kind") != "lehome_native_reference_hf_readback_v1" or publication.get("execution_receipt_sha256") != execution_sha or publication.get("readback_verified") is not True or not isinstance(publication.get("immutable_revision"), str) or _COMMIT.fullmatch(publication["immutable_revision"]) is None: raise NativeReferenceGateError("publication receipt is not a readback-verified immutable upload")
    _digest(publication.get("bundle_manifest_sha256"), "publication bundle manifest")
    stopped = _object(stopped, "stopped VM receipt")
    if set(stopped) != {"schema_version", "kind", "execution_receipt_sha256", "vm_id", "disk_id", "image", "state", "attached_disk_ids"} or stopped.get("schema_version") != 1 or stopped.get("kind") != "lehome_native_reference_vm_stopped_v1" or stopped.get("execution_receipt_sha256") != execution_sha or stopped.get("vm_id") != identity["vm_id"] or stopped.get("disk_id") != identity["disk_id"] or stopped.get("image") != identity["image"] or stopped.get("state") != "STOPPED" or stopped.get("attached_disk_ids") != [identity["disk_id"]]: raise NativeReferenceGateError("stopped VM receipt is not bound to the execution identity")
    return {"schema_version": 1, "kind": "lehome_native_reference_gate_final_receipt_v1", "status": "passed", "execution_receipt_sha256": execution_sha, "fidelity_review_sha256": canonical_sha256(fidelity), "publication_receipt_sha256": canonical_sha256(publication), "stopped_vm_receipt_sha256": canonical_sha256(stopped), "identity": identity}


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        if path.is_symlink() or not path.is_file(): raise OSError("unsafe")
        return _object(json.loads(path.read_bytes()), label)
    except (OSError, UnicodeError, json.JSONDecodeError, NativeReferenceGateError) as error: raise NativeReferenceGateError(f"{label} is invalid") from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify-execution"); verify.add_argument("--result", type=Path, required=True); verify.add_argument("--bundle-root", type=Path, required=True); verify.add_argument("--receipt", type=Path, required=True)
    compile_stage = commands.add_parser("compile-stage"); compile_stage.add_argument("--bundle-root", type=Path, required=True); compile_stage.add_argument("--stage", type=int, required=True); compile_stage.add_argument("--category", required=True); compile_stage.add_argument("--garment", required=True); compile_stage.add_argument("--identity", type=Path, required=True)
    finalize = commands.add_parser("finalize"); finalize.add_argument("--execution", type=Path, required=True); finalize.add_argument("--fidelity", type=Path, required=True); finalize.add_argument("--publication", type=Path, required=True); finalize.add_argument("--stopped", type=Path, required=True); finalize.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "compile-stage": receipt = compile_native_stage(args.bundle_root, stage=args.stage, category=args.category, garment=args.garment, identity=_read_json(args.identity, "identity"))
        elif args.command == "verify-execution": receipt = verify_native_reference_result(_read_json(args.result, "result"), bundle_root=args.bundle_root); _write_exclusive(args.receipt, receipt)
        else: receipt = finalize_native_reference_gate(_read_json(args.execution, "execution receipt"), _read_json(args.fidelity, "fidelity review"), _read_json(args.publication, "publication receipt"), _read_json(args.stopped, "stopped VM receipt")); _write_exclusive(args.receipt, receipt)
    except NativeReferenceGateError as error:
        if argv is not None: raise SystemExit(str(error)) from None
        print(f"error: {error}", file=sys.stderr); return 2
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":"))); return 0


if __name__ == "__main__": raise SystemExit(main())
