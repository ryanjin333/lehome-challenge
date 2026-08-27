#!/usr/bin/env python3
"""Build strict, atomically-emitted simple-curriculum JSON matrices."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Callable, Mapping

from lehome.flywheel.simple_curriculum import build_calibration_rows, build_curriculum_rows


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError("duplicate JSON key")
        document[key] = value
    return document


def _has_symlink_ancestor(path: Path) -> bool:
    current = path
    while True:
        try:
            if stat.S_ISLNK(current.lstat().st_mode):
                return True
        except OSError:
            return True
        if current.parent == current:
            return False
        current = current.parent


def _reject_nonfinite(value: object) -> None:
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_nonfinite(item)
    elif isinstance(value, list):
        for item in value:
            _reject_nonfinite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON number")


def _strict_json(path: Path, *, label: str) -> object:
    if not path.is_absolute() or _has_symlink_ancestor(path) or not path.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("non-finite JSON number")),
        )
        _reject_nonfinite(value)
        return value
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is malformed") from error


def _safe_absent(path: Path, *, label: str) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink() or _has_symlink_ancestor(path.parent):
        raise FileExistsError(f"{label} must be an absent absolute path")
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir() or not stat.S_ISDIR(parent.stat().st_mode):
        raise ValueError(f"{label} parent is unsafe")


def _atomic_write(path: Path, payload: bytes, *, before_publish: Callable[[], None] | None = None) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fchmod(stream.fileno(), 0o644)
            os.fsync(stream.fileno())
        if before_publish is not None:
            before_publish()
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _emit(*, output: Path, receipt: Path, rows: list[dict[str, object]], parameters: Mapping[str, object]) -> None:
    _safe_absent(output, label="output")
    _safe_absent(receipt, label="receipt")
    if output.resolve(strict=False) == receipt.resolve(strict=False):
        raise ValueError("output and receipt paths must be distinct")
    output_payload = _canonical_bytes(rows)
    receipt_payload = _canonical_bytes({
        "schema_version": 1,
        "kind": "lehome_simple_curriculum_matrix_receipt_v1",
        "parameters": dict(parameters),
        "output_sha256": sha256(output_payload).hexdigest(),
        "output_bytes": len(output_payload),
    })
    _atomic_write(output, output_payload)
    _atomic_write(receipt, receipt_payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    calibration = commands.add_parser("build-calibration")
    calibration.add_argument("--catalog", type=Path, required=True)
    calibration.add_argument("--seed-base", type=int, required=True)
    calibration.add_argument("--output", type=Path, required=True)
    calibration.add_argument("--receipt", type=Path, required=True)
    curriculum = commands.add_parser("build-curriculum")
    curriculum.add_argument("--report", type=Path, required=True)
    curriculum.add_argument("--calibration-matrix", type=Path, required=True)
    curriculum.add_argument("--approved-catalog", type=Path, required=True)
    curriculum.add_argument("--policy-identity", type=Path, required=True)
    curriculum.add_argument("--rng-seed", type=int, required=True)
    curriculum.add_argument("--output", type=Path, required=True)
    curriculum.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build-calibration":
            catalog = _strict_json(args.catalog, label="catalog")
            rows = build_calibration_rows(catalog, seed_base=args.seed_base)
            _emit(output=args.output, receipt=args.receipt, rows=rows, parameters={
                "command": args.command, "catalog": catalog, "seed_base": args.seed_base,
            })
        else:
            report = _strict_json(args.report, label="report")
            calibration = _strict_json(args.calibration_matrix, label="calibration matrix")
            approved_catalog = _strict_json(args.approved_catalog, label="approved catalog")
            policy_identity = _strict_json(args.policy_identity, label="policy identity")
            rows = build_curriculum_rows(
                report, calibration_rows=calibration, count=600, rng_seed=args.rng_seed,
                policy_identity=policy_identity, catalog=approved_catalog,
            )
            _emit(output=args.output, receipt=args.receipt, rows=rows, parameters={
                "command": args.command,
                "report_sha256": sha256(_canonical_bytes(report)).hexdigest(),
                "calibration_matrix_sha256": sha256(_canonical_bytes(calibration)).hexdigest(),
                "approved_catalog": approved_catalog,
                "policy_identity": policy_identity,
                "rng_seed": args.rng_seed,
                "count": 600,
            })
    except (ValueError, FileExistsError) as error:
        _parser().error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
