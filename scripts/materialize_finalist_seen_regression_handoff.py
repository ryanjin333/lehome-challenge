#!/usr/bin/env python3
"""Materialize one immutable seen-regression receipt for one admitted finalist."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping


_EVIDENCE_FIELDS = {
    "schema_version", "kind", "candidate_checkpoint_receipt_sha256",
    "major_seen_regression", "readback_verified", "sealed", "report_sha256",
}


def _digest(value: object, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be SHA-256")
    return value


def _canonical(value: Mapping[str, object]) -> bytes:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _load_evidence(path: Path, receipt: str) -> dict[str, object]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file() or (path.stat().st_mode & 0o777) != 0o444:
        raise ValueError("seen-regression evidence source is unsafe")
    try:
        evidence = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("seen-regression evidence is invalid") from error
    if (
        not isinstance(evidence, dict)
        or set(evidence) != _EVIDENCE_FIELDS
        or evidence.get("schema_version") != 1
        or evidence.get("kind") != "lehome_experiment_seen_regression_evidence"
        or evidence.get("candidate_checkpoint_receipt_sha256") != receipt
        or type(evidence.get("major_seen_regression")) is not bool
        or evidence.get("readback_verified") is not True
        or evidence.get("sealed") is not True
    ):
        raise ValueError("seen-regression evidence does not bind checkpoint")
    body = dict(evidence)
    report_sha256 = _digest(body.pop("report_sha256"), "seen-regression report")
    if hashlib.sha256(_canonical(body)).hexdigest() != report_sha256:
        raise ValueError("seen-regression evidence report digest mismatch")
    return evidence


def materialize_handoff(
    *, root: Path, experiment_id: str,
    checkpoint_receipt_sha256: str, evidence_path: Path,
) -> Path:
    experiment = _digest(experiment_id, "experiment ID")
    receipt = _digest(checkpoint_receipt_sha256, "checkpoint receipt")
    if not root.is_absolute() or root.is_symlink():
        raise ValueError("seen-regression handoff root is unsafe")
    evidence = _load_evidence(evidence_path, receipt)
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("seen-regression handoff root is unsafe")
    finalist_root = root / experiment
    if finalist_root.exists() or finalist_root.is_symlink():
        if finalist_root.is_symlink() or not finalist_root.is_dir():
            raise ValueError("finalist handoff directory is unsafe")
    else:
        finalist_root.mkdir(mode=0o750)
    target = finalist_root / f"{receipt}.json"
    if target.exists() or target.is_symlink():
        raise ValueError("finalist seen-regression handoff already exists")
    evidence_bytes = _canonical(evidence)
    descriptor = {
        "schema_version": 1,
        "kind": "lehome_finalist_seen_regression_handoff",
        "experiment_id": experiment,
        "checkpoint_receipt_sha256": receipt,
        "evidence": evidence,
        "evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
    }
    payload = _canonical(descriptor)
    descriptor_fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor_fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    directory_fd = os.open(finalist_root, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--checkpoint-receipt-sha256", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    arguments = parser.parse_args(argv)
    print(materialize_handoff(
        root=arguments.root,
        experiment_id=arguments.experiment_id,
        checkpoint_receipt_sha256=arguments.checkpoint_receipt_sha256,
        evidence_path=arguments.evidence,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
