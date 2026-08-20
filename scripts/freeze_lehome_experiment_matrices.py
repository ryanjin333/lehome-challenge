#!/usr/bin/env python3
"""Freeze canonical list-form unseen-20 and unseen-80 evaluator matrices."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Mapping, Sequence


_SHA256 = re.compile(r"[0-9a-f]{64}")
_CATEGORIES = ("top_long", "top_short", "pant_long", "pant_short")
_ROW_KEYS = {"trial_id", "category", "garment_name", "release_stage", "seed"}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_source(path: Path, expected_sha256: str, label: str) -> object:
    if not path.is_absolute() or _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError(f"{label} source or SHA-256 is invalid")
    try:
        metadata = path.lstat()
    except OSError:
        raise ValueError(f"{label} source is unavailable") from None
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise ValueError(f"{label} source is unsafe")
    try:
        payload = path.read_bytes()
    except OSError:
        raise ValueError(f"{label} source is unreadable") from None
    if _sha256_bytes(payload) != expected_sha256:
        raise ValueError(f"{label} source SHA-256 mismatch")
    try:
        return json.loads(payload)
    except (UnicodeError, json.JSONDecodeError):
        raise ValueError(f"{label} source is not valid JSON") from None


def _rows(document: object, label: str) -> list[dict[str, object]]:
    if isinstance(document, Mapping):
        if set(document) != {"schema_version", "training_holdouts", "trials"}:
            raise ValueError(f"{label} challenge envelope is invalid")
        if document.get("schema_version") != 1 or not isinstance(document.get("training_holdouts"), list):
            raise ValueError(f"{label} challenge envelope is invalid")
        document = document.get("trials")
    if not isinstance(document, list) or not document:
        raise ValueError(f"{label} matrix must contain trials")
    normalized: list[dict[str, object]] = []
    trial_ids: set[str] = set()
    identities: set[tuple[object, ...]] = set()
    for value in document:
        if not isinstance(value, Mapping) or set(value) != _ROW_KEYS:
            raise ValueError(f"{label} trial schema is invalid")
        row = dict(value)
        trial_id = row.get("trial_id")
        category = row.get("category")
        garment = row.get("garment_name")
        stage = row.get("release_stage")
        seed = row.get("seed")
        if (
            type(trial_id) is not str or not trial_id
            or category not in _CATEGORIES
            or type(garment) is not str or not garment
            or stage not in {"seen", "public_unseen"}
            or type(seed) is not int or seed < 0
        ):
            raise ValueError(f"{label} trial identity is invalid")
        identity = (category, garment, stage, seed)
        if trial_id in trial_ids or identity in identities:
            raise ValueError(f"{label} matrix contains duplicate trials")
        trial_ids.add(trial_id)
        identities.add(identity)
        normalized.append(row)
    return normalized


def _balanced(rows: Sequence[Mapping[str, object]], count: int, label: str) -> None:
    expected = count // len(_CATEGORIES)
    if len(rows) != count or Counter(row["category"] for row in rows) != Counter({category: expected for category in _CATEGORIES}):
        raise ValueError(f"{label} matrix is not exactly balanced")


def _canonical_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return (json.dumps(list(rows), sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def _write_file(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def freeze_experiment_matrices(
    *,
    promotion_source: Path | str,
    promotion_source_sha256: str,
    final_source: Path | str,
    final_source_sha256: str,
    output_root: Path | str,
) -> dict[str, object]:
    promotion_path = Path(promotion_source)
    final_path = Path(final_source)
    destination = Path(output_root)
    if not destination.is_absolute() or destination.exists() or destination.is_symlink():
        raise ValueError("frozen matrix output already exists or is unsafe")
    parent = destination.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("frozen matrix output parent is unsafe")

    promotion = _rows(_read_source(promotion_path, promotion_source_sha256, "promotion"), "promotion")
    final_all = _rows(_read_source(final_path, final_source_sha256, "final"), "final")
    if len(final_all) == 280:
        if Counter(row["release_stage"] for row in final_all) != Counter({"seen": 200, "public_unseen": 80}):
            raise ValueError("final challenge matrix release-stage composition is invalid")
        _balanced(final_all, 280, "final challenge")
        final = [row for row in final_all if row["release_stage"] == "public_unseen"]
    elif len(final_all) == 80:
        final = final_all
    else:
        raise ValueError("final matrix must be the canonical 280 envelope or frozen unseen-80 list")
    if any(row["release_stage"] != "public_unseen" for row in promotion):
        raise ValueError("promotion matrix must contain only public-unseen trials")
    if any(row["release_stage"] != "public_unseen" for row in final):
        raise ValueError("final matrix must contain only public-unseen trials")
    _balanced(promotion, 20, "promotion")
    _balanced(final, 80, "final")
    promotion_ids = {str(row["trial_id"]) for row in promotion}
    final_ids = {str(row["trial_id"]) for row in final}
    promotion_identities = {(row["category"], row["garment_name"], row["seed"]) for row in promotion}
    final_identities = {(row["category"], row["garment_name"], row["seed"]) for row in final}
    if not promotion_ids.isdisjoint(final_ids) or not promotion_identities.isdisjoint(final_identities):
        raise ValueError("promotion and final evaluator trials must be disjoint")

    promotion_bytes = _canonical_bytes(promotion)
    final_bytes = _canonical_bytes(final)
    temporary = Path(tempfile.mkdtemp(prefix=".lehome-frozen-matrices-", dir=parent))
    try:
        files = {
            "promotion-matrix.json": promotion_bytes,
            "final-matrix.json": final_bytes,
        }
        digests: dict[str, str] = {}
        for name, payload in files.items():
            digest = _sha256_bytes(payload)
            digests[name] = digest
            _write_file(temporary / name, payload)
            _write_file(temporary / f"{name}.sha256", (digest + "\n").encode("ascii"))
        directory = os.open(temporary, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        os.replace(temporary, destination)
        parent_descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "schema_version": 1,
        "promotion_matrix": str(destination / "promotion-matrix.json"),
        "promotion_matrix_sha256": digests["promotion-matrix.json"],
        "final_matrix": str(destination / "final-matrix.json"),
        "final_matrix_sha256": digests["final-matrix.json"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promotion-source", type=Path, required=True)
    parser.add_argument("--promotion-source-sha256", required=True)
    parser.add_argument("--final-source", type=Path, required=True)
    parser.add_argument("--final-source-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = freeze_experiment_matrices(
        promotion_source=args.promotion_source,
        promotion_source_sha256=args.promotion_source_sha256,
        final_source=args.final_source,
        final_source_sha256=args.final_source_sha256,
        output_root=args.output_root,
    )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
