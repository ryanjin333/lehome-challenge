"""Seal one LeHome challenge evaluation report into an atomic local snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lehome.flywheel.artifacts import atomic_write_json, build_sha256_manifest
from lehome_train.challenge_evaluation import (
    load_challenge_matrix,
    load_seen_dev_matrix,
    score_split,
    seal_report,
    validate_challenge_report,
)


def _read_json(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("challenge report must be a regular JSON file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("challenge report is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("challenge report must be an object")
    return value


def build_publication_plan(report_path: Path, staging_root: Path, matrix_path: Path, seen_dev_path: Path) -> dict[str, object]:
    report = _read_json(report_path)
    if report.get("kind") == "public_unseen_tops_evaluation":
        raise ValueError("top-40 diagnostic reports cannot be published as challenge evaluations")
    matrix = load_challenge_matrix(matrix_path)
    seen_dev = load_seen_dev_matrix(seen_dev_path)
    validate_challenge_report(report, matrix, seen_dev, require_digest=False)
    sealed = seal_report(report)
    validate_challenge_report(sealed, matrix, seen_dev)
    root = Path(staging_root)
    if root.exists() or root.is_symlink():
        raise ValueError("challenge evaluation staging root must not already exist")
    root.mkdir(parents=True)
    try:
        atomic_write_json(root / "challenge-evaluation-report.json", sealed)
        snapshot = {
            "schema_version": 1,
            "kind": "lehome_challenge_evaluation_snapshot",
            "candidate_key": sealed["candidate_key"],
            "matrix_sha256": sealed["matrix_sha256"],
            "identity": sealed["identity"],
            "report_sha256": sealed["report_sha256"],
            "splits": {
                name: {"trial_ids": split["trial_ids"], "metrics": score_split(split)}
                for name, split in sealed["splits"].items()
            },
        }
        atomic_write_json(root / "challenge-evaluation-manifest.json", snapshot)
        atomic_write_json(root / "SHA256SUMS.json", build_sha256_manifest(root))
    except BaseException:
        import shutil
        shutil.rmtree(root, ignore_errors=True)
        raise
    return {"root": str(root), "candidate_key": sealed["candidate_key"], "report_sha256": sealed["report_sha256"], "snapshot": snapshot}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--staging-root", required=True, type=Path)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--seen-dev-matrix", required=True, type=Path)
    args = parser.parse_args(argv)
    build_publication_plan(args.report, args.staging_root, args.matrix, args.seen_dev_matrix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
