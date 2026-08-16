"""Select a physical-test winner or immutable next-round rollout manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lehome.flywheel.artifacts import atomic_write_json
from lehome_train.challenge_evaluation import (
    CANDIDATE_KEYS,
    MATRIX_SHA256,
    load_challenge_matrix,
    load_seen_dev_matrix,
    select_challenge_winner,
    validate_evaluation_manifest,
    validate_challenge_report,
)


def _read_json(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("input must be a regular JSON file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("input JSON is invalid") from error
    if not isinstance(value, dict):
        raise ValueError("input JSON must be an object")
    return value


def load_selection_bundle(reports_dir: Path, evaluation_manifest: Path) -> dict[str, object]:
    manifest = _read_json(evaluation_manifest)
    validate_evaluation_manifest(manifest)
    reports: list[dict[str, object]] = []
    for key in CANDIDATE_KEYS:
        reports.append(_read_json(Path(reports_dir) / f"{key}.json"))
    root = Path(__file__).resolve().parents[1]
    matrix = load_challenge_matrix(root / "configs/eval_groot_n17_public_280.json")
    seen_dev = load_seen_dev_matrix(root / "configs/eval_groot_n17_seen_dev.json")
    for key, report in zip(CANDIDATE_KEYS, reports):
        if report.get("candidate_key") != key:
            raise ValueError("report candidate key does not match its filename")
        validate_challenge_report(report, matrix, seen_dev)
    return {"schema_version": 1, "evaluation_manifest": manifest, "reports": reports}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", required=True, type=Path)
    parser.add_argument("--evaluation-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        receipt = select_challenge_winner(load_selection_bundle(args.reports_dir, args.evaluation_manifest))
    except ValueError:
        return 1
    if args.output.exists() or args.output.is_symlink():
        return 1
    atomic_write_json(args.output, receipt)
    if receipt.get("kind") == "lehome_challenge_winner" and receipt.get("physical_test_approved") is True:
        return 0
    if receipt.get("kind") == "lehome_next_round_rollout":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
