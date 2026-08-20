from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from lehome_train.challenge_evaluation import (
    MATRIX_SHA256,
    evaluate_candidate_gates,
    load_challenge_matrix,
    load_seen_dev_matrix,
    public_unseen_trials,
    seal_report,
    seen_full_trials,
    validate_challenge_report,
)

ROOT = Path(__file__).resolve().parents[2]
_CATEGORIES = ("top_long", "top_short", "pant_long", "pant_short")
SPEC = importlib.util.spec_from_file_location("challenge_publish_under_test", ROOT / "scripts/publish_groot_challenge_evaluation.py")
assert SPEC and SPEC.loader
PUBLISH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PUBLISH)


def _split(trials, per_category=None, safety_count=0):
    per_category = per_category or {category: min(15, sum(trial.category == category for trial in trials)) for category in _CATEGORIES}
    trial_successes = {category: 0 for category in _CATEGORIES}
    entries = []
    for trial in trials:
        success = int(trial_successes[trial.category] < per_category[trial.category])
        trial_successes[trial.category] += success
        entries.append({"trial_id": trial.trial_id, "category": trial.category, "official_success": success, "safety_count": 0})
    if safety_count:
        entries[0]["safety_count"] = safety_count
    return {
        "episodes": len(trials), "official_successes": sum(per_category.values()), "safety_count": safety_count,
        "trial_ids": [trial.trial_id for trial in trials], "trials": entries,
        "per_category": {
            category: {
                "episodes": sum(trial.category == category for trial in trials),
                "official_successes": per_category[category],
                "safety_count": safety_count if category == trials[0].category else 0,
            }
            for category in _CATEGORIES
        },
        "safety_failures": [],
    }


def valid_report(*, unseen_per_category=None, include_seen_full=True):
    matrix = load_challenge_matrix(ROOT / "configs/eval_groot_n17_public_280.json")
    splits = {
        "public_unseen": _split(public_unseen_trials(matrix), unseen_per_category),
        "seen_dev": _split(load_seen_dev_matrix(ROOT / "configs/eval_groot_n17_seen_dev.json"), {category: 5 for category in _CATEGORIES}),
    }
    if include_seen_full:
        splits["seen_full"] = _split(seen_full_trials(matrix), {category: 45 for category in _CATEGORIES})
    return seal_report({"schema_version": 1, "kind": "lehome_challenge_evaluation", "candidate_key": "new_step_2k", "identity": {"policy_repo": "ryanjin333/lehome-groot-n17-models", "policy_revision": "1" * 40, "policy_step": 2000, "policy_artifact_sha256": "2" * 64, "code_revision": "3" * 40}, "matrix_sha256": MATRIX_SHA256, "splits": splits})


def test_admits_canonical_trials_and_hashes_seen_dev_file():
    matrix = load_challenge_matrix(ROOT / "configs/eval_groot_n17_public_280.json")
    seen_dev_path = ROOT / "configs/eval_groot_n17_seen_dev.json"
    assert len(public_unseen_trials(matrix)) == 80
    assert len(seen_full_trials(matrix)) == 200
    assert hashlib.sha256(seen_dev_path.read_bytes()).hexdigest() == "e8412ac7edcdbbb8a09b9d19e65dfe851feff717c909f93733b46c2b2176124b"
    validate_challenge_report(valid_report(), matrix, load_seen_dev_matrix(seen_dev_path))


def test_rejects_seen_dev_file_with_different_bytes_even_when_shape_is_valid(tmp_path):
    canonical = ROOT / "configs/eval_groot_n17_seen_dev.json"
    drifted = tmp_path / "seen-dev.json"
    drifted.write_bytes(canonical.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="seen-dev.*SHA-256"):
        load_seen_dev_matrix(drifted)


def test_trials_bind_order_categories_and_declared_aggregates():
    matrix = load_challenge_matrix(ROOT / "configs/eval_groot_n17_public_280.json")
    seen_dev = load_seen_dev_matrix(ROOT / "configs/eval_groot_n17_seen_dev.json")
    for mutate in (
        lambda report: report["splits"]["public_unseen"]["trials"][0].__setitem__("trial_id", "wrong"),
        lambda report: report["splits"]["public_unseen"]["trials"][0].__setitem__("category", "pant_short"),
        lambda report: report["splits"]["public_unseen"]["trials"][0].__setitem__("official_success", True),
        lambda report: report["splits"]["public_unseen"]["trials"][0].__setitem__("safety_count", True),
        lambda report: report["splits"]["public_unseen"].__setitem__("official_successes", 59),
        lambda report: report["splits"]["public_unseen"]["per_category"]["top_long"].__setitem__("official_successes", 14),
        lambda report: report["splits"]["public_unseen"].__setitem__("safety_count", 1),
    ):
        report = valid_report()
        mutate(report)
        with pytest.raises(ValueError):
            validate_challenge_report(report, matrix, seen_dev)


def test_safety_count_is_rankable_but_major_safety_token_disqualifies():
    report = valid_report()
    report["splits"]["public_unseen"]["trials"][0]["safety_count"] = 1
    report["splits"]["public_unseen"]["safety_count"] = 1
    report = seal_report(report)
    assert evaluate_candidate_gates(report)["safety_passed"] is True
    report["splits"]["public_unseen"]["outcome"] = "minor collision"
    report = seal_report(report)
    assert evaluate_candidate_gates(report)["safety_passed"] is False


def test_category_safety_count_must_match_trial_evidence():
    matrix = load_challenge_matrix(ROOT / "configs/eval_groot_n17_public_280.json")
    seen_dev = load_seen_dev_matrix(ROOT / "configs/eval_groot_n17_seen_dev.json")
    report = valid_report()
    report["splits"]["public_unseen"]["trials"][0]["safety_count"] = 1
    report["splits"]["public_unseen"]["safety_count"] = 1
    report["splits"]["public_unseen"]["per_category"]["top_long"]["safety_count"] = 0
    report = seal_report(report)
    with pytest.raises(ValueError, match="safety count"):
        validate_challenge_report(report, matrix, seen_dev)


def test_split_requires_explicit_safety_failure_declaration():
    matrix = load_challenge_matrix(ROOT / "configs/eval_groot_n17_public_280.json")
    seen_dev = load_seen_dev_matrix(ROOT / "configs/eval_groot_n17_seen_dev.json")
    report = valid_report()
    del report["splits"]["public_unseen"]["safety_failures"]
    report = seal_report(report)
    with pytest.raises(ValueError, match="incomplete"):
        validate_challenge_report(report, matrix, seen_dev)


def test_publish_refuses_tops_reports_and_recomputes_from_trials(tmp_path):
    matrix = ROOT / "configs/eval_groot_n17_public_280.json"
    seen_dev = ROOT / "configs/eval_groot_n17_seen_dev.json"
    report = valid_report()
    report_path = tmp_path / "new_step_2k.json"
    report["splits"]["public_unseen"]["official_successes"] = 1
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="aggregate"):
        PUBLISH.build_publication_plan(report_path, tmp_path / "bad", matrix, seen_dev)
    report = valid_report()
    report["kind"] = "public_unseen_tops_evaluation"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="top-40"):
        PUBLISH.build_publication_plan(report_path, tmp_path / "tops", matrix, seen_dev)


def test_publish_writes_sealed_report_snapshot_and_matching_hashes(tmp_path):
    report_path = tmp_path / "new_step_2k.json"
    report_path.write_text(json.dumps(valid_report()), encoding="utf-8")
    root = tmp_path / "publication"
    PUBLISH.build_publication_plan(
        report_path,
        root,
        ROOT / "configs/eval_groot_n17_public_280.json",
        ROOT / "configs/eval_groot_n17_seen_dev.json",
    )
    expected = {"challenge-evaluation-report.json", "challenge-evaluation-manifest.json", "SHA256SUMS.json"}
    assert {path.name for path in root.iterdir()} == expected
    sums = json.loads((root / "SHA256SUMS.json").read_text(encoding="utf-8"))
    for name in expected - {"SHA256SUMS.json"}:
        assert sums[name]["sha256"] == hashlib.sha256((root / name).read_bytes()).hexdigest()


def test_async_sweep_selection_is_a_separate_fail_closed_path_from_legacy_five_candidate_gate():
    from lehome_train.challenge_evaluation import select_async_sweep_final_winner

    assert select_async_sweep_final_winner(
        {},
        baseline_report=None,
        original_12k_checkpoint_digest="a" * 64,
        final_matrix_sha256="b" * 64,
    ) == {"decision": "baseline_evaluation_required"}
