from __future__ import annotations

import importlib.util
import json

from lehome_train.challenge_evaluation import CANDIDATE_KEYS, MATRIX_SHA256, seal_report, select_challenge_winner
from test_challenge_evaluation import ROOT, _CATEGORIES, valid_report

SPEC = importlib.util.spec_from_file_location("challenge_select_under_test", ROOT / "scripts/select_groot_challenge_winner.py")
assert SPEC and SPEC.loader
SELECT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SELECT)


def _identity(key):
    step = 1000 if key in {"previous_step_1k", "new_step_1k"} else 2000
    identity = {"policy_repo": "ryanjin333/lehome-groot-n17-models", "policy_revision": str(CANDIDATE_KEYS.index(key) + 1) * 40, "policy_step": step, "policy_artifact_sha256": str(CANDIDATE_KEYS.index(key) + 1) * 64, "code_revision": "3" * 40}
    if key == "original_baseline":
        identity.update({"policy_revision": "30ac1a84da67b099e115ad147bcd61e9d60046d3", "policy_step": 12000, "policy_artifact_sha256": "3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06", "policy_subpath": "policies/step-12000", "policy_archive_sha256": "0ddd4e7ce351dd2172cd1edd967293a50d02c15c0f2c21ca39db94692a57e0b5"})
    return identity


def _report(key, unseen=56, categories=None, include_seen_full=True):
    report = valid_report(unseen_per_category=categories, include_seen_full=include_seen_full)
    report["candidate_key"], report["identity"] = key, _identity(key)
    if categories is None:
        quotient, remainder = divmod(unseen, 4)
        categories = {category: quotient + (index < remainder) for index, category in enumerate(_CATEGORIES)}
    split = report["splits"]["public_unseen"]
    for entry in split["trials"]:
        entry["official_success"] = 0
    for category in _CATEGORIES:
        for index, entry in enumerate(entry for entry in split["trials"] if entry["category"] == category):
            entry["official_success"] = int(index < categories[category])
        split["per_category"][category]["official_successes"] = categories[category]
    split["official_successes"] = sum(categories.values())
    return seal_report(report)


def _manifest(reports):
    return {"schema_version": 1, "kind": "lehome_challenge_evaluation_manifest", "candidate_keys": list(CANDIDATE_KEYS), "matrix_sha256": MATRIX_SHA256, "seen_dev_sha256": "e8412ac7edcdbbb8a09b9d19e65dfe851feff717c909f93733b46c2b2176124b", "seen_regression_tolerance": 0.05, "major_safety_tokens": ["unsafe", "collision", "self_collision"], "tie_breakers": ["unseen_successes", "category_floor_margin", "seen_regression", "safety_count", "checkpoint_order"], "identities": {report["candidate_key"]: {key: report["identity"][key] for key in (("policy_repo", "policy_revision", "policy_step", "policy_artifact_sha256", "code_revision", "policy_subpath", "policy_archive_sha256") if report["candidate_key"] == "original_baseline" else ("policy_repo", "policy_revision", "policy_step", "policy_artifact_sha256", "code_revision"))} for report in reports}}


def _bundle(reports):
    return {"schema_version": 1, "evaluation_manifest": _manifest(reports), "reports": [seal_report(report) for report in reports]}


def _set_split_successes(report, split_name, categories):
    split = report["splits"][split_name]
    for entry in split["trials"]:
        entry["official_success"] = 0
    for category in _CATEGORIES:
        for index, entry in enumerate(entry for entry in split["trials"] if entry["category"] == category):
            entry["official_success"] = int(index < categories[category])
        split["per_category"][category]["official_successes"] = categories[category]
    split["official_successes"] = sum(categories.values())


def _five(winner="new_step_2k", winner_categories=None, include_winner_seen_full=True):
    reports = [_report(key, 52 if key != "original_baseline" else 48) for key in CANDIDATE_KEYS]
    reports[CANDIDATE_KEYS.index(winner)] = _report(winner, 56, winner_categories, include_winner_seen_full)
    return reports


def test_floor_passing_winner_is_physically_approved():
    receipt = select_challenge_winner(_bundle(_five()))
    assert receipt["kind"] == "lehome_challenge_winner"
    assert receipt["winner_key"] == "new_step_2k"


def test_55_and_11_category_are_real_floor_failures():
    receipt = select_challenge_winner(_bundle(_five(winner_categories={"top_long": 13, "top_short": 14, "pant_long": 14, "pant_short": 14})))
    assert receipt["physical_test_approved"] is False
    receipt = select_challenge_winner(_bundle(_five(winner_categories={"top_long": 15, "top_short": 15, "pant_long": 15, "pant_short": 11})))
    assert receipt["physical_test_approved"] is False
    assert receipt["kind"] in {"lehome_challenge_rejected", "lehome_next_round_rollout"}


def test_safety_count_breaks_ties_without_disqualification():
    reports = _five(winner="previous_step_1k")
    reports[CANDIDATE_KEYS.index("new_step_2k")] = _report("new_step_2k", 56)
    candidate = reports[CANDIDATE_KEYS.index("previous_step_1k")]
    candidate["splits"]["public_unseen"]["trials"][0]["safety_count"] = 1
    candidate["splits"]["public_unseen"]["safety_count"] = 1
    candidate["splits"]["public_unseen"]["per_category"]["top_long"]["safety_count"] = 1
    reports[CANDIDATE_KEYS.index("previous_step_1k")] = seal_report(candidate)
    assert select_challenge_winner(_bundle(reports))["winner_key"] == "new_step_2k"


def test_seen_regression_uses_consistent_trial_and_split_totals():
    reports = _five()
    candidate = reports[-1]
    _set_split_successes(candidate, "seen_full", {"top_long": 41, "top_short": 45, "pant_long": 45, "pant_short": 45})
    assert candidate["splits"]["seen_full"]["official_successes"] == 176
    assert select_challenge_winner(_bundle(reports))["physical_test_approved"] is False


def test_seen_full_allows_two_of_fifty_but_disqualifies_three_of_fifty():
    reports = _five()
    candidate = reports[-1]
    _set_split_successes(candidate, "seen_full", {"top_long": 43, "top_short": 45, "pant_long": 45, "pant_short": 45})
    assert select_challenge_winner(_bundle(reports))["physical_test_approved"] is True
    reports = _five()
    candidate = reports[-1]
    _set_split_successes(candidate, "seen_full", {"top_long": 42, "top_short": 45, "pant_long": 45, "pant_short": 45})
    assert select_challenge_winner(_bundle(reports))["physical_test_approved"] is False


def test_seen_dev_one_of_six_regression_disqualifies_without_seen_full():
    reports = _five(include_winner_seen_full=False)
    for index, key in enumerate(CANDIDATE_KEYS[1:4], start=1):
        reports[index] = _report(key, 48)
    baseline = reports[0]
    baseline["splits"].pop("seen_full")
    candidate = reports[-1]
    _set_split_successes(candidate, "seen_dev", {"top_long": 4, "top_short": 5, "pant_long": 5, "pant_short": 5})
    receipt = select_challenge_winner(_bundle(reports))
    assert receipt["kind"] == "lehome_challenge_rejected"
    assert receipt["next_round"] is False


def test_identity_and_contradictory_provenance_reject_not_promote():
    reports = _five()
    bundle = _bundle(reports)
    reports[-1]["identity"]["policy_step"] = 9999
    assert select_challenge_winner(bundle)["kind"] == "lehome_challenge_rejected"
    reports = _five()
    reports[-1]["provenance"] = {"identity": {**reports[-1]["identity"], "policy_step": 9999}}
    assert select_challenge_winner(_bundle(reports))["kind"] == "lehome_challenge_rejected"


def test_duplicate_revision_or_artifact_across_keys_rejects():
    reports = _five()
    reports[-1]["identity"]["policy_revision"] = reports[1]["identity"]["policy_revision"]
    assert select_challenge_winner(_bundle(reports))["kind"] == "lehome_challenge_rejected"
    reports = _five()
    reports[-1]["identity"]["policy_artifact_sha256"] = reports[1]["identity"]["policy_artifact_sha256"]
    assert select_challenge_winner(_bundle(reports))["kind"] == "lehome_challenge_rejected"


def test_key_policy_steps_are_bound_to_their_candidate_keys():
    for bad_step in (12000, 1000):
        reports = _five()
        reports[-1]["identity"]["policy_step"] = bad_step
        assert select_challenge_winner(_bundle(reports))["kind"] == "lehome_challenge_rejected"


def test_missing_seen_full_is_rejected_incomplete_not_next_round():
    receipt = select_challenge_winner(_bundle(_five(include_winner_seen_full=False)))
    assert receipt["kind"] == "lehome_challenge_rejected" and receipt["next_round"] is False
    assert any("seen-full" in reason for reason in receipt["reasons"])


def test_higher_unseen_passer_wins_and_baseline_can_win():
    reports = _five(winner="new_step_2k")
    reports[CANDIDATE_KEYS.index("previous_step_1k")] = _report("previous_step_1k", 57)
    assert select_challenge_winner(_bundle(reports))["winner_key"] == "previous_step_1k"
    reports = [_report(key, 56) for key in CANDIDATE_KEYS]
    reports[0] = _report("original_baseline", 60)
    assert select_challenge_winner(_bundle(reports))["winner_key"] == "original_baseline"


def test_nonselected_passer_without_seen_full_does_not_block_selected_winner():
    reports = _five(winner="new_step_2k")
    reports[-1] = _report("new_step_2k", 57)
    reports[CANDIDATE_KEYS.index("previous_step_1k")] = _report("previous_step_1k", 56, include_seen_full=False)
    receipt = select_challenge_winner(_bundle(reports))
    assert receipt["winner_key"] == "new_step_2k"
    assert receipt["physical_test_approved"] is True


def test_improver_below_floors_gets_next_round_and_no_improvement_rejected():
    reports = [_report(key, 48 if key == "original_baseline" else 50) for key in CANDIDATE_KEYS]
    receipt = select_challenge_winner(_bundle(reports))
    assert receipt["kind"] == "lehome_next_round_rollout"
    assert receipt["attempt_cap"] == 400 and receipt["accepted_target"] == 150
    assert select_challenge_winner(_bundle([_report(key, 48) for key in CANDIDATE_KEYS]))["kind"] == "lehome_challenge_rejected"


def test_cli_exit_codes_and_filename_key_mismatch(tmp_path):
    reports_dir, manifest = tmp_path / "reports", tmp_path / "manifest.json"
    reports_dir.mkdir()
    for reports, expected, name in ((_five(), 0, "winner"), ([_report(key, 48 if key == "original_baseline" else 50) for key in CANDIDATE_KEYS], 2, "next"), ([_report(key, 48) for key in CANDIDATE_KEYS], 1, "rejected")):
        for report in reports:
            (reports_dir / f"{report['candidate_key']}.json").write_text(json.dumps(report), encoding="utf-8")
        manifest.write_text(json.dumps(_manifest(reports)), encoding="utf-8")
        output = tmp_path / f"{name}.json"
        assert SELECT.main(["--reports-dir", str(reports_dir), "--evaluation-manifest", str(manifest), "--output", str(output)]) == expected
        assert output.exists()
    (reports_dir / "new_step_2k.json").write_text(json.dumps(_report("previous_step_2k", 50)), encoding="utf-8")
    output = tmp_path / "invalid.json"
    assert SELECT.main(["--reports-dir", str(reports_dir), "--evaluation-manifest", str(manifest), "--output", str(output)]) == 1
    assert not output.exists()
