"""Fail-closed selection for the five-candidate LeHome challenge gate."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence

from lehome.flywheel.matrix import PublicMatrix, Trial, load_public_matrix, matrix_sha256


CATEGORIES = ("top_long", "top_short", "pant_long", "pant_short")
CANDIDATE_KEYS = ("original_baseline", "previous_step_1k", "previous_step_2k", "new_step_1k", "new_step_2k")
UNSEEN_OVERALL_MIN = 56
UNSEEN_CATEGORY_MIN = 12
UNSEEN_EPISODES = 80
SEEN_DEV_EPISODES = 24
SEEN_FULL_EPISODES = 200
NEXT_ROUND_ATTEMPT_CAP = 400
NEXT_ROUND_ACCEPTED_TARGET = 150
SEEN_REGRESSION_TOLERANCE = 0.05
MATRIX_SHA256 = "a3b15b5e4df2c68be6f3ea06eae4d8c2418714c45cf6af2a7f42e982225464b9"
SEEN_DEV_SHA256 = "e8412ac7edcdbbb8a09b9d19e65dfe851feff717c909f93733b46c2b2176124b"
MAJOR_SAFETY_TOKENS = ("unsafe", "collision", "self_collision")
TIE_BREAKERS = ("unseen_successes", "category_floor_margin", "seen_regression", "safety_count", "checkpoint_order")

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TRAINER_OCI = re.compile(r"^ghcr.io/ryanjin333/lehome-groot-n17-trainer@sha256:[0-9a-f]{64}$")
_IDENTITY_FIELDS = ("policy_repo", "policy_revision", "policy_step", "policy_artifact_sha256", "code_revision")
_ORIGINAL_PIN_FIELDS = ("policy_subpath", "policy_archive_sha256")
_POLICY_STEPS = {
    "original_baseline": 12000,
    "previous_step_1k": 1000,
    "new_step_1k": 1000,
    "previous_step_2k": 2000,
    "new_step_2k": 2000,
}
_ORIGINAL_IDENTITY = {
    "policy_repo": "ryanjin333/lehome-groot-n17-models",
    "policy_revision": "30ac1a84da67b099e115ad147bcd61e9d60046d3",
    "policy_step": 12000,
    "policy_artifact_sha256": "3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06",
    "policy_subpath": "policies/step-12000",
    "policy_archive_sha256": "0ddd4e7ce351dd2172cd1edd967293a50d02c15c0f2c21ca39db94692a57e0b5",
}


def load_seen_dev_matrix(path: Path) -> tuple[object, ...]:
    """Load the frozen dev matrix through its canonical validator script."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("seen-dev matrix must be a regular file")
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError("seen-dev matrix cannot be read") from error
    if digest != SEEN_DEV_SHA256:
        raise ValueError("seen-dev matrix SHA-256 drifted")
    script = Path(__file__).resolve().parents[3] / "scripts/eval_groot_n17_matrix.py"
    spec = importlib.util.spec_from_file_location("lehome_seen_dev_matrix", script)
    if spec is None or spec.loader is None:
        raise ValueError("canonical seen-dev matrix validator is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return tuple(module.load_matrix(path))


def _require_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer at least {minimum}")
    return value


def _canonical_digest(value: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()


def report_digest(report: Mapping[str, object]) -> str:
    payload = dict(report)
    payload.pop("report_sha256", None)
    return _canonical_digest(payload)


def seal_report(report: Mapping[str, object]) -> dict[str, object]:
    payload = dict(report)
    payload.pop("report_sha256", None)
    payload["report_sha256"] = report_digest(payload)
    return payload


def load_challenge_matrix(path: Path) -> PublicMatrix:
    matrix = load_public_matrix(Path(path))
    if matrix_sha256(matrix) != MATRIX_SHA256:
        raise ValueError("canonical public matrix SHA-256 drifted")
    return matrix


def public_unseen_trials(matrix: PublicMatrix) -> tuple[Trial, ...]:
    trials = tuple(trial for trial in matrix.trials if trial.release_stage == "public_unseen")
    _validate_trials(trials, UNSEEN_EPISODES, 20, "public-unseen")
    return trials


def seen_full_trials(matrix: PublicMatrix) -> tuple[Trial, ...]:
    trials = tuple(trial for trial in matrix.trials if trial.release_stage == "seen")
    _validate_trials(trials, SEEN_FULL_EPISODES, 50, "seen-full")
    return trials


def _validate_trials(trials: Sequence[object], expected: int, per_category: int, label: str) -> None:
    if len(trials) != expected or len({getattr(trial, "trial_id", None) for trial in trials}) != expected:
        raise ValueError(f"{label} trials are not the exact frozen matrix")
    for category in CATEGORIES:
        if sum(getattr(trial, "category", None) == category for trial in trials) != per_category:
            raise ValueError(f"{label} trial categories are not balanced")


def validate_candidate_identity(candidate: Mapping[str, object]) -> Mapping[str, object]:
    identity = candidate.get("identity") if "identity" in candidate else candidate
    if not isinstance(identity, Mapping):
        raise ValueError("candidate identity is missing")
    if not set(_IDENTITY_FIELDS).issubset(identity):
        raise ValueError("candidate identity is incomplete")
    if not isinstance(identity["policy_repo"], str) or not identity["policy_repo"] or any(ch.isspace() for ch in identity["policy_repo"]):
        raise ValueError("candidate policy repository is unsafe")
    if not isinstance(identity["policy_revision"], str) or not _COMMIT.fullmatch(identity["policy_revision"]):
        raise ValueError("candidate policy revision must be an immutable commit")
    _require_int(identity["policy_step"], "candidate policy step", minimum=0)
    if not isinstance(identity["policy_artifact_sha256"], str) or not _SHA256.fullmatch(identity["policy_artifact_sha256"]):
        raise ValueError("candidate policy artifact digest is invalid")
    if not isinstance(identity["code_revision"], str) or not _COMMIT.fullmatch(identity["code_revision"]):
        raise ValueError("candidate code revision must be an immutable commit")
    if "policy_archive_sha256" in identity and (not isinstance(identity["policy_archive_sha256"], str) or not _SHA256.fullmatch(identity["policy_archive_sha256"])):
        raise ValueError("candidate policy archive digest is invalid")
    if "policy_subpath" in identity and (not isinstance(identity["policy_subpath"], str) or not identity["policy_subpath"] or identity["policy_subpath"].startswith("/") or ".." in identity["policy_subpath"].split("/")):
        raise ValueError("candidate policy subpath is unsafe")
    return identity


def _validate_original_identity(identity: Mapping[str, object]) -> None:
    for field in ("policy_repo", "policy_revision", "policy_step", "policy_artifact_sha256", *_ORIGINAL_PIN_FIELDS):
        if identity.get(field) != _ORIGINAL_IDENTITY[field]:
            raise ValueError("original baseline identity does not match pinned parent")


def _validate_candidate_key_identity(candidate_key: str, identity: Mapping[str, object]) -> None:
    if identity["policy_step"] != _POLICY_STEPS[candidate_key]:
        raise ValueError("candidate policy step does not match its predeclared key")
    if candidate_key == "original_baseline":
        _validate_original_identity(identity)


def _split_trials(split_name: str, matrix: PublicMatrix, seen_dev: Sequence[object]) -> tuple[object, ...]:
    if split_name == "public_unseen":
        return public_unseen_trials(matrix)
    if split_name == "seen_full":
        return seen_full_trials(matrix)
    if split_name == "seen_dev":
        trials = tuple(seen_dev)
        _validate_trials(trials, SEEN_DEV_EPISODES, 6, "seen-dev")
        return trials
    raise ValueError("unknown challenge split")


def score_split(report: Mapping[str, object]) -> dict[str, object]:
    episodes = _require_int(report.get("episodes"), "split episodes", minimum=1)
    successes = _require_int(report.get("official_successes"), "split official successes")
    if successes > episodes:
        raise ValueError("split successes exceed episodes")
    categories = report.get("per_category")
    if not isinstance(categories, Mapping) or set(categories) != set(CATEGORIES):
        raise ValueError("split must contain exactly four categories")
    result: dict[str, object] = {"episodes": episodes, "official_successes": successes, "per_category": {}}
    total = 0
    for category in CATEGORIES:
        item = categories[category]
        if not isinstance(item, Mapping):
            raise ValueError("category score must be an object")
        item_episodes = _require_int(item.get("episodes"), f"{category} episodes", minimum=1)
        item_successes = _require_int(item.get("official_successes"), f"{category} successes")
        if item_successes > item_episodes:
            raise ValueError("category successes exceed episodes")
        total += item_successes
        result["per_category"][category] = {"episodes": item_episodes, "official_successes": item_successes}
    if total != successes:
        raise ValueError("split aggregate successes do not equal categories")
    return result


def _validate_split_evidence(split: Mapping[str, object], expected_trials: Sequence[object], split_name: str) -> None:
    required = {"episodes", "official_successes", "safety_count", "safety_failures", "trial_ids", "trials", "per_category"}
    if not required.issubset(split):
        raise ValueError("challenge split is incomplete")
    trial_ids = split.get("trial_ids")
    if not isinstance(trial_ids, list) or trial_ids != [trial.trial_id for trial in expected_trials]:
        raise ValueError(f"{split_name} trial IDs do not match the frozen matrix")
    evidence = split.get("trials")
    if not isinstance(evidence, list) or len(evidence) != len(expected_trials):
        raise ValueError(f"{split_name} trials do not match the frozen matrix")
    successes = {category: 0 for category in CATEGORIES}
    category_safety = {category: 0 for category in CATEGORIES}
    safety_total = 0
    for expected, actual in zip(expected_trials, evidence):
        if not isinstance(actual, Mapping) or set(actual) != {"trial_id", "category", "official_success", "safety_count"}:
            raise ValueError(f"{split_name} trial evidence fields are invalid")
        if actual["trial_id"] != expected.trial_id or actual["category"] != expected.category:
            raise ValueError(f"{split_name} trials do not match the frozen matrix")
        success = _require_int(actual["official_success"], "trial official success")
        if success not in (0, 1):
            raise ValueError("trial official success must be exactly zero or one")
        safety = _require_int(actual["safety_count"], "trial safety count")
        successes[expected.category] += success
        category_safety[expected.category] += safety
        safety_total += safety
    score = score_split(split)
    if score["episodes"] != len(expected_trials):
        raise ValueError(f"{split_name} episode count does not match the frozen matrix")
    if score["official_successes"] != sum(successes.values()):
        raise ValueError(f"{split_name} declared aggregates do not match trials")
    for category in CATEGORIES:
        declared = score["per_category"][category]
        if declared["episodes"] != sum(trial.category == category for trial in expected_trials):
            raise ValueError(f"{split_name} category episode count is invalid")
        if declared["official_successes"] != successes[category]:
            raise ValueError(f"{split_name} declared aggregates do not match trials")
        if _require_int(split["per_category"][category].get("safety_count"), f"{split_name} {category} safety count") != category_safety[category]:
            raise ValueError(f"{split_name} declared category safety count does not match trials")
    if _require_int(split["safety_count"], f"{split_name} safety count") != safety_total:
        raise ValueError(f"{split_name} declared safety count does not match trials")


def _validate_safety_metadata(split: Mapping[str, object]) -> None:
    failures = split.get("safety_failures")
    if not isinstance(failures, list) or not all(isinstance(item, str) for item in failures):
        raise ValueError("safety_failures must be a list of strings")
    outcome = split.get("outcome")
    if outcome is not None and not isinstance(outcome, str) and not (isinstance(outcome, list) and all(isinstance(item, str) for item in outcome)):
        raise ValueError("outcome must be a string or string list")


def _safety_failed(split: Mapping[str, object]) -> bool:
    for value in (split.get("outcome"), split.get("safety_failures")):
        values = (value,) if isinstance(value, str) else value if isinstance(value, list) else ()
        if any(any(token in item.lower() for token in MAJOR_SAFETY_TOKENS) for item in values):
            return True
    return False


def _validate_provenance(report: Mapping[str, object], identity: Mapping[str, object]) -> None:
    provenance = report.get("provenance")
    if provenance is None:
        return
    if not isinstance(provenance, Mapping):
        raise ValueError("report provenance is invalid")
    claimed = provenance.get("identity", provenance)
    if not isinstance(claimed, Mapping) or any(claimed.get(field) != identity[field] for field in _IDENTITY_FIELDS):
        raise ValueError("report provenance contradicts identity")


def validate_challenge_report(report: Mapping[str, object], matrix: PublicMatrix, seen_dev: Sequence[object], *, require_digest: bool = True) -> None:
    if not isinstance(report, Mapping) or report.get("schema_version") != 1 or report.get("kind") != "lehome_challenge_evaluation":
        raise ValueError("not a challenge evaluation report")
    if report.get("candidate_key") not in CANDIDATE_KEYS:
        raise ValueError("candidate key is not predeclared")
    allowed = {"schema_version", "kind", "candidate_key", "identity", "matrix_sha256", "splits", "report_sha256", "provenance", "trainer_oci"}
    if not set(report).issubset(allowed):
        raise ValueError("challenge report has unrecognized fields")
    if report.get("matrix_sha256") != MATRIX_SHA256:
        raise ValueError("challenge report matrix SHA-256 does not match")
    identity = validate_candidate_identity(report)
    _validate_candidate_key_identity(report["candidate_key"], identity)
    _validate_provenance(report, identity)
    trainer_oci = report.get("trainer_oci")
    if trainer_oci is not None and (not isinstance(trainer_oci, str) or not _TRAINER_OCI.fullmatch(trainer_oci)):
        raise ValueError("trainer OCI is invalid")
    digest = report.get("report_sha256")
    if require_digest and (not isinstance(digest, str) or not _SHA256.fullmatch(digest) or digest != report_digest(report)):
        raise ValueError("challenge report digest does not bind report bytes")
    splits = report.get("splits")
    if not isinstance(splits, Mapping) or not {"public_unseen", "seen_dev"}.issubset(splits) or not set(splits).issubset({"public_unseen", "seen_dev", "seen_full"}):
        raise ValueError("challenge report split set is invalid")
    for name, split in splits.items():
        if not isinstance(split, Mapping):
            raise ValueError("challenge split must be an object")
        _validate_split_evidence(split, _split_trials(name, matrix, seen_dev), name)
        _validate_safety_metadata(split)


def validate_evaluation_manifest(manifest: Mapping[str, object]) -> Mapping[str, object]:
    expected = {"schema_version", "kind", "candidate_keys", "matrix_sha256", "seen_dev_sha256", "seen_regression_tolerance", "major_safety_tokens", "tie_breakers", "identities"}
    if not isinstance(manifest, Mapping) or set(manifest) != expected:
        raise ValueError("evaluation manifest field set is invalid")
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "lehome_challenge_evaluation_manifest":
        raise ValueError("evaluation manifest schema is invalid")
    if manifest.get("candidate_keys") != list(CANDIDATE_KEYS) or manifest.get("matrix_sha256") != MATRIX_SHA256 or manifest.get("seen_dev_sha256") != SEEN_DEV_SHA256:
        raise ValueError("evaluation manifest does not bind frozen inputs")
    if type(manifest.get("seen_regression_tolerance")) is not float or manifest["seen_regression_tolerance"] != SEEN_REGRESSION_TOLERANCE:
        raise ValueError("evaluation manifest regression tolerance is invalid")
    if manifest.get("major_safety_tokens") != list(MAJOR_SAFETY_TOKENS) or manifest.get("tie_breakers") != list(TIE_BREAKERS):
        raise ValueError("evaluation manifest gate policy is invalid")
    identities = manifest.get("identities")
    if not isinstance(identities, Mapping) or set(identities) != set(CANDIDATE_KEYS):
        raise ValueError("evaluation manifest identities are incomplete")
    for key in CANDIDATE_KEYS:
        identity = identities[key]
        fields = (*_IDENTITY_FIELDS, *_ORIGINAL_PIN_FIELDS) if key == "original_baseline" else _IDENTITY_FIELDS
        if not isinstance(identity, Mapping) or set(identity) != set(fields):
            raise ValueError("evaluation manifest identity field set is invalid")
        validate_candidate_identity(identity)
        _validate_candidate_key_identity(key, identity)
    revisions = [identities[key]["policy_revision"] for key in CANDIDATE_KEYS]
    artifacts = [identities[key]["policy_artifact_sha256"] for key in CANDIDATE_KEYS]
    if len(set(revisions)) != len(revisions) or len(set(artifacts)) != len(artifacts):
        raise ValueError("evaluation manifest candidate identities are not unique")
    return manifest


def evaluate_candidate_gates(candidate_report: Mapping[str, object]) -> dict[str, object]:
    splits = candidate_report.get("splits")
    if not isinstance(splits, Mapping):
        return {"passed": False, "reasons": ["missing splits"], "unseen_passed": False, "safety_passed": False}
    reasons: list[str] = []
    unseen_score = None
    unseen = splits.get("public_unseen")
    if isinstance(unseen, Mapping):
        try:
            unseen_score = score_split(unseen)
            if unseen_score["episodes"] != UNSEEN_EPISODES or unseen_score["official_successes"] < UNSEEN_OVERALL_MIN:
                reasons.append("unseen overall threshold failed")
            if any(unseen_score["per_category"][category]["official_successes"] < UNSEEN_CATEGORY_MIN for category in CATEGORIES):
                reasons.append("unseen category floor failed")
        except ValueError as error:
            reasons.append(str(error))
    else:
        reasons.append("missing public_unseen")
    seen_dev = splits.get("seen_dev")
    if not isinstance(seen_dev, Mapping):
        reasons.append("missing seen-dev screen")
    else:
        try:
            if score_split(seen_dev)["episodes"] != SEEN_DEV_EPISODES:
                reasons.append("seen-dev screen is incomplete")
        except ValueError as error:
            reasons.append(str(error))
    safety_passed = not any(_safety_failed(split) for split in splits.values() if isinstance(split, Mapping))
    if not safety_passed:
        reasons.append("major safety failure")
    return {"passed": not reasons, "reasons": reasons, "unseen_passed": unseen_score is not None and unseen_score["official_successes"] >= UNSEEN_OVERALL_MIN and all(unseen_score["per_category"][category]["official_successes"] >= UNSEEN_CATEGORY_MIN for category in CATEGORIES), "safety_passed": safety_passed}


def _regression(baseline: Mapping[str, object], candidate: Mapping[str, object]) -> tuple[bool, int]:
    baseline_splits = baseline["splits"]
    candidate_splits = candidate["splits"]
    common = "seen_full" if "seen_full" in baseline_splits and "seen_full" in candidate_splits else "seen_dev"
    baseline_score, candidate_score = score_split(baseline_splits[common]), score_split(candidate_splits[common])
    total, failed = 0, False
    for category in CATEGORIES:
        base, other = baseline_score["per_category"][category], candidate_score["per_category"][category]
        if base["episodes"] != other["episodes"]:
            raise ValueError("seen comparison episode counts differ")
        deficit = max(0, base["official_successes"] - other["official_successes"])
        total += deficit
        if deficit * 20 > base["episodes"]:
            failed = True
    return failed, total


def _candidate_key(report: Mapping[str, object], baseline: Mapping[str, object]) -> tuple[int, int, int, int, int]:
    unseen = score_split(report["splits"]["public_unseen"])
    margin = min(unseen["per_category"][category]["official_successes"] - UNSEEN_CATEGORY_MIN for category in CATEGORIES)
    _, regression = _regression(baseline, report)
    safety = sum(_require_int(split["safety_count"], "split safety count") for split in report["splits"].values())
    return (-unseen["official_successes"], -margin, regression, safety, CANDIDATE_KEYS.index(report["candidate_key"]))


def build_next_round_manifest(winner_identity: Mapping[str, object]) -> dict[str, object]:
    identity = validate_candidate_identity(winner_identity)
    winner_key = winner_identity.get("candidate_key")
    if winner_key not in CANDIDATE_KEYS:
        raise ValueError("next-round winner key is not predeclared")
    return {"schema_version": 1, "kind": "lehome_next_round_rollout", "physical_test_approved": False, "winner_key": winner_key, "attempt_cap": NEXT_ROUND_ATTEMPT_CAP, "accepted_target": NEXT_ROUND_ACCEPTED_TARGET, "identity": dict(identity)}


def _rejected(reasons: list[str]) -> dict[str, object]:
    return {"schema_version": 1, "kind": "lehome_challenge_rejected", "physical_test_approved": False, "next_round": False, "reasons": reasons}


def select_challenge_winner(bundle: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(bundle, Mapping) or set(bundle) != {"schema_version", "evaluation_manifest", "reports"} or bundle.get("schema_version") != 1:
        raise ValueError("selection bundle field set is invalid")
    try:
        validate_evaluation_manifest(bundle["evaluation_manifest"])
    except ValueError as error:
        return _rejected([str(error)])
    reports = bundle["reports"]
    if not isinstance(reports, list) or len(reports) != len(CANDIDATE_KEYS):
        return _rejected(["exactly five candidate reports are required"])
    by_key: dict[str, Mapping[str, object]] = {}
    for report in reports:
        if not isinstance(report, Mapping) or report.get("candidate_key") in by_key:
            return _rejected(["candidate reports must have unique predeclared keys"])
        by_key[str(report.get("candidate_key"))] = report
    if set(by_key) != set(CANDIDATE_KEYS):
        return _rejected(["candidate report keys do not match the predeclared five"])
    matrix = load_challenge_matrix(Path(__file__).resolve().parents[3] / "configs/eval_groot_n17_public_280.json")
    seen_dev = load_seen_dev_matrix(Path(__file__).resolve().parents[3] / "configs/eval_groot_n17_seen_dev.json")
    manifest_identities = bundle["evaluation_manifest"]["identities"]
    invalid: dict[str, str] = {}
    for key, report in by_key.items():
        try:
            validate_challenge_report(report, matrix, seen_dev)
            identity = report["identity"]
            fields = (*_IDENTITY_FIELDS, *_ORIGINAL_PIN_FIELDS) if key == "original_baseline" else _IDENTITY_FIELDS
            if any(identity[field] != manifest_identities[key][field] for field in fields):
                raise ValueError("report identity contradicts evaluation manifest")
        except ValueError as error:
            invalid[key] = str(error)
    if invalid:
        return _rejected([f"{key}: {reason}" for key, reason in sorted(invalid.items())])
    baseline = by_key["original_baseline"]
    baseline_unseen = score_split(baseline["splits"]["public_unseen"])["official_successes"]
    passing: list[Mapping[str, object]] = []
    improvers: list[Mapping[str, object]] = []
    for key, report in by_key.items():
        gates = evaluate_candidate_gates(report)
        regressed, _ = _regression(baseline, report)
        unseen = score_split(report["splits"]["public_unseen"])["official_successes"]
        if gates["passed"] and not regressed:
            passing.append(report)
        if key != "original_baseline" and not gates["unseen_passed"] and gates["safety_passed"] and not regressed and unseen > baseline_unseen:
            improvers.append(report)
    if passing:
        candidate = min(passing, key=lambda report: _candidate_key(report, baseline))
        if "seen_full" not in candidate["splits"] or "seen_full" not in baseline["splits"]:
            return _rejected(["incomplete seen-full evidence prevents physical promotion"])
        _, regression = _regression(baseline, candidate)
        unseen = score_split(candidate["splits"]["public_unseen"])
        return {"schema_version": 1, "kind": "lehome_challenge_winner", "physical_test_approved": True, "winner_key": candidate["candidate_key"], "unseen": unseen, "tie_breakers_applied": list(TIE_BREAKERS), "provenance": {"identity": dict(candidate["identity"]), "report_sha256": candidate["report_sha256"], "matrix_sha256": MATRIX_SHA256}, "seen_full_comparison_vs_baseline": {"baseline_key": "original_baseline", "total_positive_success_deficit": regression}}
    if improvers:
        return build_next_round_manifest(min(improvers, key=lambda report: _candidate_key(report, baseline)))
    return _rejected(["no safe, provenance-valid candidate improved unseen performance over original_baseline"])


def select_async_sweep_final_winner(
    finalists: Mapping[str, Mapping[str, object]],
    *,
    baseline_report: Mapping[str, object] | None,
    original_12k_checkpoint_digest: str,
    final_matrix_sha256: str,
) -> dict[str, object]:
    """Run the dynamic async-sweep gate without changing the legacy five-arm gate.

    The historical challenge selector above intentionally remains pinned to its
    published five-candidate manifest.  New asynchronous jobs have generated
    experiment IDs, so they use the stricter publication-v2 and sealed
    per-episode receipt contract in ``experiment_winner`` instead.
    """
    from lehome_train.groot.experiment_winner import select_async_final_winner

    return select_async_final_winner(
        finalists,
        baseline_report=baseline_report,
        original_12k_checkpoint_digest=original_12k_checkpoint_digest,
        final_matrix_sha256=final_matrix_sha256,
    )
