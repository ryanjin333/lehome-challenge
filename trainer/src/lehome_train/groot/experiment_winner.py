"""Fail-closed final unseen-80 winner and baseline-reuse gates.

This module is deliberately separate from :mod:`challenge_evaluation`: that
module still owns the historical five-candidate challenge receipt. The async
sweep has dynamic experiment IDs, so its final decision must not inherit the
old ``previous_step_*`` / ``new_step_*`` candidate names.
"""

from __future__ import annotations

import json
import re
from typing import Mapping, Protocol, Sequence

from lehome_train.groot.experiment_publication import parse_checkpoint_publication
from lehome_train.io import canonical_json_sha256


_CATEGORIES = ("top_long", "top_short", "pant_long", "pant_short")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_REPORT_FIELDS = {
    "schema_version", "kind", "candidate_id", "experiment_id",
    "checkpoint_receipt_sha256", "checkpoint_publication", "matrix_sha256",
    "policy_digest", "categories", "overall_successes", "episode_artifacts",
    "safety_failure", "major_seen_regression", "report_sha256", "sidecar",
}
_SIDECAR_FIELDS = {
    "schema_version", "kind", "report_sha256", "artifact_set_sha256",
    "seal_sha256", "readback_verified", "sealed",
}
_EPISODE_FIELDS = {
    "trial_id", "category", "official_success", "artifact_sha256",
    "receipt_sha256", "readback_verified", "sealed",
}


class FinalReportTransport(Protocol):
    """Minimal injected Hub boundary for final-receipt publication.

    The controller never reads a local sidecar as remote evidence.  An adapter
    must upload the exact canonical bytes and then return fresh bytes for the
    same immutable destination.
    """

    def upload_bytes(self, repository: str, path: str, payload: bytes) -> None: ...

    def read_bytes(self, repository: str, path: str) -> bytes: ...


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be SHA-256")
    return value


def _canonical_report_body(report: Mapping[str, object]) -> dict[str, object]:
    """The report bytes sealed by the sidecar, excluding circular seal fields."""
    body = dict(report)
    body.pop("report_sha256", None)
    body.pop("sidecar", None)
    return body


def final_report_sha256(report: Mapping[str, object]) -> str:
    return canonical_json_sha256(_canonical_report_body(report))


def _artifact_set_sha256(episodes: Sequence[Mapping[str, object]]) -> str:
    return canonical_json_sha256({"schema_version": 1, "episode_artifacts": [dict(value) for value in episodes]})


def _seal_sha256(*, report_sha256: str, artifact_set_sha256: str, checkpoint_receipt_sha256: str, matrix_sha256: str) -> str:
    return canonical_json_sha256({
        "schema_version": 1,
        "kind": "lehome_final_unseen80_seal",
        "report_sha256": report_sha256,
        "artifact_set_sha256": artifact_set_sha256,
        "checkpoint_receipt_sha256": checkpoint_receipt_sha256,
        "matrix_sha256": matrix_sha256,
    })


def seal_final_unseen80_report(report: Mapping[str, object]) -> dict[str, object]:
    """Attach a deterministic *local* intent sidecar.

    This pure function has no Hub access, so it deliberately cannot claim a
    readback or a seal.  ``publish_final_unseen80_report`` is the only path
    that turns this intent into a strict final receipt.
    """
    value = dict(report)
    value.pop("report_sha256", None)
    value.pop("sidecar", None)
    digest = final_report_sha256(value)
    episodes = value.get("episode_artifacts")
    if not isinstance(episodes, list):
        raise ValueError("final report episode artifacts are missing")
    checkpoint_receipt = _sha(value.get("checkpoint_receipt_sha256"), "checkpoint receipt")
    matrix_digest = _sha(value.get("matrix_sha256"), "matrix digest")
    artifacts = _artifact_set_sha256(episodes)
    value["report_sha256"] = digest
    value["sidecar"] = {
        "schema_version": 1,
        "kind": "lehome_final_unseen80_sidecar",
        "report_sha256": digest,
        "artifact_set_sha256": artifacts,
        "seal_sha256": _seal_sha256(
            report_sha256=digest,
            artifact_set_sha256=artifacts,
            checkpoint_receipt_sha256=checkpoint_receipt,
            matrix_sha256=matrix_digest,
        ),
        "readback_verified": False,
        "sealed": False,
    }
    return value


def _canonical_report_bytes(report: Mapping[str, object]) -> bytes:
    return json.dumps(dict(report), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8") + b"\n"


def _final_report_destination(repository: object, path: object) -> tuple[str, str]:
    if type(repository) is not str or not repository or any(value.isspace() for value in repository):
        raise ValueError("final report repository is invalid")
    if type(path) is not str or not path or path.startswith("/") or ".." in path.split("/"):
        raise ValueError("final report path is invalid")
    return repository, path


def publish_final_unseen80_report(
    report: Mapping[str, object],
    *,
    transport: FinalReportTransport,
    repository: str,
    path: str,
) -> dict[str, object]:
    """Publish a final report and require exact fresh Hub readback.

    A local sidecar is only a publishing intent.  The returned document claims
    ``readback_verified`` only after the injected transport returns the exact
    bytes uploaded at the explicit immutable destination.
    """
    repo, remote_path = _final_report_destination(repository, path)
    local = dict(report)
    sidecar = local.get("sidecar")
    if not isinstance(sidecar, Mapping):
        raise ValueError("final report local sidecar is invalid")
    if sidecar.get("readback_verified") is not False or sidecar.get("sealed") is not False:
        raise ValueError("final report must be a local unverified intent before publication")
    # Validate every structural binding while explicitly allowing the local
    # intent's two false flags.  This prevents a malformed report from being
    # uploaded merely to obtain a true-looking sidecar.
    validate_final_unseen80_report(local, require_hub_readback=False)
    published = dict(local)
    published_sidecar = dict(sidecar)
    published_sidecar.update({"readback_verified": True, "sealed": True})
    published["sidecar"] = published_sidecar
    validate_final_unseen80_report(published)
    payload = _canonical_report_bytes(published)
    transport.upload_bytes(repo, remote_path, payload)
    observed = transport.read_bytes(repo, remote_path)
    if not isinstance(observed, bytes) or observed != payload:
        raise ValueError("final report Hub readback does not bind published bytes")
    return published


def validate_final_unseen80_report(report: Mapping[str, object], *, require_hub_readback: bool = True) -> dict[str, object]:
    """Require a sealed, publication-v2-bound all-category final receipt."""
    if not isinstance(report, Mapping) or set(report) != _REPORT_FIELDS:
        raise ValueError("final unseen-80 report has unknown or missing field")
    if report.get("schema_version") != 2 or report.get("kind") != "lehome_experiment_final_unseen80":
        raise ValueError("final unseen-80 report schema is invalid")
    if type(report.get("candidate_id")) is not str or _CANDIDATE.fullmatch(report["candidate_id"]) is None:
        raise ValueError("final unseen-80 candidate ID is invalid")
    experiment_id = _sha(report.get("experiment_id"), "experiment ID")
    receipt = _sha(report.get("checkpoint_receipt_sha256"), "checkpoint receipt")
    matrix = _sha(report.get("matrix_sha256"), "matrix digest")
    policy = _sha(report.get("policy_digest"), "policy digest")
    publication_value = report.get("checkpoint_publication")
    if not isinstance(publication_value, Mapping):
        raise ValueError("final unseen-80 publication is missing")
    publication = parse_checkpoint_publication(publication_value)
    if (
        publication.relative_path is None
        or publication.descriptor_relative_path is None
        or publication.experiment_id != experiment_id
        or publication.receipt_sha256 != receipt
        or publication.artifact_sha256 != policy
    ):
        raise ValueError("final unseen-80 publication v2 identity does not bind report")
    categories = report.get("categories")
    if not isinstance(categories, Mapping) or set(categories) != set(_CATEGORIES):
        raise ValueError("final unseen-80 categories are incomplete")
    total = 0
    parsed_categories: dict[str, dict[str, int]] = {}
    for category in _CATEGORIES:
        score = categories[category]
        if not isinstance(score, Mapping) or set(score) != {"successes", "episodes"}:
            raise ValueError("final unseen-80 category score is invalid")
        successes, episodes = score.get("successes"), score.get("episodes")
        if type(successes) is not int or type(episodes) is not int or episodes != 20 or not 0 <= successes <= 20:
            raise ValueError("final unseen-80 must have exactly 20 episodes per category")
        parsed_categories[category] = {"successes": successes, "episodes": episodes}
        total += successes
    if type(report.get("overall_successes")) is not int or report["overall_successes"] != total:
        raise ValueError("final unseen-80 aggregate does not match categories")
    artifacts = report.get("episode_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 80:
        raise ValueError("final unseen-80 requires exactly 80 episode artifacts")
    ids: set[str] = set()
    observed = {name: {"episodes": 0, "successes": 0} for name in _CATEGORIES}
    parsed_artifacts: list[dict[str, object]] = []
    for artifact in artifacts:
        if not isinstance(artifact, Mapping) or set(artifact) != _EPISODE_FIELDS:
            raise ValueError("final unseen-80 episode artifact fields are invalid")
        trial_id = artifact.get("trial_id")
        category, success = artifact.get("category"), artifact.get("official_success")
        if type(trial_id) is not str or not trial_id or trial_id in ids or category not in _CATEGORIES or type(success) is not int or success not in (0, 1):
            raise ValueError("final unseen-80 episode identities are invalid")
        if artifact.get("readback_verified") is not True or artifact.get("sealed") is not True:
            raise ValueError("final unseen-80 episode artifact is not sealed/read-back verified")
        _sha(artifact.get("artifact_sha256"), "episode artifact")
        _sha(artifact.get("receipt_sha256"), "episode receipt")
        ids.add(trial_id)
        observed[str(category)]["episodes"] += 1
        observed[str(category)]["successes"] += success
        parsed_artifacts.append(dict(artifact))
    if any(observed[name]["episodes"] != 20 or observed[name]["successes"] != parsed_categories[name]["successes"] for name in _CATEGORIES):
        raise ValueError("final unseen-80 per-episode evidence does not match category totals")
    if type(report.get("safety_failure")) is not bool or type(report.get("major_seen_regression")) is not bool:
        raise ValueError("final unseen-80 safety or seen regression flag is invalid")
    digest = _sha(report.get("report_sha256"), "report digest")
    if digest != final_report_sha256(report):
        raise ValueError("final unseen-80 report digest mismatch")
    sidecar = report.get("sidecar")
    if not isinstance(sidecar, Mapping) or set(sidecar) != _SIDECAR_FIELDS:
        raise ValueError("final unseen-80 sidecar is invalid")
    if sidecar.get("schema_version") != 1 or sidecar.get("kind") != "lehome_final_unseen80_sidecar" or type(sidecar.get("readback_verified")) is not bool or type(sidecar.get("sealed")) is not bool:
        raise ValueError("final unseen-80 sidecar is not sealed/read-back verified")
    if require_hub_readback and (sidecar.get("readback_verified") is not True or sidecar.get("sealed") is not True):
        raise ValueError("final unseen-80 sidecar is not sealed/read-back verified")
    if not require_hub_readback and (sidecar.get("readback_verified") is not False or sidecar.get("sealed") is not False):
        raise ValueError("final unseen-80 local sidecar must remain unverified")
    artifact_digest = _artifact_set_sha256(parsed_artifacts)
    if sidecar.get("report_sha256") != digest or sidecar.get("artifact_set_sha256") != artifact_digest:
        raise ValueError("final unseen-80 sidecar does not bind report artifacts")
    if sidecar.get("seal_sha256") != _seal_sha256(report_sha256=digest, artifact_set_sha256=artifact_digest, checkpoint_receipt_sha256=receipt, matrix_sha256=matrix):
        raise ValueError("final unseen-80 seal digest mismatch")
    return {
        "experiment_id": experiment_id,
        "candidate_id": str(report["candidate_id"]),
        "checkpoint_receipt_sha256": receipt,
        "checkpoint_publication": dict(publication.canonical),
        "matrix_sha256": matrix,
        "policy_digest": policy,
        "categories": parsed_categories,
        "overall_successes": total,
        "safety_failure": bool(report["safety_failure"]),
        "major_seen_regression": bool(report["major_seen_regression"]),
        "report_sha256": digest,
        "seal_sha256": str(sidecar["seal_sha256"]),
    }


def baseline_reuse_decision(report: Mapping[str, object] | None, *, original_12k_checkpoint_digest: str, final_matrix_sha256: str) -> str:
    """Return an explicit no-GPU-reuse decision for the original 12K result."""
    try:
        expected_checkpoint = _sha(original_12k_checkpoint_digest, "original 12K checkpoint digest")
        expected_matrix = _sha(final_matrix_sha256, "final matrix digest")
        if report is None:
            raise ValueError("baseline report is absent")
        parsed = validate_final_unseen80_report(report)
        if parsed["matrix_sha256"] != expected_matrix or parsed["policy_digest"] != expected_checkpoint:
            raise ValueError("baseline identity does not match original 12K or final matrix")
    except ValueError:
        return "baseline_evaluation_required"
    return "baseline_reusable"


def verify_reusable_baseline(report: Mapping[str, object], *, checkpoint_digest: str, matrix_digest: str) -> bool:
    """Compatibility boolean wrapper around the strict baseline decision."""
    return baseline_reuse_decision(report, original_12k_checkpoint_digest=checkpoint_digest, final_matrix_sha256=matrix_digest) == "baseline_reusable"


def winner_gate(report: Mapping[str, object]) -> str:
    """Apply the 70% and per-category final gate without legacy candidate keys."""
    try:
        if report.get("kind") == "lehome_experiment_final_unseen80":
            parsed = validate_final_unseen80_report(report)
            categories = parsed["categories"]
            successes = int(parsed["overall_successes"])
            safety = bool(parsed["safety_failure"])
            seen = bool(parsed["major_seen_regression"])
            counts = {name: int(categories[name]["successes"]) for name in _CATEGORIES}
        else:
            # Retain the tiny legacy pure-gate surface used by existing callers.
            categories = report.get("category_successes")
            if not isinstance(categories, Mapping) or set(categories) != set(_CATEGORIES):
                return "rejected"
            successes = report.get("overall_successes")
            safety, seen = report.get("safety_regression"), report.get("seen_regression")
            counts = {name: categories[name] for name in _CATEGORIES}
        if safety or seen or type(successes) is not int or successes < 56 or any(type(counts[name]) is not int or counts[name] < 12 for name in _CATEGORIES):
            return "rejected"
    except ValueError:
        return "rejected"
    return "winner"


def select_final_winner(reports: Mapping[str, Mapping[str, object]]) -> str | None:
    """Pick a dynamic finalist; old 1K/2K candidate names are never assumed."""
    candidates: list[tuple[str, int, int]] = []
    for name, report in reports.items():
        if type(name) is not str or winner_gate(report) != "winner":
            continue
        if report.get("kind") == "lehome_experiment_final_unseen80":
            parsed = validate_final_unseen80_report(report)
            if parsed["candidate_id"] != name:
                raise ValueError("finalist key does not match report candidate ID")
            floor = min(int(parsed["categories"][category]["successes"]) for category in _CATEGORIES)
            candidates.append((name, int(parsed["overall_successes"]), floor))
        else:
            categories = report["category_successes"]
            assert isinstance(categories, Mapping)
            candidates.append((name, int(report["overall_successes"]), min(int(categories[category]) for category in _CATEGORIES)))
    return max(candidates, key=lambda item: (item[1], item[2], item[0]))[0] if candidates else None


def select_async_final_winner(finalists: Mapping[str, Mapping[str, object]], *, baseline_report: Mapping[str, object] | None, original_12k_checkpoint_digest: str, final_matrix_sha256: str) -> dict[str, object]:
    """Select only after strict baseline reuse and dynamic-finalist validation."""
    baseline = baseline_reuse_decision(
        baseline_report,
        original_12k_checkpoint_digest=original_12k_checkpoint_digest,
        final_matrix_sha256=final_matrix_sha256,
    )
    if baseline != "baseline_reusable":
        return {"decision": baseline}
    try:
        expected_matrix = _sha(final_matrix_sha256, "final matrix digest")
        for candidate_id, report in finalists.items():
            if type(candidate_id) is not str:
                raise ValueError("finalist ID is invalid")
            parsed = validate_final_unseen80_report(report)
            if parsed["candidate_id"] != candidate_id or parsed["matrix_sha256"] != expected_matrix:
                raise ValueError("finalist receipt does not bind its ID or final matrix")
    except ValueError:
        return {"decision": "invalid_finalist_receipt"}
    winner = select_final_winner(finalists)
    if winner is None:
        return {"decision": "no_finalist_passed"}
    parsed = validate_final_unseen80_report(finalists[winner])
    return {
        "decision": "winner",
        "candidate_id": winner,
        "experiment_id": parsed["experiment_id"],
        "checkpoint_receipt_sha256": parsed["checkpoint_receipt_sha256"],
        "report_sha256": parsed["report_sha256"],
        "seal_sha256": parsed["seal_sha256"],
    }
