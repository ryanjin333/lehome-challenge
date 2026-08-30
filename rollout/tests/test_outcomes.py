from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from b1k_rollout.contracts import RolloutContract
from b1k_rollout.outcomes import Outcome, classify_outcome, classify_outcome_file
from b1k_rollout.provenance import ProvenanceAuthenticator
from b1k_rollout.task_manifest import load_task_manifest


FIXTURES = Path(__file__).parent / "fixtures"
MANIFEST = load_task_manifest(Path(__file__).parents[1] / "task-manifest.json")
AUTH = ProvenanceAuthenticator(b"o" * 32, issuer="outcomes-test")


def _evidence(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _contract() -> RolloutContract:
    return RolloutContract.from_mapping(_evidence("closed-success.json")["contract"])


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        ("closed-success.json", Outcome.SUCCESS),
        ("closed-failure.json", Outcome.FAILURE),
        ("incomplete.json", Outcome.QUARANTINE),
    ],
)
def test_only_official_closed_boolean_terminal_evidence_is_classified(
    fixture: str, expected: Outcome
) -> None:
    classified = classify_outcome(_evidence(fixture), task_manifest=MANIFEST)

    assert classified.outcome is expected
    assert classified.raw_evidence == _evidence(fixture)


def test_q_score_cannot_relabel_official_failure_as_success() -> None:
    classified = classify_outcome(_evidence("closed-failure.json"), task_manifest=MANIFEST)

    assert classified.outcome is Outcome.FAILURE
    assert classified.final_q_scores == {"final": 0.99}


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda evidence: evidence["q_score"].__setitem__("final", -0.1), "q_score.final"),
        (lambda evidence: evidence["q_score"].__setitem__("final", 1.1), "q_score.final"),
        (lambda evidence: evidence["q_score"].__setitem__("final", 0.5), "success"),
        (lambda evidence: evidence["time"].__setitem__("simulator_steps", 11), "simulator steps"),
        (lambda evidence: evidence["time"].__setitem__("simulator_steps", 0), "simulator steps"),
        (lambda evidence: evidence["time"].__setitem__("simulator_time", -1.0), "simulator_time"),
        (lambda evidence: evidence["agent_distance"].__setitem__("base", -1.0), "agent_distance.base"),
        (lambda evidence: evidence["time"].__setitem__("normalized_time", -0.1), "normalized_time"),
        (lambda evidence: evidence["time"].__setitem__("normalized_time", float("-inf")), "normalized_time"),
        (lambda evidence: evidence["time"].__setitem__("normalized_time", float("nan")), "normalized_time"),
        (lambda evidence: evidence["normalized_agent_distance"].__setitem__("left", -0.1), "normalized_agent_distance.left"),
        (lambda evidence: evidence["normalized_agent_distance"].__setitem__("left", float("-inf")), "normalized_agent_distance.left"),
        (lambda evidence: evidence["normalized_agent_distance"].__setitem__("left", float("nan")), "normalized_agent_distance.left"),
    ],
)
def test_pinned_metric_invariants_fail_closed_before_terminal_classification(
    mutate: object, reason: str
) -> None:
    evidence = _evidence("closed-success.json")
    mutate(evidence)  # type: ignore[operator]

    classified = classify_outcome(evidence, task_manifest=MANIFEST)

    assert classified.outcome is Outcome.QUARANTINE
    assert reason.casefold() in classified.reason.casefold()


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda evidence: evidence["q_score"].__setitem__("final", 10**1000), Outcome.QUARANTINE),
        (lambda evidence: evidence["q_score"].__setitem__("final", -(10**1000)), Outcome.QUARANTINE),
        (lambda evidence: evidence["time"].__setitem__("simulator_time", 10**1000), Outcome.SUCCESS),
        (lambda evidence: evidence["time"].__setitem__("simulator_time", -(10**1000)), Outcome.QUARANTINE),
        (lambda evidence: evidence["agent_distance"].__setitem__("base", 10**1000), Outcome.SUCCESS),
        (lambda evidence: evidence["agent_distance"].__setitem__("base", -(10**1000)), Outcome.QUARANTINE),
        (lambda evidence: evidence["time"].__setitem__("normalized_time", 10**1000), Outcome.SUCCESS),
        (lambda evidence: evidence["time"].__setitem__("normalized_time", -(10**1000)), Outcome.QUARANTINE),
    ],
)
def test_oversized_json_integers_never_escape_terminal_classification(
    mutate: object, expected: Outcome
) -> None:
    evidence = _evidence("closed-success.json")
    mutate(evidence)  # type: ignore[operator]

    classified = classify_outcome(evidence, task_manifest=MANIFEST)

    assert classified.outcome is expected


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda evidence: evidence.pop("success"), "success"),
        (lambda evidence: evidence.__setitem__("success", 1), "success"),
        (lambda evidence: evidence.__setitem__("completed", False), "completion"),
        (lambda evidence: evidence.__setitem__("steps", 0), "steps"),
        (lambda evidence: evidence.__setitem__("task", "not_in_manifest"), "task"),
        (lambda evidence: evidence.__setitem__("instance_id", -1), "instance"),
        (lambda evidence: evidence.__setitem__("instance_id", 302), "instance"),
        (lambda evidence: evidence.__setitem__("mode", "train"), "mode"),
        (lambda evidence: evidence.pop("q_score"), "Q-score"),
        (lambda evidence: evidence.__setitem__("policy_server_crash", True), "crash"),
        (lambda evidence: evidence.__setitem__("simulator_crash", True), "crash"),
        (lambda evidence: evidence.update(timeout=True, completed=False), "timeout"),
        (lambda evidence: evidence["artifact_hashes"].__setitem__("../escape", "a" * 64), "artifact"),
        (lambda evidence: evidence["contract"].__setitem__("task_manifest_sha256", "0" * 64), "manifest"),
        (lambda evidence: evidence.__setitem__("rollout_id", -1), "rollout"),
        (lambda evidence: evidence.__setitem__("rollout_id", "wrapper-rollout"), "rollout"),
        (lambda evidence: evidence.__setitem__("task", "lehome_task"), "LeHome"),
    ],
)
def test_missing_invalid_or_crashed_evidence_fails_closed_to_quarantine(
    mutate: object, reason: str
) -> None:
    evidence = _evidence("closed-success.json")
    mutate(evidence)  # type: ignore[operator]

    classified = classify_outcome(evidence, task_manifest=MANIFEST)

    assert classified.outcome is Outcome.QUARANTINE
    assert reason.casefold() in classified.reason.casefold()
    assert classified.raw_evidence == evidence


def test_classification_requires_the_complete_contract_identity_and_hashes() -> None:
    evidence = _evidence("closed-success.json")
    evidence["contract"].pop("image_digest")

    classified = classify_outcome(evidence, task_manifest=MANIFEST)

    assert classified.outcome is Outcome.QUARANTINE
    assert "contract" in classified.reason.casefold()


def test_malformed_json_is_quarantined_without_discarding_raw_bytes() -> None:
    classified = classify_outcome(b'{"broken":', task_manifest=MANIFEST)

    assert classified.outcome is Outcome.QUARANTINE
    assert classified.raw_evidence == b'{"broken":'
    assert "JSON" in classified.reason


def test_incomplete_file_is_quarantined_even_when_its_contents_are_terminal(
    tmp_path: Path,
) -> None:
    path = tmp_path / "episode.json.incomplete"
    path.write_text((FIXTURES / "closed-success.json").read_text(encoding="utf-8"), encoding="utf-8")

    classified = classify_outcome_file(
        path,
        task_manifest=MANIFEST,
        episode_key="incomplete-file",
        contract=_contract(),
        authenticator=AUTH,
    )

    assert classified.outcome is Outcome.QUARANTINE
    assert "incomplete" in classified.reason.casefold()
    assert isinstance(classified.raw_evidence, bytes)


def test_embedded_lehome_material_is_quarantined_even_outside_the_task_name() -> None:
    evidence = _evidence("closed-success.json")
    evidence["q_score"] = {"note": "lehome provenance"}

    classified = classify_outcome(evidence, task_manifest=MANIFEST)

    assert classified.outcome is Outcome.QUARANTINE
    assert "LeHome" in classified.reason


def test_official_rollout_id_must_be_a_nonnegative_integer() -> None:
    evidence = _evidence("closed-success.json")
    evidence["rollout_id"] = "wrapper-rollout-001"

    classified = classify_outcome(evidence, task_manifest=MANIFEST)

    assert classified.outcome is Outcome.QUARANTINE


def test_raw_evidence_hashes_exact_valid_bytes_not_its_reformatted_json() -> None:
    raw = (FIXTURES / "closed-success.json").read_bytes()
    classified = classify_outcome(raw, task_manifest=MANIFEST)

    import hashlib

    assert classified.outcome is Outcome.SUCCESS
    assert classified.raw_evidence == raw
    assert classified.raw_evidence_sha256 == hashlib.sha256(raw).hexdigest()


def test_official_metric_payloads_are_retained_with_their_full_structure() -> None:
    classified = classify_outcome(_evidence("closed-success.json"), task_manifest=MANIFEST)

    assert classified.evaluator_metrics == {
        "agent_distance": {"base": 1.0, "left": 2.0, "right": 3.0},
        "normalized_agent_distance": {"base": 0.1, "left": 0.2, "right": 0.3},
        "q_score": {"final": 1.0},
        "time": {"normalized_time": 0.5, "simulator_steps": 12, "simulator_time": 1.25},
    }


def test_byte_evidence_with_upstream_infinite_normalized_metric_is_preserved_safely() -> None:
    evidence = _evidence("closed-success.json")
    evidence["time"]["normalized_time"] = float("inf")
    raw = json.dumps(evidence).encode("utf-8")

    classified = classify_outcome(raw, task_manifest=MANIFEST)

    assert classified.outcome is Outcome.SUCCESS
    assert classified.raw_evidence == raw


def test_invalid_extracted_ids_are_normalized_before_quarantine_persistence() -> None:
    evidence = _evidence("closed-success.json")
    evidence["episode_id"] = "../escaped"
    evidence["rollout_id"] = -1

    classified = classify_outcome(evidence, task_manifest=MANIFEST)

    assert classified.outcome is Outcome.QUARANTINE
    assert classified.episode_id is None
    assert classified.rollout_id is None


def test_file_classifier_rejects_a_symlink_without_reading_through_it(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_bytes((FIXTURES / "closed-success.json").read_bytes())
    link = tmp_path / "evidence.json"
    link.symlink_to(target)

    classified = classify_outcome_file(
        link,
        task_manifest=MANIFEST,
        episode_key="symlink-file",
        contract=_contract(),
        authenticator=AUTH,
    )

    assert classified.outcome is Outcome.QUARANTINE
    assert classified.raw_evidence == b""
    assert "symlink" in classified.reason
    assert classified.provenance_attestation is not None
