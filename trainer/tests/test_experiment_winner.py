"""Final unseen-80 winner gate tests."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


_CATEGORIES = ("top_long", "top_short", "pant_long", "pant_short")


class _FakeHub:
    def __init__(self) -> None:
        self.payload: bytes | None = None

    def upload_bytes(self, repository: str, path: str, payload: bytes) -> None:
        assert repository == "owner/final-reports"
        assert path == "finals/report.json"
        self.payload = payload

    def read_bytes(self, repository: str, path: str) -> bytes:
        assert repository == "owner/final-reports"
        assert path == "finals/report.json"
        assert self.payload is not None
        return self.payload


def _publication(*, experiment_id: str, receipt: str, policy: str) -> dict[str, object]:
    return {
        "schema_version": 2,
        "experiment_id": experiment_id,
        "job_digest": experiment_id,
        "target_step": 500,
        "repository": "owner/checkpoints",
        "immutable_revision": "b" * 40,
        "remote_prefix": "experiments/candidate-a",
        "artifact_sha256": policy,
        "receipt_sha256": receipt,
        "readback_verified": True,
        "relative_path": "checkpoint.tar",
        "artifact_byte_size": 1,
        "descriptor_relative_path": "checkpoint.json",
        "descriptor_sha256": "d" * 64,
        "descriptor_byte_size": 1,
    }


def _report(
    *,
    candidate: str = "candidate-a",
    experiment_id: str = "a" * 64,
    receipt: str = "c" * 64,
    policy: str = "e" * 64,
    matrix: str = "f" * 64,
    successes: dict[str, int] | None = None,
) -> dict[str, object]:
    from lehome_train.groot.experiment_winner import seal_final_unseen80_report

    per_category = successes or {"top_long": 12, "top_short": 12, "pant_long": 12, "pant_short": 20}
    artifacts: list[dict[str, object]] = []
    for category in _CATEGORIES:
        for index in range(20):
            artifacts.append({
                "trial_id": f"{category}-{index}",
                "category": category,
                "official_success": int(index < per_category[category]),
                "artifact_sha256": "1" * 64,
                "receipt_sha256": "2" * 64,
                "readback_verified": True,
                "sealed": True,
            })
    local = seal_final_unseen80_report({
        "schema_version": 2,
        "kind": "lehome_experiment_final_unseen80",
        "candidate_id": candidate,
        "experiment_id": experiment_id,
        "checkpoint_receipt_sha256": receipt,
        "checkpoint_publication": _publication(experiment_id=experiment_id, receipt=receipt, policy=policy),
        "matrix_sha256": matrix,
        "policy_digest": policy,
        "categories": {name: {"successes": value, "episodes": 20} for name, value in per_category.items()},
        "overall_successes": sum(per_category.values()),
        "episode_artifacts": artifacts,
        "safety_failure": False,
        "major_seen_regression": False,
    })
    from lehome_train.groot.experiment_winner import publish_final_unseen80_report
    return publish_final_unseen80_report(
        local,
        transport=_FakeHub(),
        repository="owner/final-reports",
        path="finals/report.json",
    )


def test_winner_gate_requires_all_category_and_safety_thresholds() -> None:
    from lehome_train.groot.experiment_winner import winner_gate

    assert winner_gate({"overall_successes": 56, "category_successes": {"top_long": 12, "top_short": 12, "pant_long": 12, "pant_short": 20}, "safety_regression": False, "seen_regression": False}) == "winner"
    assert winner_gate({"overall_successes": 56, "category_successes": {"top_long": 11, "top_short": 12, "pant_long": 12, "pant_short": 21}, "safety_regression": False, "seen_regression": False}) == "rejected"


def test_final_gate_requires_exactly_80_unique_sealed_readback_episodes_and_v2_publication() -> None:
    from lehome_train.groot.experiment_winner import validate_final_unseen80_report

    report = _report()
    parsed = validate_final_unseen80_report(report)
    assert parsed["overall_successes"] == 56
    assert parsed["checkpoint_publication"]["schema_version"] == 2

    duplicate = deepcopy(report)
    duplicate["episode_artifacts"][1]["trial_id"] = duplicate["episode_artifacts"][0]["trial_id"]  # type: ignore[index]
    with pytest.raises(ValueError, match="identities"):
        validate_final_unseen80_report(duplicate)

    missing = deepcopy(report)
    missing["episode_artifacts"] = missing["episode_artifacts"][:-1]  # type: ignore[index]
    with pytest.raises(ValueError, match="exactly 80"):
        validate_final_unseen80_report(missing)


def test_final_gate_rejects_tampered_report_or_sidecar_digest() -> None:
    from lehome_train.groot.experiment_winner import validate_final_unseen80_report

    report = _report()
    report["categories"]["top_long"]["successes"] = 13  # type: ignore[index]
    with pytest.raises(ValueError, match="aggregate|evidence|digest"):
        validate_final_unseen80_report(report)

    report = _report()
    report["sidecar"]["seal_sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(ValueError, match="seal digest"):
        validate_final_unseen80_report(report)


def test_final_report_requires_injected_hub_readback_before_claiming_a_seal() -> None:
    """A local JSON sidecar alone is not evidence that Hugging Face has it."""
    from lehome_train.groot.experiment_winner import (
        publish_final_unseen80_report,
        seal_final_unseen80_report,
        validate_final_unseen80_report,
    )

    class FakeHub:
        def __init__(self, *, tamper: bool = False) -> None:
            self.payload: bytes | None = None
            self.tamper = tamper

        def upload_bytes(self, repository: str, path: str, payload: bytes) -> None:
            assert repository == "owner/final-reports" and path == "finals/candidate-a.json"
            self.payload = payload

        def read_bytes(self, repository: str, path: str) -> bytes:
            assert repository == "owner/final-reports" and path == "finals/candidate-a.json"
            assert self.payload is not None
            return b"tampered" if self.tamper else self.payload

    local = seal_final_unseen80_report(_report())
    assert local["sidecar"]["readback_verified"] is False
    with pytest.raises(ValueError, match="sealed/read-back"):
        validate_final_unseen80_report(local)

    published = publish_final_unseen80_report(
        local,
        transport=FakeHub(),
        repository="owner/final-reports",
        path="finals/candidate-a.json",
    )
    assert published["sidecar"]["readback_verified"] is True
    assert published["sidecar"]["sealed"] is True
    assert validate_final_unseen80_report(published)["candidate_id"] == "candidate-a"

    with pytest.raises(ValueError, match="readback"):
        publish_final_unseen80_report(
            local,
            transport=FakeHub(tamper=True),
            repository="owner/final-reports",
            path="finals/candidate-a.json",
        )


def test_final_report_writer_publishes_then_writes_only_readback_verified_bytes(tmp_path) -> None:
    from lehome_train.groot.experiment_winner import seal_final_unseen80_report

    source = Path(__file__).parents[2] / "scripts" / "summarize_groot_persistent_evaluation.py"
    spec = importlib.util.spec_from_file_location("persistent_evaluation_summary_under_test", source)
    assert spec and spec.loader
    summary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summary)

    class FakeHub:
        payload: bytes | None = None

        def upload_bytes(self, repository: str, path: str, payload: bytes) -> None:
            assert repository == "owner/final-reports" and path == "finals/report.json"
            self.payload = payload

        def read_bytes(self, repository: str, path: str) -> bytes:
            assert self.payload is not None
            return self.payload

    output = tmp_path / "final.json"
    result = summary.write_final_unseen80_report(
        output,
        seal_final_unseen80_report(_report()),
        transport=FakeHub(),
        repository="owner/final-reports",
        remote_path="finals/report.json",
    )
    assert result == output
    document = __import__("json").loads(output.read_text())
    assert document["sidecar"]["readback_verified"] is True


def test_baseline_reuse_requires_exact_original_checkpoint_matrix_and_sealed_artifacts() -> None:
    from lehome_train.groot.experiment_winner import baseline_reuse_decision

    baseline = _report(candidate="original-12k")
    assert baseline_reuse_decision(baseline, original_12k_checkpoint_digest="e" * 64, final_matrix_sha256="f" * 64) == "baseline_reusable"
    assert baseline_reuse_decision(baseline, original_12k_checkpoint_digest="0" * 64, final_matrix_sha256="f" * 64) == "baseline_evaluation_required"
    assert baseline_reuse_decision(baseline, original_12k_checkpoint_digest="e" * 64, final_matrix_sha256="0" * 64) == "baseline_evaluation_required"


def test_dynamic_finalists_skip_historical_1k_2k_keys_and_require_baseline() -> None:
    from lehome_train.groot.experiment_winner import select_async_final_winner

    baseline = _report(candidate="original-12k")
    candidate = _report(candidate="experiment-7", experiment_id="9" * 64, receipt="8" * 64, policy="7" * 64, successes={"top_long": 13, "top_short": 13, "pant_long": 13, "pant_short": 20})
    result = select_async_final_winner(
        {"experiment-7": candidate},
        baseline_report=baseline,
        original_12k_checkpoint_digest="e" * 64,
        final_matrix_sha256="f" * 64,
    )
    assert result["decision"] == "winner"
    assert result["candidate_id"] == "experiment-7"
    assert select_async_final_winner({"experiment-7": candidate}, baseline_report=None, original_12k_checkpoint_digest="e" * 64, final_matrix_sha256="f" * 64) == {"decision": "baseline_evaluation_required"}
    assert select_async_final_winner({"old-new-step-2k": {"overall_successes": 80, "category_successes": {"top_long": 20, "top_short": 20, "pant_long": 20, "pant_short": 20}, "safety_regression": False, "seen_regression": False}}, baseline_report=baseline, original_12k_checkpoint_digest="e" * 64, final_matrix_sha256="f" * 64) == {"decision": "invalid_finalist_receipt"}
