from __future__ import annotations

import hashlib
import json
import math
from collections import Counter

import pytest

from lehome.flywheel.simple_curriculum import (
    SIGMA,
    build_calibration_rows,
    build_curriculum_rows,
    garment_weight,
    type_weight,
    validate_calibration_report,
)


POLICY_IDENTITY = {
    "policy_repo": "ryanjin333/lehome-groot-n17-models",
    "policy_revision": "a" * 40,
    "policy_step": 12000,
    "policy_artifact_sha256": "b" * 64,
}


def _catalog() -> dict[str, list[str]]:
    return {
        "top_long": [f"Top_Long_Seen_{index}" for index in range(1, 11)],
        "top_short": [f"Top_Short_Seen_{index}" for index in range(1, 11)],
        "pant_long": [f"Pant_Long_Seen_{index}" for index in range(1, 11)],
        "pant_short": [f"Pant_Short_Seen_{index}" for index in range(1, 11)],
    }


def _matrix_sha(rows: object) -> str:
    payload = (json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def _report(rows: list[dict[str, object]], *, catalog: object | None = None) -> dict[str, object]:
    outcomes = [
        {"attempt_id": row["attempt_id"], "trial_id": row["trial_id"], "success": index % 3 != 0}
        for index, row in enumerate(rows)
    ]
    return {
        "schema_version": 1,
        "kind": "lehome_simple_curriculum_calibration_report_v1",
        "authenticated": True,
        "calibration_matrix_sha256": _matrix_sha(rows),
        "policy_identity": POLICY_IDENTITY,
        "authenticated_policy_identity": POLICY_IDENTITY,
        "provenance": {"simulator_device": "cpu", "policy_device": "cuda:0"},
        "catalog": catalog if catalog is not None else _catalog(),
        "outcomes": outcomes,
    }


def test_calibration_builder_is_uniform_and_canonical() -> None:
    catalog = _catalog()

    rows = build_calibration_rows(catalog, seed_base=90_000)

    assert len(rows) == 400
    assert rows == build_calibration_rows(catalog, seed_base=90_000)
    assert set(Counter(row["garment"] for row in rows).values()) == {10}
    assert set(Counter(row["category"] for row in rows).values()) == {100}
    assert set(Counter(row["category"] for row in rows[:100]).values()) == {25}
    for field in ("attempt_id", "trial_id", "seed"):
        assert len({row[field] for row in rows}) == 400
    assert {
        "campaign_kind", "logical_stage", "attempt_id", "trial_id", "garment", "garment_name",
        "category", "release_stage", "seed", "source_seed", "strategy",
    } <= set(rows[0])
    assert {row["campaign_kind"] for row in rows} == {"calibration"}
    assert {row["logical_stage"] for row in rows} == {"calibration"}
    assert {row["release_stage"] for row in rows} == {"seen"}
    assert {row["strategy"] for row in rows} == {"canonical"}


def test_weights_follow_the_approved_formula() -> None:
    assert SIGMA == 0.233
    assert type_weight(0.0) == 1.0
    assert type_weight(1.0) == 0.05
    assert garment_weight(0.5) == pytest.approx(1.0)
    assert garment_weight(0.0) == pytest.approx(math.exp(-((0.0 - 0.5) ** 2) / (2 * SIGMA**2)))
    assert garment_weight(1.0) == garment_weight(0.0)
    assert garment_weight(-1.0) == 0.02


def test_curriculum_builder_authenticates_and_samples_deterministically() -> None:
    calibration = build_calibration_rows(_catalog(), seed_base=90_000)
    report = _report(calibration)

    validated = validate_calibration_report(
        report,
        matrix_sha256=_matrix_sha(calibration),
        policy_identity=POLICY_IDENTITY,
        catalog=_catalog(),
    )
    rows = build_curriculum_rows(report, calibration_rows=calibration, count=600, rng_seed=1234)

    assert len(rows) == 600
    assert rows == build_curriculum_rows(report, calibration_rows=calibration, count=600, rng_seed=1234)
    assert len({row["attempt_id"] for row in rows}) == 600
    assert len({row["trial_id"] for row in rows}) == 600
    assert len({row["seed"] for row in rows}) == 600
    assert {row["attempt_id"] for row in rows}.isdisjoint(row["attempt_id"] for row in calibration)
    assert {row["trial_id"] for row in rows}.isdisjoint(row["trial_id"] for row in calibration)
    assert {row["seed"] for row in rows}.isdisjoint(row["seed"] for row in calibration)
    assert validated["matrix_sha256"] == _matrix_sha(calibration)
    assert all(row["builder_rng_seed"] == 1234 for row in rows)
    assert all(row["calibration_matrix_sha256"] == _matrix_sha(calibration) for row in rows)
    assert all(row["sampled_category_weight"] > 0 for row in rows)
    assert all(row["sampled_garment_weight"] > 0 for row in rows)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda report: report.update(policy_identity={**POLICY_IDENTITY, "policy_step": 500}), "policy identity"),
        (lambda report: report["provenance"].update(policy_device="cpu"), "provenance"),
        (lambda report: report.update(calibration_matrix_sha256="0" * 64), "matrix SHA-256"),
        (lambda report: report.update(catalog={"top_long": _catalog()["top_long"]}), "catalog"),
        (lambda report: report.update(outcomes=report["outcomes"][:-1]), "400"),
    ],
)
def test_calibration_report_fails_closed_on_bad_authentication(mutator, message: str) -> None:
    calibration = build_calibration_rows(_catalog(), seed_base=90_000)
    report = _report(calibration)
    mutator(report)

    with pytest.raises(ValueError, match=message):
        validate_calibration_report(
            report,
            matrix_sha256=_matrix_sha(calibration),
            policy_identity=POLICY_IDENTITY,
            catalog=_catalog(),
        )
