from __future__ import annotations

import hashlib
import json
import math
from collections import Counter

import numpy as np
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
        "top_long": [f"Top_Long_Seen_{index}" for index in range(10)],
        "top_short": [f"Top_Short_Seen_{index}" for index in range(10)],
        "pant_long": [f"Pant_Long_Seen_{index}" for index in range(10)],
        "pant_short": [f"Pant_Short_Seen_{index}" for index in range(10)],
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
    assert {row["campaign_kind"] for row in rows} == {"simple_curriculum_source_v1"}
    assert {row["logical_stage"] for row in rows} == {"calibration"}
    assert {row["release_stage"] for row in rows} == {"seen"}
    assert {row["strategy"] for row in rows} == {"canonical"}
    assert all(row["source_seed"] == row["seed"] for row in rows)


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
    rows = build_curriculum_rows(
        report, calibration_rows=calibration, count=600, rng_seed=1234,
        policy_identity=POLICY_IDENTITY, catalog=_catalog(),
    )

    assert len(rows) == 600
    assert rows == build_curriculum_rows(
        report, calibration_rows=calibration, count=600, rng_seed=1234,
        policy_identity=POLICY_IDENTITY, catalog=_catalog(),
    )
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
    assert {row["campaign_kind"] for row in rows} == {"simple_curriculum_source_v1"}
    assert all(row["source_seed"] == row["seed"] for row in rows)
    for row in [*calibration, *rows]:
        assert 0 <= row["seed"] < 2**32
        np.random.RandomState(row["seed"])


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


def test_catalog_accepts_seen_zero_and_rejects_seen_ten() -> None:
    assert build_calibration_rows(_catalog(), seed_base=1)[0]["garment"] == "Top_Long_Seen_0"
    invalid = _catalog()
    invalid["top_long"][-1] = "Top_Long_Seen_10"

    with pytest.raises(ValueError, match="unapproved"):
        build_calibration_rows(invalid, seed_base=1)


def test_curriculum_requires_externally_trusted_policy_and_catalog() -> None:
    catalog = _catalog()
    calibration = build_calibration_rows(catalog, seed_base=90_000)
    forged_policy = {**POLICY_IDENTITY, "policy_artifact_sha256": "c" * 64}
    forged_report = _report(calibration)
    forged_report["policy_identity"] = forged_policy
    forged_report["authenticated_policy_identity"] = forged_policy
    forged_catalog = _catalog()
    forged_catalog["top_long"] = list(reversed(forged_catalog["top_long"]))
    forged_calibration = build_calibration_rows(forged_catalog, seed_base=90_000)
    forged_catalog_report = _report(forged_calibration, catalog=forged_catalog)

    with pytest.raises(ValueError, match="policy identity"):
        build_curriculum_rows(
            forged_report, calibration_rows=calibration, count=600, rng_seed=3,
            policy_identity=POLICY_IDENTITY, catalog=catalog,
        )
    with pytest.raises(ValueError, match="catalog"):
        build_curriculum_rows(
            forged_catalog_report, calibration_rows=forged_calibration, count=600, rng_seed=3,
            policy_identity=POLICY_IDENTITY, catalog=catalog,
        )


class _ScriptedRng:
    def __init__(self, values: list[int], *, pick_last: bool = False) -> None:
        self.values = iter(values)
        self.pick_last = pick_last
        self.choice_calls: list[tuple[tuple[object, ...], tuple[float, ...]]] = []

    def choices(self, population, *, weights, k):  # type: ignore[no-untyped-def]
        assert k == 1
        self.choice_calls.append((tuple(population), tuple(weights)))
        return [population[-1] if self.pick_last else population[0]]

    def randrange(self, _start, _stop):  # type: ignore[no-untyped-def]
        return next(self.values)


def test_curriculum_redraws_duplicate_seed_and_selects_category_then_garment() -> None:
    catalog = _catalog()
    calibration = build_calibration_rows(catalog, seed_base=90_000)
    report = _report(calibration)
    scripted = _ScriptedRng([0, 0, 1], pick_last=True)

    rows = build_curriculum_rows(
        report, calibration_rows=calibration, count=2, rng_seed=1,
        policy_identity=POLICY_IDENTITY, catalog=catalog, rng_factory=lambda _seed: scripted,
    )

    assert [row["seed"] for row in rows] == [0, 1]
    assert len({row["seed"] for row in rows}) == 2
    assert scripted.choice_calls[0][0] == ("top_long", "top_short", "pant_long", "pant_short")
    assert scripted.choice_calls[1][0] == tuple(catalog["pant_short"])
    assert scripted.choice_calls[0][1] != scripted.choice_calls[1][1]


def test_calibration_seed_range_includes_boundaries_and_rejects_overflow() -> None:
    catalog = _catalog()
    lower_rows = build_calibration_rows(catalog, seed_base=0)
    rows = build_calibration_rows(catalog, seed_base=2**32 - 400)

    assert lower_rows[0]["seed"] == 0
    assert lower_rows[-1]["seed"] == 399
    assert rows[0]["seed"] == 2**32 - 400
    assert rows[-1]["seed"] == 2**32 - 1
    assert all(0 <= row["seed"] < 2**32 for row in rows)
    for row in rows:
        np.random.RandomState(row["seed"])
    with pytest.raises(ValueError, match="seed_base"):
        build_calibration_rows(catalog, seed_base=-1)
    with pytest.raises(ValueError, match="seed_base"):
        build_calibration_rows(catalog, seed_base=2**32 - 399)
