"""Deterministic, authenticated matrices for the simple success curriculum."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
import math
import random
import re
from typing import Callable, Mapping, Sequence


SIGMA = 0.233
_CATEGORIES = ("top_long", "top_short", "pant_long", "pant_short")
_GARMENT_PATTERNS = {
    "top_long": re.compile(r"^Top_Long_Seen_[0-9]$"),
    "top_short": re.compile(r"^Top_Short_Seen_[0-9]$"),
    "pant_long": re.compile(r"^Pant_Long_Seen_[0-9]$"),
    "pant_short": re.compile(r"^Pant_Short_Seen_[0-9]$"),
}
_MAX_SEED = 2**32 - 1


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _matrix_sha256(rows: object) -> str:
    return sha256(_canonical_bytes(rows)).hexdigest()


def _catalog(catalog: object) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not isinstance(catalog, Mapping) or set(catalog) != set(_CATEGORIES):
        raise ValueError("catalog must contain exactly the four approved categories")
    normalized: list[tuple[str, tuple[str, ...]]] = []
    all_garments: set[str] = set()
    for category in _CATEGORIES:
        garments = catalog[category]
        if not isinstance(garments, Sequence) or isinstance(garments, (str, bytes)) or len(garments) != 10:
            raise ValueError("catalog must contain exactly ten garments per category")
        names = tuple(garments)
        if any(not isinstance(name, str) or _GARMENT_PATTERNS[category].fullmatch(name) is None for name in names):
            raise ValueError("catalog has an unapproved seen garment")
        if len(set(names)) != 10 or all_garments.intersection(names):
            raise ValueError("catalog garments must be globally unique")
        all_garments.update(names)
        normalized.append((category, names))
    return tuple(normalized)


def _catalog_document(catalog: tuple[tuple[str, tuple[str, ...]], ...]) -> dict[str, list[str]]:
    return {category: list(garments) for category, garments in catalog}


def _require_seed(value: object, *, field: str, maximum: int = _MAX_SEED) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def type_weight(success_rate: float) -> float:
    """Favor categories with poor authenticated calibration performance."""

    if type(success_rate) not in (int, float) or not math.isfinite(success_rate):
        raise ValueError("success rate must be finite")
    return max(1.0 - float(success_rate), 0.05)


def garment_weight(success_rate: float) -> float:
    """Favor garments around the decision boundary, retaining a floor."""

    if type(success_rate) not in (int, float) or not math.isfinite(success_rate):
        raise ValueError("success rate must be finite")
    rate = float(success_rate)
    return max(math.exp(-((rate - 0.5) ** 2) / (2 * SIGMA**2)), 0.02)


def build_calibration_rows(catalog: object, *, seed_base: int) -> list[dict[str, object]]:
    """Build the fixed 400-row, evenly interleaved seen-garment calibration."""

    normalized = _catalog(catalog)
    seed_base = _require_seed(seed_base, field="seed_base", maximum=_MAX_SEED - 399)
    by_category = dict(normalized)
    rows: list[dict[str, object]] = []
    category_attempts = {category: 0 for category in _CATEGORIES}
    for index in range(400):
        category = _CATEGORIES[index % len(_CATEGORIES)]
        category_index = category_attempts[category]
        category_attempts[category] += 1
        garment = by_category[category][category_index % 10]
        seed = seed_base + index
        rows.append({
            "campaign_kind": "simple_curriculum_source_v1",
            "logical_stage": "calibration",
            "attempt_id": f"calibration-{index:04d}",
            "trial_id": f"calibration-trial-{index:04d}",
            "garment": garment,
            "garment_name": garment,
            "category": category,
            "release_stage": "seen",
            "seed": seed,
            "source_seed": seed,
            "strategy": "canonical",
        })
    return rows


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _require_policy_identity(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("policy identity is invalid")
    required = {"policy_repo", "policy_revision", "policy_step", "policy_artifact_sha256"}
    if set(value) != required:
        raise ValueError("policy identity is invalid")
    if (not isinstance(value["policy_repo"], str) or not value["policy_repo"]
            or not isinstance(value["policy_revision"], str) or re.fullmatch(r"[0-9a-f]{40}", value["policy_revision"]) is None
            or type(value["policy_step"]) is not int or value["policy_step"] != 12000):
        raise ValueError("policy identity is invalid")
    _require_sha256(value["policy_artifact_sha256"], field="policy identity artifact")
    return dict(value)


def _valid_outcomes(report: Mapping[str, object]) -> list[dict[str, object]]:
    outcomes = report.get("outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != 400:
        raise ValueError("calibration report must authenticate exactly 400 valid outcomes")
    normalized: list[dict[str, object]] = []
    attempt_ids: set[str] = set()
    trial_ids: set[str] = set()
    for outcome in outcomes:
        if not isinstance(outcome, Mapping):
            raise ValueError("calibration report must authenticate exactly 400 valid outcomes")
        attempt_id, trial_id, success = outcome.get("attempt_id"), outcome.get("trial_id"), outcome.get("success")
        if (not isinstance(attempt_id, str) or not attempt_id or not isinstance(trial_id, str) or not trial_id
                or type(success) is not bool or attempt_id in attempt_ids or trial_id in trial_ids):
            raise ValueError("calibration report must authenticate exactly 400 valid outcomes")
        attempt_ids.add(attempt_id)
        trial_ids.add(trial_id)
        normalized.append({"attempt_id": attempt_id, "trial_id": trial_id, "success": success})
    return normalized


def validate_calibration_report(
    report: object,
    *,
    matrix_sha256: str,
    policy_identity: object,
    catalog: object,
) -> dict[str, object]:
    """Fail closed unless a report binds the exact policy, matrix, and lane."""

    expected_matrix = _require_sha256(matrix_sha256, field="matrix SHA-256")
    expected_policy = _require_policy_identity(policy_identity)
    expected_catalog = _catalog(catalog)
    if not isinstance(report, Mapping):
        raise ValueError("calibration report is invalid")
    if report.get("schema_version") != 1 or report.get("kind") != "lehome_simple_curriculum_calibration_report_v1" or report.get("authenticated") is not True:
        raise ValueError("calibration report is not authenticated")
    if report.get("calibration_matrix_sha256") != expected_matrix:
        raise ValueError("calibration report matrix SHA-256 mismatch")
    if _require_policy_identity(report.get("policy_identity")) != expected_policy:
        raise ValueError("calibration report policy identity mismatch")
    if _require_policy_identity(report.get("authenticated_policy_identity")) != expected_policy:
        raise ValueError("calibration report policy identity authentication mismatch")
    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("simulator_device") != "cpu" or not isinstance(provenance.get("policy_device"), str) or re.fullmatch(r"cuda:[0-9]+", provenance["policy_device"]) is None:
        raise ValueError("calibration report provenance must be CPU cloth with CUDA policy")
    try:
        report_catalog = _catalog(report.get("catalog"))
    except ValueError as error:
        raise ValueError("calibration report catalog is invalid") from error
    if report_catalog != expected_catalog:
        raise ValueError("calibration report catalog does not match the approved catalog")
    outcomes = _valid_outcomes(report)
    return {
        "matrix_sha256": expected_matrix,
        "policy_identity": expected_policy,
        "catalog": _catalog_document(expected_catalog),
        "outcomes": outcomes,
    }


def _calibration_index(rows: object, *, catalog: tuple[tuple[str, tuple[str, ...]], ...]) -> dict[str, dict[str, object]]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or len(rows) != 400:
        raise ValueError("calibration matrix must contain exactly 400 rows")
    expected = build_calibration_rows(_catalog_document(catalog), seed_base=_require_seed(rows[0].get("seed") if isinstance(rows[0], Mapping) else None, field="calibration seed") if rows else 0)
    if list(rows) != expected:
        raise ValueError("calibration matrix is not canonical")
    return {str(row["attempt_id"]): row for row in expected}


def _choose_weighted(items: Sequence[str], weights: Sequence[float], generator: object) -> str:
    if not items or len(items) != len(weights) or any(not math.isfinite(weight) or weight <= 0 for weight in weights):
        raise ValueError("weighted selection inputs are invalid")
    choice = getattr(generator, "choices", None)
    if not callable(choice):
        raise ValueError("curriculum RNG is invalid")
    selected = choice(items, weights=weights, k=1)
    if not isinstance(selected, Sequence) or len(selected) != 1 or selected[0] not in items:
        raise ValueError("curriculum RNG returned an invalid choice")
    return str(selected[0])


def _draw_unique_seed(generator: object, used: set[int], *, max_draws: int = 10_000) -> int:
    draw = getattr(generator, "randrange", None)
    if not callable(draw):
        raise ValueError("curriculum RNG is invalid")
    for _ in range(max_draws):
        seed = draw(0, 1 << 32)
        if type(seed) is not int or not 0 <= seed < 1 << 32:
            raise ValueError("curriculum RNG returned an invalid seed")
        if seed not in used:
            used.add(seed)
            return seed
    raise ValueError("curriculum seed space exhausted")


def build_curriculum_rows(
    report: object,
    *,
    calibration_rows: object,
    count: int,
    rng_seed: int,
    policy_identity: object,
    catalog: object,
    rng_factory: Callable[[int], object] = random.Random,
) -> list[dict[str, object]]:
    """Sample an authenticated 600-row curriculum with an injected RNG seed."""

    if type(count) is not int or count < 1:
        raise ValueError("curriculum count must be a positive integer")
    rng_seed = _require_seed(rng_seed, field="rng_seed")
    expected_catalog = _catalog(catalog)
    expected_policy = _require_policy_identity(policy_identity)
    matrix_sha = _matrix_sha256(calibration_rows)
    validated = validate_calibration_report(
        report,
        matrix_sha256=matrix_sha,
        policy_identity=expected_policy,
        catalog=_catalog_document(expected_catalog),
    )
    calibration = _calibration_index(calibration_rows, catalog=expected_catalog)
    outcomes = validated["outcomes"]
    assert isinstance(outcomes, list)
    outcome_by_attempt = {str(outcome["attempt_id"]): outcome for outcome in outcomes}
    if set(outcome_by_attempt) != set(calibration) or any(outcome_by_attempt[attempt_id]["trial_id"] != row["trial_id"] for attempt_id, row in calibration.items()):
        raise ValueError("calibration report outcomes do not authenticate the calibration matrix")
    successes: dict[str, list[bool]] = defaultdict(list)
    category_successes: dict[str, list[bool]] = defaultdict(list)
    for attempt_id, row in calibration.items():
        success = outcome_by_attempt[attempt_id]["success"]
        assert type(success) is bool
        successes[str(row["garment"])].append(success)
        category_successes[str(row["category"])].append(success)
    garment_rates = {garment: sum(values) / len(values) for garment, values in successes.items()}
    category_rates = {category: sum(values) / len(values) for category, values in category_successes.items()}
    catalog_by_category = dict(expected_catalog)
    category_weights = [type_weight(category_rates[category]) for category in _CATEGORIES]
    generator = rng_factory(rng_seed)
    calibration_attempt_ids = set(calibration)
    calibration_trial_ids = {str(row["trial_id"]) for row in calibration.values()}
    calibration_seeds = {int(row["seed"]) for row in calibration.values()}
    used_seeds = set(calibration_seeds)
    rows: list[dict[str, object]] = []
    for index in range(count):
        category = _choose_weighted(_CATEGORIES, category_weights, generator)
        garments = catalog_by_category[category]
        garment_weights = [garment_weight(garment_rates[garment]) for garment in garments]
        garment = _choose_weighted(garments, garment_weights, generator)
        seed = _draw_unique_seed(generator, used_seeds)
        attempt_id = f"curriculum-{index:04d}"
        trial_id = f"curriculum-trial-{index:04d}"
        if attempt_id in calibration_attempt_ids or trial_id in calibration_trial_ids or seed in calibration_seeds:
            raise AssertionError("curriculum identity collision")
        rows.append({
            "campaign_kind": "simple_curriculum_source_v1",
            "logical_stage": "curriculum",
            "attempt_id": attempt_id,
            "trial_id": trial_id,
            "garment": garment,
            "garment_name": garment,
            "category": category,
            "release_stage": "seen",
            "seed": seed,
            "source_seed": seed,
            "strategy": "canonical",
            "builder_rng_seed": rng_seed,
            "calibration_matrix_sha256": matrix_sha,
            "sampled_category_weight": category_weights[_CATEGORIES.index(category)],
            "sampled_garment_weight": garment_weights[garments.index(garment)],
        })
    return rows
