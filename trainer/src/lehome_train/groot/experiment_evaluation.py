"""Strict immutable evaluation evidence used by asynchronous promotion."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping, Sequence

from lehome_train.io import canonical_json_sha256


_CATEGORIES = ("top_long", "top_short", "pant_long", "pant_short")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_STRICT_FIELDS = {"schema_version", "experiment_id", "checkpoint_receipt_sha256", "matrix_sha256", "policy_digest", "categories", "episode_artifacts", "promotion_metrics", "provenance", "strict_seal", "evidence_report_sha256", "report_sha256"}
_METRICS = {"overall_successes", "overall_episodes", "overall_success_rate", "safety_failure", "paired_improvement", "gpu_seconds", "infrastructure_retry_count", "progress", "recovery", "pairing"}
_ARTIFACT = {"schedule_index", "trial_id", "attempt_id", "category", "garment", "seed", "official_success", "terminal_event", "episode_sha256", "worker_receipt_sha256"}


@dataclass(frozen=True, slots=True)
class CategoryScore:
    successes: int
    episodes: int


@dataclass(frozen=True, slots=True)
class ExperimentEvaluation:
    experiment_id: str
    checkpoint_receipt_sha256: str
    matrix_sha256: str
    policy_digest: str
    categories: Mapping[str, CategoryScore]
    overall_successes: int
    overall_episodes: int
    overall_success_rate: float
    episode_artifacts: tuple[Mapping[str, object], ...]
    paired_improvement: float
    gpu_seconds: float
    infrastructure_retry_count: int
    safety_failure: bool
    progress_metrics: Mapping[str, object]
    recovery_metrics: Mapping[str, object]
    pairing_metrics: Mapping[str, object]
    provenance: Mapping[str, object]
    evidence_report_sha256: str | None
    strict: bool
    sha256: str


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise ValueError(f"{label} must be SHA-256")
    return value


def _finite(value: object, label: str, *, nonnegative: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)) or (nonnegative and float(value) < 0):
        raise ValueError(f"{label} is invalid")
    return float(value)


def _categories(value: object) -> dict[str, CategoryScore]:
    if not isinstance(value, Mapping) or set(value) != set(_CATEGORIES):
        raise ValueError("evaluation requires exactly four categories")
    out: dict[str, CategoryScore] = {}
    for name in _CATEGORIES:
        score = value[name]
        if not isinstance(score, Mapping) or set(score) != {"successes", "episodes"} or type(score.get("successes")) is not int or type(score.get("episodes")) is not int or score["episodes"] <= 0 or not 0 <= score["successes"] <= score["episodes"]:
            raise ValueError("category score is invalid")
        out[name] = CategoryScore(score["successes"], score["episodes"])
    return out


def build_legacy_experiment_evaluation(*, experiment_id: str, checkpoint_receipt_sha256: str, matrix_sha256: str, policy_digest: str, categories: Mapping[str, Mapping[str, int]], episode_artifacts: Sequence[Mapping[str, object]], paired_improvement: float = 0.0, gpu_seconds: float = 0.0, infrastructure_retry_count: int = 0, safety_failure: bool = False, progress_metrics: Mapping[str, object] | None = None, recovery_metrics: Mapping[str, object] | None = None) -> ExperimentEvaluation:
    """Legacy in-memory score builder. Persisted reports use the strict parser."""
    for value, label in ((experiment_id, "experiment ID"), (checkpoint_receipt_sha256, "checkpoint receipt"), (matrix_sha256, "matrix"), (policy_digest, "policy")):
        _sha(value, label)
    parsed = _categories(categories)
    rows = tuple(MappingProxyType(dict(row)) for row in episode_artifacts)
    ids = [str(row.get("trial_id", "")) for row in rows]
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError("duplicate or missing episode artifact")
    paired, gpu = _finite(paired_improvement, "paired improvement"), _finite(gpu_seconds, "GPU seconds", nonnegative=True)
    if type(infrastructure_retry_count) is not int or infrastructure_retry_count < 0 or type(safety_failure) is not bool:
        raise ValueError("promotion metrics are invalid")
    progress, recovery = MappingProxyType(dict(progress_metrics or {})), MappingProxyType(dict(recovery_metrics or {}))
    raw = {"schema_version": 1, "experiment_id": experiment_id, "checkpoint_receipt_sha256": checkpoint_receipt_sha256, "matrix_sha256": matrix_sha256, "policy_digest": policy_digest, "categories": {name: {"successes": parsed[name].successes, "episodes": parsed[name].episodes} for name in _CATEGORIES}, "episode_artifacts": [dict(row) for row in rows], "promotion_metrics": {"paired_improvement": paired, "gpu_seconds": gpu, "infrastructure_retry_count": infrastructure_retry_count, "safety_failure": safety_failure, "progress": dict(progress), "recovery": dict(recovery)}}
    total_successes, total_episodes = sum(item.successes for item in parsed.values()), sum(item.episodes for item in parsed.values())
    return ExperimentEvaluation(experiment_id, checkpoint_receipt_sha256, matrix_sha256, policy_digest, MappingProxyType(parsed), total_successes, total_episodes, total_successes / total_episodes, rows, paired, gpu, infrastructure_retry_count, safety_failure, progress, recovery, MappingProxyType({"status": "legacy_unpaired"}), MappingProxyType({}), None, False, canonical_json_sha256(raw))


# Existing controller imports remain valid; this is intentionally not a
# persisted strict-report parser.
build_experiment_evaluation = build_legacy_experiment_evaluation


def _report_sha(document: Mapping[str, object]) -> str:
    raw = dict(document)
    raw.pop("report_sha256", None)
    return hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()


def _artifacts(value: object, categories: Mapping[str, CategoryScore]) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("strict evaluation report has no episode artifacts")
    rows: list[Mapping[str, object]] = []
    trials: set[str] = set(); attempts: set[str] = set(); indices: set[int] = set()
    totals = {name: [0, 0] for name in _CATEGORIES}
    for item in value:
        if not isinstance(item, Mapping) or set(item) != _ARTIFACT:
            raise ValueError("strict evaluation episode artifact schema is invalid")
        index, trial, attempt, category = item.get("schedule_index"), item.get("trial_id"), item.get("attempt_id"), item.get("category")
        if type(index) is not int or index < 0 or type(trial) is not str or not trial or type(attempt) is not str or _SHA.fullmatch(attempt) is None or category not in _CATEGORIES or type(item.get("garment")) is not str or not item["garment"] or type(item.get("seed")) is not int or type(item.get("official_success")) is not int or item["official_success"] not in {0, 1}:
            raise ValueError("strict evaluation episode artifact identity is invalid")
        if item.get("terminal_event") != ("accepted" if item["official_success"] else "rejected"):
            raise ValueError("strict evaluation episode artifact terminal outcome is invalid")
        _sha(item.get("episode_sha256"), "episode artifact"); _sha(item.get("worker_receipt_sha256"), "worker receipt")
        if trial in trials or attempt in attempts or index in indices:
            raise ValueError("strict evaluation episode artifact is duplicated")
        trials.add(trial); attempts.add(attempt); indices.add(index)
        totals[category][0] += item["official_success"]; totals[category][1] += 1
        rows.append(MappingProxyType(dict(item)))
    if indices != set(range(len(rows))) or any(tuple(totals[name]) != (categories[name].successes, categories[name].episodes) for name in _CATEGORIES):
        raise ValueError("strict evaluation artifacts do not match category totals")
    return tuple(rows)


def _provenance(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != {"trainer", "runtime", "data_sources"}:
        raise ValueError("strict evaluation provenance is invalid")
    trainer, runtime, sources = value["trainer"], value["runtime"], value["data_sources"]
    if not isinstance(trainer, Mapping) or set(trainer) != {"image_id", "oci_digest", "code_revision"} or type(trainer.get("image_id")) is not str or not trainer["image_id"] or type(trainer.get("oci_digest")) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", trainer["oci_digest"]) is None or type(trainer.get("code_revision")) is not str or _COMMIT.fullmatch(trainer["code_revision"]) is None:
        raise ValueError("strict evaluation trainer provenance is invalid")
    if not isinstance(runtime, Mapping) or set(runtime) != {"code_revision", "asset_revision", "simulator_version", "image_identity"} or type(runtime.get("code_revision")) is not str or _COMMIT.fullmatch(runtime["code_revision"]) is None or type(runtime.get("asset_revision")) is not str or _COMMIT.fullmatch(runtime["asset_revision"]) is None or type(runtime.get("simulator_version")) is not str or not runtime["simulator_version"] or type(runtime.get("image_identity")) is not str or not runtime["image_identity"]:
        raise ValueError("strict evaluation runtime provenance is invalid")
    if not isinstance(sources, list) or not sources:
        raise ValueError("strict evaluation data provenance is invalid")
    parsed_sources = []
    for source in sources:
        if not isinstance(source, Mapping) or set(source) != {"kind", "repository", "revision", "prefix", "manifest_sha256", "tree_sha256"} or any(type(source.get(key)) is not str or not source[key] for key in ("kind", "repository", "revision", "prefix")) or _COMMIT.fullmatch(source["revision"]) is None:
            raise ValueError("strict evaluation data provenance is invalid")
        _sha(source.get("manifest_sha256"), "data manifest"); _sha(source.get("tree_sha256"), "data tree")
        parsed_sources.append(MappingProxyType(dict(source)))
    return MappingProxyType({"trainer": MappingProxyType(dict(trainer)), "runtime": MappingProxyType(dict(runtime)), "data_sources": tuple(parsed_sources)})


def _pairing(value: object, *, paired_improvement: float, episodes: int) -> Mapping[str, object]:
    """Validate the baseline gate that distinguishes a real tie from no data."""
    if not isinstance(value, Mapping) or type(value.get("status")) is not str:
        raise ValueError("strict evaluation pairing evidence is invalid")
    status = value["status"]
    if status == "baseline_evaluation_required":
        if set(value) != {"status"}:
            raise ValueError("strict evaluation unavailable pairing is invalid")
        return MappingProxyType({"status": status})
    expected = {
        "status", "baseline_report_sha256", "paired_trials", "candidate_wins",
        "baseline_wins", "ties", "paired_improvement", "progress_improvement",
        "recovery_improvement",
    }
    if status != "available" or set(value) != expected:
        raise ValueError("strict evaluation pairing schema is invalid")
    _sha(value.get("baseline_report_sha256"), "paired baseline report")
    counts = (value.get("paired_trials"), value.get("candidate_wins"), value.get("baseline_wins"), value.get("ties"))
    if any(type(item) is not int for item in counts) or not 0 < counts[0] <= episodes or any(item < 0 for item in counts[1:]) or counts[1] + counts[2] + counts[3] != counts[0]:
        raise ValueError("strict evaluation paired trial totals are invalid")
    observed = _finite(value.get("paired_improvement"), "paired improvement")
    if not math.isclose(observed, paired_improvement, rel_tol=0.0, abs_tol=1e-12) or not math.isclose(observed, (counts[1] - counts[2]) / counts[0], rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("strict evaluation paired improvement is invalid")
    _finite(value.get("progress_improvement"), "paired progress improvement")
    _finite(value.get("recovery_improvement"), "paired recovery improvement")
    return MappingProxyType(dict(value))


def parse_experiment_evaluation(value: Mapping[str, object]) -> ExperimentEvaluation:
    """Fail closed on the exact strict-report schema emitted by the summarizer."""
    document = dict(value)
    if set(document) != _STRICT_FIELDS or document.get("schema_version") != 1:
        raise ValueError("strict evaluation report schema is invalid")
    for name in ("experiment_id", "checkpoint_receipt_sha256", "matrix_sha256", "policy_digest", "evidence_report_sha256", "report_sha256"):
        _sha(document.get(name), name.replace("_", " "))
    if document["report_sha256"] != _report_sha(document):
        raise ValueError("strict evaluation report SHA-256 mismatch")
    if document.get("strict_seal") is not False:
        raise ValueError("strict evaluation report must not use a collection seal")
    categories = _categories(document.get("categories"))
    artifacts = _artifacts(document.get("episode_artifacts"), categories)
    metrics = document.get("promotion_metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != _METRICS or type(metrics.get("overall_successes")) is not int or type(metrics.get("overall_episodes")) is not int or type(metrics.get("safety_failure")) is not bool or type(metrics.get("infrastructure_retry_count")) is not int or metrics["infrastructure_retry_count"] < 0:
        raise ValueError("strict evaluation promotion metrics are invalid")
    successes, episodes = sum(score.successes for score in categories.values()), sum(score.episodes for score in categories.values())
    if metrics["overall_successes"] != successes or metrics["overall_episodes"] != episodes:
        raise ValueError("strict evaluation overall metrics disagree with categories")
    overall = _finite(metrics.get("overall_success_rate"), "overall success rate", nonnegative=True)
    if overall > 1 or not math.isclose(overall, successes / episodes, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("strict evaluation overall success rate is invalid")
    paired, gpu = _finite(metrics.get("paired_improvement"), "paired improvement"), _finite(metrics.get("gpu_seconds"), "GPU seconds", nonnegative=True)
    progress, recovery = metrics.get("progress"), metrics.get("recovery")
    if not isinstance(progress, Mapping) or set(progress) != {"observed_episodes", "mean_terminal_progress"} or type(progress.get("observed_episodes")) is not int or not 0 <= progress["observed_episodes"] <= episodes or _finite(progress.get("mean_terminal_progress"), "mean terminal progress", nonnegative=True) > 1:
        raise ValueError("strict evaluation progress metrics are invalid")
    if not isinstance(recovery, Mapping) or set(recovery) != {"recovery_attempts", "successful_recoveries"} or type(recovery.get("recovery_attempts")) is not int or type(recovery.get("successful_recoveries")) is not int or not 0 <= recovery["successful_recoveries"] <= recovery["recovery_attempts"] <= episodes:
        raise ValueError("strict evaluation recovery metrics are invalid")
    pairing = _pairing(metrics.get("pairing"), paired_improvement=paired, episodes=episodes)
    return ExperimentEvaluation(document["experiment_id"], document["checkpoint_receipt_sha256"], document["matrix_sha256"], document["policy_digest"], MappingProxyType(categories), successes, episodes, overall, artifacts, paired, gpu, metrics["infrastructure_retry_count"], metrics["safety_failure"], MappingProxyType(dict(progress)), MappingProxyType(dict(recovery)), pairing, _provenance(document.get("provenance")), document["evidence_report_sha256"], True, document["report_sha256"])


def to_evaluation_score(evaluation: ExperimentEvaluation):
    """Adapt immutable evaluation evidence for the controller's pure ranking rules."""
    from lehome_train.groot.experiment_promotion import EvaluationScore
    # The in-memory legacy builder is retained for small pure unit tests.  All
    # persisted controller reports are strict and must carry authenticated
    # pairing evidence before they can be ranked.
    if evaluation.strict and evaluation.pairing_metrics.get("status") != "available":
        raise ValueError("baseline_evaluation_required")
    return EvaluationScore(experiment_id=evaluation.experiment_id, policy_digest=evaluation.policy_digest, category_successes=tuple(evaluation.categories[name].successes for name in _CATEGORIES), overall_successes=evaluation.overall_successes, paired_improvement=evaluation.paired_improvement, gpu_seconds=evaluation.gpu_seconds, safety_failure=evaluation.safety_failure)


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in items:
        if key in output:
            raise ValueError("evaluation report has duplicate fields")
        output[key] = value
    return output


def load_experiment_evaluation(path: str | Path) -> ExperimentEvaluation:
    """Read canonical strict-report bytes and their immutable hash sidecar."""
    source = Path(path); sidecar = source.with_suffix(source.suffix + ".sha256")
    if source.is_symlink() or sidecar.is_symlink() or not source.is_file() or not sidecar.is_file():
        raise ValueError("evaluation report or sidecar is unsafe")
    try:
        raw = source.read_bytes(); document = json.loads(raw, object_pairs_hook=_pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON number"))); digest = sidecar.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("evaluation report is malformed") from error
    if _SHA.fullmatch(digest or "") is None or digest != hashlib.sha256(raw).hexdigest():
        raise ValueError("evaluation report sidecar does not bind report bytes")
    if not isinstance(document, dict) or raw != json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8") + b"\n":
        raise ValueError("evaluation report is not canonical")
    return parse_experiment_evaluation(document)
