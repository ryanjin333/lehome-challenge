"""Canonical, immutable contracts for asynchronous LeHome experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType
from typing import Mapping

from lehome_train.groot.experiment_manifest import batch64_quotas
from lehome_train.io import canonical_json_sha256


_SHA = re.compile(r"^[0-9a-f]{64}$")
_REV = re.compile(r"^[0-9a-f]{40}$")
_RATIOS = {(100, 0), (95, 5), (90, 10), (85, 15), (80, 20), (70, 30)}
_SECRET = re.compile(r"(?:token|password|secret|api[_-]?key)", re.I)


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON field")
        value[key] = item
    return value


def _exact(value: object, keys: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} has unknown or missing field")
    return value


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise ValueError(f"{label} must be SHA-256")
    return value


def _revision(value: object, label: str) -> str:
    if type(value) is not str or _REV.fullmatch(value) is None:
        raise ValueError(f"{label} must be an immutable revision")
    return value


def _relative(value: object, label: str) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise ValueError(f"{label} must be a safe prefix")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be a safe prefix")
    return value


def _repository(value: object, label: str) -> str:
    if type(value) is not str or not value or value.startswith("/") or " " in value or "/" not in value:
        raise ValueError(f"{label} is invalid")
    return value


def _no_secrets(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if _SECRET.search(key):
                raise ValueError("experiment documents cannot contain secret-shaped fields")
            _no_secrets(item)
    elif isinstance(value, list):
        for item in value:
            _no_secrets(item)


@dataclass(frozen=True, slots=True)
class ArtifactBinding:
    repository: str
    revision: str
    prefix: str
    manifest_sha256: str
    tree_sha256: str
    kind: str


@dataclass(frozen=True, slots=True)
class MixtureBinding:
    bc_percent: int
    added_percent: int
    batch64_quotas: Mapping[str, int]
    sampling_strategy: str


@dataclass(frozen=True, slots=True)
class TrainingBudget:
    action_horizon: int
    batch_size: int
    seed: int
    target_step: int
    save_steps: int


@dataclass(frozen=True, slots=True)
class EvaluationBinding:
    matrix_id: str
    matrix_sha256: str
    policy_digest: str


@dataclass(frozen=True, slots=True)
class PublicationBinding:
    checkpoint_repository: str
    result_repository: str
    prefix: str


@dataclass(frozen=True, slots=True)
class ExperimentJob:
    experiment_id: str
    arm: str
    parent_checkpoint: Mapping[str, str]
    trainer: Mapping[str, str]
    data_sources: tuple[ArtifactBinding, ...]
    mixture: MixtureBinding
    training: TrainingBudget
    evaluation: EvaluationBinding
    publication: PublicationBinding
    dependencies: tuple[str, ...]
    admission: Mapping[str, str]
    raw: Mapping[str, object]


def experiment_identity(document: Mapping[str, object]) -> str:
    """Hash exactly the canonical identity, excluding only its declared ID."""
    identity = dict(document)
    identity.pop("experiment_id", None)
    return canonical_json_sha256(identity)


def _load_document(path: str | Path) -> dict[str, object]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("experiment job is missing or unsafe")
    try:
        raw = source.read_bytes()
        document = json.loads(raw, object_pairs_hook=_pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON number")))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("experiment job is malformed") from error
    if json.dumps(document, sort_keys=True, separators=(",", ":")).encode() != raw:
        raise ValueError("experiment job must use canonical JSON")
    return document


def _parse(document: dict[str, object]) -> ExperimentJob:
    _no_secrets(document)
    base_fields = {"schema_version", "experiment_id", "arm", "parent_checkpoint", "trainer", "data_sources", "mixture", "training", "evaluation", "publication", "dependencies"}
    if set(document) not in (base_fields, base_fields | {"admission"}):
        raise ValueError("experiment job has unknown or missing field")
    item = document
    if item["schema_version"] != 1 or type(item["schema_version"]) is not int:
        raise ValueError("experiment job schema is unsupported")
    if type(item["arm"]) is not str or not item["arm"]:
        raise ValueError("experiment arm is invalid")
    declared = _sha(item["experiment_id"], "experiment ID")
    if declared != experiment_identity(item):
        raise ValueError("declared experiment ID does not match canonical digest")
    parent = item["parent_checkpoint"]
    if not isinstance(parent, dict) or set(parent) not in (
        {"repository", "revision", "subpath", "artifact_sha256"},
        {"repository", "revision", "subpath", "artifact_sha256", "receipt_sha256"},
    ):
        raise ValueError("parent checkpoint has unknown or missing field")
    parent_fields = {"repository": _repository(parent["repository"], "parent repository"), "revision": _revision(parent["revision"], "parent revision"), "subpath": _relative(parent["subpath"], "parent subpath"), "artifact_sha256": _sha(parent["artifact_sha256"], "parent artifact")}
    if "receipt_sha256" in parent:
        parent_fields["receipt_sha256"] = _sha(parent["receipt_sha256"], "parent checkpoint receipt")
    parent_value = MappingProxyType(parent_fields)
    trainer = _exact(item["trainer"], {"image_id", "oci_digest", "code_revision"}, "trainer")
    if type(trainer["image_id"]) is not str or not trainer["image_id"] or type(trainer["oci_digest"]) is not str or not re.fullmatch(r"sha256:[0-9a-f]{64}", trainer["oci_digest"]):
        raise ValueError("trainer identity is invalid")
    trainer_value = MappingProxyType({"image_id": trainer["image_id"], "oci_digest": trainer["oci_digest"], "code_revision": _revision(trainer["code_revision"], "trainer code revision")})
    raw_sources = item["data_sources"]
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("data sources are invalid")
    sources: list[ArtifactBinding] = []
    for raw in raw_sources:
        source = _exact(raw, {"kind", "repository", "revision", "prefix", "manifest_sha256", "tree_sha256"}, "data source")
        if type(source["kind"]) is not str or not source["kind"]:
            raise ValueError("data source kind is invalid")
        sources.append(ArtifactBinding(_repository(source["repository"], "data repository"), _revision(source["revision"], "data revision"), _relative(source["prefix"], "data prefix"), _sha(source["manifest_sha256"], "data manifest"), _sha(source["tree_sha256"], "data tree"), source["kind"]))
    mixture = _exact(item["mixture"], {"bc_percent", "added_percent", "batch64_quotas", "sampling_strategy"}, "mixture")
    bc, added = mixture["bc_percent"], mixture["added_percent"]
    if type(bc) is not int or type(added) is not int or (bc, added) not in _RATIOS:
        raise ValueError("mixture is not an approved sweep ratio")
    expected = batch64_quotas({"bc": bc, "rollout": added, "dagger": 0})
    if mixture["batch64_quotas"] != expected or type(mixture["sampling_strategy"]) is not str or mixture["sampling_strategy"] not in {"unweighted", "awr_style_weighted_replay"}:
        raise ValueError("mixture quota or sampling strategy is invalid")
    training = _exact(item["training"], {"action_horizon", "batch_size", "seed", "target_step", "save_steps"}, "training")
    if training["action_horizon"] != 16 or training["batch_size"] != 64 or type(training["seed"]) is not int or training["target_step"] not in (500, 1000, 2000) or training["save_steps"] != 500:
        raise ValueError("training sweep invariant drift")
    evaluation = _exact(item["evaluation"], {"matrix_id", "matrix_sha256", "policy_digest"}, "evaluation")
    if type(evaluation["matrix_id"]) is not str or not evaluation["matrix_id"]:
        raise ValueError("evaluation matrix is invalid")
    matrix_sha256 = _sha(evaluation["matrix_sha256"], "matrix hash")
    policy_digest = _sha(evaluation["policy_digest"], "policy digest")
    publication = _exact(item["publication"], {"checkpoint_repository", "result_repository", "prefix"}, "publication")
    deps = item["dependencies"]
    if not isinstance(deps, list) or any(type(value) is not str or not value for value in deps) or len(set(deps)) != len(deps):
        raise ValueError("dependencies are invalid")
    admission_raw = item.get("admission", {"kind": "initial"})
    if not isinstance(admission_raw, dict) or type(admission_raw.get("kind")) is not str:
        raise ValueError("experiment admission is invalid")
    if admission_raw["kind"] == "initial":
        admission = _exact(admission_raw, {"kind"}, "initial admission")
    elif admission_raw["kind"] == "seed_repeat":
        admission = _exact(admission_raw, {"kind", "source_experiment_id"}, "seed repeat admission")
        _sha(admission["source_experiment_id"], "seed repeat source experiment")
    elif admission_raw["kind"] == "continuation":
        admission = _exact(admission_raw, {"kind", "source_experiment_id"}, "continuation admission")
        _sha(admission["source_experiment_id"], "continuation source experiment")
    elif admission_raw["kind"] == "awr_style_weighted_replay":
        admission = _exact(
            admission_raw,
            {
                "kind", "pending_admission_sha256", "matched_training_sha256",
                "progress_evidence_sha256", "progress_evidence_receipt_sha256",
                "progress_evidence_mixture_id", "progress_evidence_mixture_manifest_sha256",
                "awr_replay_config_sha256", "winning_unweighted_experiment_id",
                "winning_unweighted_report_sha256", "winning_unweighted_seal_sha256",
            },
            "AWR-style weighted replay admission",
        )
        for field in set(admission) - {"kind"}:
            _sha(admission[field], f"AWR-style admission {field}")
    else:
        raise ValueError("experiment admission kind is invalid")
    if admission["kind"] in {"initial", "seed_repeat"} and policy_digest != parent_fields["artifact_sha256"]:
        raise ValueError("evaluation baseline policy does not match the original parent checkpoint")
    return ExperimentJob(declared, item["arm"], parent_value, trainer_value, tuple(sources), MixtureBinding(bc, added, MappingProxyType(expected), mixture["sampling_strategy"]), TrainingBudget(16, 64, training["seed"], training["target_step"], 500), EvaluationBinding(evaluation["matrix_id"], matrix_sha256, policy_digest), PublicationBinding(_repository(publication["checkpoint_repository"], "checkpoint repository"), _repository(publication["result_repository"], "result_repository"), _relative(publication["prefix"], "publication prefix")), tuple(deps), MappingProxyType({key: str(value) for key, value in admission.items()}), MappingProxyType(dict(item)))


def load_experiment_job(path: str | Path) -> ExperimentJob:
    return _parse(_load_document(path))


def dump_experiment_job(path: str | Path, identity: Mapping[str, object]) -> ExperimentJob:
    document = dict(identity)
    document.pop("experiment_id", None)
    document["experiment_id"] = experiment_identity(document)
    target = Path(path)
    if target.is_absolute() and target.is_symlink():
        raise ValueError("experiment job path is unsafe")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(json.dumps(document, sort_keys=True, separators=(",", ":")).encode())
    return load_experiment_job(target)
