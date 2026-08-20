"""Contracts for the one matched deterministic AWR-style replay ablation.

The trainer does not accept per-sample loss weights.  This ablation therefore
changes replay frequency only, and is intentionally named *AWR-style weighted
replay*, never canonical AWR.  The publisher that creates the new runtime
request-set is external to this module, so absence of its read-back receipt is
an explicit pending admission rather than a runnable experiment.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Mapping

from lehome_train.groot.awr_weighting import AwrReplayConfig, authenticated_progress_evidence_receipt_sha256
from lehome_train.groot.experiment_job import ArtifactBinding, ExperimentJob, _parse, experiment_identity
from lehome_train.groot.experiment_publication import parse_checkpoint_publication
from lehome_train.groot.experiment_runtime_request import runtime_profile_sha256
from lehome_train.groot.experiment_winner import validate_final_unseen80_report, winner_gate
from lehome_train.io import canonical_json_sha256


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_FIELDS = {"kind", "repository", "revision", "prefix", "manifest_sha256", "tree_sha256"}
_PENDING_FIELDS = {
    "schema_version", "kind", "parent_experiment_id", "unweighted_checkpoint_publication",
    "unweighted_runtime_request_set", "matched_training_sha256", "progress_evidence_sha256",
    "progress_evidence_receipt_sha256", "progress_evidence_mixture_id",
    "progress_evidence_mixture_manifest_sha256",
    "winning_unweighted_report_sha256", "winning_unweighted_seal_sha256",
    "awr_replay_config", "awr_replay_config_sha256", "required_weighted_request_set",
    "controller_admission_contract",
}
_RECEIPT_FIELDS = {
    "schema_version", "kind", "pending_admission_sha256", "weighted_runtime_request_set",
    "child_runtime_profile_sha256", "matched_training_sha256", "readback_receipt_sha256",
    "authenticated_principal_sha256", "progress_evidence_sha256",
    "progress_evidence_receipt_sha256", "progress_evidence_mixture_id",
    "progress_evidence_mixture_manifest_sha256", "awr_replay_config_sha256",
    "winning_unweighted_experiment_id", "winning_unweighted_report_sha256",
    "winning_unweighted_seal_sha256", "readback_verified",
}


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be SHA-256")
    return value


def _source_document(source: ArtifactBinding) -> dict[str, str]:
    return {
        "kind": source.kind,
        "repository": source.repository,
        "revision": source.revision,
        "prefix": source.prefix,
        "manifest_sha256": source.manifest_sha256,
        "tree_sha256": source.tree_sha256,
    }


def _parse_source(value: Mapping[str, object]) -> dict[str, str]:
    if set(value) != _SOURCE_FIELDS or value.get("kind") != "runtime_request_set":
        raise ValueError("AWR-style weighted source must be one runtime request set")
    for key in ("repository", "revision", "prefix"):
        if type(value.get(key)) is not str or not value[key]:
            raise ValueError("AWR-style weighted source identity is invalid")
    for key in ("manifest_sha256", "tree_sha256"):
        _sha(value.get(key), f"AWR-style weighted source {key}")
    return {key: str(value[key]) for key in _SOURCE_FIELDS}


def _runtime_source(job: ExperimentJob) -> ArtifactBinding:
    values = [source for source in job.data_sources if source.kind == "runtime_request_set"]
    if len(values) != 1:
        raise ValueError("AWR-style ablation requires exactly one parent runtime request set")
    return values[0]


def _matched_training_document(job: ExperimentJob) -> dict[str, object]:
    """Everything that must remain fixed across unweighted and weighted runs."""
    return {
        "schema_version": 1,
        "parent_checkpoint": dict(job.parent_checkpoint),
        "trainer": dict(job.trainer),
        "data_sources": [_source_document(source) for source in job.data_sources if source.kind != "runtime_request_set"],
        "mixture": {
            "bc_percent": job.mixture.bc_percent,
            "added_percent": job.mixture.added_percent,
            "batch64_quotas": dict(job.mixture.batch64_quotas),
        },
        "training": {
            "action_horizon": job.training.action_horizon,
            "batch_size": job.training.batch_size,
            "seed": job.training.seed,
            "target_step": job.training.target_step,
            "save_steps": job.training.save_steps,
        },
        "evaluation": {
            "matrix_id": job.evaluation.matrix_id,
            "matrix_sha256": job.evaluation.matrix_sha256,
        },
    }


def matched_training_sha256(job: ExperimentJob) -> str:
    return canonical_json_sha256(_matched_training_document(job))


def pending_admission_sha256(pending: Mapping[str, object]) -> str:
    if not isinstance(pending, Mapping) or set(pending) != _PENDING_FIELDS:
        raise ValueError("AWR-style pending admission has unknown or missing field")
    return canonical_json_sha256(dict(pending))


def build_awr_style_ablation(
    parent: ExperimentJob,
    *,
    winning_unweighted_report: Mapping[str, object],
    progress_evidence_sha256: str,
    progress_evidence_receipt: Mapping[str, object],
    replay_config: AwrReplayConfig,
) -> dict[str, object]:
    """Create a strict *pending* external-materialization contract.

    It cannot be leased. The external request-set publisher must create a new,
    profile-bound bundle and return the authenticated read-back receipt consumed
    by :func:`bind_weighted_runtime_request_set`.
    """
    if parent.mixture.sampling_strategy != "unweighted" or not any(source.kind == "recovery" for source in parent.data_sources):
        raise ValueError("AWR-style ablation requires an unweighted recovery parent")
    parsed = validate_final_unseen80_report(winning_unweighted_report)
    if winner_gate(winning_unweighted_report) != "winner" or parsed["experiment_id"] != parent.experiment_id:
        raise ValueError("AWR-style ablation requires a verified winning unweighted recovery job")
    publication = parsed["checkpoint_publication"]
    if publication["target_step"] != parent.training.target_step:
        raise ValueError("AWR-style winning report does not bind parent terminal step")
    evidence = _sha(progress_evidence_sha256, "AWR-style progress evidence")
    evidence_receipt_sha = authenticated_progress_evidence_receipt_sha256(progress_evidence_receipt)
    if progress_evidence_receipt.get("evidence_sha256") != evidence:
        raise ValueError("AWR-style progress evidence receipt does not bind evidence")
    evidence_mixture_id = _sha(progress_evidence_receipt.get("mixture_id"), "AWR-style evidence mixture")
    evidence_mixture_manifest = _sha(progress_evidence_receipt.get("mixture_manifest_sha256"), "AWR-style evidence mixture manifest")
    original_source = _source_document(_runtime_source(parent))
    fixed = matched_training_sha256(parent)
    return {
        "schema_version": 1,
        "kind": "lehome_awr_style_pending_materialization",
        "parent_experiment_id": parent.experiment_id,
        "unweighted_checkpoint_publication": publication,
        "unweighted_runtime_request_set": original_source,
        "matched_training_sha256": fixed,
        "progress_evidence_sha256": evidence,
        "progress_evidence_receipt_sha256": evidence_receipt_sha,
        "progress_evidence_mixture_id": evidence_mixture_id,
        "progress_evidence_mixture_manifest_sha256": evidence_mixture_manifest,
        "winning_unweighted_report_sha256": parsed["report_sha256"],
        "winning_unweighted_seal_sha256": parsed["seal_sha256"],
        "awr_replay_config": replay_config.to_dict(),
        "awr_replay_config_sha256": replay_config.sha256,
        "required_weighted_request_set": {
            "kind": "runtime_request_set",
            "must_differ_from_manifest_sha256": original_source["manifest_sha256"],
            "must_differ_from_tree_sha256": original_source["tree_sha256"],
            "must_bind_matched_training_sha256": fixed,
            "must_bind_progress_evidence_sha256": evidence,
            "must_bind_progress_evidence_receipt_sha256": evidence_receipt_sha,
            "must_bind_progress_evidence_mixture_id": evidence_mixture_id,
            "must_bind_progress_evidence_mixture_manifest_sha256": evidence_mixture_manifest,
            "must_bind_awr_replay_config_sha256": replay_config.sha256,
        },
        "controller_admission_contract": {
            "kind": "lehome_awr_style_weighted_replay_admission",
            "requires_authenticated_readback_receipt": True,
            "requires_new_runtime_request_set": True,
            "lease_state_before_receipt": "PENDING_MATERIALIZATION",
            "lease_state_after_valid_receipt": "READY",
        },
    }


def _validate_pending(pending: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(pending, Mapping) or set(pending) != _PENDING_FIELDS:
        raise ValueError("AWR-style pending admission has unknown or missing field")
    if pending.get("schema_version") != 1 or pending.get("kind") != "lehome_awr_style_pending_materialization":
        raise ValueError("AWR-style pending admission schema is invalid")
    _sha(pending.get("parent_experiment_id"), "AWR-style parent experiment")
    source = pending.get("unweighted_runtime_request_set")
    if not isinstance(source, Mapping):
        raise ValueError("AWR-style pending source is invalid")
    parsed_source = _parse_source(source)
    _sha(pending.get("matched_training_sha256"), "AWR-style matched training")
    _sha(pending.get("progress_evidence_sha256"), "AWR-style progress evidence")
    _sha(pending.get("progress_evidence_receipt_sha256"), "AWR-style progress evidence receipt")
    _sha(pending.get("progress_evidence_mixture_id"), "AWR-style progress evidence mixture")
    _sha(pending.get("progress_evidence_mixture_manifest_sha256"), "AWR-style progress evidence mixture manifest")
    _sha(pending.get("winning_unweighted_report_sha256"), "AWR-style winning report")
    _sha(pending.get("winning_unweighted_seal_sha256"), "AWR-style winning seal")
    config = pending.get("awr_replay_config")
    if not isinstance(config, Mapping) or set(config) != {"temperature", "minimum", "maximum"}:
        raise ValueError("AWR-style replay configuration is invalid")
    parsed_config = AwrReplayConfig(**dict(config))
    if pending.get("awr_replay_config_sha256") != parsed_config.sha256:
        raise ValueError("AWR-style replay configuration digest mismatch")
    required = pending.get("required_weighted_request_set")
    if not isinstance(required, Mapping) or set(required) != {
        "kind", "must_differ_from_manifest_sha256", "must_differ_from_tree_sha256",
        "must_bind_matched_training_sha256", "must_bind_progress_evidence_sha256",
        "must_bind_progress_evidence_receipt_sha256", "must_bind_progress_evidence_mixture_id",
        "must_bind_progress_evidence_mixture_manifest_sha256",
        "must_bind_awr_replay_config_sha256",
    } or required.get("kind") != "runtime_request_set":
        raise ValueError("AWR-style weighted request-set contract is invalid")
    for key in set(required) - {"kind"}:
        _sha(required.get(key), f"AWR-style weighted request-set {key}")
    contract = pending.get("controller_admission_contract")
    if not isinstance(contract, Mapping) or set(contract) != {
        "kind", "requires_authenticated_readback_receipt", "requires_new_runtime_request_set",
        "lease_state_before_receipt", "lease_state_after_valid_receipt",
    } or contract.get("kind") != "lehome_awr_style_weighted_replay_admission" or contract.get("requires_authenticated_readback_receipt") is not True or contract.get("requires_new_runtime_request_set") is not True or contract.get("lease_state_before_receipt") != "PENDING_MATERIALIZATION" or contract.get("lease_state_after_valid_receipt") != "READY":
        raise ValueError("AWR-style controller admission contract is invalid")
    publication = pending.get("unweighted_checkpoint_publication")
    if not isinstance(publication, Mapping):
        raise ValueError("AWR-style parent publication is invalid")
    parsed_publication = parse_checkpoint_publication(publication)
    if parsed_publication.relative_path is None or parsed_publication.descriptor_relative_path is None or parsed_publication.experiment_id != pending["parent_experiment_id"]:
        raise ValueError("AWR-style parent publication is not a bound v2 receipt")
    if (
        required["must_differ_from_manifest_sha256"] != parsed_source["manifest_sha256"]
        or required["must_differ_from_tree_sha256"] != parsed_source["tree_sha256"]
        or required["must_bind_matched_training_sha256"] != pending["matched_training_sha256"]
        or required["must_bind_progress_evidence_sha256"] != pending["progress_evidence_sha256"]
        or required["must_bind_progress_evidence_receipt_sha256"] != pending["progress_evidence_receipt_sha256"]
        or required["must_bind_progress_evidence_mixture_id"] != pending["progress_evidence_mixture_id"]
        or required["must_bind_progress_evidence_mixture_manifest_sha256"] != pending["progress_evidence_mixture_manifest_sha256"]
        or required["must_bind_awr_replay_config_sha256"] != pending["awr_replay_config_sha256"]
    ):
        raise ValueError("AWR-style weighted request-set contract does not bind pending identity")
    return dict(pending)


@dataclass(frozen=True, slots=True)
class AwrStyleAdmission:
    """A runnable weighted job plus the receipt its controller must verify."""

    job: ExperimentJob
    receipt: Mapping[str, object]
    receipt_sha256: str
    pending_admission_sha256: str


def bind_weighted_runtime_request_set(
    parent: ExperimentJob,
    pending: Mapping[str, object],
    *,
    weighted_runtime_request_set: Mapping[str, object],
    materialization_receipt: Mapping[str, object],
) -> AwrStyleAdmission:
    """Bind a publisher-created weighted bundle, or reject it before leasing."""
    value = _validate_pending(pending)
    if value["parent_experiment_id"] != parent.experiment_id or value["matched_training_sha256"] != matched_training_sha256(parent):
        raise ValueError("AWR-style pending admission does not bind parent")
    if parent.mixture.sampling_strategy != "unweighted":
        raise ValueError("AWR-style parent is not unweighted")
    source = _parse_source(weighted_runtime_request_set)
    required = value["required_weighted_request_set"]
    assert isinstance(required, Mapping)
    if source["manifest_sha256"] == required["must_differ_from_manifest_sha256"] or source["tree_sha256"] == required["must_differ_from_tree_sha256"]:
        raise ValueError("AWR-style ablation may not reuse the unweighted runtime request set")
    child = deepcopy(dict(parent.raw))
    sources = child.get("data_sources")
    if not isinstance(sources, list):
        raise ValueError("AWR-style parent sources are invalid")
    replacements = 0
    for index, item in enumerate(sources):
        if isinstance(item, dict) and item.get("kind") == "runtime_request_set":
            sources[index] = dict(source)
            replacements += 1
    if replacements != 1:
        raise ValueError("AWR-style parent runtime request set is ambiguous")
    mixture = child.get("mixture")
    publication = child.get("publication")
    dependencies = child.get("dependencies")
    if not isinstance(mixture, dict) or not isinstance(publication, dict) or not isinstance(dependencies, list):
        raise ValueError("AWR-style parent contract is invalid")
    mixture["sampling_strategy"] = "awr_style_weighted_replay"
    publication["prefix"] = str(publication["prefix"]) + "-awr-style"
    # The evidence/config are explicit immutable admission fields, not opaque
    # dependency hashes that a generic recovery endpoint could accidentally
    # satisfy.
    child["admission"] = {
        "kind": "awr_style_weighted_replay",
        "pending_admission_sha256": pending_admission_sha256(value),
        "matched_training_sha256": value["matched_training_sha256"],
        "progress_evidence_sha256": value["progress_evidence_sha256"],
        "progress_evidence_receipt_sha256": value["progress_evidence_receipt_sha256"],
        "progress_evidence_mixture_id": value["progress_evidence_mixture_id"],
        "progress_evidence_mixture_manifest_sha256": value["progress_evidence_mixture_manifest_sha256"],
        "awr_replay_config_sha256": value["awr_replay_config_sha256"],
        "winning_unweighted_experiment_id": parent.experiment_id,
        "winning_unweighted_report_sha256": value["winning_unweighted_report_sha256"],
        "winning_unweighted_seal_sha256": value["winning_unweighted_seal_sha256"],
    }
    child.pop("experiment_id", None)
    child["experiment_id"] = experiment_identity(child)
    weighted_job = _parse(child)
    if matched_training_sha256(weighted_job) != value["matched_training_sha256"]:
        raise ValueError("AWR-style ablation changed a matched training field")
    receipt = dict(materialization_receipt)
    expected_pending = pending_admission_sha256(value)
    validate_awr_style_materialization_receipt(weighted_job, receipt)
    if receipt.get("pending_admission_sha256") != expected_pending:
        raise ValueError("AWR-style materialization receipt does not bind pending child")
    return AwrStyleAdmission(
        job=weighted_job,
        receipt=receipt,
        receipt_sha256=canonical_json_sha256(receipt),
        pending_admission_sha256=expected_pending,
    )


def validate_awr_style_materialization_receipt(
    job: ExperimentJob,
    receipt: Mapping[str, object],
) -> str:
    """Validate the exact authenticated receipt the controller must consume."""
    if job.admission.get("kind") != "awr_style_weighted_replay" or job.mixture.sampling_strategy != "awr_style_weighted_replay":
        raise ValueError("AWR-style materialization receipt requires a weighted job")
    value = dict(receipt)
    if set(value) != _RECEIPT_FIELDS or value.get("schema_version") != 1 or value.get("kind") != "lehome_awr_style_weighted_request_set_receipt" or value.get("readback_verified") is not True:
        raise ValueError("AWR-style materialization receipt is invalid")
    source = _parse_source(value.get("weighted_runtime_request_set", {})) if isinstance(value.get("weighted_runtime_request_set"), Mapping) else None
    expected_source = _source_document(_runtime_source(job))
    if source != expected_source:
        raise ValueError("AWR-style materialization receipt does not bind request set")
    for field in (
        "pending_admission_sha256", "matched_training_sha256", "progress_evidence_sha256",
        "progress_evidence_receipt_sha256", "progress_evidence_mixture_id",
        "progress_evidence_mixture_manifest_sha256", "awr_replay_config_sha256",
        "winning_unweighted_experiment_id", "winning_unweighted_report_sha256",
        "winning_unweighted_seal_sha256",
    ):
        if value.get(field) != job.admission.get(field):
            raise ValueError("AWR-style materialization receipt does not bind weighted job admission")
    if value.get("child_runtime_profile_sha256") != runtime_profile_sha256(job):
        raise ValueError("AWR-style materialization receipt does not bind child runtime profile")
    _sha(value.get("readback_receipt_sha256"), "AWR-style readback receipt")
    _sha(value.get("authenticated_principal_sha256"), "AWR-style authenticated principal")
    return canonical_json_sha256(value)
