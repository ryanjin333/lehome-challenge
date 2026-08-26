#!/usr/bin/env python3
"""Start the private experiment controller after local input validation."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from lehome_train.groot.experiment_controller import ExperimentController, validate_production_budget
from lehome_train.groot.experiment_deployment_gate import load_deployment_gate
from lehome_train.groot.experiment_service import ExperimentService
from lehome_train.groot.experiment_job import load_experiment_job
from lehome_train.groot.experiment_manifest import APPROVED_ORIGINAL_12K_CHECKPOINT, batch64_quotas


_BUDGET_KEYS = ("gpu_seconds_ceiling", "spend_ceiling", "estimated_gpu_seconds_per_step", "gpu_price_per_second")
_ARM_CONTRACTS = {
    "a": (100, 0, ("bc", "runtime_request_set")),
    "b": (95, 5, ("bc", "ordinary_success", "runtime_request_set")),
    "c": (70, 30, ("bc", "ordinary_success", "runtime_request_set")),
    "d": (95, 5, ("bc", "recovery", "runtime_request_set")),
    "e": (90, 10, ("bc", "recovery", "runtime_request_set")),
    "f": (85, 15, ("bc", "recovery", "runtime_request_set")),
    "g": (80, 20, ("bc", "recovery", "runtime_request_set")),
}


def validate_campaign(campaign: object) -> dict[str, float]:
    required = {"schema_version", "kind", "gradient_step_ceiling", "tied_runner_gradient_step_ceiling", "original_12k_checkpoint", "manifest_set_sha256", "jobs", *_BUDGET_KEYS}
    if not isinstance(campaign, dict) or set(campaign) != required or campaign.get("schema_version") != 1 or campaign.get("kind") != "lehome_recovery_sweep_v1" or campaign.get("gradient_step_ceiling") != 7000 or campaign.get("tied_runner_gradient_step_ceiling") != 8000 or campaign.get("original_12k_checkpoint") != dict(APPROVED_ORIGINAL_12K_CHECKPOINT) or not isinstance(campaign.get("jobs"), list) or len(campaign["jobs"]) != 7 or not all(isinstance(item, str) for item in campaign["jobs"]) or len(set(campaign["jobs"])) != 7:
        raise ValueError("campaign manifest is invalid")
    return validate_production_budget({key: campaign[key] for key in _BUDGET_KEYS})


def validate_initial_jobs(
    jobs: list[object],
    *,
    training_image_id: str | None = None,
    training_oci_digest: str | None = None,
    training_code_revision: str | None = None,
) -> None:
    if len(jobs) != 7 or {getattr(item, "arm", None) for item in jobs} != set(_ARM_CONTRACTS):
        raise ValueError("campaign must contain exactly the canonical A-G initial arms")
    first = jobs[0]
    stable_trainer = dict(first.trainer)
    stable_matrix = (first.evaluation.matrix_id, first.evaluation.matrix_sha256)
    stable_publication = (first.publication.checkpoint_repository, first.publication.result_repository)
    stable_seed = first.training.seed
    stable_bc = first.data_sources[0]
    ordinary_source = None
    recovery_source = None
    recovery_dependencies: set[str] = set()
    for item in jobs:
        bc_percent, added_percent, source_kinds = _ARM_CONTRACTS[item.arm]
        expected_quotas = batch64_quotas({"bc": bc_percent, "rollout": added_percent, "dagger": 0})
        if (
            getattr(item, "admission", {}).get("kind") != "initial"
            or getattr(item, "training", None) is None
            or (item.training.action_horizon, item.training.batch_size, item.training.target_step, item.training.save_steps) != (16, 64, 500, 500)
            or item.training.seed != stable_seed
            or dict(getattr(item, "parent_checkpoint", {})) != dict(APPROVED_ORIGINAL_12K_CHECKPOINT)
            or getattr(item, "evaluation", None) is None
            or item.evaluation.policy_digest != APPROVED_ORIGINAL_12K_CHECKPOINT["artifact_sha256"]
            or (item.evaluation.matrix_id, item.evaluation.matrix_sha256) != stable_matrix
            or dict(item.trainer) != stable_trainer
            or (training_image_id is not None and item.trainer["image_id"] != training_image_id)
            or (training_oci_digest is not None and item.trainer["oci_digest"] != training_oci_digest)
            or (training_code_revision is not None and item.trainer["code_revision"] != training_code_revision)
            or (item.publication.checkpoint_repository, item.publication.result_repository) != stable_publication
            or item.publication.prefix != f"experiments/{item.arm}-500"
            or (item.mixture.bc_percent, item.mixture.added_percent, item.mixture.sampling_strategy) != (bc_percent, added_percent, "unweighted")
            or dict(item.mixture.batch64_quotas) != expected_quotas
            or tuple(source.kind for source in item.data_sources) != source_kinds
            or item.data_sources[0] != stable_bc
        ):
            raise ValueError("campaign job does not match the canonical A-G initial contract or deployment gate")
        if item.arm in {"a", "b", "c"}:
            if item.dependencies:
                raise ValueError("ordinary initial arms cannot declare recovery dependencies")
        else:
            if len(item.dependencies) != 1:
                raise ValueError("recovery initial arms require one immutable dependency")
            recovery_dependencies.add(item.dependencies[0])
        if item.arm in {"b", "c"}:
            ordinary_source = item.data_sources[1] if ordinary_source is None else ordinary_source
            if item.data_sources[1] != ordinary_source:
                raise ValueError("ordinary initial arms must share one immutable source")
        if item.arm in {"d", "e", "f", "g"}:
            recovery_source = item.data_sources[1] if recovery_source is None else recovery_source
            if item.data_sources[1] != recovery_source:
                raise ValueError("recovery initial arms must share one immutable source")
    if len(recovery_dependencies) != 1:
        raise ValueError("recovery initial arms must share one immutable dependency")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True); parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, required=True); parser.add_argument("--audit-log", type=Path, required=True); parser.add_argument("--bind", required=True)
    parser.add_argument("--deployment-gate", type=Path, required=True); parser.add_argument("--deployment-gate-sha256", required=True)
    args = parser.parse_args()
    for path in (args.database, args.token_file, args.manifests, args.audit_log, args.deployment_gate):
        if not path.is_absolute(): raise ValueError("controller paths must be absolute")
    campaign_path = args.manifests / "campaign.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    budget = validate_campaign(campaign)
    jobs = [load_experiment_job(path) for path in sorted(args.manifests.glob("*.json")) if path.name != "campaign.json"]
    if {job.experiment_id for job in jobs} != set(campaign["jobs"]):
        raise ValueError("campaign job set is incomplete or contains an unexpected manifest")
    deployment = load_deployment_gate(args.deployment_gate, args.deployment_gate_sha256)
    validate_initial_jobs(
        jobs,
        training_image_id=deployment.training_image_id,
        training_oci_digest=deployment.training_oci_digest,
        training_code_revision=deployment.training_code_revision,
    )
    controller = ExperimentController(
        args.database,
        gradient_step_ceiling=campaign["gradient_step_ceiling"],
        tied_runner_gradient_step_ceiling=campaign["tied_runner_gradient_step_ceiling"],
        recovery_collection_admitted=deployment.recovery_collection_admitted,
        **budget,
    )
    controller.add_jobs(jobs, manifest_set_sha256=campaign["manifest_set_sha256"])
    controller.reconcile_pending_candidates(time.time_ns())
    host, port = args.bind.rsplit(":", 1)
    service = ExperimentService((host, int(port)), controller, args.token_file)
    args.audit_log.parent.mkdir(parents=True, exist_ok=True); args.audit_log.write_text("controller ready\n", encoding="utf-8")
    service.serve_forever()
if __name__ == "__main__": main()
