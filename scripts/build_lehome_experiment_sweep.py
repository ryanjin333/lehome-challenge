#!/usr/bin/env python3
"""Generate seven canonical initial jobs from explicit immutable bindings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from lehome_train.groot.experiment_job import dump_experiment_job
from lehome_train.groot.experiment_controller import validate_production_budget
from lehome_train.groot.experiment_manifest import APPROVED_ORIGINAL_12K_CHECKPOINT, batch64_quotas
from lehome_train.io import canonical_json_sha256


ARMS = (
    ("a", 100, 0, None), ("b", 95, 5, "ordinary_success"),
    ("c", 70, 30, "ordinary_success"), ("d", 95, 5, "recovery"),
    ("e", 90, 10, "recovery"), ("f", 85, 15, "recovery"),
    ("g", 80, 20, "recovery"),
)


def _artifact(request: Mapping[str, object], kind: str) -> dict[str, object]:
    artifacts = request.get("artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(artifacts.get(kind), Mapping):
        raise ValueError(f"request requires explicit immutable {kind} artifact binding")
    value = dict(artifacts[kind])
    if set(value) != {"kind", "repository", "revision", "prefix", "manifest_sha256", "tree_sha256"} or value["kind"] != kind:
        raise ValueError(f"{kind} artifact binding is malformed")
    return value


def build_initial_jobs(request: Mapping[str, object], output: Path) -> list[str]:
    template = request.get("template")
    request_sets = request.get("request_sets")
    dependency = request.get("recovery_dependency")
    if not isinstance(template, Mapping) or not isinstance(request_sets, Mapping) or type(dependency) is not str:
        raise ValueError("sweep request schema is invalid")
    budget = validate_production_budget(request.get("budget"))
    parent = template.get("parent_checkpoint")
    evaluation = template.get("evaluation")
    if (
        not isinstance(parent, Mapping)
        or dict(parent) != dict(APPROVED_ORIGINAL_12K_CHECKPOINT)
        or not isinstance(evaluation, Mapping)
        or evaluation.get("policy_digest") != APPROVED_ORIGINAL_12K_CHECKPOINT["artifact_sha256"]
    ):
        raise ValueError("sweep template must bind the approved original 12K checkpoint")
    bc = _artifact(request, "bc")
    ordinary = _artifact(request, "ordinary_success")
    recovery = _artifact(request, "recovery")
    output.mkdir(parents=True, exist_ok=True)
    emitted: list[str] = []
    documents: list[tuple[str, dict[str, object]]] = []
    for arm, bc_percent, added_percent, added_kind in ARMS:
        request_set = request_sets.get(arm)
        if not isinstance(request_set, Mapping):
            raise ValueError(f"request requires immutable runtime request-set binding for arm {arm}")
        runtime = dict(request_set)
        if set(runtime) != {"kind", "repository", "revision", "prefix", "manifest_sha256", "tree_sha256"} or runtime["kind"] != "runtime_request_set":
            raise ValueError("runtime request-set binding is malformed")
        document = json.loads(json.dumps(template))
        document["arm"] = arm
        document["data_sources"] = [bc, runtime] if added_kind is None else [bc, _artifact(request, added_kind), runtime]
        document["mixture"] = {"bc_percent": bc_percent, "added_percent": added_percent, "batch64_quotas": batch64_quotas({"bc": bc_percent, "rollout": added_percent, "dagger": 0}), "sampling_strategy": "unweighted"}
        document["training"]["target_step"] = 500
        document["publication"]["prefix"] = f"experiments/{arm}-500"
        document["dependencies"] = [] if added_kind != "recovery" else [dependency]
        job = dump_experiment_job(output / f"{arm}-500.json", document)
        emitted.append(job.experiment_id)
        documents.append((job.experiment_id, dict(job.raw)))
    documents.sort(key=lambda item: item[0])
    manifest_set_sha256 = canonical_json_sha256({"schema_version": 1, "jobs": documents})
    campaign = {"schema_version": 1, "kind": "lehome_recovery_sweep_v1", "gradient_step_ceiling": 7000, "tied_runner_gradient_step_ceiling": 8000, "original_12k_checkpoint": dict(APPROVED_ORIGINAL_12K_CHECKPOINT), "manifest_set_sha256": manifest_set_sha256, "jobs": emitted, **budget}
    (output / "campaign.json").write_text(json.dumps(campaign, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return emitted


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--request", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_initial_jobs(json.loads(args.request.read_text(encoding="utf-8")), args.output)


if __name__ == "__main__":
    main()
