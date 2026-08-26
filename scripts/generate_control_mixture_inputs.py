#!/usr/bin/env python3
"""Generate immutable pure-BC and ordinary-success control mixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from lehome_train.groot.experiment_manifest import (
    batch64_quotas,
    load_experiment_manifest,
    runtime_profile_document,
)
from lehome_train.io import canonical_json_bytes, canonical_json_sha256, sha256_file


RATIOS = {"a": (100, 0), "b": (95, 5)}


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))
    path.chmod(0o600)


def load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def control_schedule(arm: str) -> dict[str, object]:
    try:
        bc, ordinary = RATIOS[arm]
    except KeyError as error:
        raise ValueError(f"unsupported control arm: {arm}") from error
    return {
        "bc_percent": bc,
        "ordinary_percent": ordinary,
        "batch64_quotas": batch64_quotas({"bc": bc, "rollout": ordinary, "dagger": 0}),
    }


def control_selections(ranges: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    selections = [dict(row) for row in ranges]
    if not selections:
        raise ValueError("base runtime plan has no windows")
    kinds = {row.get("source_kind") for row in selections}
    if not {"organizer", "flywheel"}.issubset(kinds):
        raise ValueError("control source must retain organizer and ordinary rollout provenance")
    source_order = {"organizer": 0, "flywheel": 1}
    return sorted(
        selections,
        key=lambda row: (
            source_order.get(str(row.get("source_kind")), 2),
            str(row.get("raw_episode_id")),
            int(row.get("raw_frame_start", -1)),
            int(row.get("raw_frame_stop", -1)),
        ),
    )


def lineage_hashes(selections: list[dict[str, object]]) -> dict[str, str]:
    values: dict[str, list[str]] = {"train": [], "validation": []}
    for row in selections:
        split = str(row.get("split"))
        if split not in values:
            raise ValueError("control window has an invalid split")
        kind = "bc" if row.get("source_kind") == "organizer" else "rollout"
        values[split].append(f"{kind}:{row['raw_episode_id']}")
    return {
        f"{split}_sha256": canonical_json_sha256(
            {"split": split, "lineage_ids": sorted(lineages)}
        )
        for split, lineages in values.items()
    }


def generate_arm(
    *,
    arm: str,
    plan: dict[str, object],
    selections: list[dict[str, object]],
    lineages: dict[str, str],
    selected_document: dict[str, object],
    source_lineage_sha256: str,
    source_publications: dict[str, object],
    base_experiment: dict[str, object],
    output: Path,
) -> dict[str, object]:
    schedule = control_schedule(arm)
    bc = int(schedule["bc_percent"])
    ordinary = int(schedule["ordinary_percent"])
    root = output / arm
    selected_path = root / "selected-150.json"
    write(selected_path, selected_document)

    experiment = json.loads(json.dumps(base_experiment))
    experiment["mixture_weights"] = {"bc": bc, "dagger": 0, "rollout": ordinary}
    experiment["lineage"] = dict(lineages)
    experiment["mixture_manifest_sha256"] = "0" * 64
    scratch = root / ".experiment-profile.json"
    write(scratch, experiment)
    parsed = load_experiment_manifest(scratch)

    input_bindings = plan.get("input_bindings")
    if not isinstance(input_bindings, dict):
        raise ValueError("base runtime plan input bindings are malformed")
    bindings = dict(input_bindings)
    bindings.update(
        {
            "selected_bindings_sha256": sha256_file(selected_path),
            "source_lineage_sha256": source_lineage_sha256,
            "experiment_config_sha256": canonical_json_sha256(runtime_profile_document(parsed)),
            "runtime_schedule": {
                "action_horizon": 16,
                "batch_size": 64,
                "bc": bc,
                "rollout": ordinary,
            },
        }
    )
    plan_value = {
        "schema_version": 1,
        "kind": "runtime_mixture_plan",
        "input_bindings": bindings,
        "selected_frame_ranges": selections,
    }
    plan_sha256 = canonical_json_sha256(plan_value)
    plan_value["sha256"] = plan_sha256
    plan_state = {
        "schema_version": 1,
        "kind": "runtime_mixture_plan_state",
        "plan": plan_value,
        "plan_sha256": plan_sha256,
    }
    plan_path = root / "runtime-plan.json"
    write(plan_path, plan_state)

    experiment["mixture_manifest_sha256"] = plan_sha256
    experiment_path = root / "experiment-manifest.json"
    write(experiment_path, experiment)
    scratch.unlink()
    parsed = load_experiment_manifest(experiment_path)
    if canonical_json_sha256(runtime_profile_document(parsed)) != bindings["experiment_config_sha256"]:
        raise ValueError("final experiment profile drifted after plan binding")

    write(
        root / "build-request.json",
        {
            "schema_version": 1,
            "command": "build-runtime-mixture",
            "arguments": {
                "organizer_root": "/sources/organizer",
                "campaign_root": "/sources/rollout",
                "source_publications": "/work/source-publications.json",
                "selected_bindings": f"/work/{arm}/selected-150.json",
                "plan_state": f"/work/{arm}/runtime-plan.json",
                "destination": f"/work/{arm}/built",
                "experiment_manifest": f"/work/{arm}/experiment-manifest.json",
            },
        },
    )
    write(
        root / "publish-request.json",
        {
            "schema_version": 1,
            "command": "publish-runtime-mixture",
            "arguments": {
                "pending_root": f"/work/{arm}/built",
                "revision": "main",
                "receipt_path": f"/work/{arm}/publication-receipt.json",
            },
        },
    )
    write(
        root / "finalize-request.json",
        {
            "schema_version": 1,
            "command": "finalize-runtime-mixture",
            "arguments": {
                "pending_root": f"/work/{arm}/built",
                "publication_receipt": f"/work/{arm}/publication-receipt.json",
                "destination": f"/work/{arm}/final",
                "deployment_receipt_path": f"/work/{arm}/deployment-receipt.json",
                "source_mounts": {
                    "organizer": "/sources/organizer",
                    "rollout": "/sources/rollout",
                },
                "revision": "main",
            },
        },
    )
    summary = {
        "arm": arm,
        "bc_percent": bc,
        "ordinary_percent": ordinary,
        "batch64_quotas": schedule["batch64_quotas"],
        "bc_window_count": sum(row.get("source_kind") == "organizer" for row in selections),
        "ordinary_window_count": sum(row.get("source_kind") == "flywheel" for row in selections),
        "plan_sha256": plan_sha256,
        "experiment_manifest_sha256": parsed.identity_sha256,
        "selected_bindings_sha256": sha256_file(selected_path),
    }
    write(root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--source-lineage", type=Path, required=True)
    parser.add_argument("--source-publications", type=Path, required=True)
    parser.add_argument("--base-experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", action="append", choices=tuple(RATIOS))
    args = parser.parse_args()

    state = load(args.base_plan)
    plan = state.get("plan")
    if not isinstance(plan, dict) or not isinstance(plan.get("selected_frame_ranges"), list):
        raise ValueError("base runtime plan is malformed")
    selections = control_selections(plan["selected_frame_ranges"])
    selected_document = load(args.selected)
    publications = load(args.source_publications)
    publication_rows = publications.get("sources")
    if not isinstance(publication_rows, list):
        raise ValueError("source publication descriptor is malformed")
    for row in publication_rows:
        if not isinstance(row, dict) or not isinstance(row.get("readback_receipt_path"), str):
            raise ValueError("source publication row is malformed")
        row["readback_receipt_path"] = "/prepared/receipts/" + Path(row["readback_receipt_path"]).name
    write(args.output / "source-publications.json", publications)

    base_experiment = load(args.base_experiment)
    lineages = lineage_hashes(selections)
    source_lineage_sha256 = sha256_file(args.source_lineage)
    summaries = [
        generate_arm(
            arm=arm,
            plan=plan,
            selections=selections,
            lineages=lineages,
            selected_document=selected_document,
            source_lineage_sha256=source_lineage_sha256,
            source_publications=publications,
            base_experiment=base_experiment,
            output=args.output,
        )
        for arm in (args.arm or list(RATIOS))
    ]
    write(args.output / "summary.json", {"schema_version": 1, "arms": summaries})
    print(json.dumps({"output": str(args.output), "arms": summaries}, sort_keys=True))


if __name__ == "__main__":
    main()
