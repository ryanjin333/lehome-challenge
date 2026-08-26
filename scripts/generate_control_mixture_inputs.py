#!/usr/bin/env python3
"""Generate immutable control and targeted 90/10 runtime mixtures."""

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


RATIOS = {
    "a": (100, 0, "none"),
    "b": (95, 5, "ordinary_success"),
    "success-replay": (90, 10, "success_replay"),
    "hard-state": (90, 10, "hard_state"),
}
DEFAULT_ARMS = ("a", "b")
_CATEGORIES = ("top_long", "top_short", "pant_long", "pant_short")
_RECOVERY_CONTRACT = "authenticated_cloth_snapshot_at_fresh_h16_next_action_boundary_v2"


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
        bc, added, _source = RATIOS[arm]
    except KeyError as error:
        raise ValueError(f"unsupported control arm: {arm}") from error
    return {
        "bc_percent": bc,
        "ordinary_percent": added,
        "batch64_quotas": batch64_quotas({"bc": bc, "rollout": added, "dagger": 0}),
    }


def arm_source(arm: str) -> str:
    try:
        return RATIOS[arm][2]
    except KeyError as error:
        raise ValueError(f"unsupported control arm: {arm}") from error


def selected_arms(arms: Iterable[str] | None) -> tuple[str, ...]:
    return tuple(arms) if arms else DEFAULT_ARMS


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


def _episode_category(campaign_root: Path, episode_id: str) -> str:
    episode = load(campaign_root / "raw" / episode_id / "episode.json")
    identity = episode.get("identity")
    category = identity.get("category") if isinstance(identity, dict) else None
    if category not in _CATEGORIES:
        raise ValueError("targeted rollout episode category is invalid")
    return str(category)


def _round_robin(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_episode: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        episode_id = str(row["raw_episode_id"])
        by_episode.setdefault(episode_id, []).append(row)
    queues = {
        episode_id: sorted(
            values,
            key=lambda row: (int(row["raw_frame_start"]), int(row["raw_frame_stop"])),
        )
        for episode_id, values in by_episode.items()
    }
    interleaved: list[dict[str, object]] = []
    offset = 0
    while True:
        appended = False
        for episode_id in sorted(queues):
            values = queues[episode_id]
            if offset < len(values):
                interleaved.append(values[offset])
                appended = True
        if not appended:
            return interleaved
        offset += 1


def _success_replay_windows(
    ranges: list[dict[str, object]], campaign_root: Path
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {category: [] for category in _CATEGORIES}
    categories: dict[str, str] = {}
    for row in ranges:
        if row.get("source_kind") != "flywheel":
            continue
        episode_id = str(row.get("raw_episode_id"))
        category = categories.setdefault(
            episode_id, _episode_category(campaign_root, episode_id)
        )
        grouped[category].append(row)
    if any(not grouped[category] for category in _CATEGORIES):
        raise ValueError("balanced success replay requires every garment category")
    target = min(len(grouped[category]) for category in _CATEGORIES)
    return [
        row
        for category in _CATEGORIES
        for row in _round_robin(grouped[category])[:target]
    ]


def _hard_state_windows(
    ranges: list[dict[str, object]], campaign_root: Path, recovery_audit: Path
) -> list[dict[str, object]]:
    audit = load(recovery_audit)
    recoveries = audit.get("selected_recoveries")
    if (
        audit.get("schema_version") != 4
        or audit.get("kind") != "lehome_successful_recovery_audit"
        or audit.get("continuation_contract") != _RECOVERY_CONTRACT
        or not isinstance(recoveries, list)
        or not recoveries
    ):
        raise ValueError("hard-state source requires the authenticated recovery audit")
    available: dict[tuple[str, int, int], dict[str, object]] = {}
    templates: dict[str, dict[str, object]] = {}
    for row in ranges:
        if row.get("source_kind") != "flywheel":
            continue
        episode_id = str(row.get("raw_episode_id"))
        templates.setdefault(episode_id, row)
        key = (
            episode_id,
            int(row.get("raw_frame_start", -1)),
            int(row.get("raw_frame_stop", -1)),
        )
        if key in available:
            raise ValueError("hard-state base plan contains duplicate rollout windows")
        available[key] = row
    grouped: dict[str, list[dict[str, object]]] = {}
    seen: set[tuple[str, int, int]] = set()
    verified_annotations: set[str] = set()
    for recovery in recoveries:
        if not isinstance(recovery, dict):
            raise ValueError("hard-state recovery row is malformed")
        episode_id, category, windows = (
            recovery.get("source_episode_id"),
            recovery.get("category"),
            recovery.get("h16_ranges"),
        )
        if (
            type(episode_id) is not str
            or category not in _CATEGORIES
            or _episode_category(campaign_root, episode_id) != category
            or not isinstance(windows, list)
            or not windows
        ):
            raise ValueError("hard-state recovery lineage is malformed")
        for window in windows:
            if not isinstance(window, dict):
                raise ValueError("hard-state recovery window is malformed")
            start, stop, frame_ids = (
                window.get("start"), window.get("stop"), window.get("frame_ids")
            )
            if (
                type(start) is not int
                or type(stop) is not int
                or stop - start != 16
                or frame_ids != list(range(start, stop))
            ):
                raise ValueError("hard-state recovery window is not an exact h16 range")
            key = (episode_id, start, stop)
            row = available.get(key)
            if key in seen:
                raise ValueError("hard-state recovery window is duplicated in the audit")
            if row is None:
                template = templates.get(episode_id)
                annotation_count = recovery.get("annotation_count")
                artifacts = recovery.get("source_artifacts")
                annotations_sha256 = (
                    artifacts.get("annotations_sha256")
                    if isinstance(artifacts, dict)
                    else None
                )
                annotations = campaign_root / "raw" / episode_id / "annotations.jsonl"
                if (
                    template is None
                    or type(annotation_count) is not int
                    or start < 0
                    or stop > annotation_count
                    or type(annotations_sha256) is not str
                    or len(annotations_sha256) != 64
                    or any(character not in "0123456789abcdef" for character in annotations_sha256)
                    or annotations.is_symlink()
                    or not annotations.is_file()
                ):
                    raise ValueError("off-grid hard-state recovery lacks authenticated source data")
                if episode_id not in verified_annotations:
                    with annotations.open("rb") as stream:
                        actual_count = sum(1 for _line in stream)
                    if (
                        actual_count != annotation_count
                        or sha256_file(annotations) != annotations_sha256
                    ):
                        raise ValueError("off-grid hard-state recovery source authentication failed")
                    verified_annotations.add(episode_id)
                row = dict(template)
                row.update(
                    {
                        "raw_frame_start": start,
                        "raw_frame_stop": stop,
                        "raw_frame_ids": [str(frame) for frame in range(start, stop)],
                    }
                )
            seen.add(key)
            grouped.setdefault(str(category), []).append(row)
    required = {"top_long", "top_short", "pant_long"}
    if not required.issubset(grouped):
        raise ValueError("hard-state audit lacks one of the weak garment categories")
    target = min(len(rows) for rows in grouped.values())
    return [
        row
        for category in sorted(grouped)
        for row in _round_robin(grouped[category])[:target]
    ]


def targeted_selections(
    ranges: Iterable[dict[str, object]],
    *,
    campaign_root: Path,
    mode: str,
    recovery_audit: Path | None,
) -> list[dict[str, object]]:
    values = [dict(row) for row in ranges]
    organizer = [row for row in values if row.get("source_kind") == "organizer"]
    if not organizer:
        raise ValueError("targeted mixture has no organizer BC windows")
    if mode == "success-replay":
        if recovery_audit is not None:
            raise ValueError("success replay does not accept a recovery audit")
        targeted = _success_replay_windows(values, campaign_root)
    elif mode == "hard-state":
        if recovery_audit is None:
            raise ValueError("hard-state selection requires a recovery audit")
        targeted = _hard_state_windows(values, campaign_root, recovery_audit)
    else:
        raise ValueError("targeted mixture mode is invalid")
    return control_selections([*organizer, *targeted])


def bind_source_bundles(
    experiment: dict[str, object],
    *,
    plan: dict[str, object],
    source_publications: dict[str, object],
) -> None:
    bindings = plan.get("input_bindings")
    rows = source_publications.get("sources")
    if not isinstance(bindings, dict) or not isinstance(rows, list):
        raise ValueError("runtime source bindings are malformed")
    publications = {
        row.get("source_id"): row
        for row in rows
        if isinstance(row, dict) and row.get("source_id") in {"organizer", "rollout"}
    }
    if set(publications) != {"organizer", "rollout"}:
        raise ValueError("runtime source publications are incomplete")
    fields = {
        "bc_bundle": (
            "organizer", "organizer_manifest_sha256", "organizer_tree_sha256"
        ),
        "rollout_bundle": (
            "rollout", "campaign_receipt_sha256", "campaign_tree_sha256"
        ),
    }
    for bundle_name, (source_id, manifest_key, tree_key) in fields.items():
        bundle, publication = experiment.get(bundle_name), publications[source_id]
        if not isinstance(bundle, dict):
            raise ValueError("experiment source bundle is malformed")
        if any(type(publication.get(key)) is not str for key in ("repository", "revision", "prefix")):
            raise ValueError("runtime source publication identity is malformed")
        if any(type(bindings.get(key)) is not str for key in (manifest_key, tree_key)):
            raise ValueError("runtime source identity binding is malformed")
        bundle.update(
            {
                "repository": publication["repository"],
                "revision": publication["revision"],
                "prefix": publication["prefix"],
                "manifest_sha256": bindings[manifest_key],
                "tree_sha256": bindings[tree_key],
            }
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
    added = int(schedule["ordinary_percent"])
    root = output / arm
    selected_path = root / "selected-150.json"
    write(selected_path, selected_document)

    experiment = json.loads(json.dumps(base_experiment))
    bind_source_bundles(
        experiment, plan=plan, source_publications=source_publications
    )
    experiment["mixture_weights"] = {"bc": bc, "dagger": 0, "rollout": added}
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
                "rollout": added,
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
        "added_percent": added,
        "added_source": arm_source(arm),
        "batch64_quotas": schedule["batch64_quotas"],
        "bc_window_count": sum(row.get("source_kind") == "organizer" for row in selections),
        "added_window_count": sum(row.get("source_kind") == "flywheel" for row in selections),
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
    parser.add_argument(
        "--targeted-mode", choices=("success-replay", "hard-state")
    )
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--recovery-audit", type=Path)
    args = parser.parse_args()

    state = load(args.base_plan)
    plan = state.get("plan")
    if not isinstance(plan, dict) or not isinstance(plan.get("selected_frame_ranges"), list):
        raise ValueError("base runtime plan is malformed")
    arms = selected_arms(args.arm)
    if args.targeted_mode is None:
        if args.campaign_root is not None or args.recovery_audit is not None:
            raise ValueError("targeted source inputs require a targeted mode")
        selections = control_selections(plan["selected_frame_ranges"])
    else:
        if arms != (args.targeted_mode,) or args.campaign_root is None:
            raise ValueError("targeted mode requires its one matching arm and campaign root")
        selections = targeted_selections(
            plan["selected_frame_ranges"],
            campaign_root=args.campaign_root,
            mode=args.targeted_mode,
            recovery_audit=args.recovery_audit,
        )
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
        for arm in arms
    ]
    write(args.output / "summary.json", {"schema_version": 1, "arms": summaries})
    print(json.dumps({"output": str(args.output), "arms": summaries}, sort_keys=True))


if __name__ == "__main__":
    main()
