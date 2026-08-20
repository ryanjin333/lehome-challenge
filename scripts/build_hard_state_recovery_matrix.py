"""Rank official failed terminals into a hard-state recovery matrix.

This does not train on failed actions. It only emits restorable fail
snapshots so a later worker can restore the terminal cloth/robot state
and keep a successful recovery.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "source" / "lehome"))

from lehome.flywheel.hard_mining import FailureEvidence, rank_failures


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _progress(episode: dict, annotations: list[dict]) -> tuple[float, int]:
    if not annotations:
        return 0.0, 0
    max_progress = 0.0
    peak_index = -1
    saw_step_reward = False
    for index, row in enumerate(annotations):
        reward = row.get("reward")
        if isinstance(reward, (int, float)) and not isinstance(reward, bool) and math.isfinite(reward):
            saw_step_reward = True
            progress = min(max(float(reward), 0.0), 1.0)
            if row.get("success") is True:
                progress = 1.0
            if progress >= max_progress:
                max_progress = progress
                peak_index = index
    # Older receipts may omit per-step rewards. Preserve their normalized return fallback.
    if not saw_step_reward:
        metrics = ((episode.get("outcome") or {}).get("metrics") or [{}])[-1] if isinstance(episode.get("outcome"), dict) else {}
        ret = metrics.get("return")
        if isinstance(ret, (int, float)):
            max_progress = min(max(float(ret) / 150.0, 0.0), 0.9)
            peak_index = len(annotations) - 1
    stalled = len(annotations) if peak_index < 0 else len(annotations) - peak_index - 1
    return max_progress, stalled


def collect_failures(campaign_root: Path) -> list[dict]:
    rows: list[dict] = []
    for episode_path in campaign_root.rglob("raw/*/episode.json"):
        if episode_path.is_symlink():
            continue
        episode = _load_json(episode_path)
        if episode.get("accepted_success") is True:
            continue
        terminal = episode_path.parent / "snapshots" / "terminal.json"
        annotations_path = episode_path.parent / "annotations.jsonl"
        if not terminal.is_file() or not annotations_path.is_file():
            continue
        identity = episode.get("identity") or {}
        category = identity.get("category")
        garment = identity.get("garment_name")
        seed = identity.get("seed")
        if not isinstance(category, str) or not isinstance(garment, str) or not isinstance(seed, int):
            continue
        annotations = [json.loads(line) for line in annotations_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        max_progress, stalled = _progress(episode, annotations)
        rows.append(
            {
                "episode_id": episode.get("episode_id") or episode_path.parent.name,
                "episode_path": str(episode_path),
                "terminal_path": str(terminal),
                "category": category,
                "garment": garment,
                "seed": seed,
                "outcome": episode.get("outcome"),
                "terminal_reason": episode.get("terminal_reason"),
                "max_progress": max_progress,
                "stalled_steps": stalled,
                "length": len(annotations),
                "restorable": True,
            }
        )
    return rows


def build_matrix(rows: list[dict], *, category_success: dict[str, float], limit: int) -> list[dict]:
    evidence = [
        FailureEvidence(
            row["episode_id"],
            row["category"],
            False,
            float(row["max_progress"]),
            int(row["stalled_steps"]),
            int(row["length"]),
            True,
        )
        for row in rows
    ]
    by_id = {row["episode_id"]: row for row in rows}
    ranked = rank_failures(evidence, category_success=category_success)
    matrix = []
    for item in (ranked_item for ranked_item in ranked if ranked_item.eligible_for_recovery):
        if len(matrix) >= limit:
            break
        src = by_id[item.episode_id]
        matrix.append(
            {
                "attempt_id": f"hard-state-{src['episode_id'][:12]}-seed-{src['seed']}",
                "trial_id": f"hard-state-{src['episode_id'][:12]}-seed-{src['seed']}",
                "garment": src["garment"],
                "garment_name": src["garment"],
                "category": src["category"],
                "release_stage": "seen",
                "difficulty": "hard_state",
                "seed": src["seed"],
                "restore_snapshot": src["terminal_path"],
                "source_episode_id": src["episode_id"],
                "source_episode_path": src["episode_path"],
                "rank_score": item.score,
                "priority_reasons": list(item.priority_reasons),
                "selection_profile": "near_miss_v1",
                "selection_evidence": {
                    "max_progress": item.diagnostics["max_progress"],
                    "stall_fraction": item.diagnostics["stall_fraction"],
                    "eligible_for_recovery": item.eligible_for_recovery,
                },
            }
        )
    return matrix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="output path (defaults to CAMPAIGN_ROOT/hard-state-nearmiss.json)",
    )
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument(
        "--category-success",
        default="top_long=0.4,top_short=0.25,pant_long=0.2,pant_short=0.65",
        help="comma k=v rates used only to rank which fails to restore first",
    )
    args = parser.parse_args(argv)
    output = args.output or (args.campaign_root / "hard-state-nearmiss.json")
    category_success = {}
    for part in args.category_success.split(","):
        key, value = part.split("=", 1)
        category_success[key.strip()] = float(value)
    rows = collect_failures(args.campaign_root)
    present = {row["category"] for row in rows}
    # Ranking requires every referenced category; drop unused floors.
    category_success = {key: value for key, value in category_success.items() if key in present}
    if rows and not category_success:
        category_success = {row["category"]: 0.0 for row in rows}
    matrix = build_matrix(rows, category_success=category_success, limit=args.limit)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"failures": len(rows), "selected": len(matrix), "output": str(output), "categories": sorted(present)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
