"""Rank authenticated moment-of-ruin snapshots into a hard-state matrix.

This does not train on failed actions or terminal states. It emits only
checksum-bound CPU-cloth continuation snapshots immediately before a verified
reward drawdown so a later worker can keep a formally successful recovery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "source" / "lehome"))

from lehome.flywheel.hard_mining import FailureEvidence, rank_failures


_DROP_THRESHOLD = 0.12
_MINIMUM_PEAK = 0.25
_MINIMUM_RECOVERY_PEAK = 0.35
_CPU_CLOTH_AUTHORITY = "usd_local_points_v1"
_HARD_STATE_REPLAY_KIND = "verified_hard_state_moment_of_ruin_v1"


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


def _moment_of_ruin(
    annotations: list[dict], continuation_root: Path
) -> tuple[Path, dict] | None:
    available = []
    if continuation_root.is_dir() and not continuation_root.is_symlink():
        for path in continuation_root.iterdir():
            match = re.fullmatch(r"([0-9]{6})\.json", path.name)
            if match is not None and path.is_file() and not path.is_symlink():
                available.append((int(match.group(1)), path))
    if not available:
        return None
    available.sort()

    peak_progress = 0.0
    peak_step = -1
    for row in annotations:
        reward, step = row.get("reward"), row.get("step")
        if (
            isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not math.isfinite(reward)
            or isinstance(step, bool)
            or not isinstance(step, int)
            or step < 0
        ):
            continue
        progress = min(max(float(reward), 0.0), 1.0)
        if progress > peak_progress:
            peak_progress, peak_step = progress, step
            continue
        progress_drop = peak_progress - progress
        if peak_progress < _MINIMUM_PEAK or progress_drop < _DROP_THRESHOLD:
            continue
        candidates = [(snapshot_step, path) for snapshot_step, path in available if snapshot_step <= step]
        if not candidates:
            return None
        restore_step, restore_path = candidates[-1]
        return restore_path, {
            "signal": "dense_reward_proxy_no_success_head",
            "peak_progress": peak_progress,
            "peak_step": peak_step,
            "detection_step": step,
            "restore_step": restore_step,
            "progress_drop": progress_drop,
            "drop_threshold": _DROP_THRESHOLD,
            "minimum_peak": _MINIMUM_PEAK,
        }
    return None


def _cpu_restore_contract(path: Path, *, restore_step: int, garment: str) -> dict | None:
    """Authenticate one CPU continuation without accepting CUDA cloth state."""

    if path.is_symlink() or not path.is_file() or restore_step <= 0 or restore_step % 16:
        return None
    try:
        payload_bytes = path.read_bytes()
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    randomization = payload.get("randomization") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 3
        or payload.get("cloth_state_authority") != _CPU_CLOTH_AUTHORITY
        or payload.get("garment_name") != garment
        or not isinstance(randomization, dict)
        or randomization.get("strategy") != "canonical"
        or randomization.get("continuation_step") != restore_step
    ):
        return None
    return {
        "restore_snapshot": str(path),
        "restore_snapshot_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "restore_snapshot_cloth_frame": _CPU_CLOTH_AUTHORITY,
        "restore_snapshot_step": restore_step,
    }


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
        moment = _moment_of_ruin(annotations, episode_path.parent / "snapshots" / "continuations")
        restore_contract = (
            _cpu_restore_contract(moment[0], restore_step=moment[1]["restore_step"], garment=garment)
            if moment is not None
            else None
        )
        rows.append(
            {
                "episode_id": episode.get("episode_id") or episode_path.parent.name,
                "episode_path": str(episode_path),
                "terminal_path": str(terminal),
                "restore_snapshot": restore_contract["restore_snapshot"] if restore_contract else None,
                "restore_snapshot_sha256": (
                    restore_contract["restore_snapshot_sha256"] if restore_contract else None
                ),
                "restore_snapshot_cloth_frame": (
                    restore_contract["restore_snapshot_cloth_frame"] if restore_contract else None
                ),
                "restore_snapshot_step": (
                    restore_contract["restore_snapshot_step"] if restore_contract else None
                ),
                "moment_of_ruin": moment[1] if moment is not None else None,
                "category": category,
                "garment": garment,
                "seed": seed,
                "outcome": episode.get("outcome"),
                "terminal_reason": episode.get("terminal_reason"),
                "max_progress": max_progress,
                "stalled_steps": stalled,
                "length": len(annotations),
                "restorable": restore_contract is not None,
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
            bool(row.get("restorable", False)),
        )
        for row in rows
    ]
    by_id = {row["episode_id"]: row for row in rows}
    ranked = rank_failures(evidence, category_success=category_success)
    matrix = []
    for item in ranked:
        src = by_id[item.episode_id]
        moment = src.get("moment_of_ruin")
        moment_eligible = (
            isinstance(moment, dict)
            and bool(item.diagnostics["restorable"])
            and float(item.diagnostics["progress"]) >= _MINIMUM_RECOVERY_PEAK
        )
        if not moment_eligible:
            continue
        if len(matrix) >= limit:
            break
        selection_evidence = {
            "max_progress": item.diagnostics["max_progress"],
            "stall_fraction": item.diagnostics["stall_fraction"],
            "eligible_for_recovery": True,
            "terminal_near_miss_eligible": item.eligible_for_recovery,
        }
        if isinstance(moment, dict):
            selection_evidence["moment_of_ruin"] = moment
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
                "strategy": "canonical",
                "restore_snapshot": src["restore_snapshot"],
                "restore_snapshot_sha256": src["restore_snapshot_sha256"],
                "restore_snapshot_cloth_frame": src["restore_snapshot_cloth_frame"],
                "restore_snapshot_step": src["restore_snapshot_step"],
                "replay_kind": _HARD_STATE_REPLAY_KIND,
                "parent_episode_id": src["episode_id"],
                "lineage_id": src["episode_id"],
                "source_episode_id": src["episode_id"],
                "source_episode_path": src["episode_path"],
                "rank_score": item.score,
                "priority_reasons": list(item.priority_reasons),
                "selection_profile": "moment_of_ruin_reward_drop_v1",
                "selection_evidence": selection_evidence,
            }
        )
    return matrix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, action="append", required=True)
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
    campaign_roots: list[Path] = args.campaign_root
    if args.output is None and len(campaign_roots) != 1:
        parser.error("--output is required with multiple campaign roots")
    output = args.output or (campaign_roots[0] / "hard-state-nearmiss.json")
    category_success = {}
    for part in args.category_success.split(","):
        key, value = part.split("=", 1)
        category_success[key.strip()] = float(value)
    rows: list[dict] = []
    seen_roots: list[Path] = []
    seen_episodes: set[str] = set()
    for campaign_root in campaign_roots:
        if not campaign_root.is_absolute() or campaign_root.is_symlink() or not campaign_root.is_dir():
            raise ValueError("campaign root must be a real absolute directory")
        resolved_root = campaign_root.resolve(strict=True)
        if any(
            resolved_root == prior
            or resolved_root.is_relative_to(prior)
            or prior.is_relative_to(resolved_root)
            for prior in seen_roots
        ):
            raise ValueError("campaign roots overlap")
        seen_roots.append(resolved_root)
        for row in collect_failures(campaign_root):
            episode_id = str(row["episode_id"])
            if episode_id in seen_episodes:
                raise ValueError("campaign roots contain a duplicate episode")
            seen_episodes.add(episode_id)
            rows.append(row)
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
