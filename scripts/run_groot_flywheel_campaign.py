"""Resumable, local-only supervisor for isolated GR00T rollout processes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence

from lehome.flywheel.artifacts import verify_episode
from lehome.flywheel.capacity import CapacitySample, choose_worker_count
from lehome.flywheel.matrix import Trial


@dataclass(frozen=True, slots=True)
class CampaignState:
    output_root: Path
    trial_ids: tuple[str, ...]


def pending_trial_ids(state: CampaignState) -> tuple[str, ...]:
    """Resume only after a terminal episode and every checksum verify again."""
    pending: list[str] = []
    for trial_id in state.trial_ids:
        try:
            episode = verify_episode(state.output_root / "raw" / trial_id)
            if not isinstance(episode.get("terminal_reason"), str) or not episode["terminal_reason"]:
                raise ValueError("terminal reason is missing")
        except ValueError:
            pending.append(trial_id)
    return tuple(pending)


def _write_heartbeat(path: Path, *, worker_id: int, trial_id: str, state: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"worker_id": worker_id, "trial_id": trial_id, "state": state, "monotonic_ns": time.monotonic_ns()}, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _trial_command(args: argparse.Namespace, trial: Trial) -> list[str]:
    return [
        sys.executable, "-m", "scripts.run_groot_flywheel_trial", "--policy-path", str(args.policy_path),
        "--policy-revision-file", str(args.policy_revision_file), "--garment", trial.garment_name,
        "--seed", str(trial.seed), "--episode-id", trial.trial_id, "--output-root", str(args.output_root),
        "--max-steps", str(args.max_steps), "--headless",
    ]


def _run_one_worker(args: argparse.Namespace, *, worker_id: int, trial: Trial) -> int:
    worker_root = args.output_root / "workers" / f"worker-{worker_id:02d}"
    heartbeat = worker_root / "heartbeat.json"
    log_path = worker_root / f"{trial.trial_id}.log"
    _write_heartbeat(heartbeat, worker_id=worker_id, trial_id=trial.trial_id, state="started")
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(_trial_command(args, trial), stdout=log, stderr=subprocess.STDOUT)
        try:
            returncode = process.wait(timeout=args.worker_timeout_seconds)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            _write_heartbeat(heartbeat, worker_id=worker_id, trial_id=trial.trial_id, state="timeout")
            return 124
    _write_heartbeat(heartbeat, worker_id=worker_id, trial_id=trial.trial_id, state="terminal")
    return returncode


def _validate_sweep(values: str) -> tuple[int, ...]:
    requested = tuple(int(value) for value in values.split(",") if value)
    legal = (1, 2, 4, 6, 8)
    if not requested or requested != legal[: len(requested)]:
        raise ValueError("capacity sweep order must be 1,2,4,6, then 8 only if eligible")
    return requested


def _resource_margins() -> tuple[float, float, float]:
    """Read current host/GPU free margins; unknown telemetry fails closed."""
    host_margin = 0.0
    try:
        entries = dict(
            line.replace(":", "").split()[:2]
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
            if ":" in line
        )
        host_margin = int(entries["MemAvailable"]) / int(entries["MemTotal"])
    except (KeyError, OSError, ValueError):
        pass
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
        )
        margins = [int(free) / int(total) for line in completed.stdout.splitlines() for free, total in [line.split(",")]]
        vram_margin = min(margins) if completed.returncode == 0 and margins else 0.0
    except (OSError, ValueError, ZeroDivisionError):
        vram_margin = 0.0
    return host_margin, vram_margin, vram_margin


def _run_worker_group(args: argparse.Namespace, assignments: Sequence[tuple[int, Trial]]) -> tuple[float, int, int]:
    """Start independent one-environment workers, then wait with finite bounds."""
    started = time.monotonic()
    processes: list[tuple[int, Trial, subprocess.Popen[str], Path, object]] = []
    for worker_id, trial in assignments:
        worker_root = args.output_root / "workers" / f"worker-{worker_id:02d}"
        heartbeat = worker_root / "heartbeat.json"
        log_path = worker_root / f"{trial.trial_id}.log"
        _write_heartbeat(heartbeat, worker_id=worker_id, trial_id=trial.trial_id, state="started")
        log = log_path.open("x", encoding="utf-8")
        processes.append((worker_id, trial, subprocess.Popen(_trial_command(args, trial), stdout=log, stderr=subprocess.STDOUT), heartbeat, log))
    completed = failed = 0
    for worker_id, trial, process, heartbeat, log in processes:
        try:
            returncode = process.wait(timeout=args.worker_timeout_seconds)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            returncode = 124
            _write_heartbeat(heartbeat, worker_id=worker_id, trial_id=trial.trial_id, state="timeout")
        else:
            _write_heartbeat(heartbeat, worker_id=worker_id, trial_id=trial.trial_id, state="terminal")
        log.close()
        try:
            verify_episode(args.output_root / "raw" / trial.trial_id)
            verified = True
        except ValueError:
            verified = False
        if returncode == 0 and verified:
            completed += 1
        else:
            failed += 1
    return time.monotonic() - started, completed, failed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--policy-revision-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--capacity-sweep")
    parser.add_argument("--trials-per-worker", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--worker-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_campaign(args: argparse.Namespace) -> dict[str, object]:
    from scripts.eval_groot_n17_matrix import load_matrix

    if args.trials_per_worker <= 0 or args.worker_timeout_seconds <= 0:
        raise ValueError("worker counts and timeout must be positive")
    trials = tuple(load_matrix(args.matrix))
    args.output_root.mkdir(parents=True, exist_ok=True)
    state = CampaignState(args.output_root, tuple(trial.trial_id for trial in trials))
    by_id = {trial.trial_id: trial for trial in trials}
    pending = pending_trial_ids(state)
    records: list[dict[str, object]] = []
    if not args.dry_run:
        for worker_id, trial_id in enumerate(pending, start=1):
            if (worker_id - 1) >= args.trials_per_worker:
                break
            returncode = _run_one_worker(args, worker_id=worker_id, trial=by_id[trial_id])
            records.append({"worker_id": worker_id, "trial_id": trial_id, "returncode": returncode})
    else:
        records = [{"trial_id": trial_id, "command": _trial_command(args, by_id[trial_id])} for trial_id in pending]

    report: dict[str, object] = {
        "schema_version": 1,
        "pending_before": list(pending),
        "workers": records,
        "completed_after": [trial_id for trial_id in state.trial_ids if trial_id not in pending_trial_ids(state)],
    }
    if args.capacity_sweep:
        counts = _validate_sweep(args.capacity_sweep)
        samples: list[CapacitySample] = []
        capacity_records: list[dict[str, object]] = []
        capacity_pending = list(pending_trial_ids(state))
        for count in counts:
            decision_before = choose_worker_count(samples) if samples else None
            if count == 8 and (decision_before is None or decision_before.accepted_workers != 6):
                capacity_records.append({"workers": count, "status": "skipped", "reason": "six_workers_not_accepted"})
                break
            assignments = tuple((index + 1, by_id[trial_id]) for index, trial_id in enumerate(capacity_pending[:count]))
            if len(assignments) != count:
                capacity_records.append({"workers": count, "status": "skipped", "reason": "insufficient_pending_trials"})
                break
            capacity_pending = capacity_pending[count:]
            elapsed, completed, failed = _run_worker_group(args, assignments)
            ram_margin, inference_margin, render_margin = _resource_margins()
            sample = CapacitySample(count, elapsed, completed, failed, inference_margin, render_margin, ram_margin)
            samples.append(sample)
            capacity_records.append({"workers": count, "elapsed_seconds": elapsed, "completed_trials": completed, "failed_trials": failed, "host_ram_margin": ram_margin, "inference_vram_margin": inference_margin, "render_vram_margin": render_margin})
            if choose_worker_count(samples).accepted_workers != count:
                break
        decision = choose_worker_count(samples)
        report["capacity"] = {"requested": list(counts), "samples": capacity_records, "accepted_workers": decision.accepted_workers, "rejected": decision.rejected}
    (args.output_root / "capacity-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_campaign(args)
    except ValueError as error:
        print(f"campaign validation error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CampaignState", "pending_trial_ids", "run_campaign"]
