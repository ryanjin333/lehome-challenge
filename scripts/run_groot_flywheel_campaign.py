"""Resumable, local-only supervisor for isolated GR00T rollout processes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence

from lehome.flywheel.artifacts import verify_episode_manifest
from lehome.flywheel.capacity import CapacitySample, choose_worker_count
from lehome.flywheel.isaac_recorder import CANONICAL_VIDEO_FILENAMES
from lehome.flywheel.matrix import Trial, load_public_matrix, matrix_sha256


@dataclass(frozen=True, slots=True)
class CampaignState:
    output_root: Path
    trial_ids: tuple[str, ...]


def is_completed_trial(episode_dir: Path) -> bool:
    """Accept only terminal, non-error artifacts with canonical video evidence."""
    try:
        episode, manifest = verify_episode_manifest(episode_dir)
    except ValueError:
        return False
    if not isinstance(episode.get("terminal_reason"), str) or not episode["terminal_reason"]:
        return False
    if episode.get("outcome") == "error" or episode.get("recorder_error"):
        return False
    expected_videos = {f"videos/{filename}" for filename in CANONICAL_VIDEO_FILENAMES}
    manifest_videos = {path for path in manifest if path.startswith("videos/")}
    return manifest_videos == expected_videos and all(manifest[path]["size"] > 0 for path in expected_videos)


def _prepare_retry_attempt(output_root: Path, trial_id: str) -> None:
    """Atomically quarantine an invalid prior attempt before retrying its ID."""
    if (
        not isinstance(trial_id, str)
        or trial_id in {"", ".", ".."}
        or "/" in trial_id
        or "\\" in trial_id
        or Path(trial_id).is_absolute()
        or Path(trial_id).name != trial_id
    ):
        raise ValueError("trial ID must be a non-empty path-safe identifier")
    root = Path(output_root)
    if root.is_symlink():
        raise ValueError("campaign output root must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise ValueError("campaign output root must be a directory")
    raw = root / "raw" / trial_id
    if is_completed_trial(raw):
        return
    sources: list[tuple[str, Path]] = []
    for name in (".pending", "raw"):
        parent = root / name
        if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
            raise ValueError(f"campaign {name} root is unsafe")
        source = parent / trial_id
        if source.is_symlink() or (source.exists() and not source.is_dir()):
            raise ValueError(f"campaign {name} trial path is unsafe")
        if source.exists():
            sources.append((name.removeprefix("."), source))
    if not sources:
        return
    quarantine_root = root / "quarantine"
    if quarantine_root.is_symlink() or (quarantine_root.exists() and not quarantine_root.is_dir()):
        raise ValueError("campaign quarantine root is unsafe")
    quarantine_root.mkdir(exist_ok=True)
    attempt = 1
    while True:
        quarantine = quarantine_root / f"{trial_id}.attempt-{attempt:03d}"
        try:
            quarantine.mkdir()
        except FileExistsError:
            if quarantine.is_symlink() or not quarantine.is_dir():
                raise ValueError("campaign quarantine attempt path is unsafe")
            attempt += 1
            continue
        break
    for name, source in sources:
        source.rename(quarantine / name)


def pending_trial_ids(state: CampaignState) -> tuple[str, ...]:
    """Resume unless the terminal artifact passes the canonical completion predicate."""
    return tuple(
        trial_id
        for trial_id in state.trial_ids
        if not is_completed_trial(state.output_root / "raw" / trial_id)
    )


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
        "--policy-repo", args.policy_repo, "--policy-step", str(args.policy_step), "--code-revision", args.code_revision,
        "--asset-revision", args.asset_revision, "--simulator-version", args.simulator_version,
        "--release-assets-root", str(args.release_assets_root),
        "--category", trial.category, "--release-stage", trial.release_stage,
        "--policy-artifact-sha256", args.policy_artifact_sha256, "--image-identity", args.image_identity,
        "--seed", str(trial.seed), "--episode-id", trial.trial_id, "--output-root", str(args.output_root),
        "--strategy", args.strategy,
        "--max-steps", str(args.max_steps), "--headless",
    ]


def _attempt_log_path(worker_root: Path, trial_id: str) -> Path:
    attempt = 1
    while True:
        path = worker_root / f"{trial_id}.attempt-{attempt:03d}.log"
        if not path.exists() and not path.is_symlink():
            return path
        attempt += 1


def _run_one_worker(args: argparse.Namespace, *, worker_id: int, trial: Trial) -> int:
    worker_root = args.output_root / "workers" / f"worker-{worker_id:02d}"
    heartbeat = worker_root / "heartbeat.json"
    log_path = _attempt_log_path(worker_root, trial.trial_id)
    _prepare_retry_attempt(args.output_root, trial.trial_id)
    _write_heartbeat(heartbeat, worker_id=worker_id, trial_id=trial.trial_id, state="started")
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(_trial_command(args, trial), stdout=log, stderr=subprocess.STDOUT)
        try:
            returncode = process.wait(timeout=args.worker_timeout_seconds)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=args.terminate_grace_seconds)
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


def _positive_finite_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return seconds


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


def _cleanup_partially_launched_workers(
    args: argparse.Namespace,
    processes: Sequence[tuple[int, Trial, subprocess.Popen[str], Path, object]],
) -> list[BaseException]:
    """Bound a best-effort shutdown without hiding the launch failure that caused it."""
    errors: list[BaseException] = []
    to_reap = list(processes)
    pending: list[tuple[int, Trial, subprocess.Popen[str], Path, object]] = []
    for record in processes:
        worker_id, trial, process, heartbeat, log = record
        try:
            if process.poll() is None:
                pending.append(record)
        except BaseException as error:
            errors.append(RuntimeError(f"worker {worker_id} could not be polled during launch cleanup: {error}"))
            pending.append(record)

    for worker_id, trial, process, heartbeat, log in pending:
        try:
            process.terminate()
        except BaseException as error:
            errors.append(RuntimeError(f"worker {worker_id} could not be terminated during launch cleanup: {error}"))
        try:
            _write_heartbeat(heartbeat, worker_id=worker_id, trial_id=trial.trial_id, state="timeout")
        except BaseException as error:
            errors.append(RuntimeError(f"worker {worker_id} heartbeat cleanup failed: {error}"))

    terminate_deadline = time.monotonic() + args.terminate_grace_seconds
    while pending and time.monotonic() < terminate_deadline:
        still_pending = []
        for record in pending:
            worker_id, trial, process, heartbeat, log = record
            try:
                if process.poll() is None:
                    still_pending.append(record)
            except BaseException as error:
                errors.append(RuntimeError(f"worker {worker_id} could not be polled during launch cleanup: {error}"))
                still_pending.append(record)
        pending = still_pending
        if pending:
            time.sleep(min(0.1, max(0.0, terminate_deadline - time.monotonic())))

    for worker_id, trial, process, heartbeat, log in pending:
        try:
            process.kill()
        except BaseException as error:
            errors.append(RuntimeError(f"worker {worker_id} could not be killed during launch cleanup: {error}"))

    reap_deadline = time.monotonic() + args.terminate_grace_seconds
    for worker_id, trial, process, heartbeat, log in to_reap:
        remaining = max(0.0, reap_deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except BaseException as error:
            errors.append(RuntimeError(f"worker {worker_id} could not be reaped during launch cleanup: {error}"))

    for worker_id, trial, process, heartbeat, log in processes:
        try:
            log.close()
        except BaseException as error:
            errors.append(RuntimeError(f"worker {worker_id} log could not be closed during launch cleanup: {error}"))
    return errors


def _report_launch_cleanup_failures(launch_error: BaseException, cleanup_errors: Sequence[BaseException]) -> None:
    """Keep the launch exception primary while making cleanup faults observable on Python 3.10+."""
    if not cleanup_errors:
        return
    detail = "; ".join(str(error) for error in cleanup_errors)
    if hasattr(launch_error, "add_note"):
        launch_error.add_note(f"Additional launch cleanup failures: {detail}")
    else:
        print(f"Additional launch cleanup failures: {detail}", file=sys.stderr)


def _run_worker_group(args: argparse.Namespace, assignments: Sequence[tuple[int, Trial]]) -> tuple[float, int, int]:
    """Start workers together and apply one launch-relative deadline to all."""
    started = time.monotonic()
    processes: list[tuple[int, Trial, subprocess.Popen[str], Path, object]] = []
    launch_log: object | None = None
    try:
        for worker_id, trial in assignments:
            worker_root = args.output_root / "workers" / f"worker-{worker_id:02d}"
            heartbeat = worker_root / "heartbeat.json"
            log_path = _attempt_log_path(worker_root, trial.trial_id)
            _prepare_retry_attempt(args.output_root, trial.trial_id)
            _write_heartbeat(heartbeat, worker_id=worker_id, trial_id=trial.trial_id, state="started")
            launch_log = log_path.open("x", encoding="utf-8")
            process = subprocess.Popen(_trial_command(args, trial), stdout=launch_log, stderr=subprocess.STDOUT)
            processes.append((worker_id, trial, process, heartbeat, launch_log))
            launch_log = None
    except BaseException as launch_error:
        cleanup_errors = _cleanup_partially_launched_workers(args, processes)
        if launch_log is not None:
            try:
                launch_log.close()
            except BaseException as error:
                cleanup_errors.append(RuntimeError(f"unlaunched worker log could not be closed during launch cleanup: {error}"))
        _report_launch_cleanup_failures(launch_error, cleanup_errors)
        raise
    returncodes: dict[int, int] = {}
    pending = list(processes)
    deadline = started + args.worker_timeout_seconds
    while pending and time.monotonic() < deadline:
        still_pending: list[tuple[int, Trial, subprocess.Popen[str], Path, object]] = []
        for worker_id, trial, process, heartbeat, log in pending:
            returncode = process.poll()
            if returncode is None:
                still_pending.append((worker_id, trial, process, heartbeat, log))
                continue
            returncodes[worker_id] = returncode
            _write_heartbeat(heartbeat, worker_id=worker_id, trial_id=trial.trial_id, state="terminal")
        pending = still_pending
        if pending:
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    if pending:
        for worker_id, trial, process, heartbeat, log in pending:
            process.terminate()
            _write_heartbeat(heartbeat, worker_id=worker_id, trial_id=trial.trial_id, state="timeout")
        grace_deadline = time.monotonic() + args.terminate_grace_seconds
        while pending and time.monotonic() < grace_deadline:
            still_pending = []
            for worker_id, trial, process, heartbeat, log in pending:
                if process.poll() is None:
                    still_pending.append((worker_id, trial, process, heartbeat, log))
            pending = still_pending
            if pending:
                time.sleep(min(0.1, max(0.0, grace_deadline - time.monotonic())))
        for worker_id, trial, process, heartbeat, log in pending:
            process.kill()
        reap_deadline = time.monotonic() + args.terminate_grace_seconds
        for worker_id, trial, process, heartbeat, log in pending:
            remaining = reap_deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"worker {worker_id} did not exit after SIGKILL")
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(f"worker {worker_id} did not exit after SIGKILL") from error
        for worker_id, trial, process, heartbeat, log in processes:
            if worker_id not in returncodes:
                returncodes[worker_id] = 124

    completed = failed = 0
    for worker_id, trial, process, heartbeat, log in processes:
        returncode = returncodes[worker_id]
        log.close()
        if returncode == 0 and is_completed_trial(args.output_root / "raw" / trial.trial_id):
            completed += 1
        else:
            failed += 1
    return time.monotonic() - started, completed, failed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True, help="committed canonical public 280-trial JSON")
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--policy-revision-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--policy-repo", required=True)
    parser.add_argument("--policy-step", type=int, required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--asset-revision", required=True)
    parser.add_argument("--release-assets-root", type=Path, required=True)
    parser.add_argument("--simulator-version", required=True)
    parser.add_argument("--policy-artifact-sha256", required=True)
    parser.add_argument("--image-identity", required=True)
    parser.add_argument("--capacity-sweep")
    parser.add_argument("--strategy", choices=("canonical", "mild", "strong"), default="canonical")
    parser.add_argument("--trials-per-worker", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--worker-timeout-seconds", type=_positive_finite_seconds, default=1800.0)
    parser.add_argument("--terminate-grace-seconds", type=_positive_finite_seconds, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_campaign(args: argparse.Namespace) -> dict[str, object]:
    if (
        args.trials_per_worker <= 0
        or args.worker_timeout_seconds <= 0
        or not math.isfinite(args.worker_timeout_seconds)
        or not math.isfinite(args.terminate_grace_seconds)
        or args.terminate_grace_seconds <= 0
    ):
        raise ValueError("worker counts and timeouts must be finite and positive")
    matrix = load_public_matrix(args.matrix)
    trials = matrix.trials
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
        "matrix": {
            "schema_version": matrix.schema_version,
            "sha256": matrix_sha256(matrix),
            "trial_count": len(trials),
            "training_holdouts": list(matrix.training_holdouts),
        },
        "pending_before": list(pending),
        "workers": records,
        "completed_after": [trial_id for trial_id in state.trial_ids if trial_id not in pending_trial_ids(state)],
    }
    if args.capacity_sweep and not args.dry_run:
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
    elif args.capacity_sweep:
        report["capacity"] = {"requested": list(_validate_sweep(args.capacity_sweep)), "status": "dry_run_no_processes"}
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
