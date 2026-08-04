"""Prepare and operate a loopback-only physical SO101 DAgger collection session."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import secrets
import stat
import re
import threading
import time
from typing import Any, Callable, Iterable, Protocol
from uuid import uuid4

from lehome.flywheel.artifacts import atomic_write_json, build_sha256_manifest, verify_episode
from lehome.flywheel.bridge_receiver import BridgeReceiver, LoopbackBridgeServer
from lehome.flywheel.intervention import InterventionController
from lehome.flywheel.isaac_recorder import MixedSourceRecorder, RecordedEpisode
from lehome.flywheel.models import ActionSource, EpisodeOutcome, QualityGrade, RejectionReason
from lehome.flywheel.quality import AttemptStats, QualityResult, QualityThresholds, grade_attempt, load_quality_thresholds
from lehome.flywheel.runtime_preflight import require_isaac_sim_5_1_runtime
from lehome.flywheel.snapshots import capture_snapshot


CONTROLS = {
    "space": "activate_or_request_takeover",
    "a": "accept_after_official_success",
    "d": "discard",
    "r": "reset",
    "escape": "safe_exit",
}
LOOPBACK_HOST = "127.0.0.1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def default_secret_path() -> Path:
    return Path.home() / ".local" / "state" / "lehome-groot" / "bridge-session.secret"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("practice", "expert", "dagger"), default="practice")
    parser.add_argument("--listen-host", default=LOOPBACK_HOST)
    parser.add_argument("--listen-port", type=int, default=18080)
    parser.add_argument("--run-root", type=Path, default=Path("runs/groot-dagger"))
    parser.add_argument("--enable-training-output", action="store_true")
    parser.add_argument("--quality-thresholds", type=Path)
    parser.add_argument("--organizer-dataset-revision")
    parser.add_argument("--organizer-dataset-sha256")
    parser.add_argument("--left-calibration-sha256")
    parser.add_argument("--right-calibration-sha256")
    parser.add_argument("--policy-path", type=Path)
    parser.add_argument("--policy-revision")
    parser.add_argument("--policy-revision-file", type=Path)
    parser.add_argument("--policy-repo")
    parser.add_argument("--policy-step", type=int)
    parser.add_argument("--policy-artifact-sha256")
    parser.add_argument("--image-identity")
    parser.add_argument("--code-revision")
    parser.add_argument("--asset-revision")
    parser.add_argument("--release-assets-root", type=Path)
    parser.add_argument("--simulator-version")
    parser.add_argument("--episode-id")
    parser.add_argument("--garment")
    parser.add_argument("--category", choices=("top_long", "top_short", "pant_long", "pant_short"))
    parser.add_argument("--release-stage", choices=("seen", "public_unseen"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", choices=("canonical", "mild", "strong"), default="canonical")
    parser.add_argument("--task", default="LeHome-BiSO101-Direct-Garment-v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--interactive", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> QualityThresholds | None:
    if args.listen_host != LOOPBACK_HOST:
        raise ValueError("DAgger receiver must be loopback-only")
    if not isinstance(args.listen_port, int) or not 1 <= args.listen_port <= 65535:
        raise ValueError("collector listen port must be in the TCP port range")
    calibration_hashes = (args.left_calibration_sha256, args.right_calibration_sha256)
    if any(value is not None for value in calibration_hashes) and not all(
        isinstance(value, str) and _SHA256.fullmatch(value) for value in calibration_hashes
    ):
        raise ValueError("collector calibration hashes must be paired lowercase SHA-256 values")
    if args.interactive and not all(calibration_hashes):
        raise ValueError("interactive collection requires both expected calibration hashes")
    if args.interactive:
        from scripts.run_groot_flywheel_trial import (
            _validate_declared_production_provenance,
            build_identity,
            read_pinned_revision,
        )

        revision = args.policy_revision
        if args.policy_revision_file is not None:
            if revision is not None:
                raise ValueError("provide one policy revision source")
            revision = read_pinned_revision(args.policy_revision_file)
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("interactive collection requires a pinned policy revision")
        if args.policy_path is None or not args.policy_path.is_dir():
            raise ValueError("interactive collection requires an existing policy path")
        if args.max_steps <= 0 or args.seed < 0:
            raise ValueError("interactive collection requires non-negative seed and positive max-steps")
        args.collection_identity = build_identity(args, revision)
        _validate_declared_production_provenance(args)
        if args.mode in {"expert", "dagger"} and not args.enable_training_output:
            raise ValueError("interactive expert or dagger collection requires training output")
    if args.enable_training_output and args.mode not in {"expert", "dagger"}:
        raise ValueError("training output requires expert or dagger mode")
    if not args.enable_training_output:
        return None
    if args.quality_thresholds is None:
        raise ValueError("quality thresholds manifest is required for training output")
    if not args.organizer_dataset_revision or not args.organizer_dataset_sha256:
        raise ValueError("training output requires pinned organizer dataset revision and SHA-256")
    return load_quality_thresholds(
        args.quality_thresholds,
        expected_dataset_revision=args.organizer_dataset_revision,
        expected_dataset_sha256=args.organizer_dataset_sha256,
    )


def create_session_secret(path: Path) -> bytes:
    """Create a one-session secret atomically with owner-only permissions."""
    secret_path = Path(path)
    secret_path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    parent = secret_path.parent.stat()
    if (
        not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != os.geteuid()
        or stat.S_IMODE(parent.st_mode) & (stat.S_IWGRP | stat.S_IWOTH)
    ):
        raise RuntimeError("bridge session secret parent must be owner-private")
    try:
        descriptor = os.open(secret_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError("refusing to reuse an existing bridge session secret") from error
    created = os.fstat(descriptor)
    identity = (created.st_dev, created.st_ino)
    secret = secrets.token_bytes(32)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(secret)
            output.flush()
            os.fsync(output.fileno())
    except BaseException as error:
        try:
            remove_session_secret(secret_path, identity=identity)
        except RuntimeError as cleanup_error:
            raise cleanup_error from error
        raise
    try:
        if stat.S_IMODE(secret_path.stat().st_mode) != 0o600:
            raise RuntimeError("failed to create a mode-0600 bridge session secret")
    except BaseException as error:
        try:
            remove_session_secret(secret_path, identity=identity)
        except RuntimeError as cleanup_error:
            raise cleanup_error from error
        raise RuntimeError("failed to create a mode-0600 bridge session secret")
    return secret


def _secret_identity(path: Path) -> tuple[int, int]:
    status = path.lstat()
    if not stat.S_ISREG(status.st_mode) or stat.S_IMODE(status.st_mode) != 0o600:
        raise RuntimeError("bridge session secret is no longer a private regular file")
    return status.st_dev, status.st_ino


def remove_session_secret(path: Path, *, identity: tuple[int, int]) -> None:
    """Best-effort overwrite then unlink, without removing a replaced path."""
    secret_path = Path(path)
    try:
        status = secret_path.lstat()
    except FileNotFoundError:
        return
    if (
        not stat.S_ISREG(status.st_mode)
        or stat.S_IMODE(status.st_mode) != 0o600
        or (status.st_dev, status.st_ino) != identity
    ):
        raise RuntimeError("refusing to remove a bridge secret path not created by this session")
    descriptor = os.open(secret_path, os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != identity:
            raise RuntimeError("refusing to overwrite a replaced bridge secret")
        remaining = opened.st_size
        while remaining:
            written = os.write(descriptor, b"\0" * min(remaining, 64 * 1024))
            if written <= 0:
                raise RuntimeError("failed to overwrite the bridge session secret")
            remaining -= written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    final_status = secret_path.lstat()
    if (final_status.st_dev, final_status.st_ino) != identity:
        raise RuntimeError("refusing to remove a replaced bridge secret")
    secret_path.unlink()


@dataclass(slots=True)
class CollectorSession:
    controller: InterventionController
    session_nonce: str
    quality_thresholds: QualityThresholds | None
    bridge_server: LoopbackBridgeServer
    secret_path: Path
    secret_identity: tuple[int, int]
    controls: list[str] = field(default_factory=list)
    _secret_removed: bool = field(default=False, init=False, repr=False)
    _listener_thread: threading.Thread | None = field(default=None, init=False, repr=False)

    def record_control(self, control: str) -> None:
        if control not in CONTROLS:
            raise ValueError("unsupported DAgger control")
        self.controls.append(control)

    def status(self) -> dict[str, object]:
        bridge = self.bridge_server.receiver.current()
        return {
            "mode": self.controller.mode,
            "state": self.controller.state,
            "action_source": self.controller.action_source,
            "bridge_age_ms": bridge.sample_age_ms,
            "bridge_jitter_ms": self.bridge_server.receiver.jitter_ms,
            "bridge_state": bridge.reason or "eligible",
        }

    def start_listener(self) -> threading.Thread:
        self.bridge_server.start()

        def serve() -> None:
            try:
                self.bridge_server.serve_one_client()
            except (ConnectionError, OSError, RuntimeError, ValueError):
                # The receiver retains the fail-closed disconnect state. Do not
                # print transport details that could reveal operator context.
                if not self.bridge_server._cancel.is_set() and self.bridge_server.failure is None:
                    self.bridge_server.failure = "bridge_listener_failed"
                    self.bridge_server.receiver.close_connection()
                return

        thread = threading.Thread(target=serve, name="lehome-bridge-receiver", daemon=True)
        thread.start()
        self._listener_thread = thread
        return thread

    def wait_for_bridge_ready(self, *, poll_interval_s: float = 0.05) -> None:
        """Do not start a paid Isaac episode until a healthy bridge is live.

        Listener acceptance intentionally has no short startup deadline.  This
        wait is instead cancelled by ``close_listener``/Ctrl-C and fails
        visibly on a transport error, preventing a whole episode of silent
        ``no_sample`` holds while the operator finishes the SSH tunnel.
        """
        if poll_interval_s <= 0:
            raise ValueError("bridge readiness polling interval must be positive")
        while True:
            if self.bridge_server.failure is not None:
                raise RuntimeError("bridge listener failed before collection became ready")
            command = self.bridge_server.receiver.current()
            if self.bridge_server.receiver.handshake is not None and command.eligible:
                return
            if self.bridge_server._cancel.is_set():
                raise RuntimeError("bridge listener was cancelled before collection became ready")
            time.sleep(poll_interval_s)

    def close_listener(self) -> None:
        shutdown_timeout = False
        try:
            self.bridge_server.close()
            if self._listener_thread is not None:
                self._listener_thread.join(timeout=1.0)
                shutdown_timeout = self._listener_thread.is_alive()
        finally:
            if not self._secret_removed:
                remove_session_secret(self.secret_path, identity=self.secret_identity)
                self._secret_removed = True
        if shutdown_timeout:
            raise RuntimeError("bridge listener did not stop after bounded cancellation")


class ControlSource(Protocol):
    def poll(self) -> str | None: ...
    def metrics(self) -> dict[str, int]: ...


class ScheduledControlSource:
    """Deterministic nonblocking control queue used by tests and dry orchestration."""

    def __init__(self, controls: Iterable[str]) -> None:
        self._controls = list(controls)
        self._hesitations = 0
        self._corrections = 0

    def poll(self) -> str | None:
        return self._controls.pop(0) if self._controls else None

    def metrics(self) -> dict[str, int]:
        return {"hesitations": self._hesitations, "corrections": self._corrections}


def _observation(env: object, reset_result: object | None = None) -> dict[str, object]:
    if isinstance(reset_result, dict):
        return reset_result
    getter = getattr(env, "_get_observations", None)
    if callable(getter):
        return getter()
    getter = getattr(env, "observation", None)
    if callable(getter):
        return getter()
    raise ValueError("collection environment must provide an observation mapping")


def _step(env: object, action: tuple[float, ...]) -> tuple[dict[str, object], float, bool]:
    result = getattr(env, "step")(action)
    if isinstance(result, tuple) and len(result) == 3:
        observation, reward, success = result
        return _observation(env, observation), float(reward), bool(success)
    if isinstance(result, tuple) and len(result) == 5:
        observation, reward, terminated, truncated, info = result
        success = bool(info.get("official_success", terminated)) if isinstance(info, dict) else bool(terminated)
        return _observation(env, observation), float(reward), success
    reward = getattr(env, "reward", None)
    success = getattr(env, "official_success", None)
    if reward is None or success is None:
        raise ValueError("collection environment step must return reward and official success")
    return _observation(env), float(reward), bool(success)


def _policy_action(policy: object, observation: dict[str, object]) -> tuple[tuple[float, ...], str, int]:
    select = getattr(policy, "select_action_with_provenance", None)
    if not callable(select):
        raise ValueError("DAgger policy control requires provenance-bearing policy actions")
    selected = select(observation)
    value, request_id, chunk_offset = getattr(selected, "value", None), getattr(selected, "request_id", None), getattr(selected, "chunk_offset", None)
    if not isinstance(request_id, str) or not request_id or not isinstance(chunk_offset, int) or chunk_offset < 0:
        raise ValueError("policy action provenance is invalid")
    action = tuple(float(item) for item in value)
    if len(action) != 12 or not all(math.isfinite(item) for item in action):
        raise ValueError("policy action must be finite 12D")
    return action, request_id, chunk_offset


def _clear_policy_queue(policy: object) -> None:
    for name in ("clear_queued_actions", "clear_action_queue", "reset"):
        clear = getattr(policy, name, None)
        if callable(clear):
            clear()
            return
    raise ValueError("DAgger takeover requires a policy queue clear operation")


def _operator_resync(receiver: object):
    """Apply an explicit ``space`` resync only after the receiver's safe gate."""
    current = getattr(receiver, "current", None)
    if not callable(current):
        raise ValueError("bridge receiver does not report bridge health")
    command = current()
    if command.reason == "resync_required":
        resync = getattr(receiver, "resync", None)
        if not callable(resync):
            raise ValueError("bridge receiver does not support explicit resynchronization")
        resync()
        return current()
    return command


def _percentile(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]


def _attempt_quality(
    *,
    thresholds: QualityThresholds | None,
    official_success: bool,
    actions: list[tuple[float, ...]],
    step_dt_s: float | None,
    jitter_samples: list[float],
    stale_samples: int,
    disconnected: bool,
    holds: int,
    unsafe_commands: int,
    control_source: ControlSource,
    manual_discarded: bool,
) -> QualityResult:
    metrics = control_source.metrics()
    if unsafe_commands:
        return QualityResult("C", ("unsafe_commands",), 0.0)
    if thresholds is None or not isinstance(step_dt_s, (int, float)) or not math.isfinite(step_dt_s) or step_dt_s <= 0:
        return QualityResult("C", ("metric_evidence_unavailable",), 0.0)
    velocities = [max(abs(current - previous) for current, previous in zip(action, prior)) / step_dt_s for prior, action in zip(actions, actions[1:])]
    accelerations = [abs(current - previous) / step_dt_s for previous, current in zip(velocities, velocities[1:])]
    velocity_p95, acceleration_p95, jitter_p95 = _percentile(velocities), _percentile(accelerations), _percentile(jitter_samples)
    if velocity_p95 is None or acceleration_p95 is None or jitter_p95 is None:
        return QualityResult("C", ("metric_evidence_unavailable",), 0.0)
    if not all(isinstance(metrics.get(name), int) and metrics[name] >= 0 for name in ("hesitations", "corrections")):
        return QualityResult("C", ("metric_evidence_unavailable",), 0.0)
    return grade_attempt(
        AttemptStats(
            official_success=official_success,
            hesitations=metrics["hesitations"] + holds,
            corrections=metrics["corrections"],
            stale_samples=stale_samples,
            unsafe_commands=unsafe_commands,
            disconnected=disconnected,
            manual_discarded=manual_discarded,
            velocity_p95=velocity_p95,
            acceleration_p95=acceleration_p95,
            jitter_p95=jitter_p95,
        ),
        thresholds,
    )


def _collection_step_dt_s(env: object) -> float:
    """Return the task-declared control cadence or fail before recording."""
    value = getattr(env, "step_dt_s", None)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("collection environment step_dt_s must be a positive finite number")
    return float(value)


def _prepare_zero_step_diagnostic(recorder: MixedSourceRecorder) -> None:
    """Keep an unstarted, discarded attempt immutable without inventing a frame."""
    annotations = recorder.writer.staging / "annotations.jsonl"
    if not annotations.exists():
        annotations.touch(exist_ok=False)


def _rejection_reasons(quality: QualityResult, *, held: bool, discarded: bool) -> tuple[RejectionReason, ...]:
    mapped = {
        "stale_samples": RejectionReason.STALE_EXPERT,
        "disconnected": RejectionReason.DISCONNECTED,
        "unsafe_commands": RejectionReason.UNSAFE,
        "operator_discarded": RejectionReason.OPERATOR_DISCARDED,
        "metric_evidence_unavailable": RejectionReason.MISSING,
    }
    reasons = [mapped[reason] for reason in quality.reasons if reason in mapped]
    if held:
        reasons.append(RejectionReason.HOLD)
    if discarded:
        reasons.append(RejectionReason.OPERATOR_DISCARDED)
    if not reasons:
        reasons.append(RejectionReason.FAILED_EPISODE)
    return tuple(dict.fromkeys(reasons))


def _write_export_receipt(run_root: Path, result: RecordedEpisode) -> None:
    """Atomically publish selection-only evidence after verifying immutable raw data."""
    verify_episode(result.path)
    destination = run_root / "exports" / result.path.name
    if destination.exists() or destination.is_symlink():
        raise ValueError("refusing to overwrite an existing DAgger export receipt")
    staging = run_root / ".pending-exports" / f"{result.path.name}-{uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        atomic_write_json(staging / "selection-report.json", result.selection_report.as_dict() if result.selection_report else {})
        atomic_write_json(
            staging / "expert-windows.json",
            {"windows": [{"observation_step": window.observation_step, "future_actions": [list(action) for action in window.future_actions]} for window in result.expert_windows]},
        )
        atomic_write_json(staging / "SHA256SUMS.json", build_sha256_manifest(staging))
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(destination)
    except BaseException:
        if staging.exists():
            import shutil
            shutil.rmtree(staging)
        raise


def collect_episode(
    session: CollectorSession,
    env: object,
    policy: object,
    control_source: ControlSource,
    recorder: MixedSourceRecorder,
    thresholds: QualityThresholds | None,
    max_steps: int,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    ready_poll_interval_s: float = 0.01,
) -> RecordedEpisode:
    """Collect one real/fake episode through the same source and evidence gates."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if (
        isinstance(ready_poll_interval_s, bool)
        or not isinstance(ready_poll_interval_s, (int, float))
        or not math.isfinite(ready_poll_interval_s)
        or ready_poll_interval_s <= 0
    ):
        raise ValueError("ready polling interval must be a positive finite number")
    reset = getattr(env, "reset", None)
    if not callable(reset):
        raise ValueError("collection environment must provide reset")
    # The collector may not invent a control rate: the registered task owns it.
    step_dt_s = _collection_step_dt_s(env)
    observation = _observation(env, reset())
    official_success = False
    accept_requested = False
    discarded = False
    held = stale_samples = unsafe_commands = 0
    disconnected = False
    unsafe = False
    actions: list[tuple[float, ...]] = []
    jitter_samples: list[float] = []
    last_safe = tuple(float(value) for value in getattr(session.bridge_server.receiver, "last_safe_command", (0.0,) * 12))
    next_step_deadline: float | None = None
    reset_snapshot_recorded = False
    steps_applied = 0
    while steps_applied < max_steps:
        control = control_source.poll()
        if control is not None:
            session.record_control(control)
            if control in {"d", "escape"}:
                discarded = True
                break
            if control == "r":
                # Preserve the partial attempt as diagnostic-only, then return
                # the real task to reset before ending this immutable episode.
                observation = _observation(env, reset())
                if session.controller.state == "ready":
                    # No attempt has begun yet. Continue waiting for the
                    # operator rather than manufacturing a HOLD frame merely
                    # to make a reset control finalizable.
                    continue
                discarded = True
                break
            if control == "a" and official_success:
                accept_requested = True
            if control == "space":
                if session.controller.state == "ready":
                    (session.controller.start_policy if session.controller.mode == "dagger" else session.controller.start_expert)()
                    recorder.record_snapshot("reset", capture_snapshot(env, randomization={"strategy": "collection"}))
                    reset_snapshot_recorded = True
                elif session.controller.state in {"policy", "takeover_pending"}:
                    if session.controller.state == "policy":
                        session.controller.request_takeover()
                    command = _operator_resync(session.bridge_server.receiver)
                    if command.eligible:
                        session.controller.accept_expert(
                            current_robot=tuple(float(value) for value in observation["observation.state"]),
                            leader_command=command.command,
                        )
                        _clear_policy_queue(policy)
                        recorder.record_snapshot("takeover", capture_snapshot(env, randomization={"strategy": "collection"}))
                elif session.controller.state == "expert":
                    _operator_resync(session.bridge_server.receiver)
        if session.controller.state == "ready":
            # Before the operator explicitly starts, leave the real task
            # untouched. Polling in short intervals preserves cancel/reset
            # responsiveness without consuming an episode step or writing a
            # synthetic HOLD frame.
            sleep(float(ready_poll_interval_s))
            continue
        if not reset_snapshot_recorded:
            raise RuntimeError("collection started without a reset snapshot")
        now = clock()
        if not math.isfinite(now):
            raise ValueError("collection clock must return a finite monotonic time")
        if next_step_deadline is None:
            # The first applied action is allowed immediately after ``space``.
            next_step_deadline = now
        else:
            remaining = next_step_deadline - now
            if remaining > 0:
                sleep(remaining)
        source = ActionSource.HOLD
        provenance: dict[str, object] = {}
        if session.controller.state == "policy":
            action, request_id, chunk_offset = _policy_action(policy, observation)
            source = ActionSource.POLICY
            provenance = {"policy_request_id": request_id, "policy_chunk_offset": chunk_offset}
        elif session.controller.state == "expert":
            command = session.bridge_server.receiver.current()
            if command.eligible:
                action = tuple(float(value) for value in command.command)
                last_safe = action
                source = ActionSource.EXPERT
                provenance = {"expert_sequence": command.sequence, "expert_sample_age_ms": command.sample_age_ms}
                jitter_samples.append(float(getattr(session.bridge_server.receiver, "jitter_ms", 0.0)))
            else:
                action = last_safe
                held += 1
                stale_samples += int(command.reason == "stale_sample")
                disconnected = disconnected or command.reason == "disconnected"
        else:
            action = last_safe
            held += 1
        safety = getattr(env, "is_action_safe", None)
        if not callable(safety) or not bool(safety(action)):
            unsafe_commands += 1
            unsafe = True
            # Never invoke env.step with an action that failed the real task's
            # safety boundary.  Preserve a diagnostic hold frame so the
            # immutable artifact remains structurally valid, but exclude the
            # rejected command from both BC targets and applied-action data.
            recorder.record_step(
                observation,
                last_safe,
                reward=0.0,
                success=official_success,
                action_source=ActionSource.HOLD,
                segment=session.controller.segment,
            )
            break
        next_observation, reward, step_success = _step(env, action)
        steps_applied += 1
        # Advance from the prior deadline, rather than the end of this step,
        # so normal scheduling does not accumulate timing drift. If a step
        # overran one or more periods, skip those missed deadlines instead of
        # issuing a catch-up burst faster than the task's declared cadence.
        next_step_deadline += step_dt_s
        completed_at = clock()
        if not math.isfinite(completed_at):
            raise ValueError("collection clock must return a finite monotonic time")
        if completed_at > next_step_deadline:
            missed_periods = math.floor((completed_at - next_step_deadline) / step_dt_s) + 1
            next_step_deadline += missed_periods * step_dt_s
        official_success = official_success or step_success
        recorder.record_step(
            observation, action, reward=reward, success=official_success, action_source=source,
            segment=session.controller.segment, **provenance,
        )
        actions.append(action)
        observation = next_observation
    recorder.record_snapshot("terminal", capture_snapshot(env, randomization={"strategy": "collection"}))
    if steps_applied == 0:
        _prepare_zero_step_diagnostic(recorder)
    quality = _attempt_quality(
        thresholds=thresholds, official_success=official_success, actions=actions, step_dt_s=step_dt_s,
        jitter_samples=jitter_samples, stale_samples=stale_samples, disconnected=disconnected,
        holds=held, unsafe_commands=unsafe_commands, control_source=control_source, manual_discarded=discarded,
    )
    accepted = False
    if accept_requested and official_success and quality.trainable and not held and not discarded and session.controller.state == "expert":
        accepted = session.controller.accept(quality)
    elif session.controller.state in {"policy", "takeover_pending", "expert"}:
        session.controller.discard()
    reasons = _rejection_reasons(quality, held=bool(held), discarded=discarded)
    outcome = EpisodeOutcome(
        "unsafe" if unsafe else "success" if official_success and not discarded else "discarded" if discarded else "timeout",
        accepted,
        QualityGrade(quality.grade),
        () if accepted else reasons,
        "accept" if accepted else "discard",
    )
    result = recorder.finish(outcome=outcome, controls=session.controls)
    if accepted and session.controller.mode != "practice" and result.episode["trainable"] and result.expert_windows:
        _write_export_receipt(recorder.writer.run_root, result)
    return result


def _write_session_manifest(root: Path, session: CollectorSession, args: argparse.Namespace) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "session-manifest.json"
    if manifest.exists():
        raise ValueError("refusing to overwrite an existing DAgger session manifest")
    document: dict[str, Any] = {
        "mode": args.mode,
        "listen_host": args.listen_host,
        "listen_port": args.listen_port,
        "enable_training_output": args.enable_training_output,
        "controls": CONTROLS,
        "session_nonce": session.session_nonce,
    }
    identity = getattr(args, "collection_identity", None)
    if identity is not None:
        document["identity"] = {
            "episode_id": identity.episode_id,
            "policy_repo": identity.policy_repo,
            "policy_revision": identity.policy_revision,
            "policy_step": identity.policy_step,
            "code_revision": identity.code_revision,
            "asset_revision": identity.asset_revision,
            "simulator_version": identity.simulator_version,
            "garment_name": identity.garment_name,
            "category": identity.category,
            "release_stage": identity.release_stage,
            "seed": identity.seed,
            "instruction": identity.instruction,
            "strategy": identity.strategy,
        }
    if session.quality_thresholds is not None:
        document["quality_threshold_dataset"] = {
            "revision": session.quality_thresholds.dataset_revision,
            "sha256": session.quality_thresholds.dataset_sha256,
        }
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, manifest)


def prepare_session(args: argparse.Namespace, *, secret_path: Path | None = None) -> CollectorSession:
    thresholds = validate_args(args)
    created_secret_path = default_secret_path() if secret_path is None else Path(secret_path)
    secret = create_session_secret(created_secret_path)
    secret_identity = _secret_identity(created_secret_path)
    expected_calibrations = None
    if args.left_calibration_sha256 is not None:
        expected_calibrations = (args.left_calibration_sha256, args.right_calibration_sha256)
    receiver = BridgeReceiver(expected_calibrations=expected_calibrations)
    session_nonce = secrets.token_urlsafe(24)
    try:
        session = CollectorSession(
            controller=InterventionController(mode=args.mode),
            session_nonce=session_nonce,
            quality_thresholds=thresholds,
            bridge_server=LoopbackBridgeServer(
                secret=secret,
                session_nonce=session_nonce,
                port=args.listen_port,
                receiver=receiver,
            ),
            secret_path=created_secret_path,
            secret_identity=secret_identity,
        )
        _write_session_manifest(args.run_root, session, args)
        return session
    except BaseException:
        remove_session_secret(created_secret_path, identity=secret_identity)
        raise


def ssh_forward_command(port: int) -> str:
    """A copyable command template that exposes no secret or secret path."""
    return f"ssh -N -L {port}:127.0.0.1:{port} USER@APPROVED_NORTH_AMERICAN_HOST"


def run_interactive(session: CollectorSession) -> None:  # pragma: no cover - operator path
    print("Controls:", json.dumps(CONTROLS, sort_keys=True))
    while True:
        print(json.dumps(session.status(), sort_keys=True))
        control = input("control> ").strip().lower()
        if control == "escape":
            session.record_control(control)
            return
        if control not in CONTROLS:
            print("unrecognized control")
            continue
        session.record_control(control)
        # Robot state, official success, bridge health, and actual actions are
        # supplied by the Isaac integration loop; this control loop never emits
        # an unvalidated command by itself.


class BackgroundStdinControlSource(ScheduledControlSource):
    """Nonblocking stdin reader; collection sampling never waits on terminal input."""

    def __init__(self) -> None:  # pragma: no cover - operator path
        super().__init__(())
        import queue

        self._queue: queue.SimpleQueue[str] = queue.SimpleQueue()

        def read_controls() -> None:
            while True:
                try:
                    control = input().strip().lower()
                except EOFError:
                    return
                if control in CONTROLS:
                    self._queue.put(control)
                    if control == "escape":
                        return

        threading.Thread(target=read_controls, name="lehome-dagger-controls", daemon=True).start()

    def poll(self) -> str | None:  # pragma: no cover - operator path
        try:
            return self._queue.get_nowait()
        except Exception:
            return None


def _run_production_collection(
    args: argparse.Namespace,
    session: CollectorSession,
    *,
    runtime_preflight: Callable[[], object] | None = None,
) -> RecordedEpisode:  # pragma: no cover - requires Isaac/GR00T
    """Launch Isaac and GR00T only after strict argument validation completed."""
    (runtime_preflight or require_isaac_sim_5_1_runtime)()
    from isaaclab.app import AppLauncher
    from scripts.eval_policy import PolicyRegistry
    import scripts.eval_policy.groot_policy  # noqa: F401
    from scripts.run_groot_flywheel_trial import _live_runtime_identity, _production_env, _validate_live_runtime_identity
    from scripts.utils import common

    launch_parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(launch_parser)
    launch_args, _ = launch_parser.parse_known_args([])
    app = common.launch_app_from_args(launch_args)
    env = None
    try:
        _validate_live_runtime_identity(args, app, runtime_identity_reader=_live_runtime_identity)
        env = _production_env(args)
        policy = PolicyRegistry.create("groot", model_path=str(args.policy_path), device=args.device, task_description="fold the garment on the table")
        policy.reset()

        class IsaacCollectionEnvironment:
            """Narrow tuple-action seam over the registered Isaac task."""

            # This task's registered control cadence is part of its task contract.
            step_dt_s = 1.0 / 30.0

            def reset(self):
                env.reset()
                return env._get_observations()

            def step(self, action: tuple[float, ...]):
                import torch

                env.step(torch.tensor(action, dtype=torch.float32, device=args.device).unsqueeze(0))
                reward = env._get_rewards()
                success = env._get_success()
                return env._get_observations(), float(reward.item() if hasattr(reward, "item") else reward), bool(success.item() if hasattr(success, "item") else success)

            def flywheel_capture_state(self):
                capture = getattr(env, "flywheel_capture_state", None)
                if callable(capture):
                    return capture()
                # Match capture_snapshot's strict fallback surface. Missing state
                # remains an error: a collector may not invent reset evidence.
                return {
                    name: getattr(env, name)
                    for name in ("robot_position", "robot_velocity", "cloth_position", "cloth_velocity", "rng_state", "garment_name")
                }

            def is_action_safe(self, action: tuple[float, ...]) -> bool:
                from lehome.assets.robots.lerobot import SO101_FOLLOWER_USD_JOINT_LIMLITS

                limits = tuple(SO101_FOLLOWER_USD_JOINT_LIMLITS.values())
                return all(
                    math.isfinite(value) and math.radians(lower) <= value <= math.radians(upper)
                    for arm in (action[:6], action[6:])
                    for value, (lower, upper) in zip(arm, limits)
                )

        recorder = MixedSourceRecorder(
            args.run_root,
            identity=args.collection_identity,
            mode=args.mode,
            provenance={"policy_artifact_sha256": args.policy_artifact_sha256, "image_identity": args.image_identity},
        )
        return collect_episode(session, IsaacCollectionEnvironment(), policy, BackgroundStdinControlSource(), recorder, session.quality_thresholds, args.max_steps)
    finally:
        if env is not None and hasattr(env, "close"):
            env.close()
        common.close_app(app)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.interactive:
        # Validate before this host gate so malformed input cannot disguise the
        # actual command boundary, then reject an incompatible paid host before
        # session-secret or output creation.
        validate_args(args)
        require_isaac_sim_5_1_runtime()
    session = prepare_session(args)
    try:
        print(json.dumps({"ssh_forward": ssh_forward_command(args.listen_port), **session.status()}, sort_keys=True))
        if args.interactive:
            session.start_listener()
            session.wait_for_bridge_ready()
            # main completed the pre-output host gate above; keep the helper's
            # default guard for direct callers and avoid a second probe here.
            result = _run_production_collection(args, session, runtime_preflight=lambda: None)
            print(json.dumps({"episode_id": result.path.name, "bc_target_count": result.episode["bc_target_count"], "trainable": result.episode["trainable"]}, sort_keys=True))
        return 0
    finally:
        session.close_listener()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
