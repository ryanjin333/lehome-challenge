"""Run one long-lived PhysX-cloth Isaac worker against the append-only ledger."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, MutableMapping

from lehome.flywheel.persistent_worker import PersistentRolloutWorker
from lehome.flywheel.task_ledger import TaskLedger


_CUDA_DEVICE = re.compile(r"^cuda:([0-9]+)$")
_DEFAULT_POLICY_REPO = "ryanjin333/lehome-groot-n17-models"
_DEFAULT_POLICY_REVISION = "30ac1a84da67b099e115ad147bcd61e9d60046d3"
_DEFAULT_POLICY_ARTIFACT_SHA256 = "3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06"


class LedgerWorkerController:
    """Give a worker the small controller surface without a second scheduler."""

    def __init__(self, ledger: TaskLedger, *, lease_duration_ns: int) -> None:
        self._ledger = ledger
        self._lease_duration_ns = lease_duration_ns

    def lease_next(self, worker_id: str):
        return self._ledger.lease_next(worker_id, lease_duration_ns=self._lease_duration_ns)

    def record_terminal(self, worker_id: str, attempt_id: str, lease_id: str, raw_artifact_id: str):
        return self._ledger.record_terminal(worker_id, attempt_id, lease_id, raw_artifact_id)

    def record_interrupted(self, worker_id: str, attempt_id: str, lease_id: str, reason: str) -> None:
        self._ledger.record_interrupted(worker_id, attempt_id, lease_id, reason)

    def reject_attempt(self, worker_id: str, attempt_id: str, lease_id: str, *, reason: str) -> str:
        return self._ledger.reject_attempt(worker_id, attempt_id, lease_id, reason=reason)

    def record_infrastructure_abort(self, worker_id: str, attempt_id: str, lease_id: str, *, reason: str) -> str:
        return self._ledger.record_infrastructure_abort(worker_id, attempt_id, lease_id, reason=reason)

    def status(self, attempt_id: str) -> str:
        return self._ledger.status(attempt_id)

    def heartbeat(self, worker_id: str, attempt_id: str, lease_id: str):
        return self._ledger.heartbeat(worker_id, attempt_id, lease_id, lease_duration_ns=self._lease_duration_ns)


def prepare_persistent_cloth_launch(
    args: argparse.Namespace, *, environ: MutableMapping[str, str] | None = None,
) -> str:
    """Bind renderer/policy to CUDA and select the explicit cloth simulator."""

    renderer = _CUDA_DEVICE.fullmatch(str(getattr(args, "renderer_device", "")))
    policy = _CUDA_DEVICE.fullmatch(str(getattr(args, "policy_device", "")))
    if renderer is None or policy is None:
        raise ValueError("renderer and policy devices must be cuda:<physical GPU>")
    if renderer.group(1) != policy.group(1):
        raise ValueError("persistent worker requires policy and renderer on the same physical CUDA device")
    requested_simulator = getattr(args, "simulator_device", None)
    simulator_device = args.renderer_device if requested_simulator is None else str(requested_simulator).lower()
    if simulator_device != "cpu" and simulator_device != args.renderer_device:
        raise ValueError("simulator device must be cpu or the assigned renderer CUDA device")
    target_environ = os.environ if environ is None else environ
    target_environ["LEHOME_FLYWHEEL_WORKER_GPU"] = renderer.group(1)
    args.device = simulator_device
    args.camera_device = args.renderer_device
    return renderer.group(1)


def _load_matrix(path: Path) -> list[Mapping[str, object]]:
    from lehome.flywheel.recovery_collection import load_attempt_matrix

    return load_attempt_matrix(path)


def build_parser() -> argparse.ArgumentParser:
    from scripts.utils.parser import setup_eval_parser

    parser = setup_eval_parser()
    parser.description = __doc__
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--attempt-matrix", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lease-seconds", type=float, default=300.0)
    parser.add_argument(
        "--source-finalization-timeout-seconds", type=float, default=300.0,
        help="bounded wait for accepted or cleanly rejected snapshot-source artifacts",
    )
    parser.add_argument("--max-attempts", type=int, default=400)
    parser.add_argument("--target-accepted", type=int, default=150)
    parser.add_argument("--renderer-device", required=True, help="physical CUDA device used for renderer/cameras")
    parser.add_argument("--policy-device", required=True, help="physical CUDA device used by the policy gateway")
    parser.add_argument(
        "--simulator-device", default=None,
        help="cloth simulator device; CPU is admitted only for snapshot-source bootstrap diagnostics",
    )
    parser.add_argument("--policy-gateway-endpoint", required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--policy-repo", default=_DEFAULT_POLICY_REPO)
    parser.add_argument("--policy-revision", default=_DEFAULT_POLICY_REVISION)
    parser.add_argument("--policy-step", type=int, default=12000)
    parser.add_argument("--policy-artifact-sha256", default=_DEFAULT_POLICY_ARTIFACT_SHA256)
    parser.add_argument("--policy-timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--preparation-timeout-seconds",
        type=float,
        default=180.0,
        help="strict wall-clock bound for one native Isaac garment preparation",
    )
    parser.add_argument("--policy-ready-file", type=Path, required=True)
    parser.add_argument("--initial-garment", required=True)
    return parser


def _progress(message: str) -> None:
    """Write a flushed checkpoint the smoke log can see after a silent Kit exit."""

    line = f"persistent worker: {message}\n"
    print(line, end="", flush=True)
    try:
        path = Path("/eval/logs/worker-progress.txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
    except OSError:
        pass


def _build_cloth_session(args: argparse.Namespace):
    """Import Isaac only after AppLauncher is live, then build one reusable env."""

    _progress(f"importing gym and task for {args.task}")
    try:
        import gymnasium as gym
        import lehome.tasks.bedroom  # noqa: F401 - registers the task with Gym.
        from isaaclab_tasks.utils import parse_env_cfg
        from scripts.eval_policy.groot_policy import SessionPolicyClient
        from scripts.utils.evaluation import EvaluationSession
    except Exception as error:
        _progress(f"import failed: {type(error).__name__}: {error}")
        raise

    _progress("parse_env_cfg")
    env_cfg = parse_env_cfg(args.task, device=args.device)
    env_cfg.sim.use_fabric = False
    env_cfg.seed = args.seed
    env_cfg.random_seed = args.seed
    env_cfg.garment_cfg_base_path = args.garment_cfg_base_path
    env_cfg.particle_cfg_path = args.particle_cfg_path
    env_cfg.garment_name = args.initial_garment
    env_cfg.garment_version = "Release"
    _progress(f"gym.make {args.task} garment={args.initial_garment}")
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    _progress("initialize_obs")
    env.initialize_obs()
    _progress("obs initialized")
    if args.device != "cpu" and (args.device != args.renderer_device or _CUDA_DEVICE.fullmatch(args.device) is None):
        raise ValueError("persistent worker simulator device is not bound to the requested backend")
    args.num_episodes = 1
    args.flywheel_manifest = None
    if args.policy_ready_file.is_symlink() or not args.policy_ready_file.is_file():
        raise ValueError("policy gateway readiness receipt must be a regular file")
    try:
        readiness = json.loads(args.policy_ready_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("policy gateway readiness receipt is invalid") from error
    if (not isinstance(readiness, dict) or readiness.get("ready") is not True
            or readiness.get("policy_sha256") != args.policy_sha256
            or readiness.get("runtime_device") != args.policy_device):
        raise ValueError("policy gateway readiness does not prove the assigned runtime device")
    policy = SessionPolicyClient(
        args.policy_gateway_endpoint, args.policy_sha256, args.policy_timeout_seconds,
        session_id=args.session_id,
    )
    policy.runtime_device = readiness["runtime_device"]
    _progress("EvaluationSession constructed")
    return EvaluationSession(
        args,
        env=env,
        policy=policy,
        env_cfg=env_cfg,
        is_bimanual="bi" in args.task.lower(),
        require_deterministic_seed=True,
    )


def run(args: argparse.Namespace, *, session_factory: Any = None, ledger_factory: Any = TaskLedger) -> list[dict[str, object]]:
    if args.lease_seconds <= 0:
        raise ValueError("lease seconds must be positive")
    if (
        isinstance(args.preparation_timeout_seconds, bool)
        or not isinstance(args.preparation_timeout_seconds, (int, float))
        or not math.isfinite(args.preparation_timeout_seconds)
        or args.preparation_timeout_seconds <= 0
    ):
        raise ValueError("preparation timeout seconds must be positive")
    source_finalization_timeout_seconds = getattr(args, "source_finalization_timeout_seconds", 300.0)
    if (
        isinstance(source_finalization_timeout_seconds, bool)
        or not isinstance(source_finalization_timeout_seconds, (int, float))
        or not math.isfinite(source_finalization_timeout_seconds)
        or source_finalization_timeout_seconds <= 0
    ):
        raise ValueError("source finalization timeout seconds must be positive")
    if args.device != "cpu" and (args.device != args.renderer_device or _CUDA_DEVICE.fullmatch(args.device) is None):
        raise ValueError("persistent worker simulator device is not bound to the requested backend")
    matrix = _load_matrix(args.attempt_matrix)
    if args.device == "cpu":
        from lehome.flywheel.recovery_collection import (
            validate_snapshot_source_descriptor,
            validate_snapshot_source_discovery_descriptor,
        )

        try:
            # The retained one-row historical replay form is validated by its
            # existing checksum/frame gate.  Every fresh CPU discovery row is
            # otherwise an ordinary, same-category source descriptor.
            if len(matrix) == 1 and matrix[0].get("replay_kind") == "verified_success_reset_v1":
                validate_snapshot_source_descriptor(args.attempt_matrix)
            else:
                validate_snapshot_source_discovery_descriptor(args.attempt_matrix)
            if (
                type(args.max_attempts) is not int
                or args.max_attempts != len(matrix)
                or type(args.target_accepted) is not int
                or not 1 <= args.target_accepted <= min(4, len(matrix))
            ):
                raise ValueError("CPU source discovery attempt bounds are invalid")
        except (OSError, TypeError, ValueError) as error:
            raise ValueError(
                "CPU cloth is reserved for bounded snapshot-source bootstrap discovery"
            ) from error
    ledger = ledger_factory(
        args.database, attempt_matrix=matrix, max_attempts=args.max_attempts,
        target_accepted=args.target_accepted,
    )
    try:
        factory = session_factory or (lambda: _build_cloth_session(args))
        policy_holder: dict[str, object] = {}

        def simulator_factory():
            session = factory()
            policy_holder["policy"] = session.policy
            return session

        class _PolicyProxy:
            action_horizon = 16

            def reset(self) -> None:
                policy = policy_holder.get("policy")
                if policy is None:
                    raise RuntimeError("policy is unavailable before session startup")
                policy.reset()

            def __getattr__(self, name: str):
                policy = policy_holder.get("policy")
                if policy is None:
                    raise AttributeError(name)
                return getattr(policy, name)

        worker = PersistentRolloutWorker(
            worker_id=args.worker_id, session_id=args.session_id,
            controller=LedgerWorkerController(ledger, lease_duration_ns=round(args.lease_seconds * 1_000_000_000)),
            simulator_factory=simulator_factory, policy=_PolicyProxy(), output_root=args.output_root,
            renderer_device=args.renderer_device, policy_device=args.policy_device, simulator_device=args.device,
            heartbeat_interval_seconds=max(0.1, args.lease_seconds / 3.0),
            preparation_timeout_seconds=args.preparation_timeout_seconds,
            source_finalization_timeout_seconds=source_finalization_timeout_seconds,
        )
        return worker.run()
    finally:
        ledger.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    # Isaac only enters the worker process after pure CLI validation is set up.
    from isaaclab.app import AppLauncher
    from scripts.utils.common import close_app, launch_app_from_args

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    prepare_persistent_cloth_launch(args)
    simulation_app = launch_app_from_args(args)
    _progress("kit launched")
    try:
        _progress("run() starting")
        try:
            run(args)
        except BaseException as error:
            _progress(f"run() failed before kit close: {type(error).__name__}: {error}")
            raise
        _progress("run() returned")
    finally:
        _progress("closing kit")
        close_app(simulation_app)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as error:
        print(f"persistent worker error: {error}", file=sys.stderr, flush=True)
        try:
            Path("/eval/logs/worker-error.txt").write_text(f"{type(error).__name__}: {error}\n")
        except OSError:
            pass
        raise SystemExit(2)
    except BaseException:
        import traceback
        detail = traceback.format_exc()
        try:
            Path("/eval/logs/worker-error.txt").write_text(detail)
        except OSError:
            pass
        traceback.print_exc()
        raise
