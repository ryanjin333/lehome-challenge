"""Run one long-lived CPU-cloth Isaac worker against the append-only ledger."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, MutableMapping

from lehome.flywheel.persistent_worker import PersistentRolloutWorker
from lehome.flywheel.task_ledger import TaskLedger


_CUDA_DEVICE = re.compile(r"^cuda:([0-9]+)$")


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

    def heartbeat(self, worker_id: str, attempt_id: str, lease_id: str):
        return self._ledger.heartbeat(worker_id, attempt_id, lease_id, lease_duration_ns=self._lease_duration_ns)


def prepare_cpu_cloth_launch(args: argparse.Namespace, *, environ: MutableMapping[str, str] | None = None) -> str:
    """Validate the physical renderer before fixing simulation/cloth to CPU."""

    renderer = _CUDA_DEVICE.fullmatch(str(getattr(args, "renderer_device", "")))
    policy = _CUDA_DEVICE.fullmatch(str(getattr(args, "policy_device", "")))
    if renderer is None or policy is None:
        raise ValueError("renderer and policy devices must be cuda:<physical GPU>")
    target_environ = os.environ if environ is None else environ
    target_environ["LEHOME_FLYWHEEL_WORKER_GPU"] = renderer.group(1)
    # ``launch_app_from_args`` reads this environment variable to set Kit's
    # renderer GPU while preserving this CPU value for task/cloth config.
    args.device = "cpu"
    args.camera_device = args.renderer_device
    return renderer.group(1)


def _load_matrix(path: Path) -> list[Mapping[str, object]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("attempt matrix must be a regular JSON file")
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("attempt matrix must be valid JSON") from error
    if not isinstance(decoded, list) or not all(isinstance(item, dict) for item in decoded):
        raise ValueError("attempt matrix must be a JSON array of assignments")
    return decoded


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
    parser.add_argument("--max-attempts", type=int, default=400)
    parser.add_argument("--target-accepted", type=int, default=150)
    parser.add_argument("--renderer-device", required=True, help="physical CUDA device used for renderer/cameras")
    parser.add_argument("--policy-device", required=True, help="physical CUDA device used by the policy gateway")
    parser.add_argument("--policy-gateway-endpoint", required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--policy-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--policy-ready-file", type=Path, required=True)
    parser.add_argument("--initial-garment", required=True)
    return parser


def _build_cpu_cloth_session(args: argparse.Namespace):
    """Import Isaac only after AppLauncher is live, then build one reusable env."""

    import gymnasium as gym
    import lehome.tasks.bedroom  # noqa: F401 - registers the task with Gym.
    from isaaclab_tasks.utils import parse_env_cfg
    from scripts.eval_policy.groot_policy import SessionPolicyClient
    from scripts.utils.evaluation import EvaluationSession

    env_cfg = parse_env_cfg(args.task, device="cpu")
    env_cfg.sim.use_fabric = False
    env_cfg.seed = args.seed
    env_cfg.random_seed = args.seed
    env_cfg.garment_cfg_base_path = args.garment_cfg_base_path
    env_cfg.particle_cfg_path = args.particle_cfg_path
    env_cfg.garment_name = args.initial_garment
    env_cfg.garment_version = "Release"
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.initialize_obs()
    if args.device != "cpu":
        raise ValueError("persistent worker must force CPU physics before environment creation")
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
    return EvaluationSession(args, env=env, policy=policy, env_cfg=env_cfg, is_bimanual="bi" in args.task.lower())


def run(args: argparse.Namespace, *, session_factory: Any = None, ledger_factory: Any = TaskLedger) -> list[dict[str, object]]:
    if args.device != "cpu":
        raise ValueError("persistent worker requires CPU cloth physics")
    if args.lease_seconds <= 0:
        raise ValueError("lease seconds must be positive")
    ledger = ledger_factory(
        args.database, attempt_matrix=_load_matrix(args.attempt_matrix), max_attempts=args.max_attempts,
        target_accepted=args.target_accepted,
    )
    try:
        factory = session_factory or (lambda: _build_cpu_cloth_session(args))
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
            renderer_device=args.renderer_device, policy_device=args.policy_device,
            heartbeat_interval_seconds=max(0.1, args.lease_seconds / 3.0),
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
    prepare_cpu_cloth_launch(args)
    simulation_app = launch_app_from_args(args)
    try:
        run(args)
    finally:
        close_app(simulation_app)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as error:
        print(f"persistent worker error: {error}", file=sys.stderr)
        raise SystemExit(2)
