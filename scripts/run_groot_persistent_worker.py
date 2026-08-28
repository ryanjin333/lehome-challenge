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
from lehome.flywheel.task_ledger import MAX_CAMPAIGN_ATTEMPTS, TaskLedger


_CUDA_DEVICE = re.compile(r"^cuda:([0-9]+)$")
_DEFAULT_POLICY_REPO = "ryanjin333/lehome-groot-n17-models"
_DEFAULT_POLICY_REVISION = "30ac1a84da67b099e115ad147bcd61e9d60046d3"
_DEFAULT_POLICY_ARTIFACT_SHA256 = "3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06"
_TERMINAL_EVALUATION_CATEGORIES = ("top_long", "top_short", "pant_long", "pant_short")
_TERMINAL_EVALUATION_ROW_KEYS = {"trial_id", "category", "garment_name", "release_stage", "seed"}
_TERMINAL_80_ALIAS_KEYS = {"attempt_id", "garment"}


def simple_curriculum_collection_from_environ(environ: Mapping[str, str]) -> bool:
    """Parse the one outer-process marker without leaking environment reads inward."""

    value = environ.get("LEHOME_SIMPLE_CURRICULUM_COLLECTION", "0")
    if value not in {"0", "1"}:
        raise ValueError("LEHOME_SIMPLE_CURRICULUM_COLLECTION must be exactly 0 or 1")
    return value == "1"


class LedgerWorkerController:
    """Give a worker the small controller surface without a second scheduler."""

    def __init__(
        self,
        ledger: TaskLedger,
        *,
        lease_duration_ns: int,
        retry_infrastructure_aborts: bool = False,
        assignment_filter: Mapping[str, object] | None = None,
    ) -> None:
        self._ledger = ledger
        self._lease_duration_ns = lease_duration_ns
        self._retry_infrastructure_aborts = retry_infrastructure_aborts
        self._assignment_filter = assignment_filter

    def lease_next(self, worker_id: str):
        kwargs: dict[str, object] = {"lease_duration_ns": self._lease_duration_ns}
        if self._assignment_filter is not None:
            kwargs["assignment_filter"] = self._assignment_filter
        return self._ledger.lease_next(worker_id, **kwargs)

    def record_terminal(self, worker_id: str, attempt_id: str, lease_id: str, raw_artifact_id: str):
        return self._ledger.record_terminal(worker_id, attempt_id, lease_id, raw_artifact_id)

    def record_interrupted(self, worker_id: str, attempt_id: str, lease_id: str, reason: str) -> None:
        self._ledger.record_interrupted(worker_id, attempt_id, lease_id, reason)

    def reject_attempt(self, worker_id: str, attempt_id: str, lease_id: str, *, reason: str) -> str:
        return self._ledger.reject_attempt(worker_id, attempt_id, lease_id, reason=reason)

    def record_infrastructure_abort(self, worker_id: str, attempt_id: str, lease_id: str, *, reason: str) -> str:
        if self._retry_infrastructure_aborts:
            return self._ledger.record_interrupted(
                worker_id,
                attempt_id,
                lease_id,
                f"infrastructure_retry:{reason}",
            )
        return self._ledger.record_infrastructure_abort(worker_id, attempt_id, lease_id, reason=reason)

    def record_fidelity_abort(
        self,
        worker_id: str,
        attempt_id: str,
        lease_id: str,
        *,
        session_id: str,
        generation: int,
        fidelity_code: str,
        fidelity: Mapping[str, object],
        runtime: Mapping[str, object],
    ) -> str:
        return self._ledger.record_fidelity_abort(
            worker_id,
            attempt_id,
            lease_id,
            session_id=session_id,
            generation=generation,
            fidelity_code=fidelity_code,
            fidelity=fidelity,
            runtime=runtime,
        )

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


def _is_cpu_terminal_evaluation(matrix: list[Mapping[str, object]], args: argparse.Namespace) -> bool:
    """Accept only the terminal-evaluation contract already admitted by the shell."""

    if (
        os.environ.get("LEHOME_EVALUATION_TERMINAL_UPLOAD") != "1"
        or len(matrix) not in {20, 24, 80}
        or type(args.max_attempts) is not int
        or args.max_attempts != len(matrix)
        or type(args.target_accepted) is not int
        or args.target_accepted != len(matrix)
    ):
        return False
    seen_development = len(matrix) == 24
    seen80 = len(matrix) == 80 and all(row.get("release_stage") == "seen" for row in matrix)
    expected_release_stage = "seen" if seen_development or seen80 else "public_unseen"
    exact_eighty = len(matrix) == 80
    base_row_keys = _TERMINAL_EVALUATION_ROW_KEYS
    aliased_row_keys = base_row_keys | _TERMINAL_80_ALIAS_KEYS
    row_key_shapes = {frozenset(row) for row in matrix}
    if row_key_shapes not in ({frozenset(base_row_keys)}, {frozenset(aliased_row_keys)}):
        return False
    has_aliases = row_key_shapes == {frozenset(aliased_row_keys)}
    expected_per_category = len(matrix) // len(_TERMINAL_EVALUATION_CATEGORIES)
    category_counts = {category: 0 for category in _TERMINAL_EVALUATION_CATEGORIES}
    trial_ids: set[str] = set()
    identities: set[tuple[object, ...]] = set()
    for row in matrix:
        trial_id = row.get("trial_id")
        category = row.get("category")
        garment_name = row.get("garment_name")
        release_stage = row.get("release_stage")
        seed = row.get("seed")
        if (
            type(trial_id) is not str
            or not trial_id
            or category not in category_counts
            or type(garment_name) is not str
            or not garment_name
            or release_stage != expected_release_stage
            or type(seed) is not int
            or seed < 0
        ):
            return False
        if has_aliases and (row.get("attempt_id") != trial_id or row.get("garment") != garment_name):
            return False
        identity = (category, garment_name, seed)
        if trial_id in trial_ids or identity in identities:
            return False
        trial_ids.add(trial_id)
        identities.add(identity)
        category_counts[category] += 1
    if not all(count == expected_per_category for count in category_counts.values()):
        return False
    if not seen_development and not seen80:
        return True
    prefixes = {
        "top_long": "Top_Long", "top_short": "Top_Short",
        "pant_long": "Pant_Long", "pant_short": "Pant_Short",
    }
    if seen_development:
        expected_rows = {
            (
                f"{category.replace('_', '-')}-seen-{garment_index}-seed-{seed}",
                category,
                f"{prefixes[category]}_Seen_{garment_index}",
                "seen",
                seed,
            )
            for category in _TERMINAL_EVALUATION_CATEGORIES
            for garment_index in range(2)
            for seed in (42, 43, 44)
        }
    else:
        seed_bases = {
            "top_long": 970_000, "top_short": 971_000,
            "pant_long": 972_000, "pant_short": 973_000,
        }
        expected_rows = {
            (
                f"{category.replace('_', '-')}-seen-{garment_index}-seed-{seed}",
                category,
                f"{prefixes[category]}_Seen_{garment_index}",
                "seen",
                seed,
            )
            for category in _TERMINAL_EVALUATION_CATEGORIES
            for garment_index in range(10)
            for seed in range(
                seed_bases[category] + garment_index * 2,
                seed_bases[category] + garment_index * 2 + 2,
            )
        }
    return {
        (row["trial_id"], row["category"], row["garment_name"], row["release_stage"], row["seed"])
        for row in matrix
    } == expected_rows


def _is_exact_simple_curriculum_partition(matrix: list[Mapping[str, object]], args: argparse.Namespace) -> bool:
    """Admit only one frozen physical partition of the fresh source matrices."""

    contracts = {
        "calibration-head": (100, 100, 150, "calibration"),
        "calibration-tail": (300, 300, 400, "calibration"),
        "curriculum-a": (300, 300, 400, "curriculum"),
        "curriculum-b": (300, 300, 400, "curriculum"),
    }
    if args.device != "cpu" or getattr(args, "completion_metric", "accepted_successes") != "terminal_outcomes" or not matrix:
        return False
    partition_ids = {row.get("partition_id") for row in matrix}
    parent_hashes = {row.get("parent_matrix_sha256") for row in matrix}
    if len(partition_ids) != 1 or len(parent_hashes) != 1:
        return False
    partition_id = next(iter(partition_ids))
    parent_sha = next(iter(parent_hashes))
    contract = contracts.get(partition_id)
    if (
        contract is None or not isinstance(parent_sha, str) or re.fullmatch(r"[0-9a-f]{64}", parent_sha) is None
        or (len(matrix), args.target_accepted, args.max_attempts) != contract[:3]
    ):
        return False
    patterns = {
        "top_long": r"Top_Long_Seen_[0-9]", "top_short": r"Top_Short_Seen_[0-9]",
        "pant_long": r"Pant_Long_Seen_[0-9]", "pant_short": r"Pant_Short_Seen_[0-9]",
    }
    attempt_ids: set[str] = set()
    trial_ids: set[str] = set()
    seeds: set[int] = set()
    for row in matrix:
        attempt_id, trial_id, seed = row.get("attempt_id"), row.get("trial_id"), row.get("seed")
        category, garment = row.get("category"), row.get("garment")
        if (
            row.get("campaign_kind") != "simple_curriculum_source_v1" or row.get("logical_stage") != contract[3]
            or row.get("strategy") != "canonical" or row.get("release_stage") != "seen"
            or not isinstance(attempt_id, str) or not attempt_id or not isinstance(trial_id, str) or not trial_id
            or type(seed) is not int or not 0 <= seed < 2**32 or row.get("source_seed") != seed
            or category not in patterns or not isinstance(garment, str) or row.get("garment_name") != garment
            or re.fullmatch(patterns[category], garment) is None
            or attempt_id in attempt_ids or trial_id in trial_ids or seed in seeds
        ):
            return False
        attempt_ids.add(attempt_id)
        trial_ids.add(trial_id)
        seeds.add(seed)
    return True


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
    parser.add_argument("--completion-metric", choices=("accepted_successes", "terminal_outcomes"), default="accepted_successes")
    parser.add_argument("--renderer-device", required=True, help="physical CUDA device used for renderer/cameras")
    parser.add_argument("--policy-device", required=True, help="physical CUDA device used by the policy gateway")
    parser.add_argument(
        "--simulator-device", default=None,
        help="cloth simulator device; CPU is admitted only for authenticated recovery or terminal evaluation",
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
    if args.device == "cpu":
        env_cfg.wait_for_textures = False
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
    terminal_evaluation = os.environ.get("LEHOME_EVALUATION_TERMINAL_UPLOAD") == "1"
    affinity_value = os.environ.get("LEHOME_EVALUATION_GARMENT_AFFINITY", "0")
    if affinity_value not in {"0", "1"}:
        raise ValueError("LEHOME_EVALUATION_GARMENT_AFFINITY must be exactly 0 or 1")
    garment_affinity = affinity_value == "1"
    success_replay_campaign = os.environ.get("LEHOME_SUCCESS_REPLAY_CAMPAIGN") == "1"
    hard_state_campaign = os.environ.get("LEHOME_HARD_STATE_CAMPAIGN") == "1"
    simple_curriculum_collection = getattr(args, "simple_curriculum_collection", False)
    if type(simple_curriculum_collection) is not bool:
        raise ValueError("simple_curriculum_collection must be a boolean")
    if sum((terminal_evaluation, success_replay_campaign, hard_state_campaign, simple_curriculum_collection)) > 1:
        raise ValueError("persistent CPU campaign mode markers are mutually exclusive")
    if terminal_evaluation and args.device != "cpu":
        raise ValueError("terminal evaluation requires CPU cloth")
    matrix = _load_matrix(args.attempt_matrix)
    source_discovery = bool(matrix) and all(
        row.get("snapshot_source_bootstrap") is True for row in matrix
    )
    if terminal_evaluation and not garment_affinity:
        raise ValueError("terminal CPU evaluation requires garment affinity")
    if simple_curriculum_collection and not garment_affinity:
        raise ValueError("simple curriculum CPU collection requires garment affinity")
    if garment_affinity and not (terminal_evaluation or source_discovery or simple_curriculum_collection):
        raise ValueError("garment affinity is reserved for terminal CPU evaluation, source discovery, or simple curriculum collection")
    if simple_curriculum_collection:
        if not _is_exact_simple_curriculum_partition(matrix, args):
            raise ValueError("simple curriculum CPU partition is invalid")
        if not any(row.get("garment_name") == args.initial_garment for row in matrix):
            raise ValueError("garment affinity is absent from the simple curriculum matrix")
    elif terminal_evaluation:
        if not _is_cpu_terminal_evaluation(matrix, args):
            raise ValueError("terminal evaluation matrix is invalid")
        if garment_affinity and not any(
            row.get("garment_name") == args.initial_garment for row in matrix
        ):
            raise ValueError("garment affinity is absent from the terminal evaluation matrix")
    elif source_discovery and garment_affinity and not any(
        (row.get("garment_name") or row.get("garment")) == args.initial_garment
        for row in matrix
    ):
        raise ValueError("garment affinity is absent from the source discovery matrix")
    elif args.device == "cpu":
        controlled_teacher_smoke = (
            len(matrix) == 1
            and args.max_attempts == 1
            and args.target_accepted == 1
            and matrix[0].get("recovery_kind") == "controlled_success_recovery_snapshot_v3"
            and matrix[0].get("controlled_smoke") is True
            and matrix[0].get("controlled_smoke_zero_perturbation") is True
            and matrix[0].get("controlled_smoke_teacher_probe") is True
            and matrix[0].get("controlled_smoke_perturbation_mode")
            == "zero_perturbation_teacher_continuation_probe_v1"
        )
        if success_replay_campaign:
            from lehome.flywheel.recovery_collection import validate_success_replay_descriptor

            try:
                validate_success_replay_descriptor(args.attempt_matrix)
                if (
                    type(args.max_attempts) is not int
                    or args.max_attempts != len(matrix)
                    or type(args.target_accepted) is not int
                    or not 1 <= args.target_accepted <= min(150, len(matrix))
                ):
                    raise ValueError("CPU success replay attempt bounds are invalid")
            except (OSError, TypeError, ValueError) as error:
                raise ValueError("CPU success replay campaign is invalid") from error
        elif hard_state_campaign:
            from lehome.flywheel.recovery_collection import validate_hard_state_descriptor

            try:
                validate_hard_state_descriptor(args.attempt_matrix)
                if (
                    type(args.max_attempts) is not int
                    or args.max_attempts != len(matrix)
                    or type(args.target_accepted) is not int
                    or not 1 <= args.target_accepted <= min(150, len(matrix))
                ):
                    raise ValueError("CPU hard-state attempt bounds are invalid")
            except (OSError, TypeError, ValueError) as error:
                raise ValueError("CPU hard-state campaign is invalid") from error
        elif controlled_teacher_smoke:
            pass  # _load_matrix already authenticates the descriptor mode identity.
        elif _is_cpu_terminal_evaluation(matrix, args):
            pass  # The shell marker and frozen public-unseen matrix bind terminal evaluation.
        else:
            from lehome.flywheel.recovery_collection import (
                validate_snapshot_source_descriptor,
                validate_snapshot_source_discovery_descriptor,
            )

            try:
                # The retained one-row historical replay form is validated by its
                # existing checksum/frame gate. Every fresh CPU discovery row is
                # otherwise an ordinary bounded source descriptor.
                if len(matrix) == 1 and matrix[0].get("replay_kind") in {
                    "verified_success_reset_v1", "verified_success_early_snapshot_v1",
                }:
                    validate_snapshot_source_descriptor(args.attempt_matrix)
                else:
                    validate_snapshot_source_discovery_descriptor(args.attempt_matrix)
                if (
                    type(args.max_attempts) is not int
                    or args.max_attempts != len(matrix)
                    or type(args.target_accepted) is not int
                    or not 1 <= args.target_accepted <= min(150, len(matrix))
                ):
                    raise ValueError("CPU source discovery attempt bounds are invalid")
            except (OSError, TypeError, ValueError) as error:
                raise ValueError(
                    "CPU cloth is reserved for bounded snapshot-source bootstrap discovery or terminal public-unseen evaluation"
                ) from error
    ledger = ledger_factory(
        args.database, attempt_matrix=matrix,
        max_attempts=MAX_CAMPAIGN_ATTEMPTS if terminal_evaluation else args.max_attempts,
        target_accepted=args.target_accepted,
        completion_metric=getattr(args, "completion_metric", "accepted_successes"),
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
            controller=LedgerWorkerController(
                ledger,
                lease_duration_ns=round(args.lease_seconds * 1_000_000_000),
                retry_infrastructure_aborts=terminal_evaluation or simple_curriculum_collection,
                assignment_filter=(
                    {
                        "garment" if source_discovery else "garment_name":
                            args.initial_garment
                    }
                    if garment_affinity else None
                ),
            ),
            simulator_factory=simulator_factory, policy=_PolicyProxy(), output_root=args.output_root,
            renderer_device=args.renderer_device, policy_device=args.policy_device, simulator_device=args.device,
            heartbeat_interval_seconds=max(0.1, args.lease_seconds / 3.0),
            preparation_timeout_seconds=args.preparation_timeout_seconds,
            source_finalization_timeout_seconds=source_finalization_timeout_seconds,
            simple_curriculum_collection=getattr(args, "simple_curriculum_collection", False),
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
    args.simple_curriculum_collection = simple_curriculum_collection_from_environ(os.environ)
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
