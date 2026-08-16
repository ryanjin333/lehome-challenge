import os
import argparse
import json
import gymnasium as gym
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Mapping, Optional

from isaaclab.envs import DirectRLEnv
from isaaclab_tasks.utils import parse_env_cfg

from scripts.eval_policy import PolicyRegistry
from scripts.eval_policy.base_policy import BasePolicy

from scripts.utils.eval_utils import (
    convert_ee_pose_to_joints,
    save_videos_from_observations,
    calculate_and_print_metrics,
)

from lehome.utils.record import (
    RateLimiter,
    get_next_experiment_path_with_gap,
    append_episode_initial_pose,
)
from .common import stabilize_garment_after_reset
from lehome.utils.logger import get_logger

logger = get_logger(__name__)


def _load_flywheel_manifest(path_value: str | None) -> dict[str, object] | None:
    """Load the opt-in recorder contract without changing legacy evaluation."""
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_symlink() or not path.is_file():
        raise ValueError("flywheel manifest must be a regular file")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("flywheel manifest must be valid JSON") from error
    if not isinstance(manifest, dict) or not isinstance(manifest.get("policy_revision"), str):
        raise ValueError("flywheel manifest requires a policy_revision")
    manifest["_path"] = path
    return manifest


def _flywheel_identity(manifest: dict[str, object] | None):
    """Validate the immutable assignment that a dedicated flywheel worker owns."""
    if manifest is None:
        return None
    from lehome.flywheel.models import EpisodeIdentity

    raw_identity = manifest.get("identity")
    if not isinstance(raw_identity, dict):
        raise ValueError("flywheel manifest requires an immutable identity")
    try:
        identity = EpisodeIdentity(**raw_identity)
    except (TypeError, ValueError) as error:
        raise ValueError("flywheel manifest has an invalid immutable identity") from error
    if manifest.get("episode_id") != identity.episode_id:
        raise ValueError("flywheel manifest episode ID does not match immutable identity")
    if manifest.get("garment") != identity.garment_name:
        raise ValueError("flywheel manifest garment does not match immutable identity")
    if manifest.get("seed") != identity.seed:
        raise ValueError("flywheel manifest seed does not match immutable identity")
    return identity


def _validate_active_flywheel_garment(env: DirectRLEnv, identity) -> None:
    """Fail before recording if the created Isaac environment differs from its manifest."""
    active_cfg = getattr(env, "cfg", None)
    active_name = getattr(active_cfg, "garment_name", None)
    active_version = getattr(active_cfg, "garment_version", None)
    active_object = getattr(env, "object", None)
    active_object_name = getattr(active_object, "prim_name", None)
    if (
        active_name != identity.garment_name
        or active_version != "Release"
        or active_object_name != identity.garment_name
    ):
        raise ValueError("active environment garment does not match immutable flywheel identity")


class EvaluationSession:
    """One reusable environment/policy pair for sequential assigned episodes.

    Legacy :func:`eval` still owns its command-line parsing and garment-list
    selection.  This session only owns resources after they have been created,
    which also lets a persistent worker keep one Isaac application alive while
    changing garments through the environment's native interface.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        *,
        env: DirectRLEnv,
        policy: BasePolicy,
        env_cfg: Any,
        ee_solver: Optional[Any] = None,
        is_bimanual: bool = False,
        env_factory: Any = None,
    ) -> None:
        self.args = args
        self.env = env
        self.policy = policy
        self.env_cfg = env_cfg
        self.ee_solver = ee_solver
        self.is_bimanual = is_bimanual
        self._env_factory = env_factory or (lambda cfg: gym.make(args.task, cfg=cfg).unwrapped)

    @property
    def runtime_receipt(self) -> dict[str, object]:
        """Report the actual CPU-cloth/GPU-render split without guessing it."""

        simulation_device = str(getattr(self.env, "device", getattr(self.args, "device", ""))).lower()
        if simulation_device != "cpu":
            raise ValueError("persistent evaluation requires an environment running CPU cloth simulation")
        observed_devices = getattr(self.env, "flywheel_runtime_devices", None)
        observed_devices = observed_devices() if callable(observed_devices) else None
        if not isinstance(observed_devices, Mapping):
            observed_devices = {
                "renderer_device": getattr(self.env, "renderer_device", None),
                "camera_device": getattr(self.env, "camera_device", None),
            }
        renderer_device = str(observed_devices.get("renderer_device") or "")
        camera_device = str(observed_devices.get("camera_device") or "")
        if not renderer_device or not camera_device:
            raise ValueError("persistent evaluation cannot determine renderer/camera device")
        backend = getattr(self.env, "_flywheel_cloth_backend", None)
        readback = getattr(self.env, "_flywheel_cpu_cloth_state", None)
        canary = getattr(self.env, "flywheel_visible_garment_contact", None)
        if not callable(backend) or not callable(readback) or not callable(canary):
            raise ValueError("persistent evaluation cannot observe CPU cloth/contact canary")
        if backend() != "usd":
            raise ValueError("persistent evaluation requires observed USD CPU cloth backend")
        positions, velocities = readback()
        try:
            position_count, velocity_count = len(positions), len(velocities)
        except TypeError as error:
            raise ValueError("persistent evaluation CPU cloth readback is unavailable") from error
        if position_count <= 0 or velocity_count <= 0 or position_count != velocity_count:
            raise ValueError("persistent evaluation CPU cloth readback is invalid")
        contact = canary()
        if not isinstance(contact, Mapping) or not isinstance(contact.get("observed"), bool):
            raise ValueError("persistent evaluation visible-contact canary is unavailable")
        return {
            "simulation_device": simulation_device,
            "cloth_device": simulation_device,
            "cloth_backend": backend(),
            "cloth_readback": {"positions": position_count, "velocities": velocity_count},
            "visible_contact_canary": dict(contact),
            "renderer_device": renderer_device,
            "camera_device": camera_device,
            "policy_device": str(getattr(self.policy, "runtime_device", "")),
        }

    def prepare_episode(
        self,
        *,
        garment_name: str,
        garment_stage: str = "Release",
        seed: int,
        episode_generation: int,
        reset_policy: bool = True,
    ) -> None:
        """Switch/reset episode-local state without relaunching Isaac when possible."""

        if not isinstance(garment_name, str) or not garment_name:
            raise ValueError("garment_name must be a non-empty string")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(episode_generation, int) or episode_generation < 1:
            raise ValueError("episode_generation must be positive")
        current_name = getattr(self.env_cfg, "garment_name", None)
        current_stage = getattr(self.env_cfg, "garment_version", None)
        if current_name != garment_name or current_stage != garment_stage:
            switch = getattr(self.env, "switch_garment", None)
            if callable(switch):
                switch(garment_name, garment_stage)
            else:
                self.env.close()
                self.env_cfg.garment_name = garment_name
                self.env_cfg.garment_version = garment_stage
                self.env = self._env_factory(self.env_cfg)
                initialize_obs = getattr(self.env, "initialize_obs", None)
                if callable(initialize_obs):
                    initialize_obs()
        self.env_cfg.garment_name = garment_name
        self.env_cfg.garment_version = garment_stage
        for target in (self.env_cfg, getattr(self.env, "cfg", None)):
            if target is not None:
                setattr(target, "seed", seed)
                setattr(target, "random_seed", seed)
        set_seed = getattr(self.env, "set_seed", None)
        if callable(set_seed):
            set_seed(seed)
        reset = getattr(self.policy, "reset", None)
        # Legacy registry adapters are permitted to be stateless.  The
        # persistent worker separately requires ``reset()`` before admitting
        # a session-aware policy to a lease.
        if reset_policy and callable(reset):
            reset()

    def run_episode(
        self,
        *,
        assignment: Dict[str, Any],
        policy: BasePolicy,
        attempt_output_dir: Path | None = None,
        reset_policy: bool = False,
        cancellation_event: Any = None,
    ) -> Dict[str, Any]:
        """Run one already-prepared assignment using the existing evaluation loop."""

        garment_name = assignment.get("garment", assignment.get("garment_name"))
        if not isinstance(garment_name, str) or not garment_name:
            raise ValueError("assignment requires garment")
        # The persistent launcher uses a transparent proxy while it lazily
        # binds its policy client to the one freshly-created Isaac session.
        # Actual inference always remains on ``self.policy``.
        del policy
        if attempt_output_dir is None:
            # The ordinary eval CLI deliberately keeps the exact caller-owned
            # writer roots.  Persistent collection opts in below.
            episode_args = self.args
        else:
            episode_args = argparse.Namespace(**vars(self.args))
            # Keep diagnostics/dataset roots attempt-scoped.  This is the same
            # evaluation writer path the CLI uses, not a second writer.
            episode_args.video_dir = str(attempt_output_dir / "videos")
            episode_args.eval_dataset_path = str(attempt_output_dir / "dataset")
            episode_args.persistent_output_dir = str(attempt_output_dir)
        metrics = run_evaluation_loop(
            env=self.env, policy=self.policy, args=episode_args, ee_solver=self.ee_solver,
            is_bimanual=self.is_bimanual, garment_name=garment_name, reset_policy=reset_policy,
            cancellation_event=cancellation_event,
        )
        return {
            "metrics": metrics,
            "success": bool(metrics and metrics[-1].get("success", False)),
        }

    def close(self) -> None:
        try:
            close_policy = getattr(self.policy, "close", None)
            if callable(close_policy):
                close_policy()
        finally:
            self.env.close()


def run_evaluation_loop(
    env: DirectRLEnv,
    policy: BasePolicy,
    args: argparse.Namespace,
    ee_solver: Optional[Any] = None,
    is_bimanual: bool = False,
    garment_name: Optional[str] = None,
    reset_policy: bool = True,
    cancellation_event: Any = None,
) -> List[Dict[str, Any]]:
    """
    Core evaluation loop.
    Refactored to be agnostic of specific model implementations.
    """

    # --- Dataset Recording Setup (Optional) ---
    eval_dataset = None
    json_path = None
    episode_index = 0
    if args.save_datasets:
        # Dataset recording is optional for rollout evaluation.  Keep LeRobot
        # out of the import path unless this branch is explicitly requested.
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        features = None
        if args.dataset_root and Path(args.dataset_root).exists():
            source_dataset = LeRobotDataset(repo_id="collected_dataset", root=Path(args.dataset_root))
            features = dict(source_dataset.meta.features)
            fps = source_dataset.fps
        else:
            fps = 30  # Default FPS if no source dataset is provided
            action_names = [
                "shoulder_pan", "shoulder_lift", "elbow_flex",
                "wrist_flex", "wrist_roll", "gripper",
            ]
            if is_bimanual:
                left_names = [f"left_{n}" for n in action_names]
                right_names = [f"right_{n}" for n in action_names]
                joint_names = left_names + right_names
            else:
                joint_names = action_names
            dim = len(joint_names)
            features = {
                "observation.state": {
                    "dtype": "float32",
                    "shape": (dim,),
                    "names": joint_names,
                },
                "action": {
                    "dtype": "float32",
                    "shape": (dim,),
                    "names": joint_names,
                },
            }
            image_keys = ["top_rgb", "left_rgb", "right_rgb"] if is_bimanual else ["top_rgb", "wrist_rgb"]
            for key in image_keys:
                features[f"observation.images.{key}"] = {
                    "dtype": "video",
                    "shape": (480, 640, 3),
                    "names": ["height", "width", "channels"],
                }
        root_path = Path(args.eval_dataset_path)
        eval_dataset = LeRobotDataset.create(
            repo_id="lehome_eval",
            fps=fps,
            root=get_next_experiment_path_with_gap(root_path),
            use_videos=True,
            image_writer_threads=8,
            image_writer_processes=0,
            features=features,
        )
        json_path = eval_dataset.root / "meta" / "garment_info.json"

    all_episode_metrics = []
    flywheel_manifest = _load_flywheel_manifest(getattr(args, "flywheel_manifest", None))
    flywheel_identity = _flywheel_identity(flywheel_manifest)
    if flywheel_identity is not None:
        _validate_active_flywheel_garment(env, flywheel_identity)
        if garment_name != flywheel_identity.garment_name:
            raise ValueError("evaluation garment does not match immutable flywheel identity")
    logger.info(f"Starting evaluation: {args.num_episodes} episodes")
    rate_limiter = RateLimiter(args.step_hz)

    for i in range(args.num_episodes):
        # 1. Reset Environment & Policy
        env.reset()
        if reset_policy:
            policy.reset()
        stabilize_garment_after_reset(env, args)

        recorder = None
        reset_snapshot = None
        randomization_receipt = {}
        if flywheel_manifest is not None:
            from lehome.flywheel.isaac_recorder import AutonomousRecorder
            from lehome.flywheel.models import EpisodeIdentity
            from lehome.flywheel.randomization import sample_randomization
            from lehome.flywheel.snapshots import capture_snapshot

            strategy = flywheel_manifest.get("strategy", "canonical")
            sampled = sample_randomization(strategy, seed=flywheel_manifest.get("seed", args.seed))
            randomization_receipt = env.apply_flywheel_randomization(sampled)
            from lehome.flywheel.randomization import validate_randomization_receipt
            validate_randomization_receipt(dict(sampled.values), dict(randomization_receipt))
            identity = flywheel_identity
            if identity is None:  # pragma: no cover - guarded above, keeps type flow explicit.
                raise RuntimeError("missing flywheel identity")
            recorder = AutonomousRecorder(
                Path(flywheel_manifest["_path"]).parent,
                policy_revision=flywheel_manifest["policy_revision"],
                episode_id=flywheel_manifest.get("episode_id"),
                identity=identity,
                provenance={"policy_artifact_sha256": flywheel_manifest["policy_artifact_sha256"], "image_identity": flywheel_manifest["image_identity"], "execution_mode": flywheel_manifest["execution_mode"], "execution_backend": flywheel_manifest["execution_backend"], "simulator_device": flywheel_manifest["simulator_device"], "policy_device": flywheel_manifest.get("policy_device"), "parity_stage": flywheel_manifest.get("parity_stage"), "strategy_sampled": dict(sampled.values), "strategy_receipt": dict(randomization_receipt)},
            )
            reset_snapshot = capture_snapshot(env, randomization={"strategy": strategy, "sampled": dict(sampled.values), "receipt": dict(randomization_receipt)})
            recorder.record_snapshot("reset", reset_snapshot)

        # 2. Initial Observation (Numpy)
        object_initial_pose = env.get_all_pose() if args.save_datasets else None
        observation_dict = env._get_observations()

        # Prepare for video recording
        episode_frames = (
            {k: [] for k in observation_dict.keys() if "images" in k}
            if args.save_video
            else {}
        )

        episode_return = 0.0
        episode_length = 0
        extra_steps = 0
        success_flag = False
        success = torch.tensor(False)
        terminal_reason = "horizon"
        visible_contact = None

        for st in range(args.max_steps):
            if cancellation_event is not None and cancellation_event.is_set():
                raise InterruptedError("persistent worker cancellation requested")
            if rate_limiter:
                rate_limiter.sleep(env)

            # 3. Policy Inference (The core abstraction)
            # Input: Numpy Dict -> Output: Numpy Array
            if recorder is None:
                action_np = policy.select_action(observation_dict)
                action_provenance = None
            else:
                if not hasattr(policy, "select_action_with_provenance"):
                    raise ValueError("flywheel recording requires policy action provenance")
                action_provenance = policy.select_action_with_provenance(observation_dict)
                action_np = action_provenance.value

            # 4. Prepare Action for Environment (Tensor)
            # Convert numpy action to tensor for Isaac Lab
            action = torch.from_numpy(action_np).float().to(args.device).unsqueeze(0)

            # 5. Inverse Kinematics (Optional Helper Logic)
            # If policy outputs EE pose but env needs joints
            if args.use_ee_pose and ee_solver is not None:
                current_joints = (
                    torch.from_numpy(observation_dict["observation.state"])
                    .float()
                    .to(args.device)
                )
                action = convert_ee_pose_to_joints(
                    ee_pose_action=action.squeeze(0),
                    current_joints=current_joints,
                    solver=ee_solver,
                    is_bimanual=is_bimanual,
                    state_unit="rad",
                    device=args.device,
                ).unsqueeze(0)

            # 6. Step Environment
            env.step(action)
            if recorder is not None:
                current_contact = env.flywheel_visible_garment_contact()
                if visible_contact is None or current_contact["minimum_distance_m"] < visible_contact["minimum_distance_m"]:
                    visible_contact = current_contact
                elif current_contact["observed"]:
                    visible_contact = current_contact

            # Check success first
            if not success_flag:
                success = env._get_success()
                if success.item():
                    success_flag = True
                    extra_steps = 50  # Run a bit longer after success to settle

            # Get reward from environment (Isaac Lab stores rewards internally)
            reward_value = env._get_rewards()
            if isinstance(reward_value, torch.Tensor):
                reward = reward_value.item()
            else:
                reward = float(reward_value)

            # Accumulate reward for all steps (including post-success steps)
            episode_return += reward
            # Only count length before success (for consistency with episode termination)
            if not success_flag:
                episode_length += 1

            if recorder is not None:
                recorder.record_step(
                    observation_dict,
                    action.detach().cpu().numpy().squeeze(0),
                    reward=reward,
                    success=bool(success_flag),
                    request_id=action_provenance.request_id,
                    chunk_offset=action_provenance.chunk_offset,
                )

            # Update Observation
            observation_dict = env._get_observations()

            # Recording
            if args.save_datasets:
                frame = {
                    k: v
                    for k, v in observation_dict.items()
                    if k != "observation.top_depth"
                }
                frame["task"] = args.task_description
                eval_dataset.add_frame(frame)

            if args.save_video:
                for key, val in observation_dict.items():
                    if "images" in key:
                        episode_frames[key].append(val.copy())

            if success_flag:
                extra_steps -= 1
                if extra_steps <= 0:
                    terminal_reason = "success"
                    break

        # --- End of Episode Handling ---
        is_success = success.item() if success_flag else False

        if recorder is not None:
            from lehome.flywheel.snapshots import capture_snapshot
            recorder.record_snapshot("terminal", capture_snapshot(env, randomization={"receipt": dict(randomization_receipt)}))
            if visible_contact is None:
                raise RuntimeError("flywheel trial did not record simulator robot-garment contact evidence")
            recorder.finish(
                reason=terminal_reason,
                accepted_success=bool(is_success),
                visible_contact=visible_contact,
            )

        # Save Datasets
        if args.save_datasets:
            if success_flag:
                eval_dataset.save_episode()
                append_episode_initial_pose(
                    json_path,
                    episode_index,
                    object_initial_pose,
                    garment_name=garment_name,
                )
                episode_index += 1
            else:
                eval_dataset.clear_episode_buffer()

        # Save Videos (Using generic util)
        if args.save_video:
            save_videos_from_observations(
                episode_frames,
                success=success if success_flag else torch.tensor(False),
                save_dir=args.video_dir,
                episode_idx=i,
            )

        # Log Metrics
        all_episode_metrics.append(
            {"return": episode_return, "length": episode_length, "success": is_success}
        )
        logger.info(
            f"Episode {i + 1}/{args.num_episodes}: Return={episode_return:.2f}, Length={episode_length}, Success={is_success}"
        )

    return all_episode_metrics


def eval(args: argparse.Namespace, simulation_app: Any) -> None:
    """
    Main entry point for evaluation logic.
    """
    # One flywheel worker owns exactly one manifest garment.  Resolve this before
    # allocating policy/Isaac resources so a mismatched invocation cannot record.
    flywheel_manifest = _load_flywheel_manifest(getattr(args, "flywheel_manifest", None))
    flywheel_identity = _flywheel_identity(flywheel_manifest)
    if flywheel_identity is not None:
        if args.num_episodes != 1:
            raise ValueError("flywheel workers must run exactly one episode")
        if args.garment_name != flywheel_identity.garment_name:
            raise ValueError("requested garment does not match immutable flywheel identity")

    # 1. Environment Configuration
    env_cfg = parse_env_cfg(args.task, device=args.device)
    env_cfg.sim.use_fabric = False
    if args.use_random_seed:
        env_cfg.use_random_seed = True
    else:
        env_cfg.use_random_seed = False
        env_cfg.seed = args.seed
        env_cfg.random_seed = args.seed
        # Propagate seed to sim config if structure exists
        if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "seed"):
            env_cfg.sim.seed = args.seed

    env_cfg.garment_cfg_base_path = args.garment_cfg_base_path
    env_cfg.particle_cfg_path = args.particle_cfg_path

    # 2. Initialize Policy (Using the Policy Registry)
    # This replaces create_il_policy, make_pre_post_processors, etc.
    logger.info(f"Initializing Policy Type: {args.policy_type}")

    # Check if policy is registered
    if not PolicyRegistry.is_registered(args.policy_type):
        available_policies = PolicyRegistry.list_policies()
        raise ValueError(
            f"Policy type '{args.policy_type}' not found in registry. "
            f"Available policies: {', '.join(available_policies)}"
        )

    device = args.device if args.policy_type == "groot" else ("cuda" if torch.cuda.is_available() else "cpu")
    is_bimanual = "Bi" in args.task or "bi" in args.task.lower()

    # Create policy instance from registry with appropriate arguments
    # Different policies may require different initialization arguments
    policy_kwargs = {
        "device": device,
    }

    if args.policy_type == "lerobot":
        # LeRobot policy requires policy_path and dataset_root
        if not args.policy_path:
            raise ValueError("--policy_path is required for lerobot policy type")
        if not args.dataset_root:
            raise ValueError("--dataset_root is required for lerobot policy type")
        policy_kwargs.update(
            {
                "policy_path": args.policy_path,
                "dataset_root": args.dataset_root,
                "task_description": args.task_description,
            }
        )
    elif args.policy_type == "docker":
        # Docker policy connects to an external container
        policy_kwargs["docker_url"] = args.docker_url
    elif args.policy_type == "groot":
        if not args.policy_path:
            raise ValueError("--policy_path is required for groot policy type")
        policy_kwargs.update(
            {
                "model_path": args.policy_path,
                "task_description": args.task_description,
            }
        )
    elif args.policy_type == "groot_server":
        if not args.policy_path:
            raise ValueError("--policy_path is required for groot_server policy type")
        endpoint = getattr(args, "policy_server_endpoint", None)
        token_env = getattr(args, "policy_server_token_env", None)
        request_timeout = getattr(args, "policy_server_request_timeout", None)
        if not endpoint or not token_env or request_timeout is None:
            raise ValueError("groot_server requires the dedicated policy-server trial boundary")
        policy_kwargs.update(
            {
                "model_path": args.policy_path,
                "policy_server_endpoint": endpoint,
                "policy_server_token_env": token_env,
                "policy_server_request_timeout": request_timeout,
                "task_description": args.task_description,
            }
        )
    else:
        # For custom policies, pass policy_path as model_path if provided
        if args.policy_path:
            policy_kwargs["model_path"] = args.policy_path

    # Create policy from registry
    policy = PolicyRegistry.create(args.policy_type, **policy_kwargs)
    logger.info(f"Policy '{args.policy_type}' loaded successfully")

    # 3. Initialize IK Solver (If needed)
    ee_solver = None
    if args.use_ee_pose:
        from lehome.utils import RobotKinematics

        urdf_path = args.ee_urdf_path  # Assuming path is handled or add check logic
        joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ]
        ee_solver = RobotKinematics(
            str(urdf_path),
            target_frame_name="gripper_frame_link",
            joint_names=joint_names,
        )
        logger.info(f"IK solver loaded.")

    # 4. Load Evaluation List
    # Only loads from 'Release' directory based on garment_type
    eval_list = []  # List of (name, stage)

    # The manifest is authoritative.  Legacy custom mode remains list-driven
    # only when the caller did not opt into a flywheel recording contract.
    if flywheel_identity is not None:
        eval_list.append((flywheel_identity.garment_name, "Release"))
    # Evaluate a specific category based on garment_type
    elif args.garment_type == "custom":
        # For 'custom' type, we load from the root Release_test_list.txt
        eval_list_path = os.path.join(
            args.garment_cfg_base_path, "Release", "Release_test_list.txt"
        )
    else:
        # Map argument to specific sub-category directory
        type_map = {
            "top_long": "Top_Long",
            "top_short": "Top_Short",
            "pant_long": "Pant_Long",
            "pant_short": "Pant_Short",
        }
        file_prefix = type_map.get(args.garment_type, "Top_Long")
        # Path: Assets/objects/Challenge_Garment/Release/Top_Long/Top_Long.txt
        eval_list_path = os.path.join(
            args.garment_cfg_base_path, "Release", file_prefix, f"{file_prefix}.txt"
        )

    if flywheel_identity is None:
        logger.info(
            f"Loading evaluation list for category '{args.garment_type}' from: {eval_list_path}"
        )

        if not os.path.exists(eval_list_path):
            raise FileNotFoundError(f"Evaluation list not found: {eval_list_path}")

        with open(eval_list_path, "r") as f:
            names = [line.strip() for line in f.readlines() if line.strip()]
            for name in names:
                eval_list.append((name, "Release"))

    logger.info(f"Loaded {len(eval_list)} garments for category: {args.garment_type}")

    if not eval_list:
        raise ValueError(
            f"No garments found to evaluate for category '{args.garment_type}'."
        )

    # 5. Main Evaluation Loops
    all_garment_metrics = []

    # Init Env with first garment
    first_name, first_stage = eval_list[0]
    env_cfg.garment_name = first_name
    env_cfg.garment_version = first_stage
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.initialize_obs()
    if flywheel_identity is not None:
        _validate_active_flywheel_garment(env, flywheel_identity)
    session = EvaluationSession(
        args, env=env, policy=policy, env_cfg=env_cfg, ee_solver=ee_solver,
        is_bimanual=is_bimanual,
    )

    try:
        for garment_idx, (garment_name, garment_stage) in enumerate(eval_list):
            logger.info(
                f"Evaluating: {garment_name} ({garment_stage}) ({garment_idx+1}/{len(eval_list)})"
            )

            session.prepare_episode(
                garment_name=garment_name, garment_stage=garment_stage,
                seed=args.seed, episode_generation=garment_idx + 1, reset_policy=False,
            )
            if flywheel_identity is not None:
                _validate_active_flywheel_garment(session.env, flywheel_identity)

            # Run Loop
            metrics = session.run_episode(
                assignment={"garment": garment_name},
                policy=policy,
                reset_policy=True,
            )["metrics"]

            all_garment_metrics.append(
                {"garment_name": garment_name, "metrics": metrics}
            )

    finally:
        session.close()

    # Print summary across all garments
    logger.info("=" * 60)
    logger.info("Overall Summary")
    logger.info("=" * 60)

    if all_garment_metrics:
        # Aggregate all episode metrics
        all_episodes = []
        for garment_data in all_garment_metrics:
            for episode_metric in garment_data["metrics"]:
                episode_metric["garment_name"] = garment_data["garment_name"]
                all_episodes.append(episode_metric)

        # Print overall metrics
        calculate_and_print_metrics(all_episodes)

        # Print per-garment summary
        logger.info("=" * 60)
        logger.info("Per-Garment Summary")
        logger.info("=" * 60)
        for garment_data in all_garment_metrics:
            garment_name = garment_data["garment_name"]
            metrics = garment_data["metrics"]
            success_count = sum(1 for m in metrics if m["success"])
            success_rate = success_count / len(metrics) if metrics else 0.0
            avg_return = np.mean([m["return"] for m in metrics]) if metrics else 0.0
            logger.info(
                f"  {garment_name}: Success Rate = {success_rate:.2%}, Avg Return = {avg_return:.2f}"
            )
    else:
        logger.info("No metrics collected (all evaluations failed)")

    logger.info("=" * 60)
    logger.info("Evaluation completed successfully")
    logger.info("=" * 60)
