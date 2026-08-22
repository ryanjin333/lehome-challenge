import os
import argparse
import hashlib
import json
import re
import stat
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
from lehome.flywheel.persistent_worker import SimulatorNumericalDivergenceError

logger = get_logger(__name__)

_FLYWHEEL_POLICY_ACTION_JOINT_NAMES = (
    "left_shoulder_pan",
    "left_shoulder_lift",
    "left_elbow_flex",
    "left_wrist_flex",
    "left_wrist_roll",
    "left_gripper",
    "right_shoulder_pan",
    "right_shoulder_lift",
    "right_elbow_flex",
    "right_wrist_flex",
    "right_wrist_roll",
    "right_gripper",
)


def _flywheel_policy_action_limit_diagnostics(env: Any, action: Any) -> dict[str, object]:
    """Return bounded target-limit and live-position diagnostics for flywheel steps."""

    def numpy_array(value: Any) -> np.ndarray:
        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach()
        to_cpu = getattr(value, "cpu", None)
        if callable(to_cpu):
            value = to_cpu()
        to_numpy = getattr(value, "numpy", None)
        if callable(to_numpy):
            value = to_numpy()
        return np.asarray(value, dtype=np.float32)

    try:
        values = numpy_array(action)
        left_limits = numpy_array(env.left_arm.data.soft_joint_pos_limits)
        right_limits = numpy_array(env.right_arm.data.soft_joint_pos_limits)
        left_positions = numpy_array(env.left_arm.data.joint_pos)
        right_positions = numpy_array(env.right_arm.data.joint_pos)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return {"policy_action_limits_available": False}
    if (
        values.shape != (1, 12)
        or left_limits.shape != (1, 6, 2)
        or right_limits.shape != (1, 6, 2)
        or left_positions.shape != (1, 6)
        or right_positions.shape != (1, 6)
    ):
        return {"policy_action_limits_available": False}
    limits = np.concatenate((left_limits[0], right_limits[0]), axis=0)
    positions = np.concatenate((left_positions[0], right_positions[0]), axis=0)
    if (
        not np.isfinite(limits).all()
        or not np.isfinite(positions).all()
        or np.any(limits[:, 0] > limits[:, 1])
    ):
        return {"policy_action_limits_available": False}
    finite = np.isfinite(values[0])
    outside = finite & (
        (values[0] < limits[:, 0]) | (values[0] > limits[:, 1])
    )
    joint_diagnostics: dict[str, dict[str, bool | float]] = {}
    for index, joint_name in enumerate(_FLYWHEEL_POLICY_ACTION_JOINT_NAMES):
        target_finite = bool(finite[index])
        if target_finite:
            violation = max(
                float(limits[index, 0] - values[0, index]),
                float(values[0, index] - limits[index, 1]),
                0.0,
            )
            target_to_live_delta = abs(float(values[0, index] - positions[index]))
        else:
            violation = 0.0
            target_to_live_delta = 0.0
        joint_diagnostics[joint_name] = {
            "target_finite": target_finite,
            "outside_live_joint_limit": bool(outside[index]),
            "limit_violation_rad": round(violation, 8),
            "target_to_live_joint_position_delta_rad": round(target_to_live_delta, 8),
        }
    return {
        "policy_action_limits_available": True,
        "policy_action_dimension": 12,
        "policy_action_nonfinite_count": int((~finite).sum()),
        "policy_action_outside_live_joint_limit_count": int(outside.sum()),
        "policy_action_joint_diagnostics": joint_diagnostics,
    }


def _require_flywheel_cloth_health(
    env: Any,
    *,
    policy_action_diagnostics: Mapping[str, object] | None = None,
) -> None:
    """Stop a recording before numerical cloth divergence can become data."""

    check = getattr(env, "flywheel_cloth_physical_health", None)
    if not callable(check):
        raise SimulatorNumericalDivergenceError(
            "simulator_numerical_divergence: cloth health readback unavailable"
        )
    health = check()
    if not isinstance(health, Mapping) or health.get("healthy") is not True:
        reason = (
            health.get("reason", "simulator_numerical_divergence")
            if isinstance(health, Mapping)
            else "simulator_numerical_divergence"
        )
        evidence: list[str] = []
        if isinstance(health, Mapping):
            exceeded_metrics = health.get("exceeded_metrics")
            if isinstance(exceeded_metrics, (list, tuple)):
                for metric in exceeded_metrics:
                    if not isinstance(metric, Mapping):
                        continue
                    metric_name = metric.get("metric_name")
                    if isinstance(metric_name, str):
                        evidence.append(
                            f"{metric_name}={metric.get('metric_value')} "
                            f"limit={metric.get('metric_limit')}"
                        )
            metric_name = health.get("metric_name")
            if isinstance(metric_name, str):
                evidence.append(
                    f"{metric_name}={health.get('metric_value')} "
                    f"limit={health.get('metric_limit')}"
                )
            offending_colliders = health.get("offending_colliders")
            if isinstance(offending_colliders, (list, tuple)):
                for collider in offending_colliders[:3]:
                    if not isinstance(collider, Mapping):
                        continue
                    usd_prim = collider.get("usd_prim")
                    prim_type = collider.get("prim_type")
                    approximation = collider.get("approximation")
                    if all(isinstance(value, str) for value in (usd_prim, prim_type, approximation)):
                        evidence.append(
                            f"usd_prim={usd_prim} prim_type={prim_type} "
                            f"approximation={approximation}"
                        )
        if isinstance(policy_action_diagnostics, Mapping):
            limits_available = policy_action_diagnostics.get(
                "policy_action_limits_available"
            )
            if type(limits_available) is bool:
                evidence.append(
                    f"policy_action_limits_available={limits_available}"
                )
            for field in (
                "policy_action_nonfinite_count",
                "policy_action_outside_live_joint_limit_count",
                "policy_action_steps_outside_live_joint_limits",
                "policy_action_max_outside_live_joint_limit_count",
                "policy_action_total_steps",
            ):
                value = policy_action_diagnostics.get(field)
                if type(value) is int and value >= 0:
                    evidence.append(f"{field}={value}")
            outside_steps = policy_action_diagnostics.get(
                "policy_action_outside_live_joint_limit_step_counts"
            )
            max_violation = policy_action_diagnostics.get(
                "policy_action_max_limit_violation_rad"
            )
            max_delta = policy_action_diagnostics.get(
                "policy_action_max_target_to_live_joint_position_delta_rad"
            )
            if all(isinstance(values, Mapping) for values in (outside_steps, max_violation, max_delta)):
                joint_evidence: list[str] = []
                for joint_name in _FLYWHEEL_POLICY_ACTION_JOINT_NAMES:
                    outside_step_count = outside_steps.get(joint_name)
                    violation = max_violation.get(joint_name)
                    delta = max_delta.get(joint_name)
                    if (
                        type(outside_step_count) is not int
                        or outside_step_count < 0
                        or type(violation) is not float
                        or not float("-inf") < violation < float("inf")
                        or type(delta) is not float
                        or not float("-inf") < delta < float("inf")
                    ):
                        joint_evidence = []
                        break
                    joint_evidence.append(
                        f"{joint_name}(outside_steps={outside_step_count},"
                        f"max_violation_rad={violation},"
                        f"max_target_to_live_joint_position_delta_rad={delta})"
                    )
                if joint_evidence:
                    evidence.append(
                        "policy_action_joint_summary=" + ",".join(joint_evidence)
                    )
        diagnostic_suffix = f" ({'; '.join(evidence)})" if evidence else ""
        raise SimulatorNumericalDivergenceError(
            f"{reason}: cloth physical-health admission failed{diagnostic_suffix}"
        )

_LOWERCASE_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LOWERCASE_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DEFAULT_POLICY_REPO = "ryanjin333/lehome-groot-n17-models"
_DEFAULT_POLICY_REVISION = "30ac1a84da67b099e115ad147bcd61e9d60046d3"
_DEFAULT_POLICY_STEP = 12000
_DEFAULT_POLICY_ARTIFACT_SHA256 = "3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06"


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




def _persistent_assignment_is_complete(assignment: Mapping[str, Any]) -> bool:
    garment = assignment.get("garment", assignment.get("garment_name"))
    seed = assignment.get("seed")
    category = assignment.get("category")
    attempt_id = assignment.get("attempt_id") or assignment.get("trial_id")
    return (
        isinstance(garment, str) and bool(garment)
        and isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0
        and isinstance(category, str) and bool(category)
        and isinstance(attempt_id, str) and bool(attempt_id)
    )


def _persistent_collection_strategy(assignment: Mapping[str, Any]) -> str:
    """Resolve the recorded strategy without enabling unstable material edits."""

    explicit = assignment.get("strategy")
    if explicit is None:
        return "mild_geometry" if assignment.get("difficulty") == "randomized" else "canonical"
    if explicit in {"mild", "strong"}:
        raise ValueError("persistent collection only supports geometry-only randomization")
    if explicit not in {"canonical", "mild_geometry", "strong_geometry"}:
        raise ValueError("persistent collection has an unsupported randomization strategy")
    return str(explicit)


def _persistent_policy_identity(args: argparse.Namespace) -> tuple[str, str, int, str]:
    """Validate the exact checkpoint identity served to this worker."""

    repository = getattr(args, "policy_repo", _DEFAULT_POLICY_REPO)
    revision = getattr(args, "policy_revision", _DEFAULT_POLICY_REVISION)
    step = getattr(args, "policy_step", _DEFAULT_POLICY_STEP)
    artifact_sha256 = getattr(
        args, "policy_artifact_sha256", _DEFAULT_POLICY_ARTIFACT_SHA256,
    )
    if not isinstance(repository, str) or not repository or any(character.isspace() for character in repository):
        raise ValueError("persistent policy repository is invalid")
    if not isinstance(revision, str) or _LOWERCASE_COMMIT.fullmatch(revision) is None:
        raise ValueError("persistent policy revision must be an immutable commit")
    if type(step) is not int or step <= 0:
        raise ValueError("persistent policy step must be a positive integer")
    if not isinstance(artifact_sha256, str) or _LOWERCASE_SHA256.fullmatch(artifact_sha256) is None:
        raise ValueError("persistent policy artifact SHA-256 is invalid")
    return repository, revision, step, artifact_sha256


def _verified_restore_assignment(
    assignment: Mapping[str, Any],
) -> tuple[object | None, dict[str, object] | None]:
    """Read a digest-bound replay reset once, before Isaac can restore it."""

    restore = assignment.get("restore_snapshot") or assignment.get("hard_state_snapshot")
    expected = assignment.get("restore_snapshot_sha256")
    replay_kind = assignment.get("replay_kind")
    verification_required = replay_kind == "verified_success_reset_v1" or expected is not None
    if not verification_required:
        return restore, None
    if not isinstance(restore, (str, Path)):
        raise ValueError("verified replay restore snapshot must be an absolute regular file")
    path = Path(restore)
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("verified replay restore snapshot must be an absolute regular file")
    try:
        if not stat.S_ISREG(path.stat().st_mode):
            raise ValueError("verified replay restore snapshot must be an absolute regular file")
        payload_bytes = path.read_bytes()
    except OSError as error:
        raise ValueError("verified replay restore snapshot must be an absolute regular file") from error
    if not isinstance(expected, str) or _LOWERCASE_SHA256.fullmatch(expected) is None:
        raise ValueError("verified replay restore snapshot requires a lowercase SHA-256")
    if hashlib.sha256(payload_bytes).hexdigest() != expected:
        raise ValueError("verified replay restore snapshot SHA-256 mismatch")
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("verified replay restore snapshot must contain JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("verified replay restore snapshot must contain a JSON object")
    lineage = {
        key: assignment.get(key)
        for key in (
            "parent_episode_id", "lineage_id", "replay_kind",
            "restore_snapshot_cloth_frame",
        )
        if assignment.get(key) is not None
    }
    if replay_kind == "verified_success_reset_v1":
        if (
            not isinstance(lineage.get("parent_episode_id"), str)
            or not lineage["parent_episode_id"]
            or not isinstance(lineage.get("lineage_id"), str)
            or lineage["lineage_id"] != lineage["parent_episode_id"]
        ):
            raise ValueError("verified success replay requires matching parent and lineage IDs")
        cloth_frame = lineage.get("restore_snapshot_cloth_frame")
        if cloth_frame not in {"usd_local_points_v1", "physx_cloth_view_world_v1"}:
            raise ValueError("verified success replay requires an explicit legacy cloth frame")
        schema_version = payload.get("schema_version")
        authority = payload.get("cloth_state_authority")
        if cloth_frame == "usd_local_points_v1":
            if schema_version != 1 or authority is not None:
                raise ValueError("verified success replay USD-local cloth frame is incompatible")
            payload = dict(payload)
            payload["schema_version"] = 3
            payload["cloth_state_authority"] = "usd_local_points_v1"
        elif not (
            (schema_version == 1 and authority is None)
            or (schema_version == 2 and authority == "physx_cloth_view_world_v1")
        ):
            raise ValueError("verified success replay PhysX cloth frame is incompatible")
    return payload, {
        "restore_snapshot": str(path),
        "restore_snapshot_sha256": expected,
        **lineage,
    }


def _write_persistent_flywheel_manifest(
    attempt_output_dir: Path,
    assignment: Mapping[str, Any],
    args: argparse.Namespace,
    *,
    verified_restore: Mapping[str, object] | None = None,
) -> Path:
    """Author one attempt-scoped autonomous recorder contract for persistent collection."""

    from lehome.flywheel.artifacts import atomic_write_json
    from lehome.flywheel.models import EpisodeIdentity

    garment = assignment.get("garment", assignment.get("garment_name"))
    seed = assignment.get("seed")
    category = assignment.get("category")
    release_stage = assignment.get("release_stage", "seen")
    attempt_id = assignment.get("attempt_id") or assignment.get("trial_id")
    if not isinstance(garment, str) or not garment:
        raise ValueError("persistent flywheel assignment requires a garment")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("persistent flywheel assignment requires a non-negative seed")
    if not isinstance(category, str) or not category:
        raise ValueError("persistent flywheel assignment requires a category")
    if not isinstance(attempt_id, str) or not attempt_id:
        raise ValueError("persistent flywheel assignment requires an attempt id")
    policy_repo, policy_revision, policy_step, policy_artifact_sha256 = _persistent_policy_identity(args)
    # The material path remains disabled on this Isaac build because its USD
    # displayColor readback is unstable. Geometry-only profiles retain strict
    # readback while varying the scene properties that transfer to unseen tops.
    strategy = _persistent_collection_strategy(assignment)
    identity = EpisodeIdentity(
        episode_id=attempt_id,
        policy_repo=policy_repo,
        policy_revision=policy_revision,
        policy_step=policy_step,
        code_revision="61e60d18dcda662b144d1cc0fb05fa2beec82033",
        asset_revision="bea65fd960ad5a1bb3bd3fa77164b28001c08ef9",
        simulator_version="5.1.0.0",
        garment_name=garment,
        category=category,
        release_stage=str(release_stage),
        seed=seed,
        instruction="fold the garment on the table",
        strategy=strategy,
    )
    simulator_device = str(getattr(args, "device", "")).lower()
    if simulator_device != "cpu" and re.fullmatch(r"cuda:[0-9]+", simulator_device) is None:
        raise ValueError("persistent flywheel assignment requires cpu or a canonical CUDA simulator device")
    path = attempt_output_dir / "flywheel-manifest.json"
    payload = {
        "schema_version": 1,
        "policy_revision": identity.policy_revision,
        "seed": identity.seed,
        "garment": identity.garment_name,
        "strategy": identity.strategy,
        "episode_id": identity.episode_id,
        "identity": {
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
        },
        "policy_artifact_sha256": policy_artifact_sha256,
        "image_identity": "sha256:afb35941768cabfe2f18173df27190b78a5b3044fbbbe71c3029539ffbc821d7",
        "execution_mode": "policy_server",
        "execution_backend": "policy_server",
        "simulator_device": simulator_device,
        "policy_device": str(getattr(args, "policy_device", "cuda:0")),
        "parity_stage": "server_cpu" if simulator_device == "cpu" else "persistent_collection",
    }
    if verified_restore is not None:
        payload.update(verified_restore)
    if assignment.get("recovery_kind") == "controlled_success_recovery_snapshot_v3":
        controlled_keys = {
            "recovery_kind", "category", "garment", "source_round_id", "source_episode_id", "source_episode_digest",
            "source_immutable_revision", "source_reset", "source_reset_sha256",
            "source_annotations", "source_annotations_sha256", "source_first_success_step",
            "source_continuation_snapshot", "source_continuation_snapshot_sha256",
            "source_continuation_snapshot_relative_path", "prefix_stop", "perturbation_profile",
            "perturbation_seed", "source_seed", "source_continuation_state", "source_state_fingerprint", "perturbation_fingerprint",
            "source_state_perturbation_fingerprint", "category_acceptance_cap",
            "controlled_smoke", "controlled_smoke_teacher_probe",
        }
        payload["controlled_recovery"] = {
            key: assignment[key] for key in controlled_keys if key in assignment
        }
    attempt_output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)
    return path



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
        require_deterministic_seed: bool = False,
    ) -> None:
        self.args = args
        self.env = env
        self.policy = policy
        self.env_cfg = env_cfg
        self.ee_solver = ee_solver
        self.is_bimanual = is_bimanual
        self._env_factory = env_factory or (lambda cfg: gym.make(args.task, cfg=cfg).unwrapped)
        self._require_deterministic_seed = require_deterministic_seed

    @property
    def runtime_receipt(self) -> dict[str, object]:
        """Report the actual cloth/render binding without guessing it."""

        simulation_device = str(getattr(self.env, "device", getattr(self.args, "device", ""))).lower()
        if simulation_device != "cpu" and re.fullmatch(r"cuda:[0-9]+", simulation_device) is None:
            raise ValueError("persistent evaluation requires a canonical CPU or CUDA cloth simulator")
        # Prefer already-known env/args devices at startup. Query Kit
        # /renderer/activeGpu only after the first reset, when other workers
        # are not still owning the GPU startup path.
        observed_devices = {
            "renderer_device": getattr(self.env, "renderer_device", None) or getattr(self.args, "renderer_device", None),
            "camera_device": getattr(self.env, "camera_device", None) or getattr(self.args, "camera_device", None),
        }
        if getattr(self, "_include_live_runtime_evidence", False):
            live = getattr(self.env, "flywheel_runtime_devices", None)
            live = live() if callable(live) else None
            if isinstance(live, Mapping):
                observed_devices = live
        renderer_device = str(observed_devices.get("renderer_device") or "")
        camera_device = str(observed_devices.get("camera_device") or "")
        if not renderer_device or not camera_device:
            raise ValueError("persistent evaluation cannot determine renderer/camera device")
        if re.fullmatch(r"cuda:[0-9]+", renderer_device) is None or camera_device != renderer_device:
            raise ValueError("persistent evaluation requires renderer and cameras on one CUDA device")
        if simulation_device != "cpu" and renderer_device != simulation_device:
            raise ValueError("persistent evaluation requires CUDA cloth and rendering on one device")
        backend = getattr(self.env, "_flywheel_cloth_backend", None)
        if not callable(backend):
            raise ValueError("persistent evaluation cannot observe the cloth backend")
        if backend() != "physx_cloth_view":
            raise ValueError("persistent evaluation requires the live PhysX cloth backend")
        receipt = {
            "simulation_device": simulation_device,
            "cloth_device": simulation_device,
            "cloth_backend": backend(),
            "renderer_device": renderer_device,
            "camera_device": camera_device,
            "policy_device": str(getattr(self.policy, "runtime_device", "")),
        }
        # Startup admission validates the initialized backend identity without
        # reading live cloth particles or camera RGB while other Isaac workers
        # may still own the GPU startup path. Fresh cloth/contact evidence is
        # collected only after the episode reset.
        if getattr(self, "_include_live_runtime_evidence", False):
            readback = getattr(self.env, "_flywheel_physics_cloth_state", None)
            if not callable(readback):
                raise ValueError("persistent evaluation cannot observe PhysX cloth readback")
            positions, velocities = readback()
            try:
                position_count, velocity_count = len(positions), len(velocities)
            except TypeError as error:
                raise ValueError("persistent evaluation PhysX cloth readback is unavailable") from error
            if position_count <= 0 or velocity_count <= 0 or position_count != velocity_count:
                raise ValueError("persistent evaluation PhysX cloth readback is invalid")
            receipt["cloth_readback"] = {"positions": position_count, "velocities": velocity_count}
        if getattr(self, "_include_contact_canary", False):
            canary = getattr(self.env, "flywheel_visible_garment_contact", None)
            if not callable(canary):
                raise ValueError("persistent evaluation cannot observe visible-contact canary")
            contact = canary()
            if not isinstance(contact, Mapping) or not isinstance(contact.get("observed"), bool):
                raise ValueError("persistent evaluation visible-contact canary is unavailable")
            receipt["visible_contact_canary"] = dict(contact)
        return receipt


    def restore_hard_state(self, payload: Mapping[str, Any] | str | Path) -> None:
        """Restore a failed-episode terminal snapshot after garment/policy reset."""

        from lehome.flywheel.snapshots import Snapshot, restore_snapshot

        if not isinstance(payload, Mapping):
            path = Path(payload)
            payload = json.loads(path.read_text(encoding="utf-8"))
        snapshot = Snapshot(
            schema_version=int(payload["schema_version"]),
            robot_position=tuple(payload["robot_position"]),
            robot_velocity=tuple(payload["robot_velocity"]),
            cloth_position=tuple(tuple(point) for point in payload["cloth_position"]),
            cloth_velocity=tuple(tuple(point) for point in payload["cloth_velocity"]),
            rng_state=dict(payload["rng_state"]),
            garment_name=str(payload["garment_name"]),
            randomization=dict(payload.get("randomization") or {}),
            scene_state=dict(payload.get("scene_state") or {}),
            cloth_state_authority=payload.get("cloth_state_authority"),
        )
        restore_snapshot(self.env, snapshot)

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
        deterministic_seed = self._require_deterministic_seed or not bool(
            getattr(self.args, "use_random_seed", False)
        )
        if deterministic_seed:
            for target in (self.env_cfg, getattr(self.env, "cfg", None)):
                if target is not None:
                    setattr(target, "seed", seed)
                    setattr(target, "random_seed", seed)
            set_seed = getattr(self.env, "set_seed", None)
            if not callable(set_seed):
                raise ValueError("persistent evaluation requires environment set_seed(seed)")
            set_seed(seed)
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
        restore, verified_restore = _verified_restore_assignment(assignment)
        if restore is not None:
            self._pending_restore_snapshot = restore
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
            if getattr(episode_args, "flywheel_manifest", None) is None:
                if _persistent_assignment_is_complete(assignment):
                    episode_args.flywheel_manifest = str(
                        _write_persistent_flywheel_manifest(
                            attempt_output_dir,
                            assignment,
                            episode_args,
                            verified_restore=verified_restore,
                        )
                    )
        if getattr(self, "_pending_restore_snapshot", None) is not None:
            episode_args.restore_snapshot = self._pending_restore_snapshot
            self._pending_restore_snapshot = None
        metrics = run_evaluation_loop(
            env=self.env, policy=self.policy, args=episode_args, ee_solver=self.ee_solver,
            is_bimanual=self.is_bimanual, garment_name=garment_name, reset_policy=reset_policy,
            cancellation_event=cancellation_event,
        )
        if attempt_output_dir is not None:
            # The persistent worker validates this property immediately after
            # the episode. Only then is it safe and meaningful to touch live
            # USD cloth arrays, camera ownership, and the contact canary.
            self._include_live_runtime_evidence = True
            self._include_contact_canary = True
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
        stabilize_garment_after_reset(env, args)
        restore = getattr(args, "restore_snapshot", None)
        if restore is not None:
            from lehome.flywheel.snapshots import Snapshot, restore_snapshot
            payload = restore if isinstance(restore, Mapping) else json.loads(Path(restore).read_text(encoding="utf-8"))
            snapshot = Snapshot(
                schema_version=int(payload["schema_version"]),
                robot_position=tuple(payload["robot_position"]),
                robot_velocity=tuple(payload["robot_velocity"]),
                cloth_position=tuple(tuple(point) for point in payload["cloth_position"]),
                cloth_velocity=tuple(tuple(point) for point in payload["cloth_velocity"]),
                rng_state=dict(payload["rng_state"]),
                garment_name=str(payload["garment_name"]),
                randomization=dict(payload.get("randomization") or {}),
                scene_state=dict(payload.get("scene_state") or {}),
                cloth_state_authority=payload.get("cloth_state_authority"),
            )
            restore_snapshot(env, snapshot)
            args.restore_snapshot = None

        recorder = None
        reset_snapshot = None
        randomization_receipt = {}
        controlled_provenance = None
        if flywheel_manifest is not None:
            from lehome.flywheel.isaac_recorder import AutonomousRecorder
            from lehome.flywheel.models import EpisodeIdentity
            from lehome.flywheel.randomization import sample_randomization
            from lehome.flywheel.snapshots import capture_snapshot

            strategy = flywheel_manifest.get("strategy", "canonical")
            sampled = sample_randomization(strategy, seed=flywheel_manifest.get("seed", args.seed))
            controlled = flywheel_manifest.get("controlled_recovery")
            if controlled is not None:
                if not isinstance(controlled, Mapping) or strategy != "canonical":
                    raise ValueError("controlled recovery requires a canonical immutable manifest")
                if restore is not None:
                    raise ValueError("controlled recovery cannot combine a second restore snapshot")
                # Validate the immutable source boundary before randomization
                # can mutate the simulator. Bootstrap reloads after the
                # readback to defend against input replacement in between.
                from lehome.flywheel.recovery_collection import bootstrap_controlled_recovery, load_controlled_recovery
                load_controlled_recovery(controlled)
            randomization_receipt = env.apply_flywheel_randomization(sampled)
            from lehome.flywheel.randomization import validate_randomization_receipt
            validate_randomization_receipt(dict(sampled.values), dict(randomization_receipt))
            if controlled is not None:
                controlled_provenance = bootstrap_controlled_recovery(env, controlled)
            identity = flywheel_identity
            if identity is None:  # pragma: no cover - guarded above, keeps type flow explicit.
                raise RuntimeError("missing flywheel identity")
            recorder = AutonomousRecorder(
                Path(flywheel_manifest["_path"]).parent,
                policy_revision=flywheel_manifest["policy_revision"],
                episode_id=flywheel_manifest.get("episode_id"),
                identity=identity,
                provenance={"policy_artifact_sha256": flywheel_manifest["policy_artifact_sha256"], "image_identity": flywheel_manifest["image_identity"], "execution_mode": flywheel_manifest["execution_mode"], "execution_backend": flywheel_manifest["execution_backend"], "simulator_device": flywheel_manifest["simulator_device"], "policy_device": flywheel_manifest.get("policy_device"), "parity_stage": flywheel_manifest.get("parity_stage"), "strategy_sampled": dict(sampled.values), "strategy_receipt": dict(randomization_receipt), **({"controlled_recovery": dict(controlled_provenance)} if controlled_provenance is not None else {})},
            )
            _require_flywheel_cloth_health(env)
            reset_snapshot = capture_snapshot(env, randomization={"strategy": strategy, "sampled": dict(sampled.values), "receipt": dict(randomization_receipt)})
            recorder.record_snapshot("reset", reset_snapshot)
        if reset_policy:
            # Controlled recovery must reset its temporal policy state only
            # after the authenticated source-prefix bootstrap is complete.
            policy.reset()

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
        policy_action_steps_outside_live_joint_limits = 0
        policy_action_max_outside_live_joint_limit_count = 0
        policy_action_total_steps = 0
        policy_action_outside_live_joint_limit_step_counts: dict[str, int] = {}
        policy_action_max_limit_violation_rad: dict[str, float] = {}
        policy_action_max_target_to_live_joint_position_delta_rad: dict[str, float] = {}

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
            policy_action_diagnostics = None
            if recorder is not None:
                policy_action_diagnostics = _flywheel_policy_action_limit_diagnostics(env, action)
                policy_action_total_steps += 1
                if policy_action_diagnostics.get("policy_action_limits_available") is True:
                    outside_count = policy_action_diagnostics.get(
                        "policy_action_outside_live_joint_limit_count"
                    )
                    if type(outside_count) is int:
                        if outside_count > 0:
                            policy_action_steps_outside_live_joint_limits += 1
                        policy_action_max_outside_live_joint_limit_count = max(
                            policy_action_max_outside_live_joint_limit_count,
                            outside_count,
                        )
                    joint_diagnostics = policy_action_diagnostics.get(
                        "policy_action_joint_diagnostics"
                    )
                    if isinstance(joint_diagnostics, Mapping):
                        for joint_name, joint_diagnostic in joint_diagnostics.items():
                            if not isinstance(joint_diagnostic, Mapping):
                                continue
                            target_finite = joint_diagnostic.get("target_finite")
                            outside_limit = joint_diagnostic.get(
                                "outside_live_joint_limit"
                            )
                            violation = joint_diagnostic.get("limit_violation_rad")
                            delta = joint_diagnostic.get(
                                "target_to_live_joint_position_delta_rad"
                            )
                            policy_action_outside_live_joint_limit_step_counts.setdefault(
                                joint_name, 0
                            )
                            policy_action_max_limit_violation_rad.setdefault(
                                joint_name, 0.0
                            )
                            policy_action_max_target_to_live_joint_position_delta_rad.setdefault(
                                joint_name, 0.0
                            )
                            if (
                                target_finite is not True
                                or type(outside_limit) is not bool
                                or type(violation) is not float
                                or type(delta) is not float
                            ):
                                continue
                            if outside_limit:
                                policy_action_outside_live_joint_limit_step_counts[
                                    joint_name
                                ] += 1
                            policy_action_max_limit_violation_rad[joint_name] = max(
                                policy_action_max_limit_violation_rad[joint_name], violation
                            )
                            policy_action_max_target_to_live_joint_position_delta_rad[
                                joint_name
                            ] = max(
                                policy_action_max_target_to_live_joint_position_delta_rad[
                                    joint_name
                                ],
                                delta,
                            )
                    policy_action_diagnostics = {
                        **policy_action_diagnostics,
                        "policy_action_steps_outside_live_joint_limits": (
                            policy_action_steps_outside_live_joint_limits
                        ),
                        "policy_action_max_outside_live_joint_limit_count": (
                            policy_action_max_outside_live_joint_limit_count
                        ),
                        "policy_action_total_steps": policy_action_total_steps,
                        "policy_action_outside_live_joint_limit_step_counts": (
                            policy_action_outside_live_joint_limit_step_counts
                        ),
                        "policy_action_max_limit_violation_rad": (
                            policy_action_max_limit_violation_rad
                        ),
                        "policy_action_max_target_to_live_joint_position_delta_rad": (
                            policy_action_max_target_to_live_joint_position_delta_rad
                        ),
                    }
                else:
                    policy_action_diagnostics = {
                        **policy_action_diagnostics,
                        "policy_action_total_steps": policy_action_total_steps,
                        "policy_action_outside_live_joint_limit_step_counts": (
                            policy_action_outside_live_joint_limit_step_counts
                        ),
                        "policy_action_max_limit_violation_rad": (
                            policy_action_max_limit_violation_rad
                        ),
                        "policy_action_max_target_to_live_joint_position_delta_rad": (
                            policy_action_max_target_to_live_joint_position_delta_rad
                        ),
                    }
            env.step(action)
            if recorder is not None:
                _require_flywheel_cloth_health(
                    env, policy_action_diagnostics=policy_action_diagnostics
                )
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
            # A successful autonomous source can later be admitted for a
            # controlled recovery only from an authenticated fresh H=16 policy
            # boundary.  Capture the complete physical state *before* the next
            # policy action, never after reconstructing it with a long prefix.
            if recorder is not None and not success_flag and recorder.step > 0 and recorder.step % 16 == 0:
                recorder.record_continuation_snapshot(
                    recorder.step,
                    capture_snapshot(
                        env,
                        randomization={
                            "strategy": strategy,
                            "sampled": dict(sampled.values),
                            "receipt": dict(randomization_receipt),
                            "continuation_step": recorder.step,
                        },
                    ),
                )

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
        require_deterministic_seed=flywheel_identity is not None,
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
