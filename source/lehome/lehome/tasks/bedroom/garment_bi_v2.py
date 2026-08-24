from __future__ import annotations
from typing import Any, Dict, List
from collections.abc import Sequence

import hashlib
import json
import os
import random
import numpy as np
import torch
from omegaconf import OmegaConf

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera
from pxr import UsdShade, Sdf, UsdGeom
import omni.kit.commands
from isaacsim.core.utils.prims import is_prim_path_valid
import isaacsim.core.utils.prims as prims_utils

from lehome.tasks.bedroom.garment_bi_cfg_v2 import GarmentEnvCfg
from lehome.utils.success_checker_chanllege import success_checker_garment_fold
from lehome.utils.depth_to_pointcloud import generate_pointcloud_from_data
from lehome.assets.scenes.bedroom import MARBLE_BEDROOM_CFG
from lehome.devices.action_process import preprocess_device_action
from lehome.assets.object.Garment import GarmentObject
from lehome.assets.collider_audit import audit_current_usd_stage
from lehome.flywheel.persistent_worker import SimulatorNumericalDivergenceError
from lehome.tasks.bedroom.challenge_garment_loader import ChallengeGarmentLoader
from lehome.flywheel.isaac_camera import read_camera_world_pose, write_camera_world_pose
import logging
from lehome.utils.logger import get_logger

# Create logger for this module with DEBUG level
logger = get_logger(__name__)


class GarmentEnv(DirectRLEnv):
    cfg: GarmentEnvCfg

    def __init__(self, cfg: GarmentEnvCfg, render_mode: str | None = None, **kwargs):
        self.cfg = cfg
        self.action_scale = self.cfg.action_scale
        self.object = None  # Will be created in _setup_scene
        self._flywheel_collider_health = None

        # Cache for distance-based reward (to handle step_interval decorator)
        self._last_computed_reward = 0.0

        self.garment_loader = ChallengeGarmentLoader(cfg.garment_cfg_base_path)
        self.garment_config = self.garment_loader.load_garment_config(
            cfg.garment_name, cfg.garment_version
        )
        self.particle_config = OmegaConf.load(cfg.particle_cfg_path)

        if cfg.use_random_seed:
            # Use random seed (no fixed seed)
            self.garment_rng = np.random.RandomState()
        else:
            # Use fixed seed from config
            self.garment_rng = np.random.RandomState(cfg.random_seed)

        cfg.viewer = cfg.viewer.replace(
            eye=(0, -1.2, 1.3),
            lookat=(0, 6.4, -2.8),
        )
        super().__init__(cfg, render_mode, **kwargs)
        self.left_joint_pos = self.left_arm.data.joint_pos
        self.right_joint_pos = self.right_arm.data.joint_pos

    def set_seed(self, seed: int) -> None:
        """Bind an episode seed before creating or resetting its garment."""

        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("garment seed must be a non-negative integer")
        self.cfg.use_random_seed = False
        self.cfg.seed = seed
        self.cfg.random_seed = seed
        self.garment_rng = np.random.RandomState(seed)
        if self.object is not None:
            self.object.rng = self.garment_rng

    def _setup_scene(self):
        self.left_arm = Articulation(self.cfg.left_robot)
        self.right_arm = Articulation(self.cfg.right_robot)
        self.top_camera = TiledCamera(self.cfg.top_camera)
        self.left_camera = TiledCamera(self.cfg.left_wrist)
        self.right_camera = TiledCamera(self.cfg.right_wrist)
        cfg = MARBLE_BEDROOM_CFG.spawn
        cfg.func(
            "/World/Scene",
            cfg,
            translation=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 0.0),
        )

        # Create garment object with selected asset
        self._create_garment_object()

        # add articulation to scene
        self.scene.articulations["left_arm"] = self.left_arm
        self.scene.articulations["right_arm"] = self.right_arm
        self.scene.sensors["top_camera"] = self.top_camera
        self.scene.sensors["left_camera"] = self.left_camera
        self.scene.sensors["right_camera"] = self.right_camera
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=1200, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _create_garment_object(self):
        """
        Create a new GarmentObject with the currently selected asset.
        """
        if self.object is not None:
            self._delete_garment_object()

        # Generate prim_path based on garment_name, default to "Cloth" if not specified
        garment_name = getattr(self.cfg, "garment_name", None)
        if garment_name and garment_name.strip():
            prim_name = garment_name.strip()
        else:
            prim_name = "Cloth"

        prim_path = f"/World/Object/{prim_name}"

        try:
            if is_prim_path_valid(prim_path):
                logger.debug(
                    f"[GarmentEnv] Prim path {prim_path} still exists, deleting before creation"
                )
                omni.kit.commands.execute("DeletePrims", paths=[prim_path])
                if hasattr(self, "sim") and self.sim is not None:
                    for _ in range(5):
                        self.sim.step(render=True)
                if is_prim_path_valid(prim_path):
                    logger.warning(
                        f"[GarmentEnv] WARNING: Prim path {prim_path} still exists after deletion attempt!"
                    )
                else:
                    logger.debug(
                        f"[GarmentEnv] Prim path {prim_path} successfully deleted"
                    )
        except Exception as e:
            logger.debug(
                f"[GarmentEnv] Could not delete existing prim (may not exist): {e}"
            )

        # Garment recreation changes the composed stage.  Never reuse collider
        # evidence captured for the previous garment in a persistent worker.
        self._flywheel_collider_health = None
        self._flywheel_legacy_cpu_reset_state = None

        # Create new garment object
        try:
            logger.debug(
                f"[GarmentEnv] Creating GarmentObject at prim_path: {prim_path}"
            )
            self.object = GarmentObject(
                prim_path=prim_path,
                particle_config=self.particle_config,
                garment_config=self.garment_config,
                rng=self.garment_rng,
            )
            logger.info("[GarmentEnv] GarmentObject created successfully")
        except Exception as e:
            logger.error(f"[GarmentEnv] Failed to create GarmentObject: {e}")
            raise RuntimeError(f"Failed to create GarmentObject: {e}") from e

        # Validate created object
        self._validate_created_object()

        self.texture_cfg = self.particle_config.objects.get("texture_randomization", {})
        self.light_cfg = self.particle_config.objects.get("light_randomization", {})
        self.flywheel_randomization_cfg = self.particle_config.objects.get("flywheel_randomization", {})
        logger.debug(
            f"[GarmentEnv] Loaded texture_cfg: {bool(self.texture_cfg)}, light_cfg: {bool(self.light_cfg)}"
        )

    def _validate_created_object(self):
        """
        Validate that the GarmentObject was created successfully and has required attributes.

        Raises:
            RuntimeError: If object validation fails
        """
        logger.debug("[GarmentEnv] Validating created GarmentObject...")

        if self.object is None:
            raise RuntimeError("GarmentObject creation returned None")

        required_attrs = [
            "usd_prim_path",
            "mesh_prim_path",
            "particle_system_path",
            "particle_material_path",
        ]

        for attr in required_attrs:
            if not hasattr(self.object, attr):
                raise RuntimeError(f"GarmentObject missing required attribute: {attr}")

            attr_value = getattr(self.object, attr)
            if attr_value is None:
                raise RuntimeError(f"GarmentObject attribute {attr} is None")

        prim_paths_to_check = [
            ("usd_prim_path", self.object.usd_prim_path),
            ("mesh_prim_path", self.object.mesh_prim_path),
        ]

        for path_name, path_value in prim_paths_to_check:
            if not is_prim_path_valid(path_value):
                logger.warning(
                    f"[GarmentEnv] Prim path {path_name} '{path_value}' is not valid in stage. "
                    "This may be expected if the prim hasn't been added yet."
                )
            else:
                logger.debug(
                    f"[GarmentEnv] Prim path {path_name} '{path_value}' is valid"
                )

        logger.debug("[GarmentEnv] GarmentObject validation passed")

    def _delete_garment_object(self):
        """Delete the current garment object from the stage.

        This method ensures complete cleanup of the garment object, including:
        - USD prim deletion
        - Particle system cleanup
        - All child prims removal
        """
        if self.object is None:
            return
        
        # bug: stuck while eval
        from isaacsim.core.api import World
        world = World.instance()
        # bug: stuck while eval
        was_playing = world.is_playing()
        if was_playing:
            world.pause()
            
        try:
            # Try to get prim_path from object first, then fallback to garment_name-based path
            if hasattr(self.object, "usd_prim_path") and self.object.usd_prim_path:
                prim_path = self.object.usd_prim_path
            else:
                # Fallback: generate prim_path based on garment_name, same logic as creation
                garment_name = getattr(self.cfg, "garment_name", None)
                if garment_name and garment_name.strip():
                    prim_name = garment_name.strip()
                else:
                    prim_name = "Cloth"
                prim_path = f"/World/Object/{prim_name}"

            if hasattr(self.object, "particle_system_path"):
                particle_path = self.object.particle_system_path
                try:
                    if is_prim_path_valid(particle_path):
                        omni.kit.commands.execute("DeletePrims", paths=[particle_path])
                        logger.debug(
                            f"[GarmentEnv] Deleted particle system at {particle_path}"
                        )
                except Exception as e:
                    logger.warning(
                        f"[GarmentEnv] Failed to delete particle system: {e}"
                    )

            if is_prim_path_valid(prim_path):
                omni.kit.commands.execute("DeletePrims", paths=[prim_path])
                logger.debug(f"[GarmentEnv] Deleted garment prim at {prim_path}")
            else:
                logger.warning(
                    f"[GarmentEnv] Prim path {prim_path} is not valid, skipping deletion"
                )

        except Exception as e:
            logger.warning(f"[GarmentEnv] Failed to delete garment object: {e}")
            import traceback

            traceback.print_exc()
        # bug: stuck while eval
        if was_playing:
            world.play()
        self.object = None

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        self.left_arm.set_joint_position_target(self.actions[:, :6])
        self.right_arm.set_joint_position_target(self.actions[:, 6:])

    def _get_observations(self) -> dict:
        action = self.actions.squeeze(0)
        left_joint_pos = torch.cat(
            [self.left_joint_pos[:, i].unsqueeze(1) for i in range(6)], dim=-1
        )
        right_joint_pos = torch.cat(
            [self.right_joint_pos[:, i].unsqueeze(1) for i in range(6)], dim=-1
        )
        joint_pos = torch.cat([left_joint_pos, right_joint_pos], dim=1)
        joint_pos = joint_pos.squeeze(0)
        top_camera_rgb = self.top_camera.data.output["rgb"]
        top_camera_depth = self.top_camera.data.output["depth"].squeeze()
        left_camera_rgb = self.left_camera.data.output["rgb"]
        right_camera_rgb = self.right_camera.data.output["rgb"]

        # Convert depth from meters to millimeters (uint16)
        # Range: 0-65535 mm (0-65.535 m), precision: 1 mm
        depth_np = top_camera_depth.cpu().detach().numpy().copy()
        depth_mm = np.clip(depth_np * 1000, 0, 65535).astype(np.uint16)

        observations = {
            "action": action.cpu().detach().numpy(),
            "observation.state": joint_pos.cpu().detach().numpy(),
            "observation.images.top_rgb": top_camera_rgb.cpu()
            .detach()
            .numpy()
            .squeeze(),
            "observation.images.left_rgb": left_camera_rgb.cpu()
            .detach()
            .numpy()
            .squeeze(),
            "observation.images.right_rgb": right_camera_rgb.cpu()
            .detach()
            .numpy()
            .squeeze(),
            "observation.top_depth": depth_mm,
        }
        return observations

    def _get_workspace_pointcloud(
        self, env_index: int = 0, num_points: int = 2048, use_fps: bool = False
    ):
        """
        Retrive workspace pointcloud from specified env_id (Robot Right Arm Base Frame)。

        Args:
            env_index (int)
            num_points (int)
            use_fps (bool)

        Returns:
            points (np.ndarray)
            colors (np.ndarray) (0-255)
        """
        top_camera_rgb_tensor = self.top_camera.data.output["rgb"]
        top_camera_depth_tensor = self.top_camera.data.output["depth"]

        depth_img = top_camera_depth_tensor[env_index].clone().cpu().numpy().squeeze()

        # Converting to float32 and to meter
        depth_img = depth_img.astype(np.float32) / 1000.0

        # RGB Tensor shape (Num_Envs, H, W, 4) -> (H, W, 3/4)
        rgb_img = top_camera_rgb_tensor[env_index].clone().cpu().numpy()

        pointclouds = generate_pointcloud_from_data(
            rgb_image=rgb_img,
            depth_image=depth_img,
            num_points=num_points,
            use_fps=use_fps,
        )
        print(f"[Info] Generated pointcloud shape: {pointclouds.shape}")

        return pointclouds

    def _get_rewards(self) -> torch.Tensor:
        """Calculate distance-based reward for garment folding task.

        Reward Components:
        1. Distance-based reward: Encourages getting closer to target distances
        2. Success bonus: Large reward when all conditions are met
        3. Progress reward: Partial credit for meeting some conditions

        Returns:
            torch.Tensor: Reward value (0.0 to 1.0+)
        """
        # ========== Original Simple Reward (Sparse) ==========
        # Uncomment below to use simple binary reward (0 or 1)
        # success = self._check_success()
        # if success:
        #     total_reward = 1
        # else:
        #     total_reward = 0
        # return total_reward
        # =====================================================

        # ========== Distance-Based Reward (Dense) ==========
        # Check if object is valid
        if self.object is None:
            return 0.0
        if not hasattr(self.object, "_cloth_prim_view"):
            return 0.0

        # Get detailed success check result
        garment_type = self.garment_loader.get_garment_type(self.cfg.garment_name)
        result = success_checker_garment_fold(self.object, garment_type)

        # Handle step_interval decorator returning False
        if not isinstance(result, dict):
            # Return cached reward from last computation (maintains reward continuity)
            return self._last_computed_reward

        # Extract details
        success = result.get("success", False)
        details = result.get("details", {})

        # If success, return maximum reward
        if success:
            self._last_computed_reward = 1.0
            return 1.0

        # Calculate distance-based reward
        total_reward = 0.0
        num_conditions = len(details)

        if num_conditions == 0:
            return 0.0

        # Calculate weighted reward based on condition type
        # Primary conditions (<=): folding-related, higher weight
        # Secondary conditions (>=): shape constraints, lower weight
        primary_rewards = []
        secondary_rewards = []

        for cond_key, cond_info in details.items():
            value = cond_info.get("value", 0.0)
            threshold = cond_info.get("threshold", 0.0)
            passed = cond_info.get("passed", False)

            description = cond_info.get("description", "")
            is_less_than = "<=" in description

            if passed:
                condition_reward = 1.0
            else:
                if is_less_than:
                    # Primary folding conditions: steep penalty when far from target
                    if threshold > 0:
                        excess_ratio = max(0.0, (value - threshold) / threshold)
                        # Steeper decay for primary conditions
                        condition_reward = np.exp(-3.0 * excess_ratio)
                    else:
                        condition_reward = 0.0
                else:
                    # Secondary shape constraints: gentler reward curve
                    if threshold > 0:
                        ratio = value / threshold
                        # Less aggressive growth for secondary conditions
                        condition_reward = max(0.0, 1.0 - np.exp(-1.5 * (1.0 - ratio)))
                    else:
                        condition_reward = 0.0

            if is_less_than:
                primary_rewards.append(condition_reward)
            else:
                secondary_rewards.append(condition_reward)

        # Weighted combination: primary conditions dominate
        # Only give significant reward when primary conditions are mostly satisfied
        num_primary = len(primary_rewards)
        num_secondary = len(secondary_rewards)

        if num_primary > 0:
            avg_primary = sum(primary_rewards) / num_primary
            # Use geometric mean to heavily penalize if any primary condition is bad
            min_primary = min(primary_rewards) if primary_rewards else 0.0
            # Combine average and minimum to ensure all primary conditions matter
            primary_score = (avg_primary**0.7) * (min_primary**0.3)
        else:
            primary_score = 1.0

        if num_secondary > 0:
            avg_secondary = sum(secondary_rewards) / num_secondary
            secondary_score = avg_secondary
        else:
            secondary_score = 1.0

        # Final reward: primary conditions weighted heavily (80%), secondary (20%)
        # Scale to [0, 0.9] to reserve 1.0 for success
        final_reward = (0.8 * primary_score + 0.2 * secondary_score) * 0.9

        # Cache the computed reward for non-check steps
        self._last_computed_reward = float(final_reward)

        return float(final_reward)
        # ===================================================

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _check_success(self) -> bool:
        """Check success based on garment type."""
        if self.object is None:
            return False

        if not hasattr(self.object, "_cloth_prim_view"):
            return False

        garment_type = self.garment_loader.get_garment_type(self.cfg.garment_name)
        result = success_checker_garment_fold(self.object, garment_type)

        if isinstance(result, dict):
            return result.get("success", False)
        else:
            return bool(result)

    def _get_success(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.object is None:
            success = False
            result = None
        elif not hasattr(self.object, "_cloth_prim_view"):
            success = False
            result = None
        else:
            garment_type = self.garment_loader.get_garment_type(self.cfg.garment_name)
            result = success_checker_garment_fold(self.object, garment_type)

            if isinstance(result, dict):
                logger.info(
                    f"[Success Check] Garment type: {result.get('garment_type', 'unknown')}, Thresholds: {result.get('thresholds', [])}"
                )

                details = result.get("details", {})
                for key, condition_info in details.items():
                    status = "✓" if condition_info.get("passed", False) else "✗"
                    logger.info(
                        f"  {condition_info.get('description', '')} -> {status}"
                    )

                success = result.get("success", False)
                logger.info(
                    f"[Success Check] Final result: {'Success ✓' if success else 'Failed ✗'}"
                )
            else:
                success = bool(result)

        if isinstance(success, bool):
            success_tensor = torch.tensor(
                [success] * len(self.episode_length_buf), device=self.device
            )
        else:
            success_tensor = torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
        episode_success = success_tensor
        return episode_success

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.left_arm._ALL_INDICES
        super()._reset_idx(env_ids)

        # Reset cached reward on episode reset
        self._last_computed_reward = 0.0

        left_joint_pos = self.left_arm.data.default_joint_pos[env_ids]
        right_joint_pos = self.right_arm.data.default_joint_pos[env_ids]
        self.left_arm.write_joint_position_to_sim(
            left_joint_pos, joint_ids=None, env_ids=env_ids
        )
        self.right_arm.write_joint_position_to_sim(
            right_joint_pos, joint_ids=None, env_ids=env_ids
        )

        # Reset the garment object.  CPU source bootstrap owns USD-local
        # particles and cannot create a particle-cloth view.
        if self.object is not None:
            if str(self.device).lower() == "cpu":
                self._flywheel_reset_legacy_cpu_garment()
            else:
                self.object.reset()

        # Apply randomization if enabled in config
        if self.texture_cfg.get("enable", False):
            self._randomize_table038_texture()

        if self.light_cfg.get("enable", False):
            self._randomize_light()

    def _randomize_table038_texture(self):
        """Randomize Table038 texture based on config."""
        if not self.texture_cfg.get("enable", False):
            return

        folder = self.texture_cfg.get("folder", "")
        if not os.path.isabs(folder):
            folder = os.path.join(os.getcwd(), folder)

        min_id = int(self.texture_cfg.get("min_id", 1))
        max_id = int(self.texture_cfg.get("max_id", 1))
        shader_path = self.texture_cfg.get("prim_path", "")

        if not folder or not os.path.exists(folder):
            print(f"[Reset][Warn] Texture folder not found: {folder}")
            return
        if not shader_path:
            print("[Reset][Warn] No prim_path provided for texture randomization")
            return

        stage = self.scene.stage
        shader_prim = stage.GetPrimAtPath(shader_path)
        if not shader_prim.IsValid():
            print(f"[Reset][Warn] Shader prim not found at {shader_path}")
            return

        shader = UsdShade.Shader(shader_prim)
        idx = random.randint(min_id, max_id)
        tex_path = os.path.join(folder, f"{idx}.png")

        tex_input = shader.GetInput("file") or shader.GetInput("diffuse_texture")
        if not tex_input:
            print("[Reset][Warn] No texture input found on shader")
            return

        tex_input.Set(Sdf.AssetPath(tex_path))
        # print(f"[Reset] Texture randomized -> {tex_path}")

    def _randomize_light(self):
        """Randomize DomeLight attributes based on config."""
        if not self.light_cfg.get("enable", False):
            return

        prim_path = self.light_cfg.get("prim_path", "/World/Light")
        intensity_range = self.light_cfg.get("intensity_range", [800, 2000])
        color_range = self.light_cfg.get("color_range", [0.0, 1.0])

        stage = self.scene.stage
        light_prim = stage.GetPrimAtPath(prim_path)
        if not light_prim.IsValid():
            print(f"[Reset][Warn] Light prim not found at {prim_path}")
            return

        intensity = random.uniform(*intensity_range)
        color = tuple[float, float, float](
            random.uniform(color_range[0], color_range[1]) for _ in range(3)
        )

        light_prim.GetAttribute("inputs:intensity").Set(intensity)
        light_prim.GetAttribute("inputs:color").Set(color)

        # print(f"[Reset] Light randomized -> intensity={intensity:.1f}, color={color}")

    def preprocess_device_action(
        self, action: dict[str, Any], teleop_device
    ) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)

    def initialize_obs(self):
        if str(self.device).lower() == "cpu":
            self._flywheel_initialize_legacy_cpu_garment()
        else:
            self.object.initialize()

    def _flywheel_initialize_legacy_cpu_garment(self) -> None:
        """Initialize the source-only CPU garment from live USD particles."""

        if self.object is None:
            raise RuntimeError("cannot initialize an absent CPU source garment")
        try:
            self.object.set_world_pose(position=self.object.init_pos, orientation=self.object.init_ori)
            positions, velocities = self._flywheel_legacy_cpu_cloth_state()
        except (RuntimeError, TypeError, ValueError, AttributeError) as error:
            raise RuntimeError("CPU source garment initialization lacks live USD cloth state") from error
        self._flywheel_legacy_cpu_reset_state = (positions.copy(), velocities.copy())

    def _flywheel_reset_legacy_cpu_garment(self) -> None:
        """Restore the CPU source reset state through USD, never PhysX."""

        initial = getattr(self, "_flywheel_legacy_cpu_reset_state", None)
        if not (
            isinstance(initial, tuple)
            and len(initial) == 2
            and all(isinstance(value, np.ndarray) for value in initial)
        ):
            raise RuntimeError("CPU source reset has no live USD initialization state")
        positions, velocities = initial
        try:
            positions, velocities = self._flywheel_cloth_arrays(positions, velocities)
            config_get = getattr(self.object, "_get_config_value", None)
            if not callable(config_get):
                raise RuntimeError("CPU source garment reset configuration is unavailable")
            position_range, _ = config_get("soft_reset_pos_range", "common")
            orientation_range, _ = config_get("soft_reset_rot_range", "common")
            position_range = np.asarray(position_range, dtype=np.float32)
            orientation_range = np.asarray(orientation_range, dtype=np.float32)
            if (
                position_range.shape != (6,)
                or orientation_range.shape != (6,)
                or not np.isfinite(position_range).all()
                or not np.isfinite(orientation_range).all()
            ):
                raise RuntimeError("CPU source garment reset configuration is invalid")
            position = self.garment_rng.uniform(position_range[:3], position_range[3:])
            orientation_degrees = self.garment_rng.uniform(orientation_range[:3], orientation_range[3:])
            from isaacsim.core.utils.rotations import euler_angles_to_quat

            self.object.set_world_pose(position, euler_angles_to_quat(orientation_degrees, degrees=True))
            self.object.reset_pose = np.concatenate((position, orientation_degrees)).astype(np.float32)
            positions_attr, velocities_attr = self._flywheel_legacy_cpu_cloth_attributes()
            if not callable(getattr(positions_attr, "Set", None)) or not callable(getattr(velocities_attr, "Set", None)):
                raise RuntimeError("CPU source garment USD attributes are not writable")
            if positions_attr.Set(self._flywheel_legacy_usd_vec3f_array(positions)) is False:
                raise RuntimeError("CPU source garment USD points write failed")
            if velocities_attr.Set(self._flywheel_legacy_usd_vec3f_array(velocities)) is False:
                raise RuntimeError("CPU source garment USD velocities write failed")
            observed_positions, observed_velocities = self._flywheel_legacy_cpu_cloth_state()
        except (RuntimeError, TypeError, ValueError, AttributeError) as error:
            raise RuntimeError("CPU source garment USD reset readback failed") from error
        if not (
            np.allclose(observed_positions, positions, rtol=0.0, atol=1e-6)
            and np.allclose(observed_velocities, velocities, rtol=0.0, atol=1e-6)
        ):
            raise RuntimeError("CPU source garment USD reset readback mismatch")

    def get_all_pose(self):
        return self.object.get_all_pose()

    def set_all_pose(self, pose):
        if str(self.device).lower() == "cpu":
            garment_pose = np.asarray(pose.get("Garment"), dtype=np.float32)
            if garment_pose.shape != (6,) or not np.isfinite(garment_pose).all():
                raise ValueError("CPU source garment pose must be a finite xyz-rpy vector")
            from isaacsim.core.utils.rotations import euler_angles_to_quat

            self.object.set_world_pose(
                garment_pose[:3], euler_angles_to_quat(garment_pose[3:], degrees=True)
            )
            self.object.reset_pose = garment_pose.copy()
            return
        self.object.set_all_pose(pose)

    def _flywheel_capture_scene_state(self) -> dict[str, object]:
        """Capture every USD/sensor property mutated by flywheel randomization."""
        if self.object is None:
            raise RuntimeError("cannot capture flywheel scene state without a garment")
        stage = self.scene.stage
        table_prim = stage.GetPrimAtPath(self.texture_cfg.get("prim_path", ""))
        table_shader = UsdShade.Shader(table_prim)
        table_input = table_shader.GetInput("file") or table_shader.GetInput("diffuse_texture")
        mesh = stage.GetPrimAtPath(self.object.mesh_prim_path)
        color_attr = mesh.GetAttribute("primvars:displayColor") if mesh.IsValid() else None
        light = stage.GetPrimAtPath(self.flywheel_randomization_cfg.get("light_prim_path", "/World/Light"))
        intensity_attr = light.GetAttribute("inputs:intensity") if light.IsValid() else None
        color_light_attr = light.GetAttribute("inputs:color") if light.IsValid() else None
        if not table_prim.IsValid() or not table_input or color_attr is None or not color_attr.IsValid():
            raise RuntimeError("flywheel scene state requires table shader and garment displayColor")
        if intensity_attr is None or not intensity_attr.IsValid() or color_light_attr is None or not color_light_attr.IsValid():
            raise RuntimeError("flywheel scene state requires readable light intensity and color")
        table_value = table_input.Get()
        table_path = str(getattr(table_value, "path", table_value))
        from lehome.flywheel.randomization import read_or_author_garment_display_color

        display_color = read_or_author_garment_display_color(color_attr)
        light_color = color_light_attr.Get()
        if intensity_attr.Get() is None or light_color is None:
            raise RuntimeError("flywheel scene state has unreadable USD attributes")

        def tensor_row(value) -> list[float]:
            array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
            return [float(item) for item in np.asarray(array)[0]]

        cameras = []
        for camera in (self.top_camera, self.left_camera, self.right_camera):
            positions, orientations = read_camera_world_pose(camera)
            cameras.append({"position": tensor_row(positions), "orientation": tensor_row(orientations)})
        roots = []
        for arm in (self.left_arm, self.right_arm):
            if not hasattr(arm, "write_root_pose_to_sim") or not hasattr(arm.data, "root_pos_w") or not hasattr(arm.data, "root_quat_w"):
                raise RuntimeError("Isaac robot base does not expose restorable world pose")
            roots.append({"position": tensor_row(arm.data.root_pos_w), "orientation": tensor_row(arm.data.root_quat_w)})
        return {
            "camera_world_poses": cameras,
            "robot_root_poses": roots,
            "light_intensity": float(intensity_attr.Get()),
            "light_color": [float(value) for value in light_color],
            "table_texture_path": table_path,
            "table_shader_input": table_input.GetBaseName(),
            "garment_display_color": display_color,
            "garment_reset_pose": [float(value) for value in self.object.get_all_pose()["Garment"]],
        }

    def _flywheel_restore_scene_state(self, scene_state: dict[str, object]) -> None:
        """Restore the exact scene payload captured by :meth:`_flywheel_capture_scene_state`."""
        required = {"camera_world_poses", "robot_root_poses", "light_intensity", "light_color", "table_texture_path", "table_shader_input", "garment_display_color", "garment_reset_pose"}
        if set(scene_state) != required:
            raise ValueError("flywheel scene snapshot does not cover every randomized property")
        cameras = scene_state["camera_world_poses"]
        roots = scene_state["robot_root_poses"]
        if not isinstance(cameras, list) or len(cameras) != 3 or not isinstance(roots, list) or len(roots) != 2:
            raise ValueError("flywheel scene snapshot has invalid camera or robot root coverage")
        for camera, pose in zip((self.top_camera, self.left_camera, self.right_camera), cameras, strict=True):
            if not isinstance(pose, dict):
                raise ValueError("flywheel camera snapshot pose is invalid")
            position = torch.tensor(pose["position"], dtype=torch.float32, device=self.device).unsqueeze(0)
            orientation = torch.tensor(pose["orientation"], dtype=torch.float32, device=self.device).unsqueeze(0)
            write_camera_world_pose(camera, position, orientation)
        for arm, pose in zip((self.left_arm, self.right_arm), roots, strict=True):
            if not isinstance(pose, dict):
                raise ValueError("flywheel robot root snapshot pose is invalid")
            position = torch.tensor(pose["position"], dtype=torch.float32, device=self.device).unsqueeze(0)
            orientation = torch.tensor(pose["orientation"], dtype=torch.float32, device=self.device).unsqueeze(0)
            arm.write_root_pose_to_sim(torch.cat((position, orientation), dim=-1))
        stage = self.scene.stage
        light = stage.GetPrimAtPath(self.flywheel_randomization_cfg.get("light_prim_path", "/World/Light"))
        table_prim = stage.GetPrimAtPath(self.texture_cfg.get("prim_path", ""))
        table_input = UsdShade.Shader(table_prim).GetInput("file") or UsdShade.Shader(table_prim).GetInput("diffuse_texture")
        mesh = stage.GetPrimAtPath(self.object.mesh_prim_path)
        intensity_attr = light.GetAttribute("inputs:intensity") if light.IsValid() else None
        color_light_attr = light.GetAttribute("inputs:color") if light.IsValid() else None
        color_attr = mesh.GetAttribute("primvars:displayColor") if mesh.IsValid() else None
        if not table_input or intensity_attr is None or color_light_attr is None or color_attr is None:
            raise RuntimeError("flywheel scene snapshot restore cannot access USD attributes")
        table_input.Set(Sdf.AssetPath(str(scene_state["table_texture_path"])))
        intensity_attr.Set(float(scene_state["light_intensity"]))
        color_light_attr.Set(tuple(float(value) for value in scene_state["light_color"]))
        color_attr.Set([tuple(float(value) for value in color) for color in scene_state["garment_display_color"]])
        self.set_all_pose({"Garment": np.asarray(scene_state["garment_reset_pose"], dtype=np.float32)})
        observed = self._flywheel_capture_scene_state()
        if not self._flywheel_scene_state_matches(observed, scene_state):
            raise RuntimeError("flywheel scene snapshot readback mismatch")

    @staticmethod
    def _flywheel_scene_state_matches(observed: dict[str, object], expected: dict[str, object]) -> bool:
        """Compare USD identifiers exactly and float32 Isaac values with a tolerance."""
        if set(observed) != set(expected):
            return False
        if observed["table_texture_path"] != expected["table_texture_path"] or observed["table_shader_input"] != expected["table_shader_input"]:
            return False
        try:
            if not np.isclose(float(observed["light_intensity"]), float(expected["light_intensity"]), atol=1e-5):
                return False
            for field in ("light_color", "garment_display_color", "garment_reset_pose"):
                if not np.allclose(observed[field], expected[field], atol=1e-5):
                    return False
            for field in ("camera_world_poses", "robot_root_poses"):
                actual_poses, expected_poses = observed[field], expected[field]
                if not isinstance(actual_poses, list) or not isinstance(expected_poses, list) or len(actual_poses) != len(expected_poses):
                    return False
                for actual, target in zip(actual_poses, expected_poses, strict=True):
                    if not isinstance(actual, dict) or not isinstance(target, dict):
                        return False
                    if not np.allclose(actual["position"], target["position"], atol=1e-5) or not np.allclose(actual["orientation"], target["orientation"], atol=1e-5):
                        return False
        except (KeyError, TypeError, ValueError):
            return False
        return True

    @staticmethod
    def _flywheel_cloth_arrays(positions, velocities) -> tuple[np.ndarray, np.ndarray]:
        """Validate live PhysX cloth-view particle state."""
        try:
            local_positions = np.asarray(positions, dtype=np.float32)
            local_velocities = np.asarray(velocities, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise RuntimeError("garment PhysX cloth readback is not numeric") from error
        if (
            local_positions.ndim != 2
            or local_velocities.ndim != 2
            or local_positions.shape[1:] != (3,)
            or local_velocities.shape[1:] != (3,)
            or local_positions.shape[0] != local_velocities.shape[0]
        ):
            raise RuntimeError(
                "garment PhysX cloth readback shape mismatch: "
                f"positions_shape={local_positions.shape} "
                f"velocities_shape={local_velocities.shape} expected_shape=Nx3"
            )
        positions_nonfinite_count = int(np.size(local_positions) - np.isfinite(local_positions).sum())
        velocities_nonfinite_count = int(np.size(local_velocities) - np.isfinite(local_velocities).sum())
        if positions_nonfinite_count or velocities_nonfinite_count:
            raise RuntimeError(
                "garment PhysX cloth readback is nonfinite: "
                f"positions_nonfinite_count={positions_nonfinite_count} "
                f"velocities_nonfinite_count={velocities_nonfinite_count}"
            )
        return local_positions.copy(), local_velocities.copy()

    @staticmethod
    def _flywheel_legacy_local_to_world(
        positions, velocities, root_position, root_rotation, root_scale
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert authenticated legacy USD-local cloth state for PhysX restore."""

        local_positions = np.asarray(positions, dtype=np.float32)
        local_velocities = np.asarray(velocities, dtype=np.float32)
        translation = np.asarray(root_position, dtype=np.float32).reshape(-1)
        rotation = np.asarray(root_rotation, dtype=np.float32)
        scale_value = root_scale
        detach = getattr(scale_value, "detach", None)
        if callable(detach):
            scale_value = detach()
            to_cpu = getattr(scale_value, "cpu", None)
            if callable(to_cpu):
                scale_value = to_cpu()
            to_numpy = getattr(scale_value, "numpy", None)
            if callable(to_numpy):
                scale_value = to_numpy()
        scale = np.asarray(scale_value, dtype=np.float32).reshape(-1)
        if (
            local_positions.ndim != 2
            or local_velocities.shape != local_positions.shape
            or local_positions.shape[1:] != (3,)
            or translation.shape != (3,)
            or rotation.shape != (3, 3)
            or scale.shape != (3,)
            or not np.isfinite(local_positions).all()
            or not np.isfinite(local_velocities).all()
            or not np.isfinite(translation).all()
            or not np.isfinite(rotation).all()
            or not np.isfinite(scale).all()
            or np.any(scale <= 0.0)
        ):
            raise ValueError("legacy USD-local cloth transform is invalid")
        world_positions = (local_positions * scale) @ rotation.T + translation
        world_velocities = (local_velocities * scale) @ rotation.T
        return world_positions.astype(np.float32, copy=False), world_velocities.astype(
            np.float32, copy=False
        )

    @staticmethod
    def _flywheel_project_legacy_usd_to_physx(
        positions, velocities, asset_positions, live_rest_positions,
        welded_vertices_remap_to_orig, welded_vertices_remap_to_weld,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project legacy USD state into the cooked PhysX particle order.

        PhysX's cooked weld maps are authoritative when present.  For meshes
        without cooked maps, identity is admitted only by an ordered rest-point
        comparison before this projection runs.
        """

        source_positions = np.asarray(positions, dtype=np.float32)
        source_velocities = np.asarray(velocities, dtype=np.float32)
        asset = np.asarray(asset_positions, dtype=np.float32)
        if (
            source_positions.ndim != 2
            or source_positions.shape[1:] != (3,)
            or source_velocities.shape != source_positions.shape
            or asset.shape != source_positions.shape
            or not np.isfinite(source_positions).all()
            or not np.isfinite(source_velocities).all()
            or not np.isfinite(asset).all()
        ):
            raise SimulatorNumericalDivergenceError(
                "legacy USD to PhysX cloth topology is invalid"
            )
        if len(source_positions) == 0:
            raise SimulatorNumericalDivergenceError(
                "legacy USD to PhysX cloth topology is invalid: usd_row_count=0"
            )

        def integer_vector(value, attribute_name: str) -> np.ndarray:
            try:
                vector = np.asarray(value)
            except (TypeError, ValueError) as error:
                raise SimulatorNumericalDivergenceError(
                    f"{attribute_name} must be a one-dimensional integer array"
                ) from error
            if vector.ndim != 1 or vector.dtype.kind not in "iu":
                raise SimulatorNumericalDivergenceError(
                    f"{attribute_name} must be a one-dimensional integer array"
                )
            return vector.astype(np.int64, copy=False)

        remap_to_orig = integer_vector(
            welded_vertices_remap_to_orig, "weldedVerticesRemapToOrig"
        )
        remap_to_weld = integer_vector(
            welded_vertices_remap_to_weld, "weldedVerticesRemapToWeld"
        )
        live_physx_count = len(remap_to_orig)
        if len(remap_to_weld) != len(source_positions):
            raise SimulatorNumericalDivergenceError(
                "legacy USD and live PhysX cloth topology weldedVerticesRemapToWeld "
                "size mismatch: "
                f"usd_row_count={len(source_positions)} remap_to_weld_count={len(remap_to_weld)}"
            )
        try:
            live_rest = np.asarray(live_rest_positions, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise SimulatorNumericalDivergenceError(
                "legacy USD to PhysX cloth topology live particle positions are invalid"
            ) from error
        if live_rest.ndim != 2 or live_rest.shape[1:] != (3,) or not np.isfinite(live_rest).all():
            raise SimulatorNumericalDivergenceError(
                "legacy USD to PhysX cloth topology live particle positions are invalid"
            )
        # Live coordinates only validate the current particle-array shape.  They
        # are never used as identity evidence or a proximity admission gate:
        # PhysX may integrate cloth before this restore path runs.
        if live_physx_count == 0:
            raise SimulatorNumericalDivergenceError(
                "legacy USD to PhysX cloth topology has no live PhysX particles"
            )
        if live_physx_count != len(live_rest):
            raise SimulatorNumericalDivergenceError(
                "legacy USD and live PhysX cloth topology weldedVerticesRemapToOrig "
                "size mismatch: "
                f"live_physx_count={len(live_rest)} "
                f"remap_to_orig_count={live_physx_count}"
            )

        invalid_orig = np.flatnonzero(
            (remap_to_orig < 0) | (remap_to_orig >= len(source_positions))
        )
        if invalid_orig.size:
            index = int(invalid_orig[0])
            raise SimulatorNumericalDivergenceError(
                "legacy USD to PhysX cloth topology weldedVerticesRemapToOrig "
                f"contains out-of-range index: index={index} value={int(remap_to_orig[index])} "
                f"usd_row_count={len(source_positions)}"
            )
        if np.unique(remap_to_orig).size != live_physx_count:
            raise SimulatorNumericalDivergenceError(
                "legacy USD to PhysX cloth topology weldedVerticesRemapToOrig must be unique: "
                f"live_physx_count={live_physx_count} "
                f"unique_original_count={np.unique(remap_to_orig).size}"
            )
        invalid_weld = np.flatnonzero(
            (remap_to_weld < 0) | (remap_to_weld >= live_physx_count)
        )
        if invalid_weld.size:
            index = int(invalid_weld[0])
            raise SimulatorNumericalDivergenceError(
                "legacy USD to PhysX cloth topology weldedVerticesRemapToWeld "
                f"contains out-of-range index: index={index} value={int(remap_to_weld[index])} "
                f"live_physx_count={live_physx_count}"
            )
        mapped_weld_count = int(np.unique(remap_to_weld).size)
        if mapped_weld_count != live_physx_count:
            raise SimulatorNumericalDivergenceError(
                "legacy USD to PhysX cloth topology weldedVerticesRemapToWeld "
                "does not cover every live PhysX particle: "
                f"mapped_welded_count={mapped_weld_count} "
                f"live_physx_count={live_physx_count}"
            )

        representative_welds = remap_to_weld[remap_to_orig]
        mismatch = np.flatnonzero(
            representative_welds != np.arange(live_physx_count, dtype=np.int64)
        )
        if mismatch.size:
            welded_index = int(mismatch[0])
            original_index = int(remap_to_orig[welded_index])
            raise SimulatorNumericalDivergenceError(
                "legacy USD to PhysX cloth topology weld map inverse representative mismatch: "
                f"welded_index={welded_index} original_index={original_index} "
                f"mapped_welded_index={int(remap_to_weld[original_index])}"
            )

        representative_for_authored = remap_to_orig[remap_to_weld]
        normalized_asset = asset.copy()
        normalized_asset[normalized_asset == 0.0] = 0.0
        asset_max_abs_delta = np.max(
            np.abs(normalized_asset - normalized_asset[representative_for_authored]), axis=1
        )
        geometry_mismatch = np.flatnonzero(asset_max_abs_delta != 0.0)
        if geometry_mismatch.size:
            original_index = int(geometry_mismatch[0])
            representative = int(representative_for_authored[original_index])
            raise SimulatorNumericalDivergenceError(
                "legacy USD to PhysX cloth topology weld map groups distinct authored vertices: "
                f"original_index={original_index} representative_index={representative} "
                f"asset_max_abs_delta={float(asset_max_abs_delta[original_index]):.9g}"
            )

        position_max_abs_delta = np.max(
            np.abs(source_positions - source_positions[representative_for_authored]), axis=1
        )
        velocity_max_abs_delta = np.max(
            np.abs(source_velocities - source_velocities[representative_for_authored]), axis=1
        )
        state_mismatch = np.flatnonzero(
            (position_max_abs_delta != 0.0) | (velocity_max_abs_delta != 0.0)
        )
        if state_mismatch.size:
            duplicate_index = int(state_mismatch[0])
            representative = int(representative_for_authored[duplicate_index])
            raise SimulatorNumericalDivergenceError(
                "legacy USD duplicate seam state is inconsistent: "
                f"representative_index={representative} duplicate_index={duplicate_index} "
                f"position_max_abs_delta={float(position_max_abs_delta[duplicate_index]):.9g} "
                f"velocity_max_abs_delta={float(velocity_max_abs_delta[duplicate_index]):.9g}"
            )

        return (
            source_positions[remap_to_orig].copy(),
            source_velocities[remap_to_orig].copy(),
        )

    @staticmethod
    def _flywheel_rebase_world_cloth(
        positions, velocities, source_position, source_rotation,
        target_position, target_rotation,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rigidly move authenticated world cloth with its randomized root pose."""

        world_positions = np.asarray(positions, dtype=np.float32)
        world_velocities = np.asarray(velocities, dtype=np.float32)
        source_translation = np.asarray(source_position, dtype=np.float32).reshape(-1)
        source_matrix = np.asarray(source_rotation, dtype=np.float32)
        target_translation = np.asarray(target_position, dtype=np.float32).reshape(-1)
        target_matrix = np.asarray(target_rotation, dtype=np.float32)
        if (
            world_positions.ndim != 2
            or world_velocities.shape != world_positions.shape
            or world_positions.shape[1:] != (3,)
            or source_translation.shape != (3,)
            or target_translation.shape != (3,)
            or source_matrix.shape != (3, 3)
            or target_matrix.shape != (3, 3)
            or not np.isfinite(world_positions).all()
            or not np.isfinite(world_velocities).all()
            or not np.isfinite(source_translation).all()
            or not np.isfinite(target_translation).all()
            or not np.isfinite(source_matrix).all()
            or not np.isfinite(target_matrix).all()
        ):
            raise ValueError("authenticated world cloth rebase is invalid")
        local_positions = (world_positions - source_translation) @ source_matrix
        local_velocities = world_velocities @ source_matrix
        rebased_positions = local_positions @ target_matrix.T + target_translation
        rebased_velocities = local_velocities @ target_matrix.T
        return rebased_positions.astype(np.float32, copy=False), rebased_velocities.astype(
            np.float32, copy=False
        )

    def _flywheel_legacy_cpu_cloth_attributes(self):
        """Expose USD particles only for restoring pre-PhysX legacy snapshots."""

        prim = getattr(self.object, "_prim", None)
        get_attribute = getattr(prim, "GetAttribute", None)
        if not callable(get_attribute):
            raise RuntimeError("legacy CPU garment does not expose a USD prim for cloth state")
        positions_attr = get_attribute("points")
        velocities_attr = get_attribute("velocities")
        if positions_attr is None or velocities_attr is None:
            raise RuntimeError("legacy CPU garment USD prim is missing points or velocities")
        if (not callable(getattr(positions_attr, "Get", None))
                or not callable(getattr(velocities_attr, "Get", None))):
            raise RuntimeError("legacy CPU garment USD points or velocities are unreadable")
        return positions_attr, velocities_attr

    def _flywheel_physx_weld_maps(self, *, asset_positions=None, live_positions=None):
        """Read cooked topology maps, or prove that the live topology is identity."""

        prim = getattr(self.object, "_prim", None)
        get_attribute = getattr(prim, "GetAttribute", None)
        if not callable(get_attribute):
            raise SimulatorNumericalDivergenceError(
                "live PhysX cloth prim does not expose cooked weld maps"
            )

        def read_optional_map(attribute_name: str):
            attribute = get_attribute(attribute_name)
            is_valid = getattr(attribute, "IsValid", None)
            if attribute is None or (callable(is_valid) and not is_valid()):
                return None
            get_value = getattr(attribute, "Get", None)
            if not callable(get_value):
                raise SimulatorNumericalDivergenceError(
                    f"live PhysX cloth prim cannot read {attribute_name}"
                )
            value = get_value()
            if value is None:
                return None
            try:
                return value if len(value) else None
            except TypeError as error:
                raise SimulatorNumericalDivergenceError(
                    f"live PhysX cloth prim has invalid {attribute_name}"
                ) from error

        map_names = (
            "physxParticle:weldedVerticesRemapToOrig",
            "physxParticle:weldedVerticesRemapToWeld",
        )
        remap_to_orig, remap_to_weld = (
            read_optional_map(map_names[0]), read_optional_map(map_names[1])
        )
        if remap_to_orig is not None and remap_to_weld is not None:
            return remap_to_orig, remap_to_weld
        if (remap_to_orig is None) != (remap_to_weld is None):
            missing = map_names[0] if remap_to_orig is None else map_names[1]
            raise SimulatorNumericalDivergenceError(
                f"live PhysX cloth prim is missing {missing}"
            )

        # PhysX omits cooked weld arrays for meshes whose authored point order
        # is already the simulation point order.  NVIDIA's installed schema
        # converter welds by exact-coordinate deduplication while preserving
        # the first authored occurrence.  Therefore unique authored vertices,
        # no welded triangles, and equal authored/live cardinality prove the
        # identity topology.  Do not compare coordinates: PhysX cooking may
        # relax rest geometry without changing particle order.
        welded_triangles = read_optional_map("physxParticle:weldedTriangleIndices")
        if welded_triangles is not None:
            raise SimulatorNumericalDivergenceError(
                "live PhysX cloth prim has welded triangles without cooked vertex maps"
            )
        try:
            asset = np.asarray(asset_positions, dtype=np.float32)
            live = np.asarray(live_positions, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise SimulatorNumericalDivergenceError(
                "live PhysX cloth prim is missing cooked weld maps"
            ) from error
        if (
            asset.ndim != 2
            or asset.shape[1:] != (3,)
            or live.ndim != 2
            or live.shape[1:] != (3,)
            or len(asset) == 0
            or len(live) != len(asset)
            or not np.isfinite(asset).all()
            or not np.isfinite(live).all()
        ):
            raise SimulatorNumericalDivergenceError(
                "live PhysX cloth prim is missing cooked weld maps and identity cardinality is unproven"
            )
        normalized_asset = asset.copy()
        normalized_asset[normalized_asset == 0.0] = 0.0
        if len(np.unique(normalized_asset, axis=0)) != len(normalized_asset):
            raise SimulatorNumericalDivergenceError(
                "live PhysX cloth prim is missing cooked weld maps for non-unique authored vertices"
            )
        identity = np.arange(len(asset), dtype=np.int64)
        return identity, identity.copy()

    def _flywheel_legacy_cpu_cloth_state(self) -> tuple[np.ndarray, np.ndarray]:
        positions_attr, velocities_attr = self._flywheel_legacy_cpu_cloth_attributes()
        positions, velocities = positions_attr.Get(), velocities_attr.Get()
        if positions is None or velocities is None:
            raise RuntimeError("legacy CPU garment USD points or velocities are unset")
        return self._flywheel_cloth_arrays(positions, velocities)

    @staticmethod
    def _flywheel_legacy_usd_vec3f_array(values: np.ndarray):
        """Convert legacy CPU cloth arrays to USD's native vector representation."""

        try:
            from pxr import Gf, Vt
        except ImportError:
            return values.tolist()
        return Vt.Vec3fArray([Gf.Vec3f(*map(float, row)) for row in values])

    def _flywheel_physics_cloth_view(self):
        ensure = getattr(self.object, "_ensure_physics_cloth_view", None)
        if not callable(ensure):
            raise RuntimeError("garment PhysX cloth view is not initialized")
        return ensure()

    def _flywheel_physics_cloth_state(self) -> tuple[np.ndarray, np.ndarray]:
        cloth = self._flywheel_physics_cloth_view()

        def _numpy(value):
            return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

        try:
            positions = _numpy(cloth.get_world_positions())
            velocities = _numpy(cloth.get_velocities())
        except (RuntimeError, TypeError, ValueError, AttributeError) as error:
            raise RuntimeError("garment PhysX cloth readback API failure") from error
        if positions.ndim == 3 and positions.shape[0] == 1:
            positions = positions[0]
        if velocities.ndim == 3 and velocities.shape[0] == 1:
            velocities = velocities[0]
        positions, velocities = self._flywheel_cloth_arrays(positions, velocities)

        initial_positions = getattr(self.object, "initial_points_positions", None)
        try:
            initial_positions = _numpy(initial_positions)
        except (RuntimeError, TypeError, ValueError, AttributeError) as error:
            raise RuntimeError("garment PhysX cloth topology readback API failure") from error
        if initial_positions.ndim == 3 and initial_positions.shape[0] == 1:
            initial_positions = initial_positions[0]
        if initial_positions.ndim != 2 or initial_positions.shape[1:] != (3,):
            raise RuntimeError(
                "garment PhysX cloth topology reference invalid: "
                f"initial_positions_shape={initial_positions.shape} expected_shape=Nx3"
            )
        if positions.shape[0] != initial_positions.shape[0]:
            raise RuntimeError(
                "garment PhysX cloth topology mismatch: "
                f"live_particle_count={positions.shape[0]} "
                f"initial_particle_count={initial_positions.shape[0]}"
            )

        try:
            root_position, root_orientation = self.object.get_world_pose()
            root_position = _numpy(root_position)
            root_orientation = _numpy(root_orientation)
        except (RuntimeError, TypeError, ValueError, AttributeError) as error:
            raise RuntimeError("garment root transform readback API failure") from error
        if root_position.ndim == 2 and root_position.shape[0] == 1:
            root_position = root_position[0]
        if root_orientation.ndim == 2 and root_orientation.shape[0] == 1:
            root_orientation = root_orientation[0]
        if root_position.shape != (3,) or root_orientation.shape != (4,):
            raise RuntimeError(
                "garment root transform shape mismatch: "
                f"root_position_shape={root_position.shape} "
                f"root_orientation_shape={root_orientation.shape} "
                "expected_position_shape=(3,) expected_orientation_shape=(4,)"
            )
        root_position_nonfinite_count = int(np.size(root_position) - np.isfinite(root_position).sum())
        root_orientation_nonfinite_count = int(np.size(root_orientation) - np.isfinite(root_orientation).sum())
        if root_position_nonfinite_count or root_orientation_nonfinite_count:
            raise RuntimeError(
                "garment root transform is nonfinite: "
                f"root_position_nonfinite_count={root_position_nonfinite_count} "
                f"root_orientation_nonfinite_count={root_orientation_nonfinite_count}"
            )
        return positions, velocities

    def flywheel_collider_health(self) -> dict[str, object]:
        """Cache a fail-closed audit of live USD collision approximations.

        Collision authoring is immutable during an evaluation episode, so one
        composed-stage readback is sufficient and avoids adding per-step USD
        traversal overhead.  The returned mapping is carried through the
        existing health gate as infrastructure evidence, never as a setting
        change.
        """

        if self._flywheel_collider_health is None:
            try:
                self._flywheel_collider_health = audit_current_usd_stage()
            except (RuntimeError, TypeError, ValueError, AttributeError):
                self._flywheel_collider_health = {
                    "healthy": False,
                    "reason": "collider_static_audit_unavailable",
                    "metric_name": "unsupported_dynamic_collider_count",
                    "metric_value": "unavailable",
                    "metric_limit": 0,
                    "offending_colliders": [],
                }
        result = self._flywheel_collider_health
        if not isinstance(result, dict) or result.get("healthy") is not True:
            return result if isinstance(result, dict) else {
                "healthy": False,
                "reason": "collider_static_audit_unavailable",
                "metric_name": "unsupported_dynamic_collider_count",
                "metric_value": "unavailable",
                "metric_limit": 0,
                "offending_colliders": [],
            }
        return dict(result)

    def flywheel_cloth_physical_health(self) -> dict[str, object]:
        """Fail closed when live cloth state is numerically outside this scene's scale."""

        collider_health = self.flywheel_collider_health()
        if collider_health.get("healthy") is not True:
            return collider_health

        try:
            if str(getattr(self, "device", "")).lower() == "cpu":
                # CpuSimulationView cannot create a particle-cloth view.  The
                # exact source-bootstrap path therefore audits the current
                # authored USD-local state that it will snapshot.
                positions, velocities = self._flywheel_legacy_cpu_cloth_state()
            else:
                positions, velocities = self._flywheel_physics_cloth_state()
        except (RuntimeError, TypeError, ValueError) as error:
            return {
                "healthy": False,
                "reason": "simulator_numerical_divergence",
                "metric_name": "cloth_state_readback",
                "metric_value": str(error) or type(error).__name__,
                "metric_limit": "finite_aligned_nx3",
            }
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            return {"healthy": False, "reason": "simulator_numerical_divergence"}

        def _config_get(value, key, default):
            getter = getattr(value, "get", None)
            return getter(key, default) if callable(getter) else getattr(value, key, default)

        objects = _config_get(getattr(self, "particle_config", None), "objects", None)
        common = _config_get(objects, "common", {})
        particle_system = _config_get(objects, "particle_system", {})
        garment = getattr(self, "garment_config", None)
        scale = np.asarray(
            _config_get(garment, "scale", _config_get(common, "scale", [1.0, 1.0, 1.0])),
            dtype=np.float64,
        )
        reset_range = np.asarray(
            _config_get(
                garment,
                "soft_reset_pos_range",
                _config_get(common, "soft_reset_pos_range", [0.0] * 6),
            ),
            dtype=np.float64,
        )
        configured_max_velocity_mps = float(_config_get(particle_system, "max_velocity", 5.0))
        if (
            scale.shape != (3,)
            or reset_range.shape != (6,)
            or not np.isfinite(scale).all()
            or not np.isfinite(reset_range).all()
            or not np.isfinite(configured_max_velocity_mps)
            or configured_max_velocity_mps <= 0.0
        ):
            return {"healthy": False, "reason": "simulator_numerical_divergence"}
        # The reset configuration locates the garment and its existing scale
        # bounds its authored local mesh. This is an admission envelope only;
        # it does not alter PhysX settings or policy actions.
        reset_position_envelope_m = float(np.max(np.abs(reset_range)))
        garment_scale_envelope_m = float(np.max(np.abs(scale)))
        max_position_limit_m = reset_position_envelope_m + 2.0 * garment_scale_envelope_m
        max_extent_limit_m = 4.0 * garment_scale_envelope_m
        max_velocity_limit_mps = configured_max_velocity_mps + 1e-4
        max_position_m = float(np.max(np.abs(positions))) if positions.size else float("inf")
        max_extent_m = float(np.max(np.ptp(positions, axis=0))) if positions.size else float("inf")
        max_velocity_mps = float(np.max(np.linalg.norm(velocities, axis=1))) if velocities.size else float("inf")
        exceeded_metrics = []
        for metric_name, metric_value, metric_limit in (
            ("max_position_m", max_position_m, max_position_limit_m),
            ("max_extent_m", max_extent_m, max_extent_limit_m),
            ("max_velocity_mps", max_velocity_mps, max_velocity_limit_mps),
        ):
            if metric_value > metric_limit:
                exceeded_metrics.append(
                    {
                        "metric_name": metric_name,
                        "metric_value": metric_value,
                        "metric_limit": metric_limit,
                    }
                )
        if exceeded_metrics:
            return {
                "healthy": False,
                "reason": "simulator_numerical_divergence",
                "max_position_m": max_position_m,
                "max_extent_m": max_extent_m,
                "max_velocity_mps": max_velocity_mps,
                "max_position_limit_m": max_position_limit_m,
                "max_extent_limit_m": max_extent_limit_m,
                "max_velocity_limit_mps": max_velocity_limit_mps,
                "exceeded_metrics": exceeded_metrics,
            }
        return {
            "healthy": True,
            "max_position_m": max_position_m,
            "max_extent_m": max_extent_m,
            "max_velocity_mps": max_velocity_mps,
        }

    def _flywheel_cloth_backend(self) -> str:
        device = str(self.device).lower()
        if device == "cpu":
            # Historical source capture on CPU owns only the composed USD
            # local-particle attributes.  Never construct a PhysX view here:
            # CpuSimulationView does not provide particle-cloth views.
            return "usd_local_points_v1"
        if device == "cuda" or (device.startswith("cuda:") and device[5:].isdigit()):
            self._flywheel_physics_cloth_view()
            return "physx_cloth_view"
        raise RuntimeError(f"flywheel cloth state does not support device {self.device!r}")

    def flywheel_runtime_devices(self) -> dict[str, str]:
        """Read the live Kit renderer slot and prove all camera outputs exist.

        This intentionally does not consult launch arguments: the receipt must
        describe the renderer that Kit actually selected after startup.
        """

        try:
            import carb

            settings = carb.settings.get_settings()
            active_gpu = settings.get("/renderer/activeGpu")
        except Exception as error:
            raise RuntimeError("Kit active renderer GPU is unavailable") from error
        if not isinstance(active_gpu, int) or isinstance(active_gpu, bool) or active_gpu < 0:
            raise RuntimeError("Kit active renderer GPU is invalid")
        cameras = (self.top_camera, self.left_camera, self.right_camera)
        for camera in cameras:
            output = getattr(getattr(camera, "data", None), "output", None)
            if not isinstance(output, dict) or "rgb" not in output or output["rgb"] is None:
                raise RuntimeError("live camera output ownership is unavailable")
        device = f"cuda:{active_gpu}"
        return {"renderer_device": device, "camera_device": device}

    def flywheel_capture_state(self) -> dict[str, object]:
        """Return the complete mutable simulator state needed for hard replay."""
        if self.object is None:
            raise RuntimeError("cannot snapshot an uninitialized garment")

        def _numpy(value):
            return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

        if str(self.device).lower() == "cpu":
            # The historically stable source-only CPU path owns USD-local
            # particles.  Read the live authored values rather than rest
            # constants so the schema explicitly records its frame.
            positions, velocities = self._flywheel_legacy_cpu_cloth_state()
            cloth_state_authority = "usd_local_points_v1"
        else:
            if self._flywheel_cloth_backend() != "physx_cloth_view":
                raise RuntimeError("controlled snapshots require the live PhysX cloth view")
            positions, velocities = self._flywheel_physics_cloth_state()
            cloth_state_authority = "physx_cloth_view_world_v1"
        rng_name, rng_keys, rng_pos, rng_gauss, rng_cached = self.garment_rng.get_state()
        return {
            "robot_position": np.concatenate(
                (_numpy(self.left_arm.data.joint_pos)[0], _numpy(self.right_arm.data.joint_pos)[0])
            ),
            "robot_velocity": np.concatenate(
                (_numpy(self.left_arm.data.joint_vel)[0], _numpy(self.right_arm.data.joint_vel)[0])
            ),
            "cloth_position": positions,
            "cloth_velocity": velocities,
            "rng_state": {
                "kind": "numpy.RandomState",
                "name": rng_name,
                "keys": [int(value) for value in rng_keys.tolist()],
                "position": int(rng_pos),
                "has_gauss": int(rng_gauss),
                "cached_gaussian": float(rng_cached),
            },
            "garment_name": self.cfg.garment_name,
            "scene_state": self._flywheel_capture_scene_state(),
            "cloth_state_authority": cloth_state_authority,
        }

    def flywheel_visible_garment_contact(self) -> dict[str, object]:
        """Read actual Isaac particle and gripper geometry for rollout contact evidence."""
        if self.object is None:
            raise RuntimeError("cannot read visible contact without an initialized garment")

        def _numpy(value):
            return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

        if str(self.device).lower() == "cpu":
            # CPU source snapshots are USD-local.  Contact geometry is
            # world-space, so transform only the current live local positions
            # through the garment's current world frame; velocities are not a
            # contact input and are deliberately not translated.
            local_positions, _ = self._flywheel_legacy_cpu_cloth_state()
            try:
                root_position, root_orientation = self.object.get_world_pose()
                root_position = _numpy(root_position)
                root_orientation = _numpy(root_orientation)
                world_scale = _numpy(self.object.get_world_scale())
            except (RuntimeError, TypeError, ValueError, AttributeError) as error:
                raise RuntimeError("garment world transform readback API failure") from error
            if root_position.ndim == 2 and root_position.shape[0] == 1:
                root_position = root_position[0]
            if root_orientation.ndim == 2 and root_orientation.shape[0] == 1:
                root_orientation = root_orientation[0]
            if root_position.shape != (3,) or root_orientation.shape != (4,):
                raise RuntimeError("garment world transform has an unsupported shape")
            if not np.isfinite(root_position).all() or not np.isfinite(root_orientation).all():
                raise RuntimeError("garment world transform is nonfinite")
            try:
                from isaacsim.core.utils.rotations import quat_to_rot_matrix

                rotation = quat_to_rot_matrix(root_orientation)
                particle_positions, _ = self._flywheel_legacy_local_to_world(
                    local_positions,
                    np.zeros_like(local_positions),
                    root_position,
                    rotation,
                    world_scale,
                )
            except (RuntimeError, TypeError, ValueError, AttributeError) as error:
                raise RuntimeError("legacy CPU garment world transform is unavailable") from error
        else:
            if self._flywheel_cloth_backend() != "physx_cloth_view":
                raise RuntimeError("visible contact requires the live PhysX cloth view")
            particle_positions, _ = self._flywheel_physics_cloth_state()
        gripper_positions = []
        for arm in (self.left_arm, self.right_arm):
            names = getattr(arm, "body_names", None)
            positions = getattr(arm.data, "body_pos_w", None)
            if not isinstance(names, (list, tuple)) or positions is None:
                raise RuntimeError("Isaac robot does not expose gripper body positions for contact evidence")
            matches = [index for index, name in enumerate(names) if "gripper" in str(name).lower()]
            if not matches:
                raise RuntimeError("Isaac robot does not expose a gripper body for contact evidence")
            body_positions = _numpy(positions)
            if body_positions.ndim != 3 or body_positions.shape[0] != 1:
                raise RuntimeError("Isaac gripper body positions have an unsupported shape")
            gripper_positions.extend(body_positions[0, index] for index in matches)
        from lehome.flywheel.contact import visible_contact_from_simulator_geometry

        return visible_contact_from_simulator_geometry(particle_positions, np.asarray(gripper_positions))

    def flywheel_restore_state(self, snapshot) -> None:
        """Restore a validated flywheel snapshot without touching reset/reward logic."""
        if self.object is None:
            raise RuntimeError("cannot restore an uninitialized garment")
        if snapshot.garment_name != self.cfg.garment_name:
            raise ValueError("snapshot garment does not match the active environment")
        schema_version = getattr(snapshot, "schema_version", None)
        device_name = str(self.device).lower()
        authority = getattr(snapshot, "cloth_state_authority", None)
        legacy_local = schema_version == 3 and authority == "usd_local_points_v1"
        legacy_cpu = device_name == "cpu" and (schema_version == 1 or legacy_local)
        self._flywheel_legacy_projection_receipt = None
        cloth_position, cloth_velocity = self._flywheel_cloth_arrays(
            snapshot.cloth_position, snapshot.cloth_velocity
        )
        if legacy_local and not legacy_cpu:
            pose = np.asarray(snapshot.scene_state.get("garment_reset_pose"), dtype=np.float32)
            if pose.shape != (6,) or not np.isfinite(pose).all():
                raise ValueError("legacy USD-local cloth restore requires a finite garment reset pose")
            from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_rot_matrix

            rotation = quat_to_rot_matrix(euler_angles_to_quat(pose[3:], degrees=True))
            scale = self.object.get_world_scale()
            def to_numpy(value):
                return (
                    value.detach().cpu().numpy()
                    if callable(getattr(value, "detach", None))
                    else np.asarray(value)
                )
            asset_points = to_numpy(self.object._get_points_pose()).reshape(-1, 3)
            live_rest_world = to_numpy(self.object.initial_points_positions).reshape(-1, 3)
            initial_root_position = to_numpy(self.object.initial_root_position).reshape(3)
            initial_root_orientation = to_numpy(self.object.initial_root_orientation).reshape(4)
            initial_root_rotation = quat_to_rot_matrix(initial_root_orientation)
            asset_rest_world, _ = self._flywheel_legacy_local_to_world(
                asset_points,
                np.zeros_like(asset_points),
                initial_root_position,
                initial_root_rotation,
                scale,
            )
            welded_vertices_remap_to_orig, welded_vertices_remap_to_weld = (
                self._flywheel_physx_weld_maps(
                    asset_positions=asset_rest_world,
                    live_positions=live_rest_world,
                )
            )
            def weld_digest(value) -> str:
                vector = np.asarray(value)
                if vector.ndim != 1 or vector.dtype.kind not in "iu":
                    raise SimulatorNumericalDivergenceError(
                        "legacy USD to PhysX cloth topology map digest is invalid"
                    )
                return hashlib.sha256(vector.astype(np.int64, copy=False).tobytes()).hexdigest()

            weld_digests = {
                "welded_vertices_remap_to_orig_sha256": weld_digest(welded_vertices_remap_to_orig),
                "welded_vertices_remap_to_weld_sha256": weld_digest(welded_vertices_remap_to_weld),
            }
            weld_map_identity = hashlib.sha256(json.dumps({
                "mesh_prim_path": str(self.object.mesh_prim_path),
                **weld_digests,
            }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            cloth_position, cloth_velocity = self._flywheel_project_legacy_usd_to_physx(
                cloth_position, cloth_velocity, asset_points, live_rest_world,
                welded_vertices_remap_to_orig, welded_vertices_remap_to_weld,
            )
            cloth_position, cloth_velocity = self._flywheel_legacy_local_to_world(
                cloth_position, cloth_velocity, pose[:3], rotation, scale
            )
        if legacy_cpu:
            positions_attr, velocities_attr = self._flywheel_legacy_cpu_cloth_attributes()
            if (not callable(getattr(positions_attr, "Set", None))
                    or not callable(getattr(velocities_attr, "Set", None))):
                raise RuntimeError("legacy CPU garment USD points or velocities are not writable")
            cloth = None
        elif schema_version == 1:
            if self._flywheel_cloth_backend() != "physx_cloth_view":
                raise RuntimeError("legacy CUDA restore requires the live PhysX cloth view")
            cloth = self._flywheel_physics_cloth_view()
        elif legacy_local:
            if self._flywheel_cloth_backend() != "physx_cloth_view":
                raise RuntimeError("legacy USD-local restore requires the live PhysX cloth view")
            cloth = self._flywheel_physics_cloth_view()
        elif (schema_version == 2
                and getattr(snapshot, "cloth_state_authority", None) == "physx_cloth_view_world_v1"):
            if self._flywheel_cloth_backend() != "physx_cloth_view":
                raise RuntimeError("controlled restore requires the live PhysX cloth view")
            cloth = self._flywheel_physics_cloth_view()
        else:
            raise ValueError("snapshot has no supported cloth-state authority")
        rng_state = snapshot.rng_state
        if rng_state.get("kind") != "numpy.RandomState":
            raise ValueError("unsupported flywheel RNG snapshot")
        try:
            restored_rng_state = (
                str(rng_state["name"]),
                np.asarray(rng_state["keys"], dtype=np.uint32),
                int(rng_state["position"]),
                int(rng_state["has_gauss"]),
                float(rng_state["cached_gaussian"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("unsupported flywheel RNG snapshot") from error
        if not hasattr(self.left_arm, "write_joint_position_to_sim") or not hasattr(self.right_arm, "write_joint_position_to_sim"):
            raise RuntimeError("Isaac articulation does not expose restorable joint position")
        if not hasattr(self.left_arm, "write_joint_velocity_to_sim") or not hasattr(self.right_arm, "write_joint_velocity_to_sim"):
            raise RuntimeError("Isaac articulation does not expose restorable joint velocity")
        device = self.device
        robot_position = torch.tensor(snapshot.robot_position, dtype=torch.float32, device=device).unsqueeze(0)
        robot_velocity = torch.tensor(snapshot.robot_velocity, dtype=torch.float32, device=device).unsqueeze(0)
        self.left_arm.write_joint_position_to_sim(robot_position[:, :6])
        self.right_arm.write_joint_position_to_sim(robot_position[:, 6:])
        self.left_arm.write_joint_velocity_to_sim(robot_velocity[:, :6])
        self.right_arm.write_joint_velocity_to_sim(robot_velocity[:, 6:])
        self.garment_rng.set_state(restored_rng_state)
        if snapshot.scene_state:
            self._flywheel_restore_scene_state(snapshot.scene_state)
            if snapshot.randomization.get("strategy") == "canonical":
                self._flywheel_randomization_baseline = dict(snapshot.scene_state)
        # Scene restoration may author USD points. Apply the selected snapshot
        # cloth representation last. Controlled recovery never reaches the
        # legacy branch because its loader requires a v2 PhysX authority.
        if legacy_cpu:
            if positions_attr.Set(self._flywheel_legacy_usd_vec3f_array(cloth_position)) is False:
                raise RuntimeError("legacy CPU garment USD points write failed")
            if velocities_attr.Set(self._flywheel_legacy_usd_vec3f_array(cloth_velocity)) is False:
                raise RuntimeError("legacy CPU garment USD velocities write failed")
            observed_position, observed_velocity = self._flywheel_legacy_cpu_cloth_state()
        else:
            cloth.set_world_positions(torch.tensor(cloth_position, dtype=torch.float32, device=device).unsqueeze(0))
            cloth.set_velocities(torch.tensor(cloth_velocity, dtype=torch.float32, device=device).unsqueeze(0))
            observed_position, observed_velocity = self._flywheel_physics_cloth_state()
        if not (
            np.allclose(observed_position, cloth_position, rtol=0.0, atol=1e-6)
            and np.allclose(observed_velocity, cloth_velocity, rtol=0.0, atol=1e-6)
        ):
            raise RuntimeError("garment cloth write readback mismatch")
        restored_pose = np.asarray(self.object.get_all_pose()["Garment"], dtype=np.float32)
        if restored_pose.shape != (6,) or not np.isfinite(restored_pose).all():
            raise RuntimeError("garment restored pose readback is invalid")
        self._flywheel_preserved_restore_for_randomization = {
            "positions": np.asarray(observed_position, dtype=np.float32).copy(),
            "velocities": np.asarray(observed_velocity, dtype=np.float32).copy(),
            "pose": restored_pose.copy(),
        }
        self._flywheel_randomization_receipt = dict(snapshot.randomization)
        if legacy_local and not legacy_cpu:
            self._flywheel_legacy_projection_receipt = {
                "source_snapshot_authority": "usd_local_points_v1",
                "weld_map_identity": weld_map_identity,
                **weld_digests,
            }

    def apply_flywheel_randomization(self, randomization) -> dict[str, object]:
        """Apply opt-in rollout perturbations and return values read back from Isaac.

        This method is intentionally fail-closed: a caller must not write a
        randomized manifest when any configured simulator property cannot be
        applied and observed again.
        """
        values = dict(randomization.values if hasattr(randomization, "values") else randomization)
        preserved_restore = getattr(self, "_flywheel_preserved_restore_for_randomization", None)
        self._flywheel_preserved_restore_for_randomization = None
        # Canonical is a control, not a randomization strategy.  In particular,
        # do not restore a captured scene here: evaluation calls this after the
        # garment has settled, and scene restoration resets the cloth pose.
        if not values:
            self._flywheel_randomization_receipt = {}
            return {}
        baseline = getattr(self, "_flywheel_randomization_baseline", None)
        if baseline is None:
            baseline = self._flywheel_capture_scene_state()
            self._flywheel_randomization_baseline = baseline
        # Every non-canonical strategy starts from the same exact scene.
        self._flywheel_restore_scene_state(baseline)
        from lehome.flywheel.randomization import randomization_materials_enabled

        materials_enabled = randomization_materials_enabled(values)
        if self.object is None:
            raise RuntimeError("cannot randomize an uninitialized garment")
        stage = self.scene.stage
        table_input = None
        read_color = None
        if materials_enabled:
            texture_folder = self.texture_cfg.get("folder", "")
            if not os.path.isabs(texture_folder):
                texture_folder = os.path.join(os.getcwd(), texture_folder)
            texture_path = os.path.join(texture_folder, f"{int(values['table_texture_id'])}.png")
            if not os.path.isfile(texture_path):
                raise RuntimeError("flywheel table texture asset is missing")
            table_prim = stage.GetPrimAtPath(self.texture_cfg.get("prim_path", ""))
            if not table_prim.IsValid():
                raise RuntimeError("flywheel table shader prim is missing")
            table_shader = UsdShade.Shader(table_prim)
            table_input = table_shader.GetInput("file") or table_shader.GetInput("diffuse_texture")
            if not table_input:
                raise RuntimeError("flywheel table shader input is missing")
            table_input.Set(Sdf.AssetPath(texture_path))
            if str(table_input.Get().path) != texture_path:
                raise RuntimeError("flywheel table shader readback mismatch")
            mesh = stage.GetPrimAtPath(self.object.mesh_prim_path)
            color_attr = mesh.GetAttribute("primvars:displayColor")
            color = tuple(float(value) for value in values["garment_display_color"])
            if not mesh.IsValid() or not color_attr.IsValid():
                raise RuntimeError("flywheel garment displayColor is missing")
            color_attr.Set([color])
            read_color = tuple(float(value) for value in color_attr.Get()[0])
            if not np.allclose(read_color, color, atol=1e-6):
                raise RuntimeError("flywheel garment displayColor readback mismatch")
        camera_delta = np.asarray(values["camera_translation_m"], dtype=np.float32)
        base_delta = np.asarray(values["robot_base_translation_m"], dtype=np.float32)
        if camera_delta.shape != (3,) or base_delta.shape != (3,):
            raise ValueError("flywheel translation randomization must be 3-D")
        light_path = self.flywheel_randomization_cfg.get("light_prim_path", "/World/Light")
        light = stage.GetPrimAtPath(light_path)
        if not light.IsValid():
            raise RuntimeError(f"flywheel light prim is missing: {light_path}")
        intensity_attr = light.GetAttribute("inputs:intensity")
        base_intensity = intensity_attr.Get()
        if base_intensity is None:
            raise RuntimeError("flywheel light intensity is unreadable")
        intensity_attr.Set(float(base_intensity) * float(values["light_intensity_scale"]))

        for camera in (self.top_camera, self.left_camera, self.right_camera):
            current_positions, current_orientations = read_camera_world_pose(camera)
            positions = current_positions + torch.tensor(camera_delta, device=self.device).unsqueeze(0)
            write_camera_world_pose(camera, positions, current_orientations)

        pose = np.asarray(self.object.get_all_pose()["Garment"], dtype=np.float32)
        pose[5] += float(values["garment_yaw_deg"])
        self.set_all_pose({"Garment": pose})
        if not np.isclose(float(self.object.reset_pose[5]), float(pose[5]), atol=1e-5):
            raise RuntimeError("garment yaw readback did not match requested randomization")

        for arm in (self.left_arm, self.right_arm):
            if not hasattr(arm, "write_root_pose_to_sim") or not hasattr(arm.data, "root_pos_w"):
                raise RuntimeError("Isaac robot base does not expose restorable world pose")
            position = arm.data.root_pos_w + torch.tensor(base_delta, device=self.device).unsqueeze(0)
            root_pose = torch.cat((position, arm.data.root_quat_w), dim=-1)
            arm.write_root_pose_to_sim(root_pose)
            actual = arm.data.root_pos_w.detach().cpu().numpy()[0]
            if not np.allclose(actual, position.detach().cpu().numpy()[0], atol=1e-5):
                raise RuntimeError("robot base pose readback did not match requested randomization")

        if preserved_restore is not None:
            source_pose = np.asarray(preserved_restore["pose"], dtype=np.float32)
            target_pose = np.asarray(self.object.get_all_pose()["Garment"], dtype=np.float32)
            if (
                source_pose.shape != (6,)
                or target_pose.shape != (6,)
                or not np.isfinite(source_pose).all()
                or not np.isfinite(target_pose).all()
            ):
                raise RuntimeError("garment randomized pose readback is invalid")
            from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_rot_matrix

            source_rotation = quat_to_rot_matrix(
                euler_angles_to_quat(source_pose[3:], degrees=True)
            )
            target_rotation = quat_to_rot_matrix(
                euler_angles_to_quat(target_pose[3:], degrees=True)
            )
            cloth_position, cloth_velocity = self._flywheel_rebase_world_cloth(
                preserved_restore["positions"], preserved_restore["velocities"],
                source_pose[:3], source_rotation, target_pose[:3], target_rotation,
            )
            cloth = self._flywheel_physics_cloth_view()
            cloth.set_world_positions(
                torch.tensor(cloth_position, dtype=torch.float32, device=self.device).unsqueeze(0)
            )
            cloth.set_velocities(
                torch.tensor(cloth_velocity, dtype=torch.float32, device=self.device).unsqueeze(0)
            )
            observed_position, observed_velocity = self._flywheel_physics_cloth_state()
            if not (
                np.allclose(observed_position, cloth_position, rtol=0.0, atol=1e-6)
                and np.allclose(observed_velocity, cloth_velocity, rtol=0.0, atol=1e-6)
            ):
                raise RuntimeError("randomized authenticated cloth write readback mismatch")

        receipt = {
            "light_intensity_scale": float(intensity_attr.Get()) / float(base_intensity),
            "camera_translation_m": tuple(float(value) for value in camera_delta),
            "garment_yaw_deg": float(self.object.reset_pose[5] - (pose[5] - float(values["garment_yaw_deg"]))),
            "robot_base_translation_m": tuple(float(value) for value in base_delta),
        }
        if materials_enabled:
            receipt.update({
                "table_texture_id": int(values["table_texture_id"]),
                "table_texture_path": str(table_input.Get().path),
                "table_shader_input": table_input.GetBaseName(),
                "garment_display_color": read_color,
            })
        from lehome.flywheel.randomization import validate_randomization_receipt

        validate_randomization_receipt(values, receipt)
        self._flywheel_randomization_receipt = receipt
        return receipt

    def switch_garment(self, garment_name: str, garment_version: str = None):
        """Switch to a different garment without recreating the environment.

        This method allows reusing the same environment instance for different garments,
        which is much faster than closing and recreating the environment.

        Args:
            garment_name: Name of the garment to switch to (e.g., "Top_Long_Seen_0")
            garment_version: Version of the garment ("Release" or "Holdout"),
                            defaults to current cfg.garment_version
        """
        logger.info(
            f"[GarmentEnv] Switching garment to: {garment_name} (version: {garment_version})"
        )

        if garment_version is None:
            garment_version = self.cfg.garment_version

        # Validate the replacement while the current garment is still intact.
        # Bad assets must fail without destroying the usable scene.
        next_garment_config = self.garment_loader.load_garment_config(
            garment_name, garment_version
        )

        if self.object is not None:
            self._delete_garment_object()
            logger.info("[GarmentEnv] Old garment object deleted")

        # Update config
        self.cfg.garment_name = garment_name
        self.cfg.garment_version = garment_version

        self.garment_config = next_garment_config
        logger.debug(f"[GarmentEnv] Garment config reloaded for {garment_name}")

        # solve particle ditorition
        logger.debug(
            f"[GarmentEnv] Running physics steps to clean up old particle system..."
        )
        cleanup_steps = 20

        if hasattr(self, "sim") and self.sim is not None:
            for i in range(cleanup_steps):
                try:
                    self.sim.step(render=True)
                    # Log progress every 5 steps
                    if (i + 1) % 5 == 0:
                        logger.debug(
                            f"[GarmentEnv] Cleanup progress: {i+1}/{cleanup_steps}"
                        )
                except Exception as e:
                    logger.warning(f"[GarmentEnv] Error during cleanup step {i+1}: {e}")
                    # Continue with next step
                    continue
            logger.debug(f"[GarmentEnv] Cleanup physics steps completed")
        else:
            logger.warning(f"[GarmentEnv] sim not available, skipping cleanup steps")

        # create new garment object
        self._create_garment_object()
        logger.debug(f"[GarmentEnv] New garment object created for {garment_name}")
        logger.debug(
            f"[GarmentEnv] Running initial physics steps to register prim in stage..."
        )
        initial_steps = 5
        if hasattr(self, "sim") and self.sim is not None:
            for i in range(initial_steps):
                try:
                    self.sim.step(render=True)
                except Exception as e:
                    logger.warning(f"[GarmentEnv] Error during initial step {i+1}: {e}")
            logger.debug(f"[GarmentEnv] Initial physics steps completed")
        else:
            logger.warning(f"[GarmentEnv] sim not available, skipping initial steps")
        if hasattr(self, "render"):
            try:
                self.render()
                logger.debug(f"[GarmentEnv] Render called after initial physics steps")
            except Exception as e:
                logger.warning(
                    f"[GarmentEnv] Error during render after initial steps: {e}"
                )

        try:
            self.initialize_obs()
            logger.debug(
                f"[GarmentEnv] Observation system initialized for {garment_name}"
            )
            if hasattr(self, "render"):
                try:
                    self.render()
                    logger.debug(
                        f"[GarmentEnv] Render called after observation initialization"
                    )
                except Exception as e:
                    logger.debug(
                        f"[GarmentEnv] Error during render after observation init: {e}"
                    )
        except Exception as e:
            logger.warning(
                f"[GarmentEnv] Failed to initialize observations (may be expected): {e}"
            )

    def cleanup(self):
        """Cleanup method (defensive programming).

        Note: When environments are fully closed and recreated (as in eval.py),
        this cleanup is not strictly necessary since _create_garment_object()
        already handles checking and cleaning up existing prims when creating
        a new environment. However, this method is kept as a safety measure
        for cases where the same environment instance might be reused.
        """
        logger.debug("[GarmentEnv] Starting cleanup...")

        # Delete garment object if it exists
        if self.object is not None:
            self._delete_garment_object()
            logger.debug("[GarmentEnv] Garment object cleaned up")

        # Clear references
        self.object = None
        # Note: Don't clear garment_config and particle_config as they might be needed
        # if the environment is reset rather than recreated

        logger.debug("[GarmentEnv] Cleanup completed")

    def __del__(self):
        """Destructor to ensure cleanup on deletion."""
        try:
            if hasattr(self, "object") and self.object is not None:
                self.cleanup()
        except Exception:
            # Ignore errors during destruction
            pass
