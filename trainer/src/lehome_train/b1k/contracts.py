"""Secret-free configuration contract for the initial Behavior 1K run."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Mapping

from lehome_train.constants import (
    BEHAVIOR_1K_DATASET_REPOSITORY,
    BEHAVIOR_1K_DATASET_REVISION,
    BEHAVIOR_1K_FINAL_MODEL_REPOSITORY,
    BEHAVIOR_1K_CHECKPOINT_BUCKET,
    ISAAC_GROOT_REPOSITORY,
    ISAAC_GROOT_REVISION,
    MODEL_REVISION,
    COSMOS_REPOSITORY,
    COSMOS_REVISION,
)
from lehome_train.b1k.training import SUPPORTED_GPU_COUNTS, approved_launch_plans


_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_R1PRO_MODALITY_SHA256 = "ca3a2e406472650bee9439ed81a3f4a1531b6fe689cdc1b348d0f260a208e641"


def _value(values: Mapping[str, str], key: str, *, default: str | None = None) -> str:
    value = values.get(key, default)
    if type(value) is not str or not value.strip():
        raise ValueError(f"{key} must be non-empty")
    return value.strip()


def _integer(values: Mapping[str, str], key: str, *, default: str | None = None) -> int:
    raw = _value(values, key, default=default)
    if not raw.isdecimal():
        raise ValueError(f"{key} must be an integer")
    return int(raw)


@dataclass(frozen=True, slots=True)
class RunContract:
    """Validated identity data that deliberately excludes ``HF_TOKEN``."""

    dataset_repo: str
    model_repo: str
    checkpoint_bucket: str
    dataset_revision: str
    groot_repository: str
    groot_revision: str
    container_digest: str
    run_id: str
    cycle_id: str
    train_steps: int
    save_steps: int
    checkpoint_keep: int
    resume_policy: str
    auto_destroy: bool
    manifest_path: str | None
    parent_cycle_id: str | None
    base_model_revision: str
    task_manifest_sha256: str
    modality_sha256: str
    stats_sha256: str
    launch_plan_id: str
    world_size: int
    learning_rate: float
    cosmos_repository: str
    cosmos_revision: str
    experiment_name: str
    launch_arguments_sha256: str
    physical_batch_size: int
    global_batch_size: int
    gradient_accumulation_steps: int
    effective_global_batch_size: int
    weight_decay: float
    warmup_ratio: float

    @classmethod
    def from_environment(cls, values: Mapping[str, str]) -> "RunContract":
        # Validate the credential's presence but do not assign it to any object.
        _value(values, "HF_TOKEN")
        dataset_repo = _value(values, "HF_DATASET_REPO")
        if dataset_repo != BEHAVIOR_1K_DATASET_REPOSITORY:
            raise ValueError("HF_DATASET_REPO must be the pinned Behavior 1K dataset")
        model_repo = _value(values, "HF_MODEL_REPO")
        if model_repo != BEHAVIOR_1K_FINAL_MODEL_REPOSITORY:
            raise ValueError("HF_MODEL_REPO must be the B1K final model repository")
        checkpoint_bucket = _value(values, "HF_CHECKPOINT_BUCKET")
        if checkpoint_bucket != BEHAVIOR_1K_CHECKPOINT_BUCKET:
            raise ValueError("HF_CHECKPOINT_BUCKET must be the B1K rolling checkpoint bucket")
        dataset_revision = _value(values, "DATASET_REVISION")
        if dataset_revision != BEHAVIOR_1K_DATASET_REVISION:
            raise ValueError("DATASET_REVISION must equal the pinned Behavior 1K commit")
        groot_revision = _value(values, "GROOT_REVISION")
        if groot_revision != ISAAC_GROOT_REVISION:
            raise ValueError("GROOT_REVISION must equal the pinned ISAAC_GROOT_REVISION")
        container_digest = _value(values, "CONTAINER_DIGEST")
        if not _DIGEST.fullmatch(container_digest):
            raise ValueError("CONTAINER_DIGEST must be sha256:<64 lowercase hex characters>")
        run_id = _value(values, "RUN_ID")
        cycle_id = _value(values, "CYCLE_ID")
        for key, identifier in (("RUN_ID", run_id), ("CYCLE_ID", cycle_id)):
            if not _SAFE_IDENTIFIER.fullmatch(identifier):
                raise ValueError(f"{key} must be a safe filesystem identifier")
        train_steps = _integer(values, "TRAIN_STEPS")
        if train_steps != 15_000:
            raise ValueError("TRAIN_STEPS must be exactly 15,000 for the initial run")
        save_steps = _integer(values, "SAVE_STEPS")
        if save_steps != 1_000:
            raise ValueError("SAVE_STEPS must be exactly 1,000 for the initial run")
        checkpoint_keep = _integer(values, "CHECKPOINT_KEEP", default="2")
        if checkpoint_keep != 2:
            raise ValueError("CHECKPOINT_KEEP must be exactly 2")
        resume_policy = _value(values, "RESUME_POLICY", default="auto")
        if resume_policy not in {"auto", "never", "require"}:
            raise ValueError("RESUME_POLICY must be auto, never, or require")
        auto_destroy_raw = _value(values, "AUTO_DESTROY", default="0")
        if auto_destroy_raw != "0":
            raise ValueError("AUTO_DESTROY must be exactly 0")
        base_model_revision = _value(values, "BASE_MODEL_REVISION")
        if base_model_revision != MODEL_REVISION:
            raise ValueError("BASE_MODEL_REVISION must equal the pinned base model revision")
        fingerprints = {
            "TASK_MANIFEST_SHA256": _value(values, "TASK_MANIFEST_SHA256"),
            "MODALITY_SHA256": _value(values, "MODALITY_SHA256"),
            "STATS_SHA256": _value(values, "STATS_SHA256"),
        }
        if not all(_SHA256.fullmatch(value) for value in fingerprints.values()):
            raise ValueError("TASK_MANIFEST_SHA256, MODALITY_SHA256, and STATS_SHA256 must be SHA-256 values")
        if fingerprints["MODALITY_SHA256"] != _R1PRO_MODALITY_SHA256:
            raise ValueError("MODALITY_SHA256 must equal the pinned R1Pro modality fingerprint")
        world_size = _integer(values, "WORLD_SIZE")
        if world_size not in SUPPORTED_GPU_COUNTS:
            raise ValueError("WORLD_SIZE must be one to four")
        launch_plan_id = _value(values, "LAUNCH_PLAN_ID")
        if launch_plan_id not in {plan.identity for plan in approved_launch_plans(num_gpus=world_size)}:
            raise ValueError("LAUNCH_PLAN_ID must be an approved plan for WORLD_SIZE")
        learning_rate_raw = _value(values, "LEARNING_RATE")
        try:
            learning_rate = float(learning_rate_raw)
        except ValueError as error:
            raise ValueError("LEARNING_RATE must be a positive finite number") from error
        if learning_rate != 1e-4:
            raise ValueError("LEARNING_RATE must equal the approved 1e-4 plan value")
        cosmos_repository = _value(values, "COSMOS_REPOSITORY")
        cosmos_revision = _value(values, "COSMOS_REVISION")
        if cosmos_repository != COSMOS_REPOSITORY or cosmos_revision != COSMOS_REVISION:
            raise ValueError("COSMOS_REPOSITORY and COSMOS_REVISION must equal their pinned identities")
        experiment_name = _value(values, "EXPERIMENT_NAME")
        if experiment_name != run_id:
            raise ValueError("EXPERIMENT_NAME must equal RUN_ID for checkpoint isolation")
        launch_arguments_sha256 = _value(values, "LAUNCH_ARGUMENTS_SHA256")
        if not _SHA256.fullmatch(launch_arguments_sha256):
            raise ValueError("LAUNCH_ARGUMENTS_SHA256 must be a SHA-256 value")
        plan = next(plan for plan in approved_launch_plans(num_gpus=world_size) if plan.identity == launch_plan_id)
        manifest_path = values.get("TASK_MANIFEST")
        if manifest_path is not None:
            if type(manifest_path) is not str or not manifest_path.strip():
                raise ValueError("TASK_MANIFEST must be non-empty when supplied")
            manifest_path = manifest_path.strip()
        parent_cycle_id = values.get("PARENT_CYCLE_ID")
        if parent_cycle_id is not None:
            if type(parent_cycle_id) is not str or not _SAFE_IDENTIFIER.fullmatch(parent_cycle_id):
                raise ValueError("PARENT_CYCLE_ID must be a safe filesystem identifier")
        return cls(
            dataset_repo=dataset_repo,
            model_repo=model_repo,
            checkpoint_bucket=checkpoint_bucket,
            dataset_revision=dataset_revision,
            groot_repository=ISAAC_GROOT_REPOSITORY,
            groot_revision=groot_revision,
            container_digest=container_digest,
            run_id=run_id,
            cycle_id=cycle_id,
            train_steps=train_steps,
            save_steps=save_steps,
            checkpoint_keep=checkpoint_keep,
            resume_policy=resume_policy,
            auto_destroy=False,
            manifest_path=manifest_path,
            parent_cycle_id=parent_cycle_id,
            base_model_revision=base_model_revision,
            task_manifest_sha256=fingerprints["TASK_MANIFEST_SHA256"],
            modality_sha256=fingerprints["MODALITY_SHA256"],
            stats_sha256=fingerprints["STATS_SHA256"],
            launch_plan_id=launch_plan_id,
            world_size=world_size,
            learning_rate=learning_rate,
            cosmos_repository=cosmos_repository,
            cosmos_revision=cosmos_revision,
            experiment_name=experiment_name,
            launch_arguments_sha256=launch_arguments_sha256,
            physical_batch_size=plan.physical_batch_size,
            global_batch_size=plan.global_batch_size,
            gradient_accumulation_steps=plan.gradient_accumulation_steps,
            effective_global_batch_size=plan.effective_global_batch_size,
            weight_decay=plan.weight_decay,
            warmup_ratio=plan.warmup_ratio,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "auto_destroy": self.auto_destroy,
            "base_model_revision": self.base_model_revision,
            "checkpoint_bucket": self.checkpoint_bucket,
            "checkpoint_keep": self.checkpoint_keep,
            "container_digest": self.container_digest,
            "cycle_id": self.cycle_id,
            "dataset_repo": self.dataset_repo,
            "dataset_revision": self.dataset_revision,
            "groot_revision": self.groot_revision,
            "groot_repository": self.groot_repository,
            "manifest_path": self.manifest_path,
            "task_manifest_sha256": self.task_manifest_sha256,
            "modality_sha256": self.modality_sha256,
            "stats_sha256": self.stats_sha256,
            "launch_plan_id": self.launch_plan_id,
            "world_size": self.world_size,
            "learning_rate": self.learning_rate,
            "cosmos_repository": self.cosmos_repository,
            "cosmos_revision": self.cosmos_revision,
            "experiment_name": self.experiment_name,
            "launch_arguments_sha256": self.launch_arguments_sha256,
            "physical_batch_size": self.physical_batch_size,
            "global_batch_size": self.global_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_global_batch_size": self.effective_global_batch_size,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "model_repo": self.model_repo,
            "parent_cycle_id": self.parent_cycle_id,
            "resume_policy": self.resume_policy,
            "run_id": self.run_id,
            "save_steps": self.save_steps,
            "train_steps": self.train_steps,
        }
