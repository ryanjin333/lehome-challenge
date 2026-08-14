"""Immutable, fail-closed inputs for the pinned GR00T N1.7 launcher."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping

from lehome_train.constants import MODEL_REVISION
from lehome_train.flywheel.augmentation import (
    augmentation_profile as resolve_augmentation_profile,
    validated_augmentation_receipt,
)


ACTION_HORIZON = 16
LR_SCHEDULER_TYPE = "cosine"
DECAY_SEMANTICS = "cosine_remainder_after_warmup"
_PINNED_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_nonempty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


def _require_pinned_revision(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not _PINNED_REVISION.fullmatch(value):
        raise ValueError(f"{field_name} must be a pinned 40-character revision")


@dataclass(frozen=True, slots=True)
class FineTuneLaunchConfig:
    """The deliberately narrow contract passed to official ``launch_finetune``.

    ``base_model_path`` must be a revision-verified local snapshot before this
    adapter runs.  The official N1.7 entry point accepts a path but no separate
    Hub revision flag, so passing a floating Hub name here would make a run
    non-reproducible.  The separately recorded revision is therefore required
    and preflight is responsible for proving the snapshot has that identity.
    """

    base_model_path: str
    base_model_revision: str
    dataset_path: str
    dataset_revision: str
    modality_config_path: str
    output_dir: str
    experiment_name: str
    physical_batch_size: int
    max_steps: int
    save_steps: int
    warmup_ratio: float
    num_gpus: int = 1
    global_batch_size: int | None = None
    gradient_accumulation_steps: int = 1
    model_action_chunk_capacity: int = 40
    training_action_horizon: int = ACTION_HORIZON
    tune_llm: bool = False
    tune_visual: bool = False
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    dataloader_num_workers: int = 4
    save_total_limit: int = 5
    augmentation_profile: str = "none"
    augmentation_receipt: Mapping[str, object] | None = None
    parent_checkpoint_repository: str | None = None
    parent_checkpoint_revision: str | None = None
    parent_checkpoint_subpath: str | None = None
    parent_checkpoint_artifact_sha256: str | None = None
    runtime_mixture_manifest: str | None = None
    runtime_window_index: str | None = None
    runtime_mounts_descriptor: str | None = None
    runtime_resume_sample_offset: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "base_model_path",
            "dataset_path",
            "modality_config_path",
            "output_dir",
            "experiment_name",
        ):
            _require_nonempty(getattr(self, field_name), field_name)
        if (
            self.experiment_name in {".", ".."}
            or "/" in self.experiment_name
            or "\\" in self.experiment_name
        ):
            raise ValueError("experiment_name must be one safe path component")
        _require_pinned_revision(self.base_model_revision, "base_model_revision")
        _require_pinned_revision(self.dataset_revision, "dataset_revision")
        if self.base_model_revision != MODEL_REVISION:
            raise ValueError("base_model_revision must equal the pinned GR00T N1.7 revision")
        if self.num_gpus not in {1, 4}:
            raise ValueError("exactly one GPU or the explicit four-GPU profile is required")
        if not isinstance(self.physical_batch_size, int) or self.physical_batch_size <= 0:
            raise ValueError("physical_batch_size must be positive")
        resolved_global_batch = (
            self.physical_batch_size * self.num_gpus * self.gradient_accumulation_steps
            if self.global_batch_size is None
            else self.global_batch_size
        )
        if self.gradient_accumulation_steps != 1:
            raise ValueError("gradient accumulation must be exactly 1")
        expected_global_batch = (
            self.physical_batch_size * self.num_gpus * self.gradient_accumulation_steps
        )
        if resolved_global_batch != expected_global_batch:
            raise ValueError(
                "global batch must equal physical batch per-device times world size times gradient accumulation"
            )
        if self.num_gpus == 4 and self.physical_batch_size != 1:
            raise ValueError("four-GPU profile requires per-device batch 1")
        object.__setattr__(self, "global_batch_size", resolved_global_batch)
        if self.model_action_chunk_capacity != 40:
            raise ValueError("model action chunk capacity must be exactly 40")
        if self.training_action_horizon != ACTION_HORIZON:
            raise ValueError("training action horizon must be exactly 16")
        parent_fields = (
            self.parent_checkpoint_repository,
            self.parent_checkpoint_revision,
            self.parent_checkpoint_subpath,
            self.parent_checkpoint_artifact_sha256,
        )
        has_parent = any(value is not None for value in parent_fields)
        if has_parent and not all(value is not None for value in parent_fields):
            raise ValueError("parent checkpoint identity must be complete")
        if has_parent:
            assert self.parent_checkpoint_repository is not None
            assert self.parent_checkpoint_revision is not None
            assert self.parent_checkpoint_subpath is not None
            assert self.parent_checkpoint_artifact_sha256 is not None
            if (
                not self.parent_checkpoint_repository
                or any(character.isspace() for character in self.parent_checkpoint_repository)
            ):
                raise ValueError("parent checkpoint repository is invalid")
            _require_pinned_revision(
                self.parent_checkpoint_revision, "parent checkpoint revision"
            )
            subpath = PurePosixPath(self.parent_checkpoint_subpath)
            if (
                subpath.is_absolute()
                or ".." in subpath.parts
                or "." in subpath.parts
                or not self.parent_checkpoint_subpath
                or "\\" in self.parent_checkpoint_subpath
            ):
                raise ValueError("parent checkpoint subpath is invalid")
            if not _SHA256.fullmatch(self.parent_checkpoint_artifact_sha256):
                raise ValueError("parent checkpoint artifact SHA-256 is invalid")
            if not Path(self.base_model_path).is_absolute():
                raise ValueError("parent checkpoint base_model_path must be absolute")
        runtime_fields = (
            self.runtime_mixture_manifest,
            self.runtime_window_index,
            self.runtime_mounts_descriptor,
        )
        if any(field is not None for field in runtime_fields) and not all(field is not None for field in runtime_fields):
            raise ValueError("runtime mixture manifest, window index, and mounts descriptor must be complete")
        if all(field is not None for field in runtime_fields):
            for field_name, field in zip(("runtime_mixture_manifest", "runtime_window_index", "runtime_mounts_descriptor"), runtime_fields, strict=True):
                if type(field) is not str or not Path(field).is_absolute():
                    raise ValueError(f"{field_name} must be an absolute path")
        if type(self.runtime_resume_sample_offset) is not int or self.runtime_resume_sample_offset < 0:
            raise ValueError("runtime_resume_sample_offset must be nonnegative")
        if not self.tune_projector:
            raise ValueError("tune_projector must be true")
        if not self.tune_diffusion_model:
            raise ValueError("tune_diffusion_model must be true")
        if self.tune_llm:
            raise ValueError("tune_llm must be false")
        if self.tune_visual:
            raise ValueError("tune_visual must be false")
        for field_name in ("max_steps", "save_steps", "dataloader_num_workers", "save_total_limit"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be positive")
        for field_name in ("learning_rate", "weight_decay"):
            value = getattr(self, field_name)
            if not isinstance(value, (float, int)) or value <= 0:
                raise ValueError(f"{field_name} must be positive")
        if not isinstance(self.warmup_ratio, (float, int)) or not 0 < self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be a fraction strictly between zero and one")
        profile = resolve_augmentation_profile(
            self.augmentation_profile, receipt=self.augmentation_receipt
        )
        canonical_receipt = validated_augmentation_receipt(
            profile.name, self.augmentation_receipt
        )
        object.__setattr__(self, "augmentation_profile", profile.name)
        object.__setattr__(self, "augmentation_receipt", canonical_receipt)

    def identity(self) -> dict[str, object]:
        """Return command-relevant provenance without secret environment data."""

        return {
            "base_model_path": self.base_model_path,
            "base_model_revision": self.base_model_revision,
            "dataset_path": self.dataset_path,
            "dataset_revision": self.dataset_revision,
            "modality_config_path": self.modality_config_path,
            "output_dir": self.output_dir,
            "experiment_name": self.experiment_name,
            "physical_batch_size": self.physical_batch_size,
            "global_batch_size": self.global_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "model_action_chunk_capacity": self.model_action_chunk_capacity,
            "training_action_horizon": self.training_action_horizon,
            "warmup_ratio": float(self.warmup_ratio),
            "lr_scheduler_type": LR_SCHEDULER_TYPE,
            "decay_semantics": DECAY_SEMANTICS,
            "tune_llm": self.tune_llm,
            "tune_visual": self.tune_visual,
            "tune_projector": self.tune_projector,
            "tune_diffusion_model": self.tune_diffusion_model,
            "max_steps": self.max_steps,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "dataloader_num_workers": self.dataloader_num_workers,
            "num_gpus": self.num_gpus,
            "augmentation_profile": self.augmentation_profile,
            "augmentation_profile_sha256": resolve_augmentation_profile(
                self.augmentation_profile, receipt=self.augmentation_receipt
            ).sha256,
            "augmentation_receipt": self.augmentation_receipt,
            "parent_checkpoint_repository": self.parent_checkpoint_repository,
            "parent_checkpoint_revision": self.parent_checkpoint_revision,
            "parent_checkpoint_subpath": self.parent_checkpoint_subpath,
            "parent_checkpoint_artifact_sha256": self.parent_checkpoint_artifact_sha256,
            "runtime_mixture_manifest": self.runtime_mixture_manifest,
            "runtime_window_index": self.runtime_window_index,
            "runtime_mounts_descriptor": self.runtime_mounts_descriptor,
            "runtime_resume_sample_offset": self.runtime_resume_sample_offset,
        }

    def sample_presentations_for_optimizer_steps(self, optimizer_steps: int) -> int:
        """Return global samples consumed by a whole-number optimizer-step count."""

        if type(optimizer_steps) is not int or optimizer_steps < 0:
            raise ValueError("optimizer steps must be nonnegative")
        assert self.global_batch_size is not None
        return optimizer_steps * self.global_batch_size
