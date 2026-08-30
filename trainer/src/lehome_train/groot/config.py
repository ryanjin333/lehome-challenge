"""Immutable, fail-closed inputs for the pinned GR00T N1.7 launcher."""

from __future__ import annotations

import re
from dataclasses import dataclass

from lehome_train.constants import MODEL_REVISION


ACTION_HORIZON = 16
LR_SCHEDULER_TYPE = "cosine"
DECAY_SEMANTICS = "cosine_remainder_after_warmup"
_PINNED_REVISION = re.compile(r"^[0-9a-f]{40}$")


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
    action_horizon: int = ACTION_HORIZON
    tune_llm: bool = False
    tune_visual: bool = False
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    dataloader_num_workers: int = 4
    save_total_limit: int = 5

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
        if self.num_gpus != 1:
            raise ValueError("exactly one GPU is required")
        if not isinstance(self.physical_batch_size, int) or self.physical_batch_size <= 0:
            raise ValueError("physical_batch_size must be positive")
        resolved_global_batch = (
            self.physical_batch_size
            if self.global_batch_size is None
            else self.global_batch_size
        )
        if resolved_global_batch != self.physical_batch_size:
            raise ValueError("global batch must equal physical batch")
        object.__setattr__(self, "global_batch_size", resolved_global_batch)
        if self.gradient_accumulation_steps != 1:
            raise ValueError("gradient accumulation must be exactly 1")
        if self.action_horizon != ACTION_HORIZON:
            raise ValueError("action horizon must be exactly 16")
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
            "action_horizon": self.action_horizon,
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
        }
