"""Immutable version and repository settings for portable training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


PYTHON_VERSION: Final = "3.10.18"
CUDA_BASE_DIGEST: Final = (
    "sha256:61f6c08f2b59036cb935e56d1e31a6b64e3ae2c7ddb86d33fa0b044c7917b719"
)
ISAAC_GROOT_REVISION: Final = "23ace64f17aa5015259b8609d371eb61a357c776"
MODEL_REVISION: Final = "2fc962b973bccdd5d8ce4f67cc63b264d6886495"
DEFAULT_DATA_REPO: Final = "ryanjin333/lehome-groot-n17-data"
DEFAULT_MODEL_REPO: Final = "ryanjin333/lehome-groot-n17-models"
DEFAULT_ROLLOUT_REPO: Final = "ryanjin333/lehome-groot-n17-rollouts"


@dataclass(frozen=True, slots=True)
class TrainerSettings:
    """Pinned trainer inputs; callers must explicitly supply any override."""

    python_version: str = PYTHON_VERSION
    cuda_base_digest: str = CUDA_BASE_DIGEST
    isaac_groot_revision: str = ISAAC_GROOT_REVISION
    model_revision: str = MODEL_REVISION
    data_repo: str = DEFAULT_DATA_REPO
    model_repo: str = DEFAULT_MODEL_REPO


DEFAULT_SETTINGS: Final = TrainerSettings()
