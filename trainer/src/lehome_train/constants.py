"""Immutable version and repository settings for portable training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


PYTHON_VERSION: Final = "3.10.18"
CUDA_BASE_DIGEST: Final = (
    "sha256:61f6c08f2b59036cb935e56d1e31a6b64e3ae2c7ddb86d33fa0b044c7917b719"
)
ISAAC_GROOT_REPOSITORY: Final = "https://github.com/wensi-ai/Isaac-GR00T.git"
ISAAC_GROOT_REVISION: Final = "ace36d935b376fbf25cd56371e23877b95407c40"
BEHAVIOR_1K_DATASET_REPOSITORY: Final = "behavior-1k/2026-challenge-demos"
BEHAVIOR_1K_DATASET_REVISION: Final = "4f50b44796641a4d526a19d9aeadc8aa51e2f2c2"
BEHAVIOR_1K_FINAL_MODEL_REPOSITORY: Final = "ryanjin333/behavior1k-groot-n17-models"
BEHAVIOR_1K_CHECKPOINT_BUCKET: Final = "ryanjin333/behavior1k-groot-n17-checkpoints"
BEHAVIOR_1K_TRAINER_IMAGE_REPOSITORY: Final = "docker.io/ryanjin333/behavior1k-groot-n17-trainer"
COSMOS_REPOSITORY: Final = "nvidia/Cosmos-Reason2-2B"
COSMOS_REVISION: Final = "9ce19a195e423419c349abfc86fd07178b230561"
MODEL_REVISION: Final = "2fc962b973bccdd5d8ce4f67cc63b264d6886495"
DEFAULT_DATA_REPO: Final = "ryanjin333/lehome-groot-n17-data"
DEFAULT_MODEL_REPO: Final = "ryanjin333/lehome-groot-n17-models"


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
