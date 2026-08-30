"""Immutable, secret-free contracts for BEHAVIOR-1K rollout campaigns."""

from b1k_rollout.contracts import RolloutContract
from b1k_rollout.identity import (
    BEHAVIOR_REVISION,
    DATASET_REPO,
    GROOT_REVISION,
    MODEL_REPO,
)

__all__ = [
    "BEHAVIOR_REVISION",
    "DATASET_REPO",
    "GROOT_REVISION",
    "MODEL_REPO",
    "RolloutContract",
]
