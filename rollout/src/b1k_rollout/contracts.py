"""The frozen, complete identity required before a B1K rollout starts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from b1k_rollout.identity import (
    BEHAVIOR_REVISION,
    DATASET_REPO,
    GROOT_REVISION,
    MODEL_REPO,
    canonical_json_sha256,
    reject_credential_material,
    require_b1k_repository,
    require_image_digest,
    require_immutable_commit,
    require_sha256,
)


AUTO_DESTROY = "0"
_EVALUATOR_MODES = frozenset(("train", "public_test"))
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_FIELDS = frozenset(
    (
        "behavior_revision",
        "groot_revision",
        "model_repository",
        "model_commit",
        "dataset_repository",
        "image_digest",
        "run_id",
        "cycle_id",
        "campaign_id",
        "evaluator_mode",
        "task_manifest_sha256",
        "checkpoint_artifact_sha256",
        "auto_destroy",
    )
)
_TASK_LIST_FIELDS = frozenset(("tasks", "task_names", "task_list", "task_ids"))


@dataclass(frozen=True, slots=True)
class RolloutContract:
    """A complete, serializable identity for one immutable rollout campaign."""

    behavior_revision: str
    groot_revision: str
    model_repository: str
    model_commit: str
    dataset_repository: str
    image_digest: str
    run_id: str
    cycle_id: str
    campaign_id: str
    evaluator_mode: str
    task_manifest_sha256: str
    checkpoint_artifact_sha256: str
    auto_destroy: str = AUTO_DESTROY

    def __post_init__(self) -> None:
        if self.behavior_revision != BEHAVIOR_REVISION:
            raise ValueError("BEHAVIOR revision must equal the pinned immutable commit")
        if self.groot_revision != GROOT_REVISION:
            raise ValueError("GR00T revision must equal the pinned immutable commit")
        require_b1k_repository(
            self.model_repository, expected=MODEL_REPO, label="model repository"
        )
        require_immutable_commit(self.model_commit, label="model commit")
        require_b1k_repository(
            self.dataset_repository, expected=DATASET_REPO, label="dataset repository"
        )
        require_image_digest(self.image_digest)
        for field in ("run_id", "cycle_id", "campaign_id"):
            _require_identifier(getattr(self, field), field=field)
        if self.evaluator_mode not in _EVALUATOR_MODES:
            raise ValueError("evaluator mode must be train or public_test")
        require_sha256(self.task_manifest_sha256, label="task manifest hash")
        require_sha256(self.checkpoint_artifact_sha256, label="checkpoint artifact hash")
        if self.auto_destroy != AUTO_DESTROY:
            raise ValueError("AUTO_DESTROY must be exactly 0")
        reject_credential_material(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "RolloutContract":
        """Parse a complete contract and reject non-contract configuration."""

        reject_credential_material(value)
        keys = set(value)
        task_fields = keys & _TASK_LIST_FIELDS
        if task_fields:
            raise ValueError("arbitrary task lists are forbidden; use task_manifest_sha256")
        unknown = keys - _FIELDS
        if unknown:
            raise ValueError(f"unknown rollout contract fields: {', '.join(sorted(unknown))}")
        missing = _FIELDS - keys
        if missing:
            raise ValueError(f"missing rollout contract fields: {', '.join(sorted(missing))}")
        fields = {name: value[name] for name in _FIELDS}
        if not all(isinstance(item, str) for item in fields.values()):
            raise ValueError("rollout contract fields must be strings")
        return cls(**fields)  # type: ignore[arg-type]

    @property
    def identity(self) -> str:
        """Content address this frozen contract with canonical JSON."""

        return canonical_json_sha256(self.to_dict())

    def to_dict(self) -> dict[str, str]:
        """Return the exact secret-free payload used for serialization."""

        return asdict(self)


def _require_identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{field} must be a non-empty stable identifier")
    return value
