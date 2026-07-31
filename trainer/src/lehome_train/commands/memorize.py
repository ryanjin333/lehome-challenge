"""Fixed-budget orchestration for offline one-episode memorization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from lehome_train.models import MemorizationResult
from lehome_train.offline_eval import OfflineEvaluation, every_dimension_improved


PHYSICAL_BATCH_SIZE = 1
EVALUATION_INTERVAL = 500
CHECKPOINT_INTERVAL = 1_000
MAX_SAMPLE_PRESENTATIONS = 10_000
_REQUIRED_CONSECUTIVE_EVALUATIONS = 2
_MSE_RATIO_THRESHOLD = 0.1


class ChunkTrainer(Protocol):
    """Injected persistent trainer; production implementations may use GR00T."""

    def __call__(
        self,
        *,
        episode_id: str,
        optimizer_steps: int,
        physical_batch_size: int,
    ) -> None: ...


class EpisodeEvaluator(Protocol):
    """Injected evaluator for the current in-memory/checkpointed policy."""

    def __call__(
        self,
        *,
        episode_id: str,
        sample_presentations: int,
    ) -> OfflineEvaluation: ...


class Checkpointer(Protocol):
    """Injected checkpoint operation for the disposable experiment."""

    def __call__(self, *, episode_id: str, sample_presentations: int) -> None: ...


def _numeric_episode_id(value: str) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("prepared episode IDs must be canonical integers") from error
    if str(normalized) != value or normalized < 0:
        raise ValueError("prepared episode IDs must be canonical integers")
    return normalized


def _prepared_split(dataset: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    try:
        manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("prepared dataset manifest is malformed") from error
    if not isinstance(manifest, dict) or manifest.get("output_format") != "groot_lerobot_v2.1_per_episode":
        raise ValueError("prepared dataset manifest has an unsupported format")
    train = manifest.get("train_episode_ids")
    validation = manifest.get("validation_episode_ids")
    if (
        not isinstance(train, list)
        or not train
        or not all(type(value) is str for value in train)
        or not isinstance(validation, list)
        or not all(type(value) is str for value in validation)
    ):
        raise ValueError("prepared dataset manifest has an invalid episode split")
    train_tuple = tuple(train)
    validation_tuple = tuple(validation)
    for value in train_tuple + validation_tuple:
        _numeric_episode_id(value)
    if (
        len(set(train_tuple)) != len(train_tuple)
        or len(set(validation_tuple)) != len(validation_tuple)
        or set(train_tuple).intersection(validation_tuple)
    ):
        raise ValueError("prepared dataset manifest has an invalid episode split")
    return train_tuple, validation_tuple


def select_training_episode(
    dataset_path: str | Path,
    *,
    requested_episode_id: str | None = None,
) -> str:
    """Select the lowest numeric training ID, or validate an explicit choice."""

    train, _validation = _prepared_split(Path(dataset_path))
    if requested_episode_id is None:
        return min(train, key=_numeric_episode_id)
    if type(requested_episode_id) is not str or requested_episode_id not in train:
        raise ValueError("memorization episode must belong to the training split")
    return requested_episode_id


def _evaluate(
    evaluator: EpisodeEvaluator,
    *,
    episode_id: str,
    sample_presentations: int,
) -> OfflineEvaluation:
    evaluation = evaluator(
        episode_id=episode_id,
        sample_presentations=sample_presentations,
    )
    if not isinstance(evaluation, OfflineEvaluation):
        raise TypeError("memorization evaluator must return OfflineEvaluation")
    return evaluation


def run_memorization(
    *,
    dataset_path: str | Path,
    experiment_id: str,
    experiment_config_sha256: str,
    dataset_manifest_sha256: str,
    trainer: ChunkTrainer,
    evaluator: EpisodeEvaluator,
    checkpointer: Checkpointer,
    requested_episode_id: str | None = None,
) -> MemorizationResult:
    """Run the non-extendable one-episode diagnostic and return its gate result.

    The injected trainer is one persistent disposable experiment.  Each call
    presents the selected episode for exactly 500 physical-batch-one optimizer
    steps, allowing this function to own and enforce all cadence and budget
    decisions independently of the CUDA runtime.
    """

    episode_id = select_training_episode(
        dataset_path,
        requested_episode_id=requested_episode_id,
    )
    initialized = _evaluate(
        evaluator,
        episode_id=episode_id,
        sample_presentations=0,
    )
    final = initialized
    consecutive_qualifying = 0
    sample_presentations = 0
    offline_gate_passed = False

    while sample_presentations < MAX_SAMPLE_PRESENTATIONS:
        trainer(
            episode_id=episode_id,
            optimizer_steps=EVALUATION_INTERVAL,
            physical_batch_size=PHYSICAL_BATCH_SIZE,
        )
        sample_presentations += EVALUATION_INTERVAL
        if sample_presentations % CHECKPOINT_INTERVAL == 0:
            checkpointer(
                episode_id=episode_id,
                sample_presentations=sample_presentations,
            )
        final = _evaluate(
            evaluator,
            episode_id=episode_id,
            sample_presentations=sample_presentations,
        )
        if (
            final.frame_count != initialized.frame_count
            or final.action_dimension != initialized.action_dimension
        ):
            raise ValueError("memorization evaluation shape changed during training")
        qualifies = (
            final.normalized_mse <= initialized.normalized_mse * _MSE_RATIO_THRESHOLD
            and every_dimension_improved(initialized, final)
        )
        consecutive_qualifying = consecutive_qualifying + 1 if qualifies else 0
        if consecutive_qualifying == _REQUIRED_CONSECUTIVE_EVALUATIONS:
            offline_gate_passed = True
            break

    return MemorizationResult(
        experiment_id=experiment_id,
        experiment_config_sha256=experiment_config_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
        episode_id=episode_id,
        initialized_normalized_mse=initialized.normalized_mse,
        final_normalized_mse=final.normalized_mse,
        initialized_dimension_mse=initialized.dimension_mse,
        final_dimension_mse=final.dimension_mse,
        sample_presentations=sample_presentations,
        offline_gate_passed=offline_gate_passed,
        promotable=False,
        pending_gate="simulator_expert_replay",
    )
