from __future__ import annotations

import json
from pathlib import Path

import pytest

from lehome_train.commands.memorize import (
    CHECKPOINT_INTERVAL,
    EVALUATION_INTERVAL,
    MAX_SAMPLE_PRESENTATIONS,
    PHYSICAL_BATCH_SIZE,
    run_memorization,
    select_training_episode,
)
from lehome_train.offline_eval import OfflineEvaluation


SHA_A = "a" * 64
SHA_B = "b" * 64


def _dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "prepared"
    dataset.mkdir()
    (dataset / "manifest.json").write_text(
        json.dumps(
            {
                "output_format": "groot_lerobot_v2.1_per_episode",
                "train_episode_ids": ["11", "3"],
                "validation_episode_ids": ["7"],
            }
        ),
        encoding="utf-8",
    )
    return dataset


def _evaluation(normalized_mse: float, dimensions: tuple[float, ...]) -> OfflineEvaluation:
    return OfflineEvaluation(
        normalized_mse=normalized_mse,
        dimension_mse=dimensions,
        frame_count=4,
        action_dimension=len(dimensions),
    )


def test_episode_selection_is_deterministic_and_restricted_to_training_split(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)

    assert select_training_episode(dataset) == "3"
    assert select_training_episode(dataset, requested_episode_id="11") == "11"
    with pytest.raises(ValueError, match="training split"):
        select_training_episode(dataset, requested_episode_id="7")


def test_failed_gate_uses_the_fixed_budget_and_exact_cadences(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    training_calls: list[tuple[str, int, int]] = []
    evaluation_calls: list[tuple[str, int]] = []
    checkpoint_calls: list[tuple[str, int]] = []

    def trainer(*, episode_id: str, optimizer_steps: int, physical_batch_size: int) -> None:
        training_calls.append((episode_id, optimizer_steps, physical_batch_size))

    def evaluator(*, episode_id: str, sample_presentations: int) -> OfflineEvaluation:
        evaluation_calls.append((episode_id, sample_presentations))
        return _evaluation(1.0 if sample_presentations == 0 else 0.2, (0.2, 0.2))

    result = run_memorization(
        dataset_path=dataset,
        experiment_id="memorize-001",
        experiment_config_sha256=SHA_A,
        dataset_manifest_sha256=SHA_B,
        trainer=trainer,
        evaluator=evaluator,
        checkpointer=lambda *, episode_id, sample_presentations: checkpoint_calls.append(
            (episode_id, sample_presentations)
        ),
    )

    assert PHYSICAL_BATCH_SIZE == 1
    assert EVALUATION_INTERVAL == 500
    assert CHECKPOINT_INTERVAL == 1_000
    assert MAX_SAMPLE_PRESENTATIONS == 10_000
    assert training_calls == [
        ("3", EVALUATION_INTERVAL, PHYSICAL_BATCH_SIZE)
        for _ in range(MAX_SAMPLE_PRESENTATIONS // EVALUATION_INTERVAL)
    ]
    assert evaluation_calls == [
        ("3", step)
        for step in range(0, MAX_SAMPLE_PRESENTATIONS + 1, EVALUATION_INTERVAL)
    ]
    assert checkpoint_calls == [
        ("3", step)
        for step in range(CHECKPOINT_INTERVAL, MAX_SAMPLE_PRESENTATIONS + 1, CHECKPOINT_INTERVAL)
    ]
    assert result.sample_presentations == MAX_SAMPLE_PRESENTATIONS
    assert result.initialized_normalized_mse == 1.0
    assert result.final_normalized_mse == 0.2
    assert result.offline_gate_passed is False
    assert result.promotable is False
    assert result.pending_gate == "simulator_expert_replay"


def test_early_stopping_requires_two_consecutive_qualifying_evaluations(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    evaluations = {
        0: _evaluation(1.0, (0.01, 1.99)),
        500: _evaluation(0.01, (0.005, 0.015)),
        1_000: _evaluation(0.01, (0.01, 0.01)),
        1_500: _evaluation(0.01, (0.005, 0.015)),
        2_000: _evaluation(0.01, (0.004, 0.016)),
    }
    trained_steps: list[int] = []
    checkpoints: list[int] = []

    result = run_memorization(
        dataset_path=dataset,
        requested_episode_id="11",
        experiment_id="memorize-002",
        experiment_config_sha256=SHA_A,
        dataset_manifest_sha256=SHA_B,
        trainer=lambda **values: trained_steps.append(values["optimizer_steps"]),
        evaluator=lambda **values: evaluations[values["sample_presentations"]],
        checkpointer=lambda **values: checkpoints.append(values["sample_presentations"]),
    )

    assert trained_steps == [500, 500, 500, 500]
    assert checkpoints == [1_000, 2_000]
    assert result.episode_id == "11"
    assert result.sample_presentations == 2_000
    assert result.initialized_dimension_mse == (0.01, 1.99)
    assert result.final_dimension_mse == (0.004, 0.016)
    assert result.final_normalized_mse <= result.initialized_normalized_mse * 0.1
    assert all(
        final < initialized
        for initialized, final in zip(
            result.initialized_dimension_mse,
            result.final_dimension_mse,
            strict=True,
        )
    )
    assert result.offline_gate_passed is True
    assert result.promotable is False
    assert result.pending_gate == "simulator_expert_replay"
