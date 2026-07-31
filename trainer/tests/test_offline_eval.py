from __future__ import annotations

import math

import pytest

from lehome_train.offline_eval import (
    OfflineEvaluation,
    evaluate_action_predictions,
    every_dimension_improved,
)


def test_normalized_mse_is_computed_per_dimension_and_across_the_episode() -> None:
    evaluation = evaluate_action_predictions(
        predicted_actions=((2.0, 4.0), (2.0, 2.0)),
        expert_actions=((1.0, 2.0), (2.0, 4.0)),
        normalization_scale=(1.0, 2.0),
        action_min=(0.0, 0.0),
        action_max=(4.0, 4.0),
        prediction_frame_indices=(5, 6),
        expert_frame_indices=(5, 6),
    )

    assert evaluation == OfflineEvaluation(
        normalized_mse=0.75,
        dimension_mse=(0.5, 1.0),
        frame_count=2,
        action_dimension=2,
    )
    assert math.isfinite(evaluation.normalized_mse)
    assert all(math.isfinite(value) for value in evaluation.dimension_mse)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"predicted_actions": ((1.0, 2.0),)}, "shape"),
        ({"predicted_actions": ((1.0,), (2.0,))}, "shape"),
        ({"predicted_actions": ((1.0, float("nan")), (2.0, 4.0))}, "finite"),
        ({"normalization_scale": (1.0, 0.0)}, "normalization"),
        ({"predicted_actions": ((1.0, 5.0), (2.0, 4.0))}, "range"),
        ({"prediction_frame_indices": (5, 7)}, "temporal"),
        ({"prediction_frame_indices": (5, 5)}, "frame indices"),
        ({"prediction_frame_indices": (6, 5)}, "frame indices"),
        ({"prediction_frame_indices": (5.0, 6)}, "frame indices"),
        ({"prediction_frame_indices": (True, 6)}, "frame indices"),
        (
            {
                "prediction_frame_indices": (-1, 6),
                "expert_frame_indices": (-1, 6),
            },
            "frame indices",
        ),
    ],
)
def test_offline_evaluation_fails_closed_on_invalid_replay(
    overrides: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "predicted_actions": ((1.0, 2.0), (2.0, 4.0)),
        "expert_actions": ((1.0, 2.0), (2.0, 4.0)),
        "normalization_scale": (1.0, 2.0),
        "action_min": (0.0, 0.0),
        "action_max": (4.0, 4.0),
        "prediction_frame_indices": (5, 6),
        "expert_frame_indices": (5, 6),
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        evaluate_action_predictions(**values)  # type: ignore[arg-type]


def test_every_action_dimension_must_improve_strictly() -> None:
    initialized = OfflineEvaluation(1.0, (0.01, 1.99), 4, 2)

    assert every_dimension_improved(
        initialized,
        OfflineEvaluation(0.01, (0.005, 0.015), 4, 2),
    )
    assert not every_dimension_improved(
        initialized,
        OfflineEvaluation(0.01, (0.01, 0.01), 4, 2),
    )
    assert not every_dimension_improved(
        initialized,
        OfflineEvaluation(0.01, (0.005, 0.015, 0.01), 4, 3),
    )


def test_evaluation_rejects_aggregate_mse_that_contradicts_dimensions() -> None:
    with pytest.raises(ValueError, match="aggregate"):
        OfflineEvaluation(
            normalized_mse=0.5,
            dimension_mse=(0.1, 0.3),
            frame_count=4,
            action_dimension=2,
        )


def test_temporal_alignment_allows_strictly_increasing_gapped_subsets() -> None:
    evaluation = evaluate_action_predictions(
        predicted_actions=((1.0, 2.0), (2.0, 4.0)),
        expert_actions=((1.0, 2.0), (2.0, 4.0)),
        normalization_scale=(1.0, 2.0),
        action_min=(0.0, 0.0),
        action_max=(4.0, 4.0),
        prediction_frame_indices=(2, 5),
        expert_frame_indices=(2, 5),
    )

    assert evaluation.normalized_mse == 0.0
