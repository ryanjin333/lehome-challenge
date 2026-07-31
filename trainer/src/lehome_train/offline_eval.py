"""CPU-safe offline action replay metrics for memorization diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True, slots=True)
class OfflineEvaluation:
    """Finite normalized action errors for one temporally aligned episode."""

    normalized_mse: float
    dimension_mse: tuple[float, ...]
    frame_count: int
    action_dimension: int

    def __post_init__(self) -> None:
        if (
            type(self.normalized_mse) not in (int, float)
            or not math.isfinite(float(self.normalized_mse))
            or self.normalized_mse < 0
        ):
            raise ValueError("normalized MSE must be finite and nonnegative")
        object.__setattr__(self, "normalized_mse", float(self.normalized_mse))
        if type(self.frame_count) is not int or self.frame_count <= 0:
            raise ValueError("frame count must be positive")
        if type(self.action_dimension) is not int or self.action_dimension <= 0:
            raise ValueError("action dimension must be positive")
        if len(self.dimension_mse) != self.action_dimension:
            raise ValueError("dimension MSE shape must match action dimension")
        normalized_dimensions: list[float] = []
        for value in self.dimension_mse:
            if (
                type(value) not in (int, float)
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError("dimension MSE values must be finite and nonnegative")
            normalized_dimensions.append(float(value))
        object.__setattr__(self, "dimension_mse", tuple(normalized_dimensions))


def _finite_vector(
    values: Sequence[object],
    *,
    expected_dimension: int,
    description: str,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or len(values) != expected_dimension:
        raise ValueError(f"{description} shape is invalid")
    output: list[float] = []
    for value in values:
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise ValueError(f"{description} values must be finite")
        output.append(float(value))
    return tuple(output)


def _action_rows(
    values: Sequence[Sequence[object]],
    *,
    frame_count: int,
    action_dimension: int,
    description: str,
) -> tuple[tuple[float, ...], ...]:
    if isinstance(values, (str, bytes)) or len(values) != frame_count:
        raise ValueError(f"{description} shape is invalid")
    return tuple(
        _finite_vector(
            row,
            expected_dimension=action_dimension,
            description=description,
        )
        for row in values
    )


def evaluate_action_predictions(
    *,
    predicted_actions: Sequence[Sequence[object]],
    expert_actions: Sequence[Sequence[object]],
    normalization_scale: Sequence[object],
    action_min: Sequence[object],
    action_max: Sequence[object],
    prediction_frame_indices: Sequence[int],
    expert_frame_indices: Sequence[int],
) -> OfflineEvaluation:
    """Validate one replay and compute normalized per-dimension action MSE.

    Frame indices are explicit so an otherwise plausible one-frame shift cannot
    silently pass the diagnostic.  Invalid shape, range, or finite-value facts
    fail closed instead of producing a partial metric.
    """

    if not expert_actions or isinstance(expert_actions, (str, bytes)):
        raise ValueError("expert action shape is invalid")
    first_row = expert_actions[0]
    if isinstance(first_row, (str, bytes)) or not first_row:
        raise ValueError("expert action shape is invalid")
    frame_count = len(expert_actions)
    action_dimension = len(first_row)
    expert = _action_rows(
        expert_actions,
        frame_count=frame_count,
        action_dimension=action_dimension,
        description="expert action",
    )
    predicted = _action_rows(
        predicted_actions,
        frame_count=frame_count,
        action_dimension=action_dimension,
        description="predicted action",
    )
    scale = _finite_vector(
        normalization_scale,
        expected_dimension=action_dimension,
        description="normalization scale",
    )
    if any(value <= 0 for value in scale):
        raise ValueError("normalization scale must be positive")
    minimum = _finite_vector(
        action_min,
        expected_dimension=action_dimension,
        description="action range minimum",
    )
    maximum = _finite_vector(
        action_max,
        expected_dimension=action_dimension,
        description="action range maximum",
    )
    if any(lower > upper for lower, upper in zip(minimum, maximum, strict=True)):
        raise ValueError("action range minimum exceeds maximum")
    if (
        len(prediction_frame_indices) != frame_count
        or len(expert_frame_indices) != frame_count
        or tuple(prediction_frame_indices) != tuple(expert_frame_indices)
    ):
        raise ValueError("prediction replay is not temporally aligned")
    for rows in (expert, predicted):
        if any(
            value < minimum[dimension] or value > maximum[dimension]
            for row in rows
            for dimension, value in enumerate(row)
        ):
            raise ValueError("action value is outside the declared range")

    squared_error = [0.0] * action_dimension
    for prediction, target in zip(predicted, expert, strict=True):
        for dimension in range(action_dimension):
            error = (prediction[dimension] - target[dimension]) / scale[dimension]
            squared_error[dimension] += error * error
    dimension_mse = tuple(value / frame_count for value in squared_error)
    return OfflineEvaluation(
        normalized_mse=sum(dimension_mse) / action_dimension,
        dimension_mse=dimension_mse,
        frame_count=frame_count,
        action_dimension=action_dimension,
    )


def every_dimension_improved(
    initialized: OfflineEvaluation,
    candidate: OfflineEvaluation,
) -> bool:
    """Return whether comparable candidate error strictly improves every axis."""

    if (
        initialized.frame_count != candidate.frame_count
        or initialized.action_dimension != candidate.action_dimension
    ):
        return False
    return all(
        final < initial
        for initial, final in zip(
            initialized.dimension_mse,
            candidate.dimension_mse,
            strict=True,
        )
    )
