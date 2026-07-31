"""Loss and throughput extraction from official GR00T trainer output."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import math
import re
from typing import Iterable, Literal


_MAPPING = re.compile(r"\{.*\}")
_CHECKPOINT = re.compile(r"Saving (?:model )?checkpoint to (?P<path>\S+)")
_NUMBER = r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf|-inf)"


@dataclass(frozen=True, slots=True)
class TrainerMetric:
    """One machine-readable signal without retaining raw log text."""

    kind: Literal["loss", "checkpoint", "throughput"]
    line_number: int
    loss: float | None = None
    finite_loss: bool | None = None
    optimizer_step: int | None = None
    steps_per_second: float | None = None
    samples_per_second: float | None = None
    checkpoint_path: str | None = None


def _float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _mapping_from_line(line: str) -> dict[str, object] | None:
    matched = _MAPPING.search(line)
    if matched is None:
        return None
    try:
        decoded = ast.literal_eval(matched.group(0))
    except (SyntaxError, ValueError):
        return None
    return decoded if isinstance(decoded, dict) else None


def _metric_value(mapping: dict[str, object], *names: str) -> float | None:
    for name in names:
        value = _float(mapping.get(name))
        if value is not None:
            return value
    return None


def _finite_or_none(value: float | None) -> float | None:
    """Do not export unusable throughput as a performance measurement."""

    return value if value is not None and math.isfinite(value) else None


def _nan_aware_loss(line: str) -> float | None:
    matched = re.search(r"['\"]loss['\"]\s*:\s*" + _NUMBER, line, re.IGNORECASE)
    if matched is None:
        return None
    try:
        return float(matched.group("value"))
    except ValueError:
        return None


def parse_trainer_log_lines(lines: Iterable[str]) -> tuple[TrainerMetric, ...]:
    """Extract official Trainer signals while deliberately discarding raw logs."""

    records: list[TrainerMetric] = []
    for line_number, line in enumerate(lines, start=1):
        checkpoint = _CHECKPOINT.search(line)
        if checkpoint is not None:
            records.append(
                TrainerMetric(
                    kind="checkpoint",
                    line_number=line_number,
                    checkpoint_path=checkpoint.group("path"),
                )
            )
            continue
        mapping = _mapping_from_line(line)
        loss = _metric_value(mapping, "loss", "train_loss") if mapping else None
        if loss is None:
            loss = _nan_aware_loss(line)
        step = _metric_value(mapping, "step", "global_step") if mapping else None
        steps_per_second = (
            _metric_value(mapping, "steps_per_second", "train_steps_per_second")
            if mapping
            else None
        )
        samples_per_second = (
            _metric_value(mapping, "samples_per_second", "train_samples_per_second")
            if mapping
            else None
        )
        steps_per_second = _finite_or_none(steps_per_second)
        samples_per_second = _finite_or_none(samples_per_second)
        if loss is not None:
            records.append(
                TrainerMetric(
                    kind="loss",
                    line_number=line_number,
                    loss=loss,
                    finite_loss=math.isfinite(loss),
                    optimizer_step=int(step) if step is not None and step >= 0 else None,
                    steps_per_second=steps_per_second,
                    samples_per_second=samples_per_second,
                )
            )
        elif steps_per_second is not None or samples_per_second is not None:
            records.append(
                TrainerMetric(
                    kind="throughput",
                    line_number=line_number,
                    optimizer_step=int(step) if step is not None and step >= 0 else None,
                    steps_per_second=steps_per_second,
                    samples_per_second=samples_per_second,
                )
            )
    return tuple(records)
