"""Fixed-exposure optimizer and checkpoint scheduling."""

from __future__ import annotations

from dataclasses import dataclass
import math


TOTAL_SAMPLE_PRESENTATIONS = 768_000
CHECKPOINT_SAMPLE_PRESENTATIONS = 64_000
DEFAULT_WARMUP_FRACTION = 0.05
DEFAULT_PEAK_LEARNING_RATE = 1e-4


def _positive_integer(value: int, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def optimizer_steps_for_presentations(
    sample_presentations: int,
    physical_batch_size: int,
) -> int:
    """Return exact optimizer steps without rounding an exposure budget."""

    presentations = _positive_integer(sample_presentations, "sample presentations")
    batch = _positive_integer(physical_batch_size, "physical batch size")
    steps, remainder = divmod(presentations, batch)
    if remainder:
        raise ValueError("sample presentations must produce integral optimizer steps")
    return steps


def cosine_learning_rate_multiplier(
    optimizer_step: int,
    *,
    total_optimizer_steps: int,
    warmup_fraction: float,
) -> float:
    """Return linear-warmup/cosine-decay LR as a fraction of peak LR."""

    total = _positive_integer(total_optimizer_steps, "total optimizer steps")
    if type(optimizer_step) is not int or not 0 <= optimizer_step <= total:
        raise ValueError("optimizer step must be within the schedule")
    if type(warmup_fraction) not in (int, float) or not 0 < warmup_fraction < 1:
        raise ValueError("warmup_fraction must be strictly between zero and one")

    progress = optimizer_step / total
    warmup = float(warmup_fraction)
    if progress <= warmup:
        return progress / warmup
    decay_progress = (progress - warmup) / (1.0 - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * decay_progress))


@dataclass(frozen=True, slots=True)
class ExposureSchedule:
    """One exact-presentation training schedule for a selected physical batch."""

    physical_batch_size: int
    sample_presentations: int = TOTAL_SAMPLE_PRESENTATIONS
    checkpoint_sample_presentations: int = CHECKPOINT_SAMPLE_PRESENTATIONS
    warmup_fraction: float = DEFAULT_WARMUP_FRACTION

    def __post_init__(self) -> None:
        total_steps = optimizer_steps_for_presentations(
            self.sample_presentations,
            self.physical_batch_size,
        )
        try:
            interval_steps = optimizer_steps_for_presentations(
                self.checkpoint_sample_presentations,
                self.physical_batch_size,
            )
        except ValueError as error:
            raise ValueError(
                "checkpoint interval must produce integral optimizer steps"
            ) from error
        if self.sample_presentations % self.checkpoint_sample_presentations:
            raise ValueError("checkpoint interval must divide total sample presentations")
        if interval_steps > total_steps:
            raise ValueError("checkpoint interval exceeds the training schedule")
        if (
            type(self.warmup_fraction) not in (int, float)
            or not 0 < self.warmup_fraction < 1
        ):
            raise ValueError("warmup_fraction must be strictly between zero and one")

    @property
    def total_optimizer_steps(self) -> int:
        return optimizer_steps_for_presentations(
            self.sample_presentations,
            self.physical_batch_size,
        )

    @property
    def checkpoint_interval_steps(self) -> int:
        try:
            return optimizer_steps_for_presentations(
                self.checkpoint_sample_presentations,
                self.physical_batch_size,
            )
        except ValueError as error:
            raise ValueError(
                "checkpoint interval must produce integral optimizer steps"
            ) from error

    @property
    def checkpoint_steps(self) -> tuple[int, ...]:
        interval = self.checkpoint_interval_steps
        return tuple(range(interval, self.total_optimizer_steps + 1, interval))

    @property
    def warmup_optimizer_steps(self) -> int:
        raw_steps = self.total_optimizer_steps * float(self.warmup_fraction)
        steps = round(raw_steps)
        if not math.isclose(raw_steps, steps, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("warmup fraction must produce integral optimizer steps")
        if not 0 < steps < self.total_optimizer_steps:
            raise ValueError("warmup optimizer steps must be within the schedule")
        return steps

    def identity(self) -> dict[str, object]:
        """Return every immutable input to the canonical LR schedule."""

        return {
            "schema_version": 1,
            "physical_batch_size": self.physical_batch_size,
            "sample_presentations": self.sample_presentations,
            "total_optimizer_steps": self.total_optimizer_steps,
            "checkpoint_sample_presentations": self.checkpoint_sample_presentations,
            "checkpoint_interval_steps": self.checkpoint_interval_steps,
            "warmup_fraction": float(self.warmup_fraction),
            "warmup_optimizer_steps": self.warmup_optimizer_steps,
            "scheduler_type": "cosine",
            "decay_semantics": "cosine_remainder_after_warmup",
            "base_learning_rate": 0.0,
            "peak_learning_rate": DEFAULT_PEAK_LEARNING_RATE,
        }

    @property
    def sha256(self) -> str:
        from lehome_train.io import canonical_json_sha256

        return canonical_json_sha256(self.identity())

    def learning_rate_multiplier(self, optimizer_step: int) -> float:
        return cosine_learning_rate_multiplier(
            optimizer_step,
            total_optimizer_steps=self.total_optimizer_steps,
            warmup_fraction=float(self.warmup_fraction),
        )

    def learning_rate(self, optimizer_step: int) -> float:
        return DEFAULT_PEAK_LEARNING_RATE * self.learning_rate_multiplier(
            optimizer_step
        )
