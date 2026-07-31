"""Deterministic episode-level train/validation splitting."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EpisodeSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]


def _episode_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def split_episode_ids(
    episode_ids: list[str] | tuple[str, ...],
    *,
    seed: int,
    validation_fraction: float,
) -> EpisodeSplit:
    """Split stable IDs as indivisible units using a fixed local PRNG."""

    if type(seed) is not int:
        raise TypeError("split seed must be an integer")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    normalized = [str(value) for value in episode_ids]
    if len(set(normalized)) != len(normalized):
        raise ValueError("episode IDs must be unique")
    ordered = sorted(normalized, key=_episode_sort_key)
    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    validation_count = round(len(shuffled) * validation_fraction)
    if validation_fraction > 0 and len(shuffled) > 1:
        validation_count = max(1, validation_count)
    validation_count = min(validation_count, max(0, len(shuffled) - 1))
    validation = set(shuffled[:validation_count])
    return EpisodeSplit(
        train=tuple(value for value in ordered if value not in validation),
        validation=tuple(value for value in ordered if value in validation),
    )
