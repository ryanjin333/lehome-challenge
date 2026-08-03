"""Fail-closed expert-action selection for behavior-cloning exports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from .models import ActionSource, EpisodeFrame


@dataclass(frozen=True, slots=True)
class ExpertWindow:
    observation_step: int
    future_actions: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class SelectionReport:
    policy: int
    hold: int
    failed_episode: int
    short_tail: int
    stale_expert: int
    holdout: int
    selected: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def _validate_horizon(horizon: int) -> None:
    if not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")


def _is_complete_expert_window(
    frames: Sequence[EpisodeFrame],
    start: int,
    horizon: int,
    max_expert_sample_age_ms: float | None,
) -> bool:
    segment = frames[start : start + horizon]
    if len(segment) != horizon:
        return False
    first = segment[0]
    for frame in segment:
        if frame.action_source is not ActionSource.EXPERT or frame.segment != first.segment:
            return False
        if max_expert_sample_age_ms is not None and (
            frame.expert_sample_age_ms is None or frame.expert_sample_age_ms > max_expert_sample_age_ms
        ):
            return False
    return all(next_frame.step == frame.step + 1 for frame, next_frame in zip(segment, segment[1:]))


def select_expert_windows(
    frames: Sequence[EpisodeFrame],
    *,
    horizon: int,
    accepted_success: bool,
    holdout: bool = False,
    max_expert_sample_age_ms: float | None = None,
) -> tuple[ExpertWindow, ...]:
    """Select only contiguous expert labels from accepted, non-holdout episodes."""
    _validate_horizon(horizon)
    if max_expert_sample_age_ms is not None and max_expert_sample_age_ms < 0:
        raise ValueError("max_expert_sample_age_ms must be non-negative")
    if not accepted_success or holdout:
        return ()
    windows: list[ExpertWindow] = []
    for start in range(len(frames)):
        if not _is_complete_expert_window(frames, start, horizon, max_expert_sample_age_ms):
            continue
        segment = frames[start : start + horizon]
        windows.append(ExpertWindow(segment[0].step, tuple(frame.action for frame in segment)))
    return tuple(windows)


def build_selection_report(
    frames: Sequence[EpisodeFrame],
    *,
    horizon: int,
    accepted_success: bool,
    holdout: bool = False,
    max_expert_sample_age_ms: float | None = None,
) -> SelectionReport:
    """Summarize accepted and rejected labels without changing the raw episode."""
    _validate_horizon(horizon)
    if max_expert_sample_age_ms is not None and max_expert_sample_age_ms < 0:
        raise ValueError("max_expert_sample_age_ms must be non-negative")
    policy = sum(frame.action_source is ActionSource.POLICY for frame in frames)
    hold = sum(frame.action_source is ActionSource.HOLD for frame in frames)
    stale_expert = sum(
        frame.action_source is ActionSource.EXPERT
        and max_expert_sample_age_ms is not None
        and (frame.expert_sample_age_ms is None or frame.expert_sample_age_ms > max_expert_sample_age_ms)
        for frame in frames
    )
    short_tail = sum(
        frame.action_source is ActionSource.EXPERT and len(frames) - index < horizon
        for index, frame in enumerate(frames)
    )
    selected = len(
        select_expert_windows(
            frames,
            horizon=horizon,
            accepted_success=accepted_success,
            holdout=holdout,
            max_expert_sample_age_ms=max_expert_sample_age_ms,
        )
    )
    return SelectionReport(
        policy=policy,
        hold=hold,
        failed_episode=int(not accepted_success),
        short_tail=short_tail,
        stale_expert=stale_expert,
        holdout=int(holdout),
        selected=selected,
    )
