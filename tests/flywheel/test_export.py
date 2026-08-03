import pytest

from lehome.flywheel.export import build_selection_report, select_expert_windows
from lehome.flywheel.models import ActionSource, EpisodeFrame


def frame(
    step: int,
    source: ActionSource,
    *,
    segment: int = 1,
    age_ms: float | None = None,
) -> EpisodeFrame:
    return EpisodeFrame(
        step,
        step + 1,
        step + 2,
        (0.0,) * 12,
        (float(step),) * 12,
        source,
        0.0,
        False,
        segment,
        expert_sample_age_ms=age_ms,
    )


def test_dagger_exports_only_complete_expert_windows() -> None:
    frames = [frame(i, ActionSource.POLICY if i < 4 else ActionSource.EXPERT) for i in range(24)]

    selected = select_expert_windows(frames, horizon=16, accepted_success=True)

    assert [window.observation_step for window in selected] == [4, 5, 6, 7, 8]
    assert all(len(window.future_actions) == 16 for window in selected)
    assert all(value[0] >= 4 for window in selected for value in window.future_actions)


def test_failed_episode_and_hold_frames_export_nothing() -> None:
    frames = [frame(i, ActionSource.EXPERT) for i in range(20)]

    assert select_expert_windows(frames, horizon=16, accepted_success=False) == ()
    frames[8] = frame(8, ActionSource.HOLD)
    assert select_expert_windows(frames, horizon=16, accepted_success=True) == ()


def test_export_requires_contiguous_steps_and_a_single_expert_segment() -> None:
    broken_steps = [frame(i, ActionSource.EXPERT) for i in range(16)]
    broken_steps[10] = frame(12, ActionSource.EXPERT)
    assert select_expert_windows(broken_steps, horizon=16, accepted_success=True) == ()

    broken_segment = [frame(i, ActionSource.EXPERT, segment=1 if i < 8 else 2) for i in range(16)]
    assert select_expert_windows(broken_segment, horizon=16, accepted_success=True) == ()


def test_selection_report_counts_fail_closed_rejections() -> None:
    frames = [frame(0, ActionSource.POLICY), frame(1, ActionSource.HOLD)]
    frames.extend(frame(index, ActionSource.EXPERT, age_ms=120.0) for index in range(2, 5))
    report = build_selection_report(
        frames,
        horizon=4,
        accepted_success=False,
        holdout=True,
        max_expert_sample_age_ms=100.0,
    )

    assert report.policy == 1
    assert report.hold == 1
    assert report.failed_episode == 1
    assert report.short_tail == 3
    assert report.stale_expert == 3
    assert report.holdout == 1
    assert report.selected == 0
    with pytest.raises(ValueError, match="horizon"):
        select_expert_windows(frames, horizon=0, accepted_success=True)
