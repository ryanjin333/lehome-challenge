from __future__ import annotations

from types import SimpleNamespace

from lehome_train.groot.chunk_launch import StopAtOptimizerStep


def test_stop_callback_stops_only_at_requested_global_step() -> None:
    callback = StopAtOptimizerStep(1_000)
    control = SimpleNamespace(
        should_training_stop=False,
        should_save=False,
        should_log=False,
    )

    returned = callback.on_step_end(
        args=object(),
        state=SimpleNamespace(global_step=999),
        control=control,
    )
    assert returned.should_training_stop is False

    returned = callback.on_step_end(
        args=object(),
        state=SimpleNamespace(global_step=1_000),
        control=control,
    )
    assert returned.should_training_stop is True
    assert returned.should_save is True
    assert returned.should_log is True


def test_zero_step_callback_stops_before_any_optimizer_step() -> None:
    callback = StopAtOptimizerStep(0)
    control = SimpleNamespace(should_training_stop=False)

    returned = callback.on_train_begin(
        args=object(),
        state=SimpleNamespace(global_step=0),
        control=control,
    )

    assert returned.should_training_stop is True
