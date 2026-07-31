"""Run the exact pinned GR00T launcher with one resumable stop boundary."""

from __future__ import annotations

import argparse
import runpy
import sys
from typing import Any


class StopAtOptimizerStep:
    """Transformers callback that stops after a completed global optimizer step."""

    def __init__(self, optimizer_step: int) -> None:
        if type(optimizer_step) is not int or optimizer_step < 0:
            raise ValueError("stop optimizer step must be nonnegative")
        self.optimizer_step = optimizer_step

    def on_step_end(self, args: object, state: Any, control: Any, **_kwargs: object) -> Any:
        if state.global_step >= self.optimizer_step:
            control.should_log = True
            control.should_save = True
            control.should_training_stop = True
        return control

    def on_train_begin(
        self, args: object, state: Any, control: Any, **_kwargs: object
    ) -> Any:
        if self.optimizer_step == 0:
            control.should_training_stop = True
        return control


def _arguments(argv: list[str] | None) -> tuple[int, str, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stop-after-step", type=int, required=True)
    parser.add_argument("remainder", nargs=argparse.REMAINDER)
    parsed = parser.parse_args(argv)
    remainder = list(parsed.remainder)
    if remainder[:1] == ["--"]:
        remainder.pop(0)
    if not remainder or not remainder[0].endswith("gr00t/experiment/launch_finetune.py"):
        raise ValueError("chunk launcher requires the pinned official entrypoint")
    if parsed.stop_after_step < 0:
        raise ValueError("chunk stop optimizer step must be nonnegative")
    return parsed.stop_after_step, remainder[0], remainder[1:]


def main(argv: list[str] | None = None) -> None:
    stop_step, entrypoint, official_arguments = _arguments(argv)
    try:
        from transformers import Trainer
    except ImportError:
        raise RuntimeError("pinned Transformers runtime is unavailable") from None

    original_train = Trainer.train

    def bounded_train(trainer: Any, *args: object, **kwargs: object) -> Any:
        trainer.add_callback(StopAtOptimizerStep(stop_step))
        return original_train(trainer, *args, **kwargs)

    Trainer.train = bounded_train
    sys.argv = [entrypoint, *official_arguments]
    runpy.run_path(entrypoint, run_name="__main__")


if __name__ == "__main__":
    main()
