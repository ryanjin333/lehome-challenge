"""Run the exact pinned GR00T launcher with one resumable stop boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys
from typing import Any

try:
    from transformers import TrainerCallback
except ImportError:  # Host-side request validation does not install CUDA deps.
    class TrainerCallback:  # type: ignore[no-redef]
        pass


class StopAtOptimizerStep(TrainerCallback):
    """Transformers callback that stops after a completed global optimizer step."""

    def __init__(self, optimizer_step: int) -> None:
        if type(optimizer_step) is not int or optimizer_step < 0:
            raise ValueError("stop optimizer step must be nonnegative")
        self.optimizer_step = optimizer_step

    def on_train_begin(
        self, args: object, state: Any, control: Any, **_kwargs: object
    ) -> Any:
        """Satisfy the current Transformers callback lifecycle contract."""

        return control

    def on_step_end(self, args: object, state: Any, control: Any, **_kwargs: object) -> Any:
        if state.global_step >= self.optimizer_step:
            control.should_log = True
            control.should_save = True
            control.should_training_stop = True
        return control

def _resume_value(output_dir: str | Path) -> bool:
    """Return true only when the official output has a valid trainer checkpoint."""

    root = Path(output_dir)
    if not root.exists():
        return False
    candidates: list[tuple[int, Path]] = []
    for path in root.glob("checkpoint-*"):
        suffix = path.name.removeprefix("checkpoint-")
        if suffix.isdigit():
            candidates.append((int(suffix), path))
    if not candidates:
        return False
    step, checkpoint = max(candidates)
    state_path = checkpoint / "trainer_state.json"
    if checkpoint.is_symlink() or not checkpoint.is_dir() or state_path.is_symlink():
        raise ValueError("latest GR00T checkpoint is not a regular directory")
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("latest GR00T checkpoint has invalid trainer state") from None
    if not isinstance(state, dict) or state.get("global_step") != step:
        raise ValueError("latest GR00T checkpoint trainer state does not match its step")
    return True


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
        # The pinned upstream has already built the model/dataset and saved the
        # processor before this call, and it calls save_model immediately after
        # train returns. Entering Transformers' loop at a zero target can still
        # complete optimizer step 1 before an on_train_begin stop is honored.
        if stop_step == 0:
            return None
        trainer.add_callback(StopAtOptimizerStep(stop_step))
        if kwargs.get("resume_from_checkpoint") is True:
            kwargs["resume_from_checkpoint"] = _resume_value(trainer.args.output_dir)
        return original_train(trainer, *args, **kwargs)

    Trainer.train = bounded_train
    sys.argv = [entrypoint, *official_arguments]
    runpy.run_path(entrypoint, run_name="__main__")


if __name__ == "__main__":
    main()
