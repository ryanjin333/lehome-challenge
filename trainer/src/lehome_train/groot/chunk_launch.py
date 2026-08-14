"""Run the exact pinned GR00T launcher with one resumable stop boundary."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
from typing import Any, Mapping

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

def _resume_step(output_dir: str | Path, *, num_gpus: int = 1) -> int | None:
    """Return the authenticated latest checkpoint step, if one exists."""

    root = Path(output_dir)
    if not root.exists():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in root.glob("checkpoint-*"):
        suffix = path.name.removeprefix("checkpoint-")
        if suffix.isdigit():
            candidates.append((int(suffix), path))
    if not candidates:
        return None
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
    if num_gpus == 4:
        shard_root = checkpoint / f"global_step{step}"
        expected = {
            "mp_rank_00_model_states.pt",
            *(f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt" for rank in range(4)),
        }
        if (
            not shard_root.is_dir()
            or shard_root.is_symlink()
            or {entry.name for entry in shard_root.iterdir()} != expected
            or any(not entry.is_file() or entry.is_symlink() for entry in shard_root.iterdir())
        ):
            raise ValueError("latest GR00T checkpoint has incomplete ZeRO-2 shards")
    return step


def _resume_value(output_dir: str | Path, *, num_gpus: int = 1) -> bool:
    """Return true only when the official output has a valid trainer checkpoint."""

    return _resume_step(output_dir, num_gpus=num_gpus) is not None


def _runtime_checkpoint_binding(
    runtime_arguments: list[str], official_arguments: list[str], *, num_gpus: int
) -> tuple[int, int]:
    """Derive the sole runtime cursor from the checked checkpoint directory."""

    try:
        wrapper_options = runtime_arguments[:runtime_arguments.index("--")]
    except ValueError:
        raise ValueError("chunk runtime launcher requires a wrapper/official argument separator") from None
    if any(
        flag in wrapper_options
        for flag in ("--resume-sample-offset", "--resume-global-step", "--global-batch-size")
    ):
        raise ValueError("runtime cursor flags must be injected from an authenticated checkpoint")
    try:
        output = Path(official_arguments[official_arguments.index("--output-dir") + 1])
        experiment = official_arguments[official_arguments.index("--experiment-name") + 1]
        global_batch = int(official_arguments[official_arguments.index("--global-batch-size") + 1])
    except (ValueError, IndexError):
        raise ValueError("chunk runtime launcher requires canonical output and global batch arguments") from None
    if not output.is_absolute() or not experiment or "/" in experiment or "\\" in experiment or global_batch <= 0:
        raise ValueError("chunk runtime launcher has unsafe checkpoint binding arguments")
    step = _resume_step(output / experiment, num_gpus=num_gpus)
    return (0 if step is None else step), global_batch


def _arguments(argv: list[str] | None) -> tuple[int, str, list[str], list[str] | None]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stop-after-step", type=int, required=True)
    parser.add_argument("remainder", nargs=argparse.REMAINDER)
    parsed = parser.parse_args(argv)
    remainder = list(parsed.remainder)
    if remainder[:1] == ["--"]:
        remainder.pop(0)
    if remainder[:2] == ["-m", "lehome_train.groot.runtime_mixture_entrypoint"]:
        try:
            official_index = remainder.index("--official-launch")
            separator = remainder.index("--", official_index + 2)
            entrypoint = remainder[official_index + 1]
            official = remainder[separator + 1 :]
        except (ValueError, IndexError):
            raise ValueError("chunk runtime launcher requires canonical official launch arguments") from None
        if not entrypoint.endswith("gr00t/experiment/launch_finetune.py") or not official:
            raise ValueError("chunk runtime launcher requires the pinned official entrypoint")
        # Keep the entire authenticated wrapper.  The chunk guard has to patch
        # ``Trainer.train`` *and* let the wrapper replace DatasetFactory in
        # this same interpreter; reducing this to the official script silently
        # bypasses the runtime mixture.
        return parsed.stop_after_step, entrypoint, official, remainder[2:]
    if not remainder or not remainder[0].endswith("gr00t/experiment/launch_finetune.py"):
        raise ValueError("chunk launcher requires the pinned official entrypoint")
    if parsed.stop_after_step < 0:
        raise ValueError("chunk stop optimizer step must be nonnegative")
    return parsed.stop_after_step, remainder[0], remainder[1:], None


def _official_num_gpus(arguments: list[str]) -> int:
    """Read the pinned launcher's declared world size without inventing a flag."""

    try:
        value = arguments[arguments.index("--num-gpus") + 1]
    except (ValueError, IndexError):
        # Legacy single-GPU invocations predate the explicit upstream flag.
        return 1
    if value not in {"1", "4"}:
        raise ValueError("chunk launcher supports only one or four GPUs")
    return int(value)


def _validate_distributed_rank_device(
    *,
    world_size: str | None,
    local_rank: str | None,
    rank: str | None,
    device_count: int,
    num_gpus: int,
) -> int:
    """Fail before the official launcher when torchrun ranks are inconsistent."""

    if num_gpus != 4:
        raise ValueError("distributed rank validation requires four GPUs")
    try:
        parsed_world_size = int(world_size or "")
        parsed_local_rank = int(local_rank or "")
        parsed_rank = int(rank or "")
    except ValueError:
        raise ValueError("torchrun world size and ranks must be explicit integers") from None
    if parsed_world_size != num_gpus:
        raise ValueError("torchrun world size does not match the configured GPU count")
    if not 0 <= parsed_local_rank < num_gpus:
        raise ValueError("torchrun local rank does not name a configured GPU")
    if not 0 <= parsed_rank < num_gpus:
        raise ValueError("torchrun rank does not name a configured GPU")
    if type(device_count) is not int or device_count != num_gpus:
        raise ValueError("visible CUDA GPUs do not match the configured GPU count")
    return parsed_local_rank


def _configure_rank_device(num_gpus: int, environment: Mapping[str, str]) -> None:
    """Bind every torchrun child to exactly its local CUDA device."""

    if num_gpus == 1:
        return
    try:
        import torch
    except ImportError:
        raise RuntimeError("pinned PyTorch runtime is unavailable") from None
    local_rank = _validate_distributed_rank_device(
        world_size=environment.get("WORLD_SIZE"),
        local_rank=environment.get("LOCAL_RANK"),
        rank=environment.get("RANK"),
        device_count=torch.cuda.device_count(),
        num_gpus=num_gpus,
    )
    torch.cuda.set_device(local_rank)


def _rank_metadata_staging(
    official_arguments: list[str],
    *,
    num_gpus: int,
    environment: Mapping[str, str],
) -> tuple[list[str], Path | None, Path | None]:
    """Give follower ranks private pre-train metadata roots, never checkpoints."""

    if num_gpus == 1:
        return official_arguments, None, None
    try:
        local_rank = _validate_distributed_rank_device(
            world_size=environment.get("WORLD_SIZE"),
            local_rank=environment.get("LOCAL_RANK"),
            rank=environment.get("RANK"),
            device_count=num_gpus,
            num_gpus=num_gpus,
        )
        output_index = official_arguments.index("--output-dir") + 1
        experiment_index = official_arguments.index("--experiment-name") + 1
        output_root = Path(official_arguments[output_index])
        experiment_name = official_arguments[experiment_index]
    except (ValueError, IndexError):
        raise ValueError("chunk launcher requires canonical output and experiment arguments") from None
    if (
        not output_root.is_absolute()
        or ".." in output_root.parts
        or not experiment_name
        or "/" in experiment_name
        or "\\" in experiment_name
        or official_arguments.count("--output-dir") != 1
        or official_arguments.count("--experiment-name") != 1
    ):
        raise ValueError("chunk launcher output arguments are unsafe")
    canonical_run = output_root / experiment_name
    if local_rank == 0:
        return official_arguments, canonical_run, None
    staging_root = output_root / f".lehome-rank-metadata-{local_rank}"
    rewritten = list(official_arguments)
    rewritten[output_index] = str(staging_root)
    return rewritten, canonical_run, staging_root


def _restore_canonical_trainer_output(
    trainer: Any,
    *,
    canonical_run: Path | None,
    synchronize: Callable[[], None],
) -> None:
    """Release all ranks only after metadata setup, then share checkpoint root."""

    synchronize()
    if canonical_run is not None:
        trainer.args.output_dir = str(canonical_run)


def _cleanup_metadata_staging(metadata_staging: Path | None) -> None:
    """Remove only a validated follower-only metadata directory after runpy exits."""

    if metadata_staging is None:
        return
    try:
        resolved_staging = metadata_staging.resolve(strict=False)
        resolved_parent = metadata_staging.parent.resolve(strict=False)
        if (
            resolved_staging.parent == resolved_parent
            and resolved_staging.name.startswith(".lehome-rank-metadata-")
            and metadata_staging.exists()
            and not metadata_staging.is_symlink()
        ):
            shutil.rmtree(metadata_staging)
    except OSError:
        pass


def main(argv: list[str] | None = None) -> None:
    stop_step, entrypoint, official_arguments, runtime_arguments = _arguments(argv)
    num_gpus = _official_num_gpus(official_arguments)
    runtime_binding = (
        None
        if runtime_arguments is None
        else _runtime_checkpoint_binding(runtime_arguments, official_arguments, num_gpus=num_gpus)
    )
    expected_runtime_step = None if runtime_binding is None else runtime_binding[0]
    _configure_rank_device(num_gpus, os.environ)
    official_arguments, canonical_run, metadata_staging = _rank_metadata_staging(
        official_arguments, num_gpus=num_gpus, environment=os.environ
    )
    try:
        from transformers import Trainer
    except ImportError:
        raise RuntimeError("pinned Transformers runtime is unavailable") from None

    original_train = Trainer.train

    def synchronize() -> None:
        if num_gpus == 1:
            return
        try:
            import torch.distributed as dist
        except ImportError:
            raise RuntimeError("pinned PyTorch distributed runtime is unavailable") from None
        if not dist.is_initialized():
            raise RuntimeError("distributed metadata synchronization was not initialized")
        dist.barrier()

    def bounded_train(trainer: Any, *args: object, **kwargs: object) -> Any:
        # The pinned upstream has already built the model/dataset and saved the
        # processor before this call, and it calls save_model immediately after
        # train returns. Entering Transformers' loop at a zero target can still
        # complete optimizer step 1 before an on_train_begin stop is honored.
        _restore_canonical_trainer_output(
            trainer, canonical_run=canonical_run, synchronize=synchronize
        )
        if stop_step == 0:
            return None
        trainer.add_callback(StopAtOptimizerStep(stop_step))
        actual_step = 0
        if kwargs.get("resume_from_checkpoint") is True:
            actual = _resume_step(trainer.args.output_dir, num_gpus=num_gpus)
            kwargs["resume_from_checkpoint"] = actual is not None
            actual_step = 0 if actual is None else actual
        if expected_runtime_step is not None:
            if actual_step != expected_runtime_step:
                raise ValueError("runtime checkpoint step does not match authenticated runtime resume binding")
            dataset = getattr(trainer, "train_dataset", None)
            if dataset is None or type(getattr(dataset, "seed", None)) is not int or not callable(getattr(dataset, "reset_seed", None)):
                raise ValueError("pinned trainer did not expose a seed-resettable runtime dataset")
            # Gr00tTrainer.get_train_dataloader performs the pinned
            # reset_seed(dataset.seed + state.global_step) call.  Do not
            # pre-reset here, or an arbitrary Trainer implementation could
            # bypass that exact upstream resume contract.
        return original_train(trainer, *args, **kwargs)

    Trainer.train = bounded_train
    try:
        if runtime_arguments is None:
            sys.argv = [entrypoint, *official_arguments]
            runpy.run_path(entrypoint, run_name="__main__")
        else:
            separator = runtime_arguments.index("--")
            assert runtime_binding is not None
            step, global_batch = runtime_binding
            guarded_arguments = [
                *runtime_arguments[:separator],
                "--resume-sample-offset", str(step * global_batch),
                "--resume-global-step", str(step),
                "--global-batch-size", str(global_batch),
                "--",
                *official_arguments,
            ]
            # Importing and calling the wrapper is intentional: it retains the
            # active Trainer.train patch while it performs the narrow pinned
            # DatasetFactory substitution before its runpy invocation.
            from lehome_train.groot import runtime_mixture_entrypoint

            runtime_mixture_entrypoint.main(guarded_arguments)
    finally:
        _cleanup_metadata_staging(metadata_staging)


if __name__ == "__main__":
    main()
