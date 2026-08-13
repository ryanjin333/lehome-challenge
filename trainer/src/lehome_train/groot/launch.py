"""Secret-safe adapter for the pinned official GR00T N1.7 launcher."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys
from typing import Callable, Mapping, Sequence

from lehome_train.constants import ISAAC_GROOT_REVISION
from lehome_train.flywheel.augmentation import augmentation_profile, color_jitter_cli
from lehome_train.groot.config import FineTuneLaunchConfig
from lehome_train.groot.checkpoint_identity import policy_artifact_sha256
from lehome_train.io import atomic_write_json


_CUDA_VISIBLE_DEVICE = re.compile(r"(?:[0-9]+|GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9-]+)")
_IDENTITY_FILENAME = "lehome_launch.json"


@dataclass(frozen=True, slots=True)
class OfficialLaunch:
    """A ready-to-run official command plus a secret-stripped environment."""

    command: tuple[str, ...]
    environment: dict[str, str]


def _require_visible_gpus(value: str | None, *, expected_count: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"exactly {expected_count} visible GPU{'s' if expected_count != 1 else ''} are required")
    candidates = tuple(candidate.strip() for candidate in value.split(","))
    if (
        len(candidates) != expected_count
        or any(not _CUDA_VISIBLE_DEVICE.fullmatch(candidate) for candidate in candidates)
        or len(set(candidates)) != len(candidates)
    ):
        if expected_count == 4:
            raise ValueError("exactly four visible GPUs are required")
        raise ValueError("exactly one visible GPU is required")
    return ",".join(candidates)


def _safe_environment(
    environment: Mapping[str, str] | None,
    *,
    visible_devices: str,
) -> dict[str, str]:
    source = os.environ if environment is None else environment
    cleaned = {
        key: value
        for key, value in source.items()
        if key != "HF_TOKEN"
    }
    cleaned["CUDA_VISIBLE_DEVICES"] = visible_devices
    return cleaned


def _checkout_head(checkout: Path, environment: Mapping[str, str]) -> str:
    """Return a checkout's exact Git identity without exposing command output."""

    try:
        completed = subprocess.run(
            ("git", "-C", str(checkout), "rev-parse", "HEAD"),
            check=False,
            capture_output=True,
            text=True,
            env=dict(environment),
        )
    except OSError as error:
        raise ValueError("official GR00T checkout is not a readable Git checkout") from error
    if completed.returncode != 0:
        raise ValueError("official GR00T checkout is not a readable Git checkout")
    return completed.stdout.strip()


def _checkout_is_clean(checkout: Path, environment: Mapping[str, str]) -> bool:
    """Return whether the checkout has no staged, modified, or untracked code."""

    try:
        completed = subprocess.run(
            ("git", "-C", str(checkout), "status", "--porcelain=v1", "--untracked-files=all"),
            check=False,
            capture_output=True,
            text=True,
            env=dict(environment),
        )
    except OSError:
        return False
    return completed.returncode == 0 and not completed.stdout


def _official_entrypoint(
    official_checkout: str | os.PathLike[str],
    environment: Mapping[str, str],
) -> Path:
    checkout = Path(official_checkout)
    entrypoint = checkout / "gr00t" / "experiment" / "launch_finetune.py"
    if not entrypoint.is_file():
        raise ValueError("official GR00T entrypoint is missing")
    if _checkout_head(checkout, environment) != ISAAC_GROOT_REVISION:
        raise ValueError("official checkout is not pinned Isaac-GR00T revision")
    if not _checkout_is_clean(checkout, environment):
        raise ValueError("official GR00T checkout is not clean")
    return entrypoint


def build_launch(
    config: FineTuneLaunchConfig,
    *,
    visible_devices: str | None,
    environment: Mapping[str, str] | None,
    official_checkout: str | os.PathLike[str],
) -> OfficialLaunch:
    """Build a one- or four-GPU command for NVIDIA's pinned ``launch_finetune.py``.

    This wrapper intentionally does not alter the upstream training loop.  The
    model and dataset revisions are recorded in the immutable experiment
    identity; their local snapshots are passed to the upstream path-only API.
    """

    if config.parent_checkpoint_artifact_sha256 is not None and (
        policy_artifact_sha256(config.base_model_path)
        != config.parent_checkpoint_artifact_sha256
    ):
        raise ValueError("parent checkpoint artifact digest mismatch")
    visible_gpus = _require_visible_gpus(visible_devices, expected_count=config.num_gpus)
    safe_environment = _safe_environment(environment, visible_devices=visible_gpus)
    entrypoint = _official_entrypoint(official_checkout, safe_environment)
    command = (
        sys.executable,
        str(entrypoint),
        "--base-model-path",
        config.base_model_path,
        "--dataset-path",
        config.dataset_path,
        "--embodiment-tag",
        "NEW_EMBODIMENT",
        "--modality-config-path",
        config.modality_config_path,
        "--num-gpus",
        str(config.num_gpus),
        "--output-dir",
        config.output_dir,
        "--experiment-name",
        config.experiment_name,
        "--global-batch-size",
        str(config.global_batch_size),
        "--gradient-accumulation-steps",
        "1",
        "--max-steps",
        str(config.max_steps),
        "--save-steps",
        str(config.save_steps),
        "--save-total-limit",
        str(config.save_total_limit),
        "--warmup-ratio",
        str(float(config.warmup_ratio)),
        "--learning-rate",
        str(float(config.learning_rate)),
        "--weight-decay",
        str(float(config.weight_decay)),
        "--dataloader-num-workers",
        str(config.dataloader_num_workers),
        "--no-tune-llm",
        "--no-tune-visual",
        "--tune-projector",
        "--tune-diffusion-model",
        *color_jitter_cli(
            augmentation_profile(
                config.augmentation_profile, receipt=config.augmentation_receipt
            )
        ),
    )
    return OfficialLaunch(
        command=command,
        environment=safe_environment,
    )


def _existing_identity(output_dir: Path) -> dict[str, object] | None:
    if not output_dir.exists():
        return None
    if not output_dir.is_dir():
        raise ValueError("incompatible experiment output path is not a directory")
    identity_path = output_dir / _IDENTITY_FILENAME
    if not identity_path.exists():
        if any(output_dir.iterdir()):
            raise ValueError("incompatible experiment directory has no launch identity")
        return None
    try:
        decoded = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("incompatible experiment launch identity") from error
    if not isinstance(decoded, dict):
        raise ValueError("incompatible experiment launch identity")
    return decoded


def _write_or_verify_identity(config: FineTuneLaunchConfig) -> None:
    # The pinned upstream launcher nests named runs below output_dir.  The
    # identity must live beside checkpoints, not merely beside their parent.
    output_dir = Path(config.output_dir) / config.experiment_name
    existing = _existing_identity(output_dir)
    expected = config.identity()
    if existing is not None:
        if existing != expected:
            raise ValueError("incompatible experiment directory")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / _IDENTITY_FILENAME, expected)


Runner = Callable[..., subprocess.CompletedProcess[object]]


def launch_finetune(
    config: FineTuneLaunchConfig,
    *,
    visible_devices: str | None,
    environment: Mapping[str, str] | None,
    official_checkout: str | os.PathLike[str],
    runner: Runner = subprocess.run,
) -> subprocess.CompletedProcess[object]:
    """Record compatible experiment identity and execute the upstream launcher.

    The child receives no Hugging Face token.  Artifact transfers are separate
    trusted operations; training only reads pre-downloaded, revision-verified
    model and dataset snapshots.
    """

    launch = build_launch(
        config,
        visible_devices=visible_devices,
        environment=environment,
        official_checkout=official_checkout,
    )
    _write_or_verify_identity(config)
    if config.num_gpus == 1:
        return runner(launch.command, env=launch.environment, check=True)
    return runner(
        _distributed_chunk_command(config, stop_after_optimizer_step=config.max_steps, launch=launch),
        env=launch.environment,
        check=True,
    )


def launch_continuous_finetune(
    config: FineTuneLaunchConfig,
    *,
    visible_devices: str | None,
    environment: Mapping[str, str] | None,
    official_checkout: str | os.PathLike[str],
    runner: Runner = subprocess.run,
) -> subprocess.CompletedProcess[object]:
    """Run the one official corrective process which owns both saves."""

    if config.num_gpus != 1:
        raise ValueError("continuous corrective training requires one GPU")
    if config.global_batch_size != 64 or config.physical_batch_size != 64:
        raise ValueError("first continuous corrective run requires global batch 64")
    if config.max_steps != 2_000 or config.save_steps != 1_000:
        raise ValueError("continuous corrective training requires 1000/2000 checkpoints")
    return launch_finetune(
        config,
        visible_devices=visible_devices,
        environment=environment,
        official_checkout=official_checkout,
        runner=runner,
    )


def _distributed_chunk_command(
    config: FineTuneLaunchConfig,
    *,
    stop_after_optimizer_step: int,
    launch: OfficialLaunch,
) -> tuple[str, ...]:
    """Run the resume patch in every DDP rank with the current interpreter."""

    chunk_arguments = (
        "-m",
        "lehome_train.groot.chunk_launch",
        "--stop-after-step",
        str(stop_after_optimizer_step),
        "--",
        *launch.command[1:],
    )
    if config.num_gpus == 1:
        return (sys.executable, *chunk_arguments)
    return (
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={config.num_gpus}",
        *chunk_arguments,
    )


def launch_finetune_to_step(
    config: FineTuneLaunchConfig,
    *,
    stop_after_optimizer_step: int,
    visible_devices: str | None,
    environment: Mapping[str, str] | None,
    official_checkout: str | os.PathLike[str],
    runner: Runner = subprocess.run,
) -> subprocess.CompletedProcess[object]:
    """Resume the pinned launcher but stop at one controller-owned boundary.

    The upstream run always calls ``Trainer.train(resume_from_checkpoint=True)``.
    For target zero, the image-owned wrapper returns before entering the
    Transformers loop; upstream has already initialized the pipeline and saves
    the model after the return. Positive targets add only a stop callback. The
    official full-run ``max_steps`` remains unchanged, so resumed chunks keep
    one optimizer and warmup/cosine schedule instead of restarting it at each
    upload boundary.
    """

    if (
        type(stop_after_optimizer_step) is not int
        or stop_after_optimizer_step < 0
        or stop_after_optimizer_step > config.max_steps
    ):
        raise ValueError("chunk stop step must be within the full training schedule")
    launch = build_launch(
        config,
        visible_devices=visible_devices,
        environment=environment,
        official_checkout=official_checkout,
    )
    _write_or_verify_identity(config)
    wrapped = _distributed_chunk_command(
        config,
        stop_after_optimizer_step=stop_after_optimizer_step,
        launch=launch,
    )
    return runner(wrapped, env=launch.environment, check=True)
