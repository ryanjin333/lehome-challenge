"""Fail-closed command generator for the pinned B1K fork entrypoint."""

from __future__ import annotations

import os
from dataclasses import dataclass
import json
from pathlib import Path
import re
import signal
import subprocess
import time
from typing import Callable, Mapping

from lehome_train.b1k.rolling_checkpoints import validate_native_checkpoint
from lehome_train.b1k.training import LaunchPlan, SUPPORTED_GPU_COUNTS, approved_launch_plans
from lehome_train.constants import ISAAC_GROOT_REVISION
from lehome_train.io import canonical_json_sha256


_DEVICE = re.compile(r"(?:[0-9]+|GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9-]+)")
_EXPERIMENT = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_REQUIRED_FILES = (
    "scripts/b1k/train_b1k.py", "scripts/b1k/deploy_modality.py", "examples/b1k/r1pro.py",
    "examples/b1k/r1pro.json", "gr00t/data/dataset/lerobot_episode_loader.py",
)
_ACCEPTANCE_CHECKOUT = "/opt/isaac-groot"
_ACCEPTANCE_DATASET = "/workspace/data/b1k"
_ACCEPTANCE_MODEL = "/workspace/models/groot"
_ACCEPTANCE_OUTPUT = "/workspace/outputs"
_COMPLETED_STEP = re.compile(r"(?:global_step|step)\s*[=:]\s*15000\b", re.IGNORECASE)
_CHECKPOINT_DIRECTORY = re.compile(r"checkpoint-([1-9][0-9]*)$")
_MAX_CAPTURED_STDERR = 16 * 1024
_PROCESS_REAP_SECONDS = 30.0
_SECRET = re.compile(r"(?i)(?:hf_[a-z0-9_-]{6,}|(?:token|api[_-]?key|password)\s*[:=]\s*[^\s]+)")


@dataclass(frozen=True, slots=True)
class B1KLaunch:
    command: tuple[str, ...]
    environment: dict[str, str]
    arguments_sha256: str


class B1KLaunchFailure(RuntimeError):
    """A bounded trainer-process failure with observed durable progress."""

    def __init__(self, message: str, *, optimizer_step: int, signal_number: int | None = None) -> None:
        super().__init__(message)
        self.optimizer_step = optimizer_step
        if signal_number is not None and signal_number not in {signal.SIGTERM, signal.SIGINT}:
            raise ValueError("B1K launch signal is invalid")
        self.signal_number = signal_number


class _ProcessSignal(Exception):
    def __init__(self, number: int) -> None:
        self.number = number


def _sanitize_log(path: Path) -> None:
    """Redact credential-shaped material before a trainer log becomes evidence."""

    if path.is_symlink() or not path.is_file():
        raise ValueError("B1K trainer log is unsafe")
    raw = path.read_text(encoding="utf-8", errors="replace")
    redacted = _SECRET.sub("<redacted>", raw)
    temporary = path.with_name(f".{path.name}.redacted")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("B1K trainer log temporary path exists")
    try:
        temporary.write_text(redacted, encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sanitize_b1k_log(path: Path) -> None:
    """Make a log safe evidence: regular, redacted, and owner-readable only."""

    _sanitize_log(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o777 != 0o600:
        raise ValueError("B1K trainer log is unsafe")
    if _SECRET.search(path.read_text(encoding="utf-8", errors="replace")):
        raise ValueError("B1K trainer log contains credential-shaped material")


def _open_private_log(path: Path) -> object:
    if path.is_symlink():
        raise ValueError("B1K trainer log path is unsafe")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        return os.fdopen(descriptor, "ab")
    except BaseException:
        os.close(descriptor)
        raise


def _checkpoint_signature(root: Path) -> tuple[tuple[str, int, int], ...]:
    """Return a cheap, repeatable tree identity after strict payload validation."""

    return tuple(
        (path.relative_to(root).as_posix(), path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    )


def _durable_trainer_step(output_dir: Path) -> int:
    """Read only complete native trainer state; malformed state is not progress."""

    highest = 0
    if not output_dir.is_dir() or output_dir.is_symlink():
        return highest
    candidates = [output_dir]
    candidates.extend(output_dir.iterdir())
    for candidate in candidates:
        if candidate == output_dir:
            state = candidate / "trainer_state.json"
            try:
                value = json.loads(state.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            step = value.get("global_step") if isinstance(value, dict) else None
            if type(step) is int and step >= 0:
                highest = max(highest, step)
            continue
        matched = _CHECKPOINT_DIRECTORY.fullmatch(candidate.name)
        if matched is None or candidate.is_symlink() or not candidate.is_dir():
            continue
        state = candidate / "trainer_state.json"
        try:
            value = json.loads(state.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        step = value.get("global_step") if isinstance(value, dict) else None
        if type(step) is int and step == int(matched.group(1)):
            highest = max(highest, step)
    return highest


def _stable_native_checkpoints(
    output_dir: Path,
    *,
    world_size: int,
    observations: dict[int, tuple[tuple[str, int, int], ...]],
    published: set[int],
    resume_floor: int,
) -> tuple[int, ...]:
    """Return exactly once the checkpoint dirs unchanged across two polls."""

    ready: list[int] = []
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ValueError("B1K checkpoint output directory is unsafe")
    for candidate in sorted(output_dir.iterdir()):
        matched = _CHECKPOINT_DIRECTORY.fullmatch(candidate.name)
        if matched is None or candidate.is_symlink() or not candidate.is_dir():
            continue
        step = int(matched.group(1))
        if step not in range(1_000, 15_001, 1_000) or step <= resume_floor or step in published:
            continue
        try:
            validate_native_checkpoint(candidate, step=step, world_size=world_size)
            signature = _checkpoint_signature(candidate)
        except (OSError, ValueError):
            observations.pop(step, None)
            continue
        if observations.get(step) == signature:
            published.add(step)
            ready.append(step)
        else:
            observations[step] = signature
    return tuple(ready)


def _terminate_process_group(process: object, *, initial_signal: int = signal.SIGTERM) -> None:
    """Stop a torchrun group and always reap it after a watcher failure."""

    pid = getattr(process, "pid", None)
    wait = getattr(process, "wait", None)
    if type(pid) is not int or pid <= 0 or not callable(wait):
        return
    if initial_signal not in {signal.SIGTERM, signal.SIGINT}:
        raise ValueError("B1K process termination signal is invalid")
    try:
        os.killpg(pid, initial_signal)
    except (OSError, ProcessLookupError):
        pass
    def reap() -> bool:
        deadline = time.monotonic() + _PROCESS_REAP_SECONDS
        while True:
            try:
                wait(timeout=max(0.0, deadline - time.monotonic()))
                return True
            except InterruptedError:
                continue
            except subprocess.TimeoutExpired:
                return False

    if reap():
        return
    try:
        os.killpg(pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass
    try:
        reap()
    except OSError:
        pass


def run_b1k_launch_with_checkpoint_watch(
    launch: B1KLaunch,
    *,
    output_dir: str | Path,
    world_size: int,
    on_stable_checkpoint: Callable[[int], None],
    popen_factory: Callable[..., object] = subprocess.Popen,
    sleep: Callable[[float], None] = time.sleep,
    poll_interval_seconds: float = 2.0,
    resume_floor: int = 0,
    published_seed: tuple[int, ...] = (),
    post_exit_stability_polls: int = 2,
) -> int:
    """Run exact ``torchrun`` while publishing only stable native checkpoints.

    The child streams both outputs to disk rather than pipes, so a long run
    cannot deadlock on a full pipe.  The tail of stderr is retained only in the
    raised exception for CUDA-OOM classification and is never written to the
    durable controller status.
    """

    if not isinstance(launch, B1KLaunch) or not launch.command or launch.command[0] != "torchrun":
        raise ValueError("B1K launch execution contract is invalid")
    if world_size not in SUPPORTED_GPU_COUNTS or type(poll_interval_seconds) not in {int, float} or poll_interval_seconds <= 0:
        raise ValueError("B1K checkpoint watch configuration is invalid")
    if type(resume_floor) is not int or resume_floor not in range(0, 15_001, 1_000):
        raise ValueError("B1K checkpoint resume floor is invalid")
    if type(post_exit_stability_polls) is not int or post_exit_stability_polls < 2:
        raise ValueError("B1K checkpoint final stability polls are invalid")
    if type(published_seed) is not tuple or any(type(step) is not int or step not in range(1_000, 15_001, 1_000) or step > resume_floor for step in published_seed):
        raise ValueError("B1K checkpoint published seed is invalid")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("B1K checkpoint output directory is unsafe")
    observations: dict[int, tuple[tuple[str, int, int], ...]] = {}
    published: set[int] = set(published_seed)
    stdout_path, stderr_path = root / "trainer.stdout.log", root / "trainer.stderr.log"
    if any(path.is_symlink() for path in (stdout_path, stderr_path)):
        raise ValueError("B1K trainer log path is unsafe")
    previous_handlers: dict[int, object] = {}
    pending_signal: int | None = None
    process: object | None = None
    cleanup_in_progress = False
    def interrupted(number: int, _frame: object) -> None:
        nonlocal pending_signal
        if number not in {signal.SIGTERM, signal.SIGINT}:
            return
        if pending_signal is None:
            pending_signal = number
        # Popen has forked by the time it can return.  Retaining an early
        # signal lets the returned process group receive that exact signal.
        if process is None or cleanup_in_progress:
            return
        raise _ProcessSignal(pending_signal)
    try:
        for number in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[number] = signal.getsignal(number)
            signal.signal(number, interrupted)
        with _open_private_log(stdout_path) as stdout, _open_private_log(stderr_path) as stderr:
            process = popen_factory(launch.command, stdout=stdout, stderr=stderr, text=False, env=dict(launch.environment), start_new_session=True)
            poll = getattr(process, "poll", None)
            wait = getattr(process, "wait", None)
            if not callable(poll) or not callable(wait):
                raise ValueError("B1K launch process is invalid")
            def publish_stable() -> None:
                for step in _stable_native_checkpoints(
                    root,
                    world_size=world_size,
                    observations=observations,
                    published=published,
                    resume_floor=resume_floor,
                ):
                    on_stable_checkpoint(step)

            try:
                if pending_signal is not None:
                    raise _ProcessSignal(pending_signal)
                while poll() is None:
                    publish_stable()
                    sleep(float(poll_interval_seconds))
                for observation in range(post_exit_stability_polls):
                    publish_stable()
                    if observation + 1 < post_exit_stability_polls:
                        sleep(float(poll_interval_seconds))
                returncode = wait()
            except _ProcessSignal as interruption:
                cleanup_in_progress = True
                _terminate_process_group(process, initial_signal=interruption.number)
                raise B1KLaunchFailure(
                    f"B1K torchrun interrupted by signal {interruption.number}",
                    optimizer_step=_durable_trainer_step(root),
                    signal_number=interruption.number,
                ) from None
            except BaseException:
                cleanup_in_progress = True
                _terminate_process_group(process)
                raise
    finally:
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)
        for path in (stdout_path, stderr_path):
            if path.exists():
                sanitize_b1k_log(path)
    detail = stderr_path.read_text(encoding="utf-8", errors="replace")[-_MAX_CAPTURED_STDERR:]
    step = _durable_trainer_step(root)
    if type(returncode) is not int or returncode != 0:
        raise B1KLaunchFailure("B1K torchrun failed: " + detail, optimizer_step=step)
    if step != 15_000:
        raise B1KLaunchFailure("B1K torchrun did not prove completion at step 15000", optimizer_step=step)
    return 15_000


def execute_b1k_launch(
    launch: B1KLaunch,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> int:
    """Execute precisely the validated torchrun argv and prove final progress.

    ``build_b1k_launch`` is the only source of ``B1KLaunch`` in production.
    Keeping execution here makes the subprocess boundary auditable: no shell,
    no inherited Hub token, and no success without an explicit step-15,000
    completion marker from the upstream trainer.
    """

    if not isinstance(launch, B1KLaunch):
        raise ValueError("B1K launch is invalid")
    if not launch.command or launch.command[0] != "torchrun" or "HF_TOKEN" in launch.environment:
        raise ValueError("B1K launch execution contract is invalid")
    completed = runner(
        launch.command,
        check=False,
        capture_output=True,
        text=True,
        env=dict(launch.environment),
    )
    stdout = getattr(completed, "stdout", "")
    stderr = getattr(completed, "stderr", "")
    returncode = getattr(completed, "returncode", None)
    if type(returncode) is not int:
        raise ValueError("B1K launch returned an invalid process result")
    if returncode != 0:
        raise ValueError("B1K torchrun failed")
    if type(stdout) is not str or type(stderr) is not str or not _COMPLETED_STEP.search(stdout + "\n" + stderr):
        raise ValueError("B1K torchrun did not prove completion at step 15000")
    return 15_000


def _checkout_head(checkout: Path, environment: Mapping[str, str]) -> str:
    try:
        result = subprocess.run(("git", "-C", str(checkout), "rev-parse", "HEAD"), check=False, capture_output=True, text=True, env=dict(environment))
    except OSError as error:
        raise ValueError("B1K fork checkout is not readable") from error
    if result.returncode != 0:
        raise ValueError("B1K fork checkout is not readable")
    return result.stdout.strip()


def _checkout_is_clean(checkout: Path, environment: Mapping[str, str]) -> bool:
    try:
        result = subprocess.run(("git", "-C", str(checkout), "status", "--porcelain=v1", "--untracked-files=all"), check=False, capture_output=True, text=True, env=dict(environment))
    except OSError:
        return False
    return result.returncode == 0 and not result.stdout


def _visible_devices(value: str | None, world_size: int) -> str:
    if type(value) is not str or value != value.strip() or not value:
        raise ValueError("CUDA_VISIBLE_DEVICES must contain exactly the requested ranks")
    devices = value.split(",")
    if len(devices) != world_size or len(set(devices)) != len(devices) or not all(_DEVICE.fullmatch(item) for item in devices):
        raise ValueError("CUDA_VISIBLE_DEVICES must contain exactly the requested ranks")
    return value


def actual_b1k_output_root(output_dir: str | os.PathLike[str], experiment_name: str) -> Path:
    """Return the checkpoint root produced by the pinned upstream trainer.

    The B1K entrypoint appends ``experiment_name`` to ``--output-dir``.  Keep
    that interface fact in one place so the watcher, restoration, and final
    evidence cannot accidentally inspect the parent directory.
    """

    root = Path(output_dir)
    if not str(root) or type(experiment_name) is not str or not _EXPERIMENT.fullmatch(experiment_name):
        raise ValueError("B1K output root inputs are invalid")
    return root / experiment_name


def build_b1k_command(
    plan: LaunchPlan,
    *,
    checkout: str,
    dataset_path: str,
    base_model_path: str,
    output_dir: str,
    experiment_name: str,
    resume_from_checkpoint: bool = False,
) -> tuple[str, ...]:
    """Return the sole supported upstream argv without executing it."""

    if plan not in approved_launch_plans(num_gpus=plan.num_gpus):
        raise ValueError("B1K launch plan must be one of the approved identities")
    if type(experiment_name) is not str or not _EXPERIMENT.fullmatch(experiment_name):
        raise ValueError("experiment_name must be a safe deterministic run identity")
    if any(type(value) is not str or not value for value in (checkout, dataset_path, base_model_path, output_dir)):
        raise ValueError("B1K command paths must be non-empty")
    if type(resume_from_checkpoint) is not bool:
        raise ValueError("resume_from_checkpoint must be boolean")
    return (
        "torchrun", f"--nproc_per_node={plan.num_gpus}", "--master_port=29500",
        str(Path(checkout) / "scripts/b1k/train_b1k.py"),
        "--base-model-path", base_model_path, "--dataset-path", dataset_path, "--output-dir", output_dir,
        "--experiment-name", experiment_name, "--embodiment-tag", "NEW_EMBODIMENT",
        "--modality-config-path", str(Path(checkout) / "examples/b1k/r1pro.py"),
        "--num-gpus", str(plan.num_gpus), "--global-batch-size", str(plan.global_batch_size),
        "--gradient-accumulation-steps", str(plan.gradient_accumulation_steps),
        "--max-steps", "15000", "--save-steps", "1000", "--save-total-limit", "2",
        "--learning-rate", str(plan.learning_rate), "--weight-decay", str(plan.weight_decay),
        "--warmup-ratio", str(plan.warmup_ratio), "--decode-only-used-frames",
    ) + (("--resume-from-checkpoint",) if resume_from_checkpoint else ())


def build_b1k_launch(
    plan: LaunchPlan,
    *,
    visible_devices: str | None,
    environment: Mapping[str, str] | None,
    official_checkout: str | os.PathLike[str],
    dataset_path: str,
    base_model_path: str,
    output_dir: str,
    experiment_name: str,
    resume_from_checkpoint: bool = False,
) -> B1KLaunch:
    if not isinstance(plan, LaunchPlan) or plan.num_gpus not in SUPPORTED_GPU_COUNTS:
        raise ValueError("B1K launch plan must use one to four ranks")
    if plan.max_steps != 15_000 or plan.save_steps != 1_000 or plan.checkpoint_keep != 2:
        raise ValueError("B1K launch plan must retain the initial 15,000/1,000/latest-two contract")
    if any(type(value) is not str or not value for value in (dataset_path, base_model_path, output_dir)):
        raise ValueError("dataset_path, base_model_path, and output_dir must be non-empty")
    if type(resume_from_checkpoint) is not bool:
        raise ValueError("resume_from_checkpoint must be boolean")
    visible = _visible_devices(visible_devices, plan.num_gpus)
    cleaned = {key: value for key, value in (os.environ if environment is None else environment).items() if key != "HF_TOKEN"}
    cleaned["CUDA_VISIBLE_DEVICES"] = visible
    checkout = Path(official_checkout)
    if _checkout_head(checkout, cleaned) != ISAAC_GROOT_REVISION or not _checkout_is_clean(checkout, cleaned):
        raise ValueError("B1K fork checkout is not the exact clean pinned revision")
    for relative in _REQUIRED_FILES:
        if not (checkout / relative).is_file():
            raise ValueError(f"B1K fork required file is missing: {relative}")
    command = build_b1k_command(plan, checkout=str(checkout), dataset_path=dataset_path, base_model_path=base_model_path, output_dir=output_dir, experiment_name=experiment_name, resume_from_checkpoint=resume_from_checkpoint)
    return B1KLaunch(command=command, environment=cleaned, arguments_sha256=canonical_json_sha256(command))
