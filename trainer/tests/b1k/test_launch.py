from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import json
import signal

import pytest

from lehome_train.b1k.launch import B1KLaunch, B1KLaunchFailure, actual_b1k_output_root, build_b1k_launch, execute_b1k_launch, run_b1k_launch_with_checkpoint_watch
from lehome_train.b1k.training import approved_launch_plans
from lehome_train.constants import ISAAC_GROOT_REVISION


@pytest.fixture
def b1k_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    checkout = tmp_path / "isaac-groot"
    for relative in (
        "scripts/b1k/train_b1k.py",
        "scripts/b1k/deploy_modality.py",
        "examples/b1k/r1pro.py",
        "examples/b1k/r1pro.json",
        "gr00t/data/dataset/lerobot_episode_loader.py",
    ):
        path = checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")
    monkeypatch.setattr("lehome_train.b1k.launch._checkout_head", lambda *_args: ISAAC_GROOT_REVISION)
    monkeypatch.setattr("lehome_train.b1k.launch._checkout_is_clean", lambda *_args: True)
    return checkout


@pytest.mark.parametrize("num_gpus,visible", [(1, "0"), (2, "0,1"), (3, "0,1,2"), (4, "0,1,2,3")])
def test_b1k_launch_uses_pinned_torchrun_and_exact_initial_contract(
    b1k_checkout: Path, num_gpus: int, visible: str
) -> None:
    plan = approved_launch_plans(num_gpus=num_gpus)[0]
    launch = build_b1k_launch(
        plan,
        visible_devices=visible,
        environment={"HF_TOKEN": "not-a-real-token", "PATH": "/bin"},
        official_checkout=b1k_checkout,
        dataset_path="/prepared/b1k",
        base_model_path="/prepared/base-model",
        output_dir="/output/b1k",
        experiment_name="b1k-20260803-001",
    )

    assert launch.command[:4] == (
        "torchrun", f"--nproc_per_node={num_gpus}", "--master_port=29500",
        str(b1k_checkout / "scripts/b1k/train_b1k.py"),
    )
    assert launch.command[launch.command.index("--num-gpus") + 1] == str(num_gpus)
    assert launch.command[launch.command.index("--global-batch-size") + 1] == str(plan.global_batch_size)
    assert launch.command[launch.command.index("--gradient-accumulation-steps") + 1] == str(plan.gradient_accumulation_steps)
    assert launch.command[launch.command.index("--max-steps") + 1] == "15000"
    assert launch.command[launch.command.index("--save-steps") + 1] == "1000"
    assert launch.command[launch.command.index("--save-total-limit") + 1] == "2"
    assert "--decode-only-used-frames" in launch.command
    assert str(b1k_checkout / "examples/b1k/r1pro.py") in launch.command
    assert launch.command[launch.command.index("--base-model-path") + 1] == "/prepared/base-model"
    assert launch.command[launch.command.index("--embodiment-tag") + 1] == "NEW_EMBODIMENT"
    assert launch.command[launch.command.index("--experiment-name") + 1] == "b1k-20260803-001"
    assert launch.command[launch.command.index("--learning-rate") + 1] == "0.0001"
    assert len(launch.arguments_sha256) == 64
    assert launch.environment["CUDA_VISIBLE_DEVICES"] == visible
    assert "HF_TOKEN" not in launch.environment


def test_b1k_launch_accepts_native_resume_and_rejects_device_mismatch(b1k_checkout: Path) -> None:
    launch = build_b1k_launch(
        approved_launch_plans(num_gpus=1)[0],
        visible_devices="0",
        environment={},
        official_checkout=b1k_checkout,
        dataset_path="/prepared/b1k",
        base_model_path="/prepared/base-model",
        output_dir="/output/b1k",
        experiment_name="b1k-20260803-001",
        resume_from_checkpoint=True,
    )
    assert launch.command[-1:] == ("--resume-from-checkpoint",)
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        build_b1k_launch(
            approved_launch_plans(num_gpus=2)[0], visible_devices="0", environment={},
            official_checkout=b1k_checkout, dataset_path="/prepared/b1k", base_model_path="/prepared/base-model", output_dir="/output/b1k", experiment_name="b1k-20260803-001",
        )


def test_pinned_upstream_output_is_the_parent_once_plus_experiment_name(b1k_checkout: Path) -> None:
    launch = build_b1k_launch(
        approved_launch_plans(num_gpus=1)[0],
        visible_devices="0",
        environment={"WANDB_MODE": "offline"},
        official_checkout=b1k_checkout,
        dataset_path="/prepared/b1k",
        base_model_path="/prepared/base-model",
        output_dir="/workspace/outputs/b1k-run-001",
        experiment_name="b1k-run-001",
    )

    assert launch.command[launch.command.index("--output-dir") + 1] == "/workspace/outputs/b1k-run-001"
    assert actual_b1k_output_root("/workspace/outputs/b1k-run-001", "b1k-run-001") == Path("/workspace/outputs/b1k-run-001/b1k-run-001")
    assert launch.environment["WANDB_MODE"] == "offline"


def test_b1k_launch_rejects_plan_outside_approved_identities(b1k_checkout: Path) -> None:
    plan = replace(approved_launch_plans(num_gpus=1)[0], identity="unsafe-custom-plan")
    with pytest.raises(ValueError, match="approved"):
        build_b1k_launch(
            plan, visible_devices="0", environment={}, official_checkout=b1k_checkout,
            dataset_path="/prepared/b1k", base_model_path="/prepared/base-model", output_dir="/output/b1k", experiment_name="b1k-20260803-001",
        )


def test_b1k_launch_rejects_unsafe_experiment_name(b1k_checkout: Path) -> None:
    with pytest.raises(ValueError, match="experiment_name"):
        build_b1k_launch(
            approved_launch_plans(num_gpus=1)[0], visible_devices="0", environment={}, official_checkout=b1k_checkout,
            dataset_path="/prepared/b1k", base_model_path="/prepared/base-model", output_dir="/output/b1k", experiment_name="../unsafe",
        )


def test_execute_b1k_launch_uses_exact_argv_and_sanitized_environment(b1k_checkout: Path) -> None:
    launch = build_b1k_launch(
        approved_launch_plans(num_gpus=1)[0],
        visible_devices="0",
        environment={"HF_TOKEN": "not-a-real-token", "PATH": "/bin"},
        official_checkout=b1k_checkout,
        dataset_path="/prepared/b1k",
        base_model_path="/prepared/base-model",
        output_dir="/output/b1k",
        experiment_name="b1k-20260803-001",
    )
    observed: dict[str, object] = {}

    def run(command: tuple[str, ...], **kwargs: object) -> object:
        observed["command"] = command
        observed["environment"] = kwargs["env"]
        return type("Result", (), {"returncode": 0, "stdout": "completed global_step=15000", "stderr": ""})()

    assert execute_b1k_launch(launch, runner=run) == 15_000
    assert observed["command"] == launch.command
    assert "HF_TOKEN" not in observed["environment"]


def test_execute_b1k_launch_fails_closed_without_exact_completion() -> None:
    launch = B1KLaunch(command=("torchrun", "train.py"), environment={}, arguments_sha256="a" * 64)
    with pytest.raises(ValueError, match="15000"):
        execute_b1k_launch(
            launch,
            runner=lambda *_args, **_kwargs: type("Result", (), {"returncode": 0, "stdout": "completed global_step=14999", "stderr": ""})(),
        )


def test_live_launch_publishes_only_stable_native_checkpoints_before_a_later_failure(tmp_path: Path) -> None:
    output = tmp_path / "output"
    complete = output / "checkpoint-1000"
    complete.mkdir(parents=True)
    (complete / "trainer_state.json").write_text(json.dumps({"global_step": 1000}))
    (complete / "config.json").write_text("{}")
    for name in ("model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (complete / name).write_bytes(b"complete")
    incomplete = output / "checkpoint-2000"
    incomplete.mkdir()
    (incomplete / "trainer_state.json").write_text(json.dumps({"global_step": 2000}))
    published: list[int] = []

    class Process:
        def __init__(self) -> None: self.polls = 0
        def poll(self) -> int | None:
            self.polls += 1
            return None if self.polls < 3 else 1
        def wait(self) -> int: return 1

    with pytest.raises(B1KLaunchFailure, match="torchrun failed") as failure:
        run_b1k_launch_with_checkpoint_watch(
            B1KLaunch(command=("torchrun", "train.py"), environment={}, arguments_sha256="a" * 64),
            output_dir=output,
            world_size=1,
            on_stable_checkpoint=published.append,
            popen_factory=lambda *_args, **_kwargs: Process(),
            sleep=lambda _seconds: None,
        )
    assert failure.value.optimizer_step == 2_000
    assert published == [1_000]


def _native_checkpoint(root: Path, step: int) -> None:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    (checkpoint / "config.json").write_text("{}")
    for name in ("model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (checkpoint / name).write_bytes(b"complete")


def test_live_launch_skips_restored_checkpoint_and_stabilizes_final_checkpoint_after_exit(tmp_path: Path) -> None:
    output = tmp_path / "output"
    _native_checkpoint(output, 14_000)
    _native_checkpoint(output, 15_000)
    published: list[int] = []

    class Process:
        pid = 1234
        def poll(self) -> int: return 0
        def wait(self, timeout: float | None = None) -> int: return 0

    assert run_b1k_launch_with_checkpoint_watch(
        B1KLaunch(command=("torchrun", "train.py"), environment={}, arguments_sha256="a" * 64),
        output_dir=output,
        world_size=1,
        on_stable_checkpoint=published.append,
        resume_floor=14_000,
        published_seed=(14_000,),
        popen_factory=lambda *_args, **_kwargs: Process(),
        sleep=lambda _seconds: None,
    ) == 15_000
    assert published == [15_000]


def test_live_launch_callback_failure_terminates_and_reaps_the_process_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "output"
    _native_checkpoint(output, 1_000)
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr("lehome_train.b1k.launch.os.killpg", lambda pid, signal: signals.append((pid, signal)))

    class Process:
        pid = 4321
        polls = 0
        waits: list[float | None] = []
        def poll(self) -> int | None:
            self.polls += 1
            return None
        def wait(self, timeout: float | None = None) -> int:
            self.waits.append(timeout)
            return 0

    process = Process()
    with pytest.raises(RuntimeError, match="publisher failed"):
        run_b1k_launch_with_checkpoint_watch(
            B1KLaunch(command=("torchrun", "train.py"), environment={}, arguments_sha256="a" * 64),
            output_dir=output,
            world_size=1,
            on_stable_checkpoint=lambda _step: (_ for _ in ()).throw(RuntimeError("publisher failed")),
            popen_factory=lambda *_args, **_kwargs: process,
            sleep=lambda _seconds: None,
        )
    assert signals
    assert process.waits


@pytest.mark.parametrize("interrupt", [signal.SIGTERM, signal.SIGINT])
def test_live_launch_forwards_exact_signal_reaps_child_restores_handlers_and_persists_sanitized_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupt: int) -> None:
    output = tmp_path / "output"; handlers: dict[int, object] = {}; restored: list[tuple[int, object]] = []; signals: list[tuple[int, int]] = []
    previous = {signal.SIGTERM: object(), signal.SIGINT: object()}
    monkeypatch.setattr("lehome_train.b1k.launch.signal.getsignal", lambda number: previous[number])
    def install(number: int, handler: object) -> None:
        handlers[number] = handler
        if handler is previous[number]: restored.append((number, handler))
    monkeypatch.setattr("lehome_train.b1k.launch.signal.signal", install)
    monkeypatch.setattr("lehome_train.b1k.launch.os.killpg", lambda pid, number: signals.append((pid, number)))

    class Process:
        pid = 999
        def __init__(self, **kwargs: object) -> None:
            kwargs["stdout"].write(b"trainer stdout\n")
            kwargs["stderr"].write(b"HF_TOKEN=super-secret hf_abcdefghijklmnopqrstuvwxyz\n")
        def poll(self) -> None:
            handlers[interrupt](interrupt, None)
            return None
        def wait(self, timeout: float | None = None) -> int: return 0

    with pytest.raises(B1KLaunchFailure, match="interrupted"):
        run_b1k_launch_with_checkpoint_watch(
            B1KLaunch(command=("torchrun", "train.py"), environment={}, arguments_sha256="a" * 64),
            output_dir=output,
            world_size=1,
            on_stable_checkpoint=lambda _step: None,
            popen_factory=lambda *_args, **kwargs: Process(**kwargs),
            sleep=lambda _seconds: None,
        )
    assert (999, interrupt) in signals
    assert {number for number, _handler in restored} == {signal.SIGTERM, signal.SIGINT}
    assert (output / "trainer.stdout.log").read_text() == "trainer stdout\n"
    assert "super-secret" not in (output / "trainer.stderr.log").read_text()
    assert "hf_abcdefghijklmnopqrstuvwxyz" not in (output / "trainer.stderr.log").read_text()
    assert (output / "trainer.stderr.log").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("interrupt", [signal.SIGTERM, signal.SIGINT])
def test_live_launch_bridges_a_signal_received_after_handlers_install_but_before_spawn_returns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupt: int) -> None:
    output = tmp_path / "output"; handlers: dict[int, object] = {}; signals: list[tuple[int, int]] = []
    previous = {signal.SIGTERM: object(), signal.SIGINT: object()}
    monkeypatch.setattr("lehome_train.b1k.launch.signal.getsignal", lambda number: previous[number])
    monkeypatch.setattr("lehome_train.b1k.launch.signal.signal", lambda number, handler: handlers.__setitem__(number, handler))
    monkeypatch.setattr("lehome_train.b1k.launch.os.killpg", lambda pid, number: signals.append((pid, number)))

    class Process:
        pid = 998
        def poll(self) -> None: return None
        def wait(self, timeout: float | None = None) -> int: return 0

    def spawn(*_args: object, **_kwargs: object) -> Process:
        handlers[interrupt](interrupt, None)
        return Process()

    with pytest.raises(B1KLaunchFailure, match=rf"signal {interrupt}") as failure:
        run_b1k_launch_with_checkpoint_watch(
            B1KLaunch(command=("torchrun", "train.py"), environment={}, arguments_sha256="a" * 64),
            output_dir=output,
            world_size=1,
            on_stable_checkpoint=lambda _step: None,
            popen_factory=spawn,
            sleep=lambda _seconds: None,
        )
    assert failure.value.signal_number == interrupt
    assert (998, interrupt) in signals


def test_live_launch_coalesces_a_second_signal_while_reaping_the_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "output"; handlers: dict[int, object] = {}; signals: list[tuple[int, int]] = []
    previous = {signal.SIGTERM: object(), signal.SIGINT: object()}
    monkeypatch.setattr("lehome_train.b1k.launch.signal.getsignal", lambda number: previous[number])
    monkeypatch.setattr("lehome_train.b1k.launch.signal.signal", lambda number, handler: handlers.__setitem__(number, handler))
    monkeypatch.setattr("lehome_train.b1k.launch.os.killpg", lambda pid, number: signals.append((pid, number)))

    class Process:
        pid = 997
        waits = 0
        def poll(self) -> None:
            handlers[signal.SIGTERM](signal.SIGTERM, None)
            return None
        def wait(self, timeout: float | None = None) -> int:
            self.waits += 1
            if self.waits == 1:
                handlers[signal.SIGINT](signal.SIGINT, None)
                raise InterruptedError
            return 0

    with pytest.raises(B1KLaunchFailure, match="signal 15") as failure:
        run_b1k_launch_with_checkpoint_watch(
            B1KLaunch(command=("torchrun", "train.py"), environment={}, arguments_sha256="a" * 64),
            output_dir=output,
            world_size=1,
            on_stable_checkpoint=lambda _step: None,
            popen_factory=lambda *_args, **_kwargs: Process(),
            sleep=lambda _seconds: None,
        )
    assert failure.value.signal_number == signal.SIGTERM
    assert signals == [(997, signal.SIGTERM)]
