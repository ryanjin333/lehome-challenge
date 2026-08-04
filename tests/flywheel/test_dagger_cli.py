from __future__ import annotations

import argparse
import stat
from dataclasses import dataclass
import threading
import time

import numpy as np
import pytest

from lehome.flywheel.bridge_receiver import ExpertCommand, Handshake, LeaderSampleFrame
from lehome.flywheel.isaac_recorder import MixedSourceRecorder
from lehome.flywheel.models import EpisodeIdentity
from lehome.flywheel.quality import QualityThresholds
import scripts.collect_groot_dagger as dagger_module
from scripts.collect_groot_dagger import ScheduledControlSource, build_parser, collect_episode, main, prepare_session, validate_args


def arguments(*values: str):
    return build_parser().parse_args(list(values))


def test_collector_defaults_to_practice_and_loopback() -> None:
    args = arguments()
    assert args.mode == "practice"
    assert args.listen_host == "127.0.0.1"
    assert args.enable_training_output is False


def test_training_output_requires_mode_and_pinned_threshold_manifest(tmp_path) -> None:
    args = arguments(
        "--mode", "dagger", "--enable-training-output", "--quality-thresholds", str(tmp_path / "missing.json"),
        "--organizer-dataset-revision", "a" * 40, "--organizer-dataset-sha256", "b" * 64,
    )
    with pytest.raises(ValueError, match="quality thresholds"):
        validate_args(args)
    with pytest.raises(ValueError, match="loopback"):
        validate_args(arguments("--listen-host", "0.0.0.0"))


def test_interactive_collection_requires_known_calibration_hashes() -> None:
    with pytest.raises(ValueError, match="calibration"):
        validate_args(arguments("--interactive"))


def test_interactive_validation_rejects_before_any_isaac_launcher_import() -> None:
    with pytest.raises(ValueError, match="pinned policy revision"):
        validate_args(arguments("--interactive", "--left-calibration-sha256", "a" * 64, "--right-calibration-sha256", "b" * 64))


def test_practice_session_has_no_export_and_secret_is_mode_0600(tmp_path) -> None:
    args = arguments("--run-root", str(tmp_path / "practice"))
    session = prepare_session(args, secret_path=tmp_path / "bridge-session.secret")
    assert session.controller.mode == "practice"
    assert not (args.run_root / "exports").exists()
    assert stat.S_IMODE((tmp_path / "bridge-session.secret").stat().st_mode) == 0o600


def test_session_cleanup_removes_only_its_owned_secret(tmp_path) -> None:
    secret_path = tmp_path / "bridge-session.secret"
    session = prepare_session(arguments("--run-root", str(tmp_path / "practice")), secret_path=secret_path)

    session.close_listener()
    session.close_listener()

    assert not secret_path.exists()


def test_session_cleanup_refuses_to_delete_a_replaced_secret_path(tmp_path) -> None:
    secret_path = tmp_path / "bridge-session.secret"
    session = prepare_session(arguments("--run-root", str(tmp_path / "practice")), secret_path=secret_path)
    secret_path.unlink()
    secret_path.write_bytes(b"unrelated")
    secret_path.chmod(0o600)

    with pytest.raises(RuntimeError, match="not created by this session"):
        session.close_listener()

    assert secret_path.read_bytes() == b"unrelated"


def test_prepare_session_removes_its_secret_when_manifest_creation_fails(tmp_path) -> None:
    root = tmp_path / "existing"
    root.mkdir()
    (root / "session-manifest.json").write_text("{}", encoding="utf-8")
    secret_path = tmp_path / "bridge-session.secret"

    with pytest.raises(ValueError, match="manifest"):
        prepare_session(arguments("--run-root", str(root)), secret_path=secret_path)

    assert not secret_path.exists()


def test_prepare_session_never_removes_a_replaced_secret_on_the_error_exit(tmp_path, monkeypatch) -> None:
    secret_path = tmp_path / "bridge-session.secret"

    def fail_after_replacement(*_args, **_kwargs) -> None:
        secret_path.unlink()
        secret_path.write_bytes(b"unrelated")
        secret_path.chmod(0o600)
        raise ValueError("manifest failed after replacement")

    monkeypatch.setattr("scripts.collect_groot_dagger._write_session_manifest", fail_after_replacement)

    with pytest.raises(RuntimeError, match="not created by this session"):
        prepare_session(arguments("--run-root", str(tmp_path / "practice")), secret_path=secret_path)

    assert secret_path.read_bytes() == b"unrelated"


def test_noninteractive_main_always_removes_its_one_session_secret(tmp_path, monkeypatch) -> None:
    secret_path = tmp_path / "bridge-session.secret"
    monkeypatch.setattr("scripts.collect_groot_dagger.default_secret_path", lambda: secret_path)

    assert main(["--run-root", str(tmp_path / "practice")]) == 0
    assert not secret_path.exists()


def test_production_collection_rejects_identity_after_app_launch_before_environment_or_recorder(monkeypatch, tmp_path) -> None:
    events: list[str] = []
    app_module = type("AppModule", (), {})()

    class AppLauncher:
        @staticmethod
        def add_app_launcher_args(_parser) -> None:
            return None

    app_module.AppLauncher = AppLauncher
    common = type("Common", (), {
        "launch_app_from_args": staticmethod(lambda _args: events.append("launch") or object()),
        "close_app": staticmethod(lambda _app: events.append("close")),
    })()
    eval_policy = type("PolicyRegistryModule", (), {"PolicyRegistry": object})()
    groot_policy = type("GrootPolicyModule", (), {})()
    import types
    import scripts.run_groot_flywheel_trial as trial_module

    utils = types.ModuleType("scripts.utils")
    utils.__path__ = []
    monkeypatch.setitem(__import__("sys").modules, "isaaclab", types.ModuleType("isaaclab"))
    monkeypatch.setitem(__import__("sys").modules, "isaaclab.app", app_module)
    monkeypatch.setitem(__import__("sys").modules, "scripts.eval_policy", eval_policy)
    monkeypatch.setitem(__import__("sys").modules, "scripts.eval_policy.groot_policy", groot_policy)
    monkeypatch.setitem(__import__("sys").modules, "scripts.utils", utils)
    monkeypatch.setitem(__import__("sys").modules, "scripts.utils.common", common)
    monkeypatch.setattr(trial_module, "_production_env", lambda _args: (_ for _ in ()).throw(AssertionError("environment constructed")))
    monkeypatch.setattr(trial_module, "_validate_live_runtime_identity", lambda *_args, **_kwargs: (events.append("gate"), (_ for _ in ()).throw(ValueError("identity mismatch")))[1])

    with pytest.raises(ValueError, match="identity mismatch"):
        dagger_module._run_production_collection(argparse.Namespace(), object())

    assert events == ["launch", "gate", "close"]


def test_collection_waits_for_an_authenticated_healthy_bridge_before_starting(tmp_path) -> None:
    value = prepare_session(arguments("--run-root", str(tmp_path / "practice")), secret_path=tmp_path / "bridge-session.secret")
    receiver = value.bridge_server.receiver
    receiver.converter = lambda values: values

    def connect_after_setup_delay() -> None:
        time.sleep(0.05)
        receiver.accept_handshake(
            Handshake("nonce", 0, "left", "right", "a" * 64, "b" * 64, ((0.0, 4095.0),) * 6, ((0.0, 4095.0),) * 6, 30)
        )
        receiver.accept_sample(
            LeaderSampleFrame("nonce", 1, 1, (0.0,) * 12, (2048.0,) * 12, 1_000_000, 0)
        )

    worker = threading.Thread(target=connect_after_setup_delay)
    worker.start()
    try:
        value.wait_for_bridge_ready(poll_interval_s=0.005)
    finally:
        worker.join(timeout=1.0)
        value.close_listener()

    assert receiver.current().eligible is False  # close is fail-closed after readiness


def test_listener_runtime_failure_unblocks_bridge_readiness(tmp_path, monkeypatch) -> None:
    value = prepare_session(arguments("--run-root", str(tmp_path / "practice")), secret_path=tmp_path / "bridge-session.secret")
    serve_started = threading.Event()
    readiness_done = threading.Event()
    readiness_errors: list[BaseException] = []

    def fail_runtime() -> None:
        serve_started.set()
        raise RuntimeError("unavailable verifier")

    def wait_for_ready() -> None:
        try:
            value.wait_for_bridge_ready(poll_interval_s=0.001)
        except BaseException as error:
            readiness_errors.append(error)
        finally:
            readiness_done.set()

    monkeypatch.setattr(value.bridge_server, "start", lambda: None)
    monkeypatch.setattr(value.bridge_server, "serve_one_client", fail_runtime)
    assert value.bridge_server.failure is None
    value.start_listener()
    waiter = threading.Thread(target=wait_for_ready)
    waiter.start()
    try:
        assert serve_started.wait(timeout=1.0)
        assert readiness_done.wait(timeout=0.2)
        assert value.bridge_server.failure == "bridge_listener_failed"
        assert len(readiness_errors) == 1
        assert isinstance(readiness_errors[0], RuntimeError)
        assert str(readiness_errors[0]) == "bridge listener failed before collection became ready"
    finally:
        value.close_listener()
        waiter.join(timeout=1.0)


def identity(name: str) -> EpisodeIdentity:
    return EpisodeIdentity(name, "repo", "a" * 40, 1, "b" * 40, "c" * 40, "isaac", "Pant_Long_Seen_0", "pant_long", "seen", 1, "fold the garment", "canonical")


def thresholds() -> QualityThresholds:
    return QualityThresholds("a" * 40, "b" * 64, 1000.0, 1000.0, 10.0, 2000.0, 2000.0, 20.0, 0, 0)


def observation() -> dict[str, np.ndarray]:
    return {
        "observation.state": np.zeros(12, dtype=np.float32),
        "observation.images.top_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation.images.left_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation.images.right_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
    }


class FakeEnv:
    step_dt_s = 1.0 / 30.0

    def __init__(self, *, success_at: int = 2) -> None:
        self.success_at = success_at
        self.actions: list[tuple[float, ...]] = []
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1
        return observation()

    def step(self, action):
        copied = tuple(float(value) for value in action)
        self.actions.append(copied)
        return observation(), 1.0, len(self.actions) >= self.success_at

    def is_action_safe(self, action) -> bool:
        return True

    def flywheel_capture_state(self):
        return {
            "robot_position": np.zeros(12), "robot_velocity": np.zeros(12),
            "cloth_position": np.zeros((1, 3)), "cloth_velocity": np.zeros((1, 3)),
            "rng_state": {"seed": 1}, "garment_name": "Pant_Long_Seen_0",
        }


class FakeClock:
    """Deterministic monotonic-clock seam for collector cadence tests."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, duration_s: float) -> None:
        self.sleeps.append(duration_s)
        self.now += duration_s


@dataclass
class PolicyAction:
    value: np.ndarray
    request_id: str
    chunk_offset: int


class FakePolicy:
    def __init__(self) -> None:
        self.cleared = False

    def select_action_with_provenance(self, observation):
        return PolicyAction(np.full(12, 0.25, dtype=np.float32), "request", 0)

    def clear_queued_actions(self) -> None:
        self.cleared = True


class FakeReceiver:
    def __init__(self, *, eligible: bool = True) -> None:
        self.eligible = eligible
        self.last_safe_command = (0.0,) * 12
        self.jitter_ms = 0.0

    def current(self):
        return ExpertCommand((0.0,) * 12, self.eligible, None if self.eligible else "stale_sample", 7, 1.0)


class ResyncRequiredReceiver:
    def __init__(self) -> None:
        self.last_safe_command = (0.0,) * 12
        self.jitter_ms = 0.0
        self.resync_calls = 0
        self._resynced = False

    def current(self):
        if self._resynced:
            return ExpertCommand((0.0,) * 12, True, None, 7, 1.0)
        return ExpertCommand((0.0,) * 12, False, "resync_required", 7, 1.0)

    def resync(self) -> None:
        assert self.current().reason == "resync_required"
        self.resync_calls += 1
        self._resynced = True


def recorder(tmp_path, name: str, mode: str, monkeypatch) -> MixedSourceRecorder:
    value = MixedSourceRecorder(tmp_path, identity=identity(name), mode=mode, horizon=1)
    def encode(root, *, fps=30):
        directory = root / "videos"; directory.mkdir(exist_ok=True)
        for camera in ("top_rgb", "left_rgb", "right_rgb"):
            (directory / f"{camera}.mp4").write_bytes(b"video")
        return ("top_rgb.mp4", "left_rgb.mp4", "right_rgb.mp4")
    monkeypatch.setattr(value.video_sink, "encode", encode)
    return value


def session(tmp_path, mode: str):
    value = prepare_session(arguments("--mode", mode, "--run-root", str(tmp_path)), secret_path=tmp_path / f"{mode}.secret")
    value.quality_thresholds = thresholds()
    value.bridge_server.receiver = FakeReceiver()
    return value


def test_collect_episode_practice_creates_no_exports_for_repeated_attempts(tmp_path, monkeypatch) -> None:
    for index in range(10):
        root = tmp_path / f"practice-{index}"
        result = collect_episode(session(root, "practice"), FakeEnv(), FakePolicy(), ScheduledControlSource(["space"]), recorder(root, f"practice-{index}", "practice", monkeypatch), thresholds(), max_steps=2)
        assert result.episode["bc_target_count"] == 0
        assert not (root / "exports").exists()


def test_collect_episode_waits_for_delayed_initial_space_without_recording_ready_holds(tmp_path, monkeypatch) -> None:
    value = session(tmp_path, "dagger")
    env, clock = FakeEnv(success_at=1), FakeClock()

    result = collect_episode(
        value,
        env,
        FakePolicy(),
        ScheduledControlSource([None, None, "space"]),
        recorder(tmp_path, "delayed-start", "dagger", monkeypatch),
        thresholds(),
        max_steps=1,
        clock=clock.monotonic,
        sleep=clock.sleep,
        ready_poll_interval_s=0.01,
    )

    assert env.actions == [tuple(np.full(12, 0.25, dtype=np.float32))]
    assert [annotation["action_source"] for annotation in result.annotations] == ["policy"]
    assert clock.sleeps == [0.01, 0.01]


def test_collect_episode_accepts_reset_while_ready_without_stepping(tmp_path, monkeypatch) -> None:
    value = session(tmp_path, "dagger")
    env, clock = FakeEnv(), FakeClock()

    result = collect_episode(
        value,
        env,
        FakePolicy(),
        ScheduledControlSource([None, "r", "space"]),
        recorder(tmp_path, "ready-reset", "dagger", monkeypatch),
        thresholds(),
        max_steps=1,
        clock=clock.monotonic,
        sleep=clock.sleep,
        ready_poll_interval_s=0.01,
    )

    assert env.actions == [tuple(np.full(12, 0.25, dtype=np.float32))]
    assert [annotation["action_source"] for annotation in result.annotations] == ["policy"]
    assert env.reset_calls == 2
    assert clock.sleeps == [0.01]


def test_collect_episode_accepts_discard_while_ready_without_stepping(tmp_path, monkeypatch) -> None:
    value = session(tmp_path, "dagger")
    env, clock = FakeEnv(), FakeClock()

    result = collect_episode(
        value,
        env,
        FakePolicy(),
        ScheduledControlSource([None, "d"]),
        recorder(tmp_path, "ready-discard", "dagger", monkeypatch),
        thresholds(),
        max_steps=1,
        clock=clock.monotonic,
        sleep=clock.sleep,
        ready_poll_interval_s=0.01,
    )

    assert env.actions == []
    assert result.annotations == ()
    assert result.episode["trainable"] is False
    assert clock.sleeps == [0.01]


def test_collect_episode_paces_applied_actions_at_the_environment_deadline(tmp_path, monkeypatch) -> None:
    value = session(tmp_path, "dagger")
    env, clock = FakeEnv(success_at=99), FakeClock()

    collect_episode(
        value,
        env,
        FakePolicy(),
        ScheduledControlSource(["space"]),
        recorder(tmp_path, "paced", "dagger", monkeypatch),
        thresholds(),
        max_steps=3,
        clock=clock.monotonic,
        sleep=clock.sleep,
    )

    assert len(env.actions) == 3
    assert clock.sleeps == [env.step_dt_s, env.step_dt_s]


def test_collect_episode_skips_missed_deadlines_after_an_overrunning_step(tmp_path, monkeypatch) -> None:
    class OverrunningEnv(FakeEnv):
        def __init__(self, clock: FakeClock) -> None:
            super().__init__(success_at=99)
            self.clock = clock
            self.action_start_times: list[float] = []

        def step(self, action):
            self.action_start_times.append(self.clock.now)
            self.clock.now += self.step_dt_s * 1.5
            return super().step(action)

    value, clock = session(tmp_path, "dagger"), FakeClock()
    env = OverrunningEnv(clock)

    collect_episode(
        value,
        env,
        FakePolicy(),
        ScheduledControlSource(["space"]),
        recorder(tmp_path, "overrun", "dagger", monkeypatch),
        thresholds(),
        max_steps=2,
        clock=clock.monotonic,
        sleep=clock.sleep,
    )

    assert env.action_start_times[1] - env.action_start_times[0] >= env.step_dt_s
    assert clock.sleeps == [pytest.approx(env.step_dt_s / 2.0)]


def test_collect_episode_rejects_an_invalid_environment_step_duration(tmp_path, monkeypatch) -> None:
    class InvalidStepDurationEnv(FakeEnv):
        step_dt_s = 0.0

    with pytest.raises(ValueError, match="step_dt_s"):
        collect_episode(
            session(tmp_path, "dagger"),
            InvalidStepDurationEnv(),
            FakePolicy(),
            ScheduledControlSource(["space"]),
            recorder(tmp_path, "invalid-step-duration", "dagger", monkeypatch),
            thresholds(),
            max_steps=1,
        )


def test_collect_episode_full_expert_exports_exact_applied_expert_actions(tmp_path, monkeypatch) -> None:
    value = session(tmp_path, "expert")
    env, policy = FakeEnv(success_at=1), FakePolicy()
    result = collect_episode(value, env, policy, ScheduledControlSource(["space", None, "a"]), recorder(tmp_path, "expert", "expert", monkeypatch), thresholds(), max_steps=3)
    assert result.episode["bc_target_count"] > 0
    assert all(annotation["action_source"] == "expert" for annotation in result.annotations)
    assert all(tuple(annotation["action"]) == action for annotation, action in zip(result.annotations, env.actions))
    assert (tmp_path / "exports" / "expert" / "selection-report.json").is_file()


def test_collect_episode_dagger_exports_only_post_takeover_and_clears_policy(tmp_path, monkeypatch) -> None:
    value = session(tmp_path, "dagger")
    env, policy = FakeEnv(success_at=2), FakePolicy()
    result = collect_episode(value, env, policy, ScheduledControlSource(["space", "space", "a"]), recorder(tmp_path, "dagger", "dagger", monkeypatch), thresholds(), max_steps=3)
    assert result.annotations[0]["action_source"] == "policy"
    assert all(annotation["action_source"] == "expert" for annotation in result.annotations[1:])
    assert [tuple(annotation["action"]) for annotation in result.annotations] == env.actions
    assert policy.cleared is True
    assert [window.observation_step for window in result.expert_windows] == [1, 2]
    assert (tmp_path / "exports" / "dagger" / "expert-windows.json").is_file()


def test_collect_episode_dagger_requires_a_later_space_to_resync_before_takeover(tmp_path, monkeypatch) -> None:
    value = session(tmp_path, "dagger")
    receiver = ResyncRequiredReceiver()
    value.bridge_server.receiver = receiver
    env, policy = FakeEnv(success_at=2), FakePolicy()

    result = collect_episode(
        value,
        env,
        policy,
        ScheduledControlSource(["space", "space", "a"]),
        recorder(tmp_path, "dagger-resync", "dagger", monkeypatch),
        thresholds(),
        max_steps=3,
    )

    assert receiver.resync_calls == 1
    assert result.annotations[0]["action_source"] == "policy"
    assert all(annotation["action_source"] == "expert" for annotation in result.annotations[1:])
    assert policy.cleared is True
    assert [window.observation_step for window in result.expert_windows] == [1, 2]
    assert (tmp_path / "exports" / "dagger-resync" / "selection-report.json").is_file()


def test_collect_episode_hold_or_discard_stays_diagnostic_without_export(tmp_path, monkeypatch) -> None:
    value = session(tmp_path, "expert")
    value.bridge_server.receiver = FakeReceiver(eligible=False)
    result = collect_episode(value, FakeEnv(), FakePolicy(), ScheduledControlSource(["space", "d"]), recorder(tmp_path, "hold", "expert", monkeypatch), thresholds(), max_steps=2)
    assert result.episode["bc_target_count"] == 0
    assert not (tmp_path / "exports").exists()


def test_collect_episode_never_steps_an_unsafe_action_or_exports_it(tmp_path, monkeypatch) -> None:
    class UnsafeEnv(FakeEnv):
        def is_action_safe(self, action) -> bool:
            return False

    value = session(tmp_path, "expert")
    env = UnsafeEnv(success_at=1)
    result = collect_episode(
        value,
        env,
        FakePolicy(),
        ScheduledControlSource(["space", "a"]),
        recorder(tmp_path, "unsafe", "expert", monkeypatch),
        thresholds(),
        max_steps=3,
    )

    assert env.actions == []
    assert result.episode["trainable"] is False
    assert "unsafe" in result.episode["rejection_reasons"]
    assert not (tmp_path / "exports").exists()


def test_collect_episode_reset_resets_task_and_stays_diagnostic(tmp_path, monkeypatch) -> None:
    value, env = session(tmp_path, "expert"), FakeEnv()
    result = collect_episode(
        value,
        env,
        FakePolicy(),
        ScheduledControlSource(["space", "r"]),
        recorder(tmp_path, "reset", "expert", monkeypatch),
        thresholds(),
        max_steps=3,
    )
    assert env.reset_calls == 2
    assert result.episode["bc_target_count"] == 0
    assert not (tmp_path / "exports").exists()


def test_collect_episode_without_metric_thresholds_stays_diagnostic(tmp_path, monkeypatch) -> None:
    value = session(tmp_path, "expert")
    result = collect_episode(
        value,
        FakeEnv(success_at=1),
        FakePolicy(),
        ScheduledControlSource(["space", None, "a"]),
        recorder(tmp_path, "ungraded", "expert", monkeypatch),
        None,
        max_steps=3,
    )
    assert result.episode["bc_target_count"] == 0
    assert result.episode["quality_grade"] == "C"
    assert not (tmp_path / "exports").exists()
