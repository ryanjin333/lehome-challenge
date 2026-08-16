from __future__ import annotations

import importlib
from pathlib import Path
import sys
import types


def _cpu_cloth_evidence(env) -> None:
    if not hasattr(env, "renderer_device"):
        env.renderer_device = "cuda:0"
    if not hasattr(env, "camera_device"):
        env.camera_device = env.renderer_device
    env._flywheel_cloth_backend = lambda: "usd"
    env._flywheel_cpu_cloth_state = lambda: ([0.0], [0.0])
    env.flywheel_visible_garment_contact = lambda: {"observed": False}


def _evaluation(monkeypatch):
    """Load the session boundary without installing Isaac or torch."""
    repository = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repository))
    package = types.ModuleType("scripts.utils")
    package.__path__ = [str(repository / "scripts" / "utils")]
    modules = {
        "scripts.utils": package,
        "gymnasium": types.SimpleNamespace(make=lambda *_args, **_kwargs: None),
        "torch": types.SimpleNamespace(),
        "isaaclab": types.ModuleType("isaaclab"),
        "isaaclab.envs": types.ModuleType("isaaclab.envs"),
        "isaaclab_tasks": types.ModuleType("isaaclab_tasks"),
        "isaaclab_tasks.utils": types.ModuleType("isaaclab_tasks.utils"),
        "scripts.eval_policy": types.ModuleType("scripts.eval_policy"),
        "scripts.eval_policy.base_policy": types.ModuleType("scripts.eval_policy.base_policy"),
        "scripts.utils.eval_utils": types.ModuleType("scripts.utils.eval_utils"),
        "scripts.utils.common": types.ModuleType("scripts.utils.common"),
        "lehome.utils.record": types.ModuleType("lehome.utils.record"),
        "lehome.utils.logger": types.ModuleType("lehome.utils.logger"),
    }
    modules["isaaclab.envs"].DirectRLEnv = object
    modules["isaaclab_tasks.utils"].parse_env_cfg = lambda *_args, **_kwargs: None
    modules["scripts.eval_policy"].PolicyRegistry = object
    modules["scripts.eval_policy.base_policy"].BasePolicy = object
    modules["scripts.utils.eval_utils"].convert_ee_pose_to_joints = lambda *_args, **_kwargs: None
    modules["scripts.utils.eval_utils"].save_videos_from_observations = lambda *_args, **_kwargs: None
    modules["scripts.utils.eval_utils"].calculate_and_print_metrics = lambda *_args, **_kwargs: None
    modules["scripts.utils.common"].stabilize_garment_after_reset = lambda *_args, **_kwargs: None
    modules["lehome.utils.record"].RateLimiter = object
    modules["lehome.utils.record"].get_next_experiment_path_with_gap = lambda *_args, **_kwargs: None
    modules["lehome.utils.record"].append_episode_initial_pose = lambda *_args, **_kwargs: None
    modules["lehome.utils.logger"].get_logger = lambda *_args, **_kwargs: types.SimpleNamespace(info=lambda *_args: None)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "scripts.utils.evaluation", raising=False)
    return importlib.import_module("scripts.utils.evaluation")


def test_evaluation_session_switches_garments_without_recreating_a_switchable_environment(monkeypatch) -> None:
    evaluation = _evaluation(monkeypatch)

    calls: list[object] = []

    class Environment:
        cfg = types.SimpleNamespace(garment_name="Top_Long_Seen_0", garment_version="Release", seed=None, random_seed=None)

        def switch_garment(self, name, stage):
            calls.append(("switch", name, stage))
            self.cfg.garment_name = name
            self.cfg.garment_version = stage

        def close(self):
            calls.append("close")

    class Policy:
        def reset(self):
            calls.append("reset")

    args = types.SimpleNamespace(seed=0, num_episodes=1)
    env = Environment()
    session = evaluation.EvaluationSession(args, env=env, policy=Policy(), env_cfg=env.cfg)
    session.prepare_episode(garment_name="Top_Long_Seen_1", garment_stage="Release", seed=42, episode_generation=1)

    assert calls == [("switch", "Top_Long_Seen_1", "Release"), "reset"]
    assert env.cfg.seed == 42
    assert env.cfg.random_seed == 42


def test_cpu_cloth_runtime_receipt_requires_cpu_environment_device(monkeypatch) -> None:
    evaluation = _evaluation(monkeypatch)

    env = types.SimpleNamespace(device="cpu", cfg=types.SimpleNamespace())
    _cpu_cloth_evidence(env)
    args = types.SimpleNamespace(device="cuda:0", renderer_device="cuda:0", camera_device="cuda:0")
    session = evaluation.EvaluationSession(args, env=env, policy=object(), env_cfg=env.cfg)

    assert session.runtime_receipt["simulation_device"] == "cpu"
    assert session.runtime_receipt["cloth_device"] == "cpu"
    assert session.runtime_receipt["renderer_device"] == "cuda:0"


def test_cpu_cloth_runtime_receipt_uses_observed_renderer_and_camera_devices(monkeypatch) -> None:
    evaluation = _evaluation(monkeypatch)

    env = types.SimpleNamespace(
        device="cpu", renderer_device="cuda:2", camera_device="cuda:2", cfg=types.SimpleNamespace(),
    )
    _cpu_cloth_evidence(env)
    args = types.SimpleNamespace(device="cpu", renderer_device="cuda:0", camera_device="cuda:0")
    session = evaluation.EvaluationSession(args, env=env, policy=object(), env_cfg=env.cfg)

    assert session.runtime_receipt["renderer_device"] == "cuda:2"
    assert session.runtime_receipt["camera_device"] == "cuda:2"


def test_evaluation_session_does_not_reset_a_policy_that_the_worker_already_reset(monkeypatch, tmp_path) -> None:
    evaluation = _evaluation(monkeypatch)
    reset_calls: list[str] = []

    class Policy:
        def reset(self):
            reset_calls.append("reset")

    env = types.SimpleNamespace(device="cpu", cfg=types.SimpleNamespace(garment_name="shirt", garment_version="Release"))
    env.close = lambda: None
    args = types.SimpleNamespace(task="task", device="cpu", renderer_device="cuda:0", camera_device="cuda:0")
    session = evaluation.EvaluationSession(args, env=env, policy=Policy(), env_cfg=env.cfg)
    session.prepare_episode(garment_name="shirt", seed=42, episode_generation=1, reset_policy=False)

    assert reset_calls == []


def test_evaluation_session_keeps_legacy_writers_unless_given_an_explicit_attempt_output(monkeypatch, tmp_path) -> None:
    evaluation = _evaluation(monkeypatch)
    captured = []
    monkeypatch.setattr(evaluation, "run_evaluation_loop", lambda **kwargs: captured.append(kwargs["args"]) or [])
    env = types.SimpleNamespace(device="cpu", cfg=types.SimpleNamespace())
    args = types.SimpleNamespace(
        task="task", device="cpu", renderer_device="cuda:0", camera_device="cuda:0",
        video_dir="legacy-videos", eval_dataset_path="legacy-dataset",
    )
    session = evaluation.EvaluationSession(args, env=env, policy=object(), env_cfg=env.cfg)

    session.run_episode(assignment={"garment": "shirt"}, policy=object())
    session.run_episode(
        assignment={"garment": "shirt"}, policy=object(), attempt_output_dir=tmp_path / "attempt",
    )

    assert (captured[0].video_dir, captured[0].eval_dataset_path) == ("legacy-videos", "legacy-dataset")
    assert (captured[1].video_dir, captured[1].eval_dataset_path) == (
        str(tmp_path / "attempt" / "videos"), str(tmp_path / "attempt" / "dataset"),
    )


def test_evaluation_session_explicit_reset_flag_preserves_legacy_per_episode_resets(monkeypatch) -> None:
    evaluation = _evaluation(monkeypatch)
    reset_flags = []
    monkeypatch.setattr(evaluation, "run_evaluation_loop", lambda **kwargs: reset_flags.append(kwargs["reset_policy"]) or [])
    env = types.SimpleNamespace(device="cpu", cfg=types.SimpleNamespace())
    args = types.SimpleNamespace(task="task", video_dir="video", eval_dataset_path="dataset")
    session = evaluation.EvaluationSession(args, env=env, policy=object(), env_cfg=env.cfg)

    session.run_episode(assignment={"garment": "shirt"}, policy=object(), reset_policy=True)
    session.run_episode(assignment={"garment": "shirt"}, policy=object(), reset_policy=False)

    assert reset_flags == [True, False]
