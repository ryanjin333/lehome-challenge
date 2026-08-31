from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rollout_appliance.native_reference_site.cloth_fidelity import (
    ClothFidelityInvalid,
    install_cloth_fidelity_monitor_on_env,
    measure_cloth_health,
    validate_cloth_fidelity_evidence,
)


class _Attribute:
    def __init__(self, value):
        self.value = value

    def Get(self):
        return self.value


class _Prim:
    def __init__(self, positions, velocities):
        self.values = {"points": _Attribute(positions), "velocities": _Attribute(velocities)}

    def GetAttribute(self, name):
        return self.values.get(name)


def _env(*, positions=None, velocities=None):
    positions = [[0.0, 0.0, 0.2], [0.4, 0.2, 0.3]] if positions is None else positions
    velocities = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]] if velocities is None else velocities
    garment = {
        "scale": [1.0, 1.0, 1.0],
        "soft_reset_pos_range": [-0.5, -0.5, 0.0, 0.5, 0.5, 0.5],
    }
    particle = {"objects": {"common": {}, "particle_system": {"max_velocity": 5.0}}}
    return SimpleNamespace(
        object=SimpleNamespace(_prim=_Prim(positions, velocities)),
        garment_config=garment,
        particle_config=particle,
        cfg=SimpleNamespace(garment_name="Top_Short_Seen_0"),
    )


def test_cloth_health_uses_existing_fixed_scene_scale_and_velocity_semantics() -> None:
    health = measure_cloth_health(_env())
    assert health == {
        "healthy": True,
        "sample_count": 2,
        "max_position_m": 0.4,
        "max_extent_m": 0.4,
        "max_velocity_mps": 0.1,
        "max_position_limit_m": 2.5,
        "max_extent_limit_m": 4.0,
        "max_velocity_limit_mps": 4.75,
        "missing_cloth": False,
        "cloth_flight": False,
        "nonfinite_cloth_state": False,
    }


@pytest.mark.parametrize(
    ("env", "code"),
    [
        (SimpleNamespace(object=None), "missing_cloth"),
        (_env(positions=[[float("nan"), 0.0, 0.0]]), "nonfinite_cloth_state"),
        (_env(positions=[[3.0, 0.0, 0.0]], velocities=[[0.0, 0.0, 0.0]]), "cloth_flight"),
    ],
)
def test_cloth_health_fails_closed_for_each_measured_invalid(env, code: str) -> None:
    with pytest.raises(ClothFidelityInvalid, match=code):
        measure_cloth_health(env)


def test_monitor_records_hash_chained_post_step_and_pre_score_evidence(tmp_path: Path) -> None:
    class Env:
        def __init__(self):
            base = _env()
            self.object = base.object
            self.garment_config = base.garment_config
            self.particle_config = base.particle_config
            self.cfg = base.cfg

        def reset(self):
            return "reset"

        def step(self, action):
            return action

        def _get_success(self):
            return False

    path = tmp_path / "cloth-fidelity.jsonl"
    env = Env()
    install_cloth_fidelity_monitor_on_env(env, path)
    env.reset()
    env.step("action")
    assert env._get_success() is False
    env.reset()
    env.step("action-2")
    assert env._get_success() is False
    env.close()

    summary = validate_cloth_fidelity_evidence(
        path,
        expected_episodes=[
            ("Top_Short_Seen_0", 1),
            ("Top_Short_Seen_0", 2),
        ],
    )
    assert summary["measured_episode_count"] == 2
    assert summary["fidelity_invalid_count"] == 0
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert {
        row["stage"] for row in rows
        if row["kind"] == "lehome_native_cloth_fidelity_event_v1"
    } == {"post_step", "pre_score"}
    assert sum(
        row["kind"] == "lehome_native_cloth_fidelity_terminal_v1" for row in rows
    ) == 2
    assert all(len(row["event_sha256"]) == 64 for row in rows)


def test_monitor_batches_600_step_and_pre_score_checks_to_one_terminal_flush_per_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rollout_appliance.native_reference_site.cloth_fidelity as fidelity

    env = _env()
    env.reset = lambda: None
    env.step = lambda action: action
    env._get_success = lambda: False
    fsync_calls: list[int] = []
    print_calls: list[str] = []
    monkeypatch.setattr(fidelity.os, "fsync", lambda descriptor: fsync_calls.append(descriptor))
    monkeypatch.setattr("builtins.print", lambda value, **kwargs: print_calls.append(str(value)))
    path = tmp_path / "cloth-fidelity.jsonl"
    install_cloth_fidelity_monitor_on_env(env, path)
    for _episode in range(2):
        env.reset()
        for step in range(600):
            env.step(step)
            env._get_success()
        assert len(fsync_calls) == _episode
        assert len(print_calls) == _episode
    env.close()

    summary = validate_cloth_fidelity_evidence(
        path,
        expected_episodes=[("Top_Short_Seen_0", 1), ("Top_Short_Seen_0", 2)],
    )
    assert summary["event_count"] == 2_402
    assert len(fsync_calls) == 2
    assert len(print_calls) == 2
    assert all(value.startswith("LEHOME_CLOTH_FIDELITY_FLUSH ") for value in print_calls)


def test_monitor_flushes_invalid_fidelity_immediately_before_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rollout_appliance.native_reference_site.cloth_fidelity as fidelity

    env = _env()
    env.reset = lambda: None
    env.step = lambda action: action
    env._get_success = lambda: False
    fsync_calls: list[int] = []
    print_calls: list[str] = []
    monkeypatch.setattr(fidelity.os, "fsync", lambda descriptor: fsync_calls.append(descriptor))
    monkeypatch.setattr("builtins.print", lambda value, **kwargs: print_calls.append(str(value)))
    path = tmp_path / "cloth-fidelity.jsonl"
    install_cloth_fidelity_monitor_on_env(env, path)
    env.reset()
    env.step("healthy")
    env.object._prim.values["velocities"].value = [[9.0, 0.0, 0.0], [9.0, 0.0, 0.0]]

    with pytest.raises(ClothFidelityInvalid, match="cloth_flight"):
        env.step("invalid")

    assert len(fsync_calls) == 1
    assert len(print_calls) == 1
    summary = validate_cloth_fidelity_evidence(
        path, expected_episodes=[("Top_Short_Seen_0", 1)]
    )
    assert summary["measured_episode_count"] == 1
    assert summary["fidelity_invalid_count"] == 1


def test_monitor_process_exit_callback_flushes_final_episode_with_terminal_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rollout_appliance.native_reference_site.cloth_fidelity as fidelity

    env = _env()
    env.reset = lambda: None
    env.step = lambda action: action
    env._get_success = lambda: False
    callbacks: list[object] = []
    monkeypatch.setattr(fidelity.atexit, "register", lambda callback: callbacks.append(callback))
    monkeypatch.setattr(fidelity.atexit, "unregister", lambda callback: None)
    path = tmp_path / "cloth-fidelity.jsonl"
    install_cloth_fidelity_monitor_on_env(env, path)
    env.reset()
    env.step(None)
    env._get_success()

    assert path.read_bytes() == b""
    with pytest.raises(ValueError, match="empty"):
        validate_cloth_fidelity_evidence(
            path, expected_episodes=[("Top_Short_Seen_0", 1)]
        )
    assert len(callbacks) == 1
    callbacks[0]()

    summary = validate_cloth_fidelity_evidence(
        path, expected_episodes=[("Top_Short_Seen_0", 1)]
    )
    assert summary["measured_episode_count"] == 1
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["kind"] == "lehome_native_cloth_fidelity_terminal_v1"
    assert rows[-1]["status"] == "healthy"


def test_evidence_validator_rejects_tampered_chain(tmp_path: Path) -> None:
    path = tmp_path / "cloth-fidelity.jsonl"
    env = _env()
    env.reset = lambda: None
    env.step = lambda action: action
    env._get_success = lambda: False
    install_cloth_fidelity_monitor_on_env(env, path)
    env.reset(); env.step(None); env._get_success()
    env.close()
    rows = path.read_text(encoding="utf-8").splitlines()
    value = json.loads(rows[-2]); value["health"]["max_position_m"] = 0.5
    rows[-2] = json.dumps(value, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash chain"):
        validate_cloth_fidelity_evidence(
            path, expected_episodes=[("Top_Short_Seen_0", 1)]
        )
