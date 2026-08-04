from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import pytest

from lehome.flywheel.artifacts import EpisodeArtifactWriter
from lehome.flywheel.matrix import Trial, build_public_matrix, matrix_sha256
from scripts.run_groot_flywheel_campaign import (
    CampaignState,
    _run_one_worker,
    _run_worker_group,
    _trial_command,
    build_parser,
    pending_trial_ids,
    run_campaign,
)


def campaign_state_with_completed_trial(tmp_path, trial_id: str) -> CampaignState:
    writer = EpisodeArtifactWriter(tmp_path, trial_id)
    writer.append_annotation({"step": 0, "action_source": "policy"})
    writer.finalize({"terminal_reason": "horizon", "outcome": "timeout"})
    return CampaignState(output_root=tmp_path, trial_ids=(trial_id, "trial-002"))


def test_campaign_resume_skips_checksum_verified_trials(tmp_path) -> None:
    state = campaign_state_with_completed_trial(tmp_path, "trial-001")
    assert pending_trial_ids(state) == ("trial-002",)


def test_campaign_forwards_run_provenance_and_matrix_trial_identity(tmp_path) -> None:
    args = argparse.Namespace(
        policy_path=tmp_path / "policy", policy_revision_file=tmp_path / "revision.txt",
        policy_repo="org/policy", policy_step=12000, code_revision="a" * 40,
        asset_revision="b" * 40, simulator_version="isaac-5.1", policy_artifact_sha256="c" * 64,
        image_identity="sha256:immutable", output_root=tmp_path, max_steps=600, strategy="mild",
    )
    trial = Trial("pant_long", "Pant_Long_Seen_0", "seen", 42)
    command = _trial_command(args, trial)
    values = dict(zip(command[3::2], command[4::2], strict=False))
    assert values["--episode-id"] == trial.trial_id
    assert values["--garment"] == "Pant_Long_Seen_0"
    assert values["--category"] == "pant_long"
    assert values["--release-stage"] == "seen"
    assert values["--seed"] == "42"
    assert values["--policy-repo"] == "org/policy"
    assert values["--code-revision"] == "a" * 40
    assert values["--asset-revision"] == "b" * 40
    assert values["--policy-artifact-sha256"] == "c" * 64
    assert values["--strategy"] == "mild"


def test_campaign_missing_provenance_rejects_before_worker_launch(monkeypatch, tmp_path) -> None:
    called = False
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.subprocess.Popen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("spawned")))
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--policy-path", "policy", "--policy-revision-file", "revision", "--output-root", str(tmp_path)])


def test_campaign_uses_the_committed_public_280_trial_matrix(tmp_path) -> None:
    matrix_path = Path(__file__).parents[2] / "configs" / "eval_groot_n17_public_280.json"
    args = build_parser().parse_args([
        "--matrix", str(matrix_path), "--policy-path", "policy", "--policy-revision-file", "revision", "--output-root", str(tmp_path),
        "--policy-repo", "org/policy", "--policy-step", "1", "--code-revision", "a" * 40,
        "--asset-revision", "b" * 40, "--simulator-version", "isaac-5.1",
        "--policy-artifact-sha256", "c" * 64, "--image-identity", "sha256:image", "--dry-run",
    ])

    report = run_campaign(args)

    assert len(report["pending_before"]) == 280
    assert report["pending_before"][0] == "top-long-seen-0-seed-101"
    assert report["matrix"]["sha256"] == matrix_sha256(build_public_matrix())


def test_campaign_rejects_a_noncanonical_matrix_before_worker_launch(monkeypatch, tmp_path) -> None:
    matrix_path = tmp_path / "not-canonical.json"
    matrix_path.write_text("{}\n", encoding="utf-8")
    args = build_parser().parse_args([
        "--matrix", str(matrix_path), "--policy-path", "policy", "--policy-revision-file", "revision",
        "--output-root", str(tmp_path / "output"), "--policy-repo", "org/policy", "--policy-step", "1",
        "--code-revision", "a" * 40, "--asset-revision", "b" * 40, "--simulator-version", "isaac-5.1",
        "--policy-artifact-sha256", "c" * 64, "--image-identity", "sha256:image",
    ])
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign.subprocess.Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("worker launched")),
    )

    with pytest.raises(ValueError, match="canonical public contract"):
        run_campaign(args)


def _worker_args(tmp_path, *, worker_timeout_seconds: float = 2.0, terminate_grace_seconds: float = 0.25) -> argparse.Namespace:
    return argparse.Namespace(
        output_root=tmp_path,
        worker_timeout_seconds=worker_timeout_seconds,
        terminate_grace_seconds=terminate_grace_seconds,
        policy_path=tmp_path / "policy",
        policy_revision_file=tmp_path / "revision.txt",
        policy_repo="org/policy",
        policy_step=12000,
        code_revision="a" * 40,
        asset_revision="b" * 40,
        simulator_version="isaac-5.1",
        policy_artifact_sha256="c" * 64,
        image_identity="sha256:immutable",
        max_steps=600,
        strategy="canonical",
    )


class _TimeoutThenKillProcess:
    def __init__(self, events: list[tuple[str, float | None]]) -> None:
        self.events = events
        self.killed = False

    def wait(self, timeout=None):
        self.events.append(("wait", timeout))
        if self.killed or timeout is None:
            return 0
        raise subprocess.TimeoutExpired("trial", timeout)

    def poll(self):
        self.events.append(("poll", None))
        return None

    def terminate(self) -> None:
        self.events.append(("terminate", None))

    def kill(self) -> None:
        self.events.append(("kill", None))
        self.killed = True


class _SuccessfulProcess:
    def __init__(self, events: list[tuple[str, float | None]]) -> None:
        self.events = events

    def wait(self, timeout=None):
        self.events.append(("wait", timeout))
        return 0

    def poll(self):
        self.events.append(("poll", None))
        return 0

    def terminate(self) -> None:
        raise AssertionError("successful worker must not terminate")

    def kill(self) -> None:
        raise AssertionError("successful worker must not kill")


class _LaunchCleanupProcess:
    def __init__(self, worker_id: int, events: list[tuple[str, int, float | None]]) -> None:
        self.worker_id = worker_id
        self.events = events
        self.killed = False

    def poll(self):
        self.events.append(("poll", self.worker_id, None))
        return 0 if self.killed else None

    def terminate(self) -> None:
        self.events.append(("terminate", self.worker_id, None))

    def kill(self) -> None:
        self.events.append(("kill", self.worker_id, None))
        self.killed = True

    def wait(self, timeout=None):
        self.events.append(("wait", self.worker_id, timeout))
        assert self.killed
        return 0


class _GracefulAfterTerminateProcess:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.terminated = False

    def poll(self):
        self.events.append("poll")
        return 0 if self.terminated else None

    def terminate(self) -> None:
        self.events.append("terminate")
        self.terminated = True

    def kill(self) -> None:
        raise AssertionError("gracefully terminated worker must not be killed")

    def wait(self, timeout=None):
        self.events.append("wait")
        assert self.terminated
        return 0


def _group_trials(count: int) -> tuple[tuple[int, Trial], ...]:
    return tuple(
        (index, Trial("pant_long", f"Pant_Long_Seen_{index}", "seen", 41 + index))
        for index in range(1, count + 1)
    )


def test_worker_group_cleans_up_started_workers_when_later_log_open_fails(monkeypatch, tmp_path) -> None:
    events: list[tuple[str, int, float | None]] = []
    opened = []
    real_open = Path.open
    clock = [0.0]
    open_count = 0

    def open_log(path, *args, **kwargs):
        if (args and args[0] == "x") or kwargs.get("mode") == "x":
            nonlocal open_count
            open_count += 1
            if open_count == 2:
                raise OSError("log open failure")
        log = real_open(path, *args, **kwargs)
        if (args and args[0] == "x") or kwargs.get("mode") == "x":
            opened.append(log)
        return log

    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.Path.open", open_log)
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.time.sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign.subprocess.Popen",
        lambda *args, **kwargs: _LaunchCleanupProcess(1, events),
    )

    with pytest.raises(OSError, match="log open failure"):
        _run_worker_group(_worker_args(tmp_path), _group_trials(2))

    assert [(event, worker) for event, worker, _ in events if event in {"terminate", "kill"}] == [
        ("terminate", 1), ("kill", 1),
    ]
    assert [timeout for event, _, timeout in events if event == "wait"] == [0.25]
    assert all(log.closed for log in opened)


def test_worker_group_reaps_a_partial_launch_worker_that_exits_after_terminate(monkeypatch, tmp_path) -> None:
    events: list[str] = []
    clock = [0.0]
    launch_count = 0

    def launch(*args, **kwargs):
        nonlocal launch_count
        launch_count += 1
        if launch_count == 2:
            raise OSError("popen failure")
        return _GracefulAfterTerminateProcess(events)

    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.time.sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.subprocess.Popen", launch)

    with pytest.raises(OSError, match="popen failure"):
        _run_worker_group(_worker_args(tmp_path), _group_trials(2))

    assert events == ["poll", "terminate", "poll", "wait"]


def test_worker_group_cleans_up_and_closes_every_log_when_later_popen_fails(monkeypatch, tmp_path) -> None:
    events: list[tuple[str, int, float | None]] = []
    opened = []
    real_open = Path.open
    clock = [0.0]
    launch_count = 0

    def open_log(path, *args, **kwargs):
        log = real_open(path, *args, **kwargs)
        opened.append(log)
        return log

    def launch(*args, **kwargs):
        nonlocal launch_count
        launch_count += 1
        if launch_count == 3:
            raise OSError("popen failure")
        return _LaunchCleanupProcess(launch_count, events)

    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.Path.open", open_log)
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.time.sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.subprocess.Popen", launch)

    with pytest.raises(OSError, match="popen failure"):
        _run_worker_group(_worker_args(tmp_path), _group_trials(3))

    assert [(event, worker) for event, worker, _ in events if event in {"terminate", "kill"}] == [
        ("terminate", 1), ("terminate", 2), ("kill", 1), ("kill", 2),
    ]
    assert [timeout for event, _, timeout in events if event == "wait"] == [0.25, 0.25]
    assert clock[0] == pytest.approx(0.25)
    assert all(log.closed for log in opened)


def test_single_worker_uses_configured_shutdown_grace_only_after_main_timeout(monkeypatch, tmp_path) -> None:
    events: list[tuple[str, float | None]] = []
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign.subprocess.Popen",
        lambda *args, **kwargs: (events.append(("launch", None)), _TimeoutThenKillProcess(events))[1],
    )
    trial = Trial("pant_long", "Pant_Long_Seen_0", "seen", 42)

    assert _run_one_worker(_worker_args(tmp_path), worker_id=1, trial=trial) == 124
    assert events == [("launch", None), ("wait", 2.0), ("terminate", None), ("wait", 0.25), ("kill", None), ("wait", None)]


def test_worker_group_launches_immediately_and_uses_configured_shutdown_grace(monkeypatch, tmp_path) -> None:
    events: list[tuple[str, float | None]] = []
    clock = [0.0]
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign.time.sleep",
        lambda seconds: (events.append(("sleep", seconds)), clock.__setitem__(0, clock[0] + seconds)),
    )
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign.subprocess.Popen",
        lambda *args, **kwargs: (events.append(("launch", None)), _TimeoutThenKillProcess(events))[1],
    )
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.verify_episode", lambda path: {"terminal_reason": "horizon"})
    first_trial = Trial("pant_long", "Pant_Long_Seen_0", "seen", 42)
    second_trial = Trial("pant_long", "Pant_Long_Seen_1", "seen", 43)

    _, completed, failed = _run_worker_group(_worker_args(tmp_path), ((1, first_trial), (2, second_trial)))
    assert (completed, failed) == (0, 2)
    assert events[:2] == [("launch", None), ("launch", None)]
    assert events.count(("terminate", None)) == 2
    assert events.count(("kill", None)) == 2
    assert events.count(("wait", 0.25)) == 2


def test_worker_group_uses_one_launch_relative_deadline(monkeypatch, tmp_path) -> None:
    events: list[tuple[str, float | None]] = []
    clock = [10.0]
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign.time.sleep",
        lambda seconds: (events.append(("sleep", seconds)), clock.__setitem__(0, clock[0] + seconds)),
    )
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign.subprocess.Popen",
        lambda *args, **kwargs: (events.append(("launch", None)), _TimeoutThenKillProcess(events))[1],
    )
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.verify_episode", lambda path: {"terminal_reason": "horizon"})
    trials = (
        (1, Trial("pant_long", "Pant_Long_Seen_0", "seen", 42)),
        (2, Trial("pant_long", "Pant_Long_Seen_1", "seen", 43)),
    )

    _run_worker_group(_worker_args(tmp_path, worker_timeout_seconds=2.0), trials)

    assert events[:2] == [("launch", None), ("launch", None)]
    assert not any(event[0] == "wait" and event[1] == 2.0 for event in events)


def test_successful_worker_never_waits_for_shutdown_grace(monkeypatch, tmp_path) -> None:
    events: list[tuple[str, float | None]] = []
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign.subprocess.Popen",
        lambda *args, **kwargs: (events.append(("launch", None)), _SuccessfulProcess(events))[1],
    )
    trial = Trial("pant_long", "Pant_Long_Seen_0", "seen", 42)

    assert _run_one_worker(_worker_args(tmp_path), worker_id=1, trial=trial) == 0
    assert events == [("launch", None), ("wait", 2.0)]


@pytest.mark.parametrize("value", ("0", "-1", "nan", "inf"))
def test_terminate_grace_seconds_must_be_finite_and_positive(value: str) -> None:
    arguments = [
        "--matrix", "matrix.json", "--policy-path", "policy", "--policy-revision-file", "revision",
        "--output-root", "output", "--policy-repo", "org/policy", "--policy-step", "1",
        "--code-revision", "a", "--asset-revision", "b", "--simulator-version", "isaac-5.1",
        "--policy-artifact-sha256", "c", "--image-identity", "sha256:image",
    ]
    assert build_parser().parse_args(arguments).terminate_grace_seconds == 5.0
    with pytest.raises(SystemExit):
        build_parser().parse_args([*arguments, "--terminate-grace-seconds", value])


@pytest.mark.parametrize("value", ("0", "-1", "nan", "inf"))
def test_worker_timeout_seconds_must_be_finite_and_positive(value: str) -> None:
    arguments = [
        "--matrix", "matrix.json", "--policy-path", "policy", "--policy-revision-file", "revision",
        "--output-root", "output", "--policy-repo", "org/policy", "--policy-step", "1",
        "--code-revision", "a", "--asset-revision", "b", "--simulator-version", "isaac-5.1",
        "--policy-artifact-sha256", "c", "--image-identity", "sha256:image",
    ]
    with pytest.raises(SystemExit):
        build_parser().parse_args([*arguments, "--worker-timeout-seconds", value])
