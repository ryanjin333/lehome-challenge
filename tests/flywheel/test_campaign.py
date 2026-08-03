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
