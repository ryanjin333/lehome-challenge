from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import threading
import pytest

from lehome.flywheel.artifacts import EpisodeArtifactWriter
from lehome.flywheel.isaac_recorder import CANONICAL_VIDEO_FILENAMES
from lehome.flywheel.matrix import Trial, build_public_matrix, matrix_sha256
from scripts.run_groot_flywheel_campaign import (
    CampaignState,
    _prepare_retry_attempt,
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
    videos = writer.staging / "videos"
    videos.mkdir()
    for filename in CANONICAL_VIDEO_FILENAMES:
        (videos / filename).write_bytes(b"video")
    writer.finalize(
        {"terminal_reason": "horizon", "outcome": "timeout"},
        required_videos=CANONICAL_VIDEO_FILENAMES,
    )
    return CampaignState(output_root=tmp_path, trial_ids=(trial_id, "trial-002"))


def test_campaign_resume_skips_checksum_verified_trials(tmp_path) -> None:
    state = campaign_state_with_completed_trial(tmp_path, "trial-001")
    assert pending_trial_ids(state) == ("trial-002",)


def test_campaign_resume_retries_an_encoder_error_even_when_generic_artifact_hashes_verify(tmp_path) -> None:
    writer = EpisodeArtifactWriter(tmp_path, "trial-001")
    writer.append_annotation({"step": 0, "action_source": "policy"})
    videos = writer.staging / "videos"
    videos.mkdir()
    for filename in CANONICAL_VIDEO_FILENAMES:
        (videos / filename).write_bytes(b"video")
    writer.finalize(
        {"terminal_reason": "success", "outcome": "error", "recorder_error": "encoder failed"},
        required_videos=CANONICAL_VIDEO_FILENAMES,
    )
    state = CampaignState(output_root=tmp_path, trial_ids=("trial-001",))

    assert pending_trial_ids(state) == ("trial-001",)


def test_campaign_resume_retries_a_terminal_artifact_missing_one_canonical_video(tmp_path) -> None:
    writer = EpisodeArtifactWriter(tmp_path, "trial-001")
    writer.append_annotation({"step": 0, "action_source": "policy"})
    videos = writer.staging / "videos"
    videos.mkdir()
    for filename in CANONICAL_VIDEO_FILENAMES[:-1]:
        (videos / filename).write_bytes(b"video")
    writer.finalize(
        {"terminal_reason": "horizon", "outcome": "timeout"},
        required_videos=CANONICAL_VIDEO_FILENAMES[:-1],
    )
    state = CampaignState(output_root=tmp_path, trial_ids=("trial-001",))

    assert pending_trial_ids(state) == ("trial-001",)


def test_campaign_resume_retries_an_artifact_with_an_extra_manifest_listed_video(tmp_path) -> None:
    writer = EpisodeArtifactWriter(tmp_path, "trial-001")
    writer.append_annotation({"step": 0, "action_source": "policy"})
    videos = writer.staging / "videos"
    videos.mkdir()
    for filename in (*CANONICAL_VIDEO_FILENAMES, "debug.mp4"):
        (videos / filename).write_bytes(b"video")
    writer.finalize(
        {"terminal_reason": "horizon", "outcome": "timeout"},
        required_videos=(*CANONICAL_VIDEO_FILENAMES, "debug.mp4"),
    )

    assert pending_trial_ids(CampaignState(output_root=tmp_path, trial_ids=("trial-001",))) == ("trial-001",)


def test_sequential_retry_quarantines_failed_staging_before_reusing_the_trial_id(monkeypatch, tmp_path) -> None:
    trial = Trial("pant_long", "Pant_Long_Seen_0", "seen", 42)
    failed = EpisodeArtifactWriter(tmp_path, trial.trial_id)
    failed.append_annotation({"step": 0, "action_source": "policy"})

    def launch(*_args, **_kwargs):
        replacement = EpisodeArtifactWriter(tmp_path, trial.trial_id)
        replacement.append_annotation({"step": 0, "action_source": "policy"})
        replacement.finalize({"terminal_reason": "horizon", "outcome": "timeout"})
        return _SuccessfulProcess([])

    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.subprocess.Popen", launch)
    assert _run_one_worker(_worker_args(tmp_path), worker_id=1, trial=trial) == 0

    assert (tmp_path / "raw" / trial.trial_id).is_dir()
    assert list((tmp_path / "quarantine").glob(f"{trial.trial_id}.attempt-*/pending"))


def test_retry_preparation_rejects_parent_path_without_moving_any_output(tmp_path) -> None:
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(ValueError, match="path-safe"):
        _prepare_retry_attempt(tmp_path, "..")

    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert not (tmp_path / "quarantine").exists()


def test_symlinked_raw_parent_never_counts_external_complete_episode_or_mutates_it(tmp_path) -> None:
    external = tmp_path / "external"
    state = campaign_state_with_completed_trial(external, "trial-001")
    output = tmp_path / "output"
    output.mkdir()
    (output / "raw").symlink_to(external / "raw", target_is_directory=True)

    with pytest.raises(ValueError, match="raw root is unsafe"):
        pending_trial_ids(CampaignState(output_root=output, trial_ids=("trial-001",)))

    assert (external / "raw" / "trial-001").is_dir()


@pytest.mark.parametrize("parent", (".pending", "raw", "quarantine"))
def test_retry_preparation_rejects_symlinked_campaign_parents(tmp_path, parent) -> None:
    external = tmp_path / "external"
    external.mkdir()
    (tmp_path / parent).symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="unsafe"):
        _prepare_retry_attempt(tmp_path, "trial-001")

    assert not list(external.iterdir())


def test_repeated_retry_preparation_retains_distinct_quarantined_attempts(tmp_path) -> None:
    for _ in range(2):
        failed = EpisodeArtifactWriter(tmp_path, "trial-001")
        failed.append_annotation({"step": 0, "action_source": "policy"})
        _prepare_retry_attempt(tmp_path, "trial-001")

    attempts = sorted((tmp_path / "quarantine").glob("trial-001.attempt-*"))
    assert len(attempts) == 2
    assert all((attempt / "pending" / "annotations.jsonl").is_file() for attempt in attempts)


def test_concurrent_retry_preparation_retains_one_attempt_without_collision_or_overwrite(tmp_path) -> None:
    failed = EpisodeArtifactWriter(tmp_path, "trial-001")
    failed.append_annotation({"step": 0, "action_source": "policy"})
    failures: list[BaseException] = []
    start = threading.Barrier(2)

    def prepare() -> None:
        try:
            start.wait()
            _prepare_retry_attempt(tmp_path, "trial-001")
        except BaseException as error:
            failures.append(error)

    threads = [threading.Thread(target=prepare) for _ in range(2)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()

    assert failures == []
    attempts = list((tmp_path / "quarantine").glob("trial-001.attempt-*"))
    assert len(attempts) == 1
    assert (attempts[0] / "pending" / "annotations.jsonl").is_file()


def test_retry_preparation_fails_closed_on_an_unsafe_attempt_destination(tmp_path) -> None:
    failed = EpisodeArtifactWriter(tmp_path, "trial-001")
    failed.append_annotation({"step": 0, "action_source": "policy"})
    external = tmp_path / "external"
    external.mkdir()
    quarantine = tmp_path / "quarantine"
    quarantine.mkdir()
    (quarantine / "trial-001.attempt-001").symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="attempt path is unsafe"):
        _prepare_retry_attempt(tmp_path, "trial-001")

    assert (tmp_path / ".pending" / "trial-001").is_dir()
    assert not list(external.iterdir())


def test_parallel_retry_quarantines_invalid_raw_before_worker_reuses_the_trial_id(monkeypatch, tmp_path) -> None:
    trial = Trial("pant_long", "Pant_Long_Seen_0", "seen", 42)
    invalid = EpisodeArtifactWriter(tmp_path, trial.trial_id)
    invalid.append_annotation({"step": 0, "action_source": "policy"})
    invalid.finalize({"terminal_reason": "horizon", "outcome": "error"})
    events: list[tuple[str, float | None]] = []

    def launch(*_args, **_kwargs):
        replacement = EpisodeArtifactWriter(tmp_path, trial.trial_id)
        replacement.append_annotation({"step": 0, "action_source": "policy"})
        videos = replacement.staging / "videos"
        videos.mkdir()
        for filename in CANONICAL_VIDEO_FILENAMES:
            (videos / filename).write_bytes(b"video")
        replacement.finalize(
            {"terminal_reason": "horizon", "outcome": "timeout"},
            required_videos=CANONICAL_VIDEO_FILENAMES,
        )
        return _SuccessfulProcess(events)

    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.subprocess.Popen", launch)
    _, completed, failed = _run_worker_group(_worker_args(tmp_path), ((1, trial),))

    assert (completed, failed) == (1, 0)
    assert list((tmp_path / "quarantine").glob(f"{trial.trial_id}.attempt-*/raw"))


def test_worker_group_does_not_count_an_encoder_error_as_completed(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign.subprocess.Popen",
        lambda *args, **kwargs: _SuccessfulProcess([]),
    )
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.is_completed_trial", lambda *_args: False)

    _, completed, failed = _run_worker_group(
        _worker_args(tmp_path), ((1, Trial("pant_long", "Pant_Long_Seen_0", "seen", 42)),)
    )

    assert (completed, failed) == (0, 1)


def test_campaign_forwards_run_provenance_and_matrix_trial_identity(tmp_path) -> None:
    args = argparse.Namespace(
        policy_path=tmp_path / "policy", policy_revision_file=tmp_path / "revision.txt",
        policy_repo="org/policy", policy_step=12000, code_revision="a" * 40,
        asset_revision="b" * 40, simulator_version="isaac-5.1", policy_artifact_sha256="c" * 64,
        image_identity="sha256:immutable", release_assets_root=tmp_path / "asset-checkout" / "Release",
        output_root=tmp_path, max_steps=600, strategy="mild",
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
    assert values["--release-assets-root"] == str(tmp_path / "asset-checkout" / "Release")
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
        "--asset-revision", "b" * 40, "--release-assets-root", str(tmp_path / "assets" / "Release"), "--simulator-version", "isaac-5.1",
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
        "--code-revision", "a" * 40, "--asset-revision", "b" * 40, "--release-assets-root", str(tmp_path / "assets" / "Release"), "--simulator-version", "isaac-5.1",
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
        release_assets_root=tmp_path / "asset-checkout" / "objects" / "Challenge_Garment" / "Release",
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
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.is_completed_trial", lambda *_args: False)
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
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.is_completed_trial", lambda *_args: False)
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
        "--code-revision", "a", "--asset-revision", "b", "--release-assets-root", "assets/Release", "--simulator-version", "isaac-5.1",
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
        "--code-revision", "a", "--asset-revision", "b", "--release-assets-root", "assets/Release", "--simulator-version", "isaac-5.1",
        "--policy-artifact-sha256", "c", "--image-identity", "sha256:image",
    ]
    with pytest.raises(SystemExit):
        build_parser().parse_args([*arguments, "--worker-timeout-seconds", value])
