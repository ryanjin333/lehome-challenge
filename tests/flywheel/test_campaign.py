from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import threading
import pytest

from lehome.flywheel.artifacts import EpisodeArtifactWriter
from lehome.flywheel.capacity import CapacitySample, choose_worker_count
from lehome.flywheel.isaac_recorder import CANONICAL_VIDEO_FILENAMES
from lehome.flywheel.matrix import Trial, build_public_matrix, matrix_sha256
from scripts.run_groot_flywheel_campaign import (
    CampaignState,
    _failure_classes,
    _PolicyTelemetrySampler,
    _attempt_log_paths,
    _prepare_policy_telemetry_path,
    _prepare_retry_attempt,
    _run_one_worker,
    _run_worker_group,
    _resource_margins,
    _trial_command,
    _validate_sweep,
    _worker_gpu_indices,
    _worker_environment,
    build_parser,
    pending_trial_ids,
    run_campaign,
)


@pytest.fixture(autouse=True)
def _keep_unit_campaigns_host_independent(monkeypatch):
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign.require_isaac_sim_5_1_runtime",
        lambda: None,
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


def test_retry_preparation_closes_parent_and_lock_after_unsafe_leaf(monkeypatch, tmp_path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    pending = tmp_path / ".pending"
    pending.mkdir()
    (pending / "trial-001").symlink_to(external, target_is_directory=True)
    opened: dict[int, str] = {}
    closed: list[str] = []
    real_open, real_close = os.open, os.close

    def track_open(path, *args, **kwargs):
        fd = real_open(path, *args, **kwargs)
        name = os.fspath(path)
        if name in {".campaign.lock", ".pending"}:
            opened[fd] = name
        return fd

    def track_close(fd):
        name = opened.pop(fd, None)
        if name is not None:
            closed.append(name)
        return real_close(fd)

    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.os.open", track_open)
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.os.close", track_close)

    with pytest.raises(ValueError, match="trial path is unsafe"):
        _prepare_retry_attempt(tmp_path, "trial-001")

    assert opened == {}
    assert ".pending" in closed
    assert closed.count(".campaign.lock") == 1


def test_retry_preparation_closes_parent_fds_and_lock_after_second_move_error(monkeypatch, tmp_path) -> None:
    for parent in (".pending", "raw"):
        (tmp_path / parent / "trial-001").mkdir(parents=True)
    opened: dict[int, str] = {}
    closed: list[str] = []
    real_open, real_close, real_rename = os.open, os.close, os.rename

    def track_open(path, *args, **kwargs):
        fd = real_open(path, *args, **kwargs)
        name = os.fspath(path)
        if name in {".campaign.lock", ".pending", "raw"}:
            opened[fd] = name
        return fd

    def track_close(fd):
        name = opened.pop(fd, None)
        if name is not None:
            closed.append(name)
        return real_close(fd)

    moves = 0

    def fail_second_move(*args, **kwargs):
        nonlocal moves
        moves += 1
        if moves == 2:
            raise OSError("second move failed")
        return real_rename(*args, **kwargs)

    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.os.open", track_open)
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.os.close", track_close)
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.os.rename", fail_second_move)

    with pytest.raises(ValueError, match="output storage is unsafe") as error:
        _prepare_retry_attempt(tmp_path, "trial-001")

    assert str(error.value.__cause__) == "second move failed"
    assert moves == 2
    assert opened == {}
    assert ".pending" in closed
    assert "raw" in closed
    assert closed.count(".campaign.lock") == 1


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
        output_root=tmp_path, max_steps=600, strategy="mild", device="cuda:3",
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
    assert values["--device"] == "cuda:3"


def test_campaign_parser_exposes_device_and_forwards_it_to_trial_workers(tmp_path) -> None:
    args = build_parser().parse_args([
        "--matrix", str(Path(__file__).parents[2] / "configs" / "eval_groot_n17_public_280.json"),
        "--policy-path", "policy", "--policy-revision-file", "revision", "--output-root", str(tmp_path),
        "--policy-repo", "org/policy", "--policy-step", "1", "--code-revision", "a" * 40,
        "--asset-revision", "b" * 40, "--release-assets-root", str(tmp_path / "assets" / "Release"),
        "--simulator-version", "isaac-5.1", "--policy-artifact-sha256", "c" * 64,
        "--image-identity", "sha256:image", "--device", "cuda:2", "--dry-run",
    ])

    assert args.device == "cuda:2"
    assert "--device" in _trial_command(args, build_public_matrix().trials[0])


@pytest.mark.parametrize(
    "value",
    ("", "1", "1,2", "2,1,4", "1,2,4,6", "1,2,4,", "1,,2,4", " 1,2,4", "1,2,4 "),
)
def test_capacity_sweep_requires_exact_four_gpu_acceptance_gate(value: str) -> None:
    with pytest.raises(ValueError):
        _validate_sweep(value)


def test_capacity_sweep_accepts_the_full_four_gpu_gate() -> None:
    assert _validate_sweep("1,2,4") == (1, 2, 4)


def test_campaign_forwards_pinned_policy_server_boundary_to_each_trial(tmp_path) -> None:
    args = build_parser().parse_args([
        "--matrix", str(Path(__file__).parents[2] / "configs" / "eval_groot_n17_public_280.json"),
        "--policy-path", "policy", "--policy-revision-file", "revision", "--output-root", str(tmp_path),
        "--policy-repo", "org/policy", "--policy-step", "1", "--code-revision", "a" * 40,
        "--asset-revision", "b" * 40, "--release-assets-root", str(tmp_path / "assets" / "Release"),
        "--simulator-version", "isaac-5.1", "--policy-artifact-sha256", "c" * 64,
        "--image-identity", "sha256:image", "--groot-root", "/workspace/Isaac-GR00T",
        "--groot-revision", "d" * 40, "--groot-python", "/workspace/groot-venv/bin/python",
        "--policy-server-readiness-timeout", "30", "--policy-server-request-timeout", "2.5",
        "--policy-server-termination-grace", "4", "--dry-run",
    ])

    command = _trial_command(args, build_public_matrix().trials[0], policy_server_port=5501, policy_server_log=tmp_path / "server.log")
    values = dict(zip(command[3::2], command[4::2], strict=False))

    assert values["--groot-root"] == "/workspace/Isaac-GR00T"
    assert values["--groot-revision"] == "d" * 40
    assert values["--groot-python"] == "/workspace/groot-venv/bin/python"
    assert values["--policy-server-port"] == "5501"
    assert values["--policy-server-readiness-timeout"] == "30.0"
    assert values["--policy-server-request-timeout"] == "2.5"
    assert values["--policy-server-termination-grace"] == "4.0"
    assert values["--policy-server-log"] == str(tmp_path / "server.log")


def test_isolated_worker_forwards_its_physical_gpu_without_cuda_visibility_remapping(tmp_path, monkeypatch) -> None:
    args = _worker_args(tmp_path)
    args.device = "cuda:4"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    command = _trial_command(args, Trial("pant_long", "Pant_Long_Seen_0", "seen", 42), device="cuda:4")
    environment = _worker_environment(args, 4)

    assert command[command.index("--device") + 1] == "cuda:4"
    assert "CUDA_VISIBLE_DEVICES" not in environment
    assert environment["LEHOME_FLYWHEEL_WORKER_GPU"] == "4"


def test_vram_margin_uses_only_the_assigned_gpu(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign.subprocess.run",
        lambda *_args, **_kwargs: argparse.Namespace(
            returncode=0,
            stdout="0, 900, 1000\n1, 100, 1000\n2, 250, 1000\n",
        ),
    )

    _, combined_margin, legacy_margin = _resource_margins((2,))

    assert combined_margin == pytest.approx(0.25)
    assert legacy_margin == pytest.approx(0.25)


def test_worker_group_assigns_unique_policy_servers_and_attempt_logs(monkeypatch, tmp_path) -> None:
    args = _worker_args(tmp_path)
    args.device = "cuda:2"
    allocated_ports = iter((5511, 5512))
    launches = []

    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign._allocate_loopback_port",
        lambda: next(allocated_ports),
        raising=False,
    )
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign.subprocess.Popen",
        lambda command, **kwargs: (launches.append((command, kwargs)), _SuccessfulProcess([]))[1],
    )
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.is_completed_trial", lambda *_args: True)

    _run_worker_group(args, _group_trials(2), gpu_indices=(2, 3))

    commands = [dict(zip(command[3::2], command[4::2], strict=False)) for command, _ in launches]
    assert [command["--device"] for command in commands] == ["cuda:2", "cuda:3"]
    assert [command["--policy-server-port"] for command in commands] == ["5511", "5512"]
    server_logs = [Path(command["--policy-server-log"]) for command in commands]
    assert len(set(server_logs)) == 2
    assert all(log.name.endswith(".policy-server.log") for log in server_logs)
    assert all("CUDA_VISIBLE_DEVICES" not in kwargs["env"] for _, kwargs in launches)


def test_retry_uses_a_fresh_paired_policy_server_log(tmp_path) -> None:
    worker_root = tmp_path / "workers" / "worker-01"
    worker_root.mkdir(parents=True)
    worker_log, server_log = _attempt_log_paths(worker_root, "trial-001")
    worker_log.write_text("first worker attempt\n", encoding="utf-8")
    server_log.write_text("first server attempt\n", encoding="utf-8")

    retry_worker_log, retry_server_log = _attempt_log_paths(worker_root, "trial-001")

    assert retry_worker_log.name == "trial-001.attempt-002.log"
    assert retry_server_log.name == "trial-001.attempt-002.policy-server.log"


def test_capacity_sweep_accounts_only_for_exact_nonduplicated_wave_trials(monkeypatch, tmp_path) -> None:
    matrix_path = Path(__file__).parents[2] / "configs" / "eval_groot_n17_public_280.json"
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.pending_trial_ids", lambda state: state.trial_ids)
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._run_one_worker", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("sequential trial launched")))
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign._run_worker_group",
        lambda _args, assignments, **_kwargs: (1.0, len(assignments), 0, {worker: 0.1 for worker, _ in assignments}, [{"host_ram_margin": 1.0, "combined_vram_margin": 1.0, "peak_host_ram_bytes": 1, "peak_vram_bytes": 1, "cpu_utilization": 0.1, "run_queue": 1, "inference_latency_seconds": 0.1, "inference_queue_depth": 0}], ()),
    )
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._resource_margins", lambda *_args: (1.0, 1.0, 1.0))
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._worker_gpu_indices", lambda _args, count: tuple(range(count)))
    args = build_parser().parse_args([
        "--matrix", str(matrix_path), "--policy-path", "policy", "--policy-revision-file", "revision", "--output-root", str(tmp_path),
        "--policy-repo", "org/policy", "--policy-step", "1", "--code-revision", "a" * 40,
        "--asset-revision", "b" * 40, "--release-assets-root", str(tmp_path / "assets" / "Release"), "--simulator-version", "isaac-5.1",
        "--policy-artifact-sha256", "c" * 64, "--image-identity", "sha256:image", "--device", "cuda:0",
        "--groot-root", str(tmp_path / "Isaac-GR00T"), "--groot-revision", "d" * 40,
        "--groot-python", str(tmp_path / "groot-venv" / "bin" / "python"),
        "--capacity-sweep", "1,2,4", "--trials-per-worker", "1",
    ])

    report = run_campaign(args)

    expected = [trial.trial_id for trial in build_public_matrix().trials[:7]]
    assert report["episode_accounting"]["sequential_trial_ids"] == []
    assert report["episode_accounting"]["capacity_wave_trial_ids"] == [expected[:1], expected[1:3], expected[3:7]]
    assert report["episode_accounting"]["attempt_count"] == 7
    assert report["episode_accounting"]["attempted_unique_trial_ids"] == sorted(expected)
    assert report["capacity"]["samples"][0]["combined_vram_margin"] == 1.0
    assert "inference_vram_margin" not in report["capacity"]["samples"][0]
    assert "render_vram_margin" not in report["capacity"]["samples"][0]


def test_cpu_capacity_assignment_represents_one_cpu_worker_explicitly(tmp_path) -> None:
    args = _worker_args(tmp_path)
    args.device = "cpu"

    assert _worker_gpu_indices(args, 1) == (None,)


def test_policy_telemetry_sampler_accepts_worker_attributed_evidence(tmp_path) -> None:
    telemetry_path = _prepare_policy_telemetry_path(tmp_path, worker_id=1)
    telemetry_path.write_text(
        '{"latency_seconds":0.2,"queue_depth_after_enqueue":16,"request_id":"abc"}\n',
        encoding="utf-8",
    )

    sample = _PolicyTelemetrySampler({1: telemetry_path}, wave_started_ns=0).sample(final=True)

    assert sample == {
        "inference_latency_seconds": 0.2,
        "inference_queue_depth": 16,
        "policy_evidence_failures": (),
        "policy_evidence_records": (),
    }


def test_policy_telemetry_sampler_consumes_each_appended_record_once(tmp_path) -> None:
    telemetry_path = _prepare_policy_telemetry_path(tmp_path, worker_id=1)
    telemetry_path.write_text(
        '{"latency_seconds":0.2,"queue_depth_after_enqueue":2,"request_id":"first"}\n',
        encoding="utf-8",
    )
    sampler = _PolicyTelemetrySampler({1: telemetry_path}, wave_started_ns=0)

    sampler.sample()
    sampler.sample()
    with telemetry_path.open("a", encoding="utf-8") as handle:
        handle.write('{"latency_seconds":0.3,"queue_depth_after_enqueue":3,"request_id":"second"}\n')
    sample = sampler.sample(final=True)

    assert sampler._latencies == [0.2, 0.3]
    assert sampler._queue_depths == [2, 3]
    assert sample["inference_latency_seconds"] == 0.3
    assert sample["inference_queue_depth"] == 3


@pytest.mark.parametrize(
    ("payload", "started_ns", "expected"),
    (
        ("", 0, "policy_telemetry_missing"),
        ("not-json\n", 0, "policy_telemetry_malformed"),
        ('{"latency_seconds":0.2,"queue_depth_after_enqueue":1,"request_id":"abc"}\n', 2, "policy_telemetry_stale"),
    ),
)
def test_policy_telemetry_sampler_fails_closed_on_missing_malformed_or_stale_evidence(tmp_path, payload, started_ns, expected) -> None:
    telemetry_path = _prepare_policy_telemetry_path(tmp_path, worker_id=1)
    telemetry_path.write_text(payload, encoding="utf-8")
    if started_ns:
        os.utime(telemetry_path, ns=(1, 1))

    sample = _PolicyTelemetrySampler({1: telemetry_path}, wave_started_ns=started_ns).sample(final=True)

    assert sample["policy_evidence_failures"] == (expected,)
    assert sample["inference_latency_seconds"] is None
    assert sample["inference_queue_depth"] is None


def test_policy_telemetry_sampler_rejects_a_path_attributed_to_the_wrong_worker(tmp_path) -> None:
    telemetry_path = _prepare_policy_telemetry_path(tmp_path, worker_id=2)
    telemetry_path.write_text(
        '{"latency_seconds":0.2,"queue_depth_after_enqueue":1,"request_id":"abc"}\n',
        encoding="utf-8",
    )

    sample = _PolicyTelemetrySampler({1: telemetry_path}, wave_started_ns=0).sample(final=True)

    assert sample["policy_evidence_failures"] == ("policy_telemetry_wrong_worker",)


def test_policy_telemetry_paths_are_unique_regular_files_beneath_the_assigned_worker_root(tmp_path) -> None:
    first = _prepare_policy_telemetry_path(tmp_path, worker_id=1)
    second = _prepare_policy_telemetry_path(tmp_path, worker_id=1)

    assert first != second
    assert first.parent == tmp_path / "workers" / "worker-01"
    assert first.stat().st_mode & 0o170000 == 0o100000


def test_policy_telemetry_path_rejects_a_symlinked_workers_root(tmp_path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    (tmp_path / "workers").symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="unsafe"):
        _prepare_policy_telemetry_path(tmp_path, worker_id=1)

    assert not list(external.iterdir())


def test_policy_telemetry_sampler_rejects_replacement_before_its_first_read(tmp_path) -> None:
    provisioned = _prepare_policy_telemetry_path(tmp_path, worker_id=1)
    path = provisioned.path
    path.unlink()
    path.write_text(
        '{"latency_seconds":0.2,"queue_depth_after_enqueue":1,"request_id":"replacement"}\n',
        encoding="utf-8",
    )

    sample = _PolicyTelemetrySampler({1: provisioned}, wave_started_ns=0).sample(final=True)

    assert sample["policy_evidence_records"] == (
        {"worker_id": 1, "failure_class": "policy_telemetry_unsafe_path"},
    )


def test_worker_environment_removes_an_inherited_policy_telemetry_path(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LEHOME_FLYWHEEL_POLICY_TELEMETRY_PATH", "/untrusted/inherited.jsonl")

    environment = _worker_environment(_worker_args(tmp_path), None)

    assert "LEHOME_FLYWHEEL_POLICY_TELEMETRY_PATH" not in environment


def test_campaign_retains_multi_worker_policy_failure_records_counts_and_rejection(monkeypatch, tmp_path) -> None:
    matrix_path = Path(__file__).parents[2] / "configs" / "eval_groot_n17_public_280.json"
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.pending_trial_ids", lambda state: state.trial_ids)
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._worker_gpu_indices", lambda _args, count: tuple(range(count)))

    def sample_for(assignments, policy_records=()):
        return (
            1.0,
            len(assignments),
            0,
            {worker_id: 0.1 for worker_id, _ in assignments},
            [{
                "host_ram_margin": 1.0,
                "inference_vram_margin": 1.0,
                "render_vram_margin": 1.0,
                "peak_host_ram_bytes": 1,
                "peak_vram_bytes": 1,
                "cpu_utilization": 0.1,
                "run_queue": 1,
                "inference_latency_seconds": 0.1,
                "inference_queue_depth": 0,
                "policy_evidence_failures": tuple(record["failure_class"] for record in policy_records),
                "policy_evidence_records": policy_records,
            }],
            (),
        )

    def run_group(_args, assignments, **_kwargs):
        if len(assignments) == 1:
            return sample_for(assignments)
        return sample_for(assignments, (
            {"worker_id": 1, "failure_class": "policy_telemetry_malformed"},
            {"worker_id": 2, "failure_class": "policy_telemetry_missing"},
        ))

    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._run_worker_group", run_group)
    args = build_parser().parse_args([
        "--matrix", str(matrix_path), "--policy-path", "policy", "--policy-revision-file", "revision", "--output-root", str(tmp_path),
        "--policy-repo", "org/policy", "--policy-step", "1", "--code-revision", "a" * 40,
        "--asset-revision", "b" * 40, "--release-assets-root", str(tmp_path / "assets" / "Release"), "--simulator-version", "isaac-5.1",
        "--policy-artifact-sha256", "c" * 64, "--image-identity", "sha256:image",
        "--groot-root", str(tmp_path / "Isaac-GR00T"), "--groot-revision", "d" * 40,
        "--groot-python", str(tmp_path / "groot-venv" / "bin" / "python"), "--capacity-sweep", "1,2,4",
    ])

    report = run_campaign(args)
    second_wave = report["capacity"]["samples"][1]
    decision = choose_worker_count((
        CapacitySample(1, 1.0, 1, 0, 1.0, 1.0, cpu_utilization=0.1, run_queue=1, inference_latency_seconds=0.1, inference_queue_depth=0),
        CapacitySample(2, 1.0, 2, 0, 1.0, 1.0, cpu_utilization=0.1, run_queue=1, inference_latency_seconds=0.1, inference_queue_depth=0, policy_evidence_failures=("policy_telemetry_malformed", "policy_telemetry_missing")),
    ))

    assert second_wave["policy_evidence_records"] == [
        {"worker_id": 1, "failure_class": "policy_telemetry_malformed"},
        {"worker_id": 2, "failure_class": "policy_telemetry_missing"},
    ]
    assert second_wave["worker_failures"] == [
        {"worker_id": 1, "trial_id": build_public_matrix().trials[1].trial_id, "classes": ["policy_telemetry_malformed"]},
        {"worker_id": 2, "trial_id": build_public_matrix().trials[2].trial_id, "classes": ["policy_telemetry_missing"]},
    ]
    assert second_wave["failure_counts"] == {"policy_telemetry_malformed": 1, "policy_telemetry_missing": 1}
    assert report["capacity"]["rejected"][2] == ("policy_telemetry_malformed", "policy_telemetry_missing")
    assert report["capacity"]["requested"] == [1, 2, 4]
    assert len(report["capacity"]["samples"]) == 2
    assert decision.rejected[2] == ("policy_telemetry_malformed", "policy_telemetry_missing")


def test_failure_classification_ignores_benign_subsystem_mentions_and_requires_an_error_line(tmp_path) -> None:
    log_path = tmp_path / "worker.log"
    log_path.write_text("policy loaded with CUDA and Vulkan support\n", encoding="utf-8")

    assert _failure_classes(log_path, returncode=0, progressed=True) == ()

    log_path.write_text("INFO: policy loaded\nERROR: CUDA context initialization failed\n", encoding="utf-8")

    assert _failure_classes(log_path, returncode=1, progressed=True) == ("nonzero_exit", "cuda")


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


def test_campaign_dry_run_scans_completion_once_for_the_280_trial_report(monkeypatch, tmp_path) -> None:
    matrix_path = Path(__file__).parents[2] / "configs" / "eval_groot_n17_public_280.json"
    calls: list[CampaignState] = []

    def pending_once(state: CampaignState) -> tuple[str, ...]:
        calls.append(state)
        return state.trial_ids

    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.pending_trial_ids", pending_once)
    args = build_parser().parse_args([
        "--matrix", str(matrix_path), "--policy-path", "policy", "--policy-revision-file", "revision", "--output-root", str(tmp_path),
        "--policy-repo", "org/policy", "--policy-step", "1", "--code-revision", "a" * 40,
        "--asset-revision", "b" * 40, "--release-assets-root", str(tmp_path / "assets" / "Release"), "--simulator-version", "isaac-5.1",
        "--policy-artifact-sha256", "c" * 64, "--image-identity", "sha256:image", "--dry-run",
    ])

    report = run_campaign(args)

    assert len(calls) == 1
    assert report["completed_after"] == []
    assert report["episode_accounting"]["attempt_count"] == 0
    assert report["episode_accounting"]["sequential_trial_ids"] == []


def test_capacity_sweep_uses_post_sweep_completion_scan_for_completed_after(monkeypatch, tmp_path) -> None:
    matrix_path = Path(__file__).parents[2] / "configs" / "eval_groot_n17_public_280.json"
    pending_results: list[tuple[str, ...]] = []

    def pending_for_phase(state: CampaignState) -> tuple[str, ...]:
        result = state.trial_ids if not pending_results else state.trial_ids[1:]
        pending_results.append(result)
        return result

    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.pending_trial_ids", pending_for_phase)
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._run_one_worker", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("sequential trial launched")))
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._run_worker_group", lambda *_args, **_kwargs: (1.0, 1, 0, {1: 0.1}, [{"host_ram_margin": 1.0, "inference_vram_margin": 1.0, "render_vram_margin": 1.0, "peak_host_ram_bytes": 1, "peak_vram_bytes": 1, "cpu_utilization": 0.1, "run_queue": 1, "inference_latency_seconds": 0.1, "inference_queue_depth": 0}], ()))
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._resource_margins", lambda *_args: (1.0, 1.0, 1.0))
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._worker_gpu_indices", lambda *_args, **_kwargs: (0,))
    args = build_parser().parse_args([
        "--matrix", str(matrix_path), "--policy-path", "policy", "--policy-revision-file", "revision", "--output-root", str(tmp_path),
        "--policy-repo", "org/policy", "--policy-step", "1", "--code-revision", "a" * 40,
        "--asset-revision", "b" * 40, "--release-assets-root", str(tmp_path / "assets" / "Release"), "--simulator-version", "isaac-5.1",
        "--policy-artifact-sha256", "c" * 64, "--image-identity", "sha256:image",
        "--groot-root", str(tmp_path / "Isaac-GR00T"), "--groot-revision", "d" * 40,
        "--groot-python", str(tmp_path / "groot-venv" / "bin" / "python"), "--capacity-sweep", "1,2,4",
    ])

    report = run_campaign(args)

    assert len(pending_results) == 2
    assert report["completed_after"] == [build_public_matrix().trials[0].trial_id]
    assert report["capacity"]["requested"] == [1, 2, 4]
    assert len(report["capacity"]["samples"]) == 2


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


def test_dry_run_skips_the_runtime_preflight_even_when_injected_gate_would_fail(tmp_path) -> None:
    matrix_path = Path(__file__).parents[2] / "configs" / "eval_groot_n17_public_280.json"
    args = build_parser().parse_args([
        "--matrix", str(matrix_path), "--policy-path", "policy", "--policy-revision-file", "revision",
        "--output-root", str(tmp_path / "output"), "--policy-repo", "org/policy", "--policy-step", "1",
        "--code-revision", "a" * 40, "--asset-revision", "b" * 40,
        "--release-assets-root", str(tmp_path / "assets" / "Release"), "--simulator-version", "5.1.0.0",
        "--policy-artifact-sha256", "c" * 64, "--image-identity", "sha256:" + "d" * 64,
        "--dry-run",
    ])

    report = run_campaign(
        args,
        runtime_preflight=lambda: (_ for _ in ()).throw(AssertionError("dry-run preflight called")),
    )

    assert report["pending_before"]


def test_production_campaign_checks_host_before_matrix_or_output_creation(tmp_path) -> None:
    output_root = tmp_path / "output"
    args = build_parser().parse_args([
        "--matrix", str(tmp_path / "would-not-be-read.json"), "--policy-path", "policy",
        "--policy-revision-file", "revision", "--output-root", str(output_root),
        "--policy-repo", "org/policy", "--policy-step", "1", "--code-revision", "a" * 40,
        "--asset-revision", "b" * 40, "--release-assets-root", str(tmp_path / "assets" / "Release"),
        "--simulator-version", "5.1.0.0", "--policy-artifact-sha256", "c" * 64,
        "--image-identity", "sha256:" + "d" * 64, "--groot-root", str(tmp_path / "Isaac-GR00T"),
        "--groot-revision", "e" * 40, "--groot-python", str(tmp_path / "python3.10"),
    ])
    events: list[str] = []

    def preflight() -> None:
        events.append("preflight")
        raise ValueError("incompatible host")

    with pytest.raises(ValueError, match="incompatible host"):
        run_campaign(args, runtime_preflight=preflight)

    assert events == ["preflight"]
    assert not output_root.exists()


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
        device="cuda:0",
        groot_root=tmp_path / "Isaac-GR00T",
        groot_revision="d" * 40,
        groot_python=tmp_path / "groot-venv" / "bin" / "python",
        policy_server_readiness_timeout=30.0,
        policy_server_request_timeout=2.5,
        policy_server_termination_grace=4.0,
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
