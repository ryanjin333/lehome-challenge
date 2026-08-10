from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import pytest
import scripts.run_groot_flywheel_trial as trial_module

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
    _run_campaign_under_supervisor,
    _run_scale_cpu_canary,
    _scale_cpu_runtime_paths,
    _run_worker_group,
    _resource_margins,
    _trial_command,
    _validate_sweep,
    _worker_gpu_indices,
    _worker_environment,
    _write_invocation_checkpoint,
    _write_json_atomically,
    _episode_gate_evidence,
    _read_parity_receipt,
    _require_cpu_scale_authorization,
    _cpu_scale_live_invocation,
    _may_emit_parity_receipt,
    _validate_cuda_abort_evidence,
    _validate_scale_cpu_production_output,
    _validate_server_cpu_evidence,
    _validate_legacy_cpu_reference,
    _require_scale_parity,
    _sha256_file,
    _validate_legacy_shared_policy_receipt,
    _attempted_gate_evidence,
    _campaign_supervisor_lease,
    _top40_evaluation_invocation,
    _top40_evaluation_metrics,
    _top40_final_bindings,
    _abort_after_first_completed_cohort,
    _verify_or_write_top40_evaluation_invocation,
    _validate_top40_evaluation_output,
    build_legacy_cpu_reference_receipt,
    build_parser,
    main,
    pending_trial_ids,
    run_campaign,
    selected_trials,
)
from lehome.flywheel.parity import EpisodeGateEvidence, HISTORICAL_CONTROL_IDS, assess_reset_diversity, historical_control_trials


@pytest.fixture(autouse=True)
def _keep_unit_campaigns_host_independent(monkeypatch):
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign.require_isaac_sim_5_1_runtime",
        lambda: None,
    )
    def fake_scale_paths(args):
        root = Path(args.trial_runtime_root).resolve()
        return root, (
            root, root / "source" / "lehome",
            root / "third_party" / "IsaacLab" / "source" / "isaaclab",
            root / "third_party" / "IsaacLab" / "source" / "isaaclab_assets",
            root / "third_party" / "IsaacLab" / "source" / "isaaclab_tasks",
            root / "third_party" / "IsaacLab" / "source" / "isaaclab_mimic",
            root / "third_party" / "IsaacLab" / "source" / "isaaclab_rl",
            Path("/opt/isaacsim/python"),
            Path(getattr(args, "groot_root", "/opt/isaac-groot")).resolve(),
        )
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._scale_cpu_runtime_paths", fake_scale_paths)


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


def test_scale_cpu_runtime_paths_accepts_isaac_kit_python_layout(monkeypatch, tmp_path) -> None:
    runtime_root = tmp_path / "trial-runtime"
    for relative_path in (
        "source/lehome",
        "third_party/IsaacLab/source/isaaclab",
        "third_party/IsaacLab/source/isaaclab_assets",
        "third_party/IsaacLab/source/isaaclab_tasks",
        "third_party/IsaacLab/source/isaaclab_mimic",
        "third_party/IsaacLab/source/isaaclab_rl",
    ):
        (runtime_root / relative_path).mkdir(parents=True)

    isaacsim_root = tmp_path / "isaac-sim"
    kit_python = isaacsim_root / "kit" / "python"
    kit_interpreter = kit_python / "bin" / "python3.11"
    kit_interpreter.parent.mkdir(parents=True)
    kit_interpreter.write_text("#!/bin/sh\n")
    kit_interpreter.chmod(0o755)
    groot_python = tmp_path / "workspace" / "lehome-venv" / "bin" / "python"
    groot_python.parent.mkdir(parents=True)
    groot_python.symlink_to(kit_interpreter)
    groot_root = tmp_path / "isaac-groot"
    groot_root.mkdir()
    assert not (isaacsim_root / "python").exists()
    assert groot_python.resolve() == kit_interpreter

    monkeypatch.setenv("ISAACSIM_PATH", str(isaacsim_root))
    resolved_root, trusted_paths = _scale_cpu_runtime_paths(
        argparse.Namespace(trial_runtime_root=runtime_root, groot_root=groot_root)
    )

    assert resolved_root == runtime_root.resolve()
    assert trusted_paths[-2:] == (kit_python.resolve(), groot_root.resolve())


@pytest.mark.parametrize(
    ("groot_root_kind", "message"),
    (("missing", "trusted runtime import roots are incomplete"), ("file", "trusted runtime import roots are incomplete"),
     ("symlink", "trusted runtime import roots are incomplete"), ("undeclared", "requires a declared GR00T checkout")),
)
def test_scale_cpu_runtime_paths_rejects_missing_or_unsafe_groot_checkout(monkeypatch, tmp_path, groot_root_kind, message) -> None:
    runtime_root = tmp_path / "trial-runtime"
    for relative_path in (
        "source/lehome",
        "third_party/IsaacLab/source/isaaclab",
        "third_party/IsaacLab/source/isaaclab_assets",
        "third_party/IsaacLab/source/isaaclab_tasks",
        "third_party/IsaacLab/source/isaaclab_mimic",
        "third_party/IsaacLab/source/isaaclab_rl",
    ):
        (runtime_root / relative_path).mkdir(parents=True)
    isaacsim_root = tmp_path / "isaac-sim"
    (isaacsim_root / "kit" / "python").mkdir(parents=True)
    groot_root = tmp_path / "isaac-groot"
    if groot_root_kind == "file":
        groot_root.write_text("not a checkout", encoding="utf-8")
    elif groot_root_kind == "symlink":
        real_groot_root = tmp_path / "real-isaac-groot"
        real_groot_root.mkdir()
        groot_root.symlink_to(real_groot_root)

    monkeypatch.setenv("ISAACSIM_PATH", str(isaacsim_root))

    args = argparse.Namespace(trial_runtime_root=runtime_root)
    if groot_root_kind != "undeclared":
        args.groot_root = groot_root
    with pytest.raises(ValueError, match=message):
        _scale_cpu_runtime_paths(args)


def _run_supervisor_only(args):
    """Exercise scheduler mechanics without bypassing public scale admission tests."""
    return _run_campaign_under_supervisor(args, build_public_matrix())


def test_campaign_resume_skips_checksum_verified_trials(tmp_path) -> None:
    state = campaign_state_with_completed_trial(tmp_path, "trial-001")
    assert pending_trial_ids(state) == ("trial-002",)


def test_campaign_reads_success_contact_and_reset_evidence_only_from_verified_artifacts(tmp_path) -> None:
    writer = EpisodeArtifactWriter(tmp_path, "trial-001")
    writer.append_annotation({"step": 0, "action_source": "policy"})
    videos = writer.staging / "videos"
    videos.mkdir()
    for filename in CANONICAL_VIDEO_FILENAMES:
        (videos / filename).write_bytes(b"video")
    writer.finalize(
        {
            "terminal_reason": "success",
            "outcome": "success",
            "accepted_success": True,
            "reset_hash": "a" * 64,
            "visible_contact": {
                "observed": True,
                "source": "simulator_particle_to_gripper_distance",
                "minimum_distance_m": 0.01,
            },
        },
        required_videos=CANONICAL_VIDEO_FILENAMES,
    )

    assert _episode_gate_evidence(tmp_path, "trial-001").official_success is True
    assert _episode_gate_evidence(tmp_path, "trial-001").visible_contact is True


def test_campaign_abort_evidence_counts_terminal_attempts_without_artifacts_as_zero_evidence(tmp_path) -> None:
    evidence = _attempted_gate_evidence(tmp_path, tuple(f"trial-{index:03d}" for index in range(12)))

    assert len(evidence) == 12
    assert all(not item.official_success and not item.visible_contact and item.reset_hash is None for item in evidence)


def test_campaign_control_mode_selects_exact_historical_twelve_without_mutating_public_matrix(tmp_path) -> None:
    args = build_parser().parse_args([
        "--matrix", "configs/eval_groot_n17_public_280.json", "--policy-path", "policy", "--policy-revision-file", "revision",
        "--output-root", str(tmp_path), "--policy-repo", "org/policy", "--policy-step", "12000",
        "--code-revision", "a" * 40, "--asset-revision", "b" * 40, "--release-assets-root", str(tmp_path / "assets" / "Release"),
        "--simulator-version", "isaac-5.1", "--policy-artifact-sha256", "c" * 64, "--image-identity", "sha256:image",
        "--historical-control", "--parity-stage", "direct_cpu", "--dry-run",
    ])

    assert len(selected_trials(args, build_public_matrix())) == 12


def test_public_unseen_tops_selects_the_exact_canonical_top_forty_in_source_order() -> None:
    matrix = build_public_matrix()
    args = argparse.Namespace(historical_control=False, public_unseen_tops=True)

    trials = selected_trials(args, matrix)

    assert [trial.trial_id for trial in trials] == [
        trial.trial_id
        for trial in matrix.trials
        if trial.release_stage == "public_unseen" and trial.category in {"top_long", "top_short"}
    ]
    assert len(trials) == 40
    assert {category: sum(trial.category == category for trial in trials) for category in ("top_long", "top_short")} == {
        "top_long": 20,
        "top_short": 20,
    }
    assert len(matrix.trials) == 280


def test_campaign_parser_rejects_public_unseen_tops_with_historical_control(tmp_path) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "--matrix", "configs/eval_groot_n17_public_280.json", "--policy-path", "policy", "--policy-revision-file", "revision",
            "--output-root", str(tmp_path), "--policy-repo", "org/policy", "--policy-step", "12000",
            "--code-revision", "a" * 40, "--asset-revision", "b" * 40, "--release-assets-root", str(tmp_path / "assets" / "Release"),
            "--simulator-version", "isaac-5.1", "--policy-artifact-sha256", "c" * 64, "--image-identity", "sha256:image",
            "--historical-control", "--public-unseen-tops", "--dry-run",
        ])


def test_public_unseen_tops_rejects_the_historical_control_worker_alias(tmp_path) -> None:
    args = build_parser().parse_args([
        "--matrix", "configs/eval_groot_n17_public_280.json", "--policy-path", "policy", "--policy-revision-file", "revision",
        "--output-root", str(tmp_path), "--policy-repo", "org/policy", "--policy-step", "12000",
        "--code-revision", "a" * 40, "--asset-revision", "b" * 40, "--release-assets-root", str(tmp_path / "assets" / "Release"),
        "--simulator-version", "isaac-5.1", "--policy-artifact-sha256", "c" * 64, "--image-identity", "sha256:image",
        "--public-unseen-tops", "--historical-control-workers", "4", "--execution-mode", "policy_server", "--device", "cpu", "--dry-run",
    ])

    with pytest.raises(ValueError, match="cannot use --historical-control-workers"):
        run_campaign(args)


def test_public_unseen_tops_dry_run_is_a_diagnostic_evaluation_report_not_a_scale_release(tmp_path) -> None:
    args = build_parser().parse_args([
        "--matrix", "configs/eval_groot_n17_public_280.json", "--policy-path", "policy", "--policy-revision-file", "revision",
        "--output-root", str(tmp_path), "--policy-repo", "org/policy", "--policy-step", "12000",
        "--code-revision", "a" * 40, "--asset-revision", "b" * 40, "--release-assets-root", str(tmp_path / "assets" / "Release"),
        "--simulator-version", "isaac-5.1", "--policy-artifact-sha256", "c" * 64, "--image-identity", "sha256:image",
        "--public-unseen-tops", "--workers", "4", "--execution-mode", "policy_server", "--device", "cpu", "--dry-run",
    ])

    report = run_campaign(args)

    assert report["selection"] == {
        "kind": "public_unseen_tops_evaluation",
        "classification": "diagnostic_evaluation_only_not_training_or_production_release",
        "rft_data_eligible": False,
        "trial_count": 40,
        "trial_ids": [trial.trial_id for trial in selected_trials(args, build_public_matrix())],
        "category_counts": {"top_long": 20, "top_short": 20},
        "parity_stage": None,
    }


def test_top40_evaluation_invocation_rejects_programmatic_historical_control_collision(tmp_path) -> None:
    args = _worker_args(tmp_path)
    args.historical_control = True
    args.public_unseen_tops = True
    args.workers = 4
    args.execution_mode = "policy_server"
    args.device = "cpu"
    args.policy_device = "cuda:0"
    args.parity_stage = None
    args.matrix = Path(__file__).parents[2] / "configs" / "eval_groot_n17_public_280.json"
    args.trials_per_worker = 1
    args.max_inference_latency_seconds = 0.5
    args.max_inference_queue_depth = 16
    args.early_abort_completed_trials = 12
    args.minimum_reset_uniqueness_ratio = 1.0
    args.historical_control_workers = None
    args.capacity_sweep = None
    args.dry_run = False
    args.max_steps = 600
    args.policy_revision_file.write_text("f" * 40, encoding="utf-8")

    with pytest.raises(ValueError, match="cannot combine --historical-control and --public-unseen-tops"):
        _top40_evaluation_invocation(args, build_public_matrix(), selected_trials(args, build_public_matrix()))


def test_top40_evaluation_resume_rejects_a_foreign_raw_trial_before_launch(tmp_path) -> None:
    args = _worker_args(tmp_path)
    args.historical_control = False
    args.public_unseen_tops = True
    (tmp_path / "raw" / "foreign-trial").mkdir(parents=True)
    trials = tuple(
        trial for trial in build_public_matrix().trials
        if trial.release_stage == "public_unseen" and trial.category in {"top_long", "top_short"}
    )

    with pytest.raises(ValueError, match="extra or foreign trial artifact"):
        _validate_top40_evaluation_output(
            args, CampaignState(tmp_path, tuple(trial.trial_id for trial in trials)), {}, trials,
        )


def test_top40_evaluation_metrics_are_derived_per_category_from_verified_evidence(monkeypatch, tmp_path) -> None:
    trials = tuple(
        trial for trial in build_public_matrix().trials
        if trial.release_stage == "public_unseen" and trial.category in {"top_long", "top_short"}
    )
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign._episode_gate_evidence",
        lambda _root, trial_id: EpisodeGateEvidence(
            official_success=trial_id.endswith("seed-601"), visible_contact=not trial_id.endswith("seed-653"), reset_hash="a" * 64,
        ),
    )

    metrics = _top40_evaluation_metrics(tmp_path, trials)

    assert metrics == {
        "episodes": 40, "official_successes": 4, "success_rate": 0.1, "visible_contact_count": 36,
        "per_category": {
            "top_long": {"episodes": 20, "official_successes": 2, "success_rate": 0.1, "visible_contact_count": 18},
            "top_short": {"episodes": 20, "official_successes": 2, "success_rate": 0.1, "visible_contact_count": 18},
        },
    }


@pytest.mark.parametrize("field", ("policy_revision", "policy_artifact_sha256"))
def test_top40_resume_rejects_a_foreign_invocation_before_worker_scheduling(monkeypatch, tmp_path, field) -> None:
    args = _worker_args(tmp_path)
    args.trials_per_worker = 1
    args.max_inference_latency_seconds = 0.5
    args.max_inference_queue_depth = 16
    args.early_abort_completed_trials = 12
    args.minimum_reset_uniqueness_ratio = 1.0
    args.historical_control_workers = None
    args.capacity_sweep = None
    args.dry_run = False
    args.matrix = Path(__file__).parents[2] / "configs" / "eval_groot_n17_public_280.json"
    args.historical_control = False
    args.public_unseen_tops = True
    args.workers = 4
    args.execution_mode = "policy_server"
    args.device = "cpu"
    args.policy_device = "cuda:0"
    args.parity_stage = None
    args.policy_revision_file.write_text("f" * 40, encoding="utf-8")
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._live_groot_identity", lambda _args: {
        "groot_root": "/groot", "groot_revision": "d" * 40, "groot_python": "/groot/python",
        "groot_python_sha256": "e" * 64, "groot_python_version": "3.10.18",
    })
    trials = tuple(trial for trial in build_public_matrix().trials if trial.release_stage == "public_unseen" and trial.category in {"top_long", "top_short"})
    invocation = _top40_evaluation_invocation(args, build_public_matrix(), trials)
    foreign = dict(invocation)
    foreign[field] = "0" * (40 if field == "policy_revision" else 64)
    _write_json_atomically(tmp_path / "checkpoint-evaluation-invocation.json", foreign)
    scheduled = []
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._run_campaign_under_supervisor", lambda *_args, **_kwargs: scheduled.append(True))

    with pytest.raises(ValueError, match="does not match this resume identity"):
        run_campaign(args)
    assert scheduled == []


def _top40_trials() -> tuple[Trial, ...]:
    return tuple(trial for trial in build_public_matrix().trials if trial.release_stage == "public_unseen" and trial.category in {"top_long", "top_short"})


def test_top40_mixed_completed_episode_and_receipt_provenance_rejects_before_launch(monkeypatch, tmp_path) -> None:
    trials = _top40_trials()
    args = _worker_args(tmp_path)
    args.historical_control = False
    args.public_unseen_tops = True
    trial = trials[0]
    writer = EpisodeArtifactWriter(tmp_path, trial.trial_id)
    writer.append_annotation({"step": 0})
    videos = writer.staging / "videos"; videos.mkdir()
    for name in CANONICAL_VIDEO_FILENAMES: (videos / name).write_bytes(b"v")
    writer.finalize({"terminal_reason": "horizon", "outcome": "success", "accepted_success": True, "identity": {"episode_id": trial.trial_id}, "provenance": {}}, required_videos=CANONICAL_VIDEO_FILENAMES)
    (tmp_path / f"policy-server-receipt-{trial.trial_id}.json").write_text(json.dumps({"episode_id": trial.trial_id, "checkpoint_revision": "foreign"}), encoding="utf-8")
    invocation = {"policy_device_pool": ["cuda:0"], "policy_repo": "org/policy", "policy_revision": "f" * 40, "policy_step": 12000, "code_revision": "a" * 40, "asset_revision": "b" * 40, "simulator_version": "isaac-5.1", "strategy": "canonical", "simulator_device": "cpu", "policy_artifact_sha256": "c" * 64, "image_identity": "sha256:immutable", "groot_revision": "d" * 40, "groot_python": "/python", "groot_python_version": "3.10.18"}
    with pytest.raises(ValueError, match="episode identity"):
        _validate_top40_evaluation_output(args, CampaignState(tmp_path, tuple(item.trial_id for item in trials)), invocation, trials)


def test_top40_clean_close_binds_exactly_forty_manifests_receipts_and_metrics(monkeypatch, tmp_path) -> None:
    trials = _top40_trials(); state = CampaignState(tmp_path, tuple(trial.trial_id for trial in trials))
    for trial in trials:
        episode = tmp_path / "raw" / trial.trial_id; episode.mkdir(parents=True)
        (episode / "SHA256SUMS.json").write_text("{}", encoding="utf-8")
        (tmp_path / f"policy-server-receipt-{trial.trial_id}.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._episode_gate_evidence", lambda _root, trial_id: EpisodeGateEvidence(trial_id.endswith("601"), True, "a" * 64))
    close = _top40_final_bindings(tmp_path, state, {"policy_revision": "f" * 40}, trials)
    assert len(close["episode_manifests"]) == len(close["policy_server_receipts"]) == 40
    assert close["metrics"]["episodes"] == 40 and close["metrics"]["official_successes"] == 4


def test_top40_first_cohort_zero_success_and_contact_aborts(monkeypatch, tmp_path) -> None:
    args = argparse.Namespace(output_root=tmp_path, early_abort_completed_trials=12)
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._attempted_gate_evidence", lambda *_args: [EpisodeGateEvidence(False, False, None)] * 12)
    receipt = _abort_after_first_completed_cohort(args, trial_ids=tuple(f"trial-{index}" for index in range(12)), invocation_id="a" * 32)
    assert receipt is not None and receipt["reason"]


def test_top40_duplicate_reset_hashes_fail_completed_campaign_diversity() -> None:
    evidence = [EpisodeGateEvidence(True, True, "a" * 64) for _ in range(40)]
    diversity = assess_reset_diversity(evidence, minimum_ratio=1.0)
    assert diversity.passed is False and diversity.unique_hashes == 1


def test_historical_direct_cpu_workers_are_exactly_four() -> None:
    action = next(action for action in build_parser()._actions if "--historical-control-workers" in action.option_strings)

    assert action.type("4") == 4
    with pytest.raises(argparse.ArgumentTypeError, match="exactly 4"):
        action.type("6")


def test_non_dry_direct_cpu_is_rejected_before_preflight_or_output(tmp_path) -> None:
    args = build_parser().parse_args([
        "--matrix", "configs/eval_groot_n17_public_280.json", "--policy-path", "policy", "--policy-revision-file", "revision",
        "--output-root", str(tmp_path), "--policy-repo", "org/policy", "--policy-step", "12000",
        "--code-revision", "a" * 40, "--asset-revision", "b" * 40, "--release-assets-root", str(tmp_path / "Release"),
        "--simulator-version", "isaac-5.1", "--policy-artifact-sha256", "c" * 64, "--image-identity", "sha256:image",
        "--historical-control", "--parity-stage", "direct_cpu", "--execution-mode", "direct", "--device", "cpu",
        "--historical-control-workers", "4",
    ])
    called = []
    with pytest.raises(ValueError, match="direct_cpu is unsupported before campaign launch"):
        run_campaign(args, runtime_preflight=lambda: called.append(True))
    assert called == []
    assert not (tmp_path / "capacity-report.json").exists()
    assert not (tmp_path / "campaign-ledger").exists()


def test_canonical_public_campaign_rejects_missing_scale_stage_before_launch(tmp_path) -> None:
    args = build_parser().parse_args([
        "--matrix", "configs/eval_groot_n17_public_280.json", "--policy-path", "policy", "--policy-revision-file", "revision",
        "--output-root", str(tmp_path), "--policy-repo", "org/policy", "--policy-step", "12000",
        "--code-revision", "a" * 40, "--asset-revision", "b" * 40, "--release-assets-root", str(tmp_path / "assets" / "Release"),
        "--simulator-version", "isaac-5.1", "--policy-artifact-sha256", "c" * 64, "--image-identity", "sha256:image",
        "--groot-root", str(tmp_path / "Isaac-GR00T"), "--groot-revision", "d" * 40,
        "--groot-python", str(tmp_path / "groot-venv" / "bin" / "python"), "--workers", "1",
    ])

    with pytest.raises(ValueError, match="canonical public 280 execution requires --parity-stage scale"):
        run_campaign(args)


def test_scale_refuses_to_relax_the_280_reset_diversity_requirement(tmp_path) -> None:
    args = build_parser().parse_args([
        "--matrix", "configs/eval_groot_n17_public_280.json", "--policy-path", "policy", "--policy-revision-file", "revision",
        "--output-root", str(tmp_path), "--policy-repo", "org/policy", "--policy-step", "12000",
        "--code-revision", "a" * 40, "--asset-revision", "b" * 40, "--release-assets-root", str(tmp_path / "assets" / "Release"),
        "--simulator-version", "isaac-5.1", "--policy-artifact-sha256", "c" * 64, "--image-identity", "sha256:image",
        "--groot-root", str(tmp_path / "Isaac-GR00T"), "--groot-revision", "d" * 40,
        "--groot-python", str(tmp_path / "groot-venv" / "bin" / "python"), "--workers", "1",
        "--parity-stage", "scale", "--minimum-reset-uniqueness-ratio", "0.99",
    ])

    with pytest.raises(ValueError, match="100% distinct canonical reset hashes"):
        run_campaign(args)


def _write_server_receipt_artifacts(root: Path, *, stage: str, successes: int = 10) -> Path:
    policy_receipts = []
    for index, trial_id in enumerate(HISTORICAL_CONTROL_IDS):
        writer = EpisodeArtifactWriter(root, trial_id)
        writer.append_annotation({"step": 0, "action_source": "policy_server"})
        videos = writer.staging / "videos"
        videos.mkdir()
        for filename in CANONICAL_VIDEO_FILENAMES:
            (videos / filename).write_bytes(b"video")
        writer.finalize(
            {
                "terminal_reason": "horizon",
                "outcome": "success" if index < successes else "timeout",
                "accepted_success": index < successes,
                "visible_contact": {"observed": True, "source": "simulator_particle_to_gripper_distance", "minimum_distance_m": 0.01},
                "reset_hash": f"{index + 1:064x}",
                "identity": {
                    "policy_repo": "org/policy", "policy_revision": "a" * 40, "policy_step": 1,
                    "code_revision": "b" * 40, "asset_revision": "e" * 40,
                    "simulator_version": "isaac-5.1", "strategy": "canonical", "seed": 42,
                },
                "provenance": {
                    "policy_artifact_sha256": "c" * 64,
                    "image_identity": "sha256:image",
                    "execution_mode": "policy_server",
                    "execution_backend": "policy_server",
                    "simulator_device": "cpu" if stage == "server_cpu" else "cuda:0",
                    "policy_device": "cuda:0",
                    "parity_stage": stage,
                },
            },
            required_videos=CANONICAL_VIDEO_FILENAMES,
        )
        receipt_args = argparse.Namespace(
            output_root=root, episode_id=trial_id, parity_stage=stage, policy_path=Path("/checkpoint"),
            groot_python=Path("/venv/bin/python"), policy_server_port=5511, policy_server_request_timeout=1.0,
            policy_server_readiness_timeout=1.0, policy_artifact_sha256="c" * 64, code_revision="b" * 40,
            image_identity="sha256:image", device="cpu" if stage == "server_cpu" else "cuda:0", policy_device="cuda:0",
            seed=42,
        )
        policy_receipt = trial_module.write_policy_server_receipt(
            receipt_args, groot_revision="d" * 40, python_version="3.10.18", checkpoint_revision="a" * 40,
            command=trial_module.build_policy_server_command(receipt_args),
        )
        policy_receipts.append({
            "trial_id": trial_id,
            "path": policy_receipt.name,
            "sha256": _sha256_file(policy_receipt),
        })
    receipt = root / "receipt.json"
    receipt.write_text(json.dumps({
        "schema_version": 1,
        "parity_stage": stage,
        "trial_count": 12,
        "trial_ids": list(HISTORICAL_CONTROL_IDS),
        "official_successes": successes,
        "backend": "policy_server_cpu" if stage == "server_cpu" else "policy_server_cuda",
        "artifact_root": str(root),
        "policy_server_receipts": policy_receipts,
    }), encoding="utf-8")
    return receipt


def _write_cuda_abort_fixture(root: Path, *, contacts: int = 6) -> Path:
    _write_server_receipt_artifacts(root, stage="server_cuda", successes=0)
    for index, trial_id in enumerate(HISTORICAL_CONTROL_IDS):
        if index < contacts:
            continue
        episode_path = root / "raw" / trial_id / "episode.json"
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        episode["visible_contact"]["observed"] = False
        episode_path.write_text(json.dumps(episode), encoding="utf-8")
        manifest_path = episode_path.with_name("SHA256SUMS.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["episode.json"]["sha256"] = _sha256_file(episode_path)
        manifest["episode.json"]["size"] = episode_path.stat().st_size
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    invocation_id = "f" * 32
    receipt = {
        "status": "aborted", "reason": "zero_official_successes", "completed_trials": 12,
        "trial_ids": list(HISTORICAL_CONTROL_IDS), "official_successes": 0,
        "visible_robot_garment_contacts": contacts, "invocation_id": invocation_id,
    }
    path = root / "campaign-abort-receipt-test.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    bound_receipt = {**receipt, "receipt_path": str(path)}
    waves = []
    for index in range(0, 12, 4):
        ids = list(HISTORICAL_CONTROL_IDS[index:index + 4])
        wave = {"mode": "production", "status": "terminal", "trial_ids": ids,
                "scheduled_trial_ids": ids, "launched_trial_ids": ids,
                "completed_trials": 4, "failed_trials": 0}
        if index == 8:
            wave["abort_receipt"] = bound_receipt
        waves.append(wave)
    ledger = {"schema_version": 1, "invocation_id": invocation_id, "status": "failed", "mode": "production",
              "pending_before": list(HISTORICAL_CONTROL_IDS), "completed_after": list(HISTORICAL_CONTROL_IDS),
              "abort_receipt": bound_receipt, "waves": waves}
    (root / "campaign-ledger").mkdir()
    (root / "campaign-ledger" / f"{invocation_id}.json").write_text(json.dumps(ledger), encoding="utf-8")
    return path


def test_cuda_abort_evidence_accepts_the_production_receipt_path_binding_schema(tmp_path) -> None:
    cuda_abort = _write_cuda_abort_fixture(tmp_path / "cuda-abort")
    expected = _cpu_scale_live_invocation(
        _cpu_scale_args(tmp_path, legacy=cuda_abort, server_cpu=cuda_abort, cuda_abort=cuda_abort),
        build_public_matrix(),
    )
    receipt = json.loads(cuda_abort.read_text(encoding="utf-8"))
    ledger = json.loads((cuda_abort.parent / "campaign-ledger" / f"{'f' * 32}.json").read_text(encoding="utf-8"))

    assert "receipt_path" not in receipt
    assert ledger["abort_receipt"]["receipt_path"] == str(cuda_abort)
    assert ledger["waves"][-1]["abort_receipt"]["receipt_path"] == str(cuda_abort)
    assert _validate_cuda_abort_evidence(cuda_abort, expected)["official_successes"] == 0


def test_cuda_abort_evidence_rejects_a_receipt_path_in_the_on_disk_receipt(tmp_path) -> None:
    cuda_abort = _write_cuda_abort_fixture(tmp_path / "cuda-abort")
    expected = _cpu_scale_live_invocation(
        _cpu_scale_args(tmp_path, legacy=cuda_abort, server_cpu=cuda_abort, cuda_abort=cuda_abort),
        build_public_matrix(),
    )
    receipt = json.loads(cuda_abort.read_text(encoding="utf-8"))
    receipt["receipt_path"] = str(cuda_abort)
    cuda_abort.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="on-disk receipt must not contain a receipt path binding"):
        _validate_cuda_abort_evidence(cuda_abort, expected)


@pytest.mark.parametrize("location", ("ledger", "terminal_wave"))
@pytest.mark.parametrize("tampering", ("missing", "wrong"))
def test_cuda_abort_evidence_rejects_missing_or_tampered_receipt_path_binding(tmp_path, location, tampering) -> None:
    cuda_abort = _write_cuda_abort_fixture(tmp_path / "cuda-abort")
    expected = _cpu_scale_live_invocation(
        _cpu_scale_args(tmp_path, legacy=cuda_abort, server_cpu=cuda_abort, cuda_abort=cuda_abort),
        build_public_matrix(),
    )
    ledger_path = cuda_abort.parent / "campaign-ledger" / f"{'f' * 32}.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    bound_receipt = ledger["abort_receipt"] if location == "ledger" else ledger["waves"][-1]["abort_receipt"]
    if tampering == "missing":
        del bound_receipt["receipt_path"]
    else:
        bound_receipt["receipt_path"] = str(cuda_abort.with_name("wrong-receipt.json"))
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

    with pytest.raises(ValueError, match="receipt path does not bind the on-disk receipt"):
        _validate_cuda_abort_evidence(cuda_abort, expected)


def test_new_server_receipt_rejects_artifacts_from_a_foreign_stage(tmp_path) -> None:
    receipt = _write_server_receipt_artifacts(tmp_path, stage="server_cuda")
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload.update({"parity_stage": "server_cpu", "backend": "policy_server_cpu"})
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact backend or stage is foreign"):
        _read_parity_receipt(receipt)


def test_new_server_receipt_rejects_a_forged_claimed_success_total(tmp_path) -> None:
    receipt = _write_server_receipt_artifacts(tmp_path, stage="server_cpu")
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["official_successes"] = 12
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="claimed successes do not match terminal artifacts"):
        _read_parity_receipt(receipt)


def test_new_server_receipt_rejects_policy_server_device_binding_mismatch(tmp_path) -> None:
    receipt = _write_server_receipt_artifacts(tmp_path, stage="server_cpu")
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    binding = payload["policy_server_receipts"][0]
    policy_receipt = tmp_path / binding["path"]
    policy_payload = json.loads(policy_receipt.read_text(encoding="utf-8"))
    policy_payload["policy_device"] = "cuda:1"
    policy_receipt.write_text(json.dumps(policy_payload), encoding="utf-8")
    binding["sha256"] = _sha256_file(policy_receipt)
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="device binding mismatches terminal provenance"):
        _read_parity_receipt(receipt)


def test_server_cuda_receipt_rejects_noncanonical_simulator_device_after_rehash(tmp_path) -> None:
    receipt = _write_server_receipt_artifacts(tmp_path, stage="server_cuda")
    trial_id = HISTORICAL_CONTROL_IDS[0]
    episode_path = tmp_path / "raw" / trial_id / "episode.json"
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    episode["provenance"]["simulator_device"] = "cuda:bogus"
    episode_path.write_text(json.dumps(episode), encoding="utf-8")
    manifest_path = tmp_path / "raw" / trial_id / "SHA256SUMS.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["episode.json"]["sha256"] = _sha256_file(episode_path)
    manifest["episode.json"]["size"] = episode_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact device does not match its stage"):
        _read_parity_receipt(receipt)


def test_new_server_receipt_rejects_minimal_policy_server_receipt(tmp_path) -> None:
    receipt = _write_server_receipt_artifacts(tmp_path, stage="server_cpu")
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    binding = payload["policy_server_receipts"][0]
    path = tmp_path / binding["path"]
    path.write_text(json.dumps({"simulator_device": "cpu", "policy_device": "cuda:0"}), encoding="utf-8")
    binding["sha256"] = _sha256_file(path)
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="device binding mismatches terminal provenance"):
        _read_parity_receipt(receipt)


def _write_sha256sums(root: Path) -> Path:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    sums = root / "SHA256SUMS"
    sums.write_text("".join(f"{_sha256_file(path)}  ./{path.relative_to(root)}\n" for path in files), encoding="utf-8")
    return sums


def _write_legacy_receipt_bundle(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "archived-output.txt").write_text("archived rollout", encoding="utf-8")
    report = archive / "rollout-report.json"
    report.write_text(json.dumps({"trials": [{"trial": {"trial_id": trial_id}} for trial_id in HISTORICAL_CONTROL_IDS]}), encoding="utf-8")
    archive_sums = _write_sha256sums(archive)

    controls = tmp_path / "controls"
    for trial in historical_control_trials():
        release = controls / trial.trial_id / "garment-config" / "Release"
        release.mkdir(parents=True)
        (release / "Release_test_list.txt").write_text(trial.garment_name, encoding="utf-8")
    control_sums = _write_sha256sums(controls)

    reproduction = tmp_path / "reproduction"
    reproduction.mkdir()
    source_sha256 = {
        "run-lehome-24-shared.sh": "bbf8fee87d7efc4e09b08874e3265175fd7a4c9ea9494be8ac7e8301fd4d7f92",
        "eval_groot_n17_matrix_parallel.py": "e26d63536a6eef53fe6d0de8a22ee683616aa1b5ba4aa4dc968d4eb13a37f89a",
        "groot_policy.py": "cee2d9f78711e867ef4e4867ee615abdbfe5584e3385c3b601adfe90f25d78bf",
        "serve_groot_policy.py": "b8aa5f81e651e1db18f4189e55121e0eca67ca7613a58db361ae88753a8cb3e4",
    }
    binding = {
        "source_sha256": source_sha256,
        "archive_sha256sums_sha256": _sha256_file(archive_sums),
        "archive_rollout_report_sha256": _sha256_file(report),
        "historical_control_sha256sums_sha256": _sha256_file(control_sums),
    }
    records = []
    for index, trial_id in enumerate(HISTORICAL_CONTROL_IDS):
        trial_dir = reproduction / trial_id
        trial_dir.mkdir()
        terminal = trial_dir / "terminal.json"
        terminal.write_text(json.dumps({
            "trial_id": trial_id, "terminal": True, "backend": "legacy_shared_policy_server",
            "environment_device": "cpu", "outcome": "success" if index < 9 else "timeout",
            "accepted_success": index < 9, "integrity_binding": binding,
        }), encoding="utf-8")
        manifest = trial_dir / "manifest.json"
        manifest.write_text(json.dumps({"episode_id": trial_id}), encoding="utf-8")
        log = trial_dir / "terminal.log"
        log.write_text("terminal evidence\n", encoding="utf-8")
        records.append({
            "trial_id": trial_id,
            "terminal_record_path": str(terminal.relative_to(reproduction)), "terminal_record_sha256": _sha256_file(terminal),
            "manifest_path": str(manifest.relative_to(reproduction)), "manifest_sha256": _sha256_file(manifest),
            "log_path": str(log.relative_to(reproduction)), "log_sha256": _sha256_file(log),
        })
    receipt_payload: dict[str, object] = {
        "schema_version": 1, "parity_stage": "legacy_server_cpu", "trial_count": 12, "official_successes": 9,
        "backend": "legacy_shared_policy_server", "source_sha256": source_sha256,
        "archive_root": str(archive), "archive_sha256sums_sha256": _sha256_file(archive_sums),
        "archive_rollout_report_sha256": _sha256_file(report), "historical_control_root": str(controls),
        "historical_control_sha256sums_sha256": _sha256_file(control_sums), "reproduction_root": str(reproduction),
        "terminal_records": records,
        "identity": {
            "policy_repo": "org/policy", "policy_step": 1, "policy_revision": "a" * 40,
            "code_revision": "b" * 40, "asset_revision": "e" * 40,
            "simulator_version": "isaac-5.1", "policy_artifact_sha256": "c" * 64,
            "image_identity": "sha256:image", "strategy": "canonical",
        },
        "reproduction": {"backend": "legacy_shared_policy_server", "environment_device": "cpu", "reference_isaac_workers": 6,
                           "reference_policy_servers": 2, "actual_isaac_workers": 2, "actual_policy_servers": 2,
                           "gpu_count": 4, "concurrency_non_parity": True},
    }
    receipt = tmp_path / "legacy-receipt.json"
    receipt.write_text(json.dumps(receipt_payload), encoding="utf-8")
    return receipt, receipt_payload


def test_legacy_receipt_rejects_terminal_forgery_after_rehashing_the_terminal_record(tmp_path) -> None:
    receipt, payload = _write_legacy_receipt_bundle(tmp_path)
    assert _validate_legacy_shared_policy_receipt(receipt)["official_successes"] == 9

    reproduction = Path(str(payload["reproduction_root"]))
    record = payload["terminal_records"][0]
    terminal = reproduction / str(record["terminal_record_path"])
    terminal_payload = json.loads(terminal.read_text(encoding="utf-8"))
    terminal_payload["integrity_binding"]["archive_rollout_report_sha256"] = "0" * 64
    terminal.write_text(json.dumps(terminal_payload), encoding="utf-8")
    record["terminal_record_sha256"] = _sha256_file(terminal)
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="not bound to source/archive/frozen-config identities"):
        _validate_legacy_shared_policy_receipt(receipt)


def test_legacy_receipt_rejects_a_forged_claimed_success_total(tmp_path) -> None:
    receipt, payload = _write_legacy_receipt_bundle(tmp_path)
    payload["official_successes"] = 12
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="claimed successes do not match reproduction terminals"):
        _validate_legacy_shared_policy_receipt(receipt)


def test_legacy_receipt_rejects_duplicate_terminal_records_that_inflate_successes(tmp_path) -> None:
    receipt, payload = _write_legacy_receipt_bundle(tmp_path)
    records = payload["terminal_records"]
    payload["terminal_records"] = [*records, records[0]]
    payload["official_successes"] = 11
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly twelve terminal artifact records"):
        _validate_legacy_shared_policy_receipt(receipt)


def test_scale_rejects_a_bare_legacy_claim_before_it_can_enter_the_ladder(tmp_path) -> None:
    receipt = tmp_path / "bare-legacy.json"
    receipt.write_text(json.dumps({
        "schema_version": 1,
        "parity_stage": "legacy_server_cpu",
        "trial_count": 12,
        "official_successes": 12,
        "backend": "legacy_shared_policy_server",
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="shared-policy launcher and adapter hashes"):
        _require_scale_parity([receipt])


def _cpu_scale_args(tmp_path: Path, *, legacy: Path, server_cpu: Path, cuda_abort: Path) -> argparse.Namespace:
    revision = tmp_path / "policy-revision.txt"
    revision.write_text("a" * 40, encoding="utf-8")
    return argparse.Namespace(
        output_root=tmp_path / "production", policy_revision_file=revision,
        policy_repo="org/policy", policy_step=1, code_revision="b" * 40,
        asset_revision="e" * 40, simulator_version="isaac-5.1",
        policy_artifact_sha256="c" * 64, image_identity="sha256:image",
        strategy="canonical", groot_revision="d" * 40,
        groot_python=Path("/venv/bin/python"), device="cpu", execution_mode="policy_server",
        policy_device="cuda:0", workers=4, minimum_reset_uniqueness_ratio=1.0,
        cpu_scale_decision="authorize_cpu_simulator_policy_server_scale_v1",
        legacy_server_cpu_receipt=legacy, server_cpu_receipt=server_cpu,
        cuda_abort_receipt=cuda_abort,
    )


def _scale_cpu_scheduler_args(tmp_path: Path) -> argparse.Namespace:
    args = _worker_args(tmp_path)
    args.parity_stage = "scale_cpu"
    args.workers = 4
    args.dry_run = False
    args.capacity_sweep = None
    args.historical_control = False
    args.trials_per_worker = 1
    args.early_abort_completed_trials = 12
    args.minimum_reset_uniqueness_ratio = 1.0
    return args


def _write_scale_cpu_terminal(
    root: Path, trial: Trial, *, success: bool = True, contact: bool = True,
) -> None:
    writer = EpisodeArtifactWriter(root, trial.trial_id)
    writer.append_annotation({"step": 0, "action_source": "policy"})
    videos = writer.staging / "videos"
    videos.mkdir()
    for filename in CANONICAL_VIDEO_FILENAMES:
        (videos / filename).write_bytes(b"video")
    writer.finalize({
        "terminal_reason": "horizon",
        "outcome": "success" if success else "timeout",
        "accepted_success": success,
        "visible_contact": {
            "observed": contact,
            "source": "simulator_particle_to_gripper_distance",
            "minimum_distance_m": 0.01,
        },
        "reset_hash": f"{abs(hash(trial.trial_id)) % (1 << 256):064x}",
    }, required_videos=CANONICAL_VIDEO_FILENAMES)
    (root / f"policy-server-receipt-{trial.trial_id}.json").write_text(
        json.dumps({"schema_version": 1, "episode_id": trial.trial_id, "backend": "policy_server"}), encoding="utf-8",
    )


def _scale_cpu_state(tmp_path: Path) -> tuple[CampaignState, dict[str, Trial]]:
    trials = build_public_matrix().trials
    state = CampaignState(tmp_path, tuple(trial.trial_id for trial in trials))
    return state, {trial.trial_id: trial for trial in trials}


def test_scale_cpu_resume_validates_materialized_output_with_the_loaded_matrix(monkeypatch, tmp_path) -> None:
    args = _scale_cpu_scheduler_args(tmp_path)
    args.policy_device = "cuda:0"
    args.policy_revision_file.write_text("e" * 40, encoding="utf-8")
    matrix = build_public_matrix()
    invocation = {
        "policy_repo": args.policy_repo,
        "policy_revision": "e" * 40,
        "policy_step": args.policy_step,
        "code_revision": args.code_revision,
        "asset_revision": args.asset_revision,
        "simulator_version": args.simulator_version,
        "strategy": args.strategy,
        "policy_device": args.policy_device,
        "policy_artifact_sha256": args.policy_artifact_sha256,
        "image_identity": args.image_identity,
        "groot_revision": args.groot_revision,
        "groot_python": str(args.groot_python),
        "groot_python_version": "3.10.18",
    }

    def write_terminal(trial: Trial, reset_number: int, policy_device: str) -> None:
        writer = EpisodeArtifactWriter(tmp_path, trial.trial_id)
        writer.append_annotation({"step": 0, "action_source": "policy_server"})
        videos = writer.staging / "videos"
        videos.mkdir()
        for filename in CANONICAL_VIDEO_FILENAMES:
            (videos / filename).write_bytes(b"video")
        writer.finalize({
            "terminal_reason": "horizon", "outcome": "success", "accepted_success": True,
            "visible_contact": {"observed": True, "source": "simulator_particle_to_gripper_distance", "minimum_distance_m": 0.01},
            "reset_hash": f"{reset_number:064x}",
            "identity": {
                "episode_id": trial.trial_id, "policy_repo": invocation["policy_repo"],
                "policy_revision": invocation["policy_revision"], "policy_step": invocation["policy_step"],
                "code_revision": invocation["code_revision"], "asset_revision": invocation["asset_revision"],
                "simulator_version": invocation["simulator_version"], "garment_name": trial.garment_name,
                "category": trial.category, "release_stage": trial.release_stage, "seed": trial.seed,
                "instruction": "fold the garment on the table", "strategy": invocation["strategy"],
            },
            "provenance": {
                "execution_mode": "policy_server", "execution_backend": "policy_server", "parity_stage": "server_cpu",
                "simulator_device": "cpu", "policy_device": policy_device,
                "policy_artifact_sha256": invocation["policy_artifact_sha256"], "image_identity": invocation["image_identity"],
            },
        }, required_videos=CANONICAL_VIDEO_FILENAMES)
        (tmp_path / f"policy-server-receipt-{trial.trial_id}.json").write_text(json.dumps({
            "schema_version": 1, "episode_id": trial.trial_id, "parity_stage": "server_cpu",
            "backend": "policy_server", "checkpoint_revision": invocation["policy_revision"],
            "checkpoint_digest": invocation["policy_artifact_sha256"], "code_revision": invocation["code_revision"],
            "image_identity": invocation["image_identity"], "groot_revision": invocation["groot_revision"],
            "python_path": invocation["groot_python"], "python_version": invocation["groot_python_version"],
            "policy_seed": trial.seed, "simulator_device": "cpu", "policy_device": policy_device,
        }), encoding="utf-8")

    for reset_number, trial in enumerate(matrix.trials[:-1], start=1):
        write_terminal(trial, reset_number, f"cuda:{(reset_number - 1) % 4}")
    resumed: list[str] = []

    def run_group(_args, assignments, **_kwargs):
        resumed.extend(trial.trial_id for _, trial in assignments)
        write_terminal(assignments[0][1], len(matrix.trials), "cuda:3")
        return 1.0, 1, 0

    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._worker_gpu_indices", lambda _args, count: tuple(range(count)))
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._run_worker_group", run_group)

    report = _run_campaign_under_supervisor(args, matrix, cpu_scale_authorization={"invocation": invocation})

    assert resumed == [matrix.trials[-1].trial_id]
    assert report["completed_after"] == [trial.trial_id for trial in matrix.trials]

    trial = matrix.trials[-1]
    receipt_path = tmp_path / f"policy-server-receipt-{trial.trial_id}.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["policy_device"] = "cuda:0"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    state = CampaignState(tmp_path, tuple(item.trial_id for item in matrix.trials))
    with pytest.raises(ValueError, match="policy-server receipt policy device does not match its episode"):
        _validate_scale_cpu_production_output(args, state, matrix, {"invocation": invocation})

    receipt["policy_device"] = "cuda:4"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    episode_path = tmp_path / "raw" / trial.trial_id / "episode.json"
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    episode["provenance"]["policy_device"] = "cuda:4"
    episode_path.write_text(json.dumps(episode), encoding="utf-8")
    manifest_path = episode_path.with_name("SHA256SUMS.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["episode.json"]["sha256"] = _sha256_file(episode_path)
    manifest["episode.json"]["size"] = episode_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="policy device is not in the authorized scale_cpu worker set"):
        _validate_scale_cpu_production_output(args, state, matrix, {"invocation": invocation})


def test_scale_cpu_fresh_scheduler_finishes_first_twelve_before_trial_thirteen(monkeypatch, tmp_path) -> None:
    state, by_id = _scale_cpu_state(tmp_path)
    canary = state.trial_ids[:12]
    thirteenth = state.trial_ids[12]
    launches: list[tuple[tuple[str, ...], tuple[int | None, ...]]] = []

    def run_group(_args, assignments, *, gpu_indices=None, **_kwargs):
        ids = tuple(trial.trial_id for _, trial in assignments)
        launches.append((ids, tuple(gpu_indices or ())))
        for _, trial in assignments:
            _write_scale_cpu_terminal(tmp_path, trial)
        return 1.0, len(assignments), 0

    pending_scans = iter((state.trial_ids, (thirteenth,), ()))
    def pending_only_thirteen(_state):
        return next(pending_scans, ())

    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.pending_trial_ids", pending_only_thirteen)
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._run_worker_group", run_group)
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._worker_gpu_indices", lambda _args, count: tuple(range(count)))
    with pytest.raises(RuntimeError, match="reset diversity gate failed"):
        _run_campaign_under_supervisor(_scale_cpu_scheduler_args(tmp_path), build_public_matrix())

    assert [ids for ids, _ in launches[:3]] == [canary[:4], canary[4:8], canary[8:12]]
    assert launches[3][0] == (thirteenth,)
    assert all(thirteenth not in ids for ids, _ in launches[:3])
    assert [gpus for _, gpus in launches[:3]] == [(0, 1, 2, 3)] * 3
    receipt = json.loads((tmp_path / "cpu-scale-canary-receipt.json").read_text(encoding="utf-8"))
    assert receipt["decision"] == "pass"
    assert receipt["canary_trial_ids"] == list(canary)


def test_scale_cpu_resume_launches_only_two_missing_canaries_before_trial_thirteen(monkeypatch, tmp_path) -> None:
    state, by_id = _scale_cpu_state(tmp_path)
    canary = state.trial_ids[:12]
    thirteenth = state.trial_ids[12]
    for trial_id in canary[:10]:
        _write_scale_cpu_terminal(tmp_path, by_id[trial_id])
    launches: list[tuple[tuple[str, ...], tuple[int | None, ...]]] = []

    def run_group(_args, assignments, *, gpu_indices=None, **_kwargs):
        ids = tuple(trial.trial_id for _, trial in assignments)
        launches.append((ids, tuple(gpu_indices or ())))
        for _, trial in assignments:
            _write_scale_cpu_terminal(tmp_path, trial)
        return 1.0, len(assignments), 0

    pending_scans = iter((state.trial_ids, (thirteenth,), ()))
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.pending_trial_ids", lambda _state: next(pending_scans, ()))
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._run_worker_group", run_group)
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._worker_gpu_indices", lambda _args, count: tuple(range(count)))
    with pytest.raises(RuntimeError, match="reset diversity gate failed"):
        _run_campaign_under_supervisor(_scale_cpu_scheduler_args(tmp_path), build_public_matrix())

    assert launches[0] == (canary[10:12], (0, 1))
    assert launches[1][0] == (thirteenth,)
    assert thirteenth not in launches[0][0]


def test_scale_cpu_preexisting_zero_success_canary_aborts_before_popen(monkeypatch, tmp_path) -> None:
    state, by_id = _scale_cpu_state(tmp_path)
    for trial_id in state.trial_ids[:12]:
        _write_scale_cpu_terminal(tmp_path, by_id[trial_id], success=False, contact=True)
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign._run_worker_group",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("canary must not relaunch")),
    )

    with pytest.raises(RuntimeError, match="zero success"):
        _run_scale_cpu_canary(_scale_cpu_scheduler_args(tmp_path), state=state, by_id=by_id, authorization=None)

    assert json.loads((tmp_path / "cpu-scale-canary-receipt.json").read_text(encoding="utf-8"))["decision"] == "abort"


def test_scale_cpu_preexisting_valid_canary_permits_trial_thirteen(monkeypatch, tmp_path) -> None:
    state, by_id = _scale_cpu_state(tmp_path)
    canary = state.trial_ids[:12]
    thirteenth = state.trial_ids[12]
    for trial_id in canary:
        _write_scale_cpu_terminal(tmp_path, by_id[trial_id])
    launches: list[tuple[str, ...]] = []

    def run_group(_args, assignments, **_kwargs):
        launches.append(tuple(trial.trial_id for _, trial in assignments))
        for _, trial in assignments:
            _write_scale_cpu_terminal(tmp_path, trial)
        return 1.0, len(assignments), 0

    pending_scans = iter((state.trial_ids, (thirteenth,), ()))
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.pending_trial_ids", lambda _state: next(pending_scans, ()))
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._run_worker_group", run_group)
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._worker_gpu_indices", lambda _args, count: tuple(range(count)))
    with pytest.raises(RuntimeError, match="reset diversity gate failed"):
        _run_campaign_under_supervisor(_scale_cpu_scheduler_args(tmp_path), build_public_matrix())

    assert launches == [(thirteenth,)]


def test_scale_cpu_interrupted_canary_ledger_fails_closed(monkeypatch, tmp_path) -> None:
    state, by_id = _scale_cpu_state(tmp_path)
    trial_id = state.trial_ids[0]
    ledger_root = tmp_path / "campaign-ledger"
    ledger_root.mkdir()
    (ledger_root / f"{'f' * 32}.json").write_text(json.dumps({
        "schema_version": 1, "mode": "production", "waves": [{
            "trial_ids": [trial_id], "scheduled_trial_ids": [trial_id], "launched_trial_ids": [], "status": "started",
        }],
    }), encoding="utf-8")
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign._run_worker_group",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("ambiguous attempt must not launch")),
    )

    with pytest.raises(ValueError, match="interrupted or ambiguous"):
        _run_scale_cpu_canary(_scale_cpu_scheduler_args(tmp_path), state=state, by_id=by_id, authorization=None)


def test_scale_cpu_terminal_ledger_failure_counts_as_a_nonretryable_canary_attempt(tmp_path) -> None:
    state, by_id = _scale_cpu_state(tmp_path)
    canary = state.trial_ids[:12]
    for trial_id in canary[1:]:
        _write_scale_cpu_terminal(tmp_path, by_id[trial_id])
    ledger_root = tmp_path / "campaign-ledger"
    ledger_root.mkdir()
    (ledger_root / f"{'a' * 32}.json").write_text(json.dumps({
        "schema_version": 1, "mode": "production", "waves": [{
            "wave": 1, "trial_ids": list(canary), "scheduled_trial_ids": list(canary), "launched_trial_ids": list(canary),
            "completed_trials": 11, "failed_trials": 1, "status": "terminal",
        }],
    }), encoding="utf-8")

    _run_scale_cpu_canary(_scale_cpu_scheduler_args(tmp_path), state=state, by_id=by_id, authorization=None)

    receipt = json.loads((tmp_path / "cpu-scale-canary-receipt.json").read_text(encoding="utf-8"))
    assert receipt["decision"] == "pass"
    assert receipt["attempt_evidence"][0]["source"] == "terminal_ledger_without_valid_artifact"
    assert receipt["attempt_evidence"][0]["official_success"] is False


def test_scale_cpu_passing_canary_retries_a_frozen_terminal_failure_during_final_close(monkeypatch, tmp_path) -> None:
    state, by_id = _scale_cpu_state(tmp_path)
    canary = state.trial_ids[:12]
    for trial_id in canary[1:]:
        _write_scale_cpu_terminal(tmp_path, by_id[trial_id])
    ledger_root = tmp_path / "campaign-ledger"
    ledger_root.mkdir()
    (ledger_root / f"{'b' * 32}.json").write_text(json.dumps({
        "schema_version": 1, "mode": "production", "waves": [{
            "wave": 1, "trial_ids": list(canary), "scheduled_trial_ids": list(canary), "launched_trial_ids": list(canary),
            "completed_trials": 11, "failed_trials": 1, "status": "terminal",
        }],
    }), encoding="utf-8")
    args = _scale_cpu_scheduler_args(tmp_path)
    _run_scale_cpu_canary(args, state=state, by_id=by_id, authorization=None)
    launches: list[tuple[str, ...]] = []

    def run_group(_args, assignments, **_kwargs):
        launches.append(tuple(trial.trial_id for _, trial in assignments))
        for _, trial in assignments:
            _write_scale_cpu_terminal(tmp_path, trial)
        return 1.0, len(assignments), 0

    pending_scans = iter(((canary[0],), (canary[0],), ()))
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.pending_trial_ids", lambda _state: next(pending_scans, ()))
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._run_worker_group", run_group)
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._worker_gpu_indices", lambda _args, count: tuple(range(count)))
    with pytest.raises(RuntimeError, match="reset diversity gate failed"):
        _run_campaign_under_supervisor(args, build_public_matrix())

    assert launches == [(canary[0],)]
    receipt = json.loads((tmp_path / "cpu-scale-canary-receipt.json").read_text(encoding="utf-8"))
    assert receipt["attempt_evidence"][0]["official_success"] is False


def test_scale_cpu_stale_canary_receipt_rejects_before_later_work(tmp_path) -> None:
    state, by_id = _scale_cpu_state(tmp_path)
    for trial_id in state.trial_ids[:12]:
        _write_scale_cpu_terminal(tmp_path, by_id[trial_id])
    args = _scale_cpu_scheduler_args(tmp_path)
    _run_scale_cpu_canary(args, state=state, by_id=by_id, authorization=None)
    receipt_path = tmp_path / "cpu-scale-canary-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["authorization_sha256"] = "stale"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="canary receipt is stale"):
        _run_scale_cpu_canary(args, state=state, by_id=by_id, authorization=None)


def test_scale_cpu_canary_resume_requires_exact_all_twelve_ledger_bindings(monkeypatch, tmp_path) -> None:
    state, by_id = _scale_cpu_state(tmp_path)
    canary = state.trial_ids[:12]
    for trial_id in canary:
        _write_scale_cpu_terminal(tmp_path, by_id[trial_id])
    ledger_root = tmp_path / "campaign-ledger"
    ledger_root.mkdir()
    ledger_path = ledger_root / f"{'c' * 32}.json"
    ledger_path.write_text(json.dumps({
        "schema_version": 1, "mode": "production", "waves": [{
            "wave": 1, "trial_ids": list(canary), "scheduled_trial_ids": list(canary),
            "launched_trial_ids": list(canary), "completed_trials": 12, "failed_trials": 0, "status": "terminal",
        }],
    }), encoding="utf-8")
    args = _scale_cpu_scheduler_args(tmp_path)

    _run_scale_cpu_canary(args, state=state, by_id=by_id, authorization=None)

    receipt_path = tmp_path / "cpu-scale-canary-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert set(receipt["canary_ledger_bindings"]) == set(canary)
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign._run_worker_group",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("validated canary must not relaunch")),
    )
    _run_scale_cpu_canary(args, state=state, by_id=by_id, authorization=None)

    receipt["canary_ledger_bindings"].pop(canary[0])
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="extra or missing ledger bindings"):
        _run_scale_cpu_canary(args, state=state, by_id=by_id, authorization=None)

    receipt["canary_ledger_bindings"][canary[0]] = {
        "path": str(ledger_path), "sha256": _sha256_file(ledger_path),
    }
    receipt["canary_ledger_bindings"]["extra-trial"] = {
        "path": str(ledger_path), "sha256": _sha256_file(ledger_path),
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="extra or missing ledger bindings"):
        _run_scale_cpu_canary(args, state=state, by_id=by_id, authorization=None)


def test_cpu_scale_authorization_binds_verified_evidence_and_rejects_stale_runtime(tmp_path) -> None:
    legacy, _ = _write_legacy_receipt_bundle(tmp_path / "legacy")
    server_cpu = _write_server_receipt_artifacts(tmp_path / "server-cpu", stage="server_cpu")
    cuda_abort = _write_cuda_abort_fixture(tmp_path / "cuda-abort")
    args = _cpu_scale_args(tmp_path, legacy=legacy, server_cpu=server_cpu, cuda_abort=cuda_abort)

    authorization = _require_cpu_scale_authorization(args, build_public_matrix())

    assert authorization["invocation"]["device"] == "cpu"
    assert (args.output_root / "cpu-scale-authorization.json").is_file()
    args.image_identity = "sha256:stale"
    with pytest.raises(ValueError, match="server_cpu receipt identity does not match"):
        _require_cpu_scale_authorization(args, build_public_matrix())


def test_cpu_scale_authorization_binds_named_isaac_and_groot_import_roots(tmp_path, monkeypatch) -> None:
    legacy, _ = _write_legacy_receipt_bundle(tmp_path / "legacy")
    server_cpu = _write_server_receipt_artifacts(tmp_path / "server-cpu", stage="server_cpu")
    cuda_abort = _write_cuda_abort_fixture(tmp_path / "cuda-abort")
    args = _cpu_scale_args(tmp_path, legacy=legacy, server_cpu=server_cpu, cuda_abort=cuda_abort)
    runtime_root = tmp_path / "trial-runtime"
    for relative_path in (
        "source/lehome",
        "third_party/IsaacLab/source/isaaclab",
        "third_party/IsaacLab/source/isaaclab_assets",
        "third_party/IsaacLab/source/isaaclab_tasks",
        "third_party/IsaacLab/source/isaaclab_mimic",
        "third_party/IsaacLab/source/isaaclab_rl",
    ):
        (runtime_root / relative_path).mkdir(parents=True)
    isaacsim_root = tmp_path / "isaac-sim"
    kit_python = isaacsim_root / "kit" / "python"
    kit_python.mkdir(parents=True)
    groot_root = tmp_path / "isaac-groot"
    groot_root.mkdir()
    args.trial_runtime_root = runtime_root
    args.groot_root = groot_root
    args.policy_path = tmp_path / "policy"
    args.release_assets_root = tmp_path / "assets"
    args.matrix = tmp_path / "matrix.json"

    def clean_checkout(path, *, label):
        revisions = {"trial runtime root": "b" * 40, "release assets root": "e" * 40}
        return Path(path).resolve(), revisions[label]

    monkeypatch.setenv("ISAACSIM_PATH", str(isaacsim_root))
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._clean_git_revision", clean_checkout)
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign._live_groot_identity",
        lambda _args: {"groot_root": str(groot_root.resolve()), "groot_revision": "d" * 40,
                       "groot_python": "/venv/bin/python", "groot_python_sha256": "1" * 64,
                       "groot_python_version": "3.10.18"},
    )
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign._controller_identity",
        lambda: {"controller_root": "/controller", "controller_revision": "1" * 40,
                 "controller_campaign_sha256": "2" * 64, "controller_parity_sha256": "3" * 64},
    )
    monkeypatch.setattr(trial_module, "policy_artifact_sha256", lambda _path: "c" * 64)
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._scale_cpu_runtime_paths", _scale_cpu_runtime_paths)

    authorization = _require_cpu_scale_authorization(args, build_public_matrix())
    invocation = authorization["invocation"]

    assert invocation["isaacsim_python_path"] == str(kit_python.resolve())
    assert invocation["groot_root"] == str(groot_root.resolve())
    assert invocation["trusted_groot_root"] == str(groot_root.resolve())
    assert invocation["trusted_pythonpath"][-2:] == [str(kit_python.resolve()), str(groot_root.resolve())]


def test_cpu_scale_authorization_is_production_only(tmp_path) -> None:
    legacy, _ = _write_legacy_receipt_bundle(tmp_path / "legacy")
    server_cpu = _write_server_receipt_artifacts(tmp_path / "server-cpu", stage="server_cpu")
    cuda_abort = _write_cuda_abort_fixture(tmp_path / "cuda-abort")
    args = _cpu_scale_args(tmp_path, legacy=legacy, server_cpu=server_cpu, cuda_abort=cuda_abort)
    args.dry_run = True

    with pytest.raises(ValueError, match="production-only"):
        _require_cpu_scale_authorization(args, build_public_matrix())


@pytest.mark.parametrize(("attribute", "value", "message"), (
    ("asset_revision", "f" * 40, "server_cpu receipt identity does not match"),
    ("code_revision", "f" * 40, "server_cpu receipt identity does not match"),
))
def test_cpu_scale_authorization_rejects_stale_code_or_asset_identity(tmp_path, attribute, value, message) -> None:
    legacy, _ = _write_legacy_receipt_bundle(tmp_path / "legacy")
    server_cpu = _write_server_receipt_artifacts(tmp_path / "server-cpu", stage="server_cpu")
    cuda_abort = _write_cuda_abort_fixture(tmp_path / "cuda-abort")
    args = _cpu_scale_args(tmp_path, legacy=legacy, server_cpu=server_cpu, cuda_abort=cuda_abort)
    setattr(args, attribute, value)

    with pytest.raises(ValueError, match=message):
        _require_cpu_scale_authorization(args, build_public_matrix())


def test_cpu_scale_authorization_rejects_stale_pinned_python_receipt_identity(tmp_path) -> None:
    legacy, _ = _write_legacy_receipt_bundle(tmp_path / "legacy")
    server_cpu = _write_server_receipt_artifacts(tmp_path / "server-cpu", stage="server_cpu")
    cuda_abort = _write_cuda_abort_fixture(tmp_path / "cuda-abort")
    args = _cpu_scale_args(tmp_path, legacy=legacy, server_cpu=server_cpu, cuda_abort=cuda_abort)
    args.trial_runtime_root = tmp_path / "runtime"
    args.policy_path = tmp_path / "policy"
    args.release_assets_root = tmp_path / "assets"
    args.matrix = tmp_path / "matrix.json"

    def clean_checkout(path, *, label):
        revisions = {"trial runtime root": "b" * 40, "release assets root": "e" * 40}
        return Path(path).resolve(), revisions[label]

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr("scripts.run_groot_flywheel_campaign._clean_git_revision", clean_checkout)
        monkeypatch.setattr(
            "scripts.run_groot_flywheel_campaign._live_groot_identity",
            lambda _args: {"groot_root": "/opt/isaac-groot", "groot_revision": "d" * 40,
                           "groot_python": "/different/python", "groot_python_sha256": "1" * 64,
                           "groot_python_version": "3.10.18"},
        )
        monkeypatch.setattr(
            "scripts.run_groot_flywheel_campaign._controller_identity",
            lambda: {"controller_root": "/controller", "controller_revision": "1" * 40,
                     "controller_campaign_sha256": "2" * 64, "controller_parity_sha256": "3" * 64},
        )
        monkeypatch.setattr(trial_module, "policy_artifact_sha256", lambda _path: "c" * 64)
        with pytest.raises(ValueError, match="server_cpu receipt Python path does not match"):
            _require_cpu_scale_authorization(args, build_public_matrix())
    finally:
        monkeypatch.undo()


@pytest.mark.parametrize("stale_field", ("policy_path", "worker_timeout_seconds", "controller_campaign_sha256"))
def test_cpu_scale_authorization_resume_rejects_stale_bound_invocation_or_controller(tmp_path, monkeypatch, stale_field) -> None:
    legacy, _ = _write_legacy_receipt_bundle(tmp_path / "legacy")
    server_cpu = _write_server_receipt_artifacts(tmp_path / "server-cpu", stage="server_cpu")
    cuda_abort = _write_cuda_abort_fixture(tmp_path / "cuda-abort")
    args = _cpu_scale_args(tmp_path, legacy=legacy, server_cpu=server_cpu, cuda_abort=cuda_abort)
    args.trial_runtime_root = tmp_path / "runtime"
    args.policy_path = tmp_path / "policy"
    args.release_assets_root = tmp_path / "assets"
    args.matrix = tmp_path / "matrix.json"
    args.worker_timeout_seconds = 90.0
    controller = {"controller_root": "/controller", "controller_revision": "1" * 40,
                  "controller_campaign_sha256": "2" * 64, "controller_parity_sha256": "3" * 64}

    def clean_checkout(path, *, label):
        revisions = {"trial runtime root": "b" * 40, "release assets root": "e" * 40}
        return Path(path).resolve(), revisions[label]

    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._clean_git_revision", clean_checkout)
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign._live_groot_identity",
        lambda _args: {"groot_root": "/opt/isaac-groot", "groot_revision": "d" * 40,
                       "groot_python": "/venv/bin/python", "groot_python_sha256": "1" * 64,
                       "groot_python_version": "3.10.18"},
    )
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._controller_identity", lambda: dict(controller))
    monkeypatch.setattr(trial_module, "policy_artifact_sha256", lambda _path: "c" * 64)

    authorization = _require_cpu_scale_authorization(args, build_public_matrix())
    assert authorization == _require_cpu_scale_authorization(args, build_public_matrix())
    if stale_field == "policy_path":
        args.policy_path = tmp_path / "policy-stale"
    elif stale_field == "worker_timeout_seconds":
        args.worker_timeout_seconds = 91.0
    else:
        controller["controller_campaign_sha256"] = "4" * 64
    with pytest.raises(ValueError, match="does not exactly match"):
        _require_cpu_scale_authorization(args, build_public_matrix())


def test_cpu_scale_authorization_rejects_policy_digest_changed_since_authorization(tmp_path, monkeypatch) -> None:
    legacy, _ = _write_legacy_receipt_bundle(tmp_path / "legacy")
    server_cpu = _write_server_receipt_artifacts(tmp_path / "server-cpu", stage="server_cpu")
    cuda_abort = _write_cuda_abort_fixture(tmp_path / "cuda-abort")
    args = _cpu_scale_args(tmp_path, legacy=legacy, server_cpu=server_cpu, cuda_abort=cuda_abort)
    args.trial_runtime_root = tmp_path / "runtime"
    args.policy_path = tmp_path / "policy"
    args.release_assets_root = tmp_path / "assets"
    args.matrix = tmp_path / "matrix.json"

    def clean_checkout(path, *, label):
        revisions = {"trial runtime root": "b" * 40, "release assets root": "e" * 40}
        return Path(path).resolve(), revisions[label]

    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._clean_git_revision", clean_checkout)
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign._live_groot_identity",
        lambda _args: {"groot_root": "/opt/isaac-groot", "groot_revision": "d" * 40,
                       "groot_python": "/venv/bin/python", "groot_python_sha256": "1" * 64,
                       "groot_python_version": "3.10.18"},
    )
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign._controller_identity",
        lambda: {"controller_root": "/controller", "controller_revision": "1" * 40,
                 "controller_campaign_sha256": "2" * 64, "controller_parity_sha256": "3" * 64},
    )
    monkeypatch.setattr(trial_module, "policy_artifact_sha256", lambda _path: "0" * 64)

    with pytest.raises(ValueError, match="policy checkpoint digest"):
        _require_cpu_scale_authorization(args, build_public_matrix())


def test_cpu_scale_authorization_rejects_forged_cuda_abort_contact_total(tmp_path) -> None:
    legacy, _ = _write_legacy_receipt_bundle(tmp_path / "legacy")
    server_cpu = _write_server_receipt_artifacts(tmp_path / "server-cpu", stage="server_cpu")
    cuda_abort = _write_cuda_abort_fixture(tmp_path / "cuda-abort", contacts=6)
    payload = json.loads(cuda_abort.read_text(encoding="utf-8")); payload["visible_robot_garment_contacts"] = 11
    cuda_abort.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="ledger payload does not match"):
        _require_cpu_scale_authorization(
            _cpu_scale_args(tmp_path, legacy=legacy, server_cpu=server_cpu, cuda_abort=cuda_abort),
            build_public_matrix(),
        )


@pytest.mark.parametrize("mutation, message", (
    ("missing_ledger", "campaign ledger must be a regular file"),
    ("wrong_invocation", "campaign ledger must be a regular file"),
    ("reordered_wave", "waves do not cover"),
    ("abort_mismatch", "ledger payload does not match"),
))
def test_cpu_scale_authorization_rejects_forged_cuda_supervisor_ledger(tmp_path, mutation, message) -> None:
    legacy, _ = _write_legacy_receipt_bundle(tmp_path / "legacy")
    server_cpu = _write_server_receipt_artifacts(tmp_path / "server-cpu", stage="server_cpu")
    cuda_abort = _write_cuda_abort_fixture(tmp_path / "cuda-abort")
    ledger_path = cuda_abort.parent / "campaign-ledger" / f"{'f' * 32}.json"
    if mutation == "missing_ledger":
        ledger_path.unlink()
    elif mutation == "wrong_invocation":
        receipt = json.loads(cuda_abort.read_text(encoding="utf-8")); receipt["invocation_id"] = "e" * 32
        cuda_abort.write_text(json.dumps(receipt), encoding="utf-8")
    else:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if mutation == "reordered_wave":
            ledger["waves"][0]["trial_ids"] = list(reversed(ledger["waves"][0]["trial_ids"]))
            ledger["waves"][0]["scheduled_trial_ids"] = ledger["waves"][0]["trial_ids"]
            ledger["waves"][0]["launched_trial_ids"] = ledger["waves"][0]["trial_ids"]
        else:
            ledger["abort_receipt"]["visible_robot_garment_contacts"] = 5
        ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        _require_cpu_scale_authorization(
            _cpu_scale_args(tmp_path, legacy=legacy, server_cpu=server_cpu, cuda_abort=cuda_abort), build_public_matrix(),
        )


def test_cuda_abort_bundle_hash_changes_for_a_rechecksummed_consulted_artifact(tmp_path) -> None:
    cuda_abort = _write_cuda_abort_fixture(tmp_path / "cuda-abort")
    args = argparse.Namespace(policy_revision_file=(tmp_path / "revision"), policy_repo="org/policy", policy_step=1,
        code_revision="b" * 40, asset_revision="e" * 40, simulator_version="isaac-5.1", policy_artifact_sha256="c" * 64,
        image_identity="sha256:image", strategy="canonical", device="cpu", execution_mode="policy_server", workers=4,
        policy_device="cuda:0", groot_revision="d" * 40, groot_python=Path("/venv/bin/python"))
    args.policy_revision_file.write_text("a" * 40, encoding="utf-8")
    expected = _cpu_scale_live_invocation(args, build_public_matrix())
    before = _validate_cuda_abort_evidence(cuda_abort, expected)["bundle_sha256"]
    trial_id = HISTORICAL_CONTROL_IDS[0]
    artifact = cuda_abort.parent / "raw" / trial_id / "annotations.jsonl"
    artifact.write_text(artifact.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    manifest_path = artifact.with_name("SHA256SUMS.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["annotations.jsonl"]["sha256"] = _sha256_file(artifact)
    manifest["annotations.jsonl"]["size"] = artifact.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert _validate_cuda_abort_evidence(cuda_abort, expected)["bundle_sha256"] != before


def test_server_cpu_bundle_hash_changes_for_a_rechecksummed_consulted_artifact(tmp_path) -> None:
    receipt = _write_server_receipt_artifacts(tmp_path / "server-cpu", stage="server_cpu")
    args = _cpu_scale_args(tmp_path, legacy=receipt, server_cpu=receipt, cuda_abort=receipt)
    expected = _cpu_scale_live_invocation(args, build_public_matrix())
    before = _validate_server_cpu_evidence(receipt, expected)["_bundle_sha256"]
    trial_id = HISTORICAL_CONTROL_IDS[0]
    artifact = receipt.parent / "raw" / trial_id / "annotations.jsonl"
    artifact.write_text(artifact.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    manifest_path = artifact.with_name("SHA256SUMS.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["annotations.jsonl"]["sha256"] = _sha256_file(artifact)
    manifest["annotations.jsonl"]["size"] = artifact.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert _validate_server_cpu_evidence(receipt, expected)["_bundle_sha256"] != before


def test_server_cuda_under_ten_never_emits_a_parity_receipt_even_at_abort_threshold_24() -> None:
    assert not _may_emit_parity_receipt("server_cuda", 0)
    assert not _may_emit_parity_receipt("server_cuda", 9)
    assert _may_emit_parity_receipt("server_cuda", 10)


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    (
        ("device", "cuda:0", "--device cpu"),
        ("workers", 3, "exactly --workers 4"),
        ("execution_mode", "direct", "--execution-mode policy_server"),
    ),
)
def test_cpu_scale_authorization_rejects_a_non_cpu_four_worker_policy_server_invocation(tmp_path, attribute, value, message) -> None:
    legacy, _ = _write_legacy_receipt_bundle(tmp_path / "legacy")
    server_cpu = _write_server_receipt_artifacts(tmp_path / "server-cpu", stage="server_cpu")
    cuda_root = tmp_path / "cuda-abort"
    _write_server_receipt_artifacts(cuda_root, stage="server_cuda", successes=0)
    cuda_abort = cuda_root / "campaign-abort-receipt-test.json"
    cuda_abort.write_text(json.dumps({
        "status": "aborted", "reason": "zero_official_successes", "completed_trials": 12,
        "trial_ids": list(HISTORICAL_CONTROL_IDS), "official_successes": 0,
        "visible_robot_garment_contacts": 12,
    }), encoding="utf-8")
    args = _cpu_scale_args(tmp_path, legacy=legacy, server_cpu=server_cpu, cuda_abort=cuda_abort)
    setattr(args, attribute, value)

    with pytest.raises(ValueError, match=message):
        _require_cpu_scale_authorization(args, build_public_matrix())


def test_cpu_scale_authorization_rejects_rehashed_duplicate_server_cpu_resets(tmp_path) -> None:
    legacy, _ = _write_legacy_receipt_bundle(tmp_path / "legacy")
    server_cpu = _write_server_receipt_artifacts(tmp_path / "server-cpu", stage="server_cpu")
    receipt_root = server_cpu.parent
    episode_path = receipt_root / "raw" / HISTORICAL_CONTROL_IDS[-1] / "episode.json"
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    episode["reset_hash"] = f"{1:064x}"
    episode_path.write_text(json.dumps(episode), encoding="utf-8")
    manifest_path = episode_path.with_name("SHA256SUMS.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["episode.json"]["sha256"] = _sha256_file(episode_path)
    manifest["episode.json"]["size"] = episode_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cuda_root = tmp_path / "cuda-abort"
    _write_server_receipt_artifacts(cuda_root, stage="server_cuda", successes=0)
    cuda_abort = cuda_root / "campaign-abort-receipt-test.json"
    cuda_abort.write_text(json.dumps({
        "status": "aborted", "reason": "zero_official_successes", "completed_trials": 12,
        "trial_ids": list(HISTORICAL_CONTROL_IDS), "official_successes": 0,
        "visible_robot_garment_contacts": 12,
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="unique_resets_not_12_of_12"):
        _require_cpu_scale_authorization(
            _cpu_scale_args(tmp_path, legacy=legacy, server_cpu=server_cpu, cuda_abort=cuda_abort),
            build_public_matrix(),
        )


def test_legacy_cpu_reference_builder_derives_a_checksummed_9_of_12_without_simulation(tmp_path, monkeypatch) -> None:
    root = tmp_path / "legacy-existing"
    for index, trial_id in enumerate(HISTORICAL_CONTROL_IDS):
        trial = root / trial_id
        trial.mkdir(parents=True)
        (trial / "trial.json").write_text(
            json.dumps({"metric": {"success": index < 9}, "trial": {"trial_id": trial_id, "category": "top_long", "garment_name": "x", "seed": 42}, "environment_device": "cpu", "command": ["--device", "cpu", "--seed", "42", "--policy_path", "/workspace/checkpoints/runtime-step-12000-local-processor-v2"]}), encoding="utf-8",
        )
    _write_sha256sums(root)
    source = tmp_path / "legacy-source"
    source.mkdir()
    source_hashes = {}
    for name in ("run-lehome-24-shared.sh", "eval_groot_n17_matrix_parallel.py", "groot_policy.py", "serve_groot_policy.py"):
        path = source / name; path.write_text(name, encoding="utf-8"); source_hashes[name] = _sha256_file(path)
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._LEGACY_SHARED_POLICY_HASHES", source_hashes)
    archive = tmp_path / "archive"
    controls = archive
    for trial in historical_control_trials():
        release = controls / trial.trial_id / "garment-config" / "Release"
        release.mkdir(parents=True); (release / "Release_test_list.txt").write_text(trial.garment_name, encoding="utf-8")
    (archive / "rollout-report.json").write_text(json.dumps({"trials": [{"trial": {"trial_id": trial_id}} for trial_id in HISTORICAL_CONTROL_IDS]}), encoding="utf-8")
    _write_sha256sums(archive)
    receipt = build_legacy_cpu_reference_receipt(root, tmp_path / "legacy-reference.json", source_root=source, archive_root=archive)

    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["reference_kind"] == "checksummed_trial_json_reproduction"
    assert payload["historical_runtime_identity_status"] == "not_recorded"
    assert payload["official_successes"] == 9
    assert payload["source_sha256"] == source_hashes
    assert payload["bundle_sha256"]


@pytest.mark.parametrize("mutation", ("source", "archive_sums", "report", "controls", "reproduction", "live_source"))
def test_simple_legacy_reference_rejects_every_bound_evidence_mutation(tmp_path, monkeypatch, mutation) -> None:
    root = tmp_path / "repro"
    for index, trial_id in enumerate(HISTORICAL_CONTROL_IDS):
        path = root / trial_id; path.mkdir(parents=True)
        (path / "trial.json").write_text(json.dumps({"metric": {"success": index < 9}, "trial": {"trial_id": trial_id, "category": "top_long", "garment_name": "x", "seed": 42}, "environment_device": "cpu", "command": ["--device", "cpu", "--seed", "42", "--policy_path", "/workspace/checkpoints/runtime-step-12000-local-processor-v2"]}), encoding="utf-8")
    _write_sha256sums(root)
    source = tmp_path / "source"; source.mkdir()
    hashes = {}
    for name in ("run-lehome-24-shared.sh", "eval_groot_n17_matrix_parallel.py", "groot_policy.py", "serve_groot_policy.py"):
        path = source / name; path.write_text(name, encoding="utf-8"); hashes[name] = _sha256_file(path)
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._LEGACY_SHARED_POLICY_HASHES", hashes)
    archive = tmp_path / "archive"; controls = archive
    for trial in historical_control_trials():
        release = controls / trial.trial_id / "garment-config" / "Release"; release.mkdir(parents=True)
        (release / "Release_test_list.txt").write_text(trial.garment_name, encoding="utf-8")
    report = archive / "rollout-report.json"
    report.write_text(json.dumps({"trials": [{"trial": {"trial_id": item}} for item in HISTORICAL_CONTROL_IDS]}), encoding="utf-8")
    _write_sha256sums(archive)
    receipt = build_legacy_cpu_reference_receipt(root, tmp_path / "receipt.json", source_root=source, archive_root=archive)
    if mutation == "source":
        (source / "groot_policy.py").write_text("mutated", encoding="utf-8")
    elif mutation == "archive_sums":
        (archive / "SHA256SUMS").write_text("bad", encoding="utf-8")
    elif mutation == "report":
        report.write_text("{}", encoding="utf-8")
    elif mutation == "controls":
        (controls / HISTORICAL_CONTROL_IDS[0] / "garment-config" / "Release" / "Release_test_list.txt").write_text("wrong", encoding="utf-8")
    elif mutation == "reproduction":
        (root / HISTORICAL_CONTROL_IDS[0] / "trial.json").write_text("{}", encoding="utf-8")
    expected = {"legacy_source_root": str((tmp_path / "other-source") if mutation == "live_source" else source)}
    with pytest.raises(ValueError):
        _validate_legacy_cpu_reference(receipt, expected)


def test_campaign_rejects_a_second_production_controller(monkeypatch, tmp_path) -> None:
    matrix_path = Path(__file__).parents[2] / "configs" / "eval_groot_n17_public_280.json"
    trial_ids = tuple(trial.trial_id for trial in build_public_matrix().trials[:1])
    started = threading.Event()
    release = threading.Event()
    results: list[BaseException] = []
    scans = 0

    def pending_for_phase(_state: CampaignState) -> tuple[str, ...]:
        nonlocal scans
        scans += 1
        return trial_ids if scans == 1 else ()

    def run_group(_args, assignments, **_kwargs):
        started.set()
        assert release.wait(timeout=2.0)
        return 1.0, len(assignments), 0

    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.pending_trial_ids", pending_for_phase)
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._run_worker_group", run_group)
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._worker_gpu_indices", lambda _args, count: tuple(range(count)))
    args = build_parser().parse_args([
        "--matrix", str(matrix_path), "--policy-path", "policy", "--policy-revision-file", "revision", "--output-root", str(tmp_path),
        "--policy-repo", "org/policy", "--policy-step", "1", "--code-revision", "a" * 40,
        "--asset-revision", "b" * 40, "--release-assets-root", str(tmp_path / "assets" / "Release"), "--simulator-version", "isaac-5.1",
        "--policy-artifact-sha256", "c" * 64, "--image-identity", "sha256:image", "--device", "cuda:0",
        "--groot-root", str(tmp_path / "Isaac-GR00T"), "--groot-revision", "d" * 40,
        "--groot-python", str(tmp_path / "groot-venv" / "bin" / "python"), "--workers", "1",
    ])

    def controller_target() -> None:
        try:
            with _campaign_supervisor_lease(args.output_root):
                _run_supervisor_only(args)
        except BaseException as error:
            results.append(error)

    controller = threading.Thread(target=controller_target)
    controller.start()
    try:
        assert started.wait(timeout=2.0)
        with pytest.raises(ValueError, match="supervisor is already active"):
            with _campaign_supervisor_lease(args.output_root):
                pass
    finally:
        release.set()
        controller.join(timeout=2.0)

    assert not controller.is_alive()
    assert results == []


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


def test_sequential_worker_reaps_child_when_wait_is_interrupted(monkeypatch, tmp_path) -> None:
    events: list[tuple[str, float | None]] = []

    class InterruptingProcess:
        def __init__(self) -> None:
            self.terminated = False
            self.killed = False

        def wait(self, timeout=None):
            events.append(("wait", timeout))
            if self.killed:
                return 0
            if self.terminated:
                raise subprocess.TimeoutExpired("trial", timeout)
            raise KeyboardInterrupt()

        def terminate(self) -> None:
            events.append(("terminate", None))
            self.terminated = True

        def kill(self) -> None:
            events.append(("kill", None))
            self.killed = True

    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.subprocess.Popen", lambda *_args, **_kwargs: InterruptingProcess())

    with pytest.raises(KeyboardInterrupt):
        _run_one_worker(_worker_args(tmp_path), worker_id=1, trial=Trial("pant_long", "Pant_Long_Seen_0", "seen", 42))

    assert events == [("wait", 2.0), ("terminate", None), ("wait", 0.25), ("kill", None), ("wait", None)]


def test_first_ledger_creation_fsyncs_the_campaign_root(monkeypatch, tmp_path) -> None:
    root_inode = tmp_path.stat().st_ino
    root_fsyncs: list[int] = []
    real_fsync = os.fsync

    def fsync(fd: int) -> None:
        if os.fstat(fd).st_ino == root_inode:
            root_fsyncs.append(fd)
        real_fsync(fd)

    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.os.fsync", fsync)

    _write_invocation_checkpoint(tmp_path, "a" * 32, {"schema_version": 1})

    assert root_fsyncs


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


def test_retry_preparation_quarantines_per_trial_receipts_without_episode_staging(tmp_path) -> None:
    trial_id = "trial-001"
    receipt = tmp_path / f"policy-server-receipt-{trial_id}.json"
    manifest = tmp_path / f"flywheel-manifest-{trial_id}.json"
    receipt.write_text('{"port": 40001}\n', encoding="utf-8")
    manifest.write_text('{"episode_id": "trial-001"}\n', encoding="utf-8")

    _prepare_retry_attempt(tmp_path, trial_id)

    attempts = list((tmp_path / "quarantine").glob(f"{trial_id}.attempt-*"))
    assert len(attempts) == 1
    assert (attempts[0] / receipt.name).read_text(encoding="utf-8") == '{"port": 40001}\n'
    assert (attempts[0] / manifest.name).read_text(encoding="utf-8") == '{"episode_id": "trial-001"}\n'
    assert not receipt.exists()
    assert not manifest.exists()


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


def test_campaign_parser_rejects_conflicting_or_unapproved_production_worker_counts(tmp_path) -> None:
    base = [
        "--matrix", str(Path(__file__).parents[2] / "configs" / "eval_groot_n17_public_280.json"),
        "--policy-path", "policy", "--policy-revision-file", "revision", "--output-root", str(tmp_path),
        "--policy-repo", "org/policy", "--policy-step", "1", "--code-revision", "a" * 40,
        "--asset-revision", "b" * 40, "--release-assets-root", str(tmp_path / "assets" / "Release"),
        "--simulator-version", "isaac-5.1", "--policy-artifact-sha256", "c" * 64,
        "--image-identity", "sha256:image",
    ]

    with pytest.raises(SystemExit, match="2"):
        build_parser().parse_args([*base, "--workers", "5"])
    with pytest.raises(SystemExit, match="2"):
        build_parser().parse_args([*base, "--workers", "1", "--capacity-sweep", "1,2,4"])


def test_campaign_execution_rejects_programmatic_worker_oversubscription_before_gpu_inventory(monkeypatch, tmp_path) -> None:
    matrix_path = Path(__file__).parents[2] / "configs" / "eval_groot_n17_public_280.json"
    args = build_parser().parse_args([
        "--matrix", str(matrix_path), "--policy-path", "policy", "--policy-revision-file", "revision", "--output-root", str(tmp_path),
        "--policy-repo", "org/policy", "--policy-step", "1", "--code-revision", "a" * 40,
        "--asset-revision", "b" * 40, "--release-assets-root", str(tmp_path / "assets" / "Release"), "--simulator-version", "isaac-5.1",
        "--policy-artifact-sha256", "c" * 64, "--image-identity", "sha256:image", "--device", "cuda:0",
        "--groot-root", str(tmp_path / "Isaac-GR00T"), "--groot-revision", "d" * 40,
        "--groot-python", str(tmp_path / "groot-venv" / "bin" / "python"), "--workers", "4",
    ])
    args.workers = 5
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign._worker_gpu_indices",
        lambda *_args: (_ for _ in ()).throw(AssertionError("GPU inventory requested")),
    )

    with pytest.raises(ValueError, match="between 1 and 4"):
        run_campaign(args)


def test_campaign_execution_rejects_boolean_worker_count_before_gpu_inventory(monkeypatch, tmp_path) -> None:
    matrix_path = Path(__file__).parents[2] / "configs" / "eval_groot_n17_public_280.json"
    args = build_parser().parse_args([
        "--matrix", str(matrix_path), "--policy-path", "policy", "--policy-revision-file", "revision", "--output-root", str(tmp_path),
        "--policy-repo", "org/policy", "--policy-step", "1", "--code-revision", "a" * 40,
        "--asset-revision", "b" * 40, "--release-assets-root", str(tmp_path / "assets" / "Release"), "--simulator-version", "isaac-5.1",
        "--policy-artifact-sha256", "c" * 64, "--image-identity", "sha256:image", "--device", "cuda:0",
        "--groot-root", str(tmp_path / "Isaac-GR00T"), "--groot-revision", "d" * 40,
        "--groot-python", str(tmp_path / "groot-venv" / "bin" / "python"), "--workers", "1",
    ])
    args.workers = True
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign._worker_gpu_indices",
        lambda *_args: (_ for _ in ()).throw(AssertionError("GPU inventory requested")),
    )

    with pytest.raises(ValueError, match="between 1 and 4"):
        run_campaign(args)


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


def test_scale_cpu_worker_resolves_trial_modules_only_from_the_pinned_runtime_root(tmp_path) -> None:
    args = _worker_args(tmp_path)
    args.parity_stage = "scale_cpu"
    args.trial_runtime_root = tmp_path / "fcf-runtime"
    args.device = "cpu"

    command = _trial_command(args, build_public_matrix().trials[0])
    environment = _worker_environment(args, 0)

    assert command[0] == sys.executable
    assert command[1] == str(args.trial_runtime_root / "scripts" / "run_groot_flywheel_trial.py")
    assert command[command.index("--parity-stage") + 1] == "server_cpu"
    assert environment["PYTHONPATH"] == os.pathsep.join((
        str(args.trial_runtime_root.resolve()), str(args.trial_runtime_root.resolve() / "source" / "lehome"),
        str(args.trial_runtime_root.resolve() / "third_party" / "IsaacLab" / "source" / "isaaclab"),
        str(args.trial_runtime_root.resolve() / "third_party" / "IsaacLab" / "source" / "isaaclab_assets"),
        str(args.trial_runtime_root.resolve() / "third_party" / "IsaacLab" / "source" / "isaaclab_tasks"),
        str(args.trial_runtime_root.resolve() / "third_party" / "IsaacLab" / "source" / "isaaclab_mimic"),
        str(args.trial_runtime_root.resolve() / "third_party" / "IsaacLab" / "source" / "isaaclab_rl"), "/opt/isaacsim/python",
        str(args.groot_root.resolve()),
    ))


def test_cpu_worker_forwards_renderer_gpu_without_hiding_vulkan_devices(tmp_path, monkeypatch) -> None:
    args = _worker_args(tmp_path)
    args.device = "cpu"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")

    environment = _worker_environment(args, 2)

    assert "CUDA_VISIBLE_DEVICES" not in environment
    assert environment["LEHOME_FLYWHEEL_WORKER_GPU"] == "2"


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
    checkpoints = list((tmp_path / "campaign-ledger").glob("*.json"))
    assert len(checkpoints) == 1
    checkpoint = json.loads(checkpoints[0].read_text(encoding="utf-8"))
    assert checkpoint["mode"] == "capacity_sweep"
    assert [wave["status"] for wave in checkpoint["waves"]] == ["terminal", "terminal", "terminal"]


def test_capacity_sweep_persists_partial_launch_accounting_before_reraising(monkeypatch, tmp_path) -> None:
    matrix_path = Path(__file__).parents[2] / "configs" / "eval_groot_n17_public_280.json"

    def run_group(_args, assignments, **_kwargs):
        error = OSError("second capacity launch failed")
        error.scheduled_trial_ids = tuple(trial.trial_id for _, trial in assignments)
        error.launched_trial_ids = ()
        raise error

    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._run_worker_group", run_group)
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._worker_gpu_indices", lambda _args, count: tuple(range(count)))
    args = build_parser().parse_args([
        "--matrix", str(matrix_path), "--policy-path", "policy", "--policy-revision-file", "revision", "--output-root", str(tmp_path),
        "--policy-repo", "org/policy", "--policy-step", "1", "--code-revision", "a" * 40,
        "--asset-revision", "b" * 40, "--release-assets-root", str(tmp_path / "assets" / "Release"), "--simulator-version", "isaac-5.1",
        "--policy-artifact-sha256", "c" * 64, "--image-identity", "sha256:image", "--device", "cuda:0",
        "--groot-root", str(tmp_path / "Isaac-GR00T"), "--groot-revision", "d" * 40,
        "--groot-python", str(tmp_path / "groot-venv" / "bin" / "python"), "--capacity-sweep", "1,2,4",
    ])

    with pytest.raises(OSError, match="second capacity launch failed"):
        run_campaign(args)

    checkpoint = json.loads(next((tmp_path / "campaign-ledger").glob("*.json")).read_text(encoding="utf-8"))
    assert checkpoint["status"] == "failed"
    assert checkpoint["waves"][0]["scheduled_trial_ids"] == [build_public_matrix().trials[0].trial_id]
    assert checkpoint["waves"][0]["launched_trial_ids"] == []


def test_capacity_sweep_persists_zero_launches_when_port_allocation_fails(monkeypatch, tmp_path) -> None:
    matrix_path = Path(__file__).parents[2] / "configs" / "eval_groot_n17_public_280.json"
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign._allocate_loopback_ports",
        lambda _workers: (_ for _ in ()).throw(OSError("loopback allocation failed")),
    )
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign.subprocess.Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("trial must not launch")),
    )
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._worker_gpu_indices", lambda _args, count: tuple(range(count)))
    args = build_parser().parse_args([
        "--matrix", str(matrix_path), "--policy-path", "policy", "--policy-revision-file", "revision", "--output-root", str(tmp_path),
        "--policy-repo", "org/policy", "--policy-step", "1", "--code-revision", "a" * 40,
        "--asset-revision", "b" * 40, "--release-assets-root", str(tmp_path / "assets" / "Release"), "--simulator-version", "isaac-5.1",
        "--policy-artifact-sha256", "c" * 64, "--image-identity", "sha256:image", "--device", "cuda:0",
        "--groot-root", str(tmp_path / "Isaac-GR00T"), "--groot-revision", "d" * 40,
        "--groot-python", str(tmp_path / "groot-venv" / "bin" / "python"), "--capacity-sweep", "1,2,4",
    ])

    with pytest.raises(OSError, match="loopback allocation failed"):
        run_campaign(args)

    checkpoint = json.loads(next((tmp_path / "campaign-ledger").glob("*.json")).read_text(encoding="utf-8"))
    assert checkpoint["status"] == "failed"
    assert checkpoint["waves"][0]["launched_trial_ids"] == []


def test_campaign_workers_runs_all_pending_trials_in_finite_gpu_waves(monkeypatch, tmp_path) -> None:
    matrix_path = Path(__file__).parents[2] / "configs" / "eval_groot_n17_public_280.json"
    trial_ids = tuple(trial.trial_id for trial in build_public_matrix().trials[:7])
    scans: list[tuple[str, ...]] = []
    launches: list[tuple[tuple[int, ...], tuple[int | None, ...]]] = []

    def pending_for_phase(_state: CampaignState) -> tuple[str, ...]:
        scans.append(trial_ids)
        return trial_ids if len(scans) == 1 else ()

    def run_group(_args, assignments, *, gpu_indices=None, **_kwargs):
        launches.append((
            tuple(trial.seed for _, trial in assignments),
            tuple(gpu_indices),
        ))
        return 1.0, len(assignments), 0

    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.pending_trial_ids", pending_for_phase)
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._run_worker_group", run_group)
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign._worker_gpu_indices",
        lambda _args, count: tuple(range(count)),
    )
    args = build_parser().parse_args([
        "--matrix", str(matrix_path), "--policy-path", "policy", "--policy-revision-file", "revision", "--output-root", str(tmp_path),
        "--policy-repo", "org/policy", "--policy-step", "1", "--code-revision", "a" * 40,
        "--asset-revision", "b" * 40, "--release-assets-root", str(tmp_path / "assets" / "Release"), "--simulator-version", "isaac-5.1",
        "--policy-artifact-sha256", "c" * 64, "--image-identity", "sha256:image", "--device", "cuda:0",
        "--groot-root", str(tmp_path / "Isaac-GR00T"), "--groot-revision", "d" * 40,
        "--groot-python", str(tmp_path / "groot-venv" / "bin" / "python"), "--workers", "4",
    ])

    report = _run_supervisor_only(args)

    expected_seeds = [trial.seed for trial in build_public_matrix().trials[:7]]
    assert launches == [(tuple(expected_seeds[:4]), (0, 1, 2, 3)), (tuple(expected_seeds[4:]), (0, 1, 2))]
    assert report["production"] == {"workers": 4, "status": "completed", "waves": 2}
    assert report["episode_accounting"]["production_wave_trial_ids"] == [list(trial_ids[:4]), list(trial_ids[4:])]
    assert report["episode_accounting"]["attempted_unique_trial_ids"] == sorted(trial_ids)


def test_sequential_campaign_fails_closed_on_a_nonzero_worker(monkeypatch, tmp_path) -> None:
    matrix_path = Path(__file__).parents[2] / "configs" / "eval_groot_n17_public_280.json"
    trial_ids = tuple(trial.trial_id for trial in build_public_matrix().trials[:1])
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.pending_trial_ids", lambda _state: trial_ids)
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._run_one_worker", lambda *_args, **_kwargs: 1)
    args = build_parser().parse_args([
        "--matrix", str(matrix_path), "--policy-path", "policy", "--policy-revision-file", "revision", "--output-root", str(tmp_path),
        "--policy-repo", "org/policy", "--policy-step", "1", "--code-revision", "a" * 40,
        "--asset-revision", "b" * 40, "--release-assets-root", str(tmp_path / "assets" / "Release"), "--simulator-version", "isaac-5.1",
        "--policy-artifact-sha256", "c" * 64, "--image-identity", "sha256:image", "--device", "cuda:0",
        "--groot-root", str(tmp_path / "Isaac-GR00T"), "--groot-revision", "d" * 40,
        "--groot-python", str(tmp_path / "groot-venv" / "bin" / "python"),
    ])

    with pytest.raises(RuntimeError, match="sequential worker 1 failed"):
        _run_supervisor_only(args)

    report = json.loads((tmp_path / "capacity-report.json").read_text(encoding="utf-8"))
    assert report["workers"] == [{"mode": "sequential", "returncode": 1, "trial_id": trial_ids[0], "worker_id": 1}]
    checkpoint = json.loads(next((tmp_path / "campaign-ledger").glob("*.json")).read_text(encoding="utf-8"))
    assert checkpoint["status"] == "failed"


def test_sequential_campaign_fails_closed_on_an_incomplete_zero_exit(monkeypatch, tmp_path) -> None:
    matrix_path = Path(__file__).parents[2] / "configs" / "eval_groot_n17_public_280.json"
    trial_ids = tuple(trial.trial_id for trial in build_public_matrix().trials[:1])
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.pending_trial_ids", lambda _state: trial_ids)
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._run_one_worker", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.is_completed_trial", lambda *_args: False)
    args = build_parser().parse_args([
        "--matrix", str(matrix_path), "--policy-path", "policy", "--policy-revision-file", "revision", "--output-root", str(tmp_path),
        "--policy-repo", "org/policy", "--policy-step", "1", "--code-revision", "a" * 40,
        "--asset-revision", "b" * 40, "--release-assets-root", str(tmp_path / "assets" / "Release"), "--simulator-version", "isaac-5.1",
        "--policy-artifact-sha256", "c" * 64, "--image-identity", "sha256:image", "--device", "cuda:0",
        "--groot-root", str(tmp_path / "Isaac-GR00T"), "--groot-revision", "d" * 40,
        "--groot-python", str(tmp_path / "groot-venv" / "bin" / "python"),
    ])

    with pytest.raises(RuntimeError, match="sequential worker 1 failed: returncode=0"):
        _run_supervisor_only(args)


def test_sequential_campaign_marks_checkpoint_interrupted_before_reraising(monkeypatch, tmp_path) -> None:
    matrix_path = Path(__file__).parents[2] / "configs" / "eval_groot_n17_public_280.json"
    trial_ids = tuple(trial.trial_id for trial in build_public_matrix().trials[:1])
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.pending_trial_ids", lambda _state: trial_ids)
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign._run_one_worker",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    args = build_parser().parse_args([
        "--matrix", str(matrix_path), "--policy-path", "policy", "--policy-revision-file", "revision", "--output-root", str(tmp_path),
        "--policy-repo", "org/policy", "--policy-step", "1", "--code-revision", "a" * 40,
        "--asset-revision", "b" * 40, "--release-assets-root", str(tmp_path / "assets" / "Release"), "--simulator-version", "isaac-5.1",
        "--policy-artifact-sha256", "c" * 64, "--image-identity", "sha256:image", "--device", "cuda:0",
        "--groot-root", str(tmp_path / "Isaac-GR00T"), "--groot-revision", "d" * 40,
        "--groot-python", str(tmp_path / "groot-venv" / "bin" / "python"),
    ])

    with pytest.raises(KeyboardInterrupt):
        _run_supervisor_only(args)

    checkpoint = json.loads(next((tmp_path / "campaign-ledger").glob("*.json")).read_text(encoding="utf-8"))
    assert checkpoint["status"] == "interrupted"
    assert checkpoint["waves"][-1]["status"] == "interrupted"


def test_campaign_workers_empty_resume_does_not_inventory_gpus(monkeypatch, tmp_path) -> None:
    matrix_path = Path(__file__).parents[2] / "configs" / "eval_groot_n17_public_280.json"
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.pending_trial_ids", lambda _state: ())
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign._worker_gpu_indices",
        lambda *_args: (_ for _ in ()).throw(AssertionError("GPU inventory requested")),
    )
    args = build_parser().parse_args([
        "--matrix", str(matrix_path), "--policy-path", "policy", "--policy-revision-file", "revision", "--output-root", str(tmp_path),
        "--policy-repo", "org/policy", "--policy-step", "1", "--code-revision", "a" * 40,
        "--asset-revision", "b" * 40, "--release-assets-root", str(tmp_path / "assets" / "Release"), "--simulator-version", "isaac-5.1",
        "--policy-artifact-sha256", "c" * 64, "--image-identity", "sha256:image", "--device", "cuda:0",
        "--groot-root", str(tmp_path / "Isaac-GR00T"), "--groot-revision", "d" * 40,
        "--groot-python", str(tmp_path / "groot-venv" / "bin" / "python"), "--workers", "4",
    ])

    report = _run_supervisor_only(args)

    assert report["production"] == {"workers": 4, "status": "completed", "waves": 0}
    assert report["completed_after"] == [trial.trial_id for trial in build_public_matrix().trials]


def test_campaign_writes_atomic_production_checkpoints_before_and_after_each_wave(monkeypatch, tmp_path) -> None:
    matrix_path = Path(__file__).parents[2] / "configs" / "eval_groot_n17_public_280.json"
    trial_ids = tuple(trial.trial_id for trial in build_public_matrix().trials[:1])
    pending_scans = 0
    replacements: list[tuple[str, str]] = []
    real_replace = os.replace

    def pending_for_phase(_state: CampaignState) -> tuple[str, ...]:
        nonlocal pending_scans
        pending_scans += 1
        return trial_ids if pending_scans == 1 else ()

    def run_group(_args, assignments, **_kwargs):
        files = list((tmp_path / "campaign-ledger").glob("*.json"))
        assert len(files) == 1
        checkpoint = json.loads(files[0].read_text(encoding="utf-8"))
        assert checkpoint["status"] == "running"
        assert checkpoint["waves"][-1]["status"] == "started"
        assert checkpoint["waves"][-1]["trial_ids"] == list(trial_ids)
        return 1.0, len(assignments), 0

    def replace(source, destination, *args, **kwargs):
        replacements.append((os.fspath(source), os.fspath(destination)))
        return real_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.pending_trial_ids", pending_for_phase)
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._run_worker_group", run_group)
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._worker_gpu_indices", lambda _args, count: tuple(range(count)))
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.os.replace", replace)
    args = build_parser().parse_args([
        "--matrix", str(matrix_path), "--policy-path", "policy", "--policy-revision-file", "revision", "--output-root", str(tmp_path),
        "--policy-repo", "org/policy", "--policy-step", "1", "--code-revision", "a" * 40,
        "--asset-revision", "b" * 40, "--release-assets-root", str(tmp_path / "assets" / "Release"), "--simulator-version", "isaac-5.1",
        "--policy-artifact-sha256", "c" * 64, "--image-identity", "sha256:image", "--device", "cuda:0",
        "--groot-root", str(tmp_path / "Isaac-GR00T"), "--groot-revision", "d" * 40,
        "--groot-python", str(tmp_path / "groot-venv" / "bin" / "python"), "--workers", "1",
    ])

    _run_supervisor_only(args)

    files = list((tmp_path / "campaign-ledger").glob("*.json"))
    assert len(files) == 1
    checkpoint = json.loads(files[0].read_text(encoding="utf-8"))
    assert checkpoint["status"] == "completed"
    assert checkpoint["waves"][-1]["status"] == "terminal"
    assert replacements
    assert not list((tmp_path / "campaign-ledger").glob("*.tmp"))


def test_campaign_checkpoint_marks_an_interrupted_wave_before_reraising(monkeypatch, tmp_path) -> None:
    matrix_path = Path(__file__).parents[2] / "configs" / "eval_groot_n17_public_280.json"
    trial_ids = tuple(trial.trial_id for trial in build_public_matrix().trials[:1])
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.pending_trial_ids", lambda _state: trial_ids)
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._worker_gpu_indices", lambda _args, count: tuple(range(count)))
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign._run_worker_group",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    args = build_parser().parse_args([
        "--matrix", str(matrix_path), "--policy-path", "policy", "--policy-revision-file", "revision", "--output-root", str(tmp_path),
        "--policy-repo", "org/policy", "--policy-step", "1", "--code-revision", "a" * 40,
        "--asset-revision", "b" * 40, "--release-assets-root", str(tmp_path / "assets" / "Release"), "--simulator-version", "isaac-5.1",
        "--policy-artifact-sha256", "c" * 64, "--image-identity", "sha256:image", "--device", "cuda:0",
        "--groot-root", str(tmp_path / "Isaac-GR00T"), "--groot-revision", "d" * 40,
        "--groot-python", str(tmp_path / "groot-venv" / "bin" / "python"), "--workers", "1",
    ])

    with pytest.raises(KeyboardInterrupt):
        _run_supervisor_only(args)

    checkpoint = json.loads(next((tmp_path / "campaign-ledger").glob("*.json")).read_text(encoding="utf-8"))
    assert checkpoint["status"] == "interrupted"
    assert checkpoint["waves"][-1]["status"] == "interrupted"


def test_campaign_ledger_preserves_prior_capacity_invocations(tmp_path) -> None:
    matrix_path = Path(__file__).parents[2] / "configs" / "eval_groot_n17_public_280.json"
    base = [
        "--matrix", str(matrix_path), "--policy-path", "policy", "--policy-revision-file", "revision", "--output-root", str(tmp_path),
        "--policy-repo", "org/policy", "--policy-step", "1", "--code-revision", "a" * 40,
        "--asset-revision", "b" * 40, "--release-assets-root", str(tmp_path / "assets" / "Release"), "--simulator-version", "isaac-5.1",
        "--policy-artifact-sha256", "c" * 64, "--image-identity", "sha256:image", "--dry-run",
    ]

    run_campaign(build_parser().parse_args([*base, "--capacity-sweep", "1,2,4"]))
    run_campaign(build_parser().parse_args(base))

    checkpoints = [json.loads(path.read_text(encoding="utf-8")) for path in (tmp_path / "campaign-ledger").glob("*.json")]
    assert len(checkpoints) == 2
    assert {checkpoint["mode"] for checkpoint in checkpoints} == {"capacity_sweep", "sequential"}
    assert all(checkpoint["status"] == "completed" for checkpoint in checkpoints)


def test_campaign_workers_continues_terminal_failures_until_the_finite_cohort_is_accounted(monkeypatch, tmp_path) -> None:
    matrix_path = Path(__file__).parents[2] / "configs" / "eval_groot_n17_public_280.json"
    trial_ids = tuple(trial.trial_id for trial in build_public_matrix().trials[:7])
    launches: list[tuple[str, ...]] = []

    def run_group(_args, assignments, **_kwargs):
        launches.append(tuple(trial.trial_id for _, trial in assignments))
        return 1.0, len(assignments) - 1, 1

    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.pending_trial_ids", lambda _state: trial_ids)
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._run_worker_group", run_group)
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign._worker_gpu_indices",
        lambda _args, count: tuple(range(count)),
    )
    args = build_parser().parse_args([
        "--matrix", str(matrix_path), "--policy-path", "policy", "--policy-revision-file", "revision", "--output-root", str(tmp_path),
        "--policy-repo", "org/policy", "--policy-step", "1", "--code-revision", "a" * 40,
        "--asset-revision", "b" * 40, "--release-assets-root", str(tmp_path / "assets" / "Release"), "--simulator-version", "isaac-5.1",
        "--policy-artifact-sha256", "c" * 64, "--image-identity", "sha256:image", "--device", "cuda:0",
        "--groot-root", str(tmp_path / "Isaac-GR00T"), "--groot-revision", "d" * 40,
        "--groot-python", str(tmp_path / "groot-venv" / "bin" / "python"), "--workers", "4",
    ])

    with pytest.raises(RuntimeError, match="terminal-incomplete trials remain"):
        _run_supervisor_only(args)

    assert launches == [trial_ids[:4], trial_ids[4:]]
    report = json.loads((tmp_path / "capacity-report.json").read_text(encoding="utf-8"))
    assert report["production"] == {"workers": 4, "status": "failed", "waves": 2}
    assert report["workers"][0]["failed_trials"] == 1


def test_campaign_workers_reports_a_group_launch_error_without_starting_another_wave(monkeypatch, tmp_path) -> None:
    matrix_path = Path(__file__).parents[2] / "configs" / "eval_groot_n17_public_280.json"
    trial_ids = tuple(trial.trial_id for trial in build_public_matrix().trials[:7])
    launches: list[tuple[str, ...]] = []

    def run_group(_args, assignments, **_kwargs):
        launches.append(tuple(trial.trial_id for _, trial in assignments))
        error = OSError("policy server launch failed")
        error.scheduled_trial_ids = tuple(trial.trial_id for _, trial in assignments)
        error.launched_trial_ids = (assignments[0][1].trial_id,)
        raise error

    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.pending_trial_ids", lambda _state: trial_ids)
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._run_worker_group", run_group)
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign._worker_gpu_indices",
        lambda _args, count: tuple(range(count)),
    )
    args = build_parser().parse_args([
        "--matrix", str(matrix_path), "--policy-path", "policy", "--policy-revision-file", "revision", "--output-root", str(tmp_path),
        "--policy-repo", "org/policy", "--policy-step", "1", "--code-revision", "a" * 40,
        "--asset-revision", "b" * 40, "--release-assets-root", str(tmp_path / "assets" / "Release"), "--simulator-version", "isaac-5.1",
        "--policy-artifact-sha256", "c" * 64, "--image-identity", "sha256:image", "--device", "cuda:0",
        "--groot-root", str(tmp_path / "Isaac-GR00T"), "--groot-revision", "d" * 40,
        "--groot-python", str(tmp_path / "groot-venv" / "bin" / "python"), "--workers", "4",
    ])

    with pytest.raises(RuntimeError, match="production wave 1 failed: policy server launch failed"):
        _run_supervisor_only(args)

    assert launches == [trial_ids[:4]]
    report = json.loads((tmp_path / "capacity-report.json").read_text(encoding="utf-8"))
    assert report["production"] == {"workers": 4, "status": "failed", "waves": 1}
    assert report["workers"][0]["status"] == "launch_error"
    assert report["workers"][0]["scheduled_trial_ids"] == list(trial_ids[:4])
    assert report["workers"][0]["launched_trial_ids"] == [trial_ids[0]]
    assert report["episode_accounting"] == {
        "sequential_trial_ids": [],
        "capacity_wave_trial_ids": [],
        "production_wave_trial_ids": [[trial_ids[0]]],
        "attempt_count": 1,
        "attempted_unique_trial_ids": [trial_ids[0]],
    }
    ledger = json.loads(next((tmp_path / "campaign-ledger").glob("*.json")).read_text(encoding="utf-8"))
    assert ledger["waves"][0]["scheduled_trial_ids"] == list(trial_ids[:4])
    assert ledger["waves"][0]["launched_trial_ids"] == [trial_ids[0]]


def test_campaign_workers_persists_zero_launches_when_port_allocation_fails(monkeypatch, tmp_path) -> None:
    matrix_path = Path(__file__).parents[2] / "configs" / "eval_groot_n17_public_280.json"
    trial_ids = tuple(trial.trial_id for trial in build_public_matrix().trials[:4])
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.pending_trial_ids", lambda _state: trial_ids)
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._worker_gpu_indices", lambda _args, count: tuple(range(count)))
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign._allocate_loopback_ports",
        lambda _workers: (_ for _ in ()).throw(OSError("loopback allocation failed")),
    )
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign.subprocess.Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("trial must not launch")),
    )
    args = build_parser().parse_args([
        "--matrix", str(matrix_path), "--policy-path", "policy", "--policy-revision-file", "revision", "--output-root", str(tmp_path),
        "--policy-repo", "org/policy", "--policy-step", "1", "--code-revision", "a" * 40,
        "--asset-revision", "b" * 40, "--release-assets-root", str(tmp_path / "assets" / "Release"), "--simulator-version", "isaac-5.1",
        "--policy-artifact-sha256", "c" * 64, "--image-identity", "sha256:image", "--device", "cuda:0",
        "--groot-root", str(tmp_path / "Isaac-GR00T"), "--groot-revision", "d" * 40,
        "--groot-python", str(tmp_path / "groot-venv" / "bin" / "python"), "--workers", "4",
    ])

    with pytest.raises(RuntimeError, match="production wave 1 failed: loopback allocation failed"):
        _run_supervisor_only(args)

    report = json.loads((tmp_path / "capacity-report.json").read_text(encoding="utf-8"))
    assert report["workers"][0]["scheduled_trial_ids"] == list(trial_ids)
    assert report["workers"][0]["launched_trial_ids"] == []
    assert report["episode_accounting"]["attempt_count"] == 0
    assert report["episode_accounting"]["attempted_unique_trial_ids"] == []
    ledger = json.loads(next((tmp_path / "campaign-ledger").glob("*.json")).read_text(encoding="utf-8"))
    assert ledger["waves"][0]["launched_trial_ids"] == []


def test_campaign_workers_fails_if_a_successful_wave_leaves_pending_artifacts(monkeypatch, tmp_path) -> None:
    matrix_path = Path(__file__).parents[2] / "configs" / "eval_groot_n17_public_280.json"
    trial_ids = tuple(trial.trial_id for trial in build_public_matrix().trials[:4])
    launches: list[tuple[str, ...]] = []

    def run_group(_args, assignments, **_kwargs):
        launches.append(tuple(trial.trial_id for _, trial in assignments))
        return 1.0, len(assignments), 0

    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.pending_trial_ids", lambda _state: trial_ids)
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._run_worker_group", run_group)
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign._worker_gpu_indices",
        lambda _args, count: tuple(range(count)),
    )
    args = build_parser().parse_args([
        "--matrix", str(matrix_path), "--policy-path", "policy", "--policy-revision-file", "revision", "--output-root", str(tmp_path),
        "--policy-repo", "org/policy", "--policy-step", "1", "--code-revision", "a" * 40,
        "--asset-revision", "b" * 40, "--release-assets-root", str(tmp_path / "assets" / "Release"), "--simulator-version", "isaac-5.1",
        "--policy-artifact-sha256", "c" * 64, "--image-identity", "sha256:image", "--device", "cuda:0",
        "--groot-root", str(tmp_path / "Isaac-GR00T"), "--groot-revision", "d" * 40,
        "--groot-python", str(tmp_path / "groot-venv" / "bin" / "python"), "--workers", "4",
    ])

    with pytest.raises(RuntimeError, match="terminal-incomplete trials remain"):
        _run_supervisor_only(args)

    assert launches == [trial_ids]
    report = json.loads((tmp_path / "capacity-report.json").read_text(encoding="utf-8"))
    assert report["production"] == {"workers": 4, "status": "failed", "waves": 1}


def test_campaign_main_returns_nonzero_for_a_reported_production_failure(monkeypatch, capsys) -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args = lambda _argv: argparse.Namespace()
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.build_parser", lambda: parser)
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign.run_campaign",
        lambda _args: (_ for _ in ()).throw(RuntimeError("production wave 1 failed")),
    )

    assert main([]) == 1
    assert "campaign execution error: production wave 1 failed" in capsys.readouterr().err


def test_cpu_simulator_still_assigns_a_physical_gpu_to_the_policy_server(monkeypatch, tmp_path) -> None:
    args = _worker_args(tmp_path)
    args.device = "cpu"
    args.policy_device = "cuda:2"
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._visible_gpu_indices", lambda: (0, 1, 2, 3))

    assert _worker_gpu_indices(args, 1) == (2,)


def test_production_worker_gpu_assignment_rejects_oversubscription(monkeypatch, tmp_path) -> None:
    args = _worker_args(tmp_path)
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign._visible_gpu_indices", lambda: (0, 1))

    with pytest.raises(ValueError, match="unsupported GPU oversubscription"):
        _worker_gpu_indices(args, 3)


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


def test_worker_group_records_only_popen_successes_when_a_later_launch_fails(monkeypatch, tmp_path) -> None:
    launched = []
    assignments = _group_trials(2)

    def launch(*_args, **_kwargs):
        if launched:
            raise OSError("second Popen failed")
        launched.append(True)
        return _GracefulAfterTerminateProcess([])

    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.subprocess.Popen", launch)

    with pytest.raises(OSError, match="second Popen failed") as raised:
        _run_worker_group(_worker_args(tmp_path), assignments)

    assert raised.value.scheduled_trial_ids == tuple(trial.trial_id for _, trial in assignments)
    assert raised.value.launched_trial_ids == (assignments[0][1].trial_id,)


def test_worker_group_kills_a_policy_server_descendant_after_its_trial_parent_exits(monkeypatch, tmp_path) -> None:
    signals: list[tuple[int, int]] = []
    launches = []
    clock = [0.0]
    policy_server_alive = True
    waits = []

    class ParentExitedWhilePolicyServerLives:
        pid = 5151

        def poll(self):
            return 0

        def wait(self, timeout=None):
            waits.append(timeout)
            return 0

    def popen(*_args, **kwargs):
        launches.append(kwargs)
        return ParentExitedWhilePolicyServerLives()

    def killpg(process_group: int, signum: int) -> None:
        nonlocal policy_server_alive
        signals.append((process_group, signum))
        if signum == 0:
            if not policy_server_alive:
                raise ProcessLookupError
        elif signum == signal.SIGKILL:
            policy_server_alive = False

    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.subprocess.Popen", popen)
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.os.killpg", killpg)
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign.time.sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.is_completed_trial", lambda *_args: True)

    _, completed, failed = _run_worker_group(_worker_args(tmp_path), _group_trials(1))

    assert (completed, failed) == (1, 0)
    assert launches[0]["start_new_session"] is True
    assert (5151, signal.SIGTERM) in signals
    assert (5151, signal.SIGKILL) in signals
    assert policy_server_alive is False
    assert waits == [0.25]


def test_sequential_worker_kills_a_policy_server_descendant_after_nonzero_wait(monkeypatch, tmp_path) -> None:
    signals: list[tuple[int, int]] = []
    clock = [0.0]
    policy_server_alive = True

    class ParentExitedWhilePolicyServerLives:
        pid = 6161

        def wait(self, timeout=None):
            return 17

        def poll(self):
            return 17

    def killpg(process_group: int, signum: int) -> None:
        nonlocal policy_server_alive
        signals.append((process_group, signum))
        if signum == 0:
            if not policy_server_alive:
                raise ProcessLookupError
        elif signum == signal.SIGKILL:
            policy_server_alive = False

    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign.subprocess.Popen",
        lambda *_args, **_kwargs: ParentExitedWhilePolicyServerLives(),
    )
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.os.killpg", killpg)
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign.time.sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )

    assert _run_one_worker(_worker_args(tmp_path), worker_id=1, trial=_group_trials(1)[0][1]) == 17
    assert (6161, signal.SIGTERM) in signals
    assert (6161, signal.SIGKILL) in signals
    assert policy_server_alive is False


def test_worker_group_fails_closed_when_a_policy_server_group_survives_sigkill(monkeypatch, tmp_path) -> None:
    clock = [0.0]

    class ParentExitedWhilePolicyServerLives:
        pid = 7171

        def poll(self):
            return 17

        def wait(self, timeout=None):
            return 17

    def killpg(_process_group: int, signum: int) -> None:
        if signum not in {0, signal.SIGTERM, signal.SIGKILL}:
            raise AssertionError("unexpected signal")

    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign.subprocess.Popen",
        lambda *_args, **_kwargs: ParentExitedWhilePolicyServerLives(),
    )
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.os.killpg", killpg)
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign.time.sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.is_completed_trial", lambda *_args: False)

    with pytest.raises(RuntimeError, match="worker process group 7171 survived SIGKILL"):
        _run_worker_group(_worker_args(tmp_path), _group_trials(1))


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


def test_worker_group_reaps_all_siblings_when_polling_is_interrupted(monkeypatch, tmp_path) -> None:
    events: list[tuple[str, int, float | None]] = []
    processes = iter((_LaunchCleanupProcess(1, events), _LaunchCleanupProcess(2, events)))
    monkeypatch.setattr("scripts.run_groot_flywheel_campaign.subprocess.Popen", lambda *_args, **_kwargs: next(processes))
    monkeypatch.setattr(
        "scripts.run_groot_flywheel_campaign._trial_has_first_progress",
        lambda *_args: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        _run_worker_group(_worker_args(tmp_path), _group_trials(2))

    assert [(event, worker) for event, worker, _ in events if event in {"terminate", "kill"}] == [
        ("terminate", 1), ("terminate", 2), ("kill", 1), ("kill", 2),
    ]
    assert [(event, worker) for event, worker, _ in events if event == "wait"] == [
        ("wait", 1), ("wait", 2),
    ]


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
