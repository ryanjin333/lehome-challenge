from __future__ import annotations

from pathlib import Path
import json

import pytest

from lehome_train.b1k.lifecycle import (
    LifecycleController,
    TrainingFailure,
    assemble_run_contract,
    configure_production_controller,
    main,
)
from lehome_train.b1k.launch import B1KLaunchFailure
from lehome_train.constants import (
    BEHAVIOR_1K_CHECKPOINT_BUCKET,
    BEHAVIOR_1K_FINAL_MODEL_REPOSITORY,
)
from lehome_train.b1k.training import approved_launch_plans



def test_oom_before_progress_retries_next_approved_plan_but_after_progress_does_not(tmp_path: Path) -> None:
    attempts: list[int] = []
    def run(plan: object, _resume: bool, published: object) -> int:
        attempts.append(plan.physical_batch_size)
        if len(attempts) == 1: raise TrainingFailure("CUDA out of memory", optimizer_step=0)
        for step in range(1_000, 15_001, 1_000):
            published(step)
        return 15_000
    controller = LifecycleController(run_training=run, publish_checkpoint=lambda _step: None, world_size=1, output=tmp_path, finalize=lambda: {"immutable_commit": "f" * 40})
    assert controller.run() == 15_000 and attempts == [64, 32]
    controller = LifecycleController(run_training=lambda _p, _r, _published: (_ for _ in ()).throw(TrainingFailure("CUDA out of memory", optimizer_step=1)), publish_checkpoint=lambda _s: None, world_size=1, output=tmp_path / "second")
    with pytest.raises(TrainingFailure): controller.run()


def test_module_main_dispatches_preflight_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class Paths: output = tmp_path
    monkeypatch.setattr("lehome_train.b1k.lifecycle.production_preflight", lambda: Paths())
    assert main(["--preflight-only"]) == 0
    assert (tmp_path / "run-status.json").exists()


def test_module_main_dispatches_configured_controller_without_the_deliberate_runtime_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class Paths: output = tmp_path
    monkeypatch.setattr("lehome_train.b1k.lifecycle.production_preflight", lambda: Paths())
    configure_production_controller(
        lambda _paths: LifecycleController(
            run_training=lambda _plan, _resume, published: [published(step) for step in range(1_000, 15_001, 1_000)] and 15_000,
            publish_checkpoint=lambda _step: None,
            world_size=1,
            output=tmp_path,
            finalize=lambda: {"immutable_commit": "f" * 40},
        )
    )
    try:
        assert main([]) == 0
        assert json.loads((tmp_path / "run-status.json").read_text())["phase"] == "complete"
    finally:
        configure_production_controller(None)


def test_main_preserves_the_controller_failure_status_with_attempts_and_reason(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class Paths: output = tmp_path
    monkeypatch.setattr("lehome_train.b1k.lifecycle.production_preflight", lambda: Paths())
    configure_production_controller(
        lambda _paths: LifecycleController(
            run_training=lambda _plan, _resume, _published: (_ for _ in ()).throw(TrainingFailure("CUDA out of memory", optimizer_step=1)),
            publish_checkpoint=lambda _step: None,
            world_size=1,
            output=tmp_path,
        )
    )
    try:
        assert main([]) == 1
        status = json.loads((tmp_path / "run-status.json").read_text())
        assert status["phase"] == "failed"
        assert status["attempts"][0]["optimizer_step"] == 1
        assert status["reason"] == "TrainingFailure"
        assert status["reason"] != "production-controller-failed"
    finally:
        configure_production_controller(None)


@pytest.mark.parametrize(("number", "reason"), [(15, "signal-sigterm"), (2, "signal-sigint")])
def test_interrupted_trainer_failure_enters_the_detailed_failed_status_path(tmp_path: Path, number: int, reason: str) -> None:
    controller = LifecycleController(
        run_training=lambda _plan, _resume, _published: (_ for _ in ()).throw(B1KLaunchFailure(f"B1K torchrun interrupted by signal {number}", optimizer_step=0, signal_number=number)),
        publish_checkpoint=lambda _step: None,
        world_size=1,
        output=tmp_path,
    )

    with pytest.raises(B1KLaunchFailure, match="interrupted"):
        controller.run()
    status = json.loads((tmp_path / "run-status.json").read_text())
    assert status["phase"] == "failed"
    assert status["reason"] == reason
    assert status["attempts"][0]["optimizer_step"] == 0


def test_assembled_contract_uses_only_b1k_publication_targets() -> None:
    plan = approved_launch_plans(num_gpus=1)[0]
    contract = assemble_run_contract(
        token="not-a-real-token",
        run_id="b1k-20260808-001",
        cycle_id="cycle-001",
        container_digest="sha256:" + "a" * 64,
        world_size=1,
        task_manifest_sha256="b" * 64,
        modality_sha256="ca3a2e406472650bee9439ed81a3f4a1531b6fe689cdc1b348d0f260a208e641",
        stats_sha256="c" * 64,
        plan=plan,
        launch_arguments_sha256="d" * 64,
    )

    assert contract.model_repo == BEHAVIOR_1K_FINAL_MODEL_REPOSITORY
    assert contract.checkpoint_bucket == BEHAVIOR_1K_CHECKPOINT_BUCKET


@pytest.mark.parametrize("resume_policy", ("auto", "never", "require"))
def test_assembled_contract_preserves_the_actual_resume_policy(resume_policy: str) -> None:
    plan = approved_launch_plans(num_gpus=1)[0]

    contract = assemble_run_contract(
        token="not-a-real-token",
        run_id="b1k-20260808-001",
        cycle_id="cycle-001",
        container_digest="sha256:" + "a" * 64,
        world_size=1,
        task_manifest_sha256="b" * 64,
        modality_sha256="ca3a2e406472650bee9439ed81a3f4a1531b6fe689cdc1b348d0f260a208e641",
        stats_sha256="c" * 64,
        plan=plan,
        launch_arguments_sha256="d" * 64,
        resume_policy=resume_policy,
    )

    assert contract.resume_policy == resume_policy


def test_assembled_contract_preserves_the_actual_approved_fallback_plan_and_base_argv() -> None:
    plan = approved_launch_plans(num_gpus=1)[1]
    contract = assemble_run_contract(
        token="not-a-real-token",
        run_id="b1k-20260808-001",
        cycle_id="cycle-001",
        container_digest="sha256:" + "a" * 64,
        world_size=1,
        task_manifest_sha256="b" * 64,
        modality_sha256="ca3a2e406472650bee9439ed81a3f4a1531b6fe689cdc1b348d0f260a208e641",
        stats_sha256="c" * 64,
        plan=plan,
        launch_arguments_sha256="d" * 64,
    )

    assert contract.launch_plan_id == plan.identity
    assert contract.physical_batch_size == plan.physical_batch_size
    assert contract.effective_global_batch_size == plan.effective_global_batch_size
    assert contract.launch_arguments_sha256 == "d" * 64


def test_controller_bootstraps_resumes_publishes_all_boundaries_and_finalizes_before_complete(tmp_path: Path) -> None:
    events: list[object] = []

    def run(plan: object, resume: bool, published: object) -> int:
        events.append(("train", plan.identity, resume))
        for step in range(15_000 if resume else 1_000, 15_001, 1_000):
            published(step)
        return 15_000

    controller = LifecycleController(
        run_training=run,
        publish_checkpoint=lambda step: events.append(("checkpoint", step)),
        world_size=1,
        output=tmp_path,
        resume_policy="require",
        bootstrap=lambda: events.append("bootstrap"),
        select_resume=lambda: tmp_path / "restore" / "checkpoint-14000",
        finalize=lambda: events.append("finalize") or {"immutable_commit": "f" * 40},
    )

    assert controller.run() == 15_000
    assert events[0] == "bootstrap"
    assert events[1][0:2] == ("train", "b1k-gpu1-effective-batch256")
    assert events[1][2] is True
    assert [event[1] for event in events if isinstance(event, tuple) and event[0] == "checkpoint"] == [15_000]
    assert events[-1] == "finalize"
    status = json.loads((tmp_path / "run-status.json").read_text())
    assert status["phase"] == "complete"
    assert status["resume_policy"] == "require"


def test_controller_finalizes_a_verified_step_15000_resume_without_starting_torchrun(tmp_path: Path) -> None:
    events: list[str] = []

    controller = LifecycleController(
        run_training=lambda *_args: pytest.fail("a complete restored checkpoint must not relaunch torchrun"),
        publish_checkpoint=lambda _step: pytest.fail("a complete restored checkpoint must not republish"),
        world_size=1,
        output=tmp_path,
        select_resume=lambda: tmp_path / "checkpoint-15000",
        finalize=lambda: events.append("finalized") or {"immutable_commit": "f" * 40},
    )

    assert controller.run() == 15_000
    assert events == ["finalized"]
    status = json.loads((tmp_path / "run-status.json").read_text())
    assert status["phase"] == "complete"
    assert status["resumed"] is True


def test_controller_marks_failure_without_oom_fallback_after_remote_state_or_progress(tmp_path: Path) -> None:
    attempts: list[int] = []

    def run(plan: object, _resume: bool, _published: object) -> int:
        attempts.append(plan.physical_batch_size)
        raise TrainingFailure("CUDA out of memory hf_secret-value", optimizer_step=0)

    controller = LifecycleController(
        run_training=run,
        publish_checkpoint=lambda _step: None,
        world_size=1,
        output=tmp_path,
        remote_state_exists=lambda: True,
    )

    with pytest.raises(TrainingFailure):
        controller.run()
    assert attempts == [64]
    status = json.loads((tmp_path / "run-status.json").read_text())
    assert status["phase"] == "failed"
    assert "hf_secret-value" not in (tmp_path / "run-status.json").read_text()


def test_controller_never_marks_complete_when_final_readback_fails(tmp_path: Path) -> None:
    def reject_finalization() -> dict[str, str]:
        raise ValueError("immutable final readback failed")

    controller = LifecycleController(
        run_training=lambda _plan, _resume, published: [published(step) for step in range(1_000, 15_001, 1_000)] and 15_000,
        publish_checkpoint=lambda _step: None,
        world_size=1,
        output=tmp_path,
        finalize=reject_finalization,
    )

    with pytest.raises(ValueError, match="readback"):
        controller.run()
    assert json.loads((tmp_path / "run-status.json").read_text())["phase"] == "failed"


def test_controller_requires_a_finalizer_before_completion(tmp_path: Path) -> None:
    controller = LifecycleController(
        run_training=lambda _plan, _resume, published: [published(step) for step in range(1_000, 15_001, 1_000)] and 15_000,
        publish_checkpoint=lambda _step: None,
        world_size=1,
        output=tmp_path,
    )

    with pytest.raises(ValueError, match="finalization"):
        controller.run()
    assert json.loads((tmp_path / "run-status.json").read_text())["phase"] == "failed"


def test_controller_retains_published_checkpoint_when_training_is_interrupted(tmp_path: Path) -> None:
    published: list[int] = []

    def interrupted(_plan: object, _resume: bool, on_stable_checkpoint: object) -> int:
        on_stable_checkpoint(1_000)
        raise TrainingFailure("interrupted", optimizer_step=1_001)

    controller = LifecycleController(
        run_training=interrupted,
        publish_checkpoint=published.append,
        world_size=1,
        output=tmp_path,
    )

    with pytest.raises(TrainingFailure, match="interrupted"):
        controller.run()
    assert published == [1_000]
    status = json.loads((tmp_path / "run-status.json").read_text())
    assert status["phase"] == "failed"
    assert status["published_steps"] == [1_000]
