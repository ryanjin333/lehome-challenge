"""Offline contract tests for the one-VM simple-curriculum orchestrator."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_simple_curriculum_collection.py"
WRAPPER = ROOT / "rollout_appliance" / "run_simple_curriculum_collection.sh"


def _module():
    spec = importlib.util.spec_from_file_location("simple_curriculum_orchestrator", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeRunner:
    def __init__(self, root: Path | None = None, *, gate_decision: str = "continue", replay_result: str = "complete", fail_stop: bool = False, fail_stage: str | None = None) -> None:
        self.root = root
        self.gate_decision = gate_decision
        self.replay_result = replay_result
        self.fail_stop = fail_stop
        self.fail_stage = fail_stage
        self.calls: list[str] = []
        self.stops = 0

    def run(self, stage: str, **_kwargs):
        self.calls.append(stage)
        if stage == self.fail_stage: raise RuntimeError("preempted")
        if self.root is None:
            return {"stage": stage}
        names = {
            "calibration-matrix": ("matrix", "matrix_receipt"), "calibration-head": ("matrix", "manifest", "ledger"),
            "first-100-gate": ("report", "gate_receipt"), "calibration-tail": ("matrix", "manifest", "ledger"),
            "calibration-report": ("report",), "curriculum-matrix": ("matrix", "matrix_receipt"),
            "curriculum-a": ("matrix", "manifest", "ledger"), "curriculum-b": ("matrix", "manifest", "ledger"),
            "fresh-report": ("report", "matrix", "terminal_artifact_manifest"), "replay-matrix": ("matrix", "matrix_receipt"),
            "success-replay": ("matrix", "ledger", "readback_seal"), "final-publication": ("publication_receipt", "publication_readback"),
        }[stage]
        artifacts = {}
        for name in names:
            path = self.root / "artifacts" / f"{stage}-{name}.json"; path.parent.mkdir(parents=True, exist_ok=True)
            if name == "matrix" and stage in {"calibration-matrix", "curriculum-matrix"}:
                count = 400 if stage == "calibration-matrix" else 600
                path.write_text(json.dumps([{"attempt_id": f"{stage}-{index}", "seed": index} for index in range(count)]), encoding="utf-8")
            else:
                path.write_text(json.dumps({"stage": stage, "name": name}), encoding="utf-8")
            artifacts[name] = {"path": path.relative_to(self.root).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        result = {"artifacts": artifacts}
        if stage == "first-100-gate": result["decision"] = self.gate_decision
        if stage == "success-replay": result["result"] = self.replay_result
        return result

    def stop_gpu(self, _command: str) -> None:
        self.stops += 1
        if self.fail_stop: raise RuntimeError("stop unavailable")


def _config(module, tmp_path: Path):
    host = tmp_path / "reviewed"; host.mkdir()
    for relative in ("source/lehome", "trainer/src", "scripts", "rollout_appliance"):
        (host / relative).mkdir(parents=True)
    observed = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    observer = tmp_path / "spend.json"; observer.write_text(json.dumps({"schema_version": 1, "kind": "lehome_spend_observation_v1", "observer": "test-meter", "observed_at_utc": observed, "spent_usd": 0.0}), encoding="utf-8")
    return module.CollectionConfig(
        campaign_root=tmp_path / "campaign",
        host_code_root=host,
        run_id="fresh-run-20260828",
        round_id="fresh-12k-20260828",
        max_wall_seconds=3600.0,
        max_spend_usd=99.0,
        paid=True,
        gpu_stop_command="/usr/local/libexec/lehome-stop-gpu",
        runtime_identity={
            "rollout_image": "repo/rollout@sha256:" + "a" * 64,
            "trainer_image": "repo/trainer@sha256:" + "b" * 64,
            "policy_repo": "ryanjin333/lehome-groot-n17-models",
            "policy_revision": "30ac1a84da67b099e115ad147bcd61e9d60046d3",
            "policy_step": 12000,
            "policy_artifact_sha256": "3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06",
            "simulator_device": "cpu", "cloth_device": "cpu", "policy_device": "cuda:0", "worker_count": 4,
        },
        spend_observer=observer,
    )


def test_gate_failure_never_launches_later_stages_and_stops_once(tmp_path: Path) -> None:
    module = _module(); config = _config(module, tmp_path); runner = FakeRunner(config.campaign_root, gate_decision="fidelity_stop")

    result = module.run_collection(config, runner=runner)

    assert result == "fidelity_stop"
    assert runner.calls == ["calibration-matrix", "calibration-head", "first-100-gate", "final-publication"]
    assert runner.stops == 1


def test_continue_uses_the_exact_order_and_reports_replay_shortage(tmp_path: Path) -> None:
    module = _module(); config = _config(module, tmp_path); runner = FakeRunner(config.campaign_root, replay_result="replay_shortage")

    result = module.run_collection(config, runner=runner)

    assert result == "replay_shortage"
    assert runner.calls == [
        "calibration-matrix", "calibration-head", "first-100-gate", "calibration-tail",
        "calibration-report", "curriculum-matrix", "curriculum-a", "curriculum-b",
        "fresh-report", "replay-matrix", "success-replay", "final-publication",
    ]
    assert runner.stops == 1


def test_restart_validates_receipts_without_repeating_terminal_stages(tmp_path: Path) -> None:
    module = _module(); config = _config(module, tmp_path)
    first = FakeRunner(config.campaign_root); assert module.run_collection(config, runner=first) == "complete"
    second = FakeRunner(config.campaign_root)

    assert module.run_collection(config, runner=second) == "complete"
    assert second.calls == []
    assert second.stops == 0


def test_receipt_collision_is_fatal(tmp_path: Path) -> None:
    module = _module(); config = _config(module, tmp_path)
    root = config.campaign_root / "stage-receipts"; root.mkdir(parents=True)
    (root / "calibration-matrix.json").write_text('{"bad": true}\n', encoding="utf-8")

    try:
        module.run_collection(config, runner=FakeRunner())
    except module.ReceiptMismatchError:
        pass
    else:
        raise AssertionError("receipt collision must be fatal")


def test_partition_preserves_logical_row_bytes_and_keeps_metadata_in_manifest(tmp_path: Path) -> None:
    module = _module()
    rows = [{"attempt_id": "a", "seed": 1}, {"attempt_id": "b", "seed": 2}]

    partition, manifest = module.partition_rows(rows, parent_matrix_sha256="c" * 64, partition_id="calibration-head", start=0, end=1)

    assert partition == [{"attempt_id": "a", "seed": 1}]
    assert manifest["row_start"] == 0 and manifest["row_end"] == 1
    assert manifest["partition_id"] == "calibration-head"
    assert rows[0] == {"attempt_id": "a", "seed": 1}


def test_wrapper_requires_paid_stop_hook_and_has_no_cloud_lifecycle_command(tmp_path: Path) -> None:
    result = subprocess.run(["bash", str(WRAPPER)], cwd=ROOT, text=True, capture_output=True, env={**os.environ, "LEHOME_PAID_COLLECTION": "1"})

    assert result.returncode != 0
    assert "LEHOME_GPU_STOP_COMMAND" in result.stderr
    text = WRAPPER.read_text(encoding="utf-8").lower()
    assert all(token not in text for token in ("nebius", "terraform", "packer", " instance create", " instance start", " instance delete"))


def test_simple_partitions_publish_mixed_policy_outcomes_through_terminal_evidence_lane() -> None:
    campaign = (ROOT / "rollout_appliance" / "run_12k_campaign.sh").read_text(encoding="utf-8")

    assert '[ "${EVALUATION_TERMINAL_UPLOAD}" = "1" ] || [ "${SIMPLE_CURRICULUM_COLLECTION}" = "1" ]' in campaign
    assert 'UPLOADER_ROLE="evaluation-uploader"' in campaign
    assert 'UPLOADER_ROOT_FLAG=(--terminal-root "${CAMPAIGN_ROOT}/evaluation-terminal")' in campaign


def test_configuration_rejects_unpinned_or_noncanonical_runtime_tuple(tmp_path: Path) -> None:
    module = _module(); config = _config(module, tmp_path)
    bad_identity = dict(config.runtime_identity); bad_identity["worker_count"] = 1
    invalid = module.CollectionConfig(
        config.campaign_root, config.host_code_root, config.run_id, config.round_id,
        config.max_wall_seconds, config.max_spend_usd, config.paid, config.gpu_stop_command, bad_identity, config.spend_observer,
    )

    try:
        invalid.validate()
    except ValueError as error:
        assert "runtime tuple" in str(error)
    else:
        raise AssertionError("noncanonical runtime tuple must fail closed")


def test_configuration_rejects_the_hard_cap_itself(tmp_path: Path) -> None:
    module = _module(); config = _config(module, tmp_path)
    capped = module.CollectionConfig(
        config.campaign_root, config.host_code_root, config.run_id, config.round_id,
        config.max_wall_seconds, 100.0, config.paid, config.gpu_stop_command, config.runtime_identity, config.spend_observer,
    )

    with pytest.raises(ValueError, match="max spend"):
        capped.validate()


def test_missing_or_mutated_stage_artifact_fails_resume_closed(tmp_path: Path) -> None:
    module = _module(); config = _config(module, tmp_path)
    runner = FakeRunner()
    with pytest.raises(module.ReceiptMismatchError, match="artifact"):
        module.run_collection(config, runner=runner)


def test_command_runner_rejects_env_command_and_shell_bypass(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    monkeypatch.setenv("LEHOME_ORCHESTRATOR_CALIBRATION_MATRIX_COMMAND", "/bin/sh -c 'echo {}'")

    with pytest.raises(ValueError, match="fixed"):
        module.CommandRunner().run("calibration-matrix")


def test_fixed_adapter_has_a_canonical_non_shell_argv_for_every_paid_stage(tmp_path: Path) -> None:
    module = _module(); runner = module.CommandRunner(_config(module, tmp_path))
    root = runner.config.campaign_root
    (root / "partitions").mkdir(parents=True)
    for partition in ("calibration-head", "calibration-tail", "curriculum-a", "curriculum-b"):
        (root / "partitions" / f"{partition}.json").write_text("[]", encoding="utf-8")
    for stage in module.STAGES[:-1]:
        if stage == "final-publication":
            with pytest.raises(RuntimeError, match="Task 7 publisher"):
                runner.argv_for(stage, {})
            continue
        argv = runner.argv_for(stage, {})
        assert argv and all(token not in {"sh", "bash", "-c"} for token in argv)


def test_resume_rehashes_stage_artifacts_and_refuses_mutated_matrix(tmp_path: Path) -> None:
    module = _module(); config = _config(module, tmp_path); first = FakeRunner(config.campaign_root)
    assert module.run_collection(config, runner=first) == "complete"
    matrix = config.campaign_root / "artifacts" / "calibration-matrix-matrix.json"
    matrix.write_text("[]", encoding="utf-8")

    with pytest.raises(module.ReceiptMismatchError, match="artifact"):
        module.run_collection(config, runner=FakeRunner(config.campaign_root))


def test_failed_stop_is_durable_and_restart_returns_infrastructure_stop_failure(tmp_path: Path) -> None:
    module = _module(); config = _config(module, tmp_path); runner = FakeRunner(config.campaign_root, fail_stop=True)
    assert module.run_collection(config, runner=runner) == "infrastructure_stop_failure"
    assert runner.stops == 1

    assert module.run_collection(config, runner=runner) == "infrastructure_stop_failure"
    assert runner.stops == 1


def test_runtime_identity_rejects_extra_or_secret_like_keys(tmp_path: Path) -> None:
    module = _module(); config = _config(module, tmp_path); identity = dict(config.runtime_identity); identity["api_token"] = "never-store"
    invalid = module.CollectionConfig(config.campaign_root, config.host_code_root, config.run_id, config.round_id, config.max_wall_seconds, config.max_spend_usd, config.paid, config.gpu_stop_command, identity, config.spend_observer)

    with pytest.raises(ValueError, match="excludes secrets"):
        invalid.validate()


def test_stage_output_rejects_secret_and_unexpected_fields(tmp_path: Path) -> None:
    module = _module(); config = _config(module, tmp_path)

    with pytest.raises(module.ReceiptMismatchError, match="schema"):
        module._authenticated_output("calibration-matrix", {"artifacts": {}, "secret": "x"}, config=config)


def test_preemption_resumes_the_same_root_and_does_not_replay_completed_stages(tmp_path: Path) -> None:
    module = _module(); config = _config(module, tmp_path); interrupted = FakeRunner(config.campaign_root, fail_stage="calibration-tail")
    assert module.run_collection(config, runner=interrupted) == "infrastructure_stop_failure"
    assert interrupted.stops == 1
    resumed = FakeRunner(config.campaign_root)

    assert module.run_collection(config, runner=resumed) == "infrastructure_stop_failure"
    assert resumed.calls == []


def test_resume_rehashes_the_physical_ledger_and_rejects_changed_bytes(tmp_path: Path) -> None:
    module = _module(); config = _config(module, tmp_path); assert module.run_collection(config, runner=FakeRunner(config.campaign_root)) == "complete"
    ledger = config.campaign_root / "artifacts" / "calibration-head-ledger.json"
    ledger.write_text('{"replaced":true}', encoding="utf-8")

    with pytest.raises(module.ReceiptMismatchError, match="artifact"):
        module.run_collection(config, runner=FakeRunner(config.campaign_root))


def test_campaign_root_symlink_is_rejected_before_journal_creation(tmp_path: Path) -> None:
    module = _module(); config = _config(module, tmp_path); real = tmp_path / "real"; real.mkdir(); alias = tmp_path / "campaign-alias"; alias.symlink_to(real, target_is_directory=True)
    invalid = module.CollectionConfig(alias, config.host_code_root, config.run_id, config.round_id, config.max_wall_seconds, config.max_spend_usd, config.paid, config.gpu_stop_command, config.runtime_identity, config.spend_observer)

    with pytest.raises(ValueError, match="symlink"):
        invalid.validate()


def test_paid_budget_gate_fails_before_runner_at_exact_spend_boundary(tmp_path: Path) -> None:
    module = _module(); config = _config(module, tmp_path)
    assert config.spend_observer is not None
    observed = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    config.spend_observer.write_text(json.dumps({"schema_version": 1, "kind": "lehome_spend_observation_v1", "observer": "test-meter", "observed_at_utc": observed, "spent_usd": 99.0}), encoding="utf-8")
    runner = FakeRunner(config.campaign_root)

    with pytest.raises(module.ReceiptMismatchError, match="budget"):
        module.run_collection(config, runner=runner)
    assert runner.calls == []


def test_inflight_budget_watchdog_terminates_a_clean_child_before_returning(tmp_path: Path) -> None:
    """A live paid adapter cannot outlive a newly observed budget breach."""
    module = _module(); runner = module.CommandRunner(_config(module, tmp_path))
    observations = 0

    def observe_then_breach() -> None:
        nonlocal observations
        observations += 1
        if observations >= 2:
            raise module.BudgetLimitError("paid budget or wall-time limit reached")

    runner.budget_check = observe_then_breach
    started = time.monotonic()
    with pytest.raises(module.BudgetLimitError, match="budget"):
        runner._invoke(
            (sys.executable, "-c", "import time; time.sleep(30)"),
            stage="watchdog", inputs={},
        )

    assert observations >= 2
    assert time.monotonic() - started < 3


def test_paid_simple_wrapper_contract_requires_one_vm_marker() -> None:
    text = (ROOT / "rollout_appliance" / "run_12k_campaign.sh").read_text(encoding="utf-8")
    assert 'paid simple curriculum requires LEHOME_ONE_VM_ORCHESTRATOR=1' in text


def test_fixed_adapter_executes_real_matrix_cli_with_clean_typed_handoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The production adapter must create a real canonical output, not a marker."""
    module = _module()
    config = _config(module, tmp_path)
    config = module.CollectionConfig(
        campaign_root=config.campaign_root,
        host_code_root=ROOT,
        run_id=config.run_id,
        round_id=config.round_id,
        max_wall_seconds=config.max_wall_seconds,
        max_spend_usd=config.max_spend_usd,
        paid=config.paid,
        gpu_stop_command=config.gpu_stop_command,
        runtime_identity=config.runtime_identity,
        spend_observer=config.spend_observer,
    )
    catalog = config.campaign_root / "inputs" / "seen-catalog.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_text(json.dumps({
        "top_long": [f"Top_Long_Seen_{index}" for index in range(10)],
        "top_short": [f"Top_Short_Seen_{index}" for index in range(10)],
        "pant_long": [f"Pant_Long_Seen_{index}" for index in range(10)],
        "pant_short": [f"Pant_Short_Seen_{index}" for index in range(10)],
    }), encoding="utf-8")
    monkeypatch.setenv("LEHOME_ORCHESTRATOR_CALIBRATION_MATRIX_COMMAND", "/bin/false")
    runner = module.CommandRunner(config)
    clean = runner.environment_for("calibration-matrix", {})
    assert clean["PYTHONPATH"] == str(ROOT / "source" / "lehome")
    assert clean["LEHOME_SIMPLE_CURRICULUM_COLLECTION"] == "1"
    assert clean["LEHOME_ONE_VM_ORCHESTRATOR"] == "1"
    assert clean["LEHOME_PAID_COLLECTION"] == "1"
    assert "LEHOME_ORCHESTRATOR_CALIBRATION_MATRIX_COMMAND" not in clean

    result = runner.run("calibration-matrix")

    matrix = config.campaign_root / result["artifacts"]["matrix"]["path"]
    receipt = config.campaign_root / result["artifacts"]["matrix_receipt"]["path"]
    assert len(json.loads(matrix.read_text(encoding="utf-8"))) == 400
    assert json.loads(receipt.read_text(encoding="utf-8"))["output_sha256"] == result["artifacts"]["matrix"]["sha256"]

    def must_not_reinvoke(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a completed matrix must be adopted without rerunning its producer")

    monkeypatch.setattr(runner, "_invoke", must_not_reinvoke)
    assert runner.run("calibration-matrix") == result
    receipt.unlink()
    with pytest.raises(module.ReceiptMismatchError, match="receipt"):
        runner.run("calibration-matrix")


def test_existing_stop_state_without_immutable_stop_receipt_never_retries_stop(tmp_path: Path) -> None:
    module = _module(); config = _config(module, tmp_path); runner = FakeRunner(config.campaign_root)
    assert module.run_collection(config, runner=runner) == "complete"
    stop_receipt = config.campaign_root / "stage-receipts" / "gpu-stop.json"
    stop_receipt.unlink()

    assert module.run_collection(config, runner=runner) == "infrastructure_stop_failure"
    assert runner.stops == 1


def test_partition_adapter_passes_exact_one_vm_tuple_without_inherited_lehome_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module(); config = _config(module, tmp_path)
    rows = [{"attempt_id": f"calibration-{index:04d}", "trial_id": f"trial-{index:04d}", "seed": index} for index in range(400)]
    logical = config.campaign_root / "matrices" / "calibration.json"; logical.parent.mkdir(parents=True)
    logical.write_bytes((json.dumps(rows, sort_keys=True, separators=(",", ":")) + "\n").encode())
    matrix, manifest, details = module.materialize_partition(
        parent_matrix=logical, parent_matrix_sha256=hashlib.sha256(logical.read_bytes()).hexdigest(),
        output_directory=config.campaign_root / "partitions", partition_id="calibration-head", start=0, end=100,
    )
    monkeypatch.setenv("LEHOME_WORKER_COUNT", "1")
    runner = module.CommandRunner(config)
    inputs = {
        "partition_id": "calibration-head", "partition_matrix": matrix.relative_to(config.campaign_root).as_posix(),
        "partition_manifest": manifest.relative_to(config.campaign_root).as_posix(), "partition_sha256": details["partition_sha256"],
        "row_start": 0, "row_end": 100, "target": 100, "lease_budget": 150,
    }

    environment = runner.environment_for("calibration-head", inputs)

    assert runner.argv_for("calibration-head", inputs) == (str(config.host_code_root / "rollout_appliance" / "run_12k_campaign.sh"),)
    assert environment["LEHOME_ATTEMPT_MATRIX"] == str(matrix)
    assert environment["LEHOME_PARTITION_MANIFEST"] == str(manifest)
    assert environment["LEHOME_WORKER_COUNT"] == "4"
    assert environment["LEHOME_SIMPLE_CURRICULUM_COLLECTION"] == "1"
    assert environment["LEHOME_ONE_VM_ORCHESTRATOR"] == "1"
    assert environment["LEHOME_PAID_COLLECTION"] == "1"
    assert environment["LEHOME_RESUME_PREEMPTED_ROLLOUT"] == "0"
    assert environment["LEHOME_ROLLOUT_PREEMPTION_CONTEXT"] == str(config.campaign_root / "fresh" / "calibration-head" / "rollout-preemption.json")
    assert set(key for key in environment if key.startswith("LEHOME_")) >= {
        "LEHOME_POLICY_REPO", "LEHOME_POLICY_REVISION", "LEHOME_POLICY_ARTIFACT_SHA256",
        "LEHOME_ROLLOUT_IMAGE", "LEHOME_TRAINER_IMAGE", "LEHOME_RUN_ID", "LEHOME_ROUND_ID",
    }


def test_partition_adapter_exports_durable_preemption_contract(tmp_path: Path) -> None:
    module = _module(); config = _config(module, tmp_path)
    root = config.campaign_root
    matrix = root / "partitions/calibration-head.json"; matrix.parent.mkdir(parents=True)
    matrix.write_text("[]", encoding="utf-8")
    manifest = root / "partitions/calibration-head.manifest.json"
    manifest.write_text(json.dumps({"parent_matrix_sha256": "a" * 64}), encoding="utf-8")
    stage = root / "fresh/calibration-head"; stage.mkdir(parents=True)
    (stage / "ledger.sqlite3").write_bytes(b"sqlite")
    context = stage / "rollout-preemption.json"; context.write_text("{}", encoding="utf-8")
    inputs = {"partition_id": "calibration-head", "partition_matrix": "partitions/calibration-head.json", "partition_manifest": "partitions/calibration-head.manifest.json", "partition_sha256": hashlib.sha256(matrix.read_bytes()).hexdigest(), "row_start": 0, "row_end": 100, "target": 100, "lease_budget": 150}

    environment = module.CommandRunner(config).environment_for("calibration-head", inputs)

    assert environment["LEHOME_RESUME_PREEMPTED_ROLLOUT"] == "1"
    assert environment["LEHOME_ROLLOUT_PREEMPTION_CONTEXT"] == str(context)


def test_paid_spend_observer_rejects_stale_or_different_meter_receipts(tmp_path: Path) -> None:
    module = _module(); config = _config(module, tmp_path); journal = module.StageJournal(config)
    journal.check_budget()
    assert config.spend_observer is not None
    observed = (datetime.now(UTC) - timedelta(minutes=6)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    config.spend_observer.write_text(json.dumps({"schema_version": 1, "kind": "lehome_spend_observation_v1", "observer": "test-meter", "observed_at_utc": observed, "spent_usd": 0.0}), encoding="utf-8")

    with pytest.raises(module.ReceiptMismatchError, match="stale"):
        journal.check_budget()

    observed = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    config.spend_observer.write_text(json.dumps({"schema_version": 1, "kind": "lehome_spend_observation_v1", "observer": "other-meter", "observed_at_utc": observed, "spent_usd": 0.0}), encoding="utf-8")

    with pytest.raises(module.ReceiptMismatchError, match="observer"):
        journal.check_budget()


def test_exhausted_visual_replay_with_a_category_shortage_is_a_data_outcome(tmp_path: Path) -> None:
    """All 400 rejected parents are a replay shortage, never a fake success."""
    from lehome.flywheel.task_ledger import TaskLedger

    module = _module(); config = _config(module, tmp_path)
    rows = [
        {
            "attempt_id": f"replay-{category}-{index}", "trial_id": f"replay-{category}-{index}",
            "category": category, "strategy": "visual_only", "category_acceptance_cap": 50,
        }
        for category in ("top_long", "top_short", "pant_long", "pant_short")
        for index in range(100)
    ]
    replay = config.campaign_root / "replay" / "replay.json"; replay.parent.mkdir(parents=True)
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")) + "\n"
    replay.write_text(encoded, encoding="utf-8")
    (replay.with_suffix(".json.sha256")).write_text(hashlib.sha256(encoded.encode()).hexdigest() + "\n", encoding="ascii")
    ledger = TaskLedger(config.campaign_root / "replay" / "ledger.sqlite3", attempt_matrix=rows, max_attempts=400, target_accepted=200)
    try:
        for index in range(400):
            lease = ledger.lease_next(f"worker-{index % 4}", lease_duration_ns=10**15)
            assert lease is not None
            ledger.record_terminal(lease.worker_id, lease.attempt.attempt_id, lease.lease_id, f"raw-{index}")
            assert ledger.validate_terminal(lease.attempt.attempt_id, "rejected") == "rejected"
    finally:
        ledger.close()

    discovered = module.CommandRunner(config)._discover("success-replay", {})

    assert discovered["result"] == "replay_shortage"


def test_exact_visual_replay_seals_only_the_200_readback_verified_category_capped_successes(tmp_path: Path) -> None:
    from lehome.flywheel.fresh_replay_evidence import _episode_artifact_sha256
    from lehome.flywheel.task_ledger import TaskLedger

    module = _module(); config = _config(module, tmp_path)
    categories = ("top_long", "top_short", "pant_long", "pant_short")
    rows = [
        {
            "attempt_id": f"replay-{category}-{index}", "trial_id": f"replay-{category}-{index}",
            "category": category, "strategy": "visual_only", "category_acceptance_cap": 50,
        }
        for index in range(100) for category in categories
    ]
    replay = config.campaign_root / "replay" / "replay.json"; replay.parent.mkdir(parents=True)
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")) + "\n"
    replay.write_text(encoded, encoding="utf-8")
    (replay.with_suffix(".json.sha256")).write_text(hashlib.sha256(encoded.encode()).hexdigest() + "\n", encoding="ascii")
    ledger = TaskLedger(config.campaign_root / "replay" / "ledger.sqlite3", attempt_matrix=rows, max_attempts=400, target_accepted=200)
    try:
        for index in range(200):
            lease = ledger.lease_next(f"worker-{index % 4}", lease_duration_ns=10**15)
            assert lease is not None
            artifact = config.campaign_root / "replay" / "accepted" / lease.attempt.attempt_id
            artifact.mkdir(parents=True); (artifact / "rollout.json").write_text("{}\n", encoding="utf-8")
            digest = _episode_artifact_sha256(artifact)
            receipt = config.campaign_root / "replay" / "hf-sync-receipts" / f"{lease.attempt.attempt_id}.sync.json"
            receipt.parent.mkdir(exist_ok=True)
            receipt.write_bytes((json.dumps({
                "schema_version": 1, "attempt_id": lease.attempt.attempt_id,
                "repository": "ryanjin333/lehome-groot-n17-rollouts",
                "round_id": config.round_id + "-replay",
                "remote_prefix": f"rollout-rounds/{config.round_id}-replay/{lease.attempt.attempt_id}",
                "readback_verified": True, "episode_sha256": digest, "immutable_revision": "c" * 40,
            }, sort_keys=True, separators=(",", ":")) + "\n").encode())
            ledger.record_terminal(lease.worker_id, lease.attempt.attempt_id, lease.lease_id, str(artifact))
            assert ledger.validate_terminal(lease.attempt.attempt_id, "accepted", artifact_id=str(artifact)) == "accepted"
    finally:
        ledger.close()

    discovered = module.CommandRunner(config)._discover("success-replay", {})

    assert discovered["result"] == "complete"
    seal = config.campaign_root / discovered["artifacts"]["readback_seal"]["path"]
    payload = json.loads(seal.read_text(encoding="utf-8"))
    assert payload["readback_verified"] is True
    assert payload["accepted_by_category"] == {category: 50 for category in categories}
    assert len(payload["accepted_attempt_ids"]) == 200
