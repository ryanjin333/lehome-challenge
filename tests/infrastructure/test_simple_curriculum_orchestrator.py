"""Offline contract tests for the one-VM simple-curriculum orchestrator."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

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
            "calibration-matrix": ("matrix",), "calibration-head": ("matrix", "manifest", "ledger"),
            "first-100-gate": ("report", "gate_receipt"), "calibration-tail": ("matrix", "manifest", "ledger"),
            "calibration-report": ("report",), "curriculum-matrix": ("matrix",),
            "curriculum-a": ("matrix", "manifest", "ledger"), "curriculum-b": ("matrix", "manifest", "ledger"),
            "fresh-report": ("report",), "replay-matrix": ("matrix",),
            "success-replay": ("matrix", "ledger"), "final-publication": ("publication_receipt", "publication_readback"),
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
    return module.CollectionConfig(
        campaign_root=tmp_path / "campaign",
        host_code_root=host,
        run_id="fresh-run-20260828",
        round_id="fresh-round-20260828",
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
        config.max_wall_seconds, config.max_spend_usd, config.paid, config.gpu_stop_command, bad_identity,
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
        config.max_wall_seconds, 100.0, config.paid, config.gpu_stop_command, config.runtime_identity,
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
    invalid = module.CollectionConfig(config.campaign_root, config.host_code_root, config.run_id, config.round_id, config.max_wall_seconds, config.max_spend_usd, config.paid, config.gpu_stop_command, identity)

    with pytest.raises(ValueError, match="excludes secrets"):
        invalid.validate()


def test_stage_output_rejects_secret_and_unexpected_fields(tmp_path: Path) -> None:
    module = _module(); config = _config(module, tmp_path)

    with pytest.raises(module.ReceiptMismatchError, match="schema"):
        module._authenticated_output("calibration-matrix", {"artifacts": {}, "secret": "x"}, config=config)


def test_preemption_resumes_the_same_root_and_does_not_replay_completed_stages(tmp_path: Path) -> None:
    module = _module(); config = _config(module, tmp_path); interrupted = FakeRunner(config.campaign_root, fail_stage="calibration-tail")
    with pytest.raises(RuntimeError, match="preempted"):
        module.run_collection(config, runner=interrupted)
    resumed = FakeRunner(config.campaign_root)

    assert module.run_collection(config, runner=resumed) == "complete"
    assert resumed.calls[0] == "calibration-tail"
    assert "calibration-head" not in resumed.calls


def test_resume_rehashes_the_physical_ledger_and_rejects_changed_bytes(tmp_path: Path) -> None:
    module = _module(); config = _config(module, tmp_path); assert module.run_collection(config, runner=FakeRunner(config.campaign_root)) == "complete"
    ledger = config.campaign_root / "artifacts" / "calibration-head-ledger.json"
    ledger.write_text('{"replaced":true}', encoding="utf-8")

    with pytest.raises(module.ReceiptMismatchError, match="artifact"):
        module.run_collection(config, runner=FakeRunner(config.campaign_root))


def test_campaign_root_symlink_is_rejected_before_journal_creation(tmp_path: Path) -> None:
    module = _module(); config = _config(module, tmp_path); real = tmp_path / "real"; real.mkdir(); alias = tmp_path / "campaign-alias"; alias.symlink_to(real, target_is_directory=True)
    invalid = module.CollectionConfig(alias, config.host_code_root, config.run_id, config.round_id, config.max_wall_seconds, config.max_spend_usd, config.paid, config.gpu_stop_command, config.runtime_identity)

    with pytest.raises(ValueError, match="symlink"):
        invalid.validate()
