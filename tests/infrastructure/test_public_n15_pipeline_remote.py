"""Offline boundary tests for the bounded public N1.5 remote pipeline."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts/run_public_n15_reproduction.py"
WRAPPER = ROOT / "rollout_appliance/run_public_n15_pipeline_remote.sh"


def _load_cli():
    spec = importlib.util.spec_from_file_location("public_n15_reproduction", CLI)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lifecycle_plan_is_immutable_and_has_exact_paid_stage_order(tmp_path: Path) -> None:
    module = _load_cli()
    output = tmp_path / "pipeline-plan.json"
    assert module.main([
        "lifecycle-plan", "--run-id", "n15-20260831-a", "--budget-usd", "100",
        "--estimated-cost-usd", "99.99", "--output", str(output),
    ]) == 0
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["vm_id"] == "computeinstance-u00t6xfqhadrcmssa2"
    assert value["protected_disk_id"] == "computedisk-u00pbe55crxy7jr56x"
    assert value["budget_usd"] == 100.0
    assert value["stages"] == [
        "verify_stopped", "start", "validate_runtime", "train", "train_publish_readback",
        "focused_gate", "focused_gate_publish_readback", "harvest",
        "harvest_publish_readback", "stop",
    ]
    assert output.stat().st_mode & 0o777 == 0o444
    assert module.main([
        "lifecycle-plan", "--run-id", "n15-20260831-a", "--budget-usd", "100",
        "--estimated-cost-usd", "99.99", "--output", str(output),
    ]) == 2


def test_lifecycle_plan_refuses_over_budget_before_any_provider_action(tmp_path: Path, capsys) -> None:
    module = _load_cli()
    output = tmp_path / "pipeline-plan.json"
    assert module.main([
        "lifecycle-plan", "--run-id", "n15-20260831-b", "--budget-usd", "100",
        "--estimated-cost-usd", "100.01", "--output", str(output),
    ]) == 2
    assert not output.exists()
    assert "budget" in capsys.readouterr().err.lower()


def test_remote_wrapper_is_single_vm_fail_closed_and_receipt_resumable() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    assert 'EXACT_VM_ID="computeinstance-u00t6xfqhadrcmssa2"' in text
    assert 'PROTECTED_DISK_ID="computedisk-u00pbe55crxy7jr56x"' in text
    assert "nebius compute instance start --id" in text
    assert "nebius compute instance stop --id" in text
    assert "compute instance create" not in text
    assert "compute disk create" not in text
    assert "compute image create" not in text
    assert "trap stop_exact_vm EXIT" in text
    assert "LEHOME_N15_MAX_BUDGET_USD" in text
    assert "LEHOME_N15_ESTIMATED_COST_USD" in text
    assert "run_public_n15_reproduction.py lifecycle-plan" in text
    assert "run_public_n15_focused_gate.sh" in text
    assert "run_public_n15_harvest.sh" in text
    assert "immutable receipt" in text
    assert "anonymous" in text.lower()
    assert text.index("train_stage") < text.index("focused_stage") < text.index("harvest_stage")
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)


def test_remote_wrapper_never_runs_downstream_after_a_failed_gate() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    assert "require_stage_receipt train" in text
    assert "require_stage_receipt focused_gate" in text
    assert "require_stage_receipt harvest" in text
    assert "focused_stage || fail" in text
    assert "harvest_stage || fail" in text
    assert text.index("require_stage_receipt focused_gate") < text.rindex("harvest_stage || fail")
