"""Offline boundary tests for the bounded public N1.5 remote pipeline."""

from __future__ import annotations

import importlib.util
import json
import os
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
        "lifecycle-plan", "--run-id", "n15-20260831-a", "--repository", "ryanjin333/public-n15", "--remote-pipeline-root", "/mnt/lehome/public-n15-runs/n15-20260831-a", "--budget-usd", "100",
        "--estimated-cost-usd", "99.99", "--output", str(output),
    ]) == 0
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["vm_id"] == "computeinstance-u00t6xfqhadrcmssa2"
    assert value["protected_disk_id"] == "computedisk-u00pbe55crxy7jr56x"
    assert value["budget_usd"] == 100.0
    assert value["provider_source_image_id"] == "computeimage-u00zf6w3yf72gakhcy"
    assert value["prefixes"]["harvest"] == "n15-public/n15-20260831-a/harvest"
    assert value["stages"] == [
        "verify_stopped", "start", "validate_runtime", "train", "train_publish_readback",
        "focused_gate", "focused_gate_publish_readback", "harvest",
        "harvest_publish_readback", "stop",
    ]
    assert output.stat().st_mode & 0o777 == 0o444
    assert module.main([
        "lifecycle-plan", "--run-id", "n15-20260831-a", "--repository", "ryanjin333/public-n15", "--remote-pipeline-root", "/mnt/lehome/public-n15-runs/n15-20260831-a", "--budget-usd", "100",
        "--estimated-cost-usd", "99.99", "--output", str(output),
    ]) == 2


def test_lifecycle_plan_refuses_over_budget_before_any_provider_action(tmp_path: Path, capsys) -> None:
    module = _load_cli()
    output = tmp_path / "pipeline-plan.json"
    assert module.main([
        "lifecycle-plan", "--run-id", "n15-20260831-b", "--repository", "ryanjin333/public-n15", "--remote-pipeline-root", "/mnt/lehome/public-n15-runs/n15-20260831-b", "--budget-usd", "100",
        "--estimated-cost-usd", "100.01", "--output", str(output),
    ]) == 2
    assert not output.exists()
    assert "budget" in capsys.readouterr().err.lower()


def test_lifecycle_plan_resume_requires_exact_canonical_run_and_prefixes(tmp_path: Path) -> None:
    module = _load_cli()
    output = tmp_path / "pipeline-plan.json"
    arguments = ["--run-id", "n15-20260831-c", "--repository", "ryanjin333/public-n15", "--remote-pipeline-root", "/mnt/lehome/public-n15-runs/n15-20260831-c", "--budget-usd", "100", "--estimated-cost-usd", "3", "--output", str(output)]
    assert module.main(["lifecycle-plan", *arguments]) == 0
    assert module.main(["verify-lifecycle-plan", *arguments]) == 0
    altered = json.loads(output.read_text(encoding="utf-8")); altered["prefixes"]["harvest"] = "shared/latest"
    output.chmod(0o644); output.write_text(json.dumps(altered), encoding="utf-8")
    assert module.main(["verify-lifecycle-plan", *arguments]) == 2


def test_remote_wrapper_is_single_vm_fail_closed_and_receipt_resumable() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    assert 'EXACT_VM_ID="computeinstance-u00t6xfqhadrcmssa2"' in text
    assert 'PROTECTED_DISK_ID="computedisk-u00pbe55crxy7jr56x"' in text
    assert 'EXACT_IMAGE_ID="computeimage-u00zf6w3yf72gakhcy"' in text
    assert "LEHOME_N15_EXPECTED_IMAGE_ID" not in text
    assert "nebius compute instance start --id" in text
    assert "nebius compute instance stop --id" in text
    assert "compute instance create" not in text
    assert "compute disk create" not in text
    assert "compute image create" not in text
    assert "trap stop_exact_vm EXIT" in text
    assert "LEHOME_N15_MAX_BUDGET_USD" in text
    assert "LEHOME_N15_ESTIMATED_COST_USD" not in text
    assert "PROVIDER_HOURLY_CEILING_USD=3" in text
    assert "run_public_n15_reproduction.py lifecycle-plan" in text
    assert "run_public_n15_focused_gate.sh" in text
    assert "run_public_n15_harvest.sh" in text
    assert "immutable receipt" in text
    assert "anonymous" in text.lower()
    for required in (
        "LEHOME_OFFICIAL_RUNTIME_REVISION", "LEHOME_OFFICIAL_SOURCE_ROOT",
        "LEHOME_OFFICIAL_ASSETS_ROOT", "LEHOME_OFFICIAL_METADATA_ROOT",
        "LEHOME_N15_CANDIDATE_CHECKPOINT", "LEHOME_N15_CANDIDATE_IDENTITY_RECEIPT",
        "LEHOME_N15_FOCUSED_PROMOTION_RECEIPT", "LEHOME_N15_HARVEST_ROOT",
        "LEHOME_N15_TERMINAL_RECEIPT",
    ):
        assert required in text
    assert text.index("train_stage") < text.index("focused_stage") < text.index("harvest_stage")
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)


def test_remote_wrapper_never_runs_downstream_after_a_failed_gate() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    assert "verify_remote_training_chain" in text
    assert "verify_remote_focused_chain" in text
    assert "verify_remote_harvest_chain" in text
    assert "run_paid_stage focused_gate" in text
    assert "run_paid_stage harvest" in text
    assert "paid-deadline.json" in text
    assert text.index("verify_remote_focused_chain || fail") < text.rindex("run_paid_stage harvest")


def test_over_budget_plan_never_starts_the_mocked_exact_vm(tmp_path: Path) -> None:
    """A failing preflight may stop, but it must never make a start request."""
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    log = tmp_path / "nebius.log"
    raw = {
        "metadata": {"id": "computeinstance-u00t6xfqhadrcmssa2", "name": "lehome-rollout"},
        "status": {"state": "STOPPED"},
        "spec": {"boot_disk": {"managed_disk": {"spec": {"source_image_id": "computeimage-u00zf6w3yf72gakhcy"}}}, "secondary_disks": [{"existing_disk": {"id": "computedisk-u00pbe55crxy7jr56x"}}]},
    }
    (fake_bin / "nebius").write_text(
        "#!/usr/bin/env python3\nimport json, os, sys\nopen(os.environ['FAKE_NEBIUS_LOG'], 'a').write(' '.join(sys.argv[1:]) + '\\n')\nprint(json.dumps(" + repr(raw) + "))\n",
        encoding="utf-8",
    )
    (fake_bin / "ssh").write_text("#!/usr/bin/env bash\nexit 97\n", encoding="utf-8")
    for command in (fake_bin / "nebius", fake_bin / "ssh"): command.chmod(0o755)
    pipeline = tmp_path / "pipeline"; pipeline.mkdir()
    env = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}", "FAKE_NEBIUS_LOG": str(log),
           "LEHOME_N15_RUN_ID": "n15-over-budget", "LEHOME_N15_PIPELINE_ROOT": str(pipeline),
           "LEHOME_N15_MAX_BUDGET_USD": "71", "LEHOME_N15_SSH_TARGET": "operator@example", "LEHOME_N15_REMOTE_ROOT": "/mnt/lehome/runtime", "LEHOME_N15_REMOTE_RUNS_BASE": "/mnt/lehome/runs", "LEHOME_N15_REMOTE_PIPELINE_ROOT": "/mnt/lehome/runs/n15-over-budget", "LEHOME_N15_PUBLIC_HF_REPOSITORY": "ryanjin333/public-n15", "LEHOME_OFFICIAL_ASSETS_ROOT": "/mnt/assets", "LEHOME_OFFICIAL_METADATA_ROOT": "/mnt/source", "LEHOME_N15_REFERENCE_CHECKPOINT": "/mnt/reference", "LEHOME_N15_REFERENCE_SANITIZED_CONFIG_ROOT": "/mnt/reference-config", "LEHOME_N15_REFERENCE_COMPATIBILITY_RECEIPT": "/mnt/reference-receipt", "LEHOME_N15_NATIVE_RUNTIME_EVIDENCE_ROOT": "/mnt/evidence", "LEHOME_N15_NATIVE_DEPENDENCIES_ROOT": "/mnt/deps", "LEHOME_N15_FOCUSED_HF_CACHE_ROOT": "/mnt/cache", "LEHOME_N15_ROLLOUT_IMAGE_RECEIPT": "/mnt/image.json"}
    env.update({"LEHOME_N15_TRAINING_HF_CACHE_ROOT": "/mnt/train-cache", "LEHOME_N15_LEROBOT_WHEEL": "/mnt/lerobot.whl", "LEHOME_N15_TRAINING_ROOT": "/mnt/lehome/runs/n15-over-budget/training"})
    result = subprocess.run(["bash", str(WRAPPER)], cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert " start " not in f" {log.read_text(encoding='utf-8')} "
