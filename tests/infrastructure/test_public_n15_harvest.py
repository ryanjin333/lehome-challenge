"""Offline executable-boundary checks for the public N1.5 harvest."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts/build_public_n15_harvest.py"
WRAPPER = ROOT / "rollout_appliance/run_public_n15_harvest.sh"


def _load_cli():
    spec = importlib.util.spec_from_file_location("build_public_n15_harvest", CLI)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_build_verify_and_first_100_are_offline_and_atomic(tmp_path: Path) -> None:
    module = _load_cli()
    manifest = tmp_path / "manifest.json"
    receipt = tmp_path / "manifest.receipt.json"
    assert module.main([
        "build", "--checkpoint-tree-sha256", "a" * 64,
        "--checkpoint-receipt-sha256", "b" * 64,
        "--runtime-receipt-sha256", "c" * 64,
        "--source-tree-sha256", "d" * 64,
        "--dataset-snapshot-sha256", "e" * 64,
        "--rollout-image-sha256", "f" * 64,
        "--manifest", str(manifest), "--receipt", str(receipt),
    ]) == 0
    assert manifest.is_file() and receipt.is_file()
    assert module.main(["verify", "--manifest", str(manifest), "--receipt", str(receipt)]) == 0

    value = json.loads(manifest.read_text())
    outcomes = tmp_path / "outcomes.json"
    outcomes.write_bytes((json.dumps([
        {
            "attempt_id": row["attempt_id"],
            "official_outcome": "success" if index < 5 else "policy_failure",
            "cloth_fidelity": {"measured": True, "valid": True},
        }
        for index, row in enumerate(value["attempts"][:100])
    ], sort_keys=True, separators=(",", ":")) + "\n").encode("ascii"))
    gate = tmp_path / "first-100.json"
    assert module.main([
        "first-100", "--manifest", str(manifest), "--outcomes", str(outcomes),
        "--output", str(gate),
    ]) == 0
    assert json.loads(gate.read_text())["decision"] == "continue"
    assert module.main([
        "build", "--checkpoint-tree-sha256", "a" * 64,
        "--checkpoint-receipt-sha256", "b" * 64,
        "--runtime-receipt-sha256", "c" * 64,
        "--source-tree-sha256", "d" * 64,
        "--dataset-snapshot-sha256", "e" * 64,
        "--rollout-image-sha256", "f" * 64,
        "--manifest", str(manifest), "--receipt", str(receipt),
    ]) == 2


def test_shell_wrapper_is_native_fail_closed_and_has_no_provisioning_path() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    assert "python -P -m scripts.eval" in text
    assert "--save_datasets" in text
    assert "--garment_filter" in text
    assert "--num_episodes 25" in text
    assert "--device cpu" in text
    assert "--use_random_seed" not in text
    assert "GARMENT_FILTER" not in text
    assert "target_successes" not in text
    assert "hard_state" not in text
    assert "curriculum" not in text.lower()
    assert "computeinstance-u00t6xfqhadrcmssa2" in text
    assert "computedisk-u00pbe55crxy7jr56x" in text
    assert "nebius compute instance stop --id" in text
    assert "compute instance create" not in text
    assert "compute disk create" not in text
    assert "compute instance start" not in text
    assert text.index("verify-terminal") < text.index("HARVEST_TERMINAL_COMPLETE=1")


def test_shell_wrapper_has_four_to_two_admission_and_first_100_boundary() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    assert "LEHOME_N15_FOUR_WORKER_ADMISSION_RECEIPT" in text
    assert "LEHOME_N15_TWO_WORKER_ADMISSION_RECEIPT" in text
    assert "admit-workers" in text
    assert "first-100" in text
    assert "LEHOME_N15_FIRST_100_OUTCOMES" in text
    assert "LEHOME_N15_PUBLICATION_RECEIPT" in text
    assert "LEHOME_N15_PROVIDER_STOPPED_RECEIPT" in text
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)


def test_shell_aggregates_every_pid_and_requires_checked_in_full_collector_before_publish() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    assert "process_pids" in text and "process_ids" in text
    assert 'if wait "$pid"' in text
    assert "collect-outcomes" in text
    assert "--expected-attempt-count 1000" in text
    assert "final-outcomes.json" in text
    assert text.index("collect-outcomes") < text.index("publish-hf")
    assert "LEHOME_N15_OUTCOME_COLLECTOR" not in text
    assert "LEHOME_N15_PUBLIC_HF_PUBLISHER" not in text
    # A bare wait loses individual exit statuses and is forbidden.
    assert "then wait; fi" not in text


def test_shell_pins_task1_runtime_image_and_sanitizes_semantic_environment() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    assert "LEHOME_N15_TRAINING_IDENTITY_RECEIPT" in text
    assert "verify-runtime" in text
    assert "sha256:bec2b688ca03145dd20c010aa32b761a386e3fed57bdc45c3df5d86f9afa15c7" in text
    assert "docker image inspect" in text
    assert "--pull never" in text
    assert "--network none" in text  # runtime verification cannot fetch or mutate remote state
    assert "PYTHONHOME=" in text
    assert "PYTHONSAFEPATH=1" in text
    assert "PYTHONDONTWRITEBYTECODE=1" in text
    assert "LEHOME_NATIVE_CLOTH_FIDELITY_EVIDENCE" in text
    for forbidden in (
        "LEHOME_NATIVE_REFERENCE_CHECKPOINT_ROOT",
        "LEHOME_NATIVE_REFERENCE_SANITIZED_CONFIG_ROOT",
        "LEHOME_NATIVE_REFERENCE_CHECKPOINT_COMPATIBILITY_RECEIPT",
        "LEHOME_CPU_ACTION",
    ):
        assert forbidden not in text


def test_shell_validates_public_readback_and_provider_stop_on_every_exit() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    assert "publish-hf" in text
    assert "anonymous" in text.lower()
    assert "validate-provider-stop" in text
    assert "stop_and_observe_exact_vm" in text
    assert "trap on_exit EXIT" in text
    assert "protected_disk_preserved" in text
