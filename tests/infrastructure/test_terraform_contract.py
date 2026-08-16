"""Static contract for the protected Nebius storage and runtime Terraform."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TF_ROOT = REPO_ROOT / "infrastructure" / "nebius" / "terraform"


def _read(*parts: str) -> str:
    return (TF_ROOT.joinpath(*parts)).read_text(encoding="utf-8")


def test_provider_is_pinned_exact():
    for root in ("storage", "runtime"):
        main = _read(root, "main.tf")
        assert 'source  = "nebius/nebius"' in main
        assert 'version = "0.6.42"' in main


def test_storage_own_protected_500gib_network_ssd():
    storage = _read("storage", "main.tf")
    assert "500" in storage
    assert '"NETWORK_SSD"' in storage
    assert "forbid_deletion = true" in storage
    assert "prevent_destroy = true" in storage


def test_runtime_is_preemptible_rtx6000_with_fail_recovery():
    runtime = _read("runtime", "main.tf")
    assert '"gpu-rtx6000"' in runtime
    assert '"1gpu-24vcpu-218gb"' in runtime
    assert 'recovery_policy = "FAIL"' in runtime
    assert "on_preemption" in runtime and '"STOP"' in runtime


def test_runtime_boot_disk_is_disposable_custom_image():
    runtime = _read("runtime", "main.tf")
    assert "image_id" in runtime
    assert "boot_disk" in runtime


def test_one_existing_secondary_disk_attached_read_write():
    runtime = _read("runtime", "main.tf")
    assert '"READ_WRITE"' in runtime
    assert "device_id" in runtime


def test_single_role_at_a_time_variable_selects_role():
    runtime_vars = _read("runtime", "variables.tf")
    assert "active_role" in runtime_vars
    assert "training" in runtime_vars and "rollout" in runtime_vars


def test_no_secrets_or_hyperparameters_in_state():
    for root in ("storage", "runtime"):
        for name in ("main.tf", "variables.tf", "outputs.tf"):
            path = TF_ROOT / root / name
            if not path.exists():
                continue
            content = path.read_text(encoding="utf-8").lower()
            for pattern in ("token", "password", "api_key", "private_key"):
                assert pattern not in content, (root, name, pattern)
    # Hyperparameters live in manifests, never as loose Terraform variables.
    runtime_vars = _read("runtime", "variables.tf").lower()
    assert "learning_rate" not in runtime_vars
    assert "batch_size" not in runtime_vars


def test_example_tfvars_document_both_roles():
    training_example = _read("runtime", "training.tfvars.example")
    rollout_example = _read("runtime", "rollout.tfvars.example")
    import re

    assert re.search(r'active_role\s*=\s*"training"', training_example)
    assert re.search(r'active_role\s*=\s*"rollout"', rollout_example)
    for example in (training_example, rollout_example):
        assert "image_id" in example
        assert "manifest_uri" in example or "experiment_manifest" in example
