"""Pinned Packer/Terraform bootstrap: checksums, ignore rules, no paid calls."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "infrastructure" / "nebius" / "tools"
BOOTSTRAP = TOOLS_DIR / "bootstrap.sh"
GITIGNORE = REPO_ROOT / ".gitignore"


def test_bootstrap_pins_platform_specific_releases_with_checksums():
    script = BOOTSTRAP.read_text(encoding="utf-8")
    for platform_key in ("darwin_amd64", "darwin_arm64", "linux_amd64", "linux_arm64"):
        assert platform_key in script, f"missing platform pin: {platform_key}"
    # Every download is checksum-verified.
    assert "sha256sum --check" in script or "sha256sum -c" in script
    assert "PKCS11" not in script
    # The published checksums themselves are pinned, not fetched at runtime.
    assert script.count("sha256") >= 8


def test_bootstrap_never_invokes_paid_commands():
    script = BOOTSTRAP.read_text(encoding="utf-8")
    forbidden = (
        "terraform apply",
        "packer build",
        "nebius compute",
        "instance create",
        "disk create",
    )
    for command in forbidden:
        assert command not in script, f"bootstrap must never run: {command}"


def test_bootstrap_installs_into_ignored_tools_directory():
    script = BOOTSTRAP.read_text(encoding="utf-8")
    assert "infrastructure/nebius/.tools" in script or ".tools" in script
    gitignore = GITIGNORE.read_text(encoding="utf-8")
    assert "infrastructure/nebius/.tools/" in gitignore


def test_gitignore_covers_state_and_secrets_but_not_lockfiles():
    gitignore = GITIGNORE.read_text(encoding="utf-8")
    assert ".terraform/" in gitignore
    assert "*.tfstate" in gitignore
    assert ".auto.tfvars" in gitignore or "*.auto.tfvars" in gitignore
    # Lockfiles stay trackable: no rule may ignore them.
    assert ".terraform.lock.hcl" not in gitignore


def test_validate_entrypoint_never_runs_paid_commands():
    validate = (REPO_ROOT / "infrastructure" / "nebius" / "validate.sh").read_text(encoding="utf-8")
    # Paid commands may be printed as guidance, never executed.
    executable_lines = [
        line.strip() for line in validate.splitlines()
        if line.strip() and not line.strip().startswith("#") and not line.strip().startswith("echo")
    ]
    executable_text = "\n".join(executable_lines)
    for command in ("terraform apply", "packer build", "nebius compute"):
        assert command not in executable_text, f"validate.sh must never run: {command}"
    assert '"${PACKER}" validate' in validate
    assert '"${TERRAFORM}"' in validate and "validate" in validate
    assert "pytest" in validate
