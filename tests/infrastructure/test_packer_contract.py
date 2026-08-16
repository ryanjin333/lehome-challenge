"""Static contract for the two Nebius Packer golden-image templates."""

from __future__ import annotations

import json
from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKER_DIR = REPO_ROOT / "infrastructure" / "nebius" / "packer"
SCRIPTS_DIR = PACKER_DIR / "scripts"

TRAINING_OCI = (
    "ghcr.io/ryanjin333/lehome-groot-n17-trainer"
    "@sha256:b56c16c259b7eda99294f2069e976b53395e665aaf68174d5b13ba458a93b746"
)
CHALLENGE_SHA256 = "1a85e389962909debc4ee9988d8a8c388f905fba60686ef78b1623e6872f7123"
CHALLENGE_SIZE = 26676771349
CHALLENGE_REVISION = "a914115729bb0bfd260971b9c8d4147bff38c1fb"


def _read(name: str) -> str:
    return (PACKER_DIR / name).read_text(encoding="utf-8")


def test_plugin_block_pins_nebius_builder_version():
    plugins = _read("plugins.pkr.hcl")
    assert 'source  = "github.com/nebius/nebius"' in plugins
    assert 'version = "= 0.0.7"' in plugins


def test_cpu_builder_shape_and_base_image():
    for name in ("training.pkr.hcl", "rollout.pkr.hcl"):
        template = _read(name)
        assert 'platform = "cpu-d3"' in template, name
        assert 'preset   = "16vcpu-64gb"' in template, name
        assert 'family = "ubuntu24.04-driverless"' in template, name
        assert 'ssh_username = "ubuntu"' in template, name


def test_image_names_are_unique_and_role_specific():
    training = _read("training.pkr.hcl")
    rollout = _read("rollout.pkr.hcl")
    assert "name         = var.training_image_name" in training
    assert "name         = var.rollout_image_name" in rollout
    variables = _read("variables.pkr.hcl")
    training_default = re.search(
        r'variable\s+"training_image_name"\s*{[^}]*default\s*=\s*"([^"]+)"', variables, re.DOTALL,
    )
    rollout_default = re.search(
        r'variable\s+"rollout_image_name"\s*{[^}]*default\s*=\s*"([^"]+)"', variables, re.DOTALL,
    )
    assert training_default and training_default.group(1) == "vla-training-base"
    assert rollout_default and rollout_default.group(1) == "lehome-rollout"
    assert training_default.group(1) != rollout_default.group(1)


def test_no_secrets_baked_into_any_template():
    for name in ("variables.pkr.hcl", "training.pkr.hcl", "rollout.pkr.hcl", "plugins.pkr.hcl"):
        lowered = _read(name).lower()
        for pattern in ("hf_token", "huggingface_token", "password", "api_key", "sk_hf", "hf_"):
            assert pattern not in lowered, (name, pattern)
    # Service-account credentials may only arrive through sensitive variables
    # without defaults; the builder never stores a literal key.
    variables = _read("variables.pkr.hcl")
    for block in re.finditer(r"variable\s+\"([^\"]+)\"\s*{([^}]*)}", variables, re.DOTALL):
        var_name, body = block.group(1), block.group(2)
        if any(fragment in var_name for fragment in ("private_key", "token", "credential")):
            assert re.search(r"sensitive\s*=\s*true", body), var_name
            assert not re.search(r"default\s*=", body), f"{var_name} must not carry a default value"


def test_training_image_pins_exact_oci_and_code_revision():
    training = _read("training.pkr.hcl")
    install = (SCRIPTS_DIR / "install-training.sh").read_text(encoding="utf-8")
    combined = training + install + _read("variables.pkr.hcl")
    assert TRAINING_OCI.split("@sha256:")[1] in combined
    assert "ghcr.io/ryanjin333/lehome-groot-n17-trainer" in combined
    # The portable training image never loads the challenge tarball.
    assert "lehome-challenge.tar.gz" not in training
    assert "lehome-challenge.tar.gz" not in install


def test_rollout_image_verifies_tarball_before_use():
    rollout = _read("rollout.pkr.hcl")
    install = (SCRIPTS_DIR / "install-rollout.sh").read_text(encoding="utf-8")
    combined = rollout + install
    assert CHALLENGE_SHA256 in combined
    assert str(CHALLENGE_SIZE) in combined
    assert CHALLENGE_REVISION in combined
    assert "sha256sum" in install
    # In the actual build script, size and hash must be checked before the
    # real docker load command (not the comment mentioning it).
    verify_index = install.find("sha256sum")
    load_index = install.find("docker load --input")
    assert verify_index != -1 and load_index != -1 and verify_index < load_index


def test_rollout_boot_disk_headroom_covers_tarball_and_layers():
    rollout = _read("rollout.pkr.hcl")
    match = re.search(r"size_gibibytes\s*=\s*(\d+)", rollout)
    assert match, "rollout template must pin a boot disk size"
    size_gib = int(match.group(1))
    # 26.7 GB tarball + loaded layers + derived layer + overhead.
    assert size_gib >= 128


def test_both_templates_install_guest_services_and_cleanup():
    common = (SCRIPTS_DIR / "install-common.sh").read_text(encoding="utf-8")
    assert "lehome_workspace.py" in common
    assert "lehome_preempt.py" in common
    assert "lehome-workspace.service" in common
    assert "lehome-preempt.service" in common

    install_rollout = (SCRIPTS_DIR / "install-rollout.sh").read_text(encoding="utf-8")
    # Downloaded tarball and build caches must not survive image capture.
    assert "rm -f" in install_rollout and "lehome-challenge.tar.gz" in install_rollout
    assert "docker system prune" in install_rollout or "docker builder prune" in install_rollout


def test_builder_is_documented_as_on_demand_cpu_not_preemptible():
    for name in ("training.pkr.hcl", "rollout.pkr.hcl"):
        stripped = _read(name).replace("#", " ")
        template = re.sub(r"\s+", " ", stripped).lower()
        assert "on-demand" in template or "temporary cpu builder" in template
        assert "not preemptible" in template
