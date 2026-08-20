"""Static contract for the two Nebius Packer golden-image templates."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess

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
    assert re.search(r"name\s*=\s*var\.training_image_name", training)
    assert re.search(r"name\s*=\s*var\.rollout_image_name", rollout)
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
            if var_name == "ghcr_pull_token":
                assert re.search(r'default\s*=\s*""', body), "optional GHCR auth must default to no credential"
            else:
                assert not re.search(r"default\s*=", body), f"{var_name} must not carry a default value"

    install = (SCRIPTS_DIR / "install-training.sh").read_text(encoding="utf-8")
    assert "docker login ghcr.io" in install
    assert "docker logout ghcr.io" in install
    logout_index = install.find("docker logout ghcr.io")
    prune_index = install.find("docker system prune")
    assert logout_index != -1 and prune_index != -1 and logout_index < prune_index


def test_training_image_pins_exact_oci_and_code_revision():
    training = _read("training.pkr.hcl")
    install = (SCRIPTS_DIR / "install-training.sh").read_text(encoding="utf-8")
    variables = _read("variables.pkr.hcl")
    combined = training + install + variables
    assert TRAINING_OCI.split("@sha256:")[1] in combined
    assert "ghcr.io/ryanjin333/lehome-groot-n17-trainer" in combined
    # The portable training image never loads the challenge tarball.
    assert "lehome-challenge.tar.gz" not in training
    assert "lehome-challenge.tar.gz" not in install
    assert '[[ "${TRAINER_CODE_REVISION}" =~ ^[0-9a-f]{40}$ ]]' in install
    revision_variable = re.search(
        r'variable\s+"trainer_code_revision"\s*\{([^}]*)\}', variables, re.DOTALL,
    )
    assert revision_variable and not re.search(r"default\s*=", revision_variable.group(1))


def test_training_image_records_a_bare_digest_that_the_guest_accepts():
    training = _read("training.pkr.hcl")
    variables = _read("variables.pkr.hcl")
    install = (SCRIPTS_DIR / "install-training.sh").read_text(encoding="utf-8")

    assert "training_oci_digest" in variables
    assert "TRAINING_OCI_DIGEST='${var.training_oci_digest}'" in training
    assert '^[0-9a-f]{64}$' in install
    assert 'TRAINING_OCI_DIGEST="sha256:${TRAINING_OCI_DIGEST}"' in install
    assert '"oci_digest": "${TRAINING_OCI_DIGEST}"' in install
    assert '"${TRAINING_OCI_IMAGE%@*}@${TRAINING_OCI_DIGEST}"' in install


def test_training_install_captures_only_bare_digest_in_image_manifest(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    etc_root = tmp_path / "etc" / "lehome"
    image = TRAINING_OCI
    bare = image.rsplit(":", 1)[1]
    for name, body in {
        "systemctl": "exit 0",
        "id": "exit 0",
        "useradd": "exit 0",
        "usermod": "exit 0",
        "cp": "exit 0",
        "chown": "exit 0",
        "python3": "exit 0",
        "rm": "exit 0",
        "apt-get": "exit 0",
        "install": 'if [[ "${@: -1}" == "${LEHOME_ETC_DIR}" ]]; then /usr/bin/install "$@"; fi\nexit 0',
        "docker": f'if [[ "${{1:-}}" == inspect ]]; then printf "%s\\n" "{image}"; fi\nexit 0',
    }.items():
        path = fake_bin / name
        path.write_text(f"#!/usr/bin/env bash\nset -euo pipefail\n{body}\n", encoding="utf-8")
        path.chmod(0o755)

    guest = tmp_path / "guest"
    (guest / "bin").mkdir(parents=True)
    (guest / "systemd").mkdir()
    (guest / "bin" / "lehome-experiment-worker.sh").write_text("", encoding="utf-8")
    (guest / "systemd" / "lehome-experiment-worker.service").write_text("", encoding="utf-8")
    (tmp_path / "run_lehome_experiment_worker.py").write_text("", encoding="utf-8")
    (tmp_path / "lehome_train").mkdir()

    result = subprocess.run(
        ["/bin/bash", str(SCRIPTS_DIR / "install-training.sh")],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "TRAINING_OCI_IMAGE": image,
            "TRAINING_OCI_DIGEST": bare,
            "TRAINER_CODE_REVISION": "a" * 40,
            "LEHOME_ETC_DIR": str(etc_root),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((etc_root / "training-image-manifest.json").read_text(encoding="utf-8"))
    assert manifest["oci_image"] == image
    assert manifest["oci_digest"] == f"sha256:{bare}"


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
    # The official tarball loads as lehome-challenge:latest. The derived layer
    # must FROM a valid name:tag that still pins the challenge revision.
    assert 'docker tag' in install
    assert 'PINNED_BASE="lehome-challenge:${CHALLENGE_REVISION}"' in install
    assert 'lehome-challenge@${CHALLENGE_REVISION}' not in install
    assert 'LEHOME_BASE_IMAGE="${PINNED_BASE}"' in install


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
    assert "signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg" in common
    assert "nvidia-driver-580-server-open" in common
    assert "datacenter-gpu-manager-4-cuda12" in common
    assert "nvidia-dcgm.service" in common
    assert "nebius_observability_agent.service" in common
    assert "nebius_observability_agent_updater.service" in common
    assert "lehome-workspace.sh" in common
    assert "lehome-preempt.sh" in common

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


def test_rollout_rebuilds_can_use_versioned_image_name():
    variables = _read("variables.pkr.hcl")
    assert 'variable "rollout_image_name"' in variables
    # Rebuilds must be allowed to override the default name so Nebius does
    # not collide with the previous golden image.
    assert 'default = "lehome-rollout"' in variables


def test_rollout_host_install_carries_the_pinned_geometry_pilot_recipe():
    install = (SCRIPTS_DIR / "install-rollout.sh").read_text(encoding="utf-8")
    for name in (
        "run_12k_campaign.sh",
        "run_randomized_top_short_pilot.sh",
        "campaign_top_short_geometry_pilot.json",
        "campaign_top_short_geometry_pilot.json.sha256",
    ):
        assert name in install
    assert "chmod 0755" in install


def test_rollout_host_install_and_stage_carry_controlled_recovery_runtime():
    install = (SCRIPTS_DIR / "install-rollout.sh").read_text(encoding="utf-8")
    stage = (REPO_ROOT / "infrastructure" / "nebius" / "tools" / "stage-rollout.sh").read_text(encoding="utf-8")
    assert "run_controlled_recovery_campaign.sh" in install
    assert "/opt/lehome/rollout_appliance/run_controlled_recovery_campaign.sh" in install
    assert "build_controlled_recovery_matrix.py" in stage


def test_rollout_image_and_stage_carry_the_success_replay_campaign_tools():
    install = (SCRIPTS_DIR / "install-rollout.sh").read_text(encoding="utf-8")
    stage = (REPO_ROOT / "infrastructure" / "nebius" / "tools" / "stage-rollout.sh").read_text(encoding="utf-8")
    dockerfile = (REPO_ROOT / "rollout_appliance" / "Dockerfile").read_text(encoding="utf-8")
    for content in (install, dockerfile):
        assert "run_success_replay_campaign.sh" in content
        assert "chmod 0755" in content
        assert "bash -n" in content
    assert "build_success_replay_matrix.py" in install
    assert "build_success_replay_matrix.py" in stage
    assert "build_success_replay_matrix.py" in dockerfile
    assert 'chmod 0755 "${STAGE_DIR}/scripts/build_success_replay_matrix.py"' in stage
    assert 'chmod 0755 "${STAGE_DIR}/rollout_appliance/run_success_replay_campaign.sh"' in stage
    assert "/opt/lehome/scripts/build_success_replay_matrix.py" in dockerfile
    assert "chmod 0755 /opt/lehome/scripts/build_success_replay_matrix.py" in dockerfile


def test_rollout_host_install_carries_the_experiment_evaluator_wrapper():
    install = (SCRIPTS_DIR / "install-rollout.sh").read_text(encoding="utf-8")
    stage = (REPO_ROOT / "infrastructure/nebius/tools/stage-rollout.sh").read_text(encoding="utf-8")
    assert "run_experiment_evaluator.sh" in install
    assert "/opt/lehome/rollout_appliance/run_experiment_evaluator.sh" in install
    assert "run_lehome_experiment_evaluator.py" in stage
    assert "run_lehome_experiment_worker.py" in stage
    assert "/opt/lehome/scripts/run_lehome_experiment_evaluator.py" in install


def test_pinned_tool_bootstrap_version_probe_cannot_sigpipe_under_pipefail():
    bootstrap = (REPO_ROOT / "infrastructure/nebius/tools/bootstrap.sh").read_text(encoding="utf-8")
    assert "| head" not in bootstrap
    assert 'version | sed -n \'1p\'' in bootstrap


def test_free_validation_checks_randomized_campaign_shell_recipes():
    validation = (REPO_ROOT / "infrastructure" / "nebius" / "validate.sh").read_text(encoding="utf-8")
    assert "rollout_appliance/run_12k_campaign.sh" in validation
    assert "rollout_appliance/run_randomized_top_short_pilot.sh" in validation


def test_rollout_stage_includes_the_randomized_matrix_builder():
    stage = (REPO_ROOT / "infrastructure" / "nebius" / "tools" / "stage-rollout.sh").read_text(encoding="utf-8")
    assert "build_randomized_pilot_matrix.py" in stage


def test_rollout_stage_includes_the_persistent_evaluation_summarizer():
    stage = (REPO_ROOT / "infrastructure" / "nebius" / "tools" / "stage-rollout.sh").read_text(encoding="utf-8")
    assert "summarize_groot_persistent_evaluation.py" in stage


def test_incremental_rollout_image_uses_existing_ready_image_without_redownloading_tarball():
    patch_template = _read("rollout-patch.pkr.hcl")
    patch_install = (SCRIPTS_DIR / "install-rollout-patch.sh").read_text(encoding="utf-8")

    assert 'base_image {' in patch_template
    assert 'id = var.rollout_parent_image_id' in patch_template
    assert 'source "nebius-image" "lehome-rollout-patch"' in patch_template
    assert "rollout-stage" in patch_template
    assert "lehome-rollout:build" in patch_install
    assert "docker build" in patch_install
    assert "/opt/lehome/rollout_appliance" in patch_install
    assert "lehome-challenge.tar.gz" not in patch_template + patch_install
    assert "docker load" not in patch_install
    assert "curl " not in patch_install


def test_patch_parent_is_optional_for_full_training_and_rollout_builds():
    variables = _read("variables.pkr.hcl")
    block = re.search(r'variable\s+"rollout_parent_image_id"\s*{([^}]*)}', variables, re.DOTALL)
    assert block
    assert re.search(r'default\s*=\s*""', block.group(1))


def test_incremental_rollout_image_refreshes_guest_workspace_services():
    stage = (REPO_ROOT / "infrastructure" / "nebius" / "tools" / "stage-rollout.sh").read_text(encoding="utf-8")
    patch_install = (SCRIPTS_DIR / "install-rollout-patch.sh").read_text(encoding="utf-8")

    assert 'cp -R "${REPO_ROOT}/infrastructure/nebius/guest" "${STAGE_DIR}/guest"' in stage
    for name in (
        "lehome_workspace.py",
        "lehome_preempt.py",
        "lehome-workspace.sh",
        "lehome-preempt.sh",
        "lehome-workspace.service",
        "lehome-preempt.service",
    ):
        assert name in patch_install
    assert "systemctl daemon-reload" in patch_install


def test_incremental_rollout_image_installs_the_controlled_recovery_wrapper_executable():
    patch_install = (SCRIPTS_DIR / "install-rollout-patch.sh").read_text(encoding="utf-8")

    assert "/opt/lehome/rollout_appliance/run_controlled_recovery_campaign.sh" in patch_install
    assert "chmod 0755" in patch_install


def test_incremental_rollout_image_installs_the_experiment_evaluator_runtime():
    patch_install = (SCRIPTS_DIR / "install-rollout-patch.sh").read_text(encoding="utf-8")

    assert "/opt/lehome/rollout_appliance/run_experiment_evaluator.sh" in patch_install
    assert "cp -a /tmp/lehome-repo/scripts/. /opt/lehome/scripts/" in patch_install
    assert "/opt/lehome/scripts/summarize_groot_persistent_evaluation.py" in patch_install
    assert "/tmp/lehome-repo/guest/systemd/lehome-experiment-evaluator.service" in patch_install
    assert "/etc/systemd/system/lehome-experiment-evaluator.service" in patch_install
    assert "systemctl daemon-reload" in patch_install
    assert "systemctl disable lehome-experiment-evaluator.service" in patch_install


def test_incremental_rollout_replaces_the_canonical_host_source_without_nesting():
    patch_install = (SCRIPTS_DIR / "install-rollout-patch.sh").read_text(encoding="utf-8")

    # The parent image already contains /opt/lehome/source/lehome. Copying the
    # directory itself into that existing destination creates
    # /opt/lehome/source/lehome/lehome and leaves the stale package active.
    assert "rm -rf -- /opt/lehome/source/lehome" in patch_install
    assert "cp -a /tmp/lehome-repo/source/lehome /opt/lehome/source/lehome" in patch_install
    assert "cp -a /tmp/lehome-repo/source/lehome/. /opt/lehome/source/lehome/" not in patch_install


def test_rollout_images_bake_the_pinned_policy_server_image_for_cold_boots():
    rollout = _read("rollout.pkr.hcl")
    full_install = (SCRIPTS_DIR / "install-rollout.sh").read_text(encoding="utf-8")
    patch_install = (SCRIPTS_DIR / "install-rollout-patch.sh").read_text(encoding="utf-8")

    for install in (full_install, patch_install):
        assert TRAINING_OCI in install
        assert 'docker pull "${TRAINER_IMAGE}"' in install
    disk_size = re.search(r"disk\s*{[^}]*size_gibibytes\s*=\s*(\d+)", rollout, re.DOTALL)
    assert disk_size and int(disk_size.group(1)) >= 192


def test_incremental_rollout_retries_transient_trainer_image_pull(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker_log = tmp_path / "docker.log"
    pull_count = tmp_path / "pull-count"

    docker = fake_bin / "docker"
    docker.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "${DOCKER_LOG}"
if [ "${1:-}" = pull ]; then
  count=0
  if [ -f "${PULL_COUNT}" ]; then count="$(cat "${PULL_COUNT}")"; fi
  count="$((count + 1))"
  printf '%s\\n' "${count}" > "${PULL_COUNT}"
  if [ "${count}" -lt 3 ]; then
    echo 'error from registry: retry-after' >&2
    exit 1
  fi
fi
exit 0
""",
        encoding="utf-8",
    )
    docker.chmod(0o755)

    for command in ("systemctl", "git", "install", "cp", "chmod", "sleep", "sync", "rm"):
        executable = fake_bin / command
        executable.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "DOCKER_LOG": str(docker_log),
            "PULL_COUNT": str(pull_count),
        }
    )
    result = subprocess.run(
        ["/bin/bash", str(SCRIPTS_DIR / "install-rollout-patch.sh")],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert pull_count.read_text(encoding="utf-8").strip() == "3"
    pulls = [line for line in docker_log.read_text(encoding="utf-8").splitlines() if line.startswith("pull ")]
    assert pulls == [f"pull {TRAINING_OCI}"] * 3
