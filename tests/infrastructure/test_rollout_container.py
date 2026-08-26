"""Static contract for the LeHome-derived rollout container layer."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
APPLIANCE_DIR = REPO_ROOT / "rollout_appliance"

EXPECTED_REPOSITORY = "lehome/docker"
EXPECTED_REVISION = "a914115729bb0bfd260971b9c8d4147bff38c1fb"
EXPECTED_SIZE = 26676771349
EXPECTED_SHA256 = "1a85e389962909debc4ee9988d8a8c388f905fba60686ef78b1623e6872f7123"


def test_artifact_manifest_pins_exact_challenge_tarball():
    manifest = json.loads((APPLIANCE_DIR / "challenge-artifact.json").read_text(encoding="utf-8"))
    assert manifest["repository"] == EXPECTED_REPOSITORY
    assert manifest["revision"] == EXPECTED_REVISION
    assert manifest["size"] == EXPECTED_SIZE
    assert manifest["sha256"] == EXPECTED_SHA256
    assert manifest["filename"] == "lehome-challenge.tar.gz"
    assert manifest["schema_version"] == 1


def test_dockerfile_derives_from_loaded_challenge_image():
    dockerfile = (APPLIANCE_DIR / "Dockerfile").read_text(encoding="utf-8")
    assert "ARG LEHOME_BASE_IMAGE" in dockerfile
    from_lines = [line for line in dockerfile.splitlines() if line.strip().startswith("FROM")]
    assert len(from_lines) == 1
    assert "LEHOME_BASE_IMAGE" in from_lines[0]
    # The default base tag must pin the exact challenge revision.
    assert EXPECTED_REVISION in dockerfile
    assert f"lehome-challenge:{EXPECTED_REVISION}" in dockerfile
    assert f"lehome-challenge@{EXPECTED_REVISION}" not in dockerfile


def test_dockerfile_copies_only_runtime_code_and_no_secrets():
    dockerfile = (APPLIANCE_DIR / "Dockerfile").read_text(encoding="utf-8")
    copy_sources = [
        line.split()[1] for line in dockerfile.splitlines()
        if line.strip().startswith("COPY") and "--from" not in line
    ]
    allowed_prefixes = ("source/", "scripts/", "trainer/src", "trainer/pyproject.toml", "rollout_appliance/")
    assert copy_sources, "the layer must copy its runtime code"
    for source in copy_sources:
        assert source.startswith(allowed_prefixes), f"unexpected COPY source: {source}"
    # Scan only directive content; comments may legitimately mention hygiene.
    directives = "\n".join(
        line for line in dockerfile.splitlines() if not line.strip().startswith("#")
    ).lower()
    for secret_pattern in ("hf_token", "huggingface_token", "password", "api_key", "secret"):
        assert secret_pattern not in directives, f"Dockerfile must not reference secrets: {secret_pattern}"
    assert "--mount=type=secret" not in dockerfile


def test_entrypoint_defaults_to_appliance_supervisor_with_checks():
    entrypoint = (APPLIANCE_DIR / "entrypoint.sh").read_text(encoding="utf-8")
    assert "run_groot_rollout_appliance.py" in entrypoint
    assert "/opt/lehome-challenge/.venv/bin/python" in entrypoint
    assert "/opt/runtime/bin/python" not in entrypoint
    assert "exec" in entrypoint
    # Controller/finalizer/supervisor need no Hub secret. Only the uploader
    # receives a private token-file path at runtime.
    assert "require_env HF_TOKEN" not in entrypoint
    assert "LEHOME_HF_TOKEN_FILE" in entrypoint
    assert "--token-file" in entrypoint
    assert "-n \"" in entrypoint or ":-" in entrypoint
    assert "echo \"$HF_TOKEN\"" not in entrypoint
    assert entrypoint.startswith("#!/")


def test_dockerfile_entrypoint_and_no_baked_model():
    dockerfile = (APPLIANCE_DIR / "Dockerfile").read_text(encoding="utf-8")
    entrypoint_lines = [line for line in dockerfile.splitlines() if line.strip().startswith("ENTRYPOINT")]
    assert len(entrypoint_lines) == 1
    assert "lehome-rollout-entrypoint" in entrypoint_lines[0]
    # Official challenge image python, not the trainer-container /opt/runtime path.
    assert "/opt/lehome-challenge/.venv/bin/python" in dockerfile
    assert "/opt/runtime/bin/python" not in dockerfile

    lowered = dockerfile.lower()
    assert "safetensors" not in lowered
    # No checkpoint may be baked into the derived layer.
    for line in dockerfile.splitlines():
        if line.strip().startswith("COPY"):
            assert "checkpoint" not in line.lower()
            assert ".safetensors" not in line


def test_rollout_layer_carries_the_pinned_geometry_pilot_recipe() -> None:
    dockerfile = (APPLIANCE_DIR / "Dockerfile").read_text(encoding="utf-8")
    campaign = (APPLIANCE_DIR / "run_12k_campaign.sh").read_text(encoding="utf-8")
    for name in (
        "run_12k_campaign.sh",
        "run_randomized_top_short_pilot.sh",
        "campaign_top_short_geometry_pilot.json",
        "campaign_top_short_geometry_pilot.json.sha256",
    ):
        assert name in dockerfile
    assert "build_randomized_pilot_matrix.py" in dockerfile
    assert "build_controlled_recovery_matrix.py" in dockerfile
    assert "run_controlled_recovery_campaign.sh" in dockerfile
    assert '-e LEHOME_EVALUATION_TERMINAL_UPLOAD="${EVALUATION_TERMINAL_UPLOAD}"' in campaign


def _terminal_cpu_evaluation_environment(tmp_path: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update({
        "LEHOME_WORKSPACE": str(tmp_path / "workspace"),
        "LEHOME_CAMPAIGN_ROOT": str(tmp_path / "campaign"),
        "LEHOME_ATTEMPT_MATRIX": str(tmp_path / "missing-matrix.json"),
        "LEHOME_MATRIX_TEMPLATE": str(tmp_path / "missing-template.json"),
        "LEHOME_ATTEMPT_MATRIX_SHA256": "0" * 64,
        "LEHOME_SIMULATOR_DEVICE": "cpu",
        "LEHOME_EVALUATION_TERMINAL_UPLOAD": "1",
        "LEHOME_WORKER_COUNT": "4",
        "LEHOME_ENABLE_HF_UPLOAD": "1",
        "LEHOME_SKIP_ROUND_SEAL": "0",
        "LEHOME_CONTROLLED_RECOVERY_SMOKE": "0",
        "LEHOME_SNAPSHOT_SOURCE_BOOTSTRAP": "0",
        "LEHOME_RESUME_PREEMPTED_ROLLOUT": "0",
    })
    return environment


def test_campaign_admits_cpu_cloth_for_the_exact_terminal_evaluation_tuple(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(APPLIANCE_DIR / "run_12k_campaign.sh")],
        env=_terminal_cpu_evaluation_environment(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "missing 400-attempt matrix" in result.stderr
    assert "CPU cloth requires" not in result.stderr


def test_campaign_rejects_cpu_cloth_when_terminal_evaluation_tuple_is_weakened(tmp_path: Path) -> None:
    environment = _terminal_cpu_evaluation_environment(tmp_path)
    environment["LEHOME_RESUME_PREEMPTED_ROLLOUT"] = "1"
    result = subprocess.run(
        ["bash", str(APPLIANCE_DIR / "run_12k_campaign.sh")],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "CPU cloth requires" in result.stderr


def test_campaign_rejects_cuda_cloth_for_terminal_evaluation(tmp_path: Path) -> None:
    environment = _terminal_cpu_evaluation_environment(tmp_path)
    environment["LEHOME_SIMULATOR_DEVICE"] = "cuda:0"
    result = subprocess.run(
        ["bash", str(APPLIANCE_DIR / "run_12k_campaign.sh")],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "terminal evaluation requires LEHOME_SIMULATOR_DEVICE=cpu" in result.stderr
