"""Static contract for the LeHome-derived rollout container layer."""

from __future__ import annotations

import json
from pathlib import Path

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
    assert f"lehome-challenge@{EXPECTED_REVISION}" in dockerfile or EXPECTED_REVISION in dockerfile


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
    assert "exec" in entrypoint
    # Non-secret admission checks: presence only, never value printing.
    assert "HF_TOKEN" in entrypoint
    assert "-n \"" in entrypoint or ":-" in entrypoint
    assert "echo \"$HF_TOKEN\"" not in entrypoint
    assert entrypoint.startswith("#!/")


def test_dockerfile_entrypoint_and_no_baked_model():
    dockerfile = (APPLIANCE_DIR / "Dockerfile").read_text(encoding="utf-8")
    entrypoint_lines = [line for line in dockerfile.splitlines() if line.strip().startswith("ENTRYPOINT")]
    assert len(entrypoint_lines) == 1
    assert "lehome-rollout-entrypoint" in entrypoint_lines[0]
    lowered = dockerfile.lower()
    assert "safetensors" not in lowered
    # No checkpoint may be baked into the derived layer.
    for line in dockerfile.splitlines():
        if line.strip().startswith("COPY"):
            assert "checkpoint" not in line.lower()
            assert ".safetensors" not in line
