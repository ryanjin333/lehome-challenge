from __future__ import annotations

import re
from pathlib import Path

from b1k_rollout.identity import BEHAVIOR_REVISION, GROOT_REVISION


ROLLOUT = Path(__file__).parents[1]


def _read(relative_path: str) -> str:
    return (ROLLOUT / relative_path).read_text(encoding="utf-8")


def test_rollout_image_is_pinned_headless_and_secret_free() -> None:
    dockerfile = _read("Dockerfile")

    assert BEHAVIOR_REVISION in dockerfile
    assert GROOT_REVISION in dockerfile
    parent_image = "stanfordvl/behavior@sha256:b789b8d8efefda509b37404a676523d6cee81e2860558287cf6c34c2af3b79c7"
    assert parent_image in dockerfile
    assert len(parent_image.rsplit(":", 1)[1]) == 64
    assert 'io.lehome.behavior-parent-digest="${BEHAVIOR_PARENT_DIGEST}"' in dockerfile
    assert "OMNI_KIT_ACCEPT_EULA=YES" in dockerfile
    assert "OMNIGIBSON_DATA_PATH=/workspace/omnigibson-data" in dockerfile
    assert "HEADLESS=1" in dockerfile
    assert "HEALTHCHECK" in dockerfile and "/opt/conda/envs/behavior/bin/python -m b1k_rollout.cli healthcheck" in dockerfile
    assert "--start-period=45m" in dockerfile
    assert "USER rollout" not in dockerfile
    assert "setpriv --reuid=10001 --regid=10001" in _read("entrypoint.sh")
    assert "uv sync --frozen --no-dev" in dockerfile
    assert "--no-install-project" not in dockerfile
    assert "GROOT_PYTHON=/opt/isaac-groot/.venv/bin/python" in dockerfile
    assert dockerfile.index("rm -rf /behavior-src /opt/isaac-groot") < dockerfile.index("git init /behavior-src")
    for forbidden in ("novnc", "x11vnc", "xfce", "jupyter", "HF_TOKEN="):
        assert forbidden not in dockerfile.casefold()
    assert not re.search(r"hf_[A-Za-z0-9]{30,}", dockerfile)


def test_rollout_entrypoint_fails_closed_before_campaign_execution() -> None:
    entrypoint = _read("entrypoint.sh")

    assert "B1K_HF_TOKEN_FILE" in entrypoint
    assert '"${1:-}" == "smoke-runtime"' in entrypoint
    assert "must be a regular file" in entrypoint
    assert "must not be readable by group or other" in entrypoint
    assert "OMNI_KIT_ACCEPT_EULA" in entrypoint
    assert "OMNIGIBSON_DATA_PATH" in entrypoint
    assert "-m b1k_rollout.cli preflight" in entrypoint
    assert "-m b1k_rollout.cli campaign" in entrypoint
    assert "unset HF_TOKEN" in entrypoint
    assert "B1K_ROLLOUT_VERIFY_PRIVILEGE_DROP" in entrypoint
    assert "-m b1k_rollout.cli assets-bootstrap" in entrypoint
    root_branch = entrypoint.split('if [[ ! -f "${B1K_HF_TOKEN_FILE:-}"', 1)[0]
    assert '"$BEHAVIOR_PYTHON" -m b1k_rollout.cli assets-bootstrap' not in root_branch
    assert entrypoint.index("assets-bootstrap") > entrypoint.index("unset HF_TOKEN")


def test_image_verifier_accepts_only_immutable_rollout_digests() -> None:
    verifier = _read("scripts/verify-image.sh")

    assert "behavior1k-groot-n17@sha256" in verifier
    assert "docker image inspect" in verifier
    assert "io.lehome.behavior-parent-digest" in verifier
    assert "b1k_rollout.cli healthcheck" in verifier
    assert "b1k_rollout.policy_server" in verifier
    assert "io.lehome.behavior-revision" in verifier
    assert "credential material" in verifier
    assert 'actual_user" != ""' in verifier
    assert "B1K_ROLLOUT_VERIFY_PRIVILEGE_DROP=1" in verifier
    assert "B1K_HF_TOKEN_FILE=/workspace/.cache/huggingface/token" in verifier
    assert "/opt/conda/envs/behavior/bin/python -m b1k_rollout.cli" in verifier


def test_ci_secret_scan_excludes_only_known_synthetic_fixture_files() -> None:
    workflow = (ROLLOUT.parent / ".github/workflows/groot-trainer-image.yml").read_text(encoding="utf-8")

    for fixture in (
        "deployment/tests/test_dockerhub.py",
        "deployment/tests/test_huggingface.py",
        "deployment/tests/test_ledger.py",
        "rollout/tests/test_provenance.py",
        "rollout/tests/test_publisher.py",
    ):
        assert f":(exclude){fixture}" in workflow
    assert ":(exclude)rollout/tests/**" not in workflow
    assert ":(exclude)deployment/tests/**" not in workflow


def test_cpu_workflow_covers_the_deployment_project_in_its_own_frozen_environment() -> None:
    workflow = (ROLLOUT.parent / ".github/workflows/groot-trainer-image.yml").read_text(encoding="utf-8")

    assert '"deployment/**"' in workflow
    assert "uv lock --check --project deployment" in workflow
    assert "uv run --project deployment --frozen --no-editable pytest deployment/tests -q" in workflow
