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
    assert "OMNIGIBSON_APPDATA_PATH=/workspace/campaign/.cache/omnigibson" in dockerfile
    assert "WARP_CACHE_PATH=/workspace/campaign/.cache/warp" in dockerfile
    assert "HEADLESS=1" in dockerfile
    assert "HEALTHCHECK" in dockerfile and "/opt/conda/envs/behavior/bin/python -m b1k_rollout.cli healthcheck" in dockerfile
    assert "--start-period=45m" in dockerfile
    assert "USER rollout" not in dockerfile
    assert "setpriv --reuid=10001 --regid=10001" in _read("entrypoint.sh")
    assert "uv sync --frozen --no-dev" in dockerfile
    assert "UV_HTTP_TIMEOUT=600" in dockerfile
    assert dockerfile.index("UV_HTTP_TIMEOUT=600") < dockerfile.index("uv sync --frozen --no-dev")
    assert "--no-install-project" not in dockerfile
    assert "GROOT_PYTHON=/opt/isaac-groot/.venv/bin/python" in dockerfile
    assert '"$BEHAVIOR_PYTHON" -c \'import b1k_rollout.policy_server\'' in dockerfile
    assert "python3 -c 'import b1k_rollout.policy_server'" not in dockerfile
    assert dockerfile.index("WORKDIR /\n") < dockerfile.index(
        "rm -rf /behavior-src /opt/isaac-groot"
    )
    assert dockerfile.index("rm -rf /behavior-src /opt/isaac-groot") < dockerfile.index("git init /behavior-src")
    for forbidden in ("novnc", "x11vnc", "xfce", "jupyter", "HF_TOKEN="):
        assert forbidden not in dockerfile.casefold()
    assert not re.search(r"hf_[A-Za-z0-9]{30,}", dockerfile)


def test_rollout_image_prepares_only_omnigibson_runtime_copy_targets_for_unprivileged_launch() -> None:
    dockerfile = _read("Dockerfile")

    assert 'importlib.util.find_spec("isaacsim")' in dockerfile
    assert "spec.submodule_search_locations" in dockerfile
    assert "spec.origin" not in dockerfile
    assert 'test -d "$isaacsim_apps_path"' in dockerfile
    assert (
        "install -o rollout -g rollout -m 0644 "
        "/behavior-src/OmniGibson/omnigibson/omnigibson_5_1_0.kit "
        '"$isaacsim_apps_path/omnigibson_5_1_0.kit"'
    ) in dockerfile
    assert (
        "install -o rollout -g rollout -m 0644 "
        "/behavior-src/OmniGibson/docs/assets/OmniGibson_logo.png "
        '"$isaacsim_apps_path/OmniGibson_logo.png"'
    ) in dockerfile
    assert 'chown -R rollout:rollout "$isaacsim_apps_path"' not in dockerfile


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
    assert (
        "install -d -o 10001 -g 10001 -m 0700 \\\n"
        "    /workspace /workspace/campaign /workspace/checkpoint-source \\\n"
        "    /workspace/omnigibson-data /workspace/smoke-canary \\\n"
        "    /workspace/campaign/.cache/numba /workspace/campaign/.cache/triton \\\n"
        "    /workspace/campaign/.cache/matplotlib /workspace/campaign/.cache/omnigibson \\\n"
        "    /workspace/campaign/.cache/warp"
    ) in entrypoint
    root_branch = entrypoint.split('if [[ ! -f "${B1K_HF_TOKEN_FILE:-}"', 1)[0]
    assert "export OMNIGIBSON_APPDATA_PATH=/workspace/campaign/.cache/omnigibson" in root_branch
    assert "export NUMBA_CACHE_DIR=/workspace/campaign/.cache/numba" in root_branch
    assert "export TRITON_CACHE_DIR=/workspace/campaign/.cache/triton" in root_branch
    assert "export MPLCONFIGDIR=/workspace/campaign/.cache/matplotlib" in root_branch
    assert "export WARP_CACHE_PATH=/workspace/campaign/.cache/warp" in root_branch
    assert ": > /workspace/smoke-canary/rollout-ready" in entrypoint
    assert ": > /workspace/.b1k-rollout-smoke-ready" not in entrypoint
    assert "/workspace/.cache /workspace/.cache/huggingface" not in root_branch
    assert '"$BEHAVIOR_PYTHON" -m b1k_rollout.cli assets-bootstrap' not in root_branch
    assert entrypoint.index("assets-bootstrap") > entrypoint.index("unset HF_TOKEN")
    assert 'test -d "$OMNIGIBSON_APPDATA_PATH"' in entrypoint
    assert 'test -O "$OMNIGIBSON_APPDATA_PATH"' in entrypoint


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
    assert 'test "$OMNIGIBSON_APPDATA_PATH" = /workspace/campaign/.cache/omnigibson' in verifier
    assert 'test "$WARP_CACHE_PATH" = /workspace/campaign/.cache/warp' in verifier


def test_image_verifier_does_not_treat_generated_third_party_environments_as_release_source() -> None:
    verifier = _read("scripts/verify-image.sh")

    assert "--exclude-dir=.venv" in verifier
    assert "--exclude-dir=.git" in verifier
    assert "/opt/rollout /behavior-src /opt/isaac-groot" in verifier


def test_image_verifier_is_executable_by_ci() -> None:
    assert (ROLLOUT / "scripts/verify-image.sh").stat().st_mode & 0o111


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
