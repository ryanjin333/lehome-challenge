from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_root_build_context_includes_only_required_rollout_assets() -> None:
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    dockerfile = (ROOT / "rollout/Dockerfile").read_text(encoding="utf-8")

    for required in (
        "!rollout/",
        "!rollout/Dockerfile",
        "!rollout/pyproject.toml",
        "!rollout/uv.lock",
        "!rollout/src/",
        "!rollout/src/**",
        "!rollout/task-manifest.json",
        "!rollout/entrypoint.sh",
    ):
        assert required in dockerignore
    assert "ARG BEHAVIOR_PARENT_IMAGE" in dockerfile
    assert "FROM ${BEHAVIOR_PARENT_IMAGE}" in dockerfile
    assert dockerfile.count("ARG BEHAVIOR_PARENT_IMAGE") == 2
    assert "COPY rollout/pyproject.toml rollout/uv.lock ./" in dockerfile
