from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_ci_builds_role_prefixed_trainer_and_rollout_tags_in_the_shared_private_repository() -> None:
    workflow = (ROOT / ".github/workflows/groot-trainer-image.yml").read_text(encoding="utf-8")

    assert '"rollout/**"' in workflow
    assert "IMAGE_REPOSITORY: docker.io/ryanjin333/behavior1k-groot-n17" in workflow
    assert "type=raw,value=trainer-${{ github.sha }}" in workflow
    assert "type=raw,value=rollout-${{ github.sha }}" in workflow
    assert "rollout/Dockerfile" in workflow
    assert "target: rollout-runtime" in workflow
    assert "BEHAVIOR_PARENT_IMAGE" in workflow
    assert "rollout/scripts/verify-image.sh" in workflow
