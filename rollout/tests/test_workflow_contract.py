from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_ci_builds_trainer_and_rollout_as_separate_digest_pinned_repositories() -> None:
    workflow = (ROOT / ".github/workflows/groot-trainer-image.yml").read_text(encoding="utf-8")

    assert '"rollout/**"' in workflow
    assert "docker.io/ryanjin333/behavior1k-groot-n17-trainer" in workflow
    assert "docker.io/ryanjin333/behavior1k-groot-n17-rollout" in workflow
    assert "rollout/Dockerfile" in workflow
    assert "target: rollout-runtime" in workflow
    assert "BEHAVIOR_PARENT_IMAGE" in workflow
    assert "rollout/scripts/verify-image.sh" in workflow
