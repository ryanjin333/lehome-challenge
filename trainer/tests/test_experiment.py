from __future__ import annotations

import json
from pathlib import Path

from lehome_train.experiment import (
    create_or_resume_experiment,
    experiment_id,
    mark_stage_complete,
    pending_stages,
)
from lehome_train.models import ArtifactIdentity


def artifact() -> ArtifactIdentity:
    return ArtifactIdentity("meta/stats.json", "a" * 64, 12)


def config() -> dict[str, object]:
    return {"dataset_revision": "b" * 40, "model_revision": "c" * 40, "batch": 64}


def test_identity_is_canonical_and_matching_partial_experiment_resumes(tmp_path: Path) -> None:
    first = create_or_resume_experiment(tmp_path, resolved_config=config(), artifacts=(artifact(),))
    mark_stage_complete(first, "image_runtime_verification", duration_seconds=1.25)

    resumed = create_or_resume_experiment(tmp_path, resolved_config=dict(reversed(list(config().items()))), artifacts=(artifact(),))

    assert resumed.experiment_id == first.experiment_id == experiment_id(config(), (artifact(),))
    assert resumed.root == first.root
    assert resumed.resumed is True
    assert pending_stages(resumed)[0] == "network_measurement"
    assert (first.root / "logs" / "prepare.log").is_file()
    assert (first.root / "status.json").is_file()


def test_completed_stage_is_skipped_and_incompatible_collision_is_preserved(tmp_path: Path) -> None:
    first = create_or_resume_experiment(tmp_path, resolved_config=config(), artifacts=(artifact(),))
    for stage in pending_stages(first):
        mark_stage_complete(first, stage, duration_seconds=0.0)

    completed = create_or_resume_experiment(tmp_path, resolved_config=config(), artifacts=(artifact(),))
    assert pending_stages(completed) == ()

    identity_path = first.root / "experiment.json"
    identity_path.write_text(json.dumps({"wrong": "identity"}), encoding="utf-8")
    replacement = create_or_resume_experiment(tmp_path, resolved_config=config(), artifacts=(artifact(),))

    assert replacement.root != first.root
    assert first.root.is_dir()
    assert replacement.root.name.startswith(first.experiment_id + "-superseded-")


def test_incompatible_partial_status_is_preserved_and_never_resumed(tmp_path: Path) -> None:
    first = create_or_resume_experiment(tmp_path, resolved_config=config(), artifacts=(artifact(),))
    (first.root / "status.json").write_text('{"bad":"status"}', encoding="utf-8")

    superseded = create_or_resume_experiment(tmp_path, resolved_config=config(), artifacts=(artifact(),))

    assert superseded.root != first.root
    assert (first.root / "status.json").read_text(encoding="utf-8") == '{"bad":"status"}'
