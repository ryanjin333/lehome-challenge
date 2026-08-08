from __future__ import annotations

import json
from pathlib import Path

import pytest

from lehome.flywheel.artifacts import (
    MANIFEST_NAME,
    EpisodeArtifactWriter,
    atomic_write_json,
    build_sha256_manifest,
)
from lehome.flywheel.matrix import PublicMatrix, Trial
from lehome.flywheel.release import (
    build_release_plan,
    materialize_release,
    validate_remote_file_tree,
    verify_release_tree,
)


POLICY_REVISION = "a" * 40
CODE_REVISION = "b" * 40
ASSET_REVISION = "c" * 40
POLICY_DIGEST = "e" * 64
IMAGE_IDENTITY = "sha256:" + "d" * 64


def _matrix() -> PublicMatrix:
    return PublicMatrix(
        schema_version=1,
        trials=(
            Trial("top_long", "Top_Long_Seen_0", "seen", 101),
            Trial("pant_short", "Pant_Short_Unseen_1", "public_unseen", 601),
        ),
        training_holdouts=("Pant_Short_Unseen_1",),
    )


def _episode(root: Path, trial: Trial, *, steps: int = 2, extra_file: bool = False) -> None:
    writer = EpisodeArtifactWriter(root, trial.trial_id)
    for index in range(steps):
        writer.append_annotation({"step": index})
    for relative in (
        "snapshots/reset.json",
        "snapshots/terminal.json",
        "videos/left_rgb.mp4",
        "videos/right_rgb.mp4",
        "videos/top_rgb.mp4",
    ):
        path = writer.staging / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode("utf-8"))
    if extra_file:
        (writer.staging / "debug.log").write_text("not approved\n", encoding="utf-8")
    writer.finalize(
        {
            "episode_id": trial.trial_id,
            "identity": {
                "category": trial.category,
                "garment_name": trial.garment_name,
                "release_stage": trial.release_stage,
                "seed": trial.seed,
                "policy_repo": "ryanjin333/lehome-groot-n17-models",
                "policy_revision": POLICY_REVISION,
                "policy_step": 12000,
                "code_revision": CODE_REVISION,
                "asset_revision": ASSET_REVISION,
                "simulator_version": "5.1.0.0",
            },
            "provenance": {
                "policy_artifact_sha256": POLICY_DIGEST,
                "image_identity": IMAGE_IDENTITY,
            },
        },
        required_videos=("left_rgb.mp4", "right_rgb.mp4", "top_rgb.mp4"),
    )


def test_release_plan_requires_exact_verified_matrix_and_sorts_episodes(tmp_path: Path) -> None:
    matrix = _matrix()
    for trial in matrix.trials:
        _episode(tmp_path, trial)
    (tmp_path / "rollout-report.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "capacity-report.json").write_text("{}\n", encoding="utf-8")

    plan = build_release_plan(tmp_path, matrix, expected_steps=2)

    assert plan.episode_count == 2
    assert plan.category_counts == {"pant_short": 1, "top_long": 1}
    assert plan.release_stage_counts == {"public_unseen": 1, "seen": 1}
    assert plan.episode_paths == (
        "raw/pant_short/public_unseen/pant-short-public-unseen-1-seed-601",
        "raw/top_long/seen/top-long-seen-0-seed-101",
    )

    unexpected = tmp_path / "raw" / "unexpected"
    unexpected.mkdir()
    with pytest.raises(ValueError, match="exact matrix"):
        build_release_plan(tmp_path, matrix, expected_steps=2)


def test_release_plan_rejects_identity_and_annotation_mismatch(tmp_path: Path) -> None:
    matrix = PublicMatrix(1, (_matrix().trials[0],), ())
    _episode(tmp_path, matrix.trials[0], steps=1)
    (tmp_path / "rollout-report.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "capacity-report.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="annotation count"):
        build_release_plan(tmp_path, matrix, expected_steps=2)

    episode = tmp_path / "raw" / matrix.trials[0].trial_id / "episode.json"
    payload = json.loads(episode.read_text(encoding="utf-8"))
    payload["identity"]["seed"] = 999
    episode.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest (size|hash) mismatch"):
        build_release_plan(tmp_path, matrix, expected_steps=1)


def test_release_plan_rejects_manifested_files_outside_exact_allowlist(tmp_path: Path) -> None:
    matrix = PublicMatrix(1, (_matrix().trials[0],), ())
    _episode(tmp_path, matrix.trials[0], extra_file=True)
    (tmp_path / "rollout-report.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "capacity-report.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exact release allowlist"):
        build_release_plan(tmp_path, matrix, expected_steps=2)


def test_materialized_release_is_content_addressed_and_freshly_verifiable(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    matrix = _matrix()
    for trial in matrix.trials:
        _episode(run_root, trial)
    (run_root / "rollout-report.json").write_text("{}\n", encoding="utf-8")
    (run_root / "capacity-report.json").write_text("{}\n", encoding="utf-8")
    plan = build_release_plan(run_root, matrix, expected_steps=2)

    release = materialize_release(
        plan,
        tmp_path / "release",
        matrix_json="{\"schema_version\":1}\n",
        policy_revision=POLICY_REVISION,
        code_revision=CODE_REVISION,
        asset_revision=ASSET_REVISION,
        image_identity=IMAGE_IDENTITY,
    )

    assert release.remote_prefix == f"rollouts/groot-n17-step-12000/{release.release_id}"
    assert release.entry_count == len(verify_release_tree(release.root))
    assert (release.root / "release-manifest.json").is_file()
    assert (release.root / "SHA256SUMS.json").is_file()

    with pytest.raises(ValueError, match="does not match verified episodes"):
        materialize_release(
            plan,
            tmp_path / "wrong-release",
            matrix_json="{\"schema_version\":1}\n",
            policy_revision="f" * 40,
            code_revision=CODE_REVISION,
            asset_revision=ASSET_REVISION,
            image_identity=IMAGE_IDENTITY,
        )

    first = next(path for path in release.root.rglob("*.mp4"))
    first.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="(size|hash) mismatch"):
        verify_release_tree(release.root)


def test_remote_tree_requires_exact_prefixed_paths_and_sizes(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    matrix = PublicMatrix(1, (_matrix().trials[0],), ())
    _episode(run_root, matrix.trials[0])
    (run_root / "rollout-report.json").write_text("{}\n", encoding="utf-8")
    (run_root / "capacity-report.json").write_text("{}\n", encoding="utf-8")
    release = materialize_release(
        build_release_plan(run_root, matrix, expected_steps=2),
        tmp_path / "release",
        matrix_json="{\"schema_version\":1}\n",
        policy_revision=POLICY_REVISION,
        code_revision=CODE_REVISION,
        asset_revision=ASSET_REVISION,
        image_identity=IMAGE_IDENTITY,
    )
    observed = {
        f"unrelated/{entry.relative_path}": entry.byte_size
        for entry in release.entries
    }
    observed.update(
        {
            f"{release.remote_prefix}/{entry.relative_path}": entry.byte_size
            for entry in release.entries
        }
    )

    validate_remote_file_tree(
        observed,
        remote_prefix=release.remote_prefix,
        expected=release.entries,
    )

    observed[f"{release.remote_prefix}/unexpected.bin"] = 1
    with pytest.raises(ValueError, match="closed local allowlist"):
        validate_remote_file_tree(
            observed,
            remote_prefix=release.remote_prefix,
            expected=release.entries,
        )


def test_materialization_revalidates_copied_episode_provenance(tmp_path: Path) -> None:
    matrix = PublicMatrix(1, (_matrix().trials[0],), ())
    _episode(tmp_path, matrix.trials[0])
    (tmp_path / "rollout-report.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "capacity-report.json").write_text("{}\n", encoding="utf-8")
    plan = build_release_plan(tmp_path, matrix, expected_steps=2)

    episode_root = tmp_path / "raw" / matrix.trials[0].trial_id
    episode_path = episode_root / "episode.json"
    payload = json.loads(episode_path.read_text(encoding="utf-8"))
    payload["identity"]["policy_revision"] = "f" * 40
    episode_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    (episode_root / MANIFEST_NAME).unlink()
    atomic_write_json(episode_root / MANIFEST_NAME, build_sha256_manifest(episode_root))

    with pytest.raises(ValueError, match="copied episode provenance changed"):
        materialize_release(
            plan,
            tmp_path / "release",
            matrix_json="{\"schema_version\":1}\n",
            policy_revision=POLICY_REVISION,
            code_revision=CODE_REVISION,
            asset_revision=ASSET_REVISION,
            image_identity=IMAGE_IDENTITY,
        )
