from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from lehome_train.checkpoints import (
    GIBIBYTE,
    AsyncCheckpointUploads,
    CheckpointDescriptor,
    can_continue_without_upload,
    load_checkpoint_descriptor,
    prunable_checkpoints,
    require_compatible_checkpoint,
    retry_checkpoint_upload,
    write_checkpoint_descriptor,
)
from lehome_train.models import ArtifactIdentity, CheckpointRecord


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _checkpoint(
    step: int,
    *,
    batch: int = 64,
    remotely_verified: bool = False,
    normalization_sha256: str = SHA_C,
) -> CheckpointDescriptor:
    return CheckpointDescriptor(
        record=CheckpointRecord(
            experiment_id="experiment-001",
            optimizer_step=step,
            sample_presentations=step * batch,
            experiment_config_sha256=SHA_A,
            dataset_manifest_sha256=SHA_B,
            artifact=ArtifactIdentity(
                relative_path=f"checkpoints/step-{step}.tar.zst",
                sha256=f"{step:064x}",
                byte_size=5 * GIBIBYTE,
            ),
            resumable=True,
            remotely_verified=remotely_verified,
        ),
        normalization_sha256=normalization_sha256,
    )


def test_checkpoint_descriptor_round_trips_and_is_resumable(tmp_path: Path) -> None:
    checkpoint = _checkpoint(1_000, remotely_verified=True)
    manifest = tmp_path / "checkpoint.json"

    write_checkpoint_descriptor(manifest, checkpoint)
    loaded = load_checkpoint_descriptor(manifest)

    assert loaded == checkpoint
    assert require_compatible_checkpoint(
        loaded,
        experiment_id="experiment-001",
        experiment_config_sha256=SHA_A,
        dataset_manifest_sha256=SHA_B,
        normalization_sha256=SHA_C,
        physical_batch_size=64,
        maximum_optimizer_step=12_000,
        checkpoint_interval_steps=1_000,
    ) == checkpoint


def test_checkpoint_descriptor_loader_rejects_type_drift_and_duplicate_fields(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "checkpoint.json"
    payload = _checkpoint(1_000).to_dict()
    payload["normalization_sha256"] = 7
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="normalization"):
        load_checkpoint_descriptor(manifest)

    manifest.write_text(
        '{"schema_version":1,"schema_version":1,"normalization_sha256":"'
        + SHA_C
        + '","record":{}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="malformed"):
        load_checkpoint_descriptor(manifest)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dataset_manifest_sha256", SHA_C, "dataset manifest"),
        ("experiment_config_sha256", SHA_C, "experiment config"),
        ("normalization_sha256", SHA_B, "normalization"),
    ],
)
def test_resume_rejects_incompatible_checkpoint_identity(
    field: str,
    value: str,
    message: str,
) -> None:
    checkpoint = _checkpoint(1_000)
    values = {
        "experiment_id": "experiment-001",
        "experiment_config_sha256": SHA_A,
        "dataset_manifest_sha256": SHA_B,
        "normalization_sha256": SHA_C,
        "physical_batch_size": 64,
        "maximum_optimizer_step": 12_000,
        "checkpoint_interval_steps": 1_000,
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        require_compatible_checkpoint(checkpoint, **values)


def test_resume_requires_verified_resumable_boundary_checkpoint() -> None:
    unverified = _checkpoint(1_000)
    not_resumable = replace(
        unverified,
        record=replace(unverified.record, remotely_verified=True, resumable=False),
    )
    off_boundary = _checkpoint(999, remotely_verified=True)

    for checkpoint, message in (
        (unverified, "remotely verified"),
        (not_resumable, "resumable"),
        (off_boundary, "checkpoint boundary"),
    ):
        with pytest.raises(ValueError, match=message):
            require_compatible_checkpoint(
                checkpoint,
                experiment_id="experiment-001",
                experiment_config_sha256=SHA_A,
                dataset_manifest_sha256=SHA_B,
                normalization_sha256=SHA_C,
                physical_batch_size=64,
                maximum_optimizer_step=12_000,
                checkpoint_interval_steps=1_000,
            )


def test_pruning_preserves_latest_verified_resumable_checkpoint() -> None:
    first = _checkpoint(1_000, remotely_verified=True)
    protected = _checkpoint(2_000, remotely_verified=True)
    newest_unverified = _checkpoint(3_000)

    assert prunable_checkpoints((first, protected, newest_unverified), keep_newest=1) == (
        first,
    )


def test_disk_reserve_holds_two_more_complete_checkpoints_plus_twenty_gib() -> None:
    checkpoint_bytes = 5 * GIBIBYTE
    required = 2 * checkpoint_bytes + 20 * GIBIBYTE

    assert can_continue_without_upload(required, checkpoint_bytes) is True
    assert can_continue_without_upload(required - 1, checkpoint_bytes) is False


def test_upload_retries_exactly_five_times_with_bounded_delays() -> None:
    attempts: list[int] = []
    delays: list[float] = []

    uploaded = retry_checkpoint_upload(
        _checkpoint(1_000),
        uploader=lambda _checkpoint: attempts.append(1) or False,
        sleeper=delays.append,
    )

    assert uploaded is False
    assert len(attempts) == 5
    assert len(delays) == 4
    assert max(delays) <= 1.0


def test_async_upload_marks_only_verified_successes() -> None:
    with AsyncCheckpointUploads(
        uploader=lambda _checkpoint: True,
        sleeper=lambda _delay: None,
    ) as uploads:
        uploads.submit(_checkpoint(1_000))
        completed = uploads.finish()

    assert completed[0].record.remotely_verified is True
