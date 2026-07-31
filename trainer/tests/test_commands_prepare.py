from __future__ import annotations

from pathlib import Path

import pytest

from lehome_train.commands.prepare import prepare_training_environment
from lehome_train.models import ArtifactIdentity
from lehome_train.preflight import MINIMUM_DISK_BYTES, MINIMUM_VRAM_BYTES


def test_prepare_records_timings_and_resumes_completed_stages(tmp_path: Path) -> None:
    calls: list[str] = []
    operations = {
        name: (lambda name=name: calls.append(name))
        for name in (
            "image_runtime_verification",
            "network_measurement",
            "model_download",
            "dataset_download",
            "schema_hash_validation",
            "model_initialization",
        )
    }
    result = prepare_training_environment(
        output_root=tmp_path,
        resolved_config={"dataset_revision": "a" * 40, "model_revision": "b" * 40},
        artifacts=(ArtifactIdentity("meta/stats.json", "c" * 64, 1),),
        visible_devices="0",
        visible_vram_bytes=(MINIMUM_VRAM_BYTES,),
        writable_free_bytes=MINIMUM_DISK_BYTES,
        token="hf_explicit_but_not_persisted",
        hub_permission_check=lambda _token: True,
        stage_operations=operations,
    )

    assert calls == list(operations)
    assert result.experiment.root.is_dir()
    calls.clear()
    resumed = prepare_training_environment(
        output_root=tmp_path,
        resolved_config={"dataset_revision": "a" * 40, "model_revision": "b" * 40},
        artifacts=(ArtifactIdentity("meta/stats.json", "c" * 64, 1),),
        visible_devices="0",
        visible_vram_bytes=(MINIMUM_VRAM_BYTES,),
        writable_free_bytes=MINIMUM_DISK_BYTES,
        token="hf_explicit_but_not_persisted",
        hub_permission_check=lambda _token: True,
        stage_operations=operations,
    )
    assert calls == []
    assert resumed.experiment.resumed is True


def test_prepare_rejects_secret_bearing_resolved_config_before_writing_output(tmp_path: Path) -> None:
    operations = {
        name: (lambda: None)
        for name in (
            "image_runtime_verification",
            "network_measurement",
            "model_download",
            "dataset_download",
            "schema_hash_validation",
            "model_initialization",
        )
    }
    with pytest.raises(ValueError, match="secret"):
        prepare_training_environment(
            output_root=tmp_path,
            resolved_config={
                "dataset_revision": "a" * 40,
                "model_revision": "b" * 40,
                "hub_token": "hf_explicit_but_not_persisted",
            },
            artifacts=(ArtifactIdentity("meta/stats.json", "c" * 64, 1),),
            visible_devices="0",
            visible_vram_bytes=(MINIMUM_VRAM_BYTES,),
            writable_free_bytes=MINIMUM_DISK_BYTES,
            token="hf_explicit_but_not_persisted",
            hub_permission_check=lambda _token: True,
            stage_operations=operations,
        )
    assert not list(tmp_path.iterdir())
