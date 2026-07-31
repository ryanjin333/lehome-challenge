from __future__ import annotations

from pathlib import Path
import json

import pytest

from lehome_train.commands.prepare import prepare_training_environment
from lehome_train.io import sha256_file
from lehome_train.models import ArtifactIdentity
from lehome_train.preflight import (
    HubPermission,
    HubTarget,
    MINIMUM_DISK_BYTES,
    MINIMUM_VRAM_BYTES,
)


def snapshot(tmp_path: Path, name: str, revision: str) -> tuple[Path, Path]:
    root = tmp_path / name
    root.mkdir()
    payload = root / "payload.bin"
    payload.write_bytes(name.encode("utf-8"))
    artifact = ArtifactIdentity("payload.bin", sha256_file(payload), payload.stat().st_size)
    manifest = root / "snapshot.json"
    manifest.write_text(
        json.dumps({"revision": revision, "artifacts": [artifact.to_dict()]}),
        encoding="utf-8",
    )
    return root, manifest


def test_prepare_records_timings_and_resumes_completed_stages(tmp_path: Path) -> None:
    calls: list[str] = []
    operations = {
        name: (
            lambda root, name=name: (
                calls.append(name),
                _write_stage_record(root, name),
            )[1]
        )
        for name in (
            "image_runtime_verification",
            "network_measurement",
            "model_download",
            "dataset_download",
            "schema_hash_validation",
            "model_initialization",
        )
    }
    model_root, model_manifest = snapshot(tmp_path, "model", "b" * 40)
    dataset_root, dataset_manifest = snapshot(tmp_path, "dataset", "a" * 40)
    target = HubTarget("ryanjin333/lehome-groot-n17-models", "b" * 40)
    result = prepare_training_environment(
        output_root=tmp_path / "output",
        resolved_config={
            "dataset_revision": "a" * 40,
            "model_revision": "b" * 40,
            "dataset_repository": "ryanjin333/lehome-groot-n17-data",
            "model_repository": "ryanjin333/lehome-groot-n17-models",
        },
        artifacts=(ArtifactIdentity("meta/stats.json", "c" * 64, 1),),
        visible_devices="0",
        visible_vram_bytes=(MINIMUM_VRAM_BYTES,),
        writable_free_bytes=MINIMUM_DISK_BYTES,
        token="hf_explicit_but_not_persisted",
        hub_targets=(target,),
        hub_permission_check=lambda *_args: HubPermission(True, True, True),
        stage_operations=operations,
        model_snapshot_root=model_root,
        model_snapshot_manifest=model_manifest,
        dataset_snapshot_root=dataset_root,
        dataset_snapshot_manifest=dataset_manifest,
    )

    assert calls == list(operations)
    assert result.experiment.root.is_dir()
    calls.clear()
    resumed = prepare_training_environment(
        output_root=tmp_path / "output",
        resolved_config={
            "dataset_revision": "a" * 40,
            "model_revision": "b" * 40,
            "dataset_repository": "ryanjin333/lehome-groot-n17-data",
            "model_repository": "ryanjin333/lehome-groot-n17-models",
        },
        artifacts=(ArtifactIdentity("meta/stats.json", "c" * 64, 1),),
        visible_devices="0",
        visible_vram_bytes=(MINIMUM_VRAM_BYTES,),
        writable_free_bytes=MINIMUM_DISK_BYTES,
        token="hf_explicit_but_not_persisted",
        hub_targets=(target,),
        hub_permission_check=lambda *_args: HubPermission(True, True, True),
        stage_operations=operations,
        model_snapshot_root=model_root,
        model_snapshot_manifest=model_manifest,
        dataset_snapshot_root=dataset_root,
        dataset_snapshot_manifest=dataset_manifest,
    )
    assert calls == []
    assert resumed.experiment.resumed is True


def test_prepare_rejects_secret_bearing_resolved_config_before_writing_output(tmp_path: Path) -> None:
    operations = {
        name: (lambda _root: ())
        for name in (
            "image_runtime_verification",
            "network_measurement",
            "model_download",
            "dataset_download",
            "schema_hash_validation",
            "model_initialization",
        )
    }
    model_root, model_manifest = snapshot(tmp_path, "model", "b" * 40)
    dataset_root, dataset_manifest = snapshot(tmp_path, "dataset", "a" * 40)
    with pytest.raises(ValueError, match="secret"):
        prepare_training_environment(
            output_root=tmp_path / "output",
            resolved_config={
                "dataset_revision": "a" * 40,
                "model_revision": "b" * 40,
                "dataset_repository": "ryanjin333/lehome-groot-n17-data",
                "model_repository": "ryanjin333/lehome-groot-n17-models",
                "hub_token": "hf_explicit_but_not_persisted",
            },
            artifacts=(ArtifactIdentity("meta/stats.json", "c" * 64, 1),),
            visible_devices="0",
            visible_vram_bytes=(MINIMUM_VRAM_BYTES,),
            writable_free_bytes=MINIMUM_DISK_BYTES,
            token="hf_explicit_but_not_persisted",
            hub_targets=(HubTarget("ryanjin333/lehome-groot-n17-models", "b" * 40),),
            hub_permission_check=lambda *_args: HubPermission(True, True, True),
            stage_operations=operations,
            model_snapshot_root=model_root,
            model_snapshot_manifest=model_manifest,
            dataset_snapshot_root=dataset_root,
            dataset_snapshot_manifest=dataset_manifest,
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "extra",
    [
        {"release_note": "contains hf_" + "a" * 34},
        {"metadata": {"ordinary_description": "contains hf_" + "b" * 34}},
    ],
)
def test_prepare_rejects_token_text_at_ordinary_nested_config_keys_before_output(
    tmp_path: Path,
    extra: dict[str, object],
) -> None:
    model_root, model_manifest = snapshot(tmp_path, "model", "b" * 40)
    dataset_root, dataset_manifest = snapshot(tmp_path, "dataset", "a" * 40)
    config = {
        "dataset_revision": "a" * 40,
        "model_revision": "b" * 40,
        "dataset_repository": "ryanjin333/lehome-groot-n17-data",
        "model_repository": "ryanjin333/lehome-groot-n17-models",
    }
    config.update(extra)
    operations = {name: (lambda _root: ()) for name in (
        "image_runtime_verification",
        "network_measurement",
        "model_download",
        "dataset_download",
        "schema_hash_validation",
        "model_initialization",
    )}

    with pytest.raises(ValueError, match="secret"):
        prepare_training_environment(
            output_root=tmp_path / "output",
            resolved_config=config,
            artifacts=(ArtifactIdentity("meta/stats.json", "c" * 64, 1),),
            visible_devices="0",
            visible_vram_bytes=(MINIMUM_VRAM_BYTES,),
            writable_free_bytes=MINIMUM_DISK_BYTES,
            token="hf_explicit_but_not_persisted",
            hub_targets=(HubTarget("ryanjin333/lehome-groot-n17-models", "b" * 40),),
            hub_permission_check=lambda *_args: HubPermission(True, True, True),
            stage_operations=operations,
            model_snapshot_root=model_root,
            model_snapshot_manifest=model_manifest,
            dataset_snapshot_root=dataset_root,
            dataset_snapshot_manifest=dataset_manifest,
        )
    assert not (tmp_path / "output").exists()


def test_prepare_records_sanitized_failed_stage_before_raising(tmp_path: Path) -> None:
    model_root, model_manifest = snapshot(tmp_path, "model", "b" * 40)
    dataset_root, dataset_manifest = snapshot(tmp_path, "dataset", "a" * 40)
    operations = {
        "image_runtime_verification": lambda _root: (_ for _ in ()).throw(
            RuntimeError("hf_explicit_but_not_persisted")
        ),
        "network_measurement": lambda _root: (),
        "model_download": lambda _root: (),
        "dataset_download": lambda _root: (),
        "schema_hash_validation": lambda _root: (),
        "model_initialization": lambda _root: (),
    }
    with pytest.raises(RuntimeError, match="failed safely") as error:
        prepare_training_environment(
            output_root=tmp_path / "output",
            resolved_config={
                "dataset_revision": "a" * 40,
                "model_revision": "b" * 40,
                "dataset_repository": "ryanjin333/lehome-groot-n17-data",
                "model_repository": "ryanjin333/lehome-groot-n17-models",
            },
            artifacts=(ArtifactIdentity("meta/stats.json", "c" * 64, 1),),
            visible_devices="0",
            visible_vram_bytes=(MINIMUM_VRAM_BYTES,),
            writable_free_bytes=MINIMUM_DISK_BYTES,
            token="hf_explicit_but_not_persisted",
            hub_targets=(HubTarget("ryanjin333/lehome-groot-n17-models", "b" * 40),),
            hub_permission_check=lambda *_args: HubPermission(True, True, True),
            stage_operations=operations,
            model_snapshot_root=model_root,
            model_snapshot_manifest=model_manifest,
            dataset_snapshot_root=dataset_root,
            dataset_snapshot_manifest=dataset_manifest,
        )
    assert "hf_explicit_but_not_persisted" not in str(error.value)
    experiment_root = next((tmp_path / "output").iterdir())
    status = json.loads((experiment_root / "status.json").read_text(encoding="utf-8"))
    assert status["stages"]["image_runtime_verification"]["state"] == "failed"
    assert "hf_explicit_but_not_persisted" not in (experiment_root / "logs" / "prepare.log").read_text(encoding="utf-8")


def _write_stage_record(root: Path, stage: str) -> tuple[ArtifactIdentity, ...]:
    directory = root / "stage-records"
    directory.mkdir(exist_ok=True)
    path = directory / f"{stage}.json"
    path.write_text('{"ok":true}', encoding="utf-8")
    return (
        ArtifactIdentity(
            f"stage-records/{stage}.json", sha256_file(path), path.stat().st_size
        ),
    )
