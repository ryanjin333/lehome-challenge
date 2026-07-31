from __future__ import annotations

from pathlib import Path
import json

import pytest

from lehome_train.preflight import (
    MINIMUM_DISK_BYTES,
    MINIMUM_VRAM_BYTES,
    PreflightStage,
    HubPermission,
    HubTarget,
    check_hardware,
    run_timed_stages,
    verify_hub_upload_readback_permission,
    verify_immutable_revision,
    verify_snapshot_manifest,
)
from lehome_train.io import sha256_file
from lehome_train.models import ArtifactIdentity


def test_hardware_requires_one_visible_40gb_gpu_and_200gb_writable_disk() -> None:
    report = check_hardware(
        visible_devices="1",
        visible_vram_bytes=(MINIMUM_VRAM_BYTES,),
        writable_free_bytes=MINIMUM_DISK_BYTES,
    )

    assert report.visible_device == "1"
    assert report.vram_bytes == MINIMUM_VRAM_BYTES
    assert report.writable_free_bytes == MINIMUM_DISK_BYTES

    with pytest.raises(ValueError, match="40 GiB"):
        check_hardware(
            visible_devices="1",
            visible_vram_bytes=(MINIMUM_VRAM_BYTES - 1,),
            writable_free_bytes=MINIMUM_DISK_BYTES,
        )
    with pytest.raises(ValueError, match="exactly one visible GPU"):
        check_hardware(
            visible_devices="0,1",
            visible_vram_bytes=(MINIMUM_VRAM_BYTES, MINIMUM_VRAM_BYTES),
            writable_free_bytes=MINIMUM_DISK_BYTES,
        )
    with pytest.raises(ValueError, match="200 GiB"):
        check_hardware(
            visible_devices="1",
            visible_vram_bytes=(MINIMUM_VRAM_BYTES,),
            writable_free_bytes=MINIMUM_DISK_BYTES - 1,
        )


def test_revision_snapshot_and_hub_permission_are_explicit_and_secret_safe(tmp_path: Path) -> None:
    revision = "a" * 40
    verify_immutable_revision(
        expected_revision=revision,
        observed_revision=revision,
        label="dataset",
    )
    with pytest.raises(ValueError, match="dataset revision"):
        verify_immutable_revision(
            expected_revision=revision,
            observed_revision="b" * 40,
            label="dataset",
        )

    snapshot = tmp_path / "model"
    snapshot.mkdir()
    payload = snapshot / "weights.bin"
    payload.write_bytes(b"model")
    identity = ArtifactIdentity("weights.bin", sha256_file(payload), payload.stat().st_size)
    manifest = snapshot / "snapshot.json"
    manifest.write_text(
        json.dumps({"revision": revision, "artifacts": [identity.to_dict()]}),
        encoding="utf-8",
    )
    verified = verify_snapshot_manifest(
        snapshot_root=snapshot,
        manifest_path=manifest,
        expected_revision=revision,
        label="model",
    )
    assert verified.revision == revision
    assert verified.manifest_sha256 == sha256_file(manifest)

    observed_requests: list[tuple[str, str, str]] = []
    verify_hub_upload_readback_permission(
        token="hf_secret_token_value",
        target=HubTarget("ryanjin333/lehome-groot-n17-models", revision),
        permission_check=lambda token, repository, selected_revision: (
            observed_requests.append((token, repository, selected_revision))
            or HubPermission(can_upload=True, can_readback=True, private_repository=True)
        ),
    )
    assert observed_requests == [
        ("hf_secret_token_value", "ryanjin333/lehome-groot-n17-models", revision)
    ]

    with pytest.raises(ValueError) as error:
        verify_hub_upload_readback_permission(
            token="hf_secret_token_value",
            target=HubTarget("ryanjin333/lehome-groot-n17-models", revision),
            permission_check=lambda *_args: (_ for _ in ()).throw(
                RuntimeError("hf_secret_token_value must not leak")
            ),
        )
    assert "hf_secret_token_value" not in str(error.value)


def test_timed_preflight_stages_have_complete_contract_and_duration() -> None:
    observed: list[str] = []
    stages = tuple(
        PreflightStage(name, lambda name=name: observed.append(name))
        for name in (
            "image_runtime_verification",
            "network_measurement",
            "model_download",
            "dataset_download",
            "schema_hash_validation",
            "model_initialization",
        )
    )

    results = run_timed_stages(stages)

    assert observed == [stage.name for stage in stages]
    assert [result.name for result in results] == [stage.name for stage in stages]
    assert all(result.duration_seconds >= 0 for result in results)
    with pytest.raises(ValueError, match="preflight stages"):
        run_timed_stages(stages[:-1])
