import json
from pathlib import Path

import pytest

from lehome_train.release_manifest import (
    CUDA_BASE_DIGEST,
    ISAAC_GROOT_REVISION,
    MODEL_REVISION,
    ReleaseManifest,
    load_release_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "trainer"
LOCK_SHA256 = "08cc37e719678d7d0347b45f617e1dccc6e8d6513da501237ef541982cadf6fa"
REPOSITORY_COMMIT = "7c150162faf3ec285960f59cc72a6e5643e9d711"
OCI_DIGEST = "sha256:" + "a" * 64


def _accepted_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "accepted",
        "platform": "linux/amd64",
        "cuda_base": {
            "image": "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04",
            "digest": CUDA_BASE_DIGEST,
        },
        "python_version": "3.10.18",
        "uv_version": "0.8.22",
        "isaac_groot": {
            "repository": "https://github.com/NVIDIA/Isaac-GR00T.git",
            "commit": ISAAC_GROOT_REVISION,
        },
        "base_model": {
            "repository": "nvidia/GR00T-N1.7-3B",
            "revision": MODEL_REVISION,
        },
        "trainer_lock_sha256": LOCK_SHA256,
        "repository_commit": REPOSITORY_COMMIT,
        "image": {
            "repository": "ghcr.io/ryanjin333/lehome-groot-n17-trainer",
            "tag": REPOSITORY_COMMIT,
            "digest": OCI_DIGEST,
        },
        "gpu_acceptance": {
            "status": "passed",
            "hardware": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "network_gbps": 1.0,
            "image_pull_seconds": 120.0,
            "first_optimizer_step_seconds": 900.0,
            "memorization_passed": True,
            "batches_tested_sequentially": [16, 32, 64],
            "training_768k_started_or_resumed": True,
            "evidence_uri": "https://github.com/example/actions/runs/1",
        },
    }


def test_accepted_manifest_records_every_immutable_identity() -> None:
    manifest = ReleaseManifest.from_dict(_accepted_payload())

    assert manifest.cuda_base_digest == CUDA_BASE_DIGEST
    assert manifest.isaac_groot_commit == ISAAC_GROOT_REVISION
    assert manifest.model_revision == MODEL_REVISION
    assert manifest.trainer_lock_sha256 == LOCK_SHA256
    assert manifest.repository_commit == REPOSITORY_COMMIT
    assert manifest.oci_digest == OCI_DIGEST


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("cuda_base", "digest"), "sha256:" + "b" * 64),
        (("isaac_groot", "commit"), "main"),
        (("base_model", "revision"), "latest"),
        (("trainer_lock_sha256",), "not-a-hash"),
        (("repository_commit",), "main"),
        (("image", "tag"), "latest"),
        (("image", "digest"), "not-a-digest"),
    ],
)
def test_manifest_rejects_changed_or_mutable_identities(
    path: tuple[str, ...], value: object
) -> None:
    payload = _accepted_payload()
    target = payload
    for key in path[:-1]:
        target = target[key]  # type: ignore[index,assignment]
    target[path[-1]] = value

    with pytest.raises(ValueError):
        ReleaseManifest.from_dict(payload)


def test_manifest_rejects_unknown_and_duplicate_fields(tmp_path: Path) -> None:
    payload = _accepted_payload()
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="schema"):
        ReleaseManifest.from_dict(payload)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_release_manifest(duplicate)


def test_unreleased_example_has_pins_but_no_claimed_image_or_gpu_acceptance() -> None:
    manifest = load_release_manifest(TRAINER / "release-manifest.example.json")

    assert manifest.status == "unreleased"
    assert manifest.cuda_base_digest == CUDA_BASE_DIGEST
    assert manifest.isaac_groot_commit == ISAAC_GROOT_REVISION
    assert manifest.model_revision == MODEL_REVISION
    assert manifest.trainer_lock_sha256 == LOCK_SHA256
    assert manifest.oci_digest is None
    assert manifest.gpu_acceptance_status == "pending"


def test_accepted_manifest_requires_real_digest_and_completed_pro6000_gate() -> None:
    for field, value in (
        ("digest", None),
        ("tag", "0" * 40),
    ):
        payload = _accepted_payload()
        payload["image"][field] = value  # type: ignore[index]
        with pytest.raises(ValueError):
            ReleaseManifest.from_dict(payload)

    payload = _accepted_payload()
    payload["gpu_acceptance"]["first_optimizer_step_seconds"] = 1800.1  # type: ignore[index]
    with pytest.raises(ValueError, match="30 minutes"):
        ReleaseManifest.from_dict(payload)


def test_image_contract_files_preserve_runtime_and_secret_boundaries() -> None:
    dockerfile = (TRAINER / "Dockerfile").read_text(encoding="utf-8")
    entrypoint = (TRAINER / "docker" / "entrypoint.sh").read_text(encoding="utf-8")

    assert "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04@" + CUDA_BASE_DIGEST in dockerfile
    assert "uv sync --frozen" in dockerfile
    assert ISAAC_GROOT_REVISION in dockerfile
    assert "USER trainer" in dockerfile
    assert "LEHOME_TRAIN_RUNTIME_FACTORY=lehome_train.groot.production_runtime:create" in dockerfile
    for forbidden in ("Isaac Sim", "IsaacLab", "HF_TOKEN="):
        assert forbidden not in dockerfile
    assert "hf auth login" in entrypoint
    assert "huggingface-cli login" in entrypoint
    assert "unset HF_TOKEN" in entrypoint
    for mount in ("/cache", "/prepared", "/output"):
        assert mount in entrypoint


def test_workflow_never_embeds_a_pat_and_gpu_gate_is_explicit() -> None:
    workflow = (ROOT / ".github" / "workflows" / "groot-trainer-image.yml").read_text(
        encoding="utf-8"
    )

    assert "packages: write" in workflow
    assert "password: ${{ github.token }}" in workflow
    assert "secrets.PAT" not in workflow
    assert "self-hosted" in workflow
    assert "--gpu" in workflow
    assert "docker/metadata-action" in workflow
    assert "type=raw,value=${{ github.sha }}" in workflow


def test_image_verifier_makes_ephemeral_bind_mounts_writable_by_container_user() -> None:
    verifier = (TRAINER / "scripts" / "verify-image.sh").read_text(encoding="utf-8")

    assert 'chmod 0777 "$mount_root/cache" "$mount_root/prepared" "$mount_root/output"' in verifier


def test_production_runtime_factory_is_not_a_successful_noop() -> None:
    from lehome_train.groot.production_runtime import create

    adapter = create()
    for command in ("prepare", "memorize", "smoke", "train"):
        with pytest.raises(ValueError, match="schema"):
            getattr(adapter, command)({})
