import hashlib
import json
from pathlib import Path
import re

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
LOCK_SHA256 = "67fcd520cd75f3b3b383fcc887f244c332af5c2a5548d384d71e0376697b2432"
REPOSITORY_COMMIT = "7d367df39a94917c6c1df6befe011eef1a0ce3ca"
OCI_DIGEST = "sha256:" + "a" * 64
DATASET_REVISION = "b" * 40
DATASET_MANIFEST_SHA256 = "c" * 64
NORMALIZATION_SHA256 = "d" * 64


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
        "dataset": {
            "repository": "ryanjin333/lehome-groot-n17-data",
            "revision": DATASET_REVISION,
            "manifest_sha256": DATASET_MANIFEST_SHA256,
            "normalization_sha256": NORMALIZATION_SHA256,
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
    assert manifest.dataset_revision == DATASET_REVISION
    assert manifest.dataset_manifest_sha256 == DATASET_MANIFEST_SHA256
    assert manifest.normalization_sha256 == NORMALIZATION_SHA256


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
        (("dataset", "repository"), "somewhere/else"),
        (("dataset", "revision"), "main"),
        (("dataset", "manifest_sha256"), "not-a-hash"),
        (("dataset", "normalization_sha256"), "not-a-hash"),
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


def test_manifest_rejects_a_well_formed_hash_for_a_different_trainer_lock() -> None:
    payload = _accepted_payload()
    payload["trainer_lock_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="approved trainer lock"):
        ReleaseManifest.from_dict(payload)


def test_unreleased_example_has_pins_but_no_claimed_image_or_gpu_acceptance() -> None:
    manifest = load_release_manifest(TRAINER / "release-manifest.example.json")

    assert manifest.status == "unreleased"
    assert manifest.cuda_base_digest == CUDA_BASE_DIGEST
    assert manifest.isaac_groot_commit == ISAAC_GROOT_REVISION
    assert manifest.model_revision == MODEL_REVISION
    assert manifest.trainer_lock_sha256 == LOCK_SHA256
    assert manifest.oci_digest is None
    assert manifest.gpu_acceptance_status == "pending"
    assert manifest.repository_commit is None
    assert manifest.payload["image"]["tag"] is None  # type: ignore[index]
    assert manifest.payload["dataset"] == {
        "repository": "ryanjin333/lehome-groot-n17-data",
        "revision": None,
        "manifest_sha256": None,
        "normalization_sha256": None,
    }
    assert manifest.trainer_lock_sha256 == hashlib.sha256(
        (TRAINER / "uv.lock").read_bytes()
    ).hexdigest()


def test_dirty_build_is_labeled_and_tagged_diagnostic_and_release_verifier_rejects_it() -> None:
    builder = (TRAINER / "scripts" / "build-image.sh").read_text(encoding="utf-8")
    verifier = (TRAINER / "scripts" / "verify-image.sh").read_text(encoding="utf-8")

    assert "-dirty-diagnostic" in builder
    assert 'io.lehome.release-mode=${release_mode}' in builder
    assert 'expected_release_mode=release' in verifier
    assert 'io.lehome.release-mode' in verifier


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

    payload = _accepted_payload()
    payload["gpu_acceptance"]["image_pull_seconds"] = None  # type: ignore[index]
    with pytest.raises(ValueError, match="pull timing"):
        ReleaseManifest.from_dict(payload)


@pytest.mark.parametrize(
    "uri",
    [
        "javascript:alert(1)",
        "http://github.com/example/actions/runs/1",
        "https://user:secret@github.com/example/actions/runs/1",
        "https://localhost/evidence",
        "https://github.com/example/actions/runs/1?token=secret",
        "https://github.com/example/actions/runs/1#token",
    ],
)
def test_accepted_manifest_requires_safe_https_evidence_uri(uri: str) -> None:
    payload = _accepted_payload()
    payload["gpu_acceptance"]["evidence_uri"] = uri  # type: ignore[index]

    with pytest.raises(ValueError, match="evidence URI"):
        ReleaseManifest.from_dict(payload)


def test_image_contract_files_preserve_runtime_and_secret_boundaries() -> None:
    dockerfile = (TRAINER / "Dockerfile").read_text(encoding="utf-8")
    entrypoint = (TRAINER / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
    pyproject = (TRAINER / "pyproject.toml").read_text(encoding="utf-8")

    assert "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04@" + CUDA_BASE_DIGEST in dockerfile
    assert "ARG UBUNTU_SNAPSHOT=20260701T000000Z" in dockerfile
    assert (
        "https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/"
        in dockerfile
    )
    assert "Dir::Etc::sourcelist=/tmp/ubuntu-snapshot.list" in dockerfile
    assert "Dir::Etc::sourceparts=-" in dockerfile
    assert "Dir::State::lists=/tmp/apt-lists" in dockerfile
    assert "uv sync --frozen" in dockerfile
    assert "uv export --frozen --no-dev --no-emit-project" in dockerfile
    assert "uv pip check --python /opt/runtime/bin/python" in dockerfile
    assert "git config --system --add safe.directory /opt/isaac-groot" in dockerfile
    assert '"huggingface-hub==0.36.2"' in pyproject
    assert '"click==8.1.8"' in pyproject
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


def test_entrypoint_retains_token_only_for_controller_owned_remote_commands() -> None:
    entrypoint = (TRAINER / "docker" / "entrypoint.sh").read_text(encoding="utf-8")

    assert 'case "${2:-}:${3:-}" in' in entrypoint
    assert (
        "prepare:*|train:*|restore:*|sync:*|data:publish|data:retrieve|model:retrieve)"
        in entrypoint
    )
    assert "remote=true" in entrypoint
    assert "unset HF_TOKEN" in entrypoint
    assert "hf auth login" in entrypoint
    assert "huggingface-cli login" in entrypoint


def test_trainer_lock_aligns_every_shared_runtime_package_with_upstream() -> None:
    lock = (TRAINER / "uv.lock").read_text(encoding="utf-8")
    versions = dict(
        re.findall(r'\[\[package\]\]\nname = "([^"]+)"\nversion = "([^"]+)"', lock)
    )
    expected = {
        "certifi": "2026.2.25",
        "charset-normalizer": "3.4.6",
        "click": "8.1.8",
        "filelock": "3.25.2",
        "fsspec": "2025.3.0",
        "hf-xet": "1.4.2",
        "huggingface-hub": "0.36.2",
        "idna": "3.11",
        "markdown-it-py": "4.0.0",
        "packaging": "26.0",
        "pyarrow": "23.0.1",
        "pygments": "2.19.2",
        "requests": "2.32.5",
        "rich": "14.3.3",
        "tomli": "2.4.0",
        "tqdm": "4.67.3",
        "typing-extensions": "4.15.0",
        "urllib3": "2.6.3",
    }

    assert {name: versions[name] for name in expected} == expected


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
    assert "verify-image.sh \"${IMAGE_REPOSITORY}@${OCI_DIGEST}\"" in workflow
    assert "scanner_status=$?" in workflow
    assert "scanner failed with status" in workflow


def test_cpu_safe_workflow_disables_ansi_help_output() -> None:
    workflow = (ROOT / ".github" / "workflows" / "groot-trainer-image.yml").read_text(
        encoding="utf-8"
    )

    assert 'NO_COLOR: "1"' in workflow


def test_cpu_safe_workflow_installs_ffmpeg_for_converter_tests() -> None:
    workflow = (ROOT / ".github" / "workflows" / "groot-trainer-image.yml").read_text(
        encoding="utf-8"
    )

    assert "apt-get install -y --no-install-recommends ffmpeg" in workflow


def test_cpu_safe_workflow_excludes_linux_image_acceptance() -> None:
    workflow = (ROOT / ".github" / "workflows" / "groot-trainer-image.yml").read_text(
        encoding="utf-8"
    )

    assert "--ignore=tests/test_groot_runtime_gate.py" in workflow


def test_gpu_verifier_keeps_stdin_open_selects_gpu_zero_and_checks_sentinel() -> None:
    verifier = (TRAINER / "scripts" / "verify-image.sh").read_text(encoding="utf-8")

    assert '"${run[@]}" -i --gpus device=0' in verifier
    assert "CUDA_VISIBLE_DEVICES=0" in verifier
    assert "GPU_SENTINEL:optimizer-step-complete" in verifier
    assert "grep -Fxq 'GPU_SENTINEL:optimizer-step-complete'" in verifier
    assert "secret_status=$?" in verifier
    assert '[[ "$secret_status" -eq 1 ]]' in verifier


def test_docker_context_is_default_deny() -> None:
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()

    assert dockerignore[0] == "**"
    assert "!trainer/pyproject.toml" in dockerignore
    assert "!trainer/uv.lock" in dockerignore
    assert "!trainer/src/**" in dockerignore
    assert "!trainer/config/**" in dockerignore
    assert "!trainer/docker/entrypoint.sh" in dockerignore


def test_image_verifier_makes_ephemeral_bind_mounts_writable_by_container_user() -> None:
    verifier = (TRAINER / "scripts" / "verify-image.sh").read_text(encoding="utf-8")

    assert 'chmod 0777 "$mount_root/cache" "$mount_root/prepared" "$mount_root/output"' in verifier


def test_image_grants_trainer_only_the_git_lfs_scratch_directory() -> None:
    dockerfile = (TRAINER / "Dockerfile").read_text(encoding="utf-8")

    assert "GIT_LFS_SKIP_SMUDGE=1 git checkout --detach FETCH_HEAD" in dockerfile
    assert (
        "install -d -o trainer -g trainer -m 0750 "
        "/opt/isaac-groot/.git/lfs/tmp"
    ) in dockerfile


def test_image_verifier_restores_bind_mount_permissions_before_cleanup() -> None:
    verifier = (TRAINER / "scripts" / "verify-image.sh").read_text(encoding="utf-8")

    assert "cleanup_mount_root()" in verifier
    assert 'trap cleanup_mount_root EXIT' in verifier
    assert '--user 0:0' in verifier
    assert '--entrypoint /bin/chmod "$image_ref"' in verifier
    assert '-R a+rwX /cache /prepared /output' in verifier


def test_image_prunes_upstream_artifacts_and_verifier_requires_none() -> None:
    dockerfile = (TRAINER / "Dockerfile").read_text(encoding="utf-8")
    verifier = (TRAINER / "scripts" / "verify-image.sh").read_text(encoding="utf-8")

    assert "git ls-files -z" in dockerfile
    assert "git update-index --skip-worktree" in dockerfile
    assert "xargs -0 rm -f --" in dockerfile
    assert 'find /opt/trainer /opt/isaac-groot -type f' in verifier
    assert '-iname "*.gif"' in verifier
    assert '-iname "*.whl"' in verifier
    assert 'test -z "$bundled_artifact"' in verifier
    assert "version https://git-lfs.github.com/spec/v1" not in verifier


def test_production_runtime_factory_is_not_a_successful_noop() -> None:
    from lehome_train.groot.production_runtime import create

    adapter = create()
    for command in ("prepare", "memorize", "smoke", "train"):
        with pytest.raises(ValueError, match="schema"):
            getattr(adapter, command)({})
