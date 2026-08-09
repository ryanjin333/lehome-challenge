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
from lehome_train.constants import (
    BEHAVIOR_1K_DATASET_REVISION,
    BEHAVIOR_1K_TRAINER_IMAGE_REPOSITORY,
    ISAAC_GROOT_REPOSITORY,
)
from lehome_train.b1k.launch import build_b1k_command
from lehome_train.b1k.training import approved_launch_plans
from lehome_train.io import canonical_json_sha256


ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "trainer"
LOCK_SHA256 = "67fcd520cd75f3b3b383fcc887f244c332af5c2a5548d384d71e0376697b2432"
REPOSITORY_COMMIT = "7d367df39a94917c6c1df6befe011eef1a0ce3ca"
OCI_DIGEST = "sha256:" + "a" * 64
DATASET_REVISION = BEHAVIOR_1K_DATASET_REVISION
DATASET_MANIFEST_SHA256 = "c" * 64
NORMALIZATION_SHA256 = "d" * 64


def _accepted_payload() -> dict[str, object]:
    return {
        "schema_version": 2,
        "status": "accepted",
        "platform": "linux/amd64",
        "cuda_base": {
            "image": "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04",
            "digest": CUDA_BASE_DIGEST,
        },
        "python_version": "3.10.18",
        "uv_version": "0.8.22",
        "isaac_groot": {
            "repository": "https://github.com/wensi-ai/Isaac-GR00T.git",
            "commit": ISAAC_GROOT_REVISION,
        },
        "base_model": {
            "repository": "nvidia/GR00T-N1.7-3B",
            "revision": MODEL_REVISION,
        },
        "cosmos": {
            "repository": "nvidia/Cosmos-Reason2-2B",
            "revision": "9ce19a195e423419c349abfc86fd07178b230561",
        },
        "dataset": {
            "repository": "behavior-1k/2026-challenge-demos",
            "revision": DATASET_REVISION,
            "manifest_sha256": DATASET_MANIFEST_SHA256,
            "normalization_sha256": NORMALIZATION_SHA256,
        },
        "trainer_lock_sha256": LOCK_SHA256,
        "repository_commit": REPOSITORY_COMMIT,
        "image": {
            "repository": BEHAVIOR_1K_TRAINER_IMAGE_REPOSITORY,
            "tag": REPOSITORY_COMMIT,
            "digest": OCI_DIGEST,
        },
        "gpu_acceptance": {
            "status": "passed",
            "run_id": "b1k-acceptance-001",
            "world_size": 1,
            "launch_plan_id": "b1k-gpu1-effective-batch256",
            "effective_global_batch_size": 256,
            "physical_batch_size": 64,
            "global_batch_size": 64,
            "gradient_accumulation_steps": 4,
            "task_manifest_sha256": DATASET_MANIFEST_SHA256,
            "modality_sha256": "ca3a2e406472650bee9439ed81a3f4a1531b6fe689cdc1b348d0f260a208e641",
            "stats_sha256": NORMALIZATION_SHA256,
            "base_model_revision": MODEL_REVISION,
            "cosmos_revision": "9ce19a195e423419c349abfc86fd07178b230561",
            "learning_rate": 1e-4,
            "warmup_ratio": 0.05,
            "launch_arguments": ["torchrun", "--nproc_per_node=1", "--master_port=29500", "/opt/isaac-groot/scripts/b1k/train_b1k.py", "--base-model-path", "/workspace/models/groot", "--dataset-path", "/workspace/data/b1k", "--output-dir", "/workspace/outputs/b1k-acceptance-001", "--experiment-name", "b1k-acceptance-001", "--embodiment-tag", "NEW_EMBODIMENT", "--modality-config-path", "/opt/isaac-groot/examples/b1k/r1pro.py", "--num-gpus", "1", "--global-batch-size", "64", "--gradient-accumulation-steps", "4", "--max-steps", "15000", "--save-steps", "1000", "--save-total-limit", "2", "--learning-rate", "0.0001", "--weight-decay", "1e-05", "--warmup-ratio", "0.05", "--decode-only-used-frames"],
            "launch_arguments_sha256": "3a7f6873ac2f176b6a442b880848756e7e3cbba26ba74d2a4955aa163222d3ea",
            "hardware_model": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "hardware_count": 1,
            "vram_gib": 96,
            "cuda_optimizer_step_passed": True,
            "checkpoint_roundtrip_passed": True,
            "resume_passed": True,
            "tiny_overfit_passed": True,
            "finite_loss": True,
            "acceptance_seconds": 600.0,
            "evidence_uri": "https://github.com/example/actions/runs/1",
        },
    }


def test_accepted_manifest_records_every_immutable_identity() -> None:
    manifest = ReleaseManifest.from_dict(_accepted_payload())

    assert manifest.cuda_base_digest == CUDA_BASE_DIGEST
    assert manifest.isaac_groot_commit == ISAAC_GROOT_REVISION
    assert manifest.payload["isaac_groot"]["repository"] == ISAAC_GROOT_REPOSITORY  # type: ignore[index]
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
        "repository": "behavior-1k/2026-challenge-demos",
        "revision": DATASET_REVISION,
        "manifest_sha256": None,
        "normalization_sha256": None,
    }
    assert manifest.trainer_lock_sha256 == hashlib.sha256(
        (TRAINER / "uv.lock").read_bytes()
    ).hexdigest()


def test_unreleased_example_uses_the_b1k_training_image_identity() -> None:
    payload = json.loads((TRAINER / "release-manifest.example.json").read_text())

    assert payload["image"]["repository"] == BEHAVIOR_1K_TRAINER_IMAGE_REPOSITORY


def test_dirty_build_is_labeled_and_tagged_diagnostic_and_release_verifier_rejects_it() -> None:
    builder = (TRAINER / "scripts" / "build-image.sh").read_text(encoding="utf-8")
    verifier = (TRAINER / "scripts" / "verify-image.sh").read_text(encoding="utf-8")

    assert "-dirty-diagnostic" in builder
    assert 'io.lehome.release-mode=${release_mode}' in builder
    assert 'expected_release_mode=release' in verifier
    assert 'io.lehome.release-mode' in verifier


def test_accepted_manifest_requires_real_digest_and_b1k_identity_gate() -> None:
    for field, value in (
        ("digest", None),
        ("tag", "0" * 40),
    ):
        payload = _accepted_payload()
        payload["image"][field] = value  # type: ignore[index]
        with pytest.raises(ValueError):
            ReleaseManifest.from_dict(payload)

    payload = _accepted_payload()
    payload["gpu_acceptance"]["launch_plan_id"] = "unsafe"  # type: ignore[index]
    with pytest.raises(ValueError, match="launch plan"):
        ReleaseManifest.from_dict(payload)


@pytest.mark.parametrize("world_size", (1, 2, 3, 4))
def test_accepted_manifest_validates_each_approved_gpu_plan(world_size: int) -> None:
    payload = _accepted_payload()
    acceptance = payload["gpu_acceptance"]
    plan = approved_launch_plans(num_gpus=world_size)[0]
    command = build_b1k_command(
        plan,
        checkout="/opt/isaac-groot",
        dataset_path="/workspace/data/b1k",
        base_model_path="/workspace/models/groot",
        output_dir="/workspace/outputs/b1k-acceptance-001",
        experiment_name="b1k-acceptance-001",
        resume_from_checkpoint=False,
    )
    acceptance.update({
        "world_size": world_size,
        "launch_plan_id": plan.identity,
        "effective_global_batch_size": plan.effective_global_batch_size,
        "physical_batch_size": plan.physical_batch_size,
        "global_batch_size": plan.global_batch_size,
        "gradient_accumulation_steps": plan.gradient_accumulation_steps,
        "launch_arguments": list(command),
        "launch_arguments_sha256": canonical_json_sha256(command),
        "hardware_count": world_size,
    })

    assert ReleaseManifest.from_dict(payload).gpu_acceptance_status == "passed"


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
    assert ISAAC_GROOT_REPOSITORY in dockerfile
    for required_file in (
        "scripts/b1k/train_b1k.py",
        "scripts/b1k/deploy_modality.py",
        "examples/b1k/r1pro.py",
        "examples/b1k/r1pro.json",
        "gr00t/data/dataset/lerobot_episode_loader.py",
    ):
        assert required_file in dockerfile
    assert "USER trainer" in dockerfile
    assert "LEHOME_TRAIN_RUNTIME_FACTORY=lehome_train.groot.production_runtime:create" in dockerfile
    for forbidden in ("Isaac Sim", "IsaacLab", "HF_TOKEN="):
        assert forbidden not in dockerfile
    assert "hf auth login" in entrypoint
    assert "huggingface-cli login" in entrypoint
    assert "unset HF_TOKEN" in entrypoint
    for mount in ("/cache", "/prepared", "/output"):
        assert mount in entrypoint


def test_image_uses_parser_level_b1k_cli_verifier() -> None:
    verifier = (TRAINER / "scripts" / "verify-b1k-cli.py").read_text(encoding="utf-8")
    image_verifier = (TRAINER / "scripts" / "verify-image.sh").read_text(encoding="utf-8")
    assert "from gr00t.configs.finetune_config import FinetuneConfig" in verifier
    assert "tyro.cli(FinetuneConfig" in verifier
    assert "build_b1k_command" in verifier
    assert "for world_size in (1, 2, 3, 4)" in verifier
    assert "verify-b1k-cli" in image_verifier


def test_training_readmes_name_the_current_image_verifier_and_gpu_bounds() -> None:
    trainer_readme = (TRAINER / "README.md").read_text(encoding="utf-8")
    launchkit_readme = (TRAINER / "b1k_launchkit" / "README.md").read_text(encoding="utf-8")
    gpu_gate = re.search(
        r"On a Linux NVIDIA host.*?```bash\n(?P<command>.*?)\n```",
        trainer_readme,
        flags=re.DOTALL,
    )

    assert gpu_gate is not None
    assert gpu_gate.group("command") == (
        "REPOSITORY_COMMIT=<40-lowercase-source-commit> \\\n"
        "  CUDA_VISIBLE_DEVICES=0 trainer/scripts/verify-image.sh --gpu \\\n"
        "  docker.io/ryanjin333/behavior1k-groot-n17-trainer@sha256:<64-lowercase-hex>"
    )
    assert "REPOSITORY_COMMIT=<40-character-source-commit>" in trainer_readme
    assert "ace36d935b376fbf25cd56371e23877b95407c40" in trainer_readme
    assert "ghcr.io" not in trainer_readme.lower()
    assert "1–4 RTX PRO 6000" in launchkit_readme


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


def test_image_final_user_supports_vast_ssh_overlay() -> None:
    dockerfile = (TRAINER / "Dockerfile").read_text(encoding="utf-8")
    verifier = (TRAINER / "scripts" / "verify-image.sh").read_text(encoding="utf-8")

    assert "--create-home --home-dir /home/trainer --shell /bin/bash trainer" in dockerfile
    assert "HOME=/home/trainer" in dockerfile
    assert "HOME=/nonexistent" not in dockerfile
    assert 'test "$HOME" = /home/trainer' in verifier
    assert 'test -w "$HOME"' in verifier
    assert 'test -w "$HOME/.bashrc"' in verifier
    assert 'printf "\\n" >> "$HOME/.bashrc"' in verifier


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

    assert "IMAGE_REPOSITORY: docker.io/ryanjin333/behavior1k-groot-n17-trainer" in workflow
    assert "secrets.DOCKERHUB_USERNAME" in workflow
    assert "secrets.DOCKERHUB_TOKEN" in workflow
    assert "packages: write" not in workflow
    assert "github.token" not in workflow
    assert "ghcr.io" not in workflow.lower()
    assert "secrets.PAT" not in workflow
    assert "self-hosted" in workflow
    assert "--gpu" in workflow
    assert "docker/metadata-action" in workflow
    assert "type=raw,value=${{ github.sha }}" in workflow
    assert "verify-image.sh \"${IMAGE_REPOSITORY}@${OCI_DIGEST}\"" in workflow
    assert "digest: ${{ steps.build.outputs.digest }}" in workflow
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
