"""Render the private, digest-pinned Vast schema for B1K rollout campaigns."""

from __future__ import annotations

import json
from collections.abc import Mapping

from b1k_rollout.identity import BEHAVIOR_REVISION, DATASET_REPO, GROOT_REVISION, MODEL_REPO, require_image_digest, require_immutable_commit, require_sha256
from b1k_rollout.task_manifest import CANONICAL_MANIFEST_SHA256


ROLLOUT_IMAGE_REPOSITORY = "docker.io/ryanjin333/behavior1k-groot-n17"
ROLLOUT_ONSTART = (
    "install -d -o 10001 -g 10001 -m 0700 /workspace /workspace/campaign /workspace/checkpoint-source "
    "/workspace/omnigibson-data /workspace/smoke-canary && "
    "bash /usr/local/bin/b1k-rollout-entrypoint"
)
CAMPAIGN_ID = "b1k-r1pro-public-test-100x10"
_FORBIDDEN = ("novnc", "x11", "xfce", "jupyter", "desktop", "hf_token=", "password=")


def render_vast_template(*, image_digest: str, model_commit: str, checkpoint_artifact_sha256: str, gpu_ids: tuple[int, ...]) -> str:
    """Render the checked-in, credential-free Vast template deterministically."""

    try:
        require_image_digest(image_digest)
    except ValueError as error:
        raise ValueError("template image must use an immutable rollout digest") from error
    require_immutable_commit(model_commit, label="model commit")
    require_sha256(checkpoint_artifact_sha256, label="checkpoint artifact")
    if model_commit == "0" * 40 or checkpoint_artifact_sha256 == "0" * 64:
        raise ValueError("production template must not use zero checkpoint identities")
    if not 1 <= len(gpu_ids) <= 4 or any(type(item) is not int or item < 0 for item in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("template must use one through four explicit GPU IDs")
    template = _template(image_digest, model_commit, checkpoint_artifact_sha256, gpu_ids)
    validate_vast_template(template)
    return json.dumps(template, indent=2, sort_keys=True) + "\n"


def validate_vast_template(value: Mapping[str, object]) -> None:
    """Reject mutable, secret-bearing, or GUI-enabled rollout templates."""

    template = dict(value)
    required = {
        "env",
        "extra_filters",
        "image",
        "jup_direct",
        "name",
        "onstart",
        "private",
        "recommended_disk_space",
        "runtype",
        "ssh_direct",
        "use_ssh",
    }
    if set(template) != required:
        raise ValueError("template fields are invalid")
    image = template["image"]
    expected_prefix = ROLLOUT_IMAGE_REPOSITORY + "@"
    if not isinstance(image, str) or not image.startswith(expected_prefix):
        raise ValueError("template image must use the canonical rollout repository")
    digest = image.removeprefix(expected_prefix)
    try:
        require_image_digest(digest)
    except ValueError as error:
        raise ValueError("template image must use an immutable rollout digest") from error
    if template["private"] is not True:
        raise ValueError("template must be private")
    if template["recommended_disk_space"] != 2048:
        raise ValueError("template must request exactly 2 TB disk")
    if template["runtype"] != "ssh" or template["ssh_direct"] is not True or template["use_ssh"] is not True:
        raise ValueError("template must expose direct SSH only for observability")
    if template["jup_direct"] is not False:
        raise ValueError("template must not expose Jupyter")
    if template["onstart"] != ROLLOUT_ONSTART:
        raise ValueError("template must use the canonical rollout onstart")
    filters = template["extra_filters"]
    if not isinstance(filters, Mapping) or not isinstance(filters.get("num_gpus"), Mapping):
        raise ValueError("template must select an explicit GPU count")
    env = template["env"]
    if not isinstance(env, str):
        raise ValueError("template environment must be a string")
    required_env = (
        "AUTO_DESTROY=0",
        "B1K_ACCEPT_DATASET_TOS=YES",
        "B1K_HF_TOKEN_FILE=/workspace/.cache/huggingface/token",
        f"HF_MODEL_REPO={MODEL_REPO}",
        f"HF_DATASET_REPO={DATASET_REPO}",
        f"CAMPAIGN_ID={CAMPAIGN_ID}",
        f"BEHAVIOR_REVISION={BEHAVIOR_REVISION}",
        f"GROOT_REVISION={GROOT_REVISION}",
        f"TASK_MANIFEST_SHA256={CANONICAL_MANIFEST_SHA256}",
        "EPISODES_PER_TASK=10",
    )
    if any(item not in env for item in required_env):
        raise ValueError("template omits a required rollout campaign value")
    gpu_item = next((item.removeprefix("GPU_IDS=") for item in env.split() if item.startswith("GPU_IDS=")), None)
    if gpu_item is None:
        raise ValueError("template must provide explicit GPU_IDS")
    gpu_ids = tuple(gpu_item.split(","))
    if not 1 <= len(gpu_ids) <= 4 or any(not item.isdigit() for item in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids) or filters["num_gpus"] != {"eq": len(gpu_ids)}:
        raise ValueError("template GPU offer count must exactly match GPU_IDS")
    serialized = json.dumps(template, sort_keys=True).casefold()
    if any(item in serialized for item in _FORBIDDEN):
        raise ValueError("template must not contain GUI or credential configuration")


def render_vast_template_fixture(*, image_digest: str) -> str:
    """Render the non-runnable schema fixture without pretending it is production."""
    return json.dumps(_template(image_digest, "0" * 40, "0" * 64, (0,)), indent=2, sort_keys=True) + "\n"


def _template(image_digest: str, model_commit: str, checkpoint_artifact_sha256: str, gpu_ids: tuple[int, ...]) -> dict[str, object]:
    return {
        "env": " ".join(
            (
                "--ipc=host --shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864",
                "-e AUTO_DESTROY=0",
                "-e B1K_ACCEPT_DATASET_TOS=YES",
                f"-e CONTAINER_DIGEST={image_digest}",
                "-e RUN_ID=b1k-r1pro-100x10",
                "-e CYCLE_ID=cycle-001",
                f"-e CAMPAIGN_ID={CAMPAIGN_ID}",
                "-e EVALUATOR_MODE=public_test",
                f"-e BEHAVIOR_REVISION={BEHAVIOR_REVISION}",
                f"-e GROOT_REVISION={GROOT_REVISION}",
                f"-e TASK_MANIFEST_SHA256={CANONICAL_MANIFEST_SHA256}",
                "-e EPISODES_PER_TASK=10",
                f"-e GPU_IDS={','.join(str(item) for item in gpu_ids)}",
                "-e B1K_HF_TOKEN_FILE=/workspace/.cache/huggingface/token",
                f"-e HF_MODEL_REPO={MODEL_REPO}",
                f"-e HF_DATASET_REPO={DATASET_REPO}",
                f"-e MODEL_COMMIT={model_commit}",
                f"-e CHECKPOINT_ARTIFACT_SHA256={checkpoint_artifact_sha256}",
            )
        ),
        "extra_filters": {
            "cpu_cores_effective": {"gte": 24},
            "cpu_ram": {"gte": 128000},
            "datacenter": {"eq": True},
            "direct_port_count": {"gte": 1},
            "disk_bw": {"gte": 4000},
            "gpu_ram": {"gte": 20000},
            "inet_down": {"gte": 1000},
            "inet_up": {"gte": 500},
            "num_gpus": {"eq": len(gpu_ids)},
            "rentable": {"eq": True},
            "verified": {"eq": True},
        },
        "image": f"{ROLLOUT_IMAGE_REPOSITORY}@{image_digest}",
        "jup_direct": False,
        "name": "b1k-rollout",
        "onstart": ROLLOUT_ONSTART,
        "private": True,
        "recommended_disk_space": 2048,
        "runtype": "ssh",
        "ssh_direct": True,
        "use_ssh": True,
    }
