"""Training-only, secret-free Vast template payloads; never creates instances."""

from __future__ import annotations

import re

from lehome_train.constants import (
    BEHAVIOR_1K_CHECKPOINT_BUCKET,
    BEHAVIOR_1K_DATASET_REPOSITORY,
    BEHAVIOR_1K_DATASET_REVISION,
    BEHAVIOR_1K_FINAL_MODEL_REPOSITORY,
    BEHAVIOR_1K_TRAINER_IMAGE_REPOSITORY,
    COSMOS_REPOSITORY,
    COSMOS_REVISION,
    ISAAC_GROOT_REVISION,
    MODEL_REVISION,
)


_IMAGE = re.compile(re.escape(BEHAVIOR_1K_TRAINER_IMAGE_REPOSITORY) + r"@sha256:[0-9a-f]{64}")
_RUN_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
_TOKEN_FILE = "/workspace/.cache/huggingface/token"


def render_vast_template(*, image: str, run_id: str, cycle_id: str, schema_fixture: bool = False) -> dict[str, object]:
    if _IMAGE.fullmatch(image) is None or _RUN_ID.fullmatch(run_id) is None or _RUN_ID.fullmatch(cycle_id) is None:
        raise ValueError("invalid Vast template identity")
    zero_digest = image.endswith("sha256:" + "0" * 64)
    if zero_digest != schema_fixture:
        raise ValueError("zero image digest is reserved for the schema fixture")
    container_digest = image.rsplit("@", 1)[1]
    docker_env = " ".join((
        "-e AUTO_DESTROY=0",
        f"-e RUN_ID={run_id}",
        f"-e CYCLE_ID={cycle_id}",
        f"-e CONTAINER_DIGEST={container_digest}",
        f"-e B1K_HF_TOKEN_FILE={_TOKEN_FILE}",
        f"-e HF_DATASET_REPO={BEHAVIOR_1K_DATASET_REPOSITORY}",
        f"-e HF_MODEL_REPO={BEHAVIOR_1K_FINAL_MODEL_REPOSITORY}",
        f"-e HF_CHECKPOINT_BUCKET={BEHAVIOR_1K_CHECKPOINT_BUCKET}",
        f"-e DATASET_REVISION={BEHAVIOR_1K_DATASET_REVISION}",
        f"-e GROOT_REVISION={ISAAC_GROOT_REVISION}",
        "-e TRAIN_STEPS=15000",
        "-e SAVE_STEPS=1000",
        "-e CHECKPOINT_KEEP=2",
        "-e RESUME_POLICY=auto",
        f"-e BASE_MODEL_REVISION={MODEL_REVISION}",
        f"-e COSMOS_REPOSITORY={COSMOS_REPOSITORY}",
        f"-e COSMOS_REVISION={COSMOS_REVISION}",
        f"-e EXPERIMENT_NAME={run_id}",
    ))
    return {
        "name": "b1k-training",
        "image": image,
        "env": "--user root --ipc=host --shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864 " + docker_env,
        "onstart": "bash /opt/b1k-launchkit/onstart.sh",
        "jup_direct": False,
        "ssh_direct": True,
        "runtype": "ssh",
        "use_ssh": True,
        "recommended_disk_space": 2048,
        "private": True,
        "extra_filters": {
            "num_gpus": {"gte": 1, "lte": 4},
            "gpu_ram": {"gte": 96000},
            "cpu_ram": {"gte": 128000},
            "disk_bw": {"gte": 4000},
            "inet_down": {"gte": 1000},
            "inet_up": {"gte": 500},
            "cpu_cores_effective": {"gte": 24},
            "direct_port_count": {"gte": 1},
            "verified": {"eq": True},
            "datacenter": {"eq": True},
            "external": {"eq": False},
            "rentable": {"eq": True},
            "gpu_name": {"in": ["RTX PRO 6000 S", "RTX PRO 6000 WS"]},
        },
    }
