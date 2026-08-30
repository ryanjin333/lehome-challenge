from __future__ import annotations

import json
from pathlib import Path
import shlex

import pytest

from lehome_train.b1k.contracts import RunContract
from lehome_train.b1k.template import render_vast_template
from lehome_train.b1k.training import approved_launch_plans


_IMAGE = "docker.io/ryanjin333/behavior1k-groot-n17-trainer@sha256:" + "a" * 64
_FIXTURE_IMAGE = "docker.io/ryanjin333/behavior1k-groot-n17-trainer@sha256:" + "0" * 64
_CYCLE_ID = "b1k-cycle-001"


def _template_environment(payload: dict[str, object]) -> dict[str, str]:
    tokens = shlex.split(payload["env"])
    pairs = tuple(tokens[index + 1] for index, token in enumerate(tokens) if token == "-e")
    return {pair.split("=", 1)[0]: pair.split("=", 1)[1] for pair in pairs}


def test_template_is_a_private_b1k_only_docker_hub_payload_for_one_to_four_gpus() -> None:
    payload = render_vast_template(image=_IMAGE, run_id="b1k-run-001", cycle_id=_CYCLE_ID)

    assert payload == render_vast_template(image=_IMAGE, run_id="b1k-run-001", cycle_id=_CYCLE_ID)
    assert payload["name"] == "behavior1k-groot-n17-trainer"
    assert payload["image"] == _IMAGE
    assert payload["private"] is True
    assert payload["recommended_disk_space"] >= 2048
    assert payload["extra_filters"]["num_gpus"] == {"gte": 1, "lte": 4}
    assert payload["jup_direct"] is False and payload["runtype"] == "ssh"
    assert "AUTO_DESTROY=0" in payload["env"]
    assert "CYCLE_ID=b1k-cycle-001" in payload["env"]
    assert "CONTAINER_DIGEST=sha256:" + "a" * 64 in payload["env"]
    assert "B1K_HF_TOKEN_FILE=/workspace/.cache/huggingface/token" in payload["env"]
    for identity in (
        "HF_DATASET_REPO=behavior-1k/2026-challenge-demos",
        "HF_MODEL_REPO=ryanjin333/behavior1k-groot-n17-models",
        "HF_CHECKPOINT_BUCKET=ryanjin333/behavior1k-groot-n17-checkpoints",
    ):
        assert identity in payload["env"]
    assert "HF_TOKEN=" not in str(payload)
    assert "lehome-" not in str(payload).lower()


def test_template_environment_completes_the_runtime_contract_after_token_and_data_derived_values_are_supplied() -> None:
    payload = render_vast_template(image=_IMAGE, run_id="b1k-run-001", cycle_id=_CYCLE_ID)
    plan = approved_launch_plans(num_gpus=4)[0]
    values = {
        **_template_environment(payload),
        "HF_TOKEN": "runtime-token-file-value",
        "TASK_MANIFEST_SHA256": "a" * 64,
        "MODALITY_SHA256": "ca3a2e406472650bee9439ed81a3f4a1531b6fe689cdc1b348d0f260a208e641",
        "STATS_SHA256": "b" * 64,
        "WORLD_SIZE": "4",
        "LAUNCH_PLAN_ID": plan.identity,
        "LEARNING_RATE": str(plan.learning_rate),
        "LAUNCH_ARGUMENTS_SHA256": "c" * 64,
    }

    contract = RunContract.from_environment(values)

    assert contract.run_id == "b1k-run-001"
    assert contract.cycle_id == _CYCLE_ID
    assert contract.container_digest == "sha256:" + "a" * 64
    assert contract.world_size == 4


@pytest.mark.parametrize("cycle_id", ("../unsafe", "UPPER", "", "x" * 65))
def test_template_rejects_an_unsafe_cycle_id(cycle_id: str) -> None:
    with pytest.raises(ValueError, match="identity"):
        render_vast_template(image=_IMAGE, run_id="b1k-run-001", cycle_id=cycle_id)


@pytest.mark.parametrize(
    "image",
    (
        "ghcr.io/ryanjin333/behavior1k-groot-n17-trainer@sha256:" + "a" * 64,
        "docker.io/ryanjin333/behavior1k-groot-n17-trainer:latest",
        "docker.io/ryanjin333/other@sha256:" + "a" * 64,
    ),
)
def test_template_rejects_every_noncanonical_or_mutable_image(image: str) -> None:
    with pytest.raises(ValueError, match="identity"):
        render_vast_template(image=image, run_id="b1k-run-001", cycle_id=_CYCLE_ID)


def test_zero_digest_fixture_is_only_a_deterministic_template_schema_example() -> None:
    root = Path(__file__).parents[2]
    fixture = json.loads((root / "vast-template.example.json").read_text(encoding="utf-8"))

    assert fixture == render_vast_template(image=_FIXTURE_IMAGE, run_id="b1k-template-schema", cycle_id="b1k-template-schema-cycle", schema_fixture=True)
    assert fixture["image"] == _FIXTURE_IMAGE


def test_zero_digest_is_rejected_outside_the_schema_fixture() -> None:
    with pytest.raises(ValueError, match="schema fixture"):
        render_vast_template(image=_FIXTURE_IMAGE, run_id="b1k-run-001", cycle_id=_CYCLE_ID)


def test_image_contract_has_no_registry_or_simulator_drift() -> None:
    root = Path(__file__).parents[3]
    workflow = (root / ".github/workflows/groot-trainer-image.yml").read_text(encoding="utf-8")
    verifier = (root / "trainer/scripts/verify-image.sh").read_text(encoding="utf-8")
    dockerfile = (root / "trainer/Dockerfile").read_text(encoding="utf-8")

    assert "docker.io/ryanjin333/behavior1k-groot-n17-trainer" in workflow
    assert "secrets.DOCKERHUB_USERNAME" in workflow
    assert "secrets.DOCKERHUB_TOKEN" in workflow
    assert "ghcr.io" not in workflow.lower()
    assert "outputs:" in workflow and "digest:" in workflow
    assert "target: training-runtime" in workflow
    assert "FROM nvidia/cuda" in dockerfile and " AS training-build" in dockerfile
    assert "FROM training-build AS training-runtime" in dockerfile
    assert "docker\\.io/ryanjin333/behavior1k-groot-n17-trainer@sha256:" in verifier
    for asset in ("/isaac-sim", "/IsaacLab", "/OmniGibson"):
        assert f"test ! -e {asset}" in verifier
