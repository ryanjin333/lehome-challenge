from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from b1k_rollout.template import render_vast_template, render_vast_template_fixture, validate_vast_template


ROLLOUT = Path(__file__).parents[1]
_DIGEST = "sha256:" + "a" * 64


def test_template_renders_private_headless_campaign_with_explicit_gpu_bounds() -> None:
    rendered = render_vast_template(image_digest=_DIGEST, model_commit="b" * 40, checkpoint_artifact_sha256="c" * 64, gpu_ids=(0,))
    template = json.loads(rendered)

    assert template["private"] is True
    assert template["image"] == f"docker.io/ryanjin333/behavior1k-groot-n17-rollout@{_DIGEST}"
    assert template["recommended_disk_space"] >= 2048
    assert template["runtype"] == "ssh"
    assert template["ssh_direct"] is True
    assert template["jup_direct"] is False
    assert template["extra_filters"]["num_gpus"] == {"eq": 1}
    assert "AUTO_DESTROY=0" in template["env"]
    assert "B1K_ACCEPT_DATASET_TOS=YES" in template["env"]
    assert "B1K_HF_TOKEN_FILE=/workspace/.cache/huggingface/token" in template["env"]
    assert "HF_TOKEN=" not in template["env"]
    assert "novnc" not in rendered.casefold()
    assert "x11" not in rendered.casefold()
    assert "jupyter" not in rendered.casefold()
    validate_vast_template(template)


@pytest.mark.parametrize(
    "image_digest",
    ["latest", "sha256:" + "A" * 64, "sha256:" + "f" * 63],
)
def test_template_refuses_mutable_or_malformed_image_digests(image_digest: str) -> None:
    with pytest.raises(ValueError, match="immutable.*digest"):
        render_vast_template(image_digest=image_digest, model_commit="b" * 40, checkpoint_artifact_sha256="c" * 64, gpu_ids=(0,))


def test_checked_in_template_is_the_canonical_secret_free_schema_fixture() -> None:
    fixture = json.loads((ROLLOUT / "vast-template.example.json").read_text(encoding="utf-8"))

    validate_vast_template(fixture)
    assert fixture == json.loads(render_vast_template_fixture(image_digest="sha256:" + "0" * 64))
    assert not re.search(r"hf_[A-Za-z0-9]{30,}", json.dumps(fixture))


def test_production_template_rejects_zero_checkpoint_identity_and_mismatched_gpu_offer() -> None:
    with pytest.raises(ValueError, match="zero checkpoint"):
        render_vast_template(image_digest=_DIGEST, model_commit="0" * 40, checkpoint_artifact_sha256="c" * 64, gpu_ids=(0,))
    rendered = json.loads(render_vast_template(image_digest=_DIGEST, model_commit="b" * 40, checkpoint_artifact_sha256="c" * 64, gpu_ids=(0, 1)))
    rendered["extra_filters"]["num_gpus"] = {"eq": 1}
    with pytest.raises(ValueError, match="GPU offer"):
        validate_vast_template(rendered)
