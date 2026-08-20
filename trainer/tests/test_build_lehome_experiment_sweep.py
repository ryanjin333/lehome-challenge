"""The generator emits all arms without credentials or runtime mutation."""
from __future__ import annotations
import json
import importlib.util
from pathlib import Path

import pytest

from test_experiment_job import _document


def _module():
    spec = importlib.util.spec_from_file_location("builder", Path(__file__).resolve().parents[2] / "scripts/build_lehome_experiment_sweep.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _controller_module():
    spec = importlib.util.spec_from_file_location("controller_runner", Path(__file__).resolve().parents[2] / "scripts/run_lehome_experiment_controller.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _request() -> dict[str, object]:
    from lehome_train.groot.experiment_manifest import APPROVED_ORIGINAL_12K_CHECKPOINT

    template = _document()
    template["parent_checkpoint"] = dict(APPROVED_ORIGINAL_12K_CHECKPOINT)
    template["evaluation"]["policy_digest"] = APPROVED_ORIGINAL_12K_CHECKPOINT["artifact_sha256"]  # type: ignore[index]
    binding = lambda kind: {"kind": kind, "repository": "owner/data", "revision": "b" * 40, "prefix": kind, "manifest_sha256": "a" * 64, "tree_sha256": "a" * 64}
    return {
        "template": template,
        "recovery_dependency": "c" * 64,
        "artifacts": {kind: binding(kind) for kind in ("bc", "ordinary_success", "recovery")},
        "request_sets": {arm: binding("runtime_request_set") for arm in "abcdefg"},
        "budget": {"gpu_seconds_ceiling": 10_000, "spend_ceiling": 100, "estimated_gpu_seconds_per_step": 1, "gpu_price_per_second": 0.01},
    }


def test_builds_all_seven_initial_arms(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_manifest import APPROVED_ORIGINAL_12K_CHECKPOINT

    ids = _module().build_initial_jobs(_request(), tmp_path)
    campaign = json.loads((tmp_path / "campaign.json").read_text())
    assert len(ids) == 7 and campaign["manifest_set_sha256"]
    assert campaign["original_12k_checkpoint"] == dict(APPROVED_ORIGINAL_12K_CHECKPOINT)


def test_rejects_a_self_consistent_but_unapproved_original_parent(tmp_path: Path) -> None:
    request = _request()
    template = request["template"]
    assert isinstance(template, dict)
    template["parent_checkpoint"] = {
        "repository": "attacker/models",
        "revision": "c" * 40,
        "subpath": "policies/step-12000",
        "artifact_sha256": "d" * 64,
    }
    template["evaluation"]["policy_digest"] = "d" * 64  # type: ignore[index]

    with pytest.raises(ValueError, match="approved original 12K"):
        _module().build_initial_jobs(request, tmp_path)


@pytest.mark.parametrize(
    "field",
    ("gpu_seconds_ceiling", "spend_ceiling", "estimated_gpu_seconds_per_step", "gpu_price_per_second"),
)
def test_rejects_zero_production_budget_inputs(tmp_path: Path, field: str) -> None:
    request = _request()
    request["budget"][field] = 0  # type: ignore[index]

    with pytest.raises(ValueError, match="positive bounded"):
        _module().build_initial_jobs(request, tmp_path)


def test_production_controller_revalidates_campaign_parent_and_budget(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_job import load_experiment_job

    _module().build_initial_jobs(_request(), tmp_path)
    runner = _controller_module()
    campaign = json.loads((tmp_path / "campaign.json").read_text())
    jobs = [load_experiment_job(path) for path in sorted(tmp_path.glob("*-500.json"))]
    runner.validate_campaign(campaign)
    runner.validate_initial_jobs(jobs)

    campaign["original_12k_checkpoint"]["artifact_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="campaign manifest"):
        runner.validate_campaign(campaign)


def test_production_controller_requires_exact_gradient_budgets_and_all_canonical_arms(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_job import load_experiment_job

    _module().build_initial_jobs(_request(), tmp_path)
    runner = _controller_module()
    campaign = json.loads((tmp_path / "campaign.json").read_text())
    jobs = [load_experiment_job(path) for path in sorted(tmp_path.glob("*-500.json"))]

    altered = json.loads(json.dumps(campaign))
    altered["gradient_step_ceiling"] = 1
    altered["tied_runner_gradient_step_ceiling"] = 1
    with pytest.raises(ValueError, match="campaign manifest"):
        runner.validate_campaign(altered)

    with pytest.raises(ValueError, match="canonical A-G"):
        runner.validate_initial_jobs(jobs[:1])


def test_controller_rejects_an_unapproved_compute_image_before_any_lease(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_job import load_experiment_job

    _module().build_initial_jobs(_request(), tmp_path)
    runner = _controller_module()
    jobs = [load_experiment_job(path) for path in sorted(tmp_path.glob("*-500.json"))]

    with pytest.raises(ValueError, match="deployment gate"):
        runner.validate_initial_jobs(jobs, training_image_id="computeimage-not-approved")

    with pytest.raises(ValueError, match="deployment gate"):
        runner.validate_initial_jobs(jobs, training_oci_digest="sha256:" + "f" * 64)

    with pytest.raises(ValueError, match="deployment gate"):
        runner.validate_initial_jobs(jobs, training_code_revision="f" * 40)
