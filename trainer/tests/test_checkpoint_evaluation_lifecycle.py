from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPOSITORY = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "checkpoint_evaluation_lifecycle_under_test",
    REPOSITORY / "scripts" / "run_groot_checkpoint_evaluation_lifecycle.py",
)
assert SPEC is not None and SPEC.loader is not None
LIFECYCLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LIFECYCLE
SPEC.loader.exec_module(LIFECYCLE)


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _manifest(tmp_path: Path, **changes: object) -> Path:
    invocation = {
        "schema_version": 1,
        "kind": "public_unseen_tops_checkpoint_evaluation",
        "matrix_sha256": LIFECYCLE.canonical_matrix_sha256(),
        "selected_trial_ids": LIFECYCLE.canonical_trial_ids(),
        "policy_repo": LIFECYCLE.APPROVED_POLICY_REPOSITORY,
        "policy_revision": LIFECYCLE.APPROVED_POLICY_REVISION,
        "policy_step": LIFECYCLE.APPROVED_POLICY_STEP,
        "policy_artifact_sha256": LIFECYCLE.APPROVED_POLICY_ARTIFACT_SHA256,
        "code_revision": "c" * 40,
        "asset_revision": LIFECYCLE.APPROVED_ASSET_REVISION,
        "simulator_version": "5.1.0.0",
        "image_identity": LIFECYCLE.APPROVED_IMAGE_DIGEST,
        "execution_mode": "policy_server",
        "simulator_device": "cpu",
        "policy_device_pool": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        "workers": 4,
        "strategy": "baseline",
        "max_steps": 128,
        "groot_revision": LIFECYCLE.APPROVED_GROOT_REVISION,
        "groot_root": LIFECYCLE.APPROVED_GROOT_ROOT,
        "groot_python": LIFECYCLE.APPROVED_GROOT_PYTHON,
        "groot_python_sha256": LIFECYCLE.APPROVED_GROOT_PYTHON_SHA256,
        "groot_python_version": "3.10.18",
    }
    value: dict[str, object] = {
        "schema_version": 1,
        "kind": "groot_checkpoint_evaluation_lifecycle",
        "invocation": invocation,
        "rollout_image": LIFECYCLE.APPROVED_IMAGE_REPOSITORY + "@" + LIFECYCLE.APPROVED_IMAGE_DIGEST,
        "runtime": {
            "matrix_path": "configs/eval_groot_n17_public_280.json",
            "policy_path": "policies/step-12000",
            "release_assets_root": "Assets/objects/Challenge_Garment/Release",
            "groot_root": LIFECYCLE.APPROVED_GROOT_ROOT,
        },
        "code_bundle_sha256": "f" * 64,
        "hard_wall_seconds": LIFECYCLE.MAX_WALL_SECONDS,
        "total_dollar_ceiling_usd": LIFECYCLE.MAX_TOTAL_DOLLARS,
    }
    value.update(changes)
    return _write(tmp_path / "evaluation.json", value)


def _healthy_offer() -> dict[str, object]:
    return {
        "id": 7, "is_bid": False, "gpu_name": "RTX 3090", "num_gpus": 4,
        "cpu_cores_effective": 64, "cpu_ram": 131072, "driver_version": "580.65.06",
        "dph_total": 0.75,
    }


def _healthy_instance(instance_id: int = 99) -> dict[str, object]:
    return {
        **_healthy_offer(), "id": instance_id, "actual_status": "running",
        "ssh_host": "example.test", "ssh_port": 22,
    }


def _bind_bundle(manifest: Path, bundle: Path) -> None:
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["code_bundle_sha256"] = hashlib.sha256(bundle.read_bytes()).hexdigest()
    _write(manifest, value)


def test_malformed_preflight_makes_zero_provider_calls(tmp_path: Path) -> None:
    manifest = _write(tmp_path / "malformed.json", {"kind": "wrong"})
    invoked: list[tuple[str, ...]] = []
    with pytest.raises(ValueError, match="lifecycle manifest"):
        LIFECYCLE.rent_evaluation(manifest, lifecycle_root=tmp_path / "life", runner=lambda command: invoked.append(command))
    assert invoked == []


@pytest.mark.parametrize("field,value", [("dph_total", 1.0), ("driver_version", "570.1"), ("is_bid", True)])
def test_capture_requires_sub_dollar_on_demand_r580_offer(tmp_path: Path, field: str, value: object) -> None:
    offer = _healthy_offer(); offer[field] = value
    with pytest.raises(ValueError, match="sub-\\$1|R580|on-demand"):
        LIFECYCLE.capture_provider_evidence(offers=[offer], instances=[], volumes=[], output=tmp_path / "provider.json", now_unix=10)


def test_capture_rejects_hourly_rate_that_can_exceed_four_hour_dollar_ceiling(tmp_path: Path) -> None:
    offer = _healthy_offer(); offer["dph_total"] = 0.76
    with pytest.raises(ValueError, match="dollar ceiling"):
        LIFECYCLE.capture_provider_evidence(offers=[offer], instances=[], volumes=[], output=tmp_path / "provider.json", now_unix=10)


def test_rent_rejects_account_overlap_before_create(tmp_path: Path) -> None:
    invoked: list[tuple[str, ...]] = []
    def runner(command: tuple[str, ...]) -> str:
        invoked.append(command)
        if command[-1] == "instances":
            return json.dumps([_healthy_instance(17)])
        if command[-1] == "volumes":
            return "[]"
        raise AssertionError(command)
    with pytest.raises(ValueError, match="zero instances"):
        LIFECYCLE.rent_evaluation(_manifest(tmp_path), lifecycle_root=tmp_path / "life", runner=runner)
    assert not any(command[:4] == ("vastai", "--raw", "create", "instance") for command in invoked)


def test_rent_captures_only_r580_four_3090_host_and_hard_budget(tmp_path: Path) -> None:
    invoked: list[tuple[str, ...]] = []
    def runner(command: tuple[str, ...]) -> str:
        invoked.append(command)
        if command[-1] in {"instances", "volumes"}:
            return "[]"
        if command[:4] == ("vastai", "--raw", "search", "offers"):
            return json.dumps([_healthy_offer()])
        if command[:4] == ("vastai", "--raw", "create", "instance"):
            return '{"new_contract":99}'
        if command[:4] == ("vastai", "--raw", "show", "instance"):
            return json.dumps(_healthy_instance())
        raise AssertionError(command)
    receipt = LIFECYCLE.rent_evaluation(_manifest(tmp_path), lifecycle_root=tmp_path / "life", runner=runner)
    assert receipt["instance_id"] == 99
    assert receipt["invocation_sha256"] == LIFECYCLE.canonical_sha256(json.loads(_manifest(tmp_path).read_text())["invocation"])
    create = next(command for command in invoked if command[:4] == ("vastai", "--raw", "create", "instance"))
    assert create[create.index("--image") + 1] == LIFECYCLE.APPROVED_IMAGE_REPOSITORY + "@" + LIFECYCLE.APPROVED_IMAGE_DIGEST
    assert "--on-demand" in next(command for command in invoked if command[:4] == ("vastai", "--raw", "search", "offers"))


def test_stage_launch_uses_exact_cpu_policy_server_four_cuda_slots_and_wall_cap(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    bundle = tmp_path / "code.bundle"; bundle.write_bytes(b"bundle")
    _bind_bundle(manifest, bundle)
    token = tmp_path / "token"; token.write_text("secret", encoding="utf-8"); token.chmod(0o600)
    instance = {"kind": "groot_checkpoint_evaluation_instance", "instance_id": 99, "host": "example.test", "port": 22, "lease_deadline_unix": __import__("time").time_ns() // 1_000_000_000 + LIFECYCLE.MAX_WALL_SECONDS, "invocation_sha256": LIFECYCLE.canonical_sha256(json.loads(manifest.read_text())["invocation"])}
    invoked: list[tuple[str, ...]] = []
    def runner(command: tuple[str, ...]):
        invoked.append(command)
        if command[-1] == "instances": return json.dumps([_healthy_instance()])
        if command[-1] == "volumes": return "[]"
        if command[:4] == ("vastai", "--raw", "show", "instance"): return json.dumps(_healthy_instance())
        return type("Result", (), {"returncode": 0, "stdout": ""})()
    receipt = LIFECYCLE.stage_launch_sync_evaluation(manifest, instance, lifecycle_root=tmp_path / "life", runner=runner, code_bundle=bundle, token_file=token)
    assert receipt["status"] == "synced"
    script = next(command[-1] for command in invoked if command and command[0] == "ssh" and "-lc" in command)
    assert "--public-unseen-tops" in script and "--execution-mode policy_server" in script and "--device cpu" in script and "--workers 4" in script
    assert "--matrix" in script and "--policy-path /workspace/checkpoints/" in script and "/policies/step-12000" in script
    assert "hf download ryanjin333/lehome-groot-n17-models --revision" in script
    assert "hf download lehome/asset_challenge --repo-type dataset --revision" in script
    assert all(f"cuda:{slot}" in script for slot in range(4))
    assert "timeout --signal=TERM --kill-after=20s " in script and f"{LIFECYCLE.MAX_WALL_SECONDS + 1}s" not in script
    assert "git -C" in script and "diff --quiet" in script
    assert "/opt/lehome-challenge/.venv/bin/python -m scripts.run_groot_flywheel_campaign" in script
    assert "PYTHONPATH=" in script


def test_composed_campaign_arguments_are_parser_complete_and_fail_without_matrix(tmp_path: Path) -> None:
    manifest = json.loads(_manifest(tmp_path).read_text())
    arguments = LIFECYCLE.campaign_arguments(manifest["invocation"], manifest["runtime"], checkout="/remote/code", output_root="/remote/output")
    parser = LIFECYCLE._campaign_module().build_parser()
    assert parser.parse_args(arguments).matrix == Path("/remote/code/configs/eval_groot_n17_public_280.json")
    with pytest.raises(SystemExit):
        parser.parse_args(arguments[:arguments.index("--matrix")] + arguments[arguments.index("--matrix") + 2:])


def test_remote_failure_syncs_redacted_evidence_and_destroys_exactly_once(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    bundle = tmp_path / "code.bundle"; bundle.write_bytes(b"bundle")
    _bind_bundle(manifest, bundle)
    token = tmp_path / "token"; token.write_text("secret-token", encoding="utf-8"); token.chmod(0o600)
    instance = {"kind": "groot_checkpoint_evaluation_instance", "instance_id": 99, "host": "example.test", "port": 22, "lease_deadline_unix": __import__("time").time_ns() // 1_000_000_000 + LIFECYCLE.MAX_WALL_SECONDS, "invocation_sha256": LIFECYCLE.canonical_sha256(json.loads(manifest.read_text())["invocation"])}
    invoked: list[tuple[str, ...]] = []; destroyed = False
    def runner(command: tuple[str, ...]):
        nonlocal destroyed
        invoked.append(command)
        if command[-1] == "instances": return "[]" if destroyed else json.dumps([_healthy_instance()])
        if command[-1] == "volumes": return "[]"
        if command[:4] == ("vastai", "--raw", "show", "instance"): return "{}" if destroyed else json.dumps(_healthy_instance())
        if command[:3] == ("vastai", "destroy", "instance"):
            destroyed = True; return ""
        if command and command[0] == "ssh" and "-lc" in command: return type("Result", (), {"returncode": 1, "stdout": "secret-token"})()
        return type("Result", (), {"returncode": 0, "stdout": ""})()
    with pytest.raises(RuntimeError, match="remote launch"):
        LIFECYCLE.stage_launch_sync_evaluation(manifest, instance, lifecycle_root=tmp_path / "life", runner=runner, code_bundle=bundle, token_file=token)
    assert sum(command[:3] == ("vastai", "destroy", "instance") for command in invoked) == 1
    failure = json.loads(next((tmp_path / "life").glob("failure-*.json")).read_text())
    assert failure["instance_id"] == 99 and failure["account_instances_empty"] is True and failure["account_volumes_empty"] is True
    assert "secret-token" not in json.dumps(failure)


def test_publication_receipt_must_bind_current_instance_and_exact_invocation(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    invocation = json.loads(manifest.read_text())["invocation"]
    instance = _write(tmp_path / "instance.json", {"kind": "groot_checkpoint_evaluation_instance", "instance_id": 99, "invocation_sha256": LIFECYCLE.canonical_sha256(invocation)})
    publication = _write(tmp_path / "publication.json", {"kind": "groot_checkpoint_evaluation_publication", "disposable": True, "instance_id": 99, "instance_receipt_sha256": hashlib.sha256(instance.read_bytes()).hexdigest(), "invocation": invocation, "invocation_sha256": LIFECYCLE.canonical_sha256(invocation), "immutable_revision": "a" * 40, "remote_prefix": "evaluations/x", "repository_private": True, "tree_listing_verified": True, "fresh_readback_verified": True})
    invoked: list[tuple[str, ...]] = []
    def runner(command: tuple[str, ...]) -> str:
        invoked.append(command)
        if command[:3] == ("vastai", "destroy", "instance"): return ""
        if command[:4] == ("vastai", "--raw", "show", "instance"): return "{}"
        if command[-1] in {"instances", "volumes"}: return "[]"
        raise AssertionError(command)
    assert LIFECYCLE.destroy_after_publication(99, publication, instance, runner=runner)
    stale = json.loads(publication.read_text()); stale["instance_id"] = 17; _write(publication, stale)
    with pytest.raises(ValueError, match="publication"):
        LIFECYCLE.destroy_after_publication(99, publication, instance, runner=runner)


def test_disposal_accepts_vast_instances_null_absence_shape() -> None:
    def runner(command: tuple[str, ...]) -> str:
        if command[:4] == ("vastai", "--raw", "show", "instance"):
            return '{"instances": null}'
        return "[]"
    assert LIFECYCLE._verify_absent(99, runner) == (True, True, True)


def test_comparator_only_allows_checkpoint_step_and_artifact_hash(tmp_path: Path) -> None:
    one = json.loads(_manifest(tmp_path).read_text())["invocation"]
    two = {**one, "policy_step": 2000, "policy_artifact_sha256": "1" * 64}
    assert LIFECYCLE.compare_checkpoint_invocations(one, two) == {}
    two["code_revision"] = "2" * 40
    assert LIFECYCLE.compare_checkpoint_invocations(one, two) == {"code_revision": (one["code_revision"], two["code_revision"])}
