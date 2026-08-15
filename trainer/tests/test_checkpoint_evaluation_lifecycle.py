from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
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


@pytest.fixture(autouse=True)
def _isolated_claim_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(LIFECYCLE, "EVALUATION_RENT_CLAIM_ROOT", tmp_path / "claims-global")


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
        "strategy": "canonical",
        "max_steps": 600,
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


def _evaluation_release(invocation: dict[str, object]) -> tuple[str, str]:
    release_id = LIFECYCLE.canonical_sha256({
        "invocation": invocation,
        "trial_ids": LIFECYCLE.canonical_trial_ids(),
        "kind": "diagnostic_evaluation_not_rft",
    })
    return release_id, f"evaluations/groot-n17-step-{invocation['policy_step']}/{release_id}"


def test_malformed_preflight_makes_zero_provider_calls(tmp_path: Path) -> None:
    manifest = _write(tmp_path / "malformed.json", {"kind": "wrong"})
    invoked: list[tuple[str, ...]] = []
    with pytest.raises(ValueError, match="lifecycle manifest"):
        LIFECYCLE.rent_evaluation(manifest, lifecycle_root=tmp_path / "life", runner=lambda command: invoked.append(command))
    assert invoked == []


def test_lifecycle_uses_the_exact_public_pullthrough_mirror_manifest() -> None:
    assert LIFECYCLE.APPROVED_IMAGE_REPOSITORY == "mirror.gcr.io/ryanjin333/lehome-rollout"
    assert LIFECYCLE.APPROVED_IMAGE_DIGEST == "sha256:293c4f258f3742a7234699d706fb7088d0da8a764957bc79b244d830561abc12"


@pytest.mark.parametrize(("field", "value"), (("strategy", "mild"), ("max_steps", 128)))
def test_manifest_rejects_campaign_default_drift_before_provider_calls(tmp_path: Path, field: str, value: object) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["invocation"][field] = value
    _write(manifest, payload)
    invoked: list[tuple[str, ...]] = []
    with pytest.raises(ValueError, match="canonical CPU policy-server"):
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
    receipt = LIFECYCLE.rent_evaluation(
        _manifest(tmp_path), lifecycle_root=tmp_path / "life", runner=runner,
        watchdog_launcher=lambda instance_id, deadline, receipt: 4321,
    )
    assert receipt["instance_id"] == 99
    assert receipt["invocation_sha256"] == LIFECYCLE.canonical_sha256(json.loads(_manifest(tmp_path).read_text())["invocation"])
    create = next(command for command in invoked if command[:4] == ("vastai", "--raw", "create", "instance"))
    assert create[create.index("--image") + 1] == LIFECYCLE.APPROVED_IMAGE_REPOSITORY + "@" + LIFECYCLE.APPROVED_IMAGE_DIGEST
    assert "--on-demand" in next(command for command in invoked if command[:4] == ("vastai", "--raw", "search", "offers"))


def test_watchdog_starts_before_the_first_running_readback(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    invoked: list[tuple[str, ...]] = []
    watchdogs: list[tuple[int, int, Path]] = []
    exact_reads = 0
    create_attempted = False
    destroyed = False

    def runner(command: tuple[str, ...]) -> str:
        nonlocal exact_reads, create_attempted, destroyed
        invoked.append(command)
        if command[:4] == ("vastai", "--raw", "create", "instance"):
            create_attempted = True
            return '{"new_contract":99}'
        if command[:4] == ("vastai", "--raw", "search", "offers"):
            return json.dumps([_healthy_offer()])
        if command[:4] == ("vastai", "--raw", "show", "instance"):
            exact_reads += 1
            if exact_reads == 1:
                raise RuntimeError("running readback failed")
            return "{}" if destroyed else json.dumps(_healthy_instance())
        if command[:3] == ("vastai", "destroy", "instance"):
            destroyed = True
            return ""
        if command[-1] == "instances":
            return "[]" if destroyed or not create_attempted else json.dumps([_healthy_instance()])
        if command[-1] == "volumes":
            return "[]"
        raise AssertionError(command)

    with pytest.raises(RuntimeError, match="running readback failed"):
        LIFECYCLE.rent_evaluation(
            manifest, lifecycle_root=tmp_path / "life", runner=runner,
            watchdog_launcher=lambda instance_id, deadline, receipt: watchdogs.append((instance_id, deadline, receipt)) or 4321,
        )
    assert len(watchdogs) == 1 and watchdogs[0][0] == 99
    assert sum(command[:3] == ("vastai", "destroy", "instance") for command in invoked) == 1


def test_stage_launch_uses_exact_cpu_policy_server_four_cuda_slots_and_wall_cap(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    bundle = tmp_path / "code.bundle"; bundle.write_bytes(b"bundle")
    _bind_bundle(manifest, bundle)
    token = tmp_path / "token"; token.write_text("secret", encoding="utf-8"); token.chmod(0o600)
    instance = {"kind": "groot_checkpoint_evaluation_instance", "instance_id": 99, "host": "example.test", "port": 22, "lease_deadline_unix": __import__("time").time_ns() // 1_000_000_000 + LIFECYCLE.MAX_WALL_SECONDS, "watchdog_pid": 4321, "watchdog_receipt": str(tmp_path / "watchdog.json"), "rent_claim_path": str(LIFECYCLE.EVALUATION_RENT_CLAIM_ROOT / "active.json"), "invocation_sha256": LIFECYCLE.canonical_sha256(json.loads(manifest.read_text())["invocation"])}
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
    timeout_match = re.search(r"timeout --signal=TERM --kill-after=20s (\d+)s", script)
    assert timeout_match is not None and int(timeout_match.group(1)) <= LIFECYCLE.MAX_WALL_SECONDS - LIFECYCLE.LEASE_WATCHDOG_RESERVE_SECONDS
    assert "git -C" in script and "diff --quiet" in script
    assert "/opt/lehome-challenge/.venv/bin/python -m scripts.run_groot_flywheel_campaign" in script
    assert "PYTHONPATH=" in script
    assert script.count("test -x /opt/lehome-challenge/.venv/bin/python") == 1


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
    instance = {"kind": "groot_checkpoint_evaluation_instance", "instance_id": 99, "host": "example.test", "port": 22, "lease_deadline_unix": __import__("time").time_ns() // 1_000_000_000 + LIFECYCLE.MAX_WALL_SECONDS, "watchdog_pid": 4321, "watchdog_receipt": str(tmp_path / "watchdog.json"), "rent_claim_path": str(LIFECYCLE.EVALUATION_RENT_CLAIM_ROOT / "active.json"), "invocation_sha256": LIFECYCLE.canonical_sha256(json.loads(manifest.read_text())["invocation"])}
    claim_path = LIFECYCLE._rent_claim_path(json.loads(manifest.read_text()))
    claim = LIFECYCLE._acquire_rent_claim(claim_path, json.loads(manifest.read_text()))
    LIFECYCLE._terminalize_rent_claim(claim_path, claim, status="succeeded", instance_id=99)
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
    release_id, remote_prefix = _evaluation_release(invocation)
    instance = _write(tmp_path / "instance.json", {"kind": "groot_checkpoint_evaluation_instance", "instance_id": 99, "rent_claim_path": str(LIFECYCLE.EVALUATION_RENT_CLAIM_ROOT / "active.json"), "invocation_sha256": LIFECYCLE.canonical_sha256(invocation)})
    publication = _write(tmp_path / "publication.json", {"kind": "groot_checkpoint_evaluation_publication", "disposable": True, "instance_id": 99, "instance_receipt_sha256": hashlib.sha256(instance.read_bytes()).hexdigest(), "invocation": invocation, "invocation_sha256": LIFECYCLE.canonical_sha256(invocation), "immutable_revision": "a" * 40, "repository": "ryanjin333/lehome-groot-n17-data", "release_id": release_id, "remote_prefix": remote_prefix, "repository_private": True, "tree_listing_verified": True, "fresh_readback_verified": True})
    invoked: list[tuple[str, ...]] = []
    destroyed = False
    def runner(command: tuple[str, ...]) -> str:
        nonlocal destroyed
        invoked.append(command)
        if command[:3] == ("vastai", "destroy", "instance"):
            destroyed = True; return ""
        if command[:4] == ("vastai", "--raw", "show", "instance"): return "{}" if destroyed else json.dumps(_healthy_instance())
        if command[-1] == "instances": return "[]" if destroyed else json.dumps([_healthy_instance()])
        if command[-1] == "volumes": return "[]"
        raise AssertionError(command)
    disposal = tmp_path / "disposal.json"
    claim_path = LIFECYCLE._rent_claim_path(json.loads(manifest.read_text()))
    claim = LIFECYCLE._acquire_rent_claim(claim_path, json.loads(manifest.read_text()))
    LIFECYCLE._terminalize_rent_claim(claim_path, claim, status="succeeded", instance_id=99)
    receipt = LIFECYCLE.destroy_after_publication(99, publication, instance, disposal_receipt=disposal, runner=runner)
    assert receipt["destroyed_and_absent"] is True
    assert receipt["destroy_issued"] is True
    assert sum(command[:3] == ("vastai", "destroy", "instance") for command in invoked) == 1
    assert receipt["publication_receipt_sha256"] == hashlib.sha256(publication.read_bytes()).hexdigest()
    stale = json.loads(publication.read_text()); stale["instance_id"] = 17; _write(publication, stale)
    with pytest.raises(ValueError, match="publication"):
        LIFECYCLE.destroy_after_publication(99, publication, instance, disposal_receipt=tmp_path / "disposal-2.json", runner=runner)


@pytest.mark.parametrize(("field", "value"), (
    ("repository", "attacker/other"),
    ("release_id", "0" * 64),
    ("remote_prefix", "evaluations/forged"),
))
def test_disposal_rejects_forged_release_identity_before_provider_calls(tmp_path: Path, field: str, value: str) -> None:
    invocation = json.loads(_manifest(tmp_path).read_text())["invocation"]
    release_id, remote_prefix = _evaluation_release(invocation)
    instance = _write(tmp_path / "instance.json", {
        "kind": "groot_checkpoint_evaluation_instance", "instance_id": 99,
        "rent_claim_path": str(LIFECYCLE.EVALUATION_RENT_CLAIM_ROOT / "active.json"),
        "invocation_sha256": LIFECYCLE.canonical_sha256(invocation),
    })
    publication_value = {
        "kind": "groot_checkpoint_evaluation_publication", "disposable": True,
        "repository": "ryanjin333/lehome-groot-n17-data", "repository_private": True,
        "release_id": release_id, "remote_prefix": remote_prefix,
        "immutable_revision": "a" * 40, "instance_id": 99,
        "instance_receipt_sha256": hashlib.sha256(instance.read_bytes()).hexdigest(),
        "invocation": invocation, "invocation_sha256": LIFECYCLE.canonical_sha256(invocation),
        "tree_listing_verified": True, "fresh_readback_verified": True,
    }
    publication_value[field] = value
    publication = _write(tmp_path / "publication.json", publication_value)
    invoked: list[tuple[str, ...]] = []
    with pytest.raises(ValueError, match="publication"):
        LIFECYCLE.destroy_after_publication(
            99, publication, instance, disposal_receipt=tmp_path / "disposal.json",
            runner=lambda command: invoked.append(command),
        )
    assert invoked == []


def test_rent_claim_is_singleton_and_second_controller_is_provider_free(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    claim_path = LIFECYCLE._rent_claim_path(json.loads(manifest.read_text()))
    LIFECYCLE._acquire_rent_claim(claim_path, json.loads(manifest.read_text()))
    other = json.loads(manifest.read_text())
    other["code_bundle_sha256"] = "1" * 64
    assert LIFECYCLE._rent_claim_path(other) == claim_path
    invoked: list[tuple[str, ...]] = []
    with pytest.raises(ValueError, match="already held"):
        LIFECYCLE.rent_evaluation(manifest, lifecycle_root=tmp_path / "life", runner=lambda command: invoked.append(command))
    assert invoked == []


def test_ambiguous_create_blocks_claim_and_prevents_second_rent(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    claim_root = LIFECYCLE.EVALUATION_RENT_CLAIM_ROOT
    invoked: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...]) -> str:
        invoked.append(command)
        if command[-1] in {"instances", "volumes"}:
            return "[]"
        if command[:4] == ("vastai", "--raw", "search", "offers"):
            return json.dumps([_healthy_offer()])
        if command[:4] == ("vastai", "--raw", "create", "instance"):
            return "{}"
        raise AssertionError(command)

    with pytest.raises(RuntimeError, match="lacks instance ID"):
        LIFECYCLE.rent_evaluation(manifest, lifecycle_root=tmp_path / "life", runner=runner, ambiguous_create_polls=1)
    claim_path = LIFECYCLE._rent_claim_path(json.loads(manifest.read_text()))
    assert json.loads(claim_path.read_text())["status"] == "blocked_ambiguous_create"
    later: list[tuple[str, ...]] = []
    with pytest.raises(ValueError, match="already held"):
        LIFECYCLE.rent_evaluation(manifest, lifecycle_root=tmp_path / "later", runner=lambda command: later.append(command))
    assert later == []


def test_ambiguous_create_cleans_the_only_new_compatible_host_and_releases_claim(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    claim_root = LIFECYCLE.EVALUATION_RENT_CLAIM_ROOT
    invoked: list[tuple[str, ...]] = []
    create_attempted = False
    destroyed = False

    def runner(command: tuple[str, ...]) -> str:
        nonlocal create_attempted, destroyed
        invoked.append(command)
        if command[:4] == ("vastai", "--raw", "create", "instance"):
            create_attempted = True
            return "{}"
        if command[:4] == ("vastai", "--raw", "search", "offers"):
            return json.dumps([_healthy_offer()])
        if command[:3] == ("vastai", "destroy", "instance"):
            destroyed = True
            return ""
        if command[:4] == ("vastai", "--raw", "show", "instance"):
            return "{}" if destroyed else json.dumps(_healthy_instance())
        if command[-1] == "instances":
            return "[]" if destroyed or not create_attempted else json.dumps([_healthy_instance()])
        if command[-1] == "volumes":
            return "[]"
        raise AssertionError(command)

    with pytest.raises(RuntimeError, match="lacks instance ID"):
        LIFECYCLE.rent_evaluation(manifest, lifecycle_root=tmp_path / "life", runner=runner, ambiguous_create_polls=1)
    assert sum(command[:3] == ("vastai", "destroy", "instance") for command in invoked) == 1
    assert list(claim_root.glob("*.json")) == []


def test_ambiguous_create_cleans_a_transitional_single_new_host(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    invoked: list[tuple[str, ...]] = []
    create_attempted = False
    destroyed = False

    def runner(command: tuple[str, ...]) -> str:
        nonlocal create_attempted, destroyed
        invoked.append(command)
        if command[:4] == ("vastai", "--raw", "create", "instance"):
            create_attempted = True
            return "{}"
        if command[:4] == ("vastai", "--raw", "search", "offers"):
            return json.dumps([_healthy_offer()])
        if command[:3] == ("vastai", "destroy", "instance"):
            destroyed = True
            return ""
        if command[:4] == ("vastai", "--raw", "show", "instance"):
            return "{}" if destroyed else json.dumps({"id": 99, "actual_status": "loading"})
        if command[-1] == "instances":
            if destroyed or not create_attempted:
                return "[]"
            return json.dumps([{"id": 99, "actual_status": "loading"}])
        if command[-1] == "volumes":
            return "[]"
        raise AssertionError(command)

    with pytest.raises(RuntimeError, match="lacks instance ID"):
        LIFECYCLE.rent_evaluation(
            manifest, lifecycle_root=tmp_path / "life", runner=runner,
            ambiguous_create_polls=1,
        )
    assert sum(command[:3] == ("vastai", "destroy", "instance") for command in invoked) == 1
    assert not (LIFECYCLE.EVALUATION_RENT_CLAIM_ROOT / "active.json").exists()


def test_disposal_accepts_vast_instances_null_absence_shape() -> None:
    def runner(command: tuple[str, ...]) -> str:
        if command[:4] == ("vastai", "--raw", "show", "instance"):
            return '{"instances": null}'
        return "[]"
    assert LIFECYCLE._verify_absent(99, runner) == (True, True, True)


def test_disposal_does_not_treat_an_instance_row_as_absent() -> None:
    def runner(command: tuple[str, ...]) -> str:
        if command[:4] == ("vastai", "--raw", "show", "instance"):
            return json.dumps(_healthy_instance())
        if command[-1] == "instances":
            return json.dumps([_healthy_instance()])
        return "[]"

    assert LIFECYCLE._verify_absent(99, runner, polls=1) == (False, False, True)


def test_watchdog_destroys_once_at_deadline_and_keeps_account_empty(tmp_path: Path) -> None:
    destroyed = False
    invoked: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...]) -> str:
        nonlocal destroyed
        invoked.append(command)
        if command[:3] == ("vastai", "destroy", "instance"):
            destroyed = True
            return ""
        if command[:4] == ("vastai", "--raw", "show", "instance"):
            return "{}" if destroyed else json.dumps(_healthy_instance())
        if command[-1] == "instances":
            return "[]" if destroyed else json.dumps([_healthy_instance()])
        if command[-1] == "volumes":
            return "[]"
        raise AssertionError(command)

    receipt = LIFECYCLE.enforce_lease_deadline(
        99, 100, tmp_path / "watchdog.json", runner=runner,
        sleep=lambda _seconds: None, now=lambda: 100,
    )
    assert receipt["destroyed_and_absent"] is True and receipt["destroy_issued"] is True
    assert sum(command[:3] == ("vastai", "destroy", "instance") for command in invoked) == 1


def test_watchdog_and_explicit_disposal_share_one_destroy_lock() -> None:
    destroyed = False
    invoked: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...]) -> str:
        nonlocal destroyed
        invoked.append(command)
        if command[:3] == ("vastai", "destroy", "instance"):
            destroyed = True
            return ""
        if command[:4] == ("vastai", "--raw", "show", "instance"):
            return "{}" if destroyed else json.dumps(_healthy_instance())
        if command[-1] == "instances":
            return "[]" if destroyed else json.dumps([_healthy_instance()])
        if command[-1] == "volumes":
            return "[]"
        raise AssertionError(command)

    first = LIFECYCLE._destroy_owned_once(99, runner, polls=1)
    second = LIFECYCLE._destroy_owned_once(99, runner, polls=1)
    assert first == (True, True, True, True)
    assert second == (False, True, True, True)
    assert sum(command[:3] == ("vastai", "destroy", "instance") for command in invoked) == 1


def test_destroy_retries_inconsistent_exact_absence_when_account_still_lists_instance() -> None:
    destroyed = False
    invoked: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...]) -> str:
        nonlocal destroyed
        invoked.append(command)
        if command[:3] == ("vastai", "destroy", "instance"):
            destroyed = True
            return ""
        if command[:4] == ("vastai", "--raw", "show", "instance"):
            return '{"instances":null}'
        if command[-1] == "instances":
            return "[]" if destroyed else json.dumps([_healthy_instance()])
        if command[-1] == "volumes":
            return "[]"
        raise AssertionError(command)

    assert LIFECYCLE._destroy_owned_once(99, runner, polls=1) == (True, True, True, True)
    assert sum(command[:3] == ("vastai", "destroy", "instance") for command in invoked) == 1


def test_pretransport_validation_failure_still_cleans_host_and_releases_claim(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    bundle = tmp_path / "wrong.bundle"; bundle.write_bytes(b"wrong")
    token = tmp_path / "token"; token.write_text("secret", encoding="utf-8"); token.chmod(0o600)
    invocation = json.loads(manifest.read_text())["invocation"]
    instance = {
        "kind": "groot_checkpoint_evaluation_instance", "instance_id": 99,
        "host": "example.test", "port": 22,
        "lease_deadline_unix": __import__("time").time_ns() // 1_000_000_000 + LIFECYCLE.MAX_WALL_SECONDS,
        "watchdog_pid": 4321, "watchdog_receipt": str(tmp_path / "watchdog.json"),
        "rent_claim_path": str(LIFECYCLE.EVALUATION_RENT_CLAIM_ROOT / "active.json"),
        "invocation_sha256": LIFECYCLE.canonical_sha256(invocation),
    }
    claim_path = LIFECYCLE._rent_claim_path(json.loads(manifest.read_text()))
    claim = LIFECYCLE._acquire_rent_claim(claim_path, json.loads(manifest.read_text()))
    LIFECYCLE._terminalize_rent_claim(claim_path, claim, status="succeeded", instance_id=99)
    destroyed = False
    invoked: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...]) -> str:
        nonlocal destroyed
        invoked.append(command)
        if command[:3] == ("vastai", "destroy", "instance"):
            destroyed = True
            return ""
        if command[:4] == ("vastai", "--raw", "show", "instance"):
            return "{}" if destroyed else json.dumps(_healthy_instance())
        if command[-1] == "instances":
            return "[]" if destroyed else json.dumps([_healthy_instance()])
        if command[-1] == "volumes":
            return "[]"
        return type("Result", (), {"returncode": 0, "stdout": ""})()

    with pytest.raises(RuntimeError, match="instance disposed"):
        LIFECYCLE.stage_launch_sync_evaluation(
            manifest, instance, lifecycle_root=tmp_path / "life", runner=runner,
            code_bundle=bundle, token_file=token,
        )
    assert sum(command[:3] == ("vastai", "destroy", "instance") for command in invoked) == 1
    assert not claim_path.exists()


def test_disposal_absence_timeout_records_failure_after_exactly_one_destroy(tmp_path: Path) -> None:
    invocation = json.loads(_manifest(tmp_path).read_text())["invocation"]
    release_id, remote_prefix = _evaluation_release(invocation)
    instance = _write(tmp_path / "instance.json", {
        "kind": "groot_checkpoint_evaluation_instance", "instance_id": 99,
        "rent_claim_path": str(LIFECYCLE.EVALUATION_RENT_CLAIM_ROOT / "active.json"),
        "invocation_sha256": LIFECYCLE.canonical_sha256(invocation),
    })
    publication = _write(tmp_path / "publication.json", {
        "kind": "groot_checkpoint_evaluation_publication", "disposable": True,
        "instance_id": 99, "instance_receipt_sha256": hashlib.sha256(instance.read_bytes()).hexdigest(),
        "invocation": invocation, "invocation_sha256": LIFECYCLE.canonical_sha256(invocation),
        "immutable_revision": "a" * 40, "repository": "ryanjin333/lehome-groot-n17-data",
        "release_id": release_id, "remote_prefix": remote_prefix,
        "repository_private": True, "tree_listing_verified": True, "fresh_readback_verified": True,
    })
    invoked: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...]) -> str:
        invoked.append(command)
        if command[:3] == ("vastai", "destroy", "instance"):
            return ""
        if command[:4] == ("vastai", "--raw", "show", "instance"):
            return json.dumps(_healthy_instance())
        if command[-1] == "instances":
            return json.dumps([_healthy_instance()])
        if command[-1] == "volumes":
            return "[]"
        raise AssertionError(command)

    disposal = tmp_path / "disposal-failed.json"
    with pytest.raises(ValueError, match="did not empty"):
        LIFECYCLE.destroy_after_publication(
            99, publication, instance, disposal_receipt=disposal,
            runner=runner, absence_polls=1,
        )
    assert sum(command[:3] == ("vastai", "destroy", "instance") for command in invoked) == 1
    assert json.loads(disposal.read_text())["absence_unverified"] is True


def test_comparator_only_allows_checkpoint_step_and_artifact_hash(tmp_path: Path) -> None:
    one = json.loads(_manifest(tmp_path).read_text())["invocation"]
    two = {**one, "policy_step": 2000, "policy_artifact_sha256": "1" * 64}
    assert LIFECYCLE.compare_checkpoint_invocations(one, two) == {}
    two["code_revision"] = "2" * 40
    assert LIFECYCLE.compare_checkpoint_invocations(one, two) == {"code_revision": (one["code_revision"], two["code_revision"])}
