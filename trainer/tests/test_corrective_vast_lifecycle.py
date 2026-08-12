from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import subprocess

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "source" / "lehome"))
SPEC = importlib.util.spec_from_file_location("corrective_vast_lifecycle_under_test", REPOSITORY / "scripts" / "run_groot_corrective_vast_lifecycle.py")
assert SPEC is not None and SPEC.loader is not None
LIFECYCLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LIFECYCLE
SPEC.loader.exec_module(LIFECYCLE)
CAMPAIGN_SPEC = importlib.util.spec_from_file_location("corrective_campaign_for_lifecycle_test", REPOSITORY / "scripts" / "run_groot_corrective_campaign.py")
assert CAMPAIGN_SPEC is not None and CAMPAIGN_SPEC.loader is not None
CAMPAIGN = importlib.util.module_from_spec(CAMPAIGN_SPEC)
sys.modules[CAMPAIGN_SPEC.name] = CAMPAIGN
CAMPAIGN_SPEC.loader.exec_module(CAMPAIGN)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _canary_instance() -> dict[str, object]:
    return {"schema_version": 1, "kind": "corrective_vast_instance", "instance_id": 9,
            "host": "host", "port": 22, "wave_index": 0,
            "provider_response_sha256": "c" * 64, "provider_evidence_sha256": "d" * 64}


def _manifest(tmp_path: Path) -> Path:
    path = tmp_path / "wave.json"
    baseline = {
        "rollout_image": LIFECYCLE.APPROVED_IMAGE_REPOSITORY + "@" + LIFECYCLE.APPROVED_IMAGE_DIGEST, "image_identity": LIFECYCLE.APPROVED_IMAGE_DIGEST,
        "policy_path": "/local/policy", "policy_revision_file": "/local/revision.json",
        "release_assets_root": "/local/assets", "groot_root": "/local/groot",
        "groot_python": "/local/groot/python", "controller_python": "/local/isaac/python",
    }
    attempts = []
    for index in range(4):
        attempts.append({"attempt_id": f"attempt-{index}", "worker_slot": index, "episode_id": f"attempt-{index}", "command": ["/local/isaac/python", "scripts/run_groot_flywheel_trial.py", "--policy-path", "/local/policy", "--policy-revision-file", "/local/revision.json", "--release-assets-root", "/local/assets", "--groot-root", "/local/groot", "--groot-python", "/local/groot/python", "--output-root", str(tmp_path / "campaign"), "--policy-server-log", str(tmp_path / "campaign" / "workers" / f"worker-{index}.log")]})
    _write(path, {"schema_version": 1, "kind": "corrective_rft_wave", "wave_index": 0, "baseline": baseline, "provider": {"rental_kind": "on-demand", "instance_hourly_cost_usd": 1.0, "account_hourly_total_usd": 1.5, "offer_id": 7, "gpu_name": "RTX 3090", "num_gpus": 4}, "provider_evidence": {"evidence_id": "fresh"}, "attempts": attempts})
    return path


def _bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "approved.tar"
    bundle.write_bytes(b"bundle")
    digest = __import__("hashlib").sha256(bundle.read_bytes()).hexdigest()
    (tmp_path / "approved.tar.sha256").write_text(digest + "\n")
    _write(bundle.with_name(bundle.name + ".manifest.json"), {
        "schema_version": 1, "kind": "corrective_remote_bundle", "bundle_sha256": digest,
        "paths": {"policy_path": "policy", "policy_revision_file": "revision.json", "release_assets_root": "assets", "groot_root": "groot", "groot_python": "groot/python", "controller_python": "isaac/python", "output_root": "campaign", "trial_script": "scripts/run_groot_flywheel_trial.py"},
    })
    return bundle


def test_capture_offer_evidence_binds_live_offer_and_account_snapshot(tmp_path: Path) -> None:
    evidence = LIFECYCLE.capture_offer_evidence(
        offers=[{"id": 8, "gpu_name": "RTX 3090", "num_gpus": 4, "dph_total": 1.1, "cpu_cores_effective": 64, "cpu_ram": 131072, "is_bid": False}, {"id": 7, "gpu_name": "RTX 3090", "num_gpus": 4, "dph_total": 1.0, "cpu_cores_effective": 64, "cpu_ram": 131072, "is_bid": False}],
        instances=[{"id": 99, "dph_total": 0.5}], output=tmp_path / "offer.json", now_unix=100, ttl_seconds=60,
    )
    assert evidence["account_hourly_total_usd"] == 1.5
    assert evidence["expires_at_unix"] == 160
    assert (tmp_path / evidence["source_snapshot_path"]).is_file()
    assert evidence["offer_id"] == 7
    assert "cpu_ram>=128" in LIFECYCLE.OFFER_QUERY


def test_cli_query_and_rent_readback_rejects_spoofed_or_mismatched_instance(tmp_path: Path) -> None:
    commands = []
    destroyed = [False]
    def runner(command):
        commands.append(command)
        if command[:4] == ("vastai", "--raw", "show", "instances"):
            return "[]"
        if command[:4] == ("vastai", "--raw", "show", "volumes"):
            return "[]"
        if command[:4] == ("vastai", "--raw", "search", "offers"):
            return '[{"id":8,"gpu_name":"RTX 3090","num_gpus":4,"dph_total":1.1,"cpu_cores_effective":64,"cpu_ram":131072,"is_bid":false},{"id":7,"gpu_name":"RTX 3090","num_gpus":4,"dph_total":1.0,"cpu_cores_effective":64,"cpu_ram":131072,"is_bid":false}]'
        if command[:4] == ("vastai", "--raw", "create", "instance"):
            return '{"new_contract":99}'
        if command[:4] == ("vastai", "--raw", "show", "instance") and destroyed[0]:
            return "{}"
        if command[:4] == ("vastai", "--raw", "show", "instance"):
            return '{"id":99,"actual_status":"running","ssh_host":"host","ssh_port":22,"gpu_name":"A100","num_gpus":4,"cpu_cores_effective":64,"cpu_ram":131072,"driver_version":"550.54","dph_total":1.0,"is_bid":false}'
        if command[:3] == ("vastai", "destroy", "instance"):
            destroyed[0] = True
            return ""
        raise AssertionError(command)
    with pytest.raises(ValueError, match="readback"):
        LIFECYCLE.rent_wave(_manifest(tmp_path), lifecycle_root=tmp_path / "life", runner=runner, now_unix=100)
    assert ("vastai", "--raw", "search", "offers", LIFECYCLE.OFFER_QUERY, "--on-demand", "--storage", "300") in commands
    assert "driver_version" not in LIFECYCLE.OFFER_QUERY
    create = next(command for command in commands if command[:4] == ("vastai", "--raw", "create", "instance"))
    assert create[-3:] == ("--ssh", "--direct", "--cancel-unavail")
    assert create[create.index("--env") + 1] == "-e LEHOME_FLYWHEEL_IMAGE_IDENTITY=" + LIFECYCLE.APPROVED_IMAGE_DIGEST


def test_remote_stage_launch_and_destroy_require_live_receipts(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    bundle = _bundle(tmp_path)
    token = tmp_path / "token"; token.write_text("secret")
    invoked = []
    def runner(command):
        invoked.append(command); return ""
    instance = {"schema_version": 1, "kind": "corrective_vast_instance", "instance_id": 99, "host": "host", "port": 22, "wave_index": 0, "provider_response_sha256": "b" * 64}
    with pytest.raises(ValueError, match="baseline"):
        LIFECYCLE.remote_launch_wave(manifest, instance, lifecycle_root=tmp_path / "life", runner=runner, code_bundle=bundle, token_file=token)
    assert not invoked
    disposal = tmp_path / "disposal.json"
    _write(disposal, {"schema_version": 1, "disposable": True, "immutable_revision": "c" * 40, "fresh_readback_verified": True, "tree_listing_verified": True})
    with pytest.raises(ValueError, match="lifecycle"):
        LIFECYCLE.destroy_after_publication(99, disposal, tmp_path / "missing.json", runner=runner)


def test_capture_rejects_bid_and_stale_spend_and_deterministically_prefers_manifest_offer(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="4xRTX3090"):
        LIFECYCLE.capture_offer_evidence(offers=[{"id": 7, "gpu_name": "RTX 3090", "num_gpus": 4, "dph_total": 1.0, "is_bid": True}], instances=[], output=tmp_path / "bid.json", now_unix=1, ttl_seconds=30)
    evidence = LIFECYCLE.capture_offer_evidence(offers=[{"id": 9, "gpu_name": "RTX 3090", "num_gpus": 4, "dph_total": 0.9, "cpu_cores": 64, "ram": 128, "is_bid": False}, {"id": 7, "gpu_name": "RTX 3090", "num_gpus": 4, "dph_total": 1.0, "cpu_cores": 64, "ram": 128, "is_bid": False}], instances=[], output=tmp_path / "preferred.json", now_unix=1, ttl_seconds=30, preferred_offer_id=7)
    assert evidence["offer_id"] == 7


def test_preferred_over_cap_falls_back_to_compatible_under_cap(tmp_path: Path) -> None:
    evidence = LIFECYCLE.capture_offer_evidence(
        offers=[{"id": 7, "gpu_name": "RTX 3090", "num_gpus": 4, "dph_total": 1.8, "cpu_cores": 64, "ram": 128, "is_bid": False}, {"id": 8, "gpu_name": "RTX 3090", "num_gpus": 4, "dph_total": 0.8, "cpu_cores": 64, "ram": 128, "is_bid": False}],
        instances=[{"dph_total": 0.5}], output=tmp_path / "fallback.json", now_unix=1, ttl_seconds=30, preferred_offer_id=7,
    )
    assert evidence["offer_id"] == 8


def test_capture_rejects_actual_vast_capacity_below_four_worker_floor(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="4xRTX3090"):
        LIFECYCLE.capture_offer_evidence(
            offers=[{"id": 7, "gpu_name": "RTX 3090", "num_gpus": 4, "dph_total": 0.7, "cpu_cores_effective": 63, "cpu_ram": 131072, "is_bid": False}],
            instances=[], output=tmp_path / "cores.json", now_unix=1, ttl_seconds=30,
        )
    with pytest.raises(ValueError, match="4xRTX3090"):
        LIFECYCLE.capture_offer_evidence(
            offers=[{"id": 7, "gpu_name": "RTX 3090", "num_gpus": 4, "dph_total": 0.7, "cpu_cores_effective": 64, "cpu_ram": 127999, "is_bid": False}],
            instances=[], output=tmp_path / "ram.json", now_unix=1, ttl_seconds=30,
        )


def test_rent_readback_failure_destroys_only_created_instance(tmp_path: Path) -> None:
    calls = []
    def runner(command):
        calls.append(command)
        if command[:4] == ("vastai", "--raw", "show", "instances"): return "[]"
        if command[:4] == ("vastai", "--raw", "show", "volumes"): return "[]"
        if command[:4] == ("vastai", "--raw", "search", "offers"): return '[{"id":7,"gpu_name":"RTX 3090","num_gpus":4,"dph_total":1.0,"cpu_cores":64,"ram":128,"is_bid":false}]'
        if command[:4] == ("vastai", "--raw", "create", "instance"): return '{"new_contract":99}'
        if command[:4] == ("vastai", "--raw", "show", "instance"): return '{}'
        if command[:3] == ("vastai", "destroy", "instance"): return ""
        raise AssertionError(command)
    with pytest.raises(ValueError, match="SSH-ready"):
        LIFECYCLE.rent_wave(_manifest(tmp_path), lifecycle_root=tmp_path / "life", runner=runner, now_unix=100, sleep=lambda _: None)
    assert ("vastai", "destroy", "instance", "99") in calls


def test_remote_bundle_extracts_checkout_and_rewrites_every_local_runtime_path(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    bundle = _bundle(tmp_path)
    token = tmp_path / "token"; token.write_text("secret")
    invoked = []
    def runner(command):
        invoked.append(command)
        return ""
    instance = {"schema_version": 1, "kind": "corrective_vast_instance", "instance_id": 99, "host": "host", "port": 22, "wave_index": 0}
    with pytest.raises(ValueError, match="baseline"):
        LIFECYCLE.remote_launch_wave(manifest, instance, lifecycle_root=tmp_path / "life", runner=runner, code_bundle=bundle, token_file=token)
    assert not invoked


def test_remote_terminal_rejects_nonzero_worker_or_missing_raw_receipt(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    bundle = _bundle(tmp_path)
    token = tmp_path / "token"; token.write_text("secret")
    instance = {"schema_version": 1, "kind": "corrective_vast_instance", "instance_id": 99, "host": "host", "port": 22, "wave_index": 0}
    life = tmp_path / "life"
    remote = life / "synced-wave-000000"
    remote.mkdir(parents=True)
    _write(remote / "remote-terminal.json", {"schema_version": 1, "kind": "corrective_remote_terminal", "workers": [{"worker_slot": index, "attempt_id": f"attempt-{index}", "returncode": 1 if index == 2 else 0, "raw_receipt_path": f"raw/attempt-{index}/SHA256SUMS.json", "raw_receipt_sha256": "b" * 64} for index in range(4)]})
    with pytest.raises(ValueError, match="baseline"):
        LIFECYCLE.remote_launch_wave(manifest, instance, lifecycle_root=life, runner=lambda _: "", code_bundle=bundle, token_file=token)
    _write(remote / "remote-terminal.json", {"schema_version": 1, "kind": "corrective_remote_terminal", "workers": [{"worker_slot": index, "attempt_id": f"attempt-{index}", "returncode": 0, "raw_receipt_path": f"raw/attempt-{index}/SHA256SUMS.json", "raw_receipt_sha256": "b" * 64} for index in range(4)]})
    with pytest.raises(ValueError, match="baseline"):
        LIFECYCLE.remote_launch_wave(manifest, instance, lifecycle_root=life, runner=lambda _: "", code_bundle=bundle, token_file=token)


def test_remote_rewrite_accepts_actual_campaign_wave_manifest_without_baseline_output_root(tmp_path: Path) -> None:
    baseline = {
        "parent_checkpoint_repository": "repo", "parent_checkpoint_revision": "r", "parent_checkpoint_artifact_sha256": "a" * 64,
        "parent_checkpoint_step": 12000, "code_revision": "c" * 40, "asset_revision": "d" * 40,
        "image_identity": "sha256:" + "e" * 64, "simulator_version": "5.1", "policy_path": "/policy",
        "policy_revision_file": "/revision", "release_assets_root": "/assets", "groot_root": "/groot",
        "groot_revision": "f" * 40, "groot_python": "/groot/python", "controller_python": "/isaac/python", "rollout_image": "ghcr.io/ryanjin333/lehome-rollout@sha256:" + "e" * 64,
    }
    wave = CAMPAIGN._wave_manifest(tmp_path / "campaign", wave_index=0, categories=("top_long", "top_short", "pant_long", "pant_short"), baseline=baseline, provider={"rental_kind": "on-demand"}, provider_evidence={})
    root = LIFECYCLE._manifest_output_root(wave["attempts"])
    assert root == str(tmp_path / "campaign")
    rewritten = LIFECYCLE._remote_command(wave["attempts"][0]["command"], baseline, {"policy_path": "policy", "policy_revision_file": "revision", "release_assets_root": "assets", "groot_root": "groot", "groot_python": "python", "controller_python": "isaac/python", "output_root": "campaign", "trial_script": "scripts/run_groot_flywheel_trial.py"}, "/remote/checkout", root)
    assert str(tmp_path / "campaign") not in rewritten
    assert any(token.startswith("/remote/checkout/campaign/workers") for token in rewritten)


def test_free_bundle_builder_creates_safe_tar_and_rejects_missing_source(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    baseline = payload["baseline"]
    paths = {"policy_path": tmp_path / "policy", "policy_revision_file": tmp_path / "revision.json", "release_assets_root": tmp_path / "assets", "groot_root": tmp_path / "groot", "groot_python": tmp_path / "groot-python", "controller_python": tmp_path / "controller-python"}
    baseline.update({key: str(value) for key, value in paths.items()})
    for attempt in payload["attempts"]:
        attempt["command"] = [str(paths["groot_python"]) if item == "/local/groot/python" else str(paths["policy_path"]) if item == "/local/policy" else str(paths["policy_revision_file"]) if item == "/local/revision.json" else str(paths["release_assets_root"]) if item == "/local/assets" else str(paths["groot_root"]) if item == "/local/groot" else item for item in attempt["command"]]
    _write(manifest, payload)
    for key in ("policy_path", "policy_revision_file", "release_assets_root", "groot_root", "groot_python", "controller_python"):
        path = Path(baseline[key]); path.parent.mkdir(parents=True, exist_ok=True)
        if key in {"policy_path", "release_assets_root", "groot_root"}:
            path.mkdir(exist_ok=True)
            (path / "data").write_text("ok")
        else:
            path.write_text("ok")
    checkout = tmp_path / "checkout"; (checkout / "scripts").mkdir(parents=True); (checkout / "scripts" / "run_groot_flywheel_trial.py").write_text("print('ok')")
    bundle = tmp_path / "bundle.tar"
    receipt = LIFECYCLE.build_approved_bundle(manifest, checkout=checkout, output=bundle)
    assert receipt["bundle_sha256"] == __import__("hashlib").sha256(bundle.read_bytes()).hexdigest()
    with tarfile.open(bundle) as archive:
        assert all(".." not in Path(member.name).parts and not member.issym() for member in archive.getmembers())
    (Path(baseline["policy_revision_file"])).unlink()
    with pytest.raises(ValueError, match="missing"):
        LIFECYCLE.build_approved_bundle(manifest, checkout=checkout, output=tmp_path / "missing.tar")


def test_canary_disposal_never_accepts_150_release_receipt(tmp_path: Path) -> None:
    publication = tmp_path / "canary.json"
    _write(publication, {"schema_version": 1, "kind": "corrective_campaign_publication", "disposable": True, "immutable_revision": "a" * 40, "fresh_readback_verified": True, "tree_listing_verified": True})
    lifecycle = tmp_path / "instance.json"; _write(lifecycle, {"kind": "corrective_vast_instance", "instance_id": 9})
    with pytest.raises(ValueError, match="canary"):
        LIFECYCLE.destroy_after_canary_publication(9, publication, lifecycle, canary_attempt_id="attempt-0", runner=lambda _: "{}")


def test_canary_disposal_accepts_publisher_receipt_schema(tmp_path: Path) -> None:
    publication = tmp_path / "canary.json"
    _write(publication, {"schema_version": 1, "kind": "corrective_rft_private_canary", "repository": "private/repo", "immutable_revision": "a" * 40, "remote_prefix": "corrective-rft-canary/x", "attempt_id": "attempt-0", "episode_id": "attempt-0", "instance_id": 9, "entry_count": 3, "repository_private": True, "tree_listing_verified": True, "fresh_readback_verified": True, "training_admission": False, "disposable": True})
    lifecycle = tmp_path / "instance.json"; _write(lifecycle, {"kind": "corrective_vast_instance", "instance_id": 9})
    calls = []
    def runner(command):
        calls.append(command)
        return "{}"
    assert LIFECYCLE.destroy_after_canary_publication(9, publication, lifecycle, canary_attempt_id="attempt-0", runner=runner)
    assert calls[0] == ("vastai", "destroy", "instance", "9")


def test_canary_disposal_rejects_publication_for_another_instance(tmp_path: Path) -> None:
    publication = tmp_path / "canary.json"
    _write(publication, {"schema_version": 1, "kind": "corrective_rft_private_canary", "immutable_revision": "a" * 40, "attempt_id": "attempt-0", "instance_id": 10, "repository_private": True, "tree_listing_verified": True, "fresh_readback_verified": True, "training_admission": False, "disposable": True})
    lifecycle = tmp_path / "instance.json"; _write(lifecycle, {"kind": "corrective_vast_instance", "instance_id": 9})
    with pytest.raises(ValueError, match="canary"):
        LIFECYCLE.destroy_after_canary_publication(9, publication, lifecycle, canary_attempt_id="attempt-0", runner=lambda _: "{}")


def test_literal_canary_cli_preflights_image_native_runtime_and_syncs_abort(tmp_path: Path) -> None:
    canary = tmp_path / "canary.json"
    _write(canary, {"schema_version": 1, "kind": "corrective_rft_canary", "wave_index": 0, "episode_count": 1, "baseline": {"code_revision": "c" * 40, "parent_checkpoint_revision": "a" * 40, "parent_checkpoint_artifact_sha256": "b" * 64, "controller_python": "/opt/lehome-challenge/.venv/bin/python", "groot_root": LIFECYCLE.APPROVED_GROOT_ROOT, "groot_revision": LIFECYCLE.APPROVED_GROOT_REVISION, "groot_python": LIFECYCLE.APPROVED_GROOT_PYTHON, "policy_path": "/local/policy", "policy_revision_file": "/local/revision", "release_assets_root": "/local/assets", "image_identity": "sha256:" + "a" * 64}, "provider": {}, "attempt": {"attempt_id": "attempt-0", "worker_slot": 0, "command": ["/opt/lehome-challenge/.venv/bin/python", "scripts/run_groot_flywheel_trial.py", "--policy-path", "/local/policy", "--policy-revision-file", "/local/revision", "--release-assets-root", "/local/assets", "--groot-root", LIFECYCLE.APPROVED_GROOT_ROOT, "--groot-python", LIFECYCLE.APPROVED_GROOT_PYTHON, "--output-root", str(tmp_path / "output")]}})
    instance = _canary_instance()
    calls = []
    def runner(command):
        calls.append(command)
        return type("Result", (), {"returncode": 0 if command[0] in {"ssh", "scp"} and command[-3:-1] != ("sh", "-lc") else 1, "stdout": ""})()
    bundle = tmp_path / "code.bundle"; bundle.write_bytes(b"git bundle")
    token = tmp_path / "token"; token.write_text("secret")
    result = LIFECYCLE.remote_launch_canary(canary, instance, lifecycle_root=tmp_path / "life", runner=runner, bundle=bundle, token_file=token)
    assert result["kind"] == "corrective_canary_abort"
    assert Path(result["synced_evidence_root"]).is_dir()
    script = next(command[-1] for command in calls if command[0] == "ssh" and command[-3:-1] == ("sh", "-lc"))
    assert LIFECYCLE.APPROVED_GROOT_ROOT in script and "lfs pull" in script
    assert "ryanjin333/lehome-groot-n17-models" in script and "Cosmos-Reason2-2B" in script
    assert "policy_artifact_sha256" in script and "/opt/lehome-challenge/.venv/bin/hf download" in script
    assert "Assets/$d" in script and "LEHOME_FLYWHEEL_WORKER_GPU=0" in script
    assert "if [ -e /workspace/lehome-release-assets ]" in script
    assert "lfs ls-files --long" in script and "diff --quiet" in script
    assert "trap 'rm -f /workspace/corrective/canary-000000/hf.token; unset HF_TOKEN' EXIT" in script
    scp_sources = [command[-2] for command in calls if command[0] == "scp" and "canary-000000" in command[-2]]
    assert any(source.endswith("/code/campaign/.") for source in scp_sources)
    assert not any(source.endswith("/canary-000000/.") for source in scp_sources)


def test_canary_cli_actions_are_exposed() -> None:
    actions = LIFECYCLE.build_parser()._subparsers._group_actions[0].choices
    assert {"canary-launch", "canary-destroy", "build-bundle", "build-code-bundle", "ingest"} <= set(actions)


def test_canary_early_staging_failure_writes_nonempty_abort_evidence(tmp_path: Path) -> None:
    canary = tmp_path / "canary.json"
    _write(canary, {"schema_version": 1, "kind": "corrective_rft_canary", "wave_index": 0, "episode_count": 1, "baseline": {"code_revision": "c" * 40, "parent_checkpoint_revision": "a" * 40, "parent_checkpoint_artifact_sha256": "b" * 64, "controller_python": "/opt/lehome-challenge/.venv/bin/python", "groot_root": LIFECYCLE.APPROVED_GROOT_ROOT, "groot_revision": LIFECYCLE.APPROVED_GROOT_REVISION, "groot_python": LIFECYCLE.APPROVED_GROOT_PYTHON, "policy_path": "/local/policy", "policy_revision_file": "/local/revision", "release_assets_root": "/local/assets", "image_identity": LIFECYCLE.APPROVED_IMAGE_DIGEST}, "provider": {}, "attempt": {"attempt_id": "attempt-0", "worker_slot": 0, "command": ["/opt/lehome-challenge/.venv/bin/python", "--output-root", str(tmp_path / "out")]}})
    bundle = tmp_path / "code.bundle"; bundle.write_bytes(b"bundle")
    token = tmp_path / "token"; token.write_text("secret")
    result = LIFECYCLE.remote_launch_canary(canary, _canary_instance(), lifecycle_root=tmp_path / "life", runner=lambda _: type("Result", (), {"returncode": 1})(), bundle=bundle, token_file=token)
    evidence = Path(result["abort_evidence_root"])
    assert result["kind"] == "corrective_canary_abort" and (evidence / "setup.json").is_file()
    assert result["synced_evidence_sha256"] == LIFECYCLE._evidence_root_sha256(evidence)
    assert result["transport_returncode"] == 1 and len(result["canary_manifest_sha256"]) == len(result["staged_bundle_sha256"]) == 64


def test_canary_retry_writes_append_only_abort_evidence_and_materializes_wrapper(tmp_path: Path) -> None:
    canary = tmp_path / "canary.json"
    _write(canary, {"schema_version": 1, "kind": "corrective_rft_canary", "wave_index": 0, "episode_count": 1, "baseline": _image_native_baseline(), "provider": {}, "attempt": _canary_attempt(tmp_path, seed=11)})
    bundle = tmp_path / "code.bundle"; bundle.write_bytes(b"bundle")
    token = tmp_path / "token"; token.write_text("secret")
    runner = lambda _: type("Result", (), {"returncode": 1, "stdout": ""})()
    first = LIFECYCLE.remote_launch_canary(canary, _canary_instance(), lifecycle_root=tmp_path / "life", runner=runner, bundle=bundle, token_file=token)
    second = LIFECYCLE.remote_launch_canary(canary, _canary_instance(), lifecycle_root=tmp_path / "life", runner=runner, bundle=bundle, token_file=token)
    assert first["retry_id"] != second["retry_id"] and first["abort_evidence_root"] != second["abort_evidence_root"]
    wrapper = "\n".join(LIFECYCLE._groot_wrapper_setup())
    assert LIFECYCLE.APPROVED_GROOT_NATIVE_PYTHON in wrapper
    assert LIFECYCLE.APPROVED_GROOT_PYTHON_SHA256 in wrapper
    assert "PYTHONPATH=/opt/gr00t-runtime/lib/python3.10/site-packages:/opt/isaac-groot" in wrapper


def test_canary_rejects_unbound_or_wrong_wave_instance_before_staging(tmp_path: Path) -> None:
    canary = tmp_path / "canary.json"
    _write(canary, {"schema_version": 1, "kind": "corrective_rft_canary", "wave_index": 0, "episode_count": 1, "baseline": _image_native_baseline(), "provider": {}, "attempt": _canary_attempt(tmp_path, seed=11)})
    bundle = tmp_path / "code.bundle"; bundle.write_bytes(b"bundle")
    token = tmp_path / "token"; token.write_text("secret")
    instance = {**_canary_instance(), "wave_index": 1}
    with pytest.raises(ValueError, match="lifecycle"):
        LIFECYCLE.remote_launch_canary(canary, instance, lifecycle_root=tmp_path / "life", runner=lambda _: pytest.fail("must not stage"), bundle=bundle, token_file=token)


def test_campaign_sync_failure_returns_one_stable_publishable_abort_root(tmp_path: Path) -> None:
    baseline = _image_native_baseline()
    attempt = _canary_attempt(tmp_path, seed=11)
    canary = tmp_path / "canary.json"
    _write(canary, {"schema_version": 1, "kind": "corrective_rft_canary", "wave_index": 0, "episode_count": 1, "baseline": baseline, "provider": {}, "attempt": attempt})
    bundle = tmp_path / "code.bundle"; bundle.write_bytes(b"bundle")
    token = tmp_path / "token"; token.write_text("secret")

    def runner(command: tuple[str, ...]):
        if command[0] == "scp" and command[-2].endswith("/code/campaign/."):
            return type("Result", (), {"returncode": 1, "stdout": ""})()
        if command[0] == "scp" and command[-2].endswith("/canary.returncode"):
            Path(command[-1]).parent.mkdir(parents=True, exist_ok=True)
            Path(command[-1]).write_text("1\n")
        return type("Result", (), {"returncode": 0, "stdout": ""})()

    result = LIFECYCLE.remote_launch_canary(canary, _canary_instance(), lifecycle_root=tmp_path / "life", runner=runner, bundle=bundle, token_file=token)
    evidence = Path(result["synced_evidence_root"])
    assert result["kind"] == "corrective_canary_abort" and result["abort_evidence_root"] == str(evidence)
    assert (evidence / "setup.json").is_file() and (evidence / "transport.json").is_file() and (evidence / "canary.returncode").read_text() == "1\n"
    assert result["synced_evidence_sha256"] == LIFECYCLE._evidence_root_sha256(evidence)


def test_rc0_failed_canary_emits_publishable_non_training_abort(tmp_path: Path) -> None:
    baseline = _image_native_baseline()
    attempt = _canary_attempt(tmp_path, seed=11)
    canary = tmp_path / "canary.json"
    _write(canary, {"schema_version": 1, "kind": "corrective_rft_canary", "wave_index": 0, "episode_count": 1, "baseline": baseline, "provider": {}, "attempt": attempt})
    bundle = tmp_path / "code.bundle"; bundle.write_bytes(b"bundle")
    token = tmp_path / "token"; token.write_text("secret")
    life = tmp_path / "life"; sync = life / "canary-000000-sync"

    def runner(command: tuple[str, ...]):
        if command[0] == "scp" and command[-2].endswith("/code/campaign/."):
            _failed_canary_sync(sync, attempt, baseline)
        return type("Result", (), {"returncode": 0, "stdout": ""})()

    result = LIFECYCLE.remote_launch_canary(canary, _canary_instance(), lifecycle_root=life, runner=runner, bundle=bundle, token_file=token)
    assert result["kind"] == "corrective_canary_non_training_abort"
    assert result["schema_version"] == 1 and result["non_training_admitted"] is False
    assert result["transport_returncode"] == 0 and result["attempt_id"] == "attempt-0" and result["instance_id"] == 9
    assert all(len(result[key]) == 64 for key in ("canary_manifest_sha256", "staged_bundle_sha256", "synced_evidence_sha256", "raw_manifest_sha256", "policy_receipt_sha256"))


def _image_native_baseline() -> dict[str, object]:
    return {
        "code_revision": "c" * 40, "parent_checkpoint_revision": "a" * 40,
        "parent_checkpoint_artifact_sha256": "b" * 64,
        "controller_python": "/opt/lehome-challenge/.venv/bin/python",
        "groot_root": LIFECYCLE.APPROVED_GROOT_ROOT,
        "groot_revision": LIFECYCLE.APPROVED_GROOT_REVISION,
        "groot_python": LIFECYCLE.APPROVED_GROOT_PYTHON,
        "policy_path": "/local/policy", "policy_revision_file": "/local/revision",
        "release_assets_root": "/local/assets", "image_identity": LIFECYCLE.APPROVED_IMAGE_DIGEST,
    }


def _canary_attempt(tmp_path: Path, *, seed: int) -> dict[str, object]:
    return {"attempt_id": "attempt-0", "episode_id": "attempt-0", "worker_slot": 0, "seed": seed,
            "command": ["/opt/lehome-challenge/.venv/bin/python", "scripts/run_groot_flywheel_trial.py", "--policy-path", "/local/policy", "--policy-revision-file", "/local/revision", "--release-assets-root", "/local/assets", "--groot-root", LIFECYCLE.APPROVED_GROOT_ROOT, "--groot-python", LIFECYCLE.APPROVED_GROOT_PYTHON, "--output-root", str(tmp_path / "campaign")]}


def _failed_canary_sync(sync: Path, attempt: dict[str, object], baseline: dict[str, object]) -> None:
    import hashlib
    raw = sync / "raw" / "attempt-0"
    _write(raw / "episode.json", {"episode_id": "attempt-0", "outcome": "failure", "accepted_success": False})
    _write(raw / "SHA256SUMS.json", {"episode.json": {"sha256": hashlib.sha256((raw / "episode.json").read_bytes()).hexdigest(), "size": (raw / "episode.json").stat().st_size}})
    policy = "/workspace/checkpoints/lehome-groot-n17-models-" + str(baseline["parent_checkpoint_revision"]) + "/policies/step-12000"
    _write(sync / "policy-server-receipt-attempt-0.json", {"episode_id": "attempt-0", "backend": "policy_server", "checkpoint_revision": baseline["parent_checkpoint_revision"], "checkpoint_digest": baseline["parent_checkpoint_artifact_sha256"], "code_revision": baseline["code_revision"], "image_identity": baseline["image_identity"], "policy_device": "cuda:0", "parity_stage": "server_cpu", "simulator_device": "cpu", "groot_revision": baseline["groot_revision"], "python_path": baseline["groot_python"], "policy_seed": attempt["seed"], "port": 9100, "command": ["--model-path", policy]})


def test_build_code_bundle_requires_a_clean_exact_commit(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"; checkout.mkdir()
    subprocess.run(("git", "init"), cwd=checkout, check=True, capture_output=True)
    subprocess.run(("git", "config", "user.email", "test@example.invalid"), cwd=checkout, check=True)
    subprocess.run(("git", "config", "user.name", "Test"), cwd=checkout, check=True)
    (checkout / "tracked.txt").write_text("ok")
    subprocess.run(("git", "add", "tracked.txt"), cwd=checkout, check=True); subprocess.run(("git", "commit", "-m", "test"), cwd=checkout, check=True, capture_output=True)
    revision = subprocess.run(("git", "rev-parse", "HEAD"), cwd=checkout, check=True, capture_output=True, text=True).stdout.strip()
    bundle = tmp_path / "code.bundle"
    receipt = LIFECYCLE.build_code_bundle(checkout=checkout, revision=revision, output=bundle)
    assert receipt["revision"] == revision
    subprocess.run(("git", "bundle", "verify", str(bundle)), cwd=checkout, check=True, capture_output=True)
    (checkout / "dirty.txt").write_text("no")
    with pytest.raises(ValueError, match="clean"):
        LIFECYCLE.build_code_bundle(checkout=checkout, revision=revision, output=tmp_path / "dirty.bundle")


def test_full_wave_image_native_interface_succeeds_with_canonical_synced_evidence(tmp_path: Path) -> None:
    import hashlib
    campaign = tmp_path / "campaign"
    baseline = {
        "rollout_image": LIFECYCLE.APPROVED_IMAGE_REPOSITORY + "@" + LIFECYCLE.APPROVED_IMAGE_DIGEST,
        "image_identity": LIFECYCLE.APPROVED_IMAGE_DIGEST, "controller_python": "/opt/lehome-challenge/.venv/bin/python",
        "groot_root": LIFECYCLE.APPROVED_GROOT_ROOT, "groot_revision": LIFECYCLE.APPROVED_GROOT_REVISION,
        "groot_python": LIFECYCLE.APPROVED_GROOT_PYTHON, "code_revision": "c" * 40,
        "parent_checkpoint_revision": "a" * 40, "parent_checkpoint_artifact_sha256": "b" * 64,
        "policy_path": "/local/policy", "policy_revision_file": "/local/revision", "release_assets_root": "/local/assets",
    }
    attempts = []
    for slot in range(4):
        attempt_id = f"attempt-{slot}"
        attempts.append({"attempt_id": attempt_id, "episode_id": attempt_id, "worker_slot": slot, "seed": 100 + slot, "command": ["/opt/lehome-challenge/.venv/bin/python", "scripts/run_groot_flywheel_trial.py", "--policy-path", "/local/policy", "--policy-revision-file", "/local/revision", "--release-assets-root", "/local/assets", "--groot-root", LIFECYCLE.APPROVED_GROOT_ROOT, "--groot-python", LIFECYCLE.APPROVED_GROOT_PYTHON, "--output-root", str(campaign), "--policy-server-log", str(campaign / "workers" / f"{slot}.log")]})
    manifest = tmp_path / "wave.json"
    _write(manifest, {"schema_version": 1, "kind": "corrective_rft_wave", "wave_index": 0, "baseline": baseline, "provider": {"rental_kind": "on-demand", "instance_hourly_cost_usd": .7, "account_hourly_total_usd": 1.5, "offer_id": 7, "gpu_name": "RTX 3090", "num_gpus": 4}, "attempts": attempts})
    code_bundle = tmp_path / "code.bundle"; code_bundle.write_bytes(b"bundle")
    token = tmp_path / "token"; token.write_text("secret")
    life = tmp_path / "life"; sync = life / "synced-wave-000000"
    calls = []
    def materialize_sync() -> None:
        workers = []
        policy_root = "/workspace/checkpoints/lehome-groot-n17-models-" + baseline["parent_checkpoint_revision"]
        for attempt in attempts:
            episode = sync / "raw" / attempt["attempt_id"]
            _write(episode / "episode.json", {"episode_id": attempt["attempt_id"]})
            sha = hashlib.sha256((episode / "episode.json").read_bytes()).hexdigest()
            _write(episode / "SHA256SUMS.json", {"episode.json": {"sha256": sha, "size": (episode / "episode.json").stat().st_size}})
            manifest_sha = hashlib.sha256((episode / "SHA256SUMS.json").read_bytes()).hexdigest()
            workers.append({"worker_slot": attempt["worker_slot"], "attempt_id": attempt["attempt_id"], "returncode": 0, "raw_receipt_path": f"raw/{attempt['attempt_id']}/SHA256SUMS.json", "raw_receipt_sha256": manifest_sha})
            _write(sync / f"policy-server-receipt-{attempt['attempt_id']}.json", {"episode_id": attempt["attempt_id"], "backend": "policy_server", "checkpoint_revision": baseline["parent_checkpoint_revision"], "checkpoint_digest": baseline["parent_checkpoint_artifact_sha256"], "code_revision": baseline["code_revision"], "image_identity": baseline["image_identity"], "policy_device": f"cuda:{attempt['worker_slot']}", "parity_stage": "server_cpu", "simulator_device": "cpu", "groot_revision": baseline["groot_revision"], "python_path": baseline["groot_python"], "policy_seed": attempt["seed"], "port": 9100 + attempt["worker_slot"], "command": ["--model-path", policy_root + "/policies/step-12000"]})
        _write(sync / "remote-terminal.json", {"schema_version": 1, "kind": "corrective_remote_terminal", "workers": workers})
    def runner(command):
        calls.append(command)
        if command[0] == "scp" and command[-2].endswith("/code/campaign/."):
            materialize_sync()
        return type("Result", (), {"returncode": 0, "stdout": ""})()
    instance = {"kind": "corrective_vast_instance", "instance_id": 9, "host": "host", "port": 22, "wave_index": 0}
    receipt = LIFECYCLE.remote_launch_wave(manifest, instance, lifecycle_root=life, runner=runner, code_bundle=code_bundle, token_file=token)
    script = next(command[-1] for command in calls if command[0] == "ssh" and command[-3:-1] == ("sh", "-lc"))
    assert receipt["status"] == "remote_terminal"
    assert "git clone --no-checkout" in script and "ryanjin333/lehome-groot-n17-models" in script
    assert "asset_challenge" in script and "lfs pull" in script
    assert "if [ -e /workspace/lehome-release-assets ]" in script
    assert "lfs ls-files --long" in script and "diff --quiet" in script
    assert LIFECYCLE.APPROVED_GROOT_ROOT in script and "/opt/lehome-challenge/.venv/bin/python" in script
    assert all(f"LEHOME_FLYWHEEL_WORKER_GPU={slot}" in script for slot in range(4))
    assert any(command[0] == "scp" and command[-2].endswith("/code/campaign/.") for command in calls)
