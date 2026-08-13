from pathlib import Path
import hashlib
import json

import pytest

import importlib.util
import sys

REPOSITORY = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("persistent_training_lifecycle_under_test", REPOSITORY / "scripts" / "run_groot_persistent_training.py")
assert SPEC is not None and SPEC.loader is not None
LIFECYCLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LIFECYCLE
SPEC.loader.exec_module(LIFECYCLE)


class FailRunner:
    def __call__(self, _command: tuple[str, ...]) -> str:
        raise AssertionError("provider must not be called")


class FakeHub:
    def list_tree(self, *, repository, revision, token):
        from lehome_train.hub import HubTreeEntry
        return (HubTreeEntry("prefix/checkpoints/step-1000.tar", "file"), HubTreeEntry("prefix/checkpoints/step-2000.tar", "file"))
    def download_files(self, *, repository, revision, destination, relative_paths, token, remote_prefix=None):
        for relative in relative_paths:
            target = destination / relative; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(b"artifact")
        return revision


def test_dry_run_never_calls_provider(tmp_path: Path) -> None:
    request = tmp_path / "request.json"
    request.write_text('{"generation_sha256":"' + "a" * 64 + '","config_sha256":"' + "b" * 64 + '"}')
    report = LIFECYCLE.main_for_test(["status", "--request", str(request)], runner=FailRunner())
    assert report["paid_action"] is False


def test_destroy_requires_two_immutable_checkpoints_bound_to_instance(tmp_path: Path) -> None:
    receipt = {"kind": "continuous_corrective_training_terminal", "instance_id": 7, "immutable_checkpoint_steps": [1000]}
    with pytest.raises(ValueError, match="instance-bound disposal"):
        LIFECYCLE.destroy(instance_id=7, training_receipt=receipt)


def test_destroy_requires_publications_bound_to_terminal_identity() -> None:
    receipt = {
        "kind": "continuous_corrective_training_terminal", "instance_id": 7,
        "generation_sha256": "a" * 64, "config_sha256": "b" * 64,
        "experiment_id": "persistent-001", "immutable_checkpoint_steps": [1000, 2000],
        "immutable_checkpoint_publications": [
            {"optimizer_step": step, "repository": "wrong/repo", "immutable_revision": "c" * 40, "remote_prefix": "prefix", "relative_path": f"step-{step}.tar", "artifact_sha256": "d" * 64, "artifact_byte_size": 1, "readback_verified": True, "generation_sha256": "a" * 64, "config_sha256": "b" * 64, "experiment_id": "persistent-001"}
            for step in (1000, 2000)
        ],
    }
    with pytest.raises(ValueError, match="approved model repository"):
        LIFECYCLE.destroy(instance_id=7, training_receipt=receipt)


def test_destroy_cli_reads_private_token_and_constructs_real_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = tmp_path / "hub.token"
    token.write_text("publisher-token\n", encoding="utf-8")
    token.chmod(0o600)
    request = tmp_path / "destroy.json"
    request.write_text(json.dumps({"instance_id": 7, "kind": "continuous_corrective_training_terminal", "immutable_checkpoint_steps": [1000, 2000], "immutable_checkpoint_publications": []}), encoding="utf-8")
    captured: dict[str, object] = {}
    monkeypatch.setattr(LIFECYCLE, "destroy", lambda **kwargs: captured.update(kwargs) or {"destroy_authorized": True})

    result = LIFECYCLE.main_for_test(["destroy", "--request", str(request), "--execute", "--token-file", str(token)], runner=FailRunner())

    assert result["destroy_authorized"] is True
    assert captured["token"] == "publisher-token"
    assert type(captured["transport"]).__name__ == "HuggingFaceHubTransport"


def test_capture_rent_and_destroy_use_injected_cli_and_fresh_readback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[tuple[str, ...]] = []
    def runner(command: tuple[str, ...]) -> str:
        commands.append(command)
        if command[:4] == ("vastai", "--raw", "search", "offers"):
            return '[{"id":7,"gpu_name":"RTX PRO 6000 WS","num_gpus":1,"gpu_ram":96000,"dph_total":0.7,"min_bid":0.5,"is_bid":false,"driver_version":"595.71.05","untrusted_token":"never-persist"}]'
        if command[:4] == ("vastai", "--raw", "show", "instances"): return "[]"
        if command[:4] == ("vastai", "--raw", "show", "volumes"): return "[]"
        if command[:4] == ("vastai", "--raw", "create", "instance"): return '{"new_contract":9}'
        if command[:4] == ("vastai", "--raw", "show", "instance"): return '{"id":9,"gpu_name":"RTX PRO 6000 WS","num_gpus":1,"gpu_ram":96000,"dph_total":0.7,"ssh_host":"host","ssh_port":22,"driver_version":"595.71.05"}'
        if command[:3] == ("vastai", "destroy", "instance"): return ""
        raise AssertionError(command)
    monkeypatch.setattr(LIFECYCLE.time, "time", lambda: 100)
    image = "ghcr.io/ryanjin333/lehome-groot-n17-trainer@sha256:" + "a" * 64
    evidence = LIFECYCLE.capture_offers(runner=runner, now_unix=100) | {"trainer_image": image, "training_capability": {"hardware": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "driver_version": "595.71.05", "image_digest": image.rpartition("@")[2], "cuda_runtime": "12.8", "torch_cuda": "12.8", "compute_capability": "12.0", "optimizer_step": {"passed": True, "loss": .2}, "nvml": {"utilization_percent": 80}}}
    assert evidence["offer"]["gpu_name"] == "RTX PRO 6000 WS"
    assert "untrusted_token" not in evidence["offer"]
    instance = LIFECYCLE.rent(evidence=evidence, runner=runner)
    assert instance["instance_id"] == 9
    with pytest.raises(ValueError, match="disposal terminal"):
        LIFECYCLE.destroy(instance_id=9, training_receipt={"kind": "continuous_corrective_training_terminal", "instance_id": 9, "immutable_checkpoint_steps": [1000, 2000], "immutable_checkpoint_publications": [{"optimizer_step": 1000, "repository": "ryanjin333/lehome-groot-n17-models", "immutable_revision": "a" * 40, "remote_prefix": "prefix", "relative_path": "checkpoints/step-1000.tar", "artifact_sha256": hashlib.sha256(b"artifact").hexdigest(), "artifact_byte_size": 8, "readback_verified": True}, {"optimizer_step": 2000, "repository": "ryanjin333/lehome-groot-n17-models", "immutable_revision": "b" * 40, "remote_prefix": "prefix", "relative_path": "checkpoints/step-2000.tar", "artifact_sha256": hashlib.sha256(b"artifact").hexdigest(), "artifact_byte_size": 8, "readback_verified": True}]}, runner=runner, transport=FakeHub(), token="test-token")


def test_offer_total_conservatively_counts_requested_300gb_storage() -> None:
    def runner(command: tuple[str, ...]) -> str:
        if command[:4] == ("vastai", "--raw", "search", "offers"):
            return '[{"id":7,"gpu_name":"RTX PRO 6000 S","num_gpus":1,"gpu_ram":96000,"dph_total":0.7,"min_bid":0.5,"storage_cost":0.001}]'
        if command[:4] == ("vastai", "--raw", "show", "instances"):
            return "[]"
        if command[:4] == ("vastai", "--raw", "show", "volumes"):
            return "[]"
        raise AssertionError(command)
    evidence = LIFECYCLE.capture_offers(runner=runner, now_unix=1)
    assert evidence["requested_storage_gb"] == 300
    assert evidence["requested_storage_hourly_usd"] == pytest.approx(0.3)
    assert evidence["account_hourly_total_usd"] == pytest.approx(1.0)


def test_offer_without_storage_quote_uses_nonzero_conservative_storage_bound() -> None:
    def runner(command: tuple[str, ...]) -> str:
        if command[:4] == ("vastai", "--raw", "search", "offers"):
            return '[{"id":7,"gpu_name":"RTX PRO 6000 S","num_gpus":1,"gpu_ram":96000,"dph_total":0.7,"min_bid":0.5}]'
        if command[:4] in {("vastai", "--raw", "show", "instances"), ("vastai", "--raw", "show", "volumes")}:
            return "[]"
        raise AssertionError(command)
    evidence = LIFECYCLE.capture_offers(runner=runner, now_unix=1)
    assert evidence["requested_storage_hourly_usd"] > 0


def test_rent_requires_capability_receipt_for_exact_image(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(LIFECYCLE.time, "time", lambda: 100)
    evidence = {"offer": {"id": 7, "min_bid": .5}, "search_mode": "interruptible", "expires_at_unix": 101, "trainer_image": "ghcr.io/ryanjin333/lehome-groot-n17-trainer@sha256:" + "a" * 64}
    with pytest.raises(ValueError, match="capability"):
        LIFECYCLE.rent(evidence=evidence, runner=lambda _: "{}")


def test_status_parses_authenticated_terminal() -> None:
    instance = {"instance_id": 7, "host": "host", "port": 22}
    terminal = {"kind": "continuous_corrective_training_terminal", "instance_id": 7}
    result = LIFECYCLE.remote_action(action="status", instance=instance, request={}, runner=lambda _command: json.dumps(terminal))
    assert result["terminal"] == terminal


def test_train_requires_capability_receipt_bound_to_its_instance() -> None:
    instance = {"instance_id": 7, "host": "host", "port": 22, "trainer_image": LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE, "provider_response_sha256": "a" * 64}
    request = {
        "generation_sha256": "b" * 64,
        "config_sha256": "c" * 64,
        "capability_receipt": {
            "kind": "persistent_training_capability",
            "instance_id": 8,
            "trainer_image": LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE,
            "provider_response_sha256": "a" * 64,
            "training_capability": {"hardware": "NVIDIA RTX PRO 6000 Blackwell", "driver_version": "595.71.05", "image_digest": LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE.rpartition("@")[2], "cuda_runtime": "12.8", "torch_cuda": "12.8", "compute_capability": "12.0", "optimizer_step": {"passed": True, "loss": .2}, "nvml": {"utilization_percent": 90}},
        },
    }
    with pytest.raises(ValueError, match="capability receipt"):
        LIFECYCLE.remote_action(action="train", instance=instance, request=request, runner=FailRunner())


def test_status_reads_only_requested_terminal_beneath_output() -> None:
    instance = {"instance_id": 7, "host": "host", "port": 22}
    terminal = {"kind": "continuous_corrective_training_terminal", "instance_id": 7}
    seen: list[tuple[str, ...]] = []
    result = LIFECYCLE.remote_action(
        action="status",
        instance=instance,
        request={"terminal_path": "/output/persistent/terminal.json"},
        runner=lambda command: seen.append(command) or json.dumps(terminal),
    )
    assert result["terminal"] == terminal
    assert seen[-1][-1].endswith("cat /output/persistent/terminal.json")

    with pytest.raises(ValueError, match="terminal path"):
        LIFECYCLE.remote_action(action="status", instance=instance, request={"terminal_path": "/tmp/terminal.json"}, runner=FailRunner())
    with pytest.raises(ValueError, match="terminal path"):
        LIFECYCLE.remote_action(action="status", instance=instance, request={"terminal_path": "/output/x;whoami"}, runner=FailRunner())


def test_stage_setup_hydrates_only_operational_roots() -> None:
    command = LIFECYCLE._stage_setup_command("a" * 64)
    assert "tar --no-same-owner --no-same-permissions -xf /tmp/lehome-stage/code.bundle -C /prepared/code" in command
    assert "tar --no-same-owner --no-same-permissions -xf /tmp/lehome-stage/parent.tar -C /cache/parent" in command
    assert "mv /tmp/lehome-stage/generation /prepared/generation" in command
    assert "chmod 600 /prepared/config/publisher.token" in command
    assert "mv /tmp/lehome-stage/continuous.json /prepared/config/continuous.json" in command
    assert "policy_artifact_sha256('/cache/parent')" in command


def test_stage_requires_distinct_parent_archive_and_policy_artifact_hashes() -> None:
    request = {"parent_checkpoint_sha256": LIFECYCLE.PARENT_CHECKPOINT["artifact_sha256"], "parent_archive_sha256": "a" * 64,
               "parent_checkpoint_repository": LIFECYCLE.PARENT_CHECKPOINT["repository"], "parent_checkpoint_revision": LIFECYCLE.PARENT_CHECKPOINT["revision"], "parent_checkpoint_subpath": LIFECYCLE.PARENT_CHECKPOINT["subpath"]}
    assert LIFECYCLE._parent_identities(request) == ("a" * 64, LIFECYCLE.PARENT_CHECKPOINT["artifact_sha256"])


def test_stage_rejects_runtime_request_that_points_at_unstaged_paths(tmp_path: Path) -> None:
    launch = tmp_path / "launch.json"
    launch.write_text(json.dumps({"base_model_path": "/tmp/model", "dataset_path": "/prepared/generation", "output_dir": "/output/run", "modality_config_path": "/prepared/config/modality.py"}))
    continuous = tmp_path / "continuous.json"
    continuous.write_text(json.dumps({"launch_config": "/prepared/config/launch.json", "experiment_config": "/prepared/config/experiment.json", "generation_root": "/prepared/generation", "publisher_token_file": "/prepared/config/publisher.token"}))
    with pytest.raises(ValueError, match="base model path"):
        LIFECYCLE._validate_staged_operational_requests(launch, continuous)


def test_stage_requires_code_bundle_receipt_to_match_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "code.bundle"
    bundle.write_bytes(b"not-an-archive")
    receipt = tmp_path / "code.bundle.sha256"
    receipt.write_text("0" * 64 + "  code.bundle\n", encoding="utf-8")
    with pytest.raises(ValueError, match="code bundle receipt"):
        LIFECYCLE._verify_code_bundle_receipt(bundle, receipt)


def test_bootstrap_canary_uses_only_historical_image_and_binds_instance_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...]) -> str:
        commands.append(command)
        if command[:4] == ("vastai", "--raw", "create", "instance"):
            return '{"new_contract":9}'
        if command[:4] == ("vastai", "--raw", "show", "instance"):
            return '{"id":9,"gpu_name":"RTX PRO 6000 WS","num_gpus":1,"gpu_ram":96000,"dph_total":0.7,"ssh_host":"host","ssh_port":22}'
        if command[0] == "scp":
            return ""
        if command[0] == "ssh" and "sha256sum" in command[-1]:
            return hashlib.sha256(bundle_path.read_bytes()).hexdigest() + "  code.bundle\n"
        if command[0] == "ssh":
            return json.dumps({"hardware": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "driver_version": "595.71.05", "image_digest": LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE.rpartition("@")[2], "cuda_runtime": "12.8", "torch_cuda": "12.8", "compute_capability": "12.0", "optimizer_step": {"passed": True, "loss": .2}, "nvml": {"utilization_percent": 80}})
        raise AssertionError(command)

    monkeypatch.setattr(LIFECYCLE.time, "time", lambda: 100)
    import tarfile
    bundle_path = tmp_path / "code.bundle"
    with tarfile.open(bundle_path, "w") as archive:
        info = tarfile.TarInfo("trainer/src/lehome_train/__init__.py"); info.size = 0; archive.addfile(info)
    receipt_path = tmp_path / "code.bundle.sha256"
    receipt_path.write_text(hashlib.sha256(bundle_path.read_bytes()).hexdigest() + "  code.bundle\n", encoding="utf-8")
    receipt = LIFECYCLE.bootstrap_canary(
        evidence={"offer": {"id": 7, "min_bid": .5}, "search_mode": "interruptible", "expires_at_unix": 101, "trainer_image": LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE, "code_bundle": str(bundle_path), "code_bundle_sha256_file": str(receipt_path)},
        runner=runner,
    )

    assert receipt["instance_id"] == 9
    assert receipt["trainer_image"] == LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE
    assert receipt["instance"]["host"] == "host"
    assert receipt["training_capability"]["image_digest"] == LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE.rpartition("@")[2]
    assert any("--one-step" in command[-1] for command in commands if command[0] == "ssh")


def test_promote_canary_recovers_only_the_bound_instance_receipt() -> None:
    image = LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE
    receipt = {
        "kind": "persistent_training_capability", "instance_id": 9,
        "trainer_image": image, "provider_response_sha256": "a" * 64,
        "instance": {"instance_id": 9, "trainer_image": image, "provider_response_sha256": "a" * 64, "host": "host", "port": 22},
        "training_capability": {"hardware": "NVIDIA RTX PRO 6000 Blackwell", "driver_version": "595.71.05", "image_digest": image.rpartition("@")[2], "cuda_runtime": "12.8", "torch_cuda": "12.8", "compute_capability": "12.0", "optimizer_step": {"passed": True, "loss": .1}, "nvml": {"utilization_percent": 80}},
    }
    def runner(command: tuple[str, ...]) -> str:
        assert command == ("vastai", "--raw", "show", "instance", "9")
        return json.dumps({
            "id": 9, "actual_status": "running", "gpu_name": "RTX PRO 6000 WS",
            "gpu_ram": 96000, "num_gpus": 1, "dph_total": .7,
            "ssh_host": "host", "ssh_port": 22,
        })
    receipt["instance"]["provider_response_sha256"] = LIFECYCLE._hash(json.loads(runner(("vastai", "--raw", "show", "instance", "9"))))
    receipt["provider_response_sha256"] = receipt["instance"]["provider_response_sha256"]
    assert LIFECYCLE.promote_canary(capability_receipt=receipt, runner=runner)["instance_id"] == 9


def test_promote_canary_requires_fresh_matching_live_provider_readback() -> None:
    image = LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE
    receipt = {
        "kind": "persistent_training_capability", "instance_id": 9,
        "trainer_image": image, "provider_response_sha256": "a" * 64,
        "instance": {"instance_id": 9, "trainer_image": image, "provider_response_sha256": "a" * 64, "host": "host", "port": 22, "offer_evidence_sha256": "b" * 64},
        "training_capability": {"hardware": "NVIDIA RTX PRO 6000 Blackwell", "driver_version": "595.71.05", "image_digest": image.rpartition("@")[2], "cuda_runtime": "12.8", "torch_cuda": "12.8", "compute_capability": "12.0", "optimizer_step": {"passed": True, "loss": .1}, "nvml": {"utilization_percent": 80}},
    }
    with pytest.raises(ValueError, match="fresh live"):
        LIFECYCLE.promote_canary(capability_receipt=receipt, runner=lambda _: "{}")


def test_bootstrap_canary_stages_current_code_and_cleans_up_after_probe_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "code.bundle"
    import tarfile
    with tarfile.open(bundle, "w") as archive:
        info = tarfile.TarInfo("trainer/src/lehome_train/__init__.py"); info.size = 0; archive.addfile(info)
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    receipt = tmp_path / "code.bundle.sha256"
    receipt.write_text(digest + "  code.bundle\n", encoding="utf-8")
    commands: list[tuple[str, ...]] = []
    destroyed = False
    def runner(command: tuple[str, ...]) -> str:
        nonlocal destroyed
        commands.append(command)
        if command[:4] == ("vastai", "--raw", "create", "instance"): return '{"new_contract":9}'
        if command[:4] == ("vastai", "--raw", "show", "instance"): return "{}" if destroyed else '{"id":9,"gpu_name":"RTX PRO 6000 WS","num_gpus":1,"gpu_ram":96000,"dph_total":0.7,"ssh_host":"host","ssh_port":22}'
        if command[:3] == ("vastai", "destroy", "instance"):
            destroyed = True
            return ""
        if command[0] == "scp": return ""
        if command[0] == "ssh": raise RuntimeError("probe failed")
        raise AssertionError(command)
    monkeypatch.setattr(LIFECYCLE.time, "time", lambda: 100)
    with pytest.raises(RuntimeError, match="probe failed"):
        LIFECYCLE.bootstrap_canary(evidence={"offer": {"id": 7, "min_bid": .5}, "search_mode": "interruptible", "expires_at_unix": 101, "trainer_image": LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE, "code_bundle": str(bundle), "code_bundle_sha256_file": str(receipt)}, runner=runner)
    assert any(command[:3] == ("vastai", "destroy", "instance") for command in commands)


def test_materialize_builds_a_verified_sealed_generation(tmp_path: Path) -> None:
    # The lifecycle's free preparation action must exercise the same canonical
    # mix/materialization implementation as production, rather than accepting
    # a hand-written receipt.
    from test_flywheel_mix import _prepared_source

    organizer = _prepared_source(tmp_path / "organizer", kind="organizer", episodes=2)
    corrective_a = _prepared_source(tmp_path / "corrective-a", kind="flywheel", grade="A", episodes=1)
    corrective_b = _prepared_source(tmp_path / "corrective-b", kind="flywheel", grade="B", episodes=1)
    destination = tmp_path / "generation"
    request = tmp_path / "materialize.json"
    request.write_text(json.dumps({
        "organizer_root": str(organizer),
        "corrective_roots": [str(corrective_a), str(corrective_b)],
        "destination": str(destination),
        "seed": 20260812,
        "organizer_source_evidence": LIFECYCLE.ORGANIZER_SOURCE,
        "corrective_source_evidence": LIFECYCLE.CORRECTIVE_SOURCE | {"release_id": "b" * 64},
    }))

    report = LIFECYCLE.main_for_test(["materialize", "--request", str(request)])

    assert report["paid_action"] is False
    assert report["generation_root"] == str(destination)
    assert (destination.with_name(destination.name + ".generation.json")).is_file()


def test_prepare_requires_exact_pinned_sources_in_the_sealed_receipt(tmp_path: Path) -> None:
    from test_flywheel_mix import _prepared_source
    organizer = _prepared_source(tmp_path / "organizer", kind="organizer", episodes=2)
    corrective_a = _prepared_source(tmp_path / "corrective-a", kind="flywheel", grade="A", episodes=1)
    corrective_b = _prepared_source(tmp_path / "corrective-b", kind="flywheel", grade="B", episodes=1)
    root = tmp_path / "generation"
    LIFECYCLE._materialize({"organizer_root": str(organizer), "corrective_roots": [str(corrective_a), str(corrective_b)], "destination": str(root), "seed": 1, "organizer_source_evidence": LIFECYCLE.ORGANIZER_SOURCE, "corrective_source_evidence": LIFECYCLE.CORRECTIVE_SOURCE | {"release_id": "b" * 64}})
    request = tmp_path / "prepare.json"
    request.write_text(json.dumps({"generation_root": str(root)}), encoding="utf-8")
    assert LIFECYCLE.main_for_test(["prepare", "--request", str(request)])["paid_action"] is False


def test_prepare_requires_local_organizer_and_corrective_release_evidence(
    tmp_path: Path,
) -> None:
    receipt = {
        "schema_version": 1, "sealed": True,
        "source_revisions": {
            "organizer:x": LIFECYCLE.ORGANIZER_SOURCE["revision"],
            "flywheel:y": LIFECYCLE.CORRECTIVE_SOURCE["revision"],
        },
        "organizer_source": LIFECYCLE.ORGANIZER_SOURCE | {"manifest_sha256": "a" * 64},
        "corrective_source": LIFECYCLE.CORRECTIVE_SOURCE | {"repository": "ryanjin333/lehome-groot-n17-data", "release_id": "b" * 64},
    }
    with pytest.raises(ValueError, match="organizer evidence"):
        LIFECYCLE._verify_prepare_evidence(receipt)
