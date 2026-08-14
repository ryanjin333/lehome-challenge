from pathlib import Path
import hashlib
import json
import subprocess
import time

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
        return (
            HubTreeEntry("prefix/checkpoints/step-1000.tar", "file"),
            HubTreeEntry("prefix/checkpoints/step-1000.json", "file"),
            HubTreeEntry("prefix/checkpoints/step-2000.tar", "file"),
            HubTreeEntry("prefix/checkpoints/step-2000.json", "file"),
        )
    def download_files(self, *, repository, revision, destination, relative_paths, token, remote_prefix=None):
        for relative in relative_paths:
            target = destination / relative; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(b"artifact")
        return revision


def _canonical_resume_chain() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    instance = {
        "schema_version": 1,
        "kind": "persistent_training_instance",
        "instance_id": 10,
        "host": "replacement",
        "port": 22,
        "trainer_image": LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE,
        "offer_evidence_sha256": "a" * 64,
        "provider_response_sha256": "b" * 64,
        "account_hourly_total_usd": 0.7,
    }
    publication = {
        "optimizer_step": 1000,
        "readback_verified": True,
        "generation_sha256": "c" * 64,
        "config_sha256": "d" * 64,
        "experiment_id": "persistent-001",
        "repository": LIFECYCLE.PARENT_CHECKPOINT["repository"],
        "immutable_revision": "e" * 40,
        "remote_prefix": "prefix",
        "relative_path": "checkpoints/step-1000.tar",
        "artifact_sha256": "f" * 64,
        "artifact_byte_size": 8,
        "descriptor_relative_path": "checkpoints/step-1000.json",
        "descriptor_sha256": "a" * 64,
        "descriptor_byte_size": 9,
    }
    capability = {
        "schema_version": 1,
        "kind": "persistent_training_capability",
        "instance_id": 10,
        "trainer_image": LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE,
        "provider_response_sha256": "b" * 64,
        "instance": instance,
        "training_capability": {
            "hardware": "NVIDIA RTX PRO 6000 Blackwell",
            "driver_version": "595.71.05",
            "image_digest": LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE.rpartition("@")[2],
            "cuda_runtime": "12.8",
            "torch_cuda": "12.8",
            "compute_capability": "12.0",
            "optimizer_step": {"passed": True, "loss": 0.2},
            "nvml": {"utilization_percent": 90},
        },
    }
    descriptor = {
        "path": "/local/replacement/resume-checkpoint.json",
        "sha256": "a" * 64,
        "byte_size": 9,
        "relative_path": "checkpoints/step-1000.json",
    }
    terminal = {
        "schema_version": 1,
        "kind": "continuous_corrective_training_terminal",
        "status": "provider_interrupted",
        "instance_id": 9,
        "generation_sha256": "c" * 64,
        "config_sha256": "d" * 64,
        "experiment_id": "persistent-001",
        "provider_reason": "instance_absent",
        "immutable_checkpoint_publications": [publication],
        "resumable_checkpoint_step": 1000,
        "disposable": False,
    }
    replacement = {
        "schema_version": 1,
        "kind": "persistent_training_replacement_resume",
        "instance": instance,
        "capability_receipt": capability,
        "resume_checkpoint_publication": publication,
        "resume_checkpoint_descriptor": descriptor,
        "generation_sha256": "c" * 64,
        "config_sha256": "d" * 64,
        "resume_generation_sha256": "c" * 64,
        "resume_config_sha256": "d" * 64,
        "experiment_id": "persistent-001",
        "replaced_instance_id": 9,
    }
    request = {
        "generation_sha256": "c" * 64,
        "config_sha256": "d" * 64,
        "resume_generation_sha256": "c" * 64,
        "resume_config_sha256": "d" * 64,
        "capability_receipt": capability,
        "provider_interruption_terminal": terminal,
        "resume_checkpoint_publication": publication,
        "resume_checkpoint_descriptor": descriptor,
        "replacement_resume_receipt": replacement,
    }
    envelope = {
        "schema_version": 1,
        "command": "continuous-train",
        "arguments": {
            "launch_config": "/prepared/config/launch.json",
            "experiment_config": "/prepared/config/experiment.json",
            "generation_root": "/prepared/generation",
            "parent_checkpoint_sha256": LIFECYCLE.PARENT_CHECKPOINT["artifact_sha256"],
            "normalization_sha256": "b" * 64,
            "checkpoint_repository": LIFECYCLE.PARENT_CHECKPOINT["repository"],
            "checkpoint_revision": "main",
            "instance_id": 10,
            "result_output": "/output/results.json",
            "status_output": "/output/status.json",
            "resume_checkpoint": "/prepared/config/resume-checkpoint.json",
            "resume_publication": publication,
            "publisher_token_file": "/prepared/config/publisher.token",
        },
    }
    return instance, request, envelope


def _two_publication_resume_chain() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    instance, request, envelope = _canonical_resume_chain()
    first = request["resume_checkpoint_publication"]
    terminal = request["provider_interruption_terminal"]
    replacement = request["replacement_resume_receipt"]
    descriptor = request["resume_checkpoint_descriptor"]
    assert isinstance(first, dict)
    assert isinstance(terminal, dict)
    assert isinstance(replacement, dict)
    assert isinstance(descriptor, dict)
    latest = first | {
        "optimizer_step": 2000,
        "relative_path": "checkpoints/step-2000.tar",
        "artifact_sha256": "1" * 64,
        "descriptor_relative_path": "checkpoints/step-2000.json",
        "descriptor_sha256": "2" * 64,
    }
    terminal["immutable_checkpoint_publications"] = [first, latest]
    terminal["resumable_checkpoint_step"] = 2000
    descriptor.update({"sha256": "2" * 64, "relative_path": "checkpoints/step-2000.json"})
    replacement["resume_checkpoint_publication"] = latest
    request["resume_checkpoint_publication"] = latest
    envelope["arguments"]["resume_publication"] = latest
    return instance, request, envelope


def _assert_no_resume_execution(calls: list[tuple[str, ...]]) -> None:
    assert not any("continuous-train --request /prepared/config/resume.json" in command[-1] for command in calls)


def _corrective_release_receipt(roots: list[Path], path: Path) -> Path:
    """Fixture for the independently verified prior full-release readback."""
    from lehome_train.data.inspect import artifact_identities
    from lehome_train.io import atomic_write_json, canonical_json_sha256
    for root in roots:
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["source_revision"] = "a" * 40
        manifest["source_release_id"] = "b" * 64
        manifest["output_artifacts"] = artifact_identities(root, exclude={"manifest.json"})
        manifest["output_manifest_sha256"] = canonical_json_sha256(manifest["output_artifacts"])
        atomic_write_json(manifest_path, manifest)
    payload = {
        "published_release": LIFECYCLE.CORRECTIVE_SOURCE | {"release_id": "b" * 64},
        "local_snapshot": {
            "source_revision": "a" * 40,
            "source_release_id": "b" * 64,
            "trees": {str(root): LIFECYCLE._tree_readback_sha256(root) for root in roots},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


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
            {"optimizer_step": step, "repository": "wrong/repo", "immutable_revision": "c" * 40, "remote_prefix": "prefix", "relative_path": f"step-{step}.tar", "artifact_sha256": "d" * 64, "artifact_byte_size": 1, "descriptor_relative_path": f"step-{step}.json", "descriptor_sha256": "d" * 64, "descriptor_byte_size": 1, "readback_verified": True, "generation_sha256": "a" * 64, "config_sha256": "b" * 64, "experiment_id": "persistent-001"}
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
        if command[:4] == ("vastai", "--raw", "show", "instance"): return '{"id":9,"actual_status":"running","gpu_name":"RTX PRO 6000 WS","num_gpus":1,"gpu_ram":96000,"dph_total":0.7,"ssh_host":"host","ssh_port":22,"driver_version":"595.71.05"}'
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
        LIFECYCLE.destroy(instance_id=9, training_receipt={"kind": "continuous_corrective_training_terminal", "instance_id": 9, "immutable_checkpoint_steps": [1000, 2000], "immutable_checkpoint_publications": [{"optimizer_step": 1000, "repository": "ryanjin333/lehome-groot-n17-models", "immutable_revision": "a" * 40, "remote_prefix": "prefix", "relative_path": "checkpoints/step-1000.tar", "artifact_sha256": hashlib.sha256(b"artifact").hexdigest(), "artifact_byte_size": 8, "descriptor_relative_path": "checkpoints/step-1000.json", "descriptor_sha256": hashlib.sha256(b"artifact").hexdigest(), "descriptor_byte_size": 8, "readback_verified": True}, {"optimizer_step": 2000, "repository": "ryanjin333/lehome-groot-n17-models", "immutable_revision": "b" * 40, "remote_prefix": "prefix", "relative_path": "checkpoints/step-2000.tar", "artifact_sha256": hashlib.sha256(b"artifact").hexdigest(), "artifact_byte_size": 8, "descriptor_relative_path": "checkpoints/step-2000.json", "descriptor_sha256": hashlib.sha256(b"artifact").hexdigest(), "descriptor_byte_size": 8, "readback_verified": True}]}, runner=runner, transport=FakeHub(), token="test-token")


def test_offer_total_uses_the_300gb_vast_quote_once() -> None:
    def runner(command: tuple[str, ...]) -> str:
        if command[:4] == ("vastai", "--raw", "search", "offers"):
            return '[{"id":7,"gpu_name":"RTX PRO 6000 S","num_gpus":1,"gpu_ram":96000,"dph_total":0.7,"dph_base":0.4,"min_bid":0.5,"storage_cost":0.001}]'
        if command[:4] == ("vastai", "--raw", "show", "instances"):
            return "[]"
        if command[:4] == ("vastai", "--raw", "show", "volumes"):
            return "[]"
        raise AssertionError(command)
    evidence = LIFECYCLE.capture_offers(runner=runner, now_unix=1)
    assert evidence["requested_storage_gb"] == 300
    assert evidence["requested_storage_hourly_usd"] == pytest.approx(0.3)
    # ``dph_total`` comes from the explicit --storage 300 quote: it must not
    # be added a second time merely because its storage component is exposed.
    assert evidence["account_hourly_total_usd"] == pytest.approx(0.7)


def test_offer_without_storage_breakdown_treats_300gb_total_quote_as_conservative() -> None:
    def runner(command: tuple[str, ...]) -> str:
        if command[:4] == ("vastai", "--raw", "search", "offers"):
            return '[{"id":7,"gpu_name":"RTX PRO 6000 S","num_gpus":1,"gpu_ram":96000,"dph_total":0.7,"min_bid":0.5}]'
        if command[:4] in {("vastai", "--raw", "show", "instances"), ("vastai", "--raw", "show", "volumes")}:
            return "[]"
        raise AssertionError(command)
    evidence = LIFECYCLE.capture_offers(runner=runner, now_unix=1)
    assert evidence["requested_storage_hourly_usd"] is None
    assert evidence["account_hourly_total_usd"] == pytest.approx(0.7)


def test_offer_rejects_account_wide_total_above_one_dollar_per_hour() -> None:
    def runner(command: tuple[str, ...]) -> str:
        if command[:4] == ("vastai", "--raw", "search", "offers"):
            return '[{"id":7,"gpu_name":"RTX PRO 6000 S","num_gpus":1,"gpu_ram":96000,"dph_total":0.7,"min_bid":0.5}]'
        if command[:4] == ("vastai", "--raw", "show", "instances"):
            return '[{"id":2,"dph_total":0.31}]'
        if command[:4] == ("vastai", "--raw", "show", "volumes"):
            return "[]"
        raise AssertionError(command)
    with pytest.raises(ValueError, match=r"exceeds \$1/hr"):
        LIFECYCLE.capture_offers(runner=runner, now_unix=1)


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


def test_provider_interruption_terminal_is_resumable_only_with_last_immutable_checkpoint() -> None:
    terminal = LIFECYCLE.provider_interruption_terminal(
        instance_id=9,
        generation_sha256="a" * 64,
        config_sha256="b" * 64,
        experiment_id="persistent-001",
        publications=[{"optimizer_step": 1000, "readback_verified": True}],
        provider_reason="instance preempted",
    )
    assert terminal["status"] == "provider_interrupted"
    assert terminal["resumable_checkpoint_step"] == 1000
    with pytest.raises(ValueError, match="provider interruption"):
        LIFECYCLE.resume_identity(terminal, generation_sha256="a" * 64, config_sha256="c" * 64)


def test_provider_absence_builds_a_replacement_resume_descriptor_with_same_identities(
    tmp_path: Path,
) -> None:
    terminal = LIFECYCLE.provider_interruption_terminal(
        instance_id=9,
        generation_sha256="a" * 64,
        config_sha256="b" * 64,
        experiment_id="persistent-001",
        publications=[{
            "optimizer_step": 1000,
            "readback_verified": True,
            "generation_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "experiment_id": "persistent-001",
            "immutable_revision": "c" * 40,
            "repository": LIFECYCLE.PARENT_CHECKPOINT["repository"],
            "remote_prefix": "prefix",
            "relative_path": "checkpoints/step-1000.tar",
            "artifact_sha256": hashlib.sha256(b"artifact").hexdigest(),
            "artifact_byte_size": 8,
            "descriptor_relative_path": "checkpoints/step-1000.json",
            "descriptor_sha256": hashlib.sha256(b"artifact").hexdigest(),
            "descriptor_byte_size": 8,
        }],
        provider_reason="instance absent",
    )
    def runner(command: tuple[str, ...]) -> str:
        if command == ("vastai", "--raw", "show", "instance", "9"):
            return "{}"
        if command == ("vastai", "--raw", "show", "instance", "10"):
            return json.dumps({
                "id": 10, "actual_status": "running", "gpu_name": "RTX PRO 6000 WS",
                "gpu_ram": 96000, "num_gpus": 1, "dph_total": .7,
                "ssh_host": "replacement", "ssh_port": 22,
            })
        if command == ("vastai", "--raw", "show", "instances"):
            return '[{"id":10,"dph_total":0.7}]'
        if command == ("vastai", "--raw", "show", "volumes"):
            return "[]"
        raise AssertionError(command)
    replacement = {
        "schema_version": 1, "kind": "persistent_training_instance",
        "instance_id": 10, "host": "replacement", "port": 22,
        "trainer_image": LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE,
        "offer_evidence_sha256": "a" * 64,
        "provider_response_sha256": LIFECYCLE._stable_instance_identity({
            "id": 10, "actual_status": "running", "gpu_name": "RTX PRO 6000 WS",
            "gpu_ram": 96000, "num_gpus": 1, "dph_total": .7,
            "ssh_host": "replacement", "ssh_port": 22,
        }),
        "account_hourly_total_usd": .7,
    }
    capability = {
        "schema_version": 1, "kind": "persistent_training_capability", "instance_id": 10,
        "trainer_image": LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE,
        "provider_response_sha256": replacement["provider_response_sha256"],
        "instance": replacement,
        "training_capability": {
            "hardware": "NVIDIA RTX PRO 6000 Blackwell", "driver_version": "595.71.05",
            "image_digest": LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE.rpartition("@")[2],
            "cuda_runtime": "12.8", "torch_cuda": "12.8", "compute_capability": "12.0",
            "optimizer_step": {"passed": True, "loss": .2}, "nvml": {"utilization_percent": 90},
        },
    }
    assert LIFECYCLE.classify_provider_interruption(instance={"instance_id": 9}, runner=runner) == "instance_absent"
    descriptor = LIFECYCLE.replacement_resume_descriptor(
        terminal=terminal, capability_receipt=capability, runner=runner,
        transport=FakeHub(), token="test-token", descriptor_output=tmp_path / "resume.json",
    )
    assert descriptor["instance"]["instance_id"] == 10
    assert descriptor["resume_checkpoint_publication"]["optimizer_step"] == 1000
    assert descriptor["resume_checkpoint_descriptor"]["path"].endswith("resume.json")
    assert descriptor["generation_sha256"] == "a" * 64
    assert descriptor["resume_generation_sha256"] == "a" * 64

    # The replacement receipt retains the local hydration path.  Stage binds
    # that source to the mounted path, and resume rechecks the mounted envelope
    # immediately before it executes.
    publication = descriptor["resume_checkpoint_publication"]
    local_descriptor = descriptor["resume_checkpoint_descriptor"]
    assert isinstance(publication, dict)
    assert isinstance(local_descriptor, dict)
    stage_request = {
        "generation_sha256": "a" * 64,
        "resume_checkpoint_descriptor": str(tmp_path / "resume.json"),
        "replacement_resume_receipt": descriptor,
    }
    LIFECYCLE._validate_replacement_stage_binding(
        {
            "resume_checkpoint": "/prepared/config/resume-checkpoint.json",
            "resume_publication": publication,
        },
        stage_request,
        instance=descriptor["instance"],
    )
    resume_request = {
        "generation_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "resume_generation_sha256": "a" * 64,
        "resume_config_sha256": "b" * 64,
        "capability_receipt": capability,
        "provider_interruption_terminal": terminal,
        "resume_checkpoint_publication": publication,
        "resume_checkpoint_descriptor": local_descriptor,
        "replacement_resume_receipt": descriptor,
    }

    def resume_runner(command: tuple[str, ...]) -> str:
        if command[-1] == "cat /prepared/config/resume.json":
            return json.dumps({
                "schema_version": 1,
                "command": "continuous-train",
                "arguments": {
                    "launch_config": "/prepared/config/launch.json",
                    "experiment_config": "/prepared/config/experiment.json",
                    "generation_root": "/prepared/generation",
                    "parent_checkpoint_sha256": LIFECYCLE.PARENT_CHECKPOINT["artifact_sha256"],
                    "normalization_sha256": "c" * 64,
                    "checkpoint_repository": LIFECYCLE.PARENT_CHECKPOINT["repository"],
                    "checkpoint_revision": "main",
                    "instance_id": descriptor["instance"]["instance_id"],
                    "result_output": "/output/results.json",
                    "status_output": "/output/status.json",
                    "resume_checkpoint": "/prepared/config/resume-checkpoint.json",
                    "resume_publication": publication,
                    "publisher_token_file": "/prepared/config/publisher.token",
                },
            })
        if "sha256sum /prepared/config/resume-checkpoint.json" in command[-1]:
            return f"{local_descriptor['sha256']}  resume-checkpoint.json\n{local_descriptor['byte_size']}\n"
        if "continuous-train --request /prepared/config/resume.json" in command[-1]:
            return ""
        raise AssertionError(command)

    assert LIFECYCLE.remote_action(
        action="resume", instance=descriptor["instance"], request=resume_request,
        runner=resume_runner,
    )["action"] == "resume"


def test_resume_rechecks_the_staged_envelope_against_the_replacement_receipt(
) -> None:
    instance, request, envelope = _canonical_resume_chain()
    publication = request["resume_checkpoint_publication"]
    assert isinstance(publication, dict)
    envelope["arguments"] = envelope["arguments"] | {
        "resume_publication": publication | {"relative_path": "checkpoints/step-2000.tar"},
    }

    def runner(command: tuple[str, ...]) -> str:
        if command[-1] == "cat /prepared/config/resume.json":
            return json.dumps(envelope)
        raise AssertionError("resume command must not execute after staged envelope mismatch")

    with pytest.raises(ValueError, match="staged resume request identity"):
        LIFECYCLE.remote_action(
            action="resume",
            instance=instance,
            request=request,
            runner=runner,
        )


@pytest.mark.parametrize("terminal", (None, "not-a-provider-terminal"))
def test_resume_requires_a_provider_interruption_terminal_before_execution(
    terminal: object,
) -> None:
    instance, request, _ = _canonical_resume_chain()
    if terminal is not None:
        request["provider_interruption_terminal"] = terminal
    else:
        request.pop("provider_interruption_terminal")
    calls: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...]) -> str:
        calls.append(command)
        return ""

    with pytest.raises(ValueError, match="resume terminal schema"):
        LIFECYCLE.remote_action(action="resume", instance=instance, request=request, runner=runner)
    assert not any("continuous-train --request /prepared/config/resume.json" in command[-1] for command in calls)


def test_resume_requires_a_replacement_receipt_before_execution(
) -> None:
    instance, request, _ = _canonical_resume_chain()
    request.pop("replacement_resume_receipt")
    calls: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...]) -> str:
        calls.append(command)
        return ""

    with pytest.raises(ValueError, match="resume replacement schema"):
        LIFECYCLE.remote_action(action="resume", instance=instance, request=request, runner=runner)
    assert not any("continuous-train --request /prepared/config/resume.json" in command[-1] for command in calls)


@pytest.mark.parametrize(
    "identity",
    ("generation_sha256", "resume_generation_sha256", "config_sha256", "resume_config_sha256"),
)
def test_resume_rejects_missing_identity_before_execution(identity: str) -> None:
    instance, request, _ = _canonical_resume_chain()
    request.pop(identity)
    calls: list[tuple[str, ...]] = []

    with pytest.raises(ValueError, match="resume identity is invalid"):
        LIFECYCLE.remote_action(
            action="resume", instance=instance, request=request,
            runner=lambda command: calls.append(command) or "",
        )
    _assert_no_resume_execution(calls)


@pytest.mark.parametrize("part", ("terminal", "replacement", "capability", "instance"))
def test_resume_rejects_partial_mapping_chain_evidence_before_execution(part: str) -> None:
    instance, request, _ = _canonical_resume_chain()
    if part == "terminal":
        request["provider_interruption_terminal"] = {"schema_version": 1}
    elif part == "replacement":
        request["replacement_resume_receipt"] = {"schema_version": 1}
    elif part == "capability":
        request["capability_receipt"] = {"schema_version": 1}
    else:
        instance = {"instance_id": 10}
    calls: list[tuple[str, ...]] = []

    with pytest.raises(ValueError, match="resume (identity|terminal|replacement|capability|instance) schema"):
        LIFECYCLE.remote_action(
            action="resume", instance=instance, request=request,
            runner=lambda command: calls.append(command) or "",
        )
    _assert_no_resume_execution(calls)


def test_resume_rejects_partial_staged_envelope_before_execution() -> None:
    instance, request, envelope = _canonical_resume_chain()
    envelope["arguments"] = {"resume_checkpoint": "/prepared/config/resume-checkpoint.json"}
    calls: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...]) -> str:
        calls.append(command)
        if command[-1] == "cat /prepared/config/resume.json":
            return json.dumps(envelope)
        raise AssertionError("a partial envelope must not reach descriptor or training execution")

    with pytest.raises(ValueError, match="staged resume request schema"):
        LIFECYCLE.remote_action(action="resume", instance=instance, request=request, runner=runner)
    _assert_no_resume_execution(calls)


def test_repeated_resume_preemption_terminalizes_the_validated_resume_lineage() -> None:
    instance, request, envelope = _canonical_resume_chain()
    descriptor = request["resume_checkpoint_descriptor"]
    assert isinstance(descriptor, dict)
    calls: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...]) -> str:
        calls.append(command)
        if command[-1] == "cat /prepared/config/resume.json":
            return json.dumps(envelope)
        if "sha256sum /prepared/config/resume-checkpoint.json" in command[-1]:
            return f"{descriptor['sha256']}  resume-checkpoint.json\n{descriptor['byte_size']}\n"
        if "continuous-train --request /prepared/config/resume.json" in command[-1]:
            raise subprocess.CalledProcessError(255, command)
        assert command == ("vastai", "--raw", "show", "instance", "10")
        return '{"id":10,"actual_status":"interrupted"}'

    result = LIFECYCLE.remote_action(
        action="resume", instance=instance, request=request, runner=runner,
    )

    terminal = result["terminal"]
    assert result["action"] == "resume"
    assert terminal["instance_id"] == 10
    assert terminal["generation_sha256"] == request["generation_sha256"]
    assert terminal["config_sha256"] == request["config_sha256"]
    assert terminal["experiment_id"] == request["provider_interruption_terminal"]["experiment_id"]
    assert terminal["provider_reason"] == "provider_interrupted"
    assert terminal["resumable_checkpoint_step"] == 1000
    assert terminal["disposable"] is False
    assert terminal["immutable_checkpoint_publications"] == request["provider_interruption_terminal"]["immutable_checkpoint_publications"]
    LIFECYCLE._validate_resume_terminal(
        terminal,
        generation_sha256=request["generation_sha256"],
        config_sha256=request["config_sha256"],
    )


def test_resume_rejects_stale_terminal_resumable_step_before_remote_execution() -> None:
    instance, request, _ = _two_publication_resume_chain()
    terminal = request["provider_interruption_terminal"]
    assert isinstance(terminal, dict)
    terminal["resumable_checkpoint_step"] = 1000
    calls: list[tuple[str, ...]] = []

    with pytest.raises(ValueError, match="resume terminal resumable checkpoint is not canonical"):
        LIFECYCLE.remote_action(
            action="resume", instance=instance, request=request,
            runner=lambda command: calls.append(command) or "",
        )
    assert calls == []


def test_repeated_resume_preemption_preserves_canonical_maximum_publication_lineage() -> None:
    instance, request, envelope = _two_publication_resume_chain()
    descriptor = request["resume_checkpoint_descriptor"]
    original_terminal = request["provider_interruption_terminal"]
    assert isinstance(descriptor, dict)
    assert isinstance(original_terminal, dict)

    def runner(command: tuple[str, ...]) -> str:
        if command[-1] == "cat /prepared/config/resume.json":
            return json.dumps(envelope)
        if "sha256sum /prepared/config/resume-checkpoint.json" in command[-1]:
            return f"{descriptor['sha256']}  resume-checkpoint.json\n{descriptor['byte_size']}\n"
        if "continuous-train --request /prepared/config/resume.json" in command[-1]:
            raise subprocess.CalledProcessError(255, command)
        assert command == ("vastai", "--raw", "show", "instance", "10")
        return '{"id":10,"actual_status":"interrupted"}'

    result = LIFECYCLE.remote_action(
        action="resume", instance=instance, request=request, runner=runner,
    )

    terminal = result["terminal"]
    assert terminal["resumable_checkpoint_step"] == 2000
    assert terminal["immutable_checkpoint_publications"] == original_terminal["immutable_checkpoint_publications"]
    LIFECYCLE._validate_resume_terminal(
        terminal,
        generation_sha256=request["generation_sha256"],
        config_sha256=request["config_sha256"],
    )


def test_remote_ssh_failure_is_terminalized_only_after_provider_interruption_readback() -> None:
    instance = {"instance_id": 9, "host": "old", "port": 22, "trainer_image": LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE, "provider_response_sha256": "a" * 64}
    request = {
        "generation_sha256": "a" * 64, "config_sha256": "b" * 64,
        "experiment_id": "persistent-001",
        "immutable_checkpoint_publications": [{
            "optimizer_step": 1000, "readback_verified": True,
            "generation_sha256": "a" * 64, "config_sha256": "b" * 64,
            "experiment_id": "persistent-001", "immutable_revision": "c" * 40,
        }],
        "capability_receipt": {"kind": "persistent_training_capability", "instance_id": 9, "trainer_image": LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE, "provider_response_sha256": "a" * 64, "training_capability": {"hardware": "NVIDIA RTX PRO 6000 Blackwell", "driver_version": "595.71.05", "image_digest": LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE.rpartition("@")[2], "cuda_runtime": "12.8", "torch_cuda": "12.8", "compute_capability": "12.0", "optimizer_step": {"passed": True, "loss": .2}, "nvml": {"utilization_percent": 90}}},
    }
    calls = 0
    def runner(command: tuple[str, ...]) -> str:
        nonlocal calls
        calls += 1
        if command[:2] == ("ssh", "-o"):
            raise subprocess.CalledProcessError(255, command)
        assert command == ("vastai", "--raw", "show", "instance", "9")
        return '{"id":9,"actual_status":"interrupted"}'
    result = LIFECYCLE.remote_action(action="train", instance=instance, request=request, runner=runner)
    assert result["terminal"]["status"] == "provider_interrupted"
    assert result["terminal"]["resumable_checkpoint_step"] == 1000
    assert calls == 2


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
    assert "resume-checkpoint.json /prepared/config/resume-checkpoint.json" in command
    assert "policy_artifact_sha256('/cache/parent')" in command


def test_stage_requires_distinct_parent_archive_and_policy_artifact_hashes() -> None:
    request = {"parent_checkpoint_sha256": LIFECYCLE.PARENT_CHECKPOINT["artifact_sha256"], "parent_archive_sha256": "0ddd4e7ce351dd2172cd1edd967293a50d02c15c0f2c21ca39db94692a57e0b5",
               "parent_checkpoint_repository": LIFECYCLE.PARENT_CHECKPOINT["repository"], "parent_checkpoint_revision": LIFECYCLE.PARENT_CHECKPOINT["revision"], "parent_checkpoint_subpath": LIFECYCLE.PARENT_CHECKPOINT["subpath"]}
    assert LIFECYCLE._parent_identities(request) == ("0ddd4e7ce351dd2172cd1edd967293a50d02c15c0f2c21ca39db94692a57e0b5", LIFECYCLE.PARENT_CHECKPOINT["artifact_sha256"])
    with pytest.raises(ValueError, match="parent archive"):
        LIFECYCLE._parent_identities(request | {"parent_archive_sha256": "a" * 64})


def test_stage_rejects_runtime_request_that_points_at_unstaged_paths(tmp_path: Path) -> None:
    launch = tmp_path / "launch.json"
    launch.write_text(json.dumps({"base_model_path": "/tmp/model", "dataset_path": "/prepared/generation", "output_dir": "/output/run", "modality_config_path": "/prepared/config/modality.py"}))
    continuous = tmp_path / "continuous.json"
    continuous.write_text(json.dumps({"launch_config": "/prepared/config/launch.json", "experiment_config": "/prepared/config/experiment.json", "generation_root": "/prepared/generation", "publisher_token_file": "/prepared/config/publisher.token"}))
    with pytest.raises(ValueError, match="base model path"):
        LIFECYCLE._validate_staged_operational_requests(launch, continuous)


def test_stage_validates_the_executable_nested_continuous_envelope(
    tmp_path: Path,
) -> None:
    launch = tmp_path / "launch.json"
    launch.write_text(json.dumps({
        "base_model_path": "/cache/parent",
        "dataset_path": "/prepared/generation",
        "output_dir": "/output",
        "modality_config_path": "/prepared/config/modality.py",
        "dataset_revision": "b" * 40,
    }))
    continuous = tmp_path / "continuous.json"
    continuous.write_text(json.dumps({
        "schema_version": 1,
        "command": "continuous-train",
        "arguments": {
            "launch_config": "/prepared/config/launch.json",
            "experiment_config": "/prepared/config/experiment.json",
            "generation_root": "/prepared/generation",
            "publisher_token_file": "/prepared/config/publisher.token",
        },
    }))

    LIFECYCLE._validate_staged_operational_requests(launch, continuous)


def test_stage_generation_identity_comes_from_the_sibling_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation = tmp_path / "generation"
    generation.mkdir()
    receipt = {
        "schema_version": 1,
        "sealed": True,
        "mix_plan_sha256": "a" * 64,
        "dataset_manifest_sha256": "b" * 64,
    }
    import lehome_train.flywheel.mix as mix
    monkeypatch.setattr(mix, "load_generation_receipt", lambda _root: receipt)

    identity = LIFECYCLE._sealed_generation_identity(generation)

    assert identity == {
        "mix_plan_sha256": "a" * 64,
        "dataset_manifest_sha256": "b" * 64,
        "dataset_revision": "b" * 40,
    }
    with pytest.raises(ValueError, match="caller generation identity"):
        LIFECYCLE._sealed_generation_identity(
            generation, claimed_mix_plan_sha256="c" * 64,
        )


def test_stage_rejects_a_resume_descriptor_that_does_not_match_its_publication(
    tmp_path: Path,
) -> None:
    descriptor = tmp_path / "resume-checkpoint.json"
    descriptor.write_bytes(b"authenticated descriptor")
    descriptor_sha = hashlib.sha256(descriptor.read_bytes()).hexdigest()
    arguments = {
        "resume_checkpoint": "/prepared/config/resume-checkpoint.json",
        "resume_publication": {
            "repository": LIFECYCLE.PARENT_CHECKPOINT["repository"],
            "immutable_revision": "a" * 40,
            "remote_prefix": "checkpoint-staging/run/archive",
            "relative_path": "checkpoints/step-1000.tar",
            "artifact_sha256": "b" * 64,
            "artifact_byte_size": 1,
            "descriptor_relative_path": "checkpoints/step-1000.json",
            "descriptor_sha256": descriptor_sha,
            "descriptor_byte_size": descriptor.stat().st_size,
        },
    }
    request = {"resume_checkpoint_descriptor": str(descriptor)}

    assert LIFECYCLE._validate_resume_descriptor_for_stage(arguments, request) == (
        str(descriptor), descriptor_sha, descriptor.stat().st_size,
    )
    descriptor.write_bytes(b"caller fabricated descriptor")
    with pytest.raises(ValueError, match="source is unavailable|differs from immutable publication"):
        LIFECYCLE._validate_resume_descriptor_for_stage(arguments, request)


def test_stage_requires_the_exact_authenticated_replacement_receipt(
    tmp_path: Path,
) -> None:
    descriptor = tmp_path / "resume-checkpoint.json"
    descriptor.write_bytes(b"authenticated descriptor")
    descriptor_sha = hashlib.sha256(descriptor.read_bytes()).hexdigest()
    publication = {
        "repository": LIFECYCLE.PARENT_CHECKPOINT["repository"],
        "immutable_revision": "a" * 40,
        "remote_prefix": "checkpoint-staging/run/archive",
        "relative_path": "checkpoints/step-1000.tar",
        "artifact_sha256": "b" * 64,
        "artifact_byte_size": 1,
        "descriptor_relative_path": "checkpoints/step-1000.json",
        "descriptor_sha256": descriptor_sha,
        "descriptor_byte_size": descriptor.stat().st_size,
    }
    replacement = {
        "schema_version": 1,
        "kind": "persistent_training_replacement_resume",
        "instance": {"instance_id": 10, "host": "replacement", "port": 22},
        "generation_sha256": "c" * 64,
        "config_sha256": "d" * 64,
        "resume_generation_sha256": "c" * 64,
        "resume_config_sha256": "d" * 64,
        "experiment_id": "corrective-rft-70-30-20260813",
        "resume_checkpoint_publication": publication,
        "resume_checkpoint_descriptor": {
            "path": str(descriptor), "sha256": descriptor_sha,
            "byte_size": descriptor.stat().st_size,
            "relative_path": "checkpoints/step-1000.json",
        },
    }
    arguments = {
        "resume_checkpoint": "/prepared/config/resume-checkpoint.json",
        "resume_publication": publication,
    }
    request = {
        "generation_sha256": "c" * 64,
        "resume_checkpoint_descriptor": str(descriptor),
        "replacement_resume_receipt": replacement,
    }

    instance = replacement["instance"]
    LIFECYCLE._validate_replacement_stage_binding(arguments, request, instance=instance)
    with pytest.raises(ValueError, match="replacement publication"):
        LIFECYCLE._validate_replacement_stage_binding(
            arguments | {"resume_publication": publication | {"relative_path": "checkpoints/step-2000.tar"}},
            request,
            instance=instance,
        )


def test_descriptor_hydration_reuses_only_the_exact_authenticated_retry_file(
    tmp_path: Path,
) -> None:
    output = tmp_path / "resume.json"
    output.write_bytes(b"authenticated descriptor")
    publication = {
        "repository": LIFECYCLE.PARENT_CHECKPOINT["repository"],
        "immutable_revision": "a" * 40,
        "remote_prefix": "prefix",
        "descriptor_relative_path": "checkpoints/step-1000.json",
        "descriptor_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "descriptor_byte_size": output.stat().st_size,
    }
    assert LIFECYCLE._hydrate_resume_descriptor(
        publication=publication, transport=FailRunner(), token="test-token", output=output,
    )["path"] == str(output)
    output.write_bytes(b"wrong retry descriptor")
    with pytest.raises(ValueError, match="retry output"):
        LIFECYCLE._hydrate_resume_descriptor(
            publication=publication, transport=FailRunner(), token="test-token", output=output,
        )


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
        if command[:4] == ("vastai", "--raw", "show", "instances"):
            return "[]"
        if command[:4] == ("vastai", "--raw", "show", "volumes"):
            return "[]"
        if command[:4] == ("vastai", "--raw", "create", "instance"):
            return '{"new_contract":9}'
        if command[:4] == ("vastai", "--raw", "show", "instance"):
            return '{"id":9,"actual_status":"running","gpu_name":"RTX PRO 6000 WS","num_gpus":1,"gpu_ram":96000,"dph_total":0.7,"ssh_host":"host","ssh_port":22}'
        if command[0] == "scp":
            return ""
        if command[0] == "ssh" and command[-1].startswith("sha256sum "):
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
        evidence={"offer": {"id": 7, "min_bid": .5, "dph_total": .7}, "search_mode": "interruptible", "expires_at_unix": 101, "account_hourly_total_usd": .7, "trainer_image": LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE, "code_bundle": str(bundle_path), "code_bundle_sha256_file": str(receipt_path)},
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
        if command == ("vastai", "--raw", "show", "instance", "9"):
            return json.dumps({
                "id": 9, "actual_status": "running", "gpu_name": "RTX PRO 6000 WS",
                "gpu_ram": 96000, "num_gpus": 1, "dph_total": .7,
                "ssh_host": "host", "ssh_port": 22,
            })
        if command == ("vastai", "--raw", "show", "instances"):
            return '[{"id":9,"dph_total":0.7}]'
        if command == ("vastai", "--raw", "show", "volumes"):
            return "[]"
        raise AssertionError(command)
    receipt["instance"]["provider_response_sha256"] = LIFECYCLE._stable_instance_identity(json.loads(runner(("vastai", "--raw", "show", "instance", "9"))))
    receipt["instance"]["account_hourly_total_usd"] = .7
    receipt["provider_response_sha256"] = receipt["instance"]["provider_response_sha256"]
    assert LIFECYCLE.promote_canary(capability_receipt=receipt, runner=runner)["instance_id"] == 9


def test_promote_canary_requires_fresh_matching_live_provider_readback() -> None:
    image = LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE
    receipt = {
        "kind": "persistent_training_capability", "instance_id": 9,
        "trainer_image": image, "provider_response_sha256": "a" * 64,
        "instance": {"instance_id": 9, "trainer_image": image, "provider_response_sha256": "a" * 64, "host": "host", "port": 22, "offer_evidence_sha256": "b" * 64, "account_hourly_total_usd": .7},
        "training_capability": {"hardware": "NVIDIA RTX PRO 6000 Blackwell", "driver_version": "595.71.05", "image_digest": image.rpartition("@")[2], "cuda_runtime": "12.8", "torch_cuda": "12.8", "compute_capability": "12.0", "optimizer_step": {"passed": True, "loss": .1}, "nvml": {"utilization_percent": 80}},
    }
    with pytest.raises(ValueError, match="fresh live"):
        LIFECYCLE.promote_canary(capability_receipt=receipt, runner=lambda _: "{}")


def test_rent_and_promotion_refuse_fabricated_over_cap_account_receipts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(LIFECYCLE.time, "time", lambda: 100)
    image = "ghcr.io/ryanjin333/lehome-groot-n17-trainer@sha256:" + "a" * 64
    evidence = {
        "offer": {"id": 7, "min_bid": .5}, "search_mode": "interruptible",
        "expires_at_unix": 101, "account_hourly_total_usd": 1.01,
        "trainer_image": image,
        "training_capability": {"hardware": "NVIDIA RTX PRO 6000 Blackwell", "driver_version": "595.71.05", "image_digest": image.rpartition("@")[2], "cuda_runtime": "12.8", "torch_cuda": "12.8", "compute_capability": "12.0", "optimizer_step": {"passed": True, "loss": .2}, "nvml": {"utilization_percent": 90}},
    }
    with pytest.raises(ValueError, match=r"exceeds \$1/hr"):
        LIFECYCLE.rent(evidence=evidence, runner=lambda _: "{}")
    capability = {
        "kind": "persistent_training_capability", "instance_id": 9,
        "trainer_image": LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE,
        "provider_response_sha256": "a" * 64,
        "instance": {"instance_id": 9, "trainer_image": LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE, "provider_response_sha256": "a" * 64, "host": "host", "port": 22, "account_hourly_total_usd": 1.01},
        "training_capability": {"hardware": "NVIDIA RTX PRO 6000 Blackwell", "driver_version": "595.71.05", "image_digest": LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE.rpartition("@")[2], "cuda_runtime": "12.8", "torch_cuda": "12.8", "compute_capability": "12.0", "optimizer_step": {"passed": True, "loss": .2}, "nvml": {"utilization_percent": 90}},
    }
    with pytest.raises(ValueError, match=r"exceeds \$1/hr"):
        LIFECYCLE.promote_canary(capability_receipt=capability, runner=lambda _: "{}")


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
        if command[:4] == ("vastai", "--raw", "show", "instances"): return "[]"
        if command[:4] == ("vastai", "--raw", "show", "volumes"): return "[]"
        if command[:4] == ("vastai", "--raw", "create", "instance"): return '{"new_contract":9}'
        if command[:4] == ("vastai", "--raw", "show", "instance"): return "{}" if destroyed else '{"id":9,"actual_status":"running","gpu_name":"RTX PRO 6000 WS","num_gpus":1,"gpu_ram":96000,"dph_total":0.7,"ssh_host":"host","ssh_port":22}'
        if command[:3] == ("vastai", "destroy", "instance"):
            destroyed = True
            return ""
        if command[0] == "scp": return ""
        if command[0] == "ssh": raise RuntimeError("probe failed")
        raise AssertionError(command)
    monkeypatch.setattr(LIFECYCLE.time, "time", lambda: 100)
    with pytest.raises(RuntimeError, match="probe failed"):
        LIFECYCLE.bootstrap_canary(evidence={"offer": {"id": 7, "min_bid": .5, "dph_total": .7}, "search_mode": "interruptible", "expires_at_unix": 101, "account_hourly_total_usd": .7, "trainer_image": LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE, "code_bundle": str(bundle), "code_bundle_sha256_file": str(receipt)}, runner=runner)
    assert any(command[:3] == ("vastai", "destroy", "instance") for command in commands)


def test_materialize_builds_a_verified_sealed_generation(tmp_path: Path) -> None:
    # The lifecycle's free preparation action must exercise the same canonical
    # mix/materialization implementation as production, rather than accepting
    # a hand-written receipt.
    from test_flywheel_mix import _prepared_source

    organizer = _prepared_source(tmp_path / "organizer", kind="organizer", episodes=2)
    corrective_a = _prepared_source(tmp_path / "corrective-a", kind="flywheel", grade="A", episodes=1)
    corrective_b = _prepared_source(tmp_path / "corrective-b", kind="flywheel", grade="B", episodes=1)
    release_receipt = _corrective_release_receipt([corrective_a, corrective_b], tmp_path / "corrective-release.json")
    destination = tmp_path / "generation"
    request = tmp_path / "materialize.json"
    request.write_text(json.dumps({
        "organizer_root": str(organizer),
        "corrective_roots": [str(corrective_a), str(corrective_b)],
        "destination": str(destination),
        "persistent_staging_root": str(tmp_path / "materialize-resume"),
        "seed": 20260812,
        "organizer_source_evidence": LIFECYCLE.ORGANIZER_SOURCE,
        "corrective_release_receipt": str(release_receipt),
    }))

    report = LIFECYCLE.main_for_test(["materialize", "--request", str(request)])

    assert report["paid_action"] is False
    assert report["generation_root"] == str(destination)
    assert report["generation_sha256"] == report["sealed_generation_sha256"]
    assert report["dataset_revision"] == report["dataset_manifest_sha256"][:40]
    assert (destination.with_name(destination.name + ".generation.json")).is_file()


def test_materialize_forwards_optional_video_workers_and_rejects_invalid_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_flywheel_mix import _prepared_source
    import lehome_train.flywheel.mix as mix

    organizer = _prepared_source(tmp_path / "organizer", kind="organizer", episodes=2)
    corrective = _prepared_source(tmp_path / "corrective", kind="flywheel", grade="A", episodes=2)
    receipt = _corrective_release_receipt([corrective], tmp_path / "corrective-release.json")
    request = {
        "organizer_root": str(organizer),
        "corrective_roots": [str(corrective)],
        "destination": str(tmp_path / "generation"),
        "persistent_staging_root": str(tmp_path / "resume"),
        "seed": 1,
        "organizer_source_evidence": LIFECYCLE.ORGANIZER_SOURCE,
        "corrective_release_receipt": str(receipt),
    }
    seen: list[dict[str, object]] = []
    monkeypatch.setattr(mix, "materialize_mixed_snapshot", lambda *_args, **kwargs: seen.append(kwargs) or {})
    monkeypatch.setattr(
        mix,
        "verify_generation",
        lambda _root: {
            "organizer_training_frames": 7,
            "rft_training_frames": 3,
            "mix_plan_sha256": "a" * 64,
            "dataset_manifest_sha256": "b" * 64,
        },
    )

    for video_workers in (16, 24, 32):
        assert LIFECYCLE._materialize(request | {"video_workers": video_workers})["action"] == "materialize"
        assert seen[-1]["video_workers"] == video_workers
    assert LIFECYCLE._materialize(request)["action"] == "materialize"
    assert seen[-1]["video_workers"] == 4
    for invalid in (0, 33, True, 1.0, "8"):
        with pytest.raises(ValueError, match="video_workers"):
            LIFECYCLE._materialize(request | {"video_workers": invalid})


def test_materialize_allows_exact_existing_destination_for_resume_repair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from test_flywheel_mix import _prepared_source
    organizer = _prepared_source(tmp_path / "organizer", kind="organizer", episodes=2)
    corrective = _prepared_source(tmp_path / "corrective", kind="flywheel", grade="A", episodes=2)
    receipt = _corrective_release_receipt([corrective], tmp_path / "corrective-release.json")
    destination, staging = tmp_path / "generation", tmp_path / "resume"
    original = __import__("lehome_train.flywheel.mix", fromlist=["atomic_write_json"]).atomic_write_json
    sibling = destination.with_name(destination.name + ".generation.json")
    monkeypatch.setattr("lehome_train.flywheel.mix.atomic_write_json", lambda path, value: (_ for _ in ()).throw(RuntimeError("receipt interruption")) if path == sibling else original(path, value))
    request = {"organizer_root": str(organizer), "corrective_roots": [str(corrective)], "destination": str(destination), "persistent_staging_root": str(staging), "seed": 1, "organizer_source_evidence": LIFECYCLE.ORGANIZER_SOURCE, "corrective_release_receipt": str(receipt)}
    with pytest.raises(RuntimeError, match="receipt interruption"):
        LIFECYCLE._materialize(request)
    monkeypatch.setattr("lehome_train.flywheel.mix.atomic_write_json", original)
    assert LIFECYCLE._materialize(request)["generation_root"] == str(destination)


def test_materialize_accepts_terminal_retry_after_staging_cleanup(tmp_path: Path) -> None:
    from test_flywheel_mix import _prepared_source
    organizer = _prepared_source(tmp_path / "organizer", kind="organizer", episodes=2)
    corrective = _prepared_source(tmp_path / "corrective", kind="flywheel", grade="A", episodes=2)
    release = _corrective_release_receipt([corrective], tmp_path / "corrective-release.json")
    destination, staging = tmp_path / "generation", tmp_path / "resume"
    request = {"organizer_root": str(organizer), "corrective_roots": [str(corrective)], "destination": str(destination), "persistent_staging_root": str(staging), "seed": 1, "organizer_source_evidence": LIFECYCLE.ORGANIZER_SOURCE, "corrective_release_receipt": str(release)}
    LIFECYCLE._materialize(request)
    assert not staging.exists()
    assert LIFECYCLE._materialize(request)["generation_root"] == str(destination)


def test_derive_corrective_receipt_binds_disposal_proof_to_local_manifest_and_tree(
    tmp_path: Path,
) -> None:
    root = tmp_path / "materialized-rft-16"
    root.mkdir()
    (root / "frame.bin").write_bytes(b"accepted corrective frame")
    (root / "manifest.json").write_text(json.dumps({
        "source_format": "verified_flywheel_rft_release",
        "source_repository": LIFECYCLE.CORRECTIVE_SOURCE["repository"],
        "source_revision": "c" * 40,
        "source_release_id": "d" * 64,
    }), encoding="utf-8")
    disposal = tmp_path / "corrective-rft-disposal-receipt.json"
    disposal.write_text(json.dumps({
        "schema_version": 1, "disposable": True,
        "repository": LIFECYCLE.CORRECTIVE_SOURCE["repository"],
        "immutable_revision": LIFECYCLE.CORRECTIVE_SOURCE["revision"],
        "remote_prefix": LIFECYCLE.CORRECTIVE_SOURCE["prefix"],
        "release_id": LIFECYCLE.CORRECTIVE_SOURCE["prefix"].rpartition("/")[2],
        "fresh_readback_verified": True, "tree_listing_verified": True,
    }), encoding="utf-8")
    output = tmp_path / "corrective-release.json"
    receipt = LIFECYCLE.derive_corrective_receipt(
        disposal_receipt=disposal, snapshot_root=root, output=output,
    )
    assert receipt["published_release"]["revision"] == LIFECYCLE.CORRECTIVE_SOURCE["revision"]
    assert receipt["local_snapshot"]["source_revision"] == "c" * 40
    assert output.is_file() and not output.is_symlink()
    (root / "frame.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="tree"):
        LIFECYCLE._verified_corrective_release_evidence([root], str(output))


def test_prepare_requires_exact_pinned_sources_in_the_sealed_receipt(tmp_path: Path) -> None:
    from test_flywheel_mix import _prepared_source
    organizer = _prepared_source(tmp_path / "organizer", kind="organizer", episodes=2)
    corrective_a = _prepared_source(tmp_path / "corrective-a", kind="flywheel", grade="A", episodes=1)
    corrective_b = _prepared_source(tmp_path / "corrective-b", kind="flywheel", grade="B", episodes=1)
    release_receipt = _corrective_release_receipt([corrective_a, corrective_b], tmp_path / "corrective-release.json")
    root = tmp_path / "generation"
    LIFECYCLE._materialize({"organizer_root": str(organizer), "corrective_roots": [str(corrective_a), str(corrective_b)], "destination": str(root), "persistent_staging_root": str(tmp_path / "generation-resume"), "seed": 1, "organizer_source_evidence": LIFECYCLE.ORGANIZER_SOURCE, "corrective_release_receipt": str(release_receipt)})
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


def test_runtime_mixture_train_uses_only_authenticated_receipts_and_never_legacy_generation(
    tmp_path: Path,
) -> None:
    receipts = {
        "bc": {"repository": "ryanjin333/lehome-groot-n17-data", "immutable_revision": "a" * 40, "remote_prefix": "bc/full", "fresh_readback_verified": True, "tree_listing_verified": True},
        "rollout": {"repository": "ryanjin333/lehome-groot-n17-data", "immutable_revision": "b" * 40, "remote_prefix": "rollouts/round-1", "fresh_readback_verified": True, "tree_listing_verified": True},
        "deployment": {"repository": "ryanjin333/lehome-groot-n17-data", "immutable_revision": "c" * 40, "remote_prefix": "mixtures/" + "d" * 64, "mixture_id": "d" * 64, "pending_receipt_sha256": "e" * 64, "artifact_entries": [{"relative_path": "mixture.json", "sha256": "f" * 64, "byte_size": 1}], "fresh_readback_verified": True, "tree_listing_verified": True},
    }
    paths = {}
    for name, value in receipts.items():
        path = tmp_path / f"{name}.json"; path.write_text(json.dumps(value), encoding="utf-8"); paths[name] = str(path)
    pilot = tmp_path / "pilot.json"
    pilot.write_text(json.dumps(_schema4_pilot(
        instance_id=9, provider_sha="0" * 64, code_revision="1" * 40,
        code_sha="2" * 64, bc_revision="a" * 40, rollout_revision="b" * 40,
        deployment_revision="c" * 40,
    )), encoding="utf-8")
    calls: list[tuple[str, ...]] = []
    def runner(command: tuple[str, ...]) -> str:
        calls.append(command); return "{}"
    instance = {"schema_version": 1, "kind": "runtime_mixture_gpu_warmup_instance", "instance_id": 44, "host": "native-x86", "port": 22, "platform_arch": "x86_64", "trainer_image": LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE, "offer_evidence_sha256": "a" * 64, "provider_response_sha256": "b" * 64, "capability_sha256": "c" * 64}
    output = tmp_path / "execution.json"
    binding = {"mixture": {"repository": LIFECYCLE.CORRECTIVE_SOURCE["repository"], "revision": "c" * 40, "mixture_id": "d" * 64, "manifest_sha256": "6" * 64, "window_index_sha256": "7" * 64, "normalization_sha256": "8" * 64, "source_revisions": {"organizer": "a" * 40, "rollout": "b" * 40}}, "deployment": {"oci_image_digest": LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE.rpartition("@")[2], "provider": "vast", "capability_sha256": "c" * 64}, "code": {"repository_revision": "1" * 40, "bundle_sha256": "2" * 64, "isaac_groot_revision": "9" * 40}, "parent_checkpoint": {"repository": LIFECYCLE.PARENT_CHECKPOINT["repository"], "revision": LIFECYCLE.PARENT_CHECKPOINT["revision"], "subpath": "policies/step-12000", "artifact_sha256": LIFECYCLE.PARENT_CHECKPOINT["artifact_sha256"]}, "physical_batch_size": 64, "action_horizon": 16}
    warmup = tmp_path / "warmup.json"
    warmup.write_text(json.dumps({"kind": "runtime_mixture_gpu_warmup_lifecycle", "instance_id": 44, "provider_response_sha256": "b" * 64, "capability_sha256": "c" * 64, "code_revision": "1" * 40, "code_bundle_sha256": "2" * 64, "parent_checkpoint_artifact_sha256": LIFECYCLE.PARENT_CHECKPOINT["artifact_sha256"], "deployment_revision": "c" * 40, "cpu_pilot_sha256": LIFECYCLE.sha256_file(pilot), "runtime_warmup_binding": binding, "warmup_receipt": {}, "selected_loader_workers": 4}), encoding="utf-8")
    request = {"bc_readback_receipt": paths["bc"], "rollout_readback_receipt": paths["rollout"], "deployment_receipt": paths["deployment"], "pilot_receipt": str(pilot), "warmup_lifecycle_receipt": str(warmup), "code_revision": "1" * 40, "code_bundle_sha256": "2" * 64, "execution_receipt": str(output), "failure_receipt": str(tmp_path / "failure.json")}

    report = LIFECYCLE.remote_action(action="runtime-train", instance=instance, request=request, runner=runner)

    assert report["action"] == "runtime-train"
    command = calls[-1][-1]
    assert "runtime-mixture-train --request /prepared/config/runtime-train.json" in command
    assert "/prepared/generation" not in command and "continuous-train" not in command
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["platform_arch"] == "x86_64" and receipt["deployment_revision"] == "c" * 40


def test_runtime_pilot_provider_plan_is_on_demand_x86_and_never_rents() -> None:
    plan = LIFECYCLE.runtime_mixture_pilot_provider_plan()

    assert plan == {
        "paid_action": False, "action": "runtime-pilot-plan", "provider_action": "not_rented",
        "platform_arch": "x86_64", "purchase_option": "on_demand",
        "account_hourly_cap_usd": 1.0, "max_instances": 1,
    }


def test_runtime_pilot_rejects_a_descriptive_unbound_claim(tmp_path: Path) -> None:
    receipt = tmp_path / "pilot.json"
    receipt.write_text(json.dumps({
        "schema_version": 2, "kind": "runtime_mixture_loader_pilot",
        "model_loaded": False, "gpu_initialized": False, "native_x86_required": True,
        "canonical_worker_counts": [0, 4, 8, 16, 24],
        "canonical_completion": True, "throughput_verified": True,
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="authenticated"):
        LIFECYCLE._validated_runtime_pilot(str(receipt))


def test_runtime_pilot_rejects_legacy_schema3_even_when_its_rows_look_measured() -> None:
    receipt = _schema4_pilot(instance_id=44, provider_sha="a" * 64, code_revision="b" * 40, code_sha="c" * 64, bc_revision="d" * 40, rollout_revision="e" * 40, deployment_revision="f" * 40)
    receipt["schema_version"] = 3
    with pytest.raises(ValueError, match="canonical CPU-only"):
        LIFECYCLE._validated_runtime_pilot_value(receipt)


def test_account_cap_rejects_exactly_one_dollar_per_hour() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        LIFECYCLE._require_account_cap(1.0, label="test")


def test_platform_attestation_requires_native_x86_remote_proof() -> None:
    instance = {"host": "native", "port": 22}
    assert LIFECYCLE._attest_platform_arch(instance, runner=lambda command: "x86_64\n") == "x86_64"
    with pytest.raises(ValueError, match="x86_64"):
        LIFECYCLE._attest_platform_arch(instance, runner=lambda command: "aarch64\n")


def test_capture_runtime_cpu_pilot_offer_uses_exact_on_demand_contract() -> None:
    commands: list[tuple[str, ...]] = []
    offer = {"id": 8, "ask_contract_id": 9, "machine_id": 10, "cpu_arch": "amd64", "cpu_cores_effective": 32, "cpu_ram": 64390, "disk_space": 124.75, "disk_bw": 500, "inet_down": 1000, "reliability": .99, "num_gpus": 1, "dph_total": .18, "storage_total_cost": 0, "is_bid": False, "rentable": True, "rented": False, "gpu_name": "RTX PRO 6000 WS", "gpu_ram": 96000, "driver_version": "x"}
    def runner(command: tuple[str, ...]) -> str:
        commands.append(command)
        if command[2:4] == ("search", "offers"): return json.dumps([offer])
        return "[]"
    receipt = LIFECYCLE.capture_runtime_pilot_offer(runner=runner, now_unix=1)
    assert receipt["offer"]["id"] == 8 and receipt["account_hourly_total_usd"] < 1
    assert "--on-demand" in commands[0] and "--storage" in commands[0]


def test_capture_runtime_pilot_offer_canonicalizes_real_integral_float_cores() -> None:
    """Vast raw JSON reports hardware cores as integral floats on live rows."""
    valid = {
        "id": 45818984, "ask_contract_id": 1, "machine_id": 2,
        "cpu_arch": "amd64", "cpu_cores_effective": 36.0,
        "cpu_ram": 64129, "disk_space": 142, "disk_bw": 500,
        "inet_down": 1000, "reliability": .9973919, "num_gpus": 1,
        "dph_total": .081333, "storage_total_cost": 0, "is_bid": False,
        "rentable": True, "rented": False, "gpu_name": "RTX PRO 6000 WS",
        "gpu_ram": 96000, "driver_version": "x",
    }
    invalid = [
        valid | {"id": 1, "cpu_cores_effective": True, "dph_total": .01},
        valid | {"id": 2, "cpu_cores_effective": 32.5, "dph_total": .01},
        valid | {"id": 3, "cpu_cores_effective": float("nan"), "dph_total": .01},
        valid | {"id": 4, "cpu_cores_effective": float("inf"), "dph_total": .01},
        valid | {"id": 5, "cpu_cores_effective": "36", "dph_total": .01},
    ]

    def runner(command: tuple[str, ...]) -> str:
        if command[2:4] == ("search", "offers"):
            return json.dumps([*invalid, valid])
        if command[2:4] in {("show", "instances"), ("show", "volumes")}:
            return "[]"
        raise AssertionError(command)

    receipt = LIFECYCLE.capture_runtime_pilot_offer(runner=runner, now_unix=1)

    assert receipt["offer"]["id"] == 45818984
    assert receipt["offer"]["cpu_cores_effective"] == 36
    assert type(receipt["offer"]["cpu_cores_effective"]) is int


@pytest.mark.parametrize("value", [True, 32.5, float("nan"), float("inf"), "36"])
def test_runtime_pilot_core_canonicalizer_rejects_non_integral_numeric_values(value: object) -> None:
    with pytest.raises(ValueError, match="core count"):
        LIFECYCLE._canonical_runtime_pilot_cpu_cores(value)


def test_pro6000_offer_query_uses_vast_scaled_gibibyte_predicate() -> None:
    assert LIFECYCLE.OFFER_QUERY == "gpu_ram>=96 num_gpus=1 reliability>=0.95"


def test_capture_runtime_cpu_pilot_offer_rejects_account_total_at_one_dollar() -> None:
    offer = {"id": 8, "cpu_arch": "amd64", "cpu_cores_effective": 32, "cpu_ram": 64390, "disk_space": 124.75, "disk_bw": 500, "reliability": .99, "num_gpus": 1, "dph_total": .18, "is_bid": False, "rentable": True, "rented": False}

    def runner(command: tuple[str, ...]) -> str:
        if command[2:4] == ("search", "offers"):
            return json.dumps([offer])
        if command[2:4] == ("show", "instances"):
            return '[{"dph_total":0.82}]'
        if command[2:4] == ("show", "volumes"):
            return "[]"
        raise AssertionError(command)

    with pytest.raises(ValueError, match="exceeds"):
        LIFECYCLE.capture_runtime_pilot_offer(runner=runner, now_unix=1)


def _runtime_pilot_offer_evidence(*, failure_receipt: str) -> dict[str, object]:
    evidence_root = Path(failure_receipt).parent
    source = {
        "repository": LIFECYCLE.CORRECTIVE_SOURCE["repository"],
        "fresh_readback_verified": True, "tree_listing_verified": True,
    }
    bc = source | {"immutable_revision": "c" * 40, "remote_prefix": "bc/full"}
    rollout = source | {"immutable_revision": "d" * 40, "remote_prefix": "rollouts/round-1"}
    deployment = source | {
        "immutable_revision": "e" * 40, "mixture_id": "f" * 64,
        "remote_prefix": "mixtures/" + "f" * 64,
        "pending_receipt_sha256": "0" * 64, "artifact_entries": ["manifest.json"],
    }
    paths = {"bc": evidence_root / "bc.json", "rollout": evidence_root / "rollout.json", "deployment": evidence_root / "deployment.json"}
    for key, value in (("bc", bc), ("rollout", rollout), ("deployment", deployment)):
        paths[key].write_text(json.dumps(value), encoding="utf-8")
    offer = {
        "id": 8, "ask_contract_id": 9, "machine_id": 10, "cpu_arch": "amd64",
        "cpu_cores_effective": 32, "cpu_ram": 64390, "disk_space": 124.75,
        "disk_bw": 500, "inet_down": 1000, "reliability": .99, "num_gpus": 1,
        "dph_total": .18, "storage_total_cost": 0, "is_bid": False,
        "rentable": True, "rented": False, "gpu_name": "RTX PRO 6000 WS",
        "gpu_ram": 96000, "driver_version": "x",
    }
    return {
        "schema_version": 1, "kind": "runtime_mixture_cpu_pilot_offer", "offer": offer,
        "raw_offer_sha256": LIFECYCLE._hash(offer), "account_hourly_total_usd": .18,
        "captured_at_unix": int(time.time()), "expires_at_unix": int(time.time()) + 60,
        "search_mode": "on_demand", "platform_arch": "amd64", "storage_gb": 120,
        "trainer_image": LIFECYCLE.RUNTIME_CPU_PILOT_IMAGE,
        "image_digest": LIFECYCLE.RUNTIME_CPU_PILOT_IMAGE.rpartition("@")[2],
        "failure_receipt": failure_receipt, "code_revision": "a" * 40,
        "code_bundle_sha256": "b" * 64,
        "bc_readback_receipt": str(paths["bc"]),
        "rollout_readback_receipt": str(paths["rollout"]),
        "deployment_receipt": str(paths["deployment"]),
    }


def test_rent_runtime_cpu_pilot_uses_exact_on_demand_create_and_x86_proof(tmp_path: Path) -> None:
    commands: list[tuple[str, ...]] = []
    readiness_reads = 0
    sleeps: list[float] = []
    live = {
        "id": 44, "actual_status": "running", "cpu_arch": "amd64",
        "cpu_cores_effective": 36.0, "cpu_ram": 64390, "disk_space": 124.75,
        "machine_id": 10, "gpu_name": "RTX PRO 6000 WS", "gpu_ram": 96000,
        "driver_version": "x", "reliability": .99, "num_gpus": 1, "dph_total": .18,
        "ssh_host": "native-x86", "ssh_port": 22,
    }

    def runner(command: tuple[str, ...]) -> str:
        nonlocal readiness_reads
        commands.append(command)
        if command[:4] in {
            ("vastai", "--raw", "show", "instances"),
            ("vastai", "--raw", "show", "volumes"),
        }:
            return "[]"
        if command[:4] == ("vastai", "--raw", "create", "instance"):
            return '{"new_contract":44}'
        if command == ("vastai", "--raw", "show", "instance", "44"):
            readiness_reads += 1
            if readiness_reads < 3:
                return '{"id":44,"actual_status":"loading"}'
            return json.dumps(live)
        if command[-1] == "set -eu; uname -m":
            return "x86_64\n"
        raise AssertionError(command)

    receipt = LIFECYCLE.rent_runtime_cpu_pilot(
        evidence=_runtime_pilot_offer_evidence(failure_receipt=str(tmp_path / "failure.json")), runner=runner, sleep=sleeps.append,
    )

    assert receipt["kind"] == "runtime_mixture_cpu_pilot_instance"
    assert receipt["platform_arch"] == "x86_64"
    assert LIFECYCLE.RUNTIME_PILOT_READINESS_POLLS == 120 and sleeps == [5.0, 5.0]
    assert ("vastai", "--raw", "create", "instance", "8", "--image", LIFECYCLE.RUNTIME_CPU_PILOT_IMAGE, "--disk", "120", "--ssh", "--direct", "--cancel-unavail", "--env", "-e LEHOME_TRAIN_IMAGE=" + LIFECYCLE.RUNTIME_CPU_PILOT_IMAGE) in commands


def test_runtime_pilot_live_readback_accepts_real_null_reliability_but_binds_machine_price_and_gpu() -> None:
    offer = {
        "id": 46568462, "ask_contract_id": 1, "machine_id": 142447,
        "cpu_arch": "amd64", "cpu_cores_effective": 48, "cpu_ram": 128716,
        "disk_space": 120, "disk_bw": 500, "inet_down": 1000,
        "reliability": .9952681, "num_gpus": 1, "dph_total": .144444,
        "is_bid": False, "rentable": True, "rented": False,
        "gpu_name": "Tesla V100", "gpu_ram": 32768, "driver_version": "580.173.02",
    }
    live = {
        "id": 47723784, "actual_status": "running", "machine_id": 142447,
        "cpu_arch": "amd64", "cpu_cores_effective": 48.0, "cpu_ram": 128716,
        "disk_space": 120, "dph_total": .144444, "gpu_name": "Tesla V100",
        "gpu_ram": 32768, "driver_version": "580.173.02", "num_gpus": 1,
        "reliability": None, "ask_contract_id": None, "rented": None,
        "ssh_host": "real-host", "ssh_port": 22,
    }

    assert LIFECYCLE._runtime_pilot_live_matches(live=live, instance_id=47723784, offer=offer)
    assert LIFECYCLE._runtime_pilot_instance_identity(live) == LIFECYCLE._runtime_pilot_instance_identity(live | {"reliability": .1})
    assert not LIFECYCLE._runtime_pilot_live_matches(live=live | {"machine_id": 142448}, instance_id=47723784, offer=offer)
    assert not LIFECYCLE._runtime_pilot_live_matches(live=live | {"dph_total": .144445}, instance_id=47723784, offer=offer)
    assert not LIFECYCLE._runtime_pilot_live_matches(live=live | {"reliability": .97}, instance_id=47723784, offer=offer)
    assert not LIFECYCLE._runtime_pilot_live_matches(live={key: value for key, value in live.items() if key != "actual_status"}, instance_id=47723784, offer=offer)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), float("-inf"), "0.99"])
def test_runtime_pilot_reliability_predicate_rejects_nonfinite_or_non_numeric_values(value: object) -> None:
    with pytest.raises(ValueError, match="reliability"):
        LIFECYCLE._runtime_pilot_reliability(value)


def test_runtime_pilot_sealed_offer_rejects_infinite_reliability(tmp_path: Path) -> None:
    evidence = _runtime_pilot_offer_evidence(failure_receipt=str(tmp_path / "failure.json"))
    evidence["offer"]["reliability"] = float("inf")  # type: ignore[index]
    with pytest.raises(ValueError, match="offer evidence"):
        LIFECYCLE._runtime_pilot_offer(evidence)


def test_rent_runtime_cpu_pilot_mismatch_cleans_only_new_instance_and_proves_absence(tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []
    destroyed = False

    def runner(command: tuple[str, ...]) -> str:
        nonlocal destroyed
        calls.append(command)
        if command[:4] in {
            ("vastai", "--raw", "show", "instances"),
            ("vastai", "--raw", "show", "volumes"),
        }:
            return "[]"
        if command[:4] == ("vastai", "--raw", "create", "instance"):
            return '{"new_contract":44}'
        if command == ("vastai", "--raw", "show", "instance", "44"):
            return "{}" if destroyed else json.dumps({
                "id": 44, "actual_status": "running", "cpu_arch": "aarch64",
                "cpu_cores_effective": 32, "cpu_ram": 64390, "disk_space": 124.75,
                "reliability": .99, "num_gpus": 1, "dph_total": .18,
                "ssh_host": "wrong", "ssh_port": 22,
            })
        if command == ("vastai", "destroy", "instance", "44", "--yes"):
            destroyed = True
            return ""
        raise AssertionError(command)

    with pytest.raises(ValueError, match="readback"):
        LIFECYCLE.rent_runtime_cpu_pilot(
            evidence=_runtime_pilot_offer_evidence(failure_receipt=str(tmp_path / "failure.json")), runner=runner, sleep=lambda _: None,
        )

    assert calls[-2:] == [
        ("vastai", "destroy", "instance", "44", "--yes"),
        ("vastai", "--raw", "show", "instance", "44"),
    ]


def test_rent_runtime_cpu_pilot_timeout_cleans_new_instance_and_proves_absence(tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []
    destroyed = False

    def runner(command: tuple[str, ...]) -> str:
        nonlocal destroyed
        calls.append(command)
        if command[:4] in {
            ("vastai", "--raw", "show", "instances"),
            ("vastai", "--raw", "show", "volumes"),
        }:
            return "[]"
        if command[:4] == ("vastai", "--raw", "create", "instance"):
            return '{"new_contract":44}'
        if command == ("vastai", "--raw", "show", "instance", "44"):
            return "{}" if destroyed else '{"id":44,"actual_status":"loading"}'
        if command == ("vastai", "destroy", "instance", "44", "--yes"):
            destroyed = True
            return ""
        raise AssertionError(command)

    with pytest.raises(ValueError, match="timed out"):
        LIFECYCLE.rent_runtime_cpu_pilot(
            evidence=_runtime_pilot_offer_evidence(failure_receipt=str(tmp_path / "failure.json")), runner=runner,
            max_readiness_polls=1, sleep=lambda _: None,
        )

    assert calls[-2:] == [
        ("vastai", "destroy", "instance", "44", "--yes"),
        ("vastai", "--raw", "show", "instance", "44"),
    ]


def test_runtime_rent_rejects_absent_or_invalid_preflight_evidence_before_provider_calls(tmp_path: Path) -> None:
    """No offer/account/create command may precede immutable campaign validation."""
    calls: list[tuple[str, ...]] = []
    runner = lambda command: calls.append(command) or (_ for _ in ()).throw(AssertionError(command))

    cpu = _runtime_pilot_offer_evidence(failure_receipt=str(tmp_path / "cpu-failure.json"))
    cpu.pop("deployment_receipt")
    with pytest.raises(ValueError, match="receipt path"):
        LIFECYCLE.rent_runtime_cpu_pilot(evidence=cpu, runner=runner)

    gpu = _runtime_pilot_offer_evidence(failure_receipt=str(tmp_path / "gpu-failure.json"))
    invalid_pilot = tmp_path / "invalid-pilot.json"
    invalid_pilot.write_text("{}", encoding="utf-8")
    gpu["pilot_receipt"] = str(invalid_pilot)
    with pytest.raises(ValueError, match="canonical CPU-only"):
        LIFECYCLE.rent_runtime_gpu_warmup(evidence=gpu, runner=runner)

    assert calls == []


def _runtime_pilot_request_files(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    receipts = {
        "bc": {"repository": LIFECYCLE.CORRECTIVE_SOURCE["repository"], "immutable_revision": "a" * 40, "remote_prefix": "bc/full", "fresh_readback_verified": True, "tree_listing_verified": True},
        "rollout": {"repository": LIFECYCLE.CORRECTIVE_SOURCE["repository"], "immutable_revision": "b" * 40, "remote_prefix": "rollouts/round-1", "fresh_readback_verified": True, "tree_listing_verified": True},
        "deployment": {"repository": LIFECYCLE.CORRECTIVE_SOURCE["repository"], "immutable_revision": "c" * 40, "remote_prefix": "mixtures/" + "d" * 64, "mixture_id": "d" * 64, "pending_receipt_sha256": "e" * 64, "artifact_entries": [{"relative_path": "mixture.json", "sha256": "f" * 64, "byte_size": 1}], "fresh_readback_verified": True, "tree_listing_verified": True},
    }
    paths: dict[str, str] = {}
    for name, receipt in receipts.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(receipt), encoding="utf-8")
        paths[name] = str(path)
    instance = {
        "schema_version": 1, "kind": "runtime_mixture_cpu_pilot_instance",
        "instance_id": 44, "host": "native-x86", "port": 22, "platform_arch": "x86_64",
        "trainer_image": LIFECYCLE.RUNTIME_CPU_PILOT_IMAGE,
        "image_digest": LIFECYCLE.RUNTIME_CPU_PILOT_IMAGE.rpartition("@")[2], "offer_evidence_sha256": "1" * 64,
        "provider_response_sha256": "2" * 64, "account_hourly_total_usd": .18,
    }
    request = {
        "bc_readback_receipt": paths["bc"], "rollout_readback_receipt": paths["rollout"],
        "deployment_receipt": paths["deployment"], "code_revision": "3" * 40,
        "code_bundle_sha256": "4" * 64, "lifecycle_receipt": str(tmp_path / "pilot-lifecycle.json"),
    }
    bootstrap = {
        "schema_version": 1, "kind": "runtime_mixture_bootstrap_stage", "instance_id": 44,
        "provider_response_sha256": "2" * 64, "platform_arch": "x86_64",
        "trainer_image": LIFECYCLE.RUNTIME_CPU_PILOT_IMAGE,
        "image_digest": LIFECYCLE.RUNTIME_CPU_PILOT_IMAGE.rpartition("@")[2],
        "code_revision": "3" * 40, "code_bundle_sha256": "4" * 64,
        "bc_revision": "a" * 40, "rollout_revision": "b" * 40,
        "deployment_revision": "c" * 40, "bc_receipt_sha256": LIFECYCLE.sha256_file(Path(paths["bc"])),
        "rollout_receipt_sha256": LIFECYCLE.sha256_file(Path(paths["rollout"])),
        "deployment_receipt_sha256": LIFECYCLE.sha256_file(Path(paths["deployment"])),
        "transfers": [{"name": "code.bundle", "sha256": "5" * 64}],
    }
    bootstrap_path = tmp_path / "bootstrap.json"
    bootstrap_path.write_text(json.dumps(bootstrap), encoding="utf-8")
    request["bootstrap_receipt"] = str(bootstrap_path)
    return instance, request


def _schema4_pilot(*, instance_id: int, provider_sha: str, code_revision: str, code_sha: str, bc_revision: str, rollout_revision: str, deployment_revision: str) -> dict[str, object]:
    return {
        "schema_version": 4, "kind": "runtime_mixture_loader_pilot", "model_loaded": False,
        "gpu_initialized": False, "processor_contract": "pinned_processor_integration_required",
        "representative": {"three_cameras": True, "action_horizon": 16},
        "sample_count_per_worker": 100, "worker_counts": [0, 4, 8, 16, 24],
        "canonical_worker_counts": [0, 4, 8, 16, 24], "loader_throughput": {},
        "timing_rows": [{"worker_count": count, "decoded_samples": 100, "seconds": 1.0, "samples_per_second": 100.0, "host_cpu_seconds": 1.0, "host_max_rss_mib": 1.0, "latency_seconds_p50": .01, "latency_seconds_p95": .02} for count in [0, 4, 8, 16, 24]],
        "authenticated_evidence": {"provider_instance_id": instance_id, "provider_response_sha256": provider_sha, "platform_arch": "x86_64", "image_digest": LIFECYCLE.RUNTIME_CPU_PILOT_IMAGE.rpartition("@")[2], "code_revision": code_revision, "code_bundle_sha256": code_sha, "bc_revision": bc_revision, "rollout_revision": rollout_revision, "deployment_revision": deployment_revision},
        "cache_cap": 1, "native_x86_required": True, "timeout_seconds": 60.0, "canonical_completion": True,
    }


def test_runtime_cpu_pilot_executes_only_loader_cli_and_persists_bound_lifecycle(tmp_path: Path) -> None:
    instance, request = _runtime_pilot_request_files(tmp_path)
    pilot = {
        "schema_version": 4, "kind": "runtime_mixture_loader_pilot", "model_loaded": False,
        "gpu_initialized": False, "processor_contract": "pinned_processor_integration_required",
        "representative": {"bc_window_id": "bc", "rollout_window_id": "rollout", "three_cameras": True, "action_horizon": 16},
        "sample_count_per_worker": 100, "worker_counts": [0, 4, 8, 16, 24],
        "canonical_worker_counts": [0, 4, 8, 16, 24], "loader_throughput": {},
        "timing_rows": [{"worker_count": count, "decoded_samples": 100, "seconds": 1.0, "samples_per_second": 100.0, "host_cpu_seconds": 1.0, "host_max_rss_mib": 1.0, "latency_seconds_p50": .01, "latency_seconds_p95": .02} for count in [0, 4, 8, 16, 24]],
        "authenticated_evidence": {"provider_instance_id": 44, "provider_response_sha256": "2" * 64, "platform_arch": "x86_64", "image_digest": LIFECYCLE.RUNTIME_CPU_PILOT_IMAGE.rpartition("@")[2], "code_revision": "3" * 40, "code_bundle_sha256": "4" * 64, "bc_revision": "a" * 40, "rollout_revision": "b" * 40, "deployment_revision": "c" * 40},
        "cache_cap": 1, "native_x86_required": True, "timeout_seconds": 60.0,
        "canonical_completion": True,
    }
    calls: list[tuple[str, ...]] = []
    report = LIFECYCLE.run_runtime_cpu_pilot(
        instance=instance, request=request,
        runner=lambda command: calls.append(command) or json.dumps(pilot),
    )

    assert report["action"] == "runtime-pilot-run"
    assert "pilot-runtime-mixture --request /prepared/config/runtime-pilot.json" in calls[-1][-1]
    assert "runtime-mixture-train" not in calls[-1][-1] and "continuous-train" not in calls[-1][-1]
    persisted = json.loads(Path(str(request["lifecycle_receipt"])).read_text(encoding="utf-8"))
    assert persisted["instance_id"] == 44
    assert persisted["deployment_revision"] == "c" * 40
    assert persisted["pilot_receipt"]["gpu_initialized"] is False
    assert persisted["trainer_image"] == LIFECYCLE.RUNTIME_CPU_PILOT_IMAGE
    assert "parent_checkpoint_artifact_sha256" not in persisted
    assert persisted["staged_transfers"] == [{"name": "code.bundle", "sha256": "5" * 64}]


def test_runtime_cpu_pilot_destroy_requires_untampered_bound_pilot_lifecycle(tmp_path: Path) -> None:
    instance, request = _runtime_pilot_request_files(tmp_path)
    lifecycle = {
        "schema_version": 1, "kind": "runtime_mixture_cpu_pilot_lifecycle", "instance_id": 44,
        "provider_response_sha256": "2" * 64, "platform_arch": "x86_64",
        "trainer_image": LIFECYCLE.RUNTIME_CPU_PILOT_IMAGE,
        "image_digest": LIFECYCLE.RUNTIME_CPU_PILOT_IMAGE.rpartition("@")[2], "code_revision": "3" * 40,
        "code_bundle_sha256": "4" * 64, "bc_revision": "a" * 40, "rollout_revision": "b" * 40,
        "deployment_revision": "c" * 40, "deployment_receipt_sha256": LIFECYCLE.sha256_file(Path(str(request["deployment_receipt"]))),
        "bootstrap_receipt_sha256": LIFECYCLE.sha256_file(Path(str(request["bootstrap_receipt"]))),
        "staged_transfers": [{"name": "code.bundle", "sha256": "5" * 64}],
        "pilot_receipt": _schema4_pilot(instance_id=44, provider_sha="2" * 64, code_revision="3" * 40, code_sha="4" * 64, bc_revision="a" * 40, rollout_revision="b" * 40, deployment_revision="c" * 40),
    }
    calls: list[tuple[str, ...]] = []
    absence_reads = 0
    sleeps: list[float] = []
    def runner(command: tuple[str, ...]) -> str:
        nonlocal absence_reads
        calls.append(command)
        if command[:4] == ("vastai", "--raw", "show", "instance"):
            absence_reads += 1
            return '{"id":44,"actual_status":"exiting"}' if absence_reads == 1 else '{"instances":null}'
        return ""

    assert LIFECYCLE.destroy_runtime_cpu_pilot(instance_id=44, lifecycle_receipt=lifecycle, runner=runner, max_absence_polls=2, sleep=sleeps.append)["destroy_authorized"] is True
    assert calls.count(("vastai", "destroy", "instance", "44", "--yes")) == 1 and sleeps == [5.0]
    lifecycle["pilot_receipt"]["authenticated_evidence"]["provider_instance_id"] = 45  # type: ignore[index]
    with pytest.raises(ValueError, match="pilot"):
        LIFECYCLE.destroy_runtime_cpu_pilot(instance_id=44, lifecycle_receipt=lifecycle, runner=runner)


def test_runtime_cpu_bootstrap_receipt_rejects_source_or_deployment_receipt_drift(tmp_path: Path) -> None:
    instance, request = _runtime_pilot_request_files(tmp_path)
    path = Path(str(request["bootstrap_receipt"]))
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["deployment_revision"] = "f" * 40
    path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="CPU pilot bootstrap"):
        LIFECYCLE._runtime_bootstrap_receipt(path=path, instance=instance, request=request)


@pytest.mark.parametrize("value", [{}, None, {"instances": None}])
def test_runtime_absence_parser_accepts_only_canonical_vast_absence_shapes(value: object) -> None:
    assert LIFECYCLE._runtime_instance_is_absent(value) is True


@pytest.mark.parametrize("value", [{"instances": []}, {"instances": {}}, {"instances": None, "id": 44}, [], {"id": 44}])
def test_runtime_absence_parser_rejects_noncanonical_provider_shapes(value: object) -> None:
    assert LIFECYCLE._runtime_instance_is_absent(value) is False


def test_runtime_cpu_pilot_cli_actions_are_explicitly_gated_and_never_use_legacy_train(tmp_path: Path) -> None:
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"instance_id": 44}), encoding="utf-8")

    report = LIFECYCLE.main_for_test(["runtime-pilot-destroy", "--request", str(request)])

    assert report == {"paid_action": False, "action": "runtime-pilot-destroy", "dry_run": True, "request": {"instance_id": 44}}


def test_runtime_checkpoint_actions_are_explicit_dry_run_boundaries(tmp_path: Path) -> None:
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"instance": {"instance_id": 44}}), encoding="utf-8")

    report = LIFECYCLE.main_for_test([
        "runtime-checkpoint-replacement-resume", "--request", str(request),
    ])

    assert report["paid_action"] is False
    assert report["action"] == "runtime-checkpoint-replacement-resume"
    assert report["dry_run"] is True


def test_runtime_bootstrap_stage_is_an_explicit_pre_pilot_action(tmp_path: Path) -> None:
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"instance": {"instance_id": 44}}), encoding="utf-8")

    report = LIFECYCLE.main_for_test(["runtime-bootstrap-stage", "--request", str(request)])

    assert report["paid_action"] is False and report["action"] == "runtime-bootstrap-stage"


def test_runtime_cpu_bootstrap_stages_only_cpu_pilot_inputs_and_never_parent_material(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance, request = _runtime_pilot_request_files(tmp_path)
    bundle, bundle_sha, token = tmp_path / "code.bundle", tmp_path / "code.bundle.sha256", tmp_path / "runtime.token"
    bundle.write_bytes(b"reviewed bundle")
    bundle_sha.write_text("4" * 64 + "  code.bundle\n", encoding="utf-8")
    token.write_text("private-token", encoding="utf-8")
    token.chmod(0o600)
    hydrate = tmp_path / "hydrate.json"
    hydrate.write_text(json.dumps({"schema_version": 1, "command": "hydrate-runtime-mixture", "arguments": {"deployment_receipt": "/prepared/config/deployment-receipt.json", "source_readback_receipts": "/prepared/config", "destination": "/prepared/runtime", "mounts_descriptor": "/prepared/runtime/mounts.json"}}), encoding="utf-8")
    pilot = tmp_path / "pilot.json"
    pilot.write_text(json.dumps({"schema_version": 1, "command": "pilot-runtime-mixture", "arguments": {"mixture_manifest": "/prepared/runtime/mixture.json", "mounts_descriptor": "/prepared/runtime/mounts.json", "sample_count": 100, "worker_counts": [0, 4, 8, 16, 24], "timeout_seconds": 60, "authenticated_evidence": {}}}), encoding="utf-8")
    output = tmp_path / "cpu-bootstrap.json"
    request |= {
        "code_bundle": str(bundle), "code_bundle_sha256_file": str(bundle_sha), "token_file": str(token),
        "runtime_hydrate_request": str(hydrate), "runtime_pilot_request": str(pilot),
        "bootstrap_receipt": str(output),
    }
    monkeypatch.setattr(LIFECYCLE, "_verify_reviewed_code_bundle", lambda *_args: "4" * 64)
    sources = {
        "code.bundle": bundle, "code.bundle.sha256": bundle_sha, "runtime.token": token,
        "runtime-hydrate.json": hydrate, "runtime-pilot.json": pilot,
        "bc-readback.json": Path(str(request["bc_readback_receipt"])),
        "rollout-readback.json": Path(str(request["rollout_readback_receipt"])),
        "deployment-receipt.json": Path(str(request["deployment_receipt"])),
    }
    calls: list[tuple[str, ...]] = []
    def runner(command: tuple[str, ...]) -> str:
        calls.append(command)
        if command[0] == "scp":
            return ""
        if command[0] == "ssh" and command[-1].startswith("sha256sum "):
            name = command[-1].rpartition("/")[2]
            return LIFECYCLE.sha256_file(sources[name]) + "  " + name + "\n"
        return ""

    result = LIFECYCLE.runtime_mixture_bootstrap_stage(instance=instance, request=request, runner=runner)

    receipt = result["bootstrap_receipt"]
    assert receipt["trainer_image"] == LIFECYCLE.RUNTIME_CPU_PILOT_IMAGE
    assert receipt["image_digest"] == LIFECYCLE.RUNTIME_CPU_PILOT_IMAGE.rpartition("@")[2]
    assert {item["name"] for item in receipt["transfers"]} == set(sources)
    assert "parent_checkpoint_artifact_sha256" not in receipt
    setup = calls[-1][-1]
    assert "cd /tmp/lehome-runtime-bootstrap; sha256sum -c code.bundle.sha256" in setup
    assert "git clone --quiet --no-checkout /tmp/lehome-runtime-bootstrap/code.bundle /prepared/code" in setup
    assert "checkout --quiet --detach " + "3" * 40 in setup
    assert "chmod 600 /prepared/config/runtime.token" in setup
    assert "parent.tar" not in setup and "modality.py" not in setup and "launch.json" not in setup
    assert "experiment.json" not in setup and "cuda" not in setup.lower()
    assert LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE not in setup and "private-token" not in setup


def test_runtime_cli_dispatch_never_calls_legacy_recuts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the executable action dispatch: bootstrap -> 1K -> loss -> resume -> 2K -> dispose."""
    token = tmp_path / "token"
    token.write_text("hf_fake", encoding="utf-8")
    token.chmod(0o600)
    terminal = tmp_path / "terminal.json"
    terminal.write_text("{}", encoding="utf-8")
    request_path = tmp_path / "request.json"
    instance = {"kind": "runtime_mixture_gpu_warmup_instance", "instance_id": 44, "provider_response_sha256": "a" * 64}
    request = {"instance": instance, "failure_receipt": str(tmp_path / "failure.json"), "terminal_receipt": str(terminal), "resume_destination": str(tmp_path / "hydrate"), "checkpoint_repository": LIFECYCLE.PARENT_CHECKPOINT["repository"], "checkpoint_revision": "b" * 40, "checkpoint_experiment_id": "runtime", "checkpoint_artifact_root": str(tmp_path / "artifacts")}
    request_path.write_text(json.dumps(request), encoding="utf-8")
    calls: list[str] = []
    monkeypatch.setattr(LIFECYCLE, "runtime_mixture_bootstrap_stage", lambda **kwargs: calls.append("bootstrap") or {"paid_action": True})
    monkeypatch.setattr(LIFECYCLE, "run_runtime_cpu_pilot", lambda **kwargs: calls.append("pilot") or {"paid_action": True})
    monkeypatch.setattr(LIFECYCLE, "run_runtime_gpu_warmup", lambda **kwargs: calls.append("warmup") or {"paid_action": True})
    monkeypatch.setattr(LIFECYCLE, "remote_action", lambda **kwargs: calls.append(str(kwargs["action"])) or {"paid_action": True, "action": kwargs["action"]})
    monkeypatch.setattr(LIFECYCLE, "publish_runtime_checkpoint", lambda **kwargs: calls.append("publish") or {"paid_action": True})
    monkeypatch.setattr(LIFECYCLE, "runtime_anchor_interruption_terminal", lambda **kwargs: calls.append("interrupted") or {"paid_action": True})
    monkeypatch.setattr(LIFECYCLE, "runtime_checkpoint_terminal", lambda **kwargs: calls.append("complete") or {"terminal": {"kind": "fake"}})
    monkeypatch.setattr(LIFECYCLE, "resume_runtime_checkpoint", lambda **kwargs: calls.append("resume") or {"paid_action": True})
    monkeypatch.setattr(LIFECYCLE, "destroy_runtime_checkpoint_completion", lambda **kwargs: calls.append("dispose") or {"paid_action": True})
    monkeypatch.setattr(LIFECYCLE, "_runtime_checkpoint_terminal_output", lambda request, terminal: dict(terminal))

    def runner(command: tuple[str, ...]) -> str:
        return "{}" if command[:4] == ("vastai", "--raw", "show", "instance") else ""

    def invoke(action: str) -> None:
        LIFECYCLE.main_for_test([action, "--request", str(request_path), "--execute", "--token-file", str(token)], runner=runner)

    invoke("runtime-bootstrap-stage")
    invoke("runtime-pilot-run")
    invoke("runtime-gpu-warmup")
    invoke("runtime-train")
    invoke("runtime-checkpoint-publish")
    invoke("runtime-checkpoint-interrupted")
    invoke("runtime-checkpoint-replacement-resume")
    invoke("runtime-train")
    invoke("runtime-checkpoint-publish")
    invoke("runtime-checkpoint-complete")
    invoke("runtime-checkpoint-dispose")

    assert calls == ["runtime-bootstrap-stage", "pilot", "warmup", "runtime-train", "publish", "interrupted", "resume", "runtime-train", "publish", "complete", "dispose"]


def test_runtime_abort_cleanup_writes_redacted_bound_non_disposable_receipt(tmp_path: Path) -> None:
    output = tmp_path / "abort.json"
    calls: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...]) -> str:
        calls.append(command)
        return "{}" if command[:4] == ("vastai", "--raw", "show", "instance") else ""

    LIFECYCLE._runtime_abort_cleanup(
        instance={"kind": "runtime_mixture_gpu_warmup_instance", "instance_id": 44, "provider_response_sha256": "a" * 64},
        request={"failure_receipt": str(output), "code_revision": "b" * 40, "code_bundle_sha256": "c" * 64},
        error=RuntimeError("HF_TOKEN secret-value"), runner=runner,
    )

    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["instance_id"] == 44 and receipt["disposable"] is False
    assert receipt["cleanup_status"] == "destroyed_and_absent"
    assert receipt["error"] == "redacted remote failure"
    assert calls[0] == ("vastai", "destroy", "instance", "44", "--yes")


def test_runtime_abort_cleanup_polls_transitional_destroy_until_absent(tmp_path: Path) -> None:
    output = tmp_path / "abort-poll.json"
    absent_reads = iter(['{"id":44,"actual_status":"exiting"}', "{}"])
    calls: list[tuple[str, ...]] = []
    sleeps: list[float] = []

    def runner(command: tuple[str, ...]) -> str:
        calls.append(command)
        if command == ("vastai", "destroy", "instance", "44", "--yes"):
            return ""
        if command == ("vastai", "--raw", "show", "instance", "44"):
            return next(absent_reads)
        raise AssertionError(command)

    LIFECYCLE._runtime_abort_cleanup(
        instance={"instance_id": 44, "provider_response_sha256": "a" * 64},
        request={"failure_receipt": str(output), "code_revision": "b" * 40, "code_bundle_sha256": "c" * 64},
        error=ValueError("staged failure"), runner=runner, max_absence_polls=2, sleep=sleeps.append,
    )

    assert calls == [
        ("vastai", "destroy", "instance", "44", "--yes"),
        ("vastai", "--raw", "show", "instance", "44"),
        ("vastai", "--raw", "show", "instance", "44"),
    ]
    assert sleeps == [5.0]
    assert json.loads(output.read_text())["cleanup_status"] == "destroyed_and_absent"


def test_runtime_abort_cleanup_times_out_when_destroyed_row_persists(tmp_path: Path) -> None:
    output = tmp_path / "abort-timeout.json"
    calls: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...]) -> str:
        calls.append(command)
        if command == ("vastai", "destroy", "instance", "44", "--yes"):
            return ""
        if command == ("vastai", "--raw", "show", "instance", "44"):
            return '{"id":44,"actual_status":"exiting"}'
        raise AssertionError(command)

    with pytest.raises(RuntimeError, match="did not verify instance absence"):
        LIFECYCLE._runtime_abort_cleanup(
            instance={"instance_id": 44, "provider_response_sha256": "a" * 64},
            request={"failure_receipt": str(output), "code_revision": "b" * 40, "code_bundle_sha256": "c" * 64},
            error=ValueError("staged failure"), runner=runner, max_absence_polls=2, sleep=lambda _: None,
        )

    assert calls.count(("vastai", "destroy", "instance", "44", "--yes")) == 1
    assert json.loads(output.read_text())["cleanup_status"] == "absence_unverified"


def test_checkpoint_disposal_persistent_absence_never_double_destroys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once terminal disposal issues destroy, its failure path owns cleanup."""
    import lehome_train.groot.runtime_checkpoint_lifecycle as checkpoint_lifecycle

    output = tmp_path / "dispose-failure.json"
    monkeypatch.setattr(LIFECYCLE, "_runtime_checkpoint_identity", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(checkpoint_lifecycle, "authorize_runtime_mixture_disposal", lambda **_kwargs: {"authorized": True})
    calls: list[tuple[str, ...]] = []
    def runner(command: tuple[str, ...]) -> str:
        calls.append(command)
        if command == ("vastai", "destroy", "instance", "44", "--yes"):
            return ""
        if command == ("vastai", "--raw", "show", "instance", "44"):
            return '{"id":44,"actual_status":"exiting"}'
        raise AssertionError(command)

    with pytest.raises(RuntimeError, match="did not verify instance absence"):
        LIFECYCLE.destroy_runtime_checkpoint_completion(
            instance={"instance_id": 44}, request={"failure_receipt": str(output), "code_revision": "a" * 40, "code_bundle_sha256": "b" * 64},
            terminal={}, hub=object(), runner=runner, max_absence_polls=2, sleep=lambda _: None,
        )

    assert calls.count(("vastai", "destroy", "instance", "44", "--yes")) == 1
    assert json.loads(output.read_text())["cleanup_status"] == "absence_unverified"


def test_runtime_abort_cleanup_retains_non_disposable_receipt_when_destroy_fails(tmp_path: Path) -> None:
    output = tmp_path / "abort-failed.json"

    with pytest.raises(RuntimeError, match="could not destroy"):
        LIFECYCLE._runtime_abort_cleanup(
            instance={"instance_id": 44, "provider_response_sha256": "a" * 64},
            request={"failure_receipt": str(output), "code_revision": "b" * 40, "code_bundle_sha256": "c" * 64},
            error=ValueError("stage failed"),
            runner=lambda command: (_ for _ in ()).throw(OSError("provider down")),
        )

    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["disposable"] is False and receipt["cleanup_status"] == "destroy_failed"


def test_runtime_paid_actions_fail_closed_without_a_failure_receipt() -> None:
    with pytest.raises(ValueError, match="failure_receipt"):
        LIFECYCLE._runtime_abort_on_failure(
            instance={"kind": "runtime_mixture_cpu_pilot_instance", "instance_id": 44},
            request={}, runner=lambda command: "",
            operation=lambda: {"paid_action": True},
        )


def test_runtime_selected_workers_are_checked_against_unmodified_launch_config(tmp_path: Path) -> None:
    selected, launch = tmp_path / "selected.json", tmp_path / "launch.json"
    selected.write_text(json.dumps({"selected_loader_workers": 4}), encoding="utf-8")
    launch.write_text(json.dumps({"dataloader_num_workers": 4}), encoding="utf-8")

    workers, identity = LIFECYCLE._runtime_stage_selected_workers(selected_path=selected, launch_path=launch)

    assert workers == 4 and identity == LIFECYCLE.sha256_file(selected)
    launch.write_text(json.dumps({"dataloader_num_workers": 8}), encoding="utf-8")
    with pytest.raises(ValueError, match="workers"):
        LIFECYCLE._runtime_stage_selected_workers(selected_path=selected, launch_path=launch)


def test_runtime_provider_loss_requires_two_fresh_absence_readbacks() -> None:
    calls: list[tuple[str, ...]] = []

    loss = LIFECYCLE.classify_runtime_provider_loss(
        instance={"instance_id": 44},
        runner=lambda command: calls.append(command) or "{}",
    )

    assert loss is not None and loss["kind"] == "instance_absent"
    assert calls == [
        ("vastai", "--raw", "show", "instance", "44"),
        ("vastai", "--raw", "show", "instance", "44"),
    ]


def test_runtime_gpu_warmup_runs_exact_cli_and_binds_cpu_pilot_code_parent_and_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance, request = _runtime_pilot_request_files(tmp_path)
    instance |= {"kind": "runtime_mixture_gpu_warmup_instance", "trainer_image": LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE, "capability_sha256": "5" * 64}
    pilot = {
        "schema_version": 4, "kind": "runtime_mixture_loader_pilot", "model_loaded": False, "gpu_initialized": False,
        "processor_contract": "pinned_processor_integration_required", "representative": {"three_cameras": True, "action_horizon": 16}, "sample_count_per_worker": 100, "worker_counts": [0, 4, 8, 16, 24], "canonical_worker_counts": [0, 4, 8, 16, 24], "loader_throughput": {str(n): {"decoded_samples": 100, "samples_per_second": 1.0} for n in [0, 4, 8, 16, 24]}, "timing_rows": [{"worker_count": n, "decoded_samples": 100, "seconds": 1.0, "samples_per_second": 1.0, "host_cpu_seconds": 1.0, "host_max_rss_mib": 1.0, "latency_seconds_p50": .01, "latency_seconds_p95": .02} for n in [0, 4, 8, 16, 24]], "authenticated_evidence": {"provider_instance_id": 44, "provider_response_sha256": "2" * 64, "platform_arch": "x86_64", "image_digest": LIFECYCLE.RUNTIME_CPU_PILOT_IMAGE.rpartition("@")[2], "code_revision": "3" * 40, "code_bundle_sha256": "4" * 64, "bc_revision": "a" * 40, "rollout_revision": "b" * 40, "deployment_revision": "c" * 40}, "cache_cap": 1, "native_x86_required": True, "timeout_seconds": 60.0, "canonical_completion": True,
    }
    pilot_path = tmp_path / "pilot.json"; pilot_path.write_text(json.dumps(pilot), encoding="utf-8")
    binding = {"mixture": {"repository": LIFECYCLE.CORRECTIVE_SOURCE["repository"], "revision": "c" * 40, "mixture_id": "d" * 64, "manifest_sha256": "6" * 64, "window_index_sha256": "7" * 64, "normalization_sha256": "8" * 64, "source_revisions": {"organizer": "a" * 40, "rollout": "b" * 40}}, "deployment": {"oci_image_digest": LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE.rpartition("@")[2], "provider": "vast", "capability_sha256": "5" * 64}, "code": {"repository_revision": "3" * 40, "bundle_sha256": "4" * 64, "isaac_groot_revision": "9" * 40}, "parent_checkpoint": {"repository": LIFECYCLE.PARENT_CHECKPOINT["repository"], "revision": LIFECYCLE.PARENT_CHECKPOINT["revision"], "subpath": "policies/step-12000", "artifact_sha256": LIFECYCLE.PARENT_CHECKPOINT["artifact_sha256"]}, "physical_batch_size": 64, "action_horizon": 16}
    binding_path = tmp_path / "binding.json"; binding_path.write_text(json.dumps(binding), encoding="utf-8")
    request |= {"pilot_receipt": str(pilot_path), "runtime_warmup_binding": str(binding_path), "warmup_lifecycle_receipt": str(tmp_path / "warmup-lifecycle.json")}
    monkeypatch.setattr("lehome_train.groot.runtime_mixture_warmup.validate_warmup_binding", lambda value: dict(value))
    monkeypatch.setattr("lehome_train.groot.runtime_mixture_warmup.validate_gpu_warmup_receipt", lambda receipt, **kwargs: 4)
    calls: list[tuple[str, ...]] = []
    report = LIFECYCLE.run_runtime_gpu_warmup(instance=instance, request=request, runner=lambda command: calls.append(command) or json.dumps({"kind": "runtime_mixture_gpu_warmup"}))

    assert report["action"] == "runtime-gpu-warmup"
    assert "runtime-gpu-warmup --request /prepared/config/runtime-warmup.json" in calls[-1][-1]
    lifecycle = json.loads(Path(str(request["warmup_lifecycle_receipt"])).read_text())
    assert lifecycle["selected_loader_workers"] == 4
    assert lifecycle["trainer_image"] == LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE
    assert lifecycle["parent_checkpoint_artifact_sha256"] == LIFECYCLE.PARENT_CHECKPOINT["artifact_sha256"]
