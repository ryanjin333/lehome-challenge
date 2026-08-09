from __future__ import annotations

import json
import os
import stat
from dataclasses import replace
from pathlib import Path

import pytest

from b1k_deploy.dockerhub import CommandResult, DockerImageRelease
from b1k_deploy.production_smoke import ProductionSmokeError, SshSmokeRemote, VastCliSmokeClient, _rollout_command, _training_command
from b1k_deploy.publish import canonical_payload_hash, load_canonical_template
from b1k_deploy.smoke import SmokeTemplatePublicationReceipt

WORKSPACE = Path(__file__).resolve().parents[2]


def training_release(reference: str) -> DockerImageRelease:
    digest = reference.rsplit("@", 1)[1]
    return DockerImageRelease("training", "docker.io/ryanjin333/behavior1k-groot-n17", "trainer-" + "a" * 40, "a" * 40, digest, reference)


def rollout_release(reference: str) -> DockerImageRelease:
    digest = reference.rsplit("@", 1)[1]
    return DockerImageRelease("rollout", "docker.io/ryanjin333/behavior1k-groot-n17", "rollout-" + "a" * 40, "a" * 40, digest, reference)


def release_receipts() -> dict[str, DockerImageRelease]:
    return {
        "training_release": training_release("docker.io/ryanjin333/behavior1k-groot-n17@sha256:" + "a" * 64),
        "rollout_release": rollout_release("docker.io/ryanjin333/behavior1k-groot-n17@sha256:" + "b" * 64),
    }


def test_unified_repository_requires_explicit_role_receipts(tmp_path: Path) -> None:
    runner = VastRunner()
    client = _client(tmp_path, runner)
    identity = tmp_path / "id"
    identity.write_text("private", encoding="utf-8")
    identity.chmod(0o600)

    with pytest.raises(ProductionSmokeError, match="release receipt"):
        SshSmokeRemote(
            vast=client,
            identity_file=identity,
            known_hosts=tmp_path / "campaign" / "known_hosts",
            training_image="docker.io/ryanjin333/behavior1k-groot-n17@sha256:" + "a" * 64,
            rollout_image="docker.io/ryanjin333/behavior1k-groot-n17@sha256:" + "b" * 64,
            hub_verifier=object(),
        )


@pytest.mark.parametrize(
    "changes",
    (
        {"repository": "docker.io/ryanjin333/not-the-b1k-repository"},
        {"digest": "sha256:" + "c" * 64},
        {"tag": "trainer-" + "b" * 40},
        {"source_commit": "b" * 40},
    ),
)
def test_unified_repository_rejects_partial_training_release_identity(tmp_path: Path, changes: dict[str, str]) -> None:
    runner = VastRunner()
    client = _client(tmp_path, runner)
    identity = tmp_path / "id"
    identity.write_text("private", encoding="utf-8")
    identity.chmod(0o600)
    receipts = release_receipts()
    receipts["training_release"] = replace(receipts["training_release"], **changes)

    with pytest.raises(ProductionSmokeError, match="release receipt"):
        SshSmokeRemote(
            vast=client,
            identity_file=identity,
            known_hosts=tmp_path / "campaign" / "known_hosts",
            training_image="docker.io/ryanjin333/behavior1k-groot-n17@sha256:" + "a" * 64,
            rollout_image="docker.io/ryanjin333/behavior1k-groot-n17@sha256:" + "b" * 64,
            **receipts,
            hub_verifier=object(),
        )


def test_unified_repository_rejects_cross_source_role_pair(tmp_path: Path) -> None:
    runner = VastRunner()
    client = _client(tmp_path, runner)
    identity = tmp_path / "id"
    identity.write_text("private", encoding="utf-8")
    identity.chmod(0o600)
    receipts = release_receipts()
    receipts["rollout_release"] = replace(
        receipts["rollout_release"],
        tag="rollout-" + "b" * 40,
        source_commit="b" * 40,
    )

    with pytest.raises(ProductionSmokeError, match="source commit"):
        SshSmokeRemote(
            vast=client,
            identity_file=identity,
            known_hosts=tmp_path / "campaign" / "known_hosts",
            training_image="docker.io/ryanjin333/behavior1k-groot-n17@sha256:" + "a" * 64,
            rollout_image="docker.io/ryanjin333/behavior1k-groot-n17@sha256:" + "b" * 64,
            **receipts,
            hub_verifier=object(),
        )

class VastRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], float]] = []
        self.instances: list[dict[str, object]] = []
        self.template = load_canonical_template("training", source_root=WORKSPACE)
        self.template["image"] = "docker.io/ryanjin333/behavior1k-groot-n17@sha256:" + "a" * 64
        self.template["env"] = self.template["env"].replace("CONTAINER_DIGEST=sha256:" + "0" * 64, "CONTAINER_DIGEST=sha256:" + "a" * 64)

    def run(self, arguments, *, stdin, env, timeout):
        self.calls.append((arguments, timeout))
        if arguments[1:3] == ("search", "offers"):
            payload = [
                {"id": 11, "dph_total": 0.40, "gpu_name": "slow", "verification": "verified", "vericode": 1, "num_gpus": 1, "cpu_arch": "amd64", "cuda_max_good": 12.4, "compute_cap": 800, "gpu_ram": 24576, "driver_version": "550.54.14", "disk_space": 100.9, "cpu_ram": 32768, "inet_down": 1000.9, "duration": 3600.0},
                {"id": 12, "dph_total": 0.20, "gpu_name": "fast", "verification": "verified", "vericode": 1, "num_gpus": 1, "cpu_arch": "amd64", "cuda_max_good": 12.4, "compute_cap": 800, "gpu_ram": 24576, "driver_version": "550.54.14", "disk_space": 200.9, "cpu_ram": 65536, "inet_down": 1000.9, "duration": 3600.0},
            ]
        elif arguments[2:4] == ("search", "templates"):
            payload = [{**self.template, "id": 123, "hash_id": "canonical_template_hash"}]
        elif arguments[2:4] == ("create", "instance"):
            payload = {"success": True, "new_contract": 9876543}
        elif arguments[2:4] == ("show", "instances"):
            payload = self.instances
        elif arguments[2:4] == ("destroy", "instance"):
            payload = {"success": True}
        else:
            raise AssertionError(arguments)
        return CommandResult(0, json.dumps(payload), "")


def _client(tmp_path: Path, runner: VastRunner) -> VastCliSmokeClient:
    executable = tmp_path / "vastai"; executable.write_text("#!", encoding="utf-8"); executable.chmod(0o700)
    key = tmp_path / "vast_api_key"; key.write_text("not-a-real-key\n", encoding="utf-8"); key.chmod(0o600)
    return VastCliSmokeClient(vastai_executable=executable, api_key_file=key, runner=runner)


def test_selects_the_cheapest_verified_compatible_one_gpu_offer_and_binds_provider_template_hash(tmp_path: Path) -> None:
    runner = VastRunner()
    client = _client(tmp_path, runner)

    offer = client.select_offer("training-smoke")
    instance_id = client.create_instance({"offer_id": offer.offer_id, "template_id": "123", "idempotency_key": "b1k-smoke-" + "a" * 32, "hourly_rate_usd": offer.hourly_rate_usd, "disk_gb": 100, "purpose": "training-smoke", "image_reference": runner.template["image"], "payload_hash": canonical_payload_hash(runner.template)}, timeout_seconds=30)

    assert offer.offer_id == "12"
    assert "gpu_ram>=24" in runner.calls[0][0][3]
    assert "gpu_ram>=12288" not in runner.calls[0][0][3]
    assert offer.compatibility.ram_gb == 64
    assert offer.compatibility.maximum_duration_minutes == 60
    assert instance_id == "9876543"
    create = runner.calls[-1][0]
    assert create == (
        str(tmp_path / "vastai"), "--raw", "create", "instance", "12", "--template_hash", "canonical_template_hash",
        "--disk", "100", "--label", "b1k-smoke-" + "a" * 32, "--bid_price", "0.2", "--cancel-unavail", "--env", "-e B1K_TRAINING_SMOKE_RUNTIME=1",
    )


def test_rollout_create_uses_environment_smoke_mode_without_an_incomplete_runtime_argument(tmp_path: Path) -> None:
    runner = VastRunner(); client = _client(tmp_path, runner)
    runner.template = load_canonical_template("rollout", source_root=WORKSPACE)
    runner.template["image"] = "docker.io/ryanjin333/behavior1k-groot-n17@sha256:" + "b" * 64
    runner.template["env"] = runner.template["env"].replace("CONTAINER_DIGEST=sha256:" + "0" * 64, "CONTAINER_DIGEST=sha256:" + "b" * 64)
    client.select_offer("rollout-smoke")
    assert "gpu_ram>=24" in runner.calls[0][0][3]
    client.create_instance({"offer_id": "12", "template_id": "123", "idempotency_key": "b1k-smoke-" + "c" * 32, "hourly_rate_usd": "0.20", "disk_gb": 300, "purpose": "rollout-smoke", "image_reference": runner.template["image"], "payload_hash": canonical_payload_hash(runner.template)}, timeout_seconds=30)
    create = runner.calls[-1][0]
    assert "--args" not in create
    assert create[-2:] == ("--env", "-e B1K_ROLLOUT_SMOKE_RUNTIME=1")
    assert ("--disk", "300") in zip(create, create[1:])


@pytest.mark.parametrize("target", ["47198086", "*", "b1k-smoke-" + "a" * 32, "12;destroy instance 47198086"])
def test_destroy_rejects_any_non_exact_or_protected_target_without_running_cli(tmp_path: Path, target: str) -> None:
    runner = VastRunner(); client = _client(tmp_path, runner)
    with pytest.raises(ProductionSmokeError):
        client.destroy_instance(target, timeout_seconds=30)
    assert runner.calls == []


def test_reconciliation_requires_one_exact_label_match_and_endpoint_is_exact_id_bound(tmp_path: Path) -> None:
    runner = VastRunner(); client = _client(tmp_path, runner)
    key = "b1k-smoke-" + "b" * 32
    runner.instances = [{"id": 222, "label": key, "ssh_host": "203.0.113.9", "ssh_port": 2222, "ssh_user": "root"}]
    assert client.find_instance_by_idempotency_key(key, timeout_seconds=30) == "222"
    endpoint = client.endpoint("222", timeout_seconds=30)
    assert (endpoint.instance_id, endpoint.host, endpoint.port) == ("222", "203.0.113.9", 2222)


def test_ssh_requires_private_identity_and_uses_strict_campaign_known_hosts(tmp_path: Path) -> None:
    runner = VastRunner(); client = _client(tmp_path, runner)
    identity = tmp_path / "id"; identity.write_text("private", encoding="utf-8"); identity.chmod(0o600)
    calls: list[tuple[str, ...]] = []
    def ssh_run(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "ssh-keyscan":
            return type("Done", (), {"returncode": 0, "stdout": "[203.0.113.9]:2222 ssh-ed25519 key\n", "stderr": ""})()
        return type("Done", (), {"returncode": 0, "stdout": "", "stderr": ""})()
    class Hub:
        def bootstrap_probe(self, *args, **kwargs): raise AssertionError("not a runtime contract")
    remote = SshSmokeRemote(
        vast=client, identity_file=identity, known_hosts=tmp_path / "campaign" / "known_hosts",
        training_image="docker.io/ryanjin333/behavior1k-groot-n17@sha256:" + "a" * 64,
        rollout_image="docker.io/ryanjin333/behavior1k-groot-n17@sha256:" + "b" * 64,
        **release_receipts(),
        hub_verifier=Hub(), runner=ssh_run,
    )
    remote._endpoints["222"] = __import__("b1k_deploy.production_smoke", fromlist=["VastInstanceEndpoint"]).VastInstanceEndpoint("222", "203.0.113.9", 2222)
    remote._ssh("222", ("true",), 10)
    command = calls[-1]
    assert "StrictHostKeyChecking=yes" in command
    assert "GlobalKnownHostsFile=/dev/null" in command
    assert "-i" in command and str(identity) in command
    assert identity.read_text(encoding="utf-8") not in command

    identity.chmod(0o644)
    with pytest.raises(ProductionSmokeError, match="private"):
        SshSmokeRemote(vast=client, identity_file=identity, known_hosts=tmp_path / "bad", training_image="docker.io/ryanjin333/behavior1k-groot-n17@sha256:" + "a" * 64, rollout_image="docker.io/ryanjin333/behavior1k-groot-n17@sha256:" + "b" * 64, **release_receipts(), hub_verifier=Hub(), runner=ssh_run)


def test_image_local_commands_do_not_assume_a_nested_docker_daemon_or_fabricate_simulator_evidence() -> None:
    training = _training_command("docker.io/ryanjin333/behavior1k-groot-n17@sha256:" + "a" * 64, "b1k-bootstrap-" + "a" * 32 + "-smoke-model")
    rollout = _rollout_command("docker.io/ryanjin333/behavior1k-groot-n17@sha256:" + "b" * 64, "b1k-bootstrap-" + "b" * 32 + "-success-fixture", "b1k-bootstrap-" + "b" * 32 + "-failure-fixture")

    assert training[0] == "/opt/runtime/bin/python"
    assert "docker" not in training
    assert training[1] == "/opt/b1k-launchkit/training_smoke.py"
    source = (Path(__file__).parents[2] / "trainer" / "b1k_launchkit" / "training_smoke.py").read_text(encoding="utf-8")
    assert "scripts/b1k/train_b1k.py" in source
    assert '"torchrun"' in source
    assert "_checkpoint_bucket_probe" in source
    assert '"upload"' in source and '"download"' in source and '"delete"' in source
    assert rollout[:3] == ("/opt/conda/envs/behavior/bin/python", "-m", "b1k_rollout.cli")
    assert "docker" not in rollout
    assert "smoke-runtime" in rollout


def test_runtime_transport_uses_only_role_specific_sanitized_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = VastRunner(); client = _client(tmp_path, runner)
    identity = tmp_path / "id"; identity.write_text("private", encoding="utf-8"); identity.chmod(0o600)

    class Hub:
        def verify_remote_probe(self, *_args: object, **_kwargs: object) -> object:
            raise AssertionError("not used by runtime transport")

    remote = SshSmokeRemote(
        vast=client, identity_file=identity, known_hosts=tmp_path / "campaign" / "known_hosts",
        training_image="docker.io/ryanjin333/behavior1k-groot-n17@sha256:" + "a" * 64,
        rollout_image="docker.io/ryanjin333/behavior1k-groot-n17@sha256:" + "b" * 64,
        **release_receipts(),
        hub_verifier=Hub(), runner=lambda *_args, **_kwargs: None,
    )
    captured: list[tuple[str, ...]] = []
    monkeypatch.setenv("MUST_NOT_LEAK", "ambient-secret")

    def ssh(_instance_id: str, command: tuple[str, ...], _timeout_seconds: int) -> object:
        captured.append(command)
        digest = next(value.removeprefix("CONTAINER_DIGEST=") for value in command if value.startswith("CONTAINER_DIGEST="))
        return type("Completed", (), {"stdout": json.dumps({"container_digest": digest}) + "\n"})()

    monkeypatch.setattr(remote, "_ssh", ssh)
    remote._runtime_json("9876543", _training_command(remote._training_image, "b1k-bootstrap-" + "a" * 32 + "-smoke-model"), 60, remote._training_image)
    remote._runtime_json("9876543", _rollout_command(remote._rollout_image, "b1k-bootstrap-" + "b" * 32 + "-success-fixture", "b1k-bootstrap-" + "b" * 32 + "-failure-fixture"), 60, remote._rollout_image)

    training, rollout = captured
    for command in (training, rollout):
        assert command[:6] == ("setpriv", "--reuid=10001", "--regid=10001", "--init-groups", "env", "-i")
        assert "MUST_NOT_LEAK=ambient-secret" not in command
    assert "OMNI_KIT_ACCEPT_EULA=YES" not in training
    assert "OMNIGIBSON_DATA_PATH=/workspace/omnigibson-data" not in training
    assert "OMNI_KIT_ACCEPT_EULA=YES" in rollout
    assert "OMNIGIBSON_DATA_PATH=/workspace/omnigibson-data" in rollout


def test_lost_training_runtime_evidence_reconciles_its_exact_image_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = VastRunner(); client = _client(tmp_path, runner)
    identity = tmp_path / "id"; identity.write_text("private", encoding="utf-8"); identity.chmod(0o600)
    calls: list[tuple[str, str]] = []

    class Hub:
        def reconcile_remote_probe(self, role: str, _repository: object, *, prefix: str) -> None:
            calls.append((role, prefix))

    remote = SshSmokeRemote(
        vast=client, identity_file=identity, known_hosts=tmp_path / "campaign" / "known_hosts",
        training_image="docker.io/ryanjin333/behavior1k-groot-n17@sha256:" + "a" * 64,
        rollout_image="docker.io/ryanjin333/behavior1k-groot-n17@sha256:" + "b" * 64,
        **release_receipts(),
        hub_verifier=Hub(), runner=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(remote, "_runtime_json", lambda *_args: (_ for _ in ()).throw(ProductionSmokeError("runtime evidence lost")))

    with pytest.raises(ProductionSmokeError, match="runtime evidence lost"):
        remote.run_training_contract("b1k-smoke-" + "a" * 32, "9876543", 60)

    assert calls == [("model", "b1k-bootstrap-" + "a" * 32 + "-smoke-model")]


def test_lost_rollout_runtime_evidence_reconciles_both_probes_and_preserves_primary_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = VastRunner(); client = _client(tmp_path, runner)
    identity = tmp_path / "id"; identity.write_text("private", encoding="utf-8"); identity.chmod(0o600)
    calls: list[tuple[str, str]] = []

    class Hub:
        def reconcile_remote_probe(self, role: str, _repository: object, *, prefix: str) -> None:
            calls.append((role, prefix))
            if prefix.endswith("success-fixture"):
                raise RuntimeError("success cleanup failed")

    remote = SshSmokeRemote(
        vast=client, identity_file=identity, known_hosts=tmp_path / "campaign" / "known_hosts",
        training_image="docker.io/ryanjin333/behavior1k-groot-n17@sha256:" + "a" * 64,
        rollout_image="docker.io/ryanjin333/behavior1k-groot-n17@sha256:" + "b" * 64,
        **release_receipts(),
        hub_verifier=Hub(), runner=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(remote, "_runtime_json", lambda *_args: (_ for _ in ()).throw(ProductionSmokeError("runtime evidence lost")))

    with pytest.raises(ProductionSmokeError, match="runtime evidence lost") as error:
        remote.run_rollout_contract("b1k-smoke-" + "b" * 32, "9876543", 60)

    assert calls == [
        ("dataset", "b1k-bootstrap-" + "b" * 32 + "-success-fixture"),
        ("dataset", "b1k-bootstrap-" + "b" * 32 + "-failure-fixture"),
    ]
    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == "success cleanup failed"


def test_training_missing_post_execution_probe_evidence_reconciles_and_preserves_the_primary_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = VastRunner(); client = _client(tmp_path, runner)
    identity = tmp_path / "id"; identity.write_text("private", encoding="utf-8"); identity.chmod(0o600)
    calls: list[str] = []

    class Hub:
        def reconcile_remote_probe(self, _role: str, _repository: object, *, prefix: str) -> None:
            calls.append(prefix)
            raise RuntimeError("reconciliation failed")

    remote = SshSmokeRemote(
        vast=client, identity_file=identity, known_hosts=tmp_path / "campaign" / "known_hosts",
        training_image="docker.io/ryanjin333/behavior1k-groot-n17@sha256:" + "a" * 64,
        rollout_image="docker.io/ryanjin333/behavior1k-groot-n17@sha256:" + "b" * 64,
        **release_receipts(),
        hub_verifier=Hub(), runner=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(remote, "_runtime_json", lambda *_args: {
        "runtime_uid": 10001, "token_file_uid": 10001, "token_file_mode": 0o600,
        "gpu_count": 1, "optimizer_steps": 1, "lifecycle_preflight": "passed",
        "container_digest": "sha256:" + "a" * 64, "checkpoint_bucket_probe": "passed",
    })

    with pytest.raises(ProductionSmokeError, match="remote image did not return") as error:
        remote.run_training_contract("b1k-smoke-" + "c" * 32, "9876543", 60)

    assert calls == ["b1k-bootstrap-" + "c" * 32 + "-smoke-model"]
    assert isinstance(error.value.__cause__, RuntimeError)


def test_rollout_missing_post_execution_probe_evidence_reconciles_both_exact_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = VastRunner(); client = _client(tmp_path, runner)
    identity = tmp_path / "id"; identity.write_text("private", encoding="utf-8"); identity.chmod(0o600)
    calls: list[str] = []

    class Hub:
        def reconcile_remote_probe(self, _role: str, _repository: object, *, prefix: str) -> None:
            calls.append(prefix)

    remote = SshSmokeRemote(
        vast=client, identity_file=identity, known_hosts=tmp_path / "campaign" / "known_hosts",
        training_image="docker.io/ryanjin333/behavior1k-groot-n17@sha256:" + "a" * 64,
        rollout_image="docker.io/ryanjin333/behavior1k-groot-n17@sha256:" + "b" * 64,
        **release_receipts(),
        hub_verifier=Hub(), runner=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(remote, "_runtime_json", lambda *_args: {
        "gpu_count": 1, "eula_environment": "OMNI_KIT_ACCEPT_EULA=YES", "warp_runtime": "bundled-compatible",
        "headless_loads": 1, "resets": 1, "rgb_observation_count": 3, "action_mapping_count": 1,
        "evaluator_outcome": "terminal", "container_digest": "sha256:" + "b" * 64,
    })

    with pytest.raises(ProductionSmokeError, match="both immutable fixture upload commits"):
        remote.run_rollout_contract("b1k-smoke-" + "d" * 32, "9876543", 60)

    assert calls == [
        "b1k-bootstrap-" + "d" * 32 + "-success-fixture",
        "b1k-bootstrap-" + "d" * 32 + "-failure-fixture",
    ]


def test_ephemeral_template_creation_retries_exact_name_readback_after_an_empty_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = VastRunner(); client = _client(tmp_path, runner)
    production_payload = dict(runner.template)
    production = SmokeTemplatePublicationReceipt("123", training_release(runner.template["image"]), canonical_payload_hash(production_payload))
    name = "b1k-training-smoke-" + "c" * 32
    rows = [[], [{"id": 456, "name": name}]]
    created_payload: dict[str, object] = {}

    monkeypatch.setattr(client, "attest_template_binding", lambda **_kwargs: "provider")
    monkeypatch.setattr(client, "_template_registry_repo", lambda _template_id: "docker.io")
    def create_command(payload: dict[str, object], *, docker_login_repo: str) -> tuple[str, ...]:
        assert docker_login_repo == "docker.io"
        created_payload.update(payload)
        return ("create",)
    monkeypatch.setattr(client, "_template_create_command", create_command)
    monkeypatch.setattr(client, "_create_template_id", lambda _command: "456")
    monkeypatch.setattr(client, "_template_readback", lambda template_id: production_payload if template_id == "123" else created_payload)
    monkeypatch.setattr(client, "_rows", lambda _arguments, timeout_seconds=None: rows.pop(0))

    receipt = client.create_ephemeral_smoke_template("training", production, name=name)

    assert receipt.template_id == "456"


def test_ephemeral_template_propagates_only_the_registered_private_registry_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = VastRunner()
    client = _client(tmp_path, runner)
    production_payload = dict(runner.template)
    production = SmokeTemplatePublicationReceipt(
        "123", training_release(runner.template["image"]), canonical_payload_hash(production_payload)
    )
    name = "b1k-training-smoke-" + "a" * 32
    captured: dict[str, object] = {}

    monkeypatch.setattr(client, "attest_template_binding", lambda **_kwargs: "provider")
    monkeypatch.setattr(client, "_template_registry_repo", lambda _template_id: "docker.io")

    def create_command(payload: dict[str, object], *, docker_login_repo: str) -> tuple[str, ...]:
        captured["payload"] = payload
        captured["docker_login_repo"] = docker_login_repo
        return ("create",)

    monkeypatch.setattr(client, "_template_create_command", create_command)
    monkeypatch.setattr(client, "_create_template_id", lambda _command: "456")
    created_payload: dict[str, object] = {}
    monkeypatch.setattr(
        client,
        "_template_readback",
        lambda template_id: production_payload if template_id == "123" else created_payload,
    )

    def rows(_arguments: tuple[str, ...], timeout_seconds: int | None = None) -> list[dict[str, object]]:
        created_payload.update(captured["payload"])
        return [{"id": 456, "name": name}]

    monkeypatch.setattr(client, "_rows", rows)

    receipt = client.create_ephemeral_smoke_template("training", production, name=name)

    assert receipt.template_id == "456"
    assert captured["docker_login_repo"] == "docker.io"
    assert "docker_login_repo" not in captured["payload"]


def test_private_registry_reference_is_read_back_and_passed_without_credentials(tmp_path: Path) -> None:
    runner = VastRunner()
    runner.template["docker_login_repo"] = "docker.io"
    client = _client(tmp_path, runner)

    assert client._template_registry_repo("123") == "docker.io"
    command = client._template_create_command(
        load_canonical_template("training", source_root=WORKSPACE),
        docker_login_repo="docker.io",
    )

    assert ("--login", "docker.io") in zip(command, command[1:])
    assert not any("dckr_pat" in item.casefold() or "private-token" in item.casefold() for item in command)


@pytest.mark.parametrize("registry_reference", [None, "hub.docker.com", "docker.io private-token"])
def test_private_registry_reference_rejects_missing_or_unapproved_values(
    tmp_path: Path, registry_reference: str | None
) -> None:
    runner = VastRunner()
    runner.template["docker_login_repo"] = registry_reference
    client = _client(tmp_path, runner)

    with pytest.raises(ProductionSmokeError, match="approved private registry reference"):
        client._template_registry_repo("123")


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ("New Template: 456\n", "456"),
        ("{\"id\":456}\n", None),
        ("new Template: 456\n", None),
        ("New Template: 456\nextra\n", None),
    ],
)
def test_template_create_parser_accepts_only_the_documented_anchored_raw_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stdout: str, expected: str | None
) -> None:
    runner = VastRunner(); client = _client(tmp_path, runner)
    monkeypatch.setattr(runner, "run", lambda *_args, **_kwargs: CommandResult(0, stdout, ""))

    if expected is None:
        with pytest.raises(ProductionSmokeError, match="exact template ID"):
            client._create_template_id(("create", "template"))
    else:
        assert client._create_template_id(("create", "template")) == expected


def test_ephemeral_template_create_id_with_failed_readback_is_destroyed_and_proven_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = VastRunner(); client = _client(tmp_path, runner)
    production_payload = dict(runner.template)
    production = SmokeTemplatePublicationReceipt("123", training_release(runner.template["image"]), canonical_payload_hash(production_payload))
    name = "b1k-training-smoke-" + "d" * 32
    commands: list[tuple[str, ...]] = []
    queries: list[str] = []

    monkeypatch.setattr(client, "attest_template_binding", lambda **_kwargs: "provider")
    monkeypatch.setattr(client, "_template_registry_repo", lambda _template_id: "docker.io")
    monkeypatch.setattr(client, "_template_create_command", lambda _payload, *, docker_login_repo: ("create",))
    def json_call(command: tuple[str, ...], _timeout: int) -> object:
        commands.append(command)
        return {}
    monkeypatch.setattr(client, "_json", json_call)
    monkeypatch.setattr(client, "_create_template_id", lambda _command: "456")
    monkeypatch.setattr(client, "_template_readback", lambda template_id: production_payload if template_id == "123" else (_ for _ in ()).throw(ProductionSmokeError("exact readback failed")))
    def rows(arguments: tuple[str, ...], timeout_seconds: int | None = None) -> list[dict[str, object]]:
        queries.append(arguments[-1])
        return []
    monkeypatch.setattr(client, "_rows", rows)

    with pytest.raises(ProductionSmokeError, match="readback failed"):
        client.create_ephemeral_smoke_template("training", production, name=name)

    assert any(command[-1] == "456" for command in commands if command[:4] == (str(tmp_path / "vastai"), "--raw", "destroy", "template"))
    assert f"id==456" in queries
    assert f"name=={name}" in queries


def test_ephemeral_template_create_error_reconciles_exact_name_and_preserves_primary_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = VastRunner(); client = _client(tmp_path, runner)
    production_payload = dict(runner.template)
    production = SmokeTemplatePublicationReceipt("123", training_release(runner.template["image"]), canonical_payload_hash(production_payload))
    name = "b1k-training-smoke-" + "e" * 32
    commands: list[tuple[str, ...]] = []
    queries: list[str] = []

    monkeypatch.setattr(client, "attest_template_binding", lambda **_kwargs: "provider")
    monkeypatch.setattr(client, "_template_registry_repo", lambda _template_id: "docker.io")
    monkeypatch.setattr(client, "_template_create_command", lambda _payload, *, docker_login_repo: ("create",))
    def json_call(command: tuple[str, ...], _timeout: int) -> object:
        commands.append(command)
        return {}
    monkeypatch.setattr(client, "_json", json_call)
    monkeypatch.setattr(client, "_create_template_id", lambda _command: (_ for _ in ()).throw(ProductionSmokeError("create response lost")))
    monkeypatch.setattr(client, "_template_readback", lambda _template_id: production_payload)
    def rows(arguments: tuple[str, ...], timeout_seconds: int | None = None) -> list[dict[str, object]]:
        query = arguments[-1]
        queries.append(query)
        return [{"id": 456, "name": name}] if query == f"name=={name}" and len([item for item in queries if item == query]) == 1 else []
    monkeypatch.setattr(client, "_rows", rows)

    with pytest.raises(ProductionSmokeError, match="create response lost"):
        client.create_ephemeral_smoke_template("training", production, name=name)

    assert any(command[-1] == "456" for command in commands if command[:4] == (str(tmp_path / "vastai"), "--raw", "destroy", "template"))
    assert f"id==456" in queries
    assert f"name=={name}" in queries


def test_ephemeral_template_create_id_is_cleaned_when_name_reconciliation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = VastRunner(); client = _client(tmp_path, runner)
    production_payload = dict(runner.template)
    production = SmokeTemplatePublicationReceipt("123", training_release(runner.template["image"]), canonical_payload_hash(production_payload))
    name = "b1k-training-smoke-" + "f" * 32
    commands: list[tuple[str, ...]] = []
    name_reads = 0

    monkeypatch.setattr(client, "attest_template_binding", lambda **_kwargs: "provider")
    monkeypatch.setattr(client, "_template_registry_repo", lambda _template_id: "docker.io")
    monkeypatch.setattr(client, "_template_create_command", lambda _payload, *, docker_login_repo: ("create",))
    def json_call(command: tuple[str, ...], _timeout: int) -> object:
        commands.append(command)
        return {}
    monkeypatch.setattr(client, "_json", json_call)
    monkeypatch.setattr(client, "_create_template_id", lambda _command: "456")
    monkeypatch.setattr(client, "_template_readback", lambda _template_id: production_payload)
    def rows(arguments: tuple[str, ...], timeout_seconds: int | None = None) -> list[dict[str, object]]:
        nonlocal name_reads
        if arguments[-1] == f"name=={name}":
            name_reads += 1
            if name_reads == 1:
                raise ProductionSmokeError("provider name readback lost")
        return []
    monkeypatch.setattr(client, "_rows", rows)

    with pytest.raises(ProductionSmokeError, match="provider name readback lost"):
        client.create_ephemeral_smoke_template("training", production, name=name)

    assert any(command[-1] == "456" for command in commands if command[:4] == (str(tmp_path / "vastai"), "--raw", "destroy", "template"))
    assert name_reads > 1


def test_rollout_contract_reconciles_both_exact_fixtures_when_the_first_verification_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = VastRunner()
    client = _client(tmp_path, runner)
    identity = tmp_path / "id"; identity.write_text("private", encoding="utf-8"); identity.chmod(0o600)

    class Hub:
        def verify_remote_probe(self, *_args: object, **_kwargs: object) -> object:
            raise AssertionError("the test replaces the remote-probe boundary")

        def reconcile_remote_probe(self, *_args: object, **_kwargs: object) -> None:
            pass

    remote = SshSmokeRemote(
        vast=client, identity_file=identity, known_hosts=tmp_path / "campaign" / "known_hosts",
        training_image="docker.io/ryanjin333/behavior1k-groot-n17@sha256:" + "a" * 64,
        rollout_image="docker.io/ryanjin333/behavior1k-groot-n17@sha256:" + "b" * 64,
        **release_receipts(),
        hub_verifier=Hub(), runner=lambda *_args, **_kwargs: None,
    )
    payload = {
        "gpu_count": 1, "eula_environment": "OMNI_KIT_ACCEPT_EULA=YES", "warp_runtime": "bundled-compatible",
        "headless_loads": 1, "resets": 1, "rgb_observation_count": 3, "action_mapping_count": 1,
        "evaluator_outcome": "terminal", "container_digest": "sha256:" + "b" * 64,
        "remote_probe_upload_commits": {"success-fixture": "1" * 40, "failure-fixture": "2" * 40},
    }
    calls: list[str] = []

    monkeypatch.setattr(remote, "_runtime_json", lambda *_args: payload)
    def probe(_role: str, classification: str, _run_id: str, _payload: object) -> object:
        calls.append(classification)
        if classification == "success-fixture":
            raise ProductionSmokeError("first fixture verification failed")
        raise ProductionSmokeError("second fixture cleanup failed")
    monkeypatch.setattr(remote, "_remote_probe", probe)

    with pytest.raises(ProductionSmokeError, match="first fixture verification failed") as error:
        remote.run_rollout_contract("b1k-smoke-" + "c" * 32, "9876543", 60)

    assert calls == ["success-fixture", "failure-fixture"]
    assert isinstance(error.value.__cause__, ProductionSmokeError)
    assert str(error.value.__cause__) == "second fixture cleanup failed"
