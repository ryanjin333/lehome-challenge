from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from b1k_rollout.contracts import RolloutContract
from b1k_rollout.controller import (
    CheckpointReceipt,
    ControllerError,
    FilesystemCheckpointSource,
    RolloutController,
    SubprocessGroupLauncher,
    bounded_http_health_probe,
    build_official_evaluator_command,
    checkpoint_tree_sha256,
    hydrate_immutable_checkpoint,
)
from b1k_rollout.episodes import write_episode_envelope
from b1k_rollout.identity import BEHAVIOR_REVISION, DATASET_REPO, GROOT_REVISION, MODEL_REPO
from b1k_rollout.outcomes import classify_outcome
from b1k_rollout.provenance import ProvenanceAuthenticator
from b1k_rollout.publisher import publish_release
from b1k_rollout.task_manifest import load_task_manifest


ROLLOUT = Path(__file__).parents[1]


def _contract() -> RolloutContract:
    return RolloutContract(
        behavior_revision=BEHAVIOR_REVISION,
        groot_revision=GROOT_REVISION,
        model_repository=MODEL_REPO,
        model_commit="a" * 40,
        dataset_repository=DATASET_REPO,
        image_digest="sha256:" + "b" * 64,
        run_id="run-001",
        cycle_id="cycle-001",
        campaign_id="campaign-001",
        evaluator_mode="public_test",
        task_manifest_sha256="7ab5ee6ef1c5e48b421f4dac6ef45537f081b78feba0735ae1c805f529462d92",
        checkpoint_artifact_sha256="d" * 64,
    )


class FakeCheckpointSource:
    def __init__(self, receipt: CheckpointReceipt) -> None:
        self.receipt = receipt
        self.calls: list[tuple[str, str, str, Path]] = []

    def download(self, *, repository: str, revision: str, destination: Path) -> CheckpointReceipt:
        self.calls.append(("download", repository, revision, destination))
        destination.mkdir(parents=True, exist_ok=True)
        return self.receipt

    def readback(self, *, repository: str, revision: str, destination: Path) -> CheckpointReceipt:
        self.calls.append(("readback", repository, revision, destination))
        return self.receipt


def test_checkpoint_hydration_requires_matching_immutable_readback(tmp_path: Path) -> None:
    contract = _contract()
    receipt = CheckpointReceipt(
        model_commit=contract.model_commit,
        artifact_sha256=contract.checkpoint_artifact_sha256,
        local_path=tmp_path / "checkpoint",
    )
    source = FakeCheckpointSource(receipt)

    hydrated = hydrate_immutable_checkpoint(source, contract=contract, destination=receipt.local_path)

    assert hydrated == receipt
    assert source.calls == [
        ("download", MODEL_REPO, contract.model_commit, receipt.local_path),
        ("readback", MODEL_REPO, contract.model_commit, receipt.local_path),
    ]


def test_checkpoint_hydration_rejects_readback_identity_drift(tmp_path: Path) -> None:
    contract = _contract()
    destination = tmp_path / "checkpoint"
    source = FakeCheckpointSource(
        CheckpointReceipt(contract.model_commit, contract.checkpoint_artifact_sha256, destination)
    )
    source.readback = lambda **_: CheckpointReceipt("b" * 40, contract.checkpoint_artifact_sha256, destination)  # type: ignore[method-assign]

    with pytest.raises(ControllerError, match="checkpoint readback"):
        hydrate_immutable_checkpoint(source, contract=contract, destination=destination)


def test_filesystem_checkpoint_source_downloads_and_freshly_readbacks_one_revision(
    tmp_path: Path,
) -> None:
    revision = "a" * 40
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "weights.bin").write_bytes(b"immutable checkpoint")
    source = FilesystemCheckpointSource({revision: source_root})
    checksum = source.download(
        repository=MODEL_REPO, revision=revision, destination=tmp_path / "checksum"
    ).artifact_sha256
    contract = replace(_contract(), checkpoint_artifact_sha256=checksum)

    receipt = hydrate_immutable_checkpoint(
        source, contract=contract, destination=tmp_path / "hydrated"
    )

    assert receipt.artifact_sha256 == checksum
    assert (receipt.local_path / "weights.bin").read_bytes() == b"immutable checkpoint"


def test_filesystem_checkpoint_adapter_reuses_only_the_exact_canonical_tree(tmp_path: Path) -> None:
    revision = "a" * 40
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "nested").mkdir()
    (source_root / "nested" / "weights.bin").write_bytes(b"immutable checkpoint")
    source = FilesystemCheckpointSource({revision: source_root})
    destination = tmp_path / "destination"

    first = source.download(repository=MODEL_REPO, revision=revision, destination=destination)
    resumed = source.download(repository=MODEL_REPO, revision=revision, destination=destination)

    assert first == resumed
    assert first.artifact_sha256 == checkpoint_tree_sha256(destination)
    (destination / "nested" / "weights.bin").write_bytes(b"partial restart corruption")
    with pytest.raises(ControllerError, match="does not match immutable source"):
        source.download(repository=MODEL_REPO, revision=revision, destination=destination)


def test_filesystem_checkpoint_adapter_binds_trainer_final_manifest_not_tree_hash(tmp_path: Path) -> None:
    revision = "a" * 40
    source_root = tmp_path / "trainer-finalize"
    checkpoint = source_root / "checkpoint"
    checkpoint.mkdir(parents=True)
    payload = b"trainer-finalized-weight"
    weight = checkpoint / "model.safetensors"
    weight.write_bytes(payload)
    manifest = {
        "schema_version": 1,
        "files": {"checkpoint/model.safetensors": {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}},
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
    (source_root / "final-manifest.json").write_bytes(manifest_bytes)
    source = FilesystemCheckpointSource({revision: source_root})
    contract = replace(_contract(), checkpoint_artifact_sha256=hashlib.sha256(manifest_bytes).hexdigest())

    receipt = hydrate_immutable_checkpoint(
        source, contract=contract, destination=tmp_path / "materialized-checkpoint"
    )

    assert receipt.artifact_sha256 == hashlib.sha256(manifest_bytes).hexdigest()
    assert receipt.local_tree_sha256 == checkpoint_tree_sha256(receipt.local_path)
    assert (receipt.local_path / "model.safetensors").read_bytes() == payload


def test_subprocess_group_launcher_and_bounded_health_probe_are_real_bounded_adapters(
    tmp_path: Path,
) -> None:
    launcher = SubprocessGroupLauncher()
    process = launcher.start(
        (sys.executable, "-c", "import time; time.sleep(60)"),
        environment={},
        cwd=tmp_path,
    )
    try:
        launcher.terminate_group(process, timeout_seconds=1.0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    assert process.poll() is not None
    assert bounded_http_health_probe("http://127.0.0.1:1/healthz", 0.1) is False


def test_subprocess_launcher_keeps_required_simulator_runtime_environment_but_strips_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = SubprocessGroupLauncher()
    data_path = tmp_path / "omnigibson-data"
    data_path.mkdir()
    monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", "YES")
    monkeypatch.setenv("OMNIGIBSON_DATA_PATH", str(data_path))
    monkeypatch.setenv("B1K_HF_TOKEN", "must-not-reach-child")
    output = tmp_path / "environment.json"
    process = launcher.start(
        (
            sys.executable,
            "-c",
            "import json,os,pathlib; pathlib.Path('environment.json').write_text(json.dumps(dict(os.environ)))",
        ),
        environment={"CUDA_VISIBLE_DEVICES": "3", "B1K_HF_TOKEN": "also-secret"},
        cwd=tmp_path,
    )
    assert process.wait(2.0) == 0
    child_environment = json.loads(output.read_text(encoding="utf-8"))
    assert child_environment["OMNI_KIT_ACCEPT_EULA"] == "YES"
    assert child_environment["OMNIGIBSON_DATA_PATH"] == str(data_path)
    assert child_environment["CUDA_VISIBLE_DEVICES"] == "3"
    assert "PATH" in child_environment
    assert "B1K_HF_TOKEN" not in child_environment


def test_subprocess_group_launcher_escalates_for_a_term_ignoring_descendant(tmp_path: Path) -> None:
    launcher = SubprocessGroupLauncher()
    child = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    parent = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); time.sleep(60)"
    )
    process = launcher.start((sys.executable, "-c", parent), environment={}, cwd=tmp_path)
    try:
        launcher.terminate_group(process, timeout_seconds=1.0)
        with pytest.raises(ProcessLookupError):
            os.killpg(process.pid, 0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_official_evaluator_command_preserves_r1pro_public_test_protocol(tmp_path: Path) -> None:
    command = build_official_evaluator_command(
        task_name="turning_on_radio",
        mode="public_test",
        instance_index=3,
        port=8123,
        output_dir=tmp_path,
    )

    assert command == (
        "/opt/conda/envs/behavior/bin/python", "-m", "omnigibson.eval.eval",
        "--task-name", "turning_on_radio",
        "--robot-config", "/behavior-src/OmniGibson/omnigibson/eval/r1pro.yaml",
        "--mode", "public_test",
        "--host", "127.0.0.1",
        "--port", "8123",
        "--instance-indices", "3",
        "--num-rollouts", "1",
        "--output-dir", str(tmp_path),
        "--headless", "--write-video",
    )


@dataclass
class FakeProcess:
    pid: int
    returncode: int = 0
    timed_out: bool = False
    subprocess_timed_out: bool = False
    interrupted: bool = False
    terminated: bool = False
    exited: bool = False

    def wait(self, timeout: float) -> int:
        if self.subprocess_timed_out:
            raise subprocess.TimeoutExpired("official-evaluator", timeout)
        if self.timed_out:
            raise TimeoutError
        if self.interrupted:
            raise KeyboardInterrupt
        return self.returncode

    def poll(self) -> int | None:
        return self.returncode if self.exited else None


class FakeLauncher:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], dict[str, str], Path]] = []
        self.processes: list[FakeProcess] = []

    def start(self, command: tuple[str, ...], *, environment: dict[str, str], cwd: Path) -> FakeProcess:
        process = FakeProcess(pid=10_000 + len(self.processes))
        self.calls.append((command, environment, cwd))
        self.processes.append(process)
        return process

    def terminate_group(self, process: FakeProcess, *, timeout_seconds: float) -> None:
        process.terminated = True
        process.exited = True


class FileEvidenceLauncher(FakeLauncher):
    def start(self, command: tuple[str, ...], *, environment: dict[str, str], cwd: Path) -> FakeProcess:
        process = super().start(command, environment=environment, cwd=cwd)
        if command[:3] == ("/opt/conda/envs/behavior/bin/python", "-m", "omnigibson.eval.eval"):
            output = Path(command[command.index("--output-dir") + 1])
            target = output / "raw-evidence.json"
            target.write_text("{}", encoding="utf-8")
            (output / "evaluator.json").symlink_to(target)
        return process


class TerminalEvidenceLauncher(FakeLauncher):
    def start(self, command: tuple[str, ...], *, environment: dict[str, str], cwd: Path) -> FakeProcess:
        process = super().start(command, environment=environment, cwd=cwd)
        if command[:3] == ("/opt/conda/envs/behavior/bin/python", "-m", "omnigibson.eval.eval"):
            output = Path(command[command.index("--output-dir") + 1])
            artifact = b"controller-publisher-integration"
            (output / "videos").mkdir()
            (output / "videos" / "rollout.mp4").write_bytes(artifact)
            task = command[command.index("--task-name") + 1]
            index = int(command[command.index("--instance-indices") + 1])
            evidence = json.loads(
                (ROLLOUT / "tests" / "fixtures" / "official-evaluator-result.json").read_text()
            )
            evidence["task"] = task
            evidence["instance_id"] = 301 + index
            result_dir = output / "json"
            result_dir.mkdir()
            (result_dir / f"{task}_{301 + index}_0.json").write_text(json.dumps(evidence), encoding="utf-8")
        return process


class MismatchedTerminalEvidenceLauncher(TerminalEvidenceLauncher):
    def start(self, command: tuple[str, ...], *, environment: dict[str, str], cwd: Path) -> FakeProcess:
        process = super().start(command, environment=environment, cwd=cwd)
        if command[:3] == ("/opt/conda/envs/behavior/bin/python", "-m", "omnigibson.eval.eval"):
            output = Path(command[command.index("--output-dir") + 1])
            task = command[command.index("--task-name") + 1]
            evidence_path = output / "json" / f"{task}_301_0.json"
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence["instance_id"] = 302
            evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
        return process


class FailedTerminalEvidenceLauncher(TerminalEvidenceLauncher):
    def start(self, command: tuple[str, ...], *, environment: dict[str, str], cwd: Path) -> FakeProcess:
        process = super().start(command, environment=environment, cwd=cwd)
        if command[:3] == ("/opt/conda/envs/behavior/bin/python", "-m", "omnigibson.eval.eval"):
            output = Path(command[command.index("--output-dir") + 1])
            task = command[command.index("--task-name") + 1]
            evidence_path = output / "json" / f"{task}_301_0.json"
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence["success"] = False
            evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
        return process


class MemoryHub:
    """Tiny immutable Hub boundary used to exercise the real publisher."""

    def __init__(self) -> None:
        self.current: dict[str, bytes] = {}
        self.revisions: dict[str, dict[str, bytes]] = {}
        self._serial = 0
        self.head = self._commit()

    def get_dataset_info(self, _repo_id: str) -> Mapping[str, object]:
        return {"private": True, "sha": self.head}

    def list_tree(self, _repo_id: str, *, revision: str, prefix: str) -> Mapping[str, str]:
        return {
            path: hashlib.sha256(data).hexdigest()
            for path, data in self.revisions[revision].items()
            if path.startswith(prefix + "/")
        }

    def upload_tree(self, _repo_id: str, *, local_dir: Path, remote_prefix: str, commit_message: str) -> str:
        assert commit_message
        for path in local_dir.rglob("*"):
            if path.is_file():
                self.current[f"{remote_prefix}/{path.relative_to(local_dir).as_posix()}"] = path.read_bytes()
        return self._commit()

    def promote_prefix(self, _repo_id: str, *, staging_prefix: str, release_prefix: str, commit_message: str) -> str:
        assert commit_message
        for path, data in list(self.current.items()):
            if path.startswith(staging_prefix + "/"):
                self.current[f"{release_prefix}/{path.removeprefix(staging_prefix + '/')}"] = data
                del self.current[path]
        return self._commit()

    def delete_prefix(self, _repo_id: str, *, prefix: str) -> str:
        for path in list(self.current):
            if path.startswith(prefix + "/"):
                del self.current[path]
        return self._commit()

    def download_file_to_path(self, _repo_id: str, *, revision: str, path: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.revisions[revision][path])

    def _commit(self) -> str:
        self._serial += 1
        revision = f"{self._serial:040x}"
        self.revisions[revision] = dict(self.current)
        self.head = revision
        return revision


def test_controller_assigns_one_policy_and_one_evaluator_worker_per_gpu(
    tmp_path: Path,
) -> None:
    contract = _contract()
    checkpoint = CheckpointReceipt(contract.model_commit, contract.checkpoint_artifact_sha256, tmp_path / "checkpoint")
    launcher = FakeLauncher()
    controller = RolloutController(
        contract=contract,
        task_manifest=load_task_manifest(ROLLOUT / "task-manifest.json"),
        checkpoint_source=FakeCheckpointSource(checkpoint),
        checkpoint_dir=checkpoint.local_path,
        workspace=tmp_path / "campaign",
        gpu_ids=(2, 5),
        launcher=launcher,
        health_probe=lambda url, timeout: url.endswith("/healthz"),
    )

    assignments = controller.worker_assignments
    controller.start_policy_servers()

    assert [(item.worker_id, item.gpu_id, item.port) for item in assignments] == [
        ("gpu-2", 2, 8000), ("gpu-5", 5, 8001)
    ]
    assert [call[1]["CUDA_VISIBLE_DEVICES"] for call in launcher.calls] == ["2", "5"]
    assert all("--action-horizon" not in call[0] for call in launcher.calls)
    assert json.loads((tmp_path / "campaign" / "campaign-manifest.json").read_text())[
        "worker_assignments"
    ] == [
        {"gpu_id": 2, "port": 8000, "worker_id": "gpu-2"},
        {"gpu_id": 5, "port": 8001, "worker_id": "gpu-5"},
    ]
    controller.close()
    assert all(process.terminated for process in launcher.processes)


def test_controller_polls_policy_readiness_and_rejects_an_exited_launched_child(
    tmp_path: Path,
) -> None:
    contract = _contract()
    checkpoint = CheckpointReceipt(contract.model_commit, contract.checkpoint_artifact_sha256, tmp_path / "checkpoint")
    launcher = FakeLauncher()
    probe_calls: list[float] = []

    def eventually_ready(_url: str, timeout: float) -> bool:
        probe_calls.append(timeout)
        return len(probe_calls) == 3

    controller = RolloutController(
        contract=contract,
        task_manifest=load_task_manifest(ROLLOUT / "task-manifest.json"),
        checkpoint_source=FakeCheckpointSource(checkpoint),
        checkpoint_dir=checkpoint.local_path,
        workspace=tmp_path / "campaign",
        gpu_ids=(0,),
        launcher=launcher,
        health_probe=eventually_ready,
        shutdown_timeout_seconds=1.0,
    )
    controller.start_policy_servers()
    assert len(probe_calls) == 3
    assert all(0 < timeout <= 1.0 for timeout in probe_calls)
    controller.close()

    exited = FakeProcess(pid=30_001, returncode=17, exited=True)
    launcher = FakeLauncher()
    launcher.start = lambda *_args, **_kwargs: exited  # type: ignore[method-assign]
    exited_checkpoint = CheckpointReceipt(
        contract.model_commit, contract.checkpoint_artifact_sha256, tmp_path / "checkpoint-exited"
    )
    controller = RolloutController(
        contract=contract,
        task_manifest=load_task_manifest(ROLLOUT / "task-manifest.json"),
        checkpoint_source=FakeCheckpointSource(exited_checkpoint),
        checkpoint_dir=exited_checkpoint.local_path,
        workspace=tmp_path / "campaign-exited",
        gpu_ids=(0,),
        launcher=launcher,
        health_probe=lambda _url, _timeout: True,
    )
    with pytest.raises(ControllerError, match="exited before readiness"):
        controller.start_policy_servers()


def test_controller_rechecks_launched_child_after_a_positive_health_response(tmp_path: Path) -> None:
    contract = _contract()
    checkpoint = CheckpointReceipt(contract.model_commit, contract.checkpoint_artifact_sha256, tmp_path / "checkpoint")
    launcher = FakeLauncher()
    process = FakeProcess(pid=30_002)
    launcher.start = lambda *_args, **_kwargs: process  # type: ignore[method-assign]

    def unrelated_port_responded(_url: str, _timeout: float) -> bool:
        process.exited = True
        return True

    controller = RolloutController(
        contract=contract,
        task_manifest=load_task_manifest(ROLLOUT / "task-manifest.json"),
        checkpoint_source=FakeCheckpointSource(checkpoint),
        checkpoint_dir=tmp_path / "checkpoint",
        workspace=tmp_path / "campaign",
        gpu_ids=(0,),
        launcher=launcher,
        health_probe=unrelated_port_responded,
    )
    with pytest.raises(ControllerError, match="exited before readiness"):
        controller.start_policy_servers()


def test_controller_rejects_an_occupied_port_before_launch_and_a_listener_race_after_health(
    tmp_path: Path,
) -> None:
    contract = _contract()
    checkpoint = CheckpointReceipt(contract.model_commit, contract.checkpoint_artifact_sha256, tmp_path / "checkpoint")
    launcher = FakeLauncher()
    occupied = RolloutController(
        contract=contract,
        task_manifest=load_task_manifest(ROLLOUT / "task-manifest.json"),
        checkpoint_source=FakeCheckpointSource(checkpoint),
        checkpoint_dir=checkpoint.local_path,
        workspace=tmp_path / "campaign-occupied",
        gpu_ids=(0,),
        launcher=launcher,
        health_probe=lambda _url, _timeout: True,
        port_is_available=lambda _host, _port: False,
    )
    with pytest.raises(ControllerError, match="already occupied"):
        occupied.start_policy_servers()
    assert launcher.calls == []

    launcher = FakeLauncher()
    raced = RolloutController(
        contract=contract,
        task_manifest=load_task_manifest(ROLLOUT / "task-manifest.json"),
        checkpoint_source=FakeCheckpointSource(checkpoint),
        checkpoint_dir=checkpoint.local_path,
        workspace=tmp_path / "campaign-raced",
        gpu_ids=(0,),
        launcher=launcher,
        health_probe=lambda _url, _timeout: True,
        port_is_available=lambda _host, _port: True,
        listener_pid=lambda _host, _port: os.getpid(),
    )
    with pytest.raises(ControllerError, match="not the launched process"):
        raced.start_policy_servers()
    assert launcher.processes[0].terminated is True

    launcher = FakeLauncher()
    owned = RolloutController(
        contract=contract,
        task_manifest=load_task_manifest(ROLLOUT / "task-manifest.json"),
        checkpoint_source=FakeCheckpointSource(checkpoint),
        checkpoint_dir=checkpoint.local_path,
        workspace=tmp_path / "campaign-owned",
        gpu_ids=(0,),
        launcher=launcher,
        health_probe=lambda _url, _timeout: True,
        port_is_available=lambda _host, _port: True,
        listener_pid=lambda _host, _port: 10_000,
    )
    owned.start_policy_servers()
    owned.close()


def test_controller_refuses_publication_until_every_canonical_episode_is_closed_or_quarantined(
    tmp_path: Path,
) -> None:
    contract = _contract()
    checkpoint = CheckpointReceipt(contract.model_commit, contract.checkpoint_artifact_sha256, tmp_path / "checkpoint")
    controller = RolloutController(
        contract=contract,
        task_manifest=load_task_manifest(ROLLOUT / "task-manifest.json"),
        checkpoint_source=FakeCheckpointSource(checkpoint),
        checkpoint_dir=checkpoint.local_path,
        workspace=tmp_path / "campaign",
        gpu_ids=(0,),
        launcher=FakeLauncher(),
        health_probe=lambda _url, _timeout: True,
    )
    called = False

    def publisher(**_: object) -> object:
        nonlocal called
        called = True
        return object()

    with pytest.raises(ControllerError, match="not terminal or quarantined"):
        controller.publish_if_complete(publisher)

    assert called is False


def test_controller_quarantines_a_timed_out_evaluator_and_terminates_its_group(
    tmp_path: Path,
) -> None:
    contract = _contract()
    checkpoint = CheckpointReceipt(contract.model_commit, contract.checkpoint_artifact_sha256, tmp_path / "checkpoint")
    launcher = FakeLauncher()
    timed_out = FakeProcess(pid=44, timed_out=True)
    launcher.start = lambda *_args, **_kwargs: timed_out  # type: ignore[method-assign]
    controller = RolloutController(
        contract=contract,
        task_manifest=load_task_manifest(ROLLOUT / "task-manifest.json"),
        checkpoint_source=FakeCheckpointSource(checkpoint),
        checkpoint_dir=checkpoint.local_path,
        workspace=tmp_path / "campaign",
        gpu_ids=(0,),
        launcher=launcher,
        health_probe=lambda _url, _timeout: True,
        provenance_authenticator=ProvenanceAuthenticator(b"k" * 32, issuer="test"),
    )

    controller._run_episode(controller.requests[0])

    envelope = controller._load_envelopes_if_present()[0]
    campaign = json.loads((controller.workspace / "campaign-manifest.json").read_text())
    assert timed_out.terminated is True
    assert envelope.outcome.value == "quarantine"
    assert envelope.provenance["origin"] == "file"
    assert envelope.provenance_attestation is not None
    assert campaign["accepted_attempts"][envelope.episode_key] == {
        "attempt_root": "artifacts/b100-000-public-00/attempts/attempt-000001",
        "envelope_sha256": envelope.canonical_sha256,
    }


def test_controller_treats_subprocess_timeout_expired_as_a_timeout(tmp_path: Path) -> None:
    contract = _contract()
    checkpoint = CheckpointReceipt(contract.model_commit, contract.checkpoint_artifact_sha256, tmp_path / "checkpoint")
    process = FakeProcess(pid=48, subprocess_timed_out=True)
    launcher = FakeLauncher()
    launcher.start = lambda *_args, **_kwargs: process  # type: ignore[method-assign]
    controller = RolloutController(
        contract=contract,
        task_manifest=load_task_manifest(ROLLOUT / "task-manifest.json"),
        checkpoint_source=FakeCheckpointSource(checkpoint),
        checkpoint_dir=checkpoint.local_path,
        workspace=tmp_path / "campaign",
        gpu_ids=(0,),
        launcher=launcher,
        health_probe=lambda _url, _timeout: True,
        provenance_authenticator=ProvenanceAuthenticator(b"t" * 32, issuer="test"),
    )

    controller._run_episode(controller.requests[0])

    envelope = controller._load_envelopes_if_present()[0]
    assert process.terminated is True
    assert envelope.outcome.value == "quarantine"
    assert b'"status":"timeout"' in envelope.raw_evidence


def test_controller_rejects_a_preexisting_campaign_manifest_for_a_different_contract(
    tmp_path: Path,
) -> None:
    contract = _contract()
    checkpoint = CheckpointReceipt(contract.model_commit, contract.checkpoint_artifact_sha256, tmp_path / "checkpoint")
    workspace = tmp_path / "campaign"
    workspace.mkdir()
    (workspace / "campaign-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "contract_identity": "0" * 64,
                "task_manifest_sha256": contract.task_manifest_sha256,
                "worker_assignments": [{"worker_id": "gpu-0", "gpu_id": 0, "port": 8000}],
                "requested_episode_count": 1000,
                "accepted_attempts": {},
            }
        ),
        encoding="utf-8",
    )
    source = FakeCheckpointSource(checkpoint)
    launcher = FakeLauncher()
    controller = RolloutController(
        contract=contract,
        task_manifest=load_task_manifest(ROLLOUT / "task-manifest.json"),
        checkpoint_source=source,
        checkpoint_dir=checkpoint.local_path,
        workspace=workspace,
        gpu_ids=(0,),
        launcher=launcher,
        health_probe=lambda _url, _timeout: True,
    )

    with pytest.raises(ControllerError, match="campaign manifest"):
        controller.start_policy_servers()

    assert source.calls == []
    assert launcher.calls == []


def test_controller_terminates_a_normally_completed_evaluator_process_group(tmp_path: Path) -> None:
    contract = _contract()
    checkpoint = CheckpointReceipt(contract.model_commit, contract.checkpoint_artifact_sha256, tmp_path / "checkpoint")
    launcher = FakeLauncher()
    completed = FakeProcess(pid=46)
    launcher.start = lambda *_args, **_kwargs: completed  # type: ignore[method-assign]
    controller = RolloutController(
        contract=contract,
        task_manifest=load_task_manifest(ROLLOUT / "task-manifest.json"),
        checkpoint_source=FakeCheckpointSource(checkpoint),
        checkpoint_dir=checkpoint.local_path,
        workspace=tmp_path / "campaign",
        gpu_ids=(0,),
        launcher=launcher,
        health_probe=lambda _url, _timeout: True,
        provenance_authenticator=ProvenanceAuthenticator(b"n" * 32, issuer="test"),
    )

    controller._run_episode(controller.requests[0])

    envelope = controller._load_envelopes_if_present()[0]
    assert completed.terminated is True
    assert envelope.provenance["origin"] == "file"
    assert envelope.provenance_attestation is not None


def test_controller_contract_binds_a_crashed_evaluator_quarantine(tmp_path: Path) -> None:
    contract = _contract()
    checkpoint = CheckpointReceipt(contract.model_commit, contract.checkpoint_artifact_sha256, tmp_path / "checkpoint")
    launcher = FakeLauncher()
    crashed = FakeProcess(pid=47, returncode=1)
    launcher.start = lambda *_args, **_kwargs: crashed  # type: ignore[method-assign]
    controller = RolloutController(
        contract=contract,
        task_manifest=load_task_manifest(ROLLOUT / "task-manifest.json"),
        checkpoint_source=FakeCheckpointSource(checkpoint),
        checkpoint_dir=checkpoint.local_path,
        workspace=tmp_path / "campaign",
        gpu_ids=(0,),
        launcher=launcher,
        health_probe=lambda _url, _timeout: True,
        provenance_authenticator=ProvenanceAuthenticator(b"c" * 32, issuer="test"),
    )

    controller._run_episode(controller.requests[0])

    envelope = controller._load_envelopes_if_present()[0]
    assert crashed.terminated is True
    assert envelope.provenance["origin"] == "file"
    assert envelope.provenance_attestation is not None


def test_controller_surfaces_policy_process_group_cleanup_failure(tmp_path: Path) -> None:
    contract = _contract()
    checkpoint = CheckpointReceipt(contract.model_commit, contract.checkpoint_artifact_sha256, tmp_path / "checkpoint")
    launcher = FakeLauncher()
    controller = RolloutController(
        contract=contract,
        task_manifest=load_task_manifest(ROLLOUT / "task-manifest.json"),
        checkpoint_source=FakeCheckpointSource(checkpoint),
        checkpoint_dir=checkpoint.local_path,
        workspace=tmp_path / "campaign",
        gpu_ids=(0,),
        launcher=launcher,
        health_probe=lambda _url, _timeout: True,
    )
    controller.start_policy_servers()
    launcher.terminate_group = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("cleanup"))  # type: ignore[method-assign]

    with pytest.raises(ControllerError, match="policy process group cleanup"):
        controller.close()


def test_controller_does_not_publish_when_policy_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract()
    checkpoint = CheckpointReceipt(contract.model_commit, contract.checkpoint_artifact_sha256, tmp_path / "checkpoint")
    launcher = FakeLauncher()
    controller = RolloutController(
        contract=contract,
        task_manifest=load_task_manifest(ROLLOUT / "task-manifest.json"),
        checkpoint_source=FakeCheckpointSource(checkpoint),
        checkpoint_dir=checkpoint.local_path,
        workspace=tmp_path / "campaign",
        gpu_ids=(0,),
        launcher=launcher,
        health_probe=lambda _url, _timeout: True,
        provenance_authenticator=ProvenanceAuthenticator(b"q" * 32, issuer="test"),
    )
    request = controller.requests[0]
    monkeypatch.setattr(RolloutController, "requests", property(lambda _self: (request,)))
    original_terminate = launcher.terminate_group

    def fail_only_policy(process: FakeProcess, *, timeout_seconds: float) -> None:
        if process.pid == 10_000:
            raise RuntimeError("policy cleanup")
        original_terminate(process, timeout_seconds=timeout_seconds)

    launcher.terminate_group = fail_only_policy  # type: ignore[method-assign]
    called = False

    def publisher(**_: object) -> None:
        nonlocal called
        called = True

    with pytest.raises(ControllerError, match="policy process group cleanup"):
        controller.run(publish=publisher)

    assert called is False


def test_controller_persists_and_reloads_a_contract_bound_interrupt_quarantine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract()
    checkpoint = CheckpointReceipt(contract.model_commit, contract.checkpoint_artifact_sha256, tmp_path / "checkpoint")
    launcher = FakeLauncher()
    interrupted = FakeProcess(pid=45, interrupted=True)
    launcher.start = lambda *_args, **_kwargs: interrupted  # type: ignore[method-assign]
    controller = RolloutController(
        contract=contract,
        task_manifest=load_task_manifest(ROLLOUT / "task-manifest.json"),
        checkpoint_source=FakeCheckpointSource(checkpoint),
        checkpoint_dir=checkpoint.local_path,
        workspace=tmp_path / "campaign",
        gpu_ids=(0,),
        launcher=launcher,
        health_probe=lambda _url, _timeout: True,
        provenance_authenticator=ProvenanceAuthenticator(b"i" * 32, issuer="test"),
    )
    persisted_contracts: list[RolloutContract | None] = []
    original_write = write_episode_envelope

    def record_write(*args: object, **kwargs: object) -> Path:
        persisted_contracts.append(kwargs.get("contract"))
        return original_write(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("b1k_rollout.controller.write_episode_envelope", record_write)

    with pytest.raises(KeyboardInterrupt):
        controller._run_episode(controller.requests[0])

    assert interrupted.terminated is True
    assert persisted_contracts == [contract]
    assert controller._load_envelopes_if_present()[0].outcome.value == "quarantine"


def test_controller_retries_in_a_new_attempt_directory_without_touching_old_evidence(
    tmp_path: Path,
) -> None:
    contract = _contract()
    checkpoint = CheckpointReceipt(contract.model_commit, contract.checkpoint_artifact_sha256, tmp_path / "checkpoint")
    launcher = FakeLauncher()
    controller = RolloutController(
        contract=contract,
        task_manifest=load_task_manifest(ROLLOUT / "task-manifest.json"),
        checkpoint_source=FakeCheckpointSource(checkpoint),
        checkpoint_dir=checkpoint.local_path,
        workspace=tmp_path / "campaign",
        gpu_ids=(0,),
        launcher=launcher,
        health_probe=lambda _url, _timeout: True,
        provenance_authenticator=ProvenanceAuthenticator(b"a" * 32, issuer="test"),
    )
    request = controller.requests[0]
    old_attempt = controller.artifact_root / request.episode_key / "attempts" / "attempt-000001"
    old_attempt.mkdir(parents=True)
    old_evidence = old_attempt / "evaluator.json"
    old_evidence.write_text("old evidence", encoding="utf-8")

    controller._run_episode(request)

    evaluator_command = launcher.calls[0][0]
    output_dir = Path(evaluator_command[evaluator_command.index("--output-dir") + 1])
    assert output_dir.name == "attempt-000002"
    assert old_evidence.read_text(encoding="utf-8") == "old evidence"
    assert controller._load_envelopes_if_present()[0].outcome.value == "quarantine"


def test_controller_carries_file_observation_attestation_into_its_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract()
    checkpoint = CheckpointReceipt(contract.model_commit, contract.checkpoint_artifact_sha256, tmp_path / "checkpoint")
    controller = RolloutController(
        contract=contract,
        task_manifest=load_task_manifest(ROLLOUT / "task-manifest.json"),
        checkpoint_source=FakeCheckpointSource(checkpoint),
        checkpoint_dir=checkpoint.local_path,
        workspace=tmp_path / "campaign",
        gpu_ids=(0,),
        launcher=FileEvidenceLauncher(),
        health_probe=lambda _url, _timeout: True,
        provenance_authenticator=ProvenanceAuthenticator(b"f" * 32, issuer="test"),
    )

    request = controller.requests[0]
    monkeypatch.setattr(RolloutController, "requests", property(lambda _self: (request,)))
    controller._run_episode(request)

    envelope = controller._load_envelopes_if_present()[0]
    assert envelope.outcome.value == "quarantine"
    assert envelope.provenance["origin"] == "file"
    assert envelope.provenance_attestation is not None
    # A symlinked evaluator record is retained as authenticated quarantine state,
    # never exposed as a publisher artifact root.
    assert controller._accepted_attempt_roots((envelope,)) == {}
    published = controller.publish_if_complete(lambda **kwargs: kwargs)
    assert published["artifact_roots"] == {}


def test_controller_binds_the_exact_attempt_root_for_real_immutable_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract()
    checkpoint = CheckpointReceipt(contract.model_commit, contract.checkpoint_artifact_sha256, tmp_path / "checkpoint")
    authenticator = ProvenanceAuthenticator(b"p" * 32, issuer="test")
    controller = RolloutController(
        contract=contract,
        task_manifest=load_task_manifest(ROLLOUT / "task-manifest.json"),
        checkpoint_source=FakeCheckpointSource(checkpoint),
        checkpoint_dir=checkpoint.local_path,
        workspace=tmp_path / "campaign",
        gpu_ids=(0,),
        launcher=TerminalEvidenceLauncher(),
        health_probe=lambda _url, _timeout: True,
        provenance_authenticator=authenticator,
    )
    request = controller.requests[0]
    monkeypatch.setattr(RolloutController, "requests", property(lambda _self: (request,)))

    controller._run_episode(request)

    envelope = controller._load_envelopes_if_present()[0]
    hub = MemoryHub()
    result = controller.publish_if_complete(
        lambda **kwargs: publish_release(hub=hub, **kwargs)
    )

    assert controller._accepted_attempt_roots((envelope,)) == {
        request.episode_key: controller.artifact_root / request.episode_key / "attempts" / "attempt-000001"
    }
    assert result.release_manifest["counts"] == {"success": 1, "failure": 0, "quarantine": 0}


def test_controller_quarantines_terminal_evidence_for_the_wrong_requested_instance(
    tmp_path: Path,
) -> None:
    contract = _contract()
    checkpoint = CheckpointReceipt(contract.model_commit, contract.checkpoint_artifact_sha256, tmp_path / "checkpoint")
    controller = RolloutController(
        contract=contract,
        task_manifest=load_task_manifest(ROLLOUT / "task-manifest.json"),
        checkpoint_source=FakeCheckpointSource(checkpoint),
        checkpoint_dir=checkpoint.local_path,
        workspace=tmp_path / "campaign",
        gpu_ids=(0,),
        launcher=MismatchedTerminalEvidenceLauncher(),
        health_probe=lambda _url, _timeout: True,
        provenance_authenticator=ProvenanceAuthenticator(b"m" * 32, issuer="test"),
    )

    controller._run_episode(controller.requests[0])

    envelope = controller._load_envelopes_if_present()[0]
    assert envelope.outcome.value == "quarantine"
    assert b'official_evaluator_output' in envelope.raw_evidence


def test_controller_normalizes_a_pinned_official_failure_result(tmp_path: Path) -> None:
    contract = _contract()
    checkpoint = CheckpointReceipt(contract.model_commit, contract.checkpoint_artifact_sha256, tmp_path / "checkpoint")
    controller = RolloutController(
        contract=contract,
        task_manifest=load_task_manifest(ROLLOUT / "task-manifest.json"),
        checkpoint_source=FakeCheckpointSource(checkpoint),
        checkpoint_dir=checkpoint.local_path,
        workspace=tmp_path / "campaign",
        gpu_ids=(0,),
        launcher=FailedTerminalEvidenceLauncher(),
        health_probe=lambda _url, _timeout: True,
        provenance_authenticator=ProvenanceAuthenticator(b"o" * 32, issuer="test"),
    )

    controller._run_episode(controller.requests[0])

    envelope = controller._load_envelopes_if_present()[0]
    assert envelope.outcome.value == "failure"
    assert envelope.provenance["reason_code"] == "official_evaluator_v1"


def test_controller_requires_an_out_of_band_authenticator_before_evaluator_file_work(
    tmp_path: Path,
) -> None:
    contract = _contract()
    checkpoint = CheckpointReceipt(contract.model_commit, contract.checkpoint_artifact_sha256, tmp_path / "checkpoint")
    controller = RolloutController(
        contract=contract,
        task_manifest=load_task_manifest(ROLLOUT / "task-manifest.json"),
        checkpoint_source=FakeCheckpointSource(checkpoint),
        checkpoint_dir=checkpoint.local_path,
        workspace=tmp_path / "campaign",
        gpu_ids=(0,),
        launcher=FakeLauncher(),
        health_probe=lambda _url, _timeout: True,
    )

    with pytest.raises(ControllerError, match="provenance authenticator"):
        controller._run_episode(controller.requests[0])


def test_controller_resumes_only_the_exact_terminal_or_quarantined_campaign(
    tmp_path: Path,
) -> None:
    contract = _contract()
    checkpoint = CheckpointReceipt(contract.model_commit, contract.checkpoint_artifact_sha256, tmp_path / "checkpoint")
    launcher = FakeLauncher()
    authenticator = ProvenanceAuthenticator(b"r" * 32, issuer="test")
    controller = RolloutController(
        contract=contract,
        task_manifest=load_task_manifest(ROLLOUT / "task-manifest.json"),
        checkpoint_source=FakeCheckpointSource(checkpoint),
        checkpoint_dir=checkpoint.local_path,
        workspace=tmp_path / "campaign",
        gpu_ids=(0,),
        launcher=launcher,
        health_probe=lambda _url, _timeout: True,
        provenance_authenticator=authenticator,
    )
    classified = classify_outcome(
        {"status": "interrupted", "completed": False}, task_manifest=controller.task_manifest
    )
    controller.workspace.mkdir()
    for request in controller.requests:
        write_episode_envelope(
            controller.envelope_root,
            request.episode_key,
            classified,
            contract=contract,
            authenticator=authenticator,
        )
    for envelope in controller._load_envelopes_if_present():
        attempt = (
            controller.artifact_root
            / envelope.episode_key
            / "attempts"
            / "attempt-000001"
        )
        attempt.mkdir(parents=True)
        controller._accepted_attempts[envelope.episode_key] = {
            "attempt_root": attempt.relative_to(controller.workspace).as_posix(),
            "envelope_sha256": envelope.canonical_sha256,
        }
    controller._write_campaign_manifest()

    received: list[ProvenanceAuthenticator] = []

    def publisher(
        *,
        episodes: object,
        artifact_roots: object,
        contract: RolloutContract,
        task_manifest: object,
        authenticator: ProvenanceAuthenticator,
    ) -> int:
        del artifact_roots, contract, task_manifest
        received.append(authenticator)
        return len(episodes)  # type: ignore[arg-type]

    published = controller.run(publish=publisher)

    assert published == 1000
    assert received == [authenticator]
    assert len(launcher.calls) == 1  # The resumed campaign starts its one policy server only.

    def publisher_without_authenticator(
        *, episodes: object, artifact_roots: object, contract: object, task_manifest: object
    ) -> int:
        del artifact_roots, contract, task_manifest
        return len(episodes)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="authenticator"):
        controller.publish_if_complete(publisher_without_authenticator)


class InjectedCrash(RuntimeError):
    pass


@pytest.mark.parametrize(
    "boundary",
    ("pending_attempt_durable", "envelope_durable", "accepted_binding_durable"),
)
def test_controller_recovers_every_attempt_persistence_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    """Each durable state is either recoverable or safely retried in a new attempt."""

    contract = _contract()
    checkpoint = CheckpointReceipt(contract.model_commit, contract.checkpoint_artifact_sha256, tmp_path / "checkpoint")
    authenticator = ProvenanceAuthenticator(b"z" * 32, issuer="test")
    request_holder: dict[str, object] = {}

    def make_controller(*, hook: object = None) -> RolloutController:
        controller = RolloutController(
            contract=contract,
            task_manifest=load_task_manifest(ROLLOUT / "task-manifest.json"),
            checkpoint_source=FakeCheckpointSource(checkpoint),
            checkpoint_dir=checkpoint.local_path,
            workspace=tmp_path / "campaign",
            gpu_ids=(0,),
            launcher=TerminalEvidenceLauncher(),
            health_probe=lambda _url, _timeout: True,
            provenance_authenticator=authenticator,
            persistence_hook=hook if callable(hook) else None,
        )
        if "request" not in request_holder:
            request_holder["request"] = controller.requests[0]
        monkeypatch.setattr(RolloutController, "requests", property(lambda _self: (request_holder["request"],)))
        return controller

    def crash_at(name: str) -> None:
        if name == boundary:
            raise InjectedCrash(name)

    initial = make_controller(hook=crash_at)
    request = request_holder["request"]
    assert isinstance(request, type(initial.requests[0]))
    with pytest.raises(InjectedCrash, match=boundary):
        initial._run_episode(request)

    first_attempt = initial.artifact_root / request.episode_key / "attempts" / "attempt-000001"
    recovered = make_controller()
    if boundary == "pending_attempt_durable":
        # A durable intent without an envelope is never promoted.  Even an attacker
        # changing old bytes only causes a fresh attempt to be used.
        (first_attempt / "json" / "turning_on_radio_301_0.json").write_text('{"status":"forged"}', encoding="utf-8")
        recovered.start_policy_servers()
        recovered._run_episode(request)
        assert (first_attempt / "json" / "turning_on_radio_301_0.json").read_text(encoding="utf-8") == '{"status":"forged"}'
        assert (
            recovered.artifact_root / request.episode_key / "attempts" / "attempt-000002"
        ).is_dir()
    else:
        recovered.start_policy_servers()

    result = recovered.publish_if_complete(lambda **kwargs: publish_release(hub=MemoryHub(), **kwargs))
    assert result.release_manifest["counts"] == {"success": 1, "failure": 0, "quarantine": 0}
    recovered.close()


def test_controller_rejects_a_tampered_pending_attempt_with_a_committed_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract()
    checkpoint = CheckpointReceipt(contract.model_commit, contract.checkpoint_artifact_sha256, tmp_path / "checkpoint")
    authenticator = ProvenanceAuthenticator(b"y" * 32, issuer="test")

    def crash_after_envelope(name: str) -> None:
        if name == "envelope_durable":
            raise InjectedCrash(name)

    controller = RolloutController(
        contract=contract,
        task_manifest=load_task_manifest(ROLLOUT / "task-manifest.json"),
        checkpoint_source=FakeCheckpointSource(checkpoint),
        checkpoint_dir=checkpoint.local_path,
        workspace=tmp_path / "campaign",
        gpu_ids=(0,),
        launcher=TerminalEvidenceLauncher(),
        health_probe=lambda _url, _timeout: True,
        provenance_authenticator=authenticator,
        persistence_hook=crash_after_envelope,
    )
    request = controller.requests[0]
    monkeypatch.setattr(RolloutController, "requests", property(lambda _self: (request,)))
    with pytest.raises(InjectedCrash):
        controller._run_episode(request)
    evidence = controller.artifact_root / request.episode_key / "attempts" / "attempt-000001" / "json" / "turning_on_radio_301_0.json"
    evidence.write_text('{"status":"forged"}', encoding="utf-8")

    resumed = RolloutController(
        contract=contract,
        task_manifest=load_task_manifest(ROLLOUT / "task-manifest.json"),
        checkpoint_source=FakeCheckpointSource(checkpoint),
        checkpoint_dir=checkpoint.local_path,
        workspace=tmp_path / "campaign",
        gpu_ids=(0,),
        launcher=TerminalEvidenceLauncher(),
        health_probe=lambda _url, _timeout: True,
        provenance_authenticator=authenticator,
    )
    monkeypatch.setattr(RolloutController, "requests", property(lambda _self: (request,)))
    with pytest.raises(ControllerError, match="does not match"):
        resumed.start_policy_servers()


class InterruptingLauncher(FakeLauncher):
    def __init__(self) -> None:
        super().__init__()
        self.evaluator = FakeProcess(pid=12_001)

    def terminate_group(self, process: FakeProcess, *, timeout_seconds: float) -> None:
        super().terminate_group(process, timeout_seconds=timeout_seconds)
        if process is self.evaluator:
            self.evaluator.terminated = True


def test_controller_main_interrupt_cancels_active_groups_without_starting_more_episodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract()
    checkpoint = CheckpointReceipt(contract.model_commit, contract.checkpoint_artifact_sha256, tmp_path / "checkpoint")
    launcher = InterruptingLauncher()
    controller = RolloutController(
        contract=contract,
        task_manifest=load_task_manifest(ROLLOUT / "task-manifest.json"),
        checkpoint_source=FakeCheckpointSource(checkpoint),
        checkpoint_dir=checkpoint.local_path,
        workspace=tmp_path / "campaign",
        gpu_ids=(0,),
        launcher=launcher,
        health_probe=lambda _url, _timeout: True,
        provenance_authenticator=ProvenanceAuthenticator(b"x" * 32, issuer="test"),
    )
    first = controller.requests[0]
    second = replace(first, episode_key="b100-000-public-01", instance_index=1)
    monkeypatch.setattr(RolloutController, "requests", property(lambda _self: (first, second)))

    def interrupted_worker(_requests: object) -> None:
        # This exception is delivered through Future.result() on the orchestration
        # thread, exercising the same main-thread cancellation path as Ctrl-C.
        controller._register_active_evaluator(launcher.evaluator)
        raise KeyboardInterrupt

    monkeypatch.setattr(controller, "_run_worker", interrupted_worker)
    with pytest.raises(KeyboardInterrupt):
        controller.run()

    assert controller._cancel_event.is_set()
    assert launcher.evaluator.terminated is True
    assert len(launcher.calls) == 1  # one policy server; no evaluator command starts after cancellation


def test_controller_worker_failure_cancels_and_terminates_an_active_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract()
    checkpoint = CheckpointReceipt(contract.model_commit, contract.checkpoint_artifact_sha256, tmp_path / "checkpoint")
    launcher = InterruptingLauncher()
    controller = RolloutController(
        contract=contract,
        task_manifest=load_task_manifest(ROLLOUT / "task-manifest.json"),
        checkpoint_source=FakeCheckpointSource(checkpoint),
        checkpoint_dir=checkpoint.local_path,
        workspace=tmp_path / "campaign",
        gpu_ids=(0, 1),
        launcher=launcher,
        health_probe=lambda _url, _timeout: True,
        provenance_authenticator=ProvenanceAuthenticator(b"w" * 32, issuer="test"),
    )
    first = controller.requests[0]
    second = replace(
        first,
        episode_key="b100-000-public-01",
        instance_index=1,
        worker_id="gpu-1",
        gpu_id=1,
        port=8001,
    )
    monkeypatch.setattr(RolloutController, "requests", property(lambda _self: (first, second)))
    sibling_registered = threading.Event()
    sibling_released = threading.Event()
    original_terminate = launcher.terminate_group

    def terminate(process: FakeProcess, *, timeout_seconds: float) -> None:
        original_terminate(process, timeout_seconds=timeout_seconds)
        if process is launcher.evaluator:
            sibling_released.set()

    launcher.terminate_group = terminate  # type: ignore[method-assign]

    def worker(requests: object) -> None:
        request = tuple(requests)[0]  # type: ignore[arg-type]
        if request.worker_id == "gpu-0":
            assert sibling_registered.wait(1.0)
            raise RuntimeError("worker boom")
        controller._register_active_evaluator(launcher.evaluator)
        sibling_registered.set()
        assert sibling_released.wait(1.0)
        controller._unregister_active_evaluator(launcher.evaluator)

    monkeypatch.setattr(controller, "_run_worker", worker)
    with pytest.raises(RuntimeError, match="worker boom"):
        controller.run()

    assert controller._cancel_event.is_set()
    assert launcher.evaluator.terminated is True


def test_controller_observes_a_later_worker_failure_without_waiting_for_a_blocked_first_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract()
    checkpoint = CheckpointReceipt(contract.model_commit, contract.checkpoint_artifact_sha256, tmp_path / "checkpoint")
    launcher = InterruptingLauncher()
    controller = RolloutController(
        contract=contract,
        task_manifest=load_task_manifest(ROLLOUT / "task-manifest.json"),
        checkpoint_source=FakeCheckpointSource(checkpoint),
        checkpoint_dir=checkpoint.local_path,
        workspace=tmp_path / "campaign",
        gpu_ids=(0, 1),
        launcher=launcher,
        health_probe=lambda _url, _timeout: True,
        provenance_authenticator=ProvenanceAuthenticator(b"l" * 32, issuer="test"),
    )
    first = controller.requests[0]
    second = replace(
        first,
        episode_key="b100-000-public-01",
        instance_index=1,
        worker_id="gpu-1",
        gpu_id=1,
        port=8001,
    )
    monkeypatch.setattr(RolloutController, "requests", property(lambda _self: (first, second)))
    first_started = threading.Event()
    first_released = threading.Event()
    original_terminate = launcher.terminate_group

    def terminate(process: FakeProcess, *, timeout_seconds: float) -> None:
        original_terminate(process, timeout_seconds=timeout_seconds)
        if process is launcher.evaluator:
            first_released.set()

    launcher.terminate_group = terminate  # type: ignore[method-assign]

    def worker(requests: object) -> None:
        request = tuple(requests)[0]  # type: ignore[arg-type]
        if request.worker_id == "gpu-0":
            controller._register_active_evaluator(launcher.evaluator)
            first_started.set()
            assert first_released.wait(3.0)
            controller._unregister_active_evaluator(launcher.evaluator)
            return
        assert first_started.wait(1.0)
        raise RuntimeError("gpu-1 failed")

    monkeypatch.setattr(controller, "_run_worker", worker)
    started_at = time.monotonic()
    with pytest.raises(RuntimeError, match="gpu-1 failed"):
        controller.run()

    assert time.monotonic() - started_at < 1.0
    assert controller._cancel_event.is_set()
    assert launcher.evaluator.terminated is True


def test_controller_retains_failed_policy_cleanup_handles_for_a_bounded_retry(tmp_path: Path) -> None:
    contract = _contract()
    checkpoint = CheckpointReceipt(contract.model_commit, contract.checkpoint_artifact_sha256, tmp_path / "checkpoint")
    launcher = FakeLauncher()
    controller = RolloutController(
        contract=contract,
        task_manifest=load_task_manifest(ROLLOUT / "task-manifest.json"),
        checkpoint_source=FakeCheckpointSource(checkpoint),
        checkpoint_dir=checkpoint.local_path,
        workspace=tmp_path / "campaign",
        gpu_ids=(0,),
        launcher=launcher,
        health_probe=lambda _url, _timeout: True,
    )
    controller.start_policy_servers()
    process = launcher.processes[0]
    original_terminate = launcher.terminate_group
    attempts = 0

    def fail_once(target: FakeProcess, *, timeout_seconds: float) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary cleanup failure")
        original_terminate(target, timeout_seconds=timeout_seconds)

    launcher.terminate_group = fail_once  # type: ignore[method-assign]
    with pytest.raises(ControllerError, match="policy process group cleanup"):
        controller.close()
    assert controller._policy_processes == {"gpu-0": process}

    controller.close()
    assert controller._policy_processes == {}
    assert process.terminated is True


def test_controller_retains_failed_evaluator_cleanup_and_surfaces_worker_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract()
    checkpoint = CheckpointReceipt(contract.model_commit, contract.checkpoint_artifact_sha256, tmp_path / "checkpoint")
    launcher = InterruptingLauncher()
    controller = RolloutController(
        contract=contract,
        task_manifest=load_task_manifest(ROLLOUT / "task-manifest.json"),
        checkpoint_source=FakeCheckpointSource(checkpoint),
        checkpoint_dir=checkpoint.local_path,
        workspace=tmp_path / "campaign",
        gpu_ids=(0,),
        launcher=launcher,
        health_probe=lambda _url, _timeout: True,
        provenance_authenticator=ProvenanceAuthenticator(b"v" * 32, issuer="test"),
    )
    request = controller.requests[0]
    monkeypatch.setattr(RolloutController, "requests", property(lambda _self: (request,)))
    original_terminate = launcher.terminate_group

    def always_fail(target: FakeProcess, *, timeout_seconds: float) -> None:
        if target is launcher.evaluator:
            raise RuntimeError("evaluator remains alive")
        original_terminate(target, timeout_seconds=timeout_seconds)

    launcher.terminate_group = always_fail  # type: ignore[method-assign]

    def failed_worker(_requests: object) -> None:
        controller._register_active_evaluator(launcher.evaluator)
        raise RuntimeError("worker boom")

    monkeypatch.setattr(controller, "_run_worker", failed_worker)
    with pytest.raises(ControllerError, match="active evaluator process group cleanup"):
        controller.run()
    assert id(launcher.evaluator) in controller._active_evaluators

    launcher.terminate_group = original_terminate  # type: ignore[method-assign]
    controller._terminate_active_evaluators()
    assert controller._active_evaluators == {}
