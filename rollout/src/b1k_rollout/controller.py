"""Bounded headless orchestration for the canonical B1K rollout campaign.

The controller deliberately owns process lifetime and local campaign state, but
does not interpret R1Pro actions or evaluator success.  Those remain entirely
inside the pinned GR00T server and the official OmniGibson evaluator.
"""

from __future__ import annotations

import json
import os
import hashlib
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol

from b1k_rollout.contracts import RolloutContract
from b1k_rollout.episodes import EpisodeEnvelope, load_episode_envelopes, write_episode_envelope
from b1k_rollout.identity import MODEL_REPO, reject_credential_material, require_sha256
from b1k_rollout.outcomes import (
    ClassifiedOutcome,
    Outcome,
    classify_outcome,
    classify_outcome_file,
    raw_evidence_sha256,
)
from b1k_rollout.provenance import (
    ProvenanceAuthenticationError,
    ProvenanceAuthenticator,
    canonical_attestation_payload,
    load_local_provenance_key,
)
from b1k_rollout.task_manifest import validate_task_manifest


_R1PRO_CONFIG = "/behavior-src/OmniGibson/omnigibson/eval/r1pro.yaml"
_BEHAVIOR_PYTHON = "/opt/conda/envs/behavior/bin/python"
_POLICY_HOST = "127.0.0.1"
_FIRST_POLICY_PORT = 8000
_RUNTIME_ENVIRONMENT = frozenset(
    (
        "PATH",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "LANG",
        "LC_ALL",
        "CUDA_HOME",
        "GROOT_PYTHON",
        "HEADLESS",
        "MPLCONFIGDIR",
        "NVIDIA_DRIVER_CAPABILITIES",
        "NUMBA_CACHE_DIR",
        "OMNI_KIT_ACCEPT_EULA",
        "OMNIGIBSON_APPDATA_PATH",
        "OMNIGIBSON_DATA_PATH",
        "TRITON_CACHE_DIR",
        "VK_DRIVER_FILES",
    )
)
_LAUNCH_ENVIRONMENT = frozenset(("CUDA_VISIBLE_DEVICES",))


class ControllerError(RuntimeError):
    """The campaign cannot safely make progress or be published."""


@dataclass(frozen=True, slots=True)
class CheckpointReceipt:
    """The independently verified identity returned by each checkpoint transfer."""

    model_commit: str
    artifact_sha256: str
    local_path: Path
    local_tree_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.local_path, Path):
            raise ValueError("checkpoint local path is invalid")
        if not self.local_path.is_absolute():
            raise ValueError("checkpoint local path must be absolute")
        if not isinstance(self.model_commit, str) or len(self.model_commit) != 40:
            raise ValueError("checkpoint model commit is invalid")
        require_sha256(self.artifact_sha256, label="checkpoint artifact")
        if self.local_tree_sha256 is not None:
            require_sha256(self.local_tree_sha256, label="checkpoint tree")


class CheckpointSource(Protocol):
    """Download and fresh readback boundary for a selected model-repo commit."""

    def download(
        self, *, repository: str, revision: str, destination: Path
    ) -> CheckpointReceipt: ...

    def readback(
        self, *, repository: str, revision: str, destination: Path
    ) -> CheckpointReceipt: ...


class ManagedProcess(Protocol):
    pid: int

    def wait(self, timeout: float) -> int: ...


class ProcessLauncher(Protocol):
    def start(
        self, command: tuple[str, ...], *, environment: dict[str, str], cwd: Path
    ) -> ManagedProcess: ...

    def terminate_group(self, process: ManagedProcess, *, timeout_seconds: float) -> None: ...


class FilesystemCheckpointSource:
    """Concrete immutable-directory transport for an already materialized model revision."""

    def __init__(self, revision_roots: Mapping[str, Path]) -> None:
        self._revision_roots = {revision: Path(root).resolve() for revision, root in revision_roots.items()}

    def download(
        self, *, repository: str, revision: str, destination: Path
    ) -> CheckpointReceipt:
        source = self._source(repository=repository, revision=revision)
        materialized_source, manifest_sha256 = _manifest_checkpoint_source(source)
        source_sha256 = checkpoint_tree_sha256(materialized_source)
        destination = Path(destination)
        if destination.exists():
            if destination.is_symlink() or not destination.is_dir():
                raise ControllerError("checkpoint destination is invalid")
            existing_sha256 = checkpoint_tree_sha256(destination)
            if existing_sha256 != source_sha256:
                raise ControllerError("existing checkpoint destination does not match immutable source")
            return CheckpointReceipt(
                revision, manifest_sha256 or existing_sha256, destination.resolve(), existing_sha256
            )
        try:
            shutil.copytree(materialized_source, destination, symlinks=False, copy_function=shutil.copy2)
        except OSError as error:
            raise ControllerError("immutable checkpoint download failed") from error
        destination_sha256 = checkpoint_tree_sha256(destination)
        if destination_sha256 != source_sha256:
            raise ControllerError("immutable checkpoint download hash mismatch")
        return CheckpointReceipt(
            revision, manifest_sha256 or destination_sha256, destination.resolve(), destination_sha256
        )

    def readback(
        self, *, repository: str, revision: str, destination: Path
    ) -> CheckpointReceipt:
        source = self._source(repository=repository, revision=revision)
        _, manifest_sha256 = _manifest_checkpoint_source(source)
        destination = Path(destination)
        if not destination.is_dir() or destination.is_symlink():
            raise ControllerError("checkpoint readback path is invalid")
        tree_sha256 = checkpoint_tree_sha256(destination)
        return CheckpointReceipt(revision, manifest_sha256 or tree_sha256, destination.resolve(), tree_sha256)

    def _source(self, *, repository: str, revision: str) -> Path:
        if repository != MODEL_REPO or revision not in self._revision_roots:
            raise ControllerError("immutable checkpoint source is unavailable")
        source = self._revision_roots[revision]
        if source.is_symlink() or not source.is_dir():
            raise ControllerError("immutable checkpoint source is invalid")
        return source


class SubprocessGroupLauncher:
    """A real local launcher that owns a new process group for every child."""

    def start(
        self, command: tuple[str, ...], *, environment: dict[str, str], cwd: Path
    ) -> subprocess.Popen[bytes]:
        safe_environment = {"PATH": os.environ.get("PATH", os.defpath)}
        for name in _RUNTIME_ENVIRONMENT - {"PATH"}:
            if name in os.environ:
                safe_environment[name] = os.environ[name]
        for name, value in environment.items():
            if name in _LAUNCH_ENVIRONMENT:
                safe_environment[name] = value
        try:
            return subprocess.Popen(
                command,
                cwd=cwd,
                env=safe_environment,
                start_new_session=True,
            )
        except OSError as error:
            raise ControllerError("unable to start controlled process group") from error

    def terminate_group(self, process: ManagedProcess, *, timeout_seconds: float) -> None:
        if timeout_seconds <= 0:
            raise ControllerError("process group timeout is invalid")
        pid = process.pid
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError as error:
            raise ControllerError("unable to terminate process group") from error
        deadline = time.monotonic() + timeout_seconds
        if self._wait_for_group_exit(process, pid, deadline):
            return
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except OSError as error:
            raise ControllerError("unable to kill process group") from error
        if not self._wait_for_group_exit(process, pid, time.monotonic() + timeout_seconds):
            raise ControllerError("process group did not terminate")

    @staticmethod
    def _group_exists(pid: int) -> bool:
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            # macOS can transiently return EPERM while a just-killed orphan is
            # being reaped.  Treat that as still present and keep the bounded poll;
            # accepting it as gone would leave a descendant unverified.
            return True
        except OSError as error:
            raise ControllerError("unable to inspect process group") from error
        return True

    def _wait_for_group_exit(self, process: ManagedProcess, pid: int, deadline: float) -> bool:
        """Wait for the group, rather than merely its leader, to be gone."""

        while self._group_exists(pid):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            try:
                process.wait(min(0.1, max(0.01, remaining)))
            except (subprocess.TimeoutExpired, TimeoutError):
                pass
            # A child can remain after wait() reaps the leader.  Poll killpg(0)
            # until the complete process group has disappeared.
            time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
        return True


def bounded_http_health_probe(url: str, timeout_seconds: float) -> bool:
    """Perform one bounded health probe without exposing response bodies in logs."""

    if not isinstance(url, str) or not url.startswith("http://") or timeout_seconds <= 0:
        raise ControllerError("health probe input is invalid")
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError, ValueError):
        return False


@dataclass(frozen=True, slots=True)
class WorkerAssignment:
    worker_id: str
    gpu_id: int
    port: int


@dataclass(frozen=True, slots=True)
class EpisodeRequest:
    episode_key: str
    task_name: str
    instance_index: int
    worker_id: str
    gpu_id: int
    port: int


def hydrate_immutable_checkpoint(
    source: CheckpointSource, *, contract: RolloutContract, destination: Path
) -> CheckpointReceipt:
    """Require download and independent readback to agree with the frozen contract."""

    destination = Path(destination).resolve()
    downloaded = source.download(
        repository=MODEL_REPO, revision=contract.model_commit, destination=destination
    )
    readback = source.readback(
        repository=MODEL_REPO, revision=contract.model_commit, destination=destination
    )
    for receipt in (downloaded, readback):
        _validate_checkpoint_receipt(receipt, contract=contract, destination=destination)
    if downloaded != readback:
        raise ControllerError("checkpoint readback does not match immutable download")
    return downloaded


def build_official_evaluator_command(
    *, task_name: str, mode: str, instance_index: int, port: int, output_dir: Path
) -> tuple[str, ...]:
    """Construct only the pinned official R1Pro evaluator invocation."""

    if mode not in ("train", "public_test"):
        raise ControllerError("official evaluator mode is invalid")
    if not isinstance(instance_index, int) or instance_index < 0:
        raise ControllerError("official evaluator instance index is invalid")
    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise ControllerError("policy server port is invalid")
    if not isinstance(task_name, str) or not task_name:
        raise ControllerError("official evaluator task is invalid")
    return (
        _BEHAVIOR_PYTHON, "-m", "omnigibson.eval.eval",
        "--task-name", task_name,
        "--robot-config", _R1PRO_CONFIG,
        "--mode", mode,
        "--host", _POLICY_HOST,
        "--port", str(port),
        "--instance-indices", str(instance_index),
        "--num-rollouts", "1",
        "--output-dir", str(Path(output_dir)),
        "--headless", "--write-video",
    )


class RolloutController:
    """Run the exact B100 public-test campaign with explicit GPU ownership."""

    def __init__(
        self,
        *,
        contract: RolloutContract,
        task_manifest: Mapping[str, object],
        checkpoint_source: CheckpointSource,
        checkpoint_dir: Path,
        workspace: Path,
        gpu_ids: Sequence[int],
        launcher: ProcessLauncher,
        health_probe: Callable[[str, float], bool],
        policy_command: Sequence[str] = ("python", "-m", "groot_policy_server"),
        episode_timeout_seconds: float = 900.0,
        shutdown_timeout_seconds: float = 20.0,
        provenance_authenticator: ProvenanceAuthenticator | None = None,
        provenance_key_path: Path | None = None,
        persistence_hook: Callable[[str], None] | None = None,
        port_is_available: Callable[[str, int], bool] | None = None,
        listener_pid: Callable[[str, int], int | None] | None = None,
    ) -> None:
        validate_task_manifest(task_manifest)
        if task_manifest["provenance"]["canonical_sha256"] != contract.task_manifest_sha256:  # type: ignore[index]
            raise ControllerError("task manifest does not match rollout contract")
        if not gpu_ids or any(type(gpu) is not int or gpu < 0 for gpu in gpu_ids):
            raise ControllerError("at least one explicit non-negative GPU id is required")
        if len(set(gpu_ids)) != len(gpu_ids):
            raise ControllerError("GPU ids must be unique")
        if episode_timeout_seconds <= 0 or shutdown_timeout_seconds <= 0:
            raise ControllerError("controller timeouts must be positive")
        if not policy_command or not all(isinstance(part, str) and part for part in policy_command):
            raise ControllerError("policy command is invalid")
        if provenance_authenticator is not None and not isinstance(
            provenance_authenticator, ProvenanceAuthenticator
        ):
            raise ControllerError("provenance authenticator is invalid")
        if provenance_key_path is not None:
            try:
                key = load_local_provenance_key(Path(provenance_key_path))
            except (OSError, ProvenanceAuthenticationError) as error:
                raise ControllerError("provenance key file is invalid") from error
            if provenance_authenticator is None:
                provenance_authenticator = ProvenanceAuthenticator(
                    key, issuer=f"b1k-{contract.campaign_id}"
                )
        self.contract = contract
        self.task_manifest = dict(task_manifest)
        self.checkpoint_source = checkpoint_source
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.workspace = Path(workspace).resolve()
        self.launcher = launcher
        self.health_probe = health_probe
        self.policy_command = tuple(policy_command)
        self.episode_timeout_seconds = episode_timeout_seconds
        self.shutdown_timeout_seconds = shutdown_timeout_seconds
        self.provenance_authenticator = provenance_authenticator
        self.provenance_key_path = Path(provenance_key_path) if provenance_key_path is not None else None
        self._persistence_hook = persistence_hook
        # Production launcher paths perform kernel-level endpoint ownership checks;
        # protocol fakes must opt in through injected probes for deterministic tests.
        self._port_is_available = port_is_available or (
            _loopback_port_is_available
            if isinstance(launcher, SubprocessGroupLauncher)
            else lambda _host, _port: True
        )
        self._listener_pid = (
            listener_pid
            if listener_pid is not None
            else _linux_listener_pid if isinstance(launcher, SubprocessGroupLauncher) else None
        )
        self.worker_assignments = tuple(
            WorkerAssignment(worker_id=f"gpu-{gpu}", gpu_id=gpu, port=_FIRST_POLICY_PORT + ordinal)
            for ordinal, gpu in enumerate(gpu_ids)
        )
        self._policy_processes: dict[str, ManagedProcess] = {}
        self._checkpoint: CheckpointReceipt | None = None
        self._accepted_attempts: dict[str, dict[str, str]] = {}
        self._pending_attempts: dict[str, str] = {}
        self._state_lock = threading.RLock()
        self._cancel_event = threading.Event()
        self._active_evaluators: dict[int, ManagedProcess] = {}

    @property
    def envelope_root(self) -> Path:
        return self.workspace / "envelopes"

    @property
    def artifact_root(self) -> Path:
        return self.workspace / "artifacts"

    @property
    def requests(self) -> tuple[EpisodeRequest, ...]:
        requests: list[EpisodeRequest] = []
        for task in self.task_manifest["tasks"]:  # type: ignore[index]
            task_value = dict(task)  # type: ignore[arg-type]
            for instance in task_value["requested_instances"]:
                instance_value = dict(instance)
                ordinal = len(requests)
                worker = self.worker_assignments[ordinal % len(self.worker_assignments)]
                requests.append(
                    EpisodeRequest(
                        episode_key=f"b100-{task_value['source_task_id']:03d}-public-{instance_value['index']:02d}",
                        task_name=str(task_value["task_name"]),
                        instance_index=int(instance_value["index"]),
                        worker_id=worker.worker_id,
                        gpu_id=worker.gpu_id,
                        port=worker.port,
                    )
                )
        if len(requests) != 1000:
            raise ControllerError("canonical B100 public_test campaign must contain exactly 1000 episodes")
        return tuple(requests)

    def start_policy_servers(self) -> CheckpointReceipt:
        """Hydrate once, then start exactly one policy server per assigned GPU."""

        self._load_campaign_manifest()
        self._reconcile_attempts()
        if self._checkpoint is None:
            self._checkpoint = hydrate_immutable_checkpoint(
                self.checkpoint_source, contract=self.contract, destination=self.checkpoint_dir
            )
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._write_campaign_manifest()
        try:
            for worker in self.worker_assignments:
                if worker.worker_id in self._policy_processes:
                    continue
                if not self._port_is_available(_POLICY_HOST, worker.port):
                    raise ControllerError(f"policy server {worker.worker_id} port is already occupied")
                command = self.policy_command + (
                    "--checkpoint", str(self._checkpoint.local_path),
                    "--host", _POLICY_HOST,
                    "--port", str(worker.port),
                )
                process = self.launcher.start(
                    command,
                    environment={"CUDA_VISIBLE_DEVICES": str(worker.gpu_id)},
                    cwd=self.workspace,
                )
                self._policy_processes[worker.worker_id] = process
                self._await_policy_readiness(worker, process)
            return self._checkpoint
        except Exception:
            self.close()
            raise

    def run(self, *, publish: Callable[..., object] | None = None) -> object | None:
        """Resume completed envelopes and evaluate every remaining canonical episode."""

        policy_closed = False
        executor: ThreadPoolExecutor | None = None
        self.start_policy_servers()
        try:
            completed = {envelope.episode_key for envelope in self._load_envelopes_if_present()}
            expected = {request.episode_key for request in self.requests}
            if not completed.issubset(expected):
                raise ControllerError("resume state contains an episode outside the canonical campaign")
            remaining_by_worker = {
                worker.worker_id: tuple(
                    request
                    for request in self.requests
                    if request.worker_id == worker.worker_id and request.episode_key not in completed
                )
                for worker in self.worker_assignments
            }
            # Each submitted sequence is the sole evaluator worker for one simulator GPU.
            # It retains no policy state and starts one bounded official evaluator process
            # per requested episode, so a timeout never strands the next episode.
            executor = ThreadPoolExecutor(max_workers=len(self.worker_assignments))
            futures = [
                executor.submit(self._run_worker, requests)
                for requests in remaining_by_worker.values()
                if requests
            ]
            try:
                completed_futures, _ = (
                    wait(futures, return_when=FIRST_EXCEPTION) if futures else (set(), set())
                )
                for future in completed_futures:
                    future.result()
            except BaseException as worker_error:
                self._cancel_event.set()
                cleanup_error: ControllerError | None = None
                try:
                    self._terminate_active_evaluators()
                except ControllerError as error:
                    # Handles remain registered after a failed termination, making a
                    # bounded caller retry possible.  Do not claim cancellation won.
                    cleanup_error = error
                for future in futures:
                    future.cancel()
                # Deliberately do not wait here: a broken evaluator must never make
                # Ctrl-C hang the controller after its bounded group termination.
                executor.shutdown(wait=False, cancel_futures=True)
                executor = None
                if cleanup_error is not None:
                    raise cleanup_error from worker_error
                raise
            else:
                executor.shutdown(wait=True)
                executor = None
            self.close()
            policy_closed = True
            if publish is None:
                return None
            return self.publish_if_complete(publish)
        finally:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
            if not policy_closed:
                self.close()

    def _run_worker(self, requests: Sequence[EpisodeRequest]) -> None:
        for request in requests:
            if self._cancel_event.is_set():
                return
            self._run_episode(request)

    def publish_if_complete(self, publisher: Callable[..., object]) -> object:
        """Publish only the exact campaign set after every key has a durable envelope."""

        self._load_campaign_manifest()
        self._reconcile_attempts()
        envelopes = self._load_envelopes_if_present()
        expected = {request.episode_key for request in self.requests}
        actual = {envelope.episode_key for envelope in envelopes}
        if actual != expected or len(envelopes) != len(expected):
            raise ControllerError("requested episodes are not terminal or quarantined")
        if any(not isinstance(envelope.outcome, Outcome) for envelope in envelopes):  # pragma: no cover - enum guard
            raise ControllerError("requested episodes have an invalid disposition")
        authenticator = self._require_authenticator()
        artifact_roots = self._accepted_attempt_roots(envelopes)
        keyword_args: dict[str, object] = {
            "episodes": envelopes,
            "artifact_roots": artifact_roots,
            "contract": self.contract,
            "task_manifest": self.task_manifest,
            "authenticator": authenticator,
        }
        return publisher(**keyword_args)

    def close(self) -> None:
        """Boundedly terminate only process groups this controller started."""

        processes = tuple(self._policy_processes.items())
        errors: list[Exception] = []
        for worker_id, process in processes:
            try:
                self.launcher.terminate_group(process, timeout_seconds=self.shutdown_timeout_seconds)
            except Exception as error:
                errors.append(error)
            else:
                with self._state_lock:
                    if self._policy_processes.get(worker_id) is process:
                        del self._policy_processes[worker_id]
        if errors:
            raise ControllerError("policy process group cleanup failed") from errors[0]

    def _run_episode(self, request: EpisodeRequest) -> None:
        if self._cancel_event.is_set():
            raise ControllerError("campaign cancellation is in progress")
        authenticator = self._require_authenticator()
        output_dir = self._new_attempt_directory(request)
        command = build_official_evaluator_command(
            task_name=request.task_name,
            mode=self.contract.evaluator_mode,
            instance_index=request.instance_index,
            port=request.port,
            output_dir=output_dir,
        )
        process = self.launcher.start(
            command,
            environment={"CUDA_VISIBLE_DEVICES": str(request.gpu_id)},
            cwd=self.workspace,
        )
        self._register_active_evaluator(process)
        group_terminated = False
        try:
            returncode = process.wait(self.episode_timeout_seconds)
        except (TimeoutError, subprocess.TimeoutExpired):
            self._terminate_process_group(process)
            group_terminated = True
            classified = self._controller_quarantine(request, output_dir, "timeout", authenticator)
        except KeyboardInterrupt:
            self._terminate_process_group(process)
            group_terminated = True
            classified = self._controller_quarantine(request, output_dir, "interrupted", authenticator)
            self._persist_episode(request, output_dir, classified, authenticator)
            raise
        except Exception:
            self._terminate_process_group(process)
            group_terminated = True
            classified = self._controller_quarantine(request, output_dir, "crashed", authenticator)
        else:
            self._terminate_process_group(process)
            group_terminated = True
            if returncode != 0:
                classified = self._controller_quarantine(request, output_dir, "crashed", authenticator)
            else:
                try:
                    classified = self._discover_official_evaluator_output(
                        request, output_dir, authenticator
                    )
                except ControllerError as error:
                    classified = self._controller_quarantine(
                        request, output_dir, f"official_evaluator_output:{error}", authenticator
                    )
        finally:
            if group_terminated:
                self._unregister_active_evaluator(process)
        self._persist_episode(request, output_dir, classified, authenticator)

    def _controller_quarantine(
        self,
        request: EpisodeRequest,
        output_dir: Path,
        status: str,
        authenticator: ProvenanceAuthenticator,
    ) -> ClassifiedOutcome:
        """Attest a local controller failure as observed file evidence, never a bare mapping."""

        evidence = output_dir / "controller-outcome.json"
        _atomic_json_file(evidence, {"status": status, "completed": False})
        return classify_outcome_file(
            evidence,
            task_manifest=self.task_manifest,
            episode_key=request.episode_key,
            contract=self.contract,
            authenticator=authenticator,
        )

    def _discover_official_evaluator_output(
        self,
        request: EpisodeRequest,
        output_dir: Path,
        authenticator: ProvenanceAuthenticator,
    ) -> ClassifiedOutcome:
        """Normalize exactly one pinned upstream ``eval.py`` result into our contract.

        BEHAVIOR commit 26f2 writes only task/instance/rollout/metrics JSON; the
        contract fields below are controller-owned normalization, never required in
        the upstream result file.
        """

        resolved_instance = _resolved_instance_id(request, self.contract.evaluator_mode)
        expected_name = f"{request.task_name}_{resolved_instance}_0.json"
        result_root = output_dir / "json"
        try:
            candidates = tuple(sorted(result_root.glob("*.json")))
        except OSError as error:
            raise ControllerError("official evaluator output is unreadable") from error
        if len(candidates) != 1 or candidates[0].name != expected_name:
            raise ControllerError("official evaluator output is missing, duplicate, or not scheduled")
        evidence = candidates[0]
        if evidence.is_symlink() or not evidence.is_file():
            raise ControllerError("official evaluator output is unsafe")
        try:
            raw = evidence.read_bytes()
            record = json.loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ControllerError("official evaluator output is malformed") from error
        _validate_official_result(record, request=request, resolved_instance=resolved_instance)
        artifact_hashes = _attempt_artifact_hashes(output_dir)
        normalized = {
            "schema_version": 1,
            "episode_id": request.episode_key,
            "rollout_id": 0,
            "task": request.task_name,
            "steps": record["steps"],
            "success": record["success"],
            "q_score": record["q_score"],
            "time": record["time"],
            "agent_distance": record["agent_distance"],
            "normalized_agent_distance": record["normalized_agent_distance"],
            "completed": True,
            "mode": self.contract.evaluator_mode,
            "instance_id": resolved_instance,
            "instance_index": request.instance_index,
            "contract": self.contract.to_dict(),
            "artifact_hashes": artifact_hashes,
        }
        classified = classify_outcome(normalized, task_manifest=self.task_manifest)
        if classified.outcome is Outcome.QUARANTINE:
            raise ControllerError("official evaluator output metrics are invalid")
        provenance = {
            "origin": "file",
            "disposition": "regular",
            "basename": expected_name,
            "reason_code": "official_evaluator_v1",
            "diagnostic": f"official_evaluator_v1:{hashlib.sha256(raw).hexdigest()}",
            "raw_evidence_sha256": classified.raw_evidence_sha256,
        }
        classified = replace(classified, provenance=provenance)
        fields = {
            "episode_id": classified.episode_id,
            "rollout_id": classified.rollout_id,
            "evaluator_identity": classified.evaluator_identity,
            "outcome": classified.outcome.value,
            "reason": classified.reason,
            "raw_evidence_sha256": classified.raw_evidence_sha256,
            "final_q_scores": classified.final_q_scores,
            "evaluator_metrics": classified.evaluator_metrics,
            "provenance": classified.provenance,
        }
        return replace(
            classified,
            provenance_attestation=authenticator.sign(
                canonical_attestation_payload(self.contract, request.episode_key, fields)
            ),
        )

    def _await_policy_readiness(self, worker: WorkerAssignment, process: ManagedProcess) -> None:
        """Bound health readiness and reject a port response from an exited child."""

        url = f"http://{_POLICY_HOST}:{worker.port}/healthz"
        deadline = time.monotonic() + self.shutdown_timeout_seconds
        while True:
            if self._process_exit_status(process) is not None:
                raise ControllerError(f"policy server {worker.worker_id} exited before readiness")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ControllerError(f"policy server {worker.worker_id} failed /healthz readiness")
            try:
                healthy = self.health_probe(url, min(1.0, remaining))
            except Exception:
                healthy = False
            if healthy:
                # A success from an unrelated process on an occupied port is not
                # sufficient if our newly launched policy has already exited.
                if self._process_exit_status(process) is None:
                    self._verify_policy_listener(worker, process)
                    return
                raise ControllerError(f"policy server {worker.worker_id} exited before readiness")
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    def _verify_policy_listener(self, worker: WorkerAssignment, process: ManagedProcess) -> None:
        """When enabled, require the ready listener to be in our launched process group."""

        if self._checkpoint is None or self._checkpoint.model_commit != self.contract.model_commit:
            raise ControllerError("policy server checkpoint is not bound to the frozen contract")
        listener_pid = self._listener_pid
        if listener_pid is None:
            return
        owner = listener_pid(_POLICY_HOST, worker.port)
        if owner is None:
            raise ControllerError(f"policy server {worker.worker_id} listener ownership is unavailable")
        try:
            same_group = owner == process.pid or os.getpgid(owner) == process.pid
        except OSError as error:
            raise ControllerError(f"policy server {worker.worker_id} listener ownership is unreadable") from error
        if not same_group:
            raise ControllerError(f"policy server {worker.worker_id} health listener is not the launched process")

    @staticmethod
    def _process_exit_status(process: ManagedProcess) -> int | None:
        poll = getattr(process, "poll", None)
        if callable(poll):
            status = poll()
            if status is not None and not isinstance(status, int):
                raise ControllerError("managed process returned an invalid exit status")
            return status
        return None

    def _evidence_matches_request(
        self, classified: ClassifiedOutcome, request: EpisodeRequest
    ) -> bool:
        """Accept terminal evaluator evidence only for its exact requested task tuple."""

        if classified.outcome is Outcome.QUARANTINE:
            return True
        identity = classified.evaluator_identity
        return (
            isinstance(identity, Mapping)
            and identity.get("task") == request.task_name
            and identity.get("mode") == self.contract.evaluator_mode
            and identity.get("instance_index") == request.instance_index
        )

    def _persist_episode(
        self,
        request: EpisodeRequest,
        output_dir: Path,
        classified: ClassifiedOutcome,
        authenticator: ProvenanceAuthenticator,
    ) -> None:
        # One critical section keeps envelope persistence and the manifest binding in
        # lockstep across concurrent per-GPU workers.
        with self._state_lock:
            relative = output_dir.relative_to(self.workspace).as_posix()
            self._pending_attempts[request.episode_key] = relative
            self._write_campaign_manifest()
            self._persistence_boundary("pending_attempt_durable")
            write_episode_envelope(
                self.envelope_root,
                request.episode_key,
                classified,
                contract=self.contract,
                authenticator=authenticator,
            )
            self._persistence_boundary("envelope_durable")
            envelope = next(
                item
                for item in load_episode_envelopes(
                    self.envelope_root,
                    contract=self.contract,
                    authenticator=authenticator,
                )
                if item.episode_key == request.episode_key
            )
            self._verify_attempt_matches_envelope(output_dir, envelope)
            self._accepted_attempts[request.episode_key] = {
                "attempt_root": relative,
                "envelope_sha256": envelope.canonical_sha256,
            }
            del self._pending_attempts[request.episode_key]
            self._write_campaign_manifest()
            self._persistence_boundary("accepted_binding_durable")

    def _new_attempt_directory(self, request: EpisodeRequest) -> Path:
        """Reserve an unused evaluator output directory without touching prior attempts."""

        episode_root = self.artifact_root / request.episode_key
        attempts_root = episode_root / "attempts"
        for directory in (self.artifact_root, episode_root, attempts_root):
            try:
                directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            except OSError as error:
                raise ControllerError("unable to create evaluator attempt workspace") from error
            if directory.is_symlink() or not directory.is_dir():
                raise ControllerError("evaluator attempt workspace is unsafe")
        for ordinal in range(1, 1_000_000):
            attempt = attempts_root / f"attempt-{ordinal:06d}"
            try:
                attempt.mkdir(mode=0o700)
            except FileExistsError:
                continue
            except OSError as error:
                raise ControllerError("unable to reserve evaluator attempt workspace") from error
            return attempt
        raise ControllerError("evaluator attempt workspace is exhausted")

    def _accepted_attempt_roots(
        self, envelopes: Sequence[EpisodeEnvelope]
    ) -> dict[str, Path]:
        roots: dict[str, Path] = {}
        for envelope in envelopes:
            binding = self._accepted_attempts.get(envelope.episode_key)
            if binding is None or binding.get("envelope_sha256") != envelope.canonical_sha256:
                raise ControllerError("accepted episode attempt is not durably bound to its envelope")
            try:
                relative = Path(binding["attempt_root"])
                root = (self.workspace / relative).resolve(strict=True)
                root.relative_to(self.artifact_root.resolve(strict=True))
            except (KeyError, OSError, ValueError) as error:
                raise ControllerError("accepted episode attempt root is invalid") from error
            if root.is_symlink() or not root.is_dir():
                raise ControllerError("accepted episode attempt root is unsafe")
            if _evidence_declares_artifacts(envelope.raw_evidence):
                roots[envelope.episode_key] = root
        return roots

    def _reconcile_attempts(self) -> None:
        """Recover a crash between durable attempt intent and final envelope binding."""

        with self._state_lock:
            if not self._pending_attempts:
                return
            envelopes = {item.episode_key: item for item in self._load_envelopes_if_present()}
            changed = False
            for episode_key, relative in tuple(self._pending_attempts.items()):
                envelope = envelopes.get(episode_key)
                if envelope is None:
                    # No envelope was committed; preserve the untrusted old attempt and
                    # let resume allocate a new one rather than reusing it.
                    continue
                try:
                    attempt = (self.workspace / relative).resolve(strict=True)
                    attempt.relative_to(self.artifact_root.resolve(strict=True))
                except (OSError, ValueError) as error:
                    raise ControllerError("pending evaluator attempt root is invalid") from error
                self._verify_attempt_matches_envelope(attempt, envelope)
                self._accepted_attempts[episode_key] = {
                    "attempt_root": relative,
                    "envelope_sha256": envelope.canonical_sha256,
                }
                del self._pending_attempts[episode_key]
                changed = True
            if changed:
                self._write_campaign_manifest()

    def _verify_attempt_matches_envelope(self, attempt: Path, envelope: EpisodeEnvelope) -> None:
        provenance = envelope.provenance
        basename = provenance.get("basename") if isinstance(provenance, Mapping) else None
        if not isinstance(basename, str) or not basename:
            raise ControllerError("accepted attempt lacks file-origin evaluator evidence")
        if provenance.get("reason_code") == "official_evaluator_v1":
            evidence = attempt / "json" / basename
            diagnostic = provenance.get("diagnostic")
            expected_raw_sha256 = (
                diagnostic.removeprefix("official_evaluator_v1:")
                if isinstance(diagnostic, str) and diagnostic.startswith("official_evaluator_v1:")
                else None
            )
        else:
            evidence = attempt / basename
            expected_raw_sha256 = envelope.raw_evidence_sha256
        if evidence.is_symlink() or not evidence.is_file():
            # The evidence classifier deliberately records a symlink/unreadable
            # evaluator path as an authenticated file-origin quarantine.  It cannot
            # be read back safely, so never pass this attempt to publication (there
            # are no declared artifacts), but retain the durable quarantine binding.
            disposition = provenance.get("disposition") if isinstance(provenance, Mapping) else None
            if (
                envelope.outcome is Outcome.QUARANTINE
                and disposition in {"symlink", "unreadable"}
                and not _evidence_declares_artifacts(envelope.raw_evidence)
            ):
                return
            raise ControllerError("accepted attempt evaluator evidence is invalid")
        try:
            raw = evidence.read_bytes()
        except OSError as error:
            raise ControllerError("accepted attempt evaluator evidence is unreadable") from error
        if raw_evidence_sha256(raw) != expected_raw_sha256:
            raise ControllerError("accepted attempt does not match its envelope evidence")

    def _terminate_process_group(self, process: ManagedProcess) -> None:
        try:
            self.launcher.terminate_group(process, timeout_seconds=self.shutdown_timeout_seconds)
        except Exception as error:
            raise ControllerError("evaluator process group cleanup failed") from error

    def _register_active_evaluator(self, process: ManagedProcess) -> None:
        with self._state_lock:
            self._active_evaluators[id(process)] = process
            if not self._cancel_event.is_set():
                return
        try:
            self._terminate_process_group(process)
        except Exception:
            # Retain the handle for the next bounded cancellation/close attempt.
            raise
        self._unregister_active_evaluator(process)
        raise ControllerError("campaign cancellation is in progress")

    def _unregister_active_evaluator(self, process: ManagedProcess) -> None:
        with self._state_lock:
            self._active_evaluators.pop(id(process), None)

    def _terminate_active_evaluators(self) -> None:
        """Signal every active evaluator, taking a bounded best-effort cleanup pass."""

        with self._state_lock:
            processes = tuple(self._active_evaluators.items())
        errors: list[Exception] = []
        for handle, process in processes:
            try:
                self._terminate_process_group(process)
            except Exception as error:
                errors.append(error)
            else:
                with self._state_lock:
                    if self._active_evaluators.get(handle) is process:
                        del self._active_evaluators[handle]
        if errors:
            raise ControllerError("active evaluator process group cleanup failed") from errors[0]

    def _persistence_boundary(self, name: str) -> None:
        if self._persistence_hook is not None:
            self._persistence_hook(name)

    def _load_envelopes_if_present(self) -> tuple[EpisodeEnvelope, ...]:
        if not self.envelope_root.exists():
            return ()
        try:
            return load_episode_envelopes(
                self.envelope_root,
                contract=self.contract,
                authenticator=self._require_authenticator(),
            )
        except Exception as error:
            raise ControllerError("campaign envelope state cannot be trusted") from error

    def _require_authenticator(self) -> ProvenanceAuthenticator:
        if self.provenance_authenticator is None:
            raise ControllerError("campaign provenance authenticator is required")
        return self.provenance_authenticator

    def _write_campaign_manifest(self) -> None:
        self.workspace.mkdir(mode=0o700, parents=True, exist_ok=True)
        existing = self._read_campaign_manifest()
        if existing is not None:
            self._validate_campaign_manifest(existing)
        payload: dict[str, object] = {
            "schema_version": 2,
            "contract_identity": self.contract.identity,
            "task_manifest_sha256": self.contract.task_manifest_sha256,
            "worker_assignments": [asdict(item) for item in self.worker_assignments],
            "requested_episode_count": len(self.requests),
            "accepted_attempts": {
                key: dict(value) for key, value in sorted(self._accepted_attempts.items())
            },
            "pending_attempts": dict(sorted(self._pending_attempts.items())),
        }
        reject_credential_material(payload)
        _atomic_json_file(self.workspace / "campaign-manifest.json", payload)

    def _load_campaign_manifest(self) -> None:
        payload = self._read_campaign_manifest()
        if payload is None:
            self._accepted_attempts = {}
            self._pending_attempts = {}
            return
        self._validate_campaign_manifest(payload)
        attempts = payload["accepted_attempts"]
        assert isinstance(attempts, Mapping)
        self._accepted_attempts = {
            key: dict(value) for key, value in attempts.items()  # type: ignore[arg-type]
        }
        pending = payload["pending_attempts"]
        assert isinstance(pending, Mapping)
        self._pending_attempts = {key: value for key, value in pending.items()}  # type: ignore[misc]

    def _read_campaign_manifest(self) -> Mapping[str, object] | None:
        path = self.workspace / "campaign-manifest.json"
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ControllerError("campaign manifest is unsafe")
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ControllerError("campaign manifest is unreadable") from error
        if not isinstance(payload, Mapping):
            raise ControllerError("campaign manifest is invalid")
        return payload

    def _validate_campaign_manifest(self, payload: Mapping[str, object]) -> None:
        expected = {
            "schema_version": 2,
            "contract_identity": self.contract.identity,
            "task_manifest_sha256": self.contract.task_manifest_sha256,
            "worker_assignments": [asdict(item) for item in self.worker_assignments],
            "requested_episode_count": len(self.requests),
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ControllerError("campaign manifest does not match this immutable campaign")
        attempts = payload.get("accepted_attempts")
        if not isinstance(attempts, Mapping):
            raise ControllerError("campaign manifest accepted attempts are invalid")
        pending = payload.get("pending_attempts")
        if not isinstance(pending, Mapping):
            raise ControllerError("campaign manifest pending attempts are invalid")
        for key, binding in attempts.items():
            if not isinstance(key, str) or not isinstance(binding, Mapping):
                raise ControllerError("campaign manifest accepted attempts are invalid")
            if set(binding) != {"attempt_root", "envelope_sha256"}:
                raise ControllerError("campaign manifest accepted attempts are invalid")
            root = binding.get("attempt_root")
            digest = binding.get("envelope_sha256")
            if (
                not isinstance(root, str)
                or not root.startswith("artifacts/")
                or Path(root).is_absolute()
                or any(part in ("", ".", "..") for part in Path(root).parts)
            ):
                raise ControllerError("campaign manifest attempt root is invalid")
            try:
                require_sha256(digest, label="campaign envelope")
            except ValueError as error:
                raise ControllerError("campaign manifest envelope hash is invalid") from error
        for key, root in pending.items():
            if (
                not isinstance(key, str)
                or not isinstance(root, str)
                or not root.startswith("artifacts/")
                or Path(root).is_absolute()
                or any(part in ("", ".", "..") for part in Path(root).parts)
            ):
                raise ControllerError("campaign manifest pending attempt is invalid")


def _validate_checkpoint_receipt(
    receipt: CheckpointReceipt, *, contract: RolloutContract, destination: Path
) -> None:
    if not isinstance(receipt, CheckpointReceipt):
        raise ControllerError("checkpoint transport returned an invalid receipt")
    if receipt.model_commit != contract.model_commit or receipt.artifact_sha256 != contract.checkpoint_artifact_sha256:
        raise ControllerError("checkpoint readback does not match immutable rollout contract")
    if receipt.local_path.resolve() != destination:
        raise ControllerError("checkpoint receipt local path does not match requested destination")


def checkpoint_tree_sha256(root: Path) -> str:
    """Canonical SHA-256 of regular checkpoint files: path, size, and file digest.

    This is the single immutable-checkpoint identity used by both the runtime CLI
    and the controller adapter; it intentionally refuses symlinks and special files.
    """

    if root.is_symlink() or not root.is_dir():
        raise ControllerError("checkpoint tree is invalid")
    digest = hashlib.sha256()
    entries = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
    for path in entries:
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ControllerError("checkpoint tree must not contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ControllerError("checkpoint tree contains an unsupported entry")
        file_hash = hashlib.sha256()
        try:
            with path.open("rb") as reader:
                while chunk := reader.read(1024 * 1024):
                    file_hash.update(chunk)
        except OSError as error:
            raise ControllerError("checkpoint tree is unreadable") from error
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_hash.hexdigest().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _manifest_checkpoint_source(root: Path) -> tuple[Path, str | None]:
    """Select and verify ``checkpoint/**`` from a trainer final-manifest v1."""

    manifest_path = root / "final-manifest.json"
    if not manifest_path.exists():
        return root, None
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ControllerError("final checkpoint manifest is invalid")
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ControllerError("final checkpoint manifest is unreadable") from error
    files = manifest.get("files") if isinstance(manifest, Mapping) else None
    if manifest.get("schema_version") != 1 or not isinstance(files, Mapping):
        raise ControllerError("final checkpoint manifest schema is invalid")
    checkpoint = root / "checkpoint"
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ControllerError("final checkpoint manifest has no checkpoint subtree")
    expected = {
        path.removeprefix("checkpoint/"): value
        for path, value in files.items()
        if isinstance(path, str) and path.startswith("checkpoint/")
    }
    if not expected:
        raise ControllerError("final checkpoint manifest has no checkpoint files")
    actual = {
        path.relative_to(checkpoint).as_posix(): path
        for path in checkpoint.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if set(actual) != set(expected):
        raise ControllerError("checkpoint subtree does not match final manifest")
    for relative, entry in expected.items():
        if not isinstance(entry, Mapping):
            raise ControllerError("final checkpoint manifest file entry is invalid")
        digest = entry.get("sha256")
        size = entry.get("size")
        if not isinstance(digest, str) or type(size) is not int or size < 0:
            raise ControllerError("final checkpoint manifest file entry is invalid")
        try:
            require_sha256(digest, label="final checkpoint file")
        except ValueError as error:
            raise ControllerError("final checkpoint manifest file entry is invalid") from error
        path = actual[relative]
        if path.stat().st_size != size or _file_sha256(path) != digest:
            raise ControllerError("checkpoint subtree does not match final manifest")
    return checkpoint, hashlib.sha256(manifest_bytes).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as reader:
            while chunk := reader.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise ControllerError("checkpoint file is unreadable") from error
    return digest.hexdigest()


def _loopback_port_is_available(host: str, port: int) -> bool:
    """Return true only when the exact loopback TCP endpoint is currently closed."""

    if host != _POLICY_HOST or not 1 <= port <= 65535:
        raise ControllerError("policy port preflight input is invalid")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.settimeout(0.2)
            return client.connect_ex((host, port)) != 0
    except OSError as error:
        raise ControllerError("policy port preflight failed") from error


def _linux_listener_pid(host: str, port: int) -> int | None:
    """Resolve a loopback listening socket inode to its owning PID on Linux."""

    if host != _POLICY_HOST or not Path("/proc").is_dir():
        return None
    inodes: set[str] = set()
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            lines = table.read_text(encoding="ascii").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A":
                continue
            try:
                local_port = int(fields[1].rsplit(":", 1)[1], 16)
            except ValueError:
                continue
            if local_port == port:
                inodes.add(fields[9])
    if not inodes:
        return None
    try:
        process_entries = tuple(Path("/proc").iterdir())
    except OSError:
        return None
    for entry in process_entries:
        if not entry.name.isdecimal():
            continue
        fd_root = entry / "fd"
        try:
            descriptors = tuple(fd_root.iterdir())
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except OSError:
                continue
            if target.startswith("socket:[") and target[8:-1] in inodes:
                return int(entry.name)
    return None


def _resolved_instance_id(request: EpisodeRequest, mode: str) -> int:
    if mode == "public_test":
        return 301 + request.instance_index
    if mode == "train":
        return request.instance_index
    raise ControllerError("official evaluator mode is unsupported")


def _validate_official_result(
    value: object, *, request: EpisodeRequest, resolved_instance: int
) -> None:
    if not isinstance(value, Mapping):
        raise ControllerError("official evaluator output is malformed")
    required = {
        "task", "instance_id", "rollout_id", "steps", "success",
        "q_score", "time", "agent_distance", "normalized_agent_distance",
    }
    if set(value) != required:
        raise ControllerError("official evaluator output fields are invalid")
    if (
        value.get("task") != request.task_name
        or value.get("instance_id") != resolved_instance
        or value.get("rollout_id") != 0
        or type(value.get("steps")) is not int
        or value["steps"] <= 0
        or type(value.get("success")) is not bool
    ):
        raise ControllerError("official evaluator output does not match the scheduled request")
    q_score = value.get("q_score")
    time_metrics = value.get("time")
    distances = value.get("agent_distance")
    normalized = value.get("normalized_agent_distance")
    if (
        not isinstance(q_score, Mapping)
        or set(q_score) != {"final"}
        or not isinstance(q_score["final"], (int, float))
        or isinstance(q_score["final"], bool)
        or not isinstance(time_metrics, Mapping)
        or set(time_metrics) != {"simulator_steps", "simulator_time", "normalized_time"}
        or time_metrics.get("simulator_steps") != value["steps"]
        or not isinstance(distances, Mapping)
        or not isinstance(normalized, Mapping)
        or set(distances) != {"base", "left", "right"}
        or set(normalized) != {"base", "left", "right"}
    ):
        raise ControllerError("official evaluator output metrics are invalid")


def _attempt_artifact_hashes(root: Path) -> dict[str, str]:
    if root.is_symlink() or not root.is_dir():
        raise ControllerError("official evaluator artifact root is invalid")
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ControllerError("official evaluator artifacts must not contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ControllerError("official evaluator artifact is invalid")
        relative = path.relative_to(root).as_posix()
        digest = hashlib.sha256()
        try:
            with path.open("rb") as reader:
                while chunk := reader.read(1024 * 1024):
                    digest.update(chunk)
        except OSError as error:
            raise ControllerError("official evaluator artifact is unreadable") from error
        hashes[relative] = digest.hexdigest()
    if not hashes:
        raise ControllerError("official evaluator did not produce artifacts")
    return hashes


def _atomic_json_file(path: Path, payload: Mapping[str, object]) -> None:
    """Write controller metadata/evidence durably without retaining secrets."""

    reject_credential_material(payload)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    descriptor, staged_name = tempfile.mkstemp(prefix=".b1k-", dir=path.parent)
    staged = Path(staged_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as writer:
            writer.write(encoded)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(staged, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        staged.unlink(missing_ok=True)


def _evidence_declares_artifacts(raw_evidence: object) -> bool:
    if isinstance(raw_evidence, bytes):
        try:
            raw_evidence = json.loads(raw_evidence)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return False
    elif isinstance(raw_evidence, str):
        try:
            raw_evidence = json.loads(raw_evidence)
        except json.JSONDecodeError:
            return False
    if not isinstance(raw_evidence, Mapping):
        return False
    artifact_hashes = raw_evidence.get("artifact_hashes")
    return isinstance(artifact_hashes, Mapping) and bool(artifact_hashes)
