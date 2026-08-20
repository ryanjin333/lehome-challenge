#!/usr/bin/env python3
"""Lease immutable jobs and execute them through the production runtime adapter."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import ssl
import stat
import subprocess
import threading
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from lehome_train.groot.experiment_job import ExperimentJob
from lehome_train.groot.experiment_deployment_gate import (
    bind_training_job_identity,
    load_deployment_gate,
    load_training_image_manifest,
)
from lehome_train.groot.experiment_worker import ControllerProtocolError, ControllerUnavailable, ExperimentWorker, PreemptionRequested
from lehome_train.groot.experiment_service import load_bearer_token
from lehome_train.io import sha256_file

class HttpControllerClient:
    def __init__(self, url: str, token_file: Path, manifest_set_sha256: str, ca_file: Path | None) -> None:
        parsed = urlsplit(url)
        if parsed.scheme == "https":
            if ca_file is None:
                raise ValueError("controller HTTPS requires a private CA file")
        elif not (parsed.scheme == "http" and parsed.hostname == "127.0.0.1"):
            raise ValueError("controller URL must use private TLS or loopback")
        self.tls_context: ssl.SSLContext | None = None
        if ca_file is not None:
            private_ca = Path(ca_file)
            if (
                not private_ca.is_absolute()
                or private_ca.is_symlink()
                or not private_ca.is_file()
                or stat.S_IMODE(private_ca.stat().st_mode) & 0o022
            ):
                raise ValueError("controller private CA file is unsafe")
            self.tls_context = ssl.create_default_context(cafile=str(private_ca))
        self.url, self.token, self.manifest_set_sha256 = url.rstrip("/"), load_bearer_token(token_file), manifest_set_sha256
    def _post(self, endpoint: str, payload: dict[str, object]) -> dict[str, object]:
        request = Request(self.url + endpoint, data=json.dumps(payload, separators=(",", ":")).encode(), headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=20, context=self.tls_context) as response: result = json.loads(response.read())
        except HTTPError as error:
            if error.code == 429 or error.code >= 500:
                raise ControllerUnavailable("controller HTTP transport unavailable") from error
            raise ControllerProtocolError("controller rejected immutable request") from error
        except (URLError, TimeoutError, OSError) as error:
            raise ControllerUnavailable("controller transport unavailable") from error
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ControllerProtocolError("controller returned malformed JSON") from error
        if not isinstance(result, dict): raise ControllerProtocolError("controller returned invalid JSON")
        return result
    def lease_next(self, worker_id: str, capability: str, *, now_ns: int, lease_ns: int):
        result = self._post("/lease", {"worker_id": worker_id, "capability": capability, "now_ns": now_ns, "lease_ns": lease_ns, "manifest_set_sha256": self.manifest_set_sha256})
        if result.get("lease") is None: return None
        from lehome_train.groot.experiment_controller import JobLease
        raw = result.get("job")
        if not isinstance(raw, dict): raise ValueError("controller lease lacks immutable job")
        from lehome_train.groot.experiment_job import _parse
        publication = result.get("publication")
        parent_publication = result.get("parent_publication")
        evaluation_matrix_sha256 = result.get("evaluation_matrix_sha256")
        if publication is not None and not isinstance(publication, dict): raise ValueError("controller lease publication is invalid")
        if parent_publication is not None and not isinstance(parent_publication, dict): raise ValueError("controller lease parent publication is invalid")
        if evaluation_matrix_sha256 is not None and (type(evaluation_matrix_sha256) is not str or len(evaluation_matrix_sha256) != 64 or any(character not in "0123456789abcdef" for character in evaluation_matrix_sha256)): raise ValueError("controller lease evaluation matrix is invalid")
        return JobLease(str(result["lease_id"]), str(result["experiment_id"]), str(result.get("worker_id", worker_id)), capability, int(result["expires_ns"]), _parse(raw), publication, parent_publication, evaluation_matrix_sha256)
    def heartbeat(self, lease: object, now_ns: int, lease_ns: int):
        return self._post("/heartbeat", {"lease_id": lease.lease_id, "worker_id": lease.worker_id, "now_ns": now_ns, "lease_ns": lease_ns})
    def reconcile_terminal_receipt(self, lease: object, receipt: str, now_ns: int) -> str:
        result = self._post("/complete", {"lease_id": lease.lease_id, "experiment_id": lease.experiment_id, "worker_id": lease.worker_id, "receipt_sha256": receipt, "now_ns": now_ns})
        state = result.get("status")
        if type(state) is not str or not state:
            raise ControllerProtocolError("controller completion response is invalid")
        return state.upper()
    def complete(self, lease: object, receipt: str, now_ns: int) -> None: self._post("/complete", {"lease_id": lease.lease_id, "experiment_id": lease.experiment_id, "worker_id": lease.worker_id, "receipt_sha256": receipt, "now_ns": now_ns})
    def publication_verified(self, experiment_id: str, publication: dict[str, object], now_ns: int) -> str:
        result = self._post("/publication", {"experiment_id": experiment_id, "publication": publication, "now_ns": now_ns})
        state = result.get("status")
        if type(state) is not str or not state:
            raise ControllerProtocolError("controller publication response is invalid")
        return state.upper()
    def satisfy_dependency(self, receipt: dict[str, object], now_ns: int) -> int: return int(self._post("/dependency", {"receipt": receipt, "now_ns": now_ns})["unblocked"])
    def submit_evaluation(self, lease: object, report: dict[str, object], now_ns: int) -> None: self._post("/evaluation", {"lease_id": lease.lease_id, "experiment_id": lease.experiment_id, "worker_id": lease.worker_id, "report": report, "now_ns": now_ns})
    def retryable(self, lease: object, reason: str, now_ns: int) -> None: self._post("/retryable", {"lease_id": lease.lease_id, "experiment_id": lease.experiment_id, "worker_id": lease.worker_id, "reason": reason, "now_ns": now_ns})
    def block_infrastructure(self, lease: object, reason: str, now_ns: int) -> None: self._post("/block", {"lease_id": lease.lease_id, "experiment_id": lease.experiment_id, "worker_id": lease.worker_id, "reason": reason, "now_ns": now_ns})

class ProductionRuntimeExperimentRunner:
    def __init__(self, cache_root: Path, output_root: Path, hf_token_file: Path, *, hydrator: Any | None = None, parent_hub: Any | None = None, training_script: Path = Path("/opt/lehome/guest/bin/lehome-training.sh"), process_runner: Any = subprocess.run) -> None:
        self.cache_root, self.output_root, self.hf_token_file = cache_root, output_root, hf_token_file
        self.hydrator, self.parent_hub, self.training_script, self.process_runner = hydrator, parent_hub, training_script, process_runner
        self._result_outputs: dict[str, Path] = {}

    def _handoff_root(self) -> Path:
        root = self.output_root / "publication-handoffs"
        if root.is_symlink():
            raise ValueError("publication handoff root is unsafe")
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        return root

    @staticmethod
    def _canonical_json(value: object) -> bytes:
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

    def _handoff_path(self, experiment_id: str) -> Path:
        if type(experiment_id) is not str or len(experiment_id) != 64:
            raise ValueError("publication handoff experiment identity is invalid")
        return self._handoff_root() / (experiment_id + ".json")

    def _validate_workspace_inputs(self, workspace: Path, request_set: object) -> None:
        """Verify immutable request bytes before reusing a preempted workspace."""
        root = getattr(request_set, "root", None)
        if not isinstance(root, Path) or workspace.is_symlink() or not workspace.is_dir():
            raise ValueError("preempted workspace is unsafe")
        source_manifest, workspace_manifest = root / "bundle-manifest.json", workspace / "bundle-manifest.json"
        if source_manifest.is_symlink() or workspace_manifest.is_symlink() or not source_manifest.is_file() or not workspace_manifest.is_file() or source_manifest.read_bytes() != workspace_manifest.read_bytes():
            raise ValueError("preempted workspace request-set manifest is tampered")
        try:
            manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
            entries = manifest["files"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError("preempted workspace request-set manifest is invalid") from error
        if not isinstance(entries, list):
            raise ValueError("preempted workspace request-set manifest is invalid")
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"path", "byte_size", "sha256"}:
                raise ValueError("preempted workspace request-set manifest is invalid")
            relative = entry["path"]
            if type(relative) is not str or relative.startswith("/") or ".." in Path(relative).parts:
                raise ValueError("preempted workspace request-set path is unsafe")
            candidate = workspace / relative
            if candidate.is_symlink() or not candidate.is_file() or candidate.stat().st_size != entry["byte_size"] or sha256_file(candidate) != entry["sha256"]:
                raise ValueError("preempted workspace immutable bytes are tampered")

    def persist_pending_publication(self, lease: object, receipt: str, publication: dict[str, object]) -> None:
        """Persist an exact terminal receipt before controller acknowledgement."""
        from lehome_train.groot.experiment_job import _parse
        from lehome_train.groot.experiment_publication import bind_checkpoint_publication

        job = getattr(lease, "job", None)
        experiment_id = getattr(lease, "experiment_id", None)
        lease_id, worker_id, capability = getattr(lease, "lease_id", None), getattr(lease, "worker_id", None), getattr(lease, "capability", None)
        if (
            not isinstance(job, ExperimentJob) or experiment_id != job.experiment_id or type(receipt) is not str
            or not all(type(value) is str and value for value in (lease_id, worker_id))
            or capability != "training"
        ):
            raise ValueError("publication handoff is malformed")
        bind_checkpoint_publication(job, receipt, publication)
        result = self._result_outputs.get(experiment_id)
        if result is None or result.is_symlink() or not result.is_file() or sha256_file(result) != receipt:
            raise ValueError("publication handoff result bytes are missing or tampered")
        document = {
            "schema_version": 2,
            "experiment_id": experiment_id,
            "job": dict(job.raw),
            "lease": {"lease_id": lease_id, "worker_id": worker_id, "capability": capability},
            "receipt_sha256": receipt,
            "publication": publication,
            "result_output": str(result),
        }
        path, payload = self._handoff_path(experiment_id), self._canonical_json(document)
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise ValueError("publication handoff conflicts with existing receipt")
            return
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    def pending_publications(self) -> tuple[tuple[object, str, dict[str, object]], ...]:
        """Read only exact, byte-verified durable publication handoffs."""
        from lehome_train.groot.experiment_job import _parse
        from lehome_train.groot.experiment_publication import bind_checkpoint_publication

        entries: list[tuple[object, str, dict[str, object]]] = []
        root = self._handoff_root()
        for path in sorted(root.glob("*.json")):
            if path.is_symlink() or not path.is_file():
                raise ValueError("publication handoff is unsafe")
            try:
                raw = path.read_bytes(); document = json.loads(raw)
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError("publication handoff is malformed") from error
            schema_version = document.get("schema_version") if isinstance(document, dict) else None
            allowed = {
                1: {"schema_version", "experiment_id", "job", "receipt_sha256", "publication", "result_output"},
                2: {"schema_version", "experiment_id", "job", "lease", "receipt_sha256", "publication", "result_output"},
            }
            if not isinstance(document, dict) or schema_version not in allowed or set(document) != allowed[schema_version] or self._canonical_json(document) != raw:
                raise ValueError("publication handoff is malformed")
            job_raw = document.get("job")
            if not isinstance(job_raw, dict) or not isinstance(document.get("publication"), dict) or type(document.get("receipt_sha256")) is not str:
                raise ValueError("publication handoff is malformed")
            job = _parse(job_raw)
            if document["experiment_id"] != job.experiment_id or path.name != job.experiment_id + ".json":
                raise ValueError("publication handoff has the wrong experiment identity")
            result = Path(str(document["result_output"]))
            expected_prefix = self.output_root / "jobs" / job.experiment_id
            try:
                result.relative_to(expected_prefix)
            except ValueError as error:
                raise ValueError("publication handoff result escapes its workspace") from error
            if result.is_symlink() or not result.is_file() or sha256_file(result) != document["receipt_sha256"]:
                raise ValueError("publication handoff result bytes are missing or tampered")
            bind_checkpoint_publication(job, document["receipt_sha256"], document["publication"])
            lease_document = document.get("lease")
            if schema_version == 1:
                # Legacy durable handoffs prove a terminal result but lack the
                # original lease authority.  They may only fail closed until an
                # operator reconciles them; they must never launch training.
                lease = type("PendingPublicationLease", (), {"experiment_id": job.experiment_id, "job": job})()
            else:
                if (
                    not isinstance(lease_document, dict)
                    or set(lease_document) != {"lease_id", "worker_id", "capability"}
                    or not all(type(lease_document[key]) is str and lease_document[key] for key in ("lease_id", "worker_id"))
                    or lease_document["capability"] != "training"
                ):
                    raise ValueError("publication handoff lease identity is malformed")
                lease = type("PendingPublicationLease", (), {
                    "lease_id": lease_document["lease_id"], "experiment_id": job.experiment_id,
                    "worker_id": lease_document["worker_id"], "capability": lease_document["capability"], "job": job,
                })()
            entries.append((lease, document["receipt_sha256"], document["publication"]))
        return tuple(entries)

    def clear_pending_publication(self, experiment_id: str) -> None:
        path = self._handoff_path(experiment_id)
        if path.exists():
            if path.is_symlink() or not path.is_file():
                raise ValueError("publication handoff is unsafe")
            path.unlink()

    @staticmethod
    def _terminal_result_receipt_path(result_output: Path) -> Path:
        # Legacy results live at ``workspace/output/result.json`` while sweep
        # results live under ``workspace/output/sweep/<job>/``.  Find the one
        # mounted output root rather than assuming a fixed parent depth.
        for ancestor in result_output.parents:
            if ancestor.name == "output":
                return ancestor.parent / "private" / "terminal-result.sha256"
        raise ValueError("terminal result escapes its workspace output root")

    def _existing_terminal_publication(self, job: ExperimentJob, result_output: Path) -> dict[str, object] | None:
        """Reuse only an already sealed same-job terminal result, never a partial."""
        receipt_path = self._terminal_result_receipt_path(result_output)
        if not receipt_path.exists():
            return None
        if receipt_path.is_symlink() or not receipt_path.is_file() or result_output.is_symlink() or not result_output.is_file():
            raise ValueError("terminal result receipt is unsafe")
        receipt = receipt_path.read_text(encoding="ascii").strip()
        if len(receipt) != 64 or sha256_file(result_output) != receipt:
            raise ValueError("terminal result receipt does not bind result bytes")
        from lehome_train.groot.experiment_runtime_request import publication_from_result
        publication = publication_from_result(job, result_output)
        if publication["receipt_sha256"] != receipt:
            raise ValueError("terminal result publication receipt mismatch")
        return publication

    def _seal_terminal_result(self, result_output: Path, receipt: str) -> None:
        receipt_path = self._terminal_result_receipt_path(result_output)
        if receipt_path.exists() or receipt_path.is_symlink():
            if receipt_path.is_symlink() or not receipt_path.is_file() or receipt_path.read_text(encoding="ascii").strip() != receipt:
                raise ValueError("terminal result receipt conflicts with existing result")
            return
        receipt_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if receipt_path.parent.is_symlink() or not receipt_path.parent.is_dir():
            raise ValueError("terminal result receipt directory is unsafe")
        descriptor = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="ascii") as handle:
                handle.write(receipt + "\n"); handle.flush(); os.fsync(handle.fileno())
        except BaseException:
            receipt_path.unlink(missing_ok=True)
            raise

    def run(
        self,
        job: ExperimentJob,
        *,
        cancellation: threading.Event | None = None,
        parent_publication: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Execute the guest-controller request set, never the Python runtime directly."""
        from lehome_train.groot.experiment_runtime_request import (
            HuggingFacePromotedParentHub,
            HuggingFaceRequestSetHydrator,
            build_sweep_runtime_request,
            build_sweep_train_overlay,
            build_runtime_environment,
            copy_request_set_to_workspace,
            hydrate_promoted_parent,
            materialize_request_set,
            publication_from_result,
        )
        if cancellation is not None and cancellation.is_set():
            raise RuntimeError("training cancelled before request-set hydration")
        if not self.cache_root.is_absolute() or not self.output_root.is_absolute() or self.output_root.is_symlink():
            raise ValueError("production runtime roots are unsafe")
        self.output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        hydrator = self.hydrator or HuggingFaceRequestSetHydrator(self.hf_token_file)
        request_set = materialize_request_set(job, self.cache_root, hydrator=hydrator)
        workspace_target = self.output_root / "jobs" / job.experiment_id
        reusing_workspace = workspace_target.exists()
        if reusing_workspace:
            self._validate_workspace_inputs(workspace_target, request_set)
            workspace = workspace_target
        else:
            workspace = copy_request_set_to_workspace(request_set, workspace_target)
        if request_set.compatibility_profile_sha256 is None:
            # Preserve the pre-sweep production-request compatibility path.
            # New sweep manifests are v2 and take the stricter branch below.
            runtime_env, result_output = build_runtime_environment(
                job, request_set, workspace, hf_token_file=self.hf_token_file,
            )
        else:
            if job.training.target_step > 500 and not isinstance(parent_publication, Mapping):
                raise ValueError("promoted experiment lease lacks an authenticated parent publication")
            parent = hydrate_promoted_parent(
                job,
                publication={} if parent_publication is None else parent_publication,
                cache_root=workspace / "cache" / "promoted-parents",
                hub=self.parent_hub or HuggingFacePromotedParentHub(self.hf_token_file),
            )
            train_base = workspace / request_set.environment["LEHOME_RUNTIME_TRAIN_REQUEST"]
            launch_base = workspace / "prepared" / "config" / "launch.json"
            experiment_base = workspace / "prepared" / "config" / "experiment.json"
            overlay = workspace / "prepared" / "sweep-train-overlay.json"
            overlay = build_sweep_train_overlay(
                job, workspace=workspace, base_train_request=train_base,
                compatibility_profile_sha256=request_set.compatibility_profile_sha256,
                parent_publication=parent_publication,
            )
            dynamic = build_sweep_runtime_request(
                job, workspace=workspace, base_train_request=train_base,
                base_launch_config=launch_base, base_experiment_config=experiment_base,
                overlay=overlay, promoted_parent=parent,
            )
            runtime_env, result_output = build_runtime_environment(
                job, request_set, workspace, hf_token_file=self.hf_token_file,
                sweep_overlay=overlay, promoted_parent=parent,
                sweep_runtime_request=dynamic,
            )
        existing = self._existing_terminal_publication(job, result_output)
        if existing is not None:
            self._result_outputs[job.experiment_id] = result_output
            return {"terminal_receipt_sha256": existing["receipt_sha256"], "publication": existing}
        if cancellation is not None and cancellation.is_set():
            raise RuntimeError("training cancelled before guest controller")
        env = dict(os.environ, LEHOME_RUNTIME_ENV=str(runtime_env), LEHOME_EXPERIMENT_RESUME="1" if reusing_workspace else "0")
        try:
            if cancellation is not None and self.process_runner is subprocess.run:
                from lehome_train.groot.experiment_worker import run_subprocess_cancellable
                run_subprocess_cancellable([str(self.training_script)], env=env, cancellation=cancellation)
            else:
                self.process_runner([str(self.training_script)], env=env, check=True)
        except BaseException as error:
            if cancellation is not None and cancellation.is_set():
                self._write_preemption_receipt(job, result_output)
                raise PreemptionRequested("training cancelled with exact preemption receipt") from error
            raise
        if cancellation is not None and cancellation.is_set():
            raise RuntimeError("training cancelled before publication readback")
        publication = publication_from_result(job, result_output)
        self._seal_terminal_result(result_output, publication["receipt_sha256"])
        self._result_outputs[job.experiment_id] = result_output
        return {"terminal_receipt_sha256": publication["receipt_sha256"], "publication": publication}

    def _write_preemption_receipt(self, job: ExperimentJob, result_output: Path) -> None:
        """Record only a verified terminal output; partial bytes are never resumable evidence."""
        root = result_output.parent.parent / "private" / "preemptions"
        if root.is_symlink():
            raise ValueError("preemption receipt root is unsafe")
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        receipt: dict[str, object] = {"schema_version": 1, "experiment_id": job.experiment_id, "result_output": str(result_output), "terminal_receipt_sha256": None}
        if result_output.exists() and not result_output.is_symlink() and result_output.is_file():
            try:
                from lehome_train.groot.experiment_runtime_request import publication_from_result
                receipt["terminal_receipt_sha256"] = publication_from_result(job, result_output)["receipt_sha256"]
            except ValueError:
                # No terminal read-back exists; the guest is allowed to resume
                # from its own checkpoint protocol, never from this receipt.
                pass
        target = root / (str(os.getpid()) + "-" + str(threading.get_ident()) + ".json")
        payload = self._canonical_json(receipt)
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())

def main(argv: list[str] | None = None, *, controller_factory: Callable[..., Any] = HttpControllerClient, runtime_factory: Callable[..., Any] = ProductionRuntimeExperimentRunner) -> int:
    parser = argparse.ArgumentParser()
    for name in ("controller-url", "controller-ca-file", "worker-id", "manifest-set-sha256", "cache-root", "output-root", "controller-token-file", "hf-token-file", "deployment-gate", "deployment-gate-sha256", "training-image-manifest"): parser.add_argument("--" + name, required=True)
    parser.add_argument("--max-jobs", type=int)
    args = parser.parse_args(argv)
    cache, output, controller_token, hf_token, controller_ca, deployment_gate, training_image_manifest = Path(args.cache_root), Path(args.output_root), Path(args.controller_token_file), Path(args.hf_token_file), Path(args.controller_ca_file), Path(args.deployment_gate), Path(args.training_image_manifest)
    if any(not path.is_absolute() for path in (cache, output, controller_token, hf_token, controller_ca, deployment_gate, training_image_manifest)): raise ValueError("worker roots and credential files must be absolute")
    gate = load_deployment_gate(deployment_gate, args.deployment_gate_sha256)
    image = load_training_image_manifest(training_image_manifest)
    return ExperimentWorker(
        controller_factory(args.controller_url, controller_token, args.manifest_set_sha256, controller_ca),
        worker_id=args.worker_id,
        runner=runtime_factory(cache, output, hf_token),
        identity_preflight=lambda job: bind_training_job_identity(job, gate, image),
    ).run(max_jobs=args.max_jobs)
if __name__ == "__main__": raise SystemExit(main())
