#!/usr/bin/env python3
"""Fail-closed, resumable host journal for one simple-curriculum collection.

This controller deliberately has no provider API client.  It only sequences
already-reviewed appliance commands and persists immutable receipts, so a
preemption can resume the same matrices and ledgers without inventing work.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import tempfile
from typing import Any, Mapping, Protocol, Sequence


STAGES = (
    "calibration-matrix", "calibration-head", "first-100-gate",
    "calibration-tail", "calibration-report", "curriculum-matrix",
    "curriculum-a", "curriculum-b", "fresh-report", "replay-matrix",
    "success-replay", "final-publication", "gpu-stop",
)
COMMAND_VERSION = "simple-curriculum-one-vm-v1"
_HEX = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{7,127}$")
_DEFAULT_IDS = frozenset({"lehome-rft-70-30-v1", "round-3", "campaign-12k-round-3"})
_CLOUD_TOKENS = frozenset({"nebius", "vast", "aws", "gcloud", "terraform", "packer", "create", "start", "delete"})
_ORIGINAL_12K = {
    "policy_repo": "ryanjin333/lehome-groot-n17-models",
    "policy_revision": "30ac1a84da67b099e115ad147bcd61e9d60046d3",
    "policy_step": 12000,
    "policy_artifact_sha256": "3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06",
    "simulator_device": "cpu", "cloth_device": "cpu", "policy_device": "cuda:0", "worker_count": 4,
}
_RUNTIME_KEYS = frozenset(_ORIGINAL_12K) | frozenset({"rollout_image", "trainer_image"})
_TRUSTED_GPU_STOP = "/usr/local/libexec/lehome-stop-gpu"
_STAGE_ARTIFACTS = {
    "calibration-matrix": frozenset({"matrix"}),
    "calibration-head": frozenset({"matrix", "manifest", "ledger"}),
    "first-100-gate": frozenset({"report", "gate_receipt"}),
    "calibration-tail": frozenset({"matrix", "manifest", "ledger"}),
    "calibration-report": frozenset({"report"}),
    "curriculum-matrix": frozenset({"matrix"}),
    "curriculum-a": frozenset({"matrix", "manifest", "ledger"}),
    "curriculum-b": frozenset({"matrix", "manifest", "ledger"}),
    "fresh-report": frozenset({"report"}),
    "replay-matrix": frozenset({"matrix"}),
    "success-replay": frozenset({"matrix", "ledger"}),
    "final-publication": frozenset({"publication_receipt", "publication_readback"}),
}


class ReceiptMismatchError(RuntimeError):
    """An existing journal entry is not exactly the receipt this run owns."""


class StopHookError(RuntimeError):
    """The configured external stop hook failed after a terminal result."""


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _digest(value: object) -> str:
    return sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("receipt is missing or unsafe")
    return sha256(path.read_bytes()).hexdigest()


def _safe_directory(path: Path, *, must_exist: bool) -> None:
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("path must be an absolute non-symlink directory")
    for ancestor in (path, *path.parents):
        if ancestor.exists() and ancestor.is_symlink():
            raise ValueError("path has a symlink ancestor")
    if must_exist:
        if not path.is_dir() or not stat.S_ISDIR(path.stat().st_mode):
            raise ValueError("directory is missing or unsafe")
    elif path.exists() and (not path.is_dir() or not stat.S_ISDIR(path.stat().st_mode)):
        raise ValueError("campaign root is not a directory")


def _tree_sha(root: Path) -> str:
    digest = sha256()
    for relative in ("source/lehome", "trainer/src", "scripts", "rollout_appliance"):
        tree = root / relative
        if tree.is_symlink() or not tree.is_dir():
            raise ValueError("reviewed code root is incomplete")
        for item in sorted(tree.rglob("*")):
            if "__pycache__" in item.parts or item.suffix == ".pyc" or item.name == ".DS_Store":
                continue
            if item.is_symlink() or (item.exists() and not (item.is_file() or item.is_dir())):
                raise ValueError("reviewed code root contains unsafe entry")
            if item.is_file():
                digest.update(item.relative_to(root).as_posix().encode("utf-8") + b"\0")
                digest.update(item.read_bytes())
    return digest.hexdigest()


def _git_revision(root: Path) -> str:
    result = subprocess.run(
        ("git", "-C", str(root), "rev-parse", "HEAD"), text=True, capture_output=True, check=False,
    )
    revision = result.stdout.strip()
    if result.returncode != 0 or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        # Unit-test runners can construct a reviewed tree without a repository;
        # the command-line paid boundary below rejects this value.
        return "offline-" + _tree_sha(root)[:32]
    return revision


@dataclass(frozen=True, slots=True)
class CollectionConfig:
    campaign_root: Path
    host_code_root: Path
    run_id: str
    round_id: str
    max_wall_seconds: float
    max_spend_usd: float
    paid: bool
    gpu_stop_command: str | None
    runtime_identity: Mapping[str, object]

    def validate(self, *, require_git: bool = False) -> None:
        _safe_directory(self.host_code_root, must_exist=True)
        _safe_directory(self.campaign_root, must_exist=False)
        for identifier in (self.run_id, self.round_id):
            if identifier in _DEFAULT_IDS or _SAFE_ID.fullmatch(identifier) is None:
                raise ValueError("fresh caller-supplied run and round IDs are required")
        for value, label, ceiling in ((self.max_wall_seconds, "max wall time", 86_400.0), (self.max_spend_usd, "max spend", 100.0)):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0 or value >= ceiling:
                raise ValueError(f"{label} must be finite, positive, and bounded")
        if self.paid and self.gpu_stop_command != _TRUSTED_GPU_STOP:
            raise ValueError("paid collection requires the fixed trusted GPU stop hook")
        if not isinstance(self.runtime_identity, Mapping) or not self.runtime_identity:
            raise ValueError("pinned runtime identity is required")
        if set(self.runtime_identity) != _RUNTIME_KEYS or any(re.search(r"(?:token|secret|password|credential|api[_-]?key)", str(key), re.I) for key in self.runtime_identity):
            raise ValueError("runtime identity schema is exact and excludes secrets")
        for key in ("rollout_image", "trainer_image"):
            value = self.runtime_identity.get(key)
            if not isinstance(value, str) or re.search(r"@sha256:[0-9a-f]{64}$", value) is None:
                raise ValueError("runtime images must be digest pinned")
        if any(self.runtime_identity.get(key) != value for key, value in _ORIGINAL_12K.items()):
            raise ValueError("runtime tuple is not the pinned original-12K four-worker CPU-cloth contract")
        if require_git and not re.fullmatch(r"[0-9a-f]{40}", _git_revision(self.host_code_root)):
            raise ValueError("LEHOME_HOST_CODE_ROOT must be a checked-out reviewed Git root")

    def identity(self) -> dict[str, object]:
        return {
            "campaign_root": str(_canonical_root(self)), "run_id": self.run_id, "round_id": self.round_id,
            "host_code_root": str(self.host_code_root), "code_revision": _git_revision(self.host_code_root),
            "code_tree_sha256": _tree_sha(self.host_code_root),
            "runtime_identity": dict(self.runtime_identity),
            "max_wall_seconds": self.max_wall_seconds, "max_spend_usd": self.max_spend_usd,
        }


class Runner(Protocol):
    def run(self, stage: str, **kwargs: object) -> Mapping[str, object]: ...
    def stop_gpu(self, command: str) -> None: ...


class CommandRunner:
    """Runs only explicitly configured local/appliance commands, never cloud CLIs."""

    def run(self, stage: str, **kwargs: object) -> Mapping[str, object]:
        # No command is accepted from the environment. A paid deployment must
        # install a reviewed adapter with a fixed argv contract; until then,
        # failing here is safer than turning this host process into a shell.
        if any(name.startswith("LEHOME_ORCHESTRATOR_") and name.endswith("_COMMAND") for name in os.environ):
            raise ValueError("stage commands must use a fixed reviewed adapter, never environment commands")
        raise RuntimeError(f"no fixed reviewed adapter is installed for {stage}")

    def stop_gpu(self, command: str) -> None:
        if command != _TRUSTED_GPU_STOP:
            raise ValueError("GPU stop hook must be the fixed trusted executable")
        subprocess.run((_TRUSTED_GPU_STOP,), check=True)


def partition_rows(rows: Sequence[Mapping[str, object]], *, parent_matrix_sha256: str, partition_id: str, start: int, end: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Slice frozen logical rows verbatim; partition metadata is external."""
    if _HEX.fullmatch(parent_matrix_sha256) is None or not 0 <= start < end <= len(rows):
        raise ValueError("partition bounds or parent matrix hash are invalid")
    partition = [dict(row) for row in rows[start:end]]
    manifest = {
        "schema_version": 1, "kind": "lehome_simple_curriculum_partition_manifest_v1",
        "partition_id": partition_id, "parent_matrix_sha256": parent_matrix_sha256,
        "row_start": start, "row_end": end, "row_count": len(partition),
        "partition_sha256": _digest(partition),
    }
    return partition, manifest


def materialize_partition(*, parent_matrix: Path, parent_matrix_sha256: str, output_directory: Path,
                          partition_id: str, start: int, end: int) -> tuple[Path, Path, dict[str, object]]:
    """Atomically materialize/verify one immutable physical matrix slice."""
    parent_matrix, output_directory = Path(parent_matrix), Path(output_directory)
    if not parent_matrix.is_absolute() or parent_matrix.is_symlink() or not parent_matrix.is_file():
        raise ValueError("logical matrix is missing or unsafe")
    actual_hash = _file_sha(parent_matrix)
    if actual_hash != parent_matrix_sha256 or _HEX.fullmatch(parent_matrix_sha256) is None:
        raise ReceiptMismatchError("logical matrix hash mismatch; refusing to rebuild")
    try:
        rows = json.loads(parent_matrix.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("logical matrix is malformed") from error
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("logical matrix rows are invalid")
    partition, manifest = partition_rows(rows, parent_matrix_sha256=actual_hash, partition_id=partition_id, start=start, end=end)
    _safe_directory(output_directory, must_exist=False); output_directory.mkdir(parents=True, exist_ok=True)
    matrix_path = output_directory / f"{partition_id}.json"
    manifest_path = output_directory / f"{partition_id}.manifest.json"
    expected_matrix, expected_manifest = _canonical(partition), _canonical(manifest)
    for path, payload in ((matrix_path, expected_matrix), (manifest_path, expected_manifest)):
        if path.exists():
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise ReceiptMismatchError("immutable partition collision")
        else:
            _write_absent(path, payload)
    return matrix_path, manifest_path, manifest


def _write_absent(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise ReceiptMismatchError("immutable receipt collision")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload); stream.flush(); os.fchmod(stream.fileno(), 0o444); os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _replace_durable(path: Path, payload: bytes) -> None:
    if path.is_symlink() or not path.parent.is_dir(): raise ReceiptMismatchError("durable state path is unsafe")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload); stream.flush(); os.fchmod(stream.fileno(), 0o600); os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally: temporary.unlink(missing_ok=True)


def _canonical_root(config: CollectionConfig) -> Path:
    return config.campaign_root.resolve(strict=False)


def _authenticated_output(stage: str, output: Mapping[str, object], *, config: CollectionConfig) -> dict[str, object]:
    """Accept only a bounded stage result and hash every referenced byte now."""
    if stage == "gpu-stop":
        allowed = {"terminal_outcome", "stop_status", "stop_error_type"}
        if set(output) - allowed or output.get("stop_status") not in {"not_required", "pending", "succeeded", "failed"}:
            raise ReceiptMismatchError("GPU stop state is invalid")
        if not isinstance(output.get("terminal_outcome"), str): raise ReceiptMismatchError("GPU stop state is invalid")
        if "stop_error_type" in output and not isinstance(output["stop_error_type"], str): raise ReceiptMismatchError("GPU stop state is invalid")
        return dict(output)
    required = _STAGE_ARTIFACTS.get(stage, frozenset())
    allowed = {"artifacts"}
    if stage == "first-100-gate": allowed.add("decision")
    if stage == "success-replay": allowed.add("result")
    if set(output) - allowed or "artifacts" not in output or not isinstance(output["artifacts"], Mapping):
        raise ReceiptMismatchError("stage artifact output schema is invalid")
    artifacts = output["artifacts"]
    if set(artifacts) != required:
        raise ReceiptMismatchError("stage output artifacts are missing or unexpected")
    root = _canonical_root(config)
    checked: dict[str, dict[str, str]] = {}
    for name, descriptor in artifacts.items():
        if not isinstance(descriptor, Mapping) or set(descriptor) != {"path", "sha256"}:
            raise ReceiptMismatchError("stage artifact descriptor is invalid")
        path, claimed = descriptor["path"], descriptor["sha256"]
        if not isinstance(path, str) or not isinstance(claimed, str) or _HEX.fullmatch(claimed) is None:
            raise ReceiptMismatchError("stage artifact descriptor is invalid")
        candidate = (root / path).resolve(strict=False)
        if not candidate.is_relative_to(root) or candidate.is_symlink() or not candidate.is_file():
            raise ReceiptMismatchError("stage artifact is missing or unsafe")
        actual = _file_sha(candidate)
        if actual != claimed:
            raise ReceiptMismatchError("stage artifact hash mismatch")
        checked[str(name)] = {"path": candidate.relative_to(root).as_posix(), "sha256": actual}
    result: dict[str, object] = {"artifacts": checked}
    if stage == "first-100-gate":
        decision = output.get("decision")
        if decision not in {"continue", "fidelity_stop", "infrastructure_stop", "insufficient_source_stop"}:
            raise ReceiptMismatchError("first-100 gate receipt has no valid decision")
        result["decision"] = decision
    if stage == "success-replay":
        replay = output.get("result")
        if replay not in {"complete", "replay_shortage"}:
            raise ReceiptMismatchError("replay result is invalid")
        result["result"] = replay
    return result


def _verify_authenticated_output(stage: str, output: Mapping[str, object], *, config: CollectionConfig) -> None:
    # Re-run validation against the actual bytes on every restart.  The
    # descriptor itself is part of the receipt hash; this catches deletion,
    # rewrites, root swaps, and changed ledgers without trusting old JSON.
    if _authenticated_output(stage, output, config=config) != output:
        raise ReceiptMismatchError("stage output authentication changed")


class StageJournal:
    def __init__(self, config: CollectionConfig) -> None:
        self.config, self.directory = config, config.campaign_root / "stage-receipts"
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, stage: str) -> Path:
        if stage not in STAGES: raise ValueError("unknown stage")
        return self.directory / f"{stage}.json"

    @property
    def stop_state_path(self) -> Path: return self.directory / "gpu-stop-state.json"

    def stop_state(self, predecessor: str, outcome: str) -> dict[str, object] | None:
        path = self.stop_state_path
        if not path.exists(): return None
        try: state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error: raise ReceiptMismatchError("GPU stop state is malformed") from error
        expected = {"schema_version": 1, "kind": "lehome_simple_curriculum_gpu_stop_state_v1", "campaign_root": str(_canonical_root(self.config)), "predecessor_receipt_sha256": predecessor, "terminal_outcome": outcome}
        if (not isinstance(state, dict) or set(state) != set(expected) | {"status"}
                or any(state.get(key) != value for key, value in expected.items())
                or state.get("status") not in {"pending", "succeeded", "failed"}):
            raise ReceiptMismatchError("GPU stop state does not bind this terminal collection")
        return state

    def write_stop_state(self, predecessor: str, outcome: str, status: str) -> None:
        if status not in {"pending", "succeeded", "failed"}: raise ValueError("GPU stop status is invalid")
        _replace_durable(self.stop_state_path, _canonical({"schema_version": 1, "kind": "lehome_simple_curriculum_gpu_stop_state_v1", "campaign_root": str(_canonical_root(self.config)), "predecessor_receipt_sha256": predecessor, "terminal_outcome": outcome, "status": status}))

    def _read(self, stage: str, predecessor: str | None, inputs: Mapping[str, object]) -> dict[str, object] | None:
        path = self.path(stage)
        if not path.exists(): return None
        try: receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error: raise ReceiptMismatchError("stage receipt is malformed") from error
        if not isinstance(receipt, dict): raise ReceiptMismatchError("stage receipt is malformed")
        body = dict(receipt); stored = body.pop("receipt_sha256", None)
        required = {"schema_version", "kind", "stage", "predecessor_receipt_sha256", "command_version", "input_hashes", "output_hashes", "runtime_identity", "output", "receipt_sha256"}
        if set(receipt) != required or stored != _digest(body): raise ReceiptMismatchError("stage receipt checksum mismatch")
        if (receipt["schema_version"] != 1 or receipt["kind"] != "lehome_simple_curriculum_stage_receipt_v1"
                or receipt["stage"] != stage or receipt["predecessor_receipt_sha256"] != predecessor
                or receipt["command_version"] != COMMAND_VERSION or receipt["runtime_identity"] != self.config.identity()):
            raise ReceiptMismatchError("stage receipt does not bind this collection")
        if not isinstance(receipt["input_hashes"], dict) or not isinstance(receipt["output_hashes"], dict) or not isinstance(receipt["output"], dict):
            raise ReceiptMismatchError("stage receipt fields are invalid")
        if receipt["input_hashes"] != {
            "collection_identity": _digest(self.config.identity()), "predecessor": predecessor or "",
            "stage_inputs": _digest(dict(inputs)),
        }:
            raise ReceiptMismatchError("stage receipt input binding mismatch")
        _verify_authenticated_output(stage, receipt["output"], config=self.config)
        if receipt["output_hashes"] != {"output": _digest(receipt["output"])}:
            raise ReceiptMismatchError("stage receipt output hash mismatch")
        return receipt

    def complete(self, stage: str, predecessor: str | None, output: Mapping[str, object], *, inputs: Mapping[str, object]) -> dict[str, object]:
        existing = self._read(stage, predecessor, inputs)
        if existing is not None: return existing
        safe_output = _authenticated_output(stage, output, config=self.config)
        body: dict[str, object] = {
            "schema_version": 1, "kind": "lehome_simple_curriculum_stage_receipt_v1", "stage": stage,
            "predecessor_receipt_sha256": predecessor, "command_version": COMMAND_VERSION,
            "input_hashes": {"collection_identity": _digest(self.config.identity()), "predecessor": predecessor or "", "stage_inputs": _digest(dict(inputs))},
            "output_hashes": {"output": _digest(safe_output)}, "runtime_identity": self.config.identity(), "output": safe_output,
        }
        receipt = {**body, "receipt_sha256": _digest(body)}
        _write_absent(self.path(stage), _canonical(receipt))
        return receipt


def _stage(journal: StageJournal, runner: Runner, stage: str, predecessor: str | None, **kwargs: object) -> tuple[dict[str, object], str]:
    existing = journal._read(stage, predecessor, kwargs)
    if existing is not None:
        return existing, str(existing["receipt_sha256"])
    output = runner.run(stage, **kwargs)
    if not isinstance(output, Mapping): raise ValueError("stage runner output is invalid")
    receipt = journal.complete(stage, predecessor, output, inputs=kwargs)
    return receipt, str(receipt["receipt_sha256"])


def _partition_from_stage(config: CollectionConfig, stage_output: Mapping[str, object], *, partition_id: str,
                          start: int, end: int) -> dict[str, object]:
    """Use a written logical matrix when the producer exposes its immutable bytes."""
    artifacts = stage_output.get("artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(artifacts.get("matrix"), Mapping):
        raise ReceiptMismatchError("logical matrix stage did not bind a matrix artifact")
    path, digest = artifacts["matrix"].get("path"), artifacts["matrix"].get("sha256")
    if not isinstance(path, str) or not isinstance(digest, str): raise ReceiptMismatchError("logical matrix stage did not bind a path and hash")
    matrix, manifest, details = materialize_partition(
        parent_matrix=_canonical_root(config) / path, parent_matrix_sha256=digest,
        output_directory=config.campaign_root / "partitions", partition_id=partition_id, start=start, end=end,
    )
    root = _canonical_root(config)
    return {"partition_matrix": matrix.relative_to(root).as_posix(), "partition_manifest": manifest.relative_to(root).as_posix(), "partition_sha256": details["partition_sha256"]}


def run_collection(config: CollectionConfig, *, runner: Runner) -> str:
    """Run/recover the only permitted state machine; returns its data outcome."""
    config.validate()
    journal = StageJournal(config); predecessor: str | None = None
    calibration_matrix, predecessor = _stage(journal, runner, "calibration-matrix", predecessor)
    calibration_partition = _partition_from_stage(config, calibration_matrix["output"], partition_id="calibration-head", start=0, end=100)
    _, predecessor = _stage(journal, runner, "calibration-head", predecessor, partition_id="calibration-head", row_start=0, row_end=100, target=100, lease_budget=150, **calibration_partition)
    gate, predecessor = _stage(journal, runner, "first-100-gate", predecessor)
    decision = gate["output"].get("decision")
    if decision not in {"continue", "fidelity_stop", "infrastructure_stop", "insufficient_source_stop"}:
        raise ReceiptMismatchError("first-100 gate receipt has no valid decision")
    if decision == "continue":
        calibration_tail = _partition_from_stage(config, calibration_matrix["output"], partition_id="calibration-tail", start=100, end=400)
        _, predecessor = _stage(journal, runner, "calibration-tail", predecessor, partition_id="calibration-tail", row_start=100, row_end=400, target=300, lease_budget=400, **calibration_tail)
        _, predecessor = _stage(journal, runner, "calibration-report", predecessor)
        curriculum_matrix, predecessor = _stage(journal, runner, "curriculum-matrix", predecessor)
        curriculum_a = _partition_from_stage(config, curriculum_matrix["output"], partition_id="curriculum-a", start=0, end=300)
        _, predecessor = _stage(journal, runner, "curriculum-a", predecessor, partition_id="curriculum-a", row_start=0, row_end=300, target=300, lease_budget=400, **curriculum_a)
        curriculum_b = _partition_from_stage(config, curriculum_matrix["output"], partition_id="curriculum-b", start=300, end=600)
        _, predecessor = _stage(journal, runner, "curriculum-b", predecessor, partition_id="curriculum-b", row_start=300, row_end=600, target=300, lease_budget=400, **curriculum_b)
        for stage in ("fresh-report", "replay-matrix"):
            _, predecessor = _stage(journal, runner, stage, predecessor)
        replay, predecessor = _stage(journal, runner, "success-replay", predecessor)
        outcome = replay["output"].get("result", "complete")
        if outcome not in {"complete", "replay_shortage"}: raise ReceiptMismatchError("replay result is invalid")
    else:
        outcome = str(decision)
    _, predecessor = _stage(journal, runner, "final-publication", predecessor, terminal_outcome=outcome)
    stop_inputs = {"terminal_outcome": outcome}
    existing = journal._read("gpu-stop", predecessor, stop_inputs)
    state = journal.stop_state(predecessor, outcome)
    if state is not None and state["status"] != "succeeded":
        # Never retry after a crash/failure: the process could have died after
        # sending the external stop but before recording it. Retrying would
        # break exactly-once; a separate provider audit is required instead.
        return "infrastructure_stop_failure"
    if existing is not None:
        if not config.paid and existing["output"].get("stop_status") == "not_required":
            return str(outcome)
        if state is None:
            return "infrastructure_stop_failure"
        return str(outcome)
    if not config.paid:
        journal.complete("gpu-stop", predecessor, {"terminal_outcome": outcome, "stop_status": "not_required"}, inputs=stop_inputs)
        return str(outcome)
    journal.write_stop_state(predecessor, outcome, "pending")
    try:
        assert config.gpu_stop_command is not None
        runner.stop_gpu(config.gpu_stop_command)
    except Exception as error:
        journal.write_stop_state(predecessor, outcome, "failed")
        journal.complete("gpu-stop", predecessor, {"terminal_outcome": outcome, "stop_status": "failed", "stop_error_type": type(error).__name__}, inputs=stop_inputs)
        return "infrastructure_stop_failure"
    journal.write_stop_state(predecessor, outcome, "succeeded")
    journal.complete("gpu-stop", predecessor, {"terminal_outcome": outcome, "stop_status": "succeeded"}, inputs=stop_inputs)
    return str(outcome)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--host-code-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True); parser.add_argument("--round-id", required=True)
    parser.add_argument("--max-wall-seconds", type=float, required=True); parser.add_argument("--max-spend-usd", type=float, required=True)
    parser.add_argument("--paid", action="store_true"); parser.add_argument("--gpu-stop-command")
    parser.add_argument("--runtime-identity-json", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        identity = json.loads(args.runtime_identity_json.read_text(encoding="utf-8"))
        config = CollectionConfig(args.campaign_root, args.host_code_root, args.run_id, args.round_id, args.max_wall_seconds, args.max_spend_usd, args.paid, args.gpu_stop_command, identity)
        config.validate(require_git=True)
        print(run_collection(config, runner=CommandRunner()))
    except (OSError, ValueError, ReceiptMismatchError, StopHookError, RuntimeError) as error:
        _parser().error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
