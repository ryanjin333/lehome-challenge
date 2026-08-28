#!/usr/bin/env python3
"""Fail-closed, resumable host journal for one simple-curriculum collection.

This controller deliberately has no provider API client.  It only sequences
already-reviewed appliance commands and persists immutable receipts, so a
preemption can resume the same matrices and ledgers without inventing work.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
import subprocess
import tempfile
import threading
import time
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
_EXACT_ROLLOUT_INSTANCE_ID = "computeinstance-u00t6xfqhadrcmssa2"
_STAGE_ARTIFACTS = {
    "calibration-matrix": frozenset({"matrix", "matrix_receipt"}),
    "calibration-head": frozenset({"matrix", "manifest", "ledger"}),
    "first-100-gate": frozenset({"report", "gate_receipt"}),
    "calibration-tail": frozenset({"matrix", "manifest", "ledger"}),
    "calibration-report": frozenset({"report"}),
    "curriculum-matrix": frozenset({"matrix", "matrix_receipt"}),
    "curriculum-a": frozenset({"matrix", "manifest", "ledger"}),
    "curriculum-b": frozenset({"matrix", "manifest", "ledger"}),
    "fresh-report": frozenset({"report", "matrix", "terminal_artifact_manifest"}),
    "replay-matrix": frozenset({"matrix", "matrix_receipt"}),
    "success-replay": frozenset({"matrix", "ledger", "readback_seal"}),
    "final-publication": frozenset({"publication_receipt", "publication_readback"}),
}


class ReceiptMismatchError(RuntimeError):
    """An existing journal entry is not exactly the receipt this run owns."""


class StopHookError(RuntimeError):
    """The configured external stop hook failed after a terminal result."""


class BudgetLimitError(ReceiptMismatchError):
    """A portable paid-run walltime or spend limit was reached."""


class _PipeTail:
    """Continuously drain one child pipe while retaining only diagnostics."""

    _LIMIT = 64 * 1024

    def __init__(self, stream: object) -> None:
        self._stream = stream
        self._chunks: list[str] = []
        self._size = 0
        self._lock = threading.Lock()

    def drain(self) -> None:
        reader = getattr(self._stream, "read", None)
        if not callable(reader):
            return
        while True:
            chunk = reader(4096)
            if not chunk:
                return
            if not isinstance(chunk, str):
                chunk = str(chunk)
            with self._lock:
                self._chunks.append(chunk)
                self._size += len(chunk)
                while self._size > self._LIMIT and self._chunks:
                    excess = self._size - self._LIMIT
                    first = self._chunks[0]
                    if len(first) <= excess:
                        self._chunks.pop(0)
                        self._size -= len(first)
                    else:
                        self._chunks[0] = first[excess:]
                        self._size -= excess

    def text(self) -> str:
        with self._lock:
            return "".join(self._chunks)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _digest(value: object) -> str:
    return sha256(_canonical(value)).hexdigest()


def _parse_utc(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ReceiptMismatchError(f"{label} is malformed")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ReceiptMismatchError(f"{label} is malformed") from error
    if parsed.tzinfo != UTC:
        raise ReceiptMismatchError(f"{label} is malformed")
    return parsed


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _file_sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("receipt is missing or unsafe")
    return sha256(path.read_bytes()).hexdigest()


def _descriptors(root: Path, **paths: Path) -> dict[str, dict[str, str]]:
    descriptors: dict[str, dict[str, str]] = {}
    for name, path in paths.items():
        if not path.is_absolute() or path.is_symlink() or not path.is_file() or not path.is_relative_to(root):
            raise ReceiptMismatchError("stage output path is missing or unsafe")
        descriptors[name] = {"path": path.relative_to(root).as_posix(), "sha256": _file_sha(path)}
    return descriptors


def _strict_json_object(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ReceiptMismatchError(f"{label} is missing or unsafe")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReceiptMismatchError(f"{label} is malformed") from error
    if not isinstance(payload, dict):
        raise ReceiptMismatchError(f"{label} is malformed")
    return payload


def _validate_matrix_receipt(matrix: Path, receipt: Path, *, expected_rows: int) -> None:
    if matrix.is_symlink() or not matrix.is_file():
        raise ReceiptMismatchError("logical matrix is missing or unsafe")
    try:
        rows = json.loads(matrix.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReceiptMismatchError("logical matrix is malformed") from error
    if not isinstance(rows, list) or len(rows) != expected_rows or not all(isinstance(row, Mapping) for row in rows):
        raise ReceiptMismatchError("logical matrix has an unexpected canonical shape")
    payload = _strict_json_object(receipt, label="logical matrix receipt")
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "lehome_simple_curriculum_matrix_receipt_v1"
        or payload.get("output_sha256") != _file_sha(matrix)
        or payload.get("output_bytes") != matrix.stat().st_size
        or not isinstance(payload.get("parameters"), Mapping)
    ):
        raise ReceiptMismatchError("logical matrix receipt does not bind matrix bytes")


def _partition_parent_sha(manifest: Path) -> str:
    payload = _strict_json_object(manifest, label="partition manifest")
    value = payload.get("parent_matrix_sha256")
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise ReceiptMismatchError("partition manifest parent matrix hash is invalid")
    return value


def _validate_partition_manifest(manifest: Path, *, matrix: Path, partition_id: str, inputs: Mapping[str, object]) -> None:
    payload = _strict_json_object(manifest, label="partition manifest")
    expected = {
        "schema_version": 1,
        "kind": "lehome_simple_curriculum_partition_manifest_v1",
        "partition_id": partition_id,
        "row_start": inputs.get("row_start"),
        "row_end": inputs.get("row_end"),
        "row_count": int(inputs.get("row_end", 0)) - int(inputs.get("row_start", 0)),
        "partition_sha256": _file_sha(matrix),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ReceiptMismatchError("partition manifest does not bind physical matrix")
    _partition_parent_sha(manifest)


def _partition_ledger_terminal_count(
    ledger: Path, *, matrix: Path, max_attempts: int, target: int,
    completion_metric: str = "terminal_outcomes",
) -> int:
    """Re-open the canonical ledger against its exact physical matrix."""
    if ledger.is_symlink() or not ledger.is_file():
        raise ReceiptMismatchError("partition ledger is missing or unsafe")
    try:
        rows = json.loads(matrix.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
            raise ValueError("physical matrix is malformed")
        # TaskLedger authenticates its immutable metadata, assignment order,
        # and its own canonical (ledger) attempt IDs.  Those IDs intentionally
        # differ from the logical row IDs, so comparing the two directly would
        # reject every legitimate production partition.
        from lehome.flywheel.task_ledger import TaskLedger

        task_ledger = TaskLedger(
            ledger, attempt_matrix=rows, max_attempts=max_attempts,
            target_accepted=target, completion_metric=completion_metric,
        )
        try:
            completed = (
                task_ledger.terminal_outcome_count()
                if completion_metric == "terminal_outcomes" else task_ledger.accepted_count()
            )
        finally:
            task_ledger.close()
    except (OSError, ValueError, json.JSONDecodeError, sqlite3.Error) as error:
        raise ReceiptMismatchError("partition ledger is malformed") from error
    if completed < 0 or completed > target:
        raise ReceiptMismatchError("partition ledger terminal count is invalid")
    return completed


def _validate_partition_ledger(
    ledger: Path, *, matrix: Path, max_attempts: int, target: int,
    completion_metric: str = "terminal_outcomes",
) -> None:
    completed = _partition_ledger_terminal_count(
        ledger, matrix=matrix, max_attempts=max_attempts, target=target,
        completion_metric=completion_metric,
    )
    if completed != target:
        raise ReceiptMismatchError("partition ledger terminal count is not exact")


def _write_immutable_json(path: Path, payload: Mapping[str, object] | Sequence[object]) -> None:
    encoded = _canonical(payload)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise ReceiptMismatchError("immutable output collision")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_absent(path, encoded)


def _policy_identity(config: CollectionConfig) -> dict[str, object]:
    runtime = config.runtime_identity
    return {key: runtime[key] for key in ("policy_repo", "policy_revision", "policy_step", "policy_artifact_sha256")}


def _prepare_controller_inputs(config: CollectionConfig) -> None:
    """Materialize only controller-derived immutable inputs; the catalog is supplied."""
    root = _canonical_root(config)
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    catalog = inputs / "seen-catalog.json"
    if catalog.is_symlink() or not catalog.is_file():
        raise ReceiptMismatchError("the approved 40-garment catalog is required before collection")
    _write_immutable_json(inputs / "policy-identity.json", _policy_identity(config))


def _load_json_list(path: Path, *, label: str) -> list[dict[str, object]]:
    if path.is_symlink() or not path.is_file():
        raise ReceiptMismatchError(f"{label} is missing or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReceiptMismatchError(f"{label} is malformed") from error
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ReceiptMismatchError(f"{label} is malformed")
    return value


def _validate_gate_receipt(gate: Path, *, report: Path, matrix: Path) -> str:
    """Recompute the existing canonical gate contract before trusting its word."""
    from check_simple_curriculum_gate import build_gate_receipt

    root = matrix.parents[1]
    policy = root / "inputs/policy-identity.json"; catalog = root / "inputs/seen-catalog.json"
    actual = _strict_json_object(gate, label="first-100 gate receipt")
    expected = build_gate_receipt(
        _strict_json_object(report, label="first-100 report"), report_bytes=report.read_bytes(),
        matrix=_load_json_list(matrix, label="first-100 matrix"), matrix_bytes=matrix.read_bytes(),
        trusted_policy=_strict_json_object(policy, label="policy identity"), policy_bytes=policy.read_bytes(),
        catalog=_strict_json_object(catalog, label="seen catalog"), catalog_bytes=catalog.read_bytes(),
    )
    if actual != expected:
        raise ReceiptMismatchError("first-100 gate receipt does not authenticate report and matrix")
    decision = actual.get("decision")
    if not isinstance(decision, str):
        raise ReceiptMismatchError("first-100 gate decision is missing")
    return decision


def _report_trials(path: Path, *, field: str) -> list[dict[str, object]]:
    report = _strict_json_object(path, label="partition report")
    trials = report.get(field)
    if not isinstance(trials, list) or not all(isinstance(item, dict) for item in trials):
        raise ReceiptMismatchError("partition report has no canonical terminal trials")
    return trials


def _build_calibration_report(config: CollectionConfig) -> None:
    """Join two canonical physical reports into the existing curriculum input shape."""
    root = _canonical_root(config)
    calibration = _load_json_list(root / "matrices/calibration.json", label="calibration matrix")
    head = _report_trials(root / "reports/calibration-head.json", field="gate_trials")
    tail = _report_trials(root / "reports/calibration-tail.json", field="trials")
    outcomes: dict[str, dict[str, object]] = {}
    for trial in head:
        attempt, trial_id, success = trial.get("assignment_id"), trial.get("trial_id"), trial.get("official_success")
        if not isinstance(attempt, str) or not isinstance(trial_id, str) or type(success) is not bool:
            raise ReceiptMismatchError("head report cannot form calibration outcome")
        outcomes[attempt] = {"attempt_id": attempt, "trial_id": trial_id, "success": success}
    for trial in tail:
        # Generic partition reports expose both the durable TaskLedger id and
        # the logical assignment id.  Calibration rows are keyed by the latter.
        attempt, trial_id = trial.get("assignment_id"), trial.get("trial_id")
        success = trial.get("official_success")
        if not isinstance(attempt, str) or not isinstance(trial_id, str) or type(success) is not int or success not in {0, 1} or attempt in outcomes:
            raise ReceiptMismatchError("tail report cannot form calibration outcome")
        outcomes[attempt] = {"attempt_id": attempt, "trial_id": trial_id, "success": bool(success)}
    expected_ids = [str(row.get("attempt_id")) for row in calibration]
    if len(expected_ids) != 400 or set(outcomes) != set(expected_ids):
        raise ReceiptMismatchError("calibration reports do not cover the exact logical matrix")
    report: dict[str, object] = {
        "schema_version": 1, "kind": "lehome_simple_curriculum_calibration_report_v1", "authenticated": True,
        "calibration_matrix_sha256": _file_sha(root / "matrices/calibration.json"),
        "policy_identity": _policy_identity(config), "authenticated_policy_identity": _policy_identity(config),
        "provenance": {"simulator_device": "cpu", "policy_device": str(config.runtime_identity["policy_device"])},
        "catalog": _strict_json_object(root / "inputs/seen-catalog.json", label="seen catalog"),
        "outcomes": [outcomes[item] for item in expected_ids],
    }
    _write_immutable_json(root / "reports/calibration.json", report)


def _validate_calibration_report_artifact(config: CollectionConfig, report: Path) -> None:
    from lehome.flywheel.simple_curriculum import validate_calibration_report

    root = _canonical_root(config)
    try:
        validate_calibration_report(
            _strict_json_object(report, label="calibration report"), matrix_sha256=_file_sha(root / "matrices/calibration.json"),
            policy_identity=_policy_identity(config), catalog=_strict_json_object(root / "inputs/seen-catalog.json", label="seen catalog"),
        )
    except ValueError as error:
        raise ReceiptMismatchError("calibration report is not canonical curriculum evidence") from error


def _fresh_trial_from_terminal_report(
    config: CollectionConfig, *, partition: str, terminal: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Translate an authenticated terminal report, never a bare ledger row.

    ``attempt_id`` below is the durable TaskLedger ID.  The logical source ID
    stays only as audit metadata; replay contracts intentionally use the
    durable ID for both attempt and trial identity.
    """
    root = _canonical_root(config)
    attempt = terminal.get("attempt_id")
    assignment = terminal.get("assignment_id")
    category = terminal.get("category")
    garment = terminal.get("garment")
    event = terminal.get("terminal_event")
    runtime = terminal.get("runtime")
    fidelity = terminal.get("fidelity")
    artifact_root = terminal.get("finalized_artifact_root")
    if (
        not isinstance(attempt, str) or not isinstance(assignment, str)
        or not isinstance(category, str) or not isinstance(garment, str)
        or event not in {"accepted", "rejected"} or not isinstance(runtime, Mapping)
        or not isinstance(fidelity, Mapping) or not isinstance(artifact_root, str)
    ):
        raise ReceiptMismatchError("fresh terminal report has no typed terminal handoff")
    required_runtime = {"simulation_device", "cloth_device", "renderer_device", "camera_device", "policy_device"}
    if set(runtime) != required_runtime or runtime.get("simulation_device") != "cpu" or runtime.get("cloth_device") != "cpu":
        raise ReceiptMismatchError("fresh terminal runtime is not the reviewed CPU tuple")
    if any(not isinstance(runtime.get(key), str) for key in ("renderer_device", "camera_device", "policy_device")):
        raise ReceiptMismatchError("fresh terminal runtime is malformed")
    # Partition reports carry the canonical persistent-worker fidelity schema,
    # while fresh-replay evidence has its own three aggregate failure fields.
    # Translate only after requiring every primitive monitor to be present and
    # clean; never manufacture a clean result from an incomplete report.
    required_fidelity = {
        "missing_cloth", "cloth_flight", "nonfinite_cloth_state",
        "safety_failure", "monitor_active", "monitor_observed",
    }
    if set(fidelity) != required_fidelity or any(type(fidelity.get(key)) is not bool for key in required_fidelity):
        raise ReceiptMismatchError("fresh terminal fidelity evidence is malformed")
    if (
        fidelity["missing_cloth"] or fidelity["cloth_flight"] or fidelity["nonfinite_cloth_state"]
        or fidelity["safety_failure"] or not fidelity["monitor_active"] or not fidelity["monitor_observed"]
    ):
        raise ReceiptMismatchError("fresh terminal fidelity gate did not pass")
    success = event == "accepted"
    if terminal.get("accepted_success") is not success or terminal.get("outcome") != ("success" if success else "failure"):
        raise ReceiptMismatchError("fresh terminal outcome disagrees with finalized evidence")
    trial: dict[str, object] = {
        "attempt_id": attempt, "category": category, "garment_name": garment,
        "accepted_success": success, "official_success": success,
        "outcome": "success" if success else "failure",
        "simulator_device": runtime["simulation_device"], "cloth_device": runtime["cloth_device"],
        "renderer_device": runtime["renderer_device"], "camera_device": runtime["camera_device"],
        "policy_device": runtime["policy_device"],
        # The primitive fidelity evidence above was fully observed and clean.
        # These fields are the immutable fresh-source contract's aggregates.
        "safety_failure": False, "numerical_failure": False, "cloth_failure": False,
        "remote_prefix": f"rollout-rounds/{config.round_id}/{attempt}",
        "campaign_round_id": config.round_id, "campaign_run_id": config.run_id,
        # Keep the terminal hashes in the report as well as the aggregate
        # manifest.  Resume can then cross-bind both independent receipts.
        "episode_sha256": terminal.get("episode_sha256"),
        "worker_receipt_sha256": terminal.get("worker_receipt_sha256"),
    }
    source = {
        "attempt_id": attempt, "trial_id": attempt, "logical_attempt_id": assignment,
        "category": category, "garment_name": garment, "release_stage": "seen", "strategy": "canonical",
        "campaign_kind": "fresh_12k_success_source_v1", "logical_stage": "fresh_success_source",
        "campaign_round_id": config.round_id, "campaign_run_id": config.run_id,
    }
    aggregate = {
        "attempt_id": attempt, "terminal_event": event,
        "logical_attempt_id": assignment, "terminal_report_sha256": terminal.get("episode_sha256"),
        "worker_receipt_sha256": terminal.get("worker_receipt_sha256"),
        "finalized_artifact_root": artifact_root,
    }
    from build_success_replay_matrix import _episode_artifact_sha256

    episode_root = Path(artifact_root)
    episode = _strict_json_object(
        episode_root / "raw" / attempt / "episode.json",
        label="fresh terminal episode",
    )
    identity = episode.get("identity")
    raw_outcome = episode.get("outcome")
    if (
        not isinstance(identity, Mapping)
        or identity.get("campaign_round_id") != config.round_id
        or identity.get("campaign_run_id") != config.run_id
        or episode.get("accepted_success") is not success
        or not isinstance(raw_outcome, str)
        or (raw_outcome == "success") is not success
    ):
        raise ReceiptMismatchError("fresh terminal lacks actual campaign-bound episode evidence")
    receipt = root / "fresh" / partition / "hf-sync-receipts" / f"{attempt}.sync.json"
    sync = _strict_json_object(receipt, label="fresh Hub readback receipt")
    artifact_sha = _episode_artifact_sha256(episode_root)
    if (
        sync.get("readback_verified") is not True or sync.get("attempt_id") != attempt
        or sync.get("round_id") != config.round_id or sync.get("run_id") != config.run_id
        or sync.get("remote_prefix") != trial["remote_prefix"]
        or sync.get("episode_sha256") != artifact_sha
    ):
        raise ReceiptMismatchError("fresh terminal lacks actual campaign-bound Hub readback evidence")
    if success:
        trial.update(artifact_sha256=artifact_sha, hub_sync_receipt_sha256=_file_sha(receipt))
        aggregate["hub_sync_receipt_sha256"] = _file_sha(receipt)
        aggregate["artifact_sha256"] = artifact_sha
    return source, trial, aggregate


def _build_fresh_source_report(config: CollectionConfig) -> None:
    root = _canonical_root(config)
    if re.fullmatch(r"fresh-12k-[a-z0-9-]{1,112}", config.round_id) is None or re.fullmatch(r"fresh-run-[a-z0-9-]{1,112}", config.run_id) is None:
        raise ReceiptMismatchError("fresh report requires fresh 12K run and round identities")
    source_rows: list[dict[str, object]] = []; trials: list[dict[str, object]] = []; aggregate: list[dict[str, object]] = []
    for partition in ("calibration-head", "calibration-tail", "curriculum-a", "curriculum-b"):
        report_path = root / "reports" / "partitions" / f"{partition}.json"
        report = _strict_json_object(report_path, label="fresh partition terminal report")
        if report.get("kind") != "lehome_simple_curriculum_partition_report_v1":
            raise ReceiptMismatchError("fresh source must be constructed from a canonical partition report")
        terminal_trials = report.get("trials")
        if not isinstance(terminal_trials, list):
            raise ReceiptMismatchError("fresh partition report has no terminal evidence")
        for terminal in terminal_trials:
            if not isinstance(terminal, Mapping):
                raise ReceiptMismatchError("fresh partition terminal evidence is malformed")
            source, trial, evidence = _fresh_trial_from_terminal_report(config, partition=partition, terminal=terminal)
            source_rows.append(source); trials.append(trial); aggregate.append(evidence)
    if len(source_rows) != 1000 or len({str(row["attempt_id"]) for row in source_rows}) != 1000:
        raise ReceiptMismatchError("fresh source matrix must contain exactly 1,000 distinct outcomes")
    matrix = root / "reports/fresh-source-matrix.json"; _write_immutable_json(matrix, source_rows)
    terminal_manifest = root / "reports/fresh-terminal-artifacts.json"
    _write_immutable_json(terminal_manifest, {"schema_version": 1, "kind": "lehome_fresh_terminal_artifact_manifest_v1", "entries": aggregate})
    report: dict[str, object] = {
        "schema_version": 1, "kind": "lehome_fresh_12k_success_source_report_v1",
        "campaign_kind": "fresh_12k_success_source_v1", "logical_stage": "fresh_success_source",
        "round_id": config.round_id, "run_id": config.run_id, "matrix_sha256": _file_sha(matrix),
        "identity": _policy_identity(config), "trials": trials, "safety_failure": False,
    }
    report["report_sha256"] = _digest(report)
    _write_immutable_json(root / "reports/fresh-source-report.json", report)


def _validate_fresh_source_outputs(config: CollectionConfig, *, report: Path, matrix: Path) -> dict[str, dict[str, object]]:
    from lehome.flywheel.fresh_replay_evidence import authenticate_fresh_source_contract

    try:
        authenticated = authenticate_fresh_source_contract((report,), (matrix,))
    except ValueError as error:
        raise ReceiptMismatchError("fresh source report is not eligible replay evidence") from error
    if len(authenticated) != 1000:
        raise ReceiptMismatchError("fresh source report has an incorrect terminal outcome count")
    return authenticated


def _validate_fresh_terminal_artifact_manifest(
    config: CollectionConfig, *, authenticated: Mapping[str, Mapping[str, object]], manifest: Path,
) -> None:
    """Re-open every reported terminal root before adopting a fresh-source report.

    The fresh source report is deliberately compact, but it is only useful if
    the terminal artifacts and Hub receipts it was built from still bind the
    exact durable TaskLedger identifiers.  This makes a pre-journal crash
    adoption safe: a complete-looking report cannot mask a changed artifact,
    a swapped logical assignment, or a local-only accepted result.
    """
    from build_success_replay_matrix import _episode_artifact_sha256

    root = _canonical_root(config)
    payload = _strict_json_object(manifest, label="fresh terminal artifact manifest")
    if (
        set(payload) != {"schema_version", "kind", "entries"}
        or payload.get("schema_version") != 1
        or payload.get("kind") != "lehome_fresh_terminal_artifact_manifest_v1"
        or not isinstance(payload.get("entries"), list)
    ):
        raise ReceiptMismatchError("fresh terminal artifact manifest is invalid")
    entries = payload["entries"]
    if len(entries) != 1000:
        raise ReceiptMismatchError("fresh terminal artifact manifest has an incorrect outcome count")
    by_attempt: dict[str, Mapping[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("attempt_id"), str):
            raise ReceiptMismatchError("fresh terminal artifact manifest entry is malformed")
        attempt = str(entry["attempt_id"])
        if attempt in by_attempt or attempt not in authenticated:
            raise ReceiptMismatchError("fresh terminal artifact manifest has an unknown or duplicate attempt")
        by_attempt[attempt] = entry
    if set(by_attempt) != set(authenticated):
        raise ReceiptMismatchError("fresh terminal artifact manifest does not cover the source matrix")

    for attempt, context in authenticated.items():
        trial = context.get("trial")
        rows = context.get("source_matrix_rows")
        entry = by_attempt[attempt]
        if not isinstance(trial, Mapping) or not isinstance(rows, Mapping):
            raise ReceiptMismatchError("fresh terminal source context is malformed")
        row = rows.get(attempt)
        if not isinstance(row, Mapping):
            raise ReceiptMismatchError("fresh terminal source matrix is malformed")
        event = "accepted" if trial.get("accepted_success") is True else "rejected"
        required = {
            "attempt_id", "terminal_event", "logical_attempt_id", "terminal_report_sha256",
            "worker_receipt_sha256", "finalized_artifact_root",
        }
        if event == "accepted":
            required |= {"hub_sync_receipt_sha256", "artifact_sha256"}
        if set(entry) != required:
            raise ReceiptMismatchError("fresh terminal artifact manifest entry has an unexpected schema")
        if (
            entry.get("terminal_event") != event
            or entry.get("logical_attempt_id") != row.get("logical_attempt_id")
            or entry.get("terminal_report_sha256") != trial.get("episode_sha256")
            or entry.get("worker_receipt_sha256") != trial.get("worker_receipt_sha256")
            or not isinstance(entry.get("finalized_artifact_root"), str)
        ):
            raise ReceiptMismatchError(f"fresh terminal artifact manifest does not bind report identities: {attempt}")
        artifact_root = Path(str(entry["finalized_artifact_root"]))
        try:
            relative = artifact_root.relative_to(root)
        except ValueError as error:
            raise ReceiptMismatchError("fresh terminal artifact root escapes collection") from error
        if (
            artifact_root.is_symlink() or not artifact_root.is_dir()
            or len(relative.parts) != 4 or relative.parts[0] != "fresh"
            or relative.parts[2] not in {"accepted", "evaluation-terminal"}
            or relative.parts[3] != attempt
        ):
            raise ReceiptMismatchError("fresh terminal artifact root is not canonical")
        episode = artifact_root / "raw" / attempt / "episode.json"
        worker_receipt = artifact_root / "worker-receipt.json"
        if _file_sha(episode) != entry["terminal_report_sha256"] or _file_sha(worker_receipt) != entry["worker_receipt_sha256"]:
            raise ReceiptMismatchError("fresh terminal artifact bytes no longer match report evidence")
        artifact_sha = _episode_artifact_sha256(artifact_root)
        receipt = artifact_root.parent.parent / "hf-sync-receipts" / f"{attempt}.sync.json"
        sync = _strict_json_object(receipt, label="fresh Hub readback receipt")
        if (
            sync.get("readback_verified") is not True or sync.get("attempt_id") != attempt
            or sync.get("round_id") != config.round_id or sync.get("run_id") != config.run_id
            or sync.get("remote_prefix") != f"rollout-rounds/{config.round_id}/{attempt}"
            or sync.get("episode_sha256") != artifact_sha
        ):
            raise ReceiptMismatchError("fresh terminal artifact lacks a matching Hub readback")
        if event != "accepted":
            continue
        if (
            artifact_sha != entry.get("artifact_sha256") or artifact_sha != trial.get("artifact_sha256")
            or _file_sha(receipt) != entry.get("hub_sync_receipt_sha256")
            or _file_sha(receipt) != trial.get("hub_sync_receipt_sha256")
        ):
            raise ReceiptMismatchError("fresh terminal accepted artifact lacks a matching Hub readback")


def _validate_replay_matrix_receipt(matrix: Path, receipt: Path) -> int:
    if matrix.is_symlink() or not matrix.is_file():
        raise ReceiptMismatchError("success replay matrix receipt is invalid")
    try:
        rows = json.loads(matrix.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReceiptMismatchError("success replay matrix receipt is invalid") from error
    if (receipt.is_symlink() or not receipt.is_file()
            or _file_sha(matrix) != receipt.read_text(encoding="utf-8").strip()
            or not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows) or len(rows) > 400):
        raise ReceiptMismatchError("success replay matrix receipt is invalid")
    return len(rows)


def _discover_success_replay(config: CollectionConfig, *, matrix: Path, ledger: Path) -> Mapping[str, object]:
    """Authenticate the exact 4x100/4x50 visual replay terminal boundary.

    A 400-attempt campaign is successful only when its ledger has exactly 50
    accepted outcomes in every category and every one of those artifacts has
    a readback-verified Hub receipt.  If all 400 rows finished but any cap is
    short, preserve that as the explicit data outcome ``replay_shortage``.
    """
    from lehome.flywheel.task_ledger import TaskLedger

    root = _canonical_root(config)
    rows = _load_json_list(matrix, label="success replay matrix")
    expected_categories = Counter({"top_long": 100, "top_short": 100, "pant_long": 100, "pant_short": 100})
    if len(rows) != 400 or Counter(row.get("category") for row in rows) != expected_categories:
        raise ReceiptMismatchError("success replay matrix does not have the exact four-category attempt caps")
    if any(row.get("strategy") != "visual_only" or row.get("category_acceptance_cap") != 50 for row in rows):
        raise ReceiptMismatchError("success replay matrix does not bind the exact visual acceptance caps")
    try:
        task_ledger = TaskLedger(ledger, attempt_matrix=rows, max_attempts=400, target_accepted=200)
        try:
            attempts = task_ledger.attempts()
            statuses = {attempt.attempt_id: task_ledger.status(attempt.attempt_id) for attempt in attempts}
        finally:
            task_ledger.close()
    except (OSError, ValueError, sqlite3.Error) as error:
        raise ReceiptMismatchError("success replay ledger is malformed") from error
    if len(attempts) != 400 or any(status == "infrastructure_abort" for status in statuses.values()):
        raise ReceiptMismatchError("success replay contains unresolved infrastructure evidence")
    accepted = tuple(attempt for attempt in attempts if statuses.get(attempt.attempt_id) == "accepted")
    accepted_categories = Counter(str(attempt.assignment.get("category")) for attempt in accepted)
    if len(accepted) > 200 or any(accepted_categories[category] > 50 for category in expected_categories):
        raise ReceiptMismatchError("success replay ledger exceeds an immutable acceptance cap")
    seal = root / "replay" / "success-replay-readback-seal.json"
    if len(accepted) < 200:
        if all(status in {"accepted", "rejected"} for status in statuses.values()):
            shortage_body: dict[str, object] = {
                "schema_version": 1, "kind": "lehome_success_replay_readback_seal_v1",
                "round_id": config.round_id + "-replay", "run_id": config.run_id,
                "matrix_sha256": _file_sha(matrix),
                "accepted_attempt_ids": [attempt.attempt_id for attempt in accepted],
                "accepted_by_category": dict(sorted(accepted_categories.items())),
                "readback_receipts": {}, "readback_verified": False, "outcome": "replay_shortage",
            }
            _write_immutable_json(seal, {**shortage_body, "seal_sha256": _digest(shortage_body)})
            return {"result": "replay_shortage", "readback_seal": seal}
        raise ReceiptMismatchError("success replay is incomplete before a declared source shortage")
    if accepted_categories != Counter({category: 50 for category in expected_categories}):
        raise ReceiptMismatchError("success replay accepted terminal category counts are not exact")

    round_id = config.round_id + "-replay"
    from build_success_replay_matrix import _episode_artifact_sha256

    receipts: dict[str, dict[str, str]] = {}
    for attempt in accepted:
        attempt_id = attempt.attempt_id
        artifact = root / "replay" / "accepted" / attempt_id
        receipt_path = root / "replay" / "hf-sync-receipts" / f"{attempt_id}.sync.json"
        artifact_sha = _episode_artifact_sha256(artifact)
        receipt = _strict_json_object(receipt_path, label="success replay Hub readback receipt")
        immutable_revision = receipt.get("immutable_revision")
        if (
            receipt.get("schema_version") != 1 or receipt.get("attempt_id") != attempt_id
            or receipt.get("repository") != "ryanjin333/lehome-groot-n17-rollouts"
            or receipt.get("round_id") != round_id
            or receipt.get("run_id") != config.run_id
            or receipt.get("remote_prefix") != f"rollout-rounds/{round_id}/{attempt_id}"
            or receipt.get("readback_verified") is not True or receipt.get("episode_sha256") != artifact_sha
            or not isinstance(immutable_revision, str) or re.fullmatch(r"[0-9a-f]{40}", immutable_revision) is None
        ):
            raise ReceiptMismatchError("success replay accepted artifact lacks a matching Hub readback receipt")
        receipts[attempt_id] = {
            "receipt_sha256": _file_sha(receipt_path), "episode_sha256": artifact_sha,
            "immutable_revision": immutable_revision,
        }
    body: dict[str, object] = {
        "schema_version": 1, "kind": "lehome_success_replay_readback_seal_v1",
        "round_id": round_id, "run_id": config.run_id, "matrix_sha256": _file_sha(matrix),
        "accepted_attempt_ids": [attempt.attempt_id for attempt in accepted],
        "accepted_by_category": dict(sorted(accepted_categories.items())),
        "readback_receipts": receipts, "readback_verified": True,
    }
    payload = {**body, "seal_sha256": _digest(body)}
    _write_immutable_json(seal, payload)
    return {"result": "complete", "readback_seal": seal}


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
    spend_observer: Path | None = None
    rollout_instance_id: str = _EXACT_ROLLOUT_INSTANCE_ID

    def validate(self, *, require_git: bool = False) -> None:
        _safe_directory(self.host_code_root, must_exist=True)
        _safe_directory(self.campaign_root, must_exist=False)
        for identifier in (self.run_id, self.round_id):
            if identifier in _DEFAULT_IDS or _SAFE_ID.fullmatch(identifier) is None:
                raise ValueError("fresh caller-supplied run and round IDs are required")
        if self.paid and (
            re.fullmatch(r"fresh-run-[a-z0-9-]{1,112}", self.run_id) is None
            or re.fullmatch(r"fresh-12k-[a-z0-9-]{1,112}", self.round_id) is None
        ):
            raise ValueError("paid collection requires fresh-run and fresh-12k identities before spending")
        for value, label, ceiling in ((self.max_wall_seconds, "max wall time", 86_400.0), (self.max_spend_usd, "max spend", 100.0)):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0 or value >= ceiling:
                raise ValueError(f"{label} must be finite, positive, and bounded")
        if self.paid and self.gpu_stop_command != _TRUSTED_GPU_STOP:
            raise ValueError("paid collection requires the fixed trusted GPU stop hook")
        if self.rollout_instance_id != _EXACT_ROLLOUT_INSTANCE_ID:
            raise ValueError("collection is pinned to the one approved rollout VM")
        if self.paid and self.spend_observer is None:
            raise ValueError("paid collection requires a typed spend observer")
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
            "spend_observer": str(self.spend_observer) if self.spend_observer else None,
            "rollout_instance_id": self.rollout_instance_id,
        }


class Runner(Protocol):
    def run(self, stage: str, **kwargs: object) -> Mapping[str, object]: ...
    def stop_gpu(self, command: str) -> None: ...


class CommandRunner:
    """Fixed reviewed stage adapter; no caller-provided command surface exists."""

    def __init__(self, config: CollectionConfig | None = None) -> None:
        self.config = config
        self.budget_check: Any = None

    def _require_config(self) -> CollectionConfig:
        if self.config is None:
            raise ValueError("fixed adapter requires collection config")
        return self.config

    def environment_for(self, stage: str, inputs: Mapping[str, object]) -> dict[str, str]:
        """Return the complete inherited-free environment for one reviewed stage.

        The appliance is intentionally environment-driven, but its values are
        controller-owned here.  Do not add ``os.environ`` to this mapping:
        caller-provided LEHOME values must never alter a paid collection.
        """
        config = self._require_config()
        root = _canonical_root(config)
        environment = {
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "LC_ALL": "C",
            "PYTHONPATH": str(config.host_code_root / "source" / "lehome"),
            "LEHOME_SIMPLE_CURRICULUM_COLLECTION": "1",
            "LEHOME_ONE_VM_ORCHESTRATOR": "1",
            "LEHOME_PAID_COLLECTION": "1" if config.paid else "0",
        }
        if stage == "success-replay":
            replay = root / "replay/replay.json"
            reports = root / "reports/fresh-source-report.json"
            matrices = root / "reports/fresh-source-matrix.json"
            environment.update({
                "LEHOME_HOST_CODE_ROOT": str(config.host_code_root),
                "LEHOME_SIMPLE_CURRICULUM_COLLECTION": "0",
                # Replay is still a paid one-VM continuation.  Only the CPU
                # mode marker changes; clearing these selected baked code in
                # the wrapper and bypassed the journaled path.
                "LEHOME_ONE_VM_ORCHESTRATOR": "1",
                "LEHOME_PAID_COLLECTION": "1" if config.paid else "0",
                "LEHOME_SUCCESS_REPLAY_MATRIX": str(replay),
                "LEHOME_SUCCESS_REPLAY_MATRIX_SHA256": _file_sha(replay),
                "LEHOME_CAMPAIGN_ROOT": str(root / "replay"),
                "LEHOME_WORKSPACE": "/mnt/lehome",
                "LEHOME_WORKER_COUNT": "4", "LEHOME_MAX_ATTEMPTS": "400", "LEHOME_TARGET_ACCEPTED": "200",
                "LEHOME_RUN_ID": config.run_id, "LEHOME_ROUND_ID": config.round_id + "-replay",
                "LEHOME_ROLLOUT_IMAGE": str(config.runtime_identity["rollout_image"]),
                "LEHOME_TRAINER_IMAGE": str(config.runtime_identity["trainer_image"]),
                "LEHOME_ENABLE_HF_UPLOAD": "1", "LEHOME_SKIP_ROUND_SEAL": "1",
                "LEHOME_FRESH_SOURCE_REPORTS_JSON": json.dumps([{"path": str(reports), "sha256": _file_sha(reports)}]),
                "LEHOME_FRESH_SOURCE_MATRICES_JSON": json.dumps([{"path": str(matrices), "sha256": _file_sha(matrices)}]),
            })
            return environment
        if stage == "final-publication":
            if set(inputs) != {"terminal_outcome"} or not isinstance(inputs.get("terminal_outcome"), str):
                raise ValueError("final publication inputs are invalid")
            environment.update({
                "LEHOME_CAMPAIGN_ROOT": str(root), "LEHOME_RUN_ID": config.run_id,
                "LEHOME_ROUND_ID": config.round_id, "LEHOME_TERMINAL_OUTCOME": str(inputs["terminal_outcome"]),
                "LEHOME_ROLLOUT_INSTANCE_ID": config.rollout_instance_id,
                "LEHOME_HF_TOKEN_FILE": "/mnt/lehome/secrets/hf_token",
                "LEHOME_ROLLOUT_REPOSITORY": "ryanjin333/lehome-groot-n17-rollouts",
                "LEHOME_HF_REVISION": "main",
            })
            return environment
        if stage not in {"calibration-head", "calibration-tail", "curriculum-a", "curriculum-b"}:
            return environment
        required = {"partition_id", "partition_matrix", "partition_manifest", "partition_sha256", "row_start", "row_end", "target", "lease_budget"}
        if set(inputs) != required:
            raise ValueError("partition adapter inputs are not exact")
        partition = str(inputs["partition_id"])
        if partition not in {"calibration-head", "calibration-tail", "curriculum-a", "curriculum-b"}:
            raise ValueError("unknown partition")
        matrix = root / str(inputs["partition_matrix"])
        manifest = root / str(inputs["partition_manifest"])
        matrix_sha = _file_sha(matrix)
        if matrix_sha != inputs["partition_sha256"]:
            raise ReceiptMismatchError("partition adapter matrix hash mismatch")
        stage_root = root / "fresh" / partition
        resume_context = stage_root / "rollout-preemption.json"
        ledger_exists = (stage_root / "ledger.sqlite3").is_file()
        if ledger_exists and (resume_context.is_symlink() or not resume_context.is_file()):
            raise ReceiptMismatchError("resumed partition is missing its durable preemption context")
        runtime = config.runtime_identity
        environment.update({
            "LEHOME_HOST_CODE_ROOT": str(config.host_code_root),
            "LEHOME_WORKSPACE": "/mnt/lehome",
            "LEHOME_CAMPAIGN_ROOT": str(stage_root),
            "LEHOME_ATTEMPT_MATRIX": str(matrix),
            "LEHOME_ATTEMPT_MATRIX_SHA256": matrix_sha,
            "LEHOME_PARTITION_MANIFEST": str(manifest),
            "LEHOME_PARTITION_ID": partition,
            "LEHOME_PARENT_MATRIX_SHA256": _partition_parent_sha(manifest),
            "LEHOME_MAX_ATTEMPTS": str(inputs["lease_budget"]),
            "LEHOME_TARGET_ACCEPTED": str(inputs["target"]),
            "LEHOME_WORKER_COUNT": "4",
            "LEHOME_SIMULATOR_DEVICE": "cpu",
            "LEHOME_ENABLE_HF_UPLOAD": "1",
            "LEHOME_SKIP_ROUND_SEAL": "1",
            "LEHOME_COMPLETION_METRIC": "terminal_outcomes",
            "LEHOME_POLICY_REPO": str(runtime["policy_repo"]),
            "LEHOME_POLICY_REVISION": str(runtime["policy_revision"]),
            "LEHOME_POLICY_STEP": str(runtime["policy_step"]),
            "LEHOME_POLICY_ARTIFACT_SHA256": str(runtime["policy_artifact_sha256"]),
            "LEHOME_ROLLOUT_IMAGE": str(runtime["rollout_image"]),
            "LEHOME_TRAINER_IMAGE": str(runtime["trainer_image"]),
            "LEHOME_RUN_ID": config.run_id,
            "LEHOME_ROUND_ID": config.round_id,
            "LEHOME_HF_TOKEN_FILE": "/mnt/lehome/secrets/hf_token",
            "LEHOME_ROLLOUT_REPOSITORY": "ryanjin333/lehome-groot-n17-rollouts",
            "LEHOME_HF_REVISION": "main",
            "LEHOME_RESUME_PREEMPTED_ROLLOUT": "1" if ledger_exists else "0",
            "LEHOME_ROLLOUT_PREEMPTION_CONTEXT": str(resume_context),
        })
        return environment

    def argv_for(self, stage: str, inputs: Mapping[str, object]) -> tuple[str, ...]:
        if self.config is None: raise ValueError("fixed adapter requires collection config")
        root, code = _canonical_root(self.config), self.config.host_code_root
        py = str(Path(os.sys.executable).resolve())
        scripts = code / "scripts"; appliance = code / "rollout_appliance"
        if stage == "calibration-matrix": return (py, str(scripts / "build_simple_curriculum_matrix.py"), "build-calibration", "--catalog", str(root / "inputs/seen-catalog.json"), "--seed-base", "2026082801", "--output", str(root / "matrices/calibration.json"), "--receipt", str(root / "matrices/calibration.receipt.json"))
        if stage in {"calibration-head", "calibration-tail", "curriculum-a", "curriculum-b"}: return (str(appliance / "run_12k_campaign.sh"),)
        if stage == "first-100-gate": return (
            py, str(scripts / "check_simple_curriculum_gate.py"),
            "--report", str(root / "reports/calibration-head.json"),
            "--matrix", str(root / "partitions/calibration-head.json"),
            "--trusted-policy-identity", str(root / "inputs/policy-identity.json"),
            "--approved-garment-catalog", str(root / "inputs/seen-catalog.json"),
            "--output", str(root / "reports/first-100-gate.json"),
        )
        if stage == "calibration-report": return self._summary_argv("calibration-tail", root / "reports/calibration-tail.json", simple_partition=True)
        if stage == "fresh-report": return self._summary_argv("curriculum-b", root / "reports" / "partitions" / "curriculum-b.json", simple_partition=True)
        if stage == "curriculum-matrix": return (
            py, str(scripts / "build_simple_curriculum_matrix.py"), "build-curriculum",
            "--report", str(root / "reports/calibration.json"),
            "--calibration-matrix", str(root / "matrices/calibration.json"),
            "--approved-catalog", str(root / "inputs/seen-catalog.json"),
            "--policy-identity", str(root / "inputs/policy-identity.json"),
            "--rng-seed", "2026082802", "--output", str(root / "matrices/curriculum.json"),
            "--receipt", str(root / "matrices/curriculum.receipt.json"),
        )
        if stage == "replay-matrix":
            argv: list[str] = [py, str(scripts / "build_success_replay_matrix.py")]
            for partition in ("calibration-head", "calibration-tail", "curriculum-a", "curriculum-b"):
                argv.extend(("--accepted-root", str(root / "fresh" / partition / "accepted")))
            argv.extend((
                "--output", str(root / "replay/replay.json"), "--source-report", str(root / "reports/fresh-source-report.json"),
                "--source-matrix", str(root / "reports/fresh-source-matrix.json"), "--strategy", "visual_only",
                "--attempt-cap-per-category", "100", "--acceptance-cap-per-category", "50",
                "--max-attempts", "400", "--target-accepted", "200", "--rng-seed", "2026082803",
            ))
            return tuple(argv)
        if stage == "success-replay": return (str(appliance / "run_success_replay_campaign.sh"),)
        if stage == "final-publication":
            publisher = scripts / "publish_simple_curriculum_collection.py"
            if not publisher.is_file(): raise RuntimeError("canonical Task 7 publisher adapter is not available")
            return (py, str(publisher))
        raise ValueError("unknown fixed stage")

    def _summary_argv(self, partition: str, output: Path, *, simple_partition: bool = False) -> tuple[str, ...]:
        config = self._require_config(); root = _canonical_root(config); runtime = config.runtime_identity
        argv = (
            str(Path(os.sys.executable).resolve()), str(config.host_code_root / "scripts" / "summarize_groot_persistent_evaluation.py"),
            "--campaign-root", str(root / "fresh" / partition),
            "--matrix", str(root / "partitions" / f"{partition}.json"),
            "--matrix-sha256", _file_sha(root / "partitions" / f"{partition}.json"),
            "--candidate-key", "original_baseline", "--policy-repo", str(runtime["policy_repo"]),
            "--policy-revision", str(runtime["policy_revision"]), "--policy-step", str(runtime["policy_step"]),
            "--policy-artifact-sha256", str(runtime["policy_artifact_sha256"]), "--output", str(output),
        )
        return argv + (("--simple-partition",) if simple_partition else ())

    def _output_paths(self, stage: str, inputs: Mapping[str, object]) -> tuple[Path, ...]:
        """Fixed canonical paths used to adopt a post-crash completed stage."""
        root = _canonical_root(self._require_config())
        paths = {
            "calibration-matrix": (root / "matrices/calibration.json", root / "matrices/calibration.receipt.json"),
            "first-100-gate": (root / "reports/calibration-head.json", root / "reports/first-100-gate.json"),
            "calibration-report": (root / "reports/calibration-tail.json", root / "reports/calibration.json"),
            "curriculum-matrix": (root / "matrices/curriculum.json", root / "matrices/curriculum.receipt.json"),
            "fresh-report": (root / "reports/fresh-source-report.json", root / "reports/fresh-source-matrix.json", root / "reports/fresh-terminal-artifacts.json"),
            "replay-matrix": (root / "replay/replay.json", root / "replay/replay.json.sha256"),
            "success-replay": (
                root / "replay/replay.json", root / "replay/ledger.sqlite3",
                root / "replay/success-replay-readback-seal.json",
            ),
            "final-publication": (
                root / "reports/final-publication.json", root / "reports/final-publication-readback.json",
            ),
        }
        if stage in {"calibration-head", "calibration-tail", "curriculum-a", "curriculum-b"}:
            partition = str(inputs["partition_id"])
            return (root / str(inputs["partition_matrix"]), root / str(inputs["partition_manifest"]), root / "fresh" / partition / "ledger.sqlite3")
        return paths.get(stage, ())

    def _invoke_if_absent(self, argv: tuple[str, ...], *, stage: str, output: Path, inputs: Mapping[str, object] | None = None) -> None:
        if output.exists() or output.is_symlink():
            if output.is_symlink() or not output.is_file():
                raise ReceiptMismatchError("post-crash stage output is unsafe")
            return
        self._invoke(argv, stage=stage, inputs=inputs)

    def run(self, stage: str, **kwargs: object) -> Mapping[str, object]:
        config = self._require_config()
        root = _canonical_root(config)
        if stage in {"calibration-matrix", "curriculum-matrix"}:
            (root / "matrices").mkdir(parents=True, exist_ok=True)
        if stage in {"first-100-gate", "calibration-report", "fresh-report", "replay-matrix"}:
            (root / "reports").mkdir(parents=True, exist_ok=True)
        if stage == "replay-matrix":
            (root / "replay").mkdir(parents=True, exist_ok=True)
        partition_stage = stage in {"calibration-head", "calibration-tail", "curriculum-a", "curriculum-b"}
        if partition_stage:
            # The matrix and manifest are immutable *inputs*, materialized by
            # the controller before the appliance starts.  Only the ledger is
            # a completion output.  A valid partial ledger is resumed through
            # its durable preemption context; it is never mis-adopted.
            partition = str(kwargs["partition_id"])
            matrix = root / str(kwargs["partition_matrix"])
            manifest = root / str(kwargs["partition_manifest"])
            ledger = root / "fresh" / partition / "ledger.sqlite3"
            _validate_partition_manifest(manifest, matrix=matrix, partition_id=partition, inputs=kwargs)
            if ledger.exists() or ledger.is_symlink():
                completed = _partition_ledger_terminal_count(
                    ledger, matrix=matrix, max_attempts=int(kwargs["lease_budget"]), target=int(kwargs["target"]),
                )
                if completed == int(kwargs["target"]):
                    return self._discover(stage, kwargs)
            self._invoke(self.argv_for(stage, kwargs), stage=stage, inputs=kwargs)
            return self._discover(stage, kwargs)
        if stage == "final-publication":
            receipt, readback = self._output_paths(stage, kwargs)
            # A crash can land after the publisher durably fsyncs the immutable
            # receipt but before it writes the local readback receipt.  Never
            # treat that as a completed stage and never invoke the normal
            # mutable-ref upload path again.  The publisher's explicit
            # reconcile mode pins the receipt's commit/manifest and performs
            # fresh authenticated plus anonymous downloads only.
            if receipt.exists() or receipt.is_symlink():
                if readback.exists() or readback.is_symlink():
                    return self._discover(stage, kwargs)
                self._invoke(
                    self.argv_for(stage, kwargs) + ("--reconcile",),
                    stage=stage, inputs=kwargs,
                )
                return self._discover(stage, kwargs)
        output_paths = self._output_paths(stage, kwargs)
        if output_paths and any(path.exists() or path.is_symlink() for path in output_paths):
            # A completed output is adopted only after deep stage validation.
            # A partial/mismatched output is never overwritten or rerun.
            return self._discover(stage, kwargs)
        if stage == "first-100-gate":
            self._invoke_if_absent(self._summary_argv("calibration-head", root / "reports/calibration-head.json"), stage="calibration-head-summary", output=root / "reports/calibration-head.json")
        elif stage == "fresh-report":
            for partition in ("calibration-head", "calibration-tail", "curriculum-a"):
                output = root / "reports" / "partitions" / f"{partition}.json"
                self._invoke_if_absent(
                    self._summary_argv(partition, root / "reports" / "partitions" / f"{partition}.json", simple_partition=True),
                    stage=f"{partition}-summary",
                    output=output,
                )
        argv = self.argv_for(stage, kwargs)
        if stage == "calibration-report":
            self._invoke_if_absent(argv, stage=stage, output=root / "reports/calibration-tail.json")
        elif stage == "fresh-report":
            self._invoke_if_absent(argv, stage=stage, output=root / "reports/partitions/curriculum-b.json")
        else:
            self._invoke(argv, stage=stage, inputs=kwargs)
        if stage == "calibration-report":
            _build_calibration_report(config)
        elif stage == "fresh-report":
            _build_fresh_source_report(config)
        return self._discover(stage, kwargs)

    def _invoke(self, argv: tuple[str, ...], *, stage: str, inputs: Mapping[str, object] | None = None) -> None:
        config = self._require_config()
        process = subprocess.Popen(
            argv, cwd=config.host_code_root, env=self.environment_for(stage if inputs is not None else "summary", inputs or {}),
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        stdout_tail, stderr_tail = _PipeTail(process.stdout), _PipeTail(process.stderr)
        drainers = tuple(
            threading.Thread(target=tail.drain, name=f"lehome-{stage}-pipe", daemon=True)
            for tail in (stdout_tail, stderr_tail)
        )
        for drainer in drainers:
            drainer.start()
        try:
            while process.poll() is None:
                # Final publication/readback runs only after the exact VM has
                # been authoritatively stopped.  It is a zero-compute retry
                # boundary, so an exhausted paid budget must not strand
                # already-collected evidence on the local disk.
                if self.budget_check is not None and stage != "final-publication":
                    try:
                        self.budget_check()
                    except Exception:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill(); process.wait(timeout=10)
                        raise
                time.sleep(0.1)
        finally:
            # Draining prevents a verbose child from wedging on the OS pipe
            # buffer.  Tails are bounded and are only used in typed errors.
            for drainer in drainers:
                drainer.join(timeout=10)
        completed = subprocess.CompletedProcess(argv, process.returncode, stdout_tail.text(), stderr_tail.text())
        if completed.returncode:
            raise RuntimeError(f"fixed {stage} adapter failed: {completed.returncode}: {completed.stderr.strip()}")

    def _discover(self, stage: str, inputs: Mapping[str, object]) -> Mapping[str, object]:
        """Discover only canonical outputs at this stage's fixed locations."""
        config = self._require_config()
        root = _canonical_root(config)
        if stage == "calibration-matrix":
            matrix = root / "matrices/calibration.json"
            receipt = root / "matrices/calibration.receipt.json"
            _validate_matrix_receipt(matrix, receipt, expected_rows=400)
            return {"artifacts": _descriptors(root, matrix=matrix, matrix_receipt=receipt)}
        if stage in {"calibration-head", "calibration-tail", "curriculum-a", "curriculum-b"}:
            partition = str(inputs["partition_id"])
            matrix = root / str(inputs["partition_matrix"])
            manifest = root / str(inputs["partition_manifest"])
            ledger = root / "fresh" / partition / "ledger.sqlite3"
            _validate_partition_manifest(manifest, matrix=matrix, partition_id=partition, inputs=inputs)
            _validate_partition_ledger(
                ledger, matrix=matrix, max_attempts=int(inputs["lease_budget"]), target=int(inputs["target"]),
            )
            return {"artifacts": _descriptors(root, matrix=matrix, manifest=manifest, ledger=ledger)}
        if stage == "first-100-gate":
            report = root / "reports/calibration-head.json"; gate = root / "reports/first-100-gate.json"
            decision = _validate_gate_receipt(gate, report=report, matrix=root / "partitions/calibration-head.json")
            return {"artifacts": _descriptors(root, report=report, gate_receipt=gate), "decision": decision}
        if stage == "calibration-report":
            report = root / "reports/calibration.json"; _validate_calibration_report_artifact(config, report)
            return {"artifacts": _descriptors(root, report=report)}
        if stage == "curriculum-matrix":
            matrix = root / "matrices/curriculum.json"; receipt = root / "matrices/curriculum.receipt.json"
            _validate_matrix_receipt(matrix, receipt, expected_rows=600)
            return {"artifacts": _descriptors(root, matrix=matrix, matrix_receipt=receipt)}
        if stage == "fresh-report":
            report = root / "reports/fresh-source-report.json"; matrix = root / "reports/fresh-source-matrix.json"
            manifest = root / "reports/fresh-terminal-artifacts.json"
            authenticated = _validate_fresh_source_outputs(config, report=report, matrix=matrix)
            _validate_fresh_terminal_artifact_manifest(config, authenticated=authenticated, manifest=manifest)
            return {"artifacts": _descriptors(root, report=report, matrix=matrix, terminal_artifact_manifest=manifest)}
        if stage == "replay-matrix":
            matrix = root / "replay/replay.json"; receipt = root / "replay/replay.json.sha256"
            if not matrix.exists():
                raise ReceiptMismatchError("replay builder did not materialize a canonical shortage matrix")
            count = _validate_replay_matrix_receipt(matrix, receipt)
            output: dict[str, object] = {"artifacts": _descriptors(root, matrix=matrix, matrix_receipt=receipt)}
            if count != 400:
                output["result"] = "replay_shortage"
            return output
        if stage == "success-replay":
            ledger = root / "replay" / "ledger.sqlite3"; matrix = root / "replay/replay.json"
            result = _discover_success_replay(config, matrix=matrix, ledger=ledger)
            if result["result"] == "replay_shortage":
                seal = result.get("readback_seal")
                if not isinstance(seal, Path):
                    raise ReceiptMismatchError("success replay shortage seal is missing")
                return {"artifacts": _descriptors(root, matrix=matrix, ledger=ledger, readback_seal=seal), "result": "replay_shortage"}
            seal = result.get("readback_seal")
            if not isinstance(seal, Path):
                raise ReceiptMismatchError("success replay readback seal is missing")
            return {"artifacts": _descriptors(root, matrix=matrix, ledger=ledger, readback_seal=seal), "result": "complete"}
        if stage == "final-publication":
            receipt = root / "reports/final-publication.json"
            readback = root / "reports/final-publication-readback.json"
            _validate_final_publication_artifacts(config, receipt=receipt, readback=readback, inputs=inputs)
            return {"artifacts": _descriptors(root, publication_receipt=receipt, publication_readback=readback)}
        raise RuntimeError(f"fixed {stage} adapter has no canonical output discovery")

    def stop_gpu(self, command: str) -> None:
        if command != _TRUSTED_GPU_STOP:
            raise ValueError("GPU stop hook must be the fixed trusted executable")
        config = self._require_config()
        evidence = _canonical_root(config) / "stage-receipts" / "gpu-stop-observation.json"
        evidence.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            (_TRUSTED_GPU_STOP, "--instance-id", config.rollout_instance_id, "--evidence-path", str(evidence)),
            check=True,
            env={
                "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                "LC_ALL": "C", "LEHOME_GPU_STOP_INSTANCE_ID": config.rollout_instance_id,
                "LEHOME_GPU_STOP_EVIDENCE_PATH": str(evidence),
            },
        )


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


def _verified_stop_observation(config: CollectionConfig) -> tuple[Path, str]:
    """Require a trusted Nebius API observation, not merely stop dispatch."""
    path = _canonical_root(config) / "stage-receipts" / "gpu-stop-observation.json"
    payload = _strict_json_object(path, label="GPU stop observation")
    required = {
        "schema_version", "kind", "provider", "instance_id", "state", "verified",
        "observed_at_utc", "provider_response_sha256",
    }
    if (
        set(payload) != required or payload.get("schema_version") != 1
        or payload.get("kind") != "lehome_simple_curriculum_verified_gpu_stop_v1"
        or payload.get("provider") != "nebius_compute_api"
        or payload.get("instance_id") != config.rollout_instance_id
        or payload.get("state") != "STOPPED" or payload.get("verified") is not True
        or not isinstance(payload.get("observed_at_utc"), str)
        or not isinstance(payload.get("provider_response_sha256"), str)
        or _HEX.fullmatch(str(payload.get("provider_response_sha256"))) is None
    ):
        raise ReceiptMismatchError("GPU stop lacks an authoritative exact-VM STOPPED observation")
    return path, _file_sha(path)


def _validate_final_publication_artifacts(
    config: CollectionConfig, *, receipt: Path, readback: Path, inputs: Mapping[str, object] | None = None,
) -> None:
    published = _strict_json_object(receipt, label="final publication receipt")
    required = {
        "schema_version", "kind", "run_id", "round_id", "terminal_outcome", "repository", "remote_prefix",
        "immutable_revision", "entry_count", "entries", "bundle_sha256", "final_seal_sha256", "readback_verified",
        "public_readback_verified",
    }
    if (
        set(published) != required or published.get("schema_version") != 1
        or published.get("kind") != "lehome_simple_curriculum_publication_receipt_v1"
        or published.get("run_id") != config.run_id or published.get("round_id") != config.round_id
        or published.get("repository") != "ryanjin333/lehome-groot-n17-rollouts"
        or published.get("remote_prefix") != f"collection-rounds/{config.run_id}"
        or not isinstance(published.get("immutable_revision"), str) or re.fullmatch(r"[0-9a-f]{40}", published["immutable_revision"]) is None
        or type(published.get("entry_count")) is not int or int(published["entry_count"]) < 1
        or not isinstance(published.get("entries"), list) or len(published["entries"]) != int(published["entry_count"])
        or any(not isinstance(published.get(key), str) or _HEX.fullmatch(str(published.get(key))) is None for key in ("bundle_sha256", "final_seal_sha256"))
        or published.get("readback_verified") is not True or published.get("public_readback_verified") is not True
    ):
        raise ReceiptMismatchError("final publication receipt is malformed or incomplete")
    manifest: list[tuple[str, str, int]] = []
    for raw in published["entries"]:
        if not isinstance(raw, Mapping) or set(raw) != {"relative_path", "sha256", "byte_size"}:
            raise ReceiptMismatchError("final publication receipt manifest is malformed")
        relative, sha256, byte_size = raw.get("relative_path"), raw.get("sha256"), raw.get("byte_size")
        if (
            not isinstance(relative, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,511}", relative)
            or any(part in {"", ".", ".."} or part.startswith(".") for part in Path(relative).parts)
            or relative.split("/", 1)[0] not in {"manifests", "fresh", "replay", "reports", "seals"}
            or not isinstance(sha256, str) or _HEX.fullmatch(sha256) is None
            or type(byte_size) is not int or byte_size < 0
        ):
            raise ReceiptMismatchError("final publication receipt manifest is malformed")
        manifest.append((relative, sha256, byte_size))
    if manifest != sorted(manifest) or len({relative for relative, _sha, _size in manifest}) != len(manifest):
        raise ReceiptMismatchError("final publication receipt manifest is malformed")
    manifest_payload = [
        {"relative_path": relative, "sha256": sha256, "byte_size": byte_size}
        for relative, sha256, byte_size in manifest
    ]
    if _digest(manifest_payload) != published["bundle_sha256"]:
        raise ReceiptMismatchError("final publication receipt manifest digest mismatch")
    if inputs is not None and published.get("terminal_outcome") != inputs.get("terminal_outcome"):
        raise ReceiptMismatchError("final publication receipt terminal outcome mismatch")
    public = _strict_json_object(readback, label="final publication public readback receipt")
    expected = {
        "schema_version": 1, "kind": "lehome_simple_curriculum_public_readback_receipt_v1",
        "publication_receipt_sha256": _file_sha(receipt), "repository": published["repository"],
        "immutable_revision": published["immutable_revision"], "remote_prefix": published["remote_prefix"],
        "bundle_sha256": published["bundle_sha256"], "authenticated_readback_verified": True,
        "anonymous_readback_verified": True,
    }
    if public != expected:
        raise ReceiptMismatchError("final publication public readback receipt is malformed")


def _authenticated_output(
    stage: str, output: Mapping[str, object], *, config: CollectionConfig, deep: bool = False,
) -> dict[str, object]:
    """Accept only a bounded stage result and hash every referenced byte now."""
    if stage == "gpu-stop":
        allowed = {"terminal_outcome", "stop_status", "stop_error_type", "rollout_instance_id", "verified_stopped", "stop_observation_sha256"}
        if set(output) - allowed or output.get("stop_status") not in {"not_required", "pending", "succeeded", "failed"}:
            raise ReceiptMismatchError("GPU stop state is invalid")
        if not isinstance(output.get("terminal_outcome"), str): raise ReceiptMismatchError("GPU stop state is invalid")
        if "stop_error_type" in output and not isinstance(output["stop_error_type"], str): raise ReceiptMismatchError("GPU stop state is invalid")
        if output.get("stop_status") == "succeeded" and config.paid:
            if (
                output.get("rollout_instance_id") != config.rollout_instance_id
                or output.get("verified_stopped") is not True
                or output.get("stop_observation_sha256") != _verified_stop_observation(config)[1]
            ):
                raise ReceiptMismatchError("GPU stop receipt lacks verified stopped-state evidence")
        return dict(output)
    required = _STAGE_ARTIFACTS.get(stage, frozenset())
    allowed = {"artifacts"}
    if stage == "first-100-gate": allowed.add("decision")
    if stage == "replay-matrix": allowed.add("result")
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
    if stage == "replay-matrix" and "result" in output:
        if output["result"] != "replay_shortage":
            raise ReceiptMismatchError("replay matrix result is invalid")
        result["result"] = "replay_shortage"
    if deep and stage == "fresh-report":
        # A journal receipt for fresh source evidence is not an authority on
        # its own.  Re-run the canonical report/matrix traversal and then
        # reopen all 1,000 terminal artifacts and their accepted Hub receipts
        # before allowing a stopped campaign to resume.
        expected = {
            "report": "reports/fresh-source-report.json",
            "matrix": "reports/fresh-source-matrix.json",
            "terminal_artifact_manifest": "reports/fresh-terminal-artifacts.json",
        }
        if {name: item["path"] for name, item in checked.items()} != expected:
            raise ReceiptMismatchError("fresh report stage output is not canonical")
        try:
            authenticated = _validate_fresh_source_outputs(
                config,
                report=root / expected["report"],
                matrix=root / expected["matrix"],
            )
            _validate_fresh_terminal_artifact_manifest(
                config,
                authenticated=authenticated,
                manifest=root / expected["terminal_artifact_manifest"],
            )
        except (ValueError, OSError) as error:
            raise ReceiptMismatchError("fresh report deep authentication failed") from error
    if deep and stage == "final-publication":
        expected = {
            "publication_receipt": "reports/final-publication.json",
            "publication_readback": "reports/final-publication-readback.json",
        }
        if {name: item["path"] for name, item in checked.items()} != expected:
            raise ReceiptMismatchError("final publication stage output is not canonical")
        _validate_final_publication_artifacts(
            config, receipt=root / expected["publication_receipt"], readback=root / expected["publication_readback"],
        )
    return result


def _verify_authenticated_output(stage: str, output: Mapping[str, object], *, config: CollectionConfig) -> None:
    # Re-run validation against the actual bytes on every restart.  The
    # descriptor itself is part of the receipt hash; this catches deletion,
    # rewrites, root swaps, and changed ledgers without trusting old JSON.
    if _authenticated_output(stage, output, config=config, deep=True) != output:
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

    @property
    def budget_state_path(self) -> Path: return self.directory / "budget-state.json"

    def check_budget(self) -> None:
        if not self.config.paid: return
        observer = self.config.spend_observer
        if observer is None or not observer.is_absolute() or observer.is_symlink() or not observer.is_file():
            raise ReceiptMismatchError("paid spend observer is unavailable")
        try: payload = json.loads(observer.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error: raise ReceiptMismatchError("paid spend observer is malformed") from error
        required = {"schema_version", "kind", "observer", "observed_at_utc", "spent_usd"}
        if (not isinstance(payload, dict) or set(payload) != required
                or payload.get("schema_version") != 1 or payload.get("kind") != "lehome_spend_observation_v1"
                or not isinstance(payload.get("observer"), str) or not payload["observer"]
                or type(payload.get("spent_usd")) not in {int, float}
                or not math.isfinite(float(payload["spent_usd"])) or float(payload["spent_usd"]) < 0):
            raise ReceiptMismatchError("paid spend observer is malformed")
        observed_at = _parse_utc(payload["observed_at_utc"], label="spend observer timestamp")
        now = datetime.now(UTC)
        if observed_at < now - timedelta(minutes=5) or observed_at > now + timedelta(minutes=1):
            raise ReceiptMismatchError("paid spend observer is stale")
        state = {
            "schema_version": 1, "kind": "lehome_simple_curriculum_budget_state_v1",
            "started_at_utc": _format_utc(now), "deadline_at_utc": _format_utc(now + timedelta(seconds=self.config.max_wall_seconds)),
            "observer": payload["observer"], "last_observed_at_utc": _format_utc(observed_at), "last_spent_usd": float(payload["spent_usd"]),
        }
        if self.budget_state_path.exists():
            try: state = json.loads(self.budget_state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error: raise ReceiptMismatchError("budget state is malformed") from error
            if (not isinstance(state, dict) or set(state) != {"schema_version", "kind", "started_at_utc", "deadline_at_utc", "observer", "last_observed_at_utc", "last_spent_usd"}
                    or state.get("schema_version") != 1 or state.get("kind") != "lehome_simple_curriculum_budget_state_v1"
                    or state.get("observer") != payload["observer"]):
                raise ReceiptMismatchError("paid spend observer does not match durable budget state")
            started = _parse_utc(state.get("started_at_utc"), label="budget start timestamp")
            deadline = _parse_utc(state.get("deadline_at_utc"), label="budget deadline")
            if deadline <= started or type(state.get("last_spent_usd")) not in {int, float} or not math.isfinite(float(state["last_spent_usd"])) or float(state["last_spent_usd"]) < 0:
                raise ReceiptMismatchError("durable budget state is malformed")
            previous_observed = _parse_utc(state.get("last_observed_at_utc"), label="durable spend timestamp")
            if observed_at < previous_observed or float(payload["spent_usd"]) < float(state["last_spent_usd"]):
                raise ReceiptMismatchError("paid spend observer regressed or is stale")
            state.update(last_spent_usd=float(payload["spent_usd"]), last_observed_at_utc=_format_utc(observed_at))
        deadline = _parse_utc(state["deadline_at_utc"], label="budget deadline")
        if now >= deadline or float(payload["spent_usd"]) >= self.config.max_spend_usd:
            raise BudgetLimitError("paid budget or wall-time limit reached")
        _replace_durable(self.budget_state_path, _canonical(state))

    def stop_state(self, predecessor: str | None, outcome: str) -> dict[str, object] | None:
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

    def write_stop_state(self, predecessor: str | None, outcome: str, status: str) -> None:
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
    journal.check_budget()
    try:
        existing = journal._read(stage, predecessor, kwargs)
        if existing is not None:
            return existing, str(existing["receipt_sha256"])
        output = runner.run(stage, **kwargs)
        if not isinstance(output, Mapping): raise ValueError("stage runner output is invalid")
        receipt = journal.complete(stage, predecessor, output, inputs=kwargs)
        return receipt, str(receipt["receipt_sha256"])
    finally:
        # Observe the portable budget after all paths, including subprocess,
        # parser, receipt, and artifact-validation failures.
        journal.check_budget()


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


def _stop_terminal(journal: StageJournal, runner: Runner, config: CollectionConfig, *, predecessor: str | None, outcome: str) -> str:
    """Persist exactly-once stop evidence, never retrying an ambiguous stop."""
    stop_inputs = {"terminal_outcome": outcome}
    existing = journal._read("gpu-stop", predecessor, stop_inputs)
    state = journal.stop_state(predecessor, outcome)
    if state is not None and existing is None:
        # A process can crash after writing any durable intent/result state but
        # before the immutable receipt.  The external stop may already have
        # happened, so retrying it would violate exactly-once semantics.
        return "infrastructure_stop_failure"
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
        _observation, observation_sha = _verified_stop_observation(config)
    except Exception as error:
        journal.write_stop_state(predecessor, outcome, "failed")
        journal.complete("gpu-stop", predecessor, {"terminal_outcome": outcome, "stop_status": "failed", "stop_error_type": type(error).__name__}, inputs=stop_inputs)
        return "infrastructure_stop_failure"
    journal.write_stop_state(predecessor, outcome, "succeeded")
    journal.complete("gpu-stop", predecessor, {
        "terminal_outcome": outcome, "stop_status": "succeeded",
        "rollout_instance_id": config.rollout_instance_id, "verified_stopped": True,
        "stop_observation_sha256": observation_sha,
    }, inputs=stop_inputs)
    return str(outcome)


def _stop_then_publish(journal: StageJournal, runner: Runner, config: CollectionConfig, *, predecessor: str | None, outcome: str) -> str:
    """Stop before final publication; never turn a failed publisher into success.

    The public complete seal must bind authoritative stopped-state evidence,
    so final publication is intentionally downstream of the durable stop
    receipt.  A restart can retry only this publication boundary, never the
    paid collection or stop hook.
    """
    final_outcome = _stop_terminal(journal, runner, config, predecessor=predecessor, outcome=outcome)
    raw_stop = _strict_json_object(journal.path("gpu-stop"), label="GPU stop receipt")
    stop_receipt = raw_stop.get("receipt_sha256")
    if not isinstance(stop_receipt, str) or _HEX.fullmatch(stop_receipt) is None:
        raise ReceiptMismatchError("GPU stop receipt is missing its immutable digest")
    if final_outcome != outcome:
        # A failed/ambiguous stop is itself the terminal infrastructure result.
        # Its public seal cannot claim verified stop; do not publish a false
        # completion receipt.
        return final_outcome
    _post_stop_publication(
        journal, runner, predecessor=stop_receipt, terminal_outcome=final_outcome,
    )
    return final_outcome


def _post_stop_publication(
    journal: StageJournal, runner: Runner, *, predecessor: str, terminal_outcome: str,
) -> None:
    """Run only the zero-compute final-publication boundary after STOPPED.

    This intentionally bypasses ``check_budget``.  The caller has already
    written and authenticated the one VM's stop receipt, while the stage
    itself is limited to local sealing plus public Hub upload/readback.  It
    cannot reach a collection, replay, or provider-start command.
    """

    inputs = {"terminal_outcome": terminal_outcome}
    if journal._read("final-publication", predecessor, inputs) is not None:
        return
    output = runner.run("final-publication", **inputs)
    if not isinstance(output, Mapping):
        raise ValueError("stage runner output is invalid")
    journal.complete("final-publication", predecessor, output, inputs=inputs)


def _existing_terminal_outcome(journal: StageJournal) -> str | None:
    """Refuse to run a new paid stage after any durable stop-era state exists."""
    stop_receipt = journal.path("gpu-stop")
    if not stop_receipt.exists():
        if journal.stop_state_path.exists():
            return "infrastructure_stop_failure"
        return None
    try:
        raw = _strict_json_object(stop_receipt, label="GPU stop receipt")
        predecessor = raw.get("predecessor_receipt_sha256")
        output = raw.get("output")
        if not isinstance(output, Mapping) or not isinstance(output.get("terminal_outcome"), str):
            return "infrastructure_stop_failure"
        receipt = journal._read("gpu-stop", predecessor if isinstance(predecessor, str) else None, {"terminal_outcome": output["terminal_outcome"]})
    except (ReceiptMismatchError, ValueError):
        return "infrastructure_stop_failure"
    if receipt is None:
        return "infrastructure_stop_failure"
    _rehash_terminal_journal(journal)
    if output.get("stop_status") not in {"not_required", "succeeded"}:
        return "infrastructure_stop_failure"
    return str(output["terminal_outcome"])


def _rehash_terminal_journal(journal: StageJournal) -> None:
    """A stopped run is resumable only while every immutable output still hashes."""
    for stage in STAGES:
        path = journal.path(stage)
        if not path.exists():
            continue
        receipt = _strict_json_object(path, label="stage receipt")
        body = dict(receipt); declared = body.pop("receipt_sha256", None)
        if declared != _digest(body) or receipt.get("stage") != stage or not isinstance(receipt.get("output"), Mapping):
            raise ReceiptMismatchError("terminal stage journal receipt is malformed")
        _verify_authenticated_output(stage, receipt["output"], config=journal.config)


def run_collection(config: CollectionConfig, *, runner: Runner) -> str:
    """Run/recover the only permitted state machine; returns its data outcome."""
    config.validate()
    if isinstance(runner, CommandRunner):
        _prepare_controller_inputs(config)
    journal = StageJournal(config); predecessor: str | None = None
    if isinstance(runner, CommandRunner):
        runner.budget_check = journal.check_budget
    existing_terminal = _existing_terminal_outcome(journal)
    if existing_terminal is not None:
        # A post-stop process crash can leave the only safe retryable boundary
        # (public readback) unfinished.  Never re-enter collection or stop.
        if existing_terminal == "infrastructure_stop_failure" and not journal.path("gpu-stop").is_file():
            return existing_terminal
        raw_stop = _strict_json_object(journal.path("gpu-stop"), label="GPU stop receipt")
        stop_receipt = raw_stop.get("receipt_sha256")
        if not isinstance(stop_receipt, str) or _HEX.fullmatch(stop_receipt) is None:
            raise ReceiptMismatchError("GPU stop receipt is missing its immutable digest")
        final = journal._read("final-publication", stop_receipt, {"terminal_outcome": existing_terminal})
        if final is None:
            _post_stop_publication(
                journal, runner, predecessor=stop_receipt, terminal_outcome=existing_terminal,
            )
        return existing_terminal
    try:
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
            _, predecessor = _stage(journal, runner, "fresh-report", predecessor)
            replay_matrix, predecessor = _stage(journal, runner, "replay-matrix", predecessor)
            if replay_matrix["output"].get("result") == "replay_shortage":
                outcome = "replay_shortage"
            else:
                replay, predecessor = _stage(journal, runner, "success-replay", predecessor)
                outcome = replay["output"].get("result", "complete")
            if outcome not in {"complete", "replay_shortage"}: raise ReceiptMismatchError("replay result is invalid")
        else:
            outcome = str(decision)
    except BudgetLimitError:
        # A budget observer can fire before the first adapter, between two
        # adapters, while CommandRunner is terminating a live child, or in
        # the post-adapter observation in ``_stage``.  In every paid case the
        # exact rollout VM may therefore still be consuming compute.  Treat
        # this as a terminal infrastructure outcome: dispatch the *trusted*
        # exact-VM stop, require the provider STOPPED observation, and then
        # permit only final-publication (which deliberately has no budget
        # check and no paid command surface).  Do not re-raise here: doing so
        # used to strand an already-running VM on a pre-stage/in-flight spend
        # breach.
        return _stop_then_publish(
            journal, runner, config, predecessor=predecessor,
            outcome="infrastructure_stop",
        )
    except ReceiptMismatchError:
        _stop_then_publish(journal, runner, config, predecessor=predecessor, outcome="infrastructure_stop_failure")
        raise
    except Exception:
        return _stop_then_publish(journal, runner, config, predecessor=predecessor, outcome="infrastructure_stop_failure")
    return _stop_then_publish(journal, runner, config, predecessor=predecessor, outcome=str(outcome))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--host-code-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True); parser.add_argument("--round-id", required=True)
    parser.add_argument("--max-wall-seconds", type=float, required=True); parser.add_argument("--max-spend-usd", type=float, required=True)
    parser.add_argument("--paid", action="store_true"); parser.add_argument("--gpu-stop-command")
    parser.add_argument("--spend-observer", type=Path)
    parser.add_argument("--runtime-identity-json", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        identity = json.loads(args.runtime_identity_json.read_text(encoding="utf-8"))
        config = CollectionConfig(args.campaign_root, args.host_code_root, args.run_id, args.round_id, args.max_wall_seconds, args.max_spend_usd, args.paid, args.gpu_stop_command, identity, args.spend_observer)
        config.validate(require_git=True)
        print(run_collection(config, runner=CommandRunner(config)))
    except (OSError, ValueError, ReceiptMismatchError, StopHookError, RuntimeError) as error:
        _parser().error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
