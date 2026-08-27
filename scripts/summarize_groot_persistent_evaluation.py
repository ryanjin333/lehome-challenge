"""Seal one completed persistent-worker seen-80 or unseen-80 report."""

from __future__ import annotations

import argparse
from io import BytesIO
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import sqlite3
from typing import Mapping
from urllib.parse import quote


_CATEGORIES = ("top_long", "top_short", "pant_long", "pant_short")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SIMULATOR_VERSION = re.compile(r"^[0-9]+(?:\.[0-9]+){1,3}$")
_SAFE_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")
_CANDIDATES = ("original_baseline", "new_step_2k")
_PAIRING_UNAVAILABLE = {"status": "baseline_evaluation_required"}
_PAIRING_AVAILABLE_FIELDS = {
    "status", "baseline_report_sha256", "paired_trials", "candidate_wins",
    "baseline_wins", "ties", "paired_improvement", "progress_improvement",
    "recovery_improvement",
}
_BASELINE_FIELDS = {
    "schema_version", "kind", "matrix_sha256", "report_sha256", "policy_digest",
    "episode_artifacts", "promotion_metrics", "readback_verified", "sealed",
}
_BASELINE_ARTIFACT_FIELDS = {"trial_id", "official_success", "episode_sha256", "worker_receipt_sha256"}


def _runtime_identity_digest(identity: Mapping[str, object], provenance: Mapping[str, object]) -> str:
    """Digest the complete immutable runtime tuple used by the 100-outcome gate."""
    fields = {
        "policy_repo": provenance.get("policy_repo"),
        "policy_revision": provenance.get("policy_revision"),
        "policy_step": provenance.get("policy_step"),
        "policy_artifact_sha256": provenance.get("policy_artifact_sha256"),
        "code_revision": identity.get("code_revision"),
        "asset_revision": identity.get("asset_revision"),
        "simulator_version": identity.get("simulator_version"),
        "image_identity": provenance.get("image_identity"),
        "simulator_device": provenance.get("simulator_device"),
        "cloth_device": provenance.get("cloth_device"),
        "renderer_device": provenance.get("renderer_device"),
        "camera_device": provenance.get("camera_device"),
        "policy_device": provenance.get("policy_device"),
    }
    if (not isinstance(fields["policy_repo"], str) or not fields["policy_repo"] or any(character.isspace() for character in fields["policy_repo"])
            or not isinstance(fields["policy_revision"], str) or _COMMIT.fullmatch(fields["policy_revision"]) is None
            or type(fields["policy_step"]) is not int or fields["policy_step"] != 12000
            or not isinstance(fields["policy_artifact_sha256"], str) or _SHA256.fullmatch(fields["policy_artifact_sha256"]) is None
            or not isinstance(fields["code_revision"], str) or _COMMIT.fullmatch(fields["code_revision"]) is None
            or not isinstance(fields["asset_revision"], str) or _COMMIT.fullmatch(fields["asset_revision"]) is None
            or not isinstance(fields["simulator_version"], str) or _SIMULATOR_VERSION.fullmatch(fields["simulator_version"]) is None
            or not isinstance(fields["image_identity"], str) or re.fullmatch(r"sha256:[0-9a-f]{64}", fields["image_identity"]) is None
            or any(not isinstance(fields[key], str) or not fields[key] for key in ("simulator_device", "cloth_device", "renderer_device", "camera_device", "policy_device"))):
        raise ValueError("evaluation runtime identity is incomplete")
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()


def _augment_first_hundred_metrics(report: Mapping[str, object]) -> dict[str, object]:
    """Add immutable first-100 fields without changing legacy report fields."""
    result = dict(report)
    trials = result.get("gate_trials")
    invalid = result.get("infrastructure_invalid_executions", 0)
    if not isinstance(trials, list) or type(invalid) is not int or invalid < 0:
        raise ValueError("first-100 report evidence is invalid")
    assignment_ids: list[str] = []
    identities: set[str] = set()
    for trial in trials:
        if not isinstance(trial, Mapping):
            raise ValueError("first-100 gate trial is invalid")
        attempt_id = trial.get("assignment_id")
        terminal = trial.get("terminal_event")
        identity = trial.get("identity")
        provenance = trial.get("provenance")
        fidelity = trial.get("fidelity")
        if (not isinstance(attempt_id, str) or not attempt_id or terminal not in {"accepted", "rejected"}
                or not isinstance(identity, Mapping) or not isinstance(provenance, Mapping)
                or not isinstance(fidelity, Mapping)):
            raise ValueError("first-100 gate trial is invalid")
        if any(type(fidelity.get(field)) is not bool for field in ("missing_cloth", "cloth_flight", "nonfinite_cloth_state", "safety_failure")):
            raise ValueError("first-100 fidelity evidence is invalid")
        assignment_ids.append(attempt_id)
        identities.add(_runtime_identity_digest(identity, provenance))
    if len(set(assignment_ids)) != len(assignment_ids):
        raise ValueError("first-100 assignment identities are duplicated")
    result["valid_outcomes"] = len(assignment_ids)
    result["infrastructure_invalid_executions"] = invalid
    result["execution_count"] = len(assignment_ids) + invalid
    result["runtime_identities"] = sorted(identities)
    result["fresh_assignment_ids"] = sorted(assignment_ids)
    return result


def _simple_evidence_indexes(
    root: Path,
) -> tuple[dict[str, tuple[Path, dict[str, object]]], dict[str, tuple[Path, dict[str, object]]], set[tuple[str, str]]]:
    """Index gate evidence without following symlinked files or directories."""
    indexed = {"episode.json": {}, "worker-receipt.json": {}}
    identity_keys = {"episode.json": "episode_id", "worker-receipt.json": "attempt_id"}
    invalid: set[tuple[str, str]] = set()
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in sorted(entries, key=lambda item: item.name):
                    path = Path(entry.path)
                    relative = path.relative_to(root).as_posix()
                    if entry.is_symlink():
                        invalid.add(("unsafe", relative))
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(path)
                        continue
                    if entry.name not in indexed:
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        invalid.add(("unsafe", relative))
                        continue
                    try:
                        payload = _simple_json_object(path, entry.name, campaign_root=root)
                    except ValueError:
                        invalid.add(("malformed", relative))
                        continue
                    identity = payload.get(identity_keys[entry.name])
                    if not isinstance(identity, str) or not identity:
                        invalid.add(("identity", relative))
                        continue
                    target = indexed[entry.name]
                    if identity in target:
                        invalid.add(("duplicate", relative))
                        continue
                    target[identity] = (path, payload)
        except OSError:
            invalid.add(("unsafe", directory.relative_to(root).as_posix()))
    return indexed["episode.json"], indexed["worker-receipt.json"], invalid


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_nonfinite_json(value: object) -> None:
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_nonfinite_json(item)
    elif isinstance(value, list):
        for item in value:
            _reject_nonfinite_json(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON number")


def _simple_evidence_file(path: Path, *, campaign_root: Path, label: str) -> None:
    """Require a regular selected artifact without resolving away symlinks."""
    root = campaign_root.absolute()
    candidate = path.absolute()
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes campaign root") from error
    current = root
    if current.is_symlink() or not current.is_dir():
        raise ValueError(f"{label} campaign root is unsafe")
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"{label} has a symlink ancestor")
    if not current.is_file():
        raise ValueError(f"{label} must be a regular file")


def _simple_json_object(path: Path, label: str, *, campaign_root: Path) -> dict[str, object]:
    _simple_evidence_file(path, campaign_root=campaign_root, label=label)
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda constant: (_ for _ in ()).throw(ValueError(f"non-finite JSON constant: {constant}")),
        )
        _reject_nonfinite_json(value)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is invalid") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _verify_simple_finalized_manifest(destination: Path, *, campaign_root: Path) -> None:
    """Verify the complete JSON checksum manifest emitted by ArtifactFinalizationQueue."""

    manifest_name = "SHA256SUMS.json"
    manifest_path = destination / manifest_name
    manifest = _simple_json_object(manifest_path, manifest_name, campaign_root=campaign_root)
    actual: dict[str, Path] = {}
    pending = [destination]
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as error:
            raise ValueError("finalized artifact directory is unreadable") from error
        for entry in entries:
            path = Path(entry.path)
            if entry.is_symlink():
                raise ValueError("finalized artifact contains a symlink")
            relative = path.relative_to(destination).as_posix()
            if entry.is_dir(follow_symlinks=False):
                pending.append(path)
            elif entry.is_file(follow_symlinks=False):
                if relative != manifest_name:
                    actual[relative] = path
            else:
                raise ValueError("finalized artifact contains non-regular evidence")
    expected: dict[str, tuple[str, int]] = {}
    for relative, entry in manifest.items():
        path = PurePosixPath(relative) if isinstance(relative, str) else None
        if (
            path is None or not relative or path.is_absolute() or "." in path.parts or ".." in path.parts
            or relative == manifest_name or not isinstance(entry, Mapping) or set(entry) != {"sha256", "size"}
            or not isinstance(entry["sha256"], str) or _SHA256.fullmatch(entry["sha256"]) is None
            or type(entry["size"]) is not int or entry["size"] < 0
        ):
            raise ValueError("finalized artifact manifest entry is invalid")
        expected[relative] = (entry["sha256"], entry["size"])
    if set(expected) != set(actual):
        raise ValueError("finalized artifact manifest coverage is invalid")
    for relative, path in actual.items():
        digest, size = expected[relative]
        _simple_evidence_file(path, campaign_root=campaign_root, label="finalized manifest evidence")
        if path.stat().st_size != size or _sha256_file(path) != digest:
            raise ValueError("finalized artifact manifest hash mismatch")


def _simple_fidelity(episode: Mapping[str, object]) -> dict[str, bool]:
    fidelity = episode.get("fidelity")
    fields = ("missing_cloth", "cloth_flight", "nonfinite_cloth_state", "safety_failure")
    if not isinstance(fidelity, Mapping) or any(type(fidelity.get(field)) is not bool for field in fields):
        raise ValueError("simple curriculum episode fidelity evidence is invalid")
    aggregate = episode.get("safety_failure")
    if type(aggregate) is not bool or aggregate is not fidelity["safety_failure"]:
        raise ValueError("simple curriculum safety evidence is contradictory")
    return {field: bool(fidelity[field]) for field in fields}


def _simple_finalized_paths(
    campaign_root: Path, ledger_id: str, terminal_event: str, terminal_payload: Mapping[str, object],
) -> tuple[Path, Path]:
    """Return an exact allow-listed finalizer destination for first-100 evidence."""
    destinations = [campaign_root / "evaluation-terminal" / ledger_id]
    if terminal_event == "accepted":
        destinations.append(campaign_root / "accepted" / ledger_id)
    destination = next(
        (candidate for candidate in destinations if terminal_payload == {"artifact_id": str(candidate)}),
        None,
    )
    if destination is None:
        raise ValueError("simple curriculum terminal artifact is not the finalized destination")
    return destination / "raw" / ledger_id / "episode.json", destination / "worker-receipt.json"


def _simple_raw_output_path(
    campaign_root: Path, *, worker_id: object, session_id: object, ledger_id: str,
    lease_id: object, generation: object,
) -> Path:
    """Reconstruct the immutable persistent-worker output location from its receipt."""
    components = (worker_id, session_id, ledger_id, lease_id)
    if (any(not isinstance(component, str) or _SAFE_PATH_COMPONENT.fullmatch(component) is None
            or component in {".", ".."} for component in components)
            or type(generation) is not int or generation < 1):
        raise ValueError("simple curriculum receipt raw output identity is invalid")
    return campaign_root.joinpath(*components, f"generation-{generation}")


def _terminal_invalid_key(ledger_id: str, pending_evidence: tuple[str, str, str]) -> tuple[str, str, str, str]:
    lease_id, _worker_id, raw_artifact_id = pending_evidence
    return ("terminal-evidence", ledger_id, lease_id, raw_artifact_id)


def _build_simple_first_hundred_report(
    *, campaign_root: Path, rows: list[dict[str, object]], matrix_sha256: str, candidate_key: str,
    policy_repo: str, policy_revision: str, policy_step: int, policy_artifact_sha256: str,
) -> dict[str, object]:
    """Summarize only terminal evidence that can authenticate the 100-row gate."""
    ledger_path = campaign_root / "ledger.sqlite3"
    with _open_ledger(ledger_path) as ledger:
        attempts = list(ledger.execute("SELECT attempt_id, schedule_index, assignment_json FROM attempts ORDER BY schedule_index"))
        if len(attempts) != len(rows):
            raise ValueError("evaluation ledger does not contain exactly the frozen-matrix attempts")
        assignments: dict[str, dict[str, object]] = {}
        for index, (attempt, expected) in enumerate(zip(attempts, rows)):
            try:
                assignment = json.loads(attempt["assignment_json"])
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError("evaluation ledger assignment is invalid") from error
            if attempt["schedule_index"] != index or assignment != expected:
                raise ValueError("evaluation ledger assignments do not match the frozen matrix")
            assignments[str(attempt["attempt_id"])] = assignment
        terminal: dict[str, tuple[str, dict[str, object]]] = {}
        pending: dict[str, tuple[str, str, str]] = {}
        completed_at_ns = 0
        invalid_executions: set[tuple[object, ...]] = set()
        infrastructure_by_lease: dict[tuple[str, str], set[tuple[object, ...]]] = defaultdict(set)
        infrastructure_without_lease: list[tuple[str, tuple[object, ...]]] = []
        gate_fidelity_failures: list[dict[str, object]] = []
        for event in ledger.execute("SELECT event_id, at_ns, event_type, attempt_id, lease_id, worker_id, payload_json FROM events ORDER BY event_id"):
            event_type = event["event_type"]
            ledger_id = event["attempt_id"]
            if event_type in {"infrastructure_abort", "retryable", "lease_expired", "preempted", "interrupted"}:
                key = (
                    "infrastructure-execution", ledger_id,
                    event["lease_id"] if isinstance(event["lease_id"], str) else event["event_id"],
                )
                invalid_executions.add(key)
                if isinstance(ledger_id, str) and isinstance(event["lease_id"], str):
                    infrastructure_by_lease[(ledger_id, event["lease_id"])].add(key)
                elif isinstance(ledger_id, str):
                    infrastructure_without_lease.append((ledger_id, key))
                if event_type == "infrastructure_abort":
                    try:
                        payload = json.loads(event["payload_json"])
                    except (TypeError, json.JSONDecodeError):
                        payload = None
                    fields = {
                        "failure_class", "fidelity_code", "fidelity", "lease_id", "worker_id",
                        "session_id", "generation", "runtime",
                    }
                    fidelity_fields = {
                        "missing_cloth", "cloth_flight", "nonfinite_cloth_state", "safety_failure",
                        "monitor_active", "monitor_observed",
                    }
                    runtime_fields = {
                        "simulation_device", "cloth_device", "renderer_device", "camera_device", "policy_device",
                    }
                    if (
                        isinstance(payload, dict) and set(payload) == fields
                        and isinstance(ledger_id, str) and ledger_id in assignments
                        and payload.get("failure_class") == "fidelity"
                        and isinstance(payload.get("fidelity_code"), str)
                        and payload["fidelity_code"] in fidelity_fields
                        and payload.get("lease_id") == event["lease_id"]
                        and payload.get("worker_id") == event["worker_id"]
                        and isinstance(payload.get("session_id"), str) and payload["session_id"]
                        and type(payload.get("generation")) is int and payload["generation"] >= 1
                        and isinstance(payload.get("fidelity"), dict) and set(payload["fidelity"]) == fidelity_fields
                        and all(type(payload["fidelity"][field]) is bool for field in fidelity_fields)
                        and payload["fidelity"][payload["fidelity_code"]] is True
                        and payload["fidelity"]["monitor_active"] is True
                        and payload["fidelity"]["monitor_observed"] is True
                        and isinstance(payload.get("runtime"), dict) and set(payload["runtime"]) == runtime_fields
                        and all(isinstance(payload["runtime"][field], str) and payload["runtime"][field] for field in runtime_fields)
                    ):
                        gate_fidelity_failures.append({
                            "assignment_id": assignments[ledger_id]["attempt_id"],
                            "ledger_id": ledger_id,
                            "lease_id": payload["lease_id"], "worker_id": payload["worker_id"],
                            "session_id": payload["session_id"], "generation": payload["generation"],
                            "fidelity_code": payload["fidelity_code"], "fidelity": payload["fidelity"],
                            "runtime": payload["runtime"],
                        })
            elif event_type == "terminal_pending_validation":
                try:
                    payload = json.loads(event["payload_json"])
                except (TypeError, json.JSONDecodeError) as error:
                    raise ValueError("evaluation terminal payload is invalid") from error
                if (ledger_id not in assignments or ledger_id in pending or not isinstance(payload, dict)
                        or not isinstance(payload.get("raw_artifact_id"), str) or not payload["raw_artifact_id"]
                        or not isinstance(event["lease_id"], str) or not event["lease_id"]
                        or not isinstance(event["worker_id"], str) or not event["worker_id"]):
                    raise ValueError("evaluation terminal evidence is ambiguous")
                pending[str(ledger_id)] = (str(event["lease_id"]), str(event["worker_id"]), payload["raw_artifact_id"])
            elif event_type in {"accepted", "rejected"}:
                if ledger_id not in assignments or ledger_id in terminal:
                    raise ValueError("evaluation ledger contains ambiguous terminal outcomes")
                try:
                    payload = json.loads(event["payload_json"])
                except (TypeError, json.JSONDecodeError) as error:
                    raise ValueError("evaluation terminal payload is invalid") from error
                if not isinstance(payload, dict):
                    raise ValueError("evaluation terminal payload is invalid")
                terminal[str(ledger_id)] = (str(event_type), payload)
                completed_at_ns = max(completed_at_ns, int(event["at_ns"]))
        for ledger_id, key in infrastructure_without_lease:
            pending_evidence = pending.get(ledger_id)
            if pending_evidence is not None:
                infrastructure_by_lease[(ledger_id, pending_evidence[0])].add(key)
    episodes, receipts, invalid_evidence = _simple_evidence_indexes(campaign_root)
    evidence_execution_keys: dict[str, tuple[object, ...]] = {}

    def evidence_key_for_pending(ledger_id: str, pending_evidence: tuple[str, str, str]) -> tuple[object, ...]:
        matching = infrastructure_by_lease.get((ledger_id, pending_evidence[0]), set())
        if len(matching) == 1:
            return next(iter(matching))
        return _terminal_invalid_key(ledger_id, pending_evidence)

    def register_evidence_path(path: Path, key: tuple[object, ...]) -> None:
        try:
            relative = path.absolute().relative_to(campaign_root.absolute())
        except ValueError:
            return
        if any(component in {".", ".."} for component in relative.parts):
            return
        evidence_execution_keys[relative.as_posix()] = key

    for ledger_id, pending_evidence in pending.items():
        raw_root = Path(pending_evidence[2])
        pending_key = evidence_key_for_pending(ledger_id, pending_evidence)
        register_evidence_path(raw_root, pending_key)
        register_evidence_path(raw_root / "worker-receipt.json", pending_key)
        register_evidence_path(raw_root / "raw", pending_key)
        register_evidence_path(raw_root / "raw" / ledger_id, pending_key)
        register_evidence_path(raw_root / "raw" / ledger_id / "episode.json", pending_key)
        terminal_evidence = terminal.get(ledger_id)
        if terminal_evidence is None:
            continue
        try:
            episode_path, receipt_path = _simple_finalized_paths(
                campaign_root, ledger_id, terminal_evidence[0], terminal_evidence[1],
            )
        except ValueError:
            continue
        key = _terminal_invalid_key(ledger_id, pending_evidence)
        register_evidence_path(receipt_path.parent, key)
        register_evidence_path(episode_path.parent, key)
        register_evidence_path(episode_path.parent.parent, key)
        register_evidence_path(episode_path, key)
        register_evidence_path(receipt_path, key)
    def invalid_evidence_key(reason: str, relative: str) -> tuple[object, ...]:
        parts = PurePosixPath(relative).parts
        for length in range(len(parts), 0, -1):
            bound = evidence_execution_keys.get(PurePosixPath(*parts[:length]).as_posix())
            if bound is not None:
                return bound
        return ("evidence", reason, relative)

    invalid_evidence_keys = {
        invalid_evidence_key(reason, relative) for reason, relative in invalid_evidence
    }
    invalid_executions.update(invalid_evidence_keys)
    invalid_executions.update(("unbound-episode", ledger_id) for ledger_id in episodes if ledger_id not in assignments)
    invalid_executions.update(("unbound-receipt", ledger_id) for ledger_id in receipts if ledger_id not in assignments)
    invalid_terminal_attempts = {
        key[1] for key in invalid_evidence_keys
        if len(key) >= 2 and key[0] == "terminal-evidence" and isinstance(key[1], str)
    }
    gate_trials: list[dict[str, object]] = []
    trials: list[dict[str, object]] = []
    category_scores = {category: {"episodes": 0, "official_successes": 0} for category in _CATEGORIES}
    garment_scores: dict[str, dict[str, int]] = defaultdict(lambda: {"episodes": 0, "official_successes": 0})
    safety_failure = False
    for ledger_id, assignment in assignments.items():
        terminal_evidence = terminal.get(ledger_id)
        pending_evidence = pending.get(ledger_id)
        if terminal_evidence is None or pending_evidence is None:
            has_infrastructure_event = (
                pending_evidence is not None
                and bool(infrastructure_by_lease.get((ledger_id, pending_evidence[0])))
            )
            if ledger_id not in invalid_terminal_attempts and not has_infrastructure_event:
                invalid_executions.add(("incomplete-terminal", ledger_id))
            continue
        try:
            event, terminal_payload = terminal_evidence
            terminal_lease, terminal_worker, raw_artifact_id = pending_evidence
            expected_episode_path, expected_receipt_path = _simple_finalized_paths(
                campaign_root, ledger_id, event, terminal_payload,
            )
            _verify_simple_finalized_manifest(
                expected_episode_path.parents[2], campaign_root=campaign_root,
            )
            episode = _simple_json_object(expected_episode_path, "episode.json", campaign_root=campaign_root)
            receipt = _simple_json_object(expected_receipt_path, "worker-receipt.json", campaign_root=campaign_root)
            episode_path, receipt_path = expected_episode_path, expected_receipt_path
            identity = episode.get("identity")
            provenance = episode.get("provenance")
            if not isinstance(identity, Mapping) or not isinstance(provenance, Mapping):
                raise ValueError("identity")
            if ({key: identity.get(key) for key in ("policy_repo", "policy_revision", "policy_step")} != {
                    "policy_repo": policy_repo, "policy_revision": policy_revision, "policy_step": policy_step}
                    or provenance.get("policy_artifact_sha256") != policy_artifact_sha256):
                raise ValueError("policy")
            if (identity.get("episode_id") != ledger_id or identity.get("garment_name") != assignment.get("garment_name")
                    or identity.get("category") != assignment.get("category") or identity.get("release_stage") != assignment.get("release_stage")
                    or identity.get("seed") != assignment.get("seed")):
                raise ValueError("assignment")
            fidelity = _simple_fidelity(episode)
            receipt_outcome = receipt.get("outcome")
            official_success = event == "accepted"
            if (receipt.get("schema_version") != 1 or receipt.get("attempt_id") != ledger_id
                    or receipt.get("lease_id") != terminal_lease or receipt.get("worker_id") != terminal_worker
                    or not isinstance(receipt.get("session_id"), str) or not receipt["session_id"]
                    or receipt.get("seed") != assignment.get("seed") or receipt.get("garment") != assignment.get("garment_name")
                    or episode_path != expected_episode_path or receipt_path != expected_receipt_path
                    or receipt.get("output_dir") != str(_simple_raw_output_path(
                        campaign_root, worker_id=terminal_worker, session_id=receipt.get("session_id"),
                        ledger_id=ledger_id, lease_id=terminal_lease,
                        generation=receipt.get("episode_generation"),
                    ))
                    or raw_artifact_id != receipt.get("output_dir")
                    or not isinstance(receipt_outcome, Mapping) or receipt_outcome.get("success") is not official_success
                    or episode.get("accepted_success") is not official_success
                    or (episode.get("outcome") == "success") is not official_success):
                raise ValueError("outcome")
            simulator_device = receipt.get("simulation_device", provenance.get("simulator_device"))
            cloth_device = receipt.get("cloth_device")
            renderer_device = receipt.get("renderer_device")
            camera_device = receipt.get("camera_device")
            policy_device = receipt.get("policy_device", provenance.get("policy_device"))
            runtime = receipt.get("runtime")
            if (simulator_device != "cpu" or cloth_device != "cpu"
                    or not all(isinstance(device, str) and re.fullmatch(r"cuda:[0-9]+", device) for device in (renderer_device, camera_device, policy_device))
                    or len({renderer_device, camera_device, policy_device}) != 1
                    or not isinstance(runtime, Mapping)
                    or any(runtime.get(key) != value for key, value in {
                        "simulation_device": simulator_device, "cloth_device": cloth_device, "renderer_device": renderer_device,
                        "camera_device": camera_device, "policy_device": policy_device,
                    }.items())):
                raise ValueError("devices")
            gate_identity = {
                "policy_repo": policy_repo, "policy_revision": policy_revision, "policy_step": policy_step,
                "policy_artifact_sha256": policy_artifact_sha256, "code_revision": identity.get("code_revision"),
                "asset_revision": identity.get("asset_revision"), "simulator_version": identity.get("simulator_version"),
            }
            gate_provenance = {
                "policy_repo": policy_repo, "policy_revision": policy_revision, "policy_step": policy_step,
                "policy_artifact_sha256": policy_artifact_sha256, "image_identity": provenance.get("image_identity"),
                "simulator_device": simulator_device, "cloth_device": cloth_device, "renderer_device": renderer_device,
                "camera_device": camera_device, "policy_device": policy_device,
            }
            _runtime_identity_digest(gate_identity, gate_provenance)
        except (TypeError, ValueError):
            invalid_executions.add(_terminal_invalid_key(ledger_id, pending_evidence))
            continue
        category = str(assignment["category"]); garment = str(assignment["garment_name"])
        category_scores[category]["episodes"] += 1; garment_scores[garment]["episodes"] += 1
        if official_success:
            category_scores[category]["official_successes"] += 1; garment_scores[garment]["official_successes"] += 1
        safety_failure = safety_failure or fidelity["safety_failure"]
        trials.append({
            "schedule_index": len(trials), "trial_id": assignment["trial_id"], "attempt_id": ledger_id,
            "category": category, "garment": garment, "seed": assignment["seed"],
            "official_success": int(official_success), "terminal_event": event,
            "episode_sha256": _sha256_file(episode_path), "worker_receipt_sha256": _sha256_file(receipt_path),
        })
        gate_trials.append({
            "assignment_id": assignment["attempt_id"], "trial_id": assignment["trial_id"], "terminal_event": event,
            "official_success": official_success, "episode_sha256": _sha256_file(episode_path),
            "worker_receipt_sha256": _sha256_file(receipt_path), "identity": gate_identity,
            "provenance": gate_provenance, "fidelity": fidelity,
        })
    successes = sum(values["official_successes"] for values in category_scores.values())
    report: dict[str, object] = {
        "schema_version": 1, "kind": "lehome_simple_curriculum_first100_report_v1",
        "release_stage": "seen", "candidate_key": candidate_key,
        "identity": {"policy_repo": policy_repo, "policy_revision": policy_revision, "policy_step": policy_step, "policy_artifact_sha256": policy_artifact_sha256},
        "campaign_kind": "simple_curriculum_source_v1", "logical_stage": "calibration_head",
        "matrix_sha256": matrix_sha256, "ledger_sha256": _sha256_file(ledger_path), "completed_at_ns": completed_at_ns,
        "episodes": len(rows), "official_successes": successes, "success_rate": successes / len(gate_trials) if gate_trials else 0.0,
        "per_category": {category: {**values, "success_rate": values["official_successes"] / values["episodes"] if values["episodes"] else 0.0} for category, values in category_scores.items()},
        "per_garment": {garment: {**values, "success_rate": values["official_successes"] / values["episodes"] if values["episodes"] else 0.0} for garment, values in sorted(garment_scores.items())},
        "gates": {"overall_ge_70": False, "each_category_ge_60": False}, "safety": {"evaluated": False, "physical_approval": False},
        "infrastructure_retry_count": len(invalid_executions), "gpu_seconds": 0.0,
        "progress": {"observed_episodes": 0, "mean_terminal_progress": 0.0}, "recovery": {"recovery_attempts": 0, "successful_recoveries": 0},
        "safety_failure": safety_failure, "trials": trials, "gate_trials": gate_trials,
        "gate_fidelity_failures": sorted(gate_fidelity_failures, key=lambda item: (
            str(item["assignment_id"]), str(item["lease_id"]), int(item["generation"]),
        )),
        "infrastructure_invalid_executions": len(invalid_executions),
    }
    report = _augment_first_hundred_metrics(report)
    report["report_sha256"] = report_sha256(report)
    return report


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"evaluation evidence must be a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular JSON file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _load_matrix(path: Path, expected_sha256: str) -> list[dict[str, object]]:
    if not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("matrix SHA-256 is invalid")
    if _sha256_file(path) != expected_sha256:
        raise ValueError("matrix SHA-256 mismatch")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("evaluation matrix is invalid") from error
    if not isinstance(value, list) or not value or len(value) % len(_CATEGORIES) or not all(isinstance(item, dict) for item in value):
        raise ValueError("evaluation matrix must contain an equal nonzero count for all categories")
    rows = [dict(item) for item in value]
    counts = Counter(item.get("category") for item in rows)
    if set(counts) != set(_CATEGORIES) or len(set(counts.values())) != 1:
        raise ValueError("evaluation matrix must contain an equal count for all categories")
    trial_ids = [item.get("trial_id") for item in rows]
    if not all(isinstance(item, str) and item for item in trial_ids) or len(set(trial_ids)) != len(rows):
        raise ValueError("evaluation matrix trial IDs are invalid")
    release_stages = {item.get("release_stage") for item in rows}
    if len(release_stages) != 1 or not release_stages <= {"seen", "public_unseen"}:
        raise ValueError("evaluation matrix must use one supported release stage")
    return rows


def _validate_policy_inputs(
    candidate_key: str,
    policy_repo: str,
    policy_revision: str,
    policy_step: int,
    policy_artifact_sha256: str,
) -> None:
    if candidate_key not in _CANDIDATES:
        raise ValueError("candidate key is not approved")
    if not isinstance(policy_repo, str) or not policy_repo or any(character.isspace() for character in policy_repo):
        raise ValueError("policy repository is invalid")
    if not isinstance(policy_revision, str) or _COMMIT.fullmatch(policy_revision) is None:
        raise ValueError("policy revision must be an immutable commit")
    if type(policy_step) is not int or policy_step <= 0:
        raise ValueError("policy step must be a positive integer")
    if not isinstance(policy_artifact_sha256, str) or _SHA256.fullmatch(policy_artifact_sha256) is None:
        raise ValueError("policy artifact SHA-256 is invalid")


def _open_ledger(path: Path) -> sqlite3.Connection:
    if path.is_symlink() or not path.is_file():
        raise ValueError("evaluation ledger must be a regular file")
    # Read-only mode still follows a persisted WAL. ``immutable=1`` would
    # silently ignore terminal events that have not been checkpointed yet.
    connection = sqlite3.connect(f"file:{quote(str(path))}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _indexed_json_files(root: Path, name: str, id_key: str) -> dict[str, tuple[Path, dict[str, object]]]:
    indexed: dict[str, tuple[Path, dict[str, object]]] = {}
    for path in root.rglob(name):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"evaluation {name} evidence is unsafe")
        payload = _json_object(path, name)
        identifier = payload.get(id_key)
        if not isinstance(identifier, str) or not identifier or identifier in indexed:
            raise ValueError(f"evaluation {name} identities are invalid or duplicated")
        indexed[identifier] = (path, payload)
    return indexed


def report_sha256(report: Mapping[str, object]) -> str:
    payload = dict(report)
    payload.pop("report_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_sha256(value: Mapping[str, object], *, omitted: str = "report_sha256") -> str:
    body = dict(value)
    body.pop(omitted, None)
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be SHA-256")
    return value


def _finite(value: object, label: str, *, nonnegative: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)) or (nonnegative and float(value) < 0):
        raise ValueError(f"{label} is invalid")
    return float(value)


def _paired_metrics(
    legacy: Mapping[str, object],
    baseline_evidence: Mapping[str, object] | None,
    *,
    expected_policy_digest: str,
) -> dict[str, object]:
    """Compute paired ranking evidence, or name the missing baseline explicitly.

    An absent 12K reference is never encoded as ``0.0``.  That distinction is
    important because zero is a real tie, while a missing baseline must block
    promotion until the exact frozen-matrix evidence is supplied.
    """
    if baseline_evidence is None:
        return dict(_PAIRING_UNAVAILABLE)
    baseline = dict(baseline_evidence)
    if set(baseline) != _BASELINE_FIELDS or baseline.get("schema_version") != 1 or baseline.get("kind") != "lehome_experiment_paired_unseen20_baseline":
        raise ValueError("paired baseline evidence schema is invalid")
    if baseline.get("readback_verified") is not True or baseline.get("sealed") is not True:
        raise ValueError("paired baseline evidence is not sealed/read-back verified")
    if baseline.get("matrix_sha256") != legacy.get("matrix_sha256"):
        raise ValueError("paired baseline does not use the candidate matrix")
    baseline_digest = _digest(baseline.get("report_sha256"), "paired baseline report")
    if baseline_digest != _canonical_sha256(baseline):
        raise ValueError("paired baseline report digest mismatch")
    baseline_policy_digest = _digest(baseline.get("policy_digest"), "paired baseline policy")
    if baseline_policy_digest != expected_policy_digest:
        raise ValueError("paired baseline policy does not match the pinned original parent")
    raw_trials = baseline.get("episode_artifacts")
    if not isinstance(raw_trials, list) or not raw_trials:
        raise ValueError("paired baseline episode evidence is invalid")
    outcomes: dict[str, int] = {}
    for artifact in raw_trials:
        if not isinstance(artifact, Mapping) or set(artifact) != _BASELINE_ARTIFACT_FIELDS:
            raise ValueError("paired baseline episode evidence schema is invalid")
        trial = artifact.get("trial_id")
        if type(trial) is not str or not trial or trial in outcomes or type(artifact.get("official_success")) is not int or artifact["official_success"] not in (0, 1):
            raise ValueError("paired baseline episode identity is invalid")
        _digest(artifact.get("episode_sha256"), "paired baseline episode")
        _digest(artifact.get("worker_receipt_sha256"), "paired baseline worker receipt")
        outcomes[trial] = int(artifact["official_success"])
    candidate_trials = legacy.get("trials")
    if not isinstance(candidate_trials, list):
        raise ValueError("candidate trials are invalid")
    if {item.get("trial_id") for item in candidate_trials if isinstance(item, Mapping)} != set(outcomes):
        raise ValueError("paired baseline trial identities do not match candidate")
    candidate_wins = baseline_wins = ties = 0
    for trial in candidate_trials:
        if not isinstance(trial, Mapping) or type(trial.get("trial_id")) is not str or type(trial.get("official_success")) is not int:
            raise ValueError("candidate paired trial is invalid")
        candidate_success = int(trial["official_success"])
        baseline_success = outcomes[trial["trial_id"]]
        if candidate_success > baseline_success:
            candidate_wins += 1
        elif candidate_success < baseline_success:
            baseline_wins += 1
        else:
            ties += 1
    metrics = baseline.get("promotion_metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != {"progress", "recovery"}:
        raise ValueError("paired baseline aggregate metrics are invalid")
    progress = metrics.get("progress")
    recovery = metrics.get("recovery")
    if not isinstance(progress, Mapping) or set(progress) != {"observed_episodes", "mean_terminal_progress"} or type(progress.get("observed_episodes")) is not int or progress["observed_episodes"] < 0:
        raise ValueError("paired baseline progress evidence is invalid")
    baseline_progress = _finite(progress.get("mean_terminal_progress"), "paired baseline terminal progress", nonnegative=True)
    if baseline_progress > 1:
        raise ValueError("paired baseline terminal progress is invalid")
    if not isinstance(recovery, Mapping) or set(recovery) != {"recovery_attempts", "successful_recoveries"} or type(recovery.get("recovery_attempts")) is not int or type(recovery.get("successful_recoveries")) is not int or not 0 <= recovery["successful_recoveries"] <= recovery["recovery_attempts"]:
        raise ValueError("paired baseline recovery evidence is invalid")
    baseline_recovery = (recovery["successful_recoveries"] / recovery["recovery_attempts"]) if recovery["recovery_attempts"] else 0.0
    candidate_progress = legacy.get("progress")
    candidate_recovery = legacy.get("recovery")
    if not isinstance(candidate_progress, Mapping) or not isinstance(candidate_recovery, Mapping):
        raise ValueError("candidate aggregate metrics are invalid")
    candidate_progress_value = _finite(candidate_progress.get("mean_terminal_progress"), "candidate terminal progress", nonnegative=True)
    attempts = candidate_recovery.get("recovery_attempts")
    recoveries = candidate_recovery.get("successful_recoveries")
    if type(attempts) is not int or type(recoveries) is not int or not 0 <= recoveries <= attempts:
        raise ValueError("candidate recovery evidence is invalid")
    candidate_recovery_value = recoveries / attempts if attempts else 0.0
    paired_trials = len(candidate_trials)
    return {
        "status": "available",
        "baseline_report_sha256": baseline_digest,
        "paired_trials": paired_trials,
        "candidate_wins": candidate_wins,
        "baseline_wins": baseline_wins,
        "ties": ties,
        "paired_improvement": (candidate_wins - baseline_wins) / paired_trials,
        "progress_improvement": candidate_progress_value - baseline_progress,
        "recovery_improvement": candidate_recovery_value - baseline_recovery,
    }


def build_paired_baseline_evidence(
    *,
    campaign_root: Path,
    matrix_path: Path,
    matrix_sha256: str,
    policy_repo: str,
    policy_revision: str,
    policy_step: int,
    policy_artifact_sha256: str,
) -> dict[str, object]:
    """Seal one original-12K (or explicitly supplied) paired baseline run."""
    legacy = build_report(
        campaign_root=campaign_root,
        matrix_path=matrix_path,
        matrix_sha256=matrix_sha256,
        candidate_key="original_baseline",
        policy_repo=policy_repo,
        policy_revision=policy_revision,
        policy_step=policy_step,
        policy_artifact_sha256=policy_artifact_sha256,
    )
    # Do not allow a locally sealed evaluator record to stand in for a Hub
    # receipt.  It shares the final-80 terminal receipt gate even though its
    # output shape is intentionally smaller.
    _terminal_sync_artifacts(campaign_root=Path(campaign_root), legacy=legacy)
    evidence: dict[str, object] = {
        "schema_version": 1,
        "kind": "lehome_experiment_paired_unseen20_baseline",
        "matrix_sha256": matrix_sha256,
        "policy_digest": policy_artifact_sha256,
        "episode_artifacts": [
            {
                "trial_id": trial["trial_id"],
                "official_success": trial["official_success"],
                "episode_sha256": trial["episode_sha256"],
                "worker_receipt_sha256": trial["worker_receipt_sha256"],
            }
            for trial in legacy["trials"]
        ],
        "promotion_metrics": {"progress": legacy["progress"], "recovery": legacy["recovery"]},
        "readback_verified": True,
        "sealed": True,
    }
    evidence["report_sha256"] = _canonical_sha256(evidence)
    return evidence


def build_report(
    *,
    campaign_root: Path,
    matrix_path: Path,
    matrix_sha256: str,
    candidate_key: str,
    policy_repo: str,
    policy_revision: str,
    policy_step: int,
    policy_artifact_sha256: str,
) -> dict[str, object]:
    """Verify all 80 terminal artifacts and return a self-hashed score report."""

    _validate_policy_inputs(candidate_key, policy_repo, policy_revision, policy_step, policy_artifact_sha256)
    root = Path(campaign_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("campaign root must be a materialized directory")
    rows = _load_matrix(Path(matrix_path), matrix_sha256)
    simple_first_hundred = (
        len(rows) == 100
        and all(row.get("campaign_kind") == "simple_curriculum_source_v1" for row in rows)
        and all(row.get("logical_stage") == "calibration" for row in rows)
    )
    if simple_first_hundred:
        return _build_simple_first_hundred_report(
            campaign_root=root, rows=rows, matrix_sha256=matrix_sha256,
            candidate_key=candidate_key, policy_repo=policy_repo, policy_revision=policy_revision,
            policy_step=policy_step, policy_artifact_sha256=policy_artifact_sha256,
        )
    ledger_path = root / "ledger.sqlite3"
    with _open_ledger(ledger_path) as ledger:
        attempts = list(ledger.execute("SELECT attempt_id, schedule_index, assignment_json FROM attempts ORDER BY schedule_index"))
        if len(attempts) != len(rows):
            raise ValueError("evaluation ledger does not contain exactly the frozen-matrix attempts")
        attempt_ids: list[str] = []
        for index, (attempt, expected) in enumerate(zip(attempts, rows)):
            try:
                assignment = json.loads(attempt["assignment_json"])
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError("evaluation ledger assignment is invalid") from error
            if attempt["schedule_index"] != index or assignment != expected:
                raise ValueError("evaluation ledger assignments do not match the frozen matrix")
            attempt_ids.append(str(attempt["attempt_id"]))
        terminal_by_attempt: dict[str, str] = {}
        terminal_at_ns: dict[str, int] = {}
        infrastructure_retry_count = 0
        gpu_seconds = 0.0
        for event in ledger.execute("SELECT at_ns, event_type, attempt_id, payload_json FROM events ORDER BY event_id"):
            if event["event_type"] in {"infrastructure_abort", "retryable", "lease_expired", "preempted"}:
                infrastructure_retry_count += 1
                try:
                    payload = json.loads(event["payload_json"] or "{}")
                except (TypeError, json.JSONDecodeError) as error:
                    raise ValueError("evaluation infrastructure event payload is invalid") from error
                value = payload.get("gpu_seconds", 0.0) if isinstance(payload, dict) else 0.0
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError("evaluation infrastructure GPU seconds are invalid")
                gpu_seconds += float(value)
                continue
            if event["event_type"] in {"accepted", "rejected"}:
                attempt_id = str(event["attempt_id"])
                if attempt_id in terminal_by_attempt:
                    raise ValueError("evaluation ledger contains duplicate terminal outcomes")
                terminal_by_attempt[attempt_id] = str(event["event_type"])
                terminal_at_ns[attempt_id] = int(event["at_ns"])
    if set(terminal_by_attempt) != set(attempt_ids):
        raise ValueError("evaluation ledger is missing terminal outcomes")

    episodes = _indexed_json_files(root, "episode.json", "episode_id")
    receipts = _indexed_json_files(root, "worker-receipt.json", "attempt_id")
    if set(episodes) != set(attempt_ids) or set(receipts) != set(attempt_ids):
        raise ValueError("evaluation terminal artifact coverage is incomplete")

    category_scores = {category: {"episodes": 0, "official_successes": 0} for category in _CATEGORIES}
    garment_scores: dict[str, dict[str, int]] = defaultdict(lambda: {"episodes": 0, "official_successes": 0})
    trials: list[dict[str, object]] = []
    code_revisions: set[str] = set()
    asset_revisions: set[str] = set()
    simulator_versions: set[str] = set()
    image_identities: set[str] = set()
    progress_total = 0.0
    progress_observed = 0
    recovery_attempts = 0
    successful_recoveries = 0
    safety_failure = False
    for attempt_id, assignment in zip(attempt_ids, rows):
        episode_path, episode = episodes[attempt_id]
        receipt_path, receipt = receipts[attempt_id]
        identity = episode.get("identity")
        provenance = episode.get("provenance")
        if not isinstance(identity, dict) or not isinstance(provenance, dict):
            raise ValueError("evaluation episode lacks policy identity")
        if (
            identity.get("policy_repo") != policy_repo
            or identity.get("policy_revision") != policy_revision
            or identity.get("policy_step") != policy_step
            or provenance.get("policy_artifact_sha256") != policy_artifact_sha256
        ):
            raise ValueError("evaluation episode policy identity does not match the served checkpoint")
        if (
            identity.get("episode_id") != attempt_id
            or identity.get("garment_name") != assignment.get("garment_name")
            or identity.get("category") != assignment.get("category")
            or identity.get("release_stage") != assignment.get("release_stage")
            or identity.get("seed") != assignment.get("seed")
        ):
            raise ValueError("evaluation episode identity does not match the frozen assignment")
        simulator_device = provenance.get("simulator_device")
        policy_device = provenance.get("policy_device")
        canonical_policy_device = (
            isinstance(policy_device, str)
            and re.fullmatch(r"cuda:[0-9]+", policy_device) is not None
        )
        legacy_cuda_pair = (
            isinstance(simulator_device, str)
            and re.fullmatch(r"cuda:[0-9]+", simulator_device) is not None
            and policy_device == simulator_device
        )
        cpu_cloth_with_cuda_policy = simulator_device == "cpu" and canonical_policy_device
        if not (legacy_cuda_pair or cpu_cloth_with_cuda_policy):
            raise ValueError("evaluation episode device provenance is invalid")
        accepted_success = episode.get("accepted_success")
        official_success = accepted_success is True and episode.get("outcome") == "success"
        if type(accepted_success) is not bool:
            raise ValueError("evaluation episode success field is invalid")
        metrics = episode.get("metrics", {})
        if metrics is not None and not isinstance(metrics, dict):
            raise ValueError("evaluation episode metrics are invalid")
        metrics = metrics or {}
        terminal_progress = metrics.get("terminal_progress")
        if terminal_progress is not None:
            if type(terminal_progress) not in (int, float) or not math.isfinite(terminal_progress) or not 0 <= float(terminal_progress) <= 1:
                raise ValueError("evaluation terminal progress is invalid")
            progress_total += float(terminal_progress)
            progress_observed += 1
        recovered = metrics.get("recovered", False)
        recovery_attempted = metrics.get("recovery_attempted", recovered is True)
        if type(recovered) is not bool or type(recovery_attempted) is not bool:
            raise ValueError("evaluation recovery metric is invalid")
        recovery_attempts += int(recovery_attempted)
        successful_recoveries += int(recovered)
        episode_safety = episode.get("safety_failure", False)
        if type(episode_safety) is not bool:
            raise ValueError("evaluation safety metric is invalid")
        safety_failure = safety_failure or episode_safety
        expected_terminal = "accepted" if official_success else "rejected"
        if terminal_by_attempt[attempt_id] != expected_terminal:
            raise ValueError("evaluation ledger terminal outcome disagrees with episode evidence")
        receipt_outcome = receipt.get("outcome")
        if not isinstance(receipt_outcome, dict) or receipt_outcome.get("success") is not official_success:
            raise ValueError("evaluation worker receipt disagrees with episode evidence")
        category = str(assignment["category"])
        garment = str(assignment["garment_name"])
        category_scores[category]["episodes"] += 1
        garment_scores[garment]["episodes"] += 1
        if official_success:
            category_scores[category]["official_successes"] += 1
            garment_scores[garment]["official_successes"] += 1
        code_revisions.add(str(identity.get("code_revision")))
        asset_revisions.add(str(identity.get("asset_revision")))
        simulator_versions.add(str(identity.get("simulator_version")))
        image_identities.add(str(provenance.get("image_identity")))
        trials.append({
            "schedule_index": len(trials),
            "trial_id": assignment["trial_id"],
            "attempt_id": attempt_id,
            "category": category,
            "garment": garment,
            "seed": assignment["seed"],
            "official_success": int(official_success),
            "terminal_event": expected_terminal,
            "episode_sha256": _sha256_file(episode_path),
            "worker_receipt_sha256": _sha256_file(receipt_path),
        })
    for values in (code_revisions, asset_revisions, simulator_versions, image_identities):
        if len(values) != 1 or "None" in values:
            raise ValueError("evaluation runtime identity is inconsistent")

    successes = sum(item["official_successes"] for item in category_scores.values())
    release_stage = str(rows[0]["release_stage"])
    report: dict[str, object] = {
        "schema_version": 1,
        "kind": (
            "lehome_groot_persistent_seen80_evaluation"
            if release_stage == "seen"
            else "lehome_groot_persistent_unseen80_evaluation"
        ),
        "release_stage": release_stage,
        "candidate_key": candidate_key,
        "identity": {
            "policy_repo": policy_repo,
            "policy_revision": policy_revision,
            "policy_step": policy_step,
            "policy_artifact_sha256": policy_artifact_sha256,
            "code_revision": next(iter(code_revisions)),
            "asset_revision": next(iter(asset_revisions)),
            "simulator_version": next(iter(simulator_versions)),
            "image_identity": next(iter(image_identities)),
        },
        "matrix_sha256": matrix_sha256,
        "ledger_sha256": _sha256_file(ledger_path),
        "completed_at_ns": max(terminal_at_ns.values()),
        "episodes": len(rows),
        "official_successes": successes,
        "success_rate": successes / len(rows),
        "per_category": {
            category: {**values, "success_rate": values["official_successes"] / values["episodes"]}
            for category, values in category_scores.items()
        },
        "per_garment": {
            garment: {**values, "success_rate": values["official_successes"] / values["episodes"]}
            for garment, values in sorted(garment_scores.items())
        },
        "gates": {
            "overall_ge_70": successes / len(rows) >= 0.70,
            "each_category_ge_60": all(values["official_successes"] / values["episodes"] >= 0.60 for values in category_scores.values()),
        },
        "safety": {"evaluated": False, "physical_approval": False},
        "infrastructure_retry_count": infrastructure_retry_count,
        "gpu_seconds": gpu_seconds,
        "progress": {"observed_episodes": progress_observed, "mean_terminal_progress": progress_total / progress_observed if progress_observed else 0.0},
        "recovery": {"recovery_attempts": recovery_attempts, "successful_recoveries": successful_recoveries},
        "safety_failure": safety_failure,
        "trials": trials,
    }
    report["report_sha256"] = report_sha256(report)
    return report


def build_experiment_report(
    *,
    experiment_job: object,
    checkpoint_publication: Mapping[str, object],
    campaign_root: Path,
    matrix_path: Path,
    matrix_sha256: str,
    baseline_evidence: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the controller-facing report from terminal rollout evidence.

    This is intentionally a thin binding over the existing artifact verifier:
    it does not trust a score file from a worker, and it records the exact
    publication receipt and immutable artifact the policy server loaded.
    """
    from lehome_train.groot.experiment_publication import parse_checkpoint_publication

    publication = parse_checkpoint_publication(checkpoint_publication)
    experiment_id = getattr(experiment_job, "experiment_id", None)
    training = getattr(experiment_job, "training", None)
    evaluation = getattr(experiment_job, "evaluation", None)
    if not isinstance(experiment_id, str) or publication.experiment_id != experiment_id or publication.job_digest != experiment_id or publication.target_step != getattr(training, "target_step", None):
        raise ValueError("strict report does not bind the experiment job")
    if matrix_sha256 != getattr(evaluation, "matrix_sha256", None):
        raise ValueError("strict report does not bind the frozen evaluation matrix")
    legacy = build_report(
        campaign_root=campaign_root,
        matrix_path=matrix_path,
        matrix_sha256=matrix_sha256,
        candidate_key="new_step_2k",
        policy_repo=publication.repository,
        policy_revision=publication.immutable_revision,
        policy_step=publication.target_step,
        policy_artifact_sha256=publication.artifact_sha256,
    )
    if legacy.get("release_stage") != "public_unseen":
        raise ValueError("promotion report requires public-unseen evaluation")
    per_category = legacy["per_category"]
    if not isinstance(per_category, dict) or set(per_category) != set(_CATEGORIES):
        raise ValueError("strict report is missing category evidence")
    categories = {
        category: {
            "successes": per_category[category]["official_successes"],
            "episodes": per_category[category]["episodes"],
        }
        for category in _CATEGORIES
    }
    trainer = getattr(experiment_job, "trainer", None)
    data_sources = getattr(experiment_job, "data_sources", None)
    if not isinstance(trainer, Mapping) or not isinstance(data_sources, tuple):
        raise ValueError("strict report has no immutable trainer/data bindings")
    expected_baseline_policy = _digest(
        getattr(evaluation, "policy_digest", None),
        "pinned original baseline policy",
    )
    pairing = _paired_metrics(
        legacy,
        baseline_evidence,
        expected_policy_digest=expected_baseline_policy,
    )
    report = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "checkpoint_receipt_sha256": publication.receipt_sha256,
        "matrix_sha256": matrix_sha256,
        "policy_digest": publication.artifact_sha256,
        "categories": categories,
        "episode_artifacts": legacy["trials"],
        "promotion_metrics": {
            "overall_successes": legacy["official_successes"],
            "overall_episodes": legacy["episodes"],
            "overall_success_rate": legacy["success_rate"],
            "safety_failure": legacy["safety_failure"],
            # The controller may rank only a report with ``available`` pairing
            # evidence.  A missing baseline is intentionally typed rather than
            # silently treated as a neutral paired score.
            "paired_improvement": pairing.get("paired_improvement", 0.0),
            "gpu_seconds": legacy["gpu_seconds"],
            "infrastructure_retry_count": legacy["infrastructure_retry_count"],
            "progress": legacy["progress"],
            "recovery": legacy["recovery"],
            "pairing": pairing,
        },
        "provenance": {
            "trainer": dict(trainer),
            "runtime": {
                "code_revision": legacy["identity"]["code_revision"],
                "asset_revision": legacy["identity"]["asset_revision"],
                "simulator_version": legacy["identity"]["simulator_version"],
                "image_identity": legacy["identity"]["image_identity"],
            },
            "data_sources": [
                {"kind": source.kind, "repository": source.repository, "revision": source.revision, "prefix": source.prefix, "manifest_sha256": source.manifest_sha256, "tree_sha256": source.tree_sha256}
                for source in data_sources
            ],
        },
        "strict_seal": False,
        "evidence_report_sha256": legacy["report_sha256"],
    }
    report["report_sha256"] = report_sha256(report)
    return report


def _terminal_sync_artifacts(*, campaign_root: Path, legacy: Mapping[str, object]) -> list[dict[str, object]]:
    """Bind every final episode to its own Hub read-back receipt.

    The normal training/promotion report deliberately keeps a local evidence
    shape.  Final unseen-80 selection is stricter: every terminal episode must
    have a separate immutable Hub receipt.  This helper refuses to upgrade
    local-only outcomes into a final candidate receipt.
    """
    trials = legacy.get("trials")
    if not isinstance(trials, list):
        raise ValueError("final evaluation trial evidence is invalid")
    receipts_root = campaign_root / "hf-sync-receipts"
    if receipts_root.is_symlink() or not receipts_root.is_dir():
        raise ValueError("final evaluation requires terminal Hub sync receipts")
    final: list[dict[str, object]] = []
    for trial in trials:
        if not isinstance(trial, Mapping):
            raise ValueError("final evaluation trial evidence is invalid")
        attempt = trial.get("attempt_id")
        trial_id = trial.get("trial_id")
        category = trial.get("category")
        success = trial.get("official_success")
        if type(attempt) is not str or _SHA256.fullmatch(attempt) is None or type(trial_id) is not str or not trial_id or category not in _CATEGORIES or type(success) is not int or success not in (0, 1):
            raise ValueError("final evaluation trial identity is invalid")
        sync_path = receipts_root / f"{attempt}.sync.json"
        receipt = _json_object(sync_path, "final evaluation Hub receipt")
        required = {
            "schema_version", "attempt_id", "repository", "round_id", "remote_prefix",
            "publication_ref", "immutable_revision", "entry_count", "episode_sha256",
            "readback_verified",
        }
        if set(receipt) != required or receipt.get("schema_version") != 1 or receipt.get("attempt_id") != attempt or receipt.get("readback_verified") is not True:
            raise ValueError("final evaluation Hub receipt is not read-back verified")
        if type(receipt.get("immutable_revision")) is not str or _COMMIT.fullmatch(receipt["immutable_revision"]) is None or type(receipt.get("entry_count")) is not int or receipt["entry_count"] <= 0:
            raise ValueError("final evaluation Hub receipt identity is invalid")
        final.append({
            "trial_id": trial_id,
            "category": category,
            "official_success": success,
            "artifact_sha256": _digest(receipt.get("episode_sha256"), "final evaluation episode"),
            "receipt_sha256": _sha256_file(sync_path),
            "readback_verified": True,
            "sealed": True,
        })
    return final


def _seen_regression_flag(value: Mapping[str, object]) -> bool:
    """Require an explicit sealed/read-back seen-regression decision."""
    document = dict(value)
    expected = {
        "schema_version", "kind", "candidate_checkpoint_receipt_sha256",
        "major_seen_regression", "readback_verified", "sealed", "report_sha256",
    }
    if set(document) != expected or document.get("schema_version") != 1 or document.get("kind") != "lehome_experiment_seen_regression_evidence" or document.get("readback_verified") is not True or document.get("sealed") is not True or type(document.get("major_seen_regression")) is not bool:
        raise ValueError("final evaluation seen-regression evidence is invalid")
    _digest(document.get("candidate_checkpoint_receipt_sha256"), "seen-regression checkpoint receipt")
    digest = _digest(document.get("report_sha256"), "seen-regression report")
    if digest != _canonical_sha256(document):
        raise ValueError("final evaluation seen-regression evidence digest mismatch")
    return bool(document["major_seen_regression"])


def build_final_unseen80_report(
    *,
    experiment_job: object,
    checkpoint_publication: Mapping[str, object],
    campaign_root: Path,
    matrix_path: Path,
    matrix_sha256: str,
    candidate_id: str,
    seen_regression_evidence: Mapping[str, object],
) -> dict[str, object]:
    """Emit the sealed final-evaluation receipt consumed by winner selection.

    This is deliberately distinct from :func:`build_experiment_report`: the
    latter is an unseen-20 promotion report, while this path permits only a
    frozen 80-row, 20-per-category public-unseen matrix with terminal Hub
    receipts for every episode.
    """
    from lehome_train.groot.experiment_publication import parse_checkpoint_publication
    from lehome_train.groot.experiment_winner import seal_final_unseen80_report

    publication = parse_checkpoint_publication(checkpoint_publication)
    experiment_id = getattr(experiment_job, "experiment_id", None)
    training = getattr(experiment_job, "training", None)
    if type(experiment_id) is not str or publication.canonical.get("schema_version") != 2 or publication.experiment_id != experiment_id or publication.job_digest != experiment_id or publication.target_step != getattr(training, "target_step", None):
        raise ValueError("final evaluation does not bind a publication-v2 experiment job")
    if type(candidate_id) is not str or not candidate_id:
        raise ValueError("final evaluation candidate ID is invalid")
    rows = _load_matrix(Path(matrix_path), matrix_sha256)
    if (
        len(rows) != 80
        or any(item.get("release_stage") != "public_unseen" for item in rows)
        or any(sum(item.get("category") == category for item in rows) != 20 for category in _CATEGORIES)
    ):
        raise ValueError("final evaluation requires the exact frozen unseen-80 matrix")
    legacy = build_report(
        campaign_root=campaign_root,
        matrix_path=matrix_path,
        matrix_sha256=matrix_sha256,
        candidate_key="new_step_2k",
        policy_repo=publication.repository,
        policy_revision=publication.immutable_revision,
        policy_step=publication.target_step,
        policy_artifact_sha256=publication.artifact_sha256,
    )
    artifacts = _terminal_sync_artifacts(campaign_root=Path(campaign_root), legacy=legacy)
    categories = {
        category: {
            "successes": int(legacy["per_category"][category]["official_successes"]),
            "episodes": int(legacy["per_category"][category]["episodes"]),
        }
        for category in _CATEGORIES
    }
    seen = _seen_regression_flag(seen_regression_evidence)
    if seen_regression_evidence.get("candidate_checkpoint_receipt_sha256") != publication.receipt_sha256:
        raise ValueError("final evaluation seen-regression evidence does not bind checkpoint")
    report = seal_final_unseen80_report({
        "schema_version": 2,
        "kind": "lehome_experiment_final_unseen80",
        "candidate_id": candidate_id,
        "experiment_id": experiment_id,
        "checkpoint_receipt_sha256": publication.receipt_sha256,
        "checkpoint_publication": dict(publication.canonical),
        "matrix_sha256": matrix_sha256,
        "policy_digest": publication.artifact_sha256,
        "categories": categories,
        "overall_successes": int(legacy["official_successes"]),
        "episode_artifacts": artifacts,
        "safety_failure": bool(legacy["safety_failure"]),
        "major_seen_regression": seen,
    })
    return report


def write_experiment_report(path: Path, report: Mapping[str, object]) -> Path:
    """Atomically publish a controller report and immutable digest sidecar."""
    output = Path(path)
    write_report(output, report)
    sidecar = output.with_suffix(output.suffix + ".sha256")
    descriptor = os.open(sidecar, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(_sha256_file(output) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        sidecar.unlink(missing_ok=True)
        raise
    directory = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return output


class HuggingFaceFinalReportTransport:
    """Small explicit Hugging Face adapter used only by the final publisher."""

    def __init__(self, token_file: Path) -> None:
        token = Path(token_file)
        if token.is_symlink() or not token.is_file() or (token.stat().st_mode & 0o777) != 0o600:
            raise ValueError("final Hugging Face token file is unsafe")
        self.token = token.read_text(encoding="utf-8").strip()
        if not self.token:
            raise ValueError("final Hugging Face token is empty")

    def _api(self):
        try:
            from huggingface_hub import HfApi
        except ImportError as error:
            raise RuntimeError("huggingface_hub is required for final report publication") from error
        return HfApi(token=self.token)

    def upload_bytes(self, repository: str, path: str, payload: bytes) -> None:
        self._api().upload_file(
            path_or_fileobj=BytesIO(payload),
            path_in_repo=path,
            repo_id=repository,
            repo_type="dataset",
            commit_message="publish immutable LeHome final unseen-80 receipt",
        )

    def read_bytes(self, repository: str, path: str) -> bytes:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise RuntimeError("huggingface_hub is required for final report readback") from error
        downloaded = hf_hub_download(repo_id=repository, repo_type="dataset", filename=path, force_download=True, token=self.token)
        return Path(downloaded).read_bytes()


def write_final_unseen80_report(
    path: Path,
    report: Mapping[str, object],
    *,
    transport: object,
    repository: str,
    remote_path: str,
) -> Path:
    """Publish/read back the final receipt before persisting a local copy."""
    from lehome_train.groot.experiment_winner import publish_final_unseen80_report

    published = publish_final_unseen80_report(
        report,
        transport=transport,
        repository=repository,
        path=remote_path,
    )
    return write_experiment_report(path, published)


def write_report(path: Path, report: Mapping[str, object]) -> None:
    output = Path(path)
    if output.exists() or output.is_symlink():
        raise ValueError("evaluation report output must not already exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(report, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        directory = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        output.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--matrix-sha256", required=True)
    parser.add_argument("--candidate-key", choices=_CANDIDATES, required=True)
    parser.add_argument("--policy-repo", required=True)
    parser.add_argument("--policy-revision", required=True)
    parser.add_argument("--policy-step", type=int, required=True)
    parser.add_argument("--policy-artifact-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(
        campaign_root=args.campaign_root,
        matrix_path=args.matrix,
        matrix_sha256=args.matrix_sha256,
        candidate_key=args.candidate_key,
        policy_repo=args.policy_repo,
        policy_revision=args.policy_revision,
        policy_step=args.policy_step,
        policy_artifact_sha256=args.policy_artifact_sha256,
    )
    write_report(args.output, report)
    print(report["report_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
