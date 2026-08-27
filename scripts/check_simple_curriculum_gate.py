#!/usr/bin/env python3
"""Authenticate and gate the first 100 simple-curriculum outcomes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Mapping

from lehome.flywheel.simple_curriculum import build_calibration_rows


_DECISIONS = frozenset({"continue", "infrastructure_stop", "fidelity_stop", "insufficient_source_stop"})
_REASONS = frozenset({"passed", "valid_outcome_count", "episode_fidelity", "invalid_ratio", "official_success_floor", "mixed_runtime_identity"})
_ALLOWED_PAIRS = frozenset({
    ("continue", "passed"),
    ("infrastructure_stop", "valid_outcome_count"),
    ("infrastructure_stop", "invalid_ratio"),
    ("fidelity_stop", "episode_fidelity"),
    ("fidelity_stop", "mixed_runtime_identity"),
    ("insufficient_source_stop", "official_success_floor"),
})
_FIDELITY_FIELDS = ("missing_cloth", "cloth_flight", "nonfinite_cloth_state", "safety_failure")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class GateDecision:
    decision: str
    reason: str

    def __post_init__(self) -> None:
        if self.decision not in _DECISIONS or self.reason not in _REASONS or (self.decision, self.reason) not in _ALLOWED_PAIRS:
            raise ValueError("gate decision is invalid")

    def as_dict(self) -> dict[str, str]:
        return {"decision": self.decision, "reason": self.reason}


def _count(report: Mapping[str, object], key: str) -> int:
    value = report.get(key)
    if type(value) is not int or value < 0:
        raise ValueError(f"{key} is invalid")
    return value


def evaluate_gate(report: Mapping[str, object]) -> GateDecision:
    """Return the fixed first-100 circuit-breaker decision without coercion."""

    if not isinstance(report, Mapping):
        raise ValueError("gate report is invalid")
    valid_outcomes = _count(report, "valid_outcomes")
    invalid = _count(report, "infrastructure_invalid_executions")
    executions = _count(report, "execution_count")
    if executions != valid_outcomes + invalid:
        raise ValueError("execution_count is invalid")
    if valid_outcomes != 100:
        return GateDecision("infrastructure_stop", "valid_outcome_count")
    trials = report.get("gate_trials")
    if not isinstance(trials, list):
        raise ValueError("gate_trials are invalid")
    aggregate_safety = report.get("safety_failure")
    if type(aggregate_safety) is not bool:
        raise ValueError("safety_failure is invalid")
    if aggregate_safety:
        return GateDecision("fidelity_stop", "episode_fidelity")
    for trial in trials:
        if not isinstance(trial, Mapping):
            raise ValueError("trial is invalid")
        fidelity = trial.get("fidelity")
        if not isinstance(fidelity, Mapping) or any(type(fidelity.get(field)) is not bool for field in _FIDELITY_FIELDS):
            raise ValueError("trial fidelity is invalid")
        if any(fidelity[field] for field in _FIDELITY_FIELDS):
            return GateDecision("fidelity_stop", "episode_fidelity")
    if executions == 0:
        raise ValueError("execution_count is invalid")
    if invalid / executions > 0.02:
        return GateDecision("infrastructure_stop", "invalid_ratio")
    if _count(report, "official_successes") < 5:
        return GateDecision("insufficient_source_stop", "official_success_floor")
    runtime_identities = report.get("runtime_identities")
    if (not isinstance(runtime_identities, list) or len(runtime_identities) != 1
            or not isinstance(runtime_identities[0], str)):
        return GateDecision("fidelity_stop", "mixed_runtime_identity")
    return GateDecision("continue", "passed")


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _canonical_digest(value: Mapping[str, object]) -> str:
    body = dict(value)
    body.pop("report_sha256", None)
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_nonfinite(value: object) -> None:
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_nonfinite(item)
    elif isinstance(value, list):
        for item in value:
            _reject_nonfinite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON number")


def _has_symlink_ancestor(path: Path) -> bool:
    current = path
    while True:
        try:
            if stat.S_ISLNK(current.lstat().st_mode):
                return True
        except OSError:
            return True
        if current.parent == current:
            return False
        current = current.parent


def _strict_json(path: Path, *, label: str) -> tuple[object, bytes]:
    path = Path(path)
    if not path.is_absolute() or _has_symlink_ancestor(path) or path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    try:
        payload = path.read_bytes()
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_reject_duplicates,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("non-finite JSON number")),
        )
        _reject_nonfinite(value)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is malformed") from error
    return value, payload


def _safe_absent(path: Path) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink() or _has_symlink_ancestor(path.parent):
        raise FileExistsError("gate receipt output must be an absent safe absolute path")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink() or not stat.S_ISDIR(parent.stat().st_mode):
        raise ValueError("gate receipt parent is unsafe")


def _atomic_write(path: Path, payload: bytes, *, before_publish: object = None) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
        if before_publish is not None:
            if not callable(before_publish):
                raise ValueError("publication hook is invalid")
            before_publish()
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _policy_identity(value: object) -> dict[str, object]:
    required = {"policy_repo", "policy_revision", "policy_step", "policy_artifact_sha256"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("policy identity is invalid")
    if (not isinstance(value["policy_repo"], str) or not value["policy_repo"]
            or not isinstance(value["policy_revision"], str) or len(value["policy_revision"]) != 40
            or type(value["policy_step"]) is not int or value["policy_step"] != 12000
            or not isinstance(value["policy_artifact_sha256"], str) or _SHA256.fullmatch(value["policy_artifact_sha256"]) is None):
        raise ValueError("policy identity is invalid")
    return dict(value)


def _runtime_identity(identity: Mapping[str, object], provenance: Mapping[str, object]) -> str:
    fields = {
        "policy_repo": provenance.get("policy_repo"), "policy_revision": provenance.get("policy_revision"),
        "policy_step": provenance.get("policy_step"), "policy_artifact_sha256": provenance.get("policy_artifact_sha256"),
        "code_revision": identity.get("code_revision"), "asset_revision": identity.get("asset_revision"),
        "simulator_version": identity.get("simulator_version"), "image_identity": provenance.get("image_identity"),
        "simulator_device": provenance.get("simulator_device"), "cloth_device": provenance.get("cloth_device"),
        "renderer_device": provenance.get("renderer_device"), "camera_device": provenance.get("camera_device"),
        "policy_device": provenance.get("policy_device"),
    }
    if any(not isinstance(value, (str, int)) for value in fields.values()):
        raise ValueError("runtime identity is incomplete")
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _validate_matrix(rows: object, catalog: object) -> list[dict[str, object]]:
    if not isinstance(rows, list) or len(rows) != 100 or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("calibration-head matrix is invalid")
    try:
        seed_base = rows[0].get("seed")
        canonical = build_calibration_rows(catalog, seed_base=seed_base)[:100]  # type: ignore[arg-type]
    except (AttributeError, ValueError) as error:
        raise ValueError("calibration-head matrix is invalid") from error
    if rows != canonical:
        raise ValueError("calibration-head matrix is not canonical")
    return [dict(row) for row in canonical]


def _authenticate_report(report: object, *, report_bytes: bytes, matrix_rows: list[dict[str, object]], matrix_bytes: bytes, policy: dict[str, object]) -> dict[str, object]:
    if not isinstance(report, Mapping):
        raise ValueError("report is invalid")
    data = dict(report)
    if (data.get("schema_version") != 1 or data.get("kind") != "lehome_simple_curriculum_first100_report_v1"
            or data.get("campaign_kind") != "simple_curriculum_source_v1" or data.get("logical_stage") != "calibration_head"):
        raise ValueError("report campaign identity is invalid")
    if data.get("report_sha256") != _canonical_digest(data):
        raise ValueError("report SHA-256 mismatch")
    matrix_sha = hashlib.sha256(matrix_bytes).hexdigest()
    if data.get("matrix_sha256") != matrix_sha:
        raise ValueError("matrix SHA-256 mismatch")
    if _policy_identity(data.get("identity")) != policy:
        raise ValueError("report policy identity mismatch")
    expected_ids = {str(row["attempt_id"]) for row in matrix_rows}
    fresh = data.get("fresh_assignment_ids")
    if (not isinstance(fresh, list) or any(not isinstance(item, str) for item in fresh)
            or fresh != sorted(fresh) or len(set(fresh)) != len(fresh) or not set(fresh).issubset(expected_ids)):
        raise ValueError("report fresh assignment identities do not match matrix")
    trials = data.get("gate_trials")
    if not isinstance(trials, list) or len(trials) != len(fresh):
        raise ValueError("report gate_trials are invalid")
    by_id: dict[str, Mapping[str, object]] = {}
    runtime: set[str] = set()
    successes = 0
    rows_by_id = {str(row["attempt_id"]): row for row in matrix_rows}
    for assignment_id in fresh:
        row = rows_by_id[assignment_id]
        matches = [trial for trial in trials if isinstance(trial, Mapping) and trial.get("assignment_id") == assignment_id]
        if len(matches) != 1:
            raise ValueError("report trial assignment identities do not match matrix")
        trial = matches[0]
        if trial.get("trial_id") != row.get("trial_id") or trial.get("terminal_event") not in {"accepted", "rejected"}:
            raise ValueError("report trial identity is invalid")
        official_success = trial.get("official_success")
        if type(official_success) is not bool or (official_success != (trial["terminal_event"] == "accepted")):
            raise ValueError("report terminal outcome is invalid")
        identity, provenance, fidelity = trial.get("identity"), trial.get("provenance"), trial.get("fidelity")
        if not isinstance(identity, Mapping) or not isinstance(provenance, Mapping) or not isinstance(fidelity, Mapping):
            raise ValueError("report trial provenance is invalid")
        if _policy_identity({key: identity.get(key) for key in policy}) != policy or _policy_identity({key: provenance.get(key) for key in policy}) != policy:
            raise ValueError("report trial policy identity mismatch")
        renderer = provenance.get("renderer_device")
        camera = provenance.get("camera_device")
        policy_device = provenance.get("policy_device")
        if (provenance.get("simulator_device") != "cpu" or provenance.get("cloth_device") != "cpu"
                or not all(isinstance(device, str) and re.fullmatch(r"cuda:[0-9]+", device) for device in (renderer, camera, policy_device))
                or len({renderer, camera, policy_device}) != 1):
            raise ValueError("report device provenance is invalid")
        if any(type(fidelity.get(key)) is not bool for key in _FIDELITY_FIELDS):
            raise ValueError("report fidelity evidence is invalid")
        by_id[assignment_id] = trial
        runtime.add(_runtime_identity(identity, provenance))
        successes += int(official_success)
    if data.get("runtime_identities") != sorted(runtime):
        raise ValueError("report runtime identity digest mismatch")
    if _count(data, "valid_outcomes") != len(fresh) or _count(data, "official_successes") != successes:
        raise ValueError("report metrics do not match terminal evidence")
    invalid = _count(data, "infrastructure_invalid_executions")
    if _count(data, "execution_count") != len(fresh) + invalid:
        raise ValueError("report execution count is invalid")
    aggregate_safety = data.get("safety_failure")
    if type(aggregate_safety) is not bool or aggregate_safety != any(
        bool(trial["fidelity"]["safety_failure"]) for trial in by_id.values()
    ):
        raise ValueError("report safety evidence is invalid")
    return data


def build_gate_receipt(report: object, *, report_bytes: bytes, matrix: object, matrix_bytes: bytes, trusted_policy: object, policy_bytes: bytes, catalog: object, catalog_bytes: bytes) -> dict[str, object]:
    policy = _policy_identity(trusted_policy)
    rows = _validate_matrix(matrix, catalog)
    authenticated = _authenticate_report(report, report_bytes=report_bytes, matrix_rows=rows, matrix_bytes=matrix_bytes, policy=policy)
    decision = evaluate_gate(authenticated)
    return {
        "schema_version": 1,
        "kind": "lehome_simple_curriculum_first100_gate_receipt_v1",
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "matrix_sha256": hashlib.sha256(matrix_bytes).hexdigest(),
        "trusted_policy_identity_sha256": hashlib.sha256(policy_bytes).hexdigest(),
        "approved_garment_catalog_sha256": hashlib.sha256(catalog_bytes).hexdigest(),
        "trusted_policy_identity": policy,
        "metrics": {key: authenticated[key] for key in ("valid_outcomes", "infrastructure_invalid_executions", "execution_count", "official_successes", "runtime_identities", "fresh_assignment_ids")},
        **decision.as_dict(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--trusted-policy-identity", type=Path, required=True)
    parser.add_argument("--approved-garment-catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        _safe_absent(args.output)
        report, report_bytes = _strict_json(args.report, label="report")
        matrix, matrix_bytes = _strict_json(args.matrix, label="matrix")
        policy, policy_bytes = _strict_json(args.trusted_policy_identity, label="trusted policy identity")
        catalog, catalog_bytes = _strict_json(args.approved_garment_catalog, label="approved garment catalog")
        receipt = build_gate_receipt(report, report_bytes=report_bytes, matrix=matrix, matrix_bytes=matrix_bytes, trusted_policy=policy, policy_bytes=policy_bytes, catalog=catalog, catalog_bytes=catalog_bytes)
        _atomic_write(args.output, _canonical_bytes(receipt))
    except (OSError, ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(receipt["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
