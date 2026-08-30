"""Fail-closed interpretation of official BEHAVIOR rollout terminal records."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
import os
import stat
from pathlib import Path, PurePosixPath

from b1k_rollout.contracts import RolloutContract
from b1k_rollout.identity import canonical_json_sha256, reject_credential_material, require_sha256
from b1k_rollout.task_manifest import canonical_manifest_sha256, validate_task_manifest
from b1k_rollout.provenance import ProvenanceAuthenticator, canonical_attestation_payload


class Outcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    QUARANTINE = "quarantine"


@dataclass(frozen=True, slots=True)
class ClassifiedOutcome:
    """A classification that retains the complete raw evaluator evidence."""

    outcome: Outcome
    reason: str
    raw_evidence: object
    raw_evidence_sha256: str
    episode_id: str | None
    rollout_id: int | None
    final_q_scores: object | None
    evaluator_metrics: Mapping[str, object] | None
    evaluator_identity: Mapping[str, object] | None
    provenance: Mapping[str, object]
    provenance_attestation: Mapping[str, str] | None


class OutcomeEvidenceError(ValueError):
    """The evaluator record is not sufficient terminal evidence."""


_SCHEMA_VERSION = 1
_EVIDENCE_FIELDS = frozenset(
    (
        "schema_version",
        "episode_id",
        "rollout_id",
        "task",
        "steps",
        "success",
        "q_score",
        "time",
        "agent_distance",
        "normalized_agent_distance",
        "completed",
        "mode",
        "instance_id",
        "instance_index",
        "contract",
        "artifact_hashes",
    )
)
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ARTIFACT_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def classify_outcome(
    evidence: Mapping[str, object] | str | bytes | bytearray,
    *,
    task_manifest: Mapping[str, object],
) -> ClassifiedOutcome:
    """Classify only a complete official terminal record.

    A Q-score is evidence retained for diagnosis, never a success threshold.  Every
    malformed or incomplete record is returned as quarantine so callers can publish
    its raw evidence without accidentally treating it as a completed rollout.
    """

    raw_evidence = _preserve_raw_evidence(evidence)
    try:
        noncanonical_mapping = _mapping_has_nonfinite_values(evidence)
        record, raw_hash = _decode_evidence(evidence)
        _reject_process_failure(record)
        parsed = _validate_evidence(record, task_manifest=task_manifest)
        if noncanonical_mapping:
            raise OutcomeEvidenceError(
                "mapping evidence with non-finite values must be retained as original bytes"
            )
    except (OutcomeEvidenceError, TypeError, ValueError, UnicodeDecodeError) as error:
        return ClassifiedOutcome(
            outcome=Outcome.QUARANTINE,
            reason=str(error),
            raw_evidence=raw_evidence,
            raw_evidence_sha256=_raw_hash(raw_evidence),
            episode_id=_raw_episode_id(raw_evidence),
            rollout_id=_raw_rollout_id(raw_evidence),
            final_q_scores=_raw_final_q_scores(raw_evidence),
            evaluator_metrics=_raw_metrics(raw_evidence),
            evaluator_identity=None,
            provenance=_provenance("mapping", "mapping", "invalid", raw_evidence),
            provenance_attestation=None,
        )

    outcome = Outcome.SUCCESS if parsed.success is True else Outcome.FAILURE
    return ClassifiedOutcome(
        outcome=outcome,
        reason="",
        raw_evidence=raw_evidence,
        raw_evidence_sha256=raw_hash,
        episode_id=parsed.episode_id,
        rollout_id=parsed.rollout_id,
        final_q_scores=parsed.final_q_scores,
        evaluator_metrics=parsed.evaluator_metrics,
        evaluator_identity=parsed.evaluator_identity,
        provenance=_provenance("mapping", "mapping", "terminal", raw_evidence),
        provenance_attestation=None,
    )


def classify_outcome_file(
    path: Path, *, task_manifest: Mapping[str, object], episode_key: str, contract: RolloutContract, authenticator: ProvenanceAuthenticator
) -> ClassifiedOutcome:
    """Classify one evidence file, treating a staged file as non-terminal evidence."""

    disposition = "regular"
    try:
        raw, filename = _read_evidence_file(Path(path))
    except OutcomeEvidenceError as error:
        disposition = "symlink" if Path(path).is_symlink() else "unreadable"
        return _attest_file(_quarantine_file_evidence(str(error), filename=Path(path).name, disposition=disposition), episode_key, contract, authenticator)
    if filename.endswith(".incomplete"):
        return _attest_file(ClassifiedOutcome(
            outcome=Outcome.QUARANTINE,
            reason="official evaluator evidence is still incomplete",
            raw_evidence=raw,
            raw_evidence_sha256=hashlib.sha256(raw).hexdigest(),
            episode_id=None,
            rollout_id=None,
            final_q_scores=None,
            evaluator_metrics=None,
            evaluator_identity=None,
            provenance=_provenance("file", "incomplete", "incomplete", raw, basename=filename),
            provenance_attestation=None,
        ), episode_key, contract, authenticator)
    classified = classify_outcome(raw, task_manifest=task_manifest)
    return _attest_file(replace(
        classified,
        provenance=_provenance("file", "regular", "terminal" if classified.outcome is not Outcome.QUARANTINE else "invalid", raw, basename=filename),
    ), episode_key, contract, authenticator)


@dataclass(frozen=True, slots=True)
class _ValidatedEvidence:
    episode_id: str
    rollout_id: int
    success: bool
    final_q_scores: object
    evaluator_metrics: Mapping[str, object]
    evaluator_identity: Mapping[str, object]


def _decode_evidence(
    evidence: Mapping[str, object] | str | bytes | bytearray,
) -> tuple[Mapping[str, object], str]:
    raw_hash = raw_evidence_sha256(evidence)
    if isinstance(evidence, Mapping):
        record = evidence
    elif isinstance(evidence, (str, bytes, bytearray)):
        try:
            record = json.loads(evidence)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise OutcomeEvidenceError("official evaluator evidence is malformed JSON") from error
    else:
        raise OutcomeEvidenceError("official evaluator evidence must be a JSON object")
    if not isinstance(record, Mapping):
        raise OutcomeEvidenceError("official evaluator evidence must be a JSON object")
    try:
        return record, raw_hash
    except (TypeError, ValueError) as error:
        raise OutcomeEvidenceError("official evaluator evidence is not canonical JSON") from error


def _validate_evidence(
    record: Mapping[str, object], *, task_manifest: Mapping[str, object]
) -> _ValidatedEvidence:
    reject_credential_material(record)
    _reject_lehome_material(record)
    if record.get("schema_version") != _SCHEMA_VERSION:
        raise OutcomeEvidenceError("official evaluator evidence schema version is invalid")

    episode_id = _require_identifier(record.get("episode_id"), label="episode id")
    rollout_id = _require_rollout_id(record.get("rollout_id"))
    task = _require_task(record.get("task"))
    steps = record.get("steps")
    if type(steps) is not int or steps <= 0:
        raise OutcomeEvidenceError("episode steps must be a positive integer")

    contract = _contract(record.get("contract"))
    _validate_manifest(task_manifest, contract)
    if record.get("completed") is not True:
        raise OutcomeEvidenceError("official evaluator completion marker is missing or false")
    success = record.get("success")
    if type(success) is not bool:
        raise OutcomeEvidenceError("official evaluator success must be exactly bool true or false")
    evaluator_metrics = _validate_metrics(record)
    final_q_scores = evaluator_metrics["q_score"]
    if set(record) != _EVIDENCE_FIELDS:
        raise OutcomeEvidenceError("official evaluator evidence fields are invalid")
    mode, index, instance_id = _validate_instance(
        task_manifest, task=task, contract=contract, record=record
    )
    _validate_artifact_hashes(record.get("artifact_hashes"))
    return _ValidatedEvidence(
        episode_id=episode_id,
        rollout_id=rollout_id,
        success=success,
        final_q_scores=copy.deepcopy(final_q_scores),
        evaluator_metrics=copy.deepcopy(evaluator_metrics),
        evaluator_identity={
            "campaign_id": contract.campaign_id,
            "contract_identity": contract.identity,
            "instance_id": instance_id,
            "instance_index": index,
            "mode": mode,
            "model_commit": contract.model_commit,
            "rollout_id": rollout_id,
            "task": task,
        },
    )


def _reject_process_failure(record: Mapping[str, object]) -> None:
    """Reject known controller failure markers before generic schema validation."""

    for key in ("policy_server_crash", "simulator_crash"):
        if record.get(key) is True:
            raise OutcomeEvidenceError(f"{key.replace('_', ' ')} prevents terminal classification")
    if record.get("timeout") is True:
        if record.get("completed") is not True:
            raise OutcomeEvidenceError("timeout without official evaluator closure")
    status = record.get("status")
    if status in ("crashed", "policy_server_crash", "simulator_crash"):
        raise OutcomeEvidenceError("crash status prevents terminal classification")


def _validate_manifest(manifest: Mapping[str, object], contract: RolloutContract) -> None:
    try:
        validate_task_manifest(manifest)
    except (TypeError, ValueError) as error:
        raise OutcomeEvidenceError("canonical loaded task manifest is invalid") from error
    if canonical_manifest_sha256(manifest) != contract.task_manifest_sha256:
        raise OutcomeEvidenceError("contract task manifest identity does not bind the loaded manifest")


def _validate_instance(
    manifest: Mapping[str, object],
    *,
    task: str,
    contract: RolloutContract,
    record: Mapping[str, object],
) -> tuple[str, int, int]:
    mode = record.get("mode")
    if mode != contract.evaluator_mode:
        raise OutcomeEvidenceError("evaluator mode does not match the rollout contract")
    index = record.get("instance_index")
    if type(index) is not int or index < 0:
        raise OutcomeEvidenceError("resolved evaluator instance index is invalid")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, Sequence):
        raise OutcomeEvidenceError("canonical loaded task manifest has no tasks")
    for item in tasks:
        if not isinstance(item, Mapping) or item.get("task_name") != task:
            continue
        requests = item.get("requested_instances")
        if isinstance(requests, Sequence) and any(
            isinstance(request, Mapping) and request.get("mode") == mode and request.get("index") == index
            for request in requests
        ):
            instance_id = record.get("instance_id")
            if type(instance_id) is not int or instance_id < 0:
                raise OutcomeEvidenceError("resolved evaluator instance identity is invalid")
            if mode == "public_test" and instance_id != 301 + index:
                raise OutcomeEvidenceError(
                    "public_test resolved evaluator instance must equal 301 plus requested index"
                )
            return mode, index, instance_id
        break
    raise OutcomeEvidenceError("task or resolved evaluator instance is not bound to the canonical manifest")


def _validate_artifact_hashes(value: object) -> None:
    hashes = _mapping(value, label="artifact hashes")
    if not hashes:
        raise OutcomeEvidenceError("at least one artifact content hash is required")
    for path, digest in hashes.items():
        if not isinstance(path, str) or not _safe_artifact_path(path):
            raise OutcomeEvidenceError("artifact path is unsafe")
        try:
            require_sha256(digest, label="artifact content hash")
        except ValueError as error:
            raise OutcomeEvidenceError(str(error)) from error


def _contract(value: object) -> RolloutContract:
    if not isinstance(value, Mapping):
        raise OutcomeEvidenceError("rollout contract is missing or invalid")
    try:
        return RolloutContract.from_mapping(value)
    except (TypeError, ValueError) as error:
        raise OutcomeEvidenceError("rollout contract is invalid") from error


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise OutcomeEvidenceError(f"{label} must be an object")
    return value


def _require_identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise OutcomeEvidenceError(f"{label} is invalid")
    if "lehome" in value.casefold():
        raise OutcomeEvidenceError(f"{label} must not reference LeHome")
    return value


def _require_rollout_id(value: object) -> int:
    if type(value) is int and value >= 0:
        return value
    raise OutcomeEvidenceError("rollout id is invalid")


def _validate_metrics(record: Mapping[str, object]) -> dict[str, object]:
    """Validate exact upstream TaskMetric and AgentMetric payloads.

    Upstream may serialize positive infinity for normalized metrics when a
    denominator is zero.  Byte-backed evidence retains those original bytes; a
    mapping containing a non-finite value is quarantined because canonical JSON
    hashing cannot represent it without changing the evaluator data.
    """

    q_score = _mapping(record.get("q_score"), label="Q-score")
    if set(q_score) != {"final"}:
        raise OutcomeEvidenceError("official evaluator Q-score payload is invalid")
    q_final = _finite_number(q_score.get("final"), label="q_score.final")
    if not 0 <= q_final <= 1:
        raise OutcomeEvidenceError("official evaluator q_score.final must be in [0, 1]")
    if record.get("success") is True and q_final != 1.0:
        raise OutcomeEvidenceError("official evaluator success requires q_score.final exactly 1.0")

    time = _mapping(record.get("time"), label="time")
    if set(time) != {"simulator_steps", "simulator_time", "normalized_time"}:
        raise OutcomeEvidenceError("official evaluator time payload is invalid")
    if (
        type(time.get("simulator_steps")) is not int
        or time["simulator_steps"] <= 0
        or time["simulator_steps"] != record.get("steps")
    ):
        raise OutcomeEvidenceError("official evaluator simulator steps must exactly equal episode steps")
    _nonnegative_finite_number(time.get("simulator_time"), label="time.simulator_time")
    _normalized_number(time.get("normalized_time"), label="time.normalized_time")

    metrics: dict[str, object] = {"q_score": dict(q_score), "time": dict(time)}
    for field, normalized in (
        ("agent_distance", False),
        ("normalized_agent_distance", True),
    ):
        distances = _mapping(record.get(field), label=field)
        if set(distances) != {"base", "left", "right"}:
            raise OutcomeEvidenceError(f"official evaluator {field} payload is invalid")
        for arm, value in distances.items():
            if normalized:
                _normalized_number(value, label=f"{field}.{arm}")
            else:
                _nonnegative_finite_number(value, label=f"{field}.{arm}")
        metrics[field] = dict(distances)
    return metrics


def _finite_number(value: object, *, label: str) -> float | int:
    if type(value) is int:
        return value
    if type(value) is float:
        try:
            if math.isfinite(value):
                return value
        except OverflowError:
            pass
    raise OutcomeEvidenceError(f"official evaluator {label} must be a finite number")


def _nonnegative_finite_number(value: object, *, label: str) -> float | int:
    value = _finite_number(value, label=label)
    if value < 0:
        raise OutcomeEvidenceError(f"official evaluator {label} must be nonnegative")
    return value


def _normalized_number(value: object, *, label: str) -> float | int:
    if type(value) is int:
        if value >= 0:
            return value
        raise OutcomeEvidenceError(f"official evaluator {label} is invalid")
    if type(value) is float:
        try:
            if not math.isnan(value) and value >= 0:
                return value
        except OverflowError:
            pass
    raise OutcomeEvidenceError(f"official evaluator {label} is invalid")


def _mapping_has_nonfinite_values(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    try:
        canonical_json_sha256(value)
    except ValueError:
        return True
    return False


def _require_task(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise OutcomeEvidenceError("task is invalid")
    reject_credential_material(value)
    if "lehome" in value.casefold():
        raise OutcomeEvidenceError("task must not reference LeHome")
    return value


def _safe_artifact_path(value: str) -> bool:
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and len(path.parts) > 0
        and all(part not in (".", "..") and _ARTIFACT_PART.fullmatch(part) for part in path.parts)
        and not value.endswith(".incomplete")
    )


def _preserve_raw_evidence(value: object) -> object:
    if isinstance(value, bytearray):
        return bytes(value)
    try:
        return copy.deepcopy(value)
    except Exception:  # pragma: no cover - a hostile object must still be quarantined
        return repr(value)


def _raw_hash(value: object) -> str:
    try:
        return raw_evidence_sha256(value)
    except (TypeError, ValueError):
        return hashlib.sha256(repr(value).encode("utf-8", errors="backslashreplace")).hexdigest()


def _raw_episode_id(value: object) -> str | None:
    episode_id = value.get("episode_id") if isinstance(value, Mapping) else None
    if isinstance(episode_id, str) and _IDENTIFIER.fullmatch(episode_id) and "lehome" not in episode_id.casefold():
        return episode_id
    return None


def _raw_final_q_scores(value: object) -> object | None:
    q_score = value.get("q_score") if isinstance(value, Mapping) else None
    try:
        canonical_json_sha256(q_score)
    except (TypeError, ValueError):
        return None
    return copy.deepcopy(q_score)


def _raw_metrics(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    metrics = {key: value[key] for key in ("q_score", "time", "agent_distance", "normalized_agent_distance") if key in value}
    try:
        canonical_json_sha256(metrics)
    except (TypeError, ValueError):
        return None
    return copy.deepcopy(metrics)


def _raw_rollout_id(value: object) -> int | None:
    rollout_id = value.get("rollout_id") if isinstance(value, Mapping) else None
    return rollout_id if type(rollout_id) is int and rollout_id >= 0 else None


def raw_evidence_sha256(value: object) -> str:
    """Hash evidence exactly as retained, including upstream non-finite metrics.

    Normal mappings use the repository's strict canonical JSON.  A mapping with
    upstream ``Infinity`` uses the deterministic JSON spelling emitted by Python's
    encoder (sorted keys, compact separators, ``allow_nan=True``); it is the one
    safe retained representation used by envelope encoding and decoding.
    """

    if isinstance(value, bytes):
        return hashlib.sha256(value).hexdigest()
    if isinstance(value, bytearray):
        return hashlib.sha256(bytes(value)).hexdigest()
    if isinstance(value, str):
        return hashlib.sha256(value.encode("utf-8")).hexdigest()
    try:
        return canonical_json_sha256(value)
    except ValueError:
        try:
            encoded = json.dumps(
                value,
                allow_nan=True,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise ValueError("raw evidence has no retained JSON representation") from error
        return hashlib.sha256(encoded).hexdigest()


def _quarantine_file_evidence(
    reason: str, *, filename: str, disposition: str
) -> ClassifiedOutcome:
    raw = b""
    return ClassifiedOutcome(
        outcome=Outcome.QUARANTINE,
        reason=f"official evaluator evidence is unreadable: {disposition}",
        raw_evidence=raw,
        raw_evidence_sha256=_raw_hash(raw),
        episode_id=None,
        rollout_id=None,
        final_q_scores=None,
        evaluator_metrics=None,
        evaluator_identity=None,
        provenance=_provenance("file", disposition, "unreadable", raw, basename=filename),
        provenance_attestation=None,
    )


def _provenance(
    origin: str, disposition: str, reason_code: str, raw_evidence: object, *, basename: str | None = None
) -> Mapping[str, object]:
    if origin not in ("mapping", "file") or disposition not in (
        "mapping", "regular", "incomplete", "symlink", "unreadable"
    ):
        raise OutcomeEvidenceError("classification provenance is invalid")
    return {
        "origin": origin,
        "disposition": disposition,
        "basename": basename,
        "reason_code": reason_code,
        "diagnostic": reason_code,
        "raw_evidence_sha256": _raw_hash(raw_evidence),
    }


def revalidate_classification(
    raw_evidence: object, provenance: Mapping[str, object], *, task_manifest: Mapping[str, object]
) -> ClassifiedOutcome:
    """Re-run Task3 classification while preserving legitimate file disposition."""

    if not _valid_provenance(provenance, raw_evidence):
        raise OutcomeEvidenceError("classification provenance is invalid")
    origin = provenance["origin"]
    disposition = provenance["disposition"]
    if origin == "mapping":
        return classify_outcome(raw_evidence, task_manifest=task_manifest)
    if provenance.get("reason_code") == "official_evaluator_v1":
        # The controller normalized the pinned upstream eval.py record after
        # validating its scheduled path and raw fields.  Re-derive the normalized
        # terminal disposition without replacing its file-origin provenance.
        return replace(
            classify_outcome(raw_evidence, task_manifest=task_manifest),
            provenance=dict(provenance),
        )
    basename = provenance["basename"]
    assert isinstance(basename, str)
    if disposition == "incomplete":
        return ClassifiedOutcome(
            Outcome.QUARANTINE, "official evaluator evidence is still incomplete", raw_evidence,
            _raw_hash(raw_evidence), None, None, None, None, None,
            _provenance("file", "incomplete", "incomplete", raw_evidence, basename=basename),
            None,
        )
    if disposition in ("symlink", "unreadable"):
        return ClassifiedOutcome(
            Outcome.QUARANTINE, f"official evaluator evidence is unreadable: {disposition}", raw_evidence,
            _raw_hash(raw_evidence), None, None, None, None, None,
            _provenance("file", disposition, "unreadable", raw_evidence, basename=basename),
            None,
        )
    classified = classify_outcome(raw_evidence, task_manifest=task_manifest)
    return replace(classified, provenance=_provenance("file", "regular", "terminal" if classified.outcome is not Outcome.QUARANTINE else "invalid", raw_evidence, basename=basename))


def _attest_file(classified: ClassifiedOutcome, episode_key: str, contract: RolloutContract, authenticator: ProvenanceAuthenticator) -> ClassifiedOutcome:
    if not isinstance(episode_key, str) or not _IDENTIFIER.fullmatch(episode_key):
        raise OutcomeEvidenceError("episode key is invalid")
    return replace(classified, provenance_attestation=authenticator.sign(canonical_attestation_payload(contract, episode_key, _classification_fields(classified))))


def _classification_fields(classified: ClassifiedOutcome) -> Mapping[str, object]:
    return {"episode_id": classified.episode_id, "rollout_id": classified.rollout_id, "evaluator_identity": classified.evaluator_identity, "outcome": classified.outcome.value, "reason": classified.reason, "raw_evidence_sha256": classified.raw_evidence_sha256, "final_q_scores": classified.final_q_scores, "evaluator_metrics": classified.evaluator_metrics, "provenance": classified.provenance}


def _valid_provenance(value: object, raw_evidence: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"origin", "disposition", "basename", "reason_code", "diagnostic", "raw_evidence_sha256"}:
        return False
    if not isinstance(value.get("diagnostic"), str): return False
    if value.get("raw_evidence_sha256") != _raw_hash(raw_evidence): return False
    if value.get("origin") == "mapping": return value.get("disposition") == "mapping" and value.get("basename") is None
    return value.get("origin") == "file" and value.get("disposition") in ("regular", "incomplete", "symlink", "unreadable") and isinstance(value.get("basename"), str) and bool(value["basename"])


def _read_evidence_file(path: Path) -> tuple[bytes, str]:
    """Read one exact regular file through descriptor-anchored components only."""

    if not path.is_absolute() or any(part in (".", "..") for part in path.parts):
        raise OutcomeEvidenceError("evaluator evidence path traversal is forbidden")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_fd = os.open(path.anchor, flags)
    except OSError as error:
        raise OutcomeEvidenceError(f"official evaluator evidence is unreadable: {error}") from error
    try:
        for part in path.parts[1:-1]:
            try:
                mode = os.stat(part, dir_fd=directory_fd, follow_symlinks=False).st_mode
            except OSError as error:
                raise OutcomeEvidenceError(f"official evaluator evidence is unreadable: {error}") from error
            if stat.S_ISLNK(mode):
                raise OutcomeEvidenceError("evaluator evidence path must not contain symlinks")
            if not stat.S_ISDIR(mode):
                raise OutcomeEvidenceError("evaluator evidence path has a non-directory component")
            child_fd = os.open(part, flags, dir_fd=directory_fd)
            if not stat.S_ISDIR(os.fstat(child_fd).st_mode):
                os.close(child_fd)
                raise OutcomeEvidenceError("evaluator evidence path descriptor is invalid")
            os.close(directory_fd)
            directory_fd = child_fd
        name = path.name
        if not name:
            raise OutcomeEvidenceError("evaluator evidence path is invalid")
        mode = os.stat(name, dir_fd=directory_fd, follow_symlinks=False).st_mode
        if stat.S_ISLNK(mode):
            raise OutcomeEvidenceError("evaluator evidence path must not be a symlink")
        if not stat.S_ISREG(mode):
            raise OutcomeEvidenceError("evaluator evidence must be a regular file")
        file_fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd)
        try:
            if not stat.S_ISREG(os.fstat(file_fd).st_mode):
                raise OutcomeEvidenceError("evaluator evidence descriptor must be a regular file")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(file_fd, 65536)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks), name
        finally:
            os.close(file_fd)
    except OSError as error:
        raise OutcomeEvidenceError(f"official evaluator evidence is unreadable: {error}") from error
    finally:
        os.close(directory_fd)


def _reject_lehome_material(value: object) -> None:
    if isinstance(value, str):
        if "lehome" in value.casefold():
            raise OutcomeEvidenceError("official evaluator evidence must not reference LeHome")
    elif isinstance(value, Mapping):
        for key, nested in value.items():
            _reject_lehome_material(key)
            _reject_lehome_material(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            _reject_lehome_material(nested)
