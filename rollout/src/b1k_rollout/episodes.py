"""Atomic, canonical on-disk envelopes for completed and quarantined episodes."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from b1k_rollout.contracts import RolloutContract
from b1k_rollout.identity import canonical_json_sha256, reject_credential_material, require_sha256
from b1k_rollout.outcomes import ClassifiedOutcome, Outcome, raw_evidence_sha256
from b1k_rollout.provenance import ProvenanceAuthenticationError, ProvenanceAuthenticator, canonical_attestation_payload


class EpisodeIntegrityError(ValueError):
    """The episode store or its envelope cannot be trusted."""


@dataclass(frozen=True, slots=True)
class VerifiedArtifact:
    path: Path
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class EpisodeEnvelope:
    episode_key: str
    episode_id: str | None
    rollout_id: int | None
    evaluator_identity: Mapping[str, object] | None
    outcome: Outcome
    reason: str
    raw_evidence: object
    raw_evidence_sha256: str
    final_q_scores: object | None
    evaluator_metrics: Mapping[str, object] | None
    provenance: Mapping[str, object]
    provenance_attestation: Mapping[str, str] | None
    canonical_sha256: str


_SCHEMA_VERSION = 2
_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ENVELOPE_FIELDS = frozenset(
    (
        "schema_version",
        "episode_key",
        "episode_id",
        "rollout_id",
        "evaluator_identity",
        "outcome",
        "reason",
        "raw_evidence",
        "raw_evidence_encoding",
        "raw_evidence_sha256",
        "final_q_scores",
        "evaluator_metrics",
        "evaluator_metrics_encoding",
        "provenance",
        "provenance_attestation",
        "canonical_sha256",
    )
)


def write_episode_envelope(
    root: Path, episode_key: str, classified: ClassifiedOutcome, *, contract: RolloutContract | None = None, authenticator: ProvenanceAuthenticator | None = None
) -> Path:
    """Durably stage ``.incomplete`` then atomically publish one envelope.

    Episode keys are deliberately not paths.  The root and every discovered entry
    must be non-symlink regular filesystem objects; a staged file is retained on a
    failed write so the controller can quarantine and inspect it safely.
    """

    _validate_episode_key(episode_key)
    payload = _new_payload(episode_key, classified, contract=contract, authenticator=authenticator)
    encoded = (json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")
    root_fd = _open_safe_root(root)
    stage = f"{episode_key}.json.incomplete"
    final = f"{episode_key}.json"
    try:
        _ensure_absent(root_fd, stage)
        _ensure_absent(root_fd, final)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        stage_fd = os.open(stage, flags, 0o600, dir_fd=root_fd)
        try:
            if not stat.S_ISREG(os.fstat(stage_fd).st_mode):
                raise EpisodeIntegrityError("staged episode envelope must be a regular file")
            _write_all(stage_fd, encoded)
            os.fsync(stage_fd)
        finally:
            os.close(stage_fd)
        os.fsync(root_fd)
        os.replace(stage, final, src_dir_fd=root_fd, dst_dir_fd=root_fd)
        os.fsync(root_fd)
    except OSError as error:
        raise EpisodeIntegrityError(f"unable to atomically write episode envelope: {error}") from error
    finally:
        os.close(root_fd)
    return root / final


def load_episode_envelopes(root: Path, *, contract: RolloutContract | None = None, authenticator: ProvenanceAuthenticator | None = None) -> tuple[EpisodeEnvelope, ...]:
    """Load all envelopes, rejecting partial files, symlinks, and identity conflicts."""

    root_fd = _open_safe_root(root)
    try:
        entries = sorted(os.listdir(root_fd))
        if any(entry.endswith(".incomplete") for entry in entries):
            raise EpisodeIntegrityError("incomplete episode envelope is present")
        envelopes: list[EpisodeEnvelope] = []
        episode_keys: set[str] = set()
        episode_ids: dict[str, str] = {}
        evaluator_identities: dict[str, str] = {}
        for name in entries:
            _reject_symlink_entry(root_fd, name)
            if not name.endswith(".json"):
                raise EpisodeIntegrityError(f"unexpected episode store entry: {name}")
            payload = _read_json_file(root_fd, name)
            envelope = _parse_envelope(payload, filename=name, contract=contract, authenticator=authenticator)
            if envelope.episode_key in episode_keys:
                raise EpisodeIntegrityError("duplicate episode key")
            episode_keys.add(envelope.episode_key)
            if envelope.episode_id is not None:
                previous_key = episode_ids.get(envelope.episode_id)
                if previous_key is not None and previous_key != envelope.episode_key:
                    raise EpisodeIntegrityError("conflicting duplicate episode id")
                episode_ids[envelope.episode_id] = envelope.episode_key
            if envelope.evaluator_identity is not None:
                identity = canonical_json_sha256(envelope.evaluator_identity)
                previous_key = evaluator_identities.get(identity)
                if previous_key is not None and previous_key != envelope.episode_key:
                    raise EpisodeIntegrityError("conflicting duplicate evaluator identity")
                evaluator_identities[identity] = envelope.episode_key
            envelopes.append(envelope)
        return tuple(envelopes)
    finally:
        os.close(root_fd)


def verify_artifact_hashes(root: Path, artifact_hashes: Mapping[str, object]) -> None:
    """Verify that regular, non-symlink artifacts match their claimed SHA-256."""

    if not isinstance(artifact_hashes, Mapping) or not artifact_hashes:
        raise EpisodeIntegrityError("artifact hashes must be a non-empty object")
    root_fd = _open_safe_root(root)
    try:
        for name, expected in artifact_hashes.items():
            if not isinstance(name, str) or not _safe_relative_path(name):
                raise EpisodeIntegrityError("artifact path is unsafe")
            try:
                require_sha256(expected, label="artifact content hash")
            except ValueError as error:
                raise EpisodeIntegrityError(str(error)) from error
            actual = _hash_regular_file(root_fd, name)
            if actual != expected:
                raise EpisodeIntegrityError(f"artifact content hash mismatch: {name}")
    finally:
        os.close(root_fd)


def copy_verified_artifact(root: Path, name: str, destination: Path, expected: str) -> VerifiedArtifact:
    """Hash and copy one artifact through the same no-follow descriptor.

    The source path is resolved only once through descriptor-anchored directory
    components.  Parent replacements after that point cannot redirect either the
    hash or the copied bytes outside the verified file descriptor.
    """

    if not isinstance(name, str) or not _safe_relative_path(name):
        raise EpisodeIntegrityError("artifact path is unsafe")
    try:
        require_sha256(expected, label="artifact content hash")
    except ValueError as error:
        raise EpisodeIntegrityError("artifact content hash is invalid") from error
    root_fd = _open_safe_root(root)
    source_fd: int | None = None
    destination_fd: int | None = None
    copied = False
    try:
        source_fd = _open_regular_relative(root_fd, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            _write_all(destination_fd, chunk)
        actual = digest.hexdigest()
        if actual != expected:
            raise EpisodeIntegrityError("artifact content hash mismatch")
        copied = True
        return VerifiedArtifact(destination, size, actual)
    except OSError as error:
        raise EpisodeIntegrityError("cannot safely copy artifact") from error
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        if source_fd is not None:
            os.close(source_fd)
        os.close(root_fd)
        if not copied:
            destination.unlink(missing_ok=True)


def _new_payload(episode_key: str, classified: ClassifiedOutcome, *, contract: RolloutContract | None, authenticator: ProvenanceAuthenticator | None) -> dict[str, object]:
    _validate_classified(classified)
    raw_evidence, raw_evidence_encoding = _json_evidence(classified.raw_evidence)
    evaluator_metrics, evaluator_metrics_encoding = _json_evidence(classified.evaluator_metrics)
    payload: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "episode_key": episode_key,
        "episode_id": classified.episode_id,
        "rollout_id": classified.rollout_id,
        "evaluator_identity": classified.evaluator_identity,
        "outcome": classified.outcome.value,
        "reason": classified.reason,
        "raw_evidence": raw_evidence,
        "raw_evidence_encoding": raw_evidence_encoding,
        "raw_evidence_sha256": classified.raw_evidence_sha256,
        "final_q_scores": classified.final_q_scores,
        "evaluator_metrics": evaluator_metrics,
        "evaluator_metrics_encoding": evaluator_metrics_encoding,
        "provenance": classified.provenance,
        "provenance_attestation": classified.provenance_attestation,
    }
    if classified.provenance.get("origin") == "file":
        if authenticator is None or contract is None: raise EpisodeIntegrityError("file provenance requires an authenticator and contract")
        try:
            authenticator.verify(
                canonical_attestation_payload(contract, episode_key, _attestation_fields(payload)),
                classified.provenance_attestation,
            )
        except ProvenanceAuthenticationError as error:
            raise EpisodeIntegrityError("file provenance attestation is invalid") from error
    elif authenticator is not None and contract is not None:
        if classified.provenance_attestation is not None:
            raise EpisodeIntegrityError("mapping provenance attestation must be minted by the envelope writer")
        payload["provenance_attestation"] = authenticator.sign(
            canonical_attestation_payload(contract, episode_key, _attestation_fields(payload))
        )
    reject_credential_material(payload)
    _reject_lehome(payload)
    payload["canonical_sha256"] = canonical_json_sha256(payload)
    return payload


def _parse_envelope(value: object, *, filename: str, contract: RolloutContract | None, authenticator: ProvenanceAuthenticator | None) -> EpisodeEnvelope:
    if not isinstance(value, Mapping) or set(value) != _ENVELOPE_FIELDS:
        raise EpisodeIntegrityError("episode envelope fields are invalid")
    if value.get("schema_version") != _SCHEMA_VERSION:
        raise EpisodeIntegrityError("episode envelope schema version is invalid")
    episode_key = value.get("episode_key")
    if not isinstance(episode_key, str):
        raise EpisodeIntegrityError("episode key is invalid")
    _validate_episode_key(episode_key)
    if filename != f"{episode_key}.json":
        raise EpisodeIntegrityError("episode key does not match filename")
    episode_id = value.get("episode_id")
    if episode_id is not None and (not isinstance(episode_id, str) or not _KEY.fullmatch(episode_id)):
        raise EpisodeIntegrityError("episode id is invalid")
    try:
        outcome = Outcome(value.get("outcome"))
    except ValueError as error:
        raise EpisodeIntegrityError("episode outcome is invalid") from error
    if not isinstance(value.get("reason"), str):
        raise EpisodeIntegrityError("episode reason is invalid")
    try:
        require_sha256(value.get("raw_evidence_sha256"), label="raw evidence hash")
    except ValueError as error:
        raise EpisodeIntegrityError(str(error)) from error
    payload_without_hash = {key: nested for key, nested in value.items() if key != "canonical_sha256"}
    if value.get("canonical_sha256") != canonical_json_sha256(payload_without_hash):
        raise EpisodeIntegrityError("episode envelope canonical hash is stale or invalid")
    raw_evidence = _decode_envelope_evidence(
        value.get("raw_evidence"), value.get("raw_evidence_encoding")
    )
    _scan_decoded_retained_evidence(raw_evidence)
    evaluator_metrics = _decode_envelope_evidence(
        value.get("evaluator_metrics"), value.get("evaluator_metrics_encoding")
    )
    _scan_decoded_retained_evidence(evaluator_metrics)
    _scan_decoded_retained_evidence(value.get("final_q_scores"))
    if value["raw_evidence_sha256"] != raw_evidence_sha256(raw_evidence):
        raise EpisodeIntegrityError("raw evidence hash does not match preserved evidence")
    evaluator_identity = _validate_evaluator_identity(value.get("evaluator_identity"))
    provenance = _validate_provenance(value.get("provenance"), raw_evidence)
    attestation = value.get("provenance_attestation")
    if authenticator is not None and contract is not None:
        try: authenticator.verify(canonical_attestation_payload(contract, episode_key, _attestation_fields(value)), attestation)
        except ProvenanceAuthenticationError as error: raise EpisodeIntegrityError("file provenance attestation is invalid") from error
    elif provenance["origin"] == "file":
        raise EpisodeIntegrityError("file provenance requires an authenticator and contract")
    elif attestation is not None:
        raise EpisodeIntegrityError("mapping provenance attestation requires an authenticator and contract")
    if outcome is not Outcome.QUARANTINE and evaluator_identity is None:
        raise EpisodeIntegrityError("terminal episode envelope requires evaluator identity")
    if evaluator_identity is not None:
        _validate_identity_against_raw(evaluator_identity, raw_evidence)
    if outcome is not Outcome.QUARANTINE:
        _validate_terminal_envelope_against_raw(
            raw_evidence,
            outcome=outcome,
            reason=value["reason"],
            episode_id=episode_id,
            rollout_id=_validate_envelope_rollout_id(value.get("rollout_id")),
            final_q_scores=value["final_q_scores"],
            evaluator_metrics=evaluator_metrics,
            evaluator_identity=evaluator_identity,
        )
    reject_credential_material(value)
    _reject_lehome(value)
    return EpisodeEnvelope(
        episode_key=episode_key,
        episode_id=episode_id,
        rollout_id=_validate_envelope_rollout_id(value.get("rollout_id")),
        evaluator_identity=evaluator_identity,
        outcome=outcome,
        reason=value["reason"],
        raw_evidence=raw_evidence,
        raw_evidence_sha256=value["raw_evidence_sha256"],
        final_q_scores=value["final_q_scores"],
        evaluator_metrics=evaluator_metrics,
        provenance=provenance,
        provenance_attestation=dict(attestation) if isinstance(attestation, Mapping) else None,
        canonical_sha256=value["canonical_sha256"],
    )


def _validate_classified(classified: ClassifiedOutcome) -> None:
    if not isinstance(classified, ClassifiedOutcome):
        raise EpisodeIntegrityError("classified outcome is invalid")
    if classified.episode_id is not None:
        _validate_episode_key(classified.episode_id)
    _validate_envelope_rollout_id(classified.rollout_id)
    _validate_evaluator_identity(classified.evaluator_identity)
    try:
        require_sha256(classified.raw_evidence_sha256, label="raw evidence hash")
    except ValueError as error:
        raise EpisodeIntegrityError(str(error)) from error
    if classified.raw_evidence_sha256 != raw_evidence_sha256(classified.raw_evidence):
        raise EpisodeIntegrityError("raw evidence hash does not match the preserved evidence")
    _scan_decoded_retained_evidence(classified.raw_evidence)
    _scan_decoded_retained_evidence(classified.final_q_scores)
    _scan_decoded_retained_evidence(classified.evaluator_metrics)
    _scan_decoded_retained_evidence(classified.provenance)


def _json_evidence(value: object) -> tuple[object, str]:
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii"), "bytes"
    if isinstance(value, str):
        return value, "text"
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        try:
            return (
                json.dumps(
                    value,
                    allow_nan=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                "nonfinite_json",
            )
        except (TypeError, ValueError):
            raise EpisodeIntegrityError("raw evidence cannot be serialized safely") from None
    return value, "json"


def _decode_envelope_evidence(value: object, encoding: object) -> object:
    if encoding == "bytes":
        if not isinstance(value, str):
            raise EpisodeIntegrityError("raw byte evidence encoding is invalid")
        try:
            return base64.b64decode(value.encode("ascii"), validate=True)
        except (UnicodeEncodeError, ValueError) as error:
            raise EpisodeIntegrityError("raw byte evidence encoding is invalid") from error
    if encoding == "text":
        if not isinstance(value, str):
            raise EpisodeIntegrityError("raw text evidence encoding is invalid")
        return value
    if encoding == "json":
        try:
            json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise EpisodeIntegrityError("raw JSON evidence is invalid") from error
        return value
    if encoding == "nonfinite_json":
        if not isinstance(value, str):
            raise EpisodeIntegrityError("raw non-finite evidence encoding is invalid")
        try:
            return json.loads(value)
        except json.JSONDecodeError as error:
            raise EpisodeIntegrityError("raw non-finite evidence encoding is invalid") from error
    raise EpisodeIntegrityError("raw evidence encoding is invalid")


def _scan_decoded_retained_evidence(value: object) -> None:
    """Reject secrets and LeHome references hidden by byte/base64 transport."""

    if isinstance(value, bytes):
        # Replacement decoding preserves every non-UTF-8 position while exposing
        # ASCII credential and product markers, which are the only accepted tokens.
        value = value.decode("utf-8", errors="replace")
    try:
        reject_credential_material(value)
        _reject_lehome(value)
    except ValueError as error:
        raise EpisodeIntegrityError("decoded retained evidence is unsafe") from error


def _validate_envelope_rollout_id(value: object) -> int | None:
    if value is None:
        return None
    if type(value) is int and value >= 0:
        return value
    raise EpisodeIntegrityError("rollout id is invalid")


_EVALUATOR_IDENTITY_FIELDS = frozenset(
    (
        "campaign_id",
        "contract_identity",
        "instance_id",
        "instance_index",
        "mode",
        "model_commit",
        "rollout_id",
        "task",
    )
)


def _validate_evaluator_identity(value: object) -> Mapping[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != _EVALUATOR_IDENTITY_FIELDS:
        raise EpisodeIntegrityError("evaluator identity is invalid")
    for field in ("campaign_id", "contract_identity", "model_commit", "task", "mode"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise EpisodeIntegrityError("evaluator identity is invalid")
    for field in ("instance_id", "instance_index", "rollout_id"):
        if type(value.get(field)) is not int or value[field] < 0:
            raise EpisodeIntegrityError("evaluator identity is invalid")
    return dict(value)


def _validate_provenance(value: object, raw_evidence: object) -> Mapping[str, object]:
    fields = {"origin", "disposition", "basename", "reason_code", "diagnostic", "raw_evidence_sha256"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise EpisodeIntegrityError("classification provenance is invalid")
    if value.get("raw_evidence_sha256") != raw_evidence_sha256(raw_evidence):
        raise EpisodeIntegrityError("classification provenance raw evidence hash is invalid")
    origin, disposition, basename, reason = value.get("origin"), value.get("disposition"), value.get("basename"), value.get("reason_code")
    if not isinstance(reason, str) or not reason or not isinstance(value.get("diagnostic"), str):
        raise EpisodeIntegrityError("classification provenance is invalid")
    if origin == "mapping" and disposition == "mapping" and basename is None:
        return dict(value)
    if origin == "file" and disposition in ("regular", "incomplete", "symlink", "unreadable") and isinstance(basename, str) and _KEY.fullmatch(basename):
        return dict(value)
    raise EpisodeIntegrityError("classification provenance is invalid")


def _attestation_fields(value: Mapping[str, object]) -> Mapping[str, object]:
    """Every producer-controlled classification field is MAC-bound."""

    fields = (
        "schema_version", "episode_key", "episode_id", "rollout_id", "evaluator_identity",
        "outcome", "reason", "raw_evidence_sha256", "final_q_scores", "evaluator_metrics",
        "provenance",
    )
    # The envelope transports non-finite metric mappings as deterministic JSON
    # text, while the file classifier observes their decoded semantic values.
    # MAC every path over the latter so producer, writer, loader, and publisher
    # bind the same classification rather than transport encoding details.
    evaluator_metrics = _decode_envelope_evidence(
        value.get("evaluator_metrics"), value.get("evaluator_metrics_encoding")
    )
    return {
        field: (evaluator_metrics if field == "evaluator_metrics" else value.get(field))
        for field in fields
        if field != "schema_version"
    }


def _validate_identity_against_raw(identity: Mapping[str, object], raw_evidence: object) -> None:
    if identity != _identity_from_raw(raw_evidence):
        raise EpisodeIntegrityError("evaluator identity does not match raw evidence")


def _identity_from_raw(raw_evidence: object) -> Mapping[str, object]:
    raw_evidence = _raw_evidence_mapping(raw_evidence)
    try:
        contract = RolloutContract.from_mapping(raw_evidence.get("contract", {}))
        return {
            "campaign_id": contract.campaign_id,
            "contract_identity": contract.identity,
            "instance_id": raw_evidence["instance_id"],
            "instance_index": raw_evidence["instance_index"],
            "mode": raw_evidence["mode"],
            "model_commit": contract.model_commit,
            "rollout_id": raw_evidence["rollout_id"],
            "task": raw_evidence["task"],
        }
    except (KeyError, TypeError, ValueError) as error:
        raise EpisodeIntegrityError("evaluator identity does not match raw evidence") from error


def _raw_evidence_mapping(raw_evidence: object) -> Mapping[str, object]:
    if isinstance(raw_evidence, bytes):
        try:
            raw_evidence = json.loads(raw_evidence)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise EpisodeIntegrityError("evaluator identity does not match raw evidence") from error
    elif isinstance(raw_evidence, str):
        try:
            raw_evidence = json.loads(raw_evidence)
        except json.JSONDecodeError as error:
            raise EpisodeIntegrityError("evaluator identity does not match raw evidence") from error
    if not isinstance(raw_evidence, Mapping):
        raise EpisodeIntegrityError("evaluator identity does not match raw evidence")
    return raw_evidence


def _validate_terminal_envelope_against_raw(
    raw_evidence: object,
    *,
    outcome: Outcome,
    reason: object,
    episode_id: object,
    rollout_id: int | None,
    final_q_scores: object,
    evaluator_metrics: object,
    evaluator_identity: Mapping[str, object] | None,
) -> None:
    """Re-derive every terminal field from raw evidence before trusting an envelope."""

    raw = _raw_evidence_mapping(raw_evidence)
    try:
        success = raw["success"]
        completed = raw["completed"]
        raw_episode_id = raw["episode_id"]
        raw_rollout_id = raw["rollout_id"]
        metrics = {
            key: raw[key]
            for key in ("q_score", "time", "agent_distance", "normalized_agent_distance")
        }
    except KeyError as error:
        raise EpisodeIntegrityError("terminal envelope does not match retained raw evidence") from error
    if type(success) is not bool or completed is not True:
        raise EpisodeIntegrityError("terminal envelope does not match retained raw evidence")
    expected_outcome = Outcome.SUCCESS if success is True else Outcome.FAILURE
    if (
        outcome is not expected_outcome
        or reason != ""
        or episode_id != raw_episode_id
        or rollout_id != raw_rollout_id
        or final_q_scores != metrics["q_score"]
        or evaluator_metrics != metrics
        or evaluator_identity != _identity_from_raw(raw)
    ):
        raise EpisodeIntegrityError("terminal envelope does not match retained raw evidence")


def _validate_episode_key(value: str) -> None:
    if not _KEY.fullmatch(value) or "lehome" in value.casefold():
        raise EpisodeIntegrityError("episode key is unsafe")


def _open_safe_root(root: Path) -> int:
    root = Path(root)
    if not root.is_absolute():
        raise EpisodeIntegrityError("episode root must be an explicit absolute path")
    if any(component in (".", "..") for component in root.parts):
        raise EpisodeIntegrityError("episode root path traversal is forbidden")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_fd = os.open(root.anchor, flags)
    except OSError as error:
        raise EpisodeIntegrityError(f"cannot safely open filesystem root: {error}") from error
    try:
        for index, part in enumerate(root.parts[1:]):
            try:
                mode = os.stat(part, dir_fd=directory_fd, follow_symlinks=False).st_mode
            except FileNotFoundError:
                if index != len(root.parts[1:]) - 1:
                    raise EpisodeIntegrityError("episode root parent is missing") from None
                os.mkdir(part, mode=0o700, dir_fd=directory_fd)
                os.fsync(directory_fd)
                mode = os.stat(part, dir_fd=directory_fd, follow_symlinks=False).st_mode
            if stat.S_ISLNK(mode):
                raise EpisodeIntegrityError("episode root path must not contain symlinks")
            if not stat.S_ISDIR(mode):
                raise EpisodeIntegrityError("episode root path must contain directories only")
            child_fd = os.open(part, flags, dir_fd=directory_fd)
            if not stat.S_ISDIR(os.fstat(child_fd).st_mode):
                os.close(child_fd)
                raise EpisodeIntegrityError("episode root descriptor must be a directory")
            os.close(directory_fd)
            directory_fd = child_fd
        return directory_fd
    except OSError as error:
        os.close(directory_fd)
        raise EpisodeIntegrityError(f"cannot safely open episode root: {error}") from error
    except Exception:
        os.close(directory_fd)
        raise


def _ensure_absent(root_fd: int, name: str) -> None:
    try:
        os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise EpisodeIntegrityError("episode envelope already exists")


def _write_all(fd: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        written = os.write(fd, data[offset:])
        if written <= 0:
            raise EpisodeIntegrityError("staged episode envelope write made no progress")
        offset += written


def _reject_symlink_entry(root_fd: int, name: str) -> None:
    try:
        mode = os.stat(name, dir_fd=root_fd, follow_symlinks=False).st_mode
    except OSError as error:
        raise EpisodeIntegrityError(f"cannot inspect episode entry: {name}") from error
    if stat.S_ISLNK(mode):
        raise EpisodeIntegrityError("episode store must not contain symlinks")
    if not stat.S_ISREG(mode):
        raise EpisodeIntegrityError("episode store must contain regular files only")


def _read_json_file(root_fd: int, name: str) -> object:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(name, flags, dir_fd=root_fd)
    except OSError as error:
        raise EpisodeIntegrityError(f"cannot safely read episode envelope: {name}") from error
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise EpisodeIntegrityError("episode envelope descriptor must be a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(fd)
    try:
        return json.loads(b"".join(chunks))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise EpisodeIntegrityError("episode envelope is malformed JSON") from error


def _safe_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and bool(path.parts)
        and all(part not in (".", "..") and _KEY.fullmatch(part) for part in path.parts)
        and "lehome" not in value.casefold()
    )


def _hash_regular_file(root_fd: int, name: str) -> str:
    fd = _open_regular_relative(root_fd, name)
    try:
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.close(fd)


def _open_regular_relative(root_fd: int, name: str) -> int:
    """Open a relative file while refusing every symlink in its component chain."""

    parts = PurePosixPath(name).parts
    directory_fd = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            mode = os.stat(part, dir_fd=directory_fd, follow_symlinks=False).st_mode
            if stat.S_ISLNK(mode):
                raise EpisodeIntegrityError("artifact path must not contain symlinks")
            if not stat.S_ISDIR(mode):
                raise EpisodeIntegrityError("artifact path has a non-directory component")
            child_fd = os.open(
                part,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = child_fd
        final = parts[-1]
        mode = os.stat(final, dir_fd=directory_fd, follow_symlinks=False).st_mode
        if stat.S_ISLNK(mode):
            raise EpisodeIntegrityError("artifact path must not contain symlinks")
        if not stat.S_ISREG(mode):
            raise EpisodeIntegrityError("artifact must be a regular file")
        file_fd = os.open(final, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd)
        try:
            if not stat.S_ISREG(os.fstat(file_fd).st_mode):
                raise EpisodeIntegrityError("artifact descriptor must be a regular file")
            return file_fd
        except Exception:
            os.close(file_fd)
            raise
    except OSError as error:
        raise EpisodeIntegrityError(f"cannot safely read artifact: {name}") from error
    finally:
        os.close(directory_fd)


def _reject_lehome(value: object) -> None:
    if isinstance(value, str):
        if "lehome" in value.casefold():
            raise EpisodeIntegrityError("episode envelope must not reference LeHome")
    elif isinstance(value, Mapping):
        for key, nested in value.items():
            _reject_lehome(key)
            _reject_lehome(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_lehome(nested)
