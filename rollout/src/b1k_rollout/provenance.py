"""Out-of-band authentication for file-origin rollout provenance."""

from __future__ import annotations

import hashlib
import hmac
import os
import json
import re
import stat
from collections.abc import Mapping
from pathlib import Path

from b1k_rollout.contracts import RolloutContract

_ISSUER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ProvenanceAuthenticationError(ValueError):
    pass


class ProvenanceAuthenticator:
    """A non-serializable HMAC-SHA256 signer/verifier.

    Interface for controllers: construct once from a 32-byte local key and pass the
    same instance to ``classify_outcome_file``, ``write_episode_envelope``,
    ``load_episode_envelopes``, and ``publish_release``.  The envelope carries only
    issuer, one-way key id, and MAC.
    """

    def __init__(self, key: bytes, *, issuer: str) -> None:
        if not isinstance(key, bytes) or len(key) != 32:
            raise ProvenanceAuthenticationError("provenance key is invalid")
        if not isinstance(issuer, str) or not _ISSUER.fullmatch(issuer):
            raise ProvenanceAuthenticationError("provenance issuer is invalid")
        self._key = key
        self.issuer = issuer
        self.key_id = hashlib.sha256(b"b1k-provenance-key-v1\0" + key).hexdigest()

    def sign(self, payload: Mapping[str, object]) -> Mapping[str, str]:
        mac = hmac.new(self._key, _attestation_bytes(payload), hashlib.sha256).hexdigest()
        return {"issuer": self.issuer, "key_id": self.key_id, "mac": mac}

    def verify(self, payload: Mapping[str, object], attestation: object) -> None:
        if not isinstance(attestation, Mapping) or set(attestation) != {"issuer", "key_id", "mac"}:
            raise ProvenanceAuthenticationError("provenance attestation is invalid")
        if attestation.get("issuer") != self.issuer or attestation.get("key_id") != self.key_id:
            raise ProvenanceAuthenticationError("provenance attestation is not trusted")
        mac = attestation.get("mac")
        if not isinstance(mac, str) or len(mac) != 64:
            raise ProvenanceAuthenticationError("provenance attestation is invalid")
        expected = hmac.new(self._key, _attestation_bytes(payload), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, mac):
            raise ProvenanceAuthenticationError("provenance attestation is invalid")


def _attestation_bytes(payload: Mapping[str, object]) -> bytes:
    """Deterministically encode the retained evidence fields for HMAC binding.

    Official quarantine evidence may preserve a positive Infinity metric.  This
    encoding deliberately uses Python's stable JSON spellings for non-finite
    floats while retaining sorted keys and compact UTF-8 output.
    """

    try:
        return json.dumps(
            payload,
            allow_nan=True,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ProvenanceAuthenticationError("provenance attestation input is invalid") from error


def load_local_provenance_key(path: Path) -> bytes:
    """Read exactly one owner-only 32-byte key without following symlinks."""

    path = Path(path)
    if not path.is_absolute() or any(part in (".", "..") for part in path.parts):
        raise ProvenanceAuthenticationError("provenance key path is invalid")
    parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path.anchor, parent_flags)
    except OSError as error:
        raise ProvenanceAuthenticationError("provenance key file is invalid") from error
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, parent_flags, dir_fd=fd)
            os.close(fd)
            fd = child
        final_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        key_fd = os.open(path.name, final_flags, dir_fd=fd)
        try:
            info = os.fstat(key_fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size != 32:
                raise ProvenanceAuthenticationError("provenance key file is invalid")
            key = os.read(key_fd, 33)
            if len(key) != 32 or os.read(key_fd, 1):
                raise ProvenanceAuthenticationError("provenance key file is invalid")
            return key
        finally:
            os.close(key_fd)
    except OSError as error:
        raise ProvenanceAuthenticationError("provenance key file is invalid") from error
    finally:
        os.close(fd)


def canonical_attestation_payload(
    contract: RolloutContract, episode_key: str, fields: Mapping[str, object]
) -> Mapping[str, object]:
    """One payload shared by producer, writer, loader, and publisher."""

    if not isinstance(contract, RolloutContract) or not _ISSUER.fullmatch(episode_key):
        raise ProvenanceAuthenticationError("provenance attestation input is invalid")
    bound = (
        "episode_id", "rollout_id", "evaluator_identity", "outcome", "reason",
        "raw_evidence_sha256", "final_q_scores", "evaluator_metrics", "provenance",
    )
    return {
        "schema_version": 3,
        "contract_identity": contract.identity,
        "campaign_id": contract.campaign_id,
        "model_commit": contract.model_commit,
        "task_manifest_sha256": contract.task_manifest_sha256,
        "episode_key": episode_key,
        **{name: fields.get(name) for name in bound},
    }
