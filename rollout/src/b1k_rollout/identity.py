"""Pinned identities and deterministic serialization for rollout contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence


BEHAVIOR_REVISION = "26f2c7ef7b9cf96bd0414f81e1e751e493762779"
GROOT_REVISION = "ace36d935b376fbf25cd56371e23877b95407c40"
MODEL_REPO = "ryanjin333/behavior1k-groot-n17-models"
DATASET_REPO = "ryanjin333/behavior1k-groot-n17-rollouts"

_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_SECRET_KEY = re.compile(
    r"(?:^|[_-])(token|secret|password|credential|authorization|api[_-]?key|private[_-]?key|access[_-]?key)(?:$|[_-])",
    re.IGNORECASE,
)
_SECRET_VALUE = re.compile(
    r"(?:\bBearer\s+\S+|\bapi[_-]?key\s*=|\btoken\s*=)",
    re.IGNORECASE,
)
_CREDENTIAL_VALUE_PATTERNS = (
    re.compile(r"gh[pous]_[A-Za-z0-9]{36}"),
    re.compile(r"ghr_[A-Za-z0-9]{76}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{82}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\bxox[a-zA-Z]-\d{10,13}-\d{10,13}-[A-Za-z0-9-]{20,}\b"),
    re.compile(r"\bdckr_pat_[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{34}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
)


def canonical_json_bytes(value: object) -> bytes:
    """Encode a supported JSON value deterministically as UTF-8 bytes."""

    return json.dumps(
        _canonical_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    """Return the SHA-256 digest of a canonical JSON value."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def require_immutable_commit(value: object, *, label: str) -> str:
    """Require a complete lower-case Git commit, never a mutable ref."""

    if not isinstance(value, str) or not _GIT_COMMIT.fullmatch(value):
        raise ValueError(f"{label} must be an immutable commit")
    return value


def require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a SHA-256 hash")
    return value


def require_image_digest(value: object) -> str:
    if not isinstance(value, str) or not _IMAGE_DIGEST.fullmatch(value):
        raise ValueError("image digest must be a sha256 digest")
    return value


def reject_credential_material(value: object) -> None:
    """Fail closed if a serializable value contains credential material."""

    if isinstance(value, str):
        if _contains_credential_value(value):
            raise ValueError("serialized contract must not contain credential material")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ValueError("serialized contract keys must be strings")
            if _SECRET_KEY.search(key):
                raise ValueError("serialized contract must not contain credential fields")
            reject_credential_material(nested)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            reject_credential_material(nested)


def _contains_credential_value(value: str) -> bool:
    return bool(_SECRET_VALUE.search(value)) or any(
        pattern.search(value) for pattern in _CREDENTIAL_VALUE_PATTERNS
    )


def require_b1k_repository(value: object, *, expected: str, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a repository name")
    if "lehome" in value.casefold():
        raise ValueError(f"{label} must not reference a LeHome repository")
    if value != expected:
        raise ValueError(f"{label} must equal {expected}")
    return value


def _canonical_value(value: object) -> object:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical JSON object keys must be strings")
        return {key: _canonical_value(nested) for key, nested in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical_value(nested) for nested in value]
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("canonical JSON numbers must be finite")
        return value
    raise TypeError("value is not supported by the canonical JSON contract")
