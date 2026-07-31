"""Canonical identity for the verified train-only GR00T normalization bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from lehome_train.io import canonical_json_sha256, sha256_file


_VALIDATION_PATHS = (
    "meta/lehome_groot_modality.py",
    "meta/relative_stats.json",
    "meta/stats.json",
    "meta/validation_report.json",
)
_NORMALIZATION_PATHS = _VALIDATION_PATHS[:-1]
_SHA256_LENGTH = 64


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("prepared normalization metadata contains a duplicate field")
        result[key] = value
    return result


def _prepared_hashes(dataset: Path) -> Mapping[str, object]:
    path = dataset / "meta" / "prepared_hashes.json"
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("prepared normalization hash allowlist is unavailable") from None
    if (
        not isinstance(payload, Mapping)
        or set(payload) != {"schema_version", "artifacts"}
        or payload.get("schema_version") != 1
        or not isinstance(payload.get("artifacts"), Mapping)
    ):
        raise ValueError("prepared normalization hash allowlist is incompatible")
    artifacts = payload["artifacts"]
    assert isinstance(artifacts, Mapping)
    if set(artifacts) != set(_VALIDATION_PATHS):
        raise ValueError("prepared normalization hash allowlist is incompatible")
    return artifacts


def normalization_identity(dataset_path: str | Path) -> str:
    """Hash the exact verified modality/statistics identities used by GR00T."""

    dataset = Path(dataset_path)
    if not dataset.is_dir() or dataset.is_symlink():
        raise ValueError("prepared normalization dataset is unavailable")
    recorded = _prepared_hashes(dataset)
    identities: list[dict[str, str]] = []
    for relative_path in _VALIDATION_PATHS:
        expected = recorded[relative_path]
        if (
            type(expected) is not str
            or len(expected) != _SHA256_LENGTH
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise ValueError("prepared normalization hash allowlist is incompatible")
        artifact = dataset / relative_path
        if artifact.is_symlink() or not artifact.is_file():
            raise ValueError("prepared normalization artifact is unavailable")
        if sha256_file(artifact) != expected:
            raise ValueError("normalization artifact hash mismatch")
        if relative_path in _NORMALIZATION_PATHS:
            identities.append(
                {"relative_path": relative_path, "sha256": expected}
            )
    return canonical_json_sha256(
        {"schema_version": 1, "artifacts": identities}
    )
