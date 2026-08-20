"""Fail-closed, immutable progress evidence for deterministic AWR replay."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType

from lehome_train.io import canonical_json_sha256


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SCORE_KINDS = {"progress", "advantage"}
_EVIDENCE_RECEIPT_FIELDS = {
    "schema_version", "kind", "evidence_sha256", "mixture_id",
    "mixture_manifest_sha256", "authenticated_principal_sha256",
    "readback_receipt_sha256", "readback_verified",
}


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _exact(value: Mapping[str, object], keys: set[str], *, label: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{label} has unknown or missing field")


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a SHA-256")
    return value


def _identifier(value: object, *, label: str) -> str:
    if type(value) is not str or not value or "/" in value or "\\" in value:
        raise ValueError(f"{label} is invalid")
    return value


def _relative_path(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} is unsafe")
    path = PurePosixPath(value)
    if path.is_absolute() or "." in path.parts or ".." in path.parts or "\\" in value:
        raise ValueError(f"{label} is unsafe")
    return value


def _finite_number(value: object, *, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")
    return float(value)


@dataclass(frozen=True, slots=True)
class AwrReplayConfig:
    """Explicit replay transformation; no model-loss field is fabricated."""

    temperature: float
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        temperature = _finite_number(self.temperature, label="AWR temperature")
        minimum = _finite_number(self.minimum, label="AWR minimum")
        maximum = _finite_number(self.maximum, label="AWR maximum")
        if temperature <= 0.0:
            raise ValueError("AWR temperature must be positive")
        if minimum <= 0.0 or maximum <= 0.0 or minimum > maximum:
            raise ValueError("AWR clip bounds are invalid")
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    def to_dict(self) -> dict[str, float]:
        return {
            "temperature": self.temperature,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class ProgressEvidence:
    episode_id: str
    lineage_id: str
    score_kind: str
    score: float
    provenance_path: str
    provenance_sha256: str


@dataclass(frozen=True, slots=True)
class ProgressEvidenceSet:
    mixture_id: str
    mixture_manifest_sha256: str
    identity_sha256: str
    episodes: Mapping[str, ProgressEvidence]

    def validate_binding(self, *, mixture_id: str, mixture_manifest_sha256: str) -> None:
        if self.mixture_id != mixture_id or self.mixture_manifest_sha256 != mixture_manifest_sha256:
            raise ValueError("AWR evidence does not match the immutable runtime mixture")

    def weights(self, config: AwrReplayConfig) -> Mapping[str, float]:
        """Return bounded positive replay weights without exposing loss weights."""
        lower, upper = math.log(config.minimum), math.log(config.maximum)
        result: dict[str, float] = {}
        for episode_id, evidence in self.episodes.items():
            exponent = evidence.score / config.temperature
            if exponent <= lower:
                result[episode_id] = config.minimum
            elif exponent >= upper:
                result[episode_id] = config.maximum
            else:
                result[episode_id] = math.exp(exponent)
        return MappingProxyType(result)


def canonical_evidence_sha256(document: object) -> str:
    """The semantic immutable identity accepted by the runtime binding."""
    return canonical_json_sha256(document)


def authenticated_progress_evidence_receipt_sha256(receipt: Mapping[str, object]) -> str:
    """Validate the controller-facing read-back receipt for progress evidence.

    A locally generated evidence JSON is not sufficient to admit an AWR-style
    replay job: the receipt records both the Hub readback and the authenticated
    controller principal without storing a secret.
    """
    if not isinstance(receipt, Mapping) or set(receipt) != _EVIDENCE_RECEIPT_FIELDS:
        raise ValueError("AWR progress evidence receipt has unknown or missing field")
    if receipt.get("schema_version") != 1 or receipt.get("kind") != "lehome_awr_progress_evidence_receipt" or receipt.get("readback_verified") is not True:
        raise ValueError("AWR progress evidence receipt is not read-back verified")
    for key in (
        "evidence_sha256", "mixture_id", "mixture_manifest_sha256",
        "authenticated_principal_sha256", "readback_receipt_sha256",
    ):
        _sha256(receipt.get(key), label=f"AWR progress evidence receipt {key}")
    return canonical_json_sha256(dict(receipt))


def _evidence_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("AWR evidence path is unsafe")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError("AWR evidence path is unavailable") from error
    if resolved != path or not resolved.is_file():
        raise ValueError("AWR evidence path is unsafe")
    return resolved


def load_progress_evidence(
    path: str | Path,
    *,
    expected_sha256: str,
    mixture_id: str,
    mixture_manifest_sha256: str,
) -> ProgressEvidenceSet:
    """Load one exact train-only episode evidence document, or reject it."""
    selected = _evidence_path(path)
    expected = _sha256(expected_sha256, label="AWR expected evidence hash")
    expected_mixture = _sha256(mixture_id, label="AWR mixture ID")
    expected_manifest = _sha256(mixture_manifest_sha256, label="AWR mixture manifest hash")
    try:
        document = json.loads(selected.read_text(encoding="utf-8"), object_pairs_hook=_strict_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid AWR progress evidence") from error
    if not isinstance(document, dict):
        raise ValueError("invalid AWR progress evidence")
    if canonical_evidence_sha256(document) != expected:
        raise ValueError("AWR progress evidence digest mismatch")
    _exact(
        document,
        {"schema_version", "kind", "mixture_id", "mixture_manifest_sha256", "episodes"},
        label="AWR progress evidence",
    )
    if type(document["schema_version"]) is not int or document["schema_version"] != 1 or document["kind"] != "lehome_awr_progress_evidence":
        raise ValueError("AWR progress evidence schema drift")
    if document["mixture_id"] != expected_mixture or document["mixture_manifest_sha256"] != expected_manifest:
        raise ValueError("AWR progress evidence mixture binding mismatch")
    raw_episodes = document["episodes"]
    if not isinstance(raw_episodes, list) or not raw_episodes:
        raise ValueError("AWR progress evidence episodes are invalid")
    episodes: dict[str, ProgressEvidence] = {}
    lineages: set[str] = set()
    for raw in raw_episodes:
        if not isinstance(raw, dict):
            raise ValueError("AWR episode evidence is invalid")
        _exact(
            raw,
            {"episode_id", "lineage_id", "split", "score_kind", "score", "provenance_path", "provenance_sha256"},
            label="AWR episode evidence",
        )
        episode_id = _identifier(raw["episode_id"], label="AWR episode ID")
        lineage_id = _identifier(raw["lineage_id"], label="AWR lineage ID")
        if raw["split"] != "train":
            raise ValueError("AWR evidence must be train lineage only")
        if raw["score_kind"] not in _SCORE_KINDS:
            raise ValueError("AWR score kind is invalid")
        if episode_id in episodes or lineage_id in lineages:
            raise ValueError("duplicate AWR episode evidence")
        episodes[episode_id] = ProgressEvidence(
            episode_id=episode_id,
            lineage_id=lineage_id,
            score_kind=str(raw["score_kind"]),
            score=_finite_number(raw["score"], label="AWR score"),
            provenance_path=_relative_path(raw["provenance_path"], label="AWR provenance path"),
            provenance_sha256=_sha256(raw["provenance_sha256"], label="AWR provenance hash"),
        )
        lineages.add(lineage_id)
    return ProgressEvidenceSet(
        mixture_id=expected_mixture,
        mixture_manifest_sha256=expected_manifest,
        identity_sha256=expected,
        episodes=MappingProxyType(episodes),
    )
