"""Immutable, fail-closed augmentation contracts for GR00T fine-tuning.

This module only selects the checked upstream color-jitter interface.  It does
not alter the upstream loader and intentionally provides no blur, noise, crop,
cutout, or camera-dropout controls.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from lehome_train.data.mapping import CAMERA_KEYS
from lehome_train.io import canonical_json_sha256
from lehome_train.io import atomic_write_json


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SECRET_KEY = re.compile(r"(?:secret|token|env)", re.IGNORECASE)
_JITTER_ORDER = ("brightness", "contrast", "saturation", "hue")
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "canonical_holdout_id",
        "dataset_revision",
        "policy_revision",
        "evaluation_manifest_sha256",
        "mild_profile_sha256",
        "metric_name",
        "metric_direction",
        "baseline_metric",
        "candidate_metric",
        "max_allowed_regression",
        "non_regression_passed",
        "receipt_sha256",
    }
)
_SAMPLE_SHEET_FRAME_COUNT = 32
_SAMPLE_SHEET_CAMERA_KEYS = tuple(
    key.removeprefix("observation.images.") for key in CAMERA_KEYS
)


@dataclass(frozen=True, slots=True)
class AugmentationProfile:
    """One supported, immutable color-jitter selection."""

    name: str
    color_jitter: Mapping[str, float]

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(
            {
                "schema_version": 1,
                "name": self.name,
                "color_jitter": dict(self.color_jitter),
            }
        )


def _profile(name: str, **color_jitter: float) -> AugmentationProfile:
    if tuple(color_jitter) != _JITTER_ORDER[: len(color_jitter)]:
        raise RuntimeError("augmentation profile jitter order is not canonical")
    return AugmentationProfile(name, MappingProxyType(color_jitter))


_PROFILES = MappingProxyType(
    {
        "none": _profile("none"),
        "mild": _profile(
            "mild", brightness=0.20, contrast=0.20, saturation=0.20, hue=0.05
        ),
        "nvidia_reference": _profile(
            "nvidia_reference",
            brightness=0.30,
            contrast=0.40,
            saturation=0.50,
            hue=0.08,
        ),
    }
)


def _reject_secret_keys(value: object, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"{label} keys must be strings")
            if _SECRET_KEY.search(key):
                raise ValueError(f"{label} must not contain secret, token, or environment keys")
            _reject_secret_keys(item, label=label)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _reject_secret_keys(item, label=label)


def _require_string(value: object, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"augmentation receipt {field} must be a non-empty string")
    return value


def _require_sha256(value: object, field: str) -> str:
    parsed = _require_string(value, field)
    if not _SHA256.fullmatch(parsed):
        raise ValueError(f"augmentation receipt {field} must be a SHA-256")
    return parsed


def _require_revision(value: object, field: str) -> str:
    parsed = _require_string(value, field)
    if not _REVISION.fullmatch(parsed):
        raise ValueError(f"augmentation receipt {field} must be a pinned revision")
    return parsed


def _require_metric(value: object, field: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"augmentation receipt {field} must be finite")
    return float(value)


def _validated_nvidia_reference_receipt(receipt: Mapping[str, object] | None) -> dict[str, object]:
    """Validate explicit mild-versus-canonical-holdout comparison evidence.

    A receipt is evidence supplied by an external evaluated run, never a
    locally minted convenience value.  Its hash covers every other field so a
    persisted gate cannot be silently edited after the comparison.
    """

    if not isinstance(receipt, Mapping):
        raise ValueError("nvidia_reference requires a canonical-holdout receipt")
    _reject_secret_keys(receipt, label="augmentation receipt")
    if set(receipt) != _RECEIPT_FIELDS:
        raise ValueError("augmentation receipt has missing or unknown fields")
    if receipt["schema_version"] != 1 or type(receipt["schema_version"]) is not int:
        raise ValueError("augmentation receipt schema_version must be 1")
    _require_string(receipt["canonical_holdout_id"], "canonical_holdout_id")
    _require_revision(receipt["dataset_revision"], "dataset_revision")
    _require_revision(receipt["policy_revision"], "policy_revision")
    _require_sha256(receipt["evaluation_manifest_sha256"], "evaluation_manifest_sha256")
    if receipt["mild_profile_sha256"] != _PROFILES["mild"].sha256:
        raise ValueError("augmentation receipt must identify the immutable mild profile")
    _require_string(receipt["metric_name"], "metric_name")
    direction = receipt["metric_direction"]
    if direction not in {"higher_is_better", "lower_is_better"}:
        raise ValueError("augmentation receipt metric_direction is invalid")
    baseline = _require_metric(receipt["baseline_metric"], "baseline_metric")
    candidate = _require_metric(receipt["candidate_metric"], "candidate_metric")
    allowed_regression = _require_metric(
        receipt["max_allowed_regression"], "max_allowed_regression"
    )
    if allowed_regression < 0:
        raise ValueError("augmentation receipt max_allowed_regression must be non-negative")
    if type(receipt["non_regression_passed"]) is not bool:
        raise ValueError("augmentation receipt non_regression_passed must be boolean")
    observed_pass = (
        candidate >= baseline - allowed_regression
        if direction == "higher_is_better"
        else candidate <= baseline + allowed_regression
    )
    if receipt["non_regression_passed"] is not True or not observed_pass:
        raise ValueError("augmentation receipt must prove mild passed with no regression")
    supplied_hash = _require_sha256(receipt["receipt_sha256"], "receipt_sha256")
    evidence = {key: receipt[key] for key in sorted(_RECEIPT_FIELDS - {"receipt_sha256"})}
    if supplied_hash != canonical_json_sha256(evidence):
        raise ValueError("augmentation receipt hash does not match comparison evidence")
    return {key: receipt[key] for key in sorted(_RECEIPT_FIELDS)}


def augmentation_profile(
    name: str,
    *,
    receipt: Mapping[str, object] | None = None,
) -> AugmentationProfile:
    """Return an allowed profile after enforcing its required gate."""

    if type(name) is not str or name not in _PROFILES:
        raise ValueError("unknown augmentation profile")
    if name == "nvidia_reference":
        _validated_nvidia_reference_receipt(receipt)
    elif receipt is not None:
        raise ValueError("augmentation receipt is only valid for nvidia_reference")
    return _PROFILES[name]


def validated_augmentation_receipt(
    profile_name: str, receipt: Mapping[str, object] | None
) -> dict[str, object] | None:
    """Return canonical persistence-safe receipt evidence for one profile."""

    profile = augmentation_profile(profile_name, receipt=receipt)
    if profile.name == "nvidia_reference":
        return _validated_nvidia_reference_receipt(receipt)
    return None


def color_jitter_cli(profile: AugmentationProfile) -> tuple[str, ...]:
    """Encode only the official GR00T color-jitter flag and its eight values."""

    if not profile.color_jitter:
        return ()
    return (
        "--color-jitter-params",
        "brightness",
        str(profile.color_jitter["brightness"]),
        "contrast",
        str(profile.color_jitter["contrast"]),
        "saturation",
        str(profile.color_jitter["saturation"]),
        "hue",
        str(profile.color_jitter["hue"]),
    )


def build_sample_sheet_report(
    profile_name: str,
    *,
    seed: int,
    frames: Sequence[Mapping[str, object]],
    receipt: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Freeze 32 canonical frame references for later in-image rendering.

    This checkout has canonical video metadata but no checked image-decoding or
    augmentation-rendering seam.  The report is deliberately an offline,
    deterministic selection artifact; the accepted trainer image must still
    render these exact three-camera frames before a paid run.
    """

    profile = augmentation_profile(profile_name, receipt=receipt)
    if type(seed) is not int or seed < 0:
        raise ValueError("sample-sheet seed must be a non-negative integer")
    if isinstance(frames, (str, bytes)) or len(frames) != _SAMPLE_SHEET_FRAME_COUNT:
        raise ValueError("sample sheet must select exactly 32 frames")
    selected: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise ValueError(f"sample-sheet frame {index} must be an object")
        _reject_secret_keys(frame, label="sample-sheet frame")
        if set(frame) != {"episode_id", "frame_index"}:
            raise ValueError(f"sample-sheet frame {index} has missing or unknown fields")
        episode_id = _require_string(frame["episode_id"], "episode_id")
        frame_index = frame["frame_index"]
        if type(frame_index) is not int or frame_index < 0:
            raise ValueError(f"sample-sheet frame {index} has an invalid frame_index")
        identity = (episode_id, frame_index)
        if identity in seen:
            raise ValueError("sample-sheet frames must be unique")
        seen.add(identity)
        selected.append({"episode_id": episode_id, "frame_index": frame_index})
    return {
        "schema_version": 1,
        "profile": profile.name,
        "profile_sha256": profile.sha256,
        "augmentation_receipt": validated_augmentation_receipt(profile.name, receipt),
        "seed": seed,
        "camera_keys": list(_SAMPLE_SHEET_CAMERA_KEYS),
        "frame_count": _SAMPLE_SHEET_FRAME_COUNT,
        "selected_frames": selected,
        "render_status": "pending_accepted_trainer_image",
    }


def write_sample_sheet_report(
    output_dir: str | Path,
    profile_name: str,
    *,
    seed: int,
    frames: Sequence[Mapping[str, object]],
    receipt: Mapping[str, object] | None = None,
) -> Path:
    """Write only the selection report; the image file remains a Vast gate."""

    report = build_sample_sheet_report(
        profile_name, seed=seed, frames=frames, receipt=receipt
    )
    destination = Path(output_dir) / "augmentation-report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination, report)
    return destination
