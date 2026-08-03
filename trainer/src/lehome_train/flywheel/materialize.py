"""Materialize verified expert-only windows without mutating raw episodes."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from lehome_train.io import atomic_write_json, canonical_json_sha256


ACTION_HORIZON = 16
CAMERA_KEYS = ("top_rgb", "left_rgb", "right_rgb")
# The canonical evaluation matrix must set ``evaluation_only``.  This small
# explicit list also protects older collection manifests which predate it.
PUBLIC_UNSEEN_HOLDOUTS = frozenset({"public-unseen"})


@dataclass(frozen=True, slots=True)
class MaterializationReport:
    episode_id: str
    selected_observations: int
    rejected_by_reason: dict[str, int]
    output_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "episode_id": self.episode_id,
            "selected_observations": self.selected_observations,
            "rejected_by_reason": dict(sorted(self.rejected_by_reason.items())),
            "output_sha256": self.output_sha256,
        }


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("raw episode contains a duplicate JSON field")
        result[key] = value
    return result


def _read_raw(root: Path) -> dict[str, Any]:
    path = root / "episode.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object)
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("raw episode metadata is unavailable or malformed") from None
    if not isinstance(payload, dict):
        raise ValueError("raw episode metadata must be an object")
    checksums = root / "SHA256SUMS"
    if checksums.exists():
        for line in checksums.read_text(encoding="utf-8").splitlines():
            parts = line.split(maxsplit=1)
            if len(parts) != 2 or len(parts[0]) != 64:
                raise ValueError("raw episode checksum manifest is malformed")
            target = root / parts[1].lstrip("*")
            if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != parts[0]:
                raise ValueError("raw episode checksum verification failed")
    return payload


def _vector(value: object, *, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 12:
        raise ValueError(f"{label} must be a finite 12D vector")
    result: list[float] = []
    for item in value:
        if type(item) not in (int, float):
            raise ValueError(f"{label} must be a finite 12D vector")
        number = float(item)
        if number != number or number in (float("inf"), float("-inf")):
            raise ValueError(f"{label} must be a finite 12D vector")
        result.append(number)
    return result


def _frame(value: object, index: int) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"raw frame {index} is malformed")
    if value.get("action_source") not in {"policy", "expert", "hold"}:
        raise ValueError(f"raw frame {index} has an invalid action source")
    cameras = value.get("cameras")
    if not isinstance(cameras, Mapping) or tuple(cameras) != CAMERA_KEYS:
        raise ValueError(f"raw frame {index} lacks the canonical three cameras")
    if not all(isinstance(cameras[name], str) and cameras[name] for name in CAMERA_KEYS):
        raise ValueError(f"raw frame {index} has invalid camera evidence")
    _vector(value.get("state"), label=f"raw frame {index} state")
    _vector(value.get("action"), label=f"raw frame {index} action")
    return value


def materialize_episode(raw_root: str | Path, output_root: str | Path) -> MaterializationReport:
    """Select only complete, successful expert windows into a portable episode.

    Camera references stay as provenance in this offline format; the existing
    converter remains the canonical writer for actual LeRobot video shards.
    """

    raw = _read_raw(Path(raw_root))
    grade = raw.get("quality_grade")
    if grade not in {"A", "B", "C"}:
        raise ValueError("raw episode quality grade must be A, B, or C")
    if grade == "C":
        raise ValueError("Grade C episodes cannot enter training")
    if raw.get("official_success") is not True:
        raise ValueError("unsuccessful episodes cannot enter training")
    if raw.get("evaluation_only") is True or raw.get("garment_name") in PUBLIC_UNSEEN_HOLDOUTS:
        raise ValueError("evaluation holdout cannot enter training")
    if raw.get("fps") != 30:
        raise ValueError("raw episode must be synchronized at 30 FPS")
    episode_id = raw.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("raw episode ID is required")
    input_frames = raw.get("frames")
    if not isinstance(input_frames, list):
        raise ValueError("raw episode frames are required")
    frames = tuple(_frame(value, index) for index, value in enumerate(input_frames))
    rejected: Counter[str] = Counter()
    for frame in frames:
        source = str(frame["action_source"])
        if source != "expert":
            rejected[source] += 1
        if frame.get("eligible") is False:
            rejected[str(frame.get("rejection_reason", "ineligible"))] += 1
    selected: list[dict[str, object]] = []
    for start, frame in enumerate(frames):
        window = frames[start : start + ACTION_HORIZON]
        if len(window) != ACTION_HORIZON:
            if frame["action_source"] == "expert":
                rejected["incomplete_horizon"] += 1
            continue
        if frame["action_source"] != "expert" or frame.get("eligible") is False:
            continue
        if any(candidate["action_source"] != "expert" or candidate.get("eligible") is False for candidate in window):
            rejected["non_expert_future"] += 1
            continue
        selected.append({
            "observation_step": int(frame.get("step", start)),
            "state": _vector(frame["state"], label="state"),
            "action": _vector(frame["action"], label="action"),
            "future_actions": [_vector(candidate["action"], label="future action") for candidate in window],
            "cameras": {key: frame["cameras"][key] for key in CAMERA_KEYS},
            "action_source": "expert",
        })
    if not selected:
        raise ValueError("accepted episode contains no complete expert windows")
    output = Path(output_root)
    if output.exists():
        raise FileExistsError("refusing to overwrite materialized episode")
    output.mkdir(parents=True)
    provenance = raw.get("provenance", {})
    if not isinstance(provenance, Mapping):
        raise ValueError("raw episode provenance is malformed")
    # Retain only explicitly non-secret, immutable provenance fields.
    safe_provenance = {key: value for key, value in provenance.items() if "token" not in key.lower() and "credential" not in key.lower()}
    payload = {
        "schema_version": 1,
        "episode_id": episode_id,
        "quality_grade": grade,
        "fps": 30,
        "camera_keys": list(CAMERA_KEYS),
        "state_dimension": 12,
        "action_dimension": 12,
        "action_horizon": ACTION_HORIZON,
        "frames": selected,
    }
    atomic_write_json(output / "episode.json", payload)
    atomic_write_json(output / "provenance.json", dict(safe_provenance))
    digest = canonical_json_sha256(payload)
    report = MaterializationReport(episode_id, len(selected), dict(rejected), digest)
    atomic_write_json(output / "selection-report.json", report.to_dict())
    return report
