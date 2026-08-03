"""Deterministic, immutable 70/30 organizer/flywheel mix manifests."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random

from lehome_train.io import canonical_json_sha256


@dataclass(frozen=True, slots=True)
class MixPlan:
    seed: int
    organizer_training_frames: int
    flywheel_training_frames: int
    source_weights: dict[str, float]
    grade_weights: dict[str, float]
    organizer_episode_ids: tuple[str, ...]
    flywheel_episode_ids: tuple[str, ...]
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": 1, "seed": self.seed, "organizer_training_frames": self.organizer_training_frames,
                "flywheel_training_frames": self.flywheel_training_frames, "source_weights": self.source_weights,
                "grade_weights": self.grade_weights, "organizer_episode_ids": list(self.organizer_episode_ids),
                "flywheel_episode_ids": list(self.flywheel_episode_ids), "sha256": self.sha256}


def _read(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ValueError("mix input metadata is unavailable or malformed") from None
    if not isinstance(manifest, dict) or not isinstance(provenance, dict):
        raise ValueError("mix input metadata must be objects")
    return manifest, provenance


def _input(root: Path, *, expected_source: str) -> tuple[int, tuple[str, ...], dict[str, object]]:
    manifest, provenance = _read(root)
    if manifest.get("flywheel_source") != expected_source:
        raise ValueError(f"mix input must be {expected_source} expert data")
    frames = manifest.get("frame_count")
    ids = manifest.get("episode_ids")
    if type(frames) is not int or frames <= 0 or not isinstance(ids, list) or not ids or not all(isinstance(value, str) and value for value in ids):
        raise ValueError("mix input has invalid immutable frame metadata")
    if provenance.get("action_source") != "expert":
        raise ValueError("non-expert target cannot enter mix")
    if provenance.get("release_stage") == "public_unseen":
        raise ValueError("evaluation holdout cannot enter mix")
    return frames, tuple(sorted(ids)), provenance


def build_mix_plan(organizer: str | Path, flywheel: str | Path, *, seed: int, organizer_fraction: float = 0.70) -> MixPlan:
    """Freeze repeatable frame counts before snapshot materialization/statistics."""
    if type(seed) is not int or not 0 < organizer_fraction < 1:
        raise ValueError("mix seed and organizer fraction are invalid")
    organizer_frames, organizer_ids, _ = _input(Path(organizer), expected_source="organizer")
    flywheel_frames, flywheel_ids, provenance = _input(Path(flywheel), expected_source="flywheel")
    grade = provenance.get("quality_grade")
    if grade not in {"A", "B"}:
        raise ValueError("flywheel grade must be A or B")
    target_total = max(organizer_frames, math.ceil(flywheel_frames / (1.0 - organizer_fraction)))
    selected_organizer = round(target_total * organizer_fraction)
    selected_flywheel = target_total - selected_organizer
    # Shuffle IDs in the manifest rather than source traversal order, so any
    # later materializer has a deterministic sampling order and audit trail.
    randomizer = random.Random(seed)
    organizer_selected, flywheel_selected = list(organizer_ids), list(flywheel_ids)
    randomizer.shuffle(organizer_selected)
    randomizer.shuffle(flywheel_selected)
    base = {"schema_version": 1, "seed": seed, "organizer_training_frames": selected_organizer,
            "flywheel_training_frames": selected_flywheel, "source_weights": {"organizer": 0.7, "flywheel": 0.3},
            "grade_weights": {"A": 1.0, "B": 0.5}, "organizer_episode_ids": organizer_selected,
            "flywheel_episode_ids": flywheel_selected}
    return MixPlan(seed, selected_organizer, selected_flywheel, {"organizer": 0.7, "flywheel": 0.3}, {"A": 1.0, "B": 0.5}, tuple(organizer_selected), tuple(flywheel_selected), canonical_json_sha256(base))
