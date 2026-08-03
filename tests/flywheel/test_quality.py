from __future__ import annotations

import json
from pathlib import Path

import pytest

from lehome.flywheel.quality import AttemptStats, grade_attempt, load_quality_thresholds


def manifest(path: Path) -> Path:
    document = {
        "schema_version": 1,
        "organizer_dataset": {"revision": "a" * 40, "sha256": "b" * 64},
        "derivation": {
            "source_statistics_sha256": "c" * 64,
            "sample_count": 100,
            "quantiles": {"clean": 0.95, "maximum": 0.99},
        },
        "thresholds": {
            "clean_velocity_p95": 1.0,
            "clean_acceleration_p95": 1.0,
            "clean_jitter_p95": 1.0,
            "max_velocity_p95": 2.0,
            "max_acceleration_p95": 2.0,
            "max_jitter_p95": 2.0,
            "allowed_stale_samples": 0,
            "allowed_unsafe_commands": 0,
        },
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_quality_grades_clean_recovery_and_rejected_attempts(tmp_path: Path) -> None:
    values = load_quality_thresholds(
        manifest(tmp_path / "quality-thresholds.json"),
        expected_dataset_revision="a" * 40,
        expected_dataset_sha256="b" * 64,
    )
    assert grade_attempt(AttemptStats(official_success=True), values).grade == "A"
    assert grade_attempt(AttemptStats(official_success=True, hesitations=2), values).grade == "B"
    assert grade_attempt(AttemptStats(official_success=True, stale_samples=1), values).grade == "C"
    assert grade_attempt(AttemptStats(official_success=False), values).trainable is False


def test_quality_manifest_requires_pinned_dataset_and_statistical_derivation(tmp_path: Path) -> None:
    path = manifest(tmp_path / "quality-thresholds.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["organizer_dataset"]["revision"] = "main"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="pinned"):
        load_quality_thresholds(path, expected_dataset_revision="a" * 40, expected_dataset_sha256="b" * 64)
