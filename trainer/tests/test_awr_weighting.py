from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")


def _document() -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "lehome_awr_progress_evidence",
        "mixture_id": "a" * 64,
        "mixture_manifest_sha256": "b" * 64,
        "episodes": [
            {
                "episode_id": "attempt-low",
                "lineage_id": "lineage-low",
                "split": "train",
                "score_kind": "progress",
                "score": -1.0,
                "provenance_path": "receipts/low.json",
                "provenance_sha256": "c" * 64,
            },
            {
                "episode_id": "attempt-high",
                "lineage_id": "lineage-high",
                "split": "train",
                "score_kind": "advantage",
                "score": 2.0,
                "provenance_path": "receipts/high.json",
                "provenance_sha256": "d" * 64,
            },
        ],
    }


def test_progress_evidence_has_a_canonical_identity_and_bounded_positive_weights(
    tmp_path: Path,
) -> None:
    from lehome_train.groot.awr_weighting import (
        AwrReplayConfig,
        canonical_evidence_sha256,
        load_progress_evidence,
    )

    document = _document()
    path = tmp_path / "awr-progress.json"
    _write(path, document)
    evidence = load_progress_evidence(
        path,
        expected_sha256=canonical_evidence_sha256(document),
        mixture_id="a" * 64,
        mixture_manifest_sha256="b" * 64,
    )

    weights = evidence.weights(AwrReplayConfig(temperature=1.0, minimum=0.5, maximum=2.0))

    assert evidence.identity_sha256 == canonical_evidence_sha256(document)
    assert weights["attempt-low"] == pytest.approx(0.5)
    assert weights["attempt-high"] == pytest.approx(2.0)


def test_awr_replay_configuration_has_a_canonical_identity() -> None:
    from lehome_train.groot.awr_weighting import AwrReplayConfig
    from lehome_train.io import canonical_json_sha256

    config = AwrReplayConfig(temperature=0.75, minimum=0.5, maximum=3.0)

    assert config.to_dict() == {
        "temperature": 0.75,
        "minimum": 0.5,
        "maximum": 3.0,
    }
    assert config.sha256 == canonical_json_sha256(config.to_dict())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value["episodes"].append(dict(value["episodes"][0])), "duplicate"),
        (lambda value: value["episodes"].__setitem__(0, {**value["episodes"][0], "score": True}), "score"),
        (lambda value: value["episodes"].__setitem__(0, {**value["episodes"][0], "score": float("nan")}), "finite"),
        (lambda value: value["episodes"].__setitem__(0, {**value["episodes"][0], "provenance_path": "receipts/../low.json"}), "unsafe"),
        (lambda value: value.__setitem__("extra", 1), "unknown or missing"),
    ],
)
def test_progress_evidence_rejects_drift_and_ambiguous_identity(
    tmp_path: Path, mutate: object, message: str,
) -> None:
    from lehome_train.groot.awr_weighting import canonical_evidence_sha256, load_progress_evidence

    document = _document()
    assert callable(mutate)
    mutate(document)
    path = tmp_path / "awr-progress.json"
    _write(path, document)

    with pytest.raises(ValueError, match=message):
        load_progress_evidence(
            path,
            expected_sha256=canonical_evidence_sha256(document),
            mixture_id="a" * 64,
            mixture_manifest_sha256="b" * 64,
        )


@pytest.mark.parametrize(
    "config",
    [
        {"temperature": True, "minimum": 0.5, "maximum": 2.0},
        {"temperature": float("inf"), "minimum": 0.5, "maximum": 2.0},
        {"temperature": 0.0, "minimum": 0.5, "maximum": 2.0},
        {"temperature": 1.0, "minimum": 0.0, "maximum": 2.0},
        {"temperature": 1.0, "minimum": 2.0, "maximum": 0.5},
    ],
)
def test_awr_configuration_rejects_invalid_weight_bounds(config: dict[str, object]) -> None:
    from lehome_train.groot.awr_weighting import AwrReplayConfig

    with pytest.raises(ValueError):
        AwrReplayConfig(**config)  # type: ignore[arg-type]


def test_progress_evidence_rejects_a_digest_mismatch(tmp_path: Path) -> None:
    from lehome_train.groot.awr_weighting import load_progress_evidence

    path = tmp_path / "awr-progress.json"
    _write(path, _document())

    with pytest.raises(ValueError, match="digest mismatch"):
        load_progress_evidence(
            path,
            expected_sha256="e" * 64,
            mixture_id="a" * 64,
            mixture_manifest_sha256="b" * 64,
        )


def test_authenticated_progress_evidence_receipt_requires_exact_readback_identity() -> None:
    from lehome_train.groot.awr_weighting import authenticated_progress_evidence_receipt_sha256

    receipt = {
        "schema_version": 1,
        "kind": "lehome_awr_progress_evidence_receipt",
        "evidence_sha256": "a" * 64,
        "mixture_id": "b" * 64,
        "mixture_manifest_sha256": "c" * 64,
        "authenticated_principal_sha256": "d" * 64,
        "readback_receipt_sha256": "e" * 64,
        "readback_verified": True,
    }
    assert len(authenticated_progress_evidence_receipt_sha256(receipt)) == 64
    receipt["readback_verified"] = False
    with pytest.raises(ValueError, match="read-back"):
        authenticated_progress_evidence_receipt_sha256(receipt)
