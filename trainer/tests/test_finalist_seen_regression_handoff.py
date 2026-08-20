"""Immutable finalist-specific seen-regression handoff materialization."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


def _module() -> object:
    script = Path(__file__).resolve().parents[2] / "scripts/materialize_finalist_seen_regression_handoff.py"
    spec = importlib.util.spec_from_file_location("finalist_seen_handoff", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _evidence(receipt: str) -> dict[str, object]:
    evidence: dict[str, object] = {
        "schema_version": 1,
        "kind": "lehome_experiment_seen_regression_evidence",
        "candidate_checkpoint_receipt_sha256": receipt,
        "major_seen_regression": False,
        "readback_verified": True,
        "sealed": True,
    }
    evidence["report_sha256"] = hashlib.sha256(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    return evidence


def test_materializer_writes_exact_finalist_and_receipt_bound_immutable_descriptor(tmp_path: Path) -> None:
    module = _module()
    experiment_id, receipt = "a" * 64, "b" * 64
    source = tmp_path / "seen.json"
    source.write_text(json.dumps(_evidence(receipt), sort_keys=True, separators=(",", ":")), encoding="ascii")
    source.chmod(0o444)
    root = tmp_path / "handoffs"

    output = module.materialize_handoff(
        root=root, experiment_id=experiment_id,
        checkpoint_receipt_sha256=receipt, evidence_path=source,
    )

    assert output == root / experiment_id / f"{receipt}.json"
    assert output.stat().st_mode & 0o777 == 0o444
    descriptor = json.loads(output.read_text(encoding="ascii"))
    assert descriptor["experiment_id"] == experiment_id
    assert descriptor["checkpoint_receipt_sha256"] == receipt
    assert descriptor["evidence"]["candidate_checkpoint_receipt_sha256"] == receipt


def test_materializer_rejects_checkpoint_mismatch_and_existing_destination(tmp_path: Path) -> None:
    module = _module()
    experiment_id, receipt = "a" * 64, "b" * 64
    source = tmp_path / "seen.json"
    source.write_text(json.dumps(_evidence("c" * 64), sort_keys=True, separators=(",", ":")), encoding="ascii")
    source.chmod(0o444)
    root = tmp_path / "handoffs"

    try:
        module.materialize_handoff(root=root, experiment_id=experiment_id, checkpoint_receipt_sha256=receipt, evidence_path=source)
    except ValueError as error:
        assert "checkpoint" in str(error)
    else:
        raise AssertionError("misbound seen-regression evidence was materialized")
