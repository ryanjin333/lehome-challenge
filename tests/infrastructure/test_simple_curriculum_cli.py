from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

from lehome.flywheel.simple_curriculum import build_calibration_rows


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_simple_curriculum_matrix.py"


def _catalog() -> dict[str, list[str]]:
    return {
        "top_long": [f"Top_Long_Seen_{index}" for index in range(10)],
        "top_short": [f"Top_Short_Seen_{index}" for index in range(10)],
        "pant_long": [f"Pant_Long_Seen_{index}" for index in range(10)],
        "pant_short": [f"Pant_Short_Seen_{index}" for index in range(10)],
    }


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "source" / "lehome")},
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_builds_calibration_and_hash_bound_receipt_atomically(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    output = tmp_path / "calibration.json"
    receipt = tmp_path / "calibration.receipt.json"
    catalog.write_text(json.dumps(_catalog()), encoding="utf-8")

    result = _run(
        "build-calibration", "--catalog", str(catalog), "--seed-base", "90000",
        "--output", str(output), "--receipt", str(receipt),
    )

    assert result.returncode == 0, result.stderr
    payload = output.read_bytes()
    emitted = json.loads(payload)
    bound_receipt = json.loads(receipt.read_text(encoding="utf-8"))
    assert len(emitted) == 400
    assert bound_receipt["output_sha256"] == hashlib.sha256(payload).hexdigest()
    assert bound_receipt["parameters"]["seed_base"] == 90000
    assert not list(tmp_path.glob(".*.tmp"))


def test_cli_emits_each_atomic_artifact_in_its_own_directory(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    output_parent = tmp_path / "matrices"
    receipt_parent = tmp_path / "receipts"
    output_parent.mkdir()
    receipt_parent.mkdir()
    catalog.write_text(json.dumps(_catalog()), encoding="utf-8")

    result = _run(
        "build-calibration", "--catalog", str(catalog), "--seed-base", "90000",
        "--output", str(output_parent / "calibration.json"),
        "--receipt", str(receipt_parent / "calibration.receipt.json"),
    )

    assert result.returncode == 0, result.stderr


def test_cli_refuses_symlinks_and_existing_destinations(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(_catalog()), encoding="utf-8")
    unsafe_catalog = tmp_path / "catalog-link.json"
    unsafe_catalog.symlink_to(catalog)
    output = tmp_path / "matrix.json"
    receipt = tmp_path / "receipt.json"

    symlink = _run(
        "build-calibration", "--catalog", str(unsafe_catalog), "--seed-base", "1",
        "--output", str(output), "--receipt", str(receipt),
    )
    assert symlink.returncode != 0
    assert "unsafe" in symlink.stderr

    output.write_text("already here", encoding="utf-8")
    existing = _run(
        "build-calibration", "--catalog", str(catalog), "--seed-base", "1",
        "--output", str(output), "--receipt", str(receipt),
    )
    assert existing.returncode != 0
    assert "absent" in existing.stderr


def test_cli_refuses_a_symlinked_input_ancestor(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "catalog.json").write_text(json.dumps(_catalog()), encoding="utf-8")
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    result = _run(
        "build-calibration", "--catalog", str(linked / "catalog.json"), "--seed-base", "1",
        "--output", str(tmp_path / "matrix.json"), "--receipt", str(tmp_path / "receipt.json"),
    )

    assert result.returncode != 0
    assert "unsafe" in result.stderr


def test_cli_rejects_nonfinite_json_before_curriculum_validation(tmp_path: Path) -> None:
    catalog = _catalog()
    calibration = build_calibration_rows(catalog, seed_base=90_000)
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(calibration, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    encoded = calibration_path.read_bytes()
    report = {
        "schema_version": 1,
        "kind": "lehome_simple_curriculum_calibration_report_v1",
        "authenticated": True,
        "calibration_matrix_sha256": hashlib.sha256(encoded).hexdigest(),
        "policy_identity": {
            "policy_repo": "ryanjin333/lehome-groot-n17-models", "policy_revision": "a" * 40,
            "policy_step": 12000, "policy_artifact_sha256": "b" * 64,
        },
        "authenticated_policy_identity": {
            "policy_repo": "ryanjin333/lehome-groot-n17-models", "policy_revision": "a" * 40,
            "policy_step": 12000, "policy_artifact_sha256": "b" * 64,
        },
        "provenance": {"simulator_device": "cpu", "policy_device": "cuda:0"},
        "catalog": catalog,
        "outcomes": [
            {"attempt_id": row["attempt_id"], "trial_id": row["trial_id"], "success": True}
            for row in calibration
        ],
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report, sort_keys=True, separators=(",", ":"))[:-1] + ",\"ignored_nonfinite\":1e999}\n", encoding="utf-8")
    approved_catalog = tmp_path / "approved-catalog.json"
    policy_identity = tmp_path / "policy-identity.json"
    approved_catalog.write_text(json.dumps(catalog), encoding="utf-8")
    policy_identity.write_text(json.dumps(_policy_identity()), encoding="utf-8")

    result = _run(
        "build-curriculum", "--report", str(report_path), "--calibration-matrix", str(calibration_path),
        "--approved-catalog", str(approved_catalog), "--policy-identity", str(policy_identity),
        "--rng-seed", "4", "--output", str(tmp_path / "curriculum.json"),
        "--receipt", str(tmp_path / "curriculum.receipt.json"),
    )

    assert result.returncode != 0
    assert "malformed" in result.stderr


def _policy_identity() -> dict[str, object]:
    return {
        "policy_repo": "ryanjin333/lehome-groot-n17-models",
        "policy_revision": "a" * 40,
        "policy_step": 12000,
        "policy_artifact_sha256": "b" * 64,
    }


def _report(calibration: list[dict[str, object]], catalog: dict[str, list[str]]) -> dict[str, object]:
    payload = (json.dumps(calibration, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return {
        "schema_version": 1,
        "kind": "lehome_simple_curriculum_calibration_report_v1",
        "authenticated": True,
        "calibration_matrix_sha256": hashlib.sha256(payload).hexdigest(),
        "policy_identity": _policy_identity(),
        "authenticated_policy_identity": _policy_identity(),
        "provenance": {"simulator_device": "cpu", "policy_device": "cuda:0"},
        "catalog": catalog,
        "outcomes": [
            {"attempt_id": row["attempt_id"], "trial_id": row["trial_id"], "success": index % 2 == 0}
            for index, row in enumerate(calibration)
        ],
    }


def test_cli_builds_authenticated_curriculum_and_receipt(tmp_path: Path) -> None:
    catalog = _catalog()
    calibration = build_calibration_rows(catalog, seed_base=90_000)
    calibration_path = tmp_path / "calibration.json"
    report_path = tmp_path / "report.json"
    approved_catalog = tmp_path / "approved-catalog.json"
    policy_identity = tmp_path / "policy-identity.json"
    output = tmp_path / "curriculum.json"
    receipt = tmp_path / "curriculum.receipt.json"
    calibration_payload = (json.dumps(calibration, sort_keys=True, separators=(",", ":")) + "\n").encode()
    calibration_path.write_bytes(calibration_payload)
    report_path.write_text(json.dumps(_report(calibration, catalog)), encoding="utf-8")
    approved_catalog.write_text(json.dumps(catalog), encoding="utf-8")
    policy_identity.write_text(json.dumps(_policy_identity()), encoding="utf-8")

    result = _run(
        "build-curriculum", "--report", str(report_path), "--calibration-matrix", str(calibration_path),
        "--approved-catalog", str(approved_catalog), "--policy-identity", str(policy_identity),
        "--rng-seed", "4", "--output", str(output), "--receipt", str(receipt),
    )

    assert result.returncode == 0, result.stderr
    output_payload = output.read_bytes()
    bound_receipt = json.loads(receipt.read_text(encoding="utf-8"))
    assert len(json.loads(output_payload)) == 600
    assert bound_receipt["output_sha256"] == hashlib.sha256(output_payload).hexdigest()
    assert bound_receipt["parameters"] == {
        "calibration_matrix_sha256": hashlib.sha256(calibration_payload).hexdigest(),
        "approved_catalog": catalog,
        "command": "build-curriculum",
        "count": 600,
        "policy_identity": _policy_identity(),
        "report_sha256": hashlib.sha256((json.dumps(_report(calibration, catalog), sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest(),
        "rng_seed": 4,
    }


def test_cli_rejects_output_receipt_alias_without_creating_an_artifact(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(_catalog()), encoding="utf-8")
    shared = tmp_path / "shared.json"

    result = _run(
        "build-calibration", "--catalog", str(catalog), "--seed-base", "1",
        "--output", str(shared), "--receipt", str(shared),
    )

    assert result.returncode != 0
    assert "distinct" in result.stderr
    assert not shared.exists()


def test_cli_rejects_duplicate_json_keys_without_creating_outputs(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    output = tmp_path / "matrix.json"
    receipt = tmp_path / "receipt.json"
    catalog.write_text(
        '{"top_long":[],"top_long":[],"top_short":[],"pant_long":[],"pant_short":[]}',
        encoding="utf-8",
    )

    result = _run(
        "build-calibration", "--catalog", str(catalog), "--seed-base", "1",
        "--output", str(output), "--receipt", str(receipt),
    )

    assert result.returncode != 0
    assert "catalog is malformed" in result.stderr
    assert not output.exists()
    assert not receipt.exists()
