"""Offline CLI tests for the public GR00T N1.5 reproduction controller."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from tests.test_n15_reproduction import (
    _fixture_contract,
    _materialize_snapshots,
    _materialize_source,
    _materialize_training_output,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_public_n15_reproduction.py"


def _common(checkout: Path, source_receipt: Path, snapshots_receipt: Path, contract) -> list[str]:
    return [
        "--checkout",
        str(checkout),
        "--source-receipt",
        str(source_receipt),
        "--resolved-snapshots-receipt",
        str(snapshots_receipt),
        "--vm-id",
        contract.vm_id,
        "--disk-id",
        contract.disk_id,
    ]


def test_cli_verify_inputs_render_training_and_verify_output_are_offline_and_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.run_public_n15_reproduction import main

    checkout, source_receipt = _materialize_source(tmp_path)
    _, _, snapshots_receipt = _materialize_snapshots(tmp_path, checkout)
    contract = _fixture_contract(checkout)
    common = _common(checkout, source_receipt, snapshots_receipt, contract)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("offline CLI must not execute subprocesses"),
    )

    verified_output = tmp_path / "verified-inputs.json"
    assert main(["verify-inputs", *common, "--output", str(verified_output)], contract=contract) == 0
    verified = json.loads(verified_output.read_text(encoding="utf-8"))
    assert verified["kind"] == "lehome_public_n15_verified_inputs_v1"
    assert verified["vm_id"] == contract.vm_id
    assert verified["disk_id"] == contract.disk_id
    assert verified_output.stat().st_mode & 0o777 == 0o444

    manifest_output = tmp_path / "training-execution.json"
    assert main(["render-training", *common, "--output", str(manifest_output)], contract=contract) == 0
    manifest = json.loads(manifest_output.read_text(encoding="utf-8"))
    assert manifest["execution"]["argv"] == [
        "lerobot-train",
        "--config_path=configs/train_groot.yaml",
    ]
    assert manifest["execution"]["env"] == {
        "HF_HUB_CACHE": str((tmp_path / "hub").resolve()),
        "HF_HUB_OFFLINE": "1",
    }
    assert manifest["execution"]["shell_argv"] == (
        "lerobot-train --config_path=configs/train_groot.yaml"
    )

    training_root = _materialize_training_output(
        tmp_path,
        source_receipt=source_receipt,
        snapshots_receipt=snapshots_receipt,
    )
    training_output = tmp_path / "verified-training-output.json"
    assert main(
        [
            "verify-training-output",
            *common,
            "--training-root",
            str(training_root),
            "--output",
            str(training_output),
        ],
        contract=contract,
    ) == 0
    result = json.loads(training_output.read_text(encoding="utf-8"))
    assert result["kind"] == "lehome_public_n15_verified_training_output_v1"
    assert result["step"] == 12000
    assert training_output.stat().st_mode & 0o777 == 0o444
    assert "lerobot-train --config_path=configs/train_groot.yaml" in capsys.readouterr().out


def test_cli_fails_closed_without_overwriting_an_existing_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.run_public_n15_reproduction import main

    checkout, source_receipt = _materialize_source(tmp_path)
    _, _, snapshots_receipt = _materialize_snapshots(tmp_path, checkout)
    contract = _fixture_contract(checkout)
    output = tmp_path / "existing.json"
    output.write_text("sentinel\n", encoding="utf-8")

    result = main(
        [
            "verify-inputs",
            *_common(checkout, source_receipt, snapshots_receipt, contract),
            "--output",
            str(output),
        ],
        contract=contract,
    )

    assert result == 2
    assert output.read_text(encoding="utf-8") == "sentinel\n"
    assert "already exists" in capsys.readouterr().err


def test_cli_script_help_is_importable_without_cloud_or_training_dependencies() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "source/lehome")},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "verify-inputs" in result.stdout
    assert "render-training" in result.stdout
    assert "verify-training-output" in result.stdout
