from __future__ import annotations

from dataclasses import dataclass
import json
import subprocess

import pytest

from lehome.flywheel.runtime_preflight import (
    IsaacRuntimePreflightError,
    IsaacRuntimeReceipt,
    inspect_isaac_sim_5_1_runtime,
    require_isaac_sim_5_1_runtime,
)
import scripts.check_isaac_runtime as runtime_cli


@dataclass
class _Result:
    returncode: int = 0
    stdout: str = "580.65.06\n"


def _probe(stdout: str = "580.65.06\n", *, returncode: int = 0, **kwargs):
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def runner(command, **runner_kwargs):
        calls.append((tuple(command), runner_kwargs))
        return _Result(returncode=returncode, stdout=stdout)

    receipt = inspect_isaac_sim_5_1_runtime(
        system=lambda: kwargs.get("system", "Linux"),
        machine=lambda: kwargs.get("machine", "x86_64"),
        runner=runner,
    )
    return receipt, calls


def test_runtime_preflight_accepts_reviewed_r580_and_uses_the_exact_driver_query() -> None:
    receipt, calls = _probe()

    assert receipt.compatible is True
    assert receipt.driver_versions == ("580.65.06",)
    assert calls == [
        (
            ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"),
            {"check": False, "capture_output": True, "text": True, "timeout": 5.0},
        )
    ]


@pytest.mark.parametrize(
    ("stdout", "error_code"),
    (
        ("580.65.05\n", "unreviewed_driver_version"),
        ("595.71.05\n", "unreviewed_driver_version"),
        ("581.0.0\n", "unreviewed_driver_version"),
        ("580.65\n", "malformed_driver_version"),
        ("580.65.06\n580.66.01\n", "mixed_driver_versions"),
    ),
)
def test_runtime_preflight_rejects_unreviewed_malformed_or_mixed_versions(
    stdout: str,
    error_code: str,
) -> None:
    receipt, _ = _probe(stdout)

    assert receipt.compatible is False
    assert receipt.error_code == error_code


def test_runtime_preflight_rejects_missing_or_failed_nvidia_smi() -> None:
    def missing(*_args, **_kwargs):
        raise FileNotFoundError("nvidia-smi")

    missing_receipt = inspect_isaac_sim_5_1_runtime(
        system=lambda: "Linux", machine=lambda: "x86_64", runner=missing,
    )
    failed_receipt, _ = _probe(returncode=1)

    assert missing_receipt.error_code == "nvidia_smi_unavailable"
    assert failed_receipt.error_code == "nvidia_smi_failed"


def test_runtime_preflight_rejects_a_timed_out_nvidia_smi_probe() -> None:
    def timed_out(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("nvidia-smi", 5.0)

    receipt = inspect_isaac_sim_5_1_runtime(
        system=lambda: "Linux", machine=lambda: "x86_64", runner=timed_out,
    )

    assert receipt.error_code == "nvidia_smi_unavailable"


def test_runtime_preflight_accepts_the_later_reviewed_r580_version() -> None:
    receipt, _ = _probe("580.159.03\n")

    assert receipt.compatible is True


@pytest.mark.parametrize(
    ("system", "machine", "error_code"),
    (
        ("Darwin", "arm64", "unsupported_system"),
        ("Linux", "aarch64", "unsupported_machine"),
    ),
)
def test_runtime_preflight_rejects_non_linux_or_non_x86_without_querying_a_gpu(
    system: str,
    machine: str,
    error_code: str,
) -> None:
    receipt, calls = _probe(system=system, machine=machine)

    assert receipt.error_code == error_code
    assert calls == []


def test_runtime_preflight_fails_closed_with_actionable_r590_error() -> None:
    with pytest.raises(IsaacRuntimePreflightError, match="Switch to a reviewed R580-driver host"):
        require_isaac_sim_5_1_runtime(
            system=lambda: "Linux",
            machine=lambda: "x86_64",
            runner=lambda *_args, **_kwargs: _Result(stdout="595.71.05\n"),
        )


def test_malformed_probe_output_is_not_copied_into_the_receipt() -> None:
    receipt, _ = _probe("580.65.06 secret-token-value\n")

    encoded = json.dumps(receipt.as_dict(), sort_keys=True)
    assert receipt.error_code == "malformed_driver_version"
    assert "secret-token-value" not in encoded
    assert receipt.driver_versions == ()


def test_cli_prints_deterministic_machine_readable_secret_free_receipt(monkeypatch, capsys) -> None:
    receipt = IsaacRuntimeReceipt(
        compatible=False,
        error_code="unreviewed_driver_version",
        driver_versions=("595.71.05",),
        machine="x86_64",
        system="Linux",
    )
    monkeypatch.setattr(runtime_cli, "inspect_isaac_sim_5_1_runtime", lambda: receipt)

    assert runtime_cli.main() == 2
    assert capsys.readouterr().out == (
        '{"compatible":false,"driver_versions":["595.71.05"],'
        '"error_code":"unreviewed_driver_version","machine":"x86_64",'
        '"policy":"isaac_sim_5.1_linux_x86_64_r580","schema_version":1,'
        '"system":"Linux"}\n'
    )
