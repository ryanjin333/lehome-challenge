"""Unit contracts for the fail-closed production Nebius capacity adapter."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


_INSTANCE_ID = "computeinstance-u00rv6yj0m1m7jen5q"


class _Completed:
    def __init__(self, *, stdout: str, returncode: int = 0, stderr: str = "") -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


def test_production_runner_uses_only_exact_nebius_instance_commands() -> None:
    from lehome_train.groot.experiment_capacity import NebiusCliInstanceRunner

    calls: list[tuple[tuple[str, ...], int, bool, bool]] = []

    def run(command, *, timeout, text, capture_output, check):
        calls.append((tuple(command), timeout, text, capture_output))
        assert check is False
        return _Completed(stdout=json.dumps({"metadata": {"id": _INSTANCE_ID}, "status": {"state": "STOPPED"}}))

    runner = NebiusCliInstanceRunner(subprocess_run=run, timeout_seconds=17)

    assert runner.instance_state(_INSTANCE_ID) == "STOPPED"
    runner.start_instance(_INSTANCE_ID)
    runner.stop_instance(_INSTANCE_ID)

    assert calls == [
        (("nebius", "compute", "instance", "get", "--id", _INSTANCE_ID, "--format", "json", "--no-progress", "--timeout", "17s", "--no-browser", "--no-check-update", "--retries", "1"), 22, True, True),
        (("nebius", "compute", "instance", "start", "--id", _INSTANCE_ID, "--format", "json", "--no-progress", "--timeout", "17s", "--no-browser", "--no-check-update", "--retries", "1"), 22, True, True),
        (("nebius", "compute", "instance", "stop", "--id", _INSTANCE_ID, "--format", "json", "--no-progress", "--timeout", "17s", "--no-browser", "--no-check-update", "--retries", "1"), 22, True, True),
    ]


def test_production_runner_binds_each_exact_instance_command_to_a_root_owned_cli_config(tmp_path: Path) -> None:
    """Capacity mutations must not inherit an ambient Nebius profile."""
    from lehome_train.groot.experiment_capacity import NebiusCliInstanceRunner

    config = tmp_path / "nebius-capacity.yaml"
    config.write_text("profiles: {}\n", encoding="utf-8")
    config.chmod(0o600)
    calls: list[tuple[str, ...]] = []

    def run(command, **_kwargs):
        calls.append(tuple(command))
        return _Completed(stdout=json.dumps({"metadata": {"id": _INSTANCE_ID}, "status": {"state": "STOPPED"}}))

    runner = NebiusCliInstanceRunner(
        subprocess_run=run,
        provider_config_file=config,
        provider_config_owner_uid=os.getuid(),
    )

    runner.start_instance(_INSTANCE_ID)

    assert calls == [
        (
            "nebius", "--config", str(config), "compute", "instance", "start", "--id", _INSTANCE_ID,
            "--format", "json", "--no-progress", "--timeout", "30s", "--no-browser",
            "--no-check-update", "--retries", "1",
        )
    ]


def test_production_runner_refuses_an_unsafe_provider_auth_config_before_mutation(tmp_path: Path) -> None:
    from lehome_train.groot.experiment_capacity import NebiusCliInstanceRunner

    config = tmp_path / "nebius-capacity.yaml"
    config.write_text("profiles: {}\n", encoding="utf-8")
    config.chmod(0o644)
    calls: list[tuple[str, ...]] = []
    runner = NebiusCliInstanceRunner(
        subprocess_run=lambda command, **_kwargs: calls.append(tuple(command)),
        provider_config_file=config,
        provider_config_owner_uid=os.getuid(),
    )

    with pytest.raises(RuntimeError, match="provider authentication"):
        runner.start_instance(_INSTANCE_ID)

    assert calls == []


def test_production_runner_rejects_mismatched_json_identity_before_returning_state() -> None:
    from lehome_train.groot.experiment_capacity import NebiusCliInstanceRunner

    runner = NebiusCliInstanceRunner(
        subprocess_run=lambda *args, **kwargs: _Completed(
            stdout=json.dumps({"metadata": {"id": "computeinstance-u00different"}, "status": {"state": "RUNNING"}})
        )
    )

    with pytest.raises(RuntimeError, match="identity"):
        runner.instance_state(_INSTANCE_ID)


@pytest.mark.parametrize(
    "payload",
    [
        "not-json",
        json.dumps({"metadata": {"id": _INSTANCE_ID}, "status": {"state": "SCHEDULING"}}),
        json.dumps({"metadata": {"id": _INSTANCE_ID}}),
    ],
)
def test_production_runner_fails_closed_on_bad_state_json(payload: str) -> None:
    from lehome_train.groot.experiment_capacity import NebiusCliInstanceRunner

    runner = NebiusCliInstanceRunner(subprocess_run=lambda *args, **kwargs: _Completed(stdout=payload))

    with pytest.raises(RuntimeError, match="state"):
        runner.instance_state(_INSTANCE_ID)


def test_production_runner_rejects_failed_cli_operation() -> None:
    from lehome_train.groot.experiment_capacity import NebiusCliInstanceRunner

    runner = NebiusCliInstanceRunner(
        subprocess_run=lambda *args, **kwargs: _Completed(stdout="", stderr="request failed", returncode=1)
    )

    with pytest.raises(RuntimeError, match="failed"):
        runner.start_instance(_INSTANCE_ID)
