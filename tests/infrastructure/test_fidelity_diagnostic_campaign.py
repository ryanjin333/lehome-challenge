"""Fail-closed contract for the isolated 1+3 cloth-fidelity diagnostic."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "rollout_appliance" / "run_12k_campaign.sh"
WRAPPER = ROOT / "rollout_appliance" / "run_fidelity_diagnostic.sh"
ROLLOUT_IMAGE = "ghcr.io/ryanjin333/lehome-rollout@sha256:" + "d" * 64


def _row(stage: str, seed: int, index: int = 1) -> dict[str, object]:
    return {
        "campaign_kind": "fidelity_diagnostic_v1",
        "diagnostic_stage": stage,
        "attempt_id": f"fidelity-diagnostic-{stage.lower()}-{index}",
        "trial_id": f"fidelity-diagnostic-{stage.lower()}-{index}",
        "garment": "Top_Short_Seen_2",
        "garment_name": "Top_Short_Seen_2",
        "category": "top_short",
        "release_stage": "seen",
        "seed": seed,
        "source_seed": seed,
        "strategy": "canonical",
    }


def _environment(tmp_path: Path, stage: str = "A") -> dict[str, str]:
    rows = [_row("A", 2026082789)] if stage == "A" else [
        _row("B", seed, index)
        for index, seed in enumerate((2026082709, 2026082749, 2026082789), start=1)
    ]
    matrix = tmp_path / f"stage-{stage.lower()}.json"
    matrix.write_text(json.dumps(rows, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    count = len(rows)
    environment = os.environ.copy()
    environment.update({
        "LEHOME_WORKSPACE": str(tmp_path / "workspace"),
        "LEHOME_CAMPAIGN_ROOT": str(tmp_path / f"fidelity-diagnostic-stage-{stage.lower()}"),
        "LEHOME_ATTEMPT_MATRIX": str(matrix),
        "LEHOME_ATTEMPT_MATRIX_SHA256": hashlib.sha256(matrix.read_bytes()).hexdigest(),
        "LEHOME_FIDELITY_DIAGNOSTIC": "1",
        "LEHOME_FIDELITY_DIAGNOSTIC_STAGE": stage,
        "LEHOME_SIMULATOR_DEVICE": "cpu",
        "LEHOME_WORKER_COUNT": "1",
        "LEHOME_ENABLE_HF_UPLOAD": "0",
        "LEHOME_SKIP_ROUND_SEAL": "1",
        "LEHOME_MAX_WORKER_RESTARTS": "0",
        "LEHOME_POLICY_STEP": "12000",
        "LEHOME_COMPLETION_METRIC": "terminal_outcomes",
        "LEHOME_MAX_ATTEMPTS": str(count),
        "LEHOME_TARGET_ACCEPTED": str(count),
        "LEHOME_PARTITION_ID": f"fidelity-diagnostic-{stage.lower()}",
        "LEHOME_ROLLOUT_IMAGE": ROLLOUT_IMAGE,
        "LEHOME_HOST_CODE_ROOT": str(ROOT),
        "LEHOME_VALIDATE_MATRIX_ONLY": "1",
    })
    return environment


@pytest.mark.parametrize("stage", ["A", "B"])
def test_exact_fidelity_diagnostic_stage_is_allowlisted_before_docker(tmp_path: Path, stage: str) -> None:
    result = subprocess.run(
        ["bash", str(CAMPAIGN)], env=_environment(tmp_path, stage),
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "docker" not in result.stdout.lower() + result.stderr.lower()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("LEHOME_SIMULATOR_DEVICE", "cuda:0"),
        ("LEHOME_WORKER_COUNT", "4"),
        ("LEHOME_ENABLE_HF_UPLOAD", "1"),
        ("LEHOME_SKIP_ROUND_SEAL", "0"),
        ("LEHOME_MAX_WORKER_RESTARTS", "1"),
        ("LEHOME_COMPLETION_METRIC", "accepted_successes"),
        ("LEHOME_POLICY_STEP", "500"),
        ("LEHOME_FIDELITY_DIAGNOSTIC_STAGE", "C"),
    ],
)
def test_fidelity_diagnostic_rejects_any_weakened_runtime_tuple(
    tmp_path: Path, field: str, value: str,
) -> None:
    environment = _environment(tmp_path)
    environment[field] = value
    result = subprocess.run(
        ["bash", str(CAMPAIGN)], env=environment,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "fidelity diagnostic" in result.stderr.lower()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows[0].update(seed=2026082790, source_seed=2026082790),
        lambda rows: rows[0].update(garment="Top_Short_Seen_1", garment_name="Top_Short_Seen_1"),
        lambda rows: rows[0].update(category="pant_short"),
        lambda rows: rows[0].update(strategy="replay"),
        lambda rows: rows[0].update(extra=True),
    ],
)
def test_fidelity_diagnostic_rejects_noncanonical_descriptor_rows(tmp_path: Path, mutation) -> None:
    environment = _environment(tmp_path)
    matrix = Path(environment["LEHOME_ATTEMPT_MATRIX"])
    rows = json.loads(matrix.read_text(encoding="utf-8"))
    mutation(rows)
    matrix.write_text(json.dumps(rows, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    environment["LEHOME_ATTEMPT_MATRIX_SHA256"] = hashlib.sha256(matrix.read_bytes()).hexdigest()
    result = subprocess.run(
        ["bash", str(CAMPAIGN)], env=environment,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "fidelity diagnostic" in result.stderr.lower()


def test_wrapper_materializes_only_the_exact_isolated_one_plus_three_plan(tmp_path: Path) -> None:
    base = tmp_path / "fidelity-diagnostic-test-run"
    environment = os.environ.copy()
    environment.update({
        "LEHOME_FIDELITY_DIAGNOSTIC_ROOT": str(base),
        "LEHOME_HOST_CODE_ROOT": str(ROOT),
        "LEHOME_ROLLOUT_IMAGE": ROLLOUT_IMAGE,
        "LEHOME_FIDELITY_DIAGNOSTIC_VALIDATE_ONLY": "1",
    })
    result = subprocess.run(
        ["bash", str(WRAPPER)], env=environment,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads((base / "stage-a" / "matrix.json").read_text()) == [_row("A", 2026082789)]
    assert json.loads((base / "stage-b" / "matrix.json").read_text()) == [
        _row("B", seed, index)
        for index, seed in enumerate((2026082709, 2026082749, 2026082789), start=1)
    ]
    assert not list(base.rglob("ledger.sqlite3"))
    assert not list(base.rglob("*.strict.seal.json"))
    assert not list(base.rglob("hf-sync-receipts"))


def test_wrapper_refuses_existing_or_real_campaign_roots(tmp_path: Path) -> None:
    existing = tmp_path / "fidelity-diagnostic-existing"
    existing.mkdir()
    for root in (existing, tmp_path / "simple-curriculum-400-600"):
        environment = os.environ.copy()
        environment.update({
            "LEHOME_FIDELITY_DIAGNOSTIC_ROOT": str(root),
            "LEHOME_HOST_CODE_ROOT": str(ROOT),
            "LEHOME_ROLLOUT_IMAGE": ROLLOUT_IMAGE,
            "LEHOME_FIDELITY_DIAGNOSTIC_VALIDATE_ONLY": "1",
        })
        result = subprocess.run(
            ["bash", str(WRAPPER)], env=environment,
            capture_output=True, text=True, check=False,
        )
        assert result.returncode == 2


def test_diagnostic_path_is_structurally_nonpublishing_and_fail_closed() -> None:
    campaign = CAMPAIGN.read_text(encoding="utf-8")
    wrapper = WRAPPER.read_text(encoding="utf-8")
    assert 'LEHOME_ENABLE_HF_UPLOAD="0"' in wrapper
    assert 'LEHOME_SKIP_ROUND_SEAL="1"' in wrapper
    assert 'LEHOME_MAX_WORKER_RESTARTS="0"' in wrapper
    assert 'LEHOME_COMPLETION_METRIC="terminal_outcomes"' in wrapper
    assert 'LEHOME_SIMULATOR_DEVICE="cpu"' in wrapper
    assert 'LEHOME_WORKER_COUNT="1"' in wrapper
    assert 'campaign_has_infrastructure_abort' in campaign
    assert 'sleep 1' in campaign


@pytest.mark.parametrize("stage_a_outcome", ["accepted", "rejected"])
def test_wrapper_runs_stage_b_only_after_one_ordinary_stage_a_outcome(
    tmp_path: Path, stage_a_outcome: str,
) -> None:
    base = tmp_path / f"fidelity-diagnostic-gated-{stage_a_outcome}"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launches = tmp_path / "launches.txt"
    fake_bash = fake_bin / "bash"
    fake_bash.write_text(textwrap.dedent(f"""\
        #!/bin/sh
        set -eu
        printf '%s\\n' "$LEHOME_FIDELITY_DIAGNOSTIC_STAGE" >> {str(launches)!r}
        PYTHONPATH={str(ROOT / 'source' / 'lehome')!r} {sys.executable!r} - "$LEHOME_CAMPAIGN_ROOT" "$LEHOME_ATTEMPT_MATRIX" "$LEHOME_MAX_ATTEMPTS" "$LEHOME_TARGET_ACCEPTED" "$LEHOME_FIDELITY_DIAGNOSTIC_STAGE" <<'PY'
        import json
        import sys
        from pathlib import Path
        from lehome.flywheel.task_ledger import TaskLedger

        root, matrix_path = Path(sys.argv[1]), Path(sys.argv[2])
        rows = json.loads(matrix_path.read_text(encoding="utf-8"))
        ledger = TaskLedger(root / "ledger.sqlite3", attempt_matrix=rows,
                            max_attempts=int(sys.argv[3]), target_accepted=int(sys.argv[4]),
                            completion_metric="terminal_outcomes")
        stage = sys.argv[5]
        try:
            for index, row in enumerate(rows, start=1):
                lease = ledger.lease_next("diagnostic-worker", lease_duration_ns=10**18)
                attempt_id = lease.attempt.attempt_id
                output = root / "diagnostic-worker" / "session-1" / row["attempt_id"] / lease.lease_id / f"generation-{{index}}"
                output.mkdir(parents=True)
                (output / "worker-receipt.json").write_text(json.dumps({{
                    "episode_generation": index, "session_id": "session-1",
                }}), encoding="utf-8")
                ledger.record_terminal("diagnostic-worker", attempt_id, lease.lease_id, str(output))
                outcome = {stage_a_outcome!r} if stage == "A" else "rejected"
                ledger.validate_terminal(
                    attempt_id, outcome,
                    artifact_id=f"diagnostic-artifact-{{stage}}-{{index}}" if outcome == "accepted" else None,
                )
        finally:
            ledger.close()
        PY
    """), encoding="utf-8")
    fake_bash.chmod(0o755)
    environment = os.environ.copy()
    environment.update({
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "LEHOME_FIDELITY_DIAGNOSTIC_ROOT": str(base),
        "LEHOME_HOST_CODE_ROOT": str(ROOT),
        "LEHOME_ROLLOUT_IMAGE": ROLLOUT_IMAGE,
    })

    result = subprocess.run(
        ["/bin/bash", str(WRAPPER)], env=environment,
        capture_output=True, text=True, check=False,
    )

    assert result.returncode == 0, result.stderr
    assert launches.read_text(encoding="utf-8").splitlines() == ["A", "B"]


def test_wrapper_stops_before_stage_b_when_stage_a_has_an_abort(tmp_path: Path) -> None:
    base = tmp_path / "fidelity-diagnostic-gated-abort"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launches = tmp_path / "launches.txt"
    fake_bash = fake_bin / "bash"
    fake_bash.write_text(textwrap.dedent(f"""\
        #!/bin/sh
        set -eu
        printf '%s\\n' "$LEHOME_FIDELITY_DIAGNOSTIC_STAGE" >> {str(launches)!r}
        PYTHONPATH={str(ROOT / 'source' / 'lehome')!r} {sys.executable!r} - "$LEHOME_CAMPAIGN_ROOT" "$LEHOME_ATTEMPT_MATRIX" <<'PY'
        import json
        import sys
        from pathlib import Path
        from lehome.flywheel.task_ledger import TaskLedger

        root, matrix_path = Path(sys.argv[1]), Path(sys.argv[2])
        rows = json.loads(matrix_path.read_text(encoding="utf-8"))
        ledger = TaskLedger(root / "ledger.sqlite3", attempt_matrix=rows,
                            max_attempts=1, target_accepted=1, completion_metric="terminal_outcomes")
        try:
            lease = ledger.lease_next("diagnostic-worker", lease_duration_ns=10**18)
            ledger.record_infrastructure_abort("diagnostic-worker", lease.attempt.attempt_id, lease.lease_id,
                                               reason="runtime_evidence_invalid")
        finally:
            ledger.close()
        PY
    """), encoding="utf-8")
    fake_bash.chmod(0o755)
    environment = os.environ.copy()
    environment.update({
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "LEHOME_FIDELITY_DIAGNOSTIC_ROOT": str(base),
        "LEHOME_HOST_CODE_ROOT": str(ROOT),
        "LEHOME_ROLLOUT_IMAGE": ROLLOUT_IMAGE,
    })

    result = subprocess.run(
        ["/bin/bash", str(WRAPPER)], env=environment,
        capture_output=True, text=True, check=False,
    )

    assert result.returncode != 0
    assert launches.read_text(encoding="utf-8").splitlines() == ["A"]
    assert not (base / "stage-b" / "ledger.sqlite3").exists()
