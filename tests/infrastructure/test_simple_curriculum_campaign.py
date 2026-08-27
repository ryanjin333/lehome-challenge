"""Offline exact-partition contract for the stopped simple-curriculum campaign."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "rollout_appliance" / "run_12k_campaign.sh"


_TUPLES = {
    "calibration-head": (100, 100, 150, "calibration"),
    "calibration-tail": (300, 300, 400, "calibration"),
    "curriculum-a": (300, 300, 400, "curriculum"),
    "curriculum-b": (300, 300, 400, "curriculum"),
}


def _rows(partition_id: str, count: int, stage: str) -> list[dict[str, object]]:
    prefixes = ("Top_Long", "Top_Short", "Pant_Long", "Pant_Short")
    return [
        {
            "campaign_kind": "simple_curriculum_source_v1", "logical_stage": stage,
            "attempt_id": f"{partition_id}-{index:03d}", "trial_id": f"trial-{partition_id}-{index:03d}",
            "garment": f"{prefixes[index % 4]}_Seen_{index % 10}",
            "garment_name": f"{prefixes[index % 4]}_Seen_{index % 10}",
            "category": ("top_long", "top_short", "pant_long", "pant_short")[index % 4],
            "release_stage": "seen", "seed": 1_000_000 + index,
            "source_seed": 1_000_000 + index, "strategy": "canonical",
            "partition_id": partition_id, "parent_matrix_sha256": "a" * 64,
        }
        for index in range(count)
    ]


def _code_hash() -> str:
    digest = hashlib.sha256()
    for relative in ("source/lehome", "scripts", "rollout_appliance"):
        for path in sorted((ROOT / relative).rglob("*")):
            if path.is_file() and not path.is_symlink():
                digest.update(path.relative_to(ROOT).as_posix().encode() + b"\0")
                digest.update(path.read_bytes())
    return digest.hexdigest()


def _environment(tmp_path: Path, partition_id: str) -> dict[str, str]:
    count, target, budget, stage = _TUPLES[partition_id]
    matrix = tmp_path / f"{partition_id}.json"
    matrix.write_text(json.dumps(_rows(partition_id, count, stage), sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    environment = os.environ.copy()
    environment.update({
        "LEHOME_WORKSPACE": str(tmp_path / "workspace"), "LEHOME_CAMPAIGN_ROOT": str(tmp_path / "campaign"),
        "LEHOME_ATTEMPT_MATRIX": str(matrix), "LEHOME_ATTEMPT_MATRIX_SHA256": hashlib.sha256(matrix.read_bytes()).hexdigest(),
        "LEHOME_SIMPLE_CURRICULUM_COLLECTION": "1", "LEHOME_SIMULATOR_DEVICE": "cpu",
        "LEHOME_WORKER_COUNT": "4", "LEHOME_ENABLE_HF_UPLOAD": "1", "LEHOME_SKIP_ROUND_SEAL": "1",
        "LEHOME_POLICY_STEP": "12000", "LEHOME_COMPLETION_METRIC": "terminal_outcomes",
        "LEHOME_MAX_ATTEMPTS": str(budget), "LEHOME_TARGET_ACCEPTED": str(target),
        "LEHOME_PARTITION_ID": partition_id, "LEHOME_PARENT_MATRIX_SHA256": "a" * 64,
        "LEHOME_ROLLOUT_IMAGE": "ghcr.io/ryanjin333/lehome-rollout@sha256:" + "d" * 64,
        "LEHOME_VALIDATE_MATRIX_ONLY": "1",
    })
    return environment


@pytest.mark.parametrize("partition_id", _TUPLES)
def test_exact_simple_curriculum_partitions_admit_validation_only_before_docker(tmp_path: Path, partition_id: str) -> None:
    result = subprocess.run(["bash", str(SCRIPT)], env=_environment(tmp_path, partition_id), capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "docker" not in result.stdout.lower() + result.stderr.lower()


@pytest.mark.parametrize("field,value", [
    ("LEHOME_MAX_ATTEMPTS", "151"), ("LEHOME_TARGET_ACCEPTED", "99"),
    ("LEHOME_WORKER_COUNT", "1"), ("LEHOME_ENABLE_HF_UPLOAD", "0"),
    ("LEHOME_SKIP_ROUND_SEAL", "0"), ("LEHOME_POLICY_STEP", "11999"),
    ("LEHOME_SIMULATOR_DEVICE", "cuda:0"), ("LEHOME_EVALUATION_TERMINAL_UPLOAD", "1"),
    ("LEHOME_SUCCESS_REPLAY_CAMPAIGN", "1"), ("LEHOME_HARD_STATE_CAMPAIGN", "1"),
])
def test_exact_simple_curriculum_rejects_weakened_or_conflicting_tuple_before_docker(tmp_path: Path, field: str, value: str) -> None:
    environment = _environment(tmp_path, "calibration-head")
    environment[field] = value
    result = subprocess.run(["bash", str(SCRIPT)], env=environment, capture_output=True, text=True, check=False)
    assert result.returncode == 2
    assert "docker" not in result.stdout.lower() + result.stderr.lower()


@pytest.mark.parametrize("mutate", [
    lambda row: row.update(campaign_kind="other"), lambda row: row.update(logical_stage="curriculum"),
    lambda row: row.update(strategy="mild"), lambda row: row.update(garment="Top_Long_Unseen_0"),
    lambda row: row.update(seed=2**32), lambda row: row.update(partition_id="curriculum-a"),
    lambda row: row.update(parent_matrix_sha256="c" * 64),
])
def test_exact_simple_curriculum_rejects_bad_row_contract_before_docker(tmp_path: Path, mutate) -> None:
    environment = _environment(tmp_path, "calibration-head")
    matrix = Path(environment["LEHOME_ATTEMPT_MATRIX"])
    rows = json.loads(matrix.read_text(encoding="utf-8"))
    mutate(rows[0])
    matrix.write_text(json.dumps(rows, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    environment["LEHOME_ATTEMPT_MATRIX_SHA256"] = hashlib.sha256(matrix.read_bytes()).hexdigest()

    result = subprocess.run(["bash", str(SCRIPT)], env=environment, capture_output=True, text=True, check=False)

    assert result.returncode != 0
    assert "simple curriculum" in result.stderr
    assert "docker" not in result.stdout.lower() + result.stderr.lower()


@pytest.mark.parametrize("descriptor_update, expected_error", [
    ({"active": True}, "does not bind"),
    ({"parent_matrix_sha256": "c" * 64}, "does not bind"),
])
def test_exact_simple_curriculum_resume_refuses_active_or_mismatched_descriptor(
    tmp_path: Path, descriptor_update: dict[str, object], expected_error: str,
) -> None:
    environment = _environment(tmp_path, "calibration-head")
    from lehome.flywheel.task_ledger import TaskLedger

    matrix = json.loads(Path(environment["LEHOME_ATTEMPT_MATRIX"]).read_text(encoding="utf-8"))
    ledger_path = Path(environment["LEHOME_CAMPAIGN_ROOT"]) / "ledger.sqlite3"
    ledger_path.parent.mkdir()
    ledger = TaskLedger(ledger_path, attempt_matrix=matrix, max_attempts=150, target_accepted=100, completion_metric="terminal_outcomes")
    ledger.pause_for_preemption("spot")
    ledger.close()
    context = tmp_path / "preemption.json"
    environment.update({
        "LEHOME_VALIDATE_MATRIX_ONLY": "0", "LEHOME_RESUME_PREEMPTED_ROLLOUT": "1",
        "LEHOME_ROLLOUT_PREEMPTION_CONTEXT": str(context), "PYTHONPATH": str(ROOT / "source" / "lehome"),
    })
    descriptor = {
        "active": False, "campaign_mode": "simple_curriculum_collection", "completion_metric": "terminal_outcomes",
        "partition_id": "calibration-head", "parent_matrix_sha256": "a" * 64, "code_root_sha256": _code_hash(),
        "attempt_matrix_sha256": environment["LEHOME_ATTEMPT_MATRIX_SHA256"],
        "policy_repo": "ryanjin333/lehome-groot-n17-models", "policy_revision": "30ac1a84da67b099e115ad147bcd61e9d60046d3",
        "policy_step": 12000, "policy_artifact_sha256": "3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06",
        "policy_sha256": "e8531e9477b68ac8f7d9fc9564bb66ebfae51f828b44599c4777bd2eb3b72efa",
        "simulator_device": "cpu", "renderer_device": "cuda:0", "policy_device": "cuda:0",
        "trainer_image": "ghcr.io/ryanjin333/lehome-groot-n17-trainer@sha256:b56c16c259b7eda99294f2069e976b53395e665aaf68174d5b13ba458a93b746",
        "rollout_image": environment["LEHOME_ROLLOUT_IMAGE"],
    }
    descriptor.update(descriptor_update)
    context.write_text(json.dumps(descriptor), encoding="utf-8")

    result = subprocess.run(["bash", str(SCRIPT)], env=environment, capture_output=True, text=True, check=False)

    assert result.returncode != 0
    assert expected_error in result.stderr
    assert "docker" not in result.stdout.lower() + result.stderr.lower()


def test_exact_simple_curriculum_resume_resumes_only_a_matching_paused_ledger(tmp_path: Path) -> None:
    from lehome.flywheel.task_ledger import TaskLedger

    environment = _environment(tmp_path, "calibration-head")
    matrix = json.loads(Path(environment["LEHOME_ATTEMPT_MATRIX"]).read_text(encoding="utf-8"))
    ledger_path = Path(environment["LEHOME_CAMPAIGN_ROOT"]) / "ledger.sqlite3"
    ledger_path.parent.mkdir()
    ledger = TaskLedger(ledger_path, attempt_matrix=matrix, max_attempts=150, target_accepted=100, completion_metric="terminal_outcomes")
    ledger.pause_for_preemption("spot")
    ledger.close()
    context = tmp_path / "preemption.json"
    context.write_text(json.dumps({
        "active": False, "campaign_mode": "simple_curriculum_collection", "completion_metric": "terminal_outcomes",
        "partition_id": "calibration-head", "parent_matrix_sha256": "a" * 64, "code_root_sha256": _code_hash(),
        "attempt_matrix_sha256": environment["LEHOME_ATTEMPT_MATRIX_SHA256"],
        "policy_repo": "ryanjin333/lehome-groot-n17-models", "policy_revision": "30ac1a84da67b099e115ad147bcd61e9d60046d3",
        "policy_step": 12000, "policy_artifact_sha256": "3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06",
        "policy_sha256": "e8531e9477b68ac8f7d9fc9564bb66ebfae51f828b44599c4777bd2eb3b72efa",
        "simulator_device": "cpu", "renderer_device": "cuda:0", "policy_device": "cuda:0",
        "trainer_image": "ghcr.io/ryanjin333/lehome-groot-n17-trainer@sha256:b56c16c259b7eda99294f2069e976b53395e665aaf68174d5b13ba458a93b746",
        "rollout_image": environment["LEHOME_ROLLOUT_IMAGE"],
    }), encoding="utf-8")
    environment.update({
        "LEHOME_VALIDATE_MATRIX_ONLY": "0", "LEHOME_RESUME_PREEMPTED_ROLLOUT": "1",
        "LEHOME_ROLLOUT_PREEMPTION_CONTEXT": str(context),
        "PYTHONPATH": str(ROOT / "source" / "lehome"),
    })

    result = subprocess.run(["bash", str(SCRIPT)], env=environment, capture_output=True, text=True, check=False)

    assert result.returncode == 2  # Upload credential gate runs after the durable resume action.
    assert "LEHOME_HF_TOKEN_FILE" in result.stderr
    reopened = TaskLedger(ledger_path, attempt_matrix=matrix, max_attempts=150, target_accepted=100, completion_metric="terminal_outcomes")
    try:
        assert reopened.is_stopped is False
        assert [event.event_type for event in reopened.events()] == ["campaign_paused", "campaign_resumed"]
    finally:
        reopened.close()


def test_exact_simple_curriculum_resume_refuses_a_terminal_ledger(tmp_path: Path) -> None:
    from lehome.flywheel.task_ledger import TaskLedger

    environment = _environment(tmp_path, "calibration-head")
    matrix = json.loads(Path(environment["LEHOME_ATTEMPT_MATRIX"]).read_text(encoding="utf-8"))
    ledger_path = Path(environment["LEHOME_CAMPAIGN_ROOT"]) / "ledger.sqlite3"
    ledger_path.parent.mkdir()
    ledger = TaskLedger(ledger_path, attempt_matrix=matrix, max_attempts=150, target_accepted=100, completion_metric="terminal_outcomes")
    ledger.end_campaign("complete")
    ledger.close()
    context = tmp_path / "preemption.json"
    context.write_text(json.dumps({
        "active": False, "campaign_mode": "simple_curriculum_collection", "completion_metric": "terminal_outcomes",
        "partition_id": "calibration-head", "parent_matrix_sha256": "a" * 64, "code_root_sha256": _code_hash(),
        "attempt_matrix_sha256": environment["LEHOME_ATTEMPT_MATRIX_SHA256"],
        "policy_repo": "ryanjin333/lehome-groot-n17-models", "policy_revision": "30ac1a84da67b099e115ad147bcd61e9d60046d3",
        "policy_step": 12000, "policy_artifact_sha256": "3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06",
        "policy_sha256": "e8531e9477b68ac8f7d9fc9564bb66ebfae51f828b44599c4777bd2eb3b72efa",
        "simulator_device": "cpu", "renderer_device": "cuda:0", "policy_device": "cuda:0",
        "trainer_image": "ghcr.io/ryanjin333/lehome-groot-n17-trainer@sha256:b56c16c259b7eda99294f2069e976b53395e665aaf68174d5b13ba458a93b746",
        "rollout_image": environment["LEHOME_ROLLOUT_IMAGE"],
    }), encoding="utf-8")
    environment.update({
        "LEHOME_VALIDATE_MATRIX_ONLY": "0", "LEHOME_RESUME_PREEMPTED_ROLLOUT": "1",
        "LEHOME_ROLLOUT_PREEMPTION_CONTEXT": str(context), "PYTHONPATH": str(ROOT / "source" / "lehome"),
    })

    result = subprocess.run(["bash", str(SCRIPT)], env=environment, capture_output=True, text=True, check=False)

    assert result.returncode != 0
    assert "already terminal" in result.stderr
