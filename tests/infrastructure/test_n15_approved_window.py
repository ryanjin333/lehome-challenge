"""Approved extensions preserve historical deadlines and the total cost cap."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / 'rollout_appliance/run_public_n15_pipeline_remote.sh'


def run_window(tmp_path, override=None, stage='focused_gate'):
    now = int(time.time())
    plan = tmp_path / 'lifecycle-plan.json'
    plan.write_text('{}\n')
    plan.chmod(0o444)
    old = tmp_path / 'paid-deadline.json'
    old.write_text(json.dumps({
        'schema_version': 1, 'kind': 'lehome_public_n15_paid_deadline_v1',
        'run_id': 'test-run', 'lifecycle_plan_sha256': hashlib.sha256(plan.read_bytes()).hexdigest(),
        'started_unix_seconds': now - 172800, 'deadline_unix_seconds': now - 86400,
    }, sort_keys=True, separators=(',', ':')) + '\n')
    old.chmod(0o444)
    before = old.read_bytes()
    value = dict(run_id='test-run', started_unix_seconds=now - 10,
                 deadline_unix_seconds=now - 10 + 14400, budget_usd=100,
                 prior_reserved_usd=72, hourly_ceiling_usd=3, new_window_max_usd=12,
                 authorization='Explicit user approval')
    value.update(override or {})
    approval = tmp_path / 'approval.json'
    approval.write_text(json.dumps(value) + '\n')
    approval.chmod(0o444)
    result = subprocess.run(['bash', '-c', '''
source "$WRAPPER"
host_next_unfinished_stage() { echo "$TEST_STAGE"; }
verify_conservative_task_budget "$DEADLINE_RECEIPT"
initialize_deadline
initialize_stage_deadline focused_gate 14400 "$(initialize_deadline)"
'''], env={**os.environ, 'WRAPPER': str(WRAPPER), 'TEST_STAGE': stage,
            'LEHOME_N15_RUN_ID': 'test-run', 'LEHOME_N15_PIPELINE_ROOT': str(tmp_path),
            'LEHOME_N15_APPROVED_WINDOW': str(approval)}, capture_output=True, text=True)
    assert old.read_bytes() == before
    return result, value


def test_approved_window_admits_completed_training_without_reset(tmp_path):
    result, value = run_window(tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [str(value['deadline_unix_seconds'])] * 2
    assert not (tmp_path / 'stage-focused_gate-deadline.json').exists()


@pytest.mark.parametrize('override', [
    {'prior_reserved_usd': 0}, {'budget_usd': 101}, {'hourly_ceiling_usd': 1},
    {'new_window_max_usd': 0}, {'run_id': 'another-run'},
    {'deadline_unix_seconds': 0}, {'authorization': ''},
])
def test_invalid_approval_rejected(tmp_path, override):
    result, _ = run_window(tmp_path, override)
    assert result.returncode != 0


def test_renewal_carries_forward_prior_reservation(tmp_path):
    result, _ = run_window(tmp_path, {'prior_reserved_usd': 84})
    assert result.returncode == 0, result.stderr


def test_renewal_rejects_total_over_cap(tmp_path):
    result, _ = run_window(tmp_path, {'prior_reserved_usd': 96})
    assert result.returncode != 0


def test_extension_cannot_restart_training(tmp_path):
    result, _ = run_window(tmp_path, stage='train')
    assert result.returncode != 0
