"""Training and organizer evaluation require different pinned source trees."""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / 'rollout_appliance/run_public_n15_pipeline_remote.sh'


def test_focused_stage_routes_organizer_source_not_training_fork():
    result = subprocess.run(['bash', '-c', '''
source "$WRAPPER"
remote() { printf '%s\\n' "$@"; cat >/dev/null; }
focused_stage
'''], env={**os.environ, 'WRAPPER': str(WRAPPER),
            'LEHOME_N15_PUBLIC_SOURCE_ROOT': '/mnt/training-fork',
            'LEHOME_OFFICIAL_SOURCE_ROOT': '/mnt/organizer-source'},
        capture_output=True, text=True, check=True)
    args = result.stdout.splitlines()
    assert args[6] == '/mnt/organizer-source'
    assert '/mnt/training-fork' not in args


def test_organizer_source_is_exported_to_paid_stage_child():
    text = WRAPPER.read_text()
    export = next(line for line in text.splitlines() if line.startswith('  export REMOTE_ROOT '))
    assert ' OFFICIAL_SOURCE_ROOT ' in export


def test_organizer_source_is_required_before_paid_admission():
    text = WRAPPER.read_text()
    admission = next(line for line in text.splitlines() if 'all canonical remote inputs are required' in line)
    assert '-n "$OFFICIAL_SOURCE_ROOT"' in admission
