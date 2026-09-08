"""Training provenance must retain its original interpreter inside containers."""
from pathlib import Path

import pytest


def test_container_native_interpreter_needs_no_host_mounts():
    from scripts.prepare_n15_training_runtime_mounts import mounts_for_interpreter
    assert mounts_for_interpreter(Path('/opt/lehome-challenge/.venv/bin/python')) == []


def test_unknown_interpreter_rejected_before_execution():
    from scripts.prepare_n15_training_runtime_mounts import mounts_for_interpreter
    with pytest.raises(ValueError, match='unexpected'):
        mounts_for_interpreter(Path('/untrusted/python'))


def test_runtime_mounts_include_venv_and_base_read_only(tmp_path):
    from scripts.prepare_n15_training_runtime_mounts import runtime_mounts
    venv = tmp_path / "venv"
    base = tmp_path / "python" / "cpython-3.11"
    (venv / "bin").mkdir(parents=True)
    (base / "bin").mkdir(parents=True)
    executable = base / "bin/python3.11"
    executable.write_text("fixture")
    (venv / "bin/python").symlink_to(executable)
    result = runtime_mounts(venv / "bin/python", venv, base, allowed_root=tmp_path)
    assert result == [f"type=bind,src={venv},dst={venv},readonly",
                      f"type=bind,src={base},dst={base},readonly"]


def test_runtime_mounts_reject_escape(tmp_path):
    from scripts.prepare_n15_training_runtime_mounts import runtime_mounts
    with pytest.raises(ValueError, match="outside"):
        runtime_mounts(Path('/usr/bin/python3'), Path('/usr'), Path('/usr'),
                       allowed_root=tmp_path)


def test_exact_external_base_runtime_is_allowed_without_mounting_home(tmp_path):
    from scripts.prepare_n15_training_runtime_mounts import runtime_mounts
    tools = tmp_path / 'tools'
    venv = tools / 'venv'
    base = tmp_path / 'home/ubuntu/.local/share/uv/python/cpython-3.11.16'
    (venv / 'bin').mkdir(parents=True)
    (base / 'bin').mkdir(parents=True)
    executable = base / 'bin/python3.11'
    executable.write_text('fixture')
    (venv / 'bin/python').symlink_to(executable)
    mounts = runtime_mounts(venv / 'bin/python', venv, base,
                            allowed_root=tools, allowed_base=base)
    assert mounts == [f'type=bind,src={venv},dst={venv},readonly',
                      f'type=bind,src={base},dst={base},readonly']
    with pytest.raises(ValueError, match='outside'):
        runtime_mounts(venv / 'bin/python', venv, base.parent,
                       allowed_root=tools, allowed_base=base)


def test_runtime_mounts_reject_root_mount(tmp_path):
    from scripts.prepare_n15_training_runtime_mounts import runtime_mounts
    with pytest.raises(ValueError, match="outside"):
        runtime_mounts(tmp_path / 'bin/python', tmp_path, tmp_path,
                       allowed_root=tmp_path)


def test_intermediate_uv_alias_is_mapped_to_the_verified_base(tmp_path):
    from scripts.prepare_n15_training_runtime_mounts import runtime_mounts
    venv = tmp_path / 'venv'
    base = tmp_path / 'python/cpython-3.11.16'
    alias = base.parent / 'cpython-3.11'
    (venv / 'bin').mkdir(parents=True)
    (base / 'bin').mkdir(parents=True)
    (base / 'bin/python3.11').write_text('fixture')
    alias.symlink_to(base, target_is_directory=True)
    (venv / 'bin/python').symlink_to(alias / 'bin/python3.11')
    mounts = runtime_mounts(venv / 'bin/python', venv, base, allowed_root=tmp_path)
    assert f'type=bind,src={base},dst={alias},readonly' in mounts


def test_launchers_supply_runtime_mounts_to_identity_consumers():
    root = Path(__file__).resolve().parents[1] / 'rollout_appliance'
    focused = (root / 'run_public_n15_focused_gate.sh').read_text()
    harvest = (root / 'run_public_n15_harvest.sh').read_text()
    assert 'prepare_n15_training_runtime_mounts.py' in focused
    assert focused.count('"${training_runtime_mounts[@]}"') == 2
    assert 'prepare_n15_training_runtime_mounts.py' in harvest
    assert '"${training_runtime_mounts[@]}"' in harvest


def test_alias_outside_base_parent_is_rejected(tmp_path):
    from scripts.prepare_n15_training_runtime_mounts import runtime_mounts
    venv = tmp_path / 'venv'
    base = tmp_path / 'python/cpython-3.11.16'
    alias = tmp_path / 'unrelated-alias'
    (venv / 'bin').mkdir(parents=True)
    (base / 'bin').mkdir(parents=True)
    (base / 'bin/python3.11').write_text('fixture')
    alias.symlink_to(base, target_is_directory=True)
    (venv / 'bin/python').symlink_to(alias / 'bin/python3.11')
    with pytest.raises(ValueError, match='alias'):
        runtime_mounts(venv / 'bin/python', venv, base, allowed_root=tmp_path)
