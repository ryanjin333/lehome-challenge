"""The inference overlay must survive the evaluator's clean subprocess env."""
from pathlib import Path
import zipfile

import pytest

from scripts import run_official_lehome_comparison as comparison


def test_policy_subprocess_receives_only_explicit_dependency_overlay(tmp_path, monkeypatch):
    monkeypatch.setenv("LEHOME_N15_DEPENDENCY_SITE", "/flash/site-packages")
    monkeypatch.setenv("PYTHONPATH", "/untrusted-parent")
    env = comparison._execution_env(
        source_root=tmp_path / "source", log_root=tmp_path / "logs",
        isaaclab_root=tmp_path / "isaaclab", isaaclab_tasks_root=tmp_path / "tasks",
        native_site_root=tmp_path / "native",
        policy=comparison.PolicyDefinition("candidate-n15", "lerobot", checkpoint_root=tmp_path),
        sanitized_config_root=tmp_path / "config", compatibility_receipt=tmp_path / "receipt",
        dependency_site="/flash/site-packages",
    )
    assert "/flash/site-packages" in env["PYTHONPATH"].split(":")
    assert "/untrusted-parent" not in env["PYTHONPATH"].split(":")


def test_unrelated_lerobot_flow_ignores_inherited_overlay(tmp_path, monkeypatch):
    monkeypatch.setenv("LEHOME_N15_DEPENDENCY_SITE", "/unrelated-stale-path")
    env = comparison._execution_env(
        source_root=tmp_path, log_root=tmp_path, isaaclab_root=tmp_path,
        isaaclab_tasks_root=tmp_path, native_site_root=tmp_path,
        policy=comparison.PolicyDefinition("competitor-n15", "lerobot", checkpoint_root=tmp_path),
        sanitized_config_root=tmp_path, compatibility_receipt=tmp_path,
    )
    assert "/unrelated-stale-path" not in env["PYTHONPATH"]


def test_materialization_authenticates_all_wheels_before_writing(tmp_path, monkeypatch):
    from scripts import prepare_n15_dependency_overlay as overlay
    wheel = tmp_path / "fixture.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("fixture/__init__.py", "VALUE = 42\n")
    calls = []
    monkeypatch.setattr(overlay, "inspect_flash_attention_overlay", lambda: calls.append("flash"))
    monkeypatch.setattr(overlay, "inspect_public_pyproject_dependencies_overlay", lambda: calls.append("public"))
    monkeypatch.setattr(overlay, "FLASH_ATTENTION_WHEEL_PATH", wheel)
    monkeypatch.setattr(overlay, "PUBLIC_PYPROJECT_DEPENDENCY_WHEELS", ())
    target = tmp_path / "site-packages"
    overlay.materialize(target)
    assert calls == ["flash", "public"]
    assert (target / "fixture/__init__.py").read_text() == "VALUE = 42\n"
    with pytest.raises(FileExistsError):
        overlay.materialize(target)


def test_authentication_failure_leaves_no_partial_overlay(tmp_path, monkeypatch):
    from scripts import prepare_n15_dependency_overlay as overlay
    def reject():
        raise ValueError("wheel mismatch")
    monkeypatch.setattr(overlay, "inspect_flash_attention_overlay", reject)
    target = tmp_path / "site-packages"
    with pytest.raises(ValueError, match="wheel mismatch"):
        overlay.materialize(target)
    assert not target.exists()


def test_focused_launcher_uses_ram_overlay_for_both_dependency_consumers():
    root = Path(__file__).resolve().parents[1]
    text = (root / "rollout_appliance/run_public_n15_focused_gate.sh").read_text()
    assert "uv pip install" not in text
    assert text.count('--tmpfs /flash:rw,exec,size=2g,mode=700') == 2
    assert text.count('prepare_n15_dependency_overlay.py') == 2
    assert '--env LEHOME_N15_DEPENDENCY_SITE=/flash/site-packages' in text
