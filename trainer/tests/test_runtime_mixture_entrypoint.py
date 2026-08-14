from __future__ import annotations

from pathlib import Path

import pytest


def test_factory_replacement_is_narrow_and_is_restored_after_official_runner(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture_entrypoint import run_official_launcher

    setup = type("Setup", (), {"DatasetFactory": object})
    original = setup.DatasetFactory
    seen: dict[str, object] = {}

    def runner(path: str, argv: list[str]) -> None:
        seen["path"], seen["argv"], seen["factory"] = path, argv, setup.DatasetFactory

    run_official_launcher(
        official_launch=tmp_path / "launch_finetune.py", setup_module=setup, setup_sha256="a" * 64,
        mixture_manifest="mixture.json", window_index="windows.json", mounts_descriptor="mounts.json", resume_sample_offset=64,
        runner=runner, hash_file=lambda _path: "a" * 64,
    )
    assert seen["factory"] is not original
    assert setup.DatasetFactory is original
    assert seen["argv"] == []


def test_entrypoint_refuses_setup_hash_drift_before_patch(tmp_path: Path) -> None:
    from lehome_train.groot.runtime_mixture_entrypoint import run_official_launcher

    setup = type("Setup", (), {"DatasetFactory": object})
    with pytest.raises(ValueError, match="setup hash"):
        run_official_launcher(official_launch=tmp_path / "launch_finetune.py", setup_module=setup, setup_sha256="a" * 64, mixture_manifest="m", window_index="w", mounts_descriptor="d", hash_file=lambda _path: "b" * 64)
