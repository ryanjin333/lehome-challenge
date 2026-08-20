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


def test_entrypoint_passes_checkpoint_step_and_global_batch_to_runtime_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lehome_train.groot import runtime_mixture_entrypoint as entrypoint

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        entrypoint,
        "runtime_dataset_factory_class",
        lambda **kwargs: captured.update(kwargs) or object,
    )
    setup = type("Setup", (), {"DatasetFactory": object})
    entrypoint.run_official_launcher(
        official_launch=tmp_path / "launch_finetune.py", setup_module=setup,
        setup_sha256="a" * 64, mixture_manifest="m", window_index="w",
        mounts_descriptor="d", resume_sample_offset=640,
        resume_global_step=10, global_batch_size=64,
        runner=lambda *_: None, hash_file=lambda _path: "a" * 64,
    )
    assert captured["expected_global_step"] == 10
    assert captured["global_batch_size"] == 64


def test_entrypoint_loads_hash_bound_awr_evidence_before_patching_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace
    from lehome_train.groot import runtime_mixture_entrypoint as entrypoint

    evidence_path = tmp_path / "awr.json"
    evidence_path.write_text("{}", encoding="utf-8")
    evidence = object()
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        entrypoint,
        "load_runtime_contract",
        lambda *_args: SimpleNamespace(
            manifest=SimpleNamespace(mixture_id="a" * 64, raw={"bound": True})
        ),
        raising=False,
    )
    monkeypatch.setattr(
        entrypoint,
        "load_progress_evidence",
        lambda path, **kwargs: captured.update(path=path, **kwargs) or evidence,
        raising=False,
    )
    monkeypatch.setattr(
        entrypoint,
        "runtime_dataset_factory_class",
        lambda **kwargs: captured.update(factory=kwargs) or object,
    )
    setup = type("Setup", (), {"DatasetFactory": object})

    entrypoint.run_official_launcher(
        official_launch=tmp_path / "launch_finetune.py",
        setup_module=setup,
        setup_sha256="b" * 64,
        mixture_manifest="/runtime/mixture.json",
        window_index="/runtime/windows.json",
        mounts_descriptor="/runtime/mounts.json",
        awr_evidence_path=evidence_path,
        awr_evidence_sha256="d" * 64,
        awr_temperature=0.75,
        awr_minimum=0.5,
        awr_maximum=3.0,
        runner=lambda *_: None,
        hash_file=lambda _path: "b" * 64,
    )

    assert captured["path"] == evidence_path
    assert captured["expected_sha256"] == "d" * 64
    assert captured["mixture_id"] == "a" * 64
    factory = captured["factory"]
    assert isinstance(factory, dict)
    assert factory["awr_evidence"] is evidence
    assert factory["awr_config"].to_dict() == {
        "temperature": 0.75,
        "minimum": 0.5,
        "maximum": 3.0,
    }
