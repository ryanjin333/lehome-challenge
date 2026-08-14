from __future__ import annotations

from types import SimpleNamespace
from dataclasses import dataclass

import pytest


def test_entrypoint_injects_only_dataset_factory_and_preserves_config() -> None:
    from lehome_train.groot.runtime_mixture_entrypoint import run_runtime_mixture_finetune

    received: dict[str, object] = {}

    def run(config: object) -> str:
        received["config"] = config
        return "ran"

    upstream = SimpleNamespace(FinetuneConfig=lambda **kwargs: SimpleNamespace(**kwargs), run=run)
    result = run_runtime_mixture_finetune(
        {"model_path": "/model", "max_steps": 12, "dataset_path": "/old", "tune_llm": False},
        mixture_manifest="mixture.json", window_index="windows.json", mounts_descriptor="mounts.json", resume_sample_offset=64,
        upstream_module=upstream,
    )
    assert result == "ran"
    config = received["config"]
    assert config.model_path == "/model"
    assert config.max_steps == 12
    assert config.dataset_path == "/old"
    assert callable(config.dataset_factory)
    assert not hasattr(config, "runtime_mixture_manifest")
    assert not hasattr(config, "resume_sample_offset")


def test_entrypoint_fails_closed_when_pinned_surface_drifts() -> None:
    from lehome_train.groot.runtime_mixture_entrypoint import run_runtime_mixture_finetune

    with pytest.raises(ValueError, match="factory|run|FinetuneConfig"):
        run_runtime_mixture_finetune({}, mixture_manifest="m", window_index="w", mounts_descriptor="d", upstream_module=object())


def test_entrypoint_rejects_the_official_style_config_without_an_injection_seam() -> None:
    from lehome_train.groot.runtime_mixture_entrypoint import run_runtime_mixture_finetune

    @dataclass
    class OfficialStyleConfig:
        dataset_path: str

    with pytest.raises(ValueError, match="dataset factory"):
        run_runtime_mixture_finetune(
            {"dataset_path": "/old"}, mixture_manifest="m", window_index="w", mounts_descriptor="d",
            upstream_module=SimpleNamespace(FinetuneConfig=OfficialStyleConfig, run=lambda _config: None),
        )
