"""Pinned-GR00T entry boundary for the runtime-mixture dataset factory.

It intentionally does not edit or copy NVIDIA's training loop.  The caller
supplies normal official ``FinetuneConfig`` values; this module replaces only
the upstream dataset-factory seam and calls the upstream ``run`` unchanged.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any, Mapping

from lehome_train.groot.runtime_mixture import make_dataset_factory


def _upstream_surface(module: object) -> tuple[object, object]:
    config_type = getattr(module, "FinetuneConfig", None)
    run = getattr(module, "run", None)
    if not callable(config_type) or not callable(run):
        raise ValueError("pinned GR00T FinetuneConfig or run integration symbol is missing")
    return config_type, run


def _accepts_dataset_factory(config_type: object) -> None:
    """Reject a drifted upstream API rather than monkeypatching it."""

    try:
        signature = inspect.signature(config_type)
    except (TypeError, ValueError) as error:
        raise ValueError("pinned GR00T dataset factory integration symbol is not inspectable") from error
    if "dataset_factory" in signature.parameters:
        return
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return
    raise ValueError("pinned GR00T dataset factory integration symbol is missing")


def run_runtime_mixture_finetune(
    official_config: Mapping[str, object],
    *,
    mixture_manifest: str | Path,
    window_index: str | Path,
    mounts_descriptor: str | Path,
    resume_sample_offset: int = 0,
    upstream_module: object | None = None,
) -> Any:
    """Build official config fields unchanged, except for ``dataset_factory``.

    ``window_index`` is checked against the manifest by ``make_dataset_factory``;
    receiving it explicitly prevents a caller from accidentally believing a
    different index is selected.  It is not injected into GR00T configuration.
    """

    if resume_sample_offset < 0:
        raise ValueError("resume sample offset must be nonnegative")
    manifest = Path(mixture_manifest)
    if upstream_module is None:
        try:
            upstream_module = importlib.import_module("gr00t.experiment.launch_finetune")
        except ImportError as error:
            raise ValueError("pinned GR00T launch_finetune module is unavailable") from error
    config_type, run = _upstream_surface(upstream_module)
    _accepts_dataset_factory(config_type)
    factory = make_dataset_factory(
        mixture_manifest=manifest,
        mounts_descriptor=mounts_descriptor,
        global_sample_offset=resume_sample_offset,
        expected_window_index=window_index,
    )
    values = dict(official_config)
    if "dataset_factory" in values:
        raise ValueError("official config must not override the runtime dataset factory")
    values["dataset_factory"] = factory
    # Construction fails closed if the checked pinned class does not expose the
    # narrowly expected injection seam. No attributes are patched afterwards.
    try:
        config = config_type(**values)
    except (TypeError, ValueError) as error:
        raise ValueError("pinned GR00T config rejected runtime dataset factory") from error
    return run(config)
