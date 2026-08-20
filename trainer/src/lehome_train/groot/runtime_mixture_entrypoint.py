"""Narrow in-process DatasetFactory substitution for pinned GR00T N1.7."""

from __future__ import annotations

import argparse
import hashlib
import importlib
from pathlib import Path
import runpy
import sys
from typing import Any, Callable

from lehome_train.groot.awr_weighting import AwrReplayConfig, load_progress_evidence
from lehome_train.groot.runtime_mixture import (
    load_runtime_contract,
    runtime_dataset_factory_class,
)
from lehome_train.io import canonical_json_sha256


PINNED_SETUP_SHA256 = "bdc3cf8a6b9c92a0e3f46d79dc3de05b0a8c70c4289d44ac0aa1c75698a93f31"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run_official_launcher(*, official_launch: str | Path, setup_module: object, setup_sha256: str, mixture_manifest: str | Path, window_index: str | Path, mounts_descriptor: str | Path, resume_sample_offset: int = 0, resume_global_step: int | None = None, global_batch_size: int | None = None, awr_evidence_path: str | Path | None = None, awr_evidence_sha256: str | None = None, awr_temperature: float | None = None, awr_minimum: float | None = None, awr_maximum: float | None = None, official_argv: list[str] | None = None, runner: Callable[[str, list[str]], Any] | None = None, hash_file: Callable[[Path], str] = _sha256) -> Any:
    """Run the unchanged official script while replacing exactly one local symbol.

    The replacement is restored even when Tyro/configuration/training raises.
    No config fields are fabricated and no other GR00T import is patched.
    """
    launch = Path(official_launch)
    if (
        resume_sample_offset < 0
        or (resume_global_step is None) != (global_batch_size is None)
        or (resume_global_step is not None and (type(resume_global_step) is not int or resume_global_step < 0 or type(global_batch_size) is not int or global_batch_size <= 0 or resume_sample_offset != resume_global_step * global_batch_size))
        or hash_file(launch.parent.parent / "model" / "gr00t_n1d7" / "setup.py") != setup_sha256
    ):
        raise ValueError("pinned GR00T setup hash mismatch")
    original = getattr(setup_module, "DatasetFactory", None)
    if original is None:
        raise ValueError("pinned GR00T DatasetFactory symbol is missing")
    awr_fields = (
        awr_evidence_path,
        awr_evidence_sha256,
        awr_temperature,
        awr_minimum,
        awr_maximum,
    )
    if any(field is not None for field in awr_fields) and not all(
        field is not None for field in awr_fields
    ):
        raise ValueError("runtime AWR configuration must be complete")
    awr_evidence = None
    awr_config = None
    if awr_evidence_path is not None:
        assert awr_evidence_sha256 is not None
        assert awr_temperature is not None
        assert awr_minimum is not None
        assert awr_maximum is not None
        contract = load_runtime_contract(mixture_manifest, mounts_descriptor)
        awr_evidence = load_progress_evidence(
            awr_evidence_path,
            expected_sha256=awr_evidence_sha256,
            mixture_id=contract.manifest.mixture_id,
            mixture_manifest_sha256=canonical_json_sha256(contract.manifest.raw),
        )
        awr_config = AwrReplayConfig(
            temperature=awr_temperature,
            minimum=awr_minimum,
            maximum=awr_maximum,
        )
    replacement = runtime_dataset_factory_class(
        mixture_manifest=mixture_manifest,
        window_index=window_index,
        mounts_descriptor=mounts_descriptor,
        global_sample_offset=resume_sample_offset,
        expected_global_step=resume_global_step,
        global_batch_size=global_batch_size,
        awr_evidence=awr_evidence,
        awr_config=awr_config,
    )
    setattr(setup_module, "DatasetFactory", replacement)
    try:
        argv = list(official_argv or [])
        if runner is not None:
            return runner(str(launch), argv)
        previous = sys.argv
        sys.argv = [str(launch), *argv]
        try:
            return runpy.run_path(str(launch), run_name="__main__")
        finally:
            sys.argv = previous
    finally:
        setattr(setup_module, "DatasetFactory", original)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixture-manifest", required=True)
    parser.add_argument("--window-index", required=True)
    parser.add_argument("--mounts-descriptor", required=True)
    parser.add_argument("--awr-evidence")
    parser.add_argument("--awr-evidence-sha256")
    parser.add_argument("--awr-temperature", type=float)
    parser.add_argument("--awr-minimum", type=float)
    parser.add_argument("--awr-maximum", type=float)
    # The production chunk wrapper injects this authenticated pair after it
    # has inspected the checkpoint.  Users may not persist it in launch config.
    parser.add_argument("--resume-sample-offset", type=int, default=0)
    parser.add_argument("--resume-global-step", type=int)
    parser.add_argument("--global-batch-size", type=int)
    parser.add_argument("--official-launch", required=True)
    parser.add_argument("official_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    checkout = Path(args.official_launch).resolve().parents[2]
    sys.path.insert(0, str(checkout))
    setup = importlib.import_module("gr00t.model.gr00t_n1d7.setup")
    original = args.official_args[1:] if args.official_args[:1] == ["--"] else args.official_args
    run_official_launcher(official_launch=args.official_launch, setup_module=setup, setup_sha256=PINNED_SETUP_SHA256, mixture_manifest=args.mixture_manifest, window_index=args.window_index, mounts_descriptor=args.mounts_descriptor, resume_sample_offset=args.resume_sample_offset, resume_global_step=args.resume_global_step, global_batch_size=args.global_batch_size, awr_evidence_path=args.awr_evidence, awr_evidence_sha256=args.awr_evidence_sha256, awr_temperature=args.awr_temperature, awr_minimum=args.awr_minimum, awr_maximum=args.awr_maximum, official_argv=original)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
