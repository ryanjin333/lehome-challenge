"""Executable runtime-anchor lifecycle proof; no provider or network access."""

from __future__ import annotations

from pathlib import Path

import pytest

import lehome_train.groot.production_runtime as runtime_module


def test_resumed_runtime_accepts_the_authenticated_predecessor_anchor_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 2K resumed process must be able to chain its durable 1K anchor."""

    prepared, output, cache = (tmp_path / "prepared", tmp_path / "output", tmp_path / "cache")
    monkeypatch.setattr(runtime_module, "_ALLOWED_ROOTS", (prepared, output, cache))

    with pytest.raises(ValueError, match="launch_config"):
        runtime_module.ProductionRuntime().runtime_mixture_train(
            {
                "launch_config": str(prepared / "launch.json"),
                "experiment_config": str(prepared / "experiment.json"),
                "runtime_manifest": str(prepared / "mixture.json"),
                "runtime_window_index": str(prepared / "windows.json"),
                "runtime_normalization": str(prepared / "normalization.json"),
                "runtime_mounts_descriptor": str(prepared / "mounts.json"),
                "runtime_source_evidence": str(prepared / "sources.json"),
                "cpu_pilot_receipt": str(prepared / "pilot.json"),
                "warmup_receipt": str(prepared / "warmup.json"),
                "runtime_warmup_binding": str(prepared / "binding.json"),
                "runtime_resume_archive": str(prepared / "resume.tar"),
                "runtime_resume_descriptor": str(prepared / "resume.json"),
                "runtime_resume_cursor": {
                    "optimizer_step": 1000,
                    "global_sample_offset": 64_000,
                    "physical_batch_size": 64,
                    "action_horizon": 16,
                },
                # This durable link is returned only by anchor discovery, not
                # supplied by an untrusted trainer process.
                "runtime_resume_anchor": {
                    "immutable_anchor_revision": "a" * 40,
                    "anchor_sha256": "b" * 64,
                },
                "runtime_resume_publication": {},
                "checkpoint_repository": runtime_module.DEFAULT_MODEL_REPO,
                "checkpoint_revision": "main",
                "publisher_token_file": str(prepared / "publisher.token"),
                "instance_id": 10,
                "result_output": str(output / "result.json"),
                "status_output": str(output / "status.json"),
            }
        )
