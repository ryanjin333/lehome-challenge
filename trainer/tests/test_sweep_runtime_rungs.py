"""Short-rung supervisor behaviour isolated from the legacy 2K flow."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace


def _complete_checkpoint(root: Path, step: int) -> None:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    for name in ("model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (checkpoint / name).write_bytes(f"{name}-{step}".encode())
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "log_history": [{"step": step, "loss": 0.1}]}),
        encoding="utf-8",
    )


def test_sweep_500_publishes_its_terminal_checkpoint_without_waiting_for_later_rungs(
    tmp_path: Path,
) -> None:
    from lehome_train.groot.continuous_training import run_sweep_supervisor

    def launch() -> None:
        _complete_checkpoint(tmp_path, 500)

    published = run_sweep_supervisor(
        run_root=tmp_path,
        target_step=500,
        launch=launch,
        package=lambda checkpoint: checkpoint.optimizer_step,
        publish=lambda step: {
            "optimizer_step": step,
            "readback_verified": True,
            "immutable_revision": "a" * 40,
        },
    )

    assert [item["optimizer_step"] for item in published] == [500]


def test_sweep_records_every_local_500_boundary_before_publication(tmp_path: Path) -> None:
    from lehome_train.groot.continuous_training import run_sweep_supervisor

    local_steps: list[int] = []

    def launch() -> None:
        _complete_checkpoint(tmp_path, 500)
        _complete_checkpoint(tmp_path, 1000)

    run_sweep_supervisor(
        run_root=tmp_path,
        target_step=1000,
        launch=launch,
        record_local=lambda checkpoint: local_steps.append(checkpoint.optimizer_step),
        package=lambda checkpoint: checkpoint.optimizer_step,
        publish=lambda step: {
            "optimizer_step": step,
            "readback_verified": True,
            "immutable_revision": "a" * 40,
        },
    )

    assert local_steps == [500, 1000]


def test_sweep_preemption_returns_the_last_verified_500_step_checkpoint(
    tmp_path: Path,
) -> None:
    from lehome_train.groot.continuous_training import (
        PreemptionController,
        run_sweep_supervisor,
    )

    preemption = PreemptionController()

    def launch() -> None:
        _complete_checkpoint(tmp_path, 500)
        preemption.request()

    published = run_sweep_supervisor(
        run_root=tmp_path,
        target_step=1000,
        launch=launch,
        package=lambda checkpoint: checkpoint.optimizer_step,
        publish=lambda step: {
            "optimizer_step": step,
            "readback_verified": True,
            "immutable_revision": "a" * 40,
        },
        preemption=preemption,
    )

    assert [item["optimizer_step"] for item in published] == [500]
    assert preemption.finalized_step == 500


def test_sweep_publication_attestation_accepts_the_500_terminal_rung(
    tmp_path: Path, monkeypatch,
) -> None:
    from lehome_train.groot.runtime_checkpoint_lifecycle import (
        RuntimeMixtureTrainingIdentity,
        attest_sweep_runtime_checkpoint_publication,
    )
    import lehome_train.groot.runtime_checkpoint_lifecycle as lifecycle

    identity = RuntimeMixtureTrainingIdentity(
        mixture_id="a" * 64,
        deployment_receipt_sha256="b" * 64,
        source_revisions=(
            ("organizer", "c" * 40, "bc/full", "d" * 64),
            ("rollout", "e" * 40, "rollouts/round-1", "f" * 64),
        ),
        schedule_seed=1,
        code_bundle_sha256="1" * 64,
        code_bundle_revision="2" * 40,
        oci_image="sha256:" + "3" * 64,
        parent_step12000_artifact_sha256="4" * 64,
    )
    monkeypatch.setattr(lifecycle, "_verify_publication_readback", lambda **_kwargs: None)
    publication = attest_sweep_runtime_checkpoint_publication(
        raw_publication={
            "optimizer_step": 500,
            "repository": "ryanjin333/lehome-groot-n17-models",
            "immutable_revision": "5" * 40,
            "remote_prefix": "experiments/a-500",
            "relative_path": "checkpoints/step-500.tar",
            "artifact_sha256": "6" * 64,
            "artifact_byte_size": 1,
            "descriptor_relative_path": "checkpoints/step-500.json",
            "descriptor_sha256": "7" * 64,
            "descriptor_byte_size": 1,
            "readback_verified": True,
        },
        identity=identity,
        hub=object(),
        destination=tmp_path / "readback",
    )

    assert publication["optimizer_step"] == 500
    assert publication["fresh_tree_readback_verified"] is True


def test_production_sweep_branch_reaches_isolated_supervisor_not_legacy_2k(
    tmp_path: Path, monkeypatch,
) -> None:
    """A CPU fake proves the dynamic sweep path invokes its own supervisor."""
    import lehome_train.groot.production_runtime as production
    from lehome_train.groot.config import FineTuneLaunchConfig
    from lehome_train.groot.experiment_manifest import SweepRuntimeProfile
    from lehome_train.models import ExperimentConfig
    from lehome_train.constants import MODEL_REVISION

    profile = SweepRuntimeProfile(
        weights={"bc": 100, "rollout": 0, "dagger": 0},
        quotas={"bc": 64, "rollout": 0, "dagger": 0},
        target_step=500, save_steps=500, terminal_publish=True,
        action_horizon=16, global_batch_size=64,
    )
    config = FineTuneLaunchConfig(
        base_model_path="/cache/parent", base_model_revision=MODEL_REVISION,
        dataset_path="/prepared/runtime", dataset_revision="3" * 40,
        modality_config_path="/prepared/modality.py", output_dir="/output/sweep/" + "a" * 64,
        experiment_name="a" * 64, physical_batch_size=64, global_batch_size=64,
        max_steps=500, save_steps=500, warmup_ratio=.05,
        parent_checkpoint_repository="owner/models", parent_checkpoint_revision="4" * 40,
        parent_checkpoint_subpath="policies/step-12000", parent_checkpoint_artifact_sha256="5" * 64,
        runtime_mixture_manifest="/prepared/runtime/mixture.json",
        runtime_window_index="/prepared/runtime/windows.json",
        runtime_mounts_descriptor="/prepared/runtime/mounts.json",
        runtime_sweep_profile=profile,
    )
    experiment = ExperimentConfig(
        repository_commit="6" * 40, container_digest="sha256:" + "7" * 64,
        model_repository="nvidia/GR00T-N1.7-3B", model_revision=MODEL_REVISION,
        dataset_repository="owner/data", dataset_revision="8" * 40,
        dataset_manifest_sha256="9" * 64, physical_batch_size=64,
        gradient_accumulation_steps=1, sample_presentations=500 * 64,
        action_horizon=16, tune_language_backbone=False, tune_visual_backbone=False,
    )
    paths = {name: tmp_path / f"{name}.json" for name in (
        "runtime_manifest", "runtime_window_index", "runtime_normalization",
        "runtime_mounts_descriptor", "runtime_source_evidence", "warmup_receipt",
        "runtime_warmup_binding",
    )}
    for path in paths.values(): path.write_text("{}", encoding="utf-8")
    outputs = {"result_output": tmp_path / "result.json", "status_output": tmp_path / "status.json"}
    token = tmp_path / "token"; token.write_text("token", encoding="utf-8"); token.chmod(0o600)
    calls: list[int] = []
    launch_resumes: list[tuple[int, Path | None]] = []
    monkeypatch.setattr(production, "_runtime_sweep_campaign", lambda *_args: None)
    monkeypatch.setattr(production, "_load_nonempty_json_artifact", lambda *_args: {"binding": True})
    monkeypatch.setattr(production, "bind_warmup_to_runtime_artifacts", lambda **_kwargs: {}) if hasattr(production, "bind_warmup_to_runtime_artifacts") else None
    import lehome_train.groot.runtime_mixture_warmup as warmup
    monkeypatch.setattr(warmup, "bind_warmup_to_runtime_artifacts", lambda **_kwargs: {})
    monkeypatch.setattr(warmup, "validate_gpu_warmup_receipt", lambda *_args, **_kwargs: 4)
    class Session:
        def __init__(self, **_kwargs): pass
        def package_checkpoint_snapshot(self, _root, *, optimizer_step, **_kwargs): return optimizer_step
    class Uploader:
        def __init__(self, **_kwargs): pass
        def publish_receipt(self, step, **_kwargs):
            return {"optimizer_step": step, "repository": "ryanjin333/lehome-groot-n17-models", "immutable_revision": "a" * 40, "remote_prefix": "experiments/a", "relative_path": "checkpoints/step-500.tar", "artifact_sha256": "b" * 64, "artifact_byte_size": 1, "descriptor_relative_path": "checkpoints/step-500.json", "descriptor_sha256": "c" * 64, "descriptor_byte_size": 1, "readback_verified": True}
    monkeypatch.setattr(production, "GrootTrainingSession", Session)
    monkeypatch.setattr(production, "HubCheckpointUploader", Uploader)
    def supervisor(**kwargs):
        calls.append(kwargs["target_step"])
        kwargs["launch"]()
        return (kwargs["publish"](kwargs["target_step"]),)
    monkeypatch.setattr(production, "run_sweep_supervisor", supervisor)
    monkeypatch.setattr(production, "_publisher_token", lambda _path: "token")
    # The resume selector has dedicated full-state coverage below; this test
    # only proves the isolated sweep branch reaches its supervisor.
    monkeypatch.setattr(
        production,
        "_select_sweep_resume_checkpoint",
        lambda *, config, binding: None if config.max_steps == 500 else tmp_path / "checkpoint-500",
    )
    monkeypatch.setattr(
        production,
        "launch_sweep_finetune",
        lambda config, **kwargs: launch_resumes.append((config.max_steps, kwargs.get("resume_checkpoint"))),
    )
    runtime = production.ProductionRuntime(checkpoint_transport_factory=lambda **_kwargs: object())
    result = runtime._runtime_sweep_train(
        request={"checkpoint_repository": "ryanjin333/lehome-groot-n17-models", "checkpoint_revision": "main", "publisher_token_file": str(token), "instance_id": 1},
        outputs=outputs, config=config, experiment=experiment, paths=paths,
        contract=SimpleNamespace(), sweep_binding={"parent_publication": None},
    )
    assert calls == [500]
    assert launch_resumes == [(500, None)]
    assert result["immutable_checkpoint_publications"][0]["optimizer_step"] == 500

    # The same isolated supervisor accepts a promoted continuation rung; it
    # does not silently reuse the legacy 2K-only supervisor.
    promoted_profile = replace(profile, target_step=1000)
    promoted_config = replace(
        config, experiment_name="b" * 64, output_dir="/output/sweep/" + "b" * 64,
        base_model_path="/cache/promoted-parent/checkpoint", max_steps=1000,
        runtime_sweep_profile=promoted_profile,
    )
    promoted_experiment = replace(experiment, sample_presentations=1000 * 64)
    promoted_publication = {
        "schema_version": 2, "experiment_id": "a" * 64, "job_digest": "a" * 64,
        "target_step": 500, "repository": "owner/models", "immutable_revision": "a" * 40,
        "remote_prefix": "experiments/a-500", "relative_path": "checkpoints/step-500.tar",
        "artifact_sha256": "b" * 64, "artifact_byte_size": 1,
        "descriptor_relative_path": "checkpoints/step-500.json", "descriptor_sha256": "c" * 64,
        "descriptor_byte_size": 1, "receipt_sha256": "d" * 64, "readback_verified": True,
    }
    result = runtime._runtime_sweep_train(
        request={"checkpoint_repository": "ryanjin333/lehome-groot-n17-models", "checkpoint_revision": "main", "publisher_token_file": str(token), "instance_id": 2},
        outputs=outputs, config=promoted_config, experiment=promoted_experiment,
        paths=paths, contract=SimpleNamespace(), sweep_binding={"parent_publication": promoted_publication},
    )
    assert calls == [500, 1000]
    assert launch_resumes[-1] == (1000, tmp_path / "checkpoint-500")
    assert result["target_step"] == 1000
