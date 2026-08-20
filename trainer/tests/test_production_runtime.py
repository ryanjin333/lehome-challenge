from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import lehome_train.groot.production_adapters as adapters
import lehome_train.groot.production_runtime as runtime_module
from lehome_train.commands.sync import generate_sync_manifest
from lehome_train.constants import MODEL_REVISION
from lehome_train.data.normalization import normalization_identity
from lehome_train.checkpoints import CheckpointDescriptor, write_checkpoint_descriptor
from lehome_train.groot.config import FineTuneLaunchConfig
from lehome_train.io import canonical_json_sha256, sha256_file
from lehome_train.models import ArtifactIdentity, CheckpointRecord, ExperimentConfig, SmokeResult


@pytest.fixture(autouse=True)
def _complete_official_checkpoint_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep production recovery fixtures in the official full-state layout."""

    original_mkdir = Path.mkdir

    def mkdir(path: Path, *args: object, **kwargs: object) -> None:
        original_mkdir(path, *args, **kwargs)
        suffix = path.name.removeprefix("checkpoint-")
        if suffix.isdecimal():
            for name in ("model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
                artifact = path / name
                if not artifact.exists():
                    artifact.write_bytes(name.encode("ascii"))

    monkeypatch.setattr(Path, "mkdir", mkdir)


COMMIT = "a" * 40
DATASET_REVISION = "b" * 40
DATASET_SHA256 = "c" * 64
NORMALIZATION_SHA256 = "d" * 64
IMAGE_DIGEST = "sha256:" + "e" * 64


def test_runtime_checkpoint_source_evidence_binds_the_exact_awr_transform() -> None:
    from lehome_train.groot.awr_weighting import AwrReplayConfig
    from lehome_train.groot.runtime_checkpoint_lifecycle import (
        RuntimeMixtureTrainingIdentity,
    )

    awr_config = AwrReplayConfig(temperature=0.75, minimum=0.5, maximum=3.0)
    identity = RuntimeMixtureTrainingIdentity(
        mixture_id="a" * 64,
        deployment_receipt_sha256="b" * 64,
        source_revisions=(
            ("organizer", "c" * 40, "bc/full", "d" * 64),
            ("rollout", "e" * 40, "rollouts/round-1", "f" * 64),
        ),
        schedule_seed=17,
        code_bundle_sha256="1" * 64,
        code_bundle_revision="2" * 40,
        oci_image="sha256:" + "3" * 64,
        parent_step12000_artifact_sha256="4" * 64,
        awr_evidence_sha256="8" * 64,
        awr_config_sha256=awr_config.sha256,
    )
    parsed = runtime_module._runtime_checkpoint_identity_from_evidence(
        identity.to_dict()
    )
    config = SimpleNamespace(
        runtime_awr_evidence_sha256="8" * 64,
        runtime_awr_temperature=0.75,
        runtime_awr_minimum=0.5,
        runtime_awr_maximum=3.0,
    )
    assert parsed.awr_evidence_sha256 == "8" * 64
    runtime_module._validate_runtime_awr_training_identity(
        config=config, identity=parsed
    )
    with pytest.raises(ValueError, match="AWR.*identity"):
        runtime_module._validate_runtime_awr_training_identity(
            config=SimpleNamespace(
                runtime_awr_evidence_sha256="8" * 64,
                runtime_awr_temperature=0.75,
                runtime_awr_minimum=0.5,
                runtime_awr_maximum=4.0,
            ),
            identity=parsed,
        )


def test_runtime_awr_evidence_is_loaded_from_the_protected_mount_and_bound_to_manifest(
    tmp_path: Path,
) -> None:
    evidence = {
        "schema_version": 1,
        "kind": "lehome_awr_progress_evidence",
        "mixture_id": "a" * 64,
        "mixture_manifest_sha256": canonical_json_sha256({"manifest": "bound"}),
        "episodes": [
            {
                "episode_id": "attempt-1",
                "lineage_id": "lineage-1",
                "split": "train",
                "score_kind": "progress",
                "score": 1.0,
                "provenance_path": "receipts/attempt-1.json",
                "provenance_sha256": "b" * 64,
            }
        ],
    }
    evidence_path = tmp_path / "prepared" / "awr.json"
    _write(evidence_path, evidence)
    config = SimpleNamespace(
        runtime_awr_evidence_path=str(evidence_path),
        runtime_awr_evidence_sha256=canonical_json_sha256(evidence),
        runtime_awr_temperature=0.75,
        runtime_awr_minimum=0.5,
        runtime_awr_maximum=3.0,
    )
    contract = SimpleNamespace(
        manifest=SimpleNamespace(mixture_id="a" * 64, raw={"manifest": "bound"})
    )

    binding = runtime_module._load_runtime_awr_binding(
        config=config, contract=contract
    )

    assert binding is not None
    assert binding.checkpoint_identity() == {
        "awr_evidence_sha256": canonical_json_sha256(evidence),
        "awr_config_sha256": __import__(
            "lehome_train.groot.awr_weighting", fromlist=["AwrReplayConfig"]
        ).AwrReplayConfig(temperature=0.75, minimum=0.5, maximum=3.0).sha256,
    }


def test_runtime_warmup_dataset_factory_uses_the_same_bound_awr_objects_as_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warm-up must exercise the exact replay transform that training will use."""
    evidence = {
        "schema_version": 1,
        "kind": "lehome_awr_progress_evidence",
        "mixture_id": "a" * 64,
        "mixture_manifest_sha256": canonical_json_sha256({"manifest": "bound"}),
        "episodes": [{
            "episode_id": "attempt-1", "lineage_id": "lineage-1", "split": "train",
            "score_kind": "progress", "score": 1.0,
            "provenance_path": "receipts/attempt-1.json", "provenance_sha256": "b" * 64,
        }],
    }
    evidence_path = tmp_path / "prepared" / "awr.json"
    _write(evidence_path, evidence)
    config = SimpleNamespace(
        runtime_awr_evidence_path=str(evidence_path),
        runtime_awr_evidence_sha256=canonical_json_sha256(evidence),
        runtime_awr_temperature=0.75, runtime_awr_minimum=0.5, runtime_awr_maximum=3.0,
    )
    contract = SimpleNamespace(
        manifest=SimpleNamespace(mixture_id="a" * 64, raw={"manifest": "bound"})
    )
    captured: dict[str, object] = {}

    def capture_factory(**kwargs: object) -> type[object]:
        captured.update(kwargs)
        return object

    import lehome_train.groot.runtime_mixture as runtime_mixture

    monkeypatch.setattr(runtime_mixture, "runtime_dataset_factory_class", capture_factory)
    result = runtime_module._runtime_warmup_dataset_factory(
        config=config, contract=contract,
        manifest=tmp_path / "prepared" / "mixture.json",
        window_index=tmp_path / "prepared" / "windows.json",
        mounts=tmp_path / "prepared" / "mounts.json",
    )

    assert result is object
    assert captured["awr_evidence"].identity_sha256 == canonical_json_sha256(evidence)
    assert captured["awr_config"].sha256 == (
        __import__("lehome_train.groot.awr_weighting", fromlist=["AwrReplayConfig"])
        .AwrReplayConfig(temperature=0.75, minimum=0.5, maximum=3.0).sha256
    )


def test_runtime_warmup_dataset_factory_keeps_disabled_awr_factory_arguments_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(runtime_awr_evidence_path=None)
    contract = SimpleNamespace(manifest=SimpleNamespace(mixture_id="a" * 64, raw={}))
    captured: dict[str, object] = {}

    def capture_factory(**kwargs: object) -> type[object]:
        captured.update(kwargs)
        return object

    import lehome_train.groot.runtime_mixture as runtime_mixture

    monkeypatch.setattr(runtime_mixture, "runtime_dataset_factory_class", capture_factory)
    assert runtime_module._runtime_warmup_dataset_factory(
        config=config, contract=contract,
        manifest=tmp_path / "prepared" / "mixture.json",
        window_index=tmp_path / "prepared" / "windows.json",
        mounts=tmp_path / "prepared" / "mounts.json",
    ) is object
    assert "awr_evidence" not in captured and "awr_config" not in captured


@pytest.fixture(autouse=True)
def mounted_test_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runtime_module,
        "_ALLOWED_ROOTS",
        (tmp_path / "prepared", tmp_path / "output", tmp_path / "cache"),
    )


def _write(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return str(path)


def _launch_payload(tmp_path: Path, *, batch: int, max_steps: int) -> dict[str, object]:
    dataset = tmp_path / "prepared" / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    (dataset / "manifest.json").write_text(
        json.dumps(
            {
                "output_format": "groot_lerobot_v2.1_per_episode",
                "train_episode_ids": ["0"],
                "validation_episode_ids": ["1"],
            }
        ),
        encoding="utf-8",
    )
    model = tmp_path / "cache" / "model"
    model.mkdir(parents=True, exist_ok=True)
    modality = dataset / "meta" / "modality.py"
    modality.parent.mkdir(parents=True, exist_ok=True)
    modality.write_text("# checked\n", encoding="utf-8")
    normalization_files = {
        "meta/lehome_groot_modality.py": "# canonical modality\n",
        "meta/relative_stats.json": '{"relative":true}\n',
        "meta/stats.json": '{"stats":true}\n',
        "meta/validation_report.json": '{"valid":true}\n',
    }
    for relative_path, content in normalization_files.items():
        path = dataset / relative_path
        path.write_text(content, encoding="utf-8")
    (dataset / "meta" / "prepared_hashes.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifacts": {
                    relative_path: sha256_file(dataset / relative_path)
                    for relative_path in normalization_files
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output" / "experiment"
    output.mkdir(parents=True, exist_ok=True)
    return {
        "base_model_path": str(model),
        "base_model_revision": MODEL_REVISION,
        "dataset_path": str(dataset),
        "dataset_revision": DATASET_REVISION,
        "modality_config_path": str(modality),
        "output_dir": str(output),
        "experiment_name": "experiment-001",
        "physical_batch_size": batch,
        "max_steps": max_steps,
        "save_steps": 1000,
        "warmup_ratio": 0.05,
    }


def _experiment_payload(*, batch: int) -> dict[str, object]:
    return ExperimentConfig(
        repository_commit=COMMIT,
        container_digest=IMAGE_DIGEST,
        model_repository="nvidia/GR00T-N1.7-3B",
        model_revision=MODEL_REVISION,
        dataset_repository="ryanjin333/lehome-groot-n17-data",
        dataset_revision=DATASET_REVISION,
        dataset_manifest_sha256=DATASET_SHA256,
        physical_batch_size=batch,
        gradient_accumulation_steps=1,
        sample_presentations=768_000,
        action_horizon=16,
        tune_language_backbone=False,
        tune_visual_backbone=False,
    ).to_dict()


def test_memorize_composes_fixed_controller_before_writing_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch_path = _write(
        tmp_path / "prepared" / "memorize-launch.json",
        _launch_payload(tmp_path, batch=1, max_steps=10_000),
    )
    experiment_path = _write(
        tmp_path / "prepared" / "experiment.json",
        _experiment_payload(batch=1),
    )
    result_path = tmp_path / "output" / "reports" / "memorize.json"
    status_path = tmp_path / "output" / "status" / "memorize.json"
    observed: dict[str, object] = {}

    def fake_controller(**kwargs: object) -> SimpleNamespace:
        assert status_path.parent.is_dir()
        assert result_path.parent.is_dir()
        observed.update(kwargs)
        return SimpleNamespace(to_dict=lambda: {"status": "controller-result"})

    monkeypatch.setattr(runtime_module, "run_memorization", fake_controller)
    adapter = runtime_module.ProductionRuntime()
    returned = adapter.memorize(
        {
            "launch_config": launch_path,
            "experiment_config": experiment_path,
            "dataset_manifest_sha256": DATASET_SHA256,
            "requested_episode_id": None,
            "result_output": str(result_path),
            "status_output": str(status_path),
        }
    )

    assert observed["dataset_manifest_sha256"] == DATASET_SHA256
    assert observed["experiment_config_sha256"] == canonical_json_sha256(
        ExperimentConfig(**_experiment_payload(batch=1))
    )
    assert callable(observed["trainer"])
    assert callable(observed["evaluator"])
    assert callable(observed["checkpointer"])
    assert json.loads(result_path.read_text(encoding="utf-8")) == {
        "status": "controller-result"
    }
    assert returned["status"] == "controller-result"


def test_prepare_composes_restart_safe_preflight_before_model_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch_path = _write(
        tmp_path / "prepared" / "prepare-launch.json",
        _launch_payload(tmp_path, batch=64, max_steps=12_000),
    )
    experiment_path = _write(
        tmp_path / "prepared" / "experiment.json",
        _experiment_payload(batch=64),
    )
    model_manifest = _write(
        tmp_path / "cache" / "model" / "lehome_model_snapshot.json",
        {"revision": MODEL_REVISION, "artifacts": [{"fixture": True}]},
    )
    dataset_manifest = _write(
        tmp_path / "prepared" / "dataset.snapshot.json",
        {"revision": DATASET_REVISION, "artifacts": [{"fixture": True}]},
    )
    network = _write(
        tmp_path / "prepared" / "network.json",
        {
            "schema_version": 1,
            "downloaded_bytes": 1_000_000_000,
            "duration_seconds": 1.0,
        },
    )
    status = tmp_path / "output" / "status" / "prepare.json"
    observed: dict[str, object] = {}

    experiment_root = tmp_path / "output" / "preflight" / "exp"

    def fake_prepare(**kwargs: object) -> SimpleNamespace:
        observed.update(kwargs)
        assert tuple(kwargs["stage_operations"]) == runtime_module.PREFLIGHT_STAGE_NAMES
        experiment_root.mkdir(parents=True)
        return SimpleNamespace(
            experiment=SimpleNamespace(experiment_id="exp", root=experiment_root),
            hardware=SimpleNamespace(
                visible_device="0",
                vram_bytes=96 * 1024**3,
                writable_free_bytes=300 * 1024**3,
            ),
            completed_stages=runtime_module.PREFLIGHT_STAGE_NAMES,
        )

    monkeypatch.setattr(runtime_module, "prepare_training_environment", fake_prepare)
    monkeypatch.setattr(runtime_module, "_visible_device", lambda *_args: "0")
    monkeypatch.setattr(runtime_module, "probe_physical_vram_bytes", lambda: 96 * 1024**3)
    monkeypatch.setattr(
        runtime_module,
        "HuggingFaceHubTransport",
        lambda **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setenv("HF_TOKEN", "hf_explicit_prepare_parent_token")

    result = runtime_module.ProductionRuntime().prepare(
        {
            "launch_config": launch_path,
            "experiment_config": experiment_path,
            "model_snapshot_manifest": model_manifest,
            "dataset_snapshot_manifest": dataset_manifest,
            "network_measurement": network,
            "artifact_repository": "ryanjin333/lehome-groot-n17-models",
            "artifact_revision": "f" * 40,
            "status_output": str(status),
        }
    )

    assert observed["token"] == "hf_explicit_prepare_parent_token"
    assert len(observed["hub_targets"]) == 2
    assert result["completed_stages"] == list(runtime_module.PREFLIGHT_STAGE_NAMES)
    assert result["normalization_sha256"] == normalization_identity(
        tmp_path / "prepared" / "dataset"
    )
    assert observed["resolved_config"]["normalization_sha256"] == result[
        "normalization_sha256"
    ]
    sync_root = tmp_path / "output" / "experiment"
    assert json.loads((sync_root / "resolved-config.json").read_text()) == _experiment_payload(
        batch=64
    )
    provenance = json.loads((sync_root / "provenance.json").read_text())
    assert provenance["experiment_id"] == "experiment-001"
    assert provenance["preflight_experiment_id"] == "exp"
    assert provenance["repository_commit"] == COMMIT
    assert provenance["container_digest"] == IMAGE_DIGEST
    assert provenance["normalization_sha256"] == result["normalization_sha256"]
    assert provenance["model_snapshot_manifest_sha256"] == sha256_file(model_manifest)
    assert provenance["dataset_snapshot_manifest_sha256"] == sha256_file(dataset_manifest)
    assert json.loads((sync_root / "logs" / "prepare.json").read_text()) == {
        "schema_version": 1,
        "event": "prepared",
        "experiment_id": "experiment-001",
        "preflight_experiment_id": "exp",
        "normalization_sha256": result["normalization_sha256"],
    }


def test_smoke_uses_nvml_probe_and_canonical_controller_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch_path = _write(
        tmp_path / "prepared" / "smoke-launch.json",
        _launch_payload(tmp_path, batch=1, max_steps=100),
    )
    experiment = ExperimentConfig(**_experiment_payload(batch=64))
    experiment_path = _write(
        tmp_path / "prepared" / "experiment.json", experiment.to_dict()
    )
    report_path = tmp_path / "output" / "reports" / "smoke.json"
    selected_path = tmp_path / "output" / "reports" / "selected-smoke.json"
    status_path = tmp_path / "output" / "status" / "smoke.json"
    selected = SmokeResult(
        experiment_id="experiment-001",
        experiment_config_sha256=canonical_json_sha256(experiment),
        dataset_manifest_sha256=DATASET_SHA256,
        physical_batch_size=64,
        gradient_accumulation_steps=1,
        optimizer_steps=100,
        stable=True,
        finite_loss=True,
        physical_vram_bytes=96 * 1024**3,
        peak_reserved_vram_bytes=70 * 1024**3,
        minimum_steady_state_free_vram_bytes=20 * 1024**3,
        steady_steps_per_second=1.0,
        samples_per_second=64.0,
        failure_reason=None,
    )
    observed: dict[str, object] = {}

    class Report:
        selected_batch_size = 64
        attempts = (SimpleNamespace(result=selected),)

        def to_dict(self) -> dict[str, object]:
            return {"selected_batch_size": 64, "attempts": [selected.to_dict()]}

    def fake_controller(**kwargs: object) -> Report:
        assert status_path.parent.is_dir()
        observed.update(kwargs)
        return Report()

    monkeypatch.setattr(runtime_module, "run_smoke_tests", fake_controller)
    monkeypatch.setattr(runtime_module, "probe_physical_vram_bytes", lambda: 96 * 1024**3)
    runtime_module.ProductionRuntime().smoke(
        {
            "launch_config": launch_path,
            "experiment_config": experiment_path,
            "report_output": str(report_path),
            "selected_result_output": str(selected_path),
            "status_output": str(status_path),
        }
    )
    assert observed["physical_vram_bytes"] == 96 * 1024**3
    assert callable(observed["runner"])
    assert callable(observed["sampler_factory"])
    assert json.loads(selected_path.read_text(encoding="utf-8"))["physical_batch_size"] == 64


def test_tune_uses_only_fixed_batch64_loader_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch_path = _write(tmp_path / "prepared" / "tune-launch.json", _launch_payload(tmp_path, batch=64, max_steps=2000))
    experiment_path = _write(tmp_path / "prepared" / "experiment.json", _experiment_payload(batch=64))
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(runtime_module, "_visible_device", lambda *_args: "0")
    monkeypatch.setattr(runtime_module, "probe_physical_vram_bytes", lambda: 96 * 1024**3)
    monkeypatch.setattr(runtime_module, "GrootSmokeRunner", lambda: lambda config: calls.append((config.dataloader_num_workers, config.physical_batch_size)) or runtime_module.SmokeAttemptReceipt(100, 1, True, 0, 0, 1, 100, (), None))
    result = runtime_module.ProductionRuntime().tune({
        "launch_config": launch_path, "experiment_config": experiment_path,
        "report_output": str(tmp_path / "output" / "report.json"),
        "status_output": str(tmp_path / "output" / "status.json"),
    })
    assert result["production_physical_batch"] == 64
    assert [workers for workers, batch in calls] == [0, 4, 8, 12, 16]
    assert all(batch == 64 for _workers, batch in calls)


def test_runtime_mixture_launch_locks_2k_and_local_500_step_cadence(tmp_path: Path) -> None:
    launch = _launch_payload(tmp_path, batch=64, max_steps=2000)
    launch.update({
        "runtime_mixture_manifest": "/prepared/mixture.json",
        "runtime_window_index": "/prepared/windows.json",
        "runtime_mounts_descriptor": "/prepared/mounts.json",
        "save_steps": 1000,
    })
    with pytest.raises(ValueError, match="local checkpoint cadence"):
        FineTuneLaunchConfig(**launch)


def test_runtime_mixture_train_requires_a_cross_checked_gpu_warmup_receipt() -> None:
    with pytest.raises(ValueError, match="runtime-mixture-train"):
        runtime_module.ProductionRuntime().runtime_mixture_train({
            "launch_config": "/prepared/launch.json",
            "experiment_config": "/prepared/experiment.json",
            "runtime_manifest": "/prepared/mixture.json",
            "runtime_window_index": "/prepared/windows.json",
            "runtime_normalization": "/prepared/normalization.json",
            "runtime_mounts_descriptor": "/prepared/mounts.json",
            "runtime_source_evidence": "/prepared/sources.json",
            "result_output": "/output/result.json",
            "status_output": "/output/status.json",
        })


def test_runtime_mixture_train_envelope_rejects_cpu_pilot_receipt() -> None:
    arguments = {
        "launch_config": "/prepared/launch.json", "experiment_config": "/prepared/experiment.json",
        "runtime_manifest": "/prepared/mixture.json", "runtime_window_index": "/prepared/windows.json",
        "runtime_normalization": "/prepared/normalization.json", "runtime_mounts_descriptor": "/prepared/mounts.json",
        "runtime_source_evidence": "/prepared/sources.json", "warmup_receipt": "/prepared/warmup.json",
        "runtime_warmup_binding": "/prepared/binding.json", "runtime_resume_archive": None,
        "runtime_resume_descriptor": None, "runtime_resume_cursor": None, "runtime_resume_anchor": None,
        "runtime_resume_publication": None, "checkpoint_repository": "repo", "checkpoint_revision": "main",
        "publisher_token_file": "/prepared/token", "instance_id": 44,
        "result_output": "/output/result.json", "status_output": "/output/status.json",
        "cpu_pilot_receipt": "/prepared/cpu-pilot.json",
    }

    with pytest.raises(ValueError, match="incompatible schema"):
        runtime_module.ProductionRuntime().runtime_mixture_train(arguments)


def test_runtime_recovery_selection_prefers_local_1500_then_hf_then_parent(tmp_path: Path) -> None:
    from lehome_train.groot.local_recovery import attest_local_checkpoint

    identity = {
        "experiment_manifest_sha256": "a" * 64,
        "parent_checkpoint_artifact_sha256": "b" * 64,
        "runtime_mixture_id": "c" * 64,
        "trainer_code_sha256": "d" * 64,
        "trainer_code_revision": "e" * 40,
    }
    official = tmp_path / "output" / "run" / "checkpoint-1500"
    official.mkdir(parents=True)
    (official / "weights.bin").write_bytes(b"weights")
    (official / "trainer_state.json").write_text(json.dumps({"global_step": 1500, "log_history": [{"step": 1500, "loss": .25}]}))
    local = attest_local_checkpoint(
        checkpoint=official, metadata_root=tmp_path / "output" / "local-recovery", optimizer_step=1500,
        identity=identity,
    )
    hf = SimpleNamespace(record=SimpleNamespace(optimizer_step=1000))

    selected = runtime_module._select_runtime_recovery(local=local, hf_checkpoint=tmp_path / "hf-1000", hf_descriptor=hf, parent_checkpoint=tmp_path / "parent")
    assert selected.path == official and selected.optimizer_step == 1500 and selected.source == "local"
    selected = runtime_module._select_runtime_recovery(local=None, hf_checkpoint=tmp_path / "hf-1000", hf_descriptor=hf, parent_checkpoint=tmp_path / "parent")
    assert selected.path == tmp_path / "hf-1000" and selected.source == "hf"
    hf_2000 = SimpleNamespace(record=SimpleNamespace(optimizer_step=2000))
    selected = runtime_module._select_runtime_recovery(local=None, hf_checkpoint=tmp_path / "hf-2000", hf_descriptor=hf_2000, parent_checkpoint=tmp_path / "parent")
    assert selected.source == "terminal" and selected.optimizer_step == 2000
    selected = runtime_module._select_runtime_recovery(local=None, hf_checkpoint=None, hf_descriptor=None, parent_checkpoint=tmp_path / "parent")
    assert selected.path == tmp_path / "parent" and selected.source == "parent"


def test_runtime_recovery_uses_authenticated_hf_2k_terminal_when_local_2k_lacks_its_journal(tmp_path: Path) -> None:
    from lehome_train.groot.local_recovery import attest_local_checkpoint

    identity = {
        "experiment_manifest_sha256": "a" * 64,
        "parent_checkpoint_artifact_sha256": "b" * 64,
        "runtime_mixture_id": "c" * 64,
        "trainer_code_sha256": "d" * 64,
        "trainer_code_revision": "e" * 40,
    }
    checkpoint = tmp_path / "output" / "run" / "checkpoint-2000"
    checkpoint.mkdir(parents=True)
    (checkpoint / "weights.bin").write_bytes(b"weights")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 2000, "log_history": [{"step": 2000, "loss": .25}]}),
    )
    local = attest_local_checkpoint(
        checkpoint=checkpoint, metadata_root=tmp_path / "output" / "local-recovery",
        optimizer_step=2000, identity=identity,
    )
    hf = SimpleNamespace(record=SimpleNamespace(optimizer_step=2000))

    selected = runtime_module._select_runtime_recovery(
        local=local, hf_checkpoint=tmp_path / "hf-2000", hf_descriptor=hf,
        parent_checkpoint=tmp_path / "parent",
    )

    assert selected.source == "terminal"
    assert selected.local is None
    assert selected.path == tmp_path / "hf-2000"


def test_runtime_authenticated_hf_two_k_bypasses_an_unmarked_local_two_k_journal(tmp_path: Path) -> None:
    from lehome_train.groot.local_recovery import (
        attest_local_checkpoint,
        discover_local_recovery,
        record_immutable_publication,
    )

    identity = {
        "experiment_manifest_sha256": "a" * 64,
        "parent_checkpoint_artifact_sha256": "b" * 64,
        "runtime_mixture_id": "c" * 64,
        "trainer_code_sha256": "d" * 64,
        "trainer_code_revision": "e" * 40,
    }
    checkpoint = tmp_path / "output" / "run" / "checkpoint-2000"
    checkpoint.mkdir(parents=True)
    (checkpoint / "weights.bin").write_bytes(b"weights")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 2000, "log_history": [{"step": 2000, "loss": .25}]}),
    )
    metadata = tmp_path / "output" / "local-recovery"
    local = attest_local_checkpoint(
        checkpoint=checkpoint, metadata_root=metadata, optimizer_step=2000, identity=identity,
    )
    record_immutable_publication(
        metadata_root=metadata, checkpoint=local,
        publication={"optimizer_step": 2000, "readback_verified": True, "immutable_revision": "1" * 40},
        anchor={"immutable_anchor_revision": "1" * 40, "anchor_sha256": "2" * 64},
    )
    (metadata / "publication-2000.COMPLETE").unlink()
    recovered = discover_local_recovery(metadata_root=metadata, identity=identity)
    assert recovered is not None and recovered.terminal_immutable_publication is None

    selected = runtime_module._select_runtime_recovery(
        local=recovered, hf_checkpoint=tmp_path / "authenticated-hf-2000",
        hf_descriptor=SimpleNamespace(record=SimpleNamespace(optimizer_step=2000)),
        parent_checkpoint=tmp_path / "parent",
    )
    assert selected.source == "terminal"
    assert selected.local is None


@pytest.mark.parametrize("tamper", (None, "publication", "anchor", "float-cursor", "readback", "artifact", "descriptor", "revision", "size", "manifest"))
def test_runtime_mixture_hf_2k_terminal_restores_then_requires_bound_publication_and_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str | None,
) -> None:
    """The actual runtime may terminal at 2K only after cursor and chain validation."""
    prepared, output, cache = tmp_path / "prepared", tmp_path / "output", tmp_path / "cache"
    for root in (prepared, output, cache): root.mkdir()
    runtime_paths = {
        key: prepared / f"{key}.json"
        for key in ("manifest", "windows", "normalization", "mounts", "sources", "warmup", "binding")
    }
    for path in runtime_paths.values(): path.write_text('{"ok":true}', encoding="utf-8")
    token = prepared / "token"; token.write_text("publisher-token", encoding="utf-8"); token.chmod(0o600)
    identity = __import__("lehome_train.groot.runtime_checkpoint_lifecycle", fromlist=["RuntimeMixtureTrainingIdentity"]).RuntimeMixtureTrainingIdentity(
        mixture_id="a" * 64, deployment_receipt_sha256="b" * 64,
        source_revisions=(("organizer", "c" * 40, "bc/full", "d" * 64), ("rollout", "e" * 40, "rollouts/round-1", "f" * 64)),
        schedule_seed=1, code_bundle_sha256="1" * 64, code_bundle_revision="2" * 40,
        oci_image="sha256:" + "3" * 64, parent_step12000_artifact_sha256="4" * 64,
    )
    runtime_paths["sources"].write_text(json.dumps(identity.to_dict()), encoding="utf-8")
    experiment = ExperimentConfig(**_experiment_payload(batch=64))
    experiment_path = _write(prepared / "experiment.json", experiment.to_dict())
    archive = prepared / "resume.tar"; archive.write_bytes(b"authenticated-2k")
    descriptor = CheckpointDescriptor(
        record=CheckpointRecord("runtime", 2000, 128_000, "5" * 64, identity.mixture_id, "6" * 64,
            ArtifactIdentity("checkpoints/step-2000.tar", sha256_file(archive), archive.stat().st_size), True, False),
        normalization_sha256="7" * 64, schedule_sha256="6" * 64, locally_verified=True,
    )
    descriptor_path = prepared / "resume.json"; write_checkpoint_descriptor(descriptor_path, descriptor)
    config = SimpleNamespace(
        runtime_mixture_manifest=str(runtime_paths["manifest"]), runtime_window_index=str(runtime_paths["windows"]),
        runtime_mounts_descriptor=str(runtime_paths["mounts"]), dataset_path=str(prepared / "dataset"),
        base_model_path=str(cache / "parent"), output_dir=str(output), experiment_name="runtime", dataloader_num_workers=4, num_gpus=1,
        identity=lambda: {},
    )
    publication = {
        "schema_version": 1, "kind": "runtime_mixture_checkpoint_publication", "optimizer_step": 2000,
        "repository": runtime_module.DEFAULT_MODEL_REPO, "immutable_revision": "8" * 40,
        "remote_prefix": "runtime/checkpoint", "relative_path": "checkpoints/step-2000.tar",
        "artifact_sha256": sha256_file(archive), "artifact_byte_size": archive.stat().st_size,
        "descriptor_relative_path": "checkpoints/step-2000.json", "descriptor_sha256": sha256_file(descriptor_path),
        "descriptor_byte_size": descriptor_path.stat().st_size, "readback_verified": True,
        "identity": identity.to_dict(), "identity_sha256": identity.sha256,
        "runtime_cursor": {"optimizer_step": 2000, "global_sample_offset": 128_000, "physical_batch_size": 64, "action_horizon": 16},
        "fresh_tree_readback_verified": True,
    }
    anchor = {"immutable_anchor_revision": "8" * 40, "anchor_sha256": "9" * 64}
    if tamper == "publication": publication = None  # type: ignore[assignment]
    if tamper == "anchor": anchor = None  # type: ignore[assignment]
    if tamper == "float-cursor": publication["runtime_cursor"]["physical_batch_size"] = 64.0
    if tamper == "readback": publication["readback_verified"] = False
    if tamper == "artifact": publication["artifact_sha256"] = "0" * 64
    if tamper == "descriptor": publication["descriptor_sha256"] = "0" * 64
    if tamper == "revision": publication["immutable_revision"] = "not-a-revision"
    if tamper == "size": publication["artifact_byte_size"] = float(archive.stat().st_size)
    restored: list[Path] = []
    launched = False
    monkeypatch.setattr(runtime_module, "_load_config", lambda _path: config)
    monkeypatch.setattr(runtime_module, "_load_experiment", lambda _path: experiment)
    monkeypatch.setattr(runtime_module, "_runtime_final_campaign", lambda *_args: None)
    monkeypatch.setattr(runtime_module, "GrootTrainingSession", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(runtime_module, "HubCheckpointUploader", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(runtime_module, "launch_continuous_finetune", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("terminal 2K must not launch gradients")))
    monkeypatch.setattr(
        "lehome_train.groot.runtime_mixture.load_runtime_contract",
        lambda *_args: SimpleNamespace(
            manifest=SimpleNamespace(
                mixture_id=identity.mixture_id, cycle_size=64,
                **({} if tamper == "manifest" else {"experiment_manifest_sha256": "f" * 64}),
            ), training_windows=(),
        ),
    )
    monkeypatch.setattr("lehome_train.groot.runtime_mixture_warmup.bind_warmup_to_runtime_artifacts", lambda **_kwargs: {"binding": True})
    monkeypatch.setattr("lehome_train.groot.runtime_mixture_warmup.validate_gpu_warmup_receipt", lambda *_args, **_kwargs: 4)
    def restore(_archive: Path, **kwargs: object) -> None:
        root = kwargs["output_root"]
        assert isinstance(root, Path)
        restored.append(root)
        checkpoint = root / "runtime" / "checkpoint-2000"
        checkpoint.mkdir(parents=True)
        (checkpoint / "weights.bin").write_bytes(b"authenticated-2k")
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": 2000, "log_history": [{"step": 2000, "loss": .25}]}),
        )

    monkeypatch.setattr("lehome_train.groot.production_adapters._restore_checkpoint_archive", restore)
    request = {
        "launch_config": str(prepared / "launch.json"), "experiment_config": experiment_path,
        "runtime_manifest": str(runtime_paths["manifest"]), "runtime_window_index": str(runtime_paths["windows"]),
        "runtime_normalization": str(runtime_paths["normalization"]), "runtime_mounts_descriptor": str(runtime_paths["mounts"]),
        "runtime_source_evidence": str(runtime_paths["sources"]), "warmup_receipt": str(runtime_paths["warmup"]),
        "runtime_warmup_binding": str(runtime_paths["binding"]), "runtime_resume_archive": str(archive),
        "runtime_resume_descriptor": str(descriptor_path), "runtime_resume_cursor": publication["runtime_cursor"] if publication else {"optimizer_step": 2000, "global_sample_offset": 128_000, "physical_batch_size": 64, "action_horizon": 16},
        "runtime_resume_anchor": anchor, "runtime_resume_publication": publication,
        "local_recovery_root": str(output / "local-recovery"), "checkpoint_repository": runtime_module.DEFAULT_MODEL_REPO,
        "checkpoint_revision": "main", "publisher_token_file": str(token), "instance_id": 7,
        "result_output": str(output / "result.json"), "status_output": str(output / "status.json"),
    }
    if tamper is None:
        result = runtime_module.ProductionRuntime(checkpoint_transport_factory=lambda **_kwargs: object()).runtime_mixture_train(request)
        assert result["status"] == "runtime-mixture-terminal-hf-2000"
        assert result["immutable_checkpoint_publications"] == [publication]
        assert restored == [output]

        # Crash after local 2K attestation but before the terminal local
        # journal: the already-authenticated HF terminal remains sufficient,
        # with neither a new publication nor gradient launch.
        from lehome_train.groot.local_recovery import attest_local_checkpoint, discover_local_recovery
        local = attest_local_checkpoint(
            checkpoint=output / "runtime" / "checkpoint-2000",
            metadata_root=output / "local-recovery", optimizer_step=2000,
            identity={
                "experiment_manifest_sha256": "f" * 64,
                "parent_checkpoint_artifact_sha256": identity.parent_step12000_artifact_sha256,
                "runtime_mixture_id": identity.mixture_id,
                "trainer_code_sha256": identity.code_bundle_sha256,
                "trainer_code_revision": identity.code_bundle_revision,
            },
        )
        assert discover_local_recovery(
            metadata_root=output / "local-recovery", identity={
                "experiment_manifest_sha256": "f" * 64,
                "parent_checkpoint_artifact_sha256": identity.parent_step12000_artifact_sha256,
                "runtime_mixture_id": identity.mixture_id,
                "trainer_code_sha256": identity.code_bundle_sha256,
                "trainer_code_revision": identity.code_bundle_revision,
            },
        ).terminal_immutable_publication is None
        monkeypatch.setattr(
            runtime_module, "run_continuous_supervisor",
            lambda **_kwargs: (_ for _ in ()).throw(AssertionError("terminal HF evidence must not republish")),
        )
        recovered = runtime_module.ProductionRuntime(checkpoint_transport_factory=lambda **_kwargs: object()).runtime_mixture_train(request)
        assert recovered["status"] == "runtime-mixture-terminal-hf-2000"
        assert recovered["immutable_checkpoint_publications"] == [publication]
        assert restored[-1] != output
        assert local.terminal_immutable_publication is None
    else:
        with pytest.raises(ValueError, match="authenticated immutable cursor|authenticated predecessor anchor|does not match|cursor or checkpoint archive|experiment manifest binding"):
            runtime_module.ProductionRuntime(checkpoint_transport_factory=lambda **_kwargs: object()).runtime_mixture_train(request)


@pytest.mark.parametrize(
    ("local_step", "journal", "expect_source", "expect_restore"),
    [
        (1500, False, "local", False),
        (1000, True, "local", False),
        (1000, False, "hf", True),
    ],
)
def test_runtime_local_first_selection_validates_hf_without_restoring_until_selected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    local_step: int,
    journal: bool,
    expect_source: str,
    expect_restore: bool,
) -> None:
    """A retained local cursor is never displaced by eager HF restoration."""
    from lehome_train.groot.local_recovery import attest_local_checkpoint, record_immutable_publication
    from lehome_train.groot.runtime_checkpoint_lifecycle import RuntimeMixtureTrainingIdentity

    prepared, output, cache = tmp_path / "prepared", tmp_path / "output", tmp_path / "cache"
    for root in (prepared, output, cache):
        root.mkdir()
    artifact_paths = {
        key: prepared / f"{key}.json"
        for key in ("manifest", "windows", "normalization", "mounts", "sources", "warmup", "binding")
    }
    for path in artifact_paths.values():
        path.write_text('{"ok":true}', encoding="utf-8")
    token = prepared / "token"; token.write_text("publisher-token", encoding="utf-8"); token.chmod(0o600)
    identity = RuntimeMixtureTrainingIdentity(
        mixture_id="a" * 64, deployment_receipt_sha256="b" * 64,
        source_revisions=(("organizer", "c" * 40, "bc/full", "d" * 64), ("rollout", "e" * 40, "rollouts/round-1", "f" * 64)),
        schedule_seed=1, code_bundle_sha256="1" * 64, code_bundle_revision="2" * 40,
        oci_image="sha256:" + "3" * 64, parent_step12000_artifact_sha256="4" * 64,
    )
    artifact_paths["sources"].write_text(json.dumps(identity.to_dict()), encoding="utf-8")
    experiment = ExperimentConfig(**_experiment_payload(batch=64))
    experiment_path = _write(prepared / "experiment.json", experiment.to_dict())
    archive = prepared / "resume.tar"; archive.write_bytes(b"authenticated-1k")
    descriptor = CheckpointDescriptor(
        record=CheckpointRecord("runtime", 1000, 64_000, "5" * 64, identity.mixture_id, "6" * 64,
            ArtifactIdentity("checkpoints/step-1000.tar", sha256_file(archive), archive.stat().st_size), True, False),
        normalization_sha256="7" * 64, schedule_sha256="6" * 64, locally_verified=True,
    )
    descriptor_path = prepared / "resume.json"; write_checkpoint_descriptor(descriptor_path, descriptor)
    config = SimpleNamespace(
        runtime_mixture_manifest=str(artifact_paths["manifest"]), runtime_window_index=str(artifact_paths["windows"]),
        runtime_mounts_descriptor=str(artifact_paths["mounts"]), dataset_path=str(prepared / "dataset"),
        base_model_path=str(cache / "parent"), output_dir=str(output), experiment_name="runtime",
        dataloader_num_workers=4, num_gpus=1, identity=lambda: {},
    )
    publication = {
        "schema_version": 1, "kind": "runtime_mixture_checkpoint_publication", "optimizer_step": 1000,
        "repository": runtime_module.DEFAULT_MODEL_REPO, "immutable_revision": "8" * 40,
        "remote_prefix": "runtime/checkpoint", "relative_path": "checkpoints/step-1000.tar",
        "artifact_sha256": sha256_file(archive), "artifact_byte_size": archive.stat().st_size,
        "descriptor_relative_path": "checkpoints/step-1000.json", "descriptor_sha256": sha256_file(descriptor_path),
        "descriptor_byte_size": descriptor_path.stat().st_size, "readback_verified": True,
        "identity": identity.to_dict(), "identity_sha256": identity.sha256,
        "runtime_cursor": {"optimizer_step": 1000, "global_sample_offset": 64_000, "physical_batch_size": 64, "action_horizon": 16},
        "fresh_tree_readback_verified": True,
    }
    anchor = {"immutable_anchor_revision": "8" * 40, "anchor_sha256": "9" * 64}
    official = output / "runtime" / f"checkpoint-{local_step}"
    official.mkdir(parents=True)
    (official / "weights.bin").write_bytes(b"preserved-local")
    (official / "trainer_state.json").write_text(json.dumps({"global_step": local_step, "log_history": [{"step": local_step, "loss": .2}]}))
    local_identity = {
        "experiment_manifest_sha256": "f" * 64,
        "parent_checkpoint_artifact_sha256": identity.parent_step12000_artifact_sha256,
        "runtime_mixture_id": identity.mixture_id, "trainer_code_sha256": identity.code_bundle_sha256,
        "trainer_code_revision": identity.code_bundle_revision,
    }
    local = attest_local_checkpoint(
        checkpoint=official, metadata_root=output / "local-recovery", optimizer_step=local_step,
        identity=local_identity,
    )
    if journal:
        record_immutable_publication(
            metadata_root=output / "local-recovery", checkpoint=local,
            publication=publication, anchor=anchor,
        )
    restored_roots: list[Path] = []
    launched: list[Path | None] = []
    monkeypatch.setattr(runtime_module, "_load_config", lambda _path: config)
    monkeypatch.setattr(runtime_module, "_load_experiment", lambda _path: experiment)
    monkeypatch.setattr(runtime_module, "_runtime_final_campaign", lambda *_args: None)
    monkeypatch.setattr(runtime_module, "GrootTrainingSession", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(runtime_module, "HubCheckpointUploader", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr("lehome_train.groot.runtime_mixture.load_runtime_contract", lambda *_args: SimpleNamespace(manifest=SimpleNamespace(mixture_id=identity.mixture_id, cycle_size=64, experiment_manifest_sha256="f" * 64), training_windows=()))
    monkeypatch.setattr("lehome_train.groot.runtime_mixture_warmup.bind_warmup_to_runtime_artifacts", lambda **_kwargs: {"binding": True})
    monkeypatch.setattr("lehome_train.groot.runtime_mixture_warmup.validate_gpu_warmup_receipt", lambda *_args, **_kwargs: 4)
    def restore(_archive: Path, **kwargs: object) -> None:
        root = kwargs["output_root"]
        assert isinstance(root, Path)
        restored_roots.append(root)
        run = root / "runtime"; run.mkdir()
        (run / "lehome_launch.json").write_text("{}", encoding="utf-8")
        checkpoint = run / "checkpoint-1000"; checkpoint.mkdir()
        (checkpoint / "weights.bin").write_bytes(b"hf")
        (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 1000, "log_history": [{"step": 1000, "loss": .1}]}))
    monkeypatch.setattr("lehome_train.groot.production_adapters._restore_checkpoint_archive", restore)
    monkeypatch.setattr(runtime_module, "launch_continuous_finetune", lambda _config, **kwargs: launched.append(kwargs.get("resume_checkpoint")))
    def supervisor(**kwargs: object) -> tuple[dict[str, object], ...]:
        launch = kwargs["launch"]
        assert callable(launch)
        launch()
        return () if kwargs["already_published"] else (publication,)
    monkeypatch.setattr(runtime_module, "run_continuous_supervisor", supervisor)
    request = {
        "launch_config": str(prepared / "launch.json"), "experiment_config": experiment_path,
        "runtime_manifest": str(artifact_paths["manifest"]), "runtime_window_index": str(artifact_paths["windows"]),
        "runtime_normalization": str(artifact_paths["normalization"]), "runtime_mounts_descriptor": str(artifact_paths["mounts"]),
        "runtime_source_evidence": str(artifact_paths["sources"]), "warmup_receipt": str(artifact_paths["warmup"]),
        "runtime_warmup_binding": str(artifact_paths["binding"]), "runtime_resume_archive": str(archive),
        "runtime_resume_descriptor": str(descriptor_path), "runtime_resume_cursor": publication["runtime_cursor"],
        "runtime_resume_anchor": anchor, "runtime_resume_publication": publication,
        "local_recovery_root": str(output / "local-recovery"), "checkpoint_repository": runtime_module.DEFAULT_MODEL_REPO,
        "checkpoint_revision": "main", "publisher_token_file": str(token), "instance_id": 7,
        "result_output": str(output / "result.json"), "status_output": str(output / "status.json"),
    }

    result = runtime_module.ProductionRuntime(checkpoint_transport_factory=lambda **_kwargs: object()).runtime_mixture_train(request)

    assert result["status"] == "runtime-mixture-interrupted"
    assert bool(restored_roots) is expect_restore
    assert (official / "weights.bin").read_bytes() == b"preserved-local"
    if expect_source == "local":
        assert launched == [official]
    else:
        assert len(launched) == 1 and launched[0] is not None
        assert Path(launched[0]).parent.parent.parent == output
        assert Path(launched[0]) != official


def test_hf_resume_staging_recovers_an_interrupted_private_restore_without_touching_local_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"; output.mkdir()
    local = output / "runtime" / "checkpoint-1000"; local.mkdir(parents=True)
    (local / "weights.bin").write_bytes(b"verified-local-fallback")
    archive = tmp_path / "resume.tar"; archive.write_bytes(b"hf-archive")
    descriptor = tmp_path / "resume.json"; descriptor.write_text("{}", encoding="utf-8")
    validated = runtime_module._ValidatedRuntimeResume(
        archive=archive, descriptor_path=descriptor, descriptor=object(),
        cursor={"optimizer_step": 1000, "global_sample_offset": 64_000, "physical_batch_size": 64, "action_horizon": 16},
    )
    config = SimpleNamespace(output_dir=str(output), experiment_name="runtime", num_gpus=1, identity=lambda: {})
    calls: list[Path] = []

    monkeypatch.setattr(
        "lehome_train.groot.production_adapters._restore_checkpoint_archive",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("simulated termination")),
    )
    with pytest.raises(RuntimeError, match="termination"):
        runtime_module._restore_validated_runtime_resume_checkpoint(
            validated=validated, config=config, preserve_existing_run=True,
        )
    staging = output / f".runtime-hf-resume-1000-{sha256_file(archive)[:16]}"
    assert (staging / ".INCOMPLETE.json").is_file()
    assert (local / "weights.bin").read_bytes() == b"verified-local-fallback"

    def restore(_archive: Path, **kwargs: object) -> None:
        root = kwargs["output_root"]
        assert isinstance(root, Path)
        calls.append(root)
        run = root / "runtime"; run.mkdir()
        (run / "lehome_launch.json").write_text("{}", encoding="utf-8")
        (run / "checkpoint-1000").mkdir()

    monkeypatch.setattr("lehome_train.groot.production_adapters._restore_checkpoint_archive", restore)
    restored = runtime_module._restore_validated_runtime_resume_checkpoint(
        validated=validated, config=config, preserve_existing_run=True,
    )

    assert restored == staging / "runtime" / "checkpoint-1000"
    assert calls == [staging]
    assert not (staging / ".INCOMPLETE.json").exists()
    assert (local / "weights.bin").read_bytes() == b"verified-local-fallback"
    assert not list(output.glob(f".{staging.name}.*.incomplete"))


def test_live_device_provenance_shuts_down_nvml_without_masking_device_read_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    fake_nvml = SimpleNamespace(
        nvmlInit=lambda: events.append("init"),
        nvmlDeviceGetHandleByIndex=lambda _index: object(),
        nvmlDeviceGetUUID=lambda _handle: "GPU-1234",
        nvmlShutdown=lambda: events.append("shutdown"),
    )
    fake_cuda = SimpleNamespace(
        get_device_properties=lambda _index: (_ for _ in ()).throw(RuntimeError("device properties failed")),
        get_device_name=lambda _index: "NVIDIA RTX PRO 6000",
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)

    with pytest.raises(RuntimeError, match="device properties failed"):
        runtime_module._live_device_provenance(SimpleNamespace(cuda=fake_cuda))

    assert events == ["init", "shutdown"]

    events.clear()
    fake_nvml.nvmlDeviceGetUUID = lambda _handle: (_ for _ in ()).throw(ValueError("UUID read failed"))
    with pytest.raises(RuntimeError, match="stable GPU UUID") as raised:
        runtime_module._live_device_provenance(SimpleNamespace(cuda=fake_cuda))

    assert isinstance(raised.value.__cause__, ValueError)
    assert str(raised.value.__cause__) == "UUID read failed"
    assert events == ["init", "shutdown"]


def test_live_device_provenance_normalizes_a_torchversion_string_subclass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TorchVersion(str):
        pass

    fake_nvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda _index: object(),
        nvmlDeviceGetUUID=lambda _handle: "GPU-1234",
        nvmlShutdown=lambda: None,
    )
    fake_cuda = SimpleNamespace(
        is_available=lambda: True, is_initialized=lambda: True,
        get_device_properties=lambda _index: SimpleNamespace(total_memory=96 * 1024**3),
        get_device_name=lambda _index: "NVIDIA RTX PRO 6000",
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)

    state = runtime_module._live_device_provenance(SimpleNamespace(
        cuda=fake_cuda, __version__=TorchVersion("2.7.0+cu128"),
        version=SimpleNamespace(cuda="12.8"),
    ))

    assert type(state.torch_version) is str
    assert state.torch_version == "2.7.0+cu128"


def test_runtime_gpu_warmup_production_adapter_measures_live_loader_model_and_nvml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production factory exposes measurements, never caller-authored rows."""

    class FakeCuda:
        def __init__(self) -> None:
            self.synchronizations = 0
            self.empty_cache_calls = 0
            self.ipc_collect_calls = 0

        def is_available(self) -> bool:
            return True

        def is_initialized(self) -> bool:
            return True

        def synchronize(self) -> None:
            self.synchronizations += 1

        def empty_cache(self) -> None:
            self.empty_cache_calls += 1

        def ipc_collect(self) -> None:
            self.ipc_collect_calls += 1

        def reset_peak_memory_stats(self) -> None:
            return None

        def max_memory_allocated(self) -> int:
            return 32 * 1024**3

        def max_memory_reserved(self) -> int:
            return 40 * 1024**3

        def mem_get_info(self) -> tuple[int, int]:
            return (20 * 1024**3, 96 * 1024**3)

    class FakeTensor:
        shape = (64, 1)

        def to(self, device: str, *, non_blocking: bool) -> "FakeTensor":
            assert device == "cuda" and non_blocking is True
            return self

    class FakeLoss:
        def backward(self) -> None:
            return None

        def item(self) -> float:
            return 0.125

    class FakeModel:
        def parameters(self):
            return iter((SimpleNamespace(device=SimpleNamespace(type="cuda")),))

        def train(self) -> None:
            return None

        def __call__(self, **batch: object) -> object:
            assert isinstance(batch["pixel_values"], FakeTensor)
            return SimpleNamespace(loss=FakeLoss())

    class FakeOptimizer:
        def __init__(self) -> None:
            self.steps = 0
            self.clears = 0

        def step(self) -> None:
            self.steps += 1

        def zero_grad(self, *, set_to_none: bool) -> None:
            assert set_to_none is True
            self.clears += 1

    class FakeLoader:
        def __init__(self) -> None:
            self.shutdowns = 0

        def __iter__(self) -> "FakeLoader":
            return self

        def __next__(self) -> dict[str, FakeTensor]:
            return {"pixel_values": FakeTensor()}

        def _shutdown_workers(self) -> None:
            self.shutdowns += 1

    class FakeSampler:
        def __init__(self) -> None:
            self.closed = False

        def sample(self) -> object:
            return SimpleNamespace(gpu_utilization_percent=80.0)

        def close(self) -> None:
            self.closed = True

    class FakeAutocast:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: object) -> None:
            return None

    def fake_autocast(*, device_type: str, dtype: object) -> FakeAutocast:
        assert device_type == "cuda"
        assert dtype is fake_bfloat16
        return FakeAutocast()

    cuda = FakeCuda()
    fake_bfloat16 = object()
    fake_torch = SimpleNamespace(
        cuda=cuda, autocast=fake_autocast, bfloat16=fake_bfloat16,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    loaders: list[FakeLoader] = []
    samplers: list[FakeSampler] = []
    times = iter(
        value
            for step in range(7)
        for value in (float(step), float(step), float(step) + 0.01, float(step) + 0.20)
    )
    optimizer = FakeOptimizer()
    session = runtime_module._RuntimeMixtureWarmupSession(
        torch_module=fake_torch,
        model=FakeModel(),
        optimizer=optimizer,
        loader_factory=lambda workers: loaders.append(FakeLoader()) or loaders[-1],
        sampler_factory=lambda: samplers.append(FakeSampler()) or samplers[-1],
        runtime_state_probe=lambda: __import__("lehome_train.groot.runtime_mixture_warmup", fromlist=["RuntimeState"]).RuntimeState(
            torch_cuda_available=True, torch_cuda_initialized=True, model_loaded=True,
            hostname="gpu-host", host_architecture="x86_64", torch_version="2.7.0",
            cuda_version="12.8", gpu_device_name="NVIDIA RTX PRO 6000",
            gpu_uuid="GPU-1234", total_vram_bytes=96 * 1024**3,
        ),
        materialization_proof_factory=lambda: {
            "bc": {"source_type": "bc", "window_id": "bc-window", "action_horizon": 16, "camera_count": 3},
            "rollout": {"source_type": "rollout", "window_id": "rollout-window", "action_horizon": 16, "camera_count": 3},
        },
        clock=lambda: next(times),
    )
    production = runtime_module.ProductionRuntime()
    monkeypatch.setattr(production, "_create_runtime_gpu_warmup_session", lambda _arguments: session)

    adapter = production.runtime_gpu_warmup_adapter({"binding": {}})
    assert adapter.runtime_state().to_dict()["gpu_uuid"] == "GPU-1234"
    measured = adapter.measure(worker_count=12, burn_in_steps=2, measured_steps=5)

    assert measured.decoded_samples == 64 * 7
    assert measured.measured_steps == 5
    assert measured.oom is False and measured.error is None
    assert measured.loader_wait_seconds == pytest.approx(0.05)
    assert measured.step_seconds == pytest.approx(1.0)
    assert measured.gpu_busy_seconds == pytest.approx(0.8)
    assert measured.gpu_utilization_percent == 80.0
    assert measured.observed_batch_sizes == (64,) * 7
    assert measured.loss_min == measured.loss_max == measured.loss_final == 0.125
    assert measured.minimum_free_vram_bytes == 20 * 1024**3
    assert measured.materialization_proof is not None
    assert optimizer.steps == 7 and optimizer.clears == 7
    assert len(loaders) == len(samplers) == 1
    assert loaders[0].shutdowns == 1 and samplers[0].closed is True
    assert cuda.empty_cache_calls == cuda.ipc_collect_calls == 1
    rejected = session.measure(worker_count=24, burn_in_steps=10, measured_steps=50)
    assert rejected.error == "runtime GPU warm-up worker count is not canonical"


def test_runtime_gpu_warmup_records_oom_and_cleans_workers_without_claiming_success() -> None:
    class FakeCuda:
        def synchronize(self) -> None:
            return None

        def empty_cache(self) -> None:
            return None

        def ipc_collect(self) -> None:
            return None

        def reset_peak_memory_stats(self) -> None:
            return None

        def max_memory_allocated(self) -> int:
            return 32 * 1024**3

        def max_memory_reserved(self) -> int:
            return 40 * 1024**3

        def mem_get_info(self) -> tuple[int, int]:
            return (20 * 1024**3, 96 * 1024**3)

    class OomModel:
        def parameters(self):
            return iter((SimpleNamespace(device=SimpleNamespace(type="cuda")),))

        def train(self) -> None:
            return None

        def __call__(self, **_batch: object) -> object:
            raise RuntimeError("CUDA out of memory")

    class Loader:
        shutdowns = 0

        def __iter__(self) -> "Loader":
            return self

        def __next__(self) -> dict[str, object]:
            return {"batch": SimpleNamespace(shape=(64,), to=lambda *_args, **_kwargs: self)}

        def _shutdown_workers(self) -> None:
            self.shutdowns += 1

    loader = Loader()
    sampler = SimpleNamespace(sample=lambda: pytest.fail("NVML sampling must not follow an OOM"), closed=False)

    def sampler_factory() -> object:
        sampler.close = lambda: setattr(sampler, "closed", True)
        return sampler

    class FakeAutocast:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: object) -> None:
            return None

    fake_bfloat16 = object()

    def fake_autocast(*, device_type: str, dtype: object) -> FakeAutocast:
        assert device_type == "cuda"
        assert dtype is fake_bfloat16
        return FakeAutocast()

    session = runtime_module._RuntimeMixtureWarmupSession(
        torch_module=SimpleNamespace(
            cuda=FakeCuda(), autocast=fake_autocast, bfloat16=fake_bfloat16,
        ),
        model=OomModel(),
        optimizer=SimpleNamespace(step=lambda: None, zero_grad=lambda **_kwargs: None),
        loader_factory=lambda _workers: loader,
        sampler_factory=sampler_factory,
        clock=iter((0.0, 0.0, 0.01)).__next__,
    )

    measured = session.measure(worker_count=4, burn_in_steps=2, measured_steps=5)

    assert measured.oom is True
    assert measured.measured_steps == 0
    assert measured.decoded_samples == 64
    assert "out of memory" in (measured.error or "").lower()
    assert loader.shutdowns == 1
    assert sampler.closed is True


def test_runtime_gpu_warmup_never_substitutes_unloaded_model_or_missing_nvml() -> None:
    class FakeCuda:
        def synchronize(self) -> None:
            return None

        def empty_cache(self) -> None:
            return None

        def ipc_collect(self) -> None:
            return None

        def reset_peak_memory_stats(self) -> None:
            return None

        def max_memory_allocated(self) -> int:
            return 32 * 1024**3

        def max_memory_reserved(self) -> int:
            return 40 * 1024**3

        def mem_get_info(self) -> tuple[int, int]:
            return (20 * 1024**3, 96 * 1024**3)

    class Model:
        def __init__(self, device: str) -> None:
            self._device = device

        def parameters(self):
            return iter((SimpleNamespace(device=SimpleNamespace(type=self._device)),))

    unused_loader = lambda _workers: pytest.fail("loader must not replace an unloaded model")
    unloaded = runtime_module._RuntimeMixtureWarmupSession(
        torch_module=SimpleNamespace(cuda=FakeCuda()),
        model=Model("cpu"),
        optimizer=SimpleNamespace(),
        loader_factory=unused_loader,
        sampler_factory=lambda: pytest.fail("NVML must not replace an unloaded model"),
    )
    absent_model = unloaded.measure(worker_count=4, burn_in_steps=2, measured_steps=5)
    assert absent_model.measured_steps == absent_model.decoded_samples == 0
    assert absent_model.oom is False
    assert "not loaded" in (absent_model.error or "")

    # Python special methods are resolved on the type, so a real tiny iterator
    # keeps this focused on the unavailable NVML path.
    class OneBatchLoader:
        def __iter__(self) -> "OneBatchLoader":
            return self

        def __next__(self) -> object:
            return SimpleNamespace(shape=(64,))

        def _shutdown_workers(self) -> None:
            return None

    missing_nvml = runtime_module._RuntimeMixtureWarmupSession(
        torch_module=SimpleNamespace(cuda=FakeCuda()),
        model=Model("cuda"),
        optimizer=SimpleNamespace(),
        loader_factory=lambda _workers: OneBatchLoader(),
        sampler_factory=lambda: (_ for _ in ()).throw(RuntimeError("NVML unavailable")),
    )
    absent_nvml = missing_nvml.measure(worker_count=4, burn_in_steps=2, measured_steps=5)
    assert absent_nvml.measured_steps == absent_nvml.decoded_samples == 0
    assert absent_nvml.oom is False
    assert absent_nvml.error == "NVML unavailable"


def test_resume_downloads_and_consumes_the_immutable_authenticated_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_bytes = b"authenticated checkpoint archive"
    archive_sha = __import__("hashlib").sha256(archive_bytes).hexdigest()
    descriptor = CheckpointDescriptor(
        record=CheckpointRecord(
            experiment_id="corrective-rft-70-30-20260813",
            optimizer_step=1000,
            sample_presentations=64_000,
            experiment_config_sha256="a" * 64,
            dataset_manifest_sha256="b" * 64,
            schedule_sha256="c" * 64,
            artifact=ArtifactIdentity("checkpoints/step-1000.tar", archive_sha, len(archive_bytes)),
            resumable=True,
            remotely_verified=False,
        ),
        normalization_sha256="d" * 64,
        schedule_sha256="c" * 64,
        locally_verified=True,
    )
    staged = tmp_path / "prepared" / "resume-checkpoint.json"
    staged.parent.mkdir()
    write_checkpoint_descriptor(staged, descriptor)
    descriptor_sha = sha256_file(staged)
    downloaded: list[tuple[str, ...]] = []

    class FakeTransport:
        def list_tree(self, **_kwargs: object) -> tuple[object, ...]:
            return (
                SimpleNamespace(relative_path="prefix/checkpoints/step-1000.tar", entry_type="file"),
                SimpleNamespace(relative_path="prefix/checkpoints/step-1000.json", entry_type="file"),
            )

        def download_files(self, **kwargs: object) -> None:
            destination = kwargs["destination"]
            assert isinstance(destination, Path)
            paths = kwargs["relative_paths"]
            assert isinstance(paths, tuple)
            downloaded.append(paths)
            for relative in paths:
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(
                    archive_bytes if relative.endswith(".tar") else staged.read_bytes()
                )

    monkeypatch.setattr(runtime_module, "HuggingFaceHubTransport", lambda **_kwargs: FakeTransport())
    publication = {
        "repository": runtime_module.DEFAULT_MODEL_REPO,
        "immutable_revision": "e" * 40,
        "remote_prefix": "prefix",
        "relative_path": "checkpoints/step-1000.tar",
        "artifact_sha256": archive_sha,
        "artifact_byte_size": len(archive_bytes),
        "descriptor_relative_path": "checkpoints/step-1000.json",
        "descriptor_sha256": descriptor_sha,
        "descriptor_byte_size": staged.stat().st_size,
    }

    observed = runtime_module._resume_publication(
        value=publication,
        descriptor=str(staged),
        output_root=tmp_path / "output",
        token="publisher-token",
    )

    assert observed.record.remotely_verified is True
    assert descriptor.record.remotely_verified is False
    assert downloaded == [("checkpoints/step-1000.tar", "checkpoints/step-1000.json")]


def test_publisher_readback_promotes_an_actual_false_descriptor_for_runtime_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifacts"
    archive = root / "checkpoints" / "step-1000.tar"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"checkpoint archive")
    descriptor = CheckpointDescriptor(
        record=CheckpointRecord(
            experiment_id="corrective-rft-70-30-20260813",
            optimizer_step=1000,
            sample_presentations=64_000,
            experiment_config_sha256="a" * 64,
            dataset_manifest_sha256="b" * 64,
            schedule_sha256="c" * 64,
            artifact=ArtifactIdentity("checkpoints/step-1000.tar", sha256_file(archive), archive.stat().st_size),
            resumable=True,
            remotely_verified=False,
        ),
        normalization_sha256="d" * 64,
        schedule_sha256="c" * 64,
        locally_verified=True,
    )
    descriptor_path = root / "checkpoints" / "step-1000.json"
    write_checkpoint_descriptor(descriptor_path, descriptor)
    remote: dict[str, bytes] = {}

    monkeypatch.setattr(adapters, "HuggingFaceHubTransport", lambda **_kwargs: object())
    monkeypatch.setattr(adapters, "require_access", lambda **_kwargs: None)

    def upload(**kwargs: object) -> str:
        assert kwargs["max_attempts"] == 3
        source = kwargs["source"]
        assert isinstance(source, Path)
        for entry in kwargs["entries"]:
            remote[entry.relative_path] = (source / entry.relative_path).read_bytes()
        return "e" * 40

    def download(**kwargs: object) -> str:
        assert kwargs["max_attempts"] == 3
        destination = kwargs["destination"]
        assert isinstance(destination, Path)
        for relative in kwargs["relative_paths"]:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(remote[relative])
        return "e" * 40

    monkeypatch.setattr(adapters, "upload_files", upload)
    monkeypatch.setattr(adapters, "download_files", download)
    uploader = adapters.HubCheckpointUploader(
        repository=runtime_module.DEFAULT_MODEL_REPO,
        revision="main",
        experiment_id="corrective-rft-70-30-20260813",
        artifact_root=root,
        token="publisher-token",
    )
    publication = uploader.publish_receipt(descriptor, timeout_seconds=1)
    anchor_receipt = uploader.publish_anchor(
        {
            "schema_version": 1,
            "kind": "runtime_mixture_checkpoint_anchor",
            "repository": runtime_module.DEFAULT_MODEL_REPO,
            "anchor_ref": "main",
            "experiment_id": "corrective-rft-70-30-20260813",
        },
        timeout_seconds=1,
    )
    assert anchor_receipt == {
        "immutable_anchor_revision": "e" * 40,
        "anchor_sha256": __import__("hashlib").sha256(remote["latest.json"]).hexdigest(),
        "readback_verified": True,
    }
    staged = tmp_path / "prepared" / "resume-checkpoint.json"
    staged.parent.mkdir()
    staged.write_bytes(descriptor_path.read_bytes())

    class RuntimeTransport:
        def list_tree(self, **_kwargs: object) -> tuple[object, ...]:
            return tuple(
                SimpleNamespace(relative_path="prefix/" + path, entry_type="file")
                for path in remote
            )

        def download_files(self, **kwargs: object) -> None:
            destination = kwargs["destination"]
            assert isinstance(destination, Path)
            for relative in kwargs["relative_paths"]:
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(remote[relative])

    monkeypatch.setattr(runtime_module, "HuggingFaceHubTransport", lambda **_kwargs: RuntimeTransport())
    publication = publication | {"remote_prefix": "prefix"}
    resumed = runtime_module._resume_publication(
        value=publication,
        descriptor=str(staged),
        output_root=tmp_path / "output",
        token="publisher-token",
    )

    assert descriptor.record.remotely_verified is False
    assert resumed.record.remotely_verified is True


def test_continuous_campaign_binds_local_dataset_revision_to_the_sealed_manifest(
    tmp_path: Path,
) -> None:
    manifest_sha = "f" * 64
    launch = _launch_payload(tmp_path, batch=64, max_steps=2_000) | {
        "base_model_path": "/cache/parent",
        "dataset_path": "/prepared/generation",
        "dataset_revision": manifest_sha[:40],
        "output_dir": "/output",
        "modality_config_path": "/prepared/config/modality.py",
        "experiment_name": "corrective-rft-70-30-20260813",
            "augmentation_profile": "none",
            "save_steps": 500,
        "parent_checkpoint_repository": "ryanjin333/lehome-groot-n17-models",
        "parent_checkpoint_revision": "30ac1a84da67b099e115ad147bcd61e9d60046d3",
        "parent_checkpoint_subpath": "policies/step-12000",
        "parent_checkpoint_artifact_sha256": "3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06",
    }
    config = FineTuneLaunchConfig(**launch)
    experiment = ExperimentConfig(**(_experiment_payload(batch=64) | {
        "container_digest": "sha256:b56c16c259b7eda99294f2069e976b53395e665aaf68174d5b13ba458a93b746",
        "dataset_repository": "local/sealed-mixed-generation",
        "dataset_revision": manifest_sha[:40],
        "dataset_manifest_sha256": manifest_sha,
        "sample_presentations": 128_000,
    }))

    identity = runtime_module._continuous_campaign_identity(
        config, experiment, {"sealed": True, "mix_plan_sha256": "a" * 64, "dataset_manifest_sha256": manifest_sha},
    )

    assert identity == {"mix_plan_sha256": "a" * 64, "dataset_manifest_sha256": manifest_sha}
    with pytest.raises(ValueError, match="local dataset revision"):
        runtime_module._continuous_campaign_identity(
            config, experiment, {"sealed": True, "mix_plan_sha256": "a" * 64, "dataset_manifest_sha256": "e" * 64},
        )

def test_train_delegates_all_budget_resume_storage_and_upload_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch_path = _write(
        tmp_path / "prepared" / "train-launch.json",
        _launch_payload(tmp_path, batch=64, max_steps=12_000),
    )
    experiment = ExperimentConfig(**_experiment_payload(batch=64))
    experiment_path = _write(
        tmp_path / "prepared" / "experiment.json", experiment.to_dict()
    )
    selected = SmokeResult(
        experiment_id="experiment-001",
        experiment_config_sha256=canonical_json_sha256(experiment),
        dataset_manifest_sha256=DATASET_SHA256,
        physical_batch_size=64,
        gradient_accumulation_steps=1,
        optimizer_steps=100,
        stable=True,
        finite_loss=True,
        physical_vram_bytes=96 * 1024**3,
        peak_reserved_vram_bytes=70 * 1024**3,
        minimum_steady_state_free_vram_bytes=20 * 1024**3,
        steady_steps_per_second=1.0,
        samples_per_second=64.0,
        failure_reason=None,
    )
    smoke_path = _write(
        tmp_path / "prepared" / "selected-smoke.json", selected.to_dict()
    )
    result_path = tmp_path / "output" / "reports" / "train.json"
    status_path = tmp_path / "output" / "status" / "train.json"
    observed: dict[str, object] = {}

    def fake_controller(**kwargs: object) -> SimpleNamespace:
        assert status_path.parent.is_dir()
        observed.update(kwargs)
        return SimpleNamespace(
            status="completed",
            to_dict=lambda: {
                "status": "completed",
                "sample_presentations": 768_000,
                "checkpoints": [],
            },
        )

    monkeypatch.setattr(runtime_module, "run_fixed_exposure_training", fake_controller)
    returned = runtime_module.ProductionRuntime().train(
        {
            "launch_config": launch_path,
            "experiment_config": experiment_path,
            "selected_smoke_result": smoke_path,
            "normalization_sha256": normalization_identity(
                tmp_path / "prepared" / "dataset"
            ),
            "estimated_checkpoint_bytes": 8 * 1024**3,
            "checkpoint_repository": "ryanjin333/lehome-groot-n17-models",
            "checkpoint_revision": "training-uploads",
            "resume_checkpoint": None,
            "provider_hourly_price": 1.25,
            "instance_start_time": "2026-07-31T10:00:00Z",
            "result_output": str(result_path),
            "status_output": str(status_path),
        }
    )

    assert observed["experiment_config"] == experiment
    assert observed["selected_smoke"] == selected
    assert observed["normalization_sha256"] == normalization_identity(
        tmp_path / "prepared" / "dataset"
    )
    assert callable(observed["runner"])
    assert callable(observed["checkpointer"])
    assert callable(observed["uploader"])
    assert callable(observed["disk_probe"])
    assert callable(observed["checkpoint_deleter"])
    assert observed["resume_checkpoint"] is None
    assert observed["status_path"] == status_path
    assert returned["sample_presentations"] == 768_000
    log_payload = json.loads(
        (tmp_path / "output" / "experiment" / "logs" / "train.json").read_text()
    )
    assert log_payload == {
        "schema_version": 1,
        "event": "training_terminal",
        "experiment_id": "experiment-001",
        "experiment_config_sha256": canonical_json_sha256(experiment),
        "normalization_sha256": normalization_identity(
            tmp_path / "prepared" / "dataset"
        ),
        "status": "completed",
        "sample_presentations": 768_000,
    }


def test_production_root_artifacts_satisfy_default_sync_allowlist(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    root.mkdir()
    experiment = ExperimentConfig(**_experiment_payload(batch=64))
    runtime_module._write_immutable_json_artifact(
        root / "resolved-config.json", experiment.to_dict()
    )
    runtime_module._write_immutable_json_artifact(
        root / "provenance.json",
        {"schema_version": 1, "experiment_id": "experiment-001"},
    )
    runtime_module._write_safe_json_artifact(
        root / "logs" / "prepare.json",
        {"schema_version": 1, "event": "prepared"},
    )
    runtime_module._write_safe_json_artifact(
        root / "logs" / "train.json",
        {"schema_version": 1, "event": "training_terminal"},
    )
    (root / "reports").mkdir()
    (root / "reports" / "training-result.json").write_text("{}", encoding="utf-8")
    (root / "checkpoints").mkdir()
    (root / "checkpoints" / "step-12000.tar.zst").write_bytes(b"checkpoint")

    manifest = generate_sync_manifest(
        root,
        experiment_id="experiment-001",
        experiment_config_sha256=canonical_json_sha256(experiment),
    )

    assert {entry.relative_path for entry in manifest.entries} == {
        "checkpoints/step-12000.tar.zst",
        "logs/prepare.json",
        "logs/train.json",
        "provenance.json",
        "reports/training-result.json",
        "resolved-config.json",
    }


def test_train_rejects_caller_normalization_mismatch_before_controller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch_path = _write(
        tmp_path / "prepared" / "train-launch.json",
        _launch_payload(tmp_path, batch=64, max_steps=12_000),
    )
    experiment_path = _write(
        tmp_path / "prepared" / "experiment.json",
        _experiment_payload(batch=64),
    )
    smoke = SmokeResult(
        experiment_id="experiment-001",
        experiment_config_sha256=canonical_json_sha256(
            ExperimentConfig(**_experiment_payload(batch=64))
        ),
        dataset_manifest_sha256=DATASET_SHA256,
        physical_batch_size=64,
        gradient_accumulation_steps=1,
        optimizer_steps=100,
        stable=True,
        finite_loss=True,
        physical_vram_bytes=96 * 1024**3,
        peak_reserved_vram_bytes=70 * 1024**3,
        minimum_steady_state_free_vram_bytes=20 * 1024**3,
        steady_steps_per_second=1.0,
        samples_per_second=64.0,
        failure_reason=None,
    )
    smoke_path = _write(tmp_path / "prepared" / "smoke.json", smoke.to_dict())
    launched = False

    def forbidden_controller(**_kwargs: object) -> object:
        nonlocal launched
        launched = True
        raise AssertionError("controller must not start")

    monkeypatch.setattr(runtime_module, "run_fixed_exposure_training", forbidden_controller)
    with pytest.raises(ValueError, match="normalization identity is incompatible"):
        runtime_module.ProductionRuntime().train(
            {
                "launch_config": launch_path,
                "experiment_config": experiment_path,
                "selected_smoke_result": smoke_path,
                "normalization_sha256": "f" * 64,
                "estimated_checkpoint_bytes": 8 * 1024**3,
                "checkpoint_repository": "ryanjin333/lehome-groot-n17-models",
                "checkpoint_revision": "training-uploads",
                "resume_checkpoint": None,
                "provider_hourly_price": None,
                "instance_start_time": None,
                "result_output": str(tmp_path / "output" / "train.json"),
                "status_output": str(tmp_path / "output" / "status.json"),
            }
        )

    assert launched is False


def test_runtime_rejects_uncreatable_status_parent_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch_path = _write(
        tmp_path / "prepared" / "memorize-launch.json",
        _launch_payload(tmp_path, batch=1, max_steps=10_000),
    )
    experiment_path = _write(
        tmp_path / "prepared" / "experiment.json",
        _experiment_payload(batch=1),
    )
    blocker = tmp_path / "output" / "not-a-directory"
    blocker.write_text("blocked", encoding="utf-8")
    launched = False

    def fake_controller(**_kwargs: object) -> object:
        nonlocal launched
        launched = True
        return SimpleNamespace(to_dict=lambda: {})

    monkeypatch.setattr(runtime_module, "run_memorization", fake_controller)
    with pytest.raises(ValueError, match="output parent"):
        runtime_module.ProductionRuntime().memorize(
            {
                "launch_config": launch_path,
                "experiment_config": experiment_path,
                "dataset_manifest_sha256": DATASET_SHA256,
                "requested_episode_id": None,
                "result_output": str(blocker / "result.json"),
                "status_output": str(blocker / "status.json"),
            }
        )
    assert launched is False
