from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import lehome_train.groot.production_runtime as runtime_module
from lehome_train.commands.sync import generate_sync_manifest
from lehome_train.constants import MODEL_REVISION
from lehome_train.data.normalization import normalization_identity
from lehome_train.io import canonical_json_sha256, sha256_file
from lehome_train.models import ExperimentConfig, SmokeResult


COMMIT = "a" * 40
DATASET_REVISION = "b" * 40
DATASET_SHA256 = "c" * 64
NORMALIZATION_SHA256 = "d" * 64
IMAGE_DIGEST = "sha256:" + "e" * 64


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
