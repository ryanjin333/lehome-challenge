"""Literal runtime warm-up staging-to-session integration proof.

The paid SSH/SCP boundary is represented by a local filesystem transport.  The
stage function and production config loader stay real: this catches drift
between the staged launch location and the runtime's fixed canonical read.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from lehome_train.constants import MODEL_REVISION
from lehome_train.io import sha256_file


REPOSITORY = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "literal_runtime_warmup_stage_under_test",
    REPOSITORY / "scripts" / "run_groot_persistent_training.py",
)
assert SPEC is not None and SPEC.loader is not None
LIFECYCLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LIFECYCLE
SPEC.loader.exec_module(LIFECYCLE)


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture
def runtime_mounts(tmp_path: Path) -> dict[str, Path]:
    """Local counterparts for the three canonical mounts used over SSH."""

    mounts = {
        "prepared": tmp_path / "prepared",
        "cache": tmp_path / "cache",
        "output": tmp_path / "output",
    }
    for root in mounts.values():
        root.mkdir()
    return mounts


def _local_stage_transport(
    remote_root: Path, mounts: dict[str, Path],
) -> tuple[object, list[tuple[str, ...]]]:
    """Interpret only this stage's SSH/SCP commands against local fixture bytes."""

    calls: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...]) -> str:
        calls.append(command)
        if command[0] == "scp":
            source, destination = Path(command[-2]), command[-1]
            assert destination.startswith("root@fixture:")
            remote_path = destination.partition(":")[2]
            target = remote_root / remote_path.lstrip("/")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            return ""
        shell = command[-1]
        assert command[0] == "ssh"
        translated = shell.replace(
            "/tmp/lehome-runtime-warmup", str(remote_root / "tmp" / "lehome-runtime-warmup")
        )
        for name, root in mounts.items():
            translated = translated.replace("/" + name, str(root))
        completed = subprocess.run(
            ("/bin/sh", "-c", translated), check=True, capture_output=True, text=True
        )
        return completed.stdout

    return runner, calls


def _campaign_receipts(tmp_path: Path) -> dict[str, Path]:
    common = {
        "repository": LIFECYCLE.CORRECTIVE_SOURCE["repository"],
        "fresh_readback_verified": True,
        "tree_listing_verified": True,
    }
    return {
        "bc": _write(tmp_path / "campaign" / "bc.json", common | {
            "immutable_revision": "b" * 40, "remote_prefix": "bc/full",
        }),
        "rollout": _write(tmp_path / "campaign" / "rollout.json", common | {
            "immutable_revision": "c" * 40, "remote_prefix": "rollouts/round-1",
        }),
        "deployment": _write(tmp_path / "campaign" / "deployment.json", common | {
            "immutable_revision": "a" * 40, "remote_prefix": "mixtures/" + "d" * 64,
            "mixture_id": "d" * 64, "pending_receipt_sha256": "e" * 64,
            "artifact_entries": ["mixture.json"],
        }),
    }


def _cpu_pilot(*, instance: dict[str, object], code_revision: str, code_sha256: str) -> dict[str, object]:
    workers = [0, 4, 8, 16, 24]
    return {
        "schema_version": 4,
        "kind": "runtime_mixture_loader_pilot",
        "model_loaded": False,
        "gpu_initialized": False,
        "processor_contract": "pinned_processor_integration_required",
        "representative": {"three_cameras": True, "action_horizon": 16},
        "sample_count_per_worker": 100,
        "worker_counts": workers,
        "canonical_worker_counts": workers,
        "loader_throughput": {
            str(worker): {"decoded_samples": 100, "samples_per_second": 100.0}
            for worker in workers
        },
        "timing_rows": [
            {
                "worker_count": worker, "decoded_samples": 100, "seconds": 1.0,
                "samples_per_second": 100.0, "host_cpu_seconds": 1.0,
                "host_max_rss_mib": 1.0, "latency_seconds_p50": 0.01,
                "latency_seconds_p95": 0.02,
            }
            for worker in workers
        ],
        "authenticated_evidence": {
            "provider_instance_id": instance["instance_id"],
            "provider_response_sha256": instance["provider_response_sha256"],
            "platform_arch": "x86_64",
            "image_digest": LIFECYCLE.RUNTIME_CPU_PILOT_IMAGE.rpartition("@")[2],
            "code_revision": code_revision,
            "code_bundle_sha256": code_sha256,
            "bc_revision": "b" * 40,
            "rollout_revision": "c" * 40,
            "deployment_revision": "a" * 40,
        },
        "cache_cap": 1,
        "native_x86_required": True,
        "timeout_seconds": 60.0,
        "canonical_completion": True,
    }


def _install_low_level_runtime_fakes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> tuple[list[object], list[object]]:
    """Fake only CUDA/GR00T loader/NVML implementation details below the session."""

    import lehome_train.groot.launch as launch
    import lehome_train.groot.production_runtime as production
    from lehome_train.constants import ISAAC_GROOT_REVISION

    loaded_pipelines: list[object] = []
    created_loaders: list[object] = []

    class FakeCuda:
        def is_available(self) -> bool:
            return True

        def is_initialized(self) -> bool:
            return True

    class FakeParameter:
        requires_grad = True
        device = SimpleNamespace(type="cuda")

    class FakeModel:
        def to(self, device: str) -> None:
            assert device == "cuda"

        def train(self) -> None:
            return None

        def parameters(self):
            return iter((FakeParameter(),))

    class FakeOptimizer:
        def __init__(self, parameters: object, **_kwargs: object) -> None:
            assert list(parameters)

    class FakeLoader:
        def __init__(self, dataset: object, **kwargs: object) -> None:
            self.dataset, self.kwargs = dataset, kwargs
            created_loaders.append(self)

    fake_torch = SimpleNamespace(
        cuda=FakeCuda(),
        optim=SimpleNamespace(AdamW=FakeOptimizer),
        utils=SimpleNamespace(data=SimpleNamespace(DataLoader=FakeLoader)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeModelConfig:
        pass

    class FakePinnedConfig:
        def __init__(self) -> None:
            self.data = SimpleNamespace(multiprocessing_context="spawn")
            self.model = FakeModelConfig()
            self.training = SimpleNamespace()

        def load_dict(self, value: object) -> "FakePinnedConfig":
            self.loaded_data = value
            return self

    class FakeProcessor:
        def set_statistics(self, statistics: object, *, override: bool) -> None:
            assert statistics and override is True

    class FakePipeline:
        def __init__(self, config: object, save_cfg_dir: Path) -> None:
            self.config, self.save_cfg_dir = config, save_cfg_dir
            loaded_pipelines.append(self)

        def setup(self) -> None:
            import gr00t.model.gr00t_n1d7.setup as setup

            self.dataset, _ = setup.DatasetFactory(self.config.training).build(FakeProcessor())

        def return_model(self) -> FakeModel:
            return FakeModel()

        def return_dataset(self) -> tuple[object, None]:
            return self.dataset, None

        def return_collator(self):
            return lambda batch: batch

    gr00t = types.ModuleType("gr00t")
    gr00t.__path__ = []  # type: ignore[attr-defined]
    configs = types.ModuleType("gr00t.configs")
    configs.__path__ = []  # type: ignore[attr-defined]
    config_base = types.ModuleType("gr00t.configs.base_config")
    config_base.get_default_config = FakePinnedConfig
    data = types.ModuleType("gr00t.data")
    data.__path__ = []  # type: ignore[attr-defined]
    embodiment_tags = types.ModuleType("gr00t.data.embodiment_tags")
    embodiment_tags.EmbodimentTag = SimpleNamespace(
        resolve=lambda _tag: SimpleNamespace(value="new_embodiment")
    )
    model = types.ModuleType("gr00t.model")
    model.__path__ = []  # type: ignore[attr-defined]
    registry = types.ModuleType("gr00t.model.registry")
    registry.MODEL_REGISTRY = SimpleNamespace(get=lambda config_type: FakePipeline if config_type is FakeModelConfig else None)
    n1d7 = types.ModuleType("gr00t.model.gr00t_n1d7")
    n1d7.__path__ = []  # type: ignore[attr-defined]
    setup = types.ModuleType("gr00t.model.gr00t_n1d7.setup")
    setup.DatasetFactory = object
    for name, module in {
        "gr00t": gr00t,
        "gr00t.configs": configs,
        "gr00t.configs.base_config": config_base,
        "gr00t.data": data,
        "gr00t.data.embodiment_tags": embodiment_tags,
        "gr00t.model": model,
        "gr00t.model.registry": registry,
        "gr00t.model.gr00t_n1d7": n1d7,
        "gr00t.model.gr00t_n1d7.setup": setup,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    class FakeSampler:
        def close(self) -> None:
            return None

    checkout = tmp_path / "official"
    entrypoint = checkout / "gr00t" / "experiment" / "launch_finetune.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("# fixture\n", encoding="utf-8")
    monkeypatch.setenv("LEHOME_GROOT_ROOT", str(checkout))
    monkeypatch.setattr(launch, "_checkout_head", lambda *_args: ISAAC_GROOT_REVISION)
    monkeypatch.setattr(launch, "_checkout_is_clean", lambda *_args: True)
    monkeypatch.setattr(launch, "policy_artifact_sha256", lambda _path: LIFECYCLE.PARENT_CHECKPOINT["artifact_sha256"])
    monkeypatch.setattr(production, "_visible_device", lambda _count: "0")
    monkeypatch.setattr(production, "NvmlTelemetrySampler", FakeSampler)
    monkeypatch.delitem(sys.modules, "modality", raising=False)
    return loaded_pipelines, created_loaders


def test_runtime_warmup_stage_copies_the_canonical_launch_into_the_real_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime_mounts: dict[str, Path],
) -> None:
    """The real stage and real config loader agree on h16/batch-64 mixture launch bytes."""

    from test_runtime_mixture import _contract
    import lehome_train.groot.production_runtime as production

    prepared, cache, output = (
        runtime_mounts["prepared"], runtime_mounts["cache"], runtime_mounts["output"]
    )
    monkeypatch.setattr(production, "_ALLOWED_ROOTS", (prepared, output, cache))
    manifest, windows, mounts = _contract(prepared / "runtime")
    normalization = manifest.parent / "mixture-normalization.json"
    (prepared / "code").mkdir()
    (prepared / "config" / "modality.py").parent.mkdir(parents=True, exist_ok=True)
    (prepared / "config" / "modality.py").write_text("# fixture\n", encoding="utf-8")
    (cache / "parent").mkdir(parents=True)

    instance = {
        "kind": "runtime_mixture_gpu_warmup_instance", "instance_id": 44,
        "host": "fixture", "port": 22, "platform_arch": "x86_64",
        "trainer_image": LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE,
        "provider_response_sha256": "1" * 64, "capability_sha256": "2" * 64,
    }
    cpu_instance = {
        "instance_id": 43, "provider_response_sha256": "6" * 64,
        "platform_arch": "x86_64", "trainer_image": LIFECYCLE.RUNTIME_CPU_PILOT_IMAGE,
        "image_digest": LIFECYCLE.RUNTIME_CPU_PILOT_IMAGE.rpartition("@")[2],
    }
    code_revision, code_sha256 = "3" * 40, "4" * 64
    receipts = _campaign_receipts(tmp_path)
    binding = {
        "mixture": {
            "repository": "ryanjin333/lehome-groot-n17-data", "revision": "a" * 40,
            "mixture_id": "d" * 64, "manifest_sha256": sha256_file(manifest),
            "window_index_sha256": sha256_file(windows),
            "normalization_sha256": sha256_file(normalization),
            "source_revisions": {"bc": "b" * 40, "round-1": "c" * 40},
        },
        "deployment": {
            "oci_image_digest": LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE.rpartition("@")[2],
            "provider": "vast", "capability_sha256": instance["capability_sha256"],
        },
        "code": {
            "repository_revision": code_revision, "bundle_sha256": code_sha256,
            "isaac_groot_revision": "5" * 40,
        },
        "parent_checkpoint": {
            "repository": LIFECYCLE.PARENT_CHECKPOINT["repository"],
            "revision": LIFECYCLE.PARENT_CHECKPOINT["revision"],
            "subpath": "policies/step-12000",
            "artifact_sha256": LIFECYCLE.PARENT_CHECKPOINT["artifact_sha256"],
        },
        "physical_batch_size": 64, "action_horizon": 16,
    }
    pilot = _cpu_pilot(instance=cpu_instance, code_revision=code_revision, code_sha256=code_sha256)
    launch = {
        "base_model_path": str(cache / "parent"), "base_model_revision": MODEL_REVISION,
        "dataset_path": str(prepared / "runtime"), "dataset_revision": "6" * 40,
        "modality_config_path": str(prepared / "config" / "modality.py"), "output_dir": str(output / "run"),
        "experiment_name": "runtime-mixture-70-30", "physical_batch_size": 64,
        "global_batch_size": 64, "max_steps": 2000, "save_steps": 1000,
        "warmup_ratio": 0.05, "dataloader_num_workers": 4,
        "training_action_horizon": 16, "model_action_chunk_capacity": 40,
        "parent_checkpoint_repository": LIFECYCLE.PARENT_CHECKPOINT["repository"],
        "parent_checkpoint_revision": LIFECYCLE.PARENT_CHECKPOINT["revision"],
        "parent_checkpoint_subpath": "policies/step-12000",
        "parent_checkpoint_artifact_sha256": LIFECYCLE.PARENT_CHECKPOINT["artifact_sha256"],
        "runtime_mixture_manifest": str(manifest), "runtime_window_index": str(windows),
        "runtime_mounts_descriptor": str(mounts),
    }
    pilot_path = _write(tmp_path / "stage" / "pilot.json", pilot)
    binding_path = _write(tmp_path / "stage" / "binding.json", binding)
    warmup_request = _write(tmp_path / "stage" / "runtime-warmup.json", {
        "schema_version": 1, "command": "runtime-gpu-warmup",
        "arguments": {"cpu_pilot": pilot, "binding": binding},
    })
    launch_path = _write(tmp_path / "stage" / "warmup-launch.json", launch)
    bootstrap = _write(tmp_path / "stage" / "bootstrap.json", {
        "schema_version": 1, "kind": "runtime_mixture_bootstrap_stage",
        "instance_id": cpu_instance["instance_id"],
        "provider_response_sha256": cpu_instance["provider_response_sha256"],
        "platform_arch": "x86_64",
        "trainer_image": LIFECYCLE.RUNTIME_CPU_PILOT_IMAGE,
        "image_digest": LIFECYCLE.RUNTIME_CPU_PILOT_IMAGE.rpartition("@")[2],
        "code_revision": code_revision, "code_bundle_sha256": code_sha256,
        "bc_revision": "b" * 40, "rollout_revision": "c" * 40,
        "deployment_revision": "a" * 40,
        "bc_receipt_sha256": sha256_file(receipts["bc"]),
        "rollout_receipt_sha256": sha256_file(receipts["rollout"]),
        "deployment_receipt_sha256": sha256_file(receipts["deployment"]),
        "transfers": [{"name": "code.bundle", "sha256": "7" * 64}],
    })
    stage_receipt = tmp_path / "stage" / "warmup-stage.json"
    runner, calls = _local_stage_transport(tmp_path / "remote", runtime_mounts)
    report = LIFECYCLE.runtime_mixture_warmup_stage(
        instance=instance,
        request={
            "code_revision": code_revision, "code_bundle_sha256": code_sha256,
            "bc_readback_receipt": str(receipts["bc"]),
            "rollout_readback_receipt": str(receipts["rollout"]),
            "deployment_receipt": str(receipts["deployment"]),
            "bootstrap_receipt": str(bootstrap), "pilot_receipt": str(pilot_path),
            "runtime_warmup_binding": str(binding_path),
            "runtime_warmup_request": str(warmup_request),
            "warmup_launch_config": str(launch_path),
            "warmup_stage_receipt": str(stage_receipt),
        },
        runner=runner,
    )

    canonical_launch = prepared / "config" / "launch.json"
    assert report["action"] == "runtime-warmup-stage"

    loaded_pipelines, created_loaders = _install_low_level_runtime_fakes(monkeypatch, tmp_path)
    session = production.ProductionRuntime()._create_runtime_gpu_warmup_session(
        {"cpu_pilot": pilot, "binding": binding}
    )
    loader = session.loader_factory(4)

    assert session.model_loaded() is True
    assert json.loads(canonical_launch.read_text(encoding="utf-8")) == launch
    assert not (prepared / "warmup" / "launch.json").exists()
    assert all("/prepared/warmup/launch.json" not in command[-1] for command in calls if command[0] == "ssh")
    assert launch["training_action_horizon"] == 16
    assert launch["model_action_chunk_capacity"] == 40
    assert launch["physical_batch_size"] == launch["global_batch_size"] == 64
    assert launch["dataloader_num_workers"] == 4
    assert launch["dataset_path"] == str(prepared / "runtime")
    assert binding["mixture"] == {
        "repository": "ryanjin333/lehome-groot-n17-data", "revision": "a" * 40,
        "mixture_id": "d" * 64, "manifest_sha256": sha256_file(manifest),
        "window_index_sha256": sha256_file(windows),
        "normalization_sha256": sha256_file(normalization),
        "source_revisions": {"bc": "b" * 40, "round-1": "c" * 40},
    }
    assert len(loaded_pipelines) == 1
    assert created_loaders == [loader]
    assert loader.kwargs["batch_size"] == 64
    assert loader.kwargs["num_workers"] == 4
    assert loader.kwargs["persistent_workers"] is True
