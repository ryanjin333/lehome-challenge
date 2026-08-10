from __future__ import annotations

import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from lehome_train.b1k.lifecycle import LifecycleController, main
from lehome_train.b1k.launch import B1KLaunch, B1KLaunchFailure
from lehome_train.b1k import production
from lehome_train.b1k.training import approved_launch_plans


@pytest.mark.parametrize("visible", ["0", "0,1", "0,1,2", "0,1,2,3"])
def test_production_world_size_matches_each_visible_gpu_rank(visible: str) -> None:
    values = {"CUDA_VISIBLE_DEVICES": visible}

    assert production._world_size(values) == len(visible.split(","))
    assert values["CUDA_VISIBLE_DEVICES"] == visible


@pytest.mark.parametrize("visible", ["", "0,1,2,3,4", "0,0", "0,,1", "0, 1"])
def test_production_world_size_rejects_zero_or_more_than_four_visible_gpus(visible: str) -> None:
    values = {"CUDA_VISIBLE_DEVICES": visible}

    with pytest.raises(ValueError):
        production._world_size(values)


@pytest.mark.parametrize("resume_policy", ("auto", "never", "require"))
def test_production_controller_preserves_the_selected_resume_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, resume_policy: str
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("RESUME_POLICY", resume_policy)

    controller = production.build_production_controller(SimpleNamespace(output=tmp_path))

    assert controller.resume_policy == resume_policy


def test_main_loads_the_onstart_namespaced_adapter_without_global_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Paths:
        output = tmp_path

    monkeypatch.setattr("lehome_train.b1k.lifecycle.production_preflight", lambda: Paths())
    script = (Path(__file__).parents[2] / "b1k_launchkit" / "onstart.sh").read_text(encoding="utf-8")
    adapter = re.search(r"B1K_LIFECYCLE_ADAPTER=([^\s]+)", script)
    assert adapter is not None
    monkeypatch.setenv("B1K_LIFECYCLE_ADAPTER", adapter.group(1))
    monkeypatch.setattr(
        "lehome_train.b1k.production.build_production_controller",
        lambda _paths: LifecycleController(
            run_training=lambda _plan, _resume, published: [published(step) for step in range(1_000, 15_001, 1_000)] and 15_000,
            publish_checkpoint=lambda _step: None,
            world_size=1,
            output=tmp_path,
            finalize=lambda: {"immutable_commit": "f" * 40},
        ),
    )

    assert main([]) == 0


def test_onstart_exports_the_production_adapter_without_exposing_the_token() -> None:
    script = (Path(__file__).parents[2] / "b1k_launchkit" / "onstart.sh").read_text(encoding="utf-8")
    assert "B1K_LIFECYCLE_ADAPTER=lehome_train.b1k.production:build_production_controller" in script
    assert "HF_TOKEN=" not in script.split("exec setpriv", 1)[1]
    assert ">> /workspace/logs/controller.log 2>&1" in script
    assert "lehome_train.b1k.lifecycle > /workspace/logs/controller.log 2>&1" not in script
    assert "token_bootstrap.py" in script
    assert script.index("token_bootstrap.py") < script.index("exec setpriv")
    assert "mktemp /workspace/.cache/huggingface" not in script
    assert "B1K_TRAINING_SMOKE_RUNTIME" in script
    assert script.index("B1K_TRAINING_SMOKE_RUNTIME") < script.index("exec setpriv")
    assert ": > /workspace/smoke-canary/training-ready" in script
    assert ": > /workspace/.b1k-training-smoke-ready" not in script
    assert "WANDB_DIR=/workspace/logs/wandb" in script
    assert script.index("WANDB_DIR=/workspace/logs/wandb") < script.index("exec setpriv")
    assert "/workspace/.cache/triton" in script
    assert "TRITON_CACHE_DIR=/workspace/.cache/triton" in script
    assert script.index("TRITON_CACHE_DIR=/workspace/.cache/triton") < script.index("exec setpriv")


def test_deploy_modality_passes_the_pinned_root_as_the_positional_argument(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset = tmp_path / "b1k"; dataset.mkdir(); checkout = tmp_path / "checkout"
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []
    monkeypatch.setattr(production, "_CHECKOUT", checkout)

    def run(argv: tuple[str, ...], **kwargs: object) -> SimpleNamespace:
        calls.append((argv, kwargs))
        modality = dataset / "meta" / "modality.json"; modality.parent.mkdir(); modality.write_text("{}")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(production.subprocess, "run", run)
    assert production._deploy_modality(dataset) == dataset / "meta" / "modality.json"
    assert calls[0][0] == (production.sys.executable, str(checkout / "scripts/b1k/deploy_modality.py"), str(dataset))
    assert "cwd" not in calls[0][1]


def test_runtime_store_hash_is_identical_for_clean_and_resume_launches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    paths = SimpleNamespace(output=tmp_path / "output")
    paths.output.mkdir()
    runtime = production._Runtime(
        paths=paths,
        values={"RUN_ID": "b1k-run-001"},
        hub=SimpleNamespace(),
        world_size=1,
        bootstrap_result=SimpleNamespace(dataset=tmp_path / "dataset", derived_model=tmp_path / "model", offline_environment=lambda: {"WANDB_MODE": "offline", "WANDB_DIR": "/workspace/logs/wandb", "TRITON_CACHE_DIR": "/workspace/.cache/triton"}),
    )
    plan = approved_launch_plans(num_gpus=1)[0]
    hashes: list[str] = []
    def build(_plan: object, **kwargs: object) -> B1KLaunch:
        resume = kwargs["resume_from_checkpoint"]
        command = ("torchrun", "--resume" if resume else "--fresh")
        return B1KLaunch(command=command, environment={}, arguments_sha256="b" * 64 if resume else "a" * 64)
    class Store:
        def verified_steps(self) -> tuple[int, ...]: return ()
    monkeypatch.setattr(production, "build_b1k_launch", build)
    monkeypatch.setattr(production._Runtime, "_store", lambda _self, _plan, launch_hash: hashes.append(launch_hash) or Store())
    monkeypatch.setattr(production, "run_b1k_launch_with_checkpoint_watch", lambda *_args, **_kwargs: 15_000)

    assert runtime.run_training(plan, False, lambda _step: None) == 15_000
    assert runtime.run_training(plan, True, lambda _step: None) == 15_000
    assert hashes == ["a" * 64, "a" * 64]


def test_runtime_launch_uses_bootstrap_offline_environment_and_the_pinned_nested_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "output"; output.mkdir()
    runtime = production._Runtime(
        paths=SimpleNamespace(output=output),
        values={"RUN_ID": "b1k-run-001", "CUDA_VISIBLE_DEVICES": "0"},
        hub=SimpleNamespace(),
        world_size=1,
        bootstrap_result=SimpleNamespace(
            dataset=tmp_path / "dataset",
            derived_model=tmp_path / "model",
            offline_environment=lambda: {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "WANDB_MODE": "offline", "WANDB_DIR": "/workspace/logs/wandb", "TRITON_CACHE_DIR": "/workspace/.cache/triton"},
        ),
    )
    observed: dict[str, object] = {}
    def build(_plan: object, **kwargs: object) -> B1KLaunch:
        observed.update(kwargs)
        return B1KLaunch(command=("torchrun",), environment=kwargs["environment"], arguments_sha256="a" * 64)
    monkeypatch.setattr(production, "build_b1k_launch", build)

    runtime._launch(approved_launch_plans(num_gpus=1)[0], resume=False)

    assert observed["environment"]["WANDB_MODE"] == "offline"
    assert observed["environment"]["WANDB_DIR"] == "/workspace/logs/wandb"
    assert observed["environment"]["TRITON_CACHE_DIR"] == "/workspace/.cache/triton"
    assert observed["environment"]["HF_HUB_OFFLINE"] == "1"
    assert observed["output_dir"] == str(output)
    assert runtime._training_output() == output / "b1k-run-001"


def test_runtime_launch_rejects_a_non_workspace_wandb_directory(tmp_path: Path) -> None:
    runtime = production._Runtime(
        paths=SimpleNamespace(output=tmp_path / "output"),
        values={"RUN_ID": "b1k-run-001", "CUDA_VISIBLE_DEVICES": "0"},
        hub=SimpleNamespace(),
        world_size=1,
        bootstrap_result=SimpleNamespace(
            dataset=tmp_path / "dataset",
            derived_model=tmp_path / "model",
            offline_environment=lambda: {"WANDB_MODE": "offline", "WANDB_DIR": "/root/wandb"},
        ),
    )

    with pytest.raises(ValueError, match="WANDB_DIR"):
        runtime._launch(approved_launch_plans(num_gpus=1)[0], resume=False)


def test_step_zero_oom_fallback_records_the_lower_plan_in_final_evidence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output, logs, final, dataset = (tmp_path / name for name in ("output", "logs", "final", "dataset"))
    for directory in (output, logs, final, dataset / "meta"): directory.mkdir(parents=True)
    modality, stats = dataset / "meta" / "modality.json", dataset / "meta" / "stats.json"
    modality.write_text("{}"); stats.write_text("{}")
    selection, materialized, derivation = (tmp_path / name for name in ("selection.json", "materialized.json", "derivation.json"))
    selection.write_text('{"fingerprint":"' + "a" * 64 + '"}')
    materialized.write_text('{"fingerprint":"' + "b" * 64 + '"}')
    derivation.write_text("{}")
    paths = SimpleNamespace(output=output, logs=logs, final=final, dataset=dataset)
    training_output = output / "b1k-run-001"; training_output.mkdir()
    (training_output / "trainer.stdout.log").write_text("trainer stdout\n")
    (training_output / "trainer.stderr.log").write_text("trainer stderr\nHF_TOKEN=super-secret\n")
    (logs / "controller.log").write_text("attempt one\napi_key=controller-secret\n")
    runtime = production._Runtime(
        paths=paths,
        values={"RUN_ID": "b1k-run-001", "CYCLE_ID": "cycle-001", "CONTAINER_DIGEST": "sha256:" + "e" * 64, "RESUME_POLICY": "never"},
        hub=SimpleNamespace(),
        world_size=1,
        bootstrap_result=SimpleNamespace(dataset=dataset, derived_model=tmp_path / "model", selection_manifest=selection, materialized_manifest=materialized, modality_sha256="ca3a2e406472650bee9439ed81a3f4a1531b6fe689cdc1b348d0f260a208e641", stats_sha256="d" * 64, model_derivation=derivation, offline_environment=lambda: {"WANDB_MODE": "offline", "WANDB_DIR": "/workspace/logs/wandb", "TRITON_CACHE_DIR": "/workspace/.cache/triton"}),
        receipt_paths=[],
    )
    plans = approved_launch_plans(num_gpus=1); captured: dict[str, object] = {}; retention_calls: list[tuple[int, ...]] = []
    def build(plan: object, **kwargs: object) -> B1KLaunch:
        return B1KLaunch(command=("torchrun", plan.identity), environment={}, arguments_sha256=chr(ord("a") + plans.index(plan)) * 64)
    class PublishedStore:
        def __init__(self) -> None: self.steps: list[int] = []
        def verified_steps(self) -> tuple[int, ...]: return tuple(self.steps)
        def ensure_newest_two(self) -> tuple[int, ...]:
            retained = tuple(self.steps[-2:]); retention_calls.append(retained); return retained
    class Publisher:
        def __init__(self, **_kwargs: object) -> None: self.store = PublishedStore()
        def publish(self, step: int) -> Path:
            self.store.steps.append(step)
            receipt = training_output / "checkpoint-receipts" / f"step-{step}.json"; receipt.parent.mkdir(exist_ok=True); receipt.write_text("{}")
            return receipt
    class CapturingFinalizer:
        def __init__(self, **_kwargs: object) -> None: pass
        def finalize(self, **kwargs: object) -> dict[str, str]:
            captured["evidence"] = kwargs["evidence"]
            return {"immutable_commit": "f" * 40}
    def watch(launch: B1KLaunch, **kwargs: object) -> int:
        if launch.command[1] == plans[0].identity:
            raise B1KLaunchFailure("CUDA out of memory", optimizer_step=0)
        for step in range(1_000, 15_001, 1_000): kwargs["on_stable_checkpoint"](step)
        return 15_000
    monkeypatch.setattr(production, "read_hf_token", lambda: "not-a-real-token")
    monkeypatch.setattr(production, "build_b1k_launch", build)
    monkeypatch.setattr(production, "LocalCheckpointPublisher", Publisher)
    monkeypatch.setattr(production, "run_b1k_launch_with_checkpoint_watch", watch)
    monkeypatch.setattr(production, "Finalizer", CapturingFinalizer)
    controller = LifecycleController(
        run_training=runtime.run_training,
        publish_checkpoint=runtime.publish_checkpoint,
        world_size=1,
        output=output,
        finalize=runtime.finalize,
        remote_state_exists=lambda: runtime.publisher is not None and bool(runtime.publisher.store.verified_steps()),
    )

    assert controller.run() == 15_000
    contract = json.loads(captured["evidence"].run_contract.read_text())
    assert contract["launch_plan_id"] == plans[1].identity
    assert contract["physical_batch_size"] == plans[1].physical_batch_size
    assert contract["launch_arguments_sha256"] == "b" * 64
    assert contract["resume_policy"] == "never"
    assert {path.name for path in captured["evidence"].logs} == {"controller.log", "trainer.stdout.log", "trainer.stderr.log"}
    controller_log = logs / "controller.log"
    assert "attempt one" in controller_log.read_text()
    assert "controller-secret" not in controller_log.read_text()
    assert "super-secret" not in (training_output / "trainer.stderr.log").read_text()
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in captured["evidence"].logs)
    assert retention_calls == [(14_000, 15_000)]


def test_runtime_resume_discovers_a_lower_batch_plan_restores_into_output_and_records_remote_provenance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    paths = SimpleNamespace(output=tmp_path / "output")
    paths.output.mkdir()
    runtime = production._Runtime(
        paths=paths,
        values={"RUN_ID": "b1k-run-001"},
        hub=SimpleNamespace(),
        world_size=1,
        bootstrap_result=SimpleNamespace(dataset=tmp_path / "dataset", derived_model=tmp_path / "model", offline_environment=lambda: {"WANDB_MODE": "offline", "WANDB_DIR": "/workspace/logs/wandb", "TRITON_CACHE_DIR": "/workspace/.cache/triton"}),
        receipt_paths=[],
    )
    plans = approved_launch_plans(num_gpus=1)
    fallback = plans[1]
    hashes: list[tuple[str, str]] = []
    def build(_plan: object, **kwargs: object) -> B1KLaunch:
        resume = kwargs["resume_from_checkpoint"]
        index = plans.index(_plan)
        return B1KLaunch(command=("torchrun", _plan.identity, "--resume" if resume else "--fresh"), environment={}, arguments_sha256=chr(ord("a") + index) * 64 if not resume else chr(ord("d") + index) * 64)
    class Backend:
        def read_json(self, path: str) -> dict[str, object]:
            assert path == "runs/b1k-run-001/latest.json"
            return {"step": 15_000, "prefix": "verified/b1k-run-001/step-15000/abc", "descriptor_sha256": "d" * 64}
    class Store:
        backend = Backend()
        run_id = "b1k-run-001"
        def __init__(self, plan: object) -> None: self.compatibility = plan.identity
        def inspect_resume_compatibility(self, _policy: object) -> str: return fallback.identity
        def resume(self, _policy: object, destination: Path) -> Path:
            assert destination == paths.output / "b1k-run-001"
            assert self.compatibility == fallback.identity
            return destination / "checkpoint-15000"
        def verified_steps(self) -> tuple[int, ...]: return (14_000, 15_000)
    monkeypatch.setattr(production, "build_b1k_launch", build)
    monkeypatch.setattr(production._Runtime, "_store", lambda _self, plan, launch_hash: hashes.append((plan.identity, launch_hash)) or Store(plan))

    restored = runtime.select_resume()

    assert restored == paths.output / "b1k-run-001" / "checkpoint-15000"
    assert hashes == [(plan.identity, chr(ord("a") + index) * 64) for index, plan in enumerate(plans)] + [(fallback.identity, "b" * 64)]
    assert runtime.launch_arguments == ("torchrun", fallback.identity, "--resume")
    receipt = paths.output / "b1k-run-001" / "checkpoint-receipts" / "resume-step-15000.json"
    assert receipt in runtime.receipt_paths
    assert '"descriptor"' in receipt.read_text()
    assert (paths.output / "b1k-run-001" / "trainer.stdout.log").read_text() == "no local trainer process ran during verified resume\n"
    assert (paths.output / "b1k-run-001" / "trainer.stderr.log").read_text() == "no local trainer process ran during verified resume\n"


@pytest.mark.parametrize("failure", [ValueError("ambiguous verified checkpoint compatibility"), ValueError("incompatible or corrupt verified checkpoint")])
def test_runtime_resume_fails_closed_when_the_remote_namespace_is_ambiguous_or_corrupt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: ValueError) -> None:
    paths = SimpleNamespace(output=tmp_path / "output")
    paths.output.mkdir()
    runtime = production._Runtime(
        paths=paths,
        values={"RUN_ID": "b1k-run-001"},
        hub=SimpleNamespace(),
        world_size=1,
        bootstrap_result=SimpleNamespace(dataset=tmp_path / "dataset", derived_model=tmp_path / "model", offline_environment=lambda: {"WANDB_MODE": "offline", "WANDB_DIR": "/workspace/logs/wandb", "TRITON_CACHE_DIR": "/workspace/.cache/triton"}),
        receipt_paths=[],
    )
    def build(_plan: object, **kwargs: object) -> B1KLaunch:
        return B1KLaunch(command=("torchrun",), environment={}, arguments_sha256="a" * 64)
    class Store:
        compatibility = object()
        def inspect_resume_compatibility(self, _policy: object) -> object: raise failure
    monkeypatch.setattr(production, "build_b1k_launch", build)
    monkeypatch.setattr(production._Runtime, "_store", lambda *_args: Store())

    with pytest.raises(ValueError, match="(?:ambiguous|incompatible|corrupt)"):
        runtime.select_resume()


def test_onstart_leaves_remote_snapshot_destinations_to_the_lifecycle_and_execs_controller() -> None:
    script = (Path(__file__).parents[2] / "b1k_launchkit" / "onstart.sh").read_text(encoding="utf-8")
    for destination in ("/workspace/data/b1k", "/workspace/models/groot-upstream", "/workspace/models/cosmos", "/workspace/models/groot"):
        assert destination not in script
    assert "nohup" not in script
    assert "-x /opt/b1k-bucket-helper/bin/b1k-bucket-helper" in script
    assert "exec /opt/runtime/bin/python -m lehome_train.b1k.lifecycle" in script
