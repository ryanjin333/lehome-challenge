"""Literal dispatch-to-runtime proof for the persistent runtime checkpoint path.

This test intentionally keeps the paid/provider boundary in memory.  It must
still invoke the lifecycle script parser and the production runtime's real
continuous supervisor, checkpoint packager, anchor publisher and recovery
gates.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

from lehome_train.hub import HubAccess, HubTreeEntry
from lehome_train.io import sha256_file


REPOSITORY = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "literal_runtime_cli_under_test",
    REPOSITORY / "scripts" / "run_groot_persistent_training.py",
)
assert SPEC is not None and SPEC.loader is not None
LIFECYCLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LIFECYCLE
SPEC.loader.exec_module(LIFECYCLE)


def test_production_runtime_accepts_only_an_explicit_checkpoint_byte_transport() -> None:
    """The production default stays real; tests may supply only Hub bytes."""
    from lehome_train.groot.production_runtime import ProductionRuntime

    transport = object()
    runtime = ProductionRuntime(checkpoint_transport_factory=lambda **_kwargs: transport)

    assert runtime._checkpoint_transport_factory(timeout_seconds=30.0) is transport


def test_lifecycle_parser_accepts_only_an_explicit_hub_transport_factory(
    tmp_path: Path,
) -> None:
    """The script's executable dispatch must expose the same byte-only seam."""
    request = tmp_path / "request.json"
    request.write_text('{"instance":{"instance_id":44}}', encoding="utf-8")
    transport = object()

    result = LIFECYCLE.main_for_test(
        ["runtime-checkpoint-replacement-resume", "--request", str(request)],
        transport_factory=lambda **_kwargs: transport,
    )

    assert result["dry_run"] is True


class _MemoryHub:
    """The permitted external boundary: immutable Hub bytes only."""

    def __init__(self) -> None:
        self.objects: dict[str, dict[str, bytes]] = {}
        self.main: str | None = None
        self.uploaded: list[tuple[str, str]] = []

    def check_access(self, *, repository: str, token: str) -> HubAccess:
        assert repository == LIFECYCLE.PARENT_CHECKPOINT["repository"] and token == "fake-token"
        return HubAccess(can_read=True, can_write=True, private_repository=True)

    def upload_files(self, *, repository: str, revision: str, source: Path, entries, token: str, remote_prefix: str | None = None) -> str:
        assert repository == LIFECYCLE.PARENT_CHECKPOINT["repository"] and revision == "main"
        assert remote_prefix is not None and token == "fake-token"
        immutable = f"{len(self.objects) + 1:040x}"
        self.objects[immutable] = {
            f"{remote_prefix}/{entry.relative_path}": (source / entry.relative_path).read_bytes()
            for entry in entries
        }
        self.uploaded.append((immutable, remote_prefix))
        if remote_prefix.startswith("checkpoints/"):
            self.main = immutable
        return immutable

    def list_tree(self, *, repository: str, revision: str, token: str) -> tuple[HubTreeEntry, ...]:
        assert repository == LIFECYCLE.PARENT_CHECKPOINT["repository"] and token == "fake-token"
        return tuple(HubTreeEntry(path, "file") for path in sorted(self.objects[revision]))

    def download_files(self, *, repository: str, revision: str, destination: Path, relative_paths, token: str, remote_prefix: str | None = None) -> str:
        assert repository == LIFECYCLE.PARENT_CHECKPOINT["repository"] and token == "fake-token"
        assert remote_prefix is not None
        for relative in relative_paths:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.objects[revision][f"{remote_prefix}/{relative}"])
        return revision

    def resolve_approved_ref(self, *, repository: str, ref: str, token: str) -> str:
        assert repository == LIFECYCLE.PARENT_CHECKPOINT["repository"] and ref == "main" and token == "fake-token"
        assert self.main is not None
        return self.main


class _WarmupAdapter:
    def runtime_state(self):
        from lehome_train.groot.runtime_mixture_warmup import RuntimeState
        return RuntimeState(torch_cuda_available=True, torch_cuda_initialized=True, model_loaded=True)

    def measure(self, *, worker_count: int, burn_in_steps: int, measured_steps: int):
        from lehome_train.groot.runtime_mixture_warmup import GpuWarmupMeasurement
        return GpuWarmupMeasurement(
            decoded_samples=64 * (burn_in_steps + measured_steps), measured_steps=measured_steps,
            loader_wait_seconds=20.0 if worker_count == 0 else 1.0, step_seconds=20.0, gpu_busy_seconds=18.0,
            gpu_utilization_percent=90.0, oom=False, error=None,
        )


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_runtime_supervisor_packages_and_anchors_the_real_one_k_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real runtime orchestration publishes a readback-verified 1K anchor."""
    from test_runtime_mixture import _contract
    from lehome_train.constants import MODEL_REVISION
    from lehome_train.groot.continuous_training import ProviderInterrupted
    from lehome_train.groot.runtime_mixture_warmup import build_gpu_warmup_receipt
    import lehome_train.groot.production_runtime as production
    from lehome_train.models import ExperimentConfig

    prepared, output, cache = tmp_path / "prepared", tmp_path / "output", tmp_path / "cache"
    monkeypatch.setattr(production, "_ALLOWED_ROOTS", (prepared, output, cache))
    manifest, windows, mounts = _contract(prepared / "runtime")
    normalization = manifest.parent / "mixture-normalization.json"
    bc_receipt = manifest.parent / "source-publication" / "bc-readback.json"
    rollout_receipt = manifest.parent / "source-publication" / "rollout-readback.json"
    deployment_receipt = manifest.parent / "release-receipt.json"
    mixture_id = "d" * 64
    (cache / "parent").mkdir(parents=True)
    modality = prepared / "config" / "modality.py"; modality.parent.mkdir(parents=True); modality.write_text("# fixture\n")
    run_root = output / "run"
    code_revision, code_sha = "1" * 40, "2" * 64
    binding = {
        "mixture": {"repository": "ryanjin333/lehome-groot-n17-data", "revision": "a" * 40, "mixture_id": mixture_id,
                    "manifest_sha256": sha256_file(manifest), "window_index_sha256": sha256_file(windows),
                    "normalization_sha256": sha256_file(normalization), "source_revisions": {"bc": "b" * 40, "round-1": "c" * 40}},
        "deployment": {"oci_image_digest": LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE.rpartition("@")[2], "provider": "vast", "capability_sha256": "3" * 64},
        "code": {"repository_revision": code_revision, "bundle_sha256": code_sha, "isaac_groot_revision": "4" * 40},
        "parent_checkpoint": {"repository": LIFECYCLE.PARENT_CHECKPOINT["repository"], "revision": LIFECYCLE.PARENT_CHECKPOINT["revision"], "subpath": "policies/step-12000", "artifact_sha256": LIFECYCLE.PARENT_CHECKPOINT["artifact_sha256"]},
        "physical_batch_size": 64, "action_horizon": 16,
    }
    pilot = {"schema_version": 4, "kind": "runtime_mixture_loader_pilot", "model_loaded": False, "gpu_initialized": False,
             "processor_contract": "pinned_processor_integration_required", "representative": {"three_cameras": True, "action_horizon": 16},
             "sample_count_per_worker": 100, "worker_counts": [0, 4, 8, 16, 24], "canonical_worker_counts": [0, 4, 8, 16, 24],
             "loader_throughput": {str(i): {"decoded_samples": 100, "samples_per_second": 1.0} for i in (0, 4, 8, 16, 24)},
             "timing_rows": [{"worker_count": i, "decoded_samples": 100, "seconds": 1.0, "samples_per_second": 1.0, "host_cpu_seconds": 1.0, "host_max_rss_mib": 1.0, "latency_seconds_p50": 0.1, "latency_seconds_p95": 0.2} for i in (0, 4, 8, 16, 24)],
             "authenticated_evidence": {"provider_instance_id": 10, "provider_response_sha256": "5" * 64, "platform_arch": "x86_64", "image_digest": LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE.rpartition("@")[2], "code_revision": code_revision, "code_bundle_sha256": code_sha, "bc_revision": "b" * 40, "rollout_revision": "c" * 40, "deployment_revision": "a" * 40}, "cache_cap": 1, "native_x86_required": True, "timeout_seconds": 60.0, "canonical_completion": True}
    warmup = build_gpu_warmup_receipt(cpu_pilot=pilot, binding=binding, adapter=_WarmupAdapter())
    launch = {"base_model_path": str(cache / "parent"), "base_model_revision": MODEL_REVISION, "dataset_path": str(manifest.parent), "dataset_revision": "6" * 40,
              "modality_config_path": str(modality), "output_dir": str(run_root), "experiment_name": "runtime-mixture-70-30", "physical_batch_size": 64, "global_batch_size": 64, "max_steps": 2000, "save_steps": 1000, "warmup_ratio": .05, "dataloader_num_workers": 4,
              "parent_checkpoint_repository": LIFECYCLE.PARENT_CHECKPOINT["repository"], "parent_checkpoint_revision": LIFECYCLE.PARENT_CHECKPOINT["revision"], "parent_checkpoint_subpath": "policies/step-12000", "parent_checkpoint_artifact_sha256": LIFECYCLE.PARENT_CHECKPOINT["artifact_sha256"], "runtime_mixture_manifest": str(manifest), "runtime_window_index": str(windows), "runtime_mounts_descriptor": str(mounts)}
    experiment = ExperimentConfig(repository_commit=code_revision, container_digest="sha256:" + "7" * 64, model_repository="nvidia/GR00T-N1.7-3B", model_revision=MODEL_REVISION, dataset_repository="ryanjin333/lehome-groot-n17-data", dataset_revision="6" * 40, dataset_manifest_sha256=mixture_id, physical_batch_size=64, gradient_accumulation_steps=1, sample_presentations=128_000, action_horizon=16, tune_language_backbone=False, tune_visual_backbone=False)
    identity = {"mixture_id": mixture_id, "deployment_receipt_sha256": sha256_file(deployment_receipt), "source_revisions": [{"source_id": "organizer", "immutable_revision": "b" * 40, "prefix": "bc/full", "tree_sha256": sha256_file(bc_receipt)}, {"source_id": "rollout", "immutable_revision": "c" * 40, "prefix": "rollouts/round-1", "tree_sha256": sha256_file(rollout_receipt)}], "schedule_seed": 17, "code_bundle_sha256": code_sha, "code_bundle_revision": code_revision, "oci_image": LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE.rpartition("@")[2], "parent_step12000_artifact_sha256": LIFECYCLE.PARENT_CHECKPOINT["artifact_sha256"], "physical_batch_size": 64, "action_horizon": 16}
    paths = {"launch_config": _write(prepared / "config" / "launch.json", launch), "experiment_config": _write(prepared / "config" / "experiment.json", experiment.to_dict()), "runtime_manifest": manifest, "runtime_window_index": windows, "runtime_normalization": normalization, "runtime_mounts_descriptor": mounts, "runtime_source_evidence": _write(prepared / "runtime" / "source-evidence.json", identity), "cpu_pilot_receipt": _write(prepared / "config" / "pilot.json", pilot), "warmup_receipt": _write(prepared / "config" / "warmup.json", warmup), "runtime_warmup_binding": _write(prepared / "config" / "binding.json", binding)}
    token = prepared / "config" / "runtime.token"; token.write_text("fake-token\n"); token.chmod(0o600)
    hub = _MemoryHub()
    def launch_one_k(config, **_kwargs):
        checkpoint = Path(config.output_dir) / config.experiment_name / "checkpoint-1000"; checkpoint.mkdir(parents=True)
        (checkpoint / "weights.bin").write_bytes(b"immutable-1k")
        _write(checkpoint / "trainer_state.json", {"global_step": 1000, "log_history": [{"step": 1000, "loss": .2}]})
        run = checkpoint.parent; _write(run / "lehome_launch.json", config.identity())
        raise ProviderInterrupted("provider loss after durable 1K")
    monkeypatch.setattr(production, "launch_continuous_finetune", launch_one_k)
    result = production.ProductionRuntime(checkpoint_transport_factory=lambda **_kwargs: hub).runtime_mixture_train({**{key: str(value) for key, value in paths.items()}, "runtime_resume_archive": None, "runtime_resume_descriptor": None, "runtime_resume_cursor": None, "runtime_resume_anchor": None, "runtime_resume_publication": None, "checkpoint_repository": LIFECYCLE.PARENT_CHECKPOINT["repository"], "checkpoint_revision": "main", "publisher_token_file": str(token), "instance_id": 10, "result_output": str(output / "result.json"), "status_output": str(output / "status.json")})
    publication = result["immutable_checkpoint_publications"][0]
    assert result["status"] == "runtime-mixture-interrupted" and publication["optimizer_step"] == 1000
    assert publication["kind"] == "runtime_mixture_checkpoint_publication"
    assert publication["identity"] == identity
    assert publication["runtime_cursor"] == {"optimizer_step": 1000, "global_sample_offset": 64_000, "physical_batch_size": 64, "action_horizon": 16}
    assert publication["runtime_checkpoint_anchor"]["readback_verified"] is True
    assert publication["readback_verified"] is True
    assert publication["runtime_checkpoint_anchor"]["immutable_anchor_revision"] == hub.main
    assert all("continuous-train" not in prefix and "generation" not in prefix for _, prefix in hub.uploaded)

    # Host output is deliberately discarded.  The executable action resolves
    # the known ``main`` ref and its immutable 1K anchor after two absence
    # readbacks, rather than receiving a process result as resume evidence.
    lost_terminal = tmp_path / "lost-terminal.json"
    interruption = {
        "instance": {"instance_id": 10, "platform_arch": "x86_64", "trainer_image": LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE},
        "checkpoint_repository": LIFECYCLE.PARENT_CHECKPOINT["repository"], "checkpoint_anchor_ref": "main", "checkpoint_experiment_id": "runtime-mixture-70-30",
        "experiment_config": str(paths["experiment_config"]), "runtime_source_evidence": str(paths["runtime_source_evidence"]), "expected_checkpoint_steps": [1000], "terminal_receipt": str(lost_terminal),
    }
    interruption_path = _write(tmp_path / "interruption.json", interruption)
    terminal = LIFECYCLE.main_for_test(
        ["runtime-checkpoint-interrupted", "--request", str(interruption_path), "--execute", "--token-file", str(token)],
        runner=lambda command: "{}" if command[:4] == ("vastai", "--raw", "show", "instance") else (_ for _ in ()).throw(AssertionError(command)),
        transport_factory=lambda **_kwargs: hub,
    )["terminal"]
    assert terminal["resumable_checkpoint_step"] == 1000

    replacement = {"kind": "runtime_mixture_gpu_warmup_instance", "instance_id": 11, "host": "replacement", "port": 22, "platform_arch": "x86_64", "trainer_image": LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE, "provider_response_sha256": "b" * 64, "capability_sha256": "3" * 64}
    resume_destination = prepared / "resume-download"
    replacement_request = {
        "instance": replacement, "code_revision": code_revision, "code_bundle_sha256": code_sha, "schedule_seed": 17,
        "bc_readback_receipt": str(bc_receipt), "rollout_readback_receipt": str(rollout_receipt), "deployment_receipt": str(deployment_receipt),
        "checkpoint_experiment_id": "runtime-mixture-70-30", "experiment_config": str(paths["experiment_config"]),
        "runtime_source_evidence": str(paths["runtime_source_evidence"]), "terminal_receipt": str(lost_terminal),
        "resume_destination": str(resume_destination), "failure_receipt": str(tmp_path / "resume-failure.json"),
    }
    resumed = LIFECYCLE.main_for_test(
        ["runtime-checkpoint-replacement-resume", "--request", str(_write(tmp_path / "replacement.json", replacement_request)), "--execute", "--token-file", str(token)],
        runner=lambda command: (_ for _ in ()).throw(AssertionError(command)), transport_factory=lambda **_kwargs: hub,
    )
    assert resumed["instance_id"] == 11 and resumed["runtime_cursor"]["global_sample_offset"] == 64_000
    assert resumed["runtime_resume_anchor"] == {"immutable_anchor_revision": hub.main, "anchor_sha256": publication["runtime_checkpoint_anchor"]["anchor_sha256"]}

    # A replacement host has no durable trainer output.  It consumes only the
    # discovered immutable archive and descriptor, then the real supervisor
    # observes/packages/publishes the 2K completion boundary.
    shutil.rmtree(run_root)
    resume_archive = Path(resumed["checkpoint_archive"])
    resume_descriptor = Path(resumed["checkpoint_descriptor"])
    from lehome_train.checkpoints import load_checkpoint_descriptor
    resume_record = load_checkpoint_descriptor(resume_descriptor).record
    assert resume_record.dataset_manifest_sha256 == mixture_id
    assert resume_record.sample_presentations == 64_000
    assert resume_record.artifact.sha256 == sha256_file(resume_archive)
    def launch_two_k(config, **_kwargs):
        checkpoint = Path(config.output_dir) / config.experiment_name / "checkpoint-2000"; checkpoint.mkdir(parents=True)
        (checkpoint / "weights.bin").write_bytes(b"immutable-2k")
        _write(checkpoint / "trainer_state.json", {"global_step": 2000, "log_history": [{"step": 2000, "loss": .1}]})
    monkeypatch.setattr(production, "launch_continuous_finetune", launch_two_k)
    completed = production.ProductionRuntime(checkpoint_transport_factory=lambda **_kwargs: hub).runtime_mixture_train({**{key: str(value) for key, value in paths.items()}, "runtime_resume_archive": str(resume_archive), "runtime_resume_descriptor": str(resume_descriptor), "runtime_resume_cursor": resumed["runtime_cursor"], "runtime_resume_anchor": resumed["runtime_resume_anchor"], "runtime_resume_publication": resumed["runtime_resume_publication"], "checkpoint_repository": LIFECYCLE.PARENT_CHECKPOINT["repository"], "checkpoint_revision": "main", "publisher_token_file": str(token), "instance_id": 11, "result_output": str(output / "replacement-result.json"), "status_output": str(output / "replacement-status.json")})
    publications = completed["immutable_checkpoint_publications"]
    assert completed["status"] == "runtime-mixture-complete" and [item["optimizer_step"] for item in publications] == [1000, 2000]
    assert publications[1]["runtime_checkpoint_anchor"]["readback_verified"] is True
    assert publications[1]["runtime_checkpoint_anchor"]["immutable_anchor_revision"] == hub.main

    completion_path = tmp_path / "complete-terminal.json"
    completion_request = {**replacement_request, "instance": replacement, "runtime_checkpoint_publications": publications, "terminal_receipt": str(completion_path), "failure_receipt": str(tmp_path / "complete-failure.json")}
    complete = LIFECYCLE.main_for_test(
        ["runtime-checkpoint-complete", "--request", str(_write(tmp_path / "complete.json", completion_request)), "--execute", "--token-file", str(token)],
        runner=lambda command: (_ for _ in ()).throw(AssertionError(command)), transport_factory=lambda **_kwargs: hub,
    )["terminal"]
    assert complete["disposable"] is True and complete["resumable_checkpoint_step"] == 2000

    destroyed = False
    def destroy_runner(command: tuple[str, ...]) -> str:
        nonlocal destroyed
        if command == ("vastai", "destroy", "instance", "11", "--yes"):
            destroyed = True; return ""
        if command == ("vastai", "--raw", "show", "instance", "11"):
            return "{}" if destroyed else '{"id":11}'
        raise AssertionError(command)
    dispose_request = {**replacement_request, "instance": replacement, "terminal_receipt": str(completion_path), "failure_receipt": str(tmp_path / "dispose-failure.json")}
    disposed = LIFECYCLE.main_for_test(
        ["runtime-checkpoint-dispose", "--request", str(_write(tmp_path / "dispose.json", dispose_request)), "--execute", "--token-file", str(token)],
        runner=destroy_runner, transport_factory=lambda **_kwargs: hub,
    )
    assert destroyed and disposed["destroy_authorized"] is True and disposed["instance_id"] == 11
