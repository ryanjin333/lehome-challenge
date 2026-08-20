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
from lehome_train.hub import HubRateLimitError, HubTransientError
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
        self.download_counts: dict[str, int] = {}
        self.rate_limit_calls: dict[str, set[int]] = {}
        self.fail_next_download = False

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
        if self.fail_next_download:
            self.fail_next_download = False
            raise HubTransientError("transient immutable readback failure")
        count = self.download_counts.get(remote_prefix, 0) + 1
        self.download_counts[remote_prefix] = count
        if count in self.rate_limit_calls.get(remote_prefix, set()):
            raise HubRateLimitError("rate limited")
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
        return RuntimeState(
            torch_cuda_available=True, torch_cuda_initialized=True, model_loaded=True,
            hostname="gpu-host", host_architecture="x86_64", torch_version="2.7.0",
            cuda_version="12.8", gpu_device_name="NVIDIA RTX PRO 6000",
            gpu_uuid="GPU-fixture", total_vram_bytes=96 * 1024**3,
        )

    def measure(self, *, worker_count: int, burn_in_steps: int, measured_steps: int):
        from lehome_train.groot.runtime_mixture_warmup import GpuWarmupMeasurement
        return GpuWarmupMeasurement(
            decoded_samples=64 * (burn_in_steps + measured_steps), measured_steps=measured_steps,
            loader_wait_seconds=20.0 if worker_count == 0 else 1.0, step_seconds=20.0, gpu_busy_seconds=18.0,
            gpu_utilization_percent=90.0, oom=False, error=None,
            observed_batch_sizes=(64,) * (burn_in_steps + measured_steps),
            loss_min=.1, loss_max=.3, loss_final=.2,
            peak_memory_allocated_bytes=32 * 1024**3,
            peak_memory_reserved_bytes=40 * 1024**3,
            minimum_free_vram_bytes=20 * 1024**3,
            samples_per_second=192.0,
            step_latency_p50_seconds=.3,
            step_latency_p95_seconds=.5,
            materialization_proof={
                "bc": {"source_type": "bc", "window_id": "bc-1", "action_horizon": 16, "camera_count": 3},
                "rollout": {"source_type": "rollout", "window_id": "rollout-1", "action_horizon": 16, "camera_count": 3},
            },
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
    # Upgrade the generic mixture fixture to the production-only v3 contract.
    # The runtime path binds this experiment manifest digest into both warm-up
    # and local-recovery receipts, so legacy v2 bytes are intentionally barred.
    from lehome_train.groot.runtime_mixture import _manifest_digest_binding
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_value.update({
        "schema_version": 3,
        "experiment_manifest_sha256": "f" * 64,
        "mixture_weights": {"bc": 80, "rollout": 20, "dagger": 0},
        "source_quotas": {"bc": 51, "rollout": 13, "dagger": 0},
        "cycle_size": 64,
    })
    manifest_value["sources"][0]["quota"] = 51
    manifest_value["sources"][1]["quota"] = 13
    windows_value = json.loads(windows.read_text(encoding="utf-8"))
    windows_value["manifest_sha256"] = _manifest_digest_binding(manifest_value)
    _write(windows, windows_value)
    manifest_value["window_index"] = {
        "path": "windows.json", "sha256": sha256_file(windows),
        "byte_size": windows.stat().st_size,
    }
    _write(manifest, manifest_value)
    deployment_receipt_path = manifest.parent / "release-receipt.json"
    deployment_receipt = json.loads(deployment_receipt_path.read_text(encoding="utf-8"))
    deployment_receipt.update({
        "experiment_manifest_sha256": "f" * 64,
        "mixture_weights": {"bc": 80, "rollout": 20, "dagger": 0},
        "source_quotas": {"bc": 51, "rollout": 13, "dagger": 0},
    })
    for entry in deployment_receipt["artifact_entries"]:
        artifact = manifest.parent / entry["relative_path"]
        entry.update({"sha256": sha256_file(artifact), "byte_size": artifact.stat().st_size})
    _write(deployment_receipt_path, deployment_receipt)
    mounts_value = json.loads(mounts.read_text(encoding="utf-8"))
    mounts_value["deployment_receipt_sha256"] = sha256_file(deployment_receipt_path)
    _write(mounts, mounts_value)
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
            "mixture": {"repository": "ryanjin333/lehome-groot-n17-rollouts", "revision": "a" * 40, "mixture_id": mixture_id,
                    "manifest_sha256": sha256_file(manifest), "window_index_sha256": sha256_file(windows),
                    "normalization_sha256": sha256_file(normalization), "experiment_manifest_sha256": "f" * 64,
                    "source_revisions": {"bc": "b" * 40, "round-1": "c" * 40}},
        "deployment": {"oci_image_digest": LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE.rpartition("@")[2], "provider": "vast", "capability_sha256": "3" * 64},
        "code": {"repository_revision": code_revision, "bundle_sha256": code_sha, "isaac_groot_revision": "4" * 40},
        "parent_checkpoint": {"repository": LIFECYCLE.PARENT_CHECKPOINT["repository"], "revision": LIFECYCLE.PARENT_CHECKPOINT["revision"], "subpath": "policies/step-12000", "artifact_sha256": LIFECYCLE.PARENT_CHECKPOINT["artifact_sha256"]},
        "physical_batch_size": 64, "action_horizon": 16,
    }
    warmup = build_gpu_warmup_receipt(binding=binding, adapter=_WarmupAdapter())
    launch = {"base_model_path": str(cache / "parent"), "base_model_revision": MODEL_REVISION, "dataset_path": str(manifest.parent), "dataset_revision": "6" * 40,
              "modality_config_path": str(modality), "output_dir": str(run_root), "experiment_name": "runtime-mixture-70-30", "physical_batch_size": 64, "global_batch_size": 64, "gradient_accumulation_steps": 1, "augmentation_profile": "none", "num_gpus": 1, "max_steps": 2000, "save_steps": 500, "training_action_horizon": 16, "model_action_chunk_capacity": 40, "warmup_ratio": .05, "dataloader_num_workers": 4,
              "parent_checkpoint_repository": LIFECYCLE.PARENT_CHECKPOINT["repository"], "parent_checkpoint_revision": LIFECYCLE.PARENT_CHECKPOINT["revision"], "parent_checkpoint_subpath": "policies/step-12000", "parent_checkpoint_artifact_sha256": LIFECYCLE.PARENT_CHECKPOINT["artifact_sha256"], "runtime_mixture_manifest": str(manifest), "runtime_window_index": str(windows), "runtime_mounts_descriptor": str(mounts)}
    experiment = ExperimentConfig(repository_commit=code_revision, container_digest=LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE.rpartition("@")[2], model_repository="nvidia/GR00T-N1.7-3B", model_revision=MODEL_REVISION, dataset_repository="ryanjin333/lehome-groot-n17-rollouts", dataset_revision="a" * 40, dataset_manifest_sha256=mixture_id, physical_batch_size=64, gradient_accumulation_steps=1, sample_presentations=128_000, action_horizon=16, tune_language_backbone=False, tune_visual_backbone=False)
    identity = {"mixture_id": mixture_id, "deployment_receipt_sha256": sha256_file(deployment_receipt), "source_revisions": [{"source_id": "organizer", "immutable_revision": "b" * 40, "prefix": "bc/full", "tree_sha256": sha256_file(bc_receipt)}, {"source_id": "rollout", "immutable_revision": "c" * 40, "prefix": "rollouts/round-1", "tree_sha256": sha256_file(rollout_receipt)}], "schedule_seed": 17, "code_bundle_sha256": code_sha, "code_bundle_revision": code_revision, "oci_image": LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE.rpartition("@")[2], "parent_step12000_artifact_sha256": LIFECYCLE.PARENT_CHECKPOINT["artifact_sha256"], "physical_batch_size": 64, "action_horizon": 16}
    paths = {"launch_config": _write(prepared / "config" / "launch.json", launch), "experiment_config": _write(prepared / "config" / "experiment.json", experiment.to_dict()), "runtime_manifest": manifest, "runtime_window_index": windows, "runtime_normalization": normalization, "runtime_mounts_descriptor": mounts, "runtime_source_evidence": _write(prepared / "runtime" / "source-evidence.json", identity), "warmup_receipt": _write(prepared / "config" / "warmup.json", warmup), "runtime_warmup_binding": _write(prepared / "config" / "binding.json", binding)}
    token = prepared / "config" / "runtime.token"; token.write_text("fake-token\n"); token.chmod(0o600)
    hub = _MemoryHub()
    # Inject a 429 at the durable anchor byte readback.  Production records the
    # Hub-provided retry delay rather than manufacturing timing in this proof.
    hub.rate_limit_calls = {
        "checkpoint-staging/runtime-mixture-70-30/": set(),
        "checkpoints/runtime-mixture-70-30": {1},
    }
    retry_delays: list[float] = []
    def launch_one_k(config, **_kwargs):
        checkpoint = Path(config.output_dir) / config.experiment_name / "checkpoint-1000"; checkpoint.mkdir(parents=True)
        (checkpoint / "model.safetensors").write_bytes(b"immutable-1k")
        (checkpoint / "optimizer.pt").write_bytes(b"optimizer-1k")
        (checkpoint / "scheduler.pt").write_bytes(b"scheduler-1k")
        (checkpoint / "rng_state.pth").write_bytes(b"rng-1k")
        _write(checkpoint / "trainer_state.json", {"global_step": 1000, "log_history": [{"step": 1000, "loss": .2}]})
        run = checkpoint.parent; _write(run / "lehome_launch.json", config.identity())
        raise ProviderInterrupted("provider loss after durable 1K")
    monkeypatch.setattr(production, "launch_continuous_finetune", launch_one_k)
    result = production.ProductionRuntime(checkpoint_transport_factory=lambda **_kwargs: hub, checkpoint_retry_sleeper=retry_delays.append).runtime_mixture_train({**{key: str(value) for key, value in paths.items()}, "runtime_resume_archive": None, "runtime_resume_descriptor": None, "runtime_resume_cursor": None, "runtime_resume_anchor": None, "runtime_resume_publication": None, "local_recovery_root": str(output / "local-recovery"), "checkpoint_repository": LIFECYCLE.PARENT_CHECKPOINT["repository"], "checkpoint_revision": "main", "publisher_token_file": str(token), "instance_id": 10, "result_output": str(output / "result.json"), "status_output": str(output / "status.json")})
    publication = result["immutable_checkpoint_publications"][0]
    assert result["status"] == "runtime-mixture-interrupted" and publication["optimizer_step"] == 1000
    assert publication["kind"] == "runtime_mixture_checkpoint_publication"
    assert publication["identity"] == identity
    assert publication["runtime_cursor"] == {"optimizer_step": 1000, "global_sample_offset": 64_000, "physical_batch_size": 64, "action_horizon": 16}
    assert publication["runtime_checkpoint_anchor"]["readback_verified"] is True
    assert publication["readback_verified"] is True
    assert publication["runtime_checkpoint_anchor"]["immutable_anchor_revision"] == hub.main
    assert all("continuous-train" not in prefix and "generation" not in prefix for _, prefix in hub.uploaded)
    assert retry_delays == [300.0]

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

    # A replacement that fails before immutable recovery must be abort-cleaned
    # without consuming the durable cursor claim.  The next lease can recover
    # the same 1K anchor normally.
    failed_replacement = {"kind": "runtime_mixture_gpu_warmup_instance", "instance_id": 77, "host": "failed-replacement", "port": 22, "platform_arch": "x86_64", "trainer_image": LIFECYCLE.BOOTSTRAP_TRAINER_IMAGE, "provider_response_sha256": "c" * 64, "capability_sha256": "3" * 64}
    failed_cleanup = False
    failed_calls: list[tuple[str, ...]] = []
    def failed_runner(command: tuple[str, ...]) -> str:
        nonlocal failed_cleanup
        failed_calls.append(command)
        if command == ("vastai", "destroy", "instance", "77", "--yes"):
            failed_cleanup = True; return ""
        if command == ("vastai", "--raw", "show", "instance", "77"):
            return "{}" if failed_cleanup else '{"id":77}'
        raise AssertionError(command)
    failed_request = {
        "instance": failed_replacement, "code_revision": code_revision, "code_bundle_sha256": code_sha, "schedule_seed": 17,
        "bc_readback_receipt": str(bc_receipt), "rollout_readback_receipt": str(rollout_receipt), "deployment_receipt": str(deployment_receipt),
        "checkpoint_experiment_id": "runtime-mixture-70-30", "experiment_config": str(paths["experiment_config"]),
        "runtime_source_evidence": str(paths["runtime_source_evidence"]), "terminal_receipt": str(lost_terminal),
        "resume_destination": str(prepared / "failed-resume-download"), "failure_receipt": str(tmp_path / "failed-resume-failure.json"),
    }
    hub.fail_next_download = True
    with pytest.raises(HubTransientError, match="readback failure"):
        LIFECYCLE.main_for_test(
            ["runtime-checkpoint-replacement-resume", "--request", str(_write(tmp_path / "failed-replacement.json", failed_request)), "--execute", "--token-file", str(token)],
            runner=failed_runner, transport_factory=lambda **_kwargs: hub,
        )
    assert failed_calls == [("vastai", "destroy", "instance", "77", "--yes"), ("vastai", "--raw", "show", "instance", "77")]
    assert failed_cleanup and json.loads(Path(str(failed_request["failure_receipt"])).read_text())["cleanup_status"] == "destroyed_and_absent"
    assert not (lost_terminal.with_name(lost_terminal.name + ".resume-claim.json")).exists()

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
    # A claim is single-use even for the original replacement: replay with the
    # already materialized destination, or a different fresh destination, is
    # rejected before any cleanup/failure receipt can touch that active lease.
    replay_calls: list[tuple[str, ...]] = []
    replay_runner = lambda command: replay_calls.append(command) or (_ for _ in ()).throw(AssertionError(command))
    with pytest.raises(LIFECYCLE.RuntimeResumeAlreadyClaimed):
        LIFECYCLE.main_for_test(
            ["runtime-checkpoint-replacement-resume", "--request", str(_write(tmp_path / "replay-same.json", replacement_request)), "--execute", "--token-file", str(token)],
            runner=replay_runner, transport_factory=lambda **_kwargs: hub,
        )
    same_instance_replay = {**replacement_request, "resume_destination": str(prepared / "same-instance-fresh-destination"), "failure_receipt": str(tmp_path / "same-instance-replay-failure.json")}
    with pytest.raises(LIFECYCLE.RuntimeResumeAlreadyClaimed):
        LIFECYCLE.main_for_test(
            ["runtime-checkpoint-replacement-resume", "--request", str(_write(tmp_path / "replay-fresh.json", same_instance_replay)), "--execute", "--token-file", str(token)],
            runner=replay_runner, transport_factory=lambda **_kwargs: hub,
        )
    assert replay_calls == [] and not Path(str(same_instance_replay["failure_receipt"])).exists()

    # A different newly-rented replacement loses the same exclusive claim and
    # is still abort-cleaned with an exact absence readback.
    competing = {**replacement, "instance_id": 12, "host": "second-replacement", "provider_response_sha256": "d" * 64}
    competing_request = {**replacement_request, "instance": competing, "resume_destination": str(prepared / "second-resume-download"), "failure_receipt": str(tmp_path / "competing-resume-failure.json")}
    competing_destroyed = False
    def competing_runner(command: tuple[str, ...]) -> str:
        nonlocal competing_destroyed
        if command == ("vastai", "destroy", "instance", "12", "--yes"):
            competing_destroyed = True; return ""
        if command == ("vastai", "--raw", "show", "instance", "12"):
            return "{}" if competing_destroyed else '{"id":12}'
        raise AssertionError(command)
    with pytest.raises(ValueError, match="already claimed"):
        LIFECYCLE.main_for_test(
            ["runtime-checkpoint-replacement-resume", "--request", str(_write(tmp_path / "competing.json", competing_request)), "--execute", "--token-file", str(token)],
            runner=competing_runner, transport_factory=lambda **_kwargs: hub,
        )
    assert competing_destroyed and json.loads(Path(str(competing_request["failure_receipt"])).read_text())["cleanup_status"] == "destroyed_and_absent"

    # This branch deliberately models a fresh output mount, not a shared-disk
    # local recovery.  Discard both the trainer tree and its sidecars: leaving
    # receipts that point at deleted checkpoints is correctly fail-closed.
    # The replacement consumes only the discovered immutable archive and
    # descriptor, then observes/packages/publishes the 2K boundary.
    shutil.rmtree(run_root)
    shutil.rmtree(output / "local-recovery")
    resume_archive = Path(resumed["checkpoint_archive"])
    resume_descriptor = Path(resumed["checkpoint_descriptor"])
    from lehome_train.checkpoints import load_checkpoint_descriptor
    resume_record = load_checkpoint_descriptor(resume_descriptor).record
    assert resume_record.dataset_manifest_sha256 == mixture_id
    assert resume_record.sample_presentations == 64_000
    assert resume_record.artifact.sha256 == sha256_file(resume_archive)
    def launch_two_k(config, **_kwargs):
        checkpoint = Path(config.output_dir) / config.experiment_name / "checkpoint-2000"; checkpoint.mkdir(parents=True)
        (checkpoint / "model.safetensors").write_bytes(b"immutable-2k")
        (checkpoint / "optimizer.pt").write_bytes(b"optimizer-2k")
        (checkpoint / "scheduler.pt").write_bytes(b"scheduler-2k")
        (checkpoint / "rng_state.pth").write_bytes(b"rng-2k")
        _write(checkpoint / "trainer_state.json", {"global_step": 2000, "log_history": [{"step": 2000, "loss": .1}]})
    monkeypatch.setattr(production, "launch_continuous_finetune", launch_two_k)
    completed = production.ProductionRuntime(checkpoint_transport_factory=lambda **_kwargs: hub, checkpoint_retry_sleeper=retry_delays.append).runtime_mixture_train({**{key: str(value) for key, value in paths.items()}, "runtime_resume_archive": str(resume_archive), "runtime_resume_descriptor": str(resume_descriptor), "runtime_resume_cursor": resumed["runtime_cursor"], "runtime_resume_anchor": resumed["runtime_resume_anchor"], "runtime_resume_publication": resumed["runtime_resume_publication"], "local_recovery_root": str(output / "local-recovery"), "checkpoint_repository": LIFECYCLE.PARENT_CHECKPOINT["repository"], "checkpoint_revision": "main", "publisher_token_file": str(token), "instance_id": 11, "result_output": str(output / "replacement-result.json"), "status_output": str(output / "replacement-status.json")})
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

    # Lose the completed replacement's host result as well.  A fresh two-read
    # provider-loss proof must recover both immutable links and turn the 2K
    # anchor directly into a disposable completion terminal; it must never try
    # to resume a completed run.
    completed_lost_instance = {**replacement, "instance_id": 12, "host": "lost-complete"}
    lost_complete_path = tmp_path / "lost-complete-terminal.json"
    lost_complete_request = {
        **interruption,
        "instance": completed_lost_instance,
        "expected_checkpoint_steps": [1000, 2000],
        "terminal_receipt": str(lost_complete_path),
    }
    recovered_complete = LIFECYCLE.main_for_test(
        ["runtime-checkpoint-interrupted", "--request", str(_write(tmp_path / "lost-complete.json", lost_complete_request)), "--execute", "--token-file", str(token)],
        runner=lambda command: "{}" if command[:4] == ("vastai", "--raw", "show", "instance") else (_ for _ in ()).throw(AssertionError(command)),
        transport_factory=lambda **_kwargs: hub,
    )["terminal"]
    assert recovered_complete["kind"] == "runtime_mixture_training_complete"
    assert recovered_complete["disposable"] is True
    assert [item["optimizer_step"] for item in recovered_complete["immutable_checkpoint_publications"]] == [1000, 2000]
    latest_anchor = json.loads(hub.objects[hub.main]["checkpoints/runtime-mixture-70-30/latest.json"])
    assert latest_anchor["previous_anchor_sha256"] == publication["runtime_checkpoint_anchor"]["anchor_sha256"]

    recovered_disposed = False
    def recovered_destroy_runner(command: tuple[str, ...]) -> str:
        nonlocal recovered_disposed
        if command == ("vastai", "destroy", "instance", "12", "--yes"):
            recovered_disposed = True; return ""
        if command == ("vastai", "--raw", "show", "instance", "12"):
            return "{}" if recovered_disposed else '{"id":12}'
        raise AssertionError(command)
    recovered_dispose_request = {
        **replacement_request,
        "instance": completed_lost_instance,
        "terminal_receipt": str(lost_complete_path),
        "failure_receipt": str(tmp_path / "lost-complete-dispose-failure.json"),
    }
    recovered_disposal = LIFECYCLE.main_for_test(
        ["runtime-checkpoint-dispose", "--request", str(_write(tmp_path / "lost-complete-dispose.json", recovered_dispose_request)), "--execute", "--token-file", str(token)],
        runner=recovered_destroy_runner, transport_factory=lambda **_kwargs: hub,
    )
    assert recovered_disposed and recovered_disposal["destroy_authorized"] is True
