"""Concrete, credential-free assembly of the pinned B1K lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable

from lehome_train.b1k.bootstrap import (
    BootstrapResult,
    ProductionHubAccess,
    WorkspacePaths,
    bootstrap_remote,
    read_hf_token,
)
from lehome_train.b1k.bucket_protocol import BucketHelperClient
from lehome_train.b1k.finalize import FinalEvidence, Finalizer
from lehome_train.b1k.launch import (
    B1KLaunch,
    _visible_devices,
    actual_b1k_output_root,
    build_b1k_launch,
    run_b1k_launch_with_checkpoint_watch,
    sanitize_b1k_log,
)
from lehome_train.b1k.lifecycle import LifecycleController, assemble_run_contract
from lehome_train.b1k.contracts import RunContract
from lehome_train.b1k.rolling_checkpoints import (
    CheckpointCompatibility,
    HelperBucketBackend,
    LocalCheckpointPublisher,
    ResumePolicy,
    RollingCheckpointStore,
)
from lehome_train.b1k.training import LaunchPlan, SUPPORTED_GPU_COUNTS, approved_launch_plans
from lehome_train.constants import (
    BEHAVIOR_1K_CHECKPOINT_BUCKET,
    BEHAVIOR_1K_DATASET_REPOSITORY,
    BEHAVIOR_1K_DATASET_REVISION,
    BEHAVIOR_1K_FINAL_MODEL_REPOSITORY,
    COSMOS_REVISION,
    ISAAC_GROOT_REVISION,
    MODEL_REVISION,
)
from lehome_train.io import atomic_write_json, canonical_json_sha256


_CHECKOUT = Path("/opt/isaac-groot")
_BUCKET_HELPER = "/opt/b1k-bucket-helper/bin/b1k-bucket-helper"


def _world_size(values: dict[str, str]) -> int:
    if "CUDA_VISIBLE_DEVICES" in values:
        visible = values["CUDA_VISIBLE_DEVICES"]
        if type(visible) is not str:
            raise ValueError("CUDA_VISIBLE_DEVICES must be a canonical non-empty device list")
        count = len(visible.split(","))
    else:
        try:
            result = subprocess.run(("nvidia-smi", "--query-gpu=index", "--format=csv,noheader"), check=False, capture_output=True, text=True)
        except OSError:
            result = None
        count = len([line for line in result.stdout.splitlines() if line.strip()]) if result is not None and result.returncode == 0 else 0
        visible = ",".join(str(index) for index in range(count))
    if count not in SUPPORTED_GPU_COUNTS:
        raise ValueError("B1K production adapter requires one to four visible GPUs")
    values["CUDA_VISIBLE_DEVICES"] = _visible_devices(visible, count)
    return count


def _deploy_modality(dataset: Path) -> Path:
    """Run only the pinned upstream modality deployer with its root argument."""

    script = _CHECKOUT / "scripts/b1k/deploy_modality.py"
    result = subprocess.run((sys.executable, str(script), str(dataset)), check=False, capture_output=True, text=True, env={key: value for key, value in os.environ.items() if key != "HF_TOKEN"})
    if result.returncode != 0:
        raise ValueError("pinned B1K modality deployment failed")
    modality = dataset / "meta/modality.json"
    if modality.is_symlink() or not modality.is_file():
        raise ValueError("pinned B1K modality deployment did not create the loader artifact")
    return modality


def _fingerprint(path: Path) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    fingerprint = value.get("fingerprint") if isinstance(value, dict) else None
    if type(fingerprint) is not str or len(fingerprint) != 64:
        raise ValueError("bootstrap manifest fingerprint is invalid")
    return fingerprint


def _final_transport() -> tuple[Callable[[str], None], Callable[[str, str, Path], str], Callable[[str, str, str, Path], None]]:
    """Create short-lived explicit-token model upload/readback callbacks."""

    def ensure_branch(branch: str) -> None:
        from huggingface_hub import HfApi

        HfApi().create_branch(
            repo_id=BEHAVIOR_1K_FINAL_MODEL_REPOSITORY,
            branch=branch,
            repo_type="model",
            exist_ok=True,
            token=read_hf_token(),
        )

    def upload(branch: str, remote: str, source: Path) -> str:
        from huggingface_hub import HfApi

        result = HfApi().upload_file(
            path_or_fileobj=str(source),
            path_in_repo=remote,
            repo_id=BEHAVIOR_1K_FINAL_MODEL_REPOSITORY,
            repo_type="model",
            revision=branch,
            token=read_hf_token(),
        )
        commit = getattr(result, "oid", None)
        if type(commit) is not str:
            raise ValueError("final model upload did not return an immutable commit")
        return commit

    def download(branch: str, remote: str, commit: str, destination: Path) -> None:
        from huggingface_hub import hf_hub_download

        source = Path(hf_hub_download(
            repo_id=BEHAVIOR_1K_FINAL_MODEL_REPOSITORY,
            filename=remote,
            repo_type="model",
            revision=commit,
            force_download=True,
            token=read_hf_token(),
        ))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    return ensure_branch, upload, download


@dataclass(slots=True)
class _Runtime:
    paths: WorkspacePaths
    values: dict[str, str]
    hub: ProductionHubAccess
    world_size: int
    bootstrap_result: BootstrapResult | None = None
    publisher: LocalCheckpointPublisher | None = None
    receipt_paths: list[Path] | None = None
    resume_step: int = 0
    launch_arguments: tuple[str, ...] | None = None
    run_contract: RunContract | None = None

    def bootstrap(self) -> None:
        # PyArrow is part of the production image, but keeping the heavy dataset
        # reader lazy lets lifecycle wiring and preflight fail cleanly before
        # materialization dependencies are needed.
        from lehome_train.b1k.dataset import build_training_manifest, materialize_training_manifest

        result = bootstrap_remote(
            paths=self.paths,
            token=read_hf_token(),
            create_bucket_flag=self.values.get("CREATE_CHECKPOINT_BUCKET", "0"),
            hub=self.hub,
            build_selection=lambda root: build_training_manifest(
                root,
                repository=BEHAVIOR_1K_DATASET_REPOSITORY,
                revision=BEHAVIOR_1K_DATASET_REVISION,
            ),
            materialize=lambda root, **kwargs: materialize_training_manifest(
                root,
                build_training_manifest(
                    root,
                    repository=BEHAVIOR_1K_DATASET_REPOSITORY,
                    revision=BEHAVIOR_1K_DATASET_REVISION,
                ),
                **kwargs,
            ),
            deploy_modality=_deploy_modality,
            stats_path=self.paths.dataset / "meta/stats.json",
        )
        self.bootstrap_result = result
        self.receipt_paths = []

    def _launch(self, plan: LaunchPlan, *, resume: bool) -> B1KLaunch:
        if self.bootstrap_result is None:
            raise ValueError("B1K bootstrap was not completed")
        environment = dict(self.values)
        environment.update(self.bootstrap_result.offline_environment())
        if environment.get("WANDB_MODE") != "offline":
            raise ValueError("B1K launch requires WANDB_MODE=offline")
        return build_b1k_launch(
            plan,
            visible_devices=self.values.get("CUDA_VISIBLE_DEVICES"),
            environment=environment,
            official_checkout=_CHECKOUT,
            dataset_path=str(self.bootstrap_result.dataset),
            base_model_path=str(self.bootstrap_result.derived_model),
            output_dir=str(self.paths.output),
            experiment_name=self.values["RUN_ID"],
            resume_from_checkpoint=resume,
        )

    def _training_output(self) -> Path:
        return actual_b1k_output_root(self.paths.output, self.values["RUN_ID"])

    def _ensure_resume_logs(self) -> None:
        output = self._training_output()
        output.mkdir(parents=True, exist_ok=True)
        for name in ("trainer.stdout.log", "trainer.stderr.log"):
            path = output / name
            if path.is_symlink():
                raise ValueError("B1K trainer log path is unsafe")
            if not path.exists():
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                try:
                    os.fchmod(descriptor, 0o600)
                    os.write(descriptor, b"no local trainer process ran during verified resume\n")
                finally:
                    os.close(descriptor)

    def _base_launch(self, plan: LaunchPlan) -> B1KLaunch:
        """The resume-independent argv identity used in every descriptor."""

        return self._launch(plan, resume=False)

    def _store(self, plan: LaunchPlan, launch_sha256: str) -> RollingCheckpointStore:
        if self.bootstrap_result is None:
            raise ValueError("B1K bootstrap was not completed")
        contract = assemble_run_contract(
            token=read_hf_token(),
            run_id=self.values["RUN_ID"],
            cycle_id=self.values["CYCLE_ID"],
            container_digest=self.values["CONTAINER_DIGEST"],
            world_size=self.world_size,
            task_manifest_sha256=_fingerprint(self.bootstrap_result.selection_manifest),
            modality_sha256=self.bootstrap_result.modality_sha256,
            stats_sha256=self.bootstrap_result.stats_sha256,
            plan=plan,
            launch_arguments_sha256=launch_sha256,
            resume_policy=self.values.get("RESUME_POLICY", "auto"),
        )
        self.run_contract = contract
        compatibility = CheckpointCompatibility(
            materialized_dataset_fingerprint=_fingerprint(self.bootstrap_result.materialized_manifest),
            modality_sha256=contract.modality_sha256,
            stats_sha256=contract.stats_sha256,
            groot_revision=ISAAC_GROOT_REVISION,
            base_model_revision=MODEL_REVISION,
            cosmos_revision=COSMOS_REVISION,
            container_digest=contract.container_digest,
            cycle_id=contract.cycle_id,
            world_size=self.world_size,
            plan_identity=plan.identity,
            physical_batch_size=plan.physical_batch_size,
            global_batch_size=plan.global_batch_size,
            gradient_accumulation_steps=plan.gradient_accumulation_steps,
            effective_global_batch_size=plan.effective_global_batch_size,
            learning_rate=plan.learning_rate,
            weight_decay=plan.weight_decay,
            warmup_ratio=plan.warmup_ratio,
            launch_argv_sha256=launch_sha256,
        )
        return RollingCheckpointStore(
            backend=HelperBucketBackend(BucketHelperClient(executable=_BUCKET_HELPER), BEHAVIOR_1K_CHECKPOINT_BUCKET),
            run_id=contract.run_id,
            compatibility=compatibility,
        )

    def run_training(self, plan: LaunchPlan, resume: bool, on_stable_checkpoint: Callable[[int], None]) -> int:
        if self.bootstrap_result is None:
            raise ValueError("B1K bootstrap was not completed")
        launch = self._launch(plan, resume=resume)
        base_launch = self._base_launch(plan)
        self.launch_arguments = launch.command
        self.publisher = LocalCheckpointPublisher(
            store=self._store(plan, base_launch.arguments_sha256),
            checkpoint_root=self._training_output(),
            receipts_root=self._training_output() / "checkpoint-receipts",
        )

        def publish(step: int) -> None:
            if self.publisher is None or self.receipt_paths is None:
                raise ValueError("B1K checkpoint publisher is unavailable")
            self.receipt_paths.append(self.publisher.publish(step))
            on_stable_checkpoint(step)

        return run_b1k_launch_with_checkpoint_watch(
            launch,
            output_dir=self._training_output(),
            world_size=self.world_size,
            on_stable_checkpoint=publish,
            resume_floor=self.resume_step,
            published_seed=tuple(step for step in self.publisher.store.verified_steps() if step <= self.resume_step),
        )

    def select_resume(self) -> Path | None:
        """Find exactly one compatible approved plan before restoring it.

        A checkpoint can originate from a clean step-zero OOM fallback.  Its
        plan identity is immutable checkpoint compatibility, not a preference
        for today's largest batch.  The compatibility scan validates the full
        remote namespace before choosing a plan, so malformed or mixed state
        cannot be skipped by probing a later candidate.
        """

        policy = ResumePolicy(self.values.get("RESUME_POLICY", "auto"))
        candidates: list[tuple[LaunchPlan, B1KLaunch, RollingCheckpointStore]] = []
        for plan in approved_launch_plans(num_gpus=self.world_size):
            base_launch = self._base_launch(plan)
            candidates.append((plan, base_launch, self._store(plan, base_launch.arguments_sha256)))
        observed = candidates[0][2].inspect_resume_compatibility(policy)
        if observed is None:
            return None
        matches = tuple(candidate for candidate in candidates if candidate[2].compatibility == observed)
        if len(matches) != 1:
            raise ValueError("remote checkpoint does not identify exactly one approved launch plan")
        plan, base_launch, _candidate_store = matches[0]
        launch = self._launch(plan, resume=True)
        # Rebuild the selected store to set the final run contract to the
        # recovered plan and revalidate the namespace immediately before I/O.
        store = self._store(plan, base_launch.arguments_sha256)
        restored = store.resume(policy, self._training_output())
        if restored is not None:
            self.resume_step = int(restored.name.removeprefix("checkpoint-"))
            self.launch_arguments = launch.command
            self._ensure_resume_logs()
            self.publisher = LocalCheckpointPublisher(
                store=store,
                checkpoint_root=self._training_output(),
                receipts_root=self._training_output() / "checkpoint-receipts",
            )
            if self.receipt_paths is None:
                raise ValueError("B1K resume receipt paths are unavailable")
            receipts_root = self._training_output() / "checkpoint-receipts"
            receipts_root.mkdir(parents=True, exist_ok=True)
            receipt = receipts_root / f"resume-step-{self.resume_step}.json"
            if receipt.exists() or receipt.is_symlink():
                raise ValueError("verified resume receipt already exists")
            atomic_write_json(
                receipt,
                {
                    "descriptor": store.backend.read_json(f"runs/{store.run_id}/latest.json"),
                    "restored_checkpoint": restored.name,
                    "run_id": store.run_id,
                    "step": self.resume_step,
                    "verified_steps": list(store.verified_steps()),
                },
            )
            self.receipt_paths.append(receipt)
        return restored

    def publish_checkpoint(self, step: int) -> None:
        # Publication happens inside ``run_training`` before this accounting
        # callback; the controller must never be allowed to republish a step.
        if self.publisher is None or step not in self.publisher.store.verified_steps():
            raise ValueError("stable checkpoint was not remotely verified")

    def finalize(self) -> dict[str, str]:
        if self.bootstrap_result is None or self.receipt_paths is None or self.run_contract is None or self.publisher is None:
            raise ValueError("B1K finalization prerequisites are unavailable")
        retained = self.publisher.store.ensure_newest_two()
        if len(retained) > 2:
            raise ValueError("B1K checkpoint retention repair failed")
        ensure_branch, upload, download = _final_transport()
        output = self._training_output()
        output.mkdir(parents=True, exist_ok=True)
        contract_path = output / "run-contract.json"
        atomic_write_json(contract_path, self.run_contract.to_dict())
        argv_path = output / "launch-arguments.json"
        if self.launch_arguments is None:
            raise ValueError("B1K finalization has no launch arguments")
        atomic_write_json(argv_path, {"arguments": list(self.launch_arguments), "arguments_sha256": canonical_json_sha256(self.launch_arguments)})
        revisions = output / "revisions-image.json"
        atomic_write_json(revisions, {"groot_revision": ISAAC_GROOT_REVISION, "container_digest": self.values["CONTAINER_DIGEST"]})
        log = self.paths.logs / "controller.log"
        if log.is_symlink():
            raise ValueError("B1K controller log path is unsafe")
        if not log.exists():
            log.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.fchmod(descriptor, 0o600)
                os.write(descriptor, b"controller log retained externally\n")
            finally:
                os.close(descriptor)
        trainer_logs = (output / "trainer.stdout.log", output / "trainer.stderr.log")
        if any(path.is_symlink() or not path.is_file() for path in trainer_logs):
            raise ValueError("B1K trainer logs are unavailable for final evidence")
        for path in (log, *trainer_logs):
            sanitize_b1k_log(path)
        evidence = FinalEvidence(
            run_contract=contract_path,
            selection_manifest=self.bootstrap_result.selection_manifest,
            materialized_manifest=self.bootstrap_result.materialized_manifest,
            modality=self.paths.dataset / "meta/modality.json",
            stats=self.paths.dataset / "meta/stats.json",
            model_derivation=self.bootstrap_result.model_derivation,
            revisions_image=revisions,
            argv=argv_path,
            logs=(log, *trainer_logs),
            rolling_receipts=tuple(self.receipt_paths),
            world_size=self.world_size,
        )
        return Finalizer(upload_file=upload, download_file=download, ensure_branch=ensure_branch).finalize(
            run_id=self.values["RUN_ID"],
            checkpoint=output / "checkpoint-15000",
            evidence=evidence,
            final_dir=self.paths.final,
        )


def build_production_controller(paths: WorkspacePaths) -> LifecycleController:
    """Compose the headless B1K controller; never creates/destroys Vast."""

    values = dict(os.environ)
    if values.get("AUTO_DESTROY", "0") != "0":
        raise ValueError("AUTO_DESTROY must be 0")
    resume_policy = ResumePolicy(values.get("RESUME_POLICY", "auto"))
    values["RESUME_POLICY"] = resume_policy.value
    world_size = _world_size(values)
    runtime = _Runtime(
        paths=paths,
        values=values,
        hub=ProductionHubAccess(BucketHelperClient(executable=_BUCKET_HELPER)),
        world_size=world_size,
    )
    return LifecycleController(
        run_training=runtime.run_training,
        publish_checkpoint=runtime.publish_checkpoint,
        world_size=world_size,
        output=paths.output,
        resume_policy=resume_policy.value,
        bootstrap=runtime.bootstrap,
        select_resume=runtime.select_resume,
        finalize=runtime.finalize,
        remote_state_exists=lambda: runtime.publisher is not None and bool(runtime.publisher.store.verified_steps()),
    )
