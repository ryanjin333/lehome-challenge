"""Small, injectable controller for the paid B1K training lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Callable, Mapping

from lehome_train.b1k.bootstrap import WorkspacePaths, read_hf_token, require_hardware
from lehome_train.b1k.contracts import RunContract
from lehome_train.b1k.launch import B1KLaunchFailure
from lehome_train.b1k.training import LaunchPlan, approved_launch_plans, is_recognized_cuda_oom
from lehome_train.constants import BEHAVIOR_1K_CHECKPOINT_BUCKET, BEHAVIOR_1K_DATASET_REPOSITORY, BEHAVIOR_1K_DATASET_REVISION, BEHAVIOR_1K_FINAL_MODEL_REPOSITORY, COSMOS_REPOSITORY, COSMOS_REVISION, ISAAC_GROOT_REVISION, MODEL_REVISION
from lehome_train.io import atomic_write_json


_COMMIT = re.compile(r"[0-9a-f]{40}")
_CHECKPOINT_NAME = re.compile(r"checkpoint-([1-9][0-9]*)$")


ProductionControllerFactory = Callable[[WorkspacePaths], "LifecycleController"]
_production_controller_factory: ProductionControllerFactory | None = None


def configure_production_controller(factory: ProductionControllerFactory | None) -> None:
    """Install the explicit runtime wiring used by the container entrypoint.

    This module owns no Vast client and does not synthesize an upstream training
    command.  The image bootstrap installs the concrete callback graph after
    its pinned upstream checkout is validated.  Tests use the same seam to
    exercise the full state machine without a CUDA or Hub dependency.
    """

    global _production_controller_factory
    _production_controller_factory = factory


def _controller_from_environment(paths: WorkspacePaths, values: Mapping[str, str]) -> "LifecycleController" | None:
    """Load the image-installed production adapter, never a Vast integration.

    Task 5 supplies this namespaced adapter in the image environment.  Keeping
    the adapter string in deployment configuration lets the entrypoint compose
    real bootstrap/Hub/bucket/final transports without retaining a token in a
    controller object or teaching this module an unpinned upstream CLI.
    """

    reference = values.get("B1K_LIFECYCLE_ADAPTER")
    if reference is None:
        return None
    if type(reference) is not str or reference.count(":") != 1:
        raise ValueError("B1K_LIFECYCLE_ADAPTER is invalid")
    module_name, attribute = reference.split(":", 1)
    if not module_name.startswith("lehome_train.b1k.") or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", attribute):
        raise ValueError("B1K_LIFECYCLE_ADAPTER is invalid")
    factory = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(factory):
        raise ValueError("B1K_LIFECYCLE_ADAPTER factory is invalid")
    controller = factory(paths)
    if not isinstance(controller, LifecycleController):
        raise ValueError("B1K_LIFECYCLE_ADAPTER factory returned an invalid controller")
    return controller


class TrainingFailure(RuntimeError):
    def __init__(self, message: str, *, optimizer_step: int) -> None:
        super().__init__(message); self.optimizer_step = optimizer_step


def _failure_reason(error: BaseException) -> str:
    """Store a stable, credential-free controller failure reason."""

    if isinstance(error, B1KLaunchFailure):
        if error.signal_number == 15:
            return "signal-sigterm"
        if error.signal_number == 2:
            return "signal-sigint"
    return type(error).__name__


def production_preflight(*, environment: dict[str, str] | None = None, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> WorkspacePaths:
    """Execute the non-negotiable local gates before any Hub transfer.

    This entrypoint deliberately has no Vast API surface: instance creation and
    destruction are outside the controller's authority.
    """
    values = os.environ if environment is None else environment
    if values.get("AUTO_DESTROY", "0") != "0": raise ValueError("AUTO_DESTROY must be 0")
    run_id = values.get("RUN_ID", "")
    paths = WorkspacePaths.from_root("/workspace", run_id=run_id)
    result = runner(("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"), check=False, capture_output=True, text=True)
    if result.returncode != 0: raise ValueError("nvidia-smi preflight failed")
    require_hardware(result.stdout, free_bytes=shutil.disk_usage("/workspace").free)
    read_hf_token()
    paths.create()
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Behavior 1K lifecycle controller")
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args(argv)
    paths = production_preflight()
    status = paths.output / "run-status.json"
    temporary = status.with_suffix(".incomplete")
    temporary.write_text(json.dumps({"phase": "preflight-passed", "auto_destroy": False}, sort_keys=True)); temporary.replace(status)
    if arguments.preflight_only:
        return 0
    try:
        controller = _controller_from_environment(paths, os.environ)
        if controller is None and _production_controller_factory is not None:
            controller = _production_controller_factory(paths)
        if controller is None:
            atomic_write_json(status, {"auto_destroy": False, "phase": "failed", "reason": "production-controller-not-configured"})
            return 1
        if not isinstance(controller, LifecycleController):
            raise ValueError("production controller factory returned an invalid controller")
        return 0 if controller.run() == 15_000 else 1
    except BaseException:
        try:
            existing = json.loads(status.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            existing = None
        if not isinstance(existing, dict) or existing.get("phase") == "preflight-passed":
            atomic_write_json(status, {"auto_destroy": False, "phase": "failed", "reason": "production-controller-failed"})
        return 1


def assemble_run_contract(*, token: str, run_id: str, cycle_id: str, container_digest: str, world_size: int, task_manifest_sha256: str, modality_sha256: str, stats_sha256: str, plan: LaunchPlan, launch_arguments_sha256: str, resume_policy: str = "auto") -> RunContract:
    """Construct the complete transient contract after GPU/data detection.

    ``token`` is passed only to validation and is absent from the returned
    contract and every status manifest.
    """
    if plan not in approved_launch_plans(num_gpus=world_size):
        raise ValueError("run contract plan is not approved for the detected world size")
    return RunContract.from_environment({"HF_TOKEN": token, "HF_DATASET_REPO": BEHAVIOR_1K_DATASET_REPOSITORY, "HF_MODEL_REPO": BEHAVIOR_1K_FINAL_MODEL_REPOSITORY, "HF_CHECKPOINT_BUCKET": BEHAVIOR_1K_CHECKPOINT_BUCKET, "DATASET_REVISION": BEHAVIOR_1K_DATASET_REVISION, "GROOT_REVISION": ISAAC_GROOT_REVISION, "CONTAINER_DIGEST": container_digest, "RUN_ID": run_id, "CYCLE_ID": cycle_id, "TRAIN_STEPS": "15000", "SAVE_STEPS": "1000", "CHECKPOINT_KEEP": "2", "RESUME_POLICY": resume_policy, "AUTO_DESTROY": "0", "BASE_MODEL_REVISION": MODEL_REVISION, "TASK_MANIFEST_SHA256": task_manifest_sha256, "MODALITY_SHA256": modality_sha256, "STATS_SHA256": stats_sha256, "WORLD_SIZE": str(world_size), "LAUNCH_PLAN_ID": plan.identity, "LEARNING_RATE": str(plan.learning_rate), "COSMOS_REPOSITORY": COSMOS_REPOSITORY, "COSMOS_REVISION": COSMOS_REVISION, "EXPERIMENT_NAME": run_id, "LAUNCH_ARGUMENTS_SHA256": launch_arguments_sha256})


@dataclass(slots=True)
class LifecycleController:
    """Injectable, fail-closed orchestrator for one immutable B1K run.

    The callbacks deliberately receive no credential.  Bootstrap and transport
    adapters own their short-lived token handling; status manifests retain only
    resumable experiment identity and sanitized outcome metadata.
    """

    run_training: Callable[[object, bool, Callable[[int], None]], int]
    publish_checkpoint: Callable[[int], None]
    world_size: int
    output: Path
    resume: bool = False
    resume_policy: str = "auto"
    bootstrap: Callable[[], object] | None = None
    select_resume: Callable[[], Path | None] | None = None
    finalize: Callable[[], Mapping[str, str]] | None = None
    remote_state_exists: Callable[[], bool] | None = None

    def __post_init__(self) -> None:
        if self.resume_policy not in {"auto", "never", "require"}:
            raise ValueError("lifecycle resume policy is invalid")

    def run(self) -> int:
        attempts: list[dict[str, object]] = []
        published_steps: list[int] = []
        resumed = self.resume
        resume_step = 0
        try:
            if self.bootstrap is not None:
                self.bootstrap()
            if self.select_resume is not None:
                restored = self.select_resume()
                if restored is not None:
                    matched = _CHECKPOINT_NAME.fullmatch(restored.name) if isinstance(restored, Path) else None
                    if matched is None or int(matched.group(1)) not in range(1_000, 15_001, 1_000):
                        raise ValueError("resume checkpoint identity is invalid")
                    resumed = True
                    resume_step = int(matched.group(1))
            self._write("bootstrap-complete", attempts=attempts, published_steps=published_steps, resumed=resumed)
        except BaseException as error:
            self._write("failed", attempts=attempts, published_steps=published_steps, resumed=resumed, reason=_failure_reason(error))
            raise
        if resume_step == 15_000:
            try:
                if self.finalize is None:
                    raise ValueError("finalization callback is required before completion")
                receipt = self.finalize()
                commit = receipt.get("immutable_commit")
                if type(commit) is not str or not _COMMIT.fullmatch(commit):
                    raise ValueError("finalization did not return an immutable readback receipt")
                attempts.append({"plan": "resumed-checkpoint-15000", "result": "complete"})
                self._write("complete", attempts=attempts, published_steps=published_steps, resumed=resumed)
                return 15_000
            except BaseException as error:
                self._write("failed", attempts=attempts, published_steps=published_steps, resumed=resumed, reason=_failure_reason(error))
                raise
        for plan in approved_launch_plans(num_gpus=self.world_size):
            try:
                expected_steps = iter(range(resume_step + 1_000, 15_001, 1_000))

                def on_stable_checkpoint(step: int) -> None:
                    expected = next(expected_steps, None)
                    if step != expected:
                        raise ValueError("stable checkpoint order is invalid")
                    self.publish_checkpoint(step)
                    published_steps.append(step)
                    self._write("training", attempts=attempts, published_steps=published_steps, resumed=resumed)

                step = self.run_training(plan, resumed, on_stable_checkpoint)
                if step != 15_000:
                    raise TrainingFailure("training did not reach step 15000", optimizer_step=step)
                if next(expected_steps, None) is not None:
                    raise TrainingFailure("training completed before every stable checkpoint was published", optimizer_step=step)
                attempts.append({"plan": plan.identity, "result": "complete"})
                if self.finalize is None:
                    raise ValueError("finalization callback is required before completion")
                receipt = self.finalize()
                commit = receipt.get("immutable_commit")
                if type(commit) is not str or not _COMMIT.fullmatch(commit):
                    raise ValueError("finalization did not return an immutable readback receipt")
                self._write("complete", attempts=attempts, published_steps=published_steps, resumed=resumed)
                return step
            except (TrainingFailure, B1KLaunchFailure) as error:
                attempts.append({"plan": plan.identity, "result": "training-failed", "optimizer_step": error.optimizer_step})
                self._write("failed", attempts=attempts, published_steps=published_steps, resumed=resumed, reason=_failure_reason(error))
                remote_state = resumed or (self.remote_state_exists is not None and self.remote_state_exists())
                if error.optimizer_step != 0 or remote_state or not is_recognized_cuda_oom(str(error)):
                    raise
            except BaseException as error:
                attempts.append({"plan": plan.identity, "result": "controller-failed"})
                self._write("failed", attempts=attempts, published_steps=published_steps, resumed=resumed, reason=_failure_reason(error))
                raise
        raise TrainingFailure("all approved B1K batch plans exhausted", optimizer_step=0)

    def _write(self, phase: str, *, attempts: list[dict[str, object]], published_steps: list[int], resumed: bool, reason: str | None = None) -> None:
        if phase not in {"bootstrap-complete", "training", "complete", "failed"}:
            raise ValueError("lifecycle phase is invalid")
        self.output.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {"attempts": attempts, "auto_destroy": False, "phase": phase, "published_steps": published_steps, "resume_policy": self.resume_policy, "resumed": resumed}
        if reason is not None:
            payload["reason"] = reason
        atomic_write_json(self.output / "run-status.json", payload)


if __name__ == "__main__":
    raise SystemExit(main())
