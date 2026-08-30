"""Strict immutable provenance for an accepted trainer image."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Mapping
from urllib.parse import urlsplit

from lehome_train.constants import (
    BEHAVIOR_1K_DATASET_REPOSITORY,
    BEHAVIOR_1K_DATASET_REVISION,
    BEHAVIOR_1K_TRAINER_IMAGE_REPOSITORY,
    COSMOS_REPOSITORY,
    COSMOS_REVISION,
    CUDA_BASE_DIGEST,
    ISAAC_GROOT_REPOSITORY as _ISAAC_GROOT_REPOSITORY,
    ISAAC_GROOT_REVISION,
    MODEL_REVISION,
    PYTHON_VERSION,
)
from lehome_train.io import atomic_write_json, canonical_json_sha256, sha256_file
from lehome_train.b1k.training import SUPPORTED_GPU_COUNTS, approved_launch_plans
from lehome_train.b1k.launch import build_b1k_command


CUDA_BASE_IMAGE = "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04"
ISAAC_GROOT_REPOSITORY = _ISAAC_GROOT_REPOSITORY
MODEL_REPOSITORY = "nvidia/GR00T-N1.7-3B"
IMAGE_REPOSITORY = BEHAVIOR_1K_TRAINER_IMAGE_REPOSITORY
DATASET_REPOSITORY = BEHAVIOR_1K_DATASET_REPOSITORY
UV_VERSION = "0.8.22"
TRAINER_LOCK_SHA256 = "67fcd520cd75f3b3b383fcc887f244c332af5c2a5548d384d71e0376697b2432"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def _exact_keys(value: object, expected: set[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"release manifest {label} has an incompatible schema")
    if not all(type(key) is str for key in value):
        raise ValueError(f"release manifest {label} keys must be strings")
    return value


def _string(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"release manifest {label} must be a non-empty string")
    return value


def _optional_number(value: object, label: str) -> float | None:
    if value is None:
        return None
    if type(value) not in (int, float) or float(value) < 0:
        raise ValueError(f"release manifest {label} must be nonnegative")
    return float(value)


@dataclass(frozen=True, slots=True)
class ReleaseManifest:
    """Validated release identity and fresh-machine acceptance evidence."""

    payload: dict[str, object]

    @property
    def status(self) -> str:
        return str(self.payload["status"])

    @property
    def cuda_base_digest(self) -> str:
        return str(self.payload["cuda_base"]["digest"])  # type: ignore[index]

    @property
    def isaac_groot_commit(self) -> str:
        return str(self.payload["isaac_groot"]["commit"])  # type: ignore[index]

    @property
    def model_revision(self) -> str:
        return str(self.payload["base_model"]["revision"])  # type: ignore[index]

    @property
    def trainer_lock_sha256(self) -> str:
        return str(self.payload["trainer_lock_sha256"])

    @property
    def repository_commit(self) -> str | None:
        value = self.payload["repository_commit"]
        return None if value is None else str(value)

    @property
    def oci_digest(self) -> str | None:
        value = self.payload["image"]["digest"]  # type: ignore[index]
        return None if value is None else str(value)

    @property
    def gpu_acceptance_status(self) -> str:
        return str(self.payload["gpu_acceptance"]["status"])  # type: ignore[index]

    @property
    def dataset_revision(self) -> str | None:
        value = self.payload["dataset"]["revision"]  # type: ignore[index]
        return None if value is None else str(value)

    @property
    def dataset_manifest_sha256(self) -> str | None:
        value = self.payload["dataset"]["manifest_sha256"]  # type: ignore[index]
        return None if value is None else str(value)

    @property
    def normalization_sha256(self) -> str | None:
        value = self.payload["dataset"]["normalization_sha256"]  # type: ignore[index]
        return None if value is None else str(value)

    def to_dict(self) -> dict[str, object]:
        return json.loads(json.dumps(self.payload, allow_nan=False))

    @classmethod
    def from_dict(cls, raw: object) -> "ReleaseManifest":
        root = _exact_keys(
            raw,
            {
                "schema_version",
                "status",
                "platform",
                "cuda_base",
                "python_version",
                "uv_version",
                "isaac_groot",
                "base_model",
                "cosmos",
                "dataset",
                "trainer_lock_sha256",
                "repository_commit",
                "image",
                "gpu_acceptance",
            },
            "root",
        )
        if type(root["schema_version"]) is not int or root["schema_version"] != 2:
            raise ValueError("release manifest schema version is unsupported")
        status = _string(root["status"], "status")
        if status not in {"unreleased", "accepted"}:
            raise ValueError("release manifest status is unsupported")
        if root["platform"] != "linux/amd64":
            raise ValueError("release manifest platform must be linux/amd64")
        if root["python_version"] != PYTHON_VERSION or root["uv_version"] != UV_VERSION:
            raise ValueError("release manifest toolchain is not pinned")

        cuda = _exact_keys(root["cuda_base"], {"image", "digest"}, "cuda_base")
        if cuda["image"] != CUDA_BASE_IMAGE or cuda["digest"] != CUDA_BASE_DIGEST:
            raise ValueError("release manifest CUDA base differs from the approved digest")
        groot = _exact_keys(root["isaac_groot"], {"repository", "commit"}, "isaac_groot")
        if groot["repository"] != ISAAC_GROOT_REPOSITORY or groot["commit"] != ISAAC_GROOT_REVISION:
            raise ValueError("release manifest Isaac-GR00T identity differs from the approved commit")
        model = _exact_keys(root["base_model"], {"repository", "revision"}, "base_model")
        if model["repository"] != MODEL_REPOSITORY or model["revision"] != MODEL_REVISION:
            raise ValueError("release manifest model identity differs from the approved revision")
        cosmos = _exact_keys(root["cosmos"], {"repository", "revision"}, "cosmos")
        if cosmos["repository"] != COSMOS_REPOSITORY or cosmos["revision"] != COSMOS_REVISION:
            raise ValueError("release manifest Cosmos identity differs from the approved revision")

        dataset = _exact_keys(
            root["dataset"],
            {"repository", "revision", "manifest_sha256", "normalization_sha256"},
            "dataset",
        )
        if dataset["repository"] != DATASET_REPOSITORY:
            raise ValueError("release manifest dataset repository is unsupported")
        dataset_identity = (
            dataset["revision"],
            dataset["manifest_sha256"],
            dataset["normalization_sha256"],
        )
        has_dataset_artifact_identity = any(
            value is not None for value in dataset_identity[1:]
        )
        if has_dataset_artifact_identity:
            if type(dataset["revision"]) is not str or not _SHA.fullmatch(dataset["revision"]):
                raise ValueError("release manifest dataset revision is not immutable")
            if dataset["revision"] != BEHAVIOR_1K_DATASET_REVISION:
                raise ValueError("release manifest dataset revision differs from the approved B1K commit")
            for key in ("manifest_sha256", "normalization_sha256"):
                value = dataset[key]
                if type(value) is not str or not _SHA256.fullmatch(value):
                    raise ValueError(f"release manifest dataset {key} is invalid")
        elif dataset["revision"] not in {None, BEHAVIOR_1K_DATASET_REVISION}:
            raise ValueError("release manifest dataset revision differs from the approved B1K commit")

        lock_hash = _string(root["trainer_lock_sha256"], "trainer lock hash")
        repository_commit = root["repository_commit"]
        if not _SHA256.fullmatch(lock_hash):
            raise ValueError("release manifest trainer lock hash is invalid")
        if lock_hash != TRAINER_LOCK_SHA256:
            raise ValueError("release manifest differs from the approved trainer lock")
        if repository_commit is not None and (
            type(repository_commit) is not str or not _SHA.fullmatch(repository_commit)
        ):
            raise ValueError("release manifest repository commit is not immutable")

        image = _exact_keys(root["image"], {"repository", "tag", "digest"}, "image")
        if image["repository"] != IMAGE_REPOSITORY:
            raise ValueError("release manifest image repository is unsupported")
        tag = image["tag"]
        if tag != repository_commit:
            raise ValueError("release manifest image tag must equal the repository commit")
        digest = image["digest"]
        if digest is not None and (type(digest) is not str or not _DIGEST.fullmatch(digest)):
            raise ValueError("release manifest OCI digest is invalid")

        acceptance = _exact_keys(
            root["gpu_acceptance"],
            {
                "status",
                "run_id",
                "world_size",
                "launch_plan_id",
                "effective_global_batch_size",
                "physical_batch_size",
                "global_batch_size",
                "gradient_accumulation_steps",
                "task_manifest_sha256",
                "modality_sha256",
                "stats_sha256",
                "base_model_revision",
                "cosmos_revision",
                "learning_rate",
                "warmup_ratio",
                "launch_arguments_sha256",
                "launch_arguments",
                "hardware_model",
                "hardware_count",
                "vram_gib",
                "cuda_optimizer_step_passed",
                "checkpoint_roundtrip_passed",
                "resume_passed",
                "tiny_overfit_passed",
                "finite_loss",
                "acceptance_seconds",
                "evidence_uri",
            },
            "gpu_acceptance",
        )
        acceptance_status = _string(acceptance["status"], "GPU acceptance status")
        if acceptance_status not in {"pending", "passed"}:
            raise ValueError("release manifest GPU acceptance status is unsupported")
        if status == "accepted":
            if repository_commit is None:
                raise ValueError("accepted release manifest requires an immutable repository commit")
            if digest is None:
                raise ValueError("accepted release manifest requires a final OCI digest")
            if acceptance_status != "passed":
                raise ValueError("accepted release manifest requires passed GPU acceptance")
            if any(value is None for value in dataset_identity):
                raise ValueError("accepted release manifest requires immutable dataset evidence")
            world_size = acceptance["world_size"]
            if type(world_size) is not int or world_size not in SUPPORTED_GPU_COUNTS:
                raise ValueError("GPU acceptance world size must be one to four")
            plans = approved_launch_plans(num_gpus=world_size)
            plan = next((item for item in plans if item.identity == acceptance["launch_plan_id"]), None)
            if plan is None:
                raise ValueError("GPU acceptance launch plan is not approved")
            if acceptance["effective_global_batch_size"] != plan.effective_global_batch_size:
                raise ValueError("GPU acceptance effective batch differs from the launch plan")
            if (acceptance["physical_batch_size"], acceptance["global_batch_size"], acceptance["gradient_accumulation_steps"]) != (plan.physical_batch_size, plan.global_batch_size, plan.gradient_accumulation_steps):
                raise ValueError("GPU acceptance optimizer fields differ from the launch plan")
            if acceptance["task_manifest_sha256"] != dataset["manifest_sha256"]:
                raise ValueError("GPU acceptance manifest identity differs from the dataset")
            if acceptance["stats_sha256"] != dataset["normalization_sha256"]:
                raise ValueError("GPU acceptance stats identity differs from the dataset")
            if acceptance["modality_sha256"] != "ca3a2e406472650bee9439ed81a3f4a1531b6fe689cdc1b348d0f260a208e641":
                raise ValueError("GPU acceptance modality fingerprint differs from R1Pro")
            if acceptance["base_model_revision"] != MODEL_REVISION:
                raise ValueError("GPU acceptance base model revision differs from the pin")
            if acceptance["cosmos_revision"] != COSMOS_REVISION:
                raise ValueError("GPU acceptance Cosmos revision differs from the pin")
            if acceptance["learning_rate"] != 1e-4 or acceptance["warmup_ratio"] != 0.05:
                raise ValueError("GPU acceptance learning schedule differs from the launch contract")
            if type(acceptance["launch_arguments_sha256"]) is not str or not _SHA256.fullmatch(acceptance["launch_arguments_sha256"]):
                raise ValueError("GPU acceptance launch arguments hash is invalid")
            run_id = acceptance["run_id"]
            if type(run_id) is not str or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", run_id):
                raise ValueError("GPU acceptance run_id is unsafe")
            expected_command = build_b1k_command(
                plan, checkout="/opt/isaac-groot", dataset_path="/workspace/data/b1k",
                base_model_path="/workspace/models/groot", output_dir=f"/workspace/outputs/{run_id}",
                experiment_name=run_id, resume_from_checkpoint=False,
            )
            if acceptance["launch_arguments"] != list(expected_command) or canonical_json_sha256(expected_command) != acceptance["launch_arguments_sha256"]:
                raise ValueError("GPU acceptance launch arguments differ from their hash")
            if acceptance["hardware_model"] not in {"NVIDIA RTX PRO 6000 Blackwell Server Edition", "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"} or type(acceptance["hardware_count"]) is not int or acceptance["hardware_count"] != world_size or acceptance["vram_gib"] != 96:
                raise ValueError("GPU acceptance hardware evidence is invalid")
            if any(acceptance[key] is not True for key in ("cuda_optimizer_step_passed", "checkpoint_roundtrip_passed", "resume_passed", "tiny_overfit_passed", "finite_loss")):
                raise ValueError("GPU acceptance required checks must pass")
            if type(acceptance["acceptance_seconds"]) not in (int, float) or acceptance["acceptance_seconds"] <= 0 or acceptance["acceptance_seconds"] > 1800:
                raise ValueError("GPU acceptance timing must be positive and bounded")
            evidence_uri = _string(acceptance["evidence_uri"], "GPU evidence URI")
            parsed_evidence = urlsplit(evidence_uri)
            if (
                parsed_evidence.scheme != "https"
                or parsed_evidence.hostname is None
                or "." not in parsed_evidence.hostname
                or parsed_evidence.username is not None
                or parsed_evidence.password is not None
                or parsed_evidence.query
                or parsed_evidence.fragment
                or not parsed_evidence.path.strip("/")
            ):
                raise ValueError("GPU evidence URI must be a safe absolute HTTPS URI")
        elif acceptance_status != "pending":
            raise ValueError("unreleased manifest cannot claim passed GPU acceptance")
        elif any(
            (
                acceptance["world_size"] is not None,
                acceptance["run_id"] is not None,
                acceptance["launch_plan_id"] is not None,
                acceptance["effective_global_batch_size"] is not None,
                acceptance["physical_batch_size"] is not None,
                acceptance["global_batch_size"] is not None,
                acceptance["gradient_accumulation_steps"] is not None,
                acceptance["task_manifest_sha256"] is not None,
                acceptance["modality_sha256"] is not None,
                acceptance["stats_sha256"] is not None,
                acceptance["base_model_revision"] is not None,
                acceptance["cosmos_revision"] is not None,
                acceptance["learning_rate"] is not None,
                acceptance["warmup_ratio"] is not None,
                acceptance["launch_arguments_sha256"] is not None,
                acceptance["launch_arguments"] is not None,
                acceptance["hardware_model"] is not None,
                acceptance["hardware_count"] is not None,
                acceptance["vram_gib"] is not None,
                acceptance["cuda_optimizer_step_passed"] is not None,
                acceptance["checkpoint_roundtrip_passed"] is not None,
                acceptance["resume_passed"] is not None,
                acceptance["tiny_overfit_passed"] is not None,
                acceptance["finite_loss"] is not None,
                acceptance["acceptance_seconds"] is not None,
                acceptance["evidence_uri"] is not None,
            )
        ):
            raise ValueError("pending GPU acceptance cannot contain claimed evidence")

        payload = json.loads(json.dumps(root, allow_nan=False))
        return cls(payload=payload)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("release manifest contains a duplicate field")
        result[key] = value
    return result


def load_release_manifest(path: str | Path) -> ReleaseManifest:
    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError("release manifest numbers must be finite")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("release manifest is malformed") from None
    return ReleaseManifest.from_dict(payload)


def pending_manifest(
    *,
    repository_commit: str,
    trainer_lock_sha256: str,
    oci_digest: str | None,
    dataset_revision: str | None = BEHAVIOR_1K_DATASET_REVISION,
    dataset_manifest_sha256: str | None = None,
    normalization_sha256: str | None = None,
) -> ReleaseManifest:
    return ReleaseManifest.from_dict(
        {
            "schema_version": 2,
            "status": "unreleased",
            "platform": "linux/amd64",
            "cuda_base": {"image": CUDA_BASE_IMAGE, "digest": CUDA_BASE_DIGEST},
            "python_version": PYTHON_VERSION,
            "uv_version": UV_VERSION,
            "isaac_groot": {
                "repository": ISAAC_GROOT_REPOSITORY,
                "commit": ISAAC_GROOT_REVISION,
            },
            "base_model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION},
            "cosmos": {"repository": COSMOS_REPOSITORY, "revision": COSMOS_REVISION},
            "dataset": {
                "repository": DATASET_REPOSITORY,
                "revision": dataset_revision,
                "manifest_sha256": dataset_manifest_sha256,
                "normalization_sha256": normalization_sha256,
            },
            "trainer_lock_sha256": trainer_lock_sha256,
            "repository_commit": repository_commit,
            "image": {
                "repository": IMAGE_REPOSITORY,
                "tag": repository_commit,
                "digest": oci_digest,
            },
            "gpu_acceptance": {
                "status": "pending",
                "run_id": None,
                "world_size": None,
                "launch_plan_id": None,
                "effective_global_batch_size": None,
                "physical_batch_size": None,
                "global_batch_size": None,
                "gradient_accumulation_steps": None,
                "task_manifest_sha256": None,
                "modality_sha256": None,
                "stats_sha256": None,
                "base_model_revision": None,
                "cosmos_revision": None,
                "learning_rate": None,
                "warmup_ratio": None,
                "launch_arguments_sha256": None,
                "launch_arguments": None,
                "hardware_model": None,
                "hardware_count": None,
                "vram_gib": None,
                "cuda_optimizer_step_passed": None,
                "checkpoint_roundtrip_passed": None,
                "resume_passed": None,
                "tiny_overfit_passed": None,
                "finite_loss": None,
                "acceptance_seconds": None,
                "evidence_uri": None,
            },
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a strict pending image manifest")
    parser.add_argument("--repository-commit", required=True)
    parser.add_argument("--trainer-lock-path", required=True)
    parser.add_argument("--oci-digest")
    parser.add_argument("--dataset-revision", default=BEHAVIOR_1K_DATASET_REVISION)
    parser.add_argument("--dataset-manifest-sha256")
    parser.add_argument("--normalization-sha256")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = pending_manifest(
        repository_commit=args.repository_commit,
        trainer_lock_sha256=sha256_file(args.trainer_lock_path),
        oci_digest=args.oci_digest,
        dataset_revision=args.dataset_revision,
        dataset_manifest_sha256=args.dataset_manifest_sha256,
        normalization_sha256=args.normalization_sha256,
    )
    atomic_write_json(args.output, manifest.to_dict())


if __name__ == "__main__":
    main()
