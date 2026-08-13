"""Strict immutable provenance for an accepted trainer image."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Mapping
from urllib.parse import urlsplit

from lehome_train.constants import (
    CUDA_BASE_DIGEST,
    ISAAC_GROOT_REVISION,
    MODEL_REVISION,
    PYTHON_VERSION,
)
from lehome_train.io import atomic_write_json, sha256_file


CUDA_BASE_IMAGE = "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04"
ISAAC_GROOT_REPOSITORY = "https://github.com/NVIDIA/Isaac-GR00T.git"
MODEL_REPOSITORY = "nvidia/GR00T-N1.7-3B"
IMAGE_REPOSITORY = "ghcr.io/ryanjin333/lehome-groot-n17-trainer"
DATASET_REPOSITORY = "ryanjin333/lehome-groot-n17-data"
UV_VERSION = "0.8.22"
TRAINER_LOCK_SHA256 = "5df7702213f15d7116e6889cf9cf24d7c9983ca920c22f6852e68c5413b57964"
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


def validate_training_capability(value: object) -> Mapping[str, object]:
    """Admit Blackwell training by real optimizer capability, not rollout R580."""
    capability = _exact_keys(
        value,
        {"hardware", "driver_version", "image_digest", "cuda_runtime", "torch_cuda", "compute_capability", "optimizer_step", "nvml"},
        "training capability",
    )
    if "RTX PRO 6000 Blackwell" not in _string(capability["hardware"], "training hardware"):
        raise ValueError("training capability requires RTX PRO 6000 Blackwell")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", _string(capability["driver_version"], "training driver")):
        raise ValueError("training driver version is invalid")
    if not _DIGEST.fullmatch(_string(capability["image_digest"], "training image digest")):
        raise ValueError("training image digest is invalid")
    for key in ("cuda_runtime", "torch_cuda", "compute_capability"):
        _string(capability[key], key)
    step = capability["optimizer_step"]
    if not isinstance(step, Mapping) or step.get("passed") is not True or type(step.get("loss")) not in (int, float):
        raise ValueError("training capability requires a finite optimizer smoke")
    if not math.isfinite(float(step["loss"])):
        raise ValueError("training capability requires a finite optimizer smoke")
    nvml = capability["nvml"]
    if not isinstance(nvml, Mapping) or type(nvml.get("utilization_percent")) not in (int, float):
        raise ValueError("training capability requires NVML telemetry")
    return capability


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
                "dataset",
                "trainer_lock_sha256",
                "repository_commit",
                "image",
                "gpu_acceptance",
            },
            "root",
        )
        if type(root["schema_version"]) is not int or root["schema_version"] != 1:
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
        if any(value is not None for value in dataset_identity):
            if type(dataset["revision"]) is not str or not _SHA.fullmatch(dataset["revision"]):
                raise ValueError("release manifest dataset revision is not immutable")
            for key in ("manifest_sha256", "normalization_sha256"):
                value = dataset[key]
                if type(value) is not str or not _SHA256.fullmatch(value):
                    raise ValueError(f"release manifest dataset {key} is invalid")

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
                "hardware",
                "network_gbps",
                "image_pull_seconds",
                "first_optimizer_step_seconds",
                "memorization_passed",
                "batches_tested_sequentially",
                "training_768k_started_or_resumed",
                "evidence_uri",
            },
            "gpu_acceptance",
        )
        acceptance_status = _string(acceptance["status"], "GPU acceptance status")
        if acceptance_status not in {"pending", "passed"}:
            raise ValueError("release manifest GPU acceptance status is unsupported")
        network = _optional_number(acceptance["network_gbps"], "network bandwidth")
        first_step = _optional_number(
            acceptance["first_optimizer_step_seconds"], "first optimizer step time"
        )
        image_pull = _optional_number(acceptance["image_pull_seconds"], "image pull time")
        batches = acceptance["batches_tested_sequentially"]
        if not isinstance(batches, list) or any(type(item) is not int for item in batches):
            raise ValueError("release manifest smoke batches must be an integer array")

        if status == "accepted":
            if repository_commit is None:
                raise ValueError("accepted release manifest requires an immutable repository commit")
            if digest is None:
                raise ValueError("accepted release manifest requires a final OCI digest")
            if acceptance_status != "passed":
                raise ValueError("accepted release manifest requires passed GPU acceptance")
            if any(value is None for value in dataset_identity):
                raise ValueError("accepted release manifest requires immutable dataset evidence")
            hardware = _string(acceptance["hardware"], "GPU hardware")
            if "RTX PRO 6000" not in hardware:
                raise ValueError("accepted release manifest requires fresh RTX PRO 6000 evidence")
            if network is None or network < 1.0:
                raise ValueError("GPU acceptance requires at least 1 Gbps bandwidth")
            if image_pull is None or image_pull <= 0:
                raise ValueError("GPU acceptance requires measured image pull timing")
            if first_step is None or first_step > 1800:
                raise ValueError("GPU acceptance must reach the first optimizer step in 30 minutes")
            if acceptance["memorization_passed"] is not True:
                raise ValueError("GPU acceptance requires one-episode memorization")
            if batches != [16, 32, 64]:
                raise ValueError("GPU acceptance requires sequential batches 16, 32, and 64")
            if acceptance["training_768k_started_or_resumed"] is not True:
                raise ValueError("GPU acceptance requires starting or resuming the 768k run")
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
                acceptance["hardware"] is not None,
                network is not None,
                acceptance["image_pull_seconds"] is not None,
                first_step is not None,
                acceptance["memorization_passed"] is not False,
                batches != [],
                acceptance["training_768k_started_or_resumed"] is not False,
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
    dataset_revision: str | None = None,
    dataset_manifest_sha256: str | None = None,
    normalization_sha256: str | None = None,
) -> ReleaseManifest:
    return ReleaseManifest.from_dict(
        {
            "schema_version": 1,
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
                "hardware": None,
                "network_gbps": None,
                "image_pull_seconds": None,
                "first_optimizer_step_seconds": None,
                "memorization_passed": False,
                "batches_tested_sequentially": [],
                "training_768k_started_or_resumed": False,
                "evidence_uri": None,
            },
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a strict pending image manifest")
    parser.add_argument("--repository-commit", required=True)
    parser.add_argument("--trainer-lock-path", required=True)
    parser.add_argument("--oci-digest")
    parser.add_argument("--dataset-revision")
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
