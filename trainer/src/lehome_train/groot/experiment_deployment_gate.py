"""Immutable admission proof required before the sweep may spend GPU time."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import stat
from typing import Mapping


_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_INSTANCE = re.compile(r"computeinstance-[a-z0-9]+")
_IMAGE = re.compile(r"computeimage-[a-z0-9]+")
_SAFE = re.compile(r"[A-Za-z0-9_.-]+")

# One immutable identity connects deployment admission, evaluator leases, and
# capacity idle detection.  Keeping it here prevents those production paths
# from silently drifting apart.
PRODUCTION_EVALUATOR_WORKER_ID = "lehome-experiment-evaluator"


@dataclass(frozen=True, slots=True)
class DeploymentGate:
    sha256: str
    controller_instance_id: str
    controller_image_id: str
    training_instance_ids: tuple[str, str]
    training_image_id: str
    training_oci_digest: str
    training_code_revision: str
    rollout_instance_id: str
    rollout_image_id: str
    teacher_probe_receipt_sha256: str


@dataclass(frozen=True, slots=True)
class TrainingImageManifest:
    """The OCI identity baked into the immutable training host image."""

    oci_image: str
    oci_digest: str
    trainer_code_revision: str


def _exact(value: object, fields: set[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"deployment gate {label} is malformed")
    return value


def _identity(value: object, pattern: re.Pattern[str], label: str) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise ValueError(f"deployment gate {label} is invalid")
    return value


def _image_bound(value: object, *, worker: bool) -> tuple[str, str, str | None, int | None]:
    fields = {"instance_id", "image_id", "image_status", "readback_verified"}
    if worker:
        fields |= {"worker_id"}
    raw = _exact(value, fields | ({"slot"} if isinstance(value, Mapping) and "slot" in value else set()), "image binding")
    instance_id = _identity(raw["instance_id"], _INSTANCE, "instance identity")
    image_id = _identity(raw["image_id"], _IMAGE, "image identity")
    if raw["image_status"] != "READY" or raw["readback_verified"] is not True:
        raise ValueError("deployment gate image is not READY and read-back verified")
    worker_id = None
    if worker:
        worker_id = _identity(raw["worker_id"], _SAFE, "worker identity")
    slot = raw.get("slot")
    if slot is not None and (type(slot) is not int or slot not in (1, 2)):
        raise ValueError("deployment gate training slot is invalid")
    return instance_id, image_id, worker_id, slot


def load_deployment_gate(path: str | Path, expected_sha256: str) -> DeploymentGate:
    source = Path(path)
    if type(expected_sha256) is not str or _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("deployment gate SHA-256 is invalid")
    if (
        not source.is_absolute()
        or source.is_symlink()
        or not source.is_file()
        or stat.S_IMODE(source.stat().st_mode) != 0o444
    ):
        raise ValueError("deployment gate must be an immutable regular file")
    payload = source.read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected_sha256:
        raise ValueError("deployment gate SHA-256 mismatch")
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("deployment gate JSON is invalid") from error
    root = _exact(
        document,
        {
            "schema_version", "kind", "controller", "training_workers",
            "training_oci_digest", "training_code_revision", "rollout_worker", "teacher_probe",
        },
        "envelope",
    )
    if root["schema_version"] != 1 or root["kind"] != "lehome_experiment_pool_deployment_gate":
        raise ValueError("deployment gate schema is unsupported")
    training_oci_digest = _identity(root["training_oci_digest"], re.compile(r"sha256:[0-9a-f]{64}"), "training OCI digest")
    training_code_revision = _identity(root["training_code_revision"], _COMMIT, "training code revision")
    controller_id, controller_image, _, controller_slot = _image_bound(root["controller"], worker=False)
    if controller_slot is not None:
        raise ValueError("deployment gate controller slot is invalid")
    training = root["training_workers"]
    if not isinstance(training, list) or len(training) != 2:
        raise ValueError("deployment gate requires exactly two training workers")
    parsed = [_image_bound(item, worker=True) for item in training]
    if [item[3] for item in parsed] != [1, 2]:
        raise ValueError("deployment gate training slots are invalid")
    if [item[2] for item in parsed] != ["lehome-experiment-training-1", "lehome-experiment-training-2"]:
        raise ValueError("deployment gate training worker identities are invalid")
    if parsed[0][1] != parsed[1][1]:
        raise ValueError("deployment gate training images differ")
    rollout_id, rollout_image, rollout_worker, rollout_slot = _image_bound(root["rollout_worker"], worker=True)
    if rollout_worker != PRODUCTION_EVALUATOR_WORKER_ID or rollout_slot is not None:
        raise ValueError("deployment gate rollout worker identity is invalid")
    instance_ids = (controller_id, parsed[0][0], parsed[1][0], rollout_id)
    if len(set(instance_ids)) != len(instance_ids):
        raise ValueError("deployment gate instance identities are duplicated")
    if len({controller_image, parsed[0][1], rollout_image}) != 3:
        raise ValueError("deployment gate role images are not distinct")
    teacher = _exact(
        root["teacher_probe"],
        {
            "kind", "attempt_id", "round_id", "matrix_sha256", "materialization_sha256",
            "episode_sha256", "sync_receipt_sha256", "immutable_revision", "accepted",
            "readback_verified", "strict_seal_present",
        },
        "teacher probe",
    )
    if (
        teacher["kind"] != "zero_perturbation_teacher_continuation_probe_v1"
        or teacher["accepted"] is not True
        or teacher["readback_verified"] is not True
        or teacher["strict_seal_present"] is not False
        or type(teacher["round_id"]) is not str
        or not teacher["round_id"].startswith("controlled-recovery-smoke-")
        or not teacher["round_id"].endswith("-unsealed-staging")
    ):
        raise ValueError("deployment gate teacher probe is not accepted and read-back verified")
    for field in ("attempt_id", "matrix_sha256", "materialization_sha256", "episode_sha256", "sync_receipt_sha256"):
        _identity(teacher[field], _SHA256, f"teacher probe {field}")
    _identity(teacher["immutable_revision"], _COMMIT, "teacher probe revision")
    return DeploymentGate(
        sha256=observed,
        controller_instance_id=controller_id,
        controller_image_id=controller_image,
        training_instance_ids=(parsed[0][0], parsed[1][0]),
        training_image_id=parsed[0][1],
        training_oci_digest=training_oci_digest,
        training_code_revision=training_code_revision,
        rollout_instance_id=rollout_id,
        rollout_image_id=rollout_image,
        teacher_probe_receipt_sha256=str(teacher["sync_receipt_sha256"]),
    )


def load_training_image_manifest(path: str | Path) -> TrainingImageManifest:
    """Read the root-owned immutable OCI record baked into a training image.

    Ownership is checked by the root-only worker wrapper before it invokes the
    Python worker.  This parser repeats the non-symlink/immutable-byte checks
    so a runtime environment variable cannot redirect it to job-controlled
    data.
    """
    source = Path(path)
    if (
        not source.is_absolute()
        or source.is_symlink()
        or not source.is_file()
        or stat.S_IMODE(source.stat().st_mode) != 0o444
    ):
        raise ValueError("training image manifest must be an immutable regular file")
    try:
        document = json.loads(source.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("training image manifest JSON is invalid") from error
    value = _exact(
        document,
        {"schema_version", "kind", "oci_image", "oci_digest", "trainer_code_revision"},
        "training image manifest",
    )
    if value["schema_version"] != 1 or value["kind"] != "vla-training-base-image":
        raise ValueError("training image manifest schema is unsupported")
    digest = _identity(value["oci_digest"], re.compile(r"sha256:[0-9a-f]{64}"), "training image OCI digest")
    image = value["oci_image"]
    if type(image) is not str or re.fullmatch(r"[^@\s]+@" + re.escape(digest), image) is None:
        raise ValueError("training image OCI image is invalid")
    revision = _identity(value["trainer_code_revision"], _COMMIT, "training image code revision")
    return TrainingImageManifest(image, digest, revision)


def bind_training_job_identity(
    job: object,
    deployment: DeploymentGate,
    training_image: TrainingImageManifest,
) -> None:
    """Bind a leased job to immutable compute and baked OCI identities.

    The read-back deployment gate declares both the compute-image and OCI
    identities.  The worker then proves the baked host manifest has the same
    OCI digest. A job can state either value but cannot supply the value that
    is trusted.
    """
    trainer = getattr(job, "trainer", None)
    if not isinstance(trainer, Mapping):
        raise ValueError("training job has no immutable trainer identity")
    if trainer.get("image_id") != deployment.training_image_id:
        raise ValueError("training image identity does not match deployment gate")
    if trainer.get("oci_digest") != deployment.training_oci_digest:
        raise ValueError("training OCI identity does not match deployment gate")
    if trainer.get("code_revision") != deployment.training_code_revision:
        raise ValueError("training code revision does not match deployment gate")
    if training_image.oci_digest != deployment.training_oci_digest:
        raise ValueError("baked training OCI identity does not match deployment gate")
    if trainer.get("oci_digest") != training_image.oci_digest:
        raise ValueError("training OCI identity does not match baked image manifest")
    if training_image.trainer_code_revision != deployment.training_code_revision:
        raise ValueError("baked training code revision does not match deployment gate")
    if trainer.get("code_revision") != training_image.trainer_code_revision:
        raise ValueError("training code revision does not match baked image manifest")
