"""Pure persistent-worker manifest authoring.

The live Isaac evaluation loop imports this module after Kit is available.
Keeping the attempt manifest contract independent of Torch and Isaac makes the
same worker-to-recorder handoff verifiable on a controller-only host.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
from typing import Any, Mapping

from .artifacts import atomic_write_json
from .models import EpisodeIdentity


_LOWERCASE_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LOWERCASE_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DEFAULT_POLICY_REPO = "ryanjin333/lehome-groot-n17-models"
_DEFAULT_POLICY_REVISION = "30ac1a84da67b099e115ad147bcd61e9d60046d3"
_DEFAULT_POLICY_STEP = 12000
_DEFAULT_POLICY_ARTIFACT_SHA256 = "3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06"


def persistent_policy_identity(args: argparse.Namespace) -> tuple[str, str, int, str]:
    """Validate the exact checkpoint identity served to one worker."""

    repository = getattr(args, "policy_repo", _DEFAULT_POLICY_REPO)
    revision = getattr(args, "policy_revision", _DEFAULT_POLICY_REVISION)
    step = getattr(args, "policy_step", _DEFAULT_POLICY_STEP)
    artifact_sha256 = getattr(args, "policy_artifact_sha256", _DEFAULT_POLICY_ARTIFACT_SHA256)
    if not isinstance(repository, str) or not repository or any(character.isspace() for character in repository):
        raise ValueError("persistent policy repository is invalid")
    if not isinstance(revision, str) or _LOWERCASE_COMMIT.fullmatch(revision) is None:
        raise ValueError("persistent policy revision must be an immutable commit")
    if type(step) is not int or step <= 0:
        raise ValueError("persistent policy step must be a positive integer")
    if not isinstance(artifact_sha256, str) or _LOWERCASE_SHA256.fullmatch(artifact_sha256) is None:
        raise ValueError("persistent policy artifact SHA-256 is invalid")
    return repository, revision, step, artifact_sha256


def persistent_collection_strategy(assignment: Mapping[str, Any]) -> str:
    """Resolve the recorder strategy without widening collection scope."""

    explicit = assignment.get("strategy")
    if explicit is None:
        return "mild_geometry" if assignment.get("difficulty") == "randomized" else "canonical"
    if explicit == "visual_only":
        if os.environ.get("LEHOME_SUCCESS_REPLAY_CAMPAIGN") != "1":
            raise ValueError("visual-only randomization requires a success replay campaign")
        from .recovery_collection import validate_success_replay_descriptor

        validate_success_replay_descriptor(assignment)
        return "visual_only"
    if explicit in {"mild", "strong"}:
        raise ValueError("persistent collection only supports geometry-only randomization")
    if explicit not in {"canonical", "mild_geometry", "strong_geometry"}:
        raise ValueError("persistent collection has an unsupported randomization strategy")
    return str(explicit)


def write_persistent_flywheel_manifest(
    attempt_output_dir: Path,
    assignment: Mapping[str, Any],
    args: argparse.Namespace,
    *,
    verified_restore: Mapping[str, object] | None = None,
) -> Path:
    """Author one attempt-scoped autonomous-recorder contract."""

    garment = assignment.get("garment", assignment.get("garment_name"))
    seed = assignment.get("seed")
    category = assignment.get("category")
    release_stage = assignment.get("release_stage", "seen")
    attempt_id = assignment.get("attempt_id") or assignment.get("trial_id")
    if not isinstance(garment, str) or not garment:
        raise ValueError("persistent flywheel assignment requires a garment")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("persistent flywheel assignment requires a non-negative seed")
    if not isinstance(category, str) or not category:
        raise ValueError("persistent flywheel assignment requires a category")
    if not isinstance(attempt_id, str) or not attempt_id:
        raise ValueError("persistent flywheel assignment requires an attempt id")
    policy_repo, policy_revision, policy_step, policy_artifact_sha256 = persistent_policy_identity(args)
    campaign_round_id = getattr(args, "campaign_round_id", None)
    campaign_run_id = getattr(args, "campaign_run_id", None)
    if (campaign_round_id is None) != (campaign_run_id is None):
        raise ValueError("persistent flywheel assignment requires paired campaign provenance")
    if campaign_round_id is not None and (
        not isinstance(campaign_round_id, str) or not campaign_round_id
        or not isinstance(campaign_run_id, str) or not campaign_run_id
    ):
        raise ValueError("persistent flywheel campaign provenance is invalid")
    strategy = persistent_collection_strategy(assignment)
    identity = EpisodeIdentity(
        episode_id=attempt_id,
        policy_repo=policy_repo,
        policy_revision=policy_revision,
        policy_step=policy_step,
        code_revision="61e60d18dcda662b144d1cc0fb05fa2beec82033",
        asset_revision="bea65fd960ad5a1bb3bd3fa77164b28001c08ef9",
        simulator_version="5.1.0.0",
        garment_name=garment,
        category=category,
        release_stage=str(release_stage),
        seed=seed,
        instruction="fold the garment on the table",
        strategy=strategy,
        campaign_round_id=campaign_round_id,
        campaign_run_id=campaign_run_id,
    )
    simulator_device = str(getattr(args, "device", "")).lower()
    if simulator_device != "cpu" and re.fullmatch(r"cuda:[0-9]+", simulator_device) is None:
        raise ValueError("persistent flywheel assignment requires cpu or a canonical CUDA simulator device")
    path = attempt_output_dir / "flywheel-manifest.json"
    payload: dict[str, object] = {
        "schema_version": 2 if campaign_round_id is not None else 1,
        "policy_revision": identity.policy_revision,
        "seed": identity.seed,
        "garment": identity.garment_name,
        "strategy": identity.strategy,
        "episode_id": identity.episode_id,
        "identity": {
            "episode_id": identity.episode_id,
            "policy_repo": identity.policy_repo,
            "policy_revision": identity.policy_revision,
            "policy_step": identity.policy_step,
            "code_revision": identity.code_revision,
            "asset_revision": identity.asset_revision,
            "simulator_version": identity.simulator_version,
            "garment_name": identity.garment_name,
            "category": identity.category,
            "release_stage": identity.release_stage,
            "seed": identity.seed,
            "instruction": identity.instruction,
            "strategy": identity.strategy,
        },
        "policy_artifact_sha256": policy_artifact_sha256,
        "image_identity": "sha256:afb35941768cabfe2f18173df27190b78a5b3044fbbbe71c3029539ffbc821d7",
        "execution_mode": "policy_server",
        "execution_backend": "policy_server",
        "simulator_device": simulator_device,
        "policy_device": str(getattr(args, "policy_device", "cuda:0")),
        "parity_stage": "server_cpu" if simulator_device == "cpu" else "persistent_collection",
    }
    if campaign_round_id is not None:
        identity_payload = payload["identity"]
        assert isinstance(identity_payload, dict)
        identity_payload.update({"campaign_round_id": campaign_round_id, "campaign_run_id": campaign_run_id})
    if verified_restore is not None:
        payload.update(verified_restore)
    if assignment.get("recovery_kind") == "controlled_success_recovery_snapshot_v3":
        controlled_keys = {
            "recovery_kind", "category", "garment", "source_round_id", "source_episode_id", "source_episode_digest",
            "source_immutable_revision", "source_reset", "source_reset_sha256",
            "source_annotations", "source_annotations_sha256", "source_first_success_step",
            "source_continuation_snapshot", "source_continuation_snapshot_sha256",
            "source_continuation_snapshot_relative_path", "prefix_stop", "perturbation_profile",
            "perturbation_seed", "source_seed", "source_continuation_state", "source_state_fingerprint", "source_snapshot_schema_version", "source_snapshot_authority", "source_only_envelope", "perturbation_fingerprint",
            "source_state_perturbation_fingerprint", "category_acceptance_cap",
            "controlled_smoke", "controlled_smoke_teacher_probe",
        }
        payload["controlled_recovery"] = {key: assignment[key] for key in controlled_keys if key in assignment}
    attempt_output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)
    return path


__all__ = [
    "persistent_collection_strategy",
    "persistent_policy_identity",
    "write_persistent_flywheel_manifest",
]
