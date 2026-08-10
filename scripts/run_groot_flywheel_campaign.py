"""Resumable, local-only supervisor for isolated GR00T rollout processes."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import socket
import stat
import subprocess
import sys
import time
from typing import Callable, Sequence
from uuid import uuid4

from lehome.flywheel.artifacts import verify_episode_manifest
from lehome.flywheel.capacity import CapacityDecision, CapacitySample, choose_worker_count
from lehome.flywheel.isaac_recorder import CANONICAL_VIDEO_FILENAMES
from lehome.flywheel.matrix import Trial, load_public_matrix, matrix_sha256
from lehome.flywheel.parity import (
    AbortGate,
    EpisodeGateEvidence,
    HISTORICAL_CONTROL_IDS,
    assess_reset_diversity,
    evaluate_abort_gate,
    evaluate_cpu_scale_ladder,
    evaluate_parity_ladder,
    historical_control_trials,
)
from lehome.flywheel.runtime_preflight import require_isaac_sim_5_1_runtime


@dataclass(frozen=True, slots=True)
class CampaignState:
    output_root: Path
    trial_ids: tuple[str, ...]


_LEGACY_SHARED_POLICY_HASHES = {
    "run-lehome-24-shared.sh": "bbf8fee87d7efc4e09b08874e3265175fd7a4c9ea9494be8ac7e8301fd4d7f92",
    "eval_groot_n17_matrix_parallel.py": "e26d63536a6eef53fe6d0de8a22ee683616aa1b5ba4aa4dc968d4eb13a37f89a",
    "groot_policy.py": "cee2d9f78711e867ef4e4867ee615abdbfe5584e3385c3b601adfe90f25d78bf",
    "serve_groot_policy.py": "b8aa5f81e651e1db18f4189e55121e0eca67ca7613a58db361ae88753a8cb3e4",
}

_PUBLIC_UNSEEN_TOP_CATEGORIES = ("top_long", "top_short")
_PUBLIC_UNSEEN_TOP_CATEGORY_COUNT = 20
_PUBLIC_UNSEEN_TOP_TRIAL_COUNT = 40
_TOP40_EVALUATION_INVOCATION = "checkpoint-evaluation-invocation.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_receipt_file(root: Path, relative: object, digest: object) -> Path:
    """Return one checksummed receipt artifact without following any symlink."""
    if not isinstance(relative, str) or not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
        raise ValueError("receipt artifact path or checksum is invalid")
    path = Path(relative)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("receipt artifact path is unsafe")
    candidate = root.joinpath(*path.parts)
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("receipt artifact must not traverse a symlink")
    if not candidate.is_file() or _sha256_file(candidate) != digest:
        raise ValueError("receipt artifact integrity failed")
    return candidate


def _historical_config_root(args: argparse.Namespace, trial) -> Path | None:
    root = getattr(args, "historical_control_root", None)
    if root is None:
        return None
    candidate = Path(root) / trial.trial_id / "garment-config"
    release_list = candidate / "Release" / "Release_test_list.txt"
    if candidate.is_symlink() or not candidate.is_dir() or release_list.is_symlink() or not release_list.is_file():
        raise ValueError("historical control bundle lacks a frozen garment config")
    if release_list.read_text(encoding="utf-8").strip() != trial.garment_name:
        raise ValueError("historical control frozen garment config does not match trial garment")
    return candidate


def _validate_historical_control_bundle(root: Path, trials: Sequence[object]) -> None:
    """Verify the frozen per-trial configs against the archived bundle manifest."""
    manifest = root / "SHA256SUMS"
    if root.is_symlink() or not root.is_dir() or manifest.is_symlink() or not manifest.is_file():
        raise ValueError("historical control root must be a checksummed real directory")
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        try:
            digest, relative = line.split(maxsplit=1)
        except ValueError as error:
            raise ValueError("historical control SHA256SUMS is malformed") from error
        normalized = relative.removeprefix("./")
        if normalized in expected:
            raise ValueError("historical control SHA256SUMS has duplicate paths")
        expected[normalized] = digest
    # The bundle is the frozen input, not a loose source of selected files:
    # validate every listed config artifact before choosing the twelve trials.
    for relative, digest in expected.items():
        _verified_receipt_file(root, relative, digest)
    for trial in trials:
        relative = f"{trial.trial_id}/garment-config/Release/Release_test_list.txt"
        path = root / relative
        if expected.get(relative) is None or path.is_symlink() or not path.is_file() or _sha256_file(path) != expected[relative]:
            raise ValueError("historical control frozen garment config integrity check failed")
        if path.read_text(encoding="utf-8").strip() != trial.garment_name:
            raise ValueError("historical control frozen garment config does not match trial garment")


def _validate_legacy_shared_policy_receipt(path: Path) -> dict[str, object]:
    payload = _read_parity_receipt(path)
    if payload.get("parity_stage") != "legacy_server_cpu" or payload.get("backend") != "legacy_shared_policy_server":
        raise ValueError("legacy receipt must identify the legacy shared policy-server backend")
    if payload.get("source_sha256") != _LEGACY_SHARED_POLICY_HASHES:
        raise ValueError("legacy receipt does not bind the verified shared-policy launcher and adapter hashes")
    archive_root = payload.get("archive_root")
    if not isinstance(archive_root, str):
        raise ValueError("legacy receipt must bind the archived rollout bundle")
    root = Path(archive_root)
    sums = root / "SHA256SUMS"
    report = root / "rollout-report.json"
    if root.is_symlink() or not sums.is_file() or sums.is_symlink() or not report.is_file() or report.is_symlink():
        raise ValueError("legacy archived rollout bundle is unavailable")
    if payload.get("archive_sha256sums_sha256") != _sha256_file(sums) or payload.get("archive_rollout_report_sha256") != _sha256_file(report):
        raise ValueError("legacy receipt archived bundle hashes do not match")
    expected: dict[str, str] = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        expected[relative.removeprefix("./")] = digest
    for relative, digest in expected.items():
        _verified_receipt_file(root, relative, digest)
    archive = json.loads(report.read_text(encoding="utf-8"))
    archived_ids = [item.get("trial", {}).get("trial_id") for item in archive.get("trials", ())]
    if [trial_id for trial_id in archived_ids if trial_id in HISTORICAL_CONTROL_IDS] != list(HISTORICAL_CONTROL_IDS):
        raise ValueError("legacy archive does not bind the exact historical twelve IDs")
    control_root = payload.get("historical_control_root")
    if not isinstance(control_root, str):
        raise ValueError("legacy receipt must bind the full frozen historical config set")
    controls = Path(control_root)
    _validate_historical_control_bundle(controls, historical_control_trials())
    control_sums = controls / "SHA256SUMS"
    control_sums_sha256 = _sha256_file(control_sums)
    if payload.get("historical_control_sha256sums_sha256") != control_sums_sha256:
        raise ValueError("legacy receipt frozen config SHA256SUMS identity does not match")
    reproduction_root = payload.get("reproduction_root")
    terminal_records = payload.get("terminal_records")
    disclosure = payload.get("reproduction")
    if not isinstance(reproduction_root, str) or not isinstance(terminal_records, list) or not isinstance(disclosure, dict):
        raise ValueError("legacy receipt requires newly generated reproduction artifacts and disclosure")
    expected_disclosure = {
        "backend": "legacy_shared_policy_server",
        "environment_device": "cpu",
        "reference_isaac_workers": 6,
        "reference_policy_servers": 2,
        "actual_isaac_workers": 2,
        "actual_policy_servers": 2,
        "gpu_count": 4,
        "concurrency_non_parity": True,
    }
    if any(disclosure.get(key) != value for key, value in expected_disclosure.items()):
        raise ValueError("legacy receipt must disclose the capped four-GPU reproduction and non-parity concurrency")
    reproduced = Path(reproduction_root)
    if reproduced == root or reproduced.resolve() == root.resolve() or reproduced.is_symlink() or not reproduced.is_dir():
        raise ValueError("legacy reproduction evidence must be a new, separate real directory")
    if len(terminal_records) != len(HISTORICAL_CONTROL_IDS):
        raise ValueError("legacy receipt must include exactly twelve terminal artifact records")
    seen = set()
    derived_successes = 0
    expected_binding = {
        "source_sha256": _LEGACY_SHARED_POLICY_HASHES,
        "archive_sha256sums_sha256": _sha256_file(sums),
        "archive_rollout_report_sha256": _sha256_file(report),
        "historical_control_sha256sums_sha256": control_sums_sha256,
    }
    for record in terminal_records:
        if not isinstance(record, dict) or record.get("trial_id") not in HISTORICAL_CONTROL_IDS:
            raise ValueError("legacy receipt terminal record IDs are invalid")
        trial_id = record["trial_id"]
        if trial_id in seen:
            raise ValueError("legacy receipt terminal artifact records must not duplicate trial IDs")
        seen.add(trial_id)
        verified: dict[str, Path] = {}
        for key in ("terminal_record", "log", "manifest"):
            relative, digest = record.get(f"{key}_path"), record.get(f"{key}_sha256")
            verified[key] = _verified_receipt_file(reproduced, relative, digest)
        try:
            terminal = json.loads(verified["terminal_record"].read_text(encoding="utf-8"))
            manifest = json.loads(verified["manifest"].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("legacy terminal record and manifest must be readable JSON") from error
        manifest_trial_id = manifest.get("episode_id") or manifest.get("trial_id")
        if isinstance(manifest.get("identity"), dict):
            manifest_trial_id = manifest_trial_id or manifest["identity"].get("episode_id")
        if (
            not isinstance(terminal, dict)
            or terminal.get("trial_id") != trial_id
            or terminal.get("terminal") is not True
            or terminal.get("backend") != "legacy_shared_policy_server"
            or terminal.get("environment_device") != "cpu"
            or manifest_trial_id != trial_id
        ):
            raise ValueError("legacy reproduction terminal evidence is not attributable to its trial/backend")
        if terminal.get("integrity_binding") != expected_binding:
            raise ValueError("legacy reproduction terminal is not bound to source/archive/frozen-config identities")
        if not isinstance(terminal.get("outcome"), str) or not isinstance(terminal.get("accepted_success"), bool):
            raise ValueError("legacy reproduction terminal lacks a verifiable outcome")
        if terminal["outcome"] == "success" and terminal["accepted_success"]:
            derived_successes += 1
    if seen != set(HISTORICAL_CONTROL_IDS):
        raise ValueError("legacy receipt must include terminal artifacts for the exact historical twelve IDs")
    if payload.get("official_successes") != derived_successes:
        raise ValueError("legacy receipt claimed successes do not match reproduction terminals")
    validated = dict(payload)
    validated["official_successes"] = derived_successes
    bundle_files = {
        "legacy_receipt": _sha256_file(path),
        "archive_sha256sums": _sha256_file(sums),
        "archive_rollout_report": _sha256_file(report),
        "historical_control_sha256sums": _sha256_file(control_sums),
    }
    for relative, digest in expected.items():
        bundle_files[f"archive/{relative}"] = digest
    for relative, digest in {
        line.split(maxsplit=1)[1].removeprefix("./"): line.split(maxsplit=1)[0]
        for line in control_sums.read_text(encoding="utf-8").splitlines() if len(line.split(maxsplit=1)) == 2
    }.items():
        bundle_files[f"historical-control/{relative}"] = digest
    for record in terminal_records:
        assert isinstance(record, dict)
        trial_id = str(record["trial_id"])
        for key in ("terminal_record", "log", "manifest"):
            bundle_files[f"reproduction/{trial_id}/{key}"] = str(record[f"{key}_sha256"])
    validated["bundle_sha256"] = _canonical_sha256(bundle_files)
    return validated


def selected_trials(args: argparse.Namespace, matrix) -> tuple[Trial, ...]:
    """Keep the public 280 matrix immutable while exposing its exact historical control."""
    if getattr(args, "historical_control", False):
        return historical_control_trials()  # type: ignore[return-value]
    if getattr(args, "public_unseen_tops", False):
        trials = tuple(
            trial for trial in matrix.trials
            if trial.release_stage == "public_unseen" and trial.category in _PUBLIC_UNSEEN_TOP_CATEGORIES
        )
        category_counts = {
            category: sum(trial.category == category for trial in trials)
            for category in _PUBLIC_UNSEEN_TOP_CATEGORIES
        }
        if (
            len(matrix.trials) != 280
            or len(trials) != _PUBLIC_UNSEEN_TOP_TRIAL_COUNT
            or category_counts != {category: _PUBLIC_UNSEEN_TOP_CATEGORY_COUNT for category in _PUBLIC_UNSEEN_TOP_CATEGORIES}
        ):
            raise ValueError("public-unseen tops selection must be exactly 20 top_long and 20 top_short trials from the canonical public 280 matrix")
        return trials
    return matrix.trials


def _selection_metadata(args: argparse.Namespace, trials: Sequence[Trial]) -> dict[str, object]:
    if getattr(args, "public_unseen_tops", False):
        return {
            "kind": "public_unseen_tops_evaluation",
            "classification": "diagnostic_evaluation_only_not_training_or_production_release",
            "rft_data_eligible": False,
            "trial_count": len(trials),
            "trial_ids": [trial.trial_id for trial in trials],
            "category_counts": {
                category: sum(trial.category == category for trial in trials)
                for category in _PUBLIC_UNSEEN_TOP_CATEGORIES
            },
            "parity_stage": args.parity_stage,
        }
    return {
        "kind": "historical_control" if args.historical_control else "canonical_public_matrix",
        "trial_count": len(trials),
        "trial_ids": [trial.trial_id for trial in trials],
        "parity_stage": args.parity_stage,
    }


def _top40_evaluation_invocation(
    args: argparse.Namespace,
    matrix,
    trials: Sequence[Trial],
) -> dict[str, object]:
    """Bind one diagnostic run before its root can be resumed."""
    if getattr(args, "historical_control", False) and getattr(args, "public_unseen_tops", False):
        raise ValueError("cannot combine --historical-control and --public-unseen-tops")
    if not getattr(args, "public_unseen_tops", False):
        raise ValueError("top-40 evaluation invocation requires --public-unseen-tops")
    if args.parity_stage is not None or args.execution_mode != "policy_server" or args.device != "cpu" or args.workers != 4:
        raise ValueError("top-40 evaluation requires diagnostic policy-server CPU execution with exactly four workers")
    if len(trials) != _PUBLIC_UNSEEN_TOP_TRIAL_COUNT:
        raise ValueError("top-40 evaluation invocation must bind exactly 40 selected trials")
    groot = _live_groot_identity(args)
    return {
        "schema_version": 1,
        "kind": "public_unseen_tops_checkpoint_evaluation",
        "matrix_sha256": matrix_sha256(matrix),
        "selected_trial_ids": [trial.trial_id for trial in trials],
        "policy_repo": args.policy_repo,
        "policy_revision": _read_current_policy_revision(args.policy_revision_file),
        "policy_step": args.policy_step,
        "policy_artifact_sha256": args.policy_artifact_sha256,
        "code_revision": args.code_revision,
        "asset_revision": args.asset_revision,
        "simulator_version": args.simulator_version,
        "image_identity": args.image_identity,
        "execution_mode": args.execution_mode,
        "simulator_device": args.device,
        "policy_device_pool": [f"cuda:{index}" for index in range(args.workers)],
        "workers": args.workers,
        "strategy": args.strategy,
        "max_steps": args.max_steps,
        **groot,
    }


def _verify_or_write_top40_evaluation_invocation(
    output_root: Path,
    invocation: dict[str, object],
) -> Path:
    path = Path(output_root) / _TOP40_EVALUATION_INVOCATION
    if path.exists() or path.is_symlink():
        if _regular_json(path, label="top-40 evaluation invocation") != invocation:
            raise ValueError("top-40 evaluation invocation does not match this resume identity")
    else:
        _write_json_atomically(path, invocation)
    return path


def _validate_top40_evaluation_output(
    args: argparse.Namespace,
    state: CampaignState,
    invocation: dict[str, object],
    trials: Sequence[Trial],
) -> None:
    """Reject mixed or foreign completed top-40 artifacts before scheduling."""
    root = Path(args.output_root)
    raw = root / "raw"
    if raw.exists() and (raw.is_symlink() or not raw.is_dir()):
        raise ValueError("top-40 evaluation raw root is unsafe")
    by_id = {trial.trial_id: trial for trial in trials}
    trial_ids = set(by_id)
    for receipt_path in root.glob("policy-server-receipt-*.json"):
        trial_id = receipt_path.name.removeprefix("policy-server-receipt-").removesuffix(".json")
        if receipt_path.is_symlink() or not receipt_path.is_file() or trial_id not in trial_ids:
            raise ValueError("top-40 evaluation root contains an extra or foreign policy-server receipt")
    if not raw.exists():
        return
    for episode_root in raw.iterdir():
        if episode_root.is_symlink() or not episode_root.is_dir() or episode_root.name not in trial_ids:
            raise ValueError("top-40 evaluation raw root contains an extra or foreign trial artifact")
        trial_id = episode_root.name
        if not is_completed_trial(root, trial_id):
            continue
        episode, _manifest = verify_episode_manifest(episode_root)
        trial = by_id[trial_id]
        identity = episode.get("identity")
        provenance = episode.get("provenance")
        expected_identity = {
            "episode_id": trial.trial_id, "policy_repo": invocation["policy_repo"],
            "policy_revision": invocation["policy_revision"], "policy_step": invocation["policy_step"],
            "code_revision": invocation["code_revision"], "asset_revision": invocation["asset_revision"],
            "simulator_version": invocation["simulator_version"], "garment_name": trial.garment_name,
            "category": trial.category, "release_stage": trial.release_stage, "seed": trial.seed,
            "instruction": "fold the garment on the table", "strategy": invocation["strategy"],
        }
        if not isinstance(identity, dict) or any(identity.get(key) != value for key, value in expected_identity.items()):
            raise ValueError("top-40 evaluation episode identity does not match its immutable invocation")
        if not isinstance(provenance, dict) or any(provenance.get(key) != value for key, value in {
            "execution_mode": "policy_server", "execution_backend": "policy_server",
            "simulator_device": invocation["simulator_device"], "policy_artifact_sha256": invocation["policy_artifact_sha256"],
            "image_identity": invocation["image_identity"],
        }.items()):
            raise ValueError("top-40 evaluation episode provenance is foreign or mismatched")
        if provenance.get("policy_device") not in invocation["policy_device_pool"]:
            raise ValueError("top-40 evaluation episode policy device is outside the immutable worker topology")
        receipt = _regular_json(root / f"policy-server-receipt-{trial_id}.json", label="top-40 evaluation policy-server receipt")
        expected_receipt = {
            "schema_version": 1, "episode_id": trial_id, "parity_stage": None, "backend": "policy_server",
            "groot_revision": invocation["groot_revision"], "python_path": invocation["groot_python"],
            "python_version": invocation["groot_python_version"], "checkpoint_revision": invocation["policy_revision"],
            "checkpoint_digest": invocation["policy_artifact_sha256"], "code_revision": invocation["code_revision"],
            "image_identity": invocation["image_identity"], "simulator_device": invocation["simulator_device"],
            "policy_seed": trial.seed,
        }
        if any(receipt.get(key) != value for key, value in expected_receipt.items()) or receipt.get("policy_device") != provenance.get("policy_device"):
            raise ValueError("top-40 evaluation policy-server receipt does not match the episode and immutable invocation")


def _top40_final_bindings(output_root: Path, state: CampaignState, invocation: dict[str, object], trials: Sequence[Trial]) -> dict[str, object]:
    manifests = []
    receipts = []
    for trial_id in state.trial_ids:
        manifest = Path(output_root) / "raw" / trial_id / "SHA256SUMS.json"
        receipt = Path(output_root) / f"policy-server-receipt-{trial_id}.json"
        if manifest.is_symlink() or not manifest.is_file() or receipt.is_symlink() or not receipt.is_file():
            raise ValueError("top-40 evaluation final close requires every episode manifest and policy-server receipt")
        manifests.append({"trial_id": trial_id, "path": f"raw/{trial_id}/SHA256SUMS.json", "sha256": _sha256_file(manifest)})
        receipts.append({"trial_id": trial_id, "path": receipt.name, "sha256": _sha256_file(receipt)})
    if len(manifests) != 40 or len(receipts) != 40:
        raise ValueError("top-40 evaluation final close requires exactly 40 episode manifests and receipts")
    return {"invocation": invocation, "episode_manifests": manifests, "policy_server_receipts": receipts, "metrics": _top40_evaluation_metrics(output_root, trials)}


def _top40_evaluation_metrics(output_root: Path, trials: Sequence[Trial]) -> dict[str, object]:
    """Aggregate the exact selected trials only from checksum-verified episodes."""
    def aggregate(items: Sequence[Trial]) -> dict[str, object]:
        evidence = [_episode_gate_evidence(Path(output_root), trial.trial_id) for trial in items]
        successes = sum(item.official_success for item in evidence)
        return {"episodes": len(items), "official_successes": successes, "success_rate": successes / len(items), "visible_contact_count": sum(item.visible_contact for item in evidence)}
    by_category = {category: tuple(trial for trial in trials if trial.category == category) for category in _PUBLIC_UNSEEN_TOP_CATEGORIES}
    if len(trials) != 40 or any(len(items) != 20 for items in by_category.values()):
        raise ValueError("top-40 evaluation metrics require the exact canonical category counts")
    overall = aggregate(trials)
    return {**overall, "per_category": {category: aggregate(items) for category, items in by_category.items()}}


def _episode_gate_evidence(output_root: Path, trial_id: str) -> EpisodeGateEvidence:
    """Read gate inputs only from checksum-verified, terminal episode artifacts."""
    episode, _manifest = verify_episode_manifest(Path(output_root) / "raw" / trial_id)
    contact = episode.get("visible_contact")
    visible_contact = (
        isinstance(contact, dict)
        and contact.get("observed") is True
        and contact.get("source") == "simulator_particle_to_gripper_distance"
        and isinstance(contact.get("minimum_distance_m"), (int, float))
        and math.isfinite(float(contact["minimum_distance_m"]))
        and float(contact["minimum_distance_m"]) >= 0
    )
    reset_hash = episode.get("reset_hash")
    if not isinstance(reset_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", reset_hash):
        reset_hash = None
    return EpisodeGateEvidence(
        official_success=episode.get("outcome") == "success" and episode.get("accepted_success") is True,
        visible_contact=visible_contact,
        reset_hash=reset_hash,
    )


def _attempted_gate_evidence(output_root: Path, trial_ids: Sequence[str]) -> list[EpisodeGateEvidence]:
    """Count every terminal attempt; absent/corrupt artifacts fail closed as no evidence."""
    evidence: list[EpisodeGateEvidence] = []
    for trial_id in trial_ids:
        try:
            complete = is_completed_trial(output_root, trial_id)
        except ValueError:
            complete = False
        if complete:
            evidence.append(_episode_gate_evidence(output_root, trial_id))
        else:
            evidence.append(EpisodeGateEvidence(False, False, None))
    return evidence


def _abort_after_first_completed_cohort(
    args: argparse.Namespace,
    *,
    trial_ids: Sequence[str],
    invocation_id: str,
) -> dict[str, object] | None:
    evidence = _attempted_gate_evidence(args.output_root, trial_ids)
    receipt = evaluate_abort_gate(evidence, AbortGate(args.early_abort_completed_trials))
    if receipt is None:
        return None
    result = dict(receipt)
    cohort_ids = list(trial_ids[:args.early_abort_completed_trials])
    result.update({
        "invocation_id": invocation_id,
        "trial_ids": cohort_ids,
        "missing_or_invalid_evidence_trial_ids": [
            trial_id for trial_id, item in zip(cohort_ids, evidence[:args.early_abort_completed_trials], strict=True)
            if item.reset_hash is None
        ],
    })
    path = args.output_root / f"campaign-abort-receipt-{invocation_id}.json"
    _write_json_atomically(path, result)
    result["receipt_path"] = str(path)
    return result


def _terminal_attempted_trial_ids(records: Sequence[dict[str, object]]) -> list[str]:
    attempted: list[str] = []
    for record in records:
        if "trial_id" in record and isinstance(record["trial_id"], str):
            attempted.append(record["trial_id"])
        else:
            attempted.extend(
                trial_id for trial_id in record.get("launched_trial_ids", ())
                if isinstance(trial_id, str)
            )
    return attempted


def _scale_cpu_canary_ledgers(output_root: Path, canary_ids: Sequence[str]) -> dict[str, dict[str, object]]:
    """Return one attributable terminal launch per canary ID, or fail closed.

    A non-terminal ledger wave is deliberately not resumable evidence: it may
    have launched an Isaac process without producing an artifact.  Retrying it
    could turn the first twelve into an unaccounted multi-attempt experiment.
    """
    root = Path(output_root)
    ledger_root = root / "campaign-ledger"
    if ledger_root.exists() and (ledger_root.is_symlink() or not ledger_root.is_dir()):
        raise ValueError("scale_cpu canary ledger root is unsafe")
    if not ledger_root.exists():
        return {}
    permitted = set(canary_ids)
    attempts: dict[str, dict[str, object]] = {}
    for ledger_path in sorted(ledger_root.iterdir()):
        if ledger_path.is_symlink() or not ledger_path.is_file():
            raise ValueError("scale_cpu canary ledger contains an unsafe entry")
        ledger = _regular_json(ledger_path, label="scale_cpu canary ledger")
        if ledger.get("schema_version") != 1 or ledger.get("mode") != "production":
            raise ValueError("scale_cpu canary ledger has an invalid schema or mode")
        waves = ledger.get("waves")
        if not isinstance(waves, list):
            raise ValueError("scale_cpu canary ledger lacks wave accounting")
        for wave in waves:
            if not isinstance(wave, dict):
                raise ValueError("scale_cpu canary ledger wave is invalid")
            scheduled = wave.get("scheduled_trial_ids")
            launched = wave.get("launched_trial_ids")
            trial_ids = wave.get("trial_ids")
            touched = set()
            for values in (scheduled, launched, trial_ids):
                if isinstance(values, list):
                    touched.update(value for value in values if isinstance(value, str) and value in permitted)
            if not touched:
                continue
            if (
                wave.get("status") != "terminal"
                or not isinstance(trial_ids, list)
                or not isinstance(scheduled, list)
                or not isinstance(launched, list)
                or scheduled != trial_ids
                or launched != trial_ids
                or len(set(trial_ids)) != len(trial_ids)
                or not isinstance(wave.get("completed_trials"), int)
                or not isinstance(wave.get("failed_trials"), int)
                or wave["completed_trials"] + wave["failed_trials"] != len(trial_ids)
            ):
                raise ValueError("scale_cpu canary ledger has an interrupted or ambiguous canary wave")
            for trial_id in trial_ids:
                if trial_id not in permitted:
                    continue
                if trial_id in attempts:
                    raise ValueError("scale_cpu canary ledger records more than one terminal attempt for a canary ID")
                attempts[trial_id] = {
                    "ledger_path": str(ledger_path),
                    "ledger_sha256": _sha256_file(ledger_path),
                    "wave": wave.get("wave"),
                }
    return attempts


def _scale_cpu_canary_state(
    output_root: Path,
    canary_ids: Sequence[str],
) -> tuple[dict[str, EpisodeGateEvidence], dict[str, str], tuple[str, ...], dict[str, dict[str, object]]]:
    """Reconstruct the canary from validated artifacts plus terminal ledgers."""
    ledgers = _scale_cpu_canary_ledgers(output_root, canary_ids)
    root = Path(output_root)
    raw = root / "raw"
    evidence: dict[str, EpisodeGateEvidence] = {}
    source: dict[str, str] = {}
    missing: list[str] = []
    for trial_id in canary_ids:
        try:
            complete = is_completed_trial(root, trial_id)
        except ValueError:
            complete = False
        if complete:
            evidence[trial_id] = _episode_gate_evidence(root, trial_id)
            source[trial_id] = "validated_terminal_artifact"
        elif trial_id in ledgers:
            # A terminal launch that lacks a trustworthy success artifact is a
            # durable false/no-contact attempt, never a retry candidate.
            evidence[trial_id] = EpisodeGateEvidence(False, False, None)
            source[trial_id] = "terminal_ledger_without_valid_artifact"
        else:
            candidate = raw / trial_id
            if candidate.exists() or candidate.is_symlink():
                raise ValueError("scale_cpu canary has an invalid artifact without terminal ledger attribution")
            missing.append(trial_id)
    return evidence, source, tuple(missing), ledgers


def _scale_cpu_canary_receipt(
    *,
    args: argparse.Namespace,
    authorization: dict[str, object] | None,
    canary_ids: Sequence[str],
    evidence: dict[str, EpisodeGateEvidence],
    source: dict[str, str],
    ledgers: dict[str, dict[str, object]],
    decision: str,
) -> dict[str, object]:
    invocation = authorization.get("invocation") if isinstance(authorization, dict) else None
    ledger_bindings = {
        trial_id: {
            "path": ledgers[trial_id]["ledger_path"],
            "sha256": ledgers[trial_id]["ledger_sha256"],
        }
        for trial_id in canary_ids
        if trial_id in ledgers
    }
    attempts = [
        {
            "trial_id": trial_id,
            "source": source[trial_id],
            "official_success": evidence[trial_id].official_success,
            "visible_contact": evidence[trial_id].visible_contact,
            "reset_hash": evidence[trial_id].reset_hash,
            "policy_receipt_sha256": (
                _sha256_file(args.output_root / f"policy-server-receipt-{trial_id}.json")
                if (args.output_root / f"policy-server-receipt-{trial_id}.json").is_file()
                and not (args.output_root / f"policy-server-receipt-{trial_id}.json").is_symlink()
                else None
            ),
        }
        for trial_id in canary_ids
    ]
    return {
        "schema_version": 1,
        "kind": "scale_cpu_first_twelve_canary",
        "decision": decision,
        "canary_trial_ids": list(canary_ids),
        "official_successes": sum(item.official_success for item in evidence.values()),
        "visible_robot_garment_contacts": sum(item.visible_contact for item in evidence.values()),
        "attempt_evidence": attempts,
        "authorization_sha256": _canonical_sha256(authorization) if authorization is not None else None,
        "invocation_sha256": _canonical_sha256(invocation) if invocation is not None else None,
        "canary_ledger_bindings": ledger_bindings,
    }


def _verify_or_write_scale_cpu_canary_receipt(
    *,
    args: argparse.Namespace,
    authorization: dict[str, object] | None,
    canary_ids: Sequence[str],
    evidence: dict[str, EpisodeGateEvidence],
    source: dict[str, str],
    ledgers: dict[str, dict[str, object]],
) -> bool:
    if set(evidence) != set(canary_ids):
        raise ValueError("scale_cpu canary decision requires exactly twelve durable terminal attempts")
    decision = "abort" if not any(item.official_success for item in evidence.values()) or not any(item.visible_contact for item in evidence.values()) else "pass"
    expected = _scale_cpu_canary_receipt(
        args=args, authorization=authorization, canary_ids=canary_ids,
        evidence=evidence, source=source, ledgers=ledgers, decision=decision,
    )
    path = args.output_root / _CPU_SCALE_CANARY_RECEIPT
    if path.exists() or path.is_symlink():
        existing = _regular_json(path, label="scale_cpu canary receipt")
        immutable = (
            "schema_version", "kind", "canary_trial_ids", "authorization_sha256", "invocation_sha256",
        )
        if any(existing.get(key) != expected.get(key) for key in immutable):
            raise ValueError("scale_cpu canary receipt is stale or does not match terminal evidence and production identity")
        bindings = existing.get("canary_ledger_bindings")
        attempts = existing.get("attempt_evidence")
        if not isinstance(bindings, dict) or not isinstance(attempts, list) or len(attempts) != len(canary_ids):
            raise ValueError("scale_cpu canary receipt is stale or does not match terminal evidence and production identity")
        for trial_id, binding in bindings.items():
            current = ledgers.get(trial_id)
            if (
                not isinstance(trial_id, str) or not isinstance(binding, dict)
                or not isinstance(current, dict)
                or binding.get("path") != current.get("ledger_path")
                or binding.get("sha256") != current.get("ledger_sha256")
            ):
                raise ValueError("scale_cpu canary receipt is stale or does not match terminal evidence and production identity")
        for trial_id, item in zip(canary_ids, attempts, strict=True):
            if not isinstance(item, dict) or item.get("trial_id") != trial_id:
                raise ValueError("scale_cpu canary receipt is stale or does not match terminal evidence and production identity")
            if item.get("source") == "terminal_ledger_without_valid_artifact":
                # The first failed terminal launch is frozen evidence.  A later
                # retry may create a valid artifact, but must not rewrite the
                # canary result that authorized the post-gate remainder.
                if trial_id not in bindings or any(item.get(key) != value for key, value in {
                    "official_success": False, "visible_contact": False, "reset_hash": None,
                    "policy_receipt_sha256": None,
                }.items()):
                    raise ValueError("scale_cpu canary receipt is stale or does not match terminal evidence and production identity")
                continue
            current = {
                "trial_id": trial_id,
                "source": source.get(trial_id),
                "official_success": evidence[trial_id].official_success,
                "visible_contact": evidence[trial_id].visible_contact,
                "reset_hash": evidence[trial_id].reset_hash,
                "policy_receipt_sha256": (
                    _sha256_file(args.output_root / f"policy-server-receipt-{trial_id}.json")
                    if (args.output_root / f"policy-server-receipt-{trial_id}.json").is_file()
                    and not (args.output_root / f"policy-server-receipt-{trial_id}.json").is_symlink()
                    else None
                ),
            }
            if item != current:
                raise ValueError("scale_cpu canary receipt is stale or does not match terminal evidence and production identity")
        if existing.get("decision") not in {"pass", "abort"}:
            raise ValueError("scale_cpu canary receipt is stale or does not match terminal evidence and production identity")
        return existing["decision"] == "pass"
    else:
        _write_json_atomically(path, expected)
    return decision == "pass"


def _run_scale_cpu_canary(
    args: argparse.Namespace,
    *,
    state: CampaignState,
    by_id: dict[str, Trial],
    authorization: dict[str, object] | None,
) -> None:
    """Finish and durably decide the public matrix's first twelve before ID 13."""
    canary_ids = state.trial_ids[:12]
    if len(canary_ids) != 12:
        raise ValueError("scale_cpu requires the canonical public first twelve canary IDs")
    receipt_path = args.output_root / _CPU_SCALE_CANARY_RECEIPT
    if receipt_path.exists() or receipt_path.is_symlink():
        existing = _regular_json(receipt_path, label="scale_cpu canary receipt")
        if existing.get("schema_version") != 1 or existing.get("kind") != "scale_cpu_first_twelve_canary" or existing.get("canary_trial_ids") != list(canary_ids):
            raise ValueError("scale_cpu canary receipt schema, kind, or canonical IDs are invalid")
        invocation = authorization.get("invocation") if isinstance(authorization, dict) else None
        if existing.get("authorization_sha256") != (_canonical_sha256(authorization) if authorization is not None else None) or existing.get("invocation_sha256") != (_canonical_sha256(invocation) if invocation is not None else None):
            raise ValueError("scale_cpu canary receipt is stale: production identity differs")
        attempts = existing.get("attempt_evidence")
        bindings = existing.get("canary_ledger_bindings")
        if not isinstance(attempts, list) or len(attempts) != 12 or not isinstance(bindings, dict):
            raise ValueError("scale_cpu canary receipt is incomplete")
        successes = contacts = 0
        for trial_id, item in zip(canary_ids, attempts, strict=True):
            if not isinstance(item, dict) or item.get("trial_id") != trial_id or item.get("source") not in {"validated_terminal_artifact", "terminal_ledger_without_valid_artifact"} or not isinstance(item.get("official_success"), bool) or not isinstance(item.get("visible_contact"), bool):
                raise ValueError("scale_cpu canary receipt attempt evidence is invalid")
            if item["source"] == "validated_terminal_artifact":
                observed = _episode_gate_evidence(args.output_root, trial_id)
                receipt = _regular_json(args.output_root / f"policy-server-receipt-{trial_id}.json", label="scale_cpu frozen canary policy receipt")
                if item.get("official_success") != observed.official_success or item.get("visible_contact") != observed.visible_contact or item.get("reset_hash") != observed.reset_hash or item.get("policy_receipt_sha256") != _sha256_file(args.output_root / f"policy-server-receipt-{trial_id}.json"):
                    raise ValueError("scale_cpu canary frozen artifact evidence is stale")
            else:
                binding = bindings.get(trial_id)
                if item.get("official_success") is not False or item.get("visible_contact") is not False or item.get("reset_hash") is not None or item.get("policy_receipt_sha256") is not None or not isinstance(binding, dict):
                    raise ValueError("scale_cpu canary frozen ledger attempt is invalid")
            successes += int(item["official_success"]); contacts += int(item["visible_contact"])
        decision = "abort" if successes == 0 or contacts == 0 else "pass"
        if existing.get("official_successes") != successes or existing.get("visible_robot_garment_contacts") != contacts or existing.get("decision") != decision:
            raise ValueError("scale_cpu canary receipt aggregate or decision is tampered")
        if set(bindings) != set(canary_ids):
            raise ValueError("scale_cpu canary receipt has extra or missing ledger bindings")
        ledger_root = (args.output_root / "campaign-ledger").resolve()
        for trial_id, binding in bindings.items():
            if not isinstance(binding, dict):
                raise ValueError("scale_cpu canary receipt ledger binding is invalid")
            ledger_path = Path(str(binding.get("path", "")))
            if ledger_path.is_symlink() or not ledger_path.is_file() or not ledger_path.resolve().is_relative_to(ledger_root) or binding.get("sha256") != _sha256_file(ledger_path):
                raise ValueError("scale_cpu canary receipt ledger binding is stale")
            ledger = _regular_json(ledger_path, label="scale_cpu frozen canary ledger")
            if ledger.get("schema_version") != 1 or ledger.get("mode") != "production":
                raise ValueError("scale_cpu frozen canary ledger is invalid")
            waves = ledger.get("waves")
            matches = [wave for wave in waves if isinstance(wave, dict) and trial_id in wave.get("trial_ids", ())] if isinstance(waves, list) else []
            if len(matches) != 1:
                raise ValueError("scale_cpu frozen canary ledger does not uniquely attribute its trial")
            wave = matches[0]
            trial_ids = wave.get("trial_ids")
            if wave.get("status") != "terminal" or not isinstance(trial_ids, list) or wave.get("scheduled_trial_ids") != trial_ids or wave.get("launched_trial_ids") != trial_ids or not isinstance(wave.get("completed_trials"), int) or not isinstance(wave.get("failed_trials"), int) or wave["completed_trials"] + wave["failed_trials"] != len(trial_ids):
                raise ValueError("scale_cpu frozen canary ledger wave is invalid")
        if decision != "pass":
            raise RuntimeError("campaign aborted: scale_cpu first-twelve canary has zero success or zero contact")
        return
    evidence, source, missing, ledgers = _scale_cpu_canary_state(args.output_root, canary_ids)
    if missing:
        invocation_id = uuid4().hex
        checkpoint: dict[str, object] = {
            "schema_version": 1, "invocation_id": invocation_id, "mode": "production",
            "status": "running", "pending_before": list(missing), "waves": [],
        }
        _write_invocation_checkpoint(args.output_root, invocation_id, checkpoint)
        for wave_number, offset in enumerate(range(0, len(missing), 4), start=1):
            wave_trial_ids = missing[offset:offset + 4]
            assignments = tuple((worker_id, by_id[trial_id]) for worker_id, trial_id in enumerate(wave_trial_ids, start=1))
            gpu_indices = _worker_gpu_indices(args, len(assignments))
            wave = {
                "mode": "production", "wave": wave_number, "workers": len(assignments),
                "trial_ids": list(wave_trial_ids), "scheduled_trial_ids": list(wave_trial_ids),
                "launched_trial_ids": [], "gpu_indices": list(gpu_indices), "status": "started",
            }
            checkpoint["waves"].append(wave)
            _write_invocation_checkpoint(args.output_root, invocation_id, checkpoint)
            try:
                elapsed, completed, failed = _run_worker_group(args, assignments, gpu_indices=gpu_indices)
            except BaseException as error:
                scheduled, launched = _launch_accounting_from_error(error, wave_trial_ids)
                wave.update({"status": "interrupted", "detail": str(error), "scheduled_trial_ids": scheduled, "launched_trial_ids": launched})
                checkpoint.update({"status": "interrupted", "error_type": type(error).__name__})
                _write_invocation_checkpoint(args.output_root, invocation_id, checkpoint)
                raise
            wave.update({
                "status": "terminal", "launched_trial_ids": list(wave_trial_ids),
                "elapsed_seconds": elapsed, "completed_trials": completed, "failed_trials": failed,
            })
            _write_invocation_checkpoint(args.output_root, invocation_id, checkpoint)
        checkpoint["status"] = "canary_terminal"
        checkpoint["completed_after"] = [trial_id for trial_id in canary_ids if trial_id not in missing or is_completed_trial(args.output_root, trial_id)]
        _write_invocation_checkpoint(args.output_root, invocation_id, checkpoint)
        evidence, source, missing, ledgers = _scale_cpu_canary_state(args.output_root, canary_ids)
    if missing:
        raise ValueError("scale_cpu canary is incomplete after scheduling its missing IDs")
    if not _verify_or_write_scale_cpu_canary_receipt(
        args=args, authorization=authorization, canary_ids=canary_ids,
        evidence=evidence, source=source, ledgers=ledgers,
    ):
        raise RuntimeError("campaign aborted: scale_cpu first-twelve canary has zero success or zero contact")


def _read_parity_receipt(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("parity receipt must be a regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("parity receipt must be valid JSON") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("parity receipt schema is invalid")
    stage, count, successes = payload.get("parity_stage"), payload.get("trial_count"), payload.get("official_successes")
    if stage not in {"legacy_server_cpu", "server_cpu", "server_cuda"} or count != 12 or not isinstance(successes, int) or not 0 <= successes <= 12:
        raise ValueError("parity receipt does not prove a complete twelve-case stage")
    expected_backend = {
        "legacy_server_cpu": "legacy_shared_policy_server",
        "server_cpu": "policy_server_cpu",
        "server_cuda": "policy_server_cuda",
    }
    if payload.get("backend") != expected_backend[stage]:
        raise ValueError("parity receipt stage label does not match its recorded backend")
    if stage != "legacy_server_cpu":
        root_value = payload.get("artifact_root")
        trial_ids = payload.get("trial_ids")
        if not isinstance(root_value, str) or not isinstance(trial_ids, list) or tuple(trial_ids) != HISTORICAL_CONTROL_IDS:
            raise ValueError("new server parity receipt requires twelve attributable raw artifacts")
        root = Path(root_value)
        if root.is_symlink() or not root.is_dir():
            raise ValueError("new server parity receipt artifact root must be a real directory")
        expected_device = "cpu" if stage == "server_cpu" else None
        policy_receipts = payload.get("policy_server_receipts")
        if not isinstance(policy_receipts, list) or len(policy_receipts) != 12:
            raise ValueError("new server parity receipt requires twelve policy-server receipt bindings")
        receipt_by_trial = {item.get("trial_id"): item for item in policy_receipts if isinstance(item, dict)}
        if set(receipt_by_trial) != set(trial_ids):
            raise ValueError("new server parity receipt policy-server receipt IDs are invalid")
        derived_successes = 0
        derived_contacts = 0
        reset_hashes: list[str] = []
        identity_values: dict[str, set[object]] = {
            "policy_repo": set(), "policy_step": set(), "policy_revision": set(),
            "code_revision": set(), "asset_revision": set(), "simulator_version": set(),
            "strategy": set(), "policy_artifact_sha256": set(), "image_identity": set(),
            "groot_revision": set(), "python_version": set(), "python_path": set(), "policy_server_backend": set(),
        }
        bundle_files: dict[str, str] = {
            "parity_receipt": _sha256_file(path),
            "historical_control_ids": _historical_control_identity(),
        }
        for trial_id in trial_ids:
            if not isinstance(trial_id, str):
                raise ValueError("new server parity receipt has invalid trial ID")
            episode, manifest = verify_episode_manifest(root / "raw" / trial_id)
            manifest_path = root / "raw" / trial_id / "SHA256SUMS.json"
            bundle_files[f"raw/{trial_id}/manifest"] = _sha256_file(manifest_path)
            for relative, entry in manifest.items():
                if isinstance(entry, dict) and isinstance(entry.get("sha256"), str):
                    bundle_files[f"raw/{trial_id}/{relative}"] = entry["sha256"]
            if not isinstance(episode.get("terminal_reason"), str) or not isinstance(episode.get("outcome"), str) or episode.get("dry_run") is True:
                raise ValueError("new server parity receipt artifact is not a real terminal rollout")
            provenance = episode.get("provenance")
            if not isinstance(provenance, dict) or provenance.get("execution_mode") != "policy_server" or provenance.get("execution_backend") != "policy_server" or provenance.get("parity_stage") != stage:
                raise ValueError("new server parity receipt artifact backend or stage is foreign")
            device = provenance.get("simulator_device")
            policy_device = provenance.get("policy_device")
            if not isinstance(policy_device, str) or _cuda_device_index(policy_device) is None or policy_device == "cuda":
                raise ValueError("new server parity receipt artifact policy device does not match a physical CUDA device")
            if (expected_device is not None and device != expected_device) or (
                stage == "server_cuda" and (
                    not isinstance(device, str)
                    or re.fullmatch(r"cuda:[0-9]+", device) is None
                )
            ):
                raise ValueError("new server parity receipt artifact device does not match its stage")
            receipt_binding = receipt_by_trial[trial_id]
            receipt_path = _verified_receipt_file(root, receipt_binding.get("path"), receipt_binding.get("sha256"))
            bundle_files[f"policy_receipt/{trial_id}"] = _sha256_file(receipt_path)
            try:
                server_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError("new server parity receipt policy-server receipt is invalid") from error
            identity = episode.get("identity")
            required_receipt = {
                "schema_version": 1,
                "episode_id": trial_id,
                "parity_stage": stage,
                "backend": "policy_server",
                "simulator_device": device,
                "policy_device": policy_device,
                "checkpoint_revision": identity.get("policy_revision") if isinstance(identity, dict) else None,
                "checkpoint_digest": provenance.get("policy_artifact_sha256"),
                "code_revision": identity.get("code_revision") if isinstance(identity, dict) else None,
                "image_identity": provenance.get("image_identity"),
                "policy_seed": identity.get("seed") if isinstance(identity, dict) else None,
            }
            if any(server_receipt.get(key) != value for key, value in required_receipt.items()):
                raise ValueError("new server parity receipt policy-server device binding mismatches terminal provenance")
            command = server_receipt.get("command")
            expected_command = [
                server_receipt.get("python_path"),
                "run_groot_policy_server.py",
                "--model-path", server_receipt.get("model_path"),
                "--host", "127.0.0.1",
                "--port", str(server_receipt.get("port")),
                "--api-token-env", "LEHOME_GROOT_POLICY_API_TOKEN",
                "--device", "cuda:0",
                "--seed", str(server_receipt.get("policy_seed")),
            ]
            if (
                not isinstance(server_receipt.get("groot_revision"), str)
                or not re.fullmatch(r"[0-9a-f]{40}", server_receipt["groot_revision"])
                or not isinstance(server_receipt.get("python_path"), str)
                or not isinstance(server_receipt.get("python_version"), str)
                or not re.fullmatch(r"3\.10\.[0-9]+", server_receipt["python_version"])
                or server_receipt.get("host") != "127.0.0.1"
                or not isinstance(server_receipt.get("port"), int)
                or not 1 <= server_receipt["port"] <= 65535
                or not isinstance(server_receipt.get("request_timeout_seconds"), (int, float))
                or not isinstance(server_receipt.get("readiness_timeout_seconds"), (int, float))
                or not isinstance(command, list)
                or not isinstance(server_receipt.get("model_path"), str)
                or len(command) != len(expected_command)
                or command[0] != expected_command[0]
                or not isinstance(command[1], str) or Path(command[1]).name != expected_command[1]
                or command[2:] != expected_command[2:]
            ):
                raise ValueError("new server parity receipt policy-server schema is incomplete or unsafe")
            if episode.get("outcome") == "success" and episode.get("accepted_success") is True:
                derived_successes += 1
            evidence = _episode_gate_evidence(root, trial_id)
            derived_contacts += int(evidence.visible_contact)
            if evidence.reset_hash is not None:
                reset_hashes.append(evidence.reset_hash)
            if isinstance(identity, dict):
                for key in (
                    "policy_repo", "policy_step", "policy_revision", "code_revision",
                    "asset_revision", "simulator_version", "strategy",
                ):
                    value = identity.get(key)
                    if value is not None and (not isinstance(value, (str, int)) or isinstance(value, bool)):
                        raise ValueError("new server parity receipt identity contains an unsafe value")
                    identity_values[key].add(value)
            for key, value in {
                "policy_artifact_sha256": provenance.get("policy_artifact_sha256"),
                "image_identity": provenance.get("image_identity"),
                "groot_revision": server_receipt.get("groot_revision"),
                "python_version": server_receipt.get("python_version"),
                "python_path": server_receipt.get("python_path"),
                "policy_server_backend": provenance.get("execution_backend"),
            }.items():
                if value is not None and not isinstance(value, str):
                    raise ValueError("new server parity receipt provenance contains an unsafe value")
                identity_values[key].add(value)
        if successes != derived_successes:
            raise ValueError("new server parity receipt claimed successes do not match terminal artifacts")
        validated = dict(payload)
        validated["official_successes"] = derived_successes
        validated["_derived_evidence"] = {
            "visible_robot_garment_contacts": derived_contacts,
            "nonmissing_reset_hashes": len(reset_hashes),
            "unique_reset_hashes": len(set(reset_hashes)),
        }
        validated["_derived_identity"] = {
            key: next(iter(values)) if len(values) == 1 else None
            for key, values in identity_values.items()
        }
        validated["_historical_control_identity"] = hashlib.sha256(
            json.dumps(list(HISTORICAL_CONTROL_IDS), separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        validated["_bundle_sha256"] = _canonical_sha256(bundle_files)
        return validated
    return payload


def _require_scale_parity(receipts: Sequence[Path]) -> dict[str, object]:
    by_stage: dict[str, dict[str, object]] = {}
    for path in receipts:
        preliminary = _read_parity_receipt(path)
        payload = (
            _validate_legacy_shared_policy_receipt(path)
            if preliminary["parity_stage"] == "legacy_server_cpu"
            else preliminary
        )
        stage = str(payload["parity_stage"])
        if stage in by_stage:
            raise ValueError("scale requires exactly one receipt for each parity stage")
        by_stage[stage] = payload
    required = {"legacy_server_cpu", "server_cpu", "server_cuda"}
    if set(by_stage) != required:
        raise ValueError("scale requires legacy_server_cpu, server_cpu, and server_cuda parity receipts")
    decision = evaluate_parity_ladder(
        legacy_server_cpu_successes=int(by_stage["legacy_server_cpu"]["official_successes"]),
        server_cpu_successes=int(by_stage["server_cpu"]["official_successes"]),
        server_cuda_successes=int(by_stage["server_cuda"]["official_successes"]),
    )
    if not decision.allowed:
        raise ValueError(f"scale rejected by parity ladder: {', '.join(decision.reasons)}")
    return {"allowed": True, "receipts": [str(path) for path in receipts], "reasons": list(decision.reasons)}


def _may_emit_parity_receipt(stage: object, official_successes: int) -> bool:
    """A CUDA diagnostic never becomes a passing receipt below the ladder floor."""
    return stage != "server_cuda" or official_successes >= 10


_CPU_SCALE_DECISION = "authorize_cpu_simulator_policy_server_scale_v1"
_CPU_SCALE_AUTHORIZATION = "cpu-scale-authorization.json"
_CPU_SCALE_CANARY_RECEIPT = "cpu-scale-canary-receipt.json"


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _regular_json(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _historical_control_identity() -> str:
    return _canonical_sha256(list(HISTORICAL_CONTROL_IDS))


def _require_cpu_scale_identity(
    identity: object,
    expected: dict[str, object],
    *,
    label: str,
) -> None:
    if not isinstance(identity, dict):
        raise ValueError(f"{label} must bind complete immutable identity")
    required = (
        "policy_repo", "policy_step", "policy_revision", "code_revision",
        "asset_revision", "simulator_version", "policy_artifact_sha256",
        "image_identity", "strategy",
    )
    if any(identity.get(key) != expected[key] for key in required):
        raise ValueError(f"{label} identity does not match the live CPU scale invocation")


def _read_current_policy_revision(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("CPU scale requires a regular pinned policy revision file")
    revision = path.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("CPU scale policy revision must be a pinned 40-character SHA")
    return revision


def _clean_git_revision(root: Path, *, label: str) -> tuple[Path, str]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{label} must be a real Git checkout")
    try:
        revision = subprocess.run(("git", "-C", str(root), "rev-parse", "HEAD"), check=False, capture_output=True, text=True)
        status = subprocess.run(("git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"), check=False, capture_output=True, text=True)
    except OSError as error:
        raise ValueError(f"{label} is not readable Git evidence") from error
    value = revision.stdout.strip()
    if revision.returncode or status.returncode or status.stdout or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError(f"{label} must be a clean pinned Git checkout")
    return root.resolve(), value


def _live_groot_identity(args: argparse.Namespace) -> dict[str, object]:
    root, revision = _clean_git_revision(Path(args.groot_root), label="GR00T root")
    if revision != args.groot_revision:
        raise ValueError("GR00T root revision does not match --groot-revision")
    interpreter = Path(args.groot_python)
    if interpreter.is_symlink() or not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise ValueError("GR00T Python must be a regular executable")
    try:
        version = subprocess.run((str(interpreter), "--version"), check=False, capture_output=True, text=True, timeout=5.0)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError("GR00T Python version is unavailable") from error
    text = (version.stdout or version.stderr).strip()
    match = re.fullmatch(r"Python (3\.10\.[0-9]+)", text)
    if version.returncode or match is None:
        raise ValueError("GR00T Python must report an exact Python 3.10 version")
    return {"groot_root": str(root), "groot_revision": revision, "groot_python": str(interpreter.resolve()), "groot_python_sha256": _sha256_file(interpreter), "groot_python_version": match.group(1)}


def _scale_cpu_runtime_paths(args: argparse.Namespace) -> tuple[Path, tuple[Path, ...]]:
    """Resolve the only import roots trusted by immutable scale_cpu children."""
    runtime_root = Path(args.trial_runtime_root)
    if runtime_root.is_symlink() or not runtime_root.is_dir():
        raise ValueError("scale_cpu trial runtime root is unsafe")
    isaacsim = os.environ.get("ISAACSIM_PATH")
    if not isaacsim:
        raise ValueError("scale_cpu requires ISAACSIM_PATH for the immutable trial runtime")
    kit_python = Path(isaacsim) / "kit" / "python"
    legacy_python = Path(isaacsim) / "python"
    groot_root_value = getattr(args, "groot_root", None)
    if groot_root_value is None:
        raise ValueError("scale_cpu requires a declared GR00T checkout")
    groot_root = Path(groot_root_value)
    # Isaac Sim 5.1's production Kit layout exposes Python under kit/python;
    # keep the older direct-python location only for images that materialize it.
    isaac_python = kit_python if kit_python.is_dir() and not kit_python.is_symlink() else legacy_python
    roots = (
        runtime_root,
        runtime_root / "source" / "lehome",
        runtime_root / "third_party" / "IsaacLab" / "source" / "isaaclab",
        runtime_root / "third_party" / "IsaacLab" / "source" / "isaaclab_assets",
        runtime_root / "third_party" / "IsaacLab" / "source" / "isaaclab_tasks",
        runtime_root / "third_party" / "IsaacLab" / "source" / "isaaclab_mimic",
        runtime_root / "third_party" / "IsaacLab" / "source" / "isaaclab_rl",
        isaac_python,
        groot_root,
    )
    if any(path.is_symlink() or not path.is_dir() for path in roots):
        raise ValueError("scale_cpu trusted runtime import roots are incomplete")
    return runtime_root.resolve(), tuple(path.resolve() for path in roots)


def _scale_cpu_runtime_bindings(args: argparse.Namespace) -> tuple[Path, dict[str, Path]]:
    """Name the validated child import roots used in authorization evidence."""
    runtime_root, trusted_paths = _scale_cpu_runtime_paths(args)
    names = (
        "trial_runtime_root",
        "lehome_python_path",
        "isaaclab_python_path",
        "isaaclab_assets_python_path",
        "isaaclab_tasks_python_path",
        "isaaclab_mimic_python_path",
        "isaaclab_rl_python_path",
        "isaacsim_python_path",
        "trusted_groot_root",
    )
    if len(trusted_paths) != len(names):
        raise ValueError("scale_cpu trusted runtime import root bindings are incomplete")
    return runtime_root, dict(zip(names, trusted_paths, strict=True))


def _controller_identity() -> dict[str, object]:
    root, revision = _clean_git_revision(Path(__file__).resolve().parents[1], label="controller checkout")
    campaign = root / "scripts" / "run_groot_flywheel_campaign.py"
    parity = root / "source" / "lehome" / "lehome" / "flywheel" / "parity.py"
    if campaign.is_symlink() or parity.is_symlink() or not campaign.is_file() or not parity.is_file():
        raise ValueError("controller identity files are unavailable")
    return {"controller_root": str(root), "controller_revision": revision, "controller_campaign_sha256": _sha256_file(campaign), "controller_parity_sha256": _sha256_file(parity)}


def _cpu_scale_live_invocation(args: argparse.Namespace, matrix: object) -> dict[str, object]:
    if len(getattr(matrix, "trials", ())) != 280:
        raise ValueError("CPU scale requires the canonical public 280-trial matrix")
    revision = _read_current_policy_revision(args.policy_revision_file)
    runtime_root = runtime_revision = None
    if getattr(args, "trial_runtime_root", None) is not None:
        runtime_root, runtime_revision = _clean_git_revision(args.trial_runtime_root, label="trial runtime root")
        if runtime_revision != args.code_revision:
            raise ValueError("trial runtime root revision does not match --code-revision")
    expected = {
        "policy_repo": args.policy_repo,
        "policy_step": args.policy_step,
        "policy_revision": revision,
        "code_revision": args.code_revision,
        "asset_revision": args.asset_revision,
        "simulator_version": args.simulator_version,
        "policy_artifact_sha256": args.policy_artifact_sha256,
        "image_identity": args.image_identity,
        "strategy": args.strategy,
    }
    if any(value is None or value == "" for value in expected.values()):
        raise ValueError("CPU scale requires complete immutable invocation identities")
    return {
        **expected,
        "matrix_sha256": matrix_sha256(matrix),
        "matrix_trial_count": len(matrix.trials),
        "historical_control_identity": _historical_control_identity(),
        "device": args.device,
        "execution_mode": args.execution_mode,
        "workers": args.workers,
        "policy_device": args.policy_device,
        "groot_revision": args.groot_revision,
        "groot_python": str(args.groot_python),
        "trial_runtime_root": str(runtime_root) if runtime_root is not None else None,
        "trial_runtime_revision": runtime_revision,
        # Unit-level evidence checks intentionally use a small Namespace rather
        # than the full production CLI.  Production authorization below requires
        # every one of these values when a pinned runtime root is supplied.
        "output_root": str(Path(args.output_root).resolve()) if getattr(args, "output_root", None) else None,
        "matrix_path": str(Path(args.matrix).resolve()) if getattr(args, "matrix", None) else None,
        "policy_path": str(Path(args.policy_path).resolve()) if getattr(args, "policy_path", None) else None,
        "policy_revision_file": str(Path(args.policy_revision_file).resolve()),
        "policy_revision_file_sha256": _sha256_file(args.policy_revision_file),
        "release_assets_root": str(Path(args.release_assets_root).resolve()) if getattr(args, "release_assets_root", None) else None,
        "max_steps": getattr(args, "max_steps", 600), "trials_per_worker": getattr(args, "trials_per_worker", 1),
        "early_abort_completed_trials": getattr(args, "early_abort_completed_trials", 12),
        "minimum_reset_uniqueness_ratio": getattr(args, "minimum_reset_uniqueness_ratio", None),
        "worker_timeout_seconds": getattr(args, "worker_timeout_seconds", None),
        "terminate_grace_seconds": getattr(args, "terminate_grace_seconds", None),
        "policy_server_readiness_timeout": getattr(args, "policy_server_readiness_timeout", None),
        "policy_server_request_timeout": getattr(args, "policy_server_request_timeout", None),
        "policy_server_termination_grace": getattr(args, "policy_server_termination_grace", None),
        "max_inference_latency_seconds": getattr(args, "max_inference_latency_seconds", None),
        "max_inference_queue_depth": getattr(args, "max_inference_queue_depth", None),
    }


def _validate_server_cpu_evidence(path: Path, expected: dict[str, object]) -> dict[str, object]:
    receipt = _read_parity_receipt(path)
    if receipt.get("parity_stage") != "server_cpu" or receipt.get("backend") != "policy_server_cpu":
        raise ValueError("CPU scale requires a server_cpu policy-server receipt")
    evidence = receipt.get("_derived_evidence")
    if not isinstance(evidence, dict):
        raise ValueError("server_cpu receipt lacks re-derived terminal evidence")
    decision = evaluate_cpu_scale_ladder(
        legacy_server_cpu_successes=9,
        server_cpu_successes=int(receipt["official_successes"]),
        server_cpu_visible_contacts=int(evidence.get("visible_robot_garment_contacts", -1)),
        server_cpu_unique_resets=int(evidence.get("unique_reset_hashes", -1)),
        cuda_abort_successes=0,
        cuda_abort_terminal_trials=12,
    )
    if any(reason not in {"legacy_server_cpu_below_9_of_12"} for reason in decision.reasons):
        raise ValueError(f"server_cpu receipt rejected by CPU scale evidence: {', '.join(decision.reasons)}")
    if evidence.get("nonmissing_reset_hashes") != 12:
        raise ValueError("server_cpu receipt requires 12 nonmissing reset hashes")
    identity = receipt.get("_derived_identity")
    _require_cpu_scale_identity(identity, expected, label="server_cpu receipt")
    if not isinstance(identity, dict) or identity.get("policy_server_backend") != "policy_server":
        raise ValueError("server_cpu receipt must bind the policy-server backend")
    if identity.get("groot_revision") != expected["groot_revision"]:
        raise ValueError("server_cpu receipt GR00T revision does not match the live CPU scale invocation")
    python_version = identity.get("python_version")
    if not isinstance(python_version, str) or re.fullmatch(r"3\.10\.[0-9]+", python_version) is None:
        raise ValueError("server_cpu receipt must bind the pinned GR00T Python 3.10 runtime")
    if expected.get("groot_python") is not None and identity.get("python_path") != expected["groot_python"]:
        raise ValueError("server_cpu receipt Python path does not match the live GR00T runtime")
    if expected.get("groot_python_version") is not None and python_version != expected["groot_python_version"]:
        raise ValueError("server_cpu receipt Python version does not match the live GR00T runtime")
    if receipt.get("_historical_control_identity") != expected["historical_control_identity"]:
        raise ValueError("server_cpu receipt historical control identity does not match")
    return receipt


def _validate_cuda_abort_evidence(path: Path, expected: dict[str, object]) -> dict[str, object]:
    receipt = _regular_json(path, label="CUDA abort receipt")
    if "receipt_path" in receipt:
        raise ValueError("CUDA abort on-disk receipt must not contain a receipt path binding")
    required = {
        "status": "aborted", "reason": "zero_official_successes", "completed_trials": 12,
        "official_successes": 0,
    }
    if any(receipt.get(key) != value for key, value in required.items()) or tuple(receipt.get("trial_ids", ())) != HISTORICAL_CONTROL_IDS:
        raise ValueError("CUDA abort receipt is not the typed 0/12 historical diagnostic abort")
    root = path.parent
    invocation_id = receipt.get("invocation_id")
    if not isinstance(invocation_id, str) or re.fullmatch(r"[0-9a-f]{32}", invocation_id) is None:
        raise ValueError("CUDA abort receipt invocation ID is invalid")
    ledger = _regular_json(root / "campaign-ledger" / f"{invocation_id}.json", label="CUDA abort campaign ledger")
    if (
        ledger.get("schema_version") != 1 or ledger.get("invocation_id") != invocation_id
        or ledger.get("status") != "failed" or ledger.get("mode") != "production"
        or tuple(ledger.get("pending_before", ())) != HISTORICAL_CONTROL_IDS
        or tuple(ledger.get("completed_after", ())) != HISTORICAL_CONTROL_IDS
    ):
        raise ValueError("CUDA abort campaign ledger invocation or status is invalid")
    receipt_normalized = receipt
    actual_receipt_path = str(path)

    def normalized_bound_receipt(candidate: object, *, label: str) -> dict[str, object]:
        if not isinstance(candidate, dict) or candidate.get("receipt_path") != actual_receipt_path:
            raise ValueError(f"CUDA abort {label} receipt path does not bind the on-disk receipt")
        return {key: value for key, value in candidate.items() if key != "receipt_path"}

    if normalized_bound_receipt(ledger.get("abort_receipt"), label="campaign ledger") != receipt_normalized:
        raise ValueError("CUDA abort campaign ledger payload does not match its receipt")
    waves = ledger.get("waves")
    if not isinstance(waves, list) or len(waves) != 3:
        raise ValueError("CUDA abort campaign ledger must contain exactly three waves")
    wave_ids: list[str] = []
    for wave in waves:
        if not isinstance(wave, dict):
            raise ValueError("CUDA abort campaign ledger wave is invalid")
        trial_ids = wave.get("trial_ids")
        if (
            wave.get("mode") != "production" or wave.get("status") != "terminal"
            or not isinstance(trial_ids, list) or len(trial_ids) != 4
            or wave.get("scheduled_trial_ids") != trial_ids or wave.get("launched_trial_ids") != trial_ids
            or not isinstance(wave.get("completed_trials"), int) or not isinstance(wave.get("failed_trials"), int)
            or wave["completed_trials"] + wave["failed_trials"] != 4
        ):
            raise ValueError("CUDA abort campaign ledger wave accounting is invalid")
        if len(wave_ids) == 8:
            if normalized_bound_receipt(wave.get("abort_receipt"), label="campaign ledger terminal wave") != receipt_normalized:
                raise ValueError("CUDA abort campaign ledger terminal wave abort payload does not match its receipt")
        wave_ids.extend(trial_ids)
    if tuple(wave_ids) != HISTORICAL_CONTROL_IDS:
        raise ValueError("CUDA abort campaign ledger waves do not cover the canonical historical IDs in order")
    raw = root / "raw"
    if root.is_symlink() or raw.is_symlink() or not raw.is_dir():
        raise ValueError("CUDA abort receipt parent artifact root is unsafe")
    terminal_ids = tuple(sorted(child.name for child in raw.iterdir() if child.is_dir() and not child.is_symlink()))
    if set(terminal_ids) != set(HISTORICAL_CONTROL_IDS) or len(terminal_ids) != 12:
        raise ValueError("CUDA abort receipt parent artifact root must contain exactly twelve terminal historical IDs")
    if any(child.is_symlink() or not child.is_dir() for child in raw.iterdir()):
        raise ValueError("CUDA abort receipt parent artifact root contains unsafe entries")
    if any(root.glob("parity-receipt-server_cuda*.json")):
        raise ValueError("CUDA diagnostic abort root must not contain a CUDA success receipt")
    derived_successes = 0
    derived_contacts = 0
    identity_values: dict[str, set[object]] = {
        "policy_repo": set(), "policy_step": set(), "policy_revision": set(),
        "code_revision": set(), "asset_revision": set(), "simulator_version": set(),
        "strategy": set(), "policy_artifact_sha256": set(), "image_identity": set(),
        "groot_revision": set(), "python_version": set(), "python_path": set(),
    }
    bundle_files: dict[str, str] = {
        "abort_receipt": _sha256_file(path),
        "ledger": _sha256_file(root / "campaign-ledger" / f"{invocation_id}.json"),
    }
    for trial_id in HISTORICAL_CONTROL_IDS:
        episode, manifest = verify_episode_manifest(raw / trial_id)
        manifest_path = raw / trial_id / "SHA256SUMS.json"
        bundle_files[f"{trial_id}/manifest"] = _sha256_file(manifest_path)
        for relative, entry in manifest.items():
            if isinstance(entry, dict) and isinstance(entry.get("sha256"), str):
                bundle_files[f"{trial_id}/{relative}"] = entry["sha256"]
        provenance = episode.get("provenance")
        identity = episode.get("identity")
        if (
            not isinstance(provenance, dict)
            or provenance.get("execution_mode") != "policy_server"
            or provenance.get("execution_backend") != "policy_server"
            or provenance.get("parity_stage") != "server_cuda"
            or not isinstance(provenance.get("simulator_device"), str)
            or re.fullmatch(r"cuda:[0-9]+", provenance["simulator_device"]) is None
            or not isinstance(provenance.get("policy_device"), str)
            or _cuda_device_index(provenance["policy_device"]) is None
        ):
            raise ValueError("CUDA abort terminal provenance is not an isolated CUDA policy-server diagnostic")
        policy_path = root / f"policy-server-receipt-{trial_id}.json"
        policy_receipt = _regular_json(policy_path, label="CUDA abort policy-server receipt")
        bundle_files[f"policy-receipt/{trial_id}"] = _sha256_file(policy_path)
        if any(policy_receipt.get(key) != value for key, value in {
            "schema_version": 1, "episode_id": trial_id, "parity_stage": "server_cuda",
            "backend": "policy_server", "simulator_device": provenance["simulator_device"],
            "policy_device": provenance["policy_device"],
        }.items()):
            raise ValueError("CUDA abort policy-server receipt does not match terminal provenance")
        if episode.get("outcome") == "success" and episode.get("accepted_success") is True:
            derived_successes += 1
        derived_contacts += int(_episode_gate_evidence(root, trial_id).visible_contact)
        if isinstance(identity, dict):
            for key in ("policy_repo", "policy_step", "policy_revision", "code_revision", "asset_revision", "simulator_version", "strategy"):
                value = identity.get(key)
                if value is not None and (not isinstance(value, (str, int)) or isinstance(value, bool)):
                    raise ValueError("CUDA abort terminal identity contains an unsafe value")
                identity_values[key].add(value)
        for key, value in {
            "policy_artifact_sha256": provenance.get("policy_artifact_sha256"),
            "image_identity": provenance.get("image_identity"),
            "groot_revision": policy_receipt.get("groot_revision"),
            "python_version": policy_receipt.get("python_version"),
            "python_path": policy_receipt.get("python_path"),
        }.items():
            if value is not None and not isinstance(value, str):
                raise ValueError("CUDA abort terminal provenance contains an unsafe value")
            identity_values[key].add(value)
    if derived_successes != 0:
        raise ValueError("CUDA abort terminal artifacts contain an accepted success")
    if receipt.get("visible_robot_garment_contacts") != derived_contacts:
        raise ValueError("CUDA abort receipt visible-contact count does not match terminal artifacts")
    if derived_contacts != 6:
        raise ValueError("CUDA abort terminal artifacts must prove exactly 6 visible contacts")
    identity = {key: next(iter(values)) if len(values) == 1 else None for key, values in identity_values.items()}
    _require_cpu_scale_identity(identity, expected, label="CUDA abort receipt")
    if identity.get("groot_revision") != expected["groot_revision"]:
        raise ValueError("CUDA abort receipt GR00T revision does not match the live CPU scale invocation")
    python_version = identity.get("python_version")
    if not isinstance(python_version, str) or re.fullmatch(r"3\.10\.[0-9]+", python_version) is None:
        raise ValueError("CUDA abort receipt must bind the pinned GR00T Python 3.10 runtime")
    if expected.get("groot_python") is not None and identity.get("python_path") != expected["groot_python"]:
        raise ValueError("CUDA abort receipt Python path does not match the live GR00T runtime")
    if expected.get("groot_python_version") is not None and python_version != expected["groot_python_version"]:
        raise ValueError("CUDA abort receipt Python version does not match the live GR00T runtime")
    return {"official_successes": derived_successes, "visible_robot_garment_contacts": derived_contacts, "identity": identity, "bundle_sha256": _canonical_sha256(bundle_files)}


def _cpu_runtime_binding(invocation: dict[str, object]) -> dict[str, object]:
    descriptor = {
        "architecture": platform.machine(),
        "cpu_model": platform.processor(),
        "cpu_count": os.cpu_count(),
        "os": platform.system(),
        "kernel": platform.release(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
    }
    try:
        boot_identity = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    except OSError:
        try:
            boot = subprocess.run(
                ("sysctl", "-n", "kern.boottime"), check=False, capture_output=True, text=True, timeout=2.0,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ValueError("CPU scale authorization requires a boot/session identity") from error
        boot_identity = boot.stdout.strip() if boot.returncode == 0 else ""
    if not boot_identity or len(boot_identity) > 200:
        raise ValueError("CPU scale authorization boot/session identity is invalid")
    return {
        "fingerprint_sha256": _canonical_sha256({"descriptor": descriptor, "invocation": invocation}),
        "descriptor": descriptor,
        "boot_session_sha256": _canonical_sha256({"boot_id": boot_identity}),
    }


def _receipt_binding(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("CPU scale evidence receipt must be a regular file")
    return {"path": str(path), "sha256": _sha256_file(path)}


def _require_disjoint_real_roots(*roots: Path) -> None:
    resolved: list[Path] = []
    for root in roots:
        if root.is_symlink() or not root.is_dir():
            raise ValueError("scale_cpu evidence and production roots must be real directories")
        candidate = root.resolve()
        if any(candidate == prior or candidate.is_relative_to(prior) or prior.is_relative_to(candidate) for prior in resolved):
            raise ValueError("scale_cpu evidence roots and production output root must be pairwise disjoint")
        resolved.append(candidate)


def _verified_sha256sums(root: Path, *, label: str) -> tuple[Path, dict[str, str]]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{label} root must be a real directory")
    sums = root / "SHA256SUMS"
    if sums.is_symlink() or not sums.is_file():
        raise ValueError(f"{label} requires SHA256SUMS")
    listed: dict[str, str] = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        try:
            digest, relative = line.split(maxsplit=1)
        except ValueError as error:
            raise ValueError(f"{label} SHA256SUMS is malformed") from error
        relative = relative.removeprefix("./")
        if relative in listed or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"{label} SHA256SUMS has duplicate or invalid entries")
        listed[relative] = digest
        _verified_receipt_file(root, relative, digest)
    regular: set[str] = set()
    for directory, _dirs, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in files:
            candidate = parent / name
            details = candidate.lstat()
            if stat.S_ISLNK(details.st_mode):
                continue
            if not stat.S_ISREG(details.st_mode):
                raise ValueError(f"{label} contains an unsafe special file")
            relative = str(candidate.relative_to(root))
            if relative != "SHA256SUMS":
                regular.add(relative)
    if regular != set(listed):
        raise ValueError(f"{label} SHA256SUMS does not close the regular-file tree")
    return sums, listed


def _tree_symlink_map(root: Path, *, label: str) -> dict[str, str]:
    links: dict[str, str] = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in [*dirs, *files]:
            candidate = parent / name
            if candidate.is_symlink():
                links[str(candidate.relative_to(root))] = os.readlink(candidate)
    return links


def _legacy_source_and_archive(source_root: Path, archive_root: Path) -> dict[str, object]:
    source = Path(source_root)
    if source.is_symlink() or not source.is_dir():
        raise ValueError("legacy source root must be a real directory")
    source_hashes: dict[str, str] = {}
    for name in _LEGACY_SHARED_POLICY_HASHES:
        file = source / name
        if file.is_symlink() or not file.is_file():
            raise ValueError("legacy source root lacks a frozen source file")
        source_hashes[name] = _sha256_file(file)
    if source_hashes != _LEGACY_SHARED_POLICY_HASHES:
        raise ValueError("legacy source root files do not match the frozen shared-policy hashes")
    archive = Path(archive_root)
    sums, listed = _verified_sha256sums(archive, label="legacy archive")
    report = archive / "rollout-report.json"
    if report.is_symlink() or not report.is_file():
        raise ValueError("legacy archive requires rollout-report.json")
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
        ids = [item.get("trial", {}).get("trial_id") for item in payload.get("trials", ())]
    except (OSError, json.JSONDecodeError, AttributeError) as error:
        raise ValueError("legacy archive rollout report is invalid") from error
    if [trial_id for trial_id in ids if trial_id in HISTORICAL_CONTROL_IDS] != list(HISTORICAL_CONTROL_IDS):
        raise ValueError("legacy archive does not bind the exact historical twelve IDs")
    control_listed: dict[str, str] = {}
    for trial in historical_control_trials():
        relative = f"{trial.trial_id}/garment-config/Release/Release_test_list.txt"
        config = archive / relative
        if config.is_symlink() or not config.is_file() or listed.get(relative) != _sha256_file(config):
            raise ValueError("legacy archive frozen garment config is missing or unbound")
        if config.read_text(encoding="utf-8").strip() != trial.garment_name:
            raise ValueError("legacy archive frozen garment config does not match its trial")
        control_listed[relative] = listed[relative]
    return {
        "source_root": str(source.resolve()), "source_sha256": source_hashes,
        "archive_root": str(archive.resolve()), "archive_sha256sums_sha256": _sha256_file(sums),
        "archive_rollout_report_sha256": _sha256_file(report), "archive_bundle_sha256": _canonical_sha256(listed),
        "historical_control_root": str(archive.resolve()), "historical_control_sha256sums_sha256": _sha256_file(sums),
        "historical_control_bundle_sha256": _canonical_sha256(control_listed),
        "archive_symlinks": _tree_symlink_map(archive, label="legacy archive"),
    }


def _legacy_trial_command_declared(payload: dict[str, object], trial_id: str) -> dict[str, object] | None:
    trial = payload.get("trial")
    command = payload.get("command")
    if not isinstance(trial, dict) or trial.get("trial_id") != trial_id or trial.get("seed") != 42 or payload.get("environment_device") != "cpu" or not isinstance(command, list):
        return None
    def argument(flag: str) -> object | None:
        try:
            index = command.index(flag)
        except ValueError:
            return None
        return command[index + 1] if index + 1 < len(command) else None
    policy_path = argument("--policy_path")
    if argument("--device") != "cpu" or argument("--seed") != "42" or not isinstance(policy_path, str) or re.search(r"(^|[-_/])step-12000($|[-_/])", policy_path) is None:
        return None
    return {"environment_device": "cpu", "policy_path": policy_path, "seed": 42}


def build_legacy_cpu_reference_receipt(
    reproduction_root: Path,
    output_path: Path,
    *,
    source_root: Path,
    archive_root: Path,
) -> Path:
    """Create a checksummed 9..12/12 legacy CPU reference without rerunning Isaac.

    The source rollout remains untouched.  This only records the exact twelve
    historical ``trial.json`` artifacts and their checksums, so scale_cpu can
    consume a legacy root that pre-dates the newer terminal-record format.
    """
    root = Path(reproduction_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("legacy CPU reproduction root must be a real directory")
    sums, listed = _verified_sha256sums(root, label="legacy CPU reproduction")
    source_archive = _legacy_source_and_archive(source_root, archive_root)
    records: list[dict[str, object]] = []
    successes = 0
    for trial_id in HISTORICAL_CONTROL_IDS:
        path = root / trial_id / "trial.json"
        payload = _regular_json(path, label="legacy CPU trial artifact")
        metric = payload.get("metric")
        declared = _legacy_trial_command_declared(payload, trial_id)
        if not isinstance(metric, dict) or not isinstance(metric.get("success"), bool) or declared is None:
            raise ValueError("legacy CPU trial artifact must expose metric.success")
        successes += int(metric["success"])
        records.append({
            "trial_id": trial_id,
            "path": str(path.relative_to(root)),
            "sha256": _sha256_file(path),
        })
        if listed.get(str(path.relative_to(root))) != records[-1]["sha256"]:
            raise ValueError("legacy CPU reproduction SHA256SUMS must bind every trial.json record")
    receipt = {
        "schema_version": 1,
        "parity_stage": "legacy_server_cpu",
        "backend": "legacy_shared_policy_server",
        "reference_kind": "checksummed_trial_json_reproduction",
        "trial_count": 12,
        "trial_ids": list(HISTORICAL_CONTROL_IDS),
        "official_successes": successes,
        "reproduction_root": str(root),
        "terminal_records": records,
        "reproduction_sha256sums_path": "SHA256SUMS",
        "reproduction_sha256sums_sha256": _sha256_file(sums),
        "reproduction_bundle_sha256": _canonical_sha256(listed),
        "reproduction_symlinks": _tree_symlink_map(root, label="legacy CPU reproduction"),
        "historical_runtime_identity_status": "not_recorded",
        "command_declared": {"environment_device": "cpu", "policy_path": records and _legacy_trial_command_declared(_regular_json(root / HISTORICAL_CONTROL_IDS[0] / "trial.json", label="legacy CPU trial artifact"), HISTORICAL_CONTROL_IDS[0])["policy_path"], "seed": 42},
        **source_archive,
        "bundle_sha256": _canonical_sha256({"source_archive": source_archive, "reproduction": listed, "reproduction_symlinks": _tree_symlink_map(root, label="legacy CPU reproduction"), "terminal_records": records}),
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        existing = _regular_json(target, label="legacy CPU reference receipt")
        if existing != receipt:
            raise ValueError("refusing to overwrite a differing legacy CPU reference receipt")
    else:
        _write_json_atomically(target, receipt)
    return target


def _validate_legacy_cpu_reference(path: Path, expected: dict[str, object]) -> dict[str, object]:
    preliminary = _read_parity_receipt(path)
    if preliminary.get("reference_kind") != "checksummed_trial_json_reproduction":
        legacy = _validate_legacy_shared_policy_receipt(path)
        source_root = expected.get("legacy_source_root")
        if source_root is not None:
            root = Path(str(source_root))
            if root.is_symlink() or not root.is_dir():
                raise ValueError("legacy source root must be a real directory")
            actual = {name: _sha256_file(root / name) for name in _LEGACY_SHARED_POLICY_HASHES}
            if actual != _LEGACY_SHARED_POLICY_HASHES or legacy.get("source_sha256") != actual:
                raise ValueError("legacy source root files do not match the frozen shared-policy hashes")
            legacy["legacy_source_root"] = str(root.resolve())
            legacy["bundle_sha256"] = _canonical_sha256({"base": legacy["bundle_sha256"], "source": actual})
        _require_cpu_scale_identity(legacy.get("identity"), expected, label="legacy_server_cpu receipt")
        return legacy
    if preliminary.get("parity_stage") != "legacy_server_cpu" or preliminary.get("backend") != "legacy_shared_policy_server":
        raise ValueError("legacy CPU reference receipt has an invalid stage or backend")
    if tuple(preliminary.get("trial_ids", ())) != HISTORICAL_CONTROL_IDS or preliminary.get("trial_count") != 12:
        raise ValueError("legacy CPU reference receipt must bind the exact historical twelve IDs")
    root_value = preliminary.get("reproduction_root")
    records = preliminary.get("terminal_records")
    if not isinstance(root_value, str) or not isinstance(records, list) or len(records) != 12:
        raise ValueError("legacy CPU reference receipt lacks twelve checksummed terminal records")
    root = Path(root_value)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("legacy CPU reference root is unsafe")
    source_value, archive_value = preliminary.get("source_root"), preliminary.get("archive_root")
    if not isinstance(source_value, str) or not isinstance(archive_value, str):
        raise ValueError("legacy CPU reference requires source and archive roots")
    live_source = expected.get("legacy_source_root")
    if not isinstance(live_source, str) or Path(source_value).resolve() != Path(live_source).resolve():
        raise ValueError("legacy CPU reference source root does not match live legacy-source-root")
    source_archive = _legacy_source_and_archive(Path(source_value), Path(archive_value))
    for key, value in source_archive.items():
        if preliminary.get(key) != value:
            raise ValueError("legacy CPU reference source/archive evidence does not match")
    by_trial = {record.get("trial_id"): record for record in records if isinstance(record, dict)}
    if set(by_trial) != set(HISTORICAL_CONTROL_IDS):
        raise ValueError("legacy CPU reference receipt terminal IDs are invalid")
    successes = 0
    sums_path, listed = _verified_sha256sums(root, label="legacy CPU reproduction")
    if preliminary.get("reproduction_sha256sums_path") != "SHA256SUMS":
        raise ValueError("legacy CPU reference requires reproduction SHA256SUMS")
    if preliminary.get("reproduction_sha256sums_sha256") != _sha256_file(sums_path):
        raise ValueError("legacy CPU reproduction SHA256SUMS hash does not match")
    if preliminary.get("reproduction_bundle_sha256") != _canonical_sha256(listed):
        raise ValueError("legacy CPU reproduction bundle hash does not match")
    for trial_id in HISTORICAL_CONTROL_IDS:
        record = by_trial[trial_id]
        if record.get("path") != f"{trial_id}/trial.json" or listed.get(str(record.get("path"))) != record.get("sha256"):
            raise ValueError("legacy CPU trial record is not bound by reproduction SHA256SUMS")
        trial_path = _verified_receipt_file(root, record.get("path"), record.get("sha256"))
        try:
            artifact = json.loads(trial_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("legacy CPU terminal record is invalid") from error
        metric = artifact.get("metric") if isinstance(artifact, dict) else None
        declared = _legacy_trial_command_declared(artifact, trial_id) if isinstance(artifact, dict) else None
        if not isinstance(metric, dict) or not isinstance(metric.get("success"), bool) or declared is None:
            raise ValueError("legacy CPU terminal record lacks metric.success")
        successes += int(metric["success"])
    if preliminary.get("official_successes") != successes:
        raise ValueError("legacy CPU reference receipt claimed successes do not match terminal artifacts")
    if preliminary.get("historical_runtime_identity_status") != "not_recorded":
        raise ValueError("legacy CPU reference must declare historical runtime identity as not recorded")
    first = _legacy_trial_command_declared(_regular_json(root / HISTORICAL_CONTROL_IDS[0] / "trial.json", label="legacy CPU trial artifact"), HISTORICAL_CONTROL_IDS[0])
    if first is None or preliminary.get("command_declared") != first:
        raise ValueError("legacy CPU command declaration does not match terminal evidence")
    symlinks = _tree_symlink_map(root, label="legacy CPU reproduction")
    if preliminary.get("reproduction_symlinks") != symlinks or preliminary.get("bundle_sha256") != _canonical_sha256({"source_archive": source_archive, "reproduction": listed, "reproduction_symlinks": symlinks, "terminal_records": records}):
        raise ValueError("legacy CPU reference bundle hash does not match")
    validated = dict(preliminary)
    validated["official_successes"] = successes
    return validated


def _require_cpu_scale_authorization(args: argparse.Namespace, matrix: object) -> dict[str, object]:
    if getattr(args, "dry_run", False):
        raise ValueError("scale_cpu authorization is production-only and cannot run in --dry-run mode")
    if args.cpu_scale_decision != _CPU_SCALE_DECISION:
        raise ValueError("scale_cpu requires the fixed --cpu-scale-decision authorization enum")
    if args.device != "cpu" or args.execution_mode != "policy_server" or args.workers != 4:
        raise ValueError("scale_cpu requires --device cpu, --execution-mode policy_server, and exactly --workers 4")
    if getattr(args, "early_abort_completed_trials", 12) != 12 or getattr(args, "trials_per_worker", 1) != 1 or getattr(args, "max_steps", 600) != 600:
        raise ValueError("scale_cpu requires --early-abort-completed-trials 12, --trials-per-worker 1, and --max-steps 600")
    if args.minimum_reset_uniqueness_ratio != 1.0:
        raise ValueError("scale_cpu requires 100% distinct canonical reset hashes")
    if any(value is None for value in (args.legacy_server_cpu_receipt, args.server_cpu_receipt, args.cuda_abort_receipt)):
        raise ValueError("scale_cpu requires legacy CPU, server CPU, and CUDA abort evidence receipts")
    invocation = _cpu_scale_live_invocation(args, matrix)
    if getattr(args, "trial_runtime_root", None) is not None:
        required_live = ("output_root", "matrix_path", "policy_path", "policy_revision_file", "release_assets_root")
        if any(invocation.get(name) is None for name in required_live):
            raise ValueError("scale_cpu production authorization requires complete resolved invocation paths")
        invocation.update(_live_groot_identity(args))
        runtime_root, trusted_root_bindings = _scale_cpu_runtime_bindings(args)
        invocation.update({
            "trial_runtime_root": str(runtime_root),
            "trusted_pythonpath": [str(path) for path in trusted_root_bindings.values()],
            "isaacsim_python_path": str(trusted_root_bindings["isaacsim_python_path"]),
            "trusted_groot_root": str(trusted_root_bindings["trusted_groot_root"]),
            "isaaclab_source_root": str((runtime_root / "third_party" / "IsaacLab" / "source").resolve()),
        })
        invocation.update(_controller_identity())
        assets_root, assets_revision = _clean_git_revision(Path(args.release_assets_root), label="release assets root")
        if assets_revision != args.asset_revision:
            raise ValueError("release assets root revision does not match --asset-revision")
        invocation.update({"release_assets_root": str(assets_root), "release_assets_revision": assets_revision})
        legacy_source_root = getattr(args, "legacy_source_root", None)
        if legacy_source_root is not None:
            invocation["legacy_source_root"] = str(Path(legacy_source_root).resolve())
        from scripts.run_groot_flywheel_trial import policy_artifact_sha256
        actual_policy_digest = policy_artifact_sha256(Path(args.policy_path))
        if actual_policy_digest != args.policy_artifact_sha256:
            raise ValueError("policy checkpoint digest does not match --policy-artifact-sha256")
    server_cpu = _validate_server_cpu_evidence(args.server_cpu_receipt, invocation)
    legacy = _validate_legacy_cpu_reference(args.legacy_server_cpu_receipt, invocation)
    cuda_abort = _validate_cuda_abort_evidence(args.cuda_abort_receipt, invocation)
    args.output_root.mkdir(parents=True, exist_ok=True)
    legacy_root = Path(str(legacy["reproduction_root"]))
    _require_disjoint_real_roots(args.output_root, Path(str(server_cpu["artifact_root"])), legacy_root, args.cuda_abort_receipt.parent)
    decision = evaluate_cpu_scale_ladder(
        legacy_server_cpu_successes=int(legacy["official_successes"]),
        server_cpu_successes=int(server_cpu["official_successes"]),
        server_cpu_visible_contacts=int(server_cpu["_derived_evidence"]["visible_robot_garment_contacts"]),
        server_cpu_unique_resets=int(server_cpu["_derived_evidence"]["unique_reset_hashes"]),
        cuda_abort_successes=int(cuda_abort["official_successes"]),
        cuda_abort_terminal_trials=12,
    )
    if not decision.allowed:
        raise ValueError(f"scale_cpu rejected by CPU-only evidence: {', '.join(decision.reasons)}")
    authorization = {
        "schema_version": 1,
        "authorization": _CPU_SCALE_DECISION,
        "historical_legacy_exception": "historical_legacy_server_reference_substitutes_for_unsupported_direct_cpu_v1",
        "controller_stage": "scale_cpu",
        "episode_runtime_stage": "server_cpu",
        "selection": "canonical_public_280",
        "authorization_time_host_binding": True,
        "invocation": invocation,
        "runtime_binding": _cpu_runtime_binding(invocation),
        "evidence": {
            "legacy_server_cpu": {**_receipt_binding(args.legacy_server_cpu_receipt), "bundle_sha256": legacy.get("bundle_sha256")},
            "server_cpu": {**_receipt_binding(args.server_cpu_receipt), "bundle_sha256": server_cpu["_bundle_sha256"]},
            "server_cuda_abort": {**_receipt_binding(args.cuda_abort_receipt), "bundle_sha256": cuda_abort["bundle_sha256"]},
        },
    }
    path = args.output_root / _CPU_SCALE_AUTHORIZATION
    if path.exists() or path.is_symlink():
        existing = _regular_json(path, label="CPU scale authorization")
        if existing != authorization:
            raise ValueError("CPU scale authorization does not exactly match this live host/runtime and invocation")
    else:
        _write_json_atomically(path, authorization)
    return authorization


def _validate_scale_cpu_production_output(
    args: argparse.Namespace,
    state: CampaignState,
    matrix,
    authorization: dict[str, object] | None,
) -> None:
    """Fail closed on any already-materialized production episode before resume."""
    root = Path(args.output_root)
    raw = root / "raw"
    if raw.exists() and (raw.is_symlink() or not raw.is_dir()):
        raise ValueError("scale_cpu production raw root is unsafe")
    if not raw.exists():
        return
    invocation = authorization.get("invocation") if isinstance(authorization, dict) else _cpu_scale_live_invocation(args, matrix)
    if not isinstance(invocation, dict):
        raise ValueError("scale_cpu production identity is unavailable")
    if getattr(args, "workers", None) != 4:
        raise ValueError("scale_cpu production output requires the canonical four-worker topology")
    authorized_policy_devices = frozenset(f"cuda:{index}" for index in range(args.workers))
    invocation_policy_device = invocation.get("policy_device")
    if not isinstance(invocation_policy_device, str) or invocation_policy_device not in authorized_policy_devices:
        raise ValueError("scale_cpu production invocation policy device is outside the authorized worker set")
    trials = {trial_id: trial for trial_id, trial in zip(state.trial_ids, selected_trials(args, matrix), strict=True)}
    for child in raw.iterdir():
        if child.is_symlink() or not child.is_dir() or child.name not in trials:
            raise ValueError("scale_cpu production raw root contains an extra or foreign trial artifact")
        trial = trials[child.name]
        episode, _manifest = verify_episode_manifest(child)
        identity = episode.get("identity")
        provenance = episode.get("provenance")
        expected_identity = {
            "episode_id": trial.trial_id, "policy_repo": invocation["policy_repo"],
            "policy_revision": invocation["policy_revision"], "policy_step": invocation["policy_step"],
            "code_revision": invocation["code_revision"], "asset_revision": invocation["asset_revision"],
            "simulator_version": invocation["simulator_version"], "garment_name": trial.garment_name,
            "category": trial.category, "release_stage": trial.release_stage, "seed": trial.seed,
            "instruction": "fold the garment on the table", "strategy": invocation["strategy"],
        }
        if not isinstance(identity, dict) or any(identity.get(key) != value for key, value in expected_identity.items()):
            raise ValueError("scale_cpu production episode identity does not match the canonical matrix and invocation")
        if not isinstance(provenance, dict) or any(provenance.get(key) != value for key, value in {
            "execution_mode": "policy_server", "execution_backend": "policy_server", "parity_stage": "server_cpu",
            "simulator_device": "cpu",
            "policy_artifact_sha256": invocation["policy_artifact_sha256"], "image_identity": invocation["image_identity"],
        }.items()):
            raise ValueError("scale_cpu production episode provenance is foreign or mismatched")
        episode_policy_device = provenance.get("policy_device")
        if not isinstance(episode_policy_device, str) or episode_policy_device not in authorized_policy_devices:
            raise ValueError("scale_cpu production episode policy device is not in the authorized scale_cpu worker set")
        receipt = _regular_json(root / f"policy-server-receipt-{trial.trial_id}.json", label="scale_cpu production policy-server receipt")
        expected_receipt = {
            "schema_version": 1, "episode_id": trial.trial_id, "parity_stage": "server_cpu",
            "backend": "policy_server", "checkpoint_revision": invocation["policy_revision"],
            "checkpoint_digest": invocation["policy_artifact_sha256"], "code_revision": invocation["code_revision"],
            "image_identity": invocation["image_identity"], "groot_revision": invocation["groot_revision"],
            "python_path": invocation["groot_python"], "python_version": invocation.get("groot_python_version", receipt.get("python_version")),
            "policy_seed": trial.seed, "simulator_device": "cpu",
        }
        if any(receipt.get(key) != value for key, value in expected_receipt.items()):
            raise ValueError("scale_cpu production policy-server receipt does not match the episode and live invocation")
        if receipt.get("policy_device") != episode_policy_device:
            raise ValueError("scale_cpu production policy-server receipt policy device does not match its episode")


def _validate_trial_id(trial_id: str) -> None:
    if (
        not isinstance(trial_id, str)
        or trial_id in {"", ".", ".."}
        or "/" in trial_id
        or "\\" in trial_id
        or Path(trial_id).is_absolute()
        or Path(trial_id).name != trial_id
    ):
        raise ValueError("trial ID must be a non-empty path-safe identifier")


def _open_campaign_directory(parent_fd: int, name: str, *, create: bool) -> int | None:
    """Open one trusted campaign child directory without following a symlink."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        details = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        if not create:
            return None
        os.mkdir(name, dir_fd=parent_fd)
        details = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise ValueError(f"campaign {name} root is unsafe")
    try:
        return os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        raise ValueError(f"campaign {name} root is unsafe") from error


def _open_controller_lock(root_fd: int, name: str = ".campaign.lock") -> int:
    """Create the fixed lock once, then open it no-follow for every controller."""
    flags = os.O_RDWR | os.O_NOFOLLOW
    while True:
        try:
            return os.open(name, flags | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=root_fd)
        except FileExistsError:
            try:
                return os.open(name, flags, dir_fd=root_fd)
            except FileNotFoundError:
                # Another controller won creation but has not made the entry
                # observable yet; retry rather than accepting an unchecked path.
                continue


@contextmanager
def _locked_campaign_storage(output_root: Path):
    """Serialize cooperating controllers and expose a no-follow output-root FD."""
    if not hasattr(os, "O_NOFOLLOW") or os.name != "posix":
        raise ValueError("campaign retry storage requires POSIX no-follow support")
    root = Path(output_root)
    if root.is_symlink():
        raise ValueError("campaign output root must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("campaign output root must be a directory")
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    lock_fd = -1
    try:
        lock_fd = _open_controller_lock(root_fd)
        if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
            raise ValueError("campaign controller lock is unsafe")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        yield root, root_fd
    except OSError as error:
        raise ValueError("campaign output storage is unsafe") from error
    finally:
        if lock_fd >= 0:
            os.close(lock_fd)
        os.close(root_fd)


@contextmanager
def _campaign_supervisor_lease(output_root: Path):
    """Reject a second campaign controller before it can schedule any trials."""
    lease_fd = -1
    with _locked_campaign_storage(output_root) as (root, root_fd):
        lease_fd = _open_controller_lock(root_fd, ".campaign-supervisor.lock")
        if not stat.S_ISREG(os.fstat(lease_fd).st_mode):
            os.close(lease_fd)
            raise ValueError("campaign supervisor lock is unsafe")
        try:
            fcntl.flock(lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(lease_fd)
            raise ValueError("campaign supervisor is already active") from error
    try:
        yield root
    finally:
        if lease_fd >= 0:
            fcntl.flock(lease_fd, fcntl.LOCK_UN)
            os.close(lease_fd)


def _write_json_atomically(path: Path, value: object) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    fd = -1
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        remaining = memoryview(payload)
        while remaining:
            written = os.write(fd, remaining)
            if written <= 0:
                raise OSError("could not write campaign checkpoint")
            remaining = remaining[written:]
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_invocation_checkpoint(output_root: Path, invocation_id: str, checkpoint: dict[str, object]) -> None:
    if not re.fullmatch(r"[0-9a-f]{32}", invocation_id):
        raise ValueError("campaign invocation ID is invalid")
    with _locked_campaign_storage(output_root) as (root, root_fd):
        ledger_fd = _open_campaign_directory(root_fd, "campaign-ledger", create=False)
        ledger_created = ledger_fd is None
        if ledger_fd is None:
            ledger_fd = _open_campaign_directory(root_fd, "campaign-ledger", create=True)
        assert ledger_fd is not None
        os.close(ledger_fd)
        if ledger_created:
            os.fsync(root_fd)
        _write_json_atomically(root / "campaign-ledger" / f"{invocation_id}.json", checkpoint)


def _open_trial_directory(parent_fd: int, trial_id: str) -> os.stat_result | None:
    try:
        details = os.stat(trial_id, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise ValueError("campaign trial path is unsafe")
    return details


def _is_completed_locked(root: Path, root_fd: int, trial_id: str) -> bool:
    for name in (".pending", "quarantine"):
        parent_fd = _open_campaign_directory(root_fd, name, create=False)
        if parent_fd is not None:
            os.close(parent_fd)
    raw_fd = _open_campaign_directory(root_fd, "raw", create=False)
    if raw_fd is None:
        return False
    try:
        before = _open_trial_directory(raw_fd, trial_id)
        if before is None:
            return False
        episode_dir = root / "raw" / trial_id
        try:
            episode, manifest = verify_episode_manifest(episode_dir)
        except ValueError:
            return False
        after = _open_trial_directory(raw_fd, trial_id)
        if after is None or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise ValueError("campaign raw trial changed during verification")
    finally:
        os.close(raw_fd)
    if not isinstance(episode.get("terminal_reason"), str) or not episode["terminal_reason"]:
        return False
    if episode.get("outcome") == "error" or episode.get("recorder_error"):
        return False
    expected_videos = {f"videos/{filename}" for filename in CANONICAL_VIDEO_FILENAMES}
    manifest_videos = {path for path in manifest if path.startswith("videos/")}
    return manifest_videos == expected_videos and all(manifest[path]["size"] > 0 for path in expected_videos)


def is_completed_trial(output_root: Path, trial_id: str) -> bool:
    """Accept only terminal, non-error artifacts with canonical video evidence."""
    _validate_trial_id(trial_id)
    with _locked_campaign_storage(output_root) as (root, root_fd):
        return _is_completed_locked(root, root_fd, trial_id)


def _prepare_retry_attempt(output_root: Path, trial_id: str) -> None:
    """Atomically quarantine an invalid prior attempt before retrying its ID."""
    _validate_trial_id(trial_id)
    with _locked_campaign_storage(output_root) as (root, root_fd):
        if _is_completed_locked(root, root_fd, trial_id):
            return
        root_artifacts: list[tuple[str, os.stat_result]] = []
        for name in (
            f"policy-server-receipt-{trial_id}.json",
            f"flywheel-manifest-{trial_id}.json",
        ):
            try:
                details = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
                raise ValueError("campaign retry receipt is unsafe")
            root_artifacts.append((name, details))
        parent_fds: list[tuple[str, int, os.stat_result]] = []
        try:
            for parent_name in (".pending", "raw"):
                parent_fd = _open_campaign_directory(root_fd, parent_name, create=True)
                try:
                    details = _open_trial_directory(parent_fd, trial_id)
                except BaseException:
                    os.close(parent_fd)
                    raise
                if details is not None:
                    parent_fds.append((parent_name.removeprefix("."), parent_fd, details))
                else:
                    os.close(parent_fd)
            quarantine_fd = _open_campaign_directory(root_fd, "quarantine", create=True)
            if not parent_fds and not root_artifacts:
                os.close(quarantine_fd)
                return
            try:
                attempt = 1
                while True:
                    attempt_name = f"{trial_id}.attempt-{attempt:03d}"
                    try:
                        os.mkdir(attempt_name, dir_fd=quarantine_fd)
                    except FileExistsError:
                        existing = os.stat(attempt_name, dir_fd=quarantine_fd, follow_symlinks=False)
                        if stat.S_ISLNK(existing.st_mode) or not stat.S_ISDIR(existing.st_mode):
                            raise ValueError("campaign quarantine attempt path is unsafe")
                        attempt += 1
                        continue
                    attempt_fd = _open_campaign_directory(quarantine_fd, attempt_name, create=False)
                    break
                try:
                    for name, parent_fd, before in parent_fds:
                        current = _open_trial_directory(parent_fd, trial_id)
                        if current is None or (before.st_dev, before.st_ino) != (current.st_dev, current.st_ino):
                            raise ValueError("campaign trial changed during retry preparation")
                        try:
                            os.stat(name, dir_fd=attempt_fd, follow_symlinks=False)
                        except FileNotFoundError:
                            pass
                        else:
                            raise ValueError("campaign quarantine destination collision")
                        os.rename(trial_id, name, src_dir_fd=parent_fd, dst_dir_fd=attempt_fd)
                        moved = os.stat(name, dir_fd=attempt_fd, follow_symlinks=False)
                        if stat.S_ISLNK(moved.st_mode) or (moved.st_dev, moved.st_ino) != (before.st_dev, before.st_ino):
                            raise ValueError("campaign trial changed during quarantine")
                    for name, before in root_artifacts:
                        current = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                        if (
                            stat.S_ISLNK(current.st_mode)
                            or not stat.S_ISREG(current.st_mode)
                            or (before.st_dev, before.st_ino) != (current.st_dev, current.st_ino)
                        ):
                            raise ValueError("campaign retry receipt changed during retry preparation")
                        try:
                            os.stat(name, dir_fd=attempt_fd, follow_symlinks=False)
                        except FileNotFoundError:
                            pass
                        else:
                            raise ValueError("campaign quarantine destination collision")
                        os.rename(name, name, src_dir_fd=root_fd, dst_dir_fd=attempt_fd)
                        moved = os.stat(name, dir_fd=attempt_fd, follow_symlinks=False)
                        if stat.S_ISLNK(moved.st_mode) or (moved.st_dev, moved.st_ino) != (before.st_dev, before.st_ino):
                            raise ValueError("campaign retry receipt changed during quarantine")
                finally:
                    os.close(attempt_fd)
            finally:
                os.close(quarantine_fd)
        finally:
            for _, parent_fd, _ in parent_fds:
                os.close(parent_fd)


def pending_trial_ids(state: CampaignState) -> tuple[str, ...]:
    """Resume unless the terminal artifact passes the canonical completion predicate."""
    return tuple(
        trial_id
        for trial_id in state.trial_ids
        if not is_completed_trial(state.output_root, trial_id)
    )


def _write_heartbeat(path: Path, *, worker_id: int, trial_id: str, state: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"worker_id": worker_id, "trial_id": trial_id, "state": state, "monotonic_ns": time.monotonic_ns()}, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _trial_command(
    args: argparse.Namespace,
    trial: Trial,
    *,
    device: str | None = None,
    policy_server_port: int | None = None,
    policy_server_log: Path | None = None,
    policy_device: str | None = None,
) -> list[str]:
    runtime_root = getattr(args, "trial_runtime_root", None)
    scale_cpu = getattr(args, "parity_stage", None) == "scale_cpu"
    if scale_cpu and runtime_root is None:
        raise ValueError("scale_cpu requires --trial-runtime-root")
    command = [
        # Isaac trials require the controller's Isaac Python.  The pinned
        # GR00T 3.10 executable is forwarded below and used solely by the
        # isolated policy-server child which has no Isaac dependency.
        sys.executable,
        *( [str(Path(runtime_root) / "scripts" / "run_groot_flywheel_trial.py")] if scale_cpu else ["-m", "scripts.run_groot_flywheel_trial"] ),
        "--policy-path", str(args.policy_path),
        "--policy-revision-file", str(args.policy_revision_file), "--garment", trial.garment_name,
        "--policy-repo", args.policy_repo, "--policy-step", str(args.policy_step), "--code-revision", args.code_revision,
        "--asset-revision", args.asset_revision, "--simulator-version", args.simulator_version,
        "--release-assets-root", str(args.release_assets_root),
        "--category", trial.category, "--release-stage", trial.release_stage,
        "--policy-artifact-sha256", args.policy_artifact_sha256, "--image-identity", args.image_identity,
        "--seed", str(trial.seed), "--episode-id", trial.trial_id, "--output-root", str(args.output_root),
        "--strategy", args.strategy,
        "--max-steps", str(args.max_steps), "--device", device or getattr(args, "device", "cuda"),
        "--execution-mode", getattr(args, "execution_mode", "policy_server"),
    ]
    if getattr(args, "parity_stage", None) is not None:
        # `scale_cpu` is controller authorization; the immutable trial runtime
        # records the proven CPU-simulator policy-server configuration.
        command.extend(("--parity-stage", "server_cpu" if args.parity_stage == "scale_cpu" else args.parity_stage))
    if (policy_server_port is None) != (policy_server_log is None):
        raise ValueError("policy server port and log must be assigned together")
    if policy_server_port is not None:
        command.extend((
            "--groot-root", str(args.groot_root), "--groot-revision", args.groot_revision,
            "--groot-python", str(args.groot_python), "--policy-server-port", str(policy_server_port),
            "--policy-server-readiness-timeout", str(args.policy_server_readiness_timeout),
            "--policy-server-request-timeout", str(args.policy_server_request_timeout),
            "--policy-server-termination-grace", str(args.policy_server_termination_grace),
            "--policy-server-log", str(policy_server_log),
            "--policy-device", policy_device or getattr(args, "policy_device", ""),
        ))
    historical_config = _historical_config_root(args, trial)
    if historical_config is not None:
        command.extend(("--historical-control-config", str(historical_config)))
    command.append("--headless")
    return command


def _attempt_log_paths(worker_root: Path, trial_id: str) -> tuple[Path, Path]:
    attempt = 1
    while True:
        worker_log = worker_root / f"{trial_id}.attempt-{attempt:03d}.log"
        policy_server_log = worker_root / f"{trial_id}.attempt-{attempt:03d}.policy-server.log"
        if all(not path.exists() and not path.is_symlink() for path in (worker_log, policy_server_log)):
            return worker_log, policy_server_log
        attempt += 1


def _attempt_log_path(worker_root: Path, trial_id: str) -> Path:
    """Return the normal worker log while reserving its paired server-log suffix."""
    return _attempt_log_paths(worker_root, trial_id)[0]


def _allocate_loopback_port() -> int:
    """Reserve a candidate loopback port long enough to prevent in-wave collisions.

    The trial rechecks the candidate immediately before binding the policy server,
    because a released ephemeral port cannot be held across an exec boundary.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _allocate_loopback_ports(workers: int) -> tuple[int, ...]:
    if workers <= 0:
        raise ValueError("worker count must be positive")
    ports: set[int] = set()
    max_attempts = workers * 16
    while len(ports) != workers and max_attempts:
        ports.add(_allocate_loopback_port())
        max_attempts -= 1
    if len(ports) != workers:
        raise ValueError("could not allocate unique loopback policy-server ports")
    return tuple(sorted(ports))


def _worker_process_group_alive(process: object) -> bool:
    """Return whether the scheduler-owned process group still exists.

    Scheduler-launched trial parents are session leaders, so their PID is a
    unique process-group ID.  A parent can exit before its policy-server child;
    probing the group rather than the parent is what catches that orphan.
    """
    pid = getattr(process, "pid", None)
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return getattr(process, "poll")() is None
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _signal_worker_process_group(process: object, signum: int) -> None:
    """Signal only one scheduler-created trial group, never the supervisor."""
    pid = getattr(process, "pid", None)
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        # Test doubles and legacy injected processes have no stable group ID.
        # Production Popen instances always take the group-safe branch below.
        if signum == signal.SIGTERM:
            getattr(process, "terminate")()
            return
        if signum == signal.SIGKILL:
            getattr(process, "kill")()
            return
        raise ValueError("unsupported worker process-group signal")
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:
        # The group was already reaped between polling and signalling.
        pass


def _await_worker_process_group_clearance(process: object, *, grace_seconds: float) -> None:
    """Fail closed unless the known worker process group disappears on time."""
    deadline = time.monotonic() + grace_seconds
    while _worker_process_group_alive(process) and time.monotonic() < deadline:
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    if _worker_process_group_alive(process):
        pid = getattr(process, "pid", "unknown")
        raise RuntimeError(f"worker process group {pid} survived SIGKILL")


def _shutdown_worker_process_group(process: object, *, grace_seconds: float) -> None:
    """Terminate, kill if needed, and reap a trial process-tree boundary."""
    pid = getattr(process, "pid", None)
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        _signal_worker_process_group(process, signal.SIGTERM)
        try:
            getattr(process, "wait")(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            _signal_worker_process_group(process, signal.SIGKILL)
            getattr(process, "wait")()
        return
    if not _worker_process_group_alive(process):
        getattr(process, "wait")(timeout=grace_seconds)
        return
    _signal_worker_process_group(process, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while _worker_process_group_alive(process) and time.monotonic() < deadline:
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    if _worker_process_group_alive(process):
        _signal_worker_process_group(process, signal.SIGKILL)
        _await_worker_process_group_clearance(process, grace_seconds=grace_seconds)
    # Reap the direct child even when a descendant caused the group to survive.
    try:
        getattr(process, "wait")(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        _signal_worker_process_group(process, signal.SIGKILL)
        _await_worker_process_group_clearance(process, grace_seconds=grace_seconds)
        getattr(process, "wait")()


def _run_one_worker(args: argparse.Namespace, *, worker_id: int, trial: Trial) -> int:
    worker_root = args.output_root / "workers" / f"worker-{worker_id:02d}"
    heartbeat = worker_root / "heartbeat.json"
    log_path, policy_server_log = _attempt_log_paths(worker_root, trial.trial_id)
    policy_server_port = _allocate_loopback_ports(1)[0] if getattr(args, "execution_mode", "policy_server") == "policy_server" else None
    _prepare_retry_attempt(args.output_root, trial.trial_id)
    _write_heartbeat(heartbeat, worker_id=worker_id, trial_id=trial.trial_id, state="started")
    scale_cpu_cwd = str(_scale_cpu_runtime_paths(args)[0]) if getattr(args, "parity_stage", None) == "scale_cpu" else None
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            _trial_command(
                args,
                trial,
                policy_server_port=policy_server_port,
                policy_server_log=policy_server_log if policy_server_port is not None else None,
            ),
            stdout=log, stderr=subprocess.STDOUT, env=_worker_environment(args, _cuda_device_index(args.device)),
            start_new_session=True, **({"cwd": scale_cpu_cwd} if scale_cpu_cwd is not None else {}),
        )
        try:
            returncode = process.wait(timeout=args.worker_timeout_seconds)
        except subprocess.TimeoutExpired:
            _shutdown_worker_process_group(process, grace_seconds=args.terminate_grace_seconds)
            _write_heartbeat(heartbeat, worker_id=worker_id, trial_id=trial.trial_id, state="timeout")
            return 124
        except BaseException as worker_error:
            cleanup_errors: list[BaseException] = []
            try:
                _shutdown_worker_process_group(process, grace_seconds=args.terminate_grace_seconds)
            except BaseException as error:
                cleanup_errors.append(error)
            try:
                _write_heartbeat(heartbeat, worker_id=worker_id, trial_id=trial.trial_id, state="interrupted")
            except BaseException as error:
                cleanup_errors.append(error)
            _report_launch_cleanup_failures(worker_error, cleanup_errors)
            raise
    if isinstance(getattr(process, "pid", None), int) and not isinstance(getattr(process, "pid", None), bool):
        _shutdown_worker_process_group(process, grace_seconds=args.terminate_grace_seconds)
    _write_heartbeat(heartbeat, worker_id=worker_id, trial_id=trial.trial_id, state="terminal")
    return returncode


def _validate_sweep(values: str) -> tuple[int, ...]:
    if values != "1,2,4":
        raise ValueError("four-GPU capacity sweep must be exactly 1,2,4")
    return (1, 2, 4)


def _positive_finite_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return seconds


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _authorized_production_worker_count(value: str) -> int:
    parsed = _positive_int(value)
    if parsed > 4:
        raise argparse.ArgumentTypeError("production workers must be between 1 and 4")
    return parsed


def _resource_margins(gpu_indices: Sequence[int] | None = None) -> tuple[float, float, float]:
    """Read host and assigned-GPU free margins; unknown telemetry fails closed."""
    host_margin = 0.0
    try:
        entries = dict(
            line.replace(":", "").split()[:2]
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
            if ":" in line
        )
        host_margin = int(entries["MemAvailable"]) / int(entries["MemTotal"])
    except (KeyError, OSError, ValueError):
        pass
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
            timeout=5.0,
        )
        observed = {
            int(index): (int(free), int(total))
            for line in completed.stdout.splitlines()
            for index, free, total in [line.split(",")]
        }
        assigned = tuple(gpu_indices) if gpu_indices is not None else tuple(observed)
        if len(set(assigned)) != len(assigned) or any(index not in observed for index in assigned):
            raise ValueError("assigned GPU telemetry is unavailable")
        margins = [observed[index][0] / observed[index][1] for index in assigned]
        vram_margin = min(margins) if completed.returncode == 0 and margins else 0.0
    except (OSError, ValueError, ZeroDivisionError, subprocess.TimeoutExpired):
        vram_margin = 0.0
    # Kept as a compatibility return slot for callers still unpacking three
    # values. Campaign decisions consume this once as combined Isaac+policy
    # usage, never as independent renderer and inference evidence.
    return host_margin, vram_margin, vram_margin


def _cuda_device_index(device: str) -> int | None:
    if device == "cpu":
        return None
    if device == "cuda":
        return 0
    if device.startswith("cuda:") and device.removeprefix("cuda:").isdigit():
        return int(device.removeprefix("cuda:"))
    raise ValueError("rollout device must be cpu, cuda, or cuda:<non-negative-index>")


def _visible_gpu_indices() -> tuple[int, ...]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            check=False, capture_output=True, text=True, timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError("CUDA rollout isolation requires nvidia-smi GPU inventory") from error
    if result.returncode != 0:
        raise ValueError("CUDA rollout isolation requires nvidia-smi GPU inventory")
    try:
        indices = tuple(int(line.strip()) for line in result.stdout.splitlines() if line.strip())
    except ValueError as error:
        raise ValueError("CUDA rollout isolation received an invalid GPU inventory") from error
    if not indices or len(set(indices)) != len(indices) or any(index < 0 for index in indices):
        raise ValueError("CUDA rollout isolation received an invalid GPU inventory")
    return indices


def _worker_gpu_indices(args: argparse.Namespace, workers: int) -> tuple[int | None, ...]:
    """Assign one unique physical GPU to every concurrent Isaac+policy process."""
    if workers <= 0:
        raise ValueError("worker count must be positive")
    requested = _cuda_device_index(getattr(args, "device", "cuda"))
    if requested is None:
        if getattr(args, "execution_mode", "policy_server") == "direct" and getattr(args, "historical_control", False):
            return (None,) * workers
        if getattr(args, "execution_mode", "policy_server") == "direct":
            raise ValueError("CPU rollout workers may be parallel only for the historical direct control")
        requested = _cuda_device_index(getattr(args, "policy_device", "cuda:0"))
    available = _visible_gpu_indices()
    if requested not in available:
        raise ValueError("requested rollout CUDA device is not visible")
    ordered = (requested, *(index for index in available if index != requested))
    if workers > len(ordered):
        raise ValueError("unsupported GPU oversubscription: each rollout worker requires one isolated GPU")
    return ordered[:workers]


def _worker_environment(
    args: argparse.Namespace,
    gpu_index: int | None,
    *,
    policy_telemetry_path: Path | None = None,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("LEHOME_FLYWHEEL_POLICY_TELEMETRY_PATH", None)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment.pop("LEHOME_FLYWHEEL_WORKER_GPU", None)
    if getattr(args, "parity_stage", None) == "scale_cpu":
        _runtime_root, trusted_paths = _scale_cpu_runtime_paths(args)
        environment["PYTHONPATH"] = os.pathsep.join(str(path) for path in trusted_paths)
    if gpu_index is not None:
        # Keep every physical GPU visible to Isaac's Vulkan/CUDA interop.  The
        # worker GPU is forwarded separately so AppLauncher can select the
        # matching physical renderer without remapping CUDA device indices.
        environment["LEHOME_FLYWHEEL_WORKER_GPU"] = str(gpu_index)
    if policy_telemetry_path is not None:
        environment["LEHOME_FLYWHEEL_POLICY_TELEMETRY_PATH"] = str(policy_telemetry_path)
    return environment


def _worker_root_name(worker_id: int) -> str:
    if not isinstance(worker_id, int) or isinstance(worker_id, bool) or worker_id <= 0:
        raise ValueError("worker ID must be a positive integer")
    return f"worker-{worker_id:02d}"


@dataclass(frozen=True, slots=True)
class _ProvisionedPolicyTelemetry:
    path: Path
    device: int
    inode: int

    def __fspath__(self) -> str:
        return os.fspath(self.path)

    def __getattr__(self, name: str):
        return getattr(self.path, name)


def _prepare_policy_telemetry_path(output_root: Path, *, worker_id: int) -> _ProvisionedPolicyTelemetry:
    """Provision one exclusive append-only telemetry file under its worker root."""
    root = Path(output_root)
    if root.is_symlink():
        raise ValueError("campaign output root is unsafe")
    root.mkdir(parents=True, exist_ok=True)
    try:
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        raise ValueError("campaign output root is unsafe") from error
    workers_fd = worker_fd = telemetry_fd = -1
    filename = f"policy-telemetry-{uuid4().hex}.jsonl"
    worker_name = _worker_root_name(worker_id)
    try:
        workers_fd = _open_campaign_directory(root_fd, "workers", create=True)
        assert workers_fd is not None
        worker_fd = _open_campaign_directory(workers_fd, worker_name, create=True)
        assert worker_fd is not None
        telemetry_fd = os.open(
            filename,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=worker_fd,
        )
        details = os.fstat(telemetry_fd)
        if not stat.S_ISREG(details.st_mode):
            raise ValueError("policy telemetry path is unsafe")
    except OSError as error:
        raise ValueError("policy telemetry path is unsafe") from error
    finally:
        if telemetry_fd >= 0:
            os.close(telemetry_fd)
        if worker_fd >= 0:
            os.close(worker_fd)
        if workers_fd >= 0:
            os.close(workers_fd)
        os.close(root_fd)
    return _ProvisionedPolicyTelemetry(
        root / "workers" / worker_name / filename,
        details.st_dev,
        details.st_ino,
    )


class _PolicyTelemetrySampler:
    """Tail pre-provisioned worker files and retain only strict, attributable records."""

    _MAX_BYTES = 1_048_576
    _REQUIRED_KEYS = {"request_id", "latency_seconds", "queue_depth_after_enqueue"}

    def __init__(self, paths: dict[int, _ProvisionedPolicyTelemetry], *, wave_started_ns: int) -> None:
        self._paths = dict(paths)
        self._wave_started_ns = wave_started_ns
        self._fatal_failures: dict[int, list[str]] = {}
        self._latencies: list[float] = []
        self._queue_depths: list[int] = []
        self._offsets: dict[int, tuple[int, int, int]] = {}
        self._partial_lines: dict[int, bytes] = {}
        self._observed_workers: set[int] = set()

    def _record_failure(self, worker_id: int, reason: str) -> None:
        failures = self._fatal_failures.setdefault(worker_id, [])
        if reason not in failures:
            failures.append(reason)

    def _failure_records(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {"worker_id": worker_id, "failure_class": reason}
            for worker_id in sorted(self._fatal_failures)
            for reason in self._fatal_failures[worker_id]
        )

    def _read_worker(
        self,
        worker_id: int,
        provisioned: _ProvisionedPolicyTelemetry,
        *,
        final: bool,
    ) -> tuple[tuple[float, int], ...] | None:
        path = provisioned.path
        if path.parent.name != _worker_root_name(worker_id) or path.parent.parent.name != "workers":
            self._record_failure(worker_id, "policy_telemetry_wrong_worker")
            return None
        try:
            before = os.stat(path, follow_symlinks=False)
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
                self._record_failure(worker_id, "policy_telemetry_unsafe_path")
                return None
            if before.st_mtime_ns < self._wave_started_ns:
                self._record_failure(worker_id, "policy_telemetry_stale")
                return None
            if before.st_size > self._MAX_BYTES:
                self._record_failure(worker_id, "policy_telemetry_malformed")
                return None
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        except OSError:
            self._record_failure(worker_id, "policy_telemetry_missing")
            return None
        try:
            after = os.fstat(fd)
            if (
                not stat.S_ISREG(after.st_mode)
                or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
                or (after.st_dev, after.st_ino) != (provisioned.device, provisioned.inode)
            ):
                self._record_failure(worker_id, "policy_telemetry_unsafe_path")
                return None
            previous = self._offsets.get(worker_id)
            identity = (after.st_dev, after.st_ino)
            if previous is None:
                offset = 0
            elif previous[:2] != identity or after.st_size < previous[2]:
                self._record_failure(worker_id, "policy_telemetry_unsafe_path")
                return None
            else:
                offset = previous[2]
            remaining = after.st_size - offset
            if remaining > self._MAX_BYTES:
                self._record_failure(worker_id, "policy_telemetry_malformed")
                return None
            os.lseek(fd, offset, os.SEEK_SET)
            chunks: list[bytes] = []
            while remaining:
                chunk = os.read(fd, remaining)
                if not chunk:
                    self._record_failure(worker_id, "policy_telemetry_unsafe_path")
                    return None
                chunks.append(chunk)
                remaining -= len(chunk)
            payload = b"".join(chunks)
            self._offsets[worker_id] = (*identity, after.st_size)
        finally:
            os.close(fd)
        payload = self._partial_lines.pop(worker_id, b"") + payload
        if not payload:
            return ()
        values: list[tuple[float, int]] = []
        try:
            lines = payload.split(b"\n")
            trailing = lines.pop()
            if trailing:
                if final:
                    raise ValueError
                self._partial_lines[worker_id] = trailing
            for line in lines:
                if not line:
                    raise ValueError
                record = json.loads(line.decode("utf-8"))
                if not isinstance(record, dict) or set(record) != self._REQUIRED_KEYS:
                    raise ValueError
                request_id = record["request_id"]
                latency = record["latency_seconds"]
                queue_depth = record["queue_depth_after_enqueue"]
                if (
                    not isinstance(request_id, str)
                    or not request_id
                    or not isinstance(latency, (int, float))
                    or isinstance(latency, bool)
                    or not math.isfinite(latency)
                    or latency < 0
                    or not isinstance(queue_depth, int)
                    or isinstance(queue_depth, bool)
                    or queue_depth < 0
                ):
                    raise ValueError
                values.append((float(latency), queue_depth))
        except (UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError):
            self._record_failure(worker_id, "policy_telemetry_malformed")
            return None
        return tuple(values)

    def sample(self, *, final: bool = False) -> dict[str, object]:
        for worker_id, provisioned in self._paths.items():
            values = self._read_worker(worker_id, provisioned, final=final)
            if values is None:
                continue
            if values:
                self._observed_workers.add(worker_id)
                self._latencies.extend(latency for latency, _ in values)
                self._queue_depths.extend(depth for _, depth in values)
            elif final and worker_id not in self._observed_workers:
                self._record_failure(worker_id, "policy_telemetry_missing")
        failures = self._failure_records()
        valid = not failures and len(self._observed_workers) == len(self._paths) and self._latencies and self._queue_depths
        return {
            "inference_latency_seconds": max(self._latencies) if valid else None,
            "inference_queue_depth": max(self._queue_depths) if valid else None,
            "policy_evidence_failures": tuple(record["failure_class"] for record in failures),
            "policy_evidence_records": failures,
        }


def _trial_has_first_progress(output_root: Path, trial_id: str) -> bool:
    """The recorder's first committed annotation is stronger than a launched PID."""
    for parent in (".pending", "raw"):
        annotations = output_root / parent / trial_id / "annotations.jsonl"
        try:
            details = annotations.stat(follow_symlinks=False)
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(details.st_mode):
            continue
        if details.st_size > 0:
            return True
    return False


def _capacity_telemetry(gpu_indices: Sequence[int] | None = None) -> dict[str, object]:
    """Take one bounded host/assigned-GPU sample; unknown fields stay explicit."""
    host_margin, combined_vram_margin, _ = _resource_margins(gpu_indices)
    sample: dict[str, object] = {
        "host_ram_margin": host_margin,
        "combined_vram_margin": combined_vram_margin,
        "peak_host_ram_bytes": None,
        "peak_vram_bytes": None,
        "cpu_utilization": None,
        "run_queue": None,
        "inference_latency_seconds": None,
        "inference_queue_depth": None,
    }
    try:
        entries = {
            line.split(":", 1)[0]: int(line.split()[1]) * 1024
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines() if ":" in line
        }
        sample["peak_host_ram_bytes"] = entries["MemTotal"] - entries["MemAvailable"]
    except (KeyError, OSError, ValueError, IndexError):
        pass
    try:
        run_queue = Path("/proc/loadavg").read_text(encoding="utf-8").split()[3].split("/", 1)[0]
        sample["run_queue"] = int(run_queue)
    except (OSError, ValueError, IndexError):
        pass
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits"],
            check=False, capture_output=True, text=True, timeout=5.0,
        )
        observed = {
            int(index): (int(free), int(total))
            for line in result.stdout.splitlines()
            for index, free, total in [line.split(",")]
        }
        assigned = tuple(gpu_indices) if gpu_indices is not None else tuple(observed)
        usage = [observed[index][1] - observed[index][0] for index in assigned]
        if result.returncode == 0 and usage:
            sample["peak_vram_bytes"] = max(usage) * 1024 * 1024
    except (KeyError, OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return sample


def _cpu_counters() -> tuple[int, int] | None:
    """Return aggregate and idle CPU jiffies for a delta-based utilization reading."""
    try:
        fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
        if fields[0] != "cpu" or len(fields) < 5:
            return None
        values = tuple(int(value) for value in fields[1:])
    except (OSError, ValueError, IndexError):
        return None
    return sum(values), values[3] + (values[4] if len(values) > 4 else 0)


class _CapacityTelemetrySampler:
    """Associate each /proc/stat delta with an execution-time telemetry sample."""

    def __init__(self, gpu_indices: Sequence[int] | None = None) -> None:
        self._previous_cpu = _cpu_counters()
        self._gpu_indices = tuple(gpu_indices) if gpu_indices is not None else None

    def sample(self) -> dict[str, object]:
        sample = _capacity_telemetry(self._gpu_indices)
        current_cpu = _cpu_counters()
        if self._previous_cpu is not None and current_cpu is not None:
            total_delta = current_cpu[0] - self._previous_cpu[0]
            idle_delta = current_cpu[1] - self._previous_cpu[1]
            if total_delta > 0 and 0 <= idle_delta <= total_delta:
                sample["cpu_utilization"] = (total_delta - idle_delta) / total_delta
        self._previous_cpu = current_cpu
        return sample


def _cleanup_partially_launched_workers(
    args: argparse.Namespace,
    processes: Sequence[tuple[int, Trial, subprocess.Popen[str], Path, object, Path]],
) -> list[BaseException]:
    """Bound a best-effort shutdown without hiding the launch failure that caused it."""
    errors: list[BaseException] = []
    to_reap = list(processes)
    pending: list[tuple[int, Trial, subprocess.Popen[str], Path, object, Path]] = []
    for record in processes:
        worker_id, trial, process, heartbeat, log, log_path = record
        try:
            if process.poll() is None:
                pending.append(record)
        except BaseException as error:
            errors.append(RuntimeError(f"worker {worker_id} could not be polled during launch cleanup: {error}"))
            pending.append(record)

    for worker_id, trial, process, heartbeat, log, log_path in pending:
        try:
            _signal_worker_process_group(process, signal.SIGTERM)
        except BaseException as error:
            errors.append(RuntimeError(f"worker {worker_id} could not be terminated during launch cleanup: {error}"))
        try:
            _write_heartbeat(heartbeat, worker_id=worker_id, trial_id=trial.trial_id, state="timeout")
        except BaseException as error:
            errors.append(RuntimeError(f"worker {worker_id} heartbeat cleanup failed: {error}"))

    terminate_deadline = time.monotonic() + args.terminate_grace_seconds
    while pending and time.monotonic() < terminate_deadline:
        still_pending = []
        for record in pending:
            worker_id, trial, process, heartbeat, log, log_path = record
            try:
                if process.poll() is None:
                    still_pending.append(record)
            except BaseException as error:
                errors.append(RuntimeError(f"worker {worker_id} could not be polled during launch cleanup: {error}"))
                still_pending.append(record)
        pending = still_pending
        if pending:
            time.sleep(min(0.1, max(0.0, terminate_deadline - time.monotonic())))

    # A parent can acknowledge SIGTERM while a detached descendant remains.
    # Check the scheduler-owned group, rather than only the direct parent,
    # before releasing this wave's GPU allocation.
    groups_still_alive = [
        record for record in to_reap
        if isinstance(getattr(record[2], "pid", None), int)
        and not isinstance(getattr(record[2], "pid", None), bool)
        and _worker_process_group_alive(record[2])
    ]
    kill_records = [
        *groups_still_alive,
        *(record for record in pending if not isinstance(getattr(record[2], "pid", None), int)),
    ]
    for worker_id, trial, process, heartbeat, log, log_path in kill_records:
        try:
            _signal_worker_process_group(process, signal.SIGKILL)
        except BaseException as error:
            errors.append(RuntimeError(f"worker {worker_id} could not be killed during launch cleanup: {error}"))
    for worker_id, trial, process, heartbeat, log, log_path in groups_still_alive:
        try:
            _await_worker_process_group_clearance(process, grace_seconds=args.terminate_grace_seconds)
        except BaseException as error:
            errors.append(RuntimeError(f"worker {worker_id} process group did not clear during launch cleanup: {error}"))

    reap_deadline = time.monotonic() + args.terminate_grace_seconds
    for worker_id, trial, process, heartbeat, log, log_path in to_reap:
        remaining = max(0.0, reap_deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except BaseException as error:
            errors.append(RuntimeError(f"worker {worker_id} could not be reaped during launch cleanup: {error}"))

    for worker_id, trial, process, heartbeat, log, log_path in processes:
        try:
            log.close()
        except BaseException as error:
            errors.append(RuntimeError(f"worker {worker_id} log could not be closed during launch cleanup: {error}"))
    return errors


def _report_launch_cleanup_failures(launch_error: BaseException, cleanup_errors: Sequence[BaseException]) -> None:
    """Keep the launch exception primary while making cleanup faults observable on Python 3.10+."""
    if not cleanup_errors:
        return
    detail = "; ".join(str(error) for error in cleanup_errors)
    if hasattr(launch_error, "add_note"):
        launch_error.add_note(f"Additional launch cleanup failures: {detail}")
    else:
        print(f"Additional launch cleanup failures: {detail}", file=sys.stderr)


def _attach_launch_accounting(
    error: BaseException,
    assignments: Sequence[tuple[int, Trial]],
    processes: Sequence[tuple[int, Trial, subprocess.Popen[str], Path, object, Path]],
) -> None:
    """Preserve exactly which trial Popen calls succeeded before a launch fault."""
    scheduled = tuple(trial.trial_id for _, trial in assignments)
    launched = tuple(trial.trial_id for _, trial, *_ in processes)
    try:
        setattr(error, "scheduled_trial_ids", scheduled)
        setattr(error, "launched_trial_ids", launched)
    except (AttributeError, TypeError):
        # Built-in exception subclasses normally permit attributes.  If an
        # unusual BaseException does not, its original identity stays primary.
        if hasattr(error, "add_note"):
            error.add_note(
                f"Launch accounting: scheduled={list(scheduled)!r}; launched={list(launched)!r}"
            )


def _launch_accounting_from_error(
    error: BaseException,
    scheduled_trial_ids: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Read validated partial-launch metadata without trusting injected errors."""
    scheduled = list(scheduled_trial_ids)
    launched_value = getattr(error, "launched_trial_ids", None)
    scheduled_value = getattr(error, "scheduled_trial_ids", None)
    if (
        isinstance(scheduled_value, (tuple, list))
        and all(isinstance(trial_id, str) for trial_id in scheduled_value)
        and list(scheduled_value) == scheduled
    ):
        scheduled = list(scheduled_value)
    if (
        isinstance(launched_value, (tuple, list))
        and all(isinstance(trial_id, str) for trial_id in launched_value)
        and len(set(launched_value)) == len(launched_value)
        and set(launched_value).issubset(scheduled)
    ):
        return scheduled, list(launched_value)
    # Older/injected worker-group implementations cannot report partial
    # progress.  Conservatively preserve the historical all-launched contract.
    return scheduled, list(scheduled)


def _failure_classes(log_path: Path, *, returncode: int, progressed: bool) -> tuple[str, ...]:
    classes: list[str] = []
    if returncode:
        classes.append("timeout" if returncode == 124 else "nonzero_exit")
    if not progressed:
        classes.append("no_first_progress")
    if returncode:
        try:
            payload = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            payload = ""
        for marker, label in (
            (r"stale\s+ipc", "stale_ipc"),
            (r"vulkan", "vulkan"),
            (r"cuda", "cuda"),
            (r"policy", "policy"),
            (r"encoder", "video_encoder"),
        ):
            if re.search(rf"(?im)^\s*(?:error|fatal|critical)\b[^\n]*{marker}\b", payload):
                classes.append(label)
    return tuple(dict.fromkeys(classes))


def _run_worker_group(
    args: argparse.Namespace,
    assignments: Sequence[tuple[int, Trial]],
    *,
    gpu_indices: Sequence[int | None] | None = None,
    collect_telemetry: bool = False,
):
    """Start workers together and apply one launch-relative deadline to all."""
    started = time.monotonic()
    wave_started_ns = time.time_ns()
    processes: list[tuple[int, Trial, subprocess.Popen[str], Path, object, Path]] = []
    if gpu_indices is not None and len(gpu_indices) != len(assignments):
        raise ValueError("worker GPU assignment does not match the launched worker group")
    telemetry_sampler = _CapacityTelemetrySampler(gpu_indices) if collect_telemetry else None
    policy_telemetry_paths: dict[int, _ProvisionedPolicyTelemetry] = {}
    first_progress: dict[int, float] = {}
    launch_log: object | None = None
    try:
        # Reserve every per-worker port inside the accounting boundary: a
        # collision here means no trial Popen succeeded, not a full wave.
        policy_server_ports = _allocate_loopback_ports(len(assignments)) if getattr(args, "execution_mode", "policy_server") == "policy_server" else ()
        for index, (worker_id, trial) in enumerate(assignments):
            worker_root = args.output_root / "workers" / f"worker-{worker_id:02d}"
            heartbeat = worker_root / "heartbeat.json"
            log_path, policy_server_log = _attempt_log_paths(worker_root, trial.trial_id)
            _prepare_retry_attempt(args.output_root, trial.trial_id)
            _write_heartbeat(heartbeat, worker_id=worker_id, trial_id=trial.trial_id, state="started")
            launch_log = log_path.open("x", encoding="utf-8")
            physical_gpu = gpu_indices[index] if gpu_indices is not None else None
            policy_telemetry_path = (
                _prepare_policy_telemetry_path(args.output_root, worker_id=worker_id)
                if collect_telemetry
                else None
            )
            policy_server_port = policy_server_ports[index] if policy_server_ports else None
            process = subprocess.Popen(
                _trial_command(
                    args,
                    trial,
                    device=(getattr(args, "device", "cuda:0") if getattr(args, "device", "cuda:0") == "cpu" else f"cuda:{physical_gpu}"),
                    policy_server_port=policy_server_port,
                    policy_server_log=policy_server_log if policy_server_port is not None else None,
                    policy_device=f"cuda:{physical_gpu}" if physical_gpu is not None else None,
                ), stdout=launch_log, stderr=subprocess.STDOUT,
                env=_worker_environment(
                    args,
                    physical_gpu,
                    policy_telemetry_path=(policy_telemetry_path.path if policy_telemetry_path else None),
                ),
                start_new_session=True,
                **({"cwd": str(_scale_cpu_runtime_paths(args)[0])} if getattr(args, "parity_stage", None) == "scale_cpu" else {}),
            )
            if policy_telemetry_path is not None:
                policy_telemetry_paths[worker_id] = policy_telemetry_path
            processes.append((worker_id, trial, process, heartbeat, launch_log, log_path))
            launch_log = None
    except BaseException as launch_error:
        _attach_launch_accounting(launch_error, assignments, processes)
        cleanup_errors = _cleanup_partially_launched_workers(args, processes)
        if launch_log is not None:
            try:
                launch_log.close()
            except BaseException as error:
                cleanup_errors.append(RuntimeError(f"unlaunched worker log could not be closed during launch cleanup: {error}"))
        _report_launch_cleanup_failures(launch_error, cleanup_errors)
        raise
    try:
        return _monitor_worker_group(
            args,
            processes,
            started=started,
            wave_started_ns=wave_started_ns,
            collect_telemetry=collect_telemetry,
            telemetry_sampler=telemetry_sampler,
            policy_telemetry_paths=policy_telemetry_paths,
        )
    except BaseException as worker_error:
        cleanup_errors = _cleanup_partially_launched_workers(args, processes)
        _report_launch_cleanup_failures(worker_error, cleanup_errors)
        raise


def _monitor_worker_group(
    args: argparse.Namespace,
    processes,
    *,
    started: float,
    wave_started_ns: int,
    collect_telemetry: bool,
    telemetry_sampler: _CapacityTelemetrySampler | None,
    policy_telemetry_paths: dict[int, _ProvisionedPolicyTelemetry],
):
    returncodes: dict[int, int] = {}
    telemetry_samples: list[dict[str, object]] = []
    first_progress: dict[int, float] = {}
    pending = list(processes)
    policy_telemetry_sampler = (
        _PolicyTelemetrySampler(policy_telemetry_paths, wave_started_ns=wave_started_ns)
        if collect_telemetry
        else None
    )
    if collect_telemetry:
        sample = telemetry_sampler.sample()
        sample.update(policy_telemetry_sampler.sample())
        telemetry_samples.append(sample)
    deadline = started + args.worker_timeout_seconds
    while pending and time.monotonic() < deadline:
        still_pending: list[tuple[int, Trial, subprocess.Popen[str], Path, object, Path]] = []
        for worker_id, trial, process, heartbeat, log, log_path in pending:
            if worker_id not in first_progress and _trial_has_first_progress(args.output_root, trial.trial_id):
                first_progress[worker_id] = time.monotonic() - started
            returncode = process.poll()
            if returncode is None:
                still_pending.append((worker_id, trial, process, heartbeat, log, log_path))
                continue
            returncodes[worker_id] = returncode
            _write_heartbeat(heartbeat, worker_id=worker_id, trial_id=trial.trial_id, state="terminal")
        pending = still_pending
        if collect_telemetry:
            sample = telemetry_sampler.sample()
            sample.update(policy_telemetry_sampler.sample())
            telemetry_samples.append(sample)
        if pending:
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    if pending:
        for worker_id, trial, process, heartbeat, log, log_path in pending:
            _signal_worker_process_group(process, signal.SIGTERM)
            _write_heartbeat(heartbeat, worker_id=worker_id, trial_id=trial.trial_id, state="timeout")
        grace_deadline = time.monotonic() + args.terminate_grace_seconds
        while pending and time.monotonic() < grace_deadline:
            still_pending = []
            for worker_id, trial, process, heartbeat, log, log_path in pending:
                if process.poll() is None:
                    still_pending.append((worker_id, trial, process, heartbeat, log, log_path))
            pending = still_pending
            if pending:
                time.sleep(min(0.1, max(0.0, grace_deadline - time.monotonic())))
        # Do not treat a reaped trial parent as proof that its policy-server
        # descendant is gone.  Its PID remains the known group ID.
        groups_still_alive = [
            record for record in processes
            if isinstance(getattr(record[2], "pid", None), int)
            and not isinstance(getattr(record[2], "pid", None), bool)
            and _worker_process_group_alive(record[2])
        ]
        # Legacy injected process doubles have no group ID, so retain the
        # direct-parent pending list for that compatibility seam.
        kill_records = [
            *groups_still_alive,
            *(record for record in pending if not isinstance(getattr(record[2], "pid", None), int)),
        ]
        for worker_id, trial, process, heartbeat, log, log_path in kill_records:
            _signal_worker_process_group(process, signal.SIGKILL)
        for worker_id, trial, process, heartbeat, log, log_path in groups_still_alive:
            _await_worker_process_group_clearance(process, grace_seconds=args.terminate_grace_seconds)
        reap_deadline = time.monotonic() + args.terminate_grace_seconds
        for worker_id, trial, process, heartbeat, log, log_path in pending:
            remaining = reap_deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"worker {worker_id} did not exit after SIGKILL")
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(f"worker {worker_id} did not exit after SIGKILL") from error
        for worker_id, trial, process, heartbeat, log, log_path in processes:
            if worker_id not in returncodes:
                returncodes[worker_id] = 124

    completed = failed = 0
    worker_failures: list[dict[str, object]] = []
    for worker_id, trial, process, heartbeat, log, log_path in processes:
        returncode = returncodes[worker_id]
        if worker_id not in first_progress and _trial_has_first_progress(args.output_root, trial.trial_id):
            first_progress[worker_id] = time.monotonic() - started
        log.close()
        complete = returncode == 0 and is_completed_trial(args.output_root, trial.trial_id)
        if isinstance(getattr(process, "pid", None), int) and not isinstance(getattr(process, "pid", None), bool):
            # Successful artifacts do not prove that the policy server obeyed
            # trial teardown.  Verify the group is gone before the next wave.
            _shutdown_worker_process_group(process, grace_seconds=args.terminate_grace_seconds)
        if complete:
            completed += 1
        else:
            failed += 1
        worker_failures.append({
            "worker_id": worker_id,
            "trial_id": trial.trial_id,
            "classes": list(_failure_classes(log_path, returncode=returncode, progressed=worker_id in first_progress)),
        })
    elapsed = time.monotonic() - started
    if not collect_telemetry:
        return elapsed, completed, failed
    sample = telemetry_sampler.sample()
    sample.update(policy_telemetry_sampler.sample(final=True))
    telemetry_samples.append(sample)
    return elapsed, completed, failed, first_progress, telemetry_samples, tuple(worker_failures)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    selection = parser.add_mutually_exclusive_group()
    parser.add_argument("--matrix", type=Path, required=True, help="committed canonical public 280-trial JSON")
    selection.add_argument("--historical-control", action="store_true", help="run only the immutable historical twelve-case control")
    selection.add_argument(
        "--public-unseen-tops",
        action="store_true",
        help="run only the canonical 40-trial public-unseen top diagnostic evaluation",
    )
    parser.add_argument(
        "--parity-stage",
        choices=("legacy_server_cpu", "direct_cpu", "server_cpu", "server_cuda", "scale", "scale_cpu"),
        help="record diagnostic/new parity stages, or require proven legacy-server CPU parity before public scale",
    )
    parser.add_argument(
        "--parity-receipt",
        type=Path,
        action="append",
        default=[],
        help="immutable direct/server parity receipt required by --parity-stage scale",
    )
    parser.add_argument("--server-cpu-receipt", type=Path, help="verified server_cpu receipt required by --parity-stage scale_cpu")
    parser.add_argument("--cuda-abort-receipt", type=Path, help="typed 0/12 server_cuda diagnostic abort required by --parity-stage scale_cpu")
    parser.add_argument("--trial-runtime-root", type=Path, help="clean pinned checkout used by scale_cpu trial children")
    parser.add_argument(
        "--cpu-scale-decision",
        choices=(_CPU_SCALE_DECISION,),
        help="fixed explicit authorization for CPU simulation with isolated CUDA policy servers",
    )
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--policy-revision-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--policy-repo", required=True)
    parser.add_argument("--policy-step", type=int, required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--asset-revision", required=True)
    parser.add_argument("--release-assets-root", type=Path, required=True)
    parser.add_argument("--simulator-version", required=True)
    parser.add_argument("--policy-artifact-sha256", required=True)
    parser.add_argument("--image-identity", required=True)
    mode.add_argument("--capacity-sweep")
    parser.add_argument("--strategy", choices=("canonical", "mild", "strong"), default="canonical")
    parser.add_argument("--device", default="cuda:0", help="Isaac simulator device forwarded to every trial")
    parser.add_argument("--policy-device", default="cuda:0", help="physical CUDA device pool for isolated GR00T policy servers")
    parser.add_argument("--execution-mode", choices=("direct", "policy_server"), default="policy_server")
    parser.add_argument("--historical-control-root", type=Path, help="checksummed historical bundle supplying each frozen garment-config")
    parser.add_argument("--legacy-server-cpu-receipt", type=Path, help="newly generated legacy shared-policy reproduction receipt")
    parser.add_argument("--legacy-source-root", type=Path, help="pinned legacy diagnostic source root for scale_cpu evidence")
    parser.add_argument("--groot-root", type=Path, help="pinned materialized GR00T checkout for policy-server children")
    parser.add_argument("--groot-revision", help="pinned GR00T checkout revision for policy-server children")
    parser.add_argument("--groot-python", type=Path, help="Python 3.10 interpreter in the pinned GR00T environment")
    parser.add_argument("--policy-server-readiness-timeout", type=_positive_finite_seconds, default=30.0)
    parser.add_argument("--policy-server-request-timeout", type=_positive_finite_seconds, default=2.5)
    parser.add_argument("--policy-server-termination-grace", type=_positive_finite_seconds, default=5.0)
    parser.add_argument("--trials-per-worker", type=int, default=1)
    mode.add_argument(
        "--workers",
        type=_authorized_production_worker_count,
        help="run every pending trial in finite waves of this many isolated GPU workers",
    )
    parser.add_argument("--historical-control-workers", type=lambda value: 4 if value == "4" else (_ for _ in ()).throw(argparse.ArgumentTypeError("historical control workers must be exactly 4")))
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--worker-timeout-seconds", type=_positive_finite_seconds, default=1800.0)
    parser.add_argument("--terminate-grace-seconds", type=_positive_finite_seconds, default=5.0)
    parser.add_argument("--max-inference-latency-seconds", type=_positive_finite_seconds, default=0.5)
    parser.add_argument("--max-inference-queue-depth", type=_positive_int, default=16)
    parser.add_argument(
        "--early-abort-completed-trials",
        type=int,
        default=12,
        help="fail closed after 12..24 completed trials with zero official success or visible contact",
    )
    parser.add_argument(
        "--minimum-reset-uniqueness-ratio",
        type=float,
        default=1.0,
        help="minimum distinct canonical reset-hash fraction required for a completed campaign",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_campaign(
    args: argparse.Namespace,
    *,
    runtime_preflight: Callable[[], object] | None = None,
) -> dict[str, object]:
    workers = getattr(args, "workers", None)
    historical_workers = getattr(args, "historical_control_workers", None)
    if historical_workers is not None:
        if workers is not None:
            raise ValueError("choose either --workers or --historical-control-workers")
        workers = historical_workers
        args.workers = workers
    if (
        args.trials_per_worker <= 0
        or args.worker_timeout_seconds <= 0
        or not math.isfinite(args.worker_timeout_seconds)
        or not math.isfinite(args.terminate_grace_seconds)
        or args.terminate_grace_seconds <= 0
        or not math.isfinite(args.max_inference_latency_seconds)
        or args.max_inference_latency_seconds <= 0
        or args.max_inference_queue_depth <= 0
        or not isinstance(args.early_abort_completed_trials, int)
        or not 12 <= args.early_abort_completed_trials <= 24
        or not isinstance(args.minimum_reset_uniqueness_ratio, (int, float))
        or not math.isfinite(args.minimum_reset_uniqueness_ratio)
        or not 0.0 < args.minimum_reset_uniqueness_ratio <= 1.0
        or (workers is not None and (not isinstance(workers, int) or isinstance(workers, bool) or not 1 <= workers <= 4))
    ):
        raise ValueError("production workers must be between 1 and 4")
    if not args.dry_run and args.parity_stage == "direct_cpu":
        raise ValueError("direct_cpu is unsupported before campaign launch: Isaac Python 3.11 cannot directly load pinned GR00T Python 3.10")
    if not args.dry_run and getattr(args, "parity_stage", None) != "legacy_server_cpu":
        (runtime_preflight or require_isaac_sim_5_1_runtime)()
    matrix = load_public_matrix(args.matrix)
    if args.historical_control and args.public_unseen_tops:
        raise ValueError("cannot combine --historical-control and --public-unseen-tops")
    if args.parity_stage == "legacy_server_cpu":
        if args.legacy_server_cpu_receipt is None:
            raise ValueError("legacy_server_cpu requires a newly generated --legacy-server-cpu-receipt")
        legacy_receipt = _validate_legacy_shared_policy_receipt(args.legacy_server_cpu_receipt)
        args.output_root.mkdir(parents=True, exist_ok=True)
        report = {
            "schema_version": 1,
            "selection": {"kind": "legacy_server_cpu_receipt_import", "parity_stage": "legacy_server_cpu"},
            "parity_receipt": str(args.legacy_server_cpu_receipt),
            "official_successes": legacy_receipt["official_successes"],
            "backend": legacy_receipt["backend"],
        }
        _write_json_atomically(args.output_root / "capacity-report.json", report)
        return report
    if args.public_unseen_tops:
        if historical_workers is not None:
            raise ValueError("public-unseen tops cannot use --historical-control-workers")
        if args.parity_stage is not None:
            raise ValueError("public-unseen tops is a diagnostic evaluation and cannot declare a parity or scale stage")
        if args.execution_mode != "policy_server" or args.device != "cpu" or workers != 4:
            raise ValueError("public-unseen tops requires --execution-mode policy_server, --device cpu, and exactly --workers 4")
    if args.parity_stage in {"direct_cpu", "server_cpu", "server_cuda"} and not args.historical_control:
        raise ValueError("a parity stage must use --historical-control")
    if args.parity_stage in {"direct_cpu", "server_cpu", "server_cuda"} and args.historical_control_root is None:
        raise ValueError("a historical-control parity stage requires --historical-control-root with frozen per-trial configs")
    if args.parity_stage in {"direct_cpu", "server_cpu", "server_cuda"}:
        _validate_historical_control_bundle(args.historical_control_root, historical_control_trials())
    expected_stage_mode = {
        "direct_cpu": ("direct", "cpu"),
        "server_cpu": ("policy_server", "cpu"),
        "server_cuda": ("policy_server", None),
    }
    if args.parity_stage in expected_stage_mode:
        required_mode, required_device = expected_stage_mode[args.parity_stage]
        if args.execution_mode != required_mode or (required_device is not None and args.device != required_device):
            raise ValueError(f"{args.parity_stage} requires --execution-mode {required_mode} and --device {required_device or 'cuda:<physical GPU>'}")
        if args.parity_stage == "server_cuda" and _cuda_device_index(args.device) is None:
            raise ValueError("server_cuda requires --device cuda:<physical GPU>")
        if args.parity_stage == "direct_cpu" and workers != 4:
            raise ValueError("direct_cpu diagnostic requires exactly four CPU workers")
    if args.parity_stage == "scale":
        if args.historical_control or args.public_unseen_tops:
            raise ValueError("scale must retain the canonical public 280-trial matrix")
        if args.minimum_reset_uniqueness_ratio != 1.0:
            raise ValueError("scale requires 100% distinct canonical reset hashes")
        _require_scale_parity(args.parity_receipt)
    if args.parity_stage == "scale_cpu":
        if args.historical_control or args.public_unseen_tops:
            raise ValueError("scale_cpu must retain the canonical public 280-trial matrix")
        if args.trial_runtime_root is None:
            raise ValueError("scale_cpu requires --trial-runtime-root")
        if args.legacy_source_root is None:
            raise ValueError("scale_cpu requires --legacy-source-root")
    elif args.parity_stage not in {"scale", "scale_cpu"} and not args.dry_run and not args.historical_control and not args.public_unseen_tops and not args.capacity_sweep:
        raise ValueError("canonical public 280 execution requires --parity-stage scale")
    if not args.dry_run:
        if args.execution_mode == "policy_server":
            required_server_values = (args.groot_root, args.groot_revision, args.groot_python)
            if any(value is None for value in required_server_values):
                raise ValueError("policy-server campaign execution requires pinned GR00T policy-server arguments")
            if not re.fullmatch(r"[0-9a-f]{40}", args.groot_revision):
                raise ValueError("GR00T revision must be a pinned 40-character SHA")
            policy_index = _cuda_device_index(args.policy_device)
            if policy_index is None or args.policy_device != f"cuda:{policy_index}":
                raise ValueError("policy-server campaign requires --policy-device cuda:<physical GPU>")
        device_index = _cuda_device_index(args.device)
        if device_index is not None and args.device != f"cuda:{device_index}":
            raise ValueError("campaign CUDA execution requires --device cuda:<physical GPU>")
    with _campaign_supervisor_lease(args.output_root):
        cpu_scale_authorization = None
        top40_evaluation_invocation = None
        if args.parity_stage == "scale_cpu":
            cpu_scale_authorization = _require_cpu_scale_authorization(args, matrix)
        if args.public_unseen_tops and not args.dry_run:
            top40_evaluation_invocation = _top40_evaluation_invocation(args, matrix, selected_trials(args, matrix))
            _verify_or_write_top40_evaluation_invocation(args.output_root, top40_evaluation_invocation)
        return _run_campaign_under_supervisor(
            args, matrix, cpu_scale_authorization=cpu_scale_authorization,
            top40_evaluation_invocation=top40_evaluation_invocation,
        )


def _run_campaign_under_supervisor(
    args: argparse.Namespace,
    matrix,
    *,
    cpu_scale_authorization: dict[str, object] | None = None,
    top40_evaluation_invocation: dict[str, object] | None = None,
) -> dict[str, object]:
    if getattr(args, "historical_control", False) and getattr(args, "public_unseen_tops", False):
        raise ValueError("cannot combine --historical-control and --public-unseen-tops")
    trials = selected_trials(args, matrix)
    args.output_root.mkdir(parents=True, exist_ok=True)
    state = CampaignState(args.output_root, tuple(trial.trial_id for trial in trials))
    by_id = {trial.trial_id: trial for trial in trials}
    if args.parity_stage == "scale_cpu" and cpu_scale_authorization is not None:
        _validate_scale_cpu_production_output(args, state, matrix, cpu_scale_authorization)
    if getattr(args, "public_unseen_tops", False) and top40_evaluation_invocation is not None:
        _validate_top40_evaluation_output(args, state, top40_evaluation_invocation, trials)
    pending = pending_trial_ids(state)
    if args.parity_stage == "scale_cpu":
        canary_ids = state.trial_ids[:12]
        _run_scale_cpu_canary(
            args, state=state, by_id=by_id, authorization=cpu_scale_authorization,
        )
        # Terminal canary failures are frozen as first-attempt evidence, then
        # remain pending for the real 280/280 close after the pass decision.
        pending = pending_trial_ids(state)
    records: list[dict[str, object]] = []
    production_failure: str | None = None
    sequential_failure: str | None = None
    sequential_terminal_failure: str | None = None
    invocation_id = uuid4().hex
    checkpoint: dict[str, object] = {
        "schema_version": 1,
        "invocation_id": invocation_id,
        "mode": "capacity_sweep" if args.capacity_sweep else "production" if args.workers is not None else "sequential",
        "status": "running",
        "pending_before": list(pending),
        "waves": [],
    }
    _write_invocation_checkpoint(args.output_root, invocation_id, checkpoint)
    if not args.dry_run and not args.capacity_sweep:
        if args.workers is not None:
            if pending:
                gpu_indices = _worker_gpu_indices(args, args.workers)
                for wave_number, offset in enumerate(range(0, len(pending), args.workers), start=1):
                    wave_trial_ids = pending[offset:offset + args.workers]
                    assignments = tuple(
                        (worker_id, by_id[trial_id])
                        for worker_id, trial_id in enumerate(wave_trial_ids, start=1)
                    )
                    wave_gpu_indices = gpu_indices[:len(assignments)]
                    checkpoint_wave = {
                        "mode": "production",
                        "wave": wave_number,
                        "workers": len(assignments),
                        "trial_ids": list(wave_trial_ids),
                        "scheduled_trial_ids": list(wave_trial_ids),
                        "launched_trial_ids": [],
                        "gpu_indices": list(wave_gpu_indices),
                        "status": "started",
                    }
                    checkpoint["waves"].append(checkpoint_wave)
                    _write_invocation_checkpoint(args.output_root, invocation_id, checkpoint)
                    try:
                        elapsed, completed, failed = _run_worker_group(
                            args,
                            assignments,
                            gpu_indices=wave_gpu_indices,
                        )
                    except BaseException as error:
                        scheduled_trial_ids, launched_trial_ids = _launch_accounting_from_error(
                            error, wave_trial_ids,
                        )
                        if not isinstance(error, Exception):
                            checkpoint_wave.update({
                                "status": "interrupted", "detail": str(error),
                                "scheduled_trial_ids": scheduled_trial_ids,
                                "launched_trial_ids": launched_trial_ids,
                            })
                            checkpoint.update({"status": "interrupted", "error_type": type(error).__name__})
                            _write_invocation_checkpoint(args.output_root, invocation_id, checkpoint)
                            raise
                        checkpoint_wave.update({
                            "status": "failed", "detail": str(error),
                            "scheduled_trial_ids": scheduled_trial_ids,
                            "launched_trial_ids": launched_trial_ids,
                        })
                        _write_invocation_checkpoint(args.output_root, invocation_id, checkpoint)
                        records.append({
                            "mode": "production",
                            "wave": wave_number,
                            "workers": len(assignments),
                            "trial_ids": list(wave_trial_ids),
                            "scheduled_trial_ids": scheduled_trial_ids,
                            "launched_trial_ids": launched_trial_ids,
                            "gpu_indices": list(wave_gpu_indices),
                            "status": "launch_error",
                            "detail": str(error),
                        })
                        production_failure = f"production wave {wave_number} failed: {error}"
                        break
                    checkpoint_wave.update({
                        "status": "terminal",
                        "launched_trial_ids": list(wave_trial_ids),
                        "elapsed_seconds": elapsed,
                        "completed_trials": completed,
                        "failed_trials": failed,
                    })
                    _write_invocation_checkpoint(args.output_root, invocation_id, checkpoint)
                    records.append({
                        "mode": "production",
                        "wave": wave_number,
                        "workers": len(assignments),
                        "trial_ids": list(wave_trial_ids),
                        "scheduled_trial_ids": list(wave_trial_ids),
                        "launched_trial_ids": list(wave_trial_ids),
                        "gpu_indices": list(wave_gpu_indices),
                        "elapsed_seconds": elapsed,
                        "completed_trials": completed,
                        "failed_trials": failed,
                    })
                    # Terminal trial failures remain attributable attempts.
                    # Continue finite waves until the abort cohort can make
                    # its explicit zero-success/contact decision.
                    abort_receipt = None if args.parity_stage == "scale_cpu" else _abort_after_first_completed_cohort(
                        args, trial_ids=_terminal_attempted_trial_ids(records), invocation_id=invocation_id,
                    )
                    if abort_receipt is not None:
                        checkpoint_wave["abort_receipt"] = abort_receipt
                        checkpoint.update({"status": "aborted", "abort_receipt": abort_receipt})
                        _write_invocation_checkpoint(args.output_root, invocation_id, checkpoint)
                        production_failure = f"campaign aborted: {abort_receipt['reason']}"
                        break
        else:
            for worker_id, trial_id in enumerate(pending, start=1):
                if (worker_id - 1) >= args.trials_per_worker and (
                    sequential_terminal_failure is None or (worker_id - 1) >= args.early_abort_completed_trials
                ):
                    break
                checkpoint_wave = {
                    "mode": "sequential",
                    "worker_id": worker_id,
                    "trial_ids": [trial_id],
                    "status": "started",
                }
                checkpoint["waves"].append(checkpoint_wave)
                _write_invocation_checkpoint(args.output_root, invocation_id, checkpoint)
                try:
                    returncode = _run_one_worker(args, worker_id=worker_id, trial=by_id[trial_id])
                except BaseException as error:
                    checkpoint_wave.update({"status": "interrupted", "detail": str(error)})
                    checkpoint.update({"status": "interrupted", "error_type": type(error).__name__})
                    _write_invocation_checkpoint(args.output_root, invocation_id, checkpoint)
                    raise
                complete = returncode == 0 and is_completed_trial(args.output_root, trial_id)
                checkpoint_wave.update({"status": "terminal" if complete else "failed", "returncode": returncode})
                _write_invocation_checkpoint(args.output_root, invocation_id, checkpoint)
                records.append({"worker_id": worker_id, "trial_id": trial_id, "returncode": returncode, "mode": "sequential"})
                abort_receipt = None if args.parity_stage == "scale_cpu" else _abort_after_first_completed_cohort(
                    args, trial_ids=_terminal_attempted_trial_ids(records), invocation_id=invocation_id,
                )
                if abort_receipt is not None:
                    checkpoint_wave["abort_receipt"] = abort_receipt
                    checkpoint.update({"status": "aborted", "abort_receipt": abort_receipt})
                    _write_invocation_checkpoint(args.output_root, invocation_id, checkpoint)
                    sequential_failure = f"campaign aborted: {abort_receipt['reason']}"
                    break
                # A terminal failed trial is still an attributable attempt.
                # Keep launching this finite cohort so the abort gate can
                # decide from 12--24 terminal attempts rather than silently
                # stopping after the first failed episode.
                if not complete:
                    sequential_terminal_failure = f"sequential worker {worker_id} failed: returncode={returncode}"
    else:
        records = [{"trial_id": trial_id, "command": _trial_command(args, by_id[trial_id])} for trial_id in pending]

    if args.capacity_sweep and not args.dry_run:
        # A four-GPU sweep is made exclusively of documented 1/2/4 waves.
        # Do not consume a hidden sequential pilot before its 1-worker wave.
        capacity_pending = list(pending)
    elif args.dry_run:
        pending_after = pending
    else:
        pending_after = pending_trial_ids(state)
    if (
        args.workers is not None
        and not args.dry_run
        and not args.capacity_sweep
        and production_failure is None
        and pending_after
    ):
        production_failure = (
            f"production terminal-incomplete trials remain after finite waves: {len(pending_after)}"
        )

    report: dict[str, object] = {
        "schema_version": 1,
        "matrix": {
            "schema_version": matrix.schema_version,
            "sha256": matrix_sha256(matrix),
            "trial_count": len(matrix.trials),
            "training_holdouts": list(matrix.training_holdouts),
        },
        "selection": _selection_metadata(args, trials),
        "pending_before": list(pending),
        "workers": records,
        "completed_after": [],
        "paths": {
            "raw_episodes": str(args.output_root / "raw"),
            "worker_logs": str(args.output_root / "workers"),
            "capacity_report": str(args.output_root / "capacity-report.json"),
        },
    }
    if args.parity_stage == "scale_cpu":
        report["scale_cpu_authorization"] = {
            "authorization_path": str(args.output_root / _CPU_SCALE_AUTHORIZATION),
            "canary_receipt_path": str(args.output_root / _CPU_SCALE_CANARY_RECEIPT),
            "evidence": cpu_scale_authorization.get("evidence") if isinstance(cpu_scale_authorization, dict) else None,
        }
    if args.workers is not None and not args.dry_run and not args.capacity_sweep:
        report["production"] = {
            "workers": args.workers,
            "status": "failed" if production_failure else "completed",
            "waves": len(records),
        }
    if args.capacity_sweep and not args.dry_run:
        counts = _validate_sweep(args.capacity_sweep)
        samples: list[CapacitySample] = []
        capacity_records: list[dict[str, object]] = []
        for count in counts:
            assignments = tuple((index + 1, by_id[trial_id]) for index, trial_id in enumerate(capacity_pending[:count]))
            if len(assignments) != count:
                capacity_records.append({"workers": count, "status": "skipped", "reason": "insufficient_pending_trials"})
                break
            try:
                gpu_indices = _worker_gpu_indices(args, count)
            except ValueError as error:
                capacity_records.append({
                    "workers": count,
                    "trial_ids": [trial.trial_id for _, trial in assignments],
                    "status": "skipped",
                    "reason": "unsupported_gpu_oversubscription",
                    "detail": str(error),
                })
                break
            capacity_pending = capacity_pending[count:]
            checkpoint_wave = {
                "mode": "capacity_sweep",
                "workers": count,
                "trial_ids": [trial.trial_id for _, trial in assignments],
                "scheduled_trial_ids": [trial.trial_id for _, trial in assignments],
                "launched_trial_ids": [],
                "gpu_indices": list(gpu_indices),
                "status": "started",
            }
            checkpoint["waves"].append(checkpoint_wave)
            _write_invocation_checkpoint(args.output_root, invocation_id, checkpoint)
            try:
                result = _run_worker_group(args, assignments, gpu_indices=gpu_indices, collect_telemetry=True)
            except BaseException as error:
                checkpoint_status = "failed" if isinstance(error, Exception) else "interrupted"
                scheduled_trial_ids, launched_trial_ids = _launch_accounting_from_error(
                    error,
                    [trial.trial_id for _, trial in assignments],
                )
                checkpoint_wave.update({
                    "status": checkpoint_status,
                    "detail": str(error),
                    "scheduled_trial_ids": scheduled_trial_ids,
                    "launched_trial_ids": launched_trial_ids,
                })
                checkpoint.update({"status": checkpoint_status, "error_type": type(error).__name__})
                _write_invocation_checkpoint(args.output_root, invocation_id, checkpoint)
                raise
            elapsed, completed, failed = result[:3]
            if len(result) == 3:
                first_progress, telemetry_samples, worker_failures = {}, [], ()
            else:
                _, _, _, first_progress, telemetry_samples, worker_failures = result
            if telemetry_samples:
                ram_margin = min(float(item["host_ram_margin"]) for item in telemetry_samples)
                combined_vram_margin = min(
                    float(item["combined_vram_margin"] if "combined_vram_margin" in item else item["inference_vram_margin"])
                    for item in telemetry_samples
                )
                peak_ram = max((item["peak_host_ram_bytes"] for item in telemetry_samples if item["peak_host_ram_bytes"] is not None), default=None)
                peak_vram = max((item["peak_vram_bytes"] for item in telemetry_samples if item["peak_vram_bytes"] is not None), default=None)
                max_run_queue = max((item["run_queue"] for item in telemetry_samples if item["run_queue"] is not None), default=None)
                max_cpu_utilization = max((item["cpu_utilization"] for item in telemetry_samples if item["cpu_utilization"] is not None), default=None)
                max_inference_latency = max((item["inference_latency_seconds"] for item in telemetry_samples if item["inference_latency_seconds"] is not None), default=None)
                max_inference_queue_depth = max((item["inference_queue_depth"] for item in telemetry_samples if item["inference_queue_depth"] is not None), default=None)
                policy_records_by_key: dict[tuple[object, object], dict[str, object]] = {}
                for item in telemetry_samples:
                    for record in item.get("policy_evidence_records", ()):
                        policy_records_by_key.setdefault(
                            (record["worker_id"], record["failure_class"]),
                            record,
                        )
                policy_evidence_records = tuple(policy_records_by_key.values())
                policy_evidence_failures = tuple(
                    record["failure_class"] for record in policy_evidence_records
                )
            else:
                ram_margin, combined_vram_margin, _ = _resource_margins(gpu_indices)
                peak_ram = peak_vram = max_run_queue = max_cpu_utilization = max_inference_latency = max_inference_queue_depth = None
                policy_evidence_records = ({"worker_id": worker_id, "failure_class": "policy_telemetry_missing"} for worker_id, _ in assignments)
                policy_evidence_records = tuple(policy_evidence_records)
                policy_evidence_failures = tuple(
                    record["failure_class"] for record in policy_evidence_records
                )
            attributed_worker_failures: dict[int, dict[str, object]] = {
                worker_id: {"worker_id": worker_id, "trial_id": trial.trial_id, "classes": []}
                for worker_id, trial in assignments
            }
            for worker_failure in worker_failures:
                worker_id = worker_failure.get("worker_id")
                if worker_id not in attributed_worker_failures:
                    continue
                classes = attributed_worker_failures[worker_id]["classes"]
                assert isinstance(classes, list)
                for failure_class in worker_failure.get("classes", ()):
                    if failure_class not in classes:
                        classes.append(failure_class)
            for record in policy_evidence_records:
                worker_id = record["worker_id"]
                failure_class = record["failure_class"]
                worker_failure = attributed_worker_failures.get(worker_id)
                if worker_failure is None:
                    continue
                classes = worker_failure["classes"]
                assert isinstance(classes, list)
                if failure_class not in classes:
                    classes.append(failure_class)
            worker_failures = tuple(attributed_worker_failures.values())
            sample = CapacitySample(
                # Isaac and the colocated GR00T service share each assigned GPU;
                # use their observed headroom once, not as two fake resources.
                count, elapsed, completed, failed, combined_vram_margin, 1.0, ram_margin,
                first_progress_workers=len(first_progress) if first_progress or telemetry_samples else None,
                stale_ipc_count=sum(
                    1
                    for worker_failure in worker_failures
                    if "stale_ipc" in worker_failure.get("classes", ())
                ),
                peak_host_ram_bytes=peak_ram, peak_vram_bytes=peak_vram, cpu_utilization=max_cpu_utilization,
                run_queue=max_run_queue, inference_latency_seconds=max_inference_latency,
                inference_queue_depth=max_inference_queue_depth,
                policy_evidence_failures=policy_evidence_failures,
                failure_classes=tuple(
                    failure_class
                    for worker_failure in worker_failures
                    for failure_class in worker_failure.get("classes", ())
                ),
            )
            samples.append(sample)
            failure_counts: dict[str, int] = {}
            for worker_failure in worker_failures:
                for failure_class in worker_failure.get("classes", ()):
                    failure_counts[failure_class] = failure_counts.get(failure_class, 0) + 1
            capacity_records.append({
                "workers": count, "trial_ids": [trial.trial_id for _, trial in assignments], "gpu_indices": list(gpu_indices),
                "scheduled_trial_ids": [trial.trial_id for _, trial in assignments],
                "launched_trial_ids": [trial.trial_id for _, trial in assignments],
                "elapsed_seconds": elapsed, "completed_trials": completed, "failed_trials": failed,
                "first_progress_seconds": {str(worker): seconds for worker, seconds in first_progress.items()},
                "host_ram_margin": ram_margin, "combined_vram_margin": combined_vram_margin,
                "peak_host_ram_bytes": peak_ram,
                "peak_vram_bytes": peak_vram, "max_cpu_utilization": max_cpu_utilization, "max_run_queue": max_run_queue,
                "inference_latency_seconds": max_inference_latency,
                "inference_queue_depth": max_inference_queue_depth,
                "policy_evidence_failures": list(policy_evidence_failures),
                "policy_evidence_records": list(policy_evidence_records),
                "worker_failures": list(worker_failures),
                "failure_counts": failure_counts,
            })
            checkpoint_wave.update({
                "status": "terminal",
                "launched_trial_ids": [trial.trial_id for _, trial in assignments],
                "elapsed_seconds": elapsed,
                "completed_trials": completed,
                "failed_trials": failed,
            })
            _write_invocation_checkpoint(args.output_root, invocation_id, checkpoint)
            if choose_worker_count(
                samples,
                max_inference_latency_seconds=args.max_inference_latency_seconds,
                max_inference_queue_depth=args.max_inference_queue_depth,
            ).accepted_workers != count:
                break
        decision = choose_worker_count(
            samples,
            max_inference_latency_seconds=args.max_inference_latency_seconds,
            max_inference_queue_depth=args.max_inference_queue_depth,
        )
        if not samples:
            decision = CapacityDecision(0, {counts[0]: ("no_valid_capacity_sample",)})
        report["capacity"] = {
            "requested": list(counts),
            "max_inference_latency_seconds": args.max_inference_latency_seconds,
            "max_inference_queue_depth": args.max_inference_queue_depth,
            "samples": capacity_records,
            "accepted_workers": decision.accepted_workers,
            "rejected": decision.rejected,
        }
        pending_after = pending_trial_ids(state)
    elif args.capacity_sweep:
        report["capacity"] = {"requested": list(_validate_sweep(args.capacity_sweep)), "status": "dry_run_no_processes"}
    pending_after_set = set(pending_after)
    report["completed_after"] = [trial_id for trial_id in state.trial_ids if trial_id not in pending_after_set]
    capacity = report.get("capacity", {})
    capacity_samples = capacity.get("samples", []) if isinstance(capacity, dict) else []
    wave_trial_ids = [
        record.get("launched_trial_ids", record["trial_ids"])
        for record in capacity_samples
        if isinstance(record, dict) and "trial_ids" in record and record.get("status") != "skipped"
    ]
    sequential_trial_ids = [
        record["trial_id"] for record in records
        if record.get("mode") == "sequential" and "trial_id" in record
    ]
    production_wave_trial_ids = [
        record.get("launched_trial_ids", record["trial_ids"]) for record in records
        if record.get("mode") == "production" and "trial_ids" in record
    ]
    attempted = set(sequential_trial_ids)
    for wave in wave_trial_ids:
        attempted.update(wave)
    for wave in production_wave_trial_ids:
        attempted.update(wave)
    report["episode_accounting"] = {
        "sequential_trial_ids": sequential_trial_ids,
        "capacity_wave_trial_ids": wave_trial_ids,
        "production_wave_trial_ids": production_wave_trial_ids,
        "attempt_count": len(sequential_trial_ids) + sum(len(wave) for wave in wave_trial_ids) + sum(len(wave) for wave in production_wave_trial_ids),
        "attempted_unique_trial_ids": sorted(attempted),
    }
    if sequential_failure is None and sequential_terminal_failure is not None:
        sequential_failure = sequential_terminal_failure
    campaign_failure = production_failure or sequential_failure
    if not args.dry_run and not pending_after and campaign_failure is None and (args.output_root / "raw").is_dir():
        if args.parity_stage == "scale_cpu" and cpu_scale_authorization is not None:
            _validate_scale_cpu_production_output(args, state, matrix, cpu_scale_authorization)
        evidence = _attempted_gate_evidence(args.output_root, state.trial_ids)
        diversity = assess_reset_diversity(
            evidence,
            minimum_ratio=args.minimum_reset_uniqueness_ratio,
        )
        report["reset_diversity"] = {
            "minimum_ratio": args.minimum_reset_uniqueness_ratio,
            "completed_episodes": diversity.completed_episodes,
            "unique_hashes": diversity.unique_hashes,
            "required_unique_hashes": diversity.required_unique_hashes,
            "missing_hashes": diversity.missing_hashes,
            "passed": diversity.passed,
        }
        if not diversity.passed:
            campaign_failure = "campaign reset diversity gate failed"
    if args.parity_stage in {"direct_cpu", "server_cpu", "server_cuda"} and campaign_failure is None:
        evidence = _attempted_gate_evidence(args.output_root, state.trial_ids)
        if len(evidence) != 12:
            campaign_failure = "parity stage requires twelve completed historical control artifacts"
        elif not _may_emit_parity_receipt(args.parity_stage, sum(item.official_success for item in evidence)):
            # A CUDA diagnostic abort is rejection evidence, never a passing
            # server_cuda receipt regardless of a widened abort threshold.
            campaign_failure = "server_cuda parity requires at least 10 official successes"
        else:
            receipt = {
                "schema_version": 1,
                "parity_stage": args.parity_stage,
                "trial_count": 12,
                "trial_ids": list(state.trial_ids),
                "official_successes": sum(item.official_success for item in evidence),
                "visible_robot_garment_contacts": sum(item.visible_contact for item in evidence),
                "reset_diversity": report.get("reset_diversity"),
                "backend": {
                    "direct_cpu": "direct_diagnostic_cpu",
                    "server_cpu": "policy_server_cpu",
                    "server_cuda": "policy_server_cuda",
                }[args.parity_stage],
                "artifact_root": str(args.output_root),
                "policy_server_receipts": [
                    {
                        "trial_id": trial_id,
                        "path": f"policy-server-receipt-{trial_id}.json",
                        "sha256": _sha256_file(args.output_root / f"policy-server-receipt-{trial_id}.json"),
                    }
                    for trial_id in state.trial_ids
                ] if args.execution_mode == "policy_server" else [],
            }
            receipt_path = args.output_root / f"parity-receipt-{args.parity_stage}.json"
            _write_json_atomically(receipt_path, receipt)
            report["parity_receipt"] = str(receipt_path)
    if args.parity_stage == "scale_cpu" and campaign_failure is None and not pending_after:
        receipts: list[dict[str, object]] = []
        for trial_id in state.trial_ids:
            receipt_path = args.output_root / f"policy-server-receipt-{trial_id}.json"
            receipt = _regular_json(receipt_path, label="scale_cpu final policy-server receipt")
            receipts.append({"trial_id": trial_id, "path": str(receipt_path), "sha256": _sha256_file(receipt_path), "receipt": receipt})
        if len(receipts) != 280:
            raise ValueError("scale_cpu final close requires exactly 280 policy-server receipts")
        report["production_policy_server_receipts"] = receipts
        report["scale_cpu_authorization_full"] = cpu_scale_authorization
        canary_path = args.output_root / _CPU_SCALE_CANARY_RECEIPT
        report["scale_cpu_canary_receipt"] = {"path": str(canary_path), "sha256": _sha256_file(canary_path), "receipt": _regular_json(canary_path, label="scale_cpu canary receipt")}
    if getattr(args, "public_unseen_tops", False) and top40_evaluation_invocation is not None and campaign_failure is None and not pending_after:
        _validate_top40_evaluation_output(args, state, top40_evaluation_invocation, trials)
        report["checkpoint_evaluation"] = _top40_final_bindings(
            args.output_root, state, top40_evaluation_invocation, trials,
        )
    checkpoint["status"] = "failed" if campaign_failure else "completed"
    checkpoint["completed_after"] = report["completed_after"]
    _write_invocation_checkpoint(args.output_root, invocation_id, checkpoint)
    _write_json_atomically(args.output_root / "capacity-report.json", report)
    if campaign_failure:
        raise RuntimeError(campaign_failure)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_campaign(args)
    except ValueError as error:
        print(f"campaign validation error: {error}", file=sys.stderr)
        return 2
    except RuntimeError as error:
        print(f"campaign execution error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CampaignState", "build_legacy_cpu_reference_receipt", "pending_trial_ids", "run_campaign"]
