"""Private immutable publication of a fully audited corrective RFT release."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Mapping

from lehome_train.constants import DEFAULT_DATA_REPO
from lehome_train.flywheel.corrective import (
    APPROVED_PARENT_ARTIFACT_SHA256,
    APPROVED_PARENT_REPOSITORY,
    APPROVED_PARENT_REVISION,
    APPROVED_PARENT_STEP,
    CorrectiveAttemptArtifact,
    CorrectivePublicationBundle,
    TARGET_UNIQUE_SUCCESSES,
    verify_corrective_publication_bundle,
)
from lehome_train.hub import HubTreeEntry, HubTransport, download_files, list_repository_tree, require_access, upload_files
from lehome_train.io import atomic_write_json, canonical_json_bytes, canonical_json_sha256
from lehome_train.models import SyncEntry
from lehome_train.redaction import generate_upload_allowlist


_REMOTE_ROOT = "corrective-rft"


@dataclass(frozen=True, slots=True)
class CorrectivePublicationResult:
    """The evidence that permits disposal only after immutable readback."""

    repository: str
    immutable_revision: str
    remote_prefix: str
    release_id: str
    entries: tuple[SyncEntry, ...]
    disposable: bool


@dataclass(frozen=True, slots=True)
class CorrectiveCanaryPublicationBundle:
    """One canary attempt tied to its scheduled and paid execution evidence."""

    attempt: CorrectiveAttemptArtifact
    canary_manifest_path: str
    canary_manifest_sha256: str
    source_wave_manifest_path: str
    source_wave_manifest_sha256: str
    provider_evidence_path: str
    provider_evidence_sha256: str
    provider_snapshot_path: str
    provider_snapshot_sha256: str
    instance_receipt_path: str
    instance_receipt_sha256: str
    terminal_receipt_path: str
    terminal_receipt_sha256: str
    synced_evidence_root: str
    canary_sha256: str


@dataclass(frozen=True, slots=True)
class CorrectiveCanaryAbortPublicationBundle:
    """Abort-only evidence for a canary that never produced a raw episode."""

    canary_manifest_path: str
    canary_manifest_sha256: str
    source_wave_manifest_path: str
    source_wave_manifest_sha256: str
    provider_evidence_path: str
    provider_evidence_sha256: str
    provider_snapshot_path: str
    provider_snapshot_sha256: str
    instance_receipt_path: str
    instance_receipt_sha256: str
    abort_receipt_path: str
    abort_receipt_sha256: str
    synced_evidence_root: str
    abort_sha256: str


@dataclass(frozen=True, slots=True)
class CorrectiveReleasePublicationBundle:
    """The all-attempt release plus one immutable paid-instance receipt per wave."""

    corrective: CorrectivePublicationBundle
    instance_receipt_paths: Mapping[int, str]
    instance_receipt_sha256s: Mapping[int, str]
    instance_ids: Mapping[int, int]
    release_provenance_sha256: str


def _publish_staged_release(
    staging: Path,
    *,
    release_id: str,
    revision: str,
    transport: HubTransport,
    remote_root: str,
    staging_root: Path,
) -> tuple[str, str, tuple[SyncEntry, ...]]:
    entries = generate_upload_allowlist(staging, _regular_paths(staging))
    prefix = f"{remote_root}/{release_id}"
    immutable_revision = upload_files(
        transport=transport, repository=DEFAULT_DATA_REPO, revision=revision, source=staging,
        entries=entries, remote_prefix=prefix, max_attempts=1,
    )
    tree = list_repository_tree(
        transport=transport, repository=DEFAULT_DATA_REPO, revision=immutable_revision, max_attempts=1,
    )
    if not _tree_matches(tree, prefix, entries):
        raise ValueError("corrective immutable remote tree does not match the release")
    readback = Path(tempfile.mkdtemp(prefix="lehome-corrective-readback-", dir=staging_root))
    try:
        download_files(
            transport=transport, repository=DEFAULT_DATA_REPO, revision=immutable_revision,
            destination=readback, relative_paths=tuple(item.relative_path for item in entries),
            remote_prefix=prefix, max_attempts=1,
        )
        _verify_readback(readback, entries)
    finally:
        shutil.rmtree(readback, ignore_errors=True)
    return immutable_revision, prefix, entries


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("corrective snapshot JSON has a duplicate field")
        result[key] = value
    return result


def _sha256(path: Path, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"corrective canary {label} must be a regular file")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canary_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object)
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError(f"corrective canary {label} is malformed") from None
    if not isinstance(value, dict):
        raise ValueError(f"corrective canary {label} must be an object")
    return value


def _canary_body(bundle: CorrectiveCanaryPublicationBundle) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "corrective_rft_canary_publication",
        "attempt_id": bundle.attempt.attempt_id,
        "attempt_receipt_sha256": canonical_json_sha256(bundle.attempt.attempt_receipt),
        "episode_manifest_sha256": bundle.attempt.episode_manifest_sha256,
        "policy_receipt_sha256": bundle.attempt.policy_receipt_sha256,
        "canary_manifest_sha256": bundle.canary_manifest_sha256,
        "source_wave_manifest_sha256": bundle.source_wave_manifest_sha256,
        "provider_evidence_sha256": bundle.provider_evidence_sha256,
        "provider_snapshot_sha256": bundle.provider_snapshot_sha256,
        "instance_receipt_sha256": bundle.instance_receipt_sha256,
        "terminal_receipt_sha256": bundle.terminal_receipt_sha256,
        "synced_evidence_sha256": _evidence_sha256(Path(bundle.synced_evidence_root)),
    }


def _abort_body(bundle: CorrectiveCanaryAbortPublicationBundle) -> dict[str, object]:
    return {
        "schema_version": 1, "kind": "corrective_rft_canary_abort_publication",
        "canary_manifest_sha256": bundle.canary_manifest_sha256,
        "source_wave_manifest_sha256": bundle.source_wave_manifest_sha256,
        "provider_evidence_sha256": bundle.provider_evidence_sha256,
        "provider_snapshot_sha256": bundle.provider_snapshot_sha256,
        "instance_receipt_sha256": bundle.instance_receipt_sha256,
        "abort_receipt_sha256": bundle.abort_receipt_sha256,
        "synced_evidence_sha256": _evidence_sha256(Path(bundle.synced_evidence_root)),
    }


def _evidence_sha256(root: Path) -> str:
    return canonical_json_sha256({
        relative: _sha256(root / relative, "synchronized evidence")
        for relative in _regular_paths(root)
    })


def build_corrective_canary_publication_bundle(
    *,
    attempt_receipt: Mapping[str, object], raw_episode_root: str | Path,
    policy_receipt_path: str | Path, canary_manifest_path: str | Path,
    source_wave_manifest_path: str | Path, provider_evidence_path: str | Path,
    provider_snapshot_path: str | Path, instance_receipt_path: str | Path,
    terminal_receipt_path: str | Path, synced_evidence_root: str | Path,
) -> CorrectiveCanaryPublicationBundle:
    """Construct a typed success bundle from exact lifecycle and campaign files."""

    raw, policy = Path(raw_episode_root), Path(policy_receipt_path)
    attempt_id = attempt_receipt.get("attempt_id") if isinstance(attempt_receipt, Mapping) else None
    if not isinstance(attempt_id, str) or not attempt_id:
        raise ValueError("corrective canary attempt receipt is invalid")
    artifact = CorrectiveAttemptArtifact(
        attempt_id, dict(attempt_receipt), str(raw), _sha256(raw / "SHA256SUMS.json", "raw manifest"),
        str(policy), _sha256(policy, "policy receipt"),
    )
    paths = tuple(Path(item) for item in (
        canary_manifest_path, source_wave_manifest_path, provider_evidence_path,
        provider_snapshot_path, instance_receipt_path, terminal_receipt_path,
    ))
    provisional = CorrectiveCanaryPublicationBundle(
        artifact, *(str(path) for path in paths[:1]), _sha256(paths[0], "manifest"),
        *(str(path) for path in paths[1:2]), _sha256(paths[1], "source wave manifest"),
        *(str(path) for path in paths[2:3]), _sha256(paths[2], "provider evidence"),
        *(str(path) for path in paths[3:4]), _sha256(paths[3], "provider snapshot"),
        *(str(path) for path in paths[4:5]), _sha256(paths[4], "instance receipt"),
        *(str(path) for path in paths[5:6]), _sha256(paths[5], "terminal receipt"),
        str(synced_evidence_root), "0" * 64,
    )
    bundle = CorrectiveCanaryPublicationBundle(
        provisional.attempt, provisional.canary_manifest_path, provisional.canary_manifest_sha256,
        provisional.source_wave_manifest_path, provisional.source_wave_manifest_sha256,
        provisional.provider_evidence_path, provisional.provider_evidence_sha256,
        provisional.provider_snapshot_path, provisional.provider_snapshot_sha256,
        provisional.instance_receipt_path, provisional.instance_receipt_sha256,
        provisional.terminal_receipt_path, provisional.terminal_receipt_sha256,
        provisional.synced_evidence_root, canonical_json_sha256(_canary_body(provisional)),
    )
    verify_corrective_canary_publication_bundle(bundle)
    return bundle


def build_corrective_canary_abort_publication_bundle(
    *,
    canary_manifest_path: str | Path, source_wave_manifest_path: str | Path,
    provider_evidence_path: str | Path, provider_snapshot_path: str | Path,
    instance_receipt_path: str | Path, abort_receipt_path: str | Path,
    synced_evidence_root: str | Path,
) -> CorrectiveCanaryAbortPublicationBundle:
    """Construct an abort-only bundle without requiring episode or policy files."""

    paths = tuple(Path(item) for item in (
        canary_manifest_path, source_wave_manifest_path, provider_evidence_path,
        provider_snapshot_path, instance_receipt_path, abort_receipt_path,
    ))
    provisional = CorrectiveCanaryAbortPublicationBundle(
        *(str(path) for path in paths[:1]), _sha256(paths[0], "manifest"),
        *(str(path) for path in paths[1:2]), _sha256(paths[1], "source wave manifest"),
        *(str(path) for path in paths[2:3]), _sha256(paths[2], "provider evidence"),
        *(str(path) for path in paths[3:4]), _sha256(paths[3], "provider snapshot"),
        *(str(path) for path in paths[4:5]), _sha256(paths[4], "instance receipt"),
        *(str(path) for path in paths[5:6]), _sha256(paths[5], "abort receipt"),
        str(synced_evidence_root), "0" * 64,
    )
    bundle = CorrectiveCanaryAbortPublicationBundle(
        provisional.canary_manifest_path, provisional.canary_manifest_sha256,
        provisional.source_wave_manifest_path, provisional.source_wave_manifest_sha256,
        provisional.provider_evidence_path, provisional.provider_evidence_sha256,
        provisional.provider_snapshot_path, provisional.provider_snapshot_sha256,
        provisional.instance_receipt_path, provisional.instance_receipt_sha256,
        provisional.abort_receipt_path, provisional.abort_receipt_sha256,
        provisional.synced_evidence_root, canonical_json_sha256(_abort_body(provisional)),
    )
    verify_corrective_canary_abort_publication_bundle(bundle)
    return bundle


def _release_provenance_body(
    corrective: CorrectivePublicationBundle,
    receipts: Mapping[int, str],
    hashes: Mapping[int, str],
    ids: Mapping[int, int],
) -> dict[str, object]:
    return {
        "schema_version": 1, "kind": "corrective_rft_release_instance_provenance",
        "publication_sha256": corrective.publication_sha256,
        "instances": [
            {"wave_index": wave, "instance_id": ids[wave], "receipt_sha256": hashes[wave]}
            for wave in sorted(ids)
        ],
    }


def build_corrective_release_publication_bundle(
    corrective: CorrectivePublicationBundle,
    instance_receipts: Mapping[int, str | Path],
) -> CorrectiveReleasePublicationBundle:
    """Bind every collected wave to the exact private instance that ran it."""

    verify_corrective_publication_bundle(corrective)
    waves = set(corrective.wave_evidence)
    if set(instance_receipts) != waves:
        raise ValueError("corrective release instance receipts must cover every wave exactly")
    paths: dict[int, str] = {}
    hashes: dict[int, str] = {}
    ids: dict[int, int] = {}
    for wave_index, supplied in instance_receipts.items():
        path = Path(supplied)
        digest = _sha256(path, "release instance receipt")
        receipt = _canary_json(path, "release instance receipt")
        evidence = corrective.wave_evidence[wave_index]
        if (
            receipt.get("schema_version") != 1 or receipt.get("kind") != "corrective_vast_instance"
            or receipt.get("wave_index") != wave_index
            or not isinstance(receipt.get("instance_id"), int) or receipt["instance_id"] <= 0
            or receipt.get("provider_evidence_sha256") != canonical_json_sha256(evidence.provider_evidence)
        ):
            raise ValueError("corrective release instance provenance is stale or cross-bound")
        paths[wave_index], hashes[wave_index], ids[wave_index] = str(path), digest, receipt["instance_id"]
    body = _release_provenance_body(corrective, paths, hashes, ids)
    return CorrectiveReleasePublicationBundle(
        corrective, paths, hashes, ids, canonical_json_sha256(body),
    )


def verify_corrective_release_publication_bundle(
    bundle: CorrectiveReleasePublicationBundle,
) -> CorrectiveReleasePublicationBundle:
    if not isinstance(bundle, CorrectiveReleasePublicationBundle):
        raise ValueError("corrective release publication bundle is invalid")
    rebuilt = build_corrective_release_publication_bundle(bundle.corrective, bundle.instance_receipt_paths)
    if (
        rebuilt.instance_receipt_sha256s != bundle.instance_receipt_sha256s
        or rebuilt.instance_ids != bundle.instance_ids
        or rebuilt.release_provenance_sha256 != bundle.release_provenance_sha256
    ):
        raise ValueError("corrective release instance provenance is stale")
    return rebuilt


def verify_corrective_canary_publication_bundle(
    bundle: CorrectiveCanaryPublicationBundle,
) -> dict[str, object]:
    """Verify one scheduled canary attempt before publication or disposal."""

    if not isinstance(bundle, CorrectiveCanaryPublicationBundle):
        raise ValueError("corrective canary publication bundle is invalid")
    attempt = bundle.attempt
    if not isinstance(attempt, CorrectiveAttemptArtifact):
        raise ValueError("corrective canary attempt artifact is invalid")
    files = {
        "manifest": (Path(bundle.canary_manifest_path), bundle.canary_manifest_sha256),
        "source wave manifest": (Path(bundle.source_wave_manifest_path), bundle.source_wave_manifest_sha256),
        "provider evidence": (Path(bundle.provider_evidence_path), bundle.provider_evidence_sha256),
        "provider snapshot": (Path(bundle.provider_snapshot_path), bundle.provider_snapshot_sha256),
        "instance receipt": (Path(bundle.instance_receipt_path), bundle.instance_receipt_sha256),
        "terminal receipt": (Path(bundle.terminal_receipt_path), bundle.terminal_receipt_sha256),
    }
    for label, (path, expected) in files.items():
        if not isinstance(expected, str) or len(expected) != 64 or _sha256(path, label) != expected:
            raise ValueError(f"corrective canary {label} hash is stale")
    if canonical_json_sha256(attempt.attempt_receipt) != _canary_body(bundle)["attempt_receipt_sha256"]:
        raise ValueError("corrective canary attempt receipt hash is stale")
    manifest = _canary_json(files["manifest"][0], "manifest")
    wave = _canary_json(files["source wave manifest"][0], "source wave manifest")
    provider_evidence = _canary_json(files["provider evidence"][0], "provider evidence")
    instance = _canary_json(files["instance receipt"][0], "instance receipt")
    terminal = _canary_json(files["terminal receipt"][0], "terminal receipt")
    synced_root = Path(bundle.synced_evidence_root)
    synced_sha = _evidence_sha256(synced_root)
    expected_parent = {
        "parent_checkpoint_repository": APPROVED_PARENT_REPOSITORY,
        "parent_checkpoint_revision": APPROVED_PARENT_REVISION,
        "parent_checkpoint_artifact_sha256": APPROVED_PARENT_ARTIFACT_SHA256,
        "parent_checkpoint_step": APPROVED_PARENT_STEP,
    }
    baseline = manifest.get("baseline")
    scheduled = manifest.get("attempt")
    if (
        manifest.get("schema_version") != 1 or manifest.get("kind") != "corrective_rft_canary"
        or manifest.get("episode_count") != 1 or not isinstance(baseline, Mapping)
        or not isinstance(scheduled, Mapping) or manifest.get("attempt") != scheduled
        or any(baseline.get(key) != value for key, value in expected_parent.items())
        or manifest.get("source_wave_sha256") != bundle.source_wave_manifest_sha256
        or wave.get("schema_version") != 1 or wave.get("kind") != "corrective_rft_wave"
        or wave.get("wave_index") != manifest.get("wave_index")
        or wave.get("baseline") != dict(baseline)
        or wave.get("provider") != manifest.get("provider")
        or wave.get("provider_evidence") != provider_evidence
        or not isinstance(wave.get("attempts"), list)
        or sum(1 for item in wave["attempts"] if item == dict(scheduled)) != 1
        or scheduled.get("attempt_id") != attempt.attempt_id
        or scheduled.get("episode_id") != attempt.attempt_receipt.get("episode_id")
        or scheduled.get("wave_index") != attempt.attempt_receipt.get("wave_index")
        or scheduled.get("worker_slot") != attempt.attempt_receipt.get("worker_slot")
        or not isinstance(scheduled.get("command"), list)
        or not all(isinstance(item, str) for item in scheduled["command"])
        or provider_evidence.get("schema_version") != 1
        or provider_evidence.get("kind") != "external_provider_offer_evidence"
        or provider_evidence.get("source_snapshot_sha256") != bundle.provider_snapshot_sha256
        or provider_evidence.get("offer_id") != attempt.attempt_receipt.get("provider", {}).get("offer_id")
        or instance.get("schema_version") != 1 or instance.get("kind") != "corrective_vast_instance"
        or instance.get("wave_index") != manifest.get("wave_index")
        or not isinstance(instance.get("instance_id"), int) or instance["instance_id"] <= 0
        or terminal.get("schema_version") != 1
        or terminal.get("kind") != "corrective_canary_terminal"
        or terminal.get("attempt_id") != attempt.attempt_id
        or terminal.get("instance_id") != instance["instance_id"]
        or terminal.get("canary_manifest_sha256") != bundle.canary_manifest_sha256
        or not isinstance(terminal.get("staged_bundle_sha256"), str)
        or len(terminal["staged_bundle_sha256"]) != 64
        or terminal.get("transport_returncode") != 0
        or terminal.get("raw_manifest_sha256") != attempt.episode_manifest_sha256
        or terminal.get("policy_receipt_sha256") != attempt.policy_receipt_sha256
        or terminal.get("synced_evidence_sha256") != synced_sha
        or Path(attempt.raw_episode_root) != synced_root / "raw" / attempt.attempt_id
        or Path(attempt.policy_receipt_path) != synced_root / f"policy-server-receipt-{attempt.attempt_id}.json"
        or canonical_json_sha256(_canary_body(bundle)) != bundle.canary_sha256
    ):
        raise ValueError("corrective canary provenance is stale or forged")
    return {"instance_id": instance["instance_id"], "attempt": dict(scheduled), "baseline": dict(baseline)}


def verify_corrective_canary_abort_publication_bundle(
    bundle: CorrectiveCanaryAbortPublicationBundle,
) -> dict[str, object]:
    """Verify early-failure evidence without inventing an episode or policy receipt."""

    if not isinstance(bundle, CorrectiveCanaryAbortPublicationBundle):
        raise ValueError("corrective canary abort publication bundle is invalid")
    files = {
        "manifest": (Path(bundle.canary_manifest_path), bundle.canary_manifest_sha256),
        "source wave manifest": (Path(bundle.source_wave_manifest_path), bundle.source_wave_manifest_sha256),
        "provider evidence": (Path(bundle.provider_evidence_path), bundle.provider_evidence_sha256),
        "provider snapshot": (Path(bundle.provider_snapshot_path), bundle.provider_snapshot_sha256),
        "instance receipt": (Path(bundle.instance_receipt_path), bundle.instance_receipt_sha256),
        "abort receipt": (Path(bundle.abort_receipt_path), bundle.abort_receipt_sha256),
    }
    for label, (path, expected) in files.items():
        if not isinstance(expected, str) or len(expected) != 64 or _sha256(path, label) != expected:
            raise ValueError(f"corrective canary {label} hash is stale")
    manifest = _canary_json(files["manifest"][0], "manifest")
    wave = _canary_json(files["source wave manifest"][0], "source wave manifest")
    provider = _canary_json(files["provider evidence"][0], "provider evidence")
    instance = _canary_json(files["instance receipt"][0], "instance receipt")
    abort = _canary_json(files["abort receipt"][0], "abort receipt")
    root = Path(bundle.synced_evidence_root)
    scheduled_attempt_id = manifest.get("attempt", {}).get("attempt_id") if isinstance(manifest.get("attempt"), Mapping) else None
    raw_manifest = root / "raw" / str(scheduled_attempt_id) / "SHA256SUMS.json"
    policy_receipt = root / f"policy-server-receipt-{scheduled_attempt_id}.json"
    if (
        manifest.get("schema_version") != 1 or manifest.get("kind") != "corrective_rft_canary"
        or manifest.get("episode_count") != 1 or not isinstance(manifest.get("attempt"), Mapping)
        or manifest.get("source_wave_sha256") != bundle.source_wave_manifest_sha256
        or wave.get("schema_version") != 1 or wave.get("kind") != "corrective_rft_wave"
        or wave.get("wave_index") != manifest.get("wave_index")
        or wave.get("provider_evidence") != provider
        or not isinstance(wave.get("attempts"), list) or manifest["attempt"] not in wave["attempts"]
        or instance.get("schema_version") != 1 or instance.get("kind") != "corrective_vast_instance"
        or instance.get("wave_index") != manifest.get("wave_index")
        or not isinstance(instance.get("instance_id"), int) or instance["instance_id"] <= 0
        or abort.get("schema_version") != 1 or abort.get("kind") not in {"corrective_canary_abort", "corrective_canary_terminal", "corrective_canary_non_training_abort"}
        or abort.get("attempt_id") != manifest["attempt"].get("attempt_id")
        or abort.get("instance_id") != instance["instance_id"]
        or abort.get("canary_manifest_sha256") != bundle.canary_manifest_sha256
        or not isinstance(abort.get("staged_bundle_sha256"), str) or len(abort["staged_bundle_sha256"]) != 64
        or not isinstance(abort.get("transport_returncode"), int)
        or (
            abort.get("kind") == "corrective_canary_terminal"
            and (
                abort.get("transport_returncode") != 0
                or abort.get("accepted_success") is not False
                or abort.get("outcome") not in {"failure", "timeout", "error"}
            )
        )
        or (
            abort.get("kind") == "corrective_canary_non_training_abort"
            and (
                abort.get("transport_returncode") != 0
                or abort.get("non_training_admitted") is not False
                or not isinstance(abort.get("raw_manifest_sha256"), str)
                or not isinstance(abort.get("policy_receipt_sha256"), str)
                or _sha256(raw_manifest, "non-training raw manifest") != abort["raw_manifest_sha256"]
                or _sha256(policy_receipt, "non-training policy receipt") != abort["policy_receipt_sha256"]
            )
        )
        or abort.get("synced_evidence_sha256") != _evidence_sha256(root)
        or not _regular_paths(root)
        or canonical_json_sha256(_abort_body(bundle)) != bundle.abort_sha256
    ):
        raise ValueError("corrective canary abort provenance is stale or forged")
    return {
        "instance_id": instance["instance_id"], "attempt_id": manifest["attempt"]["attempt_id"],
        "canary_type": "task_failure_canary" if abort["kind"] in {"corrective_canary_terminal", "corrective_canary_non_training_abort"} else "abort_canary",
    }


def publish_private_corrective_canary(
    bundle: CorrectiveCanaryPublicationBundle,
    *,
    revision: str,
    transport: HubTransport,
    disposal_receipt: str | Path,
    staging_root: str | Path | None = None,
) -> CorrectivePublicationResult:
    """Publish one proven-success canary, never a bare attempt artifact."""

    canary = verify_corrective_canary_publication_bundle(bundle)
    attempt = bundle.attempt
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("corrective canary revision must be explicit")
    receipt = Path(disposal_receipt)
    if receipt.exists() or receipt.is_symlink():
        raise FileExistsError("corrective canary disposal receipt must not already exist")
    stage_parent = Path(staging_root) if staging_root is not None else Path(attempt.raw_episode_root).parent
    if not stage_parent.is_dir() or stage_parent.is_symlink():
        raise ValueError("corrective canary staging root must be a materialized directory")
    require_access(transport=transport, repository=DEFAULT_DATA_REPO, read=True, write=True)
    staging = Path(tempfile.mkdtemp(prefix="lehome-corrective-canary-", dir=stage_parent))
    try:
        raw_root = Path(attempt.raw_episode_root)
        try:
            from lehome.flywheel.artifacts import verify_episode_manifest
            episode, _manifest = verify_episode_manifest(raw_root)
        except (ImportError, ValueError):
            raise ValueError("corrective canary raw episode manifest verification failed") from None
        if episode.get("episode_id") != attempt.attempt_receipt.get("episode_id"):
            raise ValueError("corrective canary episode identity is stale")
        if (
            _sha256(raw_root / "SHA256SUMS.json", "raw manifest") != attempt.episode_manifest_sha256
            or _sha256(Path(attempt.policy_receipt_path), "policy receipt") != attempt.policy_receipt_sha256
        ):
            raise ValueError("corrective canary declared artifact hashes are stale")
        for relative in _regular_paths(raw_root):
            _copy(raw_root / relative, staging / "raw" / relative)
        _copy(Path(attempt.policy_receipt_path), staging / "policy-receipt.json")
        policy = _canary_json(Path(attempt.policy_receipt_path), "policy receipt")
        if (
            not attempt.attempt_receipt.get("accepted_success")
            or attempt.attempt_receipt.get("outcome") != "success"
            or episode.get("episode_id") != raw_root.name
            or not isinstance(episode.get("identity"), Mapping)
            or episode["identity"].get("release_stage") != "seen"
            or policy.get("episode_id") != attempt.attempt_receipt["episode_id"]
            or not isinstance(policy.get("command"), list)
            or "--model-path" not in policy["command"]
            or not isinstance(policy.get("model_path"), str)
            or policy["command"][policy["command"].index("--model-path") + 1] != policy["model_path"]
            or any(policy.get(key) != value for key, value in {
                "checkpoint_revision": APPROVED_PARENT_REVISION,
                "checkpoint_digest": APPROVED_PARENT_ARTIFACT_SHA256,
                "code_revision": attempt.attempt_receipt["code_revision"],
                "image_identity": attempt.attempt_receipt["image_identity"],
            }.items())
        ):
            raise ValueError("corrective canary raw or policy provenance is stale")
        (staging / "attempt-receipt.json").write_bytes(canonical_json_bytes(attempt.attempt_receipt))
        for name, path in (
            ("canary-manifest.json", bundle.canary_manifest_path),
            ("source-wave-manifest.json", bundle.source_wave_manifest_path),
            ("provider-evidence.json", bundle.provider_evidence_path),
            ("provider-source-snapshot.json", bundle.provider_snapshot_path),
            ("instance-receipt.json", bundle.instance_receipt_path),
            ("terminal-receipt.json", bundle.terminal_receipt_path),
        ):
            _copy(Path(path), staging / name)
        for relative in _regular_paths(Path(bundle.synced_evidence_root)):
            _copy(Path(bundle.synced_evidence_root) / relative, staging / "synced-evidence" / relative)
        release_id = canonical_json_sha256({
            "schema_version": 1, "kind": "corrective_rft_private_canary",
            "canary_sha256": bundle.canary_sha256,
        })
        immutable_revision, prefix, entries = _publish_staged_release(
            staging, release_id=release_id, revision=revision, transport=transport,
            remote_root="corrective-rft-canary", staging_root=stage_parent,
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    result = CorrectivePublicationResult(DEFAULT_DATA_REPO, immutable_revision, prefix, release_id, entries, True)
    atomic_write_json(receipt, {
        "schema_version": 1, "kind": "corrective_rft_private_canary", "repository": result.repository,
        "immutable_revision": result.immutable_revision, "remote_prefix": result.remote_prefix,
        "canary_type": "success_canary", "attempt_id": attempt.attempt_id,
        "episode_id": attempt.attempt_receipt["episode_id"], "instance_id": canary["instance_id"],
        "canary_sha256": bundle.canary_sha256,
        "entry_count": len(entries), "repository_private": True, "tree_listing_verified": True,
        "fresh_readback_verified": True, "training_admission": False, "disposable": True,
    })
    return result


def publish_private_corrective_canary_abort(
    bundle: CorrectiveCanaryAbortPublicationBundle,
    *,
    revision: str,
    transport: HubTransport,
    disposal_receipt: str | Path,
    staging_root: str | Path | None = None,
) -> CorrectivePublicationResult:
    """Preserve a failed canary's synchronized evidence before authorizing disposal."""

    abort = verify_corrective_canary_abort_publication_bundle(bundle)
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("corrective canary revision must be explicit")
    receipt = Path(disposal_receipt)
    if receipt.exists() or receipt.is_symlink():
        raise FileExistsError("corrective canary disposal receipt must not already exist")
    instance_path = Path(bundle.instance_receipt_path)
    sync_root = Path(bundle.synced_evidence_root)
    stage_parent = Path(staging_root) if staging_root is not None else sync_root.parent
    if not stage_parent.is_dir() or stage_parent.is_symlink():
        raise ValueError("corrective canary staging root must be a materialized directory")
    require_access(transport=transport, repository=DEFAULT_DATA_REPO, read=True, write=True)
    staging = Path(tempfile.mkdtemp(prefix="lehome-corrective-canary-abort-", dir=stage_parent))
    try:
        for name, path in (
            ("canary-manifest.json", bundle.canary_manifest_path),
            ("source-wave-manifest.json", bundle.source_wave_manifest_path),
            ("provider-evidence.json", bundle.provider_evidence_path),
            ("provider-source-snapshot.json", bundle.provider_snapshot_path),
            ("instance-receipt.json", bundle.instance_receipt_path),
            ("abort-receipt.json", bundle.abort_receipt_path),
        ):
            _copy(Path(path), staging / name)
        for relative in _regular_paths(sync_root):
            _copy(sync_root / relative, staging / "synced-evidence" / relative)
        release_id = canonical_json_sha256({
            "schema_version": 1, "kind": "corrective_rft_private_canary_abort",
            "abort_sha256": bundle.abort_sha256,
        })
        immutable_revision, prefix, entries = _publish_staged_release(
            staging, release_id=release_id, revision=revision, transport=transport,
            remote_root="corrective-rft-canary-abort", staging_root=stage_parent,
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    result = CorrectivePublicationResult(DEFAULT_DATA_REPO, immutable_revision, prefix, release_id, entries, True)
    atomic_write_json(receipt, {
        "schema_version": 1, "kind": "corrective_rft_private_canary", "canary_type": abort["canary_type"],
        "repository": result.repository, "immutable_revision": result.immutable_revision,
        "remote_prefix": result.remote_prefix, "attempt_id": abort["attempt_id"],
        "instance_id": abort["instance_id"], "abort_sha256": bundle.abort_sha256,
        "entry_count": len(entries), "repository_private": True, "tree_listing_verified": True,
        "fresh_readback_verified": True, "training_admission": False, "disposable": True,
    })
    return result


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object)
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("corrective materialized snapshot JSON is unavailable or malformed") from None
    if not isinstance(payload, Mapping):
        raise ValueError("corrective materialized snapshot JSON must be an object")
    return payload


def _regular_paths(root: Path) -> tuple[str, ...]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("corrective artifact root must be a materialized directory")
    paths: list[str] = []
    pending = [root]
    while pending:
        current = pending.pop()
        with os.scandir(current) as children:
            for child in children:
                relative = Path(child.path).relative_to(root).as_posix()
                if child.is_symlink():
                    raise ValueError("corrective artifact root must not contain symlinks")
                if child.is_dir(follow_symlinks=False):
                    pending.append(Path(child.path))
                elif child.is_file(follow_symlinks=False):
                    paths.append(relative)
                else:
                    raise ValueError("corrective artifact root contains an unsupported path type")
    if not paths:
        raise ValueError("corrective artifact root is empty")
    return tuple(sorted(paths))


def _require_snapshot(bundle: CorrectivePublicationBundle, snapshot: Path) -> tuple[str, ...]:
    verify_corrective_publication_bundle(bundle)
    paths = _regular_paths(snapshot)
    required = {"manifest.json", "meta/rft-selection.json"}
    if not required <= set(paths):
        raise ValueError("corrective materialized snapshot lacks required control files")
    manifest = _read_json(snapshot / "manifest.json")
    selection = _read_json(snapshot / "meta" / "rft-selection.json")
    future_actions = manifest.get("future_actions")
    manifest_horizon = (
        future_actions.get("horizon") if isinstance(future_actions, Mapping) else None
    )
    if manifest_horizon != 16 or selection.get("action_horizon") != 16:
        raise ValueError("corrective materialized snapshot action horizon must be exactly 16")
    campaign = selection.get("corrective_campaign")
    if (
        manifest.get("source_format") != "verified_flywheel_rft_release"
        or not isinstance(campaign, Mapping)
        or campaign.get("campaign_receipt_sha256") != bundle.selection.campaign_receipt["receipt_sha256"]
    ):
        raise ValueError("materialized snapshot is not bound to the corrective selection")
    bindings = campaign.get("selected_bindings")
    expected = [
        {"attempt_id": item.attempt_id, "episode_id": item.episode_id, "episode_manifest_sha256": item.episode_manifest_sha256}
        for item in bundle.selection.bindings
    ]
    if len(bundle.selection.bindings) != TARGET_UNIQUE_SUCCESSES or bindings != expected:
        raise ValueError("materialized snapshot selected binding index is stale or incomplete")
    return paths


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _stage_release(
    bundle: CorrectiveReleasePublicationBundle,
    snapshot: Path,
    staging_root: Path,
) -> tuple[Path, str, tuple[SyncEntry, ...]]:
    verified = verify_corrective_release_publication_bundle(bundle)
    corrective = verified.corrective
    snapshot_paths = _require_snapshot(corrective, snapshot)
    staging = Path(tempfile.mkdtemp(prefix="lehome-corrective-rft-", dir=staging_root))
    try:
        _copy_receipt = staging / "campaign-receipt.json"
        _copy_receipt.write_bytes(canonical_json_bytes(corrective.selection.campaign_receipt))
        selected_index = {
            "schema_version": 1,
            "selection_sha256": corrective.selection.selection_sha256,
            "selected_bindings": [
                {"attempt_id": item.attempt_id, "episode_id": item.episode_id, "episode_manifest_sha256": item.episode_manifest_sha256}
                for item in corrective.selection.bindings
            ],
        }
        (staging / "selected-150.json").write_bytes(canonical_json_bytes(selected_index))
        for wave in corrective.wave_evidence.values():
            wave_root = staging / "waves" / f"wave-{wave.wave_index:06d}"
            _copy(Path(wave.wave_manifest_path), wave_root / "manifest.json")
            (wave_root / "provider-evidence.json").write_bytes(
                canonical_json_bytes(wave.provider_evidence)
            )
            _copy(
                Path(wave.provider_snapshot_path),
                wave_root / "provider-source-snapshot.json",
            )
            _copy(
                Path(verified.instance_receipt_paths[wave.wave_index]),
                wave_root / "instance-receipt.json",
            )
        for item in verify_corrective_publication_bundle(corrective):
            attempt_root = staging / "attempts" / item.attempt_id
            (attempt_root / "attempt-receipt.json").parent.mkdir(parents=True, exist_ok=True)
            (attempt_root / "attempt-receipt.json").write_bytes(canonical_json_bytes(item.attempt_receipt))
            raw_root = Path(item.raw_episode_root)
            try:
                from lehome.flywheel.artifacts import verify_episode_manifest

                episode, _manifest = verify_episode_manifest(raw_root)
            except (ImportError, ValueError):
                raise ValueError("corrective raw episode manifest verification failed") from None
            if (
                episode.get("episode_id") != item.attempt_receipt.get("episode_id")
                or episode.get("episode_id") != raw_root.name
                or not isinstance(episode.get("identity"), Mapping)
                or episode["identity"].get("release_stage") != item.attempt_receipt.get("release_stage")
            ):
                raise ValueError("corrective raw episode identity does not match its attempt receipt")
            for relative in _regular_paths(raw_root):
                _copy(raw_root / relative, attempt_root / "raw" / relative)
            _copy(Path(item.policy_receipt_path), attempt_root / "policy-receipt.json")
        for relative in snapshot_paths:
            _copy(snapshot / relative, staging / "materialized-snapshot" / relative)
        candidate_paths = _regular_paths(staging)
        candidate_entries = generate_upload_allowlist(staging, candidate_paths)
        release_id = canonical_json_sha256({
            "schema_version": 1,
            "publication_sha256": corrective.publication_sha256,
            "release_provenance_sha256": verified.release_provenance_sha256,
            "entries": [entry.to_dict() for entry in candidate_entries],
        })
        manifest = {
            "schema_version": 1,
            "kind": "corrective_rft_private_release",
            "release_id": release_id,
            "publication_sha256": corrective.publication_sha256,
            "release_provenance_sha256": verified.release_provenance_sha256,
            "campaign_receipt_sha256": corrective.selection.campaign_receipt["receipt_sha256"],
            "selection_sha256": corrective.selection.selection_sha256,
            "attempt_count": len(corrective.attempt_artifacts),
            "selected_success_count": len(corrective.selection.bindings),
            "instance_ids": {str(wave): verified.instance_ids[wave] for wave in sorted(verified.instance_ids)},
            "entries": [entry.to_dict() for entry in candidate_entries],
        }
        (staging / "release-manifest.json").write_bytes(canonical_json_bytes(manifest))
        entries = generate_upload_allowlist(staging, _regular_paths(staging))
        return staging, release_id, entries
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _tree_matches(tree: Iterable[HubTreeEntry], prefix: str, entries: tuple[SyncEntry, ...]) -> bool:
    expected = {f"{prefix}/{item.relative_path}" for item in entries}
    expected_directories = {
        f"{prefix}/" + "/".join(parts[:index])
        for item in entries
        for parts in (item.relative_path.split("/"),)
        for index in range(1, len(parts))
    }
    observed: set[str] = set()
    for entry in tree:
        if entry.relative_path == prefix:
            if entry.entry_type != "directory":
                return False
            continue
        if not entry.relative_path.startswith(prefix + "/"):
            continue
        if entry.entry_type == "directory":
            if entry.relative_path not in expected_directories:
                return False
            continue
        if entry.entry_type != "file" or entry.relative_path not in expected:
            return False
        observed.add(entry.relative_path)
    return observed == expected


def _verify_readback(root: Path, entries: tuple[SyncEntry, ...]) -> None:
    observed = generate_upload_allowlist(root, tuple(item.relative_path for item in entries))
    if observed != entries or set(_regular_paths(root)) != {item.relative_path for item in entries}:
        raise ValueError("corrective immutable readback does not match the staged release")


def publish_verified_corrective_rft(
    bundle: CorrectiveReleasePublicationBundle,
    materialized_snapshot: str | Path,
    *,
    revision: str,
    transport: HubTransport,
    disposal_receipt: str | Path,
    staging_root: str | Path | None = None,
) -> CorrectivePublicationResult:
    """Publish all audit evidence and authorize disposal only after full readback."""

    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("corrective publish revision must be explicit")
    receipt = Path(disposal_receipt)
    if receipt.exists() or receipt.is_symlink():
        raise FileExistsError("corrective disposal receipt must not already exist")
    verified = verify_corrective_release_publication_bundle(bundle)
    snapshot = Path(materialized_snapshot)
    stage_parent = Path(staging_root) if staging_root is not None else snapshot.parent
    if not stage_parent.is_dir() or stage_parent.is_symlink():
        raise ValueError("corrective staging root must be a materialized directory")
    # Hub helpers obtain HF_TOKEN only from this publisher process environment.
    require_access(transport=transport, repository=DEFAULT_DATA_REPO, read=True, write=True)
    staging, release_id, entries = _stage_release(verified, snapshot, stage_parent)
    prefix = f"{_REMOTE_ROOT}/{release_id}"
    try:
        immutable_revision = upload_files(
            transport=transport, repository=DEFAULT_DATA_REPO, revision=revision, source=staging,
            entries=entries, remote_prefix=prefix, max_attempts=1,
        )
        tree = list_repository_tree(
            transport=transport, repository=DEFAULT_DATA_REPO, revision=immutable_revision,
            max_attempts=1,
        )
        if not _tree_matches(tree, prefix, entries):
            raise ValueError("corrective immutable remote tree does not match the release")
        readback = Path(tempfile.mkdtemp(prefix="lehome-corrective-readback-", dir=stage_parent))
        try:
            download_files(
                transport=transport, repository=DEFAULT_DATA_REPO, revision=immutable_revision,
                destination=readback, relative_paths=tuple(item.relative_path for item in entries),
                remote_prefix=prefix, max_attempts=1,
            )
            _verify_readback(readback, entries)
        finally:
            shutil.rmtree(readback, ignore_errors=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    result = CorrectivePublicationResult(DEFAULT_DATA_REPO, immutable_revision, prefix, release_id, entries, True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(receipt, {
        "schema_version": 1, "repository": result.repository,
        "immutable_revision": result.immutable_revision, "remote_prefix": result.remote_prefix,
        "release_id": result.release_id, "entry_count": len(result.entries),
        "instance_ids": {str(wave): verified.instance_ids[wave] for wave in sorted(verified.instance_ids)},
        "release_provenance_sha256": verified.release_provenance_sha256,
        "tree_listing_verified": True, "fresh_readback_verified": True, "disposable": True,
    })
    return result


__all__ = (
    "CorrectiveCanaryAbortPublicationBundle",
    "CorrectiveCanaryPublicationBundle",
    "CorrectivePublicationResult",
    "build_corrective_canary_abort_publication_bundle",
    "build_corrective_canary_publication_bundle",
    "publish_private_corrective_canary",
    "publish_private_corrective_canary_abort",
    "publish_verified_corrective_rft",
    "verify_corrective_canary_publication_bundle",
    "verify_corrective_canary_abort_publication_bundle",
)
