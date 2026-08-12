"""Fail-closed admission rules for a bounded corrective RFT rollout campaign."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable, Mapping

from lehome_train.io import canonical_json_bytes, sha256_file


_CATEGORIES = ("top_long", "top_short", "pant_long", "pant_short")
_SHORT_CATEGORIES = ("top_short", "pant_short")
ON_DEMAND_RENTAL = "on-demand"
MAX_ATTEMPTS = 400
TARGET_UNIQUE_SUCCESSES = 150
MAX_HOURLY_COST_USD = 2.0
CATEGORY_SUCCESS_FLOORS = {
    "top_long": 30,
    "top_short": 45,
    "pant_long": 30,
    "pant_short": 45,
}
APPROVED_PARENT_REPOSITORY = "ryanjin333/lehome-groot-n17-models"
APPROVED_PARENT_REVISION = "a9076779c970f382bf0341a1015275bf15f13822"
APPROVED_PARENT_ARTIFACT_SHA256 = "3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06"
APPROVED_PARENT_STEP = 12000
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMAGE = re.compile(r"^sha256:[0-9a-f]{64}$")
_REQUIRED_ATTEMPT_KEYS = frozenset({
    "schema_version", "attempt_id", "wave_index", "worker_slot", "episode_id",
    "category", "release_stage", "outcome", "accepted_success", "reset_sha256",
    "randomization_sha256", "hard_state_sha256", "parent_checkpoint_repository",
    "parent_checkpoint_revision", "parent_checkpoint_artifact_sha256",
    "parent_checkpoint_step", "code_revision", "asset_revision", "image_identity",
    "simulator_version", "provider",
})


@dataclass(frozen=True, slots=True)
class CorrectiveEpisode:
    """The minimum immutable identity required before a success can enter RFT."""

    episode_id: str
    category: str
    release_stage: str
    accepted_success: bool


@dataclass(frozen=True, slots=True)
class CorrectiveCampaignPolicy:
    """Hard launch limits and category floors for one corrective collection run."""

    max_attempts: int = MAX_ATTEMPTS
    max_hourly_cost_usd: float = MAX_HOURLY_COST_USD
    category_success_floors: Mapping[str, int] | None = None
    unique_success_floor: int = TARGET_UNIQUE_SUCCESSES

    def __post_init__(self) -> None:
        floors = dict(CATEGORY_SUCCESS_FLOORS if self.category_success_floors is None else self.category_success_floors)
        if set(floors) != set(_CATEGORIES) or any(
            type(value) is not int or value <= 0 for value in floors.values()
        ):
            raise ValueError("corrective category success floors are invalid")
        if any(
            type(value) is not int or value <= 0
            for value in (self.max_attempts, self.unique_success_floor)
        ):
            raise ValueError("corrective campaign integer limits must be positive")
        if (
            type(self.max_hourly_cost_usd) not in (int, float)
            or not math.isfinite(float(self.max_hourly_cost_usd))
            or self.max_hourly_cost_usd <= 0
        ):
            raise ValueError("corrective campaign hourly cost limit must be finite and positive")
        if floors["top_short"] < floors["top_long"] or floors["pant_short"] < floors["pant_long"]:
            raise ValueError("short-garment floors must not be below their long-garment floors")
        if sum(floors.values()) != self.unique_success_floor:
            raise ValueError("corrective category floors must sum to the unique success target")
        object.__setattr__(self, "category_success_floors", floors)

    def floor_for(self, category: str) -> int:
        if category not in _CATEGORIES:
            raise ValueError("corrective episode category is unsupported")
        assert self.category_success_floors is not None
        return self.category_success_floors[category]


@dataclass(frozen=True, slots=True)
class CorrectiveCampaignReport:
    category_successes: dict[str, int]
    missing_successes: dict[str, int]
    unique_successes: int
    priority_categories: tuple[str, ...]
    launch_allowed: bool
    collection_complete: bool


@dataclass(frozen=True, slots=True)
class CorrectiveEpisodeBinding:
    """A selected attempt bound to one freshly verified raw episode artifact."""

    attempt_id: str
    episode_id: str
    root: str
    episode_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class CorrectiveSelectionBundle:
    """The complete typed handoff from campaign evidence to data publication."""

    campaign_receipt: Mapping[str, object]
    bindings: tuple[CorrectiveEpisodeBinding, ...]
    selected_attempt_receipts: tuple[Mapping[str, object], ...]
    selection_sha256: str


@dataclass(frozen=True, slots=True)
class CorrectiveAttemptArtifact:
    """One terminal attempt receipt and its complete raw/policy evidence."""

    attempt_id: str
    attempt_receipt: Mapping[str, object]
    raw_episode_root: str
    episode_manifest_sha256: str
    policy_receipt_path: str
    policy_receipt_sha256: str


@dataclass(frozen=True, slots=True)
class CorrectiveWaveEvidence:
    """A complete non-secret provider and scheduling record for one wave."""

    wave_index: int
    wave_manifest_path: str
    wave_manifest_sha256: str
    provider_evidence: Mapping[str, object]
    provider_snapshot_path: str
    provider_snapshot_sha256: str


@dataclass(frozen=True, slots=True)
class CorrectivePublicationBundle:
    """The selected training index plus every attempt required for audit."""

    selection: CorrectiveSelectionBundle
    attempt_artifacts: Mapping[str, CorrectiveAttemptArtifact]
    wave_evidence: Mapping[int, CorrectiveWaveEvidence]
    publication_sha256: str


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"corrective {label} is invalid")
    return value


def _validate_attempt(value: Mapping[str, object]) -> dict[str, object]:
    if set(value) != _REQUIRED_ATTEMPT_KEYS or value.get("schema_version") != 1:
        raise ValueError("corrective attempt receipt schema is invalid")
    attempt = dict(value)
    if not isinstance(attempt["attempt_id"], str) or not attempt["attempt_id"]:
        raise ValueError("corrective attempt ID is invalid")
    if type(attempt["wave_index"]) is not int or attempt["wave_index"] < 0:
        raise ValueError("corrective wave index is invalid")
    if type(attempt["worker_slot"]) is not int or attempt["worker_slot"] not in range(4):
        raise ValueError("corrective worker slot is invalid")
    if not isinstance(attempt["episode_id"], str) or not attempt["episode_id"]:
        raise ValueError("corrective episode ID is invalid")
    if attempt["category"] not in _CATEGORIES or attempt["release_stage"] != "seen":
        raise ValueError("corrective collection must be seen-only across all categories")
    if attempt["outcome"] not in {"success", "failure", "timeout", "error"}:
        raise ValueError("corrective attempt outcome is invalid")
    if not isinstance(attempt["accepted_success"], bool) or (
        attempt["accepted_success"] and attempt["outcome"] != "success"
    ):
        raise ValueError("corrective success evidence is invalid")
    for key in ("reset_sha256", "randomization_sha256", "hard_state_sha256", "parent_checkpoint_artifact_sha256"):
        _require_sha256(attempt[key], key)
    if (
        attempt["parent_checkpoint_repository"] != APPROVED_PARENT_REPOSITORY
        or attempt["parent_checkpoint_revision"] != APPROVED_PARENT_REVISION
        or attempt["parent_checkpoint_artifact_sha256"] != APPROVED_PARENT_ARTIFACT_SHA256
        or attempt["parent_checkpoint_step"] != APPROVED_PARENT_STEP
        or _COMMIT.fullmatch(str(attempt["code_revision"])) is None
        or _COMMIT.fullmatch(str(attempt["asset_revision"])) is None
        or _IMAGE.fullmatch(str(attempt["image_identity"])) is None
        or not isinstance(attempt["simulator_version"], str)
        or not attempt["simulator_version"]
    ):
        raise ValueError("corrective attempt provenance is invalid")
    provider = attempt["provider"]
    if (
        not isinstance(provider, Mapping)
        or set(provider) != {
            "rental_kind", "instance_hourly_cost_usd", "account_hourly_total_usd",
            "offer_id", "gpu_name", "num_gpus",
        }
        or provider["rental_kind"] != ON_DEMAND_RENTAL
        or type(provider["instance_hourly_cost_usd"]) not in (int, float)
        or not math.isfinite(float(provider["instance_hourly_cost_usd"]))
        or float(provider["instance_hourly_cost_usd"]) <= 0
        or type(provider["account_hourly_total_usd"]) not in (int, float)
        or not math.isfinite(float(provider["account_hourly_total_usd"]))
        or float(provider["account_hourly_total_usd"]) < float(provider["instance_hourly_cost_usd"])
        or float(provider["account_hourly_total_usd"]) > MAX_HOURLY_COST_USD
        or type(provider["offer_id"]) is not int
        or provider["offer_id"] <= 0
        or provider["gpu_name"] != "RTX 3090"
        or provider["num_gpus"] != 4
    ):
        raise ValueError("corrective provider facts do not prove an on-demand 4x3090 under the shared $2/hr cap")
    return attempt


def _validate_attempts(attempts: Iterable[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    validated = tuple(_validate_attempt(item) for item in attempts)
    if not validated or len(validated) > MAX_ATTEMPTS:
        raise ValueError("corrective campaign must contain between one and 400 attempts")
    attempt_ids = [str(item["attempt_id"]) for item in validated]
    if len(set(attempt_ids)) != len(attempt_ids):
        raise ValueError("corrective attempt IDs must be unique")
    episode_ids = [str(item["episode_id"]) for item in validated]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("corrective episode IDs must be unique")
    parent_identity = tuple(validated[0][key] for key in (
        "parent_checkpoint_repository", "parent_checkpoint_revision", "parent_checkpoint_artifact_sha256",
        "parent_checkpoint_step", "code_revision", "asset_revision", "image_identity", "simulator_version",
    ))
    if any(tuple(item[key] for key in (
        "parent_checkpoint_repository", "parent_checkpoint_revision", "parent_checkpoint_artifact_sha256",
        "parent_checkpoint_step", "code_revision", "asset_revision", "image_identity", "simulator_version",
    )) != parent_identity for item in validated):
        raise ValueError("corrective attempts must share one immutable parent and runtime identity")
    by_wave: dict[int, list[dict[str, object]]] = {}
    for item in validated:
        by_wave.setdefault(int(item["wave_index"]), []).append(item)
    ordered_waves = sorted(by_wave)
    if ordered_waves != list(range(len(ordered_waves))) or any(
        len(by_wave[wave]) != 4 or {item["worker_slot"] for item in by_wave[wave]} != set(range(4))
        for wave in ordered_waves
    ):
        raise ValueError("corrective campaign requires complete four-worker waves")
    if any(
        any(item["provider"] != by_wave[wave][0]["provider"] for item in by_wave[wave])
        for wave in ordered_waves
    ):
        raise ValueError("corrective wave attempts must share one verified provider identity")
    return validated


def _priority_categories(successes: Mapping[str, int]) -> tuple[str, ...]:
    missing = {category for category in _CATEGORIES if successes[category] < CATEGORY_SUCCESS_FLOORS[category]}
    priority = tuple(category for category in _SHORT_CATEGORIES if category in missing)
    return priority or tuple(category for category in _CATEGORIES if category in missing)


def _next_wave_categories(successes: Mapping[str, int]) -> tuple[str, ...]:
    """Fill all four worker slots, alternating the remaining priority buckets."""
    priority = _priority_categories(successes)
    if not priority:
        return ()
    return tuple(priority[index % len(priority)] for index in range(4))


def build_corrective_campaign_receipt(attempts: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Derive a complete campaign receipt only from terminal attempt receipts."""
    validated = _validate_attempts(attempts)
    successes = {category: 0 for category in _CATEGORIES}
    for item in validated:
        if item["accepted_success"]:
            successes[str(item["category"])] += 1
    body = {
        "schema_version": 1,
        "kind": "corrective_rft_campaign",
        "attempt_count": len(validated),
        "attempt_ids": [item["attempt_id"] for item in validated],
        "attempt_receipts": [dict(item) for item in validated],
        "attempt_receipt_sha256s": [_canonical_sha256(item) for item in validated],
        "success_counts": successes,
        "next_wave_categories": list(_next_wave_categories(successes)),
        "parent_identity": {
            key: validated[0][key]
            for key in (
                "parent_checkpoint_repository", "parent_checkpoint_revision", "parent_checkpoint_artifact_sha256",
                "parent_checkpoint_step", "code_revision", "asset_revision", "image_identity", "simulator_version",
            )
        },
        "provider_by_wave": {
            str(wave): dict(items[0]["provider"])
            for wave, items in sorted({
                wave: [item for item in validated if item["wave_index"] == wave]
                for wave in {item["wave_index"] for item in validated}
            }.items())
        },
    }
    return {**body, "receipt_sha256": _canonical_sha256(body)}


def select_corrective_successes(attempts: Iterable[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    """Deterministically select exactly 150 successful, independently reset states."""
    validated = _validate_attempts(attempts)
    successes = [item for item in validated if item["accepted_success"]]
    if len(successes) < TARGET_UNIQUE_SUCCESSES:
        raise ValueError("corrective campaign has fewer than 150 accepted successes")
    fingerprints = [
        (item["reset_sha256"], item["randomization_sha256"], item["hard_state_sha256"])
        for item in successes
    ]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("corrective successes must have distinct state fingerprints")
    grouped = {category: sorted(
        (item for item in successes if item["category"] == category),
        key=lambda item: (str(item["attempt_id"]), str(item["episode_id"])),
    ) for category in _CATEGORIES}
    if any(len(grouped[category]) < CATEGORY_SUCCESS_FLOORS[category] for category in _CATEGORIES):
        raise ValueError("corrective success category floors are incomplete")
    selected = [
        *(
            grouped[category][:CATEGORY_SUCCESS_FLOORS[category]]
            for category in _CATEGORIES
        )
    ]
    flat = tuple(item for category in selected for item in category)
    if len(flat) != TARGET_UNIQUE_SUCCESSES:
        raise ValueError("corrective selection must contain exactly 150 successes")
    return flat


def bind_corrective_episode_artifacts(
    selected_attempts: Iterable[Mapping[str, object]],
    verified_episodes: Mapping[str, Mapping[str, object]],
) -> tuple[CorrectiveEpisodeBinding, ...]:
    """Bind selected receipts to fresh episode manifests and immutable identities."""
    selected = tuple(selected_attempts)
    if len(selected) != TARGET_UNIQUE_SUCCESSES:
        raise ValueError("corrective artifact binding requires exactly 150 selected attempts")
    bindings: list[CorrectiveEpisodeBinding] = []
    for attempt in selected:
        attempt_id = attempt.get("attempt_id")
        episode_id = attempt.get("episode_id")
        if not isinstance(attempt_id, str) or not isinstance(episode_id, str):
            raise ValueError("corrective selected attempt identity is invalid")
        artifact = verified_episodes.get(attempt_id)
        if not isinstance(artifact, Mapping):
            raise ValueError("corrective selected attempt is missing a verified episode artifact")
        if artifact.get("episode_id") != episode_id or artifact.get("release_stage") != "seen":
            raise ValueError("corrective verified episode identity does not match its attempt")
        root, manifest = artifact.get("root"), artifact.get("episode_manifest_sha256")
        if not isinstance(root, str) or not root or _SHA256.fullmatch(str(manifest)) is None:
            raise ValueError("corrective verified episode artifact binding is invalid")
        bindings.append(CorrectiveEpisodeBinding(attempt_id, episode_id, root, str(manifest)))
    if len({item.episode_manifest_sha256 for item in bindings}) != len(bindings):
        raise ValueError("corrective selected episode manifests must be distinct")
    return tuple(bindings)


def _selection_body(
    campaign_receipt: Mapping[str, object],
    bindings: Iterable[CorrectiveEpisodeBinding],
    selected_attempt_receipts: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "corrective_rft_selection",
        "campaign_receipt_sha256": campaign_receipt.get("receipt_sha256"),
        "selected_attempt_receipts": [dict(item) for item in selected_attempt_receipts],
        "bindings": [
            {
                "attempt_id": item.attempt_id,
                "episode_id": item.episode_id,
                "root": item.root,
                "episode_manifest_sha256": item.episode_manifest_sha256,
            }
            for item in bindings
        ],
    }


def build_corrective_selection_bundle(
    attempts: Iterable[Mapping[str, object]],
    verified_episodes: Mapping[str, Mapping[str, object]],
) -> CorrectiveSelectionBundle:
    """Close the attempt -> exact selection -> verified artifact handoff."""

    frozen_attempts = tuple(attempts)
    receipt = build_corrective_campaign_receipt(frozen_attempts)
    selected = select_corrective_successes(frozen_attempts)
    bindings = bind_corrective_episode_artifacts(selected, verified_episodes)
    body = _selection_body(receipt, bindings, selected)
    return CorrectiveSelectionBundle(receipt, bindings, tuple(dict(item) for item in selected), _canonical_sha256(body))


def verify_corrective_selection_bundle(
    bundle: CorrectiveSelectionBundle,
) -> tuple[CorrectiveEpisodeBinding, ...]:
    """Reject a stale or hand-assembled selection bundle."""

    if not isinstance(bundle, CorrectiveSelectionBundle):
        raise ValueError("corrective selection bundle is invalid")
    receipt = bundle.campaign_receipt
    receipt_sha = receipt.get("receipt_sha256") if isinstance(receipt, Mapping) else None
    if _SHA256.fullmatch(str(receipt_sha)) is None:
        raise ValueError("corrective campaign receipt is invalid")
    body_without_hash = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if _canonical_sha256(body_without_hash) != receipt_sha:
        raise ValueError("corrective campaign receipt hash is stale")
    recorded_attempts = receipt.get("attempt_receipts")
    if not isinstance(recorded_attempts, list) or not all(isinstance(item, Mapping) for item in recorded_attempts):
        raise ValueError("corrective campaign receipt lacks canonical attempt receipts")
    validated_attempts = _validate_attempts(recorded_attempts)
    if build_corrective_campaign_receipt(validated_attempts) != dict(receipt):
        raise ValueError("corrective campaign receipt attempt ledger is stale")
    selected_receipts = tuple(bundle.selected_attempt_receipts)
    expected_selected = select_corrective_successes(validated_attempts)
    if tuple(dict(item) for item in selected_receipts) != expected_selected:
        raise ValueError("corrective selected receipt ledger is stale or forged")
    bindings = tuple(bundle.bindings)
    if (
        len(bindings) != TARGET_UNIQUE_SUCCESSES
        or len({item.attempt_id for item in bindings}) != TARGET_UNIQUE_SUCCESSES
        or len({item.episode_id for item in bindings}) != TARGET_UNIQUE_SUCCESSES
        or len({item.episode_manifest_sha256 for item in bindings}) != TARGET_UNIQUE_SUCCESSES
    ):
        raise ValueError("corrective selection bundle must bind exactly 150 distinct artifacts")
    if any(
        binding.attempt_id != receipt_item["attempt_id"]
        or binding.episode_id != receipt_item["episode_id"]
        for binding, receipt_item in zip(bindings, selected_receipts, strict=True)
    ):
        raise ValueError("corrective selected receipt and artifact binding differ")
    if _canonical_sha256(_selection_body(receipt, bindings, selected_receipts)) != bundle.selection_sha256:
        raise ValueError("corrective selection bundle hash is stale")
    return bindings


def build_corrective_publication_plan(
    bundle: CorrectiveSelectionBundle,
) -> dict[str, object]:
    """Describe a private immutable publish/readback gate without calling the Hub."""
    bindings = verify_corrective_selection_bundle(bundle)
    receipt_sha = _require_sha256(
        bundle.campaign_receipt.get("receipt_sha256"), "campaign receipt hash"
    )
    return {
        "schema_version": 1,
        "kind": "corrective_rft_private_publication",
        "campaign_receipt_sha256": receipt_sha,
        "selection_sha256": bundle.selection_sha256,
        "repository_private": True,
        "selected_attempts": [
            {
                "attempt_id": item.attempt_id,
                "episode_id": item.episode_id,
                "episode_manifest_sha256": item.episode_manifest_sha256,
            }
            for item in bindings
        ],
        "required_verification": ["immutable_revision", "tree_listing", "fresh_readback"],
        "disposable": False,
    }


def _regular_file_sha256(path_value: str, label: str) -> str:
    path = Path(path_value)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"corrective {label} is not a materialized regular file")
    return sha256_file(path)


def _publication_body(
    selection: CorrectiveSelectionBundle,
    artifacts: Iterable[CorrectiveAttemptArtifact],
    wave_evidence: Iterable[CorrectiveWaveEvidence],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "corrective_rft_publication_bundle",
        "campaign_receipt_sha256": selection.campaign_receipt["receipt_sha256"],
        "selection_sha256": selection.selection_sha256,
        "attempt_artifacts": [
            {
                "attempt_id": item.attempt_id,
                "attempt_receipt_sha256": _canonical_sha256(item.attempt_receipt),
                "raw_episode_root": item.raw_episode_root,
                "episode_manifest_sha256": item.episode_manifest_sha256,
                "policy_receipt_sha256": item.policy_receipt_sha256,
            }
            for item in artifacts
        ],
        "wave_evidence": [
            {
                "wave_index": item.wave_index,
                "wave_manifest_sha256": item.wave_manifest_sha256,
                "provider_evidence": dict(item.provider_evidence),
                "provider_snapshot_sha256": item.provider_snapshot_sha256,
            }
            for item in wave_evidence
        ],
    }


def build_corrective_publication_bundle(
    selection: CorrectiveSelectionBundle,
    artifacts: Mapping[str, Mapping[str, object] | CorrectiveAttemptArtifact],
    wave_evidence: Mapping[int, Mapping[str, object] | CorrectiveWaveEvidence],
) -> CorrectivePublicationBundle:
    """Bind every terminal attempt body and raw/policy artifact to a selection."""

    bindings = verify_corrective_selection_bundle(selection)
    receipt = selection.campaign_receipt
    attempt_ids = receipt.get("attempt_ids")
    receipt_hashes = receipt.get("attempt_receipt_sha256s")
    if not isinstance(attempt_ids, list) or not isinstance(receipt_hashes, list):
        raise ValueError("corrective campaign receipt lacks an attempt ledger")
    if len(attempt_ids) != len(receipt_hashes) or not all(isinstance(item, str) for item in attempt_ids):
        raise ValueError("corrective campaign receipt attempt ledger is invalid")
    if set(artifacts) != set(attempt_ids):
        raise ValueError("corrective attempt artifact IDs must exactly match the campaign receipt")
    expected_hashes = dict(zip(attempt_ids, receipt_hashes, strict=True))
    normalized: list[CorrectiveAttemptArtifact] = []
    for attempt_id in attempt_ids:
        supplied = artifacts[attempt_id]
        if isinstance(supplied, CorrectiveAttemptArtifact):
            supplied = {
                "attempt_receipt": supplied.attempt_receipt,
                "root": supplied.raw_episode_root,
                "episode_manifest_sha256": supplied.episode_manifest_sha256,
                "policy_receipt_path": supplied.policy_receipt_path,
                "policy_receipt_sha256": supplied.policy_receipt_sha256,
            }
        if not isinstance(supplied, Mapping):
            raise ValueError("corrective attempt artifact binding is invalid")
        attempt = supplied.get("attempt_receipt")
        root = supplied.get("root")
        manifest_sha = supplied.get("episode_manifest_sha256")
        policy_path = supplied.get("policy_receipt_path")
        policy_sha = supplied.get("policy_receipt_sha256")
        if (
            not isinstance(attempt, Mapping)
            or attempt.get("attempt_id") != attempt_id
            or _canonical_sha256(attempt) != expected_hashes[attempt_id]
            or not isinstance(root, str)
            or not isinstance(manifest_sha, str)
            or not isinstance(policy_path, str)
            or not isinstance(policy_sha, str)
        ):
            raise ValueError("corrective attempt artifact binding is invalid")
        raw_root = Path(root)
        manifest_path = raw_root / "SHA256SUMS.json"
        if not raw_root.is_dir() or raw_root.is_symlink() or _regular_file_sha256(str(manifest_path), "raw manifest") != manifest_sha:
            raise ValueError("corrective raw episode manifest is stale or unavailable")
        if _regular_file_sha256(policy_path, "policy receipt") != policy_sha:
            raise ValueError("corrective policy receipt is stale or unavailable")
        normalized.append(CorrectiveAttemptArtifact(
            attempt_id=attempt_id,
            attempt_receipt=dict(attempt),
            raw_episode_root=root,
            episode_manifest_sha256=manifest_sha,
            policy_receipt_path=policy_path,
            policy_receipt_sha256=policy_sha,
        ))
    artifact_by_id = {item.attempt_id: item for item in normalized}
    for binding in bindings:
        attempt = artifact_by_id[binding.attempt_id]
        if (
            attempt.attempt_receipt.get("episode_id") != binding.episode_id
            or attempt.episode_manifest_sha256 != binding.episode_manifest_sha256
            or attempt.raw_episode_root != binding.root
        ):
            raise ValueError("corrective selected artifact binding differs from the full attempt ledger")
    by_wave = {item["wave_index"] for item in receipt["attempt_receipts"]}
    if set(wave_evidence) != by_wave:
        raise ValueError("corrective wave evidence must exactly cover every attempt wave")
    normalized_waves: list[CorrectiveWaveEvidence] = []
    for wave_index in sorted(by_wave):
        supplied_wave = wave_evidence[wave_index]
        if isinstance(supplied_wave, CorrectiveWaveEvidence):
            supplied_wave = {
                "wave_manifest_path": supplied_wave.wave_manifest_path,
                "wave_manifest_sha256": supplied_wave.wave_manifest_sha256,
                "provider_evidence": supplied_wave.provider_evidence,
                "provider_snapshot_path": supplied_wave.provider_snapshot_path,
                "provider_snapshot_sha256": supplied_wave.provider_snapshot_sha256,
            }
        if not isinstance(supplied_wave, Mapping):
            raise ValueError("corrective wave evidence binding is invalid")
        manifest_path = supplied_wave.get("wave_manifest_path")
        manifest_sha = supplied_wave.get("wave_manifest_sha256")
        provider = supplied_wave.get("provider_evidence")
        snapshot_path = supplied_wave.get("provider_snapshot_path")
        snapshot_sha = supplied_wave.get("provider_snapshot_sha256")
        if (
            not isinstance(manifest_path, str) or not isinstance(manifest_sha, str)
            or not isinstance(provider, Mapping) or not isinstance(snapshot_path, str)
            or not isinstance(snapshot_sha, str)
            or _regular_file_sha256(manifest_path, "wave manifest") != manifest_sha
            or _regular_file_sha256(snapshot_path, "provider source snapshot") != snapshot_sha
            or provider.get("schema_version") != 1
            or provider.get("kind") != "external_provider_offer_evidence"
            or provider.get("source_snapshot_sha256") != snapshot_sha
            or provider.get("offer_id") != receipt["provider_by_wave"][str(wave_index)]["offer_id"]
            or dict(provider).get("rental_kind") != "on-demand"
            or dict(provider).get("gpu_name") != "RTX 3090"
            or dict(provider).get("num_gpus") != 4
        ):
            raise ValueError("corrective wave evidence binding is invalid")
        manifest = Path(manifest_path)
        try:
            manifest_body = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise ValueError("corrective wave evidence manifest is invalid") from None
        wave_attempt_ids = [item["attempt_id"] for item in receipt["attempt_receipts"] if item["wave_index"] == wave_index]
        manifest_attempts = manifest_body.get("attempts") if isinstance(manifest_body, Mapping) else None
        if (
            not isinstance(manifest_body, Mapping)
            or manifest_body.get("schema_version") != 1
            or manifest_body.get("kind") != "corrective_rft_wave"
            or manifest_body.get("wave_index") != wave_index
            or manifest_body.get("provider_evidence") != dict(provider)
            or manifest_body.get("provider") != receipt["provider_by_wave"][str(wave_index)]
            or not isinstance(manifest_attempts, list)
            or [item.get("attempt_id") if isinstance(item, Mapping) else None for item in manifest_attempts] != wave_attempt_ids
        ):
            raise ValueError("corrective wave evidence does not bind its exact attempts")
        normalized_waves.append(CorrectiveWaveEvidence(
            wave_index=wave_index, wave_manifest_path=manifest_path, wave_manifest_sha256=manifest_sha,
            provider_evidence=dict(provider), provider_snapshot_path=snapshot_path, provider_snapshot_sha256=snapshot_sha,
        ))
    frozen = tuple(normalized)
    frozen_waves = tuple(normalized_waves)
    return CorrectivePublicationBundle(
        selection=selection,
        attempt_artifacts={item.attempt_id: item for item in frozen},
        wave_evidence={item.wave_index: item for item in frozen_waves},
        publication_sha256=_canonical_sha256(_publication_body(selection, frozen, frozen_waves)),
    )


def verify_corrective_publication_bundle(
    bundle: CorrectivePublicationBundle,
) -> tuple[CorrectiveAttemptArtifact, ...]:
    """Revalidate a typed all-attempt publication handoff before upload."""

    if not isinstance(bundle, CorrectivePublicationBundle):
        raise ValueError("corrective publication bundle is invalid")
    rebuilt = build_corrective_publication_bundle(bundle.selection, {
        attempt_id: {
            "attempt_receipt": item.attempt_receipt,
            "root": item.raw_episode_root,
            "episode_manifest_sha256": item.episode_manifest_sha256,
            "policy_receipt_path": item.policy_receipt_path,
            "policy_receipt_sha256": item.policy_receipt_sha256,
        }
        for attempt_id, item in bundle.attempt_artifacts.items()
    }, bundle.wave_evidence)
    if rebuilt.publication_sha256 != bundle.publication_sha256:
        raise ValueError("corrective publication bundle hash is stale")
    return tuple(rebuilt.attempt_artifacts[item] for item in rebuilt.selection.campaign_receipt["attempt_ids"])


def _require_launch_contract(
    *,
    policy: CorrectiveCampaignPolicy,
    attempted_episodes: int,
    offered_hourly_cost_usd: float,
    rental_kind: str,
) -> None:
    if rental_kind != ON_DEMAND_RENTAL:
        raise ValueError("corrective rollout rental must be on-demand")
    if type(attempted_episodes) is not int or attempted_episodes < 0:
        raise ValueError("corrective attempted episode count is invalid")
    if attempted_episodes > policy.max_attempts:
        raise ValueError("corrective campaign attempt limit has been reached")
    if (
        type(offered_hourly_cost_usd) not in (int, float)
        or not math.isfinite(float(offered_hourly_cost_usd))
        or offered_hourly_cost_usd < 0
    ):
        raise ValueError("corrective rollout hourly cost is invalid")
    if offered_hourly_cost_usd > policy.max_hourly_cost_usd:
        raise ValueError("corrective rollout hourly cost exceeds the shared $2/hr cap")


def assess_corrective_campaign(
    episodes: Iterable[CorrectiveEpisode | Mapping[str, object]],
    *,
    policy: CorrectiveCampaignPolicy,
    attempted_episodes: int,
    offered_hourly_cost_usd: float,
    rental_kind: str,
) -> CorrectiveCampaignReport:
    """Validate collection evidence and return the only allowed next-category order.

    Public-unseen episodes are rejected instead of ignored, so callers cannot
    accidentally feed fixed-evaluation evidence into corrective RFT.
    """

    _require_launch_contract(
        policy=policy,
        attempted_episodes=attempted_episodes,
        offered_hourly_cost_usd=offered_hourly_cost_usd,
        rental_kind=rental_kind,
    )
    successes = {category: 0 for category in _CATEGORIES}
    seen_success_ids: set[str] = set()
    for supplied in episodes:
        if isinstance(supplied, Mapping):
            try:
                episode = CorrectiveEpisode(
                    episode_id=supplied["episode_id"],
                    category=supplied["category"],
                    release_stage=supplied["release_stage"],
                    accepted_success=supplied["accepted_success"],
                )
            except (KeyError, TypeError):
                raise ValueError("corrective campaign episode evidence is invalid") from None
        elif isinstance(supplied, CorrectiveEpisode):
            episode = supplied
        else:
            raise ValueError("corrective campaign episode evidence is invalid")
        if episode.category not in _CATEGORIES:
            raise ValueError("corrective episode category is unsupported")
        if episode.release_stage != "seen":
            raise ValueError("corrective RFT rejects unseen evaluation episodes")
        if not isinstance(episode.episode_id, str) or not episode.episode_id:
            raise ValueError("corrective episode identity is invalid")
        if not isinstance(episode.accepted_success, bool):
            raise ValueError("corrective episode success evidence is invalid")
        if not episode.accepted_success:
            continue
        if episode.episode_id in seen_success_ids:
            raise ValueError("corrective RFT successes must be distinct episodes")
        seen_success_ids.add(episode.episode_id)
        successes[episode.category] += 1

    missing = {
        category: policy.floor_for(category) - count
        for category, count in successes.items()
        if count < policy.floor_for(category)
    }
    priority = tuple(category for category in _SHORT_CATEGORIES if category in missing)
    if not priority:
        priority = tuple(category for category in _CATEGORIES if category in missing)
    complete = not missing and len(seen_success_ids) == policy.unique_success_floor
    return CorrectiveCampaignReport(
        category_successes=successes,
        missing_successes=missing,
        unique_successes=len(seen_success_ids),
        priority_categories=priority,
        launch_allowed=(attempted_episodes < policy.max_attempts and not complete),
        collection_complete=complete,
    )


__all__ = (
    "CATEGORY_SUCCESS_FLOORS",
    "APPROVED_PARENT_ARTIFACT_SHA256",
    "APPROVED_PARENT_REPOSITORY",
    "APPROVED_PARENT_REVISION",
    "APPROVED_PARENT_STEP",
    "CorrectiveCampaignPolicy",
    "CorrectiveCampaignReport",
    "CorrectiveEpisode",
    "CorrectiveEpisodeBinding",
    "CorrectiveAttemptArtifact",
    "CorrectivePublicationBundle",
    "CorrectiveSelectionBundle",
    "MAX_ATTEMPTS",
    "MAX_HOURLY_COST_USD",
    "ON_DEMAND_RENTAL",
    "TARGET_UNIQUE_SUCCESSES",
    "assess_corrective_campaign",
    "bind_corrective_episode_artifacts",
    "build_corrective_campaign_receipt",
    "build_corrective_publication_plan",
    "build_corrective_publication_bundle",
    "build_corrective_selection_bundle",
    "select_corrective_successes",
    "verify_corrective_selection_bundle",
    "verify_corrective_publication_bundle",
)
