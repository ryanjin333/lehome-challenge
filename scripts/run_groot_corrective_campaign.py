"""Prepare and close bounded corrective RFT rollout waves without provider actions.

This controller intentionally does not import or call a Vast client.  A separate
lifecycle layer must preflight the selected on-demand offer, execute the emitted
four-worker manifest, and leave only terminal local artifacts for this process
to verify on the next invocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Iterable, Mapping, Sequence

from lehome.flywheel.artifacts import verify_episode_manifest
from lehome.flywheel.matrix import CATEGORY_PREFIX
from lehome_train.flywheel.corrective import (
    APPROVED_PARENT_ARTIFACT_SHA256,
    APPROVED_PARENT_REPOSITORY,
    APPROVED_PARENT_REVISION,
    APPROVED_PARENT_STEP,
    CATEGORY_SUCCESS_FLOORS,
    MAX_ATTEMPTS,
    MAX_HOURLY_COST_USD,
    build_corrective_campaign_receipt,
    select_corrective_successes,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
APPROVED_IMAGE_REPOSITORY = "docker.io/ryanjin333/lehome-rollout"
APPROVED_IMAGE_DIGEST = "sha256:293c4f258f3742a7234699d706fb7088d0da8a764957bc79b244d830561abc12"
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_CATEGORIES = tuple(CATEGORY_PREFIX)
_BASELINE_KEYS = frozenset({
    "schema_version", "kind", "parent_checkpoint_repository", "parent_checkpoint_revision",
    "parent_checkpoint_artifact_sha256", "parent_checkpoint_step", "code_revision",
    "asset_revision", "image_identity", "simulator_version", "policy_path",
    "policy_revision_file", "release_assets_root", "groot_root", "groot_revision", "groot_python",
    "rollout_image", "controller_python",
})
_PROVIDER_FACT_KEYS = frozenset({
    "rental_kind", "instance_hourly_cost_usd", "account_hourly_total_usd", "offer_id", "gpu_name", "num_gpus",
})
_PROVIDER_EVIDENCE_KEYS = _PROVIDER_FACT_KEYS | frozenset({
    "schema_version", "kind", "evidence_id", "queried_at_unix", "expires_at_unix", "source_snapshot_path",
    "source_snapshot_sha256", "source_response_sha256",
})


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular JSON file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _write_new_json(path: Path, value: Mapping[str, object]) -> None:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"refusing to overwrite differing {path.name}")
        return
    path.write_text(encoded, encoding="utf-8")


def _baseline(path: Path) -> dict[str, object]:
    baseline = _read_json(path, label="corrective baseline")
    if set(baseline) != _BASELINE_KEYS or baseline.get("schema_version") != 1 or baseline.get("kind") != "corrective_rft_baseline":
        raise ValueError("corrective baseline schema is invalid")
    required = {
        "parent_checkpoint_repository": APPROVED_PARENT_REPOSITORY,
        "parent_checkpoint_revision": APPROVED_PARENT_REVISION,
        "parent_checkpoint_artifact_sha256": APPROVED_PARENT_ARTIFACT_SHA256,
        "parent_checkpoint_step": APPROVED_PARENT_STEP,
    }
    if any(baseline.get(key) != value for key, value in required.items()):
        raise ValueError("corrective baseline is not pinned to approved step12000")
    if not _COMMIT.fullmatch(str(baseline["code_revision"])) or not _COMMIT.fullmatch(str(baseline["asset_revision"])):
        raise ValueError("corrective baseline code or assets must use immutable revisions")
    if not _SHA256.fullmatch(str(baseline["image_identity"])[7:]) or not str(baseline["image_identity"]).startswith("sha256:"):
        raise ValueError("corrective baseline image identity is invalid")
    rollout_repository, separator, rollout_digest = str(baseline["rollout_image"]).partition("@")
    if rollout_repository != APPROVED_IMAGE_REPOSITORY or separator != "@" or rollout_digest != APPROVED_IMAGE_DIGEST:
        raise ValueError("corrective baseline rollout image must be a pinned approved repository digest")
    if rollout_digest != baseline["image_identity"]:
        raise ValueError("corrective rollout image digest must equal the OCI image identity")
    if not _COMMIT.fullmatch(str(baseline["groot_revision"])) or not all(isinstance(baseline[key], str) and baseline[key] for key in ("simulator_version", "policy_path", "policy_revision_file", "release_assets_root", "groot_root", "groot_python", "controller_python")):
        raise ValueError("corrective baseline runtime invocation is incomplete")
    return baseline


def _provider(path: Path, *, now_unix: int) -> dict[str, object]:
    provider = _read_json(path, label="corrective provider receipt")
    if set(provider) != _PROVIDER_EVIDENCE_KEYS or provider.get("schema_version") != 1 or provider.get("kind") != "external_provider_offer_evidence":
        raise ValueError("corrective provider evidence schema is invalid")
    if not isinstance(provider.get("evidence_id"), str) or not provider["evidence_id"]:
        raise ValueError("corrective provider evidence ID is invalid")
    if type(provider.get("queried_at_unix")) is not int or type(provider.get("expires_at_unix")) is not int or provider["queried_at_unix"] > now_unix or provider["expires_at_unix"] <= now_unix:
        raise ValueError("corrective provider evidence is expired or has an invalid query window")
    snapshot_relative = provider.get("source_snapshot_path")
    if not isinstance(snapshot_relative, str) or Path(snapshot_relative).is_absolute() or not Path(snapshot_relative).parts or any(part in {"", ".", ".."} for part in Path(snapshot_relative).parts):
        raise ValueError("corrective provider evidence snapshot path is unsafe")
    snapshot = path.parent.joinpath(*Path(snapshot_relative).parts)
    if snapshot.is_symlink() or not snapshot.is_file() or not _SHA256.fullmatch(str(provider.get("source_snapshot_sha256"))) or hashlib.sha256(snapshot.read_bytes()).hexdigest() != provider["source_snapshot_sha256"]:
        raise ValueError("corrective provider evidence snapshot binding is invalid")
    if not _SHA256.fullmatch(str(provider.get("source_response_sha256"))):
        raise ValueError("corrective provider evidence source response hash is invalid")
    if provider.get("rental_kind") != "on-demand" or provider.get("gpu_name") != "RTX 3090" or provider.get("num_gpus") != 4:
        raise ValueError("corrective provider receipt must prove an on-demand 4xRTX3090 offer")
    if type(provider.get("offer_id")) is not int or provider["offer_id"] <= 0:
        raise ValueError("corrective provider receipt offer ID is invalid")
    for key in ("instance_hourly_cost_usd", "account_hourly_total_usd"):
        if type(provider.get(key)) not in (int, float) or not math.isfinite(float(provider[key])) or float(provider[key]) <= 0:
            raise ValueError("corrective provider receipt hourly cost is invalid")
    if float(provider["instance_hourly_cost_usd"]) > float(provider["account_hourly_total_usd"]) or float(provider["account_hourly_total_usd"]) > MAX_HOURLY_COST_USD:
        raise ValueError("corrective provider receipt exceeds the shared $2/hr cap")
    return provider


def _provider_facts(provider_evidence: Mapping[str, object]) -> dict[str, object]:
    return {key: provider_evidence[key] for key in _PROVIDER_FACT_KEYS}


def _manifest_path(root: Path, wave_index: int) -> Path:
    return root / "waves" / f"wave-{wave_index:06d}.json"


def _manifest_attempts(manifest: Mapping[str, object]) -> list[dict[str, object]]:
    attempts = manifest.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != 4 or not all(isinstance(item, dict) for item in attempts):
        raise ValueError("corrective wave manifest must contain four attempts")
    return [dict(item) for item in attempts]


def _load_manifests(root: Path, baseline: Mapping[str, object]) -> list[dict[str, object]]:
    waves_dir = root / "waves"
    if waves_dir.is_symlink():
        raise ValueError("corrective waves directory is unsafe")
    paths = sorted(waves_dir.glob("wave-*.json")) if waves_dir.exists() else []
    manifests: list[dict[str, object]] = []
    for expected_index, path in enumerate(paths):
        manifest = _read_json(path, label="corrective wave manifest")
        if manifest.get("schema_version") != 1 or manifest.get("kind") != "corrective_rft_wave" or manifest.get("wave_index") != expected_index or manifest.get("baseline") != baseline or not isinstance(manifest.get("provider_evidence"), dict) or not isinstance(manifest.get("provider"), dict):
            raise ValueError("corrective wave manifest identity is stale or foreign")
        attempts = _manifest_attempts(manifest)
        if {item.get("worker_slot") for item in attempts} != set(range(4)):
            raise ValueError("corrective wave manifest worker slots are invalid")
        for item in attempts:
            worker_slot = item.get("worker_slot")
            category = item.get("category")
            if (
                item.get("attempt_id") != _attempt_id(expected_index, worker_slot)
                or item.get("episode_id") != item.get("attempt_id")
                or item.get("wave_index") != expected_index
                or category not in _CATEGORIES
                or item.get("release_stage") != "seen"
                or item.get("garment_name") != f"{CATEGORY_PREFIX[category]}_Seen_{(expected_index * 4 + worker_slot) % 10}"
                or item.get("seed") != expected_index * 4 + worker_slot + 100_000
                or item.get("command") != _trial_command(root, expected_index, worker_slot, category, baseline)
            ):
                raise ValueError("corrective wave manifest command or immutable seen-only attempt identity is invalid")
        manifests.append(manifest)
    attempt_ids = [item["attempt_id"] for manifest in manifests for item in _manifest_attempts(manifest)]
    if len(set(attempt_ids)) != len(attempt_ids):
        raise ValueError("corrective wave manifest attempt IDs must be unique")
    return manifests


def _read_snapshot(artifact: Path, name: str) -> dict[str, object]:
    path = artifact / "snapshots" / f"{name}.json"
    return _read_json(path, label=f"corrective {name} terminal artifact")


def _verify_policy_receipt(root: Path, episode_id: str, baseline: Mapping[str, object]) -> None:
    receipt = _read_json(root / f"policy-server-receipt-{episode_id}.json", label="corrective provider receipt")
    attempt = _attempt_from_id(episode_id)
    expected = {
        "episode_id": episode_id, "backend": "policy_server",
        "checkpoint_revision": baseline["parent_checkpoint_revision"],
        "checkpoint_digest": baseline["parent_checkpoint_artifact_sha256"],
        "code_revision": baseline["code_revision"], "image_identity": baseline["image_identity"],
        "policy_device": f"cuda:{attempt[1]}", "parity_stage": "server_cpu", "simulator_device": "cpu",
        "groot_revision": baseline["groot_revision"], "python_path": baseline["groot_python"],
        "policy_seed": attempt[0] * 4 + attempt[1] + 100_000, "port": 9100 + attempt[1],
    }
    if any(receipt.get(key) != value for key, value in expected.items()) or not isinstance(receipt.get("command"), list) or not receipt["command"]:
        raise ValueError("corrective provider receipt does not match immutable runtime")


def _attempt_from_id(attempt_id: str) -> tuple[int, int]:
    matched = re.fullmatch(r"corrective-wave-([0-9]{6})-worker-([0-3])", attempt_id)
    if matched is None:
        raise ValueError("corrective policy receipt has an invalid attempt identity")
    return int(matched.group(1)), int(matched.group(2))


def _attempt_category(root: Path, attempt_id: str) -> str:
    for manifest_path in (root / "waves").glob("wave-*.json"):
        manifest = _read_json(manifest_path, label="corrective wave manifest")
        for attempt in _manifest_attempts(manifest):
            if attempt.get("attempt_id") == attempt_id and isinstance(attempt.get("category"), str):
                return attempt["category"]
    raise ValueError("corrective policy receipt is not bound to a scheduled attempt")


def _terminal_receipt(root: Path, attempt: Mapping[str, object], baseline: Mapping[str, object], provider: Mapping[str, object]) -> dict[str, object]:
    attempt_id, episode_id = attempt.get("attempt_id"), attempt.get("episode_id")
    if not isinstance(attempt_id, str) or not isinstance(episode_id, str):
        raise ValueError("corrective manifest attempt identity is invalid")
    artifact = root / "raw" / attempt_id
    try:
        episode, _manifest = verify_episode_manifest(artifact)
    except ValueError as error:
        raise ValueError(f"corrective missing terminal artifact for {attempt_id}") from error
    reset, terminal = _read_snapshot(artifact, "reset"), _read_snapshot(artifact, "terminal")
    identity = episode.get("identity")
    if not isinstance(identity, Mapping) or episode.get("episode_id") != episode_id or episode.get("terminal_reason") is None:
        raise ValueError("corrective terminal episode identity is invalid")
    randomization = reset.get("randomization")
    if (
        identity.get("category") != attempt.get("category") or identity.get("garment_name") != attempt.get("garment_name")
        or identity.get("release_stage") != "seen" or episode.get("outcome") not in {"success", "failure", "timeout", "error"}
        or not isinstance(episode.get("accepted_success"), bool) or not _SHA256.fullmatch(str(episode.get("reset_hash")))
        or not isinstance(randomization, Mapping)
    ):
        raise ValueError("corrective terminal episode is not a valid seen-only attempt")
    _verify_policy_receipt(root, episode_id, baseline)
    return {
        "schema_version": 1, "attempt_id": attempt_id, "wave_index": attempt["wave_index"], "worker_slot": attempt["worker_slot"],
        "episode_id": episode_id, "category": attempt["category"], "release_stage": "seen", "outcome": episode["outcome"],
        "accepted_success": episode["accepted_success"], "reset_sha256": episode["reset_hash"],
        "randomization_sha256": _canonical_sha256(randomization), "hard_state_sha256": _canonical_sha256(terminal),
        "parent_checkpoint_repository": baseline["parent_checkpoint_repository"], "parent_checkpoint_revision": baseline["parent_checkpoint_revision"],
        "parent_checkpoint_artifact_sha256": baseline["parent_checkpoint_artifact_sha256"], "parent_checkpoint_step": baseline["parent_checkpoint_step"],
        "code_revision": baseline["code_revision"], "asset_revision": baseline["asset_revision"], "image_identity": baseline["image_identity"],
        "simulator_version": baseline["simulator_version"], "provider": dict(provider),
    }


def _quarantine(root: Path, wave_index: int, error: ValueError) -> None:
    _write_new_json(root / "quarantine" / f"wave-{wave_index:06d}.json", {
        "schema_version": 1, "kind": "corrective_rft_quarantine", "wave_index": wave_index, "reason": str(error),
    })


def _closed_attempts(root: Path, manifests: Iterable[Mapping[str, object]], baseline: Mapping[str, object]) -> tuple[list[dict[str, object]], bool]:
    closed: list[dict[str, object]] = []
    frozen_manifests = tuple(manifests)
    for position, manifest in enumerate(frozen_manifests):
        attempts = _manifest_attempts(manifest)
        wave_index = int(manifest["wave_index"])
        missing = any(not (root / "raw" / str(item.get("attempt_id"))).exists() for item in attempts)
        if missing:
            if position == len(frozen_manifests) - 1:
                return closed, False
            raise ValueError("corrective campaign has a non-terminal wave before a later wave")
        try:
            terminal = [_terminal_receipt(root, attempt, baseline, manifest["provider"]) for attempt in attempts]
            build_corrective_campaign_receipt([*closed, *terminal])
            fingerprints = [
                (item["reset_sha256"], item["randomization_sha256"], item["hard_state_sha256"])
                for item in [*closed, *terminal] if item["accepted_success"]
            ]
            if len(fingerprints) != len(set(fingerprints)):
                raise ValueError("corrective accepted successes must have unique state fingerprints")
        except ValueError as error:
            _quarantine(root, wave_index, error)
            raise
        for receipt in terminal:
            _write_new_json(root / "receipts" / f"{receipt['attempt_id']}.json", receipt)
        closed.extend(terminal)
    return closed, True


def _next_categories(attempts: list[dict[str, object]]) -> tuple[str, ...]:
    if not attempts:
        return _CATEGORIES
    return tuple(build_corrective_campaign_receipt(attempts)["next_wave_categories"])


def _attempt_id(wave_index: int, worker_slot: int) -> str:
    return f"corrective-wave-{wave_index:06d}-worker-{worker_slot}"


def _trial_command(root: Path, wave_index: int, worker_slot: int, category: str, baseline: Mapping[str, object]) -> list[str]:
    seed = wave_index * 4 + worker_slot + 100_000
    garment = f"{CATEGORY_PREFIX[category]}_Seen_{(wave_index * 4 + worker_slot) % 10}"
    attempt_id = _attempt_id(wave_index, worker_slot)
    return [
        str(baseline["controller_python"]), "scripts/run_groot_flywheel_trial.py", "--policy-path", str(baseline["policy_path"]),
        "--policy-revision-file", str(baseline["policy_revision_file"]), "--policy-repo", str(baseline["parent_checkpoint_repository"]),
        "--policy-step", "12000", "--code-revision", str(baseline["code_revision"]), "--asset-revision", str(baseline["asset_revision"]),
        "--simulator-version", str(baseline["simulator_version"]), "--release-assets-root", str(baseline["release_assets_root"]),
        "--policy-artifact-sha256", str(baseline["parent_checkpoint_artifact_sha256"]), "--image-identity", str(baseline["image_identity"]),
        "--groot-root", str(baseline["groot_root"]), "--groot-revision", str(baseline["groot_revision"]), "--groot-python", str(baseline["groot_python"]),
        "--execution-mode", "policy_server", "--parity-stage", "server_cpu", "--device", "cpu", "--policy-device", f"cuda:{worker_slot}", "--category", category,
        "--release-stage", "seen", "--garment", garment, "--seed", str(seed), "--episode-id", attempt_id, "--output-root", str(root),
        "--policy-server-port", str(9100 + worker_slot), "--policy-server-log", str(root / "workers" / f"worker-{worker_slot:02d}" / f"{attempt_id}.policy-server.log"),
        "--policy-server-readiness-timeout", "300", "--policy-server-request-timeout", "600", "--policy-server-termination-grace", "5", "--headless",
    ]


def _wave_manifest(root: Path, *, wave_index: int, categories: Sequence[str], baseline: Mapping[str, object], provider: Mapping[str, object], provider_evidence: Mapping[str, object]) -> dict[str, object]:
    if len(categories) != 4 or any(category not in _CATEGORIES for category in categories):
        raise ValueError("corrective next wave must contain exactly four supported seen categories")
    attempts = []
    for worker_slot, category in enumerate(categories):
        seed = wave_index * 4 + worker_slot + 100_000
        garment = f"{CATEGORY_PREFIX[category]}_Seen_{(wave_index * 4 + worker_slot) % 10}"
        attempt_id = _attempt_id(wave_index, worker_slot)
        command = _trial_command(root, wave_index, worker_slot, category, baseline)
        attempts.append({"attempt_id": attempt_id, "episode_id": attempt_id, "wave_index": wave_index, "worker_slot": worker_slot, "category": category, "garment_name": garment, "seed": seed, "release_stage": "seen", "command": command})
    return {"schema_version": 1, "kind": "corrective_rft_wave", "wave_index": wave_index, "baseline": dict(baseline), "provider": dict(provider), "provider_evidence": dict(provider_evidence), "attempts": attempts, "external_lifecycle_required": True, "provider_actions": "forbidden"}


def build_corrective_canary_manifest(wave_manifest: Path, *, output: Path) -> dict[str, object]:
    """Bind exactly one scheduled episode for the paid rollout canary."""
    wave = _read_json(wave_manifest, label="corrective wave manifest")
    attempts = _manifest_attempts(wave)
    if wave.get("kind") != "corrective_rft_wave" or len(attempts) != 4:
        raise ValueError("canary requires a complete four-worker corrective wave")
    canary = {"schema_version": 1, "kind": "corrective_rft_canary", "wave_index": wave["wave_index"],
              "source_wave_sha256": hashlib.sha256(wave_manifest.read_bytes()).hexdigest(), "baseline": wave["baseline"],
              "provider": wave["provider"], "attempt": attempts[0], "episode_count": 1,
              "external_lifecycle_required": True}
    _write_new_json(output, canary)
    return canary


def materialize_corrective_canary_attempt_receipt(
    canary_manifest: Path, *, synced_campaign_root: Path, output: Path,
) -> dict[str, object]:
    """Emit one publication-ready receipt after canonical canary verification."""
    canary = _read_json(canary_manifest, label="corrective canary manifest")
    attempt, baseline, provider = canary.get("attempt"), canary.get("baseline"), canary.get("provider")
    if (
        canary.get("schema_version") != 1 or canary.get("kind") != "corrective_rft_canary"
        or canary.get("episode_count") != 1 or not isinstance(attempt, Mapping)
        or not isinstance(baseline, Mapping) or not isinstance(provider, Mapping)
    ):
        raise ValueError("corrective canary manifest is invalid")
    receipt = _terminal_receipt(synced_campaign_root, attempt, baseline, provider)
    _write_new_json(output, receipt)
    return receipt


def run_corrective_campaign(args: argparse.Namespace, *, now_unix: int | None = None) -> dict[str, object]:
    """Validate completed waves and emit at most one external four-worker wave."""
    if not args.preflight:
        raise ValueError("corrective launcher is manifest-only; pass --preflight for free validation")
    root = args.campaign_root
    if root.is_symlink():
        raise ValueError("corrective campaign root is unsafe")
    now = int(time.time()) if now_unix is None else now_unix
    baseline, provider_evidence = _baseline(args.baseline), _provider(args.provider_receipt, now_unix=now)
    provider = _provider_facts(provider_evidence)
    manifests = _load_manifests(root, baseline)
    attempts, fully_closed = _closed_attempts(root, manifests, baseline)
    if not fully_closed:
        existing = _manifest_path(root, len(manifests) - 1)
        return {"status": "awaiting_external_wave", "launch_manifest": str(existing), "closed_wave_count": len(attempts) // 4, "external_lifecycle_required": True}
    if len(attempts) > MAX_ATTEMPTS:
        raise ValueError("corrective campaign exceeded 400 attempts")
    if attempts:
        try:
            selected = select_corrective_successes(attempts)
        except ValueError:
            selected = ()
        if len(selected) == sum(CATEGORY_SUCCESS_FLOORS.values()):
            receipt = build_corrective_campaign_receipt(attempts)
            _write_new_json(root / "campaign-receipt.json", receipt)
            return {"status": "collection_complete", "campaign_receipt": str(root / "campaign-receipt.json"), "closed_wave_count": len(attempts) // 4, "external_lifecycle_required": True}
    if len(attempts) == MAX_ATTEMPTS:
        return {"status": "hard_max_reached", "closed_wave_count": len(attempts) // 4, "external_lifecycle_required": True}
    wave_index = len(manifests)
    if manifests and provider_evidence["evidence_id"] == manifests[-1]["provider_evidence"].get("evidence_id"):
        raise ValueError("corrective next wave requires fresh provider evidence")
    manifest = _wave_manifest(root, wave_index=wave_index, categories=_next_categories(attempts), baseline=baseline, provider=provider, provider_evidence=provider_evidence)
    path = _manifest_path(root, wave_index)
    _write_new_json(path, manifest)
    return {"status": "wave_scheduled", "launch_manifest": str(path), "closed_wave_count": len(attempts) // 4, "external_lifecycle_required": True}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True, help="immutable approved-step12000 baseline JSON")
    parser.add_argument("--provider-receipt", type=Path, required=True, help="external lifecycle on-demand 4xRTX3090 receipt")
    parser.add_argument("--preflight", action="store_true", help="free validation and manifest generation only")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = run_corrective_campaign(build_parser().parse_args(argv))
    except ValueError as error:
        print(f"corrective campaign validation error: {error}")
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
