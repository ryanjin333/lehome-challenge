"""Tracked release coverage for the external corrective-rollout launcher."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "source" / "lehome"))
SPEC = importlib.util.spec_from_file_location(
    "corrective_campaign_under_test", REPOSITORY / "scripts" / "run_groot_corrective_campaign.py"
)
assert SPEC is not None and SPEC.loader is not None
CAMPAIGN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CAMPAIGN
SPEC.loader.exec_module(CAMPAIGN)

from lehome_train.flywheel.corrective import (  # noqa: E402
    APPROVED_PARENT_ARTIFACT_SHA256, APPROVED_PARENT_REPOSITORY, APPROVED_PARENT_REVISION,
)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _baseline() -> dict[str, object]:
    return {"schema_version": 1, "kind": "corrective_rft_baseline", "parent_checkpoint_repository": APPROVED_PARENT_REPOSITORY, "parent_checkpoint_revision": APPROVED_PARENT_REVISION, "parent_checkpoint_artifact_sha256": APPROVED_PARENT_ARTIFACT_SHA256, "parent_checkpoint_step": 12000, "code_revision": "c" * 40, "asset_revision": "a" * 40, "image_identity": CAMPAIGN.APPROVED_IMAGE_DIGEST, "rollout_image": CAMPAIGN.APPROVED_IMAGE_REPOSITORY + "@" + CAMPAIGN.APPROVED_IMAGE_DIGEST, "controller_python": "/isaac/python", "simulator_version": "5.1.0.0", "policy_path": "/model/step-12000", "policy_revision_file": "/model/revision", "release_assets_root": "/assets", "groot_root": "/groot", "groot_revision": "b" * 40, "groot_python": "/venv/bin/python"}


def _provider(*, evidence_id: str = "evidence-1", expiry: int = 2_000_000_000, cost: float = 1.75) -> dict[str, object]:
    return {"schema_version": 1, "kind": "external_provider_offer_evidence", "evidence_id": evidence_id, "queried_at_unix": 1_000_000_000, "expires_at_unix": expiry, "source_snapshot_path": f"provider-{evidence_id}.json", "source_snapshot_sha256": "0" * 64, "source_response_sha256": "1" * 64, "rental_kind": "on-demand", "instance_hourly_cost_usd": 1.25, "account_hourly_total_usd": cost, "offer_id": 40705900, "gpu_name": "RTX 3090", "num_gpus": 4}


def _provider_file(path: Path, value: dict[str, object]) -> dict[str, object]:
    snapshot = path.parent / str(value["source_snapshot_path"])
    _json(snapshot, {"offer_id": value["offer_id"], "external": True})
    value = {**value, "source_snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest()}
    _json(path, value)
    return value


def _args(root: Path, baseline: Path, provider: Path):
    return CAMPAIGN.build_parser().parse_args(["--campaign-root", str(root), "--baseline", str(baseline), "--provider-receipt", str(provider), "--preflight"])


def _setup(tmp_path: Path):
    root, baseline, provider = tmp_path / "run", tmp_path / "baseline.json", tmp_path / "provider.json"
    base, offer = _baseline(), _provider()
    _json(baseline, base)
    return root, baseline, provider, base, _provider_file(provider, offer)


def test_trial_uses_pinned_isaac_controller_and_separate_groot_child(tmp_path: Path) -> None:
    baseline = _baseline()
    command = CAMPAIGN._trial_command(tmp_path, 0, 0, "top_long", baseline)
    assert command[0] == baseline["controller_python"]
    assert command[command.index("--groot-python") + 1] == baseline["groot_python"]
    assert command[command.index("--policy-server-readiness-timeout") + 1] == "300"
    assert command[command.index("--policy-server-request-timeout") + 1] == "600"


def test_canary_manifest_binds_one_of_a_complete_wave(tmp_path: Path) -> None:
    baseline = _baseline()
    wave = CAMPAIGN._wave_manifest(tmp_path, wave_index=0, categories=("top_long", "top_short", "pant_long", "pant_short"), baseline=baseline, provider={"rental_kind": "on-demand"}, provider_evidence={})
    source = tmp_path / "wave.json"; _json(source, wave)
    canary = CAMPAIGN.build_corrective_canary_manifest(source, output=tmp_path / "canary.json")
    assert canary["episode_count"] == 1 and canary["attempt"] == wave["attempts"][0]


def test_canary_materializes_one_canonical_publication_receipt(tmp_path: Path) -> None:
    baseline = _baseline()
    wave = CAMPAIGN._wave_manifest(tmp_path, wave_index=0, categories=("top_long", "top_short", "pant_long", "pant_short"), baseline=baseline, provider={"rental_kind": "on-demand", "offer_id": 7}, provider_evidence={})
    wave_path = tmp_path / "wave.json"; _json(wave_path, wave)
    canary_path = tmp_path / "canary.json"; canary = CAMPAIGN.build_corrective_canary_manifest(wave_path, output=canary_path)
    sync = tmp_path / "sync"; _complete(sync, canary["attempt"], baseline)
    receipt = CAMPAIGN.materialize_corrective_canary_attempt_receipt(canary_path, synced_campaign_root=sync, output=tmp_path / "attempt.json")
    assert receipt["attempt_id"] == canary["attempt"]["attempt_id"] and receipt["accepted_success"] is True


def _snapshot(*, garment: str, randomization: dict[str, object]) -> dict[str, object]:
    """Exact production ``Snapshot.to_dict()`` shape; no launcher-only fields."""
    return {
        "schema_version": 1, "robot_position": [0.0] * 12, "robot_velocity": [0.0] * 12,
        "cloth_position": [[0.0, 0.0, 0.0]], "cloth_velocity": [[0.0, 0.0, 0.0]],
        "rng_state": {"seed": 1}, "garment_name": garment, "randomization": randomization,
        "scene_state": {},
    }


def _complete(
    root: Path,
    attempt: dict[str, object],
    baseline: dict[str, object],
    *,
    terminal_reason: str = "success",
) -> None:
    attempt_id = str(attempt["attempt_id"])
    artifact = root / "raw" / attempt_id
    reset = _snapshot(garment=str(attempt["garment_name"]), randomization={"seed": attempt["seed"]})
    terminal = _snapshot(garment=str(attempt["garment_name"]), randomization={"seed": attempt["seed"]})
    terminal["scene_state"] = {"terminal": attempt_id}
    episode = {"episode_id": attempt_id, "terminal_reason": terminal_reason, "outcome": "success", "accepted_success": True, "reset_hash": _hash(reset), "identity": {"episode_id": attempt_id, "policy_repo": baseline["parent_checkpoint_repository"], "policy_revision": baseline["parent_checkpoint_revision"], "policy_step": 12000, "code_revision": baseline["code_revision"], "asset_revision": baseline["asset_revision"], "simulator_version": baseline["simulator_version"], "garment_name": attempt["garment_name"], "category": attempt["category"], "release_stage": "seen", "seed": attempt["seed"], "instruction": "fold", "strategy": "mild"}}
    _json(artifact / "episode.json", episode); _json(artifact / "snapshots" / "reset.json", reset); _json(artifact / "snapshots" / "terminal.json", terminal)
    manifest = {p.relative_to(artifact).as_posix(): {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "size": p.stat().st_size} for p in sorted(artifact.rglob("*.json"))}
    _json(artifact / "SHA256SUMS.json", manifest)
    command = attempt["command"]
    _json(root / f"policy-server-receipt-{attempt_id}.json", {"episode_id": attempt_id, "backend": "policy_server", "checkpoint_revision": baseline["parent_checkpoint_revision"], "checkpoint_digest": baseline["parent_checkpoint_artifact_sha256"], "code_revision": baseline["code_revision"], "image_identity": baseline["image_identity"], "policy_device": f"cuda:{attempt['worker_slot']}", "parity_stage": "server_cpu", "simulator_device": "cpu", "groot_revision": baseline["groot_revision"], "python_path": baseline["groot_python"], "policy_seed": attempt["seed"], "port": 9100 + attempt["worker_slot"], "command": command})


def test_preflight_emits_seen_four_worker_server_cpu_manifest(tmp_path: Path) -> None:
    root, baseline, provider, _, offer = _setup(tmp_path)
    result = CAMPAIGN.run_corrective_campaign(_args(root, baseline, provider))
    manifest = json.loads(Path(result["launch_manifest"]).read_text())
    assert result["status"] == "wave_scheduled" and manifest["provider_evidence"] == offer
    assert {a["worker_slot"] for a in manifest["attempts"]} == set(range(4))
    assert all(a["release_stage"] == "seen" and "Unseen" not in a["garment_name"] and "--parity-stage server_cpu" in " ".join(a["command"]) for a in manifest["attempts"])


def test_resume_closes_verified_wave_with_fresh_evidence_and_strict_receipts(tmp_path: Path) -> None:
    root, baseline, provider, base, _ = _setup(tmp_path)
    first = CAMPAIGN.run_corrective_campaign(_args(root, baseline, provider))
    for attempt in json.loads(Path(first["launch_manifest"]).read_text())["attempts"]: _complete(root, attempt, base)
    _provider_file(provider, _provider(evidence_id="evidence-2"))
    result = CAMPAIGN.run_corrective_campaign(_args(root, baseline, provider))
    assert result["closed_wave_count"] == 1 and len(list((root / "receipts").glob("*.json"))) == 4


def test_terminal_receipt_does_not_accept_success_at_horizon(tmp_path: Path) -> None:
    root, baseline, provider, base, _ = _setup(tmp_path)
    first = CAMPAIGN.run_corrective_campaign(_args(root, baseline, provider))
    attempts = json.loads(Path(first["launch_manifest"]).read_text())["attempts"]
    _complete(root, attempts[0], base, terminal_reason="horizon")
    for attempt in attempts[1:]:
        _complete(root, attempt, base)
    _provider_file(provider, _provider(evidence_id="evidence-2"))

    CAMPAIGN.run_corrective_campaign(_args(root, baseline, provider))

    receipt = json.loads((root / "receipts" / f"{attempts[0]['attempt_id']}.json").read_text())
    assert receipt["accepted_success"] is False


def test_resume_ignores_canary_sidecar_and_schedules_next_canonical_wave(tmp_path: Path) -> None:
    root, baseline, provider, base, _ = _setup(tmp_path)
    first = CAMPAIGN.run_corrective_campaign(_args(root, baseline, provider))
    wave_path = Path(first["launch_manifest"])
    wave = json.loads(wave_path.read_text(encoding="utf-8"))
    for attempt in wave["attempts"]:
        _complete(root, attempt, base)
    canary_path = root / "waves" / "wave-000000-canary.json"
    canary = CAMPAIGN.build_corrective_canary_manifest(wave_path, output=canary_path)
    _provider_file(provider, _provider(evidence_id="evidence-2"))
    result = CAMPAIGN.run_corrective_campaign(_args(root, baseline, provider))
    assert result["status"] == "wave_scheduled"
    assert Path(result["launch_manifest"]).name == "wave-000001.json"
    assert json.loads(canary_path.read_text(encoding="utf-8")) == canary


@pytest.mark.parametrize("broken, message", [("terminal", "missing terminal"), ("policy", "provider receipt")])
def test_resume_quarantines_missing_terminal_and_provider_mismatch(tmp_path: Path, broken: str, message: str) -> None:
    root, baseline, provider, base, _ = _setup(tmp_path); first = CAMPAIGN.run_corrective_campaign(_args(root, baseline, provider)); attempts = json.loads(Path(first["launch_manifest"]).read_text())["attempts"]
    for attempt in attempts: _complete(root, attempt, base)
    target = attempts[0]["attempt_id"]
    if broken == "terminal": (root / "raw" / target / "snapshots" / "terminal.json").unlink()
    else: _json(root / f"policy-server-receipt-{target}.json", {"episode_id": target})
    with pytest.raises(ValueError, match=message): CAMPAIGN.run_corrective_campaign(_args(root, baseline, provider))
    assert (root / "quarantine" / "wave-000000.json").is_file()


def test_rejects_over_cap_before_wave_write(tmp_path: Path) -> None:
    root, baseline, provider, _, _ = _setup(tmp_path); _provider_file(provider, _provider(cost=2.01))
    with pytest.raises(ValueError, match="shared \\$2/hr"): CAMPAIGN.run_corrective_campaign(_args(root, baseline, provider))
    assert not (root / "waves").exists()


def test_resume_rejects_mutated_deterministic_command(tmp_path: Path) -> None:
    root, baseline, provider, _, _ = _setup(tmp_path); first = CAMPAIGN.run_corrective_campaign(_args(root, baseline, provider)); path = Path(first["launch_manifest"]); manifest = json.loads(path.read_text()); manifest["attempts"][0]["command"].remove("--headless"); _json(path, manifest)
    with pytest.raises(ValueError, match="command"): CAMPAIGN.run_corrective_campaign(_args(root, baseline, provider))


def test_rejects_expired_and_snapshot_unbound_provider_evidence(tmp_path: Path) -> None:
    root, baseline, provider, _, _ = _setup(tmp_path); _provider_file(provider, _provider(expiry=999))
    with pytest.raises(ValueError, match="expired"): CAMPAIGN.run_corrective_campaign(_args(root, baseline, provider), now_unix=1_000)
    invalid = _provider_file(provider, {**_provider(), "queried_at_unix": 999}); _json(provider, {**invalid, "source_snapshot_sha256": "0" * 64})
    with pytest.raises(ValueError, match="snapshot"): CAMPAIGN.run_corrective_campaign(_args(root, baseline, provider), now_unix=1_000)


def test_next_wave_requires_fresh_provider_evidence(tmp_path: Path) -> None:
    root, baseline, provider, base, _ = _setup(tmp_path); first = CAMPAIGN.run_corrective_campaign(_args(root, baseline, provider))
    for attempt in json.loads(Path(first["launch_manifest"]).read_text())["attempts"]: _complete(root, attempt, base)
    with pytest.raises(ValueError, match="fresh provider evidence"): CAMPAIGN.run_corrective_campaign(_args(root, baseline, provider))


def test_terminal_fingerprint_accepts_exact_snapshot_schema_without_launcher_hard_state(tmp_path: Path) -> None:
    root, baseline, provider, base, _ = _setup(tmp_path)
    first = CAMPAIGN.run_corrective_campaign(_args(root, baseline, provider))
    for attempt in json.loads(Path(first["launch_manifest"]).read_text())["attempts"]:
        _complete(root, attempt, base)
    _provider_file(provider, _provider(evidence_id="evidence-2"))
    assert CAMPAIGN.run_corrective_campaign(_args(root, baseline, provider))["closed_wave_count"] == 1
