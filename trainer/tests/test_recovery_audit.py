from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

from lehome_train.io import canonical_json_sha256


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")


def _package_digest(root: Path) -> str:
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.relative_to(root).as_posix() != "SHA256SUMS.json":
            rows.append({
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "byte_size": path.stat().st_size,
            })
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _round(
    tmp_path: Path,
    *,
    round_id: str,
    episodes: dict[str, tuple[str, list[float], float]],
    success_tail: int = 2,
) -> tuple[Path, Path, Path]:
    from lehome.flywheel.artifacts import build_sha256_manifest
    from lehome_train.io import canonical_json_sha256

    root = tmp_path / round_id / "accepted"
    receipts = tmp_path / round_id / "receipts"
    digests: dict[str, str] = {}
    revisions: dict[str, str] = {}
    for offset, (attempt_id, (category, rewards, state_base)) in enumerate(sorted(episodes.items())):
        package = root / attempt_id
        raw = package / "raw" / attempt_id
        raw.mkdir(parents=True)
        annotations = [
            {
                "step": step,
                "monotonic_ns": 1000 + step,
                "wall_time_ns": 2000 + step,
                "reward": reward,
                "success": step >= len(rewards) - success_tail,
                "state": [state_base + step / 1000] * 12,
                "action": [float(step)] * 12,
                "action_source": "policy",
                "segment": 0,
                "policy_request_id": f"request-{attempt_id}-{step // 16}",
                "policy_chunk_offset": step % 16,
                "expert_sequence": None,
                "expert_sample_age_ms": None,
            }
            for step, reward in enumerate(rewards)
        ]
        (raw / "annotations.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in annotations), encoding="utf-8",
        )
        _write(raw / "snapshots" / "reset.json", {
            "schema_version": 2,
            "robot_position": [state_base] * 12,
            "robot_velocity": [0.0] * 12,
            "cloth_position": [[0.0, 0.0, 0.0]],
            "cloth_velocity": [[0.0, 0.0, 0.0]],
            "rng_state": {"seed": offset},
            "garment_name": f"{category}-seen-0",
            "randomization": {"strategy": "canonical"},
            "scene_state": {},
            "cloth_state_authority": "physx_cloth_view_world_v1",
        })
        for step in range(16, len(annotations), 16):
            _write(raw / "snapshots" / "continuations" / f"{step:06d}.json", {
                "schema_version": 2,
                "robot_position": [state_base + step / 1000] * 12,
                "robot_velocity": [0.0] * 12,
                "cloth_position": [[0.0, 0.0, 0.0]],
                "cloth_velocity": [[0.0, 0.0, 0.0]],
                "rng_state": {"seed": offset},
                "garment_name": f"{category}-seen-0",
                "randomization": {"strategy": "canonical", "continuation_step": step},
                "scene_state": {},
                "cloth_state_authority": "physx_cloth_view_world_v1",
            })
        _write(raw / "episode.json", {
            "episode_id": attempt_id, "accepted_success": True, "outcome": "success",
            "terminal_reason": "success", "bc_target_count": 0,
            "provenance": {
                "execution_backend": "policy_server", "execution_mode": "policy_server",
                "parity_stage": "persistent_collection", "policy_artifact_sha256": "c" * 64,
                "policy_device": "cuda:0", "simulator_device": "cuda:0",
            },
            "identity": {
                "release_stage": "seen", "category": category,
                "garment_name": f"{category}-seen-0", "seed": 50_000 + offset,
            },
        })
        _write(raw / "SHA256SUMS.json", build_sha256_manifest(raw))
        _write(package / "flywheel-manifest.json", {"attempt_id": attempt_id})
        _write(package / "worker-receipt.json", {"attempt_id": attempt_id})
        digest = _package_digest(package)
        revision = f"{offset + 10:040x}"
        digests[attempt_id], revisions[attempt_id] = digest, revision
        _write(receipts / f"{attempt_id}.sync.json", {
            "schema_version": 1, "attempt_id": attempt_id,
            "repository": "ryanjin333/lehome-groot-n17-rollouts", "round_id": round_id,
            "remote_prefix": f"rollout-rounds/{round_id}/{attempt_id}", "publication_ref": "main",
            "immutable_revision": revision, "entry_count": 2, "episode_sha256": digest,
            "readback_verified": True,
        })
    body = {
        "round_id": round_id, "repository": "ryanjin333/lehome-groot-n17-rollouts",
        "episode_sha256s": digests, "immutable_revisions": revisions,
    }
    seal = tmp_path / round_id / "seal.json"
    _write(seal, {
        "schema_version": 2, "kind": "rollout_round_seal", **body,
        "episode_count": len(digests), "readback_verified": True,
        "seal_sha256": canonical_json_sha256(body),
    })
    return root, receipts, seal


def _reseal_episode(root: Path, receipts: Path, seal: Path, attempt_id: str) -> None:
    """Rebuild fixture provenance after a semantic-only annotation mutation."""
    from lehome.flywheel.artifacts import build_sha256_manifest
    from lehome_train.io import canonical_json_sha256

    package = root / attempt_id
    raw = package / "raw" / attempt_id
    (raw / "SHA256SUMS.json").unlink()
    _write(raw / "SHA256SUMS.json", build_sha256_manifest(raw))
    digest = _package_digest(package)
    receipt = receipts / f"{attempt_id}.sync.json"
    receipt_value = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_value["episode_sha256"] = digest
    _write(receipt, receipt_value)
    seal_value = json.loads(seal.read_text(encoding="utf-8"))
    seal_value["episode_sha256s"][attempt_id] = digest
    body = {
        "round_id": seal_value["round_id"], "repository": seal_value["repository"],
        "episode_sha256s": seal_value["episode_sha256s"],
        "immutable_revisions": seal_value["immutable_revisions"],
    }
    seal_value["seal_sha256"] = canonical_json_sha256(body)
    _write(seal, seal_value)


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [item for child in value.values() for item in _strings(child)]
    if isinstance(value, list):
        return [item for child in value for item in _strings(child)]
    return []


def test_audit_selects_only_h16_corrective_windows_and_reports_shortfalls(tmp_path: Path) -> None:
    from lehome_train.groot.recovery_audit import audit_successful_recoveries
    from lehome_train.io import canonical_json_sha256

    # Only the drawdown episode has adverse evidence; the monotonic success is excluded.
    recovery = [0.0] * 16 + [0.4, 0.4, 0.2] + [0.2] * 13 + [0.35, 0.4, 0.5] + [0.5] * 17
    easy = [step / 100 for step in range(24)]
    root, receipts, seal = _round(tmp_path, round_id="original-round", episodes={
        "recover-top": ("top_long", recovery, 1.0),
        "easy-top": ("top_long", easy, 2.0),
    })
    output = tmp_path / "result.json"

    result = audit_successful_recoveries(
        accepted_roots=(root,), receipt_roots=(receipts,), round_seal_paths=(seal,), output_path=output,
    )

    document = json.loads(output.read_text(encoding="utf-8"))
    assert result["ready"] is False
    assert document["semantic_sha256"] == canonical_json_sha256({
        key: value for key, value in document.items() if key != "semantic_sha256"
    })
    assert document["per_category_counts"] == {
        "pant_long": 0, "pant_short": 0, "top_long": 1, "top_short": 0,
    }
    assert document["shortfalls"]["top_long"] == 4
    assert [row["source_episode_id"] for row in document["selected_recoveries"]] == ["recover-top"]
    selected = document["selected_recoveries"][0]
    assert selected["recovery_event"]["kind"] == "reward_drawdown"
    assert selected["h16_ranges"] == [{"start": 18, "stop": 34, "frame_ids": list(range(18, 34))}]
    assert not (root / "recover-top/SHA256SUMS.json").exists()
    assert selected["source_artifacts"]["package_sync_digest"] == selected["source_episode_digest"]
    assert set(selected["source_artifacts"]) == {
        "package_sync_digest", "flywheel_manifest_sha256", "worker_receipt_sha256",
        "raw_checksum_manifest_sha256", "episode_manifest_sha256", "annotations_sha256",
    }
    assert {row["reason"] for row in document["exclusions"]} == {"no_meaningful_recovery"}
    assert output.with_suffix(".json.sha256").read_text(encoding="ascii").strip() == hashlib.sha256(output.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update({"kind": "rollout_round_seal"}),
        lambda value: value.update({"source_only": False}),
        lambda value: value.update({"readback_verified": False}),
        lambda value: value.update({"episode_count": 0}),
        lambda value: value["episode_sha256s"].update({"second": "a" * 64}) or value["immutable_revisions"].update({"second": "b" * 40}),
        lambda value: value.update({"immutable_revisions": {"different": "b" * 40}}),
        lambda value: value.update({"episode_sha256s": {"one": "z" * 64}}),
        lambda value: value.update({"immutable_revisions": {"one": "z" * 40}}),
        lambda value: value.update({"round_id": ""}),
        lambda value: value.update({"round_id": "../unsafe"}),
        lambda value: value.update({"repository": ""}),
        lambda value: value.update({"repository": "owner/../unsafe"}),
        lambda value: value.update({"unexpected": True}),
        lambda value: value.pop("repository"),
    ],
)
def test_source_only_envelope_is_fail_closed_and_not_a_strict_round_seal(tmp_path: Path, mutate: object) -> None:
    from lehome_train.groot.recovery_audit import _audit_source_seal
    from lehome_train.groot.rollout_source_adapter import _seal
    assert callable(mutate)
    attempt_id = "1" * 64
    body = {"schema_version": 1, "kind": "snapshot_source_bootstrap_envelope", "round_id": "snapshot-source-bootstrap-abc-unsealed-source", "repository": "ryanjin333/lehome-groot-n17-rollouts", "episode_count": 1, "episode_sha256s": {attempt_id: "a" * 64}, "immutable_revisions": {attempt_id: "b" * 40}, "readback_verified": True, "source_only": True}
    from lehome_train.io import canonical_json_sha256
    mutate(body)
    body["envelope_sha256"] = canonical_json_sha256(body)
    path = tmp_path / "source-envelope.json"; _write(path, body)
    with pytest.raises(ValueError):
        _audit_source_seal(path)
    with pytest.raises(ValueError):
        _seal(path)


def test_source_only_envelope_rejects_raw_checksum_tampering_but_a_canonical_one_is_audit_only(tmp_path: Path) -> None:
    from lehome_train.groot.recovery_audit import _audit_source_seal
    from lehome_train.groot.rollout_source_adapter import _seal
    from lehome_train.io import canonical_json_sha256

    attempt_id = "1" * 64
    body = {"schema_version": 1, "kind": "snapshot_source_bootstrap_envelope", "round_id": "snapshot-source-bootstrap-abc-unsealed-source", "repository": "ryanjin333/lehome-groot-n17-rollouts", "episode_count": 1, "episode_sha256s": {attempt_id: "a" * 64}, "immutable_revisions": {attempt_id: "b" * 40}, "readback_verified": True, "source_only": True}
    body["envelope_sha256"] = canonical_json_sha256(body)
    path = tmp_path / "source-envelope.json"; _write(path, body)
    assert _audit_source_seal(path)["seal_sha256"] == body["envelope_sha256"]
    with pytest.raises(ValueError):
        _seal(path)


@pytest.mark.parametrize("payload", [
    '{"schema_version":1,"schema_version":1}',
    '{"schema_version":NaN}',
])
def test_source_only_envelope_rejects_duplicate_and_nonfinite_json_before_mapping_access(
    tmp_path: Path, payload: str,
) -> None:
    from lehome_train.groot.recovery_audit import _audit_source_seal

    path = tmp_path / "source-envelope.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="source envelope"):
        _audit_source_seal(path)


@pytest.mark.parametrize("schema_version", [True, 1.0])
def test_source_only_envelope_requires_an_exact_integer_schema_version(
    tmp_path: Path, schema_version: object,
) -> None:
    from lehome_train.groot.recovery_audit import _audit_source_seal
    from lehome_train.io import canonical_json_sha256

    attempt_id = "1" * 64
    body = {"schema_version": schema_version, "kind": "snapshot_source_bootstrap_envelope", "round_id": "snapshot-source-bootstrap-abc-unsealed-source", "repository": "ryanjin333/lehome-groot-n17-rollouts", "episode_count": 1, "episode_sha256s": {attempt_id: "a" * 64}, "immutable_revisions": {attempt_id: "b" * 40}, "readback_verified": True, "source_only": True}
    body["envelope_sha256"] = canonical_json_sha256(body)
    path = tmp_path / "source-envelope.json"
    _write(path, body)

    with pytest.raises(ValueError, match="source envelope"):
        _audit_source_seal(path)


def test_source_only_audit_envelope_admits_a_bounded_multi_source_set_but_remains_non_trainable(tmp_path: Path) -> None:
    from lehome_train.groot.recovery_audit import _audit_source_seal
    from lehome_train.groot.rollout_source_adapter import _seal

    path = tmp_path / "sources.json"
    first_id, second_id = "1" * 64, "2" * 64
    body = {"schema_version": 1, "kind": "snapshot_source_bootstrap_envelope", "round_id": "snapshot-source-bootstrap-abc-unsealed-source", "repository": "ryanjin333/lehome-groot-n17-rollouts", "episode_count": 2, "episode_sha256s": {first_id: "a" * 64, second_id: "b" * 64}, "immutable_revisions": {first_id: "c" * 40, second_id: "d" * 40}, "readback_verified": True, "source_only": True}
    body["envelope_sha256"] = canonical_json_sha256(body)
    path.write_text(json.dumps(body), encoding="utf-8")

    assert _audit_source_seal(path)["episode_count"] == 2
    with pytest.raises(ValueError):
        _seal(path)
    body["episode_sha256s"][first_id] = "c" * 64
    _write(path, body)
    with pytest.raises(ValueError, match="checksum"):
        _audit_source_seal(path)


def test_source_only_audit_envelope_admits_150_sources_but_rejects_151(tmp_path: Path) -> None:
    from lehome_train.groot.recovery_audit import _audit_source_seal
    from lehome_train.io import canonical_json_sha256

    episode_ids = [hashlib.sha256(f"episode-{index}".encode()).hexdigest() for index in range(151)]
    body = {
        "schema_version": 1,
        "kind": "snapshot_source_bootstrap_envelope",
        "round_id": "snapshot-source-bootstrap-abc-unsealed-source",
        "repository": "ryanjin333/lehome-groot-n17-rollouts",
        "episode_count": 150,
        "episode_sha256s": {episode_id: "a" * 64 for episode_id in episode_ids[:150]},
        "immutable_revisions": {episode_id: "b" * 40 for episode_id in episode_ids[:150]},
        "readback_verified": True,
        "source_only": True,
    }
    body["envelope_sha256"] = canonical_json_sha256(body)
    path = tmp_path / "source-envelope.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    assert _audit_source_seal(path)["episode_count"] == 150

    body["episode_count"] = 151
    body["episode_sha256s"][episode_ids[150]] = "a" * 64
    body["immutable_revisions"][episode_ids[150]] = "b" * 40
    body["envelope_sha256"] = canonical_json_sha256({key: value for key, value in body.items() if key != "envelope_sha256"})
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="source envelope"):
        _audit_source_seal(path)


@pytest.mark.parametrize("episode_id", ["not-a-task-ledger-id", "../unsafe", "A" * 64])
def test_source_only_audit_envelope_requires_canonical_task_ledger_attempt_ids(
    tmp_path: Path, episode_id: str,
) -> None:
    from lehome_train.groot.recovery_audit import _audit_source_seal

    body = {
        "schema_version": 1, "kind": "snapshot_source_bootstrap_envelope",
        "round_id": "snapshot-source-bootstrap-abc-unsealed-source",
        "repository": "ryanjin333/lehome-groot-n17-rollouts", "episode_count": 1,
        "episode_sha256s": {episode_id: "a" * 64},
        "immutable_revisions": {episode_id: "b" * 40},
        "readback_verified": True, "source_only": True,
    }
    body["envelope_sha256"] = canonical_json_sha256(body)
    path = tmp_path / "source-envelope.json"; _write(path, body)

    with pytest.raises(ValueError, match="lineage"):
        _audit_source_seal(path)


def _source_only_envelope(path: Path, strict_seal: Path) -> Path:
    from lehome_train.io import canonical_json_sha256

    strict = json.loads(strict_seal.read_text(encoding="utf-8"))
    body = {
        "schema_version": 1, "kind": "snapshot_source_bootstrap_envelope",
        "round_id": strict["round_id"], "repository": strict["repository"],
        "episode_count": 1, "episode_sha256s": strict["episode_sha256s"],
        "immutable_revisions": strict["immutable_revisions"],
        "readback_verified": True, "source_only": True,
    }
    body["envelope_sha256"] = canonical_json_sha256(body)
    _write(path, body)
    return path


def test_audit_rejects_duplicate_source_envelope_rounds_and_cross_run_episode_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lehome_train.groot.recovery_audit import audit_successful_recoveries

    materialize = types.ModuleType("lehome_train.flywheel.materialize")
    materialize._is_autonomous_policy_success = lambda _: True  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lehome_train.flywheel.materialize", materialize)
    rewards = [0.0] * 16 + [0.4, 0.1] + [0.1] * 14 + [0.5] * 16
    shared_attempt = "d" * 64
    root, receipts, strict = _round(tmp_path, round_id="source-round-one", episodes={shared_attempt: ("top_long", rewards, 1.0)})
    envelope = _source_only_envelope(tmp_path / "source-one-envelope.json", strict)
    with pytest.raises(ValueError, match="round seal ID collision"):
        audit_successful_recoveries(
            accepted_roots=(root, root), receipt_roots=(receipts, receipts),
            round_seal_paths=(envelope, envelope), output_path=tmp_path / "duplicate-round.json",
        )
    second_root, second_receipts, second_strict = _round(tmp_path, round_id="source-round-two", episodes={shared_attempt: ("top_long", rewards, 2.0)})
    second_envelope = _source_only_envelope(tmp_path / "source-two-envelope.json", second_strict)
    with pytest.raises(ValueError, match="cross-round episode-ID collision"):
        audit_successful_recoveries(
            accepted_roots=(root, second_root), receipt_roots=(receipts, second_receipts),
            round_seal_paths=(envelope, second_envelope), output_path=tmp_path / "duplicate-episode.json",
        )


def test_audit_advances_an_adverse_start_inside_a_cached_chunk_to_the_next_fresh_policy_boundary(tmp_path: Path) -> None:
    """A reset continuation may start only at an authenticated offset-zero action."""
    from lehome_train.groot.recovery_audit import audit_successful_recoveries

    # The reward drawdown starts at step 18, inside request 1.  Its recovery is
    # confirmed after step 32, so that step is the only usable new-request boundary.
    rewards = [0.0] * 16 + [0.4, 0.4, 0.2] + [0.2] * 13 + [0.35, 0.4, 0.5] + [0.5] * 4
    root, receipts, seal = _round(tmp_path, round_id="original-round", episodes={
        "recover-boundary": ("top_long", rewards, 3.0),
    })
    annotations = root / "recover-boundary/raw/recover-boundary/annotations.jsonl"
    rows = [json.loads(line) for line in annotations.read_text(encoding="utf-8").splitlines()]
    rows[32]["state"] = [3.031] * 12
    annotations.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    _reseal_episode(root, receipts, seal, "recover-boundary")
    output = tmp_path / "result.json"

    audit_successful_recoveries(
        accepted_roots=(root,), receipt_roots=(receipts,), round_seal_paths=(seal,), output_path=output,
    )

    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["schema_version"] == 4
    selected = document["selected_recoveries"]
    assert len(selected) == 1
    continuation = selected[0]["continuation_start"]
    assert continuation["annotation_index"] == 32
    assert continuation["step"] == 32
    assert continuation["policy_chunk_offset"] == 0
    assert continuation["policy_request_id"] == "request-recover-boundary-2"
    assert continuation["state"] == [3.032] * 12
    assert continuation["snapshot_continuation_step"] == 32
    assert continuation["snapshot_schema_version"] == 2
    assert continuation["snapshot_cloth_state_authority"] == "physx_cloth_view_world_v1"
    assert selected[0]["fingerprint"] == continuation["state_fingerprint"]


def test_audit_rejects_a_snapshot_whose_declared_continuation_step_is_wrong(tmp_path: Path) -> None:
    from lehome_train.groot.recovery_audit import audit_successful_recoveries

    rewards = [0.0] * 16 + [0.4, 0.4, 0.2] + [0.2] * 13 + [0.35, 0.4, 0.5] + [0.5] * 4
    root, receipts, seal = _round(tmp_path, round_id="original-round", episodes={
        "recover-boundary": ("top_long", rewards, 3.0),
    })
    continuation = root / "recover-boundary/raw/recover-boundary/snapshots/continuations/000032.json"
    payload = json.loads(continuation.read_text(encoding="utf-8"))
    payload["randomization"]["continuation_step"] = 16
    _write(continuation, payload)
    _reseal_episode(root, receipts, seal, "recover-boundary")
    output = tmp_path / "wrong-step.json"

    audit_successful_recoveries(
        accepted_roots=(root,), receipt_roots=(receipts,), round_seal_paths=(seal,), output_path=output,
    )

    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["selected_recoveries"] == []
    assert document["exclusions"] == [{
        "source_round_id": "original-round", "source_episode_id": "recover-boundary",
        "reason": "no_authenticated_h16_snapshot_before_recovery_confirmation",
    }]


def test_audit_rejects_a_continuation_from_a_different_reset_randomization(tmp_path: Path) -> None:
    from lehome_train.groot.recovery_audit import audit_successful_recoveries

    rewards = [0.0] * 16 + [0.4, 0.4, 0.2] + [0.2] * 13 + [0.35, 0.4, 0.5] + [0.5] * 4
    root, receipts, seal = _round(tmp_path, round_id="original-round", episodes={
        "recover-boundary": ("top_long", rewards, 3.0),
    })
    continuation = root / "recover-boundary/raw/recover-boundary/snapshots/continuations/000032.json"
    payload = json.loads(continuation.read_text(encoding="utf-8"))
    payload["randomization"]["strategy"] = "geometry"
    _write(continuation, payload)
    _reseal_episode(root, receipts, seal, "recover-boundary")
    output = tmp_path / "wrong-randomization.json"

    audit_successful_recoveries(
        accepted_roots=(root,), receipt_roots=(receipts,), round_seal_paths=(seal,), output_path=output,
    )

    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["selected_recoveries"] == []
    assert document["exclusions"] == [{
        "source_round_id": "original-round", "source_episode_id": "recover-boundary",
        "reason": "no_authenticated_h16_snapshot_before_recovery_confirmation",
    }]


def test_audit_rejects_missing_or_malformed_policy_chunk_provenance(tmp_path: Path) -> None:
    from lehome_train.groot.recovery_audit import audit_successful_recoveries

    rewards = [0.0] * 16 + [0.4, 0.2, 0.4, 0.5]
    root, receipts, seal = _round(tmp_path, round_id="original-round", episodes={
        "recover": ("top_long", rewards, 4.0),
    })
    annotations = root / "recover/raw/recover/annotations.jsonl"
    rows = [json.loads(line) for line in annotations.read_text(encoding="utf-8").splitlines()]
    del rows[0]["policy_request_id"]
    annotations.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    _reseal_episode(root, receipts, seal, "recover")

    with pytest.raises(ValueError, match="policy chunk provenance"):
        audit_successful_recoveries(
            accepted_roots=(root,), receipt_roots=(receipts,), round_seal_paths=(seal,),
            output_path=tmp_path / "result.json",
        )


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda rows: rows[0].update({"policy_chunk_offset": 1}), "nonzero_initial_offset"),
        (lambda rows: rows[3].update({"policy_chunk_offset": 4}), "offset_jump"),
        (lambda rows: rows[3].update({"policy_request_id": "request-recover-1"}), "early_request_change"),
        (lambda rows: rows[16].update({"policy_request_id": "request-recover-0"}), "reused_request"),
        (lambda rows: rows[16].update({"policy_chunk_offset": 1}), "invalid_post_chunk_transition"),
    ],
)
def test_audit_rejects_noncanonical_h16_policy_chunk_traces(
    tmp_path: Path,
    mutate: object,
    reason: str,
) -> None:
    from lehome_train.groot.recovery_audit import audit_successful_recoveries

    rewards = [0.0] * 16 + [0.4, 0.4, 0.2] + [0.2] * 13 + [0.35, 0.4, 0.5] + [0.5] * 4
    root, receipts, seal = _round(tmp_path, round_id="original-round", episodes={
        "recover": ("top_long", rewards, 4.0),
    })
    annotations = root / "recover/raw/recover/annotations.jsonl"
    rows = [json.loads(line) for line in annotations.read_text(encoding="utf-8").splitlines()]
    assert callable(mutate)
    mutate(rows)
    annotations.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    _reseal_episode(root, receipts, seal, "recover")

    with pytest.raises(ValueError, match="policy chunk trace"):
        audit_successful_recoveries(
            accepted_roots=(root,), receipt_roots=(receipts,), round_seal_paths=(seal,),
            output_path=tmp_path / f"{reason}.json",
        )


def test_audit_excludes_a_recovery_when_the_only_fresh_boundary_is_the_confirmation(tmp_path: Path) -> None:
    from lehome_train.groot.recovery_audit import audit_successful_recoveries

    # The boundary at step 32 is the confirmation itself.  It follows the
    # corrective action and cannot prove a fresh recovery continuation.
    rewards = [0.0] * 16 + [0.4, 0.4, 0.2] + [0.2] * 13 + [0.4, 0.5] + [0.5] * 4
    root, receipts, seal = _round(tmp_path, round_id="original-round", episodes={
        "no-boundary": ("pant_short", rewards, 5.0),
    })
    output = tmp_path / "result.json"

    audit_successful_recoveries(
        accepted_roots=(root,), receipt_roots=(receipts,), round_seal_paths=(seal,), output_path=output,
    )

    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["selected_recoveries"] == []
    assert document["exclusions"] == [{
        "source_round_id": "original-round", "source_episode_id": "no-boundary",
        "reason": "no_authenticated_h16_snapshot_before_recovery_confirmation",
    }]


def test_audit_document_is_portable_across_absolute_source_relocations(tmp_path: Path) -> None:
    from lehome_train.groot.recovery_audit import audit_successful_recoveries

    rewards = [0.0] * 16 + [0.4, 0.2, 0.4, 0.5]
    first = _round(tmp_path / "hydration-a", round_id="original-round", episodes={
        "recover": ("top_long", rewards, 1.0),
    })
    second = _round(tmp_path / "hydration-b", round_id="original-round", episodes={
        "recover": ("top_long", rewards, 1.0),
    })
    first_output, second_output = tmp_path / "first.json", tmp_path / "second.json"
    audit_successful_recoveries(
        accepted_roots=(first[0],), receipt_roots=(first[1],), round_seal_paths=(first[2],), output_path=first_output,
    )
    audit_successful_recoveries(
        accepted_roots=(second[0],), receipt_roots=(second[1],), round_seal_paths=(second[2],), output_path=second_output,
    )
    assert first_output.read_bytes() == second_output.read_bytes()
    assert first_output.with_name("first.json.sha256").read_bytes() == second_output.with_name("second.json.sha256").read_bytes()
    assert not any(item.startswith("/") for item in _strings(json.loads(first_output.read_text(encoding="utf-8"))))


def test_audit_deduplicates_recovery_start_states_deterministically(tmp_path: Path) -> None:
    from lehome_train.groot.recovery_audit import audit_successful_recoveries

    rewards = [0.0] * 16 + [0.4, 0.4, 0.2] + [0.2] * 13 + [0.35, 0.4, 0.5] + [0.5] * 17
    first, first_receipts, first_seal = _round(tmp_path, round_id="original-round", episodes={
        "a-winner": ("top_short", rewards, 7.0),
    })
    second, second_receipts, second_seal = _round(tmp_path, round_id="replay-round", episodes={
        "z-duplicate": ("top_short", rewards, 7.0),
    })
    output = tmp_path / "result.json"

    audit_successful_recoveries(
        accepted_roots=(first, second), receipt_roots=(first_receipts, second_receipts),
        round_seal_paths=(first_seal, second_seal), output_path=output,
    )
    document = json.loads(output.read_text(encoding="utf-8"))
    assert [row["source_episode_id"] for row in document["selected_recoveries"]] == ["a-winner"]
    assert document["duplicates"] == [{
        "duplicate_episode_id": "z-duplicate", "fingerprint": document["selected_recoveries"][0]["fingerprint"],
        "reason": "duplicate_continuation_start_state", "winner_episode_id": "a-winner",
    }]


def test_audit_never_emits_a_short_tail_when_recovery_is_near_episode_end(tmp_path: Path) -> None:
    from lehome_train.groot.recovery_audit import audit_successful_recoveries

    rewards = [0.0] * 31 + [0.4, 0.2, 0.4, 0.5]
    root, receipts, seal = _round(tmp_path, round_id="original-round", episodes={
        "late-recovery": ("pant_long", rewards, 4.0),
    })
    output = tmp_path / "result.json"

    audit_successful_recoveries(
        accepted_roots=(root,), receipt_roots=(receipts,), round_seal_paths=(seal,), output_path=output,
    )

    selected = json.loads(output.read_text(encoding="utf-8"))["selected_recoveries"][0]
    assert selected["h16_ranges"] == [{"start": 19, "stop": 35, "frame_ids": list(range(19, 35))}]


def test_audit_excludes_a_recovery_without_one_full_h16_window(tmp_path: Path) -> None:
    from lehome_train.groot.recovery_audit import audit_successful_recoveries

    root, receipts, seal = _round(tmp_path, round_id="original-round", episodes={
        "too-short": ("pant_short", [0.0, 0.2, 0.4, 0.2, 0.4, 0.5], 5.0),
    })
    output = tmp_path / "result.json"

    audit_successful_recoveries(
        accepted_roots=(root,), receipt_roots=(receipts,), round_seal_paths=(seal,), output_path=output,
    )

    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["selected_recoveries"] == []
    assert document["exclusions"] == [{
        "source_round_id": "original-round", "source_episode_id": "too-short",
        "reason": "no_authenticated_h16_snapshot_before_recovery_confirmation",
    }]


def test_audit_rejects_a_cached_reward_plateau_without_drawdown_evidence(tmp_path: Path) -> None:
    from lehome_train.groot.recovery_audit import audit_successful_recoveries

    rewards = [0.2] + [0.2] * 16 + [0.35, 0.5]
    root, receipts, seal = _round(tmp_path, round_id="original-round", episodes={
        "stalled": ("pant_short", rewards, 6.0),
    })
    output = tmp_path / "result.json"

    audit_successful_recoveries(
        accepted_roots=(root,), receipt_roots=(receipts,), round_seal_paths=(seal,), output_path=output,
    )

    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["selected_recoveries"] == []
    assert document["detector_mode"] == "reward_drawdown_only_no_reward_freshness_annotations"


def test_audit_fails_closed_on_untrusted_annotation_or_output_overwrite(tmp_path: Path) -> None:
    from lehome_train.groot.recovery_audit import audit_successful_recoveries

    rewards = [0.0, 0.2, 0.4, 0.35, 0.2, 0.25, 0.4] + [0.5] * 17
    root, receipts, seal = _round(tmp_path, round_id="original-round", episodes={
        "recover-top": ("top_long", rewards, 1.0),
    })
    annotations = root / "recover-top/raw/recover-top/annotations.jsonl"
    rows = [json.loads(line) for line in annotations.read_text(encoding="utf-8").splitlines()]
    rows[3]["action_source"] = "operator"
    annotations.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    _reseal_episode(root, receipts, seal, "recover-top")
    # Fully resealed provenance reaches the semantic admission check.
    with pytest.raises(ValueError, match="policy"):
        audit_successful_recoveries(
            accepted_roots=(root,), receipt_roots=(receipts,), round_seal_paths=(seal,),
            output_path=tmp_path / "result.json",
        )

    output = tmp_path / "existing.json"
    output.write_text("do not replace", encoding="utf-8")
    with pytest.raises(FileExistsError):
        audit_successful_recoveries(
            accepted_roots=(root,), receipt_roots=(receipts,), round_seal_paths=(seal,), output_path=output,
        )


def test_audit_rejects_shallow_drawdown_without_minimum_recovery_gain(tmp_path: Path) -> None:
    from lehome_train.groot.recovery_audit import audit_successful_recoveries

    rewards = [0.0] * 16 + [0.4, 0.34, 0.4, 0.5]
    root, receipts, seal = _round(tmp_path, round_id="original-round", episodes={
        "shallow": ("top_long", rewards, 8.0),
    })
    output = tmp_path / "result.json"

    audit_successful_recoveries(
        accepted_roots=(root,), receipt_roots=(receipts,), round_seal_paths=(seal,), output_path=output,
    )

    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["selected_recoveries"] == []
    assert document["exclusions"] == [{
        "source_round_id": "original-round", "source_episode_id": "shallow",
        "reason": "no_meaningful_recovery",
    }]


def test_audit_accepts_a_latched_success_tail_and_rejects_an_unlatched_one(tmp_path: Path) -> None:
    from lehome_train.groot.recovery_audit import audit_successful_recoveries

    rewards = [0.0] * 16 + [0.4, 0.4, 0.2] + [0.2] * 13 + [0.35, 0.4, 0.5] + [0.5] * 17
    root, receipts, seal = _round(tmp_path, round_id="original-round", episodes={
        "latched": ("top_short", rewards, 9.0),
    })
    valid = tmp_path / "valid.json"
    audit_successful_recoveries(
        accepted_roots=(root,), receipt_roots=(receipts,), round_seal_paths=(seal,), output_path=valid,
    )
    assert json.loads(valid.read_text(encoding="utf-8"))["admitted_episodes"][0]["official_success_step"] == 50

    annotations = root / "latched/raw/latched/annotations.jsonl"
    rows = [json.loads(line) for line in annotations.read_text(encoding="utf-8").splitlines()]
    rows[-1]["success"] = False
    annotations.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    _reseal_episode(root, receipts, seal, "latched")
    with pytest.raises(ValueError, match="latched"):
        audit_successful_recoveries(
            accepted_roots=(root,), receipt_roots=(receipts,), round_seal_paths=(seal,),
            output_path=tmp_path / "invalid.json",
        )


def test_audit_rejects_relative_symlink_and_overlapping_inputs(tmp_path: Path) -> None:
    from lehome_train.groot.recovery_audit import audit_successful_recoveries

    rewards = [0.0] * 16 + [0.4, 0.4, 0.2] + [0.2] * 13 + [0.35, 0.4, 0.5] + [0.5] * 17
    root, receipts, seal = _round(tmp_path, round_id="original-round", episodes={
        "recover": ("pant_long", rewards, 10.0),
    })
    with pytest.raises(ValueError, match="absolute"):
        audit_successful_recoveries(
            accepted_roots=("relative",), receipt_roots=(receipts,), round_seal_paths=(seal,),
            output_path=tmp_path / "relative.json",
        )
    linked = tmp_path / "linked-accepted"
    linked.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="non-symlink|symlink"):
        audit_successful_recoveries(
            accepted_roots=(linked,), receipt_roots=(receipts,), round_seal_paths=(seal,),
            output_path=tmp_path / "linked.json",
        )
    with pytest.raises(ValueError, match="overlap"):
        audit_successful_recoveries(
            accepted_roots=(root,), receipt_roots=(receipts,), round_seal_paths=(seal,),
            output_path=root / "must-not-write.json",
        )


def test_audit_publishes_checksum_before_visible_json_commit_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lehome_train.groot import recovery_audit

    rewards = [0.0] * 16 + [0.4, 0.2, 0.4, 0.5]
    root, receipts, seal = _round(tmp_path, round_id="original-round", episodes={
        "recover": ("pant_long", rewards, 11.0),
    })
    destinations: list[str] = []
    original_link = recovery_audit.os.link

    def tracking_link(source: str | bytes | os.PathLike[str], destination: str | bytes | os.PathLike[str], *args: object, **kwargs: object) -> None:
        destinations.append(Path(destination).name)
        original_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(recovery_audit.os, "link", tracking_link)
    audit = recovery_audit.audit_successful_recoveries
    audit(
        accepted_roots=(root,), receipt_roots=(receipts,), round_seal_paths=(seal,),
        output_path=tmp_path / "result.json",
    )
    assert destinations == ["result.json.sha256", "result.json"]


def test_cli_exits_zero_when_all_categories_have_five_distinct_recoveries(tmp_path: Path) -> None:
    categories = ("top_long", "top_short", "pant_long", "pant_short")
    rewards = [0.0] * 16 + [0.4, 0.4, 0.2] + [0.2] * 13 + [0.35, 0.4, 0.5] + [0.5] * 17
    episodes = {
        f"{category}-{index}": (category, rewards, float(category_index * 10 + index))
        for category_index, category in enumerate(categories)
        for index in range(5)
    }
    root, receipts, seal = _round(tmp_path, round_id="complete-round", episodes=episodes)
    repo = Path(__file__).parents[2]
    environment = os.environ | {"PYTHONPATH": f"{repo / 'source/lehome'}:{repo / 'trainer/src'}"}
    completed = subprocess.run(
        [sys.executable, str(repo / "scripts/audit_successful_recoveries.py"),
         "--accepted-root", str(root), "--receipts-root", str(receipts), "--round-seal", str(seal),
         "--output", str(tmp_path / "complete.json")],
        cwd=repo, env=environment, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["ready"] is True


def test_cli_exits_two_for_a_valid_shortfall(tmp_path: Path) -> None:
    rewards = [0.0] * 16 + [0.4, 0.2, 0.4, 0.5]
    root, receipts, seal = _round(tmp_path, round_id="shortfall-round", episodes={
        "only-one": ("top_long", rewards, 12.0),
    })
    repo = Path(__file__).parents[2]
    environment = os.environ | {"PYTHONPATH": f"{repo / 'source/lehome'}:{repo / 'trainer/src'}"}
    completed = subprocess.run(
        [sys.executable, str(repo / "scripts/audit_successful_recoveries.py"),
         "--accepted-root", str(root), "--receipts-root", str(receipts), "--round-seal", str(seal),
         "--output", str(tmp_path / "shortfall.json")],
        cwd=repo, env=environment, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 2, completed.stderr
    assert json.loads(completed.stdout)["ready"] is False
