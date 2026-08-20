from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pytest


def _write(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state_fingerprint(*, category: str, garment: str, state: list[float]) -> str:
    rounded = ["0.000000" if value == 0.0 else format(value, ".6f") for value in state]
    return hashlib.sha256(json.dumps({"category": category, "garment": garment, "state_rounding": "fixed_6dp", "state": rounded}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _accepted(root: Path, round_id: str, episode: str, category: str) -> dict[str, str]:
    raw = root / episode / "raw" / episode
    reset = raw / "snapshots" / "reset.json"
    annotations = raw / "annotations.jsonl"
    reset_hash = _write(reset, {"schema_version": 1, "robot_position": [0.0] * 12, "robot_velocity": [0.0] * 12, "cloth_position": [[0.0, 0.0, 0.0]], "cloth_velocity": [[0.0, 0.0, 0.0]], "rng_state": {}, "garment_name": f"{category}-seen"})
    annotations.parent.mkdir(parents=True, exist_ok=True)
    annotations.write_text("".join(json.dumps({"step": index, "action": [float(index)] * 12, "action_source": "policy", "reward": float(index), "success": index >= 19, "state": [float(index)] * 12, "policy_request_id": f"request-{index // 16}", "policy_chunk_offset": index % 16}, sort_keys=True) + "\n" for index in range(20)), encoding="utf-8")
    annotation_hash = hashlib.sha256(annotations.read_bytes()).hexdigest()
    episode_hash = _write(raw / "episode.json", {"episode_id": episode, "accepted_success": True, "outcome": "success", "terminal_reason": "success", "identity": {"category": category, "garment_name": f"{category}-seen"}})
    manifest = {
        path.relative_to(raw).as_posix(): {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size": path.stat().st_size}
        for path in (reset, annotations, raw / "episode.json")
    }
    raw_manifest_hash = _write(raw / "SHA256SUMS.json", manifest)
    package_hash = hashlib.sha256((round_id + episode).encode()).hexdigest()
    garment = f"{category}-seen"
    state = [16.0] * 12
    fingerprint = _state_fingerprint(category=category, garment=garment, state=state)
    return {"source_round_id": round_id, "source_episode_id": episode, "source_episode_digest": package_hash, "source_immutable_revision": "a" * 40, "category": category, "garment": garment, "fingerprint": fingerprint, "continuation_start": {"annotation_index": 16, "step": 16, "policy_request_id": "request-1", "policy_chunk_offset": 0, "state": state, "state_fingerprint": fingerprint}, "recovery_event": {"adverse_start": 15, "recovery_confirmation": 18}, "source_artifacts": {"package_sync_digest": package_hash, "raw_checksum_manifest_sha256": raw_manifest_hash, "episode_manifest_sha256": episode_hash, "annotations_sha256": annotation_hash, "reset_sha256": reset_hash}}


def test_builder_emits_portable_96_attempt_schedule_with_a_separate_hydration_descriptor(tmp_path: Path) -> None:
    from scripts.build_controlled_recovery_matrix import build_controlled_recovery_matrix

    accepted = tmp_path / "accepted"
    selected = [_accepted(accepted, "round-1", "long", "pant_long"), _accepted(accepted, "round-1", "tl", "top_long"), _accepted(accepted, "round-1", "ts", "top_short")]
    audit = {"schema_version": 2, "kind": "lehome_successful_recovery_audit", "semantic_sha256": "", "selected_recoveries": selected, "shortfalls": {"pant_long": 4, "top_long": 1, "top_short": 3, "pant_short": 0}}
    audit["semantic_sha256"] = hashlib.sha256(json.dumps({key: value for key, value in audit.items() if key != "semantic_sha256"}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    audit_path = tmp_path / "audit.json"
    _write(audit_path, audit)
    sidecar = tmp_path / "audit.json.sha256"
    sidecar.write_text(hashlib.sha256(audit_path.read_bytes()).hexdigest() + "\n", encoding="ascii")
    first, second = tmp_path / "one.json", tmp_path / "two.json"
    first_receipt = build_controlled_recovery_matrix(audit_path=audit_path, accepted_roots=[accepted], output=first)
    build_controlled_recovery_matrix(audit_path=audit_path, accepted_roots=[accepted], output=second)
    matrix = json.loads(first.read_text(encoding="utf-8"))
    assert matrix["kind"] == "controlled_success_recovery_matrix_v1"
    rows = matrix["rows"]
    assert len(rows) == 96
    assert {category: sum(row["category"] == category for row in rows) for category in ("pant_long", "top_long", "top_short")} == {
        "pant_long": 32, "top_long": 32, "top_short": 32,
    }
    assert not any(row["category"] == "pant_short" for row in rows)
    assert all(row["recovery_kind"] == "controlled_success_recovery_v1" for row in rows)
    assert all(row["source_continuation_state"] == [16.0] * 12 for row in rows)
    assert all("source_reset" not in row and "source_annotations" not in row for row in rows)
    assert not any(str(tmp_path) in first.read_text(encoding="utf-8") for _ in [0])
    assert first.read_bytes() == second.read_bytes()
    assert first_receipt["matrix_sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    hydration = Path(first_receipt["materialization_path"])
    assert hydration.is_file()
    local_rows = json.loads(hydration.read_text(encoding="utf-8"))["rows"]
    assert len(local_rows) == 96
    assert all(Path(row["source_reset"]).is_absolute() for row in local_rows)
    published = (first, Path(str(first) + ".sha256"), hydration, Path(str(hydration) + ".sha256"))
    assert {path.stat().st_mode & 0o777 for path in published} == {0o644}


def test_builder_honors_lower_attempt_bound_and_keeps_later_retries_after_failures(tmp_path: Path) -> None:
    from scripts.build_controlled_recovery_matrix import build_controlled_recovery_matrix

    accepted = tmp_path / "accepted"
    selected = [_accepted(accepted, "round-1", "long", "pant_long"), _accepted(accepted, "round-1", "tl", "top_long"), _accepted(accepted, "round-1", "ts", "top_short")]
    audit = {"schema_version": 2, "kind": "lehome_successful_recovery_audit", "semantic_sha256": "", "selected_recoveries": selected, "shortfalls": {"pant_long": 4, "top_long": 1, "top_short": 3, "pant_short": 0}}
    audit["semantic_sha256"] = hashlib.sha256(json.dumps({key: value for key, value in audit.items() if key != "semantic_sha256"}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    audit_path = tmp_path / "audit.json"
    _write(audit_path, audit)
    (tmp_path / "audit.json.sha256").write_text(hashlib.sha256(audit_path.read_bytes()).hexdigest() + "\n", encoding="ascii")
    result = build_controlled_recovery_matrix(audit_path=audit_path, accepted_roots=[accepted], output=tmp_path / "matrix.json", max_attempts=12)
    rows = json.loads(Path(result["materialization_path"]).read_text(encoding="utf-8"))["rows"]
    assert len(rows) == 12
    assert len({row["perturbation_seed"] for row in rows}) == 12
    assert Counter(row["category"] for row in rows) == {"pant_long": 4, "top_long": 4, "top_short": 4}


def test_eight_row_lower_bound_starts_with_the_exact_reachable_acceptance_caps(tmp_path: Path) -> None:
    from scripts.build_controlled_recovery_matrix import build_controlled_recovery_matrix

    accepted = tmp_path / "accepted"
    selected = [_accepted(accepted, "round-1", "long", "pant_long"), _accepted(accepted, "round-1", "tl", "top_long"), _accepted(accepted, "round-1", "ts", "top_short")]
    audit = {"schema_version": 2, "kind": "lehome_successful_recovery_audit", "semantic_sha256": "", "selected_recoveries": selected, "shortfalls": {"pant_long": 4, "top_long": 1, "top_short": 3, "pant_short": 0}}
    audit["semantic_sha256"] = hashlib.sha256(json.dumps({key: value for key, value in audit.items() if key != "semantic_sha256"}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    audit_path = tmp_path / "audit.json"; _write(audit_path, audit)
    audit_path.with_name("audit.json.sha256").write_text(hashlib.sha256(audit_path.read_bytes()).hexdigest() + "\n", encoding="ascii")
    result = build_controlled_recovery_matrix(audit_path=audit_path, accepted_roots=[accepted], output=tmp_path / "matrix.json", max_attempts=8)
    rows = json.loads(Path(result["materialization_path"]).read_text(encoding="utf-8"))["rows"]
    assert Counter(row["category"] for row in rows) == {"pant_long": 4, "top_long": 1, "top_short": 3}


def test_builder_uses_only_the_authenticated_explicit_continuation_boundary(tmp_path: Path) -> None:
    from scripts.build_controlled_recovery_matrix import build_controlled_recovery_matrix

    accepted = tmp_path / "accepted"
    selected = [
        _accepted(accepted, "round-1", "long", "pant_long"),
        _accepted(accepted, "round-1", "tl", "top_long"),
        _accepted(accepted, "round-1", "ts", "top_short"),
    ]
    # The recovery event begins at one, but only the authenticated offset-zero
    # continuation at sixteen can be safely replayed after a policy-session reset.
    audit = {"schema_version": 2, "kind": "lehome_successful_recovery_audit", "semantic_sha256": "", "selected_recoveries": selected, "shortfalls": {"pant_long": 4, "top_long": 1, "top_short": 3, "pant_short": 0}}
    audit["semantic_sha256"] = hashlib.sha256(json.dumps({key: value for key, value in audit.items() if key != "semantic_sha256"}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    audit_path = tmp_path / "audit.json"; _write(audit_path, audit)
    audit_path.with_name("audit.json.sha256").write_text(hashlib.sha256(audit_path.read_bytes()).hexdigest() + "\n", encoding="ascii")

    result = build_controlled_recovery_matrix(
        audit_path=audit_path, accepted_roots=[accepted], output=tmp_path / "matrix.json", max_attempts=8,
    )

    rows = json.loads(Path(result["materialization_path"]).read_text(encoding="utf-8"))["rows"]
    assert {row["prefix_stop"] for row in rows} == {16}
    assert {row["source_state_fingerprint"] for row in rows} == {
        selected[0]["continuation_start"]["state_fingerprint"],
        selected[1]["continuation_start"]["state_fingerprint"],
        selected[2]["continuation_start"]["state_fingerprint"],
    }
    assert {tuple(row["source_continuation_state"]) for row in rows} == {(16.0,) * 12}


def test_builder_carries_the_exact_authenticated_continuation_state_into_portable_rows(tmp_path: Path) -> None:
    from scripts.build_controlled_recovery_matrix import build_controlled_recovery_matrix

    accepted = tmp_path / "accepted"
    selected = [_accepted(accepted, "round-1", "long", "pant_long"), _accepted(accepted, "round-1", "tl", "top_long"), _accepted(accepted, "round-1", "ts", "top_short")]
    selected[0]["continuation_start"]["state"] = [16.0 + index / 10 for index in range(12)]
    raw = accepted / "long" / "raw" / "long"
    annotation_rows = [json.loads(line) for line in (raw / "annotations.jsonl").read_text(encoding="utf-8").splitlines()]
    annotation_rows[16]["state"] = selected[0]["continuation_start"]["state"]
    (raw / "annotations.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in annotation_rows), encoding="utf-8")
    selected[0]["continuation_start"]["state_fingerprint"] = _state_fingerprint(category="pant_long", garment="pant_long-seen", state=selected[0]["continuation_start"]["state"])
    selected[0]["fingerprint"] = selected[0]["continuation_start"]["state_fingerprint"]
    manifest = json.loads((raw / "SHA256SUMS.json").read_text(encoding="utf-8"))
    manifest["annotations.jsonl"] = {"sha256": hashlib.sha256((raw / "annotations.jsonl").read_bytes()).hexdigest(), "size": (raw / "annotations.jsonl").stat().st_size}
    selected[0]["source_artifacts"]["annotations_sha256"] = hashlib.sha256((raw / "annotations.jsonl").read_bytes()).hexdigest()
    selected[0]["source_artifacts"]["raw_checksum_manifest_sha256"] = _write(raw / "SHA256SUMS.json", manifest)
    audit = {"schema_version": 2, "kind": "lehome_successful_recovery_audit", "semantic_sha256": "", "selected_recoveries": selected, "shortfalls": {"pant_long": 4, "top_long": 1, "top_short": 3, "pant_short": 0}}
    audit["semantic_sha256"] = hashlib.sha256(json.dumps({key: value for key, value in audit.items() if key != "semantic_sha256"}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    audit_path = tmp_path / "audit.json"; _write(audit_path, audit)
    audit_path.with_name("audit.json.sha256").write_text(hashlib.sha256(audit_path.read_bytes()).hexdigest() + "\n", encoding="ascii")

    result = build_controlled_recovery_matrix(audit_path=audit_path, accepted_roots=[accepted], output=tmp_path / "matrix.json", max_attempts=8)
    portable_rows = json.loads((tmp_path / "matrix.json").read_text(encoding="utf-8"))["rows"]
    hydrated_rows = json.loads(Path(result["materialization_path"]).read_text(encoding="utf-8"))["rows"]
    assert {tuple(row["source_continuation_state"]) for row in portable_rows if row["category"] == "pant_long"} == {tuple(selected[0]["continuation_start"]["state"])}
    assert {tuple(row["source_continuation_state"]) for row in hydrated_rows if row["category"] == "pant_long"} == {tuple(selected[0]["continuation_start"]["state"])}


def test_builder_rejects_missing_or_nonboundary_continuation_provenance(tmp_path: Path) -> None:
    from scripts.build_controlled_recovery_matrix import build_controlled_recovery_matrix

    accepted = tmp_path / "accepted"
    selected = [
        _accepted(accepted, "round-1", "long", "pant_long"),
        _accepted(accepted, "round-1", "tl", "top_long"),
        _accepted(accepted, "round-1", "ts", "top_short"),
    ]
    selected[0]["continuation_start"]["policy_chunk_offset"] = 1
    audit = {"schema_version": 2, "kind": "lehome_successful_recovery_audit", "semantic_sha256": "", "selected_recoveries": selected, "shortfalls": {"pant_long": 4, "top_long": 1, "top_short": 3, "pant_short": 0}}
    audit["semantic_sha256"] = hashlib.sha256(json.dumps({key: value for key, value in audit.items() if key != "semantic_sha256"}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    audit_path = tmp_path / "audit.json"; _write(audit_path, audit)
    audit_path.with_name("audit.json.sha256").write_text(hashlib.sha256(audit_path.read_bytes()).hexdigest() + "\n", encoding="ascii")

    with pytest.raises(ValueError, match="continuation"):
        build_controlled_recovery_matrix(
            audit_path=audit_path, accepted_roots=[accepted], output=tmp_path / "matrix.json", max_attempts=8,
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rows: rows[3].update({"policy_chunk_offset": 4}),
        lambda rows: rows[3].update({"policy_request_id": "request-2"}),
        lambda rows: rows[16].update({"policy_request_id": "request-0"}),
    ],
)
def test_builder_rejects_hash_consistent_sources_with_broken_h16_chunk_traces(
    tmp_path: Path,
    mutate: Callable[[list[dict[str, object]]], None],
) -> None:
    from scripts.build_controlled_recovery_matrix import build_controlled_recovery_matrix

    accepted = tmp_path / "accepted"
    selected = [
        _accepted(accepted, "round-1", "long", "pant_long"),
        _accepted(accepted, "round-1", "tl", "top_long"),
        _accepted(accepted, "round-1", "ts", "top_short"),
    ]
    raw = accepted / "long" / "raw" / "long"
    annotations = raw / "annotations.jsonl"
    rows = [json.loads(line) for line in annotations.read_text(encoding="utf-8").splitlines()]
    mutate(rows)
    annotations.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    manifest = json.loads((raw / "SHA256SUMS.json").read_text(encoding="utf-8"))
    manifest["annotations.jsonl"] = {"sha256": hashlib.sha256(annotations.read_bytes()).hexdigest(), "size": annotations.stat().st_size}
    raw_manifest_hash = _write(raw / "SHA256SUMS.json", manifest)
    selected[0]["source_artifacts"]["annotations_sha256"] = hashlib.sha256(annotations.read_bytes()).hexdigest()
    selected[0]["source_artifacts"]["raw_checksum_manifest_sha256"] = raw_manifest_hash
    audit = {"schema_version": 2, "kind": "lehome_successful_recovery_audit", "semantic_sha256": "", "selected_recoveries": selected, "shortfalls": {"pant_long": 4, "top_long": 1, "top_short": 3, "pant_short": 0}}
    audit["semantic_sha256"] = hashlib.sha256(json.dumps({key: value for key, value in audit.items() if key != "semantic_sha256"}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    audit_path = tmp_path / "audit.json"; _write(audit_path, audit)
    audit_path.with_name("audit.json.sha256").write_text(hashlib.sha256(audit_path.read_bytes()).hexdigest() + "\n", encoding="ascii")

    with pytest.raises(ValueError, match="policy chunk trace"):
        build_controlled_recovery_matrix(
            audit_path=audit_path, accepted_roots=[accepted], output=tmp_path / "matrix.json", max_attempts=8,
        )


def test_portable_matrix_is_byte_identical_after_accepted_root_relocation(tmp_path: Path) -> None:
    from scripts.build_controlled_recovery_matrix import build_controlled_recovery_matrix

    def build_fixture(base: Path) -> tuple[Path, Path]:
        accepted = base / "accepted"
        selected = [_accepted(accepted, "round-1", "long", "pant_long"), _accepted(accepted, "round-1", "tl", "top_long"), _accepted(accepted, "round-1", "ts", "top_short")]
        audit = {"schema_version": 2, "kind": "lehome_successful_recovery_audit", "semantic_sha256": "", "selected_recoveries": selected, "shortfalls": {"pant_long": 4, "top_long": 1, "top_short": 3, "pant_short": 0}}
        audit["semantic_sha256"] = hashlib.sha256(json.dumps({key: value for key, value in audit.items() if key != "semantic_sha256"}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        audit_path = base / "recovery-audit-v2.json"
        _write(audit_path, audit)
        audit_path.with_name(audit_path.name + ".sha256").write_text(hashlib.sha256(audit_path.read_bytes()).hexdigest() + "\n", encoding="ascii")
        return accepted, audit_path

    first_root, first_audit = build_fixture(tmp_path / "hydration-a")
    second_root, second_audit = build_fixture(tmp_path / "hydration-b")
    build_controlled_recovery_matrix(audit_path=first_audit, accepted_roots=[first_root], output=tmp_path / "first.json", max_attempts=12)
    build_controlled_recovery_matrix(audit_path=second_audit, accepted_roots=[second_root], output=tmp_path / "second.json", max_attempts=12)
    assert (tmp_path / "first.json").read_bytes() == (tmp_path / "second.json").read_bytes()


def test_builder_rejects_bad_audit_sidecar_and_unsafe_output(tmp_path: Path) -> None:
    from scripts.build_controlled_recovery_matrix import build_controlled_recovery_matrix

    audit = tmp_path / "audit.json"
    _write(audit, {})
    (tmp_path / "audit.json.sha256").write_text("0" * 64 + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="SHA-256"):
        build_controlled_recovery_matrix(audit_path=audit, accepted_roots=[tmp_path], output=tmp_path / "matrix.json")
