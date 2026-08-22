from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import textwrap

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE = REPO_ROOT / "rollout_appliance" / "run_controlled_recovery_smoke.sh"
PRODUCTION = REPO_ROOT / "rollout_appliance" / "run_controlled_recovery_campaign.sh"
SNAPSHOT_SOURCE_BOOTSTRAP = REPO_ROOT / "rollout_appliance" / "run_snapshot_source_bootstrap.sh"


def _write(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state_fingerprint(*, category: str, garment: str, state: list[float]) -> str:
    rounded = ["0.000000" if value == 0.0 else format(value, ".6f") for value in state]
    return hashlib.sha256(json.dumps({"category": category, "garment": garment, "state_rounding": "fixed_6dp", "state": rounded}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _artifacts(tmp_path: Path) -> dict[str, str]:
    """Build the real, full 8-row materialization required by the smoke."""
    from scripts.build_controlled_recovery_matrix import build_controlled_recovery_matrix

    accepted = tmp_path / "accepted"
    selected = []
    for index, category in enumerate(("pant_long", "top_long", "top_short")):
        episode = f"episode-{index}"
        raw = accepted / episode / "raw" / episode
        reset = raw / "snapshots" / "reset.json"
        reset_hash = _write(reset, {"schema_version": 2, "robot_position": [0.0] * 12, "robot_velocity": [0.0] * 12, "cloth_position": [[0.0, 0.0, 0.0]], "cloth_velocity": [[0.0, 0.0, 0.0]], "rng_state": {}, "garment_name": f"{category}-seen", "randomization": {"strategy": "canonical"}, "scene_state": {}, "cloth_state_authority": "physx_cloth_view_world_v1"})
        annotations = raw / "annotations.jsonl"; annotations.parent.mkdir(parents=True, exist_ok=True)
        annotations.write_text("".join(json.dumps({"step": step, "action": [float(step)] * 12, "action_source": "policy", "success": step >= 19, "state": [float(step)] * 12, "policy_request_id": f"request-{step // 16}", "policy_chunk_offset": step % 16}) + "\n" for step in range(20)), encoding="utf-8")
        annotations_hash = hashlib.sha256(annotations.read_bytes()).hexdigest()
        continuation = raw / "snapshots" / "continuations" / "000016.json"
        continuation_hash = _write(continuation, {"schema_version": 2, "robot_position": [16.0] * 12, "robot_velocity": [0.0] * 12, "cloth_position": [[0.0, 0.0, 0.0]], "cloth_velocity": [[0.0, 0.0, 0.0]], "rng_state": {}, "garment_name": f"{category}-seen", "randomization": {"strategy": "canonical", "continuation_step": 16}, "scene_state": {}, "cloth_state_authority": "physx_cloth_view_world_v1"})
        episode_hash = _write(raw / "episode.json", {"episode_id": episode, "accepted_success": True, "outcome": "success", "terminal_reason": "success", "identity": {"category": category, "garment_name": f"{category}-seen", "seed": 50110}})
        manifest_hash = _write(raw / "SHA256SUMS.json", {item.relative_to(raw).as_posix(): {"sha256": hashlib.sha256(item.read_bytes()).hexdigest(), "size": item.stat().st_size} for item in (reset, annotations, continuation, raw / "episode.json")})
        digest = hashlib.sha256(("round" + episode).encode()).hexdigest()
        garment = f"{category}-seen"
        state = [16.0] * 12
        fingerprint = _state_fingerprint(category=category, garment=garment, state=state)
        selected.append({"source_round_id": "round", "source_episode_id": episode, "source_episode_digest": digest, "source_immutable_revision": "a" * 40, "category": category, "garment": garment, "fingerprint": fingerprint, "continuation_start": {"annotation_index": 16, "step": 16, "policy_request_id": "request-1", "policy_chunk_offset": 0, "policy_observation_state": state, "state": state, "state_fingerprint": fingerprint, "snapshot_relative_path": "snapshots/continuations/000016.json", "snapshot_sha256": continuation_hash, "snapshot_continuation_step": 16, "snapshot_robot_position": state}, "recovery_event": {"adverse_start": 15, "recovery_confirmation": 18}, "source_artifacts": {"package_sync_digest": digest, "raw_checksum_manifest_sha256": manifest_hash, "episode_manifest_sha256": episode_hash, "annotations_sha256": annotations_hash, "reset_sha256": reset_hash}})
    audit = {"schema_version": 4, "kind": "lehome_successful_recovery_audit", "continuation_contract": "authenticated_physx_cloth_view_snapshot_at_fresh_h16_next_action_boundary_v1", "semantic_sha256": "", "selected_recoveries": selected, "shortfalls": {"pant_long": 4, "top_long": 1, "top_short": 3, "pant_short": 0}}
    audit["semantic_sha256"] = hashlib.sha256(json.dumps({key: value for key, value in audit.items() if key != "semantic_sha256"}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    audit_path = tmp_path / "audit.json"; _write(audit_path, audit)
    audit_path.with_suffix(".json.sha256").write_text(hashlib.sha256(audit_path.read_bytes()).hexdigest() + "\n", encoding="ascii")
    return build_controlled_recovery_matrix(audit_path=audit_path, accepted_roots=[accepted], output=tmp_path / "matrix.json", max_attempts=8)


def _fake_base(path: Path, *, terminal: str = "accepted", receipt_ok: bool = True) -> None:
    path.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        python3 - "${{LEHOME_CAMPAIGN_ROOT}}" "${{LEHOME_ATTEMPT_MATRIX}}" "${{LEHOME_ROUND_ID}}" <<'PY'
        import hashlib, json, os, sqlite3, sys
        from pathlib import Path
        root, descriptor, round_id = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
        (root / 'base-launch.json').write_text(json.dumps({{'garment': os.environ.get('LEHOME_INITIAL_GARMENT'), 'max_worker_restarts': os.environ.get('LEHOME_MAX_WORKER_RESTARTS')}}))
        row = json.loads(descriptor.read_text())[0]
        canonical = json.dumps(row, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
        attempt = hashlib.sha256(json.dumps({{'schedule_index': 0, 'assignment': json.loads(canonical)}}, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
        root.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(root / 'ledger.sqlite3')
        con.execute('create table attempts (attempt_id text, schedule_index integer, assignment_json text)')
        con.execute('create table events (event_id integer primary key, event_type text, attempt_id text)')
        con.execute('insert into attempts values (?, 0, ?)', (attempt, canonical))
        con.execute('insert into events(event_type, attempt_id) values (?, ?)', ('leased', attempt))
        con.execute('insert into events(event_type, attempt_id) values (?, ?)', ('terminal_pending_validation', attempt))
        con.execute('insert into events(event_type, attempt_id) values (?, ?)', ('{terminal}', attempt))
        con.commit(); con.close()
        if '{terminal}' == 'accepted':
            accepted = root / 'accepted' / attempt; accepted.mkdir(parents=True)
            (accepted / 'episode.json').write_text('{{"ok":true}}')
            entries = [{{'relative_path': 'episode.json', 'sha256': hashlib.sha256((accepted / 'episode.json').read_bytes()).hexdigest(), 'byte_size': (accepted / 'episode.json').stat().st_size}}]
            receipt = {{'attempt_id': attempt, 'round_id': round_id, 'remote_prefix': f'rollout-rounds/{{round_id}}/{{attempt}}', 'readback_verified': {str(receipt_ok)}, 'episode_sha256': hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(',', ':')).encode()).hexdigest(), 'immutable_revision': 'a' * 40}}
            receipts = root / 'hf-sync-receipts'; receipts.mkdir(); (receipts / f'{{attempt}}.sync.json').write_text(json.dumps(receipt))
        PY
    """), encoding="utf-8")
    path.chmod(0o755)


def _fake_snapshot_source_base(path: Path) -> None:
    """Emit only the terminal evidence the bootstrap wrapper is allowed to inspect."""
    path.write_text(textwrap.dedent("""\
        #!/usr/bin/env bash
        set -euo pipefail
        python3 - "${LEHOME_CAMPAIGN_ROOT}" "${LEHOME_ATTEMPT_MATRIX}" "${LEHOME_ROUND_ID}" "${LEHOME_SNAPSHOT_SOURCE_TEST_CASE}" <<'PY'
        import hashlib, json, os, sqlite3, sys
        from pathlib import Path
        root, descriptor, round_id, case = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
        if case == 'absent-ledger': raise SystemExit(1)
        row = json.loads(descriptor.read_text())[0]
        attempt = hashlib.sha256(json.dumps({'schedule_index': 0, 'assignment': row}, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        root.mkdir(parents=True, exist_ok=True)
        (root / 'base-launch.json').write_text(json.dumps({'initial_garment': os.environ.get('LEHOME_INITIAL_GARMENT'), 'simulator_device': os.environ.get('LEHOME_SIMULATOR_DEVICE')}))
        con = sqlite3.connect(root / 'ledger.sqlite3')
        con.execute('create table events (event_type text, attempt_id text)')
        con.execute('insert into events values (?, ?)', ('rejected' if case == 'rejected' else 'accepted', attempt))
        con.commit(); con.close()
        if case == 'rejected': raise SystemExit(0)
        raw = root / 'accepted' / attempt / 'raw' / attempt; raw.mkdir(parents=True)
        garment, category, seed = 'Top_Long_Seen_0', 'top_long', 50110
        episode_payload = {'episode_id': attempt, 'mode': 'autonomous', 'accepted_success': True, 'outcome': 'success', 'terminal_reason': 'success', 'bc_target_count': 0, 'identity': {'category': category, 'garment_name': garment, 'seed': (50111 if case == 'seed-mismatch' else seed)}}
        episode = raw / 'episode.json'; episode.write_text(json.dumps(episode_payload, sort_keys=True))
        reset = raw / 'snapshots' / 'reset.json'; reset.parent.mkdir(parents=True, exist_ok=True)
        reset.write_text(json.dumps({'schema_version': 2, 'robot_position': [0.0] * 12, 'robot_velocity': [0.0] * 12, 'cloth_position': [[0.0, 0.0, 0.0]], 'cloth_velocity': [[0.0, 0.0, 0.0]], 'rng_state': {}, 'garment_name': garment, 'randomization': {'strategy': 'canonical'}, 'scene_state': {}, 'cloth_state_authority': 'physx_cloth_view_world_v1'}, sort_keys=True))
        annotations = raw / 'annotations.jsonl'
        boundary = 32 if case == 'after-success' else 16
        annotation_rows = []
        for step in range(33):
            state = [float(boundary)] * 12 if step == boundary else [float(step)] * 12
            if case == 'lagged-policy-state' and step == boundary: state = [float(boundary - 1)] * 12
            annotation_rows.append({'step': step, 'action': [float(step)] * 12, 'action_source': 'policy', 'policy_request_id': (f'request-{step // 16}' if case != 'bad-request-chunk' or step != 17 else 'request-bad'), 'policy_chunk_offset': step % 16, 'state': state, 'reward': 1.0, 'success': (step >= 19 and not (case == 'success-unlatched' and step == 20)), 'category': category, 'garment_name': garment, 'seed': seed})
        annotations.write_text(''.join(json.dumps(item, sort_keys=True) + '\\n' for item in annotation_rows))
        entries = {'episode.json': {'sha256': hashlib.sha256(episode.read_bytes()).hexdigest(), 'size': episode.stat().st_size}, 'annotations.jsonl': {'sha256': hashlib.sha256(annotations.read_bytes()).hexdigest(), 'size': annotations.stat().st_size}, 'snapshots/reset.json': {'sha256': hashlib.sha256(reset.read_bytes()).hexdigest(), 'size': reset.stat().st_size}}
        if case != 'no-h16':
            filename = '000015.json' if case == 'bad-boundary-filename' else f'{boundary:06d}.json'
            h16 = raw / 'snapshots' / 'continuations' / filename; h16.parent.mkdir(parents=True)
            continuation_step = boundary + 16 if case == 'wrong-continuation-step' else boundary
            snapshot_strategy = 'geometry' if case == 'randomization-mismatch' else 'canonical'
            snapshot = {'schema_version': (1 if case == 'bad-snapshot' else 2), 'robot_position': [float(boundary)] * 12, 'robot_velocity': [0.0] * 12, 'cloth_position': [[0.0, 0.0, 0.0]], 'cloth_velocity': [[0.0, 0.0, 0.0]], 'rng_state': {}, 'garment_name': ('Other_Garment' if case == 'garment-mismatch' else garment), 'randomization': {'strategy': snapshot_strategy, 'continuation_step': continuation_step}, 'scene_state': {}, 'cloth_state_authority': 'physx_cloth_view_world_v1'}
            h16.write_text(json.dumps(snapshot, sort_keys=True))
            entries[h16.relative_to(raw).as_posix()] = {'sha256': ('0' * 64 if case == 'bad-manifest' else hashlib.sha256(h16.read_bytes()).hexdigest()), 'size': h16.stat().st_size}
        (raw / 'SHA256SUMS.json').write_text(json.dumps(entries, sort_keys=True))
        if case == 'parent-symlink':
            real_raw = raw.with_name('real-raw'); raw.rename(real_raw); raw.symlink_to(real_raw, target_is_directory=True)
        package_entries = []
        for current, _, names in __import__('os').walk(root / 'accepted' / attempt):
            for name in names:
                item = Path(current) / name
                relative = item.relative_to(root / 'accepted' / attempt).as_posix()
                if relative != 'SHA256SUMS.json': package_entries.append({'relative_path': relative, 'sha256': hashlib.sha256(item.read_bytes()).hexdigest(), 'byte_size': item.stat().st_size})
        package_entries.sort(key=lambda item: item['relative_path'])
        digest = hashlib.sha256(json.dumps(package_entries, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        receipt = {'attempt_id': attempt, 'round_id': round_id, 'repository': 'ryanjin333/lehome-groot-n17-rollouts', 'remote_prefix': f'rollout-rounds/{round_id}/{attempt}', 'readback_verified': case != 'receipt-not-readback', 'episode_sha256': ('a' * 64 if case == 'receipt-mismatch' else digest), 'entry_count': len(package_entries), 'immutable_revision': 'b' * 40}
        receipts = root / 'hf-sync-receipts'; receipts.mkdir(); (receipts / f'{attempt}.sync.json').write_text(json.dumps(receipt))
        if case == 'strict-seal': (root / 'source.strict.seal.json').write_text('{}')
        if case == 'envelope-collision': (root / 'snapshot-source-bootstrap.envelope.json').write_text('{"sentinel":true}')
        raise SystemExit(1 if case == 'nonzero' else 0)
        PY
    """), encoding="utf-8")
    path.chmod(0o755)


def _run_snapshot_source_bootstrap(
    tmp_path: Path,
    case: str,
    *,
    garment: str = "Top_Long_Seen_0",
    category: str = "top_long",
    runtime_source_root: Path | None = None,
    clear_pythonpath: bool = False,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    # The contract rejects every symlink ancestor. pytest's ordinary macOS
    # temporary root is addressed through /var, a symlink to /private.
    if tmp_path.is_relative_to("/var"):
        tmp_path = Path("/System/Volumes/Data/private/var") / tmp_path.relative_to("/var")
    elif tmp_path.is_relative_to("/private"):
        tmp_path = Path("/System/Volumes/Data/private") / tmp_path.relative_to("/private")
    if runtime_source_root is not None and runtime_source_root.is_relative_to("/var"):
        runtime_source_root = Path("/System/Volumes/Data/private/var") / runtime_source_root.relative_to("/var")
    elif runtime_source_root is not None and runtime_source_root.is_relative_to("/private"):
        runtime_source_root = Path("/System/Volumes/Data/private") / runtime_source_root.relative_to("/private")
    descriptor = tmp_path / "source-descriptor.json"
    _write(descriptor, [{"snapshot_source_bootstrap": True, "category": category, "garment": garment, "seed": 50110, "source_seed": 50110}])
    digest = hashlib.sha256(descriptor.read_bytes()).hexdigest()
    run_id = hashlib.sha256(f"{tmp_path}:{case}".encode()).hexdigest()[:32]
    identity = hashlib.sha256(f"{run_id}:{digest}".encode("ascii")).hexdigest()[:20]
    root = tmp_path / "eval" / f"snapshot-source-bootstrap-{identity}"
    base = tmp_path / "snapshot-source-base.sh"; _fake_snapshot_source_base(base)
    fake_bin = tmp_path / "fake-bin"; fake_bin.mkdir(parents=True)
    docker = fake_bin / "docker"
    docker.write_text(textwrap.dedent("""\
        #!/usr/bin/env python3
        import os, sys
        if len(sys.argv) < 4 or sys.argv[1] != "run" or sys.argv[-3] != "-":
            raise SystemExit("unexpected validator docker invocation")
        for index, argument in enumerate(sys.argv[:-1]):
            if argument == "-e":
                key, value = sys.argv[index + 1].split("=", 1)
                os.environ[key] = value
        os.execv(sys.executable, [sys.executable, *sys.argv[-3:]])
    """), encoding="utf-8")
    docker.chmod(0o755)
    env = {**os.environ, "LEHOME_WORKSPACE": str(tmp_path), "LEHOME_SNAPSHOT_SOURCE_DESCRIPTOR": str(descriptor), "LEHOME_SNAPSHOT_SOURCE_DESCRIPTOR_SHA256": digest, "LEHOME_SNAPSHOT_SOURCE_RUN_ID": run_id, "LEHOME_SNAPSHOT_SOURCE_BASE_CAMPAIGN": str(base), "LEHOME_SNAPSHOT_SOURCE_TEST_CASE": case, "LEHOME_SNAPSHOT_SOURCE_RUNTIME_ROOT": str(runtime_source_root or (REPO_ROOT / "source" / "lehome"))}
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    if clear_pythonpath:
        env["PYTHONPATH"] = ""
    return subprocess.run(["/bin/bash", str(SNAPSHOT_SOURCE_BOOTSTRAP)], env=env, text=True, capture_output=True, check=False), root


def _run(tmp_path: Path, *, terminal: str = "accepted", receipt_ok: bool = True, fault: str = "", input_name: str = "inputs", zero_perturbation: bool = False, teacher_probe: bool = False) -> tuple[subprocess.CompletedProcess[str], Path]:
    generated = _artifacts(tmp_path / input_name)
    fake = tmp_path / "fake-base.sh"; _fake_base(fake, terminal=terminal, receipt_ok=receipt_ok)
    run_id = hashlib.sha256(str(tmp_path).encode()).hexdigest()[:32]
    root = tmp_path / "eval" / f"controlled-recovery-smoke-{run_id}"
    env = {**os.environ, "LEHOME_WORKSPACE": str(tmp_path), "LEHOME_CONTROLLED_RECOVERY_MATRIX": generated["matrix_path"], "LEHOME_CONTROLLED_RECOVERY_MATRIX_SHA256": generated["matrix_sha256"], "LEHOME_CONTROLLED_RECOVERY_MATERIALIZATION": generated["materialization_path"], "LEHOME_CONTROLLED_RECOVERY_MATERIALIZATION_SHA256": generated["materialization_sha256"], "LEHOME_CONTROLLED_RECOVERY_SMOKE_BASE_CAMPAIGN": str(fake), "LEHOME_CONTROLLED_RECOVERY_SMOKE_RUN_ID": run_id, "LEHOME_SMOKE_DESCRIPTOR_FAULT": fault, "LEHOME_CONTROLLED_RECOVERY_SMOKE_ZERO_PERTURBATION": "1" if zero_perturbation else "0", "LEHOME_CONTROLLED_RECOVERY_SMOKE_TEACHER_PROBE": "1" if teacher_probe else "0"}
    return subprocess.run(["/bin/bash", str(SMOKE)], env=env, text=True, capture_output=True, check=False), root


def _run_with_selected_row_mutation(
    tmp_path: Path,
    mutate: Callable[[dict[str, object], dict[str, object]], None],
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run a hash-consistent full artifact whose selected row is adversarial."""
    generated = _artifacts(tmp_path / "inputs")
    matrix = Path(generated["matrix_path"])
    materialization = Path(generated["materialization_path"])
    matrix_payload = json.loads(matrix.read_text(encoding="utf-8"))
    materialization_payload = json.loads(materialization.read_text(encoding="utf-8"))
    mutate(matrix_payload["rows"][0], materialization_payload["rows"][0])
    matrix_bytes = (json.dumps(matrix_payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    matrix.write_bytes(matrix_bytes)
    matrix_sha256 = hashlib.sha256(matrix_bytes).hexdigest()
    matrix.with_name(matrix.name + ".sha256").write_text(matrix_sha256 + "\n", encoding="ascii")
    materialization_payload["matrix_sha256"] = matrix_sha256
    for row in materialization_payload["rows"]:
        row["controlled_matrix_sha256"] = matrix_sha256
    materialization_bytes = (json.dumps(materialization_payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    materialization.write_bytes(materialization_bytes)
    materialization_sha256 = hashlib.sha256(materialization_bytes).hexdigest()
    materialization.with_name(materialization.name + ".sha256").write_text(materialization_sha256 + "\n", encoding="ascii")
    fake = tmp_path / "fake-base.sh"; _fake_base(fake)
    run_id = hashlib.sha256(str(tmp_path).encode()).hexdigest()[:32]
    root = tmp_path / "eval" / f"controlled-recovery-smoke-{run_id}"
    env = {**os.environ, "LEHOME_WORKSPACE": str(tmp_path), "LEHOME_CONTROLLED_RECOVERY_MATRIX": str(matrix), "LEHOME_CONTROLLED_RECOVERY_MATRIX_SHA256": matrix_sha256, "LEHOME_CONTROLLED_RECOVERY_MATERIALIZATION": str(materialization), "LEHOME_CONTROLLED_RECOVERY_MATERIALIZATION_SHA256": materialization_sha256, "LEHOME_CONTROLLED_RECOVERY_SMOKE_BASE_CAMPAIGN": str(fake), "LEHOME_CONTROLLED_RECOVERY_SMOKE_RUN_ID": run_id}
    return subprocess.run(["/bin/bash", str(SMOKE)], env=env, text=True, capture_output=True, check=False), root


def test_smoke_materializes_one_verified_row_and_requires_exact_success_receipt(tmp_path: Path) -> None:
    result, root = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    descriptors = list((root / "smoke-descriptors").glob("*.json"))
    assert len(descriptors) == 1
    row = json.loads(descriptors[0].read_text())[0]
    assert row["controlled_smoke"] is True
    assert row["controlled_smoke_matrix_sha256"]
    assert row["controlled_smoke_materialization_sha256"]
    assert not list(root.rglob("*.strict.seal.json"))


def test_smoke_passes_the_authenticated_selected_garment_and_disables_restarts(tmp_path: Path) -> None:
    result, root = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    launched = json.loads((root / "base-launch.json").read_text(encoding="utf-8"))
    descriptor = next((root / "smoke-descriptors").glob("*.json"))
    row = json.loads(descriptor.read_text(encoding="utf-8"))[0]
    assert launched == {"garment": row["garment"], "max_worker_restarts": "0"}


def test_smoke_zero_perturbation_control_derives_a_distinct_provenance_bound_descriptor(tmp_path: Path) -> None:
    normal, normal_root = _run(tmp_path / "normal")
    zero, zero_root = _run(tmp_path / "zero", zero_perturbation=True)
    assert normal.returncode == 0, normal.stderr
    assert zero.returncode == 0, zero.stderr
    normal_descriptor = next((normal_root / "smoke-descriptors").glob("*.json"))
    zero_descriptor = next((zero_root / "smoke-descriptors").glob("*.json"))
    normal_row = json.loads(normal_descriptor.read_text(encoding="utf-8"))[0]
    zero_row = json.loads(zero_descriptor.read_text(encoding="utf-8"))[0]
    assert normal_row["controlled_smoke_perturbation_mode"] == "bounded_perturbation_v1"
    assert zero_row["controlled_smoke_perturbation_mode"] == "zero_perturbation_control_v1"
    assert zero_row["controlled_smoke_zero_perturbation"] is True
    assert set(zero_row["perturbation_profile"].values()) == {0.0}
    assert zero_row["perturbation_fingerprint"] != normal_row["perturbation_fingerprint"]
    assert zero_row["source_state_perturbation_fingerprint"] != normal_row["source_state_perturbation_fingerprint"]
    assert zero_descriptor.read_bytes() != normal_descriptor.read_bytes()


def test_smoke_teacher_probe_is_identity_distinct_and_bound_into_descriptor_and_resume_validation(tmp_path: Path) -> None:
    normal, normal_root = _run(tmp_path / "normal")
    teacher, teacher_root = _run(tmp_path / "teacher", teacher_probe=True)
    zero_teacher, zero_teacher_root = _run(tmp_path / "zero-teacher", zero_perturbation=True, teacher_probe=True)
    assert normal.returncode == teacher.returncode == zero_teacher.returncode == 0
    normal_row = json.loads(next((normal_root / "smoke-descriptors").glob("*.json")).read_text())[0]
    teacher_row = json.loads(next((teacher_root / "smoke-descriptors").glob("*.json")).read_text())[0]
    zero_teacher_row = json.loads(next((zero_teacher_root / "smoke-descriptors").glob("*.json")).read_text())[0]
    assert teacher_row["controlled_smoke_teacher_probe"] is True
    assert teacher_row["controlled_smoke_perturbation_mode"] == "teacher_continuation_probe_v1"
    assert zero_teacher_row["controlled_smoke_perturbation_mode"] == "zero_perturbation_teacher_continuation_probe_v1"
    assert teacher_row["controlled_smoke_mode_identity"] not in {normal_row["controlled_smoke_mode_identity"], zero_teacher_row["controlled_smoke_mode_identity"]}


def test_smoke_rejects_a_missing_primary_garment_even_when_alias_is_valid(tmp_path: Path) -> None:
    def mutate(matrix_row: dict[str, object], hydrated_row: dict[str, object]) -> None:
        matrix_row["garment"] = ""
        hydrated_row["garment"] = ""

    result, _ = _run_with_selected_row_mutation(tmp_path, mutate)
    assert result.returncode != 0
    assert "selected row garment must be a safe non-empty identifier" in result.stderr


def test_smoke_rejects_an_unsafe_primary_garment_even_when_alias_is_valid(tmp_path: Path) -> None:
    def mutate(matrix_row: dict[str, object], hydrated_row: dict[str, object]) -> None:
        matrix_row["garment"] = "not a garment"
        hydrated_row["garment"] = "not a garment"

    result, _ = _run_with_selected_row_mutation(tmp_path, mutate)
    assert result.returncode != 0
    assert "selected row garment must be a safe non-empty identifier" in result.stderr


def test_smoke_rejects_a_garment_name_alias_that_differs_from_primary(tmp_path: Path) -> None:
    def mutate(matrix_row: dict[str, object], hydrated_row: dict[str, object]) -> None:
        matrix_row["garment_name"] = "top_long-seen"
        hydrated_row["garment_name"] = "top_long-seen"

    result, _ = _run_with_selected_row_mutation(tmp_path, mutate)
    assert result.returncode != 0
    assert "selected row garment_name must exactly match garment" in result.stderr


def test_smoke_rejects_an_unsafe_garment_name_alias(tmp_path: Path) -> None:
    def mutate(matrix_row: dict[str, object], hydrated_row: dict[str, object]) -> None:
        matrix_row["garment_name"] = "not an alias"
        hydrated_row["garment_name"] = "not an alias"

    result, _ = _run_with_selected_row_mutation(tmp_path, mutate)
    assert result.returncode != 0
    assert "selected row garment_name must be a safe non-empty identifier" in result.stderr


def test_smoke_reports_policy_rejection_distinctly_and_never_seals(tmp_path: Path) -> None:
    result, root = _run(tmp_path, terminal="rejected")
    assert result.returncode == 10
    assert "policy attempt was rejected" in result.stderr
    assert not list(root.rglob("*.strict.seal.json"))


def test_smoke_rejects_readback_receipt_mismatch(tmp_path: Path) -> None:
    result, root = _run(tmp_path, receipt_ok=False)
    assert result.returncode == 20
    assert not list(root.rglob("*.strict.seal.json"))


def test_smoke_descriptor_faults_leave_no_committed_partial_descriptor(tmp_path: Path) -> None:
    for fault in ("temp-sidecar", "temp-descriptor", "publish-sidecar", "publish-descriptor"):
        result, root = _run(tmp_path / fault, fault=fault)
        assert result.returncode != 0
        descriptors = root / "smoke-descriptors"
        if descriptors.exists():
            assert not list(descriptors.glob("*.json"))
            assert not list(descriptors.glob("*.json.sha256"))


def test_smoke_requires_fresh_run_id_and_refuses_terminal_resume(tmp_path: Path) -> None:
    first, root = _run(tmp_path)
    assert first.returncode == 0
    second, _ = _run(tmp_path, input_name="inputs-second")
    assert second.returncode != 0
    assert "already has a campaign root" in second.stderr
    generated = _artifacts(tmp_path / "second-inputs")
    fake = tmp_path / "resume-base.sh"; _fake_base(fake)
    run_id = hashlib.sha256(str(tmp_path).encode()).hexdigest()[:32]
    env = {**os.environ, "LEHOME_WORKSPACE": str(tmp_path), "LEHOME_CONTROLLED_RECOVERY_MATRIX": generated["matrix_path"], "LEHOME_CONTROLLED_RECOVERY_MATRIX_SHA256": generated["matrix_sha256"], "LEHOME_CONTROLLED_RECOVERY_MATERIALIZATION": generated["materialization_path"], "LEHOME_CONTROLLED_RECOVERY_MATERIALIZATION_SHA256": generated["materialization_sha256"], "LEHOME_CONTROLLED_RECOVERY_SMOKE_BASE_CAMPAIGN": str(fake), "LEHOME_CONTROLLED_RECOVERY_SMOKE_RUN_ID": run_id, "LEHOME_CONTROLLED_RECOVERY_SMOKE_RESUME": "1"}
    before = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))
    resumed = subprocess.run(["/bin/bash", str(SMOKE)], env=env, text=True, capture_output=True, check=False)
    assert resumed.returncode != 0
    assert "resume inputs are missing or unsafe" in resumed.stderr
    assert sorted(path.relative_to(root).as_posix() for path in root.rglob("*")) == before


def test_smoke_refuses_a_hash_valid_but_malformed_full_materialization(tmp_path: Path) -> None:
    generated = _artifacts(tmp_path / "inputs")
    materialization = Path(generated["materialization_path"])
    payload = json.loads(materialization.read_text())
    payload["rows"][0]["source_reset"] = "/unsafe/path.json"
    materialization.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(materialization.read_bytes()).hexdigest()
    materialization.with_name(materialization.name + ".sha256").write_text(digest + "\n", encoding="ascii")
    fake = tmp_path / "fake-base.sh"; _fake_base(fake)
    run_id = hashlib.sha256(str(tmp_path).encode()).hexdigest()[:32]
    env = {**os.environ, "LEHOME_WORKSPACE": str(tmp_path), "LEHOME_CONTROLLED_RECOVERY_MATRIX": generated["matrix_path"], "LEHOME_CONTROLLED_RECOVERY_MATRIX_SHA256": generated["matrix_sha256"], "LEHOME_CONTROLLED_RECOVERY_MATERIALIZATION": str(materialization), "LEHOME_CONTROLLED_RECOVERY_MATERIALIZATION_SHA256": digest, "LEHOME_CONTROLLED_RECOVERY_SMOKE_BASE_CAMPAIGN": str(fake), "LEHOME_CONTROLLED_RECOVERY_SMOKE_RUN_ID": run_id}
    result = subprocess.run(["/bin/bash", str(SMOKE)], env=env, text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert "materialization paths are unsafe" in result.stderr


def test_base_skip_seal_gate_rejects_any_non_smoke_tuple_before_docker(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.json"; matrix.write_text("[]", encoding="utf-8")
    digest = hashlib.sha256(matrix.read_bytes()).hexdigest()
    env = {**os.environ, "LEHOME_WORKSPACE": str(tmp_path), "LEHOME_ATTEMPT_MATRIX": str(matrix), "LEHOME_ATTEMPT_MATRIX_SHA256": digest, "LEHOME_SKIP_ROUND_SEAL": "1", "LEHOME_WORKER_COUNT": "4", "LEHOME_MAX_ATTEMPTS": "8", "LEHOME_TARGET_ACCEPTED": "8"}
    runner = REPO_ROOT / "rollout_appliance" / "run_12k_campaign.sh"
    result = subprocess.run(["/bin/bash", str(runner)], env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 2
    assert "reserved for the exact controlled-recovery smoke tuple" in result.stderr


def test_base_cpu_cloth_requires_the_exact_unsealed_snapshot_source_tuple(tmp_path: Path) -> None:
    matrix = tmp_path / "source.json"
    matrix.write_text(
        json.dumps([{
            "snapshot_source_bootstrap": True,
            "category": "top_short",
            "garment": "Top_Short_Seen_2",
            "seed": 50066,
            "source_seed": 50066,
        }]),
        encoding="utf-8",
    )
    digest = hashlib.sha256(matrix.read_bytes()).hexdigest()
    env = {
        **os.environ,
        "LEHOME_WORKSPACE": str(tmp_path),
        "LEHOME_CAMPAIGN_ROOT": str(tmp_path / "run"),
        "LEHOME_ATTEMPT_MATRIX": str(matrix),
        "LEHOME_ATTEMPT_MATRIX_SHA256": digest,
        "LEHOME_WORKER_COUNT": "1",
        "LEHOME_MAX_ATTEMPTS": "1",
        "LEHOME_TARGET_ACCEPTED": "1",
        "LEHOME_SIMULATOR_DEVICE": "cpu",
        "LEHOME_SNAPSHOT_SOURCE_BOOTSTRAP": "1",
        "LEHOME_ENABLE_HF_UPLOAD": "1",
        "LEHOME_SKIP_ROUND_SEAL": "0",
        "LEHOME_VALIDATE_MATRIX_ONLY": "1",
    }
    runner = REPO_ROOT / "rollout_appliance" / "run_12k_campaign.sh"
    result = subprocess.run(["/bin/bash", str(runner)], env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 2
    assert "exact unsealed snapshot-source bootstrap tuple" in result.stderr


def test_actual_base_campaign_smoke_shims_drain_the_upload_and_never_invoke_sealer(tmp_path: Path) -> None:
    """Exercise the real base shell runner with Docker/worker shims, not a fake base."""
    generated = _artifacts(tmp_path / "inputs")
    run_id = "d" * 32
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    log = tmp_path / "docker.log"
    (fake_bin / "mkdir").write_text("#!/bin/sh\nargs=''\nhas_path=0\nfor a in \"$@\"; do case \"$a\" in /eval*|/kitcache*) ;; -*) args=\"$args '$a'\" ;; *) args=\"$args '$a'\"; has_path=1 ;; esac; done\n[ \"$has_path\" = 1 ] || exit 0\neval /bin/mkdir $args\n", encoding="utf-8")
    (fake_bin / "stat").write_text("#!/bin/sh\nif [ \"$1\" = \"-c\" ]; then echo 1234:600; else exec /usr/bin/stat \"$@\"; fi\n", encoding="utf-8")
    docker = fake_bin / "docker"
    docker.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        printf '%s\\n' "$*" >> {str(log)!r}
        name=""; previous=""
        for item in "$@"; do if [ "$previous" = "--name" ]; then name="$item"; fi; previous="$item"; done
        if [ "$name" = "lehome-12k-policy" ]; then
          mkdir -p "$LEHOME_CAMPAIGN_ROOT/policy-receipts"
          printf '%s' '{{"ready":true,"policy_sha256":"e8531e9477b68ac8f7d9fc9564bb66ebfae51f828b44599c4777bd2eb3b72efa","runtime_device":"cuda:0"}}' > "$LEHOME_CAMPAIGN_ROOT/policy-receipts/ready.json"
        elif [[ "$name" = lehome-camp12k-w* ]]; then
          printf 'worker-garment=%s restarts=%s\\n' "$LEHOME_INITIAL_GARMENT" "$LEHOME_MAX_WORKER_RESTARTS" >> {str(log)!r}
          PYTHONPATH={str(REPO_ROOT)!r}/source/lehome python3 - "$LEHOME_CAMPAIGN_ROOT" "$LEHOME_ATTEMPT_MATRIX" "$LEHOME_ROUND_ID" <<'PY'
        import hashlib, json, sys
        from pathlib import Path
        from lehome.flywheel.task_ledger import TaskLedger
        root, matrix_path, round_id = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
        rows = json.loads(matrix_path.read_text())
        ledger = TaskLedger(root / 'ledger.sqlite3', attempt_matrix=rows, max_attempts=1, target_accepted=1)
        lease = ledger.lease_next('worker-1', lease_duration_ns=10**12)
        ledger.record_terminal('worker-1', lease.attempt.attempt_id, lease.lease_id, str(root / 'raw'))
        ledger.validate_terminal(lease.attempt.attempt_id, 'accepted', artifact_id=str(root / 'accepted' / lease.attempt.attempt_id))
        attempt = lease.attempt.attempt_id; ledger.close()
        accepted = root / 'accepted' / attempt; accepted.mkdir(parents=True, exist_ok=True)
        data = b'{{"ok":true}}'; (accepted / 'episode.json').write_bytes(data)
        entries = [{{'relative_path':'episode.json','sha256':hashlib.sha256(data).hexdigest(),'byte_size':len(data)}}]
        (root / 'hf-sync-receipts').mkdir(parents=True, exist_ok=True)
        (root / 'hf-sync-receipts' / f'{{attempt}}.sync.json').write_text(json.dumps({{'attempt_id':attempt,'round_id':round_id,'remote_prefix':f'rollout-rounds/{{round_id}}/{{attempt}}','readback_verified':True,'episode_sha256':hashlib.sha256(json.dumps(entries,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'immutable_revision':'a'*40}}))
        PY
        fi
        exit 0
    """), encoding="utf-8")
    for tool in (fake_bin / "mkdir", fake_bin / "stat", docker): tool.chmod(0o755)
    token = tmp_path / "token"; token.write_text("test-token", encoding="utf-8")
    root = tmp_path / "eval" / f"controlled-recovery-smoke-{run_id}"
    env = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}", "LEHOME_WORKSPACE": str(tmp_path), "LEHOME_CONTROLLED_RECOVERY_MATRIX": generated["matrix_path"], "LEHOME_CONTROLLED_RECOVERY_MATRIX_SHA256": generated["matrix_sha256"], "LEHOME_CONTROLLED_RECOVERY_MATERIALIZATION": generated["materialization_path"], "LEHOME_CONTROLLED_RECOVERY_MATERIALIZATION_SHA256": generated["materialization_sha256"], "LEHOME_CONTROLLED_RECOVERY_SMOKE_RUN_ID": run_id, "LEHOME_HF_TOKEN_FILE": str(token), "LEHOME_ROLLOUT_PREEMPTION_CONTEXT": str(tmp_path / "preemption.json"), "LEHOME_MAX_WORKER_RESTARTS": "0"}
    result = subprocess.run(["/bin/bash", str(SMOKE)], env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "--role sealer" not in log.read_text(encoding="utf-8")
    assert "worker-garment=pant_long-seen restarts=0" in log.read_text(encoding="utf-8")
    assert not list(root.rglob("*.strict.seal.json"))


def test_production_wrapper_remains_the_strict_four_worker_eight_target_contract() -> None:
    source = PRODUCTION.read_text(encoding="utf-8")
    assert 'WORKER_COUNT="${LEHOME_WORKER_COUNT:-4}"' in source
    assert 'TARGET_ACCEPTED="${LEHOME_TARGET_ACCEPTED:-8}"' in source
    assert 'LEHOME_WORKER_COUNT="4"' in source
    assert 'LEHOME_TARGET_ACCEPTED="8"' in source
    assert "LEHOME_CONTROLLED_RECOVERY_SMOKE_ZERO_PERTURBATION" not in source
    assert "LEHOME_CONTROLLED_RECOVERY_SMOKE_TEACHER_PROBE" not in source


def test_rollout_image_packages_the_unconditionally_sourced_worker_supervisor() -> None:
    dockerfile = (REPO_ROOT / "rollout_appliance" / "Dockerfile").read_text(encoding="utf-8")
    assert "rollout_appliance/worker_supervisor.sh" in dockerfile
    assert "chmod 0755 /opt/lehome/rollout_appliance" in dockerfile
    assert "bash -n /opt/lehome/rollout_appliance/worker_supervisor.sh" in dockerfile


def test_snapshot_source_bootstrap_is_a_separate_one_worker_unsealed_tuple() -> None:
    bootstrap = REPO_ROOT / "rollout_appliance" / "run_snapshot_source_bootstrap.sh"
    source = bootstrap.read_text(encoding="utf-8")
    base = (REPO_ROOT / "rollout_appliance" / "run_12k_campaign.sh").read_text(encoding="utf-8")
    installer = (REPO_ROOT / "infrastructure/nebius/packer/scripts/install-rollout.sh").read_text(encoding="utf-8")
    assert "LEHOME_WORKER_COUNT=1 LEHOME_MAX_ATTEMPTS=1 LEHOME_TARGET_ACCEPTED=1" in source
    assert "LEHOME_ENABLE_HF_UPLOAD=1 LEHOME_SKIP_ROUND_SEAL=1 LEHOME_SNAPSHOT_SOURCE_BOOTSTRAP=1" in source
    assert "LEHOME_SIMULATOR_DEVICE=cpu" in source
    assert "LEHOME_SIMULATOR_DEVICE=cuda:0" not in source
    assert "fresh absent run root; resume is forbidden" in source
    assert "readback_verified" in source
    assert "SNAPSHOT_SOURCE_BOOTSTRAP" in base and "snapshot-source-bootstrap" in base
    assert '--simulator-device "${SIMULATOR_DEVICE}"' in base
    assert "CPU cloth requires the exact unsealed snapshot-source bootstrap tuple" in base
    assert "run_snapshot_source_bootstrap.sh" in installer


def test_snapshot_source_bootstrap_classifies_rejection_and_infrastructure_distinctly(tmp_path: Path) -> None:
    rejected, rejected_root = _run_snapshot_source_bootstrap(tmp_path / "rejected", "rejected")
    assert rejected.returncode == 3
    assert not (rejected_root / "snapshot-source-bootstrap.envelope.json").exists()
    absent, absent_root = _run_snapshot_source_bootstrap(tmp_path / "absent", "absent-ledger")
    assert absent.returncode == 4
    assert not (absent_root / "snapshot-source-bootstrap.envelope.json").exists()
    nonzero, nonzero_root = _run_snapshot_source_bootstrap(tmp_path / "nonzero", "nonzero")
    assert nonzero.returncode == 4
    assert not (nonzero_root / "snapshot-source-bootstrap.envelope.json").exists()


def test_snapshot_source_bootstrap_passes_a_nondefault_descriptor_garment_before_launch(tmp_path: Path) -> None:
    result, root = _run_snapshot_source_bootstrap(
        tmp_path / "pant-long",
        "rejected",
        garment="Pant_Long_Seen_4",
        category="pant_long",
    )
    assert result.returncode == 3
    assert json.loads((root / "base-launch.json").read_text(encoding="utf-8")) == {
        "initial_garment": "Pant_Long_Seen_4",
        "simulator_device": "cpu",
    }


def test_snapshot_source_bootstrap_forces_the_isolated_cpu_simulator_lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEHOME_SIMULATOR_DEVICE", "cuda:0")
    result, root = _run_snapshot_source_bootstrap(tmp_path / "cpu-source", "rejected")
    assert result.returncode == 3
    assert json.loads((root / "base-launch.json").read_text(encoding="utf-8")) == {
        "initial_garment": "Top_Long_Seen_0",
        "simulator_device": "cpu",
    }


@pytest.mark.parametrize("case", ["no-h16", "bad-manifest", "receipt-not-readback", "receipt-mismatch", "strict-seal"])
def test_snapshot_source_bootstrap_fails_closed_after_accepted_terminal(tmp_path: Path, case: str) -> None:
    result, root = _run_snapshot_source_bootstrap(tmp_path / case, case)
    assert result.returncode == 4, result.stderr
    assert not (root / "snapshot-source-bootstrap.envelope.json").exists()


@pytest.mark.parametrize("case", ["bad-boundary-filename", "bad-snapshot", "garment-mismatch", "seed-mismatch", "wrong-continuation-step", "randomization-mismatch", "after-success", "parent-symlink"])
def test_snapshot_source_bootstrap_requires_a_usable_authenticated_h16_source(tmp_path: Path, case: str) -> None:
    result, root = _run_snapshot_source_bootstrap(tmp_path / case, case)
    assert result.returncode == 4, result.stderr
    assert not (root / "snapshot-source-bootstrap.envelope.json").exists()


def test_snapshot_source_bootstrap_uses_the_physical_boundary_when_the_policy_observation_lags(tmp_path: Path) -> None:
    result, root = _run_snapshot_source_bootstrap(tmp_path / "lagged", "lagged-policy-state")
    assert result.returncode == 0, result.stderr
    assert (root / "snapshot-source-bootstrap.envelope.json").is_file()


@pytest.mark.parametrize("case", ["bad-request-chunk", "success-unlatched"])
def test_snapshot_source_bootstrap_requires_an_audit_v4_trace(tmp_path: Path, case: str) -> None:
    result, root = _run_snapshot_source_bootstrap(tmp_path / case, case)
    assert result.returncode == 4, result.stderr
    assert not (root / "snapshot-source-bootstrap.envelope.json").exists()


def test_snapshot_source_bootstrap_writes_one_audit_only_envelope_atomically(tmp_path: Path) -> None:
    from lehome_train.groot.rollout_source_adapter import _seal

    result, root = _run_snapshot_source_bootstrap(tmp_path / "accepted", "accepted")
    assert result.returncode == 0, result.stderr
    envelope = root / "snapshot-source-bootstrap.envelope.json"
    payload = json.loads(envelope.read_text(encoding="utf-8"))
    assert payload["kind"] == "snapshot_source_bootstrap_envelope"
    assert payload["source_only"] is True
    assert payload["readback_verified"] is True
    assert len(payload["episode_sha256s"]) == len(payload["immutable_revisions"]) == 1
    assert json.loads((root / "base-launch.json").read_text(encoding="utf-8")) == {
        "initial_garment": "Top_Long_Seen_0",
        "simulator_device": "cpu",
    }
    assert not list(root.glob("*.strict.seal.json"))
    with pytest.raises(ValueError):
        _seal(envelope)
    collision, collision_root = _run_snapshot_source_bootstrap(tmp_path / "collision", "envelope-collision")
    assert collision.returncode == 4
    assert json.loads((collision_root / "snapshot-source-bootstrap.envelope.json").read_text()) == {"sentinel": True}


def test_snapshot_source_bootstrap_imports_from_the_explicit_packaged_runtime_root(tmp_path: Path) -> None:
    staged = tmp_path / "opt" / "lehome" / "source" / "lehome"
    shutil.copytree(REPO_ROOT / "source" / "lehome", staged)
    result, root = _run_snapshot_source_bootstrap(
        tmp_path / "staged", "accepted", runtime_source_root=staged, clear_pythonpath=True,
    )
    assert result.returncode == 0, result.stderr
    assert (root / "snapshot-source-bootstrap.envelope.json").is_file()


def test_snapshot_source_bootstrap_uses_the_dependency_complete_rollout_runtime_for_validation() -> None:
    source = SNAPSHOT_SOURCE_BOOTSTRAP.read_text(encoding="utf-8")
    assert "docker run --rm --user 1234:1234 --network none -i" in source
    assert '-v "${DESCRIPTOR}:${DESCRIPTOR}:ro"' in source
    assert "--entrypoint /opt/lehome-challenge/.venv/bin/python" in source
    assert 'PYTHONPATH="${RUNTIME_SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" python3' not in source


def test_snapshot_source_bootstrap_missing_packaged_runtime_root_is_infrastructure_failure(tmp_path: Path) -> None:
    result, root = _run_snapshot_source_bootstrap(
        tmp_path / "missing", "accepted", runtime_source_root=tmp_path / "missing-package", clear_pythonpath=True,
    )
    assert result.returncode == 4, result.stderr
    assert "post-acceptance evidence verification failed" in result.stderr
    assert not (root / "snapshot-source-bootstrap.envelope.json").exists()


def test_snapshot_source_bootstrap_rejects_a_packaged_runtime_root_with_a_symlink_ancestor(tmp_path: Path) -> None:
    staged_parent = tmp_path / "staged-parent"
    staged = staged_parent / "source" / "lehome"
    shutil.copytree(REPO_ROOT / "source" / "lehome", staged)
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(staged_parent, target_is_directory=True)
    result, root = _run_snapshot_source_bootstrap(
        tmp_path / "symlinked", "accepted", runtime_source_root=linked_parent / "source" / "lehome", clear_pythonpath=True,
    )
    assert result.returncode == 4, result.stderr
    assert not (root / "snapshot-source-bootstrap.envelope.json").exists()
