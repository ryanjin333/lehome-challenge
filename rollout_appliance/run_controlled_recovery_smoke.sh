#!/usr/bin/env bash
# Exactly one real controlled-recovery attempt before the four-worker campaign.
#
# This command produces an *unsealed staging* artifact.  A successful smoke is
# evidence that restore -> audited prefix replay -> bounded perturbation ->
# autonomous policy continuation works; it is never training data by itself.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FULL_WRAPPER="${SCRIPT_DIR}/run_controlled_recovery_campaign.sh"
BASE_CAMPAIGN="${LEHOME_CONTROLLED_RECOVERY_SMOKE_BASE_CAMPAIGN:-${SCRIPT_DIR}/run_12k_campaign.sh}"
WORKSPACE="${LEHOME_WORKSPACE:-/mnt/lehome}"
MATRIX="${LEHOME_CONTROLLED_RECOVERY_MATRIX:-}"
MATRIX_SHA256="${LEHOME_CONTROLLED_RECOVERY_MATRIX_SHA256:-}"
MATERIALIZATION="${LEHOME_CONTROLLED_RECOVERY_MATERIALIZATION:-}"
MATERIALIZATION_SHA256="${LEHOME_CONTROLLED_RECOVERY_MATERIALIZATION_SHA256:-}"
RUN_ID="${LEHOME_CONTROLLED_RECOVERY_SMOKE_RUN_ID:-}"
ROW_INDEX="${LEHOME_CONTROLLED_RECOVERY_SMOKE_ROW_INDEX:-0}"
RESUME="${LEHOME_CONTROLLED_RECOVERY_SMOKE_RESUME:-0}"
ZERO_PERTURBATION="${LEHOME_CONTROLLED_RECOVERY_SMOKE_ZERO_PERTURBATION:-0}"
TEACHER_PROBE="${LEHOME_CONTROLLED_RECOVERY_SMOKE_TEACHER_PROBE:-0}"
PREEMPTION_CONTEXT="${LEHOME_ROLLOUT_PREEMPTION_CONTEXT:-/run/lehome/rollout-preemption.json}"
EXPECTED_ROOT="${WORKSPACE}/eval/controlled-recovery-smoke-${RUN_ID}"
CAMPAIGN_ROOT="${LEHOME_CAMPAIGN_ROOT:-${EXPECTED_ROOT}}"
ORIGINAL_12K_POLICY_SHA256="e8531e9477b68ac8f7d9fc9564bb66ebfae51f828b44599c4777bd2eb3b72efa"
ORIGINAL_12K_CHECKPOINT="${LEHOME_ORIGINAL_12K_CHECKPOINT:-${WORKSPACE}/eval/policies/original_baseline}"

for path in "${FULL_WRAPPER}" "${BASE_CAMPAIGN}"; do
  if [ -L "${path}" ] || [ ! -f "${path}" ]; then
    echo "controlled recovery smoke appliance is incomplete or unsafe" >&2
    exit 2
  fi
done
for value in "${MATRIX_SHA256}" "${MATERIALIZATION_SHA256}"; do
  if ! [[ "${value}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "controlled recovery smoke requires pinned lowercase SHA-256 values" >&2
    exit 2
  fi
done
if ! [[ "${RUN_ID}" =~ ^[0-9a-f]{32}$ ]] || ! [[ "${ROW_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "controlled recovery smoke requires a 32-hex run ID and a non-negative row index" >&2
  exit 2
fi
if [ "${RESUME}" != "0" ] && [ "${RESUME}" != "1" ]; then
  echo "LEHOME_CONTROLLED_RECOVERY_SMOKE_RESUME must be exactly 0 or 1" >&2
  exit 2
fi
if [ "${ZERO_PERTURBATION}" != "0" ] && [ "${ZERO_PERTURBATION}" != "1" ]; then
  echo "LEHOME_CONTROLLED_RECOVERY_SMOKE_ZERO_PERTURBATION must be exactly 0 or 1" >&2
  exit 2
fi
if [ "${TEACHER_PROBE}" != "0" ] && [ "${TEACHER_PROBE}" != "1" ]; then
  echo "LEHOME_CONTROLLED_RECOVERY_SMOKE_TEACHER_PROBE must be exactly 0 or 1" >&2
  exit 2
fi
if [ "${CAMPAIGN_ROOT}" != "${EXPECTED_ROOT}" ]; then
  echo "controlled recovery smoke campaign root must bind the supplied run ID" >&2
  exit 2
fi

# Reuse the production wrapper's canonical full-artifact verifier.  The
# validation-only switch stops before its four-worker execution path.
FULL_ATTEMPTS="$(python3 - "${MATERIALIZATION}" <<'PY'
import json, sys
from pathlib import Path
try:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not 8 <= len(rows) <= 96: raise ValueError
    print(len(rows))
except (OSError, ValueError, json.JSONDecodeError):
    raise SystemExit("controlled recovery smoke materialization attempt count is invalid")
PY
)"
env \
  LEHOME_CONTROLLED_RECOVERY_MATRIX="${MATRIX}" \
  LEHOME_CONTROLLED_RECOVERY_MATRIX_SHA256="${MATRIX_SHA256}" \
  LEHOME_CONTROLLED_RECOVERY_MATERIALIZATION="${MATERIALIZATION}" \
  LEHOME_CONTROLLED_RECOVERY_MATERIALIZATION_SHA256="${MATERIALIZATION_SHA256}" \
  LEHOME_WORKER_COUNT=4 LEHOME_MAX_ATTEMPTS="${FULL_ATTEMPTS}" LEHOME_TARGET_ACCEPTED=8 \
  LEHOME_CONTROLLED_RECOVERY_VALIDATE_ONLY=1 \
  bash "${FULL_WRAPPER}"

# A resume is a read-only authentication step until every persisted identity
# agrees.  In particular, it must not repair or recreate a descriptor from a
# merely similar materialization after a preemption.
if [ "${RESUME}" = "1" ]; then
  python3 - "${CAMPAIGN_ROOT}" "${PREEMPTION_CONTEXT}" "${MATRIX}" "${MATRIX_SHA256}" "${MATERIALIZATION}" "${MATERIALIZATION_SHA256}" "${RUN_ID}" "${ROW_INDEX}" "${ZERO_PERTURBATION}" "${TEACHER_PROBE}" <<'PY'
import hashlib, json, sqlite3, stat, sys
from pathlib import Path

root, context_path = Path(sys.argv[1]), Path(sys.argv[2])
matrix, matrix_sha, materialization, materialization_sha, run_id, row_index, zero, teacher = Path(sys.argv[3]), sys.argv[4], Path(sys.argv[5]), sys.argv[6], sys.argv[7], int(sys.argv[8]), sys.argv[9], sys.argv[10]
if any(path.is_symlink() or not path.is_file() for path in (context_path, matrix, materialization)):
    raise SystemExit("controlled smoke resume inputs are missing or unsafe")
try:
    context = json.loads(context_path.read_text(encoding="utf-8"))
    hydrated = json.loads(materialization.read_text(encoding="utf-8"))
except (OSError, ValueError) as error: raise SystemExit(f"controlled smoke resume inputs are unreadable: {error}")
identity = hashlib.sha256(f"{run_id}:{matrix_sha}:{materialization_sha}".encode("ascii")).hexdigest()[:20]
mode = ("zero_perturbation_teacher_continuation_probe_v1" if zero == "1" else "teacher_continuation_probe_v1") if teacher == "1" else ("zero_perturbation_control_v1" if zero == "1" else "bounded_perturbation_v1")
mode_identity = hashlib.sha256(f"{identity}:{mode}".encode("ascii")).hexdigest()[:20]
directory = root / "smoke-descriptors"
descriptor = directory / f"controlled-recovery-smoke-{identity}-{mode_identity}-row-{row_index}.json"
sidecar = Path(str(descriptor) + ".sha256")
if (root.is_symlink() or not root.is_dir() or directory.is_symlink()
        or any(path.is_symlink() or not path.is_file() for path in (descriptor, sidecar))):
    raise SystemExit("controlled smoke resume descriptor is missing or unsafe")
descriptor_bytes = descriptor.read_bytes(); descriptor_sha = hashlib.sha256(descriptor_bytes).hexdigest()
if sidecar.read_text(encoding="ascii").strip() != descriptor_sha:
    raise SystemExit("controlled smoke resume descriptor sidecar mismatch")
try: rows = json.loads(descriptor_bytes)
except ValueError: raise SystemExit("controlled smoke resume descriptor is malformed")
if not isinstance(hydrated, dict) or not isinstance(hydrated.get("rows"), list) or not 0 <= row_index < len(hydrated["rows"]):
    raise SystemExit("controlled smoke resume row index is invalid")
expected = dict(hydrated["rows"][row_index])
if zero == "1":
    profile = expected.get("perturbation_profile")
    required = {"cloth_displacement_m", "cloth_velocity_mps", "gripper_offset_rad"}
    if not isinstance(profile, dict) or set(profile) != required or any(type(profile[key]) not in (int, float) for key in required):
        raise SystemExit("controlled smoke zero control source profile is malformed")
    zero_profile = {key: 0.0 for key in sorted(required)}
    seed, episode_digest, snapshot = expected.get("perturbation_seed"), expected.get("source_episode_digest"), expected.get("source_continuation_snapshot_sha256")
    source_state = expected.get("source_state_fingerprint")
    if type(seed) is not int or not all(isinstance(value, str) and len(value) == 64 for value in (episode_digest, snapshot, source_state)):
        raise SystemExit("controlled smoke zero control source provenance is malformed")
    perturbation = hashlib.sha256((json.dumps({**zero_profile, "seed": seed, "source_episode_digest": episode_digest, "continuation_snapshot_sha256": snapshot}, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()).hexdigest()
    expected.update({"perturbation_profile": zero_profile, "perturbation_fingerprint": perturbation, "source_state_perturbation_fingerprint": hashlib.sha256((json.dumps({"source_state_fingerprint": source_state, "perturbation_fingerprint": perturbation}, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()).hexdigest()})
expected.update({"controlled_smoke": True, "controlled_smoke_run_id": run_id, "controlled_smoke_row_index": row_index, "controlled_smoke_identity": identity, "controlled_smoke_mode_identity": mode_identity, "controlled_smoke_perturbation_mode": mode, "controlled_smoke_zero_perturbation": zero == "1", "controlled_smoke_teacher_probe": teacher == "1", "controlled_smoke_matrix_sha256": matrix_sha, "controlled_smoke_materialization_sha256": materialization_sha})
if not isinstance(rows, list) or rows != [expected]: raise SystemExit("controlled smoke resume descriptor does not match the requested row")
required_context = {
    "schema_version": 1, "kind": "lehome_rollout_preemption_context", "active": True,
    "run_id": run_id, "run_root": str(root), "database": str(root / "ledger.sqlite3"),
    "attempt_matrix": str(descriptor), "attempt_matrix_sha256": descriptor_sha,
    "max_attempts": 1, "target_accepted": 1, "controlled_recovery_smoke": True,
    "controlled_recovery_smoke_run_id": run_id,
    "controlled_recovery_smoke_matrix_sha256": matrix_sha,
    "controlled_recovery_smoke_materialization_sha256": materialization_sha,
    "controlled_recovery_smoke_row_index": row_index,
}
if context != required_context: raise SystemExit("controlled smoke resume preemption context does not match the exact smoke identity")
ledger = root / "ledger.sqlite3"
if ledger.is_symlink() or not ledger.is_file(): raise SystemExit("controlled smoke resume ledger is missing or unsafe")
try:
    con = sqlite3.connect(f"{ledger.as_uri()}?mode=ro", uri=True)
    attempts = con.execute("SELECT attempt_id, assignment_json FROM attempts ORDER BY schedule_index").fetchall()
    events = con.execute("SELECT event_type FROM events ORDER BY event_id").fetchall()
finally:
    try: con.close()
    except NameError: pass
if len(attempts) != 1 or json.loads(attempts[0][1]) != expected:
    raise SystemExit("controlled smoke resume ledger does not bind the requested descriptor row")
derived_attempt = hashlib.sha256(json.dumps({"schedule_index": 0, "assignment": expected}, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
if attempts[0][0] != derived_attempt or not events or events[-1][0] != "campaign_paused":
    raise SystemExit("controlled smoke resume requires one paused, nonterminal preempted attempt")
if any(event[0] in {"accepted", "rejected", "infrastructure_abort", "campaign_ended", "campaign_resumed"} for event in events):
    raise SystemExit("controlled smoke resume refuses terminal or already-resumed state")
PY
fi

descriptor_info="$(python3 - "${CAMPAIGN_ROOT}" "${MATRIX}" "${MATRIX_SHA256}" "${MATERIALIZATION}" "${MATERIALIZATION_SHA256}" "${RUN_ID}" "${ROW_INDEX}" "${RESUME}" "${LEHOME_SMOKE_DESCRIPTOR_FAULT:-}" "${ZERO_PERTURBATION}" "${TEACHER_PROBE}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
matrix_path = Path(sys.argv[2]); matrix_digest = sys.argv[3]
materialization_path = Path(sys.argv[4]); materialization_digest = sys.argv[5]
run_id, row_index, resume, fault, zero, teacher = sys.argv[6], int(sys.argv[7]), sys.argv[8], sys.argv[9], sys.argv[10], sys.argv[11]
if root.is_symlink(): raise SystemExit("controlled smoke campaign root is unsafe")
if root.exists():
    if not root.is_dir(): raise SystemExit("controlled smoke campaign root is unsafe")
    ledger = root / "ledger.sqlite3"
    if resume != "1": raise SystemExit("controlled smoke run ID already has a campaign root; use a fresh run ID")
    if not ledger.is_file() or ledger.is_symlink(): raise SystemExit("controlled smoke resume requires an existing real ledger")
    try:
        con = __import__("sqlite3").connect(f"{ledger.as_uri()}?mode=ro", uri=True)
        terminal = con.execute("SELECT COUNT(*) FROM events WHERE event_type IN ('accepted','rejected','infrastructure_abort')").fetchone()[0]
    except Exception as error: raise SystemExit(f"controlled smoke resume ledger is unreadable: {error}")
    finally:
        try: con.close()
        except NameError: pass
    if terminal: raise SystemExit("controlled smoke resume refuses an already terminal-settled run")
elif resume == "1":
    raise SystemExit("controlled smoke resume requires an existing campaign root")
else:
    root.mkdir(parents=True, mode=0o755)
root = root.resolve()
for path, digest in ((matrix_path, matrix_digest), (materialization_path, materialization_digest)):
    if not path.is_absolute() or path.is_symlink() or not path.is_file(): raise SystemExit("controlled smoke input is unsafe")
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest: raise SystemExit("controlled smoke input hash changed after validation")
try:
    hydrated = json.loads(materialization_path.read_text(encoding="utf-8"))
except (OSError, ValueError) as error: raise SystemExit(f"controlled smoke materialization unreadable: {error}")
if not isinstance(hydrated, dict) or not isinstance(hydrated.get("rows"), list) or not 0 <= row_index < len(hydrated["rows"]) or not isinstance(hydrated["rows"][row_index], dict):
    raise SystemExit("controlled smoke row index is outside the full materialization")
row = dict(hydrated["rows"][row_index])
garment = row.get("garment")
if not isinstance(garment, str) or re.fullmatch(r"[A-Za-z0-9_.-]+", garment) is None:
    raise SystemExit("controlled smoke selected row garment must be a safe non-empty identifier")
garment_name = row.get("garment_name")
if garment_name is not None:
    if not isinstance(garment_name, str) or re.fullmatch(r"[A-Za-z0-9_.-]+", garment_name) is None:
        raise SystemExit("controlled smoke selected row garment_name must be a safe non-empty identifier")
    if garment_name != garment:
        raise SystemExit("controlled smoke selected row garment_name must exactly match garment")
identity = hashlib.sha256(f"{run_id}:{matrix_digest}:{materialization_digest}".encode("ascii")).hexdigest()[:20]
mode = ("zero_perturbation_teacher_continuation_probe_v1" if zero == "1" else "teacher_continuation_probe_v1") if teacher == "1" else ("zero_perturbation_control_v1" if zero == "1" else "bounded_perturbation_v1")
mode_identity = hashlib.sha256(f"{identity}:{mode}".encode("ascii")).hexdigest()[:20]
if zero == "1":
    profile = row.get("perturbation_profile")
    required = {"cloth_displacement_m", "cloth_velocity_mps", "gripper_offset_rad"}
    if not isinstance(profile, dict) or set(profile) != required or any(type(profile[key]) not in (int, float) for key in required):
        raise SystemExit("controlled smoke zero control source profile is malformed")
    zero_profile = {key: 0.0 for key in sorted(required)}
    seed, episode_digest, snapshot = row.get("perturbation_seed"), row.get("source_episode_digest"), row.get("source_continuation_snapshot_sha256")
    source_state = row.get("source_state_fingerprint")
    if type(seed) is not int or not all(isinstance(value, str) and len(value) == 64 for value in (episode_digest, snapshot, source_state)):
        raise SystemExit("controlled smoke zero control source provenance is malformed")
    perturbation = hashlib.sha256((json.dumps({**zero_profile, "seed": seed, "source_episode_digest": episode_digest, "continuation_snapshot_sha256": snapshot}, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()).hexdigest()
    row.update({"perturbation_profile": zero_profile, "perturbation_fingerprint": perturbation, "source_state_perturbation_fingerprint": hashlib.sha256((json.dumps({"source_state_fingerprint": source_state, "perturbation_fingerprint": perturbation}, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()).hexdigest()})
row.update({"controlled_smoke": True, "controlled_smoke_run_id": run_id, "controlled_smoke_row_index": row_index, "controlled_smoke_identity": identity, "controlled_smoke_mode_identity": mode_identity, "controlled_smoke_perturbation_mode": mode, "controlled_smoke_zero_perturbation": zero == "1", "controlled_smoke_teacher_probe": teacher == "1", "controlled_smoke_matrix_sha256": matrix_digest, "controlled_smoke_materialization_sha256": materialization_digest})
directory = root / "smoke-descriptors"
directory.mkdir(mode=0o755, exist_ok=True)
if directory.is_symlink() or not directory.is_dir() or not directory.resolve().is_relative_to(root): raise SystemExit("controlled smoke descriptor directory is unsafe")
path = directory / f"controlled-recovery-smoke-{identity}-{mode_identity}-row-{row_index}.json"
sidecar = Path(str(path) + ".sha256")
payload = (json.dumps([row], sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
digest = hashlib.sha256(payload).hexdigest()
def fsync_dir() -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)
def safe_existing(target: Path, data: bytes) -> bool:
    return target.exists() and not target.is_symlink() and target.is_file() and target.read_bytes() == data
sidecar_bytes = (digest + "\n").encode("ascii")
if path.exists():
    if not safe_existing(path, payload) or not safe_existing(sidecar, sidecar_bytes): raise SystemExit("controlled smoke descriptor already exists with different or partial content")
elif sidecar.exists() and not safe_existing(sidecar, sidecar_bytes):
    raise SystemExit("controlled smoke descriptor sidecar is unsafe or mismatched")
else:
    created_sidecar = False
    temporary_descriptor = directory / f".{path.name}.{os.getpid()}.descriptor.tmp"
    temporary_sidecar = directory / f".{path.name}.{os.getpid()}.sidecar.tmp"
    def temp_write(target: Path, data: bytes, phase: str) -> None:
        if fault == phase: raise OSError(f"injected {phase} fault")
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream: stream.write(data); stream.flush(); os.fsync(stream.fileno())
    try:
        temp_write(temporary_sidecar, sidecar_bytes, "temp-sidecar")
        temp_write(temporary_descriptor, payload, "temp-descriptor")
        if not sidecar.exists():
            if fault == "publish-sidecar": raise OSError("injected publish-sidecar fault")
            os.link(temporary_sidecar, sidecar); created_sidecar = True; fsync_dir()
        elif not safe_existing(sidecar, sidecar_bytes): raise SystemExit("controlled smoke descriptor sidecar is unsafe or mismatched")
        if fault == "publish-descriptor": raise OSError("injected publish-descriptor fault")
        os.link(temporary_descriptor, path); fsync_dir()
    except Exception:
        if created_sidecar:
            sidecar.unlink(missing_ok=True); fsync_dir()
        raise
    finally:
        temporary_descriptor.unlink(missing_ok=True); temporary_sidecar.unlink(missing_ok=True)
print(path); print(digest); print(identity); print(garment)
PY
)"
descriptor_line_count="$(printf '%s\n' "${descriptor_info}" | wc -l | tr -d '[:space:]')"
if [ "${descriptor_line_count}" != "4" ]; then
  echo "controlled smoke descriptor creation returned malformed output" >&2
  exit 2
fi
DESCRIPTOR="$(printf '%s\n' "${descriptor_info}" | sed -n '1p')"
DESCRIPTOR_SHA256="$(printf '%s\n' "${descriptor_info}" | sed -n '2p')"
IDENTITY="$(printf '%s\n' "${descriptor_info}" | sed -n '3p')"
INITIAL_GARMENT="$(printf '%s\n' "${descriptor_info}" | sed -n '4p')"
ROUND_ID="controlled-recovery-smoke-${IDENTITY}-unsealed-staging"

set +e
env \
  LEHOME_WORKSPACE="${WORKSPACE}" \
  LEHOME_POLICY_SHA256="${ORIGINAL_12K_POLICY_SHA256}" \
  LEHOME_RUN_ID="${RUN_ID}" \
  LEHOME_CHECKPOINT_DIR="${ORIGINAL_12K_CHECKPOINT}" \
  LEHOME_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" \
  LEHOME_ATTEMPT_MATRIX="${DESCRIPTOR}" \
  LEHOME_ATTEMPT_MATRIX_SHA256="${DESCRIPTOR_SHA256}" \
  LEHOME_WORKER_COUNT=1 LEHOME_MAX_ATTEMPTS=1 LEHOME_TARGET_ACCEPTED=1 \
  LEHOME_INITIAL_GARMENT="${INITIAL_GARMENT}" LEHOME_MAX_WORKER_RESTARTS=0 \
  LEHOME_ENABLE_HF_UPLOAD=1 LEHOME_ROUND_ID="${ROUND_ID}" \
  LEHOME_SKIP_ROUND_SEAL=1 LEHOME_CONTROLLED_RECOVERY_SMOKE=1 \
  LEHOME_RESUME_PREEMPTED_ROLLOUT="${RESUME}" \
  LEHOME_CONTROLLED_RECOVERY_SMOKE_RUN_ID="${RUN_ID}" \
  LEHOME_CONTROLLED_RECOVERY_SMOKE_ROW_INDEX="${ROW_INDEX}" \
  LEHOME_CONTROLLED_RECOVERY_MATRIX_SHA256="${MATRIX_SHA256}" \
  LEHOME_CONTROLLED_RECOVERY_MATERIALIZATION_SHA256="${MATERIALIZATION_SHA256}" \
  bash "${BASE_CAMPAIGN}"
base_status=$?
set -e
if [ "${base_status}" -ne 0 ]; then
  echo "controlled recovery smoke infrastructure failure (base campaign status ${base_status})" >&2
  exit 20
fi

set +e
python3 - "${CAMPAIGN_ROOT}" "${DESCRIPTOR}" "${DESCRIPTOR_SHA256}" "${ROUND_ID}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys

root, descriptor, descriptor_sha, round_id = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
if root.is_symlink() or not root.is_dir() or descriptor.is_symlink() or not descriptor.is_file(): raise SystemExit(20)
if hashlib.sha256(descriptor.read_bytes()).hexdigest() != descriptor_sha: raise SystemExit(20)
try: rows = json.loads(descriptor.read_text(encoding="utf-8"))
except (OSError, ValueError): raise SystemExit(20)
if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict): raise SystemExit(20)
if any(root.rglob("*.strict.seal.json")): raise SystemExit(20)
database = root / "ledger.sqlite3"
try:
    connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
    attempts = connection.execute("SELECT attempt_id, assignment_json FROM attempts ORDER BY schedule_index").fetchall()
    if len(attempts) != 1 or json.loads(attempts[0][1]) != rows[0]: raise SystemExit(20)
    attempt_id = attempts[0][0]
    events = connection.execute("SELECT event_type FROM events WHERE attempt_id=? ORDER BY event_id", (attempt_id,)).fetchall()
finally:
    try: connection.close()
    except NameError: pass
if not events or events[-1][0] != "accepted" or not any(event[0] == "terminal_pending_validation" for event in events):
    # A clean, terminal policy rejection is a real smoke result, not an
    # infrastructure fault.  The wrapper's stable status is intentionally 10.
    raise SystemExit(10)
accepted = root / "accepted" / attempt_id
receipt = root / "hf-sync-receipts" / f"{attempt_id}.sync.json"
if accepted.is_symlink() or not accepted.is_dir() or receipt.is_symlink() or not receipt.is_file(): raise SystemExit(20)
entries = []
for current, directories, names in os.walk(accepted, followlinks=False):
    current_path = Path(current)
    if any((current_path / name).is_symlink() for name in directories): raise SystemExit(20)
    for name in names:
        path = current_path / name
        if path.is_symlink(): raise SystemExit(20)
        relative = path.relative_to(accepted).as_posix()
        if relative == "SHA256SUMS.json": continue
        data = path.read_bytes()
        entries.append({"relative_path": relative, "sha256": hashlib.sha256(data).hexdigest(), "byte_size": len(data)})
entries.sort(key=lambda item: item["relative_path"])
episode_sha = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
try: payload = json.loads(receipt.read_text(encoding="utf-8"))
except (OSError, ValueError): raise SystemExit(20)
if (payload.get("attempt_id") != attempt_id or payload.get("round_id") != round_id
        or payload.get("remote_prefix") != f"rollout-rounds/{round_id}/{attempt_id}"
        or payload.get("readback_verified") is not True
        or payload.get("episode_sha256") != episode_sha
        or not isinstance(payload.get("immutable_revision"), str)
        or re.fullmatch(r"[0-9a-f]{40}", payload["immutable_revision"]) is None): raise SystemExit(20)
PY
status=$?
set -e
case "${status}" in
  0) exit 0 ;;
  10) echo "controlled recovery smoke completed but the policy attempt was rejected" >&2; exit 10 ;;
  *) echo "controlled recovery smoke evidence is incomplete or inconsistent" >&2; exit 20 ;;
esac
