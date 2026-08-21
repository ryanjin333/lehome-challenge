#!/usr/bin/env bash
# Collect exactly one fresh ordinary autonomous success with H16 snapshots.
# This is a source bootstrap, not controlled recovery and never emits a seal.
set -euo pipefail

WORKSPACE="${LEHOME_WORKSPACE:-/mnt/lehome}"
DESCRIPTOR="${LEHOME_SNAPSHOT_SOURCE_DESCRIPTOR:?LEHOME_SNAPSHOT_SOURCE_DESCRIPTOR is required}"
DESCRIPTOR_SHA256="${LEHOME_SNAPSHOT_SOURCE_DESCRIPTOR_SHA256:?LEHOME_SNAPSHOT_SOURCE_DESCRIPTOR_SHA256 is required}"
RUN_ID="${LEHOME_SNAPSHOT_SOURCE_RUN_ID:?LEHOME_SNAPSHOT_SOURCE_RUN_ID is required}"
BASE="${LEHOME_SNAPSHOT_SOURCE_BASE_CAMPAIGN:-/opt/lehome/rollout_appliance/run_12k_campaign.sh}"
ROLLOUT_IMAGE="${LEHOME_ROLLOUT_IMAGE:-lehome-rollout:build}"
# This is appliance configuration, never descriptor-controlled input.  The
# post-acceptance validator mounts it into the dependency-complete rollout
# runtime and never inherits the host Python environment.
RUNTIME_SOURCE_ROOT="${LEHOME_SNAPSHOT_SOURCE_RUNTIME_ROOT:-/opt/lehome/source/lehome}"

safe_runtime_source_root() {
  local current="$1"
  [[ "${current}" == /* ]] && [ -d "${current}" ] || return 1
  while :; do
    [ -L "${current}" ] && return 1
    [ "${current}" = "/" ] && return 0
    current="${current%/*}"
    [ -n "${current}" ] || current="/"
  done
}

if ! [[ "${RUN_ID}" =~ ^[0-9a-f]{32}$ ]] || ! [[ "${DESCRIPTOR_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "snapshot source bootstrap requires fixed run and descriptor identities" >&2; exit 2
fi
if [[ "${DESCRIPTOR}" != /* ]] || [ -L "${DESCRIPTOR}" ] || [ ! -f "${DESCRIPTOR}" ] || [ ! -x "${BASE}" ]; then
  echo "snapshot source bootstrap input is unsafe" >&2; exit 2
fi
if [ "$(sha256sum "${DESCRIPTOR}" | awk '{print $1}')" != "${DESCRIPTOR_SHA256}" ]; then
  echo "snapshot source bootstrap descriptor hash changed" >&2; exit 2
fi
INITIAL_GARMENT="$(python3 - "${DESCRIPTOR}" <<'PY'
import json, re, sys
from pathlib import Path

try:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, ValueError) as error:
    raise SystemExit(f"snapshot source bootstrap descriptor is unreadable: {error}")
if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
    raise SystemExit("snapshot source bootstrap descriptor must contain exactly one assignment")
row = payload[0]
garment = row.get("garment")
if not isinstance(garment, str) or re.fullmatch(r"[A-Za-z0-9_.-]+", garment) is None:
    raise SystemExit("snapshot source bootstrap garment is invalid")
garment_name = row.get("garment_name")
if garment_name is not None and garment_name != garment:
    raise SystemExit("snapshot source bootstrap garment identity is inconsistent")
print(garment)
PY
)" || exit 2
identity="$(python3 - "${RUN_ID}" "${DESCRIPTOR_SHA256}" <<'PY'
import hashlib, sys
print(hashlib.sha256(f"{sys.argv[1]}:{sys.argv[2]}".encode("ascii")).hexdigest()[:20])
PY
)"
ROOT="${WORKSPACE}/eval/snapshot-source-bootstrap-${identity}"
if [ -e "${ROOT}" ] || [ -L "${ROOT}" ]; then
  echo "snapshot source bootstrap requires a fresh absent run root; resume is forbidden" >&2; exit 2
fi

set +e
env LEHOME_WORKSPACE="${WORKSPACE}" LEHOME_CAMPAIGN_ROOT="${ROOT}" \
  LEHOME_ATTEMPT_MATRIX="${DESCRIPTOR}" LEHOME_ATTEMPT_MATRIX_SHA256="${DESCRIPTOR_SHA256}" \
  LEHOME_RUN_ID="${RUN_ID}" LEHOME_ROUND_ID="snapshot-source-bootstrap-${identity}-unsealed-source" \
  LEHOME_WORKER_COUNT=1 LEHOME_MAX_ATTEMPTS=1 LEHOME_TARGET_ACCEPTED=1 \
  LEHOME_INITIAL_GARMENT="${INITIAL_GARMENT}" \
  LEHOME_SIMULATOR_DEVICE=cpu \
  LEHOME_ENABLE_HF_UPLOAD=1 LEHOME_SKIP_ROUND_SEAL=1 LEHOME_SNAPSHOT_SOURCE_BOOTSTRAP=1 \
  LEHOME_CONTROLLED_RECOVERY_SMOKE=0 LEHOME_RESUME_PREEMPTED_ROLLOUT=0 \
  bash "${BASE}"
status=$?
set -e
terminal="$(python3 - "${ROOT}/ledger.sqlite3" <<'PY'
import sqlite3, sys
from pathlib import Path
p=Path(sys.argv[1])
try:
    if not p.is_file() or p.is_symlink(): raise ValueError("unsafe ledger")
    con=sqlite3.connect(f"{p.as_uri()}?mode=ro", uri=True)
    rows={row[0] for row in con.execute("select event_type from events")}
except Exception:
    print("infrastructure")
else:
    con.close()
    print("rejected" if "rejected" in rows else ("accepted" if "accepted" in rows else "incomplete"))
PY
)"
if [ "${terminal}" = "rejected" ]; then
  echo "snapshot source bootstrap policy rejection; no source is admitted" >&2; exit 3
fi
if [ "${terminal}" = "infrastructure" ]; then
  echo "snapshot source bootstrap infrastructure ledger failure; no source is admitted" >&2; exit 4
fi
if [ "${status}" != 0 ]; then
  echo "snapshot source bootstrap infrastructure failure; no source is admitted" >&2; exit 4
fi
if [ "${terminal}" != "accepted" ]; then
  echo "snapshot source bootstrap has no accepted terminal outcome" >&2; exit 4
fi
set +e
if ! safe_runtime_source_root "${RUNTIME_SOURCE_ROOT}"; then
  echo "snapshot source bootstrap packaged runtime source is missing or unsafe" >&2
  verification_status=1
else
docker run --rm --user 1234:1234 --network none -i \
  -v "${WORKSPACE}:${WORKSPACE}" \
  -v "${DESCRIPTOR}:${DESCRIPTOR}:ro" \
  -v "${RUNTIME_SOURCE_ROOT}:${RUNTIME_SOURCE_ROOT}:ro" \
  -e "PYTHONPATH=${RUNTIME_SOURCE_ROOT}" \
  --entrypoint /opt/lehome-challenge/.venv/bin/python \
  "${ROLLOUT_IMAGE}" - "${ROOT}" "${DESCRIPTOR}" <<'PY'
import hashlib, json, os, re, sqlite3, sys, tempfile
from pathlib import Path
root, descriptor = Path(sys.argv[1]), Path(sys.argv[2]); ledger = root / "ledger.sqlite3"; receipts = root / "hf-sync-receipts"
if not ledger.is_file() or ledger.is_symlink() or not receipts.is_dir() or receipts.is_symlink(): raise SystemExit("snapshot source bootstrap receipt evidence is missing")
con = sqlite3.connect(f"{ledger.as_uri()}?mode=ro", uri=True)
try:
    accepted = con.execute("SELECT COUNT(*) FROM events WHERE event_type='accepted'").fetchone()[0]
finally: con.close()
files = sorted(receipts.glob("*.sync.json"))
if accepted != 1 or len(files) != 1: raise SystemExit("snapshot source bootstrap requires exactly one accepted terminal and one receipt")
receipt = json.loads(files[0].read_text(encoding="utf-8"))
if receipt.get("readback_verified") is not True: raise SystemExit("snapshot source bootstrap receipt is not readback-verified")
attempt_id = receipt.get("attempt_id")
if not isinstance(attempt_id, str) or not attempt_id: raise SystemExit("snapshot source bootstrap receipt attempt identity is invalid")
accepted = root / "accepted" / attempt_id
if accepted.is_symlink() or not accepted.is_dir(): raise SystemExit("snapshot source bootstrap accepted artifact is missing")
entries = []
for current, directories, names in os.walk(accepted, followlinks=False):
    current_path = Path(current)
    if any((current_path / name).is_symlink() for name in directories): raise SystemExit("snapshot source bootstrap accepted artifact is unsafe")
    for name in names:
        candidate = current_path / name
        if candidate.is_symlink() or not candidate.is_file(): raise SystemExit("snapshot source bootstrap accepted artifact is unsafe")
        relative = candidate.relative_to(accepted).as_posix()
        if relative == "SHA256SUMS.json": continue
        data = candidate.read_bytes()
        entries.append({"relative_path": relative, "sha256": hashlib.sha256(data).hexdigest(), "byte_size": len(data)})
entries.sort(key=lambda item: item["relative_path"])
episode_sha256 = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
round_id, repository, revision = receipt.get("round_id"), receipt.get("repository"), receipt.get("immutable_revision")
if (not isinstance(round_id, str) or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,127}", round_id) is None
        or not isinstance(repository, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}/[A-Za-z0-9][A-Za-z0-9._-]{0,127}", repository) is None
        or receipt.get("remote_prefix") != f"rollout-rounds/{round_id}/{attempt_id}"
        or receipt.get("episode_sha256") != episode_sha256
        or receipt.get("entry_count") != len(entries)
        or not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None): raise SystemExit("snapshot source bootstrap receipt lineage is invalid")
from lehome.flywheel.recovery_collection import validate_snapshot_source_bootstrap_evidence
validate_snapshot_source_bootstrap_evidence(accepted_root=accepted, descriptor_path=descriptor)
if list(root.glob("*.strict.seal.json")): raise SystemExit("snapshot source bootstrap must not create a strict trainable round seal")
body = {"schema_version": 1, "kind": "snapshot_source_bootstrap_envelope", "round_id": round_id, "repository": repository, "episode_count": 1, "episode_sha256s": {attempt_id: episode_sha256}, "immutable_revisions": {attempt_id: revision}, "readback_verified": True, "source_only": True}
if not isinstance(body["round_id"], str) or not isinstance(body["repository"], str) or not all(isinstance(value, str) and value for mapping in (body["episode_sha256s"], body["immutable_revisions"]) for value in mapping.values()): raise SystemExit("snapshot source bootstrap receipt lineage is invalid")
body["envelope_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
path = root / "snapshot-source-bootstrap.envelope.json"
if path.exists() or path.is_symlink(): raise SystemExit("snapshot source bootstrap envelope target already exists")
fd, temporary = tempfile.mkstemp(prefix=".snapshot-source-bootstrap.", suffix=".tmp", dir=root)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, path)
    directory = os.open(root, os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)
finally:
    Path(temporary).unlink(missing_ok=True)
PY
verification_status=$?
fi
set -e
if [ "${verification_status}" != 0 ]; then
  echo "snapshot source bootstrap post-acceptance evidence verification failed; no source is admitted" >&2; exit 4
fi
