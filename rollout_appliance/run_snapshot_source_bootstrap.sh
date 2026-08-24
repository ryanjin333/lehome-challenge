#!/usr/bin/env bash
# Collect a bounded set of fresh ordinary autonomous H16 snapshot sources.
# This is source discovery, not controlled recovery and never emits a seal.
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
TARGET_ACCEPTED="${LEHOME_SNAPSHOT_SOURCE_TARGET_ACCEPTED:-1}"
ROLLOUT_REPOSITORY="${LEHOME_ROLLOUT_REPOSITORY:-ryanjin333/lehome-groot-n17-rollouts}"
HF_REVISION="${LEHOME_HF_REVISION:-main}"
SOURCE_FINALIZATION_TIMEOUT_SECONDS="${LEHOME_SOURCE_FINALIZATION_TIMEOUT_SECONDS:-300}"
# Appliance/operator input only.  Empty is invalid rather than silently
# defaulting, so an intended CUDA source collection cannot fall back to CPU.
SNAPSHOT_SOURCE_SIMULATOR_DEVICE="${LEHOME_SNAPSHOT_SOURCE_SIMULATOR_DEVICE-cpu}"

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
case "${SNAPSHOT_SOURCE_SIMULATOR_DEVICE}" in
  cpu|cuda:0) ;;
  *) echo "snapshot source bootstrap simulator device must be exactly cpu or cuda:0" >&2; exit 2 ;;
esac
if [[ "${DESCRIPTOR}" != /* ]] || [ -L "${DESCRIPTOR}" ] || [ ! -f "${DESCRIPTOR}" ] || [ ! -x "${BASE}" ]; then
  echo "snapshot source bootstrap input is unsafe" >&2; exit 2
fi
if [ "$(sha256sum "${DESCRIPTOR}" | awk '{print $1}')" != "${DESCRIPTOR_SHA256}" ]; then
  echo "snapshot source bootstrap descriptor hash changed" >&2; exit 2
fi
if ! safe_runtime_source_root "${RUNTIME_SOURCE_ROOT}"; then
  echo "snapshot source bootstrap packaged runtime source is missing or unsafe" >&2; exit 4
fi
if ! docker run --rm --user 1234:1234 --network none -i \
  -v "${WORKSPACE}:${WORKSPACE}:ro" \
  -v "${DESCRIPTOR}:${DESCRIPTOR}:ro" \
  -v "${RUNTIME_SOURCE_ROOT}:${RUNTIME_SOURCE_ROOT}:ro" \
  -e "PYTHONPATH=${RUNTIME_SOURCE_ROOT}" \
  --entrypoint /opt/lehome-challenge/.venv/bin/python \
  "${ROLLOUT_IMAGE}" - "${DESCRIPTOR}" <<'PY'
import sys
import json
from pathlib import Path
from lehome.flywheel.recovery_collection import (
    validate_snapshot_source_descriptor,
    validate_snapshot_source_discovery_descriptor,
)
rows = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if len(rows) == 1 and rows[0].get("replay_kind") == "verified_success_reset_v1":
    validate_snapshot_source_descriptor(sys.argv[1])
else:
    validate_snapshot_source_discovery_descriptor(sys.argv[1])
PY
then
  echo "snapshot source bootstrap descriptor preflight failed" >&2; exit 2
fi
INITIAL_GARMENT="$(python3 - "${DESCRIPTOR}" <<'PY'
import json, re, sys
from pathlib import Path

try:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, ValueError) as error:
    raise SystemExit(f"snapshot source bootstrap descriptor is unreadable: {error}")
if not isinstance(payload, list) or not 1 <= len(payload) <= 16 or not isinstance(payload[0], dict):
    raise SystemExit("snapshot source bootstrap descriptor must contain 1..16 assignments")
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
SOURCE_ROW_COUNT="$(python3 - "${DESCRIPTOR}" <<'PY'
import json, sys
from pathlib import Path
try:
    rows = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, ValueError) as error:
    raise SystemExit(f"snapshot source bootstrap descriptor is unreadable: {error}")
if not isinstance(rows, list) or not 1 <= len(rows) <= 16:
    raise SystemExit("snapshot source bootstrap descriptor must contain 1..16 assignments")
print(len(rows))
PY
)" || exit 2
case "${TARGET_ACCEPTED}" in
  ''|*[!0-9]*) echo "snapshot source discovery target must be a positive integer" >&2; exit 2 ;;
esac
if [ "${TARGET_ACCEPTED}" -lt 1 ] || [ "${TARGET_ACCEPTED}" -gt 4 ] || [ "${TARGET_ACCEPTED}" -gt "${SOURCE_ROW_COUNT}" ]; then
  echo "snapshot source discovery target must be in 1..min(4, rows)" >&2; exit 2
fi
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
  LEHOME_WORKER_COUNT=1 LEHOME_MAX_ATTEMPTS="${SOURCE_ROW_COUNT}" LEHOME_TARGET_ACCEPTED="${TARGET_ACCEPTED}" \
  LEHOME_INITIAL_GARMENT="${INITIAL_GARMENT}" \
  LEHOME_MAX_WORKER_RESTARTS=0 \
  LEHOME_ROLLOUT_REPOSITORY="${ROLLOUT_REPOSITORY}" LEHOME_HF_REVISION="${HF_REVISION}" \
  LEHOME_SOURCE_FINALIZATION_TIMEOUT_SECONDS="${SOURCE_FINALIZATION_TIMEOUT_SECONDS}" \
  LEHOME_SIMULATOR_DEVICE="${SNAPSHOT_SOURCE_SIMULATOR_DEVICE}" \
  LEHOME_ENABLE_HF_UPLOAD=1 LEHOME_SKIP_ROUND_SEAL=1 LEHOME_SNAPSHOT_SOURCE_BOOTSTRAP=1 \
  LEHOME_CONTROLLED_RECOVERY_SMOKE=0 LEHOME_RESUME_PREEMPTED_ROLLOUT=0 \
  bash "${BASE}"
status=$?
set -e
terminal="$(python3 - "${ROOT}/ledger.sqlite3" "${TARGET_ACCEPTED}" <<'PY'
import sqlite3, sys
from pathlib import Path
p, target = Path(sys.argv[1]), int(sys.argv[2])
try:
    if not p.is_file() or p.is_symlink(): raise ValueError("unsafe ledger")
    con=sqlite3.connect(f"{p.as_uri()}?mode=ro", uri=True)
    rows=list(con.execute("select event_type, attempt_id from events"))
except Exception:
    print("infrastructure")
else:
    con.close()
    kinds={row[0] for row in rows}
    accepted={row[1] for row in rows if row[0] == "accepted" and isinstance(row[1], str)}
    if "infrastructure_abort" in kinds or len(accepted) > target:
        print("infrastructure")
    elif not accepted and "rejected" in kinds:
        print("rejected")
    elif accepted:
        print("accepted")
    else:
        print("incomplete")
PY
)"
if [ "${terminal}" = "infrastructure" ]; then
  echo "snapshot source bootstrap infrastructure ledger failure; no source is admitted" >&2; exit 4
fi
if [ "${status}" != 0 ]; then
  echo "snapshot source bootstrap infrastructure failure; no source is admitted" >&2; exit 4
fi
if [ "${terminal}" = "rejected" ]; then
  echo "snapshot source bootstrap policy rejection; no source is admitted" >&2; exit 3
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
  "${ROLLOUT_IMAGE}" - "${ROOT}" "${DESCRIPTOR}" "${TARGET_ACCEPTED}" \
  "${ROLLOUT_REPOSITORY}" "snapshot-source-bootstrap-${identity}-unsealed-source" "${HF_REVISION}" <<'PY'
import hashlib, json, os, re, sqlite3, sys, tempfile
from pathlib import Path
root, descriptor, target = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]); expected_repository, expected_round_id, expected_publication_ref = sys.argv[4:7]; ledger = root / "ledger.sqlite3"; receipts = root / "hf-sync-receipts"
if not ledger.is_file() or ledger.is_symlink() or not receipts.is_dir() or receipts.is_symlink(): raise SystemExit("snapshot source bootstrap receipt evidence is missing")
con = sqlite3.connect(f"{ledger.as_uri()}?mode=ro", uri=True)
try:
    accepted_ids = [row[0] for row in con.execute("SELECT DISTINCT attempt_id FROM events WHERE event_type='accepted' ORDER BY attempt_id")]
    attempt_rows = dict(con.execute("SELECT attempt_id, assignment_json FROM attempts"))
    infrastructure = con.execute("SELECT COUNT(*) FROM events WHERE event_type='infrastructure_abort'").fetchone()[0]
finally: con.close()
files = sorted(receipts.iterdir())
if infrastructure or not 1 <= len(accepted_ids) <= target or len(files) != len(accepted_ids): raise SystemExit("snapshot source bootstrap accepted set is invalid")
if any(not isinstance(attempt_id, str) or re.fullmatch(r"[0-9a-f]{64}", attempt_id) is None for attempt_id in accepted_ids): raise SystemExit("snapshot source bootstrap accepted attempt identity is invalid")
if any(path.is_symlink() or not path.is_file() for path in files): raise SystemExit("snapshot source bootstrap receipt path is unsafe")
try: descriptor_rows = json.loads(descriptor.read_text(encoding="utf-8"))
except (OSError, ValueError) as error: raise SystemExit(f"snapshot source descriptor is malformed: {error}")
from lehome.flywheel.recovery_collection import (
    validate_snapshot_source_bootstrap_evidence,
    validate_snapshot_source_descriptor,
    validate_snapshot_source_discovery_descriptor,
)
if len(descriptor_rows) == 1 and descriptor_rows[0].get("replay_kind") == "verified_success_reset_v1":
    descriptor_rows = [validate_snapshot_source_descriptor(descriptor)]
else:
    descriptor_rows = validate_snapshot_source_discovery_descriptor(descriptor)
canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
descriptor_by_assignment = {canonical(row): row for row in descriptor_rows}
if len(descriptor_by_assignment) != len(descriptor_rows): raise SystemExit("snapshot source descriptor assignments are not unique")
receipts_by_attempt = {}
receipt_fields = {
    "schema_version", "attempt_id", "repository", "round_id", "remote_prefix",
    "publication_ref", "immutable_revision", "entry_count", "episode_sha256",
    "readback_verified",
}
def strict_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value: raise ValueError("duplicate receipt field")
        value[key] = item
    return value
def reject_constant(value): raise ValueError(f"invalid JSON constant: {value}")
def canonical_mutable_ref(value):
    if not isinstance(value, str) or len(value) > 128 or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}", value) is None:
        return False
    return all(component not in {"", ".", ".."} and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", component) is not None for component in value.split("/")) and re.fullmatch(r"[0-9a-f]{40}", value) is None
if (re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}/[A-Za-z0-9][A-Za-z0-9._-]{0,127}", expected_repository) is None
        or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,127}", expected_round_id) is None
        or not canonical_mutable_ref(expected_publication_ref)):
    raise SystemExit("snapshot source bootstrap active uploader configuration is invalid")
for path in files:
    try: receipt = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=strict_object, parse_constant=reject_constant)
    except (OSError, ValueError) as error: raise SystemExit(f"snapshot source receipt is malformed: {error}")
    if not isinstance(receipt, dict): raise SystemExit("snapshot source bootstrap receipt attempt identity is invalid")
    attempt_id = receipt.get("attempt_id")
    if (set(receipt) != receipt_fields
            or type(receipt.get("schema_version")) is not int or receipt["schema_version"] != 1
            or not isinstance(attempt_id, str) or re.fullmatch(r"[0-9a-f]{64}", attempt_id) is None
            or path.name != f"{attempt_id}.sync.json"
            or attempt_id in receipts_by_attempt):
        raise SystemExit("snapshot source bootstrap receipt attempt identity is invalid")
    receipts_by_attempt[attempt_id] = receipt
if set(receipts_by_attempt) != set(accepted_ids): raise SystemExit("snapshot source bootstrap receipts do not match accepted attempts")
episode_sha256s, immutable_revisions, round_ids, repositories = {}, {}, set(), set()
for attempt_id in accepted_ids:
    assignment_json = attempt_rows.get(attempt_id)
    if not isinstance(assignment_json, str): raise SystemExit("snapshot source accepted attempt has no immutable assignment")
    try: descriptor_row = descriptor_by_assignment[canonical(json.loads(assignment_json))]
    except (ValueError, KeyError): raise SystemExit("snapshot source accepted attempt does not match an immutable descriptor row")
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
    receipt = receipts_by_attempt[attempt_id]
    round_id, repository, revision = receipt.get("round_id"), receipt.get("repository"), receipt.get("immutable_revision")
    publication_ref = receipt.get("publication_ref")
    if (receipt.get("readback_verified") is not True
            or not isinstance(round_id, str) or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,127}", round_id) is None
            or not isinstance(repository, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}/[A-Za-z0-9][A-Za-z0-9._-]{0,127}", repository) is None
            or round_id != expected_round_id or repository != expected_repository
            or not canonical_mutable_ref(publication_ref) or publication_ref != expected_publication_ref
            or receipt.get("remote_prefix") != f"rollout-rounds/{round_id}/{attempt_id}"
            or not isinstance(receipt.get("episode_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", receipt["episode_sha256"]) is None
            or receipt["episode_sha256"] != episode_sha256
            or type(receipt.get("entry_count")) is not int or receipt["entry_count"] <= 0
            or receipt["entry_count"] != len(entries)
            or not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None): raise SystemExit("snapshot source bootstrap receipt lineage is invalid")
    validate_snapshot_source_bootstrap_evidence(accepted_root=accepted, descriptor_row=descriptor_row)
    episode_sha256s[attempt_id] = episode_sha256; immutable_revisions[attempt_id] = revision
    round_ids.add(round_id); repositories.add(repository)
if len(round_ids) != 1 or len(repositories) != 1: raise SystemExit("snapshot source bootstrap receipts do not share a round lineage")
if list(root.glob("*.strict.seal.json")): raise SystemExit("snapshot source bootstrap must not create a strict trainable round seal")
body = {"schema_version": 1, "kind": "snapshot_source_bootstrap_envelope", "round_id": round_ids.pop(), "repository": repositories.pop(), "episode_count": len(accepted_ids), "episode_sha256s": episode_sha256s, "immutable_revisions": immutable_revisions, "readback_verified": True, "source_only": True}
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
