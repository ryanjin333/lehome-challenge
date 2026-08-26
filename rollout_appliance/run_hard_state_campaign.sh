#!/usr/bin/env bash
# Run checksum-bound CPU moment-of-ruin recoveries through the proven 12K appliance.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_CAMPAIGN="${SCRIPT_DIR}/run_12k_campaign.sh"
WORKSPACE="${LEHOME_WORKSPACE:-/mnt/lehome}"
MATRIX="${LEHOME_HARD_STATE_MATRIX:-${WORKSPACE}/eval/campaign-12k-round-3/hard-state-nearmiss.json}"
EXPECTED_SHA256="${LEHOME_HARD_STATE_MATRIX_SHA256:-}"
WORKER_COUNT="${LEHOME_WORKER_COUNT:-4}"
MAX_ATTEMPTS="${LEHOME_MAX_ATTEMPTS:-24}"
TARGET_ACCEPTED="${LEHOME_TARGET_ACCEPTED:-8}"
MAX_WORKER_RESTARTS="${LEHOME_MAX_WORKER_RESTARTS:-8}"
FRESH_LEDGER="${LEHOME_FRESH_LEDGER:-1}"
CAMPAIGN_ROOT="${LEHOME_CAMPAIGN_ROOT:-${WORKSPACE}/eval/campaign-hard-state-nearmiss-1}"
ROUND_ID="${LEHOME_ROUND_ID:-hard-state-nearmiss-round-1}"
RUN_ID="${LEHOME_RUN_ID:-hard-state-nearmiss-run-1}"
ORIGINAL_12K_POLICY_SHA256="e8531e9477b68ac8f7d9fc9564bb66ebfae51f828b44599c4777bd2eb3b72efa"
ORIGINAL_12K_CHECKPOINT="${LEHOME_ORIGINAL_12K_CHECKPOINT:-${WORKSPACE}/eval/policies/original_baseline}"

if [ ! -f "${BASE_CAMPAIGN}" ]; then
  echo "missing reusable 12K campaign appliance" >&2
  exit 2
fi
case "${MATRIX}" in
  /*) ;;
  *) echo "LEHOME_HARD_STATE_MATRIX must be an absolute path" >&2; exit 2 ;;
esac
case "${FRESH_LEDGER}" in
  "0"|"1") ;;
  *) echo "LEHOME_FRESH_LEDGER must be exactly 0 or 1" >&2; exit 2 ;;
esac
if [ -L "${MATRIX}" ] || [ ! -f "${MATRIX}" ] || [ -L "${MATRIX}.sha256" ] || [ ! -f "${MATRIX}.sha256" ]; then
  echo "hard-state matrix and receipt must be regular files" >&2
  exit 2
fi
if ! [[ "${EXPECTED_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "LEHOME_HARD_STATE_MATRIX_SHA256 must be a lowercase 64-character SHA-256" >&2
  exit 2
fi
ACTUAL_SHA256="$(sha256sum "${MATRIX}" | awk '{print $1}')"
RECEIPT_SHA256="$(tr -d '\r\n' < "${MATRIX}.sha256")"
if [ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ] || [ "${RECEIPT_SHA256}" != "${EXPECTED_SHA256}" ]; then
  echo "hard-state matrix SHA-256 mismatch" >&2
  exit 2
fi
PYTHONPATH="/opt/lehome/source/lehome:/opt/lehome${PYTHONPATH:+:${PYTHONPATH}}" python3 - \
  "${MATRIX}" "${MAX_ATTEMPTS}" "${TARGET_ACCEPTED}" <<'PY'
import sys
from pathlib import Path
from lehome.flywheel.recovery_collection import validate_hard_state_descriptor

matrix, max_attempts, target_accepted = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
rows = validate_hard_state_descriptor(matrix)
caps = {row["category"]: row["category_acceptance_cap"] for row in rows}
if len(rows) != max_attempts:
    raise SystemExit("hard-state matrix must match LEHOME_MAX_ATTEMPTS")
if sum(caps.values()) != target_accepted:
    raise SystemExit("hard-state category caps do not match LEHOME_TARGET_ACCEPTED")
PY

if [ "${FRESH_LEDGER}" = "1" ]; then
  rm -f "${CAMPAIGN_ROOT}/ledger.sqlite3" "${CAMPAIGN_ROOT}/ledger.sqlite3-wal" "${CAMPAIGN_ROOT}/ledger.sqlite3-shm"
fi

exec env \
  LEHOME_POLICY_SHA256="${ORIGINAL_12K_POLICY_SHA256}" \
  LEHOME_CHECKPOINT_DIR="${ORIGINAL_12K_CHECKPOINT}" \
  LEHOME_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" \
  LEHOME_ATTEMPT_MATRIX="${MATRIX}" \
  LEHOME_ATTEMPT_MATRIX_SHA256="${EXPECTED_SHA256}" \
  LEHOME_WORKER_COUNT="${WORKER_COUNT}" \
  LEHOME_SIMULATOR_DEVICE="cpu" \
  LEHOME_HARD_STATE_CAMPAIGN="1" \
  LEHOME_ENABLE_HF_UPLOAD="1" \
  LEHOME_SKIP_ROUND_SEAL="0" \
  LEHOME_MAX_ATTEMPTS="${MAX_ATTEMPTS}" \
  LEHOME_MAX_WORKER_RESTARTS="${MAX_WORKER_RESTARTS}" \
  LEHOME_TARGET_ACCEPTED="${TARGET_ACCEPTED}" \
  LEHOME_ROUND_ID="${ROUND_ID}" \
  LEHOME_RUN_ID="${RUN_ID}" \
  bash "${BASE_CAMPAIGN}"
