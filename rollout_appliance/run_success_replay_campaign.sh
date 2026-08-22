#!/usr/bin/env bash
# Replay checksum-verified original-12K success resets through the proven 12K appliance.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_CAMPAIGN="${SCRIPT_DIR}/run_12k_campaign.sh"
WORKSPACE="${LEHOME_WORKSPACE:-/mnt/lehome}"
MATRIX="${LEHOME_SUCCESS_REPLAY_MATRIX:-}"
EXPECTED_SHA256="${LEHOME_SUCCESS_REPLAY_MATRIX_SHA256:-}"
WORKER_COUNT="${LEHOME_WORKER_COUNT:-4}"
MAX_ATTEMPTS="${LEHOME_MAX_ATTEMPTS:-200}"
TARGET_ACCEPTED="${LEHOME_TARGET_ACCEPTED:-150}"
MAX_WORKER_RESTARTS="${LEHOME_MAX_WORKER_RESTARTS:-8}"
CAMPAIGN_ROOT="${LEHOME_CAMPAIGN_ROOT:-${WORKSPACE}/eval/campaign-success-replay-12k-round-1}"
ROUND_ID="${LEHOME_ROUND_ID:-success-replay-12k-round-1}"
ORIGINAL_12K_POLICY_SHA256="e8531e9477b68ac8f7d9fc9564bb66ebfae51f828b44599c4777bd2eb3b72efa"
ORIGINAL_12K_CHECKPOINT="${LEHOME_ORIGINAL_12K_CHECKPOINT:-${WORKSPACE}/eval/policies/original_baseline}"

if [ ! -x "${BASE_CAMPAIGN}" ] && [ ! -f "${BASE_CAMPAIGN}" ]; then
  echo "missing reusable 12K campaign appliance" >&2
  exit 2
fi
if [ -z "${MATRIX}" ]; then
  echo "LEHOME_SUCCESS_REPLAY_MATRIX is required" >&2
  exit 2
fi
if [ -z "${EXPECTED_SHA256}" ]; then
  echo "LEHOME_SUCCESS_REPLAY_MATRIX_SHA256 is required" >&2
  exit 2
fi
case "${MATRIX}" in
  /*) ;;
  *) echo "LEHOME_SUCCESS_REPLAY_MATRIX must be an absolute path" >&2; exit 2 ;;
esac
if [ -L "${MATRIX}" ] || [ ! -f "${MATRIX}" ] || [ -L "${MATRIX}.sha256" ] || [ ! -f "${MATRIX}.sha256" ]; then
  echo "success replay matrix and receipt must be regular files" >&2
  exit 2
fi
if ! [[ "${EXPECTED_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "LEHOME_SUCCESS_REPLAY_MATRIX_SHA256 must be a lowercase 64-character SHA-256" >&2
  exit 2
fi
if ! [[ "${TARGET_ACCEPTED}" =~ ^([1-9]|[1-9][0-9]|1[0-4][0-9]|150)$ ]]; then
  echo "LEHOME_TARGET_ACCEPTED must be an integer in 1..150" >&2
  exit 2
fi
ACTUAL_SHA256="$(sha256sum "${MATRIX}" | awk '{print $1}')"
RECEIPT_SHA256="$(tr -d '\r\n' < "${MATRIX}.sha256")"
if [ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ] || [ "${RECEIPT_SHA256}" != "${EXPECTED_SHA256}" ]; then
  echo "success replay matrix SHA-256 mismatch" >&2
  exit 2
fi
python3 - "${MATRIX}" <<'PY'
import json
import re
import sys
from collections import Counter
from pathlib import Path

matrix = Path(sys.argv[1])
try:
    rows = json.loads(matrix.read_text(encoding="utf-8"))
except (OSError, ValueError) as error:
    raise SystemExit(f"success replay matrix is malformed: {error}")
categories = ("top_long", "top_short", "pant_long", "pant_short")
if not isinstance(rows, list) or len(rows) != 200:
    raise SystemExit("success replay matrix must contain exactly 200 attempts")
if Counter(row.get("category") for row in rows if isinstance(row, dict)) != Counter({key: 50 for key in categories}):
    raise SystemExit("success replay matrix must contain 50 attempts per category")
for row in rows:
    if not isinstance(row, dict):
        raise SystemExit("success replay matrix row is malformed")
    if row.get("replay_kind") != "verified_success_reset_v1":
        raise SystemExit("success replay matrix row has an invalid replay kind")
    if not isinstance(row.get("restore_snapshot"), str) or not row["restore_snapshot"].startswith("/"):
        raise SystemExit("success replay matrix row has an unsafe restore snapshot")
    if not isinstance(row.get("restore_snapshot_sha256"), str) or re.fullmatch(r"[0-9a-f]{64}", row["restore_snapshot_sha256"]) is None:
        raise SystemExit("success replay matrix row has an invalid restore digest")
    if row.get("restore_snapshot_cloth_frame") not in {"usd_local_points_v1", "physx_cloth_view_world_v1"}:
        raise SystemExit("success replay matrix row has an invalid cloth frame")
    if not isinstance(row.get("parent_episode_id"), str) or not row["parent_episode_id"] or row.get("lineage_id") != row["parent_episode_id"]:
        raise SystemExit("success replay matrix row has inconsistent lineage")
PY

exec env \
  LEHOME_POLICY_SHA256="${ORIGINAL_12K_POLICY_SHA256}" \
  LEHOME_CHECKPOINT_DIR="${ORIGINAL_12K_CHECKPOINT}" \
  LEHOME_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" \
  LEHOME_ATTEMPT_MATRIX="${MATRIX}" \
  LEHOME_ATTEMPT_MATRIX_SHA256="${EXPECTED_SHA256}" \
  LEHOME_WORKER_COUNT="${WORKER_COUNT}" \
  LEHOME_MAX_ATTEMPTS="${MAX_ATTEMPTS}" \
  LEHOME_MAX_WORKER_RESTARTS="${MAX_WORKER_RESTARTS}" \
  LEHOME_TARGET_ACCEPTED="${TARGET_ACCEPTED}" \
  LEHOME_ROUND_ID="${ROUND_ID}" \
  bash "${BASE_CAMPAIGN}"
