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
python3 - "${MATRIX}" "${MAX_ATTEMPTS}" "${TARGET_ACCEPTED}" <<'PY'
import json
import re
import sys
from collections import Counter
from pathlib import Path

matrix, max_attempts, target_accepted = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
try:
    rows = json.loads(matrix.read_text(encoding="utf-8"))
except (OSError, ValueError) as error:
    raise SystemExit(f"success replay matrix is malformed: {error}")
categories = ("top_long", "top_short", "pant_long", "pant_short")
if not isinstance(rows, list) or not 1 <= len(rows) <= 400 or len(rows) != max_attempts:
    raise SystemExit("success replay matrix must match LEHOME_MAX_ATTEMPTS in 1..400")
counts = Counter(row.get("category") for row in rows if isinstance(row, dict))
if any(category not in categories for category in counts):
    raise SystemExit("success replay matrix contains an invalid category")
has_caps = ["category_acceptance_cap" in row for row in rows if isinstance(row, dict)]
if any(has_caps) and not all(has_caps):
    raise SystemExit("success replay matrix category caps must be present on every row")
cap_mode = all(has_caps)
if not cap_mode and (len(rows) != 200 or counts != Counter({key: 50 for key in categories})):
    raise SystemExit("legacy success replay matrix must contain 50 attempts per category")
caps = {}
for row in rows:
    if not isinstance(row, dict):
        raise SystemExit("success replay matrix row is malformed")
    replay_kind = row.get("replay_kind")
    if replay_kind not in {"verified_success_reset_v1", "verified_success_early_snapshot_v1"}:
        raise SystemExit("success replay matrix row has an invalid replay kind")
    if (
        replay_kind == "verified_success_early_snapshot_v1"
        and row.get("restore_snapshot_step") != 16
    ) or (
        replay_kind == "verified_success_reset_v1"
        and "restore_snapshot_step" in row
    ):
        raise SystemExit("success replay matrix row has an invalid restore boundary")
    if not isinstance(row.get("restore_snapshot"), str) or not row["restore_snapshot"].startswith("/"):
        raise SystemExit("success replay matrix row has an unsafe restore snapshot")
    if not isinstance(row.get("restore_snapshot_sha256"), str) or re.fullmatch(r"[0-9a-f]{64}", row["restore_snapshot_sha256"]) is None:
        raise SystemExit("success replay matrix row has an invalid restore digest")
    if row.get("restore_snapshot_cloth_frame") not in {"usd_local_points_v1", "physx_cloth_view_world_v1"}:
        raise SystemExit("success replay matrix row has an invalid cloth frame")
    if not isinstance(row.get("parent_episode_id"), str) or not row["parent_episode_id"] or row.get("lineage_id") != row["parent_episode_id"]:
        raise SystemExit("success replay matrix row has inconsistent lineage")
    if cap_mode:
        cap = row.get("category_acceptance_cap")
        category = row.get("category")
        if type(cap) is not int or not 0 <= cap <= 150 or (category in caps and caps[category] != cap):
            raise SystemExit("success replay matrix category cap is invalid")
        caps[category] = cap
if cap_mode and (any(caps[category] > counts[category] for category in caps) or sum(caps.values()) != target_accepted):
    raise SystemExit("success replay matrix caps do not match LEHOME_TARGET_ACCEPTED")
PY

exec env \
  LEHOME_POLICY_SHA256="${ORIGINAL_12K_POLICY_SHA256}" \
  LEHOME_CHECKPOINT_DIR="${ORIGINAL_12K_CHECKPOINT}" \
  LEHOME_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" \
  LEHOME_ATTEMPT_MATRIX="${MATRIX}" \
  LEHOME_ATTEMPT_MATRIX_SHA256="${EXPECTED_SHA256}" \
  LEHOME_WORKER_COUNT="${WORKER_COUNT}" \
  LEHOME_SIMULATOR_DEVICE="cpu" \
  LEHOME_SUCCESS_REPLAY_CAMPAIGN="1" \
  LEHOME_MAX_ATTEMPTS="${MAX_ATTEMPTS}" \
  LEHOME_MAX_WORKER_RESTARTS="${MAX_WORKER_RESTARTS}" \
  LEHOME_TARGET_ACCEPTED="${TARGET_ACCEPTED}" \
  LEHOME_ROUND_ID="${ROUND_ID}" \
  bash "${BASE_CAMPAIGN}"
