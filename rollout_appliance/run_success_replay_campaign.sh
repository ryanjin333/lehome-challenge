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
ORIGINAL_12K_POLICY_REPO="ryanjin333/lehome-groot-n17-models"
ORIGINAL_12K_POLICY_REVISION="30ac1a84da67b099e115ad147bcd61e9d60046d3"
ORIGINAL_12K_POLICY_ARTIFACT_SHA256="3fadfea79b662a8b8e10fe3cae284c6a49d66a9855ed540d6e4d97d66a0f9f06"
ORIGINAL_12K_CHECKPOINT="${LEHOME_ORIGINAL_12K_CHECKPOINT:-${WORKSPACE}/eval/policies/original_baseline}"
FRESH_SOURCE_REPORTS="${LEHOME_FRESH_SOURCE_REPORTS_JSON:-}"
FRESH_SOURCE_MATRICES="${LEHOME_FRESH_SOURCE_MATRICES_JSON:-}"

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
if ! [[ "${TARGET_ACCEPTED}" =~ ^([1-9]|[1-9][0-9]|1[0-9][0-9]|200)$ ]]; then
  echo "LEHOME_TARGET_ACCEPTED must be an integer in 1..200" >&2
  exit 2
fi
if (( 10#${TARGET_ACCEPTED} > 150 )) && [ "${TARGET_ACCEPTED}" != "200" ]; then
  echo "LEHOME_TARGET_ACCEPTED must be in 1..150 unless using the exact fresh visual-only tuple" >&2
  exit 2
fi
ACTUAL_SHA256="$(sha256sum "${MATRIX}" | awk '{print $1}')"
RECEIPT_SHA256="$(tr -d '\r\n' < "${MATRIX}.sha256")"
if [ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ] || [ "${RECEIPT_SHA256}" != "${EXPECTED_SHA256}" ]; then
  echo "success replay matrix SHA-256 mismatch" >&2
  exit 2
fi
python3 - "${MATRIX}" "${MAX_ATTEMPTS}" "${TARGET_ACCEPTED}" "${FRESH_SOURCE_REPORTS}" "${FRESH_SOURCE_MATRICES}" <<'PY'
import json
import re
import sys
from collections import Counter
from pathlib import Path

matrix, max_attempts, target_accepted = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
report_evidence, matrix_evidence = sys.argv[4], sys.argv[5]
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
fresh_fields = {
    "source_episode_sha256", "source_episode_root", "source_episode_path", "source_reset_sha256", "source_annotations_sha256",
    "source_continuation_snapshot_sha256", "source_state_fingerprint",
    "source_report_sha256", "source_matrix_sha256", "source_receipt_sha256", "source_receipt_path",
    "source_remote_prefix", "source_immutable_revision", "source_round_id", "source_run_id",
    "source_report_path", "source_matrix_path",
}
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
if target_accepted == 200:
    if (
        len(rows) != 400
        or counts != Counter({key: 100 for key in categories})
        or caps != {key: 50 for key in categories}
        or any(row.get("strategy") != "visual_only" for row in rows)
    ):
        raise SystemExit("200 accepted requires the exact fresh visual-only matrix tuple")
    for row in rows:
        if not fresh_fields <= set(row):
            raise SystemExit("200 accepted requires bound fresh-source provenance")
        if any(not isinstance(row[field], str) or re.fullmatch(r"[0-9a-f]{64}", row[field]) is None for field in fresh_fields - {"source_episode_root", "source_episode_path", "source_report_path", "source_matrix_path", "source_receipt_path", "source_remote_prefix", "source_immutable_revision", "source_round_id", "source_run_id"}):
            raise SystemExit("200 accepted fresh-source hashes are invalid")
        if (
            not isinstance(row["source_round_id"], str)
            or re.fullmatch(r"fresh-12k-[a-z0-9-]{1,112}", row["source_round_id"]) is None
            or row["source_remote_prefix"] != f"rollout-rounds/{row['source_round_id']}/{row['parent_episode_id']}"
            or re.fullmatch(r"[0-9a-f]{40}", str(row["source_immutable_revision"])) is None
        ):
            raise SystemExit("200 accepted fresh-source receipt binding is invalid")
elif sum(caps.values()) > 150:
    raise SystemExit("legacy success replay acceptance caps must remain at most 150")
PY

if [ "${TARGET_ACCEPTED}" = "200" ]; then
  python3 "${SCRIPT_DIR}/validate_fresh_replay_evidence.py" \
    --matrix "${MATRIX}" --max-attempts "${MAX_ATTEMPTS}" \
    --source-reports-json "${FRESH_SOURCE_REPORTS}" \
    --source-matrices-json "${FRESH_SOURCE_MATRICES}"
fi

exec env \
  LEHOME_POLICY_SHA256="${ORIGINAL_12K_POLICY_SHA256}" \
  LEHOME_POLICY_REPO="${ORIGINAL_12K_POLICY_REPO}" \
  LEHOME_POLICY_REVISION="${ORIGINAL_12K_POLICY_REVISION}" \
  LEHOME_POLICY_STEP="12000" \
  LEHOME_POLICY_ARTIFACT_SHA256="${ORIGINAL_12K_POLICY_ARTIFACT_SHA256}" \
  LEHOME_CHECKPOINT_DIR="${ORIGINAL_12K_CHECKPOINT}" \
  LEHOME_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" \
  LEHOME_ATTEMPT_MATRIX="${MATRIX}" \
  LEHOME_ATTEMPT_MATRIX_SHA256="${EXPECTED_SHA256}" \
  LEHOME_WORKER_COUNT="${WORKER_COUNT}" \
  LEHOME_SIMULATOR_DEVICE="cpu" \
  LEHOME_SUCCESS_REPLAY_CAMPAIGN="1" \
  LEHOME_FRESH_SOURCE_REPORTS_JSON="${FRESH_SOURCE_REPORTS}" \
  LEHOME_FRESH_SOURCE_MATRICES_JSON="${FRESH_SOURCE_MATRICES}" \
  LEHOME_MAX_ATTEMPTS="${MAX_ATTEMPTS}" \
  LEHOME_MAX_WORKER_RESTARTS="${MAX_WORKER_RESTARTS}" \
  LEHOME_TARGET_ACCEPTED="${TARGET_ACCEPTED}" \
  LEHOME_ROUND_ID="${ROUND_ID}" \
  bash "${BASE_CAMPAIGN}"
