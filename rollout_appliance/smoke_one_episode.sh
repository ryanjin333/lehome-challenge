#!/usr/bin/env bash
# Provenance-safe one-episode gate before a four-worker unseen-80 evaluation.
set -euo pipefail

WORKSPACE="${LEHOME_WORKSPACE:-/mnt/lehome}"
RUNNER="/opt/lehome/rollout_appliance/run_12k_campaign.sh"
MATRIX="/opt/lehome/rollout_appliance/eval_unseen80_smoke_v1.json"
MATRIX_SHA_FILE="/opt/lehome/rollout_appliance/eval_unseen80_smoke_v1.json.sha256"

for name in LEHOME_POLICY_SHA256 LEHOME_POLICY_REVISION LEHOME_POLICY_STEP \
            LEHOME_POLICY_ARTIFACT_SHA256 LEHOME_CHECKPOINT_DIR; do
  if [ -z "${!name:-}" ]; then
    echo "${name} is required for an attributable rollout smoke" >&2
    exit 2
  fi
done
if [ ! -x "${RUNNER}" ] || [ ! -f "${MATRIX}" ] || [ ! -f "${MATRIX_SHA_FILE}" ]; then
  echo "the baked rollout smoke recipe is incomplete" >&2
  exit 2
fi
MATRIX_SHA256="$(tr -d '[:space:]' < "${MATRIX_SHA_FILE}")"
if ! [[ "${MATRIX_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "the baked rollout smoke matrix digest is invalid" >&2
  exit 2
fi

SMOKE_ID="$(uuidgen | tr '[:upper:]' '[:lower:]')"
CAMPAIGN_ROOT="${LEHOME_CAMPAIGN_ROOT:-${WORKSPACE}/eval/smoke-${LEHOME_POLICY_STEP}-${SMOKE_ID}}"

exec env \
  LEHOME_WORKSPACE="${WORKSPACE}" \
  LEHOME_POLICY_SHA256="${LEHOME_POLICY_SHA256}" \
  LEHOME_POLICY_REPO="${LEHOME_POLICY_REPO:-ryanjin333/lehome-groot-n17-models}" \
  LEHOME_POLICY_REVISION="${LEHOME_POLICY_REVISION}" \
  LEHOME_POLICY_STEP="${LEHOME_POLICY_STEP}" \
  LEHOME_POLICY_ARTIFACT_SHA256="${LEHOME_POLICY_ARTIFACT_SHA256}" \
  LEHOME_CHECKPOINT_DIR="${LEHOME_CHECKPOINT_DIR}" \
  LEHOME_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" \
  LEHOME_ATTEMPT_MATRIX="${MATRIX}" \
  LEHOME_ATTEMPT_MATRIX_SHA256="${MATRIX_SHA256}" \
  LEHOME_WORKER_COUNT=1 \
  LEHOME_MAX_ATTEMPTS=1 \
  LEHOME_TARGET_ACCEPTED=1 \
  LEHOME_MAX_STEPS="${LEHOME_MAX_STEPS:-600}" \
  LEHOME_INITIAL_GARMENT=Top_Long_Unseen_0 \
  LEHOME_PREPARATION_TIMEOUT_SECONDS="${LEHOME_PREPARATION_TIMEOUT_SECONDS:-240}" \
  LEHOME_MAX_WORKER_RESTARTS=1 \
  LEHOME_ENABLE_HF_UPLOAD=0 \
  "${RUNNER}"
