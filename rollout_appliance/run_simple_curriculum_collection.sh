#!/usr/bin/env bash
# Fail-closed launcher for the reviewed, one-VM simple-curriculum journal.
# It intentionally delegates only to the local host orchestrator; provider
# lifecycle actions belong to an explicitly supplied stop hook outside here.
set -euo pipefail

PAID_COLLECTION="${LEHOME_PAID_COLLECTION:-0}"
case "${PAID_COLLECTION}" in 0|1) ;; *) echo "LEHOME_PAID_COLLECTION must be 0 or 1" >&2; exit 2 ;; esac
if [ "${PAID_COLLECTION}" = "1" ] && [ -z "${LEHOME_GPU_STOP_COMMAND:-}" ]; then
  echo "paid collection requires LEHOME_GPU_STOP_COMMAND" >&2
  exit 2
fi

HOST_ROOT="${LEHOME_HOST_CODE_ROOT:-}"
if [ -z "${HOST_ROOT}" ] || [[ "${HOST_ROOT}" != /* ]] || [ -L "${HOST_ROOT}" ] || [ ! -d "${HOST_ROOT}" ]; then
  echo "LEHOME_HOST_CODE_ROOT must be an absolute regular non-symlink checked-out root" >&2
  exit 2
fi
for path in source/lehome trainer/src scripts rollout_appliance; do
  if [ -L "${HOST_ROOT}/${path}" ] || [ ! -d "${HOST_ROOT}/${path}" ]; then
    echo "LEHOME_HOST_CODE_ROOT is missing reviewed ${path}" >&2
    exit 2
  fi
done

for required in LEHOME_CAMPAIGN_ROOT LEHOME_RUN_ID LEHOME_ROUND_ID LEHOME_MAX_WALL_SECONDS LEHOME_MAX_SPEND_USD LEHOME_RUNTIME_IDENTITY_JSON; do
  if [ -z "${!required:-}" ]; then
    echo "${required} is required" >&2
    exit 2
  fi
done
if [[ "${LEHOME_CAMPAIGN_ROOT}" != /* ]] || [[ "${LEHOME_RUNTIME_IDENTITY_JSON}" != /* ]]; then
  echo "campaign root and runtime identity must be absolute paths" >&2
  exit 2
fi

arguments=(
  --campaign-root "${LEHOME_CAMPAIGN_ROOT}"
  --host-code-root "${HOST_ROOT}"
  --run-id "${LEHOME_RUN_ID}"
  --round-id "${LEHOME_ROUND_ID}"
  --max-wall-seconds "${LEHOME_MAX_WALL_SECONDS}"
  --max-spend-usd "${LEHOME_MAX_SPEND_USD}"
  --runtime-identity-json "${LEHOME_RUNTIME_IDENTITY_JSON}"
)
if [ "${PAID_COLLECTION}" = "1" ]; then
  arguments+=(--paid --gpu-stop-command "${LEHOME_GPU_STOP_COMMAND}")
fi

exec python3 "${HOST_ROOT}/scripts/run_simple_curriculum_collection.py" "${arguments[@]}"
