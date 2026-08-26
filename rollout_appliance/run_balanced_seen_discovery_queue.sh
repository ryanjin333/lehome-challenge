#!/usr/bin/env bash
# Run the fixed follow-up shards for the balanced 1,000-attempt seen pool.
# The already-running pant-long shard is deliberately external to this queue.
set -uo pipefail

WORKSPACE="${LEHOME_WORKSPACE:-/mnt/lehome}"
QUEUE_FILE="${LEHOME_BALANCED_SEEN_QUEUE_FILE:?LEHOME_BALANCED_SEEN_QUEUE_FILE is required}"
PREDECESSOR_UNIT="${LEHOME_BALANCED_SEEN_PREDECESSOR_UNIT:?LEHOME_BALANCED_SEEN_PREDECESSOR_UNIT is required}"
ROLLOUT_IMAGE="${LEHOME_ROLLOUT_IMAGE:-lehome-rollout:build}"
HF_TOKEN_FILE="${LEHOME_HF_TOKEN_FILE:-${WORKSPACE}/secrets/hf_token}"
BOOTSTRAP="${LEHOME_SNAPSHOT_SOURCE_BOOTSTRAP:-/opt/lehome/rollout_appliance/run_snapshot_source_bootstrap.sh}"
STATUS_LOG="${LEHOME_BALANCED_SEEN_STATUS_LOG:-${QUEUE_FILE}.status.tsv}"

if [ ! -f "${QUEUE_FILE}" ] || [ -L "${QUEUE_FILE}" ]; then
  echo "balanced seen queue file is missing or unsafe" >&2
  exit 2
fi

while systemctl is-active --quiet "${PREDECESSOR_UNIT}"; do
  sleep 30
done

predecessor_status="$(systemctl show "${PREDECESSOR_UNIT}" -p ExecMainStatus --value 2>/dev/null || true)"
case "${predecessor_status}" in
  0|3) ;;
  *)
    echo "balanced seen predecessor failed an infrastructure gate: ${predecessor_status:-unknown}" >&2
    exit 4
    ;;
esac

overall=0
while IFS=$'\t' read -r category descriptor descriptor_sha run_id target; do
  [ -n "${category}" ] || continue
  if ! [[ "${category}" =~ ^(pant_short|top_long|top_short)$ ]] \
      || ! [[ "${descriptor}" == "${WORKSPACE}/operator/balanced1000-discovery-v2/"*.json ]] \
      || ! [[ "${descriptor_sha}" =~ ^[0-9a-f]{64}$ ]] \
      || ! [[ "${run_id}" =~ ^[0-9a-f]{32}$ ]] \
      || [ "${target}" != "125" ]; then
    echo "balanced seen queue row is invalid" >&2
    exit 2
  fi

  started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  set +e
  env \
    LEHOME_WORKSPACE="${WORKSPACE}" \
    LEHOME_SNAPSHOT_SOURCE_DESCRIPTOR="${descriptor}" \
    LEHOME_SNAPSHOT_SOURCE_DESCRIPTOR_SHA256="${descriptor_sha}" \
    LEHOME_SNAPSHOT_SOURCE_RUN_ID="${run_id}" \
    LEHOME_SNAPSHOT_SOURCE_TARGET_ACCEPTED="${target}" \
    LEHOME_SNAPSHOT_SOURCE_WORKER_COUNT=4 \
    LEHOME_SNAPSHOT_SOURCE_SIMULATOR_DEVICE=cpu \
    LEHOME_HF_TOKEN_FILE="${HF_TOKEN_FILE}" \
    LEHOME_ROLLOUT_IMAGE="${ROLLOUT_IMAGE}" \
    LEHOME_SOURCE_FINALIZATION_TIMEOUT_SECONDS=900 \
    bash "${BOOTSTRAP}"
  status=$?
  set -e
  finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s\t%s\t%s\t%s\n' "${category}" "${started_at}" "${finished_at}" "${status}" >> "${STATUS_LOG}"

  case "${status}" in
    0) ;;
    3)
      # A zero-success shard is policy evidence, not an infrastructure failure.
      overall=3
      ;;
    *)
      echo "balanced seen shard failed an infrastructure gate: ${category} status=${status}" >&2
      exit 4
      ;;
  esac
done < "${QUEUE_FILE}"

exit "${overall}"
