#!/usr/bin/env bash
set -Eeuo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUTPUT_ROOT=${OUTPUT_ROOT:-/workspace/outputs}
LOG_DIR=${LOG_DIR:-/workspace/logs/b1k}
CYCLE_ID=${CYCLE_ID:-cycle-000}
R2_REMOTE=${R2_REMOTE:-}
R2_BUCKET=${R2_BUCKET:-}
POLL_SECONDS=${CHECKPOINT_POLL_SECONDS:-300}
STATE_FILE="${LOG_DIR}/uploaded-checkpoints.txt"

if [[ -z "${R2_REMOTE}" || -z "${R2_BUCKET}" ]]; then
  echo "R2_REMOTE and R2_BUCKET are required for checkpoint watching." >&2
  exit 1
fi

if (( DRY_RUN )); then
  echo "Watch ${OUTPUT_ROOT}/**/checkpoint-* every ${POLL_SECONDS}s"
  echo "Wait until no file is newer than two minutes (-mmin -2)"
  echo "Call ${SCRIPT_DIR}/push_artifacts.sh and record ${STATE_FILE}"
  exit 0
fi

mkdir -p "${LOG_DIR}"
touch "${STATE_FILE}"

while true; do
  while IFS= read -r -d '' CHECKPOINT_PATH; do
    if grep -Fqx "${CHECKPOINT_PATH}" "${STATE_FILE}"; then
      continue
    fi
    if [[ -z "$(find "${CHECKPOINT_PATH}" -type f -print -quit)" ]]; then
      continue
    fi
    if find "${CHECKPOINT_PATH}" -type f -mmin -2 -print -quit | grep -q .; then
      continue
    fi
    EXPERIMENT_NAME=$(basename "$(dirname "${CHECKPOINT_PATH}")")
    "${SCRIPT_DIR}/push_artifacts.sh" \
      --checkpoint "${CHECKPOINT_PATH}" \
      --cycle "${CYCLE_ID}/${EXPERIMENT_NAME}"
    printf '%s\n' "${CHECKPOINT_PATH}" >> "${STATE_FILE}"
  done < <(find "${OUTPUT_ROOT}" -type d -name 'checkpoint-*' -print0 | sort -z)

  if [[ "${CHECKPOINT_WATCH_ONCE:-0}" == "1" ]]; then
    exit 0
  fi
  sleep "${POLL_SECONDS}"
done
