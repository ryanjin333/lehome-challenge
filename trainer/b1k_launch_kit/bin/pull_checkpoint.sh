#!/usr/bin/env bash
set -Eeuo pipefail

CYCLE_ID=${CYCLE_ID:-}
DESTINATION=${CHECKPOINT_DESTINATION:-/workspace/checkpoints/incoming}
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cycle) CYCLE_ID=$2; shift 2 ;;
    --destination) DESTINATION=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Usage: $0 --cycle cycle-NNN [--destination PATH] [--dry-run]" >&2; exit 2 ;;
  esac
done

R2_REMOTE=${R2_REMOTE:-}
R2_BUCKET=${R2_BUCKET:-}
if [[ -z "${CYCLE_ID}" ]]; then
  echo "--cycle is required." >&2
  exit 2
fi
if [[ -z "${R2_REMOTE}" || -z "${R2_BUCKET}" ]]; then
  echo "R2_REMOTE and R2_BUCKET must be configured." >&2
  exit 1
fi
SOURCE="${R2_REMOTE}:${R2_BUCKET}/checkpoints/${CYCLE_ID}"

if (( DRY_RUN )); then
  echo "rclone copy ${SOURCE} ${DESTINATION}"
  echo "cd ${DESTINATION} && sha256sum --check SHA256SUMS"
  exit 0
fi

mkdir -p "${DESTINATION}"
rclone copy "${SOURCE}" "${DESTINATION}" --transfers 16 --checkers 32 --progress
(
  cd "${DESTINATION}"
  sha256sum --check SHA256SUMS
)
echo "Checkpoint downloaded and verified in ${DESTINATION}."
