#!/usr/bin/env bash
set -Eeuo pipefail

CYCLE_ID=""
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cycle) CYCLE_ID=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Usage: $0 --cycle cycle-NNN [--dry-run]" >&2; exit 2 ;;
  esac
done

R2_REMOTE=${R2_REMOTE:-}
R2_BUCKET=${R2_BUCKET:-}
B1K_DATA_ROOT=${B1K_DATA_ROOT:-/workspace/datasets/2026-challenge-demos}
RESTORE_ROOT=${RESTORE_ROOT:-/workspace/restore}
LOG_DIR=${LOG_DIR:-/workspace/logs/b1k}
RESTORED_CHECKPOINT_FILE=${RESTORED_CHECKPOINT_FILE:-${LOG_DIR}/restored-checkpoint.txt}

if [[ -z "${CYCLE_ID}" || -z "${R2_REMOTE}" || -z "${R2_BUCKET}" ]]; then
  echo "--cycle, R2_REMOTE, and R2_BUCKET are required." >&2
  exit 1
fi

SOURCE="${R2_REMOTE}:${R2_BUCKET}/runs/${CYCLE_ID}"
DESTINATION="${RESTORE_ROOT}/${CYCLE_ID}"

if (( DRY_RUN )); then
  echo "rclone copy ${SOURCE} ${DESTINATION}"
  echo "rclone check ${SOURCE} ${DESTINATION} --one-way"
  echo "cd ${DESTINATION} and sha256sum --check SHA256SUMS"
  echo "tar --zstd -xf checkpoint archive"
  echo "restore metadata/meta/stats.json to ${B1K_DATA_ROOT}/meta/stats.json"
  echo "restore metadata/meta/modality.json to ${B1K_DATA_ROOT}/meta/modality.json"
  echo "write checkpoint path to ${RESTORED_CHECKPOINT_FILE}"
  exit 0
fi

mkdir -p "${DESTINATION}" "${LOG_DIR}" "${B1K_DATA_ROOT}/meta"
rclone copy "${SOURCE}" "${DESTINATION}" --transfers 16 --checkers 32 --progress
rclone check "${SOURCE}" "${DESTINATION}" --one-way
(
  cd "${DESTINATION}"
  sha256sum --check SHA256SUMS
)

if ! grep -qx 'status=success' "${DESTINATION}/run-manifest.txt"; then
  echo "Parent cycle ${CYCLE_ID} was not a successful training run." >&2
  exit 1
fi
mapfile -t CHECKPOINT_ARCHIVES < <(find "${DESTINATION}" -maxdepth 1 -type f -name 'checkpoint-*.tar.zst' -print)
if (( ${#CHECKPOINT_ARCHIVES[@]} != 1 )); then
  echo "Expected exactly one checkpoint archive for ${CYCLE_ID}; found ${#CHECKPOINT_ARCHIVES[@]}." >&2
  exit 1
fi

mkdir -p "${DESTINATION}/checkpoint"
tar --zstd -xf "${CHECKPOINT_ARCHIVES[0]}" -C "${DESTINATION}/checkpoint"
mapfile -t CHECKPOINT_DIRS < <(find "${DESTINATION}/checkpoint" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -print)
if (( ${#CHECKPOINT_DIRS[@]} != 1 )); then
  echo "Expected exactly one restored checkpoint directory; found ${#CHECKPOINT_DIRS[@]}." >&2
  exit 1
fi

cp "${DESTINATION}/metadata/meta/stats.json" "${B1K_DATA_ROOT}/meta/stats.json"
cp "${DESTINATION}/metadata/meta/modality.json" "${B1K_DATA_ROOT}/meta/modality.json"
printf '%s\n' "${CHECKPOINT_DIRS[0]}" > "${RESTORED_CHECKPOINT_FILE}"
echo "Restored verified parent cycle ${CYCLE_ID}: ${CHECKPOINT_DIRS[0]}"
