#!/usr/bin/env bash
set -Eeuo pipefail

CHECKPOINT_PATH=""
CYCLE_ID=${CYCLE_ID:-}
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT_PATH=$2; shift 2 ;;
    --cycle) CYCLE_ID=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Usage: $0 --checkpoint PATH --cycle cycle-NNN [--dry-run]" >&2; exit 2 ;;
  esac
done

R2_REMOTE=${R2_REMOTE:-}
R2_BUCKET=${R2_BUCKET:-}
HANDOFF_ROOT=${HANDOFF_ROOT:-/workspace/handoff}

if [[ -z "${CHECKPOINT_PATH}" || -z "${CYCLE_ID}" ]]; then
  echo "--checkpoint and --cycle are required." >&2
  exit 2
fi
if [[ -z "${R2_REMOTE}" || -z "${R2_BUCKET}" ]]; then
  echo "R2_REMOTE and R2_BUCKET must name a configured rclone remote and bucket." >&2
  exit 1
fi

DESTINATION="${R2_REMOTE}:${R2_BUCKET}/checkpoints/${CYCLE_ID}"
ARCHIVE_NAME="$(basename "${CHECKPOINT_PATH}").tar.zst"
STAGE_DIR="${HANDOFF_ROOT}/${CYCLE_ID}"

if (( DRY_RUN )); then
  echo "tar --zstd -cf ${STAGE_DIR}/${ARCHIVE_NAME} -C $(dirname "${CHECKPOINT_PATH}") $(basename "${CHECKPOINT_PATH}")"
  echo "sha256sum ${ARCHIVE_NAME} > ${STAGE_DIR}/SHA256SUMS"
  echo "rclone copy ${STAGE_DIR} ${DESTINATION} --s3-upload-concurrency 8"
  echo "rclone check ${STAGE_DIR} ${DESTINATION} --one-way"
  exit 0
fi

if [[ ! -d "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint directory not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi
mkdir -p "${STAGE_DIR}"
tar --zstd -cf "${STAGE_DIR}/${ARCHIVE_NAME}" \
  -C "$(dirname "${CHECKPOINT_PATH}")" "$(basename "${CHECKPOINT_PATH}")"
(
  cd "${STAGE_DIR}"
  sha256sum "${ARCHIVE_NAME}" > SHA256SUMS
)
rclone copy "${STAGE_DIR}" "${DESTINATION}" \
  --transfers 16 --checkers 32 --s3-upload-concurrency 8 --s3-chunk-size 64M \
  --progress
rclone check "${STAGE_DIR}" "${DESTINATION}" --one-way
echo "Verified checkpoint handoff: ${DESTINATION}/${ARCHIVE_NAME}"
