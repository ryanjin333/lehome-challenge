#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

CYCLE_ID=${CYCLE_ID:-cycle-000}
LOG_DIR=${LOG_DIR:-/workspace/logs/b1k}
OUTPUT_ROOT=${OUTPUT_ROOT:-/workspace/outputs}
FINAL_CHECKPOINT_FILE=${FINAL_CHECKPOINT_FILE:-${LOG_DIR}/final-checkpoint.txt}
PARENT_CYCLE_ID=${PARENT_CYCLE_ID:-}
RESTORED_CHECKPOINT_FILE=${RESTORED_CHECKPOINT_FILE:-${LOG_DIR}/restored-checkpoint.txt}
export CYCLE_ID LOG_DIR FINAL_CHECKPOINT_FILE RESTORED_CHECKPOINT_FILE

if [[ -n "${PARENT_CYCLE_ID}" && ( -z "${R2_REMOTE:-}" || -z "${R2_BUCKET:-}" ) ]]; then
  echo "R2_REMOTE and R2_BUCKET are currently required only when restoring a parent cycle." >&2
  exit 1
fi

if (( DRY_RUN )); then
  echo "${SCRIPT_DIR}/bootstrap_training.sh --prepare-only"
  if [[ -n "${PARENT_CYCLE_ID}" ]]; then
    echo "${SCRIPT_DIR}/pull_run_bundle.sh --cycle ${PARENT_CYCLE_ID}"
    echo "BASE_MODEL_PATH=<restored-checkpoint>"
  fi
  echo "${SCRIPT_DIR}/start_training.sh"
  echo "${SCRIPT_DIR}/push_run_bundle.sh --checkpoint <final-checkpoint> --cycle ${CYCLE_ID} --status success"
  echo "${SCRIPT_DIR}/destroy_instance.sh"
  exit 0
fi

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/disposable-run.log") 2>&1

RUN_STATUS=success
TRAIN_STATUS=0
"${SCRIPT_DIR}/bootstrap_training.sh" --prepare-only || TRAIN_STATUS=$?
if (( TRAIN_STATUS == 0 )); then
  if [[ -n "${PARENT_CYCLE_ID}" ]]; then
    "${SCRIPT_DIR}/pull_run_bundle.sh" --cycle "${PARENT_CYCLE_ID}" || TRAIN_STATUS=$?
    if (( TRAIN_STATUS == 0 )); then
      if [[ ! -s "${RESTORED_CHECKPOINT_FILE}" ]]; then
        echo "Parent cycle restored without a checkpoint marker." >&2
        TRAIN_STATUS=1
      else
        BASE_MODEL_PATH=$(<"${RESTORED_CHECKPOINT_FILE}")
        export BASE_MODEL_PATH
      fi
    fi
  fi
fi
if (( TRAIN_STATUS == 0 )); then
  "${SCRIPT_DIR}/start_training.sh" || TRAIN_STATUS=$?
fi

CHECKPOINT_ARGS=()
if (( TRAIN_STATUS == 0 )); then
  if [[ ! -s "${FINAL_CHECKPOINT_FILE}" ]]; then
    echo "Training completed without a final checkpoint marker." >&2
    exit 1
  fi
  CHECKPOINT_ARGS=(--checkpoint "$(<"${FINAL_CHECKPOINT_FILE}")")
else
  RUN_STATUS=failed
  if [[ -d "${OUTPUT_ROOT}" ]]; then
    LATEST_RECOVERABLE_CHECKPOINT=$(find "${OUTPUT_ROOT}" -type d -name 'checkpoint-*' -print \
      | sort -V | tail -n 1)
    if [[ -n "${LATEST_RECOVERABLE_CHECKPOINT}" ]]; then
      echo "Including latest recoverable checkpoint in failed run bundle: ${LATEST_RECOVERABLE_CHECKPOINT}"
      CHECKPOINT_ARGS=(--checkpoint "${LATEST_RECOVERABLE_CHECKPOINT}")
    fi
  fi
fi

"${SCRIPT_DIR}/push_run_bundle.sh" \
  "${CHECKPOINT_ARGS[@]}" --cycle "${CYCLE_ID}" --status "${RUN_STATUS}"
"${SCRIPT_DIR}/destroy_instance.sh"
exit "${TRAIN_STATUS}"
