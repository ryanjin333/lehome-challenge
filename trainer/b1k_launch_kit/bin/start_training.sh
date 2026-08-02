#!/usr/bin/env bash
set -Eeuo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

if [[ -z "${GROOT_DIR:-}" ]]; then
  if [[ -d /opt/isaac-groot/.git ]]; then
    GROOT_DIR=/opt/isaac-groot
  else
    GROOT_DIR=/workspace/Isaac-GR00T
  fi
fi
if [[ -z "${GROOT_PYTHON:-}" ]]; then
  if [[ -x /opt/runtime/bin/python ]]; then
    GROOT_PYTHON=/opt/runtime/bin/python
  else
    GROOT_PYTHON=${GROOT_DIR}/.venv/bin/python
  fi
fi
export PATH=/opt/runtime/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
B1K_DATA_ROOT=${B1K_DATA_ROOT:-/workspace/datasets/2026-challenge-demos}
OUTPUT_ROOT=${OUTPUT_ROOT:-/workspace/outputs}
LOG_DIR=${LOG_DIR:-/workspace/logs/b1k}
MAX_STEPS=${MAX_STEPS:-15000}
GPU_COUNT=${GPU_COUNT:-}
if [[ -z "${GPU_COUNT}" ]]; then
  if (( DRY_RUN )); then
    GPU_COUNT=1
  elif command -v nvidia-smi >/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
  else
    echo "nvidia-smi is unavailable; cannot detect GPU count." >&2
    exit 1
  fi
fi
if (( GPU_COUNT == 1 )); then
  CUDA_VISIBLE_DEVICES_VALUE=${CUDA_VISIBLE_DEVICES:-0}
  BATCH_CANDIDATES_VALUE="256 128 64"
elif (( GPU_COUNT == 2 )); then
  CUDA_VISIBLE_DEVICES_VALUE=${CUDA_VISIBLE_DEVICES:-0,1}
  BATCH_CANDIDATES_VALUE="512 256 128"
else
  echo "GPU_COUNT must be 1 or 2; got ${GPU_COUNT}." >&2
  exit 1
fi
if [[ -n "${BATCH_CANDIDATES:-}" ]]; then
  BATCH_CANDIDATES_VALUE=${BATCH_CANDIDATES}
fi
FINAL_CHECKPOINT_FILE=${FINAL_CHECKPOINT_FILE:-${LOG_DIR}/final-checkpoint.txt}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-nvidia/GR00T-N1.7-3B}
ENABLE_CHECKPOINT_WATCHER=${ENABLE_CHECKPOINT_WATCHER:-0}
if [[ -z "${WANDB_MODE:-}" ]]; then
  WANDB_MODE=offline
fi
OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export WANDB_MODE OMP_NUM_THREADS

if (( DRY_RUN )); then
  cat <<EOF
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_VALUE} WANDB_MODE=${WANDB_MODE} ${GROOT_PYTHON} -m torch.distributed.run --nproc_per_node="${GPU_COUNT}" --master_port=29500 scripts/b1k/train_b1k.py --num-gpus "${GPU_COUNT}" --global-batch-size <${BATCH_CANDIDATES_VALUE}> --max-steps ${MAX_STEPS} --save-steps 1500 --save-total-limit 5 --decode-only-used-frames --resume-from-checkpoint
EOF
  exit 0
fi

if [[ ! -x "${GROOT_PYTHON}" ]]; then
  echo "GR00T Python runtime not found: ${GROOT_PYTHON}" >&2
  exit 1
fi
for required_path in scripts/b1k/train_b1k.py examples/b1k/r1pro.py; do
  if [[ ! -f "${GROOT_DIR}/${required_path}" ]]; then
    echo "GR00T checkout is missing B1K entrypoint: ${required_path}" >&2
    exit 1
  fi
done
if ! "${GROOT_PYTHON}" -c 'import torch; assert torch.cuda.is_available()'; then
  echo "GR00T Python runtime cannot access CUDA." >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"
cd "${GROOT_DIR}"

CHECKPOINT_WATCHER_PID=""
cleanup_watcher() {
  if [[ -n "${CHECKPOINT_WATCHER_PID}" ]] && kill -0 "${CHECKPOINT_WATCHER_PID}" 2>/dev/null; then
    kill "${CHECKPOINT_WATCHER_PID}" 2>/dev/null || true
    wait "${CHECKPOINT_WATCHER_PID}" 2>/dev/null || true
  fi
}
trap cleanup_watcher EXIT INT TERM
if [[ "${ENABLE_CHECKPOINT_WATCHER}" == "1" && -n "${R2_REMOTE:-}" && -n "${R2_BUCKET:-}" ]]; then
  SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
  "${SCRIPT_DIR}/watch_checkpoints.sh" >"${LOG_DIR}/checkpoint-watcher.log" 2>&1 &
  CHECKPOINT_WATCHER_PID=$!
  echo "Continuous checkpoint upload enabled; watcher PID ${CHECKPOINT_WATCHER_PID}."
fi

for GLOBAL_BATCH_SIZE in ${BATCH_CANDIDATES_VALUE}; do
  ATTEMPT_NAME="b1k-all100-gbs${GLOBAL_BATCH_SIZE}"
  ATTEMPT_OUTPUT="${OUTPUT_ROOT}/${ATTEMPT_NAME}"
  ATTEMPT_LOG="${LOG_DIR}/train-${ATTEMPT_NAME}.log"
  RESUME_ARGS=()
  if find "${ATTEMPT_OUTPUT}" -maxdepth 1 -type d -name 'checkpoint-*' -print -quit 2>/dev/null | grep -q .; then
    RESUME_ARGS+=(--resume-from-checkpoint)
  fi

  echo "Starting ${ATTEMPT_NAME}; log: ${ATTEMPT_LOG}"
  set +e
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" "${GROOT_PYTHON}" -m torch.distributed.run \
    --nproc_per_node="${GPU_COUNT}" --master_port=29500 scripts/b1k/train_b1k.py \
    --experiment-name "${ATTEMPT_NAME}" \
    --base-model-path "${BASE_MODEL_PATH}" \
    --dataset-path "${B1K_DATA_ROOT}" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/b1k/r1pro.py \
    --num-gpus "${GPU_COUNT}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --output-dir "${OUTPUT_ROOT}" \
    --save-steps 1500 --save-total-limit 5 --max-steps "${MAX_STEPS}" \
    --dataloader-num-workers 16 --decode-only-used-frames \
    "${RESUME_ARGS[@]}" 2>&1 | tee "${ATTEMPT_LOG}"
  STATUS=${PIPESTATUS[0]}
  set -e

  if (( STATUS == 0 )); then
    FINAL_CHECKPOINT=$(find "${ATTEMPT_OUTPUT}" -maxdepth 1 -type d -name 'checkpoint-*' -print \
      | sort -V | tail -n 1)
    if [[ -z "${FINAL_CHECKPOINT}" ]]; then
      echo "Training exited successfully but no checkpoint was found in ${ATTEMPT_OUTPUT}." >&2
      exit 1
    fi
    printf '%s\n' "${FINAL_CHECKPOINT}" > "${FINAL_CHECKPOINT_FILE}"
    exit 0
  fi
  if grep -Eq "CUDA out of memory|OutOfMemoryError|CUDNN_STATUS_ALLOC_FAILED" "${ATTEMPT_LOG}"; then
    echo "Global batch ${GLOBAL_BATCH_SIZE} exhausted VRAM; retrying the next candidate automatically."
    continue
  fi
  echo "Training stopped for a non-OOM error; inspect ${ATTEMPT_LOG}." >&2
  exit "${STATUS}"
done

echo "All configured global batch sizes exhausted VRAM." >&2
exit 1
