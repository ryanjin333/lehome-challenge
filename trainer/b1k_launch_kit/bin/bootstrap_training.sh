#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DRY_RUN=0
PREPARE_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --prepare-only) PREPARE_ONLY=1; shift ;;
    *) echo "Usage: $0 [--dry-run] [--prepare-only]" >&2; exit 2 ;;
  esac
done

B1K_WORK_ROOT=${B1K_WORK_ROOT:-/workspace/b1k}
B1K_DATA_ROOT=${B1K_DATA_ROOT:-/workspace/datasets/2026-challenge-demos}
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
HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
LOG_DIR=${LOG_DIR:-/workspace/logs/b1k}
MIN_FREE_BYTES=${MIN_FREE_BYTES:-1500000000000}
if [[ -z "${B1K_PREBUILT_IMAGE:-}" ]]; then
  if [[ -d /opt/isaac-groot/.git && -x /opt/runtime/bin/python ]]; then
    B1K_PREBUILT_IMAGE=1
  else
    B1K_PREBUILT_IMAGE=0
  fi
fi
export PATH=/opt/runtime/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export B1K_DATA_ROOT GROOT_DIR GROOT_PYTHON HF_HOME
export HF_XET_HIGH_PERFORMANCE=1

# Vast account-level environment variables are injected during container startup,
# but are not necessarily present in a later SSH login. The template's on-start
# hook persists HF_TOKEN to this root-only Hugging Face token file so reruns can
# authenticate without putting the credential in the template or shell history.
if [[ -z "${HF_TOKEN:-}" && -s "${HF_HOME}/token" ]]; then
  HF_TOKEN=$(<"${HF_HOME}/token")
  export HF_TOKEN
fi

if (( DRY_RUN )); then
  cat <<EOF
Preflight: require HF_TOKEN, 1-2 GPUs, and ${MIN_FREE_BYTES} free bytes
Install: git, git-lfs, curl, zstd, tmux, rclone, uv, huggingface_hub
Clone: https://github.com/wensi-ai/Isaac-GR00T -> ${GROOT_DIR}
Prepare: uv sync --frozen --python 3.10
Pre-cache: nvidia/GR00T-N1.7-3B and nvidia/Cosmos-Reason2-2B
Download: ${SCRIPT_DIR}/download_dataset.sh
Validate: ${SCRIPT_DIR}/validate_dataset.py ${B1K_DATA_ROOT}
Deploy: python scripts/b1k/deploy_modality.py ${B1K_DATA_ROOT}
EOF
  "${SCRIPT_DIR}/download_dataset.sh" --dry-run
  if (( PREPARE_ONLY == 0 )); then
    echo "Autostart: ${SCRIPT_DIR}/start_training.sh"
    "${SCRIPT_DIR}/start_training.sh" --dry-run
  fi
  exit 0
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required. Save it as a Vast account environment variable or ${HF_HOME}/token." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi is unavailable; this must run inside a GPU container." >&2
  exit 1
fi
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
if (( GPU_COUNT < 1 || GPU_COUNT > 2 )); then
  echo "One or two GPUs are required; detected ${GPU_COUNT}." >&2
  exit 1
fi

mkdir -p "${B1K_WORK_ROOT}" "${LOG_DIR}" "${HF_HOME}" "$(dirname "${B1K_DATA_ROOT}")"
FREE_BYTES=$(df -PB1 "$(dirname "${B1K_DATA_ROOT}")" | awk 'NR==2 {print $4}')
if (( FREE_BYTES < MIN_FREE_BYTES )); then
  echo "Insufficient free disk: ${FREE_BYTES} bytes; require ${MIN_FREE_BYTES}." >&2
  exit 1
fi

if [[ "${B1K_PREBUILT_IMAGE}" == "1" ]]; then
  echo "Using prebuilt training image; verifying baked tools and GR00T environment."
  for tool in git git-lfs curl zstd tmux rclone uv hf; do
    command -v "${tool}" >/dev/null || {
      echo "Prebuilt image is missing required tool: ${tool}" >&2
      exit 1
    }
  done
  if [[ ! -d "${GROOT_DIR}/.git" || ! -x "${GROOT_PYTHON}" ]]; then
    echo "Prebuilt GR00T checkout or environment is missing from ${GROOT_DIR}." >&2
    exit 1
  fi
  for required_path in \
    scripts/b1k/train_b1k.py \
    scripts/b1k/deploy_modality.py \
    examples/b1k/r1pro.py; do
    if [[ ! -f "${GROOT_DIR}/${required_path}" ]]; then
      echo "Prebuilt GR00T checkout is missing B1K entrypoint: ${required_path}" >&2
      exit 1
    fi
  done
else
  if command -v apt-get >/dev/null; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      git git-lfs curl ca-certificates zstd tmux rclone ffmpeg libgl1 libglib2.0-0 python3-pip
  fi
  python3 -m pip install -U uv "huggingface_hub>=0.32"
  git lfs install --skip-repo
fi

# Fail before the terabyte transfer if the gated backbone is inaccessible.
mkdir -p "${HF_HOME}/gate-check"
hf download nvidia/Cosmos-Reason2-2B config.json \
  --local-dir "${HF_HOME}/gate-check/cosmos"

echo "Starting dataset, environment, and model preparation concurrently."
"${SCRIPT_DIR}/download_dataset.sh" >"${LOG_DIR}/dataset-download.log" 2>&1 &
DATASET_PID=$!

(
  if [[ "${B1K_PREBUILT_IMAGE}" == "1" ]]; then
    echo "GR00T environment supplied by prebuilt image."
  elif [[ ! -d "${GROOT_DIR}/.git" ]]; then
    git clone https://github.com/wensi-ai/Isaac-GR00T "${GROOT_DIR}"
    cd "${GROOT_DIR}"
    uv sync --frozen --python 3.10
    uv pip install --python .venv/bin/python websockets
  else
    cd "${GROOT_DIR}"
    uv sync --frozen --python 3.10
    uv pip install --python .venv/bin/python websockets
  fi
) >"${LOG_DIR}/groot-setup.log" 2>&1 &
GROOT_PID=$!

(
  hf download nvidia/GR00T-N1.7-3B
  hf download nvidia/Cosmos-Reason2-2B
) >"${LOG_DIR}/model-download.log" 2>&1 &
MODEL_PID=$!

FAILED=0
wait "${DATASET_PID}" || FAILED=1
wait "${GROOT_PID}" || FAILED=1
wait "${MODEL_PID}" || FAILED=1
if (( FAILED )); then
  echo "Preparation failed. Inspect ${LOG_DIR}/*.log; rerunning this script resumes downloads." >&2
  exit 1
fi

"${GROOT_PYTHON}" "${SCRIPT_DIR}/validate_dataset.py" "${B1K_DATA_ROOT}" \
  | tee "${LOG_DIR}/dataset-validation.json"
cd "${GROOT_DIR}"
git rev-parse HEAD > "${LOG_DIR}/groot-commit.txt"
"${GROOT_PYTHON}" -c \
  'from huggingface_hub import HfApi; print(HfApi().dataset_info("behavior-1k/2026-challenge-demos").sha)' \
  > "${LOG_DIR}/dataset-revision.txt"
"${GROOT_PYTHON}" scripts/b1k/deploy_modality.py "${B1K_DATA_ROOT}"

if (( PREPARE_ONLY )); then
  echo "Preparation complete."
  exit 0
fi
echo "Preparation complete; starting training without another prompt."
exec "${SCRIPT_DIR}/start_training.sh"
