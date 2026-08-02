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

HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
LOG_DIR=${LOG_DIR:-/workspace/logs/b1k-smoke}
B1K_DATA_ROOT=${B1K_DATA_ROOT:-/workspace/smoke-dataset}
OUTPUT_ROOT=${OUTPUT_ROOT:-/workspace/smoke-outputs}
CYCLE_ID=${CYCLE_ID:-smoke-${CONTAINER_ID:-manual}-$(date -u +%Y%m%dT%H%M%SZ)}
UPLOAD_VERIFIED_MARKER=${UPLOAD_VERIFIED_MARKER:-${LOG_DIR}/UPLOAD_VERIFIED}
export HF_HOME LOG_DIR B1K_DATA_ROOT OUTPUT_ROOT CYCLE_ID UPLOAD_VERIFIED_MARKER

if [[ -z "${HF_TOKEN:-}" && -s "${HF_HOME}/token" ]]; then
  HF_TOKEN=$(<"${HF_HOME}/token")
  export HF_TOKEN
fi
if [[ ! "${HF_TOKEN:-}" =~ ^hf_[A-Za-z0-9]+$ ]]; then
  echo "A token-shaped HF_TOKEN or ${HF_HOME}/token is required." >&2
  exit 1
fi

if (( DRY_RUN )); then
  echo "Validate HF authentication; no dataset or model download"
  echo "Create tiny synthetic checkpoint and normalization metadata"
  echo "${SCRIPT_DIR}/push_run_bundle.sh --checkpoint ${OUTPUT_ROOT}/checkpoint-smoke --cycle ${CYCLE_ID} --status success"
  echo "${SCRIPT_DIR}/destroy_instance.sh"
  exit 0
fi

mkdir -p "${LOG_DIR}" "${B1K_DATA_ROOT}/meta" "${OUTPUT_ROOT}/checkpoint-smoke"
exec > >(tee -a "${LOG_DIR}/lifecycle-smoke.log") 2>&1

echo "Validating injected Hugging Face credential without downloading model data."
curl --fail --silent --show-error --retry 3 \
  -H "Authorization: Bearer ${HF_TOKEN}" \
  --output /dev/null https://huggingface.co/api/whoami-v2

printf '{"smoke": {"mean": [0.0], "std": [1.0]}}\n' > "${B1K_DATA_ROOT}/meta/stats.json"
printf '{"smoke": {"modality_keys": ["smoke"]}}\n' > "${B1K_DATA_ROOT}/meta/modality.json"
printf 'lifecycle-smoke-no-dataset\n' > "${LOG_DIR}/dataset-revision.txt"
printf 'lifecycle-smoke-no-groot-checkout\n' > "${LOG_DIR}/groot-commit.txt"
printf 'bounded lifecycle smoke artifact\n' > "${OUTPUT_ROOT}/checkpoint-smoke/model.safetensors"

"${SCRIPT_DIR}/push_run_bundle.sh" \
  --checkpoint "${OUTPUT_ROOT}/checkpoint-smoke" \
  --cycle "${CYCLE_ID}" --status success
"${SCRIPT_DIR}/destroy_instance.sh"
