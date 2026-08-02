#!/usr/bin/env bash
set -Eeuo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

B1K_DATA_ROOT=${B1K_DATA_ROOT:-/workspace/datasets/2026-challenge-demos}
HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export HF_HOME
export HF_XET_HIGH_PERFORMANCE=1

DOWNLOAD_COMMAND=(
  hf download behavior-1k/2026-challenge-demos
  --repo-type dataset
  --revision main
  --local-dir "${B1K_DATA_ROOT}"
  --include "annotations/**"
  --include "data/**"
  --include "meta/**"
  --include "videos/observation.rgb.zed_link_camera_0/**"
  --include "videos/observation.rgb.left_realsense_link_camera_0/**"
  --include "videos/observation.rgb.right_realsense_link_camera_0/**"
)

if (( DRY_RUN )); then
  printf 'HF_XET_HIGH_PERFORMANCE=1 '
  printf '%q ' "${DOWNLOAD_COMMAND[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${B1K_DATA_ROOT}" "${HF_HOME}"
echo "Downloading the 100-task RGB-only dataset to ${B1K_DATA_ROOT}. Existing files will resume."
exec "${DOWNLOAD_COMMAND[@]}"
