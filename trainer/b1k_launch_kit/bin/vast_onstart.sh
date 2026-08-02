#!/usr/bin/env bash
set -Eeuo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

B1K_KIT_URL=${B1K_KIT_URL:-}
B1K_KIT_SHA256=${B1K_KIT_SHA256:-}
KIT_DIR=${KIT_DIR:-/workspace/b1k-launch-kit}
KIT_ARCHIVE=${KIT_ARCHIVE:-/workspace/b1k-groot-launch-kit.tar.gz}
RUN_LOG=${RUN_LOG:-/workspace/disposable-training.log}
LIFECYCLE_SMOKE=${LIFECYCLE_SMOKE:-0}

if [[ -z "${B1K_KIT_URL}" || -z "${B1K_KIT_SHA256}" ]]; then
  echo "B1K_KIT_URL and B1K_KIT_SHA256 are required." >&2
  exit 1
fi

if (( DRY_RUN )); then
  echo "curl ${B1K_KIT_URL} to ${KIT_ARCHIVE}"
  echo "sha256sum --check ${KIT_ARCHIVE}"
  echo "extract kit to ${KIT_DIR}"
  echo "if LIFECYCLE_SMOKE=1: nohup ${KIT_DIR}/bin/run_lifecycle_smoke.sh > ${RUN_LOG}"
  echo "otherwise: nohup ${KIT_DIR}/bin/run_disposable_training.sh > ${RUN_LOG}"
  exit 0
fi

mkdir -p "${KIT_DIR}"
curl --fail --location --retry 12 --retry-all-errors \
  "${B1K_KIT_URL}" --output "${KIT_ARCHIVE}"
printf '%s  %s\n' "${B1K_KIT_SHA256}" "${KIT_ARCHIVE}" | sha256sum --check -
tar -xzf "${KIT_ARCHIVE}" -C "${KIT_DIR}"
chmod +x "${KIT_DIR}"/bin/*.sh
RUNNER="${KIT_DIR}/bin/run_disposable_training.sh"
if [[ "${LIFECYCLE_SMOKE}" == "1" ]]; then
  RUNNER="${KIT_DIR}/bin/run_lifecycle_smoke.sh"
fi
nohup "${RUNNER}" >"${RUN_LOG}" 2>&1 &
echo "Disposable training launched as PID $!; log: ${RUN_LOG}"
