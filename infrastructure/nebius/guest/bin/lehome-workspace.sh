#!/usr/bin/env bash
# Admit the shared workspace disk and take the active-role lease.
set -euo pipefail
RUNTIME_ENV="${LEHOME_RUNTIME_ENV:-/etc/lehome/runtime.env}"
if [[ -f "${RUNTIME_ENV}" ]]; then
  # shellcheck disable=SC1091
  source "${RUNTIME_ENV}"
fi
ROLE="${LEHOME_ROLE:?LEHOME_ROLE is required}"
RUN_ID="${LEHOME_RUN_ID:?LEHOME_RUN_ID is required}"
DEVICE="${LEHOME_WORKSPACE_DEVICE:-/dev/disk/by-id/virtio-lehome}"
export PYTHONPATH=/opt/lehome/guest
args=(--device "${DEVICE}" --role "${ROLE}" --run-id "${RUN_ID}")
if [[ -n "${LEHOME_WORKSPACE_UUID:-}" ]]; then
  args+=(--expected-uuid "${LEHOME_WORKSPACE_UUID}")
fi
if [[ "${1:-}" == "--release" ]]; then
  args+=(--release)
elif [[ $# -ne 0 ]]; then
  echo "usage: lehome-workspace.sh [--release]" >&2
  exit 2
fi
exec /usr/bin/python3 -m lehome_workspace "${args[@]}"
