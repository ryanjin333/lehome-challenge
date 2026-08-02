#!/usr/bin/env bash
set -Eeuo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

load_vast_runtime_variable() {
  local variable_name=$1
  local entry
  if [[ -n "${!variable_name:-}" || ! -r /proc/1/environ ]]; then
    return
  fi
  while IFS= read -r -d '' entry; do
    case "$entry" in
      "${variable_name}="*)
        printf -v "$variable_name" '%s' "${entry#*=}"
        export "$variable_name"
        return
        ;;
    esac
  done < /proc/1/environ
}

load_vast_runtime_variable CONTAINER_ID
load_vast_runtime_variable CONTAINER_API_KEY

UPLOAD_VERIFIED_MARKER=${UPLOAD_VERIFIED_MARKER:-/workspace/logs/b1k/UPLOAD_VERIFIED}
if [[ "${AUTO_DESTROY:-0}" != "1" ]]; then
  echo "Refusing to destroy instance unless AUTO_DESTROY=1." >&2
  exit 1
fi
if [[ ! -s "${UPLOAD_VERIFIED_MARKER}" ]]; then
  echo "Refusing to destroy instance without a verified upload marker: ${UPLOAD_VERIFIED_MARKER}" >&2
  exit 1
fi
if [[ ! "${CONTAINER_ID:-}" =~ ^[0-9]+$ ]]; then
  echo "CONTAINER_ID must be a numeric Vast instance ID." >&2
  exit 1
fi
if [[ -z "${CONTAINER_API_KEY:-}" ]]; then
  echo "CONTAINER_API_KEY is required for instance-scoped self-destruction." >&2
  exit 1
fi

if (( DRY_RUN )); then
  echo "DELETE https://console.vast.ai/api/v0/instances/${CONTAINER_ID}/ using the injected instance-scoped API key"
  exit 0
fi
echo "Run bundle verified at $(<"${UPLOAD_VERIFIED_MARKER}"); destroying Vast instance ${CONTAINER_ID}."
DESTROY_RESPONSE=$(curl --fail --silent --show-error \
  --retry 3 --retry-all-errors --connect-timeout 10 --max-time 30 \
  --request DELETE \
  --header "Authorization: Bearer ${CONTAINER_API_KEY}" \
  "https://console.vast.ai/api/v0/instances/${CONTAINER_ID}/")
if ! grep -Eq '"success"[[:space:]]*:[[:space:]]*true' <<<"${DESTROY_RESPONSE}"; then
  echo "Vast did not confirm instance destruction: ${DESTROY_RESPONSE}" >&2
  exit 1
fi
echo "Vast confirmed destruction of instance ${CONTAINER_ID}."
