#!/bin/bash
# Rollout appliance entrypoint. Validates non-secret runtime prerequisites
# (presence only, never values) and defaults to the appliance supervisor.
set -euo pipefail

MODE="${1:-supervisor}"

require_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "entrypoint: required runtime variable ${name} is not set" >&2
    echo "entrypoint: inject it at runtime; it is never baked into images" >&2
    exit 64
  fi
}

require_env LEHOME_WORKSPACE

if [ ! -d "${LEHOME_WORKSPACE}" ]; then
  echo "entrypoint: shared workspace ${LEHOME_WORKSPACE} is not mounted" >&2
  exit 65
fi

case "${MODE}" in
  supervisor)
    exec /opt/lehome-challenge/.venv/bin/python /opt/lehome/scripts/run_groot_rollout_appliance.py "${@:2}"
    ;;
  controller)
    exec /opt/lehome-challenge/.venv/bin/python /opt/lehome/scripts/run_groot_rollout_controller.py "${@:2}"
    ;;
  finalizer)
    exec /opt/lehome-challenge/.venv/bin/python /opt/lehome/scripts/run_groot_artifact_sync.py --role finalizer "${@:2}"
    ;;
  uploader)
    token_file="${LEHOME_HF_TOKEN_FILE:-${LEHOME_WORKSPACE}/secrets/hf_token}"
    exec /opt/lehome-challenge/.venv/bin/python /opt/lehome/scripts/run_groot_artifact_sync.py --role uploader --token-file "${token_file}" "${@:2}"
    ;;
  *)
    echo "entrypoint: unknown mode ${MODE}" >&2
    exit 64
    ;;
esac
