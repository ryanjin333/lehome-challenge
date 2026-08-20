#!/usr/bin/env bash
# Bounded preemption receipt writer. Durable local state is never deleted.
set -euo pipefail
RUNTIME_ENV="${LEHOME_RUNTIME_ENV:-/etc/lehome/runtime.env}"
if [[ -f "${RUNTIME_ENV}" ]]; then
  # shellcheck disable=SC1091
  source "${RUNTIME_ENV}"
fi
ROLE="${LEHOME_ROLE:?LEHOME_ROLE is required}"
RECEIPTS_DIR="${LEHOME_RECEIPTS_DIR:-/mnt/lehome/receipts}"
ROLLOUT_CONTEXT="${LEHOME_ROLLOUT_PREEMPTION_CONTEXT:-/run/lehome/rollout-preemption.json}"
WORKSPACE_ROOT="${LEHOME_WORKSPACE_ROOT:-/mnt/lehome}"
# The training service owns the actual child PID.  Signal it before writing
# the generic lifecycle receipt, so its SIGTERM trap can retain a complete
# local checkpoint/recovery cursor while the 60-second Nebius window remains.
if [[ "${ROLE}" == "training" ]]; then
  RUN_ID="${LEHOME_RUN_ID:?LEHOME_RUN_ID is required}"
  CONTROL_BIN="${LEHOME_TRAINING_CONTROL_BIN:-/opt/lehome/guest/bin/lehome-training-control.sh}"
  if ! "${CONTROL_BIN}" stop; then
    echo "training stop was not confirmed; refusing a successful preemption receipt" >&2
    exit 75
  fi
  TRAINING_STOP_STATUS=stopped
else
  TRAINING_STOP_STATUS=not-applicable
fi
export PYTHONPATH=/opt/lehome/guest:/opt/lehome/source/lehome:/opt/lehome/trainer/src:/opt/lehome
COMMON_ARGS=(
  --role "${ROLE}" --receipts-dir "${RECEIPTS_DIR}"
  --training-stop-status "${TRAINING_STOP_STATUS}"
  --rollout-context "${ROLLOUT_CONTEXT}" --workspace-root "${WORKSPACE_ROOT}"
)
if [[ "${ROLE}" == "training" ]]; then
  exec /usr/bin/python3 -m lehome_preempt "${COMMON_ARGS[@]}" --run-id "${RUN_ID}"
fi
# A rollout derives its identity from the root-authored context, never from
# stale runtime.env state that may describe a prior campaign.
exec /usr/bin/python3 -m lehome_preempt "${COMMON_ARGS[@]}"
