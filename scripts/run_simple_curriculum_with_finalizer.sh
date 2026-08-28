#!/usr/bin/env bash
# Operator-local safety wrapper: the EXIT trap always invokes the exact-VM
# finalizer, including a remote controller or handoff-write failure.
set -euo pipefail

for required in LEHOME_OPERATOR_SSH_TARGET LEHOME_OPERATOR_CAMPAIGN_ROOT LEHOME_OPERATOR_RUN_ID LEHOME_OPERATOR_ROUND_ID LEHOME_OPERATOR_HF_TOKEN_FILE LEHOME_REMOTE_CONTROLLER_COMMAND; do
  test -n "${!required:-}" || { echo "${required} is required" >&2; exit 2; }
done
test -f "$LEHOME_OPERATOR_HF_TOKEN_FILE" && test ! -L "$LEHOME_OPERATOR_HF_TOKEN_FILE" && test -s "$LEHOME_OPERATOR_HF_TOKEN_FILE"
test "$(stat -f '%u %Lp' "$LEHOME_OPERATOR_HF_TOKEN_FILE")" = "0 600"

finalize() {
  uv run --project trainer python3 scripts/finalize_simple_curriculum_collection.py \
    --ssh-target "$LEHOME_OPERATOR_SSH_TARGET" --ssh-port "${LEHOME_OPERATOR_SSH_PORT:-22}" \
    --remote-campaign-root "$LEHOME_OPERATOR_CAMPAIGN_ROOT" \
    --run-id "$LEHOME_OPERATOR_RUN_ID" --round-id "$LEHOME_OPERATOR_ROUND_ID" \
    --hf-token-file "$LEHOME_OPERATOR_HF_TOKEN_FILE" --stop-timeout-seconds "${LEHOME_OPERATOR_STOP_TIMEOUT_SECONDS:-300}"
}
trap finalize EXIT
ssh -o ClearAllForwardings=yes -o BatchMode=yes -p "${LEHOME_OPERATOR_SSH_PORT:-22}" \
  "$LEHOME_OPERATOR_SSH_TARGET" "$LEHOME_REMOTE_CONTROLLER_COMMAND"
