#!/usr/bin/env bash
# Operator-local safety wrapper. It admits no arbitrary remote command: the
# exact approved controller argv is constructed here and finalization always
# runs on EXIT, including a remote/controller/handoff-write failure.
set -euo pipefail

for required in LEHOME_OPERATOR_SSH_TARGET LEHOME_OPERATOR_CAMPAIGN_ROOT LEHOME_OPERATOR_RUN_ID LEHOME_OPERATOR_ROUND_ID LEHOME_OPERATOR_REVIEWED_REVISION LEHOME_OPERATOR_HF_TOKEN_FILE; do
  test -n "${!required:-}" || { echo "${required} is required" >&2; exit 2; }
done
[[ "$LEHOME_OPERATOR_RUN_ID" =~ ^fresh-run-[a-z0-9-]{1,112}$ ]] || { echo "invalid operator run ID" >&2; exit 2; }
[[ "$LEHOME_OPERATOR_ROUND_ID" =~ ^fresh-12k-[a-z0-9-]{1,112}$ ]] || { echo "invalid operator round ID" >&2; exit 2; }
[[ "$LEHOME_OPERATOR_REVIEWED_REVISION" =~ ^[0-9a-f]{40}$ ]] || { echo "invalid reviewed revision" >&2; exit 2; }
[[ "$LEHOME_OPERATOR_CAMPAIGN_ROOT" == /mnt/lehome/eval/fresh-run-* ]] || { echo "invalid campaign root" >&2; exit 2; }
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
# The persisted remote invocation record is the only remote shell input. It
# must exactly match validated local IDs/revision/root before fixed controller
# argv execution; no operator-provided remote command is accepted.
read -r -d '' REMOTE_SCRIPT <<'SH' || true
set -eu
. /mnt/lehome/operator/simple-curriculum-invocation.env
test "$LEHOME_REVIEWED_REVISION" = "$1"
test "$LEHOME_RUN_ID" = "$2"
test "$LEHOME_ROUND_ID" = "$3"
test "$LEHOME_CAMPAIGN_ROOT" = "$4"
host=/mnt/lehome/runtime-code/$1
test -d "$host" && test ! -L "$host"
exec sudo env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  PYTHONPATH="$host:$host/source/lehome:$host/trainer/src" \
  LEHOME_PAID_COLLECTION=1 LEHOME_HOST_CODE_ROOT="$host" LEHOME_CAMPAIGN_ROOT="$4" \
  LEHOME_RUN_ID="$2" LEHOME_ROUND_ID="$3" LEHOME_MAX_WALL_SECONDS=86399 LEHOME_MAX_SPEND_USD=99.00 \
  LEHOME_RUNTIME_IDENTITY_JSON=/mnt/lehome/operator/runtime-identity.json \
  LEHOME_SPEND_OBSERVER=/mnt/lehome/operator/spend-observation.json \
  "$host/rollout_appliance/run_simple_curriculum_collection.sh"
SH
printf '%s\n' "$REMOTE_SCRIPT" | ssh -o ClearAllForwardings=yes -o BatchMode=yes -p "${LEHOME_OPERATOR_SSH_PORT:-22}" \
  "$LEHOME_OPERATOR_SSH_TARGET" sh -s -- \
  "$LEHOME_OPERATOR_REVIEWED_REVISION" "$LEHOME_OPERATOR_RUN_ID" "$LEHOME_OPERATOR_ROUND_ID" "$LEHOME_OPERATOR_CAMPAIGN_ROOT"
