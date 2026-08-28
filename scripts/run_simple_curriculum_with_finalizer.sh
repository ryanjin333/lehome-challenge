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
[[ "$LEHOME_OPERATOR_SSH_TARGET" =~ ^[A-Za-z_][A-Za-z0-9_.-]{0,63}@[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?([.][A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*$ ]] || { echo "invalid operator SSH target" >&2; exit 2; }
LEHOME_OPERATOR_SSH_PORT="${LEHOME_OPERATOR_SSH_PORT:-22}"
[[ "$LEHOME_OPERATOR_SSH_PORT" =~ ^[1-9][0-9]{0,4}$ ]] && (( LEHOME_OPERATOR_SSH_PORT <= 65535 )) || { echo "invalid operator SSH port" >&2; exit 2; }
test -f "$LEHOME_OPERATOR_HF_TOKEN_FILE" && test ! -L "$LEHOME_OPERATOR_HF_TOKEN_FILE" && test -s "$LEHOME_OPERATOR_HF_TOKEN_FILE"
test "$(stat -f '%u' "$LEHOME_OPERATOR_HF_TOKEN_FILE")" = "$(id -u)"
test "$(stat -f '%Lp' "$LEHOME_OPERATOR_HF_TOKEN_FILE")" = 600

controller_status=0
finalizer_status=0
finalize() {
  set +e
  uv run --project trainer python3 scripts/finalize_simple_curriculum_collection.py \
    --ssh-target "$LEHOME_OPERATOR_SSH_TARGET" --ssh-port "$LEHOME_OPERATOR_SSH_PORT" \
    --remote-campaign-root "$LEHOME_OPERATOR_CAMPAIGN_ROOT" \
    --run-id "$LEHOME_OPERATOR_RUN_ID" --round-id "$LEHOME_OPERATOR_ROUND_ID" \
    --hf-token-file "$LEHOME_OPERATOR_HF_TOKEN_FILE" --stop-timeout-seconds "${LEHOME_OPERATOR_STOP_TIMEOUT_SECONDS:-300}"
  finalizer_status=$?
  set -e
}
finish() {
  trap - EXIT
  finalize
  if [ "$controller_status" -ne 0 ] || [ "$finalizer_status" -ne 0 ]; then exit 1; fi
}
trap finish EXIT
# The persisted remote invocation record is the only remote shell input. It
# must exactly match validated local IDs/revision/root before fixed controller
# argv execution; no operator-provided remote command is accepted.
read -r -d '' REMOTE_SCRIPT <<'SH' || true
set -eu
record=/mnt/lehome/operator/simple-curriculum-invocation.env
test -f "$record" && test ! -L "$record"
test "$(stat -c '%a' "$record")" = 600
test "$(stat -c '%u' "$record")" = "$(id -u)"
awk -F= 'BEGIN { split("LEHOME_REVIEWED_REVISION LEHOME_CAMPAIGN_ROOT LEHOME_RUN_ID LEHOME_ROUND_ID LEHOME_SPEND_BASELINE_USD LEHOME_SPEND_BASELINE_AT_UTC LEHOME_MAX_HOURLY_BURN_USD LEHOME_SPEND_OBSERVER_COMMAND", a, " "); for (i in a) ok[a[i]]=1 } /^[A-Z_]+=[A-Za-z0-9._:\/+-]+$/ { if (!ok[$1] || seen[$1]++) exit 1; next } { exit 1 } END { for (i in ok) if (!seen[i]) exit 1 }' "$record"
revision=$(sed -n 's/^LEHOME_REVIEWED_REVISION=//p' "$record")
campaign=$(sed -n 's/^LEHOME_CAMPAIGN_ROOT=//p' "$record")
run=$(sed -n 's/^LEHOME_RUN_ID=//p' "$record")
round=$(sed -n 's/^LEHOME_ROUND_ID=//p' "$record")
test "$revision" = "$1" || exit 1
test "$run" = "$2" || exit 1
test "$round" = "$3" || exit 1
test "$campaign" = "$4" || exit 1
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
if ! printf '%s\n' "$REMOTE_SCRIPT" | ssh -o ClearAllForwardings=yes -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=10 \
  -p "$LEHOME_OPERATOR_SSH_PORT" -- "$LEHOME_OPERATOR_SSH_TARGET" sh -s -- \
  "$LEHOME_OPERATOR_REVIEWED_REVISION" "$LEHOME_OPERATOR_RUN_ID" "$LEHOME_OPERATOR_ROUND_ID" "$LEHOME_OPERATOR_CAMPAIGN_ROOT"; then
  controller_status=1
fi
