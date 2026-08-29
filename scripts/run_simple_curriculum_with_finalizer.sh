#!/usr/bin/env bash
# Operator-local safety wrapper. It admits no arbitrary remote command: the
# exact approved controller argv is constructed here and finalization always
# runs on EXIT, including a remote/controller/handoff-write failure.
set -euo pipefail

controller_status=0
finalizer_status=0
validated=0
# This known-good fallback is deliberately independent of all operator input:
# malformed metadata must never strand the exact billed VM before the wrapper
# can reach the local exact-ID stop adapter.
emergency_stop_timeout=300
finalize() {
  set +e
  if [ "$validated" -eq 1 ]; then
    uv run --project trainer python3 scripts/finalize_simple_curriculum_collection.py \
      --ssh-target "$LEHOME_OPERATOR_SSH_TARGET" --ssh-port "$LEHOME_OPERATOR_SSH_PORT" \
      --remote-campaign-root "$LEHOME_OPERATOR_CAMPAIGN_ROOT" \
      --run-id "$LEHOME_OPERATOR_RUN_ID" --round-id "$LEHOME_OPERATOR_ROUND_ID" \
      --hf-token-file "$LEHOME_OPERATOR_HF_TOKEN_FILE" --stop-timeout-seconds "$LEHOME_OPERATOR_STOP_TIMEOUT_SECONDS"
  else
    uv run --project trainer python3 scripts/finalize_simple_curriculum_collection.py \
      --emergency-stop-only --stop-timeout-seconds "$emergency_stop_timeout"
  fi
  finalizer_status=$?
  set -e
}
finish() {
  trap - EXIT
  finalize
  if [ "$controller_status" -ne 0 ] || [ "$finalizer_status" -ne 0 ]; then exit 1; fi
}
trap finish EXIT
reject() {
  controller_status=1
  echo "$1" >&2
  exit 2
}

for required in LEHOME_OPERATOR_SSH_TARGET LEHOME_OPERATOR_CAMPAIGN_ROOT LEHOME_OPERATOR_RUN_ID LEHOME_OPERATOR_ROUND_ID LEHOME_OPERATOR_REVIEWED_REVISION LEHOME_OPERATOR_HF_TOKEN_FILE; do
  [[ -n "${!required:-}" ]] || reject "${required} is required"
done
[[ "$LEHOME_OPERATOR_RUN_ID" =~ ^fresh-run-[a-z0-9-]{1,112}$ ]] || reject "invalid operator run ID"
[[ "$LEHOME_OPERATOR_ROUND_ID" =~ ^fresh-12k-[a-z0-9-]{1,112}$ ]] || reject "invalid operator round ID"
[[ "$LEHOME_OPERATOR_REVIEWED_REVISION" =~ ^[0-9a-f]{40}$ ]] || reject "invalid reviewed revision"
[[ "$LEHOME_OPERATOR_CAMPAIGN_ROOT" == "/mnt/lehome/eval/$LEHOME_OPERATOR_RUN_ID" ]] || reject "invalid campaign root"
[[ "$LEHOME_OPERATOR_SSH_TARGET" =~ ^[A-Za-z_][A-Za-z0-9_.-]{0,63}@[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?([.][A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*$ ]] || reject "invalid operator SSH target"
LEHOME_OPERATOR_SSH_PORT="${LEHOME_OPERATOR_SSH_PORT:-22}"
[[ "$LEHOME_OPERATOR_SSH_PORT" =~ ^[1-9][0-9]{0,4}$ ]] && (( LEHOME_OPERATOR_SSH_PORT <= 65535 )) || reject "invalid operator SSH port"
LEHOME_OPERATOR_STOP_TIMEOUT_SECONDS="${LEHOME_OPERATOR_STOP_TIMEOUT_SECONDS:-300}"
[[ "$LEHOME_OPERATOR_STOP_TIMEOUT_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]] && awk -v value="$LEHOME_OPERATOR_STOP_TIMEOUT_SECONDS" 'BEGIN { exit !(value > 0 && value <= 600) }' || reject "invalid stop timeout"
# The remote controller itself is capped at 86,399 seconds.  Give SSH one
# bounded minute of teardown margin, while keeping the local EXIT finalizer
# reachable even if an established session remains alive but never returns.
LEHOME_OPERATOR_SSH_SESSION_TIMEOUT_SECONDS="${LEHOME_OPERATOR_SSH_SESSION_TIMEOUT_SECONDS:-86460}"
[[ "$LEHOME_OPERATOR_SSH_SESSION_TIMEOUT_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]] && awk -v value="$LEHOME_OPERATOR_SSH_SESSION_TIMEOUT_SECONDS" 'BEGIN { exit !(value > 0 && value <= 86520) }' || reject "invalid SSH session timeout"
test -f "$LEHOME_OPERATOR_HF_TOKEN_FILE" || reject "operator HF token file is unavailable"
test ! -L "$LEHOME_OPERATOR_HF_TOKEN_FILE" || reject "operator HF token file must not be a symlink"
test -s "$LEHOME_OPERATOR_HF_TOKEN_FILE" || reject "operator HF token file is empty"
test "$(stat -f '%u' "$LEHOME_OPERATOR_HF_TOKEN_FILE")" = "$(id -u)" || reject "operator HF token file owner is invalid"
test "$(stat -f '%Lp' "$LEHOME_OPERATOR_HF_TOKEN_FILE")" = 600 || reject "operator HF token file mode is invalid"
validated=1
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
if ! printf '%s\n' "$REMOTE_SCRIPT" | python3 -c '
import subprocess
import sys

try:
    completed = subprocess.run(
        sys.argv[2:], stdin=sys.stdin.buffer, check=False,
        timeout=float(sys.argv[1]),
    )
except subprocess.TimeoutExpired:
    raise SystemExit(124)
raise SystemExit(completed.returncode)
' "$LEHOME_OPERATOR_SSH_SESSION_TIMEOUT_SECONDS" \
  ssh -o ClearAllForwardings=yes -o BatchMode=yes -o IdentitiesOnly=yes \
  -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -p "$LEHOME_OPERATOR_SSH_PORT" -- "$LEHOME_OPERATOR_SSH_TARGET" sh -s -- \
  "$LEHOME_OPERATOR_REVIEWED_REVISION" "$LEHOME_OPERATOR_RUN_ID" "$LEHOME_OPERATOR_ROUND_ID" "$LEHOME_OPERATOR_CAMPAIGN_ROOT"; then
  controller_status=1
fi
